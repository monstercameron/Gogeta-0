"""SQ206 / X0: causality perturbation test for every byte-model variant (CPU, seeded, a few minutes).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python -m experiments.byte_model.causality

Criterion (written before running, notes/analysis-coarse-to-fine.md section 2): row j of the logits predicts byte j+1.
Perturb input byte s (patch starts, middles and ends). Then
  * ar / flat models: every row j < s (all predictions of bytes <= s) must be BITWISE unchanged, in every pass;
  * parallel_refine: every row j whose target patch (j+1)//P is <= s//P must be bitwise unchanged (the whole target
    patch is predicted without its own bytes), in every pass;
  * non-vacuous: at least one row that is allowed to see byte s does change (ar: row s; parallel: the first row of
    the next target patch);
  * control: the deliberately broken variant (patch_shift=False) must FAIL (some protected row changes).
GDN runs through the token-by-token CPU reference (cpu_ref.py), fp32. Writes results/sq206_causality.json.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.byte_model import cpu_ref  # noqa: E402

cpu_ref.install()
from experiments.byte_model.model_bytes import FusionDance, ModelConfig, make_model  # noqa: E402

TINY_GLOBAL = dict(d_model=32, n_heads=2, mlp_hidden=64, prelude=["swa", "gdn", "attn"], core=[], coda=[], window=4)
TINY_LOCAL = dict(d_local=16, local_heads=2, local_mlp_hidden=32, local_window=6)

# name -> extra config (patched variants); "flat" is FusionDance on bytes
VARIANTS = {
    "ar_r1": dict(refine_passes=1),
    "ar_r2_shared_gated": dict(refine_passes=2),
    "ar_r4_shared_contractive": dict(refine_passes=4, refine_update="contractive"),
    "ar_r2_unshared": dict(refine_passes=2, refine_shared=False),
    "ar_r2_residual": dict(refine_passes=2, refine_update="residual", refine_loss="last"),
    "ar_lswa_local": dict(refine_passes=2, local_encoder=["conv", "lswa"], refine_blocks=["conv", "gdn", "lswa"]),
    "ar_looped_global": dict(refine_passes=2, prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=2,
                             loops_min=1, loops_max=3, bptt_loops=2),
    "ar_ternary_global": dict(refine_passes=2, ternary=True),
    "par_r1": dict(decoder="parallel_refine", refine_passes=1),
    "par_r2_soft": dict(decoder="parallel_refine", refine_passes=2),
    "par_r3_soft": dict(decoder="parallel_refine", refine_passes=3),
    "par_r2_argmax": dict(decoder="parallel_refine", refine_passes=2, refine_feedback="argmax"),
    "par_r2_lswa": dict(decoder="parallel_refine", refine_passes=2, refine_blocks=["conv", "gdn", "lswa"]),
}
BROKEN = {"BROKEN_ar_noshift": dict(refine_passes=2, patch_shift=False),
          "BROKEN_par_noshift": dict(decoder="parallel_refine", refine_passes=2, patch_shift=False)}


def build(name: str, P: int, T: int, seed: int = 0):
    torch.manual_seed(seed)
    if name == "flat":
        cfg = ModelConfig(vocab_size=256, max_seq_len=T, **{**TINY_GLOBAL, "prelude": ["swa", "gdn", "gdn", "swa",
                                                                                        "gdn", "attn"]})
        return FusionDance(cfg).eval()
    extra = {**VARIANTS, **BROKEN}[name]
    cfg = ModelConfig(vocab_size=256, max_seq_len=T, patch_size=P, **{**TINY_GLOBAL, **TINY_LOCAL, **extra})
    return make_model(cfg).eval()


def positions(P: int, T: int) -> list[int]:
    """Patch starts, middles and ends, across the first, an inner and the last patch."""
    ps = {0, P // 2, P - 1, P, P + P // 2, 2 * P - 1, 3 * P, 3 * P + P // 2, T - P, T - P // 2, T - 1}
    return sorted(s for s in ps if 0 <= s < T)


def all_logits(model, x):
    with torch.no_grad():
        out = model(x, return_all=True)[0] if hasattr(model, "passes") else model(x)[0]
    return [lg for _, lg in out] if isinstance(out, list) else [out]


def protected_rows(model, s: int, T: int) -> torch.Tensor:
    j = torch.arange(T)
    if getattr(model, "P", 0) and model.cfg.decoder == "parallel_refine":
        P = model.P
        return (j + 1) // P <= s // P
    return j < s


def must_change_row(model, s: int, T: int) -> int | None:
    if getattr(model, "P", 0) and model.cfg.decoder == "parallel_refine":
        P = model.P
        j = (s // P + 1) * P - 1
        return j if j < T else None
    return s


def check(name: str, P: int, T: int = 64, seed: int = 0, batch: int = 2) -> dict:
    model = build(name, P, T, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randint(0, 256, (batch, T), generator=g)
    base = all_logits(model, x)
    rec = {"variant": name, "P": P, "T": T, "passes": len(base), "positions": [], "ok": True, "non_vacuous": True}
    for s in positions(P, T):
        xp = x.clone()
        xp[:, s] = (xp[:, s] + 1 + torch.randint(0, 255, (batch,), generator=g)) % 256
        pert = all_logits(model, xp)
        prot = protected_rows(model, s, T)
        leaks, changed = [], True
        for k, (a, b) in enumerate(zip(base, pert)):
            same = (a == b).all(dim=-1).all(dim=0)          # [T]: row bitwise identical for every batch row
            bad = (prot & ~same).nonzero().flatten().tolist()
            if bad:
                leaks.append({"pass": k, "rows": bad[:8], "n": len(bad)})
        j = must_change_row(model, s, T)
        if j is not None:
            changed = not torch.equal(base[-1][:, j], pert[-1][:, j])
        rec["positions"].append({"s": s, "leaks": leaks, "allowed_row_changed": changed})
        rec["ok"] &= not leaks
        rec["non_vacuous"] &= changed
    return rec


def run_all(Ps=(2, 4, 8), T: int = 64, extra_P16: bool = True) -> dict:
    t0 = time.time()
    results = [check("flat", 1, T)]
    for P in Ps:
        for name in VARIANTS:
            results.append(check(name, P, T))
    if extra_P16:
        for name in ("ar_r2_shared_gated", "par_r2_soft"):
            results.append(check(name, 16, T))
    broken = [check(name, P, T) for P in Ps for name in BROKEN]
    passed = all(r["ok"] and r["non_vacuous"] for r in results)
    control_fails = all(not r["ok"] for r in broken)
    return {"id": "SQ206", "criterion": __doc__.split("Criterion")[1].split("GDN runs")[0].strip(),
            "torch": torch.__version__, "dtype": "float32", "gdn": "cpu_ref.gated_delta_rule_ref (recurrent)",
            "n_variants": len(results), "n_perturbations": sum(len(r["positions"]) for r in results),
            "all_correct_variants_pass": passed, "broken_controls_all_fail": control_fails,
            "verdict": "PASS" if passed and control_fails else "FAIL",
            "wall_s": round(time.time() - t0, 1), "results": results, "broken_controls": broken}


if __name__ == "__main__":
    torch.set_num_threads(2)
    out = run_all()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sq206_causality.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out["results"]:
        print(f"{r['variant']:28s} P={r['P']:2d} passes={r['passes']} perturbations={len(r['positions'])} "
              f"{'OK' if r['ok'] else 'LEAK'} {'non-vacuous' if r['non_vacuous'] else 'VACUOUS'}")
    for r in out["broken_controls"]:
        n = sum(1 for p in r["positions"] if p["leaks"])
        print(f"{r['variant']:28s} P={r['P']:2d} {'FAILS as required' if not r['ok'] else 'PASSES (control broken!)'}"
              f" ({n}/{len(r['positions'])} perturbations leak)")
    print(f"SQ206 {out['verdict']}: {out['n_variants']} variants, {out['n_perturbations']} perturbations, "
          f"broken controls fail: {out['broken_controls_all_fail']} ({out['wall_s']} s)")
