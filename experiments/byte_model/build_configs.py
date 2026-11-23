"""Byte main quest (E12-E16): generate the staged configs and the param / FLOP / cost table. CPU only, counting only.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python -m experiments.byte_model.build_configs

Writes (never under configs/ or gogeta/):
  experiments/staged-configs/byte/e1<N><arm>_*.json        200M-byte configs (the default budget)
  experiments/staged-configs/byte/e1<N><arm>_*_20m.json    20M-byte sweep variants (same model)
  experiments/byte_model/flops-table.json / .md             params, FLOPs per byte and per BPE-token-equivalent, x E1
Budget naming: byte configs carry train.max_bytes (and train.seq_len_unit = "bytes"); max_tokens is absent on purpose,
so the unpatched trainer refuses them (KeyError) instead of silently reading bytes as tokens.
FLOPs are COMPUTED (6N rule + attention/GDN terms, FusionDance's convention); GPU-minutes are ESTIMATES from a
bandwidth proxy calibrated on E1's measured 86k tok/s (staged-table.md), +-50%, to be replaced by SQ189.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.byte_model import cpu_ref  # noqa: E402

cpu_ref.install()
from experiments.byte_model.model_bytes import (BYTES_PER_TOKEN, FusionDance, ModelConfig,  # noqa: E402
                                                make_model)

OUT = ROOT / "experiments" / "staged-configs" / "byte"
HERE = ROOT / "experiments" / "byte_model"
T = 4096
PENDING_FIELDS = ("swa_impl",)  # fields that exist only once notes/patches/swa-flex-attention.patch is applied
E1_TOK_PER_S = 86_334            # measured, E1-20261122-124440-s0 (experiments/design/staged-table.md)
E1_LAYERS = 6

E1_BODY = {"d_model": 384, "n_heads": 6, "mlp_hidden": 1024,
           "prelude": ["swa", "gdn", "gdn", "swa", "gdn", "attn"], "core": [], "coda": []}
LOCAL = {"d_local": 192, "local_heads": 3, "local_mlp_hidden": 512, "local_window": 64,
         "local_encoder": ["conv", "gdn"], "refine_blocks": ["conv", "gdn"]}
TRAIN = {"dataset": "code-py-v1-bytes", "seq_len": T, "seq_len_unit": "bytes", "batch_size": 32,
         "max_bytes": 200_000_000, "lr": 0.006, "min_lr_frac": 0.1, "warmup_steps": 76,
         "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
         "eval_every": 250, "eval_seqs": 128, "seed": 0}
NOTE_LR = "lr 0.006 = K7's tuned E1 value, NOT tuned for bytes: run the SQ188 LR check (20M-byte variants) first."


def patched(P: int, **kw) -> dict:
    m = {"vocab_size": 256, "max_seq_len": T, **E1_BODY, "window": 1024 // P, "patch_size": P, **LOCAL,
         "refine_passes": 1, "refine_shared": True, "refine_update": "gated", "refine_loss": "linear",
         "decoder": "ar", "ternary": False}
    m.update(kw)
    return m


SPECS = [
    ("E12", "e12_flat_w1024", "Flat byte-level E1: E1's six mixers at d=384 on raw bytes, vocab 256, T=4,096 bytes, "
     "SWA window 1,024 bytes (about E1's 256-token span). Needs the flex-attention SWA patch (swa_impl flex): the "
     "dense-mask SWA at T=4,096 is O(T^2) (C12), about 6 GB of scores per layer.",
     {"vocab_size": 256, "max_seq_len": T, **E1_BODY, "window": 1024, "ternary": False, "swa_impl": "flex"}, {}),
    ("E13a", "e13a_p2_r1", "Patched coarse-to-fine, P=2, r=1, AR byte decoder (granularity sweep). Global body = E1's "
     "mixers on 2,048 patches, window 512 patches (1,024 bytes).", patched(2), {}),
    ("E13b", "e13b_p4_r1", "Patched coarse-to-fine, P=4, r=1, AR byte decoder (granularity sweep). Global body = E1's "
     "mixers on 1,024 patches, window 256 patches (1,024 bytes).", patched(4), {}),
    ("E13c", "e13c_p8_r1", "Patched coarse-to-fine, P=8, r=1, AR byte decoder (granularity sweep). Global body on 512 "
     "patches, window 128 patches.", patched(8), {}),
    ("E13d", "e13d_p16_r1", "Patched coarse-to-fine, P=16, r=1, AR (sweep end point; the literature reports collapse "
     "at P=16 on code from patch lag).", patched(16), {}),
    ("E14a", "e14a_p4_r1", "Loop sweep arm r=1 (same model as E13b: run once, reuse).", patched(4), {}),
    ("E14b", "e14b_p4_r2_shared", "Loop sweep: P=4, byte refinement loop r=2, SHARED weights, gated updates, "
     "pass-index embedding, re-injection, per-pass loss (linear weights).", patched(4, refine_passes=2), {}),
    ("E14c", "e14c_p4_r4_shared", "Loop sweep: P=4, r=4 shared, gated, per-pass loss.", patched(4, refine_passes=4),
     {"eval_passes": [1, 2, 4, 6, 8]}),
    ("E14d", "e14d_p4_r2_unshared", "H-C2F control: P=4, two UNSHARED refinement passes (equal FLOPs to E14b, more "
     "params).", patched(4, refine_passes=2, refine_shared=False), {}),
    ("E14e", "e14e_p4_rsampled", "Optional: P=4 shared loop trained with r ~ U{1..4} (reports the full L(r) curve, "
     "analysis section 4).", patched(4, refine_passes=2, refine_min=1, refine_max=4),
     {"eval_passes": [1, 2, 3, 4, 6, 8]}),
    ("E15a", "e15a_par_p4_r1", "parallel_refine, P=4, r=1: all 4 bytes of the next patch from the global state "
     "(an MTP-style patch head; no within-patch byte input).", patched(4, decoder="parallel_refine"), {}),
    ("E15b", "e15b_par_p4_r2", "parallel_refine, P=4, r=2, soft feedback of pass-1 guesses.",
     patched(4, decoder="parallel_refine", refine_passes=2), {}),
    ("E15c", "e15c_par_p4_r3", "parallel_refine, P=4, r=3, soft feedback.",
     patched(4, decoder="parallel_refine", refine_passes=3), {}),
    ("E16a", "e16a_p4_r2_looped_global", "E14 winner (placeholder: r=2 shared, replace once E14 is scored) with an "
     "E2-style looped global core on patches: prelude [swa] -> core [gdn, gdn, swa] x r_g -> coda [attn], "
     "r_g ~ U{1..7}, bptt 4, eval r_g=4.",
     patched(4, refine_passes=2, prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=4, loops_min=1,
             loops_max=7, bptt_loops=4), {"eval_loops": [1, 2, 4, 6, 8]}),
    ("E16b", "e16b_p4_r2_ternary_global", "E14 winner (placeholder: r=2 shared) with a TERNARY global body (BitLinear "
     "in the global blocks only; byte encoder/decoder, patch projections and head stay full precision, K4).",
     patched(4, refine_passes=2, ternary=True), {}),
]


def model_of(m: dict):
    mm = {k: v for k, v in m.items() if k not in PENDING_FIELDS}
    return make_model(ModelConfig(**mm))


def est_gpu_min(m: dict, n_bytes: float) -> float:
    """Bandwidth proxy (estimate): time ~ sum over block applications of positions x width. Narrow (d=192) byte
    layers get a 1.5x efficiency penalty (low arithmetic intensity, more launches; speculation). Calibrated so E1
    (6 blocks at d=384 per token) runs at its measured 86k tok/s."""
    P = m.get("patch_size", 0)
    r_g = m.get("loops", 4) if m.get("core") else 0
    n_glob = len(m["prelude"]) + len(m["coda"]) + r_g * len(m.get("core", []))
    if not P:
        le = n_glob * m["d_model"] / 384  # flat byte model: every block at byte resolution
    else:
        w = lambda kinds: sum(0.5 if k == "conv" else 1.0 for k in kinds)  # noqa: E731
        r = m.get("refine_passes", 1)
        if m.get("refine_max", 0) > m.get("refine_min", 0):  # r ~ U{min..max}: the mean number of passes
            r = (m["refine_min"] + m["refine_max"]) / 2
        local = (w(m["local_encoder"]) + r * w(m["refine_blocks"])) * m["d_local"] / 384 * 1.5
        le = n_glob * m["d_model"] / 384 / P + local
    if m.get("ternary"):
        le *= 1.33  # E3/E1 measured wall ratio 52.2/39.4 (staged-table.md), applied to the whole step (upper bound)
    per_byte_s = le / E1_LAYERS / E1_TOK_PER_S
    return n_bytes * per_byte_s / 60


def write_configs() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for eid, stem, desc, model, train_extra in SPECS:
        for budget, suffix in ((200_000_000, ""), (20_000_000, "_20m")):
            tr = {**copy.deepcopy(TRAIN), **train_extra, "max_bytes": budget}
            steps = -(-budget // (tr["batch_size"] * tr["seq_len"]))
            tr["warmup_steps"] = max(8, round(0.05 * steps))
            cfg = {"experiment": eid, "description": desc + ("" if not suffix else " [20M-byte sweep variant]"),
                   "budget": {"bytes": budget, "token_equivalents": round(budget / BYTES_PER_TOKEN),
                              "steps": steps, "note": "max_bytes counts training byte positions (B x T per step); "
                              f"token_equivalents = bytes / {BYTES_PER_TOKEN} (measured val bytes per BPE token)"},
                   "notes": NOTE_LR, "model": model, "train": tr}
            if stem == "e14a_p4_r1":
                cfg["same_model_as"] = "E13b"
            (OUT / f"{stem}{suffix}.json").write_text(json.dumps(cfg, indent=2) + "\n")
            rows.append({"id": eid, "file": f"experiments/staged-configs/byte/{stem}{suffix}.json", "cfg": cfg})
    return rows


def table(rows: list[dict]) -> dict:
    e1 = FusionDance(ModelConfig(**{**E1_BODY, "vocab_size": 8192, "window": 256, "max_seq_len": 1024}))
    e1_f = e1.flops_per_token(1024)
    e1_p = e1.param_counts()
    out = {"E1_reference": {"flops_per_token_train": e1_f, "params_total": e1_p["total"],
                            "params_non_embedding": e1_p["non_embedding"]},
           "bytes_per_token": BYTES_PER_TOKEN, "rows": []}
    for row in rows:
        cfg = row["cfg"]
        m = cfg["model"]
        model = model_of(m)
        pc = model.param_counts()
        if m.get("patch_size"):
            br = model.flops_breakdown(T)
            fb = br["total"]
            fb_inf = model.flops_breakdown(T, training=False)["total"]
            if m.get("refine_max", 0) > m.get("refine_min", 0):  # sampled r: mean training cost over U{min..max}
                rs = range(m["refine_min"], m["refine_max"] + 1)
                fb = sum(model.flops_per_token(T, passes=r) for r in rs) / len(rs)
        else:
            fb = model.flops_per_token(T)
            fb_inf = model.flops_per_token(T, training=False)
            br = {}
        ft = fb * BYTES_PER_TOKEN
        out["rows"].append({
            "id": row["id"], "file": row["file"], "bytes": cfg["train"]["max_bytes"],
            "params_total": pc["total"], "params_non_embedding": pc["non_embedding"],
            "params_global": pc.get("global", pc["total"]), "params_local": pc.get("local", 0),
            "flops_per_byte_train": fb, "flops_per_token_equiv_train": ft, "x_E1": ft / e1_f,
            "flops_per_byte_inference": fb_inf, "breakdown_per_byte_train": br,
            "train_flops_total": fb * cfg["train"]["max_bytes"],
            "est_gpu_min": est_gpu_min(m, cfg["train"]["max_bytes"])})
    return out


def analysis_crosscheck() -> dict:
    """notes/analysis-coarse-to-fine.md section 3 estimates vs our counting (per token-equivalent, training)."""
    res = {}
    flat = FusionDance(ModelConfig(**{**E1_BODY, "vocab_size": 256, "window": 1024, "max_seq_len": T}))
    res["flat_byte_E1_w1024"] = {"analysis_M": 325, "ours_M": flat.flops_per_token(T) * BYTES_PER_TOKEN / 1e6}
    for r, ref in ((1, 95), (2, 108), (4, 135)):
        ours = model_of(patched(4, refine_passes=r)).flops_per_token(T) * BYTES_PER_TOKEN / 1e6
        # the analysis assumed a local pass = GDN + SWA + SwiGLU; closest construction: blocks [gdn, lswa]
        alt = model_of(patched(4, refine_passes=r, refine_blocks=["gdn", "lswa"], local_encoder=["gdn"])
                       ).flops_per_token(T) * BYTES_PER_TOKEN / 1e6
        res[f"c2f_P4_r{r}"] = {"analysis_M": ref, "ours_M": ours, "ours_with_analysis_pass_gdn_swa_M": alt}
    m1 = model_of(patched(4, refine_passes=1)).flops_per_token(T)
    m2 = model_of(patched(4, refine_passes=2)).flops_per_token(T)
    res["extra_pass_M_per_token_equiv"] = {"analysis_M": 13, "ours_M": (m2 - m1) * BYTES_PER_TOKEN / 1e6}
    return res


def write_table(tab: dict, xc: dict) -> None:
    (HERE / "flops-table.json").write_text(json.dumps({**tab, "analysis_crosscheck": xc}, indent=1) + "\n")
    e1 = tab["E1_reference"]
    L = ["# Byte main quest: params, FLOPs, cost (generated by build_configs.py; do not edit)", "",
         f"Computed with `experiments/byte_model/model_bytes.py` (FusionDance's counting rule: 2 x params per "
         f"application, x3 for training, plus attention/GDN terms; the global body counted once per patch; one head "
         f"per loss-carrying pass in training). Per token-equivalent = per byte x {BYTES_PER_TOKEN}. E1 reference: "
         f"{e1['flops_per_token_train']:,.0f} train FLOPs/token, {e1['params_total']:,} params "
         f"({e1['params_non_embedding']:,} non-embedding). T = 4,096 bytes. **est. GPU-min is an estimate** "
         "(bandwidth proxy calibrated on E1's measured 86k tok/s, +-50%); SQ189 measures it. The ternary arm "
         "credits ternary with no FLOP saving.", "",
         "| id | file | bytes | params total | global | local | train FLOPs/byte | train FLOPs/token-eq | x E1 "
         "| inference FLOPs/byte | est. GPU-min |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab["rows"]:
        L.append(f"| {r['id']} | `{pathlib.Path(r['file']).name}` | {r['bytes']/1e6:.0f}M | {r['params_total']:,} | "
                 f"{r['params_global']:,} | {r['params_local']:,} | {r['flops_per_byte_train']/1e6:.2f}M | "
                 f"{r['flops_per_token_equiv_train']/1e6:.1f}M | {r['x_E1']:.3f} | "
                 f"{r['flops_per_byte_inference']/1e6:.2f}M | {r['est_gpu_min']:.1f} |")
    L += ["", "## Cross-check against notes/analysis-coarse-to-fine.md section 3 (train FLOPs per token-equivalent)",
          "", "| design | analysis estimate | ours (staged blocks: conv + GDN) | ours with the analysis's pass "
          "(GDN + SWA) |", "|---|---|---|---|"]
    for k, v in xc.items():
        alt = v.get("ours_with_analysis_pass_gdn_swa_M")
        L.append(f"| {k} | {v['analysis_M']}M | {v['ours_M']:.1f}M | {'' if alt is None else f'{alt:.1f}M'} |")
    br = next(r for r in tab["rows"] if r["id"] == "E14b" and r["bytes"] == 200_000_000)["breakdown_per_byte_train"]
    L += ["", "Breakdown, E14b (P=4, r=2), train FLOPs per byte: " +
          ", ".join(f"{k} {v/1e6:.2f}M" for k, v in br.items()), ""]
    (HERE / "flops-table.md").write_text("\n".join(L))


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    rows = write_configs()
    tab = table(rows)
    xc = analysis_crosscheck()
    write_table(tab, xc)
    print((HERE / "flops-table.md").read_text())
