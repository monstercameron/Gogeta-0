"""C3 witness (CPU): every block type is causal, and sliding-window attention cuts off exactly at the window.

Saved-output evidence for C3 (quick review 2026-11-23 17:54; tests/test_model.py needs CUDA). GDN runs through the torch
reference recurrence of gogeta/decode.py (fla's kernel needs a GPU), patched in for this check only.

Checks, seeded, fp32, CPU, small model (d 64, window 8, T 32):
  (1) causality per block type: perturbing token t changes no logit at positions < t (max |diff| must be exactly 0);
  (2) SWA cutoff: in a single-SWA-layer model, perturbing token 0 changes logits at positions < window and leaves
      positions >= window exactly unchanged (max |diff| == 0).
    python3 experiments/checks/c3_window_witness.py         # CPU, torch reference recurrence -> results/c3_window_witness.json
    python3 experiments/checks/c3_window_witness.py --gpu   # CUDA production kernels (fla), no patch -> results/c3_window_witness_gpu.json
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from gogeta.decode import _gdn_recurrence_torch  # noqa: E402
from gogeta.model import FusionDance, ModelConfig  # noqa: E402

BASE = dict(vocab_size=97, d_model=64, n_heads=4, mlp_hidden=128, window=8, max_seq_len=64, loops=2, loops_min=2,
            loops_max=2, bptt_loops=2)
T = 32


def ref_kernel(q, k, v, g, beta, scale=None, **_):
    B, Tq, H, K = q.shape
    o, *_r = _gdn_recurrence_torch(q.float(), k.float(), v.float(), g.float(), beta.float(), scale,
                                   torch.zeros(B, H, K, v.shape[-1]), Tq)
    return o.to(q.dtype), None


GPU = "--gpu" in sys.argv
DEV = "cuda" if GPU else "cpu"


def build(**kw):
    torch.manual_seed(0)
    m = FusionDance(ModelConfig(**{**BASE, **kw})).eval().to(DEV)
    if not GPU:  # the production fla kernel needs CUDA; on CPU use the torch reference recurrence
        for b in [*m.prelude, *m.core, *m.coda]:
            if b.kind == "gdn":
                b.mixer._kernel = ref_kernel
    return m


def logits(m, x):
    with torch.no_grad():
        return m(x)[0].double()


def causality(m, t):
    torch.manual_seed(1)
    x = torch.randint(0, 97, (1, T)).to(DEV)
    y = x.clone(); y[0, t] = (y[0, t] + 1) % 97
    a, b = logits(m, x), logits(m, y)
    return {"t": t, "max_diff_before_t": float((a[:, :t] - b[:, :t]).abs().max()) if t else 0.0,
            "max_diff_at_or_after_t": float((a[:, t:] - b[:, t:]).abs().max())}


def main():
    out = {"config": BASE, "T": T, "dtype": "float32 model, float64 diffs", "device": DEV, "kernels": "production (fla on CUDA, bf16 autocast inside GDN)" if GPU else "torch reference recurrence", "causality": {}}
    kinds = {"attn": dict(prelude=["attn"], core=[], coda=[]), "swa": dict(prelude=["swa"], core=[], coda=[]),
             "gdn": dict(prelude=["gdn"], core=[], coda=[]),
             "looped_hybrid": dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"])}
    ok = True
    for name, kw in kinds.items():
        m = build(**kw)
        rows = [causality(m, t) for t in (5, 16, 31)]
        out["causality"][name] = rows
        good = all(r["max_diff_before_t"] == 0.0 and r["max_diff_at_or_after_t"] > 0 for r in rows)
        ok &= good
        print(f"causality {name:14s} {'PASS' if good else 'FAIL'}  " +
              "  ".join(f"t={r['t']}: before {r['max_diff_before_t']:.1e} after {r['max_diff_at_or_after_t']:.1e}" for r in rows))
    m = build(prelude=["swa"], core=[], coda=[])
    torch.manual_seed(1)
    x = torch.randint(0, 97, (1, T)).to(DEV); y = x.clone(); y[0, 0] = (y[0, 0] + 1) % 97
    a, b = logits(m, x), logits(m, y)
    w = BASE["window"]
    inside = float((a[:, :w] - b[:, :w]).abs().max()); outside = float((a[:, w:] - b[:, w:]).abs().max())
    swa_ok = inside > 1e-4 and outside == 0.0
    ok &= swa_ok
    out["swa_cutoff"] = {"window": w, "perturbed_token": 0, "max_diff_positions_lt_window": inside,
                         "max_diff_positions_ge_window": outside, "pass": swa_ok}
    print(f"swa cutoff (window {w}): inside max {inside:.2e}, outside max {outside:.2e} -> {'PASS' if swa_ok else 'FAIL'}")
    out["all_pass"] = ok
    json.dump(out, open(os.path.join(ROOT, "experiments", "checks", "results", "c3_window_witness_gpu.json" if GPU else "c3_window_witness.json"), "w",
                        newline="\n"), indent=1)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
