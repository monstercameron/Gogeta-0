"""Tests of the mixed-precision budget itself (A6, C198): it must cover the GPU data we already have, be reproducible on CPU,
and be tighter than the ad-hoc tolerances it replaces.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python experiments/staged-tests/test_error_budget.py"""
from __future__ import annotations

from _harness import HERE, run_module

import json

import torch

import error_budget as EB
import numref as R
import tolerances as TL

BUDGET_JSON = HERE / "error_budget.json"


def test_budget_covers_recorded_gpu_errors():
    """The GPU errors already in the repo (SQ70 / C1), converted to scaled error with the CPU-emulated |ref| max, sit inside the budget."""
    rows = json.loads(BUDGET_JSON.read_text())["gpu_vs_cpu_emulation"]
    names = {"chunk": "gdn_chunk", "recurrent": "gdn_recurrent", "chunk_neg": "gdn_chunk_neg"}
    util = {}
    for key, r in rows.items():
        kind, dtype = key.split("/")
        e = r["gpu_recorded"] / r["ref_absmax"]
        t = TL.tol(names[kind], dtype, T=300, d=64)
        util[key] = e / t
        assert e <= t, f"GPU record {key}: scaled {e:.2e} > budget {t:.2e}"
    return "budget utilisation of the GPU records: " + ", ".join(f"{k}:{v:.2f}" for k, v in util.items())


def test_budget_beats_the_ad_hoc_tolerances():
    """Old: 2e-3 (fp32) and 3e-2 (bf16), absolute, for every kernel. The derived budgets are tighter (by 3 orders of magnitude for
    the true-fp32 recurrent path), so a 100x regression that used to pass is now caught."""
    assert TL.tol("gdn_recurrent", "fp32", T=300, d=64) < 2e-3 / 1000
    assert TL.tol("gdn_chunk", "bf16", T=300, d=64) < 3e-2
    assert TL.tol("attn", "fp32", T=1024, d=64) < 1e-5 and TL.tol("attn", "bf16", T=1024, d=64) < 3e-2
    return (f"gdn_recurrent fp32 {TL.tol('gdn_recurrent', 'fp32', T=300, d=64):.1e} vs 2e-3; "
            f"gdn_chunk fp32 {TL.tol('gdn_chunk', 'fp32', T=300, d=64):.1e} (TF32 dot); "
            f"gdn_chunk bf16 {TL.tol('gdn_chunk', 'bf16', T=300, d=64):.1e} vs 3e-2")


def test_budget_ordering_is_sane():
    for blk in ("attn", "swa", "mlp", "gdn_chunk", "gdn_recurrent"):
        assert TL.tol(blk, "fp32", T=256, d=64) < TL.tol(blk, "bf16", T=256, d=64)
    assert TL.tol("gdn_recurrent", "fp32", T=256, d=64) < TL.tol("gdn_chunk", "fp32", T=256, d=64)
    assert TL.tol("model", "bf16", L=23) > TL.tol("model", "bf16", L=6) > TL.tol("model", "fp32", L=6)


def test_budget_reproduces_on_cpu_fresh_seeds():
    """Re-measure a few cells with NEW seeds (the budget was fitted on seeds 0-2): all must be within the budget."""
    worst = 0.0
    for block, T, d in [("gdn_chunk", 256, 32), ("gdn_recurrent", 1024, 16), ("gdn_chunk_neg", 512, 32)]:
        fn = EB.cell_gdn(block[4:], "bf16")
        for seed in (7, 8):
            e = fn(T, d, seed)
            worst = max(worst, e / TL.tol(block, "bf16", T=T, d=d))
            assert e <= TL.tol(block, "bf16", T=T, d=d), (block, T, d, seed, e)
    fn = EB.CELLS["attn"][0]
    for dtype in ("fp32", "bf16"):
        e = fn(512, 32, 9, dtype)
        worst = max(worst, e / TL.tol("attn", dtype, T=512, d=32))
        assert e <= TL.tol("attn", dtype, T=512, d=32), (dtype, e)
    return f"worst budget utilisation on fresh seeds: {worst:.2f}"


def test_bf16_rounding_alone_exceeds_fp32_budget():
    """Budget is not vacuous: rounding a perfect result to bf16 breaks the true-fp32 tolerance and respects the bf16 one."""
    q, k, v, g, beta = (x.double() for x in EB.gdn_inputs(256, 32, 0))
    o, _ = R.delta_rule_chunked(q, k, v, g, beta, 32 ** -0.5, chunk=64)
    e = TL.scaled_err(o.to(torch.bfloat16), o)
    assert e > TL.tol("gdn_recurrent", "fp32", T=256, d=32)
    assert e < TL.tol("gdn_chunk", "bf16", T=256, d=32)
    # FINDING (not an assertion): the GPU-calibrated fp32 *chunk* budget is as loose as bf16 output rounding, because fla's chunk
    # kernel in fp32 behaves like a TF32 kernel. A test at that budget cannot tell fp32 from bf16-rounded output.
    return f"bf16 rounding {e:.1e}; chunk-fp32 budget {TL.tol('gdn_chunk', 'fp32', T=256, d=32):.1e}; recurrent-fp32 budget {TL.tol('gdn_recurrent', 'fp32', T=256, d=32):.1e}"


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
