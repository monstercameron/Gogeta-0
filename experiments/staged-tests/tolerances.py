"""Derived numerical tolerances (A6, C198). Replaces the ad-hoc 2e-3 (fp32) / 3e-2 (bf16) of tests/test_gdn_kernels.py.

    from tolerances import tol, scaled_err, assert_close
    assert_close(kernel_out, reference_fp64, "gdn_chunk", "bf16", T=300, d=64)

All tolerances are on the SCALED error  max|x - ref| / max|ref|  (scale-free). The constants live in error_budget.json,
written by `error_budget.py --write` (CPU emulation, SAFETY=2 over the worst observed cell). To change a tolerance, re-derive
it; do not edit by hand. Blocks: attn swa mlp bitlinear gdn_chunk gdn_recurrent gdn_chunk_neg model.
dtype names: "fp32" | "bf16" (for gdn_chunk "fp32" means the TF32-dot regime Triton uses for fp32 operands).
"""
from __future__ import annotations

import json
import pathlib

_J = json.loads((pathlib.Path(__file__).resolve().parent / "error_budget.json").read_text())
ALIASES = {"gdn": "gdn_chunk", "gdn_neg": "gdn_chunk_neg", "sdpa": "attn", "gdn_kernel": "gdn_chunk"}
ABSOLUTE_FLOOR = 1e-12


def tol(block: str, dtype: str, T: int | None = None, d: int | None = None, L: int | None = None,
        ternary: bool = False, what: str = "logits") -> float:
    block = ALIASES.get(block, block)
    if block == "model":
        f = _J["model_fit"][dtype]
        if what == "loss":   # ABSOLUTE |dloss|
            return f["ternary_loss_abs_flat"] if ternary else f["c_loss_abs"] * _J["units"][dtype] * L
        return f["ternary_logits_flat"] if ternary else f["c_logits"] * _J["units"][dtype] * L
    b = _J["blocks"][f"{block}/{dtype}"]
    # extrapolation beyond the measured grid is allowed but flagged by `in_grid`; exponents were clipped to [0, 1]
    return max(ABSOLUTE_FLOOR, b["c"] * b["u"] * float(T) ** b["a"] * float(d) ** b["b"])


def in_grid(block: str, dtype: str, T: int, d: int) -> bool:
    cells = _J["blocks"][f"{ALIASES.get(block, block)}/{dtype}"]["cells"]
    Ts = sorted({int(k.split(",")[0]) for k in cells})
    ds = sorted({int(k.split(",")[1]) for k in cells})
    return Ts[0] <= T <= Ts[-1] and ds[0] <= d <= ds[-1]


def scaled_err(x, ref) -> float:
    return ((x.double() - ref.double()).abs().max() / ref.double().abs().max().clamp(min=1e-30)).item()


def assert_close(x, ref, block, dtype, T=None, d=None, L=None, ternary=False, label=""):
    e, t = scaled_err(x, ref), tol(block, dtype, T, d, L, ternary)
    assert e <= t, f"{label or block}/{dtype}: scaled error {e:.2e} exceeds the derived budget {t:.2e} (T={T}, d={d}, L={L})"
    return e / t   # budget utilisation; < 1 passes


def units() -> dict:
    return dict(_J["units"])
