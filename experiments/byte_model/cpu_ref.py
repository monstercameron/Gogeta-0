"""CPU stand-in for fla's GPU-only gated delta rule kernel, so byte models build and run on CPU.

Same recurrence as experiments/viz/export_trace.py (gated_delta_rule_ref) and experiments/staged-tests/numref.py
(delta_rule_recurrent): token-by-token, so row t depends on rows <= t only, bit for bit (what the causality test
needs; a chunked form would mix future rows into masked products). Supports the neg-eigval call signature too.

    from experiments.byte_model import cpu_ref; cpu_ref.install()   # before building a model with "gdn" blocks
Installs a stub only when fla is not importable or when force=True (tests force it so CPU runs never touch Triton).
"""
from __future__ import annotations

import sys
import types

import torch


def gated_delta_rule_ref(q, k, v, g, beta, scale=None, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                         **_):
    """q,k:[B,T,H,K] v:[B,T,H,V] g (log decay), beta:[B,T,H] -> (o [B,T,H,V], S [B,H,K,V])."""
    B, T, H, dk = q.shape
    scale = dk ** -0.5 if scale is None else scale
    dt = q.dtype
    q, k, v, g, beta = (t.to(torch.float32 if dt in (torch.bfloat16, torch.float16) else dt) for t in (q, k, v, g, beta))
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    S = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t] - pred)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
    return torch.stack(outs, 1).to(dt), S


def install(force: bool = True) -> None:
    if not force:
        try:
            import fla.ops.gated_delta_rule  # noqa: F401
            return
        except Exception:
            pass
    fla = types.ModuleType("fla")
    ops = types.ModuleType("fla.ops")
    gdr = types.ModuleType("fla.ops.gated_delta_rule")
    gdr.chunk_gated_delta_rule = gated_delta_rule_ref
    fla.ops, ops.gated_delta_rule = ops, gdr
    sys.modules.update({"fla": fla, "fla.ops": ops, "fla.ops.gated_delta_rule": gdr})
