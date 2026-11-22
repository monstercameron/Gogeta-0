"""Mixed-precision error budget (A6, C198): derive, on CPU, the expected max error of every block in fp32 / bf16 relative to an
fp64 reference, as a function of sequence length T and width d, and turn it into the tolerances the tests cite.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/staged-tests/error_budget.py [--quick] [--write]

Quantity: scaled error  e = max|x - ref64| / max|ref64|  (so the budget is independent of the block's output scale).
Model:    e_max(T, d) <= c * u * T^a * d^b,  u = unit roundoff of the compute precision, (a, b) fitted by least squares on the
          per-cell maxima over seeds, c = SAFETY * the largest observed ratio  (so every measured cell is covered with margin).
Regimes (what the project really runs, train.py uses torch.autocast bf16 with fp32 master weights):
    fp32      module in fp32 vs fp64 (same weights/inputs)
    bf16      the same fp32 module under torch.autocast(cpu, bfloat16): matmuls in bf16, norm/softmax in fp32
    gdn kernels (CPU emulation, see below): fp32 recurrence; fp32 chunk with TF32-rounded dot operands; bf16 chunk/recurrent.
Emulation caveat (named GPU confirmation step: queue item "A6-gpu-numerics"): the GDN chunk kernel is emulated by rounding
every matmul OPERAND to the dot precision (tf32: 10-bit mantissa, bf16: 7-bit), accumulating in fp32, rounding the output to the
storage dtype. The emulation is checked against the GPU numbers already in the repo (C1: chunk fp32 2.2e-4, bf16 8.0e-4 absolute at
T=300, d=64; SQ70: recurrent fp32 3.0e-8, bf16 4.0e-4); the GPU run re-measures the whole grid and flags any cell above budget.
"""
from __future__ import annotations

import json
import math
import sys
import time
from contextlib import nullcontext

from _harness import HERE, run_module  # noqa: F401

import torch
import torch.nn.functional as F

import numref as R
from gogeta.model import Attention, BitLinear, FusionDance, GatedDeltaNet, ModelConfig, Rotary, SwiGLU

D64 = torch.float64
U = {"fp64": 2.0 ** -53, "fp32": 2.0 ** -24, "tf32": 2.0 ** -11, "bf16": 2.0 ** -8}   # unit roundoff (half-ulp, relative)
SAFETY = 2.0
OUT = HERE / "error_budget.json"


# ------------------------------------------------------------------ rounding emulation
def rnd_tf32(x):
    b = x.float().contiguous().view(torch.int32)
    return ((b + 0x1000) & ~0x1FFF).view(torch.float32)


RND = {"fp32": lambda x: x.float(), "tf32": rnd_tf32, "bf16": lambda x: x.to(torch.bfloat16).float()}


def gdn_chunk_emul(q, k, v, g, beta, scale, dot, chunk=64):
    """chunk_gated_delta_rule emulation: every matmul operand rounded to `dot` precision, fp32 accumulation / elementwise."""
    r = RND[dot]
    mm = lambda a, b: r(a) @ r(b)  # noqa: E731
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    B, T, H, K = q.shape
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    g, beta = g.transpose(1, 2), beta.transpose(1, 2)
    S = q.new_zeros(B, H, K, v.shape[-1])
    outs = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        C = e - s
        qc, kc, vc, gc, bc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], g[:, :, s:e], beta[:, :, s:e]
        G = gc.cumsum(-1)
        D = G[..., :, None] - G[..., None, :]
        strict = torch.tril(torch.ones(C, C, dtype=torch.bool), -1)
        incl = torch.tril(torch.ones(C, C, dtype=torch.bool))
        A = bc[..., :, None] * D.masked_fill(~strict, float("-inf")).exp() * mm(kc, kc.transpose(-1, -2))
        gam = G.exp()
        rhs = bc[..., None] * (vc - gam[..., None] * mm(kc, S))
        W = torch.linalg.solve_triangular(torch.eye(C) + A, rhs, upper=False)
        P = D.masked_fill(~incl, float("-inf")).exp() * mm(qc, kc.transpose(-1, -2))
        outs.append(scale * (gam[..., None] * mm(qc, S) + mm(P, W)))
        tail = (G[..., -1:] - G).exp()
        S = G[..., -1].exp()[..., None, None] * S + mm(kc.transpose(-1, -2), tail[..., None] * W)
    return torch.cat(outs, 2).transpose(1, 2)


def gdn_rec_emul(q, k, v, g, beta, scale):
    return R.delta_rule_recurrent(*(x.float() for x in (q, k, v, g, beta)), scale)[0]


# ------------------------------------------------------------------ cells: each returns (scaled error) for one seed
def scaled(x, ref):
    return ((x.double() - ref).abs().max() / ref.abs().max().clamp(min=1e-30)).item()


def gdn_inputs(T, d, seed, H=2, beta_max=1.0):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    q, k = F.normalize(r(1, T, H, d), dim=-1), F.normalize(r(1, T, H, d), dim=-1)
    v = r(1, T, H, d)
    g = F.logsigmoid(r(1, T, H) + 2)
    beta = beta_max * torch.sigmoid(r(1, T, H))
    return q, k, v, g, beta


def cell_gdn(kind, dtype):
    def f(T, d, seed):
        q, k, v, g, beta = gdn_inputs(T, d, seed, beta_max=2.0 if kind.endswith("neg") else 1.0)
        if dtype == "bf16":   # inputs are stored in bf16 (as in the model on CUDA): both sides see the rounded inputs
            q, k, v, beta = (x.to(torch.bfloat16).float() for x in (q, k, v, beta))
        ref, _ = R.delta_rule_recurrent(*(x.double() for x in (q, k, v, g, beta)), d ** -0.5)
        if kind.startswith("chunk"):
            out = gdn_chunk_emul(q, k, v, g, beta, d ** -0.5, "tf32" if dtype == "fp32" else "bf16")
        else:
            out = gdn_rec_emul(q, k, v, g, beta, d ** -0.5)
        if dtype == "bf16":
            out = out.to(torch.bfloat16)
        return scaled(out, ref)
    return f


def _cfg(d, H=2, window=64, **kw):
    return ModelConfig(vocab_size=64, d_model=H * d, n_heads=H, mlp_hidden=2 * H * d, window=window, max_seq_len=4096, **kw)


def _module_cell(make, in_dim):
    def f(T, d, seed, dtype, wscale=1.0):
        torch.manual_seed(seed)
        mod, D = make(d)
        with torch.no_grad():
            for p in mod.parameters():
                p.normal_(0, wscale / math.sqrt(D))
        x = torch.randn(1, T, in_dim(D))
        ref = mod.double()(x.double())
        mod = mod.float()
        with torch.autocast("cpu", dtype=torch.bfloat16) if dtype == "bf16" else nullcontext():
            out = mod(x)
        return scaled(out, ref.detach())
    return f


def _attn(window):
    def make(d):
        cfg = _cfg(d, window=window or 4096)
        return Attention(cfg, window, Rotary(cfg.head_dim, 4096, cfg.rope_base)), cfg.d_model
    return make


CELLS = {  # block -> (cell function, grid of T, grid of d, dtypes)
    "attn": (_module_cell(_attn(None), lambda D: D), [64, 256, 1024, 2048], [16, 32, 64], ["fp32", "bf16"]),
    "swa": (_module_cell(_attn(64), lambda D: D), [128, 512, 2048], [16, 32, 64], ["fp32", "bf16"]),
    "mlp": (_module_cell(lambda d: (SwiGLU(_cfg(d)), 2 * d), lambda D: D), [64, 256, 1024], [16, 32, 64, 128], ["fp32", "bf16"]),
    "bitlinear": (_module_cell(lambda d: (BitLinear(2 * d, 2 * d, bias=False), 2 * d), lambda D: D), [64, 256, 1024], [16, 32, 64, 128],
                  ["fp32", "bf16"]),
    "gdn_chunk": (None, [64, 256, 1024, 4096], [16, 32, 64, 128], ["fp32", "bf16"]),
    "gdn_recurrent": (None, [64, 256, 1024, 4096], [16, 32, 64, 128], ["fp32", "bf16"]),
    "gdn_chunk_neg": (None, [64, 256, 1024, 4096], [16, 32, 64], ["fp32", "bf16"]),
}
UNIT = {("attn", "fp32"): "fp32", ("attn", "bf16"): "bf16", ("swa", "fp32"): "fp32", ("swa", "bf16"): "bf16",
        ("mlp", "fp32"): "fp32", ("mlp", "bf16"): "bf16", ("bitlinear", "fp32"): "fp32", ("bitlinear", "bf16"): "bf16",
        ("gdn_chunk", "fp32"): "tf32", ("gdn_chunk", "bf16"): "bf16", ("gdn_chunk_neg", "fp32"): "tf32",
        ("gdn_chunk_neg", "bf16"): "bf16", ("gdn_recurrent", "fp32"): "fp32", ("gdn_recurrent", "bf16"): "bf16"}


def fit(cells):
    """cells: {(T,d): max error}. OLS of log e on log T, log d -> (c0, a, b); then covering coefficient."""
    import numpy as np
    keys = sorted(cells)
    X = np.array([[1.0, math.log(T), math.log(d)] for T, d in keys])
    y = np.array([math.log(max(cells[k], 1e-30)) for k in keys])
    (c0, a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(min(max(a, 0.0), 1.0)), float(min(max(b, 0.0), 1.0))       # clip: error never shrinks with size, ~sqrt..linear
    return a, b


def run(quick=False, seeds=3):
    res = {}
    for block, (fn, Ts, ds, dtypes) in CELLS.items():
        if quick:
            Ts, ds = Ts[:3], ds[:2]
        for dtype in dtypes:
            cells = {}
            for T in Ts:
                for d in ds:
                    errs = []
                    for s in range(seeds):
                        if fn is None:
                            e = cell_gdn(block.replace("gdn_", "") if False else block[4:], dtype)(T, d, s)
                        else:
                            e = fn(T, d, s, dtype)
                        errs.append(e)
                    cells[(T, d)] = max(errs)
            a, b = fit(cells)
            u = U[UNIT[(block, dtype)]]
            ratio = max(e / (u * T ** a * d ** b) for (T, d), e in cells.items())
            res[f"{block}/{dtype}"] = dict(unit=UNIT[(block, dtype)], u=u, a=round(a, 3), b=round(b, 3), c=round(SAFETY * ratio, 3),
                                           c_fit_max=round(ratio, 3), worst_observed=max(cells.values()),
                                           cells={f"{T},{d}": v for (T, d), v in cells.items()})
            print(f"{block:14s} {dtype:5s} u={u:.2e} fit a(T)={a:.2f} b(d)={b:.2f}  c={SAFETY * ratio:6.2f}  "
                  f"worst scaled err {max(cells.values()):.2e}  tol@(T={Ts[-1]},d={ds[-1]})="
                  f"{SAFETY * ratio * u * Ts[-1] ** a * ds[-1] ** b:.2e}", flush=True)
    return res


# ------------------------------------------------------------------ whole-model budgets (depth dependence)
def gdn_outnorm_regime():
    """Informational: how often does GatedDeltaNet.out_norm (RMSNorm over the kernel output o, eps=finfo(fp32).eps=1.2e-7) see
    mean(o^2) of the same order as eps? There the norm is not a normalisation and fp64 != fp32 semantically."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=128, n_heads=4, mlp_hidden=256, window=32, max_seq_len=512)
    g = GatedDeltaNet(cfg).eval()
    g._kernel = R.make_cpu_kernel()
    seen = {}
    g.out_norm.register_forward_hook(lambda m, i, o: seen.setdefault("o", i[0].detach()))
    with torch.no_grad():
        g(torch.randn(2, 256, 128))
    ms = seen["o"].pow(2).mean(-1)
    eps = torch.finfo(torch.float32).eps
    r = dict(median_rms=ms.median().sqrt().item(), sqrt_eps=eps ** 0.5, frac_ms_below_eps=(ms < eps).float().mean().item(),
             frac_ms_below_10eps=(ms < 10 * eps).float().mean().item())
    print("GDN out_norm regime at default init (x~N(0,1)):", {k: round(v, 6) for k, v in r.items()})
    return r


def model_cells(seeds=3):
    R.RMS_EPS = torch.finfo(torch.float32).eps   # reference uses the run precision's eps: isolate rounding, not eps semantics
    out = {}
    for name, kw in {"E0-like attn x6": dict(prelude=["attn"], core=["attn"], coda=["attn"], loops=4),
                     "looped swa/gdn x4": dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=4),
                     "looped swa/gdn x7": dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=7)}.items():
        for ternary in (False, True):
            for dtype in ("fp32", "bf16"):
                errs, derr = [], []
                for s in range(seeds):
                    torch.manual_seed(s)
                    cfg = ModelConfig(vocab_size=128, d_model=64, n_heads=2, mlp_hidden=128, window=32, max_seq_len=512,
                                      ternary=ternary, **kw)
                    m = FusionDance(cfg).eval()
                    with torch.no_grad():
                        for p in m.parameters():
                            if p.dim() >= 2:
                                p.normal_(0, 1.0 / math.sqrt(64))
                    for mod in m.modules():
                        if isinstance(mod, GatedDeltaNet):
                            mod._kernel = R.make_cpu_kernel()      # fp64 kernel: GDN-kernel rounding is budgeted separately
                    idx = torch.randint(0, 128, (2, 256))
                    tgt = torch.randint(0, 128, (2, 256))
                    ref_logits = R.model_ref(m, idx)
                    ref_loss = F.cross_entropy(ref_logits.reshape(-1, 128), tgt.reshape(-1))
                    with torch.no_grad(), (torch.autocast("cpu", dtype=torch.bfloat16) if dtype == "bf16" else nullcontext()):
                        lg, loss = m(idx, tgt)
                    errs.append(scaled(lg, ref_logits.detach()))
                    derr.append(abs(loss.item() - ref_loss.item()))
                out[f"{name}{' ternary' if ternary else ''}/{dtype}"] = dict(
                    L_eff=m.effective_depth(), logits_scaled_err=max(errs), loss_abs_err=max(derr))
                print(f"model {name:20s}{' ternary' if ternary else '        '} {dtype:5s} L_eff={m.effective_depth():2d} "
                      f"logits scaled err {max(errs):.2e}  |dloss| {max(derr):.2e}", flush=True)
    return out


# absolute max errors recorded ON THE GPU (experiments/checks/results/sq_hw_fla_recurrent.json, B=2 T=300 H=4 K=V=64) and C1
GPU_RECORDS = {   # (kind, dtype) -> recorded max |kernel - fp32 loop reference| (absolute)
    ("chunk", "fp32"): 2.17e-4, ("chunk", "bf16"): 7.29e-4, ("recurrent", "fp32"): 2.98e-8, ("recurrent", "bf16"): 3.96e-4,
    ("chunk_neg", "fp32"): 4.79e-4, ("chunk_neg", "bf16"): 1.44e-3,
}


def compare_with_gpu(seeds=3):
    """Run the CPU emulation at the exact GPU test configuration and put the numbers next to the recorded GPU ones."""
    rows = {}
    for (kind, dtype), rec in GPU_RECORDS.items():
        errs, mx = [], []
        for s in range(seeds):
            gen = torch.Generator().manual_seed(100 + s)
            r = lambda *sh: torch.randn(*sh, generator=gen)  # noqa: E731
            B, T, H, d = 2, 300, 4, 64
            q, k = F.normalize(r(B, T, H, d), dim=-1), F.normalize(r(B, T, H, d), dim=-1)
            v, g = r(B, T, H, d), F.logsigmoid(r(B, T, H) + 2)
            beta = (2.0 if kind.endswith("neg") else 1.0) * torch.sigmoid(r(B, T, H))
            if dtype == "bf16":
                q, k, v, beta = (x.to(torch.bfloat16).float() for x in (q, k, v, beta))
            ref, _ = R.delta_rule_recurrent(*(x.double() for x in (q, k, v, g, beta)), d ** -0.5)
            out = (gdn_chunk_emul(q, k, v, g, beta, d ** -0.5, "tf32" if dtype == "fp32" else "bf16") if kind.startswith("chunk")
                   else gdn_rec_emul(q, k, v, g, beta, d ** -0.5))
            if dtype == "bf16":
                out = out.to(torch.bfloat16)
            errs.append((out.double() - ref).abs().max().item())
            mx.append(ref.abs().max().item())
        rows[f"{kind}/{dtype}"] = dict(gpu_recorded=rec, cpu_emulated_median=sorted(errs)[len(errs) // 2], cpu_emulated_max=max(errs),
                                       ref_absmax=sum(mx) / len(mx))
        print(f"GPU vs CPU-emulation  {kind:10s} {dtype:5s} recorded {rec:.2e}   emulated {rows[f'{kind}/{dtype}']['cpu_emulated_median']:.2e}"
              f" (max {max(errs):.2e})   ratio {rec / rows[f'{kind}/{dtype}']['cpu_emulated_median']:.2f}", flush=True)
    return rows


def fit_model(mod):
    """tol_model = c * u * L_eff  (non-ternary), flat flip-dominated budget for ternary. c = SAFETY * worst ratio."""
    fit = {}
    for dtype, u in (("fp32", U["fp32"]), ("bf16", U["bf16"])):
        rat = [v["logits_scaled_err"] / (u * v["L_eff"]) for k, v in mod.items() if k.endswith("/" + dtype) and "ternary" not in k]
        flat = [v["logits_scaled_err"] for k, v in mod.items() if k.endswith("/" + dtype) and "ternary" in k]
        lrat = [v["loss_abs_err"] / (u * v["L_eff"]) for k, v in mod.items() if k.endswith("/" + dtype) and "ternary" not in k]
        lflat = [v["loss_abs_err"] for k, v in mod.items() if k.endswith("/" + dtype) and "ternary" in k]
        fit[dtype] = dict(c_logits=SAFETY * max(rat), c_loss_abs=SAFETY * max(lrat), ternary_logits_flat=SAFETY * max(flat),
                          ternary_loss_abs_flat=SAFETY * max(lflat))
    return fit


def main(argv):
    quick = "--quick" in argv
    t0 = time.time()
    res = run(quick=quick)
    gpu = compare_with_gpu()
    regime = gdn_outnorm_regime()
    mod = model_cells(seeds=2 if quick else 3)
    # GPU calibration: where the recorded GPU error exceeds the emulation's worst case, scale the coefficient up by the ratio
    # (the emulation is a lower bound there). Visible in the JSON as gpu_calibration; the GPU queue item re-measures it.
    calib = {}
    for key, r in gpu.items():
        kind, dtype = key.split("/")
        blk = {"chunk": "gdn_chunk", "recurrent": "gdn_recurrent", "chunk_neg": "gdn_chunk_neg"}[kind]
        f = max(1.0, r["gpu_recorded"] / r["cpu_emulated_max"])
        if f > 1.0:
            res[f"{blk}/{dtype}"]["c"] = round(res[f"{blk}/{dtype}"]["c"] * f, 3)
            calib[f"{blk}/{dtype}"] = round(f, 3)
    print("GPU calibration factors applied to c:", calib)
    mfit = fit_model(mod)
    print("model fit:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in mfit.items()})
    if "--write" in argv:
        OUT.write_text(json.dumps({"safety": SAFETY, "units": U, "blocks": {k: {kk: vv for kk, vv in v.items() if kk != "cells"} | {"cells": v["cells"]}
                                                                            for k, v in res.items()},
                                   "model": mod, "model_fit": mfit, "gpu_vs_cpu_emulation": gpu, "gpu_calibration": calib, "gdn_outnorm_regime": regime, "generated_by": "error_budget.py", "elapsed_s": round(time.time() - t0)}, indent=1))
        print("wrote", OUT)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
