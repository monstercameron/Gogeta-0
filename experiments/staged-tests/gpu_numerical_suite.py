"""GPU numerical suite for the fla kernels (A6; proposed queue item "A6-gpu-numerics", <= 10 GPU-minutes).

    ~/.venvs/gogeta/bin/python experiments/staged-tests/gpu_numerical_suite.py              # GPU. Only when the GPU slot is free.
    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python experiments/staged-tests/gpu_numerical_suite.py --selftest   # CPU: shim kernels

Without CUDA this exits 77 (SKIP) so run_numerical_suite.py never counts a skip as a pass. --selftest swaps the Triton kernels for
fp32 shims built on numref so the comparison / budget / reporting logic can be exercised on CPU (it validates THIS FILE, not fla).

What it adds over tests/test_gdn_kernels.py and SQ70 (which it supersedes once moved into tests/):
  1. behavioural API contract: return arity, state shape/dtype, (o, None) without output_final_state, keyword-typo swallowing,
     the positional-beta trap of fused_recurrent_gated_delta_rule (still non-finite/wrong), neg-eig flag semantics.
  2. kernel vs fp64 reference over a T x d grid (fp32 and bf16; chunk, recurrent, neg-eig): scaled error vs tolerances.tol(...),
     reported as budget utilisation (the number the CPU emulation has to be calibrated against).
  3. backward vs fp64 autograd (provisional budget = 8x the forward budget; the measured utilisation sets the real one).
  4. chunked vs streaming vs recurrent at long T (8192 and 16384, small d), carried state across arbitrary split points, and
     chunk-prefill + token decode.
  5. the real GatedDeltaNet block (fla chunk kernel, bf16/fp32) vs the fp64 numref model reference at the model budget.
  6. attention / SWA / SwiGLU / BitLinear on GPU in fp32 and bf16-autocast vs fp64 (confirms the CPU-derived module budgets).
Results: experiments/checks/results/a6_gpu_numerics.json (utilisation per cell; >1 means the CPU-derived budget is too small).
"""
from __future__ import annotations

import json
import math
import sys
import time

from _harness import ROOT, maxerr  # noqa: F401

import torch
import torch.nn.functional as F

import numref as R
import tolerances as TL

SELFTEST = "--selftest" in sys.argv
DEV = "cpu" if SELFTEST else "cuda"
RESULTS: dict = {}
D64 = torch.float64


# ------------------------------------------------------------------ kernels (real or shim)
def _shim():
    def chunk(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, **kw):
        if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta.float()) * (2.0 if allow_neg_eigval else 1.0)
        o, S = R.delta_rule_chunked(q.float(), k.float(), v.float(), g.float(), beta.float(), scale or q.shape[-1] ** -0.5,
                                    initial_state, chunk=64)
        return o.to(q.dtype), (S if output_final_state else None)

    def rec(q, k, v, g=None, gk=None, gv=None, beta=None, scale=None, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, **kw):
        if gk is not None:
            return torch.full_like(v, float("nan")), None      # the trap: what the real kernel does with garbage
        if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta.float()) * (2.0 if allow_neg_eigval else 1.0)
        beta = torch.ones_like(g) if beta is None else beta
        o, S = R.delta_rule_recurrent(q.float(), k.float(), v.float(), g.float(), beta.float(), scale or q.shape[-1] ** -0.5, initial_state)
        return o.to(q.dtype), (S if output_final_state else None)
    return chunk, rec


if SELFTEST:
    chunk_k, rec_k = _shim()
else:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_k, fused_recurrent_gated_delta_rule as rec_k  # noqa: E402


def sync():
    if not SELFTEST:
        torch.cuda.synchronize()


def record(key, util, **extra):
    RESULTS[key] = {"utilisation": util, **extra}
    return util


def inputs(B, T, H, d, seed, dtype, beta_max=1.0, raw_beta=False):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    q, k = F.normalize(r(B, T, H, d), dim=-1), F.normalize(r(B, T, H, d), dim=-1)
    v, g = r(B, T, H, d), F.logsigmoid(r(B, T, H) + 2)
    b_raw = r(B, T, H)
    beta = b_raw if raw_beta else beta_max * torch.sigmoid(b_raw)
    cast = (lambda x: x.to(torch.bfloat16)) if dtype == "bf16" else (lambda x: x)
    return [cast(x).to(DEV) for x in (q, k, v)] + [g.to(DEV), cast(beta).to(DEV)]


def ref64(q, k, v, g, beta, scale, S0=None, raw_neg=False):
    if raw_neg:
        beta = 2 * torch.sigmoid(beta.double())
    return R.delta_rule_chunked(*(x.double().cpu() for x in (q, k, v, g, beta)), scale, S0=None if S0 is None else S0.double().cpu(), chunk=64)


# ------------------------------------------------------------------ 1. behavioural API contract
def test_behavioural_contract():
    q, k, v, g, beta = inputs(1, 128, 2, 32, 0, "fp32")
    o, S = chunk_k(q, k, v, g, beta, scale=32 ** -0.5, output_final_state=True)
    assert o.shape == v.shape and S.shape == (1, 2, 32, 32) and S.dtype == torch.float32, (o.shape, S.shape, S.dtype)
    o2, S2 = chunk_k(q, k, v, g, beta, scale=32 ** -0.5)
    assert S2 is None, "without output_final_state the second return value must be None"
    o3, S3 = rec_k(q, k, v, g=g, beta=beta, scale=32 ** -0.5, output_final_state=True)
    assert S3.shape == S.shape and S3.dtype == torch.float32
    # the C2 trap: beta in the 5th positional slot is gk. Must stay detectably wrong (non-finite or far off); if fla ever
    # validates shapes and raises, that is also fine and this test records it.
    ref, _ = ref64(q, k, v, g, beta, 32 ** -0.5)
    try:
        bad, _ = rec_k(q, k, v, g, beta, scale=32 ** -0.5)
        sync()
        e = float("inf") if not torch.isfinite(bad).all() else (bad.double().cpu() - ref).abs().max().item()
        RESULTS["positional_beta_trap"] = {"err": e, "still_wrong": e > 1.0}
        assert e > 1.0, "positional beta no longer wrong: fla changed; re-audit the CONTRACT safe_positional for fused_recurrent"
    except (RuntimeError, ValueError, AssertionError) as exc:
        if isinstance(exc, AssertionError) and "no longer wrong" in str(exc):
            raise
        RESULTS["positional_beta_trap"] = {"raised": type(exc).__name__}
    good, _ = rec_k(q, k, v, g=g, beta=beta, scale=32 ** -0.5)
    assert TL.scaled_err(good.cpu(), ref) < TL.tol("gdn_recurrent", "fp32", 128, 32)
    # unknown keyword: swallowed by **kwargs (documented hazard) or rejected; record which
    try:
        o4, _ = chunk_k(q, k, v, g, beta, scale=32 ** -0.5, use_beta_sigmoid_in_kernal=True)
        RESULTS["kwarg_typo"] = {"swallowed": bool(torch.equal(o4, o2))}
    except TypeError:
        RESULTS["kwarg_typo"] = {"swallowed": False}
    # neg-eig flag semantics: raw logits + both flags == explicit beta = 2*sigmoid(raw)
    q, k, v, g, raw = inputs(1, 128, 2, 32, 1, "fp32", raw_beta=True)
    on, _ = chunk_k(q, k, v, g, raw, scale=32 ** -0.5, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    refn, _ = ref64(q, k, v, g, raw, 32 ** -0.5, raw_neg=True)
    assert TL.scaled_err(on.cpu(), refn) < TL.tol("gdn_chunk_neg", "fp32", 128, 32), "allow_neg_eigval semantics changed"


# ------------------------------------------------------------------ 2/3. kernels vs fp64 over a grid, fwd and bwd
GRID = [(300, 64), (64, 32), (1024, 32), (1024, 64), (1024, 128), (4096, 64)]


def test_kernel_vs_fp64_grid():
    worst = 0.0
    for dtype in ("fp32", "bf16"):
        for kind in ("chunk", "recurrent", "chunk_neg"):
            for T, d in GRID:
                H = 4 if (T, d) == (300, 64) else 2
                B = 2 if (T, d) == (300, 64) else 1
                neg = kind.endswith("neg")
                q, k, v, g, beta = inputs(B, T, H, d, 3, dtype, raw_beta=neg)
                ref, _ = ref64(q, k, v, g, beta, d ** -0.5, raw_neg=neg)
                if kind == "recurrent":
                    if T > 1024:
                        continue
                    o, _ = rec_k(q, k, v, g=g, beta=beta, scale=d ** -0.5)
                else:
                    kw = dict(use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True) if neg else {}
                    o, _ = chunk_k(q, k, v, g, beta, scale=d ** -0.5, **kw)
                sync()
                blk = {"chunk": "gdn_chunk", "recurrent": "gdn_recurrent", "chunk_neg": "gdn_chunk_neg"}[kind]
                u = TL.scaled_err(o.cpu(), ref) / TL.tol(blk, dtype, T, d)
                record(f"fwd/{kind}/{dtype}/T{T}/d{d}", u)
                worst = max(worst, u)
    over = {k: round(v["utilisation"], 2) for k, v in RESULTS.items() if k.startswith("fwd/") and v["utilisation"] > 1}
    assert not over, f"forward error above the CPU-derived budget (recalibrate or investigate): {over}"
    return f"worst forward budget utilisation {worst:.2f}"


def test_kernel_backward_vs_fp64():
    worst = 0.0
    for dtype in ("fp32", "bf16"):
        for kind in ("chunk", "recurrent"):
            T, d, H = 300, 64, 4
            q, k, v, g, beta = [x.requires_grad_() for x in inputs(2, T, H, d, 5, dtype)]
            w = torch.randn(2, T, H, d, generator=torch.Generator().manual_seed(9)).to(DEV)
            o = chunk_k(q, k, v, g, beta, scale=d ** -0.5)[0] if kind == "chunk" else rec_k(q, k, v, g=g, beta=beta, scale=d ** -0.5)[0]
            gk = torch.autograd.grad((o.float() * w).sum(), (q, k, v, g, beta), allow_unused=True)
            qd, kd, vd, gd, bd = [x.detach().double().cpu().requires_grad_() for x in (q, k, v, g, beta)]
            od, _ = R.delta_rule_recurrent(qd, kd, vd, gd, bd, d ** -0.5)
            gr = torch.autograd.grad((od * w.double().cpu()).sum(), (qd, kd, vd, gd, bd))
            errs = [TL.scaled_err(a.cpu(), b) for a, b in zip(gk, gr) if a is not None]
            u = max(errs) / (8 * TL.tol("gdn_chunk" if kind == "chunk" else "gdn_recurrent", dtype, T, d))
            record(f"bwd/{kind}/{dtype}", u, per_input=errs, provisional_budget="8x forward")
            worst = max(worst, u)
    assert worst <= 1.0, f"backward error above 8x the forward budget: {worst:.2f}"
    return f"worst backward utilisation (budget = 8x fwd) {worst:.2f}"


# ------------------------------------------------------------------ 4. chunked vs streaming vs recurrent at long T
def test_long_T_streaming_equivalence():
    d, H = 32, 2
    for T in (8192, 16384):
        for dtype in ("fp32", "bf16"):
            q, k, v, g, beta = inputs(1, T, H, d, 21, dtype, beta_max=2.0)
            g = g * 0.1                                     # mild decay so the state really carries information for 16k tokens
            ref, Sref = ref64(q, k, v, g, beta, d ** -0.5)
            full, Sf = chunk_k(q, k, v, g, beta, scale=d ** -0.5, output_final_state=True)
            tol = TL.tol("gdn_chunk", dtype, min(T, 4096), d) * 2           # extrapolated in T: allow 2x
            e_full = TL.scaled_err(full.cpu(), ref)
            cuts = [0, 1, 63, 64, 1000, 4097, T // 2 + 3, T - 1, T]
            S, outs = None, []
            for a, b in zip(cuts[:-1], cuts[1:]):
                o, S = chunk_k(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], scale=d ** -0.5, initial_state=S, output_final_state=True)
                outs.append(o)
            e_stream = TL.scaled_err(torch.cat(outs, 1).cpu(), ref)
            e_state = TL.scaled_err(S.cpu(), Sref)
            P = T - 128
            _, S = chunk_k(q[:, :P], k[:, :P], v[:, :P], g[:, :P], beta[:, :P], scale=d ** -0.5, output_final_state=True)
            dec = []
            for t in range(P, T):
                o, S = rec_k(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g=g[:, t:t + 1], beta=beta[:, t:t + 1], scale=d ** -0.5,
                             initial_state=S, output_final_state=True)
                dec.append(o)
            e_dec = TL.scaled_err(torch.cat(dec, 1).cpu(), ref[:, P:])
            record(f"long/T{T}/{dtype}", max(e_full, e_stream, e_dec) / tol, full=e_full, streamed=e_stream, state=e_state, prefill_decode=e_dec)
            assert max(e_full, e_stream, e_dec) <= tol, f"T={T} {dtype}: full {e_full:.1e} stream {e_stream:.1e} decode {e_dec:.1e} > {tol:.1e}"
            # the carried state through arbitrary splits must reproduce the unsplit run to within the same budget
            assert TL.scaled_err(torch.cat(outs, 1).cpu(), full.cpu()) <= tol


# ------------------------------------------------------------------ 5/6. model-level and module-level on the GPU
def test_gdn_block_and_model_vs_fp64_reference():
    from gogeta.model import FusionDance, GatedDeltaNet, ModelConfig
    for neg in (False, True):
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=128, d_model=128, n_heads=2, mlp_hidden=256, window=64, max_seq_len=512, gdn_neg_eigval=neg,
                          prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=3)
        m = FusionDance(cfg).eval()
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() >= 2:
                    p.normal_(0, 1 / math.sqrt(128))
        idx = torch.randint(0, 128, (2, 256))
        R.RMS_EPS = torch.finfo(torch.float32).eps
        ref = R.model_ref(m, idx)
        if SELFTEST:
            for mod in m.modules():
                if isinstance(mod, GatedDeltaNet):
                    mod._kernel = chunk_k
        m = m.to(DEV)
        for dtype in ("fp32", "bf16"):
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16, enabled=(dtype == "bf16")):
                lg, _ = m(idx.to(DEV))
            # a model containing GDN inherits the chunk kernel's (TF32-like) error: budget = max(model budget, kernel budget)
            budget = max(TL.tol("model", dtype, L=m.effective_depth()), TL.tol("gdn_chunk_neg" if neg else "gdn_chunk", dtype, 256, 64))
            u = TL.scaled_err(lg.cpu(), ref) / budget
            record(f"model/{'neg' if neg else 'pos'}/{dtype}", u)
            assert u <= 1.0, f"model {dtype} neg={neg}: utilisation {u:.2f}"


def test_modules_vs_fp64_on_device():
    import error_budget as EB
    for block in ("attn", "swa", "mlp", "bitlinear"):
        fn = EB.CELLS[block][0]
        for dtype in ("fp32", "bf16"):
            for T, d in ((256, 32), (1024, 64)):
                e = _module_on_device(fn, T, d, dtype)
                u = e / TL.tol(block, dtype, T, d)
                record(f"module/{block}/{dtype}/T{T}/d{d}", u)
                assert u <= 1.0, f"{block} {dtype} T={T} d={d}: utilisation {u:.2f}"


def _module_on_device(cell_fn, T, d, dtype):
    """Re-run an error_budget module cell, but evaluate the low-precision side on DEV (fp64 reference stays on CPU)."""
    import contextlib
    import error_budget as EB
    orig_ac = torch.autocast
    patched = {}

    def to_dev(mod_fp32, x):
        mod_fp32 = mod_fp32.to(DEV)
        with (orig_ac(DEV, dtype=torch.bfloat16) if dtype == "bf16" else contextlib.nullcontext()):
            return mod_fp32(x.to(DEV)).cpu()
    # error_budget's cells build + evaluate in one function; the device variant re-implements the 3 lines it needs
    block_make = {"attn": EB._attn(None), "swa": EB._attn(64),
                  "mlp": lambda d_: (EB.SwiGLU(EB._cfg(d_)), 2 * d_), "bitlinear": lambda d_: (EB.BitLinear(2 * d_, 2 * d_, bias=False), 2 * d_)}
    name = next(k for k, v in EB.CELLS.items() if v[0] is cell_fn)
    torch.manual_seed(11)
    mod, D = block_make[name](d)
    with torch.no_grad():
        for p in mod.parameters():
            p.normal_(0, 1 / math.sqrt(D))
    x = torch.randn(1, T, D)
    ref = mod.double()(x.double()).detach()
    out = to_dev(mod.float(), x)
    return TL.scaled_err(out, ref)


def main():
    if not SELFTEST and not torch.cuda.is_available():
        print("SKIP: no CUDA device (GPU suite). Use --selftest to exercise this file on CPU with shim kernels.")
        return 77
    t0 = time.time()
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        t = time.time()
        try:
            out = f()
            print(f"PASS  {n}  ({time.time() - t:.0f}s)  {out or ''}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {n}  ({time.time() - t:.0f}s)  {type(e).__name__}: {str(e)[:400]}")
    path = ROOT / "experiments" / "checks" / "results" / ("a6_gpu_numerics_selftest.json" if SELFTEST else "a6_gpu_numerics.json")
    if "--no-save" not in sys.argv:
        path.write_text(json.dumps({"selftest": SELFTEST, "device": DEV, "elapsed_s": round(time.time() - t0), "results": RESULTS}, indent=1, default=str))
        print("wrote", path)
    print("ALL OK" if not bad else f"{bad} FAILED", f"({time.time() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
