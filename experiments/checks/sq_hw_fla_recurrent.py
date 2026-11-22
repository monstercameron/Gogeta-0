"""SQ70 (hardware research): is the fla fused-recurrent GDN "NaN on this GPU" (claim C2) a call-site bug?

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_fla_recurrent.py          # GPU, ~1-3 min (Triton JIT)

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, section "fla recurrent NaN". Claims C110, C111 (and C2).

Why we suspect a call-site bug (read from the installed fla 0.6.0 source, not run):
  fla/ops/gated_delta_rule/fused_recurrent.py defines
      fused_recurrent_gated_delta_rule(q, k, v, g=None, gk=None, gv=None, beta=None, scale=None, ...)
  but tests/test_gdn_kernels.py (and sq_kda_kernel.py's GDN re-run) call it as
      fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=scale)
  so `beta` lands in `gk` (a per-KEY log decay of shape [B,T,H,K]) and the real beta defaults to ones.
  The kernel then reads gk with stride H*K from a [B,T,H] tensor: out-of-bounds reads + exp() of garbage.
  chunk_gated_delta_rule's 5th positional argument IS beta, which is why the chunk test passes.

What is measured
  1. The old positional call (reproduces the reported NaN / garbage?) vs the keyword call g=..., beta=...,
     both against a float32 loop reference, fp32 and bf16 inputs (B=2, T=300, H=4, K=V=64, as the test).
  2. The real decode path: chunk prefill (first 100 tokens, output_final_state=True), then 1-token steps of
     fused_recurrent with the carried state, vs the reference on the remaining tokens.
  3. Negative-eigenvalue mode (use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True; beta = 2*sigmoid(raw)),
     as used by E11 / SQ2.
  4. Speed of one decode step at B=16, H=6 (the RL / speculative-decoding regime): fused_recurrent vs chunk T=1.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CHECK_ID, FILE_STEM = "SQ70", "sq_hw_fla_recurrent"
CLAIMS = ["C110", "C111", "C2"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "The positional call reproduces the problem (non-finite output, or max|err| > 1 vs the reference) in fp32 and "
    "bf16. The keyword call matches the fp32 reference like the chunk kernel does (fp32 max|err| ~1e-5..1e-3, bf16 "
    "~1e-2). The prefill + 1-token-step decode path matches the reference within the same tolerances, and so does "
    "the negative-eigenvalue mode. A 1-token fused_recurrent step at B=16, H=6 takes 0.02-0.2 ms (launch-bound), "
    "at least 2x faster than calling the chunk kernel with T=1."
)
CRITERION = (
    "C110 (the C2 NaN is a call-site bug in our tests, not a GPU/kernel defect) is VERIFIED if the positional call "
    "is non-finite or has max|err| > 1 AND the keyword call is finite with max|err| < 2e-3 (fp32) and < 3e-2 (bf16), "
    "the tests/test_gdn_kernels.py tolerances. C111 (fused_recurrent_gated_delta_rule is usable for decoding on this "
    "GPU) is VERIFIED if, in fp32 and bf16, both the prefill+step path and the negative-eigenvalue path are finite "
    "and within those tolerances. If C110 is VERIFIED, C2 changes from REFUTED to WRONG TEST (re-test passes)."
)
SEED = 0
DEV = "cuda"
B, T, H, K, V = 2, 300, 4, 64, 64
PREFILL = 100
TOL = {torch.float32: 2e-3, torch.bfloat16: 3e-2}


def reference(q, k, v, g, beta, scale, h0=None):
    """float32 loop: S <- exp(g) S; S <- S + k (beta (v - S^T k))^T; o = scale * S^T q (per head)."""
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    Bq, Tq, Hq, Kq = q.shape
    S = torch.zeros(Bq, Hq, Kq, v.shape[-1], device=q.device) if h0 is None else h0.float().clone()
    out = []
    for t in range(Tq):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt = k[:, t], v[:, t]
        pred = torch.einsum("bhkv,bhk->bhv", S, kt)
        u = beta[:, t][..., None] * (vt - pred)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, u)
        out.append(scale * torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1), S


def inputs(dtype, raw_beta=False):
    q = F.normalize(torch.randn(B, T, H, K, device=DEV), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, device=DEV), dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, device=DEV).to(dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=DEV) + 2).float()
    b_raw = torch.randn(B, T, H, device=DEV)
    beta = b_raw.to(dtype) if raw_beta else torch.sigmoid(b_raw).to(dtype)
    return q, k, v, g, beta


def err(a, ref):
    a = a.float()
    if not torch.isfinite(a).all():
        return float("inf")
    return (a - ref).abs().max().item()


def main():
    t_start = time.time()
    seed_all(SEED)
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    print("fla", fla.__version__, "| device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    scale = K ** -0.5
    data = {"fla_version": fla.__version__, "sizes": dict(B=B, T=T, H=H, K=K, V=V, prefill=PREFILL), "rows": []}
    c110 = c111 = True
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            tol = TOL[dtype]
            q, k, v, g, beta = inputs(dtype)
            ref, _ = reference(q, k, v, g, beta, scale)
            row = {"dtype": str(dtype)}
            # 1a. the old positional call (beta -> gk); may raise or read garbage
            try:
                o_pos, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=scale)
                torch.cuda.synchronize()
                row["positional_err"] = err(o_pos, ref)
            except Exception as e:  # an exception also counts as "the positional call is broken"
                row["positional_err"] = float("inf")
                row["positional_exception"] = f"{type(e).__name__}: {e}"[:300]
            # 1b. keyword call
            o_kw, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, scale=scale)
            row["keyword_err"] = err(o_kw, ref)
            o_ch, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)
            row["chunk_err"] = err(o_ch, ref)
            # 2. decode: chunk prefill -> 1-token recurrent steps carrying the state
            _, S = chunk_gated_delta_rule(q[:, :PREFILL], k[:, :PREFILL], v[:, :PREFILL], g[:, :PREFILL],
                                          beta[:, :PREFILL], scale=scale, output_final_state=True)
            outs = []
            for t in range(PREFILL, T):
                o_t, S = fused_recurrent_gated_delta_rule(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                                                          g=g[:, t:t + 1], beta=beta[:, t:t + 1], scale=scale,
                                                          initial_state=S, output_final_state=True)
                outs.append(o_t)
            row["decode_err"] = err(torch.cat(outs, 1), ref[:, PREFILL:])
            # 3. negative eigenvalues: raw logits, kernel applies 2*sigmoid
            qn, kn, vn, gn, braw = inputs(dtype, raw_beta=True)
            ref_n, _ = reference(qn, kn, vn, gn, 2 * torch.sigmoid(braw.float()), scale)
            o_n, _ = fused_recurrent_gated_delta_rule(qn, kn, vn, g=gn, beta=braw, scale=scale,
                                                      use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
            row["negeig_err"] = err(o_n, ref_n)
            o_nc, _ = chunk_gated_delta_rule(qn, kn, vn, gn, braw, scale=scale,
                                             use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
            row["negeig_chunk_err"] = err(o_nc, ref_n)
            row["tol"] = tol
            ok110 = row["positional_err"] > 1 and row["keyword_err"] < tol
            ok111 = row["decode_err"] < tol and row["negeig_err"] < tol
            c110 &= ok110
            c111 &= ok111
            data["rows"].append(row)
            print(f"{str(dtype):15s} positional {row['positional_err']:.2e} | keyword {row['keyword_err']:.2e} | "
                  f"chunk {row['chunk_err']:.2e} | decode {row['decode_err']:.2e} | neg-eig {row['negeig_err']:.2e} "
                  f"(chunk {row['negeig_chunk_err']:.2e}) tol {tol:g}")

        # 4. speed of one decode step (B=16, H=6, K=V=64, bf16, state carried in fp32)
        SB, SH = 16, 6
        qs = F.normalize(torch.randn(SB, 1, SH, K, device=DEV), dim=-1).bfloat16()
        ks = F.normalize(torch.randn(SB, 1, SH, K, device=DEV), dim=-1).bfloat16()
        vs = torch.randn(SB, 1, SH, V, device=DEV).bfloat16()
        gs = F.logsigmoid(torch.randn(SB, 1, SH, device=DEV) + 2)
        bs = torch.sigmoid(torch.randn(SB, 1, SH, device=DEV)).bfloat16()
        S0 = torch.zeros(SB, SH, K, V, device=DEV)
        rec_ms = bench_ms(lambda: fused_recurrent_gated_delta_rule(qs, ks, vs, g=gs, beta=bs, scale=scale,
                                                                   initial_state=S0, output_final_state=True),
                          warmup=10, iters=200)
        chunk_ms = bench_ms(lambda: chunk_gated_delta_rule(qs, ks, vs, gs, bs, scale=scale, initial_state=S0,
                                                           output_final_state=True), warmup=10, iters=200)
        data["decode_step_ms"] = {"fused_recurrent": round(rec_ms, 4), "chunk_T1": round(chunk_ms, 4),
                                  "sizes": dict(B=SB, H=SH, K=K, V=V)}
        print(f"decode step B={SB} H={SH}: fused_recurrent {rec_ms:.3f} ms, chunk T=1 {chunk_ms:.3f} ms")

    verdicts = {
        "C110": "VERIFIED (positional call broken, keyword call correct)" if c110 else "REFUTED or inconclusive",
        "C111": "VERIFIED (prefill+step and neg-eig decode within tolerance)" if c111 else "REFUTED",
        "C2": ("WRONG TEST: re-test with keyword args passes" if (c110 and c111) else "unchanged (see rows)"),
        "speed_prediction": ("hit" if rec_ms <= 0.2 and chunk_ms >= 2 * rec_ms else "miss"),
    }
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, verdicts, data, t_start, filename=FILE_STEM)
    print("PASS" if (c110 and c111) else "FAIL")


if __name__ == "__main__":
    main()
