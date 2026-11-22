"""SQ50: fla's Kimi Delta Attention (KDA) kernels on this GPU (claims C70, C71, C72; C2 re-checked).

    ~/.venvs/gogeta/bin/python experiments/checks/sq_kda_kernel.py          # GPU, ~1-5 min (autotune)

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: Kimi Linear (arXiv 2510.26692) eq. 1; kernels in fla 0.6.0 `fla.ops.kda` (chunk_kda, fused_recurrent_kda).

KDA, per head, state S in R^{K x V}, channel-wise log decay g_t in R^K (g <= 0), beta_t in (0, 1):
    S_t = Diag(exp(g_t)) S_{t-1}                    (GDN: the same with one scalar g_t per head)
    S_t = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T  (delta rule, unchanged from GDN)
    o_t = scale * S_t^T q_t
with q, k L2-normalised. Kimi's gate: g = -exp(A_log[h]) * softplus(W_up W_down x + dt_bias[h, :]).

What is measured
  1. chunk_kda forward + backward vs a float32 loop reference (fp32 and bf16 inputs), for
     (a) mild decay, (b) strong decay down to log-decay -5, (c) the in-kernel gate path (use_gate_in_kernel=True:
     raw gate input + A_log + dt_bias, gradients to all of them).
  2. Reduction to GDN: chunk_kda with g tied across the K channels vs chunk_gated_delta_rule with that scalar g.
  3. Decode kernels: fused_recurrent_kda (grad mode, no-grad mode, and chunk-prefill -> 1-token steps carrying the
     state, i.e. the real generation path). GDN's fused_recurrent_gated_delta_rule is re-run beside it (claim C2).
     Reported only: a fallback decode path that calls the *chunk* kernels (KDA and GDN) with T=1 per step.
  4. Speed at B=16, T=1024, H=6, K=V=64 (bf16): chunk_kda vs chunk_gated_delta_rule, fwd and fwd+bwd; and a 1-token
     decode step at B=16 for both recurrent kernels.
"""
import inspect
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "chunk_kda matches the fp32 reference in every regime (fp32 fwd err ~1e-4..1e-3, bf16 ~1e-2) with grads in "
    "tolerance, and reduces to chunk_gated_delta_rule when g is tied (diff < 1e-3). fused_recurrent_kda: uncertain "
    "(~50/50); if GDN's recurrent NaN comes from a shared Triton/cc12.1 issue, KDA's will NaN too. Speed: chunk_kda "
    "fwd+bwd 1.2-2.0x chunk_gated_delta_rule at K=64 (the paper reports 'negligible' model-level prefill overhead "
    "vs GDN-H, at K=128 and 48B scale, which we do not adopt)."
)
CRITERION = (
    "C70 (chunk_kda correct on this GPU) is VERIFIED if, in all three regimes, fwd max|err| < 2e-3 (fp32) / 3e-2 "
    "(bf16) and the max per-tensor relative grad error < 2e-2 (fp32) / 3e-1 (bf16) (the tests/test_gdn_kernels.py "
    "tolerances), AND the tied-gate KDA-vs-GDN fp32 diff < 1e-3. C71 (fused_recurrent_kda usable for decoding) is "
    "VERIFIED if all outputs are finite and the grad-mode, no-grad and prefill+step paths are each within the fwd "
    "tolerance; REFUTED if any NaN/inf. C72 ('KDA costs about the same as GDN') is VERIFIED at this size if the "
    "chunk_kda fwd+bwd time <= 1.5x chunk_gated_delta_rule's, PARTIAL if <= 2.5x, REFUTED otherwise. "
    "Runtime must stay <= 300 s."
)
CHECK_ID = "sq_kda_kernel"
CLAIMS = ["C70", "C71", "C72", "C2"]
DEV = "cuda"
TOL = {torch.float32: 2e-3, torch.bfloat16: 3e-2}
GRAD_FACTOR = 10            # grad tolerance = 10 x fwd tolerance (as in tests/test_gdn_kernels.py)
TIED_TOL = 1e-3
SPEED_VERIFIED, SPEED_PARTIAL = 1.5, 2.5
B, T, H, K, V = 2, 300, 4, 64, 64     # correctness size (T not a multiple of the chunk size on purpose)
SB, ST, SH = 16, 1024, 6               # speed size
PREFILL = 256                          # prefill length for the chunk -> step decode path
SEED = 0


# ---------------------------------------------------------------------------- reference
def reference(q, k, v, g, beta, scale, S0=None):
    """q,k: [B,T,H,K]  v: [B,T,H,V]  g: [B,T,H,K] (log decay)  beta: [B,T,H]  -> (o [B,T,H,V], S [B,H,K,V])."""
    Bq, Tq, Hq, Kq = q.shape
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    S = q.new_zeros(Bq, Hq, Kq, v.shape[-1]) if S0 is None else S0.float().clone()
    out = []
    for t in range(Tq):
        S = S * g[:, t].exp()[..., None]                          # channel-wise decay of the key rows
        kt, vt, bt = k[:, t], v[:, t], beta[:, t][..., None]
        v_old = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, bt * (vt - v_old))
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(out, 1), S


def kda_gate(g_raw, A_log, dt_bias):
    """Kimi/fla gate: -exp(A_log[h]) * softplus(g_raw + dt_bias[h, :])  (same formula as fla naive_kda_gate)."""
    Hh, Kk = g_raw.shape[-2:]
    return -A_log.float().exp().view(Hh, 1) * F.softplus(g_raw.float() + dt_bias.float().view(Hh, Kk))


def max_rel(ga, gb):
    return max((a.float() - b.float()).abs().max().item() / (a.float().abs().max().item() + 1e-6)
               for a, b in zip(ga, gb))


def finite(*xs):
    return all(bool(torch.isfinite(x.float()).all()) for x in xs)


def qkvb(dtype, Bq=B, Tq=T, Hq=H):
    q = F.normalize(torch.randn(Bq, Tq, Hq, K, device=DEV), dim=-1).to(dtype)
    k = F.normalize(torch.randn(Bq, Tq, Hq, K, device=DEV), dim=-1).to(dtype)
    v = torch.randn(Bq, Tq, Hq, V, device=DEV).to(dtype)
    beta = torch.sigmoid(torch.randn(Bq, Tq, Hq, device=DEV)).to(dtype)
    return q, k, v, beta


def main():
    sys.stdout.reconfigure(line_buffering=True)  # partial progress survives a timeout kill
    t_start = time.time()
    seed_all(SEED)
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    sig_chunk = str(inspect.signature(chunk_kda))
    sig_rec = str(inspect.signature(fused_recurrent_kda))
    print("fla", fla.__version__, "| device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("chunk_kda", sig_chunk[:160], "...")
    scale = K ** -0.5
    data = {"fla_version": fla.__version__, "sizes": dict(B=B, T=T, H=H, K=K, V=V),
            "speed_sizes": dict(B=SB, T=ST, H=SH, K=K, V=V), "chunk": {}, "decode": {}, "speed": {}}

    # ------------------------------------------------------------------ 1. chunk_kda vs reference
    chunk_ok = True
    for dtype in (torch.float32, torch.bfloat16):
        tol = TOL[dtype]
        for regime in ("mild", "strong", "in_kernel_gate"):
            q, k, v, beta = qkvb(dtype)
            if regime == "in_kernel_gate":
                g_raw = torch.randn(B, T, H, K, device=DEV).to(dtype)
                A_log = torch.log(torch.empty(H, device=DEV).uniform_(1, 16))
                dt = torch.exp(torch.rand(H * K, device=DEV) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
                dt_bias = dt + torch.log(-torch.expm1(-dt))
                leaves = [x.requires_grad_() for x in (q, k, v, g_raw, beta, A_log, dt_bias)]
                ref, _ = reference(q, k, v, kda_gate(g_raw, A_log, dt_bias), beta, scale)
                out, _ = chunk_kda(q, k, v, g_raw, beta, scale=scale, use_gate_in_kernel=True,
                                   A_log=A_log, dt_bias=dt_bias)
            else:
                if regime == "mild":
                    g = F.logsigmoid(torch.randn(B, T, H, K, device=DEV) + 2)
                else:  # strong: per-channel log decay spread over [-5, 0)
                    g = -5 * torch.sigmoid(2 * torch.randn(B, T, H, K, device=DEV))
                leaves = [x.requires_grad_() for x in (q, k, v, g, beta)]
                ref, _ = reference(q, k, v, g, beta, scale)
                out, _ = chunk_kda(q, k, v, g, beta, scale=scale)
            e_fwd = (out.float() - ref).abs().max().item()
            w = torch.randn_like(ref)
            gr = torch.autograd.grad((ref * w).sum(), leaves)
            gc = torch.autograd.grad((out.float() * w).sum(), leaves)
            e_bwd = max_rel(gr, gc)
            ok = finite(out, *gc) and e_fwd < tol and e_bwd < GRAD_FACTOR * tol
            chunk_ok &= ok
            key = f"{str(dtype).split('.')[-1]}/{regime}"
            data["chunk"][key] = {"fwd_max_abs_err": e_fwd, "bwd_max_rel_err": e_bwd, "ok": ok}
            print(f"chunk_kda {key:24s} fwd {e_fwd:.2e}  bwd rel {e_bwd:.2e}  -> {'OK' if ok else 'MISMATCH'}")

    # ------------------------------------------------------------------ 2. tied gate == GDN
    q, k, v, beta = qkvb(torch.float32)
    g_s = F.logsigmoid(torch.randn(B, T, H, device=DEV) + 2)
    with torch.no_grad():
        o_kda, _ = chunk_kda(q, k, v, g_s[..., None].expand(B, T, H, K).contiguous(), beta, scale=scale)
        o_gdn, _ = chunk_gated_delta_rule(q, k, v, g_s, beta, scale=scale)
    tied_diff = (o_kda.float() - o_gdn.float()).abs().max().item()
    tied_ok = tied_diff < TIED_TOL
    data["tied_gate_kda_vs_gdn_max_abs_diff"] = tied_diff
    print(f"tied-gate chunk_kda vs chunk_gated_delta_rule: {tied_diff:.2e} -> {'OK' if tied_ok else 'MISMATCH'}")

    # ------------------------------------------------------------------ 3. decode kernels
    dec_ok = True
    for dtype in (torch.float32, torch.bfloat16):
        tol = TOL[dtype]
        dn = str(dtype).split('.')[-1]
        q, k, v, beta = qkvb(dtype)
        g = F.logsigmoid(torch.randn(B, T, H, K, device=DEV) + 2)
        ref, _ = reference(q, k, v, g, beta, scale)
        # grad mode (as in tests/test_gdn_kernels.py)
        qq, kk, vv, gg, bb = (x.clone().requires_grad_() for x in (q, k, v, g, beta))
        o_g, _ = fused_recurrent_kda(qq, kk, vv, gg, bb, scale=scale)
        with torch.no_grad():
            o_ng, _ = fused_recurrent_kda(q, k, v, g, beta, scale=scale)
            # real generation path: chunk prefill -> carry the fp32 state -> 1-token recurrent steps
            o_pre, S = chunk_kda(q[:, :PREFILL], k[:, :PREFILL], v[:, :PREFILL], g[:, :PREFILL],
                                 beta[:, :PREFILL], scale=scale, output_final_state=True)
            steps = []
            for t in range(PREFILL, T):
                sl = slice(t, t + 1)
                o_t, S = fused_recurrent_kda(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], scale=scale,
                                             initial_state=S.float(), output_final_state=True)
                steps.append(o_t)
            o_step = torch.cat([o_pre] + steps, 1)
            # GDN recurrent kernel beside it (claim C2), scalar gate = channel mean
            g_h = g.mean(-1)
            ref_gdn, _ = reference(q, k, v, g_h[..., None].expand(B, T, H, K), beta, scale)
            o_gdn_rec, _ = fused_recurrent_gated_delta_rule(q, k, v, g_h, beta, scale=scale)
            # fallback decode path (reported, not part of a criterion): chunk kernels called with T=1 per step,
            # carrying the state. If it works for GDN it is a stopgap for C2 on our existing GDN checkpoints.
            fb = {}
            for nm, fn, gg_ in (("kda_chunk_T1_steps", chunk_kda, g), ("gdn_chunk_T1_steps", chunk_gated_delta_rule, g_h)):
                o_pre, S = fn(q[:, :PREFILL], k[:, :PREFILL], v[:, :PREFILL], gg_[:, :PREFILL], beta[:, :PREFILL],
                              scale=scale, output_final_state=True)
                outs = [o_pre]
                for t in range(PREFILL, T):
                    sl = slice(t, t + 1)
                    o_t, S = fn(q[:, sl], k[:, sl], v[:, sl], gg_[:, sl], beta[:, sl], scale=scale,
                                initial_state=S.float(), output_final_state=True)
                    outs.append(o_t)
                fb[nm] = torch.cat(outs, 1)
        r = {}
        for nm, o in fb.items():
            rf = ref if nm.startswith("kda") else ref_gdn
            fin = finite(o)
            r[nm] = {"finite": fin, "max_abs_err": (o.float() - rf).abs().max().item() if fin else float("nan")}
            print(f"fallback {dn:8s} {nm:20s} finite={fin} err {r[nm]['max_abs_err']:.2e}")
        for name, o in (("grad_mode", o_g), ("no_grad", o_ng), ("prefill_then_steps", o_step)):
            fin = finite(o)
            err = (o.float() - ref).abs().max().item() if fin else float("nan")
            r[name] = {"finite": fin, "max_abs_err": err, "ok": fin and err < tol}
            dec_ok &= r[name]["ok"]
            print(f"fused_recurrent_kda {dn:8s} {name:20s} finite={fin} err {err:.2e}")
        fin = finite(o_gdn_rec)
        r["gdn_fused_recurrent_no_grad"] = {"finite": fin, "max_abs_err":
                                            (o_gdn_rec.float() - ref_gdn).abs().max().item() if fin else float("nan")}
        print(f"fused_recurrent_gated_delta_rule {dn:8s} finite={fin} err {r['gdn_fused_recurrent_no_grad']['max_abs_err']:.2e}")
        data["decode"][dn] = r

    # ------------------------------------------------------------------ 4. speed
    q, k, v, beta = qkvb(torch.bfloat16, SB, ST, SH)
    g = F.logsigmoid(torch.randn(SB, ST, SH, K, device=DEV) + 2)
    g_h = g.mean(-1).contiguous()
    leaves_kda = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
    leaves_gdn = [x.clone().requires_grad_() for x in (q, k, v, g_h, beta)]
    do = torch.randn(SB, ST, SH, V, device=DEV, dtype=torch.bfloat16)

    def fwd(fn, xs):
        with torch.no_grad():
            fn(*xs, scale=scale)

    def fwdbwd(fn, xs):
        o, _ = fn(*xs, scale=scale)
        o.backward(do)

    sp = {
        "kda_fwd_ms": bench_ms(lambda: fwd(chunk_kda, leaves_kda), warmup=3, iters=20),
        "gdn_fwd_ms": bench_ms(lambda: fwd(chunk_gated_delta_rule, leaves_gdn), warmup=3, iters=20),
        "kda_fwdbwd_ms": bench_ms(lambda: fwdbwd(chunk_kda, leaves_kda), warmup=3, iters=20),
        "gdn_fwdbwd_ms": bench_ms(lambda: fwdbwd(chunk_gated_delta_rule, leaves_gdn), warmup=3, iters=20),
    }
    # 1-token decode step at B=16 (state carried in fp32)
    S_k = torch.zeros(SB, SH, K, V, device=DEV)
    sl = slice(0, 1)
    with torch.no_grad():
        sp["kda_decode_step_ms"] = bench_ms(lambda: fused_recurrent_kda(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], scale=scale, initial_state=S_k,
            output_final_state=True), warmup=5, iters=50)
        sp["gdn_decode_step_ms"] = bench_ms(lambda: fused_recurrent_gated_delta_rule(
            q[:, sl], k[:, sl], v[:, sl], g_h[:, sl], beta[:, sl], scale=scale, initial_state=S_k,
            output_final_state=True), warmup=5, iters=50)
    ratio = sp["kda_fwdbwd_ms"] / sp["gdn_fwdbwd_ms"]
    sp["kda_over_gdn_fwdbwd"] = ratio
    sp["kda_over_gdn_fwd"] = sp["kda_fwd_ms"] / sp["gdn_fwd_ms"]
    data["speed"] = sp
    for kk_, vv_ in sp.items():
        print(f"  {kk_:24s} {vv_:.3f}")

    # ------------------------------------------------------------------ verdicts
    runtime = time.time() - t_start
    c70 = chunk_ok and tied_ok
    any_nan = any(not r_["finite"] for d_ in data["decode"].values() for n_, r_ in d_.items()
                  if n_ in ("grad_mode", "no_grad", "prefill_then_steps"))
    fb_ok = {nm: all(d_[nm]["finite"] and d_[nm]["max_abs_err"] < TOL[torch.float32 if dk == "float32" else torch.bfloat16]
                     for dk, d_ in data["decode"].items())
             for nm in ("kda_chunk_T1_steps", "gdn_chunk_T1_steps")}
    gdn_nan = any(not d_["gdn_fused_recurrent_no_grad"]["finite"] for d_ in data["decode"].values())
    verdicts = {
        "C70": ("VERIFIED" if c70 else "REFUTED") + f": chunk regimes ok={chunk_ok}, tied-gate diff {tied_diff:.2e}",
        "C71": ("VERIFIED" if dec_ok else ("REFUTED (NaN/inf)" if any_nan else "REFUTED (error above tolerance)")),
        "C72": ("VERIFIED" if ratio <= SPEED_VERIFIED else "PARTIAL" if ratio <= SPEED_PARTIAL else "REFUTED")
               + f": chunk_kda fwd+bwd = {ratio:.2f}x chunk_gated_delta_rule (B={SB},T={ST},H={SH},K=V={K}, bf16)",
        "C2_recheck_gdn_recurrent": "still NaN" if gdn_nan else "finite now (re-test tests/test_gdn_kernels.py)",
        "chunk_T1_decode_fallback_ok (reported)": fb_ok,
        "runtime_within_300s": runtime <= 300,
        "signatures": {"chunk_kda": sig_chunk, "fused_recurrent_kda": sig_rec},
    }
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, verdicts, data, t_start)
    sys.exit(0 if c70 else 1)


if __name__ == "__main__":
    main()
