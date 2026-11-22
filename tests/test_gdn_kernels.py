"""Check flash-linear-attention's Gated DeltaNet kernels against a plain PyTorch reference.

    ~/.venvs/gogeta/bin/python tests/test_gdn_kernels.py

Gated delta rule, per head, state S in R^{K x V}:
    S_t = exp(g_t) * S_{t-1}
    S_t = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T
    o_t = scale * S_t^T q_t
with q, k L2-normalised, g_t <= 0 (log decay), beta_t in (0, 1).
"""
import inspect
import sys
import time

import torch
import torch.nn.functional as F


def reference(q, k, v, g, beta, scale):
    """q,k: [B,T,H,K]  v: [B,T,H,V]  g,beta: [B,T,H]  -> o: [B,T,H,V] (float32 loop)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V)
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt, bt = k[:, t], v[:, t], beta[:, t][..., None]
        v_old = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, bt * (vt - v_old))
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(out, 1)


def main():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    print("chunk sig:", inspect.signature(chunk_gated_delta_rule))
    torch.manual_seed(0)
    dev = "cuda"
    B, T, H, K, V = 2, 300, 4, 64, 64
    scale = K ** -0.5

    def inputs(dtype):
        q = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dtype)
        k = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dtype)
        v = torch.randn(B, T, H, V, device=dev).to(dtype)
        g = F.logsigmoid(torch.randn(B, T, H, device=dev) + 2).to(torch.float32)
        beta = torch.sigmoid(torch.randn(B, T, H, device=dev)).to(dtype)
        return [x.requires_grad_() for x in (q, k, v, g, beta)]

    ok = True
    for dtype, tol in ((torch.float32, 2e-3), (torch.bfloat16, 3e-2)):
        q, k, v, g, beta = inputs(dtype)
        ref = reference(q, k, v, g, beta, scale)
        o_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)
        o_rec, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale=scale)
        e1 = (o_chunk.float() - ref).abs().max().item()
        e2 = (o_rec.float() - ref).abs().max().item()
        # backward: compare grads of a random projection of the output
        w = torch.randn_like(ref)
        gr = torch.autograd.grad((ref * w).sum(), (q, k, v, g, beta))
        gc = torch.autograd.grad((o_chunk.float() * w).sum(), (q, k, v, g, beta))
        ge = max((a.float() - b.float()).abs().max().item() / (a.float().abs().max().item() + 1e-6)
                 for a, b in zip(gr, gc))
        # The recurrent kernel is only needed for token-by-token decoding; report it, don't require it.
        with torch.no_grad():
            o_rec_ng, _ = fused_recurrent_gated_delta_rule(q.detach(), k.detach(), v.detach(), g.detach(),
                                                           beta.detach(), scale=scale)
        e3 = (o_rec_ng.float() - ref).abs().max().item()
        good = e1 < tol and ge < 10 * tol
        ok &= good
        print(f"{str(dtype):15s} CHUNK fwd max|chunk-ref|={e1:.2e} bwd max rel grad err={ge:.2e} "
              f"-> {'OK' if good else 'MISMATCH'} | recurrent (info): grad-mode {e2:.2e}, no-grad {e3:.2e}")

    # speed: chunk kernel at training-like size
    q, k, v, g, beta = inputs(torch.bfloat16)
    Bs, Ts, Hs = 16, 1024, 6
    q, k = (F.normalize(torch.randn(Bs, Ts, Hs, 64, device=dev), dim=-1).bfloat16() for _ in range(2))
    v = torch.randn(Bs, Ts, Hs, 64, device=dev).bfloat16()
    g = F.logsigmoid(torch.randn(Bs, Ts, Hs, device=dev) + 2)
    beta = torch.sigmoid(torch.randn(Bs, Ts, Hs, device=dev)).bfloat16()
    for _ in range(3):
        chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)
    torch.cuda.synchronize()
    print(f"chunk fwd B={Bs} T={Ts} H={Hs}: {(time.time()-t0)/20*1e3:.2f} ms")
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
