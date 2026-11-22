"""Independent pure-PyTorch references for every Gogeta block kind (written for fp64, valid in any dtype).

Nothing here imports gogeta.model or fla. The formulations deliberately differ from the implementation:
  * gated delta rule: (a) per-token recurrence and (b) chunkwise WY/UT form with a triangular solve
    (the model uses fla's chunk kernel; tests/test_gdn_kernels.py's reference is only the loop (a));
  * attention: explicit score matrix, boolean window mask, hand-written softmax (no SDPA);
  * RoPE: complex multiplication (the model uses a stack/flatten of real pairs);
  * BitLinear: integer ternary codes x int8 activation codes with a late rescale (the model fake-quantises);
  * RMSNorm / SwiGLU / conv: explicit formulas.
Delta rule (per head, S in R^{K x V}):  S_t = e^{g_t} (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T ,  o_t = scale S_t^T q_t.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ gated delta rule
def delta_rule_recurrent(q, k, v, g, beta, scale, S0=None):
    """q,k:[B,T,H,K] v:[B,T,H,V] g,beta:[B,T,H] -> (o [B,T,H,V], S [B,H,K,V]). Token-by-token loop."""
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]) if S0 is None else S0.to(q.dtype).clone()
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], beta[:, t][..., None] * (v[:, t] - pred))
        out.append(scale * torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1), S


def delta_rule_chunked(q, k, v, g, beta, scale, S0=None, chunk=64):
    """Same function as delta_rule_recurrent, computed chunkwise (UT/WY form):
    w = (I + A)^-1 beta (v - e^G k S0),  A_ij = beta_i e^{G_i-G_j} k_i.k_j (j<i),
    o_i = scale (e^{G_i} q_i S0 + sum_{j<=i} e^{G_i-G_j} (q_i.k_j) w_j),  S' = e^{G_C} S0 + sum_j e^{G_C-G_j} k_j w_j^T."""
    B, T, H, K = q.shape
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # [B,H,T,*]
    g, beta = g.transpose(1, 2), beta.transpose(1, 2)
    S = q.new_zeros(B, H, K, v.shape[-1]) if S0 is None else S0.to(q.dtype).clone()
    outs = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        C = e - s
        qc, kc, vc, gc, bc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], g[:, :, s:e], beta[:, :, s:e]
        G = gc.cumsum(-1)
        D = G[..., :, None] - G[..., None, :]
        strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)
        incl = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        A = bc[..., :, None] * D.masked_fill(~strict, float("-inf")).exp() * (kc @ kc.transpose(-1, -2))
        gam = G.exp()
        rhs = bc[..., None] * (vc - gam[..., None] * (kc @ S))
        eye = torch.eye(C, dtype=q.dtype, device=q.device)
        W = torch.linalg.solve_triangular(eye + A, rhs, upper=False)
        P = D.masked_fill(~incl, float("-inf")).exp() * (qc @ kc.transpose(-1, -2))
        outs.append(scale * (gam[..., None] * (qc @ S) + P @ W))
        tail = (G[..., -1:] - G).exp()
        S = G[..., -1].exp()[..., None, None] * S + kc.transpose(-1, -2) @ (tail[..., None] * W)
    return torch.cat(outs, 2).transpose(1, 2), S


# ------------------------------------------------------------------ attention / RoPE
def rope_table(max_len, head_dim, base, dtype=torch.float64):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    return torch.outer(torch.arange(max_len, dtype=dtype), inv)  # angles [max_len, hd/2]


def rope_apply(x, angles):
    """x:[B,H,T,hd] with interleaved (even, odd) pairs treated as complex numbers."""
    T = x.shape[2]
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    rot = angles[:T] if angles.is_complex() else torch.polar(torch.ones_like(angles[:T]), angles[:T])  # complex = given cos+i sin
    rot = rot.to(z.dtype)
    return torch.view_as_real(z * rot).flatten(-2)


def window_mask(T, window=None, device=None):
    i = torch.arange(T, device=device)
    d = i[:, None] - i[None, :]
    m = d >= 0
    return m if window is None else (m & (d < window))


def attention_ref(q, k, v, window=None):
    """q,k,v:[B,H,T,hd]; explicit causal (and optional sliding-window) softmax attention."""
    T, hd = q.shape[2], q.shape[3]
    s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    s = s.masked_fill(~window_mask(T, window, q.device), float("-inf"))
    s = s - s.amax(-1, keepdim=True)
    p = s.exp()
    return (p / p.sum(-1, keepdim=True)) @ v


# ------------------------------------------------------------------ BitLinear / norm / MLP
def ternary_codes(W):
    """absmean ternary: codes in {-1,0,1} (int-valued tensor) and the scale gamma."""
    gamma = W.abs().mean().clamp(min=1e-5)
    return torch.floor(W / gamma + 0.5).clamp(-1, 1), gamma


def bitlinear_ref(x, W):
    """BitNet b1.58 forward as integer arithmetic: y = (A_int @ T_int^T) * gamma / s, with A_int the int8 absmax codes."""
    T_int, gamma = ternary_codes(W)
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    A_int = torch.floor(x * s + 0.5).clamp(-128, 127)
    return (A_int @ T_int.transpose(-1, -2)) * gamma / s


def linear_ref(x, W, ternary):
    return bitlinear_ref(x, W) if ternary else x @ W.transpose(-1, -2)


RMS_EPS = None  # set to finfo(float32).eps to compare an fp32/bf16 model against an fp64 reference with the SAME eps


def rmsnorm_ref(x, w, eps=None):
    eps = (RMS_EPS if RMS_EPS is not None else torch.finfo(x.dtype).eps) if eps is None else eps  # nn.RMSNorm(eps=None): finfo(fp32).eps for fp32 and bf16, finfo(fp64).eps for fp64
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def swiglu_ref(x, Wup, Wdown, ternary):
    up = linear_ref(x, Wup, ternary)
    a, b = up[..., : up.shape[-1] // 2], up[..., up.shape[-1] // 2:]
    return linear_ref(a * torch.sigmoid(a) * b, Wdown, ternary)


def causal_dwconv_silu(x, w):
    """x:[B,T,C], w:[C,1,k] depthwise causal conv (left zero-pad k-1) then SiLU, written as shifted sums."""
    k = w.shape[-1]
    y = torch.zeros_like(x)
    for j in range(k):
        sh = k - 1 - j
        xs = F.pad(x, (0, 0, sh, 0))[:, : x.shape[1]] if sh else x
        y = y + xs * w[:, 0, j]
    return y * torch.sigmoid(y)


# ------------------------------------------------------------------ whole-model reference (reads a FusionDance's weights)
def _p(sd, name, dtype):
    return sd[name].to(dtype)


def block_ref(sd, pre, kind, cfg, x, ang, dtype, delta_fn):
    ter = cfg.ternary
    D, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
    B, T, _ = x.shape
    h = rmsnorm_ref(x, _p(sd, f"{pre}.norm1.weight", dtype))
    if kind in ("attn", "swa"):
        qkv = linear_ref(h, _p(sd, f"{pre}.mixer.qkv.weight", dtype), ter)
        q, k, v = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = rope_apply(q, ang), rope_apply(k, ang)
        y = attention_ref(q, k, v, cfg.window if kind == "swa" else None)
        mix = linear_ref(y.transpose(1, 2).reshape(B, T, D), _p(sd, f"{pre}.mixer.o.weight", dtype), ter)
    else:
        m = f"{pre}.mixer"
        proj = lambda n: linear_ref(h, _p(sd, f"{m}.{n}.weight", dtype), ter)  # noqa: E731
        cv = lambda n, z: causal_dwconv_silu(z, _p(sd, f"{m}.conv_{n}.conv.weight", dtype))  # noqa: E731
        unit = lambda z: z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # noqa: E731
        q = unit(cv("q", proj("q")).view(B, T, H, hd))
        k = unit(cv("k", proj("k")).view(B, T, H, hd))
        v = cv("v", proj("v")).view(B, T, H, hd)
        a_in = h @ _p(sd, f"{m}.a.weight", dtype).T
        g = -_p(sd, f"{m}.A_log", dtype).exp() * F.softplus(a_in + _p(sd, f"{m}.dt_bias", dtype))
        beta = torch.sigmoid(h @ _p(sd, f"{m}.b.weight", dtype).T) * (2.0 if cfg.gdn_neg_eigval else 1.0)
        o, _ = delta_fn(q, k, v, g, beta, hd ** -0.5)
        o = rmsnorm_ref(o, _p(sd, f"{m}.out_norm.weight", dtype)) * F.silu(proj("gate").view(B, T, H, hd))
        mix = linear_ref(o.reshape(B, T, H * hd), _p(sd, f"{m}.o.weight", dtype), ter)
    x = x + mix
    h = rmsnorm_ref(x, _p(sd, f"{pre}.norm2.weight", dtype))
    return x + swiglu_ref(h, _p(sd, f"{pre}.mlp.up.weight", dtype), _p(sd, f"{pre}.mlp.down.weight", dtype), ter)


def model_ref(model, idx, loops=None, dtype=torch.float64, delta_fn=None, rope_from_model=True, sd=None, bptt=None):
    """Independent forward of a gogeta FusionDance: prelude -> core x r with re-injection -> coda -> head.
    Full BPTT unless `bptt=k`: then h is detached after each of the first r-k loops (the model's no_grad semantics).
    `sd` overrides the weights (name -> tensor), e.g. to gradcheck with respect to them."""
    cfg = model.cfg
    sd = model.state_dict(keep_vars=True) if sd is None else sd
    delta_fn = delta_fn or (lambda *a: delta_rule_chunked(*a, chunk=16))
    if rope_from_model:  # isolate the table's fp32 rounding (tested separately in test_reference_blocks)
        ang = torch.complex(model.rope.cos[0, 0].double(), model.rope.sin[0, 0].double()).to(torch.complex128)
    else:
        ang = rope_table(cfg.max_seq_len, cfg.head_dim, cfg.rope_base, dtype)
    x = _p(sd, "embed.weight", dtype)[idx]
    for i, kd in enumerate(cfg.prelude):
        x = block_ref(sd, f"prelude.{i}", kd, cfg, x, ang, dtype, delta_fn)
    if cfg.core:
        r = loops if loops is not None else cfg.loops
        e = h = x
        for it in range(r):
            hn = rmsnorm_ref(h, _p(sd, "inj_norm_h.weight", dtype))
            en = rmsnorm_ref(e, _p(sd, "inj_norm_e.weight", dtype))
            h = h + linear_ref(torch.cat([hn, en], -1), _p(sd, "inject.weight", dtype), cfg.ternary)
            for i, kd in enumerate(cfg.core):
                h = block_ref(sd, f"core.{i}", kd, cfg, h, ang, dtype, delta_fn)
            if bptt is not None and it < r - bptt:
                h = h.detach()
        x = h
    for i, kd in enumerate(cfg.coda):
        x = block_ref(sd, f"coda.{i}", kd, cfg, x, ang, dtype, delta_fn)
    x = rmsnorm_ref(x, _p(sd, "norm_out.weight", dtype))
    return x @ _p(sd, "head.weight", dtype).T


def make_cpu_kernel(delta_fn=delta_rule_recurrent):
    """Drop-in for fla's chunk_gated_delta_rule on CPU, mirroring the keyword contract the model uses.
    Honours use_beta_sigmoid_in_kernel / allow_neg_eigval like fla (beta -> 2*sigmoid or sigmoid(raw)). Returns (o, S)."""
    def kernel(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, **kw):
        assert not kw, f"unmodelled kwargs {sorted(kw)}"
        assert not use_qk_l2norm_in_kernel
        if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta.double()) * (2.0 if allow_neg_eigval else 1.0)
        dt = q.dtype
        o, S = delta_fn(q.double(), k.double(), v.double(), g.double(), beta.double(),
                        scale if scale is not None else q.shape[-1] ** -0.5, initial_state)
        return o.to(dt), S
    return kernel
