"""fp64 reference implementations, known-answer tests, finite-difference gradient checks, model-vs-reference
equivalence, BPTT semantics (A6, C193-C197).  CPU only, no fla kernel (GatedDeltaNet's kernel is replaced by an
independent fp64 reference through the keyword contract the model uses; the real kernel is checked on GPU by
gpu_numerical_suite.py).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/staged-tests/test_reference_blocks.py [filter]

Sections: reference self-consistency | known answers | STE gradient | gradcheck | model vs reference | loop & BPTT.
"""
from __future__ import annotations

from _harness import ROOT, maxerr, run_module  # noqa: F401

import math

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

import numref as R
from gogeta.model import Attention, BitLinear, FusionDance, GatedDeltaNet, ModelConfig, Rotary, SwiGLU

D64 = torch.float64
TINY = dict(vocab_size=48, d_model=16, n_heads=2, mlp_hidden=16, window=5, max_seq_len=64, conv_size=4)


def seq(B=1, T=20, H=2, K=4, V=3, seed=0, beta_max=1.0, g_scale=0.3, unit_k=True):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=D64)  # noqa: E731
    q, k, v = r(B, T, H, K), r(B, T, H, K), r(B, T, H, V)
    if unit_k:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    g = -g_scale * torch.rand(B, T, H, generator=gen, dtype=D64)
    beta = beta_max * torch.sigmoid(r(B, T, H))
    return q, k, v, g, beta


def tiny_model(seed=0, std=0.4, patch_kernel=True, **kw):
    """fp64 CPU FusionDance with O(1) weights (default init gives ~0 logits and would hide errors)."""
    torch.manual_seed(seed)
    m = FusionDance(ModelConfig(**{**TINY, **kw})).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, std)
    if patch_kernel:
        for mod in m.modules():
            if isinstance(mod, GatedDeltaNet):
                mod._kernel = R.make_cpu_kernel()
    return m


# ============================================================ 1. reference self-consistency
def test_chunked_equals_recurrent_reference():
    """Two independent formulations of the gated delta rule agree to fp64 rounding, incl. beta in (0,2) and ragged T."""
    worst = 0.0
    for seed, (T, beta_max, chunk) in enumerate([(37, 1.0, 8), (64, 2.0, 16), (129, 2.0, 64), (50, 2.0, 1), (33, 1.0, 100)]):
        q, k, v, g, beta = seq(T=T, seed=seed, beta_max=beta_max, g_scale=1.0)
        o1, S1 = R.delta_rule_recurrent(q, k, v, g, beta, 0.5)
        o2, S2 = R.delta_rule_chunked(q, k, v, g, beta, 0.5, chunk=chunk)
        worst = max(worst, maxerr(o1, o2), maxerr(S1, S2))
    assert worst < 1e-11, f"chunked vs recurrent reference differ by {worst:.2e}"
    return f"max diff {worst:.1e}"


def test_reference_attention_vs_sdpa_and_rope():
    torch.manual_seed(0)
    B, H, T, hd = 2, 3, 17, 8
    q, k, v = (torch.randn(B, H, T, hd, dtype=D64) for _ in range(3))
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert maxerr(R.attention_ref(q, k, v), full) < 1e-12
    for w in (1, 4, 17, 40):
        sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=R.window_mask(T, w))
        assert maxerr(R.attention_ref(q, k, v, w), sdpa) < 1e-12, w
    rope = Rotary(hd, 64, 10000.0)
    ang = R.rope_table(64, hd, 10000.0)
    x = torch.randn(B, H, T, hd, dtype=D64)
    rope = rope.double()
    assert maxerr(rope(x), R.rope_apply(x, ang)) < 1e-4, "RoPE (model table, fp32-derived) differs from the fp64 complex form"


def test_rope_table_fp32_error_and_relative_property():
    """The model builds cos/sin tables from fp32 angles. Quantify the table error vs fp64 as a function of position,
    and check the defining RoPE property <R(m)q, R(n)k> depends on m-n only (in the fp64 form)."""
    hd, base = 16, 10000.0
    r = Rotary(hd, 16384, base)
    ang64 = R.rope_table(16384, hd, base)
    err = (r.cos[0, 0].double() - ang64.cos()).abs().max().item(), (r.sin[0, 0].double() - ang64.sin()).abs().max().item()
    e1k = (r.cos[0, 0, :1024].double() - ang64[:1024].cos()).abs().max().item()
    assert e1k < 1e-4, f"RoPE table error at 1k context {e1k:.1e}"
    assert max(err) < 2e-3, f"RoPE table error at 16k context {max(err):.1e}"
    q, k = torch.randn(1, 1, 1, hd, dtype=D64), torch.randn(1, 1, 1, hd, dtype=D64)
    dots = []
    for m, n in [(10, 7), (110, 107), (5000, 4997)]:
        qm = R.rope_apply(q.expand(1, 1, m + 1, hd), ang64)[0, 0, m]
        kn = R.rope_apply(k.expand(1, 1, n + 1, hd), ang64)[0, 0, n]
        dots.append((qm * kn).sum().item())
    assert max(dots) - min(dots) < 1e-9, f"RoPE not relative: {dots}"
    return f"table max error: {e1k:.1e} at <=1k positions, {max(err):.1e} at <=16k (fp32 angle rounding)"


# ============================================================ 2. known-answer tests
def test_known_delta_rule_orthogonal_keys_exact_retrieval():
    """beta=1, g=0, orthonormal keys: the state is sum_j e_j v_j^T, so q=e_s returns scale*v_s exactly (any s<=t)."""
    T = K = 10
    torch.manual_seed(1)
    k = torch.eye(K, dtype=D64)[None, :, None, :]                      # [1,T,1,K]
    v = torch.randn(1, T, 1, 5, dtype=D64)
    src = torch.tensor([max(0, t - 3) for t in range(T)])
    q = torch.eye(K, dtype=D64)[src][None, :, None, :]
    g, beta = torch.zeros(1, T, 1, dtype=D64), torch.ones(1, T, 1, dtype=D64)
    for fn in (R.delta_rule_recurrent, lambda *a: R.delta_rule_chunked(*a, chunk=4)):
        o, S = fn(q, k, v, g, beta, 2.0)
        assert maxerr(o[0, :, 0], 2.0 * v[0, src, 0]) < 1e-14
        assert maxerr(S[0, 0], k[0, :, 0].T @ v[0, :, 0]) < 1e-14


def test_known_delta_rule_overwrite_identity():
    """beta=1, g=0, ||k||=1: after each write S^T k_t == v_t EXACTLY, whatever was stored before (the 'delta' in delta rule).
    More generally with beta, g: S_t^T k_t = (1-beta) e^g S_{t-1}^T k_t + beta v_t."""
    q, k, v, g, beta = seq(T=60, K=6, V=4, seed=3)
    z, one = torch.zeros_like(g), torch.ones_like(beta)
    for fn in (R.delta_rule_recurrent, lambda *a: R.delta_rule_chunked(*a, chunk=7)):
        o, _ = fn(k, k, v, z, one, 1.0)            # q = k  ->  o_t = S_t^T k_t
        assert maxerr(o, v) < 1e-12, "beta=1, g=0: S_t^T k_t must equal v_t"
    # general identity, against the recurrent state (checks the chunked output path with decay and beta<1)
    o, _ = R.delta_rule_chunked(k, k, v, g, beta, 1.0, chunk=9)
    S = torch.zeros(1, 2, 6, 4, dtype=D64)
    for t in range(60):
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t]) * g[:, t].exp()[..., None]
        want = (1 - beta[:, t])[..., None] * pred + beta[:, t][..., None] * v[:, t]
        assert maxerr(o[:, t], want) < 1e-12, t
        S = R.delta_rule_recurrent(k[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], beta[:, t:t + 1], 1.0, S0=S)[1]


def test_known_delta_rule_decay_limits():
    q, k, v, g, beta = seq(T=40, K=5, V=3, seed=4)
    S0 = torch.randn(1, 2, 5, 3, dtype=D64)
    # beta = 0: no write; S_t = exp(sum g) S_0, o_t = scale * exp(G_t) S_0^T q_t
    b0 = torch.zeros_like(beta)
    G = g.cumsum(1)
    want = 0.7 * G.exp()[..., None] * torch.einsum("bhkv,bthk->bthv", S0, q)
    for fn in (R.delta_rule_recurrent, lambda *a, **kw: R.delta_rule_chunked(*a, chunk=6, **kw)):
        o, S = fn(q, k, v, g, b0, 0.7, S0=S0)
        assert maxerr(o, want) < 1e-12 and maxerr(S, G[:, -1].exp()[..., None, None] * S0) < 1e-12
    # g -> -inf (here -60): total forgetting, S_t = beta k v^T  =>  o_t = scale beta (q.k) v
    gm = torch.full_like(g, -60.0)
    o, _ = R.delta_rule_chunked(q, k, v, gm, beta, 0.7, S0=S0, chunk=8)
    want = 0.7 * beta[..., None] * (q * k).sum(-1, keepdim=True) * v
    assert maxerr(o, want) < 1e-12


def test_known_delta_rule_zero_inputs_and_zero_state():
    q, k, v, g, beta = seq(T=30, seed=5)
    z = torch.zeros_like
    for fn in (R.delta_rule_recurrent, lambda *a: R.delta_rule_chunked(*a, chunk=8)):
        o, S = fn(z(q), z(k), z(v), g, beta, 1.0)
        assert o.abs().max() == 0 and S.abs().max() == 0, "all-zero input must give zero output and zero state"
        o, S = fn(q, k, z(v), g, beta, 1.0)  # zero values, zero initial state: nothing is ever written
        assert o.abs().max() == 0 and S.abs().max() == 0, "v=0, S0=0 must keep the state at exactly 0"


def test_known_negative_eigenvalue_householder():
    """beta=2, g=0, ||k||=1: I - 2kk^T is a Householder reflection. With v=0 it preserves the state's Frobenius norm and flips
    the component of S along k (eigenvalue -1); for beta in (0,2) the map is a contraction (never grows the state)."""
    q, k, v, g, beta = seq(T=50, K=6, V=4, seed=6)
    z = torch.zeros_like(v)
    S0 = torch.randn(1, 2, 6, 4, dtype=D64)
    _, S = R.delta_rule_recurrent(q, k, z, torch.zeros_like(g), torch.full_like(g, 2.0), 1.0, S0=S0)
    assert abs(S.norm().item() - S0.norm().item()) < 1e-11, "beta=2 must be norm-preserving (reflection)"
    _, S1 = R.delta_rule_recurrent(q[:, :1], k[:, :1], z[:, :1], g[:, :1] * 0, torch.full_like(g[:, :1], 2.0), 1.0, S0=S0)
    kk = k[:, 0]                                                                  # [1,H,K]
    assert maxerr(torch.einsum("bhk,bhkv->bhv", kk, S1), -torch.einsum("bhk,bhkv->bhv", kk, S0)) < 1e-12, "component along k not negated"
    for bmax in (0.5, 1.0, 1.9):
        _, S = R.delta_rule_chunked(q, k, z, g, torch.full_like(g, bmax), 1.0, S0=S0, chunk=16)
        assert S.norm() <= S0.norm() * (1 + 1e-12), f"beta={bmax} grew the state"


def test_known_swa_window_equals_full_when_window_covers_T():
    torch.manual_seed(2)
    B, H, T, hd = 1, 2, 12, 4
    q, k, v = (torch.randn(B, H, T, hd, dtype=D64) for _ in range(3))
    full = R.attention_ref(q, k, v)
    assert maxerr(R.attention_ref(q, k, v, window=T), full) == 0.0
    assert maxerr(R.attention_ref(q, k, v, window=10 * T), full) == 0.0
    # window = T-1 differs from full ONLY in the last row (the single query that sees T keys)
    d = (R.attention_ref(q, k, v, window=T - 1) - full).abs().amax(dim=(0, 1, 3))
    assert d[: T - 1].max() == 0 and d[T - 1] > 1e-6
    # window = 1: each token attends only to itself -> output = v
    assert maxerr(R.attention_ref(q, k, v, window=1), v) < 1e-14
    # zero queries/keys -> uniform attention over the allowed keys: running mean of v
    z = torch.zeros_like(q)
    mean = v.cumsum(2) / torch.arange(1, T + 1, dtype=D64)[None, None, :, None]
    assert maxerr(R.attention_ref(z, z, v), mean) < 1e-14


def test_known_model_swa_mask_and_influence_support():
    """The model's SWA mask equals the window definition, window>=T collapses to tril, and the exact influence structure
    d out[i] / d x[j] is nonzero iff 0 <= i-j < window (checked with the autograd Jacobian, not just a few perturbations)."""
    cfg = ModelConfig(**{**TINY, "window": 3})
    rope = Rotary(cfg.head_dim, 64, cfg.rope_base).double()
    a = Attention(cfg, 3, rope).double()
    T = 9
    assert torch.equal(a._mask(T, "cpu"), R.window_mask(T, 3))
    a_full = Attention(cfg, T, rope).double()
    assert torch.equal(a_full._mask(T, "cpu"), R.window_mask(T, None)), "window == T must be exactly the causal mask"
    torch.manual_seed(0)
    x = torch.randn(1, T, cfg.d_model, dtype=D64)
    J = torch.autograd.functional.jacobian(lambda z: a(z), x)            # [1,T,D, 1,T,D]
    infl = J[0, :, :, 0, :, :].abs().amax(dim=(1, 3)) > 1e-12            # [i, j]
    assert torch.equal(infl, R.window_mask(T, 3)), "influence support differs from the window mask"


def test_known_ternary_fixed_point_and_scale_shrink():
    """Weights {-1,0,1}*s: the ternary CODES are a fixed point of absmean quantisation (idempotent pattern). The value is only
    fixed if no weight is zero: gamma = s*(fraction nonzero) so zeros shrink the magnitude to s*p (documented behaviour)."""
    torch.manual_seed(0)
    s = 0.37
    full = (torch.randint(0, 2, (8, 16)) * 2 - 1).to(D64) * s              # no zeros: p = 1
    c, gamma = R.ternary_codes(full)
    assert abs(gamma.item() - s) < 1e-15 and maxerr(c * gamma, full) < 1e-15, "all +-s weights must be an exact fixed point"
    sparse = torch.randint(-1, 2, (8, 16)).to(D64) * s
    p = (sparse != 0).double().mean().item()
    c, gamma = R.ternary_codes(sparse)
    assert torch.equal(c, sparse / s), "ternary pattern must be preserved"
    assert abs(gamma.item() - s * p) < 1e-15, "absmean of {-1,0,1}*s is s*p"
    # and the model's BitLinear agrees with the integer-arithmetic reference on these weights
    bl = BitLinear(16, 8, bias=False).double()
    with torch.no_grad():
        bl.weight.copy_(sparse)
    x = torch.randn(3, 16, dtype=D64)
    assert maxerr(bl(x), R.bitlinear_ref(x, sparse)) < 1e-12
    # idempotence of the pattern under repeated quantisation of the *quantised* weights
    q1 = c * gamma
    c2, _ = R.ternary_codes(q1)
    assert torch.equal(c2, c)


def test_known_zero_input_blocks_output_zero():
    cfg = ModelConfig(**TINY)
    rope = Rotary(cfg.head_dim, 64, cfg.rope_base).double()
    z = torch.zeros(1, 7, cfg.d_model, dtype=D64)
    for mod in (Attention(cfg, None, rope), Attention(cfg, 3, rope), SwiGLU(cfg)):
        assert mod.double()(z).abs().max() == 0, type(mod)
    cfg_t = ModelConfig(**{**TINY, "ternary": True})
    assert SwiGLU(cfg_t).double()(z).abs().max() == 0, "ternary SwiGLU on zeros"
    g = GatedDeltaNet(cfg).double()
    g._kernel = R.make_cpu_kernel()
    assert g(z).abs().max() == 0, "GDN mixer on zero input must be exactly zero (conv/normalize/kernel/out_norm all zero-safe)"


# ============================================================ 3. STE gradient (analytic)
def test_bitlinear_ste_gradient_is_exactly_the_identity_ste():
    """Analytic STE gradient. With w_q = round-clip(w/gamma)*gamma and x_q = round-clip(x*s)/s, forward y = x_q w_q^T.
    The code uses  w + (w_q - w).detach()  and  x + (x_q - x).detach(), i.e. dw_q/dw = I and dx_q/dx = I
    (no clipping mask, no gradient through gamma or s). Therefore, for loss L = <G, y>:
         dL/dx = G w_q          dL/dw = G^T x_q          dL/dbias = n/a (bias=False)
    This is the BitNet b1.58 STE, evaluated at the QUANTISED operands."""
    torch.manual_seed(0)
    for (n_in, n_out, shape) in [(16, 8, (3, 5)), (12, 12, (4,))]:
        bl = BitLinear(n_in, n_out, bias=False).double()
        x = torch.randn(*shape, n_in, dtype=D64, requires_grad=True)
        Gm = torch.randn(*shape, n_out, dtype=D64)
        (bl(x) * Gm).sum().backward()
        W = bl.weight.detach()
        T_int, gamma = R.ternary_codes(W)
        wq = T_int * gamma
        s = 127.0 / x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-5)
        xq = torch.floor(x.detach() * s + 0.5).clamp(-128, 127) / s
        assert maxerr(x.grad, Gm @ wq) < 1e-12, "dL/dx is not G w_q"
        assert maxerr(bl.weight.grad, Gm.reshape(-1, n_out).T @ xq.reshape(-1, n_in)) < 1e-12, "dL/dW is not G^T x_q"


def test_bitlinear_ste_surrogate_passes_gradcheck():
    """The STE gradient is the TRUE gradient of the surrogate f(x,w) = (x + c_x) (w + c_w)^T with the quantisation offsets
    c = Q(.) - (.) frozen at the base point: finite differences on that smooth function must equal the model's autograd.
    (gradcheck on BitLinear itself is meaningless: it is piecewise constant, true derivative 0 a.e.)"""
    torch.manual_seed(1)
    bl = BitLinear(10, 6, bias=False).double()
    x = torch.randn(4, 10, dtype=D64)
    with torch.no_grad():
        W = bl.weight
        gamma = W.abs().mean().clamp(min=1e-5)
        cw = (W / gamma).round().clamp(-1, 1) * gamma - W
        s = 127.0 / x.abs().amax(-1, keepdim=True)
        cx = (x * s).round().clamp(-128, 127) / s - x
    Gm = torch.randn(4, 6, dtype=D64)
    xs, Ws = x.clone().requires_grad_(), W.detach().clone().requires_grad_()
    assert gradcheck(lambda a, b: F.linear(a + cx, b + cw), (xs, Ws), eps=1e-6, atol=1e-8, rtol=1e-6)
    xa = x.clone().requires_grad_()
    (bl(xa) * Gm).sum().backward()
    ga, gb = torch.autograd.grad((F.linear(xs + cx, Ws + cw) * Gm).sum(), (xs, Ws))
    assert maxerr(xa.grad, ga) < 1e-12 and maxerr(bl.weight.grad, gb) < 1e-12


# ============================================================ 4. finite-difference gradient checks (fp64, tiny shapes)
def _fc(mod, x, atol=1e-7, rtol=1e-5, eps=1e-6):
    names = [n for n, _ in mod.named_parameters()]
    ps = [p.detach().clone().requires_grad_() for _, p in mod.named_parameters()]

    def f(x_, *ps_):
        return torch.func.functional_call(mod, dict(zip(names, ps_)), (x_,))
    return gradcheck(f, (x.clone().requires_grad_(), *ps), eps=eps, atol=atol, rtol=rtol)


def test_gradcheck_delta_rule_references():
    for name, fn in (("recurrent", R.delta_rule_recurrent), ("chunked", lambda *a: R.delta_rule_chunked(*a, chunk=3))):
        q, k, v, g, beta = seq(B=1, T=7, H=1, K=3, V=2, seed=7, beta_max=2.0, g_scale=1.0, unit_k=False)
        S0 = torch.randn(1, 1, 3, 2, dtype=D64)
        xs = [t.clone().requires_grad_() for t in (q, k, v, g, beta, S0)]
        assert gradcheck(lambda q, k, v, g, b, S0: fn(q, k, v, g, b, 0.6, S0), xs, eps=1e-6, atol=1e-7, rtol=1e-5), name


def test_gradcheck_model_attention_swa_mlp():
    """The model's own non-fla blocks, d=8, T=6: gradient w.r.t. input AND every parameter."""
    cfg = ModelConfig(**{**TINY, "d_model": 8, "n_heads": 2, "mlp_hidden": 6})
    rope = Rotary(cfg.head_dim, 64, cfg.rope_base).double()
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8, dtype=D64)
    for name, mod in (("attn", Attention(cfg, None, rope)), ("swa(w=3)", Attention(cfg, 3, rope)), ("swiglu", SwiGLU(cfg))):
        mod = mod.double()
        with torch.no_grad():
            for p in mod.parameters():
                p.normal_(0, 0.5)
        assert _fc(mod, x), name


def test_gradcheck_whole_loop_model_and_reference():
    """Loop injection / looped core, end to end: (a) the model's forward (attn+swa core, 2 loops) and (b) the independent
    reference with a GDN core, differentiated w.r.t. EVERY parameter (fast_mode: random-projection Jacobian check)."""
    idx = torch.randint(0, 20, (1, 6))
    m = tiny_model(vocab_size=20, d_model=8, n_heads=2, mlp_hidden=6, window=3, prelude=["swa"], core=["attn", "swa"],
                   coda=["attn"], loops=2, bptt_loops=4)
    names = [n for n, _ in m.named_parameters()]
    ps = [p.detach().clone().requires_grad_() for _, p in m.named_parameters()]
    f = lambda *a: torch.func.functional_call(m, dict(zip(names, a)), (idx,))[0]  # noqa: E731
    assert gradcheck(f, tuple(ps), eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True), "model forward"
    m2 = tiny_model(vocab_size=20, d_model=8, n_heads=2, mlp_hidden=6, window=3, prelude=["swa"], core=["gdn", "swa"],
                    coda=["gdn"], loops=2, patch_kernel=False)
    sd = m2.state_dict(keep_vars=True)
    keys = [k for k in sd if sd[k].requires_grad]
    ps2 = [sd[k].detach().clone().requires_grad_() for k in keys]
    f2 = lambda *a: R.model_ref(m2, idx, sd={**sd, **dict(zip(keys, a))})  # noqa: E731
    assert gradcheck(f2, tuple(ps2), eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True), "reference forward (GDN core)"


# ============================================================ 5. model vs independent reference
def _compare_model(m, idx, tol, loops=None, label=""):
    with torch.no_grad():
        a = m(idx, loops=loops)[0]
        b = R.model_ref(m, idx, loops=loops)
    rel = maxerr(a, b) / max(b.abs().max().item(), 1e-12)
    assert rel < tol, f"{label}: model vs fp64 reference relative error {rel:.2e} >= {tol:.0e}"
    return rel


def test_model_matches_reference_every_block_kind():
    idx = torch.randint(0, TINY["vocab_size"], (2, 24))
    out = {}
    for name, kw, tol in [
        ("attn", dict(prelude=["attn"], core=[], coda=[]), 1e-10),
        ("swa", dict(prelude=["swa"], core=[], coda=[]), 1e-10),
        ("ternary attn+swa", dict(prelude=["attn"], core=[], coda=["swa"], ternary=True), 1e-9),
        # GDN glue computes g and beta in fp32 even for fp64 inputs (model.py: .float()): ~1e-7 relative noise
        ("gdn", dict(prelude=["gdn"], core=[], coda=[]), 1e-5),
        ("gdn neg-eig", dict(prelude=["gdn"], core=[], coda=[], gdn_neg_eigval=True), 1e-5),
        ("ternary gdn", dict(prelude=["gdn"], core=[], coda=[], ternary=True), 1e-5),
        ("looped swa/gdn/attn x4", dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=4), 1e-5),
        ("looped ternary neg-eig x3", dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=3, ternary=True,
                                           gdn_neg_eigval=True), 1e-5),
    ]:
        out[name] = _compare_model(tiny_model(**kw), idx, tol, label=name)
    return "rel errs " + ", ".join(f"{k}:{v:.0e}" for k, v in out.items())


def test_model_matches_reference_with_default_init_scale():
    """Same comparison at the model's real init (std 0.02): outputs are tiny, so check absolute error."""
    torch.manual_seed(0)
    m = FusionDance(ModelConfig(**{**TINY, "core": ["gdn", "swa"], "prelude": ["swa"], "coda": ["attn"], "loops": 3})).double().eval()
    for mod in m.modules():
        if isinstance(mod, GatedDeltaNet):
            mod._kernel = R.make_cpu_kernel()
    idx = torch.randint(0, TINY["vocab_size"], (1, 20))
    with torch.no_grad():
        err = maxerr(m(idx)[0], R.model_ref(m, idx))
    assert err < 1e-8, f"absolute error at default init {err:.1e}"


def test_model_gradients_match_reference_gradients():
    """All parameter gradients of the model (autograd through its code) equal those of the independent reference, which is
    itself gradchecked above. Includes the loop, ternary STE (reference uses floor(x+.5) integer codes, gradient surrogate not
    defined there, so the ternary comparison is done on the forward only)."""
    idx = torch.randint(0, TINY["vocab_size"], (1, 18))
    W = torch.randn(1, 18, TINY["vocab_size"], dtype=D64)
    m = tiny_model(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=3, bptt_loops=8)
    (m(idx)[0] * W).sum().backward()
    g_model = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    (R.model_ref(m, idx) * W).sum().backward()
    worst = 0.0
    for n, p in m.named_parameters():
        gm, gr = g_model[n], p.grad
        scale = max(gr.abs().max().item(), 1e-12)
        worst = max(worst, maxerr(gm, gr) / scale)
    assert worst < 1e-4, f"gradient mismatch (relative to each tensor's max) {worst:.2e}"
    return f"worst relative grad err {worst:.1e} over {len(g_model)} tensors"


# ============================================================ 6. looping and BPTT truncation
def test_bptt_truncation_semantics_match_reference():
    """Training mode with r > bptt_loops: the first r-k iterations run under no_grad, so parameter gradients come ONLY from the
    last k iterations and from e (prelude output) via the injection; nothing flows through h of earlier iterations."""
    idx = torch.randint(0, TINY["vocab_size"], (1, 16))
    W = torch.randn(1, 16, TINY["vocab_size"], dtype=D64)
    kw = dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=5, bptt_loops=2)
    m = tiny_model(**kw)

    def grads(fwd):
        m.zero_grad()
        (fwd() * W).sum().backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    def rel(a, b):
        return max(maxerr(a[n], b[n]) / max(b[n].abs().max().item(), 1e-12) for n in b)

    m.train()
    g_train = grads(lambda: m(idx, loops=5)[0])
    g_ref_trunc = grads(lambda: R.model_ref(m, idx, loops=5, bptt=2))
    g_ref_full = grads(lambda: R.model_ref(m, idx, loops=5))
    assert rel(g_train, g_ref_trunc) < 1e-4, f"train-mode gradients differ from the detach-reference: {rel(g_train, g_ref_trunc):.1e}"
    assert rel(g_train, g_ref_full) > 1e-3, "truncation had no effect: bptt_loops is not doing anything"
    # same forward value: truncation only changes the backward
    with torch.no_grad():
        assert maxerr(m(idx, loops=5)[0], R.model_ref(m, idx, loops=5)) < 1e-4 * R.model_ref(m, idx, loops=5).abs().max().item()
    # eval mode (even with grad enabled): NO truncation -> full BPTT
    m.eval()
    g_eval = grads(lambda: m(idx, loops=5)[0])
    assert rel(g_eval, g_ref_full) < 1e-4, "eval mode must backprop through all loops"
    # r <= bptt_loops: train mode equals full BPTT
    m.train()
    g_short = grads(lambda: m(idx, loops=2)[0])
    g_short_ref = grads(lambda: R.model_ref(m, idx, loops=2))
    assert rel(g_short, g_short_ref) < 1e-4
    return "train(r=5,k=2) == detach reference; eval == full BPTT; r<=k == full BPTT"


def test_loop_injection_matches_hand_written_recurrence():
    """h_{i+1} = core(h_i + inject([norm_h(h_i), norm_e(e)])), e = prelude(embed): verified by a hand-rolled loop built from the
    model's own sub-modules and from the independent reference, for several r, including r=0 and r=1."""
    m = tiny_model(prelude=["swa"], core=["swa", "attn"], coda=["swa"], loops=3, bptt_loops=7)
    idx = torch.randint(0, TINY["vocab_size"], (2, 14))
    with torch.no_grad():
        for r in (0, 1, 2, 5):
            x = m.embed(idx)
            for b in m.prelude:
                x = b(x)
            e = h = x
            for _ in range(r):
                h = h + m.inject(torch.cat([m.inj_norm_h(h), m.inj_norm_e(e)], -1))
                for b in m.core:
                    h = b(h)
            for b in m.coda:
                h = b(h)
            hand = m.head(m.norm_out(h))
            assert maxerr(m(idx, loops=r)[0], hand) < 1e-12, f"r={r}"
            assert maxerr(R.model_ref(m, idx, loops=r), hand) < 1e-9, f"reference r={r}"
    # more loops must change the output (the injection is live), loops=0 skips the core entirely
    with torch.no_grad():
        assert maxerr(m(idx, loops=1)[0], m(idx, loops=2)[0]) > 1e-6


# ============================================================ 7. chunked vs streaming vs recurrent at long T (fp64, small d)
def test_long_T_chunked_streaming_recurrent_equivalence():
    """T = 8192, B=1, H=2, K=V=8: full chunked == token loop; arbitrary split points with the carried state == full;
    chunk size does not matter; chunk-prefill then token-by-token decode == full. State stays finite and bounded."""
    T = 8192
    q, k, v, g, beta = seq(B=1, T=T, H=2, K=8, V=8, seed=11, beta_max=2.0, g_scale=0.05)
    o_rec, S_rec = R.delta_rule_recurrent(q, k, v, g, beta, 8 ** -0.5)
    o_c, S_c = R.delta_rule_chunked(q, k, v, g, beta, 8 ** -0.5, chunk=64)
    scale = o_rec.abs().max().item()
    e_full = maxerr(o_rec, o_c) / scale
    assert e_full < 1e-10, f"chunked vs recurrent at T={T}: {e_full:.1e}"
    cuts = [0, 1, 7, 64, 65, 1000, 4097, 6000, 8191, T]
    S, outs = None, []
    for a, b in zip(cuts[:-1], cuts[1:]):
        o, S = R.delta_rule_chunked(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], 8 ** -0.5, S0=S, chunk=64)
        outs.append(o)
    e_stream = maxerr(torch.cat(outs, 1), o_rec) / scale
    assert e_stream < 1e-10 and maxerr(S, S_rec) < 1e-9, f"streaming with carried state: {e_stream:.1e}"
    for ch in (16, 128, 1000):
        assert maxerr(R.delta_rule_chunked(q, k, v, g, beta, 8 ** -0.5, chunk=ch)[0], o_rec) / scale < 1e-10, ch
    # prefill + decode
    P = T - 200
    _, S = R.delta_rule_chunked(q[:, :P], k[:, :P], v[:, :P], g[:, :P], beta[:, :P], 8 ** -0.5, chunk=64)
    dec = []
    for t in range(P, T):
        o, S = R.delta_rule_recurrent(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], beta[:, t:t + 1], 8 ** -0.5, S0=S)
        dec.append(o)
    assert maxerr(torch.cat(dec, 1), o_rec[:, P:]) / scale < 1e-10
    assert torch.isfinite(S_rec).all() and S_rec.norm().item() < 1e3, "state not bounded"
    return f"T={T}: chunked/recurrent {e_full:.1e}, streamed {e_stream:.1e}"


def test_long_T_16k_state_bounded_with_negative_eigenvalues():
    """beta up to 2 for 16k tokens with mild decay: finite, bounded state (|I - beta k k^T| <= 1 for unit k, beta <= 2)."""
    T = 16384
    q, k, v, g, beta = seq(B=1, T=T, H=1, K=8, V=8, seed=12, beta_max=2.0, g_scale=0.02)
    o, S = R.delta_rule_chunked(q, k, v, g, beta, 1.0, chunk=128)
    assert torch.isfinite(o).all() and torch.isfinite(S).all()
    assert S.norm().item() < 50.0, f"state norm {S.norm().item():.1f}"
    # with no decay at all the bound is still linear in T: ||S_T|| <= sum beta ||v||
    _, S0 = R.delta_rule_chunked(q, k, v, torch.zeros_like(g), beta, 1.0, chunk=128)
    assert torch.isfinite(S0).all() and S0.norm().item() < 2 * v.norm(dim=-1).sum().item()


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
