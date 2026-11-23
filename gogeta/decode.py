"""Incremental (cached) inference and lossless self-speculative decoding for gogeta.model.FusionDance.

    logits_last, cache = prefill(model, idx, loops=r)        # idx [B,T] -> logits [B,V] at the last position
    logits = step(model, cache, tokens, loops=r)             # tokens [B,k] (k >= 1) -> logits [B,k,V]
    cache.rollback(n)                                        # forget the last n cached tokens (n <= max_rollback)
    out, stats = speculative_generate(model, prompt, n_new, r_draft, r_full, k)

The model is never modified: every function only calls its submodules (and their weights) in the same order as
FusionDance.forward. Per block kind, the cache holds:

  attn  full causal attention: K and V of every past position (RoPE applied at the absolute position).
  swa   sliding window w: the last w-1 (+ rollback log) keys/values; a query at position t attends t-w+1..t (the
        training mask `0 <= t - s < w`, applied over absolute positions, bottom-right aligned for k > 1).
  gdn   the gated-delta-rule state S [B,H,K,V] (fp32) plus the raw q/k/v projections feeding the ShortConvs
        (conv_size-1 past inputs, + rollback log; zeros at the start, which is the training zero padding).

Rollback: every state keeps a log of its last max_rollback tokens (GDN: S after each of them plus the conv inputs;
SWA: window-1 + max_rollback entries), so cache.rollback(n <= max_rollback) is exact with no replay, across any
number of steps. Memory cost: max_rollback GDN states per GDN layer and pass.

Loops: the shared core is applied r times; pass i of core block j has its OWN state (cache.core[i][j]), because pass
i sees different inputs at every position. A cache is built for one r; the self-speculative draft uses a second
cache built for r_draft.

GDN math (as GatedDeltaNet.forward feeds fla's chunk kernel): q, k = l2norm(silu(conv(proj(x)))),
v = silu(conv(proj(x))), g = -exp(A_log) * softplus(a(x) + dt_bias), beta = sigmoid(b(x)) (or 2*sigmoid(b(x)) with
gdn_neg_eigval), then per head  S <- exp(g) S;  S <- S + k (beta (v - S^T k))^T;  o = S^T (q * hd^-0.5),
followed by out_norm(o) * silu(gate(x)) and the o projection. On CPU the recurrence runs in torch (fp32). On CUDA
it uses fla's fused_recurrent_gated_delta_rule (keyword arguments only: positional arguments are the C110 bug) and
falls back to the torch path if that call raises. The CUDA path is NOT tested here (no GPU in this environment).

Not supported: `inputs_embeds` (the byte front end) and the training-time truncated BPTT (inference only).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from gogeta.model import Attention, FusionDance, GatedDeltaNet

USE_FLA_ON_CUDA = True   # set False to force the torch recurrence on CUDA too
_FLA_FAILED: list = []    # exception messages of failed fla calls (after the first, the torch path is used)


# ----------------------------------------------------------------------------- per-layer states
# Every state keeps a rollback log of its last `n_log` tokens (n_log <= the cache's max_rollback, across any number of
# steps): rollback(n) for n <= n_log is exact and needs no recomputation.
class AttnState:
    """K/V cache of one attention layer (window=None: full attention; else sliding window)."""

    def __init__(self, window: int | None):
        self.window = window
        self.k = None        # [B,H,L,hd]
        self.v = None
        self.n_log = 0

    def trim(self):
        """Sliding window: keep the window-1 entries the next query needs, plus the rollback log."""
        if self.window is not None and self.k is not None:
            keep = self.window - 1 + self.n_log
            if self.k.shape[2] > keep:
                self.k = self.k[:, :, self.k.shape[2] - keep:]
                self.v = self.v[:, :, self.v.shape[2] - keep:]

    def rollback(self, n: int):
        L = self.k.shape[2]
        self.k, self.v = self.k[:, :, :L - n], self.v[:, :, :L - n]
        self.n_log -= n

    def nbytes(self) -> int:
        return 0 if self.k is None else 2 * self.k.numel() * self.k.element_size()


class GDNState:
    """Gated DeltaNet recurrent state + ShortConv rolling inputs of one GDN layer."""

    def __init__(self):
        self.S = None          # [B,H,K,V] fp32, state after the last cached token
        self.conv = None       # 3 tensors [B, conv_size-1 + n_log, C]: raw q/k/v projections
        self.S_base = None     # state before the first logged token
        self.hist = []         # states after each logged token (len n_log)
        self.n_log = 0

    def rollback(self, n: int):
        keep = self.n_log - n
        self.S = self.hist[keep - 1] if keep > 0 else self.S_base
        self.hist = self.hist[:keep]
        self.conv = [c[:, :c.shape[1] - n] for c in self.conv]
        self.n_log = keep

    def nbytes(self) -> int:
        n = 0 if self.S is None else self.S.numel() * self.S.element_size()
        n += sum(h.numel() * h.element_size() for h in self.hist)
        return n + sum(c.numel() * c.element_size() for c in (self.conv or []))


def _new_state(block):
    m = block.mixer
    if isinstance(m, GatedDeltaNet):
        return GDNState()
    if isinstance(m, Attention):
        return AttnState(m.window)
    raise TypeError(f"unsupported mixer {type(m).__name__}")


class DecodeCache:
    """All per-layer states of one FusionDance at one loop count r (core states are per pass)."""

    def __init__(self, model: FusionDance, loops: int, max_rollback: int = 16):
        self.loops = loops if model.core else 0
        self.pos = 0                      # number of cached positions
        self.max_rollback = max_rollback  # the last this-many tokens can be rolled back
        self.n_log = 0
        self.prelude = [_new_state(b) for b in model.prelude]
        self.core = [[_new_state(b) for b in model.core] for _ in range(self.loops)]
        self.coda = [_new_state(b) for b in model.coda]

    def states(self):
        yield from self.prelude
        for p in self.core:
            yield from p
        yield from self.coda

    def rollback(self, n: int):
        """Forget the last n cached tokens (n <= min(max_rollback, tokens cached))."""
        if n == 0:
            return
        if not 0 < n <= self.n_log:
            raise ValueError(f"rollback of {n} tokens; only the last {self.n_log} can be rolled back")
        for s in self.states():
            s.rollback(n)
        self.pos -= n
        self.n_log -= n

    def nbytes(self) -> int:
        return sum(s.nbytes() for s in self.states())


# ----------------------------------------------------------------------------- mixers
def _rope_at(rope, x: torch.Tensor, pos0: int) -> torch.Tensor:
    """gogeta.model.Rotary.forward with a position offset (x: [B,H,T,hd] at positions pos0..pos0+T-1)."""
    T = x.shape[2]
    if pos0 + T > rope.cos.shape[2]:
        raise ValueError(f"position {pos0 + T} exceeds max_seq_len {rope.cos.shape[2]}")
    c, s = rope.cos[:, :, pos0:pos0 + T].to(x.dtype), rope.sin[:, :, pos0:pos0 + T].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


def _attn_step(m: Attention, st: AttnState, x: torch.Tensor, pos0: int, n_log: int) -> torch.Tensor:
    B, T, D = x.shape
    H, hd = m.cfg.n_heads, m.cfg.head_dim
    q, k, v = m.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k = _rope_at(m.rope, q, pos0), _rope_at(m.rope, k, pos0)
    Lc = 0 if st.k is None else st.k.shape[2]
    K = k if st.k is None else torch.cat([st.k, k], dim=2)
    V = v if st.v is None else torch.cat([st.v, v], dim=2)
    if Lc == 0 and (m.window is None or m.window >= T):
        y = F.scaled_dot_product_attention(q, K, V, is_causal=True)      # identical call to training's
    else:   # causal (+ window) mask over absolute positions, bottom-right aligned
        qp = torch.arange(pos0, pos0 + T, device=x.device)
        kp = torch.arange(pos0 - Lc, pos0 + T, device=x.device)
        d = qp[:, None] - kp[None, :]
        mask = d >= 0
        if m.window is not None:
            mask = mask & (d < m.window)
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
    st.k, st.v, st.n_log = K, V, n_log
    st.trim()
    return m.o(y.transpose(1, 2).reshape(B, T, D))


def _conv_step(conv_mod, buf: torch.Tensor, x: torch.Tensor):
    """ShortConv on new inputs x [B,T,C] given >= conv_size-1 previous raw inputs buf [B,n,C] -> (out, buf + x)."""
    w = conv_mod.conv.weight                                     # [C,1,cs]
    cs = w.shape[-1]
    inp = torch.cat([buf, x], dim=1)
    win = inp[:, inp.shape[1] - (cs - 1 + x.shape[1]):]
    y = F.conv1d(win.transpose(1, 2), w, groups=w.shape[0])     # valid conv: T outputs
    return F.silu(y.transpose(1, 2)), inp


def _gdn_recurrence_torch(q, k, v, g, beta, scale, S, keep_from: int):
    """q,k [B,T,H,K] v [B,T,H,V] g,beta [B,T,H] (fp32), S [B,H,K,V].

    Returns o [B,T,H,V], the final S, the states after tokens keep_from..T-1, and the state before token keep_from."""
    outs, hist, S_pre = [], [], S
    for t in range(q.shape[1]):
        if t == keep_from:
            S_pre = S
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t] - pred)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
        if t >= keep_from:
            hist.append(S)
    if keep_from >= q.shape[1]:
        S_pre = S
    return torch.stack(outs, 1), S, hist, S_pre


def _gdn_recurrence_fla(q, k, v, g, b_arg, scale, S, keep_from: int, neg_eigval: bool):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fr

    kw = dict(scale=scale, output_final_state=True)
    if neg_eigval:
        kw.update(use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    dt = torch.bfloat16

    def call(sl, S):   # keyword arguments only (C110: the 5th positional argument is gk, not beta)
        o, S = fr(q=q[:, sl].to(dt), k=k[:, sl].to(dt), v=v[:, sl].to(dt), g=g[:, sl], beta=b_arg[:, sl].to(dt),
                  initial_state=S, **kw)
        return o.float(), S.float()

    outs, hist = [], []
    if keep_from > 0:
        o, S = call(slice(0, keep_from), S)
        outs.append(o)
    S_pre = S
    for t in range(keep_from, q.shape[1]):   # one token per call so every logged state is kept for rollback
        o, S = call(slice(t, t + 1), S)
        outs.append(o)
        hist.append(S)
    return torch.cat(outs, 1), S, hist, S_pre


def _gdn_step(m: GatedDeltaNet, st: GDNState, x: torch.Tensor, n_log: int) -> torch.Tensor:
    B, T, _ = x.shape
    H, hd = m.H, m.hd
    C = H * hd
    cs = m.conv_q.conv.weight.shape[-1]
    if st.S is None:
        st.S = torch.zeros(B, H, hd, hd, device=x.device, dtype=torch.float32)
        st.conv = [x.new_zeros(B, cs - 1, C) for _ in range(3)]
    (cq, iq), (ck, ik), (cv, iv) = (_conv_step(mod, buf, proj(x)) for mod, buf, proj in
                                    zip((m.conv_q, m.conv_k, m.conv_v), st.conv, (m.q, m.k, m.v)))
    q = F.normalize(cq.view(B, T, H, hd), dim=-1)
    k = F.normalize(ck.view(B, T, H, hd), dim=-1)
    v = cv.view(B, T, H, hd)
    g = -m.A_log.float().exp() * F.softplus(m.a(x).float() + m.dt_bias.float())
    raw = m.b(x).float()
    scale = hd ** -0.5
    n_new_logged = min(T, n_log)
    keep_from = T - n_new_logged
    o = None
    if x.is_cuda and USE_FLA_ON_CUDA and not _FLA_FAILED:
        try:
            b_arg = raw if m.neg_eigval else torch.sigmoid(raw)
            o, S, hist_new, S_pre = _gdn_recurrence_fla(q, k, v, g, b_arg, scale, st.S, keep_from, m.neg_eigval)
        except Exception as exc:  # noqa: BLE001  (fall back to the torch recurrence, remember why)
            _FLA_FAILED.append(f"{type(exc).__name__}: {exc}")
            o = None
    if o is None:
        dt = x.dtype   # training casts q/k/v/beta to x.dtype on CPU before the kernel
        beta = (2.0 * torch.sigmoid(raw.to(dt).float())) if m.neg_eigval else torch.sigmoid(raw).to(dt).float()
        o, S, hist_new, S_pre = _gdn_recurrence_torch(q.to(dt).float(), k.to(dt).float(), v.to(dt).float(), g, beta,
                                                      scale, st.S, keep_from)
    c_old = n_log - n_new_logged            # logged tokens carried over from earlier steps
    if c_old > 0:
        st.S_base = ([st.S_base] + st.hist)[len(st.hist) - c_old]
        st.hist = st.hist[len(st.hist) - c_old:] + hist_new
    else:
        st.S_base, st.hist = S_pre, hist_new
    st.S, st.n_log = S, n_log
    keep = (cs - 1) + n_log
    st.conv = [i[:, i.shape[1] - keep:] for i in (iq, ik, iv)]
    o = m.out_norm(o.to(x.dtype)) * F.silu(m.gate(x).view(B, T, H, hd))
    return m.o(o.reshape(B, T, H * hd))


def _block_step(b, st, x: torch.Tensor, pos0: int, n_log: int) -> torch.Tensor:
    xn = b.norm1(x)
    if isinstance(st, GDNState):
        x = x + _gdn_step(b.mixer, st, xn, n_log)
    else:
        x = x + _attn_step(b.mixer, st, xn, pos0, n_log)
    return x + b.mlp(b.norm2(x))   # SwiGLU or MoEFFN (aux=None): both per-token, stateless


# ----------------------------------------------------------------------------- model-level API
@torch.no_grad()
def _forward_cached(model: FusionDance, cache: DecodeCache, idx: torch.Tensor) -> torch.Tensor:
    cfg = model.cfg
    T = idx.shape[1]
    pos0 = cache.pos
    n_log = min(cache.max_rollback, cache.n_log + T)
    x = model.embed(idx)
    for b, st in zip(model.prelude, cache.prelude):
        x = _block_step(b, st, x, pos0, n_log)
    if model.core:
        e, h = x, x
        for i in range(cache.loops):
            inj = model.inject(torch.cat([model.inj_norm_h(h), model.inj_norm_e(e)], dim=-1))
            if cfg.inject_contractive:
                h = torch.sigmoid(model.lam_logit) * h + inj
            else:
                h = h + inj
            if cfg.loop_embed:
                h = h + model.loop_emb[min(i, model.loop_emb.shape[0] - 1)]
            p = min(i, cfg.loops_max - 1)
            for j, b in enumerate(model.core):
                h = _block_step(b, cache.core[i][j], h, pos0, n_log)
                if cfg.loop_adapter_rank:
                    h = h + (model.adapt_norm(h) @ model.adapt_down[p, j]) @ model.adapt_up[p, j]
        x = h
    for b, st in zip(model.coda, cache.coda):
        x = _block_step(b, st, x, pos0, n_log)
    cache.pos += T
    cache.n_log = n_log
    return model.head(model.norm_out(x))


def new_cache(model: FusionDance, loops: int | None = None, max_rollback: int = 16) -> DecodeCache:
    return DecodeCache(model, loops if loops is not None else model.cfg.loops, max_rollback)


def prefill(model: FusionDance, idx: torch.Tensor, loops: int | None = None, return_all: bool = False,
            max_rollback: int = 16):
    """Run the prompt idx [B,T] through the model, filling a fresh cache.

    Returns (logits [B,V] at the last position, cache), or (logits [B,T,V], cache) with return_all=True."""
    cache = new_cache(model, loops, max_rollback)
    logits = _forward_cached(model, cache, idx)
    return (logits if return_all else logits[:, -1]), cache


def step(model: FusionDance, cache: DecodeCache, tokens: torch.Tensor, loops: int | None = None) -> torch.Tensor:
    """Process tokens [B,k] (k >= 1) at positions cache.pos..cache.pos+k-1 -> logits [B,k,V]; appends to cache.

    `loops` must match the loop count the cache was built for (None = the cache's)."""
    if loops is not None and model.core and loops != cache.loops:
        raise ValueError(f"cache was built for loops={cache.loops}, step called with loops={loops}")
    if tokens.dim() == 1:
        tokens = tokens[None]
    return _forward_cached(model, cache, tokens)


@torch.no_grad()
def greedy_generate(model: FusionDance, prompt, n_new: int, loops: int | None = None) -> list:
    """Plain cached greedy decoding (one token per step): the losslessness reference."""
    idx = _as_idx(prompt, model)
    logits, cache = prefill(model, idx, loops, max_rollback=0)
    out = [int(logits[0].argmax())]
    while len(out) < n_new:
        logits = step(model, cache, torch.tensor([[out[-1]]], device=idx.device))
        out.append(int(logits[0, -1].argmax()))
    return out


def _as_idx(prompt, model) -> torch.Tensor:
    dev = model.embed.weight.device
    idx = prompt if torch.is_tensor(prompt) else torch.tensor(prompt, dtype=torch.long)
    idx = idx.to(dev)
    if idx.dim() == 1:
        idx = idx[None]
    if idx.shape[0] != 1:
        raise ValueError("generation is batch-1")
    return idx


@torch.no_grad()
def speculative_generate(model: FusionDance, prompt, n_new: int, r_draft: int, r_full: int, k: int,
                         return_stats: bool = True):
    """Lossless greedy self-speculative decoding.

    Draft = the same model with r_draft core passes and its own cache; it proposes up to k tokens greedily. The full
    model (r_full passes) verifies [pending, d_1..d_k] in ONE step call (k+1 positions: the pending token is the
    full model's previous correction/bonus token, not yet in either cache). The longest prefix of drafts equal to the
    full model's argmax is accepted, plus the full model's own token at the first mismatch (or the bonus token after
    d_k). Both caches are rolled back to the accepted prefix (KV/SWA: truncate; GDN: logged-state lookup), so the
    output equals greedy_generate(model, prompt, n_new, r_full) token for token, up to floating-point near-ties
    between a k-token verify and a 1-token step.

    Returns (tokens, stats): proposed/accepted drafts, acceptance rate, full-model verify calls, draft calls, tokens
    per verify call and wall-clock seconds."""
    if k < 1:
        raise ValueError("k >= 1")
    idx = _as_idx(prompt, model)
    dev = idx.device
    if idx.shape[1] + n_new + k + 1 > model.cfg.max_seq_len:
        raise ValueError("prompt + n_new + k exceeds max_seq_len")
    t0 = time.perf_counter()
    lf, cf = prefill(model, idx, r_full, max_rollback=k + 1)
    _, cd = prefill(model, idx, r_draft, max_rollback=k + 1)
    out = [int(lf[0].argmax())]
    pending = out[0]
    proposed = accepted = verify_calls = draft_calls = rounds_full_accept = 0
    accept_hist = [0] * (k + 1)
    while len(out) < n_new:
        kk = min(k, n_new - len(out))
        drafts, t = [], pending
        for _ in range(kk):
            lg = step(model, cd, torch.tensor([[t]], device=dev))
            draft_calls += 1
            t = int(lg[0, -1].argmax())
            drafts.append(t)
        # the draft cache now holds pending + drafts[:kk-1]
        lv = step(model, cf, torch.tensor([[pending] + drafts], device=dev))
        verify_calls += 1
        pred = lv[0].argmax(-1).tolist()          # pred[i] = full model's token after position i of the verify block
        m = 0
        while m < kk and drafts[m] == pred[m]:
            m += 1
        cf.rollback(kk - m)                         # keep pending + d_1..d_m
        if m < kk:
            cd.rollback(kk - 1 - m)
        else:                                       # all accepted: the draft still needs d_kk in its cache
            step(model, cd, torch.tensor([[drafts[-1]]], device=dev))
            draft_calls += 1
            rounds_full_accept += 1
        out.extend(drafts[:m] + [pred[m]])
        pending = pred[m]
        proposed += kk
        accepted += m
        accept_hist[m] += 1
    out = out[:n_new]
    stats = {"n_new": n_new, "r_draft": r_draft, "r_full": r_full, "k": k, "proposed": proposed, "accepted": accepted,
             "acceptance_rate": accepted / proposed if proposed else None,
             "verify_calls": verify_calls, "full_calls_incl_prefill": verify_calls + 1, "draft_calls": draft_calls,
             "tokens_per_verify_call": (n_new - 1) / verify_calls if verify_calls else None,
             "full_calls_per_token_incl_prefill": (verify_calls + 1) / n_new,
             "accept_len_hist": accept_hist, "rounds_all_accepted": rounds_full_accept,
             "seconds": time.perf_counter() - t0}
    return (out, stats) if return_stats else out


# ----------------------------------------------------------------------------- CPU reference kernel (tests/bench)
def gated_delta_rule_ref(q, k, v, g, beta, scale=None, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                         **_):
    """Torch stand-in for fla's chunk_gated_delta_rule (same signature subset), used to run the model's OWN full
    forward on CPU as the reference (the pattern of tests/test_moe.cpu_gdn). Not used by the decode path."""
    B, T, H, dk = q.shape
    scale = dk ** -0.5 if scale is None else scale
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    S = q.new_zeros(B, H, dk, v.shape[-1])
    o, S, _, _ = _gdn_recurrence_torch(q, k, v, g, beta, scale, S, T)
    return o, S


def use_reference_gdn(model: FusionDance) -> FusionDance:
    """Point every GatedDeltaNet instance's _kernel at gated_delta_rule_ref (instance attribute; source untouched)."""
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m._kernel = gated_delta_rule_ref
    return model
