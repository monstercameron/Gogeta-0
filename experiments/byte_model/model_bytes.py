"""FusionDance: prelude -> looped core x r -> coda, with pluggable sequence mixers.

Block kinds
  attn  full causal softmax attention (RoPE)              O(n^2)
  swa   sliding-window causal attention (RoPE, window w)  O(n*w) in principle*
  gdn   Gated DeltaNet (fla chunk kernel)                 O(n), fixed-size state

  * implemented with a dense boolean mask via SDPA for now, so the *kernel* is still
    O(n^2) at our 1k context. Swap for a block-sparse kernel before long-context tests.

Every block is pre-norm:  x += mixer(norm(x));  x += SwiGLU(norm(x)).

Looping (when `core` is non-empty):
  e = prelude(embed(tokens))
  h = e
  repeat r times:  h = core(h + inject([norm(h), norm(e)]))      (input re-injection)
  logits = head(norm(coda(h)))
Training samples r uniformly in [loops_min, loops_max] per step, and backprops only
through the last `bptt_loops` iterations (earlier ones run under no_grad).

`ternary=True` swaps the large projections (attention q/k/v/o, GDN q/k/v/gate/o, MLP,
re-injection) for BitNet b1.58 BitLinear: absmean ternary weights + 8-bit absmax
activations, both with straight-through estimators. Embeddings, norms, the tiny GDN
gate projections (a, b) and the short convolutions stay full precision.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- config
@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_heads: int = 6              # attention + GDN heads (head_dim = d_model / n_heads)
    mlp_hidden: int = 1024        # SwiGLU hidden size
    prelude: list = field(default_factory=lambda: ["swa"])
    core: list = field(default_factory=lambda: ["gdn", "gdn", "swa"])
    coda: list = field(default_factory=lambda: ["attn"])
    loops: int = 4                # loops used at eval / by default
    loops_min: int = 1            # training: r ~ U{loops_min..loops_max}
    loops_max: int = 7
    bptt_loops: int = 4           # backprop through the last k loops only
    window: int = 256             # sliding-window size
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    conv_size: int = 4            # GDN short conv
    gdn_neg_eigval: bool = False  # GDN write strength beta in (0, 2) instead of (0, 1): negative eigenvalues (SQ2/E11)
    ternary: bool = False
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in BYTE_DEFAULTS.items():  # token models: config dict identical to before the byte extension
            if d.get(k) == v:
                d.pop(k)
        return d

    # ---- byte-level coarse-to-fine (PatchedByteLM). Opt-in: with patch_size = 0 none of these fields is read, the
    # model is the token FusionDance (bit-identical), and to_dict() omits them while they are at their defaults.
    # In patched mode d_model/n_heads/mlp_hidden/window/prelude/core/coda/loops* describe the GLOBAL body on patches
    # (window in patches), max_seq_len is in BYTES, and `ternary` applies to the global body only (K4).
    patch_size: int = 0           # P > 0: bytes -> fixed P-byte patches -> global body -> byte decoder
    patch_shift: bool = True      # one-patch causal shift; False is a deliberately BROKEN variant (causality tests only)
    d_local: int = 192            # width of the byte-level encoder / decoder
    local_heads: int = 3          # heads of local GDN / attention (head_dim = d_local / local_heads)
    local_mlp_hidden: int = 512   # local SwiGLU hidden size
    local_window: int = 64        # window (bytes) of local "swa" / "lswa" blocks
    local_encoder: list = field(default_factory=lambda: ["conv", "gdn"])   # byte encoder (causal) before patching
    refine_blocks: list = field(default_factory=lambda: ["conv", "gdn"])   # one refinement pass (causal)
    refine_passes: int = 1        # r: passes of the byte refinement loop
    refine_min: int = 0           # training: r ~ U{refine_min..refine_max} when refine_max > refine_min (shared only)
    refine_max: int = 0
    refine_shared: bool = True    # False: r passes with separate weights (the equal-FLOP unshared control)
    refine_update: str = "gated"  # residual: h <- f(u) | gated: h <- h + sig(g_k)*(f(u)-h) | contractive: alpha_max*sig
    refine_alpha_max: float = 0.5
    refine_loss: object = "linear"  # per-pass loss weights: "last" | "uniform" | "linear" (w_k = k) | [w_1, ..., w_r]
    decoder: str = "ar"           # ar: byte-autoregressive in the patch | parallel_refine: all P bytes, refined r times
    refine_feedback: str = "soft"  # parallel_refine: previous-pass guesses as "soft" (expected) or "argmax" embeddings


BYTE_FIELDS = ("patch_size", "patch_shift", "d_local", "local_heads", "local_mlp_hidden", "local_window",
               "local_encoder", "refine_blocks", "refine_passes", "refine_min", "refine_max", "refine_shared",
               "refine_update", "refine_alpha_max", "refine_loss", "decoder", "refine_feedback")
BYTE_DEFAULTS = {k: v for k, v in asdict(ModelConfig()).items() if k in BYTE_FIELDS}
BYTES_PER_TOKEN = 3.735  # measured: val bytes / BPE tokens (notes/research-byte-level.md, seq_stats.json)


# ----------------------------------------------------------------------------- BitLinear
class BitLinear(nn.Linear):
    """BitNet b1.58 linear: ternary weights (absmean) + int8 activations (absmax), STE."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        w_q = (w / scale).round().clamp(-1, 1) * scale
        w = w + (w_q - w).detach()
        s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * s).round().clamp(-128, 127) / s
        x = x + (x_q - x).detach()
        return F.linear(x, w, self.bias)


def make_linear(cfg: ModelConfig, d_in: int, d_out: int) -> nn.Linear:
    return (BitLinear if cfg.ternary else nn.Linear)(d_in, d_out, bias=False)


# ----------------------------------------------------------------------------- pieces
class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos()[None, None], persistent=False)  # [1,1,T,hd/2]
        self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,H,T,hd]
        T = x.shape[2]
        c, s = self.cos[:, :, :T].to(x.dtype), self.sin[:, :, :T].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = make_linear(cfg, cfg.d_model, 2 * cfg.mlp_hidden)
        self.down = make_linear(cfg, cfg.mlp_hidden, cfg.d_model)
        self.down.is_residual_out = True

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, window: int | None, rope: Rotary):
        super().__init__()
        self.cfg, self.window, self.rope = cfg, window, rope
        self.qkv = make_linear(cfg, cfg.d_model, 3 * cfg.d_model)
        self.o = make_linear(cfg, cfg.d_model, cfg.d_model)
        self.o.is_residual_out = True
        self._mask_cache: dict = {}

    def _mask(self, T: int, device) -> torch.Tensor:
        key = (T, device)
        if key not in self._mask_cache:
            i = torch.arange(T, device=device)
            d = i[:, None] - i[None, :]
            self._mask_cache[key] = (d >= 0) & (d < self.window)  # True = attend
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)  # each [B,H,T,hd]
        q, k = self.rope(q), self.rope(k)
        if self.window is None or self.window >= T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(T, x.device))
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class ShortConv(nn.Module):
    """Causal depthwise conv over time + SiLU."""

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, k, groups=channels, padding=k - 1, bias=False)

    def forward(self, x):  # [B,T,C]
        T = x.shape[1]
        return F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet mixer (Yang et al. 2024) on fla's chunk kernel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # verified by tests/test_gdn_kernels.py

        self._kernel = chunk_gated_delta_rule
        D, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.H, self.hd = H, hd
        self.q, self.k, self.v = (make_linear(cfg, D, H * hd) for _ in range(3))
        self.gate = make_linear(cfg, D, H * hd)
        self.o = make_linear(cfg, H * hd, D)
        self.o.is_residual_out = True
        self.a = nn.Linear(D, H, bias=False)       # decay input (full precision)
        self.b = nn.Linear(D, H, bias=False)       # write strength beta (full precision)
        self.conv_q, self.conv_k, self.conv_v = (ShortConv(H * hd, cfg.conv_size) for _ in range(3))
        # Mamba-2 style decay: g = -exp(A_log) * softplus(a(x) + dt_bias)
        A = torch.empty(H).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.out_norm = nn.RMSNorm(hd)
        # Negative eigenvalues (Grazzi et al.; verified locally by SQ6 arm (e)): beta = 2*sigmoid(b(x)) in (0, 2),
        # so (I - beta k k^T) can flip sign. fla only allows this with the sigmoid applied inside the kernel.
        self.neg_eigval = cfg.gdn_neg_eigval

    def forward(self, x):
        B, T, _ = x.shape
        H, hd = self.H, self.hd
        q = F.normalize(self.conv_q(self.q(x)).view(B, T, H, hd), dim=-1)
        k = F.normalize(self.conv_k(self.k(x)).view(B, T, H, hd), dim=-1)
        v = self.conv_v(self.v(x)).view(B, T, H, hd)
        g = -self.A_log.float().exp() * F.softplus(self.a(x).float() + self.dt_bias.float())
        dtype = torch.bfloat16 if x.is_cuda else x.dtype
        if self.neg_eigval:
            raw = self.b(x).float()  # logits; the kernel applies 2*sigmoid
            o, _ = self._kernel(q.to(dtype), k.to(dtype), v.to(dtype), g, raw.to(dtype), scale=hd ** -0.5,
                                use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
        else:
            beta = torch.sigmoid(self.b(x).float())
            o, _ = self._kernel(q.to(dtype), k.to(dtype), v.to(dtype), g, beta.to(dtype), scale=hd ** -0.5)
        o = self.out_norm(o.to(x.dtype)) * F.silu(self.gate(x).view(B, T, H, hd))
        return self.o(o.reshape(B, T, H * hd))


class BlockLocalAttention(Attention):
    """Exact sliding-window causal attention ("lswa") computed block-locally: queries in block b of size w = window
    attend to keys in blocks b-1 and b under the window mask, so cost and memory are O(T * 2w), not O(T^2) like the
    dense-mask "swa" path (C12). Same parameters and RoPE as Attention; meant for small byte-level windows."""

    def _bmask(self, w: int, nb: int, device) -> torch.Tensor:
        key = ("blk", w, nb, device)
        if key not in self._mask_cache:
            i = torch.arange(w, device=device)[:, None]
            j = torch.arange(2 * w, device=device)[None, :]
            d = w + i - j                                   # query position minus key position
            m = ((d >= 0) & (d < self.window)).expand(nb, w, 2 * w).clone()
            m[0, :, :w] = False                             # block 0 has no previous block
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        H, hd, w = self.cfg.n_heads, self.cfg.head_dim, self.window
        q, k, v = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q), self.rope(k)
        if T <= w:  # the window covers the whole sequence: plain causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, D))
        pad = (-T) % w
        if pad:  # right padding: causal, so it cannot reach real positions
            q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
        nb = (T + pad) // w
        qb = q.view(B, H, nb, w, hd)
        kb, vb = (z.view(B, H, nb, w, hd) for z in (k, v))
        kb = torch.cat([F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :nb], kb], dim=3)  # [B,H,nb,2w,hd]: prev block + own
        vb = torch.cat([F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :nb], vb], dim=3)
        y = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=self._bmask(w, nb, x.device))
        y = y.reshape(B, H, nb * w, hd)[:, :, :T]
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, kind: str, cfg: ModelConfig, rope: Rotary):
        super().__init__()
        self.kind = kind
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        if kind == "attn":
            self.mixer = Attention(cfg, None, rope)
        elif kind == "swa":
            self.mixer = Attention(cfg, cfg.window, rope)
        elif kind == "gdn":
            self.mixer = GatedDeltaNet(cfg)
        elif kind == "lswa":
            self.mixer = BlockLocalAttention(cfg, cfg.window, rope)
        else:
            raise ValueError(f"unknown block kind {kind!r}")
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ----------------------------------------------------------------------------- model
class FusionDance(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        mk = lambda kinds: nn.ModuleList(Block(k, cfg, self.rope) for k in kinds)
        self.prelude, self.core, self.coda = mk(cfg.prelude), mk(cfg.core), mk(cfg.coda)
        if self.core:
            self.inj_norm_h, self.inj_norm_e = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
            self.inject = make_linear(cfg, 2 * cfg.d_model, cfg.d_model)
            self.inject.is_residual_out = True
        self.norm_out = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self._init_weights()

    # depth used to scale residual-output init: effective blocks at the default loop count
    def effective_depth(self, loops: int | None = None) -> int:
        r = (loops or self.cfg.loops) if self.core else 0
        return len(self.prelude) + len(self.coda) + r * len(self.core)

    def _init_weights(self):
        std_res = 0.02 / math.sqrt(2 * max(1, self.effective_depth()))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std_res if getattr(m, "is_residual_out", False) else 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None):
        x = self.embed(idx)
        for b in self.prelude:
            x = b(x)
        if self.core:
            r = loops if loops is not None else self.cfg.loops
            n_nograd = max(0, r - self.cfg.bptt_loops) if self.training else 0
            e, h = x, x
            for i in range(r):
                with torch.no_grad() if i < n_nograd else nullcontext():
                    h = h + self.inject(torch.cat([self.inj_norm_h(h), self.inj_norm_e(e)], dim=-1))
                    for b in self.core:
                        h = b(h)
            x = h
        for b in self.coda:
            x = b(x)
        logits = self.head(self.norm_out(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ------------------------------------------------------------------ accounting
    def param_counts(self) -> dict:
        def n(mods):
            return sum(p.numel() for m in mods for p in m.parameters())

        emb = self.embed.weight.numel()
        pre, core, coda = n(self.prelude), n(self.core), n(self.coda)
        other = sum(p.numel() for p in self.parameters()) - emb - pre - core - coda
        return {"total": sum(p.numel() for p in self.parameters()), "embedding": emb,
                "prelude": pre, "core": core, "coda": coda, "other": other,
                "non_embedding": pre + core + coda + other}

    def flops_per_token(self, seq_len: int, loops: int | None = None, training: bool = True) -> float:
        """Approximate FLOPs per token (6N rule for matmuls + attention/GDN terms).

        Matmul params counted once per *application* (core counted r times). Attention adds
        4*d*ctx per layer per token forward (QK^T and AV), ctx = mean attended length.
        GDN adds ~4*H*hd^2 per layer per token forward for state read/write. Training = 3x forward.
        """
        cfg = self.cfg
        r = (loops or cfg.loops) if self.core else 0
        pc = self.param_counts()
        inj = 2 * cfg.d_model * cfg.d_model if self.core else 0
        matmul_params = pc["prelude"] + pc["coda"] + r * (pc["core"] + inj) + cfg.d_model * cfg.vocab_size
        fwd = 2 * matmul_params
        kinds = list(cfg.prelude) + list(cfg.coda) + r * list(cfg.core)
        for k in kinds:
            if k == "attn":
                fwd += 4 * cfg.d_model * (seq_len / 2)
            elif k == "swa":
                fwd += 4 * cfg.d_model * min(cfg.window, seq_len / 2)
            elif k == "gdn":
                fwd += 4 * cfg.n_heads * cfg.head_dim ** 2
        return fwd * (3 if training else 1)


# ----------------------------------------------------------------------------- byte-level coarse-to-fine
# Indexing (row t of the input predicts byte t+1 and may use bytes <= t only):
#   * patch p holds input bytes [pP, (p+1)P); the global body runs on patches, so its output G[p] has seen bytes
#     <= (p+1)P - 1;
#   * one-patch causal shift (MegaByte): row t reads Gs[(t+1)//P] with Gs = [start, G[0], ..., G[N-1]], i.e. G of the
#     last patch that is complete before the target byte t+1. Its slot in the target patch is (t+1) % P;
#   * ar: the decoder adds the previous-byte embedding (byte t itself) and runs causal byte-level mixers over the
#     whole sequence, so bytes of the current patch reach row t only through causal local layers;
#   * parallel_refine: row t may use real bytes <= P*((t+1)//P) - 1 only (everything before its target patch). Rows
#     are grouped by target patch (one leading dummy row aligns the groups); slot 0 sees the real previous byte,
#     the other slots a learned mask embedding; mixers run inside each group only (causal over slots), and pass
#     k > 0 adds embeddings of pass k-1's guesses (own slot and previous slot of the same group).
class ConvBlock(nn.Module):
    """Pre-norm causal depthwise conv residual: x + SiLU(conv(norm(x))) (block kind "conv", byte layers only)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model)
        self.conv = ShortConv(cfg.d_model, cfg.conv_size)

    def forward(self, x):
        return x + self.conv(self.norm(x))


def _local_cfg(cfg: ModelConfig) -> ModelConfig:
    return replace(cfg, d_model=cfg.d_local, n_heads=cfg.local_heads, mlp_hidden=cfg.local_mlp_hidden,
                   window=cfg.local_window, ternary=False, patch_size=0, prelude=[], core=[], coda=[])


def _global_cfg(cfg: ModelConfig) -> ModelConfig:
    return replace(cfg, max_seq_len=cfg.max_seq_len // cfg.patch_size + 1, patch_size=0)


def _local_blocks(kinds, lcfg: ModelConfig, rope: Rotary) -> nn.ModuleList:
    return nn.ModuleList(ConvBlock(lcfg) if k == "conv" else Block(k, lcfg, rope) for k in kinds)


class GlobalTrunk(nn.Module):
    """FusionDance's prelude -> core x r (re-injection, bptt window) -> coda, on patch vectors (no embed/head)."""

    def __init__(self, cfg: ModelConfig, rope: Rotary):
        super().__init__()
        self.cfg = cfg
        mk = lambda kinds: nn.ModuleList(Block(k, cfg, rope) for k in kinds)  # noqa: E731
        self.prelude, self.core, self.coda = mk(cfg.prelude), mk(cfg.core), mk(cfg.coda)
        if self.core:
            self.inj_norm_h, self.inj_norm_e = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
            self.inject = make_linear(cfg, 2 * cfg.d_model, cfg.d_model)
            self.inject.is_residual_out = True

    def forward(self, x, loops=None):
        for b in self.prelude:
            x = b(x)
        if self.core:
            r = loops if loops is not None else self.cfg.loops
            n_nograd = max(0, r - self.cfg.bptt_loops) if self.training else 0
            e, h = x, x
            for i in range(r):
                with torch.no_grad() if i < n_nograd else nullcontext():
                    h = h + self.inject(torch.cat([self.inj_norm_h(h), self.inj_norm_e(e)], dim=-1))
                    for b in self.core:
                        h = b(h)
            x = h
        for b in self.coda:
            x = b(x)
        return x


class PatchedByteLM(nn.Module):
    """Coarse-to-fine byte model (notes/analysis-coarse-to-fine.md): byte encoder -> P-byte patches -> global body
    (E1/E2-style, optionally looped / ternary) -> one-patch shift -> expansion -> byte refinement loop x r -> head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        P = cfg.patch_size
        assert P > 0 and cfg.max_seq_len % P == 0, "max_seq_len (bytes) must be a multiple of patch_size"
        assert cfg.decoder in ("ar", "parallel_refine"), cfg.decoder
        assert cfg.refine_update in ("residual", "gated", "contractive"), cfg.refine_update
        assert cfg.refine_feedback in ("soft", "argmax"), cfg.refine_feedback
        assert cfg.refine_shared or cfg.refine_max <= cfg.refine_min, "pass sampling needs shared weights"
        self.cfg, self.P = cfg, P
        self.lcfg, self.gcfg = _local_cfg(cfg), _global_cfg(cfg)
        lc, gc, dl, dg = self.lcfg, self.gcfg, cfg.d_local, cfg.d_model
        self.max_passes = max(cfg.refine_passes, cfg.refine_max)
        # byte level (always full precision, K4)
        self.lrope = Rotary(lc.head_dim, cfg.max_seq_len + P, cfg.rope_base)
        self.embed = nn.Embedding(cfg.vocab_size, dl)
        self.encoder = _local_blocks(cfg.local_encoder, lc, self.lrope)
        self.patch_in = nn.Linear(P * dl, dg, bias=False)
        # patch level (ternary if cfg.ternary)
        self.grope = Rotary(gc.head_dim, gc.max_seq_len, cfg.rope_base)
        self.body = GlobalTrunk(gc, self.grope)
        self.body_norm = nn.RMSNorm(dg)
        self.patch_start = nn.Parameter(torch.zeros(dg))     # Gs[0]: context of the first patch
        self.up = nn.Linear(dg, dl, bias=False)
        self.slot_emb = nn.Embedding(P, dl)                 # within-patch position of the target byte
        # refinement loop
        n_unique = 1 if cfg.refine_shared else cfg.refine_passes
        self.passes = nn.ModuleList(_local_blocks(cfg.refine_blocks, lc, self.lrope) for _ in range(n_unique))
        self.inj_norm_h, self.inj_norm_e = nn.RMSNorm(dl), nn.RMSNorm(dl)
        self.inject = nn.Linear(2 * dl, dl, bias=False)
        self.inject.is_residual_out = True
        self.pass_emb = nn.Embedding(self.max_passes, dl)
        if cfg.refine_update != "residual":
            self.gate = nn.Parameter(torch.zeros(self.max_passes, dl))   # sigmoid(0) = 0.5 at init
        if cfg.decoder == "parallel_refine":
            self.mask_emb = nn.Parameter(torch.zeros(dl))
            self.feedback = nn.Linear(2 * dl, dl, bias=False)
            self.feedback.is_residual_out = True
        self.norm_out = nn.RMSNorm(dl)
        self.head = nn.Linear(dl, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self.last_aux: dict = {}
        self._init_weights()

    @property
    def core(self):  # the trainer's "looped?" test reads model.core
        return self.body.core

    def depths(self) -> tuple[int, int]:  # (global, local) blocks applied, for the residual init scale
        g = len(self.body.prelude) + len(self.body.coda) + (self.cfg.loops if self.body.core else 0) * len(self.body.core)
        l = len(self.encoder) + self.cfg.refine_passes * len(self.cfg.refine_blocks)
        return g, l

    def _init_weights(self):
        g_depth, l_depth = self.depths()
        glob = {id(m) for m in self.body.modules()}
        for m in self.modules():
            depth = g_depth if id(m) in glob else l_depth
            std_res = 0.02 / math.sqrt(2 * max(1, depth))
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std_res if getattr(m, "is_residual_out", False) else 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.patch_start, std=0.02)
        if self.cfg.decoder == "parallel_refine":
            nn.init.normal_(self.mask_emb, std=0.02)

    # ------------------------------------------------------------------ helpers
    def loss_weights(self, r: int) -> list[float]:
        spec = self.cfg.refine_loss
        if isinstance(spec, (list, tuple)):
            w = [float(spec[min(k, len(spec) - 1)]) for k in range(r)]
        elif spec == "last":
            w = [0.0] * (r - 1) + [1.0]
        elif spec == "uniform":
            w = [1.0] * r
        elif spec == "linear":
            w = [float(k + 1) for k in range(r)]
        else:
            raise ValueError(f"unknown refine_loss {spec!r}")
        s = sum(w)
        return [x / s for x in w]

    def _pass(self, k: int):
        return self.passes[k if not self.cfg.refine_shared else 0]

    def _group(self, x, n_groups):  # [B,T,d] -> [B*(N+1), P, d], one leading dummy row, right padding
        B, T, d = x.shape
        P = self.P
        xx = torch.cat([x.new_zeros(B, 1, d), x, x.new_zeros(B, n_groups * P - T - 1, d)], dim=1)
        return xx.view(B * n_groups, P, d)

    def _ungroup(self, y, B, T):
        return y.reshape(B, -1, y.shape[-1])[:, 1:T + 1]

    # ------------------------------------------------------------------ forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None,
                passes: int | None = None, return_all: bool = False):
        cfg, P = self.cfg, self.P
        B, T0 = idx.shape
        pad = (-T0) % P
        if pad:  # generation with a prompt that is not a multiple of P: right padding only (causal)
            idx = F.pad(idx, (0, pad))
        T = idx.shape[1]
        N = T // P
        r = passes if passes is not None else cfg.refine_passes
        if not cfg.refine_shared:
            assert r == cfg.refine_passes, "unshared refinement runs exactly refine_passes passes"
        e = self.embed(idx)                                            # [B,T,dl]
        h = e
        for b in self.encoder:
            h = b(h)
        g = self.body(self.patch_in(h.reshape(B, N, P * cfg.d_local)), loops=loops)
        g = self.body_norm(g)
        t = torch.arange(T, device=idx.device)
        if cfg.patch_shift:
            gs = torch.cat([self.patch_start.to(g.dtype).expand(B, 1, -1), g], dim=1)  # [B,N+1,dg]
            gi, slot = (t + 1) // P, (t + 1) % P
        else:  # BROKEN on purpose: row t reads its own patch, which contains bytes > t
            gs, gi, slot = g, t // P, t % P
        x0 = self.up(gs)[:, gi] + self.slot_emb(slot)                  # shifted global context + slot embedding
        par = cfg.decoder == "parallel_refine"
        if par:
            first = (slot == 0)[None, :, None]
            x0 = x0 + torch.where(first, e, self.mask_emb.to(e.dtype).expand_as(e))
            n_groups = -(-(T + 1) // P)                                # ceil((T + 1) / P): dummy row + T rows
        else:
            x0 = x0 + e                                                # previous-byte embedding
        need_logits = set(range(r)) if (self.training or return_all or par) else {r - 1}
        w = self.loss_weights(r)
        if self.training and targets is not None and not (return_all or par):
            need_logits = {k for k in range(r) if w[k] > 0} | {r - 1}
        h, outs, guess = x0, [], None
        for k in range(r):
            u = h + self.inject(torch.cat([self.inj_norm_h(h), self.inj_norm_e(x0)], dim=-1))
            u = u + self.pass_emb.weight[min(k, self.max_passes - 1)]
            if par and guess is not None:
                prev = torch.cat([guess.new_zeros(B, 1, guess.shape[-1]), guess[:, :-1]], dim=1)
                prev = torch.where(first, torch.zeros_like(prev), prev)      # never read across a patch boundary
                u = u + self.feedback(torch.cat([prev, guess], dim=-1))
            y = self._group(u, n_groups) if par else u
            for b in self._pass(k):
                y = b(y)
            y = self._ungroup(y, B, T) if par else y
            if cfg.refine_update == "residual":
                h = y
            else:
                a = torch.sigmoid(self.gate[min(k, self.max_passes - 1)]).to(y.dtype)
                if cfg.refine_update == "contractive":
                    a = a * cfg.refine_alpha_max
                h = h + a * (y - h)
            if k in need_logits:
                logits = self.head(self.norm_out(h))
                outs.append((k, logits))
                if par and k < r - 1:
                    if cfg.refine_feedback == "soft":
                        guess = torch.softmax(logits.float(), dim=-1).to(self.embed.weight.dtype) @ self.embed.weight
                    else:
                        guess = self.embed(logits.argmax(dim=-1))
                    guess = guess.to(u.dtype)
        if pad:
            outs = [(k, lg[:, :T0]) for k, lg in outs]
        logits = outs[-1][1]
        self.last_aux = {}
        if targets is None:
            return (outs if return_all else logits), None
        V = logits.size(-1)
        ce = {k: F.cross_entropy(lg.float().reshape(-1, V), targets.reshape(-1), reduction="none").view(B, -1)
              for k, lg in outs}
        last = ce[r - 1]
        self.last_aux = {"pass_loss": [ce[k].mean().detach() for k in sorted(ce)], "pass_ids": sorted(ce),
                         "slot_loss": torch.zeros(P, device=last.device).index_add_(
                             0, slot[:T0], last.detach().sum(0)) / (B * torch.bincount(slot[:T0], minlength=P).clamp(min=1))}
        if self.training:
            loss = sum(w[k] * ce[k].mean() for k in ce if w[k] > 0)
        else:
            loss = last.mean()   # eval: the final pass (what generation uses)
        return (outs if return_all else logits), loss

    # ------------------------------------------------------------------ accounting
    def param_counts(self) -> dict:
        def n(*mods):
            return sum(p.numel() for m in mods for p in m.parameters())

        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        enc = n(self.encoder)
        glob = n(self.body, self.body_norm)
        patch_io = n(self.patch_in, self.up, self.slot_emb) + self.patch_start.numel()
        dec = total - emb - enc - glob - patch_io - n(self.norm_out)
        return {"total": total, "embedding": emb, "encoder": enc, "patch_io": patch_io, "global": glob,
                "global_core": n(self.body.core), "decoder": dec, "head_norm": n(self.norm_out),
                "local": total - glob, "non_embedding": total - emb}

    @staticmethod
    def _mixer_terms(kinds, c: ModelConfig, ctx: float) -> float:
        """Forward FLOPs per position beyond the matmuls, FusionDance's rule (attention: 4 d ctx; GDN: 4 H hd^2)."""
        f = 0.0
        for k in kinds:
            if k == "attn":
                f += 4 * c.d_model * (ctx / 2)
            elif k in ("swa", "lswa"):
                f += 4 * c.d_model * min(c.window, ctx / 2)
            elif k == "gdn":
                f += 4 * c.n_heads * c.head_dim ** 2
        return f

    def flops_breakdown(self, seq_len: int, loops: int | None = None, passes: int | None = None,
                        training: bool = True) -> dict:
        """Forward(x3 if training) FLOPs per BYTE position, by part. Same convention as FusionDance.flops_per_token:
        2 x params per application (6N for training), plus attention/GDN terms; the global body is applied once per
        patch (so /P per byte); the head is applied once per pass that carries loss in training."""
        cfg, P = self.cfg, self.P
        N = seq_len // P
        r = passes if passes is not None else cfg.refine_passes
        rg = (loops or cfg.loops) if self.body.core else 0
        n = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())  # noqa: E731
        gc, lc, dl, dg, V = self.gcfg, self.lcfg, cfg.d_local, cfg.d_model, cfg.vocab_size
        inj_g = 2 * dg * dg if self.body.core else 0
        g_mat = n(self.body.prelude, self.body.coda) + rg * (n(self.body.core) + inj_g)
        g_kinds = list(cfg.prelude) + list(cfg.coda) + rg * list(cfg.core)
        f = {"encoder": 2 * n(self.encoder) + self._mixer_terms(cfg.local_encoder, lc, seq_len),
             "patch_in": 2 * dl * dg,
             "global": (2 * g_mat + self._mixer_terms(g_kinds, gc, N)) / P,
             "up": 2 * dg * dl * (N + 1) / seq_len}
        par = cfg.decoder == "parallel_refine"
        dec_ctx = P if par else seq_len
        per_pass = 2 * (n(self.passes[0]) + 2 * dl * dl) + self._mixer_terms(cfg.refine_blocks, lc, dec_ctx)
        f["decoder"] = r * per_pass + (2 * (r - 1) * (2 * dl * dl + V * dl) if par else 0)
        w = self.loss_weights(r)
        if par:
            heads = r                         # every pass's guesses feed the next pass
        elif training:
            heads = len({k for k in range(r) if w[k] > 0} | {r - 1})
        else:
            heads = 1
        f["head"] = heads * 2 * dl * V
        mult = 3 if training else 1
        f = {k: v * mult for k, v in f.items()}
        f["total"] = sum(f.values())
        return f

    def flops_per_token(self, seq_len: int, loops: int | None = None, training: bool = True,
                        passes: int | None = None) -> float:
        """Per BYTE position (the trainer's 'token' is a byte here). x BYTES_PER_TOKEN for a BPE-token-equivalent."""
        return self.flops_breakdown(seq_len, loops=loops, passes=passes, training=training)["total"]

    def flops_per_token_equiv(self, seq_len: int, **kw) -> float:
        return BYTES_PER_TOKEN * self.flops_per_token(seq_len, **kw)


def make_model(cfg: ModelConfig) -> nn.Module:
    """FusionDance for token (and flat byte) models; PatchedByteLM when cfg.patch_size > 0."""
    return PatchedByteLM(cfg) if cfg.patch_size > 0 else FusionDance(cfg)


def build_model(model_cfg: dict) -> nn.Module:
    return make_model(ModelConfig(**model_cfg))
