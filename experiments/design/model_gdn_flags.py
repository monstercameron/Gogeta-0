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
from dataclasses import asdict, dataclass, field

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
    gdn_conv: bool = True         # GDN short causal convs on q/k/v (False: identity)   [E1 decomposition]
    gdn_gate: bool = True         # GDN SiLU output gate (False: no gate projection)    [E1 decomposition]
    gdn_out_norm: bool = True     # GDN per-head output RMSNorm (False: identity)       [E1 decomposition]
    ternary: bool = False
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


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
        self.gate = make_linear(cfg, D, H * hd) if cfg.gdn_gate else None
        self.o = make_linear(cfg, H * hd, D)
        self.o.is_residual_out = True
        self.a = nn.Linear(D, H, bias=False)       # decay input (full precision)
        self.b = nn.Linear(D, H, bias=False)       # write strength beta (full precision)
        self.conv_q, self.conv_k, self.conv_v = (
            ShortConv(H * hd, cfg.conv_size) if cfg.gdn_conv else nn.Identity() for _ in range(3))
        # Mamba-2 style decay: g = -exp(A_log) * softplus(a(x) + dt_bias)
        A = torch.empty(H).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.out_norm = nn.RMSNorm(hd) if cfg.gdn_out_norm else nn.Identity()
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
        o = self.out_norm(o.to(x.dtype))
        if self.gate is not None:
            o = o * F.silu(self.gate(x).view(B, T, H, hd))
        return self.o(o.reshape(B, T, H * hd))


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


def build_model(model_cfg: dict) -> FusionDance:
    return FusionDance(ModelConfig(**model_cfg))
