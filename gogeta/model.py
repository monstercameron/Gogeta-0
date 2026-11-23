"""FusionDance: prelude -> looped core x r -> coda, with pluggable sequence mixers.

Block kinds
  attn  full causal softmax attention (RoPE)              O(n^2)
  swa   sliding-window causal attention (RoPE, window w)  O(n*w) in principle*
  gdn   Gated DeltaNet (fla chunk kernel)                 O(n), fixed-size state

  * implemented with a dense boolean mask via SDPA for now, so the *kernel* is still
    O(n^2) at our 1k context. Swap for a block-sparse kernel before long-context tests.

Every block is pre-norm:  x += mixer(norm(x));  x += SwiGLU(norm(x)).
With moe_experts > 0 the SwiGLU is a dropless top-k MoE of SwiGLU experts (gogeta/moe.py; off by default).

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

from gogeta.moe import MoEFFN, combine_aux


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
    # weight init (SQ234/SQ200). "gogeta": std 0.02, residual-output projections scaled by
    # 1/sqrt(2 * effective_depth). "flat": std 0.02 everywhere, no residual scaling (HF Llama's
    # initializer_range). Default "gogeta" is the pre-existing behaviour, bit-identical.
    init: str = "gogeta"
    # Code-MoE FFN (E60, gogeta/moe.py): moe_experts > 0 replaces every block's SwiGLU with a dropless top-k MoE of
    # SwiGLU experts of width moe_hidden. 0 = dense (default): bit-identical to the pre-MoE code, and to_dict()
    # omits every moe_* key so existing run configs and their hashes are unchanged (tests/test_moe.py).
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_hidden: int = 512
    moe_balance_coef: float = 0.01   # Switch load-balance loss weight (per MoE layer)
    moe_zloss_coef: float = 1e-3     # router z-loss weight (per MoE layer)
    moe_router_frozen: bool = False  # router weights requires_grad=False (frozen random router)
    moe_impl: str = "gather"         # "gather" (active compute only) | "dense" (all experts, static shapes)
    # Assembly compensations (v0.4.109, A1 campaign). Every default reproduces the pre-assembly model bit for bit,
    # and to_dict() omits a key while it holds its default, so existing configs and their hashes are unchanged.
    loop_embed: bool = False          # learned per-pass vector added to h before each core pass (zero-init)
    inject_contractive: bool = False  # h = lam * h + inject(...), lam = sigmoid(per-channel param), init 0.9 (Parcae-style)
    ternary_acts: bool = True         # ternary=True: also int8 absmax activations (BitNet b1.58); False = weights only
    ternary_fp_ends: bool = False     # ternary=True: prelude, coda and inject stay full precision (only the core is ternary)
    loop_adapter_rank: int = 0        # > 0: per-pass low-rank adapter after each core block (SQ357), passes >= loops_max share the last

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.moe_experts:
            d = {k: v for k, v in d.items() if not k.startswith("moe_")}
        for k, v in _ASSEMBLY_DEFAULTS.items():
            if d.get(k) == v:
                d.pop(k)
        return d


_ASSEMBLY_DEFAULTS = {"loop_embed": False, "inject_contractive": False, "ternary_acts": True, "ternary_fp_ends": False,
                      "loop_adapter_rank": 0}


# ----------------------------------------------------------------------------- BitLinear
class BitLinear(nn.Linear):
    """BitNet b1.58 linear: ternary weights (absmean) + int8 activations (absmax), STE.

    quant_acts=False keeps activations in full precision (weight-only ternary). quant_on=False makes the layer a plain
    linear (used by train.ternary_start_frac: full-precision training first, ternary QAT from that fraction on)."""
    quant_acts: bool = True
    quant_on: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_on:
            return F.linear(x, self.weight, self.bias)
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        w_q = (w / scale).round().clamp(-1, 1) * scale
        w = w + (w_q - w).detach()
        if self.quant_acts:
            s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            x_q = (x * s).round().clamp(-128, 127) / s
            x = x + (x_q - x).detach()
        return F.linear(x, w, self.bias)


def make_linear(cfg: ModelConfig, d_in: int, d_out: int) -> nn.Linear:
    if not cfg.ternary:
        return nn.Linear(d_in, d_out, bias=False)
    lin = BitLinear(d_in, d_out, bias=False)
    if not cfg.ternary_acts:
        lin.quant_acts = False
    return lin


def set_ternary_quant(model: nn.Module, on: bool) -> int:
    """Switch every BitLinear's quantisation on or off; returns how many layers changed."""
    n = 0
    for m in model.modules():
        if isinstance(m, BitLinear) and m.quant_on != on:
            m.quant_on = on
            n += 1
    return n


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
        self.moe = bool(cfg.moe_experts)
        self.mlp = MoEFFN(cfg) if self.moe else SwiGLU(cfg)

    def forward(self, x, aux: list | None = None):
        x = x + self.mixer(self.norm1(x))
        if self.moe:
            return x + self.mlp(self.norm2(x), aux)
        return x + self.mlp(self.norm2(x))


# ----------------------------------------------------------------------------- model
class FusionDance(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.moe_experts and cfg.ternary:
            raise ValueError("moe_experts > 0 with ternary=True is not supported (no BitLinear for batched experts)")
        self.cfg = cfg
        self.rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        ends_cfg = replace(cfg, ternary=False) if (cfg.ternary and cfg.ternary_fp_ends) else cfg
        mk = lambda kinds, c: nn.ModuleList(Block(k, c, self.rope) for k in kinds)
        self.prelude, self.core, self.coda = mk(cfg.prelude, ends_cfg), mk(cfg.core, cfg), mk(cfg.coda, ends_cfg)
        for i, b in enumerate([*self.prelude, *self.core, *self.coda]):
            if b.moe:
                b.mlp.layer_idx = i
        if self.core:
            self.inj_norm_h, self.inj_norm_e = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
            self.inject = make_linear(ends_cfg, 2 * cfg.d_model, cfg.d_model)
            self.inject.is_residual_out = True
            if cfg.loop_embed:  # one vector per pass index (passes beyond loops_max reuse the last), zero-init below
                self.loop_emb = nn.Parameter(torch.zeros(max(cfg.loops_max, cfg.loops), cfg.d_model))
            if cfg.inject_contractive:  # lam = sigmoid(lam_logit) in (0, 1), init 0.9
                self.lam_logit = nn.Parameter(torch.full((cfg.d_model,), math.log(0.9 / 0.1)))
            if cfg.loop_adapter_rank:  # per-pass, per-core-block low-rank adapters (SQ357): down N(0, 0.02), up zero-init
                P, J, d, rk = cfg.loops_max, len(cfg.core), cfg.d_model, cfg.loop_adapter_rank
                self.adapt_norm = nn.RMSNorm(d)
                self.adapt_down = nn.Parameter(torch.zeros(P, J, d, rk))  # N(0, 0.02) drawn LAST in _init_weights
                self.adapt_up = nn.Parameter(torch.zeros(P, J, rk, d))
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
        if self.cfg.init not in ("gogeta", "flat"):
            raise ValueError(f"unknown init {self.cfg.init!r} (gogeta | flat)")
        std_res = 0.02 if self.cfg.init == "flat" else 0.02 / math.sqrt(2 * max(1, self.effective_depth()))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std_res if getattr(m, "is_residual_out", False) else 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, MoEFFN):  # batched experts are not nn.Linear: up std 0.02, down std_res (as SwiGLU)
                m.reset_parameters(0.02, std_res)
        if getattr(self, "adapt_down", None) is not None:  # drawn after every other init, so they stay bit-identical
            nn.init.normal_(self.adapt_down, std=0.02)

    # ------------------------------------------------------------------ forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None,
                return_aux: bool = False):
        """(logits, loss) with loss = LM cross-entropy only. With MoE on and return_aux=True, returns
        (logits, loss, aux) where aux["loss"] is the weighted MoE auxiliary loss that the trainer adds to the LM
        loss (gogeta.train.forward_backward), plus detached routing diagnostics (gogeta.moe.combine_aux)."""
        aux = [] if self.cfg.moe_experts else None
        x = self.embed(idx)
        for b in self.prelude:
            x = b(x, aux)
        if self.core:
            r = loops if loops is not None else self.cfg.loops
            n_nograd = max(0, r - self.cfg.bptt_loops) if self.training else 0
            e, h = x, x
            for i in range(r):
                with torch.no_grad() if i < n_nograd else nullcontext():
                    inj = self.inject(torch.cat([self.inj_norm_h(h), self.inj_norm_e(e)], dim=-1))
                    if self.cfg.inject_contractive:
                        h = torch.sigmoid(self.lam_logit) * h + inj
                    else:
                        h = h + inj
                    if self.cfg.loop_embed:
                        h = h + self.loop_emb[min(i, self.loop_emb.shape[0] - 1)]
                    p = min(i, self.cfg.loops_max - 1)
                    for j, b in enumerate(self.core):
                        h = b(h, aux)
                        if self.cfg.loop_adapter_rank:  # per-pass low-rank residual after each core block (zero-init up)
                            h = h + (self.adapt_norm(h) @ self.adapt_down[p, j]) @ self.adapt_up[p, j]
            x = h
        for b in self.coda:
            x = b(x, aux)
        logits = self.head(self.norm_out(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_aux:
            return logits, loss, (self.moe_aux(aux) if aux else None)
        return logits, loss

    def moe_aux(self, calls: list) -> dict:
        return combine_aux(calls, self.cfg.moe_balance_coef, self.cfg.moe_zloss_coef)

    # ------------------------------------------------------------------ accounting
    def param_counts(self) -> dict:
        def n(mods):
            return sum(p.numel() for m in mods for p in m.parameters())

        emb = self.embed.weight.numel()
        pre, core, coda = n(self.prelude), n(self.core), n(self.coda)
        other = sum(p.numel() for p in self.parameters()) - emb - pre - core - coda
        d = {"total": sum(p.numel() for p in self.parameters()), "embedding": emb,
             "prelude": pre, "core": core, "coda": coda, "other": other,
             "non_embedding": pre + core + coda + other}
        if self.cfg.moe_experts:  # MoE: total counts every expert; active counts top_k experts (+ router) per token
            inact = {s: sum(b.mlp.inactive_params() for b in getattr(self, s) if b.moe) for s in ("prelude", "core", "coda")}
            n_in = sum(inact.values())
            d.update({"moe_inactive": n_in, "active_total": d["total"] - n_in,
                      "active_non_embedding": d["non_embedding"] - n_in,
                      **{f"active_{s}": d[s] - v for s, v in inact.items()}})
        return d

    def flops_per_token(self, seq_len: int, loops: int | None = None, training: bool = True) -> float:
        """Approximate FLOPs per token (6N rule for matmuls + attention/GDN terms).

        Matmul params counted once per *application* (core counted r times). Attention adds
        4*d*ctx per layer per token forward (QK^T and AV), ctx = mean attended length.
        GDN adds ~4*H*hd^2 per layer per token forward for state read/write. Training = 3x forward.
        MoE: only *active* params count (top_k experts + router per token), whatever moe_impl computes.
        """
        cfg = self.cfg
        r = (loops or cfg.loops) if self.core else 0
        pc = self.param_counts()
        inj = 2 * cfg.d_model * cfg.d_model if self.core else 0
        pre, core, coda = (pc.get(f"active_{s}", pc[s]) for s in ("prelude", "core", "coda"))
        adapt = 2 * cfg.d_model * cfg.loop_adapter_rank * len(cfg.core) if (self.core and cfg.loop_adapter_rank) else 0
        matmul_params = pre + coda + r * (core + inj + adapt) + cfg.d_model * cfg.vocab_size
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
