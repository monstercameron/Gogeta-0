"""Coarse-to-fine byte model (C2F) around the assembled FusionDance backbone. Two patching modes:

patching = "tokens" (tokens as patches; SQ207 / X1 design, notes/analysis-coarse-to-fine.md section 6):
    BPE tokens -> GLOBAL model = gogeta.model.FusionDance on tokens (its own token embedding; return_hidden hook)
               -> H[i] (has seen tokens <= i) -> up-projection to d_local
               -> byte decoder SPELLS target token i+1's bytes from H[i]: rows [BOT, b_1 .. b_L] predict
                  [b_1 .. b_L, EOT] (variable-length patches = the token's byte span; EOT ends the token)
               -> byte refinement loop (r_b passes, shared weights by default) -> 257-way head (256 bytes + EOT)
    The rows of all target tokens of a sequence are concatenated into one causal byte stream, so local mixers see
    earlier tokens' bytes too. Loss = nats per TOKEN (sum of byte + EOT cross-entropies / number of target tokens): the
    same unit as a BPE model's val_loss on the same val tokens, so bits/char = loss / ln 2 / chars_per_token compares
    directly with E1/A2s/A3 (it scores text jointly with its canonical tokenization, exactly as the BPE model does).
    Optional frozen trunk (freeze_global + load_global_checkpoint): only the byte decoder trains (the cheap X1 test).

patching = "fixed" (BLT-lite / MegaByte, fixed P-byte patches):
    raw bytes -> byte embedding -> causal local encoder (d_local) -> P-byte patches (concat, linear to d_global)
              -> FusionDance on patches (inputs_embeds hook) -> one-patch causal shift -> patch-to-byte expansion
              (+ within-patch slot embedding) -> byte refinement loop -> 256-way byte head.
    SQ374: fixed patches were far too costly at our scale (E13b P=4 +63% bpc vs E1 at ~equal FLOPs), so "tokens" is
    the default a4 design; "fixed" stays as the alternative.

Design notes: notes/analysis-coarse-to-fine.md. Devlog: experiments/devlogs/C2F.md. Tests: tests/test_c2f.py.
Uses the real gogeta.model.FusionDance as the global trunk, so every backbone option (GDN/SWA/attention, shared loop,
ternary, MoE) works unchanged. The older experiments/byte_model/model_bytes.py ran on a forked model copy.

Causality, fixed (row t predicts byte t+1 and may use bytes <= t):
  * patch p = bytes [pP, (p+1)P); G[p] has seen bytes <= (p+1)P - 1 (causal global mixers);
  * row t reads Gs[(t+1)//P], Gs = [start, G[0..N-1]]: the last patch COMPLETE before byte t+1; slot (t+1) % P;
  * the byte stream adds the embedding of byte t; every local mixer is causal at byte level.
Causality, tokens (input tokens x[0..N-1], targets y[i] = x[i+1]; span i spells y[i]):
  * span i's rows read H[i] (tokens <= i) plus, through causal local mixers, bytes of spans < i (y[<i] = x[1..i]) and
    the earlier bytes of y[i] itself: never a byte of y[i] at or after the row's target;
  * changing token x[s] (= y[s-1]) may change H[>= s], the rows of span s-1 from slot 1 on (their inputs), and all
    later spans; every earlier row and span s-1's slot-0 (BOT) row stay bitwise unchanged (tests check this).
Refinement loop (both modes; r_b passes, pass 0 is the plain decoder):
  x0 = ctx + e (ctx = expanded global context + slot embedding, e = embedding of the row's input byte)
  pass 0: h = blocks_0(x0);  pass k >= 1: h = blocks_k(lam * h + inject([norm h, norm e, norm ctx]))
  (lam = sigmoid(lam_logit), init 0.9, with refine_contractive, else 1); logits_k = head(norm_out(h)).
  r_b = 1 builds no inject: a plain single-pass decoder. blocks_k shared (H-C2F) or separate (unshared control).
  Train objective = CE(last pass) + refine_aux_weight * mean CE(earlier passes). Eval loss = last pass.
Precision is explicit per part: the global model follows its own config; the byte encoder/decoder are full precision
unless encoder_ternary / decoder_ternary. Residual-output init uses each stream's own depth (encoder:
len(local_encoder); decoder: r_b * len(refine_blocks)); the global model keeps FusionDance's own init.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import math
import os
import pathlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from gogeta.model import Attention, Block, FusionDance, ModelConfig, Rotary, ShortConv, SwiGLU, make_linear

BYTES_PER_TOKEN = 3.735   # measured val bytes per BPE token (notes/research-byte-level.md)
REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = "gogeta/assets/tokenizer-code-py-v1.json"   # code-py-v1 and code-py-v2 share it (build_v2.py)
LOCAL_KINDS = ("conv", "gdn", "lswa", "swa", "attn")


# ----------------------------------------------------------------------------- config
@dataclass
class C2FConfig:
    patching: str = "fixed"          # "tokens" (BPE tokens as variable-length patches) | "fixed" (P-byte patches)
    vocab_size: int = 256            # bytes; "tokens" adds one id (256) used as BOT input and EOT target
    max_seq_len: int = 4096          # positions of the training sequence: BYTES (fixed) or TOKENS (tokens)
    patch_size: int = 4              # P (fixed only)
    d_local: int = 192
    local_heads: int = 3             # head_dim = d_local / local_heads (even, for RoPE)
    local_mlp_hidden: int = 512
    local_window: int = 64           # bytes, for "lswa" / "swa" local blocks
    local_encoder: list = field(default_factory=lambda: ["conv", "gdn"])   # fixed only (tokens: must be [])
    refine_blocks: list = field(default_factory=lambda: ["lswa"])          # one decoder pass (block-local SWA + SwiGLU)
    refine_passes: int = 2           # r_b
    refine_shared: bool = True       # False: r_b passes with separate weights (unshared control)
    refine_mlp_hidden: int = 0       # 0 = local_mlp_hidden; smaller for the parameter-matched unshared control (A'')
    refine_contractive: bool = False  # pass k >= 1: h = lam * h + inject(...), lam = sigmoid(per-channel), init 0.9
    refine_aux_weight: float = 0.0   # weight of the mean CE of passes 0..r_b-2 added to the last pass's CE (training)
    encoder_ternary: bool = False    # byte encoder projections BitLinear (default full precision, K4)
    decoder_ternary: bool = False    # byte decoder projections + inject BitLinear (default full precision, K4)
    conv_size: int = 4
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    global_model: dict = field(default_factory=dict)   # gogeta.model.ModelConfig fields (tokens: incl. vocab_size)
    # tokens mode only
    tokenizer: str = DEFAULT_TOKENIZER   # tokenizer.json (repo-relative or absolute): token id -> byte string
    freeze_global: bool = False      # global model frozen (eval mode, no grad): only the byte side trains (X1)
    global_init: str = ""            # checkpoint (FusionDance model.pt) for the global model; loaded by the trainer
    rows_per_token: float = BYTES_PER_TOKEN + 1   # decoder rows per token (bytes + EOT), FLOP accounting only

    def __post_init__(self):
        if self.patching not in ("fixed", "tokens"):
            raise ValueError(f"unknown patching {self.patching!r} (fixed | tokens)")
        if self.patching == "fixed" and self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.patching == "tokens" and self.local_encoder:
            raise ValueError("patching 'tokens' has no byte encoder (the global model embeds tokens): local_encoder=[]")
        if self.patching == "fixed" and (self.freeze_global or self.global_init):
            raise ValueError("freeze_global / global_init are for patching 'tokens' (a token trunk)")
        if self.refine_passes < 1:
            raise ValueError("refine_passes must be >= 1")
        if self.d_local % self.local_heads or (self.d_local // self.local_heads) % 2:
            raise ValueError("d_local / local_heads must be an even integer (RoPE)")
        for k in [*self.local_encoder, *self.refine_blocks]:
            if k not in LOCAL_KINDS:
                raise ValueError(f"unknown local block kind {k!r} ({' | '.join(LOCAL_KINDS)})")

    @property
    def byte_vocab(self) -> int:
        return self.vocab_size + (1 if self.patching == "tokens" else 0)

    @property
    def n_patches_max(self) -> int:
        return -(-self.max_seq_len // self.patch_size)

    def global_config(self) -> ModelConfig:
        g = dict(self.global_model)
        g.pop("max_seq_len", None)
        if self.patching == "fixed":
            g.pop("vocab_size", None)
            # vocab 1: FusionDance's token embedding / head are never used (inputs_embeds + return_hidden)
            return ModelConfig(**g, vocab_size=1, max_seq_len=self.n_patches_max + 1)
        if "vocab_size" not in g:
            raise ValueError("patching 'tokens' needs global_model.vocab_size (the token vocabulary)")
        return ModelConfig(**g, max_seq_len=self.max_seq_len)

    def local_config(self, mlp_hidden: int | None = None, ternary: bool = False, rows: int | None = None) -> ModelConfig:
        """rows: longest byte-row sequence (RoPE table); default max_seq_len + P (fixed patches)."""
        rows = rows or self.max_seq_len + self.patch_size
        return ModelConfig(vocab_size=self.byte_vocab, d_model=self.d_local, n_heads=self.local_heads,
                           mlp_hidden=mlp_hidden or self.local_mlp_hidden, window=self.local_window,
                           max_seq_len=rows, conv_size=self.conv_size,
                           rope_base=self.rope_base, ternary=ternary, prelude=[], core=[], coda=[])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arch"] = "c2f"
        d["global_model"] = self.global_config().to_dict()   # the effective global config (incl. defaults)
        return d


# ----------------------------------------------------------------------------- token -> bytes
def _bytes_to_unicode() -> dict:
    """GPT-2 / HF ByteLevel byte <-> printable-unicode map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_byte_table(tokenizer_path: str) -> list[bytes]:
    """Byte string of every token id of a ByteLevel BPE tokenizer.json. Special tokens (<|eos|>, <|pad|>) map to
    b"\\x00", the EOS byte of the byte datasets (build_bytes.py), so a document end costs one byte, as there."""
    p = pathlib.Path(tokenizer_path)
    p = p if p.is_absolute() else REPO / p
    tj = json.loads(p.read_text(encoding="utf-8"))
    if tj["model"]["type"] != "BPE" or (tj.get("pre_tokenizer") or {}).get("type") != "ByteLevel":
        raise ValueError(f"{p}: need a ByteLevel BPE tokenizer")
    inv = {c: b for b, c in _bytes_to_unicode().items()}
    special = {t["id"] for t in tj.get("added_tokens", []) if t.get("special", True)}
    vocab = tj["model"]["vocab"]
    V = max(max(vocab.values()), *special) + 1
    table = [b"\x00"] * V
    for s, i in vocab.items():
        if i not in special:
            table[i] = bytes(inv[c] for c in s)
    if any(len(b) == 0 for b in table):
        raise ValueError("empty token byte string")
    return table


def load_global_checkpoint(path: str) -> tuple[dict, dict, dict]:
    """(state_dict, model_config, info) of a gogeta.train FusionDance checkpoint. `path` may use ~, $VARS and one glob
    (the sorted first match is used and recorded). info = {path, sha256, dataset, train} for the run config."""
    cands = sorted(_glob.glob(os.path.expandvars(os.path.expanduser(path))))
    if not cands:
        raise FileNotFoundError(f"no checkpoint matches {path!r}")
    p = cands[0]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    info = {"path": p, "sha256": h.hexdigest(), "n_matches": len(cands), "dataset": ck.get("train", {}).get("dataset"),
            "train": ck.get("train", {})}
    return ck["model"], ck["model_config"], info


# ----------------------------------------------------------------------------- local blocks
class ConvBlock(nn.Module):
    """Pre-norm causal depthwise conv residual: x + SiLU(conv(norm(x)))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model)
        self.conv = ShortConv(cfg.d_model, cfg.conv_size)

    def forward(self, x, aux=None):
        return x + self.conv(self.norm(x))


class BlockLocalAttention(Attention):
    """Exact sliding-window causal attention computed block-locally (queries in block b of width w attend to blocks
    b-1 and b under the window mask): O(T * 2w) memory instead of the dense O(T^2) mask of gogeta.model's "swa"
    (C12). Same projections and RoPE as Attention. Same algorithm as model_bytes.BlockLocalAttention."""

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


class LocalSWABlock(nn.Module):
    """Pre-norm block: x += block-local SWA(norm(x)); x += SwiGLU(norm(x)) (block kind "lswa")."""

    def __init__(self, cfg: ModelConfig, rope: Rotary):
        super().__init__()
        self.kind = "lswa"
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.mixer = BlockLocalAttention(cfg, cfg.window, rope)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, aux=None):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def local_blocks(kinds, cfg: ModelConfig, rope: Rotary) -> nn.ModuleList:
    def one(k):
        if k == "conv":
            return ConvBlock(cfg)
        if k == "lswa":
            return LocalSWABlock(cfg, rope)
        return Block(k, cfg, rope)   # gogeta.model's gdn / swa (dense mask) / attn block
    return nn.ModuleList(one(k) for k in kinds)


def mixer_flops(kinds, cfg: ModelConfig, ctx: float) -> float:
    """Forward FLOPs per position beyond the matmuls, FusionDance.flops_per_token's rule."""
    f = 0.0
    for k in kinds:
        if k == "attn":
            f += 4 * cfg.d_model * (ctx / 2)
        elif k in ("swa", "lswa"):
            f += 4 * cfg.d_model * min(cfg.window, ctx / 2)
        elif k == "gdn":
            f += 4 * cfg.n_heads * cfg.head_dim ** 2
    return f


def _n(*mods) -> int:
    return sum(p.numel() for m in mods for p in m.parameters())


# ----------------------------------------------------------------------------- shared byte side
class _C2FBase(nn.Module):
    """Byte embedding, global trunk, expansion, refinement loop, head, init, losses (both patching modes)."""

    def _build(self, cfg: C2FConfig, n_slots: int, rows: int | None = None):
        self.cfg = cfg
        dl, V = cfg.d_local, cfg.byte_vocab
        self.lcfg = cfg.local_config(ternary=cfg.encoder_ternary, rows=rows)
        self.rcfg = cfg.local_config(cfg.refine_mlp_hidden or None, ternary=cfg.decoder_ternary, rows=rows)
        self.gcfg = cfg.global_config()
        self.lrope = Rotary(self.lcfg.head_dim, self.lcfg.max_seq_len, cfg.rope_base)
        self.embed = nn.Embedding(V, dl)
        self.encoder = local_blocks(cfg.local_encoder, self.lcfg, self.lrope)
        self.glob = FusionDance(self.gcfg)          # the assembled backbone (initialises itself)
        self.up = nn.Linear(self.gcfg.d_model, dl, bias=False)
        self.slot_emb = nn.Embedding(n_slots, dl)
        r = cfg.refine_passes
        self.passes = nn.ModuleList(local_blocks(cfg.refine_blocks, self.rcfg, self.lrope)
                                    for _ in range(1 if cfg.refine_shared else r))
        if r > 1:
            self.inj_norm_h, self.inj_norm_e, self.inj_norm_c = (nn.RMSNorm(dl) for _ in range(3))
            self.inject = make_linear(self.rcfg, 3 * dl, dl)
            self.inject.is_residual_out = True
            if cfg.refine_contractive:
                self.lam_logit = nn.Parameter(torch.full((dl,), math.log(0.9 / 0.1)))
        self.norm_out = nn.RMSNorm(dl)
        self.head = nn.Linear(dl, V, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self.last_aux: dict = {}

    @property
    def core(self):   # the trainer's "looped?" test reads model.core
        return self.glob.core

    # Residual-output init uses each residual stream's OWN effective depth (outside review 2026-11-23 13:10, I3):
    #   encoder stream (fixed only): embed -> encoder blocks       depth = len(local_encoder)
    #   decoder stream: x0 = ctx + e -> r_b passes of refine_blocks depth = r_b * len(refine_blocks)
    # The encoder output reaches the decoder only through the global model, so the streams share no residual.
    # Shared r_b = 2 and the unshared 2-pass controls get the same scale; the global model keeps its own.
    def encoder_depth(self) -> int:
        return len(self.cfg.local_encoder)

    def decoder_depth(self) -> int:
        return self.cfg.refine_passes * len(self.cfg.refine_blocks)

    @staticmethod
    def std_res(depth: int) -> float:
        return 0.02 / math.sqrt(2 * max(1, depth))

    def _init_local(self):
        """std 0.02 everywhere; residual-output projections 0.02 / sqrt(2 * depth of their own stream)."""
        glob = {id(m) for m in self.glob.modules()}
        enc = {id(m) for m in self.encoder.modules()}
        s_enc, s_dec = self.std_res(self.encoder_depth()), self.std_res(self.decoder_depth())
        for m in self.modules():
            if id(m) in glob:
                continue
            if isinstance(m, nn.Linear):
                res = s_enc if id(m) in enc else s_dec
                nn.init.normal_(m.weight, std=res if getattr(m, "is_residual_out", False) else 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _pass(self, k: int) -> nn.ModuleList:
        return self.passes[0 if self.cfg.refine_shared else k]

    def _check_passes(self, passes):
        cfg = self.cfg
        r = cfg.refine_passes if passes is None else passes
        if r != cfg.refine_passes and not cfg.refine_shared:
            raise ValueError("an unshared decoder runs exactly refine_passes passes")
        if r > 1 and cfg.refine_passes == 1:
            raise ValueError("refine_passes = 1 builds no inject: cannot run more passes")
        return r

    def _refine(self, e, ctx, r: int, want: set) -> list:
        """The refinement loop; returns [(k, logits_k)] for the passes in `want`."""
        outs, h = [], ctx + e
        for k in range(r):
            if k > 0:
                inj = self.inject(torch.cat([self.inj_norm_h(h), self.inj_norm_e(e), self.inj_norm_c(ctx)], dim=-1))
                h = (torch.sigmoid(self.lam_logit).to(h.dtype) * h + inj) if self.cfg.refine_contractive else h + inj
            for b in self._pass(k):
                h = b(h)
            if k in want:
                outs.append((k, self.head(self.norm_out(h))))
        return outs

    def _want(self, r: int, targets, return_all: bool) -> set:
        if return_all or (self.training and targets is not None and self.cfg.refine_aux_weight > 0):
            return set(range(r))
        return {r - 1}

    def _objective(self, ce_by_pass: dict, r: int, norm: float):
        """ce_by_pass[k]: summed CE of pass k; norm: divisor (rows, or target tokens). Returns (loss, last)."""
        last = ce_by_pass[r - 1] / norm
        loss = last
        if self.training and self.cfg.refine_aux_weight > 0 and r > 1:
            loss = loss + self.cfg.refine_aux_weight * torch.stack([ce_by_pass[k] / norm for k in range(r - 1)]).mean()
        return loss, last

    def _global_head_flops(self) -> float:
        return 2 * self.gcfg.d_model * self.gcfg.vocab_size   # FusionDance counts its head; we never apply it

    def _global_train_flops(self, ctx: int, loops, frozen: bool, training: bool) -> float:
        """Global FLOPs per global position (patch / token) with FusionDance's rule, without its head."""
        rg = (loops or self.gcfg.loops) if self.glob.core else None
        fwd = lambda r: self.glob.flops_per_token(ctx, loops=r, training=False) - self._global_head_flops()  # noqa
        if not training or frozen:
            return fwd(rg)
        if rg is not None and rg > self.gcfg.bptt_loops:   # no-grad prefix: forward only (as gogeta.train does)
            return fwd(rg) + 2 * fwd(self.gcfg.bptt_loops)
        return 3 * fwd(rg)

    def _decoder_row_flops(self, r: int, ctx_rows: float, training: bool) -> tuple[float, float]:
        """(decoder, head) forward FLOPs per decoder row."""
        dl, V = self.cfg.d_local, self.cfg.byte_vocab
        per_pass = 2 * _n(self.passes[0]) + mixer_flops(self.cfg.refine_blocks, self.rcfg, ctx_rows)
        heads = r if (training and self.cfg.refine_aux_weight > 0 and r > 1) else 1
        return r * per_pass + (r - 1) * 2 * 3 * dl * dl, heads * 2 * dl * V


# ----------------------------------------------------------------------------- fixed P-byte patches
class C2FModel(_C2FBase):
    """patching = "fixed": bytes [B,T] -> logits [B,T,256] (row t predicts byte t+1)."""

    def __init__(self, cfg: C2FConfig):
        super().__init__()
        if cfg.patching != "fixed":
            raise ValueError("C2FModel is the fixed-patch model; use build_c2f / C2FTokenModel for 'tokens'")
        self.P = cfg.patch_size
        self._build(cfg, n_slots=cfg.patch_size)
        dg = self.gcfg.d_model
        self.patch_in = nn.Linear(cfg.patch_size * cfg.d_local, dg, bias=False)
        self.patch_start = nn.Parameter(torch.zeros(dg))     # Gs[0]: context for the first patch
        self._broken_noshift = False   # TESTS ONLY: deliberately broken causality control; never set in training
        self._init_local()
        nn.init.normal_(self.patch_start, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None,
                passes: int | None = None, return_aux: bool = False, return_all: bool = False):
        """idx [B,T] bytes -> (logits [B,T,V], loss) or with return_aux (logits, loss, moe_aux|None).
        loops: global core loops. passes: refinement passes (another value than refine_passes only with shared
        weights). return_all: list of every pass's logits. loss: training objective in train mode, last-pass mean
        CE per byte in eval mode."""
        cfg, P = self.cfg, self.P
        B, T0 = idx.shape
        pad = (-T0) % P
        if pad:  # e.g. generation: right padding only (causal), sliced off below
            idx = F.pad(idx, (0, pad))
        T = idx.shape[1]
        N = T // P
        r = self._check_passes(passes)
        e = self.embed(idx)
        h = e
        for b in self.encoder:
            h = b(h)
        pin = self.patch_in(h.reshape(B, N, P * cfg.d_local))
        g, aux = self.glob(None, loops=loops, inputs_embeds=pin, return_hidden=True, return_aux=True)
        gs = torch.cat([self.patch_start.to(g.dtype).expand(B, 1, -1), g], dim=1)   # one-patch shift
        t = torch.arange(T, device=idx.device)
        gi, slot = (t + 1) // P, (t + 1) % P
        if self._broken_noshift:   # row t reads its own patch's state (sees bytes > t): must FAIL causality
            gi = t // P + 1
        ctx = self.up(gs)[:, gi] + self.slot_emb(slot)
        outs = [(k, lg[:, :T0]) for k, lg in self._refine(e, ctx, r, self._want(r, targets, return_all))]
        logits = outs[-1][1]
        loss = None
        self.last_aux = {}
        if targets is not None:
            V = logits.size(-1)
            ce = {k: F.cross_entropy(lg.float().reshape(-1, V), targets.reshape(-1), reduction="none").view(B, -1)
                  for k, lg in outs}
            s = slot[:T0]
            loss, last = self._objective({k: v.sum() for k, v in ce.items()}, r, float(ce[r - 1].numel()))
            self.last_aux = {
                "ce_last": last.detach(), "pass_ids": sorted(ce), "pass_loss": [ce[k].mean().detach() for k in sorted(ce)],
                "slot_loss": torch.zeros(P, device=s.device).index_add_(0, s, ce[r - 1].detach().float().sum(0))
                / (B * torch.bincount(s, minlength=P).clamp(min=1))}
            if not self.training:
                loss = last
        res = [lg for _, lg in outs] if return_all else logits
        return (res, loss, aux) if return_aux else (res, loss)

    # ------------------------------------------------------------------ accounting
    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())     # tied head counted once
        emb = self.embed.weight.numel()
        gpc = self.glob.param_counts()
        unused = self.glob.embed.weight.numel()                # FusionDance's vocab-1 embedding/head (never used)
        enc = _n(self.encoder)
        patch_io = _n(self.patch_in, self.up, self.slot_emb) + self.patch_start.numel()
        dec = total - emb - enc - patch_io - gpc["total"] - _n(self.norm_out)
        d = {"total": total, "embedding": emb, "encoder": enc, "patch_io": patch_io, "global": gpc["total"] - unused,
             "global_core": gpc["core"], "global_unused_embed": unused, "decoder": dec,
             "decoder_blocks": _n(self.passes), "decoder_block_per_pass": _n(self.passes[0]),
             "head_norm": _n(self.norm_out), "local": total - gpc["total"], "non_embedding": total - emb}
        if "moe_inactive" in gpc:
            d.update({"global_moe_inactive": gpc["moe_inactive"], "global_active": d["global"] - gpc["moe_inactive"],
                      "active_total": total - gpc["moe_inactive"],
                      "active_non_embedding": total - emb - gpc["moe_inactive"]})
        return d

    def _global_fwd_per_patch(self, n_patches: int, loops: int | None) -> float:
        return self.glob.flops_per_token(n_patches, loops=loops, training=False) - self._global_head_flops()

    def flops_breakdown(self, seq_len: int, loops: int | None = None, passes: int | None = None,
                        training: bool = True) -> dict:
        """FLOPs per BYTE position, by part (FusionDance's rule: 2 x params per application, attention/GDN terms,
        training = 3 x forward; global once per patch, so / P per byte; global loops beyond bptt_loops count the
        no-grad prefix forward only; one head per loss-carrying pass in training)."""
        cfg, P = self.cfg, self.P
        N = -(-seq_len // P)
        r = cfg.refine_passes if passes is None else passes
        dl, dg = cfg.d_local, self.gcfg.d_model
        mult = 3 if training else 1
        dec, head = self._decoder_row_flops(r, seq_len, training)
        f = {"encoder": mult * (2 * _n(self.encoder) + mixer_flops(cfg.local_encoder, self.lcfg, seq_len)),
             "patch_in": mult * 2 * dl * dg,
             "global": self._global_train_flops(N, loops, False, training) / P,
             "up": mult * 2 * dg * dl * (N + 1) / seq_len,
             "decoder": mult * dec, "head": mult * head}
        f["total"] = sum(f.values())
        return f

    def flops_per_byte(self, seq_len: int, loops: int | None = None, passes: int | None = None,
                       training: bool = True) -> float:
        return self.flops_breakdown(seq_len, loops=loops, passes=passes, training=training)["total"]

    def flops_per_token(self, seq_len: int, loops: int | None = None, training: bool = True,
                        passes: int | None = None) -> float:
        """Trainer-compatible: per POSITION of the training sequence (a byte here)."""
        return self.flops_per_byte(seq_len, loops=loops, passes=passes, training=training)

    def flops_per_token_equiv(self, seq_len: int, **kw) -> float:
        return BYTES_PER_TOKEN * self.flops_per_byte(seq_len, **kw)


# ----------------------------------------------------------------------------- tokens as patches
class C2FTokenModel(_C2FBase):
    """patching = "tokens": tokens [B,N] + target tokens [B,N] -> per-row byte logits [B,R,257] over the flattened
    spelling of the targets (R = max over the batch of sum_i (len(y_i) + 1)), loss in nats per target token."""

    BOT = EOT = 256

    def __init__(self, cfg: C2FConfig, table: list[bytes] | None = None):
        super().__init__()
        if cfg.patching != "tokens":
            raise ValueError("C2FTokenModel needs patching = 'tokens'")
        table = table if table is not None else token_byte_table(cfg.tokenizer)
        V_tok = cfg.global_config().vocab_size
        if len(table) != V_tok:
            raise ValueError(f"tokenizer has {len(table)} ids, global vocab_size is {V_tok}")
        L = max(len(b) for b in table)
        tb = torch.zeros(V_tok, L, dtype=torch.long)
        for i, b in enumerate(table):
            tb[i, :len(b)] = torch.tensor(list(b), dtype=torch.long)
        self.register_buffer("tok_bytes", tb)                                 # persistent: saved with checkpoints
        self.register_buffer("tok_len", torch.tensor([len(b) for b in table], dtype=torch.long))
        self.max_token_bytes = L
        self._build(cfg, n_slots=L + 1, rows=cfg.max_seq_len * (L + 1))      # slot k = k-th row of a span
        self._broken_lookahead = False   # TESTS ONLY: deliberately broken causality control; never set in training
        self._init_local()
        if cfg.freeze_global:
            self.glob.requires_grad_(False)
            self.glob.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_global:
            self.glob.eval()   # frozen trunk: eval semantics always (default loops, no truncated-BPTT sampling)
        return self

    def load_global(self, state_dict: dict):
        """Load a FusionDance checkpoint's state dict into the global model (strict)."""
        self.glob.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------ the flattened byte stream
    def spans(self, targets: torch.Tensor):
        """targets [B,N] token ids -> (inp, tgt, span, slot), each [B,R]: row inputs (BOT then the token's bytes),
        row targets (the bytes then EOT; -100 on padding rows), the span (target-token index) and the slot of every
        row. Rows are in span order, so a causal mixer over rows never sees a later span or a later byte."""
        B, N = targets.shape
        L, dev = self.max_token_bytes, targets.device
        K = L + 1
        lens = self.tok_len[targets]                                          # [B,N]
        byt = self.tok_bytes[targets]                                         # [B,N,L]
        k = torch.arange(K, device=dev)
        inp = torch.cat([torch.full((B, N, 1), self.BOT, device=dev, dtype=torch.long), byt], dim=-1)[..., :K]
        tgt = torch.cat([byt, torch.zeros(B, N, 1, device=dev, dtype=torch.long)], dim=-1)
        tgt = torch.where(k == lens[..., None], torch.full_like(tgt, self.EOT), tgt)
        valid = k <= lens[..., None]                                          # [B,N,K]
        span = torch.arange(N, device=dev)[None, :, None].expand(B, N, K)
        slot = k.expand(B, N, K)
        valid, inp, tgt, span, slot = (z.reshape(B, N * K) for z in (valid, inp, tgt, span, slot))
        pos = valid.long().cumsum(1) - 1
        R = int(valid.sum(1).max().item())
        bi = torch.arange(B, device=dev)[:, None].expand_as(pos)[valid]
        pi = pos[valid]
        out = []
        for z, fill in ((inp, 0), (tgt, -100), (span, 0), (slot, 0)):
            o = torch.full((B, R), fill, dtype=torch.long, device=dev)
            o[bi, pi] = z[valid]
            out.append(o)
        return tuple(out)

    def _global_states(self, idx, loops):
        with torch.no_grad() if self.cfg.freeze_global else nullcontext():
            return self.glob(idx, loops=loops, return_hidden=True, return_aux=True)

    def _rows(self, H, inp, span, slot, r, want):
        if self._broken_lookahead:   # TESTS ONLY: span i reads H[i+1], which has seen the token being spelled
            span = (span + 1).clamp(max=H.shape[1] - 1)
        ctx =torch.gather(self.up(H), 1, span[..., None].expand(-1, -1, self.cfg.d_local)) + self.slot_emb(slot)
        return self._refine(self.embed(inp), ctx, r, want)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None,
                passes: int | None = None, return_aux: bool = False, return_all: bool = False):
        """idx [B,N] tokens, targets [B,N] next tokens (required: the decoder spells them; generation uses
        next_byte_logits). Returns (logits [B,R,257] per flattened row, loss in nats per target token), with
        return_aux also the global model's MoE aux (None when dense or frozen). Train mode: the objective (last
        pass + refine_aux_weight * mean earlier passes); eval: last pass. last_aux carries the row layout."""
        if targets is None:
            raise ValueError("C2FTokenModel.forward spells the target tokens: pass targets (generation: "
                             "next_byte_logits)")
        r = self._check_passes(passes)
        H, aux = self._global_states(idx, loops)
        if self.cfg.freeze_global:
            aux = None
        inp, tgt, span, slot = self.spans(targets)
        outs = self._rows(H, inp, span, slot, r, self._want(r, targets, return_all))
        B, R = tgt.shape
        V = outs[-1][1].size(-1)
        ce = {k: F.cross_entropy(lg.float().reshape(-1, V), tgt.reshape(-1), ignore_index=-100, reduction="none")
              .view(B, R) for k, lg in outs}
        n_tok = float(targets.numel())
        loss, last = self._objective({k: v.sum() for k, v in ce.items()}, r, n_tok)
        valid = tgt != -100
        n_rows = valid.sum()
        S = self.slot_emb.num_embeddings
        cl = ce[r - 1].detach()
        is_eot = tgt == self.EOT
        self.last_aux = {
            "ce_last": last.detach(), "pass_ids": sorted(ce), "pass_loss": [ce[k].sum().detach() / n_tok for k in sorted(ce)],
            "rows": n_rows.detach(), "tokens": n_tok,
            "nats_per_row": cl.sum() / n_rows.clamp(min=1),
            "eot_nats_per_token": cl[is_eot].sum() / n_tok,
            "slot_loss": torch.zeros(S, device=cl.device).index_add_(0, slot[valid], cl[valid].float())
            / torch.bincount(slot[valid], minlength=S).clamp(min=1),
            "layout": (tgt, span, slot)}
        if not self.training:
            loss = last
        res = [lg for _, lg in outs] if return_all else outs[-1][1]
        return (res, loss, aux) if return_aux else (res, loss)

    @torch.no_grad()
    def next_byte_logits(self, idx: torch.Tensor, prefix: list[int], loops: int | None = None) -> torch.Tensor:
        """Generation: idx [1,n] context tokens; prefix = bytes already spelled of the next token. Returns the
        257-way logits for the next byte (256 = end of token)."""
        H, _ = self._global_states(idx, loops)
        dev = idx.device
        if idx.shape[1] > 1:
            inp, _, span, slot = self.spans(idx[:, 1:])          # spans 0..n-2 spell the known tokens x[1..n-1]
        else:
            inp = span = slot = torch.zeros(1, 0, dtype=torch.long, device=dev)
        n = idx.shape[1]
        p_inp = torch.tensor([[self.BOT, *prefix]], device=dev)
        p_span = torch.full_like(p_inp, n - 1)
        p_slot = torch.arange(p_inp.shape[1], device=dev).clamp(max=self.slot_emb.num_embeddings - 1)[None]
        inp, span, slot = (torch.cat([a, b], 1) for a, b in ((inp, p_inp), (span, p_span), (slot, p_slot)))
        r = self.cfg.refine_passes
        return self._rows(H, inp, span, slot, r, {r - 1})[-1][1][0, -1]

    # ------------------------------------------------------------------ accounting
    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        gpc = self.glob.param_counts()
        dec = total - emb - gpc["total"] - _n(self.up, self.slot_emb, self.norm_out)
        d = {"total": total, "embedding": emb, "global": gpc["total"], "global_token_embedding": gpc["embedding"],
             "global_core": gpc["core"], "patch_io": _n(self.up, self.slot_emb), "decoder": dec,
             "decoder_blocks": _n(self.passes), "decoder_block_per_pass": _n(self.passes[0]),
             "head_norm": _n(self.norm_out), "local": total - gpc["total"], "non_embedding": total - emb - gpc["embedding"],
             "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
        if "moe_inactive" in gpc:
            d.update({"global_moe_inactive": gpc["moe_inactive"], "global_active": gpc["total"] - gpc["moe_inactive"],
                      "active_total": total - gpc["moe_inactive"]})
        return d

    def flops_breakdown(self, seq_len: int, loops: int | None = None, passes: int | None = None,
                        training: bool = True) -> dict:
        """FLOPs per TOKEN (the training position), by part: the global model once per token (FusionDance's rule,
        without its unused 8k head; forward only when frozen), the up-projection once per token, and the decoder
        and byte head once per decoder row, rows_per_token (= measured bytes per token + 1 EOT row) per token."""
        cfg = self.cfg
        r = cfg.refine_passes if passes is None else passes
        mult = 3 if training else 1
        rows = cfg.rows_per_token
        dec, head = self._decoder_row_flops(r, seq_len * rows, training)
        f = {"global": self._global_train_flops(seq_len, loops, cfg.freeze_global, training),
             "up": mult * 2 * self.gcfg.d_model * cfg.d_local,
             "decoder": mult * dec * rows, "head": mult * head * rows}
        f["total"] = sum(f.values())
        return f

    def flops_per_token(self, seq_len: int, loops: int | None = None, training: bool = True,
                        passes: int | None = None) -> float:
        return self.flops_breakdown(seq_len, loops=loops, passes=passes, training=training)["total"]

    def flops_per_byte(self, seq_len: int, **kw) -> float:
        return self.flops_per_token(seq_len, **kw) / BYTES_PER_TOKEN


def build_c2f(model_cfg: dict, table: list[bytes] | None = None) -> nn.Module:
    """C2FModel (patching 'fixed') or C2FTokenModel (patching 'tokens') from a config dict."""
    cfg = C2FConfig(**{k: v for k, v in model_cfg.items() if k != "arch"})
    return C2FTokenModel(cfg, table) if cfg.patching == "tokens" else C2FModel(cfg)
