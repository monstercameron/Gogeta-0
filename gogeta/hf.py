"""Thin Hugging Face `transformers` wrapper around FusionDance, so TRL trainers can use it.

    from gogeta.hf import GogetaConfig, GogetaForCausalLM, load_tokenizer, from_gogeta_checkpoint

    model = from_gogeta_checkpoint("~/fusiondance-data/runs/<run>/ckpt/model.pt")
    tok = load_tokenizer()

Written against transformers 5.24 / TRL 1.18 (see requirements-rl.txt and
post_training/SETUP.md). `gogeta/model.py` is not modified: the wrapper holds a FusionDance
subclass whose forward returns the final hidden states, so the parameter names (and a
FusionDance checkpoint's state dict) are unchanged under the `model.` prefix.

What the wrapper provides, and why:
  * `GogetaConfig(PreTrainedConfig)`: every ModelConfig field, plus the HF fields
    (eos/pad/bos ids, `tie_word_embeddings`, `use_cache=False`).
  * `GogetaForCausalLM(PreTrainedModel, GenerationMixin)`:
      - forward(input_ids, attention_mask, labels, loops, logits_to_keep) -> CausalLMOutput.
        `labels` follow the HF convention (unshifted, -100 = ignore); the loss is shifted inside.
      - `base_model` (= `.model`) returns `last_hidden_state` and `get_output_embeddings()`
        returns the head: TRL's GRPO/DPO/KTO trainers patch the forward with a fused LM head
        that needs exactly these two (trl/trainer/utils.py: add_fused_lm_head).
      - generation through `GenerationMixin.generate` with NO KV cache: every new token
        recomputes the whole sequence (O(T^2) work per sample). `use_cache` is accepted and
        ignored. A cached / recurrent decode path is future work (see SETUP.md, blockers).
  * Padding. FusionDance has no attention-mask input. Right padding is harmless (causal).
    LEFT padding (what TRL uses for prompts) is handled here exactly: rows are grouped by
    their number of leading pad tokens and each group runs on its un-padded suffix, so pads
    never enter attention and RoPE positions start at 0 at the first real token. Masks with
    holes in the middle (packing / padding-free) are rejected.
  * Loops: a fixed `loops` (default `config.loops`, or the `loops=` argument). In training
    mode FusionDance's truncated BPTT applies (only the last `bptt_loops` iterations get
    gradients), as in pretraining.
  * "gdn" blocks need flash-linear-attention and CUDA; "attn"/"swa" run anywhere.
"""
from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from gogeta.model import FusionDance, GatedDeltaNet, ModelConfig, Rotary

TOKENIZER_PATH = pathlib.Path(__file__).resolve().parent / "assets" / "tokenizer-code-py-v1.json"
EOS, PAD = "<|eos|>", "<|pad|>"


# ----------------------------------------------------------------------------- config
class GogetaConfig(PreTrainedConfig):
    """HF config holding the ModelConfig fields. `tie_embeddings` is stored as the HF name
    `tie_word_embeddings`; everything else keeps its ModelConfig name."""

    model_type = "gogeta"
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "n_heads",
                     "max_position_embeddings": "max_seq_len", "intermediate_size": "mlp_hidden"}

    vocab_size: int = 8192
    d_model: int = 384
    n_heads: int = 6
    mlp_hidden: int = 1024
    prelude: list = field(default_factory=lambda: ["swa"])
    core: list = field(default_factory=lambda: ["gdn", "gdn", "swa"])
    coda: list = field(default_factory=lambda: ["attn"])
    loops: int = 4
    loops_min: int = 1
    loops_max: int = 7
    bptt_loops: int = 4
    window: int = 256
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    conv_size: int = 4
    ternary: bool = False
    gdn_neg_eigval: bool = False  # v0.4.7 (E11): beta in (0, 2) via the kernel's 2*sigmoid
    init: str = "gogeta"          # ModelConfig.init: "gogeta" (scaled residual-output init) | "flat"
    # ModelConfig.moe_* (E60 Code-MoE FFN, gogeta/moe.py); defaults = dense. Not inherited from to_dict() of a
    # dense ModelConfig (it omits moe_* keys), so these defaults must equal ModelConfig's.
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_hidden: int = 512
    moe_balance_coef: float = 0.01
    moe_zloss_coef: float = 0.001
    moe_router_frozen: bool = False
    moe_impl: str = "gather"
    # ModelConfig assembly switches (v0.5.x; gogeta.model._ASSEMBLY_DEFAULTS); defaults = off, equal to ModelConfig's,
    # because to_dict() omits them when they are at their defaults.
    loop_embed: bool = False
    inject_contractive: bool = False
    ternary_acts: bool = True
    ternary_fp_ends: bool = False
    loop_adapter_rank: int = 0
    ternary_mixer: bool = True
    ternary_ffn: bool = True
    moe_core_only: bool = False
    core_mlp_hidden: int = 0
    gdn_beta_fixed: float = 0.0
    tie_word_embeddings: bool = True
    # HF / generation fields (ids match gogeta/assets/tokenizer-code-py-v1.json)
    eos_token_id: int | None = 0
    pad_token_id: int | None = 1
    bos_token_id: int | None = None
    use_cache: bool = False
    initializer_range: float = 0.02
    # TRL's create_model_from_path (used for the beta>0 reference model) reads config.auto_map
    # directly; Hub remote-code configs have it, ours is registered in-process instead.
    auto_map: dict = field(default_factory=dict)

    @classmethod
    def from_model_config(cls, mc: ModelConfig | dict, **kwargs) -> "GogetaConfig":
        d = dict(mc.to_dict() if isinstance(mc, ModelConfig) else mc)
        d["tie_word_embeddings"] = d.pop("tie_embeddings", True)
        return cls(**d, **kwargs)

    def to_model_config(self) -> ModelConfig:
        """Exact ModelConfig. Fails loudly if model.py gains a field this config lacks."""
        d = {}
        for f in dataclasses.fields(ModelConfig):
            name = "tie_word_embeddings" if f.name == "tie_embeddings" else f.name
            if not hasattr(self, name):
                raise AttributeError(f"GogetaConfig is missing ModelConfig field {f.name!r}; update gogeta/hf.py")
            v = getattr(self, name)
            d[f.name] = list(v) if isinstance(v, (list, tuple)) else v
        return ModelConfig(**d)


# ----------------------------------------------------------------------------- backbone
class FusionDanceBackbone(FusionDance):
    """FusionDance whose forward stops before the LM head and returns `last_hidden_state`.

    Same parameters and state-dict keys as FusionDance. `hidden()` mirrors
    FusionDance.forward up to `norm_out`; post_training/test_hf.py checks that
    head(hidden(x)) == FusionDance.forward(x) exactly, so a change in model.py that is not
    mirrored here fails that test.
    """

    def hidden(self, idx: torch.Tensor, loops: int | None = None) -> torch.Tensor:
        if idx.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {idx.size(1)} > max_seq_len {self.cfg.max_seq_len} "
                             "(RoPE table size); shorten prompt + completion")
        # v0.5.24: delegate to FusionDance.forward's return_hidden path (v0.5.12) instead of a hand-written mirror,
        # so every ModelConfig switch (loop_embed, contractive inject, adapters, MoE, ternary scope) is honoured.
        # The mirror predated those switches and would have silently ignored them.
        return FusionDance.forward(self, idx, loops=loops, return_hidden=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                loops: int | None = None, **kwargs) -> BaseModelOutput:
        if attention_mask is None or bool(attention_mask.all()):
            return BaseModelOutput(last_hidden_state=self.hidden(input_ids, loops))
        am = attention_mask.bool()
        B, T = am.shape
        pos = torch.arange(T, device=am.device)
        start = (~am).long().cumprod(dim=1).sum(dim=1)                 # leading pads per row
        end = T - (~am).long().flip(1).cumprod(dim=1).sum(dim=1)       # one past the last real token
        expected = (pos[None] >= start[:, None]) & (pos[None] < end[:, None])
        if not torch.equal(am, expected):
            raise ValueError("attention_mask has holes; only left/right padding is supported "
                             "(no packing / padding-free batches)")
        out = None
        for s in torch.unique(start).tolist():
            if s >= T:
                continue  # fully padded row: stays zero
            rows = (start == s).nonzero(as_tuple=True)[0]
            h = self.hidden(input_ids[rows, s:], loops)
            if out is None:
                out = h.new_zeros(B, T, h.size(-1))
            out = out.index_put((rows[:, None], torch.arange(s, T, device=h.device)[None, :]), h)
        if out is None:
            out = self.embed.weight.new_zeros(B, T, self.cfg.d_model)
        return BaseModelOutput(last_hidden_state=out)


# ----------------------------------------------------------------------------- causal LM
class GogetaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GogetaConfig
    config: GogetaConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False   # set activation/gradient_checkpointing=False in TRL configs
    _no_split_modules = ["Block"]
    _tied_weights_keys = {"model.head.weight": "model.embed.weight"}  # only applied if tie_word_embeddings

    def __init__(self, config: GogetaConfig):
        super().__init__(config)
        self.model = FusionDanceBackbone(config.to_model_config())
        self.post_init()

    # --- weight init: FusionDance's scheme, also used by from_pretrained for anything not
    # in the checkpoint. Must recompute the RoPE tables: they are non-persistent buffers,
    # which transformers 5 re-creates EMPTY when loading (modeling_utils: _move_missing_keys...).
    @torch.no_grad()
    def _init_weights(self, module):
        from transformers import initialization as init

        if isinstance(module, Rotary):
            hd = self.config.d_model // self.config.n_heads
            inv = 1.0 / (self.config.rope_base ** (torch.arange(0, hd, 2).float() / hd))
            freqs = torch.outer(torch.arange(self.config.max_seq_len).float(), inv)
            init.copy_(module.cos, freqs.cos()[None, None].to(module.cos.device))
            init.copy_(module.sin, freqs.sin()[None, None].to(module.sin.device))
        elif isinstance(module, nn.Linear):
            depth = self.model.effective_depth() if hasattr(self, "model") else 1
            scaled = getattr(module, "is_residual_out", False) and self.config.init != "flat"
            std = 0.02 / (2 * max(1, depth)) ** 0.5 if scaled else 0.02
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            if module.weight is not None:
                init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            init.kaiming_uniform_(module.weight, a=5 ** 0.5)  # torch's default Conv1d init
        elif isinstance(module, GatedDeltaNet):
            H = module.H
            init.copy_(module.A_log, torch.log(torch.empty(H).uniform_(1, 16)))
            lo, hi = torch.log(torch.tensor(0.001)), torch.log(torch.tensor(0.1))
            dt = torch.exp(torch.rand(H) * (hi - lo) + lo).clamp(min=1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))

    # --- embeddings accessors (used by TRL's fused LM head, resize_token_embeddings, tying)
    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def get_output_embeddings(self):
        return self.model.head

    def set_output_embeddings(self, value):
        self.model.head = value

    @property
    def lm_head(self):  # some TRL options (cast_lm_head_to_fp32, continuous batching) look for `lm_head`
        return self.model.head

    # --- forward
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, loops: int | None = None,
                logits_to_keep: int = 0, use_cache: bool | None = None,
                num_items_in_batch: int | torch.Tensor | None = None, **kwargs) -> CausalLMOutput:
        h = self.model(input_ids, attention_mask=attention_mask, loops=loops).last_hidden_state
        loss = None
        if labels is not None:
            logits = self.model.head(h)
            shift_logits = logits[:, :-1].float().reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
            if num_items_in_batch is None:
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            else:
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
                loss = loss / num_items_in_batch
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:]
        else:
            logits = self.model.head(h[:, -logits_to_keep:] if logits_to_keep else h)
        return CausalLMOutput(loss=loss, logits=logits)

    # --- generation without a KV cache
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False  # no Cache object is ever built for this model

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, loops=None, **kwargs):
        # Full sequence every step (no cache). generate() extends attention_mask itself.
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if loops is not None:
            inputs["loops"] = loops
        return inputs


# ----------------------------------------------------------------------------- helpers
def load_tokenizer(path: str | pathlib.Path = TOKENIZER_PATH, padding_side: str = "left",
                   model_max_length: int = 1024) -> PreTrainedTokenizerFast:
    """Our byte-level BPE as a PreTrainedTokenizerFast with eos=<|eos|> (id 0), pad=<|pad|> (id 1),
    no bos and no chat template. Left padding by default: TRL's GRPO requires it."""
    tok = PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token=EOS, pad_token=PAD,
                                  padding_side=padding_side, truncation_side="left",
                                  model_max_length=model_max_length)
    assert tok.eos_token_id == 0 and tok.pad_token_id == 1, (tok.eos_token_id, tok.pad_token_id)
    return tok


def from_gogeta_checkpoint(ckpt: str | pathlib.Path | dict, map_location: str | torch.device = "cpu",
                           **config_overrides) -> GogetaForCausalLM:
    """Build the wrapper from a gogeta/train.py checkpoint {"model": state_dict, "model_config": dict}.

    `config_overrides` set GogetaConfig fields, e.g. loops=6 for a different test-time depth.
    """
    if not isinstance(ckpt, dict):
        ckpt = torch.load(pathlib.Path(ckpt).expanduser(), map_location=map_location, weights_only=True)
    cfg = GogetaConfig.from_model_config(ckpt["model_config"], **config_overrides)
    model = GogetaForCausalLM(cfg)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}  # torch.compile prefix
    model.model.load_state_dict(sd, strict=True)
    return model.to(map_location) if not isinstance(map_location, dict) else model


def to_gogeta_checkpoint(model: GogetaForCausalLM) -> dict:
    """Inverse of from_gogeta_checkpoint: a dict gogeta/train.py-style code can load."""
    return {"model": model.model.state_dict(), "model_config": model.config.to_model_config().to_dict()}


def register_auto_classes() -> None:
    """Let AutoConfig / AutoModelForCausalLM resolve model_type "gogeta" (in this process).

    Needed whenever TRL loads the model by path, e.g. GRPO/DPO with beta > 0 build their
    reference model with AutoModelForCausalLM.from_pretrained(model.config._name_or_path).
    Called at import time; safe to call again.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        AutoConfig.register("gogeta", GogetaConfig)
    except ValueError:
        pass  # already registered
    try:
        AutoModelForCausalLM.register(GogetaConfig, GogetaForCausalLM)
    except ValueError:
        pass


register_auto_classes()
