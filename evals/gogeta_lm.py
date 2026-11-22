"""lm-evaluation-harness model class for Gogeta-0 (`--model gogeta`).

Registered as "gogeta" in lm-eval's model registry when this module is imported
(`evals/run_suite.py` does that; with the lm-eval CLI use `--plugins evals.gogeta_lm` or
`--include_path` plus an import). It is a thin subclass of lm-eval's `HFLM` that builds
`gogeta.hf.GogetaForCausalLM` itself and hands HFLM the already-initialised model object
(HFLM's documented "pass a PreTrainedModel instead of a name" path), so every request type
(loglikelihood, loglikelihood_rolling, generate_until) uses HFLM's own, standard code.

    model_args:
      checkpoint=<path>        gogeta/train.py checkpoint ({"model", "model_config"}) or an HF
                               directory written by GogetaForCausalLM.save_pretrained
      random_init=<preset>     instead of a checkpoint: a randomly initialised model
                               ("tiny_attn" = attn/swa only, CPU-safe). Wiring tests only.
      loops=<int>              test-time loop count for looped configs (default: config.loops)
      device=cpu|cuda          batch_size=<int>   max_length=<int> (default: config.max_seq_len)

Why these HFLM settings:
  * backend="causal": our config has no model_type HFLM knows.
  * prefix_token_id = <|eos|> (id 0), because the tokenizer has no BOS. HFLM's default
    already falls back to eos; we pin it so it cannot drift. In pretraining every document
    follows an <|eos|>, so "eos, then the document" is exactly the training distribution.
  * add_bos_token=False: there is no BOS.
  * max_length = config.max_seq_len (1024): the RoPE table size. Rolling-loglikelihood tasks
    split longer documents into disjoint 1,024-token windows (lm_eval.utils.
    get_rolling_token_windows with context_len=1).
  * The wrapper has no KV cache, so generation recomputes the full sequence per token
    (gogeta/hf.py). Fine for loglikelihood tasks; slow for generation tasks.
  * "gdn" blocks need CUDA + flash-linear-attention (not installed in gogeta-eval yet).
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

import dataclasses  # noqa: E402

from gogeta.hf import GogetaConfig, GogetaForCausalLM, load_tokenizer  # noqa: E402
from gogeta.model import ModelConfig  # noqa: E402

# Random-init presets: wiring/smoke tests only, never results.
PRESETS = {
    # same shape family as post_training/test_hf.py's TINY, but with our real 1,024 context
    "tiny_attn": dict(vocab_size=8192, d_model=64, n_heads=4, mlp_hidden=128, prelude=["swa"],
                      core=["attn", "swa"], coda=["attn"], loops=2, loops_min=1, loops_max=3,
                      bptt_loops=2, window=64, max_seq_len=1024),
}


def _fill_missing_fields(cfg: GogetaConfig) -> list[str]:
    """Compatibility shim: if gogeta/model.py's ModelConfig gained a field that gogeta/hf.py's
    GogetaConfig does not declare yet, set it on this config instance to the ModelConfig
    DEFAULT (i.e. exactly what an older checkpoint / config means). Returns the filled names,
    which go into the results' provenance. Fields given explicitly are never touched."""
    filled = []
    for f in dataclasses.fields(ModelConfig):
        name = "tie_word_embeddings" if f.name == "tie_embeddings" else f.name
        if not hasattr(cfg, name):
            default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            setattr(cfg, name, default)
            filled.append(name)
    return filled


def build_model(checkpoint: str | None = None, random_init: str | None = None, loops: int | None = None,
                seed: int = 0) -> tuple[GogetaForCausalLM, dict]:
    """Return (model in eval mode on CPU, provenance dict)."""
    overrides = {"loops": int(loops)} if loops is not None else {}
    if (checkpoint is None) == (random_init is None):
        raise ValueError("give exactly one of checkpoint=... or random_init=...")
    if random_init is not None:
        if random_init not in PRESETS:
            raise ValueError(f"unknown random_init preset {random_init!r}; choose from {sorted(PRESETS)}")
        torch.manual_seed(int(seed))
        cfg = GogetaConfig(**{**PRESETS[random_init], **overrides})
        filled = _fill_missing_fields(cfg)
        model = GogetaForCausalLM(cfg)
        prov = {"kind": "random_init", "preset": random_init, "seed": int(seed)}
    else:
        p = pathlib.Path(checkpoint).expanduser()
        if p.is_dir():
            cfg = GogetaConfig.from_pretrained(str(p), **overrides)
            filled = _fill_missing_fields(cfg)
            model = GogetaForCausalLM.from_pretrained(str(p), config=cfg)
            prov = {"kind": "hf_dir", "path": str(p)}
        else:  # same as gogeta.hf.from_gogeta_checkpoint, plus the shim
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            cfg = GogetaConfig.from_model_config(ckpt["model_config"], **overrides)
            filled = _fill_missing_fields(cfg)
            model = GogetaForCausalLM(cfg)
            sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
            model.model.load_state_dict(sd, strict=True)
            prov = {"kind": "gogeta_checkpoint", "path": str(p), "bytes": p.stat().st_size}
    prov["config_fields_filled_with_defaults"] = filled
    kinds = set(model.config.prelude) | set(model.config.core) | set(model.config.coda)
    prov.update(n_params=sum(t.numel() for t in model.parameters()), block_kinds=sorted(kinds),
                loops=model.config.loops, max_seq_len=model.config.max_seq_len)
    return model.eval(), prov


@register_model("gogeta")
class GogetaLM(HFLM):
    """HFLM around GogetaForCausalLM. Construct via model_args (see module docstring)."""

    def __init__(self, checkpoint: str | None = None, random_init: str | None = None,
                 loops: int | None = None, seed: int = 0, device: str = "cpu",
                 batch_size: int | str = 8, max_length: int | None = None, **kwargs):
        model, prov = build_model(checkpoint, random_init, loops, seed)
        if "gdn" in prov["block_kinds"] and not str(device).startswith("cuda"):
            raise RuntimeError("this model has 'gdn' blocks, which need CUDA + flash-linear-attention")
        model = model.to(device)
        tok = load_tokenizer(padding_side="left", model_max_length=model.config.max_seq_len)
        self.gogeta_provenance = prov
        # kwargs HFLM would only use when loading by name are dropped by HFLM itself (it warns).
        super().__init__(pretrained=model, tokenizer=tok, backend="causal",
                         max_length=int(max_length or model.config.max_seq_len),
                         batch_size=int(batch_size) if str(batch_size).isdigit() else batch_size,
                         add_bos_token=False, prefix_token_id=tok.eos_token_id, **kwargs)

    # HFLM's default is 256; our completions (solutions) are short, tasks set their own anyway.
    @property
    def max_gen_toks(self) -> int:
        return int(os.environ.get("GOGETA_MAX_GEN_TOKS", 384))
