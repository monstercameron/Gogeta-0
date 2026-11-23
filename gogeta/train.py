"""Train a FusionDance model from a JSON config (run inside WSL).

    ~/.venvs/gogeta/bin/python -m gogeta.train --config configs/e0_dense.json [--require-clean]
    ~/.venvs/gogeta/bin/python -m gogeta.train --config configs/e2_looped.json --bench 30   # throughput only

Logs through gogeta.tracking (Trackio by default). Per run, in $GOGETA_DATA/runs/<name>/:
  meta.json, summary.json (from tracking), samples.txt, and $GOGETA_DATA/checkpoints/<name>/model.pt

Architectures (top-level config key "arch", default "fusiondance"):
  fusiondance  gogeta.model.FusionDance, built from cfg["model"] (ModelConfig fields)
  hf_llama     Hugging Face's reference transformers.LlamaForCausalLM, built from cfg["model"]
               (LlamaConfig fields; vocab_size comes from the dataset). Wrapped by HFLlama so the
               loop, data, eval, tracking, FLOP accounting and checkpointing are the same code.

--lr overrides train.lr (used by the K7 LR sweep); the effective value is what the run config
records under "train", and every CLI override is listed under "cli_overrides".

Opt-in options (all OFF by default; defaults are bit-identical to the pre-option code):
  model.init            "gogeta" (default) | "flat" (FusionDance); for arch hf_llama: "hf" (default, HF's own
                        init) | "gogeta" (std 0.02, residual outputs o_proj/down_proj scaled by 1/sqrt(2*layers))
  train.schedule        "cosine" (default) | "wsd" (warmup, constant, linear decay over the last train.decay_frac)
  train.micro_batch     default = batch_size (no accumulation); else gradient accumulation, --micro-batch overrides
  model.moe_experts     0 (default, dense SwiGLU) | E > 0: Code-MoE FFN (gogeta/moe.py). The MoE auxiliary loss
                        (balance + z-loss) is added to the LM loss for backprop; train/loss and val/loss stay LM-only,
                        moe/* and train/loss_total are logged separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from gogeta.model import FusionDance, ModelConfig, set_ternary_quant
from gogeta.tracking import DATA, Run

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = [
    "def fibonacci(n):\n",
    "class Stack:\n",
    "Write a python function to check whether a string is a palindrome.\n\n```python\n",
    "def binary_search(arr, target):\n    \"\"\"",
]


# ----------------------------------------------------------------------------- data
class TokenData:
    def __init__(self, name: str, seq_len: int, seed: int):
        d = DATA / "datasets" / name
        self.meta = json.loads((d / "meta.json").read_text())
        self.train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
        self.T = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.chars_per_token = self.meta["splits"]["train"]["chars_per_token"]

    def batch(self, B: int, device: str):
        ix = torch.randint(len(self.train) - self.T - 1, (B,), generator=self.g).tolist()
        buf = np.stack([self.train[i:i + self.T + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(buf).pin_memory().to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def val_batches(self, n_seqs: int | None, B: int, device: str):
        """Deterministic, non-overlapping val sequences (first n_seqs; None = all)."""
        total = (len(self.val) - 1) // self.T
        n = total if n_seqs is None else min(n_seqs, total)
        for s in range(0, n, B):
            rows = [self.val[i * self.T:i * self.T + self.T + 1] for i in range(s, min(s + B, n))]
            t = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
            yield t[:, :-1].contiguous(), t[:, 1:].contiguous()


# ----------------------------------------------------------------------------- reference model
class HFLlama(torch.nn.Module):
    """HF `LlamaForCausalLM` behind the small interface this loop uses from FusionDance:
    forward(idx, targets, loops) -> (logits, loss), cfg.max_seq_len, core (empty: not looped),
    param_counts(), flops_per_token(). The HF model itself is unmodified (its own init, RoPE,
    RMSNorm, SDPA attention); the loss is computed here exactly as FusionDance computes it.
    """

    def __init__(self, model_cfg: dict, vocab_size: int, seq_len: int):
        super().__init__()
        from transformers import LlamaConfig, LlamaForCausalLM

        kw = {**model_cfg, "vocab_size": vocab_size, "use_cache": False}
        init = kw.pop("init", "hf")   # not a LlamaConfig field: "hf" (HF's own init, default) | "gogeta" | "flat"
        if init not in ("hf", "gogeta", "flat"):
            raise ValueError(f"unknown hf_llama init {init!r} (hf | gogeta | flat)")
        self.hf_config = LlamaConfig(**kw)
        if self.hf_config.max_position_embeddings < seq_len:
            raise ValueError(f"max_position_embeddings {self.hf_config.max_position_embeddings} < seq_len {seq_len}")
        self.lm = LlamaForCausalLM(self.hf_config)
        self.init = init
        if init == "gogeta":
            self._gogeta_init()
        self.core = torch.nn.ModuleList()   # never looped
        self.cfg = SimpleNamespace(max_seq_len=seq_len, to_dict=self.config_dict)

    @torch.no_grad()
    def _gogeta_init(self):
        """FusionDance's scheme on HF Llama: std 0.02 for every Linear/Embedding, and the residual-output
        projections (attention o_proj, MLP down_proj) at 0.02 / sqrt(2 * num_hidden_layers). HF's own init is
        already std 0.02 (initializer_range), so this only adds the residual scaling. Tied head follows the embedding."""
        std_res = 0.02 / math.sqrt(2 * max(1, self.hf_config.num_hidden_layers))
        for name, m in self.lm.named_modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=std_res if name.endswith(("o_proj", "down_proj")) else 0.02)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

    def config_dict(self) -> dict:
        return {"arch": "hf_llama", **self.hf_config.to_dict(), **({"init": self.init} if self.init != "hf" else {})}

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None):
        logits = self.lm(input_ids=idx, use_cache=False).logits
        if targets is None:
            return logits, None
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())   # tied head counted once
        emb = self.lm.model.embed_tokens.weight.numel()
        layers = sum(p.numel() for p in self.lm.model.layers.parameters())
        return {"total": total, "embedding": emb, "layers": layers, "other": total - emb - layers,
                "non_embedding": total - emb}

    def flops_per_token(self, seq_len: int, loops: int | None = None, training: bool = True) -> float:
        """Same rule as FusionDance.flops_per_token: forward = 2 x (block params + d x vocab for the
        head) + 4 x d_attn x (seq_len / 2) per attention layer (QK^T and AV); training = 3x forward.
        With E0's shapes this gives exactly E0's number."""
        c = self.hf_config
        d_attn = c.num_attention_heads * c.head_dim
        fwd = 2 * (self.param_counts()["layers"] + c.hidden_size * c.vocab_size)
        fwd += c.num_hidden_layers * 4 * d_attn * (seq_len / 2)
        return fwd * (3 if training else 1)


def build_model(cfg: dict, vocab_size: int, seq_len: int):
    """Returns (model, model_config_object_with_to_dict). Call after torch.manual_seed."""
    arch = cfg.get("arch", "fusiondance")
    if arch == "fusiondance":
        mc = ModelConfig(**{**cfg["model"], "vocab_size": vocab_size, "max_seq_len": seq_len})
        return FusionDance(mc), mc
    if arch == "hf_llama":
        m = HFLlama(cfg["model"], vocab_size, seq_len)
        return m, m.cfg
    raise ValueError(f"unknown arch {arch!r} (fusiondance | hf_llama)")


# ----------------------------------------------------------------------------- helpers
def lr_at(step: int, total: int, tc: dict) -> float:
    """LR at `step`. Linear warmup, then tc["schedule"]: "cosine" (default) decays to lr*min_lr_frac;
    "wsd" holds lr constant, then decays linearly to lr*min_lr_frac over the final tc["decay_frac"] (0.2) of steps."""
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    sched = tc.get("schedule", "cosine")
    if sched == "wsd":
        decay_start = total - int(round(tc.get("decay_frac", 0.2) * total))
        p = min(1.0, max(0.0, (step - decay_start) / max(1, total - decay_start)))
        return tc["lr"] * (1 - (1 - tc["min_lr_frac"]) * p)
    if sched != "cosine":
        raise ValueError(f"unknown schedule {sched!r} (cosine | wsd)")
    p = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    return tc["lr"] * (tc["min_lr_frac"] + (1 - tc["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def micro_batches(tc: dict) -> tuple[int, int]:
    """(micro_batch, n_micro) from train.micro_batch (default batch_size = no accumulation)."""
    B = tc["batch_size"]
    mb = tc.get("micro_batch") or B
    assert mb > 0 and B % mb == 0, f"batch_size {B} must be divisible by micro_batch {mb}"
    return mb, B // mb


def is_moe(model) -> bool:
    return bool(getattr(getattr(model, "cfg", None), "moe_experts", 0))


def forward_backward(fwd_model, x, y, r, n_micro: int = 1, stats: dict | None = None):
    """Forward + backward on one optimizer step's batch; returns the (mean) loss.
    n_micro == 1 is the plain full-batch step. Otherwise the same batch is split into n_micro equal row
    chunks, each backpropagated with loss/n_micro, so the summed gradient equals the full-batch mean-loss
    gradient (equal token counts per chunk) up to float associativity; same data order, same effective batch.
    MoE models (model.moe_experts > 0) backprop LM loss + the MoE auxiliary loss; the returned loss is still
    the LM loss only, and `stats` (if given) receives the detached aux diagnostics (see _forward_backward_moe)."""
    if is_moe(fwd_model):
        return _forward_backward_moe(fwd_model, x, y, r, n_micro, stats)
    if n_micro == 1:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd_model(x, y, loops=r)
        loss.backward()
        return loss
    total = 0.0
    for xs, ys in zip(x.chunk(n_micro), y.chunk(n_micro)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd_model(xs, ys, loops=r)
        (loss / n_micro).backward()
        total = total + loss.detach() / n_micro
    return total


MOE_STATS = ("aux", "balance", "z", "max_load", "min_load", "entropy")


def _forward_backward_moe(fwd_model, x, y, r, n_micro, stats):
    """As forward_backward, but backprops (lm + aux["loss"]) / n_micro per chunk. stats gets the micro-batch
    means of the detached tensors moe/{aux,balance,z,max_load,min_load,entropy} (no host sync here)."""
    total, acc = 0.0, {k: 0.0 for k in MOE_STATS}
    for xs, ys in (zip(x.chunk(n_micro), y.chunk(n_micro)) if n_micro > 1 else [(x, y)]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, aux = fwd_model(xs, ys, loops=r, return_aux=True)
        ((loss + aux["loss"]) / n_micro).backward()
        total = total + loss.detach() / n_micro
        for k in MOE_STATS:
            acc[k] = acc[k] + (aux["loss"] if k == "aux" else aux[k]).detach().float() / n_micro
    if stats is not None:
        stats.update(acc)
    return total


@torch.no_grad()
def evaluate(model, data, n_seqs, B, device, loops=None) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for x, y in data.val_batches(n_seqs, B, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, loops=loops)
        tot += loss.item() * x.numel()
        cnt += x.numel()
    model.train()
    return tot / cnt


@torch.no_grad()
def sample(model, tok, prompt: str, n_new: int, device, temperature=0.8, top_k=40, loops=None) -> str:
    model.eval()
    ids = tok.encode(prompt).ids
    eos = tok.token_to_id("<|eos|>")
    x = torch.tensor([ids], device=device)
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, -model.cfg.max_seq_len:], loops=loops)
        logits = logits[0, -1].float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() == eos:
            break
        x = torch.cat([x, nxt[None]], dim=1)
    model.train()
    return tok.decode(x[0].tolist())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-tokens", type=int, help="override train.max_tokens")
    ap.add_argument("--seed", type=int, help="override train.seed")
    ap.add_argument("--lr", type=float, help="override train.lr (peak LR; recorded in the run config)")
    ap.add_argument("--micro-batch", type=int, help="override train.micro_batch (gradient accumulation; batch_size % micro == 0)")
    ap.add_argument("--require-clean", action="store_true", help="refuse to run with uncommitted changes")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--bench", type=int, default=0, help="only time N steps, no tracking/eval/ckpt")
    args = ap.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not os.path.isabs(args.config) else pathlib.Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    arch = cfg.get("arch", "fusiondance")
    tc = dict(cfg["train"])
    if args.max_tokens:
        tc["max_tokens"] = args.max_tokens
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.lr is not None:
        tc["lr"] = args.lr
    if args.micro_batch is not None:
        tc["micro_batch"] = args.micro_batch
    micro_batches(tc)   # validate early
    overrides = {k: v for k, v in (("max_tokens", args.max_tokens or None), ("seed", args.seed),
                                   ("lr", args.lr), ("micro_batch", args.micro_batch)) if v is not None}
    device = "cuda"
    torch.manual_seed(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    data = TokenData(tc["dataset"], tc["seq_len"], tc["seed"])
    model, mc = build_model(cfg, data.meta["vocab_size"], tc["seq_len"])
    model = model.to(device)
    fwd_model = torch.compile(model) if args.compile else model
    pc = model.param_counts()
    looped = len(model.core) > 0
    loop_rng = random.Random(tc["seed"] + 1)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), fused=True)

    tokens_per_step = tc["batch_size"] * tc["seq_len"]
    total_steps = args.bench or math.ceil(tc["max_tokens"] / tokens_per_step)
    print(f"params: total {pc['total']/1e6:.2f}M non-emb {pc['non_embedding']/1e6:.2f}M | "
          f"steps {total_steps} x {tokens_per_step} tok | looped={looped}", flush=True)

    run = None
    if not args.bench:
        run_cfg = {"config_file": str(cfg_path.relative_to(ROOT)), "arch": arch, "cli_overrides": overrides,
                   "model": mc.to_dict(), "train": tc,
                   "params": pc, "flops_per_token_at_default_loops": model.flops_per_token(tc["seq_len"]),
                   "compile": args.compile, "dataset_meta": {k: data.meta[k] for k in ("name", "source", "vocab_size")}}
        run = Run(cfg["experiment"], config=run_cfg, seed=tc["seed"], require_clean=args.require_clean)
        print(f"run: {run.name}", flush=True)

    t_start = time.time()
    tokens = flops = 0.0
    t_log, tok_log = time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(total_steps):
            lr = lr_at(step, total_steps, tc)
            for gp in opt.param_groups:
                gp["lr"] = lr
            r = loop_rng.randint(mc.loops_min, mc.loops_max) if looped else None
            if tc.get("ternary_start_frac"):  # full-precision training first, ternary QAT from this fraction on (A1)
                if set_ternary_quant(model, step >= tc["ternary_start_frac"] * total_steps):
                    print(f"step {step}: ternary quantisation {'on' if step else 'off'}", flush=True)
            x, y = data.batch(tc["batch_size"], device)
            moe_stats = {} if is_moe(model) else None
            loss = forward_backward(fwd_model, x, y, r, micro_batches(tc)[1], moe_stats)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)

            tokens += tokens_per_step
            if looped and r > mc.bptt_loops:  # passes before the last bptt_loops run under no_grad: forward only (v0.5.3)
                g = mc.bptt_loops
                flops += (model.flops_per_token(tc["seq_len"], loops=r, training=False)
                          + 2 * model.flops_per_token(tc["seq_len"], loops=g, training=False)) * tokens_per_step
            else:
                flops += model.flops_per_token(tc["seq_len"], loops=r) * tokens_per_step
            tok_log += tokens_per_step
            if step % 10 == 0 or step == total_steps - 1:
                torch.cuda.synchronize()
                dt = time.time() - t_log
                tps = tok_log / dt if dt > 0 else 0.0
                t_log, tok_log = time.time(), 0
                m = {"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": gnorm.item(),
                     "tokens": tokens, "flops": flops, "perf/tokens_per_s": tps}
                if looped:
                    m["train/loops"] = r
                if moe_stats:   # train/loss stays the LM loss; the optimised total is LM + moe/aux
                    m.update({f"moe/{k}": v.item() for k, v in moe_stats.items()})
                    m["train/loss_total"] = loss.item() + m["moe/aux"]
                if run:
                    run.log(m, step=step)
                if step % 100 == 0:
                    print(f"step {step:6d}/{total_steps} loss {loss.item():.4f} lr {lr:.2e} "
                          f"gn {gnorm.item():.2f} {tps/1e3:.0f}k tok/s r={r}"
                          + (f" moe_aux {m['moe/aux']:.4f} max_load {m['moe/max_load']:.3f}" if moe_stats else ""),
                          flush=True)
            if run and (step % tc["eval_every"] == 0 or step == total_steps - 1) and step > 0:
                vl = evaluate(model, data, tc["eval_seqs"], tc["batch_size"], device)
                run.log({"val/loss": vl, "tokens": tokens, "flops": flops}, step=step)
                print(f"  eval step {step}: val/loss {vl:.4f}", flush=True)

        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**30
        if args.bench:
            print(f"BENCH {cfg['experiment']}: {total_steps} steps, {tokens/wall/1e3:.1f}k tok/s overall "
                  f"(incl. warmup), peak mem {peak:.2f} GB", flush=True)
            return

        # ---- final evaluation on the full validation split
        summary = {"val_loss": evaluate(model, data, None, tc["batch_size"], device)}
        summary["val_bits_per_char"] = summary["val_loss"] / math.log(2) / data.chars_per_token
        if looped:
            for r_eval in tc.get("eval_loops", [mc.loops]):
                summary[f"val_loss_r{r_eval}"] = evaluate(model, data, None, tc["batch_size"], device, loops=r_eval)
        summary.update({
            "tokens": tokens, "flops": flops, "steps": total_steps, "wall_min": wall / 60,
            "tokens_per_s": tokens / wall, "peak_mem_gb": peak,
            "params_total": pc["total"], "params_non_embedding": pc["non_embedding"],
            "flops_per_token_default": model.flops_per_token(tc["seq_len"]),
        })
        run.summary.update(summary)
        run.log({f"final/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}, step=total_steps)
        print("FINAL " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}),
              flush=True)

        # ---- samples + checkpoint
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(DATA / "datasets" / tc["dataset"] / "tokenizer.json"))
        torch.manual_seed(0)
        texts = [sample(model, tok, p, 160, device) for p in PROMPTS]
        (run.dir / "samples.txt").write_text("\n\n" + "\n\n=====\n\n".join(texts) + "\n", encoding="utf-8")
        ck = DATA / "checkpoints" / run.name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "model_config": mc.to_dict(), "train": tc}, ck / "model.pt")
        run.summary["checkpoint"] = str(ck / "model.pt")
        run.finish("done")
        print(f"done: {run.dir}", flush=True)
    except BaseException as e:
        if run and run._t is not None:
            run.finish("failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
