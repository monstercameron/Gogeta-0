"""Train a FusionDance model from a JSON config (run inside WSL).

    ~/.venvs/gogeta/bin/python -m gogeta.train --config configs/e0_dense.json [--require-clean]
    ~/.venvs/gogeta/bin/python -m gogeta.train --config configs/e2_looped.json --bench 30   # throughput only

Logs through gogeta.tracking (Trackio by default). Per run, in $GOGETA_DATA/runs/<name>/:
  meta.json, summary.json (from tracking), samples.txt, and $GOGETA_DATA/checkpoints/<name>/model.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import time

import numpy as np
import torch

from gogeta.model import FusionDance, ModelConfig
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


# ----------------------------------------------------------------------------- helpers
def lr_at(step: int, total: int, tc: dict) -> float:
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    return tc["lr"] * (tc["min_lr_frac"] + (1 - tc["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


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
    ap.add_argument("--require-clean", action="store_true", help="refuse to run with uncommitted changes")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--bench", type=int, default=0, help="only time N steps, no tracking/eval/ckpt")
    args = ap.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not os.path.isabs(args.config) else pathlib.Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    tc = dict(cfg["train"])
    if args.max_tokens:
        tc["max_tokens"] = args.max_tokens
    if args.seed is not None:
        tc["seed"] = args.seed
    device = "cuda"
    torch.manual_seed(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    data = TokenData(tc["dataset"], tc["seq_len"], tc["seed"])
    mc = ModelConfig(**{**cfg["model"], "vocab_size": data.meta["vocab_size"], "max_seq_len": tc["seq_len"]})
    model = FusionDance(mc).to(device)
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
        run_cfg = {"config_file": str(cfg_path.relative_to(ROOT)), "model": mc.to_dict(), "train": tc,
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
            x, y = data.batch(tc["batch_size"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd_model(x, y, loops=r)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)

            tokens += tokens_per_step
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
                if run:
                    run.log(m, step=step)
                if step % 100 == 0:
                    print(f"step {step:6d}/{total_steps} loss {loss.item():.4f} lr {lr:.2e} "
                          f"gn {gnorm.item():.2f} {tps/1e3:.0f}k tok/s r={r}", flush=True)
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
