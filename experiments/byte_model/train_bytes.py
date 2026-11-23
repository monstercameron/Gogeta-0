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

Byte datasets (meta.json "unit": "bytes", e.g. code-py-v1-bytes, uint8, EOS = 0x00): the budget is train.max_bytes
(--max-bytes), and max_tokens is refused, so a byte budget is never read as tokens (or vice versa). A "token" in the
loop is then one byte position; runs also log bytes and token_equiv (bytes / 3.735). Configs with
model.patch_size > 0 build gogeta.model.PatchedByteLM (coarse-to-fine); refinement passes are sampled per step when
refine_max > refine_min, and per-pass / per-slot losses are logged. --device cpu runs a CPU smoke test (no autocast,
no fused AdamW, no CUDA calls).
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

if os.environ.get("GOGETA_CPU_GDN"):  # MIRROR-ONLY
    from experiments.byte_model import cpu_ref  # MIRROR-ONLY
    cpu_ref.install()  # MIRROR-ONLY
from experiments.byte_model.model_bytes import BYTES_PER_TOKEN, FusionDance, ModelConfig, make_model  # MIRROR
from gogeta.tracking import DATA, Run

ROOT = pathlib.Path(__file__).resolve().parents[2]  # MIRROR
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
        dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.train = np.memmap(d / "train.bin", dtype=dtype, mode="r")
        self.val = np.memmap(d / "val.bin", dtype=dtype, mode="r")
        self.unit = self.meta.get("unit", "tokens")   # "bytes" for code-py-v1-bytes
        self.T = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.chars_per_token = self.meta["splits"]["train"]["chars_per_token"]   # chars per position (incl. EOS)

    def batch(self, B: int, device: str):
        ix = torch.randint(len(self.train) - self.T - 1, (B,), generator=self.g).tolist()
        buf = np.stack([self.train[i:i + self.T + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(buf)
        t = t.pin_memory().to(device, non_blocking=True) if device == "cuda" else t.to(device)
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
class ByteTokenizer:
    """The tokenizer interface sample() uses, for byte datasets: ids are byte values, EOS = meta eos_id (0x00)."""

    def __init__(self, eos_id: int = 0):
        self.eos_id = eos_id

    def encode(self, text: str):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i != self.eos_id and 0 <= i < 256).decode("utf-8", errors="replace")

    def token_to_id(self, name: str):
        return self.eos_id if name == "<|eos|>" else None


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
        self.hf_config = LlamaConfig(**kw)
        if self.hf_config.max_position_embeddings < seq_len:
            raise ValueError(f"max_position_embeddings {self.hf_config.max_position_embeddings} < seq_len {seq_len}")
        self.lm = LlamaForCausalLM(self.hf_config)
        self.core = torch.nn.ModuleList()   # never looped
        self.cfg = SimpleNamespace(max_seq_len=seq_len, to_dict=self.config_dict)

    def config_dict(self) -> dict:
        return {"arch": "hf_llama", **self.hf_config.to_dict()}

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
        return make_model(mc), mc   # FusionDance, or PatchedByteLM when model.patch_size > 0
    if arch == "hf_llama":
        m = HFLlama(cfg["model"], vocab_size, seq_len)
        return m, m.cfg
    raise ValueError(f"unknown arch {arch!r} (fusiondance | hf_llama)")


# ----------------------------------------------------------------------------- helpers
def lr_at(step: int, total: int, tc: dict) -> float:
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    return tc["lr"] * (tc["min_lr_frac"] + (1 - tc["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def autocast(device: str):
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")


def is_patched(model) -> bool:
    return getattr(getattr(model, "cfg", None), "patch_size", 0) > 0


@torch.no_grad()
def evaluate(model, data, n_seqs, B, device, loops=None, passes=None) -> float:
    model.eval()
    kw = {"passes": passes} if passes is not None else {}
    tot, cnt = 0.0, 0
    for x, y in data.val_batches(n_seqs, B, device):
        with autocast(device):
            _, loss = model(x, y, loops=loops, **kw)
        tot += loss.item() * x.numel()
        cnt += x.numel()
    model.train()
    return tot / cnt


@torch.no_grad()
def evaluate_detail(model, data, n_seqs, B, device) -> dict:
    """Patched byte models: val loss of every refinement pass and of every within-patch slot (patch lag)."""
    model.eval()
    P, r = model.cfg.patch_size, model.cfg.refine_passes
    passes, slots, cnt = torch.zeros(r), torch.zeros(P), 0
    for x, y in data.val_batches(n_seqs, B, device):
        with autocast(device):
            model(x, y, return_all=True)
        passes += torch.stack([v.float().cpu() for v in model.last_aux["pass_loss"]]) * x.numel()
        slots += model.last_aux["slot_loss"].float().cpu() * x.numel()
        cnt += x.numel()
    model.train()
    out = {f"val_loss_pass{k + 1}": v for k, v in enumerate((passes / cnt).tolist())}
    out.update({f"val_loss_slot{j}": v for j, v in enumerate((slots / cnt).tolist())})
    return out


@torch.no_grad()
def sample(model, tok, prompt: str, n_new: int, device, temperature=0.8, top_k=40, loops=None) -> str:
    model.eval()
    ids = tok.encode(prompt).ids
    eos = tok.token_to_id("<|eos|>")
    x = torch.tensor([ids], device=device)
    for _ in range(n_new):
        with autocast(torch.device(device).type):
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
    ap.add_argument("--max-tokens", type=int, help="override train.max_tokens (token datasets)")
    ap.add_argument("--max-bytes", type=int, help="override train.max_bytes (byte datasets)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: smoke tests only")
    ap.add_argument("--seed", type=int, help="override train.seed")
    ap.add_argument("--lr", type=float, help="override train.lr (peak LR; recorded in the run config)")
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
    if args.max_bytes:
        tc["max_bytes"] = args.max_bytes
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.lr is not None:
        tc["lr"] = args.lr
    overrides = {k: v for k, v in (("max_tokens", args.max_tokens or None), ("max_bytes", args.max_bytes or None),
                                   ("seed", args.seed), ("lr", args.lr)) if v is not None}
    if args.device != "cuda":
        overrides["device"] = args.device
    device = args.device
    cuda = device == "cuda"
    torch.manual_seed(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    data = TokenData(tc["dataset"], tc["seq_len"], tc["seed"])
    budget_key = "max_bytes" if data.unit == "bytes" else "max_tokens"
    other_key = "max_tokens" if budget_key == "max_bytes" else "max_bytes"
    if other_key in tc or budget_key not in tc:
        raise SystemExit(f"dataset {tc['dataset']} counts {data.unit}: the budget must be train.{budget_key} "
                         f"(found {sorted(k for k in tc if k.startswith('max_'))})")
    model, mc = build_model(cfg, data.meta["vocab_size"], tc["seq_len"])
    model = model.to(device)
    fwd_model = torch.compile(model) if args.compile else model
    pc = model.param_counts()
    looped = len(model.core) > 0
    loop_rng = random.Random(tc["seed"] + 1)
    patched = is_patched(model)
    sample_passes = patched and mc.refine_max > mc.refine_min
    pass_rng = random.Random(tc["seed"] + 2)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), fused=cuda)

    tokens_per_step = tc["batch_size"] * tc["seq_len"]   # positions: tokens, or bytes for byte datasets
    total_steps = args.bench or math.ceil(tc[budget_key] / tokens_per_step)
    unit = "bytes" if data.unit == "bytes" else "tok"
    extra = f" | patched P={mc.patch_size} r={mc.refine_passes} {mc.decoder}" if patched else ""
    print(f"params: total {pc['total']/1e6:.2f}M non-emb {pc['non_embedding']/1e6:.2f}M | "
          f"steps {total_steps} x {tokens_per_step} {unit} | looped={looped}{extra}", flush=True)

    run = None
    if not args.bench:
        cfg_name = str(cfg_path.relative_to(ROOT) if cfg_path.is_relative_to(ROOT) else cfg_path)
        run_cfg = {"config_file": cfg_name, "arch": arch, "cli_overrides": overrides,
                   "model": mc.to_dict(), "train": tc,
                   "params": pc, "flops_per_token_at_default_loops": model.flops_per_token(tc["seq_len"]),
                   "compile": args.compile, "dataset_meta": {k: data.meta.get(k) for k in ("name", "source", "vocab_size", "unit")},
                   "position_unit": data.unit}
        run = Run(cfg["experiment"], config=run_cfg, seed=tc["seed"], require_clean=args.require_clean)
        print(f"run: {run.name}", flush=True)

    t_start = time.time()
    tokens = flops = 0.0
    t_log, tok_log = time.time(), 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(total_steps):
            lr = lr_at(step, total_steps, tc)
            for gp in opt.param_groups:
                gp["lr"] = lr
            r = loop_rng.randint(mc.loops_min, mc.loops_max) if looped else None
            npass = pass_rng.randint(mc.refine_min, mc.refine_max) if sample_passes else None
            kw = {"passes": npass} if npass is not None else {}
            x, y = data.batch(tc["batch_size"], device)
            with autocast(device):
                _, loss = fwd_model(x, y, loops=r, **kw)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)

            tokens += tokens_per_step
            flops += model.flops_per_token(tc["seq_len"], loops=r, **kw) * tokens_per_step
            tok_log += tokens_per_step
            if step % 10 == 0 or step == total_steps - 1:
                if cuda:
                    torch.cuda.synchronize()
                dt = time.time() - t_log
                tps = tok_log / dt if dt > 0 else 0.0
                t_log, tok_log = time.time(), 0
                m = {"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": gnorm.item(),
                     "tokens": tokens, "flops": flops, "perf/tokens_per_s": tps}
                if looped:
                    m["train/loops"] = r
                if data.unit == "bytes":
                    m["bytes"] = tokens
                    m["token_equiv"] = tokens / BYTES_PER_TOKEN
                if patched and model.last_aux:
                    for k, v in zip(model.last_aux["pass_ids"], model.last_aux["pass_loss"]):
                        m[f"train/loss_pass{k + 1}"] = v.item()
                    if npass is not None:
                        m["train/passes"] = npass
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
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
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
        if patched:
            summary.update(evaluate_detail(model, data, None, tc["batch_size"], device))
            if mc.refine_shared:
                for p_eval in tc.get("eval_passes", []):
                    summary[f"val_loss_passes{p_eval}"] = evaluate(model, data, None, tc["batch_size"], device,
                                                                   passes=p_eval)
        if data.unit == "bytes":
            summary["bytes"] = tokens
            summary["token_equiv"] = tokens / BYTES_PER_TOKEN
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
        if data.unit == "bytes":
            tok = ByteTokenizer(data.meta.get("eos_id", 0))
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(DATA / "datasets" / tc["dataset"] / "tokenizer.json"))
        torch.manual_seed(0)
        n_new = 160 * (4 if data.unit == "bytes" else 1)   # about the same text length in bytes
        texts = [sample(model, tok, p, n_new, device) for p in PROMPTS]
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
