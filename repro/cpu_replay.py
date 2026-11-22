"""Cheap blind replay on CPU: rebuild a run from its manifest and reproduce the first training step(s).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python repro/cpu_replay.py <manifest> [--steps 1] [--micro 2]

What it checks, from the manifest alone (no GPU, no original logs needed except to compare):
  * seed handling, model init and data order: step 0 sees the same batch and the same weights, so the step-0 training
    loss must match the log's `step 0 loss` within a small tolerance (fp32 on CPU vs bf16 autocast on GPU: ~1e-2).
  * with --steps N (N > 1) it also takes N-1 AdamW steps with the same LR schedule and compares `step N-1` if the log
    printed it (only every 100th step is printed). N = 100 on CPU takes hours: that is the GPU "mid tier".
Mirrors gogeta/train.py's sampling code (`torch.randint(..., generator=g)`), which pins memory and so cannot run without
CUDA; if train.py's data code changes, the step-0 check here fails loudly rather than drifting silently.
Only attention-only configs (no fla/Triton kernels, e.g. E0) run on CPU. Micro-batching keeps RAM small (the 32 x 1024 x
8192 fp32 logits alone are 1 GB); the micro-batch losses are token-averaged, equal to the full batch in exact arithmetic.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "repro"))


def log_step_loss(log: pathlib.Path, step: int):
    for l in log.read_text(errors="replace").splitlines():
        m = re.match(rf"\s*step\s+{step}/\d+\s+loss\s+([\d.]+)", l)
        if m:
            return float(m.group(1))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--micro", type=int, default=2, help="sequences per micro-batch")
    ap.add_argument("--tol", type=float, default=0.02, help="|CPU fp32 - logged bf16 GPU| allowed at the compared step")
    a = ap.parse_args(argv)
    import numpy as np
    import torch
    from common import DATA
    from gogeta.model import FusionDance, ModelConfig
    from gogeta.train import lr_at

    m = json.loads(pathlib.Path(a.manifest).read_text())
    eff = m["config"]["effective"]
    if (eff.get("arch") or "fusiondance") != "fusiondance":
        print("cpu_replay supports arch=fusiondance only"); return 2
    tc, mcfg = eff["train"], eff["model"]
    if any(b not in ("attn", "swa") for k in ("prelude", "core", "coda") for b in mcfg.get(k, [])):
        print("this config uses fla (GDN/Mamba) blocks: they need CUDA/Triton; use the GPU tiers"); return 2
    d = DATA / "datasets" / tc["dataset"]
    train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    T, B = tc["seq_len"], tc["batch_size"]
    torch.manual_seed(tc["seed"])
    g = torch.Generator().manual_seed(tc["seed"])           # TokenData: generator seeded after manual_seed, same value
    mc = ModelConfig(**{**{k: v for k, v in mcfg.items() if k not in ("vocab_size", "max_seq_len")},
                        "vocab_size": mcfg["vocab_size"], "max_seq_len": T})
    model = FusionDance(mc)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=tc["lr"],
                            betas=(tc["beta1"], tc["beta2"]))
    total_steps = -(-tc["max_tokens"] // (B * T))
    losses = []
    for step in range(a.steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, total_steps, tc)
        ix = torch.randint(len(train) - T - 1, (B,), generator=g).tolist()
        buf = torch.from_numpy(np.stack([train[i:i + T + 1] for i in ix]).astype(np.int64))
        tot = 0.0
        for s in range(0, B, a.micro):
            x, y = buf[s:s + a.micro, :-1].contiguous(), buf[s:s + a.micro, 1:].contiguous()
            _, loss = model(x, y)
            (loss * x.size(0) / B).backward()
            tot += loss.item() * x.size(0) / B
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot)
        print(f"cpu step {step}: loss {tot:.4f}", flush=True)
    rc = 0
    lg = m["artifacts"].get("train_log")
    if isinstance(lg, dict):
        lp = DATA / "logs" / lg["name"]
        if lp.exists():
            for st in sorted({0, a.steps - 1}):
                ref = log_step_loss(lp, st)
                if ref is None:
                    print(f"step {st}: not printed in the log (only steps divisible by 100 are)"); continue
                ok = abs(ref - losses[st]) <= a.tol
                rc |= (not ok)
                print(f"step {st}: CPU fp32 {losses[st]:.4f} vs logged GPU bf16 {ref:.4f}  |d|={abs(ref-losses[st]):.4f} "
                      f"<= {a.tol}: {'OK' if ok else 'FAIL'}")
        else:
            print(f"log {lp} not available: nothing to compare against (replay losses printed above)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
