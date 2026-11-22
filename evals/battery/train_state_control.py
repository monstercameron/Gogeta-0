"""Positive control for the state-tracking probe: a tiny attention LM trained ON probe-style snippets only
(different seeds from the eval items: train seeds 1000+, eval seed 0), then scored with the SAME scorer.

It answers "is the probe solvable, and does the scoring pipeline (eos prefix, tokenisation boundary, choice
log-likelihood) work end to end?". It does NOT say anything about pretrained checkpoints (E0 is at chance on the
probe, see notes/eval-battery.md). Loss is on the answer tokens only. CPU-friendly (2 layers, d=128, ~1.2M params
besides the 8k embedding).

    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta-eval/bin/python -m evals.battery.train_state_control --steps 2500
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.battery.state_track import make_item  # noqa: E402
from evals.battery.score_state_track import bootstrap_ci, score_items  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--train-k", default="1,2,4"); ap.add_argument("--train-d", default="0,8,16")
    ap.add_argument("--eval-cells", default="assign:1:0,assign:4:16,assign:8:64,incr:2:0,incr:4:16")
    ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    from gogeta.hf import load_tokenizer
    from gogeta.model import FusionDance, ModelConfig
    tok = load_tokenizer(); eos = tok.eos_token_id
    cfg = ModelConfig(vocab_size=8192, d_model=a.d, n_heads=2, mlp_hidden=2 * a.d, prelude=["attn"] * a.layers,
                      core=[], coda=[], max_seq_len=1024)
    m = FusionDance(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    ks = [int(v) for v in a.train_k.split(",")]; ds = [int(v) for v in a.train_d.split(",")]
    rng = random.Random(a.seed); t0 = time.time(); m.train(); idx = 0
    for s in range(a.steps):
        lr = a.lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        seqs, tgts = [], []
        for _ in range(a.batch):
            mode = rng.choice(["assign", "incr"]); k = rng.choice(ks if mode == "assign" else [max(2, v) for v in ks])
            it = make_item(mode, k, rng.choice(ds), 1000 + a.seed, idx); idx += 1
            p = [eos] + tok.encode(it["prompt"], add_special_tokens=False)
            c = tok.encode(it["choices"][it["gold"]], add_special_tokens=False)
            seqs.append(p + c); tgts.append(len(c))
        L = max(map(len, seqs))
        x = torch.full((a.batch, L), tok.pad_token_id); y = torch.full((a.batch, L), -100)
        for r, (sq, nt) in enumerate(zip(seqs, tgts)):
            x[r, :len(sq)] = torch.tensor(sq); y[r, len(sq) - nt - 1:len(sq) - 1] = torch.tensor(sq[-nt:])
        loss = m(x, y)[1]; loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
        if s % 250 == 0: print(f"  step {s} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    m.eval()

    class W:  # adapter: the scorer calls model(input_ids=...).logits
        def __call__(self, input_ids, **kw):
            class O: pass
            o = O(); o.logits = m(input_ids)[0]; return o
    res = {"steps": a.steps, "train_s": round(time.time() - t0, 1), "cells": {}}
    from evals.battery.state_track import make_cell
    for c in a.eval_cells.split(","):
        mode, k, d = c.split(":")
        items = make_cell(mode, int(k), int(d), a.n_eval, 0)
        pick = score_items(W(), tok, items)
        hits = (pick == np.array([it["gold"] for it in items])).astype(float)
        acc, lo, hi = bootstrap_ci(hits)
        res["cells"][c] = {"acc": acc, "ci95": [lo, hi], "n": len(items)}
        print(f"CONTROL {c}: acc {acc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    if a.out:
        p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
