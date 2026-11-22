"""Standalone scorer for the state-tracking probe (same items as the lm-eval tasks gogeta_state_*).
Runs in ~/.venvs/gogeta (has fla, so GDN checkpoints work on GPU) or in ~/.venvs/gogeta-eval (attention-only, CPU).

    ~/.venvs/gogeta-eval/bin/python -m evals.battery.score_state_track --checkpoint <model.pt> \
        [--n 1100] [--cells assign:1:0,incr:4:16] [--device cpu|cuda] [--loops R] [--out file.json]

Per cell: accuracy (argmax of summed token log-prob of the choice given the snippet; prompt starts with <|eos|>
like pretraining docs), item-level percentile bootstrap 95% CI (2000 resamples, seed 0), and the share of
errors that picked a STALE value of the same variable (the signature of failing to track state, as opposed to
a random/other-variable error). Inference only. Deterministic on CPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.battery.state_track import GRID, N_PER_CELL, make_cell  # noqa: E402


def bootstrap_ci(hits: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), (n_boot, len(hits)))
    m = hits[idx].mean(1)
    return float(hits.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


@torch.no_grad()
def score_items(model, tok, items, device="cpu", loops=None, bs=16, max_len=1024):
    eos = tok.eos_token_id
    picks = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        seqs, spans, owner = [], [], []
        for j, it in enumerate(chunk):
            p = [eos] + tok.encode(it["prompt"], add_special_tokens=False)
            for c in it["choices"]:
                ci = tok.encode(c, add_special_tokens=False)
                ids = (p + ci)[-max_len:]
                seqs.append(ids); spans.append(len(ci)); owner.append(j)
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        for r, s in enumerate(seqs):                       # right padding is safe: causal model, we read earlier positions
            x[r, :len(s)] = torch.tensor(s)
        kw = {"loops": loops} if loops is not None else {}
        logits = model(input_ids=x.to(device), **kw).logits.float()
        lp = torch.log_softmax(logits, -1).cpu()
        sc = []
        for r, s in enumerate(seqs):
            n = spans[r]; T = len(s)
            tgt = torch.tensor(s[T - n:])
            sc.append(lp[r, T - n - 1:T - 1].gather(1, tgt[:, None]).sum().item())
        sc = np.array(sc).reshape(len(chunk), -1)
        picks.extend(sc.argmax(1).tolist())
    return np.array(picks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint"); ap.add_argument("--random-init")
    ap.add_argument("--n", type=int, default=N_PER_CELL)
    ap.add_argument("--cells", default="", help="comma list mode:k:distance (default: the full frozen grid)")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--loops", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    from gogeta.hf import from_gogeta_checkpoint, load_tokenizer     # no lm-eval import: runs in ~/.venvs/gogeta too
    if a.random_init:
        from evals.gogeta_lm import build_model                       # needs lm-eval (eval venv); wiring tests only
        model, prov = build_model(None, a.random_init, a.loops)
    else:
        model = from_gogeta_checkpoint(a.checkpoint, **({"loops": a.loops} if a.loops else {})).eval()
        kinds = set(model.config.prelude) | set(model.config.core) | set(model.config.coda)
        prov = {"checkpoint": a.checkpoint, "block_kinds": sorted(kinds), "loops": model.config.loops,
                "n_params": sum(t.numel() for t in model.parameters())}
    model.to(a.device); tok = load_tokenizer()
    cells = ([tuple(c.split(":")) for c in a.cells.split(",")] if a.cells
             else [(m, k, d) for m, cs in GRID.items() for k, d in cs])
    res = {"provenance": prov, "n_per_cell": a.n, "loops": a.loops, "cells": {}}
    for mode, k, d in cells:
        t0 = time.time()
        items = make_cell(mode, int(k), int(d), a.n, 0)
        pick = score_items(model, tok, items, a.device, a.loops, a.bs)
        gold = np.array([it["gold"] for it in items]); hits = (pick == gold).astype(float)
        acc, lo, hi = bootstrap_ci(hits)
        wrong = [(it, p) for it, p in zip(items, pick) if p != it["gold"]]
        stale_share = (sum(it["choices"][p] in it["stale"] for it, p in wrong) / len(wrong)) if wrong else None
        res["cells"][f"{mode}_k{k}_d{d}"] = {"acc": acc, "ci95": [lo, hi], "n": len(items), "chance": 0.25, "stale_share_of_errors": stale_share,
                                             "mean_lines": float(np.mean([it["lines"] for it in items])),
                                             "secs": round(time.time() - t0, 1)}
        print(f"{mode:6s} k={k} d={d:>3}: acc {acc:.3f} [{lo:.3f},{hi:.3f}] n={len(items)} ({time.time()-t0:.0f}s)", flush=True)
    if a.out:
        p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
