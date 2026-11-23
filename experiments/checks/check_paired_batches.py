"""Empirical check that same-seed arms draw identical training batches (quick review 2026-11-23 02:44, data engineer).

Interleaves model construction (which consumes the global torch RNG) with batch draws for two different arms (E1
hybrid and the E10 HF-Llama control), at the same seed, and compares sha256 of the token ids for the first 20 steps.
Run (WSL, CPU):  CUDA_VISIBLE_DEVICES= python experiments/checks/check_paired_batches.py
"""
import hashlib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from gogeta import train as T  # noqa: E402
from gogeta.model import ModelConfig, FusionDance  # noqa: E402


def draws(seed, build_model, steps=20, B=32, L=1024):
    torch.manual_seed(seed)
    data = T.TokenData("code-py-v1", L, seed)
    build_model()  # consumes the global RNG, as training does
    out = []
    for _ in range(steps):
        # identical to TokenData.batch (gogeta/train.py:65-66) minus pin_memory, which needs CUDA
        ix = torch.randint(len(data.train) - data.T - 1, (B,), generator=data.g).tolist()
        x = torch.from_numpy(__import__("numpy").stack([data.train[i:i + data.T + 1] for i in ix]).astype("int64"))
        torch.randn(1000)  # simulate other global-RNG use between steps (dropout, sampling)
        out.append(hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16])
    return out


def e1():
    cfg = json.load(open(os.path.join(ROOT, "configs", "e1_hybrid.json")))["model"]
    FusionDance(ModelConfig(**cfg))


def other():
    torch.randn(5_000_000)  # a different model would consume a different amount of global RNG


res = {}
for seed in (0, 1):
    a, b = draws(seed, e1), draws(seed, other)
    res[seed] = {"identical_all_steps": a == b, "first": a[:3]}
c0 = draws(0, e1)
res["different_seeds_differ"] = c0 != draws(1, e1)
print(json.dumps(res, indent=1))
assert all(res[s]["identical_all_steps"] for s in (0, 1)) and res["different_seeds_differ"]
print("PAIRED BATCHES OK")
