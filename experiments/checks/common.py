"""Shared helpers for the verification checks (AGENTS.md section 0).

Each check script: states PREDICTION and CRITERION as constants (committed before running), measures,
applies the criterion, prints PASS/FAIL, and writes experiments/checks/results/<id>.json.
Run from the repo root inside WSL:  ~/.venvs/gogeta/bin/python experiments/checks/k1_memory_vs_length.py
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import random
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "experiments" / "checks" / "results"

import numpy as np  # noqa: E402
import torch  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def bench_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def loglog_slope(xs, ys) -> float:
    """Least-squares slope of log(y) vs log(x): the scaling exponent."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])


def cosine_lr(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def train_steps(model, batch_fn, steps, lr=1e-3, warmup=100, wd=0.1, loops=None, log_every=0):
    """Minimal bf16 AdamW loop. batch_fn(step) -> (x, y). Returns the list of losses."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    model.train()
    losses = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(s, steps, lr, warmup)
        x, y = batch_fn(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, loops=loops)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if log_every and s % log_every == 0:
            print(f"    step {s:5d} loss {losses[-1]:.4f}", flush=True)
    return losses


def save(check_id: str, claims: list[str], prediction: str, criterion: str, verdicts: dict, data: dict,
         started: float, filename: str | None = None) -> pathlib.Path:
    """filename: optional results file stem when it differs from the check id (e.g. id SQ6 -> sq_06.json)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rec = {
        "check": check_id, "claims": claims,
        "prediction": prediction, "criterion": criterion,
        "verdicts": verdicts, "data": data,
        "ran_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - started, 1),
        "git_commit": _git("rev-parse", "HEAD"), "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "torch": torch.__version__, "device": dev,
    }
    out = RESULTS / f"{filename or check_id}.json"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"\n== {check_id} verdicts ==")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"  -> {out.relative_to(ROOT)}  ({rec['runtime_s']} s)")
    return out
