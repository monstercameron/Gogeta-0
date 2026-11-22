"""Shared plumbing for the staged numerical-verification tests (A6).

Works from experiments/staged-tests/ or from tests/ (after the main agent moves it): the repo root is found
by walking up from this file until gogeta/model.py exists, and this directory is put on sys.path.
Tests are plain `test_*` functions with asserts (pytest-compatible); `run_module(globals())` is a tiny runner so
`python <file>.py` works without pytest. CPU only unless a test says otherwise.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "gogeta" / "model.py").exists())
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def run_module(ns: dict, only: str | None = None) -> int:
    tests = [(k, v) for k, v in ns.items() if k.startswith("test_") and callable(v)]
    if only is None and len(sys.argv) > 1:
        only = sys.argv[1]
    bad = 0
    for name, fn in tests:
        if only and only not in name:
            continue
        t0 = time.time()
        try:
            out = fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)" + (f"  {out}" if isinstance(out, str) else ""))
        except AssertionError as e:
            bad += 1
            print(f"FAIL  {name}  ({time.time() - t0:.1f}s)\n      " + str(e).replace("\n", "\n      "))
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"ERROR {name}\n" + traceback.format_exc())
    print("ALL OK" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def maxerr(a, b) -> float:
    return (a.double() - b.double()).abs().max().item()
