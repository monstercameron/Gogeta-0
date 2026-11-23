"""Validate the CPU inference path (sq339_cpu.py: fp32, GDN via the chunkwise torch kernel) against the GPU numbers.

For seed 0 of cells 000, 100, 010, 111 at their best LR: score ALL 2,333 clean80 docs (window 1024, K8's protocol) in fp32 on
CPU and compare with the GPU/bf16 clean80 bits/char in sq263_runs.json (fleet scorer, K8 under bf16 autocast). Also the
kernel self-check (chunkwise vs the per-token fp32 reference). Descriptive engineering check, not registered.

Run (WSL): PYTHONPATH=. python experiments/checks/sq339_cpu_validate.py -> experiments/checks/results/sq339_cpu_validation.json
"""
import json
import math
import sys
import time

import sq339_cpu as C


def main():
    log = C.make_log(str(C.RES / "sq339_cpu_validate.log"))
    runs = C.best_runs()
    gpu = {r["run"]: r["clean80"] for r in json.load(open(C.RES / "sq263_runs.json"))["runs"]}
    docs = C.clean80_docs()
    log(f"clean80 docs: {len(docs)}")
    out = {"check": "SQ339-cpu-validation", "git": C.git_commit(), "n_docs": len(docs), "protocol": "K8 windowing, window 1024, fp32 CPU",
           "kernel_selfcheck": C.kernel_selfcheck(), "cells": {}}
    for cell in ("000", "100", "010", "111"):
        run = runs[cell][0]
        t = time.time()
        m = C.load_model(run)
        sc = C.score_docs(C.fwd_default(m), docs)
        b = C.bpc(sc)
        out["cells"][cell] = {"run": run, "cpu_bpc": b, "gpu_bpc": gpu[run], "rel_gap_pct": 100 * (b / gpu[run] - 1),
                              "tokens": sum(sc["n_tok"]), "sec": time.time() - t}
        log(f"{cell} {run}: cpu {b:.5f} gpu {gpu[run]:.5f} gap {100 * (b / gpu[run] - 1):+.3f}%")
        json.dump(out, open(C.RES / "sq339_cpu_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
