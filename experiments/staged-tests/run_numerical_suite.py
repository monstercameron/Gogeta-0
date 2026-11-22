"""Run the whole CPU numerical-verification suite (and, on a GPU box, the GPU one), optionally stamping the env fingerprint.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/staged-tests/run_numerical_suite.py [--stamp] [--gpu]

Order: API contracts first (they fail loudly on a signature change), then reference/known-answer/gradcheck/equivalence, then the
error-budget tests, then the mutation controls. --gpu additionally runs gpu_numerical_suite.py (needs the GPU slot; <= 10 min).
--stamp (only if everything passed) writes env_fingerprint.json with the suite result, which is what env_fingerprint.py checks.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from _harness import HERE

CPU_MODULES = ["test_api_contract", "test_reference_blocks", "test_error_budget", "test_mutation_controls"]


def run(mod: str, env=None) -> tuple[int, str]:
    t0 = time.time()
    p = subprocess.run([sys.executable, str(HERE / f"{mod}.py")], capture_output=True, text=True, env=env, cwd=str(HERE))
    out = p.stdout + p.stderr
    n_pass = sum(1 for ln in out.splitlines() if ln.startswith("PASS"))
    bad = [ln for ln in out.splitlines() if ln.startswith(("FAIL", "ERROR"))]
    print(f"{mod:28s} {'OK ' if p.returncode == 0 else ('SKIP' if p.returncode == 77 else 'FAIL')}  {n_pass} passed, {len(bad)} failed  ({time.time() - t0:.0f}s)")
    for ln in bad:
        print("    ", ln)
    return p.returncode, out


def main(argv):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if "--gpu" not in argv else os.environ.get("CUDA_VISIBLE_DEVICES", ""),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
    cpu_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
    results, counts = {}, 0
    for m in CPU_MODULES:
        rc, out = run(m, cpu_env)
        results[m] = rc == 0
        counts += sum(1 for ln in out.splitlines() if ln.startswith("PASS"))
    gpu_ok = None
    if "--gpu" in argv:
        rc, _ = run("gpu_numerical_suite", env)
        gpu_ok = None if rc == 77 else rc == 0     # 77 = SKIP (no CUDA): never counted as a pass
    ok = all(results.values()) and gpu_ok is not False
    print("SUITE", "PASS" if ok else "FAIL", f"({counts} CPU checks)")
    if ok and "--stamp" in argv:
        import env_fingerprint as EF
        EF.record({"cpu_suite": f"{counts} checks pass", "gpu_suite": ("pass" if gpu_ok else None)})
        print("stamped env_fingerprint.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
