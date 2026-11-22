"""Re-validation trigger (A6, C192): fail if the numerical environment differs from the one the numerical suite was last run in.

    ~/.venvs/gogeta/bin/python experiments/staged-tests/env_fingerprint.py            # check (exit 1 on any difference)
    ~/.venvs/gogeta/bin/python experiments/staged-tests/env_fingerprint.py --show     # print the current fingerprint
    ~/.venvs/gogeta/bin/python experiments/staged-tests/run_numerical_suite.py --stamp  # run everything; on all-pass, re-record

What is fingerprinted: python / torch / CUDA / triton / fla / numpy / transformers / trl / lm_eval versions; the full
inspect.signature of every external kernel in CONTRACT (test_api_contract.py) and the SDPA signature parsed from torch's docstring;
a sha256 of the SOURCE of the four fla op wrappers (a same-version local patch changes it); the platform. The committed
env_fingerprint.json also records WHEN and HOW the suite last passed against that fingerprint. A fingerprint recorded without a
suite pass ("validated": null) fails the check on purpose: bumping a dependency must be followed by running the suite.
The GPU model is recorded for information only (a different GPU does not fail the check; it triggers a warning to run the GPU item).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as md
import inspect
import json
import platform
import sys
import warnings

from _harness import HERE

warnings.filterwarnings("ignore")
FILE = HERE / "env_fingerprint.json"
PACKAGES = ["torch", "triton", "flash-linear-attention", "fla-core", "numpy", "transformers", "trl", "lm_eval", "tokenizers"]
SOURCE_HASHED = ["fla.ops.gated_delta_rule.chunk_gated_delta_rule", "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule",
                 "fla.ops.kda.chunk_kda", "fla.ops.kda.fused_recurrent_kda"]


def _resolve(dotted):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001
            continue
        for a in parts[i:]:
            obj = getattr(obj, a)
        return obj
    raise ImportError(dotted)


def current() -> dict:
    import test_api_contract as AC   # the contract keys are the single source of truth for "which functions we call"
    import torch
    fp = {"python": platform.python_version(), "platform": platform.system() + "-" + platform.machine(),
          "torch_cuda": torch.version.cuda, "packages": {}, "signatures": {}, "source_sha256": {}}
    for p in PACKAGES:
        try:
            fp["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            fp["packages"][p] = None
    for key in AC.CONTRACT:
        try:
            fp["signatures"][key] = str(inspect.signature(_resolve(key)))
        except Exception as e:  # noqa: BLE001
            fp["signatures"][key] = f"<unavailable: {type(e).__name__}>"
    fp["signatures"]["torch.nn.functional.scaled_dot_product_attention"] = ", ".join(AC._sdpa_signature())
    for key in SOURCE_HASHED:
        try:
            fp["source_sha256"][key] = hashlib.sha256(inspect.getsource(_resolve(key)).encode()).hexdigest()[:16]
        except Exception as e:  # noqa: BLE001
            fp["source_sha256"][key] = f"<unavailable: {type(e).__name__}>"
    return fp


def digest(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


def gpu_info() -> dict | None:
    import torch
    if not torch.cuda.is_available():
        return None
    return {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}


def diff(a: dict, b: dict, path="") -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff(va, vb, f"{path}{k}.")
        elif va != vb:
            out.append(f"{path}{k}:\n      committed: {va}\n      installed: {vb}")
    return out


def record(validated_by: dict | None):
    import datetime
    fp = current()
    FILE.write_text(json.dumps({"fingerprint": fp, "digest": digest(fp), "gpu": gpu_info(),
                                "validated": validated_by and {**validated_by, "date": datetime.date.today().isoformat(),
                                                               "digest": digest(fp)}}, indent=1) + "\n")
    return fp


def check() -> int:
    if not FILE.exists():
        print("FAIL: no committed env_fingerprint.json. Run run_numerical_suite.py --stamp once.")
        return 1
    committed = json.loads(FILE.read_text())
    fp = current()
    problems = diff(committed["fingerprint"], fp)
    v = committed.get("validated")
    if problems:
        print("FAIL: the numerical environment changed since the suite was last run. Differences:")
        print("  " + "\n  ".join(problems))
        print("\nNext: (1) read the diff, (2) python experiments/staged-tests/run_numerical_suite.py (CPU) and the GPU item "
              "A6-gpu-numerics, (3) audit call sites if a signature moved, (4) run_numerical_suite.py --stamp to re-record.")
        return 1
    if not v or v.get("digest") != committed["digest"]:
        print("FAIL: fingerprint recorded but the numerical suite has not been stamped as passing against it.")
        return 1
    g = gpu_info()
    if g and committed.get("gpu") and g != committed["gpu"]:
        print(f"WARN: GPU differs ({g} vs {committed['gpu']}): run the GPU numerical item (gpu_numerical_suite.py).")
    if g and not v.get("gpu_suite"):
        print("WARN: a GPU is present but the GPU numerical suite has never been stamped for this fingerprint.")
    print(f"OK: environment matches the committed fingerprint {committed['digest']} "
          f"(suite validated {v['date']}: cpu={v.get('cpu_suite')}, gpu={v.get('gpu_suite')})")
    return 0


if __name__ == "__main__":
    if "--show" in sys.argv:
        print(json.dumps(current(), indent=1))
        raise SystemExit(0)
    if "--record" in sys.argv:       # records WITHOUT a suite stamp -> the check will keep failing until the suite is stamped
        record(None)
        print("recorded (unvalidated). Now run run_numerical_suite.py --stamp.")
        raise SystemExit(0)
    raise SystemExit(check())
