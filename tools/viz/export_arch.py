#!/usr/bin/env python
"""Export every config's architecture description to docs/data/architectures.json.

GitHub Pages is static, so the 3D page (docs/architecture.html) cannot call the live viz server. This
script builds each model from the REAL code (it imports tools/viz/server.py's `describe`, so nothing is
re-counted here) and writes a snapshot. Re-run it after any change to gogeta/model.py or configs/*.json.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python tools/viz/export_arch.py
    bash tools/launch_cpu_job.sh --name arch-export -- ~/.venvs/gogeta/bin/python tools/viz/export_arch.py

CPU only; never touches the GPU.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="" and thread caps on import)

OUT = server.REPO / "docs" / "data" / "architectures.json"
LOOP_RANGE = range(1, 13)  # the 3D page's loops slider


def git_describe() -> str:
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=server.REPO,
                              capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    mt_key = (0, server.MODEL_PY.stat().st_mtime_ns)
    models, errors = [], []
    for item in server.list_configs():
        name = item["name"]
        if "error" in item:
            errors.append(f"{name}: {item['error']}")
            continue
        try:
            d = server.describe(name, None, mt_key)
            d["config_file"] = f"configs/{name}.json"
            d["title"] = item.get("title")
            d["status"] = item.get("status")
            d["val_loss"] = item.get("val_loss")
            if d["looped"]:
                # FLOPs / effective depth for every loop count on the slider (same code path, different r)
                d["by_loops"] = []
                for r in LOOP_RANGE:
                    x = server.describe(name, r, mt_key)
                    d["by_loops"].append({"loops": r, "effective_depth": x["effective_depth"],
                                          "train_per_token": x["flops"]["train_per_token"],
                                          "infer_per_token": x["flops"]["infer_per_token"],
                                          "applications": x["applications"]})
            models.append(d)
            print(f"ok  {name}: total={d['params']['total']:,}", flush=True)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    if errors:
        print("FAILED:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    out = {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "git": git_describe(), "ctx_stops": server.CTX_STOPS, "bytes_per_elem": server.BYTES_BF16,
           "models": models}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KiB, {len(models)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
