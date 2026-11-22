"""Check the tracking setup end to end with a fake training curve (no GPU, no model).

    ~/.venvs/gogeta/bin/python tools/smoke_tracking.py

Logs to the separate Trackio project "gogeta-0-smoke", so real results stay clean.
"""
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gogeta.tracking import Run  # noqa: E402

random.seed(0)
cfg = {"d_model": 384, "loops": 4, "ternary": False, "lr": 3e-3}
with Run("smoke", config=cfg, seed=0, project="gogeta-0-smoke") as run:
    for step in range(200):
        loss = 4.0 * math.exp(-step / 60) + 1.6 + random.gauss(0, 0.03)
        run.log({"train/loss": loss, "tokens": (step + 1) * 32_768}, step=step)
    run.summary["val_loss"] = round(loss, 4)

print(f"run:     {run.name}")
print(f"git:     {run.git['describe']} (dirty={run.git['dirty']})")
print(f"folder:  {run.dir}")
print(f"tracker: {run.backend}")
