"""SQ376 (descriptive, not registered): the goal-6 data gate from existing Phase 2b logs.

Is the 14M E1/E10 model still improving by more than 1% between 100M and 200M training tokens (roughly one epoch of v2's ~0.2B tokens)?
Source: the in-training eval lines `eval step N: val/loss X` (128 val sequences of 1,024 tokens, every 250 steps, 32,768 tokens/step) in the launcher logs
under ~/fusiondance-data/logs and the fleet copies, for runs whose log header shows steps 6104 (= 200M tokens) and experiment E1 or E10.
Gain = 100 * (1 - L(200M) / L(100M)) with L(100M) = the eval at step 3000 (98.3M tokens; the nearest eval at or before 100M) and L(200M) = the last eval
(step 6000 = 196.6M, plus the end-of-run summary val_loss when present). Reported per seed and run, n, mean, sd.
Caveat that matters: every run uses a cosine schedule to 10% LR over its OWN 200M tokens, so the 100M point is mid-schedule at about 55% of peak LR: the gain
includes annealing and overstates how much a longer-than-1-epoch run would still improve; the in-training val/loss is on 128 sequences (noisy at the 0.5% level),
v1 data (val contains the near-duplicates K9 found), not clean80. Nothing is re-scored here (no checkpoints at 100M exist).

Run (WSL): python experiments/checks/sq376_data_gate.py -> results/sq376.json
"""
import glob
import json
import os
import re
import statistics as st
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOME = os.path.expanduser("~/fusiondance-data")


def parse(path):
    txt = open(path, errors="replace").read()
    run = re.search(r"^run: (([A-Za-z0-9]+)-\d+-\d+-s(\d+))", txt, re.M)
    steps = re.search(r"steps (\d+) x (\d+) tok", txt)
    if not (run and steps):
        return None
    cmd = re.search(r"=== cmd: (.*)", txt)
    lr = re.search(r"--lr (\S+)", cmd.group(1)) if cmd else None
    if lr is None:  # the 5090 fleet logs carry no header: take the lr from the run's meta.json
        for mp in glob.glob(f"{HOME}/cloud/fleet/*/fusiondance-data/runs/{run.group(1)}/meta.json") + glob.glob(f"{HOME}/runs/{run.group(1)}/meta.json"):
            lrv = json.load(open(mp))["config"]["train"]["lr"]
            break
        else:
            lrv = None
    else:
        lrv = float(lr.group(1))
    evals = {int(a): float(b) for a, b in re.findall(r"eval step (\d+): val/loss ([0-9.]+)", txt)}
    return {"exp": run.group(2), "seed": int(run.group(3)), "run": run.group(1), "lr": lrv, "steps": int(steps.group(1)), "tok_per_step": int(steps.group(2)),
            "evals": evals, "finished": "done:" in txt, "log": path, "group": "p2b_fleet" if os.path.basename(path).startswith("p2b-") else "launcher"}


def main():
    paths = sorted(set(glob.glob(f"{HOME}/logs/*.log") + glob.glob(f"{HOME}/cloud/fleet/*/fusiondance-data/logs/*.log")))
    runs = {}
    for p in paths:
        r = parse(p)
        if r and r["exp"] in ("E1", "E10") and r["steps"] == 6104 and r["finished"] and 3000 in r["evals"] and max(r["evals"]) >= 5750 and not (r["lr"] and r["lr"] in (1e-3, 2e-3, 1.2e-2)):
            runs[r["run"]] = r
    rows = []
    for r in sorted(runs.values(), key=lambda r: (r["group"], r["exp"], r["seed"], r["lr"] or 0)):
        last = max(r["evals"])
        l100, l200 = r["evals"][3000], r["evals"][last]
        rows.append({"group": r["group"], "exp": r["exp"], "run": r["run"], "seed": r["seed"], "lr": r["lr"], "eval_step_100M": 3000, "eval_step_last": last,
                     "tokens_100M": 3000 * r["tok_per_step"], "tokens_last": last * r["tok_per_step"], "val_loss_100M": l100, "val_loss_last": l200,
                     "gain_pct": 100 * (1 - l200 / l100), "gain_nats": l100 - l200,
                     "curve": {str(k): v for k, v in sorted(r["evals"].items()) if k in (1000, 2000, 3000, 4000, 5000, 5750, 6000)}})
    out = {"check": "SQ376", "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "status": "descriptive, not registered", "n_logs_scanned": len(paths), "runs": rows, "groups": {}}
    for grp, exp in ((g_, e_) for g_ in ("p2b_fleet", "launcher") for e_ in ("E1", "E10")):
        for lr in sorted({r["lr"] for r in rows if r["exp"] == exp and r["group"] == grp}, key=lambda x: x or 0):
            g = [r for r in rows if r["exp"] == exp and r["lr"] == lr and r["group"] == grp]
            gains = [r["gain_pct"] for r in g]
            out["groups"][f"{grp}/{exp}@{lr}"] = {"n": len(g), "seeds": [r["seed"] for r in g], "gain_pct": gains, "mean_gain_pct": st.mean(gains),
                                            "sd_gain_pct": st.stdev(gains) if len(g) > 1 else None,
                                            "mean_val_loss_100M": st.mean(r["val_loss_100M"] for r in g), "mean_val_loss_last": st.mean(r["val_loss_last"] for r in g),
                                            "all_above_1pct": all(x > 1 for x in gains)}
    json.dump(out, open(os.path.join(ROOT, "experiments", "checks", "results", "sq376.json"), "w"), indent=1)
    for k, v in out["groups"].items():
        print(k, "n", v["n"], "seeds", v["seeds"], "gain %", [round(x, 2) for x in v["gain_pct"]], "mean", round(v["mean_gain_pct"], 2))


if __name__ == "__main__":
    main()
