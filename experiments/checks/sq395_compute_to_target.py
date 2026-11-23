"""SQ395 (descriptive, not registered; zero GPU): compute-to-target from the existing in-training eval curves.

How many tokens does A2s need to reach E27's loss, and is that ratio stable along training or an endpoint effect (C209's 17% headline)?
Source: the launcher logs of the 200M-token runs (fleet pulls and ~/fusiondance-data/logs): `eval step N: val/loss X` every 250 steps (128 val sequences of the
run's dataset, code-py-v2 for both; same eval set for both arms), 32,768 tokens/step. A2s and E27 seeds 2, 3, 4 (paired by seed: same data order).
For each seed and each E27 eval step t (1000..last): L = E27 val loss at t; t* = the first A2s step at which its (piecewise-linear interpolated) val loss falls to L; ratio(t) = t*/t
(< 1: A2s is ahead, i.e. needs fewer tokens). Also the endpoint: tokens for A2s to reach E27's FINAL eval loss. FLOPs ratio: A2s costs about 3.79e16 / 3.42e16 = 1.108 x E27's per token,
so ratio x 1.108 is the FLOP-matched version (total training FLOPs from each run's FINAL line). Caveat: both arms use the SAME cosine schedule to 200M tokens, so an early
step of a run is mid-schedule at a high LR; the ratio compares equal-step states of equal-schedule runs, not equal-compute independently tuned runs. In-training val loss is nats/token
on 128 sequences (noisy at the 0.5% level), not clean80 bits/char.

Run (WSL): python experiments/checks/sq395_compute_to_target.py -> results/sq395.json
"""
import glob
import json
import os
import re
import statistics as st
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOME = os.path.expanduser("~/fusiondance-data")
TOK_PER_STEP = 32768


def parse(path):
    txt = open(path, errors="replace").read()
    run = re.search(r"^run: (([A-Za-z0-9]+)-\d+-\d+-s(\d+))", txt, re.M)
    steps = re.search(r"steps (\d+) x (\d+) tok", txt)
    if not (run and steps) or "done:" not in txt:
        return None
    lr = re.search(r"--lr (\S+)", txt)
    fl = re.search(r'^FINAL .*?"flops": ([0-9.e+]+)', txt, re.M)
    return {"flops_total": float(fl.group(1)) if fl else None, "exp": run.group(2), "seed": int(run.group(3)), "run": run.group(1), "lr": float(lr.group(1)) if lr else None, "steps": int(steps.group(1)),
            "log": path, "evals": {int(a): float(b) for a, b in re.findall(r"eval step (\d+): val/loss ([0-9.]+)", txt)}}


def first_crossing(curve, target):
    """First step at which the piecewise-linear curve (step -> loss) is <= target, or None."""
    items = sorted(curve.items())
    for (s0, l0), (s1, l1) in zip(items, items[1:]):
        if l1 <= target < l0 or l0 <= target:
            if l0 <= target:
                return float(s0)
            return s0 + (l0 - target) / (l0 - l1) * (s1 - s0)
    return None


def main():
    paths = sorted(set(glob.glob(f"{HOME}/logs/*.log") + glob.glob(f"{HOME}/cloud/fleet/*/fusiondance-data/logs/*.log")))
    runs = {}
    for p in paths:
        r = parse(p)
        if r and r["exp"] in ("A2s", "E27") and r["steps"] == 6104 and len(r["evals"]) >= 20 and "lrhi" not in r["log"] and "lrlo" not in r["log"]:
            runs.setdefault((r["exp"], r["seed"]), r)
    out = {"check": "SQ395", "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "status": "descriptive, not registered", "runs": {f"{e}_s{s}": r["run"] for (e, s), r in runs.items()}, "per_seed": {}}
    seeds = sorted(s for (e, s) in runs if e == "A2s" and ("E27", s) in runs)
    flops = {e: st.mean(runs[(e, s)]["flops_total"] for s in seeds) for e in ("A2s", "E27")}
    out["train_flops_total_mean"] = flops
    fr = flops["A2s"] / flops["E27"]
    out["flops_ratio_A2s_over_E27"] = fr
    for s in seeds:
        a, e = runs[("A2s", s)]["evals"], runs[("E27", s)]["evals"]
        last = max(e)
        rows = []
        for t in sorted(e):
            if t < 1000:
                continue
            ts = first_crossing(a, e[t])
            rows.append({"step": t, "tokens": t * TOK_PER_STEP, "e27_val": e[t], "a2s_val_at_same_step": a.get(t), "a2s_step_to_reach": ts,
                         "ratio_tokens": (ts / t) if ts else None})
        out["per_seed"][str(s)] = {"e27_final_step": last, "e27_final_val": e[last], "a2s_final_val": a[max(a)], "final_gap_pct": 100 * (a[max(a)] / e[last] - 1),
                                   "a2s_tokens_to_reach_e27_final": (first_crossing(a, e[last]) or float("nan")) * TOK_PER_STEP,
                                   "ratio_at_endpoint": (first_crossing(a, e[last]) / last) if first_crossing(a, e[last]) else None, "curve": rows}
    # stability: mean over seeds of ratio at several fractions
    steps = sorted({r["step"] for s in out["per_seed"].values() for r in s["curve"]})
    out["ratio_by_step_mean"] = {}
    for t in steps:
        v = [r["ratio_tokens"] for s in out["per_seed"].values() for r in s["curve"] if r["step"] == t and r["ratio_tokens"]]
        if len(v) == len(seeds):
            out["ratio_by_step_mean"][str(t)] = {"mean": st.mean(v), "n": len(v), "per_seed": v, "flop_matched_mean": st.mean(v) * fr if fr else None}
    json.dump(out, open(os.path.join(ROOT, "experiments", "checks", "results", "sq395.json"), "w"), indent=1)
    print("seeds", seeds, "flops ratio", fr)
    for s, r in out["per_seed"].items():
        print("seed", s, "final gap %", round(r["final_gap_pct"], 2), "tokens to E27 final", round(r["a2s_tokens_to_reach_e27_final"] / 1e6, 1), "M", "endpoint ratio", r["ratio_at_endpoint"])
    for t, v in out["ratio_by_step_mean"].items():
        print(t, round(v["mean"], 3), [round(x, 3) for x in v["per_seed"]])


if __name__ == "__main__":
    main()
