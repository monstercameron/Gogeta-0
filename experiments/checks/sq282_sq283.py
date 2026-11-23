"""SQ282 (LR-fair depth vs width) and SQ283 (C197 seed replication), registered in experiments/checks/PREDICTIONS.md,
"Fleet wave 6, part 4". Metric: clean80 bpc on v1 val, RTX 5090.

SQ282: E32 (14 unshared blocks), E34 (5 wide blocks, ~same unique params/FLOPs), E2 (looped) at lr {3e-3, 6e-3, 1.2e-2},
       seeds 0-1, 20M tokens. Best LR per arm = min 2-seed mean (G3).
       W1: E32 beats E34 by >= 5% at best LRs.  W2: E34 at best LR beats E2 at best LR.
       SQ282's own runs only (dispatched 04:3x-04:5x UTC); the later SQ308 E32 runs at the same LR are excluded.
SQ283: E32 and E2 at 200M, lr 6e-3, seeds 0-3 (seed 0 = the C197 runs).
       R1: E32 beats E2 by >= 10% on the 4-seed means.  R2: E32 beats the E1 7-seed mean (Phase 2b-5090) by >= 5%.
Run -> lr/tokens/seed come from run meta.json (experiments/checks/results/sq282_sq283_runmap.json).

Run: python experiments/checks/sq282_sq283.py -> experiments/checks/results/sq282_sq283.json
"""
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
T3 = 3.182


def bpc(run):
    return json.load(open(os.path.join(DL, run + ".json")))["bpc"]["clean80"]


def main():
    runs = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "sq282_sq283_runmap.json")))["runs"]
    sweep, long = {}, {}
    for run, m in runs.items():
        arm = run.split("-")[0]
        hhmm = run.split("-")[2][:4]
        if m["max_tokens"] == 20000000 and "0430" <= hhmm <= "0500":
            sweep.setdefault(f"{arm}@{m['lr']}", {})[int(m["seed"])] = bpc(run)
        elif m["max_tokens"] == 200000000 and abs(m["lr"] - 0.006) < 1e-12 and arm in ("E32", "E2"):
            long.setdefault(arm, {})[int(m["seed"])] = bpc(run)
    means = {k: st.mean(v.values()) for k, v in sweep.items() if len(v) == 2}
    best = {a: min((k for k in means if k.startswith(a + "@")), key=means.get) for a in ("E32", "E34", "E2")}
    rel = lambda a, b: means[a] / means[b] - 1
    w1_gap, w2_gap = rel(best["E32"], best["E34"]), rel(best["E34"], best["E2"])
    e1_7 = st.mean(json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "phase2b_5090.json")))["E1"].values())
    seeds = sorted(set(long["E32"]) & set(long["E2"]))
    d = [math.log(long["E32"][s] / long["E2"][s]) for s in seeds]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    r1 = {"rel_paired": math.expm1(m), "ci95": [math.expm1(m - T3 * se), math.expm1(m + T3 * se)],
          "rel_means": st.mean(long["E32"].values()) / st.mean(long["E2"].values()) - 1, "per_seed": [math.expm1(x) for x in d]}
    r2 = st.mean(long["E32"].values()) / e1_7 - 1
    res = {"sweep": sweep, "two_seed_means": means, "best_lr": best,
           "W1_E32_vs_E34_best": w1_gap, "W1": "PASS" if w1_gap <= -0.05 else "FAIL",
           "W2_E34_vs_E2_best": w2_gap, "W2": "PASS" if w2_gap < 0 else "FAIL",
           "long_200M": long, "R1": r1, "R1_verdict": "PASS" if r1["rel_means"] <= -0.10 else "FAIL",
           "R2_E32_vs_E1_7seed": r2, "R2_verdict": "PASS" if r2 <= -0.05 else "FAIL", "E1_7seed_mean": e1_7}
    json.dump(res, open(os.path.join(ROOT, "experiments", "checks", "results", "sq282_sq283.json"), "w"), indent=1)
    for k in sorted(means):
        print(f"{k:12s} {means[k]:.4f}  {sweep[k]}")
    print("best LR:", best)
    print(f"W1 E32 vs E34 at best LRs {w1_gap * 100:+.2f}% -> {res['W1']} | W2 E34 vs E2 {w2_gap * 100:+.2f}% -> {res['W2']}")
    print("200M:", {a: {s: round(v, 4) for s, v in sorted(x.items())} for a, x in long.items()})
    print(f"R1 E32 vs E2 (4-seed means) {r1['rel_means'] * 100:+.2f}% (paired {r1['rel_paired'] * 100:+.2f}%, CI "
          f"[{r1['ci95'][0] * 100:+.2f}, {r1['ci95'][1] * 100:+.2f}]) -> {res['R1_verdict']}")
    print(f"R2 E32 vs E1 7-seed mean {r2 * 100:+.2f}% -> {res['R2_verdict']}")


if __name__ == "__main__":
    main()
