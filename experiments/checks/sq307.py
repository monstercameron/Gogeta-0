"""SQ307: fresh-seed confirmation of SQ301 at d = 256 (registered in experiments/checks/PREDICTIONS.md, section SQ307).

Primary: E1 at lr 1.2e-2 vs E10 at lr 3e-3, seeds 2-5, paired by seed (n = 4), clean80 bpc.
Secondary: E1 at lr 1.8e-2 (is the E1 optimum beyond the old grid?).
Run -> LR/seed from each run's meta.json (experiments/checks/results/sq307_sq308_runmap.json).

Run: python experiments/checks/sq307.py -> experiments/checks/results/sq307.json
"""
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
T3 = 3.182  # two-sided 95% t, df = 3


def main():
    runs = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "sq307_sq308_runmap.json")))["runs"]
    cells = {}
    for run, m in runs.items():
        if not run.startswith("SQ243E"):
            continue
        arm = "E1" if run.startswith("SQ243E1d") else "E10"
        f = os.path.join(DL, run + ".json")
        if not os.path.exists(f):
            continue
        cells.setdefault(f"{arm}@{m['lr']}", {})[int(m["seed"])] = json.load(open(f))["bpc"]["clean80"]
    e1, e10, e1b = cells["E1@0.012"], cells["E10@0.003"], cells.get("E1@0.018", {})
    seeds = sorted(set(e1) & set(e10))
    d = [math.log(e1[s] / e10[s]) for s in seeds]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    gap, ci = math.expm1(m), [math.expm1(m - T3 * se), math.expm1(m + T3 * se)]
    w3 = gap <= -0.010 and ci[1] < 0
    if w3:
        rule = "W3 holds: C198's d=256 component supported on fresh seeds (stays PARTIAL: d=384/512 'ties' need TOST)"
    elif abs(gap) <= 0.005:
        rule = "d=256 effect refuted as selection + noise"
    else:
        rule = "inconclusive"
    sec = {s: math.expm1(math.log(e1b[s] / e1[s])) for s in sorted(set(e1b) & set(e1))}
    res = {"cells": cells, "seeds": seeds, "per_seed_rel": [math.expm1(x) for x in d], "gap_rel": gap, "ci95": ci,
           "df": len(d) - 1, "W3": "PASS" if w3 else "FAIL", "decision": rule,
           "secondary_E1_1.8e-2_vs_1.2e-2_per_seed": sec}
    json.dump(res, open(os.path.join(ROOT, "experiments", "checks", "results", "sq307.json"), "w"), indent=1)
    for k in sorted(cells):
        print(k, {s: round(v, 4) for s, v in sorted(cells[k].items())})
    print(f"E1@1.2e-2 - E10@3e-3: {gap * 100:+.2f}%  CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] df {len(d) - 1}",
          [f"{math.expm1(x) * 100:+.2f}" for x in d])
    print("W3", res["W3"], "|", rule)
    print("secondary 1.8e-2 vs 1.2e-2:", {s: f"{v * 100:+.2f}%" for s, v in sec.items()})


if __name__ == "__main__":
    main()
