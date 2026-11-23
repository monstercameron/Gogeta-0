"""SQ301: LR-fair re-test of C198 at d = 256 (registered in experiments/checks/PREDICTIONS.md, section SQ301).

Arms: SQ243 d = 256 configs, E1 and E10, lr {3e-3, 6e-3, 1.2e-2}, seeds 0-1 (the SQ243 cells E1 6e-3 / E10 3e-3 reused).
Best LR per arm = lowest 2-seed mean clean80 bpc. Gap at best LRs = paired by seed, relative E1 vs E10.
Run -> LR comes from each run's meta.json (experiments/checks/results/sq301_lrmap.json).

Run: python experiments/checks/sq301.py -> experiments/checks/results/sq301.json
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
LRS = (0.003, 0.006, 0.012)


def main():
    lrmap = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "sq301_lrmap.json")))["runs"]
    cells = {}  # (arm, lr) -> {seed: bpc}
    for f in glob.glob(os.path.join(DL, "SQ243E*d256-2026*.json")):
        d = json.load(open(f))
        run = d["run"]
        arm = "E1" if run.startswith("SQ243E1d") else "E10"
        lr = lrmap[run]["lr"]
        cells.setdefault(f"{arm}@{lr}", {})[int(d["seed"])] = d["bpc"]["clean80"]
    means = {k: st.mean(v[s] for s in (0, 1)) for k, v in cells.items() if 0 in v and 1 in v}
    best = {a: min((lr for lr in LRS if f"{a}@{lr}" in means), key=lambda lr: means[f"{a}@{lr}"]) for a in ("E1", "E10")}
    e1, e10 = cells[f"E1@{best['E1']}"], cells[f"E10@{best['E10']}"]
    per_seed = {s: e1[s] / e10[s] - 1 for s in (0, 1)}
    gap = st.mean(per_seed.values())
    w1 = "PASS" if best["E10"] >= 0.006 else "FAIL"
    w2 = "PASS" if -0.010 <= gap <= 0.0025 else "FAIL"
    if abs(gap) <= 0.005:
        rule = "C198 REFUTED (LR artifact)"
    elif gap <= -0.010 and all(v < 0 for v in per_seed.values()):
        rule = "C198 keeps PARTIAL, with SQ301 as evidence (n = 2; init confound remains)"
    else:
        rule = "C198 stays PARTIAL, inconclusive at 2 seeds"
    res = {"cells": cells, "two_seed_means": means, "best_lr": best, "gap_at_best_per_seed": per_seed,
           "gap_at_best_mean": gap, "W1": w1, "W2": w2, "decision": rule}
    json.dump(res, open(os.path.join(ROOT, "experiments", "checks", "results", "sq301.json"), "w"), indent=1, default=str)
    for k in sorted(means):
        print(f"{k:12s} {means[k]:.4f}  {cells[k]}")
    print("best LR", best, f"gap {gap * 100:+.2f}%", {s: f"{v * 100:+.2f}%" for s, v in per_seed.items()})
    print("W1", w1, "| W2", w2, "|", rule)


if __name__ == "__main__":
    main()
