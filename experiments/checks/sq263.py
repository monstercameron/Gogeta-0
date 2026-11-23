"""SQ263: 200M gate on the five F2 factorial cells (registered in experiments/checks/PREDICTIONS.md, section SQ263).

Input: experiments/checks/results/sq263_runs.json (24 runs at 200M on code-py-v2, cell/lr/seed from each run's
meta.json; built by the gather step recorded in the v0.4.103 devlog). Metric: clean80 doc-level bpc, log scale.
H2': in >= 2 of the 3 six-e-3 cells (E21, E22, E27) the 2e-3 arm beats the 6e-3 arm by >= 0.5% (3-seed means).
H3' (descriptive, registered as underpowered at n = 3): the 200M interaction I at each cell's better LR, with a Welch CI
and a seed-blocked estimate (seeds 0-2 shared by every cell).
C201's registered refutation criterion: "at 200M (SQ263) ... the seed-blocked I's 95% CI includes 0 or lies above 0".

Run: python experiments/checks/sq263.py -> experiments/checks/results/sq263.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "experiments", "checks"))
from tstat import t_quantile as _t_exact  # noqa: E402  (exact Student t; v0.5.45 replaced a Cornish-Fisher approximation)
import json
import math
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments", "design"))
import factorial_analysis as fa  # noqa: E402

CELL = {"E20": "000", "E21": "001", "E22": "010", "E24": "100", "E27": "111"}


def t975(df):
    return _t_exact(0.05, df)  # exact two-sided 95% t quantile (was a Cornish-Fisher approximation, 10% low at df 2)


def main():
    runs = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "sq263_runs.json")))["runs"]
    arms = {}
    for r in runs:
        arms.setdefault((r["cell"], r["lr"]), {})[r["seed"]] = math.log(r["clean80"])
    mean = {k: st.mean(v.values()) for k, v in arms.items()}
    h2 = {}
    for c in ("E21", "E22", "E27"):
        d = math.expm1(mean[(c, 0.002)] - mean[(c, 0.006)])  # < 0 means 2e-3 is better
        h2[c] = {"rel_2e3_vs_6e3": d, "2e3_better_by_0.5pct": d <= -0.005}
    h2_pass = sum(v["2e3_better_by_0.5pct"] for v in h2.values()) >= 2
    best = {c: min((k for k in arms if k[0] == c), key=lambda k: mean[k]) for c in CELL}
    groups = {CELL[c]: np.array(list(arms[best[c]].values())) for c in CELL}
    I, se, df = fa.welch_ci(groups, fa.W5)
    t = t975(df)
    # seed-blocked: y = cell + seed (seeds 0-2 shared)
    rows = [(CELL[c], s, v) for c in CELL for s, v in arms[best[c]].items()]
    cells = sorted({r[0] for r in rows}); seeds = sorted({r[1] for r in rows})
    X = np.zeros((len(rows), len(cells) + len(seeds) - 1)); y = np.array([r[2] for r in rows])
    for i, (c, s, _) in enumerate(rows):
        X[i, cells.index(c)] = 1
        if seeds.index(s) > 0:
            X[i, len(cells) + seeds.index(s) - 1] = 1
    XtX = np.linalg.pinv(X.T @ X); beta = XtX @ X.T @ y; res = y - X @ beta
    dfb = len(rows) - np.linalg.matrix_rank(X); s2 = float(res @ res / dfb)
    w = np.zeros(X.shape[1])
    for c, k in fa.W5.items():
        w[cells.index(c)] = k
    Ib = float(w @ beta); seb = math.sqrt(s2 * float(w @ XtX @ w)); tb = t975(dfb)
    blocked_ci = [Ib - tb * seb, Ib + tb * seb]
    vs_dense = {CELL[c]: math.expm1(mean[best[c]] - mean[best["E20"]]) for c in CELL}
    out = {"n_runs": len(runs), "mean_bpc": {f"{c}@{lr}": math.exp(v) for (c, lr), v in mean.items()},
           "H2prime": h2, "H2prime_verdict": "PASS" if h2_pass else "FAIL",
           "best_lr": {c: best[c][1] for c in CELL}, "vs_dense_at_best_lr": vs_dense,
           "H3prime_welch": {"I": I, "ci95": [I - t * se, I + t * se], "df": df},
           "H3prime_seed_blocked": {"I": Ib, "ci95": blocked_ci, "df": int(dfb)},
           "C201_refutation_check": ("met: the 200M seed-blocked CI lies above 0 (anti-synergy)" if blocked_ci[0] > 0 else
                                     "met: the 200M seed-blocked CI includes 0" if blocked_ci[0] <= 0 <= blocked_ci[1] else
                                     "not met: the 200M seed-blocked CI lies below 0")}
    json.dump(out, open(os.path.join(ROOT, "experiments", "checks", "results", "sq263.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "mean_bpc"}, indent=1))


if __name__ == "__main__":
    main()
