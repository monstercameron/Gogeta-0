"""F2 registered analysis: the 2x2x2-factorial interaction I on the five F2 cells (E20 000, E21 001, E22 010, E24 100,
E27 111), 20M tokens on code-py-v2, clean80 doc-level bpc (log scale). Committed BEFORE it was first run (v0.4.97).

Registration (experiments/checks/PREDICTIONS.md):
  - estimand I = y111 - y100 - y010 - y001 + 2*y000 (experiments/design/factorial_analysis.py; I < 0 = synergy);
  - wave-5 amendment: the PRIMARY analysis is the seed-blocked factorial contrast (seed fixed effect + cell effects on
    log clean80 bpc); the unblocked Welch analysis (factorial_analysis.welch_ci) is the registered sensitivity;
  - decision, delta = 0.01: SYNERGY if the 95% CI lies below 0; ANTI-SYNERGY if above 0; NO INTERACTION if inside
    (-delta, +delta); else INCONCLUSIVE;
  - seed-0 runs used for LR selection are excluded (winner's curse);
  - registered prediction: I = +0.5% +- 1.5%; P(I <= -1%) = 0.10, P(|I| < 1%) = 0.55, P(I >= +1%) = 0.35;
    full fusion (E27) the worst of the cells.
Only runs at the registered LR, 20M tokens and code-py-v2 are used (from each run's meta.json).

    python3 experiments/design/f2_analysis.py --doclevel <dir of E2x score JSONs> --runs-glob "<dirs with run folders>"
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import factorial_analysis as fa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
CELL = {"E20": "000", "E21": "001", "E22": "010", "E24": "100", "E27": "111"}
LR = {"E20": 0.003, "E21": 0.006, "E22": 0.006, "E24": 0.003, "E27": 0.006}
DELTA = 0.01


def t975(df):
    # two-sided 95% t quantile (normal approximation refined; df >= 3)
    from math import sqrt
    z = 1.959963984540054
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def decide(lo, hi):
    if hi < 0:
        return "SYNERGY"
    if lo > 0:
        return "ANTI-SYNERGY"
    if lo > -DELTA and hi < DELTA:
        return "NO INTERACTION (equivalence)"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doclevel", required=True)
    ap.add_argument("--runs-glob", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "checks", "results", "f2_analysis.json"))
    ap.add_argument("--metric", default="clean80", help="bpc key in the doclevel JSON: clean80 (primary), dup80 "
                    "(registered memorisation readout), clean50, all")
    a = ap.parse_args()
    meta = {}
    for d in glob.glob(a.runs_glob):
        for m in glob.glob(os.path.join(d, "E2[0-7]-*", "meta.json")):
            j = json.load(open(m))
            cfg = j.get("config") or {}
            tr = cfg.get("train", {})
            co = j.get("cli_overrides") or {}
            lr = co.get("lr", tr.get("lr")) if isinstance(co, dict) else tr.get("lr")
            mt = (co.get("max_tokens") if isinstance(co, dict) else None) or tr.get("max_tokens")
            meta[os.path.basename(os.path.dirname(m))] = (lr, mt, tr.get("dataset"))
    y = {}
    for f in sorted(glob.glob(os.path.join(a.doclevel, "E2[0-7]-2026*.json"))):
        d = json.load(open(f))
        run, arm, seed = d["run"], d["run"].split("-")[0], int(d["seed"])
        if arm not in CELL or run not in meta or seed == 0:
            continue
        lr, mt, ds = meta[run]
        if lr is None or abs(lr - LR[arm]) > 1e-12 or mt != 20000000 or ds != "code-py-v2":
            continue
        y[(CELL[arm], seed)] = math.log(d["bpc"][a.metric])
    rows = sorted(y.items())
    cells = sorted({c for (c, _), _ in rows})
    seeds = sorted({s for (_, s), _ in rows})
    groups = {c: np.array([v for (cc, _), v in rows if cc == c]) for c in cells}
    # sensitivity (registered): unblocked Welch
    I_u, se_u, df_u = fa.welch_ci(groups, fa.W5)
    t_u = t975(df_u)
    unblocked = {"I": I_u, "se": se_u, "df": df_u, "ci95": [I_u - t_u * se_u, I_u + t_u * se_u]}
    unblocked["decision"] = decide(*unblocked["ci95"])
    # primary (registered): seed-blocked OLS, y = cell + seed + e; I as a contrast of the cell coefficients
    X = np.zeros((len(rows), len(cells) + len(seeds) - 1))
    yv = np.array([v for _, v in rows])
    for i, ((c, s), _) in enumerate(rows):
        X[i, cells.index(c)] = 1
        if seeds.index(s) > 0:
            X[i, len(cells) + seeds.index(s) - 1] = 1
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ yv
    resid = yv - X @ beta
    df_b = len(rows) - np.linalg.matrix_rank(X)
    s2 = float(resid @ resid / df_b)
    w = np.zeros(X.shape[1])
    for c, coef in fa.W5.items():
        w[cells.index(c)] = coef
    I_b = float(w @ beta)
    se_b = math.sqrt(s2 * float(w @ XtX_inv @ w))
    t_b = t975(df_b)
    blocked = {"I": I_b, "se": se_b, "df": int(df_b), "sigma": math.sqrt(s2), "ci95": [I_b - t_b * se_b, I_b + t_b * se_b]}
    blocked["decision"] = decide(*blocked["ci95"])
    cell_means = {c: float(groups[c].mean()) for c in cells}
    ranking = sorted(cells, key=lambda c: cell_means[c])
    res = {"metric": a.metric, "n_runs": len(rows), "n_per_cell": {c: int(len(groups[c])) for c in cells}, "seeds_excluded": [0],
           "primary_seed_blocked": blocked, "sensitivity_unblocked": unblocked,
           "cell_mean_log_bpc": cell_means, "cell_mean_bpc": {c: math.exp(v) for c, v in cell_means.items()},
           "ranking_best_to_worst": ranking,
           "registered_prediction": "I = +0.5% +- 1.5%; P(I<=-1%)=0.10, P(|I|<1%)=0.55, P(I>=+1%)=0.35; 111 worst",
           "prediction_full_fusion_worst": ranking[-1] == "111"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cell_mean_log_bpc"}, indent=1))


if __name__ == "__main__":
    main()
