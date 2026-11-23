"""F2 blinded variance re-estimation (registered: experiments/checks/PREDICTIONS.md, "Wave-5 amendment").

Reads clean80 doc-level bpc for the five F2 cells (E20, E21, E22, E24, E27) at their registered LRs, fits
log(bpc) = cell + seed + e (seed-blocked, the registered primary model) and prints ONLY:
  - n runs per cell (counts, not values), the residual df,
  - the pooled residual sigma (seed-blocked) and the unblocked within-cell sigma,
  - the final N for 80% power at |I| = 1% by the registered sigma^2 scaling (N = 235 at sigma 0.009), capped at 246,
  - the stage-2 size = final N - runs already done.
No cell means, contrasts or interaction estimates are printed or written (blinding). The fitted cell effects are used
internally only to form residuals, as the registration's "within-cell, seed-blocked residuals" requires.

Run-to-LR comes from each run's meta.json (pass --runs-glob for the pulled fleet run dirs).

    python experiments/design/f2_blinded_sigma.py --runs-glob "/root/fusiondance-data/cloud/fleet/*/fusiondance-data/runs"
"""
import argparse
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELL_LR = {"E20": 0.003, "E21": 0.006, "E22": 0.006, "E24": 0.003, "E27": 0.006}
N_REF, SIGMA_REF, N_CAP = 235, 0.009, 246


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", required=True, help="dir(s) holding run folders with meta.json")
    ap.add_argument("--doclevel", default=os.path.join(ROOT, "experiments", "stats", "doclevel"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "checks", "results", "f2_blinded_sigma.json"))
    a = ap.parse_args()
    lr_of = {}
    for d in glob.glob(a.runs_glob):
        for m in glob.glob(os.path.join(d, "E2[0-7]-*", "meta.json")):
            j = json.load(open(m))
            lr = (j.get("config") or {}).get("train", {}).get("lr")
            co = j.get("cli_overrides") or {}
            if isinstance(co, dict) and "lr" in co:
                lr = co["lr"]
            tok = (j.get("config") or {}).get("train", {}).get("max_tokens")
            lr_of[os.path.basename(os.path.dirname(m))] = (lr, tok)
    rows = []  # (cell, seed, log bpc); one run per (cell, seed): the latest file wins
    seen = {}
    for f in sorted(glob.glob(os.path.join(a.doclevel, "E2[0-7]-2026*.json"))):
        d = json.load(open(f))
        run, cell = d["run"], d["run"].split("-")[0]
        if cell not in CELL_LR or run not in lr_of:
            continue
        lr, _ = lr_of[run]
        if lr is None or abs(lr - CELL_LR[cell]) > 1e-12:
            continue
        seen[(cell, int(d["seed"]))] = math.log(d["bpc"]["clean80"])
    rows = [(c, s, y) for (c, s), y in sorted(seen.items())]
    cells = sorted({r[0] for r in rows})
    seeds = sorted({r[1] for r in rows})
    n = len(rows)
    X = np.zeros((n, len(cells) + len(seeds) - 1))
    y = np.array([r[2] for r in rows])
    for i, (c, s, _) in enumerate(rows):
        X[i, cells.index(c)] = 1
        if seeds.index(s) > 0:
            X[i, len(cells) + seeds.index(s) - 1] = 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rank = np.linalg.matrix_rank(X)
    df = n - rank
    sigma_blocked = float(math.sqrt(resid @ resid / df))
    # unblocked within-cell sigma (sensitivity)
    ss, dfu = 0.0, 0
    for c in cells:
        v = np.array([r[2] for r in rows if r[0] == c])
        ss += float(((v - v.mean()) ** 2).sum())
        dfu += len(v) - 1
    sigma_unblocked = math.sqrt(ss / dfu)
    n_final = min(N_CAP, math.ceil(N_REF * (sigma_blocked / SIGMA_REF) ** 2))
    stage2 = max(0, n_final - n)
    out = {"n_runs": n, "n_per_cell": {c: sum(1 for r in rows if r[0] == c) for c in cells}, "n_seeds": len(seeds),
           "resid_df": int(df), "sigma_seed_blocked": sigma_blocked, "sigma_unblocked": sigma_unblocked,
           "rule": "N_final = min(246, ceil(235 * (sigma_blocked / 0.009)^2)); stage2 = N_final - n_runs",
           "n_final": n_final, "stage2_runs": stage2,
           "blinding": "no cell means, contrasts or interaction estimates are computed for output or stored"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
