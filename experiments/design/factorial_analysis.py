"""A3 pre-registered analysis of the 2x2x2 factorial (GDN x shared loops x ternary). CPU, numpy only.

    python3 experiments/design/factorial_analysis.py --csv runs.csv      # columns: cell,seed,y   (cell = '000'..'111')
    python3 experiments/design/factorial_analysis.py --simulate          # checks the power claims by simulation

y = log of the PRIMARY metric: doc-level bits/char on the CLEAN val split (near-duplicates removed, K9).
Run the same analysis a second time with y = log(bits/char on the near-dup docs) (memorisation readout) and on
g = bpc_neardup - bpc_clean (memorisation gap). Seed-0 runs used for LR selection are excluded (winner's curse).

Estimand (primary, inferential): I = y111 - y100 - y010 - y001 + 2*y000, cell means over seeds, log scale.
  I < 0 : the full model beats the additive prediction of its three single-ingredient models (interaction advantage).
  Decision (fixed now; delta = 0.01 log units, i.e. about 1%):
    SYNERGY      : 95% CI of I entirely below 0 (the test the power calculation is for: 80% power at true I = -delta)
    ANTI-SYNERGY : 95% CI entirely above 0
    NO INTERACTION (equivalence): 95% CI inside (-delta, +delta)
    otherwise INCONCLUSIVE (report the CI and the MDE at the achieved n).
Secondary (Holm-adjusted over 4): the three 2-way interactions and the 3-way term from all 8 cells, when all 8 exist.
Main effects are descriptive (they depend on the allocation rule: depth budget 14, FLOPs-matched, params differ).
"""
import argparse
import csv
import itertools
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))), 'checks'))
from tstat import t_quantile as _t_exact  # noqa: E402

W5 = {"111": 1, "100": -1, "010": -1, "001": -1, "000": 2}
Z = 1.96


def welch_ci(groups, w):
    """groups: {cell: array of y}. Returns (I, se, df)."""
    I = sum(w[c] * groups[c].mean() for c in w)
    terms = {c: w[c] ** 2 * groups[c].var(ddof=1) / len(groups[c]) for c in w}
    se = math.sqrt(sum(terms.values()))
    df = se ** 4 / sum(t ** 2 / (len(groups[c]) - 1) for c, t in terms.items())
    return I, se, df


def tcrit(df):   # exact two-sided 95% t quantile (v0.5.45; was a Cornish-Fisher expansion, about 10% low at df 2)
    return _t_exact(0.05, max(df, 1))


def decide(I, se, df, delta=0.01):
    h = tcrit(df) * se
    lo, hi = I - h, I + h
    if hi < 0:
        v = "SYNERGY"
    elif lo > 0:
        v = "ANTI-SYNERGY"
    elif lo > -delta and hi < delta:
        v = "NO INTERACTION (equivalence within +-%.1f%%)" % (100 * delta)
    else:
        v = "INCONCLUSIVE"
    return v, lo, hi


def effects8(groups):
    """Classical factorial effects from cell means (needs all 8 cells): main A,B,C; AB,AC,BC; ABC (difference of means)."""
    m = {c: groups[c].mean() for c in groups}
    out = {}
    for name, idx in [("A_gdn", (0,)), ("B_loop", (1,)), ("C_tern", (2,)), ("AB", (0, 1)), ("AC", (0, 2)), ("BC", (1, 2)), ("ABC", (0, 1, 2))]:
        s = 0.0
        for c in m:
            sign = 1
            for i in idx:
                sign *= 1 if c[i] == "1" else -1
            s += sign * m[c]
        out[name] = s / 4.0
    return out


def read_csv(path):
    g = {}
    for r in csv.DictReader(open(path)):
        g.setdefault(r["cell"], []).append(float(r["y"]))
    return {k: np.array(v) for k, v in g.items()}


def simulate(sigma=0.004, delta=0.01, N=46, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    per = math.ceil(N / 6)
    n = {"000": 2 * per, "100": per, "010": per, "001": per, "111": per}
    hits = {0.0: 0, delta: 0}
    for true in (0.0, -delta):
        ok = 0
        for _ in range(reps):
            g = {c: rng.normal(true if c == "111" else 0.0, sigma, n[c]) for c in n}
            I, se, df = welch_ci(g, W5)
            v, lo, hi = decide(I, se, df, delta)
            ok += (v == "SYNERGY") if true < 0 else (v == "SYNERGY")
        hits[true if true == 0 else delta] = ok / reps
    return n, hits[0.0], hits[delta]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if a.simulate:
        for sigma, N in [(0.004, 46), (0.004, 54), (0.004, 30), (0.0067, 127), (0.0067, 140)]:
            n, fp, pw = simulate(sigma=sigma, N=N)
            print(f"sigma {sigma} N {N} n/cell {n}: P(declare synergy | true I=0) [false-positive rate] = {fp:.3f}, power at true I=-1.0% = {pw:.3f}")
        sys.exit()
    g = read_csv(a.csv)
    I, se, df = welch_ci(g, W5)
    v, lo, hi = decide(I, se, df)
    print(f"I = {100*I:+.2f}%  (95% CI {100*lo:+.2f}% .. {100*hi:+.2f}%), se {100*se:.2f}%, df {df:.1f}: {v}")
    if len(g) == 8:
        for k, e in effects8(g).items():
            print(f"{k:7s} {100*e:+.2f}%")
