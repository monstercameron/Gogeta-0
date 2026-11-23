"""Independent check of the v0.5.45 exact-t correction (quick review 2026-11-23 18:19, reproducibility auditor).

(1) tstat.t_quantile vs scipy.stats.t.ppf over integer and fractional (Welch) df.
(2) Every A2 paired interval recomputed from the saved per-seed ratios in results/a2_assembled.json with SciPy only
    (log1p of each seed's ratio, mean +- t.ppf(0.975, n-1) * sd/sqrt(n), expm1), compared with the stored ci95.
Pass: every quantile within 1e-6 and every bound within 0.01 percentage points.
    python3 experiments/checks/tq_independent.py      # needs scipy (WSL venv gogeta-eval) -> results/tq_independent.json
"""
import json
import math
import os
import statistics as st
import sys

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tstat import t_quantile  # noqa: E402  (only for part 1, the comparison itself)

out = {"scipy": __import__("scipy").__version__, "quantiles": [], "a2": {}}
ok = True
for df in (1, 2, 3, 4, 5, 7.3, 10, 20.6, 60, 140, 163.26):
    a, b = t_quantile(0.05, df), float(stats.t.ppf(0.975, df))
    good = abs(a - b) < 1e-6
    ok &= good
    out["quantiles"].append({"df": df, "tstat": a, "scipy": b, "abs_diff": abs(a - b), "pass": good})
d = json.load(open(os.path.join(HERE, "results", "a2_assembled.json")))
for sec in ("predictions", "decisions"):
    for name, p in d.get(sec, {}).items():
        ps = p.get("per_seed") if isinstance(p, dict) else None
        if not ps or "ci95" not in p:
            continue
        x = [math.log1p(v) for v in ps.values()]
        n = len(x)
        h = float(stats.t.ppf(0.975, n - 1)) * st.stdev(x) / math.sqrt(n)
        m = sum(x) / n
        ci = [math.expm1(m - h), math.expm1(m + h)]
        diff_pp = 100 * max(abs(ci[0] - p["ci95"][0]), abs(ci[1] - p["ci95"][1]))
        good = diff_pp < 0.01
        ok &= good
        out["a2"][f"{sec}/{name}"] = {"n": n, "scipy_ci95": ci, "stored_ci95": p["ci95"], "max_diff_pp": diff_pp, "pass": good}
out["all_pass"] = ok
json.dump(out, open(os.path.join(HERE, "results", "tq_independent.json"), "w", newline="\n"), indent=1)
for q in out["quantiles"]:
    print(f"df {q['df']:>7}: tstat {q['tstat']:.8f} scipy {q['scipy']:.8f} diff {q['abs_diff']:.1e}")
for k, r in out["a2"].items():
    print(f"{k:32s} scipy [{100*r['scipy_ci95'][0]:+.3f}, {100*r['scipy_ci95'][1]:+.3f}]  stored [{100*r['stored_ci95'][0]:+.3f}, {100*r['stored_ci95'][1]:+.3f}]  diff {r['max_diff_pp']:.1e} pp")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
