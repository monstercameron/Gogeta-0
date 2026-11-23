"""A2 multiplicity replay (quick review 2026-11-23 13:44): Holm over A2's registered comparisons (P1-P5 + amendment 1),
from the per-seed paired log differences stored in results/a2_assembled.json (paired t, df 2). Descriptive audit of the
decisions; the A2 registration did not name a family, so the family here is every registered A2 comparison.
Run: python3 experiments/checks/a2_holm.py -> results/a2_holm.json
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tstat import t_two_sided_p
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = json.load(open(os.path.join(R, "a2_assembled.json")))["predictions"]
rows = []
for k, v in d.items():
    x = [math.log1p(e) for e in v["per_seed"].values()]
    n = len(x); m = sum(x) / n; sd = math.sqrt(sum((a - m) ** 2 for a in x) / (n - 1))
    t = m / (sd / math.sqrt(n)); p = t_two_sided_p(t, n - 1)
    rows.append({"comparison": k, "n": n, "rel": math.expm1(m), "t": t, "df": n - 1, "p": p})
rows.sort(key=lambda r: r["p"]); m = len(rows)
for i, r in enumerate(rows):
    r["holm_alpha"] = 0.05 / (m - i)
    r["holm_reject"] = all(rows[j]["p"] < rows[j].get("holm_alpha", 0.05 / (m - j)) for j in range(i + 1))
json.dump({"family": [r["comparison"] for r in rows], "alpha": 0.05, "rows": rows}, open(os.path.join(R, "a2_holm.json"), "w", newline="\n"), indent=1)
for r in rows: print(f"{r['comparison']:16s} {r['rel']*100:+7.2f}%  t={r['t']:7.1f} df={r['df']} p={r['p']:.2e}  holm_alpha={r['holm_alpha']:.4f} reject={r['holm_reject']}")
