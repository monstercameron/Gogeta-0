"""TOST equivalence check (+-1% margin, alpha 0.05 per side, i.e. a 90% CI) on A3's reference comparisons, from the paired
per-seed log differences in results/a3_assembled.json (quick review 2026-11-23 14:09). -> results/a3_tost.json"""
import json, math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstat import t_quantile, t_two_sided_p  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MARGIN = math.log(1.01)


def one_sided(t, df):  # P(T > t)
    p = t_two_sided_p(abs(t), df) / 2
    return p if t > 0 else 1 - p


out = {}
d = json.load(open(os.path.join(R, "a3_assembled.json")))["predictions"]
for k in ("Q4_A3mx_vs_A2f", "Q5a_A3mx_vs_E24", "Q5b_A3mx_vs_E20"):
    x = [math.log1p(v) for v in d[k]["per_seed"].values()]
    n = len(x); m = st.mean(x); se = st.stdev(x) / math.sqrt(n); df = n - 1; q = t_quantile(0.10, df)
    p = max(one_sided((m + MARGIN) / se, df), one_sided((MARGIN - m) / se, df))
    out[k] = {"n": n, "rel": math.expm1(m), "ci90": [math.expm1(m - q * se), math.expm1(m + q * se)], "margin": 0.01,
              "tost_p": p, "equivalent": p < 0.05}
    print(k, f"{out[k]['rel'] * 100:+.2f}% 90% CI [{out[k]['ci90'][0] * 100:+.2f}, {out[k]['ci90'][1] * 100:+.2f}] TOST p={p:.3f}")
json.dump(out, open(os.path.join(R, "a3_tost.json"), "w", newline="\n"), indent=1)
