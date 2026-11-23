"""SQ308: init-controlled depth test of C197 (registered in experiments/checks/PREDICTIONS.md, section SQ308).

E1 (6 blocks) and E32 (14 unshared blocks) x {gogeta, flat} init, 20M tokens, lr 6e-3, seeds 0-2, clean80 bpc.
G1: E32-flat vs E32-gogeta within +-1%.  G2: E32 beats E1 by >= 3% under BOTH inits (paired, CI excluding 0).
Arm membership (depth, init, LR, tokens) comes from each run's meta.json (experiments/checks/results/sq307_sq308_runmap.json).

Run: python experiments/checks/sq308.py -> experiments/checks/results/sq308.json
"""
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
T2 = 4.303  # two-sided 95% t, df = 2


def paired(a, b):
    s = sorted(set(a) & set(b))
    d = [math.log(a[k] / b[k]) for k in s]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    return {"seeds": s, "rel": math.expm1(m), "ci95": [math.expm1(m - T2 * se), math.expm1(m + T2 * se)],
            "per_seed": [math.expm1(x) for x in d], "df": len(d) - 1}


def main():
    runs = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "sq307_sq308_runmap.json")))["runs"]
    arms = {}
    for run, m in runs.items():
        if run.startswith("SQ243") or m.get("max_tokens") != 20000000 or abs(m["lr"] - 0.006) > 1e-12:
            continue
        name = ("E32" if m["n_prelude"] == 14 else "E1") + "-" + m["init"]
        f = os.path.join(DL, run + ".json")
        if os.path.exists(f):
            arms.setdefault(name, {})[int(m["seed"])] = json.load(open(f))["bpc"]["clean80"]
    g1 = paired(arms["E32-flat"], arms["E32-gogeta"])
    e1_init = paired(arms["E1-flat"], arms["E1-gogeta"])
    g2_gogeta = paired(arms["E32-gogeta"], arms["E1-gogeta"])
    g2_flat = paired(arms["E32-flat"], arms["E1-flat"])
    G1 = abs(g1["rel"]) <= 0.010
    G2 = all(t["rel"] <= -0.03 and t["ci95"][1] < 0 for t in (g2_gogeta, g2_flat))
    if G1 and G2:
        rule = "C197's depth lever is not an init artifact at 20M (evidence; still exploratory at 200M)"
    elif abs(g2_flat["rel"]) <= 0.010 and g2_gogeta["rel"] <= -0.03:
        rule = "C197's lever is an init artifact: limitation on C197, README scale wording withdrawn"
    else:
        rule = "inconclusive, reported as such"
    res = {"arms": arms, "means": {k: st.mean(v.values()) for k, v in arms.items()},
           "G1_E32_flat_vs_gogeta": g1, "E1_flat_vs_gogeta": e1_init,
           "G2_E32_vs_E1_gogeta": g2_gogeta, "G2_E32_vs_E1_flat": g2_flat,
           "G1": "PASS" if G1 else "FAIL", "G2": "PASS" if G2 else "FAIL", "decision": rule}
    json.dump(res, open(os.path.join(ROOT, "experiments", "checks", "results", "sq308.json"), "w"), indent=1)
    for k in sorted(arms):
        print(f"{k:11s} mean {res['means'][k]:.4f}  {dict(sorted((s, round(v, 4)) for s, v in arms[k].items()))}")
    for k in ("G1_E32_flat_vs_gogeta", "E1_flat_vs_gogeta", "G2_E32_vs_E1_gogeta", "G2_E32_vs_E1_flat"):
        t = res[k]
        print(f"{k:24s} {t['rel'] * 100:+.2f}%  CI [{t['ci95'][0] * 100:+.2f}, {t['ci95'][1] * 100:+.2f}] df {t['df']}")
    print("G1", res["G1"], "| G2", res["G2"], "|", rule)


if __name__ == "__main__":
    main()
