"""SQ200 init 2x2 (+ SQ234): registered predictions I1-I4 (experiments/checks/PREDICTIONS.md, "Fleet wave 6, part 2").

Metric: clean80 doc-level bpc on v1 val, log scale; RTX 5090 only. E10-hf and E1-gogeta are the Phase 2b-5090 7-seed arms
(experiments/checks/results/phase2b_5090.json). Init effect within an architecture is paired by seed on seeds 0-3
(the Phase 2b arms have those seeds), with a Welch comparison against all 7 reference seeds reported too.

Run: python experiments/checks/sq200.py -> experiments/checks/results/sq200.json
"""
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
T3 = 3.182  # two-sided 95% t, df = 3


def arm(prefix):
    out = {}
    for f in glob.glob(os.path.join(DL, prefix + "-2026*.json")):
        d = json.load(open(f))
        out[int(d["seed"])] = d["bpc"]["clean80"]
    return dict(sorted(out.items()))


def paired_log(a, b):
    """mean of log(a/b) over common seeds (negative = a better), with a 95% t CI; reported as relative %."""
    s = sorted(set(a) & set(b))
    d = [math.log(a[k] / b[k]) for k in s]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    return {"seeds": s, "rel": math.expm1(m), "ci95": [math.expm1(m - T3 * se), math.expm1(m + T3 * se)],
            "per_seed": [math.expm1(x) for x in d]}


def welch_log(a, b):
    la, lb = [math.log(x) for x in a.values()], [math.log(x) for x in b.values()]
    m = st.mean(la) - st.mean(lb)
    se = math.sqrt(st.variance(la) / len(la) + st.variance(lb) / len(lb))
    return {"rel": math.expm1(m), "approx_ci95": [math.expm1(m - 2.2 * se), math.expm1(m + 2.2 * se)],
            "n": [len(la), len(lb)]}


def main():
    p2b = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "phase2b_5090.json")))
    ref = {"E10_hf": {int(k): v for k, v in p2b["E10"].items()}, "E1_gogeta": {int(k): v for k, v in p2b["E1"].items()}}
    arms = {"E0_gogeta": arm("SQ200_e0_gogeta"), "E0_flat": arm("SQ200_e0_flat"),
            "E10_gogeta": arm("SQ200_e10_gogeta"), "E1_flat": arm("SQ200_e1_flat"), **ref}
    res = {"metric": "clean80 bpc, v1 val (log scale)", "arms": arms,
           "means": {k: st.mean(v.values()) for k, v in arms.items()}, "n": {k: len(v) for k, v in arms.items()}}
    i1 = welch_log(arms["E0_flat"], arms["E0_gogeta"]) | {"paired": paired_log(arms["E0_flat"], arms["E0_gogeta"])}
    i1["verdict"] = "PASS" if i1["paired"]["rel"] <= -0.015 else "FAIL"
    i2 = {"paired_s0_3": paired_log(arms["E10_gogeta"], arms["E10_hf"]), "welch_vs_7": welch_log(arms["E10_gogeta"], arms["E10_hf"])}
    i2["verdict"] = "PASS" if i2["paired_s0_3"]["rel"] >= 0.010 else "FAIL"
    i3 = {"paired_s0_3": paired_log(arms["E1_flat"], arms["E1_gogeta"]), "welch_vs_7": welch_log(arms["E1_flat"], arms["E1_gogeta"])}
    i3["verdict"] = "PASS" if -0.025 <= i3["paired_s0_3"]["rel"] <= -0.005 else "FAIL"
    i4 = {"triggered": i3["verdict"] == "PASS",
          "E1_flat_vs_E10_hf_paired_s0_3": paired_log(arms["E1_flat"], arms["E10_hf"]),
          "note": "descriptive only unless I3 holds (registered)"}
    e0_e10 = {"E0_gogeta_vs_E10_hf": welch_log(arms["E0_gogeta"], arms["E10_hf"]),
              "E0_flat_vs_E10_hf": welch_log(arms["E0_flat"], arms["E10_hf"]),
              "E0_flat_vs_E10_hf_paired_s0_3": paired_log(arms["E0_flat"], arms["E10_hf"])}
    res["tests"] = {"I1": i1, "I2": i2, "I3": i3, "I4": i4, "architecture_within_init": e0_e10}
    out = os.path.join(ROOT, "experiments", "checks", "results", "sq200.json")
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res["means"].items():
        print(f"{k:11s} {v:.5f} n={res['n'][k]}")
    for k in ("I1", "I2", "I3"):
        t = res["tests"][k]
        p = t.get("paired") or t["paired_s0_3"]
        print(k, t["verdict"], f"paired {p['rel']*100:+.2f}% CI [{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}]")
    p = i4["E1_flat_vs_E10_hf_paired_s0_3"]
    print("I4 (descriptive)", f"E1-flat vs E10-hf {p['rel']*100:+.2f}% CI [{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}]")
    for k, v in e0_e10.items():
        print(k, f"{v['rel']*100:+.2f}%")


if __name__ == "__main__":
    main()
