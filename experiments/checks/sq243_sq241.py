"""SQ243 (two-size check) and SQ241 (param-matched E1): registered tests S1/S2/S3 and P1/P2.

Registration: experiments/checks/PREDICTIONS.md, "Fleet wave 4" (SQ243, SQ241). Metric: clean80 doc-level bpc on v1 val
(experiments/stats/score_doclevel.py output in experiments/stats/doclevel/). All arms on RTX 5090s.
The d = 384 point and the SQ241 references are the Phase 2b-5090 7-seed arms (experiments/checks/results/phase2b_5090.json).

Run: python experiments/checks/sq243_sq241.py  -> experiments/checks/results/sq243_sq241.json
"""
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
T975 = {3: 3.182, 6: 2.447}  # two-sided 95% t quantiles for df = 3 (paired n=4) and df = 6 (Welch fallback bound)


def arm(prefix):
    out = {}
    for f in glob.glob(os.path.join(DL, prefix + "-2026*.json")):
        d = json.load(open(f))
        out[int(d["seed"])] = d["bpc"]["clean80"]
    return dict(sorted(out.items()))


def rel(a, b):
    return a / b - 1.0


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    diffs = [rel(a[s], b[s]) for s in seeds]  # per-seed relative difference a vs b
    m = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(len(diffs))
    t = T975[len(diffs) - 1]
    return {"seeds": seeds, "mean_rel": m, "ci95": [m - t * se, m + t * se], "per_seed": diffs}


def main():
    p2b = json.load(open(os.path.join(ROOT, "experiments", "checks", "results", "phase2b_5090.json")))
    e1_384 = {int(k): v for k, v in p2b["E1"].items()}
    e10_384 = {int(k): v for k, v in p2b["E10"].items()}
    arms = {k: arm(k) for k in ["SQ243E1d256", "SQ243E10d256", "SQ243E1d512", "SQ243E10d512", "SQ241"]}
    res = {"metric": "clean80 bpc, v1 val", "arms": arms,
           "means": {k: st.mean(v.values()) for k, v in arms.items()}}
    res["means"]["E1_d384_phase2b"] = st.mean(e1_384.values())
    res["means"]["E10_d384_phase2b"] = st.mean(e10_384.values())

    s1 = paired(arms["SQ243E1d512"], arms["SQ243E10d512"])
    s2 = paired(arms["SQ243E1d256"], arms["SQ243E10d256"])
    s1["verdict"] = "PASS" if abs(s1["mean_rel"]) <= 0.01 else "FAIL"
    s2["verdict"] = "PASS" if abs(s2["mean_rel"]) <= 0.01 else "FAIL"
    trend = {"d256": s2["mean_rel"], "d384": rel(res["means"]["E1_d384_phase2b"], res["means"]["E10_d384_phase2b"]),
             "d512": s1["mean_rel"]}
    s3 = {"gap_by_width": trend,
          "scaling_effect_called": any(abs(trend[k]) > 0.009 and (lo > 0 or hi < 0)
                                       for k, (lo, hi) in (("d256", s2["ci95"]), ("d512", s1["ci95"]))),
          "rule": "only a gap beyond the 0.9% MDE at an end point (with its CI excluding 0) is called a scaling effect"}

    # SQ241 is not seed-paired with Phase 2b in data order beyond the seed id; seeds 0-3 exist in both, so pair by seed
    # (same seed -> same data order and init stream), and also report the unpaired mean vs the 7-seed means as registered.
    p1 = {"paired_vs_E10_seeds0_3": paired(arms["SQ241"], e10_384),
          "mean_rel_vs_E10_7seed": rel(res["means"]["SQ241"], res["means"]["E10_d384_phase2b"])}
    p1["verdict"] = "PASS" if abs(p1["mean_rel_vs_E10_7seed"]) <= 0.01 else "FAIL"
    p2 = {"paired_vs_E1_seeds0_3": paired(arms["SQ241"], e1_384),
          "mean_rel_vs_E1_7seed": rel(res["means"]["SQ241"], res["means"]["E1_d384_phase2b"])}
    p2["verdict"] = "PASS" if p2["mean_rel_vs_E1_7seed"] <= 0.005 else "FAIL"
    res["tests"] = {"S1_d512_tie": s1, "S2_d256_tie": s2, "S3_trend": s3, "P1_parammatch_vs_E10": p1,
                    "P2_parammatch_vs_E1": p2}
    out = os.path.join(ROOT, "experiments", "checks", "results", "sq243_sq241.json")
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res["means"].items():
        print(f"{k:22s} {v:.5f}  n={len(arms.get(k, {})) or 7}")
    for k, v in res["tests"].items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("per_seed",)}, default=str)[:400])


if __name__ == "__main__":
    main()
