"""SQ238: dedup-only control for code-py-v2 (registered in experiments/checks/PREDICTIONS.md, "Fleet wave 6, part 1").

Arms (200M tokens, lr 6e-3, RTX 5090, clean80 bpc on v1 val):
  A  E1 on code-py-v1sub (seeds 0-3)       SQ238v1sub-* runs
  B  E1 on code-py-v2 (E53, seeds 0-3)     E53 200M runs (seed 0 = the H5 bridge run; the 20M E53 run is excluded)
  C  E1 on code-py-v1 (Phase 2b-5090, 7 seeds)
Contrasts: dedup D = log A - log B (paired by seed); coverage/size S = log C - log A (Welch).
D1: D > 0 and 1.0% <= D <= 3.0%.  D2: |S| <= 1.0%.  D3: D + S ~ the H5 gap (3.0%) within the CIs.

Run: python experiments/checks/sq238.py -> experiments/checks/results/sq238.json
"""
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, "experiments", "stats", "doclevel")
T3 = 3.182
# 200M E53 runs (token counts checked: fleet_2026-11-22.json for s0, worker.status FINAL tokens=200015872 for s1-s3)
B_RUNS = ["E53-20261123-030901-s0", "E53-20261123-053922-s1", "E53-20261123-054336-s2", "E53-20261123-054751-s3"]


def load(run):
    d = json.load(open(os.path.join(DL, run + ".json")))
    return int(d["seed"]), d["bpc"]["clean80"]


def main():
    A = dict(load(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(DL, "SQ238v1sub-2026*.json")))
    B = dict(load(r) for r in B_RUNS)
    C = {int(k): v for k, v in json.load(open(os.path.join(ROOT, "experiments", "checks", "results",
                                                         "phase2b_5090.json")))["E1"].items()}
    seeds = sorted(set(A) & set(B))
    d = [math.log(A[s] / B[s]) for s in seeds]
    Dm, Dse = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    D = {"rel": math.expm1(Dm), "ci95": [math.expm1(Dm - T3 * Dse), math.expm1(Dm + T3 * Dse)],
         "per_seed": [math.expm1(x) for x in d], "df": len(d) - 1}
    la, lc = [math.log(v) for v in A.values()], [math.log(v) for v in C.values()]
    Sm = st.mean(lc) - st.mean(la)
    Sse = math.sqrt(st.variance(la) / len(la) + st.variance(lc) / len(lc))
    S = {"rel": math.expm1(Sm), "approx_ci95": [math.expm1(Sm - 2.3 * Sse), math.expm1(Sm + 2.3 * Sse)],
         "n": [len(lc), len(la)]}
    h5 = math.log(0.2679 / 0.2599)  # the registered H5 gap (v1 0.2679 vs v2 0.2599)
    res = {"A_v1sub": A, "B_v2": B, "C_v1_mean": st.mean(C.values()), "D_dedup": D, "S_coverage": S,
           "D1": "PASS" if (D["ci95"][0] > 0 and 0.010 <= D["rel"] <= 0.030) else "FAIL",
           "D2": "PASS" if abs(S["rel"]) <= 0.010 else "FAIL",
           "D3_sum_vs_H5": {"D_plus_S": math.expm1(Dm + Sm), "H5_gap": math.expm1(h5)}}
    res["decision"] = ("A3 v2 results stand as 'with dedup' (D's CI excludes 0 and is positive)"
                       if D["ci95"][0] > 0 else
                       ("v2's gain is a coverage artefact" if abs(S["rel"]) > abs(D["rel"]) else "inconclusive"))
    json.dump(res, open(os.path.join(ROOT, "experiments", "checks", "results", "sq238.json"), "w"), indent=1)
    print("A v1sub", {k: round(v, 4) for k, v in sorted(A.items())}, "| B v2", {k: round(v, 4) for k, v in sorted(B.items())},
          f"| C v1 mean {res['C_v1_mean']:.4f}")
    print(f"D (dedup)  {D['rel'] * 100:+.2f}%  CI [{D['ci95'][0] * 100:+.2f}, {D['ci95'][1] * 100:+.2f}] df 3",
          [f"{x * 100:+.2f}" for x in D["per_seed"]])
    print(f"S (coverage) {S['rel'] * 100:+.2f}%  approx CI [{S['approx_ci95'][0] * 100:+.2f}, {S['approx_ci95'][1] * 100:+.2f}]")
    print(f"D+S {res['D3_sum_vs_H5']['D_plus_S'] * 100:+.2f}% vs H5 {res['D3_sum_vs_H5']['H5_gap'] * 100:+.2f}%")
    print("D1", res["D1"], "| D2", res["D2"], "|", res["decision"])


if __name__ == "__main__":
    main()
