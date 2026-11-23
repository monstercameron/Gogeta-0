"""E60 stage 1 registered analysis (experiments/checks/PREDICTIONS.md, "E60 stage 1"; registered before dispatch).

Arms on code-py-v1, 200M tokens, RTX 5090, clean80 doc-level bpc (v1 val), log scale:
  E60a: E1 + dropless 4x512 top-2 MoE FFN (learned router); E60b: E60a with a frozen random router;
  E60c: dense SwiGLU 2048, total-param matched (1.47x E1 FLOPs). Reference: E1, the 7 Phase 2b-5090 seeds.
LR (G3): each arm's argmin over {3e-3, 6e-3, 1.2e-2} at 20M, seed 0; 200M seeds 0-5 at that LR.
M1: E60a - E1 = -1.8% (-0.8% to -3.5%).  M2: E60c - E1 = -3.0% (-1.5% to -4.5%); E60a worse than E60c by 0.3-2.0%.
M3: |E60a - E60b| <= 0.8%.  D1: Welch CI of E60a - E1 below 0 and |effect| > MDE 0.8% -> new claim.
D2: CI of E60a - E60b below 0 -> learned routing matters.  D3: E60a vs E60c per measured GPU-minute.
M5 (expert shares) needs router statistics that the runs did not save: reported as not measured.
M8: tok/s of E60a and E60c vs E1 (descriptive; E1 tok/s from the 5090 Phase 2b summaries if present locally).

Input: experiments/checks/results/e60_stage1_runs.json (built from run summary.json + doclevel files on the fleet copies).
Run: python experiments/checks/e60_stage1.py -> experiments/checks/results/e60_stage1.json
"""
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "checks", "results")


def tq(df):
    z = 1.959963984540054
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d, se = st.mean(a) - st.mean(b), math.sqrt(va + vb)
    t = tq(df)
    return {"rel": math.expm1(d), "ci95": [math.expm1(d - t * se), math.expm1(d + t * se)], "df": df,
            "n": [len(a), len(b)], "dz_like": d / math.sqrt((st.variance(a) + st.variance(b)) / 2)}


def main():
    runs = json.load(open(os.path.join(RES, "e60_stage1_runs.json")))["runs"]
    grid = {}
    for r in runs:
        if r["tokens"] < 1e8:
            grid.setdefault(r["arm"], {})[r["lr"]] = r["clean80"]
    best = {a: min(g, key=g.get) for a, g in grid.items()}
    long = {}
    for r in runs:
        if r["tokens"] > 1e8 and abs(r["lr"] - best[r["arm"]]) < 1e-12:
            long.setdefault(r["arm"], {})[r["seed"]] = r
    y = {a: [math.log(v["clean80"]) for _, v in sorted(d.items())] for a, d in long.items()}
    e1 = [math.log(v) for v in json.load(open(os.path.join(RES, "phase2b_5090.json")))["E1"].values()]
    m1, m2, m2b, m3 = welch(y["E60a"], e1), welch(y["E60c"], e1), welch(y["E60a"], y["E60c"]), welch(y["E60a"], y["E60b"])
    # paired-by-seed version of the a-b contrast (same data order per seed)
    seeds = sorted(set(long["E60a"]) & set(long["E60b"]))
    dd = [math.log(long["E60a"][s]["clean80"] / long["E60b"][s]["clean80"]) for s in seeds]
    mdd, sdd = st.mean(dd), st.stdev(dd) / math.sqrt(len(dd))
    m3_paired = {"rel": math.expm1(mdd), "ci95": [math.expm1(mdd - tq(len(dd) - 1) * sdd), math.expm1(mdd + tq(len(dd) - 1) * sdd)],
                 "df": len(dd) - 1, "per_seed": [math.expm1(x) for x in dd]}
    perf = {a: {"tokens_per_s_mean": st.mean(v["tokens_per_s"] for v in d.values()),
                "wall_min_mean": st.mean(v["wall_min"] for v in d.values()),
                "params_non_embedding": next(iter(d.values()))["params_non_embedding"]} for a, d in long.items()}
    # D3: bpc per GPU-minute is not a single number; report each arm's gain vs E1 next to its wall time
    out = {
        "grid_20M_seed0": grid, "best_lr": best, "n_per_arm_200M": {a: len(d) for a, d in long.items()},
        "mean_clean80": {a: math.exp(st.mean(v)) for a, v in y.items()} | {"E1_7seed": math.exp(st.mean(e1))},
        "M1_E60a_vs_E1": m1, "M1_in_range": -0.035 <= m1["rel"] <= -0.008,
        "M2_E60c_vs_E1": m2, "M2_in_range": -0.045 <= m2["rel"] <= -0.015,
        "M2b_E60a_vs_E60c": m2b, "M2b_in_range": 0.003 <= m2b["rel"] <= 0.020,
        "M3_E60a_vs_E60b_welch": m3, "M3_E60a_vs_E60b_paired": m3_paired, "M3_pass": abs(m3["rel"]) <= 0.008,
        "D1": "new claim (CI below 0, |effect| > 0.8%)" if m1["ci95"][1] < 0 and abs(m1["rel"]) > 0.008 else "MoE buys nothing measurable",
        "D2": "learned routing matters (CI of a-b below 0)" if m3["ci95"][1] < 0 else "the gain is capacity, not routing",
        "D3_perf": perf, "M5": "not measured (router statistics were not saved by the runs)",
    }
    json.dump(out, open(os.path.join(RES, "e60_stage1.json"), "w", newline="\n"), indent=1)
    for k in ("best_lr", "mean_clean80", "M1_E60a_vs_E1", "M1_in_range", "M2_E60c_vs_E1", "M2_in_range", "M2b_E60a_vs_E60c",
              "M2b_in_range", "M3_E60a_vs_E60b_welch", "M3_E60a_vs_E60b_paired", "M3_pass", "D1", "D2", "D3_perf"):
        print(k, json.dumps(out[k]))


if __name__ == "__main__":
    main()
