"""A2 assembled model: registered analysis (experiments/checks/PREDICTIONS.md, "A2 assembled model" P1-P5, plus
"A2 amendment 1" (A1e on fresh seeds) and the loop go/no-go rule of "A2 amendment 2" (disclosed post hoc)).

Runs: 200M tokens of code-py-v2, fresh seeds 2-4, clean80 doc-level bpc (log scale), paired by seed (shared data order).
Arms: A2s (assembled), A2f (A2s without ternary), A2a (A2s + per-pass adapters), A1e (ternary core only, amendment 1);
references E20 (dense, 3e-3), E24 (GDN hybrid unshared, 3e-3), E27 (full fusion, 6e-3); seed 2 of the references comes
from SQ263 (sq263_runs.json), seeds 3-4 from this wave.

    python3 experiments/checks/a2_assembled.py --gather   # WSL: pulled fleet dirs -> results/a2_runs.json
    python experiments/checks/a2_assembled.py             # -> results/a2_assembled.json
"""
import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "checks", "results")
LR = {"E20": 0.003, "E24": 0.003, "E27": 0.006}


def gather():
    base = os.path.expanduser("~/fusiondance-data/cloud/fleet")
    out = []
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-a2-*/fusiondance-data/runs/*")):
        name = os.path.basename(d)
        arm = name.split("-")[0]
        if arm not in ("A2s", "A2f", "A2a", "A1e", "E20", "E24", "E27"):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        m, tr = s.get("metrics", {}), s["config"]["train"]
        dl = glob.glob(f"{base}/*/Gogeta-0/experiments/stats/doclevel/{name}.json")
        out.append({"run": name, "arm": arm, "seed": s["seed"], "lr": tr.get("lr"), "tokens": m.get("tokens"),
                    "clean80": json.load(open(dl[0]))["bpc"]["clean80"] if dl else None,
                    "tokens_per_s": m.get("tokens_per_s"), "wall_min": m.get("wall_min"), "flops": m.get("flops"),
                    "params_non_embedding": m.get("params_non_embedding"), "params_total": m.get("params_total"),
                    "git": s.get("git", {}).get("commit"), "git_describe": s.get("git", {}).get("describe"),
                    "dataset": tr.get("dataset")})
    json.dump({"check": "A2-runs", "source": "pulled fleet run summary.json + doclevel clean80", "runs": out},
              open(os.path.join(RES, "a2_runs.json"), "w", newline="\n"), indent=1)
    print(len(out), "runs gathered")


def tq(df):
    z = 1.959963984540054
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [math.log(a[s] / b[s]) for s in seeds]
    m = st.mean(d)
    out = {"n": len(d), "seeds": seeds, "rel": math.expm1(m), "per_seed": {s: math.expm1(x) for s, x in zip(seeds, d)}}
    if len(d) >= 2:
        se = st.stdev(d) / math.sqrt(len(d)); t = tq(len(d) - 1)
        out.update({"ci95": [math.expm1(m - t * se), math.expm1(m + t * se)], "df": len(d) - 1})
    return out


def main():
    runs = [r for r in json.load(open(os.path.join(RES, "a2_runs.json")))["runs"] if r["clean80"] and r["tokens"] and r["tokens"] > 1e8]
    bad = [r["run"] for r in runs if not str(r.get("git") or "").startswith("f85ada5") or "dirty" in str(r.get("git_describe") or "")
           or r.get("dataset") != "code-py-v2"]
    print("provenance:", "OK" if not bad else f"MISMATCH {bad}")
    y = {}
    for r in runs:
        if r["arm"] in LR and abs(r["lr"] - LR[r["arm"]]) > 1e-12:
            continue
        y.setdefault(r["arm"], {})[r["seed"]] = r["clean80"]
    for r in json.load(open(os.path.join(RES, "sq263_runs.json")))["runs"]:   # reference seed 2 from SQ263
        if r["seed"] == 2 and r["cell"] in LR and abs(r["lr"] - LR[r["cell"]]) < 1e-12:
            y.setdefault(r["cell"], {}).setdefault(2, r["clean80"])
    P = {"P1_A2s_vs_E27": (paired(y["A2s"], y["E27"]), (-0.25, -0.10)),
         "P2_A2s_vs_E20": (paired(y["A2s"], y["E20"]), (0.0, 0.18)),
         "P3_A2s_vs_A2f": (paired(y["A2s"], y["A2f"]), (0.03, 0.12)),
         "P4_A2f_vs_E24": (paired(y["A2f"], y["E24"]), (0.02, 0.10)),
         "P5_A2a_vs_A2s": (paired(y.get("A2a", {}), y["A2s"]), (-0.03, 0.01)),
         "AM1_A1e_vs_E27": (paired(y.get("A1e", {}), y["E27"]), (-0.09, -0.03))}
    res = {}
    for k, (c, (lo, hi)) in P.items():
        c["in_predicted_range"] = lo <= c["rel"] <= hi if c["n"] else None
        res[k] = c
    d = res
    dec = {
        "stack_confirmed (A2s beats E27 by >= 5%, CI below 0)": d["P1_A2s_vs_E27"]["rel"] <= -0.05 and d["P1_A2s_vs_E27"].get("ci95", [0, 0])[1] < 0,
        "competitive_per_FLOP (A2s or A2f within 5% of E20)": abs(d["P2_A2s_vs_E20"]["rel"]) <= 0.05 or abs(paired(y["A2f"], y["E20"])["rel"]) <= 0.05,
        "A2f_vs_E20": paired(y["A2f"], y["E20"]),
        "A2a_replaces_A2s (P5 CI below 0)": d["P5_A2a_vs_A2s"].get("ci95", [0, 0])[1] < 0,
        "A1e_confirmed (amendment 1: CI below 0 and <= -1%)": d["AM1_A1e_vs_E27"]["rel"] <= -0.01 and d["AM1_A1e_vs_E27"].get("ci95", [0, 0])[1] < 0,
        "loops_stay_in_E61_default (amendment 2, post hoc: A2f beats E24 by >= 1%)": d["P4_A2f_vs_E24"]["rel"] <= -0.01,
    }
    # secondary readouts: weight memory (ternary core 1.58 bits/param, the rest 16 bits) and throughput
    sec = {}
    for arm in ("A2s", "A2f", "A2a", "E20", "E24", "E27"):
        rr = [r for r in runs if r["arm"] == arm and (arm not in LR or abs(r["lr"] - LR[arm]) < 1e-12)]
        if not rr:
            continue
        sec[arm] = {"tokens_per_s": st.mean(r["tokens_per_s"] for r in rr), "wall_min": st.mean(r["wall_min"] for r in rr),
                    "params_total": rr[0]["params_total"], "params_non_embedding": rr[0]["params_non_embedding"]}
    out = {"clean80": {a: {s: v for s, v in sorted(d2.items())} for a, d2 in y.items()}, "predictions": res, "decisions": dec,
           "secondary_throughput_params": sec,
           "note": "per-weight-memory readout needs the ternary/full-precision param split per arm (computed separately)"}
    json.dump(out, open(os.path.join(RES, "a2_assembled.json"), "w", newline="\n"), indent=1, default=str)
    for k, v in res.items():
        print(k, f"{v['rel'] * 100:+.2f}%", "CI", [round(x * 100, 2) for x in v.get("ci95", [])], "n", v["n"],
              "per-seed", {s: round(x * 100, 2) for s, x in v["per_seed"].items()}, "in-range", v["in_predicted_range"])
    for k, v in dec.items():
        print(k, v if not isinstance(v, dict) else f"{v['rel'] * 100:+.2f}% n={v['n']}")


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else main()
