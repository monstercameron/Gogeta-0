"""A2 amendment 3 (registered v0.5.22): compute-matched E27 for the A2 headline (C209).

E27 (e27_fact111.json, lr 6e-3) at 221.6M tokens (training FLOPs matched to A2s's 3.79e16), seeds 2-4, vs A2s at 200M
(seeds 2-4, results/a2_runs.json), paired by seed, clean80 doc-level bpc, log scale, df = n - 1.
Prediction: A2s - E27(221.6M) = -15.5% (range -17% to -13%). Rule: C209 gains a compute-matched evidence event if the
paired 95% CI lies below -5%; if the CI reaches -5% or above, C209 is REFUTED by its own criterion.

    python3 experiments/checks/a2_compute_matched.py --gather   # WSL as root: pulled a3 fleet dirs -> results/a2_cm_runs.json
    python experiments/checks/a2_compute_matched.py             # -> results/a2_compute_matched.json
"""
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstat import t_quantile, t_two_sided_p  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "checks", "results")


def gather():
    base = "/root/fusiondance-data/cloud/fleet"
    out = []
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-a3-*/fusiondance-data/runs/E27-*")):
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f)); m = s.get("metrics", {}); name = os.path.basename(d)
        dl = glob.glob(f"{base}/*/Gogeta-0/experiments/stats/doclevel/{name}.json")
        out.append({"run": name, "arm": "E27", "seed": s["seed"], "lr": s["config"]["train"].get("lr"),
                    "tokens": m.get("tokens"), "flops": m.get("flops"), "dataset": s["config"]["train"].get("dataset"),
                    "clean80": json.load(open(dl[0]))["bpc"]["clean80"] if dl else None,
                    "git": (s.get("git") or {}).get("commit"), "git_describe": (s.get("git") or {}).get("describe")})
    json.dump({"check": "A2-amendment3-runs", "runs": out}, open(os.path.join(RES, "a2_cm_runs.json"), "w", newline="\n"), indent=1)
    print(len(out), "E27 runs from the a3 fleet")


def main():
    e = {r["seed"]: r for r in json.load(open(os.path.join(RES, "a2_cm_runs.json")))["runs"]
         if r["tokens"] and r["tokens"] > 2.1e8 and r["clean80"] and abs(r["lr"] - 0.006) < 1e-12}
    a = {r["seed"]: r for r in json.load(open(os.path.join(RES, "a2_runs.json")))["runs"]
         if r["arm"] == "A2s" and r["clean80"] and r["tokens"] > 1e8}
    seeds = sorted(set(e) & set(a))
    x = [math.log(a[s]["clean80"] / e[s]["clean80"]) for s in seeds]
    out = {"seeds": seeds, "n": len(x), "per_seed": {s: math.expm1(v) for s, v in zip(seeds, x)},
           "flops": {"A2s": {s: a[s].get("flops") for s in seeds}, "E27_cm": {s: e[s]["flops"] for s in seeds}},
           "tokens_E27_cm": {s: e[s]["tokens"] for s in seeds},
           "provenance": {s: [e[s]["git"][:7], e[s]["git_describe"], e[s]["dataset"]] for s in seeds},
           "predicted_range": [-0.17, -0.13], "refutation_bound": -0.05}
    if len(x) >= 2:
        m = st.mean(x); se = st.stdev(x) / math.sqrt(len(x)); df = len(x) - 1; q = t_quantile(0.05, df)
        out.update({"rel": math.expm1(m), "df": df, "p": t_two_sided_p(m / se, df),
                    "ci95": [math.expm1(m - q * se), math.expm1(m + q * se)]})
        out["in_predicted_range"] = -0.17 <= out["rel"] <= -0.13
        out["c209_compute_matched_ok (CI below -5%)"] = out["ci95"][1] < -0.05
        out["final"] = len(x) >= 3
    json.dump(out, open(os.path.join(RES, "a2_compute_matched.json"), "w", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("flops", "provenance")}, indent=1))


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else main()
