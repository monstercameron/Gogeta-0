"""A1 assembly screen (registered in experiments/checks/PREDICTIONS.md, "A1 assembly screen", v0.5.0, before dispatch).

Eight arms, each ONE compensation on F2 cell 111 (E27), seeds 0-1, 200M tokens of code-py-v2, lr 6e-3. Reference: the SQ263
E27 runs at 6e-3, seeds 0-1 (same data order per seed). Metric: clean80 doc-level bpc, log scale.
Effect = mean over seeds of log(arm_s / E27_s). Winner: effect <= -1.0% and both seeds < 0.
Inputs: experiments/checks/results/a1_runs.json (built by --gather in WSL from the pulled fleet dirs) and sq263_runs.json.

    python3 experiments/checks/a1_screen.py --gather   # WSL: build a1_runs.json from ~/fusiondance-data/cloud/fleet
    python experiments/checks/a1_screen.py             # -> experiments/checks/results/a1_screen.json
"""
import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "checks", "results")
PRED = {"A1a": (-0.015, -0.04, 0.005), "A1b": (-0.005, -0.02, 0.005), "A1c": (-0.005, -0.025, 0.015),
        "A1d": (-0.02, -0.05, 0.0), "A1e": (-0.03, -0.06, -0.01), "A1f": (-0.03, -0.06, -0.01),
        "A1g": (-0.02, -0.05, 0.01), "A1h": (-0.01, -0.03, 0.01),
        "A1i": (-0.005, -0.02, 0.01), "A1z": (0.0, -0.005, 0.005), "A1k": (0.0, -0.01, 0.01)}


def gather():
    base = os.path.expanduser("~/fusiondance-data/cloud/fleet")
    out = []
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-a[12]-*/fusiondance-data/runs/A1*")):
        s = json.load(open(os.path.join(d, "summary.json")))
        name, m = os.path.basename(d), s.get("metrics", {})
        dl = glob.glob(f"{base}/*/Gogeta-0/experiments/stats/doclevel/{name}.json")
        tr = s["config"]["train"]
        arm = s["experiment"]
        if arm == "A1z" and abs(tr.get("lr", 0) - 0.006) > 1e-12:  # A1k = the placebo config at lr 1.2e-2 (amendment 1)
            arm = "A1k"
        out.append({"run": name, "arm": arm, "seed": s["seed"], "lr": tr.get("lr"), "tokens": m.get("tokens"),
                    "status": s.get("status"), "clean80": json.load(open(dl[0]))["bpc"]["clean80"] if dl else None,
                    "tokens_per_s": m.get("tokens_per_s"), "wall_min": m.get("wall_min"),
                    "params_non_embedding": m.get("params_non_embedding"), "git": s.get("git", {}).get("commit"),
                    "git_describe": s.get("git", {}).get("describe"), "dataset": tr.get("dataset")})
    json.dump({"check": "A1-runs", "source": "pulled fleet run summary.json + doclevel clean80 (v2 val)", "runs": out},
              open(os.path.join(RES, "a1_runs.json"), "w", newline="\n"), indent=1)
    print(len(out), "runs gathered")


def main():
    runs = [r for r in json.load(open(os.path.join(RES, "a1_runs.json")))["runs"] if r["clean80"] and r["tokens"] and r["tokens"] > 1e8
            and r["seed"] in (0, 1)]  # the screen is seeds 0-1; A1e seeds 2-4 belong to A2 amendment 1
    # provenance (quick review 11:39): registered commits only, clean tree, code-py-v2
    ok_commits = ("36acccc", "482af19", "f85ada5")  # f85ada5: A1 re-runs in the A2 wave (v0.5.4)
    bad = [r["run"] for r in runs if not str(r.get("git") or "").startswith(ok_commits)
           or "dirty" in str(r.get("git_describe") or "") or r.get("dataset") != "code-py-v2"]
    print("provenance:", "OK" if not bad else f"MISMATCH {bad}")
    ref = {r["seed"]: r for r in json.load(open(os.path.join(RES, "sq263_runs.json")))["runs"]
           if r["cell"] == "E27" and abs(r["lr"] - 0.006) < 1e-12}
    e27_tps = None
    arms = {}
    for r in runs:
        arms.setdefault(r["arm"], {})[r["seed"]] = r
    res = {}
    for a in sorted(arms):
        per = {s: math.log(v["clean80"] / ref[s]["clean80"]) for s, v in arms[a].items() if s in ref}
        eff = st.mean(per.values()) if per else None
        p, lo, hi = PRED.get(a, (None, None, None))
        res[a] = {"n_seeds": len(per), "per_seed": {s: math.expm1(x) for s, x in per.items()},
                  "effect": math.expm1(eff) if eff is not None else None,
                  "winner": eff is not None and len(per) == 2 and math.expm1(eff) <= -0.01 and all(x < 0 for x in per.values()),
                  "prediction": p, "in_predicted_range": (lo <= math.expm1(eff) <= hi) if eff is not None and lo is not None else None,
                  "mean_clean80": st.mean(v["clean80"] for v in arms[a].values()),
                  "tokens_per_s": st.mean(v["tokens_per_s"] for v in arms[a].values() if v["tokens_per_s"]),
                  "params_non_embedding": next(iter(arms[a].values()))["params_non_embedding"]}
    out = {"reference": {"E27_seeds_0_1": {s: ref[s]["clean80"] for s in (0, 1)}}, "arms": res,
           "winners": [a for a, v in res.items() if v["winner"]],
           "any_arm_improves_2pct": any(v["effect"] is not None and v["effect"] <= -0.02 for v in res.values())}
    json.dump(out, open(os.path.join(RES, "a1_screen.json"), "w", newline="\n"), indent=1)
    for a, v in res.items():
        print(f"{a}: effect {v['effect'] * 100:+.2f}% per-seed {[round(x * 100, 2) for x in v['per_seed'].values()]} "
              f"winner={v['winner']} pred {v['prediction'] * 100:+.1f}% in-range={v['in_predicted_range']} "
              f"tok/s {v['tokens_per_s']:.0f} params {v['params_non_embedding'] / 1e6:.2f}M")
    print("winners:", out["winners"])


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else main()
