"""GPU batch B1 readout (experiments/checks/PREDICTIONS.md: "GPU batch B1", "B1 amendment 1" (SQ350), "B1 amendment 2"
(Holm family G1-G3), "B1 amendment 3" (SQ404 reading)).

20M tokens of code-py-v2, exploratory screens, clean80 doc-level bpc, log scale, paired by seed (shared data order),
df = n - 1. Same-seed replay floor on this GPU class: 0.2-0.3% (SQ232, v0.4.79).

    python3 experiments/checks/b1_readout.py --gather   # WSL as root: pulled fleet dir -> results/b1_runs.json
    python experiments/checks/b1_readout.py             # -> results/b1_readout.json
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
COMMITS = ("907065f", "c384e67")  # v0.5.31 (the 23 registered jobs), v0.5.33 (SQ350 jobs; adds only configs/docs)


def gather():
    base = "/root/fusiondance-data/cloud/fleet"
    out = []
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-b1-*/fusiondance-data/runs/*")):
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f)); m = s.get("metrics", {}); name = os.path.basename(d)
        dl = glob.glob(f"{base}/gogeta-cockpit-b1-*/Gogeta-0/experiments/stats/doclevel/{name}.json")
        out.append({"run": name, "arm": name.split("-")[0], "seed": s["seed"], "lr": s["config"]["train"].get("lr"),
                    "core_lr_mult": s["config"]["train"].get("core_lr_mult", 1.0), "tokens": m.get("tokens"),
                    "clean80": json.load(open(dl[0]))["bpc"]["clean80"] if dl else None, "val_loss": m.get("val_loss"),
                    "flops": m.get("flops"), "params_total": m.get("params_total"),
                    "params_non_embedding": m.get("params_non_embedding"), "tokens_per_s": m.get("tokens_per_s"),
                    "git": (s.get("git") or {}).get("commit"), "git_describe": (s.get("git") or {}).get("describe"),
                    "dataset": s["config"]["train"].get("dataset"), "finished": s.get("finished")})
    json.dump({"check": "B1-runs", "runs": out}, open(os.path.join(RES, "b1_runs.json"), "w", newline="\n"), indent=1)
    print(len(out), "runs gathered;", sum(r["clean80"] is not None for r in out), "with clean80")


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    x = [math.log(a[s] / b[s]) for s in seeds]
    o = {"n": len(x), "seeds": seeds, "per_seed": {s: math.expm1(v) for s, v in zip(seeds, x)}}
    if x:
        o["rel"] = math.expm1(st.mean(x)); o["_m"] = st.mean(x)
    if len(x) >= 2:
        se = st.stdev(x) / math.sqrt(len(x)); df = len(x) - 1
        o.update({"df": df, "_se": se, "p": t_two_sided_p(o["_m"] / se, df) if se else 0.0,
                  "ci95": [math.expm1(o["_m"] - t_quantile(0.05, df) * se), math.expm1(o["_m"] + t_quantile(0.05, df) * se)]})
    return o


def main():
    runs = json.load(open(os.path.join(RES, "b1_runs.json")))["runs"]
    bad = [r["run"] for r in runs if not str(r["git"]).startswith(COMMITS) or "dirty" in str(r["git_describe"])
           or r["dataset"] != "code-py-v2" or not r["clean80"] or r["tokens"] < 1.9e7]
    print("provenance:", "OK" if not bad else f"CHECK {bad}")
    y = {}
    for r in runs:
        if r["clean80"]:
            y.setdefault(r["arm"], {})[r["seed"]] = r["clean80"]
    g = lambda k: y.get(k, {})  # noqa: E731
    R = {}

    def add(key, a, b, lo, hi, kind):
        c = paired(g(a), g(b)); c.update({"arms": [a, b], "predicted_range": [lo, hi], "kind": kind})
        if "rel" in c and lo is not None:
            c["in_predicted_range"] = lo <= c["rel"] <= hi
        R[key] = c
    # SQ401
    add("SQ401_A3m_vs_A2s", "A3m", "A2s", -0.05, -0.01, "exploratory")
    add("G1_SQ401_dense2048_vs_A3m", "SQ401d", "A3m", -0.03, 0.01, "gate")
    # SQ402
    add("SQ402_beta1_vs_learned", "SQ402b1", "E53", 0.0, 0.04, "exploratory")
    add("SQ402_beta05_vs_learned", "SQ402b05", "E53", -0.005, 0.03, "exploratory")
    # SQ404: gaps to E24 per multiplier; G2 = best multiplier vs mult 1
    add("SQ404_A2f_mult1_vs_E24", "A2f", "E24", 0.0, 0.05, "exploratory")
    gaps = {}
    for arm, mult in (("SQ404c05", 0.5), ("A2f", 1.0), ("SQ404c2", 2.0)):
        c = paired(g(arm), g("E24")); gaps[mult] = c
    best = min(gaps, key=lambda m: gaps[m].get("_m", 9))
    best_arm = {0.5: "SQ404c05", 1.0: "A2f", 2.0: "SQ404c2"}[best]
    if best != 1.0:
        add("G2_SQ404_best_mult_vs_mult1", best_arm, "A2f", None, None, "gate")
    g1 = gaps[1.0].get("_m"); gb = gaps[best].get("_m")
    closure = (g1 - gb) / g1 if (g1 and g1 > 0) else None
    # SQ350 (B1 amendment 1): loops at equal params
    if g("SQ350r1"):
        add("G3_SQ350_A2f_6x2_vs_6x1", "A2f", "SQ350r1", -0.08, -0.01, "gate")
    # Holm over the gates present (m fixed at 3 per amendment 2; a missing gate cannot reject)
    gates = [k for k, v in R.items() if v["kind"] == "gate" and "p" in v]
    still = True
    for i, k in enumerate(sorted(gates, key=lambda k: R[k]["p"])):
        a = 0.05 / (3 - i); still = still and R[k]["p"] < a
        R[k]["holm"] = {"step": i + 1, "alpha": a, "reject": still}
    sq404 = {"gap_to_E24_by_mult": {str(m): {"rel": math.expm1(c["_m"]) if "_m" in c else None,
                                             "per_seed": c.get("per_seed")} for m, c in gaps.items()},
             "best_mult": best, "closure_of_gap": closure,
             "rule (closure > 0.5 -> 200M re-run at that multiplier)": (closure is not None and closure > 0.5)}
    for c in R.values():
        c.pop("_m", None); c.pop("_se", None)
    sec = {}
    for arm in sorted(y):
        rr = [r for r in runs if r["arm"] == arm]
        sec[arm] = {"n": len(rr), "flops": rr[0]["flops"], "params_total": rr[0]["params_total"],
                    "params_non_embedding": rr[0]["params_non_embedding"],
                    "tokens_per_s": st.mean(r["tokens_per_s"] for r in rr if r["tokens_per_s"])}
    out = {"metric": "clean80 doc-level bpc, 20M tokens, paired by seed", "replay_floor": "0.2-0.3% (SQ232)",
           "clean80": y, "contrasts": R, "SQ404": sq404, "secondary": sec, "provenance_check": bad}
    json.dump(out, open(os.path.join(RES, "b1_readout.json"), "w", newline="\n"), indent=1, default=str)
    for k, c in R.items():
        if "rel" not in c:
            print(k, "no data"); continue
        h = c.get("holm")
        print(f"{k:32s} {c['rel'] * 100:+6.2f}%  per-seed {[round(v * 100, 2) for v in c['per_seed'].values()]}"
              + (f"  CI {[round(v * 100, 2) for v in c['ci95']]} p={c['p']:.3f}" if "ci95" in c else "")
              + (f"  range {c.get('in_predicted_range')}" if c.get("predicted_range", [None])[0] is not None else "")
              + (f"  holm {h['step']} reject={h['reject']}" if h else ""))
    print("SQ404 gaps to E24:", {m: round(v["rel"] * 100, 2) for m, v in sq404["gap_to_E24_by_mult"].items() if v["rel"] is not None},
          "best", best, "closure", None if closure is None else round(closure, 2))
    for a, v in sec.items():
        print(f"  {a:9s} flops {v['flops']:.3g} params {v['params_total'] / 1e6:.2f}M tok/s {v['tokens_per_s'] / 1e3:.0f}k")


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else main()
