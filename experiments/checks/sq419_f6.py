"""SQ419 readout (registered in PREDICTIONS.md, section "SQ419", before dispatch; this file's sha256-16 is recorded there).

F-6 = A1g (`experiments/staged-configs/a1/a1g.json`): E27's recipe (all-ternary, random r, plain injection, bptt_loops 4) with ONE
change, a 6-block core [gdn,gdn,swa,gdn,gdn,swa] with r ~ U{1..3}: the same 14 block applications per token as E27, about 1.6x
its unique non-embedding parameters. It is the parameter-matched control for C209 (A2s vs E27): A2s has the same 6-block core,
so F-6 vs E27 prices the shape and its extra parameters, and A2s vs F-6 prices the compensations (fixed r = 2, full-precision
prelude/coda/inject with a ternary core, contractive injection) at matched shape.

Runs: F-6 at lr 6e-3, 200M tokens of code-py-v2, seeds 2, 3, 4 (paired with the existing A2s and E27 runs of A2 / C209), plus an
LR bracket on seed 5 at 3e-3, 6e-3 and 1.2e-2 (A3 amendment 1's rule: 6e-3 is LR-robust if it is within 1% of the bracket's best).

    python3 experiments/checks/sq419_f6.py --gather   # WSL: pulled fleet dirs -> results/sq419_runs.json (F-6 + comparators)
    python experiments/checks/sq419_f6.py             # -> results/sq419.json

Metric: clean80 doc-level bpc, paired by seed (shared data order), log scale, df = n - 1, exact t. Secondary (descriptive):
clean50 and the 50-80 overlap band (bits and chars derived from the clean80 and clean50 aggregates of each run's doclevel file).
Primary estimand: S = mean_s log(F6_s / E27_s) / mean_s log(A2s_s / E27_s), the share of C209's log gap that the shape (with its
parameters) accounts for. Rules are evaluated here and written into the JSON; see PREDICTIONS.md for the registered text.
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
SEEDS = (2, 3, 4)
LR_MAIN = 0.006
BRACKET_SEED = 5
# Amendment 1 (v0.5.51, before any C1 result existed; outside review 19:18): 1.5e-3 added as the lower edge for F-6 AND
# A2s (A2s's seed-5 bracket was best at its lowest point, 3e-3), the FLOP-matched E27_cm as a second R1 comparator, an
# interval for S (the S rule now reads the interval), and a comparator provenance list.
BRACKET = (0.0015, 0.003, 0.006, 0.012)
A2S_BRACKET_PRIOR = {0.003: 0.27291090098024345, 0.006: 0.27449415501561863, 0.012: 0.2836742872320863}  # a3_assembled.json
PRED = {"R1_F6_vs_E27": (-0.14, -0.09), "R2_A2s_vs_F6": (-0.08, -0.03), "S": (0.5, 0.8)}


def _doclevel(base, run):
    hits = glob.glob(f"{base}/*/Gogeta-0/experiments/stats/doclevel/{run}.json")
    return json.load(open(hits[0])) if hits else None


def gather():
    base = os.path.expanduser("~/fusiondance-data/cloud/fleet")
    rows = []
    # F-6 runs: arm A1g in the C1 fleet dirs
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-c1-*/fusiondance-data/runs/A1g-*")):
        run = os.path.basename(d)
        s = json.load(open(os.path.join(d, "summary.json")))
        m, tr = s.get("metrics", {}), s["config"]["train"]
        rows.append({"run": run, "arm": "F6", "seed": s["seed"], "lr": tr.get("lr"), "tokens": m.get("tokens"),
                     "flops": m.get("flops"), "params_non_embedding": m.get("params_non_embedding"),
                     "wall_min": m.get("wall_min"), "git": s.get("git", {}).get("commit"),
                     "git_describe": s.get("git", {}).get("describe"), "dataset": tr.get("dataset"),
                     "doclevel": _doclevel(base, run)})
    # comparators: the exact A2 P1 runs (a2_runs.json; E27 seed 2 from sq263_runs.json, lr 6e-3), clean50/chars from doclevel
    a2 = [r for r in json.load(open(os.path.join(RES, "a2_runs.json")))["runs"]
          if r["arm"] in ("A2s", "E27") and r["seed"] in SEEDS and r["tokens"] and r["tokens"] > 1e8
          and abs(r["lr"] - LR_MAIN) < 1e-12]
    e27s2 = [r for r in json.load(open(os.path.join(RES, "sq263_runs.json")))["runs"]
             if r["cell"] == "E27" and r["seed"] == 2 and abs(r["lr"] - LR_MAIN) < 1e-12]
    for r in a2 + [{**r, "arm": "E27"} for r in e27s2]:
        rows.append({"run": r["run"], "arm": r["arm"], "seed": r["seed"], "lr": r["lr"], "clean80_ref": r["clean80"],
                     "git": r.get("git"), "doclevel": _doclevel(base, r["run"])})
    # amendment 1: FLOP-matched E27 (221.6M tokens, a2_cm_runs.json) and A2s's 1.5e-3 seed-5 bracket run (C1 box)
    for r in json.load(open(os.path.join(RES, "a2_cm_runs.json")))["runs"]:
        rows.append({"run": r["run"], "arm": "E27cm", "seed": r["seed"], "lr": r["lr"], "clean80_ref": r["clean80"],
                     "flops": r.get("flops"), "git": r.get("git"), "doclevel": _doclevel(base, r["run"])})
    for d in sorted(glob.glob(f"{base}/gogeta-cockpit-c1-*/fusiondance-data/runs/A2s-*")):
        run = os.path.basename(d)
        s = json.load(open(os.path.join(d, "summary.json")))
        rows.append({"run": run, "arm": "A2s_bracket", "seed": s["seed"], "lr": s["config"]["train"].get("lr"),
                     "tokens": s.get("metrics", {}).get("tokens"), "git": s.get("git", {}).get("commit"),
                     "doclevel": _doclevel(base, run)})
    json.dump({"check": "SQ419-runs", "rows": rows}, open(os.path.join(RES, "sq419_runs.json"), "w", newline="\n"), indent=1)
    print(f"gathered {len(rows)} rows; missing doclevel: {[r['run'] for r in rows if not r['doclevel']]}")


def metric(row, which):
    d = row.get("doclevel")
    if which == "clean80" and not d:
        return row.get("clean80_ref")
    if not d:
        return None
    b = d["bpc"]
    if which in ("clean80", "clean50"):
        return b[which]
    if which == "band50_80":  # chars-weighted: bits(clean80) - bits(clean50) over chars(clean80) - chars(clean50)
        c80, c50 = d["chars"]["clean80"], d["chars"]["clean50"]
        return (b["clean80"] * c80 - b["clean50"] * c50) / (c80 - c50)
    raise KeyError(which)


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    dl = [math.log(a[s] / b[s]) for s in seeds]
    m = st.mean(dl)
    out = {"n": len(dl), "seeds": seeds, "mean_log": m, "rel": math.expm1(m),
           "per_seed": {s: math.expm1(x) for s, x in zip(seeds, dl)}}
    if len(dl) >= 2:
        se = st.stdev(dl) / math.sqrt(len(dl))
        t = t_quantile(0.05, len(dl) - 1)
        out.update(df=len(dl) - 1, ci95=[math.expm1(m - t * se), math.expm1(m + t * se)],
                   p=t_two_sided_p(m / se, len(dl) - 1) if se > 0 else 0.0)
    return out


def main():
    rows = json.load(open(os.path.join(RES, "sq419_runs.json")))["rows"]
    f6 = [r for r in rows if r["arm"] == "F6"]
    bad = [r["run"] for r in f6 if r.get("dataset") != "code-py-v2" or "dirty" in str(r.get("git_describe") or "")
           or not r.get("tokens") or r["tokens"] < 1.99e8]
    out = {"check": "SQ419 (F-6 = A1g on fresh seeds; parameter-matched control for C209)", "provenance_bad": bad,
           "predicted_ranges": PRED, "metrics": {}}
    for which in ("clean80", "clean50", "band50_80"):
        y = {}
        for r in rows:
            if r["arm"] == "F6" and (r["seed"] not in SEEDS or abs(r["lr"] - LR_MAIN) > 1e-12):
                continue
            v = metric(r, which)
            if v is not None:
                y.setdefault(r["arm"], {})[r["seed"]] = v
        if not all(k in y for k in ("F6", "E27", "A2s")):
            out["metrics"][which] = {"missing": [k for k in ("F6", "E27", "A2s") if k not in y]}
            continue
        r1, r2, c209 = paired(y["F6"], y["E27"]), paired(y["A2s"], y["F6"]), paired(y["A2s"], y["E27"])
        S = r1["mean_log"] / c209["mean_log"]
        per_seed_S = {s: math.log(y["F6"][s] / y["E27"][s]) / math.log(y["A2s"][s] / y["E27"][s])
                      for s in sorted(set(y["F6"]) & set(y["E27"]) & set(y["A2s"]))}
        vals = list(per_seed_S.values())
        s_ci = None
        if len(vals) >= 2:  # amendment 1: t-interval over the per-seed shares (both numerator and denominator vary)
            h = t_quantile(0.05, len(vals) - 1) * st.stdev(vals) / math.sqrt(len(vals))
            s_ci = [st.mean(vals) - h, st.mean(vals) + h]
        r1_cm = paired(y["F6"], y["E27cm"]) if "E27cm" in y else None
        out["metrics"][which] = {"R1_F6_vs_E27": r1, "R2_A2s_vs_F6": r2, "C209_A2s_vs_E27": c209, "S": S,
                                 "S_per_seed": per_seed_S, "S_ci95_per_seed_t": s_ci,
                                 "R1cm_F6_vs_E27_flop_matched": r1_cm}
    pm = out["metrics"].get("clean80", {})
    if "S" in pm:
        S, r1, r2 = pm["S"], pm["R1_F6_vs_E27"], pm["R2_A2s_vs_F6"]
        out["in_predicted_range"] = {"R1": PRED["R1_F6_vs_E27"][0] <= r1["rel"] <= PRED["R1_F6_vs_E27"][1],
                                     "R2": PRED["R2_A2s_vs_F6"][0] <= r2["rel"] <= PRED["R2_A2s_vs_F6"][1],
                                     "S": PRED["S"][0] <= S <= PRED["S"][1]}
        lo, hi = pm["S_ci95_per_seed_t"] or (S, S)
        out["rule_S"] = ("MOSTLY SHAPE: C209 wording gains 'most of the gap is the 6-block core shape and its extra unique "
                         "parameters' (S interval lower bound >= 0.5)" if lo >= 0.5 else
                         "MOSTLY COMPENSATIONS: the compensations carry most of C209's gap (S interval upper bound < 1/3)"
                         if hi < 1 / 3 else
                         "BOTH: the S interval spans a threshold; shape and compensations each carry a share "
                         "(point estimate S = %.2f)" % S)
        if pm.get("R1cm_F6_vs_E27_flop_matched"):
            c = pm["R1cm_F6_vs_E27_flop_matched"]
            out["R1cm_note"] = ("R1 at matched training FLOPs (F-6 200M vs E27 221.6M): %+.2f%% %s" %
                                (100 * c["rel"], [round(100 * x, 2) for x in c.get("ci95", [])]))
        out["rule_R1"] = ("A1g's parameter-changed gain REPLICATES on fresh seeds (CI below 0)" if r1["ci95"][1] < 0
                          else "A1g's gain does not replicate on fresh seeds (CI reaches 0)")
        out["rule_R2"] = ("NEW CLAIM: the compensations improve the model at matched shape (paired CI below 0)"
                          if r2["ci95"][1] < 0 else "no compensation effect shown at matched shape (CI reaches 0)")
    # LR bracket on seed 5
    br = {r["lr"]: metric(r, "clean80") for r in f6 if r["seed"] == BRACKET_SEED and r["lr"] in BRACKET}
    if all(lr in br and br[lr] for lr in BRACKET):
        best = min(br.values())
        gap = br[LR_MAIN] / best - 1
        out["lr_bracket_seed5"] = {"clean80": br, "gap_of_6e-3_to_best": gap,
                                   "lr_robust": gap <= 0.01, "best_lr": min(br, key=br.get)}
    else:
        out["lr_bracket_seed5"] = {"missing": [lr for lr in BRACKET if lr not in br]}
    out["lr_bracket_seed5"]["note"] = "one-seed check (seed 5); not a seed-averaged LR optimum"
    # amendment 1: A2s's bracket extended below with 1.5e-3 (3e-3 / 6e-3 / 1.2e-2 from A3 amendment 1)
    a2b = dict(A2S_BRACKET_PRIOR)
    for r in rows:
        if r["arm"] == "A2s_bracket" and r["seed"] == BRACKET_SEED and abs(r["lr"] - 0.0015) < 1e-12:
            a2b[0.0015] = metric(r, "clean80")
    if a2b.get(0.0015):
        best = min(a2b.values())
        out["a2s_lr_bracket_seed5"] = {"clean80": a2b, "best_lr": min(a2b, key=a2b.get),
                                       "gap_of_6e-3_to_best": a2b[LR_MAIN] / best - 1,
                                       "lr_robust": a2b[LR_MAIN] / best - 1 <= 0.01}
    else:
        out["a2s_lr_bracket_seed5"] = {"missing": [0.0015]}
    out["comparator_provenance"] = sorted({(r["arm"], r["seed"], r.get("git") or "?") for r in rows if r["arm"] != "F6"})
    out["eval_depth"] = "each arm scored at its config's `loops` (E27 4, F-6 2, A2s 2), as in every A-series readout"
    json.dump(out, open(os.path.join(RES, "sq419.json"), "w", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=1))
    if "S" in pm:
        print("clean80: R1 %+.2f%% %s | R2 %+.2f%% %s | S %.2f" % (100 * pm["R1_F6_vs_E27"]["rel"], [round(100 * x, 2) for x in pm["R1_F6_vs_E27"]["ci95"]],
              100 * pm["R2_A2s_vs_F6"]["rel"], [round(100 * x, 2) for x in pm["R2_A2s_vs_F6"]["ci95"]], pm["S"]))


if __name__ == "__main__":
    gather() if "--gather" in sys.argv else main()
