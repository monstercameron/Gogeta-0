"""SQ400 (registered in PREDICTIONS.md, section SQ400): the out-of-corpus test of C213 with seeds 2-7 and paired-by-seed CIs.

Re-uses the per-document scores of SQ379 (seeds 2-4: A2s, A2f, E27, E24, E20, from results/sq379_part*.json) and scores the missing checkpoints (seeds 5-7 of A2s, A2f,
E20, E24, and A3mx 5-7) on the SQ99 out-of-corpus set (330 files, sha-verified) plus the clean80 300-doc subset, with the CPU scorer (fp32, K8 windowing, window 1024).
  python sq400_ooc_registered.py --part 0|1   score half of the missing runs -> results/sq400_part{0,1}.json
  python sq400_ooc_registered.py --analyze    all effects with BOTH intervals -> results/sq400.json
Effects (per the registration): mean over paired seeds of 100 x (bpc_arm / bpc_base - 1) on aggregate bits/char, out of corpus (all 330 files; the stdlib slice; the
site-packages slice) and on clean80-300; intervals: (i) paired-by-seed t-interval (df = n - 1, seed variance included; the interval the rules use), (ii) package-clustered
bootstrap over documents (B = 2000, as SQ379; document sampling only). Rules are evaluated at the end and written into the JSON.
"""
import glob
import json
import math
import sys
import time

import numpy as np

import sq339_cpu as C
from sq379_a2_out_of_corpus import boot, find_runs as find_runs_379, ooc_docs

OUT = C.RES / "sq400.json"
ARMS = {"A2s": 0.006, "A2f": 0.006, "E20": 0.003, "E24": 0.003, "E27": 0.006, "A3mx": 0.006}
SEEDS = (2, 3, 4, 5, 6, 7)


def find_runs():
    import sq379_a2_out_of_corpus as Q

    Q.ARMS, Q.SEEDS = ARMS, SEEDS
    return Q.find_runs()


def main():
    log = C.make_log(str(C.RES / "sq400.log"))
    runs = find_runs()
    if "--analyze" in sys.argv:
        return analyze(runs, log)
    part = int(sys.argv[sys.argv.index("--part") + 1])
    have = set()
    for f in glob.glob(str(C.RES / "sq379_part*.json")):
        have |= set(json.load(open(f))["scores"])
    todo = sorted((k for k in runs if f"{k[0]}_s{k[1]}" not in have), key=lambda k: (k[1], k[0]))
    mine = todo[part::2]
    docs, bad = ooc_docs()
    small = C.subset(300, 0)
    log(f"part {part}: {len(mine)} of {len(todo)} new runs {mine}; ooc docs {len(docs)}")
    res = {"scores": {}}
    for k in mine:
        t0 = time.time()
        model = C.load_model(runs[k])
        o = C.score_docs(C.fwd_default(model), [(i, t) for i, (_, t) in enumerate(docs)])
        s = C.score_docs(C.fwd_default(model), small)
        res["scores"][f"{k[0]}_s{k[1]}"] = {"run": runs[k], "ooc": o, "clean80_300": s, "ooc_bpc": C.bpc(o), "clean80_300_bpc": C.bpc(s)}
        log(f"{k} {runs[k]}: ooc {C.bpc(o):.4f} clean80-300 {C.bpc(s):.4f} ({time.time() - t0:.0f}s)")
        json.dump(res, open(C.RES / f"sq400_part{part}.json", "w"))


def tint(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return {"mean": m, "n": n, "ci95": [float("nan")] * 2}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}[n - 1]
    return {"mean": m, "sd": sd, "n": n, "ci95": [m - t * sd / math.sqrt(n), m + t * sd / math.sqrt(n)]}


def analyze(runs, log):
    scores = {}
    for f in sorted(glob.glob(str(C.RES / "sq379_part*.json")) + glob.glob(str(C.RES / "sq400_part*.json"))):
        d = json.load(open(f))
        scores.update(d["scores"])
        if "ooc_docs" in d:
            docs_meta = d["ooc_docs"]
    groups = [d["source"] for d in docs_meta]
    chars = np.array([d["chars"] for d in docs_meta], float)
    stdlib = np.array([g == "stdlib" for g in groups])
    out = {"check": "SQ400", "git": C.git_commit(), "status": "registered (PREDICTIONS.md SQ400)", "n_ooc_docs": len(groups), "n_stdlib_docs": int(stdlib.sum()),
           "runs": {k: v["run"] for k, v in scores.items()}, "effects": {}}
    comps = [("A2s", "A2f"), ("A2s", "E27"), ("A2f", "E20"), ("E27", "E20"), ("A2f", "E24"), ("A3mx", "E20"), ("A3mx", "E24"), ("A3mx", "A2s"), ("A2s", "E20"), ("A2s", "E24")]
    for a, b in comps:
        seeds = [s for s in SEEDS if f"{a}_s{s}" in scores and f"{b}_s{s}" in scores]
        if len(seeds) < 2:
            continue
        pairs = [(f"{a}_s{s}", f"{b}_s{s}") for s in seeds]
        nll = {k: v["ooc"]["nll_nats"] for k, v in scores.items()}

        def agg(key, mask=None):
            x = np.asarray(nll[key]) if mask is None else np.asarray(nll[key])[mask]
            c = chars if mask is None else chars[mask]
            return x.sum() / c.sum()

        per = {"all": [100 * (agg(x) / agg(y) - 1) for x, y in pairs],
               "stdlib": [100 * (agg(x, stdlib) / agg(y, stdlib) - 1) for x, y in pairs],
               "site_packages": [100 * (agg(x, ~stdlib) / agg(y, ~stdlib) - 1) for x, y in pairs],
               "clean80_300": [100 * (scores[x]["clean80_300_bpc"] / scores[y]["clean80_300_bpc"] - 1) for x, y in pairs]}
        e = {"seeds": seeds, "per_seed_pct": per, "paired_t": {k: tint(v) for k, v in per.items()}}
        e["cluster_boot_all"] = boot(nll, chars, groups, pairs)
        e["ratio_ooc_over_clean80"] = e["paired_t"]["all"]["mean"] / e["paired_t"]["clean80_300"]["mean"] if abs(e["paired_t"]["clean80_300"]["mean"]) > 1e-9 else None
        out["effects"][f"{a}_vs_{b}"] = e
        t = e["paired_t"]
        log(f"{a} vs {b} n={len(seeds)}: ooc {t['all']['mean']:+.2f}% paired CI [{t['all']['ci95'][0]:+.2f}, {t['all']['ci95'][1]:+.2f}] cluster CI "
            f"[{e['cluster_boot_all']['ci95_cluster_boot'][0]:+.2f}, {e['cluster_boot_all']['ci95_cluster_boot'][1]:+.2f}]; stdlib {t['stdlib']['mean']:+.2f} site {t['site_packages']['mean']:+.2f}; clean80-300 {t['clean80_300']['mean']:+.2f}")
    E = out["effects"]
    rules = {}
    if "A2s_vs_A2f" in E:
        t = E["A2s_vs_A2f"]["paired_t"]
        rules["P1"] = {"mean_within_1.5": abs(t["all"]["mean"]) < 1.5, "ci_includes_0": t["all"]["ci95"][0] <= 0 <= t["all"]["ci95"][1], "clean80_above_7": t["clean80_300"]["mean"] > 7}
        rules["P6"] = {"same_sign_slices": (t["stdlib"]["mean"] > 0) == (t["site_packages"]["mean"] > 0),
                       "no_slice_ci_above_+3": not (t["stdlib"]["ci95"][0] > 3 or t["site_packages"]["ci95"][0] > 3)}
    if "A2s_vs_E27" in E:
        t = E["A2s_vs_E27"]["paired_t"]["all"]
        rules["P2"] = {"mean_in_-13_-9": -13 <= t["mean"] <= -9, "ci_excludes_-5": t["ci95"][1] < -5}
    if "A2f_vs_E20" in E:
        rules["P3"] = {"mean_in_0.5_3": 0.5 <= E["A2f_vs_E20"]["paired_t"]["all"]["mean"] <= 3}
    if "E27_vs_E20" in E:
        rules["P4"] = {"ratio_in_0.4_0.65": E["E27_vs_E20"]["ratio_ooc_over_clean80"] is not None and 0.4 <= E["E27_vs_E20"]["ratio_ooc_over_clean80"] <= 0.65}
    if "A3mx_vs_E20" in E:
        t = E["A3mx_vs_E20"]["paired_t"]
        rules["P5"] = {"same_sign": (t["all"]["mean"] > 0) == (t["clean80_300"]["mean"] > 0), "smaller_magnitude": abs(t["all"]["mean"]) < abs(t["clean80_300"]["mean"])}
    if "A2f_vs_E24" in E:
        t = E["A2f_vs_E24"]["paired_t"]["all"]
        rules["P7"] = {"ci_includes_0_or_favours_loops": t["ci95"][0] <= 0}
    if "A2s_vs_A2f" in E and "A2f_vs_E24" in E:
        a = E["A2s_vs_A2f"]["paired_t"]
        l = E["A2f_vs_E24"]["paired_t"]["all"]
        rules["triage_C213_changes_priorities"] = {"ternary_ooc_ci_upper_below_+3": a["all"]["ci95"][1] < 3, "ternary_clean80_above_+5": a["clean80_300"]["mean"] > 5,
                                                   "n_seeds_ternary": a["all"]["n"], "loop_ooc_ci_includes_0_or_favours": l["ci95"][0] <= 0, "n_seeds_loop": l["n"]}
        r = rules["triage_C213_changes_priorities"]
        r["both_hold_with_n_ge_3"] = bool(r["ternary_ooc_ci_upper_below_+3"] and r["ternary_clean80_above_+5"] and r["loop_ooc_ci_includes_0_or_favours"] and r["n_seeds_ternary"] >= 3 and r["n_seeds_loop"] >= 3)
    out["rules"] = rules
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    log(json.dumps(rules, default=bool))


if __name__ == "__main__":
    main()
