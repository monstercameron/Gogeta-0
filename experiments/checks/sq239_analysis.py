"""Analysis of the SQ239 per-doc re-score: SQ239 (source config, parseability, clean50/clean80/full), SQ410 (overlap dose-response), SQ385 (position and length bins).

Input: results/sq239_nll/<run>.npz (sq239_perdoc_rescore.py), K9 overlap, docs_val src, results/sq239_parse_flags.npy. Output: results/sq239.json, sq410.json, sq385.json.
Effect of comparison a vs b on a doc subset M, per paired seed: 100 x (bpc_a(M) / bpc_b(M) - 1) with bpc(M) = sum nll / ln 2 / sum chars over M; mean over seeds and the
paired-by-seed 95% t-CI (df = n - 1). Comparisons (a vs b): A2s/A2f (ternary cost), A2f/E24 (loop cost), A2s/E27 (assembly gain), A2f/E20, E27/E20, A3mx/E20, A3mx/E24, A2s/E20, A2s/E24, A3mx/A2s.
SQ239 flags: (i) sign of the effect differs between the two source configs, or the configs differ by > 2 points; (ii) the clean50 effect differs from the clean80 effect by > 30% of
the clean80 effect or has the opposite sign (the rule of the 2026-11-23 15:37 extension).
SQ410: five K9 overlap bands over all 4,000 val docs; the same three headline ratios per band. SQ385: nats/token by position bin (1-128, 129-512, 513-1024, 1025+) and the first-128 vs
rest split, and aggregate bits/char by document length bin (<=128, 129-256, 257-512, 513-1024, >1024 tokens), same comparisons, clean80 docs.
Descriptive/exploratory: never changes a claim status (SQ225 row).
"""
import glob
import json
import math
import os

import numpy as np

import sq339_cpu as C

D = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
COMPS = [("A2s", "A2f"), ("A2f", "E24"), ("A2s", "E27"), ("A2f", "E20"), ("E27", "E20"), ("A3mx", "E20"), ("A3mx", "E24"), ("A2s", "E20"), ("A2s", "E24"), ("A3mx", "A2s")]
HEAD = [("A2s", "A2f"), ("A2f", "E24"), ("A2s", "E27")]


def tint(v):
    n = len(v)
    m = float(np.mean(v))
    if n < 2:
        return {"mean": m, "n": n}
    sd = float(np.std(v, ddof=1))
    h = T[n - 1] * sd / math.sqrt(n)
    return {"mean": m, "sd": sd, "n": n, "ci95": [m - h, m + h]}


def main():
    ov = np.array([0.0 if o is None else o for o in json.load(open(C.RES / "K9.json"))["data"]["val_per_doc_overlap"]])
    rows = [json.loads(l) for l in open(os.path.join(D, "docs_val.jsonl"), encoding="utf-8")]
    src = np.array([r["src"] for r in rows])
    cfgs = sorted(set(src))
    runs = {}
    for f in glob.glob(str(C.RES / "sq239_nll" / "*.npz")):
        z = np.load(f)
        runs[(str(z["arm"]), int(z["seed"]))] = {k: z[k] for k in ("nll_doc", "ntok_doc", "bin_nll", "bin_n", "chars")}
    chars = next(iter(runs.values()))["chars"]
    ln2 = math.log(2)

    def bpc(k, M):
        r = runs[k]
        return r["nll_doc"][M].sum() / ln2 / chars[M].sum()

    def eff(a, b, M, fn=bpc):
        seeds = sorted(s for (x, s) in runs if x == a and (b, s) in runs)
        return seeds, [100 * (fn((a, s), M) / fn((b, s), M) - 1) for s in seeds]

    masks = {"full": np.ones(len(ov), bool), "clean80": ov < 0.8, "clean50": ov < 0.5}
    out = {"check": "SQ239", "git": C.git_commit(), "status": "exploratory, descriptive (never changes a claim status)", "n_runs": len(runs),
           "runs": sorted(f"{a}_s{s}" for a, s in runs), "mask_sizes": {k: int(v.sum()) for k, v in masks.items()}}
    # --- thresholds and source configs
    thr, srcres, flags = {}, {}, []
    for a, b in COMPS:
        key = f"{a}_vs_{b}"
        thr[key] = {}
        for name, M in masks.items():
            seeds, e = eff(a, b, M)
            if e:
                thr[key][name] = {"seeds": seeds, "per_seed": e, **tint(e)}
        srcres[key] = {}
        for base in ("full", "clean80"):
            for c in cfgs:
                M = masks[base] & (src == c)
                seeds, e = eff(a, b, M)
                if e:
                    srcres[key][f"{base}/{c}"] = {"n_docs": int(M.sum()), "per_seed": e, **tint(e)}
        for base in ("full", "clean80"):
            if all(f"{base}/{c}" in srcres[key] for c in cfgs):
                m = [srcres[key][f"{base}/{c}"]["mean"] for c in cfgs]
                if (m[0] > 0) != (m[1] > 0) or abs(m[0] - m[1]) > 2:
                    flags.append({"comparison": key, "base": base, "type": "source", "means_by_config": dict(zip(cfgs, m)), "sign_reversal": (m[0] > 0) != (m[1] > 0), "diff_points": abs(m[0] - m[1])})
        if "clean80" in thr[key] and "clean50" in thr[key]:
            c80, c50 = thr[key]["clean80"]["mean"], thr[key]["clean50"]["mean"]
            if (c80 > 0) != (c50 > 0) or (abs(c80) > 1e-9 and abs(c50 - c80) / abs(c80) > 0.30):
                flags.append({"comparison": key, "type": "threshold", "clean80": c80, "clean50": c50, "full": thr[key].get("full", {}).get("mean")})
    out["thresholds"], out["by_source"], out["flags"] = thr, srcres, flags
    pf = C.RES / "sq239_parse_flags.npy"
    out["parseability_census"] = json.load(open(C.RES / "sq239_parseability.json"))["adds_parseability_stratum"] if (C.RES / "sq239_parseability.json").exists() else None
    json.dump(out, open(C.RES / "sq239.json", "w"), indent=1)
    # --- SQ410 overlap bands
    bands = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    o410 = {"check": "SQ410", "git": C.git_commit(), "status": "exploratory, descriptive", "bands": [f"[{lo},{hi})" for lo, hi in bands], "band_docs": [int(((ov >= lo) & (ov < hi)).sum()) for lo, hi in bands], "effects": {}}
    for a, b in HEAD + [("A2f", "E20"), ("E27", "E20")]:
        key = f"{a}_vs_{b}"
        o410["effects"][key] = {}
        for (lo, hi), nm in zip(bands, o410["bands"]):
            M = (ov >= lo) & (ov < hi)
            seeds, e = eff(a, b, M)
            if e:
                o410["effects"][key][nm] = {"per_seed": e, **tint(e)}
    # slope across bands (mean effect vs band index), as a plain descriptive number
    o410["slope_points_per_band"] = {}
    for key, d in o410["effects"].items():
        ms = [d[nm]["mean"] for nm in o410["bands"] if nm in d]
        if len(ms) >= 3:
            o410["slope_points_per_band"][key] = float(np.polyfit(range(len(ms)), ms, 1)[0])
    json.dump(o410, open(C.RES / "sq410.json", "w"), indent=1)
    # --- SQ385 position and length bins (clean80 docs)
    M80 = masks["clean80"]
    pos_names = ["1-128", "129-512", "513-1024", "1025+"]

    def nats_bin(k, M, bins):
        r = runs[k]
        return r["bin_nll"][M][:, bins].sum() / r["bin_n"][M][:, bins].sum()

    o385 = {"check": "SQ385", "git": C.git_commit(), "status": "exploratory, descriptive; distractor arm not run", "position_bins": pos_names, "effects_position": {}, "effects_first128_vs_rest": {}, "effects_length": {}}
    for a, b in COMPS:
        key = f"{a}_vs_{b}"
        o385["effects_position"][key], o385["effects_first128_vs_rest"][key], o385["effects_length"][key] = {}, {}, {}
        for i, nm in enumerate(pos_names):
            seeds = sorted(s for (x, s) in runs if x == a and (b, s) in runs)
            e = [100 * (math.exp(nats_bin((a, s), M80, [i]) - nats_bin((b, s), M80, [i])) - 1) for s in seeds]
            if e:
                o385["effects_position"][key][nm] = {"tokens_scored": int(runs[(a, seeds[0])]["bin_n"][M80][:, i].sum()), "per_seed": e, **tint(e)}
        for nm, bins in (("first128", [0]), ("rest", [1, 2, 3])):
            seeds = sorted(s for (x, s) in runs if x == a and (b, s) in runs)
            e = [100 * (math.exp(nats_bin((a, s), M80, bins) - nats_bin((b, s), M80, bins)) - 1) for s in seeds]
            if e:
                o385["effects_first128_vs_rest"][key][nm] = {"per_seed": e, **tint(e)}
        ntok = next(iter(runs.values()))["ntok_doc"]
        for lo, hi in ((0, 128), (129, 256), (257, 512), (513, 1024), (1025, 10 ** 9)):
            M = M80 & (ntok >= lo) & (ntok <= hi)
            if M.sum() < 20:
                continue
            seeds, e = eff(a, b, M)
            if e:
                o385["effects_length"][key][f"{lo}-{hi}"] = {"n_docs": int(M.sum()), "per_seed": e, **tint(e)}
    json.dump(o385, open(C.RES / "sq385.json", "w"), indent=1)
    for a, b in HEAD:
        key = f"{a}_vs_{b}"
        print(key, "thr", {k: round(v["mean"], 2) for k, v in thr[key].items()}, "src", {k: round(v["mean"], 2) for k, v in srcres[key].items()})
        print("  bands", {k: round(v["mean"], 2) for k, v in o410["effects"][key].items()})
        print("  pos", {k: round(v["mean"], 2) for k, v in o385["effects_position"][key].items()}, "len", {k: round(v["mean"], 2) for k, v in o385["effects_length"][key].items()})
    print("flags", json.dumps(flags)[:1500])


if __name__ == "__main__":
    main()
