"""SQ379 (descriptive, not registered): do the A2 results hold out of corpus?

Scores the 200M A2s (ternary assembled), A2f (full-precision assembled), E27 (full fusion), E24 (GDN), E20 (dense) checkpoints, seeds 2-4
(whatever exists on the fleet pulls; E27/E24/E20 seed 2 come from the SQ263 runs), on the SQ99 out-of-corpus set: the local Python
library files of experiments/checks/results/sq_99.json (K9-scanner overlap < 20% with train; paths + sha256 re-verified here, mismatches dropped),
with the CPU scorer of sq339_cpu.py (fp32, GDN via the chunkwise torch kernel, K8 windowing, window 1024).
Effects (paired by seed, log-ratio of aggregate bits/char, in %): A2s vs E27, A2f vs E20, A2f vs E24, mean over the paired seeds, with a
package-clustered bootstrap over the `source` packages (B = 2000, percentile 95% CI; the resampled doc set is shared by both arms and all
seeds, the statistic is the mean over seeds of the paired log ratio) and the seed spread. Also the same effects on the in-corpus clean80
300-doc subset (the SQ339 subset) for orientation. Per-doc NLL is stored.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq379_a2_out_of_corpus.py --part 0   (and --part 1, then --analyze)
"""
import glob
import hashlib
import json
import math
import random
import sys
import time

import torch

import sq339_cpu as C

ARMS = {"A2s": 0.006, "A2f": 0.006, "E27": 0.006, "E24": 0.003, "E20": 0.003}
SEEDS = (2, 3, 4)
OUT = C.RES / "sq379.json"


def find_runs():
    runs = {}
    for mp in sorted(glob.glob(f"{C.FLEET}/*/fusiondance-data/runs/*/meta.json")):
        m = json.load(open(mp))
        e, c = m.get("experiment"), m["config"]
        if e in ARMS and c["train"].get("max_tokens") == 200_000_000 and abs(c["train"]["lr"] - ARMS[e]) < 1e-12 and c["train"]["seed"] in SEEDS:
            run = mp.split("/")[-2]
            if glob.glob(mp.replace("/runs/", "/checkpoints/").replace("meta.json", "model.pt")):
                runs.setdefault((e, c["train"]["seed"]), run)
    return runs


def ooc_docs():
    out = json.load(open(C.RES / "sq_99.json"))["data"]["out_docs"]
    docs, bad = [], 0
    for d in out:
        try:
            t = open(d["path"], "rb").read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
        except (OSError, UnicodeDecodeError):
            bad += 1
            continue
        if hashlib.sha256(t.encode("utf-8")).hexdigest() != d["sha256"]:
            bad += 1
            continue
        docs.append((d["source"], t))
    return docs, bad


def main():
    log = C.make_log(str(C.RES / "sq379.log"))
    runs = find_runs()
    if "--analyze" in sys.argv:
        return analyze(runs, log)
    part = int(sys.argv[sys.argv.index("--part") + 1])
    keys = sorted(runs, key=lambda k: (k[1], k[0]))
    mine = keys[part::2]
    docs, bad = ooc_docs()
    log(f"part {part}: {len(mine)} of {len(keys)} runs; ooc docs {len(docs)} (dropped {bad}); runs {runs}")
    small = C.subset(300, 0)
    res = {"ooc_docs": [{"source": s, "chars": len(t)} for s, t in docs], "scores": {}}
    for k in mine:
        t0 = time.time()
        model = C.load_model(runs[k])
        o = C.score_docs(C.fwd_default(model), [(i, t) for i, (_, t) in enumerate(docs)])
        s = C.score_docs(C.fwd_default(model), small)
        res["scores"][f"{k[0]}_s{k[1]}"] = {"run": runs[k], "ooc": o, "clean80_300": s, "ooc_bpc": C.bpc(o), "clean80_300_bpc": C.bpc(s)}
        log(f"{k} {runs[k]}: ooc {C.bpc(o):.4f} clean80-300 {C.bpc(s):.4f} ({time.time() - t0:.0f}s)")
        json.dump(res, open(C.RES / f"sq379_part{part}.json", "w"))


def boot(nll, chars, groups, pairs, B=2000, seed=0):
    """pairs: list of (arm_nll_key, base_nll_key). Statistic: mean over pairs of 100*(bpc_arm/bpc_base - 1); cluster bootstrap over packages."""
    import numpy as np

    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in names}
    chars = np.asarray(chars, float)

    def stat(sel):
        v = []
        for a, b in pairs:
            ba = np.asarray(nll[a])[sel].sum() / chars[sel].sum()
            bb = np.asarray(nll[b])[sel].sum() / chars[sel].sum()
            v.append(100 * (ba / bb - 1))
        return float(np.mean(v)), v

    point, per = stat(np.arange(len(groups)))
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(names), len(names))
        sel = np.concatenate([idx[names[p]] for p in pick])
        bs.append(stat(sel)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"effect_pct": point, "per_seed_pct": per, "ci95_cluster_boot": [float(lo), float(hi)], "n_pairs": len(pairs), "n_packages": len(names)}


def analyze(runs, log):
    parts = [json.load(open(p)) for p in sorted(glob.glob(str(C.RES / "sq379_part*.json")))]
    docs = parts[0]["ooc_docs"]
    scores = {}
    for p in parts:
        scores.update(p["scores"])
    # E27/E24/E20 seed 2 from SQ263 runs (not in the a2 pulls) are scored by the same parts through find_runs (glob over all fleet dirs)
    groups = [d["source"] for d in docs]
    chars = [d["chars"] for d in docs]
    out = {"check": "SQ379", "git": C.git_commit(), "status": "descriptive, not registered", "n_ooc_docs": len(docs),
           "n_packages": len(set(groups)), "runs": {k: v["run"] for k, v in scores.items()},
           "bpc": {k: {"ooc": v["ooc_bpc"], "clean80_300": v["clean80_300_bpc"]} for k, v in scores.items()}, "effects": {}}
    comps = [("A2s", "E27"), ("A2f", "E20"), ("A2f", "E24"), ("E27", "E20"), ("A2s", "A2f")]
    for a, b in comps:
        seeds = [s for s in SEEDS if f"{a}_s{s}" in scores and f"{b}_s{s}" in scores]
        if not seeds:
            continue
        pairs = [(f"{a}_s{s}", f"{b}_s{s}") for s in seeds]
        nll = {k: v["ooc"]["nll_nats"] for k, v in scores.items()}
        e = boot(nll, chars, groups, pairs)
        e["seeds"] = seeds
        # in-corpus clean80 (300 docs) for orientation: per-seed ratio of aggregate bpc, no clustering
        e["clean80_300_per_seed_pct"] = [100 * (scores[x]["clean80_300_bpc"] / scores[y]["clean80_300_bpc"] - 1) for x, y in pairs]
        out["effects"][f"{a}_vs_{b}"] = e
        log(f"{a} vs {b} seeds {seeds}: ooc {e['effect_pct']:+.2f}% CI {e['ci95_cluster_boot'][0]:+.2f}..{e['ci95_cluster_boot'][1]:+.2f}, per seed "
            f"{[round(x, 2) for x in e['per_seed_pct']]}; clean80-300 {[round(x, 2) for x in e['clean80_300_per_seed_pct']]}")
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
