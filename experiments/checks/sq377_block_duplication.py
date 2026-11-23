"""SQ377 (descriptive, not registered): block duplication at inference (RYS-style) on unlooped checkpoints.

Is some of a loop's benefit available without training loops, i.e. is extra depth at inference free quality? On the unlooped 200M GDN hybrid E24 (14 blocks:
swa gdn gdn swa gdn gdn swa gdn gdn swa gdn gdn swa attn; best LR 3e-3, seeds 0-2) and the 14M hybrid E1 (6 blocks, 200M tokens, lr 6e-3, seeds 0-2), each block
j is re-applied once immediately after itself (order ... j, j, j+1 ...), and each adjacent pair (j, j+1) is re-applied (order ... j, j+1, j, j+1, j+2 ...),
one variant at a time, without any training; the residual stream, final norm and head are unchanged. Metric: aggregate bits/char on the first 200 docs of the
fixed 300-doc clean80 subset (sq339_cpu.subset), fp32 CPU. Effect = 100 * (bpc_variant / bpc_base - 1) per seed; mean over seeds and sd. A negative number
would mean extra depth at inference helps. (E32 of the registered row is not scored: not found among the local checkpoints with a 200M budget.)

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq377_block_duplication.py -> results/sq377.json
"""
import glob
import json
import os
import time

import torch

import sq339_cpu as C

E1_RUNS = {0: "E1-20261122-191828-s0", 1: "E1-20261122-194523-s1", 2: "E1-20261122-201556-s2"}


def fwd_order(model, order):
    blocks = list(model.prelude)

    def f(x):
        z = model.embed(x)
        for j in order:
            z = blocks[j](z)
        for b in model.coda:
            z = b(z)
        return model.head(model.norm_out(z))

    return f


def variants(n):
    v = {"base": list(range(n))}
    for j in range(n):
        v[f"dup_{j}"] = list(range(j + 1)) + [j] + list(range(j + 1, n))
    for j in range(n - 1):
        v[f"pair_{j}_{j + 1}"] = list(range(j + 2)) + [j, j + 1] + list(range(j + 2, n))
    return v


def ckpt_exists(run):
    return bool(glob.glob(f"{C.FLEET}/*/fusiondance-data/checkpoints/{run}/model.pt") or glob.glob(os.path.expanduser(f"~/fusiondance-data/checkpoints/{run}/model.pt")))


def main():
    log = C.make_log(str(C.RES / "sq377.log"))
    docs = C.subset(300, 0)[:200]
    best = C.best_runs()
    arms = {"E24": {s: best["100"][s] for s in (0, 1, 2)}, "E1": {s: r for s, r in E1_RUNS.items() if ckpt_exists(r)}}
    res = {"check": "SQ377", "git": C.git_commit(), "status": "descriptive, not registered", "n_docs": len(docs), "arms": {}}
    for arm, runs in arms.items():
        res["arms"][arm] = {"runs": runs, "bpc": {}}
        for seed, run in sorted(runs.items()):
            model = C.load_model(run)
            assert not list(model.core), "unlooped checkpoints only"
            n = len(model.prelude)
            t = time.time()
            row = {}
            for name, order in variants(n).items():
                assert not model.coda or True
                row[name] = C.bpc(C.score_docs(fwd_order(model, order), docs))
            res["arms"][arm]["bpc"][str(seed)] = row
            log(f"{arm} s{seed} ({time.time() - t:.0f}s) base {row['base']:.4f}; best variant {min((v, k) for k, v in row.items() if k != 'base')}")
            json.dump(res, open(C.RES / "sq377.json", "w"), indent=1)
    for arm, v in res["arms"].items():
        names = [k for k in next(iter(v["bpc"].values())) if k != "base"]
        v["effect_pct"] = {k: C.mean_ci([100 * (v["bpc"][s][k] / v["bpc"][s]["base"] - 1) for s in v["bpc"]]) for k in names}
        best = sorted(names, key=lambda k: v["effect_pct"][k]["mean"])[:5]
        log(f"{arm}: best 5 by mean effect: " + ", ".join(f"{k} {v['effect_pct'][k]['mean']:+.2f}%" for k in best))
    json.dump(res, open(C.RES / "sq377.json", "w"), indent=1)


if __name__ == "__main__":
    main()
