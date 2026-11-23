"""SQ339 (descriptive, not registered): split the ternary cost post hoc by eval-time fake quantisation.

The 200M F2 cells 000 (dense attention) and 100 (GDN hybrid), trained in full precision, are evaluated with the
BitLinear operations of gogeta/model.py applied at eval time to every `make_linear` layer (attention qkv/o, SwiGLU
up/down, GDN q/k/v/gate/o, re-injection; NOT embeddings/head, norms, GDN a/b, short convs):
  none | weight-only (absmean ternary) | activation-only (per-token int8 absmax) | both.
Arms: 4 x cells {000, 100} x seeds 0-2 at the best LR (sq263.json). Metric: aggregate bits/char over a fixed subset of
300 clean80 docs (random.Random(0) over the 2,333), fp32 CPU, GDN via the chunkwise torch kernel (validated in
sq339_cpu_validate.py). Penalty = 100 * (bpc_arm / bpc_none - 1), i.e. log-scale ratio reported as %.
Reference: the TRAINED ternary cost, cell 001 vs 000 at their best LRs (sq263.json vs_dense_at_best_lr).
Note: no training adapts to the quantiser here, so post-training penalties overstate what QAT would see; this is an
upper-bound style decomposition, and the weight/activation split need not add (reported with the 'both' arm).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq339_ternary_split.py [--summarize]
"""
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sq339_cpu as C

MODE = {"w": False, "a": False}


def quantisable(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not name.endswith((".a", ".b")) and name != "head":
            yield name, m


def install(model):
    n = 0
    for _, m in quantisable(model):
        def fwd(x, m=m):
            w = m.weight
            if MODE["w"]:
                scale = w.abs().mean().clamp(min=1e-5)
                w = (w / scale).round().clamp(-1, 1) * scale
            if MODE["a"]:
                s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                x = (x * s).round().clamp(-128, 127) / s
            return F.linear(x, w, m.bias)

        m.forward = fwd
        n += 1
    return n


ARMS = {"none": (False, False), "weight_ternary": (True, False), "act_int8": (False, True), "both": (True, True)}


def main():
    log = C.make_log(str(C.RES / "sq339.log"))
    import sys
    if "--summarize" in sys.argv:   # recompute the summary from the stored per-seed rows
        return summarize(json.load(open(C.RES / "sq339.json")), log)
    runs = C.best_runs()
    docs = C.subset(300, 0)
    log(f"docs {len(docs)}; git {C.git_commit()}")
    res = {"check": "SQ339", "git": C.git_commit(), "status": "descriptive, not registered", "n_docs": len(docs),
           "doc_subset": "random.Random(0).sample of the 2333 clean80 v1-val docs (K9 overlap < 0.8), n=300", "seeds": [0, 1, 2],
           "metric": "aggregate bits/char, fp32 CPU, window 1024", "cells": {}}
    for cell in ("000", "100"):
        res["cells"][cell] = {"per_seed": {}}
        for seed in (0, 1, 2):
            run = runs[cell][seed]
            model = C.load_model(run)
            n_lin = install(model)
            row = {"run": run, "n_quantised_linears": n_lin}
            for arm, (w, a) in ARMS.items():
                MODE["w"], MODE["a"] = w, a
                t = time.time()
                sc = C.score_docs(C.fwd_default(model), docs)
                row[arm] = C.bpc(sc)
                row.setdefault("nll_doc", {})[arm] = sc["nll_nats"]
                log(f"{cell} s{seed} {arm}: bpc {row[arm]:.5f} ({time.time() - t:.0f}s)")
            row["tokens"], row["chars"] = sum(sc["n_tok"]), sum(sc["chars"])
            for arm in ARMS:
                row[f"pen_pct_{arm}"] = 100 * (row[arm] / row["none"] - 1)
            res["cells"][cell]["per_seed"][str(seed)] = row
            json.dump(res, open(C.RES / "sq339.json", "w"), indent=1)
    summarize(res, log)


def summarize(res, log):
    for cell, v in res["cells"].items():
        ps = v["per_seed"]
        v["penalty_pct"] = {arm: C.mean_ci([ps[s][f"pen_pct_{arm}"] for s in ps]) for arm in ARMS if arm != "none"}
    sq = json.load(open(C.RES / "sq263.json"))
    res["trained_ternary_reference_pct"] = 100 * sq["vs_dense_at_best_lr"]["001"]
    res["trained_ternary_reference_note"] = "cell 001 vs 000, best LRs, 3-seed means, full clean80 (sq263.json)"
    json.dump(res, open(C.RES / "sq339.json", "w"), indent=1)
    for cell, v in res["cells"].items():
        for arm, d in v["penalty_pct"].items():
            log(f"{cell} {arm}: mean {d['mean']:+.2f}% sd {d['sd']:.2f} ci95 [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
