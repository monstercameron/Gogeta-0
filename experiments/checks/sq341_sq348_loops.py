"""SQ341 (loop-count curve) and SQ348 (per-pass logit lens) on the 200M F2 loop cells, one shared pass. Descriptive, not registered.

Cells 010 (dense attention core x r) and 111 (full fusion: ternary GDN/SWA core x r), best-LR runs, seeds 0-2. For each run
the trunk is run ONCE with r = 12 passes and the hidden state h_k after every pass k = 1..12 is read out two ways:
  with_coda : head(norm_out(coda(h_k)))   == the model evaluated with loops = k  (SQ341; verified against model(x, loops=k))
  no_coda   : head(norm_out(h_k))         the logit lens that bypasses the coda (SQ348)
The models were trained with r ~ U{1..7} (default eval r = 4): r = 8..12 is extrapolation.
Metric: aggregate bits/char (and nats/token) on the fixed 300 clean80 docs (sq339_cpu.subset), fp32 CPU, window 1024.
SQ348 caveat: the head/norm_out were trained on coda outputs, so no_coda numbers are only relative comparisons
(across k, across cells); they are not the model's loss.
If experiments/checks/results/sq339.json exists, the 'none' arm of cells 000/100 (same docs) gives the dense / GDN
reference so the loops penalty can be put on the same footing.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq341_sq348_loops.py
-> experiments/checks/results/sq341.json, sq348.json
"""
import json
import math
import time

import torch

import sq339_cpu as C

R = 12


def trunk_states(model, x, R=R):
    h_list = []
    z = model.embed(x)
    for b in model.prelude:
        z = b(z)
    e = h = z
    for _ in range(R):
        h = h + model.inject(torch.cat([model.inj_norm_h(h), model.inj_norm_e(e)], dim=-1))
        for b in model.core:
            h = b(h)
        h_list.append(h)
    return h_list


def readouts(model, h):
    z = h
    for b in model.coda:
        z = b(z)
    return model.head(model.norm_out(z)), model.head(model.norm_out(h))


def fwd_all(model):
    def f(x):
        out = {}
        for k, h in enumerate(trunk_states(model, x), 1):
            wc, nc = readouts(model, h)
            out[f"with_coda_{k}"], out[f"no_coda_{k}"] = wc, nc
            del wc, nc
        return out
    return f


def verify_equivalence(model, docs):
    """with_coda_k must equal model(x, loops=k) on a few docs (k = 1, 4, 12)."""
    sc = C.score_docs_multi(fwd_all(model), docs)
    err = {}
    for k in (1, 4, 12):
        ref = C.score_docs(C.fwd_default(model, loops=k), docs)
        err[k] = max(abs(a - b) for a, b in zip(sc[f"with_coda_{k}"]["nll_nats"], ref["nll_nats"]))
    return err


def main():
    log = C.make_log(str(C.RES / "sq341_sq348.log"))
    runs = C.best_runs()
    docs = C.subset(300, 0)
    meta = {"git": C.git_commit(), "status": "descriptive, not registered", "n_docs": len(docs),
            "doc_subset": "random.Random(0).sample of the 2333 clean80 v1-val docs (K9 overlap < 0.8), n=300",
            "seeds": [0, 1, 2], "metric": "aggregate bits/char (+ nats/token), fp32 CPU, window 1024",
            "trained_loops": "U{1..7}; default eval r=4; r>7 extrapolates"}
    res = {"cells": {}}
    for cell in ("010", "111"):
        res["cells"][cell] = {"per_seed": {}}
        for seed in (0, 1, 2):
            run = runs[cell][seed]
            model = C.load_model(run)
            if seed == 0:
                eq = verify_equivalence(model, docs[:12])
                log(f"{cell} shared-trunk vs model(loops=k) max |dnll| per doc (nats): {eq}")
                res["cells"][cell]["equivalence_check_max_abs_nll_diff_12docs"] = eq
            t = time.time()
            sc = C.score_docs_multi(fwd_all(model), docs)
            row = {"run": run, "with_coda_bpc": {}, "no_coda_bpc": {}, "with_coda_nats_per_tok": {}, "no_coda_nats_per_tok": {}}
            for k in range(1, R + 1):
                for kind in ("with_coda", "no_coda"):
                    s = sc[f"{kind}_{k}"]
                    row[f"{kind}_bpc"][k] = C.bpc(s)
                    row[f"{kind}_nats_per_tok"][k] = sum(s["nll_nats"]) / sum(s["n_tok"])
            row["tokens"], row["chars"] = sum(s["n_tok"]), sum(s["chars"])
            row["nll_doc_with_coda"] = {k: sc[f"with_coda_{k}"]["nll_nats"] for k in range(1, R + 1)}
            log(f"{cell} s{seed} ({time.time() - t:.0f}s) with_coda bpc: " + " ".join(f"{row['with_coda_bpc'][k]:.4f}" for k in range(1, R + 1)))
            log(f"{cell} s{seed} no_coda   bpc: " + " ".join(f"{row['no_coda_bpc'][k]:.4f}" for k in range(1, R + 1)))
            res["cells"][cell]["per_seed"][str(seed)] = row
            json.dump({"check": "SQ341+SQ348", **meta, **res}, open(C.RES / "sq341_sq348_raw.json", "w"), indent=1)

    # reference cells on the same docs (from SQ339 'none' arm) if available
    ref = {}
    try:
        s339 = json.load(open(C.RES / "sq339.json"))
        for c in ("000", "100"):
            ref[c] = {s: v["none"] for s, v in s339["cells"][c]["per_seed"].items()}
    except Exception as e:
        log(f"no SQ339 reference: {e}")
    summ = {}
    for cell, v in res["cells"].items():
        ps = v["per_seed"]
        summ[cell] = {}
        for kind in ("with_coda", "no_coda"):
            summ[cell][kind] = {}
            for k in range(1, R + 1):
                vals = [ps[s][f"{kind}_bpc"][k] for s in ps]
                d = {"bpc": C.mean_ci(vals), "per_seed_bpc": vals,
                     "log_ratio_vs_r4_pct": C.mean_ci([100 * (ps[s][f"{kind}_bpc"][k] / ps[s][f"{kind}_bpc"][4] - 1) for s in ps])}
                if ref.get("000"):
                    d["vs_dense_000_pct_per_seed"] = [100 * (ps[s][f"{kind}_bpc"][k] / ref["000"][s] - 1) for s in ps]
                summ[cell][kind][k] = d
    out_341 = {"check": "SQ341", **meta, "equivalence": {c: res["cells"][c].get("equivalence_check_max_abs_nll_diff_12docs") for c in res["cells"]},
               "reference_bpc_same_docs": ref, "per_seed_with_coda": {c: {s: v["per_seed"][s]["with_coda_bpc"] for s in v["per_seed"]} for c, v in res["cells"].items()},
               "summary_with_coda": {c: summ[c]["with_coda"] for c in summ}}
    out_348 = {"check": "SQ348", **meta, "note": "head trained on coda outputs: relative comparisons only",
               "per_seed_no_coda": {c: {s: v["per_seed"][s]["no_coda_bpc"] for s in v["per_seed"]} for c, v in res["cells"].items()},
               "per_seed_with_coda": out_341["per_seed_with_coda"],
               "summary_no_coda": {c: summ[c]["no_coda"] for c in summ}, "summary_with_coda": out_341["summary_with_coda"]}
    json.dump(out_341, open(C.RES / "sq341.json", "w"), indent=1)
    json.dump(out_348, open(C.RES / "sq348.json", "w"), indent=1)
    for cell in summ:
        for kind in ("with_coda", "no_coda"):
            log(f"{cell} {kind} mean bpc by k: " + " ".join(f"{summ[cell][kind][k]['bpc']['mean']:.4f}" for k in range(1, R + 1)))


if __name__ == "__main__":
    main()
