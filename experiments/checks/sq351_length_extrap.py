"""SQ351 (descriptive, not registered): length extrapolation of the 200M F2 cells 000 (dense attention, RoPE) and 100 (GDN hybrid).

Models were trained at sequence length 1,024. Here docs with MORE than 1,200 tokens are scored in ONE window
of up to 2,048 tokens (first 2,048 tokens of the doc, BOS prepended): the RoPE cos/sin tables are rebuilt for 2,048 positions at
eval time by overwriting the buffers of the model's shared `Rotary` module (no change to gogeta/). Per-position NLL (nats/token)
is averaged in bins of the target position: (0,512], (512,1024], (1024,2048]; the same docs for every cell and seed, so the bins
are comparable (a doc enters the last bin only for its positions > 1024).
Sanity control: bins 1 and 2 at window 2048 must equal the window-1024 numbers (causal models, same RoPE angles): max per-token |diff| is reported.
Cells 000 / 100, best LR, seeds 0-2, fp32 CPU, GDN via the chunkwise torch kernel.
Doc sets: A = clean80 val docs > 1,200 tokens as specified (only 3 exist: val docs have median 221 tokens, max 1,674); B = the SQ99
out-of-corpus local library files (< 20% 13-gram overlap with train; sha256-verified against sq_99.json) with > 1,200 tokens: many
more long docs but a distribution shift, so read the SHAPE over position (growth of bin 3 over bin 2, GDN vs dense by bin).
Descriptive: n = 3 seeds per cell; the docs are one fixed set (item CIs over docs are not seed CIs).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq351_length_extrap.py -> results/sq351.json
"""
import json
import math
import time

import torch

import sq339_cpu as C

WIN = 2048
BINS = [(0, 512), (512, 1024), (1024, 2048)]   # target position in (lo, hi]; position i = index in ids of the target token (1 = first doc token)


def rebuild_rope(model, n):
    from gogeta.model import Rotary

    r = Rotary(model.cfg.head_dim, n, model.cfg.rope_base)
    model.rope.cos, model.rope.sin = r.cos.clone(), r.sin.clone()
    assert model.rope.cos.shape[2] == n


@torch.no_grad()
def per_position_nll(model, docs_ids, win, budget=8192):
    """docs_ids: list of id lists (BOS first), each truncated to win. Returns list of per-target-position NLL arrays (len L-1)."""
    order = sorted(range(len(docs_ids)), key=lambda i: len(docs_ids[i]))
    out = [None] * len(docs_ids)
    i = 0
    while i < len(order):
        L0 = len(docs_ids[order[i]])
        j = i
        while j < len(order) and (j - i + 1) * len(docs_ids[order[j]]) <= max(budget, L0):
            j += 1
        idx = order[i:j]
        T = max(len(docs_ids[k]) for k in idx)
        x = torch.zeros(len(idx), T, dtype=torch.long)
        for b, k in enumerate(idx):
            x[b, :len(docs_ids[k])] = torch.tensor(docs_ids[k])
        lp = torch.log_softmax(model(x)[0][:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
        for b, k in enumerate(idx):
            out[k] = nll[b, :len(docs_ids[k]) - 1].double()
        i = j
    return out


def bin_means(pp):
    res = []
    for lo, hi in BINS:
        tot, n = 0.0, 0
        for a in pp:                      # a[t-1] = nll of target at position t
            seg = a[lo:min(hi, len(a))]   # targets lo+1..hi
            tot += float(seg.sum())
            n += len(seg)
        res.append({"bin": f"{lo}-{hi}", "n_tokens": n, "nats_per_token": tot / n if n else None})
    return res


def load_sets(tok, bos):
    """Doc sets (id lists, BOS first, cut to WIN tokens). A: clean80 val docs > 1200 tokens (the spec; only a handful exist,
    the v1/v2 val docs have median 221 tokens, max 1674). B: the SQ99 out-of-corpus local files (K9 scanner: < 20% 13-gram
    overlap with train; paths and sha256 in sq_99.json, re-verified here) with > 1200 tokens: enough long docs, but a
    distribution shift (raw library code), so absolute levels are high and the READ is the shape over position."""
    import hashlib
    import os

    sets = {}
    docs = C.clean80_docs()
    ids_all = [[bos] + tok.encode(t).ids for _, t in docs]
    sel = [i for i, ids in enumerate(ids_all) if len(ids) - 1 > 1200]
    sets["A_clean80_gt1200"] = {"ids": [ids_all[i][:WIN] for i in sel], "desc": [docs[i][0] for i in sel]}
    out = json.load(open(C.RES / "sq_99.json"))["data"]["out_docs"]
    ids_b, desc_b, bad = [], [], 0
    for d in out:
        try:
            t = open(d["path"], "rb").read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
        except (OSError, UnicodeDecodeError):
            bad += 1
            continue
        if hashlib.sha256(t.encode("utf-8")).hexdigest() != d["sha256"]:
            bad += 1
            continue
        ids = [bos] + tok.encode(t).ids
        if len(ids) - 1 > 1200:
            ids_b.append(ids[:WIN])
            desc_b.append(d["path"])
    sets["B_out_of_corpus_gt1200"] = {"ids": ids_b, "desc": desc_b, "sha_mismatch_or_missing": bad}
    return sets


def main():
    log = C.make_log(str(C.RES / "sq351.log"))
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    sets = load_sets(tok, bos)
    for k, v in sets.items():
        log(f"{k}: {len(v['ids'])} docs, scored tokens {sum(len(d) - 1 for d in v['ids'])}, tokens beyond 1024: {sum(max(0, len(d) - 1025) for d in v['ids'])}")
    runs = C.best_runs()
    res = {"check": "SQ351", "git": C.git_commit(), "status": "descriptive, not registered", "window": WIN,
           "bins": [f"{a}-{b}" for a, b in BINS], "metric": "mean NLL nats/token by target-position bin, fp32 CPU",
           "sets": {k: {"n_docs": len(v["ids"]), "n_tokens": sum(len(d) - 1 for d in v["ids"]), "docs": v["desc"],
                        **({"sha_mismatch_or_missing": v["sha_mismatch_or_missing"]} if "sha_mismatch_or_missing" in v else {})}
                    for k, v in sets.items()}, "cells": {}}
    for cell in ("000", "100"):
        res["cells"][cell] = {"per_seed": {}}
        for seed in (0, 1, 2):
            run = runs[cell][seed]
            model = C.load_model(run)
            t = time.time()
            rebuild_rope(model, WIN)
            row = {"run": run}
            for name, st in sets.items():
                pp = per_position_nll(model, st["ids"], WIN)
                # control: the same first-1024-token prefixes scored in a 1024 window must match (causal)
                pp1 = per_position_nll(model, [d[:1024] for d in st["ids"]], 1024)
                ctrl = max(float((a[:1023] - b[:1023]).abs().max()) for a, b in zip(pp, pp1))
                row[name] = {"bins_window2048": bin_means(pp), "bins_window1024_control": bin_means(pp1)[:2],
                             "control_max_abs_token_nll_diff_first1023": ctrl}
                log(f"{cell} s{seed} {name} ({time.time() - t:.0f}s) 2048-bins: " + " ".join(f"{b['nats_per_token']:.4f}" for b in row[name]["bins_window2048"])
                    + " | 1024-ctrl: " + " ".join(f"{b['nats_per_token']:.4f}" for b in row[name]["bins_window1024_control"]) + f" | ctrl diff {ctrl:.2e}")
            res["cells"][cell]["per_seed"][str(seed)] = row
            json.dump(res, open(C.RES / "sq351.json", "w"), indent=1)
    res["summary"] = {}
    for name in sets:
        p0, p1 = res["cells"]["000"]["per_seed"], res["cells"]["100"]["per_seed"]
        g = lambda p, s, i: p[s][name]["bins_window2048"][i]["nats_per_token"]   # noqa: E731
        res["summary"][name] = {
            "nats_per_token_by_bin": {c: {b: C.mean_ci([g(res["cells"][c]["per_seed"], s, i) for s in p0]) for i, b in enumerate(res["bins"])}
                                      for c in res["cells"]},
            "gdn_vs_dense_pct_by_bin": {b: C.mean_ci([100 * (math.exp(g(p1, s, i) - g(p0, s, i)) - 1) for s in p0]) for i, b in enumerate(res["bins"])},
            "growth_bin3_minus_bin2_nats": {c: C.mean_ci([g(res["cells"][c]["per_seed"], s, 2) - g(res["cells"][c]["per_seed"], s, 1) for s in p0]) for c in res["cells"]},
            "growth_bin3_over_bin2_pct": {c: C.mean_ci([100 * (math.exp(g(res["cells"][c]["per_seed"], s, 2) - g(res["cells"][c]["per_seed"], s, 1)) - 1) for s in p0]) for c in res["cells"]}}
        log(f"{name}: gdn vs dense % by bin " + json.dumps({b: round(d['mean'], 2) for b, d in res['summary'][name]['gdn_vs_dense_pct_by_bin'].items()})
            + " | growth bin3 vs bin2 % " + json.dumps({c: round(d['mean'], 2) for c, d in res['summary'][name]['growth_bin3_over_bin2_pct'].items()}))
    json.dump(res, open(C.RES / "sq351.json", "w"), indent=1)


if __name__ == "__main__":
    main()
