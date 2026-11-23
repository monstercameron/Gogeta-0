"""SQ365 (descriptive, not registered): ternary threshold sensitivity on the 200M F2 checkpoints.

Cells 001 (dense, ternary) and 111 (full fusion, ternary) vs their full-precision counterparts 000 and 100, best LR, seeds 0-2.
For every BitLinear layer of the ternary cells (and the same-named make_linear layers of 000/100, as the control: the weights there
were never trained against a quantiser), the latent weight w is perturbed by Gaussian noise with std = 5% of the layer's absmean
scale s = mean|w| (3 independent draws, torch.Generator seeds 1000..1002), then the model is scored: bits/char on the fixed 300-doc
clean80 subset (sq339_cpu.subset), fp32 CPU, window 1024. Ternary cells requantise inside the forward (scale recomputed from the
jittered w), exactly as in training. Penalty = 100 * (bpc_jittered / bpc_base - 1), mean over the 3 draws, per seed.
Also, per cell and seed, from the un-jittered weights: the share of weights within a margin of a rounding threshold,
|(|w|/s) - 0.5| < 0.05 (the zone where 5%-of-scale noise can flip a ternary code), and the share of codes that actually flip under a
draw (round(w/s).clamp(-1,1) vs the jittered, rescaled one), averaged over layers weighted by weight count.
Not registered. Interpretation: a large penalty and a large near-threshold share would say the ternary model sits on rounding
boundaries (fragile to optimiser noise / precision); the control cells show how much of the sensitivity is generic to weight noise.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq365_ternary_jitter.py -> results/sq365.json
"""
import json
import time

import torch
import torch.nn as nn

import sq339_cpu as C
from sq339_ternary_split import quantisable

REL, MARGIN, DRAWS = 0.05, 0.05, 3
CELLS = ("001", "111", "000", "100")


def target_layers(model, ternary):
    from gogeta.model import BitLinear

    out = []
    for name, m in quantisable(model):
        if ternary and not isinstance(m, BitLinear):
            continue
        out.append((name, m))
    return out


def codes(w):
    s = w.abs().mean().clamp(min=1e-5)
    return (w / s).round().clamp(-1, 1), s


def weight_stats(layers):
    n = near = 0
    for _, m in layers:
        w = m.weight.detach()
        s = w.abs().mean().clamp(min=1e-5)
        a = (w.abs() / s)
        near += int(((a - 0.5).abs() < MARGIN).sum())
        n += w.numel()
    return n, near


def flip_share(layers, gen):
    n = flips = 0
    for _, m in layers:
        w = m.weight.detach()
        c0, _ = codes(w)
        c1, _ = codes(w + REL * w.abs().mean() * torch.randn(w.shape, generator=gen))
        flips += int((c0 != c1).sum())
        n += w.numel()
    return flips / n


def main():
    log = C.make_log(str(C.RES / "sq365.log"))
    runs = C.best_runs()
    docs = C.subset(300, 0)
    res = {"check": "SQ365", "git": C.git_commit(), "status": "descriptive, not registered", "n_docs": len(docs),
           "jitter": f"Gaussian, std = {REL} x layer absmean scale", "draws": DRAWS, "margin": MARGIN, "seeds": [0, 1, 2],
           "metric": "aggregate bits/char, fp32 CPU, window 1024", "cells": {}}
    for cell in CELLS:
        ternary = cell in ("001", "111")
        res["cells"][cell] = {"ternary": ternary, "per_seed": {}}
        for seed in (0, 1, 2):
            model = C.load_model(runs[cell][seed])
            layers = target_layers(model, ternary)
            n, near = weight_stats(layers)
            base = C.bpc(C.score_docs(C.fwd_default(model), docs))
            orig = [m.weight.detach().clone() for _, m in layers]
            row = {"run": runs[cell][seed], "n_layers": len(layers), "n_weights": n, "near_threshold_share": near / n,
                   "base_bpc": base, "draws": []}
            for d in range(DRAWS):
                gen = torch.Generator().manual_seed(1000 + d)
                for (_, m), w0 in zip(layers, orig):
                    m.weight.data.copy_(w0 + REL * w0.abs().mean() * torch.randn(w0.shape, generator=gen))
                b = C.bpc(C.score_docs(C.fwd_default(model), docs))
                row["draws"].append({"bpc": b, "penalty_pct": 100 * (b / base - 1)})
                for (_, m), w0 in zip(layers, orig):
                    m.weight.data.copy_(w0)
            row["flip_share_draw0"] = flip_share(layers, torch.Generator().manual_seed(1000))
            row["penalty_pct_mean_over_draws"] = C.mean([x["penalty_pct"] for x in row["draws"]])
            res["cells"][cell]["per_seed"][str(seed)] = row
            log(f"{cell} s{seed}: base {base:.5f}, penalties {[round(x['penalty_pct'], 3) for x in row['draws']]}, "
                f"near-threshold {100 * near / n:.2f}%, flips {100 * row['flip_share_draw0']:.2f}%")
            json.dump(res, open(C.RES / "sq365.json", "w"), indent=1)
    for cell, v in res["cells"].items():
        ps = v["per_seed"]
        v["summary"] = {"penalty_pct": C.mean_ci([ps[s]["penalty_pct_mean_over_draws"] for s in ps]),
                        "near_threshold_share": C.mean_ci([ps[s]["near_threshold_share"] for s in ps]),
                        "flip_share": C.mean_ci([ps[s]["flip_share_draw0"] for s in ps])}
        log(f"{cell}: penalty {v['summary']['penalty_pct']['mean']:+.3f}% (sd {v['summary']['penalty_pct']['sd']:.3f}), near-threshold "
            f"{100 * v['summary']['near_threshold_share']['mean']:.2f}%, flips {100 * v['summary']['flip_share']['mean']:.2f}%")
    json.dump(res, open(C.RES / "sq365.json", "w"), indent=1)


if __name__ == "__main__":
    main()
