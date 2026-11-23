"""SQ340: fp32 vs bf16-emulated scoring of the 200M F2 cells (registered in experiments/checks/PREDICTIONS.md, section SQ340).

Question: does bf16 rounding in the GDN mixer inflate (or create) the 200M interaction I? The CPU path here is fp32 everywhere
(GDN via the chunkwise torch kernel, validated against the fp32 loop reference). The GPU path (gogeta/model.py
GatedDeltaNet.forward) casts q, k, v and beta to bf16 before the kernel and the kernel returns o in bf16. Arms, cells 000, 100, 111
(and the GDN-free cells 001, 010 in fp32 only, needed for I; they have no GDN mixer, so the emulation does not touch them):
  (a) fp32 everywhere.
  (b) registered arm: q, k, v and the mixer output o rounded to bf16 (round-trip through torch.bfloat16) at the GPU cast points.
  (c) (b) plus beta rounded to bf16 (the GPU path also casts beta); reported as a sensitivity arm.
Emulation caveats: only the GDN mixer inputs/outputs are rounded; the GPU run also uses bf16 autocast for every matmul, and fla's
kernel accumulates in its own precision internally; this emulation does not reproduce those.
Metric: aggregate bits/char on the fixed 300 clean80 docs (sq339_cpu.subset), window 1024, log scale. shift = 100*(bpc_b/bpc_a - 1)
per cell and seed. I = y111 - y100 - y010 - y001 + 2*y000 (log bpc) with the seed-blocked estimator of experiments/checks/sq263.py
(y = cell + seed effects, weights W5), seeds 0-2 shared.
Registered prediction: no cell's shift exceeds 1 point and |shift(111) - shift(100)| < 1 point; falsifier: either >= 1 point.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq340_bf16_emulation.py -> results/sq340.json
"""
import json
import math
import time

import numpy as np
import torch

import sq339_cpu as C

W = {"000": 2, "001": -1, "010": -1, "100": -1, "111": 1}   # factorial_analysis.W5 (E20..E27), as sq344.py
rnd = lambda t: t.to(torch.bfloat16).float()                  # noqa: E731


def set_mode(model, mode):
    from gogeta.model import GatedDeltaNet

    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            def kern(q, k, v, g, beta, scale=None, mode=mode, **kw):
                if mode != "fp32":
                    q, k, v = rnd(q), rnd(k), rnd(v)
                if mode == "bf16_qkv_beta_out":
                    beta = rnd(beta)
                o, S = C.gdn_chunk_cpu(q, k, v, g, beta, scale)
                return (rnd(o) if mode != "fp32" else o), S

            m._kernel = kern


def seed_blocked_I(y):
    """y: {cell: {seed: log bpc}} -> (I, ci95, df) with the seed-blocked estimator of sq263.py."""
    rows = [(c, s, v) for c in W for s, v in y[c].items()]
    cells = sorted({r[0] for r in rows}); seeds = sorted({r[1] for r in rows})
    X = np.zeros((len(rows), len(cells) + len(seeds) - 1)); yy = np.array([r[2] for r in rows])
    for i, (c, s, _) in enumerate(rows):
        X[i, cells.index(c)] = 1
        if seeds.index(s) > 0:
            X[i, len(cells) + seeds.index(s) - 1] = 1
    XtX = np.linalg.pinv(X.T @ X); beta = XtX @ X.T @ yy; res = yy - X @ beta
    dfb = len(rows) - np.linalg.matrix_rank(X); s2 = float(res @ res / dfb)
    w = np.zeros(X.shape[1])
    for c, k in W.items():
        w[cells.index(c)] = k
    I = float(w @ beta); se = math.sqrt(s2 * float(w @ XtX @ w))
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}[int(dfb)]
    return I, [I - t * se, I + t * se], int(dfb)


def main():
    log = C.make_log(str(C.RES / "sq340.log"))
    runs = C.best_runs()
    docs = C.subset(300, 0)
    res = {"check": "SQ340", "git": C.git_commit(), "status": "registered prediction in PREDICTIONS.md (section SQ340)",
           "n_docs": len(docs), "seeds": [0, 1, 2], "metric": "aggregate bits/char, fp32 CPU emulation, window 1024", "bpc": {}}
    for cell in ("000", "001", "010", "100", "111"):
        modes = ("fp32", "bf16_qkv_out", "bf16_qkv_beta_out") if cell in ("100", "111") else ("fp32",)
        res["bpc"][cell] = {}
        for seed in (0, 1, 2):
            model = C.load_model(runs[cell][seed])
            row = {}
            for mode in modes:
                set_mode(model, mode)
                t = time.time()
                row[mode] = C.bpc(C.score_docs(C.fwd_default(model), docs))
                log(f"{cell} s{seed} {mode}: {row[mode]:.5f} ({time.time() - t:.0f}s)")
            res["bpc"][cell][str(seed)] = row
            json.dump(res, open(C.RES / "sq340.json", "w"), indent=1)
    # shifts
    shifts = {}
    for cell in ("100", "111"):
        shifts[cell] = {mode: {s: 100 * (res["bpc"][cell][s][mode] / res["bpc"][cell][s]["fp32"] - 1) for s in res["bpc"][cell]}
                        for mode in ("bf16_qkv_out", "bf16_qkv_beta_out")}
    res["shift_pct_per_seed"] = shifts
    res["shift_pct_summary"] = {c: {m: C.mean_ci(list(v.values())) for m, v in d.items()} for c, d in shifts.items()}
    # dense (000) has no GDN: shift is 0 by construction
    res["shift_diff_111_minus_100_pct"] = {m: C.mean_ci([shifts["111"][m][s] - shifts["100"][m][s] for s in shifts["100"][m]])
                                           for m in ("bf16_qkv_out", "bf16_qkv_beta_out")}
    Is = {}
    for label, mode in (("fp32", "fp32"), ("bf16_qkv_out", "bf16_qkv_out"), ("bf16_qkv_beta_out", "bf16_qkv_beta_out")):
        y = {c: {int(s): math.log(res["bpc"][c][s][mode if (mode in res["bpc"][c][s]) else "fp32"]) for s in res["bpc"][c]} for c in W}
        I, ci, df = seed_blocked_I(y)
        per_seed = [sum(W[c] * y[c][s] for c in W) for s in (0, 1, 2)]
        Is[label] = {"I_log_points": 100 * I, "ci95_log_points": [100 * ci[0], 100 * ci[1]], "df": df, "I_per_seed_points": [100 * p for p in per_seed]}
        log(f"I [{label}]: {100 * I:+.2f} points, 95% CI [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] (df {df})")
    res["I_seed_blocked"] = Is
    res["I_shift_bf16_minus_fp32_points"] = {m: Is[m]["I_log_points"] - Is["fp32"]["I_log_points"] for m in ("bf16_qkv_out", "bf16_qkv_beta_out")}
    mx = max(abs(d["mean"]) for c in res["shift_pct_summary"].values() for d in c.values())
    res["prediction_check"] = {"max_abs_mean_shift_points": mx,
                               "cells_all_below_1_point": mx < 1.0,
                               "abs_shift_diff_111_minus_100_below_1_point": {m: abs(d["mean"]) < 1.0 for m, d in res["shift_diff_111_minus_100_pct"].items()}}
    json.dump(res, open(C.RES / "sq340.json", "w"), indent=1)
    log(json.dumps(res["prediction_check"]))


if __name__ == "__main__":
    main()
