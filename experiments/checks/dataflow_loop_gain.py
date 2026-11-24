"""SQ195 on the assembled loop (quick review 2026-11-23 20:32, ML theorist): local gain of each loop pass, in vs out of corpus.

For each pass k the pass map is f_k(h) = core(lam * h + inject([norm h; norm e])) (with lam = 1 for plain injection).
Finite differences at the visited state h_k (fp32, CPU, GDN torch reference): perturb h_k by eps * ||h_k|| along a unit
direction u (per token), and report the RELATIVE gain (||f_k(h_k + d) - f_k(h_k)|| / ||f_k(h_k)||) / (||d|| / ||h_k||), which
is scale-free (an absolute gain mostly measures the norm jump of the injection). eps is swept (1e-3, 3e-2, 1e-1): in
ternary models the int8 activation rounding makes tiny-eps differences jump, so read the gain where it stabilises:
  * mean gain over 3 random directions;
  * top gain by 4 steps of power iteration (u <- normalised response), a lower bound on the largest singular value.
Gain < 1 means the pass contracts perturbations (a fixed-point-like refinement); >= 1 means it does not.
    python3 experiments/checks/dataflow_loop_gain.py [--arms A2s,A2f,E27] [--seeds 2,3,4] [--windows 4]
      -> experiments/checks/results/dataflow_loop_gain.json
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sq339_cpu as C  # noqa: E402
from dataflow_atlas import find_runs, windows  # noqa: E402
from sq379_a2_out_of_corpus import ooc_docs  # noqa: E402

torch.set_grad_enabled(False)
EPS_LIST = (1e-3, 3e-2, 1e-1)  # int8 activation rounding makes tiny-eps finite differences jump; read gain where it stabilises


def pass_map(model, e, i):
    cfg = model.cfg
    lam = torch.sigmoid(model.lam_logit) if cfg.inject_contractive else None

    def f(h):
        inj = model.inject(torch.cat([model.inj_norm_h(h), model.inj_norm_e(e)], dim=-1))
        h = (lam * h if lam is not None else h) + inj
        if cfg.loop_embed:
            h = h + model.loop_emb[min(i, model.loop_emb.shape[0] - 1)]
        for b in model.core:
            h = b(h)
        return h
    return f


def gains(f, h, gen, eps):
    """Relative (scale-free) gains: (||df|| / ||f(h)||) / (||dh|| / ||h||)."""
    base = f(h)
    scale = eps * h.norm(dim=-1, keepdim=True)
    k = (h.norm() / base.norm()).item()

    def resp(u):
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = u * scale
        return (f(h + d) - base), d

    rnd = []
    for _ in range(3):
        r, d = resp(torch.randn(h.shape, generator=gen))
        rnd.append(k * (r.norm() / d.norm()).item())
    u = torch.randn(h.shape, generator=gen)
    top = 0.0
    for _ in range(4):
        r, d = resp(u)
        top = k * (r.norm() / d.norm()).item()
        u = r
    return sum(rnd) / len(rnd), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A2s,A2f,E27")
    ap.add_argument("--seeds", default="2,3,4")
    ap.add_argument("--windows", type=int, default=4)
    a = ap.parse_args()
    runs = find_runs(a.arms.split(","), [int(s) for s in a.seeds.split(",")])
    inc = [t for _, t in C.subset(400, seed=439)]
    ooc, _ = ooc_docs()
    slices = {"in_corpus": windows(inc, a.windows)[0], "out_of_corpus": windows([t for _, t in ooc], a.windows)[0]}
    out = {"check": "SQ195 loop-pass local gain (finite differences)", "eps": list(EPS_LIST), "gain": "relative: (||df||/||f(h)||) / (||dh||/||h||)", "windows": a.windows, "git": C.git_commit(), "runs": {}}
    for key in sorted(runs):
        model = C.load_model(runs[key])
        gen = torch.Generator().manual_seed(195)
        res = {"run": runs[key]}
        for sl, x in slices.items():
            h = model.embed(x)
            for b in model.prelude:
                h = b(h)
            e = h
            rows = []
            for i in range(model.cfg.loops):
                f = pass_map(model, e, i)
                row = {"pass": i + 1}
                for eps in EPS_LIST:
                    g_mean, g_top = gains(f, h, gen, eps)
                    row[f"eps{eps:g}"] = {"gain_mean": g_mean, "gain_top": g_top}
                rows.append(row)
                h = f(h)
            res[sl] = rows
        out["runs"][f"{key[0]}_s{key[1]}"] = res
        print(key, {sl: [(r['pass'],) + tuple(round(r[f'eps{e:g}']['gain_mean'], 3) for e in EPS_LIST) for r in res[sl]] for sl in slices}, flush=True)
        json.dump(out, open(os.path.join(C.RES, "dataflow_loop_gain.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
