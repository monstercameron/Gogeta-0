"""SQ265, CPU version (descriptive, not registered): per-pass gradient interference in the shared loop core.

For each looped checkpoint (A2f and A2s seeds 2-4; comparators E22 (dense loop) and E27 (ternary fusion) seeds 0-2) and each of 6 fixed batches (4 sequences x 512 tokens of
concatenated clean80 val docs), the LM loss is back-propagated r times, once per pass k: in run k the loop-shared parameters are used as-is in pass k and as detached copies in
every other pass, so the resulting gradient G_k is the part of the total gradient that flows into the parameters through pass k's use (the dependence of later passes on pass k's output
stays). Parameter sets: "core" = the parameters of the core blocks; "loop" = core plus the loop-shared inject / inj_norm_h / inj_norm_e / lam_logit / loop_emb parameters. Check: the sum of
the G_k equals the gradient of the ordinary full backward (max relative error reported). Metrics per model and batch: pairwise cosine of (G_i, G_j), ||sum_k G_k|| / sum_k ||G_k||
(1 = aligned, 1/sqrt(r) = orthogonal, 0 = cancelling for r = 2) and, per core block, the cosine of the first and last pass. Averaged over the 6 batches (sd over batches). fp32 CPU,
model in eval mode (no BPTT truncation), GDN via the chunkwise torch kernel (differentiable).
Under AdamW it is the direction of the update that matters (B1 amendment 3), not its size; cosine is scale-free.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq265_pass_gradient_interference.py -> results/sq265.json
"""
import glob
import json
import math

import torch

import sq339_cpu as C
import sq400_ooc_registered as Q

B, T, NB = 4, 512, 6


def batches():
    tok = C.tokenizer()
    stream = []
    for _, t in C.clean80_docs():
        stream.extend([0] + tok.encode(t).ids)
    x = torch.tensor(stream[:NB * B * (T + 1)]).view(NB, B, T + 1)
    return [(b[:, :-1].contiguous(), b[:, 1:].contiguous()) for b in x]


def param_sets(model):
    core = [(m, n, p) for blk in model.core for m in blk.modules() for n, p in list(m._parameters.items()) if p is not None]
    loop = list(core)
    for name in ("inject", "inj_norm_h", "inj_norm_e"):
        mod = getattr(model, name, None)
        if mod is not None:
            loop += [(m, n, p) for m in mod.modules() for n, p in list(m._parameters.items()) if p is not None]
    for name in ("lam_logit", "loop_emb"):
        p = getattr(model, name, None)
        if p is not None:
            loop.append((model, name, p))
    return core, loop


def grads_by_pass(model, x, y, r, entries):
    """Returns [G_k] (flat vectors over `entries`) and the full gradient."""
    shared = [e for e in entries]
    state = {"k": None, "calls": 0}

    def swap(i, k):
        for m, n, p in shared:
            m._parameters[n] = p if (k is None or i == k) else p.detach()

    def pre(mod, args):
        swap(state["calls"], state["k"])
        state["calls"] += 1

    h = model.inj_norm_h.register_forward_pre_hook(pre)
    out = []
    try:
        for k in list(range(r)) + [None]:
            state.update(k=k, calls=0)
            model.zero_grad(set_to_none=True)
            loss = model(x, y, loops=r)[1]
            loss.backward()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for _, _, p in shared])
            out.append(g.clone())
            for m, n, p in shared:
                m._parameters[n] = p
    finally:
        h.remove()
        for m, n, p in shared:
            m._parameters[n] = p
    return out[:-1], out[-1]


def per_block_cos(model, x, y, r):
    """cosine between first-pass and last-pass gradients, per core block."""
    res = []
    for bi in range(len(model.core)):
        ent = [(m, n, p) for m in model.core[bi].modules() for n, p in list(m._parameters.items()) if p is not None]
        gs, _ = grads_by_pass(model, x, y, r, ent)
        res.append(float(torch.nn.functional.cosine_similarity(gs[0], gs[-1], dim=0)))
    return res


def main():
    log = C.make_log(str(C.RES / "sq265.log"))
    Q.SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
    allruns = {}
    import sq379_a2_out_of_corpus as Q9
    Q9.ARMS = {"A2s": 0.006, "A2f": 0.006, "E22": 0.006, "E27": 0.006}
    Q9.SEEDS = (0, 1, 2, 3, 4)
    allruns = Q9.find_runs()
    want = [(a, s) for a in ("A2f", "A2s") for s in (2, 3, 4)] + [(a, s) for a in ("E22", "E27") for s in (0, 1, 2)]
    bs = batches()
    res = {"check": "SQ265", "git": C.git_commit(), "status": "descriptive, not registered", "batches": f"{NB} x ({B} x {T}) tokens", "models": {}}
    for k in want:
        if k not in allruns:
            log(f"missing {k}")
            continue
        model = C.load_model(allruns[k]).eval()
        r = model.cfg.loops
        core, loop = param_sets(model)
        row = {"run": allruns[k], "loops": r, "n_params_core": sum(p.numel() for _, _, p in core), "n_params_loop": sum(p.numel() for _, _, p in loop), "sets": {}}
        for sname, ent in (("core", core), ("loop", loop)):
            cos, ratio, err = [], [], []
            for x, y in bs:
                gs, full = grads_by_pass(model, x, y, r, ent)
                tot = sum(gs)
                err.append(float((tot - full).norm() / full.norm()))
                ratio.append(float(tot.norm() / sum(g.norm() for g in gs)))
                cos.append({f"{i}-{j}": float(torch.nn.functional.cosine_similarity(gs[i], gs[j], dim=0)) for i in range(r) for j in range(i + 1, r)})
            row["sets"][sname] = {"cos_by_pair_mean": {p: sum(c[p] for c in cos) / len(cos) for p in cos[0]}, "cos_by_pair_sd": {p: float(torch.tensor([c[p] for c in cos]).std()) for p in cos[0]},
                                  "norm_ratio_mean": sum(ratio) / len(ratio), "norm_ratio_sd": float(torch.tensor(ratio).std()), "sum_check_max_rel_err": max(err),
                                  "per_batch_cos_first_last": [c[f"0-{r - 1}"] for c in cos]}
        row["per_block_cos_first_last_batch0"] = per_block_cos(model, *bs[0], r)
        res["models"][f"{k[0]}_s{k[1]}"] = row
        s = row["sets"]["core"]
        log(f"{k} loops {r}: core cos(first,last) {s['cos_by_pair_mean'][f'0-{r - 1}']:+.3f} norm ratio {s['norm_ratio_mean']:.3f} check err {s['sum_check_max_rel_err']:.1e}; loop-set cos "
            f"{row['sets']['loop']['cos_by_pair_mean'][f'0-{r - 1}']:+.3f}; per-block {[round(c, 2) for c in row['per_block_cos_first_last_batch0']]}")
        json.dump(res, open(C.RES / "sq265.json", "w"), indent=1)


if __name__ == "__main__":
    main()
