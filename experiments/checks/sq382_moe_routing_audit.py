"""SQ382 (descriptive, not registered): per-pass MoE routing audit on the A3m / A3mx checkpoints.

`combine_aux` averages routing statistics over passes and hides pass-specific behaviour. Here, inference only (CPU, fp32): for every checkpoint of the A3m and A3mx
experiments found on the fleet pulls (200M tokens, seeds 5-7; 4 experts, top-2, MoE only in the looped core), the router's top-2 choices are recorded at every call
(pass k, core block j) on 32 fixed sequences of 512 tokens (the clean80 val docs concatenated, token stream cut at 512; same sequences for every model).
Reported per model, averaged over the core blocks unless stated:
  * expert load per pass (share of the N*k assignments) and its entropy / max share;
  * routing agreement between consecutive passes k -> k+1 and between the first and last pass, for the same token and the same block: top-1 identical, and the mean
    overlap |A ∩ B| / k of the two top-2 sets; chance level for top-1 agreement = sum_e f_e(k) f_e(k+1) (independent routing with the observed loads);
  * the same agreement for two DIFFERENT blocks in the same pass (a reference for how much routing is token-driven rather than layer-specific is not available
    because blocks have their own routers: we report the token-level top-1 agreement across blocks too, for orientation).
Interpretation guide (outside review 2026-11-23 13:10, I2): agreement across passes > 0.9 would mean the MoE rescue is unique parameters (C197), not per-pass specialisation.
Descriptive; n = 3 seeds per arm, single fixed 16k-token sample.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq382_moe_routing_audit.py -> results/sq382.json
"""
import glob
import json
import math

import torch

import sq339_cpu as C

N_SEQ, SEQ = 32, 512


def runs():
    out = {}
    for mp in sorted(glob.glob(f"{C.FLEET}/*/fusiondance-data/runs/*/meta.json")):
        m = json.load(open(mp))
        if m.get("experiment") in ("A3m", "A3mx") and m["config"]["train"].get("max_tokens") == 200_000_000:
            run = mp.split("/")[-2]
            if glob.glob(mp.replace("/runs/", "/checkpoints/").replace("meta.json", "model.pt")):
                out.setdefault((m["experiment"], m["config"]["train"]["seed"]), run)
    return out


def sequences():
    tok = C.tokenizer()
    stream = []
    for _, t in C.clean80_docs():
        stream.extend([0] + tok.encode(t).ids)
    x = torch.tensor(stream[:N_SEQ * SEQ]).view(N_SEQ, SEQ)
    return x


def audit(model, x):
    from gogeta.moe import MoEFFN

    mods = [m for m in model.modules() if isinstance(m, MoEFFN)]
    rec = [[] for _ in mods]
    for i, m in enumerate(mods):
        orig = m.route

        def route(x2, orig=orig, i=i):
            out = orig(x2)
            rec[i].append(out[2].detach().clone())   # top_i [N, k]
            return out

        m.route = route
    with torch.no_grad():
        model(x)
    E, k = mods[0].E, mods[0].k
    passes = len(rec[0])
    res = {"n_moe_modules": len(mods), "E": E, "k": k, "passes_recorded_per_module": passes}
    # modules are the core blocks (moe_core_only): recorded calls are one per pass
    load = torch.zeros(len(mods), passes, E)
    for i in range(len(mods)):
        for p in range(passes):
            ti = rec[i][p]
            load[i, p] = torch.bincount(ti.flatten(), minlength=E).float() / ti.numel()
    res["load_by_pass_mean_over_blocks"] = load.mean(0).tolist()
    res["load_entropy_bits_by_pass"] = [float(-(l * (l + 1e-12).log2()).sum()) for l in load.mean(0)]
    res["load_max_share_by_pass"] = [float(l.max()) for l in load.mean(0)]
    res["load_max_share_any_block_pass"] = float(load.max())
    agree1, ov, chance = [], [], []
    for i in range(len(mods)):
        a1, o, c = [], [], []
        for p in range(passes - 1):
            A, B = rec[i][p], rec[i][p + 1]
            a1.append(float((A[:, 0] == B[:, 0]).float().mean()))
            inter = (A[:, :, None] == B[:, None, :]).any(-1).float().sum(-1) / k
            o.append(float(inter.mean()))
            c.append(float((load[i, p] * load[i, p + 1]).sum()))
        agree1.append(a1), ov.append(o), chance.append(c)
    res["top1_agreement_consecutive_by_pass_pair_mean_over_blocks"] = torch.tensor(agree1).mean(0).tolist()
    res["top2_overlap_consecutive_by_pass_pair_mean_over_blocks"] = torch.tensor(ov).mean(0).tolist()
    res["top1_chance_consecutive_by_pass_pair_mean_over_blocks"] = torch.tensor(chance).mean(0).tolist()
    first_last = []
    for i in range(len(mods)):
        A, B = rec[i][0], rec[i][-1]
        first_last.append((float((A[:, 0] == B[:, 0]).float().mean()), float(((A[:, :, None] == B[:, None, :]).any(-1).float().sum(-1) / k).mean())))
    res["first_vs_last_pass_top1_and_top2overlap_by_block"] = first_last
    res["mean_consecutive_top1_agreement"] = float(torch.tensor(agree1).mean())
    res["mean_consecutive_top2_overlap"] = float(torch.tensor(ov).mean())
    res["mean_consecutive_top1_chance"] = float(torch.tensor(chance).mean())
    # across blocks, same pass (token-level, orientation only)
    ab = []
    for p in range(passes):
        for i in range(len(mods) - 1):
            ab.append(float((rec[i][p][:, 0] == rec[i + 1][p][:, 0]).float().mean()))
    res["top1_agreement_adjacent_blocks_same_pass_mean"] = float(torch.tensor(ab).mean())
    return res


def main():
    log = C.make_log(str(C.RES / "sq382.log"))
    rr = runs()
    log(f"runs: {rr}")
    x = sequences()
    res = {"check": "SQ382", "git": C.git_commit(), "status": "descriptive, not registered", "sample": f"{N_SEQ} x {SEQ} tokens of concatenated clean80 val docs", "models": {}}
    for (e, seed), run in sorted(rr.items()):
        model = C.load_model(run)
        r = audit(model, x)
        r["run"] = run
        res["models"][f"{e}_s{seed}"] = r
        log(f"{e} s{seed}: consecutive top1 agreement {r['mean_consecutive_top1_agreement']:.3f} (chance {r['mean_consecutive_top1_chance']:.3f}), top2 overlap "
            f"{r['mean_consecutive_top2_overlap']:.3f}, load max share {r['load_max_share_any_block_pass']:.3f}, first-vs-last mean top1 {sum(a for a, _ in r['first_vs_last_pass_top1_and_top2overlap_by_block']) / len(r['first_vs_last_pass_top1_and_top2overlap_by_block']):.3f}")
        json.dump(res, open(C.RES / "sq382.json", "w"), indent=1)
    for e in ("A3m", "A3mx"):
        ms = [v for k, v in res["models"].items() if k.startswith(e + "_")]
        if ms:
            res.setdefault("summary", {})[e] = {"n": len(ms), **{k: C.mean_ci([m[k] for m in ms]) for k in ("mean_consecutive_top1_agreement", "mean_consecutive_top2_overlap", "mean_consecutive_top1_chance", "top1_agreement_adjacent_blocks_same_pass_mean")}}
    json.dump(res, open(C.RES / "sq382.json", "w"), indent=1)


if __name__ == "__main__":
    main()
