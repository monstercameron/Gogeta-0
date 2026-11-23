"""SQ353 (descriptive, not registered): loop-state perturbation gain, pass-to-pass dynamics and CKA on the 200M looped models.

Is the full fusion's penalty a recurrence-stability problem (ternary noise amplified across passes)?  Inference only, fp32 CPU, 8 fixed sequences of 256 tokens
(concatenated clean80 val docs), default loops, every 200M-token checkpoint found on the fleet pulls of: E22 (010), E27 (111), A1 arms, A2s (ternary assembled), A2f
(full-precision assembled), A3x/A3y (ternary variants), plus E24 (unlooped GDN hybrid) for the CKA comparison.
(1) Perturbation gain: before pass k's first core block the state h gets a random perturbation with norm eps x ||h|| per token (eps = 1e-3 and 1e-2, fixed seed);
    gain_k = mean over tokens of ||h_coda_in(perturbed) - h_coda_in(clean)|| / ||perturbation||, the change measured at the coda input (the loop's final state);
    gain_rel_k = gain_k x mean||h at pass k|| / mean||coda input||, i.e. relative output change per relative perturbation (the scale-free number: ~ 1 neutral,
    < 1 contractive, > 1 amplifying); the absolute gain also reflects the state norm growing along the loop. One value per k = 1..r. Finite differences are checked against the second eps.
(2) Pass-to-pass dynamics: after each pass k, the state h_k: cosine(h_k, h_{k+1}) and relative step ||h_{k+1} - h_k|| / ||h_k||, mean over tokens.
(3) Linear CKA (centred) between the pass outputs h_1..h_r for the looped models, and between the block outputs of E24 (all 14, and the 4 depth-matched ones at
    blocks 3, 6, 10, 13, 1-based, spaced like the passes of a 4-pass model); summarised as mean adjacent-pass CKA and CKA(first, last).
Per-seed values and group means are in the JSON; no inferential claims (n = 2-6 per arm).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq353_loop_stability.py -> results/sq353.json
"""
import glob
import json
import math
import re

import torch

import sq339_cpu as C

N_SEQ, SEQ = 8, 256
EXPS = ("E22", "E27", "A1a", "A1b", "A1e", "A1f", "A1i", "A1z", "A2s", "A2f", "A3x", "A3y", "E24")


def find_runs():
    out = {}
    for mp in sorted(glob.glob(f"{C.FLEET}/*/fusiondance-data/runs/*/meta.json")):
        m = json.load(open(mp))
        c = m["config"]
        e = m.get("experiment")
        if e in EXPS and c["train"].get("max_tokens") == 200_000_000 and glob.glob(mp.replace("/runs/", "/checkpoints/").replace("meta.json", "model.pt")):
            out.setdefault((e, c["train"]["seed"], c["train"]["lr"]), mp.split("/")[-2])
    return out


def sequences():
    tok = C.tokenizer()
    stream = []
    for _, t in C.clean80_docs():
        stream.extend([0] + tok.encode(t).ids)
    return torch.tensor(stream[:N_SEQ * SEQ]).view(N_SEQ, SEQ)


def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    num = (X.T @ Y).pow(2).sum()
    return float(num / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


def run_looped(model, x):
    core = model.core
    states, coda_in, state_in = [], {}, {}
    pert = {"k": None, "eps": 0.0, "gen": None, "calls": 0}

    def pre_core(mod, args):
        c = pert["calls"]
        pert["calls"] += 1
        if c == 0:
            state_in["h0"] = args[0].detach().clone()
        if pert["k"] is not None and c == pert["k"]:
            h = args[0]
            d = torch.randn(h.shape, generator=pert["gen"])
            d = d / d.norm(dim=-1, keepdim=True) * h.norm(dim=-1, keepdim=True) * pert["eps"]
            pert["delta_norm"] = d.norm(dim=-1)
            pert["h_norm"] = float(h.norm(dim=-1).mean())
            return (h + d,) + tuple(args[1:])

    h1 = core[0].register_forward_pre_hook(pre_core)
    h2 = core[-1].register_forward_hook(lambda m, a, o: states.append(o.detach().clone()))
    h3 = model.coda[0].register_forward_pre_hook(lambda m, a: coda_in.__setitem__("h", a[0].detach().clone()))

    def go(k=None, eps=0.0):
        pert.update(k=k, eps=eps, gen=torch.Generator().manual_seed(123), calls=0)
        states.clear()
        with torch.no_grad():
            model(x)
        return [s.clone() for s in states], coda_in["h"].clone(), pert.get("delta_norm")

    clean_states, clean_coda, _ = go()
    r = len(clean_states)
    out = {"loops": r}
    # (2) dynamics
    flat = [s.reshape(-1, s.shape[-1]) for s in clean_states]
    out["cos_consecutive"] = [float(torch.nn.functional.cosine_similarity(flat[i], flat[i + 1], dim=-1).mean()) for i in range(r - 1)]
    out["rel_step_consecutive"] = [float(((flat[i + 1] - flat[i]).norm(dim=-1) / flat[i].norm(dim=-1)).mean()) for i in range(r - 1)]
    h0 = state_in["h0"].reshape(-1, flat[0].shape[-1])
    out["rel_step_first"] = float(((flat[0] - h0).norm(dim=-1) / h0.norm(dim=-1)).mean())
    out["state_norm_by_pass"] = [float(f.norm(dim=-1).mean()) for f in flat]
    # (3) CKA among pass outputs
    out["cka_pass_matrix"] = [[cka(flat[i], flat[j]) for j in range(r)] for i in range(r)]
    # (1) perturbation gain
    for eps in (1e-3, 1e-2):
        gains, rel = [], []
        cn = float(clean_coda.norm(dim=-1).mean())
        for k in range(r):
            _, coda_p, dn = go(k, eps)
            diff = (coda_p - clean_coda).norm(dim=-1)
            g = float((diff / dn).mean())
            gains.append(g)
            rel.append(g * pert["h_norm"] / cn)    # relative change of the final state per relative perturbation of the state at pass k
        out[f"gain_eps{eps:g}"] = gains
        out[f"gain_rel_eps{eps:g}"] = rel
    h1.remove(), h2.remove(), h3.remove()
    return out


def run_unlooped_cka(model, x):
    outs = []
    hs = [b.register_forward_hook(lambda m, a, o: outs.append(o.detach().reshape(-1, o.shape[-1]).clone())) for b in list(model.prelude)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    n = len(outs)
    full = [[cka(outs[i], outs[j]) for j in range(n)] for i in range(n)]
    sel = [2, 5, 9, 12]
    return {"n_blocks": n, "cka_block_matrix": full, "depth_matched_blocks_1based": [s + 1 for s in sel],
            "cka_depth_matched": [[full[i][j] for j in sel] for i in sel]}


def adj(m):
    n = len(m)
    return sum(m[i][i + 1] for i in range(n - 1)) / (n - 1), m[0][n - 1]


def main():
    log = C.make_log(str(C.RES / "sq353.log"))
    runs = find_runs()
    x = sequences()
    res = {"check": "SQ353", "git": C.git_commit(), "status": "descriptive, not registered", "sample": f"{N_SEQ} x {SEQ} tokens of concatenated clean80 val docs",
           "models": {}}
    for (e, seed, lr), run in sorted(runs.items()):
        try:
            model = C.load_model(run)
            if e == "E24":
                r = run_unlooped_cka(model, x)
                a, fl = adj(r["cka_depth_matched"])
                log(f"{e} s{seed}: depth-matched CKA adjacent {a:.3f}, first-last {fl:.3f}")
            else:
                r = run_looped(model, x)
                a, fl = adj(r["cka_pass_matrix"])
                log(f"{e} s{seed} lr {lr}: loops {r['loops']} gain(1e-3) {[round(g, 2) for g in r['gain_eps0.001']]} cos {[round(c, 3) for c in r['cos_consecutive']]} CKA adj {a:.3f} first-last {fl:.3f}")
        except Exception as ex:  # keep going
            r = {"error": f"{type(ex).__name__}: {ex}"}
            log(f"{e} s{seed}: ERROR {r['error']}")
        r["run"], r["lr"] = run, lr
        res["models"][f"{e}_s{seed}_lr{lr:g}"] = r
        json.dump(res, open(C.RES / "sq353.json", "w"), indent=1)
    # group summaries
    groups = {}
    for k, v in res["models"].items():
        if "error" in v:
            continue
        groups.setdefault(re.sub(r"_s\d+", "", k), []).append(v)
    res["groups"] = {}
    for g, ms in groups.items():
        d = {"n": len(ms)}
        if "gain_eps0.001" in ms[0]:
            r = min(m["loops"] for m in ms)
            d["gain_eps0.001_by_k_mean"] = [sum(m["gain_eps0.001"][i] for m in ms) / len(ms) for i in range(r)]
            d["gain_rel_eps0.001_by_k_mean"] = [sum(m["gain_rel_eps0.001"][i] for m in ms) / len(ms) for i in range(r)]
            d["gain_eps0.01_by_k_mean"] = [sum(m["gain_eps0.01"][i] for m in ms) / len(ms) for i in range(r)]
            d["cos_consecutive_mean"] = [sum(m["cos_consecutive"][i] for m in ms) / len(ms) for i in range(r - 1)]
            d["rel_step_consecutive_mean"] = [sum(m["rel_step_consecutive"][i] for m in ms) / len(ms) for i in range(r - 1)]
            d["cka_adjacent_mean"] = sum(adj(m["cka_pass_matrix"])[0] for m in ms) / len(ms)
            d["cka_first_last_mean"] = sum(adj(m["cka_pass_matrix"])[1] for m in ms) / len(ms)
        else:
            d["cka_depth_matched_adjacent_mean"] = sum(adj(m["cka_depth_matched"])[0] for m in ms) / len(ms)
            d["cka_depth_matched_first_last_mean"] = sum(adj(m["cka_depth_matched"])[1] for m in ms) / len(ms)
        res["groups"][g] = d
    json.dump(res, open(C.RES / "sq353.json", "w"), indent=1)


if __name__ == "__main__":
    main()
