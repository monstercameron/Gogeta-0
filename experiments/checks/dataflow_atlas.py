"""Data-flow atlas of the assembled model (SQ439; the human asked, 2026-11-23 ~20:10, for the data passed between layers
to be measured and refined into new techniques). Descriptive and diagnostic only: no claim changes and no registration.

What it measures, per checkpoint, on packed 1024-token windows of two slices (in-corpus clean80 v1-val docs, and the
SQ99 out-of-corpus files), CPU, fp32, GDN on the torch reference kernel (sq339_cpu):
  * residual stream at every block boundary (embed, prelude, each injection, every core block of every pass, coda, out):
    RMS, effective rank (participation ratio of the token covariance spectrum), anisotropy (|mean unit vector|^2),
    crest factor (max|x| / rms per token, median), and a logit lens (next-token CE of head(norm_out(h)));
  * every block's two residual writes (mixer, MLP): ||delta|| / ||x|| and cos(delta, x);
  * loop core: pass-to-pass cosine of h and size of each pass's change; injection: the contractive gate lam (stats), and
    ||lam*h|| vs ||inject(...)|| (how much carried state vs re-injected prelude signal enters each pass);
  * every BitLinear: relative output error of the ternary layer vs the same layer with its latent fp weights and fp
    activations, split into weight-only and activation-only parts, plus the input crest factor (int8 headroom);
  * GDN: per-head decay horizon 1 / (1 - exp(g)) (median, tokens) and write strength beta (mean);
  * attention: entropy of the attention distribution (nats) and mean attended distance (tokens), per head.
    python3 experiments/checks/dataflow_atlas.py [--arms A2s,A2f,E27,E24,E20] [--seeds 2,3,4] [--windows 12]
      -> experiments/checks/results/dataflow_atlas.json (+ dataflow_atlas_summary.md)
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sq339_cpu as C  # noqa: E402
from sq379_a2_out_of_corpus import ooc_docs  # noqa: E402

LR = {"A2s": 0.006, "A2f": 0.006, "E27": 0.006, "E24": 0.003, "E20": 0.003, "A3mx": 0.006}
T = 1024
torch.set_grad_enabled(False)


def find_runs(arms, seeds):
    import glob
    runs = {}
    for mp in sorted(glob.glob(f"{C.FLEET}/*/fusiondance-data/runs/*/meta.json")):
        m = json.load(open(mp))
        e, c = m.get("experiment"), m["config"]
        if e in arms and c["train"].get("max_tokens") == 200_000_000 and abs(c["train"]["lr"] - LR[e]) < 1e-12 \
                and c["train"]["seed"] in seeds:
            run = mp.split("/")[-2]
            if glob.glob(mp.replace("/runs/", "/checkpoints/").replace("meta.json", "model.pt")):
                runs.setdefault((e, c["train"]["seed"]), run)
    return runs


def windows(texts, n):
    tok = C.tokenizer()
    ids = []
    for t in texts:
        ids += tok.encode(t + "\n\n").ids
        if len(ids) >= n * T + 1:
            break
    x = torch.tensor(ids[: n * T + 1])
    return x[:-1].view(n, T)[:, :], torch.cat([x[1:]]).view(n, T)


# ----------------------------------------------------------------------------- stats
def state_stats(h, model, y):
    """h [B,T,d] -> dict of residual-stream statistics, incl. a logit-lens CE against next tokens y."""
    B, Tn, d = h.shape
    flat = h.reshape(-1, d)
    rms = flat.pow(2).mean(-1).sqrt()
    crest = (flat.abs().amax(-1) / rms.clamp(min=1e-8)).median().item()
    cen = flat - flat.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh((cen.T @ cen) / cen.shape[0]).clamp(min=0)
    eff_rank = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item()
    unit = F.normalize(flat, dim=-1)
    aniso = unit.mean(0).pow(2).sum().item()
    ce = 0.0
    for b in range(B):  # chunked logit lens
        lg = model.head(model.norm_out(h[b]))
        ce += F.cross_entropy(lg.float(), y[b], reduction="sum").item()
    return {"rms": rms.mean().item(), "eff_rank": eff_rank, "anisotropy": aniso, "crest": crest,
            "lens_ce": ce / (B * Tn)}


def rel(a, b):
    return (a.norm() / b.norm().clamp(min=1e-12)).item()


def cosmean(a, b):
    return F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).mean().item()


def block_step(blk, x, name, rec):
    dm = blk.mixer(blk.norm1(x))
    x1 = x + dm
    dl = blk.mlp(blk.norm2(x1)) if not blk.moe else blk.mlp(blk.norm2(x1), [])
    rec[name] = {"kind": blk.kind, "mix_rel": rel(dm, x), "mix_cos": cosmean(dm, x),
                 "mlp_rel": rel(dl, x1), "mlp_cos": cosmean(dl, x1)}
    return x1 + dl


# ----------------------------------------------------------------------------- module hooks
def install_hooks(model, store):
    from gogeta.model import Attention, BitLinear, GatedDeltaNet
    hs = []
    names = {m: n for n, m in model.named_modules()}

    def bitlin(mod, inp, out):
        x = inp[0]
        w = mod.weight
        scale = w.abs().mean().clamp(min=1e-5)
        wq = (w / scale).round().clamp(-1, 1) * scale
        s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        xq = (x * s).round().clamp(-128, 127) / s
        y_fp = F.linear(x, w)
        d = store.setdefault("bitlinear", {}).setdefault(names[mod], [])
        d.append({"err_total": rel(out - y_fp, y_fp), "err_weight": rel(F.linear(x, wq) - y_fp, y_fp),
                  "err_act": rel(F.linear(xq, w) - y_fp, y_fp),
                  "in_crest": (x.abs().amax(-1) / x.pow(2).mean(-1).sqrt().clamp(min=1e-8)).median().item()})

    def gdn(mod, inp):
        x = inp[0]
        g = -mod.A_log.float().exp() * F.softplus(mod.a(x).float() + mod.dt_bias.float())   # [B,T,H], <= 0
        a = g.exp().clamp(max=1 - 1e-6)
        hor = (1.0 / (1.0 - a)).flatten(0, 1).median(0).values                                 # per head, tokens
        beta = torch.sigmoid(mod.b(x).float()).flatten(0, 1).mean(0) if not mod.beta_fixed else torch.full((mod.H,), mod.beta_fixed)
        store.setdefault("gdn", {}).setdefault(names[mod], []).append({"horizon_med": hor.tolist(), "beta_mean": beta.tolist()})

    def attn(mod, inp):
        x = inp[0]
        B, Tn, D = x.shape
        H, hd = mod.cfg.n_heads, mod.cfg.head_dim
        q, k, _ = mod.qkv(x).view(B, Tn, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = mod.rope(q), mod.rope(k)
        i = torch.arange(Tn)
        dist = (i[:, None] - i[None, :]).float()
        mask = dist >= 0
        if mod.window is not None and mod.window < Tn:
            mask &= dist < mod.window
        ent, md = torch.zeros(H), torch.zeros(H)
        for b in range(min(B, 4)):  # 4 windows are enough for per-head attention statistics
            sc = (q[b] @ k[b].transpose(-1, -2)) * hd ** -0.5
            p = sc.masked_fill(~mask, float("-inf")).softmax(-1)
            ent += -(p * p.clamp(min=1e-12).log()).sum(-1).mean(-1)
            md += (p * dist.clamp(min=0)).sum(-1).mean(-1)
        n = min(B, 4)
        store.setdefault("attn", {}).setdefault(names[mod], []).append(
            {"window": mod.window, "entropy": (ent / n).tolist(), "mean_dist": (md / n).tolist()})

    for m in model.modules():
        if isinstance(m, BitLinear):
            hs.append(m.register_forward_hook(bitlin))
        elif isinstance(m, GatedDeltaNet):
            hs.append(m.register_forward_pre_hook(gdn))
        elif isinstance(m, Attention):
            hs.append(m.register_forward_pre_hook(attn))
    return hs


# ----------------------------------------------------------------------------- instrumented forward
def atlas_forward(model, x, y):
    cfg = model.cfg
    rec_blocks, states, loop = {}, {}, {}
    h = model.embed(x)
    states["embed"] = state_stats(h, model, y)
    for j, b in enumerate(model.prelude):
        h = block_step(b, h, f"prelude.{j}", rec_blocks)
        states[f"prelude.{j}"] = state_stats(h, model, y)
    if model.core:
        r = cfg.loops
        e = h
        lam = torch.sigmoid(model.lam_logit) if cfg.inject_contractive else None
        if lam is not None:
            q = torch.quantile(lam.float(), torch.tensor([0.05, 0.5, 0.95]))
            loop["lam"] = {"mean": lam.mean().item(), "p05": q[0].item(), "p50": q[1].item(), "p95": q[2].item()}
        prev = None
        for i in range(r):
            inj = model.inject(torch.cat([model.inj_norm_h(h), model.inj_norm_e(e)], dim=-1))
            carried = lam * h if lam is not None else h
            loop[f"pass{i + 1}.inject"] = {"carried_rms": carried.pow(2).mean().sqrt().item(),
                                           "inject_rms": inj.pow(2).mean().sqrt().item(),
                                           "inject_share": rel(inj, carried + inj), "cos_inj_carried": cosmean(inj, carried)}
            h = carried + inj
            if cfg.loop_embed:
                h = h + model.loop_emb[min(i, model.loop_emb.shape[0] - 1)]
            states[f"pass{i + 1}.in"] = state_stats(h, model, y)
            start = h
            for j, b in enumerate(model.core):
                h = block_step(b, h, f"pass{i + 1}.core.{j}", rec_blocks)
            states[f"pass{i + 1}.out"] = state_stats(h, model, y)
            loop[f"pass{i + 1}.change"] = {"rel": rel(h - start, start), "cos_in_out": cosmean(h, start)}
            if prev is not None:
                loop[f"pass{i}->pass{i + 1}"] = {"cos": cosmean(h, prev), "rel_diff": rel(h - prev, prev)}
            prev = h
    for j, b in enumerate(model.coda):
        h = block_step(b, h, f"coda.{j}", rec_blocks)
        states[f"coda.{j}"] = state_stats(h, model, y)
    return {"blocks": rec_blocks, "states": states, "loop": loop}


def agg_hooks(store):
    out = {}
    for kind, d in store.items():
        out[kind] = {}
        for name, lst in d.items():
            keys = lst[0].keys()
            agg = {}
            for k in keys:
                v = [e[k] for e in lst]
                if isinstance(v[0], list):
                    agg[k] = [sum(col) / len(col) for col in zip(*v)]
                elif isinstance(v[0], (int, float)) and v[0] is not None:
                    agg[k] = sum(v) / len(v)
                else:
                    agg[k] = v[0]
            out[kind][name] = agg
    return out


def run_one(run, slices):
    model = C.load_model(run)
    res = {"run": run, "config": {k: getattr(model.cfg, k) for k in ("prelude", "core", "coda", "loops", "ternary",
                                                                       "ternary_fp_ends", "inject_contractive")}}
    for sname, (x, y) in slices.items():
        store = {}
        hs = install_hooks(model, store)
        res[sname] = atlas_forward(model, x, y)
        res[sname]["modules"] = agg_hooks(store)
        for hh in hs:
            hh.remove()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A2s,A2f,E27,E24,E20")
    ap.add_argument("--seeds", default="2,3,4")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(C.RES, "dataflow_atlas.json"))
    a = ap.parse_args()
    arms, seeds = a.arms.split(","), [int(s) for s in a.seeds.split(",")]
    runs = find_runs(arms, seeds)
    print(f"{len(runs)} runs: {sorted(runs)}", flush=True)
    inc = [t for _, t in C.subset(400, seed=439)]
    ooc, _ = ooc_docs()
    slices = {"in_corpus": windows(inc, a.windows), "out_of_corpus": windows([t for _, t in ooc], a.windows)}
    out = {"check": "SQ439 data-flow atlas (descriptive)", "T": T, "windows_per_slice": a.windows,
           "git": C.git_commit(), "runs": {}}
    for key in sorted(runs):
        t0 = time.time()
        out["runs"][f"{key[0]}_s{key[1]}"] = run_one(runs[key], slices)
        print(f"{key} {runs[key]} done in {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(a.out, "w", newline="\n"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
