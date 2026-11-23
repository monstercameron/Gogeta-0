"""SQ392 (registered in PREDICTIONS.md, section SQ392 + protocol clarification): positional control for the long-context result.

SQ351 found the GDN hybrid (cell 100) far better than dense (cell 000) past the 1,024 training length. Does linear RoPE position interpolation close the gap?
Dense cell 000 (best LR, seeds 0-2) is scored at window 2,048 on the same two doc sets as SQ351 (A: the 3 clean80 docs > 1,200 tokens; B: the 153 SQ99 out-of-corpus
files > 1,200 tokens) in two arms, both built with max_seq_len 2048 so the RoPE table covers every scored position: raw positions 0-2047 (baseline, = SQ351), and
interpolated positions (position / 2 before the cos/sin lookup; scale 1/2 for 2048 / 1024). Nothing else differs. The hybrid (cell 100) arm is taken unchanged from
experiments/checks/results/sq351.json (same code path, same docs, same seeds). Output: mean NLL (nats/token) by target-position bin (0-512, 512-1024, 1024-2048) per
seed; the hybrid's advantage over dense in the last bin = 100 * (exp(mean_nats_hybrid - mean_nats_dense) - 1), paired by seed, without and with interpolation.
Control: the raw dense arm must reproduce SQ351's dense numbers (max abs diff reported).
Registered prediction: the advantage in 1024-2048 shrinks from -60.7% (set B) to about -30% (range -45% to -10%). Rule: still beyond -50% -> stands as stated; inside -30% ->
restated as "vs unscaled RoPE".

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq392_rope_interpolation.py -> results/sq392.json
"""
import json
import math
import time

import torch

import sq339_cpu as C
from sq351_length_extrap import BINS, WIN, bin_means, load_sets, per_position_nll, rebuild_rope


def rebuild_rope_interp(model, n, scale):
    from gogeta.model import Rotary

    r = Rotary(model.cfg.head_dim, n, model.cfg.rope_base)
    hd = model.cfg.head_dim
    inv = 1.0 / (model.cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(n).float() * scale
    freqs = torch.outer(t, inv)
    model.rope.cos, model.rope.sin = freqs.cos()[None, None].clone(), freqs.sin()[None, None].clone()
    assert model.rope.cos.shape[2] == n


def main():
    log = C.make_log(str(C.RES / "sq392.log"))
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    sets = load_sets(tok, bos)
    runs = C.best_runs()
    s351 = json.load(open(C.RES / "sq351.json"))
    res = {"check": "SQ392", "git": C.git_commit(), "status": "registered (PREDICTIONS.md SQ392)", "window": WIN, "bins": [f"{a}-{b}" for a, b in BINS],
           "sets": {k: len(v["ids"]) for k, v in sets.items()}, "dense": {}}
    for seed in (0, 1, 2):
        model = C.load_model(runs["000"][seed])
        t = time.time()
        row = {"run": runs["000"][seed]}
        for arm in ("raw", "interp"):
            if arm == "raw":
                rebuild_rope(model, WIN)
            else:
                rebuild_rope_interp(model, WIN, 1024 / WIN)   # scale 1/2: positions / 2
            row[arm] = {}
            for name, st in sets.items():
                pp = per_position_nll(model, st["ids"], WIN)
                row[arm][name] = bin_means(pp)
            log(f"dense s{seed} {arm} ({time.time() - t:.0f}s): " + " | ".join(f"{n}: " + " ".join(f"{b['nats_per_token']:.4f}" for b in row[arm][n]) for n in sets))
        res["dense"][str(seed)] = row
        json.dump(res, open(C.RES / "sq392.json", "w"), indent=1)
    # control: raw arm equals SQ351's dense numbers
    ctrl = 0.0
    for s, row in res["dense"].items():
        for name in sets:
            for a, b in zip(row["raw"][name], s351["cells"]["000"]["per_seed"][s][name]["bins_window2048"]):
                ctrl = max(ctrl, abs(a["nats_per_token"] - b["nats_per_token"]))
    res["control_raw_vs_sq351_max_abs_nats"] = ctrl
    res["summary"] = {}
    for name in sets:
        out = {}
        for arm in ("raw", "interp"):
            adv = []
            for s in res["dense"]:
                hyb = s351["cells"]["100"]["per_seed"][s][name]["bins_window2048"]
                den = res["dense"][s][arm][name]
                adv.append([100 * (math.exp(h["nats_per_token"] - d["nats_per_token"]) - 1) for h, d in zip(hyb, den)])
            out[arm] = {"hybrid_vs_dense_pct_by_bin": {b: C.mean_ci([a[i] for a in adv]) for i, b in enumerate(res["bins"])}}
            out[arm]["dense_nats_by_bin"] = {b: C.mean_ci([res["dense"][s][arm][name][i]["nats_per_token"] for s in res["dense"]]) for i, b in enumerate(res["bins"])}
        out["interp_change_in_dense_nats_pct_by_bin"] = {b: C.mean_ci([100 * (res["dense"][s]["interp"][name][i]["nats_per_token"] / res["dense"][s]["raw"][name][i]["nats_per_token"] - 1) for s in res["dense"]]) for i, b in enumerate(res["bins"])}
        res["summary"][name] = out
        log(f"{name}: hybrid vs dense % last bin raw {out['raw']['hybrid_vs_dense_pct_by_bin']['1024-2048']['mean']:+.1f} interp {out['interp']['hybrid_vs_dense_pct_by_bin']['1024-2048']['mean']:+.1f}; "
            f"interp changes dense nats by bin " + " ".join(f"{v['mean']:+.1f}%" for v in out["interp_change_in_dense_nats_pct_by_bin"].values()))
    json.dump(res, open(C.RES / "sq392.json", "w"), indent=1)


if __name__ == "__main__":
    main()
