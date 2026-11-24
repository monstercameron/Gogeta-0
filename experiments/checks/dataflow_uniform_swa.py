"""SQ441 (eval-only, CPU): is the first sliding-window attention layer just a causal moving average?

The SQ439 atlas found the first SWA layer's heads at entropy ~5.3 nats and mean distance ~111 tokens in E24 and A2f, which
is exactly uniform attention over a causal 256-token window at T = 1024. This check replaces that layer's attention
weights by the exact uniform causal window (y_t = mean of v over the last min(t+1, 256) positions; same v and o
projections, no queries or keys) and measures next-token CE on the same windows as the atlas, in and out of corpus.
A CE change near 0 means an attention-free first layer (a causal mean over a window) is a drop-in replacement.
    python3 experiments/checks/dataflow_uniform_swa.py [--arms A2s,A2f,E27,E24] [--seeds 2,3,4] [--windows 12]
      -> experiments/checks/results/dataflow_uniform_swa.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sq339_cpu as C  # noqa: E402
from dataflow_atlas import find_runs, windows  # noqa: E402
from sq379_a2_out_of_corpus import ooc_docs  # noqa: E402

torch.set_grad_enabled(False)


def uniform_forward(mod):
    def fwd(x):
        B, T, D = x.shape
        H, hd = mod.cfg.n_heads, mod.cfg.head_dim
        _, _, v = mod.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)        # v [B,H,T,hd]
        w = mod.window if mod.window is not None else T
        cs = torch.cumsum(v, dim=2)
        shifted = torch.zeros_like(cs)
        if w < T:
            shifted[:, :, w:] = cs[:, :, :-w]
        n = torch.clamp(torch.arange(1, T + 1), max=w).view(1, 1, T, 1).to(v.dtype)
        y = (cs - shifted) / n
        return mod.o(y.transpose(1, 2).reshape(B, T, D))
    return fwd


def ce(model, x, y):
    tot = 0.0
    for b in range(x.shape[0]):
        lg, _ = model(x[b:b + 1])
        tot += F.cross_entropy(lg[0].float(), y[b], reduction="sum").item()
    return tot / x.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A2s,A2f,E27,E24")
    ap.add_argument("--seeds", default="2,3,4")
    ap.add_argument("--windows", type=int, default=12)
    a = ap.parse_args()
    runs = find_runs(a.arms.split(","), [int(s) for s in a.seeds.split(",")])
    inc = [t for _, t in C.subset(400, seed=439)]
    ooc, _ = ooc_docs()
    slices = {"in_corpus": windows(inc, a.windows), "out_of_corpus": windows([t for _, t in ooc], a.windows)}
    out = {"check": "SQ441 uniform first-SWA swap (eval-only)", "windows_per_slice": a.windows, "git": C.git_commit(), "runs": {}}
    for key in sorted(runs):
        model = C.load_model(runs[key])
        first = model.prelude[0].mixer
        assert first.window is not None, "first prelude block is not SWA"
        res = {"run": runs[key]}
        for sl, (x, y) in slices.items():
            base = ce(model, x, y)
            orig = first.forward
            first.forward = uniform_forward(first)
            swp = ce(model, x, y)
            first.forward = orig
            res[sl] = {"ce_base": base, "ce_uniform": swp, "delta_pct": 100 * (swp / base - 1)}
        out["runs"][f"{key[0]}_s{key[1]}"] = res
        print(key, json.dumps({sl: round(res[sl]["delta_pct"], 2) for sl in slices}), flush=True)
        json.dump(out, open(os.path.join(C.RES, "dataflow_uniform_swa.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
