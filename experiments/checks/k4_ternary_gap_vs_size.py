"""K4: does the ternary gap shrink with model size at our scale? (claim C17). Design: experiments/checks/PREDICTIONS.md."""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import save, seed_all, train_steps  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gogeta.model import FusionDance, ModelConfig  # noqa: E402

PREDICTION = ("The gap shrinks as d grows (for example ~12% -> ~8% -> ~5%), and all gaps are positive.")
CRITERION = ("C17's local analogue ('the gap shrinks with size at our scale') is VERIFIED if the mean relative gap "
             "strictly decreases from d=64 to d=128 to d=256. The 3B-parity part stays UNVERIFIED (not testable "
             "here), so the overall status is at best PARTIAL.")

DATA = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
T, BATCH, LR, WARMUP = 512, 64, 3e-3, 100
STEPS = int(os.environ.get("K4_STEPS", 600))
DS, SEEDS, N_VAL = [64, 128, 256], [0, 1], 64
DEV = "cuda"


def main():
    started = time.time()
    train = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
    vs = np.stack([np.asarray(val[i * T:(i + 1) * T + 1], dtype=np.int64) for i in range(N_VAL)])
    vx, vy = torch.from_numpy(vs[:, :-1]).to(DEV), torch.from_numpy(vs[:, 1:]).to(DEV)

    runs = []
    for seed in SEEDS:
        offs = np.random.default_rng(seed).integers(0, len(train) - T - 1, size=(STEPS, BATCH))

        def batch_fn(s):
            b = np.stack([np.asarray(train[o:o + T + 1], dtype=np.int64) for o in offs[s]])
            b = torch.from_numpy(b).to(DEV)
            return b[:, :-1], b[:, 1:]

        for d in DS:
            for ternary in [False, True]:
                t0 = time.time()
                seed_all(seed)
                cfg = ModelConfig(vocab_size=8192, d_model=d, n_heads=d // 64, mlp_hidden=3 * d,
                                  prelude=["gdn", "attn"], core=[], coda=[], max_seq_len=T, ternary=ternary)
                m = FusionDance(cfg).to(DEV)
                losses = train_steps(m, batch_fn, STEPS, lr=LR, warmup=WARMUP)
                m.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, vl = m(vx, vy)
                r = {"seed": seed, "d": d, "ternary": ternary, "val_loss": float(vl),
                     "final_train_loss": losses[-1], "params": m.param_counts(),
                     "train_s": round(time.time() - t0, 1)}
                runs.append(r)
                print(f"seed {seed} d {d:3d} ternary {ternary!s:5s}: val {r['val_loss']:.4f} "
                      f"train {losses[-1]:.4f} ({r['train_s']} s)", flush=True)
                del m
                torch.cuda.empty_cache()

    def vl(seed, d, t):
        return next(r["val_loss"] for r in runs if r["seed"] == seed and r["d"] == d and r["ternary"] == t)

    gaps = {d: {str(s): (vl(s, d, True) - vl(s, d, False)) / vl(s, d, False) for s in SEEDS} for d in DS}
    mean_gap = {d: float(np.mean(list(gaps[d].values()))) for d in DS}
    shrinks = mean_gap[64] > mean_gap[128] > mean_gap[256]
    gstr = ", ".join(f"d={d}: {100 * mean_gap[d]:.2f}%" for d in DS)
    verdicts = {
        "C17_local": (f"VERIFIED (gap strictly shrinks: {gstr}) -> ledger status at best PARTIAL" if shrinks else
                      f"NOT VERIFIED (gap does not strictly shrink: {gstr})"),
        "criterion_pass": bool(shrinks),
        "all_gaps_positive": bool(all(g > 0 for d in DS for g in gaps[d].values())),
        "mean_rel_gap_pct": {str(d): round(100 * mean_gap[d], 3) for d in DS},
    }
    data = {"model": "2 blocks [gdn, attn] (prelude, no loop), heads=d/64, mlp=3d, vocab 8192",
            "train": {"steps": STEPS, "batch": BATCH, "seq_len": T, "tokens": STEPS * BATCH * T, "lr": LR,
                      "warmup": WARMUP, "schedule": "cosine to 0.1x",
                      "optimizer": "AdamW(0.9,0.95) wd 0.1, clip 1.0, bf16 autocast",
                      "data_order": "random train windows, np.random.default_rng(seed); same for fp and ternary"},
            "val": f"first {N_VAL} non-overlapping {T}-token windows of val.bin (+1 for targets)",
            "runs": runs, "rel_gap_by_seed": {str(d): gaps[d] for d in DS},
            "mean_rel_gap": {str(d): mean_gap[d] for d in DS}}
    save("K4", ["C17"], PREDICTION, CRITERION, verdicts, data, started)


if __name__ == "__main__":
    main()
