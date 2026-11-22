"""K5: state tracking (running parity), single pass vs looped (claim C18). Design: experiments/checks/PREDICTIONS.md."""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import save, seed_all, train_steps  # noqa: E402

import torch  # noqa: E402

from gogeta.model import FusionDance, ModelConfig  # noqa: E402

PREDICTION = "(i) < 75%; (ii) better than (i), but by an uncertain amount; (iii) ~ (ii); (iv) < 75%."
CRITERION = ("'Linear-time mixers are weak at state tracking' is supported if (i) < 90%. 'Looping adds that depth' "
             "is VERIFIED if (ii) >= (i) + 10 points, and REFUTED if (ii) < (i) + 3 points. Comparing (ii) with "
             "(iii) says whether it's the loop or just depth (reported, no criterion).")

T, BATCH, LR, WARMUP = 64, 128, 1e-3, 100
STEPS = int(os.environ.get("K5_STEPS", 3000))
SEED, EVAL_BATCHES = 0, 20
DEV = "cuda"
BASE = dict(vocab_size=2, d_model=128, n_heads=2, mlp_hidden=256, max_seq_len=T)

MODELS = {
    "i_gdn_1pass": dict(prelude=["gdn"], core=[], coda=[]),
    "ii_gdn_looped_x4": dict(prelude=[], core=["gdn"], coda=[], loops=4, loops_min=4, loops_max=4, bptt_loops=4),
    "iii_gdn_4_unshared": dict(prelude=["gdn"] * 4, core=[], coda=[]),
    "iv_attn_1": dict(prelude=["attn"], core=[], coda=[]),
}


def parity_batch(B, gen):
    x = torch.randint(0, 2, (B, T), generator=gen, device=DEV)
    y = torch.cumsum(x, dim=1) % 2  # XOR of all bits so far
    return x, y


@torch.no_grad()
def evaluate(m, loops):
    m.eval()
    gen = torch.Generator(device=DEV).manual_seed(10_000 + SEED)
    correct = total = 0
    for _ in range(EVAL_BATCHES):
        x, y = parity_batch(BATCH, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(x, loops=loops)
        pred = logits.argmax(-1)[:, 32:64]
        correct += (pred == y[:, 32:64]).sum().item()
        total += pred.numel()
    return correct / total


def main():
    started = time.time()
    res = {}
    for name, spec in MODELS.items():
        t0 = time.time()
        seed_all(SEED)
        m = FusionDance(ModelConfig(**BASE, **spec)).to(DEV)
        loops = 4 if spec["core"] else None
        gen = torch.Generator(device=DEV).manual_seed(SEED)
        losses = train_steps(m, lambda s: parity_batch(BATCH, gen), STEPS, lr=LR, warmup=WARMUP, loops=loops,
                             log_every=500)
        acc = evaluate(m, loops)
        res[name] = {"acc_pos_32_63": acc, "final_train_loss": losses[-1],
                     "train_loss_at": {str(s): losses[s] for s in range(0, STEPS, max(1, STEPS // 6))},
                     "params": sum(p.numel() for p in m.parameters()), "train_s": round(time.time() - t0, 1)}
        print(f"{name}: acc {acc:.4f} final loss {losses[-1]:.4f} ({res[name]['train_s']} s)", flush=True)
        del m
        torch.cuda.empty_cache()
    a1, a2 = 100 * res["i_gdn_1pass"]["acc_pos_32_63"], 100 * res["ii_gdn_looped_x4"]["acc_pos_32_63"]
    a3, a4 = 100 * res["iii_gdn_4_unshared"]["acc_pos_32_63"], 100 * res["iv_attn_1"]["acc_pos_32_63"]
    weak = a1 < 90
    if a2 >= a1 + 10:
        loop_v = f"VERIFIED ((ii) {a2:.1f}% >= (i) {a1:.1f}% + 10)"
    elif a2 < a1 + 3:
        loop_v = f"REFUTED ((ii) {a2:.1f}% < (i) {a1:.1f}% + 3)"
    else:
        loop_v = f"NEITHER (between +3 and +10 points: (ii) {a2:.1f}% vs (i) {a1:.1f}%)"
    verdicts = {
        "linear_mixers_weak_at_state_tracking": (f"SUPPORTED ((i) {a1:.1f}% < 90%)" if weak
                                                 else f"NOT SUPPORTED ((i) {a1:.1f}% >= 90%)"),
        "C18_looping_adds_depth": loop_v,
        "loop_vs_depth_reported": f"(ii) looped {a2:.1f}% vs (iii) 4 unshared {a3:.1f}% (diff {a2 - a3:+.1f} pts)",
        "acc_pct": {"i": round(a1, 2), "ii": round(a2, 2), "iii": round(a3, 2), "iv": round(a4, 2)},
    }
    data = {"task": {"seq_len": T, "input": "uniform random bits", "target": "running XOR (parity so far)",
                     "loss": "all positions", "scored": "positions 32-63 of held-out sequences",
                     "eval": f"{EVAL_BATCHES} batches x {BATCH}, generator seed {10_000 + SEED}"},
            "base_config": BASE, "models": MODELS,
            "train": {"steps": STEPS, "batch": BATCH, "lr": LR, "warmup": WARMUP, "schedule": "cosine to 0.1x",
                      "optimizer": "AdamW(0.9,0.95) wd 0.1, clip 1.0, bf16 autocast", "seed": SEED},
            "results": res}
    save("K5", ["C18"], PREDICTION, CRITERION, verdicts, data, started)


if __name__ == "__main__":
    main()
