"""SQ73 (hardware research): the LM head + loss. Does a chunked fused linear-cross-entropy (fla, already installed)
beat our `logits.float()` + F.cross_entropy on this bandwidth-limited GPU? (claim C117)

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_loss_head.py          # GPU, ~2-4 min

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, proposal "fused linear-CE". Patch: notes/patches/fused-linear-ce.patch.

Why: per training step (B=32, T=1024, vocab 8192) our head writes 32768 x 8192 bf16 logits (537 MB), upcasts them
to fp32 for the loss (1.07 GB), and writes/reads the logits gradient again in backward. On a ~300 GB/s part that
traffic alone is several ms of a ~156 ms E0 step (computed). fla's FusedLinearCrossEntropyLoss computes the loss
and both gradients chunk by chunk without materialising the full logits. With chunk = 1024 tokens a bf16 logits
chunk is 16 MiB, which fits in a 24 MiB L2 (GB10 figure; SQ71 checks the N1X's L2).

Arms (h: [32, 1024, 384] bf16 requiring grad; W: [8192, 384] fp32 parameter, tied like ours)
  eager     ours, exactly as gogeta/model.py: F.linear under bf16 autocast, then CE on logits.float()
  compiled  the same function under torch.compile (what training actually runs)
  flce_8 / flce_32   fla FusedLinearCrossEntropyLoss(num_chunks=8 / 32) on h and W.to(bf16)
Measured: fwd+bwd ms, peak extra memory, |loss diff|, cosine of grad_h and grad_W vs eager.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CHECK_ID, FILE_STEM = "SQ73", "sq_hw_loss_head"
CLAIMS = ["C117"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "eager head+loss fwd+bwd 15-35 ms; compiled 12-25 ms (it fuses the upcast but still materialises the logits); "
    "fla flce_32 0.55-0.85x compiled, flce_8 a bit slower than flce_32; peak extra memory: eager/compiled 1.5-3 GB, "
    "flce < 0.5 GB. |loss diff| < 2e-3 and grad cosines > 0.999 vs eager for every arm."
)
CRITERION = (
    "C117 (a chunked fused linear-CE cuts the head+loss time by >= 20% vs our compiled path at B=32, T=1024, with "
    "the same loss) is VERIFIED if the best flce arm is <= 0.8x the compiled arm's fwd+bwd time with |loss diff| < "
    "2e-3 and grad cosines > 0.999 (grad_h and grad_W); REFUTED if the best flce arm is >= 1.0x."
)
SEED = 0
DEV = "cuda"
B, T, D, VOCAB = 32, 1024, 384, 8192


def ours(h, W, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = F.linear(h, W)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))


def peak_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def main():
    t_start = time.time()
    seed_all(SEED)
    from fla.modules import FusedLinearCrossEntropyLoss

    h = (torch.randn(B, T, D, device=DEV) * 1.0).bfloat16().requires_grad_(True)
    W = torch.nn.Parameter(torch.randn(VOCAB, D, device=DEV) * 0.02)
    y = torch.randint(0, VOCAB, (B, T), device=DEV)
    compiled = torch.compile(ours)
    arms = {"eager": lambda: ours(h, W, y), "compiled": lambda: compiled(h, W, y)}
    for nc in (8, 32):
        crit = FusedLinearCrossEntropyLoss(num_chunks=nc)
        arms[f"flce_{nc}"] = (lambda c=crit: c(h, y, W.to(torch.bfloat16)))

    data = {"sizes": dict(B=B, T=T, D=D, vocab=VOCAB), "arms": {}}
    ref = None
    for name, fn in arms.items():
        try:
            loss = fn()
            loss.backward()
            gh, gw = h.grad.detach().float().clone(), W.grad.detach().float().clone()
            h.grad = W.grad = None
            if ref is None:
                ref = (loss.item(), gh, gw)

            def step(fn=fn):
                fn().backward()
                h.grad = W.grad = None
            ms = bench_ms(step, warmup=3, iters=15)
            mb = peak_mb(step)
            row = {"fwd_bwd_ms": round(ms, 3), "peak_extra_mb": round(mb, 1), "loss": loss.item(),
                   "loss_diff": abs(loss.item() - ref[0]),
                   "grad_h_cos": F.cosine_similarity(gh.flatten(), ref[1].flatten(), dim=0).item(),
                   "grad_W_cos": F.cosine_similarity(gw.flatten(), ref[2].flatten(), dim=0).item()}
        except Exception as e:
            row = {"error": f"{type(e).__name__}: {e}"[:300]}
        data["arms"][name] = row
        print(f"  {name:9s} {row}")

    a = data["arms"]
    flce = [a[k] for k in ("flce_8", "flce_32") if "fwd_bwd_ms" in a.get(k, {})]
    if flce and "fwd_bwd_ms" in a.get("compiled", {}):
        best = min(flce, key=lambda r: r["fwd_bwd_ms"])
        r = best["fwd_bwd_ms"] / a["compiled"]["fwd_bwd_ms"]
        same = best["loss_diff"] < 2e-3 and best["grad_h_cos"] > 0.999 and best["grad_W_cos"] > 0.999
        v = ("VERIFIED" if (r <= 0.8 and same) else ("REFUTED" if r >= 1.0 or not same else "inconclusive"))
        v117 = f"{v}: best flce / compiled = {r:.2f}, numerics {'equal' if same else 'DIFFER'}"
    else:
        v117 = "inconclusive: an arm failed (see data.arms)"
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, {"C117": v117}, data, t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
