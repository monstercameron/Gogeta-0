"""SQ76 (hardware research): BitLinear cost paths on the N1X (claims C120, C121; context for C14).

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_bitlinear_paths.py          # GPU, ~3-5 min

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, proposal "ternary/int path for BitLinear".

K2 measured BitLinear 4.1-5.6x slower than nn.Linear forward, but in EAGER mode, where each quantisation op
(abs, mean, round, clamp, amax, ...) is its own kernel and its own pass over memory. Training runs compiled.
Two questions:
  (1) how much of the overhead survives torch.compile (fusion)?            -> C120
  (2) does a real int8 tensor-core path (ternary weights stored as int8 {-1,0,1}, per-token int8 activations,
      torch._int_mm -> int32, rescale) make ternary INFERENCE faster than a bf16 nn.Linear at our shapes? -> C121
Shapes: x [16*1024, d_in] (K2's input), d_in->d_out in {384->1024, 384->2048, 1024->384}.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from gogeta.model import BitLinear  # noqa: E402

CHECK_ID, FILE_STEM = "SQ76", "sq_hw_bitlinear_paths"
CLAIMS = ["C120", "C121", "C14"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "Compiled BitLinear vs compiled nn.Linear: forward 1.2-2.5x slower (K2 eager: 4.1-5.6x), fwd+bwd 1.1-1.8x. The "
    "int8 _int_mm ternary inference path (activation quantisation included) is 0.6-1.2x the compiled bf16 "
    "nn.Linear forward time: no reliable speedup at d=384, because these matmuls are bandwidth-limited and the "
    "quantise/rescale passes cost about as much as the matmul. Its output matches BitLinear's forward within a "
    "relative error of 1e-2."
)
CRITERION = (
    "C120 (most of K2's BitLinear overhead is eager-mode elementwise cost that torch.compile removes) is VERIFIED if "
    "the compiled forward slowdown is <= 2.0x on every shape (vs 4.1-5.6x eager in K2). C121 (an int8 tensor-core "
    "path makes ternary inference faster than bf16 at our shapes) is VERIFIED if int8 forward <= 0.8x compiled bf16 "
    "Linear on >= 2 of 3 shapes with rel err <= 1e-2; otherwise REFUTED, and C14 stays WRONG FOR OUR CODE."
)
SEED = 0
DEV = "cuda"
M = 16 * 1024
SHAPES = [(384, 1024), (384, 2048), (1024, 384)]


def ternary_int8(w):
    """BitNet b1.58 absmean ternary weights as int8 {-1, 0, 1} plus the fp32 scale (same as BitLinear)."""
    scale = w.abs().mean().clamp(min=1e-5)
    return (w / scale).round().clamp(-1, 1).to(torch.int8), scale


def int8_forward(x, w_t_int8, w_scale):
    """Per-token absmax int8 activations x ternary int8 weights on the int8 tensor cores."""
    s = 127.0 / x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    xq = (x.float() * s).round().clamp(-128, 127).to(torch.int8)
    acc = torch._int_mm(xq, w_t_int8)                 # int32 [M, N]
    return (acc.float() * (w_scale / s)).to(torch.bfloat16)


def main():
    t_start = time.time()
    seed_all(SEED)
    data = {"M": M, "rows": []}
    c120 = True
    wins = 0
    for d_in, d_out in SHAPES:
        row = {"shape": f"{d_in}->{d_out}"}
        x = torch.randn(M, d_in, device=DEV, dtype=torch.bfloat16)
        xg = x.clone().requires_grad_(True)
        for name, cls in (("linear", nn.Linear), ("bitlinear", BitLinear)):
            seed_all(0)
            mod = cls(d_in, d_out, bias=False).to(DEV)
            for tag, m in (("eager", mod), ("compiled", torch.compile(mod))):
                def fwd(m=m):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        return m(x)

                def fwdbwd(m=m, mod=mod):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = m(xg)
                    y.float().sum().backward()
                    mod.weight.grad = None
                    xg.grad = None
                row[f"{name}_{tag}_fwd_ms"] = round(bench_ms(fwd, warmup=5, iters=50), 4)
                row[f"{name}_{tag}_fwdbwd_ms"] = round(bench_ms(fwdbwd, warmup=5, iters=30), 4)
            if name == "bitlinear":
                bit = mod
        for tag in ("eager", "compiled"):
            row[f"{tag}_fwd_slowdown_x"] = round(row[f"bitlinear_{tag}_fwd_ms"] / row[f"linear_{tag}_fwd_ms"], 3)
            row[f"{tag}_fwdbwd_slowdown_x"] = round(row[f"bitlinear_{tag}_fwdbwd_ms"] /
                                                    row[f"linear_{tag}_fwdbwd_ms"], 3)
        c120 &= row["compiled_fwd_slowdown_x"] <= 2.0
        # int8 inference path
        try:
            wq, ws = ternary_int8(bit.weight.detach())
            w_t = wq.t().contiguous().t()          # [K, N] column-major view, the cuBLASLt int8 layout
            f8 = torch.compile(int8_forward)
            with torch.no_grad():
                y_int = f8(x, w_t, ws)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y_bit = bit(x)
                rel = ((y_int.float() - y_bit.float()).norm() / y_bit.float().norm()).item()
                row["int8_fwd_ms"] = round(bench_ms(lambda: f8(x, w_t, ws), warmup=5, iters=50), 4)
            row["int8_rel_err_vs_bitlinear"] = rel
            row["int8_over_linear_compiled"] = round(row["int8_fwd_ms"] / row["linear_compiled_fwd_ms"], 3)
            wins += int(row["int8_over_linear_compiled"] <= 0.8 and rel <= 1e-2)
        except Exception as e:
            row["int8_error"] = f"{type(e).__name__}: {e}"[:300]
        data["rows"].append(row)
        print(" ", row, flush=True)
    verdicts = {
        "C120": "VERIFIED" if c120 else "REFUTED (compiled BitLinear still > 2x on some shape)",
        "C121": f"{'VERIFIED' if wins >= 2 else 'REFUTED'}: int8 path <= 0.8x bf16 on {wins}/3 shapes",
        "C14": "stays WRONG FOR OUR CODE" if wins < 2 else "int8 inference path is faster: update C14's wording",
    }
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, verdicts, data, t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
