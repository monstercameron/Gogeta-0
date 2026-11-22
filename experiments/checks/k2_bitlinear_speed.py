"""K2: does BitLinear make compute cheaper in our code? (claim C14). Design: experiments/checks/PREDICTIONS.md."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import bench_ms, save, seed_all  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from gogeta.model import BitLinear  # noqa: E402

PREDICTION = "BitLinear is 1.5-5x slower (it simulates rounding in floating point)."
CRITERION = ("C14 'ternary makes compute cheaper' holds for our code only if BitLinear forward time <= Linear "
             "forward time for both shapes. Otherwise it's WRONG FOR OUR CODE (confirmed).")


def peak_mem_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def main():
    started = time.time()
    seed_all(0)
    dev = "cuda"
    shapes = [(384, 1024), (384, 2048)]
    data = {"input_shape": [16, 1024, 384], "dtype": "bf16 autocast, fp32 params", "rows": []}
    ok_all = True
    for d_in, d_out in shapes:
        row = {"shape": f"{d_in}->{d_out}"}
        x = torch.randn(16, 1024, d_in, device=dev)
        for name, cls in [("linear", nn.Linear), ("bitlinear", BitLinear)]:
            seed_all(0)
            m = cls(d_in, d_out, bias=False).to(dev)
            xg = x.clone().requires_grad_(True)

            def fwd():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return m(x)

            def fwdbwd():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = m(xg)
                y.float().sum().backward()
                m.weight.grad = None
                xg.grad = None

            row[f"{name}_fwd_ms"] = round(bench_ms(fwd, warmup=10, iters=50), 4)
            row[f"{name}_fwdbwd_ms"] = round(bench_ms(fwdbwd, warmup=10, iters=50), 4)
            row[f"{name}_fwd_peak_mb"] = round(peak_mem_mb(fwd), 2)
            row[f"{name}_fwdbwd_peak_mb"] = round(peak_mem_mb(fwdbwd), 2)
        row["fwd_slowdown_x"] = round(row["bitlinear_fwd_ms"] / row["linear_fwd_ms"], 3)
        row["fwdbwd_slowdown_x"] = round(row["bitlinear_fwdbwd_ms"] / row["linear_fwdbwd_ms"], 3)
        row["fwd_mem_ratio"] = round(row["bitlinear_fwd_peak_mb"] / row["linear_fwd_peak_mb"], 3)
        row["fwdbwd_mem_ratio"] = round(row["bitlinear_fwdbwd_peak_mb"] / row["linear_fwdbwd_peak_mb"], 3)
        ok_all &= row["bitlinear_fwd_ms"] <= row["linear_fwd_ms"]
        data["rows"].append(row)
        print(row, flush=True)
    verdicts = {
        "C14": ("HOLDS (BitLinear fwd <= Linear fwd for both shapes)" if ok_all
                else "WRONG FOR OUR CODE (confirmed): BitLinear forward is slower than nn.Linear"),
        "criterion_pass": bool(ok_all),
    }
    save("K2", ["C14"], PREDICTION, CRITERION, verdicts, data, started)


if __name__ == "__main__":
    main()
