"""SQ71 (hardware research): the N1X roofline: tensor-core peaks per precision, DRAM bandwidth, our layer shapes.

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_roofline.py          # GPU, ~5-8 min

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, sections "Spec table" and "Roofline". Claims C112, C113, C114.

Open question this settles: NVIDIA publishes no BF16/FP8 peak for the N1X. GeForce Blackwell (sm_120) parts run
BF16-with-FP32-accumulate at HALF the FP16-accumulate rate (RTX Blackwell whitepaper, Table 3); GB10 (DGX Spark)
apparently does not (community MAMF ~99.8 TFLOP/s BF16 at 48 SMs). Our N1X reports "Product Brand: GeForce".
We measure FP16 with FP32 accumulate vs FP16 with FP16 accumulate (torch allow_fp16_accumulation): a ~2x ratio
means GeForce-style half rate; ~1x means full rate. We also sample the SM clock with nvidia-smi during each run so
the result can be expressed per clock per SM (FLOP/clk/SM), which does not depend on the laptop's power mode.

What is measured (sustained ~2 s per arm, so clocks settle)
  1. device properties torch reports (SMs, L2, shared memory per block opt-in, registers per SM, memory)
  2. square matmuls N=8192: bf16, fp16 (fp32 acc), fp16 (fp16 acc), tf32, fp8 e4m3 (torch._scaled_mm, per-tensor
     scales), int8 (torch._int_mm)
  3. DRAM: 2 GiB device copy (read+write) and 2 GiB read-only sum; an 8 MiB copy loop (L2-resident)
  4. our training shapes (M = 32768 tokens): qkv 384->1152, mlp up 384->2048, mlp down 1024->384, head 384->8192
  5. host->device: 64 MiB pageable vs pinned copy; time of a fresh 256 KiB .pin_memory() (what TokenData does)
"""
import pathlib
import shutil
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import save, seed_all  # noqa: E402

import torch  # noqa: E402

CHECK_ID, FILE_STEM = "SQ71", "sq_hw_roofline"
CLAIMS = ["C112", "C113", "C114", "C129"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "bf16 8192^3: 70-100 TFLOP/s at an SM clock of 1.8-2.4 GHz, i.e. 850-1100 FLOP/clk/SM (full rate, like GB10). "
    "fp16 with fp16 accumulate is within 1.3x of fp16 with fp32 accumulate (no GeForce half-rate). fp8 e4m3 "
    "1.6-2.1x bf16; int8 1.6-2.1x bf16; tf32 0.2-0.5x bf16. DRAM copy 200-270 GB/s (67-90% of the 300 GB/s spec), "
    "read-only 220-280 GB/s. torch reports 48 SMs, L2 = 24 MiB (same as GB10), 99 KiB shared per block, 64K regs/SM. "
    "Our shapes: qkv/up/down (K or N = 384) 20-50 TFLOP/s (bandwidth-limited), head 50-85 TFLOP/s. Pinned H2D "
    "1.5-4x pageable; a fresh 256 KiB pin_memory() costs < 0.5 ms after the first call (caching host allocator). "
    "torch sees exactly one CUDA device (the NVIDIA NPU is not a CUDA device)."
)
CRITERION = (
    "C112 (N1X tensor cores run BF16/FP16 with FP32 accumulate at full rate, ~1024 FLOP/clk/SM, so the dense BF16 "
    "peak is ~98 TFLOP/s at 2.0 GHz and ~118 at 2.4 GHz) is VERIFIED if fp16acc16/fp16acc32 <= 1.3 and the bf16 rate "
    "is >= 800 FLOP/clk/SM (when a clock was sampled); REFUTED (GeForce half rate, ~512 FLOP/clk/SM) if the ratio is "
    ">= 1.7. C113 (achievable DRAM bandwidth >= 75% of the 300 GB/s spec) is VERIFIED if max(copy, read) >= 225 GB/s. "
    "C114 (N1X L2 = 24 MiB like GB10) is VERIFIED if torch reports L2_cache_size == 24 MiB. C129 (the NPU is not "
    "usable from our CUDA stack) is VERIFIED if torch.cuda.device_count() == 1 inside WSL."
)
SEED = 0
DEV = "cuda"
SECONDS = 2.0
NSMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"


class ClockSampler:
    """Polls nvidia-smi (SM clock, power, temperature) in a thread while a benchmark runs."""

    def __init__(self, period=0.4):
        self.period, self.rows, self._stop = period, [], threading.Event()

    def _run(self):
        q = "--query-gpu=clocks.sm,power.draw,temperature.gpu"
        while not self._stop.is_set():
            try:
                out = subprocess.run([NSMI, q, "--format=csv,noheader,nounits", "-i", "0"], capture_output=True,
                                     text=True, timeout=5).stdout.strip().splitlines()[0]
                clk, pw, tp = (x.strip() for x in out.split(","))
                self.rows.append((float(clk), float(pw) if pw[0].isdigit() else None, float(tp)))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=10)

    def summary(self):
        if not self.rows:
            return {"sm_clock_mhz": None}
        clk = [r[0] for r in self.rows]
        pw = [r[1] for r in self.rows if r[1] is not None]
        return {"sm_clock_mhz": statistics.median(clk), "power_w": statistics.median(pw) if pw else None,
                "temp_c_max": max(r[2] for r in self.rows), "samples": len(self.rows)}


def sustained(fn, seconds=SECONDS):
    """Run fn repeatedly for ~seconds after warm-up; return (ms per call, clock summary)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-6)
    n = max(5, int(seconds / one))
    with ClockSampler() as cs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / n * 1e3, cs.summary()


def main():
    t_start = time.time()
    seed_all(SEED)
    p = torch.cuda.get_device_properties(0)
    props = {"name": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
             "total_memory_gib": round(p.total_memory / 2**30, 2),
             "cuda_device_count": torch.cuda.device_count(),           # C129: the NPU is not a CUDA device
             "arch_list": torch.cuda.get_arch_list()}
    for attr in ("L2_cache_size", "shared_memory_per_block_optin", "shared_memory_per_multiprocessor",
                 "regs_per_multiprocessor", "max_threads_per_multi_processor", "warp_size", "is_integrated"):
        if hasattr(p, attr):
            props[attr] = getattr(p, attr)
    print("device:", props)
    sms = p.multi_processor_count
    data = {"device": props, "matmul": {}, "memory": {}, "shapes": {}, "h2d": {}}

    # ------------------------------------------------------------------ 2. matmul peaks
    N = 8192
    a32 = torch.randn(N, N, device=DEV)
    b32 = torch.randn(N, N, device=DEV)
    flops = 2 * N ** 3

    def record(name, fn):
        try:
            ms, clk = sustained(fn)
            tf = flops / (ms * 1e-3) / 1e12
            row = {"ms": round(ms, 3), "tflops": round(tf, 2), **clk}
            if clk.get("sm_clock_mhz"):
                row["flop_per_clk_per_sm"] = round(tf * 1e12 / (clk["sm_clock_mhz"] * 1e6) / sms, 1)
        except Exception as e:
            row = {"error": f"{type(e).__name__}: {e}"[:300]}
        data["matmul"][name] = row
        print(f"  {name:22s} {row}")

    ab, bb = a32.bfloat16(), b32.bfloat16()
    ah, bh = a32.half(), b32.half()
    record("bf16", lambda: ab @ bb)
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    record("fp16_acc32", lambda: ah @ bh)
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        record("fp16_acc16", lambda: ah @ bh)
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_tf32 = True
    record("tf32", lambda: a32 @ b32)
    torch.backends.cuda.matmul.allow_tf32 = False
    a8 = ab.to(torch.float8_e4m3fn)
    b8 = bb.t().contiguous().to(torch.float8_e4m3fn).t()  # column-major second operand
    one = torch.ones((), device=DEV)
    record("fp8_e4m3", lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    ai = torch.randint(-127, 127, (N, N), device=DEV, dtype=torch.int8)
    bi = torch.randint(-127, 127, (N, N), device=DEV, dtype=torch.int8).t().contiguous().t()
    record("int8", lambda: torch._int_mm(ai, bi))
    del a32, b32, ab, bb, ah, bh, a8, b8, ai, bi
    torch.cuda.empty_cache()
    m = data["matmul"]
    ratio = (m.get("fp16_acc16", {}).get("tflops", 0) / m["fp16_acc32"]["tflops"]
             if m.get("fp16_acc32", {}).get("tflops") else None)
    data["fp16acc16_over_fp16acc32"] = ratio

    # ------------------------------------------------------------------ 3. DRAM / L2 bandwidth
    big = torch.empty(2**31 // 4, device=DEV)           # 2 GiB fp32
    big.normal_()
    dst = torch.empty_like(big)
    ms, clk = sustained(lambda: dst.copy_(big))
    data["memory"]["copy_2GiB_GBps"] = round(2 * big.numel() * 4 / (ms * 1e-3) / 1e9, 1)
    ms, clk = sustained(lambda: big.sum())
    data["memory"]["read_2GiB_GBps"] = round(big.numel() * 4 / (ms * 1e-3) / 1e9, 1)
    del big, dst
    small = torch.randn(2**23 // 4, device=DEV)         # 8 MiB
    sdst = torch.empty_like(small)
    ms, _ = sustained(lambda: sdst.copy_(small), seconds=1.0)
    data["memory"]["copy_8MiB_L2_GBps"] = round(2 * small.numel() * 4 / (ms * 1e-3) / 1e9, 1)
    print("  memory:", data["memory"])
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 4. our shapes (bf16 forward matmul)
    M = 32 * 1024
    for name, (kk, nn_) in {"qkv_384x1152": (384, 1152), "mlp_up_384x2048": (384, 2048),
                            "mlp_down_1024x384": (1024, 384), "head_384x8192": (384, 8192)}.items():
        x = torch.randn(M, kk, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(nn_, kk, device=DEV, dtype=torch.bfloat16)
        ms, clk = sustained(lambda: torch.nn.functional.linear(x, w), seconds=1.0)
        fl = 2 * M * kk * nn_
        bytes_ = 2 * (M * kk + kk * nn_ + M * nn_)
        data["shapes"][name] = {"ms": round(ms, 4), "tflops": round(fl / (ms * 1e-3) / 1e12, 2),
                                "arith_intensity_flop_per_byte": round(fl / bytes_, 1),
                                "implied_GBps_if_bw_bound": round(bytes_ / (ms * 1e-3) / 1e9, 1)}
        print(f"  {name:20s} {data['shapes'][name]}")
        del x, w

    # ------------------------------------------------------------------ 5. host -> device
    h = torch.randn(64 * 2**20 // 4)
    hp = h.pin_memory()
    d = torch.empty(h.numel(), device=DEV)
    ms, _ = sustained(lambda: d.copy_(h), seconds=1.0)
    data["h2d"]["pageable_64MiB_GBps"] = round(h.numel() * 4 / (ms * 1e-3) / 1e9, 2)
    ms, _ = sustained(lambda: d.copy_(hp, non_blocking=True), seconds=1.0)
    data["h2d"]["pinned_64MiB_GBps"] = round(h.numel() * 4 / (ms * 1e-3) / 1e9, 2)
    small_h = torch.randint(0, 8192, (32, 1025), dtype=torch.int64)   # = one training batch, 262 KB
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        small_h.clone().pin_memory()
        times.append((time.perf_counter() - t0) * 1e3)
    data["h2d"]["fresh_pin_256KiB_ms_first"] = round(times[0], 3)
    data["h2d"]["fresh_pin_256KiB_ms_median_rest"] = round(statistics.median(times[1:]), 3)
    print("  h2d:", data["h2d"])

    # ------------------------------------------------------------------ verdicts
    bf = m.get("bf16", {})
    rate = bf.get("flop_per_clk_per_sm")
    if ratio is None:
        v112 = "inconclusive (no fp16-accumulation switch in this torch)"
    elif ratio >= 1.7:
        v112 = f"REFUTED: GeForce-style half rate (fp16acc16/acc32 = {ratio:.2f})"
    elif ratio <= 1.3 and (rate is None or rate >= 800):
        v112 = f"VERIFIED: full rate (ratio {ratio:.2f}, bf16 {rate} FLOP/clk/SM)"
    else:
        v112 = f"inconclusive (ratio {ratio:.2f}, bf16 {rate} FLOP/clk/SM)"
    bw = max(data["memory"]["copy_2GiB_GBps"], data["memory"]["read_2GiB_GBps"])
    l2 = props.get("L2_cache_size")
    verdicts = {
        "C112": v112,
        "C113": f"{'VERIFIED' if bw >= 225 else 'REFUTED'}: best {bw} GB/s of 300 GB/s spec",
        "C114": (f"{'VERIFIED' if l2 == 24 * 2**20 else 'REFUTED'}: L2 = {l2 / 2**20:.1f} MiB" if l2 else
                 "inconclusive (torch does not expose L2_cache_size)"),
        "C129": ("VERIFIED: one CUDA device (the NPU is not visible to CUDA in WSL)" if props["cuda_device_count"] == 1
                 else f"REFUTED: {props['cuda_device_count']} CUDA devices"),
    }
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, verdicts, data, t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
