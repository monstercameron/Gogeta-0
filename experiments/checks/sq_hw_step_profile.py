"""SQ75 (hardware research): where does a compiled training step's GPU time go? Tests the roofline verdict in
notes/hardware-spark.md ("bandwidth/elementwise-heavy, not compute- or launch-bound") (claim C119).

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_step_profile.py          # GPU, ~5-8 min (3 compiles)

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.

For E0, E1 and E2 (r fixed at 4), B=32, T=1024, torch.compile default mode, seeded random tokens on the GPU, the
train.py step (autocast fwd, bwd, clip, fused AdamW). After 8 warm-up steps, torch.profiler records 3 steps.
Kernels are bucketed by name (heuristic; the top-30 kernel names are saved so the buckets can be re-checked):
  gemm        cuBLAS/cuBLASLt/CUTLASS/nvjet matmuls
  attention   flash / mem-efficient / cuDNN SDPA kernels
  fla_gdn     fla Triton kernels (chunk_*, *delta*, wy_*, recompute_*, solve_tril, ...)
  conv        cuDNN / depthwise conv kernels (GDN short conv)
  optimizer   fused AdamW / multi_tensor_apply / clip-norm reductions
  inductor    inductor-generated Triton kernels (triton_poi_/triton_per_/triton_red_: norms, RoPE, SwiGLU, adds, CE)
  copy        memcpy / memset
  other       everything else
Reports per arm: wall ms/step, GPU-busy ms/step (sum of kernel time), busy fraction, kernel launches per step,
and the share of each bucket.
"""
import json
import pathlib
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, save, seed_all  # noqa: E402

import torch  # noqa: E402

CHECK_ID, FILE_STEM = "SQ75", "sq_hw_step_profile"
CLAIMS = ["C119"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "E0: GPU busy >= 90% of wall time (not launch-bound); gemm 35-55% of GPU time, attention 5-15%, inductor "
    "(elementwise/norm/loss) 25-45%, optimizer 2-6%; 400-1500 kernel launches per step. E1: SWA masked attention "
    "and fla GDN kernels together 30-50%, conv 3-12%, gemm 25-40%. E2 (r=4): like E1 with a larger attention share."
)
CRITERION = (
    "C119 (E0's compiled step is bandwidth/elementwise-heavy, not compute- or launch-bound) is VERIFIED if E0's "
    "non-gemm share of GPU time >= 45% AND busy/wall >= 0.9; REFUTED if gemm >= 70% (compute-bound) or busy/wall < "
    "0.75 (launch/host-bound)."
)
SEED = 0
B, T, R = 32, 1024, 4
WARM, PROF = 8, 3
CONFIGS = {"E0": "configs/e0_dense.json", "E1": "configs/e1_hybrid.json", "E2": "configs/e2_looped.json"}
BUCKETS = [
    ("copy", r"memcpy|memset|Memcpy|Memset"),
    ("inductor", r"^triton_(poi|per|red|tem)_"),
    ("optimizer", r"adam|Adam|multi_tensor|foreach|lpnorm"),
    ("attention", r"flash|fmha|attention|Attention|efficient|cudnn_sdpa|sdpa"),
    ("fla_gdn", r"chunk_|delta|wy_|recompute|solve_tril|fused_recurrent|l2norm|kkt|_fwd_kernel|_bwd_kernel"),
    ("conv", r"conv|Conv|cudnn"),
    ("gemm", r"gemm|Gemm|GEMM|nvjet|cutlass|xmma|sm\d+_|cublas|Kernel2|matmul"),
]


def bucket(name):
    for b, pat in BUCKETS:
        if re.search(pat, name):
            return b
    return "other"


def build(exp):
    from gogeta.model import FusionDance, ModelConfig
    cfg = json.loads((ROOT / CONFIGS[exp]).read_text())
    return FusionDance(ModelConfig(**{**cfg["model"], "vocab_size": 8192, "max_seq_len": T})).cuda(), cfg["train"]


def run_arm(exp):
    torch.manual_seed(SEED)
    model, tc = build(exp)
    loops = R if len(model.core) else None
    fwd = torch.compile(model)
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
                             {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.95), fused=True)
    g = torch.Generator(device="cuda").manual_seed(SEED)
    data = [torch.randint(0, 8192, (B, T + 1), device="cuda", generator=g) for _ in range(2)]

    def step(i):
        b = data[i % 2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loops=loops)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for i in range(WARM):
        step(i)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for i in range(PROF):
            step(i)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / PROF * 1e3
    per_bucket, per_kernel, n_kernels = defaultdict(float), defaultdict(float), 0
    from torch.autograd import DeviceType
    for ev in prof.events():
        if ev.device_type != DeviceType.CUDA:
            continue
        dev_t = ev.time_range.elapsed_us()
        if dev_t > 0:
            per_bucket[bucket(ev.name)] += dev_t / 1e3          # us -> ms
            per_kernel[ev.name] += dev_t / 1e3
            n_kernels += 1
    busy = sum(per_bucket.values()) / PROF
    share = {k: round(v / PROF / busy, 4) for k, v in sorted(per_bucket.items(), key=lambda kv: -kv[1])} if busy else {}
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:30]
    res = {"exp": exp, "loops": loops, "wall_ms_per_step": round(wall, 2), "gpu_busy_ms_per_step": round(busy, 2),
           "busy_fraction": round(busy / wall, 3) if wall else None, "kernels_per_step": n_kernels // PROF,
           "share": share, "top30": [(n[:120], round(t / PROF, 3), bucket(n)) for n, t in top],
           "tok_per_s_under_profiler": round(B * T / (wall / 1e3))}
    del model, fwd, opt
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return res


def main():
    t_start = time.time()
    seed_all(SEED)
    rows = {}
    for exp in CONFIGS:
        try:
            rows[exp] = run_arm(exp)
        except Exception as e:
            rows[exp] = {"exp": exp, "error": f"{type(e).__name__}: {e}"[:400]}
        r = rows[exp]
        print(f"  {exp}: " + json.dumps({k: r.get(k) for k in ("wall_ms_per_step", "gpu_busy_ms_per_step",
                                                                 "busy_fraction", "kernels_per_step", "share",
                                                                 "error")}), flush=True)
    e0 = rows.get("E0", {})
    if "share" in e0 and e0["share"]:
        gemm = e0["share"].get("gemm", 0.0)
        bf = e0["busy_fraction"]
        if (1 - gemm) >= 0.45 and bf >= 0.9:
            v = f"VERIFIED: E0 non-gemm {1 - gemm:.2f}, busy/wall {bf:.2f}"
        elif gemm >= 0.70 or bf < 0.75:
            v = f"REFUTED: E0 gemm {gemm:.2f}, busy/wall {bf:.2f}"
        else:
            v = f"inconclusive: E0 gemm {gemm:.2f}, busy/wall {bf:.2f}"
    else:
        v = "inconclusive: E0 profile failed"
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, {"C119": v}, {"batch": B, "seq": T, "e2_loops": R, "arms": rows},
         t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
