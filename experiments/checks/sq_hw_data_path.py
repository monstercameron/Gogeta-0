"""SQ78 (hardware research): the batch path. Is TokenData.batch (numpy gather + int64 + pin_memory + H2D) a cost
on this unified-memory laptop under WSL, and is a GPU-resident dataset bit-identical? (claim C124)

    ~/.venvs/gogeta/bin/python experiments/checks/sq_hw_data_path.py          # GPU, ~1-2 min; needs the dataset

GPU check: run it only when `bash experiments/launch/status.sh` shows the GPU slot free AND no GPU process.
Source: notes/hardware-spark.md, proposal "data path / pin_memory". Patch: notes/patches/gpu-resident-data.patch.

Background (docs, not run): on integrated GPUs "any copy between host and device memory is superfluous" (CUDA
Programming Guide), but under WSL the Linux VM's RAM is not the GPU's carve-out, pinned memory "availability is
limited" (CUDA on WSL guide), and cudaMallocHost memory is not L2-cacheable on RTX Spark (porting guide).
The training set (262M uint16 tokens, ~525 MB) fits easily in the GPU's 96 GiB carve-out.

Arms, B=32, T=1024, dataset code-py-v1, seed 0, 200 batches each:
  current    gogeta.train.TokenData(...).batch(B, "cuda")  (exactly what training does)
  no_pin     the same without .pin_memory()
  gpu_gather the same CPU generator draws the same start indices; tokens are gathered from a uint16 copy of
             train.bin on the GPU (what the patch does)
All three must produce identical (x, y) for the same seed.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import save, seed_all  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gogeta.train import TokenData  # noqa: E402

CHECK_ID, FILE_STEM = "SQ78", "sq_hw_data_path"
CLAIMS = ["C124"]
# ---------------------------------------------------------------------------- written before running
PREDICTION = (
    "current: 0.5-3 ms per batch (numpy stack of 32 x 1025 tokens + int64 + pinned copy), i.e. <= 2% of a 156 ms E0 "
    "step; no_pin within +-30% of current; gpu_gather <= 0.3 ms per batch (it is a CPU randint + one gather kernel). "
    "All three arms give bit-identical batches. Uploading train.bin to the GPU (as int32) takes < 3 s and ~1.05 GB."
)
CRITERION = (
    "C124 (the data path is not a bottleneck: < 3% of an E0 step) is VERIFIED if current <= 4.7 ms per batch "
    "(3% of the 156 ms E0 step at ~210k tok/s), REFUTED if > 4.7 ms. The GPU-resident patch is safe only if "
    "gpu_gather is bit-identical to current on all 200 batches."
)
SEED = 0
B, T, N = 32, 1024, 200
DATASET = "code-py-v1"


def main():
    t_start = time.time()
    seed_all(SEED)
    data = {"B": B, "T": T, "batches": N}

    def timed(make_batch):
        out, ts = [], []
        for _ in range(N):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x, y = make_batch()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
            out.append((x, y))
        return out, ts

    # current
    td = TokenData(DATASET, T, SEED)
    cur, t_cur = timed(lambda: td.batch(B, "cuda"))

    # no pin
    td2 = TokenData(DATASET, T, SEED)

    def nopin():
        ix = torch.randint(len(td2.train) - T - 1, (B,), generator=td2.g).tolist()
        buf = np.stack([td2.train[i:i + T + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(buf).to("cuda")
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()
    nop, t_nop = timed(nopin)

    # gpu-resident gather, same generator stream
    td3 = TokenData(DATASET, T, SEED)
    t0 = time.perf_counter()
    gpu_tokens = torch.from_numpy(np.asarray(td3.train).astype(np.int32)).to("cuda")   # int32 holds uint16 exactly
    torch.cuda.synchronize()
    upload_s = time.perf_counter() - t0
    offs = torch.arange(T + 1, device="cuda")

    def gather():
        ix = torch.randint(len(td3.train) - T - 1, (B,), generator=td3.g).to("cuda", non_blocking=True)
        t = gpu_tokens[ix[:, None] + offs].long()
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()
    gat, t_gat = timed(gather)

    same_nop = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(cur, nop))
    same_gat = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(cur, gat))
    med = lambda ts: round(float(np.median(ts[5:])), 4)  # noqa: E731
    data.update({"current_ms": med(t_cur), "no_pin_ms": med(t_nop), "gpu_gather_ms": med(t_gat),
                 "current_first_ms": round(t_cur[0], 3), "upload_s": round(upload_s, 2),
                 "gpu_tokens_gb": round(gpu_tokens.numel() * 4 / 1e9, 3),
                 "identical_no_pin": same_nop, "identical_gpu_gather": same_gat,
                 "share_of_e0_step_pct": round(med(t_cur) / 156.0 * 100, 2)})
    print(" ", data, flush=True)
    v = "VERIFIED" if data["current_ms"] <= 4.7 else "REFUTED"
    verdicts = {"C124": f"{v}: current {data['current_ms']} ms/batch ({data['share_of_e0_step_pct']}% of E0 step)",
                "gpu_resident_patch_safe": bool(same_gat)}
    save(CHECK_ID, CLAIMS, PREDICTION, CRITERION, verdicts, data, t_start, filename=FILE_STEM)


if __name__ == "__main__":
    main()
