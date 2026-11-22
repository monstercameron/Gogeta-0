# refs/12-hardware: sources for the training GPU (NVIDIA RTX Spark N1X)

Downloaded 2026-11-22 by the hardware-research subagent (download approved by the human for this task).
PDFs are git-ignored (`refs/**/*.pdf`); this file is the record. Re-download from the URLs below
and check the sha256. The analysis that uses these is `notes/hardware-spark.md`.

The five `rtx-spark-porting-guide-*.pdf` files and `rtx-spark-specifications.pdf` are HTML pages
printed to PDF with headless Chrome (`--print-to-pdf`), since NVIDIA publishes them only as HTML.
Their hashes are of our print, not of a file NVIDIA serves. All other files are NVIDIA's own PDFs.

**Scope warning.** Only the RTX Spark spec page, the RTX Spark porting guide and our own
`nvidia-smi` / torch readings are about the **N1X**. The GB10 whitepaper and the DGX Spark guide
are about **GB10 / DGX Spark**, a related part. Don't read GB10 numbers as N1X facts without
saying so (see the confidence labels in `notes/hardware-spark.md`).

| File | Title (publisher, version/date) | URL | Size (bytes) | sha256 | Why it matters |
|---|---|---|---|---|---|
| `rtx-spark-specifications.pdf` | RTX Spark N1X Specifications (NVIDIA product page, printed 2026-11-22) | https://www.nvidia.com/en-us/products/rtx-spark/specifications/ | 403,113 | `033db7c23e52cfe2a46b5d7ecb26c1e36f9158846de67d339cc10edcfedd83a5` | **N1X-specific.** 6144-core laptop SKU: TDP 45-80 W, "1 Petaflop" AI, up to 128 GB LPDDR5X, 256-bit, 5th-gen tensor cores, CUDA capability 12.1, NVIDIA NPU |
| `rtx-spark-porting-guide-system-overview.pdf` | RTX Spark Porting Guide 0.1.0: System Overview (NVIDIA docs) | https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/overview.html | 229,288 | `efcd52b595cd412f92ab0a10d476781b50ae395f08d8aee35de7651ab459f8d6` | **N1X-specific.** 256-bit memory at **300 GB/s**, fully coherent; CPU 2 x (5 X925 + 5 A725) |
| `rtx-spark-porting-guide-gpu-identification.pdf` | RTX Spark Porting Guide 0.1.0: GPU Identification | https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/gpu-identification.html | 210,785 | `ea8cfcb6d74bc11bb099a5a703ead22cd738aa7837e430924398cb3d1a460ad8` | PCI device IDs: `0x2E03` = 6144-core laptop (ours, per `nvidia-smi -q`), `0x2E13` desktop, `0x2E06` 5120-core laptop |
| `rtx-spark-porting-guide-unified-memory.pdf` | RTX Spark Porting Guide 0.1.0: Unified Memory Architecture | https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/uma/index.html | 135,990 | `0b3bd7eb52eb1e3b7ba4a8f51a8c0f131356739a3ab6a35177a7011fefb38880` | Windows splits DRAM into a GPU **carveout** (shown as dedicated memory), a shared segment and CPU-only memory; coherence maintained by the platform |
| `rtx-spark-porting-guide-uma-cuda.pdf` | RTX Spark Porting Guide 0.1.0: UMA with CUDA | https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/uma/umacuda.html | 560,391 | `3909dc3d45f7a03548c854d7c00fac8bc49369123d766b52b3100b6d687d8d57` | Use `cudaMalloc` for GPU-only data; `cudaMallocHost` memory is coherent but **not L2-cacheable** on the iGPU; avoid `cudaMallocManaged`; don't allocate the whole budget |
| `rtx-spark-porting-guide-power.pdf` | RTX Spark Porting Guide 0.1.0: Power | https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/power/index.html | 281,631 | `10284557d77639b655d126f166af8a6b150205193316ea3a2ece1b02a05f24c0` | P0 = AC/best performance, P4 = balanced, P8 = battery; Windows power mode selects the regime (no TDP or clock numbers) |
| `nvidia-gb10-superchip-whitepaper.pdf` | NVIDIA GB10 Superchip White Paper, WP-12718-001_v02 (NVIDIA, 2026-04-02) | https://www.nvidia.com/content/dam/en-zz/Solutions/dgx-spark/nvidia-gb10-superchip-whitepaper.pdf | 755,407 | `241fb42e4206dd99cdbe1ee2e1267d009af1eb70219cacdf0fbe500d555782b9` | **GB10-only.** 48 SMs, 128 CUDA cores/SM, 2.4 GHz boost, **24 MiB L2**, tensor FP4/FP8/FP16/BF16, LPDDR5X 8533 Mbps, 140 W, NVLink-C2C with AMBA CHI coherency, "1 PFLOP NVFP4 Sparse", CPU caches |
| `dgx-spark-user-guide.pdf` | DGX Spark User Guide (NVIDIA docs, dated 2026-11-18) | https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf | 18,310,280 | `610f72bf0ded6640fc01c49286b960214b4cd152a14a67c67de8e785f5e3baa0` | **GB10-only.** 273 GB/s, 16-channel LPDDR5X 8533, 140 W TDP, UMA memory-reporting notes (nvidia-smi "Memory-Usage: Not Supported" on Linux; ours under WDDM shows a 96 GiB carveout instead) |
| `nvidia-rtx-blackwell-gpu-architecture.pdf` | NVIDIA RTX Blackwell GPU Architecture whitepaper, V1.1 (NVIDIA) | https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf | 8,283,392 | `906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa` | The sm_120 SM: 128 cores, 4 tensor cores, 256 KB register file, 128 KB L1/shared per SM; per-precision peak tables (GeForce parts run BF16 with FP32 accumulate at half the FP16-accumulate rate) used to bound N1X peaks |
| `cuda-blackwell-tuning-guide.pdf` | Blackwell Tuning Guide, Release 13.5 (NVIDIA CUDA docs) | https://docs.nvidia.com/cuda/pdf/Blackwell_Tuning_Guide.pdf | 157,052 | `74f9dde10f048281cdb8fdd2c2b4f45b260f375d38cc1f754e03127bc84e5856` | CC 12.0: 48 warps/SM, 64K registers/SM, 128 KB shared capacity per SM, **99 KB max per block**; clusters supported (portable size 8) |
| `cuda-blackwell-compatibility-guide.pdf` | Blackwell Compatibility Guide, Release 13.5 | https://docs.nvidia.com/cuda/pdf/Blackwell_Compatibility_Guide.pdf | 165,197 | `7deef841609c19c31fefc86dab3e1078dae4da18c27fd3d717f30f8ae849c493` | Cubins run on the same major CC with an equal or higher minor, so **sm_120 builds run on 12.1**; PTX JIT fallback |
| `cuda-programming-guide-13.5.pdf` | CUDA C++ Programming Guide, Release 13.4 (legacy edition, served at the 13.5 URL on 2026-11-22) | https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf | 4,660,774 | `d1e2b16bc6ed17df22e0fe0043fea6b095d982e7f7471d89121706ec5b10e71e` | CC table: 1536 threads / 48 warps / 24 blocks per SM for 12.x, 100 KB shared per SM, 99 KB per block; "arch-specific (`a`) code runs only on the exact CC"; integrated GPUs: copies are superfluous, use mapped page-locked memory |
| `ptx-isa-9.5.pdf` | PTX ISA, Release 9.5 (NVIDIA, Last-Modified 2026-11-04) | https://docs.nvidia.com/cuda/pdf/ptx_isa_9.5.pdf | 31,105,028 | `e2934fdad556d003e43a756dca896f87bfcff0c9fff3578722161dbbd1bde41d` | `wgmma` requires sm_90a; `tcgen05` / Tensor Memory only on sm_100a/101a(110a)/103a families, **not sm_12x**; `mma.sync` block-scaled `kind::mxf4nvf4` on sm_120a/**sm_121a**; `cp.async.bulk.tensor` (TMA) on sm_120a with restrictions; sm_121/121a/121f targets since PTX 8.8 |
| `cuda-on-wsl-user-guide.pdf` | CUDA on WSL User Guide, Release 13.5 | https://docs.nvidia.com/cuda/pdf/CUDA_on_WSL_User_Guide.pdf | 400,419 | `11499cf5c4f2b042ace91e696ec4c2edcbb5241c7d278d2f5f1ea8ffff514aa4` | WSL limits: no full managed memory, **pinned memory availability is limited**, no concurrent CPU/GPU access, NVML queries incomplete |

Total: 14 files, 65,658,747 bytes (62.6 MiB).

## Local observations (not downloads), `observed/`
Text files, committed with the notes (not git-ignored). Device UUIDs, PDI and subsystem IDs are redacted, and the
desktop process list is omitted.
- `observed/nvidia-smi-q-2026-11-22.txt`: `nvidia-smi -q` on Windows (driver 621.24, CUDA UMD 13.5): product name,
  PCI device `0x2E03`, "GPU C2C Mode: Enabled", 98,304 MiB carve-out, max clock 3,003 MHz, T.Limit margin, and the
  second device "NVIDIA NPU" (NPU3, MCDM, DLA).
- `observed/nvidia-smi-sample-2026-11-22-k7.csv`: 2 s samples (temperature, utilization, SM clock, power, P-state,
  throttle reasons) from 17:42:38 to 17:46:00, covering K7 resume-campaign item 3 (E10, lr 6e-3, 20M tokens; ran 17:42:22-17:44:33) and the
  60 s gap after it (item 4, E1 lr 1e-3, started 17:45:37). Read-only `nvidia-smi` queries; no GPU compute.

## Not downloaded (read online only)
- NVIDIA developer-forum threads on GB10 measurements (community numbers, not vendor specs):
  https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993 (mmapeak table),
  StorageReview DGX Spark review (MAMF ~99.8 TFLOP/s BF16, ~207.7 TFLOP/s FP8):
  https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops
- Triton sm_121 bring-up thread: https://github.com/triton-lang/triton/issues/10331 (Triton >= 3.6 bundles a Blackwell ptxas that handles sm_121a)
- fla chunk-kernel NaN after a -inf log gate: https://github.com/fla-org/flash-linear-attention/issues/1758 (not our bug, see notes)
- PyTorch MXFP8 `_scaled_mm` layout limits on consumer Blackwell: https://github.com/pytorch/pytorch/issues/198126
- flex_attention OutOfResources on 99 KB-shared-memory GPUs: https://github.com/pytorch/pytorch/issues/143832
