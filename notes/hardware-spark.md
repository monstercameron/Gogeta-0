# The training GPU: NVIDIA RTX Spark N1X (and how it relates to DGX Spark / GB10)

Hardware research side quest, 2026-11-22 (subagent; CPU-only, `nvidia-smi` reads only, no GPU compute: a K7
campaign was running). References: `refs/12-hardware/SOURCES.md` (14 NVIDIA documents, sha256-recorded) and the
local observations in `refs/12-hardware/observed/`. Ledger rows C110-C129 and side quests SQ70-SQ83 are proposed
at the end of `experiments/claims.md` and `experiments/side-quests.md`. Nothing below was benchmarked on the GPU
by this research. Numbers are labelled **measured** (earlier, in the lab notebook), **observed** (passive
`nvidia-smi` sampling today, raw files on disk), **reported by <source>**, **computed** or **predicted**.

## 1. Verified spec table

Confidence labels:
- **confirmed-N1X**: an NVIDIA document about the RTX Spark / N1X, or our own reading on this laptop.
- **CC 12.x doc**: NVIDIA documents a property of compute capability 12.x, which our GPU reports (12.1).
- **GB10-only**: documented only for GB10 / DGX Spark.
- **inferred**: our arithmetic from the above; the check that would settle it is named.

| Property | N1X (this laptop) | Label | GB10 / DGX Spark, for comparison | Source |
|---|---|---|---|---|
| Identity | "NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)", PCI device `0x2E03` = 6144-core **laptop** SKU; brand "GeForce"; driver model WDDM; "GPU C2C Mode: Enabled" | confirmed-N1X | "NVIDIA GB10" | `nvidia-smi -q` (observed, `refs/12-hardware/observed/nvidia-smi-q-2026-11-22.txt`); porting guide, GPU identification |
| SMs, CUDA cores | 48 SMs (measured, torch); 6,144 CUDA cores, i.e. 128 per SM | confirmed-N1X | 48 SMs, 6,144 cores (128/SM), 4 GPCs, 2.4 GHz boost | spec page; GB10 whitepaper Table 4 |
| Tensor cores | 5th generation | confirmed-N1X | "48 (1 per SM)", 5th gen | spec page; GB10 WP Table 4. (The GeForce GB202 whitepaper counts 4 per SM: counting conventions differ, so don't compare counts.) |
| Tensor precisions | FP4/NVFP4/MXFP4, FP6, FP8 (E4M3, E5M2), MXFP8 (block-scaled `mma.sync` `kind::mxf8f6f4` / `kind::mxf4nvf4` on **sm_121a**), FP16, BF16, TF32, INT8 | CC 12.x doc | Tensor "FP4, FP8, FP16, BF16" (TF32 not listed) | PTX ISA 9.5 (`mma` target notes); CUDA Programming Guide CC 12.0 section; GB10 WP |
| AI peak (FP4, sparse) | "1 Petaflop" (precision and sparsity not stated on the N1X page) | confirmed-N1X | "1 PFLOP NVFP4 Sparse" | spec page; GB10 WP Table 2 |
| Implied boost clock of that figure | 2.54 GHz (= 1e15 / (8,192 sparse FP4 FLOP/clk/SM x 48)) | inferred | same | computed from the GeForce Blackwell per-SM FP4 rate (RTX Blackwell WP Table 3: 5090 1,676 dense TFLOPS at 170 SMs x 2.407 GHz = 4,096 FLOP/clk/SM) |
| **Dense BF16 peak** | not published. **~98 TFLOP/s at the observed 2.0 GHz training clock; 118 at 2.4 GHz; 125 at 2.54 GHz**, if BF16 with FP32 accumulate runs at full rate (1,024 FLOP/clk/SM). If it ran at the GeForce half rate (512), the peak at 2.0 GHz would be 49, below our **measured 72.1** (which would then need >= 2.93 GHz). | inferred (SQ71 settles it) | not published; community MAMF **~99.8 TFLOP/s** (StorageReview), which implies full rate at ~2.03 GHz | computed; RTX Blackwell WP (5090: BF16 fp32-acc 209.5 vs FP16 fp16-acc 419 TFLOPS = GeForce half rate) |
| Dense FP8 peak | ~196 TFLOP/s at 2.0 GHz, ~250 at 2.54 GHz (2x BF16) | inferred (SQ71) | community: MAMF ~207.7, CUTLASS 188 TFLOP/s | computed; forum posts (community, not vendor) |
| Our measured 72.1 TFLOP/s (bf16 4096^2) | = 73% of the full-rate peak at the observed 2.0 GHz; 4096^2 is smaller than a peak-finding sweep and the clock during that run is unknown | measured / inferred | | notebook 12:00 |
| TF32 | supported (CC 12.0 doc); rate unknown, expected 0.25-0.5x BF16 | CC 12.x doc / inferred | mmapeak forum table 53.3 (its BF16 row, 212.9, conflicts with MAMF, so that table is unreliable) | CUDA guide; forum |
| Memory | unified LPDDR5X, up to 128 GB, **256-bit, 300 GB/s**, "fully coherent". nvidia-smi reports the memory clock 9,400 MHz (on the NPU device), and 9,400 MT/s x 32 B = 300.8 GB/s | confirmed-N1X | 128 GB LPDDR5X-**8533**, 256-bit, **273 GB/s**; community sustained reads ~231-235 GB/s (~86%) | spec page; porting guide overview; observed; DGX Spark user guide |
| How Windows/WSL see it | Windows reserves a GPU **carve-out** shown as dedicated memory: `nvidia-smi` FB total **98,304 MiB (96 GiB)** (observed); torch reports **110.5 GB** usable (measured; probably carve-out plus part of the shared segment, inferred); the WSL VM sees **~14 GB** RAM (observed `free -g`). The carve-out is DRAM, not VRAM; allocations spill to the shared segment (smaller pages, slower) once it is full | confirmed-N1X + observed | DGX OS (Linux): no carve-out; `nvidia-smi` shows "Memory-Usage: Not Supported" | porting guide UMA pages; DGX Spark user guide 2.2.2 |
| L2 cache | not published | inferred: 24 MiB (SQ71 reads `L2_cache_size`) | **24 MiB** | GB10 WP Table 4 |
| L1 / shared memory per SM | 128 KB unified L1/shared; shared carve-out 0-100 KB; **max 99 KB per block** (opt-in) | CC 12.x doc | same (CC 12.x) | RTX Blackwell WP (SM); Blackwell tuning guide 1.4.1.1; CUDA guide Table 27 |
| Register file | 64K x 32-bit = 256 KB per SM (12 MiB total, computed); <= 255 per thread | CC 12.x doc | same | tuning guide; CUDA guide |
| Occupancy limits | 1,536 threads, 48 warps, 24 blocks per SM | CC 12.x doc | same | CUDA guide Table 27 (the tuning guide says 32 blocks; the guide's table says 24) |
| TMA (`cp.async.bulk.tensor`) | **yes**, with sm_120a restrictions (no sub-byte types / `.swizzle_atomicity` for `.shared::cluster.global`) | CC 12.x doc | same | PTX ISA 9.5 |
| `wgmma` | **no** (requires sm_90a) | CC 12.x doc | no | PTX ISA 9.5, 9.7.17 target notes |
| `tcgen05` / Tensor Memory | **no** (sm_100a/101a(110a)/103a/107a families only) | CC 12.x doc | no | PTX ISA 9.5, 9.7.18 target notes |
| Clusters / distributed shared memory | yes (portable cluster size 8) | CC 12.x doc | yes | tuning guide 1.4.1.2 |
| Binary compatibility | sm_120 cubins run on 12.1 (same major, higher minor); **arch-specific `sm_120a` code does not** (exact-CC only; needs `sm_121a` or the `sm_120f` family target) | CC 12.x doc; observed (our sm_120 torch trains) | same | Blackwell compatibility guide 1.2; CUDA guide 20.1.1 |
| CPU-GPU link | NVLink-C2C ("GPU C2C Mode: Enabled"); coherent | confirmed-N1X (observed + porting guide) | NVLink-C2C, AMBA CHI-G coherency, "about five times" a comparable PCIe link (no GB/s figure) | nvidia-smi; GB10 WP |
| CPU | 20 cores: 2 clusters x (5 Cortex-X925 + 5 Cortex-A725), Arm v9.2 | confirmed-N1X | same core mix; X925 up to 4 GHz, A725 2.8 GHz; L3 16 + 8 MiB | porting guide; GB10 WP Table 3 |
| Power | laptop TDP **45-80 W** (desktop N1X 140 W). Observed under our training: **35-56 W (median 43 W), SM clock 1.8-2.2 GHz (median 2.0), 65-78 C**, P0. Max clock reported 3,003 MHz | confirmed-N1X + observed | GB10 SoC 140 W (240 W PSU) | spec page; `refs/12-hardware/observed/nvidia-smi-sample-2026-11-22-k7.csv`; DGX user guide |
| Thermal limit | not published; `nvidia-smi` "GPU Current T.Limit Temp 39 C" at 54 C idle, i.e. ~93 C max operating (inferred: T.Limit is the margin to the limit) | observed / inferred | operating ambient 5-30 C | nvidia-smi -q |
| Power modes | P0 = AC / best performance, P4 = balanced, P8 = battery; chosen by the Windows power mode | confirmed-N1X | n/a | porting guide, power |
| NPU | "NVIDIA NPU" on laptop SKUs (not on the desktop SKU). `nvidia-smi`: architecture "NPU3", brand "NPU", driver model **MCDM**, UUID prefix `DLA-`, engine clock up to 1,562 MHz, shares the LPDDR5X. An NVDLA-family inference engine exposed to Windows ML-style APIs, **not a CUDA device**; no published TOPS. Useless for our training | confirmed-N1X (exists) / inferred (what it is; C129, SQ71) | none | spec page; nvidia-smi -q |

**N1X vs GB10 in one line:** same 48-SM / 6,144-core Blackwell (CC 12.1) and 20-core CPU, but the laptop N1X has
**faster memory (9,400 vs 8,533 MT/s: 300 vs 273 GB/s)** and **about a third to a half of the power budget (45-80
W vs 140 W)**. GB10 measurements (99.8 TFLOP/s BF16) are an upper bound for us, not a prediction.

## 2. Software support on sm_121 / aarch64 (what matters for our stack)

| Component | Status | Evidence |
|---|---|---|
| torch 2.14.1+cu130 | arch list tops out at sm_120; runs on 12.1 through binary compatibility (we train with it). Inductor has an **sm_12x flex_attention default config** (bf16, head_dim 64: 128x64 tiles, 2 stages, 4 warps) sized for the 99 KB/block limit; `allow_fp16_accumulation` exists | measured (notebook); read in `torch/_inductor/heuristics/template/triton.py` |
| Triton 3.8.0 | works (all compiled runs). sm_121a needs Triton >= 3.6 (bundled Blackwell ptxas); `TRITON_PTXAS_PATH` does not apply to Blackwell targets (`TRITON_PTXAS_BLACKWELL_PATH` does). Triton on sm_12x uses `mma.sync` (no wgmma/tcgen05) | triton-lang/triton#10331 (community thread, one maintainer comment) |
| fla 0.6.0 chunk GDN | verified on this GPU (tests/test_gdn_kernels.py: fp32 2.2e-4, bf16 8.0e-4) | notebook 12:05 |
| **fla fused_recurrent GDN "NaN" (C2)** | **very likely a call-site bug in our tests, not a GPU problem.** fla 0.6.0's signature is `fused_recurrent_gated_delta_rule(q, k, v, g=None, gk=None, gv=None, beta=None, scale=...)`; our test calls it `(q, k, v, g, beta, scale=)`, so `beta` is read as `gk`, a per-key log decay of shape [B,T,H,K], out of bounds of a [B,T,H] tensor, with `exp()` applied, and the real beta defaults to ones. `chunk_gated_delta_rule`'s 5th parameter *is* beta, which is why the chunk test passes. `experiments/checks/sq_kda_kernel.py` (SQ50, line ~193) repeats the same positional call for its GDN re-run. fla's own docstring example has the same mistake | read in the installed fla source; check **SQ70**; fix `notes/patches/fla-recurrent-keyword-args.patch` |
| fla on Blackwell | fla registers a Triton global-scratch allocator for major 10/12; its `IS_NVIDIA_SM120` flag is exactly (12, 0), so 12.1 gets no SM120-specific path (FlashQLA rejects 12.1); TMA off unless `FLA_USE_TMA=1`; autotune results cached by default (`FLA_CACHE_RESULTS=1`, Triton cache in `~/.triton/cache`, 519 MB observed) | read in `fla/utils/hardware.py`, `fla/utils/env.py` |
| fla known bug (not ours) | chunk kernels give NaN after a `-inf` log gate (fla-org#1758). Our g = -exp(A_log) * softplus(.) is finite in fp32 unless softplus overflows; keep in mind for any gate that can saturate | GitHub issue |
| SDPA backends | FLASH (PyTorch's built-in FA2-derived kernels, sm_120 build) works for full causal; our SWA passes a dense bool mask, which rules flash out. cuDNN 9.24 is installed; whether its SDPA backend is faster on 12.1 is SQ72 | measured (runs); observed (`torch.backends.cudnn.version()` = 92400) |
| FlashAttention package | not installed. FA3 is Hopper-only (wgmma/TMA sm_90a); FA2 builds for sm_121 reportedly need `TORCH_CUDA_ARCH_LIST="12.0"` and ~75 min on ARM (community). Not needed: SDPA flash + flex cover us | NVIDIA forum / community posts |
| torchao (FP8 training) | not installed (install needs the human). Consumer Blackwell `_scaled_mm` MXFP8 supports only the TN layout, so dgrad/wgrad GEMMs fail (pytorch#198126); per-tensor/rowwise FP8 status on sm_12x unverified | GitHub issue |
| WSL2 | full managed memory unsupported; **pinned memory "availability is limited"**; NVML queries incomplete (our WSL `nvidia-smi` does report clocks, power, temperature: observed) | CUDA on WSL guide 4.1-4.2 |
| Windows iGPU memory APIs | `cudaMalloc` for GPU data; `cudaMallocHost` is coherent but **not L2-cacheable** on the iGPU; avoid `cudaMallocManaged`; don't allocate the whole budget (the system can stall) | porting guide, UMA with CUDA |
| CUDA graphs under WDDM/WSL | unknown here; SQ74's `reduce-overhead` arm tests it | |

## 3. Roofline: what limits our workloads?

Inputs: E0 = 96.76 MFLOP/token for training (computed, C90), E1 91, E2 193 (computed, notebook 12:10);
throughputs at B=32, T=1024, compiled, bf16 autocast (measured): E0 ~210k tok/s (E10 in today's K7 logs: 201-204k),
E1 ~100k, E2 ~44k.

| | step (computed) | achieved TFLOP/s | vs 72.1 measured matmul | vs ~98 inferred peak at 2.0 GHz | matmul-time lower bound at 72.1 |
|---|---|---|---|---|---|
| E0 | 156 ms | 20.3 | 28% | 21% | 44 ms (28% of the step) |
| E1 | 328 ms | 9.1 | 13% | 9% | 41 ms (13%) |
| E2 (r ~ U{1..7}) | 745 ms | 8.5 | 12% | 9% | 88 ms (12%) |

**Bandwidth side (computed estimate, order of magnitude).** A pre-norm block at d=384 touches each token's
activations ~37 times in d-sized units per forward (norms, qkv, RoPE, attention in/out, residual adds, SwiGLU's
2,048/1,024-wide tensors), about 28 KB/token/layer in bf16; with backward ~3x that. Six layers plus the head and
loss (32768 x 8192 logits written, read, gradient written and read twice) give **~590 KB/token, ~19.4 GB per E0
step, ~78 ms at 250 GB/s** (83% of the 300 GB/s spec; GB10 users sustain ~86% of theirs). The step's arithmetic
intensity is then ~160 FLOP/byte, **below the ridge point** (72 TFLOP/s / 250 GB/s = ~290 FLOP/B; ~390 at the
inferred peak). Matmul time (~44 ms) + traffic (~78 ms) = ~122 ms of the 156 ms step; the rest is kernel
inefficiency, serialisation and launch overhead.

**Launch overhead (computed):** a compiled E0 step is a few hundred to ~1,500 kernels; at 5-10 us each that is
<= 4-10% of 156 ms. **Not launch-bound at B=32.** Batch-1 work (K1's timings, `sample()`, decoding, RL rollouts)
**is** launch-bound: ~100+ kernels per token at a few us each caps eager single-sequence decoding at ~1-2k tok/s.

**E1/E2:** the hybrid spends 2.1x E0's time at slightly fewer FLOPs. The extra time is not matmuls (41 ms
lower bound). Candidates, to be measured by SQ75: the dense-mask SWA (all T^2 scores computed through the
non-flash SDPA kernel), the fla chunk kernels (0.9-1.2 ms forward at B=16, measured; maybe ~7 ms fwd+bwd per
layer per step at B=32, computed), and the GDN layer's extra elementwise work (three depthwise `nn.Conv1d` with
transposes, F.normalize, SiLU gates, per-head RMSNorm). E2's step is close to E1's per-layer-application cost
times its 14 applications at r=4 (328 ms x 14/6 = 765 ms vs 745 measured; computed), so E2 has no separate
bottleneck: it inherits E1's.

**Verdict:** at d=384 our training is **memory-bandwidth / elementwise-bound, not compute-bound and not
launch-bound** (C119, to be confirmed by SQ75's profiler breakdown). Consequences: (1) lower-precision matmuls
(FP8, int8, ternary kernels) can't buy much, because matmuls are under half of the step; (2) the wins are in
fewer passes over memory: fused kernels, not materialising logits, not computing masked-out attention; (3) the
biggest single unknown is the power mode: the GPU drew a median 43 W of its 45-80 W envelope under our load.

## 4. Optimisation proposals, ranked by expected gain per effort

All patches are unified diffs, **not applied**. Each one touches code paths (`gogeta/`, `tools/`, `tests/`), so
applying it is its own tagged version, only while no campaign is queued, and only after its check passes. Every
model-side change is **opt-in** (config field or CLI flag) with the default bit-identical to today: checked on CPU
for the three new `ModelConfig` fields (same state_dict and loss on a tiny SWA/attention model). All six patches
apply individually and together, in either order (`git apply --check`, `py_compile`, `bash -n`).
Gains are **predictions**; "PASS" means the named check's criterion.

| # | Change | Why this hardware | Predicted gain | Check / pass criterion | Patch / flags |
|---|---|---|---|---|---|
| **P1** | Call `fused_recurrent_gated_delta_rule` with `g=`, `beta=` keywords (tests + SQ50's GDN re-run) | Not hardware at all: the "sm_121 NaN" is an argument-order bug, so the decode kernel likely works | **Compatibility:** a working 1-token GDN decode step (C2 -> WRONG TEST), unblocks SQ53 / C19 speculative decoding / C84 RL sampling for existing GDN checkpoints, no new kernel | **SQ70**: positional call broken AND keyword, prefill+step and neg-eig paths within 2e-3 (fp32) / 3e-2 (bf16) | `notes/patches/fla-recurrent-keyword-args.patch` (+ one line in `sq_kda_kernel.py`, see the patch header) |
| **P2** | Measure first: per-kernel step profile + device roofline | Every gain below depends on where the 156/328/745 ms go | none by itself; prevents optimising the wrong thing | **SQ75** (C119: non-gemm >= 45%, busy/wall >= 0.9), **SQ71** (C112-C114, C129) | none |
| **P8** | Temperature-aware campaign gap: `--cool-to 62 --gap 180` ends the gap as soon as the GPU reads < 62 C (min 10 s) | Observed: after a 2-min run at 77 C the GPU read 59.5 C ~15 s later, then idled; in today's K7 resume campaign every item was followed by 64 s before the next started (runs: E10 131 s, E1 176-214 s; campaign state file), so the gap was 25-33% of the campaign's wall time | **~20-25% less wall time for 20M-token campaigns** (K7-type), ~3% for 200M-token runs; zero effect on results | **SQ79** (C126: median time to 65 C <= 30 s) | `notes/patches/campaign-adaptive-gap.patch` |
| **P5** | GDN mixer: fla's fused Triton `causal_conv1d` (SiLU fused, same weights) + `use_qk_l2norm_in_kernel=True` (config `gdn_fused_kernels: true`) | Bandwidth-bound: removes ~6 transposes, 3 separate SiLU passes and 2 normalize passes (fwd+bwd) per GDN layer; cuDNN depthwise-conv backward is a weak spot | **E1/E2 +5-20% tok/s** (wide: depends on SQ75's conv share) | **SQ77** (C122: module fwd diff < 2e-2, grad cos > 0.999, time <= 0.9x) | `notes/patches/gdn-fused-conv-l2norm.patch` |
| **P4** | Fused linear cross-entropy (fla `FusedLinearCrossEntropyLoss`, already installed), 32 chunks of 1,024 tokens (config `fused_ce_chunks: 32`) | Never writes the 537 MB bf16 / 1.07 GB fp32 logits; a 16 MiB chunk fits a 24 MiB L2 | head+loss 0.55-0.85x; **E0 +4-8%**, E1/E2 +2-4%; ~1-2 GB less peak memory | **SQ73** (C117: <= 0.8x compiled path, same loss/grads) | `notes/patches/fused-linear-ce.patch` |
| **P3** | SWA through `flex_attention` with a sliding-window BlockMask (config `swa_impl: "flex"`); falls back to the mask path for variable-length sampling | Computes ~3 of 8 key blocks per query block instead of all T^2 through the non-flash kernel; torch has an sm_12x flex config for the 99 KB/block limit | at T=1024: SWA layer 0.4-0.6x, **E1 +2-6%, E2 +3-8%**; at T >= 4k the SWA layer >= 3x faster with O(n) memory: **fixes ledger C12** for our code | **SQ72** (C115: fwd diff < 2e-2, grad cos > 0.999, <= 0.6x masked; flex scaling exponents <= 1.3) | `notes/patches/swa-flex-attention.patch` |
| **P10** | Windows power mode "Best performance" on AC (and the OEM performance profile, if any) before campaigns | Observed median 43 W in a 45-80 W envelope; the GPU is power-limited, not thermally limited (peak 78 C vs ~93 C limit) | **+0-30% clocks** (unknown: the OEM may cap the GPU regardless) | re-run SQ71 + SQ74's E0 arm after the switch (SQ82) | **human action** (a system setting) |
| **P6** | `--compile-mode max-autotune-no-cudagraphs` (also `reduce-overhead`, `max-autotune`) | Autotuned GEMM tiles for small-K (384) shapes on 48 SMs | +0-8% (E0); CUDA graphs ~0% for training (not launch-bound) but the tool for decoding later | **SQ74** (C118; adopt a mode only if >= 1.08x at matched loss) | `notes/patches/train-perf-flags.patch` (`--compile-mode`) |
| **P7** | `--data-on-gpu`: train.bin on the GPU as int32 (~1.05 GB of 96 GiB), batches gathered there with the same CPU RNG draws | On a UMA laptop under WSL the H2D copy is avoidable work, and WSL limits pinned memory; batches stay bit-identical | <= 1-2% tok/s; removes a WSL pinned-memory failure mode | **SQ78** (C124: current path <= 4.7 ms/batch; gpu_gather bit-identical) | `notes/patches/train-perf-flags.patch` (`--data-on-gpu`) |
| **P9** | Per-run GPU telemetry (`<log>.gpu.csv`, every 10 s) + `TORCHINDUCTOR_CACHE_DIR=$DATA/cache/inductor` | Inductor's default `/tmp/torchinductor_<user>` is emptied at WSL boot, so each restart recompiles cold; power/clock state belongs next to every result | no steady-state change; saves the cold compile after a WSL restart (predicted 1-3 min/run); provenance | SQ79 reads the same fields | `notes/patches/campaign-adaptive-gap.patch` |

**Considered and not recommended now**

| Idea | Verdict | Why (with what would change our mind) |
|---|---|---|
| Bigger batch / sequence packing | **no change** | Data are already packed (random windows of one token stream: no padding). Per-token cost is bandwidth-set, so B=64 is predicted <= +5% (SQ74) while changing the optimisation (LR/batch confound). |
| Fused AdamW / foreach | **already on** | `train.py` uses `AdamW(..., fused=True)`; `clip_grad_norm_` uses foreach by default on CUDA. |
| TF32 / bf16 settings | **keep** | Matmuls run in bf16 under autocast, so `allow_tf32` only touches fp32 ops. Don't enable `allow_fp16_accumulation` (fp16 accumulate in training). Optional hygiene: torch's newer `torch.backends.cuda.matmul.fp32_precision = "tf32"` API. |
| FP8 training (torchao float8) | **not now; needs an install** | Predicted <= 1.0x at d=384: GEMMs are under half the step and per-tensor scaling adds bandwidth; MXFP8 backward layouts fail on consumer Blackwell (pytorch#198126). Worth a look at d >= 1024 (SQ83, needs the human to `pip install torchao`). SQ71 measures the raw FP8 `_scaled_mm` rate without any install. |
| Ternary / int8 kernels for BitLinear (C14) | **training: no; inference: measure** | Training needs bf16 backward (STE). K2's 4-5.6x slowdown was eager; compiled BitLinear should be much closer (C120). An int8 `_int_mm` ternary path is predicted 0.6-1.2x bf16 at our shapes (C121): bandwidth-bound, no win (SQ76). Dedicated ternary kernels (bitnet.cpp-style, BitBLAS) would need installs. |
| `pin_memory` / host-device copies | **irrelevant for speed** | 262 KB per step. The CUDA guide calls copies "superfluous" on integrated GPUs, but under WSL the Linux VM's RAM is not the GPU carve-out, so a copy (or P7) is still needed. |
| fla autotune caching | **already on** | `FLA_CACHE_RESULTS=1` by default; Triton's cache is on the WSL filesystem. SQ77(d) verifies the second process skips autotuning (C123). `FLA_USE_TMA=1`: predicted no gain at K=V=64 (SQ77(e)). |
| FlashAttention 2/3/4 packages | **not needed** | FA3 needs wgmma (sm_90a), FA4 needs tcgen05 (sm_100): neither exists on sm_121. SDPA's flash kernel covers full attention; flex covers SWA. |
| CUDA graphs for training | **not needed** | Not launch-bound at B=32 (C118). Keep `reduce-overhead` for the decode path after P1 (SQ53). |

**Combined prediction** if P3, P4 and P5 all pass: E1/E2-style training **+10-30% tok/s** at T=1024, E0
**+5-12%**; P8 independently shortens short-run campaigns by ~20-25%. P10 could add on top of all of these.
Each must enter one ingredient at a time (SQ81: an E1 A/B per patch at matched tokens), since even opt-in kernels
change bf16 rounding.

## 5. Benchmark scripts and queue

All GPU checks are `experiments/checks/sq_hw_*.py` with `PREDICTION` and `CRITERION` constants written before any
run; they follow `common.py` (seeded, PASS/FAIL verdicts, `experiments/checks/results/sq_hw_*.json`). **None has
been run.** They need registering in a tagged version first (queue item `HW-REG`).

| ID | Script | Claims | GPU min (est.) |
|---|---|---|---|
| SQ70 | `sq_hw_fla_recurrent.py` | C110, C111, C2 | 3 |
| SQ71 | `sq_hw_roofline.py` | C112, C113, C114, C129 | 8 |
| SQ72 | `sq_hw_sdpa_swa.py` | C115, C116, C12 | 10 |
| SQ73 | `sq_hw_loss_head.py` | C117 | 4 |
| SQ74 | `sq_hw_compile_modes.py` | C118 | 15 |
| SQ75 | `sq_hw_step_profile.py` | C119 | 8 |
| SQ76 | `sq_hw_bitlinear_paths.py` | C120, C121, C14 | 5 |
| SQ77 | `sq_hw_gdn_fusions.py` | C122, C123 | 12 |
| SQ78 | `sq_hw_data_path.py` | C124 | 2 |
| SQ79 | `sq_hw_thermal_log.py` (CPU, `nvidia-smi` only, runs during a campaign) | C125, C126 | 0 (15 CPU) |

Suggested order (gain per GPU-minute): SQ70, SQ75, SQ71, SQ77, SQ73, SQ72, SQ78, SQ76, SQ74; SQ79 alongside the
next campaign.

## 6. For the human
- **Power mode (P10):** switch Windows to "Best performance" on AC (and the laptop vendor's performance profile)
  before long campaigns; then let SQ82 re-measure. Agents must not change system settings.
- **Optional install (SQ83):** `torchao` matching torch 2.14.1+cu130 in `~/.venvs/gogeta` (run `pip install
  --dry-run` first: it must not replace the CUDA torch). Low priority: predicted no gain at our size.
- No other step here needs a download, install or system change.

## 7. Caveats
- The BF16/FP8 peaks are inferred; the strongest evidence for "full rate" is that our 72.1 TFLOP/s would need a
  >= 2.93 GHz sustained clock under the GeForce half-rate hypothesis, while we observed ~2.0 GHz under training
  load (the clock during the original 4096^2 benchmark is unknown). SQ71 settles it per clock.
- The bandwidth model in section 3 is a hand count with ~+-30% error; SQ75 replaces it with a profile.
- The power and temperature observations are 3.5 minutes of one 20M-token E10 run (2 s sampling).
- GB10 community numbers (MAMF, CUTLASS, mmapeak) are forum posts, not vendor specs.
- The C2 diagnosis is from reading fla's source and our call sites; it is a prediction until SQ70 runs.
