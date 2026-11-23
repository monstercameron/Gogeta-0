# Compute rental options (snapshot)

**Source:** vast.ai/pricing (live marketplace), read in the browser pane on **2026-11-22 ~19:50 local**. Prices change hourly; "from" is the cheapest current offer, "median" the platform median. **Bandwidth figures are spec numbers recalled by the main agent: verify before relying on them.** Not a purchase; renting is the human's decision (AGENTS §0a).

## What our workloads need
- **Memory:** ~9–10.5 GB VRAM at B=32 × T=1024 for the ~14M models (E10 8.9 GB, E1 10.4 GB on the N1X, `status.sh`). E2 (looped) and the T=4,096-byte models need more (unmeasured; SQ189).
- **Generation:** **Ampere or newer** (bf16 + fla 0.6.0 Triton kernels). Turing, Pascal and Volta are excluded.
- **Bandwidth:** training is **memory-bandwidth-bound** (`notes/hardware-spark.md`), so speed tracks bandwidth more than TFLOPs. The N1X has ~300 GB/s.

## Candidates (Ampere+, ≥ 12 GB)
| GPU | VRAM | From / median $/hr | Bandwidth (approx.) | Median $ per 100 GB/s-hr | Fit |
|---|---|---|---|---|---|
| **RTX 3090** | 24 GB | **0.11 / 0.20** | ~936 GB/s | **0.021** | Best value; fits all |
| RTX 3060 | 12 GB | 0.03 / 0.07 | ~360 GB/s | 0.019 | Tight for E1; no E2/bytes |
| RTX A4000 | 16 GB | 0.07 / 0.11 | ~448 GB/s | 0.025 | E0/E1 |
| RTX 3090 Ti | 24 GB | 0.17 / 0.25 | ~1,008 GB/s | 0.025 | All |
| RTX 5060 Ti | 16 GB | 0.09 / 0.13 | ~448 GB/s | 0.029 | E0/E1 |
| RTX 5090 | 32 GB | 0.27 / 0.67 | ~1,792 GB/s | 0.037 | All; fastest consumer card |
| RTX 4090 | 24 GB | 0.28 / 0.47 | ~1,008 GB/s | 0.047 | All |
| A100 PCIe / SXM4 | 80 GB | 0.36 / 0.65 · 0.33 / 0.84 | ~1.9–2.0 TB/s | 0.034–0.041 | Overkill on memory |
| GB10 (DGX Spark) | 128 GB unified | 0.27 / 0.39 | ~273 GB/s | 0.14 | **Same sm_121 family as our N1X:** good for exact reproduction, not speed |
| H100 SXM | 80 GB | 1.43 / 2.21 | ~3.35 TB/s | 0.066 | Fast; we can't use its scale |

Other GPUs on the page (all 79), summarised:
- **Enterprise:** H200 NVL, B300, B200, H200, H100 NVL/PCIe, GB200 ($2.33–$10.91/hr median).
- **Consumer and pro Blackwell/Ada:** RTX PRO 6000 (96 GB, ~$1.4–1.7), L40S, RTX 6000 Ada, 5080/5070/4080/4070/4060 variants (8–16 GB).
- **Ampere workstation:** A6000, A5000, A40, A10.
- **Turing / Volta / Pascal** (excluded: no bf16/Triton fit).

## Recommendation (main agent, 2026-11-22)
- **Cheapest GPU that safely runs everything: RTX 3090** (~$0.20/hr median). Expected ~2–3× the N1X per run (bandwidth-based estimate; benchmark first).
- **Backlog:** ~35–60 N1X-GPU-hours (A3 controls, decomposition and factorial; the A4 battery; Byte Quest E12–E16; side quests), or ~15–25 RTX 3090-hours.
- **Scale: 8× RTX 3090** (~$1.60/hr). That clears the backlog in ~2–4 h of compute (plus waits between dependent stages), ~$5–15 in total. The largest single stage is the factorial (N=46 runs) and most stages are 6–12 runs, so cap at 16.
- **Rules for safe use:**
  - **one GPU SKU per comparison** (numerics and nondeterminism differ);
  - a one-seed **bridge run** (E10, E1) vs the N1X before scaling, since K7's LRs were tuned on the N1X;
  - **on-demand** until mid-run resume (`notes/patches/cockpit-resilience-train.patch`) is applied; after that, interruptible (~50% cheaper);
  - a small **remote-runner extension** to the cockpit (AGENTS §0g) is needed, since today's launchers are single-machine;
  - the corpus (525 MB) and the repo are fine to upload (public repo; check the corpus licence terms).
- **First step, if the human approves spending:** 1× RTX 3090 on-demand for ~1 h: a throughput benchmark plus the bridge run. Then scale.
