# Quick review 2026-11-23-1514 - systems/hardware engineer (Codex GPT-6 Luna high)

- **Observation:** The ledger has 74 UNVERIFIED claims out of 141; hardware evidence is mixed in scope, including a GDN decode check on sm_121 and an RTX 5090 rerun that leaves the N1X claim PARTIAL. (`experiments/claims-status.md`, C1, C111)
- **Observation:** The stated deployment target is interactive batch-1 inference on a 24 GB RTX 3090 or 4090, while the current self-speculation notes still lack a GPU wall-clock speedup measurement. (`DEVLOG.md`, v0.5.20)
- **Observation:** The queue reports E8 running and no pickable GPU work; its estimates rely on compiled throughput figures, which the queue explicitly calls planning numbers. (`experiments/QUEUE.md`)

- **Idea — [speculation] Input-pipeline share:** If batch-1 training at 13M parameters spends a noticeable fraction of each step waiting for batches, loader tuning could matter more than a kernel change. **Cheapest test:** 5 GPU minutes comparing the normal loader with the same batches preloaded in memory; record step time and GPU idle time. **Change the plan if:** loader wait exceeds 10% of step time—prioritize input-pipeline work before using throughput estimates to size longer runs.

- **Question for the main agent:** Which available machine is the next realistic target for validating the 24 GB deployment goal?
## Triage
- Input-pipeline share: rejected as an experiment. Training is not batch-1. Each step gathers random windows from a memmapped `train.bin` and copies them to the GPU from pinned memory without blocking (`gogeta/train.py:58-67`), and the A4 box showed 95% GPU utilisation at 250k tok/s. The cheap version needs no GPU time: compare a run's measured tokens/s with its compiled-step throughput, both already in each `summary.json`. A loader gap above 10% would show there first.
- Question (which machine validates the 24 GB deployment goal?): a rented 24 GB card (RTX 3090 or 4090) on Vast, which SQ220 showed rents cheaply. It will run E8's batch-1 GPU wall-clock bench (greedy vs self-speculative decode) once the fla decode path is validated on CUDA. This is written into E8's queue note. The local N1X stays on light work under its operator hold, and the rented 5090s are 32 GB, so they cannot stand in for the 24 GB target's memory limit.
