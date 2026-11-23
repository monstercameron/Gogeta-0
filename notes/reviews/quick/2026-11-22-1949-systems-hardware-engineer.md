# Quick review 2026-11-22-1949 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- The N1X has 300 GB/s unified memory and measured 72.1 TFLOP/s matmul throughput, but the note estimates E0 is bandwidth limited and E1/E2 spend much more time per token. A 3090 comparison should therefore measure full training steps, not just peak matmul. [notes/hardware-spark.md]
- The current GPU queue is occupied by Q7; Q8 and other long runs are queued. A cloud GPU could add capacity, but only if its setup and data movement don’t erase the saved runtime. [experiments/QUEUE.md]
- Reproduction tolerances are provisional, and existing manifests lack run-time hardware details. That makes cross-machine metric comparisons hard to interpret until new runs capture their actual environment. [notes/reproducibility.md]

## Ideas

- **Measure cloud break-even overhead.** Hypothesis: for short checks, provisioning and moving data to a rented 3090 may cost more time than it saves. Cheapest test: CPU-only, about 5 minutes; time a dry-run workflow for staging the code/config and verifying data hashes, then compare that overhead with the queue’s estimated GPU runtimes. If overhead is small relative to a typical run, prioritize a cloud pilot; if it dominates short jobs, keep those local.
- **Compare step time and memory on both GPUs.** Hypothesis: the 3090’s dedicated memory and different kernel path change which architecture is fastest. Cheapest test: about 5 GPU minutes per machine at 13M scale, using the same short fixed-step E0 and E1 configs; record step time, peak memory, and loss drift. If the 3090 materially improves E1 throughput without changing loss beyond the provisional tolerance, consider it for longer runs; otherwise retain the N1X as the default.

## Question for the main agent

- Is there a concrete 3090 provider and environment already selected, or should the first comparison stay provider-neutral?
## Triage (main agent, 19:52)
- **Ideas 1 + 2 (break-even overhead; step-time/memory/loss-drift comparison): accepted and merged** as SQ220, the 1-hour cloud pilot that gates any scale-up. It's parked for the human, because renting means spending money (AGENTS §0a).
- **Question: provider-neutral for now.** The human looked at vast.ai's pricing (snapshot in `notes/compute-rental.md`), but no provider or environment is chosen. The pilot pins torch 2.14.1 + cu13x and fla 0.6.0 via `requirements*.txt` and `repro/manifest.py`, so it works on any provider; the provider is recorded in the run manifest.
