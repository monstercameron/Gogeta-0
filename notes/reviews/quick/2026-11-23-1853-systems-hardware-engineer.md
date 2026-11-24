# Quick review 2026-11-23-1853 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- The hardware note calls E0/E1/E2 bandwidth or elementwise bound, but marks that verdict as awaiting SQ75’s profiler breakdown. Its bandwidth estimate is computed, not a measured per-step traffic count. [notes/hardware-spark.md, §3; C119]
- SQ75 profiles three compiled steps per model with device-resident random inputs. That isolates GPU work, but does not show whether the real training input pipeline keeps the GPU supplied. [experiments/checks/sq_hw_step_profile.py]
- SQ75 is the next GPU pick, estimated at 8 minutes; Q15 is the next CPU pick, estimated at 60 minutes. The queue says to use a free GPU slot for short diagnostics. [experiments/QUEUE.md]

## Ideas

- **Measure input-pipeline overhead.** **Hypothesis:** the synthetic SQ75 profile represents training throughput if the real pipeline adds little wait. **Cheapest test:** GPU, about 3 minutes at 13M parameters; compare steady-state `train.py` batch time with device-resident step time, recording loader wait and GPU busy time. **Change the plan if:** loader wait exceeds 10% of wall time; prioritize data delivery before kernel tuning. This appears distinct from the listed hardware side quests. [experiments/checks/sq_hw_step_profile.py; experiments/side-quests.md]

## Question

- Can the SQ75 readout include a short real-loader timing comparison, or is that pipeline already excluded from the throughput numbers used for planning?

## Triage
- Idea (input-pipeline overhead): accepted as an SQ75 extension, not a new SQ. `train.py --bench` exercises the real loader: memmap random windows, `pin_memory` plus a `non_blocking` copy, synchronous in the step loop with no prefetch (`gogeta/train.py:58-76`). SQ75's readout now compares its steady-state step time with the device-resident step and reports the loader share. Prediction: under 2% (a 32x1024 uint16 batch is 64 KB, and the about 400 MB code-py-v2 corpus fits in page cache). The trigger is the review's: over 10% puts a prefetching loader first.
- Question (is the loader excluded from planning throughputs?): no. The planning throughputs in QUEUE.md (E0 210k, E1 100k, E2 44k tok/s) are described as "compiled throughputs" in the QUEUE policy; I have not traced whether each came from a full `train.py` run (which includes the loader) or a bench step. The SQ75 extension measures both, so the planning numbers get a stated source. SQ75's synthetic profile is for the kernel breakdown only, and the extension above checks that the two agree.
- Observations: agreed. C119's bandwidth verdict stays computed until SQ75 runs.
