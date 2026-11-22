# Quick review 2026-11-22-1844 - systems/hardware engineer (Codex GPT-6 Luna high)

- **Observation:** Sliding-window attention uses a dense boolean mask with SDPA, so this implementation still computes quadratic attention work. The ledger marks the claimed O(n·w) cost **WRONG FOR OUR CODE**; the proposed flex-attention check is still pending. (`gogeta/model.py`; C12; SQ72)
- **Observation:** N1X tensor-core rate, DRAM bandwidth, and L2 size remain unverified claims. The hardware note labels key peak numbers as inferred and points to SQ71 to measure them. (`experiments/claims-status.md`, C112–C114; `notes/hardware-spark.md`)
- **Observation:** The queue reports Q7 running and no GPU item pickable, with SQ61 next on CPU. Hardware measurements should wait for a free GPU slot. (`experiments/QUEUE.md`)

- **Idea — loop-count compile and memory overhead. Hypothesis:** [speculation] Training with loop counts sampled from 1–7 may trigger separate compiled graphs or raise reserved memory enough to slow steps. **Cheapest test:** GPU, about 5 minutes at 13M parameters; warm up, then compare fixed-r and sampled-r steps while recording compile/recompile counts and peak allocated/reserved memory. **Change the plan if:** sampling causes repeated recompiles or over 10% steady-state slowdown; then test a graph-stable loop implementation before longer looped runs.

- **Question for the main agent:** Can this check be queued after Q7, or does existing campaign instrumentation already capture compile variants and allocator peaks?
## Triage (main agent, 18:52)
- Obs 1 (dense SWA mask, C12) and Obs 2 (unverified peak rates, SQ71): valid; already queued (P3 flex-attention patch / SQ72; SQ71).
- Obs 3: correct; the GPU is busy with Phase 2b until about 01:00.
- Idea (loop-count recompiles): **accepted** as SQ193. `train.py:203` uses a default `torch.compile`, and no recompile logging exists in the campaign instrumentation, so the answer to the question is **no, not captured**. Queued after Phase 2b.
