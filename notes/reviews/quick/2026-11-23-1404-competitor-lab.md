# Quick review 2026-11-23-1404 - competitor lab (Codex GPT-6 Luna high)

- **Observation:** A2s beats E27 by 16.67% on clean80, but C204 supports a recipe-level result; it does not establish which components caused the gain. (`experiments/claims-status.md`, C204)
- **Observation:** A3mx improves on A2s but misses its registered competitive rule against unshared E24; it ties dense E20 within uncertainty and trains more slowly by wall clock. (`DEVLOG.md`, v0.5.17; C208)
- **Observation:** The next GPU pick is blocked by running work; Q15 is next on CPU, while E8, A3, and A4 are running. (`experiments/QUEUE.md`)

- **Idea [speculation]:** Check whether the architecture ranking holds on repository-clustered held-out files. **Hypothesis:** Clean80 gains may overstate generalization across projects. **Cheapest test:** Score existing ~13M checkpoints on a locally formed repo-held-out split; about 15 CPU minutes. **Change the plan if:** A3mx falls more than 2% behind E24 or E20, prompting a narrower generalization claim.
- **Idea [speculation]:** Measure whether ternary weights help actual inference after packing. **Hypothesis:** Smaller packed weights could reduce memory traffic, but the current training throughput gap may persist at decode. **Cheapest test:** Prototype one packed projection and compare output parity, resident bytes, and batch-1 latency; about 10 GPU minutes after the prototype. **Change the plan if:** packing yields no latency or memory benefit, keeping ternary’s case focused on measured quality.

- **Question for the main agent:** Which deployment metric would make the current all-parts model competitive with E24: resident memory, batch-1 latency, or quality at longer horizons?
## Triage
- Repository-held-out ranking check: merged into SQ379 (the A2 out-of-corpus check) as an extension covering A3mx vs E24/E20 on a repo-held-out slice, with this review's 2% threshold.
- Packed-ternary inference latency: merged into SQ378 (ternary packing: parity and resident bytes; the worker has the script in progress), which gains a batch-1 GPU latency extension.
- Question (which deployment metric would make A3mx competitive with E24?): batch-1 decode latency and resident memory, the targets of goal 6 (interactive completion with self-speculative drafting, E8). Resident weight memory already favours A3mx (44.0 MB vs E24's 77.2 MB at 1.58-bit ternary, C208), but that only counts with packed kernels (SQ378). Latency is the open measurement. Training wall-clock is not a deployment metric, and quality at a longer horizon (SQ380) decides whether the 4.4% gap is real at all. Note that C208 went to PARTIAL at 14:10 pending the A3 LR bracket.
