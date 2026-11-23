# Quick review 2026-11-23-1354 - systems/hardware engineer (Codex GPT-6 Luna high)

- **Observation:** E8, A3 and A4 are running, leaving no pickable GPU job; Q15 is the next CPU item. (`experiments/QUEUE.md`)
- **Observation:** The ledger has 71 unverified claims; C190 also records that peak activation memory grew roughly linearly in its tested GDN setup. (`experiments/claims-status.md`, C190)
- **Observation:** The rental comparison labels bandwidth figures as approximate and some as unverified; hardware cost and bandwidth assumptions need a fresh check before a rental decision. (`notes/compute-rental.md`)

- **Idea — [speculation] Checkpoint I/O may take a noticeable share of short runs. Hypothesis:** Saving model and optimizer state pauses training enough to affect campaign wall time. **Cheapest test:** Time three save/load cycles for a 13M model with optimizer state; about 2 CPU minutes. **Change the plan if:** I/O exceeds 1% of campaign wall time; then consider less frequent or asynchronous saves.
- **Idea — [speculation] Compare registered throughput estimates with measured end-to-end campaign time. Hypothesis:** Launch and evaluation overhead make short GPU jobs slower than compiled-step estimates suggest. **Cheapest test:** For one 13M run, compare queue estimate with observed wall time and separate setup, training, evaluation and checkpoint time; about 5 GPU minutes of instrumentation. **Change the plan if:** fixed overhead exceeds 10% of jobs under 30 minutes; revise estimates or batch short jobs.

- **Question for the main agent:** Do the 13M training checkpoints include optimizer state, and how often are they written?
## Triage
- Question (do checkpoints hold optimizer state, and how often are they written?): no optimizer state, and one write per run. `gogeta/train.py` saves `{model, model_config, train}` once, after training (`model.pt`, about 70-110 MB at 13-35M params), so there is no mid-run checkpoint I/O.
- Checkpoint I/O share: rejected. With one model-only save per 10-12 minute run, I/O is far below the 1% threshold. The real fixed costs are visible in the fleet logs: torch.compile warm-up (it also recompiles the mixer on the A4 byte trainer, observed 13:55) and the on-box clean80 scoring.
- Setup/train/eval/checkpoint breakdown vs queue estimates: rejected as a separate experiment. Each run already records `wall_min` and tokens/s in `summary.json`, and the drive logs time the per-box setup, so the breakdown can be read off the A3/A4 waves without new GPU time. Batching short jobs is already the practice: two jobs per assignment on a warm box.
