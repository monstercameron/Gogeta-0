# Quick review 2026-11-23-1619 - cheapest-experiment hunter (Codex GPT-6 Luna high)

- **Observation:** The claims ledger still has 74 UNVERIFIED claims, so cheap checks that settle implementation or evaluation assumptions can have high value. (`experiments/claims-status.md`)
- **Observation:** B1 is running a 23-run, 20M-token screen; its three gating contrasts are exploratory and can open follow-ups, but cannot promote claims. (`DEVLOG.md`, v0.5.31; v0.5.33)
- **Observation:** No GPU item is currently pickable; Q15 is the next CPU item, estimated at 60 minutes. (`experiments/QUEUE.md`)

- **Idea [speculation]: Effective-batch invariance check.** Hypothesis: splitting the same examples and effective batch across different microbatch/gradient-accumulation settings leaves the update and short-run loss effectively unchanged. **Cheapest test:** at 13M parameters, run two fixed-seed, fixed-token-order configurations for 1–2M tokens with matched effective batch and optimizer steps; about 5–10 GPU minutes. **Change the plan if:** validation loss differs by more than 0.2%; standardize accumulation settings for comparisons and investigate trainer math before relying on short screens.

- **Question for the main agent:** Can the trainer hold token order and effective batch fixed while varying microbatch size?
## Triage
- Effective-batch invariance: already built and tested, so not added. `gogeta/train.py` supports `train.micro_batch` / `--micro-batch` (gradient accumulation). `forward_backward` splits the same batch into equal row chunks, each backpropagated at loss / n_micro, so the summed gradient equals the full-batch mean-loss gradient "up to float associativity; same data order, same effective batch" (lines 175-191). `tests/test_options.py` covers it. Every B1 run, like every A-series run, uses micro_batch = batch_size (no accumulation), so the comparisons do not depend on it.
- Question (can the trainer hold token order and effective batch fixed while varying micro-batch size?): yes. Data order comes from the seeded batch sampler (`train.py:61-65`), independent of micro_batch, and accumulation keeps the effective batch. The remaining difference is float summation order (bf16 under autocast). SQ231 measures the throughput side per GPU SKU.
