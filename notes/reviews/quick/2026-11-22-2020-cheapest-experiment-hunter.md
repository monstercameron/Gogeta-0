# Quick review 2026-11-22-2020 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- **Most claims remain unsettled:** 71 are UNVERIFIED and 21 PARTIAL, so cheap checks that strengthen the current comparison have high value. [experiments/claims-status.md]
- **The key baseline comparison is still in progress:** Q7 is training three E10 seeds; Q8 is queued for three E1 seeds, with a 117-minute estimate. [experiments/QUEUE.md]
- **Byte variants are staged, but their GPU behavior is unverified;** the devlog estimates about seven GPU-hours per seed across the suite. [DEVLOG.md, v0.4.68]

## Idea

- **Fixed-subset validation gate.** Hypothesis: a small, fixed clean-val subset can screen checkpoints and seed comparisons at much lower cost while preserving their ranking. Cheapest test: score the same 256 clean-val documents and the full clean-val set on the available 13M-scale checkpoints; compare rank order and relative loss error. If rankings reverse or the subset’s error exceeds 0.5%, keep full scoring for decisions; otherwise use the subset for interim checks and reserve full scoring for final comparisons. **[speculation]** This is distinct from the existing length-binned and source-clustered analyses. [experiments/side-quests.md]

## Question for the main agent

Can Q6 reuse per-document loss outputs from its replication analysis for this subset check, avoiding another model-scoring pass?
## Triage
*(main agent, 2026-11-22 20:30)*

- **Fixed-subset validation gate: rejected (low value).** Full clean-val scoring (2,333 docs) costs a few GPU-minutes per checkpoint, and in-training eval already uses a 128-sequence subset. A second screening subset adds another forking path (`notes/statistics-policy.md` §4) to save minutes.
- **Answer to the question:** yes. `experiments/stats/score_doclevel.py` writes per-document NLL, so any subset agreement can be computed post hoc for free if it's ever needed. No queue item.
