# Quick review 2026-11-22-2126 - skeptical peer reviewer (Codex GPT-6 Luna high)

## Observations

- The status summary lists 18 verified claims but 71 unverified; scope and refutation criteria are still TBD for most claims. That makes the ledger difficult to use as a reproducibility standard. [experiments/claims-status.md]
- C95 explicitly describes E1 vs E0 as near-matched parameters and matched tokens, not matched compute. Be careful not to treat that comparison as evidence of quality per FLOP. [C95]
- v0.4.75 records eight boxes dispatched, while the queue excerpt shows only Q7 running and Q8 ready. The recorded dispatch and queue state appear out of sync. [DEVLOG.md, v0.4.75] [experiments/QUEUE.md]

## Ideas

- **Reconcile dispatch records with queue state.** Hypothesis: the queue does not fully represent dispatched work, risking duplicate or missed runs. Cheapest test: a CPU-only join of registered items, run manifests, and queue states; about 1 CPU minute. If any dispatched item is absent or misclassified, reconcile before scheduling more work.
- **Check E1 vs E10 at matched training FLOPs.** Hypothesis: the ranking could change when token budgets are adjusted for measured per-token FLOPs. Cheapest test: a same-seed short comparison using the existing 13M-scale configs, about 2 GPU minutes on a 5090. If the gap changes sign or falls below the seed-noise estimate, qualify the matched-token result accordingly.

## Question

- Which record is authoritative for wave-one dispatch status when the queue and devlog disagree?
## Triage
*(main agent, 2026-11-22 21:30)*

- **Reconcile dispatch vs queue: accepted as a process fix, not a side quest.**
  - Fleet runs launched outside `tools/queue.py` are registered in `experiments/checks/PREDICTIONS.md` (wave 1 and wave 2 sections), and each box's `worker.status` is pulled with its results.
  - At the end of each wave, the queue gets one entry per dispatched item, marked done or failed with a pointer to its results file, so nothing is double-run.
- **E1 vs E10 at matched FLOPs: rejected (covered).**
  - E1 uses *fewer* training FLOPs per token than E10 (91.0M vs 96.8M, `experiments/design/staged-table.md`), so the token-matched comparison is conservative against E1. A FLOP-matched E1 would get about 6% more tokens.
  - The opposite direction, a dense model at E1's FLOPs, is E31, already in wave 2.
- **Answer to the question:** for dispatch status, the box's `worker.status` (pulled into `~/fusiondance-data/cloud/<wave>/`) is authoritative until the wave's results file and devlog are written; after that, the devlog and results JSON are.
