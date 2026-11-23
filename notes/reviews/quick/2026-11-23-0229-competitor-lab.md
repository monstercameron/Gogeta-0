# Quick review 2026-11-23-0229 - competitor lab (Codex GPT-6 Luna high)

## Observations

- F2 reports a significant negative interaction, but the full fusion still scores 11.2% worse than dense; GDN alone is the strongest cell at 20M tokens. The interaction is evidence of overlapping penalties, not a win over the baseline. (DEVLOG.md, v0.4.97; C201)
- The result is tied to a specific horizon and dataset, and the devlog says SQ263 is checking the 200M horizon. That follow-up matters more than treating the 20M interaction as an architecture-level conclusion. (DEVLOG.md, v0.4.97; C201)
- E60 adds MoE as an off-by-default option with bit-identity checks, but the gather path graph-breaks under `torch.compile` and GPU throughput remains unmeasured. So the registered quality comparison does not yet establish a practical speed or efficiency gain. (DEVLOG.md, v0.4.96)

## Ideas

- **F2 seed influence audit.** Hypothesis: the negative interaction is not driven by a small number of influential seeds. **Cheapest test:** about 1 CPU-minute; recompute the seed-blocked interaction after omitting each seed in turn. **Change the plan if:** any omission flips the sign or removes the registered evidence for an interaction.
- **F2 throughput check.** Hypothesis: FLOPs-matched cells have materially different wall-clock cost. **Cheapest test:** about 5 GPU-minutes at 13M parameters; benchmark tokens/sec for the relevant mixer and loop configurations at the same sequence length. **Change the plan if:** the slowest cell is over 10% behind the fastest; report practical quality-per-time separately from the FLOPs-matched result.

## Question

- Has code-py-v2’s clean80 evaluation set had a near-duplicate audit comparable to the code-py-v1 finding in K9?

## Triage
- F2 seed influence audit: done (`experiments/design/f2_loso.py`, `experiments/checks/results/f2_loso_wall.json`). Leaving out each of the 73 seeds keeps I within [-3.69%, -3.42%]; there is no sign flip and the decision is unchanged.
- F2 throughput: done from the logged wall times, with no new GPU time. Median minutes per 20M run, by cell:

  | Cell | Median minutes |
  |---|---|
  | 000 (dense) | 0.58 |
  | 001 (ternary) | 0.60 |
  | 010 (loops) | 0.69 |
  | 100 (GDN) | 0.94 |
  | 111 (full fusion) | 0.92 |

  The GDN cells take about 1.6x the wall time of dense. This is added to C201 as a limitation: report quality per wall time separately from the FLOPs-matched result.
- Question (does v2 have a K9-style audit?): F2 is scored on **v1 val clean80**, i.e. the v1 val docs with < 80% 13-gram overlap against v1 train. v2 train is a subset of v1 train with the v1-val families removed (notes/data-v2.md), so those docs are at least as clean against v2 train. v2's own redrawn val has its novelty filter built into `build_v2.py`.
