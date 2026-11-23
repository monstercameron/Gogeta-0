# Quick review 2026-11-23-0225 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- The F2 entry reports 217 runs after excluding seed 0, but also says 250 per-run score files were committed. The entry doesn’t reconcile those counts, so it’s unclear which files entered the analysis. (`DEVLOG.md`, v0.4.97; C201)
- F2’s analysis script was committed before its first run, and the entry points to a saved JSON result. That gives the analysis a useful audit trail; the reported primary and sensitivity estimates are close. (`DEVLOG.md`, v0.4.97)
- The newest quick review records SQ61’s three fail-closed sandbox gates in its triage, while the queue excerpt still places SQ61 as the next CPU item. The review’s acceptance should be reflected in the item’s acceptance criteria when it is run. (`notes/reviews/quick/2026-11-23-0149-safety-sandbox-reviewer.md`; `experiments/QUEUE.md`)

## Ideas

- **Reconcile F2 inputs.** Hypothesis: all 250 score files can be accounted for as included, excluded, or invalid, and the 217-run analysis is reproducible from that manifest. **Cheapest test:** 1 CPU-minute; enumerate the existing 13M-scale run artifacts and compare them with the analysis inclusion set. **Change the plan if:** any file’s disposition is missing or the recomputed count differs; resolve the discrepancy before relying on C201.
- **Check the interaction implementation.** [speculation] Hypothesis: a small synthetic fixture with known cell means can catch a sign or seed-blocking error in the F2 interaction calculation. **Cheapest test:** under 1 CPU-minute; run the analysis formula on hand-constructed 13M-scale-shaped data with a known interaction. **Change the plan if:** the output misses the known value or its sign; audit and rerun the F2 analysis before extending the interpretation.

## Question

- What explains the difference between 250 committed F2 score files and 217 reported runs?

## Triage
- Reconcile F2 inputs: done. `experiments/checks/results/f2_input_manifest.json` (built by `experiments/design/f2_input_manifest.py`) gives every file's disposition:

  | Disposition | Files |
  |---|---|
  | Included | 217 |
  | Other factorial cells (E23/E25/E26, not in F2) | 16 |
  | Seed 0, excluded as registered | 5 |
  | LR-sweep runs at non-registered LRs | 12 |

  That is 250 in total, with no duplicate (cell, seed) among the included runs.
- Check the interaction implementation: done. `experiments/design/check_f2_analysis_synthetic.py` builds fake inputs with a known I = -3.5% plus seed effects:
  - with no noise, the seed-blocked estimate returns -3.500% exactly;
  - with noise 0.008, it returns -3.97%, CI [-4.87%, -3.07%], which covers the truth.
- Question (250 vs 217): answered by the manifest above.
- SQ61 observation: its acceptance criteria already carry the three fail-closed gates in `experiments/side-quests.md`. QUEUE.md shows only the title.
