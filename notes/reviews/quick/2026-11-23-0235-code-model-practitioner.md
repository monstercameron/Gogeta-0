# Quick review 2026-11-23-0235 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The F2 interaction is real in this run, but full fusion is still 11.2% worse than dense; GDN alone has the best loss and costs about 1.6× the wall time. For code-model use, quality per time still needs to shape the architecture choice. (DEVLOG.md, v0.4.98; C201)
- Clean evaluation matters: C191 reports that 41.68% of code-py-v1 validation docs have at least 80% 13-gram overlap with train. Treat familiar-file performance cautiously. (experiments/claims-status.md; C191)
- The model has no KV cache, so generation recomputes the prefix each token; the benchmark notes also say execution tasks floor at this model scale. Likelihood probes are currently more informative for comparing checkpoints. (notes/benchmark-suite.md; gogeta/model.py)

## Idea

- **Import-context ablation.** **Hypothesis:** E1’s relative advantage may come from local continuation patterns rather than useful project/API context. **Cheapest test:** about 5 GPU-minutes at 13M parameters; on source-disjoint files, score imported-name targets with the original import block and with that block removed, for E1 and E10. **Change the plan if:** removing imports changes the E1–E10 gap by over 1% or flips its sign; prioritize context/data work before drawing architecture conclusions.

## Question

- Is Q14 intentionally still marked HUMAN in the queue, given the benchmark suite’s existing harness and evaluation work?

## Triage
- Import-context ablation: accepted as SQ325 (inference on the 7+7 checkpoints, source-disjoint files).
- Question (why Q14 is still HUMAN): intentionally. Its `needs_human` flag stays, because tier-1 needs dataset downloads (`evals.fetch_data hf / uncheatable`) and lm-eval installs, and AGENTS §0a makes downloads and installs human-only. Its other "missing" items are stale: notes/benchmark-suite.md is written and evals/ is tracked. The only real blocker is that human approval.
