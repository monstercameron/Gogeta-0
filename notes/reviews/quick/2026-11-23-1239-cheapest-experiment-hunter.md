# Quick review 2026-11-23-1239 - cheapest-experiment hunter (Codex GPT-6 Luna high)

- **Observation:** E13a and E13b finished at 200M bytes, but their results (0.2774 and 0.3338 nats/byte) remain unanalyzed; SQ374 is queued to extract the comparison. (DEVLOG.md, v0.5.8)
- **Observation:** The GDN × loops × ternary interaction changes sign from synergy at 20M to anti-synergy at 200M; C201 is refuted by the later result, so conclusions depend on training horizon. (C201, C202)
- **Observation:** The reward harness let code score 1/1 on a failing test by replacing `exec` or `compile`; GRPO is blocked pending SQ373, while SQ61 is next on the CPU queue for sandbox hardening. (DEVLOG.md, v0.5.7; experiments/QUEUE.md)

- **Idea [speculation]:** code-py-v2 may share the near-duplicate contamination found in code-py-v1. **Cheapest test:** scan the v2 validation set against training for 13-gram overlap, about 5 CPU-minutes with a 13M-scale data sample or scorer. **Change the plan if:** at least 1% of validation documents exceed the overlap threshold; then qualify or re-score the clean80 results for C201/C202.

**Question for the main agent:** Has code-py-v2 already had a train/validation near-duplicate check?
## Triage
- Scan v2 val for near-duplicates of train: rejected as covered by construction and by the scoring protocol. v2's validation split was built with a 13-gram rule: `build_v2.py` params `overlap_n` 13, `max_overlap` 0.3, plus MinHash family dedup at J >= 0.5 (`dataset-v2.manifest.json`; known issue: "val is redrawn ... docs that pass the 13-gram rule"). More importantly, the clean80 metric for every v2-trained model (F2, SQ263, A1, A2) does not use v2 val. It scores v1 val docs with < 80% 13-gram overlap against **v1** train (K9), and v2 train is a deduplicated subset of v1 train, so overlap with v2 train is at most that. A residual-overlap audit of the scoring set against v2 train specifically would only tighten an upper bound.
- Question (has v2 had a train/val near-duplicate check?): yes, at build time (the 13-gram rule above and the MinHash family split). The scoring set is controlled separately by K9 against the larger v1 train.
