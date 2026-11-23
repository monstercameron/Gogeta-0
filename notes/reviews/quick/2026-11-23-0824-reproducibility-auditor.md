# Quick review 2026-11-23-0824 - reproducibility auditor (Codex GPT-6 Luna high)

- **Observation:** 71 of 129 claims are UNVERIFIED, and 26 are PARTIAL; the ledger’s scope fields are also frequently TBD. The status report shows how much remains difficult to reproduce from a claim alone. (experiments/claims-status.md)
- **Observation:** C191 records conflicting commit provenance between its card and K9 results JSON. That makes it unclear which code version produced the cited measurement. (C191, experiments/claims-status.md)
- **Observation:** The queue snapshot was updated at 03:16, after the 03:08 devlog entry. Its next picks are SQ50 and SQ61, while the devlog reports E60 runs still finishing; readers may need a clear link between queue state and active campaign state. (experiments/QUEUE.md; DEVLOG.md)

- **Idea — artifact commit-provenance lint:** **Hypothesis:** Comparing each result artifact’s recorded `git_commit` with its card and ledger evidence would catch mismatched provenance like C191. **Cheapest test:** About 3 CPU-minutes at 13M scale: check K9 and a synthetic mismatch. **Change the plan if:** a mismatch passes; require matching run-commit provenance before using results to support claims.
- **Idea — generated-status consistency check:** **Hypothesis:** A small check that regenerates `claims-status.md` and flags stale claim-count summaries in project tracking would catch drift such as SQ319’s older `117 of 127` count. **Cheapest test:** About 1 CPU-minute at 13M scale. **Change the plan if:** regeneration is not byte-identical or the stale count remains; add the check to release validation.

- **Question for the main agent:** When card and result artifact provenance conflict, which should be treated as canonical evidence for the claim?

## Triage
- Artifact commit-provenance lint: merged into SQ274 (the release-lint evidence check). It adds two conditions: a result JSON's `git_commit` and `git_describe` must agree, and they must match the card's cited run commit.
- Generated-status consistency check: rejected as covered. `python experiments/claims_list.py --check` already fails on stale generated views ("stale: ... run claims_list.py"), and the version cutter regenerates them on every release. Counts quoted inside side-quest rows (SQ319's "117 of 127") are dated snapshots, not generated text.
- Question (card vs artifact provenance): the machine-written artifact outranks the hand-written card. When the artifact is itself inconsistent, as K9.json is (commit 60f4dcd, describe gb64199d-dirty), neither is canonical, and the fix is to re-run. K9 is a ~3.5-min CPU job, so it is being re-run now at a clean committed revision, and C191 will cite the reproduction.
