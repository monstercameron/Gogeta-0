# Quick review 2026-11-23-0133 - competitor lab (Codex GPT-6 Luna high)

## Observations

- **C5 is refuted**, while the claims ledger still lists 71 claims as unverified. The current evidence base does not support a broad claim that the hybrid beats its reference. (`experiments/claims-status.md`)
- The F2 plan has 37 seeds shared across all five cells for the primary contrasts; E20’s additional seeds inform only the unblocked sensitivity analysis and its own variance. (`DEVLOG.md`, v0.4.91)
- E9 is queued to compare the looped model against the same blocks without weight sharing. That control directly tests whether extra compute from looping pays off. (`experiments/QUEUE.md`)

## Idea

- **Test source-disjoint generalization:** [speculation] clean80 performance may still benefit from related documents across train and evaluation, even when uncertainty is clustered. Hypothesis: a repository or source-package holdout could change the E1–E10 ranking. Cheapest test: score existing 13M checkpoints on a preselected, source-disjoint slice and report paired seed results by source; about 5–10 GPU minutes. If the gap changes sign or exceeds one pooled seed standard deviation, make source-disjoint evaluation a gate for headline comparisons. This extends the uncertainty analysis in SQ213 rather than repeating its clustered bootstrap.

## Question

- Will headline architecture comparisons require source-disjoint evaluation before their claims are upgraded?

## Triage
- Source-disjoint generalization: accepted as SQ314, re-scoring the 7+7 tuned 5090 checkpoints on the SQ99 out-of-corpus set with a package-clustered bootstrap. SQ99/C192 only covered single N1X checkpoints.
- Question (source-disjoint gate before upgrading headline claims): yes, going forward. The v3 corpus plan (notes/research-corpus-v3.md) builds a repo-disjoint `code-py-v3-test`, and E61's go/no-go uses repo-disjoint development and test panels. Until then, SQ314 is the check for the current headline.
