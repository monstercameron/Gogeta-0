# Quick review 2026-11-23-1710 - skeptical peer reviewer (Codex GPT-6 Luna high)

## Observations

- C214 is explicitly limited to one out-of-corpus set. Its six-seed interval supports that comparison, but the result does not yet establish generalization to other code populations. [C214; DEVLOG.md, v0.5.38]
- The ternary cost estimate is close to zero on SQ99, but its 95% CI spans both a modest gain and a modest loss; “vanishes” is stronger wording than this estimate alone warrants. [DEVLOG.md, v0.5.38; C213]
- The claims ledger has 75 UNVERIFIED claims, and several claim-definition fields remain TBD. That makes scope and refutation criteria a substantial source of uncertainty alongside model results. [experiments/claims-status.md]

## Ideas

- **[speculation] Check whether the out-of-corpus loop result is driven by long files.** Hypothesis: A2f’s advantage over E24 is concentrated in longer-context files. **Cheapest test:** CPU, about 5 minutes; stratify the existing per-document scores by file length and compare paired effects across bins. **Change the plan if:** the advantage appears only in the longest bin; narrow the claim and prioritize a context-length control.
- **[speculation] Check whether the ternary comparison depends on token volume per file.** Hypothesis: A2s’s near-zero average cost hides a consistent penalty on longer files. **Cheapest test:** CPU, about 5 minutes; plot paired A2s–A2f per-document differences against token count using the existing SQ400 scores. **Change the plan if:** the penalty grows with file length; qualify the “cost vanishes” conclusion and test length-stratified scoring before using it for model selection.

## Question

- Will C214 remain scoped to this one dataset until a second independent out-of-corpus set is evaluated?
## Triage
- Length dependence of the out-of-corpus loop result, and of the ternary result: both merged into SQ385 (length bins) as an extension that runs on SQ400's per-file out-of-corpus scores, with this review's thresholds (advantage only in the longest bin; penalty growing with length).
- Observation 2 ("vanishes" is too strong for +0.42% [-0.92, +1.77]): agreed. v0.5.38's title and entry say "the ternary cost vanishes". The correct wording is "not distinguishable from 0 out of corpus (95% CI [-0.92, +1.77], n 6; no TOST, so equivalence is not shown)". An erratum is in the next version. C213's ledger text already gives the CI.
- Question (will C214 stay scoped to one set until a second independent out-of-corpus set is evaluated?): yes. C214's scope field names the SQ99 set, its limitation says "one out-of-corpus set", and its refutation criterion is a second set (the SQ405 temporal holdout or another held-out package set). Any README or design wording cites it as "on the SQ99 out-of-corpus set".
