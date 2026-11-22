# Proposed PREDICTIONS.md addition (A2)

Paste the section **"Addendum A2 (2026-11-22)"** of `experiments/stats/phase2b-analysis-plan.md` (everything below its first `---`) at the end of `experiments/checks/PREDICTIONS.md`, commit it BEFORE the first Phase 2b run finishes, and tag it. It contains: the primary-metric statement (clean-val bits/char, K9), the unchanged registered rules, the secondary estimation layer, the Holm/BH families, the memorisation-gap metric, and the n=3 power statement.

Decisions only the main agent can make (not settled by the addendum):
1. Whether the C5 ledger status should require "95% CI excludes 0" in addition to the registered rule. The registered rule sits at p ~ 0.07 at its threshold (addendum section E); the addendum only requires that the CI be reported.
2. Whether GPU time for `score_doclevel.py` (about 25 min) is added to the campaign tail. Without it the primary-metric verdicts are INCOMPLETE and the `val_loss` verdicts stand, flagged "contaminated".
3. Whether to make the 1-line change to K8's `score()` (return `nll_doc`), which enables per-doc paired bootstrap (SQ112).
