<!-- Paste-ready for experiments/checks/PREDICTIONS.md (append at the end, below "Replication protocol (Phase 2b) ..."). Everything from the next heading to the end of this file is the addendum. Notes for the main agent are in the HTML comments. -->

---

## Addendum A2 (2026-11-22): the statistical layer and the primary metric of Phase 2b
*Append-only. Written and committed **before any Phase 2b run has finished** (no 200M-token E10/E1/E11/E9/E2 result exists at the time of writing; K7's 20M-token sweep is the only LM data seen). It does not edit or weaken any criterion above. Implementation: `experiments/stats/replication_stats.py` (library `experiments/stats/core.py`, doc-level scorer `experiments/stats/score_doclevel.py`). Output: `experiments/checks/results/REPLICATION-stats.json`.*

### A. The primary decision rule is unchanged; only the metric it is applied to is stated
**Reason (K9, 2026-11-22):** 41.7% of val docs (1,667/4,000) have >= 80% of their 13-grams in train, and removing them raises E0's bits/char by 25.8% (0.2209 -> 0.2779). The training-time `val_loss` in `summary.json` is computed on that contaminated val, so it is demoted to a secondary, descriptive metric.

| Role | Metric | Definition |
|---|---|---|
| **Primary** | **clean-val doc-level bits/char** | K8's doc-level scorer (`score_checkpoint`, window 1024, `<|eos|>` prefix) on the val docs whose per-doc 13-gram overlap with train (K9 `val_per_doc_overlap`) is **< 0.80** (2,333 docs; docs with no overlap value count as clean). One number per finished run. |
| Sensitivity | clean-val bits/char, overlap **< 0.50** | same, threshold 0.50 (1,174 docs) |
| Secondary (descriptive) | training-time `val_loss` (full val, packed, contaminated) | as registered |
| Memorisation gap | bits/char on near-dup docs (>= 0.80) minus bits/char on clean docs (< 0.80), per run | see section D |

Criteria 1 (C5), 4 (E0 parity), 5 (loop question E9 vs E2) and the E11 rule above are applied **mechanically and with identical thresholds** to the primary metric (`replication_stats.py` re-runs `replication_analysis.py`'s own code on the clean-val numbers). Their noise units (pooled sd, s_E1) are computed from the same metric. Verdicts on the sensitivity subset and on `val_loss` are reported next to them; **if they disagree with the primary verdict, the disagreement is reported in the devlog entry and the ledger evidence, and the primary-metric verdict is the one recorded as the status.**
The following stay **as registered, on `val_loss`** because their numbers were registered in `val_loss` units: criterion 2 (predicted seed sd <= 0.004; also reported on the primary metric, not scored), criterion 3 (E1 reproducibility 0.5846 +- 0.002; N/A because LR_E1 = 6e-3), E10's registry prediction (mean 0.595-0.615, sd <= 0.004, throughput), E11's "within +-1.0% of E1" prediction (scored on both metrics).
**Fallback:** if the doc-level scoring of a run is missing (GPU slot unavailable), that run's primary-metric verdicts are INCOMPLETE, and the `val_loss` verdicts of `replication_analysis.py` stand, flagged "contaminated val".

### B. Secondary layer (estimation; it never changes a registered verdict)
For every comparison, always printed with n, the seed list, the protocol (LR, tokens, code commit via the run names) and the df:

| Comparison | Design | Reported |
|---|---|---|
| **E1 vs E10** | n = 3 seeds each, paired by seed index (**same data order, different init**; see `experiments/stats/pairing.md`) | mean paired difference E1 - E10 (negative = E1 better), sd of differences, **t-based 95% CI with df = 2**, relative gap % (and its CI), dz, Hedges' gz; Welch 95% CI (df ~ 4) and Cohen's d as sensitivity; exact sign-flip p (n = 3: the smallest possible two-sided p is 0.25, so it cannot be significant: reported only to show that) |
| **E11 vs E1** | n = 3, paired (**same init and same data order**, only the neg-eig flag differs) | the same, plus **TOST** (paired t, df = 2) with margin = **max(1% of mean(E1), 2 x pooled sd)**, exactly the registered band; 90% CI vs the margin; Welch-TOST as sensitivity; `min_margin_at_zero_diff` (the smallest margin this data could ever certify) |
| **E2 vs E9** | one seed each | **descriptive only**: difference, relative gap %, and the difference in units of sqrt(2) x s_E1. No CI, no p-value, no test. The registered criterion 5 stays the only decision rule. |
| E0 vs E10 | one E0 run vs three E10 seeds | 95% prediction interval for one new run (t, df 2): descriptive; the registered parity band stays the rule |

### C. Multiplicity policy for the Phase 2b family (fixed now)
- **Confirmatory family (m = 2, fixed, alpha = 0.05, Holm-Bonferroni):**
  - **H1** E1 vs E10 on the primary metric (two-sided paired t; p from df = 2).
  - **H2** E11 vs E1 equivalence on the primary metric (TOST p = max of the two one-sided p).
  - A missing member is not dropped (m stays 2). "Supported" = Holm-adjusted p < 0.05. This is reported next to, not instead of, the registered verdicts of C5 and E11.
- **Exploratory family (Benjamini-Hochberg, q = 0.10, hypothesis-generating only, never changes a ledger status):** the same comparisons on the < 50% subset and on `val_loss`, and the memorisation-gap difference E1 vs E10.
- **Descriptive (no inference):** E2 vs E9, E0 vs E10, per-run throughput/FLOPs tables, the variance decomposition.
- **Not in any family:** K7's LR choice (a tuning decision, n = 1), K8/K9 checks.
- Any comparison added after seeing results is exploratory by definition and labelled "post hoc".

### D. Memorisation gap (pre-registered metric)
`gap = bpc(near-dup docs, overlap >= 0.80) - bpc(clean docs, overlap < 0.80)` per run (also at 0.50), plus the ratio. E1 vs E10 is compared by the same paired CI. **Flag rule:** if the 95% paired CI of `gap(E1) - gap(E10)` excludes 0, E1 and E10 differ in how much they profit from repeated docs (an attention-vs-recurrent-state recall confound); the C5 and E11 sentences are then written "on clean val, with a memorisation difference of X" and the < 50% sensitivity result is shown beside the primary one. The flag never mechanically changes a verdict. Expected sign: negative for every model (repeated docs are easier).

### E. What n = 3 can and cannot show (design-level, independent of the results)
With the registered seed sd of 0.004 (val_loss units, 0.67% of 0.60):
- Two-sample (Welch/pooled) n = 3 per arm, alpha = 0.05, power 0.80: **minimum detectable effect 0.0123 = 3.07 sd = 2.05% of 0.60** (df 4). Paired n = 3: 3.26 sd of the differences (df 2). With Holm m = 2 (alpha 0.025): 3.74 sd (2.5%).
- The registered prediction "E1 beats E10 by 1.5-4%" therefore has power ~0.55 at 1.5%, ~0.78 at 2.0%, ~0.98 at 3.0%, ~1.0 at 4.0%. A true 1.5% gap is more likely than not to give a CI that includes 0.
- **The registered VERIFIED rule (gap > 2 pooled sd and complete seed separation) is weaker than a 5% test:** at the threshold the Welch t is 2.45 (df 4, p ~ 0.07), and 3 vs 3 complete separation has exact rank-test p = 0.10 (two-sided). A 95% CI excludes 0 only if the gap exceeds ~2.27 pooled sd (Welch) or ~2.48 sd of the paired differences. The ledger must therefore state, whenever C5 is recorded as VERIFIED, whether the 95% CI excludes 0 (see A2 policy, `notes/statistics-policy.md` section 2). The registered rule is not changed.
- TOST with the registered margin: with margin = 2 pooled sd, equivalence can be certified only if |diff| < ~0.26 sd (Welch); with margin = 1% of mean (0.006) it depends on the observed sd. `min_margin_at_zero_diff` in the output tells which.
- To detect a 1% (0.006) gap at sd 0.004 with 80% power one needs about 9 seeds per arm; 5 seeds per arm give an MDE of 2.0 sd.

### F. Seeds, variance decomposition, scope
- n = 3 seeds (0, 1, 2) per arm as registered. Variance is decomposed across the (seed x model) matrix (E10/E1/E11, and E10/E1 for the data-order-only share) with 2-4 df: indicative only. Separating init from data-order variance needs a code change (SQ110) and is not claimed.
- E2 and E9 are single-seed: every statement about the loop question is descriptive/exploratory, and says "n = 1".
- All numbers are for this corpus, 14M parameters, 200M tokens, and the tuned LRs of K7 (itself a single-seed, 20M-token choice).
- Run: `~/.venvs/gogeta/bin/python experiments/stats/score_doclevel.py --checkpoint <each qualifying run's model.pt>` (GPU slot, after the campaign), then `~/.venvs/gogeta/bin/python experiments/stats/replication_stats.py`.
