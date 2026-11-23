# Statistics policy (A2, proposed 2026-11-22)

Scope: how we turn run numbers into claim statuses. Tools: `experiments/stats/core.py` (library + CLI), `experiments/stats/replication_stats.py`. What is paired in our runs: `experiments/stats/pairing.md`. Phase 2b registration: `experiments/stats/phase2b-analysis-plan.md`.
Why: an external audit rated statistical inference 4/10 (one-seed results, no CIs or effect sizes, no multiplicity policy). Several current claims rest on one run whose seed noise is ~0.4-0.7% of the loss, comparable to the effects reported.

## 1. Evidence classes
| Class | Examples | Unit of replication | Seeds needed |
|---|---|---|---|
| **Deterministic / engineering** | kernel vs reference, causality tests, param counts, FLOP arithmetic, sandbox probes, eval of ONE fixed checkpoint | none (report the tolerance) | 0 |
| **Training comparison** (LM, probe tasks) | E1 vs E10, ternary gap, parity accuracy | seed (a whole training run) | see section 2 |
| **Benchmark / eval comparison** | HumanEval, MBPP, per-doc bits/char of two checkpoints | item (problem, doc) | item bootstrap (section 5) *and* seeds if the claim is about training |
| **Timing / memory** | K1, K2 | repeated measurement | report median + spread of repeats, hardware, clocks |

A benchmark CI over items measures test-set sampling noise only. It says nothing about training noise: a claim "architecture A is better" needs seeds regardless of how many items were scored.

## 2. When a claim may be called VERIFIED (training comparisons)
All of the following, and the claim row must say so:
1. **n >= 3 seeds per arm**, seeds listed, paired by seed where the data order is shared (E10/E1/E11 are; `pairing.md`).
2. **Pre-registered** criteria (committed before the runs), with the metric, split, tokens, LR and the primary vs secondary status of the comparison named. Primary metric on **clean** data (K9: val docs with < 80% 13-gram overlap with train).
3. **Matched tokens; params and FLOPs reported** next to the loss; **tuned or LR-robust** for both arms (the K7 rule), not one arm at a convenient LR.
4. **The 95% CI of the paired (or Welch) difference excludes 0**, with df stated, **and** the point estimate is beyond the practical margin delta registered for the claim (default: 1% relative, or 2 x pooled seed sd if larger).
5. For a **multiplicity family** (section 4), the Holm-adjusted p is < alpha_family.
6. For "no effect / free / no loss" claims: a **TOST** equivalence test with a stated margin. A CI that merely includes 0 never supports "no difference".

Statuses:
| Status | Training-comparison meaning |
|---|---|
| `VERIFIED` | 1-6 all met, where 4 applies to difference claims and 6 to equivalence claims: an equivalence ("no effect / free / no loss") claim is VERIFIED by a passing TOST at its registered margin together with 1-3 and 5, even when the ordinary difference CI includes 0 (as C196). (Clarified v0.5.29 after quick review 2026-11-23 15:34.) |
| `PARTIAL` | n >= 2 seeds, or n = 3 with a CI that includes 0 but the registered rule met; or n = 1 *plus* an independent deterministic confirmation. The evidence column must say which, with the n, the CI and the df. |
| `PARTIAL` + tag **`[exploratory n=1]`** | One seed: the number is a lead, not a result. **A single-seed LM comparison is never VERIFIED.** |
| `UNVERIFIED` | no measurement, or measurement invalid (e.g. contaminated metric, untuned control). |
| `REFUTED` | the registered criterion fails with n >= 3 and the CI excludes the claimed direction; with n < 3 a refutation is also tagged exploratory. |
| `INCONCLUSIVE` | the check ran but cannot decide, because its own registered validity gate failed (e.g. K3's attention control below 90%) or its criterion is indeterminate. This follows the ledger schema (`experiments/claims/README.md`); unlike UNVERIFIED, the evidence exists and is cited. A contaminated metric or an untuned control stays UNVERIFIED. (Added v0.4.108 after claim audit 2026-11-23-1103 found that the policy and the ledger schema disagreed on C16.) |

**External-baseline exception (narrow):** a claim of the form "model M beats a *non-trained-by-us* baseline (xz, SmolLM2) by a large margin" may be VERIFIED from n = 1 if the margin exceeds **10 x** the seed sd measured for M's family and recipe (from a >= 3-seed group, e.g. Phase 2b's E10/E1) and the baseline was re-measured on the same data subset.
**Known property of the registered Phase 2b rule (not changed):** "gap > 2 pooled sd and complete separation" at n = 3 is p ~ 0.07 (Welch), so a registered `VERIFIED` there can still have a 95% CI that includes 0. The ledger evidence must then read "VERIFIED by the registered rule; 95% CI [..] (df=2) includes 0". New registrations should choose a rule that implies CI-excludes-0 (gap > ~2.3 pooled sd at n = 3, or n = 5 seeds).

## 3. Reporting rules (every number that enters a ledger row, devlog, README, site)
- **n always**, the seed list, and the **protocol**: metric and split, tokens, LR, params, FLOPs, code version/run names.
- **Mean +- sd** (sample sd, ddof = 1) for a group; write "sd" or "SEM", never a bare "+-".
- **Difference with a 95% CI and its df** (e.g. "-0.0100 [-0.0148, -0.0052], paired t, df 2"), the **relative gap %**, and an **effect size** (dz for paired, Cohen's d / Hedges' g for unpaired). A p-value only next to the CI, with its test and family.
- n = 1: no CI, no p-value, no "significant", no "beats" without the word "exploratory"; say "one run".
- Report negative and null results with the same detail (a CI that includes 0 and the MDE: what gap *could* have been seen).
- State what was selected after looking (best-of-k LR, best checkpoint, subsets): the winner's number is biased high; confirm on fresh seeds.
- Use the clean metric. Anything on the contaminated val (K9) is labelled so.

## 4. Multiple comparisons
- **Family-wise error budget per arc:** the experiment registry groups experiments into arcs. Each arc gets alpha = 0.05 for its **confirmatory** comparisons, split by **Holm-Bonferroni** over the comparisons *pre-registered* in that arc (m fixed at registration; a comparison that is dropped later still counts).
- **Architecture sweeps (k variants vs a control):** the family is the k variant-vs-control comparisons, Holm over k. "Best of k" is an exploratory result (winner's curse): confirm it with >= 3 fresh seeds as a single pre-registered comparison before any claim.
- **Exploratory comparisons** (anything not pre-registered, subsets, post hoc cuts): Benjamini-Hochberg at q = 0.10, labelled "exploratory", never changes a claim status, only creates queue items.
- **Tuning is not testing:** an LR grid picks a value (K7); it creates no claim, and its winner's loss is not a confirmatory number.
- Practical consequence at n = 3 per arm: with m = 2, alpha_adj = 0.025 => minimum detectable effect 3.7 sd (vs 3.1 sd at 0.05). Keep families small: **a confirmatory family should have <= 3 members.**

## 5. Benchmarks and functional evals
- Resample **items** (problems/docs), never samples within a problem; for pass@k use per-problem scores.
- Two models on the same items: **paired bootstrap** (same item indices drawn for both), BCa interval, B >= 10,000, fixed seed recorded (`core.paired_bootstrap`). Report n_items, the CI of the difference, and the counts of items won by each model.
- Drop contaminated items (HumanEval/MBPP overlap list from C103) as the primary analysis; the full set is the sensitivity analysis.
- Small n_items (< ~100): percentile intervals are unreliable; use BCa and say so; a 0/1 pass@1 over 164 items has a roughly +-8 point CI at 50%.
- Per-item outputs (per-doc NLL, per-problem pass/fail) must be **stored** with the result so CIs can be recomputed.

## 6. Single-seed status of what we have (the exploratory ledger)
Everything produced before Phase 2b on LM training is single-seed (E0-E3, E9; K7 sweep; K4 has 2 seeds). Treated as **exploratory** until replicated under section 2; Phase 2b (E10, E1, E11 at n = 3) is the first confirmatory comparison. The retro-audit is in `experiments/stats/proposed-claims.md`.

## 7. Seeds and what they vary
`train.seed` sets init and data order together (`pairing.md`). Report "seed" as that bundle. Architecture changes change the init draw, so paired designs share only data order unless parameter lists match (E1/E11 do). A `--data-seed` flag (SQ110) would separate them.

## 8. Proposed AGENTS.md lines (section 5, Experiments)
```
- **Statistics policy: `notes/statistics-policy.md` is binding.** A training comparison is VERIFIED only with >= 3 seeds per arm, a pre-registered criterion on the clean metric, and a 95% CI (df stated) that excludes 0 beyond the registered margin; one seed is "exploratory" (PARTIAL at best).
- Every reported number carries n, the seed list and the protocol; differences carry a CI and an effect size, never a bare p-value. Use `experiments/stats/core.py`, not hand-rolled arithmetic.
- Multiple comparisons: the confirmatory family of an arc is fixed at registration and Holm-corrected; sweeps are exploratory until the winner is re-run on fresh seeds; "free/no loss" claims need TOST with a stated margin.
- Benchmarks: bootstrap over items (paired, BCa) and store per-item outputs; item CIs do not substitute for seeds.
- The primary LM metric is bits/char on the clean val (K9: < 80% 13-gram overlap); the training-time `val_loss` is secondary.
```
