# Gogeta-0 devlog

Every entry is one commit and one annotated git tag (rules in [AGENTS.md](AGENTS.md)). Newest first.

## High-level goals

1. **FusionDance:** fuse three efficiency ideas into one causal LLM architecture, and show with **Gogeta-0** (its first model) that they stack rather than fight:
   - **Linear-time sequence mixing** (Gated DeltaNet / Mamba-style state + sliding-window attention): no quadratic cost inside the network, and only one layer whose memory grows with context.
   - **Looped depth**: one shared block applied r times, so reasoning depth comes from compute, not parameters.
   - **Ternary weights** (BitNet b1.58): ~1.6 bits per weight.
2. **Think in loops, not out loud.** Test whether latent loops can replace part of chain-of-thought, trading tokens (memory + quadratic cost) for compute.
3. **The Rival Loop.** Build critique → fix into the recurrence: per-loop supervision, a grounded critic head for halting and feedback, adversarial damage-and-repair training.
4. **Make the loops pay at inference.** Self-speculative decoding: draft with few loops, verify with many.
5. **Do it on a laptop.** Experiments are small enough to train and run locally (RTX Spark N1X via WSL) or on cheap rented GPUs. Small models, clean ablations, honest results, including failures.
6. **Ship it on a gaming GPU.** End the experiments with one final Gogeta model, sized by our own scaling fit and training budget, that runs interactive batch-1 inference with a KV/state cache and lossless speculative decoding on a single 24 GB RTX 3090 or 4090. Plan: [`notes/final-model-plan.md`](notes/final-model-plan.md) (primary target ~1.2B params, the final training run is rented).

**Headline metric:** quality per bit of weight memory and per FLOP, plus how memory and compute *scale with context length*.

---

## v0.5.40: SQ411 closes the __eq__ reward-forgery route (14 arms 1/1 -> 0/1, test_rewards 34/34); scoring invariant for aggregates (SQ407); 20M screens predict sign, understate size (SQ414)
*2026-11-23*

**SQ411 closes the `__eq__` reward-forgery route (14 forgery arms 1/1 -> 0/1, test_rewards 34/34). Scoring is order- and batch-invariant for aggregates (SQ407). 20M screens predict 200M signs but understate effects 1.5-2.5x (SQ414).**

- **SQ411** (worker; the main agent re-ran `post_training.test_rewards`: 34/34 OK):
  - How it works: tests are AST-rewritten in the grader before the fork. Every non-identity comparison and every assert/`not` truthiness goes through a harness-bound strict check over exact builtin value types (or containers of them).
  - Probe (`results/reward_forgery_probe_sq411.txt`, committed harness vs working tree): eq, reflected eq, ne, lt, contains, truthy, not-falsy, int/list/str subclasses, nested list, dict value, chained and and/or all go from 1/1 to 0/1. Honest controls and the SQ373 arms are unchanged.
  - Status: the route no longer blocks GRPO. C83 records the evidence and the routes still open (solution-defined helpers, `.equals`, any/all over forged iterables). Honest solutions returning custom classes now fail closed, by design.
- **SQ407** (CPU fp32, A2s/A2f/E27, 200 docs): doc order leaves per-doc NLL bit-identical. Batch 8 vs 1 moves only a few docs (max relative 7.1e-5 for A2s, 5.0e-4 for E27; median 0), and aggregates are identical to 6 decimals. The A2s-E27 contrast moves 1e-4 log-points against a threshold of 0.1, so published aggregates stand. Per-doc analyses pin batch size 1. The GPU bf16 half is not run.
- **SQ414:** across 11 arms, the in-run 20M rank vs the 200M end gives Spearman 0.84 with 16% of pairs flipped. B1's arms against their 200M counterparts give Spearman 1.0 with no flips. Every named contrast keeps its sign, and magnitudes grow 1.5-2.5x by 200M. So 20M screens are fine for sign and understate size, which fits B1's small 20M effects.
- **SQ225 parseability census:** 99.8-99.95% of docs parse; source and split gaps are below 0.2 points, so no stratum is added.
- **Running (worker):** the per-doc re-score of 30 checkpoints (about 2 h), feeding SQ239, SQ410 and SQ385; then SQ265.
- **Quick reviews:**
  - 17:14: the A2s-A1a comparator check is labelled exploratory on C209; a two-way bootstrap for C214 is added to SQ213.
  - 17:19: SQ418 (tokenizer identity in score artifacts and box checks).
  - 17:24: SQ75 becomes a sequence-length sweep.

---

## v0.5.39: Outside review 17:10: A2 headline holds against fixed-depth E27 (-12.6%); C214 style-shift limitation; decode frontier is E8's next step; readouts frozen before dispatch; erratum on 'vanishes'
*2026-11-23*

**Outside review 17:10: the A2 headline holds against a fixed-depth E27 (-12.6%); C214 gains a style-shift limitation; a batch-1 decode frontier becomes E8's next step; readout scripts are now frozen before dispatch. Erratum: the ternary cost did not "vanish".**

- **Comparator check (review item D, no GPU):** against the fixed-r E27 family A1a (seeds 0-2, 200M, 1.9% more FLOPs), A2s is **-12.56% [-12.98, -12.15]** (Welch, unpaired), and -12.4% on the shared seed 2. About 12.5 of the 16.7 points survive a fixed-depth comparator (fixed r alone is about 3.2%, C212), so the headline is mostly the assembly. Recorded as C209 evidence.
- **C214 limitation:** the SQ99 out-of-corpus files are both unseen and library-style, so C214 cannot separate generalisation from a style shift. New SQ415 (training filtered against the SQ99 set) and SQ416 (a prose slice) address it.
- **E8's next GPU step (queue):** a registered batch-1 decode frontier on a rented 24 GB card. Per-token latency for A2s, A2f, E27, E24 and E20, fp16 plus packed ternary, greedy vs self-speculative. It measures whether loops (r reads of the shared core per token) and ternary cost anything on goal 6's metric.
- **Process rule** (PREDICTIONS.md): every batch registration names its readout script's sha256-16 before dispatch, and in-flight amendments state what had been seen.
- **SQ417:** WSD vs cosine schedule robustness of the assembly gain.
- **Erratum (v0.5.38):** "the ternary cost vanishes" overstates +0.42% [-0.92, +1.77] (n 6, no TOST). The correct wording is "not distinguishable from 0 out of corpus". The README is corrected, and its stale "C213 UNVERIFIED" now reads PARTIAL.
- **Quick review 17:10 (skeptical peer reviewer):** length dependence of C214 and of the ternary result merged into SQ385; C214 stays scoped to the one out-of-corpus set.

---

## v0.5.38: Out of corpus, looped models beat unshared GDN (loop cost reverses: A2f vs E24 -1.91%, C214 VERIFIED); ternary cost vanishes; C213 PARTIAL and its design rule fires; SQ403 fixed-r tuned to its pass count
*2026-11-23*

**Out of corpus, the looped assembled models beat unshared GDN. The loop cost reverses sign (C214, VERIFIED on one out-of-corpus set), and the ternary cost vanishes. C213 goes to PARTIAL, and its design-change rule fires: the final model is judged on out-of-corpus quality. Five more worker results.**

- **SQ400** (registered ~15:20 by the side-quest worker at the main agent's request, committed in v0.5.31 at 15:55, before its first result at 16:34; the main agent recomputed every CI below from `results/sq400.json`). 330-file SQ99 out-of-corpus set, paired-by-seed 95% CIs:

  | Comparison | n | Out of corpus | clean80-300 |
  |---|---|---|---|
  | A2f vs E24 (looped vs unshared, full precision) | 6 | **-1.91% [-3.40, -0.40]** | +2.49% |
  | A3mx vs E24 (all parts vs unshared) | 3 | **-2.82% [-5.04, -0.54]** | +3.68% |
  | A2s vs E24 (ternary looped vs unshared) | 6 | **-1.50% [-2.36, -0.63]** | +11.65% |
  | A2s vs A2f (ternary cost) | 6 | +0.42% [-0.92, +1.77] | +8.94% |
  | A2s vs E27 (assembly gain) | 3 | -11.26% [-11.59, -10.93] | -16.47% |
  | A3mx vs E20 (vs dense) | 3 | +0.81% [-0.24, +1.86] | +0.91% |
  | A2f vs E20 | 6 | +1.45% [+0.47, +2.45] | -0.71% |

  - All seven registered predictions held.
  - **The loop cost changes sign** out of corpus. Scale compression (C213's limitation) can shrink effects toward 0 but cannot flip them, so this is not a compression artifact.
  - **C214 (new, VERIFIED, scoped to this set):** A2f beats E24 out of corpus by about 1.9% at matched FLOPs (registered P7, 6 seeds, CI excludes 0). Its refutation is a second out-of-corpus set (SQ405 temporal holdout, or another package set).
  - **C213** (the costs are mostly in-distribution memorisation): UNVERIFIED -> **PARTIAL**. The registered rule is met, but the mechanism itself is not measured (SQ410 dose-response).
- **Design decision (C213's registered rule fires):** the final model (E61) is sized and judged on **out-of-corpus quality** next to clean80. Ternary and loops are no longer treated as costs on that metric. Measured there, the all-parts model A3mx beats unshared GDN (-2.8%) and is within about 1% of dense E20 (+0.8%, CI includes 0).
- **SQ403** (registered eval-time pass count on A2s): r = 1 is +27.9%, above the registered range; r = 3 is +7.6%, prediction NOT met (it was predicted within 2%). A fixed-r = 2 model is sharply tuned to two passes, so test-time extra passes hurt and an r = 1 draft is off-distribution. That raises the priority of draft-aware or variable-r training (SQ381).
- **SQ392** (registered RoPE positional control): zero-shot linear interpolation makes dense worse everywhere, so by the rule the long-context result stands (the hybrid's 1024-2048 advantage is -78.3%). The control is weak (no NTK/YaRN, no fine-tune).
- **SQ395:** A2s reaches E27's final loss with 54-59% of its tokens, and the ratio holds from step 1,000, so the headline is not an endpoint effect (C209 evidence).
- **SQ377:** duplicating a block never helps (0/27 on E24, 0/11 on E1).
- **Worker:** its list is finished; nothing was committed by it. `experiments/checks/sq339_cpu.py` (its one-line checkpoint-path fallback, used by every CPU side quest) is committed here.

---

## v0.5.37: Claim audit 2026-11-23-1702: 48 claims, 33 SOLID; C104 and C17 to UNVERIFIED; C83/C82 test outputs saved (30/30, smoke PASS); C5/C197/C196 objections rejected with caveats
*2026-11-23*

**Claim audit 2026-11-23-1702: 48 claims out of cool-down, 8 auditors. 33 SOLID, 15 non-SOLID, all checked. Two downgrades (C104, C17). Missing test outputs for C83 and C82 were re-run and saved. Three objections rejected with caveats.**

- **Scope:** 48 claims, 8 Codex GPT-6 Luna auditors (`notes/reviews/claims/2026-11-23-1702-b1..b8.md`), all ok.
- **Downgrades:**
  - **C104** (OpenCoder's annealing corpus not decontaminated): PARTIAL -> **UNVERIFIED**. The ledger cites only a secondary summary and our corpus scan, which cannot establish the paper's procedure.
  - **C17** (the ternary gap shrinks with size): PARTIAL -> **UNVERIFIED**. K4 scored the contaminated val (K9: 41.7% near-duplicates), and policy keeps contaminated metrics UNVERIFIED until a clean80 re-score.
- **Evidence gaps fixed with saved runs:**
  - **C83:** `post_training/test_rewards.py` 30/30 OK (`test-runs/2026-11-23-C83.txt`).
  - **C82:** GRPO/SFT/DPO smoke tests PASS, with every logged number identical to SETUP.md's prose (`test-runs/2026-11-23-C82.txt`).
- **Wording and scope limitations added** (confirmed WEAK, no status change): C109 (lm-eval 0.4.13 scope; stale limitation resolved), C8 (clean-subset scope), C202 (analytic vs actual compute), C106 ("protects" narrowed, plus the /proc gap), C128 (sm_12xa clause), C199 and C198 (no difference detected, not equivalence), C100 (stale pending-output limitation resolved).
- **Rejected, with footers:**
  - C5's downgrade: the registered rule was met, and the 7-seed CI excludes a practical 1% advantage; a caveat is added.
  - C197's per-doc requirement: it is a training claim with seed as the unit, and section 5 covers benchmarks.
  - C196's downgrade: the proposition is scoped to lr 6e-3; a caveat is added.
- **Quick reviews:**
  - 16:59: stale E11/Q17 queue items fixed (E11 is C196); SQ414 (do 20M screens predict 200M?).
  - 17:04: in its file.

---

## v0.5.36: GPU batch B1 read out: MoE core matches dense-2048 at 31% fewer FLOPs; learned GDN beta not needed; loop cost not an LR artifact; unique params beat a second pass ~2.4x per FLOP (20M screens)
*2026-11-23*

**GPU batch B1 read out (20M screens, registered). The MoE core matches a dense FFN of equal width at 31% fewer FLOPs. GDN's learned write strength is not needed. The loop cost is not a learning-rate artifact. Per FLOP, unique parameters beat a second loop pass about 2.4x. No Holm gate opens.**

`experiments/checks/b1_readout.py` -> `results/b1_readout.json`. 25 runs, 20M tokens of code-py-v2, clean80 paired by seed, provenance OK (907065f for the 23 registered jobs, c384e67 for SQ350). Same-seed replay floor 0.2-0.3% (SQ232). Every contrast is an exploratory screen (B1 amendment 2).

| Contrast | Seeds | Result [95% CI] | Predicted | Reading |
|---|---|---|---|---|
| SQ401: A3m (ternary MoE core) vs A2s | 3 | -1.01% [-1.83, -0.18] | -3% (-5 to -1) | in range (edge); C206's direction at 20M |
| **G1** SQ401: dense core FFN 2048 vs A3m | 3 | +0.15% [-0.88, +1.20] | -1% (-3 to +1) | a tie, but dense uses **45% more FLOPs** (5.49e15 vs 3.79e15): the MoE buys equal-width capacity per FLOP |
| SQ402: GDN beta fixed 1.0 vs learned | 2 | -0.47% | +1.5% (0 to +4) | **out of range**: no cost |
| SQ402: beta fixed 0.5 vs learned | 2 | -0.06% | +1% (-0.5 to +3) | in range; both within 0.5%, so the learned write strength is not needed at 20M |
| SQ404: A2f vs E24 (loop cost at 20M) | 2 | +1.80% | +2% (0 to +5) | in range |
| **G2** SQ404: best core LR (x0.5) vs x1 | 2 | -0.16% | closes < half | closes **9%** of the gap (x2: +7.19%), so the loop cost is not an LR artifact and there is no 200M re-run |
| **G3** SQ350: A2f 6x2 vs 6x1 (equal params) | 2 | -1.33% [-3.59, +0.99] | -4% (-8 to -1) | in range (edge); E24 is about -3.1% vs 6x1 for the same extra FLOPs, so unique params buy about 2.4x more per FLOP |

- **Holm (G1-G3, m = 3):** no gate rejects (p 0.09 / 0.51 / 0.59), as expected at df 1-2. No ledger status changes.
- **Ledger evidence:** C206 (MoE vs dense capacity) and C197 (unique params beat loops per FLOP, now on the assembled shape too).
- **Disclosure:** SQ404's seed-0 runs had finished on the box (unread) before B1 amendment 3 was pushed at 16:40. That amendment changed only the reading, not the rule.
- **Design implications:**
  - Loops' cost is a unique-parameter cost, as C197 says, not an optimizer artifact. The remaining levers are pass identity (SQ408), capacity in the core (the MoE already pays its way per FLOP), and the horizon (SQ380).
  - A fixed GDN beta is a candidate simplification, pending a 200M check.
- **Fleet:** one box for 1.0 h, destroyed at 16:55; 0 instances. X7 = $0.74, total spend $44.01.
- **Quick reviews:**
  - 16:39: target workflow stays prefix completion; distractor NLL added to SQ385.
  - 16:44: **SQ411 blocks GRPO**, closing the documented `__eq__` reward-forgery route; SQ412 adds a cockpit spend readout.
  - 16:49: gradient interference is already SQ265, now extended to A2f/A2s.
  - 16:54: the B1 bundle's provenance is described in the triage footer.

---

## v0.5.35: Outside review 16:37: SQ404's larger-step premise withdrawn (AdamW scale-invariant; clipping never fires), B1 amendment 3 before any SQ404 result; SQ410 overlap dose-response
*2026-11-23*

**Outside review 16:37: SQ404's "larger step" premise withdrawn (AdamW is scale-invariant and clipping never fires), registered as B1 amendment 3 before any SQ404 result. Overlap dose-response added as SQ410.**

- **Outside review** (`notes/reviews/2026-11-23-1637-review.md`, Haiku), triaged in its footer:
  - **SQ404's premise fails.** AdamW normalises each parameter's update by its own gradient RMS, so summed shared gradients do not enlarge the step. The only scale-dependent path, global clipping at 1.0, never fires after warm-up: a log scan of 33 pulled 200M runs (A2f, A2s, A3mx, E20, E24, E27) finds no logged step after step 300 above the clip, with median grad norm 0.11-0.14. This descriptive scan ran before registration and is disclosed as such. There is no clipping component in the loop cost.
  - **B1 amendment 3** (registered before any SQ404 result; rule unchanged): SQ404 now reads as "does the shared core's best LR differ from the global LR?", with the 0.5 arm being the core at E24's LR. The readout shows the paired per-seed differences, because df 1 gates are judged on point estimates.
  - **SQ410:** overlap dose-response of each assembly cost (C213's mechanism), on CPU after SQ239.
  - **Merged:** position and length profiles into SQ385; a same-denominator check into SQ387's precondition.
  - **Rejected:** mean vs sum over passes. Under AdamW without clipping it differs only through eps.
- **Quick review 16:34 (newcomer):** the README's first-run path exists (steps 1-5). The current thesis test is C208 (the all-parts model vs E24/E20).

---

## v0.5.34: Docs refresh (hourly); B1 amendment 2: Holm family for B1's gating contrasts, registered before any B1 result
*2026-11-23*

**Docs refresh (hourly). B1 amendment 2 (the Holm family for B1's gating contrasts) registered before any B1 result.**

- **B1 amendment 2** (PREDICTIONS.md; quick review 16:14), committed while B1 is still training and before any result is read:
  - The gating contrasts form one Holm family, alpha 0.05: G1 SQ401 dense-2048 vs A3m, G2 SQ404 best core multiplier vs 1 (gap closure against E24), and G3 SQ350 A2f(6x2) vs A2f(6x1).
  - Everything else is an exploratory readout, reported unadjusted.
  - No B1 contrast can promote a claim: 20M screens with 2-3 seeds can only open a follow-up or add a limitation.
- **Docs** (Haiku; diff checked by the main agent; statuses match the ledger; no B1 result stated):
  - README status: the compute-matched A2 headline (-16.2% [-16.9, -15.5], v0.5.30), spend $43.27, B1 running, C213's confound; a new results bullet for C212 (PARTIAL, fresh seed -3.1%).
  - The sketch's status and decisions log go to v0.5.33.
  - The campaigns README gains a B1 row. SETUP.md is unchanged because it is current.
- **Quick reviews:**
  - 16:14: the Holm family above.
  - 16:19: micro-batch invariance is already tested (`tests/test_options.py`).
  - 16:24: run order and clocks cannot move loss; the SQ232 same-seed replay floor (0.2-0.3%) is the noise floor for B1.
  - 16:29: parseability census, SQ225.
- **B1 progress:** 8 of 23 runs done by 16:25, all exit code 0, about 2 min each.

---

## v0.5.33: Outside review 16:07: SQ350 parameter-matched loop control added to the running B1 batch (amendment 1); SQ408, SQ409; C212 selection-seed limitation; B1 box training
*2026-11-23*

**Outside review 16:07: SQ350 (parameter-matched loop control) promoted into the running B1 batch as amendment 1. Two new side quests. C212 gains a selection-seed limitation. The B1 box is up and training.**

- **B1 status:** the box (RTX 5090) passed its GPU tests at v0.5.31: `test_model` 12/12 and `test_assembly` 16/16, including the new switches on CUDA. v1/v2 data are rebuilt and verified. The worker launched at 16:11 with the 23 registered jobs.
- **Outside review** (`notes/reviews/2026-11-23-1607-review.md`, Haiku), triaged in its footer:
  - **B1 amendment 1 (registered before these runs): SQ350.** A2f at one pass (6 x 1, same unique params as A2f, half the core FLOPs), seeds 0-1, queued on the B1 box (config `b1/sq350_a2f_r1.json` [603ff93bdcc9d59e]). With B1's A2f (6 x 2) and E24 (12 x 1), it splits the D3 loop cost into a loop effect at equal params and a unique-param effect at equal FLOPs. Prediction: A2f(6x2) vs A2f(6x1) = -4% (-8% to -1%).
  - **C212 limitation:** seeds 0-1 both chose and tested A1a, so the CI is not a fresh-seed interval. The fresh seed alone gives -3.11%.
  - **New:** SQ408 (zero-parameter conjugated passes, pass identity) and SQ409 (line-permutation re-score: does the assembly gain need document structure?).
  - **Merged:** an SWA-only arm into SQ392 (window vs RoPE at 2,048), and a shared-vs-unshared ternary threshold sweep into SQ365.
  - **Partly rejected:** "E27 is untuned". Its LR is bracketed (v0.5.13); it is the naive fusion with default choices, as C209 states.
- **Queue:** B1 added and running. E8 is corrected from "running" to "queued", since no GPU job ran for it; its batch-1 bench waits for a 24 GB card.
- **Quick reviews:**
  - 16:04 (competitor lab): renaming is already in SQ194; a large renaming gap would narrow C209's scope, not refute it.
  - 16:09 (systems): throughput is a secondary readout; per-run tokens/s are logged, but GPU clocks and temperature are not.

---

## v0.5.32: Claim audit 2026-11-23-1602: C213 gains a scale-compression confound (stays UNVERIFIED); B1 decode tests pass
*2026-11-23*

**Claim audit 2026-11-23-1602: one claim due (C213). A scale-compression confound is added; the hypothesis stays UNVERIFIED. The B1 decode tests pass.**

- **Scope:** one claim was due (C213, the out-of-corpus hypothesis); the other 140 are cooling down. One Codex GPT-6 Luna auditor (`notes/reviews/claims/2026-11-23-1602-b1.md`), ok.
- **C213 (WEAK, confirmed):** the numbers match `sq379.json`, but the causal wording "memorisation capacity" is not established. Out-of-corpus losses are 6-7x higher (1.65-1.9 bits/char), which compresses every relative effect toward 0. A limitation now records the confound. The attenuation is differential (the assembly gain keeps about two thirds of its size while the ternary cost goes to about 0), so compression alone does not explain it. No downgrade: C213 was already UNVERIFIED. SQ400 must also report absolute bits/char differences.
- **v0.5.31's pending test:** `tests/test_decode.py` ALL PASS with the new `gdn_beta_fixed` branch. Every suite touched by B1's switches is green (assembly 16/16, moe 15/15, c2f 17/17, decode, hf 11/11).
- **B1 fleet:** one RTX 5090 (Vast, $0.66/h with storage) rented at about 15:57 and bootstrapping at v0.5.31.
- **Quick review 15:59 (ML theorist):** pointer chasing is added to SQ399. The screen-to-scale rule was already registered. The claim whose refutation would most change direction is C197 (unique parameters beat loops at matched FLOPs).

---

## v0.5.31: GPU batch B1 registered: SQ401 (MoE core vs dense capacity), SQ402 (fixed GDN beta), SQ404 (core LR multiplier); three off-by-default switches added and tested
*2026-11-23*

**GPU batch B1 registered before dispatch (the human: "Yes, rent one"): SQ401 (is the MoE core's gain capacity?), SQ402 (does GDN's learned write strength matter?), SQ404 (is part of the loop cost step size?). It adds three switches, all off by default.**

- **New switches** (defaults bit-identical; `to_dict()` omits them at their defaults, so existing config hashes are unchanged):
  - `ModelConfig.core_mlp_hidden` (SQ401): the shared core's dense SwiGLU width, with prelude and coda keeping `mlp_hidden`. FLOPs follow from the actual parameter counts.
  - `ModelConfig.gdn_beta_fixed` (SQ402): a constant GDN write strength instead of sigmoid(b(x)). It is mutually exclusive with `gdn_neg_eigval`, and the decode path honours it.
  - `train.core_lr_mult` (SQ404): an LR multiplier for the shared core's parameters only, via `gogeta.train.param_groups`. The default returns the original two AdamW groups.
  - Both new model fields are also added to the Hugging Face wrapper's config (`gogeta/hf.py`), so the v0.5.0-v0.5.23 break (C80) cannot repeat.
- **Tests** (CPU): `tests/test_assembly.py` 16/16, including three new tests:
  - core width: equal width gives identical output, only the core widens, params and FLOPs rise;
  - fixed beta: it changes the output, `b` gets no gradient, the neg-eigval combination raises (torch reference recurrence);
  - core LR groups: they cover every parameter, and the core gets the multiplier.

  `tests/test_moe.py` 15/15, `tests/test_c2f.py` 17/17, `post_training/test_hf.py` 11/11. `tests/test_decode.py` is running and is reported in the next version.
- **Registered** (PREDICTIONS.md, "GPU batch B1"), 20M tokens, exploratory screens, paired by seed on clean80:
  - **SQ401** (seeds 0-2): A2s vs A3m vs A2s with a dense core FFN of 2048. Prediction: dense-2048 vs A3m -1% (-3% to +1%). If dense matches or beats the MoE, C206's step is capacity.
  - **SQ402** (seeds 0-1): E1 with learned beta vs beta fixed at 1.0 and 0.5. Prediction: +1.5% and +1%. Within 0.5% means gating carries the mixer.
  - **SQ404** (seeds 0-1): A2f with core LR x0.5 / x1 / x2 vs E24. Prediction: A2f vs E24 +2% at 20M, with the best multiplier (0.5) closing less than half of it. If more than half closes, the comparison is re-run at 200M.
- **Configs:** `experiments/staged-configs/b1/` (5 new, sha256-16 in the registration). The batch is 23 runs on one rented RTX 5090, about 1-1.5 h and about $1.
- **Quick reviews:**
  - 15:44: path-class fixture audit (SQ61).
  - 15:52: claim promotion gate (SQ274) and scorer invariance for ternary arms (SQ407).

---

## v0.5.30: A2 headline holds at matched compute: A2s vs FLOP-matched E27 -16.18% [-16.87, -15.48] (A2 amendment 3, C209); assembly fleet torn down ($13.09)
*2026-11-23*

**A2 headline holds at matched compute: A2s beats a FLOP-matched E27 by -16.18% [-16.87, -15.48]. The assembly fleet is fully torn down ($13.09 for A1-A4).**

- **A2 amendment 3, final** (registered v0.5.22; `experiments/checks/a2_compute_matched.py` -> `results/a2_compute_matched.json`; provenance 4c716bd, code-py-v2):
  - Setup: E27 at 221.6M tokens, which matches A2s's training FLOPs (3.78-3.80e16 vs 3.79e16), seeds 2-4, paired with A2s at 200M on clean80.
  - Result: **A2s - E27_cm = -16.18%, 95% CI [-16.87, -15.48]**, per seed -16.45 / -15.89 / -16.21, df 2, p 1.2e-4. Prediction -15.5% (-17% to -13%): in range.
  - Giving E27 its missing 10% of compute moves the headline by about half a point (from -16.67%). The CI lies far below C209's -5% refutation bound.
  - C209 gains the compute-matched evidence event, and v0.5.22's erratum is resolved: the gap is not a compute artifact.
- **Fleet:** every A3/A4 box is destroyed (15:46, "ALL A3 BOXES DESTROYED"), 0 instances on Vast, and no controller loop is left running.
- **Expenses:** X6 (assembly A1-A4, including the LR brackets and the compute-matched runs) is final at **$13.09**, inside the agent's $25 self-cap. Reconciliation at 15:47: total spend $43.27, credit $56.73.
- **Quick reviews:**
  - 15:39: fill-in-the-middle becomes SQ406; left-to-right completion stays the target.
  - 15:44: a synthetic-fixture audit of readable host paths, added as an SQ61 extension; the sandbox's in-scope path classes are stated.

---

## v0.5.29: Outside review 15:37: optimizer confound of weight sharing (SQ404), temporal holdout (SQ405); statistics policy clarified for equivalence claims (TOST)
*2026-11-23*

**Outside review 15:37: the optimizer confound of sharing (SQ404, top pick), temporal holdout (SQ405), threshold sensitivity and a training-free halting signal folded into existing items. Statistics policy clarified for equivalence claims.**

- **Outside review** (`notes/reviews/2026-11-23-1537-review.md`, Haiku), triaged in its footer:
  - **SQ404 (top pick):** shared weights sum their gradients over passes at one global LR, so they step larger than unshared weights. The loop cost that decides D3 (A2f vs E24 +3.2%) may partly be step size. The test is a core-only LR multiplier of 0.5/1/2 vs unshared, at 20M tokens, 2 seeds. It needs a per-group LR option, and its prediction is registered before it runs.
  - **SQ405:** temporal holdout (post-snapshot code). It needs the human for any download.
  - **Merged:** a training-free halting signal into SQ403 (per-token KL between r = 1 and r = 2 vs the loss gain); clean50/clean80/full-val sensitivity of the A2/A3 headlines into SQ239; a target-alignment audit as a precondition of SQ387.
  - **Rejected after checking the code:** the claim that the compute-matched readout is on a different scale from C209. Both store `expm1` of the mean log ratio. The provisional matched gap (n 2) is -16.2% [-19.6, -12.5], against C209's -16.67%.
  - **Covered:** the seed-variance floor (SQ388 uses Phase 2b's 7-seed groups) and data-matching E27_cm (the registered match is on FLOPs).
- **Statistics policy, clarified** (quick review 15:34): criterion 4 (CI excludes 0) applies to difference claims and criterion 6 (TOST) to equivalence claims. An equivalence claim is VERIFIED by a passing TOST at its registered margin even when the difference CI includes 0, as C196 already was. No claim's status changes.
- **Quick reviews triaged, all footers in `notes/reviews/quick/`:**
  - 15:10: identifier renaming, SQ194; C213's design-change threshold sent for SQ400's registration.
  - 15:14: the 24 GB bench target is a rented 3090/4090, in E8's queue note.
  - 15:19: leave-one-package-out is required in SQ400.
  - 15:24: input manifest, SQ299.
  - 15:29: the main target is quality per FLOP and per byte on code.
  - 15:34: bootstrap calibration, SQ213.
- **Running:** compute-matched E27 seed 4 (A2 amendment 3, last box).

---

## v0.5.28: Docs refresh (hourly); A3 amendment 2: A2f LR-robust, ternary cost +9.5% at each arm's best LR
*2026-11-23*

**Docs refresh (hourly), plus A3 amendment 2: A2f is LR-robust, so the ternary cost is +9.5% at each arm's best LR.**

- **Docs** (Haiku, diff checked by the main agent; every cited status matches `experiments/claims-status.md`):
  - README and `notes/sketch.md` brought to v0.5.27: C206-C208 VERIFIED again (v0.5.23), C210/C211 UNVERIFIED (v0.5.25), C213 as an out-of-corpus hypothesis with SQ400, the A2 headline's compute erratum (v0.5.22), and the hf.py fix (C80, v0.5.24).
  - The sketch's decisions log gains v0.5.22-v0.5.27, and its claim counts match the ledger (141 claims).
  - `post_training/SETUP.md` gains the wrapper breakage and fix and the current test counts (test_hf 11, test_rewards 30).
  - The campaigns README is refreshed.
- **A3 amendment 2** (registered v0.5.22; A2f seed 5, 200M, clean80; `results/a3_assembled.json`):

  | | 3e-3 | 6e-3 | 1.2e-2 |
  |---|---|---|---|
  | A2f | 0.25104 | **0.24921** | 0.25853 |
  | A2s vs A2f (ternary cost) | +8.71% | +10.15% | +9.73% |

  A2f is LR-robust at 6e-3, which is its best. At each arm's best LR the ternary cost is +9.51%, so the full-precision gap is not an LR artifact. Recorded as evidence on C207.
- **A2 amendment 3** (compute-matched E27, seeds 2-4) is running; seed 4 is on the last box. The readout script `experiments/checks/a2_compute_matched.py` is committed, and its result will be read out when the last seed lands.

---

## v0.5.27: Outside review 15:07: dense capacity control for the MoE core (SQ401), GDN beta ablation (SQ402), eval-time pass count on A2s (SQ403, registered); init depth checked, no confound
*2026-11-23*

**Outside review 15:07: a dense capacity control for the MoE core (SQ401), a GDN beta ablation (SQ402) and an eval-time pass-count check on A2s (SQ403, registered). The init-depth concern was checked and is clean.**

- **Outside review** (`notes/reviews/2026-11-23-1507-review.md`, Haiku), triaged in its footer:
  - **SQ401:** A2s with a dense 2048-hidden core FFN vs the MoE core (A3m). C206's -3.3% is measured against a dense FFN with half the core FFN params, and C203 found a param-matched dense FFN beats the MoE on the E1 stack. If dense ties or wins, the MoE step is capacity, not routing.
  - **SQ402:** GDN with fixed vs learned write strength (beta).
  - **SQ403** (prediction registered in PREDICTIONS.md): clean80 at eval r = 1/2/3 on the A2s checkpoints. Prediction: r = 1 is at least 5% worse and r = 3 within 2% of r = 2.
  - **Init depth vs trained r, checked now:** no confound. The A2/A3 configs set loops = loops_min = loops_max = 2. The random-r arms use loops = 4, the mean of U{1..7}. A1a and A1i share loops = 4, so C212 has no init difference.
  - **Agreed as reading rules:** ternary arms are counted at full-precision FLOPs, so "matched compute" means quality at equal analytic FLOPs and ternary's benefit is bytes. The assembly headlines are FLOP-matched, not param-matched; each claim's scope states its params.
  - **Not added:** the memory-matched ternary control duplicates SQ393.

---

## v0.5.26: Worker results: ternary and loop costs nearly vanish out of corpus while the assembly gain survives (SQ379, hypothesis C213; SQ400 registered follow-up); coda attention carries long-range copy (SQ383); ternary loops not contractive (SQ353); packed ternary 16x (SQ378)
*2026-11-23*

**Worker results: out of corpus, the ternary and loop costs nearly vanish while the assembly gain survives (SQ379, hypothesis C213). The coda's full attention carries copy-at-distance (SQ383). Ternary loops are not contractive (SQ353). Packed ternary works (SQ378).**

All six are descriptive side quests by the Sonnet worker, with no registered predictions. The main agent spot-checked the headline numbers against the result files (`experiments/checks/results/sq3xx.json`; devlogs `experiments/devlogs/SQ3xx.md`).

- **SQ379, A2 out of corpus** (330 SQ99 files, 31 packages; seeds 2-4; package-clustered 95% CIs, which do not include seed variance):

  | Comparison | Out of corpus | clean80 |
  |---|---|---|
  | A2s vs E27 (assembly) | -11.26% [-12.02, -10.56] | -16.7% |
  | A2s vs A2f (ternary cost) | +0.22% | +9.5% |
  | A2f vs E24 (loops vs unshared) | -0.41% [-0.69, -0.13] (n 2) | +3.2% |
  | A2f vs E20 (vs dense) | +1.65% [+1.21, +2.15] | -0.9% |
  | E27 vs E20 (full fusion) | +14.8% | +28.7% |

  **The assembly gain survives out of corpus. The two costs, ternary and loops' unique-parameter deficit, nearly vanish, and dense parity does not hold.** This is recorded as **hypothesis C213** (UNVERIFIED): the costs are mostly in-distribution memorisation capacity. **SQ400 (top pick)** is its registered out-of-corpus replication with seed variance, adding A3mx.
- **SQ383, coda attention and copy-at-distance** (A2s, seeds 2-4): with the coda's full attention swapped for a 256-token window, second-copy NLL changes 0.0% at gap 128, **+41.9% [+34.9, +48.9] at gap 256** and +13.3% [+7.8, +18.8] at gap 512, back to the no-copy level, while bits/char moves only +2.76%. The one full-attention layer outside the loop carries long-range copy, so the design keeps it.
- **SQ353, loop stability** (55 looped checkpoints plus 8 E24):
  - Full-precision loops are contractive: relative perturbation gain 0.45-0.85 (E22), 0.57/0.74 (A2f).
  - Ternary loops are not: gain at eps 1e-2 is 1.17-0.92 for E27 and 1.59/1.10 for A2s. At eps 1e-3 the gain is about 8.6x larger, which is quantiser-flip noise.
  - Adjacent-pass CKA is 0.92-0.995 in every loop, vs 0.77 between adjacent unshared E24 blocks.
  This is a mechanism for the ternary cost inside loops, and it supports the full-precision mixer fix (C207).
- **SQ378, ternary packing** (A2s s2): the 36 BitLinear layers (11.2M weights) pack 16.0x (44.8 MB fp32 -> 2.80 MB), and the whole file is 0.42x the checkpoint. Packed reload gives -0.004% bits/char vs fake-quant: 99.82% argmax agreement, not bit-identical. **A naive dequant through stock BitLinear gives +107%**, a trap documented in the devlog. This makes C208's 1.58-bit memory figures measured for A2s's ternary layers.
- **SQ382, MoE routing across passes** (A3m, A3mx, 3 seeds): top-1 expert agreement between pass 1 and pass 2 is 0.73 / 0.71 (chance 0.25), top-2 overlap 0.80. Load is balanced (max share about 0.27). Each pass partly reuses its experts.
- **SQ376, data gate:** in-training val loss still falls 22.0% (E1) and 19.9% (E10) from 100M to 197M tokens, confounded by the cosine schedule ending at 200M. The data budget is not saturated.
- **Also:**
  - the 15:04 quick review (ML theorist) is triaged: SQ399 adds bracket-depth extrapolation for goal 2;
  - the worker's log is updated;
  - SQ377 (block duplication) is still running.

---

## v0.5.25: Claim audit 2026-11-23-1502: C210/C211 to UNVERIFIED until the byte decoder's clean80 score (SQ387); C209 LR objection rejected (A2s bracket exists); C212 SOLID
*2026-11-23*

**Claim audit 2026-11-23-1502: four auditors on the four newest claims. C210 and C211 go to UNVERIFIED until clean80; C209's LR objection is rejected with the bracket as evidence.**

- **Scope:** four claims were due (C209-C212); the other 136 are cooling down. One Codex GPT-6 Luna auditor per claim (`notes/reviews/claims/2026-11-23-1502-b1..b4.md`), all ok.
- **Downgrades:**
  - **C210** (byte decoder +14.4%, WEAK, confirmed): PARTIAL -> **UNVERIFIED**. The A4 registry entry itself says "clean80 via SQ387 before any claim", and the policy requires the clean metric. The +14.39% stands as a measured result on the registered in-training v2 val metric.
  - **C211** (second shared byte pass -1.7%): PARTIAL -> **UNVERIFIED**, on the same grounds, applied for consistency. A limitation is also added that "deep supervision adds nothing" only means no benefit was detected (+0.55%, no TOST).
- **Rejected:** C209 (A2 headline) WEAK for "no A2s LR robustness". The robustness exists: A3 amendment 1 bracketed A2s, and 6e-3 is within 0.58% of its best (3e-3), so at its best LR the effect would be about 0.6 points larger. An evidence event now cites it, and the status stays VERIFIED.
- **SOLID:** C212 (fixed vs random depth, PARTIAL: selection seeds, LR not bracketed).
- **README:** C210's status wording is updated to match.
- **Quick review 14:59 (cheapest-experiment hunter):** a known-answer scorer fixture is added to Q15, and Q16's GPU check waits on it.

---

## v0.5.24: Fix: Hugging Face wrapper broken since v0.5.0 (missing assembly fields; hidden() mirror ignored them); C80 re-verified with saved output (11/11, 6/6)
*2026-11-23*

**Fix: the Hugging Face wrapper (`gogeta/hf.py`) could not build any model from v0.5.0 to v0.5.23. C80 records the break and is re-verified with saved test output.**

- **Found by** quick review 14:54 (reproducibility), which asked for C80's saved test output. The re-run at v0.5.23 failed at setUpClass with `GogetaConfig is missing ModelConfig field 'loop_embed'`.
  - v0.5.0 (36acccc) added the eight assembly switches to `ModelConfig` but not to the wrapper's config. The wrapper's guard failed loudly, so nothing diverged silently.
  - No training or evaluation result used the wrapper; it is the post-training (TRL/GRPO) entry point.
  - The wrapper's `FusionDanceBackbone.hidden()` was also a hand-written mirror of `FusionDance.forward`. It would have ignored loop_embed, contractive injection, adapters and the ternary scope even once the config built.
- **Fix:**
  - `GogetaConfig` gains the eight fields, with defaults equal to `_ASSEMBLY_DEFAULTS`.
  - `hidden()` now delegates to `FusionDance.forward(..., return_hidden=True)` (the v0.5.12 hook), so the mirror can no longer drift.
  - New test `TestAssembledConfig` checks exact logit equality (max diff 0.0) for a config with loop_embed, contractive inject, rank-4 adapters, ternary_fp_ends and ternary_mixer=False.
- **Saved evidence** (`experiments/checks/results/test-runs/2026-11-23-C80.txt`): `post_training/test_hf.py` 11/11 OK (venv gogeta-rl, including TRL's fused-head test); `evals/test_wiring.py` 6/6 OK (venv gogeta-eval).
- **Ledger:** C80 goes PARTIAL ("true at v0.4.25; broken v0.5.0-v0.5.23"), then VERIFIED on the saved run. Its prose-only limitation is resolved; "GDN through the wrapper on CUDA untested" stays.
- **Process lesson:** SQ274 gains an extension. The release lint should re-run implementation claims' tests, not only check that their evidence pointers exist. C80 stayed VERIFIED for 23 versions while its test could not build the model.

---

## v0.5.23: A3 LR bracket passes (A2s and A3mx robust at 6e-3): C206-C208 back to VERIFIED; sandbox /proc gap confirmed and filed (SQ61)
*2026-11-23*

**A3 LR bracket passes: C206-C208 return to VERIFIED. The sandbox's /proc gap is confirmed and filed.**

- **A3 amendment 1, final** (registered v0.5.18; seed 5, 200M tokens, clean80; `experiments/checks/results/a3_assembled.json`, key `A3_amendment1_lr_bracket`):

  | Arm | 3e-3 | 6e-3 | 1.2e-2 | Best | 6e-3 vs best |
  |---|---|---|---|---|---|
  | A2s | 0.27291 | 0.27449 | 0.28367 | 3e-3 | +0.58% |
  | A3mx | 0.25333 | **0.25176** | 0.25793 | 6e-3 | 0 |

  Both arms are LR-robust at 6e-3 by the registered rule (within 1% of the best), so **C206 (ternary MoE core -3.3%), C207 (ternary cost splits evenly) and C208 (all-parts model's standing) return to VERIFIED**. A3mx is at its best LR, so its +4.35% gap to the tuned E24 is not an LR artifact. A limitation is added: A2s's 3e-3 is 0.58% better, so effects against A2s may be overstated by up to about 0.6 points, and no decision changes.
- **Still running on the last box:** A3 amendment 2 (A2f at 3e-3 / 1.2e-2), then A2 amendment 3 (E27 at 221.6M tokens, seeds 2-4).
- **Quick review 14:47 (safety/sandbox):** its `/proc` question was checked directly. Under the sandbox's `unshare -rnm`, a child sees all 32 host processes and can read every cmdline (no PID namespace). Filed as an SQ61 extension: `unshare -rnmpf --mount-proc` plus a fail-closed probe, with env/fd sentinel tests. C106's "not a security boundary" limitation stands. No model-generated code has been executed at scale, and Q14 stays human-blocked.

---

## v0.5.22: Outside review 14:41: compute-matched E27 for the A2 headline and an A2f LR bracket registered and queued; expenses reconciled ($42.75 total); erratum: A2 headline not at matched FLOPs
*2026-11-23*

**Outside review 14:41: the A2 headline gets a compute-matched control and A2f an LR bracket (both registered and queued); expenses reconciled ($42.75 total, $12.57 for assembly); four new side quests. Erratum: the A2 headline was never "at matched FLOPs".**

- **Outside review** (`notes/reviews/2026-11-23-1441-review.md`, Haiku), triaged in its footer:
  - **A2 amendment 3** (registered before running): E27 at 221.6M tokens (FLOP-matched to A2s), seeds 2-4. Prediction: A2s - E27 = -15.5% (-17% to -13%). C209 is REFUTED by its own criterion if the CI reaches -5%.
  - **A3 amendment 2** (registered): A2f at 3e-3 and 1.2e-2, seed 5, so the ternary cost (A2s vs A2f) is fair on LR too.
  - Both run on the A3 fleet's last box after the A3 amendment 1 pair.
  - **SQ392 protocol clarification** before it runs: max_seq_len 2048 in both dense arms.
  - **New side quests:** SQ395 (compute-to-target from existing curves), SQ396 (minimal-pair likelihood), SQ397 (learned global tokens for SWA), SQ398 (A2s vs E27 at 20M). The token-class reweighting idea is merged into SQ318.
- **Erratum (v0.5.10):** "the A1 winners add up to -16.7% at matched FLOPs" is wrong. A2s uses about 10% more training compute than E27 (3.79e16 vs 3.41-3.43e16; C209). A2 amendment 3 measures the compute-matched gap.
- **Expenses reconciled** (`experiments/expenses.json`): there were no entries after 2026-11-22 22:35. Added from provider credit:
  - X5: fleet wave 4, $23.22, including the ~$8.5 idle time disclosed in v0.4.103;
  - X6: the assembly waves A1-A4, $12.57 to 14:45;
  - a reconciliation at 14:45: total spend $42.75, credit $57.25.
  The assembly phase is under the agent's self-imposed $25 cap, within the human's 5090-or-better grant (v0.5.0).
- **Queue:** A3 and A4 are marked done, and E8 stays running (decode built, GPU speedup not measured). QUEUE.md is re-rendered.
- **Quick review 14:41 (statistician):** leave-one-seed-out on every A3/A4 contrast. No sign flips and no decision changes. `a3_assembled.py` and `a4_fullstack.py` now write `loso` to their result JSON. The LR bracket is paired on data order, because the batch sampler has its own seeded generator (`gogeta/train.py:61-65`).

---

## v0.5.21: Docs refresh (hourly)
*2026-11-23*

**Docs refresh (hourly): README, sketch and campaign notes brought up to v0.5.20; five README result statements checked against their ledger status.**

- **README:**
  - The status paragraph now carries the A3 and A4 finals and the speculative-decoding result. "Ties dense" is gone everywhere: the full-precision A2f is "0.9% below dense in point estimate, no equivalence test", and A3mx is "not shown to tie dense" (v0.5.19 erratum).
  - A new results bullet covers A3/A4 (C208, C210 PARTIAL; SQ394).
  - The Byte Quest table no longer says the 200M E13 runs are unanalysed (SQ374, v0.5.15).
- **Status-wording check** (quick review 14:24), five README statements vs `experiments/claims-status.md`. Four were fixed and one already matched:
  - "Loops don't pay yet" now says C6 PARTIAL.
  - The A2 headline now says C209 VERIFIED at recipe level (it said "exploratory").
  - "Ties dense" became a point estimate.
  - The GRPO-block bullet now says C83 PARTIAL.
  - "Does not stack" (C202 PARTIAL) already matched.
  The long-context bullet is labelled "no ledger claim yet".
- **notes/sketch.md:** status rewritten from v0.5.14 to v0.5.20, with claim counts matching the ledger (140 claims). Seven decisions-log entries were added (v0.5.10-v0.5.20).
- **experiments/launch/campaigns/README.md:** the wave row now covers A1-A4, and the stale QUEUE timestamp is corrected.
- **Checks:** the main agent verified the diff, and every cited status matches the ledger. `post_training/SETUP.md` was already current.

---

## v0.5.20: A4 final: full stack not viable yet (byte decoder +14.4% vs BPE head; second shared byte pass -1.7%); fixed loop depth beats random -3.2% (3 seeds); self-speculative decoding built, not lossless under int8 activations
*2026-11-23*

**A4 final: the full stack is not viable yet. The byte decoder costs +14.4% against the BPE head. The second shared byte pass works (-1.7%). Fixed loop depth beats random (-3.2%, 3 seeds). Self-speculative decoding is built but is not lossless under int8 activations.**

- **A4, final** (seeds 5-7, 200M tokens, v0.5.16; `experiments/checks/a4_fullstack.py` -> `results/a4_fullstack.json`, provenance OK). Registered metric: in-training v2 val, nats per target token incl. EOT. It is not clean80 (SQ387), and that limitation is stated:
  - **R1: A4a (byte decoder) vs A3mx (BPE head) +14.39% [+13.39, +15.39]**, far outside the predicted -2% to +6%. **The registered viability rule (within 3%) fails.** EOT accounts for 0.072 nats per token of the gap; without it A4a is still about +7%.
  - **R2': second shared byte pass (A4d vs A4b) -1.68% [-1.73, -1.62]**, Holm step 1 rejects, in range.
  - R2'': deep supervision (A4a vs A4d) +0.55% [+0.00, +1.10], not significant: the aux loss does not help.
  - R3: shared vs unshared param-matched (A4a vs A4c) -1.10% [-2.04, -0.15], not significant after Holm.
  - Throughput: 210-250k tok/s vs A3mx's 288k.
- **New claims (PARTIAL: registered metric not clean80, decoder LR not bracketed):** C210 (the byte decoder's 14% cost), C211 (second shared pass -1.7%; deep supervision null).
- **Design-around registered as SQ394 (top pick):**
  - a wider decoder (d_local 256/384);
  - a hybrid head that keeps the BPE loss on the trunk;
  - a frozen-trunk A4d, to separate decoder capacity from the trunk losing its token-level signal.
  The human's directive keeps the byte decoder in the model, so its cost is designed around, not dropped.
- **A1 amendment 2, final: fixed loop depth beats random depth -3.17% [-3.73, -2.60]** (seeds 0-2, df 2), in the registered range. **C212 is PARTIAL**: A1a and A1i ran at E27's bracketed LR without their own bracket, and seeds 0-1 were the selection seeds.
- **Self-speculative decoding built** (E8; `gogeta/decode.py`, `tests/test_decode.py` ALL PASS on a rerun here, `experiments/devlogs/SPEC-DECODE.md`, `results/spec_decode_bench.json`):
  - The decoder caches KV, the SWA window and the GDN state for every (pass, core block), with exact rollback. Drafting uses r_draft passes and verification checks the k drafts in one step.
  - CPU bench on the A2s s2 checkpoint (r_full = 2): r_draft = 1 accepts 70% at k = 4 (3.7 tokens per verify call) and 57% at k = 8 (5.2).
  - **Lossless: FAIL** on 16 of 120 runs, all at near-tie steps. The int8 activation rounding differs between a 1-token and a k-token pass; with activation quantisation off, 30 of 30 runs are lossless. The criteria were revised after the smoke run (stated in the script), and the original criterion fails too.
  - Exact losslessness needs batch-invariant matmuls in the verify path. The GPU (fla) path and wall-clock speedup are not measured yet.
- **A3 amendment 1 (LR bracket), half in:** at 3e-3, A3mx is 0.6% worse than at 6e-3, and A2s is 0.6% better (within the 1% margin). The 1.2e-2 pair is training on the last box, and C206-C208 stay PARTIAL until it lands.
- **Fleet:** six of seven A3/A4 boxes finished and were destroyed by the controller (about $4.2). One box remains for the LR pair.
- **Quick reviews:**
  - 14:14: per-source split, merged into SQ239.
  - 14:19: pass@1 by loop count rejected at 13M, because the pass@1 floor is about 2%.
  - 14:24: the A3/A4 success criterion is answered (A3mx vs E24).

---

## v0.5.19: Outside review 14:09: SQ392 positional control for the long-context result (registered), SQ391 MoD across loop passes, SQ393 bits-matched quality; erratum: A3mx does not tie dense (TOST p 0.58)
*2026-11-23*

**Outside review 14:09: the long-context result gets a positional control (SQ392, registered), three new side quests, and a FLOP-counter flag checked and rejected. Erratum: A3mx does not "tie" dense.**

- **Outside review** (`notes/reviews/2026-11-23-1409-review.md`, Haiku), triaged in its footer:
  - **Accepted:**
    - SQ392 (top pick): RoPE position interpolation on SQ351's dense checkpoints, to test whether GDN's -60.7% past 1,024 tokens is mostly unscaled-RoPE degradation. Its prediction is registered in PREDICTIONS.md: about -30% after interpolation, and the result is restated if it falls inside -30%.
    - SQ391: Mixture-of-Depths routing across loop passes, to design around loops' compute cost.
    - SQ393: bits-matched quality, ternary-wide vs fp16-narrow at equal weight bytes.
  - **Merged:** sparse top-k coda retrieval into SQ383; same-shape A2f -> A2s self-distillation into SQ356.
  - **Fixes:** C208 gains a limitation that its 1.58-bit memory figures assume packing that has not been built (SQ378). The README's "does not stack at 200M" bullet now carries "(C202, PARTIAL)".
  - **Rejected after checking the code:** the claim that the FLOP counter overstates truncated-BPTT arms. Logged run compute has counted no-grad passes as forward-only since v0.5.3 (`gogeta/train.py:345-346`), so C209's ~10% compute gap stands.
- **Erratum (v0.5.17):** "the all-parts model ... ties dense per FLOP" is wrong. TOST at +-1% (`experiments/checks/a3_tost.py`, `results/a3_tost.json`) gives A3mx vs E20 +1.10%, 90% CI [-0.13, +2.33], p = 0.58: **equivalence not shown, and no advantage over dense shown**. A3mx vs A2f (+1.36%) and vs E24 (+4.35%) are not equivalent either. Quick review 14:09 raised this.
- **Fleet:** the A3 amendment 1 LR bracket started on box 4 (A2s/A3mx at 3e-3). The 1.2e-2 pair is queued, and A4 continues on the other boxes.

---

## v0.5.18: Claim audit 2026-11-23-1404: C206-C208 to PARTIAL pending an LR bracket (registered, running); C204 superseded by C209 (compute difference stated); C205 reporting fixed
*2026-11-23*

**Claim audit 2026-11-23-1404: five auditors on the five newest claims, all WEAK and all confirmed after checking. Three downgrades to PARTIAL, one supersession, one reporting fix, and an LR bracket registered to settle the downgrades.**

- **Scope:** only five claims were due (C204-C208); every other claim is in its 6-hour cool-down. One Codex GPT-6 Luna auditor per claim (`notes/reviews/claims/2026-11-23-1404-b1..b5.md`), all ok.
- **Downgrades (VERIFIED -> PARTIAL):**
  - **C206** (ternary MoE core -3.3%) and **C207** (ternary cost splits evenly): statistics policy 2.3 requires both arms tuned or LR-robust. A2s and the A3 arms all ran at 6e-3, inherited from E27's bracket, and were never bracketed themselves.
  - **C208** (all-parts model's standing): the same gap, and here it matters for direction, since A3mx (untuned) trails E24 and E20 at their tuned 3e-3. Limitations added: E20 used 6% more training FLOPs than A3mx (4.01e16 vs 3.79e16), and "+1.1% vs E20" is an inconclusive point estimate, not equivalence.
  - C207 also gains an attribution fix: the +9.5% full-precision gap is A2's P3, not C204.
- **Superseded:** **C204 -> C209** (VERIFIED). C204 said "matched training FLOPs", but E27 trains on about 10% less compute (truncated BPTT; 3.41-3.43e16 vs 3.79e16). C209 states the compute difference. The effect (-16.67% [-17.74, -15.58], dz -31) is far larger than the A1 screen's compute-matched lever (about 3%).
- **Reporting fix, status kept:** **C205** gains dz -30.1, identical FLOPs and params for A1e and E27, and the LR note. E27's 6e-3 is bracketed, and A1e is the untuned winner, so a better LR could only enlarge the effect.
- **A3 amendment 1 (registered before running):** A2s and A3mx at lr 3e-3 and 1.2e-2, seed 5, on the A3/A4 fleet (the low-LR pair started on box 4 at 14:07). Rule: 6e-3 within 1% of the best restores C206-C208 to VERIFIED; otherwise the affected comparisons re-run at the better LR. `a3_assembled.py` now keeps the bracket runs out of the paired comparisons.
- **Quick review 14:04 (competitor lab):** repo-held-out ranking merged into SQ379, packed-ternary batch-1 latency merged into SQ378; its deployment-metric question is answered in the triage footer.

---

## v0.5.17: A3 final: all four fixes count (MoE core -3.3%, FP mixers -5.0%, FP FFN -5.2%, all parts -7.7%); the all-parts model ties full precision but trails unshared GDN by 4.4% (C206-C208)
*2026-11-23*

**A3 final: every fix counts, and the all-parts model ties the full-precision version, but it still trails unshared GDN. A2's decisions survive a post hoc Holm replay.**

- **A3, final** (seeds 5-7, 200M tokens of code-py-v2, clean80 paired by seed; `experiments/checks/results/a3_assembled.json`, provenance OK). Every registered prediction (Q1-Q5) landed in its range, and all four fixes vs the assembled base A2s pass the registered Holm family:
  - Q1: ternary MoE core (A3m) **-3.29% [-4.24, -2.33]**;
  - Q2: full-precision mixers (A3x) **-5.02% [-6.11, -3.92]**, full-precision FFN (A3y) **-5.16% [-5.31, -5.01]**. The cost of ternary splits evenly, not mainly into the mixers as predicted, and the two recoveries sum to about the +9.5% full-precision gap;
  - Q3: all parts (A3mx) **-7.67% [-9.03, -6.28]**, near-additive (A3m + A3x predicts -8.1%);
  - Q4: A3mx vs A2f +1.36% [+0.09, +2.64]: most of ternary's cost is designed away;
  - Q5: A3mx vs E24 **+4.35% [+2.39, +6.36]**, vs E20 +1.10% [-0.70, +2.93]. **The registered competitive rule (within 2% of E24 and at or below E20) is not met.**
- **Secondary readouts:**
  - weight memory (ternary at 1.58 bits): A3mx 44.0 MB vs E24 77.2 MB and E20 74.7 MB (`results/a3_memory.json`);
  - training throughput: A3mx 288k tok/s vs E24 368k and E20 586k (RTX 5090).
  So the all-parts model is the smallest and ties dense per FLOP, but is slower per wall-clock and 4.4% behind unshared GDN.
- **New claims (VERIFIED, 3 fresh seeds, registered):**
  - C206: ternary MoE core -3.3%;
  - C207: ternary's cost splits evenly between mixers and FFN;
  - C208: the all-parts model's standing, with its competitive rule not met and limitations for wall-clock and horizon.
- **What this means for the design:** the remaining gap is loops' unique-parameter deficit vs unshared GDN (A2f vs E24 +3.2%), not ternary. The next lever is a longer-horizon point (SQ380), since the F2 interaction flipped sign between 20M and 200M.
- **A2 multiplicity replay** (quick review 13:44; `experiments/checks/a2_holm.py`, `results/a2_holm.json`). A2 registered per-prediction rules, not a family. A post hoc Holm over all six A2 comparisons rejects P1-P4 and A1e and keeps P5's near-null, so no decision changes. The t-test helpers moved to `experiments/checks/tstat.py` (stdlib; p-values checked against the df 1/2 closed forms).
- **A1 amendment 2:** A1a and A1i at seed 2 are training. A1a - A1i over seeds 0-1 is -3.19%.
- **A4 incident:** the first four A4 jobs exited in 1 s, because the fleet worker ran `train_c2f.py` without `PYTHONPATH` and `import experiments` failed. Fixed in the worker and requeued at 13:53. A4 now trains at about 250k tok/s, and the four A4 variants (a-d) are running on seeds 5-7. The A4 runs are in the registry as E65 (running). A2 is marked done as E63 and A3 is added as E64.
- **Quick reviews:**
  - 13:44: A2 Holm replay done; evidence-bundle check merged into SQ274.
  - 13:51: A1-vs-A2 calibration answered (A1e -8.3% screen vs -7.6% fresh).
  - 13:54: checkpoints are model-only and written once.
  - 13:59: SQ390 (contractive x precision 2x2); curvature probe merged into SQ365.

---

## v0.5.16: A2 final: C204 (assembled recipe -16.7% vs full fusion) and C205 (ternary core only -7.6%) VERIFIED; A4 amendment 1 (A4d) fixes an aux-loss confound before any A4 run
*2026-11-23*

**A2 final (all seeds) and two new VERIFIED claims; A4 amendment 1 fixes a confound before any A4 run; outside review 13:40; queue brought up to date.**

- **A2, final** (the slow box's last pair is in; `experiments/checks/results/a2_assembled.json`, provenance OK). Every registered prediction landed in its range:
  - P1: A2s vs E27 **-16.67% [-17.74, -15.58]**;
  - P2: A2s vs E20 +8.58% [+8.05, +9.11];
  - P3: ternary cost in the stack +9.52% [+8.34, +10.71];
  - **P4: A2f vs E24 +3.21% [+2.00, +4.44]** (now 3 seeds);
  - P5: adapters -0.05% [-0.13, +0.02];
  - A1e on fresh seeds -7.56% [-8.09, -7.02].
- **New claims:**
  - **C204 (VERIFIED):** the assembled recipe beats the full fusion by about 17% at matched FLOPs. It is recipe-level only, with limitations for horizon, out-of-corpus, the 3.2% gap to unshared GDN, and E27's 11% compute deficit.
  - **C205 (VERIFIED):** a ternary core with full-precision ends improves the full fusion by 7.6%.
- **A4 amendment 1** (outside review 13:40):
  - A4a (shared r_b = 2) trains with the per-pass aux loss and A4b (r_b = 1) cannot, so R2 mixed "second pass" with "deep supervision".
  - **A4d** (A4a without the aux loss) is added. R2 is split into R2' (A4d vs A4b, the second pass) and R2'' (A4a vs A4d, deep supervision) under Holm, and A4c's 4% FLOP shortfall is reported with R3.
  - No A4 job had started.
- **New side quests:** SQ387 (clean80 scorer for the byte decoder, before any A4 claim), SQ388 (minimum detectable effect at n = 3), SQ389 (a powered frozen-trunk byte-decoder sweep), SQ386 (rootless sandbox parity).
- **Ledger:** C83 records that the forgery is fixed for every probed route, with the remaining risks listed.
- **Queue:** `experiments/queue.json` / `QUEUE.md` regenerated.
  - Running: A3, A4 and E8 (speculative decoding).
  - Done: SQ61 (core) and SQ373.
  - Dropped, with reasons: SQ50 (KDA, parked), the E2-era loop diagnostics (SQ14, SQ21, SQ37, Q13) and E3, all superseded by the assembly ladder.
- **Quick reviews:** 13:24 (stop-point calibration SQ384), 13:29 (README A2 status fixed; SQ385), 13:38 (C83 reconciliation; SQ386).
- **Fleet:** the A2 wave is fully torn down (its last box cost $0.92 at 4x slower throughput). A3 runs on 7 boxes; A4 is queued.

---

## v0.5.15: Full stack: coarse-to-fine byte decoder around the all-parts backbone (tokens as patches); A4 registered; fixed byte patches +63% worse (SQ374)
*2026-11-23*

**The full stack exists: coarse-to-fine byte decoding around the all-parts backbone (`gogeta/c2f.py`). A4 is registered. Docs refreshed.**

- **SQ374, existing byte runs on clean80** (worker, one seed each): fixed byte patches are far too costly at this scale. E13b (P = 4, about the token model's FLOPs) scores 0.553 bpc vs E1's 0.339 (+63%); E13a (P = 2, 1.9x FLOPs) scores 0.462 (+36%). Loss rises 7-9% across the slots of a patch.
  - **Design-around:** tokens as patches. The global model stays on BPE tokens, and a byte decoder spells each next token.
- **Coarse-to-fine module** (C2F builder subagent; tests re-run by the main agent: `tests/test_c2f.py` 17/17, `tests/test_assembly.py` 13/13):
  - **Default `tokens` mode:** the real `FusionDance` runs on BPE tokens. From the trunk state at token i, a byte decoder spells token i+1 as [BOT, b1..bL] -> [b1..bL, EOT] with a 257-way head. Loss is in nats per target token, so it compares directly with a BPE head.
  - **Byte refinement loop:** shared-weight (r_b), with a single-pass and a parameter-matched unshared (A'') control, optional contractive updates and a per-pass loss. Its residual init uses its own depth.
  - **Frozen-trunk mode:** for the cheap X1 test.
  - **`fixed` P-byte mode:** an alternative through the `inputs_embeds` / `return_hidden` hooks.
  - **Causality:** bitwise perturbation tests pass for 26 configs (including MoE and the GDN reference), and deliberately broken controls do leak.
  - **Trainer:** `experiments/byte_model/train_c2f.py`, with the same CLI and run layout as `gogeta.train`; CPU smoke runs completed.
- **A4, registered before dispatch** (fresh seeds 5-7, paired with A3mx): the A3mx trunk with a byte decoder.
  - **Arms:** A4a (shared refinement, r_b = 2), A4b (single pass), A4c (unshared, parameter-matched). Computed FLOPs are 1.003x / 0.949x / 0.963x A3mx's.
  - **Predictions:** A4a +2% vs A3mx; sharing -1.5% vs single pass; 0% vs the parameter-matched unshared decoder.
  - **Viability rule:** A4a within 3% of A3mx.
- **Fleet:** the worker script now launches `C2F ...` jobs through `train_c2f.py`. A4 jobs join the A3 wave at this commit.
- **SQ349, syntax proxy (low power):** bracket validity is 100% in every cell and indent consistency 93-97%. No difference.
- **Docs (hourly refresh, checked):**
  - README results gain A2, SQ351 (GDN extrapolation) and SQ347 (copy deficit); "Where we are" describes the full-stack phase; the glossary gets corrected A3 and new A4 entries.
  - `post_training/SETUP.md` section 7 now describes the fixed harness (SQ373) and SQ61.
  - The stale A2 line in the README status block is fixed.
- **Still building:** `gogeta/decode.py` (self-speculative decoding), not in this commit.

---

## v0.5.14: A1 amendment 2: third seed for fixed vs random loop depth (registered before running)
*2026-11-23*

**A1 amendment 2 registered: a third seed for "fixed depth beats random depth"** (quick review 13:14, statistician).
- **What runs:** A1a (fixed r = 4) and A1i (random r, full backprop), which use equal training compute, at seed 2, paired with E27 seed 2. They run at the tail of the A3 wave.
- **Prediction:** A1a - A1i = -3% (range -5% to -1%). The comparison is confirmed only if the 3-seed paired CI lies below 0.
- **Status until then:** exploratory. The assembled recipe's fixed r is a design choice made on that basis.
- **Status:** the A3 all-parts arms are at 55-70% of training; first results are due in about 10 minutes.

---

## v0.5.13: A1 amendment: placebo stable, fixed depth beats random depth ~3% at equal compute, E27 LR bracketed; outside review 13:10 (full-stack risks)
*2026-11-23*

**A1 amendment arms read out (placebo, truncation control, LR edge), and the outside review at 13:10 on full-stack integration risks.**

A1 amendment 1 (registered before any A1 result; seeds 0-1, paired with E27; per seed: s0, s1):

| Arm | Effect vs E27 | Reading |
|---|---|---|
| **A1z** placebo (E27 re-run on the new fleet) | -0.14% (-0.04, -0.24) | the reference is stable; screen noise is small (prediction \|effect\| < 0.5%: met) |
| **A1i** full backprop, random r | -2.1% (-1.9, -2.3) | same training compute as A1a (both +17% vs E27) |
| A1a fixed r = 4 | -5.2% | at equal compute, **fixed depth beats random depth by about 3%**; removing the truncation is worth about 2% |
| **A1k** E27 at lr 1.2e-2 | +1.4% (+1.3, +1.4) | worse, as 2e-3 was (+0.6%): E27's 6e-3 is bracketed, not at an edge |
| A1b, A1f (now 2 seeds) | -0.1%, +0.1% | null confirmed |

- **Consequence:** the assembled model's fixed passes with full backprop is the right loop-training recipe.
- **Outside review 13:10** (the full-stack integration risks):
  - Draft acceptance at the untrained r = 1 operating point and the numerics margin for "lossless" go to the decode builder's bench. SQ381 (draft-aware training with an auxiliary first-pass loss) is the fallback; random-r training stays rejected per A1a vs A1i.
  - SQ382: a per-pass MoE routing audit.
  - SQ383: does the coda attention carry copy-at-distance? Agreed: the GDN copy-deficit fix goes OUTSIDE the loop (more full attention in the prelude or coda), not inside the shared core, where it would be quadratic and paid r times.
  - The byte-decoder init depth and precision go to the coarse-to-fine builder as a required test.
  - A3 reports its ternary parameter share; A3mx's ternary sits in the core MoE experts.
- **Quick review 13:07:** the N1X kernel check is rejected, since no current result depends on the N1X; the decode-timing gate belongs to the 3090/4090 check (SQ285).
- **Fleet:** the A2 wave is down to its slow last box. A3 trains on 4 boxes, with 2 more loading.
- **Claim audit 13:07:** skipped; no claim was due (all 131 in cool-down).

---

## v0.5.12: Full stack (everything): FusionDance hooks for coarse-to-fine; builders start c2f + self-speculative decoding
*2026-11-23*

**Human: "I mean everything: ssm + sliding windows + loops + coarse to fine + bitnet + MoE + speculative decode". The full-stack build starts.**

| Component | Status |
|---|---|
| SSM-family mixer (Gated DeltaNet) + sliding-window + full attention | in the assembled model (A2/A3) |
| Shared loop (6-block core x 2, contractive injection, fixed r) | in |
| BitNet ternary (core; full-precision ends; mixer/FFN scope under test in A3) | in |
| MoE (ternary experts in the shared core) | in A3m/A3mx, training now |
| Coarse-to-fine (bytes -> patches -> backbone -> shared-weight byte refinement loop) | builder subagent writing `gogeta/c2f.py` around the real `FusionDance` (the older `model_bytes.py` carries a stale forked copy) |
| Lossless self-speculative decoding (draft = fewer loop passes, verify = full) | builder subagent writing `gogeta/decode.py`: KV/SWA/GDN-state caches per (pass, block), parallel k-token verification, a greedy-losslessness test, and an acceptance-rate bench on a 200M A2s checkpoint |

- **Code:** `FusionDance.forward` gains `inputs_embeds` (a byte front end feeds patch embeddings) and `return_hidden` (a byte decoder reads the normalised hidden state). Defaults are unchanged (`tests/test_assembly.py` 13/13, `tests.test_moe` OK).
- **Plan:** A3 (training now) picks the backbone. A4 trains the full stack (coarse-to-fine around the A3 backbone), registered before dispatch, with an unshared single-pass byte decoder and a parameter-matched control. Speculative decoding is measured on the trained checkpoints.
- **Decision logged** in `notes/sketch.md`.

---

## v0.5.11: Worker results: GDN extrapolates past 1,024 tokens (-60.7% vs dense), GDN copy deficit beyond the window, C202 not a numerics artifact; reward forgery fixed (SQ373)
*2026-11-23*

**Side-quest worker results integrated (Sonnet worker, verified by the main agent), and the reward-forgery hole is closed.**

| Side quest | Result |
|---|---|
| **SQ351** length extrapolation (cells 000 vs 100, 3 seeds, 153 long out-of-corpus files) | beyond the 1,024-token training length, the GDN hybrid's loss is **-60.7% [-69.8, -51.5]** vs dense; dense degrades +160% past its window. This is the first measured (not analytic) evidence for the linear-time pillar. |
| **SQ347** copy-deficit probe (registered) | prediction not met: the interaction on copying is positive but its CIs include 0 at gaps 256/512. Descriptively, **the GDN cells copy far worse than dense beyond the SWA window** (+32%/+37% at gap 256): an issue the assembled model must design around (attention inside the shared core is the candidate fix) |
| **SQ340** fp32 vs bf16 scoring (registered) | shifts of at most 0.0035%; the 200M interaction is unchanged (+6.36 vs +6.37). C202 is not a numerics artifact; prediction held |
| **SQ339** ternary split, post hoc | int8 activations cost +0.1% (confirms A1d). Ternary weights without QAT are catastrophic (+769% / +197%), so the trained cost (+15.8%) is a QAT-adapted weight cost |
| **SQ341/SQ348** loop curve and per-pass lens | the 200M loop cells are best at their trained r = 4 (r = 1 +8%, r = 12 +5%), and the per-pass readout improves up to r = 4 |
| **SQ365** ternary jitter | 5% weight noise costs ternary cells +1.4% / +2.6% vs +0.2% for full precision; 10-14% of ternary weights sit near a rounding threshold |

- **Infrastructure:**
  - **SQ373, reward forgery fixed.** The grader/worker split binds the real exec/compile before the solution runs, sends the nonce only after it finishes, and returns the result over a pipe. All forgery probes now score 0/1 (`reward_forgery_probe_2026-11-23.txt`, after-fix section); `post_training.test_rewards` is OK (30 tests incl. 11 forgery cases). Residual risks are documented in `rewards.py`. The harness-level GRPO block is lifted, but real RL should still add container isolation. Recorded on C83.
  - **SQ61 core:** namespace hardening upstreamed into `rewards.py`, failing closed for GRPO. Resource-budget and descendant-cleanup tests are still open.
  - **SQ346/SQ352:** the sandbox self-test probes `$HOME` and gates execution at `run_tests` (16/16 with the absolute repo path the callers use).
  - **SQ345/SQ366:** score files carry dataset hashes, `check_compatible()` refuses mixed hashes, and `--store-nll` keeps per-doc NLL (default off).
- **Still running in the worker:** SQ349 (syntax proxy) and SQ374 (E13 byte runs); SQ379 (A2 out of corpus) is next.
- **A3 (the all-parts model) is training on the fleet.**

---

## v0.5.10: A2 confirmed: assembled model -16.7% vs full fusion, full-precision version ties dense; human: keep all parts -> A3 (MoE core + ternary scope) registered
*2026-11-23*

**A2 reads out on all its registered predictions; the human asks for every part to stay. A3 is registered: MoE rescues the loop, and full-precision scope removes ternary's cost.**

**A2, the assembled model** (fresh seeds 2-4, 200M tokens of code-py-v2, clean80 paired by seed; `experiments/checks/a2_assembled.py`, provenance OK). Every registered prediction landed in its range:

| Comparison | Result | 95% CI | Prediction |
|---|---|---|---|
| **P1** assembled A2s vs full fusion E27 | **-16.7%** | [-17.7, -15.6] | -17% (in range) |
| **P2** A2s vs dense E20 | +8.6% | [+8.1, +9.1] | +10% (in range) |
| **P3** ternary inside the stack (A2s vs A2f) | +9.5% | [+8.3, +10.7] | +7% (in range) |
| A2f (full-precision assembled) vs dense E20 | **-0.9%** | n = 3 | competitive |
| **P4** A2f vs unshared GDN E24 | +3.5% | [+2.3, +4.8] (2 seeds; seed 3 pending) | +6% (in range) |
| **P5** per-pass adapters (A2a vs A2s) | -0.1% | [-0.2, +0.1] | no effect |
| **A1e alone** (ternary core only) vs E27, fresh seeds | **-7.6%** | [-8.1, -7.0] | -6% (confirmed) |

- **Decisions:**
  - The stack is **confirmed**: the A1 winners add up to -16.7% at matched FLOPs.
  - The full-precision assembled model is **competitive with dense** per FLOP, with 39% fewer non-embedding params (15.1M vs 24.8M).
  - The adapters are not adopted.
  - The two remaining costs are ternary (+9.5%) and loops vs the unshared GDN stack (+3.5%).
- **Human decision:** "I want all the parts assembled and their issues designed around or fixed!" Every ingredient stays. A2 amendment 2's loop go/no-go default is superseded; its comparison is reported as a cost to fix.
- **A3, all parts** (registered before dispatch, fresh seeds 5-7):
  - **A3m:** a ternary MoE FFN in the shared core (4 x 512, top-2), the same active FLOPs with more unique params. This is the loop rescue E60 stage 2 had planned.
  - **A3x / A3y:** mixers or FFN kept in full precision, to locate ternary's cost.
  - **A3mx:** all parts combined.
  - **References:** A2s, A2f, E20, E24 on the same seeds.
  - **Predictions:** MoE -3%; full-precision mixers -5%; full-precision FFN -3%; all parts -8% vs A2s, within about 1% of E24.
- **Code** (off by default, bit-identical): MoE experts now support ternary (BitLinear-style fake quantisation with STE; this used to raise), plus `moe_core_only`, `ternary_mixer` and `ternary_ffn`. Tests: `tests/test_assembly.py` 12/12 and `tests.test_moe` 15/15 on CPU. The old "MoE + ternary raises" assertion is replaced by its new behaviour.
- **A2 fleet:**
  - 6 boxes ran; box 3 runs at about 89k tokens/s (4x slower than the rest), so its A2a seed 2 / E24 seed 3 pair finishes last.
  - The A1 re-runs (A1i, A1z, A1k, A1b, A1f) are finishing and get analysed with the next version.
  - Boxes are destroyed as they finish; wave spend so far is about $3.

---

## v0.5.9: Outside review 12:40: side quests pruned to decision nodes D1-D6 (34 parked); loop go/no-go rule (disclosed post-hoc); data gate for E61
*2026-11-23*

**Outside review 12:40 (Haiku) and quick reviews 12:29, 12:34 and 12:39, triaged. The side-quest log is pruned around decision nodes, and a loop go/no-go rule for the final model is written down.**

- **Decision nodes:** D1-D6 now head `experiments/side-quests.md`:
  - D1: assembled model vs unshared GDN at matched FLOPs;
  - D2: does ternary buy anything;
  - D3: is looping worth more than unique params;
  - D4: batch-1 decode on a 3090/4090;
  - D5: transfer and the data gate;
  - D6: safe code execution.
  The byte / coarse-to-fine track the human asked about stays open.
- **Pruning:** 34 open rows that move no node are parked with a reason (MoE research, most of the KDA family, SQ7, the E2 looped-core fixes, the hardware micro-benchmarks). They are not deleted, and each can be un-parked with a reason.
- **Loop go/no-go rule (A2 amendment 2), disclosed as written after A2 seeds 2 and 4 were seen.** Loops stay in the final model's default only if the full-precision assembled model A2f beats the unshared GDN hybrid E24 by >= 1% at matched FLOPs. Otherwise they survive only as a decode-time option (SQ362) or as the byte-refinement loop (E14). The seen seed (A2f -3.7% behind E24) makes "loops leave E61's default" the likely outcome, and that is stated in advance.
- **Data gate (SQ376):** v2 has about 0.2B tokens, and a 1.2B final model needs about 20B+. Data binds the final run before compute, so code-py-v3 (human download approval) is a prerequisite for E61.
- **New:** SQ375 (CUDA-graph decode), SQ376 (data gate from existing curves), SQ377 (block duplication at inference on unlooped checkpoints).
- **Rejected:**
  - the FLOP audit, done in v0.5.3, and A2's counts are exact;
  - the matched-applications ladder, merged into A3;
  - throughput telemetry, covered by SQ79 (now parked);
  - a v2 val dedup scan, covered by v2's 13-gram build rule and the K9-based clean80 scoring set.
- **Goal-6 order** (12:34): test the 3090 (sm_86) and 4090 (sm_89) in one rental (SQ285), the 3090 first if only one is possible.

---

## v0.5.8: Docs refresh (hourly)
*2026-11-23*

**Docs refresh (hourly, Haiku; checked by the main agent), with an erratum and a found-but-unanalysed result.**

- **README:**
  - The status block now carries the A1 results (exploratory) and says A2 is running and that GRPO is blocked until the reward-forgery fix (SQ373).
  - "Where we are, in plain words" is brought up to date.
  - A new **Glossary** covers A1/A2/A3, E-numbers, the F2 cells, clean80/clean50, matched FLOPs, "parameter-changed arm", log-points and what A2 can claim (recipe-level only).
- **Cold-start check:** the refresh agent followed the README quick start literally, and every referenced file, config, script and flag exists. The test count and skip notes were not re-run.
- **Byte Quest line, and an erratum to v0.5.6:**
  - The README said "the E13 patch-size sweep is underway (P = 2 is best so far; the 200M runs are in progress)". The 12:15 outside review flagged it, and the v0.5.6 triage wrongly said the phrase did not exist; the main agent had misread a truncated grep. The erratum is appended to that review's triage.
  - The runs are real: E13a and E13b at 200M bytes (seed 0, lr 3e-3) finished on the w4 fleet before teardown and were pulled, but never analysed (full-val 0.2774 and 0.3338 nats/byte).
  - The README now says so, and **SQ374** analyses them: clean80 bits/char, matched-FLOP comparison with the token models, and loss by position in the patch. This is the first step toward the coarse-to-fine byte stage the human asked about.
- **notes/sketch.md:** status at v0.5.7.

---

## v0.5.7: Reward forgery confirmed in the test harness (C83, SQ373: GRPO blocked); A2 amendment 1 (A1e on fresh seeds) registered
*2026-11-23*

**Reward forgery found and blocking GRPO; A2 amendment 1 registered; quick reviews 12:14, 12:19 and 12:24 triaged.**

- **Reward forgery (C83):** `experiments/checks/reward_forgery_probe.py` (output in `results/reward_forgery_probe_2026-11-23.txt`) shows that solution code which sets `builtins.exec` or `builtins.compile` to a no-op scores **1/1 on a failing test**, in both the committed harness and the working tree. The honest control scores 0/1. Swallowing AssertionError and `os._exit(0)` do not forge.
  - C83 gains this evidence and a limitation.
  - The fix is SQ373: high priority, owned by the side-quest worker. Bind the real exec/compile before the solution runs, use a fresh namespace per test and an unreachable reporting channel, with the probe cases as must-score-0 tests.
  - **GRPO stays blocked until SQ373 passes.** `evals/sandbox.py` uses the same harness, so exec-based pass@k is forgeable in principle. No model has been trained against this reward, so existing pass@k numbers are not affected.
- **A2 amendment 1** (registered before these runs): A1e (ternary core only) is confirmed or rejected on fresh seeds 2-4, paired with the E27 references already in the A2 wave. Prediction -6% (range -9% to -3%), shrunk for winner's curse. A3's leave-one-out contrasts will be Holm-adjusted as one family.
- **First A2 numbers** (not yet analysed; registered analysis when the wave completes): A2f seed 2, the full-precision assembled GDN + loops model, scores clean80 0.2501. Dense E20 seed 2 scores 0.2506 and E24 seed 2 scores 0.2412.
- **Quick reviews:** 12:14 (newcomer) gives a glossary and cold-start check task for the next docs refresh. 12:19 (safety) is the forgery above. 12:24 (statistician) is the A1e confirmation and the Holm rule for A3.

---

## v0.5.6: Outside review 12:15: A3 leave-one-out ladder planned; SQ369-SQ372; A2 fleet: one box failed setup, jobs requeued
*2026-11-23*

**Outside review 12:15 (Haiku) and quick review 12:09 (ML theorist), triaged. A3 is planned.**

- **A3 plan, adopted from the review:** a leave-one-out ladder on the assembled model A2s, on fresh seeds 5-9, registered after A2 reads out. The arms are A2s minus contractive injection, minus fixed r, with a 3x4 core (same 12 applications, half the unique core params: C197 inside the stack), A2f (no ternary), and A2s itself. It attributes the stack to its components, which the "do the components work together" question needs.
- **What A2 can claim** (12:09 review): a recipe-level claim only (A2s vs E27, and within 5% of dense if P2 passes). Mechanism claims need factorials on the recipe (SQ369 ternary x loop count, and the A3 ladder).
- **New side quests:**
  - SQ369: ternary x loop count on the assembled recipe.
  - SQ370: is the 20M->200M sign flip a cosine-schedule artifact (WSD vs cosine at 20M)?
  - SQ371: goal-6 decision memo. At 1.2B, neither ternary nor loops changes what fits in 24 GB, so their case rests on decode speed.
  - SQ372: amend E13 to compare patch sizes at matched FLOPs.
- **Merged into existing side quests:** ternary code churn into SQ365; a 200M state-tracking probe for loops into SQ68; a parameter-matched byte decoder arm (A'') into SQ207.
- **Flags checked:** the A2 configs' 20M max_tokens is overridden by `--max-tokens 200000000` on every job, the same pattern as F2. The README's "P = 2 best so far" does not exist. E24 is A2's unshared matched-FLOP reference. SQ207 and the assembly items are missing from the stale `queue.json` and go in the next queue update.
- **A2 fleet:**
  - Box 4 failed setup (the torch download died after 526 s). It was taken out of rotation, its two jobs (A2s seed 3, E27 seed 3) were requeued first, and it was destroyed for $0.12.
  - Six boxes are training, and box 5 is still loading.

---

## v0.5.5: Claim audit 2026-11-23-1202: 35 claims, no downgrades (C203 Holm checked, C30 WRONG rejected); A1 holds at equal wall-clock
*2026-11-23*

**Claim audit 2026-11-23-1202, plus quick reviews 12:01 (skeptical peer) and 12:04 (data engineer).**

- **Audit:** 8 Codex auditors, all ok, reviewed 35 claims: 14 SOLID, 7 WEAK, 13 NEEDS-EVIDENCE, 1 WRONG. Each was checked against primary files. **No downgrades.**
  - **C203 WEAK (Holm adjustment), rejected after computing it:** the four registered E60 comparisons have |t| = 21.0, 24.8, 10.0 and 14.2 (df 10.3, 10.0, 7.8, 5.0), so each p < 2e-4, under Holm's 0.0125 for m = 4. The t-statistics are recorded as evidence; VERIFIED stands.
  - **C30 WRONG, rejected:** Ouro Table 13's OlympiadBench row reads 76.44 at T = 4 and 39.26 at T = 8, exactly as claimed. The auditor read SuperGPQA's row.
  - **Confirmed, limitations added:** C93 (its stale "no multi-seed 200M model" note is updated) and C74, C35, C76, C75, C38 (paper-reported scope).
  - The 13 NEEDS-EVIDENCE verdicts are on claims already UNVERIFIED.
- **A1 table labels (12:01 review):** the parenthesised pairs in v0.5.4's A1 table are the two per-seed effects (seed 0, seed 1), not confidence intervals; a 2-seed screen computes none.
- **A1 at equal wall-clock** (in-training eval loss, SQ338 method): every winner still beats E27. A1g -6.6% (equal steps -8.9%), A1e -5.2%, A1a -3.0%, A1h -2.5%. A2 keeps the 6x2 shape.
- **Data provenance (12:04 review):** every A1/A2 box verifies v2 against the committed manifest's sha256 values before training. New side quests: SQ367 (atomic local dataset builds) and SQ368 (platform-tagged test pins).
- **A2 wave setup:**
  - Six of 8 RTX 5090s are training; two are still being set up.
  - On box 2 at f85ada5: `tests/test_model.py` 12/12 and `tests/test_assembly.py` 9/9 pass on the GPU.
  - `tests.test_moe` fails 15/15 on the x86-64 box. These are last-bit differences in pins recorded on the ARM64 host (CPU RNG and float reductions differ across architectures), not regressions; it passes 15/15 on ARM64 with the current code. Fix: SQ368.
  - A wave-2 driver bug was found and fixed before any job ran: `drive.sh` still wrote into the A1 folder, so boxes 2 and 4 got their A2 job lists from copies.
- **Side-quest worker:**
  - SQ366 (per-doc NLL storage) and SQ341/SQ348 are done: the 200M loop cells are best at their trained r = 4 (r = 1 about 8% worse, r = 12 about 5% worse).
  - These results get integrated when the worker hands back.

---

## v0.5.4: A1 results: ternary-core-only -8.3%, 6x2 loop shape -11.6%, fixed-r -5.2%; A2 assembled model registered; fleet-loss incident disclosed
*2026-11-23*

**A1 assembly screen: results (exploratory), and A2 (the assembled model) registered.**

A1: each arm is one compensation on the full-fusion cell E27, 200M tokens of code-py-v2, seeds 0-1, clean80, paired by seed (`experiments/checks/a1_screen.py`; provenance OK: registered commit, clean tree, v2):

| Arm | Change | Effect vs E27 | Winner rule |
|---|---|---|---|
| A1g | core of 6 blocks x r = 2 (+59% unique params) | **-11.6%** (-12.3, -11.0) | yes (parameter-changed); 13% slower |
| A1e | ternary in the core only (prelude, coda, inject full precision) | **-8.3%** (-8.2, -8.4) | yes; no speed cost |
| A1h | core of 4 blocks x r = 3 (+20% params) | -5.7% (-6.2, -5.2) | yes (parameter-changed) |
| A1a | fixed r = 4 training | -5.2% (-5.7, -4.8) | yes, but with +12% training compute |
| A1c | contractive injection | -1.6% (-1.7, -1.5) | yes (marginal) |
| A1d | weight-only ternary (no int8 activations) | +0.1% | no: the int8 activations are free |
| A1f | late ternary (full precision for the first half) | 0.0% (1 seed) | no |
| A1b | per-pass loop embedding | +0.2% (1 seed) | no |

- **Reading** (exploratory; confirmation is A2's job):
  - Ternary's cost sits at the edges of the network: keeping the first and last blocks and the injection in full precision recovers 8%.
  - The loop's cost is mostly a unique-parameter deficit, and more blocks with fewer passes recovers the most (C197 again).
  - Random-r training with truncated backprop costs about 5%, part of which is the compute deficit found in v0.5.3.
  - The predictions underestimated every winner: none was in its registered range except A1c.
- **Lost runs (disclosure):**
  - All 7 A1 boxes were destroyed while the amendment arms (A1i, A1z, A1k) and A1b/A1f seed 0 were running.
  - Cause: a teardown loop stopped with TaskStop kept running in Git Bash and raced the new dispatcher. Each time a box finished, the dispatcher assigned new jobs and the orphaned loop destroyed the box (identified from `vastai show audit-logs`).
  - All 14 completed runs had been pulled first; about 10 in-flight runs were lost.
  - Fix: one controller loop with a kill-switch file, and a `ps` check after every TaskStop (agent memory).
  - A1 fleet spend: about $3.4.
- **A2, registered before dispatch** (`experiments/checks/PREDICTIONS.md` "A2 assembled model"; E63). Fresh seeds 2-4, 200M, v2.
  - **A2s, the assembled model:** prelude SWA (full precision) -> [contractive inject -> core of 6 ternary blocks GDN, GDN, SWA, GDN, GDN, SWA] x 2, fixed, full backprop -> coda attention (full precision). It has 15.1M non-embedding params at 0.189 GFLOP/token, FLOP-matched to dense E20 and to the unshared GDN hybrid E24.
  - **Other arms:** A2f (A2s without ternary) and A2a (A2s plus per-pass adapters).
  - **References:** E20, E24 and E27 at seeds 2-4.
  - **Predictions:** stack vs E27 -17%; vs dense +10%; ternary inside the stack +7%; loops vs unshared +6%; adapters -1%.
  - **Decision rules:** the stack is confirmed if it beats E27 by >= 5% with CI below 0, and it is competitive if it lies within 5% of dense per FLOP. Weight-memory and wall-clock readouts are registered secondaries.
  - The lost A1 amendment arms re-run in the same wave.

---

## v0.5.3: Outside review 12:00: F2's loop cells were ~11% under-computed (truncated BPTT); FLOP accounting fixed; per-pass adapters (A2 candidate)
*2026-11-23*

**Outside review 12:00 (Haiku) and quick reviews 11:29 (statistician), 11:34 (systems) and 11:39 (reproducibility), triaged. A compute-accounting finding is fixed.**

- **Finding: F2's loop cells were under-computed.**
  - Loop cells train with r ~ U{1..7} and backprop through only the last 4 passes, so passes 5-7 run forward-only. The trainer counted every pass at full training cost.
  - Real training compute for E27-style cells is 1.715e8 FLOPs/token, against the recorded 1.928e8 (-11.1%). So F2's "FLOPs-matched" loop cells (010, 111) got about 11% less compute than the dense and GDN cells.
  - Part of the 200M loop penalty may therefore be a compute deficit. This is now a C202 limitation.
  - A1a (fixed r = 4) and A1i (full backprop) remove the deficit and use 12.4% more compute than E27; they are read with that caveat.
  - `gogeta/train.py` now counts no_grad passes as forward only. Training itself is unchanged.
- **A1 false-winner calibration** (`experiments/checks/a1_false_winner_sim.py`, 200,000 null screens): with 11 arms the screen nominates at least one false winner 16.7% of the time, but only 0.35% would also pass A2's 3-fresh-seed confirmation. The two-stage design holds.
- **A2 candidate code:** per-pass low-rank adapters (`loop_adapter_rank`, SQ357). There is one rank-r residual adapter per core block and pass, zero-init up, and its down init is drawn last, so every other weight keeps its init. Off by default, `to_dict()` omits it, and its FLOPs are counted. Tests: `tests/test_assembly.py` 9/9, `tests.test_moe` OK.
- **A1 provenance check:** `experiments/checks/a1_screen.py` asserts the registered commits (36acccc / 482af19), a clean tree and code-py-v2 for every run.
- **New side quests:** SQ359 (scorer determinism), SQ360 (elastic unshared depth), SQ361 (warm-started tying), SQ362 (decode-time frontier for goal 6), SQ363 (shared MLP with per-pass mixers), SQ364 (GDN state capacity), SQ365 (ternary threshold sensitivity, CPU). SQ342 gains fresh seeds 3-5.
- **A1 in flight:** the first runs are in (A1e, A1g, A1d, A1f, full-val loss only). Clean80 scoring follows on each box, and the registered analysis runs once all are pulled.

---

## v0.5.2: Docs refresh (hourly)
*2026-11-23*

**Docs refresh (hourly, Haiku; checked by the main agent).**
- **README:** the pitch keeps its ambition but now says the 200M penalties compound (C202, PARTIAL), and that the project is in an assembly phase testing one compensation at a time (A1, E62). The status block says the same.
- **README, new block:** "Where we are, in plain words" under "Start reading here", accepted from quick review 11:03 (newcomer). In six sentences it explains F2, clean80/clean50, the 200M result and its LR caveat, E60/C203 (with the compute caveat), and that A1 is running. It has one link each to claims-status, side-quests and PREDICTIONS.
- **notes/sketch.md:** status at v0.5.1: assembly phase, claims 19/27/9/2/1/71/2 = 131, A1 fleet (7 of 8 rented boxes running; the 8th never left "loading" and was destroyed for $0.08).

---

## v0.5.1: Outside review 12:10: A1 amended before results (A1i truncation control, A1z placebo, A1k LR check); E60c is not compute-matched
*2026-11-23*

**Outside review 12:10 (Haiku) and quick review 11:24 (ML theorist), triaged. The A1 screen is amended before any A1 result exists.**

- **A1 amendment 1** (registered in `experiments/checks/PREDICTIONS.md`):
  - **A1i:** E27 with full backprop through every pass. A1a (fixed r = 4) also removes E27's gradient truncation when r > 4, so it was not a single change; A1i separates the two.
  - **A1z:** a placebo. E27 is re-run unchanged on the new fleet, to measure environment drift and the screen's false-positive rate. With 10 arms the -1.0% rule has roughly a 20-25% chance of at least one false winner.
  - **A1k:** E27 at lr 1.2e-2, checking whether 6e-3 sits at an LR edge.
  - **New reading rules:** A1g and A1h are "parameter-changed" arms, not compensations; dense E20 joins the table, so "share of the gap closed" is paired; throughput is reported next to each effect; winners count only after A2 confirms them on fresh seeds 2-4.
- **Correction to v0.5.0:** E60c (dense 2048 FFN) vs E60a (MoE) is parameter-matched, not compute-matched. E60c's FFN matmuls cost about 2x E60a's active experts. Their near-equal wall time reflects MoE gather overhead, not equal compute. C203 gains that limitation; its main result (E60a vs E1 at matched active FLOPs) stands.
- **New side quests:**
  - SQ353: loop-state perturbation gain.
  - SQ354: loss sharpness.
  - SQ355: LR grid for the ternary arms.
  - SQ356: distillation into the fused cell.
  - SQ357: per-pass low-rank adapters, an A2 candidate.
  - SQ358: ternary MoE experts; MoE with ternary currently raises in `gogeta/model.py`.
- **Fleet:** 8 RTX 5090s were rented for A1 (about $4.4/h). Seven have finished setup and the 16 original A1 jobs are dispatching; box 3 is still loading. The extra arms run at this commit on the first boxes to finish.

---

## v0.5.0: Assembly begins: compensation hooks + A1 screen registered; E60 done (MoE -2.7%, dense-wide FFN -4.2%; C203)
*2026-11-23*

**Assembly begins** (human goal: put the model together, compensate its challenges, use the new dataset, rent 5090-class GPUs). This version also completes **E60 stage 1** (Code-MoE, registered).

**E60 stage 1** (code-py-v1, 200M tokens, 6 seeds per arm vs E1's 7 Phase 2b-5090 seeds; `experiments/checks/e60_stage1.py`):

| Arm | vs E1, clean80 | 95% CI | Prediction |
|---|---|---|---|
| E60a: MoE FFN 4x512 top-2 (learned router) | **-2.72%** | [-3.00, -2.44] | M1 PASS |
| E60b: frozen random router | -1.43% | (E60a beats it by 1.31%, paired [-1.54, -1.07]) | M3 FAIL: routing matters |
| E60c: dense SwiGLU 2048, same total params | **-4.21%** | [-4.58, -3.84] | M2 PASS |

- **Reading:** the dense, wider FFN beats the MoE by 1.56% at about the same wall time per run (5.19 vs 5.05 min on a 5090), so at this scale unique parameters in a plain FFN beat routed experts. This fits C197, where unique params are the lever.
- **Ledger:** new claim **C203** (VERIFIED, 6 seeds, registered). Limitations: LR picked at 20M, v1 data, router balance not measured.

**Assembly, stage A1 (registered, not yet run).** F2's 200M result is the problem to solve: the full fusion is 30.8% worse than dense, with anti-synergy (C202). A1 tests one compensation at a time on the full-fusion cell E27, seeds 0-1, 200M tokens of code-py-v2:
- fixed r = 4 training (E27 trains with r ~ U{1..7});
- a learned per-pass loop embedding;
- contractive injection;
- weight-only ternary;
- ternary in the core only;
- late ternary (full precision for the first half);
- two loop shapes that hold 14 block applications but give the core more unique parameters (6 x 2, 4 x 3).

The predictions and the winner rule (<= -1.0% paired, about 2 seed sd) are in `experiments/checks/PREDICTIONS.md`. Winners get stacked in A2.

- **Code:** new `ModelConfig` switches `loop_embed`, `inject_contractive`, `ternary_acts`, `ternary_fp_ends`; `BitLinear.quant_on` and `set_ternary_quant()`; `train.ternary_start_frac`. All are off by default, and `to_dict()` omits them at their defaults, so existing configs and hashes are unchanged. Tests: `tests/test_assembly.py` 7/7 and `tests/test_moe.py` 12/12 on CPU. The GPU suite (`tests/test_model.py`) runs on the rented box before training.
- **Also registered:** SQ340 (fp32 vs bf16 scoring) and SQ347 (copy-deficit probe), for the side-quest worker.
- **Side-quest worker:** a Sonnet subagent works through the CPU side quests (SQ339, SQ341, SQ348, SQ346, SQ352, SQ345, SQ340, SQ347, SQ351, SQ61) on files disjoint from the assembly work.
- **Quick reviews triaged:** 10:48 (safety: SQ352), 10:54 (code-model: SQ194 extended to the F2 cells), 11:03 (newcomer: a plain-words README block for the next docs refresh).
- **Budget:** the human authorised 5090-or-better rentals. The agent caps itself at $25 for A1+A2 and at most 8 boxes, and destroys each box once its jobs are pulled.

---

## v0.4.108: Claim audit 2026-11-23-1103: 48 claims, no downgrades; C36 WRONG rejected against the paper table; INCONCLUSIVE added to the statistics policy
*2026-11-23*

**Claim audit 2026-11-23-1103.** 8 Codex GPT-6 Luna auditors reviewed 48 claims (8 batches of 6, all 8 runs ok). Verdicts: 16 SOLID, 13 WEAK, 18 NEEDS-EVIDENCE, 1 WRONG. Each non-SOLID verdict was checked against the primary files. **No downgrades.**

- **Confirmed, limitations added:**
  - C201: the 20M measurement stands; REFUTED means only that the synergy fails its own 200M criterion, which was registered at creation.
  - C191: the clean K9 re-run supersedes the provenance concern.
  - C61, C24, C21: paper-reported, not tested locally.
  - C51: the 45.9% is an analytic projection-FLOP figure, not an end-to-end saving.
  - C127: documented ISA facts are separate from inferred incompatibility.
  - C125: "power-limited" is unsupported (1/55 power-cap samples).
- **Rejected:**
  - **C36 WRONG:** pdftotext of Hyperloop Table 1 shows the claim's percentages are right (Looped +2.2/+2.2/+3.3%, Transformer +1.4/+0.8/+1.3%, Hyperloop +1.9/+1.7/+1.2%); the auditor misattributed the rows. A limitation notes that the quoted sizes are the looped models'.
  - **C201 "downgrade to PARTIAL":** the registered criterion and policy 2's REFUTED rule are both met.
  - **Already handled by existing limitations or qualifiers:** C73, C193, C29, C190.
  - **The 18 NEEDS-EVIDENCE verdicts:** they are on claims that are already UNVERIFIED, so they are consistent with status.
- **Policy alignment:** `notes/statistics-policy.md` gains an INCONCLUSIVE row matching the ledger schema (a check whose own validity gate failed). C16 stays INCONCLUSIVE; the audit's WEAK verdict was really about this missing row. The decision is logged in `notes/sketch.md`.
- **Process fix:** the first `record` calls passed Windows backslash paths, which the ledger rejected as evidence pointers. `claim-reviews.json` entries were normalised, and the 48 `audit_verdict` events were appended with forward-slash paths.
- **Also:**
  - The lm-eval snapshot files archived in v0.4.106 are renamed `*.snapshot`, so lm-eval can never register the upstream `utils.py`, which loads HF `code_eval` in-process (quick review 10:48). The sha256 values are unchanged, and C109's evidence points at the new path.
  - The v0.4.107 erratum now attributes the blanked spans correctly: an unquoted heredoc when the main agent wrote the v0.4.106 spec.

---

## v0.4.107: Outside review 10:38: loop-cost wording tied to unique params, E9 card fixed, SQ347-SQ351; v0.4.106 entry text restored
*2026-11-23*

**Outside review 10:38 (Haiku) and quick reviews 08:44 (systems), 08:49 (cheapest experiment), 08:54 (skeptical peer) and 10:38 (data engineer), triaged.**

- **Fixed:**
  - README: "loops cost 10.6%" now says that looping at matched FLOPs also halves the unique parameters (12.3M vs 27.9M), so it is partly C197's effect. Per-FLOP figures are flagged as analytic, because sliding-window layers run as a dense-masked kernel (C12).
  - The E9 card was stale ("planned, not run"). It now shows its v0.4.79 result (clean80 0.2773 vs E2 0.2845, n = 1) and what it controls for: parameter-matched to E2 (9.17M vs 9.46M), not FLOP-matched (0.079 vs 0.193 GFLOP/token).
- **New side quests:**
  - CPU, on the local 200M checkpoints: SQ347 (copy-deficit probe, the review's pick for next, prediction to register first), SQ348 (per-pass logit lens), SQ349 (syntax-validity proxy), SQ351 (length extrapolation to 2,048).
  - GPU, needs approval: SQ350 (parameter-matched loop control).
  - SQ342 gains a power note: about 12 seeds per cell are needed to resolve a 2-point I; the replicate must pre-register clean80 and one LR rule.
- **Quick reviews:**
  - Thermal drift: rejected. SQ79 covers it, and the cross-box timing spread is small next to the 1.62x gap.
  - Wall-clock frontier: rejected. SQ310/SQ338 cover it, and truncated-cosine reads are biased.
  - Strict held-out slice: done as SQ344 (in v0.4.106).
  - Doc-join audit: clean; SQ345.
- **Erratum, v0.4.106 entry:** a shell-quoting error in how the main agent wrote that entry's spec (an unquoted heredoc, so bash ran the backticked spans; one tried to run `sandbox_home_probe.py` as a shell script and failed with a syntax error, no effect) blanked five inline-code spans in that entry's C109 and C106 bullets. They are restored here; the v0.4.106 tag itself is unchanged.

---

## v0.4.106: Claim audit 2026-11-23-1037: 48 claims, no downgrades; C109 and C106 evidence archived, C197 R2 CI stored, C202 caveats + SQ344
*2026-11-23*

**Claim audit 2026-11-23-1037.** 8 Codex GPT-6 Luna auditors reviewed 48 claims (8 batches of 6, all 8 runs ok). Verdicts: 38 SOLID, 9 WEAK, 1 NEEDS-EVIDENCE. Each non-SOLID verdict was checked against the primary files. **No downgrades.**

Confirmed and fixed with new evidence:
- **C109:** lm-eval's `mbpp_plus` scores MBPP's 3 base asserts. The unmodified lm_eval 0.4.13 task files are archived in `evals/upstream/lm_eval-0.4.13/`, with sha256 values matching the wheel's RECORD. VERIFIED stands.
- **C106:** the sandbox hides $HOME. A probe with negative and positive controls (`experiments/checks/sandbox_home_probe.py`) shows $HOME is unlistable and $HOME/.venvs is invisible inside `run_tests`. VERIFIED stands; SQ346 adds the probe to `self_test()`.
- **C197:** R2's CI is now stored in the result JSON: -7.15%, Welch 95% CI [-7.59%, -6.71%], df 5.8. The 20M-only LR tuning is now a limitation. VERIFIED stands.
- **C202:** limitations added (selected LR on the same seeds; SQ337 is not clean80), plus new evidence: SQ344, I by overlap stratum. The anti-synergy holds on clean50 (+5.49 log-points, every seed positive) and grows with overlap (dup80 +10.00). PARTIAL stands.

Rejected, with reasons in each audit file's triage:
- **Already handled by existing limitations:** C5 (registered rule kept with its CI qualifier), C196 (TOST is the registered equivalence test), C18, C128, C198.
- **C104:** still PARTIAL. Archiving the paper text needs a human PDF fetch.

Also in this version, from quick-review triage:
- SQ344 (done) and SQ345 (pin the doc-level join by hashing the val file);
- the 10:38 data-engineer join audit: all 24 SQ263 score files have identical per-stratum doc, char and token counts.

---

## v0.4.105: Docs refresh (hourly)
*2026-11-23*

**Docs refresh (hourly, Haiku; checked by the main agent).** The shared documents now match v0.4.103–v0.4.104:
- **README:**
  - status paragraph: the 200M F2 reversal as log-points, the SQ337 trajectory, SQ338's 1.62x wall time and the torn-down fleet;
  - a new "The fusion does not stack at 200M" result bullet;
  - the "Start reading here" C201 line is corrected (overlap at 20M, compound at 200M);
  - a stale "stage 2 is dispatching, cells blinded" sentence now says F2 is analysed.
- **notes/sketch.md:** decisions-log lines for v0.4.98 to v0.4.104; status at v0.4.104 (claims 18/27/9/2/1/71/2 = 130; fleet 0 instances; queue).
- **experiments/launch/campaigns/README.md:** queue timestamp updated.
- **post_training/SETUP.md:** status date updated; the sandbox list now includes /var/tmp, self-test 12/12.

---

## v0.4.104: Outside review 08:37: the 200M anti-synergy grows through training (SQ337); GDN loses at equal wall-clock (SQ338); LR-artifact claim corrected
*2026-11-23*

**Outside review 08:37 (Haiku) and quick reviews 08:34 (statistician) and 08:39 (ML theorist), triaged.** Three of their ideas could be answered from runs already on disk, so they were run at once (descriptive, not registered):

| Check | Result |
|---|---|
| SQ336, leave-one-seed-out on the 200M I | no sign flip; per-seed I = +8.83%, +3.94%, +5.86% |
| SQ337, I along training (in-training eval loss, not clean80) | **-4.0% at step 250, 0 near ~107M tokens, +4.5% at the last eval**: one rising trajectory, not a transient |
| SQ338, equal wall-clock | the GDN cell takes 1.62x the dense time; its -3.4% per FLOP becomes **+8.2% worse** at equal wall-clock |

- **Corrections:**
  - The F2 devlog's "the reversal is not an LR artifact" overstated H2'. Cells 000 and 100 ran only at 3e-3 at 200M, so an LR confound remains open (SQ342; C202 limitation added).
  - The I figures are log-points (+6.21 log-points = +6.4% as a ratio).
  - The README now says C202 is PARTIAL, and that GDN's lead holds per FLOP only.
- **Noted:** looping halves the unique parameters at matched block applications (12.3M vs 27.9M), so the loops factor is also a unique-params factor (consistent with C197).
- **New side quests:**
  - CPU: SQ339 (split the ternary cost into weights vs int8 activations, post hoc), SQ340 (fp32 reference scoring of cells 000, 100 and 111), SQ341 (loop-count curve on the loop cells). The 200M checkpoints are local.
  - GPU, needing a budget approval: SQ342 (200M LR completion), SQ343 (loop placement by block kind).
  - Merged: decay-only cell 111 into SQ235; per-name init generators into SQ332.
  - Rejected: A5 (per-cell LR at 20M), superseded by SQ342 at 200M; A8 (identifier renaming), covered by SQ194.
- **Fleet:** 0 instances; GPU budget spent.

---

## v0.4.103: SQ263: at 200M the fusion ingredients compound (I = +6.2%, anti-synergy); C201 REFUTED, C202; K9 reproduced exactly; fleet killed
*2026-11-23*

**SQ263, the registered 200M gate on the five F2 cells**, is analysed (24 runs on code-py-v2, 3 seeds per arm; cell, LR and tokens read from run `meta.json`; `experiments/checks/sq263.py`).

| Cell | 200M clean80, vs dense at each cell's best LR |
|---|---|
| 100 GDN | **−4.0%** |
| 010 loops | +10.6% |
| 001 ternary | +15.8% |
| 111 full fusion | **+30.8%** |

- **Interaction:** I = **+6.21%**, seed-blocked 95% CI [+4.20%, +8.22%], df 8 (Welch: +6.21%, [+4.44%, +7.97%]). **Anti-synergy:** at the registered horizon, the ingredients' penalties compound.
- **H2' FAIL:** the 20M-selected 6e-3 still beats 2e-3 at 200M in all three six-e-3 cells (2e-3 is worse by +0.93%, +0.20% and +0.64%). The reversal is not an LR-transfer artifact.
- **C201 is REFUTED** by its own registered refutation criterion. Its 20M synergy (−3.5%) is real at 20M only; the interaction's sign depends on the horizon.
- **New claim C202:** anti-synergy at 200M. It is PARTIAL, because H3' was registered as descriptive and there are 3 seeds per cell.
- **What this means for the thesis:** the full fusion is the worst cell at 200M. GDN alone remains the only ingredient that helps. The README and the F2 devlog are updated.
- **Analysis timing:** this was analysed in the session after the human's "kill the fleet"; the data had been collected before.

**K9 reproduced exactly** (quick review 08:24): a re-run at clean commit `d10f5a2` (CPU, 58 s) gives identical values for every measured field, including 1,667/4,000 near-duplicates and all per-doc overlaps. `K9.json` now records the clean commit, and C191's provenance limitation is superseded.

**Queue reconciled:** E9, Q10, Q11 and Q12 are marked done (all ran in Phase 2b on the N1X or the 5090). The rendered next picks are now SQ50 (GPU) and SQ61 (CPU).

**Fleet killed** (the human: "kill the fleet"):
- The last three boxes (w4-3, w4-4, w4-6) were pulled and destroyed at 08:24 EST. All E60 runs were saved: 18 at 200M plus the 9-run LR grid.
- **Disclosure:**
  - the boxes sat idle for about 5 h after finishing (03:15–03:22). The pull-and-destroy watcher ran inside the agent session and stopped when the session was interrupted. Cost: about $8.5.
  - **Total compute spend is $30.13**, against the approved $30 (credit $69.82).
  - Lesson saved to agent memory: destroy each box as its last job is pulled, and check instances at the start of every session.

**Quick reviews 08:24 (reproducibility) and 08:29 (competitor lab):**
- SQ274 gains a commit-provenance check.
- SQ335: the per-document spread of the F2 interaction.

---

## v0.4.102: Outside review 03:08: GDN memory horizon, init-vs-data seed split, dynamic eval (SQ330-SQ334); SQ222 promoted; README first-run path
*2026-11-23*

**Outside review 03:08 (Haiku), triaged:**
- **SQ330:** the GDN memory horizon, from learned decay half-lives (CPU). If most heads forget within 256 tokens, the long-memory reading of C18/C43 needs qualifying.
- **SQ331:** dynamic evaluation (per-document online adaptation), E1 vs E10.
- **SQ332:** an init-seed × data-seed variance split at d = 256. It needs a `--data-seed` option, because one seed drives both today.
- **SQ333:** an exploratory pairwise reading of C201 from the existing 2-ingredient cells.
- **SQ334:** a binding probe at distance.
- **SQ222** (fp32 vs bf16 recurrence) is promoted, since C197/C201 have no precision check.
- **Flags answered:**
  - C201's 5-cell estimand is the registered design.
  - The log scale is the registered metric; a raw-scale I of about −4.2% is a sensitivity number, not the test.

**Quick review 03:09 (newcomer):** the README quick start now opens with a **no-download, no-GPU first check** (`tests.test_options` + `tests.test_moe` on CPU: 28 tests, about 10 s). It also adds a "start reading here" pointer to the three defining results:
- C5: no detected hybrid advantage over a tuned Llama;
- C197: unique parameters beat loops at matched FLOPs;
- C201: the fusion penalties overlap.

**Fleet:** E60's 200M runs are on their last seeds. The pull-and-destroy watcher takes each box down when it is idle.

---

## v0.4.101: Claim audit 2026-11-23-0302: 48 claims, no downgrades; C30 'WRONG' rejected against the Ouro PDF; F2 interaction holds in every overlap stratum
*2026-11-23*

**Claim audit 2026-11-23-0302:** 8 Codex auditors reviewed 48 due claims; 79 are in cool-down, and all batches ran.
- **Verdicts:** 22 SOLID, 15 NEEDS-EVIDENCE, 10 WEAK, 1 WRONG.
- **No status downgrades.** Most non-SOLID verdicts concern UNVERIFIED claims, where the auditors agree they stay UNVERIFIED.
- **C30 "WRONG" is rejected after checking the PDF.** The ledger quotes Ouro-2.6B-Thinking's OlympiadBench as 76.4 at T = 4 and 39.3 at T = 8. Table 13 of `refs/11-looped-2026/2510.25741 - Ouro (LoopLM).pdf` lists, for T = 1–8: 18.96, 68.59, 75.56, **76.44**, 71.85, 69.19, 57.63, **39.26**. The auditor read the T = 5 column as T = 4.
- **Confirmed, with limitations or qualifiers:**
  - **C193:** post-hoc length bins with no committed script; exploratory.
  - **C191:** K9 provenance is inconsistent. The card cites `962492d`; the JSON records `60f4dcd` with describe `gb64199d-dirty`. The result itself is reproducible from the pipeline.
  - **C190:** exact streaming was checked at a chunk boundary only.
  - **C73, C75, C76, C78:** scoped to the Kimi paper's reported results.
  - **C125:** thermal only; the power limit is not established.
  - **C38:** the negative statements are scoped to the examined papers.
  - **C35:** scoped to the paper's tasks.

**F2 interaction by overlap stratum** (quick review 02:59, ML theorist; `f2_analysis.py --metric`; dup80 is the registered memorisation readout):

| Stratum | Seed-blocked I | 95% CI |
|---|---|---|
| clean50 | −3.76% | [−4.61%, −2.91%] |
| clean80 (primary) | −3.54% | |
| all docs | −3.16% | [−3.94%, −2.38%] |
| dup80 | −1.41% | [−2.29%, −0.54%] |

- It is negative everywhere and strongest on the most novel docs, so it is not a memorisation artifact. C201 gets the evidence.
- The source-disjoint check is folded into SQ314.

**Fleet:** E60's 200M runs are finishing on three boxes. A pull-and-destroy watcher takes each down when it is idle.

---

## v0.4.100: Eval sandbox: /var/tmp leak found and fixed (12/12 self-test); paired batches verified; E60 LR 6e-3 for all arms
*2026-11-23*

**Eval sandbox fix** (quick review 02:54, safety):
- **The leak:** a canary written to `/var/tmp` from inside `evals.sandbox.run_tests` persisted on the host, so one eval run could leave files for a later one. `/tmp`, `/dev/shm`, `/run` and `/mnt` were already private.
- **The fix:**
  - `/var/tmp` joins the private tmpfs mounts;
  - a new self-test probe, `shared_tmp_private`, checks that a write inside does not reach the host.
- **Result:**
  - the re-probe leaves no canary on the host;
  - `python -m evals.sandbox` passes **12/12** (`experiments/checks/results/test-runs/2026-11-23-sandbox-selftest.json`);
  - C106 gets the evidence.
- **SQ61's gates now include:** a world-writable-path sweep and resource budgets (02:39), alongside the network, filesystem and descendant-cleanup gates.

**Paired batches verified empirically** (quick review 02:44, data engineer):
- **The check:** `experiments/checks/check_paired_batches.py` hashes the token ids for 20 steps under seeds 0 and 1. It compares two arms that use very different amounts of global RNG.
- **The result:** the hashes are identical at every step, and different seeds differ. The loader's own generator (`gogeta/train.py:61`) makes every seed-paired comparison truly paired.

**E60 LR grid** (registered G3 rule, clean80 at 20M):

| Arm | lr 3e-3 | lr 6e-3 | lr 1.2e-2 |
|---|---|---|---|
| E60a | 0.4493 | **0.4423** | 0.4574 |
| E60b | 0.4507 | **0.4434** | 0.4663 |
| E60c | 0.4474 | **0.4438** | 0.4795 |

- All three arms take **6e-3**, an interior minimum, so no extension point is needed.
- The 18 runs at 200M (3 arms × 6 seeds) are running on three 5090s.

**Quick reviews 02:39, 02:44 and 02:54:** triaged in their files.

**Fleet:**
- **w4-1 is destroyed** (the human: "tear down any vps … if we don't use it").
- A pull-and-destroy watcher will take down w4-3, w4-4 and w4-6 when E60 finishes. Nothing more is queued.
- **Near-miss, disclosed:** the watcher's first version called the destroy helper under Windows PowerShell 5.1. There, `ConvertFrom-Json` does not enumerate arrays, so the label filter matched every instance and the destroy received all four ids.
  - The Vast CLI rejected the multi-id call, and no box was lost (all four were verified running).
  - The helper now force-enumerates, requires exactly one exact-label match and refuses otherwise. It is dry-run-checked under PowerShell 5.1 and 7.

---

## v0.4.99: Outside review 02:37: init-matched F2 check (SQ326), context-truncation intervention (SQ327); data-order pairing confirmed by code read
*2026-11-23*

**Outside review 02:37 (Haiku), triaged:**
- **SQ326, init-matched factorial sensitivity:** accepted. The gogeta init costs 6-block attention about 3.5% (SQ200) but helps the 14-block hybrid about 7% (SQ308). F2's cells use the default gogeta init, so its main effects may be init-dependent. To be registered before running.
- **SQ327:** causal context truncation at eval (1024 → 512 → 256 visible tokens) on the 7+7 checkpoints. It is an intervention test of GDN state beyond the SWA window.
- **SQ328:** LR-column robustness of the F2 interaction, descriptive (n = 1 per column).
- **SQ329:** an eval-time re-injection gain sweep on E2, testing the over-looping failure (C7).
- **Merges:** the continuous overlap regression goes into SQ323, and SQ321 is a duplicate of SQ215.
- **Rejected, the data-order pairing doubt:** `TokenData` draws batches from its own generator seeded only by the train seed (`gogeta/train.py:61`). Same-seed arms therefore see identical batches, and the paired CIs are paired.

**Quick review 02:35 (code-model practitioner):**
- SQ325: an import-context ablation on source-disjoint files.
- Q14 (the standard code benchmarks) stays human-gated, because it needs dataset downloads and eval installs (AGENTS §0a).

**E60 first GPU runs:** the LR grid is running on w4-6.
- E60a's in-training val loss at 20M: 0.991 / **0.979** / 1.013 at lr 3e-3 / 6e-3 / 1.2e-2.
- E60b: 0.994 / **0.980** / 1.033.
- The registered choice uses clean80 after scoring.
- MoE runs cost about 0.55 min per 20M run in steady state (the first run, 0.81 min, included compile).

---

## v0.4.98: Docs refresh (hourly); F2 audits: inputs reconciled, estimator validated, seed-robust, wall-clock caveat (C201)
*2026-11-23*

**F2 audits** (from quick reviews 02:25, reproducibility auditor, and 02:29, competitor lab), all CPU-only:
- **Inputs reconciled** (`experiments/checks/results/f2_input_manifest.json`): the 250 committed E2x score files break down as follows, with no duplicate (cell, seed) among the included runs.

  | Disposition | Files |
  |---|---|
  | Included | 217 |
  | Other factorial cells (E23/E25/E26) | 16 |
  | Seed 0, excluded as registered | 5 |
  | LR-sweep runs at non-registered LRs | 12 |

- **Estimator validated** (`experiments/design/check_f2_analysis_synthetic.py`): on synthetic data with a known I = −3.5% plus seed effects, the seed-blocked estimate returns −3.500% exactly with no noise. With noise 0.008 it returns −3.97%, CI [−4.87%, −3.07%], covering the truth.
- **Seed-robust:** leaving out each of the 73 seeds keeps I in [−3.69%, −3.42%], with no sign flip (`f2_loso_wall.json`).
- **The wall-clock differs from the FLOPs match.** Median minutes per 20M run:

  | Cell | Median minutes |
  |---|---|
  | 000 (dense) | 0.58 |
  | 001 (ternary) | 0.60 |
  | 010 (loops) | 0.69 |
  | 100 (GDN) | 0.94 |
  | 111 (full fusion) | 0.92 |

  So the best-quality cell (GDN) costs about 1.6× the wall time at this scale. This is added to C201 as a limitation.

**Docs refresh (hourly, Haiku; diff checked by the main agent):**
- README:
  - C5 is worded "0.18% worse in the point estimate; the CI spans zero";
  - C197 is VERIFIED on 4 seeds, with the lever being unique params;
  - the F2 prediction-was-wrong note is added;
  - C198 is shown as PARTIAL.
- `notes/sketch.md`: status refreshed to v0.4.97 with claim counts (19 VERIFIED, 26 PARTIAL, 129 claims), plus decision lines for v0.4.95–v0.4.97.

**Fleet:**
- w4-6 now runs the E60 stage-1 LR grid (9 runs at 20M), the first GPU use of the MoE code.
- SQ263 (200M F2 cells) is finishing on three boxes.

---

## v0.4.97: F2 result: the fusion ingredients interact (I = -3.5%, SYNERGY; C201) - the registered prediction was wrong
*2026-11-23*

**F2, the registered 2×2×2 factorial interaction**, is analysed.
- **Factors:** GDN, shared loops and ternary weights.
- **Design:** 14 block applications, FLOPs-matched; 20M tokens on code-py-v2.
- **Runs:** 217 (seed 0 excluded as registered).
- **Analysis:** `experiments/design/f2_analysis.py` was committed in `ead5926` before its first run. Output: `experiments/checks/results/f2_analysis.json`.

| Analysis | I = y111 − y100 − y010 − y001 + 2·y000 | 95% CI | df | Decision |
|---|---|---|---|---|
| Primary, seed-blocked | **−3.54%** | [−4.34%, −2.74%] | 140 | **SYNERGY** |
| Sensitivity, unblocked Welch | −3.25% | [−4.00%, −2.50%] | 163 | SYNERGY |

| Cell | Ingredients | Mean clean80 bpc | vs dense |
|---|---|---|---|
| 100 | GDN | **0.4243** | −7.3% |
| 000 | dense | 0.4578 | |
| 001 | ternary | 0.5025 | +9.8% |
| 111 | full fusion | 0.5092 | +11.2% |
| 010 | loops | 0.5172 | +13.0% |

- **Read:** the three ingredients' penalties overlap, so the full fusion is about 3.5% better than the sum of its parts predicts. It is not a gain over dense: the full fusion is still 11% worse at this 20M horizon, where loops and ternary each cost 10–13%. GDN alone is the best cell.
- **My registered prediction was wrong on both counts.** It said I = +0.5% ± 1.5% (P(I ≤ −1%) = 0.10) and that the full fusion would be the worst cell; loops-only is the worst.
- **New claim C201, VERIFIED, with limitations:**
  - the 20M horizon only; SQ263, the registered 200M check, is running and is unblinded now that this analysis is recorded;
  - the display-blinding exposure in stage 1 (v0.4.88–v0.4.89);
  - "synergy" means overlapping penalties.
- **Micro devlog:** `experiments/devlogs/F2.md`.
- **Data:** the F2 per-run score files (250) are now committed.
- **README:** the status summary has the F2 sentence.

---

## v0.4.96: Code-MoE layer for E60 (off by default, bit-identical); E60 stage 1 registered; outside review 02:09 (SQ320-SQ324)
*2026-11-23*

**Code-MoE FFN for E60** (subagent, Opus; the main agent re-ran every suite):
- **`gogeta/moe.py`:** a dropless top-k MoE FFN with SwiGLU experts.
  - Routing is softmax top-k with renormalised gates, depends only on the current token, and has no capacity dropping and no expert-choice routing.
  - It adds a Switch balance loss and a router z-loss, with an optional frozen router.
  - The `gather` dispatch computes only the active experts; the `dense` dispatch is an option.
- **Off by default:** new `ModelConfig` fields `moe_*` (and mirrored in `gogeta/hf.py`), with `moe_experts = 0` meaning dense. With them off, models are **bit-identical**, pinned in `tests/test_moe.py` against values captured on the unmodified code. For E0/E1/E2/E3/E9/E11 it pins:
  - the initial state_dict sha256;
  - the RNG state after construction;
  - `to_dict` and `param_counts`;
  - FLOPs as float hex;
  - the first 3 AdamW losses for E0/E1/E2.
- **`train.py`:** the aux loss is added to the backward pass only for MoE models. Reported and val losses stay LM-only, and `moe/*` stats are logged.
- **Tests, re-run by the main agent** (`experiments/checks/results/test-runs/2026-11-23-C100-C80.txt`):
  - `tests.test_options` + `tests.test_moe`: 28 OK, 1 skipped (needs a GPU);
  - `post_training.test_hf`: 10/10;
  - `evals.test_wiring`: 6/6.
  - Causality holds (perturbing future tokens leaves past logits within 1e-6).
  - Gather equals dense dispatch to 1e-5.
- **E60 configs** (`experiments/staged-configs/e60/`):

  | Arm | What it is | Non-embedding params | FLOPs vs E1 |
  |---|---|---|---|
  | E60a | 4 × 512 top-2 MoE | 18.2M total, 11.1M active | 1.00× |
  | E60b | E60a with a frozen router, balance/z-loss off | as E60a | 1.00× |
  | E60c | dense SwiGLU 2048, total-param matched | 18.2M | 1.47× |

- **Not yet measured:** GPU throughput, and inductor compile of the gather path (it graph-breaks; the dense dispatch is the fallback).

**E60 stage 1 registered** (`experiments/checks/PREDICTIONS.md`):
- **LR:** a G3 grid at 20M (9 runs), then 3 arms × 6 seeds at 200M.
- **Predictions:**
  - **M1:** E60a − E1 = −1.8%;
  - **M2:** E60c − E1 = −3.0%;
  - **M3:** |E60a − E60b| ≤ 0.8%.
- **Decisions:** D1–D3 as in the design.
- **Cost:** about 2.5 GPU-h, about $1.5, which fits the remaining budget.
- E60 is ungated by SQ282 and SQ238.

**Outside review 02:09 (Haiku), triaged:**
- **Accepted:**
  - SQ320: position-in-window loss split;
  - SQ321: a long causal conv mixer vs GDN;
  - SQ322: a one-rule TOST symmetry pass, a fair hit, because C197's "unique params" lever rests on SQ282's registered tie rule, not an equivalence test;
  - SQ323: clean-threshold sensitivity;
  - SQ324: a horizon replot.
- **Rejected:** a ternary width sweep, already covered by the F2 ternary factor, E3 and SQ290.

---

## v0.4.95: Claim audit 2026-11-23-0209: 48 claims, 17 non-SOLID; C200, C1, C131 downgraded to PARTIAL; C5 qualifier
*2026-11-23*

**Claim audit 2026-11-23-0209:** 8 Codex GPT-6 Luna auditors reviewed 48 due claims (79 are in cool-down). All 8 batches ran.
- **Verdicts:** 31 SOLID, 10 WEAK, 7 NEEDS-EVIDENCE.
- **Every non-SOLID verdict was checked by the main agent** against the ledger, the evidence files and `notes/statistics-policy.md`.

**Downgrades (confirmed):**
- **C200** (dedup is worth 1.6%): VERIFIED → **PARTIAL**. All arms ran at one LR (tuned on v1), with no LR robustness shown (policy §2(3)). This is the same rule applied to C199 in v0.4.89.
- **C1** (the GDN kernel matches the reference on the N1X): VERIFIED → **PARTIAL**. The N1X numbers exist only as a notebook transcription; the machine-readable rerun is on an RTX 5090.
- **C131** ("any random split leaks"): VERIFIED → **PARTIAL**. SQ225 shows overlap varies by source (52.3% vs 22.9%), so the universal is an inference.

**Qualifier corrections:**
- **C7:** [exploratory n=1] added.
- **C6:** the stale "E9 pending" note is replaced; E9 and the clean80 re-score exist, still one run each.
- **C5 (partly accepted):** it stays **REFUTED by its registered rule** ("REFUTED if mean(E1) ≥ mean(E10)"), because changing a registered rule after the fact would be a forking path. Its qualifier now states that both CIs include 0 (N1X n = 3, 5090 n = 7): no difference detected, not "E1 is worse".

**Limitations added:**
- C104: the primary paper text is not archived.
- C109: an unarchived upstream read.
- C18: n = 2, no CI.
- C128: the sm_120a clause is narrowed.
- C83: fork-bomb resistance is unverified (now in SQ61).
- C129: SQ71 does not test it.
- C91: the stale "replay pending" note is replaced by the observed 101-step replay.

**Evidence gaps closed:** C80 and C100 cited pass counts only in prose. Their test suites were re-run and saved to `experiments/checks/results/test-runs/2026-11-23-C100-C80.txt`: `post_training.test_hf` 10/10 and `evals.test_wiring` 6/6.

**Agreed, no change:** C28 stays UNVERIFIED.

---

## v0.4.94: C197 VERIFIED (4 seeds); the lever is unique params, not depth (SQ282); dedup alone is worth 1.6% (SQ238, C200 VERIFIED)
*2026-11-23*

**SQ283, C197's seed replication** (registered R1/R2; 200M tokens, lr 6e-3, seeds 0–3):

| Model | Mean clean80 bpc | vs E32 |
|---|---|---|
| E32 (14 unshared blocks) | **0.2488** | |
| E2 (looped) | 0.2834 | E32 is **12.2%** better (paired CI [−13.1%, −11.3%], R1 PASS) |
| E1 (7 seeds) | 0.2679 | E32 is **7.15%** better (Welch CI [−7.60%, −6.71%], R2 PASS) |

- **C197 is upgraded from PARTIAL to VERIFIED.**
- **Caveats:** the deep stack needs its depth-scaled init (SQ308), and the 200M arms all ran at 6e-3. At 20M, E32 is flat between 3e-3 and 6e-3.

**SQ282, depth vs width with fair LRs** (registered W1/W2; 20M tokens, 2 seeds, lr {3e-3, 6e-3, 1.2e-2}):
- **Best LRs:** E32 3e-3 (0.4371), E34 3e-3 (**0.4331**), E2 6e-3 (0.4919).
- **W1 FAIL:** at their best LRs, the wide 5-block E34 *ties* the deep E32 (E32 is +0.93%, within 2 seed sd of 1.7%).
  - The v0.4.84 "E34 is 14% worse" was a learning-rate artifact: E34 is very LR-sensitive (+26% at 1.2e-2).
  - By the registered rule, **C197's lever is unique parameters at matched FLOPs, not distinct depth.**
- **W2 PASS:** E34 beats the looped E2 by 12%.
- **Consequence:** E60 (Code-MoE, which adds width-like capacity) regains its prior. Its gates (SQ282 and SQ238) are both now closed.
- **README:** the scale sentence is corrected.

**SQ238, the dedup-only control** (registered D1–D3; 200M, E1, lr 6e-3, 4 seeds):

| Corpus | Mean clean80 bpc |
|---|---|
| code-py-v1sub (no dedup, v2's doc count) | 0.2663 |
| code-py-v2 (deduplicated) | 0.2620 |
| code-py-v1 (full) | 0.2679 |

- **Dedup effect** D = **+1.62%**, paired CI [+0.92%, +2.31%], all 4 seeds positive. **D1 PASS.**
- **Coverage/size effect** S = +0.64%. **D2 PASS.**
- **D3:** D + S = 2.27%, against the H5 gap of 3.08% (single seed).
- **Rule:** the A3 v2 results stand as "with dedup".
- **New claim C200, VERIFIED** (one architecture and one LR, noted as limitations).

**Fleet:**
- **w4-3 sat idle about 65 min** (00:55–02:00 EST) after SQ238 arm B. Its JOBS DONE line was lost during the operator hold and the monitor restarts, as happened with w4-5. It now runs SQ263.
- New practice (also saved as an agent memory): after any monitor gap, check every rented box directly.
- SQ263 (24 runs at 200M, blinded) is running on four boxes.
- **Spend:** about $19 of the approved $30.

**Quick reviews:**
- 01:44 (reproducibility auditor): SQ319, filling the TBD claim scopes; 117 of 127 definitions have one.
- 01:49 (safety): SQ61 gains a descendant-cleanup gate.

---

## v0.4.93: Outside review 01:38: packing, locality-mask and MTP controls (SQ315-SQ318); SQ290 gains a ternary PTQ arm
*2026-11-23*

**Outside review 01:38 (Haiku), triaged by the main agent:**
- **SQ315: document-masked packing in training.** Accepted (I1).
  - Training samples flat windows from an `<|eos|>`-separated stream with no attention mask or GDN state reset (`TokenData.batch()`; confirmed by reading the code).
  - Full attention, SWA and GDN use cross-document context differently, so the E1–E10 gap and C193's length effect may partly be packing.
  - SQ227 only changes scoring; this changes training.
- **SQ316: locality-prior control.** Accepted (I3): a sliding-window (256) Llama E10 at d = 256 against dense E10 and E1. If it matches E1, C198's small-width edge is the mask, not the linear mixer.
- **SQ317:** the base-loss cost of MTP auxiliary heads on E1, for goal 6's drafter (I4).
- **SQ318:** a per-token-class split of the E1–E10 gap on the 7+7 checkpoints (I5).
- **SQ290** gains a post-hoc ternary absmean arm (I6).
- **Rejected:** I2, a duplicate of SQ235 (the delta-rule ablation).

**Quick review 01:39 (newcomer):** the guided-trace idea is rejected (the learn page and run forms already cover it), and the question "what is the headline while F2 is blinded" is answered in the triage footer.

**Fleet:** F2 stage 2 is running on four boxes (scores blinded) and SQ238 arm B is on w4-3.

---

## v0.4.92: SQ308: C197's depth edge needs the depth-scaled init; SQ307 inconclusive; docs refresh (hourly); F2 stage 2 dispatched
*2026-11-23*

**SQ308, the init-controlled depth test of C197** (registered v0.4.90; 20M tokens, lr 6e-3, 3 seeds; `experiments/checks/results/sq308.json`):

| Comparison | Effect | 95% CI (df 2) | Prediction |
|---|---|---|---|
| E32 flat vs E32 gogeta | **+7.02%** | [+0.86%, +13.55%] | G1 FAIL |
| E1 flat vs E1 gogeta | +1.74% | [−0.34%, +3.86%] | |
| E32 vs E1, gogeta init | **−2.64%** | [−2.97%, −2.31%] | G2 FAIL (needs ≤ −3%) |
| E32 vs E1, flat init | **+2.42%** | [−1.78%, +6.79%] | G2 FAIL |

- **By the registered rule:** inconclusive.
- **Read:** the 14-block stack depends on the depth-scaled init. Under flat init it trains 7% worse and does not beat E1. C197's lever is therefore "distinct depth *with* a depth-appropriate init", the standard deep-net recipe, not depth alone.
- **The gap depends on horizon:** about 2.6% at 20M against about 7% at 200M.
- **Ledger:**
  - C197 gets a limitation (event 5);
  - C199 gets evidence that init effects depend on depth and horizon (E1 is +1.7% at 20M, +0.04% at 200M).
- **Micro devlog:** `experiments/devlogs/SQ308.md`.

**SQ307, the fresh-seed check of SQ301** (registered; seeds 2–5): E1 at 1.2e-2 vs E10 at 3e-3 at d = 256 is **−1.64%**, CI [−3.66%, +0.43%].
- All four seeds favour E1, but the CI includes 0, so **W3 FAILS: inconclusive**.
- E10's seed noise at this width is the limit.
- E1 at 1.8e-2 is worse than at 1.2e-2 on every seed.
- C198 gets this as evidence (event 6).

**Docs refresh (hourly, Haiku; diff checked by the main agent):**
- README:
  - "tie" wording replaced by "no difference detected";
  - an F2 stage-1/stage-2 sentence added (cell results stay blinded);
  - the fleet is down to five boxes;
  - the SQ307 and SQ308 outcomes added by the main agent.
- `notes/sketch.md`: status refreshed and decision lines for v0.4.89–v0.4.91 added. The main agent corrected two stale points: SQ200's init finding, and the Byte Quest status.
- The campaigns README's queue date is updated.

**F2 stage 2 is dispatched:** 87 runs on four boxes, at commit `5d86866` (after the sigma record).

**Quick review 01:33 (competitor lab):** SQ314, a source-disjoint re-check of the tuned 7+7 headline checkpoints on the SQ99 out-of-corpus set.

---

## v0.4.91: F2 stage 1 done: blinded sigma 0.0087, N_final 221 (stage 2 = 87 runs); code-py-v1sub verified on two boxes; fleet down to 5
*2026-11-23*

**F2 blinded variance re-estimation** (registered in the wave-5 amendment). This is recorded **before** stage 2 dispatches.
- **Estimator:** `experiments/design/f2_blinded_sigma.py`, committed in v0.4.89 before it was run. Output is in `experiments/checks/results/f2_blinded_sigma.json`.
- **Stage 1:** 135 runs, 27 seeds × 5 cells, including the pilot's seeds 0–2.
- **Sigma:** seed-blocked log-sd **0.00873** (residual df 104); unblocked 0.00945.
- **Registered rule:** N_final = min(246, ⌈235·(σ_blocked/0.009)²⌉) = **221**.
- **Allocation:** the registered shape (E20 twice the others) gives E20 = 74 and the other cells 37 each. **Stage 2 = 87 runs**: E20 seeds 27–73, the other cells seeds 27–36, at the registered LRs.
- **Nothing else was computed or looked at:** no cell means, contrasts or interaction estimates. The display-blinding disclosure is in v0.4.89, and more is below.
- **Design note, found now:** under the seed-blocked primary model, a seed run by only one cell adds no information about the cell contrasts. So E20's seeds 37–73 inform only the unblocked sensitivity analysis and E20's own variance.
  - The registered allocation is followed unchanged; changing it after seeing stage-1 sigma would itself be a deviation.
  - The primary analysis's effective N for contrasts is the 37 seeds shared by all five cells.
  - The analysis must report this alongside the power.
- **More blinding exposure** (quick review 01:10, SQ313):
  - **67 F2 stage-1 per-run score files were committed and pushed in v0.4.88**, so they are publicly readable;
  - they are no longer copied into the repo until this step;
  - review agents are told not to compute per-cell F2 results.

**code-py-v1sub (SQ238's control data) is verified:**
- two boxes, w4-1 and w4-5, built it independently in 43 s and 37 s, with byte-identical output (train.bin sha256 `7e0f15f1…`);
- 698,714 docs, 196.5M tokens;
- manifest: `experiments/data_integrity/dataset-v1sub.manifest.json`;
- details in `notes/data-v2.md`.

**Fleet:**
- **w4-5 is destroyed** (2.78 h, about $1.65). It sat idle for about 40 min after its last batch: its JOBS DONE event was lost while the operator hold and monitor restarts were in progress. Its inference outputs (knockout, ensemble, docnll) were pulled first.
- Five boxes remain.
- **Queue reconciled:** Q7/Q8 are marked done; they ran on the N1X and are superseded by the 7-seed 5090 arms.

**Quick reviews:**
- **01:10 and 01:24 (safety):** SQ61 now requires a fail-closed GRPO reward path that refuses to score when network or filesystem isolation is missing (C83). SQ313 is the blinding redaction check.
- **01:18 (cheapest-experiment hunter):** the Q8 preflight idea is rejected, since Q8 was a stale entry and the launcher already runs a preflight.

---

## v0.4.90: Outside review 01:07: C197's depth result is confounded with init (SQ308 registered); README width wording fixed; SQ309-SQ312
*2026-11-23*

**Outside review 01:07 (Haiku), triaged:**
- **Idea A (accepted, registered as SQ308):** the default "gogeta" init scales residual outputs by 0.02/√(2·depth) (`gogeta/model.py`, `_init_weights`).
  - That gives std 0.0058 for 6-block E1 and 0.0038 for 14-block E32.
  - So C197's "14 unshared blocks beat E1 by 7%" is also an init comparison, and C199 shows init can move all-attention stacks by 3.5–3.8%.
  - **SQ308 design:** E1 and E32 at 20M tokens × {gogeta, flat} × seeds 0–2, 12 runs. Configs are sha-bound; `staged-configs/sq308/e32_flat.json` is new.
  - **G1:** the E32 init effect is within ±1%.
  - **G2:** E32 beats E1 by ≥ 3% under both inits.
  - **Rule:** a flat-init gap within ±1% means C197's lever is an init artifact.
- **Also accepted:**
  - SQ309: length-bucketed sampling, testing whether C193's long-document deficit is data exposure;
  - SQ310: an equal wall-clock replot from existing logs (CPU);
  - SQ311: E32 on the out-of-corpus length bins;
  - SQ312: zero-init residual branches.
- **README (the review's flag, AGENTS §0 wording lag):** the width sentences now say:
  - "no difference detected" at d = 512, instead of "ties";
  - E1 is 1.2–1.7% better at d = 256 (exploratory, with SQ307 running);
  - the init result is C199 PARTIAL, and SQ308 checks the C197 init confound.

---

## v0.4.89: Claim audit 2026-11-23-0102: C199 downgraded to PARTIAL, C198 caveats; F2 blinding disclosure + blinded estimator committed
*2026-11-23*

**Claim audit 2026-11-23-0102** (2 claims due; 125 in their 6-hour cool-down; Codex GPT-6 Luna). Both verdicts were WEAK, and the main agent confirmed both against `notes/statistics-policy.md`:
- **C199 is downgraded from VERIFIED to PARTIAL.**
  - The init effects were measured at fixed LRs, and VERIFIED needs tuned or LR-robust arms (§2.3).
  - "E1 is insensitive to init" is a no-effect claim: +0.04% [−0.23%, +0.31%] means no difference was detected, not equivalence, and no TOST was registered (§2.6).
  - The I1/I2 effects stand at those LRs.
- **C198 stays PARTIAL, with two limitations:**
  - "they tie at d = 384/512" overstates what CIs spanning zero show;
  - SQ301 picked each arm's best LR on the same two seeds it reports, which biases the gap upward.
  - New **SQ307**: a fresh-seed confirmation at the selected LRs, plus 1.8e-2 for E1 because 1.2e-2 sat at the grid edge.
- **Micro devlogs:** `experiments/devlogs/SQ200.md` and `experiments/devlogs/SQ243.md`.

**Disclosure: F2 stage-1 blinding was breached in display.**
- The wave-5 amendment requires that no cell means, contrasts or interaction estimates are computed or looked at before the blinded variance re-estimation.
- The main agent's fleet monitor relayed each F2 run's final val loss and clean80 score, labelled by cell, into the session, from the first F2 runs until 01:00 EST.
- No cell means, contrasts or interaction estimates were computed, tabulated or used for any decision. The stage-2 size will come only from the registered sigma rule.
- The monitor now redacts F2 scores until the re-estimation.
- The estimator, `experiments/design/f2_blinded_sigma.py`, is committed **before** it is run. It prints only counts, the seed-blocked and unblocked pooled sigma, and N_final = min(246, ⌈235·(σ/0.009)²⌉).
- Readers should weigh the F2 interaction result knowing the per-run scores were visible during stage 1.

**Fleet:**
- F2 stage 1 is finishing on five boxes; SQ238 arm B is running on w4-3.
- Checkpoint pulls are now incremental: about 1 min instead of several, because already-pulled checkpoints are skipped.

---

## v0.4.88: SQ200: the init handicap explained (C199 VERIFIED); SQ301: E1's d=256 edge survives LR tuning; GPT-6.2 Astra feasibility review + plan errata
*2026-11-23*

**SQ200 init 2×2 + SQ234** (registered I1–I4, 4 seeds per new arm, RTX 5090, clean80; `experiments/checks/results/sq200.json`):

| Comparison | Effect | Paired 95% CI | Prediction |
|---|---|---|---|
| E0 flat vs E0 gogeta init | **−3.48%** | [−4.08%, −2.87%] | I1 PASS |
| E10 gogeta vs E10 hf init | **+3.81%** | [+3.35%, +4.28%] | I2 PASS |
| E1 flat vs E1 gogeta init | +0.04% | [−0.23%, +0.31%] | I3 FAIL |
| E1 flat vs E10 hf | +0.07% | [−0.56%, +0.69%] | I4 not triggered (descriptive) |

- **Read:**
  - our scaled-residual init costs all-attention stacks about 3.5–3.8%;
  - the GDN/SWA hybrid E1 does not care which init it gets;
  - with flat init, E0 trails E10 by only +0.44%.
- **Two consequences:**
  - E1's original "3.6% better than E0" headline was an init artifact;
  - C5's tie is not an init handicap on E1.
- **New claim C199, VERIFIED** [4 seeds per arm; LRs not re-tuned per init]. C5 gets this as evidence (event 12).

**SQ301, the LR-fair re-test of C198 at d = 256** (registered v0.4.87; seeds 0–1; run LRs read from each run's `meta.json`):

| Model | lr 3e-3 | lr 6e-3 | lr 1.2e-2 |
|---|---|---|---|
| E1 | 0.3359 | 0.3311 | **0.3305** |
| E10 | **0.3363** | 0.3371 | 0.3490 |

- **At the best LRs**, E1 − E10 = **−1.71%** (s0 −1.68%, s1 −1.74%).
- **W1 FAIL:** E10's best LR stays at 3e-3.
- **W2 FAIL:** the gap widens instead of shrinking.
- **Registered rule:** C198 keeps PARTIAL, with this as evidence.
- **Caveats:**
  - E1's best LR sits at the grid edge, so its edge may be understated;
  - the init difference does not favour E1 (SQ200);
  - two seeds only.
- **Read:** at the smallest width the hybrid really is better, and the advantage fades by d = 384.

**GPT-6.2 Astra feasibility review** (the human's request; Codex read-only; `notes/feasibility-final-training-astra.md`).
- **Verdict:** advance the unlooped hybrid G-1B to an inference prototype plus a real-code scaling pilot, and hold the final-training rental.
- **Per ingredient:**
  - GDN/SWA: keep;
  - loops: gate, unshared by default;
  - ternary: gate against int8/int4 at equal deployed bytes;
  - MoE: omit from the base run;
  - bytes: a separate track.
- **Four plan errors confirmed by the main agent and appended to `notes/final-model-plan.md` as errata:**
  - G-3B at 128k leaves about 12.4 GB, not more than 15 GB;
  - the drafter cost ratio is about 0.25, so the speedup is about 1.44×;
  - 32k prefill takes at least ~2.45 s;
  - the pilot is 25% of G-1B, not 1/10.
- **Missing infrastructure it found:** epoch coverage (the loader samples with replacement), multi-GPU training, mid-run resume, the dense SWA mask and the RoPE offset. New SQ305 (deterministic coverage loader) and SQ306 (DDP/FSDP plus restore rehearsal).
- **Decisions it leaves for the human:** data downloads, a new spending ceiling above $30/12h, G-3B, and the v1.0.0 definition.

**Tooling:**
- `notes/reviews/codex_run.ps1` gains optional `-Model`/`-Effort` (default unchanged: GPT-6 Luna, high).
- `cockpit/remote/fetch_data.sh`'s final clean check now ignores untracked files: inference outputs had made a verified rebuild report a mismatch on w4-5.
- 140 new clean80 score files are committed (F2 stage 1 so far, SQ282, SQ283, Inkling, SQ200, SQ238 arm A, SQ301).

**Fleet:**
- The operator hold (the human's interrupt, 00:3x) is lifted ("Resume").
- Idle boxes are reassigned to F2 stage 1.
- Every rented instance is accounted for (six, w4-1 … w4-6).

---

## v0.4.87: Outside review 23:39: C198's d=256 result is noisy and LR-confounded (SQ301 registered); README scale caveat; SQ300-SQ304
*2026-11-22*

**Outside review 23:39 (Haiku), triaged.** Fact-checked against `experiments/checks/results/sq243_sq241.json`.
- **C198's d = 256 endpoint is weak:**
  - E10's four seeds have sd 0.0021, against 0.0003 for E1 (7×).
  - E1 does beat E10 on all four seeds, so the review's "driven by three seeds" is an overstatement; but the size of the gap rests on E10's noise.
  - The LRs were never re-tuned per width.
  - E1 also uses FusionDance's init and E10 uses HF's. SQ200, running now, measures that confound at d = 384.
- **SQ301 registered and queued** (`experiments/checks/PREDICTIONS.md`): an LR-fair re-test at d = 256.
  - E1 and E10 at lr {3e-3, 6e-3, 1.2e-2} × seeds 0–1, reusing the existing cells: 8 new runs.
  - **W1:** E10's best LR is ≥ 6e-3.
  - **W2:** the best-vs-best gap lies in [−1.0%, +0.25%].
  - **Rule:** a gap within ±0.5% refutes C198.
- **README:** the "scale looks like the lever" sentence now carries the E34 caveat from v0.4.84 (a wide 5-block model at E32's parameters and FLOPs is 14% worse, so the lever may be distinct depth).
- **New side quests:**
  - SQ302: LR transfer across width (muP-style), which feeds the final-model LR;
  - SQ303: grow unique depth onto a trained E1;
  - SQ304: per-seed loss-curve diagnosis.
- **Rejected:** EMA as a power tool (SQ217 covers EMA, and a headline-metric change needs its own registration), and an E32/E2 redundancy probe (covered by SQ248 and SQ265).

**Quick review 23:39 (code-model practitioner, Codex):**
- SQ300, a patch-repair pilot: a capability gate for the final model, gated on `generate()`.
- The identifier reuse-distance idea is merged into SQ209.
- The whitespace probe is rejected as a duplicate of SQ194.

**Fleet:**
- The SQ238 dedup-control dataset `code-py-v1sub` is built on w4-1: 698,714 docs, 196.5M tokens, 179,069 v1-val-linked docs excluded, train.bin sha256 `7e0f15f1...`, built in 43 s.
- Its 4 seeds are training.
- Its manifest is committed only after a second box reproduces it byte for byte.
- 136 jobs are queued (wave 6 remainder, SQ238 arm B, F2 stage 1).

---

## v0.4.86: Docs refresh (hourly); date errata (v0.4.85 and wave-6 headers); SQ299 from quick review
*2026-11-22*

**Docs refresh (hourly, Haiku subagent; diff checked by the main agent):**
- **README:**
  - the status paragraph and the "Scale, so far" bullet now carry the SQ243/SQ241 results (C198 PARTIAL);
  - quick links gain the final-model plan;
  - the Byte Quest section no longer says "No byte run has started" (main-agent fix): the SQ188 LR check picked 3e-3 and the E13 sweep is running.
- **`notes/sketch.md` status:** updated to v0.4.85; the claim counts are now 23 PARTIAL of 126.
- **`experiments/launch/campaigns/README.md`:**
  - lists the two watchdog resume campaigns, not run because of the operator hold;
  - says `--lr` is passed through (Q2 done).
- **One agent sentence replaced:** "F2 ... not mentioned in v0.4.85" now says F2 is queued after SQ282/SQ283 and the SQ263 gate.

**Date errata.** The local clock is EST. The cloud boxes log in UTC, and several labels were written an hour or more ahead:
- **v0.4.85** was dated 2026-11-23; it was committed 2026-11-22 23:31 EST. Fixed in its DEVLOG date line, the `notes/sketch.md` decision lines, and E61's note.
- **The "outside review 2026-11-23 00:10"** (v0.4.84) was written at about 23:10 EST on 2026-11-22, and its file name `notes/reviews/2026-11-23-0010-review.md` is wrong. The three side-quest citations now say 2026-11-22 ~23:10, and the file keeps its name so existing links still work.
- **The wave-6 part 3/4 registration headers** in `experiments/checks/PREDICTIONS.md` give 00:20/00:30. The commits are `2526428`/`9b78441` at 23:11/23:12 EST, before any of those jobs were dispatched. An erratum is appended there and the registered text is unchanged.

**Quick review 23:33 (data engineer, Codex):** SQ299, a v3 build drop ledger (by source, licence and reason) with hard-fail split-disjointness asserts, recorded in the v3 manifest. Triage is in the review file.

---

## v0.4.85: Goal 6: a final model on a 24 GB RTX 3090/4090 (plan, E61); SQ243 width effect (C198); SQ241; byte LR 3e-3; corpus v3 plan
*2026-11-22*

**New goal 6 (the human's request): ship the final model on a gaming GPU.** End the experiments with one model that runs batch-1 inference with a KV/state cache and lossless speculative decoding on a single 24 GB RTX 3090 or 4090. Added to the goals in DEVLOG and README; goal 5 now says the experiments run locally *or on cheap rented GPUs*.
- **Plan:** `notes/final-model-plan.md` (planning subagent; the main agent re-derived the headline numbers).
- **What binds is training money and unique tokens, not VRAM:**
  - 24 GB holds about 9B params in bf16;
  - every size we can afford fits with room for more than 1M tokens of cache;
  - training costs about $2.15 per 1e18 FLOPs on a 5090 (computed from E1's measured 743k tok/s at $0.523/h, which assumes the 14M-scale efficiency carries over; SQ297 measures it).
- **Primary target G-1B:**
  - about 1.23B params: 24 blocks in E1's pattern (12 GDN, 8 SWA, 4 full attention), d = 2048, GQA with 4 KV heads;
  - about 24–32B tokens, about $500–650 all-in;
  - estimated (not measured) decode on a 3090: about 300 tok/s in bf16 and about 450–600 in int8;
  - cache: 8 KiB per token, so about 1 GB at 128k context.
- **Stretch G-3B:** about $2.8k. It needs Stack v3 data and sharded training, and is worth it only if the scaling fit predicts at least 5% better than G-1B.
- **Speculative decoding:** a separate drafter or MTP heads are estimated at 1.5–2.9×. Self-speculation by fewer loops comes to about 1.3–1.6×.
  - `gogeta/model.py` checks out: nothing in a loop pass depends on the total r, so the draft's pass caches can be reused.
  - None of these acceptance rates has been measured.
- **Gated, not assumed:** loops, ternary, MoE and the spec-decode scheme each enter the final model only if their gates pass. Milestones run from M1 (inference stack plus a 3090/4090 rental smoke test, under $2) to M4 (the final train). **M4 needs the human's approval.**
- **Registered:** E61 "Final model" (planned) in `experiments/experiments.json`, and side quests SQ285–SQ298: sm_86/sm_89 kernels, KV/SWA cache with `generate()`, self-spec cache reuse, decode benchmark, memory envelope, int8/int4 PTQ, scaling-law fit, repeat-epoch penalty, GQA, long context, the 4090 L2 ternary core, drafter/MTP acceptance.
- **Open for the human:** AGENTS §2 reserves v1.0.0 for the full fusion. If loops or ternary fail their gates, the final model is hybrid-only. Should it still be 1.0.0?

**SQ243 two-size check** (16 runs on RTX 5090s, clean80 bpc, E1 at lr 6e-3 and E10 at 3e-3 at every width; `experiments/checks/results/sq243_sq241.json`):

| Width | E1 − E10 | Paired 95% CI | Prediction |
|---|---|---|---|
| d = 512 | +0.05% | [−0.33%, +0.44%] | S1 PASS (tie) |
| d = 384 | +0.08% | (Phase 2b, 7 seeds) | |
| d = 256 | **−1.24%** | [−2.24%, −0.25%] | **S2 FAIL** (E1 better) |

- **S3:** a scaling effect is called at the d = 256 end point.
- **Read:** E1's linear mixers help only at the smallest width and the edge is gone by d = 384.
- **But the LRs were not re-tuned per width** (a registered limitation). Small models often want higher LRs, so E10 at 3e-3 may be under-tuned at d = 256.
- **New claim C198, PARTIAL** [exploratory: LRs not re-tuned per width]. Its refutation test is a per-width LR sweep.

**SQ241 param-matched E1** (mlp_hidden 952, 10.62M non-embedding params, 4 seeds):
- **vs E10:** +0.61%. P1 PASS (within ±1%).
- **vs full E1:** +0.53%. P2 FAIL, by 0.03 points over the 0.5% bound (paired CI [+0.05%, +1.21%]).
- **So** part of E1's tie with E10 rests on its 0.5M extra MLP params. This is added to C5 as evidence (event 11).

**Byte Quest:**
- **SQ188 LR sweep, e13b at 20M bytes** (val loss, nats/byte):
  - lr 1.5e-3: 2.757 (the registered extension point);
  - lr 3e-3: **2.712**;
  - lr 6e-3: 2.967;
  - lr 1.2e-2: 2.869.
  - L1 (minimum at 6e-3 or above) and L2 (a flat basin) are both **refuted**. The registered rule puts every E13 arm at **3e-3**.
  - These runs are short (153 steps), and the 6e-3 > 1.2e-2 ordering suggests noise at this length.
- **E15b crashed and is fixed:**
  - `Triton Error [CUDA]: invalid argument` in fla's chunk GDN kernel;
  - cause: the parallel-refine decoder runs GDN over 32 × 1,025 patch groups × 3 heads, more than 65,535 rows on one CUDA grid axis;
  - fix: `GatedDeltaNet._run_kernel` splits the batch, which is exact because sequences are independent (WIP `c064d29`; `test_byte_model` 24/24);
  - the retry benches at 276k tok/s, 10.25 GB.
- **SQ189 benches** (tok/s and peak memory):
  - E13b: 82.1k, 7.41 GB;
  - E14b: 110k, 9.39 GB;
  - E14c: 700k, 13.4 GB;
  - E13a: 368k, 10.7 GB.
- **Now running:** the E13 patch-size sweep at 3e-3 plus E13b at 200M.

**Corpus v3 research (the human's request):** research agent, `notes/research-corpus-v3.md`.
- **Primary:** The Stack v3 train split (rev `159792e7`).
  - ODC-By, not gated;
  - repo-level rows, so we can finally split at repo level;
  - Python is about 5% of bytes, so 2.5B tokens means about 80–105 GB of shards.
- **Fallback:** codeparrot-clean (12.8 GB, about 8B permissive tokens).
- **Downloads are human-only and not done.** SQ276 (a metadata-only shard census, a few GB) comes first.
- **SQ276–SQ281 added** to the side quests.

**Fleet:**
- Wave 6 is running on 5 boxes: the SQ200 init 2×2 and the Inkling loop-variant runs, with SQ282 started.
- w4-1 is on Byte Quest.
- The inference batch finished: the docnll scorer matches the reference scorer within 1e-6 on every checkpoint it scored (max diff seen 3.6e-7). Analysis is pending.
- **Early SQ200 signal** (seed 0/2, analysis pending): E0 with flat init 0.5889/0.5858 vs gogeta init 0.6076/0.6068 in-training val, which points to the init handicap.

---

## v0.4.84: Outside review 00:10: C197 confounded by depth (E34 14% worse at matched params); Inkling 2 Mini loop research; queue reordered
*2026-11-22*

**Outside review 00:10 (Haiku), triaged:**
- **C197 is confounded.** E34 (E9's 5-block layout widened to d=576) has about the same unique params and FLOPs as E32, yet scores **0.2834 vs 0.2486, 14% worse**. So "unique params at matched FLOPs" is not sufficient: at matched params and FLOPs, *distinct depth* beat width here. Both ran at untuned LRs, n=1.
- **Ledger:** C197 gets a limitation (event 4); it stays PARTIAL [exploratory n=1].
- **New items, registered and queued:**
  - **SQ282:** LR-fair depth vs width. E32, E34 and E2 × 3 LRs × 2 seeds at 20M.
  - **SQ283:** C197 seed replication. E32 and E2 at 200M, seeds 1–3.
  - **SQ284:** per-pass unique low-rank adapters in the shared loop. Accepted, needs a model option.
- **Queue reordered:** wave 6, then SQ282/SQ283, then SQ263 (the 200M factorial gate), then F2.
- **E60 (Code-MoE) is gated on SQ282 and SQ238,** because its prior assumes width-like capacity helps.

**Inkling 2 Mini research (the human's request):** research agent, `notes/research-inkling2-mini.md`, cited sources; not every fact re-verified by the main agent.
- **Model:** Thinking Machines Lab, released 2026-11-20. A 26.4B/3.4B-active MoE whose layers 6–17 run 3× with shared weights (layers 0–5 and 18–23 run once).
- **The loop slightly loses on training compute** (7.5%) but wins on parameters and inference.
- **Ingredients we lack:**
  - a closed-gate restart, so the looped net starts out *exactly* its unlooped twin;
  - no residual carry between passes;
  - a fixed r = 3 (we sample U{1..7});
  - thick outer layers (a quarter of the layers on each side; ours is 1/3/1);
  - gated messages plus attention residuals.
- **SQ266–SQ273 added.** The config-only SQ266 (fixed-r sweep) and SQ271 (thick outer layers) are registered as wave 6 part 3: 21 runs, predictions K1–K4.
- **Correction:** `notes/research-looped.md`'s "Inkling 2 is not looped" is true only for Inkling 2 *Small*.
- 3 papers were listed in `refs/fetch_refs.py` and **not downloaded** (human-only).

**Wave 4 so far:**
- SQ243 at d = 512: E1 0.5262–0.5295 vs E10 0.5265–0.5268 in-training val.
- SQ241 param-matched E1: 0.5875–0.5881. Clean80 scoring is pending.
- Inference (SQ248 knockouts) is running.

**Quick reviews:** 23:05 (SQ275, bound the SWA mask cache) and 23:09 (rejected as covered by C190/K1/SQ244). A research agent is searching for a better code corpus (the human's request).

---

## v0.4.83: Claim audit 2302; Code-MoE researched and registered as E60 (planned); SQ238/SQ200 wave-6 registrations
*2026-11-22*

**Claim audit 23:02** (1 claim due; the rest are in their 6-hour cool-down from 22:03):
- C197: the auditor's verdict was WEAK, citing an LR mismatch with E2.
- **Rejected after checking the run:** E2's Phase 2b-5090 run trained at lr 0.006 by CLI override (`meta.json` cli_overrides), the same LR as E32/E33. The 0.003 the auditor saw is the config file's default.
- No ledger change; C197 stays PARTIAL [exploratory n=1].

**Code-MoE (the human's proposal) researched and registered as E60, status planned** (research agent, Opus: `notes/research-code-moe.md`, `experiments/design/code-moe-draft.md`).
- **Literature:** no MoE result exists at our scale; MoEUT's smallest model is 44M, at about +0.5–1.2% vs dense at matched total params.
- **An MoE FFN adds unique params at about zero FLOPs.** That is C197's lever, so the decisive controls are a dense total-param-matched arm and a frozen random router.
- **Semantic expert specialisation is weak in published analyses and largely unmeasurable on code-py-v1:**
  - async, subprocess, hashlib and pickle appear in 0% of val docs;
  - `ast.parse` fails on all algorithmic docs;
  - only about 4 label classes are computable.
- **Stage 1 (about 2.5 GPU-h):**
  - E60a: E1 + a dropless 4×512 top-2 MoE (18.2M non-embedding, 1.0× E1 FLOPs);
  - E60b: a frozen router;
  - E60c: dense SwiGLU 2048, total-param matched;
  - 6 seeds each.
- **Stage 2:** the {E9, E2} × {dense, MoE} 2×2 plus a loop-index router bias. Does MoE rescue the loop?
- **Stage 3:** supervised semantic routing, only if learned routing beats the frozen router.
- **Causality rules:** no capacity dropping or expert-choice routing; region routing uses the previous segment or a prefix summary.
- **Corrected:** 8 experts × 512 is 4× the dense MLP, not 2×.
- **Before any E60 run:** its predictions get registered and `gogeta/moe.py` is written (off by default, bit-identical).
- **Queue:** SQ254–SQ261 added (label tooling, knockout specialisation, router collapse, token/segment/prefix routing, aux objectives, loop routing divergence, Byte Quest × MoE, CPU causality check).
- **References:** 29 papers added to `refs/fetch_refs.py` (section 15-moe). They are **not downloaded**: downloads are a human-only action (AGENTS §0a).

**Wave-6 registrations** (WIP commits before dispatch):
- **SQ238** dedup-only control: the v1-subset builder, config, and predictions D1–D3.
- **SQ200 init 2×2 + SQ234:** E0-gogeta, E0-flat, E10-gogeta and E1-flat at 200M, 4 seeds each. Predictions I1–I4.

**Quick review 23:02 (reproducibility auditor):** SQ274, evidence-value matching in the release lint.

---

## v0.4.82: Opt-in training options (init, LR schedule, gradient accumulation), defaults bit-identical; SQ238 dedup control registered
*2026-11-22*

**Three opt-in training options for wave 6**, each OFF by default, with **defaults bit-identical** to the previous code (subagent; tests re-run by the main agent):
- **`model.init`:**
  - FusionDance gets `"gogeta"` (default: the scaled residual-output init) or `"flat"` (std 0.02 everywhere, HF Llama's scheme).
  - The HF Llama arm (E10) gets `"hf"` (default) or `"gogeta"`/`"flat"`.
  - Unblocks SQ234 (E1 under E10's init) and the SQ200 init 2×2.
- **`train.schedule`:** `"cosine"` (default) or `"wsd"` (warmup, constant, then linear decay over the last `decay_frac`). Unblocks SQ229.
- **`train.micro_batch` / `--micro-batch`:** gradient accumulation at the same effective batch and data order. Gradients match full-batch to 1e-5 relative. Unblocks SQ231.
- **Proof:**
  - A reference captured from the pre-change code (initial state_dict sha256 + 8 training losses as float hex) matches the new code exactly.
  - `tests/test_options.py`: 12 ok, 1 skipped (GDN needs CUDA).
  - The byte model's mirrored `ModelConfig` gets the same field (`test_byte_model` 24/24).
  - The post-training CPU suites pass.
- **GPU verification still to do** (wave-6 smoke): a GDN config with `init: flat` and `--micro-batch` under bf16 autocast; `schedule: wsd` on real data.

**Also in this version:**
- **SQ238** (dedup-only control) is registered and committed in WIP `e449672`: the v1-subset builder (tests 7/7), the config, and predictions D1–D3.
- **Quick review 22:56 (ML theorist):** SQ265, per-pass gradient interference in the shared loop core, the mechanistic test of C197.
- **Research agents running:** Code-MoE (the human's proposal → an E-series item) and "Inkling 2 mini" recurrent design (the human's request).

---

## v0.4.81: Outside review 23:10; waves 4-5 registered and wave 4 running; F2 made seed-blocked with blinded re-estimation; AGENTS 0h pipeline rule
*2026-11-22*

**Outside review 23:10 (Haiku), triaged:**
- **Ideas A/B adopted before any F2 run.** Seeds share the data order across cells, so F2's primary analysis is now **seed-blocked**: a common seed effect explains 35% of the pilot's residual SS, and the blocked sd is 0.0082 vs 0.0090.
- **F2 sizing critique adopted.** The pilot sigma rests on 10 df, so N could plausibly be 80–500. F2 now runs in **two stages with a blinded variance re-estimation** after 120 runs (Kieser & Friede: no alpha inflation), capped at N = 246. Registered in `experiments/checks/PREDICTIONS.md`, wave-5 amendment.
- **New queue items:** SQ263 (a 200M gate on the factorial cells) and SQ264 (reading the factorial per FLOP and per bit).
- **Accepted as a limitation:** SQ243's 4-seed MDE (0.9%) can't resolve a 0.2% gap, so a tie is the expected registered outcome.

**Registered and committed before dispatch** (WIP commits a79b1de…4895120):
- AGENTS §0h **pipeline rule** (the human: "queue the next wave up before the current wave ends");
- **wave 4:** SQ243 two-size check (E1/E10 at d 256 and 512, tokens per param matched, 4 seeds) and SQ241 param-matched E1 (4 seeds); power from SQ253 (MDE about 0.9%);
- the wave-4 inference scripts: per-doc NLL, layer knockout (SQ248), logit ensemble (SQ249);
- Byte Quest stage 1: the SQ188 LR sweep, SQ189 probe and SQ221 determinism via the new `cockpit/remote/fetch_bytes.sh`;
- **wave 5:** factorial F2, re-sized from the pilot (sigma 0.009 → N up to 246), now seed-blocked and two-stage.

**Fleet:**
- Wave 4 is running on 6 rented RTX 5090s ($0.50–0.62/hr).
- Two slow hosts delayed setup: Hugging Face downloads at about 200 MB/min, and a 6-min bootstrap. The next rental needs a higher download floor.
- Stage 1 of F2, the inference jobs and Byte Quest stage 1 queue onto boxes as they free up, with no teardown in between.

**Wave 6 in preparation (subagents):**
- opt-in training options with bit-identical defaults: init (SQ234/SQ200), schedule (SQ229), gradient accumulation (SQ231);
- the v1-subset builder for the dedup-only control SQ238;
- a Code-MoE research review that becomes an E-series item (the human's proposal).

**Hourly docs refresh** (README, sketch, campaigns README) reflects v0.4.79–v0.4.80. Spot-checked by the main agent.

**Quick reviews:** 22:48 data engineer gave SQ262 (family-clustered doc-level CIs).

---

## v0.4.80: Fleet waves 1-3b done (106 runs on 8x RTX 5090, $6.54); claim audit 2203; outside review 22:30; C197 (scale is the lever)
*2026-11-22*

**Fleet waves 1–3b are complete.** 106 training runs, 0 failed, every checkpoint scored on clean80 (`experiments/checks/results/fleet_2026-11-22.json`).
- **Hardware:** 8 rented RTX 5090s (one rental failed and never started), $0.50–0.70/hr each, about 1.2 h per box.
- **Spend:** **$6.54**, reconciled against the provider (total $6.96 to date; $93.04 credit left; `experiments/expenses.json` X4).
- **Avoidable cost:** about $1 of idle time, from two of my mistakes:
  - a worker script written with CRLF line endings, so reassigned boxes died at once (about 10–15 min, 6 boxes, until the human caught it);
  - five boxes pulled but not destroyed for about 10 min.

  Both are fixed, and every box is now destroyed when its queue empties (AGENTS §0h).

**Results** (clean80 bpc, RTX 5090; registered A3/HPO stages, judged by their registrations):
- **Phase 2b-5090** (v0.4.79): E1 ties E10; E11 ≈ E1 (C196).
- **K7-5090 LR grid** (tuning): best LRs are E1 6e-3 (0.4477) and E10 3e-3 (0.4613), the same as the N1X's K7.
- **H1 second seeds:** the same order (E1 6e-3 0.4464 < 3e-3 0.4518; E10 3e-3 0.4572 < 2e-3 0.4764).
- **H2 horizon (200M):** E10 at 2e-3 is 0.2672, at 3e-3 0.2672 (7-seed mean 0.2677), at 6e-3 0.2753. E1 at 3e-3 is 0.2680, against 6e-3's 0.2679. The 20M optima hold at 200M within noise.
- **H4 warmup-30:** worse at every LR (E50 at 3e-3 0.6376; E51 at 6e-3 0.6679), so warmup 300 stays.
- **A3 matched arms (200M, v1, 1 seed):**
  - E30, dense param-matched: 0.2770;
  - E31, dense FLOP-matched: 0.2814;
  - **E32, 14 unshared blocks at E2's FLOPs: 0.2486**;
  - **E33, 12 unshared blocks: 0.2484**;
  - E34, E9 widened: 0.2834.

  E32/E33 are 7.2% better than E1 and 12.6% better than E2, beyond their registered prediction ranges. At matched FLOPs, unique-parameter capacity is the lever, and loops survive at most as parameter efficiency. **New claim C197:** PARTIAL [exploratory n=1].
- **A3 on code-py-v2 (20M):**
  - F0 LR picks by G3: E20 3e-3, E21 6e-3, E22 6e-3, E23 6e-3, E24 3e-3, E25 6e-3, E26 1.2e-2, E27 6e-3, with edge extensions run for E20, E24 and E26.
  - **F1 pilot seed sd is about 0.6–1.2% (log)**, against the 0.4% the factorial power calculation assumed, so **F2 at N = 46 is under-powered**. Sizing F2 needs a new registration; nothing runs before that.
  - F4 loop partners: E35 0.4716 (3-seed mean), E36 0.4403.
  - E1 decomposition (minimal tier, 1 seed): E1 control 0.4342; E40 (GDN → attention) 0.4646; E41 (SWA only) 0.4631; E46 0.4392; E47 0.4380; E48 0.4341. The GDN layers matter (removing them costs about 7%), and moving the attention layer barely matters.
  - H5 dedup bridge (200M): E52 (E10 on v2) 0.2614 and E53 (E1 on v2) 0.2599, both better than on v1 (0.2677 / 0.2679). **Not read as a dedup effect before the dedup-only control SQ238** (v2 changes both coverage and duplication).

**Claim audit 22:03** (8 Codex auditors, 28 claims; verified by a subagent with spot-checks by the main agent):
- No status changes; 6 limitations added (C60, C29, C63, C19, C50, C65); C195 gets evidence.
- The auditor's push to downgrade C196 was rejected: the registered rule is TOST equivalence, and it passed. A limitation records the width of the ±1% margin.

**SQ233 closed:** C1 and C3 now have measured GPU artifacts (RTX 5090: GDN kernels ALL OK; model tests 12/12), in `C1.json` and `C3.json`.

**Outside review 22:30 and quick review 22:31 (statistician):** SQ248 layer knockout (top pick), SQ249 E1+E10 logit ensemble, SQ250 token- vs doc-weighting, SQ251 distillation, SQ252 overlap-cutoff sensitivity, SQ253 power for the two-size check. The token-ratio and ternary legs are folded into SQ243.

---

## v0.4.79: Phase 2b-5090 done: E1 ties E10 (7 seeds), E11 equivalent to E1 (C196), loops don't pay; SQ232 replay floor 0.002
*2026-11-22*

**Phase 2b-5090 is complete** (`experiments/devlogs/PHASE2B-5090.md`, `experiments/checks/results/phase2b_5090.json`).
- **Design:** registered before dispatch (v0.4.74). 19 runs at 200M tokens on 6 rented RTX 5090s, commit `8441ad0`, each box's dataset rebuilt and verified byte-identical. Every checkpoint is scored on clean80.
- **Clean80 bpc means:**
  - E10: 0.26772 (7 seeds);
  - E1: 0.26795 (7 seeds);
  - E11: 0.26710 (3 seeds);
  - E9: 0.2773 (1 seed);
  - E2: 0.2845 (1 seed).
- **R1, E1 vs E10:** E1 +0.08%, paired 95% CI [−0.00087, +0.00132], p 0.64. **A tie.** The sign matches the N1X test (C5 REFUTED, v0.4.78), but the CI spans 0. Over both machines (10 seeds per arm), E1 is within about ±0.4% of a tuned transformer and not better.
- **R3, E11 vs E1:** −0.13%, TOST p 0.002 inside the registered ±1% margin, so they are **equivalent**. New claim **C196, VERIFIED** (3 seeds, clean80, RTX 5090).
- **Holm (m = 2):** neither comparison rejects equality.
- **R2, cross-SKU:** **pass**. 6/6 matched items within 0.0010 val_loss of the N1X; clean80 of N1X vs 5090 checkpoints within 0.0008. C195 gets evidence; it stays PARTIAL because the 3090 half is 20M only.
- **Loops (descriptive, 1 seed):** E2 (looped) is 2.6% worse than its unlooped twin E9 and 6.2% worse than E1. C6 gets evidence.

**SQ232, same-SKU replay floor** (`sq_232.json`):
- Same-seed 5090 replays differ by 0.0022 (E10) and 0.0028 (E1) at 20M tokens. P-R1 (≤ 0.002) fails narrowly.
- So cross-SKU deltas of up to about 0.002 are within run-to-run noise (C195 limitation).

**Registry:** E10, E11 and E9 are now `done` in `experiments/experiments.json`, with their clean80 results; E1 gets its Phase 2b numbers. This fixes the stale "E10 planned" in INDEX that the reviews flagged.

**What this means:** at 13.8M params and 200M tokens, the hybrid mixers (E1) tie a well-tuned transformer, and negative-eigenvalue GDN (E11) changes nothing measurable. The open questions move to:
- scale (SQ243, the two-size check);
- implementation parity of the control (SQ234);
- real wall-clock efficiency (SQ244);
- whether loops or ternary can earn their place: the A3 factorial and controls now running on the fleet.

**Quick review 21:47 (newcomer):** SQ246 (calibration, descriptive) accepted. The main open test is now the FusionDance combination vs its controls.

---

## v0.4.78: C5 REFUTED on the registered primary metric: E1 is 0.18% worse than the E10 control; outside review 21:40
*2026-11-22*

**The registered Phase 2b test (Q12) is in, and C5 is REFUTED.**
- **Inputs:** clean80 doc-level bits/char, the pre-registered primary metric (Addendum A2), on the six N1X Phase 2b checkpoints at tuned LRs (E10 3e-3, E1 6e-3; 3 seeds each). Scoring ran on a rented RTX 5090 so the laptop stayed cool.
- **Scores:**
  - E10: 0.2662, 0.2676, 0.2677; mean **0.2672**;
  - E1: 0.2664, 0.2680, 0.2687; mean **0.2677**.
- **Registered C5 rule:** REFUTED.
  - E1 is **0.18% worse**.
  - The paired difference (n = 3, df 2) is +0.00048 bpc, 95% CI [−0.00056, +0.00151], p = 0.18.
  - Neither condition holds: the gap vs 2 × pooled sd 0.0020 fails, and max(E1) 0.2687 > min(E10) 0.2662.
- **The secondary val_loss verdict agrees:** REFUTED, −0.14%.
- **The memorisation-confound flag is False.**
- **Status:** recorded as **REFUTED** (qualified "vs E10 at tuned LRs, clean80, 3 seeds, N1X"; ledger event 9), per A2: the primary-metric verdict is the status.
- **Why the headline flipped:** the earlier "E1 is 3.6% better" was against our own E0, which is handicapped (the E0-parity rule fails: E0 0.6063 vs E10 0.5845). The outside reviews of 20:37 and 21:40 suspected exactly this.
- **Still open:** the confirmatory Holm family stays INCOMPLETE on the N1X (E11 never ran there). The independent Phase 2b-5090 replication at 7 seeds per arm, running now, is the second check.
- **Early 5090 clean80 scores** (not yet a test): E1 s0, s1, s3, s4 0.2667 / 0.2677 / 0.2684 / 0.2686; E10 s0, s4, s5 0.2660 / 0.2670 / 0.2677.
- **README:** the headline, the results table (an E10 control row added) and the "in plain English" bullet are rewritten to the E10 comparison.

**Outside review 21:40 (Haiku), triaged:**
- **Critiques accepted:**
  - 1: the headline used the wrong control (now fixed).
  - 2: E10 also differs in RoPE layout and eps (a limitation).
  - 3–4: param-matching and size (SQ241, SQ243).
  - 5–7: R1 is weak as registered, so it is reported with its CI and never called a confirmation; the clean80 sd is now measured.
  - 8: source reweighting (SQ239).
  - 9: LSH recall (SQ245).
  - 10–12: the v2 val is novelty-selected and longer, and coverage is confounded. **No v1-vs-v2 comparison is read before the dedup-only control SQ238.**
  - 13: LR picks below the 20M seed sd are ties.
  - 14: copy probe (SQ240).
- **New queue items:** SQ238–SQ245.

**Quick review 21:37 (code-model practitioner):** executable pass@1 is already battery family F, and it can't resolve differences at 13M.

---

## v0.4.77: code-py-v2 built and verified byte-identical on two machines; docs refresh (hourly)
*2026-11-22*

**code-py-v2 is built** (`experiments/data_integrity/build_v2.py` at `b494ff8`, CPU only, run on two rented RTX 5090 boxes alongside their training).
- **Runs:** 26 s on 48 CPUs and 77 s on 64 CPUs, peak RSS about 4 GB.
- **Determinism:** all four files (`train.bin`, `train_keep_ids.npy`, `val.bin`, `docs_val.jsonl`) and the tokenizer are byte-identical across the two machines; only the manifest's `created` timestamp differs. The manifest is committed as `experiments/data_integrity/dataset-v2.manifest.json`.
- **Size:**
  - train: 698,714 docs (70.8% of v1's 987,154) and 202.5M tokens (77.2% of 262.4M);
  - val: 4,000 docs, redrawn (1.39M tokens).
- v2 validation numbers are **not comparable** with v1 val.
- **Unblocks** most of the A3 design (E20–E27, E35, E36, E40, E41, E46–E48, E52, E53). E42–E45 also need the GDN-flags patch.

**Docs refresh (hourly, Haiku), checked by the main agent:**
- README's status line now reflects the N1X hold (6/11 Phase 2b items done), the running Phase 2b-5090, SQ220/C195, SQ214 and v2, and links the expense ledger. The main agent corrected two lines: the box count is seven live of eight rented, and v2 is built, not "scheduled".
- `notes/sketch.md` gets the decisions for v0.4.72–v0.4.75 and the current claim counts.
- The phase2b row in the campaigns README is updated.

**First Phase 2b-5090 results** (200M tokens, in-training val_loss on the contaminated v1 val; clean80 scoring is pending): E1 s0 0.5841 and E1 s2 0.5856 (the N1X had 0.5835 and 0.5866).

---

## v0.4.76: code-py-v2 builder (deterministic dedup); CPU analyses SQ225, SQ228, SQ233
*2026-11-22*

**code-py-v2 builder** (`experiments/data_integrity/build_v2.py`, `notes/data-v2.md`; subagent, tests re-run by the main agent: 7/7 pass).
- **Families:** 64-perm MinHash over token 8-grams, est. Jaccard ≥ 0.5, single linkage over train and val.
- **Train** keeps one doc per family.
- **Val** is redrawn: the v1 val docs first, then seeded train singletons, each with under 30% of its 13-grams in train. Every val doc's whole family leaves train, and so does any train doc sharing a rare def/class name with a val doc (one hop).
- **Checks:** the build asserts the final overlap and family audit.
- **Determinism:** a 1-worker build is byte-identical to a 3-worker build.
- **Status:** it unblocks most of A3 (E20–E27, E35, E36, E40–E53). The full build runs next on a rented 5090's CPU, twice on two boxes, to prove the manifests match. v2 val numbers are not comparable with v1 val (it is redrawn).

**CPU analyses (subagent; numbers spot-checked by the main agent):**
- **SQ225, source mix:**
  - The val/train config mix matches (algorithmic_corpus 63.9% vs 62.9%, chi2 1.70).
  - The K9 near-duplicate overlap is concentrated by source: 52.3% of algorithmic val docs vs 22.9% of synthetic (+29.3 pts, CI 26.6–32.2). So clean-val results lean toward synthetic_code_snippet.
  - **C131:** limitation added.
- **SQ228, repetition vs length** (330 out-of-corpus files, package-clustered bootstrap):
  - 8-gram repeat share correlates 0.48 with log length.
  - Jointly, it absorbs about 27% of the length coefficient (1.55 → 1.13 per SD). Length survives.
  - **C193:** evidence added, no status change (exploratory).
- **SQ233, C1/C3:**
  - C3's attention and SWA causality checks were re-run on CPU and pass (past-token diff 0.0).
  - C1 and the GDN/looped parts of C3 need fla's Triton GPU kernels. Their JSON stores the recorded values, labelled `recorded`.
  - Next: a GPU re-run on a rented 5090.

**Quick review 21:26 (skeptical peer reviewer):**
- **Accepted:** queue reconciliation at the end of each fleet wave.
- **Rejected:** E1 vs E10 at matched FLOPs, because the token-matched comparison is already conservative against E1 (91.0M vs 96.8M FLOPs/token) and E31 covers the dense side.

**PREDICTIONS erratum:** the wave-2 heading's "~21:45" was an estimate. It was actually written at about 21:22, still before dispatch.

---

## v0.4.75: Fleet wave 1 dispatched on 8x RTX 5090; wave 2 (registered A3/HPO items on v1) pinned to one SKU
*2026-11-22*

**The fleet is running** (the human: "spin up several of the cheapest 5090s max 8 concurrent and run all the experiments"; the N1X limited to light work).
- **Boxes:** 8 RTX 5090s rented at $0.50–0.70/hr each, about $5.0/hr in total. They are outside mainland China (Hugging Face is blocked there), verified, reliability > 0.97, and labelled `gogeta-cockpit-w1-1`…`8`.
- **Spend control:** within the human's $30 budget, about $29.58 of it left. The cockpit spend approval (human-only, TTY) was left open for the human to type; until then the boxes are rented through the Vast CLI under the human's chat go-ahead, and each is pulled and destroyed when its job list ends (AGENTS §0h).
- **Wave 1 (registered in v0.4.74):**
  - Phase 2b-5090: E10 and E1 × 7 seeds, E11 × 3, E9 and E2 × 1, at 200M tokens;
  - SQ232 replay pairs;
  - clean80 scoring of every checkpoint on the box;
  - the six N1X Phase 2b checkpoints (E10/E1 × 3, the registered confirmatory Q12 set) uploaded to one 5090 for clean-val scoring, so the laptop stays cool.

  Six boxes were up first, so wave 1 runs on them, balanced at 17–21 min of training each. The other two take wave 2.
- **Wave 2 (registered here, before dispatch):** registered A3/HPO items that run on code-py-v1, judged by their v0.4.40 criteria on one SKU:
  - the K7-5090 LR grid (tuning, no claim);
  - H1 second seeds, H4 warmup-30 (E50/E51) and H2 horizon stage A;
  - the A3 matched arms E30–E34 (1 seed each, as registered).
- **Blocked:** everything on `code-py-v2` (most of A3), until its builder lands. A subagent is writing it, and the build will run on a rented box's CPU. E42–E45 also need a patch.
- **`fetch_data.sh`:** a new `KEEP=1` option keeps `docs_train.jsonl` for the byte and v2 dataset builds.

---

## v0.4.74: Expense ledger; outside review 21:10; fleet wave 1 registered (Phase 2b on 5090s at 7 seeds, SQ232)
*2026-11-22*

**Expenses are now tracked** (the human: "lets track our expenses").
- **Ledger:** `experiments/expenses.json`, append-only: credit purchases, per-rental compute (estimated), and reconciliations against the provider's own numbers. Cockpit access: `bash cockpit/cockpit.sh --expenses [reconcile]`, MCP `gogeta_expenses` and `gogeta_expenses_reconcile` (80/80 tests).
- **Public-repo rule:** amounts and purposes only. No card digits, receipt numbers or links, addresses or e-mails; the reconcile path keeps only the two numbers it needs.
- **To date:**
  - **$100.00** Vast.ai credit bought by the human (2026-11-22 16:54 PST);
  - **$0.4239** compute actually charged, against $0.34 estimated across the 3090 and 5090 pilot boxes; the $0.08 gap is boot, storage and bandwidth time;
  - **$99.58** credit left.
- Auto-billing is off (credit-only), so Vast cannot charge the card on its own.

**Outside review 21:10 (Haiku), triaged:**
- **Critiques accepted:**
  - SQ214's "w64 worse" is not significant at n = 4 (erratum in its devlog), and the C193 limitation rests on the aggregate, not the > 1,200-token bin.
  - E1 vs E10 mixes mixer and init (SQ234).
  - **At n = 3 the Phase 2b verdict sits at about 0.5 SE**, so 7 seeds per arm are needed for a 0.3% effect.
- **New queue items:** SQ234 (E1 under E10's init), SQ235 (delta-rule ablation), SQ236 (membership gap), SQ237 (GDN-only length extrapolation).
- **Quick reviews:** 21:10 reproducibility auditor gave SQ232 (same-SKU replay floor) and SQ233 (machine-readable outputs for the VERIFIED C1/C3).

**Fleet wave 1 registered before dispatch** (`experiments/checks/PREDICTIONS.md`). The human asked for "several of the cheapest 5090s, max 8 concurrent", running every experiment, CPU jobs included, and the N1X limited to light work after its 86.6 °C thermal kill.
- **Phase 2b-5090,** an independent single-SKU replication: E10 and E1 at **7 seeds each**, E11 × 3, E9 and E2 × 1, all at 200M tokens, scored on clean80.
  - The N1X 3 + 3-seed E1-vs-E10 comparison stays the registered confirmatory Q12 test.
  - The two SKUs are never pooled.
  - Predictions R1–R3. The resolvable effect is stated up front: a half-width of about 0.0018 at n = 7, under an assumed sd.
- **SQ232:** same-SKU replay pairs for E10 and E1. Prediction P-R1: |Δ| ≤ 0.002.

---

## v0.4.73: Cloud pilot results: the 5090 is fastest and cheapest per token; SQ214 window sweep (w256 best); cockpit Vast integration; thermal-kill fixes
*2026-11-22*

**SQ220 done: cloud bridge across three machines** (`experiments/devlogs/SQ220.md`, `experiments/checks/results/sq_220.json`).
- **Setup:** the same training code (`gogeta/`, `configs/`, `tools/` identical at `9712842` and `bd03492`), the same data sha256, seed 0, lr 3e-3, 20M tokens.
- **Loss transfers.** val_loss vs the N1X:
  - E10: 3090 +0.0020, 5090 +0.0011;
  - E1: 3090 +0.0007, 5090 −0.0020.

  All are within the ±0.004 tolerance. New claim **C195** (PARTIAL: seed 0, 20M tokens, two SKUs). E1 < E10 on every machine.
- **Speed (steady-state median tokens/s):**

| | N1X | RTX 3090 | RTX 5090 |
|---|---|---|---|
| E10 | 202k | 272.5k | **845k** |
| E1 | 138k | 254.5k | **743k** |
| $/hr | ~0.02 (electricity, assumed) | 0.232 | 0.523 |
| $ per 1B E1 tokens | ~0.04 | 0.253 | **0.196** |
| 200M-token E1 run | 24.2 min | 13.1 min | **4.5 min** |

- **The 5090 wins both value per dollar** (23–27% cheaper per token than the 3090) **and time** (about 3× the 3090). The 3090-part "overhead-bound" reading was wrong.
- **Prediction outcomes:** P3 was a partial miss (3090 only 1.35×/1.84×). 5-P2 and 5-P3 passed.
- **Per-box fixed cost** is about 3.3 min: 76 s env from scratch, about 90 s on-box dataset rebuild, about 30 s cold compile. Nothing is uploaded from the operator's machine.
- **Spend:** about $0.35 in total. Both boxes were destroyed as soon as their work ended (AGENTS §0h).

**SQ214 done (window sweep part; exploratory)** (`experiments/devlogs/SQ214.md`, `experiments/checks/results/sq_214.json`):
- **Design:** E1's two SWA layers at window 64/256/1024, 4 seeds each, 20M tokens, one RTX 5090.
- **Clean80 bpc means:** w64 0.4509, **w256 0.4475**, w1024 0.4533.
- **Full context is worse:** +1.30% (paired t 5.1, df 3, all four seeds the same sign). w64 is +0.76%.
- **Registered predictions:** W1 fail, W2 pass, W3 fail. The decision rule says no confirmatory run.
- **C193** gets a limitation (event 4): the "SWA window caps long-doc context" explanation of its > 1,200-token reversal loses support. SQ226 (scorer chunking) and SQ219 (context removal) are next.
- The 20M-token seed sd is 0.0017 bpc (0.39%), a by-product for SQ216.

**Incident: N1X thermal kill, then an unwanted auto-resume** (post-mortem):
- **Cause:**
  - The N1X reached 86.6 °C (limit 85 °C) during Phase 2b item 7 (E11 s0), and the human ordered a kill. The campaign runner and the run were SIGTERMed; the partial E11 run is discarded.
  - The watchdog classified the runner's exit 143 as "transient" and auto-launched `phase2b-replication-resume-1`. E1 s0 ran on the N1X from 20:53 until a session/WSL restart killed it around 21:01.
  - Separately, the launcher only gates *new* launches on temperature; nothing throttles a running job.
- **Fix:**
  - Exit 143/130 is classified `operator` and never auto-resumed.
  - A new operator hold (`cockpit.sh --hold`, human-only `--release-hold`) blocks every resume, and the TUI header shows it.
  - Two regression tests were added (watchdog 24/24).
- **Prevention:**
  - The hold is set now, per the human's "Hold for now". Phase 2b stays at 6 of 11 until the human releases it.
  - A running-job thermal guard is queued.

**Cockpit Vast integration** (cockpit subagent, verified by the main agent: 77/77 tests, MCP probe, headless TUI check):
- **Plugins:** a provider interface with Vast plus a bring-your-own SSH box.
- **57 MCP tools:** read-only status/offers/cost; confirm-gated setup/data/run/pull/down; budget-gated up/session/dispatch.
- **Guards:** spend checked before every rent under a human-only approval (`cockpit.sh --approve-spend`, TTY). AGENTS §0h is enforced in code: pull then destroy on an empty queue, a 10-min idle kill, teardown in `finally`.
- **TUI:** Cloud (`k`) and Fleet (`j`) tabs. The Fleet tab now polls live on open and every 15 s; it used to poll only on its Refresh button.
- **Launcher:** the `cockpit.ps1` wslpath bug is fixed.

**Reconnection** (the human: "make sure the reconnection is smooth and easy to do if not automatic"):
- `cockpit.sh --reconnect` restarts the watchdog if it is down, then prints the hold, GPU, campaigns, incidents and live cloud instances.
- Opening the TUI also starts the watchdog. After the restart the watchdog had been down, and nothing auto-started it.
- AGENTS §0f now makes reconnecting the first step after any restart, and records the stop-means-stopped rule.

**Quick review 20:50 (systems/hardware):** SQ231 accepted (a micro-batch throughput curve with gradient accumulation, which keeps the optimisation identical).

---

## v0.4.72: Outside review 20:40; cloud kill-on-empty rule (AGENTS 0h); on-box dataset rebuild verified byte-identical
*2026-11-22*

**Outside review 20:40 (Haiku), triaged** (`notes/reviews/2026-11-22-2040-review.md`, footer):
- **SQ226 (top pick, CPU): reset-matched scoring control.** C193's > 1,200-token reversal sits exactly where the scorer switches to chunked 1,024/512 windows, which restarts GDN state, so the "length effect" may be a scorer artifact.
- **SQ227:** document-boundary carry-over.
- **SQ228:** in-document repetition vs length as the covariate.
- **SQ229:** schedule robustness of the E1-vs-E10 order.
- **Idea E adopted before any SQ214 run.** The window sweep goes from 2 to **4 seeds per arm** as a pre-run amendment in `experiments/checks/PREDICTIONS.md`, with W1–W3 unchanged and a scoring caveat pointing to SQ226.
- **Rejected:** moving the Phase 2b verdict to the frozen test split (a post-registration change of the confirmatory set is a forking path; it stays an extra check, SQ218).

**Quick review 20:40 (ML theorist):** SQ230 (a loop-pass representation probe on E2) is accepted.

**New rule, AGENTS §0h (the human, 2026-11-22: "kill them unless we have no work queued up... if there is no more work pull down"):**
- Rented GPUs live only while work is queued: rent just in time, size the fleet to the queue.
- When a box's last job ends, pull its results and destroy it. A box idle for more than 10 min is destroyed.
- Batch runs per box to amortise setup.
- Status updates list every live box with its spend vs the $30 / 12 h budget.
- Applied right away: the 3090 pilot (instance 62070434) was destroyed when its work was done, after about 40 min, roughly $0.15.

**The setup is now clean end to end:**
- `cockpit/remote/fetch_data.sh` rebuilt the whole `code-py-v1` dataset on a fresh RTX 5090 from the public Hugging Face source (2.79 GB in about 40 s on the box's link) in **46 s** of build time.
- All four files match `experiments/data_integrity/dataset.manifest.json` byte for byte: `train.bin` `db61140a…`, `val.bin` `32e4b0ed…`, `docs_val.jsonl` `f821d119…`, `tokenizer.json` `0e75f39f…`. The builder is deterministic on a different machine.
- `bootstrap.sh` from scratch took 76 s on the 5090.
- From rent to ready, nothing is uploaded from the operator's machine (the 3090 pilot spent 494 s uploading).
- **Two fixes:**
  - `python -P`, because `tools/queue.py` shadowed the stdlib `queue` that pyarrow imports;
  - an inline comment had swallowed the script's error handler.
- **Live:** RTX 5090 instance 62073471 ($0.523/hr) is running SQ220 part 2 (bridge) and the SQ214 sweep (12 runs plus clean-val scoring); results come in the next entry.

---

## v0.4.71: Cloud pilot: the 3090 reproduces N1X losses; claim audit 2005 integrated; SQ214 window sweep registered
*2026-11-22*

**What:** the first rented GPU run (SQ220, 1x RTX 3090 on Vast.ai, approved by the human: "get 1 working first", $30 / 12 h cap). Also the 20:05 claim-audit verifier's findings, three quick reviews, and the SQ214 window sweep registered for a rented RTX 5090.

**SQ220, the 3090 part** (`experiments/devlogs/SQ220.md`):
- **Setup:** the same commit (`9712842`, clean clone, `--require-clean`), the same sha256-verified data and the same software stack as the N1X, with K7's bridge runs at seed 0, lr 3e-3, 20M tokens.
- **Loss transfers:**
  - E10: 1.0306 vs the N1X's 1.0286 (Δ +0.0020);
  - E1: 1.0078 vs 1.0071 (Δ +0.0007).

  Both are within the provisional ±0.004 reproduction tolerance (P1 passes, `STATISTICAL`). E1 < E10 holds on both GPUs.
- **Speed (steady-state median tokens/s):**
  - E10: 272.5k vs 202.0k, **1.35×**;
  - E1: 254.5k vs 138.0k, **1.84×**.

  The registered 2–3× (P3) is a partial miss: at 13.8M params the runs look overhead-bound, not bandwidth-bound.
- **Whole-run throughput on a fresh box misleads:** E1 was 0.94× because of the cold Triton/fla autotune cache, about 1 min of fixed cost per box.
- **Incidents:**
  - the `pytorch/pytorch` image has no C compiler, so Inductor failed (fixed with apt build-essential);
  - the 525 MB dataset took 494 s over the operator's link.

**Making the next rental quick (the human asked: "make this easier for the next time"):**
- `cockpit/remote/bootstrap.sh`: a one-shot, idempotent box setup. It runs apt (compiler, git), uv, the pinned Python 3.12 venv (torch 2.14.1+cu130, triton 3.8.0, fla 0.6.0, plus transformers 5.24.0 and einops 0.8.2, which `requirements.txt` omits), the clone at the exact commit, and a `torch.compile` GPU smoke test. Verified on the 3090: a re-run takes 19 s and ends in `ENV OK`.
- `cockpit/remote/fetch_data.sh`: rebuilds the dataset on the box from the public Hugging Face source with the committed tokenizer, and verifies it file by file against `experiments/data_integrity/dataset.manifest.json`. A mismatch falls back to the sha256-checked upload and is never used for training. The live rebuild test was still running at commit time; its result goes in the next entry.
- The cockpit's provider-pluggable fleet tooling (Vast first, MCP-driven, budget-guarded) is being built by a subagent and lands separately.

**Claim audit 2005, checked by the main agent:**
- **C192 PARTIAL → REFUTED.** The registered pooled criterion refutes it, and the earlier move to PARTIAL rested on a post-hoc length cut, which `notes/statistics-policy.md:49` says can never change a status. The length finding stays in C193.
- **C71 is superseded by C194**, the same KDA decode claim without the contrast with GDN that our own data contradicts.
- **C16 UNVERIFIED → INCONCLUSIVE.** K3 ran but its validity gate failed (an invalid control is INCONCLUSIVE per the ledger README).
- Limitations were added to C193, C121, C125 and C93, and evidence pointers to C67, C68 and C69.
- Three auditor verdicts were rejected after checking the PDFs (C30, C35, C76).

**Quick reviews:**
- 20:20 cheapest-experiment hunter: rejected.
- 20:24 statistician: SQ224, leave-one-seed-pair-out fragility.
- 20:32 data engineer: SQ225, source-config mix audit.
- The 20:10 safety review (SQ223) is filed.

**Registered (before any run):**
- SQ214 window sweep: E1 with SWA window 64/256/1024, seeds 0–1, 20M tokens, on one rented RTX 5090. It is exploratory, with clean-val bpc as the primary metric. Predictions W1–W3, configs hash-bound in `experiments/checks/PREDICTIONS.md`.
- SQ220 part 2 (5090 bridge): 5-P1 to 5-P3.

**Docs refresh (hourly):** README and sketch were updated with the C192/C193/C16 states and the Byte Quest registration facts. The refresh agent's "no instance rented" line was corrected by the main agent.

---

## v0.4.70: Outside review 20:37: E1's headline lead over E0 may be a baseline artifact
*2026-11-22*

### Outside review 2026-11-22 20:37 (Haiku)
`notes/reviews/2026-11-22-2037-review.md`, with the main agent's triage.
**Key finding (computed; contaminated val; provisional):**
- E10 (HF Llama at E0's exact shapes) averages 0.5845 over 3 seeds, **~3.7% below our E0 (0.6063)**, about the size of the C5 headline (3.6%).
- **E1 seed 0 at its tuned LR (0.5835) is only ~0.2% below E10's mean.**
- So E1's headline lead over E0 may be largely an E0 baseline artifact (init; SQ200).
- The registered Phase 2b verdict already compares E1 with **E10** on clean-val doc-level bits/char over 3 seeds.
- Recorded as a **C5 limitation event** and in the README status paragraph (with C193).

**Ideas:**
- I1 (packed vs doc-level protocol re-score, with length strata): merged into A2's post-campaign doc-level re-score plus SQ214;
- **I2 → SQ222** (dtype parity E1 vs E10);
- I3 (LR horizon check): already in A3's design and queued for the cloud fleet.

### README version badge fixed
The badge was stuck at v0.4.65. v0.4.66 skipped the README bump (another agent was editing it), and later bumps searched for the exact previous version. The release helper now replaces any version string.
### Cloud monitoring
The cockpit TUI is running in the human's terminal. The Vast.ai fleet integration (SSH monitoring, just-in-time rentals, 5-minute idle teardown) is being built; the SSH key is set up; no instance is rented yet.

---

## v0.4.69: Byte predictions bound to config hashes; Vast.ai CLI installed (no rentals)
*2026-11-22*

### Byte Quest registration hardening (quick review 19:56, reproducibility auditor, Codex)
- **The E12–E16 predictions are bound to exact configs:** the sha256 of all 30 staged byte configs was appended to `experiments/checks/PREDICTIONS.md` ("Config binding…"), still before any byte run. Any change needs a new dated entry.
- The EOS 0x00 collision check was already in the builder.
- **SQ221:** byte-dataset rebuild determinism (before the first byte run).

### Vast.ai CLI (the human's request: install it on Windows and hook it to the cockpit)
- `vastai` 1.11.2 is installed for the Windows user Python (ARM64) at `%APPDATA%/Python/Python312-arm64/Scripts/vastai.exe` (not on PATH; the cockpit will call it by full path). It's also installed in the WSL cockpit venv.
- **Windows ARM64 workaround:** vastai pins `cryptography==49.0.0`, `pillow==12.2.0` and `pycares==4.11.0`, which have no ARM64 wheels, so it was installed with `--no-deps` plus binary-only dependencies (cryptography 46.0.3, pillow 13.0.0, pycares 5.2.1). **Tested:** `--version`; `show instances` authenticates (0 instances); `search offers` returns 37 verified RTX 3090 offers, cheapest $0.113/hr.
- **The API key is NOT in the repo** (checked with `git grep` and a tree scan). It lives in the human's `Desktop/shared/api keys.md` (added as `VAST_API_KEY` in its env block, at the human's request) and in vastai's own config (`~/.config/vastai/vast_api_key`, Windows).
- **No instance was rented.** Renting spends money and needs the human's explicit go-ahead per rental (AGENTS §0a); the SQ220 pilot is still parked.

---

## v0.4.68: Byte Quest: model family built (staged), byte dataset, causality test passes; predictions registered
*2026-11-22*

### Byte Quest build (E12–E16; subagent, verified by the main agent)
**What**:
- The byte-level FusionDance family, as **staged patches** (`notes/patches/byte-model.patch`, `byte-train.patch`; `git apply --check` OK; applied between campaigns) plus a CPU-testable working copy in `experiments/byte_model/`.
- Variants: flat byte-E1; patched coarse-to-fine with P = 2/4/8/16, the one-patch causal shift, a byte encoder, an E1-style global body on patches, expansion, and a shared byte refinement loop × r (gated or contractive updates, re-injection, pass embedding, per-pass loss); an AR or **parallel_refine** decoder; optionally a looped or ternary global core.
- 30 staged configs in `experiments/staged-configs/byte/` (200M-byte and 20M-byte variants). Every config sets `max_seq_len` to its byte context (the RoPE cap from the 20:07 review). The windows match across the granularity sweep (1,024 bytes); byte layers are GDN + conv.

**Dataset `code-py-v1-bytes`**: the same docs and split as code-py-v1 (val 4,000 docs / 3.94M bytes; train 987,154 docs / 976.9M bytes; uint8; EOS = 0x00, never inside a doc). **Every doc round-trips byte for byte**, and the clean-val ids map one-to-one (the manifest sha256 matches). The locked test splits weren't read.
**SQ206 causality: PASS.** 42 variant × P combinations and 412 perturbations at patch starts, middles and ends are all bitwise-unchanged. The **broken no-shift control leaks in every case** (4–11 of 8–11 perturbations).
**Tests**: `test_byte_model.py` **24/24 OK** (re-run by the main agent, 44 s). Defaults are bit-identical to `gogeta.model` for E1/E2/E3/E11-style configs. The trainer smoke test passes on CPU.
**Costs** (estimates, training FLOPs per BPE-token-equivalent):
- E12 flat 325M (3.57× E1);
- E13 P = 2/4/8/16: 172/95/60/43M;
- E14 r = 2 shared = 2-pass unshared 109M, r = 4 138M;
- E15 parallel r = 1/2/3: 95/112/129M;
- E16 looped 204M, ternary 109M.

They match `notes/analysis-coarse-to-fine.md`. About 7 GPU-h per seed overall (to be replaced by SQ189's measurements).
**Predictions REGISTERED** (`experiments/checks/PREDICTIONS.md`, "Registered Byte Quest predictions E12–E16"; plan `notes/byte-quest.md`), e.g.:
- E12 1.15–1.40× BPE-E1 bits/char at matched FLOPs;
- E13: P = 4 best or tied with P = 2, P = 16 ≥ 5% worse;
- **E14: shared r = 2 within ±0.5% of equal-FLOP unshared;**
- E15: parallel r = 2 within 1% of AR;
- E16: ternary global core +3–9%.

Every byte result is reported per doc-length bucket (C193).
**Not verified**: anything on GPU (fla, bf16, compile, throughput, memory); byte LRs (SQ188 runs first); E12 needs the flex-attention patch.

### Also
- `notes/compute-rental.md`: a Vast.ai price snapshot (19:50). The RTX 3090 (~$0.20/hr median) is the cheapest card that runs everything; 8× is recommended; renting is the human's call.
- **SQ220** cloud pilot (parked for the human).
- **SQ219** context-removal intervention (from the 19:44 ML-theorist review).

---

## v0.4.67: Byte Quest on the site and README: diagram, 3D concept model, Start-here path
*2026-11-22*

**What** (the human: "update the uarch graphics and documentation in the gh pages site and the readme, byte level tokens are a thing now"; a subagent built it, the main agent verified it):
- **README:** a "Byte Quest: bytes are a main quest now" section, covering:
  - the new SVG diagram `docs/assets/byte-quest-c2f.svg`: byte encoder → patches → global stack → **causal one-patch shift** → byte refinement loop × r → 256-way head; transparent and mid-tone, for GitHub light and dark mode;
  - why bytes, the coarse-to-fine idea and its estimated cost;
  - the E12–E16 table linked to the cards;
  - the honest caveat (bytes trail BPE at our scale; the claims are about granularity and loops).

  Also from the newcomer review: the **stale SQ99 sentence is replaced** ("no evidence of E1's in-corpus advantage out of corpus; CI −0.1%..+1.0%"), and there's a **"Start here" path** with "if the README and the ledger disagree, the ledger wins".
- **Landing page:** a `#bytes` section with the diagram, the E12–E16 table, and links to cards, devlogs and the 3D page.
- **3D explorer:** a **"C2F-byte (concept)" model** (key 8, `?model=c2f_byte`): byte cubes, encoder, patch tiles, E1's real global stack on patches, the shift lines, a raw-byte re-inject skip, a byte refinement ring with ghost pillars per extra pass, and the 256-way head. It has a **P slider (2/4/8)**, r limited to 1–4, 5 new legend rows and a "design concept: estimates, not a trained model" banner. The power-level readout reproduces the analysis table at P = 4 (95/108/135 M = 1.04/1.19/1.48× E1); P = 2/8 are labelled extrapolations.
- The site build lists E12–E16 automatically (experiment pages E12–E16.html).
- The Learn page is generated from `site-src/learn.md`, so the "Bytes vs tokens" explainer is the unapplied patch `notes/patches/site-bytes-learn.patch`.

**Verified by the main agent** (browser pane):
- **labels stay at 22 → 22** across full sweeps of every slider and the self-test;
- `__arch3dSelfTest(3)` passes;
- a second run is perfectly flat (geometries 45 → 45, textures 3 → 3); the first-run +5 is a one-time cache fill;
- no console errors;
- the scene renders as designed (screenshot checked).

The subagent reports 375 px width with no horizontal scroll.
**Not verified**: the SVG on a pure white background; the 3D visuals at P = 2/8.

---

## v0.4.66: SQ99 reversed: E1's advantage is length-dependent, not corpus-specific; outside review 20:07
*2026-11-22*

### SQ99 length re-analysis (from outside review 20:07, idea I1; CPU, saved per-doc NLL)
The out-of-corpus docs average about 1,210 tokens vs about 300 in corpus (computed), so the reviewer suspected a length confound. **Confirmed.** E1-vs-E0 gap by doc length (package-clustered 95% CIs):
- 0–300 tokens: **−3.86%** (−4.94..−2.60, 29 docs);
- 300–600: −2.39% (−3.39..−1.32, 52);
- 600–1,200: −0.56% (−1.13..+0.11, 96);
- **> 1,200: +1.09% (+0.48..+1.74, 153).**

**On short out-of-corpus files E1's advantage matches its in-corpus size (−3.44%);** the pooled null is driven by long files.
- **Reading:** SQ99's "does not transfer" was mostly a **length/long-context effect**, not a corpus effect. E1 (1 attention, 2 SWA-256, 3 GDN) may fall behind E0 (6 full-attention layers) when context beyond a few hundred tokens matters. Length is confounded with file type, and this is exploratory (n=1).
- **Ledger:** new claim **C193** (the gap shrinks and reverses with doc length), PARTIAL; **C192 REFUTED → PARTIAL** ("transfers for short docs, reverses for long").
- Equal-source weighting (statistician quick review 19:39) barely changes the pooled gap (+0.31% vs +0.42%; median source +0.04%).

### Outside review 2026-11-22 20:07 (Haiku)
**Accepted:**
- **SQ214, the top pick:** per-position loss bins plus an E1 SWA window sweep 64/256/1,024;
- SQ215: a recurrence-free long-conv control;
- SQ216: decoder-only noise floor before X1;
- SQ217: weight EMA;
- SQ218: the frozen test split as the confirmatory out-of-corpus set.

**Design fixes:**
- X1 can't cache features (about 154 GB), so it recomputes on the fly on A1's deduplicated train;
- X1 adds arm A″ (param-matched to the shared arm);
- **RoPE is capped at max_seq_len = 1,024**, so byte configs must raise it (forwarded to the byte-model builder).

### Also in this version
- The hourly docs refresh (19:3x), which updated `notes/sketch.md`.
- Quick reviews triaged: 19:25 skeptic (SQ213 cluster CI, in v0.4.65), 19:29 cheapest-experiment hunter (SQ189 scope extended), 19:34 newcomer (README fix and Start-here path, being applied by the site agent), 19:39 statistician (source weighting).

---

## v0.4.65: Byte Quest becomes a main quest (E12-E16); SQ99 cluster-robust re-analysis
*2026-11-22*

### Byte Quest: a main-quest arc (the human: "add byte level tokens as a main quest, let's experiment with its granularity and loops")
**Registered** in `experiments/experiments.json` (arc "Byte Quest (granularity x loops)"), with cards and micro devlogs:
- **E12** flat byte-level E1 (vocab 256, T = 4,096 bytes);
- **E13** patch-granularity sweep, P = 2/4/8, single-pass byte decoder;
- **E14** byte refinement-loop sweep, r = 1/2/4 shared vs an equal-FLOP unshared control (H-C2F);
- **E15** parallel-then-refine decoding within a patch (r < P);
- **E16** the best decoder × looped global core × ternary global core.

**Design**: `notes/analysis-coarse-to-fine.md`. The primary metric is clean-val doc-level bits/char, plus exact-span accuracy (SQ209).
**Prerequisites**: the SQ206 causality test, the SQ189 throughput probe, the flex-attention SWA patch (C12) and SQ207 (speller on frozen E1).
**In progress:** a subagent is building the byte model family as staged patches (Phase 2b is running, so code paths can't change): byte dataset `code-py-v1-bytes`, causality tests, param/FLOP accounting, staged configs and `notes/byte-quest.md` with draft predictions. **Predictions are registered before any run.**
### SQ99 re-analysis (quick review 19:25, skeptical peer reviewer, Codex)
The per-doc bootstrap ignored package clustering. From the saved per-doc NLL:
- **by-source cluster bootstrap 95% CI −0.13% to +1.03% (includes 0);**
- leave-one-source-out +0.26% to +0.56% (the sign never flips);
- E1 better on 14/31 sources.

**The verdict DOES NOT TRANSFER stands** (no evidence the in-corpus advantage carries out of corpus), **but "E1 is worse out of corpus" is withdrawn.** Recorded as a C192 limitation event; SQ213 done.

---

## v0.4.64: README carries the SQ99 warning (correction to v0.4.63)
*2026-11-22*

**Correction**: v0.4.63's entry said the README status carries the SQ99 warning, but the edit didn't apply (the hourly docs refresh had reworded the anchor sentence), so v0.4.63's README didn't contain it. This version adds it to the README status paragraph: on out-of-corpus Python, E1's advantage over E0 disappears (+0.42% worse, vs −3.44% better in corpus); exploratory; it will be re-checked with the tuned, replicated Phase 2b checkpoints.

---

## v0.4.63: SQ99: E1's advantage does not transfer out of corpus (exploratory)
*2026-11-22*

### SQ99: out-of-corpus transfer (registered v0.4.54)
**What**: `experiments/checks/sq_99_out_of_corpus.py` on the CPU (26 min, fp32, E1's GDN via the plain-PyTorch reference). It scored the v0.3.0 E0 and E1 seed-0 checkpoints on **330 local Python files** (stdlib + installed packages; mean 13-gram overlap with train 0.31%; 400k tokens) and on a **token-matched sample of 1,340 val_clean80 docs**, both through the same scorer.
**Results** (`experiments/checks/results/sq_99.json`):
- Out-of-corpus bits/char is **6.3–6.5×** the in-corpus value for both models (format plus content shift).
- **The E1-vs-E0 gap is +0.42% out of corpus (E1 worse; 95% paired BCa CI +0.06% to +0.78%), vs −3.44% in corpus (CI −3.82% to −2.88%).**
- **Verdict: DOES NOT TRANSFER.** Prediction (2), that E1 still wins out of corpus by less, failed.

**Interpretation (cautious)**:
- The hybrid's in-corpus advantage may be specific to this corpus: memorising its repeated templates (K9, A1 families).
- **But** this is exploratory: single-seed checkpoints at an untuned LR, and **E0 may be a handicapped baseline** (its scaled residual init differs from Llama's, SQ200). The handicap would act on both sets, so it's unclear how it moves the gap. Out-of-corpus is also confounded with document format.
- **The decisive re-run** uses the Phase 2b checkpoints (E10 as the baseline, E1 at its tuned LR, 3 seeds).

**Ledger**: new claim **C192** ("E1's lower loss transfers to out-of-corpus Python") is **REFUTED** [exploratory n=1]; a limitation was added to **C5**. Card `experiments/cards/SQ99.md`, micro devlog `experiments/devlogs/SQ99.md`. README status carries the warning.

---

## v0.4.62: A4 eval battery (final): MQAR fixed by a mastery curriculum; state-tracking probe; iterative suite
*2026-11-22*

### A4 evaluation battery (rigor workstream, final hand-back; snapshot was v0.4.47)
**What**: battery v1 spec (`notes/eval-battery.md`): 37 cells in 4 families (LM, functional coding, memory/retrieval, iterative computation), each with a statistic, chance level, **a required positive control with a pass gate**, CI arithmetic and a CPU/GPU split. Code in `experiments/evalbattery/` (`mqar.py`, `itercomp.py`) and `evals/battery/` (a code state-tracking probe, scorers, 23 lm-eval task YAMLs). **Tests: 10/10 OK (re-run by the main agent).**
**Findings** (reported by the subagent, CPU, tiny models; single-seed diagnostics):
- **K3's MQAR failure diagnosed and fixed.** The task was fine; training sat on the induction plateau. P=16 pairs reached only 14–24% with any lr, width or fixed curriculum, but a **mastery curriculum** (double the pairs whenever train accuracy reaches 90%) reaches **100% at P=16 on seeds 0 and 1**, with the no-memory control at 2.5% (gate 11.3%). The K3 geometry (P=32/64) is a GPU item (QB1).
- **Iterative suite:** modadd5 and parity pass their validity gates. graph_hops/graph_reach and program are not yet validated (shortcuts or at chance), so they're excluded until the GPU gate passes. A hint: looped beats a 4-layer stack at 2× length on modadd5 (37.0% vs 28.2%, one seed).
- **Code state-tracking probe** (the outside review's idea, SQ90): E0 is at floor (31% vs 25% chance, flat in k and distance), while the **positive control solves it** (a tiny LM trained on probe snippets: 100% / 96.3%), so the probe is valid and E0 just can't do it. Proposed rule: a cell counts only if its CI lower bound is ≥ 35%.
- **Line-shuffle control:** bpb 0.240 → 0.787 (gap 0.547 against a 0.15 gate).
- **CI arithmetic:** 1,068 items → ±3 points. HumanEval+ (164) can't resolve below 3 points, so it's a sanity check only.

**GPU plan:** 10 queue proposals (QB1–QB10, about 375 GPU-min; the minimum informative set is about 175 min), all `ready: false` until their predictions are registered.
**Blocker for the human (parked):** `flash-linear-attention==0.6.0` in `~/.venvs/gogeta-eval`, so lm-eval can score GDN checkpoints (QB10).
**Caveat:** `proposed-predictions.md` was written after the CPU diagnostics; those numbers are labelled diagnostics, not predictions.
### Quick review 19:21 (safety/sandbox reviewer, Codex)
**SQ211** (sandbox escape probes: fd inheritance, /proc, symlinks) and **SQ212** (secret canaries through results, logs, the cockpit DB and run forms) accepted. No RL runs on untrusted completions exist; SQ61 gates any real GRPO.

---

## v0.4.61: Claim audit 2026-11-22-1902: 23 findings verified, 5 VERIFIED claims downgraded
*2026-11-22*

### Hourly claim audit 2026-11-22-1902 (AGENTS §0d-ter; first run)
**What**: 8 read-only Codex GPT-6 Luna auditors ran in parallel over 48 claims (settled first), each in 1.5–3.7 min, with no failures. Verdicts: **25 SOLID, 15 WEAK, 7 NEEDS-EVIDENCE, 1 WRONG.** Every non-SOLID finding was checked against the primary files by a verifier subagent (Opus), then spot-checked by the main agent. All changes went in as ledger events through the API. Audit files with `## Triage` footers are in `notes/reviews/claims/2026-11-22-1902-b1…b8.md`.

**Downgrades:**
- VERIFIED → PARTIAL:
  - **C18** (looping solves parity: n=2 seeds, and neg-eig GDN solves it in one pass);
  - **C8** (compression ratio on contaminated val; exploratory n=1);
  - **C43** (neg-eig parity: n=2 seeds);
  - **C44** (2 seeds, 2 protocols);
  - **C100** (harness wiring, tested on attention/SWA only).
- **C127** → UNVERIFIED: the PTX doc covers instructions only, and the Triton and FA3/FA4 compatibility parts were inferences.
- C5, C6 and C92 are re-qualified as PARTIAL [exploratory n=1, K9-contaminated].
- C108 and C22 stay REFUTED, with an exploratory tag.
- **C15 → SUPERSEDED** by **C190**. C15 bundled GDN (fixed memory, true) with SWA (K1: memory exponent 1.953, false). C190 states the GDN half alone, VERIFIED from K1.
- **C107's misfiled K9 row** was split out as **C191** ("the val split is exact-dedup only", VERIFIED from K9.json and `tools/prepare_data.py:9`); C107 reverted to PARTIAL.

**Strengthened with pointers:** C10, C3, C1 and C109 got evidence events and stay VERIFIED. C109 was re-read in the installed lm-eval 0.4.13 YAML.
**Auditor errors caught:** C107's limit-8 eval JSON isn't empty (pass@1 0.875 on 8 of 2,555 problems). C108 → PARTIAL would contradict the policy, which keeps REFUTED.
**Result**: claims-status is now **18 VERIFIED (was 23), 20 PARTIAL, 6 REFUTED, 2 WRONG FOR OUR CODE, 72 UNVERIFIED, 1 SUPERSEDED; 119 claims.** All 48 verdicts are recorded as `audit_verdict` events and in `claim-reviews.json`, putting those claims in their 6-hour cool-down (71 due next hour). Lint: 0 fail, 0 warn.
**Notes**:
- The events carry version stamp v0.4.59 (the verifier's working version); they're committed here in v0.4.61.
- The actor is `subagent:main-agent-verifier`, as the lint requires.
- Unresolved: C104 needs the OpenCoder paper in refs; C127 needs a local probe.

### Quick review 19:16 (code-model practitioner, Codex)
The repo-edit generate-and-test benchmark was rejected because it floors near 0 at 13M (C102). It's accepted in modified form as **SQ210**, a patch-likelihood probe on SQ99's out-of-corpus pool.

---

## v0.4.60: Coarse-to-fine architecture analysis; MambaByte speculative decoding research
*2026-11-22*

### Coarse-to-fine FusionDance (the human's architecture proposal, analysed by the main agent)
`notes/analysis-coarse-to-fine.md`: a patch backbone, then patch-to-byte expansion, then a shared-weight byte refinement loop.
- **The skeleton is known** (MegaByte, BLT, H-Net, AU-Net), and the causal one-patch shift is MegaByte's.
- **What's new:** recurrence used as high-resolution refinement after a cheap global pass. The hypothesis is H-C2F: shared refinement beats both a single-pass decoder and **an equal-FLOP unshared decoder**.
- **Estimated cost** at P=4 and d_local=192: r=1/2/4 is 1.04×/1.19×/1.48× E1, because dropping the 8k head (21% of E1) pays for the byte encoder. "Refine only the active region" saves compute at inference, not training. Dense SWA at T=4,096 needs the flex-attention patch.
- **E2's lessons apply:** gated or contractive updates, per-pass loss, pass embedding, raw-byte re-injection, and the full L(r) curve.
- **A cheaper first test: SQ207, a speller on frozen E1** (BPE tokens as patches), which isolates the loop question in minutes.
- Side quests **SQ206–SQ209** (causality test, speller, BLT-lite with the winning decoder, exact-span eval).

### Speculative decoding: tokens draft, bytes verify (MambaByte, research only)
`notes/research-speculative-bytes.md`:
- **MambaByte's 2.6× is lossy:** top-3 acceptance, measured against plain MambaByte-972M with a 110M subword drafter (Table 6). The "2.6×" in `notes/research-byte-level.md` is corrected.
- **At our 13M scale** the drafter/verifier cost ratio is about 1, so the estimates are 1.0–1.4× for tokens→bytes and 1.2–1.5× for r=1→r=4 loops.
- **GDN rollback** = state snapshot + replay. ReplaySSM (2610.13778) does this as a pointer move.
- Side quests **SQ196–SQ199**. The first is the acceptance oracle on E2 r=1 vs r=4, about 2–3 GPU-min.

### Watchdog
INC-20261122-002 (eval-battery diagnostic mqarH exited 143) was triaged as an operator stop by that subagent and closed, the same as INC-001.

---

## v0.4.59: Outside review 19:37: E0 may be a handicapped yardstick (init differs from Llama)
*2026-11-22*

### Outside review 2026-11-22 19:37 (Haiku, AGENTS §0d)
`notes/reviews/2026-11-22-1937-review.md`, with the main agent's triage.
**Key finding:** E0 may be a handicapped yardstick.
- E10 (HF Llama) has E0's exact params and FLOPs, the same step-0 loss and the same LR (3e-3), yet gets **0.5822 / 0.5854 vs E0's 0.6063** (seeds 0/1; contaminated val).
- **Verified in code:** E0 uses GPT-2-style scaled residual-output init (std 0.02/√(2·depth), `gogeta/model.py:246-248`); HF Llama uses a flat 0.02. Init is a plausible cause, not yet the established one.
- The E0-parity criterion is evaluated in Q12 on E10's 3-seed mean and clean val; it isn't due yet.
- Phase 2b's confirmatory comparison (H1) uses E10, not E0, so it's protected.
- **SQ99's baseline is E0;** it'll be re-run with the E10 checkpoint.

**Accepted** (new review range SQ200+):
- **SQ200 (top pick): an init-matched 2×2 swap of E0/E10 inits** (~15–20 GPU-min);
- SQ201: E1-vs-E10 horizon curve at a matched LR. E1 leads by 2.1% at 20M, and E10 leads by 0.4% at 200M (seed 0, contaminated);
- SQ202: mixer-conversion test;
- SQ203: per-token gain decomposition;
- SQ204: an n-gram copy floor on the near-dup band (CPU).

**Ordering adopted:** SQ200 and SQ204 run right after Phase 2b, before the Q12 verdicts are written up. **Rejected:** "Q9 wrongly marked done". Its condition (LR_E10 ≠ 3e-3) is false, so skipping was correct.
### Quick review 19:09 (reproducibility auditor, Codex)
**Provenance audit done:** K9's and SQ70's lint FAILs are lookup gaps (predictions were committed in their scripts before the results). **K7-parity is a genuine pre-registration lapse:** its script and results were first committed together in v0.4.12, so it's treated as exploratory. The fix is SQ205; the lint report gets regenerated with each science version.

---

## v0.4.58: Claims ledger v2: immutable definitions + append-only events
*2026-11-22*

### Claims ledger redesign (the human's spec)
**What** (a subagent built it; the main agent verified it): the ledger separates each claim's **immutable definition** from its **evolving evidence**.
- **Layout:** one file per claim, `experiments/claims/ledger/C<n>.jsonl`.
  - Line 1 is the `created` event holding the definition: ID, proposition (verbatim), scope (dataset, architecture, model size, precision, hardware), type (empirical / theoretical / implementation / external finding / hypothesis), refutation criteria, dependencies, origin.
  - Every later line is an append-only event: evidence_added, status_change, limitation_added, audit_verdict, erratum, superseded. Each has a version, actor, evidence pointers and `verified_at` {commit, date}.
  - Status, evidence, limitations and "last verified" are **derived**.
- **The immutability rule:** the committed file must be an exact prefix of the working file.
- **Writes go only through the API:** `python experiments/claims_list.py add-event` / `new-claim`, with per-claim locks.
- **Views:** `experiments/claims/views/C<n>.md` and `experiments/claims-status.md`. The schema and rationale are in `experiments/claims/README.md`.
- **Lint (`--check`, the release gate):** required fields, the type enum, evidence pointers that exist on disk, last-verified, dependency cycles, dependents of refuted claims, the definition-immutability and append-only checks, and the seed rule for VERIFIED.
- `experiments/claims.md` is **frozen** as the v1 archive (only line 1 changed, to a frozen header; its sha256 is in `MIGRATION.json`).
- `claim_review_batches.py` also counts `audit_verdict` events toward the 6-hour cool-down.

**Migration** (`experiments/claims/migrate_v1.py`):
- 117 IDs and 167 v1 rows → 167 events, matched 1:1 by `--verify`. 198 events in total.
- Latest statuses are identical to v1: 23 VERIFIED, 14 PARTIAL, 6 REFUTED, 2 WRONG FOR OUR CODE, 72 UNVERIFIED.
- Types: 56 external finding, 26 implementation, 18 hypothesis, 14 empirical, 3 theoretical (50 inferred, flagged).
- 49 limitations were added, quoted verbatim, with the source line checked.
- **Gaps are left visible, not invented:** "TBD (not stated at creation)" for scope (99–115 per field), refutation criteria (100) and dependencies (114).

**Verified by the main agent**:
- `claims_list.py --check` exits 0 (lint 0 fail, 0 warn);
- `migrate_v1.py --verify`: 117 IDs, 167 rows, 167 events;
- ledger tests pass (reported 15/15);
- `claim_review_batches.py status` works unchanged.

**AGENTS.md**: §0d-ter, §5 and §7 updated by the subagent; §0's "source of truth" line by the main agent. The audit brief now points auditors at the ledger files and views.
**Known issues** (recorded, not hidden):
- C107 holds two different propositions (it needs a new ID for the K9 one);
- C5's refutation criteria name E10 while its proposition names E0;
- `verified_at` for migrated rows is approximate;
- the cockpit, `test_guard.py` and `repro/lint_science.py` still read the frozen v1 file (follow-ups).

**Leak note**: the release helper had staged in-progress versions of `claims_list.py` (v0.4.50/51) and a v2 view (v0.4.53) before the store existed. This version completes that state.

---

## v0.4.57: Byte-level refs (32 papers, 2025-26 survey); quick review 18:57 triaged
*2026-11-22*

### Byte-level literature, 2025–2026 (the human's request)
**What**: refs section **`14-byte-level`** in `refs/fetch_refs.py` holds **32 papers, each verified against the arXiv API** (title and date match): 8 foundational (ByT5, CANINE, Charformer, MegaByte, MambaByte, SpaceByte, BLT, bGPT) and 24 from 2025–26. 32 PDFs were downloaded (33.2 MB, git-ignored); `refs/README.md` was regenerated. The survey and its impact on SQ180–SQ189 are in `notes/research-byte-level-2025-26.md`. Numbers are from abstracts and HTML summaries, labelled "reported".
**Design changes folded into the side quests**:
- per-position loss within each patch (patch lag);
- a patch sweep up to 16;
- fixed-shape top-K dynamic boundaries (ByteFlow Net);
- **bytes are expected to lose at our scale** (the tokenizer leads below ~4B on code), with a cheapest rescue arm.

Also: bpb isn't tokenizer-agnostic for raw training loss, and entropy patching has a blind spot at our scale. No paper combines ternary weights with bytes.
### Quick review 18:57 (ML theorist, Codex GPT-6 Luna)
**SQ195**, the loop-map Jacobian gain: power iterations on E2 activations at r = 4/8. Norm growth (K6) alone doesn't show local instability. It informs the loop fixes but doesn't hard-gate them.

---

## v0.4.56: Run report forms (final): every finished run has a form, filled by an agent through the cockpit MCP
*2026-11-22*

### Run report forms (final hand-back; snapshot was v0.4.45)
**What** (the human: "I want to see the agent use the cockpit and fill out a form for each run... explains the experiment and its metadata and shows any useful graph"):
- `cockpit/runform.py` gives the cockpit draft, submit, render and list for forms.
- MCP tools: `gogeta_run_form_draft`, `gogeta_run_form_submit` (state-changing; refuses empty narratives and claims cited without evidence pointers) and `gogeta_run_forms`. The MCP server now has 31 tools.
- The TUI's Runs tab has a forms view, and `brief()` lists finished runs that have no form.
- **Each form** has machine-filled metadata (experiment and prediction, architecture, params and FLOPs, hyperparameters, commit, environment, metrics, the K9 data caveat, linked claims) and agent-written narrative fields. **Charts** are static SVGs, transparent so they work in dark mode, at most 9 KB each: loss vs tokens with val points, LR, grad norm, throughput, a sibling comparison (val vs LR for K7), and L(r) for E2.
- **README:** a "Run report forms" section explains how to view and fill forms.

**Results**:
- **15 of 15 finished runs have a form,** filled **by a subagent calling the cockpit's MCP server** (82 MCP tool calls logged in the cockpit's SQLite, 0 failed). One refusal demo (empty purpose, claim evidence "see above") was refused and wrote nothing.
- Cockpit tests: 35 + 22 unit tests OK, and the headless TUI check is FAILED: 0 twice. A flaky TUI wait was hardened, and tab keys stopping after a button click was fixed.

**Flagged by the form-filling agent** (from the run data; **not a verdict**: contaminated val, and the registered parity and C5 analyses wait for the 3-seed means on clean val):
- **E10 seed 0 (0.5822) is about 4% better than our E0 (0.6063)** at identical shapes, which is outside the registered E0-parity tolerance;
- **E10 at 3e-3 also edges out E1 at 3e-3 (0.5846)** by 0.4%.

Both bear directly on C5.
**Discrepancies noted on the forms**: max grad-norm after step 300 from Trackio is 0.485/0.490, vs 0.38 in the micro devlogs; E1 reaches E0's final val at 172M tokens (form) vs 169M (devlog interpolation).

---

## v0.4.55: A1 data integrity (final): clean-val lists, frozen test split, families, strict benchmark contamination
*2026-11-22*

### A1 data integrity (rigor workstream, final hand-back; snapshot was v0.4.46)
**What**: `experiments/data_integrity/`:
- MinHash/LSH + substring dedup;
- corpus family clustering;
- a frozen test-split builder with an access guard;
- strict benchmark contamination;
- dataset, clean-val and test-split manifests.

The method and policy are in `notes/data-integrity.md`. Test data lives only on the WSL disk.
**Results** (reported by the subagent; the main agent re-checked the manifests):
- **The dedup tool works:** planted-duplicate precision 1.00; recall 1.00 for exact and reformatted copies, 0.95 for a 60% prefix, 0.85–1.00 for renamed identifiers.
- **No unused Python docs exist** (all 991,154 are in train or val), and the corpus has **no source/repo id** (columns: type, lang, text). A first-function-name proxy: 93.5% of val docs share a name with a train doc.
- **Families** (MinHash, Jaccard ≥ 0.5): 27.1% of docs and 19.6% of tokens are in families of size ≥ 2; 741,056 families, the largest with 5,095 docs; 555 families span train and val. Redundancy is mostly templates, not copies.
- **Clean val** (`clean-val.manifest.json`, built from K9): `val_clean80` = 2,333 docs (the primary set for claims) and `val_clean50` = 1,174 docs (the strict sensitivity set).
- **Strict benchmark contamination** (vs the old C103 screen): **HumanEval 74/164** (C103 said 32), HumanEval+ 78/164, **MBPP 103/500** (C103 said 43), MBPP+ 65/378, CRUXEval 9/800, fresh GitHub 1/500. **All earlier code-benchmark numbers should be treated as contaminated.**
- **Frozen test split `code-py-v1-test`:** 462 UncheatableEval 2026-10 files, 1,000,269 tokens, max 13-gram overlap with train/val 0.060, no family link; `--verify` passes. `code-py-v1-test-indist` (3,062 train docs) is valid only for a retrain that excludes them.
- **A deduplicated v2** (one doc per family) would keep 737.6k docs / 214.7M tokens (74.7% of docs, 81.8% of tokens).
- The main agent's re-run of `build_dataset_manifest.py --check` gives `all_ok: true`.

**Predictions that missed** (recorded): family share (predicted > 40%, observed 27%), test-split size, HumanEval strict count (predicted ≤ 70, observed 74), and MBPP vs MBPP+ agreement (79% vs a predicted 95%).
**Next**: re-score every checkpoint on `val_clean80` / `val_clean50` (SQ100); decide on `code-py-v2` before the next big replication (SQ102); swap in the strict contaminated-ids list after the campaign (SQ101). The ledger rows wait for the ledger redesign (C140–C149 proposals in the folder).

---

## v0.4.54: SQ99 registered (out-of-corpus check); byte-level side quests adopted
*2026-11-22*

### SQ99 (registered, not run yet): out-of-corpus check
**Why**: K9 showed val is 41.7% near-duplicated, and every planned test scores in-corpus docs. Does E1's advantage over E0 hold on Python the corpus cannot contain? (outside review 19:07, idea I4.)
**What**: `experiments/checks/sq_99_out_of_corpus.py` (from a subagent; selftest 27/27).
- **Out-of-corpus set:** 330 local Python files (stdlib plus 30 installed packages; paths and sha256 only, no contents committed) with < 20% 13-gram overlap with train (mean 0.31%). 400k tokens.
- **In-corpus control:** a token-matched sample of 1,340 K9-clean val docs.
- Both models are scored through the **same CPU fp32 path** (E1's GDN via the plain-PyTorch reference used by the 3D trace), so the comparison is same-scorer.
- Estimated 13–37 CPU-min.

**Prediction (registered here)**: out-of-corpus bits/char is > 1.5× in-corpus for both models; E1 still beats E0 out of corpus, but by less than in corpus; the ordering holds per source.
**Criterion**: "transfers" if gap = bpc(E1)/bpc(E0) − 1 < 0 with a paired BCa 95% CI (B = 10,000) excluding 0; otherwise "does not transfer".
**Caveats** (stored with the result): exploratory (single-seed v0.3.0 checkpoints, untuned LR); out-of-corpus is confounded with format and doc length; fp32 CPU, not the bf16 fla path.

### Byte-level tokenization side quests (the human's request)
SQ180–SQ189, from the byte-level research (v0.4.50), are adopted into `experiments/side-quests.md`.
- **Run order:** SQ189 (memory/throughput probe), SQ186 (causality test), SQ180 (byte-E1 vs BPE-E1 at matched FLOPs and matched bytes) and SQ181 (BLT-lite, patch 4).
- **Blocker:** dense-masked SWA (C12) needs the flex-attention patch first.
- A subagent is searching for 2025–2026 byte-level papers to add as `refs` section `14-byte-level`.

---

## v0.4.53: Commit-agent exception in AGENTS §4; quick reviews 18:44/18:50 triaged
*2026-11-22*

**What**:
- **AGENTS §4 exception** (the human: "start committing, too many uncommitted changes... let the agent do it"): one designated commit subagent may turn piled-up work into versions using the release rules. Snapshots of still-running work are labelled "in progress", with secrets and size checks, protected paths untouched, and no force-push or tag moves. Its first use was v0.4.44–v0.4.52.
- **Main-agent check of v0.4.44–v0.4.52**:
  - all 9 tags are on origin; protected paths are clean;
  - `cockpit.sh --test` in WSL: 57 unit tests OK;
  - the TUI/probe phase printed "FAILED: 2" once and nothing on an immediate re-run (flaky). The run-forms subagent is fixing it, and making the 2 tests that fail on Windows-only Python skip off-WSL (the live DB exists only in WSL).
- **Quick reviews (Codex GPT-6 Luna):**
  - 18:44, systems engineer: **SQ193**, `torch.compile` recompiles per sampled loop count r. No recompile logging exists today.
  - 18:50, competitor lab: **SQ194**, formatting robustness of E0/E1 on reformatted clean-val docs (CPU, reuses SQ99's scorer).

---

## v0.4.52: Tokenizer research: final hand-back
*2026-11-22*

The tokenizer research subagent has finished. Its files were committed earlier as the in-progress snapshot v0.4.49 (tags are never moved); this entry records the final status and changes no files. No tokenizer model or vocab files are committed (results JSON only; nothing over 2 MB).

- Reported by the subagent (CPU-measured, not yet re-verified by the main agent): the shipped 8k BPE gets 3.733 chars/token (3.811 on clean val), better than SmolLM2's 49k tokenizer (3.461).
- 9% of tokens are whitespace (newline + indent); 19% of the vocab is duplicated `word` / ` word` pairs.
- Vocab size is a weak lever (4k: -3.1%, 16k: +1.2%).
- Line-level superwords give +67%, but that is likely inflated by corpus duplication.
- Recommendation: keep the 8k BPE for now.

---

## v0.4.51: A6 numerics: final hand-back
*2026-11-22*

The A6 numerics subagent has finished. Its files were committed earlier as the in-progress snapshot v0.4.48 (tags are never moved); this entry records the final status and changes no files.

- Verified by the main agent: 49/49 CPU checks pass (`run_numerical_suite.py`: SUITE PASS; api_contract 10, reference_blocks 24, error_budget 5, mutation_controls 10). GPU part (SQ150) written, not yet run. The commit agent did not re-run it (no torch on the Windows side).
- Reported by the subagent, not yet verified: the positional-call scan finds 1 unsafe site (the deliberate C2 reproduction, allow-listed) and 29 fragile-but-correct positional calls (SQ154).
- Reported by the subagent: the error budget replaces the ad-hoc 2e-3 / 3e-2 tolerances.
- Reported by the subagent: the fp32 chunk kernel shows 3.1x more error on the GPU than the emulation predicts, unexplained (SQ153).
- Reported by the subagent: `nn.RMSNorm`'s default eps makes GDN `out_norm` not scale-free for about 0.6% of tokens at init (proposed C199).

---

## v0.4.50: Byte-level research: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `experiments/byte_level/` (flop_arith.py and its output, seq_stats.py and seq_stats.json, proposed claims and side-quests) and `notes/research-byte-level.md`; plus the quick review `notes/reviews/quick/2026-11-22-1844-systems-hardware-engineer.md`.
- Checks: `python -m py_compile` on the .py files passed. The scripts were not executed. No secrets and no files over 2 MB.
- The FLOP arithmetic, sequence statistics, notes and the review are reported by the subagent, not yet verified by the main agent. Nothing was run on the GPU.

---

## v0.4.49: Tokenizer research: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `experiments/tokenizer/` (audit.py, train_variants.py, run.sh, probe_env.sh, results/audit.json, results/variants.json) and `notes/research-tokenizer.md`.
- Checks: `python -m py_compile` on the .py files passed. The scripts were not executed. No secrets and no files over 2 MB.
- The audit and variants results and the notes are reported by the subagent, not yet verified by the main agent. Nothing was run on the GPU.

---

## v0.4.48: A6 numerics: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `experiments/staged-tests/` (numref, tolerances, error_budget, env_fingerprint, gpu_numerical_suite, run_numerical_suite, test_* files, proposed claims / predictions / queue / side-quests, JSON outputs) and `notes/numerical-verification.md`.
- Checks: `python -m py_compile` on every .py file passed. The test_* files were not executed because they import torch, which is not installed on the Windows side. The GPU suite was not run. No secrets and no files over 2 MB.
- The error_budget and env_fingerprint JSON and the notes are reported by the subagent, not yet verified by the main agent.

---

## v0.4.47: A4 eval battery: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `experiments/evalbattery/` (mqar.py, itercomp.py, summarize.py, diag*.sh, results/diag, results/itercomp, state_track results), `evals/battery/`, `evals/tasks/battery/`, `evals/results/battery-smoke-E0-limit150-tier1-20261122-183529.json`, and `notes/eval-battery.md`.
- Checks: `python -m py_compile` on every .py file passed. No tests were executed (the battery needs torch, which is not installed on the Windows side). No secrets and no files over 2 MB.
- The diag and itercomp results and the E0 smoke result are reported by the subagent, not yet verified by the main agent. Nothing was run on the GPU.

---

## v0.4.46: A1 data integrity: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `experiments/data_integrity/` (dedup, corpus_families, group_proxy, benchmark_contamination, build_dataset_manifest, build_test_split, test_guard, dataset and test-split manifests, proposed-queue.json, results/*.json). `notes/data-integrity.md` does not exist yet and is not included.
- Checks: `python -m py_compile` on every .py file passed. `python experiments/data_integrity/test_guard.py` runs on CPU and printed its guard description with no error; it is not an assertion suite. No secrets and no files over 2 MB.
- Any figures in the manifests and results JSON are reported by the subagent, not yet verified by the main agent. Nothing was run on the GPU.

---

## v0.4.45: Run-forms cockpit: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: cockpit changes (`core.py`, `mcp_server.py`, `tui.py`, `tui_check.py`, `test_core.py`, README, BACKLOG), new `cockpit/runform.py` and `cockpit/mcp_forms.py`, the generated run forms and charts under `experiments/run-forms/`, and the README "Run report forms" section.
- Checks: `python -m py_compile cockpit/*.py` passed. `python cockpit/test_core.py` on Windows: 35 tests, 2 failures, 1 skipped. Failing: `LiveRepoTests.test_search_index_runs` (0 indexed runs, the live DB is absent here) and `LogDbTests.test_writers_never_raise_when_db_unavailable` (got 2, expected None). Not investigated; the cockpit normally runs under WSL.
- The run forms are reported by the subagent, not yet verified by the main agent. Nothing was run on the GPU.

---

## v0.4.44: Activation-function research: snapshot (in progress)
*2026-11-22*

**Work-in-progress snapshot** committed on the human's instruction to reduce uncommitted changes; the subagent's final hand-back will be its own version.

- Included: `notes/research-activations.md`, `notes/research-activations/` (act_stats.py, proposed claims / queue / side-quests, results/act_stats.json), `notes/patches/activation-mlp-act.patch`, and the new activation section in `refs/README.md` plus `refs/fetch_refs.py` changes.
- Checks: `python -m py_compile` on act_stats.py and fetch_refs.py passed. Nothing was run on the GPU.
- Any numbers in the notes and act_stats.json are reported by the subagent, not yet verified by the main agent.
- The patch is not applied to `gogeta/`; the proposed claims and queue are proposals only, not ledger entries.

---

## v0.4.43: Outside review 19:07 + quick reviews triaged (out-of-corpus check is top pick)
*2026-11-22*

### Outside review 2026-11-22 19:07 (Haiku, AGENTS §0d)
`notes/reviews/2026-11-22-1907-review.md`, with the main agent's triage footer.
**Critiques** that hold:
- the clean-val cut is one threshold;
- dedup changes the training regime (epochs) as well as the data;
- cost is compared in FLOPs, not GPU-minutes;
- **the ternary factor bundles ternary weights with int8 activations** (model.py:73-78);
- the loop is the least replicated ingredient;
- n = 3 can't verify;
- r = 4 was chosen after looking;
- **Rotary is capped at max_seq_len** (model.py:52/98/227, verified), so SQ52's 2048-length eval can't run; it's marked not ready;
- nothing tests transfer beyond the corpus.

**Accepted** (6 new side quests, none rejected):
- **SQ99 (top pick): an out-of-corpus check** on local Python files with < 20% 13-gram overlap with train. It's CPU-feasible: E1 via the plain-PyTorch GDN reference;
- SQ96: equal-GPU-time efficiency;
- SQ97: hybrid gap vs width;
- SQ98: the E8 draft-agreement premise;
- SQ190: repetition as an axis;
- SQ191: no global layer;
- SQ192: split the ternary factor (full-precision weights + int8 activations arm).

The full L(r), r = 1..7, will be pre-registered before the factorial.

### Quick reviews (Codex GPT-6 Luna, AGENTS §0d-bis)
- **18:36 statistician** (committed in v0.4.39).
- **18:41 data engineer:**
  - **SQ95**: scorer parity. K9's +25.8% mixes a GPU-bf16 full-val number with a CPU-fp32 near-dup number; the all-GPU doc-level re-score will add a parity column.
  - Source- and group-aware splits were forwarded to A1 for `code-py-v2`.

---

## v0.4.42: Docs refresh (hourly)
*2026-11-22*

**What**: the hourly Haiku docs refresh (the human's standing request), reviewed by the main agent for accuracy against the repo.
- **README status paragraph**: the K7-tuned LRs (E10 3e-3, E1 6e-3), the Phase 2b start, the decode kernel working (C2/C110), and a stronger K9 caveat. This edit landed inside v0.4.41's commit (the version bump stages README); it's recorded here.
- **`notes/sketch.md`**: decisions-log bullets for K9, the Phase 2b registration, the ledger-status rule and claim audits, the K7 LRs, SQ70/C110 and C105, plus a short status section.
- **`experiments/launch/campaigns/README.md`**: rows for the K7 resume and extension campaigns (done) and Phase 2b (running).
- **`post_training/SETUP.md` §8.2**: the GDN recurrent kernel is resolved (it was a call-site bug). Decoding is still not wired into `generate` (SQ53), and SWA/attention have no cache.

**Checked**: every statement traced to the ledger, DEVLOG or a results file. Nothing reverted.

---

## v0.4.41: Research: image and audio inputs (nice-to-have, research only)
*2026-11-22*

**What** (human: "research subagent about supporting audio and image inputs, ... nice to have for later, keep it research only"): two Sonnet research subagents, no code, no downloads, no GPU.
- **`notes/research-image-input.md`**: 6 approaches compared (frozen encoder + projector, patch-as-tokens, discrete image tokens, cross-attention adapters, early fusion; tiny VLMs; Mamba/linear-time VLMs; ternary vision encoders; looped multimodal). **Recommendation (medium confidence)**: a narrow synthetic proof of concept, a tiny 1-3M-param conv/patch stem feeding 32-64 prefix tokens, trained on self-rendered matplotlib/turtle images paired with their Python source (no licence issues); fallback: Flamingo-style gated cross-attention from the coda. Key constraints: image tokens inside the looped core cost r times (keep them <= 64); a bidirectional 2D encoder feeding the causal core (VL-Mamba/Cobra pattern); ternary vision exists (ViT-1.58b, BitVLA, TernaryCLIP) with the first layer kept high-precision; the smallest published VLM LM is ~135M (10x ours). Several arXiv ids are marked unverified.
- **`notes/research-audio-input.md`**: **recommendation (medium confidence, ~60%)**: discrete semantic audio tokens at 12.5-25 Hz added to the vocab (~2k rows; Mimi codebook-0, GLM-4-Voice tokenizer or HuBERT units), interleaved audio/text training: no LM change, fits GDN's streaming state; T=1024 = ~82 s of audio at 12.5 Hz. Fallback: frozen Moonshine-/Whisper-tiny encoder + projector (as big as our LM). Evidence for GDN on ASR is thin (no LibriSpeech GDN result found); Speech Slytherin finds Mamba's ASR advantage minor; no ternary from-scratch audio model found. Cheapest PoC described (LibriSpeech-100 with codec tokens, shuffled-audio control). Licences flagged (EnCodec weights CC-BY-NC; Mimi licence conflicting between sources).
- **Status**: both are post-v1.0.0 arcs, after the K9 cleanup and the core thesis. Paper lists (~60 ids, some flagged unverified) will be checked before being added to `refs/fetch_refs.py`.

---

## v0.4.40: Rigor design (A3): fair HPO, matched arms, E1 decomposition, 2x2x2 factorial (registered)
*2026-11-22*

### A3 experimental design (rigor workstream; predictions REGISTERED)
**What**: from a subagent, reviewed by the main agent; the main doc is `notes/design-rigor.md`.
- **28 staged configs** in `experiments/staged-configs/`, to move into `configs/` between campaigns. Each was rebuilt on CPU from its JSON, with params and FLOPs printed.
- An opt-in GDN ablation-flag patch (`gdn_conv`, `gdn_gate`, `gdn_out_norm`, defaults identical; E1/E2/E3 construct bit-identically under it).
- Power/cost and factorial-analysis scripts in `experiments/design/`.

**Design**:
- **Fair HPO:** an equal 5-point LR grid per model; E10 gets 1.2e-2 (3 min). **Horizon check at 200M tokens:** E1@3e-3, E10@6e-3 and E10@2e-3 (88 GPU-min). The optimum transfers if no neighbour beats the Phase 2b mean by more than 0.0092 nats. No schedule sweep; one warmup pair (13 min).
- **Matched arms** (4.9 GPU-h):
  - E30 dense at E1's params; E31 dense at E1's FLOPs;
  - E32/E33 unshared at E2's FLOPs; E34 E9 widened to d=576.

  Measured cost ≠ FLOPs: E1 has 0.94× E0's FLOPs but runs at 0.60× its throughput (dense SWA mask, C12).
- **E1 decomposition** (E40–E48): GDN→attention, SWA-only, no conv / gate / out-norm, bare GDN, attention placement, all-GDN. 0.55 GPU-h minimal, 2.4 rigorous.
- **2×2×2 factorial** (GDN × loops × ternary; the cell abc is E(20+4a+2b+c)):
  - matched at 14 block applications (FLOPs matched, params differ; E35/E36 are params-matched partners);
  - estimand I = y111 − y100 − y010 − y001 + 2·y000 on log clean-val loss;
  - synergy if the 95% CI is below 0;
  - N=46 runs detects 1% at sd 0.004;
  - the cheapest powered version is 6.4 GPU-h (14.5 if sd = 0.0067);
  - **registered prediction: I = +0.5% ± 1.5% (no synergy expected)**.
- **Data:** the factorial and decomposition run on a deduplicated `code-py-v2` (being built by A1). A bridge (E10 + E1 seed 0 on v2, 64 GPU-min) checks whether rankings change.
- **Roadmap:** about 15 GPU-h for a defensible minimum, about 30 GPU-h for everything.

**Registered**: A3's predictions are appended to `experiments/checks/PREDICTIONS.md` (dated 18:47), before any of these runs exists.
**Caveats**: the wall-time estimates are model-based; ternary speed (0.75×) is assumed; the v2 configs can't launch until A1 builds `code-py-v2`; the power figures assume a seed sd of 0.004 until Phase 2b measures it.

---

## v0.4.39: Reproducibility (A5): manifests, reproduce, CPU replay, science lint; ledger-status rule
*2026-11-22*

### A5 reproducibility (rigor workstream)
**What**: from a subagent, reviewed by the main agent.
- **`repro/manifest.py`** builds an immutable per-run manifest after the fact: code SHA plus per-file hashes, config sha256, dataset sha256s, checkpoint/summary/log hashes, precision, deterministic flag, launch argv. It has a self-hash and refuses to overwrite. Anything not recorded at run time (torch/CUDA/driver/pip-freeze at train time) is written literally as "unknown (not recorded at run time)", never guessed.
- **12 back-filled manifests** in `repro/manifests/` (E0, E1 ×6, E2, E10 ×4).
- **`repro/reproduce.py`** is read-only: it checks the tree, data and config hashes against a manifest, prints the exact launch command, and `--compare` gives BITWISE / STATISTICAL / NOT REPRODUCED against the tolerances in `notes/reproducibility.md`.
- **`repro/cpu_replay.py`** is the cheapest blind-replay tier: rebuilding E0 from its manifest on CPU gives a step-0 loss of **9.0899 vs the logged 9.0900**. That confirms seed, init and data order. It's attention-only configs only.
- **`repro/lint_science.py`** is a registry-consistency linter.
- **`notes/reproducibility.md`**: the nondeterminism inventory, bitwise vs statistical reproduction, provisional tolerances (|Δ val_loss| ≤ max(0.004, 2 × seed sd)), and a blind-replication checklist.
- **`notes/patches/repro-manifest-at-run-end.patch`** (unapplied): `meta.json` records env info, a `--deterministic` flag, and the manifest is written at run end.

**Results**:
- Main-agent re-run: `repro.test_repro` **13/13 OK**; no secrets in the files.
- `reproduce.py` on E0: data and config MATCH; code MISMATCH because HEAD has moved on (expected).
- **Lint at v0.4.36, 12 FAILs:**
  - E0/E1/E2 lack a DEVLOG `### <ID>` block;
  - K7-parity's results were committed before its PREDICTIONS section;
  - K9 and SQ70 have their predictions in their scripts, not in PREDICTIONS.md;
  - 6 done queue items cite no version.

  These are queued as fixes; the K7-parity ordering one is recorded as a real pre-registration lapse.

### Ledger-status rule (registered before any E1/E11 Phase 2b result)
The first Codex quick review (`notes/reviews/quick/2026-11-22-1836-statistician.md`, GPT-6 Luna, 49 s) pointed out that the registered C5 rule can pass while the 95% CI includes 0. A dated clarification was appended to `experiments/checks/PREDICTIONS.md`: **the ledger status is the stricter of the registered verdict and the statistics policy** (Holm-adjusted CI excluding 0, or TOST); otherwise it's PARTIAL with the reason. Leave-one-seed-out sensitivity is mandatory (new SQ94). The doc-bootstrap idea was merged into SQ112.

---

## v0.4.38: Hourly claim audit: 8 Codex auditors, 6-hour cool-down per claim
*2026-11-22*

**What** (the human: "every hour spawn 8 gpt 6 luna subagents to review the claims and make sure they are rock solid, the cool down on each claim after review is 6 hours ... may need multiple passes to catch issues"):
- **AGENTS §0d-ter.** Every hour, 8 read-only Codex GPT-6 Luna (high) auditors run in parallel.
  - Each audits a batch of up to 6 claims: it reads all ledger rows, **opens every evidence pointer and checks the numbers match**, and judges the status against the statistics policy, K9 and later results.
  - Verdicts: SOLID / WEAK / WRONG / NEEDS-EVIDENCE / UNCHECKABLE, each with a pointer and a fix.
  - The main agent verifies every non-SOLID verdict itself before appending a ledger row. Rejected findings get a reason in the audit file.
  - One PATCH version per hourly audit, pushed.
- **`experiments/claim_review_batches.py`** (stdlib): `plan` picks due claims (settled first, never-reviewed first, then least recently reviewed) and skips any reviewed in the last **6 h**; `record` logs verdicts in `experiments/claim-reviews.json`; `status` reports due and cooling counts. **Now:** 117 claims due, 0 cooling, 8 batches of 6.
- **`notes/reviews/codex_run.ps1`** is the shared read-only Codex runner (newest desktop-app binary, `gpt-6-luna`, high reasoning, `-s read-only`, `-o` output). **`notes/reviews/claim-audit-brief.md`** is the auditor brief.
- **Timers:** the claim audit runs hourly at :47 (first run about 19:03). The 5-minute quick-review timer now uses the in-repo runner. Its first tick at 18:32 failed on a PowerShell quoting bug (`$ts:` parsed as a scoped variable), fixed by passing the brief as a file.
- **Cost guard:** audits never fall back to 8 Claude agents; failed batches stay due for the next hour.

---

## v0.4.37: Phase 2b analysis plan registered (statistics layer, clean-val primary metric)
*2026-11-22*

### A2 statistics: Phase 2b analysis addendum (REGISTERED)
**What**: rigor workstream A2 (statistics), from a subagent. `experiments/stats/core.py` covers paired/Welch CIs, effect sizes, BCa and paired bootstrap, sign-flip permutation, Holm/BH, TOST, variance components and MDE/power, with selftests including CI-coverage simulations. Also:
- `replication_stats.py` (the secondary statistical layer over the Q6 discovery);
- `score_doclevel.py` (GPU: doc-level bits/char per checkpoint on the K9 clean and near-dup subsets);
- `pairing.md`;
- the addendum, **appended to `experiments/checks/PREDICTIONS.md`**.

**The registered content**:
- The original Phase 2b criteria stay the primary decision rule. They're now applied to the **primary metric, doc-level bits/char on clean val** (val docs with < 80% 13-gram overlap with train: 2,333 docs; sensitivity at < 50%). Training-time `val_loss` is secondary (K9).
- The **confirmatory family is fixed at m=2, Holm at 0.05**: H1 E1 vs E10 (paired by seed; seed s fixes the data order, which is shared across architectures) and H2 E11 vs E1 (TOST with margin max(1%, 2 pooled sd)). Everything else is BH q=0.10 or descriptive. E2 vs E9 is n=1, descriptive.
- A **memorisation gap** (near-dup minus clean bits/char) is reported per model. It's a caveat flag and never changes a verdict.
- **Power at n=3:** the MDE is about 2% at sd 0.004, so power for a 1.5% gap is about 0.55. The exact sign-flip test can't go below p=0.25 at n=3.

**Registration timing (honest)**: the addendum file was written at 18:30:17, before the first Phase 2b result (E10 seed 0, finished 18:31:38). It was committed a few minutes after that result existed. The main agent didn't open the result before committing, and an E10-only result can't inform H1 or H2.
**Open decisions (main agent)**:
- `score_doclevel.py` needs about 25 GPU-min after the campaign. It's queued, or the primary-metric verdicts stay INCOMPLETE.
- A2 notes that the registered C5 rule ("gap > 2 pooled sd + separation") is weaker than a 5% test. It's left unchanged, and the CI is reported beside it.
- The retro-audit downgrades (C8, C18, C43) are pending the main agent's review, in the next version.

---

## v0.4.36: Micro devlogs for every experiment; every settled claim cites evidence
*2026-11-22*

**What** (the human's rules, AGENTS §5, v0.4.31): every experiment has a micro devlog, and every settled claim points to its evidence.
- **31 micro devlogs** in `experiments/devlogs/`: E0–E11, K1–K8 (K7 split into stages), K9 (main agent), SQ2, SQ6, SQ50, SQ70 and EVAL-SUITE, plus `README.md` and `_TEMPLATE.md`.
  - They're reconstructed by a subagent from recorded sources only (DEVLOG, lab notebook, git log, queue history, run `meta`/`summary`, campaign logs). Each entry cites its source, and gaps say "not recorded".
  - E0/E1/E2/E10/K7/K8/EVAL-SUITE carry the K9 caveat.
  - The main agent spot-checked E1's log against the recorded numbers.
- **Evidence-pointer audit:** 44 settled claims; 16 already had valid pointers.
  - **27 got append-only pointer rows,** and every cited path was checked to exist (C5's row keeps its "pending clean-val re-score" qualifier from K9).
  - **C28 is downgraded PARTIAL → UNVERIFIED** (no findable evidence: SQ4 never ran).
  - C106 got a pointer to the main agent's v0.4.19 re-run.
- **Lint:** `experiments/claims_list.py` now flags any settled claim whose latest row lacks a pointer (a path, a run id or a version tag; it matches the full untruncated cell). It **exits non-zero**, so the release script refuses to cut a version until the list is fixed.

**Results**: lint is at 0 unpointed settled claims. The list has 117 claims: 23 VERIFIED, 14 PARTIAL, 6 REFUTED, 2 WRONG FOR OUR CODE, 72 UNVERIFIED.

---

## v0.4.35: Quick reviews run on Codex GPT-6 Luna (Windows desktop CLI)
*2026-11-22*

**What** (the human: "chatgpt desktop app/codex is already installed... not in wsl, in windows"):
- Found the desktop app's CLI under `%LOCALAPPDATA%/OpenAI/Codex/bin/<hash>/codex.exe` (codex-cli 0.167.0-alpha.3.2), already logged in. The auth file wasn't opened. The model id is `gpt-6-luna`.
- **Smoke test:** `codex exec -m gpt-6-luna -c model_reasoning_effort=high -s read-only -C <repo> --ephemeral -o <file>` read `experiments/claims-status.md` and answered correctly, in 16 s and 32,032 tokens.
- **AGENTS §0d-bis** now names the exact Windows command. The 5-minute quick-review cron job was replaced so that it calls Codex first and falls back to Haiku only on error. The review file is written via `-o`, so Codex itself stays read-only.
- **Cost note:** about 32k tokens per review, roughly 380k tokens an hour, from the human's OpenAI plan.

---

## v0.4.34: Quick perspective reviews every 5 minutes
*2026-11-22*

**What** (the human: "every 5 minutes spawn a random codex/chatgpt gpt 6 luna high subagent to review the project from different perspectives and give feedback to the main agent... inject new ideas possibly but for cheap"):
- **AGENTS §0d-bis.** Every 5 minutes a cheap reviewer from a **different model family** critiques the project from one random perspective out of 11: skeptic, statistician, data engineer, ML theorist, systems engineer, reproducibility auditor, competitor lab, code practitioner, newcomer, safety reviewer, cheapest-experiment hunter.
- It reads at most ~8 files and writes at most 40 lines to `notes/reviews/quick/`: observations with citations, 1–3 ideas with their cheapest test, and one question.
- The main agent triages every idea. Quick reviews are committed in batches with the 30-minute review (§0d), not one commit per tick.
- **Reviewer model:** the Codex CLI (GPT-6 Luna, high reasoning, read-only) when installed and logged in. **It isn't installed on this machine** (checked on Windows and in WSL), and the main agent can only spawn Claude models, so the reviewer **falls back to Claude Haiku** and says so in each file. The install and OpenAI sign-in are parked for the human (they use the human's own account).
- **Timer:** a session cron job at minutes 2-59/5 (first tick 18:29).

---

## v0.4.33: K9 FAILS: 41.7% of val is near-duplicated train text
*2026-11-22*

### K9: train/val near-duplicate leakage (FAIL)
**What**: `experiments/checks/k9_leakage.py`, registered in v0.4.30 and run as CPU job `k9-20261122-181814` (200 s). It hashes every 13-gram of the 4,000 val docs (the same docs as K8) against 262.4M train tokens, with a pseudo-val of 4,000 held-out train docs and a shuffled control as baselines.
**Results** (`experiments/checks/results/K9.json`, card `experiments/cards/K9.md`, micro devlog `experiments/devlogs/K9.md`):
- **1,667 of 4,000 val docs (41.7%) are near-duplicates of train** (≥ 80% of their 13-grams in train); 70.7% are ≥ 50%. The gate was < 1%: **FAIL**.
- E0 scores **0.101 bits/char on the near-duplicates**. Without them, its full-val figure goes **0.2209 → 0.2779 (+25.8%)**. The gate was < 2%: **FAIL**.
- **The pseudo-val overlaps exactly as much** (per-doc mean 0.663 vs 0.665; ≥ 50%: 69.5% vs 70.7%), the shuffled control is 0%, and excluding boilerplate barely changes anything. So the corpus itself is self-duplicated: many LeetCode/MBPP-style problems recur in near-identical variants. It's not a split bug.
- Prediction: 1 hit (val ≈ pseudo-val), 2 misses.

**What it means**: every training-time val number so far (E0, E1, E2, E3, E9, E10, K7) is inflated by near-memorisable docs, and comparisons may be confounded by how well each architecture memorises repeats. K8's "beats SmolLM2 by more than 2×" is explained in part.
**Actions**:
- Re-score every checkpoint doc-level on a **clean val** (near-dups removed; A1 builds the list and manifest).
- The Phase 2b primary metric becomes clean-val bits/char plus a memorisation gap. A2 is registering this deviation, dated, before any Phase 2b result.
- A deduplicated `code-py-v2` is being sized (A1), and the factorial will use deduplicated data (A3).
- Phase 2b keeps running; its checkpoints get re-scored.

**Ledger**: **C94 → REFUTED**; C107 → VERIFIED (val is exact-dedup only); **C131 new, VERIFIED** (the corpus is self-duplicated); C5 and C92 stay PARTIAL, pending the clean-val re-score. The README status line now carries the caveat.

---

## v0.4.32: Outside review 18:07 triaged; README claims match the ledger
*2026-11-22*

### Outside review 2026-11-22 18:07 (Haiku reviewer, AGENTS §0d)
**What**: the first 30-minute outside review, `notes/reviews/2026-11-22-1807-review.md`, with a state-of-project summary, 9 critiques, 6 alternative ideas and a next-experiment pick. The main agent triaged every item; the triage is in the review's footer.
**Accepted**:
- SQ90: code state-tracking probe on real Python (in rigor workstream A4);
- **SQ91: E1 decomposition**, one factor at a time, the reviewer's pick (A3, high priority);
- SQ92: oracle per-token loop count, an upper bound for E5 (queued after SQ14);
- SQ93: dense arm at E1's non-embedding params (A3);
- **SQ19 promoted into the queue** (full vs truncated BPTT).

**Rejected**: "dense at E1's FLOPs". E0 already uses more FLOPs than E1 and still loses, so it can't close the gap. It was replaced by the matched-*params* arm. "Unshared at matched FLOPs" was merged into SQ32 and A3.
**README overclaim fixed** (critique 1): the status line now says the 3.6% is one seed at an untuned LR (C5: partial), mentions the running replication, and says absolute losses are provisional until K9. "We know why" is now "our best guess, not settled (C7 partial)". This edit went out inside v0.4.30's commit, because the version bump stages README; it's recorded here.

---

## v0.4.31: Cockpit: TUI + MCP + SQLite logs + self-heal watchdog; evidence and micro-devlog rules
*2026-11-22*

**What** (the human's requests: a TUI + MCP for the project; SQLite run and error logging; self-healing; "be adaptive"; every settled claim points to evidence; every experiment has a micro devlog). The cockpit was built by a subagent and verified by the main agent.
- **`cockpit/`** (new top-level folder):
  - `core.py` (stdlib; one JSON-returning function per question, incl. a ≤ 4 KB `brief()` for agents);
  - `mcp_server.py`: **28 MCP tools**. Read-only ones: brief, status, queue, experiment, claims, search, runs, results, devlog, log tail, errors, incidents, SQL (SELECT-only). State-changing ones are confirm-gated: start/stop server, launch queue item, resolve error, incident update, resume campaign, watchdog;
  - `tui.py` (Textual): 10 tabs, from Dashboard with a GPU card, campaign progress, CPU slots and server tiles with Start/Stop/Open, to Queue, Experiments, Claims, Devlog, Logs, Errors, Incidents, Runs and Search;
  - `logdb.py`: the SQLite store at `~/fusiondance-data/cockpit/gogeta.sqlite` (data, never committed; tokens redacted on every write; incremental ingest of run folders, campaign logs, CPU jobs and tracebacks);
  - `watchdog.py`: the self-heal daemon. It turns 15 kinds of fault (dead/hung runs, NaN/OOM/Triton errors, tracebacks, aborted campaigns, preflight REFUSED, failed CPU jobs, servers down, GPU ≥ 85 °C, low disk) into **incidents** with a context bundle and an agent brief. It does only safe recovery: server restarts with backoff, one retry for transient faults, a thermal pause, and resume-campaign files with exactly the unfinished items, auto-launched only for transient or thermal causes. It never edits code, deletes data or kills a healthy process.
- **The venv** `~/.venvs/gogeta-cockpit` (mcp 1.30.0, textual 8.2.8, pinned in `requirements-cockpit.txt`; ~81 MB) was installed under the human's TUI+MCP request. `.mcp.json` registers the server for Claude Code.
- **Patches, unapplied** (between campaigns): `notes/patches/cockpit-logging-*.patch` (launchers and trainer log straight to SQLite) and `cockpit-resilience-*.patch` (per-item campaign status and unfinished-item files; `--ckpt-every` / `--resume` for mid-run training resume, which is impossible today).
- **AGENTS.md:**
  - **§0e** cockpit first: one interface, SQLite logs;
  - **§0f** self-heal: the detect → contain → triage → fix → test → version → push → resume → verify loop;
  - **§0g** be adaptive: a gap is a cockpit bug, so extend it (pattern: one `core.py` function plus `expose(...)`), with `cockpit/BACKLOG.md` for known gaps;
  - **§5, new rules from the human:** every settled ledger claim must cite a concrete evidence pointer, and every experiment keeps a micro devlog `experiments/devlogs/<ID>.md` (plan, launches, observations, incidents, deviations, results, decisions). A subagent is backfilling the devlogs and auditing evidence pointers.

**Results** (re-run by the main agent):
- `cockpit.sh --test`: **48 tests OK**, plus the TUI's headless check across all tabs with 0 failures;
- MCP stdio probe: initialize OK, 28 tools listed, and 4 live calls returned without errors;
- the subagent's live ingest: 14 runs, 42 events, 4 unresolved errors (the K7 preflight REFUSED lines);
- no tokens or personal data in the committed files.

**Not verified**: the TUI driven by a human; `cockpit.ps1` in Windows Terminal; a long-running watchdog loop (it starts in the next step); the patches (only `git apply --check`).

---

## v0.4.30: K9 registered: train/val leakage check
*2026-11-22*

### K9 (registered, not run yet): train/val near-duplicate leakage
**Why**: K8a failed (our models 0.47–0.49× SmolLM2's bits/char, against a 0.5 gate). Its registered consequence is to rule out train/val leakage before trusting absolute loss numbers.
**What**: `experiments/checks/k9_leakage.py`, written by a subagent and reviewed. It hashes every 13-gram of the 4,000 val docs, the same docs as K8, with a rolling 64-bit hash, and counts hits in train (262.4M tokens, 987,154 docs). Baselines:
- a **pseudo-val** of 4,000 train docs held out of the reference set;
- a **shuffled control**;
- the numbers again **without the 1,000 most common boilerplate 13-grams**.

It flags val docs with ≥ 80% (and ≥ 50%) of their 13-grams in train. If near-duplicates exist, it re-scores E0's bits/char without them (CPU).
**Prediction (registered here, before the run)**: fewer than 1% of val docs are near-duplicates; val overlap is within ±5 points of the pseudo-val baseline; and dropping near-duplicates moves E0's bits/char by < 2%. In other words, K8's 0.47–0.49 ratio is in-distribution training, not leakage.
**Criterion**: PASS if near-duplicate val docs are < 1% AND (if computed) excluding them changes E0's bits/char by < 2%. Otherwise FAIL, which means rebuilding the val split with near-duplicate removal and re-scoring. It maps to C94 and C107.
**Results**: the selftest passes 13/13 (hash vs a pure-python reference, exact planted counts, chunk-boundary invariance, controls). Planned cost: about 2–5 CPU-min, peak ~1 GB of RAM. It runs next as a CPU job, beside Phase 2b.

---

## v0.4.29: SQ70: GDN recurrent decode kernel works (C2 verified)
*2026-11-22*

### SQ70: fla recurrent-kernel re-test (keyword vs positional call)
**What**: `experiments/checks/sq_hw_fla_recurrent.py` (prediction registered in v0.4.27), 18 s on the GPU. It compares a positional and a keyword call against an fp32 reference, plus prefill + 1-token decode, neg-eig decode, and speed.
**Results** (`experiments/checks/results/sq_hw_fla_recurrent.json`), max abs error vs reference:

| | positional (old call) | keyword | chunk | decode | neg-eig decode |
|---|---|---|---|---|---|
| fp32 | **inf** | 2.98e-08 | 2.17e-04 | 5.97e-05 | 5.22e-08 |
| bf16 | **inf** | 3.96e-04 | 7.29e-04 | 4.91e-04 | 7.73e-04 |

- A decode step takes **0.036 ms** (B=16, H=6), vs 0.296 ms for the chunk-kernel fallback: about 8× faster.
- Verdicts: **C110 VERIFIED** (the old NaN was our call), **C111 VERIFIED** (decoding works, including neg-eig), **C2 VERIFIED** (it was REFUTED by a wrong test until v0.4.27). Speed prediction: hit.

**What it unblocks**: a fast decode path for every existing GDN checkpoint: SQ53 (KDA/GDN decode), C19, C84 (RL sampling cost without a cache) and E8. Any earlier result that relied on the positional call is invalid. Only `tests/test_gdn_kernels.py` and SQ50 used it, and both were reporting-only, already fixed in v0.4.27.
**Note**: the queue's GPU-slot record lagged (K8 still marked running) when SQ70 launched; the device itself was free and status.sh showed no process. Fixed in the queue.

---

## v0.4.28: K8: external reference, plausibility gate fails narrowly; K9 next
*2026-11-22*

### K8: external bits/char reference (SmolLM2-135M)
**What**: our 4,000 val docs scored doc-level in bits/char, a tokenizer-independent measure, for SmolLM2-135M (local snapshot, rev 93efa2f) and the v0.3.0 E0 and E1 seed-0 checkpoints, using `experiments/checks/k8_external_reference.py` on the GPU (71 s). **Deviation, logged in advance in the queue:** the registered K8 scores the *tuned* E10/E1 checkpoints. This early run uses the existing ones; Q11 repeats it on the tuned checkpoints after Phase 2b.
**Results** (`experiments/checks/results/K8.json`):
- **SmolLM2-135M 0.4490**, E0 0.2209, E1 0.2126 bits/char.
- **K8a plausibility FAIL:** ours / SmolLM2 = 0.49 / 0.47 against a ≥ 0.5 gate. A 14M model beating a 135M general model by about 2.1× may just be in-distribution training on this synthetic corpus. **The registered rule still requires ruling out train/val near-duplicate leakage before trusting absolute loss numbers.** That's **K9**: a 13-gram overlap check of val vs train with a pseudo-val baseline, being written by a subagent, with its prediction committed before it runs.
- **K8b arithmetic FAIL:** doc-level vs training-time differ by 6.0% / 6.2% against a ≤ 5% gate. It's the packed-vs-per-document protocol gap already explained in C101, which the gate didn't allow for. It's recorded as FAIL; the gate isn't moved after the fact.
- **Prediction:** 2 hits (SmolLM2 range, our range), 1 narrow miss (margin 51–53% vs the predicted 20–50%), 1 miss (K8b).

**Ledger**: C94 stays **UNVERIFIED** (pending K9). Absolute loss numbers are provisional; same-split model comparisons are less affected.
**Decision**: Phase 2b (the replication) was launched at 18:13 anyway. Leakage would affect every model on the same val split, so the runs are still needed. Their verdicts (Q12) wait for K9.

---

## v0.4.27: Hardware research registered; the C2 'NaN' was our call
*2026-11-22*

### Hardware research: NVIDIA RTX Spark N1X (registered, nothing run on the GPU)
**What**: a subagent researched the GPU and the main agent reviewed it.
- **`notes/hardware-spark.md`**: a spec table for the N1X vs GB10, with confidence labels.
  - 48 SMs, 6,144 cores, CC 12.1. 5th-gen tensor cores; there is **no wgmma and no tcgen05**, and sm_120 binaries run on 12.1.
  - LPDDR5X, 300 GB/s, coherent over NVLink-C2C. Windows reserves 96 GiB; WSL sees about 14 GB of RAM. Observed 35–56 W under training.
  - Dense BF16 is about 98–118 TFLOP/s *if* the tensor cores run at full rate. Our measured 72.1 is 61–73% of that; SQ71 settles it.
- **Roofline**:
  - E0 reaches 20.3 TFLOP/s, E1 9.1 and E2 8.5 (28%, 13% and 12% of the measured peak).
  - Training is **memory-bandwidth / elementwise-bound**: about 160 FLOP/byte for E0, below the ridge of about 290. Batch-1 decoding is launch-bound.
  - So FP8, int8 or ternary kernels can't buy much at our shapes.
  - This is an estimate until SQ75 profiles it.
- **Ranked, opt-in patches** in `notes/patches/` (defaults identical; they apply alone and together):
  - P1: fla keyword fix;
  - P8: temperature-aware campaign gap plus a persistent inductor cache (K7-type campaigns ~20–25% shorter);
  - P5: fused GDN conv + q/k norm (E1/E2 +5–20%);
  - P4: fused linear cross-entropy;
  - P3: flex-attention sliding window (fixes C12);
  - P6/P7: compile mode, data kept on the GPU.

  They're applied one at a time, each checked by its own SQ, between campaigns.
- **Checks:** 10 GPU check scripts `experiments/checks/sq_hw_*.py`, with PREDICTION and CRITERION written before any run. They're registered by this version (queue item HW-REG) and queued as SQ70–SQ78; SQ79 is CPU.
- **Ledger and queue:** C110–C129 and the queue items were appended before v0.4.26 was cut, so they went into that commit; they're recorded here. The side-quest rows SQ70–SQ83 are in this version.
- **References:** 14 NVIDIA documents (62.6 MiB, PDFs git-ignored) are listed with URL, size, sha256 and reason in `refs/12-hardware/SOURCES.md`. Observed `nvidia-smi` readings are in `refs/12-hardware/observed/`, with UUIDs redacted (checked).
- **Needs the human (parked):** Windows "Best performance" power mode (P10/SQ82); optionally install torchao (SQ83).

### C2 corrected: the recurrent-kernel "NaN" was our bug
fla 0.6.0's signature is `fused_recurrent_gated_delta_rule(q, k, v, g=None, gk=None, gv=None, beta=None, scale=...)`. The main agent read this in the installed source. `tests/test_gdn_kernels.py` called it positionally as `(q, k, v, g, beta, scale=)`, so **beta was passed as `gk`** (a per-key log decay) and the real beta defaulted to ones. The chunk kernel's 5th argument *is* beta, which is why that test passed.
- The test and `sq_kda_kernel.py` (SQ50) now use keyword arguments.
- **C2: REFUTED → UNVERIFIED** (the old verdict is withdrawn, because the test was invalid). SQ70 re-tests it.
- If SQ70 passes, existing GDN checkpoints get a decode path, which unblocks SQ53, C19, C84 and E8.
- The `notes/patches/fla-recurrent-keyword-args.patch` fix is applied directly to the test here.

---

## v0.4.26: K7 done: tuned LRs E10 3e-3, E1 6e-3; Phase 2b campaign written
*2026-11-22*

### K7 (final): LR tuning
**What**: the grid extension required by the pre-registered rule: one E1 run at lr 1.2e-2, 20M tokens, seed 0 (`phase2a-k7-lr-ext.campaign`). Then `k7_lr_sweep.py --write-campaign` re-applied the rule.
**Results**:
- E1 at 1.2e-2: **1.0656** (+7.4% vs 6e-3), so **E1's tuned LR is 6e-3 (val 0.9925), bracketed**.
- **E10's tuned LR is 3e-3 (1.0286)**.
- Over 9 runs, E1 beats E10 at every LR tested; at their own best LRs the gap is −3.5% (single seed, 20M tokens).
- The rule generated `experiments/launch/campaigns/phase2b-replication.campaign`: 13 runs × 200M tokens, ~7 GPU-hours.
  - E10 seeds 0–2 at 3e-3; E1 seeds 0–2, E11 seeds 0–2, E9 seed 0 and E2 seed 0 at 6e-3.
  - E0 is skipped: its v0.3.0 seed-0 run is already at 3e-3.

**Ledger**: C92 final row, PARTIAL (3e-3 is right for E10, too low for E1; the optimum at 200M tokens may shift). The K7 card is done. **Queue**: K7 and Q5 done.
**Next**: K8 (~6 GPU-min; external reference and leakage gate on all our loss numbers), then launch Phase 2b.

---

## v0.4.25: K7 stage 1: E10 tuned at 3e-3, E1 wants a higher LR; hf.py fixed
*2026-11-22*

### K7 (stage 1 of 2): LR sweep, 8 runs × 20M tokens, seed 0
**What**: the pre-registered sweep {1e-3, 2e-3, 3e-3, 6e-3} for the control E10 and the hybrid E1. Campaigns `phase2a-k7-lr` (item 1) and `phase2a-k7-lr-resume` (items 2–8; see v0.4.16 for the interruption); code paths identical across all 8 runs. `experiments/checks/k7_lr_sweep.py` applied the decision rule → `experiments/checks/results/K7.json`.

**Results** (final val loss, packed protocol):

| LR | 1e-3 | 2e-3 | 3e-3 | 6e-3 |
|---|---|---|---|---|
| E10 | 1.2385 | 1.0650 | **1.0286** | 1.0889 |
| E1 | 1.1625 | 1.0367 | 1.0071 | **0.9925** |

- **E10: tuned LR = 3e-3** (bracketed).
- **E1: best at the top edge (6e-3)**, so rule step 2 extends the grid to 1.2e-2. That one run is next (`phase2a-k7-lr-ext.campaign`).
- E1 beats E10 at every LR tested.
- **Prediction scorecard:** 3 hits (E10's range, no divergence, no low-edge winner) and 3 misses (E1 wanted a higher LR than predicted; the curves are sharper than "flat within 1%"; 1e-3 was +17–20% worse, not +3–8%).
- **Implication**: the v0.3.0 FusionDance results at 3e-3 probably ran below their best LR while E0/E10 sat near theirs, so C5 (E1 beats E0) may understate the hybrid's edge. The replication at tuned LRs (Phase 2b) will tell.

**Ledger** (AGENTS §5 rule):
- **C92 → PARTIAL** (3e-3 is right for E10, not for E1);
- **C80 → VERIFIED** again after the fix below.

### Q30: `gogeta/hf.py` fixed
`GogetaConfig` gains `gdn_neg_eigval`. `GogetaForCausalLM` builds again and round-trips the neg-eig flag. `post_training/test_hf.py` 10/10 OK, `evals/test_wiring.py` 6/6 OK (CPU). It's a code-path edit, made in the gap between campaigns. `evals/gogeta_lm.py`'s fill-missing-fields shim is now a no-op and can go later.

---

## v0.4.24: Claims list (claims-status.md); run artifacts out of git again
*2026-11-22*

**What** (the human: "i dont want the run artifacts I want the list of claims"):
- **Reverted v0.4.23's run mirroring:** `experiments/runs/` (33 files: run `meta`/`summary`/`samples` and `sync.sh`) is removed from git, and so are the `.gitignore` exception and the §5 sync rule. The run folders still exist on the WSL disk; nothing was deleted there. The no-worktree rule from v0.4.23 stays.
- **`experiments/claims-status.md`** is a generated, readable list of **every claim with its latest status**, grouped by status, with the latest evidence and a link to its ledger line. `experiments/claims_list.py` (stdlib) builds it.
  - **Why it's needed:** the ledger `experiments/claims.md` is append-only, across 12 sections, so a claim's first row can be stale. For example, C15 and C22 still read UNVERIFIED in the main table but were settled in the K1–K6 verdict section. The list keeps the last status per ID and flags changes ("⟵ was UNVERIFIED").
  - It's regenerated with every version (AGENTS §5).
- **C80 updated to PARTIAL:** the Hugging Face wrapper was exact at v0.4.6 but has been broken since v0.4.7 (`gdn_neg_eigval` missing; fix queued as Q30). It's recorded in a new ledger section, "Status updates".

- **AGENTS §5 rule** (the human: "each experiment must update it at the end"): every experiment, check, side quest, sweep or eval **ends by appending ledger rows** for each claim it tested (status + evidence + results file + version) and regenerating `claims-status.md`. No version can be cut without it. The ledger stays at the root of `experiments/`.

- **Near-miss, caught before commit:** the list was first named `CLAIMS.md`. On Windows that's the same file as `claims.md`, so generating it overwrote the ledger. Git showed `claims.md` losing ~190 lines; the ledger was restored from HEAD and my C80 row re-added. It's now `claims-status.md`, and the script refuses to write if its output resolves to the ledger or if it parses 0 claims.

**Results**: 96 claims: **17 VERIFIED, 15 PARTIAL, 6 REFUTED, 2 WRONG FOR OUR CODE, 56 UNVERIFIED**. The unverified ones are mostly paper claims proposed by the research subagents and still pending review.

---

## v0.4.23: Run records in the repo; no-worktree rule restated
*2026-11-22*

**What** (the human, about `experiments/checks/results/` and worktrees: why isn't this in the repo?):
- **Audit:**
  - The repo has exactly one checkout and no worktrees (`git worktree list`: only `C:/Users/mreca/Desktop/Gogeta-0 [main]`).
  - All 8 check results (K1–K6, K7-parity, sq_06) are tracked and on GitHub.
  - **The gap was the training runs:** 15 run folders (`meta.json`, `summary.json`, `samples.txt`) existed only on the WSL disk under `~/fusiondance-data/runs/`. That includes E0/E1/E2's records and the K7 sweep runs, so the devlog's training numbers had no committed source file.
  - The old `C:\Users\mreca\Desktop\FusionDance` folder is a stale v0.1.1 copy; it has no results.
- **Fix:** `experiments/runs/sync.sh` mirrors each finished run's record files into `experiments/runs/<run>/` (a few KB each; refuses token-like strings; never checkpoints or data). This version commits 11 runs: E0 s0, E1 s0 (v0.3.0), E2 s0, a smoke run, and the 7 K7 sweep runs finished so far. The 4 without a `summary.json` are skipped: E9 s0 and E2 s1 were killed before finishing, one smoke run, and the K7 run still in flight.
- **Root cause:** the generic `runs/` rule in `.gitignore` (from v0.1.0, meant for training output folders) also matched `experiments/runs/`, so run records could never be committed. `.gitignore` now re-includes `/experiments/runs/`.
- **AGENTS §5:**
  - every result lives in the repo; run `sync.sh` before each version, and only checkpoints, data, caches and raw logs stay outside git;
  - one checkout, no worktrees, for subagents too;
  - never read from or write to the stale FusionDance copy.

---

## v0.4.22: 3D page: FizzBuzz Play trace, legend, polish, label-leak fix
*2026-11-22*

**What** (the human asked for a Play button tracing "write fizzbuzz in python", a visual pass on every model, fun interactions, a full legend, and a fix for labels that never cleared):
- **Play mode:** play / pause / step / back / restart, speed 0.5–4×, a stage readout (Embedding → Prelude → Core loop k/r → Coda → Output) and a follow-tokens camera.
  - The 9 **real** tokenizer pieces of `# write fizzbuzz in python\n` (`#` `␣write` `␣f` `izz` `bu` `zz` `␣in` `␣python` `↵`) rise as chips.
  - Attention and SWA arcs show the measured top-3 attention weights; GDN tokens stream into the state cube; each re-injection flashes.
  - Per-loop Δh bars fill as loops finish, and output chips carry top-5 probabilities.
- **Trace (C130, PARTIAL):** `experiments/viz/export_trace.py` writes `docs/data/token-trace.json` from the E2 checkpoint (r=4) on CPU, using a plain-PyTorch GDN reference.
  - The traced forward equals `model.forward` to 9.5e-6.
  - CPU fp32 val loss is 0.6423 vs the recorded 0.6186 (bf16 with fla kernels), so the numbers are close, not identical.
  - Per-loop ‖Δh‖/‖h‖: 10.82, 0.72, 0.40, 0.28.
  - Next-token top-5: `↵` 36%, a code fence 34%, `assert` 8%, `#` 8%, `def` 4%.
  - The greedy continuation is a `find_unique_numbers` function plus an assert. **It's not FizzBuzz**, as you'd expect from a 13M model.
  - Other models and other r are labelled ILLUSTRATIVE.
- **Legend:** 23 rows generated from the scene's own style tables. Hovering a row highlights the matching objects, clicking flies the camera to them, rows for parts the model lacks are dimmed, and decorative and Play-mode rows are tagged.
- **Polish:** fit-to-bounding-box framing with fly-to, wider block spacing, ground grid and glow disc, rim and warm lights, a calmer starfield, level-of-detail labels, ring spinners that speed up with r, hover pulse, click-to-focus with a plain-English line, a fusion flash and burst on model switch, keyboard shortcuts, Save PNG.
- **Label/memory leak fixed:** removing a model group left its nested CSS2D label divs in the DOM and never disposed geometries, materials or textures. Now one `disposeGroup()` / `clearModels()` path handles every rebuild. `window.__arch3dSelfTest(rounds)` stress-tests it.

**Results**:
- Subagent stress test, before the fix: label divs 36 → 114 and geometries 41 → 250 over 5 rounds. After the fix, 20 rounds: labels 23 → 23, geometries 33 → 33, textures 1 → 3 (the ternary speckle cache, by design).
- **The main agent re-ran both checks:** `__arch3dSelfTest(5)` passes, and an independent count of positioned CSS2D elements stays at 23 → 23 across 21 manual model switches plus full sweeps of every slider.
- Before/after screenshots are in `experiments/viz/screenshots/`.

**Lesson:** the v0.4.18 check (load once, no console errors, one screenshot) missed the leak; the human caught it. UI checks must now drive the page through its states and compare counts afterwards.

**Not verified**: label crowding at r=12 and in compare mode (still there); chips overlapping labels at 375 px; touch; the OS reduced-motion setting; Play combined with compare; the trace on GPU with fla.

---

## v0.4.21: Outside review every 30 minutes
*2026-11-22*

**What** (the human: "every 30 minutes spawn a haiku subagent to review the model and to give alternative ideas and analysis on the state of the project so we can add differences and think wider about the project and experiment more"):
- **AGENTS §0d, "Outside review every 30 minutes: think wider".** A fresh Haiku reviewer reads the README, the devlog, the index, the queue, the ledgers, `gogeta/model.py` and the previous review. It writes one file, `notes/reviews/<time>-review.md`, containing:
  - the state of the project (cited facts);
  - a critique of the architecture and the method;
  - 5–8 alternative ideas, each with a hypothesis, the cheapest test and the result that would change our plans;
  - the one experiment it would run next.
- The main agent triages every idea: promising ones go to the side quests and the queue, rejected ones get a one-line reason. Each review is committed as its own PATCH version and pushed.
- Ideas feed the queue but never preempt the running campaign or the validity-first order.
- Scheduled as a session cron job at minutes :07 and :37. The §7 docs table lists `notes/reviews/`.

---

## v0.4.20: Always push to the remote
*2026-11-22*

**What** (the human: "add this to the agents.md to always push to the remote"):
- **AGENTS §3:** after every version is committed and tagged, run `git push origin main --follow-tags` straight away, so GitHub, the tags page and the Pages site always match the local repo; then check the Pages build. Failed pushes are reported and fixed, never forced. The rule covers only `main` and its version tags on `origin`.
- **§0a and §5b:** pushing to `origin` is no longer on the safety list. Publishing anywhere else still needs the human.
- This push sends v0.4.1 to v0.4.20. Earlier pushes had been held back for the human.
- The generated per-experiment site pages and `learn.html` (nav-link regeneration) are included.

---

## v0.4.19: Standard benchmark suite (lm-eval) + sandbox gap found
*2026-11-22*

### Benchmark suite (tooling + first E0 probes)
**What**: `evals/` adds an lm-evaluation-harness 0.4.13 adapter (`gogeta_lm.py`, registered as `gogeta`) and tiered tasks run by `python -m evals.run_suite`. The venv is `~/.venvs/gogeta-eval`, pinned in `requirements-eval.txt`. A subagent built it; the main agent checked its key claims.
- **Tier 1, informative at our size:**
  - per-document bits per byte on our val set and on fresh GitHub Python (UncheatableEval, Oct 2026);
  - likelihood of the reference solutions for HumanEval and MBPP, each with a `_clean` twin;
  - CRUXEval-O as a likelihood probe;
  - functional pass@1 on our own held-out problems.
- **Tier 2:** HumanEval(+) and MBPP(+) pass@k. Expected to floor near zero: Codex-12M got 2% after 100B tokens.
- **Tier 3:** a recall probe; it floors and needs a redesign.

The survey, tiers and predictions are in `notes/benchmark-suite.md`.

**Results** (E0 seed 0, CPU, reported by the subagent):
- wiring tests 6/6;
- E0 val **0.2208 bits/char per document** vs train.py's 0.2351 packed. The −6.1% missed the ±3% prediction.
  - Re-running train.py's own method reproduces its loss to 0.014%, so the gap is protocol, not a bug: packed windows start mid-document with no context.
  - **The two numbers must never be compared directly** (C101).
- HumanEval reference-solution bpb 0.5106 (clean 0.5174); MBPP 0.6398 (clean 0.6544);
- CRUXEval-O exact match 8.9% (always answering "False" scores 3.6%);
- recall at chance.
- **Contamination:** 32 of 164 HumanEval problems have at least half of their long lines verbatim in our training data; 43 of 500 MBPP statements appear verbatim (C103).

**Verified by the main agent:**
- `evals/sandbox.self_test()` passes **11/11**.
- A control write into the repo is **blocked** by `evals/sandbox.py` but **succeeds** through `post_training/rewards.run_tests`.
- `GogetaForCausalLM` fails to build, because `gogeta/hf.py` lacks the `gdn_neg_eigval` field added in v0.4.7.

### Safety finding: the reward sandbox can write the repo (C105, REFUTED claim)
Under WSL, `/mnt/c` is mode 777, so model code run by `rewards.run_tests` can write this repo and all of C:, even as `nobody`. `post_training/SETUP.md` and AGENTS §6 are corrected: evals run generated code only through `evals/sandbox.py`, and lm-eval's built-in code tasks are never loaded. The fix is queued as **SQ61** (move the hardening into rewards.py; it must come before any GRPO run). `.sandbox_probe_*` is git-ignored.

**Queue**: Q30 (fix `gogeta/hf.py`, a code-path edit, so it runs in the gap after K7) and SQ61 were added and prioritised. **Ledger**: C100–C109 and SQ60–SQ69 appended.
**Downloads** (eval data, approved in the eval-suite request): about 14 MB of datasets into `~/fusiondance-data/eval-cache` (outside the repo). No model weights.

---

## v0.4.18: 3D architecture page; README quick links; repo topics
*2026-11-22*

**What**:
- **`docs/architecture.html`** is a three.js (r160, pinned through an import map) 3D model explorer, built by a subagent and verified by the main agent. Every shape encodes a number:
  - **plinth height** = the block's parameters;
  - **full attention** = a triangular wedge whose area grows as n² with context;
  - **SWA** = a fixed-width band that only grows in length;
  - **GDN** = a fixed cube the size of its state that never grows with context;
  - **MLP** = a bicone (d → hidden → d);
  - **the looped core** = a gold "CORE × r" ring with ghost pillars per loop ("same weights, 0 new params");
  - **ternary parts** carry a −1/0/+1 speckle and a 1.58-bit badge.

  The page also has context and loop sliders, unroll, compare mode, hover tooltips with exact numbers, a scouter power-level panel, reduced-motion support and a WebGL fallback.
- **`tools/viz/export_arch.py`** builds the model from each committed config on CPU and writes `docs/data/architectures.json` (7 models). Parameter totals match the earlier records (E0 13,767,552; E1 14,237,796; E2 12,606,488). The page adds a nav link from every site page (`tools/site_pages.py`) and a call-to-action on the landing page.
- **README**: a Quick-links row (site, devlog, learn, 3D model, experiment index, queue, claims ledger, tags). It was added by a Haiku subagent. That edit was saved while v0.4.17 was being cut, so it landed in v0.4.17's commit; it's recorded here. A duplicate Pages badge was removed.
- **GitHub repo metadata** (the human asked for GitHub tags):
  - 12 topics: llm, language-model, pytorch, linear-attention, gated-deltanet, state-space-models, looped-transformer, bitnet, ternary-weights, code-generation, python, research;
  - a description;
  - homepage = the Pages site.

  Pages was already enabled (main /docs, latest build "built").
- **Hourly docs refresh** (the human's request): a session cron job spawns a Haiku subagent each hour to refresh README, sketch.md and the setup guides from the repo. The main agent verifies its diff and cuts a PATCH version only when something meaningful changed.
- **K7 resume**: `experiments/launch/campaigns/phase2a-k7-lr-resume.campaign` holds items 2–8 of the LR sweep.

**Results** (main agent, Browser pane):
- the page loads with no console errors;
- the canvas renders (screenshot checked: E2's looped core with r = 4, scouter panel showing 12,606,488 params);
- all 7 models are in the selector.

The subagent's checks:
- scene object counts scale with loops (71 / 142 / 230 at r = 1/4/8);
- the attention wedge goes 0.4 → 12.8 from 1k → 32k context while the SWA band width stays 0.1;
- no horizontal overflow at 375 px.

**Not verified**: touch gestures, the reduced-motion and no-WebGL paths, clutter at r = 12 with 32k context.

---

## v0.4.17: Kimi Linear research; SQ50 attempt timed out
*2026-11-22*

### Kimi Linear (research)
**What**: `notes/research-kimi-linear.md` (subagent research, reviewed). It compares Kimi Delta Attention (KDA) with our GDN.
- **GDN is exactly KDA with the decay tied across channels.** KDA uses per-key-channel decay `Diag(α)` from a low-rank projection, a low-rank sigmoid output gate and head dim 128.
- Kimi Linear interleaves 3 KDA layers to 1 global layer, and its global layers are MLA with **NoPE**.
- Reported ablations (validation perplexity, paper numbers, not ours):
  - ratio: 3:1 5.65, 1:1 5.66, 7:1 5.70, full attention 5.77;
  - output gate: none 5.81 vs gated 5.67;
  - no short conv: 5.70.
- The paper's decode-speed figures are internally inconsistent (6.3× "theoretical" vs 2.3×); this is recorded in C78.
- **There is no isolated ablation of scalar vs channel decay**, so C73 is only weakly supported.

**Ledger**: C70–C79 appended as UNVERIFIED, pending review. SQ50–SQ59 are already in `experiments/side-quests.md` and the queue (top picks: SQ50 kernel check, SQ51 KDA vs GDN at d=256, SQ52 NoPE vs RoPE in the single full-attention layer).

### SQ50 (attempted, no result)
**What**: `experiments/checks/sq_kda_kernel.py` checks fla 0.6.0's `chunk_kda` / `fused_recurrent_kda` against an fp32 reference, the tied-decay reduction to GDN, and speed. PREDICTION and CRITERION are fixed at the top of the script before the run.
**Result**: **none.** The GPU run hit the subagent's 300 s timeout with no output.
- During it, other GPU jobs (SQ6, K7-parity) were started, so the one-GPU-job rule (AGENTS §5) was broken during that window. The SQ6 and K7-parity results were already checked separately; they are timing-insensitive accuracy checks.
- On CPU, the subagent confirmed our reference loop matches fla's pure-PyTorch KDA (≤ 4.4e-7) and that tied decay equals the GDN reference exactly. That checks the maths, not the kernel.
- C70–C72 stay UNVERIFIED.

**Next**: re-run SQ50 alone on an idle GPU with a ≥ 600 s budget (first-run autotune) through `tools/launch_cpu_job.sh`-style detachment, not a subagent's foreground shell.
**Refs to add (checked before fetching)**: 2505.06708, 2504.08719, 2501.18795, 2506.02475, 2505.16381, 2502.16982.

---

## v0.4.16: Q6: replication analysis script
*2026-11-22*

### Q6: replication analysis (tooling for Phase 2b)
**What**: `experiments/checks/replication_analysis.py` (stdlib + numpy; written by a subagent, reviewed and self-tested by the main agent). It applies the **already-committed** Phase 2b criteria from `experiments/checks/PREDICTIONS.md` and the registry to run folders: C5 (E1 beats E10: VERIFIED / PARTIAL / REFUTED), seed noise, old-E1 reproducibility, E0 parity, the loop question (E9 vs E2, threshold 2√2·s_E1), E11 (free / helps / hurts), and the E10 prediction scorecard. No new prediction: it only mechanises criteria registered in v0.4.11 and v0.4.13.
**Run selection**: status done, `max_tokens` = 200M (drops K7's 20M sweep runs), LR equal to the K7-tuned LR (read from `results/K7.json`), code not dirty, latest run per (experiment, seed). Any verdict with missing runs is INCOMPLETE and lists what's missing. Nothing is computed on partial data.
**Results**: `--selftest` PASS: every verdict branch is exercised on synthetic run folders through the real discovery and filter code, including five excluded run types. On real data (`--no-save`) every verdict is INCOMPLETE, as expected: K7 isn't finished and Phase 2b hasn't run.
**Caveats**: per-step train-loss jumps (E10 prediction item 5) can't be checked from `summary.json`, so they're reported NOT_CHECKED. The repo save path (`common.save`) is untested until the first real run.

### K7 (in progress): campaign interrupted after item 1/8
Item 1 finished: E10, lr 1e-3, 20M tokens → val 1.2385, bpc 0.4802, 1.86 min. Item 2's preflight then **refused**, because a subagent had created an untracked file under `tools/` (the 3D page's `tools/viz/export_arch.py`), and the campaign aborted by design (code paths must be committed). No result was affected. Items 2–8 will relaunch once that work is committed. Lesson: subagent briefs must not create files under code paths while a campaign runs.

**Queue**: Q3 done (K8 `--plan` resolved the snapshot), Q6 done.

---

## v0.4.15: Autonomy rule: never block on the human
*2026-11-22*

**What** (from the human: "don't ask me what to do, I want you to be autonomous; if you ask me while I'm sleeping we waste time"):
- **AGENTS.md §0a, "Be autonomous":** decide and proceed; thesis-direction calls are made by the agent with a written decision memo; never end a turn with a question while work remains; failures are diagnosed and fixed, not escalated.
- **The only actions not taken alone** are the safety list: downloads/installs, push/publish, deleting data, spending money. Even these **never block**: they're parked under "For the human" in `experiments/QUEUE.md` while other work continues.
- **§5b updated to match:** `needs_human` now covers the safety list only. Multi-hour GPU items and thesis decisions are autonomous.
- **Queue:** `needs_human` cleared on Q23 (loop-purpose decision memo) and E4 (full fusion, multi-hour); still parked: Q14 (eval-suite downloads/installs) and SQ7 (Mamba-3 package build). K7 and Q3 are marked running (launched from the queue at 17:25).

**Results**: `tools/queue.py validate`: ok (45 items).

---

## v0.4.14: Experiment queue + autonomous-queue rule
*2026-11-22*

**What**:
- **`experiments/queue.json`**: 45 prioritized items with dependencies, resources (GPU/CPU, minutes), `ready`/`missing`, `needs_human` and its reason, exact `launch` commands, and sources (queue subagent; reviewed by the main agent).
- **`tools/queue.py`** (stdlib): `list`, `next --gpu|--cpu [--max-minutes N]`, `show`, `set`, `add`, `move`, `validate`, `render`. It validates ids and cycles, never reorders silently, refuses a 2nd GPU or a 9th CPU item, and `next` skips `needs_human` items.
- **`experiments/QUEUE.md`**: the generated view.
- **AGENTS.md §5b:** the main agent works through the queue without asking the human, and asks only for `needs_human` items (download or install, > 3 GPU-hours in one item, deleting data, pushing, thesis-direction change) or when a result contradicts an item's premise.
- Q1 (predictions registered: v0.4.11, v0.4.13) and Q2 (`--lr` + campaign runner: v0.4.10) marked done.

**Results**: `validate`: ok (45 items, no unknown dependencies, no cycles). `next --gpu` → **K7** LR sweep (43 min); `next --cpu` → **Q3** K8 `--plan` (2 min).
**Ordering policy**: validity first (controls, tuning, seeds), then cheap inference-only diagnostics on existing checkpoints, then thesis-critical runs (E11, neg-eig in the loop, E3 at the tuned LR, loop fixes), then side quests, then long-term items.
**Needs the human (queued, not auto-run)**: Q14 eval suite tier 1 (downloads/installs), Q23 loop-purpose decision memo (thesis direction), E4 full fusion (> 3 GPU-h), SQ7 Mamba-3 kernels (install/build).
**Next**: run K7 (GPU) and Q3 (CPU) from the queue.

---

## v0.4.13: K7 and K8 registered: LR sweep and external reference
*2026-11-22*

**What**: registers the two checks that the Phase 2 campaign runs first and last. Predictions and decision rules are committed here, **before any run**.
- `experiments/checks/k7_lr_sweep.py` (applies the K7 rule and writes the next campaign; rule logic tested on 8 synthetic cases), `experiments/launch/campaigns/phase2a-k7-lr.campaign` (8 short runs) and `phase2b-replication.template`.
- `experiments/checks/k8_external_reference.py`: doc-level bits/char for our checkpoints and SmolLM2-135M. It loads only from the local cache, at the pinned revision `93efa2f`, and refuses on a mismatch.
- `experiments/checks/PREDICTIONS.md`: K7, K8, and the **replication protocol** with its comparison criteria. Cards K7 and K8.
- AGENTS.md: campaign runner and `--lr` (§5); controls are LR-tuned before use and E10 becomes the control (§6); docs table rows.
- Site regenerated (`docs/`) and `experiments/INDEX.md` regenerated from committed sources; version 0.4.13.

### K7: LR tuning for E10 (control) and E1 (best hybrid)
- **Question:** what's each model's best peak LR? Every earlier run used an untuned 3e-3.
- **Result:** not run. Phase 2a: lr ∈ {1e-3, 2e-3, 3e-3, 6e-3} × {E10, E1}, seed 0, 20M tokens, ~43 min GPU.
- **Verdict vs prediction:**
  - Prediction: E10's best is 3e-3 or 6e-3, E1's is 2e-3 or 3e-3, both curves flat near the optimum, and 1e-3 is 3–8% worse.
  - Rule: the lowest full-val loss wins (ties go to the smaller LR). If the winner is at the grid edge, extend one step (at most twice per side).
- **What it means:** the replication (E10 / E1 / E11 × 3 seeds, E9, conditional E0 and E2) then runs at the tuned LRs.
- **Card:** experiments/cards/K7.md

### K8: External sanity reference (SmolLM2-135M, bits/char)
- **Question:** are our loss numbers in a sane range against a well-known public model?
- **Result:** not run (runs after Phase 2b, ~5 min).
- **Verdict vs prediction:**
  - Prediction: SmolLM2 0.28–0.45 bits/char, ours 0.21–0.26.
  - K8a: ours ≥ 0.5 × SmolLM2, else suspect train/val leakage.
  - K8b: doc-level bits/char within 5% of the training-time value.
- **What it means:** a sanity ceiling only (10× larger, different data, possible contamination), not a control.
- **Card:** experiments/cards/K8.md

**Phase 2 GPU budget** (estimated from v0.3.0 wall times): ~6.5 h if both tuned LRs are 3e-3, up to ~8.5 h with conditional runs and grid extensions.

---

## v0.4.12: K7-PARITY: E10 construction matches E0; FusionDance unchanged
*2026-11-22*

### K7-PARITY: E10 construction parity and FusionDance unchanged
- **Question:** does the new `train.py` builder leave FusionDance bit-identical, and does E10 match E0 exactly in params and FLOPs?
- **Result** (CPU, `results/K7-parity.json`):
  - state_dicts, model config, RNG state and FLOPs are **bit-identical** to the old construction for E0, E1, E2, E3 and E9 (re-checked after the `gdn_neg_eigval` change);
  - E10: total 13,767,552 = E0, non-embedding 10,621,824 = E0, FLOPs per token 9.676e7 = E0;
  - E10 at init: loss 9.094, tied embeddings.
- **Verdict vs prediction:** PASS.
- **What it means:** E10 vs E0 is a clean implementation comparison. Caveat (ledger C91 PARTIAL): a 101-step GPU replay of E0 matched the v0.3.0 log at step 0 exactly (9.0900) but read 2.8370 vs 2.8460 at step 100. Likely GPU nondeterminism, but unproven; an old-vs-new replay is the first Phase 2 item.
- **Card:** experiments/cards/K7-parity.md

---

## v0.4.11: E10 registered: Hugging Face Llama reference control
*2026-11-22*

**What**: registers **E10**: Hugging Face's reference `LlamaForCausalLM` (transformers 5.24.0) at E0's exact size, trained by our own loop, data, eval and tracking. **Not run yet.**
- `gogeta/train.py`: a `"arch": "hf_llama"` config key (default `fusiondance`), an `HFLlama` adapter with loss computed identically, FLOPs by the same rule as FusionDance, and a `--lr` override. Run configs record `arch` and `cli_overrides`.
- `configs/e10_llama_ref.json`, `experiments/launch/E10.sh`, card, plain-English page, registry entry with its **prediction**; E9's card, and a note that E9 runs at E1's tuned LR.

### E10: Reference control: Hugging Face Llama at E0's size
- **Question:** what does a well-known reference implementation reach on our data with a tuned LR? Is our E0 baseline weak?
- **Result:** not run. It runs in the Phase 2 campaign: K7 LR sweep, then 3 seeds.
- **Verdict vs prediction:** prediction (registry): 3-seed mean 0.595–0.615, seed sd ≤ 0.004; E0 at the same LR within mean ± max(3 sd, 0.5%); E1 still beats it by 1.5–4%.
- **What it means:** E10 becomes **the** control for every comparison once it lands.
- **Card:** experiments/cards/E10.md

---

## v0.4.10: Campaign runner + launcher --lr
*2026-11-22*

**What**:
- **`tools/launch_campaign.sh <file>`** runs one launch per line via `launch_experiment.sh --foreground --wait-cool`, sequentially and detached (setsid nohup, PID and state files). It waits while the GPU slot is held, any GPU process exists, or the GPU is ≥ 85 °C; it retries transient refusals and aborts on persistent ones; it has `--status`, `--resume`, `--wait` and `--stop`.
- **`launch_experiment.sh --lr X`:** validated, passed to `gogeta.train --lr`, shown in the plan with a WARN, and recorded in the run config. Written by the cleanup agent as a patch outside its permissions; reviewed and applied by the main agent.

**Results**: the runner was tested end to end by the cleanup agent with a stub launcher and a fake `nvidia-smi` (two bugs fixed). The main agent re-tested `--lr` with dry runs (`--lr 2e-3` accepted and recorded; `--lr abc` rejected).
**Not verified**: a real campaign hasn't run yet.

---

## v0.4.9: Science gaps: wrong claims corrected everywhere, gap list
*2026-11-22*

**What** (control + science-cleanup agent, Phase 1; reviewed by the main agent):
- **Three wrong statements fixed everywhere they appeared**, now matching the ledger and K1/K2:
  - C12 (sliding window "O(n·w)": our dense-mask kernel is O(n²)): `docs/index.html`, `README.md`, `site-src/learn.md`;
  - C13 ("~10× smaller" → computed 7.8× for the body, 3.1× for the whole model): `site-src/learn.md`;
  - C14 ("ternary makes compute cheap" → only with ternary kernels; ours are simulated): `docs/index.html`, `README.md`, writeups E3/E4.
- **`notes/science-gaps.md`:** every known gap, its fix and its status. The six original gaps are fixed or designed. New ones found:
  - E1-vs-E0 "matched compute" is ambiguous (+4.4% params, −6% FLOPs; C95);
  - single-seed claims are stated as fact in the README and primer (left until the replication decides C5);
  - `val_bits_per_char` bias ~0.05%;
  - train/val near-duplicate leakage unchecked;
  - one compute-budget violation (two out-of-launcher GPU jobs at ~17:05);
  - detached-run survival unverified.
- No prediction was ever edited after the fact (checked against the registry's git history).

**Results**: n/a (docs). The site HTML is regenerated in v0.4.13.

---

## v0.4.8: Model architecture visualizer (local server)
*2026-11-22*

**What**: a local web app that **reads the real model code** and draws each config's architecture. `tools/viz/server.py` (stdlib, 127.0.0.1 only, CPU-only) builds `FusionDance` from `configs/*.json` via `gogeta/model.py` on every request, so new configs and code changes show up automatically. Start it with `bash tools/launch_viz.sh start` (WSL) or `tools\launch_viz.ps1 start -Open` (Windows) → http://127.0.0.1:7870. Guide: `tools/viz/README.md`.
- **Shapes carry meaning:**
  - full attention = a filled triangle (n² scores);
  - sliding window = a diagonal band;
  - Gated DeltaNet = a fixed-size square memory tile;
  - MLP = an expand/contract trapezoid;
  - block height ∝ params;
  - the core is drawn once with a "× r" loop arrow, plus an unroll toggle with shared-weight chips;
  - ternary = a −1/0/+1 texture with a 1.58-bit badge.
- **Controls:** a context slider 1k → 32k with live memory/compute growth, a loops slider 1–12, side-by-side compare, tooltips, a plain-English legend, and a DBZ scouter "power level" panel.

**Results**: no training. API numbers match the model code: E0 13,767,552 total; E1 14,237,796 / 91.0 MFLOPs per token; E2 non-embedding 9,460,760, depth 14 at r=4. The page structure was checked in the browser (7 configs, sliders, 21 diagrams, no console errors, no overflow at 1,156 px and 375 px).
**Findings**: the palette check (dataviz validator, dark surface) passes for blue/orange/aqua; yellow vs orange fails all-pairs, but that pair (MLP vs GDN) is always separated by shape, label and position. E10 (`hf_llama`) is drawn as the equivalent 6 × (attention + SwiGLU) stack, with a note.
**Not verified**: the visual look (the browser pane was too small to screenshot); touch tooltips on a real phone.
**Next**: unchanged (cleanup Phase 1 → E10/K7/K8 registered → Phase 2 campaign incl. E11).

---

## v0.4.7: E11 registered: negative-eigenvalue GDN for language modelling
*2026-11-22*

**What**: registers **E11**: E1's layout with GDN's write strength β ∈ (0, 2) (negative eigenvalues). It's one change vs E1, with the same params and FLOPs. **Not run yet.**
- `gogeta/model.py`: new `ModelConfig.gdn_neg_eigval` (default False; the default path is unchanged). When it's on, raw β logits go to fla with `use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True`.
- `tests/test_model.py`: 12/12 pass, including two new tests: causality, and that the computation differs from the default with identical weights (max logit diff 0.25). The config loop now skips non-FusionDance configs.
- `configs/e11_negeig.json`, `experiments/launch/E11.sh`, card, plain-English page; the registry entry with its **prediction** (committed here, before any run).
- `experiments/INDEX.md` generated (from committed sources only); `pyproject.toml` / README badge → 0.4.7.

### E11: Negative-eigenvalue Gated DeltaNet for language modelling
- **Question:** does β ∈ (0, 2) cost or help LM, given that SQ6 shows it solves state tracking in one pass?
- **Result:** not run. It runs at E1's tuned LR from K7, with 3 seeds, in the cleanup campaign right after E1's seeds.
- **Verdict vs prediction:** the prediction is no LM cost, within ±1% of E1's 3-seed mean. Criteria: free / helps / hurts by max(1%, 2× pooled seed std).
- **What it means:** if it's free or helps, neg-eig GDN becomes the default mixer and the loop question is re-asked with it.
- **Card:** experiments/cards/E11.md

---

## v0.4.6: SQ6 (+SQ2): rotation vs loops vs negative eigenvalues on parity
*2026-11-22*

**What**: side quest SQ6 (Mamba-3-style rotating state) with SQ2 (negative-eigenvalue GDN) folded in; prediction committed in `4a3e361` before it ran. This version also commits the **ledger** (`claims.md`: verdicts for C12, C14–C18, C20, C22, new C43–C44; the pending-review sections C30–C42, C50–C69 and C80–C84), the **side-quest log** (SQ outcomes; pending-review SQ14–SQ21, SQ30–SQ41, SQ50–SQ59) and the **lab notebook** entries for all checks and reviews.

### SQ6: Rotation vs loops vs negative eigenvalues on parity
- **Question:** can a single-pass layer track state (parity) as well as or better than looping? (C22; SQ2)
- **Result** (2 seeds; positions 32–63 / 64–127 at T=128):
  - GDN 1 pass 61.7 / 50.3;
  - looped ×4 97.9 / 61.2;
  - RotSSM 84.0 / 76.5;
  - RotSSM, θ=0: 50.2 / 50.1;
  - **GDN with negative eigenvalues: 100.0 / 99.99** (`results/sq_06.json`).
- **Verdict vs prediction:**
  - C22 REFUTED at our scale (84 < 90), although rotation beats looping on length generalisation;
  - rotation does not beat looping in-length;
  - **SQ2 strongly confirmed (+38 points)**.
- **What it means:** a one-flag change to GDN (β ∈ (0, 2)) solves state tracking in one pass and length-generalises, where looping doesn't generalise. This challenges what FusionDance's loop is for, and motivates E11.
- **Card:** experiments/cards/SQ6.md

**Findings**: single-pass GDN parity accuracy has high variance (K5 81.5% vs SQ6 73.3% / 50.1% with identical setup), so ≥ 2 seeds are needed (C44).

---

## v0.4.5: K6: Does E2's loop converge?
*2026-11-22*

**What**: check K6 (Does E2's loop converge?): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K6: Does E2's loop converge?
- **Question:** Does each extra pass refine E2's state? (C20)
- **Result:** Relative change k1 7.20 → k4 0.283 → k12 **0.085**; ‖h_k‖/‖h₀‖ 7.4 → 58.9; loss after the coda best at k=4 (0.722) (`results/K6.json`, inference on the E2 checkpoint).
- **Verdict vs prediction:** Hit (not convergent): C20 REFUTED for E2. The reviews read it as constant-size steps aligned with h, so the norm grows linearly.
- **What it means:** E2's loop inflates its state rather than refining it; loop fixes must bound the state (SQ14–SQ18).
- **Card:** experiments/cards/K6.md

---

## v0.4.4: K5: State tracking: single pass vs looped
*2026-11-22*

**What**: check K5 (State tracking: single pass vs looped): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K5: State tracking: single pass vs looped
- **Question:** Are linear mixers weak at state tracking, and does looping add that depth? (C18)
- **Result:** Running parity, positions 32–63: GDN 1 pass **81.5%**, looped ×4 **97.0%**, 4 unshared GDN 99.8%, attention 50.2% (`results/K5.json`).
- **Verdict vs prediction:** Hit: weakness supported (< 90%); C18 VERIFIED (+15.4 points). Looped ≈ unshared depth (−2.9 points).
- **What it means:** Looping buys state-tracking depth on parity, but so does plain depth. SQ6 then found a cheaper fix.
- **Card:** experiments/cards/K5.md

---

## v0.4.3: K4: Ternary gap vs model size
*2026-11-22*

**What**: check K4 (Ternary gap vs model size): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K4: Ternary gap vs model size
- **Question:** Does the ternary quality gap shrink with size at our scale? (C17)
- **Result:** Mean relative gap **15.32% / 13.76% / 10.55%** at d = 64/128/256 (2 seeds, decreasing within each seed; `results/K4.json`).
- **Verdict vs prediction:** Hit on the trend; the gaps are larger than predicted (12/8/5%). Local analogue VERIFIED, so C17 is PARTIAL (3B parity untestable here).
- **What it means:** Ternary costs ~10%+ at tiny sizes; E3 should run at E1's tuned LR and be judged against this trend.
- **Card:** experiments/cards/K4.md

---

## v0.4.2: K3: MQAR associative recall (invalid)
*2026-11-22*

**What**: check K3 (MQAR associative recall (invalid)): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K3: MQAR associative recall (invalid)
- **Question:** Does GDN recall better than plain linear attention? (C16)
- **Result:** attn 8.6%, gdn 3.3%, linear attention 4.9%, swa(32) 0.4% (2 seeds, `results/K3.json`).
- **Verdict vs prediction:** **Inconclusive**: the positive control (attention ≥ 90%) failed, so the task was too hard or undertrained. C16 stays UNVERIFIED.
- **What it means:** The check needs a redesign (fewer pairs, more steps, or Zoology's recipe) before any recall claim.
- **Card:** experiments/cards/K3.md

---

## v0.4.1: K2: BitLinear speed vs nn.Linear
*2026-11-22*

**What**: check K2 (BitLinear speed vs nn.Linear): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K2: BitLinear speed vs nn.Linear
- **Question:** Does ternary make compute cheaper in our code? (C14)
- **Result:** BitLinear forward is **5.58× / 4.06× slower** (384→1024 / 384→2048); fwd+bwd 1.92× / 1.59× (`results/K2.json`).
- **Verdict vs prediction:** Hit on direction; the slowdown slightly exceeds the predicted 1.5–5×. C14 confirmed WRONG FOR OUR CODE.
- **What it means:** Ternary is a memory win only until we have real ternary kernels; never claim speed from it.
- **Card:** experiments/cards/K2.md

---

## v0.4.0: K1: Memory and time vs context length
*2026-11-22*

**What**: check K1 (Memory and time vs context length): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K1: Memory and time vs context length
- **Question:** Do our layers scale as claimed? (C12 SWA O(n·w); C15 GDN fixed memory)
- **Result:** Exponents over n = 1k..16k: attn time 1.26 / mem 0.98; **SWA 1.74 / 1.95** (2.9 GB at 16k); **GDN 0.66 / 0.98**. GDN streaming split at 2048: max diff **0.0**, state [1,6,64,64] at every n (`results/K1.json`, commit `ea3aa5b`).
- **Verdict vs prediction:** Hit: SWA ≈ quadratic, as predicted; GDN linear, as predicted. C12 confirmed WRONG FOR OUR CODE; **C15 VERIFIED** (kernel level).
- **What it means:** GDN's constant-memory streaming is real on this GPU; our sliding window needs a block-sparse kernel (SQ3) before any long-context claim.
- **Card:** experiments/cards/K1.md

---

## v0.3.6: Research reviews, experiment cards + index, process rules
*2026-11-22*

**What**
- **Rules (AGENTS.md):**
  - §0b side quests;
  - §0c status updates every 10 minutes;
  - §5 compute budget (1 GPU job + ≤ 8 CPU jobs, 85 °C cool-down), launcher usage, a TRL venv and sandbox rule;
  - §6 one card per experiment plus a generated `experiments/INDEX.md`;
  - §1 **every experiment of any kind is saved, version-tagged and gets its own devlog block**, with the E-series reserved for the main thesis;
  - §1 **no WIP commits: every commit is a tagged version**.
- **Experiment cards** (`experiments/cards/`, from `_TEMPLATE.md`) for E0–E3, and `tools/build_experiment_index.py` (also run by `build_site_data.py`; supports `--registry` / `--tracked-only`).
- **Research:**
  - `notes/research-looped.md`: 12 looped-transformer papers, 2025–26;
  - `notes/review-competitors.md`: fact-check of a pasted analysis on MixerLoop, Auron and TR-Mamba2-Attn;
  - `notes/smollm2-reference.md`: SmolLM2-135M downloaded (rev `93efa2f`, sha256 matches the Hub, Apache-2.0) as an external sanity reference;
  - refs: `10-qwen` (Qwen3.8-Next architecture, FlashNext-RL, Qwen3) and `11-looped-2026` (12 papers).

**Results**: no training.
**Findings**:
- "Inkling 2" is Thinking Machines' Inkling 2 Small: a mixture-of-experts transformer, not looped.
- There's no Qwen4 paper on arXiv; Qwen3.8-Flash-Next is the official Qwen4 preview.
- LT2 (2605.20670) already loops GDN hybrids, but no one combines loops with ternary weights.
- The pasted analysis overstated MixerLoop (41.5% of the gain at 110M, not "much").
- At matched FLOPs, loops lose on LM loss in every paper that checks.

**Notes**: the ledger and side-quest rows proposed by these reviews are committed with v0.4.6. The site isn't regenerated in this version, because its sources are mid-edit by the cleanup agent.
**History**: unpushed untagged WIP commits made before the no-WIP rule: `f1b4ebd` (K1–K6 predictions), `4a3e361` (SQ6 prediction, compute rules), `21fb049` (launch scripts), `ea3aa5b` (TRL scaffold). Check results cite their hashes as provenance, so they stay.
**Next**: one tagged version per check (v0.4.0–v0.4.6), then E11 registered.

---

## v0.3.5: Trust but verify: a local check for every claim
*2026-11-22*

**What**
- **New AGENTS.md §0 "Trust but verify"**, binding on everything else. Every claim we rely on gets the smallest local experiment that could prove it wrong (≤ ~15 min, tiny models, a seeded script in `experiments/checks/`, its pass criterion written first). Paper results motivate experiments but are never adopted as facts. Nothing is called "measured" without on-disk evidence.
- **Claims ledger** `experiments/claims.md`: 20 claims from the README, site, primer and write-ups, each with source, status and evidence, plus six proposed checks (K1–K6).

**Results**: no training. Audit outcome: 5 verified (incl. 1 computed), 6 partial (mostly single seed), 6 unverified paper/theory claims, and **3 that are wrong as currently stated**:
- the "~10× smaller" ternary claim (computed: 7.8× body, 3.1× whole);
- sliding-window "O(n·w)": our dense-mask kernel is O(n²);
- "ternary makes compute cheap": it's simulated, so no speedup.

Plus 1 refuted assumption: fla's recurrent kernel NaNs on this GPU.
**Findings**: on-disk check confirmed **E3 has never been run**; only E0/E1/E2 (seed 0) have results.
**Next**: run checks K1–K6 (~50 min GPU total, all tiny) and fix the three wrong claims everywhere they appear.

---

## v0.3.4: E3 designed: ternary weights on the E1 stack
*2026-11-22*

**What**
- **E3 fully designed** (it was a placeholder): `configs/e3_ternary.json` is E1's exact layout and recipe with `ternary: true`, so the only change is weight precision. Ternary × loops moves to E4.
- **Prediction committed before any run:** val loss 0.605–0.645 (+3.5% to +10% vs E1), most likely ~0.62; stable; 10–25% slower per token (simulated quantisation). Decision rules: ≤3% nearly free, 3–10% expected, >10% → try BitNet's full recipe.
- **E3 page rewritten:** plain-English summary, the one-change design table, how ternary works step by step, the memory payoff, what the BitNet papers found, how it will be judged, risks, and status.

**Results**: no training. Measured on CPU: **33** projection layers become ternary, covering **11.06M of 11.09M** non-embedding params. Weights shrink from 28.5 MB to **9.1 MB** (2-bit packed); the model body from 22.2 MB to **2.8 MB (7.8×)**. From the papers: BitNet b1.58 only matches full precision from 3B params (700M: perplexity +4.4%).
**Findings**: after ternarisation the **word table (6.3 MB) is ~2/3 of the model**, so at our size the next memory win is the embeddings, not the layers.
**Next**: unchanged: training stays paused; when it resumes: E9 → seeds → E3.

---

## v0.3.3: Live on GitHub Pages; version tags published
*2026-11-22*

**What**
- **GitHub Pages enabled**: legacy build from `main` / `docs`, HTTPS enforced. Live at **https://monstercameron.github.io/Gogeta-0/** and set as the repo's homepage link.
- **All 7 version tags pushed** (`v0.1.0` → `v0.3.2`). The repo now has `push.followTags = true`, so future pushes carry their annotated tags automatically.
- README: "Project site" and "Start here" links point at the live site; new tags/site badges; the publishing section is now "Project site". `pyproject.toml` Homepage → the site, plus a Repository URL. AGENTS.md documents the remote, follow-tags and the post-push Pages check.

**Results**: first Pages build `built` in ~20 s. All key paths return HTTP 200 (landing, devlog, primer, E0/E2/E9 pages, CSS/JS, site data, banner). In the browser: version badge v0.3.2, 10 roadmap links resolve, E2's 2 charts render, no console errors.
**Next**: unchanged. Training stays paused until the human asks; then E9 and the seed replications (detached).

---

## v0.3.2: Repo cleanup for GitHub: README, MIT license, packaging
*2026-11-22*

**What**
- **README rewritten** with branding: an SVG banner (`docs/assets/banner.svg`), badges, the pitch, a Mermaid architecture diagram, goals, a results table with a plain-English summary, quick start, repo layout, how the research is run, publishing instructions, credits (including the dataset's ODC-By attribution) and the fan-homage disclaimer.
- **MIT `LICENSE`** (chosen by the human). **`pyproject.toml`**: `pip install -e .` installs the `gogeta` package with its tokenizer asset (torch stays a separate CUDA-index install).
- **Organised:** `sketch.md` → `notes/sketch.md`, next to the lab notebook. References updated; older devlog entries keep their original wording as history.
- **`.gitignore` hardened:** raw/parquet data, tracker DBs, logs, partial downloads, packaging/test caches, keys, and `.claude/*` except the shared `launch.json`. No tracked file is caught by the new rules.
- **Scrubbed** the one machine-specific path in AGENTS.md. AGENTS.md documents the new files and adds a rule to bump `pyproject.toml` + the README badge with every version.

**Results**: no training. Checks: every `.py` compiles; the package installs and imports in WSL and finds its tokenizer; full-history scan found **no tokens/keys/passwords**; the largest committed file is the 536 KB tokenizer; the Mermaid diagram renders.
**Findings**: the Windows username still appears in **2 historical diff lines** (an old AGENTS.md path). It's harmless, and removing it would mean rewriting history and moving tags, which AGENTS.md forbids, so it stays.
**Next**: push to GitHub and enable Pages (`main` / `docs`) when the human is ready; training stays paused until asked.

---

## v0.3.1: Plain-English experiment pages and a newcomer primer
*2026-11-22*

**What**
- **One page per experiment** on the site (`docs/experiments/E0–E9.html`), generated from `experiments/writeups/E<n>.md`. Each opens with **"What it means, in plain English"**, then the setup, charts, a breakdown of the findings, the prediction scorecard, caveats and next steps. Planned experiments get a short "what it will test" page.
- **"Start here" primer** (`docs/learn.html` from `site-src/learn.md`): how a model learns, every number on the site decoded (loss, perplexity, bits/char, FLOPs, seeds), the three tricks as analogies, and what we've found so far.
- **Charts:** interactive learning curves and loss-vs-loops (hover crosshair/tooltips, legend, direct labels, "show the numbers" tables) from `experiments/curves/E<n>.json`. The palette was validated for colour-blind safety on the site's dark surfaces.
- **AGENTS.md rule:** every finished experiment gets its write-up, curve file and generated page before its version is cut.
- Devlog cards and roadmap items link to the experiment pages; "Start here" is in every nav.

**Results**: no training (paused by request). Yardsticks measured on CPU for the explainers: the models compress held-out Python to ~3% of its size vs 15.4% for `xz -9` and 25.1% for `gzip -9`; E1 reached E0's final quality at 84% of the training budget.
**Findings / surprises**
- ⚠ **The replication queue was killed** when the previous agent session ended (E9 at step 600, E2-s1 at step 200). There are no new results, E9 is back to `planned`, and the replication predictions stay open. Future queues should run detached (`nohup`/`setsid`).
- **Correction to v0.3.0:** E2's scorecard is **5 of 7** predicted claims, not "5 of 8".
**Next**: when training resumes: E9, then seeds 1–2 (detached), then the functional-correctness eval.

---

## v0.3.0: First training arcs: E0 dense, E1 hybrid, E2 looped (initial arch)
*2026-11-22*

**What**
- **Data** (`tools/prepare_data.py`): OpenCoder `opc-annealing-corpus` (ODC-By), algorithmic + synthetic snippets, **Python only**, exact-deduped → 991k docs → 8k byte-level BPE → **262.4M train / 1.06M val tokens** (3.72 chars/token). Tokenizer is versioned in `gogeta/assets/`.
- **Model** (`gogeta/model.py`): `FusionDance`, prelude → core × r → coda with `attn` / `swa` / `gdn` blocks (GDN on fla's chunk kernel, verified against a reference), random-r training with truncated BPTT, BitLinear switch. 10/10 correctness tests.
- **Training** (`gogeta/train.py`, `tools/run_queue.sh`): compiled bf16, AdamW, cosine schedule, tracked runs with eval, samples and checkpoints.
- **Notes:** `notes/LABNOTEBOOK.md` (timestamped evidence) plus a new AGENTS.md rule to keep it.
- **Site:** E0–E2 results and verdicts show on the devlog page. The power-level chart's axis now starts near the best result and labels the % gaps (zero-based bars hid a 6% difference). E9 registered.

**Results**: 200M tokens each, seed 0, commit `0dc7491`, full val split

| | Non-emb params | Val loss | Δ vs E0 | Train FLOPs | Wall |
|---|---|---|---|---|---|
| E0 dense (6× attn) | 10.62M | 0.6063 | n/a | 1.9e16 | 23 min |
| **E1 hybrid** (1 attn + 3 GDN + 2 SWA) | 11.09M | **0.5846** | **−3.6%** | 1.8e16 | 39 min |
| E2 looped, r=4 (the initial arch) | 9.46M | 0.6186 | +2.0% | 3.9e16 | 80 min |

E2 loss vs eval loops: r1 0.669 · r2 0.627 · **r4 0.619** · r6 0.624 · r8 0.633 · r12 0.654.

**Findings**
- The **hybrid beats dense** at every checkpoint with only one full-attention layer. It's the first positive result for the linear-time-mixer fighter.
- **Looping doesn't pay yet:** E2 is 5.8% worse than E1 at 2.1× the FLOPs.
- **The loop isn't convergent:** loss bottoms out at r=4 (the mean training r) and degrades when looping more, so it learned "about 4 passes", not "each pass improves". That's a measured motivation for the Rival Loop's per-loop supervision.
- Prediction scorecard: E0 loss range ✗ (data far easier than assumed), E1 direction ✓ / size ✗, E2 5 of 8 claims ✓. Details in the lab notebook.
- ⚠ fla's recurrent GDN kernel returns NaN on this GPU (training kernel fine). SWA still uses a dense mask (O(n²) kernel). Triton needs `python3.12-dev`.

**Decisions**: E1's layout is the current best mixer stack. Added **E9** (unlooped control) to separate looping from depth. Single seed so far: gaps aren't claims until 3 seeds.

**Next**: E9 control; 2 more seeds for E0–E2; a functional-correctness eval (execute generated code against held-out asserts); per-loop diagnostics (‖h‖, per-loop loss) before E3 ternary.

---

## v0.2.0: Local experiment tracking (Trackio) and research hygiene
*2026-11-22*

**What**
- WSL training environment: `~/.venvs/gogeta` (Python 3.12.3) with **Trackio 0.45.1**, pinned in `requirements.txt`.
- `gogeta/tracking.py`: a tracker-agnostic `Run` (`GOGETA_TRACKER=trackio|wandb|none`). It logs git commit / describe / dirty flag, config and seed, writes `meta.json` and `summary.json` per run, records crashes as `failed`, and has a `require_clean` guard for runs that feed tagged versions.
- `tools/record_result.py` (run summary → registry), `tools/dashboard.py` (local dashboard, localhost only, sharing off), `tools/smoke_tracking.py`.
- Every experiment in the registry gets a `prediction` field, shown on the site's devlog page. It's written before the run.
- AGENTS.md §6: tracking and hygiene rules (predictions first, ≥3 seeds for headline claims, tokens/FLOPs x-axes).

**Results**: smoke run `smoke-20261122-115609-s0` logged 200 steps to `~/fusiondance-data/trackio`. The dashboard renders from Windows at 127.0.0.1:7861, and every network request stayed on 127.0.0.1. `require_clean` refuses on a dirty tree.
**Decisions**: Trackio over W&B local: pip-only (no Docker or license; the W&B server image's ARM64 support is unconfirmed), local SQLite storage, and a wandb-compatible API so switching later is one variable. Checked: no analytics/telemetry code in the package.
**Findings**: Trackio reserves config keys starting with `_`, so provenance uses flat `meta_*` keys. From WSL, this repo needs `safe.directory` (WSL git config) and `core.filemode=false` (repo config); otherwise every file looks changed.
**Next**: PyTorch (CUDA, aarch64) in the venv and check `flash-linear-attention`; write E0's prediction; build the E0 baseline.

---

## v0.1.2: Project site (GitHub Pages) and experiment registry
*2026-11-22*

**What**
- `docs/`: a static GitHub Pages site, DBZ-flavoured but scientific. The landing page covers the goals, the three fighters, the FusionDance diagram, the three budgets, the big ideas and the training-arc roadmap. The devlog page has the experiment log with status filters, a val-loss "power level" chart, and the version timeline.
- `experiments/experiments.json`: a registry of the nine planned arcs (E0–E8), from the baseline to self-speculative decoding.
- `tools/build_site_data.py`: generates `docs/data/site-data.js` from `DEVLOG.md` plus the registry, so the site can't drift from the devlog.
- `.gitignore`: `data/` → `/data/` (root only), because it was also hiding `docs/data/`.

**Results**: n/a. Site checked in the browser at desktop and phone widths: no console errors, no horizontal scroll.
**Decisions**: The site reads generated JS data (not fetched JSON), so it also works when opened from disk. Experiment status is tracked in the registry, not in prose.
**Next**: push to GitHub and enable Pages (main, `/docs`) when ready; then WSL setup and E0.

---

## v0.1.1: Naming: repo Gogeta-0, architecture FusionDance
*2026-11-22*

**What**: renamed the repo/project to **Gogeta-0**. **FusionDance** is now the name of the architecture (linear-time mixers + loops + ternary); Gogeta-0 is its first model. Updated README, DEVLOG, AGENTS, sketch and the refs index.
**Results**: n/a
**Decisions**: Repo = Gogeta-0, architecture = FusionDance.
**Next**: unchanged from v0.1.0.

---

## v0.1.0: Project kickoff: design, references, ground rules
*2026-11-22*

**What**
- `sketch.md`: the full design scratchpad: core thesis, v0 architecture, sub-quadratic parts bin, speculative decoding, the Rival Loop, data & evals, local hardware feasibility, decisions log.
- `refs/`: 55 papers across 9 topics, with an index (`refs/README.md`) and a re-runnable fetch script. PDFs are git-ignored.
- `README.md` (pitch), `AGENTS.md` (repo rules), this devlog.

**Decisions**
- Target is a **regular causal LLM**: loops add per-token latent depth.
- Device: the laptop. No hard memory target: we explore the design space with a configurable sweep harness.
- v0 layout: prelude (SWA) → core × r (GDN, GDN, SWA) → coda (one full-attention layer, run once). No quadratic layer inside the loop.
- Intermediate loops decode through a cheap exit head, never the coda.

**Findings**
- Native Windows ARM64 has CPU-only PyTorch. **WSL Ubuntu sees the GPU** and Linux aarch64 CUDA wheels exist, so we train in WSL.
- Closest prior art: *Tiny Recursive Reasoning with Mamba-2 Attention Hybrid* (arXiv 2602.12078). Must read before building.

**Next**
- Read the closest prior art, then lock the novelty story.
- Set up the WSL environment (PyTorch CUDA, check `flash-linear-attention` on aarch64 + Blackwell).
- Smoke-test harness: TinyStories, ~10M params, full precision, loop vs no-loop.

---

<!-- ENTRY TEMPLATE: copy above the previous entry

## vX.Y.Z: <short title>
*YYYY-MM-DD*

**What**: what changed (code, configs, docs)
**Results**: numbers + the config/seed that produced them (or "n/a")
**Decisions**: what we chose and why
**Findings / surprises**: incl. failures
**Next**: the next 1–3 steps

One block per experiment recorded in this version (E-series, checks K, side quests SQ, sweeps, evals…):

### <ID>: <title>
- **Question:** what it tested (and which claim/side quest)
- **Result:** the numbers, with run id / result file and commit
- **Verdict vs prediction:** hit / miss / inconclusive, against the committed criterion
- **What it means:** one or two sentences for the project
- **Card:** experiments/cards/<ID>.md

-->
