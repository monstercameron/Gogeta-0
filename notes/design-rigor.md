# Design for scientific rigor (A3): HPO protocol, matched arms, factorial, decomposition, roadmap

Written 2026-11-22. Scope: the audit's four findings (unfair HPO, incomplete matched controls, no compute- or parameter-matched comparisons, no factorial interaction test). Nothing here touched `gogeta/`, `configs/`, `tools/` or `tests/`. Everything is staged under `experiments/staged-configs/` (28 files, move into `configs/` between campaigns) and proposed under `experiments/design/`.

Regenerate and re-verify everything (CPU, about 1 minute): `build_staged.py`, then `power_cost.py`, then `make_proposals.py` (all in `experiments/design/`, run with `~/.venvs/gogeta/bin/python` under `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2`). IDs E20-E59 are reserved for A3 so they cannot collide with E12 (Q22's proposed neg-eig loop); renumber on integration if needed.

## 0. Dedup (K9) and the metric

K9 failed: 41.7% of val docs are near-duplicates of train and E0's bits/char moves +25.8% without them. Decisions:

1. **Primary metric of every comparison below: doc-level bits/char on the clean val split.** Every checkpoint is also scored on the near-dup docs; the **memorisation gap** g = bpc(near-dup) - bpc(clean) is reported beside it and analysed with the same contrasts (claim C169). Recall-capable mixers (full attention) may win on memorised repeats, which confounds GDN vs attention (C5, C163), so a GDN main effect that holds on clean bpc but not on the gap (or vice versa) is read as such.
2. **Which train set.**

| | v1 (code-py-v1, as Phase 2b) | v2 (deduplicated, A1) |
|---|---|---|
| Comparable with every run so far | yes | no: absolute losses and possibly rankings move |
| Memorisation confound in training dynamics | present (scored out on val, not in train) | removed |
| Cost | none | bridge runs |

   **Decision:** the factorial (E20-E27, E35, E36) and the decomposition (E40-E48) are configured on **code-py-v2**, since they are new and their conclusions are about mechanism. The matched arms (E30-E34) and the horizon check stay on **v1**, because they exist to sit next to Phase 2b's E10/E1/E2/E9; they are scored on the clean val split post hoc, which removes the val-leakage half of the problem. A **bridge** (E52 = E10, E53 = E1, seed 0, 200M, v2, Phase 2b LRs) re-anchors the two worlds: **64 GPU-min** (E10 24 + E1 39; plus 19 conditional minutes to re-check the LRs on v2 at 20M). The full alternative, re-running Phase 2b on v2 (13 runs, about 7 GPU-h), is SQ129, justified only if the bridge shows a ranking change (C168). If v2 has fewer than 200M unique tokens, the bridge repeats data: A1 must say how many epochs, and the bridge reports it.
3. All staged v2 configs name `"dataset": "code-py-v2"`; they will not launch until A1's dataset exists (the launcher checks the dataset).

## 1. Fair HPO protocol

Problem: K7 tuned only E10 (4 LR points) and E1 (5 points), at 20M tokens, one seed; nothing else (E2, E9, E3, E11, every future arm) was tuned, and E1's LR (6e-3) differs from E2/E9's old 3e-3.

**Protocol P1** (details and decision rules in `experiments/design/proposed-predictions.md`, section H):

1. **Same grid shape.** The K7 grid {1e-3, 2e-3, 3e-3, 6e-3, 1.2e-2}, 20M tokens, seed 0. Equal budget means 5 points per model: E10 lacks 1.2e-2 (**H0**: 1 run, 3 min). The same one-step extension rule for an edge winner. New arms (the factorial cells, decomposition arms) use the 3-point sub-grid {3e-3, 6e-3, 1.2e-2} plus the extension rule, which is the cheapest equal rule since 1e-3 and 2e-3 lost by 17-20% and 4-5% in K7.
2. **Same horizon rule.** A 20M winner is provisional; a claim at 200M needs a 3-point bracket at 200M, the same count for every model compared.
3. **Seeds.** The selection run is never reused as evidence (winner's curse). Replicate the top-2 LRs at a second seed to estimate the proxy's own noise (**H1**: 4 runs, 13 min).
4. **Report both numbers**: each model at its own best LR and at the shared LR.

**Long-horizon check (stage A, 88 GPU-min).** Phase 2b already gives E10@3e-3 and E1@6e-3 at 200M (3 seeds). Add one seed each of **E1@3e-3, E10@6e-3, E10@2e-3** at 200M. The first two are the coordinator-suggested cheapest additions; I added E10@2e-3 because K7's caveat and scaling folklore both say the optimum moves down with horizon, and E10@6e-3 only tests the opposite direction. (A v0.3.0 E1@3e-3 200M run exists, 0.5846, but under older code and a GPU-replay difference of unknown origin, K7-parity; re-run it.) Costs: E1 39 + E10 24 + E10 24 = 88 min.
- **Prediction:** optima hold. E10: 3e-3 best of the three (P 60%; 2e-3 P 30%). E1: 6e-3 beats 3e-3 by 0.5-2.5% (P 60%).
- **Decision rule:** with s = 0.004, a single run minus a 3-seed mean has sd 0.0046, so T = 2 x 0.0046 = 0.0092 nats (1.6% at L = 0.58). The optimum **transfers** iff no neighbour beats the Phase 2b mean by more than T. If not, bracket beyond the winner, repeat that model's 3 seeds at the new LR, and restate the C5 gap as a range (shared LR, own best LR). Stage B (E1@1.2e-2 at 200M, 39 min) only if E1@6e-3 beats 3e-3 by more than T.

**Warmup and schedule: no sweep.** Reasons: (i) at 200M, 300 steps are 4.9% of 6104 steps, standard; the schedule (cosine to 0.1) is shared, so it cannot differentially favour a model unless interactions with LR exist; (ii) sweeping warmup or schedule per model adds a dimension to each model's tuning budget and multiplies cost (3x) for a second-order effect, and "equal budget" then needs equal grids in both axes; (iii) the one real risk is the 20M proxy: 300 steps is **49%** of its 611 steps, so the proxy spends half its steps warming up, which may shift its LR ranking relative to 200M (direction unknown). That risk is a **question about the proxy, not a tuning axis**, so one registered sensitivity pair (**E50, E51**, warmup 30, 4 runs, 13 min: C167) answers it. If either winner moves, treat the K7 LRs as provisional and rely on the 200M bracket.

## 2. Matched-compute and matched-parameter arms

Counts come from `param_counts()` and `flops_per_token(1024)` (training, 3x forward; the formula credits SWA with O(window) and ternary with no saving). Every config below was **rebuilt from its written JSON file on CPU** and the numbers are what the script printed (`experiments/design/staged-table.md`).

| Arm | Config | Layout | non-emb params | train FLOPs/tok | vs E1 | vs E2 | est. 200M wall |
|---|---|---|---|---|---|---|---|
| E10 (control) | configs/e10 | 6 attn, mlp 1024 | 10,621,824 | 96,758,784 | 1.063 | 0.502 | 24 min |
| E1 | configs/e1 | swa,gdn,gdn,swa,gdn,attn | 11,092,068 | 91,027,800 | 1.000 | 0.472 | 39 min |
| E2 | configs/e2 | [swa] [gdn,gdn,swa] x4 [attn] | 9,460,760 | 192,793,152 | 2.118 | 1.000 | 82 min |
| E9 | configs/e9 | swa,gdn,gdn,swa,attn | 9,165,080 | 79,170,960 | 0.870 | 0.411 | 34 min |
| **E30** dense, E1's params | e30_dense_parammatch | 6 attn, mlp 1096 | **11,119,488** (+0.24%) | 99,744,768 | 1.096 | 0.517 | 25 min |
| **E31** dense, E1's FLOPs | e31_dense_flopmatch | 6 attn, mlp 888 | 9,681,792 | **91,118,592** (+0.10%) | 1.001 | 0.473 | 23 min |
| **E32** unshared, E2's FLOPs | e32_unshared14_e2flops | E2's 14 executed mixers, no loop | 26,037,728 | **185,715,264** (-3.7%) | 2.040 | 0.963 | 79 min |
| **E33** 12 unshared blocks | e33_unshared12_e2flops | 12 blocks, mlp 1392 | 27,427,732 | **192,580,728** (-0.1%) | 2.116 | 0.999 | 82 min |
| **E34** E9 widened to E2's FLOPs | e34_e9_wide_e2flops | d=576, 9 heads, mlp 2168 | 26,071,652 | **192,700,632** (-0.05%) | 2.117 | 1.000 | 82 min |

Notes:
- **E30/E31 bracket E1** from both sides on the two axes. E31 is low priority: E1 has *fewer* FLOPs than E0/E10 and still wins, so a dense model with E1's FLOPs has less compute and cannot close the gap by construction (the review rejected it as stated, for this reason). It is cheap (23 min) and the audit asks for it.
- **E2 vs E9 at matched FLOPs** has no single answer: E9 (79M) needs 2.4x more compute to meet E2. I stage three ways of giving the unlooped model E2's FLOPs: depth (E32: 14 unshared blocks, the review's "12 unshared core blocks + prelude + coda"), a literal 12-block total with a wider MLP (E33), and width (E34). E34 is not LR-retuned at d=576 (flagged in its prediction).
- **Parameter allocation is explicit:** the unshared arms have 2.75-2.9x E2's unique params. That is the point of the comparison ("loops improve efficiency" is a parameter-efficiency claim; E2's value is its 36% of E32's params at what loss gap), so report loss at matched FLOPs *and* the loss-per-parameter trade-off.
- LRs in the files: dense arms 3e-3 (LR_E10), hybrid arms 6e-3 (LR_E1); they must be overridden by the K7/horizon outcome.

**Measured cost next to FLOPs** (from `~/fusiondance-data/runs/*/summary.json`; tok/s = tokens/wall incl. compile and evals; TFLOP/s = tok/s x our analytic FLOPs/token):

| Run | tokens | tok/s | TFLOP/s achieved | peak GB |
|---|---|---|---|---|
| E0 v0.3.0 | 200M | 143,083 | 13.8 | 4.75 |
| E1 v0.3.0 | 200M | 86,334 | 7.9 | 7.10 |
| E2 v0.3.0 | 200M | 41,531 | 8.0 | 12.07 |
| E10 K7 (4 LRs) | 20M | 177-186k | 17.1-18.0 | 4.39 |
| E1 K7 (5 LRs) | 20M | 108-145k | 9.9-13.2 | 7.10 |

- **FLOPs flatter the hybrid.** E1 has 0.94x E0's FLOPs but ran at 0.60x its throughput at 200M (86k vs 143k tok/s), and uses 1.5x the memory. SWA runs dense-masked (C12: time exponent 1.74), so its measured cost is O(n^2) at 1k context, and the GDN kernels are bandwidth-bound; a FLOPs-matched table therefore understates the hybrid's real cost. **Every matched-arm result must state loss vs FLOPs *and* loss vs measured GPU-minutes** (E1 vs E10 at measured tok/s is roughly 0.6x speed for a 3.6% loss gain, a different and less flattering trade-off). SQ125 re-measures after SQ72's flex-attention SWA.
- The 20M rates are higher than the 200M rates (E1 134k vs 86k); the v0.3.0 200M runs predate compile/launcher changes and the GPU is power-limited (SQ79). Refresh `TF` in `build_staged.py` from Phase 2b's 200M measurements; estimates here use the 200M rates for 200M runs and the K7 rates for 20M runs.
- Ternary cost is **unmeasured** under compile: K2 says BitLinear is slower (C14 WRONG FOR OUR CODE); estimates assume 0.75x speed (SQ76 pending). FLOPs do not credit ternary; its claim is memory/inference efficiency on hardware this repo does not have, so the ternary "practical efficiency" claim can only be supported as a memory-and-loss trade-off (SQ123), not as measured speed.

## 3. The 2x2x2 factorial (GDN x shared loops x ternary)

### Cell definitions (configs `e20`-`e27`, on code-py-v2)
Cell `abc` = (GDN, loops, ternary); id = E(20 + 4a + 2b + c). Fixed everywhere: d_model 384, 6 heads, mlp 1024, vocab 8192, window 256, seq 1024, batch 32, 20M tokens, AdamW and cosine schedule as E1, loops r ~ U{1..7} train / r=4 eval where looped.

| cell | id | mixers | unique blocks | non-emb params | train FLOPs/tok | note |
|---|---|---|---|---|---|---|
| 000 | E20 | 14 full attention, unlooped | 14 | 24,783,744 | 200,604,672 | dense control at the depth budget |
| 001 | E21 | same, ternary | 14 | 24,783,744 | 200,604,672 | |
| 010 | E22 | [attn] [attn x3] x4 [attn], re-injection | 5 | 9,147,264 | 207,682,560 | dense looped |
| 011 | E23 | same, ternary | 5 | 9,147,264 | 207,682,560 | |
| 100 | E24 | E2's 14 executed mixers, unlooped | 14 | 26,037,728 | 185,715,264 | = E32 on v2 |
| 101 | E25 | same, ternary | 14 | 26,037,728 | 185,715,264 | |
| 110 | E26 | = E2's layout | 5 | 9,460,760 | 192,793,152 | |
| 111 | E27 | E2's layout, ternary (the full fusion at tiny scale) | 5 | 9,460,760 | 192,793,152 | |

(Only E20-E27, plus E35/E36 below, are staged; all were built from the written files on CPU, so the counts are measured, not assumed.)

### Matching rule (decided and justified)
**Match depth budget = block applications (14) and therefore FLOPs, not parameters.** Looped cells apply their 5 unique blocks 14 times; unlooped cells have 14 unique blocks. FLOPs agree within 3.8% inside each GDN level (looped cells +3.5% (dense) / +3.8% (hybrid) from the re-injection layer); across the GDN level dense is +8% (E20 vs E24), reported not hidden.
- Why FLOPs and not params: the audit asks for compute-matched comparisons, and the looped model's claimed benefit is parameter efficiency, which is an *outcome* to read off, not a nuisance to remove. Matching params instead would give the unlooped cell 5 blocks and the looped cell 2.4x the FLOPs, so the loop main effect would reward extra compute.
- The cost: the loop main effect is confounded with a 2.7x difference in unique params (24.8M vs 9.1M). **Allocation is therefore explicit**: the loop main effect is "loops vs unshared depth at equal FLOPs", read together with params. The matching-rule sensitivity is **E35** (5 unshared attention blocks) and **E36** (E9's layout): params-matched unlooped partners of E22 and E26. If the sign of the full-model interaction I depends on the rule, the claim is stated for both.
- Width, token count, LR schedule, init (std scaled by effective depth 14 in both) are fixed. LR is tuned **per cell** (P1, F0), because ternary and looping change the optimal LR; a shared LR would reintroduce the unfair-HPO finding inside the factorial.

### Pre-registered analysis (`experiments/design/factorial_analysis.py`, simulation-checked)
y = log(clean-val bits/char), cell mean over seeds; seed-0 LR-selection runs excluded.
- **Primary estimand (inferential):** I = y111 - y100 - y010 - y001 + 2 y000. The full model's departure from the additive prediction of its single-ingredient effects; I < 0 is an interaction advantage, on the log-loss scale (I = -0.01 means 1%). It needs only five cells.
- **Decision (fixed):** SYNERGY if the 95% CI (Welch, per-cell variances) lies entirely below 0; ANTI-SYNERGY if entirely above 0; NO INTERACTION if the CI lies inside (-1%, +1%); otherwise INCONCLUSIVE with the MDE reported.
- **Secondary (Holm over 4, only if all 8 cells exist):** the three 2-way interactions (GDN x loops, GDN x ternary, loops x ternary; C166 is the last) and the 3-way term, classical +-1 contrasts.
- **Main effects:** descriptive only (they depend on the allocation rule above).
- **Cost-side readouts:** FLOPs, non-embedding params, weight bytes (ternary 1.58 bits + fp embeddings, SQ123), measured tok/s, and the memorisation gap g for every cell (same contrasts on g; a GDN main effect that disappears on clean bpc but persists on g would be a memorisation artefact).

### Power and cost (A2's assumption: seed sd 0.004; `experiments/design/power-cost.md`)
Var(I-hat) = sigma^2 sum(w_i^2/n_i); the optimal allocation gives cell 000 (weight 2) twice the seeds, Var = 36 sigma^2/N. Two-sided alpha 0.05, power 80%: MDE = 2.80 SE. sigma is the sd of log loss, 0.004/L (0.004 at L = 1.0 as at the 20M scale, 0.0067 at L = 0.6).

| sigma (log) | N for 1.0% | for 1.5% | for 2.0% |
|---|---|---|---|
| 0.003 | 26 | 12 | 7 |
| **0.004** | **46** (n = 8 per cell, 16 for 000) | 21 | 12 |
| 0.0067 | 127 | 57 | 32 |

Simulation (`factorial_analysis.py --simulate`): at sigma 0.004 and N = 46 the power at a true I = -1% is **0.804** and the false-positive rate 2.6% (one tail of the two-sided test); at N = 30 only 0.57; at sigma 0.0067 N = 127 gives 0.81.

**Cheapest 1%-powered version = F0 + F1 + F2 = 6.4 GPU-h** (sigma 0.004, 20M tokens, d = 384, five cells):
| stage | what | GPU-h |
|---|---|---|
| F0 | G3 LR grid x 5 primary cells, seed 0, +0.5 extension run each | 1.75 |
| F1 | pilot: seeds 1, 2 for the five cells (measures sigma and the cross-cell correlation rho; MDE about 3%) | 1.00 |
| F2 | top up to N = 46 (n 16/8/8/8/8) | 3.65 |
| F3 | remaining cells E23, E25, E26 (tune + 3 seeds): full 8-cell table, pairwise interactions *not* powered for 1% | 2.20 |
| F4 | E35, E36 params-matched sensitivity, 3 seeds each | 0.29 |

If the pilot shows sigma 0.0067, F2 becomes N = 127 and the total 14.5 GPU-h; the pilot decides before the money is spent. Common random numbers (SQ120) could cut N by about 30% if rho = 0.3. A **200M scale-up check** (one seed per primary cell, 5.9 GPU-h, SQ124) is not powered and only checks the sign and size of I at the claim horizon. A smaller (d = 192, shallower) tiny scale was considered and rejected: the measured hardware is bandwidth-bound, so time scales roughly with depth x width x tokens, and the saving (about 1/3 per halving of d) does not justify losing comparability with the E-series and the fla head_dim 64.

Prediction (registered): I = +0.5% +- 1.5%; P(I <= -1%) = 10%, P(|I| < 1%) = 55%, P(I >= +1%) = 35%; full fusion worst of the eight cells in loss (the paper's synergy claim is the null-adjacent outcome I am least confident of, 10%).

## 4. E1 decomposition (SQ91, outside review idea B)
E1 differs from E0 in several things at once (mixer type, SWA, the conv/gate/norm inside GDN, attention placement). One factor at a time vs E1 (6 blocks, d = 384, 20M tokens, v2):

| id | arm | change vs E1 | non-emb params | FLOPs vs E1 | predicted vs E1 |
|---|---|---|---|---|---|
| E40 | GDN -> attention | [swa,attn,attn,swa,attn,attn] | 10,621,824 | 1.037 | +0.5% to +3% |
| E41 | SWA-only | [swa x5, attn] | 10,621,824 | 0.998 | +1% to +4% |
| E42 | no short conv | `gdn_conv=False` | 11,078,244 | 0.999 | +0.7% to +2.5% |
| E43 | no output gate | `gdn_gate=False` | 10,649,700 | 0.971 | +0.3% to +1.5% |
| E44 | no output norm | `gdn_out_norm=False` | 11,091,876 | 1.000 | within +-0.6% |
| E45 | bare GDN | all three off | 10,635,684 | 0.970 | +1.5% to +5% |
| E46 | attention first | [attn,gdn,gdn,swa,gdn,swa] | 11,092,068 | 1.000 | +0.8% to +3% |
| E47 | attention mid | [swa,gdn,gdn,attn,gdn,swa] | 11,092,068 | 1.000 | within +-0.7% |
| E48 | all-GDN | [gdn x5, attn] | 11,405,564 | 1.001 | within +-0.7% |

- **Needs a code change** (E42-E45): `gdn_conv`, `gdn_gate`, `gdn_out_norm` flags on `ModelConfig`/`GatedDeltaNet` (defaults True). Proposed as `experiments/staged-configs/patches/gdn_ablation_flags.patch` (apply between campaigns); a patched copy `experiments/design/model_gdn_flags.py` was used for all verification, and E1, E2 and E3 construct **bit-identically** under it with defaults (same state_dict and RNG state, checked on CPU). E40, E41, E46-E48 are config-only. E40 has exactly E10's parameter count (all attention-family blocks), so E40 vs E1 is also a clean parameter-matched mixer test.
- **Cost.** Minimal tier (the review's version): 9 arms + the E1 control, seed 0, LR 6e-3: 10 runs = **0.55 GPU-h**. It resolves effects above about 1.1%. Rigorous tier: every arm and the E1 and E10 controls get LR {3e-3, 6e-3} at seed 0 (equal budget: arms like E40 behave like E10, whose optimum is 3e-3, so one shared LR would be unfair), then 2 more seeds at the better LR: 44 runs = **2.4 GPU-h**; arm-vs-control SE = 0.33%, MDE (80%) = 0.92%.
- **Reading rules (fixed now):** an arm "matters" if it is worse than E1 by more than 2 x pooled sd (0.8% at sd 0.4%) on clean bpc with the same sign on the near-dup readout; "within noise" otherwise. If E40 does not lose by >= 2 sd the "running notes" story is not supported and the hybrid's C5 lead is a SWA/placement/memorisation effect. A winner at the grid edge gets the extension rule.

## 5. Roadmap, in the audit's priority order

GPU-hours from the measured-throughput cost model (`staged-table.md`); 1 = hybrid > attention, 2 = loops improve efficiency, 3 = ternary practical efficiency, 4 = synergy.

| # | Priority | Item (queue id) | GPU-h | Claims |
|---|---|---|---|---|
| 1 | Complete the controls | A3-01 equalise K7 grid + second seeds (H0, H1) | 0.25 | 1,2 |
| 2 | | A3-02 horizon check stage A (E1@3e-3, E10@6e-3, E10@2e-3 @200M) | 1.5 (+0.65 stage B) | 1,2 |
| 3 | | A3-11 warmup sensitivity (E50, E51) | 0.2 | 1,2 |
| 4 | | A3-03 matched arms E30, E32, E33, E34, E31 @200M | 4.9 | 1 (E30, E31), 2 (E32-34) |
| 5 | | A3-04 dedup bridge E52, E53 @200M (+0.3 conditional) | 1.1 | 1,2 |
| 6 | | A3-12 memorisation-gap readout (CPU/GPU 20 min) | 0.3 | 1 |
| 7 | Stats | Phase 2b already running (E10 x3, E1 x3, E11 x3; E9, E2 single seed); optional E2 and E9 seeds 1,2 (SQ127) | 3.8 | 2 |
| 8 | | A3-06 / SQ126 decomposition seeds (A3-05 first, minimal tier) | 0.55 then 2.4 | 1 |
| 9 | External validity | Decomposition arms that surprise, at 200M (SQ126); dense/hybrid pair on a second corpus is A4's | about 0.5 per arm | 1 |
| 10 | | 200M factorial scale-up check (SQ124) | 5.9 | 4 |
| 11 | Factorial | A3-07 F0 LR grids | 1.75 | 2,3,4 |
| 12 | | A3-08 F1 pilot | 1.0 | 4 |
| 13 | | A3-09 F2 confirm to 1% power | 3.65 | 4 |
| 14 | | A3-10 F3 + F4 (full table, params-matched partners) | 2.5 | 2,3,4 |

Subtotals: controls **8.2 GPU-h** (+0.65 conditional); stats (E2 and E9 extra seeds) 3.8; decomposition 0.55 / 2.9; factorial powered for 1% **6.4**, full table **9.2**; scale-up 5.9. A defensible minimum for an "audit-closed" paper: controls + decomposition minimal + factorial F0-F2 = **15.2 GPU-h** (about 2 nights at one run at a time); everything = about 30 GPU-h. Sequence: A3-01, A3-11 and A3-02 can start as soon as the campaign slot is free and need no new code; A3-04, A3-05 and the factorial wait for A1's dedup set, and A3-05 for the model patch.

Mapping of claims to which results settle them: **(1) hybrid > attention**: HPO (A3-01/02/11), E30, E31, bridge, E40-E48; **(2) loops improve efficiency**: E32-E34 (loss at matched FLOPs, params at 36%), E35/E36, the loop main effect and its interactions in the factorial; **(3) ternary practical efficiency**: the ternary main effect, GDN x ternary and loops x ternary terms, weight-bytes view (SQ123), and E3 at its tuned LR (queue); measured speed is a negative result until SQ76; **(4) synergy**: only the factorial's I (C165), with the dedup bridge guarding the premise.

## 6. Verification (all on CPU)
- 28 staged configs written; each was re-read from its JSON and constructed (FusionDance via the patched copy where needed; the two HF Llama configs through `gogeta.train.build_model`); param and FLOP numbers above are the printed values. No GDN forward was run (fla is GPU-only).
- Patch parity (flags default True) bit-identical for E1, E2, E3.
- The decision rule's power claim was checked by simulation (4,000 reps).
- Not verified: anything involving GPU speed (all wall estimates are model-based), code-py-v2 (does not exist), and the predictions themselves.
