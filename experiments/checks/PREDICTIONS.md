# Checks K1–K6: design, predictions and pass criteria

Written and committed **before any of these checks ran** (AGENTS.md §0). Scripts must implement these designs. Any deviation is reported, not silently changed. Results go to `experiments/checks/results/<id>.json`; verdicts then go into the [claims ledger](../claims.md).

Common: single GPU (RTX Spark N1X), bf16 autocast, seeds fixed, each check ≤ ~15 min.

## K1: memory and time vs context length (claims C12, C15)
**Design**
- (a) For one mixer of each kind (`attn`, `swa` with w=256, `gdn`; d=384, 6 heads, as in E1), forward-only, batch 1, n ∈ {1024, 2048, 4096, 8192, 16384}: time and peak extra memory. Clear SWA's mask cache before each memory measurement.
- Scaling exponent = log-log slope over n ≥ 4096.
- (b) GDN streaming: run fla's `chunk_gated_delta_rule` (fp32) on n=4096 in one go vs as two halves, carrying the final state of the first half as `initial_state` into the second. Compare outputs, and check that the state shape doesn't depend on n.

**Prediction:** SWA time and memory exponents ≈ 2 (dense mask); full attention time ≈ 2, memory ≈ 1; GDN time and memory ≈ 1; streaming max |diff| ≈ 1e-5.

**Criteria**
- **C12** "SWA costs O(n·w)" holds for our code only if the SWA time exponent ≤ 1.3 **and** the memory exponent ≤ 1.3. Otherwise it stays `WRONG FOR OUR CODE`.
- **C15** "GDN keeps fixed memory / linear time" is `VERIFIED` (kernel level) if the GDN time exponent ≤ 1.3 **and** the streaming max |diff| ≤ 1e-3 **and** the state shape is constant across n.

## K2: does BitLinear make compute cheaper in our code? (claim C14)
**Design:** `nn.Linear` vs `gogeta.model.BitLinear`, shapes 384→1024 and 384→2048, input [16, 1024, 384], bf16 autocast. Time forward-only and forward+backward, plus peak memory.

**Prediction:** BitLinear is 1.5–5× **slower** (it simulates rounding in floating point).

**Criterion:** C14 "ternary makes compute cheaper" holds for our code only if BitLinear forward time ≤ Linear forward time for both shapes. Otherwise it's `WRONG FOR OUR CODE` (confirmed).

## K3: associative recall, Gated DeltaNet vs plain linear attention (claim C16)
**Design**
- **Task:** MQAR. 64 key–value pairs (key vocab 256, value vocab 256), then the 64 keys queried again in shuffled order, each followed by its value. Sequence length 256; loss and accuracy only on the value positions in the query half.
- **Models:** 2 blocks, d=128, 2 heads, MLP 256. Mixers:
  - `attn` (positive control);
  - `gdn`;
  - plain causal linear attention (elu+1 feature map, cumsum form, written in the check);
  - `swa` with w=32 (negative control: it can't see the keys).
- **Training:** 4,000 steps, batch 64, AdamW lr 1e-3, warm-up 200, cosine. Held-out eval on 20 freshly generated batches. 2 seeds.

**Prediction:** attn ≥ 95%; gdn 60–95%; linear attention 10–40%; swa(32) ≈ chance (< 5%).

**Criteria**
- The check is **valid** only if attn ≥ 90% and swa(32) ≤ 30%. If not, it's inconclusive and must be redesigned.
- **C16** (vs plain linear attention) is `VERIFIED` if mean gdn accuracy ≥ mean linear-attention accuracy + 10 points over the 2 seeds. Mamba-2 isn't tested, so the best possible status is `PARTIAL`, worded as "vs plain linear attention".

## K4: does the ternary gap shrink with model size at our scale? (claim C17)
**Design**
- **Models:** 2 blocks `[gdn, attn]`, d ∈ {64, 128, 256} (heads = d/64, MLP = 3d), full precision vs `ternary=True`, same seed.
- **Data:** real `code-py-v1` data, sequence 512, batch 64, 600 steps (≈ 20M tokens), lr 3e-3, warm-up 100, cosine. Val loss on the first 64 val sequences of 512.
- 2 seeds. Relative gap = (ternary − fp) / fp.

**Prediction:** the gap shrinks as d grows (for example ~12% → ~8% → ~5%), and all gaps are positive.

**Criterion:** C17's local analogue ("the gap shrinks with size at our scale") is `VERIFIED` if the mean relative gap strictly decreases from d=64 to d=128 to d=256. The 3B-parity part stays `UNVERIFIED` (not testable here), so the overall status is at best `PARTIAL`.

## K5: state tracking, single pass vs looped (claim C18)
**Design**
- **Task:** running parity. Input is 64 random bits; the target at each position is the XOR of all bits so far.
- **Models:** d=128, 2 heads, MLP 256:
  - (i) 1 GDN block, one pass;
  - (ii) the same single GDN block looped ×4 (FusionDance core, fixed r=4, bptt 4);
  - (iii) 4 unshared GDN blocks (a depth control);
  - (iv) 1 attention block (reference).
- **Training:** 3,000 steps, batch 128, lr 1e-3. Accuracy on positions 32–63 of held-out sequences.

**Prediction:** (i) < 75%; (ii) better than (i), but by an uncertain amount; (iii) ≈ (ii); (iv) < 75%.

**Criteria**
- "Linear-time mixers are weak at state tracking" is supported if (i) < 90%.
- "Looping adds that depth" is `VERIFIED` if (ii) ≥ (i) + 10 points, and `REFUTED` if (ii) < (i) + 3 points.
- Comparing (ii) with (iii) says whether it's the *loop* or just *depth* (reported, no criterion).

## K6: does E2's loop converge? (claim C20)
**Design:** load the trained E2 checkpoint (`~/fusiondance-data/checkpoints/E2-20261122-132340-s0/model.pt`). On 16 validation sequences of 1,024 tokens, run prelude → loops k = 1..12 (the same injection as in training) and record h_k. Report:
- the relative change ‖h_k − h_{k−1}‖ / ‖h_{k−1}‖;
- the norm ratio ‖h_k‖ / ‖h_0‖;
- the loss after the coda at each k.

**Prediction:** **not** convergent. The relative change stays > 0.1 out to k=12 and the norm keeps growing; the loss is lowest near k=4.

**Criterion:** C20 "each pass refines (convergent loop)" is `VERIFIED` only if the relative change is non-increasing for k ≥ 4 **and** < 0.05 at k=12. Otherwise it's `REFUTED` for E2.

---

## SQ6 (+ SQ2): single-pass state tracking via rotation, vs loops (claims C22, C18)
*Side quest. Written and committed before running. It runs in the GPU slot after K1–K6 finish.*

**Design**
- **Task:** identical to K5: running parity over 64 random bits. Accuracy on positions 32–63 of held-out sequences, plus length generalisation on positions 64–127 of 128-bit sequences.
- **Models:** d=128, 2 heads, MLP 256, one block unless noted.
  - **(a)** GDN, 1 pass (same as K5 (i)).
  - **(b)** GDN looped ×4 (same as K5 (ii)).
  - **(c) RotSSM, 1 pass:** a simplified Mamba-3-style SISO layer written in the script in plain PyTorch with a sequential scan. Per head, a complex diagonal state h ∈ ℂ^{dv×N}, N=32:
    - λ_t = exp(−softplus(W_a x_t)) · exp(i·π·tanh(W_θ x_t)) (data-dependent decay **and rotation**);
    - h_t = λ_t ⊙ h_{t−1} + v_t ⊗ B_t;
    - y_t = Re(h_t C_t), with v, B, C linear in x_t.
  - **(d) RotSSM without rotation** (θ ≡ 0: real decay only, ≈ Mamba-2-style), as an ablation.
  - **(e) GDN with negative eigenvalues**, 1 pass (fla `allow_neg_eigval=True`, β ∈ (0, 2)): side quest SQ2.
- **Training:** identical to K5 (3,000 steps, batch 128, lr 1e-3), 2 seeds. Thread/GPU budget per AGENTS.md §5. ≤ 15 min.

**Prediction**
- (c) RotSSM ≥ 95% on positions 32–63, and largely holds on 64–127.
- (d) without rotation ≤ 60%.
- (a) GDN 1-pass < 75%.
- (e) GDN with negative eigenvalues ≥ 90%.
- (b) looped GDN: whatever K5 measured, below (c).

**Criteria**
- **C22** (local analogue: "a rotating state solves state tracking in one pass") is `VERIFIED` if the mean of (c) ≥ 90% **and** (c) ≥ (d) + 25 points. Otherwise `REFUTED` at our scale.
- **"Rotation beats looping for state tracking"** is supported if (c) ≥ (b) + 10 points. This bears on FusionDance's core bet (C18), and is reported in the ledger notes.
- **SQ2** is promising if (e) ≥ (a) + 20 points.

---

## K7: learning-rate tuning for the control (E10) and the best hybrid (E1), plus the replication protocol
*Written before any K7, E10 or replication run (science-gaps fix: every model so far used lr 3e-3 untuned, and every result is one seed). The runs use the GPU slot through `tools/launch_campaign.sh`.*

### K7-parity (prerequisite, CPU, already run)
`experiments/checks/k7_e10_parity.py` checks two things:
- The new `"arch"` switch in `gogeta/train.py` leaves FusionDance bit-identical for E0/E1/E2/E3/E9: state_dict, model config, global RNG state after construction, and FLOPs.
- E10 matches E0's params and analytic FLOPs.

Result in `results/K7-parity.json`: PASS. All 5 configs are bit-identical. E10 has 13,767,552 / 10,621,824 params and 96,758,784 train FLOPs/token, identical to E0.

### K7 design
- **Models:** E10 (`configs/e10_llama_ref.json`, HF `LlamaForCausalLM` at E0's shapes) and E1 (`configs/e1_hybrid.json`, our best hybrid).
- **Grid:** peak lr ∈ {1e-3, 2e-3, 3e-3, 6e-3}.
- **Per run:** **20M tokens** (`--max-tokens 20000000`, 611 steps), **seed 0**, everything else as registered: batch 32 × 1024, warmup 300 steps, cosine to 10%, AdamW β 0.9/0.95, wd 0.1, clip 1.0, compiled. The LR is passed with `--lr` and recorded in each run's config (`train.lr`, `cli_overrides`).
- **Runs (8):** `experiments/launch/campaigns/phase2a-k7-lr.campaign`, in this order: E10 at 1e-3, 2e-3, 3e-3, 6e-3, then E1 at 1e-3, 2e-3, 3e-3, 6e-3.
- **Metric:** final full-validation loss (`summary.json` `val_loss`).
- **Analysis:** `experiments/checks/k7_lr_sweep.py` reads the run folders, applies the rule below, and writes `results/K7.json` and the next campaign.

### K7 decision rule (fixed now)
1. For each model, the LR with the lowest final full-val loss wins.
   - A run that fails, or ends with a non-finite loss, counts as diverged (worst).
   - An exact tie (equal to 4 decimals) goes to the smaller LR.
2. If the winner is at an edge of the grid, add one step on that side and apply rule 1 again.
   - Steps below: 5e-4, then 2.5e-4. Steps above: 1.2e-2, then 2.4e-2.
   - At most two extensions per side. If the winner is still at the edge, use it and flag it as "not bracketed".
3. The tuned LRs (LR_E10, LR_E1) are used for every Phase 2b run. They are not re-tuned after seeing Phase 2b results.

### K7 prediction
- E10: best LR 3e-3 or 6e-3 (6e-3 slightly more likely at this short horizon). E1: best LR 2e-3 or 3e-3.
- Both curves are flat near the optimum: best vs second best < 1%.
- lr 1e-3 is clearly worse at 20M tokens: +3% to +8% vs the best.
- No run diverges at 6e-3 (grad clip 1.0), and no winner is at the low edge.

### K7 caveats (stated before running)
- **Short-horizon proxy:** the LR is picked at 20M tokens and used at 200M.
  - The optimal peak LR usually falls as the horizon grows, so the pick may be slightly too high for 200M.
  - This affects both models the same way. The *comparison* is fairer than before, but neither model is guaranteed optimal at 200M.
- **Warmup is 49% of a 20M-token run** (300 of 611 steps). This is kept on purpose, so the early phase matches the 200M runs.
- **One seed per LR:** LR differences smaller than the seed noise (expected ≤ 0.7%) can't be told apart. The rule still picks the minimum.
- E2 (looped) and E3 (ternary) are not tuned here. That's a remaining gap; an E2 sweep would cost ≈ 40 min.

## Replication protocol (Phase 2b) and the comparison criteria
**Runs:** `phase2b-replication.campaign`. Once K7 is decided, `k7_lr_sweep.py --write-campaign` generates it from `phase2b-replication.template`. Every run is 200M tokens (the registered `max_tokens`):
- E10 seeds 0, 1, 2 at LR_E10; E1 seeds 0, 1, 2 at LR_E1.
- **E11** (negative-eigenvalue GDN on the E1 stack, registered separately by the main agent) seeds 0, 1, 2 at LR_E1, right after E1's seeds. It's judged by its own registry prediction against E1's 3-seed mean at the same LR (criterion max(1%, 2 × pooled seed std)), not by the criteria below.
- **E9** seed 0 at **LR_E1**, because E9 is E1's layout minus one GDN block. Its committed registry prediction assumed lr 3e-3 and is **not edited**: it's scored as written, with the LR difference stated.
- **E0** seed 0 at LR_E10, **only if LR_E10 ≠ 3e-3** (otherwise the v0.3.0 E0 run is already at that LR). Purpose: implementation parity with the control, not another control.
- **E2** seed 0 at LR_E1, **only if LR_E1 ≠ 3e-3**, so that E2 and E9 (the loop question) are compared at the same LR.

Decision on E0/E2 seeds:
- E10 (3 seeds) replaces E0 as the dense control, so E0 doesn't need 3 seeds.
- E2 stays single-seed in this campaign (80 min per seed). Its comparisons use E1's seed spread as the noise estimate (below), so they stay `PARTIAL` at best.

**What counts as matched:**
- Every comparison is at matched tokens (200M) and near-matched parameters. Non-embedding: E10 = E0 10.62M, E1 11.09M (+4.4%), E9 9.17M, E2 9.46M shared.
- FLOPs differ, and are always reported next to the loss: E1 0.94× E10, E9 0.82×, E2 2.0×.

**Statistics:**
- For each model with 3 seeds: mean and sample std (ddof 1).
- Pooled sd of two models = √((s_A² + s_B²)/2).
- A difference between two single runs has noise sd ≈ √2 × s, with s = E1's seed std (the closest architecture).

**Criteria (fixed now)**
1. **C5, "the hybrid (E1) beats the dense control":**
   - `VERIFIED` if mean(E10) − mean(E1) > 2 × pooled sd **and** max(E1 seeds) < min(E10 seeds).
   - `PARTIAL` if mean(E1) < mean(E10) but either condition fails.
   - `REFUTED` if mean(E1) ≥ mean(E10).
   - The ledger row and every "E1 beats E0" sentence get restated against E10 at tuned LRs.
2. **Seed noise:** predicted s ≤ 0.004 for both E10 and E1 (the notebook's earlier replication prediction). Reported, not a pass criterion.
3. **Reproducibility of the old E1 run** (only if LR_E1 = 3e-3): the new E1 seed 0 run reproduces the v0.3.0 value 0.5846 within 0.002. Config, seed and model code are the same; only `train.py` changed, and K7-parity shows FusionDance construction is bit-identical.
4. **Implementation parity (E0 vs E10):** PASS if E0 at LR_E10 lies within mean(E10) ± max(3 × s_E10, 0.5% of mean(E10)).
   - PASS means the published E0 numbers were a fair (if untuned) dense yardstick.
   - FAIL means every "vs E0" claim is restated vs E10.
5. **The loop question (E2 vs E9, both single seed, same LR):** feeds C6.
   - "Looping beats one pass of the same blocks" is supported if E9 − E2 > 2√2 × s_E1.
   - "Looping hurts at this scale" is supported if E2 − E9 > 2√2 × s_E1.
   - Otherwise it's unresolved. Either way, E2 spends 2.4× E9's FLOPs per token, and that's reported with the result.
6. **E10's own registry prediction** is scored as written.

**GPU budget:**
- Estimated from v0.3.0 throughputs (compiled: E0 ~210k tok/s, E1 ~100k, E2 ~44k) and v0.3.0 wall times (incl. compile and evals). E10 is assumed to run like E0, and E9 at about 5/6 of E1's time.
- K7 ≈ 43 min: 8 runs, incl. ~2 min overhead per run and 60 s gaps. Grid extensions add ≈ 4–6 min each, if needed.
- Phase 2b ≈ 5.8 h: E10 3 × 23 min, E1 3 × 39 min, E11 3 × 39 min (assumed like E1), E9 33 min, plus gaps. Conditional runs: E0 +23 min, E2 +80 min.
- **Total ≈ 6.5 h if both tuned LRs are 3e-3, up to ≈ 8.5 h** with both conditional runs and two grid extensions. K8 (below) adds ≈ 5 min.

---

## K8: an external sanity reference in bits per character
*Written before any K8 run. Script: `experiments/checks/k8_external_reference.py`. It needs `HuggingFaceTB/SmolLM2-135M` (download approved by the human; fetched by another agent into `~/fusiondance-data/hf-cache/`, documented in `notes/smollm2-reference.md`; the script loads it local-only and refuses a revision mismatch):*
- *base model, apache-2.0, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;*
- *~271 MB: config.json, generation_config.json, model.safetensors (269.1 MB), tokenizer.json, tokenizer_config.json, special_tokens_map.json.*

**Design**
- **Data:** the 4,000 held-out docs (`docs_val.jsonl`, 3,938,783 chars).
- **Scoring:** each doc is scored on its own, with the model's own tokenizer.
  - It's prefixed by the model's BOS (SmolLM2) or our `<|eos|>` (our training stream puts `<|eos|>` between docs).
  - Only the doc's own tokens are scored.
  - Docs longer than the window use a sliding window (stride = window/2): 2,048 tokens for SmolLM2, 1,024 for ours.
- **Metric:** bits/char = Σ NLL / ln 2 / Σ chars.
- **Models scored:** SmolLM2-135M, plus our E10 and E1 seed-0 checkpoints at their tuned LRs (same protocol, after Phase 2b).
- The windowing was unit-tested on CPU: uniform and previous-token dummy models give the same result for every window size.

**Prediction**
- SmolLM2-135M: 0.28–0.45 bits/char (most likely ~0.33).
- Our checkpoints: 0.21–0.26 bits/char doc-level, i.e. 20–50% better than SmolLM2, because they're trained in distribution.
- Our doc-level bits/char agrees with the training-time `val_bits_per_char` within 5%.

**Criteria**
- **K8a plausibility:** PASS if every one of our checkpoints scores ≥ 0.5 × SmolLM2's bits/char.
  - A 14M model beating a 135M model trained on ~10,000× more tokens by more than 2× would be suspicious.
  - On FAIL, check for train/val near-duplicate leakage before trusting any of our loss numbers.
- **K8b arithmetic:** PASS if each checkpoint's doc-level bits/char is within 5% of its run's `val_bits_per_char`. Otherwise our bits/char conversion is suspect.

**Caveats**
- SmolLM2-135M is ~10× our size and was trained on ~2T tokens of different data (web, code, math).
- Its data may overlap this corpus; contamination isn't checked.
- It's a sanity yardstick, **not a control**: it says whether ~0.23 bits/char is plausible for this corpus, not whether our architecture is good.


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

> **Registration timing (main agent, recorded honestly):** this addendum's file was written at 18:30:17 local, before the first Phase 2b run (E10 seed 0) finished at 18:31:38. It was committed a few minutes later (v0.4.37). The main agent did not open E10 seed 0's results before this commit, and an E10-only result cannot inform either confirmatory comparison (both need E1/E11 runs, none finished).

### Clarification (2026-11-22 18:42, main agent, before any E1/E11/E9/E2 Phase 2b result exists): which rule sets the LEDGER status
The registered Phase 2b criteria (above, applied to the clean-val primary metric per Addendum A2) decide each comparison's **verdict** (VERIFIED / PARTIAL / REFUTED as defined there). The **ledger status** of the claim is the *stricter* of that verdict and `notes/statistics-policy.md`: a claim is recorded `VERIFIED` only if the registered criterion passes **and** the Holm-adjusted 95% CI of the paired difference excludes 0 (for H1) or the TOST passes at the registered margin (for H2). If the registered criterion passes but the CI/Holm condition fails, the ledger says `PARTIAL (registered criterion met; CI includes 0 at n=3)`. Leave-one-seed-out paired differences are reported for every confirmatory comparison; if dropping any one seed flips the sign, the result is flagged *fragile* in the ledger row. Raised by quick review `notes/reviews/quick/2026-11-22-1836-statistician.md`.

---

## Registered A3 design predictions (2026-11-22 18:47, committed in v0.4.40 before any of these runs exists)
*Source: rigor workstream A3 (`notes/design-rigor.md`, `experiments/design/`). Staged configs in `experiments/staged-configs/` (E20-E27, E30-E36, E40-E48, E50-E53) move into `configs/` between campaigns. Primary metric: clean-val doc-level bits/char (K9); the ledger status rule of the 18:42 clarification applies.*

All outcomes are relative differences in the **primary metric: doc-level bits/char on the clean val split** (near-duplicates removed, K9). Every analysis is repeated on the near-dup docs, and the memorisation gap g = bpc(near-dup) - bpc(clean) is reported next to it. Until A1's clean list exists the same numbers are computed on full val and flagged. Seed sd is assumed 0.004 in loss (0.4%) until Phase 2b measures it; the "2 sd" thresholds below are recomputed from the measured value.

## H. Fair HPO protocol and horizon check (claims C160, C167)
**Protocol P1 (applies to every model that appears in a claim):**
1. Same grid shape: the K7 grid {1e-3, 2e-3, 3e-3, 6e-3, 1.2e-2}, 20M tokens, seed 0 (5 points; E10 lacks 1.2e-2: run H0). An edge winner gets one extension step, the same rule for all.
2. Same horizon rule: a proxy winner is provisional. The claim-horizon (200M) check is a 3-point bracket around it (winner plus one neighbour each side where the grid allows), same count for every model of a comparison.
3. Same schedule: cosine to 0.1, warmup 300 steps at 200M. At 20M the same 300 steps are 49% of the run, so warmup is NOT swept per model; one registered sensitivity pair (E50, E51: 30 steps) checks that the proxy ranking is not an artefact.
4. Seeds: the selection run is never reused as evidence; claims use fresh seeds at the selected LR.
5. Report both: each model at its own best LR, and each model at the shared LR, so the tuning effect is visible.

**Horizon check, stage A (campaign b):** E1@3e-3, E10@6e-3, E10@2e-3 at 200M, seed 0. The coordinator's suggestion was the first two; E10@2e-3 is added because horizon theory predicts the optimum moves DOWN, and E10@6e-3 tests only the opposite direction. (A v0.3.0 E1 run at 3e-3 already exists, val 0.5846, but under older code; a clean re-run is preferred, and it doubles as the criterion-3 reproducibility check.)
- Prediction: the 20M optima hold at 200M. E10: 3e-3 stays best of {2e-3, 3e-3, 6e-3} (P 60%; 2e-3 wins P 30%; 6e-3 wins P 10%). E1: 6e-3 beats 3e-3 by 0.5-2.5% (P 60%); 3e-3 wins P 35%.
- Noise: a single run minus a 3-seed mean has sd s*sqrt(1+1/3) = 0.0046 at s = 0.004. Threshold T = 2 x 0.0046 = 0.0092 nats (about 1.6% at L = 0.58).
- Decision rule: for each model, "the optimum transfers" iff no neighbour beats the Phase 2b LR's 3-seed mean by more than T. Transfers -> keep the LRs, close C160. Does not transfer for model M -> (i) run the next point beyond the winner (bracket), (ii) repeat M's three seeds at the new LR, (iii) restate every comparison with M at its 200M-best LR; the C5 gap is then reported as a range [shared LR, own best LR]. Neighbour lower but within T: report "flat optimum", keep.
- Stage B (conditional): E1@1.2e-2 at 200M if E1@6e-3 beats 3e-3 by more than T (the edge rule again).

**Warmup sensitivity (E50, E51; claim C167):** K7's ranking is unchanged for E10 (winner 3e-3, P 70%) and E1 (winner 6e-3 or 1.2e-2, P 75%). If either winner changes, warmup interacts with the LR ranking: add a 200M check at the new winner for that model and treat the K7 LRs as provisional.

## Per-experiment predictions
- **E20** (e20_fact000.json): Cell 000 (dense, unlooped, fp), 20M, v2. Clean-val bpc within 3% of E52 run at 20M (same recipe, 14 vs 6 blocks); seed sd 0.3-0.6%. Best LR 3e-3 or 6e-3.
- **E21** (e21_fact001.json): Cell 001 (dense, unlooped, ternary): +4% to +10% vs E20 (K4: gap 10-15% at d<=256 and falling; 20M tokens favour fp). Best LR 6e-3 or 1.2e-2.
- **E22** (e22_fact010.json): Cell 010 (dense, looped r=4, fp, 5 unique blocks): +2% to +6% vs E20 (3x fewer unique params at equal FLOPs).
- **E23** (e23_fact011.json): Cell 011 (dense, looped, ternary): +6% to +14% vs E20; the ternary x loops term BC is within +-1% (P 55%) else positive = anti-synergy (P 35%).
- **E24** (e24_fact100.json): Cell 100 (hybrid, unlooped, fp): -1% to -4% vs E20 (E1's v1 gap was -3.6% at 6 blocks; on dedup data the attention advantage on memorised repeats shrinks, so the gap is smaller).
- **E25** (e25_fact101.json): Cell 101 (hybrid, unlooped, ternary): +4% to +10% vs E24.
- **E26** (e26_fact110.json): Cell 110 (hybrid, looped, fp = E2 layout on v2): +1% to +5% vs E24; E2 vs E1 on v1 (+5.8%) is the prior.
- **E27** (e27_fact111.json): Cell 111 (full fusion at 20M): worst loss of the eight. Full-model interaction I = y111-y100-y010-y001+2*y000 = +0.5% (+-1.5%): P(I <= -1%) = 10%, P(|I| < 1%) = 55%, P(I >= +1%) = 35%.
- **E30** (e30_dense_parammatch.json): Dense, mlp 1096, params = E1's (+0.24%). vs E10 mean: -0.2% to -1.0% (extra MLP capacity only). E1 still beats E30 by >= 2.5% (P 85%).
- **E31** (e31_dense_flopmatch.json): Dense, mlp 888, FLOPs = E1's (+0.1%). vs E10 mean: +0.3% to +1.2%. E1 still beats E31 by >= 3.5% (P 85%). Low priority: a dense model with fewer FLOPs than E10 cannot close the gap by construction.
- **E32** (e32_unshared14_e2flops.json): 14 unshared hybrid blocks at E2's FLOPs (-3.7%), 2.75x E2's unique params. Beats E2 at the same LR (6e-3) by 3% to 8% (P 85%) and E1 by 0% to 4%. If so, 'loops improve efficiency' survives only as parameter efficiency.
- **E33** (e33_unshared12_e2flops.json): 12 unshared blocks (review wording), mlp 1392, FLOPs = E2's. Within +-1.5% of E32; beats E2 by >= 2% (P 75%).
- **E34** (e34_e9_wide_e2flops.json): E9 layout widened to d=576 (FLOPs = E2's). Beats E2 by 1% to 6% (P 70%); within +-3% of E32. Not LR-retuned: a shortfall vs E32 larger than 3% is uninformative about width.
- **E35** (e35_dense5_unlooped.json): Dense 5-block unlooped (params-matched partner of E22): E22 vs E35 within +-1.5% (looping the dense stack 4x at 2.4x the FLOPs buys < 1.5%).
- **E36** (e36_e9layout_v2.json): E9 layout on v2 (params-matched partner of E26): E26 vs E36 within +-2.5%; direction unresolved.
- **E40** (e40_gdn2attn.json): E1 with GDN -> attention: +0.5% to +3% vs E1 on clean val (P 75% that it loses by >= 2 seed-sd = 0.8%). The review predicted >= 2 sd; v1's -3.6% E1 lead partly came from memorised repeats.
- **E41** (e41_swaonly.json): SWA-only stack: +1% to +4% vs E1; worse than E40 (window 256 sees only a quarter of the 1024 context).
- **E42** (e42_gdn_noconv.json): No short conv: +0.7% to +2.5% vs E1 (conv supplies local mixing before the delta rule). P(within 1 sd of E1) = 20%.
- **E43** (e43_gdn_nogate.json): No output gate: +0.3% to +1.5% vs E1. P(within 1 sd) = 35%.
- **E44** (e44_gdn_noonorm.json): No output RMSNorm: within +-0.6% of E1 (P 50%); risk of instability at 6e-3 (P 15% diverges).
- **E45** (e45_gdn_bare.json): Bare GDN (no conv, gate, out-norm): +1.5% to +5% vs E1; E45 - E1 >= E42 - E1 (P 80%).
- **E46** (e46_attnfirst.json): Attention first (last slot SWA): +0.8% to +3% vs E1 (the global layer sees only embedding-level features and nothing refines its output).
- **E47** (e47_attnmid.json): Attention in the middle (slot 4): within +-0.7% of E1 (P 55%).
- **E48** (e48_gdn_noswa.json): All-GDN + last attention (no SWA): within +-0.7% of E1 (P 55%); better than E1 by >= 0.7% (P 25%). Tests the Qwen3.8 report that pure-GDN hybrids beat SWA hybrids (SQ9).
- **E50** (e50_e10_warm30.json): E10 with warmup 30 at 20M: K7's LR ranking unchanged (winner 3e-3, P 70%); winner-loss lower than K7's 1.0286 by 1% to 4%.
- **E51** (e51_e1_warm30.json): E1 with warmup 30 at 20M: winner stays 6e-3 or moves to 1.2e-2 (P 75% that it does not move down); loss lower than K7's 0.9925 by 1% to 5%.
- **E52** (e52_e10_v2.json): E10 on dedup v2, 200M: clean-val bpc higher than the v1-trained E10's clean-val bpc by 0% to 6% (less duplicated data to memorise); the memorisation gap shrinks by >= 30%.
- **E53** (e53_e1_v2.json): E1 on dedup v2, 200M: E1 vs E52 gap on clean val within 2 points of the v1 gap (P 60%); smaller than the v1 gap (P 60%).

---

## Registered Byte Quest predictions E12-E16 (2026-11-22 ~19:55, committed in v0.4.68 before any byte run exists)
*Source: `notes/byte-quest.md` §6 (byte-model subagent), registered by the main agent. Primary metric: doc-level bits/char on val_clean80 (K9/A1), reported per doc-length bucket (C193). Ledger status rule of the 18:42 clarification applies. Configs staged in `experiments/staged-configs/byte/`.*

Priors: byte models trail BPE below about 4B params on code (reported by 2602.04289); an AR byte model needs 7.9x the
compute at 1.0 bpb (reported by 2605.12928); fixed patches suffer patch lag and P = 16 collapses on code (reported by
2605.09630); H-C2F priors from analysis section 7 (sharing within +-0.5% of equal-FLOP unshared, from C6). Our scale
(13M params, ~2e16 FLOPs) is below every regime those papers tested, so the priors are directional only.

**E12 (flat byte-E1).** Predicted: clean-val bits/char ratio E12 / E1@53.5M-tok (matched text, E12 at 3.6x FLOPs)
1.00-1.12; E12 / E1@200M-tok (matched FLOPs, 0.96x) 1.15-1.40. E12 is worse than E13b (same text, 0.29x the FLOPs)
or within 2% of it. Criteria: "flat bytes viable at matched FLOPs" := ratio vs E1@200M <= 1.05; refute := > 1.15
(then flat bytes are dead at our scale and only the patched arms matter). Long docs: E12's gap is larger on docs over
1,200 token-equivalents than on short docs (from C193).

**E13 (granularity).** Predicted at 200M bytes: P = 4 is best or tied with P = 2 (|diff| <= 1%); P = 8 is 1-4% worse
than P = 4; P = 16 is >= 5% worse than P = 4 (patch lag); E13b / E1@53.5M-tok (matched text, 1.04x FLOPs) 1.02-1.12
(central 1.06). Per-slot loss rises with staleness (slot P-1 vs slot 0 at fixed P), by <= 3% at P = 4 and more at
P = 8/16 (weak, direction only). Criteria: the chosen P = argmin clean-val bits/char at 200M bytes, ties (<= 1%) broken
toward the cheaper (larger) P; "patching pays" := best E13 arm <= E12 + 1% at <= 0.5x E12's FLOPs.

**E14 (loops, H-C2F).** Predicted: (1) E14d (2 unshared) beats E14a (r = 1) by >= 0.5% (more decoder compute helps);
(2) **E14b (shared r = 2) is within +-0.5% of E14d** (sharing neither helps nor hurts at matched FLOPs); (3) E14c
(r = 4) beats E14b by <= 0.5% (diminishing); (4) with gated updates the pass-to-pass hidden-norm ratio stays below 1.5
(K6 counterpoint: E2's grew 7.4 -> 58.9x); (5) per-pass val loss decreases monotonically with pass index for every
shared arm. Criteria: H-C2F supported := E14b <= E14d - 0.5% and E14b <= E14a - 0.5% (3 seeds); sharing refuted :=
E14b > E14d + 0.5%; "loops buy nothing" := E14b within +-0.5% of E14a. Optional E14e: L(r) is flat within 0.5% for
r >= 2 and degrades by < 2% at r = 6-8 (no E2-style blow-up past trained r, because of gating and per-pass loss).

**E15 (parallel_refine).** Predicted: E15a (r = 1, no within-patch bytes) is 3-10% worse than ar E13b; **E15b
(r = 2) within 1% of ar E13b**; E15c (r = 3) within 0.5% of E15b. Slot 0 matches ar (same information); the gap
grows with slot index. Criteria: "parallel decoding viable" := E15b <= E13b + 1%; refute := E15b > E13b + 3%.
Decode-speed benefit (P bytes per r decoder passes) is not measured by E15 and must not be claimed from it.

**E16 (fusion).** Predicted: (a) the looped global core (2.24x E1 FLOPs per text) is within +-1% of the unlooped E14
winner (loops in the global body don't pay, as C6 found for E2); refute of that prior := E16a <= winner - 1.5%.
(b) Ternary global body: +3-9% clean-val bits/char vs the bf16 winner (K4 width trend, partly offset by keeping
all byte layers in bf16); "ternary viable" := gap <= 5%.

### Config binding for the Byte Quest predictions (added 2026-11-22 ~19:58, v0.4.69, still before any byte run)
The predictions registered in v0.4.68 apply to exactly these staged configs (sha256, first 16 hex). Any config change before a run needs a new dated entry here. Raised by quick review `notes/reviews/quick/2026-11-22-1956-reproducibility-auditor.md`.

| Config | sha256[:16] |
|---|---|
| `e12_flat_w1024.json` | `30da6e313e4748dd` |
| `e12_flat_w1024_20m.json` | `1a672b98d49f75ba` |
| `e13a_p2_r1.json` | `a6828021eccc0556` |
| `e13a_p2_r1_20m.json` | `d23d0548df93a34a` |
| `e13b_p4_r1.json` | `d6e874f834cff234` |
| `e13b_p4_r1_20m.json` | `3d247fd4104acbb1` |
| `e13c_p8_r1.json` | `5d65dc75691d0882` |
| `e13c_p8_r1_20m.json` | `3145b1864e283f7b` |
| `e13d_p16_r1.json` | `0a062ad26c58327d` |
| `e13d_p16_r1_20m.json` | `f445e019a94033b8` |
| `e14a_p4_r1.json` | `e34cbf455d946e2d` |
| `e14a_p4_r1_20m.json` | `2a6cbd6953e8b574` |
| `e14b_p4_r2_shared.json` | `d6e583dce05e2840` |
| `e14b_p4_r2_shared_20m.json` | `91f1866acbc876b8` |
| `e14c_p4_r4_shared.json` | `9c94b5312d71eb44` |
| `e14c_p4_r4_shared_20m.json` | `a40cbf4782df4a56` |
| `e14d_p4_r2_unshared.json` | `db24fb57e0dc0bc5` |
| `e14d_p4_r2_unshared_20m.json` | `b5fab554619c856c` |
| `e14e_p4_rsampled.json` | `e792c39ccff81c6f` |
| `e14e_p4_rsampled_20m.json` | `78737c4784bbe35f` |
| `e15a_par_p4_r1.json` | `1d300274a857895e` |
| `e15a_par_p4_r1_20m.json` | `90a6211751071b30` |
| `e15b_par_p4_r2.json` | `1e58e3f85d6b7f74` |
| `e15b_par_p4_r2_20m.json` | `b021bfe3f21bc086` |
| `e15c_par_p4_r3.json` | `4fe7086bfeaa9f57` |
| `e15c_par_p4_r3_20m.json` | `372664662845ca45` |
| `e16a_p4_r2_looped_global.json` | `e9b0958e18353fef` |
| `e16a_p4_r2_looped_global_20m.json` | `832992308bf8e547` |
| `e16b_p4_r2_ternary_global.json` | `6d83780a6e24029e` |
| `e16b_p4_r2_ternary_global_20m.json` | `65d10e883b2f4580` |
