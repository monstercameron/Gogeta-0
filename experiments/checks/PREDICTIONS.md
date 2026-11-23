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

## Registered SQ214 window-sweep and SQ220 part-2 (RTX 5090) predictions (2026-11-22 20:38, committed in v0.4.71 before any 5090 run exists)

### SQ214: E1 sliding-window sweep (configs `experiments/staged-configs/sq214/e1_w{64,256,1024}_20m.json`)
- **Question:** is E1's SWA window of 256 what costs it long documents (C193: −3.86% on ≤ 300-token docs, +1.09% on > 1,200 tokens vs E0)?
- **Design:**
  - E1 with both SWA layers' window ∈ {64, 256, 1024 (= T, full causal)}; everything else identical. E1 already has one global full-attention layer (the last), so the effects should be small.
  - Seeds 0 and 1, 20M tokens, lr 6e-3 (E1's K7 best), `--compile`.
  - **One SKU:** all six runs on the same rented RTX 5090. The comparison is never pooled with N1X runs.
  - **Compute:** every arm does the same arithmetic, because the SWA mask is dense O(T²), C12. Analytic FLOPs differ only in bookkeeping (`min(window, T/2)`), so the comparison is at matched wall-clock compute.
- **Primary metric:** clean-val doc-level bits/char (`val_clean80`, `experiments/stats/score_doclevel.py`), with the mean over 2 seeds. Secondary: in-training val_loss (contaminated val, labelled so).
- **Status:** **exploratory** (n = 2 seeds at 20M tokens). Per `notes/statistics-policy.md` §4 it never changes a claim status; it only creates queue items.
- **Predictions:**
  - **W1:** |bpc(w1024) − bpc(w256)| ≤ 1.0% (relative). The global layer already gives full context.
  - **W2:** bpc(w64) is worse than bpc(w256) by 0.3–2.0%.
  - **W3:** the ordering w1024 ≤ w256 ≤ w64 holds in the 2-seed means.
- **Decision rule:**
  - If w1024 beats w256 by more than 2× the seed spread (|s0 − s1| averaged over the arms), register a 200M-token confirmatory run, plus length-binned scoring (SQ214's binned part, which needs per-doc NLL), before any claim.
  - If not, C193's window explanation loses support, and SQ219 (context-removal) becomes the next test.

### SQ220 part 2: the RTX 5090 bridge (same protocol as the 3090 part, `experiments/devlogs/SQ220.md`)
- **Runs:** E10 and E1 at seed 0, lr 3e-3, 20M tokens, `--compile`, the same as the N1X's K7 runs and the 3090.
- **Throughput metric:** the median of per-step tokens/s from step 100 on. The whole-run tokens/s mixes in the cold-box compile and autotune cost, which is reported separately.
- **Predictions:**
  - **5-P1:** |val_loss(5090) − val_loss(N1X)| ≤ 0.004 for both.
  - **5-P2:** steady-state tokens/s on the 5090 ≥ 1.25× the 3090's for E1 (3090: 254.5k) and ≥ 1.10× for E10 (3090: 272.5k). At 13.8M params the runs look overhead-bound, so the 5090's ~1.9× bandwidth won't fully show.
  - **5-P3:** cost per 1M training tokens on the 5090 is within ±30% of the 3090's.
- **Config binding** (sha256, first 16 hex): `e1_w64_20m.json` `afa384e4fee34fff`, `e1_w256_20m.json` `3827c59a85eef8c9`, `e1_w1024_20m.json` `cd39fde744e58807`.
- **Pre-run amendment, 2026-11-22 20:50** (before any SQ214 run started; outside review 20:40 idea E):
  - Seeds go from {0, 1} to **{0, 1, 2, 3}** per arm, so 12 runs.
  - The seed spread in the decision rule becomes the sd across the 4 seeds of each arm (pooled over arms).
  - W1–W3 are unchanged. The 20M-token seed sd and its CI are reported as a by-product (SQ216).
  - **Scoring caveat (SQ226):** docs longer than 1,024 tokens are scored in chunks, which restarts the recurrent state. The primary metric is unchanged, but any window effect is read with that caveat until SQ226 runs.

## Registered fleet wave 1 (2026-11-22 ~21:20, before any of these runs exists): Phase 2b on RTX 5090 plus SQ232

### Phase 2b-5090: an independent single-SKU replication
- **Design:**
  - The same 11 items as `experiments/launch/campaigns/phase2b-replication.campaign`: E10 × 3 seeds at lr 3e-3; E1 × 3 and E11 × 3 at 6e-3; E9 s0 and E2 s0 at 6e-3. 200M tokens each.
  - All of them run on rented RTX 5090s (one SKU), with the same code as the N1X campaign (`gogeta/`, `configs/` and `tools/` are unchanged since `9712842`).
  - Checkpoints are pulled; clean-val doc-level scoring (`experiments/stats/score_doclevel.py`) runs on every checkpoint.
- **Status relative to the N1X campaign (no forking path):**
  - The N1X Phase 2b E1-vs-E10 comparison (3 + 3 seeds, complete) **stays the registered confirmatory test**, Q12, per Addendum A2. Its clean-val scoring is still to run.
  - The 5090 run is a **second, independent replication** of the same design, registered here before any 5090 Phase 2b number exists.
  - It is confirmatory for the arms the N1X campaign did not finish (E11 vs E1; E9 and E2, single seed, so descriptive only). It replicates E1 vs E10.
  - The two SKUs are never pooled in one test: C195's ±0.004 tolerance is larger than the effects in question.
- **Predictions:**
  - **R1:** on the 5090, the E1-vs-E10 sign on clean80 bpc matches the N1X's sign (whatever that turns out to be).
  - **R2:** 5090 val_loss per item within ±0.004 of the matching N1X item, for the 6 items the N1X finished (extends C195 to 200M tokens).
  - **R3:** E11 vs E1, the registered E11 hypothesis (`experiments/checks/PREDICTIONS.md`, E11 section), judged on the 5090 alone.

### SQ232: same-SKU replay floor
- **Design:** E10 and E1 at seed 0, lr 3e-3, 20M tokens, `--compile`, each run **twice** on RTX 5090s (4 runs), compared with the earlier 5090 bridge runs (E10 1.0297, E1 1.0051).
- **Prediction P-R1:** |Δval_loss| between same-seed same-SKU replays ≤ 0.002, smaller than the cross-SKU differences in C195 (up to 0.0020). If replays differ by ≥ 0.002, C195's agreement is within run-to-run noise and the tolerance discussion changes.
- **Wave-1 amendment (same session, before dispatch; outside review 21:10, critique 4 and idea I6):**
  - Phase 2b-5090 runs **E10 and E1 at seeds 0–6 (7 per arm)** instead of 0–2, because at n = 3 the E1-vs-E10 difference sits at about 0.5 SE, and resolving 0.3% at 80% power needs about 7 seeds per arm (computed in the review).
  - E11 stays at seeds 0–2, and E9 and E2 at seed 0.
  - The registered test for R1 is the paired-by-seed difference over seeds 0–6 (Welch as a sensitivity check), on clean80 bpc, with Holm across the two Phase 2b-5090 confirmatory comparisons (E1 vs E10; E11 vs E1).
  - **The resolvable effect is stated before any result:** with an assumed per-arm sd of 0.0020 on val_loss (the N1X 3-seed sd), the 95% half-width of the paired mean difference at n = 7 is about 0.0018 (computed with sd_diff about 0.0020, an assumption).

## Fleet wave 2 (2026-11-22 ~21:45, before dispatch): registered A3 items that run on code-py-v1, all on RTX 5090
Nothing new is predicted here. This executes items registered in v0.4.40 (sections "Registered A3 design predictions" and "H. Fair HPO protocol and horizon check"), judged by their registered criteria, on **one SKU** (RTX 5090). References come from the same SKU: Phase 2b-5090 (wave 1) and K7-5090 below. Never pooled with N1X runs.
- **K7-5090 (tuning, no claim):** the K7 LR grid re-run on the 5090 at seed 0, 20M tokens: E10 at {1e-3, 2e-3, 3e-3, 6e-3, 1.2e-2} (1.2e-2 is H0) and E1 at {1e-3, 2e-3, 3e-3, 6e-3, 1.2e-2}. A different chosen LR than the N1X's K7 is reported, not acted on mid-wave.
- **H1:** seed 1 at the top-2 LRs, 20M: E10 at {2e-3, 3e-3}, E1 at {3e-3, 6e-3}.
- **H4:** warmup-30 sensitivity, 20M: E50 (E10) at {3e-3, 6e-3}, E51 (E1) at {6e-3, 1.2e-2}.
- **H2:** horizon check stage A at 200M, seed 0: E1 at 3e-3, E10 at 6e-3, E10 at 2e-3. H3 runs only if its registered condition holds.
- **A3 matched arms** at 200M, seed 0, at their file LRs: E30, E31, E32, E33, E34. Registered design: "1 seed each".
- **Blocked:** everything on code-py-v2 (E20–E27, E35, E36, E40–E53 except E50/E51). The builder is in progress, and the v2 items wait for its manifest. E42–E45 also need the GDN-flags patch.
- **Erratum (timestamps):** the wave-2 heading's "~21:45" was an estimate. The section was written and committed at about 21:22 local (v0.4.75), still before any wave-2 dispatch. The wave-1 "~21:20" heading is consistent with v0.4.74's commit time.

## Fleet wave 3a (2026-11-22 ~21:45, before dispatch): registered A3 stages on code-py-v2, all on RTX 5090
This executes only stages registered in v0.4.40 (`experiments/design/power-cost.md`, A3 predictions), on one SKU, and adds no new predictions.
- **Data:** v2 as committed in v0.4.77, rebuilt on each box from `build_v2.py@b494ff8` and verified against `dataset-v2.manifest.json` before use.
- **Scoring:** v1 clean80. v1 val docs were removed, with their families, from v2 train, so this metric is comparable across v1 and v2 models.
- **Runs (seed 0, 20M tokens unless stated):**
  - **F0:** LR tune of the 5 primary factorial cells (E20, E21, E22, E24, E27) at {3e-3, 6e-3, 1.2e-2}: 15 runs.
  - **F3 (tune part):** E23, E25, E26 at {3e-3, 6e-3, 1.2e-2}: 9 runs.
  - **F4:** E35 and E36 × seeds {0, 1, 2} at their file LR: 6 runs.
  - **E1 decomposition, minimal tier, runnable arms:** E40, E41, E46, E47, E48, plus the E1 control (e53 at 20M) at 6e-3: 6 runs. E42–E45 wait for the GDN-flags patch.
  - **H5 dedup bridge:** E52 (E10) and E53 (E1) at 200M: 2 runs.
- **Waits for wave 3a:** F1, F2 and F3's seeds use the LRs chosen in F0/F3 by the registered G3 rule, so they are dispatched only after F0/F3 results exist.

## Fleet wave 4 (2026-11-22 ~22:50, before dispatch): SQ243 two-size check + SQ241 param-matched E1, RTX 5090
- **Power (SQ253):** in Phase 2b-5090 the paired E1−E10 difference has an sd of about 0.0012 bpc at 200M, about 0.45% (computed from `phase2b_5090.json`'s paired CI). With **4 seeds per arm**, the 95% half-width is about 0.0019 (0.7%), and the 80%-power detectable effect is about 0.0025 (0.9%). Assumption: the same sd at other widths.
- **Primary metric:** clean80 bpc on v1 val (same as Phase 2b). All runs on RTX 5090, compared only with RTX 5090 runs.

### SQ243: two-size check (configs `experiments/staged-configs/sq243/`, sha256-16 in brackets)
- **Arms:**
  - E1 at d = 256 (`e1_d256` [a898c7d0a2d380b5]; 7.06M total, 99M tokens) and d = 512 (`e1_d512` [789ef80f3507bd30]; 24.0M, 337M tokens);
  - E10 at hidden 256 (`e10_d256` [29a5563361330ffa]; 6.84M, 96M tokens) and 512 (`e10_d512` [7bc8aba3ab274330]; 23.2M, 326M tokens).
- **Tokens:** 14.05 tokens per total param, the same ratio as the d = 384 Phase 2b runs (outside review 22:30, idea E).
- **Seeds:** 0–3 per arm, 16 runs. The d = 384 point is Phase 2b-5090 (7 + 7 seeds).
- **LRs:** E1 6e-3 and E10 3e-3 at every width, not re-tuned. That is a limitation: if an arm's 20M optimum shifts with width, the comparison is confounded.
- **Predictions:**
  - **S1:** at d = 512, E1 − E10 is within ±1% (the tie persists).
  - **S2:** at d = 256, within ±1%.
  - **S3** (exploratory): the gap's trend across 256/384/512 is reported. Only a gap beyond the 0.9% MDE at an end point is called a scaling effect.

### SQ241: param-matched E1 (`experiments/staged-configs/sq241/e1_parammatch.json` [0b99bffe757efaed])
- **Arm:** E1 with mlp_hidden 952, so non-embedding params are 10,622,052 against E10's 10,621,824. 200M tokens, lr 6e-3, seeds 0–3.
- **Predictions:**
  - **P1:** E1-parammatch vs E10 (7-seed 5090 mean) is within ±1% (still a tie).
  - **P2:** it is no more than 0.5% worse than full E1 (the 7-seed mean).

## Fleet wave 5 (2026-11-22 ~23:05, before dispatch): factorial F2 re-sized from the F1 pilot
- **Pilot result** (F1, `experiments/checks/results/fleet_2026-11-22.json`):
  - per-cell log-sd at 20M on v2: E20 0.0064, E21 0.0105, E22 0.0117, E24 0.0018, E27 0.0108;
  - **pooled sigma 0.0090**.
- **Why re-size:** the registered design (`notes/design-rigor.md` §3) planned N = 46 at sigma 0.004 and N = 127 if sigma were 0.0067; "the pilot decides before the money is spent". 0.009 is outside both pre-specified cases.
- **F2'' (new registration, written before any F2 run):** N scales with sigma², giving **N ≈ 235** for 1% at 80% power. With the registered allocation shape kept (E20, the 000 cell, gets 2× the others), the targets are E20 = 82 seeds, E21 = E22 = E24 = E27 = 41 seeds each (N = 246). Existing seeds 0–2 count, so **231 new runs**.
  - All at 20M tokens on code-py-v2, at the F0/G3-chosen LRs: E20 3e-3, E21 6e-3, E22 6e-3, E24 3e-3, E27 6e-3. Seeds 3…n−1.
  - Rented RTX 5090s only. Estimated about 4 GPU-h, about $2.5.
- **Analysis (unchanged, as registered):** `experiments/design/factorial_analysis.py` on clean80 bpc (log scale): the interaction I, its 95% CI and power, plus the registered prediction **I = +0.5% ± 1.5%; P(I ≤ −1%) = 10%; P(|I| < 1%) = 55%; P(I ≥ +1%) = 35%; full fusion the worst of the cells**.
- **Disclosure:** the sample size changed after the pilot's variance was seen, as the design intended. No pilot *means* enter the sizing.

## Registered SQ188 byte LR sweep + SQ189 probe (2026-11-22 ~23:15, committed before any byte run; drafted by the Byte Quest prep subagent, reviewed by the main agent)
- **Question:** does the K7 value lr 6e-3 (tuned for BPE E1; the byte configs' default) transfer to byte-level patched models, or does a neighbour beat it? Needed before the E13 sweep so the arms are not confounded by an untuned LR.
- **Design:** `e13b_p4_r1_20m.json` (sha256[:16] `3d247fd4104acbb1`, v0.4.69 binding), lr in {3e-3, 6e-3, 1.2e-2} via `--lr`, seed 0, 20M bytes (153 steps of B=32 x T=4,096, warmup 8), `--compile`, one RTX 5090 box (never pooled with N1X runs). Jobs `SQ188_e13b_lr*` in `experiments/byte_model/wave4_jobs.txt`.
- **Metric:** final full-val `val_loss` of the run (bits/byte = loss / ln 2; same text for all three arms, so raw loss is comparable). Tuning only: no claim, no ledger status change, status exploratory (n = 1 seed, 20M bytes).
- **Predictions:**
  - **L1:** the loss minimum is at 6e-3 or 1.2e-2, not 3e-3 (the byte configs inherit K7's E1 value; a 256-way head and 4-byte patches give a noisier, lower-entropy target that should tolerate at least that LR).
  - **L2:** the 3e-3 / 6e-3 gap is at most 2% of the loss; 1.2e-2 is within 3% of the best (a flat basin, as in K7). No arm diverges (non-finite loss or loss above the lr 3e-3 arm by > 10%).
- **Decision rule (before any result):** the E13 sweep and the 200M runs use the lr with the lowest val_loss, except that if 6e-3 is within 0.5% of the best, keep 6e-3 (the configs' registered default, no change). If the best lr is 1.2e-2 or 3e-3 by more than 0.5%, use it for all E13 arms (`--lr`), recorded in each run config. If the best arm sits at the grid edge (3e-3 or 1.2e-2) by more than 2%, add one extension point (1.8e-2 or 1.5e-3) before E13; no other re-tuning for E13 to E16.
- **Caveat:** single seed and 153 steps; the optimum at 200M bytes (1,526 steps) may be lower, so this is a transfer check, not a tuned optimum (K7 used 20M tokens, same horizon caveat).

- **SQ189 (memory/throughput probe):** 50-step `--bench` runs of e13b, e14b, e14c, e15b and e13a on one RTX 5090. It is informational: it replaces the flops-table time estimates and its peak memory gates E13a. No prediction is scored.
- **SQ221 (byte-dataset determinism):** `cockpit/remote/fetch_bytes.sh` rebuilds `code-py-v1-bytes` on the box and must print `BYTES OK` against the committed `dataset-bytes.manifest.json`. A mismatch blocks every byte run.

### Wave-5 amendment (2026-11-22 ~23:30, before any F2 run; outside review 23:10, ideas A/B)
- **Seed blocking:** every cell's seed s uses the same data order, since the order is set by the train seed and all five cells train on code-py-v2 with T = 1,024.
  - On the 15 pilot runs, a common seed effect explains 35% of the residual SS; the seed-blocked log-sd is 0.0082 (df 8) vs 0.0090 unblocked (df 10).
  - **The primary F2 analysis is the seed-blocked factorial contrast**: a seed fixed effect plus cell effects, on log clean80 bpc. The unblocked analysis is the registered sensitivity.
- **Blinded variance re-estimation:**
  - F2 runs in two stages. Stage 1 is the first 120 queued runs (interleaved across cells).
  - Then sigma is re-estimated from **within-cell, seed-blocked residuals only**: no cell means, contrasts or interaction estimates are computed or looked at.
  - Stage 2's size is set so the final N gives 80% power for |I| = 1%, capped at the registered N = 246.
  - If stage 1 already reaches that N, F2 stops.
  - Because the re-estimation is blinded, the alpha of the final test is unchanged (Kieser & Friede 2003). The stage-1 sigma and the stage-2 size are recorded in the devlog before stage 2 dispatches.

## Fleet wave 6, part 1 (2026-11-22 ~23:40, before dispatch): SQ238 dedup-only control
- **Question:** the H5 bridge showed E1 on code-py-v2 at 0.2599 clean80 bpc, against 0.2679 on v1 (−3.0%). v2 changes both duplication and coverage: 200M tokens is about 99% of v2's train but about 76% of v1's. How much of the gap is **dedup**?
- **Data:** `code-py-v1sub` (`experiments/data_integrity/build_v1_subset.py`, tests 7/7) is a seeded uniform random v1 subset with **v2's doc count (698,714)**, the v1-val families excluded, and **no dedup**. It is built on the box, and its manifest is committed after a 2-box determinism check (as for v2).
- **Arms** (200M tokens, lr 6e-3, RTX 5090, clean80 primary):
  - **A:** E1 on v1sub (`experiments/staged-configs/sq238/e1_v1sub.json` [85c36365c0f4f55a]), seeds 0–3;
  - **B:** E1 on v2 (E53), seeds 1–3, plus the existing seed 0;
  - **C:** E1 on v1, the existing Phase 2b-5090 7 seeds (0.26795).
- **Contrasts:**
  - **dedup effect** D = log bpc(A) − log bpc(B), paired by seed;
  - **coverage/size effect** S = log bpc(C) − log bpc(A), Welch.
- **Predictions:**
  - **D1:** D > 0 (dedup helps), with D between 1.0% and 3.0%.
  - **D2:** |S| ≤ 1.0% (a smaller but undeduplicated set is about as good as v1 at 200M).
  - **D3:** D + S ≈ the H5 gap (3.0%), within the CIs.
- **Decision rule:**
  - If D's 95% CI excludes 0 and is positive, the A3 v2 results stand as "with dedup".
  - If D's CI includes 0 and S explains the gap, v2's apparent gain is a coverage artefact, and A3 conclusions must not cite dedup.
- **Seeds:** 4 per arm. The MDE is about 0.9% (the SQ253 sd), so a D below about 1% would read as inconclusive.

## Fleet wave 6, part 2 (2026-11-22 ~23:55, before dispatch): SQ200 init 2x2 + SQ234 E1 under flat init (RTX 5090, 200M tokens)
- **Question:** E0 (our dense FusionDance) is about 3.7% worse than E10 (HF Llama) at identical shapes. Is that the **init**? And did E1, which shares E0's init, start with the same handicap, so that C5's tie understates the mixer?
- **Options** (v0.4.82, defaults bit-identical): `model.init` = "gogeta" (scaled residual-output) / "flat" (std 0.02) for FusionDance; "hf" / "gogeta" for E10.
- **Arms** (seeds 0–3, 200M tokens; configs in `experiments/staged-configs/sq200/`, sha256-16 in brackets):
  - E0-gogeta [4ba49a5374940ea0], the default, re-run on the 5090 because E0 has only 1 N1X seed;
  - E0-flat [b2b8ce6f9dff80f9];
  - E10-gogeta [c931615db8de0e52];
  - E10-hf: the 7 Phase 2b-5090 seeds;
  - E1-flat [9883c9c3e11105c9];
  - E1-gogeta: the 7 Phase 2b-5090 seeds.
  - LRs: E0/E10 3e-3, E1 6e-3. Not re-tuned per init (a limitation).
- **Primary:** clean80 bpc, log scale. Init effect within architecture: paired by seed where both arms have it, Welch otherwise. Architecture effect within init. Interaction.
- **Predictions:**
  - **I1:** E0-flat beats E0-gogeta by ≥ 1.5%, i.e. init explains at least about half of the E0→E10 gap.
  - **I2:** E10-gogeta is ≥ 1.0% worse than E10-hf. The scaled init hurts the HF stack too.
  - **I3:** E1-flat beats E1-gogeta by 0.5–2.5%.
  - **I4** (consequence, registered now): if I3 holds, E1-flat vs E10-hf is re-tested on the same seeds; an E1-flat advantage beyond the 0.9% MDE would reopen C5 as a new claim ("E1 beats E10 when both use their best init"), not as a status change to C5.
- **Decision rule:**
  - If I1 fails (E0-flat within ±0.5% of E0-gogeta), the E0 handicap is not the init. SQ240/SQ234-adjacent items (RoPE layout, eps) become next.
  - All four configs double as the GPU smoke test for the new `init` option. Any rc ≠ 0 stops the wave.

## Fleet wave 6, part 3 (2026-11-23 ~00:20, before dispatch): Inkling-2-Mini loop ingredients, config-only (SQ266, SQ271)
Source: `notes/research-inkling2-mini.md`, from the research agent's cited primary sources; the main agent has not independently verified every fact. Everything runs at 20M tokens, lr 6e-3, seeds 0–2, on an RTX 5090. Primary metric: clean80 bpc, plus the L(r) curve from in-training eval at r = 1…12.
- **Configs:** `experiments/staged-configs/inkling/`:
  - e2_uniform [05dd8ce02798d5a0]: the registered E2, r ~ U{1..7};
  - e9 [2a20bac0d684deda];
  - e2_fixed_r2 [71296d7ae90fa1e1];
  - e2_fixed_r3 [c2374ebb2111d555];
  - e2_fixed_r4 [29cef12f74aa97fd];
  - thick_pre [b9b343718a305a81];
  - thick_coda [812a052a942e35af].
- **SQ266 (fixed r):**
  - **K1:** e2_fixed_r4 beats e2_uniform by 0.5–1.5%.
  - **K2:** among the fixed-r arms, loss at the trained r decreases with r (r4 < r3 < r2), but every fixed-r arm degrades beyond its trained r (as C7 found).
  - **K3:** e9 still beats every looped arm at 20M. That is the C6 direction; a looped arm beating E9 by more than 2× the seed sd would be the first evidence for loops.
- **SQ271 (thick outer layers):** both variants have E2's unique-parameter count (12,606,488), with 13 block visits at fixed r = 5.
  - **K4:** thick_pre and thick_coda are each within ±1% of e2_fixed_r4 (14 visits), with thick_pre the better of the two (Inkling and 2610.40325 favour looping the middle).
- **Seeds and MDE:** 3 seeds per arm, so the MDE is about 1–1.5% at the 20M seed sd (0.4–1.2%). Smaller differences are ties.

## Fleet wave 6, part 4 (2026-11-23 ~00:30, before dispatch): SQ282 depth vs width (LR-fair), SQ283 C197 seed replication
Source: outside review 2026-11-23 00:10. C197's limitation (event 4): E34, which is E9 widened with about the same unique params and FLOPs as E32, is 14% worse than E32 at untuned LRs.
- **SQ282:**
  - **Runs:** E32 (14 unshared blocks, `staged-configs/e32_unshared14_e2flops.json`), E34 (5 wide blocks, `e34_e9_wide_e2flops.json`) and E2 (`configs/e2_looped.json`), each at lr ∈ {3e-3, 6e-3, 1.2e-2}, seeds 0–1, **20M tokens** (`--max-tokens 20000000`). 18 runs, RTX 5090, clean80.
  - **W1:** at each arm's best LR (G3, the min of the 2-seed mean), E32 beats E34 by ≥ 5%. Depth beats width at matched params and FLOPs, even LR-fair.
  - **W2:** E34 at its best LR beats E2 at its best LR.
  - **Decision:** if E32 and E34 tie within 2× the seed sd at their best LRs, C197's lever is "unique params" (as first read). If E32 wins, the lever is "distinct depth", and E60/MoE (which adds width-like capacity) loses its prior.
- **SQ283:** E32 and E2 at 200M, their file LR (6e-3), seeds 1–3, joining the existing seed 0. 6 runs.
  - **R1:** E32 beats E2 by ≥ 10% on the 4-seed means, so the C197 gap replicates.
  - **R2:** E32 beats the E1 7-seed mean by ≥ 5%.
- **Queue order (changed, not a registration change):** wave 6 (SQ200/SQ234 init, Inkling SQ266/SQ271, SQ238), then SQ282/SQ283, then SQ263 (the 200M factorial gate), then F2. **E60 is gated** on SQ282 and SQ238 (the review argues E60's prior rests on width).

## Erratum (v0.4.86): wall-clock labels in wave-6 parts 3 and 4
- The headers "Fleet wave 6, part 3 (2026-11-23 ~00:20 ...)" and "part 4 (2026-11-23 ~00:30 ...)" carry wrong times. The git commits that added them are `2526428` (2026-11-22 23:11 EST) and `9b78441` (23:12 EST); the outside review they cite was written at about 23:10 EST (its file `notes/reviews/2026-11-23-0010-review.md` is misnamed). The cloud boxes log in UTC, which is the likely source of the confusion.
- Both commits precede every SQ266/SQ271/SQ282/SQ283 dispatch (wave-6 jobs started after 23:17 EST, box logs 04:17 UTC and later), so "registered before dispatch" holds by git. The header text is left as registered.

## SQ301 (2026-11-22 ~23:50 EST, before dispatch): LR-fair re-test of C198 at d = 256
- **Why:** C198 (E1 1.24% better than E10 at d = 256) used E1 6e-3 and E10 3e-3 at every width (registered limitation). E10's d = 256 seeds are also 7× noisier than E1's (sd 0.0021 vs 0.0003, `experiments/checks/results/sq243_sq241.json`). Outside review 2026-11-22 23:39, idea 1.
- **Arms:** the SQ243 configs, unchanged (`e1_d256` [a898c7d0a2d380b5], `e10_d256` [29a5563361330ffa]; 99M / 96M tokens), each at lr {3e-3, 6e-3, 1.2e-2}, seeds 0 and 1. The existing SQ243 cells (E1 6e-3 and E10 3e-3, seeds 0–3) are reused; **8 new runs**. RTX 5090, clean80 bpc.
- **Best LR per arm:** the lowest 2-seed (s0, s1) mean clean80 across the three LRs.
- **Predictions:**
  - **W1:** E10's best LR at d = 256 is 6e-3 or 1.2e-2, not 3e-3 (smaller models tolerate higher LRs).
  - **W2:** at each arm's best LR, the paired (s0, s1) gap E1 − E10 lies in [−1.0%, +0.25%], i.e. the d = 256 advantage shrinks.
- **Decision rule for C198:**
  - Gap at best LRs within ±0.5% → C198 REFUTED (an LR artifact) and the README's d = 256 line is withdrawn.
  - Gap ≤ −1.0% at best LRs, both seeds the same sign → C198 keeps PARTIAL with this as evidence (still n = 2, and the init confound remains, see below).
  - Anything else → C198 stays PARTIAL, marked inconclusive at 2 seeds.
- **Known remaining confound:** E1 uses FusionDance's init and E10 HF's. SQ200 (running, d = 384) measures the init effect; it is not re-tested here.


## SQ307 (2026-11-23 ~01:10 EST, before dispatch): fresh-seed confirmation of SQ301 at d = 256
- **Why:** SQ301 picked each arm's best LR on seeds 0–1 and reported the gap on those same seeds (claim audit 2026-11-23-0102, C198). E1's best LR (1.2e-2) sat at the grid edge.
- **Arms:** the SQ243 d = 256 configs, unchanged: E1 at lr 1.2e-2 and 1.8e-2, and E10 at 3e-3. Fresh seeds 2–5, so 12 runs. RTX 5090, clean80 bpc.
- **Primary:** E1 at 1.2e-2 (the SQ301-selected LR) vs E10 at 3e-3, paired by seed (n = 4). E1 at 1.8e-2 is secondary and only tells us whether the E1 optimum lies beyond the old grid.
- **Prediction W3:** the paired gap E1 − E10 is ≤ −1.0%, with a 95% CI excluding 0. [Reasoning: the SQ243 6e-3 result, −1.24% on 4 seeds, already sits below −1%.]
- **Decision rule for C198:**
  - W3 holds → C198's d = 256 component is supported on fresh seeds. It stays PARTIAL, because "ties" at d = 384/512 would still need TOST.
  - Gap within ±0.5% → the d = 256 effect is refuted as selection plus noise.
  - Otherwise → inconclusive.

## SQ308 (2026-11-23 ~01:15 EST, before dispatch): init-controlled depth test of C197
- **Why:** the default "gogeta" init scales residual-output weights by 0.02/√(2·depth) (`gogeta/model.py`, `_init_weights`): std 0.0058 for 6-block E1 and 0.0038 for 14-block E32. So C197's depth comparison is also an init comparison, and C199 shows init moves all-attention stacks by about 3.5–3.8%. Outside review 2026-11-23 01:07, idea A.
- **Arms:** 20M tokens, lr 6e-3, seeds 0–2, RTX 5090, clean80 bpc. 12 runs (sha256-16 in brackets):
  - E1-gogeta: `configs/e1_hybrid.json` [622b918b6b7195fe] with `--lr 6e-3`;
  - E1-flat: `experiments/staged-configs/sq200/e1_flat.json` [9883c9c3e11105c9];
  - E32-gogeta: `experiments/staged-configs/e32_unshared14_e2flops.json` [53513fe81a78512c];
  - E32-flat: `experiments/staged-configs/sq308/e32_flat.json` [56422fa326dedc9c].
  - All with `--max-tokens 20000000`.
- **Predictions:**
  - **G1:** E32-flat vs E32-gogeta is within ±1.0%. The hybrid pattern was init-insensitive in SQ200 (I3), and E32 uses the same mixers.
  - **G2:** E32 beats E1 by ≥ 3% under **both** inits (paired by seed, CI excluding 0).
- **Decision rule:**
  - G1 holds and G2 holds → C197's depth lever is not an init artifact at 20M. C197 gets this as evidence; it stays exploratory at 200M.
  - The flat-init gap is within ±1% while the gogeta gap is ≥ 3% → the C197 lever is an init artifact. C197 gets a limitation, and the README's scale wording is withdrawn.
  - Anything else → inconclusive, reported as such.
- **Limitation:** this runs at 20M tokens, whereas C197 is a 200M result. LRs are not re-tuned per init.

## SQ263 (2026-11-23 ~01:45 EST, before dispatch): 200M gate on the five F2 factorial cells
- **Why:** F2 (the registered factorial) runs at 20M tokens, at LRs chosen at 20M. At 200M, E10 at 6e-3 was 3.0% worse than at 2e-3 (H2), so the 20M LR choice and the 20M interaction may not transfer to the registered 200M horizon. Outside review 2026-11-22 ~23:10, C.
- **Arms:** code-py-v2, 200M tokens (`--max-tokens 200000000`), seeds 0–2, RTX 5090, clean80 bpc (sha256-16 in brackets):
  - E20 `e20_fact000` [11ce08abb096e0db] at 3e-3;
  - E24 `e24_fact100` [b9864df4d62a9978] at 3e-3;
  - E21 `e21_fact001` [b8bb3c560919bf67], E22 `e22_fact010` [4d2348d0ab3b6a9b] and E27 `e27_fact111` [e7953eabf8f62475], each at **both** 6e-3 (their F0/G3 LR) and 2e-3.
  - 24 runs in total.
- **Predictions:**
  - **H2':** in at least 2 of the 3 six-e-3 cells, the 2e-3 arm beats the 6e-3 arm at 200M by ≥ 0.5% (3-seed means).
  - **H3'** (descriptive): the 200M interaction contrast, computed at each cell's better LR, is reported with its CI. At n = 3 it is underpowered and is not a confirmatory test.
- **Decision rule:** if H2' holds, F2's 20M interaction is conditional on 20M-selected LRs, and any 200M statement must use the per-cell best LR. F2's registered 20M analysis is unchanged either way.
- **Blinding:** SQ263 results are not analysed or looked at until F2's registered analysis has been run and recorded. The monitor redacts these runs too, because they carry E2x names.

## E60 stage 1 (2026-11-23 ~02:30 EST, before dispatch): Code-MoE core (E60a/E60b/E60c)
- **Design:** `experiments/design/code-moe-draft.md` §3–§5. The code lands in v0.4.96 (`gogeta/moe.py`): off by default, and bit-identical for dense models, pinned by `tests/test_moe.py`.
- **Arms** (code-py-v1, 200M tokens, RTX 5090, clean80 bpc; sha256-16 in brackets):
  - **E60a** `staged-configs/e60/e60a_moe.json` [b06cbada9d0604ac]: E1 with a dropless 4 × 512 top-2 MoE FFN, balance 0.01, z-loss 1e-3. 18.2M total non-embedding params, 11.1M active.
  - **E60b** `e60b_frozen_router.json` [6d3c11ecb472bc18]: E60a with a frozen random router, balance and z-loss off.
  - **E60c** `e60c_dense2048.json` [2bce1f8d7ff0eb42]: dense SwiGLU 2048, total-param matched (1.47× E1 FLOPs).
  - **Reference:** E1, the 7 Phase 2b-5090 seeds.
- **LR (G3 rule):** each arm at lr ∈ {3e-3, 6e-3, 1.2e-2}, seed 0, 20M tokens (9 runs). The 200M runs use each arm's argmin. If the argmin is at a grid edge by more than 2%, one extension point is run.
- **200M:** E60a, E60b and E60c × seeds 0–5 (18 runs). E60d (8 experts) is deferred.
- **Predictions** (log clean80, relative):
  - **M1:** E60a − E1 = −1.8% (−0.8% to −3.5%).
  - **M2:** E60c − E1 = −3.0% (−1.5% to −4.5%). E60a is worse than E60c by 0.3–2.0%.
  - **M3:** |E60a − E60b| ≤ 0.8%.
  - **M5** (descriptive): max per-expert share ≤ 0.35 in every layer at the end of training, and no expert below 2%.
  - **M8** (descriptive): measured tok/s of E60a vs E1 is 0.55–0.80, and of E60c vs E1 is 0.70–0.85.
- **Decisions:**
  - **D1:** the Welch 95% CI of E60a − E1 lies below 0 and the effect exceeds the MDE (about 0.8%) → a new claim; otherwise record "MoE buys nothing measurable at 14M / 200M".
  - **D2:** the CI of E60a − E60b lies below 0 → learned routing matters; otherwise "the gain is capacity, not routing".
  - **D3:** compare E60a and E60c per measured GPU-minute.
- **Context:** SQ282 (v0.4.94) found the C197 lever is unique params, not depth, so E60 is ungated. The stage-1 "max expert load > 0.6 or NaN" instability check gates stage 2.

## A1 assembly screen (2026-11-23 ~11:40 EST, before dispatch): one compensation at a time on the full fusion
- **Question:** at 200M the full fusion (F2 cell 111, E27) is 30.8% worse than dense and the ingredients interact with anti-synergy (C202). Which single compensation recovers the most of that gap?
- **Base:** E27 `experiments/staged-configs/e27_fact111.json` at lr 6e-3, 200M tokens (`--max-tokens 200000000`), code-py-v2. The existing SQ263 runs, seeds 0-2, are the reference: clean80 0.3303 / 0.3268 / 0.3288, per-seed log sd 0.0053.
- **Arms** (`experiments/staged-configs/a1/`, sha256-16 in brackets), each ONE change from E27, seeds 0 and 1, same LR and tokens:
  - **A1a** [c7c46ee31e10fc94]: fixed r = 4 in training (r ~ U{1..7} in E27).
  - **A1b** [c46c03ee356ea882]: learned per-pass loop embedding (zero-init).
  - **A1c** [42c9df11a85e0538]: contractive injection, h = lam*h + inject, lam = sigmoid(per-channel), init 0.9.
  - **A1d** [fb07718b3ca3a70e]: weight-only ternary (activations not quantised).
  - **A1e** [26d96983d6d349ba]: ternary core only (prelude, coda and inject in full precision).
  - **A1f** [0644beb7f8936d63]: late ternary: full precision for the first 50% of steps, ternary QAT after.
  - **A1g** [fc2cd08286cce5f9]: loop shape, core of 6 blocks x r = 2 (r ~ U{1..3}): 15.1M vs 9.5M non-embedding params, same 14 applications.
  - **A1h** [343590a98c4c6673]: loop shape, core of 4 blocks x r = 3 (r ~ U{1..5}): 11.4M non-embedding params.
- **Metric:** clean80 doc-level bpc (v2 val), log scale. Effect = mean over seeds 0-1 of log(arm / E27 same seed): paired by seed, since the data order is shared.
- **Predictions** (effect vs E27; negative = better):
  - **A1a:** -1.5% (range -4% to +0.5%). **A1b:** -0.5% (-2% to +0.5%). **A1c:** -0.5% (-2.5% to +1.5%).
  - **A1d:** -2% (-5% to 0%). **A1e:** -3% (-6% to -1%). **A1f:** -3% (-6% to -1%).
  - **A1g:** -2% (-5% to +1%). **A1h:** -1% (-3% to +1%).
  - At least one arm improves by >= 2%; none closes more than half of the 30.8% gap to dense.
- **Decision rule (screen, exploratory under policy §4):** an arm is a **winner** if its paired 2-seed effect is <= -1.0% (about 2 seed sd) and both seeds point the same way. Winners are stacked in A2 (registered separately, 3 seeds, with dense E20 and GDN E24 as references). A1 changes no claim status.
- **Throughput:** each arm's tokens/s vs E27 is reported (A1g and A1h change kernel shapes).

## SQ340 (2026-11-23 ~11:40 EST, before running): fp32 vs bf16-emulated scoring of the 200M F2 cells
- **Arms:** cells 000, 100 and 111 at their best LR, seeds 0-2, CPU, same clean v2 val doc subset: (a) fp32 everywhere, (b) q/k/v and the GDN output rounded to bf16 at the points where the GPU path casts (`gogeta/model.py` GatedDeltaNet.forward).
- **Prediction:** for each cell, shift = log bpc under (b) minus under (a). |shift(111) - shift(100)| < 1 point, and no cell's shift exceeds 1 point. **Falsifier:** a cell moving by >= 1 point, or 111 moving >= 1 point more than 100, would mean the 200M F2 scores must be re-scored in fp32 before C202's wording stands.

## SQ347 (2026-11-23 ~11:40 EST, before running): copy-deficit probe on the 200M F2 cells
- **Design:** all 5 cells at their best LR, seeds 0-2, CPU. Synthetic sequences: filler from clean v2 val docs with a 32-token span copied after a gap of 128, 256 or 512 tokens, 200 sequences per gap (fixed seed 0). y = log of the mean NLL on the second copy's tokens.
- **Estimand:** I_copy = y111 - y100 - y010 - y001 + 2·y000, with the seed-blocked estimator of `experiments/checks/sq263.py`.
- **Prediction:** at gaps 256 and 512, I_copy > 0 with a seed-blocked 95% CI excluding 0, and |I_copy| within 0.5x to 2x of I_bpc (+6.2 log-points). **Falsifier:** |I_copy| < 1 point or the opposite sign at both gaps: the reversal would then be general modelling, not a recall deficit.

### A1 amendment 1 (2026-11-23 ~11:35 EST, before any A1 result exists): truncation control, placebo, LR check
Prompted by outside review 2026-11-23 12:10 (`notes/reviews/2026-11-23-1210-review.md`). Added before any A1 run finished; no A1 number had been seen.
- **New arms** (seeds 0 and 1, 200M tokens, same paired analysis vs E27):
  - **A1i** [0798341738a823ff]: E27 with `bptt_loops = 7`, i.e. full backprop through every pass. A1a (fixed r = 4) also removes the truncation that E27 applies when r > 4, so A1a vs A1i separates "fixed depth" from "no truncation". Prediction: -0.5% (-2% to +1%).
  - **A1z** [f95bfccd70ea2fa7], the **placebo**: E27 unchanged, re-run on the A1 fleet at the A1 commit. Prediction: |effect| < 0.5%. If |effect| >= 1.0% or both seeds move the same way by more than 0.7%, the screen's reference is unreliable and every A1 effect is re-read against A1z instead of the SQ263 E27 runs.
  - **A1k:** E27 at lr 1.2e-2 (`--lr 1.2e-2`; SQ263 already has 2e-3, which lost 0.64%). It checks whether the fused cell's LR is at an edge. Prediction: within +/-1% of 6e-3.
- **Reading rules added:**
  - A1g and A1h change unique parameters, C197's known lever. They are reported as "parameter-changed arms", not as compensations, and are compared with the per-parameter expectation from C197.
  - The dense E20 (SQ263, seeds 0-1, 3e-3) is added to the table, so each arm's "share of the E27-to-dense gap closed" is a paired number.
  - Each arm's tokens/s vs E27 is reported next to its effect. A winner that costs more than 10% throughput is flagged and not stacked without a wall-clock note.
  - Multiplicity: with 10 arms and a 2-seed paired sd of about 0.55%, the -1.0% rule has a roughly 20-25% chance of at least one false winner. A1 winners therefore count only after A2 confirms them on **fresh seeds (2-4)**; A2 never reuses seeds 0-1.

## A2 assembled model (2026-11-23 ~12:10 EST, before dispatch): stacked A1 winners, fresh seeds
- **A1 outcome that this builds on** (exploratory screen; `experiments/checks/results/a1_screen.json`): the winners under the registered rule were A1g -11.6% (parameter-changed), A1e -8.3%, A1h -5.7% (parameter-changed), A1a -5.2% (+12% compute) and A1c -1.6%. A1b, A1d and A1f show no effect. A1 amendment arms (A1i, A1z, A1k) and A1b/A1f seed 0 were lost when the boxes were destroyed mid-run (orphaned teardown loop, disclosed in v0.5.4). They are re-run in this wave under their original amendment-1 registration.
- **Arms** (200M tokens of code-py-v2, lr 6e-3, **fresh seeds 2, 3, 4**, never used for A1 selection; `experiments/staged-configs/a2/`):
  - **A2s** [bf15dfbc94ad6b7a], the **assembled model**: E27 with a core of 6 blocks [gdn, gdn, swa, gdn, gdn, swa] x r = 2 (A1g shape), fixed r = 2 with full backprop (the A1a/A1i lever for this shape), ternary in the core only (A1e), contractive injection (A1c). 15.1M non-embedding params, 0.189 GFLOP/token training (FLOP-matched to E20 0.201 and E24 0.186, all 14 applications with gradients).
  - **A2f** [081a9dd4a2f13d23]: A2s without ternary (full precision everywhere), the assembled GDN + loops model. It prices ternary inside the stack.
  - **A2a** [537cbbad83647e9d]: A2s plus per-pass low-rank adapters (rank 32, SQ357).
  - **References**, seeds 2-4, each at its F2/G3 LR (seed 2 exists from SQ263; seeds 3-4 are new): E20 dense (3e-3), E24 GDN hybrid unshared at matched FLOPs (3e-3), E27 full fusion (6e-3).
- **Metric:** clean80 doc-level bpc, log scale. Comparisons are paired by seed (shared data order), with a 95% CI at df = 2 and the per-seed values listed.
- **Predictions:**
  - **P1 (stacking works, sub-additively):** A2s vs E27 = -17% (range -25% to -10%). Full log-additivity of the A1 winners would give about -24%.
  - **P2 (competitiveness):** A2s vs dense E20 = +10% (range 0% to +18%): the assembled ternary model still trails dense per training FLOP, at about a third of its unique parameters.
  - **P3 (ternary inside the stack):** A2s vs A2f = +7% (range +3% to +12%).
  - **P4 (loops vs unshared at matched FLOPs):** A2f vs E24 = +6% (range +2% to +10%).
  - **P5 (adapters):** A2a vs A2s = -1% (range -3% to +1%).
- **Decision rules:**
  - The stack is **confirmed** if A2s beats E27 by >= 5% with a paired 95% CI below 0. That creates a claim, while the A1 winners stay exploratory one by one.
  - The assembled model is **competitive per FLOP** if A2s or A2f lies within 5% of E20.
  - **Per weight-memory** (ternary core at 1.58 bits/param, everything else at 16 bits) and **per wall-clock** (measured tokens/s) are reported next to it as registered secondary readouts.
  - A2a replaces A2s as the base only if P5's CI lies below 0.

### A2 amendment 1 (2026-11-23 ~12:28 EST, before these runs; quick review 12:24): confirm A1's best single compensation on fresh seeds
- **Added arm:** A1e (ternary core only, `experiments/staged-configs/a1/a1e.json` [26d96983d6d349ba]) at seeds 2, 3, 4, lr 6e-3, 200M tokens. It is paired with the A2 wave's E27 references at the same seeds (seed 2 from SQ263, seeds 3-4 run in this wave).
- **Prediction:** A1e vs E27 = -6% (range -9% to -3%), shrunk from the screen's -8.3% for winner's curse.
- **Rule:** the single compensation is **confirmed** if the paired 95% CI (df 2) lies below 0 and the effect is <= -1%. Otherwise A1's A1e result stays an unconfirmed screen. This is one registered comparison, not a family.
- **A3 rule (added now, applied when A3 is registered):** A3's leave-one-out contrasts form one family, Holm-adjusted. A component is attributed only if its Holm-adjusted interval excludes 0 and clears 1%.

### A2 amendment 2 (2026-11-23 ~12:45 EST): loop go/no-go rule for the final model (E61), with disclosure
- **Disclosure:** written AFTER some A2 runs were scored and seen by the main agent: A2f seed 2 = 0.2501, A2s seeds 2 and 4 = 0.2727 / 0.2716, A2a seed 4 = 0.2714, and E27 seed 4 = 0.3265, against E20 seed 2 = 0.2506 and E24 seed 2 = 0.2412 (both from SQ263). So this is not a blind registration. It fixes a decision rule; it does not change A2's registered predictions P1-P5 or their decision rules.
- **Rule (D1/D3):** loops stay in the final-model design (E61) only if, at matched training FLOPs on clean80, the assembled full-precision model A2f beats the unshared GDN hybrid E24 by >= 1% over seeds 2-4 (paired by seed). Otherwise, or if A3 shows a benefit from looping that unique parameters do not explain, loops leave E61's default. They then stay only as a decode-time option, if SQ362 shows quality per decode-second favours them, or as the byte-refinement loop of the coarse-to-fine design (H-C2F, E14), which is a separate question.
- **Expected outcome from the seen seed** (stated so the rule cannot be bent later): A2f seed 2 trails E24 seed 2 by about 3.7%, so the rule will most likely drop loops from E61's default. P4 (registered before dispatch) predicted A2f vs E24 = +6% (+2% to +10%).

## A3 all-parts assembly (2026-11-23 ~13:05 EST, before dispatch): keep every component, design around its cost
- **Human decision (2026-11-23, chat):** "I want all the parts assembled and their issues designed around or fixed". Loops and ternary therefore stay in the model. A2 amendment 2's go/no-go rule ("loops leave E61's default") is **superseded by the human's decision**. Its comparison (A2f vs E24) is still reported, as a cost to design around, not as a gate.
- **A2 readout this builds on** (`experiments/checks/results/a2_assembled.json`, seeds 2-4):
  - A2s vs E27 -16.7% [-17.7, -15.6];
  - A2s vs E20 +8.6%;
  - ternary inside the stack (A2s vs A2f) +9.5% [+8.3, +10.7];
  - A2f vs E20 -0.9%;
  - A2f vs E24 +3.5% (2 seeds; seed 3 pending);
  - per-pass adapters 0.
  So the two remaining costs are **ternary (+9.5%)** and **loops' unique-parameter deficit vs unshared GDN (+3.5%)**.
- **Fixes under test** (one at a time on A2s, then combined; `experiments/staged-configs/a3/`; code in v0.5.10: ternary MoE experts, `moe_core_only`, `ternary_mixer`, `ternary_ffn`):
  - **A3m** [8a28dbb4df6101cc], loops' fix: a ternary MoE FFN in the shared core (4 x 512, top-2). Active FFN width 1024 equals the dense FFN, so FLOPs match (0.189 GFLOP/token), with 22.2M total vs 15.1M active non-embedding params.
  - **A3x** [a30c32f07a09ace9], ternary's fix, part 1: mixer projections in full precision, FFN ternary.
  - **A3y** [09bc189f87337544], ternary's fix, part 2: FFN in full precision, mixers ternary.
  - **A3mx** [53364a1125b533bd], **all parts**: GDN hybrid + shared loop 6x2 + contractive injection + ternary MoE core + full-precision mixers and ends.
  - **References**, same fresh seeds: A2s (the assembled base), A2f (full precision), E20 (dense), E24 (unshared GDN).
- **Seeds:** 5, 6 and 7 (fresh), 200M tokens of code-py-v2, lr 6e-3 for the A arms and the references' own LRs (E20/E24 3e-3). Paired by seed; clean80, log scale.
- **Predictions** (each vs A2s unless stated):
  - **Q1:** A3m -3% (range -6% to -1%).
  - **Q2:** A3x -5% (-8% to -2%) and A3y -3% (-6% to 0%). The ternary cost sits more in the mixers than in the FFN.
  - **Q3:** A3mx -8% (-12% to -4%).
  - **Q4:** A3mx vs A2f +1% (-2% to +4%): most of ternary's cost is designed away.
  - **Q5:** A3mx vs E24 +1% (-3% to +5%), and vs E20 -1% (-5% to +3%): the all-parts model is competitive with both references at matched active FLOPs.
- **Decision rules:**
  - A fix counts if its paired 95% CI (df 2) lies below 0 and the effect is <= -1%.
  - The four fixes form one family, Holm-adjusted.
  - The all-parts model is **competitive** if A3mx lies within 2% of E24 and at or below E20.
  - Throughput and weight memory are reported next to it.

### A1 amendment 2 (2026-11-23 ~13:20 EST, before these runs; quick review 13:14): third seed for fixed vs random depth
- **Added runs:** A1a (fixed r = 4) and A1i (random r, full backprop) at seed 2, 200M tokens, lr 6e-3, paired with E27 seed 2 (SQ263). Both arms use the same training compute (+17% vs E27).
- **Estimand:** A1a - A1i (log clean80, paired by seed, seeds 0-2, df 2).
- **Prediction:** -3% (range -5% to -1%). **Rule:** "fixed depth beats random depth at equal compute" counts as confirmed if the 3-seed paired 95% CI lies below 0 and the effect is <= -1%. Until then it stays exploratory, as the A1 screen is (the assembled recipe already uses fixed r on that exploratory basis).

## A4 full stack (2026-11-23 ~13:40 EST, before dispatch): every component in one model
- **Human instruction (v0.5.12):** "everything: ssm + sliding windows + loops + coarse to fine + bitnet + MoE + speculative decode".
- **Model** (`gogeta/c2f.py`, tokens-as-patches; `experiments/devlogs/C2F.md`):
  - The global trunk is the A3 all-parts backbone (A3mx: GDN + SWA + attention mixers, shared 6x2 loop with contractive injection, ternary MoE core, full-precision mixers and ends) on BPE tokens.
  - A byte decoder (d_local 128) spells token i+1 from the trunk state at token i. It is causal: only completed tokens are visible, which the bitwise perturbation tests check.
  - Self-speculative decoding is built separately (`gogeta/decode.py`) and measured on the trained checkpoints.
- **Arms** (`experiments/staged-configs/a4/`, sha256-16), 200M tokens of code-py-v2, lr 6e-3, seeds 5, 6, 7 (paired with A3mx at the same seeds):
  - **A4a** [92651a7ca10e9dc2]: shared byte refinement loop, r_b = 2 (H-C2F);
  - **A4b** [2477497ecfc336df]: single-pass byte decoder, r_b = 1;
  - **A4c** [f5d8f46678fc399b]: unshared 2-block decoder, parameter-matched to A4a (the A'' control).
  - Computed training FLOPs per token: 1.003x / 0.949x / 0.963x A3mx's (the decoder replaces the 8k-way head).
- **Metric:** in-training val loss in nats per target token on the v2 val split. The byte decoder's loss counts all bytes plus EOT of the target token, so it is directly comparable with the BPE head's val_loss on the same tokens. Paired by seed vs A3mx. Clean80 doc-level scoring is not wired for the byte decoder yet, which is a stated limitation.
- **Predictions:**
  - **R1:** A4a vs A3mx +2% (range -2% to +6%). Spelling a token byte by byte costs a little against a direct 8k-way head at this scale.
  - **R2 (H-C2F, sharing helps):** A4a vs A4b -1.5% (range -4% to 0%).
  - **R3 (sharing vs unique params):** A4a vs A4c 0% (range -1.5% to +1.5%).
- **Decision rules:**
  - The full stack is **viable** if A4a lies within 3% of A3mx (paired 95% CI upper bound <= +3%).
  - The byte refinement loop **earns its place** if R2's CI lies below 0. If A4a ties A4c, it is unique parameters, not sharing.
  - Throughput and memory are reported next to the results. Speculative-decoding acceptance on A4a is a secondary readout.

### A4 amendment 1 (2026-11-23 ~13:50 EST, before any A4 run; outside review 13:40)
- **Confound found:** A4a (shared r_b = 2) trains with the per-pass auxiliary loss at weight 0.5, and A4b (r_b = 1) cannot use one, so R2 (A4a vs A4b) mixes "second pass" with "deep supervision". A4c matches A4a's aux loss but uses about 4% fewer computed FLOPs (0.963x vs 1.003x A3mx), so R3 is reported with that difference.
- **Added arm:** **A4d** [142c51551569b410], which is A4a with `refine_aux_weight = 0`, seeds 5-7.
- **Re-specified tests:**
  - **R2'** (second shared pass): A4d vs A4b, prediction -1% (range -3% to +0.5%).
  - **R2''** (deep supervision): A4a vs A4d, prediction -0.5% (range -2% to +1%).
  - R1 and the viability rule (A4a within 3% of A3mx) are unchanged.
  - R2 (A4a vs A4b) is still reported, labelled as the combined effect.
- **Multiplicity:** R2', R2'' and R3 form one Holm family.

### A3 amendment 1 (2026-11-23 ~14:10 EST, before these runs; claim audit 2026-11-23-1404): learning-rate bracket for the A arms
- **Why:** statistics policy section 2.3 requires both arms of a VERIFIED training comparison to be tuned or LR-robust. Every A2/A3 arm ran at 6e-3, inherited from E27's bracket (v0.5.13), and A2s/A3mx were never bracketed. C206-C208 are PARTIAL until this is settled.
- **Runs:** A2s and A3mx at lr 3e-3 and 1.2e-2, seed 5, 200M tokens of code-py-v2 (4 runs; configs `a2/a2s.json`, `a3/a3mx.json`), on the A3/A4 fleet.
- **Rule:** an arm is **LR-robust at 6e-3** if its seed-5 clean80 at 6e-3 is within 1% of the best of {3e-3, 6e-3, 1.2e-2}. If both A2s and A3mx pass, C206-C208 return to VERIFIED on their registered results. If either fails, the affected A3 comparisons are re-run at the better LR (seeds 5-7) before any upgrade.
- **Prediction:** both pass (6e-3 within 1% of the best), because E27's family optimum sits at 6e-3 and the A arms share its recipe.

## SQ392 positional control for the long-context result (2026-11-23 ~14:15 EST, before running; outside review 14:09, I-b)
- **Setup:** SQ351's 200M checkpoints of cell 000 (dense, RoPE, trained at 1,024 tokens) and cell 100 (GDN hybrid), 3 seeds, on the same long clean80 files SQ351 used, scored at 2,048 tokens beyond position 1,024. Dense is scored with and without linear RoPE position interpolation (scale 1/2 for 2,048; positions divided by 2); the hybrid is scored as in SQ351. Eval only, no training.
- **Estimand:** the hybrid's advantage over dense past 1,024 tokens (log NLL ratio, paired by seed), without vs with interpolation on the dense arm.
- **Prediction:** interpolation removes about half of the dense arm's post-1,024 degradation, so the advantage shrinks from -60.7% to about -30% (range -45% to -10%).
- **Rule:** if the advantage with interpolation is still beyond -50%, the long-context result stands as stated. If it falls inside -30%, it is restated as "vs unscaled RoPE", and the README's long-context bullet is rewritten before it is cited again.

### A2 amendment 3 (2026-11-23 ~14:50 EST, before running; outside review 14:41, critique 2.1): compute-matched E27 for the A2 headline (C209)
- **Why:** C209's A2s-vs-E27 gap (-16.67%) compares A2s at 3.79e16 training FLOPs with E27 at 3.41-3.43e16 (truncated BPTT). DEVLOG v0.5.10 called it "matched FLOPs", which was wrong (erratum in v0.5.22).
- **Runs:** E27 (`experiments/staged-configs/e27_fact111.json`, lr 6e-3) at **221.6M tokens** (200M x 3.79/3.42, so its training FLOPs match A2s's), seeds 2, 3, 4, paired with A2s at the same seeds (shared data order).
- **Estimand:** A2s - E27(221.6M), log clean80, paired by seed, df 2.
- **Prediction:** -15.5% (range -17% to -13%). The extra 10.8% of tokens is worth about 1-2% to E27 at this horizon.
- **Rule:** C209 gains a compute-matched evidence event if the paired 95% CI lies below -5% (its refutation bound). If the CI reaches -5% or above, C209 is REFUTED by its own criterion and is restated.

### A3 amendment 2 (2026-11-23 ~14:50 EST, before running; outside review 14:41, critique 2.2 / I-6): bracket the full-precision reference A2f
- **Why:** A3 amendment 1 brackets A2s and A3mx only. The ternary cost (A2s vs A2f, A2 P3 +9.5%, used in C207's wording) is fair only if A2f's 6e-3 is also near its optimum.
- **Runs:** A2f (`a2/a2f.json`) at lr 3e-3 and 1.2e-2, seed 5, 200M tokens.
- **Rule:** A2f is LR-robust at 6e-3 if its seed-5 clean80 at 6e-3 is within 1% of the best of {3e-3, 6e-3, 1.2e-2}. If not, the ternary cost is restated at A2f's best LR, and C207 gains an LR limitation.
- **Prediction:** robust (within 1%), as A2s was at 3e-3 (+0.6%).
- **SQ392 protocol clarification (2026-11-23 ~14:50, before running; outside review 14:41, critique 2.4):** both dense arms are built with `max_seq_len = 2048`, so the RoPE cos/sin table covers every scored position. The baseline uses raw positions 0-2047, and the interpolated arm divides positions by 2 before the table lookup. Nothing else differs between the two arms.

## SQ403 eval-time pass count on A2s (2026-11-23 ~15:15 EST, before running; outside review 15:07, I-4)
- **Setup:** A2s 200M checkpoints, seeds 2-4 (trained at fixed r = 2), clean80 doc-level bpc scored at r = 1, 2 and 3 (eval only, the same scorer as `score_doclevel.py`; eval r recorded in the result JSON).
- **Prediction:** r = 1 is at least 5% worse than r = 2 (range +3% to +25%), because the decode bench's teacher-forced argmax agreement of r = 1 with r = 2 is only 0.869. r = 3 is within +-2% of r = 2 (range -1% to +4%), since SQ341/SQ348 found loss best at the trained r.
- **Reading:** a large r = 1 gap shows the second pass is used. It does not show the pass beats unshared depth, because a model trained at r = 2 and run at r = 1 is a distribution shift. SQ401 and A2f vs E24 cover the capacity question.

## SQ400 registered out-of-corpus test of the "costs are memorisation" hypothesis, C213 (2026-11-23 ~15:20 EST, before running the new seeds; written by the side-quest worker at the main agent's request)
- **Question:** SQ379 (seeds 2-4, package-clustered CI only) found the ternary cost (A2s vs A2f: +8-10% on the 300-doc clean80 subset) and the loop/fusion cost shrink or vanish out of corpus, while the assembly gain survives. Is that a property of the models (C213: the costs are mostly in-distribution capacity / memorisation) once seed variance is included?
- **Arms (200M tokens, best LR; seeds where checkpoints exist):** A2s (2-7), A2f (2-7), E20 dense (2-7), E24 GDN (2, 4, 5, 6, 7; seed 3 is missing), E27 full fusion (2-4 only), A3mx (5-7, A3 references A2s/A2f/E20/E24 at the same seeds). The seeds 2-4 scores are re-used from SQ379 (same scorer, same docs); seeds 5-7 are new.
- **Data:** (1) the SQ99 out-of-corpus set (330 local library files, K9-scanner overlap < 20%, sha256-verified), (2) the same files split into stdlib vs site-packages (a held-out-by-source slice; the locked `code-py-v1-test` set is NOT used), (3) the in-corpus clean80 300-doc subset of SQ339 for the same models, as the reference for the shrinkage ratio.
- **Estimand:** per comparison, the effect = mean over paired seeds of 100 x (bpc_arm / bpc_base - 1) on aggregate bits/char, with TWO intervals: the t-interval over the per-seed paired effects (df = n - 1, includes seed variance) and the package-clustered bootstrap over documents (B = 2000, as SQ379). The paired-by-seed interval is the one used for the rules below.
- **Predictions (effects on the SQ99 out-of-corpus set unless stated):**
  - **P1 ternary cost, A2s vs A2f (seeds 2-7):** mean within +-1.5% and the paired-by-seed 95% CI includes 0 (SQ379: +0.22% at n = 3). In-corpus (clean80-300) it stays above +7%.
  - **P2 assembly gain, A2s vs E27 (seeds 2-4):** between -9% and -13%; the paired-by-seed CI excludes -5% (SQ379: -11.3%).
  - **P3 parity with dense, A2f vs E20 (seeds 2-7):** between +0.5% and +3% (SQ379: +1.65%): out of corpus A2f is NOT better than dense, unlike on clean80 (about -1%).
  - **P4 fusion cost, E27 vs E20 (seeds 2-4):** the out-of-corpus effect is 0.4-0.65 of the clean80-300 effect (SQ379: +14.8% vs +28.7%, ratio 0.52).
  - **P5 MoE, A3mx vs E20 (seeds 5-7):** the out-of-corpus effect is smaller in magnitude than the clean80-300 effect, with the same sign (if A3mx is worse than dense in corpus it is less worse out of corpus; if better, less better).
  - **P6 slices:** the ternary-cost estimate (P1) has the same sign on the stdlib slice and the site-packages slice, and neither slice's paired-by-seed CI excludes 0 above +3%.
- **Rule for C213:** SUPPORTED (as an evidence event, not VERIFIED) if P1 holds (CI includes 0, |mean| < 1.5%) AND P4 holds (ratio < 0.65) AND P3 holds. WEAKENED if P1's paired-by-seed CI excludes 0 with a mean of +2% to +5% (the cost shrinks but does not vanish). REFUTED if P1's mean is above +5% with the CI above 0 (the cost is the same out of corpus). Report the table whether or not the rule fires; the CIs do not cover doc-sampling and seed variance jointly, and n differs by comparison.
- **Falsifiers by prediction:** P2 fails if the A2s-vs-E27 gap is above -5% (assembly gain not out of corpus); P3 fails if A2f is below -1% vs dense out of corpus (parity holds there too).
  - **P7 loop cost, A2f vs E24 (seeds 2, 4, 5, 6, 7):** the out-of-corpus paired-by-seed CI includes 0 or favours the looped A2f (SQ379: -0.41% at n = 2; clean80-300: +2.5% to +3.3%).
- **Decision rule (main agent's triage of quick review 15:10, verbatim):** C213 changes design priorities only if, with a paired-by-seed 95% CI over at least 3 seeds per arm, BOTH hold: (1) the out-of-corpus ternary cost (A2s vs A2f) has its CI upper bound below +3% while its clean80 cost stays above +5%; (2) the out-of-corpus loop cost (A2f vs E24) has a CI that includes 0 or favours loops. Otherwise C213 stays a hypothesis.

## GPU batch B1 (2026-11-23 ~16:00 EST, before dispatch; human: "Yes, rent one"): SQ401, SQ402, SQ404 at 20M tokens
All runs use code-py-v2, 20M tokens and one rented RTX 5090. They are exploratory 20M screens (statistics-policy section 4), not confirmatory. Each comparison is paired by seed (shared data order, `gogeta/train.py:61-65`) on clean80 doc-level bpc, log scale, df = n - 1. Configs are in `experiments/staged-configs/b1/` (sha256-16 in brackets); code is v0.5.31 (`core_mlp_hidden`, `gdn_beta_fixed`, `train.core_lr_mult`, all off by default and bit-identical).

### SQ401: is the MoE core's gain capacity or routing? (C206)
- **Arms, seeds 0-2, lr 6e-3:** A2s (`a2/a2s.json`, dense core FFN 1024); A3m (`a3/a3m.json`, ternary MoE core 4 x 512 top-2); **A2s-core2048** [11ad9cba0f7e6fb4] (dense core FFN 2048, the MoE bank's total width; about 2x the core FFN FLOPs). Params and FLOPs are reported per arm.
- **Predictions:**
  - A3m vs A2s = -3% (range -5% to -1%), the C206 direction at 20M;
  - A2s-core2048 vs A3m = -1% (range -3% to +1%), following C203 on the E1 stack.
- **Rule:**
  - If A2s-core2048 is at or below A3m (paired mean <= 0), C206 gains the limitation "at 20M the MoE step is matched or beaten by a dense FFN of the same total width (more FLOPs)", and the MoE's case rests on active FLOPs.
  - If A2s-core2048 is worse than A3m by more than 1%, routing adds something beyond capacity.

### SQ402: does the delta rule's learned write strength matter? (GDN mixer)
- **Arms, seeds 0-1, lr 6e-3:** E1 on v2 (`e53_e1_v2.json`, learned beta = sigmoid(b(x))); **beta fixed at 1.0** [50f7fd633718a8d6]; **beta fixed at 0.5** [bc2dac73445f9c49].
- **Predictions:** fixed 1.0 vs learned = +1.5% (range 0% to +4%); fixed 0.5 vs learned = +1% (range -0.5% to +3%).
- **Rule:** if both fixed arms lie within 0.5% of learned beta, the mixer's gain is attributed to gating and decay rather than the learned delta-rule write strength (exploratory, 2 seeds), and C18/C43's state-tracking motivation gets that caveat.

### SQ404: is part of the loop cost step size? (sharing confound, D3)
- **Arms, seeds 0-1:** A2f (`a2/a2f.json`, lr 6e-3) with core_lr_mult 0.5 [d0458a387e118e8a], 1 (the base config) and 2 [40944c715bf98d50]; and the unshared GDN hybrid E24 (`e24_fact100.json`, its tuned lr 3e-3).
- **Predictions:**
  - A2f (mult 1) vs E24 at 20M = +2% (range 0% to +5%).
  - The best multiplier closes less than half of that gap. Predicted best: 0.5, with the gap shrinking to +1.5% (range 0% to +4%).
- **Rule:**
  - If the best multiplier closes more than half of the 20M gap, the A2f vs E24 comparison (C209 limitation, D3) is re-run at 200M at that multiplier before any loop-cost conclusion.
  - If the best multiplier is 1, or it closes less than half the gap, the loop cost is not a step-size artifact at 20M.

### B1 amendment 1 (2026-11-23 ~16:15 EST, before these runs; outside review 16:07, 2.2): SQ350, parameter-matched loop control
- **Why:** the D3 loop cost (A2f vs E24 +3.2%) mixes looping with the unique-parameter deficit, because A2f's 6-block core run twice has half the unique core parameters of E24's 12 unshared blocks (C197's lever).
- **Added arm:** **A2f at one pass** (`b1/sq350_a2f_r1.json`: 6 core blocks x 1, same unique params as A2f, half its core FLOPs), seeds 0-1, lr 6e-3, 20M tokens. It is paired with B1's SQ404 arms A2f (6 x 2, mult 1) and E24 (12 x 1, lr 3e-3) at the same seeds.
- **Decomposition:**
  - A2f(6x2) - A2f(6x1) is the loop effect at equal unique params;
  - E24(12x1) - A2f(6x2) is the unique-param effect at equal FLOPs.
- **Prediction:** A2f(6x2) vs A2f(6x1) = -4% (range -8% to -1%): the second pass helps at equal params, but less than the doubling of unique params that E24 gets.
- **Rule** (exploratory, 2 seeds):
  - If the loop effect at equal params is at least 3x the size of A2f's remaining gap to E24, loops are a real compute lever and the D3 cost is mainly a unique-parameter cost (C197). C197's wording then gains "at matched FLOPs, not matched params".
  - If the loop effect is under 1%, the second pass does little even at equal params.

### B1 amendment 2 (2026-11-23 ~16:20 EST, before any B1 result; quick review 16:14): multiplicity for the B1 readout
- **Gating contrasts** (each triggers a follow-up or a ledger change) form one Holm family, alpha 0.05. They are reported Holm-adjusted next to the unadjusted estimates:
  - G1, SQ401: A2s-core2048 vs A3m (a C206 limitation);
  - G2, SQ404: best core multiplier vs mult 1, as gap closure against E24 (a 200M re-run);
  - G3, SQ350: A2f(6x2) vs A2f(6x1) (C197 wording).
- **Exploratory readouts**, reported unadjusted and labelled exploratory: SQ401's A3m vs A2s, SQ402's fixed vs learned beta, SQ404's A2f vs E24 gap itself, and every throughput number.
- With 2-3 seeds (df 1-2), every B1 contrast stays at screening strength whatever its p-value. A gate opens only a follow-up run or a limitation, never a VERIFIED status.

### B1 amendment 3 (2026-11-23 ~16:45 EST, before any SQ404 result; outside review 16:37): SQ404's reading, rule unchanged
- **The premise "shared weights take a larger step" is withdrawn.** AdamW (`gogeta/train.py`, fused) normalises each parameter's update by the running RMS of its own gradient, so summing the gradient over r passes does not enlarge the step (only eps differs). The one scale-dependent path, global grad-norm clipping at 1.0, never fires after warm-up. In a log scan of the pulled 200M runs (A2f 8, A2s 8, A3mx 5, E20 5, E24 5, E27 2), no logged step after step 300 had a pre-clip norm above 1.0, and the median norm was 0.11-0.14.
- **What SQ404 still tests:** whether the shared core's best LR differs from the global LR. The 0.5 arm puts only the core (`core.*` parameters) at 3e-3, which is E24's tuned LR, while embeddings, head, inject, prelude and coda stay at 6e-3.
- **The rule is unchanged.** A multiplier that closes more than half the A2f-vs-E24 gap still sends that comparison to a 200M re-run, read as "the core's LR was mis-set", not as "sharing enlarges steps". The readout shows the paired per-seed differences, because at df 1 the rule is judged on point estimates.
