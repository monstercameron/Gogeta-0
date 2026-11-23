# Research: how Inkling 2 Mini loops its transformer (and what Gogeta's E2 lacks)

Written 2026-11-22 by a research subagent (web only: no GPU, no training, no git). Labels follow AGENTS.md §0:
**[doc]** = stated by a Thinking Machines (TML) source, **[code]** = read from the public inference code in
Hugging Face transformers, **[2nd]** = third-party write-up, **[inferred]** = my reading, **[pred]** = prediction.
Nothing here is VERIFIED; every number is "reported by TML" unless marked computed.
Correction: `notes/research-looped.md` §1 says "Inkling 2 is not looped". That holds for **Inkling 2 Small**;
**Inkling 2 Mini** (released two days ago) is looped.

## 1. What it is (found: yes)

- **Lab:** Thinking Machines Lab (TML). **Released** 2026-11-20 (press date) as open weights under Apache 2.0,
  together with a research checkpoint, **TML-d24-varloop-research**, that accepts a variable number of passes.
- **Size [doc]:** multimodal MoE, 26.4B total / 3.4B active parameters. Each extra pass adds about 1.0B effective
  active params, so 3 passes come to about 5.4B per token. Hidden size 2304, 8 of 128 SwiGLU experts per layer
  visit, 1M context. The checkpoint is about 20 GB in NVFP4/MXFP8.
- **Primary sources:**
  - TML announcement: https://thinkingmachines.ai/news/inkling-2-mini/
  - model card: https://huggingface.co/thinkingmachines/Inkling-2-Mini (plus `config.json`)
  - varloop card: https://thinkingmachines.ai/model-card/tml-d24-varloop-research/
  - HF blog: https://huggingface.co/blog/inkling2mini
  - transformers docs: https://huggingface.co/docs/transformers/en/model_doc/inkling2_loop
  - code: `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`
- **Secondary:** Goodfire (goodfire.com/research/interpreting-inkling-2-mini/); press (the-decoder, VentureBeat, unite.ai).
- **No paper or technical report exists.** The training code is not public. Anything about training below comes
  from the blog and model cards only.

## 2. The recurrent design

| Ingredient | Inkling 2 Mini | Source |
|---|---|---|
| Where the loop sits | 24 physical layers in 4 blocks of 6: prelude = layers 0-5 (once), core = 6-17 (looped), coda = 18-23 (once). **Half the unique layers are looped; a quarter on each side are not.** | doc, config (`core_start 6, core_end 18`) |
| Iterations | **Fixed at 3** (`num_passes 3`), giving 48 layer visits. No halting, no adaptive depth. | doc, config |
| Weight sharing | All 3 passes share every core weight, including norms. No per-pass embedding and no per-pass norm. | code |
| Input injection | **Every pass restarts from the prelude output**, both the hidden state and the prelude's block states. The previous pass's residual stream is **not** carried over. | doc, code |
| State carry | Only through **cross-pass messages**. In passes 2 and 3, each core layer builds 4 messages (for its q, k, v and MoE inputs). Each message is a softmax "depth attention" with a learned, zero-initialised query, run over the previous pass's RMS-normalised block outputs and its final hidden state. | code, doc |
| Gate | Per layer and per input (4 per layer), rank 64. `z = W_down[rms(h); rms(m)]`, `f = tanh(W_f z)`, `k = tanh(W_k z)`, and the new input is `(1 − f)·h + k·m`. It runs in fp32. TML says the gate **starts fully closed**, so each pass first behaves like a non-looped model, and the learned message gain then grows during training, which made looping stable to introduce. | code (formula), doc (closed start) |
| Residual stream | **Block Attention Residuals** (Kimi, arXiv 2603.15031). There is no single running sum. Each layer's attention and MLP input is a softmax over the earlier 6-layer block outputs (RMS-normalised, zero-init queries, so a uniform average at init) plus the within-block partial sum. The coda reads the prelude blocks plus the **last** pass's blocks. | code, doc |
| Caches | Each of the 48 visits keeps its own KV cache and short-conv state. | doc, code |
| Other stabilisers (generic, not loop-specific) | qk L2-norm clip, value norm without gamma, learned attention sink, attention output gate, logit softcap 30, MLP softcaps, 5:1 sliding-window:global attention | config |
| Normalisation | RMSNorm (eps 1e-6), an embedding norm, unweighted RMSNorm inside messages and gates. TML discloses nothing loop-specific. | code |
| Initialisation | Only the depth-attention queries are explicitly zero-init in the inference code. The gate's "closed at start" must come from training-time init (e.g. zero `W_f`/`W_k`), which is not public. | code / [inferred] |

**Goodfire's reading [2nd]:**
- In passes 2-3 the messages attend almost entirely to the previous pass's second core block.
- One outlier channel spikes at the end of passes 1-2, apparently signalling "another pass is coming", and goes
  negative in pass 3.
- Logit lens only works near the coda. The passes are therefore *not* interchangeable refinements of one answer.

## 3. Training recipe and the evidence that it works

- **Recipe [doc]:**
  - It comes from TML's mainline scaling ladder. The two changes are expert pruning and adding loops.
  - Training starts with 8 of 384 experts. In midtraining, two-thirds of the experts are pruned (the 128 most
    used are kept).
  - It is then healed on **1T tokens distilled from a larger Inkling**.
  - Post-training is SFT, then RL (2.2T tokens, 2,000 steps), then on-policy distillation.
  - It is trained in NVFP4 from pretraining through RL.
- **Looping during training [doc]:**
  - Looping was introduced "during training" behind the closed gate. Whether that was from step 0 is not stated.
  - The production model always trains at r = 3.
  - **Not disclosed:** truncated vs full BPTT, per-pass losses, any loop curriculum.
- **Varloop model [doc]:**
  - It shares pretraining and midtraining stage 1 with Mini.
  - For the rest of midtraining, r ∈ {1..5} is sampled in a **1:2:6:2:1** ratio (peaked at 3; the in-order
    mapping is [inferred]).
  - Post-training is at r = 3 only.
  - The card warns it may behave unpredictably at r ≠ 3.
- **Evidence for the loop [doc]:**
  - (a) **Position ablation** at training step 2,000, equal training compute, 3 passes:
    - looping the middle 12 layers was worth a 1.51-1.91× compute-equivalent gain over a no-loop baseline;
    - looping the last 12 was worth 1.24-1.36×;
    - middle won 27 of 28 evals.
  - (b) **Unpruned loop vs no-loop** on their ladder, at the same loss on 512k-token agentic traces: the no-loop
    model needed **7.5% less training compute** but **140% more inference compute**. Sizes and the definition of
    "inference compute" are not given.
  - (c) Varloop by pass count (r = 1→5): SWE-bench Verified 56.3 / 76.5 / 78.7 / 79.3 / 79.6, and GPQA 69.6 → 84.2
    at r = 3, then 81.9 and 83.1 (non-monotone).
  - (d) Gains are largest on long-context retrieval (MRCR) and reasoning, while closed-book factual QA barely moves.
    This is consistent with our C42.
  - (e) No-CoT time horizon goes 16 s (r = 1) → 37 s (r = 3), then flat. N-hop lookup keeps improving to r = 10,
    beyond the trained 5.
  - (f) **No ablation of the gate, the messages, restart-vs-carry or attention residuals is published.**
- **[inferred] The key reading for us:** TML's own result (b) says the loop **loses on training FLOPs**. The
  non-looped model reaches the same loss with 7.5% less training compute. The win is parameters, memory and
  inference cost, which is exactly C197's direction.
  - But the deficit is small. Under an assumed loss ∝ C^−0.05, a 7.5% compute gap is about 0.4% loss (computed).
  - E2 is 12.6% worse in bpc than unique blocks at matched FLOPs, and 2.6% worse than its unlooped twin E9.
  - "Working so well" therefore means **the loop almost never costs quality per FLOP**, not that it beats
    unique depth per FLOP. Our loop is roughly 30× further from parity in loss terms.

## 4. Gap analysis vs E2

E2: `[swa] → [gdn, gdn, swa] × r → [attn]`, d = 384, r ~ U{1..7}, BPTT through the last 4, state carry
`h ← core(h + inject([rms h; rms e]))`, pre-norm running residual, 9.46M unique non-emb params, 200M tokens.

| # | Ingredient | Inkling | E2 | Plausibly explains our +2.6% vs E9? |
|---|---|---|---|---|
| G1 | **Closed-gate containment**: at init the looped net *is* its unlooped twin | yes (restart + closed gate means passes 2-3 reproduce pass 1) | no: at init E2 is a 14-visit deep shared net with an ungated, unit-scale inject (`inject.is_residual_out` only scales its init) | **High.** E2 can end up worse than E9; Inkling's design can't start worse. E2's r = 1 loss (0.6694) is 10.7% above E9's (0.6048): the shared weights are a poor 1-pass model (caveat: E9's number is from a 5090 run at lr 6e-3, E2's from v0.3.0 at 3e-3) |
| G2 | **Restart from the prelude each pass** (no residual carry) | yes | no: E2 carries h, and its norm grows about linearly (C39, C59) | **High.** This is the same mechanism as our norm drift and the C7 degradation |
| G3 | Loop count: fixed r, or peaked sampling | fixed 3 (varloop 1:2:6:2:1) | uniform 1..7: about 43% of steps run at r ≤ 3, and r = 4 gets only 1/7 of steps | **Medium-high.** Uniform r spends capacity on r = 1 and on r = 5-7 |
| G4 | Thick outer layers (25% / 50% / 25%) | 6 / 12 / 6 layers | 1 / 3 / 1 blocks (20 / 60 / 20 by unique blocks; at r = 4, 86% of visits are core) | **Medium.** TML and 2610.40325 both find middle-stack loops best at limited FLOPs; our single prelude and coda block must also do the token↔latent "translation" |
| G5 | Per-layer, per-input gated messages (q/k/v/MLP) | 4 per layer, rank 64 | one inject at loop entry only | Medium-low (no TML ablation). Maybe the main source of Inkling's quality once G1/G2 hold |
| G6 | Block attention residuals (bounded, normalised depth-mixing) | yes | plain pre-norm sum | Medium for C7 (scale stays bounded); unknown for C6 |
| G7 | Full BPTT | not disclosed | last 4 of up to 7 | Low at r ≤ 4 (no truncation there); SQ19 covers it |
| G8 | Scale, tokens/param, distillation, pruning | 26B MoE, >10T tokens, 1T distilled | 9.5M, ~21 tokens/param | Real but untestable here (C67, SQ33) |
| G9 | Generic stabilisers (qk clip, sinks, output gate, softcaps) | yes | none in loop | Low (SQ35 covers the output gate) |

**[inferred] Most likely culprits for the 2.6%:** G1 + G2 (we let the loop start, and drift, away from the unlooped
solution), then G3 (uniform r dilution), then G4. G5 and G6 are the "make it better than unlooped" ingredients,
worth testing only once G1-G3 have closed the gap.

## 5. Experiments at our scale (rented RTX 5090: 20M tokens ≈ 40 s at 14M params, 200M ≈ 5 min)

The baseline is E2 (`configs/e2_looped.json`) and the control is E9, both at 20M tokens with 3 seeds unless
stated. Report clean80 bpc and L(r) for r = 1..12. Report FLOPs from `flops_per_token` (matched params,
compute reported), per AGENTS §5.

1. **SQ266, fixed-r sweep (config only).**
   - **Hypothesis (G3):** uniform U{1..7} dilutes training, and a fixed r recovers most of the gap.
   - **Change:** `loops_min = loops_max = loops = bptt_loops = r` for r ∈ {2, 3, 4}, vs E2 and E9.
   - **Cost:** 5 arms × 3 seeds × about 1 min.
   - **[pred]:** fixed r = 4 is 0.5-1.5% better than E2 at r = 4 and closes about half the gap to E9, without
     flipping it. Fixed r = 3 is within 0.5% of r = 4 per FLOP.
   - **Changes plans if:** fixed r beats E9 by more than 2 pooled sd. Then the sampling is the main culprit:
     make fixed (or peaked) r the default and re-run C6 at 200M.
2. **SQ267, peaked loop sampling.**
   - **Change:** a new `loops_weights` option in `train.py:301`, replacing `loop_rng.randint` with
     `loop_rng.choices(range(loops_min, loops_max + 1), weights)`. Weights 1:2:6:2:1 over r = 2..6 (peak 4), vs
     uniform and vs fixed r = 4.
   - **[pred]:** L(4) within 0.3% of fixed r = 4; L(2) and L(6) within 1% of uniform; X12 = L(12)/L(4) − 1 unchanged.
   - **Changes plans if:** peaked sampling matches fixed r at r = 4 *and* keeps the variable-r range. Then
     varloop's recipe is free and we adopt it for any adaptive-depth work (E5).
3. **SQ268, closed-gate restart loop (the core Inkling ingredient; G1 + G2).**
   - **Change:** a `loop_mode: "restart_gate"` in `FusionDance.forward`. For pass k ≥ 2:
     - `x_k = (1 − tanh(W_f z)) ⊙ e + tanh(W_g z) ⊙ rms(h_{k−1})`, with `z = W_d[rms e; rms h_{k−1}]`;
     - rank 16 (computed: 24.6k params, replacing the 295k-param `inject`);
     - `W_f` and `W_g` zero-init (closed), fp32, then `h_k = core(x_k)`;
     - pass 1 is `h_1 = core(e)`.
   - **Arms:** r = 3 fixed and r ~ U{1..7}.
   - **[pred]:** never worse than E9 by more than 1 sd (containment), and 0-1.5% better at r = 3. X12 at most
     half of E2's (+5.7% computed from 0.6540/0.6186), because the gate can stay shut past the trained r.
   - **Changes plans if:** it ties E9 at 3× core FLOPs. Then loops buy nothing at our scale even when they cannot
     hurt: write the "loop = parameter/memory saver, not quality" decision memo and stop loop-quality tuning.
4. **SQ269, per-block gated messages (G5).**
   - **Change:** on top of SQ268, each core block b gets 2 gates (mixer input, MLP input), mixing in the previous
     pass's input to block b (6 gates in total, about 148k params, computed). Use Inkling's
     `(1 − f)·h + k·m` with closed init.
   - **Cost:** 20M tokens, then 200M if it wins.
   - **[pred]:** a further 0.3-1% over SQ268.
   - **Changes plans if:** it gains less than 0.3%. Then the entry gate is enough; keep the simpler one.
5. **SQ270, depth-softmax (attention-residual) readout at the loop boundary (G6).**
   - **Change:**
     - the coda input becomes `Σ_j softmax(q·rms(s_j)/√d) rms(s_j)` over s ∈ {e, h_1..h_r}, with zero-init q
       (a uniform average at init);
     - a second arm uses the same readout as the core's entry state.
   - **Compare:** SQ15/16/18 (the other bounded-scale fixes) on L(r ≤ 16) and N12.
   - **[pred]:** X12 ≤ +1%, and L(4) within ±0.5% of E2.
   - **Changes plans if:** L(4) improves by at least 1%. Then try Block AttnRes in the unlooped E1 too (a separate
     question).
6. **SQ271, thick outer layers (25/50/25; config only, G4).**
   - **Change:** `prelude [swa, gdn]`, `core [gdn, swa]`, `coda [attn]` at r = 5 (2 + 10 + 1 = 13 visits vs E2's
     14; the same 5 unique blocks).
   - **Second arm:** prelude `[swa]`, core `[gdn, swa]`, coda `[gdn, attn]`.
   - **[pred]:** 0.3-1% better than E2 at about 7% fewer FLOPs (computed from the visit counts).
   - **Changes plans if:** the thick prelude wins by more than 1%. Then the loop needs a real "translator" in front
     of it; rebalance all FusionDance configs.
7. **SQ272, grow-in looping (Inkling's "introduce looping during training").**
   - **Change:** train E9-shaped weights for 100M tokens with SQ268's gate frozen shut, then unfreeze for the last
     100M. Compare with E9 at 200M, SQ268 at 200M, and E2.
   - **[pred]:** within 0.3% of SQ268 from scratch at about 2/3 of its FLOPs.
   - **Changes plans if:** grow-in beats from-scratch looping. Then loop upcycling from a dense checkpoint is our
     cheapest path (and matches TML).
8. **SQ273, containment sanity check (gates SQ268-SQ272).**
   - **Change:** SQ268 with `W_f`/`W_g` frozen at zero. At any r it must be functionally E9 (the passes are
     identical).
   - **Pass criteria:** |L − L_E9| ≤ 1 seed-sd at 20M (3 seeds), and per-pass outputs equal to 1e-6 at init (CPU
     unit test).
   - **Changes plans if:** it fails. Then the implementation leaks state across passes; fix it before reading any
     SQ268 result.

**Top 3:** SQ268 (with SQ273 as its gate), SQ266 (config only, minutes), SQ271 (config only).

## 6. Caveats

- No TML ablation isolates the gate, the messages or restart-vs-carry. The "closed gate made looping stable"
  claim is TML's statement, not a published experiment.
- Inkling's wins are on benchmarks, long context and agentic tasks at 26B, where pruning, distillation and NVFP4
  also act. Our metric is code bpc at 1k context, where TML's data (factual QA flat) suggests loops help least.

## Draft side-quest rows

| SQ266 | **Fixed-r loop sweep** (config only): `loops_min = loops_max = loops = bptt_loops = r`, r ∈ {2, 3, 4} vs E2 (U{1..7}) and E9, 20M tokens, 3 seeds; clean80 + L(r) for r = 1..12, FLOPs reported | Inkling 2 Mini trains at a fixed r = 3; our uniform r puts only 1/7 of steps at the eval r = 4. Overlaps SQ36's fixed-r arm (this adds E9 and the r sweep). Targets C6 (research: notes/research-inkling2-mini.md) | `idea` | ~15 GPU-min (5 arms × 3 seeds); pred: fixed r = 4 is 0.5-1.5% better than E2 |
| SQ267 | **Peaked loop-count sampling**: `loops_weights` option (train.py:301, `random.choices`), 1:2:6:2:1 over r = 2..6 vs uniform vs fixed r = 4 | TML's varloop model trained r ∈ 1..5 in a 1:2:6:2:1 ratio and kept a usable variable depth. Tests whether a peak buys fixed-r quality while keeping adaptive range (E5) (research: notes/research-inkling2-mini.md) | `idea` | ~10 GPU-min; needs a 5-line train.py option |
| SQ268 | **(top pick) Closed-gate restart loop**: each pass restarts from the prelude output e; pass k ≥ 2 input = (1 − tanh(W_f z))·e + tanh(W_g z)·rms(h_{k−1}), z = W_d[rms e; rms h_{k−1}], rank 16, W_f/W_g zero-init; replaces `inject`; r = 3 fixed and U{1..7} | Inkling 2 Mini's loop: at init the looped net equals its unlooped twin, so it can't start worse than E9; restart also removes our state-norm drift (C39/C59). Targets C6, C7, C197 (research: notes/research-inkling2-mini.md) | `idea` | ~5 GPU-min at 20M, then 200M; gated by SQ273; pred: ≥ E9 − 1 sd, X12 ≤ half of E2's |
| SQ269 | **Per-block gated messages**: on SQ268, gate each core block's mixer and MLP inputs with the previous pass's input to that block, (1 − f)·h + k·m, closed init (6 gates, ~148k params) | Inkling uses 4 rank-64 gates per looped layer (q/k/v/MoE inputs) fed by depth-attention over the previous pass; tests whether per-layer messages add to the entry gate (research: notes/research-inkling2-mini.md) | `idea` | ~5 GPU-min after SQ268; pred: +0.3-1% over SQ268 |
| SQ270 | **Depth-softmax readout at the loop boundary**: coda input = softmax over rms(e, h_1..h_r) with a zero-init query (uniform at init); arm 2 uses it as the core's entry state too | Inkling replaces the running residual with Block Attention Residuals (2603.15031), which keeps scale bounded across passes; a bounded-scale alternative to SQ15/16/18. Targets C7 (research: notes/research-inkling2-mini.md) | `idea` | ~5 GPU-min; pred: X12 ≤ +1%, L(4) ±0.5% |
| SQ271 | **Thick outer layers** (config only): prelude [swa, gdn], core [gdn, swa] × 5, coda [attn] (13 visits) and prelude [swa], core [gdn, swa] × 5, coda [gdn, attn], vs E2 (14 visits) | Inkling loops the middle 50% with 25% on each side; TML (middle beat last on 27/28 evals) and 2610.40325 find middle-stack loops best at limited FLOPs. Ours is 1/3/1 (research: notes/research-inkling2-mini.md) | `idea` | ~5 GPU-min; pred: 0.3-1% better than E2 at ~7% fewer FLOPs |
| SQ272 | **Grow-in looping**: SQ268's model with the gate frozen shut for 100M tokens (= E9), then unfrozen for the last 100M; vs E9, SQ268 and E2 at 200M | TML introduced looping during training behind a closed gate; if upcycling works, a dense checkpoint becomes a cheap loop start (research: notes/research-inkling2-mini.md) | `idea` | ~15 GPU-min (3 × 200M); after SQ268 wins at 20M |
| SQ273 | **Containment sanity check**: SQ268 with W_f/W_g frozen at zero must equal E9 (per-pass outputs identical to 1e-6 at init in a CPU test; \|L − L_E9\| ≤ 1 seed-sd at 20M) | Without it, an SQ268 win or loss can't be attributed to the gate (a state leak between passes would confound it) (research: notes/research-inkling2-mini.md) | `idea` | CPU unit test + ~3 GPU-min |
