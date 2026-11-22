# Research side quest: looped / recurrent-depth transformers, 2025–2026

Written 2026-11-22 by a research subagent (CPU only, no training run). Everything below is either
**reported by <paper>** (quoted, not adopted, AGENTS.md §0), **computed** from files on disk, or **predicted**.
Nothing here is `VERIFIED`. Proposed ledger rows are C30–C42 in `experiments/claims.md`, and the side quests are
SQ14–SQ21 in `experiments/side-quests.md` (both in "pending review" sections; SQ22 = existing SQ35).

Papers were read from the PDFs (pdftotext): method, main tables, ablations and limitations. arXiv IDs, titles
and authors were checked against the arXiv API. The 12 added papers are in `refs/11-looped-2026/`
(`refs/fetch_refs.py`).

---

## 0. TL;DR

1. **Our E2 symptom is the norm, not a small-scale fluke.** Ouro (1.4B/2.6B, 7.7T tokens) also peaks at its
   trained T=4 and degrades at T=5–8. Parcae and FLT also report that test-time looping saturates or fluctuates
   beyond the trained depth. Only models with explicitly stabilised dynamics plateau instead of degrading.
2. **K6 already holds a sharp clue (computed here from `experiments/checks/results/K6.json`).** From k≥4, each
   loop adds an increment of almost constant size (‖Δ_k‖ ≈ 4.6–5.0·‖h_0‖) that is nearly **parallel to the state
   itself**: cos(Δ_k, h_{k−1}) = 0.95 → 0.97. So the loop mostly *inflates h along its own direction* (linear
   norm growth, relative change ≈ 1/k). That is what Loop-µP predicts for shared-weight increments, and what
   Parcae calls a marginally stable injection (ρ(A) ≥ 1). A plausible mechanism for the rising loss: the coda is
   a pre-norm residual block, so as ‖h‖ grows, its fixed-size update shrinks relative to h ("coda drowning").
   This can be tested in minutes without training (SQ14).
3. **Prior art is closer than we thought.** LT2 (arXiv 2605.20670) already publishes looped Gated DeltaNet +
   sparse/window attention (even a "Looped GDN+Window" variant) at 0.6B/1.3B. CHASE (2607.10110) publishes looped
   Mamba-2 and looped Mamba–Transformer hybrids. **I found no paper that combines looping with ternary
   weights** (arXiv searches listed in §3). FusionDance's remaining distinct pieces are ternary, the
   prelude/coda split with the only full-attention layer outside the loop, and the Rival Loop.
4. **First runs I'd pick:** SQ14 (inference-only drift/rescale test on the E2 checkpoint), SQ15 (Parcae-style
   contractive injection) and SQ16 (1/r step-size residual scaling). All three target the drift that K6 shows.

---

## 1. "Inkling 2": resolved, and it is not a looped model

Searches: web `"Inkling 2" model`; `Inkling looped language model recurrent depth`;
`Thinking Machines Inkling architecture looped OR recurrent OR "shared layers"`; `"Ouro 2"`; arXiv API title
search `ti:Inkling`, `abs:"Inkling 2"` (no ML hits).

- **Identified:** *Inkling 2 Small* by **Thinking Machines Lab**, announced 2026-11-05 (weights on Hugging Face
  per a 2026-11-12 update on the announcement page). From the official model card: 594B total / 17B active MoE,
  48-layer decoder-only, 8 of 384 experts routed, "a hybrid of local and global" attention, up to 1M context,
  pretrained, midtrained and post-trained natively in **NVFP4** (4-bit), multimodal. "Variable effort" is
  trained, not architectural: the announcement says they "randomly sampled and conditioned on an effort level,
  and applied a length penalty in the reward function".
- The earlier *Inkling* (secondary sources and a Hugging Face mirror, `NarsAI/Inkling-2`, which says it is
  "Duplicated from thinkingmachines/Inkling"): 66 layers, 975B total / 41B active, local/global attention
  (secondary sources say 5 sliding-window layers per 1 global).
- **Not looped:** neither the model card nor the announcement mentions recurrent depth, weight sharing or loops.
  There is no arXiv paper.
- **Tangential relevance:** (a) training natively in a 4-bit float format at frontier scale is a data point for
  low-bit training, but NVFP4 is not ternary; (b) "sample an effort level and condition on it" is the same
  pattern as LoopFormer's budget conditioning, applied to CoT length instead of loop count; (c) the 5:1
  sliding-window:global layout, if the secondary sources are right, resembles our SWA-heavy E1/E2 layouts.
- "Ouro 2" doesn't exist as a name. Search hits were Ouro-2.6B (a parameter count).

---

## 2. Papers

Format: what it is → what they **report** → relevance to FusionDance / E2 → prior-art overlap.
Our E2 facts: loss vs eval loops r1 0.669, r2 0.627, **r4 0.619**, r6 0.624, r8 0.633, r12 0.654; train
r ~ U{1..7}, bptt last 4; max grad norm 0.67 vs 0.38 non-looped. K6: norm ratio ‖h_k‖/‖h_0‖ = 22.4 at k=4,
58.9 at k=12; relative change 0.28 → 0.085.

Our loop, for reference (`gogeta/model.py`): `e = prelude(embed(x)); h = e; repeat r: h = h + W·[norm(h); norm(e)];
h = core(h)`. Core blocks are pre-norm residual `[GDN, GDN, SWA]`; no loop-boundary norm; GDN state restarts
every loop.

### 2.1 Ouro / LoopLM: arXiv 2510.25741
*Scaling Latent Reasoning via Looped Language Models.* Zhu, Wang, Hua, Zhang, Li, … Bengio, Eshraghian.
ByteDance Seed + UCSC, Princeton, Mila and others. v1 2025-10-29 (v6 2026-10-11).

- **Idea:** the whole decoder stack is looped (no prelude/coda), with a fixed T_max = 4. An exit gate at every
  step gives p(t|x). The training loss is the **exit-weighted sum of per-step LM losses** minus β·entropy (a
  uniform prior over exit steps), so every loop gets supervised. Sandwich RMSNorm. 7.7T tokens.
- **Reported:** 1.4B (24 layers) and 2.6B (48 layers, upcycled) at T=4 "match the results of up to 12B" models.
  §4 (stability): first runs with **8 recurrent steps had loss spikes and gradient oscillations, so they reduced
  to 4**; they also grew the batch from 4M to 8M tokens and lowered β from 0.1 to 0.05 for stability.
  §5.3 (depth sweep, trained at T=4, evaluated at T=1..8): base models "moderate degradation when
  extrapolating". Ouro-1.4B-Thinking OlympiadBench T4 71.55, T5 72.30, T8 66.81; AIME24 T4 65.0, T8 38.7.
  2.6B-Thinking OlympiadBench T4 76.4, T8 39.3. Safety improves with T even past 4.
- **Relevance:** the **same inverse-U as E2**, at 100–200× our parameters and ~40,000× our tokens, and *despite*
  per-step supervision. So (i) C7 is probably not a small-scale artifact; (ii) per-loop supervision, at least in
  Ouro's exit-weighted form, does not by itself buy extrapolation past the trained max (relevant to SQ5/E6).
  Ouro renormalises the state at each loop boundary (per 2610.31943, §2.11 below).
- **Prior art:** loop only; full attention, full precision.

### 2.2 LT2: Linear-Time Looped Transformers: arXiv 2605.20670 (**closest prior art**)
Deng, Zhang, Zhu, Xu, Liu, Ng, Chen. Rice, Apple, UCSC, CMU. 2026-05-20. Code and an Ouro-hybrid-1.4B
checkpoint are released.

- **Idea:** replace softmax attention inside an Ouro-style looped model (everything looped, fixed T=4) with
  linear (GDN, KDA, Mamba2, DeltaNet, RetNet, HGRN2) or sparse (window, NSA, DSA) mixers, and hybrids. They add a
  **zero-initialised per-channel, per-loop residual gate** h(τ) = h̃(τ) + ρ_τ ⊙ h(τ−1), and a sigmoid **SDPA
  output gate** in softmax blocks. Theory: T loops of a DPLR (delta-rule) mixer give a rank-T effective state
  transition, but only if the loop-specific keys differ (identical keys give no gain). T loops of a window-w
  block reach as far as T stacked window layers.
- **Reported** (0.6B/1.3B, 100B FineWeb-Edu tokens, T=4): at 1.3B, Looped Transformer avg 59.27 (PPL 9.87);
  Looped GDN 59.92 (9.75); Looped Hybrid (GDN+DSA) 60.73 (9.50); **Looped Hybrid (Full+GDN, 1:4)** 62.89 (9.12) in
  Table 2. The text says "61.39 vs. 59.27"; **this is inconsistent within the paper**. Decode speedup ×2.7
  (§3.1; the intro says ∼5×). Full:GDN ratio sweep gives an inverse-U with the optimum at 1:4; spreading attention
  along depth beats concentrating it. **§3.5 stability:** Looped GDN has "the smoothest loss curve" and a lower
  grad norm than the looped Transformer for "essentially the entire run". Looped RetNet diverges. Mamba2 and
  DeltaNet are "noticeably better but still display occasional spikes". Their attribution: you need both a
  data-dependent gate and the delta rule. §3.4: the attention sink "intensifies across loops" (sawtooth) and max
  FFN activation grows ~20× by the end of loop 4. The output gate gives Looped Transformer PPL 9.87 → 9.39.
  Synthetic state-tracking+recall (4 layers, d=256): Looped Transformer caps at stage 4 (n_max=64), while Looped
  NSA, GDN+Window and GDN+NSA reach stage 5 (n_max=128). Limitations: no cross-loop recurrent-state carry; no
  full loop-level hybridisation.
- **Relevance:** supports our mixer choice (GDN is the stablest linear mixer in a loop), our 1-attention-per-~4
  ratio, and C18 (looping helps sub-quadratic mixers more than softmax on state tracking). I found **no
  evaluation at loop counts different from the trained T** in the main text, so it says nothing about our
  extrapolation failure. The SDPA output gate is a cheap add for our in-loop SWA (SQ22, overlaps SQ8).
- **Prior-art overlap:** high. Loop + GDN + sparse/window attention is published, at 1.3B. Differences from
  FusionDance: (1) LT2 loops the *whole* stack; we use prelude/core/coda, with the only full attention *outside*
  the loop (their Full+GDN keeps full attention *inside* the loop, so it pays O(n²) T times); (2) no ternary;
  (3) fixed T, no extrapolation study; (4) no critic or Rival Loop.

### 2.3 CHASE (Looped Mamba): arXiv 2607.10110
*CHASE: Cache-Hole-Adapted Skip Exit for Looped State-Space Language Models.* Yu, Kojima, Matsuo, Iwasawa.
Rikkyo, U. Tokyo. 2026-07 (v2 2026-09-02).

- **Idea:** loop shared Mamba-2 (or Mamba–Transformer hybrid) blocks. Then make Ouro's exit gate usable on an
  SSM: exiting early leaves deeper loops' recurrent states missing for later tokens (a "cache hole").
  Continued training aligns with skipped-state inference.
- **Reported** (50M–370M): iso-parameter looping lowers PPL (Mamba-2 6⊗4: 19.98 vs 6⊗1: 21.96), but under
  **strict iso-FLOPs the deeper non-looped 24⊗1 model has the best validation PPL** (Mamba-2 12⊗1: 18.66 vs
  12⊗2: 18.21 shows the gap shrinking). The stock Ouro gate with skipping blows up PPL (by up to +4788%); CHASE
  stays within +3–4% while executing about half the loops.
- **Relevance:** independent confirmation of C6's shape (loops lose at iso-FLOP in LM loss). Directly relevant to
  E5 (adaptive loops) and E8 (self-speculative decoding): **early exit in a recurrent-state mixer leaves holes**
  that must be trained for. Our GDN layers have exactly this issue.
- **Prior art:** loop + SSM published. No ternary.

### 2.4 Parcae: arXiv 2604.12946 (**most useful for our failure**)
*Parcae: Scaling Laws For Stable Looped Language Models.* Prairie, Novack, Berg-Kirkpatrick, Fu. UCSD, Together
AI. 2026-04-14.

- **Idea:** view the loop as a dynamical system over the residual stream, h_{t+1} = A h_t + B e + R(h_t, e).
  Linearise (drop R): additive injection gives A = I (ρ = 1, marginally stable); concat-projection injection
  (Huginn, and **ours**) gives an unconstrained A = W1. Divergent runs learn ρ(A) ≥ 1. Fix: **A = exp(Δ ⊙
  −exp(log A)), diagonal, so ρ(A) < 1**; B = ΔB unconstrained; **normalise the prelude output** (e = LN(P(s))).
  Also: sample the depth **per sequence** (Poisson, mean µ_rec), decouple the depth distribution from the
  backprop window, and use µ_bwd = ⌈µ_rec/2⌉.
- **Reported:** up to 6.3% lower val PPL than prior looped models (RDM/Huginn setup). Table 2: the baseline
  converges only at LR 2e-4; Parcae converges up to 1e-3. Table 4 (140M retrofit of a Transformer): RDM
  diverges; +constrained A gives val loss T1/T4/T8 = 8.99/3.15/2.97; +per-sequence sampling 3.38/3.01/2.98;
  +prelude norm 3.28/2.97/2.95. At 1.3B: +2.99 CORE / +1.18 Core-Extended over a parameter- and data-matched
  Transformer. §5.3: test-time looping follows **L(T) = L∞ + Z·e^{−zT} (monotone, saturating)**; "gains plateau
  near µ_rec, suggesting training depth determines the test-time scaling ceiling". App. H: Geiping's
  sampling (depth truncated by µ_bwd) hurts val loss at T = 4, 16, 64 vs their decoupled sampling. App. J: a
  1.3B run had a state explosion at the *first* injection; the prelude norm fixed it.
- **Relevance:** our injection `h + W[norm h; norm e]` is the unconstrained/marginal case. K6's linearly growing
  norm (§0) is the signature of ρ ≈ 1 plus a constant drive. Parcae's prediction for a contractive A is that
  extra loops **plateau instead of degrade**. That is SQ15. Our sampler already decouples r from bptt (good),
  but samples per *batch*, not per sequence (SQ20).
- **Prior art:** loop only; full-attention blocks.

### 2.5 Loop-µP: arXiv 2606.18524
*Loop-µP: Residual Scaling and Hyperparameter Transfer in Looped Transformers.* Wang, Li, Zhu, Zhang, Huang,
Yan, Li. Tsinghua, ByteDance Seed, M-A-P. 2026-06-16 (v3 2026-10-18).

- **Idea:** distinct layers add nearly orthogonal increments (norm ~ √L). A shared layer adds **positively
  correlated** increments every loop, so the norm grows ~ N. The unique stable-and-maximal-update choice is a
  residual multiplier of **1/N** (combined with Depth-µP's L^{−1/2}), with LR independent of N.
- **Reported:** FineWeb-Edu 10B tokens, L ∈ {12, 24, 48}, N ∈ {1, 2, 4, 8}. Under 1/N the best base LR stays
  1.25e-3 for all N and reaches the lowest loss. Unscaled, the optimum falls to 7.5e-4 at N=8 and LR ≥ 3e-3
  diverges at N ≥ 4. After training, pass increments stay positively correlated (Fig. 5).
- **Relevance:** explains both E2's larger grad norm (the same weight update enters all r passes coherently; the
  paper's one-step change scales as η·N^{1−α}) and K6's linear norm growth. **Caveat:** Loop-µP trains at a
  fixed N and does not test extrapolation. Using a per-call 1/r with variable r (SQ16) is *our* synthesis with
  LoopFormer's step-size view, not a claim of either paper.
- **Prior art:** loop only.

### 2.6 Fully Looped Transformer (FLT): arXiv 2605.18797
*Simply Stabilizing the Loop via Fully Looped Transformer.* Fu, Yang, Zhang, Ma, Chen, Li, Chang. HKBU, Jilin.
2026-05-11.

- **Idea:** two parameter-free changes: (1) feed the previous loop's output to *every* layer, not just the first;
  (2) "Attention Injection": for loops t > 1, attention uses the previous loop's output as **Q** and the current
  stream as K/V (reusing the same projections), so the recurrent signal enters through a softmax-normalised
  mixture instead of being added.
- **Reported:** 127M (6 layers) / 318M (12 layers), 20 tokens/param, FineWeb-Edu. Base size: plain LT avg 40.52
  (3 loops), 36.56 (6), collapses at 9. LT with input injection 41.54 / 40.20 / 41.00. FLT 39.90 / 41.38 /
  41.72. Only FLT trains at 12 loops. Beyond the trained loop count, "performance begins to fluctuate". Single
  run per config (their limitation).
- **Relevance:** names our two symptoms (residual growth + oscillating gradients). Input injection alone already
  fixes most of the degradation at ≤ 9 loops; FLT's gain over it is small and single-seed. The Q-injection idea
  is softmax-specific. A GDN analogue (previous-loop state as the GDN *query* only) is a speculative twist, not
  tested by anyone. Low priority.

### 2.7 Hyperloop Transformers: arXiv 2604.21254
Zeitoun, Torroba-Hennigen, Kim. MIT. 2026-04-23 (v3 2026-07-02).

- **Idea:** middle-cycle looping (begin/middle×3/end) plus **loop-level hyper-connections**: n=4 parallel
  residual streams, mixed after each loop by an input-dependent *diagonal sigmoid* matrix, plus a loop position
  embedding. The parameters are loop-specific (cheap: +150–300K).
- **Reported:** about 50% fewer parameters than depth-matched Transformers and better PPL at 136M / 580M / 990M
  (12.5B / 50B / 100B tokens), e.g. 990M Hyperloop 8.49 vs 2.0B Transformer 8.60. Diminishing returns past 3
  loops and past n=4 streams. **INT4 GPTQ (computed from Table 1): plain looped models lose +2.2%, +2.2%, +3.3%
  PPL vs Transformers +1.4%, +0.8%, +1.3%; Hyperloop +1.9%, +1.7%, +1.2%.**
- **Relevance:** (a) consistent with the Iso-Depth paper (§2.10: hyper-connections raise loop value) and with
  our SQ10 idea. (b) **Ternary warning:** loops appear to amplify quantisation error (the same weights' error
  enters every pass). That is the E3/E4 fusion risk, and it is testable post-hoc on E2 (SQ21). (c) Loop-specific
  parameters and position embeddings make extrapolation impossible as designed (the authors note sharing them
  "to enable generalization to longer loops" as future work).

### 2.8 LoopFormer: arXiv 2602.11451
*Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation.* Jeddi, Ciccone, Taati.
Toronto, Vector. ICLR 2026. 2026-02-11.

- **Idea:** condition every loop on normalised time t and step size Δt (sin-cos embeddings → adaLN-style scales
  and gates, DiT-like). Train the full trajectory (L steps) plus a sampled shorter one (S ~ U{1..L−1}, random
  step schedule summing to 1). Loss = L_L + 0.1·L_S + 0.1·L_cons, where L_cons is a **stop-gradient consistency
  loss pulling short-trajectory logits towards the full trajectory's**.
- **Reported:** ~1B-compute comparisons on The Pile (25B tokens), k ∈ {1, 2, 3} blocks, L ∈ {8, 12, 24}. Looped
  models still trail non-looped on PPL; LoopFormer "closes much of this gap". Naive early-exit looped baselines
  "stagnate" (flat curvature/entropy, high cross-step CKA). For a fixed budget, schedules with large early steps
  and fine late steps are best (spread ~1.4 PPL).
- **Relevance:** the consistency loss is a cheap per-loop-supervision variant (SQ17). It only supports
  **M ≤ L**, so it doesn't address extrapolation. The step-size framing motivates SQ16.

### 2.9 Learned Stochastic Stopping: arXiv 2606.29983
*Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping.* Kuo, Chayti, Reizinger,
Brendel, Jaggi. EPFL, MPI-IS/ELLIS. 2026-06-29 (v2 2026-10-14). Algorithmic tasks only.

- **Idea:** looping creates a *selection problem*: some training runs find extrapolating computations and others
  don't. Randomising the train loop count (K = n ± 5) or learning a stopping distribution (REINFORCE
  "RL-Halting", one sampled depth per example) stabilises it.
- **Reported:** 3-layer shared block, train lengths n < 20, test 20–60. Across-run OOD std is 0.009 / 0.005 for
  3- and 60-layer Transformers vs 0.087 (K=20) and 0.122 (K=n) for looped models. Addition, OOD %/std:
  K=20 34.2/7.4; K=n±5 30.9/5.9; RL-Halt 45.0/2.7. "Low variance and high extrapolation accuracy remain distinct
  objectives". Explicit multi-depth loss averaging also reduces variance (their App. C.3).
- **Relevance:** (a) **methodology:** loop-extrapolation results are high-variance across seeds, so C7 (single
  seed) needs ≥ 2–3 seeds before it's VERIFIED. (b) We already randomise r. Their per-example sampling matches
  Parcae's per-sequence sampling (SQ20).

### 2.10 Iso-Depth Scaling Laws: arXiv 2604.21106
*How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models.* Schwethelm, Rückert,
Kaissis. TUM, Imperial, HPI. 2026-04-22 (v4 2026-10-28).

- **Idea:** fit L = E + A(N_once + r^φ·N_rec)^{−α} + B·D^{−β} over r ∈ {1, 2, 4, 8} (~50× compute) with a
  prelude/recur/coda architecture and linear input injection. φ = 1 means a loop is worth a unique block;
  φ = 0 means worthless.
- **Reported:** **φ = 0.46**. At r=4, a 410M looped model ≈ a 580M non-looped one in loss, at the training cost
  of a 1B one. **Truncated BPTT (r_bwd = ⌈r/2⌉) lowers val loss at fixed compute but lowers φ to 0.38**: "the
  loop mechanism is poorly trained under truncation". Loop-boundary hyper-connections (2 lanes, full BPTT) raise
  φ to 0.65.
- **Relevance:** quantifies C6. At matched FLOPs, looping should lose in LM loss. Its case is memory/params (and
  maybe reasoning), and that's how E2 should be framed. Our bptt=4 with r up to 7 truncates 1–3 loops on 3/7 of
  the steps (SQ19).

### 2.11 Agreement rather than normalisation: arXiv 2610.31943
*Agreement rather than normalisation keeps depth reusable in looped language models.* Viakhirev, Borodin,
Abramov, Mkrtchian. lab260 (Yerevan), BitmanagerAI, SPC RAS. 2026-10-22. Small lab, under review. Treat as
medium-confidence.

- **Idea:** Ouro renormalises the state at the end of every loop ("loop-boundary normalisation"); Huginn doesn't.
  Is that norm what makes extra loops usable?
- **Reported:** removing it from the released Ouro-1.4B/2.6B drives accuracy to 0 at the trained budget on a
  binary task, while the state norm stays within 2.021×. Trained from scratch (19M params, 2 synthetic tasks,
  **16 seeds**, r0 = 4 fixed): without the norm the model is "never worse … and extrapolates better" (Cliff's
  δ up to 0.867 beyond the budget). Refitting the head recovers a third of the loss; refitting the last block
  layer recovers all of it. Methodology: comparing looped vs stacked depth "at an equal inference loop count is
  confounded by unequal overshoot of the trained budget".
- **Relevance:** directly contradicts a stronger theoretical claim (Labovich, arXiv 2604.15259, read but not
  added: "recall with outer normalization reliably produces" the stable, input-dependent regime). A cheap,
  genuinely open A/B for us (SQ18). Their overshoot point matters for E9 vs E2.

### 2.12 Adaptive Loops and Memory: arXiv 2603.08391
*Adaptive Loops and Memory in Transformers: Think Harder or Know More?* Frey, Shomali, Bashir, Berghaus,
Koehler, Ali. Lamarr Institute, Fraunhofer IAIS, U. Bonn. ICLR 2026 workshop. 2026-03-09.

- **Idea:** *per-layer* looping (each block iterates up to N_max ∈ {3, 5, 7}) with a PonderNet-style halting
  router (input includes t/N_max). The output is the halting-weighted mixture of iterates. **Learnable per-step
  loop scales softplus(α_t), initialised at α = −7 (≈ identity)**. Gated local/global memory banks.
- **Reported:** 12 layers, ~200M, 14B tokens. Loop-3 math BPB 1.687 vs base 2.163 vs 36-layer iso-FLOP 1.801
  (better), but commonsense BPB 0.813 vs iso-FLOP 0.780 (worse). More loops give small gains (Loop-7 +1.7% over
  Loop-3 on math), and commonsense slightly declines. Memory recovers commonsense.
- **Relevance:** looping helps "algorithmic" tasks and not knowledge, which matches iso-FLOP LM loss being a poor
  yardstick (C6). The near-identity loop-scale init is a cheap stabiliser (folded into SQ16 as a variant).

### Read but not added
- Labovich, *Stability and Generalization in Looped Transformers*, arXiv 2604.15259 (Brown, Liquid AI): a
  fixed-point framework; recall + outer normalisation. Cited under §2.11.
- Yang et al., *Learning to Extrapolate in Depth with Looped Transformers*, arXiv 2610.14456 (UCLA, Princeton):
  theory on graph connectivity. Training at depth K learns an update that balances propagation against fitting
  the residual at K, giving a **finite extrapolation window** in which too many loops "create false
  connections". Deeper K is harder to optimise at a fixed budget (K = 6, 8 stuck on a plateau). A clean
  theoretical story for an inverse-U like E2's.
- Liang, Jin, Pan, *Beyond the Training Horizon*, arXiv 2609.33144: a mechanistic study. Abstract only.
- Shen and Li, *Deeper Backpropagation or More Updates?*, arXiv 2610.38422: backward-window trade-off across
  budgets. Abstract only. Shorter windows lose their advantage as budgets grow.
- Many Oct–Nov 2026 Ouro follow-ups (halting heads, KV cache, speculative decoding: 2610.32246, 2611.02333,
  2611.19896, 2610.21594, …). Not read. Relevant later to E5/E8.

---

## 3. Prior-art overlap (precise)

| Ingredient combo | Published? | Where |
|---|---|---|
| Loop + Gated DeltaNet | yes | LT2 (Looped GDN, 0.6B/1.3B, 100B tokens) |
| Loop + GDN + sliding-window / sparse attention in the loop | yes | LT2 (Looped Hybrid GDN+DSA; "Looped GDN+Window" on synthetic tasks) |
| Loop + GDN + full attention in the loop | yes | LT2 (Full+GDN 1:4) |
| Loop + Mamba-2 / Mamba–Transformer hybrid | yes | CHASE; LT2 (Looped Mamba2); TRM+Mamba-2 (2602.12078, already in refs) |
| Prelude/coda split with the only full attention outside the loop | not found as such | Huginn/Parcae/Hyperloop use prelude/coda but softmax everywhere |
| Loop + ternary weights | **not found** | arXiv API: `all:looped AND all:ternary`, `"recurrent depth" AND ternary`, `looped AND BitNet`, `"weight sharing" AND ternary AND "language model"`. Web: only unreviewed parameter-golf entries and a 3.7M HF toy (Tern-0.1) |
| Loop + quantisation evidence | partial | Hyperloop: INT4 PTQ hurts looped models more |

So the defensible novelty claim is the **full fusion (loop + linear-time mixer + ternary)**, not "looped linear
attention". README and site wording that implies otherwise should cite LT2.

---

## 4. Clues we can test here

**Overlap with the competitor review** (`notes/review-competitors.md`, SQ30–SQ41 / C50–C69, written in parallel):
SQ14 extends **SQ30** (same drift test, plus a rescale-before-coda arm). SQ15 is **SQ31 V3**, and SQ18 is
**SQ31 V1** plus a parity arm, so merge them. SQ20 overlaps **SQ34** (heavy-tailed r). SQ21 complements **SQ38**
(post-hoc on E2 vs trained ternary). SQ22 duplicates **SQ35** and is not added to the ledger as its own row.
Claims C39 and C40 refine **C59**, C37 conflicts with **C58**, and C38 refines **C65**. If both write-ups
are kept, their TINY harness (d=128) can replace H0 below.

**Common harness H0 ("mini-E2")**, so every clue is one ingredient vs one baseline at matched tokens:
d=192, 3 heads, MLP 512, prelude `[swa]`, core `[gdn, gdn, swa]`, coda `[attn]`, window 128, seq 512, batch 32,
AdamW 2e-3, warm-up 100, cosine, 1,500 steps (≈ 25M tokens of `code-py-v1`), r ~ U{1..7}, bptt 4, seed 0 (+ seed 1
for any result we record). Runtime (estimated, not measured): ~3–6 min per arm. **Validity gate:** the
baseline must reproduce the E2 shape, X12 := L(12)/L(4) − 1 ≥ +1%. Otherwise shrink less or fall back to
the K5 parity task (train r ~ U{1..7}, eval r = 1..16).

**Metrics (all clues):** L(r) for r ∈ {1, 2, 3, 4, 6, 8, 12, 16} on 64 fixed val sequences; **X12** and **X16**
(extrapolation penalties vs r=4; E2: X12 = +5.7%); best r; K6's rel change at k=12 and **N12 :=
‖h_12‖/‖h_4‖** (E2: 0.085 and 2.63, computed from K6.json); the new **increment alignment**
cos(Δ_k, h_{k−1}); pre-clip grad-norm p99.9 and max.

Ranked by (information per minute) × (relevance to C7/C20):

### SQ14 (TOP PICK): drift and "coda drowning", inference only on the E2 checkpoint (~2 min)
Supports: Loop-µP (aligned increments), Parcae (marginal stability), K6. No training.
- **Change:** none to training. Extend K6: per-token cos(Δ_k, h_{k−1}), per-token ‖Δ_k‖, and two eval-time
  interventions: (a) **rescale before the coda**: x = h_k · ‖h_4‖/‖h_k‖ (per token, with ‖h_4‖ from the same
  forward pass) before running the coda; (b) rescale after every loop k > 4 to the per-token ‖h_4‖.
- **Prediction (committed):** median per-token cos(Δ_k, h_{k−1}) ≥ 0.8 for k ≥ 4 (the global value is
  0.95–0.97). (a) recovers **≥ 50%** of the gap L(12) − L(4).
- **Pass (C41 "coda drowning"):** L_a(12) ≤ L(12) − 0.5·(L(12) − L(4)). If (a) fails but (b) passes, the
  damage is inside the core (the scale seen by the core's residuals). If both fail, it's the direction, not the
  scale: go to SQ15/SQ17.
- **Why first:** minutes, no training, and it tells us whether the cure is "bound the norm" (SQ15/16/18) or
  "change what the loop computes" (SQ17).

### SQ15 (TOP PICK): Parcae-style contractive injection (H0, 2 arms, ~10 min)
Supports: Parcae §3–4 (+ Table 4's prelude norm).
- **Change** (`FusionDance.forward`, loop body): replace `h = h + inject(cat(norm_h(h), norm_e(e)))` with
  `h = a * h + B(norm_e(e))`, where `a = exp(-softplus(dt) * exp(A_log))` is a per-channel vector in (0, 1)
  and `B = Linear(d, d)` (residual-out init). Initialise A_log/dt so a ∈ ~[0.5, 0.9]. The paper's text gives no
  init, so check their code; this is our choice. Then `h = core(h)` as now. e is already normed via `norm_e`
  (= Parcae's prelude norm). Keep h0 = e (Parcae uses random h0; that would be a second ingredient).
- **Prediction:** N12 ≤ 1.5 (baseline ≈ 2.6); X12 ≤ +1.5%; L(4) within ±1% of baseline; grad-norm max lower.
- **Pass:** X12 ≤ 0.5 × baseline X12 **and** L(4) ≤ 1.01 × baseline L(4).

### SQ16 (TOP PICK): 1/r step-size residual scaling (H0, 2–3 arms, ~15 min)
Supports: Loop-µP (1/N branch scaling, positively correlated increments), LoopFormer (step size Δ=1/M), Frey
(loop scale initialised near identity). The extrapolation logic is *our synthesis*, not theirs.
- **Change:** multiply every core-block branch output (mixer and MLP) **and** the inject output by
  s = 4/r, where r is the loop count of this forward call (train and eval). r=4 is identical to the baseline.
  More loops then mean *finer steps over the same depth-time*, so extrapolation becomes refinement. Variant
  (optional third arm): a learnable scalar loop scale softplus(α)·s with α initialised so softplus(α) ≈ 1.
- **Prediction:** L(r) non-increasing for r ≥ 4 up to 16 (within 0.2% noise); L(4) within +1.5% of baseline;
  L(1) worse than baseline (4× steps); grad-norm p99.9 lower than baseline. Risk: r=1 training steps at s=4 may
  be unstable. If so, report it; don't silently switch to U{2..7}.
- **Pass:** L(12) ≤ 1.002·L(4) **and** L(16) ≤ 1.005·L(4) **and** L(4) ≤ 1.015 × baseline L(4).

### SQ17: per-loop consistency supervision (LoopFormer / Ouro); tests E6's premise (H0, 2 arms)
- **Change:** loss = CE(r) + 0.1·CE(s) + 0.1·KL(stopgrad(p_r) ‖ p_s), with s ~ U{1..r−1} (skip when r=1) and
  p_s from running the coda + head on h_s. Cost: +1 coda/head pass.
- **Prediction (Ouro-informed):** L(1) and L(2) improve by ≥ 2%, but **X12 is not halved**. In other words,
  per-loop supervision does not fix extrapolation beyond the trained max (supporting C34).
- **Pass for C34 locally:** X12 ≥ 0.5 × baseline X12 while L(2) improves ≥ 2%. If X12 halves, C34 is refuted
  here and E6 gets stronger.

### SQ18: loop-boundary RMSNorm (Ouro vs the Agreement paper vs Labovich) (H0 + K5 parity, 2 arms each)
- **Change:** after each loop, `h = norm_loop(h)` (one shared learnable RMSNorm). The coda then sees a normed
  state.
- **Prediction:** N12 ≈ 1 by construction. On the LM, X12 ≤ 0.5 × baseline (we side with "bounding the norm
  helps" because of SQ14's hypothesis). On parity (16 positions past 64, eval r = 1..16), no worse than the
  baseline. Low confidence: the papers disagree.
- **Pass:** LM X12 ≤ 0.5 × baseline **and** L(4) ≤ 1.01 × baseline. Report the parity accuracy curve either way.

### SQ19: full vs truncated BPTT (Iso-Depth §5.1, Parcae App. H) (H0, 2 arms)
- **Change:** `bptt_loops = 7` (full; never truncates) vs 4. Report at matched tokens and note the extra
  backward compute (estimate +~20–30%).
- **Prediction:** L(6) and L(8) improve 0.3–1%; L(4) roughly unchanged; X12 within ±1 pt of baseline.
- **Pass:** L(6) and L(8) both ≥ 0.3% better at matched tokens.

### SQ20: per-sequence Poisson loop sampling (Parcae §4.2 / Alg. 2; Stochastic Stopping §4.2) (H0, 2 arms)
- **Change:** r_i = clip(1 + Poisson(3), 1, 12) per sequence (mean ≈ 4, matched), run r_max per batch and let
  sequence i skip (identity) the first r_max − r_i loops, keeping bptt on the last 4 (Parcae's Algorithm 2).
  Baseline: per-batch U{1..7}. Report compute (≈ r_max per batch, so higher) alongside the results.
- **Prediction:** L(8) and L(12) improve ≥ 0.5% (the tails are seen in training); fewer loss spikes; L(4)
  within ±0.5%.
- **Pass:** both L(8) and L(12) ≥ 0.5% better and L(4) ≤ 1.005 × baseline.

### SQ21: do loops amplify quantisation error? Post-hoc on the E2 checkpoint (~3 min, inference only)
Supports: Hyperloop Table 1 (INT4 PTQ hurts looped models ~1.6–2.6× more). Relevant to the E3/E4 fusion bet.
- **Change:** fake-quantise only the core's linear weights (per-output-channel absmax INT8 and INT4; also our
  BitLinear absmean ternary rounding without retraining, expected to be catastrophic, for reference). Measure
  ΔL(r) = L_q(r) − L_fp(r) for r = 1..12.
- **Prediction:** ΔL grows with r for INT4: ΔL(8) ≥ 1.5 × ΔL(2).
- **Pass (C37 locally):** that ratio ≥ 1.5 for INT4. If it holds, ternary training (QAT) of the loop is riskier
  than ternary training of E1, and K4-style ternary-gap checks should include a looped arm.

### SQ22 (= SQ35 in the ledger, not added separately): attention sink across loops + sigmoid output gate on the in-loop SWA (LT2 §3.4) (low priority)
- **Diagnostic** (E2 checkpoint): per loop k, the attention mass on the earliest key in each window of the
  in-loop SWA, and the max |activation| of the core MLPs. LT2 predicts a sawtooth that grows each loop.
- **Arm** (H0): `o = o * sigmoid(W_g x)` before the SWA output projection (overlaps SQ8).
- **Prediction:** sink mass and max activation increase with k on E2; the gate lowers L(4) by ≥ 0.5% and cuts
  X12 by ≥ 25%.

**Run first: SQ14, then SQ15 and SQ16.** SQ14 is free and decides between the two families of fixes (bound the
state vs change the computation). SQ15 and SQ16 are the two most direct, one-ingredient fixes for the measured
drift, each with a strong, falsifiable prediction (loss plateaus or improves past r=4). SQ17 is the cheap test
of whether E6's per-loop supervision will fix C7, and Ouro predicts it won't.

---

## 5. Open questions and surprises

1. **Surprise: E2's loop is mostly inflating h along itself** (computed from K6.json): for k ≥ 4,
   cos(Δ_k, h_{k−1}) = 0.95–0.97 and ‖Δ_k‖ ≈ 4.6–5.0·‖h_0‖. These are global (flattened) norms, so a few
   high-norm tokens or channels (e.g. a massive-activation channel, which LT2 sees growing across loops) could
   dominate. SQ14 measures it per token.
2. **Surprise: the inverse-U is the norm at scale.** Ouro (2.6B, 7.7T tokens, per-step supervised) and FLT show
   degradation or fluctuation past the trained loop count. Only Parcae (contractive injection) reports a clean
   saturating curve. The README's caveat "papers that report wins from looping use much bigger models" holds for
   in-range quality, but not for extrapolation.
3. **At matched FLOPs, looping loses in LM loss in every paper that checks** (Iso-Depth φ = 0.46, CHASE, LoopFormer,
   Frey on commonsense). C6 is the expected result, not an anomaly. The case for loops is parameters/memory and
   algorithmic tasks (C18/K5, Frey's math, LT2's state-tracking). E2/E9 should be framed that way.
4. **Conflict:** loop-boundary normalisation. Labovich says it's needed; the Agreement paper says removing it
   helps from-scratch extrapolation (16 seeds, but synthetic tasks, 19M params, fixed r0). SQ18 is a genuine
   tie-breaker at our scale.
5. **Is our GDN loop "wasting" the rank-T effect?** LT2's theory needs different keys per loop. Our GDN state
   restarts every loop, so the cross-loop product is over *separate* passes, not one state. Is that what their
   theory assumes? Their Eq. 5 multiplies per-loop transitions at a position, which suggests a fresh pass per
   loop, like ours. Unresolved.
6. **Early exit for GDN loops leaves "cache holes"** (CHASE): relevant to E5 (adaptive loops) and E8
   (self-speculative decoding with fewer loops). Plan for it before building them.
7. **Single-seed risk:** extrapolation is high-variance across seeds (Stochastic Stopping), so C7 needs ≥ 2
   seeds before VERIFIED.
8. **Paper hygiene:** LT2's text and Table 2 disagree (61.39 vs 62.89 avg; ×2.7 vs ∼5× decode). Quote the
   table and flag it.
9. **Inkling 2** is a Thinking Machines Lab MoE (594B/17B, NVFP4-native), not a looped model.
