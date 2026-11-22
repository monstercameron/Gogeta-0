# Claims ledger

Every claim Gogeta-0 relies on or states publicly, with its source and **whether this repo has verified it locally**. Rules: [AGENTS.md §0](../AGENTS.md).

**Statuses**
- `VERIFIED`: a local check passed.
- `PARTIAL`: verified in a narrower form (for example a single seed, or computed rather than measured).
- `REFUTED`: a local check failed.
- `WRONG FOR OUR CODE`: true in theory, but not true of our implementation.
- `UNVERIFIED`: not checked here yet.

Only `VERIFIED` claims may be stated as plain fact in the README, the site, the write-ups or the devlog.

*Audited 2026-11-22 (v0.3.5).*

## Verified or measured here

| ID | Claim | Source | Status | Evidence |
|---|---|---|---|---|
| C1 | fla's chunk Gated DeltaNet kernel matches a reference implementation, forward and backward, on this GPU | fla docs (assumed) | `VERIFIED` | `tests/test_gdn_kernels.py`: fp32 fwd 2.2e-4 / bwd rel 2.1e-3; bf16 8.0e-4 / 7.4e-3 (commit `0dc7491`) |
| C2 | fla's fused-recurrent GDN kernel (needed for step-by-step decoding) works on this GPU | fla docs (assumed) | `REFUTED` | Same test: returns NaN with and without grad mode |
| C3 | Every block type is causal, and sliding-window attention cuts off exactly at the window | our design | `VERIFIED` | `tests/test_model.py` 10/10: past-token diff exactly 0, outside-window diff 0 |
| C4 | BitLinear produces weights in {−1, 0, +1}·scale and trains without NaNs | BitNet b1.58 | `PARTIAL` | `tests/test_model.py`: levels {−1, 0, 1}, finite backward through a looped ternary model. *That it learns well* is untested (E3). |
| C5 | The hybrid stack (E1) beats dense attention (E0) at 14M params / 200M tokens of Python | our hypothesis | `PARTIAL` | −3.6% val loss, ahead at 25/25 checkpoints, reaches E0's final at 84% of budget. **Single seed**; replication killed. |
| C6 | Looping E2's core does not beat E0/E1 at ~2× the compute | our experiment | `PARTIAL` | E2 0.619 vs E0 0.606 / E1 0.585 at 2.0× E0 FLOPs. Single seed; the E9 control is pending. |
| C7 | E2's loss gets worse when looping beyond the trained range | our experiment | `PARTIAL` | r=4 0.619 → r=8 0.633 → r=12 0.654. Single seed. |
| C8 | Our models compress held-out Python to ~3% of its size vs 15.4% (`xz -9`) and 25.1% (`gzip -9`) | our measurement | `VERIFIED` | Measured on the 4,000 val docs (notebook, v0.3.1). Model side is single-seed. |
| C9 | Ternary shrinks the E1 body 22.2 → 2.8 MB (7.8×), the whole model 28.5 → 9.1 MB | computed | `PARTIAL` | Computed from the built model (33 BitLinear, 11.06M ternary params, 2-bit packing arithmetic). No packed file has been written and measured yet. |
| C10 | Looping gives depth without new parameters (E2: 9.46M non-emb, behaves like 14 blocks) | computed | `VERIFIED` | `param_counts()` / `effective_depth()` in `gogeta/model.py` |
| C11 | Triton on this WSL setup needs `python3.12-dev` | observed | `VERIFIED` | First kernel test failed with `Python.h` missing; passed after install (notebook) |

## Stated somewhere but not true of our code

| ID | Claim | Where it appears | Status | What's actually true / check |
|---|---|---|---|---|
| C12 | Sliding-window attention costs O(n·w) | `docs/index.html` (legend, rule box, mixer card, science table), README (mixer row), `site-src/learn.md` | `WRONG FOR OUR CODE` | Our SWA uses SDPA with a **dense boolean mask**, so the kernel is O(n²). **Confirmed by K1** (`experiments/checks/results/K1.json`): SWA time exponent 1.74, memory exponent 1.95 for n ≥ 4096 (criterion: both ≤ 1.3). Wording fixed in the science-gaps version: "O(n·w) in principle; our current kernel uses a dense mask, so it is still O(n²)". |
| C13 | Ternary weights make the model "~10× smaller" | `site-src/learn.md` | `REFUTED` | Computed: body **7.8×**, whole model **3.1×** (C9). Wording fixed in the science-gaps version: "layers about 7.8× smaller, the whole model about 3.1× (computed, not yet measured on a packed file)". |
| C14 | Ternary turns matrix multiplies into additions / makes compute cheap | README, site, primer, E3 and E4 pages | `WRONG FOR OUR CODE` | True only with ternary kernels. Ours simulates rounding in floating point. **Confirmed by K2** (`experiments/checks/results/K2.json`): BitLinear forward 5.6× (384→1024) and 4.1× (384→2048) *slower* than `nn.Linear`. Wording fixed in the science-gaps version: "could ... with dedicated ternary kernels; ours are simulated, so no speedup yet". |

## Not yet verified here (quoted from papers or theory)

| ID | Claim | Source | Status | Smallest local check (≤ 15 min) |
|---|---|---|---|---|
| C15 | Gated DeltaNet / sliding-window layers keep **fixed memory** as the input grows | Gated DeltaNet paper; theory | `UNVERIFIED` | **K1**: forward peak memory and time for one attn / SWA / GDN layer at n = 1k, 2k, 4k, 8k, 16k |
| C16 | Gated DeltaNet recalls better than plain linear attention / Mamba-2 | Gated DeltaNet paper | `UNVERIFIED` | **K3**: MQAR (multi-query associative recall), tiny 2-layer models (d=128), GDN vs SWA vs attn, ~5 min each |
| C17 | The ternary quality gap shrinks as models grow (parity around 3B) | BitNet b1.58, Table 1 | `UNVERIFIED` | **K4**: FP vs ternary at 3 tiny sizes (d=64/128/256, 2 layers, 20M tokens): does the gap shrink with size *at our scale*? |
| C18 | Linear-time mixers are weak at multi-step state tracking, and looping adds that depth | theory (expressivity papers) | `UNVERIFIED` | **K5**: parity / running-sum task, tiny GDN model 1 pass vs looped ×4 at equal params |
| C19 | Speculative decoding gives identical output, just faster (DFlash reports up to ~6×) | DFlash, speculative-decoding papers | `UNVERIFIED` | Blocked by C2 (no working step-by-step kernel). Later: greedy-output equality test + tokens/s |
| C20 | Each loop pass should refine the answer (a convergent loop) | looped-transformer papers | `UNVERIFIED`, and C7 points the other way | **K6**: measure ‖h_k − h_{k−1}‖ / ‖h_k‖ per pass on the trained E2 checkpoint (inference only, minutes) |
| C21 | Mamba-3 beats Gated DeltaNet (+0.6 pts avg downstream at 1.5B; MIMO +1.8) | Mamba-3, arXiv 2603.15569 | `UNVERIFIED` | **SQ7**: needs Mamba-3 kernels on this GPU, then a tiny E1-style swap |
| C22 | A complex-valued (rotary) state update solves state tracking (parity, modular arithmetic) in a single pass where Mamba-2 can't | Mamba-3, arXiv 2603.15569 | `UNVERIFIED` | **SQ6**: the K5 parity task with a data-dependent-RoPE single-pass layer |
| C23 | MIMO raises decode FLOPs up to 4× at the same decode latency; state 64 matches Mamba-2 at 128 | Mamba-3, arXiv 2603.15569 | `UNVERIFIED` | After a decode path exists: decode tokens/s and memory, SISO vs MIMO |
| C24 | A GDN hybrid with 1 full-attention layer per 4 beats a full-attention baseline (8/9 benchmarks) and an SWA hybrid (7/9) | Qwen3.8-Next, arXiv 2608.30320 §2.1.1 | `UNVERIFIED` (our E1 vs E0 is consistent, single seed) | **SQ9** |
| C25 | Sigmoid gates beat SiLU/tanh (GDN output gate, attention, residual gates) | Qwen3.8-Next §2.1.1, §2.2 | `UNVERIFIED` | **SQ8** |
| C26 | A gated, 4-branch widened residual lowers loss and cuts gradient-norm spikes and activation outliers | Qwen3.8-Next §2.2, §3.3 | `UNVERIFIED` | **SQ10**, **SQ12** |
| C27 | Muon (with the architecture) raises the optimal LR and batch size, removes batch-size warmup, improves stability | Qwen3.8-Next §3.1–3.3 | `UNVERIFIED` | **SQ11** |
| C28 | Lower loss doesn't guarantee better downstream ability (larger n-gram vocab: loss keeps falling, benchmarks saturate) | Qwen3.8-Next abstract | `PARTIAL`: consistent with our E0–E2 samples (best loss ≠ best code), not tested directly | **SQ4** (functional eval) |
| C29 | Model-specific RL infrastructure (kernels, memory layout, freezing parts) beats general-purpose RL frameworks for unusual architectures | FlashNext-RL, arXiv 2610.28998 (third party) | `UNVERIFIED` | Relevant when hooking up TRL (post_training/) |

## Proposed checks (not run yet)

| Check | Verifies | Cost | Script (when written) |
|---|---|---|---|
| K1 | C12, C15: memory/time vs context length per layer type | ~5 min GPU, no training | `experiments/checks/k1_memory_vs_length.py` |
| K2 | C14: BitLinear vs Linear speed (training + inference) | ~2 min GPU | `experiments/checks/k2_bitlinear_speed.py` |
| K3 | C16: MQAR recall, GDN vs SWA vs attn | ~15 min GPU (3 tiny models) | `experiments/checks/k3_mqar_recall.py` |
| K4 | C17: ternary gap vs model size at tiny scale | ~15 min GPU | `experiments/checks/k4_ternary_gap_vs_size.py` |
| K5 | C18: state tracking, single pass vs looped | ~10 min GPU | `experiments/checks/k5_state_tracking.py` |
| K6 | C20: does E2's loop converge? | ~3 min GPU, inference on the existing checkpoint | `experiments/checks/k6_loop_convergence.py` |

## Verified by the TRL setup work (2026-11-22)

| ID | Claim | Source | Status | Evidence |
|---|---|---|---|---|
| C80 | The Hugging Face wrapper (`gogeta/hf.py`) is numerically identical to FusionDance (logits, loss, left padding, save/load round trip) | our code | `VERIFIED` (CPU, attn/swa blocks) | `post_training/test_hf.py`: max logit diff 0.0; padding ≤ 1e-5; round trip exact. Re-run by the main agent: 27/27 OK. **GDN blocks through the wrapper on CUDA are untested.** |
| C81 | TRL's fused LM-head log-probs match `log_softmax` of our logits | TRL 1.18.2 source | `VERIFIED` (CPU) | `test_hf.py`, ≤ 1e-4 |
| C82 | GRPO, SFT and DPO run end to end through TRL with our wrapper | TRL docs | `PARTIAL` | CPU smoke runs on a 697K-param attn/swa model: GRPO 3 steps, 29/29 tensors updated; SFT loss 8.947 → 8.210; DPO 0.693 → 0.530. Not run on GPU, with GDN, or at real scale. |
| C83 | The reward sandbox blocks network access, memory blow-ups, fork bombs and infinite loops | our code | `PARTIAL` | `post_training/test_rewards.py` (19 cases). The **filesystem is not isolated** (it can read world-readable files and write `/tmp`); use a container or nsjail for real RL runs. |
| C84 | Without a cached/recurrent decode path, RL sampling is ~340× more work than cached decoding (256-token prompt + 512-token completion) | computed | `PARTIAL` (computed, not measured) | `post_training/SETUP.md` blockers |

## Proposed by competitor review (pending review)

Source: [`notes/review-competitors.md`](../notes/review-competitors.md) (fact-check of an externally pasted analysis against primary sources, 2026-11-22). Nothing here has been run locally. "XP-n" refers to the experiments in that note; SQ30–SQ41 are their side-quest rows.

| ID | Claim | Source | Status | Smallest local check (≤ 15 min) |
|---|---|---|---|---|
| C50 | Looping only the GDN mixer (FFN once per layer) keeps part of full-block recurrence's gain at ~54% of its backbone projection FLOPs: v1 110M keeps 41.5% of the CORE gain (NLL NoLoop 2.401 / MixerLoop 2.377 / FullLoop 2.342); v2 100M reports MixerLoop ≥ FullLoop on CORE (11.24 vs 10.89) | MixerLoop, arXiv 2608.18230 v1 §4.2, v2 Table 2 | `UNVERIFIED` | **XP-3 / SQ32**: tiny MixerLoop T=4 vs FullLoop r=2 (matched FLOPs *and* params, computed 1.314M vs 1.317M core matmul params/token) |
| C51 | Pasted analysis: "MixerLoop keeps much of the recurrence benefit while cutting projection FLOPs 45.9% at 110M" | pasted analysis, checked against 2608.18230 | `UNVERIFIED`; **overstated**. 45.9% is analytic (backbone projections only; end-to-end 43.0%); v1 keeps 41.5%, not "much"; v2 changed setup (13M/100M, 10B tokens) and reports 43%. The arXiv abstract field still carries the v1 numbers | Cite version-specific numbers only |
| C52 | MixerLoop's successive mixer passes keep lowering cross-entropy up to the trained T=4 (100M: 5.016 → 2.558 nats, steps 1.586/0.693/0.179); never evaluated beyond T=4 | 2608.18230 v2 §4.4, Fig. 5 | `UNVERIFIED` | XP-3: eval M2 at T = 1..8 |
| C53 | Auron (GDN:GQA 3:1, unique bottom + shared top × 3 loops, x0 injection) reaches val loss 3.188 / 3.035 / 3.180 at 279M / 510M / 1.1B; the 1.1B regressed | HF model cards + `training.log` of nyxia/Auron-*; HF blog 2026-04-16 | `UNVERIFIED` (numbers match the published logs; not reproducible here: code repo `Fy-/Auron` is 404) | n/a (no code) |
| C54 | Auron's 1.1B "scaling wall" is caused by representation saturation at dim 2048 (+ head_dim-64 "fragmentation") | Auron blog §5.3 | `UNVERIFIED`, and **confounded**: same LR and 32,768 tokens/step × 250k steps (≈8.2B tokens processed, ~1.6 epochs of 5B) at every scale; the 1.1B also changes topology (24 vs 16 virtual layers); its final *train* loss (2.86) is higher than the 510M's (2.72); the author's own Limitations say the cause was not isolated. The pasted "cause unresolved" is right in substance | **XP-4 / SQ33** (tokens-per-param regime) tests the under-training alternative at tiny scale |
| C55 | Weight-shared loops create repetition "attractor wells": Auron needs presence penalty ≥ 1.5 when sampling | Auron blog §5.7, model card | `UNVERIFIED` | **XP-8 / SQ37**: repeated-4-gram rate, E1 vs E2 at r = 2/4/8/12 (inference only) |
| C56 | Fully weight-shared stacks have a "parsing bottleneck"; a few unique bottom layers fix it | Auron blog §3.3 (anecdotal, no ablation) | `UNVERIFIED` | SQ39 (prelude depth 1 vs 2 at matched params) |
| C57 | TR-Mamba2-Attn (6.86M vs 6.83M params) improves ARC-AGI-1 pass@2 43.88% → 45.88% (pass@1 −0.25) but is **worse on Sudoku-Extreme (66.5% vs 72.2%)**; the pasted analysis omits the Sudoku result | arXiv 2602.12078 Tables 1–2 | `UNVERIFIED` (numbers confirmed in the paper) | n/a (bidirectional puzzle model, not a causal LM) |
| C58 | Post-norm (or a loop-boundary norm) is essential for stable recursion: pre-norm residual streams grow with recursion depth | 2602.12078 §3.3; Huginn 2502.05171 §3.2 ("required at scale") | `UNVERIFIED`; **consistent with our K6 numbers**: E2's ‖h_k‖/‖h_0‖ = 22.4 at k=4 → 58.9 at k=12, with a near-constant absolute increment ‖h_k − h_{k−1}‖/‖h_0‖ ≈ 5.0 → 4.6 (computed from `K6.json`) | **XP-2 / SQ31** |
| C59 | E2's loop is a near-constant drift in a fixed direction (h_k ≈ h_1 + (k−1)·v); the falling relative change in K6 is mostly norm growth, not convergence | our K6 data, re-analysed (computed) | `UNVERIFIED` (hypothesis) | **XP-1 / SQ30**: increment alignment cos(Δ_k, Δ_{k−1}) and norm-clamped eval on the E2 checkpoint, ~3 min |
| C60 | Looped GDN trains with the smoothest loss and smallest grad norms of all looped mixers (data-dependent gate + delta rule bound the loop); looped RetNet diverges | LT2, arXiv 2605.20670 §3.5 | `UNVERIFIED` | XP-2 grad-norm p99.9 (side measurement) |
| C61 | Attention sinks and massive activations compound across loop iterations; a head-wise sigmoid SDPA output gate inside the loop fixes it (1.3B Full+GDN looped hybrid: PPL 9.31 → 9.03) | LT2 §3.4, Table 4 | `UNVERIFIED` | **XP-6 / SQ35** |
| C62 | Looped GDN/mLSTM hybrids need sufficiently frequent exact-attention anchors; adding a Parcae input-reinjection wrapper alone degrades accuracy | arXiv 2611.19446 | `UNVERIFIED` | n/a yet (our core has one *windowed* anchor per 3 blocks) |
| C63 | INT4 PTQ of a looped model fails mainly at the non-residual loop-entry adapter ("feedback exposure"), not the residual core | arXiv 2609.30820 (Huginn-3.5B) | `UNVERIFIED`; our `inject` is residual (`h + inject(...)`) but is ternarised when `ternary=True` | **XP-9 / SQ38** (ternary vs FP `inject`) |
| C64 | Quantisation error introduced early in the recurrence is attenuated by later steps, so low precision belongs in early loops | RAMP, arXiv 2610.13041 | `UNVERIFIED` | XP-9 variant: ternary only in loops 1..r−1 (SQ38 note) |
| C65 | No published work combines a GDN-hybrid mixer, recurrent-depth weight sharing and ternary (BitNet-style) QAT in one causal LM | pasted analysis; arXiv API searches 2026-11-22 | `UNVERIFIED` (not refuted by the searches). Two of three are already combined: LT2 (looped GDN / Full+GDN hybrids, FP) and 2611.19446. Looped-model quantisation found is PTQ only (LoopQ, Loopy, RAMP, 2609.30820; MixerLoop FPGA W8A8); ternary QAT found only for non-looped SSMs (2606.18114) | Re-search before any public "first" claim |
| C66 | Starting the loop from s0 = e makes the iteration depend on boundary conditions; random s0 + per-step injection promotes path independence; step-conditioned cores fail to extrapolate | Huginn 2502.05171 §3.1 | `UNVERIFIED` | **XP-5 / SQ34** (overlap with the looped-transformer review) |
| C67 | Weight sharing pays mainly when parameter-limited: published small looped wins train at ~2,000+ tokens per non-embedding param (MixerLoop 13M: 10B tokens, ~4.7M non-emb), vs E2 at ~21 | computed from 2608.18230 Tables 5–6 and our E2 config | `UNVERIFIED` (hypothesis) | **XP-4 / SQ33** |
| C68 | On our core, MixerLoop T=4 costs the same projection FLOPs as our full-block loop at r=2 (E2 dims: 11.84M vs 11.82M matmul params per token; tiny d=128/mlp 344: 1.314M vs 1.317M), and the MLPs are ~60% of a core pass | computed by hand from `gogeta/model.py` layer shapes | `UNVERIFIED` (computed; confirm with `param_counts()` / `flops_per_token()`) | One-line script, CPU |
| C69 | Auron injects the input as `+ λ_l·x0` with one learned scalar per *virtual* layer, conditions each shared pass with a per-(loop, block) d-vector added to the GDN input only, and uses U-Net skip vectors and per-block residual scalars (init √1.1); its checkpoint has no learnable RMSNorm weights | Auron-510M safetensors header (tensor names/shapes only) + blog §3.4–3.5 | `UNVERIFIED` (training/inference code unavailable) | n/a |

**Overlap note (competitor review):** C59 (E2 loop = near-constant drift) is the same reading as the looped-transformer review's "coda drowning" claim (C41 in `notes/research-looped.md`). Keep one row when merging. C61 overlaps its SQ22 source (LT2 §3.4).

## Proposed by looped-transformer research (pending review)

Source: [`notes/research-looped.md`](../notes/research-looped.md) (papers in `refs/11-looped-2026/`). All rows come from papers or from re-analysis, so all are `UNVERIFIED`. Overlaps with the competitor-review rows: C38 refines C65, C39–C40 refine C59, and C37 conflicts with C58.

| ID | Claim | Source | Status | Smallest local check |
|---|---|---|---|---|
| C30 | Looped LMs at scale also peak at the trained loop count and degrade beyond it: Ouro 1.4B/2.6B (7.7T tokens, trained T=4) show "moderate degradation" at T=5–8 (2.6B-Thinking OlympiadBench 76.4 at T=4 → 39.3 at T=8), so our C7 shape need not be a small-scale artifact | Ouro, arXiv 2510.25741 §5.3, Tables 10–13 | `UNVERIFIED` | H0/TINY baseline arm in SQ15–SQ20 must reproduce X12 = L(12)/L(4) − 1 ≥ +1% (2 seeds) |
| C31 | Loop instability comes from an injection whose linearised transition has spectral radius ≥ 1 (additive: ρ = 1; concat-projection, like ours: unconstrained). A contractive diagonal A = exp(Δ·−exp(log A)) plus a prelude norm stabilises training, and test-time loss then saturates (L∞ + Z·e^{−zT}, plateau near µ_rec) instead of degrading | Parcae, arXiv 2604.12946 §3, §4.1, §5.3, Tables 2 and 4 | `UNVERIFIED` | **SQ15** (= SQ31 V3) |
| C32 | Shared-weight loop increments are positively correlated, so the residual norm grows ~linearly in the loop count N (√L for distinct layers); scaling each residual branch by 1/N keeps the forward pass bounded and the best LR fixed for N ≤ 8 | Loop-µP, arXiv 2606.18524 §3.2, §4.1–4.2, Figs. 4–5 | `UNVERIFIED` | **SQ14** (alignment) + **SQ16** (1/r scaling, grad-norm p99.9) |
| C33 | At matched training compute, a recurrence is worth r^0.46 unique blocks (r=4: 410M looped ≈ 580M non-looped at a 1B model's training cost). Truncated BPTT lowers the exponent to 0.38 even though val loss falls; loop-level hyper-connections raise it to 0.65. Iso-FLOP deeper non-looped models also win on PPL in looped Mamba-2 (CHASE) | Iso-Depth, arXiv 2604.21106 §4–5.2; CHASE, arXiv 2607.10110 §5.2 | `UNVERIFIED` | **SQ19** (full vs truncated BPTT); C6 framing |
| C34 | Per-loop supervision in Ouro's exit-weighted form does not by itself buy extrapolation beyond the trained max loop count (Ouro is trained that way and still degrades at T>4) | inference from Ouro, arXiv 2510.25741 §3.3 + §5.3 | `UNVERIFIED` | **SQ17** (LoopFormer consistency loss: does X12 halve?) |
| C35 | Loop extrapolation is high-variance across seeds (OOD std 0.087–0.122 looped vs ≤ 0.009 fixed-depth on binary addition); randomised or RL-learned loop counts cut it (std 7.4 → 2.7 pts), but low variance ≠ extrapolation | Learned Stochastic Stopping, arXiv 2606.29983 §4.1–4.3, Table 1 | `UNVERIFIED` | ≥ 2 seeds for any C7/X12 result; **SQ20** (per-sequence r) |
| C36 | Loops amplify quantisation error: INT4 GPTQ costs plain looped models +2.2/+2.2/+3.3% PPL vs +1.4/+0.8/+1.3% for depth-matched Transformers (136M/580M/990M); loop-level hyper-connections cut it to +1.9/+1.7/+1.2% | Hyperloop, arXiv 2604.21254 Table 1 (percentages computed here) | `UNVERIFIED` | **SQ21** (post-hoc INT8/INT4 on the E2 core, ΔL(r) for r = 1..12); complements SQ38 |
| C37 | Loop-boundary normalisation is *not* what keeps extra loops usable: 19M models trained without it (16 seeds, r0 = 4) are never worse and extrapolate better (Cliff's δ up to 0.867). **Conflicts with C58** and Labovich (arXiv 2604.15259: recall + outer norm needed) | Agreement not Normalisation, arXiv 2610.31943 §4.3 | `UNVERIFIED` | **SQ18** (= SQ31 V1 + K5 parity arm) |
| C38 | Loop + Gated DeltaNet + sliding-window/sparse attention inside the loop is published (LT2 Looped Hybrid GDN+DSA at 0.6B/1.3B, "Looped GDN+Window" on synthetic tasks), as is loop + Mamba-2 (CHASE). None of them use ternary weights or put full attention only outside the loop | LT2, arXiv 2605.20670 Table 2, Fig. 9; CHASE, arXiv 2607.10110 | `UNVERIFIED` (literature) | Re-run the novelty search before any "first" wording (SQ41) |
| C39 | From k ≥ 4, each E2 loop adds an increment of near-constant size (‖Δ_k‖ ≈ 4.6–5.0·‖h_0‖) almost parallel to the state, cos(Δ_k, h_{k−1}) = 0.95 → 0.97 (global flattened norms) | computed here from `experiments/checks/results/K6.json` (refines C59) | `UNVERIFIED` (computed, not per-token) | **SQ14** / SQ30: per-token cos and ‖Δ_k‖ |
| C40 | Hypothesis: E2's loss rises past r=4 because the growing state "drowns" the pre-norm residual coda (its fixed-size update shrinks relative to ‖h‖) | our hypothesis from C39 + Loop-µP | `UNVERIFIED` | **SQ14**: rescale h_k to ‖h_4‖ before the coda; pass if it recovers ≥ 50% of L(12) − L(4) |
| C41 | Early exit in a looped recurrent-state (SSM) model leaves "cache holes" (missing deeper-loop states for later tokens): the stock Ouro gate with skipping blows PPL up (by up to +4788%); training for skipped states keeps it within +3–4% at ~half the loops | CHASE, arXiv 2607.10110 §3.2, Fig. 2 | `UNVERIFIED` | Relevant to E5/E8; no check until a decode path exists (C2) |
| C42 | Looping helps algorithmic tasks more than knowledge: per-layer adaptive loops (Nmax=3) beat a 3×-deeper iso-FLOP model on math BPB (1.687 vs 1.801) but lose on commonsense BPB (0.813 vs 0.780) | Adaptive Loops and Memory, arXiv 2603.08391 Table 1 | `UNVERIFIED` | K5-style parity/MQAR vs LM loss at matched FLOPs (fold into the E9 analysis) |

## Science-gaps fix (2026-11-22): tuned reference control, LR tuning, replication

| ID | Claim | Source | Status | Evidence / check |
|---|---|---|---|---|
| C90 | Hugging Face's `LlamaForCausalLM` at E0's shapes (E10) has exactly E0's parameter count (13,767,552 total / 10,621,824 non-embedding) and analytic train FLOPs (96,758,784 per token at seq 1024) | computed | `VERIFIED` (computed) | `experiments/checks/k7_e10_parity.py` → `results/K7-parity.json` |
| C91 | Adding `"arch": "hf_llama"` to `gogeta/train.py` leaves FusionDance training unchanged | our code | `PARTIAL` | Construction bit-identical for E0/E1/E2/E3/E9 (state_dict, config, RNG state, FLOPs; K7-parity, CPU). The loop body is unchanged by inspection. A GPU replay of the first 100 steps against the v0.3.0 E0 log is pending (see notes/science-gaps.md). |
| C92 | lr 3e-3 is a good learning rate for our models at 200M tokens (every result so far used it, untuned) | implicit assumption | `UNVERIFIED` | **K7**: 20M-token sweep {1e-3, 2e-3, 3e-3, 6e-3} for E10 and E1 (`experiments/checks/PREDICTIONS.md`) |
| C93 | Seed-to-seed std of the final val loss is ≤ 0.004 (so a 0.02 gap is real) | our prediction (notebook, v0.3.0) | `UNVERIFIED` | Replication protocol: 3 seeds of E10 and E1 at tuned LRs |
| C94 | Our ~0.23 bits/char on held-out Python is plausible for this corpus (not a bug or train/val leakage) | our assumption | `UNVERIFIED` | **K8**: SmolLM2-135M and our checkpoints scored doc-level in bits/char (`k8_external_reference.py`) |
| C95 | E1 vs E0 is a matched-tokens, near-matched-params comparison (E1 +4.4% non-embedding params, −6% FLOPs), not a matched-compute one | computed | `VERIFIED` (computed) | `param_counts()` / `flops_per_token()`; registry results. Every "E1 beats E0" sentence must say which is matched. |

## Verdicts from checks K1–K6 and SQ6 (recorded for v0.4.0)
These update the status of earlier rows. Where they differ, this section is the current status, until the rows are consolidated.

| Claim | New status | Evidence |
|---|---|---|
| C12 SWA is O(n·w) | `WRONG FOR OUR CODE` (confirmed) | K1: SWA time exponent 1.74, memory 1.95 (2.9 GB at 16k vs 15.6 MB at 1k) |
| C14 ternary makes compute cheaper | `WRONG FOR OUR CODE` (confirmed) | K2: BitLinear forward 4.1–5.6× slower than nn.Linear |
| C15 GDN fixed memory / linear time | `VERIFIED` (kernel level) | K1: GDN time exponent 0.66, memory 0.98 (activations); streaming in two halves with a carried state is bit-identical (max diff 0.0); state [1,6,64,64] at every n |
| C16 GDN recalls better than linear attention | still `UNVERIFIED` | K3 **invalid**: the attention positive control reached only 8.6% (needs ≥ 90%). The MQAR setup needs a redesign. |
| C17 ternary gap shrinks with size | `PARTIAL` (local analogue verified) | K4: 15.3% → 13.8% → 10.5% at d = 64/128/256 (2 seeds, also within each seed); 3B parity untestable here |
| C18 looping adds state-tracking depth | `VERIFIED` (parity, T=64) | K5: 1 pass 81.5% → looped ×4 97.0%; SQ6 replication: 61.7% → 97.9% (2 seeds). But 4 unshared layers reach 99.8%, and the looped model **doesn't length-generalise** (61.2% on positions 64–127). |
| C20 E2's loop converges | `REFUTED` for E2 | K6: relative change 0.085 at k=12 (needs < 0.05); ‖h_k‖/‖h₀‖ grows 7.4 → 58.9; constant-size steps (~4.6–5 × ‖h₀‖). Best loss at k=4. |
| C22 a rotating state solves parity in one pass | `REFUTED` at our scale | SQ6 (c) RotSSM 84.0% (needs ≥ 90%), though +33.8 points over no rotation; it does generalise better than looping (76.5% vs 61.2% on positions 64–127) |
| **C43 (new)** GDN with negative eigenvalues (β ∈ (0, 2)) solves state tracking in **one pass** and length-generalises | `VERIFIED` (parity, 2 seeds, our scale) | SQ6 (e): **100.0%** on positions 32–63 and **99.99%** on 64–127, vs looped GDN 97.9% / 61.2%. One fla flag (`allow_neg_eigval`, in-kernel sigmoid). |
| **C44 (new)** single-pass GDN parity accuracy has high run-to-run variance | `VERIFIED` (observed) | K5 (i) 81.5% vs SQ6 (a) seed 0 73.3% / seed 1 50.1%, with identical init/data; 1-pass GDN numbers need ≥ 2 seeds |

## Proposed by Kimi Linear research (pending review)

Source: Kimi Linear, arXiv 2510.26692v2 (48B-total / 3B-active MoE, 1.4T tokens unless noted), and fla 0.6.0 `fla.ops.kda`. Details in [`notes/research-kimi-linear.md`](../notes/research-kimi-linear.md). All paper numbers are *reported*, at a scale ~3,000× ours.

| ID | Claim | Source | Status | Smallest local check (≤ 15 min) |
|---|---|---|---|---|
| C70 | fla's `chunk_kda` (`from fla.ops.kda import chunk_kda`) matches an fp32 reference fwd/bwd on this GPU, and reduces to `chunk_gated_delta_rule` when `g` is tied across channels | fla 0.6.0 (assumed) | `UNVERIFIED` (GPU). Only the *reference* is checked: on CPU our loop matches fla's `naive_recurrent_kda` (2.2e-8) and `naive_chunk_kda` (4.4e-7), and tied-gate = GDN reference exactly (0.0), CPU job `kda-cpu-ref-20261122-165206`. First GPU attempt hit the 300 s timeout with no output while other agents' GPU checks started alongside; no result file | **SQ50** `experiments/checks/sq_kda_kernel.py` |
| C71 | fla's `fused_recurrent_kda` (decode kernel) is finite and correct on this GPU, unlike GDN's (C2) | fla 0.6.0 (assumed) | `UNVERIFIED` | SQ50 (grad mode, no-grad, chunk prefill → 1-token steps) |
| C72 | KDA's chunk kernel costs about the same as GDN's (paper: "negligible" model-level prefill overhead vs GDN-H, Fig. 7a; ~2× faster than general DPLR, Fig. 2) | Kimi Linear §3.2, §5.6 | `UNVERIFIED` | SQ50 speed section (B=16, T=1024, H=6, K=V=64) |
| C73 | Channel-wise (per-key-dim) decay beats GDN's head-wise scalar decay in a hybrid: Kimi Linear vs GDN-H, e.g. MMLU-Pro 51.0 vs 47.9, RULER 84.3 vs 80.5, long-context avg 54.5 vs 51.2; faster MQAR/palindrome convergence (Fig. 4). **No isolated scalar-vs-channel PPL ablation is reported** | Kimi Linear Tables 3–5, Fig. 4 | `UNVERIFIED` | **SQ51** (d=256 E1-style, GDN vs KDA, 20M tokens, 2 seeds) |
| C74 | NoPE global-attention layers in a KDA hybrid beat RoPE on long context (avg 54.5 vs 51.8; RULER 84.3 vs 78.8) with similar short-context scores | Kimi Linear Table 5, §5.2 | `UNVERIFIED` | **SQ52** (NoPE in our one `attn` layer, train T=512, eval to 2048) |
| C75 | A 3:1 linear:full ratio is the best trade-off (val PPL 3:1 5.65; 1:1 5.66; 7:1 5.70; 15:1 5.82; full attention only 5.77) | Kimi Linear Table 1 | `UNVERIFIED` | SQ9 / SQ57 |
| C76 | A sigmoid output gate beats swish/SiLU and no gate (val PPL 5.65 vs 5.81 vs 5.67); a low-rank gate performs comparably to full rank | Kimi Linear Table 1, §4 | `UNVERIFIED` (same direction as C25) | **SQ55** (with SQ8) |
| C77 | Short convolutions still matter in a hybrid (val PPL 5.70 without vs 5.65 with) | Kimi Linear Table 1 | `UNVERIFIED` | Small E1 ablation (drop `ShortConv`) |
| C78 | Kimi Linear cuts KV cache by up to 75%, with up to 6× (TPOT 1.84 vs 11.48 ms, 6.3×) decoding at 1M and 2.9× prefill at 1M vs MLA. **The paper is inconsistent:** §6.3 calls 6.3× "theoretical" (larger batches) and reads Fig. 7b as 2.3× at 1M | Kimi Linear Fig. 1b, 7; §6.3 | `UNVERIFIED` (not testable at our context length; K1 checks the fixed-state part, C15) | n/a at our scale |
| C79 | Kimi Linear has ~1.16× the compute efficiency of MLA in a scaling-law fit (653M–1.7B active, 38.8B–128B tokens) | Kimi Linear Fig. 5 | `UNVERIFIED` | Not cheaply testable |

## Proposed by benchmark-suite research (pending review)

Source: `notes/benchmark-suite.md` (survey, predictions written before running, scorecard) and `evals/`. All runs CPU only, ≤ 2 threads, via `tools/launch_cpu_job.sh`, 2026-11-22; the E0 checkpoint is `~/fusiondance-data/checkpoints/E0-20261122-122059-s0/model.pt` (seed 0).

| ID | Claim | Source | Status | Evidence / smallest check |
|---|---|---|---|---|
| C100 | lm-evaluation-harness 0.4.13, through `evals/gogeta_lm.py` (an `HFLM` subclass handed our `GogetaForCausalLM`), returns exactly our model's log-probs for rolling (multi-window) and (context, continuation) requests | lm-eval docs + our code | `VERIFIED` (CPU, random attn/swa model) | `evals/test_wiring.py` 6/6: equals a hand-written forward pass (≤ 1e-3 nats; a 3,112-token doc over 4 windows rel ≤ 1e-4). GDN through the harness is untested |
| C101 | lm-eval's per-document val bpc is **6.1% lower** than train.py's packed val bpc (E0: 0.2208 vs 0.2351), and the gap is the protocol (packed windows start mid-document), not a wiring error | our measurement | `VERIFIED` (E0, one checkpoint) | `evals/checks/packed_vs_doc.py`: train.py's protocol re-run on CPU fp32 = 0.606254 vs 0.606341 recorded (Δ 0.014%); loss on the first 64 positions of a window 1.060 vs 0.576 after. Never mix the two protocols in one comparison |
| C102 | HumanEval pass@1 floors at our size: Codex-12M 2.00% (pass@100 8.58%) after 100B tokens, PolyCoder-160M 2.13%, CodeParrot-110M 3.80%; one problem = 0.61 pts, so variants ≲ 5 problems apart can't be told apart | Codex (arXiv 2107.03374 Table 1), PolyCoder (arXiv 2202.13169), computed SE | `UNVERIFIED` (reported; we have not run HumanEval on a trained checkpoint) | `python -m evals.run_suite --checkpoint <E0/E1/E2> --tier 2` on the GPU slot (~164 greedy generations each) |
| C103 | Our training data contains HumanEval/MBPP material: 32/164 HumanEval problems have ≥ 50% of their ≥ 40-char docstring/solution lines verbatim in `docs_train.jsonl` (43/164 with ≥ 2 lines); 43/500 MBPP test statements appear verbatim (0 of their assert lines) | our measurement | `VERIFIED` (line-match screen; a lower bound, misses paraphrases) | `evals/contamination.py` → `evals/results/contamination-2026-11-22.json`, flags in `evals/tasks/contaminated_ids.json`. Effect on E0: solution bpb full vs clean HumanEval 0.5106 vs 0.5174, MBPP 0.6398 vs 0.6544 |
| C104 | OpenCoder's annealing corpus was not decontaminated against HumanEval/MBPP (the paper describes 10-gram decontamination for its SFT data only) | OpenCoder, arXiv 2411.04905 §4.4 (as read) | `PARTIAL` | Consistent with C103; the paper was read only partly (first 100k characters) |
| C105 | `post_training/rewards.py`'s sandbox child cannot write to the repo (SETUP.md: "cannot write to /root, the repo, etc.") | post_training/SETUP.md | `REFUTED` (on this WSL setup) | Control: `rewards.run_tests` wrote a file under `/mnt/c/...` (drvfs shows every file as mode 777, so `nobody` can write the repo and all of C:). `/root` is protected. Affects GRPO rewards too (C83) |
| C106 | `evals/sandbox.py` (rewards.py's harness + new user/net/mount namespaces, tmpfs over /mnt, /tmp, /dev/shm, /run, nested userns as uid 65534) hides and protects /mnt/c, the repo and $HOME, blocks network, and cannot be unmounted from inside | our code | `VERIFIED` (this machine) | `python -m evals.sandbox`: 11/11 probes. Controls: the same write succeeds via rewards.py and fails here; a single-namespace variant CAN `umount /mnt` (verified), the nested one has CapEff=0. Not a security boundary: world-readable files stay readable |
| C107 | E0 writes working solutions for in-distribution held-out problems (val docs with asserts), so a functional metric does not floor for us | our measurement | `PARTIAL` (first 8 of 2,555 problems: 7/8 greedy; inspected) | `evals/results/check-E0-s0-exec-limit8-*.json`. Full run on the GPU slot pending; near-duplicate val/train docs not ruled out (the val split is exact-dedup only) |
| C108 | A 14M all-attention model trained on our corpus does in-context key-value recall from Python assignments (`gogeta_recall_d*`) | our expectation (P4) | `REFUTED` (E0, this probe design) | E0 exact recall 0.5/0/0/0% at 64/256/512/960 tokens; value bpb 2.60 at d=64 ≈ uniform chance 2.63 (computed), rising to 3.41 at d=960. The probe needs a redesign (SQ63) before it can rank architectures |
| C109 | lm-eval's `mbpp_plus` task scores only MBPP's 3 base asserts (`test_list`), not EvalPlus's extra tests | lm-eval 0.4.13 source (`tasks/mbpp/mbpp.yaml` `doc_to_target`, inherited by `mbpp_plus.yaml`) | `VERIFIED` (source read) | Our `gogeta_mbpp_plus_exec` runs the EvalPlus `test` program instead |

## Proposed by the 3D page work (main agent reviewed)

| ID | Claim | Source | Status | Evidence / notes |
|---|---|---|---|---|
| C130 | The 3D page Play trace for E2 (r=4, checkpoint E2-20261122-132340-s0 step 6104) is the model's real computation: real tokenizer pieces, real softmax attention weights, per-loop Δh, top-5 and greedy continuation | `experiments/viz/export_trace.py` (subagent) | `PARTIAL` | Traced forward = `model.forward` to max abs dlogit 9.5e-6, but on CPU fp32 with a plain-PyTorch GDN reference instead of the fla bf16 kernels: val loss 0.6423 there vs 0.6186 recorded, so values are close, not identical. Other models/r are labelled ILLUSTRATIVE on the page. Upgrade to VERIFIED by regenerating on the GPU with fla. |
