# Research side quest: activation functions for FusionDance

Written 2026-11-22 by a research subagent (CPU only, no training run, nothing committed). Every statement is **reported by <paper>**
(quoted, not adopted, AGENTS.md section 0), **measured here** (CPU, one checkpoint, section 4), **computed**, or **[speculation]**.
Nothing here is `VERIFIED`. Proposals: claims C200-C209, side quests SQ160-SQ166, queue items, all in
`notes/research-activations/proposed-*`. Unapplied code: `notes/patches/activation-mlp-act.patch`.

Papers were read from the PDFs (pdftotext) or the arXiv HTML; the 15 papers relied on are in `refs/13-activations/`
(`refs/fetch_refs.py`, section `13-activations`). Things I cite from memory without a downloaded paper are tagged **[recall]**.

## 0. TL;DR

1. **Nothing here is a free lunch at our scale; the evidence that any activation beats SwiGLU is at >= 0.12B params, mostly
   >= 1B.** The only direct small-scale evidence (0.12B, 20 tokens/param) is MoA (arXiv 2605.26647): a *learnable linear
   blend of activations* in a SwiGLU gains ~0.015-0.016 loss (about 0.5%) over plain SwiGLU. The sizes of all other gains are
   ~1% perplexity (xIELU, PolyNorm) or are about inference sparsity, not quality.
2. **For BitLinear with int8 activations, outliers are not the reason to change activation.** Our CPU measurement (section 4)
   shows per-token absmax int8 reproduces the down-projection input to 1.7-3.6% relative L2 error for every candidate. The
   outlier problem in the literature (BitNet a4.8, v2) is an **int4** problem. Choose by quality, sparsity and kernel cost.
3. **Top recommendation (confidence: low-medium): run a param-matched A/B of ungated ReLU^2 (and xIELU) against SwiGLU, both
   fp and ternary, before changing anything.** ReLU^2 is what BitNet b1.58 2B4T (2504.12285) ships in its FFN, Nemotron-4
   15B (2402.16819) uses, and BitNet a4.8 (2411.04965) uses (as a GLU) to get >80% sparse down-projection inputs. But the one
   short-budget ablation we have (xIELU table, 1.1B, 4B tokens: SwiGLU 17.58 vs ReLU^2 18.12 ppl) says GLUs win early, and our
   budget is small. So: SwiGLU stays the default until SQ160/SQ161 say otherwise.
4. **Combination verdict:** a learnable per-channel blend (SiLU <-> ReLU^2) inside the gate is the one combination with
   direct (small-scale) evidence (MoA's "LA"/"bi-LA": -0.015 to -0.016 vs SwiGLU at 0.12B) and costs +h scalars.
   Different activations per region (prelude/core/coda), per-layer routing (PolyGLU) and polynomial compositions
   (PolyCom) have **no controlled small-scale evidence for us**; PolyGLU has no SwiGLU control at all.
5. **For the looped core the activation is not the lever.** MLP and mixer inputs are RMS-normed, so no activation can
   change the injection's scale; K6's linear norm growth comes from the shared increment adding coherently (Loop-µP
   2606.18524, Parcae 2604.12946). I found no paper studying activation functions for loop stability. Keep the
   activation unchanged there, and fix the drift with injection/step-size changes (SQ15/SQ16).

## 1. Survey table

Quality column = the best controlled comparison I could read, with scale. "GLU width" = hidden size needed to match
SwiGLU params (GLU: 3 matrices of d x h; ungated: 2 matrices, so h_ungated = 1.5 h_GLU). FLOPs follow params, so
param-matched means FLOP-matched at the matmul level.

| Activation | Quality evidence (scale) | Params / FLOPs | Sparsity | Outliers / int8 | Stability, sharing, loops | Fusion cost (bandwidth-bound) |
|---|---|---|---|---|---|---|
| **ReLU** (ungated) | Shazeer 2002.05202: T5-base-like, 524k steps, log-ppl 1.677 (vs GEGLU 1.633, ReGLU 1.636, GELU 1.679); ReLU Strikes Back 2310.04564: from scratch "no significant difference" between ReLU/GELU/SiLU | h = 1.5 h_GLU | OPT: > 90% per layer (2310.04564); Llama-SiLU far lower | Measured: crest 7.7, kurt 16, int8 err 0.9% (section 4) | Unbounded; gradient 0 for x<0 (dead units) | Trivial elementwise |
| **GELU / SiLU(Swish)** (ungated) | Shazeer: 1.679 / 1.683 at 524k steps; xIELU table: SiLU 18.57 vs SwiGLU 17.58 ppl at 1.1B / 4B tokens | h = 1.5 h_GLU | ~0-3% | kurt 16-18, int8 err 1.8% | Smooth, bounded below | Tanh/erf slightly dearer than SiLU |
| **GLU family** (GEGLU, SwiGLU, ReGLU, Bilinear) | Shazeer, T5-base-like: 65k steps GEGLU 1.942, SwiGLU 1.944, ReGLU 1.953, Bilinear 1.960, ReLU 1.997; 524k steps GEGLU 1.633, ReGLU 1.636, Bilinear 1.648 (all param/FLOP-matched). Paper offers no explanation | 3 matrices; h_GLU = (2/3) h_ungated | SwiGLU/GeGLU limited (TurboSparse: SwiGLU 40% -> ReGLU ~70%) | Measured: product forms have kurt 5.7k-28k, crest 15-20 vs ~8 ungated, but int8 err 2-3.6% | The b branch is unbounded and multiplies a: product grows ~quadratically in the input scale | Reads 2h, writes h; fine when fused into the up-matmul epilogue |
| **ReLU^2** (Primer) | Primer 2109.08668: gain attributed jointly to squared ReLU + depthwise conv after QKV; 4x less training compute at 500M, 1/3 at 1.9B; Fig. 5 (110M): squared ReLU beats ReGLU/GELU/ReLU. xIELU Table 2: ReLU^2 10.35 vs SwiGLU 10.52 ppl at 1.1B, 125B tokens, but 18.12 vs 17.58 at 4B tokens (SwiGLU better early) | 2 matrices; h = 1.5 h_GLU | a4.8 (ReLU^2-GLU, up to 7B): down-proj input > 80% sparse, gate output 67.5% | Measured: relu2 ungated crest 14.7, kurt 4.3k (heavier tail than ReLU: squaring), int8 err 1.9% | Quadratic growth; 2B4T pairs it with sub-LN "for stability" | Trivial; cheapest to fuse |
| **Squared-ReLU-GLU** (ReLU^2 gate x up) | a4.8: used for sparsity; "slightly better training ppl than Swish" at INT8 (3B, Fig. 5) | 3 matrices | gate 67.5%, down input > 80% (a4.8) | Measured: kurt 28.5k, max/RMS 196, the worst tails of all (quartic in the input scale) | Most outlier-prone | Same as SwiGLU |
| **dReLU** (ReLU(a) x ReLU(b); TurboSparse 2406.05955) | Matches SwiGLU at 300M pretraining; Mistral-7B/Mixtral sparsified (> 150B tokens) | 3 matrices | ~90% (TurboSparse); ReGLU only ~70% | Measured: 85% sparse, kurt 6.1k, int8 err 1.7% (lowest of the gated forms) | Sparse; no negative branch to cancel | Trivial |
| **ProSparse** (2402.13516) | ReLUfication with progressive L1; "comparable performance" to Swish at ~7B and MiniCPM-1B | no change | 87.9-89.3% | n/a | Fine-tuning recipe, not a from-scratch activation | n/a |
| **xIELU** (2411.13010) | 1.1B/125B tokens: 10.21 ppl vs ReLU^2 10.35 vs SwiGLU 10.52; 3B/125B: 9.56 vs SwiGLU 9.71; 4B-token ablation 17.52 vs SwiGLU 17.58 vs ReLU^2 18.12; initially behind SwiGLU, matches it near 20B tokens (reported) | ungated, h = 1.5 h_GLU (they use 9216 vs 6144); 2 scalars/layer | Positive side is ReLU^2-like, negative side small | Measured with init alphas (ungated): crest 12.4, kurt 2.4k, int8 err 3.0% | bf16 OK; `expm1` needed near 0 (catastrophic cancellation); FP8 untested | ~2% slower per iteration than SwiGLU, no custom kernel (torch.compile) |
| **PolyCom** (2411.03884) | 1.3B dense, 250B tokens: loss/ppl SwiGLU 2.19/3.22, PolyNorm 2.17/3.17; OLMoE 1.3B-active, 200B tokens: val loss 2.41 -> 2.39; no seeds | extra 3 coefficients; PolyNorm needs per-power norms (extra reductions) | n/a | Order 4 showed no gain and overflow risk in low precision (their words) | PolyNorm "designed to reduce bf16 instability"; PolyReLU may be unstable outside transformers | Extra reductions: more elementwise passes |
| **PolyGLU** (2603.13347) | 597M, 10B tokens, **no SwiGLU control** (they say so); selects GELU early, Tanh in layers 15-27 | +0.23% params; all K activations are computed | Routing near one-hot | n/a | Weight-decay bug on routing parameters for ~10k steps | K x elementwise cost |
| **Mixture of Activations / Learnable Activations** (2605.26647) | 0.12B dense, 20 tokens/param: LA/MoA vs SwiGLU **-0.015 to -0.016** relative loss (bi-LA, bi-MoA); Type-I: LA +0.003 (worse than the ReLU^2 baseline), MoA -0.015; adding tanh to the dictionary hurts; sigmoid gate best; param-controlled control (Llama-large) gains ~0 | +O(d) params; 1.02x wall-clock on 0.5B | n/a | "Tolerate larger peak LR" (their remark) | Dictionary g,s,r2,l (GELU, SiLU, ReLU^2, LeakyReLU) best for ReLU-type | 1.02x with torch.compile + Triton |
| **KAN** (2404.19756) | Interpretable small tasks; not shown competitive for LM pretraining in anything I read | Spline per edge: params and compute several times an MLP [recall, not verified in the PDF] | none | Spline grids are not int8/ternary friendly [speculation] | Slow, hard to fuse | Not viable on this GPU |
| **SoLU, Mish** | Not downloaded. SoLU (Anthropic, not arXiv) aims at interpretability and pays in quality [recall]; Mish is a vision-era smooth activation, no LM evidence I know of [recall] | | | | | Not pursued |
| **Sigmoid vs SiLU as an output gate** (Gated Attention 2505.06708) | 15B-A2.5B MoE, 400B tokens: elementwise sigmoid gate after SDPA ppl 5.761 vs SiLU gate 5.822 vs baseline 6.026; gating reduces loss spikes and attention sinks over 3.5T tokens | negligible | Sigmoid scores are sparse | Fewer massive activations (their claim) | Sigmoid is bounded in (0,1): that is the stabilising part | Cheap |

Note on xIELU: Table 2 of 2411.13010 gives 1.1B: SwiGLU 12.01 / 10.52, xIELU 11.72 / 10.21 at 100B / 125B tokens (about 3% perplexity); 3B: SwiGLU 10.89 / 9.71, xIELU 10.74 / 9.56 (about 1.5%). Seeds are not reported.

Other reported anchors: BitNet b1.58 2B4T (2504.12285) uses ReLU^2 FFNs "for sparsity and computational characteristics in the
1-bit context", sub-LN, per-token absmax int8, but reports no activation ablation. BitNet a4.8 (2411.04965): INT4 on the
down-proj input with STE **diverges**; they sparsify (Top-K, 50%) and keep it int8; ReLU^2 sparsifies those inputs > 80%.
BitNet v2 (2504.18415): down-proj and o-proj inputs carry the outliers; a Hadamard transform before quantisation fixes
it. Massive Activations (2402.17762): fixed dimensions of the *residual stream* are ~100,000x the median in LLaMA2-13B;
that is a residual-stream effect, not an MLP-activation one.

## 2. What matters for our setting

- **Scale**: 13.8M params (10.6M non-embedding), d=384, mlp_hidden=1024, ~200M tokens, i.e. ~19 tokens per non-embedding
  parameter. Only Primer (110M-500M), MoA (0.12B, 20 tokens/param) and the TurboSparse 300M pretrain are near that range.
- **Param matching.** Our SwiGLU is `up: d -> 2h`, `down: h -> d`. An ungated activation matches it at h' = 1.5 h = 1536
  (computed: 2*384*1536 = 3*384*1024 weights). Both have 3.54M weights per MLP. xIELU's authors do the same (9216 vs 6144).
- **Bandwidth.** Unfused elementwise traffic per token is equal (computed: SwiGLU 2048 write + 2048 read + 1024 write +
  1024 read = 6144 elements; ungated at 1536: 4 x 1536 = 6144). With fusion into the matmul epilogue the ungated one is
  simpler (one input). I have no measurement; the real answer needs the GPU (SQ164).
- **BitLinear sees per-token absmax int8** (`gogeta/model.py` BitLinear: scale = 127/absmax over the feature dim). The
  quantity that matters is the per-token crest factor max|h|/rms(h) at the *input of `down`*. Ungated ReLU has 7.7 and the
  gated products 15-20 on E0's real pre-activations (section 4), but the resulting int8 error is 2-3.6% relative L2 for all
  of them: **[speculation]** the int8 error is dominated by the few largest entries and is not the bottleneck for loss; the
  ternary weights are.
- **Looped core.** Pre-norm means every MLP input has fixed RMS. The only way an activation changes the loop dynamics is
  through the size/direction of the increment it produces. A bounded activation (tanh) cannot contract anything because the
  increment is added to an unbounded residual. MoA's finding that tanh hurts in ReLU-type FFNs (0.132 worse than ReLU^2 alone)
  is a warning against "make it bounded".

## 3. Recommendations per component

| Component | Recommendation | Confidence | Why |
|---|---|---|---|
| (a) MLP under ternary BitLinear | Keep SwiGLU as the control. First challenger: **ungated ReLU^2, h = 1.5 h** (`mlp_act: relu2`); second: **xIELU**. Decide by SQ160 (fp) and SQ161 (ternary) | Low-medium | 2B4T, Nemotron-4, a4.8 all use ReLU^2 at scale; sparsity and a simpler kernel; but small-budget evidence favours GLUs and nobody ablated it under ternary at our size |
| (b) GDN short-conv and gates | **Keep SiLU in ShortConv** (convention in Mamba/GDN; no contrary evidence found). Output gate: test **sigmoid vs SiLU** (this is SQ8, already queued as an idea) | Medium for the gate (a gate ablation exists: sigmoid 5.761 vs SiLU 5.822 ppl, 15B MoE), low for conv | Bounded sigmoid gate gives sparsity and fewer spikes in 2505.06708; the beta/decay paths are already sigmoid/softplus by design |
| (c) Looped core | **No activation change.** Do not try tanh or other bounded MLP activations. Treat K6's drift as an injection/step-size problem | Medium | Pre-norm fixes the scale; Loop-µP/Parcae put the growth in the shared increment; MoA: tanh hurts |
| (d) Injection (`inject` linear on [norm h; norm e]) | Linear, no activation, as now | Medium | Parcae's analysis is about a (diagonal, contractive) linear injection; adding a nonlinearity makes ρ(A) harder to bound |
| MLP *inside* the core vs prelude/coda | Same activation everywhere at first; per-region mixing only as SQ166 | Low | No evidence either way |

## 4. CPU measurement (measured here, one checkpoint)

`notes/research-activations/act_stats.py` -> `notes/research-activations/results/act_stats.json`. E0 checkpoint
(`E0-20261122-122059-s0`, 6 attention blocks, SwiGLU, fp32, CPU 2 threads, ~1 min), 8 sequences x 512 val tokens = 4,096
tokens, all 6 layers averaged. I hook each MLP's `up` output (a, b), apply each candidate to those *same real pre-activations*,
and measure the tensor that `down` (a BitLinear in ternary models) would quantise. **Caveat: the weights were trained for
SwiGLU, so this shows the distribution shape each form imposes, not what a model trained with it would learn.** Ungated
variants use only `a`. xIELU uses its init alphas.

| Candidate (input of `down`) | sparsity (|h| < 0.01 rms) | excess kurtosis | per-token crest max/rms | global max/rms | int8 per-token rel. L2 err |
|---|---|---|---|---|---|
| SwiGLU silu(a)*b | 0.056 | 6777 | 15.0 | 107 | 0.0357 |
| GEGLU | 0.101 | 6208 | 15.5 | 104 | 0.0349 |
| ReGLU | 0.706 | 5671 | 15.0 | 101 | 0.0208 |
| ReLU^2-GLU | 0.778 | 28545 | 20.1 | 196 | 0.0217 |
| dReLU | 0.852 | 6088 | 16.3 | 123 | 0.0170 |
| ReLU^2 (ungated) | 0.727 | 4258 | 14.7 | 90 | 0.0191 |
| ReLU (ungated) | 0.695 | 16 | 7.7 | 18.5 | 0.0093 |
| GELU (ungated) | 0.028 | 18 | 8.1 | 19.9 | 0.0176 |
| SiLU (ungated) | 0.006 | 16 | 7.6 | 19.9 | 0.0173 |
| xIELU (init) | 0.046 | 2427 | 12.4 | 76 | 0.0297 |
| reference: norm2 output (the `up` input) | 0.008 | 0.9 | 4.0 | 7.9 | 0.0091 |
| reference: N(0,1) | 0.008 | 0.0 | 3.5 | 5.2 | 0.0079 |

Reading it: (i) every product form (GLU) is 300-1700x heavier-tailed in kurtosis than the ungated ReLU/GELU/SiLU, so the
"GLU makes outliers" folk claim holds on this checkpoint; (ii) ReLU-type gates give 70-85% structured zeros at the `down` input
(consistent with a4.8's > 80%) even on weights never trained for them; (iii) int8 per-token absmax error stays at 1-3.6%
regardless, so **int8 BitLinear is not outlier-limited here**; the statistic only gets scary for int4. This is a single
fp32 attention-only checkpoint, 4,096 tokens, one seed; the trained-with-the-activation answer is SQ161's secondary readout.

## 5. Combining activations (the human's question)

| Idea | Evidence | Verdict |
|---|---|---|
| **Learnable linear blend of activations** (per layer or per channel), input-independent | MoA/LA (0.12B, 20 tokens/param): bi-LA -0.016, one-LA -0.015 vs SwiGLU; but in ReLU-type FFN the LA was +0.003 (no gain over ReLU^2). Older: SG-Blend (2505.23942, per-layer blend of Swish/GELU, vision-scale; not downloaded; found via search) | The best-supported combination. Cheap. Test as `swiglu_relu2mix` (SQ162) |
| **Token-adaptive mixing** (MoA gate) | Same paper: MoA = LA in Type-II at best (-0.016 both), better in Type-I (-0.015 vs +0.003) | More machinery than we need for a 13M model; skip until LA works |
| **Per-neuron routed choice** (PolyGLU) | One 597M run, no SwiGLU control; routing collapses to one-hot; deep layers pick Tanh | Interesting pattern (early GELU, deep Tanh) but zero evidence of benefit; MoA says tanh hurts in ReLU-type FFNs. Do not build |
| **Polynomial composition** (PolyCom) | 1.3B/250B tokens: 2.19 -> 2.17 loss; OLMoE 2.41 -> 2.39; no seeds; order 3 fixed | xIELU already is a trainable quadratic with a negative branch, with a published ablation; PolyNorm's extra norms cost elementwise passes. Lower priority |
| **Activation x gating** (ReLU^2-GLU, dReLU) | a4.8, TurboSparse: sparsity, not quality | Only if we want > 80% structured sparsity; worst tails in section 4 |
| **Different activation per region** (prelude / core / coda) | None found | **[speculation]**: the core is applied r times with shared weights, so a *less nonlinear* core (xIELU's learned alphas shrink with depth: "less nonlinear for deeper layers" per 2411.13010) might be gentler; SQ166 |
| **KAN-style splines** | No LM evidence | No |

## 6. Experiments (cheapest first)

All at TINY scale (d=128, 2 heads of 64, mlp_hidden 256, the E1-style layout `prelude [swa] / core [gdn,gdn,swa] x3 / coda [attn]`,
10M tokens, code-py-v2, 3 seeds), est. <= 5 GPU-min per arm (an estimate, not measured). Not run (a 7-hour campaign is using the GPU). Configs are not committed (`configs/` is off limits): they are
`{**E1_tiny, "mlp_act": X}` plus the `mlp_hidden` unchanged (the patch handles the 1.5x width).

**SQ160 (fp, 4 arms: swiglu, relu2, xielu, swiglu_relu2mix; matched params, matched tokens).**
Prediction (before running): relu2 and xielu land within +-1% of SwiGLU's val loss (nats); relu2 is worse than SwiGLU by
0.3-1% (early-budget GLU advantage; xIELU's own 4B-token ablation: 18.12 vs 17.58 ppl = +3.0% ppl = +0.030 nats on ~2.9);
xielu <= relu2; the mix ends within +-0.3% of SwiGLU (LA gain at 0.12B was 0.5%, below our seed noise).
Criterion: challenger "wins" only if it beats SwiGLU by > max(0.5%, 2 x seed std) on 3 seeds; "loses" if worse by the same
margin; otherwise "tie" and SwiGLU stays.

**SQ161 (same 4 arms with `ternary: true`; the arm that matters).**
Prediction: gaps shrink or flip toward relu2 [speculation]: ternary-gap (val loss ternary - fp) for relu2 is smaller than
for SwiGLU by >= 0.01 nats. Secondary readout, free: down-input sparsity, kurtosis and int8 error of each trained model with
`act_stats.py`-style hooks (predict relu2 sparsity > 60%; SwiGLU < 10%). Criterion: relu2's ternary gap < SwiGLU's by > 0.01 nats
and 2 seed std.

**SQ162 (the blend `swiglu_relu2mix`, diagnostic, rides on SQ160's runs).** Read the learned s = sigmoid(mix) per layer after training.
Prediction: s stays < 0.2 in most channels (SiLU wins, mean s < 0.15); if s moves to > 0.5 in deep layers, the blend is
choosing ReLU^2 and the A/B in SQ160 should show relu2 <= swiglu. Criterion: report mean and spread of s per layer.

**SQ163 (conv activation / output gate; extends SQ8):** ShortConv SiLU vs identity, GDN output gate SiLU vs sigmoid.
Prediction: sigmoid gate >= SiLU gate by 0.2-0.6% loss (Gated Attention: ppl 5.761 vs 5.822, i.e. -1.0% ppl); removing the
conv SiLU changes loss by < 0.3% (GDN can absorb it in the next linear). Criterion: 0.5% / 2 seed std.

**SQ164 (GPU microbenchmark, 2 min, after the campaign):** fused fwd+bwd time of SwiGLU vs ReLU^2 vs xIELU MLP at B=32, T=1024, d=384
in bf16 with `torch.compile`; reuse `experiments/checks/sq_hw_step_profile.py`. Prediction: ReLU^2 within -5% of SwiGLU's MLP
time; xIELU +0-10% (expm1 branch); neither moves the whole step by > 3%.

**SQ165 (loop probe, inference only, rides on SQ161's checkpoints):** K6 on the swiglu and relu2 ternary looped models.
Prediction: norm growth N12 = ||h_12||/||h_4|| changes by < 15% between activations (prediction: the activation is not the lever). Criterion: if relu2's N12 differs by > 25%, revise (c).

**SQ166 (per-region activation, only if SQ160 finds a winner):** relu2 only in prelude/coda vs only in the core. **[speculation]**.

## 7. What is not verified

- No training was run; all quality numbers are reported by the papers, at 0.12B-3B (the smallest controlled evidence is
  MoA at 0.12B; most are >= 1B), with at most one seed. xIELU's/PolyCom's gains are 1-3% perplexity without variance.
- The CPU table is on SwiGLU-trained pre-activations; it says nothing about what a ReLU^2-trained ternary model does.
- I read the PDFs by text extraction; a few table cells in MoA (relative-loss units, row labels) were garbled and I used the
  paper's own prose to disambiguate. Their "relative loss" unit is not defined in the part I read.
- KAN's cost, SoLU, Mish are **[recall]**. 2025-2026 search covered activations for ternary, SSM gates and loops by web search
  only; the only activation-adjacent loop paper hits were about normalisation and residual scaling (Parcae 2604.12946,
  2607.10681 LayerNorm gain: d <= 32 CPU study, DeepLoop 2607.13491 residual scaling; the last two not read).
- I did not look for FP8/int4-specific activation papers beyond BitNet a4.8/v2.
