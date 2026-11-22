# Review: projects close to Gogeta-0, and what to test because of them

*Research side quest, 2026-11-22. Ledger rows C50–C69 and side quests SQ30–SQ41, all pending review.*
*Nothing here was run on a GPU. Every number below is quoted from a source ("reported by …") or computed by hand ("computed"); none is verified locally (AGENTS.md §0).*

Started from an externally pasted analysis whose claims were unverified. Each one was checked against primary sources: arXiv API metadata, the paper PDFs (`pdftotext`), GitHub source, Hugging Face model cards, configs and training logs. Cross-references:
- The concurrent looped-transformer review (`notes/research-looped.md`, C30+/SQ14–SQ22) appeared only **after** this note was drafted. Both reviews independently reached the same K6 "drift" reading. Overlaps are mapped in **"Overlap map"** at the end of (d). **Short version:** run its SQ14/SQ15/SQ18 instead of this note's XP-1/XP-2 (fold in the extra arms listed there). The items unique to this review are XP-3 (MixerLoop matched), XP-4 (regime), XP-9's QAT arm, XP-7 and XP-8.
- The Kimi Linear review (`notes/research-kimi-linear.md`, C70–C79/SQ50–SQ59) exists. Its SQ54 (channel-wise gating in the looped core) touches XP-2 below.

## Sources actually read

| Source | What was read |
|---|---|
| arXiv 2608.18230 (MixerLoop) **v1 and v2** | Both PDFs in full, plus the arXiv API record. The GitHub repo `ruhai-lin/MixerLoop`: `custom_models/mixerloop/layers.py`, `modeling_mixerloop.py`, `fullloop/modeling_fullloop.py` |
| Auron | HF API and model cards for `nyxia/Auron-279M`, `-510M`, `-1.1B`; their `config.json`; all three `training.log` files; the HF blog post *"Auron: Depth-Efficient Language Models via Hybrid Recurrent-Attention Weight Sharing"* (2026-04-16); the **safetensors header only** of Auron-510M (a 15.7 KB HTTP range request for tensor names and shapes, no weights); Wayback CDX for the GitHub repo |
| arXiv 2602.12078 (TR-Mamba2-Attn) | The local PDF in `refs/05` in full |
| arXiv 2605.20670 (LT2) | The PDF: method, §3.1–3.5, appendix C. **Not in the pasted analysis, but the closest published competitor** |
| arXiv 2611.19446 (looped hybrids + KV replay) | The PDF: method, PARCAE section, anchor-stride results |
| arXiv 2502.05171 (Huginn) | The local PDF: §3–4.3 |
| arXiv 2411.15242 (Zamba2) | The PDF: architecture section |
| arXiv API abstracts | 2605.22791, 2510.26692, 2402.17764, 2510.25741, 2605.09165, 2604.12946, 2605.16343, 2610.05265, 2609.30820, 2610.13041, 2606.18114, 2604.21254, 2609.29397 |
| arXiv API searches (2026-11-22) | looped/recurrent-depth × ternary/BitNet; weight sharing × ternary; looped × DeltaNet; looped LM × quantization; ternary × linear attention/SSM |
| Our repo | `README.md`, `gogeta/model.py`, `configs/e1_hybrid.json`, `configs/e2_looped.json`, `experiments/writeups/E2.md`, `claims.md`, `side-quests.md`, `checks/PREDICTIONS.md`, `checks/results/K3,K5,K6.json` |

---

## (a) Fact-check of the pasted analysis

| # | Pasted claim | Verdict | What's actually true | Source |
|---|---|---|---|---|
| 1 | MixerLoop exists, Aug 2026, arXiv 2608.18230 | **confirmed** | *"Allocating Recurrent Compute in Looped Language Models"*, Lin, Guo, Zhu, Ye, Eshraghian (UC Santa Cruz). v1 posted 2026-08-18, **v2 2026-11-12** | [arXiv API](http://export.arxiv.org/api/query?id_list=2608.18230) |
| 2 | Gated DeltaNet with shared-weight recurrence that repeats **only** the GDN mixer and runs the FFN once | **confirmed, with a nuance** | The loop is **per physical layer**: each layer's GDN mixer is applied T=4 times in a row with the same weights, then that layer's FFN runs once. Physical layers keep distinct weights. There is no shared stack looped as a whole. The backbone is **pure GDN** (no attention). Code: `MixerLoopBlock.forward` | [layers.py](https://github.com/ruhai-lin/MixerLoop/blob/main/custom_models/mixerloop/layers.py) |
| 3 | "Keeps much of the recurrence benefit while cutting recurrent-backbone projection FLOPs by 45.9% in a 110M comparison" | **version-dependent, and "much" overstates v1** | **v1** (15M/110M, 52.43B tokens each): at 110M MixerLoop keeps **41.5%** of FullLoop's CORE gain (NoLoop 14.16, MixerLoop 15.56, FullLoop 17.52). Held-out NLL: 2.401 / 2.377 / 2.342, so ~41% of the NLL gain. 45.9% is an *analytic* projection-FLOP figure (FFN = 61.2% of per-layer projections). End-to-end FLOP saving is 43.0% (110M) and 33.9% (15M). At 15M MixerLoop has the best CORE (6.52 vs 5.52), but FullLoop has the better NLL. **v2** (13M/100M, 10B tokens, new seeds) reports **43%** and **MixerLoop ≥ FullLoop on CORE at both scales** (100M: 11.24 vs 10.89 vs NoLoop 8.94). The arXiv abstract field (API record of v2) still carries the v1 numbers; the v2 PDF abstract does not | v1 §3.3, §4.2; v2 Table 2, §4.3 |
| 4 | Auron: GitHub `Fy-/Auron`, HF `nyxia/Auron-510M`, Apr 2026 | **partly wrong** | The HF models exist, **created 2026-03-22**. The paper is dated March 2026; the HF blog was published 2026-04-16. **`github.com/Fy-/Auron` returns 404** (as of 2026-11-22; the GitHub user exists). Wayback has only the PDF blob page (2026-03-23), whose file tree shows an inference package `ouro/` and `run_infer.py`, no training code. Not on arXiv: the blog says the submission was rejected | HF API; [blog](https://huggingface.co/blog/nyxia/hybrid-recurrent-attention-weight-sharing); Wayback CDX |
| 5 | Auron = GDN + attention + shared-weight depth + repeated access to the original input; "Chimera" = unique lower layers, shared upper layers | **confirmed** | GDN:GQA = 3:1 (`attn_interval=4`). 510M: 4 unique bottom blocks + 4 physical top blocks × 3 loops = 16 virtual. x0 injection: `x0_lambdas.0..15`, one learned **scalar per virtual layer** (checkpoint header) | `config.json`; safetensors header; blog §3 |
| 6 | Trained at 279M / 510M / 1.1B; 510M best; 1.1B regressed | **confirmed (as reported)** | Final val loss 3.188 / **3.035** / 3.180 (in the training logs too) | model cards; `training.log` |
| 7 | 1.1B regression, "cause unresolved" | **partly right** | The author *asserts* a root cause ("representation saturation" at dim 2048, plus head_dim=64 "fragmentation"), but the Limitations section says it was "not isolated". A "Chimera 1B v2" (head_dim 128) test was planned; no result found. **Confounds the author doesn't discuss** (from the logs): identical LR (Muon 8e-4, embeddings 4e-4), batch 16 × 2048 = 32,768 tokens/step and 250k steps at every scale, so **≈8.2B tokens processed (≈1.6 epochs of a 5B-token mix)** for all three. That is ~7.6 tokens/param for 1.1B. The 1.1B also changes topology (6+6×3 = 24 virtual vs 16). And **the 1.1B ends with a *higher train* loss (2.86) than the 510M (2.72)**: it under-fits rather than saturates. The blog's "~4.1B tokens at batch 16" contradicts its own logs ("Tokens/step: 32,768" at batch 16) | blog §5.3, §6; logs |
| 8 | TR-Mamba2-Attn, arXiv 2602.12078, already in refs/05 | **confirmed** | *"Tiny Recursive Reasoning with Mamba-2 Attention Hybrid"*, Wang & Reid, 2026-02-12 (v2 2026-03-13), ICLR 2026 Latent & Implicit Thinking workshop. The file is in `refs/05-looped-recursive/` | arXiv API; local PDF |
| 9 | Mamba-2/attention hybrid inside TRM, ~7M params | **confirmed** | 6.86M (hybrid) vs 6.83M (TRM-attn). Block = Mamba-2 → Mamba-2 → Attention → MLP with **post-norm**. Attention is **bidirectional** (grid puzzles, not causal LM) | §3.1–3.2 |
| 10 | ARC-AGI-1 pass@2 43.88% → 45.88% | **confirmed, but cherry-picked** | Correct, and pass@1 is −0.25. The pasted analysis omits that on **Sudoku-Extreme the hybrid is worse: 66.5% vs 72.2%** | Tables 1–2 |
| 11 | Huginn = recurrent depth | **confirmed** | arXiv 2502.05171 (Geiping et al.) | arXiv API |
| 12 | Zamba2 = shared attention blocks | **confirmed** | Two alternating shared attention blocks (one in the 1.2B), non-shared LoRAs per position | arXiv 2411.15242 §II |
| 13 | Kimi Linear, BitNet b1.58 | **confirmed (exist)** | 2510.26692, 2402.17764 | arXiv API |
| 14 | Gated DeltaNet-2, arXiv 2605.22791 | **confirmed** | *"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"*, Hatamizadeh, Choi, Kautz, 2026-05-21 (v2 2026-11-06) | arXiv API |
| 15 | Nobody has published GDN-hybrid + recurrent-depth weight sharing + BitNet ternary together | **couldn't verify absence, and not refuted by my searches** | But **two of the three are already combined in public work** the pasted analysis missed: **LT2** (2605.20670) loops GDN and Full+GDN hybrids at 0.6B/1.3B, and 2611.19446 loops GDN/mLSTM hybrids with attention anchors. On the quantisation side there's only PTQ for looped models (LoopQ W4A4, Loopy, RAMP, "feedback exposure" INT4; MixerLoop's FPGA build is W8A8) and ternary QAT for *non-looped* SSMs (Ternary Mamba 2606.18114). **No ternary-QAT looped model was found** | arXiv API searches listed above |
| 16 | Suggested: loop only the GDN mixer (MixerLoop-style) vs our full-block loop at matched FLOPs/params | **good idea, and neither MixerLoop paper did it** | Both MixerLoop versions match **params and tokens, not FLOPs**: FullLoop spends ~1.85× MixerLoop's backbone FLOPs and NoLoop ~0.46×. No FLOP-matched NoLoop (deeper/wider) or FLOP-matched FullLoop (r=2) exists. See XP-3 | computed from v1 Eq. 6–10 |

---

## (b) Per-project comparison against Gogeta-0

Gogeta-0 (from `gogeta/model.py`, `configs/e2_looped.json`): prelude `[SWA]` → core `[GDN+MLP, GDN+MLP, SWA+MLP] × r` → coda `[full attn]`, d=384. Each pass: `h = h + inject([norm(h); norm(e)])`, with `inject` a full 2d→d linear, then the 3 pre-norm blocks. s0 = e. Train r ~ U{1..7}, backprop through the last 4 loops.

| Axis | **Gogeta-0 (E2)** | **MixerLoop** (2608.18230) | **Auron** (HF blog + checkpoint) | **TR-Mamba2-Attn** (2602.12078) | **LT2** (2605.20670) | **Huginn** (2502.05171), for reference |
|---|---|---|---|---|---|---|
| Mixer type / ratio | GDN : SWA : full = 2:1 in the core, plus SWA prelude and full-attn coda | Pure GDN, no attention | GDN : GQA = 3:1 | Mamba-2 : bidir. attention = 2:1, then one MLP | GDN, KDA, DSA…; hybrids 4:1 (linear:full) | Full attention |
| What's shared and looped | 3 full blocks (mixer+MLP) as one stack, r times | **Each layer's mixer, T=4 times in a row; FFN once; layers not shared** | Top zone: 4 (6 for 1.1B) physical full blocks × 3 loops; 4 (6) unique bottom blocks | TRM recursion: H=3 outer × L=4–6 inner over one 4-sublayer operator | Whole physical stack (16–25 layers) × T=4 | 4-layer core × r |
| Input re-injection | `h + W·[norm(h); norm(e)]` (2d² params), once per loop | **None from the embedding.** Per-iteration channel gain: `h ← h + mixer(norm h) + ρ_i ⊙ h_in`, ρ zero-init, per iteration, shared across layers | `+ λ_l · x0` per **virtual** layer (scalar); per-virtual-layer "layer-ID" vector added to the **GDN mixer input only**; U-Net skip vectors (8 × d) | TRM's z_H/z_L state with input embedding (unchanged from TRM) | **None from the embedding.** Per-loop `h(τ) = stack(h(τ−1)) + ρ_τ ⊙ h(τ−1)`, ρ zero-init | Concatenate [s; e] → adapter A (2h→h), **not residual** |
| Loop-stability tricks | Pre-norm blocks; residual-out init scaled by effective depth at r=4. **No norm at the loop boundary** | ρ (can learn contraction); mixer out-proj init 0.02/√(2LT); final norm | Residual scalars `resid_attn`/`resid_mlp` per physical block, **init √1.1 (amplifying)**; embedding RMSNorm; **no learnable RMSNorm weights in the checkpoint** (only GDN `o_norm`), so norms are parameter-free | **Post-norm everywhere** (argues pre-norm residual grows ~√t or worse) | ρ_τ; recommends a head-wise sigmoid **SDPA output gate** (fixes compounding sinks); reports looped GDN has the smoothest grad norms | Sandwich norm (4 norms/layer) + core-end norm n_c; embedding scale; random s0 |
| Train loop count / BPTT | r ~ U{1..7} (mean 4), last 4 loops get gradients | **Fixed T=4**, full backprop | **Fixed 3 loops**, full backprop (as far as described) | Fixed TRM schedule (deep supervision from TRM) | **Fixed T=4**, full backprop | r ~ log-normal Poisson (mean 33, heavy tail), last k=8 |
| Halting / adaptive depth | None (E5 planned) | None | None (Part II talks of "adaptive depth" but trains fixed) | TRM Q-halt head | Discussed in appendix A, not used | Zero-shot KL-threshold exit |
| Extrapolation beyond trained r | Measured: **worse** (r4 0.619 → r12 0.654) | Not tested; only truncation T=1..4 (100M CE 5.016 → 2.558: T=1 is catastrophic, not graceful) | Not possible as built (layer-IDs exist for 3 loops only); not tested | Not applicable | Not tested | Gains up to ~20–30+ iterations, task-dependent saturation; warns that step-conditioned cores can't extrapolate |
| Quantisation | BitLinear ternary QAT planned (E3), **includes `inject`** | FPGA decoder W8A8 (groups of 32), states FP32 (post-hoc) | None | None | None | None (later third-party PTQ: 2609.30820) |
| Scale / data | 12.6M total (9.5M non-emb), 200M Python tokens, 1 seed | 13M/100M (v2, 10B ClimbMix tokens, 3 seeds at 13M); 15M/110M (v1, 52B tokens); 450M on 100B FineWeb-Edu | 279M/510M/1.1B total (123M/277M/761M unique), ~8.2B processed tokens (5B mix: web, code, math, chat), 1 run each | 6.9M, ARC-AGI-1 / Sudoku / Maze | 0.6B/1.3B, 100B FineWeb-Edu, 1 seed each | 3.5B, 800B tokens |
| Results | Loop costs 2× FLOPs, +2% vs E0, +6% vs E1 | v1: Mixer between No and Full at 110M; v2: Mixer ≥ Full on CORE | 510M val 3.035; no dense baseline ("no controlled ablation", their Limitations) | ARC pass@2 +2.0, Sudoku −5.7 | Looped GDN ≈ looped Transformer (better at 1.3B); Full+GDN hybrid with gate best (PPL 9.03 vs 9.39) | Large gains on reasoning benchmarks with more recurrence |
| Reported failure modes | Non-convergent loop; worse beyond r≈4 | v1: at 110M FullLoop clearly better than MixerLoop | 1.1B "scaling wall"; "attractor wells" needing presence penalty ≥ 1.5 when sampling; fully shared stacks had a "parsing bottleneck" ("geometric gravity", repetition) → two-zone design | — | Looped RetNet diverges; looped DeltaNet less stable; sinks and massive activations compound per loop; residual RMS grows ~20× by loop 4 | Bad runs 1–2: token-correlation collapse; a run that **ignored the recurrent state** (same PPL at r=1 and r=32), fixed by sandwich norm and a lower LR |

### Auron: what the code does (and what couldn't be checked)

The training/inference code **could not be read**: the repo is 404 and no source snapshot is archived. What is known about the loop and injection comes from (1) the blog's §3.4, (2) the `config.json` flags `use_x0_inject`, `use_resid_lambdas`, `use_skip_connections`, and (3) tensor names and shapes in the 510M checkpoint header:
- `x0_lambdas.0` … `x0_lambdas.15`, shape `[1]`: one scalar per **virtual** layer, so each loop pass gets its own injection strength even though the block weights are shared. Per the blog, `layer_input += λ_l · x0`, where x0 is the normed embedding output.
- `top_layer_ids`, shape `[3, 4, 1536]`: one d-vector per (loop, top block). The blog says it's added to the **GDN mixer input only**, "not the residual stream". That's iteration conditioning: the shared block knows which pass it's in.
- `skip_weights.0..7`, shape `[1536]`: U-Net skips (layer i → layer 15−i).
- `top_blocks.N.resid_attn` / `resid_mlp`: scalars per *physical* block, shared across loops, init √1.1.
- Attention blocks: `q_proj [1536,1536]`, `k_proj/v_proj [384,1536]` (GQA, 6 KV heads); no QK-norm weights; no learnable RMSNorm weights at all.
- The pattern (x0 lambdas, U-Net skips, resid lambdas, parameter-free norm, Muon) closely resembles the modded-nanogpt speedrun recipe. That's an observation, not a claim from the author.

One more claim **can't be verified and might hide a bug**. The blog says GDN suits sharing because "the hidden state naturally evolves across iterations … the accumulated recurrent state from prior loops". If that means passing the GDN *final* state of loop i as `initial_state` to loop i+1, it would leak future tokens into every position during training (non-causal). We should never do that. In our code each loop's GDN starts from a zero state (`_kernel(...)` with no `initial_state`), as does MixerLoop's.

---

## (c) Approach differences that matter for our E2 failure

**First, a re-reading of our own K6 result.** K6 (`checks/results/K6.json`) recorded ‖h_k‖/‖h_0‖ and the relative change. The absolute per-loop increment can be recovered from them (computed: rel_change_k × norm_ratio_{k−1}):

| k | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| ‖h_k‖/‖h_0‖ | 12.7 | 17.6 | 22.4 | 31.7 | 40.9 | 49.9 | 58.9 |
| ‖h_k − h_{k−1}‖/‖h_0‖ | 5.60 | 5.18 | 4.98 | 4.79 | 4.70 | 4.65 | 4.61 |

The norm grows by ~4.5–4.9 per loop, while each increment is ~4.6–5.0. Norm growth ≈ increment size means the increments are **almost parallel to each other**. The E2 loop behaves like **h_k ≈ h_1 + (k−1)·v**: a near-constant drift in a fixed direction. The "falling relative change" is mostly the denominator growing, not convergence. Since every block and the coda are pre-norm, the input-dependent part of h is diluted by ~1/k. Each later pass also writes a residual update that is ever smaller *relative* to the stream. Both effects predict a slow loss rise beyond the trained range, which is what we see. This is a hypothesis built from K6's own numbers; XP-1 tests it directly.

What the others do differently, ranked by how directly it bears on that failure:

1. **Every competitor controls the state's scale at the loop boundary. We don't.** TR-Mamba2 uses post-norm, and argues pre-norm grows with t. Huginn uses sandwich norm plus a core-end norm, and says it was *required* at scale. LT2 and MixerLoop learn a zero-init per-channel ρ on the loop residual, which can learn a contraction. Parcae (2604.12946, cited by 2611.19446) uses a decaying diagonal adapter, designed "to prevent recurrent drift". Auron uses residual scalars plus per-layer x0 scalars. → **XP-2.**
2. **Nobody else trains variable depth *and* claims extrapolation, except Huginn.** MixerLoop, LT2 and Auron fix T and never evaluate past it. So C7 (worse beyond r≈4) isn't evidence that we're uniquely broken; the field mostly sidesteps the question. Huginn, the one that scales r at test time, trains with a heavy-tailed r (mean 33, tail past 100), random s0 and k=8. Ours is U{1..7} with s0 = e, which Huginn says makes the solution depend on boundary conditions. → **XP-5** *(overlap: looped review)*.
3. **Regime: weight sharing wins where parameters are the bottleneck.** Computed from MixerLoop's Tables 5–6: its 13M model has ~4.7M non-embedding params and saw 10B tokens (≈2,100 tokens per non-embedding param); v1's 15M saw 52B (thousands per param). E2 saw **≈21 tokens per non-embedding param** (200M / 9.46M). At 21 tokens/param a model is data-limited, not parameter-limited, so extra depth-for-free buys little. Auron's 1.1B (≈7.6 tokens/param) "regression" fits the same story. → **XP-4.**
4. **The comparison axis.** MixerLoop, LT2 and Auron all compare at **matched params** (and Auron against no baseline at all), with the looped model spending T× the FLOPs. E2 *also* lost at matched params, a harder negative than anything they report, but at 2× FLOPs. None of them reports a FLOP-matched non-looped control, so "loops help" in the literature is not "loops help per FLOP". → XP-3 and XP-4 include FLOP-matched controls.
5. **Loop granularity.** We loop the MLPs too, which are ~63% of the core's projection FLOPs (computed below). MixerLoop's evidence says the mixer is where the loop's value is. → **XP-3.**
6. **Mixer placement.** LT2 finds looped GDN the most stable looped mixer, and finds softmax sinks compound across loops. Our core loops one SWA per pass with no output gate. 2611.19446 finds looped hybrids "require sufficiently frequent attention anchors"; our in-loop anchor is a *windowed* one. → **XP-6.**
7. **Iteration conditioning.** Auron (layer-IDs, per-virtual-layer λ), MixerLoop and LT2 (per-iteration ρ) all tell the shared block which pass it's in. Huginn tried that and found it prevents extrapolation. We have none, which is consistent with wanting variable r, but untested. → **XP-7.**

**Computed FLOP split** (analytic, E2 dims d=384, mlp_hidden=1024, matmul params per token-pass): GDN mixer ≈ 0.742M, SWA mixer 0.590M, SwiGLU 1.180M, inject 0.295M. One core pass = 2.074M (mixers) + 3.539M (MLPs) + 0.295M (inject) = 5.91M, so the MLPs are 60% of a pass (63% excluding inject). **MixerLoop T=4 on our core = 4×2.074 + 3.539 = 11.84M ≈ our full-block loop at r=2 (11.82M).** The same equality holds at d=128 with mlp_hidden=344 (1.314M vs 1.317M). That makes a clean FLOP- and param-matched comparison possible.

---

## (d) Ranked experiments

Common **tiny-LM harness** ("TINY"), proposed so these runs are comparable. It should go in `experiments/checks/common.py` when written:
- d=128, 2 heads (head_dim 64), mlp_hidden=344 (keeps E2's mixer:MLP FLOP ratio), our 8k tokenizer, seq_len 256, SWA window 64 (E2's 256/1024 scaled down);
- layout tiny-E2 = prelude `[swa]`, core `[gdn, gdn, swa]`, coda `[attn]`;
- batch 64 × 256 = 16,384 tokens/step, 1,200 steps ≈ **19.7M tokens** of `code-py-v1` train, LR 3e-3, cosine to 0.1×, warmup 100, AdamW (0.9, 0.95), wd 0.1, clip 1.0, seed 0;
- eval on a fixed set of 128 val windows × 256 tokens.

Computed: tiny-E2 has ≈1.05M non-embedding params, so 19.7M tokens ≈ **19 tokens per non-embedding param, the same regime as E2 (21)**. If a run takes over 3 min, cut steps proportionally and report it.

Metrics used throughout:
- **L(r)**: val loss at eval loops r ∈ {1, 2, 3, 4, 6, 8, 12, 16}. r > 7 is extrapolation for U{1..7} training.
- **K6 metrics**: ‖h_k − h_{k−1}‖/‖h_{k−1}‖, plus the two that K6 showed are needed alongside it: the norm ratio ‖h_k‖/‖h_0‖ and the absolute increment ‖h_k − h_{k−1}‖/‖h_0‖.
- **grad-norm p99.9**: pre-clip global grad norm over steps after warmup.
- **FLOPs**: `flops_per_token` at the mean training r (computed).

Single seed unless stated. Treat differences < 0.5% as ties.

### ★ XP-1 (top pick 1): is E2's loop a constant drift? Inference-only, on the existing E2 checkpoint, ~3 min GPU
- **Change:** none to training. Extend `k6_loop_convergence.py` on the same 16 val sequences, k = 1..16:
  - (a) the increment alignment cos(Δ_k, Δ_{k−1}), with Δ_k = h_k − h_{k−1}, and cos(Δ_k, h_{k−1});
  - (b) the absolute increment ‖Δ_k‖/‖h_0‖;
  - (c) a **norm-clamped eval**: for k > 4, rescale each token's h_k to its own ‖h_4‖ before the next pass and before the coda; loss at k = 6, 8, 12, 16;
  - (d) the same clamp applied only before the coda (isolates "coda sees a diluted input" from "later passes have shrinking relative influence").
- **Comparison:** same checkpoint, same FLOPs; the unclamped losses are K6's.
- **Prediction:** (a) cos(Δ_k, Δ_{k−1}) > 0.9 for all k in 3..12. (b) ‖Δ_k‖/‖h_0‖ in 4.4–5.2 for k = 3..12 (K6 implies 5.18 → 4.61). (c) The clamp cuts the k=12 excess loss L(12) − L(4) (K6 subset: 0.7598 − 0.7224 = 0.037 nats) by at least half. (d) Removes less than (c).
- **Pass criterion:** the drift hypothesis is **SUPPORTED** if (a) holds and (c) cuts the k=12 excess by ≥ 50%. It is **REFUTED** if any cos(Δ_k, Δ_{k−1}) < 0.5 for k ≥ 3, or (c) cuts the excess by < 10%.

### ★ XP-2 (top pick 2): loop-boundary normalisation, TINY, 4 runs × ~3 min
- **Change** (train r ~ U{1..7}, bptt 4, as E2):
  - **V0:** ours.
  - **V1:** Huginn-style core-end norm: `h = RMSNorm_c(core(h + inject(...)))` each loop (+d params).
  - **V2:** post-norm in the core blocks only: `x = norm(x + mixer(x)); x = norm(x + mlp(x))` (TR-Mamba2).
  - **V3:** Parcae-style decaying injection replacing `inject`: `h ← exp(−softplus(δ)·exp(A_log)) ⊙ h + softplus(δ) ⊙ W_e·norm(e)` (per-channel decay in (0,1); W_e is d×d, so d² fewer params than ours, reported).
  - Optional **V4:** one shared zero-init per-channel ρ on the loop residual (`h ← core(h_in) + ρ ⊙ h_in`, the LT2/MixerLoop gain, shared across iterations so it works with variable r).
- **Comparison:** tokens identical; FLOPs identical to <0.1%; params within 1.6% (V3 has −d² = −16k).
- **Metrics:** L(r) for r = 1..16; K6 norm ratio ‖h_12‖/‖h_4‖ and absolute increments; grad-norm p99.9.
- **Prediction:** V0 reproduces E2's shape (‖h_12‖/‖h_4‖ > 2, L(12) > L(4) by > 1%). V1 and V2 keep ‖h_12‖/‖h_4‖ < 1.3 and L(16) ≤ L(4)·1.01, with L(4) within ±1% of V0. V3 uncertain. V1/V2 have a grad-norm p99.9 ≤ V0's.
- **Pass criterion:** "boundary normalisation fixes extrapolation" **VERIFIED** if V1 or V2 has L(12) ≤ L(4)·1.005 **and** L(4) ≤ V0's L(4)·1.01. **REFUTED** if neither V1 nor V2 halves V0's relative excess L(12)/L(4) − 1.
- *Overlap:* Kimi review SQ54 (channel-wise gating in the core), SQ10 (gated residual). Run this first: it's the simplest control for both.

### ★ XP-3 (top pick 3): MixerLoop at matched FLOPs **and** params, TINY, 5 runs × ~3 min (+ a parity sub-check)
- **Change:** fixed loop counts, no r sampling, so FLOPs are exact.
  - **M0 NoLoop:** tiny-E2 at r=1 (= a tiny E9).
  - **M1 FullLoop r=2** with our inject.
  - **M2 MixerLoop T=4:** no outer loop and no `inject`. In each core block, apply `h ← h + mixer(norm1(h)) + ρ_i ⊙ h_in` four times, then `h ← h + mlp(norm2(h))` once. ρ_i is a zero-init d-vector per iteration, shared across the 3 core blocks (exactly `MixerLoopBlock`). GDN state starts from zero on every iteration.
  - **M3 FullLoop r=4** (E2's compute, 2× M1).
  - **M4 Unshared depth:** core `[gdn, gdn, swa, gdn, gdn, swa]`, r=1.
- **Comparison** (computed, core matmul params per token): M0 0.626M, **M1 1.317M, M2 1.314M**, M3 2.634M, M4 1.251M. Params: M0–M3 within 3.2% of each other (inject 32.8k; ρ 0.5k); M4 has +0.63M. **M1 vs M2 is the matched-FLOPs-and-params test.** M4 is "the same FLOPs spent on new parameters"; M3 is "the same params, double the FLOPs".
- **Metrics:** final val loss; loss vs cumulative train FLOPs; M2 eval at T = 1..8 (truncate or extend the inner loop); M1 eval at r = 1..8; per-block K6 norm ratio; grad-norm p99.9; wall-clock tokens/s. MixerLoop v1 measured only 1.11–1.12× faster at batch 1, so expect no speedup at tiny size.
- **Prediction:**
  - M2 ≤ M1 + 0.5%, i.e. at least a tie at matched FLOPs and params.
  - M2 < M0 by ≥ 1%.
  - M4 < M2: new parameters beat reuse at 19 tokens/param.
  - M3 < M1 by < 1.5%.
  - M2 wall-clock within ±10% of M1.
  - M2 doesn't extrapolate: L(T=8) > L(T=4).
- **Pass criterion:** "loop the mixer, not the block" **VERIFIED** if M2 < M1 by ≥ 0.5% on seeds 0 and 1. **REFUTED** if M2 > M1 by ≥ 0.5% on both. Otherwise a tie, which still favours MixerLoop: it frees the MLP for ternary/weight-streaming (see (e)).
- **Sub-check XP-3b** (K5 parity harness, ~1.5 min/run): core `[gdn]` as MixerLoop T=4 (MLP once) vs K5's (ii) (looped ×4, MLP ×4: 97.0%) vs FullLoop r=2. Prediction: MixerLoop T=4 ≥ 95%, because parity needs mixer depth, not MLP depth. **Supported** if ≥ 95%; **refuted** if < 90%.

### XP-4: tokens-per-parameter regime, TINY at d=64, ~4–5 runs, ≤ 15 min total
- **Change:** a tiny-E2 shrink (d=64, mlp 172, window 64; computed ≈ 0.26M non-emb params). Models: **NoLoop** (r=1) and **Loop** (fixed r=4). Budgets: 5M tokens (≈19 tokens/param, E2's regime) and 100M tokens (≈380 tokens/param). Add **NoLoop at 2.4× tokens** (FLOP-matched with Loop) at the 5M budget, and at the 100M budget if time allows. Seeds 0 and 1 at 5M.
- **Comparison:** Loop vs NoLoop at matched params and tokens; Loop vs NoLoop-2.4× at matched params **and** FLOPs (the deployment question: for fixed memory, spend FLOPs on loops or on data?).
- **Metric:** relative gap g = (L_Loop − L_NoLoop)/L_NoLoop at each budget; the same for the FLOP-matched pair.
- **Prediction:** g ≥ −0.5% at 5M (loops don't help, as in E2), and g ≤ −1.5% at 100M (loops help once parameter-limited). NoLoop-2.4× beats Loop at 5M by > 1%.
- **Pass criterion:** "weight sharing pays only when parameter-limited" **SUPPORTED** if g(100M) − g(5M) ≤ −1.5 points. **REFUTED** if g(100M) ≥ g(5M) − 0.3 points.

### XP-5: path independence and the r distribution, TINY, 3 runs *(overlap: looped review, SQ5)*
- **Change:**
  - **P0:** ours.
  - **P1:** random s0 ~ N(0, σ²) per token with σ matched to the RMS of e (Huginn), inject unchanged.
  - **P2:** P1 + heavy-tailed r (log-normal Poisson with mean 4, Huginn's Eq. 1–2 with σ=0.5, cap 24), bptt 4.
- **Comparison:** P0/P1 at identical FLOPs. P2's mean r is 4, but report its FLOPs (the tail raises cost slightly).
- **Metrics:** L(r) to r=24; a path-independence gap ‖h_k^(s0=e) − h_k^(s0=rand)‖/‖h_k‖ at k = 4, 8, 16.
- **Prediction:** P0's gap stays > 0.3 at k=16. P1/P2 bring the gap below 0.1 at k=16. P2 has L(16) ≤ L(4)·1.01, at a cost of ≤ 1% at r=4.
- **Pass criterion:** **SUPPORTED** if P2 meets both conditions; **REFUTED** if P2's L(16)/L(4) is no better than P0's by ≥ 1 point.

### XP-9 → rank 6: ternary × loop, TINY, 3 runs (the distinctive one; see (e))
- **Change:**
  - **T0:** FP tiny-E2.
  - **T1:** `ternary=True` (our current BitLinear, **including `inject`**).
  - **T2:** ternary with `inject` kept in FP. Motivation: 2609.30820's "feedback exposure", where INT4 PTQ failed mainly at Huginn's *non-residual* loop-entry adapter. Ours is residual, so this may not apply.
- **Comparison:** identical param counts and nominal FLOPs (ternary is simulated).
- **Metric:** ternary penalty ΔL(r) = L_tern(r) − L_FP(r) for r = 1..12; K6 norm ratio.
- **Prediction:** the penalty grows with depth: ΔL(12) ≥ 1.2·ΔL(4) for T1. T2's ΔL(12) ≤ 0.8 × T1's.
- **Pass criterion:** "ternary error accumulates through the loop" **VERIFIED** if T1's ΔL(12)/ΔL(4) ≥ 1.2. "Keep the loop-entry adapter in FP" **VERIFIED** if T2's ΔL(12) ≤ 0.8 × T1's, **REFUTED** if ≥ 0.95 ×.
- *Note:* best run after XP-2, since unbounded norm growth would confound the accumulation question.

### XP-7 → rank 7: iteration conditioning vs extrapolation, TINY, 4 runs
- **Change:**
  - **C0:** ours (no step signal, r ~ U{1..7}).
  - **C1:** C0 + a learned per-loop d-vector added to the GDN mixer inputs only (Auron's layer-ID); 7 entries, reuse the last for r > 7.
  - **C2:** fixed r=4 + per-iteration zero-init ρ_i (LT2/MixerLoop).
  - **C3:** fixed r=4, no conditioning.
- **Comparison:** FLOPs equal at r=4; params +7d (C1), +4d (C2).
- **Metric:** L(4); L(r) − L(4) for r = 6, 8, 12.
- **Prediction:** C1 and C2 improve L(4) by ≥ 0.5% over C0/C3, but extrapolate worse: L(8) − L(4) is larger than C0's (Huginn's warning).
- **Pass criterion:** "step conditioning trades extrapolation for in-range loss" **SUPPORTED** if both hold for C1. **REFUTED** if C1 is better or equal at both L(4) and L(8).

### XP-6 → rank 8: SDPA output gate on the in-loop SWA, TINY, 2 runs
- **Change:** add a head-wise sigmoid gate `y ← y ⊙ σ(W_g x)` after SDPA, before `o`, in the core SWA only (LT2 §3.4). Adds +d² params; shrink that block's MLP hidden by d²/(3d) ≈ 43 units to keep params matched.
- **Metrics:** first-token attention mass per loop (sink), max |h| per loop, grad-norm p99.9, L(4), L(12).
- **Prediction:** at 256 context and tiny scale the sink compounds only weakly. The baseline's loop-4 first-token mass is < 1.5× loop 1's. The gate improves L(4) by 0.2–1%.
- **Pass criterion:** LT2's "sinks compound across loops" **SUPPORTED locally** if the baseline ratio is ≥ 1.5×. The gate fix **VERIFIED** if it brings the ratio below 1.2× and lowers L(4) by ≥ 0.2%.

### XP-8 → rank 9: repetition "attractor wells", inference-only, existing E1/E2 checkpoints, ~10 min GPU
- **Change:** none. 64 val prompts (first 128 tokens each), 256-token greedy continuations (full recompute; no recurrent decode kernel, C2), E1, and E2 at r ∈ {2, 4, 8, 12}.
- **Metric:** fraction of repeated 4-grams within each continuation; longest repeated span.
- **Prediction:** E2@4's repeated-4-gram rate ≥ 1.2× E1's, rising for r > 4.
- **Pass criterion:** Auron's "weight sharing creates attractor wells" **SUPPORTED** for us if E2@4 ≥ 1.2× E1 and E2@12 > E2@4. **REFUTED** if E2@4 ≤ E1.

### Overlap map with the looped-transformer review (`notes/research-looped.md`)

| This note | Looped review | Recommendation |
|---|---|---|
| XP-1 / SQ30 (drift diagnosis, inference only) | **SQ14** (same idea: cos(Δ_k, h_{k−1}), rescale-before-coda, rescale-every-loop) | Run **SQ14**. Add XP-1's increment-alignment cos(Δ_k, Δ_{k−1}) as an extra printed number. SQ30's top-pick status transfers to SQ14 |
| XP-2 / SQ31 V1, V2 (core-end norm, post-norm) | **SQ18** (loop-boundary RMSNorm) | Merge into SQ18. Add V2 (post-norm core blocks, TR-Mamba2) as an arm |
| XP-2 / SQ31 V3 (Parcae decay injection) | **SQ15** | Duplicate: run SQ15 |
| XP-2 / SQ31 V4 (shared ρ gain) | SQ16 (1/r step scaling; near-identity loop-scale init variant) | Related, not identical. Add as an SQ16 arm if cheap |
| XP-5 / SQ34 (random s0, heavy-tailed r) | **SQ20** (per-sequence Poisson sampling) | Heavy-tail part overlaps SQ20. The **random-s0 / path-independence gap** part is new; add it to SQ20 |
| XP-6 / SQ35 (SDPA gate on the in-loop SWA) | **SQ22** | Duplicate: run SQ22 |
| XP-9 / SQ38 (ternary × loop) | SQ21 (post-hoc fake-quant of E2's core) | Complementary: SQ21 is post-hoc, SQ38 is **QAT** with the FP-vs-ternary `inject` arm. Run SQ21 first (free), then SQ38 |
| XP-3 / SQ32, XP-4 / SQ33, XP-7 / SQ36, XP-8 / SQ37, SQ39–SQ41 | none | Unique to this review |

After dedup, this review's own top picks are **XP-3 (SQ32)**, **XP-4 (SQ33)** and **XP-9 (SQ38)**, run after SQ14/SQ15/SQ18 have bounded the drift.

**Not proposed** (dropped as not cheap or not attributable): the pasted "3-axis efficiency benchmarking". It's mostly K1 (memory/time vs length) + K2 (BitLinear speed) + FLOP accounting, which already exist as checks; add a summary row once those run. Auron's two-zone prelude depth is logged as a side quest only (SQ39).

---

## (e) Where Gogeta-0 could genuinely be distinctive (based only on what was verified above)

1. **Ternary QAT inside a weight-shared loop.** No looped model trained with ternary (or any QAT) weights turned up. The closest are PTQ of looped transformers (W4A4) and ternary QAT of a non-looped Mamba-2. A tiny XP-9 result would already be new. This is not a proof of absence: re-search before any public "first" claim.
2. **Variable-depth training, with extrapolation measured, on a linear-mixer hybrid.** LT2, MixerLoop and Auron all fix T and never evaluate past it. E2's loss-vs-r curve, and a fix for it (XP-2/XP-5), is a question they don't answer.
3. **FLOP-matched controls.** None of the three closest competitors reports a FLOP-matched non-looped baseline. Reporting params-matched *and* FLOP-matched numbers (XP-3, XP-4) is a methodological edge, even if the answer is "loops don't pay per FLOP at this size".
4. **MixerLoop + ternary for memory-bound decoding (hypothesis).** MixerLoop's FPGA result (reported: NoLoop and 4-step MixerLoop both ≈152 tok/s on KV260) rests on reusing cached mixer weights while streaming FFN weights. Ternary FFN weights shrink the streamed bytes ~5× vs their 8-bit build (computed: 1.58 vs 8 bits). That widens the window for extra mixer passes. Nobody combines the two, but it rests entirely on unverified paper numbers and needs a decode path (C2) first.
5. **Small, honest, reproducible.** The competitors are single-run (LT2 1 seed; Auron 1 run per size with untuned LR across scales; MixerLoop 1 seed at 100M). A ledger of pre-registered tiny checks is a real difference in kind, not just in tone.

---

## Papers to add to `refs/` (for the main agent; `refs/fetch_refs.py` not edited)

| arXiv | Title | Why |
|---|---|---|
| 2608.18230 | Allocating Recurrent Compute in Looped Language Models (MixerLoop) | XP-3; note v1 and v2 differ |
| 2605.20670 | LT2: Linear-Time Looped Transformers | Closest published competitor: looped GDN/Full+GDN hybrids, SDPA gate, stability (XP-2, XP-6) |
| 2611.19446 | Architectures and KV Replay for Depth-Recurrent Language Models | Looped GDN/mLSTM hybrids need frequent attention anchors; cross-iteration KV sharing (relevant to decoding a looped model) |
| 2604.12946 | Parcae: Scaling Laws for Stable Looped Language Models | Drift-preventing decay adapter (XP-2 V3) *(overlap: looped review)* |
| 2609.30820 | Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness | XP-9 (loop-entry adapter precision) |
| 2610.13041 | RAMP: Recurrent Adaptive Mixed Precision… | Early-loop quantisation error is attenuated → a per-loop precision schedule idea |
| 2605.16343 | LoopQ: Quantization for Looped Language Models | Looped-PTQ failure sources (recursive error accumulation) |
| 2606.18114 | Ternary Mamba: Grouped QAT of W1.58A16 SSMs | Ternary on a recurrent mixer; "zero-ratio collapse"; post-hoc fixes fail through the recurrence |
| 2609.29397 | Baseline Shape Decides the Verdict: … Ternary LMs at 60K Parameters | Directly relevant to our tiny-scale ternary checks (K4): baseline shape swamps architecture effects; 3 seeds |
| 2605.09165 | Sparse Layers are Critical to Scaling Looped Language Models | Dense looped models scale worse than unique layers; Looped-MoE fixes it. Bears on Auron's wall *(overlap: looped review)* |
| 2605.22791 | Gated DeltaNet-2 | A mixer upgrade candidate (decoupled erase/write gates) |
| 2411.15242 | The Zamba2 Suite | Shared attention blocks + per-position LoRA (we have only Zamba1) |

Auron isn't on arXiv. Cite the HF blog and model cards. Its code repo is currently unavailable.
