# Gogeta-0: FusionDance architecture sketch 🕺🕺 → 💥

> LLM + SSM + looped transformers + BitNet... *fu-sion-HA!*

A scratchpad for ideas and feasibility checks. Nothing here is settled.

---

## The core idea

Each of the three techniques attacks a **different cost axis**, so in principle they stack instead of competing:

| Fighter | What it shrinks | Trade-off |
|---|---|---|
| **SSM** (Mamba / Mamba-2 style) | Cost of **sequence length**: linear time, constant-size state instead of a growing KV cache | Weaker at exact recall / copying than attention |
| **Looped transformer** (shared block applied N times) | **Parameter count**: depth comes from repetition, not new weights | More compute per token; training stability is harder |
| **BitNet b1.58** (ternary weights {-1, 0, +1}) | **Bits per parameter**: memory and bandwidth; matmuls become adds/subtracts | Usually needs training from scratch (quantization-aware), not post-hoc |

**Thesis:** a small, ternary, looped, hybrid SSM/attention block could give "big model" reasoning depth on a tiny memory footprint, and it would scale to long contexts.

The nice part is that looping costs extra compute and BitNet makes compute cheap. They cover each other's weak spots.

---

## Decisions log
- **2026-11-22:** Target is a **regular causal LLM**: loops add latent depth *per token*. Not TRM-style whole-answer refinement (maybe later as a second mode).
- **2026-11-22:** Device is **the laptop** (RTX Spark N1X, via WSL). **No hard memory target.** Memory efficiency is a design value, not a constraint, so we're exploring.
- **2026-11-22:** Build it as a **configurable sweep harness**, not one fixed model. Each axis gets toggled and measured.
- **2026-11-22:** Intermediate loops get a **cheap exit head** (norm + tied unembed), not the coda. Decoding every loop through the coda's full attention would bring back r × O(n²).
- **2026-11-22:** Naming: the **repo/project is Gogeta-0**, and the **architecture is FusionDance**. Gogeta-0 is the first FusionDance model.
- **2026-11-22 (v0.3.0):** First results: the E1 hybrid mixer stack beats dense by 3.6%. Looping (E2) does not pay yet and is non-convergent (best at the mean training r). Next comes the unlooped control E9 and per-loop supervision. See DEVLOG v0.3.0 and notes/LABNOTEBOOK.md.
- **2026-11-22 (v0.3.1):** Every finished experiment gets a site page with a plain-English explainer (now an AGENTS.md rule). Training is paused by request; E9 and the seed replications are pending and must run detached.
- **2026-11-22 (v0.3.5):** Trust but verify: every claim we rely on needs its own small local check (AGENTS.md §0, ledger in experiments/claims.md). Paper results motivate experiments but are never adopted as facts.

---

## The fusion dance metaphor (it maps surprisingly well)

- **Matching power levels.** Fusion only works if both fighters are matched. Here that means the hidden dims, norms and activation scales of the SSM and attention paths have to line up so neither one dominates the residual stream.
- **The dance must be perfectly synced.** One wrong finger and you get fat or skinny Gotenks. The interfaces matter: where the SSM and attention layers interleave, how norms sit around the looped block (Huginn-style sandwich norm?), and how ternary quantization interacts with the recurrent state.
- **30-minute time limit.** Fusion is temporary, so think adaptive loop count / halting: easy tokens exit early and hard tokens loop longer. See Mixture-of-Recursions and adaptive computation time.
- **Dance vs Potara earrings.**
  - *Dance* = fusion at runtime: merge or route between separately trained models, or toggle the number of loops at inference.
  - *Potara* = permanent fusion: one architecture trained end to end as a hybrid.
  - Potara is the realistic path. The Dance is the fun experiment.
- **Failed fusion.** Training divergence. Looped models and ternary models are both known to be finicky, so combining them doubles the chance of a skinny Gotenks.

---

## Gogeta-0: the first FusionDance model

```
tokens
  │
[Prelude]       1 unshared layer: embed → latent          (ternary linear)
  │
┌─[Core block] × r loops ────────────────────────────────────┐
│   Gated DeltaNet ×2   ← linear-time mixing, good recall     │
│                         (ablate vs Mamba-2 / Mamba-3)       │
│   Sliding-window attn ← local precision, O(n·w) not O(n²)   │
│   MLP (ternary)                                             │
│   + input re-injection each loop (Huginn-style)             │
└─────────────────────────────────────────────────────────────┘
  │   r sampled randomly in training, adaptive at inference
[Coda]          1 FULL attention layer (runs once, not r times) + MLP → logits
```

**Rule of thumb: never put a quadratic layer inside the loop.** Anything quadratic in the core block costs r × O(n²). Full attention only goes in the coda, where it runs once. (Revised 2026-11-22; v0 had Mamba-2 plus one maybe-windowed attention layer in the loop.)

Open design choices:
- Should the SSM state **carry across loops**, or reset each loop? Carrying it gives a 2D recurrence (time × depth). Interesting, but maybe cursed.
- Ternary everywhere, or keep the SSM's A/Δ/state params at higher precision? (Probably the latter: the scan is sensitive.)
- Attention every loop, or only every k loops? Zamba reuses one *shared* attention block across depth, which is already halfway to looping.

---

## Sub-quadratic parts bin (and where each fits in Gogeta-0)

Here n = sequence length. "Decode" means the cost per generated token.

### A. Recurrent / linear-time state: O(n) training, O(1) per token at decode
| Part | Idea | Gogeta-0 fit |
|---|---|---|
| **Gated DeltaNet** (ICLR 2025) / **Gated DeltaNet-2** (NVIDIA, May 2026) / **KDA** (Kimi Linear) | Linear attention with a delta-rule update: it *overwrites* memory instead of just adding to it, so recall is much better. GDN-2 splits erase and write into separate channel-wise gates. | ⭐ **Core-loop mixer, first choice.** Used in production hybrids (Qwen3-Next, Kimi Linear). |
| **Mamba-2 / Mamba-3** | Selective SSM scan | Core-loop mixer, **ablation baseline** |
| RWKV-7, GLA, RetNet, mLSTM (xLSTM) | Older members of the same family | Extra ablations only if time allows |
| **Titans / TTT layers** | Memory that updates itself with gradient steps at test time | 🃏 Wild card: "loop = more memory-update steps" is a cute match with looping. Expensive. Later. |
| Hyena / long convolutions | FFT-based, O(n log n) | Skip: weak at recall |

### B. Restricted softmax attention: fewer token pairs
| Part | Idea | Gogeta-0 fit |
|---|---|---|
| **Sliding-window attention** | Each token attends to only the last w tokens: O(n·w) | ⭐ **The attention inside the loop** |
| **DSA** (DeepSeek Sparse Attention; also in GLM-5) | A tiny learned "indexer" picks the top-k tokens for each query. The indexer is technically O(n²) but very cheap. | **Coda global layer, v1** for long context |
| HISA / MISA (2026) | Hierarchical / mixture-of-indexer versions that make the DSA indexer cheaper | Same slot as DSA, later |
| NSA, MoBA | Block-level sparse selection. Reported to lag DSA on retrieval. | Fallback options for the same slot |
| Longformer/BigBird, Reformer (LSH), Linformer/Performer/Nyströmformer | Older global-token / hashing / low-rank approximations | Mostly superseded; skip |

### C. In-between / mixed
| Part | Idea | Gogeta-0 fit |
|---|---|---|
| **Log-linear attention** (2025) | Hierarchical states, O(n log n): sits between a fixed-size RNN state and full KV | Alternative coda layer if SWA + GDN can't recall well enough |
| **NAtS-L** (ICML 2026) | Each *token* picks linear or softmax attention within the same layer | 🃏 Fits the adaptive-compute theme; per-token loop count plus per-token attention type. Later. |

### Not actually sub-quadratic (but often mistaken for it)
- **FlashAttention:** linear *memory*, still quadratic *compute*. We'll still use it for the SWA and coda layers.
- **Ring attention:** spreads the quadratic cost across GPUs. **MLA / KV compression:** shrinks the cache, not the compute.

### Notes for the fusion
- **Ternary:** the q/k/v/out projections of GDN, Mamba and SWA can go ternary. Gates, decays and the recurrent state stay in higher precision.
- **Speculative decoding:** rolling back SWA is trivial (truncate the cache). GDN/Mamba state needs per-position checkpoints (see the inference section).
- **Kernels:** `flash-linear-attention` (fla) has Triton kernels for GDN, Mamba-2, GLA, RWKV and more. **Check that it runs on aarch64 + Blackwell**; if not, fall back to pure-PyTorch chunked forms.
- **Smoke-test honesty:** at 512–1k tokens, quadratic attention costs almost nothing. These choices show their value only in long-context tests: add an **8k–32k needle-in-a-haystack / RULER-style eval** to the ladder.

---

## Prior art to read / check

- **Hybrids:** Jamba (AI21), Zamba / Zamba2 (Zyphra, shared attention block), Samba (Microsoft), Mamba-2 (SSD).
- **Looped / recurrent depth:** Universal Transformers; "Looped Transformers as Programmable Computers"; Geiping et al. 2025 "recurrent depth" (Huginn, 3.5B); Mixture-of-Recursions (2025).
- **BitNet:** BitNet (1-bit), BitNet b1.58, BitNet b1.58 2B4T (open weights), bitnet.cpp.
- **Crossovers:** Bi-Mamba (binarized Mamba). Is anyone doing ternary + looped? → **search this first; it's the novelty check.**
- **⚠️ Closest prior art found so far:** "Tiny Recursive Reasoning with Mamba-2 Attention Hybrid" (arXiv 2602.12078), a TRM-style recursion with a Mamba-2/attention hybrid. That's 2 of our 3 fighters, so **read it before building anything.** Also: TRM, "Less is More: Recursive Reasoning with Tiny Networks" (arXiv 2510.04871); HRM; Probabilistic TRM (arXiv 2605.19943).

---

## Feasibility

### What's plausible at hobby scale
- Train **~20–150M param** models from scratch on TinyStories, a FineWeb-Edu subset, or synthetic reasoning tasks (arithmetic, copying, graph search). Looped models shine on algorithmic tasks, so they're a good place for quick, cheap signal.
- Ternary QAT is just a custom `Linear` with straight-through estimator (STE) and absmean quantization. Easy to implement in PyTorch.
- Mamba-2 has reference kernels (`mamba_ssm`). A pure-PyTorch fallback is slow but fine for small runs.

### Hard parts / risks
- **Stability squared.** Looping and ternary weights are both unstable on their own. Expect a lot of LR / norm / init tuning.
- **No speedup without kernels.** Ternary weights only pay off with custom kernels. In plain PyTorch they're fake-quantized fp16, so they'll train *slower*. Fine for research, but don't expect wall-clock wins.
- **Backprop through loops.** Memory grows with r. Use truncated backprop through the last k loops, like Huginn.
- **The SSM scan in low precision.** Likely needs fp16/fp32 state, so only the projections go ternary.
- **Evaluation.** Small models score at the noise floor on standard benchmarks. Use synthetic tasks plus perplexity-per-bit / per-FLOP comparisons.

### Local setup (checked 2026-11-22)
- **GPU:** RTX Spark N1X (Blackwell, 6144 cores). nvidia-smi reports ~96 GB, so **memory is not the constraint; compute and bandwidth are.**
- **Native Windows (ARM64):** only CPU builds of PyTorch exist, so it's no good for training.
- **WSL Ubuntu 24.04 (aarch64):** sees the GPU ✅, and CUDA PyTorch wheels exist for Linux aarch64 (cu128/cu130) ✅. **Train here.** It still needs pip/venv installed, and WSL only has ~14 GB of system RAM (can be raised in `.wslconfig`).
- Start with a **pure-PyTorch Mamba-2** (SSD chunked scan). The official `mamba_ssm` CUDA kernels on aarch64 + Blackwell may not compile, so try them later.

**Size tiers** (rough guesses; measure real throughput first):
| Tier | Unique params | Data | Wall-clock |
|---|---|---|---|
| Smoke test | ~10M (acts like ~25M at 4 loops) | TinyStories (~0.5B tok) + synthetic tasks | ~1–3 h |
| Real test | ~50M | FineWeb-Edu ~1B tok | ~overnight to ~1 day |
| Ceiling-ish | ~150M+ | 3B+ tok | days to weeks: compute-bound |

Keep the **vocab small** (~8k BPE). At ~10M params, a GPT-2 50k vocab would make the embedding bigger than the rest of the model.

### Hardware tie-in? 🤔
Ternary math is adds and subtracts only, which suits FPGAs well. Possible crossover with the RISC-V FPGA research later.

---

## Inference side: speculative decoding (DFlash / DSpark)

- **DFlash** (ICML 2026): a small *block-diffusion* drafter proposes a whole block of tokens in parallel, conditioned on the target model's hidden features. The target verifies the block in one pass. The authors report up to ~6× lossless speedup.
- **DSpark** (2026): confidence-scheduled speculative decoding with semi-autoregressive drafting.
- **Why it helps in general:** generating text is limited by memory bandwidth, not compute. Verifying k drafted tokens costs about the same as generating 1, so speculative decoding turns idle compute into throughput.

**How it fits with FusionDance:**
- 🟡 **Overlap.** Ternary weights mean ~10× fewer bytes per token, and looping reuses the same weights r times. Both already make generation less bandwidth-bound, so speculative decoding has less to win. At ~10M params the whole model fits in GPU cache anyway, so it's pointless for the smoke test. It only starts to matter at much larger scale.
- 🔴 **SSM rollback.** If drafted tokens get rejected, attention just truncates its cache. The SSM's recurrent state has to be checkpointed at each drafted position, or recomputed. Looping multiplies that: one state per loop per position.
- 🟢 **Free drafter.** A looped model can draft for itself. Run 1–2 loops to draft and the full r loops to verify, with the same weights and no extra model (self-speculation, like LayerSkip-style early exit). Training with a random loop count (already planned) should make low-loop drafts decent for free.
- 🟢 **DFlash-style head.** Attach a small block-diffusion drafter to early-loop features instead of the final layer. This needs our own drafter training, and DFlash's training recipe wasn't public at last check.

**Plan:** don't build it yet; design for it now.
1. Make the SSM state checkpointable per position.
2. Keep random-loop training.
3. Once the model trains, measure **draft acceptance rate for r=1/2 vs full r**. That number decides whether self-speculation is worth building.

---

## The Rival Loop: critique → fix inside the recurrence

Goku and Vegeta are rivals, so build the rivalry in. In the agent version, a generator drafts, a critic finds flaws, a fixer revises, and this repeats until the critic is happy. The latent version: each loop iteration is a refinement step, so make some loops *critique* and others *fix*.

**The catch.** Agent critique loops work because the critic gets *outside* information: tests, compiler errors, a fresh context. A critic that shares the same weights and the same latent has no new information, and can learn to rubber-stamp. (LLMs mostly fail at self-correction without external feedback; Huang et al. 2023.) So the critic needs a **training signal grounded in the truth**, and the value at inference is mostly in **knowing when to stop** and **keeping each fix actually better**.

### Options, cheapest/most stable first
1. **Deep supervision + "fixes must fix" penalty.** Decode every loop through the coda and compute the loss at each loop. Add a penalty when loop k+1 is *worse* than loop k. This forces monotonic refinement. Almost free.
2. **Critic head.** A tiny head reads the latent at loop k and predicts *how wrong the current prediction is* (the per-token loss, or "will the next loop change the answer?"). It's trained against the real loss, so it's grounded. It gets used three ways:
   - **Halting:** stop looping when the critic says "good enough". This is the 30-minute fusion timer, learned (PonderNet / ACT style).
   - **Feedback:** inject the critic's error vector into the next loop's input, so the fixer gets told *what* is wrong, not just to try again.
   - **Speculative decoding:** the critic's confidence also predicts whether a low-loop draft will be accepted. One head with three jobs.
3. **Role-alternating loops.** Same shared weights, but add a learned role embedding (propose / critique) on alternate loops. Optionally add tiny role-specific adapters: two fighters sharing one body. Compare against a plain loop at the **same total r**: is an explicit critic step worth giving up a refinement step?
4. **Damage-and-repair training.** Corrupt the latent between loops and train later loops to recover. Start with random noise, then use *adversarial* noise: a PGD/FGSM-style worst-case perturbation of the hidden state. This is the genuinely adversarial version, and it makes each loop a learned denoiser, close to diffusion, which links up with the DFlash drafter idea. It costs about 1.5–2× training compute for the extra backward pass.
5. ~~GAN-style discriminator on outputs.~~ Discrete-text GANs are notoriously unstable. Ternary + looping + GAN would be three instabilities stacked: a guaranteed skinny Gotenks. Skip.

### What to measure
- Does loss go down **monotonically** as r increases (1 → 2 → … → 8), and keep improving past the training r?
- With critic halting: **average loops used at equal loss** compared with fixed r.
- On synthetic tasks with injected errors (e.g. arithmetic with a corrupted carry): does the loop *repair* them?
- How well critic confidence predicts self-speculation acceptance.

### Outside-the-model version (for later)
For code specifically, the real critic is a test runner, which is the agent loop itself. A tiny model won't write code, but a FusionDance-style model at scale could use **adaptive loop depth as the inner critic** and **tests as the outer critic**.

---

## Training data & evals

### Data, by experiment stage
| Stage | Data | Size / notes |
|---|---|---|
| **Smoke test (~10M)** | **TinyStories** (`roneneldan/TinyStories`) | ~0.5B tokens, simple English. Tiny models produce coherent text on it. |
| **Loop / recall probes** | **Generated on the fly** (no download): multi-digit arithmetic, copying, selective copy, induction, parity, graph reachability, **MQAR** (multi-query associative recall) | Infinite and controllable. Train short, test long for length generalization. Corrupted-carry arithmetic for the Rival Loop. |
| | **Reasoning Gym** | ~100 procedurally generated reasoning tasks with verifiers |
| **Puzzles (looping showcase)** | **Sudoku-Extreme, Maze-Hard, ARC-AGI-1/2** (the HRM / TRM datasets) | Small training sets plus heavy augmentation. **Published tiny baselines to beat:** TRM (7M) gets 87.4% on Sudoku-Extreme, 85.3% on Maze-Hard, 45% on ARC-AGI-1. |
| **Real text (~50M)** | **FineWeb-Edu** `sample-10BT` (stream ~1B tokens) | Default choice. Educational web text hits above its weight for small models. |
| | **SmolLM-Corpus** (Cosmopedia v2 synthetic textbooks + Python-Edu + FineWeb-Edu dedup) | A ready-made mix that includes code |
| | **FineMath** | For a math-leaning mix |
| | Alternatives: DCLM-baseline, Zyda-2, **Common Corpus** (~2T, fully open licenses), **MixtureVitae** (422B, low copyright risk), **FinePhrase** (486B synthetic rephrased web) | Swap in if licensing matters or for data ablations |
| **Sample-efficiency angle** | **BabyLM** (10M / 100M word tracks) | Tests whether looping helps when data is scarce, which is the actual claim of looping |

Tokenizer: train our own **~8k BPE** on whatever mix we use.

### Evals
| What it tests | Eval | Notes |
|---|---|---|
| Language modeling | Held-out loss / perplexity, LAMBADA | The primary metric at small scale |
| Grammar | **BLiMP** (minimal pairs) | Gives real signal even for tiny models |
| Story quality | LLM-as-judge on TinyStories completions | Same idea as the original TinyStories paper |
| Zero-shot commonsense | lm-evaluation-harness: PIQA, ARC-Easy, HellaSwag, SciQ, WinoGrande | ⚠️ Near chance at ≤50M. Compare **differences between variants**, not absolute scores. |
| **Recall (linear vs attention)** | MQAR; SWDE / FDA / SQuAD-completion | The standard recall suite from the Based / Zoology / Gated DeltaNet papers. This is where GDN vs Mamba vs SWA shows up. |
| **Long context** | RULER, BABILong, passkey / needle-in-a-haystack at 8k–32k | Tests the sub-quadratic design |
| **Looping / Rival Loop** | Accuracy vs r; length generalization; error-repair rate; Sudoku/Maze | Should look like a curve that keeps improving as loops increase |
| **Efficiency** | Train/decode tokens/sec, peak memory, **loss vs FLOPs and vs weight bits**, average loops used, speculative acceptance rate | The FusionDance headline metric |

**Skip for now:** HumanEval, MBPP, GSM8K, MMLU. These sit at chance level below a few hundred million parameters.

---

## Experiment ladder (each step is one fusion at a time)

0. **Baseline:** tiny dense transformer at a fixed param budget.
1. **+SSM:** hybrid Mamba/attention, same params. Does long context get cheaper without losing recall?
2. **+Loop:** share the core block, loop r times. Does a smaller param count match the baseline on algorithmic tasks?
3. **+Ternary:** BitNet the linears. How much quality do we lose? Plot memory vs loss.
4. **Full fusion:** all three. Compare loss vs params, vs bits, and vs FLOPs.
5. **Bonus, the Dance:** adaptive loop count at inference. Does "thinking longer" help on harder inputs?
6. **Rival Loop:** deep supervision + critic head (halting/feedback), then role-alternating loops, then adversarial damage-and-repair. Each is compared at equal compute.

Metric to beat: **quality per bit of weight memory** at matched compute.

---

## Questions for me

- What GPU / compute budget do I have? (Sets the max model size.)
- Goal: a paper-style ablation, a toy that runs on a laptop/FPGA, or just learning?
- Dance (runtime merge) or Potara (train-from-scratch hybrid) first?

## Scratch / random ideas

-
