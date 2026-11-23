# Byte-level (tokenizer-free) language models, and whether FusionDance should be one

Research side quest, 2026-11-22 (subagent; no GPU, no installs, no downloads, no git). The only code that ran is two
tiny CPU scripts in `experiments/byte_level/` (`seq_stats.py`, `flop_arith.py`). Labels follow AGENTS.md §0:
**measured** (our CPU check, output on disk), **computed** (arithmetic from repo code, shown), **reported by <paper>**
(quoted, not adopted), **predicted**, **speculation**.

How the paper numbers were obtained: the abstract pages of all papers, plus the full HTML of MegaByte, MambaByte,
SpaceByte, BLT and H-Net (read through a fetch tool that summarises the page; I did not open the PDFs, so every
number below should be re-checked against the PDF before it enters the claims ledger). Items marked (abs) are from the
abstract only. No paper's number is a claim about our code.

## 1. Local measurement (CPU, 2 min): the sequence-length blow-up

`experiments/byte_level/seq_stats.py` on our 4,000 val docs with the shipped 8k BPE (`seq_stats.json`, **measured**):

| Quantity | Value |
|---|---|
| chars / bytes / tokens | 3,938,783 / 3,940,661 / 1,055,160 |
| **bytes per token** | **3.735** (chars per token 3.733; bytes per char 1.0005) |
| non-ASCII | 69 of 4,000 docs; 1,878 extra bytes (0.05%) |
| tokens per doc, p50 / p90 / p99 | 221 / 459 / 762 |
| bytes per doc, p50 / p90 / p99 | 749 / 1,854 / 3,078 |
| docs > 1,024 tokens / > 3,584 bytes / > 4,096 bytes | 0.23% / 0.38% / 0.18% |

Consequences: (a) the blow-up is **3.73x**, not 3.5x (K8's figure was chars/token; same thing here). (b) bits/char
equals bits/byte to 0.05% on this corpus (Python is almost pure ASCII), so K8/K9's bits/char is directly a
bits/byte and byte models are scored on exactly the same denominator: no tokenizer-induced scoring difference.
(c) T = 4,096 bytes covers 99.8% of docs whole; 1,024 BPE tokens is about 3,824 bytes. (d) Training windows are
packed random windows (C101), so docs are not the unit; the ratio on the training split was not measured (assume 3.73).

## 2. Survey

"Matched" means what the paper says. All bits are per byte (bpb) unless noted. Our scale is 14M params, ~0.75 GB of text.

| Model (arXiv) | Idea | Scale reported | Compute vs token model | Quality vs BPE (as reported) | Robustness | Decode speed |
|---|---|---|---|---|---|---|
| **ByT5** (2105.13626) | T5 on raw bytes, unchanged otherwise | T5-size family (sizes not in abstract) | abstract: "characterize trade-offs" in params, training FLOPs, inference speed; byte models are slower per text (full-length sequences) | competitive with mT5 at same size (abs) | "significantly more robust to noise", better on spelling/pronunciation tasks (abs) | slower than token (longer sequences) |
| **CANINE** (2103.06874) | char-level encoder: downsample, deep stack, upsample | ~28% fewer params than mBERT (abs) | downsampling cuts the deep stack's length | +2.8 F1 over comparable mBERT on TyDi QA (abs) | no vocabulary | encoder only |
| **Charformer** (2106.12672) | GBST: soft learned subword blocks on bytes | (abs) | faster than vanilla byte and subword Transformers by 28-100% (abs) | competitive (abs) | byte input | n/a |
| **MegaByte** (2305.07185) | fixed patches P=8; global Transformer over patches, local Transformer inside | 758M global+262M local, 80B bytes, T=8,192 (controlled); 1.3B+218M/350M on 400B bytes (PG-19) | matched-compute table; attention O(T^2/P^2+TP); fwd FLOPs ~2T(m_g/P+m_l) | bpb PG-19 1.000, Stories 0.978, Books 1.007, arXiv 0.678, Code 0.411 vs byte Transformer 1.057/1.064/1.097/0.816/0.575 | not studied | 93 s vs 132 s for the byte Transformer, 8,192 bytes (1.3B+218M vs 350M); init std 0.006 truncated reported to help stability |
| **MambaByte** (2401.13660) | Mamba SSM on bytes, T=8,192 | 353M (30B bytes), 972M (150B), 1.6B | FLOPs/byte 1.02x MegaByte-758M+262M; 0.54x MegaByte-1.3B+350M. "Transformer loss in under 1/3 of compute" (Fig 1) | bpb (353M, 30B bytes): PG19 0.930, Stories 0.908, Books 0.966, arXiv 0.663, **Code 0.396**; MegaByte 1.000/0.978/1.007/0.678/0.411; byte Transformer 1.057/.../0.575. PG-19 word-PPL 33.0 vs subword Mamba-1.03B 33.9 (equal-ish, at 150B bytes). **Subword Mamba trains ~2.2x faster to near-optimal** | word-PPL increase under noise: swap 0.3: +28.7 vs +630.6 (subword Mamba); drop 0.3: +31.7 vs +213.2; antspeak +28 vs +58,300 | 8,192 bytes: 29 s (972M) vs 93 s MegaByte vs 132 s Transformer. Speculative decoding (small subword drafter, byte verifier): 2.6x |
| **SpaceByte** (2404.14408) | byte Transformer, big "global" blocks run only after space-like bytes (~6 B/patch text, ~8 code) | 793M+184M, 30B bytes; sweeps at 1e19 FLOPs | matched training FLOPs; FLOP/byte inference proxy | bpb at 1e19 FLOPs PG-19 / arXiv / GitHub: SpaceByte 1.009/0.748/0.500; subword (SentencePiece) 0.989/0.768/0.508; subword (GPT-2) 1.013/0.796/0.554; byte Transformer 1.138/0.909/0.655; MegaByte 1.083/0.822/0.570; fixed-patch SpaceByte 1.112/0.804/0.552 (**the fixed-patch ablation loses ~6-12% to the dynamic rule**) | not studied | irregular patches make batched sampling harder (paper's limitation); worse than subword on Chinese |
| **BLT** (2412.09871) | tiny local encoder, entropy-patched latent Transformer, local decoder, cross-attention, hash n-gram embeddings (n=3-8) | up to 8B params, 4T bytes; first FLOP-controlled byte scaling study | entropy patcher (100M, 14 layers); patch 4.5 B avg (= Llama-3 tokens 4.4), 6 and 8 in the scaling study; inference FLOPs inversely proportional to patch size, "up to ~50%" saving at 8 | 8B, avg of 7 tasks: BLT-entropy 61.1 vs Llama 3 60.0; **HumanEval 35.4 vs 31.1, MBPP 41.8 vs 40.2**; MMLU 57.4 vs 58.1 (BLT-space, 6.1 B/patch, is 58.0, worse) | noised HellaSwag 64.3 vs 56.9 (Llama 3); CUTE 54.1 vs 27.5 | fewer latent steps at larger patches; no wall-clock table found |
| **bGPT** (2402.19155) | next-byte prediction over any file type | (abs) no sizes | n/a | ABC-to-MIDI 0.0011 bpb error; >99.99% CPU-state simulation (abs) | n/a | n/a |
| **EvaByte** (HKU blog, 2025; no arXiv id found) | 6.5B byte LM, EVA linearised attention, 8 multibyte-prediction heads, vocab 320 | 1.5T bytes (Dolma, Stack v2, FineWeb-Edu, DCLM) | ~5x less data than tokenizer LMs (blog's claim) | "rivals" open tokenizer LMs; HumanEval/MBPP strong; beats BLT with 3-4x fewer bytes (blog's claim) | tokenizer-free | up to 2x faster than tokenizer LMs, 5-10x faster than vanilla byte models (blog's claim; self-speculative multibyte heads) |
| **H-Net** (2507.07955) | learned dynamic chunking (cosine-similarity router, EMA smoothing, STE, ratio loss alpha=0.03) inside a recursive U-Net; Mamba-2 encoder/decoder (4 layers), Transformer main net | 680M-1.6B, 100B FineWeb-Edu tokens, 8,192-byte sequences | matched compute/data to BPE Transformer; 1-stage averages ~4.8 B/chunk (N=6 target) | bpb Large: Transformer 0.756, H-Net 1-stage 0.755, 2-stage 0.743; XL 0.730 / 0.728 / 0.715. **GitHub (XL): Transformer 0.3376 vs H-Net 0.3161**. 1-stage crosses over after ~100B bytes, 2-stage after ~30B (**the byte model loses early in training**). DNA: 3.6x data efficiency | robustness score on perturbed HellaSwag 20.2 -> 39.0 (L), 22.2 -> 42.8 (XL) | no throughput reported |
| **AU-Net** (2506.14761) | autoregressive U-Net: bytes pooled to words, word pairs, 4 words | (abs) sizes not given | controlled compute | shallow hierarchies tie strong BPE baselines; deeper = "promising trend" (abs) | character-level tasks, low-resource transfer (abs) | n/a |
| **T-Free HAT** (2603.15953, Aleph Alpha) | hierarchical AR transformer: byte encoder -> word-level backbone -> byte decoder; reuses Llama-3.1 backbones minus embedding/head | 7B from scratch (~4T words), 8B and 70B retrofits | n/a (abs) | better compression and spelling robustness; English/German often above Llama 3.1 (abs) | yes | n/a |
| **Proxy Compression** (2602.04289, ICML 2026) | train one LM jointly on raw bytes and compressed views (e.g. BPE/other compressors); inference on raw bytes only; **code LM experiments** | not in abstract | fixed compute | beats pure-byte baselines at fixed compute; at larger scale matches or exceeds tokenizer models (abs) | n/a | bytes at inference |
| **Efficiency Gap in Byte Modeling** (2605.12928) | counter-evidence: byte modelling carries a compute-matched scaling overhead; worse for masked diffusion than for AR | not in abstract | compute-matched | byte penalty not uniform in scale; "context fragility" hypothesis (abs) | n/a | n/a |
| **Byte-wise sampling from a BPE LM** (2506.14123) | not a byte model: converts a BPE LM's distribution to bytes; motivated by token-boundary misalignment in code | n/a | n/a | n/a | n/a | n/a |

Honest read of the table:
1. **At matched FLOPs, a plain byte Transformer is clearly worse than BPE** (SpaceByte 1.138 vs 0.989 on PG-19, 0.655 vs 0.508 on GitHub; reported by SpaceByte). Every winning byte model gets there by *not* spending full-width compute on every byte (patching, SSM, hierarchy).
2. **Dynamic patching beats fixed patching** by a few percent to ~10% (SpaceByte table; BLT-entropy 61.1 vs BLT-space 58.0 mixes patch size and rule, so not clean).
3. **Everything ties BPE only at hundreds of millions to billions of params and 30B-4T bytes.** H-Net says the byte model is behind early in training and crosses over after ~30-100B bytes. At our ~0.75B bytes and 14M params, we are **far below every regime tested**. Our scale is the least favourable place for tokenizer-free models: capacity spent learning spelling is a fixed cost that dilutes most when the model is tiny. (Speculation, but consistent with the byte-modelling overhead in 2605.12928 (abs) and the crossover in H-Net.)
4. **Code looks like the most favourable domain**: MambaByte's Code column (0.396 vs MegaByte 0.411 vs byte Transformer 0.575), H-Net GitHub (0.3161 vs 0.3376), BLT HumanEval/MBPP. Reasons offered by the papers: identifiers, indentation and punctuation are poorly served by BPE merges. No paper tests a model below ~300M.
5. **Robustness gains are large and consistent** (MambaByte, BLT, H-Net). For a code model this matters for typos and rare identifiers, but our K9 shows our val set is 41.7% near-dups of train, so a robustness advantage will not show in clean-val bits/char; it needs its own perturbed eval.
6. **Decode speed**: pure byte models decode 3.7x more steps; MambaByte's recurrent state makes each step cheap (29 s vs 132 s Transformer at 8,192 bytes, reported), patch models amortise the big model over P bytes (MegaByte 93 s), speculative or multibyte heads recover more (MambaByte 2.6x, EvaByte up to 2x vs tokenizer LMs, both reported).

## 3. Why it might suit FusionDance (speculation, labelled)

1. **Linear-time state for 3.7x longer sequences.** *Speculation.* GDN is O(n) with a fixed state (C15 VERIFIED at kernel level: time exponent 0.66, constant state) so a 4k-byte context costs the GDN layers 4x the positions, not 16x. MambaByte's result (reported) is the closest precedent; GDN is the delta-rule cousin of Mamba. **But in our code SWA is a dense-masked SDPA, so its kernel is O(n^2) (C12 WRONG FOR OUR CODE, time exponent 1.744, memory 1.953)**: at T=4,096 the SWA layers are the problem, not GDN. Prerequisite: the flex-attention SWA patch (P3, `notes/patches`), which is not applied.
2. **Removing the embedding.** Computed: E1 has 14,237,796 params, of which 3,145,728 (22.1%) are the tied 8192x384 embedding. A 256x384 embedding is 98,304, so the freed 3.05M can go to depth/width/loops (e.g. +27% non-embedding parameters at the same total, or +2 GDN blocks). **Caveat:** the freed parameters were not the cost that matters for compute: at E1's 91 MFLOP/token, the head is 2·384·8192 = 6.3M FLOPs fwd of 30.3M (21% of E1's train FLOPs, computed), so dropping it saves real compute per position; but the byte model's cost is the 3.7x positions.
3. **Prelude / looped core / coda maps onto local encoder / latent core / local decoder.** *Speculation, the strongest structural fit.* BLT and H-Net have the same three-part shape, and FusionDance already re-injects the prelude output `e` into every loop. A byte prelude (small, byte resolution) -> P-byte patches -> looped core at patch resolution -> byte coda would make the loop cost **per patch** (1/P of the per-byte cost): looping stays affordable even though bytes are 3.7x more numerous. This is the one idea that *reduces* FusionDance's costs, rather than adding them.
4. **Ternary with a 256-entry embedding.** The byte embedding is 98k params, negligible and full precision anyway (BitLinear leaves embeddings alone, `model.py` docstring). The downside: the local byte layers are small (d_l ~192). K4 (reported in the INDEX, local) found the ternary gap is larger at smaller width (15.3% at d=64, 13.8% d=128, 10.5% d=256), so I would keep local layers in bf16 and ternarise only the latent core. Then the shipped model size is dominated by the local layers (1.83M params = 3.65 MB bf16 vs ~2.8 MB ternary body, computed from C9's figure), which weakens the "tiny model" story.
5. **Tokenizer-free bits/char gives a clean comparison.** Measured above: bytes/char = 1.0005, so K8's scorer's denominator is identical for a byte model; no tokenizer-specific normalisation is needed. A byte model can also be scored under the same doc-level protocol (K8/K9 scorer) once `score_checkpoint` is told vocab=256.
6. Also relevant to code: byte models need no tokenizer file, and indentation/whitespace runs are modelled exactly. And for an RL stage, per-byte rollouts are 3.7x longer (cost, not benefit).

## 4. Costs and risks at our scale

- **Context.** 1,024 BPE tokens = ~3,824 bytes, so the byte E1 needs T=4,096 (99.8% of val docs fit whole). Attention windows must scale with text span: window 256 BPE tokens is ~955 bytes, so window 256 *bytes* (the unchanged config) is a 3.7x shorter text span. I therefore compute both (window 256 and 1,024).
- **Compute (computed, `flop_arith.out`).** Byte-E1 at T=4,096, window 1,024: 86.9 MFLOP/byte train = 325 MFLOP per BPE-token-equivalent = **3.57x E1's 91 MFLOP/token** (3.27x with window 256). The head shrinks (-6.2 MFLOP/position fwd), but the full-attention layer now has mean context 2,048 (+2.4 MFLOP/byte fwd) and the SWA layers 4x wider. Same text as E1 costs 6.49e16 FLOPs vs 1.82e16.
- **Throughput on our bandwidth-bound GPU.** `notes/hardware-spark.md` (measured/computed): E1 ~100k tok/s at B=32,T=1024; the step is bandwidth/elementwise bound, ~28 KB per token per layer. Bandwidth scales with **positions x layers**, not FLOPs, so the byte model costs ~3.7x per unit of text, except the logits traffic: 8192-wide logits are ~131 KB/position of the ~590 KB/position of E0 (computed, E0; smaller share in E1), a 256-wide head removes most of it. Prediction: **85-115k bytes/s** (about 100k), i.e. ~3.7x slower per text. Matched text: 746.9M bytes / 100k = **~125 min** (108-147) vs E1's ~33 min.
- **Memory.** Positions per step at matched text = 122k vs 32.8k: activations ~3.7x. The 96 GiB carve-out should hold it but is **unmeasured**; fallback: B=15 with 2 accumulation steps (changes nothing in the optimisation).
- **SWA kernel.** Dense bool-mask SDPA at T=4,096, B=30, 6 heads: score tensor 30·6·4096² bf16 = 6.0 GB per SWA layer per direction, plus the mask path forbids the flash kernel. **Do not run until P3 (flex SWA) or an equivalent exists.** Full attention is flash-eligible (SDPA flash works for causal).
- **GDN chunk kernels and dynamic patching.** fla's `chunk_gated_delta_rule` works on dense [B,T,H,K] with a fixed internal chunk (64 in our install per fla's chunk kernels; I did not re-read the source this session, **verify**); T=4,096 is a multiple of 64. With *fixed* patches the core sequence is a constant 1,024 patches, so nothing changes, including loop-count handling (same shape every iteration). With *dynamic* patches (BLT, H-Net) the number of patches differs per row: either pad to the max (wasted compute, and the GDN state must not see pad tokens; padded positions after the last real patch are harmless for a causal model but cost) or use fla's variable-length path (`cu_seqlens`, believed supported, **unverified here**). Looping adds one more issue: the same patch boundaries must be used for every iteration (they are, since segmentation happens before the core), but "compute-adaptive" patching changes the FLOPs/step, which breaks matched-compute accounting. Dynamic patching also needs either a separate entropy model (BLT: 100M params, 7x our whole model) or a learned router with a ratio loss (H-Net), i.e. more moving parts than fit one-ingredient-at-a-time (AGENTS.md §5). **Recommendation: fixed patches first.**
- **Training stability, from the papers.** MegaByte: truncated-normal init std 0.006 (our init: 0.02, residual-scaled), grad-clip 1.0, 500 warm-up. H-Net: RMSNorm at the end of each component, near-zero init of the residual projection, per-stage LR scaling, ratio loss alpha 0.03, and "the byte model is behind early, crosses over after 30-100B bytes". BLT: no instability reported; context reset at newlines for the patcher. None tested at <300M; our lr 3e-3/6e-3 regime (K7) may not transfer, so each byte arm needs its own K7-style LR check (cheap at 20M tokens).
- **Contamination.** C131: our corpus self-duplicates heavily. A byte model has a longer window of exact context and could memorise near-dup docs *differently* from a BPE model (e.g. the 41.7% near-dup val docs score 0.101 bits/char for E0). All comparisons must be on the K9-clean val subset, otherwise the winner is whichever memorises better.
- **Tying and the head.** Byte head is 192/384 x 256; tying with a 256 embedding is fine; label smoothing not needed.
- **Honest prior.** At 14M params and ~0.75 GB of bytes, my prior is that plain byte-E1 loses to BPE-E1 at matched FLOPs by a double-digit percentage, ties or loses slightly at matched text, and the interesting arm is the patched one.

## 5. FusionDance-Byte sketch (computed; `flop_arith.py`)

**Variant A: Byte-E1 (control).** Same E1 body (SWA, GDN, GDN, SWA, GDN, attn), d=384, vocab 256, tied. Params: 11,091,684 body + 98,304 emb + 384 = **11,190,372** (E1: 14,237,796, so -21.4%). T=4,096, window 1,024 (the full-attn layer unchanged).

**Variant B: BLT-lite, fixed P=4 (MegaByte-style cut, GDN/SWA blocks).**

```
bytes (4096) -> embed(256 x 192, tied with the byte head)
 -> local encoder: 1 GDN block, d_l=192, 3 heads, mlp 512       (byte resolution)
 -> patch-in: concat 4 byte vectors (768) -> Linear -> 384        (1024 patches)
 -> FusionDance body at d=384 on 1024 patches: E1's 6 blocks, or prelude + core x r + coda (E2)
 -> up: Linear 384 -> 192 on the patch latent of the *previous* patch, broadcast to its 4 bytes' successors
 -> local decoder on bytes: input = embed(previous byte) + up(h_{patch-1}); GDN block + SWA(window 32) block, d_l=192
 -> RMSNorm -> head (tied embed, 192 -> 256)
```
Causality: patch j is encoded from bytes 4j..4j+3 and used only to predict bytes 4(j+1)..4(j+1)+3, as in MegaByte; so the first byte of each patch is predicted from the latent alone (a known weakness of fixed patches, SpaceByte's fixed-patch ablation).

Parameters (computed):

| Part | Params |
|---|---|
| byte embedding 256x192 (tied head) | 49,152 |
| encoder GDN block, d=192 (5d^2=184,320; a,b 1,152; conv 2,304; gates/norm ~0.5k; SwiGLU 3·512·192=294,912) | 483,142 |
| patch-in 768x384 | 294,912 |
| up 384x192 | 73,728 |
| decoder GDN block + attn block, d=192 | 925,894 |
| local total | **1,826,828** |
| body (E1 stack) | 11,091,684 |
| **total** | **12,918,896** (-9.3% vs E1; the 3.05M embedding is replaced by 1.83M of byte layers) |

FLOPs per byte, train (computed): core: (2·11.09M matmul + 1.87M attention/GDN terms) / 4 = 6.01M fwd; local: 3.22M fwd (enc 0.97M, dec 1.85M, head 0.10M, patch-in/up amortised 0.18M, attn/GDN terms 0.12M); total 9.23M fwd; **train 27.7 MFLOP/byte = 103M per token-equivalent = 1.14x E1**. So at matched text, BLT-lite is **nearly FLOP-matched to BPE E1 (+14%)** where Byte-E1 is 3.57x.

Bandwidth proxy (the bound that matters here): per patch, core 6 layers at d=384 on 1 position, plus local 3 layers at d=192 on 4 byte-positions = 3·4·(192/384) = 6 layer-equivalents: **12 vs E1's 6 layer-equivalents per patch, i.e. ~2x E1's time**. Predicted **~45-55k BPE-token-equivalents/s** (E1 ~100k): a matched-text run of 200M tok-equiv takes ~60-75 min (computed from 2x of E1's ~33 min, then the range). Since the local layers are small and elementwise-bound, the 2x could be better with fused kernels, or worse with 4k-wide byte tensors.

**Variant C (later):** the E2 looped core on patches: loops cost 1/4 per byte versus the byte-E2, so FusionDance's weakest cost (C6: 2x FLOPs for no gain) is diluted by P. Only worth it if B works. Variant D: dynamic patching (SpaceByte rule: after non-alphanumeric bytes; for Python that is about identifier/whitespace boundaries) as a rule-based, parameter-free upgrade of B, if B beats A.

## 6. The cheapest decisive experiments (designs only; not run)

Common: corpus code-py-v1 split as in E1; same optimiser/LR protocol; seed 0 first, K7-style LR check at 20M tokens (about 5 min per LR for BLT-lite, per the 2x time estimate), scored in **clean-val bits/char (K9 clean subset) with the K8 doc-level scorer**, and also reported on full val for continuity. All arms get a matching BPE-E1 re-score on the clean subset (not the contaminated 0.2126).

### BX1: byte E1 vs BPE E1 (SQ180)
- **Question:** does tokenizer removal cost quality at our scale, and at which compute match?
- **Arms:** (i) BPE-E1 (existing config, lr tuned 0.006 per K7); (ii) Byte-E1 matched **FLOPs** (209.6M bytes = 56.2M token-eq = 28% of the text); (iii) Byte-E1 matched **bytes seen** (746.9M bytes = same text and same 6,104 steps at B=30 x T=4,096 = 122,880 bytes/step, computed from 200M tokens x 3.7347).
- **Prediction (before running):** clean-val bits/char ratio byte/BPE: matched FLOPs **1.10-1.35** (byte loses clearly; SpaceByte reports a byte Transformer 15% worse on PG-19 and ~30% worse on GitHub at matched FLOPs, and we are smaller); matched bytes **0.97-1.10** (a tie or a small loss). A byte win at matched FLOPs would be a surprise.
- **Criteria:** report the ratio with 3 seeds if (ii) or (iii) is within +/-3%. "Byte matches BPE" := (iii) ratio <= 1.03. "Byte viable under the FLOP budget" := (ii) ratio <= 1.05. Refute := (ii) ratio > 1.15 (then the byte direction is dead at our scale under matched compute, and only BX2 matters).
- **GPU-minutes (estimated from `hardware-spark.md`):** (ii) ~35 (210M bytes at ~100k/s, range 30-41); (iii) ~125 (108-147); LR checks 2 x ~5; total ~3 GPU-hours with the P3 flex-SWA prerequisite; **blocked until P3 exists** (dense-mask SWA at T=4,096, ~6 GB per layer, see section 4).

### BX2: BLT-lite P=4 vs BPE E1 (SQ181)
- **Question:** does a patch-4 byte interface recover BPE-level quality at about BPE-level FLOPs, and does it do so without a tokenizer or an 8k embedding?
- **Arms:** BLT-lite B (E1 body on patches) at matched text (746.9M bytes; 1.14x E1's FLOPs, ~2x E1's time); vs BPE-E1 and vs Byte-E1 (iii).
- **Prediction:** clean-val bits/char ratio to BPE-E1: **1.00-1.12** (central 1.05): fixed patches lose their first byte's local context, 14M params give the local layers little room, but the core sees the same number of positions as E1 and gets to use all of its capacity on them. BLT-lite beats Byte-E1 (iii) at ~1/2 the training FLOPs-time, ratio vs Byte-E1 (iii) 0.97-1.05 at 0.3x the FLOPs.
- **Criteria:** "pays" := ratio <= 1.03 (vs BPE-E1) **and** bits/char <= Byte-E1 (iii) + 1%; "promising" := ratio <= 1.08; "not worth it" := ratio > 1.12. Also record the cost ratio at the matched-text point (FLOPs 1.14x computed, time to be measured).
- **GPU-minutes:** ~65 (60-75; 200M tok-eq at 45-55k/s) plus a 20M-token LR check (~7). Needs a new module (`gogeta/` is not editable by this task: ~100 lines: embed, pool, upsample, local blocks) and a byte data path (the `train.bin` is uint16 BPE; bytes are `docs_*.jsonl` text encoded to utf-8, no tokenizer).
- **Ablation if BX2 passes:** E2 looped core on patches (Variant C) vs unlooped: the loop's cost per byte is 1/4 of the byte-level equivalent.

### Why these two
BX1 answers "is the byte interface alone viable" (also the MambaByte-style question for our mixers); BX2 answers "can the interface be made nearly free". If BX1(ii) is within 5% and BX2 is within 3%, the byte direction is worth a CPU+GPU investment (dynamic patching); if both fail, the cheap answer is "no, not at 14M" and costs ~4.5 GPU-hours total.

### Tiny CPU checks that could precede the GPU
1. (Done) `seq_stats.py`: 3.73 bytes/token.
2. Causality unit test of the BLT-lite wiring (no leakage across patch boundaries; same style as C3), CPU, minutes; part of the build.
3. A CPU-sized pilot (d=64, 2k bytes, 20M bytes seen) of byte-E1 vs BPE-E1 trend, to check that the matched-FLOP direction matches the papers. Direction only; our rules (AGENTS.md §0) say real claims need the full run.

## 7. Papers to add to refs (arXiv id + one line)

Not added to `refs/fetch_refs.py` (another agent owns it). Suggested `refs/14-byte-level/`:

| arXiv | Why |
|---|---|
| 2105.13626 ByT5 | the original "byte models are robust but slow" baseline |
| 2103.06874 CANINE | downsample/upsample around a deep stack, character level |
| 2106.12672 Charformer | learned soft block pooling over bytes |
| 2305.07185 MegaByte | fixed patches, local/global split, FLOP formulas, init std 0.006 |
| 2401.13660 MambaByte | SSM on raw bytes, the closest precedent to GDN bytes; speculative byte decoding |
| 2404.14408 SpaceByte | rule-based dynamic patches; fixed-patch ablation; the cleanest matched-FLOPs byte vs subword table |
| 2412.09871 BLT | entropy patching at 8B/4T bytes, HumanEval/MBPP, hash n-gram embeddings |
| 2507.07955 H-Net | learned chunking, byte-level GitHub numbers, crossover timing |
| 2506.14761 AU-Net | autoregressive U-net from bytes to multi-word units |
| 2603.15953 T-Free HAT | byte encoder / word backbone / byte decoder, Llama retrofits |
| 2602.04289 Proxy Compression | byte LM with a compressed-view auxiliary; code LM experiments at fixed compute |
| 2605.12928 Efficiency Gap in Byte Modeling | the counter-evidence on compute-matched byte overhead |
| 2506.14123 Sampling from your LM one byte at a time | token-boundary issues for code in BPE models |
| 2402.19155 bGPT | next-byte prediction across file types (lower priority) |
| EvaByte (blog, hkunlp.github.io/blog/2025/evabyte; no arXiv id found) | byte LM with multibyte heads; HumanEval/MBPP; note the claims are blog-only |

## 8. Recommendation

1. Run **no byte GPU work now**: prerequisites are P3 (flex SWA, otherwise T=4,096 is infeasible), a byte data path, and the K9-clean scorer (SQ95 parity). Start with BX2's build (fixed patches), since it is the only arm that is not 3.6x the FLOPs; BX1 is the control that makes BX2 interpretable.
2. Expectation to hold ourselves to: a byte interface is a **cost** at 14M params, not a gain, and the strongest argument for it (a looped core on patches that makes loops cheap) is a hypothesis that BX2 followed by Variant C must earn.
3. Everything in the survey is reported, not verified; ledger rows are proposed in `experiments/byte_level/proposed-claims.md`.

---
**Correction (main agent, v0.4.60):** MambaByte's "2.6x" decoding speedup (2401.13660, Table 6) is **lossy**: a drafted byte is accepted if it's in the verifier's top-3, and faithfulness to greedy MambaByte is 0.89. It's measured against plain MambaByte-972M with a 110M subword drafter, on PG19, hardware and batch not stated in that table. The paper's other "2.6x" (MambaByte vs MegaByte, Table 5) is a different comparison. See `notes/research-speculative-bytes.md`.
