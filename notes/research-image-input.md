# Research: how could Gogeta-0 / FusionDance accept image inputs?

Status: research only, a nice-to-have for later. Nothing here has been run. Per AGENTS.md section 0, every claim below is a hypothesis until a local check verifies it.
Date: 2026-11-22. Method: web searches plus background knowledge. **Items marked [unverified id] were not confirmed by a search hit and should be checked before going into `refs/fetch_refs.py`.** Anything marked **[speculation]** is my own inference, not a sourced result.

## 0. The target, in numbers

- ~13M params, d=384, 6 heads (head_dim 64), 8k BPE, T=1024, window 256 (`gogeta/model.py`).
- Layout: prelude SWA, then core [GDN, GDN, SWA] x r (shared weights), then a coda with one full-attention layer.
- BitNet b1.58 ternary weights are planned (E-ternary), so a vision path should be ternary-compatible.
- Budget: 200M tokens, one laptop GPU.
- Consequence for images: every image token spends one of the 1024 positions. A 256-token image leaves 768 for code. Anything inside the loop is paid r times.

## 1. The approaches

### 1a. Frozen encoder plus projector (LLaVA-style)
- LLaVA (arXiv 2304.08485 [unverified id]) uses CLIP (2103.00020) or SigLIP (2303.15343) with an MLP projector into the LM embedding space. Two stages: align the projector, then instruction-tune.
- Small-model versions:
  - **SmolVLM** (arXiv 2504.05299, https://arxiv.org/abs/2504.05299). 256M model = 93M SigLIP-B/16 + SmolLM2-135M. 500M model = same encoder + SmolLM2-360M. Pixel-shuffle compresses tokens by r^2. A 512x512 image gives 1024 raw tokens at B/16 (the paper's number), and PS=4 cuts that 16x to 64. The paper finds small LMs (135M) benefit less from big encoders and prefer more aggressive shuffle [paraphrase of search summary, check].
  - **nanoVLM** (HF blog, https://huggingface.co/blog/nanovlm). SigLIP-B/16-224 + SmolLM2-135M, pixel shuffle + linear projector, a pure-PyTorch trainer. No paper. This is the closest code template for a PoC.
  - **MobileVLM / V2** (arXiv 2312.16886, 2402.03766). CLIP ViT-L/14 + MobileLLaMA 1.4B/2.7B. The LDP projector is a depthwise-conv downsampler (V2: 144 tokens from 576).
  - **TinyLLaVA** (arXiv 2402.14289 [unverified id]), **PaliGemma** (arXiv 2407.07726 [unverified id]; SigLIP-So400m + Gemma-2B, 256 tokens at 224px), **Moondream 0.5B** (no paper; int8/int4 QAT, per the efficient-VLM survey arXiv 2504.09724).
- Gap: the smallest published LMs here are 135M, about 10x our size. **[speculation]** No one has shown a useful 13M LM VLM; the encoder would dwarf the LM (93M vs 13M).

### 1b. Encoder-free / pixel patches as tokens
- **Fuyu-8B** (Adept blog, no paper): linear projection of raw patches straight into the decoder.
- **EVE** (arXiv 2406.11832, https://arxiv.org/abs/2406.11832): patch embedding layer plus patch aligning layer, with alignment to a vision backbone's patch features. About 35M public image-text pairs are enough to rival encoder-based VLMs at 7B. **EVEv2** (arXiv 2502.06788): 100M pairs, modality-specific weights inside the decoder.
- **SOLO** (arXiv 2407.06438 [unverified id]): a single transformer over raw patches, 7B.
- Fit: elegant (one extra linear layer), but all known results rely on a big LM that can learn vision itself. **[speculation]** At 13M, the LM has no spare capacity to learn low-level vision from scratch. A cheap hybrid (tiny conv stem, see section 6) is more plausible.

### 1c. Discrete image tokenizers (early fusion with a shared vocabulary)
- VQ-VAE (1711.00937), VQGAN (2012.09841), MAGVIT-v2 / LFQ (2310.05737) [unverified ids]; Chameleon (arXiv 2405.09818 [unverified id]; early-fusion, 1024 tokens per image from an 8192-code VQ); Emu3 (arXiv 2409.18869, https://arxiv.org/abs/2409.18869; single transformer, next-token only, tokenizer for image/video).
- **TiTok** (arXiv 2406.07550, https://arxiv.org/abs/2406.07550): a 1D tokenizer, 256x256 image in as few as 32 discrete tokens. The only approach whose token count is small enough to be comfortable at T=1024.
- Fit: the cleanest match for a pure causal LM (just extend the vocab, same loss). Cost: the 8k text vocab would grow by 1k to 8k image codes (the embedding/head matrices are a big share of 13M params), and token-prediction supervision on images is data-hungry. Needs a separate pretrained tokenizer (download; licence check). **[speculation]** For understanding-only (image in, code out) you do not need the decoder half or image generation, so a tiny tokenizer on small images could do.

### 1d. Cross-attention adapters
- **Flamingo** (arXiv 2204.14198): Perceiver Resampler gives a fixed 64 visual tokens; tanh-gated cross-attention (gate initialized at 0) interleaved into a frozen LM. Llama 3.2-Vision (Llama 3 herd, arXiv 2407.21783 [unverified id]) uses the same family of adapters.
- Fit: image tokens never enter the LM's sequence, so they do not eat T or hit the GDN state. Fits our rule "nothing quadratic inside the loop" only if cross-attention is placed in the coda (once), and the number of image tokens K is small (64). Added params: ~4 d^2 per cross-attn layer = ~0.6M at d=384, so cheap. **[speculation]** Gated-zero init is a natural fit with ternary training because the base text model stays intact while the gate opens.

### 1e. Early fusion, joint training from scratch
- Chameleon / Emu3 / EVE-style: image and text trained together from the start. Needs far more tokens than our 200M budget and risks hurting the code-only objective. Not recommended before the text model is settled.

## 2. Tiny-model reality check

| Model | LM | Vision | Image tokens | Training data (as I recall; verify) |
|---|---|---|---|---|
| SmolVLM-256M | 135M | SigLIP-B/16 93M | ~64 at 512px, PS=4 | large mixed corpus (Cauldron, Docmatix) |
| nanoVLM | 135M | SigLIP-B/16-224 | 49-64 after shuffle | ~1.7M samples (Cauldron subset), 1 GPU for hours |
| MobileVLM V2 1.7B | 1.4B | CLIP ViT-L/14 | 144 | 1.2M captions + 3.6M instruct |
| PaliGemma | 2B | SigLIP-So400m | 256 at 224px | broad |
| EVE | 7B | none | patches | 35M pairs |

None goes below ~100M LM params. Our ~13M LM would be the smallest by an order of magnitude. **[speculation]** Expect narrow-domain success (synthetic UI, plots, diagrams) rather than general VQA.

## 3. Fit with linear-time / SSM / hybrid mixers

Sources:
- **Vision Mamba (Vim)** arXiv 2401.09417 [unverified id]: bidirectional scan over the patch sequence.
- **VMamba** arXiv 2401.10166 (https://arxiv.org/abs/2401.10166): SS2D / cross-scan, four scan directions over the patch grid.
- **Cobra** arXiv 2403.14520 (https://arxiv.org/abs/2403.14520): Mamba LM + fused DINOv2/SigLIP features; 3-4x faster than efficient baselines; matches LLaVA-class at ~43% of params.
- **VL-Mamba** arXiv 2403.13600 (https://arxiv.org/abs/2403.13600): Vision Selective Scan connector, 2D-aware, between the encoder and the Mamba LM.
- **ML-Mamba** arXiv 2410.13613 [unverified id, not found by search]: Mamba-2 multimodal LM.
- **ViG** arXiv 2405.18425: gated linear attention for vision, with bidirectional scan. **Vision-RWKV** arXiv 2403.02308 [unverified id]. **ENA** arXiv 2508.11921: a DeltaNet-based component in an N-dimensional vision encoder. **DeltaMIL** arXiv 2512.19331: gated delta rule on whole-slide images. Qwen3-Next/3.5 VLMs reportedly pair gated-delta layers with image tokens (secondary source only; check).

Findings relevant to us:
1. **Scan order matters for a causal recurrent LM.** A GDN layer sees patches strictly in order, so a row-major raster means each patch only sees up and left of itself. Mitigations in the literature: multi-direction scans (VMamba, 4x cost), bidirectional (Vim; cannot be done in a causal LM without a separate non-causal encoder), or a connector that does 2D mixing before the LM (VL-Mamba). In practice Cobra/VL-Mamba keep the vision encoder bidirectional and feed an ordered token sequence into a causal Mamba LM, so the LM only needs the global summary. **[speculation]** For us the same trick applies: do any 2D mixing in a small bidirectional image front end; the causal core only reads the result in raster order.
2. **Cost, per token, of long image sequences.**
   - GDN: O(1) per token (state of 6 heads x 64 x 64), so 256 or 1024 image tokens are fine in time and memory, but the fixed-size state may blur fine spatial detail. Delta-rule overwrite is better than plain linear attention for recall, which helps.
   - SWA (window 256): O(256) per token. A 256-token image fits exactly in one window; at 1024 tokens, a patch cannot see the top of the image by SWA alone, so any global image-to-code retrieval must go through GDN state or the single full-attention coda layer. Note our current SWA implementation uses a dense mask (README), so it is O(n^2) in practice today.
   - Full attention (coda): O(n^2) in the sequence, runs once. 1024 tokens is cheap on our scale.
   - Looped core: all image tokens are processed r times (r=4: 4x cost). **[speculation]** Prefer inserting image tokens in the prelude/coda path or reducing to <= 64 tokens so loop cost stays small.
3. Code-after-image is the important order: image first, then code. A causal recurrent model only reads the image once, left to right, before the code starts, which is the right order for screenshot-to-code. The reverse (code, then image) is the harder one.

## 4. Ternary / low-bit vision

- **ViT-1.58b** (arXiv 2406.18051, https://arxiv.org/abs/2406.18051): ternary-weight ViT with 8-bit activations on CIFAR-10 / ImageNet-1k; needs training from scratch with the ternary scheme.
- **BitVLA** (arXiv 2506.07530, https://arxiv.org/abs/2506.07530): BitNet b1.58 2B4T LM + SigLIP quantized to 1.58-bit by distillation-aware training. The closest precedent to a fully ternary VLM.
- **TernaryCLIP** (arXiv 2510.21879, https://arxiv.org/abs/2510.21879): ternary CLIP via distillation.
- **LLaVA-OLMoBitNet1B** (arXiv 2408.13402): ternary LM with a full-precision CLIP.
- **BitNet b1.58 Reloaded** (arXiv 2407.09527): ternary also works on small networks (relevant to us, 13M scale).
- Takeaway: ternary vision works with distillation from a full-precision teacher or from-scratch training; patch-embed (first) and projector layers are usually kept in higher precision. **[speculation]** Keep the first patch/conv layer full-precision (tiny anyway) and let the ternary core read it, like our embeddings/head already stay full precision.

## 5. Looped / recurrent-depth models with multimodal input

All very recent preprints; ids come from search results and I did not open the papers, so treat claims as the authors' own and re-check ids.
- **RecursiveVLM** arXiv 2602.09080: recursive transformer for LMMs, recursive connector + monotonic recursion loss.
- **LoopVL** arXiv 2609.38426 [unverified id]: loop transformers extended to VLMs.
- **HIVE** arXiv 2602.05359: looped transformer with hierarchical visual-cue injection for latent reasoning.
- **RD-VLA** arXiv 2602.07845: recurrent-depth vision-language-action.
- **LoopViT** arXiv 2602.02156: looped ViT for ARC visual puzzles.
- **Loop Before You Leap** arXiv 2611.06206 [unverified id]: looped models under a 100M-word budget, text-only and multimodal.
- Relevance: the "re-inject visual cues each loop" idea (HIVE) fits our existing `inject` of the embedding at every loop (see `inj_norm_e` in `model.py`). **[speculation]** Cross-attention or re-injection of image features into each loop pass could give the loop something to "look at" repeatedly. No one has done this with GDN + SWA.

## 6. Code-relevant tasks and datasets

| Task | Dataset | Size | Licence (check before use) |
|---|---|---|---|
| Screenshot to HTML/CSS (synthetic) | **WebSight** (arXiv 2403.09029) | ~2M pairs (v0.1 is 823k; v0.2 is 2M, I recall) | CC-BY-4.0 on the card; users must follow source licences and disclose use |
| Real webpage to code | **WebCode2M** (arXiv 2404.06369) | ~2.5M | check card [unverified] |
| Benchmark | **Design2Code** (arXiv 2403.03163) | 484 pages | built on C4; paper says ODC-By; repo says research-use only |
| Plot / chart to code | **ChartMimic** (arXiv 2406.09961), **Plot2Code** (arXiv 2405.07990 [unverified id]), Chart2Code (arXiv 2510.17932) | 1-4k | ChartMimic images derive from CC-BY arXiv figures; dataset licence not confirmed |
| GUI to markup, 2017 baseline | **pix2code** (arXiv 1705.07962 [unverified id]) | ~1.7k synthetic apps | check repo |
| Screenshot parsing pretraining | **Pix2Struct** (arXiv 2210.03347 [unverified id]) | web-scale | model Apache-2.0; data from C4 |

Fit to our model:
- Our training corpus is Python only. WebSight targets HTML+Tailwind CSS, which is a separate distribution that would need new BPE merges (the 8k vocab was trained on Python).
- **[speculation]** The best-matched, cheapest image-to-Python tasks are *synthetic*: render matplotlib plots from Python scripts we generate ourselves (plot to code), render turtle/PIL drawings, render a snippet as a syntax-highlighted image (OCR-of-code). All are self-generated, so no licence issues and unlimited data. The inverse direction (code to image) is not needed.

## 7. Comparison table

Params-added are for d=384; "fits 13M linear-time?" is my judgement.

| Approach | Params added | Image tokens in LM | Fits 13M linear-time? | Data / compute | Licence notes |
|---|---|---|---|---|---|
| Frozen SigLIP-B/16 + pixel-shuffle + linear (SmolVLM/nanoVLM) | 93M frozen + ~0.1-0.5M projector | 49-64 (PS=4) | Tokens yes. Encoder 7x our LM. **Poor balance**; can't run on an edge budget | 1-2M pairs to align; encoder frozen (cheap) | SigLIP Apache-2.0 (check); needs download |
| Tiny trained-from-scratch ViT/conv stem + projector (distilled from SigLIP, ternary ok) | 1-5M | 64-256 | **Yes** (mid) | Synthetic data + optional distillation; hours on 1 GPU | We own it |
| Linear patch embed only (Fuyu/EVE/SOLO) | ~0.05-0.2M | 64-256 | Marginal: LM must learn vision; **[speculation]** works only for synthetic/clean images | Low-level vision must be learned from the 200M-token budget | None |
| Discrete tokens, large VQ (Chameleon/Emu3) | +1k-8k vocab rows (0.4-3M) + a tokenizer | 256-1024 | Poor: tokens eat T; large vocab | Large | Tokenizer licences differ; check |
| 1D tokenizer (TiTok) | +1k-4k vocab rows + tokenizer | 32-128 | Plausible for understanding | Needs a pretrained TiTok or training our own | Code Apache-2.0 [unverified] |
| Flamingo-style cross-attn in the coda + Perceiver Resampler | ~0.6M per xattn layer + ~0.5M resampler | 0 in sequence (K=16-64 side tokens) | **Yes**; no T cost; loop-free | Low | None |
| Early fusion from scratch | n/a | any | No (budget) | Very high | n/a |

## 8. Recommended path

1. **[confidence: medium]** Treat the first PoC as *narrow-domain, synthetic, image-first, then Python*: small grayscale or RGB images (64x64 to 128x128) rendered from Python we generate (matplotlib plots, turtle drawings, rendered code), and predict the source script. This avoids any dataset licences and any large encoder.
2. **[confidence: medium]** Front end = tiny conv or ViT stem trained from scratch (1-3M params) that outputs 32-64 tokens at d=384, inserted as soft embeddings before the code tokens (prefix). Keep the stem in bf16; core stays as is. Bidirectional 2D mixing happens inside the stem (the VL-Mamba / Cobra lesson); the causal core only reads the resulting ordered tokens.
3. **[confidence: low-medium]** If prefix tokens interfere with the loop or text quality, fall back to Flamingo-style gated cross-attention from the coda to the image tokens (gate init 0, text behaviour preserved).
4. **[confidence: low]** Discrete tokens (TiTok-like) only if soft tokens train badly; and a frozen SigLIP only as a *distillation teacher* (never as a runtime component) to give the stem a better target, as in BitVLA / TernaryCLIP.
5. Ternary: train the stem full-precision first (one ingredient at a time, per AGENTS.md section 5), ternarize later.
6. Do not do this before E-ternary and the full fusion are settled; images are a separate capability, not a fusion ingredient.

### Cheapest proof of concept (described, not run)
- Check script in `experiments/checks/`, seeded, CPU/small GPU, **<= 15 min** (AGENTS.md section 0).
- **Claim:** a tiny causal LM can use an image prefix. **Data:** 20k synthetic images (32x32 grayscale), each rendered from a one-line Python snippet such as `plt.plot` of a few random points, or a grid of `n` squares from `draw(n)`. The target is the snippet text (numbers vary by sample).
- **Model:** d=128, 2-4 layers, GDN+SWA mixture, patch stem of 4x4 patches giving 64 tokens (a linear projection plus a 1-layer conv), prefix then code tokens, loss only on code tokens.
- **Criteria (written before running):** with the image, the held-out digit/count accuracy in the generated code is >= 90%; with the image zeroed or shuffled it is at chance (control). A second arm compares layouts at matched params: all-attention vs GDN+SWA vs GDN-only with raster order, to measure whether scan order hurts (raster vs a 2D-aware stem).
- **Fail criteria:** the image-on arm does not beat the shuffled-image arm by >= 20 points. If it fails, the next cheapest arm is cross-attention instead of prefix.

## 9. Risks

- **Capacity.** 13M params may be too small to be both a good code LM and a vision reader; the image path could degrade text loss. Measure Python val loss with and without images (and note the known validation-contamination caveat, K9, in the README).
- **Context budget.** T=1024 total; 256 image tokens take a quarter. Prefer <= 64.
- **Loop cost.** Image tokens inside the loop cost r times; also the "nothing quadratic in the loop" rule.
- **Scan order.** Causal raster order loses vertical context; mitigation in section 3.
- **Vocabulary/tokenizer.** The 8k BPE was trained on Python; HTML/CSS tasks need a new tokenizer, which breaks comparability with existing checkpoints.
- **Ternary.** First-layer and projector precision is a known weak spot; ViT-1.58b needed scratch training.
- **Data licences.** WebSight is CC-BY-4.0 but with attribution/disclosure requirements; Design2Code repo says research-only; always check the live cards. Use self-generated synthetic data where possible.
- **Evaluation realism.** Synthetic success may not transfer to real screenshots. **[speculation]** Plan on it not transferring.
- **Reference reliability.** Several ids above are from memory or from search summaries; verify.

## 10. Papers to add to refs (arXiv id + one line)

Ids not marked are confirmed by a search hit; **(v)** = verify first.

- 2504.05299 SmolVLM: small-VLM design study (encoder size vs LM size, pixel shuffle).
- 2403.14520 Cobra: Mamba-LM multimodal model, 3-4x faster than efficient baselines.
- 2403.13600 VL-Mamba: Vision Selective Scan connector for 2D-to-sequence.
- 2401.10166 VMamba: SS2D four-direction scan for images.
- 2401.09417 (v) Vision Mamba (Vim): bidirectional SSM for images.
- 2405.18425 ViG: gated linear attention vision backbone with bidirectional scan.
- 2508.11921 ENA: DeltaNet-based N-D attention for vision.
- 2412.06464 Gated Delta Networks (already listed in refs presumably; cited for DeltaNet vision context).
- 2406.11832 EVE: encoder-free VLM, 35M pairs.
- 2502.06788 EVEv2: improved encoder-free baselines, 100M pairs.
- 2406.07550 TiTok: 256px image in 32 discrete tokens.
- 2409.18869 Emu3: next-token prediction on discrete image/video tokens.
- 2405.09818 (v) Chameleon: early-fusion mixed-modal token model.
- 2204.14198 Flamingo: Perceiver Resampler + gated cross-attention.
- 2312.16886 MobileVLM and 2402.03766 MobileVLM V2: mobile VLM with LDP projector.
- 2402.14289 (v) TinyLLaVA: small LLaVA-style VLM framework.
- 2407.07726 (v) PaliGemma: SigLIP + Gemma 3B VLM.
- 2504.09724 Survey on efficient VLMs (includes Moondream/SmolVLM comparisons).
- 2406.18051 ViT-1.58b: ternary vision transformer.
- 2506.07530 BitVLA: 1.58-bit vision-language-action model with distilled ternary SigLIP.
- 2510.21879 TernaryCLIP: ternary CLIP via distillation.
- 2408.13402 LLaVA-OLMoBitNet1B: ternary LM LLaVA.
- 2407.09527 BitNet b1.58 Reloaded: ternary at small scale.
- 2602.09080 RecursiveVLM: recursive transformer for multimodal models.
- 2602.05359 HIVE: looped latent reasoning with visual cue injection.
- 2609.38426 (v) LoopVL: loop transformers for VLMs.
- 2611.06206 (v) Loop Before You Leap: recurrent depth in data-limited multimodal pretraining.
- 2403.09029 WebSight: 2M synthetic HTML/screenshot pairs.
- 2403.03163 Design2Code: screenshot-to-code benchmark.
- 2404.06369 WebCode2M: real-world webpage-to-code dataset.
- 2406.09961 ChartMimic: chart-to-code benchmark.
- 2510.17932 Chart2Code: hierarchical chart-to-code benchmark.
- 2303.15343 (v) SigLIP: sigmoid-loss image encoder (distillation teacher).
- 2304.08485 (v) LLaVA: the projector-plus-instruction-tuning template.
- 2210.03347 (v) Pix2Struct: screenshot parsing pretraining.
- 2405.07990 (v) Plot2Code: plot-to-code benchmark.
- 1705.07962 (v) pix2code: original GUI-to-code.

## 11. What I could not confirm

- ML-Mamba, Vision-RWKV, Chameleon, Fuyu, SOLO, TinyLLaVA, Plot2Code, Web2Code and pix2code licences/details did not come back in searches.
- Training-set sizes in section 2 for nanoVLM, MobileVLM and PaliGemma are from memory.
- No primary source found for Gated DeltaNet used inside a published VLM (only a secondary article on Qwen's delta-rule VLMs).
