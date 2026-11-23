# Byte-level / tokenizer-free LMs, 2025-2026 literature update

Research side quest, 2026-11-22 (subagent; web search + arXiv API only; no GPU, no installs, no git). Companion to
`notes/research-byte-level.md` (which covers the foundations and our own CPU arithmetic) and to the byte side quests
SQ180-SQ189 (`experiments/byte_level/proposed-side-quests.md`). PDFs are in `refs/14-byte-level/` (32 files, 33.2 MB).

**Verification method.** Every arXiv id below was checked against the arXiv API (`export.arxiv.org/api/query?id_list=...`):
the returned title and first-submission date match. "Verified: yes" means id + title + date confirmed that way.
Numbers in the table come from the **abstract**, or, where marked (page), from the paper's HTML as summarised by a fetch
tool; I did **not** read PDFs, so treat every number as **reported by <paper>**, to be re-checked before it enters the
claims ledger. Labels: **reported**, **[speculation]**, confidence high/med/low.

## 1. Table

Scale = parameters / data as stated. "Ours" = relevance to a ~14M-param, ~0.75 GB, linear-time looped ternary code model.

| arXiv | Title | Date | Method | Scale | Headline result relevant to us | Verified |
|---|---|---|---|---|---|---|
| 2105.13626 | ByT5 | 2021-05 | T5 on raw bytes | T5 sizes | Robust, slower; baseline | yes |
| 2103.06874 | CANINE | 2021-03 | char encoder, down/up-sample | mBERT-size | Pooled hierarchy precedent | yes |
| 2106.12672 | Charformer | 2021-06 | GBST learned soft blocks | base | Learned chunking precedent | yes |
| 2305.07185 | MegaByte | 2023-05 | fixed patches, global/local | 1.3B, 400B bytes | Our BLT-lite template | yes |
| 2401.13660 | MambaByte | 2024-01 | Mamba on bytes | 353M-1.6B | SSM on bytes ties/beats MegaByte; Code 0.396 bpb | yes |
| 2404.14408 | SpaceByte | 2024-04 | rule-based patches | 793M+184M | Dynamic rule beats fixed 6-12% | yes |
| 2412.09871 | BLT | 2024-12 | entropy patches, local enc/dec | up to 8B, 4T bytes | HumanEval 35.4 vs 31.1 | yes |
| 2402.19155 | bGPT | 2024-02 | next-byte over any file | n/a | low priority | yes |
| 2502.14553 | MBLM, Multiscale Byte LM | 2025-02-20 | hierarchical decoder stack, Transformer or Mamba blocks | contexts to 5M bytes | Hybrids (Mamba+attn) efficient on very long byte seqs; near-linear generation | yes |
| 2506.14761 | AU-Net (From Bytes to Ideas) | 2025-06-17 | autoregressive U-Net: bytes -> words -> 2 -> 4 words | not in abstract | Shallow hierarchies tie BPE at controlled compute | yes |
| 2506.14123 | Sampling from Your LM One Byte at a Time | 2025-06-17 | BPE LM -> byte distribution | n/a | Token-boundary bias (code) is a real artefact | yes |
| 2507.07955 | H-Net / Dynamic Chunking | 2025-07-10 | learned chunker, Mamba-2 outer layers | 680M-1.6B, 100B | GitHub 0.316 vs 0.338 bpb; crossover after 30-100B bytes | yes |
| 2508.05628 | H-Net++ | 2025-08-07 | H-Net + 1.9M cross-chunk mixer + latent hyper-prior | 1.4B tokens Persian | Works on small data: 12% better compression than BPE GPT-2-fa; staged seq-length curriculum | yes |
| 2511.11346 | MTPC (multi-byte prediction w/ probabilistic circuits) | 2025-11-14 | MTP + spec. decoding on EvaByte, byteified Llama 3.2 3B | 3B+ | Byte decode speed lever; lossless vs verifier | yes |
| 2512.15586 | Bolmo | 2025-12-17 | byteify existing subword LMs, 2-stage | 1B, 7B | +16.5 abs. STEM over BLT-7B (summary); <1% pretrain cost | yes |
| 2601.22805 | SOMBRERO | 2026-01-30 | boundary-enrichment metric B + confidence-alignment loss + input-level smoothing | 1B | Better accuracy-efficiency trade-off on text, code, math | yes |
| 2602.04289 | Proxy Compression | 2026-02-04 | joint raw-byte + compressed-view training, bytes only at inference | 0.5B-14B, 100B-320B symbols, RefineCode | Tokenizer leads <4B; proxy equals/exceeds tokenizer only at 7-14B (page); proxy >> pure byte at fixed compute | yes |
| 2603.03583 | ByteFlow Net | 2026-03-03 | coding-rate chunking, Top-K static graph, sliding-window attn + Canon layers (no SSM) | 0.6B (50B), 1.3B (500B) | val bpb 0.86 vs Llama-BPE 0.89 (600M); avg acc 50.9 vs 49.2 (page) | yes |
| 2603.15953 | T-Free HAT | 2026-03-16 | byte enc -> word backbone -> byte dec | 7B from scratch, 8B/70B retrofits | Better compression and spelling robustness | yes |
| 2604.27263 | Decoupling Subword Benefits via Byte-level Simulation | 2026-04-29 | controlled byte pretraining isolating throughput, vocab, boundary priors | 1.7B (8k ctx), plus 68M appendix | Gap is mostly **training throughput** and boundary priors; vocab size adds little (page) | yes |
| 2605.01188 | Compute Optimal Tokenization | 2026-05-02 | 988 BLT models with controllable compression | 50M-7B | Params scale with **bytes**, not tokens; optimal compression falls with compute | yes |
| 2605.08044 | Fast BLT | 2026-05-08 | BLT-D diffusion, BLT-S self-speculation, BLT-DV | BLT-scale | >50% lower decode memory-bandwidth cost | yes |
| 2605.09630 | Scratchpad Patching | 2026-05-10 | entropy-triggered scratchpads inside patches to fix "patch lag" | ~2B and 500M, ~400B bytes, code+text+math | Fixed p=8 MBPP 24.1 -> 32.1 with SP; p=16 + SP ~ byte baseline at 16x smaller trunk KV (page) | yes |
| 2605.12928 | Efficiency Gap in Byte Modeling | 2026-05-13 | compute-matched byte vs BPE, AR and masked diffusion | 48M-1.2B, 6e18-2e21 FLOPs | AR gap: 7.9x compute at 1.0 bpb, 2.3x at 0.8, parity extrapolated ~1.3e22 (page) | yes |
| 2605.30080 | ATDC | 2026-05-28 | curriculum compression ratio for H-Net chunker | FineWeb-Edu 100B | Low->high compression curriculum stabilises chunking | yes |
| 2608.15454 | Dynamic Multi-Byte Prediction | 2026-08-16 | variable-length multi-byte decode aligned to latent segments | hierarchical LM | Pareto-optimal speed/quality, no extra params | yes |
| 2609.12303 | Breaking the Token Ceiling | 2026-09-11 | 1B byte vs token, overtrained to 1T bytes, with distillation | 1B | Token wins at low FLOPs, byte overtakes later | yes |
| 2610.05978 | Byte LMs: Scaling, Emergent Abstractions | 2026-10-05 | plain byte Transformers + token-superposition training + hash embeddings | not in abstract | Beats subword Transformers as size scales; <=25% of mid layers can be restricted to local reps; 3.4x spec. acceptance | yes |
| 2610.11790 | Boundary dependence vs entropy patching | 2026-10-08 | loss-change boundary signal | BLT-1B; from-scratch 1M-50M, 128-byte ctx | Entropy rule blind to computed values; effect absent at 1M, present at 12M-50M | yes |
| 2610.22100 | Entropy patching skips computed values | 2026-10-15 | two-byte lookup-table patcher | same 1M-50M | Table beats entropy at 50M by 36.8 pts on computed results; no rule won in >1 of 6 domains | yes |
| 2611.04215 | Tokenizer Myths | 2026-11-03 | metric analysis | n/a | BPB is **not** tokenizer-agnostic; corr. with downstream weak | yes |
| 2611.13607 | One Byte, One Token | 2026-11-10 | vocab 248k -> 384, Qwen3.8-27B (48 GDN + 16 attn), no patching | 27B, 4.77 GB supervised bytes | Pure-byte GDN hybrid works; 3-5x less output per step, prefill 6.5x slower at 64K | yes |

Not added (verified but off-topic or weak): 2608.03599 (position paper, no experiments), 2610.22071 (UTF-16 boundaries for
Brahmic scripts), 2610.22906 (federated sentiment), 2608.05806 (not byte-specific), 2608.18062 TokEval, 2506.15889 (BPE
pre-tokenisation). **Dropped as unverifiable: none**, but EvaByte has no arXiv paper (blog only; it is known only through
2511.11346, which I did verify).

## 2. What is new since BLT and H-Net (synthesis)

1. **Byte models are no longer all from scratch.** Bolmo (1B/7B) and "One Byte, One Token" (27B GDN hybrid) convert
   pretrained subword models cheaply; T-Free HAT retrofits Llama 3.1. For us, the scratch-only setting is the hard one.
2. **Patching is now a boundary-selection problem with its own metrics.** SOMBRERO (boundary enrichment), ByteFlow
   (coding rate), ATDC (compression curriculum), 2610.11790/2610.22100 (entropy's blind spot) all say the rule matters,
   and that entropy is not uniformly best. Scratchpad Patching adds a different fix: keep cheap boundaries but refresh context
   inside the patch.
3. **Scale threshold evidence got sharper.** Efficiency Gap: AR byte models need 7.9x compute at 1.0 bpb, 2.3x at 0.8 bpb, parity
   extrapolated near 1e22 FLOPs. Proxy Compression: tokenizer leads below ~4B on code. Token Ceiling: tokens win at low FLOPs.
   Counter-voice: Byte LMs 2610.05978 and ByteFlow claim wins at 0.6B-1.3B, with extra tricks (hash embeddings, superposition,
   adaptive patching).
4. **Why subword wins is being dissected.** 2604.27263: mostly training throughput (positions per second) and boundary priors,
   not vocabulary. That points at patching/compression as the lever, not at the embedding table.
5. **Decode speed work**: Fast BLT, MTPC, Dynamic MBP, and 2610.05978's 3.4x acceptance all target the byte-by-byte decode
   cost; all reported at 1B+.
6. **Evaluation caution**: Tokenizer Myths says BPB is not tokenizer-agnostic.

## 3. Impact on SQ180-SQ189 (design changes)

Reasoning is mine; every paper number is **reported**, not ours.

| SQ | Finding | Suggested change | Confidence |
|---|---|---|---|
| SQ180 (BX1 byte-E1 vs BPE-E1) | Efficiency Gap: AR byte overhead is large at small compute and shrinks with scale; at 48M and 6e18 FLOPs it is well above the 2.3x seen near 0.8 bpb (page). Our run is ~1e16-1e17 FLOPs, below the studied range. | Keep the prediction (matched-FLOPs ratio 1.10-1.35) but add a note that the paper's trend implies we sit in the worst region; do not read a loss as a verdict on bytes at scale. A cheap sanity arm: **byte-E1 with 4-byte compression warm-up** (2604.27263) rather than pure bytes. | med; the extrapolation to 14M is **[speculation]** |
| SQ180/181 | 2604.27263: the main bridge from byte to subword is throughput; compressing 4 bytes to one latent for the first 50k steps then un-patching beat the plain byte baseline (page). | Add an arm to SQ181: **patch-4 training phase, then fall back to P=1 (or smaller P)**. Costs one config, tests a mechanism, not a size. | med |
| SQ181 / SQ182 (fixed P) | Scratchpad Patching: fixed patches suffer "patch lag" (first byte of a patch is predicted from the previous patch's stale state) and larger p loses fast: p=16 MBPP 18.2 vs 26.3 byte (page); entropy-triggered scratchpads recover most of it. This is exactly our BLT-lite weakness noted in the first note (first byte predicted from latent alone). | In SQ181 decoder, give the local decoder access to the already-seen bytes of the current patch (it does) and **log per-position-in-patch loss** to measure the lag directly before adding any scratchpad. Scratchpad itself is a follow-up SQ, not part of SQ181. | med-high that lag is real; low that a 14M model needs SP |
| SQ184 (rule-based dynamic patches) | 2610.22100/2610.11790: entropy patching misses computed values; a two-byte lookup table or hand-written marker rule beats it on math at 50M from scratch (1M-50M, 128-byte context, so our scale!). SpaceByte-rule vs fixed was +6-12%. ByteFlow Net uses a **static graph with Top-K** boundaries. | (a) Use **Top-K per window** (fixed number of patches per row, variable boundaries) instead of padding or `cu_seqlens`: it keeps constant shapes for GDN chunk kernels and for loops, which removes most of SQ185's worry. (b) For code, test a rule keyed on `=`, `(`, newline, indent changes versus the SpaceByte rule. | med for Top-K idea (design logic, **[speculation]** on quality); the code-boundary rule is untested |
| SQ185 (fla varlen) | ByteFlow's static-graph Top-K sidesteps variable length. | Make SQ185 conditional: do it only if Top-K fails. Lower its priority. | med |
| SQ183 (looped core on patches) | Proxy: no. Closest: ByteFlow and H-Net put the expensive stack at patch level; One Byte One Token shows GDN hybrids tolerate pure bytes. No paper loops the core. | No change; this remains our novel combination. Note H-Net found Mamba-2 outer layers at byte level are what make byte models work, supporting a **GDN local encoder/decoder**. | med |
| SQ187 (robustness eval) | Tokenizer Myths: BPB is not tokenizer-agnostic. Our bits/char on the same chars (bytes/char 1.0005) is a **characters-denominator** metric and fine; but BPE-E1 vs byte-E1 on training-time loss are not comparable. | Always score in doc-level bits/char via the K8/K9 scorer; never compare raw training loss. Add one non-bpb downstream proxy (exact-match on identifier completion) for the byte-vs-BPE claim. | high |
| SQ188 (byte LR check) | SQ188 already planned; ATDC reports **curriculum on compression ratio stabilises** chunker training (H-Net ratio loss alpha 0.03 is fragile). | If dynamic patching is run, add a ratio-curriculum arm only after the fixed-P results; one ingredient at a time. | med |
| New (not in SQ180-189) | Bolmo / 27B GDN byteification: byte capability can be added by continued training on a trained model. | **[speculation]** Cheap test: take the trained BPE-E1 body, swap embedding/head for a 256-entry byte interface (Variant B local layers), continue-train briefly; compare to scratch BX2 at equal bytes. Needs to be scoped as a new side quest. | low |
| Compute accounting | Compute Optimal Tokenization: compute-optimal params scale with data in **bytes**, optimal compression decreases as compute grows. | At our tiny compute the implication is *higher* compression is fine (P=4-8 not P=2), contradicting my first-note guess that the best P is 2-4. Make SQ182's sweep {2,4,8,16}. | low-med; the paper's range is 50M-7B, not 14M |
| Ternary and bytes | None of the 32 papers combine ternary with bytes. 2610.05978 suggests up to 25% of intermediate layers can be local. | No change; local layers in bf16, ternarise the latent core. | high that this is a gap in the literature |

### Three findings that most change plans
1. **Patch-lag fix and per-position loss logging (Scratchpad Patching):** fixed-patch cost is mostly the first bytes of each patch.
2. **Static-graph Top-K boundaries (ByteFlow Net) make dynamic patching compatible with fixed-shape GDN kernels**, lowering SQ184/185 cost.
3. **Scale-threshold evidence (Efficiency Gap, Proxy Compression, Token Ceiling):** all say 14M / ~1e17 FLOPs is the least favourable place for bytes; the 4-byte warm-up/compression idea (2604.27263) is the cheapest rescue arm.

## 4. Caveats
- Page-level numbers marked (page) come from a summarising fetch tool over arXiv HTML; the first 100k characters only for the long papers.
- 2610.11790/2610.22100 are small, non-peer-reviewed, two related versions of one project; the second states the work is closed.
- Bolmo's summary line (+16.5 STEM over BLT) is from a search-result snippet of the paper, not the abstract; re-check.
