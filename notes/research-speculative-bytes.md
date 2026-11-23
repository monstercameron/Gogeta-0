# Speculative decoding across granularities (MambaByte "2.6x") and Gogeta-0

Research only. Nothing run. Labels: **reported** (a paper says it), **verified-local** (read in the PDF in `refs/14-byte-level/`), **computed** (arithmetic from stated inputs), **assumed** (an input I could not measure), **[speculation]**.
Sources read in full: `refs/14-byte-level/2401.13660 - MambaByte.pdf` (COLM 2024 version), `2610.05978` (Byte LMs, Table 3 and App. G), `2605.08044` (Fast BLT, abstract + grep), `2506.14123` (ByteSampler, abstract). Other papers: arXiv abstract pages only (fetched this session).

## 1. What MambaByte actually reports (arXiv 2401.13660, COLM 2024)

| Item | What the paper says | Where |
|---|---|---|
| Claim | "adaptation of speculative decoding with tokenized drafting and byte-level verification. This results in a 2.6x inference speedup to the standard MambaByte implementation, showing similar decoding efficiency as the subword Mamba" | Abstract |
| Drafter | **Mamba-110M** subword model, greedy decoding. The tokenizer is not named in the text I read. | Sec 5.3 "Generation via subword speculation" |
| Verifier | **MambaByte-972M**, context 8,192 bytes | Table 6 |
| Draft block | **3 subwords per iteration** (m = 3) | Table 6 caption |
| Acceptance rule | A drafted byte is accepted if it is **in the verifier's top-3 candidates** (tolerance nu = 3). Not argmax-equal, not rejection sampling. The bifurcation position c is the first byte outside the top-3. | Table 6 caption; Sec 3; Algorithm 1 (App. F) |
| Correction | Drafts after c are discarded. Because the drafter is subword, they cannot fix one byte and continue: the **verifier generates bytes autoregressively until a boundary byte (e.g. space)**, so the corrected partial subword is never out-of-vocabulary. The drafter then restarts from its saved hidden state at the last verified subword. | Sec 3; App. F |
| Verification kernel | Parallel scan of the drafted bytes by MambaByte; they **extended the Mamba CUDA kernel so the scan can resume from the mismatch position**. Hidden states at every byte are recorded (per-position state capture). | Sec 3 |
| Exact (lossless)? | **No.** Top-3 acceptance changes the distribution. They measure "faithfulness" as a log-odds ratio of the greedy MambaByte reference sequence under MambaByte vs the speculative output: **0.89** for speculative MambaByte vs **0.10** for subword Mamba-1.03B (1.0 = identical to greedy MambaByte). Approximately greedy-faithful, not provably identical. | Table 6, Sec 5.3 |
| Speedup | Table 6: MambaByte-972M **1x**; Mamba-1.03B subword (context 2,048) **2.8x**; MambaByte-972M + Mamba-110M speculation **2.6x**. Speed measured generating **8,192 bytes**, averaged over **100 prompts** built from common PG19 phrases. | Table 6, Sec 5.3 |
| Hardware / batch | Not stated for Table 6. Table 5 (generation-speed table) says A100 80GB PCIe. Batch size is never stated; **assumed batch 1** (recurrent, latency-bound decode). | Sec 5.3 |
| Conditions it holds / degrades | One configuration only (no sweep over m, nu, drafter size, domain). No acceptance rate or accepted-bytes-per-iteration is given. The ceiling is the subword model's own speed (2.8x); speculation reaches 2.6/2.8 = 93% of it. Domain is PG19 prose; nothing on code. | Table 6 |

**Is "2.6x" accurate?** Yes for the speculative claim, with caveats: (i) it is relative to the **plain MambaByte-972M recurrent decode**, not a Transformer; (ii) it is **lossy** (top-3 rule, 0.89 faithfulness), not the lossless speculative decoding of Leviathan et al.; (iii) the drafter is about 9x smaller (110M vs 972M, **computed** 8.8x), a cost ratio we do not have; (iv) one setting, no variance. **Trap:** Sec 5.3 also says MambaByte is "2.6x faster" than MegaByte in a parameter-matched setting (Table 5). That is a different 2.6x (MegaByte 93 s vs MambaByte-1.6B ~36 s by my reading of the garbled table text; MambaByte-972M is 29 s). Our `notes/research-byte-level.md` cites the speculative number correctly but should add "lossy, top-3, vs plain MambaByte, 110M drafter".

## 2. Follow-ups, 2024-2026 (arXiv ids verified on abs pages unless noted)

| arXiv id | What | Relevance |
|---|---|---|
| **2502.05202** (Timor et al., ICML'25 oral) | Lossless speculative decoding for **heterogeneous vocabularies**: SLEM (string-level exact match), TLI (token-level intersection), plus a third method; up to **2.8x** vs autoregressive; shipped in Hugging Face Transformers. Limitation: cost grows with longer-token drafters, "most suitable for drafters with shorter tokens". | The lossless counterpart of MambaByte's lossy rule. A BPE drafter with a byte verifier is the extreme heterogeneous case. |
| **2510.15545** TokenTiming (Xiao et al., ACL 2026) | Re-encodes draft tokens into target tokens, uses Dynamic Time Warping to map probabilities; **1.57x** vs prior SOTA SD. | Alternative alignment; token-level, not byte-level. |
| **2506.14123** ByteSampler (Hayase et al., v3 May 2026) | Inference-time method that **converts any BPE LM into a byte-level LM** (byte-level next-byte distribution), solving the prompt boundary problem; unifies vocabularies across tokenizers (ensembling, proxy-tuning). Already in our refs. | The route to q(byte given prefix) from our 8k-BPE drafter, which lossless rejection sampling against a byte verifier needs. Cost per byte is higher than one BPE step; measure. |
| **2610.05978** (Wang et al., Oct 2026; local PDF) | Byte vs subword Transformers as draft/target pairs, **teacher-forced oracle**, same-family 1B targets: byte drafts accept 90.6% (200M) / 84.6% (50M) vs subword 74.1% / 62.0%; accepted tokens per forward 9.527 vs 2.831 (200M) and 5.471 vs 1.618 (50M); accepted bytes per forward ~equal (9.527 vs 9.523; 5.471 vs 5.111) (Table 3). They warn it is a proxy, not generated histories (App. G). | Direct acceptance evidence at byte granularity, but byte->byte and subword->subword, **not** cross-granularity, at 50M-1B. Method to copy: teacher-forced consecutive-agreement oracle. |
| **2605.08044** Fast BLT (Kallini et al., ICML 2026) | BLT-D (block diffusion), **BLT-S self-speculation** (the light local decoder drafts bytes past patch boundaries, the full model verifies), BLT-DV. Abstract: >50% lower memory-bandwidth cost than BLT (an **estimate**, not wall-clock). | Patch-model analogue of cheap-draft/expensive-verify; relevant if SQ181 (BLT-lite) is built: the local decoder is a free drafter with no second tokenizer. |
| **2511.11346** MTPC (multi-byte prediction with probabilistic circuits) | Notes Hydra/EAGLE-style drafting may not transfer to byte LMs; multi-byte heads instead (EvaByte reports up to 2x vs tokenizer LMs). In refs. | Self-speculative alternative. |
| **2505.14969** STree (Wu et al., v2 Oct 2025) | First scalable **tree** verification for SSM and hybrid SSM-Transformer targets, via accumulated state-transition matrices. | Verification is not a plain parallel attention pass for SSMs; trees need this. We would use a linear chain: lower priority. |
| **2610.13778** ReplaySSM (Liou ... Tri Dao, Oct 2026) | Stores SSM state as a checkpoint plus a buffer of recent inputs; **speculative rollback becomes a pointer move**; implemented in vLLM for **Mamba-2 and Gated DeltaNet**; ~1.8x throughput over vLLM speculative decoding on two 120B-class models (batch 256, reported). | The most direct answer to GDN rollback. Large-batch regime, not our latency-bound batch 1. |
| **2408.15237** Mamba in the Llama (Wang et al.) | Hardware-aware speculative decoding algorithm and kernel for Mamba and hybrids (id from a search result; abs page not fetched, verify before adding). | Early SSM-target SD with state rollback. |
| Snakes and Ladders (Wu et al., NeurIPS 2024 ENLSP workshop; PMLR v262 wu24a) | Two backtracking solutions ("Joint Attainment and Advancement", "Activation Replay"); 6 tokens for 1.47x the cost of one; 1.82x average wall-clock (search-summary figures, not read). **No arXiv id found.** | Same problem as ReplaySSM, earlier. |
| **2506.01206** Mamba Drafters (Choi et al., Findings EMNLP 2025) | Mamba as the **drafter** for a Transformer target. | Opposite direction; our GDN drafter shares the constant-time property. |
| **2509.19873** SpecMamba (FPGA) | Hidden-state backtracking and tree-verification incompatibility named as key difficulties for SSM SD (search result only, abs page not fetched). | Background. |
| **2412.09871** BLT, **2507.07955** H-Net | Patch and dynamic-chunk models; BLT decode amortised over patches (already in refs). I have no speculative-decoding result to cite for H-Net. | Context. |

Not found: any paper doing subword-draft -> byte-verify beyond MambaByte, or byte-draft -> subword-verify with measured wall-clock. The lossless cross-granularity case with a recurrent verifier looks open (web search was standard mode only; treat "not found" as "not found quickly").

## 3. Fit to FusionDance

### 3.1 GDN rollback cost (SSM state vs KV)
- State size (**assumed** from `configs/e1_hybrid.json`: d=384, 6 heads, so head dim 64; GDN key/value dims taken as 64): 6 x 64 x 64 x 4 B = **~98 KB per GDN layer** in fp32, 3 GDN layers = ~295 KB per sequence, plus the short-conv tail. **Constant in context length**, unlike a KV cache. A snapshot copy is microseconds of bandwidth; in practice launch-bound.
- Three ways to roll back after verifying n drafted tokens and accepting a:
  1. **Snapshot + replay the accepted prefix** (Snakes and Ladders "Activation Replay"; ReplaySSM's idea): keep the pre-verify state, run the chunk verify over n tokens for logits only, then recompute the state over the accepted a tokens (chunk or `fused_recurrent`, T=a). Cost: one extra pass of the GDN layers over a tokens, **about +0.3-0.5 of a verify pass** (3 of 6 layers are GDN; **assumed**, to be measured in SQ198).
  2. **Per-token state capture** (MambaByte's way: hidden state saved at every byte, scan resumes at the mismatch): needs the verify kernel to emit per-position states. fla's chunk kernel exposes only chunk-boundary and final states, so this needs a custom or `fused_recurrent` path. Memory n x 295 KB: n = 15 bytes is ~4.4 MB (**computed**), fine.
  3. **Verify with `fused_recurrent` over T=n** emitting states: serial inside one launch. Decode step is 0.036 ms (C2/C111, B=16, H=6, T=1, kernel only, not a whole-model step); at n ~ 11-15 that is ~0.4-0.5 ms per GDN layer **if it scaled linearly (assumed)**, so option 1 is better for large n.
- **SWA (window 256) and full attention** need a KV cache that does not exist yet (SQ53; only the GDN decode path is unblocked by C111). KV rollback is free (move the length pointer) once it exists. Offline acceptance measurement does not need a cache (teacher-forced full forward).

### 3.2 Inputs for the arithmetic
- bytes/token **3.735** (`notes/research-byte-level.md`, measured on our corpus with the 8k BPE).
- GDN decode step 0.036 ms (C111, kernel only). Whole-model step latency is **not measured**; for a 14M model at batch 1 I **assume** a step is launch-latency bound, so per-step time is flat up to ~16 tokens per pass. That is the only regime where speculation can win.
- E1 training ~100k tok/s steady (`experiments/cards/E1.md`: 38.6 min, 1.82e16 FLOPs): compute-bound, ~10 us/token. It says a batch of tokens costs ~10 us each, so a single-sequence decode step of hundreds of us leaves >95% of the GPU idle, which is the headroom speculation exploits. It does not give decode latency.
- Drafter/verifier per-step cost ratio c: byte-E1 and BPE-E1 share blocks (d=384, 6 layers, ~9M non-embedding params); BPE adds an 8,192 x 384 head (3.1M) vs 256 x 384 for bytes. Compute-bound c = (9+3.1)/(9+0.1) = **~1.3**; latency-bound c = **~1.0** (**computed from params, then assumed**). A dedicated smaller drafter might reach c ~0.5 (**assumed**; launch floors apply).
- E2: layout [SWA][GDN,GDN,SWA][attn], core of 3 blocks looped; r=1 is 1+3+1 = 5 block-passes, r=4 is 1+12+1 = 14 (**computed**; blocks treated as equal cost, **assumed**), so c = 5/14 = 0.36. E2 val loss r1 0.6694 vs r4 0.6186 (E2 card): r=1 is **8% worse** and was never trained to exit.

### 3.3 Theoretical speedup
Standard formula (Leviathan et al. 2023, 2211.17192): acceptance alpha, draft length gamma, cost ratio c, one verifier step per verify: expected tokens per iteration (1 - alpha^(gamma+1))/(1 - alpha); speedup = that / (gamma c + 1). Same-size, same-vocabulary: with c = 1 and alpha 0.8 it is 0.81x, 0.74x, 0.67x at gamma 2, 3, 4: a same-size drafter never pays.

Cross-granularity version used below (**my adaptation**): bytes per iteration = 3.735 x sum_{i=1..gamma} alpha^i + 1 (the bonus/corrected byte). Cost per iteration in units of one byte-verifier step = gamma c (draft) + (1 + 0.02 n) (one chunked verify, n = 3.735 gamma bytes; 0.02 per byte is **assumed**) + P(reject) x (0.5 state replay + c drafter resync). Baseline is 1 byte per unit. alpha is **per BPE token (all its bytes must match)**. The script is a ten-line Python function; rerun in SQ198.

**Scheme A: BPE-E1 drafts, byte-E1 verifies** (latency-bound, c = 1.0):

| alpha \ gamma | 2 | 3 | 4 | 6 |
|---|---|---|---|---|
| 0.50 | 0.89 | 0.77 | 0.67 | 0.52 |
| 0.65 | 1.25 | 1.14 | 1.03 | 0.84 |
| 0.80 | 1.73 | 1.67 | 1.59 | 1.41 |
| 0.90 | 2.15 | 2.18 | 2.16 | 2.06 |

With a cheaper drafter (c = 0.5): alpha 0.65 gives 1.84/1.75/1.63 and alpha 0.8 gives 2.54/2.58/2.53 (gamma 2/3/4); alpha 0.9 reaches ~3.4. In the **compute-bound** regime (large batch, verification costs n steps) the same arithmetic gives **0.2-0.7x, always a slowdown**: speculation only converts idle latency into work.
Reading it: the byte model's 3.735x extra steps is the headroom, but a same-size drafter spends a full step per draft token, so alpha must exceed ~0.7 just to break even at c=1. MambaByte had c ~0.11, which is why it reached 2.6x. **Sanity check:** my model with c = 0.11, 4.1 bytes/token (PG19), gamma = 3 gives 4.8x at alpha 0.8, versus the paper's 2.6x; so either its per-token alpha is much lower (~0.5-0.6), or real overheads (kernel resume, boundary correction) matter. The paper reports neither alpha nor overheads, so I cannot separate them.

**Scheme B: looped r=1 drafts, r=4 verifies** (same vocabulary; units of one block-pass B; AR r=4 costs 14 B per token). Draft step 5 B serial; the verify pass over gamma tokens runs loops 2-4 and the coda in parallel = 10 B, reusing the drafter's loop-1 work. (Loop-1 states depend only on loop-1 states of earlier tokens, so they are valid for the verifier: **my reasoning**; C41's cache-hole concern is about skipped deeper states for later tokens and does not apply if every accepted token gets loops 2-4 in the verify pass; to confirm in implementation.) Speedup = 14 (1 + sum alpha^i) / (5 gamma + 10 + 3 P(reject)):

| alpha \ gamma | 2 | 3 | 4 | 6 |
|---|---|---|---|---|
| 0.50 | 1.10 | 0.95 | 0.83 | 0.65 |
| 0.65 | 1.34 | 1.21 | 1.09 | 0.89 |
| 0.80 | 1.62 | 1.56 | 1.48 | 1.31 |
| 0.90 | 1.84 | 1.87 | 1.85 | 1.76 |

Same shape: gamma 2-3 is best, and the compute-bound regime loses.

### 3.4 The two schemes compared
| | A: BPE drafter -> byte verifier | B: r=1 drafter -> r=4 verifier |
|---|---|---|
| What speculation buys | Hides the 3.735x step overhead of a byte model | Hides the 2.8x loop overhead (14 vs 5 block-passes) |
| Needs | A byte model (SQ180/181 unrun; blocked on flex SWA, P3), BPE<->byte alignment, correction-to-boundary logic, drafter re-sync, ByteSampler or a lossy top-k rule | E2 + a decode path (SQ53) |
| Exact? | Lossy (MambaByte top-3), or lossless via exact-match or ByteSampler probabilities (extra cost, lower acceptance) | Lossless with standard rejection sampling (same vocab) |
| Draft-quality problem | Two independent 13M models must agree on every byte of a token | Same weights, but r=1 is 8% worse in loss and untrained for exit; K6 shows state norm grows ~4.6-5x per loop, so r=1 states may be off-distribution for what r=4 expects |
| Best case | 2.2x (alpha 0.9, c=1); ~3.4x with a c=0.5 drafter | 1.9x (alpha 0.9); 1.5-1.6x at alpha 0.8 |
| Realistic at 13M | 1.0-1.7x if alpha 0.65-0.8 | 1.1-1.5x |
| Payoff | Only if a byte model is the product | Main FusionDance path (E2/E4/E5/E8) |

Recommendation: B is cheaper, lossless and on the main path; do its offline measurement (SQ98, extended as SQ196) first. A's measurement waits for a byte model. [speculation] A BLT-lite local decoder (SQ181) as a free drafter, as in Fast BLT's BLT-S, avoids the second tokenizer.

### 3.5 Expected acceptance at our scale (predictions, not measurements)
- Reference points: 2610.05978 Table 3 (large 1B targets with 50-200M drafts): byte-token acceptance 85-91%, subword-token 62-74%, accepted bytes per forward ~5-9.5. Our pair is same-size and tiny: no capacity gap, and both models are far from the entropy floor, so expect lower agreement than a 200M -> 1B pair.
- Scheme A per-BPE-token alpha (greedy, all ~3.7 bytes match): **predicted 0.55-0.75** (code, E1 loss 0.58 nats/token; most bytes inside a token are forced, only the first byte of a token is uncertain). With a top-3 byte rule higher but lossy.
- Scheme B per-token alpha (greedy r=1 vs r=4): **predicted 0.65-0.85**.
- Resulting wall-clock (computed from the tables at gamma 2-3): A ~1.0-1.4x, B ~1.2-1.5x; far from MambaByte's 2.6x because c is ~1.0 / 0.36 instead of ~0.11.

## 4. Risks
1. **No whole-model decode latency exists** (SQ53). All speedups assume verify of n tokens costs about one step, true only if latency-bound. In batched decoding speculation loses (3.3).
2. **Re-tokenization at the boundary**: BPE merges can change earlier tokens when a corrected byte is appended, so the drafter state must roll back to a stable boundary and re-ingest. MambaByte's "generate to a space" is a poor fit for code identifiers and indentation [speculation].
3. **Lossless vs lossy**: MambaByte's top-3 is not exact. Lossless needs byte-level q from the BPE drafter (ByteSampler cost) or exact-match verification, which lowers acceptance.
4. **Kernel interface**: fla chunk returns only the final state, so per-token state needs replay or a custom kernel. Neg-eig GDN (E11) and bf16 (error ~4e-4, C110) add precision concerns for state replay.
5. **Our val set is 41.7% near-dup of train (K9)**, which inflates agreement; measure on K9-clean val.
6. **The byte model does not exist**; SQ180/181 are blocked on flex SWA (P3).
7. E2's r=1 is not trained for exit; real agreement may be low, pushing toward exit-aware or consistency training (SQ17), a different experiment.

## 5. Cheapest experiment (designed, not run)

**SQ196 (offline acceptance oracle, scheme B first).** SQ98 (E8 premise) extended. How it differs from SQ98: SQ98 reports greedy argmax agreement and the mean agreeing run in blocks of 4 for r=1/2 vs r=4. SQ196 also reports (a) **expected rejection-sampling acceptance, sum over x of min(p_r4(x), p_r1(x))** per position, the lossless quantity the speedup formula needs; (b) the full consecutive-agreement length distribution rather than blocks of 4; (c) the speedup computed from the tables above with explicit c = 5/14; (d) strata by token type and by r=4 entropy, to see whether an entropy-gated draft (E5) beats a fixed one.
- Method: E2 checkpoint, teacher-forced on K9-clean val, 2k docs of 1,024 tokens; forwards at r=1 and r=4 (no KV cache); protocol of 2610.05978 App. G.2 (teacher forcing, each round ends at the first mismatch).
- **Prediction (register before running):** greedy per-token agreement r=1 vs r=4 **0.65-0.85**; sampling acceptance **0.55-0.75** at T=1; mean consecutive-agreement run **2.5-5 tokens**; computed speedup at gamma 3, c = 0.36: **1.2-1.55x**.
- **Criterion:** build E8 (needs SQ53) if the computed speedup is **>= 1.3x** at gamma <= 3 under the latency-bound assumption; refuted if greedy agreement < 0.60 (speedup < ~1.15x), in which case the fix is exit-aware training (SQ17), not decoding.
- Cost ~2-3 GPU-min.

**SQ197 (scheme A, after a byte model exists).** Same oracle between BPE-E1 and byte-E1, aligned by string: for each BPE token of the reference text, does the byte model's greedy (and top-3) byte sequence equal the token's bytes? Prediction alpha 0.55-0.75 greedy, 0.75-0.9 top-3. Criterion: at c=1, gamma 3, 1.3x needs alpha >= ~0.8; below ~0.7 a same-size drafter is dead and only a smaller drafter (c <= 0.5) or a BLT-S-style local drafter is worth trying. Blocked on SQ180.

**SQ198 (decode-regime microbench, no trained model).** Measure the three numbers every line above assumed: whole-model decode step latency (E1 shape, GDN via `fused_recurrent`, SWA/attention via a stub or the SQ53 cache), chunked verify of n in {4, 8, 16}, GDN state replay for a in {1..16}. Prediction: step 0.4-1.0 ms eager; verify(16)/step < 1.5; replay <= 0.5 of verify. Criterion: if verify(16)/step > 3, speculation is dead at batch 1 whatever the acceptance.

**SQ199 (lossless byte distribution from BPE).** ByteSampler (2506.14123) on E1: check exactness on 1k prefixes and measure per-byte cost vs a BPE step. Decides lossless vs MambaByte-style lossy verification for scheme A.

## 6. Papers to add to refs (verified arXiv ids)
- 2502.05202 Lossless speculative decoding for heterogeneous vocabularies (SLEM/TLI), ICML'25.
- 2510.15545 TokenTiming: universal speculative decoding via dynamic token alignment, ACL 2026.
- 2610.13778 ReplaySSM: cache SSM inputs not state; Mamba-2 and Gated DeltaNet speculative rollback.
- 2505.14969 STree: speculative tree decoding for hybrid SSMs.
- 2506.01206 Mamba Drafters for speculative decoding.
- 2408.15237 Mamba in the Llama (id from search results; abs page not fetched, verify).
- 2509.19873 SpecMamba (FPGA, state backtracking; id from search results, abs page not fetched).
- Already in refs and relevant: 2401.13660, 2506.14123, 2610.05978, 2605.08044, 2511.11346, 2211.17192 (Leviathan), 2404.16710 (LayerSkip, the self-speculative early-exit precedent for E8).
- No arXiv id: Snakes and Ladders (NeurIPS 2024 ENLSP workshop, PMLR v262 wu24a).
