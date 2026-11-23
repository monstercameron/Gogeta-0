# Final model plan: Gogeta on a 24 GB RTX 3090 / 4090

*Planning note, 2026-11-22. Asked for by the human: "lets start making a plan for the final model size, I want inference
with caching, spec decode etc on a 3090/4090 with 24GB ram, add that to the project goals so we can get there at the end
of these experiments". Nothing here was run on a GPU. Arithmetic script (CPU only, not committed):
`%TEMP%/claude/.../scratchpad/finalplan/calc.py`; its formulas are restated in each table.*

**Labels (AGENTS §0):** **(m)** measured, with the source path · **(c)** computed, formula shown · **(a)** assumed ·
**(r)** reported by a vendor or paper, not verified here. Every size, speed and $ figure below is a plan, not a result.

**Bottom line.**
1. VRAM is not the binding constraint. 24 GB holds ~9B params in bf16 (c). Everything we can afford to train fits with room for >1M tokens of cache.
2. Training compute, money and unique tokens are the binding constraints. A 1.2B model at ~20-27 tokens/param costs ~$370-500 of 5090 time (c). A 3.3B model costs ~$2.8k and needs more permissive Python than codeparrot has (c). A 7B model costs ~$12.6k (c).
3. **Primary target: ~1.2B params ("G-1B").** **Stretch target: ~3.3B ("G-3B").** Both are hybrid E1-pattern stacks; whether they include loops, ternary weights or MoE is decided by the gates in §6.

---

## 1. Hard target

| Item | RTX 3090 | RTX 4090 | Source |
|---|---|---|---|
| Arch / CUDA capability | Ampere, sm_86 | Ada, sm_89 | (r) vendor |
| DRAM bandwidth | ~936 GB/s | ~1,008 GB/s | (r), as quoted in `notes/compute-rental.md` |
| L2 cache | 6 MB | 72 MB | (r) vendor |
| FP8 tensor cores | no | yes | (r) vendor. bf16, int8 and int4 tensor ops exist on both (r) |
| VRAM | 24 GiB = 25.8e9 B | same | (r) |
| **Usable budget** | **~21.5 GB** | ~21.5 GB | (a): 25.8 GB minus CUDA context (~0.5 GB), a desktop on the same card (0.5-1.5 GB, more under Windows WDDM), allocator fragmentation (~8%) and a 1 GB margin. SQ289 measures it |
| Rental price | $0.11 / 0.20 /h (from / median) | $0.28 / 0.47 /h | (m) `notes/compute-rental.md` (Vast listing) |

**Workload:** batch 1, interactive; one prompt prefilled in chunks, then token-by-token decode. It needs a KV cache (full attention), a ring buffer (SWA), the recurrent state (GDN), lossless speculative decoding with rollback, and CUDA graphs.

**Kernel compatibility is a gate (SQ285):**

| Path | N1X sm_121 | 5090 sm_120 | 3090 sm_86 | 4090 sm_89 |
|---|---|---|---|---|
| fla chunk GDN fwd+bwd, `torch.compile` (training/prefill) | yes (m) | yes (m) Phase 2b, `experiments/devlogs/PHASE2B-5090.md` | **yes (m)**: SQ220 trained E1/E2 on a rented 3090, loss within 0.002 of N1X (`experiments/devlogs/SQ220.md`, C195) | **untested** |
| fla `fused_recurrent` GDN decode | yes (m) C2/C111, `experiments/checks/results/sq_hw_fla_recurrent.json` | untested | untested | untested |
| head_dim 128 (the final shapes below) | untested: every config run so far uses hd 64 (E1/E2/E3, E32, E34, SQ243) | untested | untested | untested |
| SDPA flash backend at long T, ternary/int4 GEMV kernels | none exist in the repo | | | |

Correction to the brief: sm_86 is **not** untested for training; SQ220 ran the chunk kernels there. Decode, hd 128 and sm_89 are untested. sm_86 has less shared memory per SM than Hopper (r), so the Triton autotune configs at hd 128 are the most likely failure.

## 2. Memory envelope

### 2.1 Candidate shapes
Params are computed with `gogeta/model.py`'s formulas:
- SwiGLU: 3·d·m.
- GDN: 5·d·d (q, k, v, gate, o) + 2·d·H + convs.
- Attention: 2·d² + 2·d·kv·hd (with GQA, a new option: model.py has none today).
- Re-injection: 2d².
- Embedding: tied, 8192·d.

All shapes use hd = 128 and the E1 pattern `[swa, gdn, gdn, swa, gdn, attn]`.

| Shape | Layout | d / heads / kv heads / mlp | Unique params (c) | Params applied per token (c) | Layers: attn / swa / gdn |
|---|---|---|---|---|---|
| **S** pilot / drafter | E1 pattern × 4 = 24 blocks, unlooped | 1024 / 8 / 2 / 2816 | 0.311B | 0.311B | 4 / 8 / 12 |
| **P** primary | E1 pattern × 4 = 24 blocks, unlooped | 2048 / 16 / 4 / 5632 | **1.226B** | 1.226B | 4 / 8 / 12 |
| **X** stretch | E1 pattern × 5 = 30 blocks, unlooped | 3072 / 24 / 4 / 8192 | **3.331B** | 3.331B | 5 / 10 / 15 |
| **PL** looped P | prelude [swa, gdn, gdn] → core [gdn, gdn, swa, gdn, gdn, swa] × 4 → coda [gdn, swa, attn] | 2048 / 16 / 4 / 5632 | 0.640B (core 0.321B) | 1.604B (= outer + 4·core) | 1 / 10 / 19 (per token, r = 4) |
| **PL2** L2-core variant | [swa, gdn, gdn, attn] → [gdn, gdn, swa] × 4 → [gdn, swa, gdn, attn] | 2048 / 16 / 4 / 5632 | 0.585B (core 0.165B) | 1.079B | 2 / 6 / 12 |

**Design rule (c):** never put full attention inside the looped core. Each pass sees different inputs, so every core layer needs its own cache **per pass**. A core full-attention layer would multiply the growing KV cache by r; SWA and GDN only multiply constant-size state.

### 2.2 Cache per token and fixed state

All caches are bf16 KV and fp32 GDN state:
- **KV per token** = n_attn · 2 · kv · hd · 2 B.
- **SWA** = n_swa · 2 · kv · hd · 2 B · window.
- **GDN** = n_gdn · H · hd² · 4 B.

| Shape | KV per token (c) | KV at 32k / 128k (c) | SWA at w = 256 / 4096 (c) | GDN state (c) | GDN per-position checkpoints for rollback, γ = 4 / 8 (c) |
|---|---|---|---|---|---|
| S | 4.0 KiB | 0.13 / 0.54 GB | 2.1 / 34 MB | 6.3 MB | 25 / 50 MB |
| P | 8.0 KiB | 0.27 / 1.07 GB | 4.2 / 67 MB | 12.6 MB | 50 / 101 MB |
| X | 10.0 KiB | 0.34 / 1.34 GB | 5.2 / 84 MB | 23.6 MB | 94 / 189 MB |
| PL (r = 4) | 2.0 KiB | 0.07 / 0.27 GB | 5.2 / 84 MB | 19.9 MB | 80 / 159 MB |

- **Without GQA** (kv = 16), P's KV is 32 KiB/token, i.e. 4.3 GB at 128k (c). That still fits. GQA is a decode-**speed** lever, not a fit lever: at 128k a decode step reads all of the KV, so P bf16 on a 3090 drops from ~213 to ~111 tok/s (c: 0.8 · 936 / (2.45 + 1.07 or 4.29) GB). It must be ablated (SQ293) before adoption (one ingredient at a time).
- **Short-conv tails:** 3 convs × 3 past inputs × d per GDN layer. Kilobytes; also rolled back.
- **Draft cost:**
  - self-speculation adds no weights;
  - a separate drafter S adds 0.62 GB in bf16;
  - MTP heads add ~1 block each (~55M for P, 0.1 GB).
- **Activations and workspace** at batch 1 decode are under 100 MB. Chunked prefill (2k-token chunks) needs logits of 2048 × 8192 × 4 B = 67 MB (c) plus hidden states. CUDA-graph pools and cuBLAS workspace: **1 GB budgeted (a)**.

### 2.3 What fits in ~21.5 GB

Weight bytes per param, with the embedding and head kept bf16:
- bf16: 2;
- int8 (per-channel): ~1.01;
- int4 (group 128, fp16 scales): ~0.52;
- ternary packed 2-bit: ~0.27 (model.py's BitLinear uses one absmean scale per matrix, so ~0.25).

| | bf16 | int8 | int4 | ternary 2-bit |
|---|---|---|---|---|
| Max params with 18 GB for weights (c, 18e9 / bytes per param) | 9.0B | 17.8B | 34.9B | 67.8B |
| P weights (c) | 2.45 GB | 1.25 GB | 0.66 GB | 0.35 GB |
| X weights (c) | 6.66 GB | 3.39 GB | 1.75 GB | 0.93 GB |
| PL weights (c) | 1.28 GB | 0.66 GB | 0.36 GB | 0.20 GB |
| Context that fits after weights, SWA at 4096, GDN state and 1 GB workspace (c) | P: ~2.2M tokens; X: ~1.3M | P: 2.3M | P: 2.4M | P: 2.4M |

**32k and 128k fit at every size and precision with >15 GB to spare (c).** The real context limit is what we train and evaluate, not VRAM:
- RoPE is precomputed to `max_seq_len = 1024`;
- SWA uses a dense T×T mask (SQ3/SQ275);
- `Rotary.forward` has no position offset.

## 3. The binding constraint: training compute and data (proved with numbers)

**Inputs:**
- **Throughput (m):** E1 on an RTX 5090 runs at 743k tok/s steady (`experiments/devlogs/SQ220.md`). At 9.1e7 FLOP/token (E1 card: 1.82e16 FLOP / 200M tokens, `experiments/cards/E1.md`), that is **6.76e13 FLOP/s achieved** (c), about 32% MFU against a ~209 TFLOPS bf16 peak (r, vendor; unverified).
- **Price (m):** $0.523/h for a 5090 (SQ220), so **$2.15 per 1e18 FLOP** (c).
- **Assumption (a):** the 14M-scale MFU transfers. Bigger matmuls usually raise MFU, but multi-GPU communication over PCIe lowers it, so treat the figures as ±50%.
- **Data:**
  - code-py-v2 has 0.2B train tokens (m, `notes/data-v2.md`);
  - codeparrot-clean has ~8B permissive tokens (c, extrapolated from partial stats in `notes/research-corpus-v3.md`). It may be ~8-20% more with our tokenizer on real code, since 3.0-3.4 chars/token is predicted vs 3.7;
  - Stack v3 Python is 154B tokens upstream (r, card), but its permissive share is unknown until SQ276;
  - repeating up to ~4 epochs costs little, as reported by Muennighoff et al. 2023 (r; not verified here, SQ292).

**Training cost** = 6·N·D FLOP ÷ 6.76e13 FLOP/s × $0.523/h. The wall-clock column assumes one 8-GPU host with ideal scaling (a).

| N | D (tokens/param) | FLOP (c) | 5090-hours (c) | $ (c) | Wall on 8 GPUs (c) | Data source that supplies D |
|---|---|---|---|---|---|---|
| 125M | 2.5B (20) | 1.9e18 | 8 | **4** | 1 h | codeparrot (1 pass) |
| 125M | 12.5B (100) | 9.4e18 | 39 | 20 | 5 h | codeparrot × 1.6 |
| 350M | 7B (20) | 1.5e19 | 60 | 32 | 8 h | codeparrot |
| 350M | 32B (91) | 6.7e19 | 276 | 144 | 1.4 d | codeparrot × 4 |
| **1.2B** | **24B (20)** | 1.7e20 | 710 | **371** | 3.7 d | codeparrot × 3 |
| **1.2B** | **32B (27)** | 2.3e20 | 947 | **495** | 4.9 d | codeparrot × 4 (the ceiling) |
| 1.5B | 30B (20) | 2.7e20 | 1,109 | 580 | 5.8 d | codeparrot × 3.75 |
| **3.3B** | **66B (20)** | 1.3e21 | 5,369 | **2,808** | 28 d | **Stack v3 only** (codeparrot × 4 = 32B is too few) |
| 3.3B | 32B (10) | 6.3e20 | 2,603 | 1,361 | 13.6 d | codeparrot × 4, under-trained |
| 7B | 140B (20) | 5.9e21 | 24,157 | **12,634** | 126 d | Stack v3, if enough of it is permissive |
| 7B | 32B (5) | 1.3e21 | 5,522 | 2,888 | 29 d | badly under-trained |

**Training memory (a; standard AdamW mixed-precision accounting):** about 16 B/param of states.
- 1.2B needs ~20 GB, which fits on one 32 GB 5090 with activation checkpointing (SQ297 measures it).
- 3.3B needs ~53 GB, which needs ZeRO/FSDP across 5090s over PCIe, or an 80 GB A100/H100 at a median $0.65-2.21/h (m, `notes/compute-rental.md`). Their throughput is unmeasured.

**Verdict: the claim holds (c).**
- **VRAM:** the 24 GB card could serve a 9B bf16 or ~35B int4 model. We cannot train even a 7B: it is $12.6k at our measured rate (420× the $30 standing approval in AGENTS §0h), 126 GPU-days on 8 boxes, and more permissive Python than our known sources hold.
- **Data:** at 20 tokens/param, codeparrot × 4 caps us at ~1.6B params. Beyond that we need Stack v3 (SQ276).
- **Money:** both recommended sizes need a new budget approval from the human. The primary needs ~$400-650 including pilots and overhead; the stretch needs ~$3k+.

### 3.1 Loops: unique params vs effective depth

**Training cost per unique core param per token (c):**
- the forward pass runs E[r] = 4 times for r ~ U{1..7};
- the backward pass runs E[min(r, 4)] = 22/7 = 3.14 times (bptt 4);
- so the cost is 2·4 + 4·3.14 = **20.6 FLOP, vs 6 unlooped: 3.4× per unique param**.

| | P (unlooped) | PL (looped, r = 4) | PL2 (looped, small core) |
|---|---|---|---|
| Unique params (c) | 1.226B | 0.640B | 0.585B |
| Params applied per token, i.e. effective depth × width (c) | 1.226B | 1.604B (2.5× unique) | 1.079B (1.8×) |
| Train FLOP/token (c) | 7.4 G | 8.5 G | 5.9 G |
| Cost at 24B tokens (c) | $371 | $440 | $305 |
| bf16 weights (c) | 2.45 GB | 1.28 GB | 1.17 GB |
| Applied params per weight byte: bf16 / ternary (c) | 0.5 / 3.5 per B | 1.25 / 8.0 per B | 0.92 / 6.0 per B |

**What this says:**
- Loops and ternary buy far more effective depth per VRAM byte: up to ~16× P-bf16's applied params per byte (c, 8.0 vs 0.5).
- **But VRAM is the resource we have in excess.** Loops cost more training FLOPs per unique param, and so far they lose at matched FLOPs: C6 and C7 are PARTIAL [exploratory n=1], E2 is 2.6% worse than E9 (PHASE2B-5090), and C197's unique-params lever is PARTIAL n=1, confounded per v0.4.84.
- **Loops and ternary earn a place only if they win on quality per training FLOP, or on decode tok/s at equal quality.** Saving memory alone is not a reason.

## 4. Recommended targets

**Decode model:**
- **Bandwidth-bound tok/s (c)** = 0.8 × bandwidth ÷ bytes per token. The 0.8 efficiency is assumed. Bytes per token = outer weights + r × core weights + LM head (bf16) + context × KV per token + SWA ring + 2 × GDN state.
- **Launch floor (a):** ~10 kernels per block. At ~8 µs each in eager mode, a 24-block model takes ~1.9 ms per token (~520 tok/s cap). With CUDA graphs at ~3 µs each, it takes ~0.7 ms (~1,400 tok/s cap).
- **Realistic tok/s** = min(bandwidth, launch floor). SQ288 calibrates both.

| Shape, precision | 3090 @4k / @32k (c) | 4090 @4k / @32k (c) | Realistic with CUDA graphs (a) |
|---|---|---|---|
| P bf16 | 298 / 272 | 321 / 293 | ~250-320 |
| P int8 weight-only | 568 / 482 | 612 / 519 | ~450-600 |
| P int4 g128 | 1,040 / 784 | 1,120 / 845 | ~700-1,000 (launch floor starts to bind) |
| P ternary 2-bit | 1,793 / 1,148 | 1,931 / 1,236 | ~1,000-1,400 (floor-bound; needs fused kernels) |
| X bf16 / int8 / int4 | 111 / 215 / 405 at 4k | 119 / 231 / 436 at 4k | as computed |
| PL (r = 4) bf16 / ternary | 230 / 1,472 at 4k | 247 / 1,585 at 4k | ternary floor-bound (30 layer passes) |
| PL2 ternary, core in 4090 L2 | n/a (6 MB L2) | 2,235 from DRAM; **3,514 if the 44 MB core stays L2-resident** (c, core read once from DRAM per token) | **hypothesis**: needs persisting-L2 control on GeForce (unverified) **and** a persistent/megakernel decode, since 3.5k tok/s is ~0.28 ms per 20-layer × 4-pass token, far below the launch floor. SQ295 |

**Prefill:** 32k prompt tokens × 2 × 1.226e9 = 8.0e13 FLOP. At an assumed 40 TFLOP/s achieved, that is about 2 s on either card (c/a).

**Speculative speedup (c):** speedup = (1 − α^(γ+1)) / (1 − α) ÷ (γ·c + 1 + verify overhead), where α is the acceptance rate, γ the draft length, and c the cost ratio. Formula from Leviathan et al. 2023, as in `notes/research-speculative-bytes.md`.

| Draft scheme | c (c) | α 0.6 / 0.7 / 0.8 at best γ (c) |
|---|---|---|
| Separate drafter S for a bf16 P target (c ≈ 0.62/2.45 GB by bytes, raised by S's launch floor) | ~0.15 | 1.50 / 1.75 / 2.11 (γ 3-5) |
| MTP heads (1 block each) | ~0.05 | 1.91 / 2.35 / 2.95 (γ 5) |
| Self-spec on PL, r_d = 1 → r = 4, reusing pass-1 caches (verify reads 3 core passes + coda = 0.70 of an AR step) | 0.40 | 1.31 / 1.46 / 1.63 (γ 2) |
| Self-spec on PL, r_d = 2 | 0.60 | 1.15 / 1.29 / 1.44 (γ 2) |

- Quantization and speculation compete for the same headroom: an int4 or ternary target is already near the launch floor, so a drafter's c grows toward ~0.3-0.5 and the gain shrinks (c from the table: c 0.3 gives 1.15-1.55×).
- MoE (E60) helps the fit: batch-1 decode reads only the active experts. Verifying γ tokens reads the union of their experts, which erodes the spec gain (a).

### 4.1 Primary: **G-1B** (shape P)

- **Layout:** 24 blocks in the E1 pattern (12 GDN, 8 SWA, 4 full attention); d 2048, 16 heads of 128; GQA with 4 KV heads on the attention and SWA layers; SwiGLU 5632; BPE 8192, tied (or the SQ277 tokenizer).
- **SWA window:** 256, or the window SQ214/SQ294 pick at this scale.
- **Context:** trained at 8k, extended to 32k.
- **Training:** bf16, 24-32B tokens (codeparrot permissive × 3-4 epochs, or Stack v3), **~$371-495 of compute (c) plus ~30% for the pilot, evals and restarts (a): ~$500-650**, or ~4-5 days on 8 × 5090.
- **Deployment:**
  - int8 weight-only by default (~450-600 tok/s on a 3090 or 4090);
  - bf16 as the reference (~300 tok/s);
  - spec decode with S as the drafter: ~1.5-2× on bf16 (a, α unmeasured).
  - VRAM use: ~2-4 GB at 32k, so 128k is affordable.
- **Rationale:** the largest size the realistic corpus supports at ≥20 tokens/param within 4 epochs, trainable on single 32 GB GPUs without sharding, at a few hundred dollars.
- **Variants, if their gates pass (§6):**
  - **PL:** loops, at ~the same training cost, with self-spec;
  - **ternary QAT:** about 1.7-2× faster decode;
  - **MoE FFN:** spends spare VRAM to buy quality at equal active FLOPs.

### 4.2 Stretch: **G-3B** (shape X)

- **Layout:** 30 blocks in the E1 pattern; d 3072, 24 heads, 4 KV heads; SwiGLU 8192.
- **Training:** **66B tokens, ~$2.8k (c)**; Stack v3 is required.
- **Hardware:** ZeRO/FSDP or 80 GB cards, with throughput unmeasured.
- **Deployment:** int8 ~200-230 tok/s, int4 ~400 tok/s, bf16 ~110-120 tok/s (c); P or S as the drafter.
- **Do it only if:**
  - M2's scaling fit predicts a meaningful gain over G-1B on our eval battery;
  - the human approves the budget.

## 5. Inference stack plan

| Component | Plan | Status / evidence |
|---|---|---|
| Full-attention KV cache | Preallocated [layers, 2, kv, T_max, hd] in bf16; `Rotary` gets a position offset; for verify, SDPA needs an explicit bottom-right causal mask (PyTorch's `is_causal` aligns top-left when q_len < kv_len (r, PyTorch docs; check in SQ286)) | Not built (SQ53 idea) |
| SWA cache | Ring buffer of `window` entries per SWA layer (per pass if looped); rollback = pointer move; window ≥ γ + 1 | Not built |
| GDN decode | `fused_recurrent_gated_delta_rule` with keyword args, fp32 state, plus short-conv tail buffers | **C2/C111 VERIFIED on sm_121 only** (`sq_hw_fla_recurrent.json`, bf16 err 3.96e-4) |
| Prefill | fla chunk kernels with `output_final_state`, 2-8k chunks; flex/block-sparse SWA (SQ3) instead of the dense mask | Chunk kernel verified for training on sm_86/120/121 (m) |
| CUDA graphs / `torch.compile` | Capture the decode step (static shapes; γ+1-token verify graph per γ); compile prefill | Compile works for training on 3090/5090 (m, SQ220); decode graphs untested |
| Quantization | 1) int8 weight-only per-channel (torch `_weight_int8pack_mm` or equivalent; works without FP8, so on the 3090 too); 2) int4 g128 (Marlin-class kernels, r: sm_80+); 3) ternary only from QAT (E3/E4) + a 2-bit packed GEMV kernel; keep embeddings, head, GDN a/b gates, A_log/dt and the state in bf16/fp32; FP8 only as a 4090 extra | Nothing built; SQ290 checks quality first |
| Rollback | KV / SWA: move the length pointer. GDN: (i) snapshot + replay of accepted tokens (ReplaySSM, arXiv 2610.13778, r), or (ii) per-position state from `fused_recurrent` over the γ verify tokens (50-160 MB, §2.2) | Designed in `notes/research-speculative-bytes.md` §3.1; SQ198 measures the replay cost |
| Sampling | Standard lossless rejection sampling (accept with min(1, p/q), resample from norm(max(0, p − q))); greedy = argmax match | Exactness test in SQ287 |

### 5.1 Speculative options, ranked

**(a) Self-speculation via fewer loops.** Only if loops survive §6.
- **Checked against `gogeta/model.py` (lines 261-282):** pass k computes `h = h + inject([norm(h), norm(e)])` followed by the core blocks, so it depends only on h_{k−1}, e and that pass's caches from earlier positions. Nothing depends on the total r:
  - `effective_depth()` is used only for init scaling;
  - `n_nograd` / `bptt_loops` change only gradients, at training time;
  - there is no loop-index or loop-count embedding.

  So **passes 1..r_d of the draft are bit-for-bit the same computation as the verifier's passes 1..r_d** (up to chunk-vs-recurrent bf16 kernel differences, ~4e-4 (m, C2)). Their per-pass SWA and GDN caches can be reused. The verifier computes only passes r_d+1..r plus the coda for the γ drafted tokens, in parallel.
- **Two traps:**
  1. The **coda** sees h_{r_d} in the draft but h_r in the verifier, so its caches differ. Either keep a second coda cache for the draft (exact draft), or let the draft's coda attend to the verifier's full-r KV for already-verified tokens. That is cheaper and maybe better, and still lossless, since only the verifier must be exact. Its acceptance is untested (SQ296).
  2. **Future features must keep pass k independent of the total r.** A per-pass role embedding (Rival Loop) or per-pass LoRA (SQ284) indexed by k is fine. Conditioning on r, or gates computed from r (e.g. a closed-gate restart keyed to r), would break reuse: add a test.
- **Cost:** no new weights; training with r ~ U{1..7} already covers r_d = 1-2. But r = 1 is 8% worse in loss (E2 card) and K6 shows the state norm grows per loop, so α is unknown: SQ196 measures it.

**(b) MTP heads** (one extra block per future token, trained jointly or post-hoc). Cheapest c (~0.05). Works for unlooped P, so it is the **default if loops are dropped**. Our recipe has none yet; quality at our scale is unmeasured.

**(c) Small separate drafter.** The M3 pilot (S, same tokenizer) doubles as the drafter: c ~0.15. Simplest and lossless, at the cost of 0.6 GB and a second cache.

## 6. Decision gates and milestones

| Ingredient | Decided by | Include in G-1B if (criterion fixed now) |
|---|---|---|
| Loops (+ self-spec (a)) | C6/C7 (PARTIAL), SQ266 (fixed r), SQ271 (thick outer), SQ284 (per-pass LoRA), SQ282/SQ283, corpus-v3 R3 (E9 vs E2) | Looped ≤ unlooped clean bpc at matched training FLOPs (3 seeds), or within 1% while decoding ≥ 1.3× faster with self-spec (SQ196 + SQ198) |
| Ternary | E3 (ternary E1), E4 (the dance), SQ290 (int4 PTQ as the competitor) | Ternary QAT within 2% of bf16 **and** better than int4 PTQ at equal weight bytes |
| Width vs depth, scale | C197, SQ243, SQ282, SQ283, SQ291 (scaling fit) | Sets d versus the block count of P/X; X only if the fit predicts ≥ 5% lower bpc than P on v3-test |
| Data | SQ276-SQ281 (corpus v3), SQ277 (tokenizer), SQ292 (repeat penalty) | Permissive tokens ≥ 20·N within 4 epochs; else shrink N |
| MoE | E60 (gated on SQ282/SQ238) | Beats dense at matched active FLOPs; then G-1B-MoE uses spare VRAM |
| Spec decode viability | SQ196 (acceptance), SQ198 (verify/step cost), SQ287/SQ296 | verify(γ = 4)/step ≤ 1.5 and a computed speedup ≥ 1.3× |
| Hardware | SQ285, SQ288, SQ289 | All decode paths pass on sm_86 **and** sm_89 |

### 6.1 Milestone ladder

| M | What | Exit criterion | Cost (c/a) |
|---|---|---|---|
| M1 | Inference stack on current 14M models: SQ286 cache/generate, SQ287 self-spec reuse, SQ288 bench, SQ289 envelope (random-init P/X), SQ285 + SQ288 on a rented 3090 and 4090 | Greedy identity vs re-forward; lossless test passes; measured tok/s within 2× of §4; P fits as predicted | < $2 (one hour each of a 3090 and a 4090) |
| M2 | Scaling-law fit on v3: 14M → 50M → 100M (corpus-v3 R4) + SQ291/SQ292; the loop, ternary and depth gates resolved | Fitted L(N, D) with CI; predicted P/X loss on v3-test; each ingredient in/out recorded in the decisions log | ~$10-30 |
| M3 | Pilot at ~1/10 size: S (0.31B) with the full recipe chosen in M2, 7-12B tokens; also becomes the drafter | Within the M2 fit's CI; no instabilities; decode stack runs it on a 3090/4090 | ~$30-60 |
| M4 | **E61 final train** (G-1B; G-3B only if M2 supports it and the budget is approved) | Trains to plan; loss within the fit's CI | ~$500-650 (G-1B), ~$3k+ (G-3B); **needs human approval** |
| M5 | Release checks: eval battery (clean, decontaminated), reproducibility from config + seed, cache/spec identity tests, tok/s and memory on a 3090 and 4090, model card | All pass; numbers in the ledger | ~$2-5 |

- **Tie to v1.0.0:** AGENTS §2 reserves 1.0.0 for "the full fusion (SSM/linear mixer + loops + ternary) trained, evaluated and reproducible". M4 + M5 is v1.0.0 **only if loops and ternary pass their gates**.
- **If they don't:** the final model is a hybrid-only release. Under the current rule it stays 0.y. **Decision for the human:** keep that rule, or redefine 1.0.0 as "the final model, whatever the gates kept".

## 7. Proposed side quests and E61

Added to `experiments/side-quests.md` by the main agent (v0.4.85):

| SQ | idea | why | status | cost |
|---|---|---|---|---|
| SQ285 | **sm_86/sm_89 kernel compatibility** on a rented 3090 and 4090: fla chunk fwd/bwd and `fused_recurrent` (keyword args, neg-eig) at hd 64 **and 128**, SDPA flash backend at T = 8k, `torch.compile`, int8 weight-only matmul; errors vs fp32 reference (reuse `sq_hw_fla_recurrent.py`) | The final model must decode on sm_86 and sm_89; only training chunk kernels at hd 64 are verified on sm_86 (SQ220), and decode only on sm_121 (C111) | `idea` | ~15 GPU-min per card, ~$0.20; rental within the standing budget |
| SQ286 | **KV + SWA ring cache and `generate()`** (implements SQ53): prefill + step, RoPE position offset, bottom-right causal verify mask, short-conv tails; greedy token identity vs full re-forward on 1k clean-val prefixes for E1 and E2 (r = 4) | Nothing in the repo can decode with a cache; it unblocks C19, C84, E8 and every number in §4 | `idea` | CPU + ~5 N1X-min |
| SQ287 | **Self-spec cache-reuse correctness**: E2 drafts at r_d = 1/2, the verifier runs only passes r_d+1..4 + coda reusing pass-1..r_d caches; compare logits to a full r = 4 forward (tolerance bf16 ~1e-3), plus an exactness test of rejection sampling (chi-square over 1e5 samples at a fixed prefix) | Verifies §5.1(a)'s reasoning from model.py; a failure means self-spec needs recomputation and loses its edge | `idea` (after SQ286) | ~5 GPU-min |
| SQ288 | **Decode tok/s benchmark harness** (extends SQ198): batch 1, ctx 1k/8k/32k, eager vs CUDA graphs vs compile, γ+1 verify cost, GDN replay cost; also reports measured vs bandwidth-model bytes per token | Calibrates the 0.8 efficiency and launch floor assumed in §4 | `idea` | ~10 GPU-min per SKU |
| SQ289 | **Memory envelope measurement**: random-init P and X shapes on a 24 GB card, bf16 and int8, KV at 32k/128k, γ = 8 checkpoints; `max_memory_allocated` + `nvidia-smi` vs §2 | Confirms "VRAM is not binding" and the usable-budget assumption (21.5 GB) | `idea` | ~10 GPU-min on a 3090 |
| SQ290 | **Weight-only PTQ quality**: int8 per-channel and int4 g128 on E1, E10, E2 (and E3 when trained), clean80 bpc delta, per-mixer sensitivity (GDN vs attention vs MLP) | Decides the default deploy precision and is the competitor ternary must beat; pass if int8 ≤ 0.5% and int4 ≤ 2% | `idea` | CPU/GPU ~10 min |
| SQ291 | **Scaling-law fit**: E1-pattern models at 14M/50M/100M (+ the M3 pilot) on v3, tokens ∝ params, fit L(N, D) (Chinchilla form) with bootstrap CI; predict P and X | Chooses between G-1B and G-3B on evidence, not hope; folds in corpus-v3 R4 | `idea` (after v3 build) | ~$10-15 |
| SQ292 | **Repeat-epoch penalty**: 14M E1 on 50M unique × 4 epochs vs 200M unique tokens, 3 seeds | codeparrot × 4 epochs is the G-1B data plan; Muennighoff et al. is unverified here | `idea` | ~15 GPU-min on 5090s |
| SQ293 | **GQA ablation**: E1 with KV heads 6 → 2 → 1 on attention + SWA, 3 seeds, 200M | GQA is an inference ingredient; add it one at a time | `idea` | ~30 GPU-min |
| SQ294 | **Long-context readiness**: flex/block-sparse SWA (SQ3), RoPE to 8k/32k, chunked prefill memory, needle test at 8k/32k on a 14M model | The 32k target is meaningless with a dense T×T mask and 1k RoPE | `idea` | ~1 h dev + ~10 GPU-min |
| SQ295 | **4090 L2-resident ternary core microbench**: re-read a 44 MB packed 2-bit matrix r times per "token" via GEMV, with and without a persisting-L2 access window, 3090 vs 4090 | Tests §4's L2-core hypothesis before anyone builds a megakernel | `idea` | ~10 GPU-min per card |
| SQ296 | **Mixed-depth coda cache for the draft**: draft coda attends to the verifier's full-r KV for verified tokens vs its own r_d cache; offline α (teacher-forced) on K9-clean val | Cheaper draft cache, maybe higher α; lossless either way | `idea` (with SQ196) | ~3 GPU-min |
| SQ297 | **Training throughput at P shape**: 50-step `--bench` of the 1.2B P config on one 5090 (and 2 × 5090 FSDP), tok/s, MFU, peak memory with activation checkpointing | Replaces the 14M MFU extrapolation behind every $ figure in §3 | `idea` | ~10 GPU-min, ~$0.10 |
| SQ298 | **Drafter / MTP acceptance at pilot scale**: S-shaped 125M drafting for the M3 pilot, plus a 1-block MTP head; α and computed speedup | Picks spec option (b) vs (c) if loops are dropped | `idea` (after M3) | ~10 GPU-min |

**E61 "Final model" (planned; to register in `experiments/experiments.json` when M2 closes).**
- **Question:** does the recipe chosen by the M2 gates (the E1-pattern hybrid at ~1.2B; loops, ternary and MoE only if they passed) land within the scaling fit's prediction?
- **Deployment criterion:** does it run interactively on one 24 GB RTX 3090 and 4090 (batch 1, KV/state cache, lossless spec decode) at the tok/s and memory this note predicts?
- **Config and seed:** committed before launch, with the prediction (loss CI from SQ291, tok/s from SQ288).
- **Matched-compute control:** none at this size (too expensive). The M3 pilot and the fit are the controls.
- **Budget:** ~$500-650 for G-1B, needing the human's approval.
- **Success:** v1.0.0 if the full fusion survived the gates; otherwise a hybrid-only 0.y release (the human decides, §6.1).

## 8. Proposed goal 6 (one line each)

- **DEVLOG "High-level goals":** `6. **Ship it on a gaming GPU.** End the experiments with one final Gogeta model, sized by our own scaling fit and training budget, that runs interactive batch-1 inference with a KV/state cache and lossless speculative decoding on a single 24 GB RTX 3090 or 4090 (plan: notes/final-model-plan.md).`
- **README "Goals":** `6. **Ship it on a gaming GPU:** a final model, sized by our scaling fit, that runs with caching and lossless speculative decoding on one 24 GB RTX 3090/4090.`

**Wording note:** goal 5 ("Do it on a laptop") needs a clause, because the final training is rented and only inference is local.

## Errata (v0.4.88, from the GPT-6.2 Astra feasibility review, each checked by the main agent)

Full review: [`notes/feasibility-final-training-astra.md`](feasibility-final-training-astra.md). The corrections below override the text above.

- **§2 ">15 GB to spare at every size and precision" is wrong.** G-3B in bf16 at 128k context leaves about **12.4 GB** (21.5 GB usable − 6.66 GB weights − 1.34 GB KV − 1 GB workspace − state). Both targets still fit with ample headroom; the conclusion "VRAM is not binding" stands.
- **§4 separate-drafter cost ratio c ≈ 0.15 is inconsistent with its own inputs.** By bytes it is 0.62 / 2.45 ≈ **0.25**, and a launch floor can only raise it. At acceptance 0.7 and γ = 3 the idealised speedup is **≈ 1.44×**, not 1.5–2.1×, before verify and rollback costs. SQ298 must measure c directly.
- **§4 prefill "~2 s at 32k" ignores the full-attention term.** Four causal full-attention layers add about 17.6 TFLOP at 32k, so prefill is **≥ ~2.45 s** at the assumed 40 TFLOP/s.
- **§6 "M3 pilot at ~1/10 size" is wrong.** S (0.311B) is **25%** of G-1B. The pilot's role (out-of-sample check of the fit, drafter) is unchanged.
- **Missing from the plan, now tracked:**
  - the trainer has no epoch coverage: `TokenData.batch()` samples random windows with replacement;
  - there is no multi-GPU path (DDP/FSDP, distributed sampler, global token accounting);
  - mid-run checkpoint/resume is only a draft patch;
  - SWA builds a dense T×T mask (8 GiB of integer temporaries at 32k);
  - RoPE has no position offset.

  These gate M3/M4 alongside SQ286/SQ294/SQ297.
- **Cost check:** with the exact 1.226B shape, 6ND gives **$379–506** for 24–32B tokens at the measured E1 rate, before overhead (the plan's $371–495 used a rounded 1.2B). At 75% of the assumed throughput, compute alone is about $506–675.
