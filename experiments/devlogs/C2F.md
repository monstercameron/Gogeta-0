# C2F: coarse-to-fine byte model around the assembled FusionDance

**Status:** built (code + CPU tests + CPU smoke runs; no GPU training yet) · **Verdict:** none yet; nothing trained beyond 20-step CPU smoke runs · **Links:** module `gogeta/c2f.py` · tests `tests/test_c2f.py` · trainer `experiments/byte_model/train_c2f.py` · draft configs `experiments/staged-configs/a4/` · design `notes/analysis-coarse-to-fine.md` · side quests SQ206, SQ207 (X1), SQ374

## Log (oldest first)

### 2026-11-23 13:00 · uncommitted (on v0.5.12 working tree) · Build
Builder subagent; files: `gogeta/c2f.py`, `tests/test_c2f.py`, `experiments/byte_model/train_c2f.py`, `experiments/staged-configs/a4/*`, this file. No git writes, no GPU.

**Design.** The global trunk is the real `gogeta.model.FusionDance` (GDN + SWA + attention mixers, shared looped core, ternary BitNet, MoE), reached through the v0.5.12 hooks (`inputs_embeds`, `return_hidden`, `return_aux`). Two patching modes, one byte decoder:

| | `patching: "tokens"` (default a4 design) | `patching: "fixed"` (alternative) |
|---|---|---|
| global input | BPE tokens, FusionDance's own embedding | P concatenated byte-encoder states -> `patch_in` -> `inputs_embeds` (FusionDance built with vocab 1, its embed/head unused) |
| patch = | the byte span of a token (variable length; ByteLevel table from `gogeta/assets/tokenizer-code-py-v1.json`, shared by code-py-v1 and v2; specials -> 0x00) | P bytes |
| decoder rows | for target token y[i]: inputs [BOT, b1..bL] -> targets [b1..bL, EOT] (257-way head), all spans concatenated into one causal byte stream | row t predicts byte t+1 |
| context | `up(H[i])` + slot embedding (row index within the span) | `up(Gs[(t+1)//P])` + slot ((t+1) % P), Gs = [start, G...] (one-patch shift) |
| loss unit | nats per target token (same unit and val tokens as the BPE model) | nats per byte |
| frozen trunk | `freeze_global` + `global_init` (FusionDance checkpoint; eval mode, default loops, no grad) | n/a |

Refinement loop (both modes): x0 = ctx + e; pass 0 h = blocks(x0); pass k >= 1 h = blocks(lam * h + inject([norm h, norm e, norm ctx])) with lam = sigmoid(per-channel), init 0.9, when `refine_contractive` (FusionDance's `inject_contractive` form), else lam = 1. Shared blocks (H-C2F) or one block set per pass (unshared control); `refine_mlp_hidden` shrinks the unshared blocks for the parameter-matched A'' control. r_b = 1 builds no inject at all. Per-pass loss: objective = CE(last) + `refine_aux_weight` * mean CE(earlier passes); eval = last pass. Default decoder block `lswa`: exact sliding-window attention computed block-locally (O(T * 2w), not the dense O(T^2) mask of the trunk's `swa`, C12) + SwiGLU.

Precision is explicit: the trunk follows its own config; `encoder_ternary` / `decoder_ternary` (default false, K4) switch the byte layers. Residual-output init (outside review 13:10, I3): each residual stream uses its own depth, encoder 0.02/sqrt(2 * len(local_encoder)), decoder 0.02/sqrt(2 * r_b * len(refine_blocks)); the trunk keeps FusionDance's own scale. Shared r_b = 2 and the unshared 2-pass controls therefore get the same scale.

**Why tokens are the default.** SQ374 (main agent, this session): fixed byte patches cost too much at this scale (E13b P = 4 at ~E1 FLOPs: 0.553 vs 0.339 bpc on clean80; loss rises 7-9% across the slots of a patch). The token mode keeps the BPE trunk exactly as trained today and replaces only its 8k head by a speller.

### 2026-11-23 13:30 · uncommitted · Causality proof and tests
*Fixed:* G[p] has seen bytes <= (p+1)P - 1; row t reads G[(t+1)//P - 1] (or `start`), which covers bytes <= (t+1)//P * P - 1 <= t; the byte stream adds byte t's embedding; every local mixer is causal. *Tokens:* span i (spelling y[i] = x[i+1]) reads H[i] (tokens <= i); local mixers run over the concatenated rows in span order, so a row sees only earlier spans (bytes of y[<i] = x[1..i]) and earlier bytes of y[i]. Changing token x[s] may change H[>= s], span s-1 from slot 1 on, and later spans; spans < s-1 and span s-1's BOT row must not change. Re-injection reads the same row's e and ctx, so it adds no path.

`CUDA_VISIBLE_DEVICES= python tests/test_c2f.py` (WSL, CPU, fp32, GDN on `experiments/byte_model/cpu_ref.py`'s token-by-token recurrence): **ALL PASS, 17 tests.**
- Fixed: perturbation at patch starts/middles/ends (7 positions at P = 2, 10 at P = 4, T = 48): rows < s bitwise unchanged in every pass, row s changes; P in {2, 4} x r_b in {1, 2} (+ contractive at r_b = 2) x global {dense GDN/SWA/attn; all parts: loops 2 + ternary core + MoE core-only, `moe_impl` dense; same with `gather`}: 18 configurations pass, bitwise even with MoE gather. Broken control (no shift) leaks: 6/14 (P = 2) and 12/20 (P = 4) (pass, perturbation) pairs.
- Tokens: 5 perturbations each, r_b in {1, 2} x {dense, all parts with gather} x {trained, frozen} trunk: 8 configurations pass bitwise; broken control (span i reads H[i+1]) leaks 10/10.
- Also: block-local SWA == dense-mask SWA (atol 1e-6); r_b = 1 == hand-written head(norm(blocks(ctx + e))) bitwise, and contractive is a no-op at r_b = 1; backward reaches every parameter (loops + ternary + MoE core-only global, contractive r_b = 2, aux loss), MoE aux passes through; frozen trunk gets no gradients and is unchanged by an optimizer step; token loss = summed row CE / target tokens; `next_byte_logits` (generation) == teacher-forced rows (1e-5); token table round-trips real code incl. non-ASCII through the `tokenizers` library (8192 ids, longest token 41 bytes); FLOP and param accounting add up.
- Init (I3), residual-output std measured on built models (d_local 128): decoder r_b = 1 0.01415 (want 0.01414); r_b = 2 shared 0.00996, unshared 0.01001, param-matched unshared 0.00998 (want 0.01000); encoder 0.00995-0.01000 (want 0.01000); trunk 0.00634 (want FusionDance's 0.00632). Token-mode decoders checked the same way.

### 2026-11-23 13:40 · uncommitted · FLOP accounting (computed, not measured)
FusionDance's rule throughout (2 x params per application + attention/GDN terms, training = 3 x forward; MoE active params; global loops beyond `bptt_loops` count the no-grad prefix forward only, as `gogeta.train` does; the trunk's unused head excluded; one byte head per loss-carrying pass). Fixed mode is per byte (global / P); token mode is per token (decoder rows per token = 3.735 measured bytes + 1 EOT = 4.735; a frozen trunk counts forward only).
Cross-check: with E14b's shapes (E1 prelude global, conv+gdn encoder, P = 4), encoder 3.05M, patch_in 0.44M, global 18.04M, up 0.11M per byte equal `model_bytes.PatchedByteLM` and `flops-table.md` exactly (test). Decoder/head differ by design (no injection on pass 0, 3-way inject on passes >= 1, single head without aux loss): 6.77M + 0.29M vs the table's 6.99M + 0.59M.

Draft configs (`experiments/staged-configs/a4/`, not registered; numbers computed by `gogeta.c2f` and embedded in each file's `computed` block):

| file | design | train FLOPs | vs reference |
|---|---|---|---|
| `a4a_tok_r2_shared` | tokens, A3mx trunk verbatim (d 384), d_local 128, SwiGLU 512, r_b 2 shared, contractive, aux 0.5 | 189.89M / token | 1.003 x A3mx BPE (189.36M) |
| `a4b_tok_r1_single` | same, r_b = 1 | 179.64M / token | 0.949 x |
| `a4c_tok_r2_unshared_parammatched` | A'': 2 unshared blocks, SwiGLU 170 (decoder blocks 262,144 vs 262,400 params) | 182.43M / token | 0.963 x |
| `a4x1{a,b,c}_frozenA2s_*` | same three decoders on the frozen A2s seed-2 trunk (sha256 a0c15385...) | 76.19 / 65.94 / 68.73M / token | 0.403 / 0.348 / 0.363 x A2s BPE (189.25M) |
| `a4fp{a,b,c}_fixedp4_*` | fixed P = 4, A3mx structure at d 256 (mlp 704, moe 352), d_local 128, SwiGLU 320 (A'': 74) | 24.64 / 22.92 / 23.51M / byte | 1.011 / 0.940 / 0.965 x E1/3.735 (24.37M) |

Token-mode sizing rule: the r2_shared decoder (17.24M decoder + 1.87M heads + 0.30M up per token) costs about what A3mx's 8k head costs (18.87M), so arm a matches A3mx's own train FLOPs/token. Fixed-P sizing: A3mx at d 384 costs 42.62M/byte in the global model alone at P = 4 (1.75 x E1/3.735), so the trunk width was scaled to 256. Params: a4a 25.72M total (global 25.32M incl. 3.15M token embedding; byte side 0.40M); a4fpa 10.74M (global 10.09M).

### 2026-11-23 13:45 · uncommitted · CPU smoke runs (WSL, CPU, GOGETA_TRACKER=none, GOGETA_DATA = a scratch dir with symlinked datasets; not results)
`python -m experiments.byte_model.train_c2f --config experiments/staged-configs/a4/<smoke>.json --device cpu`, 20 steps each, all `status: done` with summary.json, samples.txt and model.pt:
- `a4_smoke_cpu_tokens.json` (tiny all-parts trunk, tokens, code-py-v1, 64 tokens x 4): train loss 23.72 -> 21.61 nats/token (step 0 -> 19), final val_loss 16.72 on 8 val seqs; MoE aux 0.024 -> 0.022.
- `a4_smoke_cpu_frozen_a2s.json` (frozen A2s trunk, 0.04M trainable params): 23.70 -> 21.92, final val_loss 16.97; the trainer warns that A2s was trained on code-py-v2 and this run uses code-py-v1.
- `a4_smoke_cpu_fixedp.json` (fixed P = 4, code-py-v1-bytes, 256 bytes x 4): 5.548 -> 4.196 nats/byte, final val_loss 4.298; slot losses 4.30/4.30/4.31/4.29.
- CLI overrides: `--max-tokens 8192 --micro-batch 2 --seed 1 --lr 0.002` recorded in `cli_overrides` and applied (8 steps).
These only show the plumbing works; 20 steps of a tiny model say nothing about quality.

### 2026-11-23 13:50 · uncommitted · Open issues
1. **Fleet launch path.** The cockpit's remote launcher (`cockpit/fleet.py`) builds `python -m gogeta.train ...` and its busy check greps `gogeta[.]train`; `train_c2f` runs as `python -m experiments.byte_model.train_c2f` (jobs files with full commands, like `wave4_jobs.txt`, work). A hook in the cockpit (dispatch `"arch": "c2f"` configs to `train_c2f`, and widen the busy check) is the main agent's call.
2. **X1 checkpoint location.** `global_init` holds this machine's path to the A2s seed-2 checkpoint; a fleet box needs that file (the run records its sha256). code-py-v2 (A2s's data) is not on this machine, so local frozen runs use code-py-v1 (same tokenizer, different val split).
3. **Comparability.** Token-mode val_loss scores the bytes jointly with the canonical BPE segmentation (EOT marks token ends), the same quantity a BPE model's val_loss scores, so it is directly comparable on the same val tokens; it is an upper bound on the bytes-only likelihood. Fixed-mode bits/char are per byte. Clean80 scoring (SQ374's scorer) is not wired to `C2FTokenModel` yet.
4. **Not measured:** wall-clock and memory. Token mode's decoder runs on ~4.7 x the token count (B x ~4,850 rows at 1,024 tokens, padded to the longest row in the batch); `spans()` costs one host sync per step. The frozen trunk could be run in bf16 without autograd state, which the current code already does under no_grad.
5. **Untuned:** lr (0.006 from A3mx), warmup, aux weight 0.5, d_local 128. Sampling in token mode re-tokenizes when a spelled byte string is not a vocab token (qualitative samples only).
6. **No hooks needed** in `gogeta/model.py` / `moe.py` / `train.py`; `train_c2f` imports `lr_at`, `micro_batches` and `MOE_STATS` from `gogeta.train` and `TokenData`, `sample`, `autocast` from `train_bytes`.
