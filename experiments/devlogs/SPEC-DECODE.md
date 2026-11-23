# SPEC-DECODE micro devlog: cached inference and self-speculative decoding

**Status:** built, CPU-tested; lossless only up to floating-point near-ties on a ternary (int8-activation) checkpoint · goal 6 (notes/final-model-plan.md §5) · code `gogeta/decode.py` · tests `tests/test_decode.py` · bench `experiments/checks/spec_decode_bench.py` → `experiments/checks/results/spec_decode_bench.json`

### 2026-11-23 · builder subagent · what was built
- **`gogeta/decode.py`** (new; `gogeta/model.py` untouched, no hook needed). `prefill(model, idx, loops)`, `step(model, cache, tokens[B,k])` for k >= 1, `cache.rollback(n)`, `greedy_generate`, `speculative_generate(model, prompt, n_new, r_draft, r_full, k)`.
  - attn: KV cache, RoPE at absolute positions; for k > 1, an explicit bottom-right causal mask. SWA: the last window-1 K/V plus the rollback log, with the training mask `0 <= t - s < w`. GDN: fp32 state S per head plus the raw q/k/v inputs to the ShortConvs (conv_size-1 past inputs). The recurrence is in torch on CPU: l2-normalised q/k, decay g, beta = sigmoid(b) or 2·sigmoid(b) (neg-eigval), scale hd^-0.5, out_norm and the output gate. On CUDA it calls fla `fused_recurrent_gated_delta_rule` with keyword arguments (C110) and falls back to torch if that call raises.
  - Loops: every (pass, core block) pair has its own state. Covers inject (incl. contractive), loop_embed, loop adapters, MoE FFN (the model's own module, aux=None), ternary BitLinear and ternary experts, moe_core_only and ternary_fp_ends, all through the model's own submodules.
  - Rollback: every state logs its last `max_rollback` tokens across steps (GDN: S after each token plus conv inputs; SWA: window-1+log entries; full attention: truncate). A rollback is a lookup with no replay. The first version logged only the most recent step, which broke the draft cache (several 1-token steps); the 1-prompt smoke bench caught this, and the tests now cover it.
  - Self-speculation: the draft is the same model at r_draft passes with its own cache, proposing k tokens greedily. The full model verifies [pending, d_1..d_k] in ONE `step` call, accepts the longest agreeing prefix plus its own token, and both caches roll back.
- **Not done:** reuse of the draft's pass-1..r_d caches by the verifier (notes/final-model-plan.md §5.1(a)); `inputs_embeds` (byte front end); sampling (greedy only); CUDA/bf16/fla path (written, never run: no GPU here).

### Tests (`python tests/test_decode.py`, CPU fp32): ALL PASS
9 tiny configs: attn; swa (window 6 < seq); gdn; gdn neg-eigval; loops (gdn+swa core, r = 1..4); contractive inject + loop_embed + rank-4 adapters; ternary; ternary_fp_ends + contractive; MoE core-only with ternary experts. Results:
- (a) prefill + mixed-size steps vs the model's own full forward (GDN via the torch reference kernel), max error <= 2.9e-6;
- (b) one k-token step vs k single steps, <= 1.4e-6;
- (c) rollback within one step, across several steps, and twice in a row, <= 1.9e-6; a rollback past the log raises;
- (d) speculative output == cached greedy == full re-forward greedy, for 3 seeds × 5 (r_draft, r_full) pairs × k ∈ {1, 2, 4, 7}. With r_draft == r_full, every draft is accepted. Looped configs are scaled so that drafts do get rejected (mean acceptance 0.20-0.56), which exercises the rollback path.

### Bench: A2s-20261123-170724-s2 (CPU fp32, 20 val_clean50 prompts of 128 tokens, 64 new tokens, r_full = 2)
The checkpoint has **18.2M params (computed)**, not 200M: ternary (int8 activations), fp ends, contractive inject, trained at a FIXED r = 2.

| r_draft, k | lossless prompts | acceptance (pooled) | tokens / verify call | full calls / token (incl. prefill) | draft calls / token |
|---|---|---|---|---|---|
| 1, 2 | 17/20 | 0.805 | 2.57 | 0.398 | 1.05 |
| 1, 4 | 16/20 | 0.701 | 3.68 | 0.283 | 1.19 |
| 1, 8 | 18/20 | 0.566 | 5.23 | 0.204 | 1.50 |
| 3, 2 | 17/20 | 0.894 | 2.75 | 0.373 | 1.02 |
| 3, 4 | 18/20 | 0.834 | 4.19 | 0.251 | 1.10 |
| 3, 8 | 18/20 | 0.740 | 6.63 | 0.164 | 1.23 |

r_draft = 3 costs more than the target, so it is an agreement measurement, not a useful draft.

- **r agreement** (r = 1 is untrained for this model): teacher-forced per-position argmax agreement with r = 2 is 0.869 for r = 1 and 0.934 for r = 3. Free-running greedy matches r = 2 for a mean of 4.1 tokens (r = 1) and 13.2 tokens (r = 3).
- **Numerics:**
  - Cached vs the model's own full forward: max |Δlogit| = 0.19.
  - The full forward itself moves by up to 0.21 between 1 and 4 CPU threads (its noise floor). With every BitLinear's int8 activation quantisation off, cached vs full is 1.7e-5. So the decode path computes the same function, and int8 activation rounding amplifies fp32 reordering noise to ~0.1-0.2 logits.
  - Top-1 minus top-2 margin of the full model: min 4.9e-4. Steps with margin below the cached-vs-full gap: 15/1280; below the thread noise floor: 18/1280; below 0.2: 52/1280.
  - Plain cached greedy itself disagrees with the full forward's argmax on 3/20 prompts.
- **Lossless: FAIL on the ternary-activation checkpoint.** 16 of 120 (prompt, config) runs diverged from 1-token cached greedy, on 5 prompts. The worst-documented case is doc 34, step 36: full-model margin 4.9e-4 between tokens 330 and 430. The 1-token step and the full forward pick 330; the k-token verify picks 430. These are numerics flips at near-ties, not a logic error. Diagnostic (`--diag-lossy`, same JSON, `diagnostic_lossy_prompts`): every one of the 16 divergences is at a step whose full-model margin (4.9e-4 to 0.099) is below that step's cached-vs-full gap (0.060 to 0.105). Re-running the 5 lossy prompts with int8 activation quantisation off gives 30/30 lossless (prompt, config) runs. On tiny fp32 configs, including ternary ones, the tests show exact token identity. Verdicts in the JSON: original criterion (lossless and gap < 1e-3) FAIL. Revised: (1) lossless FAIL, (2) quant-off gap < 1e-4 PASS, (3) gap <= thread noise floor PASS.
- **Lossless per kernel path:** CPU torch fp32 path: algorithmically lossless, numerically lossless except at near-ties (above). bf16 / fla fused_recurrent on GPU: not checked.
- CPU wall clock: 44 min for 20 prompts with 5 workers × 4 threads, sharing the CPU with another check. CPU timings are informational; GPU timing is out of scope.

### Limitations and next steps
- Exact greedy identity under int8 activation quantisation needs batch-invariant numerics: the verify path's k-row matmuls must round like the 1-row step. Options: a batch-invariant GEMM, or verifying in fp64 on CPU. Alternatively, define "lossless" as identity with the full model's own verify-path decisions, and report the near-tie rate (margin < noise floor) as the residual.
- CPU decode is slow (~0.2 s/token at 8 threads): BitLinear re-quantises its weights on every call, and the GDN recurrence is a Python loop over tokens (prefill of 128 tokens ~7 s). A decode-time cache of quantised weights would need the same expression `w + (w_q - w)` to stay bit-identical.
- The draft does not reuse the verifier's pass-1 caches. That reuse is the main cost lever in final-model-plan §5.1(a).
