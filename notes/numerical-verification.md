# Numerical verification (rigor workstream A6)

Status: staged in `experiments/staged-tests/` (nothing under `tests/` or `gogeta/` was touched; a campaign is running). CPU part run and passing: 49 checks, `python experiments/staged-tests/run_numerical_suite.py --stamp`, WSL, fla 0.6.0, torch 2.14.1+cu130, triton 3.8.0, 2026-11-22. GPU part written, self-tested on CPU with shim kernels, **not run** (queue item SQ150, <= 10 GPU-min). Proposals: `experiments/staged-tests/proposed-{claims,side-quests,predictions}.md`, `proposed-queue.json`.

## Why this exists
The audit called C2 a textbook example: `fused_recurrent_gated_delta_rule` was declared NaN on our GPU, but our test passed `beta` as the 5th positional argument, which in fla 0.6.0 is `gk`. Every layer of defence below would have caught it earlier: the contract test (pinned order and a scan for positional arguments), a reference implementation (a wrong answer is not a NaN story), a known-answer test (S^T k_t = v_t after a beta = 1 write), the fingerprint (a signature that changes forces a re-run).

## Files (all in `experiments/staged-tests/`; they work from there or from `tests/`)
| File | What |
|---|---|
| `test_api_contract.py` | Pinned signatures, AST scan of positional/keyword use, SDPA / RMSNorm / normalize behavioural contracts (10 checks, CPU) |
| `numref.py` | Independent pure-PyTorch references: gated delta rule (token loop **and** chunkwise UT/WY form), explicit-mask attention / SWA, complex RoPE, integer-code BitLinear, RMSNorm, SwiGLU, causal conv, and a whole-model reference `model_ref(...)` that reads a `FusionDance`'s weights (with `bptt=` for truncation, `sd=` for gradcheck) |
| `test_reference_blocks.py` | Reference self-consistency, known answers, STE analytics, gradchecks, model vs reference (values and gradients), loop injection, BPTT, long-T equivalence (24 checks, CPU) |
| `error_budget.py`, `error_budget.json`, `tolerances.py`, `test_error_budget.py` | The mixed-precision budget: derivation, data, the `tol(block, dtype, T, d)` API, tests (5 checks) |
| `test_mutation_controls.py` | Negative controls: injects the C2 call, a swapped signature, an off-by-one window, a missing STE, a flipped RoPE, bptt off by one, a decay sign error, a bumped version, and requires the suite to catch each (10 checks) |
| `env_fingerprint.py`, `env_fingerprint.json`, `run_numerical_suite.py` | The re-validation trigger and the runner that stamps it |
| `gpu_numerical_suite.py` | GPU confirmation (SKIP, exit 77, without CUDA; `--selftest` runs it on CPU with shims) |

## Coverage matrix (block x test type)
`C` = CPU, run and passing; `G` = written, GPU run pending (SQ150); `-` = not applicable; `gap` = not covered.

| Block | API contract | fp64 reference | Known answers | Finite-diff gradient | Model vs reference | Chunk/stream/recurrent | Precision budget |
|---|---|---|---|---|---|---|---|
| fla `chunk_gated_delta_rule` (the kernel) | C signature + calls; G behaviour | C (2 formulations); G kernel vs ref | C (on the ref); G | C on the ref; G kernel bwd vs fp64 autograd | G (real block + model) | C ref 8k; G kernel 8k/16k | C emulated, GPU-calibrated; G |
| fla `fused_recurrent_gated_delta_rule` | C incl. the positional trap; G | C | C | G | G | G prefill + decode | C emulated; G |
| fla `chunk_kda`, `fused_recurrent_kda` | C signatures, kwargs-only `A_log`/`dt_bias` | gap (KDA reference lives in `sq_kda_kernel.py`, not rebuilt here) | gap | gap | gap | gap | gap (KDA is not in the model) |
| GDN mixer glue (convs, gates, normalise, out-norm, beta mapping, neg-eig) | C kwarg contract of the model's call | C | C (zero input -> 0; Householder) | C via model-vs-reference gradients (the fp32 cast of g, beta defeats finite differences) | C, 3e-8..1e-7 | C | C (kernel stand-in is fp64; kernel error budgeted separately) |
| Full attention (SDPA + RoPE) | C SDPA positional order, mask semantics | C | C (zero q,k -> running mean) | C gradcheck, input + all params | C, 7e-16 | - | C fp32/bf16 |
| SWA (mask path) | C | C | C (window=T equals full, window=T-1 differs only in the last row, window=1 -> v, influence matrix == mask) | C | C | - | C |
| SwiGLU MLP | - | C | C (zero -> 0) | C | C | - | C |
| BitLinear (absmean ternary, int8 act, STE) | - | C (integer-code form) | C (+-s fixed point; pattern fixed point) | C: STE analytic, and gradcheck of the frozen-offset surrogate | C (ternary models) | - | C; flip-dominated at model level |
| RoPE | - | C (complex form) | C (relative-position property) | via attention | via attention | - | C table error vs position |
| RMSNorm / `F.normalize` | C eps semantics | C | C (normalize(0) = 0) | via blocks | via blocks | - | gap (no separate budget) |
| Loop injection + BPTT truncation | - | C | C (r = 0, 1, 2, 5 hand loop) | C (looped model gradcheck, fast mode) | C | - | C (c u L_eff) |
| Whole model | - | C | - | C | C (8 configs) | G | C; G |
| Environment drift | C fingerprint | - | - | - | - | - | - |

## Results
**API contract.** Installed signatures match the pin for all nine external functions plus SDPA. The scan covered 72 kernel-related call sites in `gogeta/`, `experiments/checks/`, `evals/`, `post_training/`, `tests/`, `tools/`.
- **Unsafe positional calls: 1**, `experiments/checks/sq_hw_fla_recurrent.py:111`, the deliberate reproduction of the C2 bug (allow-listed with its reason).
- **Fragile but correct** (positional past the 3rd argument, safe only because of the pinned order): 30 sites, including the deliberate one: 21 `chunk_gated_delta_rule` (`gogeta/model.py:193,197` and several `experiments/checks/` scripts), 4 `chunk_kda`, 4 `fused_recurrent_kda`, 1 `fused_recurrent_gated_delta_rule` (the deliberate one). Note `fused_recurrent_kda` and `chunk_kda` agree on `q,k,v,g,beta` and then **diverge** (`A_log, dt_bias` vs `scale`): a second positional trap that nobody has hit because everyone passes keywords after `beta`.
- **Unresolvable** (called through a variable, a static scan cannot see the callee): 7 (`fn(...)` in `k3_mqar_recall`, `sq_kda_kernel`; `self._kernel` in two monkeypatched copies of the GDN forward in `sq_06_rotation_parity.py:132` and `sq_hw_gdn_fusions.py:76`).
- All keywords we pass are named parameters, except `chunk_kda(..., A_log=, dt_bias=)`, which fla reads from `**kwargs` and silently drops unless `use_gate_in_kernel=True` (pinned and checked). All five fla ops accept `**kwargs`, so a typo'd keyword is not an error (behaviour to be confirmed on the GPU).

**References and model equivalence** (fp64, CPU). Chunked vs token-by-token reference 2e-15. Model vs independent reference, relative to the logit max: attention / SWA / ternary 5e-16 to 9e-16; models with GDN 3e-8 to 1e-7. The GDN residue is real and explained: `gogeta/model.py` computes `g` and `beta` in fp32 even when the model is fp64 (lines 189, 192, 196). Parameter gradients agree with the reference's to 4e-7 relative (40 tensors).

**Gradient checks.** `gradcheck` (fp64, eps 1e-6, atol 1e-7) passes for: both delta-rule references wrt q, k, v, g, beta, S0 (beta up to 2); the model's attention, SWA (w=3) and SwiGLU wrt input and every parameter; the whole looped model (prelude swa, core attn+swa x2, coda attn) and the independent reference with a GDN core wrt every parameter (fast mode). **The BitLinear STE gradient**, stated analytically: forward `y = x_q w_q^T`, `w + (w_q - w).detach()` makes dw_q/dw = I and likewise dx_q/dx = I, so for L = <G, y>: dL/dx = G w_q, dL/dW = G^T x_q, evaluated at the quantised operands, with no clipping mask (weights beyond the clamp still receive gradient) and no gradient through the absmean scale or the activation scale. Verified exactly (1e-12), and the same gradient is the true gradient of the smooth surrogate `(x + c_x)(W + c_w)^T` with the quantisation offsets frozen (gradcheck passes on it). `gradcheck` on BitLinear itself is meaningless (piecewise constant).

**Known answers** (see C195). One side finding: a weight matrix of {-1, 0, +1} * s is a fixed point of the ternary *pattern* but not of the magnitude (absmean = s * fraction-nonzero, so each re-quantisation shrinks the scale); only all-nonzero matrices are an exact fixed point. `torch.round` is round-half-even where BitNet's RoundClip is floor(x + 0.5); they differ only at exact ties (the reference uses floor(x+0.5); no tie occurred in fp64 tests).

**BPTT.** With r = 5, `bptt_loops` = 2 in `train()`, the model's parameter gradients equal a reference that detaches h after each of the first 3 loops; they differ from full-BPTT gradients by > 1e-3 relative (the truncation is active); `eval()` and r <= `bptt_loops` give full BPTT; forward values are unchanged.

**Long T** (fp64 reference). T = 8192, B = 1, H = 2, K = V = 8, beta up to 2: chunked vs token loop 7.9e-16; 9-piece streaming with carried state 6.9e-16; chunk sizes 16/128/1000 identical; prefill + 200 decode steps identical; 16k tokens with beta up to 2 keep a bounded state.

**Mixed-precision budget.** Metric: scaled error max|x - ref64| / max|ref64|. Form: `c u T^a d^b` (fitted; a, b clipped to [0, 1]; c = 2 x the worst observed ratio).
- fp32 modules (attention, SWA, SwiGLU): 5e-7 to 1.2e-6 observed (about 8 to 20 u_fp32); tolerances 1e-6 to 3e-6.
- bf16 autocast modules: 0.9e-2 to 1.1e-2 observed (about 2.4 u_bf16), essentially independent of T and d (fitted exponents 0 to 0.17); tolerances 1.8e-2 to 2.2e-2.
- GDN, CPU emulation (operands rounded to TF32 / bf16, fp32 accumulate): recurrent fp32 6e-7, recurrent bf16 3.3e-3, chunk bf16 7.7e-3, chunk fp32 (TF32 dot) 7.5e-4, with no growth in T up to 4096 (exponent <= 0.04).
- **Emulation vs the GPU numbers already in the repo** (T = 300, d = 64, B = 2, H = 4): bf16 chunk 1.05x, bf16 recurrent 1.32x, neg-eig bf16 1.10x, fp32 recurrent 0.69x, but **fp32 chunk 3.1x and neg-eig fp32 3.5x** (recorded 2.2e-4 and 4.8e-4 vs 7e-5 and 1.4e-4 emulated). So the Triton chunk kernel in fp32 is less accurate than TF32 operand rounding explains. The budget carries a GPU calibration factor for those two cells (2.7-2.8x), visible in the JSON. SQ153 looks for the cause.
- Whole model, logits scaled error ~ c u L_eff: non-ternary bf16 2.4e-2 (L_eff 6) to 7.6e-2 (L_eff 23), fp32 1.5e-6 to 3.8e-6. **Ternary models are dominated by quantisation-code flips**, not rounding: fp32 vs fp64 gives 0.6e-2 to 1.3e-2 logits error, bf16 3e-2 to 6e-2, roughly independent of depth. A comparison of a ternary model across precisions needs a flip budget, not an epsilon budget. Loss differences: 3e-7 (fp32) to 3e-3 (bf16) nats.
- **Replaces** the ad-hoc tolerances: for the recurrent kernel fp32, 2e-3 -> 8e-7 (2,400x tighter, so a 100x regression no longer passes); chunk fp32 4e-3 scaled (no tighter than 2e-3 absolute: the old number was about right); chunk bf16 1.5e-2 scaled vs 3e-2. Re-derive with `error_budget.py --write`; tests cite `tolerances.tol(block, dtype, T, d)`.
- Validation of the budget: it covers all six recorded GPU errors with utilisation 0.26-0.44, and eight fresh-seed CPU cells with utilisation <= 0.39; rounding a perfect result to bf16 exceeds the fp32-recurrent budget and respects the bf16 one.

**Mutation controls** (a suite that cannot fail proves nothing): the scanner flags the exact C2 call and a keyword typo; a swapped `gk`/`beta` order fails the pin; beta handed to the kernel in the decay slot fails model-vs-reference; an SWA window off by one fails the influence/mask test; a missing STE fails the STE test; a reversed RoPE fails model-vs-reference; bptt off by one is distinguishable; a decay sign error fails the known answers; a bumped fla version, signature and source hash each trip the fingerprint.

**Model-code findings from this work** (details in C199): (1) `nn.RMSNorm` default eps is the fp32 eps for bf16, so GDN's `out_norm` is not scale-free where mean(o^2) ~ 1e-7 (0.6% of tokens at init, 6% below 10 eps); fp64 and fp32 evaluations of the same weights then differ by up to 50% on a small-`o` GDN block, which is why the fp64 reference takes the run precision's eps. (2) RoPE tables are built from fp32 angles: 1.7e-5 error at 1k positions, 2.7e-4 at 16k. (3) `g`, `beta` are fp32 in an fp64 model.

## Re-validation procedure (the audit's "when versions or signatures change")
1. `python experiments/staged-tests/env_fingerprint.py` fails when torch / triton / fla / numpy / transformers / trl / lm_eval versions, any pinned signature, SDPA's signature or the source hash of the four fla op wrappers differ from `env_fingerprint.json`, **or** when the fingerprint was recorded without a passing suite stamp.
2. Then: `run_numerical_suite.py` (CPU, ~1 min, fails loudly with the signature diff), audit call sites if a signature moved, run SQ150 on the GPU, and `run_numerical_suite.py --gpu --stamp` re-records.
3. Before a campaign or a version tag, run step 1 (it is instant). The GPU model is informational only (warning).

## Gaps (honest list)
- **No GPU result yet.** The fla kernels themselves are checked on the GPU only by the old tests (C1, C111) until SQ150 runs. Everything above that says "kernel" is the CPU emulation or the fp64 reference.
- The CPU emulation under-predicts the fp32 chunk kernel 3x (above); the budget for it rests on one T = 300 GPU datum, extended by calibration, not by understanding.
- KDA kernels: contract pinned, but no reference or numerical test (not in the model; only `sq_kda_kernel.py`).
- Budgets come from random-weight blocks (std 1/sqrt(d)), N(0,1) inputs, 3 seeds, T <= 4096, d <= 128; trained models have sharper attention and different state norms (SQ158). T = 8192/16k are extrapolated with a 2x allowance.
- GDN glue gradients are compared to the reference but not finite-differenced (fp32 casts); the in-kernel gate (`use_gate_in_kernel`) path is not exercised by the model and not tested.
- Backward budget for the kernels is provisional (8x forward); the GPU run sets it.
- Indirect kernel calls (7) cannot be scanned; test code that monkeypatches `_kernel` (SQ6, SQ77) is outside the scan's resolution.
- No fuzzing of edge shapes on the GPU (T not a multiple of 64, T = 1, GQA shapes, non-contiguous inputs): SQ159.
- bf16 *pure* (not autocast) module paths were not budgeted except for the GDN kernels; the project trains with autocast.
- Python-level hygiene is not fixed: the 30 fragile positional calls remain (SQ154 proposes keyword-only calls; it is a code change, deferred until the campaign ends).
