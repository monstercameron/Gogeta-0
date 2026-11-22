# Research: Kimi Linear / Kimi Delta Attention (KDA), and what it can do for Gogeta-0

*Research side quest, 2026-11-22. Ledger rows C70–C79, side quests SQ50–SQ59 (all pending review).*

**Status words in this note** (AGENTS.md §0): *reported* = stated by the paper or Moonshot's repo, not checked here.
*Verified here* = a local run with on-disk output. *Computed* = arithmetic from code or config. *Predicted* = my guess,
written before any run.

## Sources actually read

| Source | What I read |
|---|---|
| Paper: *Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv **2510.26692v2** (1 Nov 2025), `refs/02-linear-time-mixers/2510.26692 - Kimi Linear (KDA).pdf` | Full text via `pdftotext` (with and without `-layout`): §1–§7, App. A–D (proofs, the pseudocode listing, the 5.7T results) |
| fla **0.6.0** as installed in `~/.venvs/gogeta` | `fla/ops/kda/{__init__,chunk,fused_recurrent,gate,naive}.py`, `fla/ops/kda/backends/flash_kda.py`, `fla/layers/kda.py` |
| `huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct`: `config.json`, `modeling_kimi.py` | Layer layout, MLA config, NoPE handling, the KDA layer's gate/β/output-gate code (read through a fetch tool, so treat as *reported*) |
| `github.com/MoonshotAI/Kimi-Linear` README | Model list and headline numbers (*reported*) |

## 1. KDA vs our GDN

### Update equations, side by side

Per head, state `S ∈ R^{d_k×d_v}`, `q, k` L2-normalised, `β_t ∈ (0,1)`.

| | Our GDN (`gogeta/model.py`, fla `chunk_gated_delta_rule`) | KDA (paper eq. 1; fla `chunk_kda`) |
|---|---|---|
| State update | `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ` | `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ` |
| Decay | `α_t = exp(g_t) ∈ (0,1]`, **one scalar per head** | `α_t = exp(g_t) ∈ (0,1]^{d_k}`, **one per key channel** (a diagonal over the key rows of `S`; the value axis is not gated) |
| Decay parametrisation | `g_t = −exp(A_log[h]) · softplus(w_{a,h}ᵀ x_t + dt_bias[h])`: `W_a: D→H` (full rank, tiny) | `g_t = −exp(A_log[h]) · softplus(W_up W_down x_t + dt_bias[h, :])`: low-rank `W_down: D→d_head`, `W_up: d_head→H·d_k`; `A_log` still per head, `dt_bias` now per (head, channel) |
| Readout | `o_t = S_tᵀ q_t · d_k^{-1/2}` | same |
| β | `sigmoid(W_b x)`, scalar per head | same (Kimi's released code: `b_proj(x).sigmoid()`). fla also offers `allow_neg_eigval` (β ∈ (0,2)); Kimi doesn't use it |
| Delta rule | unchanged | unchanged. Order matters now: decay first, then the delta correction reads the *decayed* state (eq. 1; fla's `naive_recurrent_kda` does the same) |
| q/k/v path | `ShortConv(k=4)` + SiLU, then L2Norm on q, k | same: `L2Norm(Swish(ShortConv(W x)))`; v: `Swish(ShortConv(W x))` |
| Output | `W_o( RMSNorm_head(o) ⊙ SiLU(W_g x) )`, `W_g` full rank `D→D` | `W_o( RMSNorm_head(o) ⊙ Sigmoid(W_g↑ W_g↓ x) )` (eq. 10), **low-rank sigmoid gate** (rank = head dim; fla's layer adds a bias on `W_g↑`) |
| Head dim | 64 (d=384, 6 heads) | 128 (Kimi-Linear-48B: 32 heads × 128) |

As a constrained DPLR (§6.2): `S_t = (D − a_t b_tᵀ) S_{t−1} + k_t v_tᵀ` with `D = Diag(α_t)`, `a_t = β_t k_t`,
`b_t = k_t ⊙ α_t`. Tying both low-rank vectors to `k` is what makes the kernel cheaper than general DPLR.

**GDN is the special case of KDA with `g` tied across the `d_k` channels.** Verified here on CPU for the math
(reference loops, below). The GPU check also tests it for fla's kernels.

### Chunkwise algorithm (paper §3.1, App. A/C; fla `fla/ops/kda/`)

- Split the sequence into chunks of `C = 64` (fla allows 32 or 64). Inside a chunk, the product of the rank-1
  Householder-like transitions is packed with a **WY representation** (following Comba's form of `P`, which avoids an
  extra matrix inverse), and the **UT transform** turns the recurrences for `w_t`, `u_t` into one lower-triangular
  inverse per chunk: `M = (I + StrictTril(Diag(β) (Γ ⊙ K)(K/Γ)ᵀ))⁻¹ Diag(β)`, solved by forward substitution.
- Inter-chunk: `S_{[t+1]} = Diag(γ^C) S_{[t]} + (Γ^{i→C} ⊙ K)ᵀ (U − W S_{[t]})`. Output = inter-chunk `(Γ ⊙ Q) S` + intra-chunk
  masked `(QKᵀ with decay ratios)` times the "pseudo-value" `U − W S`.
- The per-channel decay ratio `exp(g_i − g_j)` is the numerical risk (division by a cumulative decay in eq. 9). GLA
  solves it with log-space secondary chunking in full precision. KDA ties `a = b = k`, which drops two of the four
  secondary-chunked matrices and about three matmuls. The paper reports **~2× faster than a general DPLR kernel up to
  64k** (Fig. 2, batch 1, 16 heads). FLOPs per head (eq. 13): `6T d_h² + 3T C d_h + T C²`, the same form as the gated
  delta rule.
- Inference: chunk kernel for prefill, recurrent kernel for decoding. Fixed state `d_k × d_v` per head (§6.3).

## 2. Kimi Linear's hybrid design

- **Layerwise 3:1 KDA : full attention**, uniform interleave. The paper chose layerwise over head-wise mixing for
  infrastructure simplicity and training stability (§4).
- Global layers are **MLA (Multi-head Latent Attention) with NoPE**: no positional encoding at all. KDA is "the primary
  position-aware operator" (§4, §6.1). Their argument: the gated delta rule is a data-dependent multiplicative position
  encoding (a relaxed RoPE). Channel-wise decay restores RoPE's per-dimension "frequency" diversity that a scalar GDN
  decay lacks. Practical reasons they give: NoPE MLA converts to pure MQA at inference, and long-context extension needs
  no RoPE base tuning or YaRN.
- Released checkpoint (config, *reported*): 27 layers, hidden 2304; KDA at 20 layers, MLA at 7 (layers 4, 8, 12, 16, 20,
  24, 27). So the last block is also global and the real ratio is 20:7, not exactly 3:1. MLA `kv_lora_rank` 512,
  `qk_nope_head_dim` 128. A `qk_rope_head_dim: 64` field exists, but `modeling_kimi.py` asserts `use_nope` and never
  rotates. MoE 256 experts, 8 routed + 1 shared, first layer dense; 48B total / 3B active. Short conv size 4.
- **Compared with ours:**

| | Kimi Linear | Qwen3.8-Next (ledger C24) | Our E1 |
|---|---|---|---|
| Linear mixer | KDA (channel-wise decay) | GDN (head-wise) | GDN (head-wise) |
| Pattern | 3 KDA : 1 global | 3 GDN : 1 global | `swa, gdn, gdn, swa, gdn, attn`: 3 GDN, 2 SWA, 1 full |
| Global share | 1/4 (7/27 in the release) | 1/4 | 1/6 full, plus 2/6 local softmax |
| Positions in global layer | **NoPE** | (see their paper) | RoPE |
| Positions elsewhere | only KDA (+ short conv) | — | RoPE in SWA; GDN + short conv |
| Output gate | low-rank **sigmoid** | sigmoid | full-rank **SiLU** |

## 3. Reported results (Kimi Linear paper; none verified here)

**Fair comparison set:** MLA vs GDN-H vs Kimi Linear, all 48B-A3B MoE, same recipe (1.4T tokens, 4k context, MuonClip,
WSD, LR 1.1e-3, 32M-token batch). GDN-H also uses the sigmoid output gate (§5.2). The paper does not say whether GDN-H's
global layers are NoPE.

| Benchmark (1.4T) | MLA | GDN-H | Kimi Linear |
|---|---|---|---|
| Base: MMLU-Pro | 47.2 | 47.9 | **51.0** |
| Base: MMLU | 71.6 | 72.2 | **73.8** |
| Base: BBH | 71.6 | 70.6 | **72.9** |
| Base: GSM8K | 83.7 | 81.7 | **83.9** |
| Base: EvalPlus | 59.5 | **63.1** | 60.2 |
| SFT: MMLU-Pro | 65.7 | 64.8 | **67.4** |
| SFT: GPQA-Diamond (avg@8) | 57.1 | 58.6 | **62.1** |
| SFT: LiveCodeBench v6 | 25.1 | 25.4 | **26.0** |
| SFT: EvalPlus | **62.6** | 62.5 | 61.0 |
| SFT: MATH500 | 80.8 | **83.0** | 81.2 |

**Long context (Table 5, 128k)**, with the RoPE ablation:

| | RULER | MRCR | HELMET-ICL | LongBench V2 | Frames | RepoQA | LCA lib | LCA commit | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| MLA | 81.3 | 22.6 | 88.0 | **36.1** | **60.5** | 63.0 | 32.8 | **33.2** | 52.2 |
| GDN-H | 80.5 | 23.9 | 85.5 | 32.6 | 58.7 | 63.0 | 34.7 | 30.5 | 51.2 |
| Kimi Linear (RoPE) | 78.8 | 22.0 | 88.0 | 35.4 | 59.9 | 66.5 | 31.3 | 32.5 | 51.8 |
| Kimi Linear (NoPE) | **84.3** | **29.6** | **90.0** | 35.0 | 58.8 | **68.5** | **37.1** | 32.7 | **54.5** |

The paper says the RoPE variant "attains similar scores on short-context tasks". It gives that only in words, with no
table for it.

**Ablations (Table 1)**: 16 heads, 16 layers, same FLOPs; perplexity, train / validation (validation is on a shifted
distribution):

| Variant | Train PPL | Val PPL |
|---|---|---|
| **3:1 (default)** | **9.23** | **5.65** |
| 0:1 (full attention only) | 9.45 | 5.77 |
| 1:1 | 9.29 | 5.66 |
| 7:1 | 9.23 | 5.70 |
| 15:1 | 9.34 | 5.82 |
| no output gate | 9.25 | 5.67 |
| swish (SiLU) output gate | 9.43 | 5.81 |
| no short conv | 9.29 | 5.70 |

- **There is no isolated "channel-wise vs scalar decay" perplexity ablation.** The evidence for channel-wise gating is
  (a) Kimi Linear vs GDN-H at 1.4T (the tables above) and (b) synthetic tasks (Fig. 4). The synthetic setup is 2 layers,
  2 heads, head dim 128, ≤ 20k steps, LR grid {5e-5, 1e-4, 5e-4, 1e-3}, best curve shown. On Palindrome, MQAR and Stack,
  KDA has the highest accuracy across lengths 256–2048 and converges faster than GDN on Palindrome and MQAR. Mamba-2
  fails all three. The numbers are only in a figure, so I don't quote any.
- **Scaling law (Fig. 5)**: 653M–1.7B active params, 38.8B–128B tokens. MLA `L = 2.3092·C^−0.0536`, Kimi
  `L = 2.2879·C^−0.0527`: "~1.16× computational efficiency".
- **Efficiency**: "KV cache usage reduced by up to 75%" (3 of 4 layers have no KV cache). Prefill is about the same as
  GDN-H at all lengths and as MLA at 4k–16k, then **2.3× at 512k and 2.9× at 1M** vs MLA (Fig. 7a, batch 1). Decoding:
  TPOT **1.84 ms vs 11.48 ms (6.3×) at 1M** (Fig. 1b), "6× decoding throughput at 1M". **Caveat:** §6.3 calls 6.3× a
  *theoretical* speedup from reallocating the freed memory to larger batches, and the same section reads Fig. 7b as
  "2.3× at 1M". So the paper is internally inconsistent about the batch-1 decode speedup.
- **RL** (Fig. 6, curves only): with identical RLVR settings, Kimi Linear's training and test accuracy (MATH500,
  AIME 2025) rise faster than MLA's.
- **Relevance of scale:** all of this is at 48B-A3B MoE, 1.4T tokens. We train 13–14M dense models on 200M tokens.
  None of these numbers transfers; they only motivate checks.

## 4. Kernels on our stack (fla 0.6.0, Blackwell cc 12.1, ARM64)

**Exists: yes.** fla 0.6.0 ships `fla/ops/kda/` (the paper's footnote points to the same path upstream).

```python
from fla.ops.kda import chunk_kda, fused_recurrent_kda        # ops
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate    # gate helpers
from fla.ops.kda.naive import naive_recurrent_kda, naive_chunk_kda   # pure-torch references
from fla.layers.kda import KimiDeltaAttention                   # full layer (fla.models.kda too)

chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
          use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
          allow_neg_eigval=False, safe_gate=False, lower_bound=None, disable_recompute=False,
          return_intermediate_states=False, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
          cp_context=None, use_graph=False, max_num_seqs=None, **kwargs)   # kwargs: A_log, dt_bias, chunk_size (32|64)
fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
          output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
          use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, lower_bound=None, state_v_first=False,
          cu_seqlens=None, **kwargs)
```

Shapes: `q, k [B,T,H,K]`, `v [B,T,HV,V]` (GVA if HV > H), **`g [B,T,HV,K]` in log space**, `beta [B,T,HV]`;
`K ≤ 256`; `initial_state` must be float32. With `use_gate_in_kernel=True`, `g` is the raw gate input and the kernel
computes `−exp(A_log[h])·softplus(g + dt_bias)` (A_log `[HV]`, dt_bias `[HV·K]`) with gradients to both. With
`safe_gate=True` plus `lower_bound ∈ [−5, 0)`, the gate becomes `lower_bound·sigmoid(...)` and enables an M=16
tensor-core path. Also present: a FlashKDA CUTLASS backend (inference only, bf16, K = V = 128, needs the `flash_kda`
package), which doesn't apply at our K=64; `fla/ops/precond_kda`; and `fla/layers/gdn2.py` (not investigated).
fla's layer and Kimi's HF code switch to `fused_recurrent_kda` when `q_len ≤ 64` at inference.

**Verified here (CPU, math only):** our float32 reference loop (in `experiments/checks/sq_kda_kernel.py`) matches fla's
`naive_recurrent_kda` (max|Δo| 2.2e-8, max|ΔS| 6.0e-8) and `naive_chunk_kda` (4.4e-7 / 4.0e-6) at B=2, T=128, H=3,
K=V=32 with strong per-channel decay. With `g` tied across channels it equals the GDN reference in
`tests/test_gdn_kernels.py` exactly (0.0). CPU job `kda-cpu-ref-20261122-165206` (log under
`~/fusiondance-data/run/cpu/`; the script was a scratch file outside the repo). This checks the reference, **not the
GPU kernels**.

**GPU kernel check:** `experiments/checks/sq_kda_kernel.py` (SQ50). Prediction and pass criterion are constants at the
top, written before any run. It checks:
- `chunk_kda` forward and backward vs the reference, fp32 and bf16, in three regimes: mild decay; strong decay down to
  −5; the in-kernel gate with gradients to `A_log` and `dt_bias`;
- tied-gate `chunk_kda` vs `chunk_gated_delta_rule`;
- `fused_recurrent_kda` in grad mode, in no-grad mode, and as chunk prefill → 1-token steps carrying the state;
- GDN's recurrent kernel, re-run beside it (C2);
- a reported-only fallback that decodes with the chunk kernels at T=1;
- speed vs `chunk_gated_delta_rule` at B=16, T=1024, H=6, K=V=64, plus a 1-token decode step.

**Run status: attempted once and timed out with no result (§6). C70–C72 are unverified.**

## 5. What it can do for us: ranked, cheap experiments

All are tiny (d ≈ 128–256), at most ~15 min of GPU each, and change one ingredient at a time. Parameter numbers are
*computed* from `gogeta/model.py`'s layer shapes. Nothing here has been run. **No edits to `gogeta/model.py` were
made.**

**Computed parameter deltas per mixer layer.** At d=384, H=6, hd=64 (E1 size), replacing GDN's `a: D→H` (2,304
params) with KDA's low-rank `f: D→hd→H·hd` (49,152) and a per-channel `dt_bias` (+378) adds **+47,226 per layer**,
i.e. +141,678 (+1.3%) on E1's three GDN layers. Kimi's low-rank sigmoid output gate (`D→hd→D` + bias = 49,536)
instead of our full-rank SiLU gate (147,456) saves 97,920 per layer. A "Kimi-faithful" KDA layer is therefore
**50,694 params smaller** than our GDN layer. At d=256, H=4: KDA's gate adds +31,996 per layer.

### ★ SQ50 (top pick 1): run the KDA kernel check (prerequisite)
- **Change:** none to the model; run `experiments/checks/sq_kda_kernel.py` when the GPU is free.
- **Metric and pass criterion:** in the script.
- **Why first:** everything below needs `chunk_kda` to be correct on cc 12.1. And if `fused_recurrent_kda` works where
  GDN's NaNs, it gives us a decode path even for our **existing GDN checkpoints**: GDN = KDA with tied `g`, so expand
  the scalar `g` over K and call the KDA decode kernel. That route needs no retraining (SQ53).
- **Cost:** ~1–5 min, mostly autotuning.

### ★ SQ51 (top pick 2): KDA vs GDN as the mixer in a small E1-style model
- **Change (describe, don't edit):** add a block kind `"kda"` to `Block`. Its mixer = `GatedDeltaNet` with
  - `self.a = nn.Linear(D, H)` replaced by `f_down = nn.Linear(D, hd)` and `f_up = nn.Linear(hd, H*hd)`, full precision
    like `a`;
  - `dt_bias` of shape `[H*hd]` (same log-uniform init per channel);
  - `A_log` unchanged (per head);
  - `g = -exp(A_log).view(H,1) * softplus(f_up(f_down(x)).view(B,T,H,hd) + dt_bias.view(H,hd))`;
  - `self._kernel = fla.ops.kda.chunk_kda`, called with the same q/k/v/β/scale (or `use_gate_in_kernel=True` with the
    raw input, `A_log`, `dt_bias`).

  Everything else is identical, including the SiLU full-rank output gate: one ingredient. `flops_per_token` adds
  `2·(D·hd + hd·H·hd)` for the gate per KDA layer.
- **Setup:** `vocab 8192, d=256, H=4, mlp_hidden=768`, `prelude = [swa, X, X, swa, X, attn]` with X = gdn | kda,
  T=1024, batch 16, lr 3e-3, 20M tokens of `code-py-v1`.
  - Three arms: GDN; KDA with rank hd = 64 (+96k params, +~1.8% non-emb); KDA with rank 16 (≈ param-matched, +~22k).
  - 2 seeds each. Report params, analytic FLOPs and tok/s (so this is matched *params* for the rank-16 arm, and
    matched *tokens* with compute reported).
- **Metric:** final val loss (fixed 128-sequence val subset). Also loss split by position (< 512 vs ≥ 512), because
  the claimed benefit is better memory management over distance.
- **Prediction:** KDA-64 is 0.3–1.5% lower than GDN, with most of the gap at positions ≥ 512; KDA-16 is in between.
  KDA tok/s is 0.8–1.0× GDN's.
- **Pass:** "KDA helps at our scale" if KDA-64 < GDN on both seeds and the mean Δ ≤ −0.5%. Null if |Δ| < 0.3% or the
  sign flips across seeds. Refuted if KDA is worse on both seeds by > 0.3%. (Our seed noise is unmeasured: the
  replication runs were killed. So a 2-seed spread is part of the result.)
- **Cost:** ~6 runs × ~2–3 min (predicted from E1's 86–100k tok/s at d=384).

### ★ SQ52 (top pick 3): NoPE vs RoPE in our single full-attention layer
- **Change:** give `Attention` a `rope: bool`. Add a config field `attn_nope: bool` that turns RoPE off **only** for
  kind `"attn"`; SWA keeps RoPE. Zero params, identical FLOPs, so the comparison is exactly matched.
- **Setup:** the small E1 layout from SQ51 with X = gdn. Train at **T=512** (`max_seq_len=2048` so the RoPE buffer
  covers eval), 20M tokens, 2 seeds per arm.
- **Metric:** val loss at T=512 (short), and per-position loss when evaluating at T=2048 (positions 512–2047 =
  extrapolation). Kimi's claim is mainly about long context and extension.
- **Prediction:** short loss within ±0.5%. Beyond 512 the RoPE model's loss rises (a known failure of unseen RoPE
  phases); the NoPE model rises less, because SWA (window 256) and GDN are length-agnostic.
- **Pass:** NoPE's mean loss on positions 512–2047 is ≥ 2% lower than RoPE's on both seeds, with the short-context loss
  no more than 0.5% worse. Refuted if NoPE is worse at 2048 or > 1% worse at 512.
- **Note:** this tests "NoPE global + position-aware linear layers" in *our* stack (GDN, not KDA). A KDA arm can follow
  SQ51.

### SQ53: a decode path through KDA's recurrent kernel (our RL / speculative-decoding blocker)
- **Condition:** SQ50 says `fused_recurrent_kda` (or the T=1 chunk fallback) is finite and within tolerance.
- **Change:** an inference-only `step()` on GDN/KDA blocks. It carries `(conv_state, S)` and calls `fused_recurrent_kda`
  with `g` expanded to `[B,1,H,hd]` for GDN blocks (exact, by the tied-gate identity). Attention and SWA layers get a
  KV cache. The model gets a `generate()` that prefills with the chunk kernels and then steps.
- **Metric:**
  - greedy-output **token identity** vs "re-run the full forward each token" on 20 prompts × 128 new tokens with the
    E1 checkpoint;
  - tokens/s at batch 1 and 16;
  - peak memory vs generated length.
- **Prediction:** identical tokens on ≥ 19/20 prompts (bf16 ties can flip an argmax), ≥ 5× tokens/s at 128 new tokens.
- **Pass:** ≥ 19/20 identical and ≥ 3× faster.
- **Why it matters:** it would unblock C19 (speculative decoding) and the GRPO/TRL plan, **without retraining** our
  GDN models.
- **Ranking:** top pick if SQ50's C71 passes.

### SQ54: channel-wise gating inside the looped core (stability / convergence)
- **Change:** K5's `ii_gdn_looped_x4` with `core=["kda"]`, plus a single-pass `["kda"]`; K5 task and settings
  (parity, T=64, d=128, H=2, 3000 steps). Then a K6-style convergence probe on a tiny looped LM (d=128, `core=[kda]`,
  r ∈ {2, 4, 8, 12}).
- **Metric:** parity accuracy on positions 32–63; `‖h_k − h_{k−1}‖/‖h_k‖` at k=12.
- **Prediction:** **no material gain on parity.** Finer forgetting doesn't add negative eigenvalues: β stays in (0, 1)
  and decays in (0, 1]. The parity limit is about eigenvalue sign (Grazzi et al., the paper's ref [27]), so single-pass
  KDA ≈ GDN (81.5% in K5, ±5 pts) and looped ≈ looped (97.0%). Convergence: no change beyond noise.
- **Pass:** KDA ≥ GDN + 5 pts in the same arm, or the k=12 relative change drops below 0.05 (K6 measured 0.085 for E2).
  The more promising variant is KDA + `allow_neg_eigval=True` (fla supports it for KDA too), folded into SQ2.

### SQ55: Kimi's output gate (low-rank sigmoid) vs our full-rank SiLU gate
- **Change:** in `GatedDeltaNet`, `F.silu(self.gate(x))` becomes `torch.sigmoid(g_up(g_down(x)))` (rank hd, bias on
  `g_up`). **Saves 97,920 params per layer at d=384.**
- **Overlap:** SQ8 (Qwen3.8-Next's sigmoid gate). Run it as SQ8's second arm: full-rank sigmoid vs low-rank sigmoid
  vs SiLU.
- **Prediction:** sigmoid ≤ SiLU in loss (Kimi Table 1 reports val PPL 5.65 sigmoid vs 5.81 swish); low-rank costs
  < 0.3%.
- **Pass:** low-rank sigmoid ≤ SiLU + 0.3% at fewer params.

### SQ56: bounded "safe" gate (`lower_bound=−5`, `safe_gate=True`)
Kernel speed (M=16 tensor-core path) and loss vs the softplus gate. A low priority: a speed knob, not a quality claim.

### SQ57: "Kimi-style" small model, after SQ51/SQ52/SQ9
`[kda, kda, kda, attn(NoPE)] ×` (no SWA) vs E1-small. This changes several ingredients at once, so it's only
interpretable after the single-ingredient results. It also overlaps SQ9 (pure 3:1).

### SQ58: chunk-kernel T=1 decoding as a stopgap for GDN (C2)
Reported by SQ50's fallback rows: `chunk_gated_delta_rule(..., initial_state=S, output_final_state=True)` called one
token at a time. If it's correct, SQ53 can start on GDN today, even if both recurrent kernels NaN (slower per step, but
correct).

### SQ59: synthetic replication of Fig. 4 at our scale
MQAR, palindrome and stack; 2 layers, d=128, GDN vs KDA. **Blocked on a working MQAR harness:** K3 was inconclusive
(attention only reached 8.6%). Redesign K3 first (the paper's settings: head dim 128, LR grid, up to 20k steps). Low
priority.

**Ranking:** SQ50 → SQ51 → SQ52 (top 3); SQ53 becomes #2 if SQ50's decode check passes. Then SQ55 (cheap, saves
params), SQ58, SQ54, SQ57, SQ56, SQ59.

## 6. Kernel check outcome

**Not completed. No GPU result exists.**

- **16:5x:** `status.sh` showed the slot free and no GPU process, so I started the check under `timeout 300`.
- It was killed at 300 s with **no output**, and no `results/sq_kda_kernel.json` was written. stdout was block-buffered
  and lost on the kill; that's fixed now with line buffering.
- While it ran, other agents' GPU checks (`sq_06_rotation_parity.py`, `K7-parity`) started, so the GPU was shared.
  `sq_06` was still on the GPU afterwards, so I did not retry.
- Likely cause: first-call Triton autotuning of the many KDA kernels, slowed further by the contention. **Unconfirmed.**
- **Next:** rerun alone on an idle GPU, with a 600 s limit for the first (autotune-populating) run. If it still exceeds
  5 min, shrink T to 128 and skip the bf16 in-kernel-gate regime.
- C70–C72 stay `UNVERIFIED`.

## Papers to add to `refs/` (via `refs/fetch_refs.py`, not edited by me)

Cited by Kimi Linear and directly relevant to SQ50–SQ59:
- arXiv **2505.06708**, Qiu et al., *Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free* (the
  sigmoid output gate, SQ55/SQ8).
- arXiv **2504.08719**, *SWAN-GPT* (interleaved RoPE + NoPE layers, SQ52).
- arXiv **2501.18795**, *Rope to Nope and Back Again: A New Hybrid Attention Strategy* (SQ52).
- arXiv **2506.02475**, *Comba* (the WY form KDA's chunk algorithm follows).
- arXiv **2505.16381**, *PaTH Attention* (accumulated-Householder position encoding, the "GDN as PE" argument).
- arXiv **2502.16982**, *Muon is Scalable for LLM Training* (Moonlight backbone; relevant to SQ11).
- Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR 2025): arXiv
  **2411.12537** (ID from memory, **check before fetching**). Relevant to SQ2/SQ54.
- Optional: Gated Linear Attention (Yang et al., ICML 2024, arXiv **2312.06635**, ID from memory, **check**), the
  channel-wise-decay ancestor.
