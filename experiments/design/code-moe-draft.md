# Draft: E60 Code-MoE (MoE FFN on the FusionDance backbone), for the main agent to register

*Research subagent, 2026-11-22. DRAFT: not registered and not binding until the main agent commits it into
`experiments/experiments.json` and `experiments/checks/PREDICTIONS.md` as a tagged version (AGENTS §1, §6). Rationale
and literature: `notes/research-code-moe.md`. Param and FLOP numbers are **computed** analytically from
`gogeta/model.py` (SwiGLU = 3·d·h params; `flops_per_token` rule: 6 × matmul params per application plus the
attention/GDN terms, seq 1024, core counted r = 4 times). The base rows match `experiments/design/staged-table.md`.
Wall-clock and cost figures are **estimates**. Nothing ran.*

## 0. Which E id

| Range | Status |
|---|---|
| E0-E16 | in `experiments/experiments.json` (E12-E16: Byte Quest) |
| E17-E19 | **not referenced anywhere in the repo** (searched experiments/, notes/, tools/, configs/, cockpit/) |
| E20-E59 | reserved for A3 by `notes/design-rigor.md` §0 and `experiments/design/build_staged.py:12` |
| E20-E27, E30-E36, E40-E48, E50-E53 | staged and run (fleet waves 1-3b); not yet in the registry |
| E54-E59 | unused, but inside A3's reservation |
| **E60** | first id above every reservation. **Proposed.** |

E17 is also free. I chose E60 so the registry never interleaves Code-MoE with A3's reserved block; the main agent may
renumber. Arms are sub-ids E60a-E60k, as with E13a etc.

## 1. Registry entry (draft)

```json
{
  "id": "E60",
  "arc": "Code-MoE",
  "title": "Code-MoE stage 1-2: MoE FFN as capacity (E1) and as a loop rescue (E2 vs E9)",
  "question": "At matched active FLOPs, does a dropless top-2 MoE FFN (4 experts x 512) lower clean80 bpc vs E1, how much of a dense total-param-matched control's gain does it capture, does learned routing beat a frozen random router, and does MoE help the looped E2 more than its unlooped twin E9 (interaction I)?",
  "prediction": "TO REGISTER: experiments/checks/PREDICTIONS.md section 'Registered Code-MoE predictions E60' (draft in experiments/design/code-moe-draft.md §5), before any run.",
  "ingredients": ["MoE FFN 4x512 top-2 (dropless)", "Switch balance loss 0.01 + z-loss 1e-3", "frozen-router control", "dense total-param control", "loop-index router bias (stage 2)"],
  "metrics": ["clean80 doc-level bpc (primary)", "memorisation gap g", "per-category bpc (tokenize classes)", "expert load / router entropy", "loop-to-loop routing divergence", "measured tok/s"],
  "status": "planned",
  "version": null,
  "config": "experiments/staged-configs/code-moe/ (to be staged; needs the code-moe patch)",
  "results": {},
  "notes": "Human proposal 2026-11-22 (programming-specialised MoE in the recurrent loop). Semantic supervision (stage 3) and the Byte Quest combo (stage 4) are conditional on stage 1 (SQ255/SQ258/SQ260). Primary metric clean80 bpc; RTX 5090 only, never pooled with N1X runs."
}
```

## 2. Shapes, params and FLOPs (computed; d = 384, 6 heads, vocab 8192, seq 1024)

- **Building blocks:**
  - dense SwiGLU h = 1024: **1,179,648** params;
  - expert SwiGLU h_e = 512: 589,824;
  - h_e = 256: 294,912;
  - router: d·E (+ 8·E for a per-loop bias table).
- **Correction to the proposal:** **8x512 top-2 is 4.0x the dense MLP's total params**, not 2x. 4x512 top-2 is 2.0x.
  Both keep active MLP params equal to dense (2 × 512 = 1024).
- MoE replaces the SwiGLU in **every block** (E1: 6 blocks; E9/E2: 5 unique blocks), so E2 vs E9 adds the same
  unique params.

| Arm | Backbone | FFN | non-emb params | x base | train FLOPs/tok | x E1 FLOPs | est. 200M wall (5090) |
|---|---|---|---|---|---|---|---|
| E1 (control, existing 7 seeds) | E1 | dense 1024 | 11,092,068 | 1.000 | 91,027,800 | 1.000 | ~5 min |
| **E60a** | E1 | MoE 4x512 k2, learned | 18,179,172 | 1.639 | 91,083,096 | 1.001 | ~7.5 min (x1.5 dispatch, unmeasured) |
| **E60b** | E1 | MoE 4x512 k2, **frozen random router** | 18,179,172 | 1.639 | 91,083,096 | 1.001 | ~7.5 min |
| **E60c** | E1 | dense **2048** (total-param-matched to E60a, +0.05%) | 18,169,956 | 1.638 | 133,495,128 | 1.467 | ~6.5 min |
| E60d | E1 | MoE 8x512 k2 (the proposal's shape) | 32,344,164 | 2.916 | 91,138,392 | 1.001 | ~8.5 min |
| (E60d') | E1 | MoE 8x256 k4 (fine-grained, 2x total) | 18,188,388 | 1.640 | 91,138,392 | 1.001 | optional |
| E9 (control) | E9 | dense | 9,165,080 | 1.000 | 79,170,960 | 0.870 | ~4.4 min |
| **E60f** | E9 | MoE 4x512 k2 | 15,071,000 | 1.644 | 79,217,040 | 0.870 | ~6.6 min |
| E2 (control) | E2, r = 4 | dense | 9,460,760 | 1.000 | 192,793,152 | 2.118 | ~10.5 min |
| **E60g** | E2 | MoE 4x512 k2, router input h only | 15,366,680 | 1.624 | 192,922,176 | 2.119 | ~16 min |
| **E60h** | E2 | MoE 4x512 k2 + per-loop router bias | 15,366,840 | 1.624 | 192,922,176 | 2.119 | ~16 min |
| (E60k) | E2 | dense 2048 (total-param control for E60g) | 15,359,000 | 1.623 | 291,883,584 | 3.207 | optional, ~14 min |

- **Matching rules:**
  - **Matched active compute:** E60a/b/d vs E1; E60f vs E9; E60g/h vs E2. Each within 0.12% FLOPs; the router is
    the only extra.
  - **Matched total params:** E60c vs E60a (−0.05%), at 1.47x FLOPs. E60k vs E60g (optional).
  - Both are reported against FLOPs **and** measured GPU-minutes (design-rigor §2).
- The E1 control is Phase 2b-5090 (7 seeds, clean80 mean 0.26795, `experiments/checks/results/phase2b_5090.json`).
  E2 and E9 have 1 seed each at 200M on the 5090 (0.2845, 0.2773); stage 2 adds 3 seeds each.

### 2.1 Routing granularity variants (SQ257; same params as E60a, causal by construction)

| Variant | Router input for token t | Decisions per 1,024-token window | Causal because |
|---|---|---|---|
| `token` (default) | norm(h_t) | 1,024 | h_t sees tokens ≤ t |
| `seg64` (region, Lory-style) | mean of h over the **previous** 64-token segment [64⌊t/64⌋ − 64, 64⌊t/64⌋); segment 0 uses a learned start vector | 16 | only completed segments |
| `prefix` (program level) | causal running mean of h over tokens ≤ t (cumsum / count) | 1,024 (slowly varying) | prefix only |

**Forbidden variants:** route by "this region's" summary; expert-choice routing; capacity-factor dropping. Each makes
token t depend on later tokens (SQ261 tests this).

## 3. Arms, stages and seeds

- **Common settings:**
  - code-py-v1, 200M tokens, batch 32 x 1024, warmup 300, cosine to 0.1, AdamW as E1;
  - primary metric **clean80 doc-level bpc** (K8/K9 scorer, as Phase 2b);
  - RTX 5090 only;
  - same seeds across arms (common data order), so a paired-by-seed analysis is available as a secondary.
- **LR:** A3 protocol P1 G3 sub-grid {3e-3, 6e-3, 1.2e-2} at 20M tokens, seed 0, for every new arm (E60a, b, c, d,
  f, g); E60h inherits E60g's LR. Edge winners get the one-step extension rule. The selection run is never reused as
  evidence.
- **MoE defaults:**
  - top-2 softmax gating, renormalised over the selected 2 (Mixtral style);
  - Switch balance loss α = 0.01 (batch level; it is a loss, not a routing decision, so it is causal-safe);
  - router z-loss 1e-3;
  - router in fp32;
  - experts initialised like SwiGLU (up std 0.02, down std_res);
  - router weight std 0.02 and zero bias;
  - E60b's router is frozen at init (`requires_grad = False`), with balance and z-loss off.

| Stage | Runs at 200M | Seeds | Gate |
|---|---|---|---|
| 1 | E60a, E60b, E60c x 6 seeds; E60d x 4 | 0-5 (0-3) | none |
| 2 | E9, E2 + 3 seeds each (seeds 1-3); E60f, E60g, E60h x 4 | 0-3 | runs unless stage 1 shows collapse or instability (max expert load > 0.6 or a NaN) |
| 3 | supervised routing (from SQ258's best λ) on E60a's shape x 6 | 0-5 | only if D2 (learned > frozen) holds, or SQ255 shows region selectivity beyond token identity |
| 4 | MoE in E13b's patched global body (SQ260) | 3 | E13 done and D1 holds |

### 3.1 Power (sigma = per-run sd of log clean80 bpc)

- **Measured at 200M:** E1's 7 seeds span 0.2667-0.2687 (sd about 0.3%); the Phase 2b paired E1−E10 difference sd
  is 0.45%. MoE adds router noise; looped cells were the noisiest at 20M (F1 pilot: E22 1.17%, E27 1.08%, against
  0.18% for E24). **Assumption:** sigma 0.5% for unlooped MoE arms and 0.8% for looped arms.
- **MDE (80% power, two-sided α 0.05):** 2.80 · sigma · sqrt(1/n1 + 1/n2).
  - **Stage 1:** E60a (n = 6) vs E1 (n = 7) at sigma 0.5% gives **MDE 0.78%**; at 0.3%, 0.47%. E60a vs E60b or
    E60c (6 vs 6): 0.81%.
  - **Stage 2 main effect,** E60g vs E2 (4 vs 4), sigma 0.8%: **MDE 1.58%**.
  - **Stage 2 interaction** I (4 cells x 4): SE = sigma · sqrt(4/4) = 0.8%, so **MDE 2.24%**. It is **under-powered
    for |I| = 1%**, and n = 4 is a **pilot**.
- **Sizing rule (registered now, variance only, never means):** from the pilot's pooled sigma, set n per cell to
  ceil((2.80 · 2 · sigma / 0.01)²) for a 1% MDE on I. That is n = 6 at sigma 0.4% and n = 21 at sigma 0.8%. If n > 12,
  stop at the pilot and report I as INCONCLUSIVE with its MDE (as F2'' did for the factorial).

## 4. Cost (estimate; RTX 5090, about $0.5-0.7/h; E1 200M ≈ 5 min, 20M ≈ 40 s per the fleet; MoE dispatch x1.5 unmeasured)

| Item | Runs | GPU-min (est.) |
|---|---|---|
| Stage 1 LR grid, 20M | 4 arms x 3 LRs | ~25 (incl. compile) |
| Stage 1, 200M | 6 x (7.5 + 7.5 + 6.5) + 4 x 8.5 | ~163 |
| Stage 2 LR grid, 20M | 2 arms x 3 LRs (+ E60h inherits) | ~12 |
| Stage 2, 200M | E2 3 x 10.5, E9 3 x 4.4, E60f 4 x 6.6, E60g 4 x 16, E60h 4 x 16 | ~199 |
| **Stages 1 + 2** | 58 runs (18 at 20M, 40 at 200M) | **~6.7 GPU-h, about $4** (within the $30 / 12 h standing budget, §0h) |

- The cheapest decisive first experiment is the **Stage 1 core**: E60a, E60b and E60c, 6 seeds each, plus their 20M
  LR grid (9 runs). That is about 2.5 GPU-h (about $1.5), and it decides D1-D3.
- Wall factors are estimates. SQ256's first 20M runs measure tok/s, and those measurements replace every row.

## 5. Predictions and decision rules (draft for PREDICTIONS.md; written before any run)

All are on the log clean80 bpc scale, as relative differences (−1% means 1% lower = better). The E1 reference is the
7-seed Phase 2b-5090 mean.

### Stage 1
- **M1:** E60a − E1 = **−1.8%** (range −0.8% to −3.5%). P(E60a beats E1 by more than the MDE) = 0.70.
- **M2:** E60c − E1 = **−3.0%** (−1.5% to −4.5%). E60a is worse than E60c by **0.3-2.0%**: MoE captures about 40-80%
  of the dense capacity gain at 0.68x its FLOPs. P(E60a ≤ E60c) = 0.2.
- **M3:** |E60a − E60b| ≤ 0.8% (P 0.6). Learned beats frozen by more than the MDE with P 0.35 (prior: Fan et al.
  2402.13089; Hash Layers 2106.04426).
- **M4:** E60d − E60a = −0.3% to −1.5% (more experts help a little).
- **M5 (descriptive):**
  - with the balance loss, max per-expert assignment share ≤ 0.35 (4 experts; uniform 0.25) in every layer at the
    end of training;
  - no dead expert (< 2% share);
  - E60b's frozen router is near-uniform by construction (checked, not predicted).
- **M6 (descriptive):** normalised MI between E60a's top-1 expert and the 4-way category label (SQ254) ≥ 0.10 in at
  least one layer. **Conditional MI given token id ≤ 0.05 bits per token in every layer**: routing ≈ token identity,
  as in OpenMoE 2402.01739.
- **M7 (descriptive):** E60a's memorisation gap g (near-dup minus clean bpc) ≥ E1's. More params memorise more.
- **M8 (descriptive):** measured tok/s ratios E60a/E1 = 0.55-0.80 and E60c/E1 = 0.70-0.85.

**Decisions (fixed now):**
- **D1, "MoE pays at matched active FLOPs":** the Welch 95% CI of E60a − E1 lies below 0 and the effect exceeds the
  MDE. Pass gives a new claim at PARTIAL/VERIFIED by policy (n ≥ 3 seeds per arm). If |E60a − E1| < MDE, then
  **"MoE buys nothing measurable at 14M / 200M"** is recorded, and stage 3 is dropped.
- **D2, "learned routing matters":** the CI of E60a − E60b lies below 0. If not, record **"the gain is capacity, not
  routing"**. Stage 3 then runs only via the SQ255 exception (region selectivity beyond token id).
- **D3, practical efficiency:** compare E60a and E60c on clean80 vs **measured GPU-minutes**. If E60c is no worse per
  GPU-minute, MoE is a FLOP-analytic win only on this hardware, the same caveat as C12 for SWA. Say so wherever the
  result is quoted.
- **D4, relation to C197:** report E60a's gain per added unique param next to E32/E33's (C197). This is descriptive;
  no claim is changed by it.

### Stage 2 (pilot, n = 4 per cell)
- **Q1:** E60g − E2 = **−2.0%** (−0.5% to −4%).
- **Q2:** I = (E60g − E2) − (E60f − E9) = **−0.5% ± 1.0%**. P(I ≤ −1%) = 0.30, P(|I| < 1%) = 0.60, P(I ≥ +1%) = 0.10.
- **Q3:** E60g is still worse than E60f (loops still don't pay at 2.4x the FLOPs): P 0.70.
- **Q4 (descriptive):** at r = 4 eval in the core blocks, the fraction of tokens whose top-2 set is **disjoint**
  between consecutive loops is 15-55%, and the fraction with an identical set ≤ 25% (reported by 2605.09165: 25-53%
  and 4-14%).
- **Q5:** |E60h − E60g| ≤ 0.5%: the loop index adds nothing beyond h.
- **Q6:** E60g's L(r) still degrades beyond the trained range (r = 8 and 12 worse than r = 4; C7): P 0.75.

**Decisions:**
- "MoE rescues the loop" iff I's 95% CI lies entirely below 0;
- "no interaction" iff the CI lies inside (−1%, +1%);
- otherwise INCONCLUSIVE with the MDE, and the §3.1 sizing rule decides whether to top up;
- the loop-index ingredient is kept only if E60h < E60g beyond the MDE.

### Stage 3 (conditional; registered later with SQ258's numbers)
- **S1:** supervised routing at the chosen λ vs E60a: +0.0% to +1.0% (ties or costs a little).
- **S2:** router-label agreement ≥ 0.7.
- **S3:** the knockout selectivity index (SQ255) is ≥ 1.5x E60a's.
- "Supervision helps" iff it beats E60a by more than the MDE. **"Encouraged but useless specialisation"** is the
  predicted outcome: S2 and S3 met, S1 not better.

## 6. Code to write (staged as `notes/patches/code-moe.patch`, applied between campaigns)

- **`gogeta/moe.py` (new)**, `class MoEFFN(nn.Module)`:
  - **Parameters:**
    - `w_up: [E, d, 2·h_e]`, `w_down: [E, h_e, d]`;
    - `router: nn.Linear(d, E)`, fp32;
    - optional `loop_bias: [loops_max + 1, E]`, initialised to zeros; index clamped, row 0 used outside the core;
    - optional `seg_start: [d]` for `seg64`.
  - **Init:** `reset_parameters(std, std_res)`, called from `FusionDance._init_weights`, because batched params are
    not `nn.Linear` and would otherwise skip the depth-scaled init.
  - **Forward:** `forward(x, loop_idx=None) -> y`. It records `self.last_aux = {"balance", "z", "load"[E],
    "entropy"}` and, when `self.record` is set, `self.last_topk` ([B, T, k] int8) for diagnostics.
  - **Dispatch:**
    - `moe_impl="gather"` (dropless; per expert, `index_select` the assigned tokens, run SwiGLU, then `index_add_`
      the gate-weighted output);
    - `moe_impl="dense_mask"` (every expert on every token, masked by the gates: about E/k x the MLP FLOPs, but
      static shapes for `torch.compile`).

    A test asserts the two match to 1e-5 in fp32. Analytic FLOPs always count **active** params (flag, as for C12).
  - **Routing modes:** `moe_router = "learned" | "frozen"`; `moe_route_ctx = "token" | "seg" | "prefix"` with
    `moe_seg_len = 64`. **No capacity factor, no token dropping, no expert choice.**
  - **Supervised auxiliary loss (stage 3):** `moe_aux_coef` with `moe_label_map` (category to expert). The loss is
    cross-entropy of the router softmax against the mapped label, ignore index 255. Labels are targets only; the
    module never reads them as inputs.
- **`gogeta/model.py`:**
  - **`ModelConfig` fields:** `moe_experts: int = 0` (0 = dense), `moe_topk: int = 2`, `moe_hidden: int = 512`,
    `moe_router: str = "learned"`, `moe_loop_bias: bool = False`, `moe_route_ctx: str = "token"`,
    `moe_seg_len: int = 64`, `moe_impl: str = "gather"`, `moe_balance_coef: float = 0.01`,
    `moe_z_coef: float = 0.001`, `moe_aux_coef: float = 0.0`.
  - **`Block`:** `self.mlp = MoEFFN(cfg) if cfg.moe_experts else SwiGLU(cfg)`. The forward passes `loop_idx` only to
    MoE.
  - **`FusionDance.forward`:** pass `loop_idx = i + 1` in core iterations (0 for prelude and coda). When MoE is on,
    return `loss = lm + Σ_blocks mean_over_calls(α·balance + β·z [+ λ·aux])`. Log the LM part separately, so
    val_loss stays comparable.
  - **Accounting:** `param_counts()` gains `active_non_embedding`; `flops_per_token` uses active matmul params.
  - **Guards:** assert `not (ternary and moe_experts)` (BitLinear for batched experts is out of scope).
- **`gogeta/train.py`:**
  - log `moe/*` (balance, z, max load, min load, router entropy per layer) every eval;
  - for stage 3, read an optional aligned label shard (`train_labels.bin`, uint8) with the same window indices as
    tokens;
  - `--require-clean` and provenance are unchanged.
- **Bit-identical defaults (the C91/byte-patch standard):**
  - with `moe_experts = 0`, no new module, parameter or buffer is created and **no RNG is consumed**;
  - `ModelConfig.to_dict()` **omits every `moe_*` key while `moe_experts == 0`**, so existing run configs and hashes
    are unchanged;
  - the loss path is untouched, with no aux term added.
  - **Test `tests/test_moe.py`** (CPU, GDN stubbed as in K7-parity):
    1. E0, E1, E2, E3, E9 construct with identical state_dict, RNG state after construction, `to_dict()`,
       `param_counts()` and `flops_per_token()` before and after the patch;
    2. gather equals dense_mask;
    3. causality: perturb token t and assert logits at positions < t are bitwise unchanged, for token, seg and
       prefix routing, at the segment boundaries t = 63, 64, 65;
    4. the frozen router has no grad and its weights are unchanged after a step;
    5. the balance loss equals 1.0 for exactly uniform routing (Switch normalisation);
    6. the table in §2 reproduces from `param_counts()` / `flops_per_token()`.
- **Tooling (CPU; registered with the cockpit as queue items per §0e):**
  - `experiments/code_moe/label_tokens.py` (SQ254): `tokenize` plus line-prefix rules produce 4 categories, aligned
    to BPE positions via the tokenizer's offsets, written as a sha-manifested uint8 shard;
  - `experiments/checks/moe_routing_stats.py` (SQ256, SQ259): load, entropy, NMI and conditional MI given token id,
    loop-to-loop divergence;
  - `experiments/checks/moe_knockout.py` (SQ255): extends `layer_knockout.py` with a per-expert gate mask instead of
    a mixer zero, scored per category on clean80.
- **Out of scope for the PoC:**
  - "task context" (no task labels exist in pretraining; `prefix` routing is the stand-in);
  - an extra gated state update on the merged expert output (the top-k gates already weight it, and a further gate
    is a separate ingredient; see SQ10 Gated Residual);
  - mypy, pyright or execution supervision (0 of 2,556 algorithmic docs parse with `ast`; measured,
    `notes/research-code-moe.md` §2).

## 7. Side-quest rows (draft for `experiments/side-quests.md`; the main agent appends them)

| SQ254 | **Code-category token labels**: `tokenize` + line-prefix labeller (lexical/syntax, identifiers+literals, assert/test region, prose/comment) aligned to BPE positions on code-py-v1/v2, with a coverage table per category and per source, plus the share of docs touching each of the proposal's 8 dimensions | Every specialisation metric and any supervised routing needs labels; a CPU read (research note §2) already found 0% concurrency/security/subprocess markers and `ast.parse` failing on all 2,556 algorithmic val docs, so 4 of 8 proposed experts would be starved (research subagent, notes/research-code-moe.md) | `idea` | ~10 CPU-min; prerequisite for SQ255/SQ258 |
| SQ255 | **Expert knockout specialisation test**: mask each expert's gate in turn in E60a (and frozen-router E60b), score clean80 bpc per SQ254 category; selectivity = own-category Δbpc / other-category Δbpc, with a token-identity control (same tokens, other categories) | The proposal's causal test of 'expertise'; the literature reports attribution, not knockout (2604.02178), and token-id routing (2402.01739) would fake selectivity unless controlled (research subagent) | `idea` | ~3 GPU-min inference on E60 checkpoints; reuses layer_knockout.py's scorer |
| SQ256 | **Router-collapse diagnostics + MoE throughput**: E60a's shape at 20M tokens with balance coef {0, 0.001, 0.01, 0.1}, 2 seeds; max/min expert share, dead experts, router entropy, loss, and measured tok/s of gather vs dense_mask dispatch | Collapse and dispatch overhead are the two practical MoE risks at d=384; the tok/s numbers replace the x1.5 wall estimate in the E60 cost table (research subagent) | `idea` | ~10 GPU-min |
| SQ257 | **Token- vs region- vs program-level routing**: E60a with `moe_route_ctx` token / seg64 (previous segment decides, Lory 2405.03133) / causal prefix mean, 200M tokens, 3 seeds each; clean80 plus routing NMI with the prose/test/code regions | The proposal's key design question; sequence-level routing gives topic-like specialisation and token-level syntactic (2402.13089), and our corpus has a natural prose -> asserts -> solution region structure (research subagent) | `idea` | ~70 GPU-min after E60 stage 1; needs the causality test (SQ261) |
| SQ258 | **Supervised-routing aux-objective ablation (stage 3 pilot)**: router cross-entropy to SQ254 categories, lambda {0, 0.03, 0.1, 0.3}, E60a's shape, 20M tokens, 3 seeds; LM loss cost vs router-label agreement vs knockout selectivity | Tests 'encouraging expertise' cheaply before any 200M supervised arm; predicts agreement rises while LM loss ties or worsens (research subagent) | `idea` | ~12 GPU-min; needs SQ254 labels and the label-shard reader |
| SQ259 | **Loop routing divergence probe**: on E60g/E60h checkpoints, the fraction of tokens with disjoint / overlapping / identical top-2 sets between consecutive loops per core block, at r = 1..12 (inside and beyond the trained range) | Looped-MoE (2605.09165) attributes its win to 25-53% disjoint routing between loops; this checks whether our loop does the same and whether routing drifts past r = 7 (C7) (research subagent) | `idea` | ~2 GPU-min inference |
| SQ260 | **Byte Quest x MoE**: MoE 4x512 top-2 in the patched global body of E13b (P = 4) vs the dense E13b, 200M bytes, 3 seeds | Stage 4 of the proposal; only meaningful after E13 lands and E60 stage 1 passes D1 (research subagent) | `idea` | ~75 GPU-min; needs the byte patch and the code-moe patch together |
| SQ261 | **MoE causality check (CPU)**: perturbation test showing that capacity-factor token dropping, expert-choice routing and 'current region' routing leak future tokens, while dropless token / previous-segment / prefix routing do not | Justifies the dropless design and guards SQ257; a leak would silently flatter the MoE arms (research subagent) | `idea` | ~2 CPU-min; part of tests/test_moe.py |
