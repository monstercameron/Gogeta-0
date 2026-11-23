# Research: a programming-specialised Mixture-of-Experts (Code-MoE) on the FusionDance backbone

*Research subagent, 2026-11-22. Scope: the human's Code-MoE proposal (8 software-dimension experts, top-2 routing
inside the recurrent loop, router input = hidden state + loop index + task context, gated merge, supervised or hybrid
specialisation, stages 1-4). Labels follow AGENTS.md §0: **reported by <paper>** (not checked here), **computed**
(from `gogeta/model.py` shapes), **measured** (on disk or from a CPU read of our data), **predicted**, **speculation**.
Nothing here ran on a GPU. The draft experiment is `experiments/design/code-moe-draft.md` (E60, SQ254-SQ261).*

## 0. Bottom line

1. **MoE at our scale: plausible but small, and mostly a capacity effect.** The smallest published looped or
   shared-layer MoE results sit at 44M params (MoEUT) and 5e16 FLOPs (Looped-MoE). Our 200M-token E1 run is
   about 1.8e16 FLOPs (computed: 91.0M FLOPs/token x 2e8), so we are below every regime tested. MoEUT's reported gains at
   matched *total* params are about 0.5-1.2% in loss (computed from its perplexities). C197 (v0.4.80) says unique parameters are our lever. So the first test is
   simple: does an MoE FFN buy part of C197's gain **without** C197's 2x FLOPs?
2. **Semantic specialisation is weakly supported in the literature, and close to unmeasurable at 14M on our data.**
   Learned routers specialise mostly on **token identity and syntax**, not topic: OpenMoE (by token ID, fixed early),
   Mixtral (no domain specialisation), Fan et al. (token-level routing is syntactic, sequence-level weakly topical),
   and the "Standing Committee" (a fixed coalition of experts takes most routing mass in every domain). On code-py-v1 (measured below), 4 of the human's 8 dimensions have essentially no training signal.
3. **Encouraging expertise is not guaranteeing it, and a router can "specialise" without helping.** Fan et al. report
   that a **frozen random router performs about as well as a learned one**. So any Code-MoE gain must be compared
   against a frozen-router MoE before it is credited to routing, let alone to semantics.
4. **The loop angle is the most interesting falsifiable piece.** Looped-MoE (2605.09165) reports that sparse experts
   fix the scaling deficit of looped models, because 25-53% of tokens get a *disjoint* expert set on the second pass.
   Our E2 is 2.6% worse than its unlooped twin E9 (C6, v0.4.79). "Does MoE rescue our loop?" is therefore a
   one-interaction test, and the cheap way to ask the proposal's "different specialists on successive passes".
5. **Recommended design:** Stage 1, an unsupervised 4x512 top-2 MoE on E1 at matched active FLOPs, against two dense
   controls (E1 itself; E1 with SwiGLU 2048 at matched *total* params) and a frozen-router MoE. Stage 2, the 2x2
   {E9, E2} x {dense, MoE}. Stage 3 (semantic supervision) only if learned routing beats frozen routing. About
   6.7 GPU-h, about $4, on a rented RTX 5090.

## 1. Literature (arXiv ids; all numbers are reported by the cited paper and were not checked here)

### 1.1 MoE for code models
- **DeepSeek-Coder-V2 (2406.11931)** has the largest code-model evidence. It is a DeepSeekMoE (2401.06066) model,
  16B/2.4B-active and 236B/21B-active, continued from DeepSeek-V2 on 6T more tokens. The experts are **generic**
  (fine-grained plus shared experts), not code-dimension experts. The code win comes from data and scale; the paper
  shows no expert semantics.
- **Qwen MoE models** (Qwen2 2407.10671: 57B-A14B; Qwen3 2505.09388, already in refs: 30B-A3B, 235B-A22B) are also
  generic MoE.
- **MultiPL-MoE (2508.19268, EMNLP Findings 2025):** upcycled token-level experts plus **segment-level** experts
  (sliding-window segments, expert-choice routing) to capture programming-language syntax. Some experts prefer
  particular languages.

  Caveat: expert-choice routing over a window that contains future tokens is **non-causal** for pretraining (§3,
  item 6).
- **Takeaway:** no production code model uses code-dimension experts. "Code-specialised" in practice means generic
  MoE plus code data.

### 1.2 MoE plus recurrent, universal or looped depth
- **SUT (2310.07096):** a Universal Transformer with MoE FFN and MoE attention, plus stick-breaking halting. It reports
  WMT'14 parity with half the compute and params, and gains on formal tasks (Logical Inference, CFQ).
- **MoEUT (2405.16039):** shared layer groups (G = 2 below 300M), σ-MoE FFN with hundreds of small experts, and
  SwitchHead attention. At **matched total params** it reports C4 perplexity 18.30 vs 18.97 at 44M (computed: about
  1.2% lower loss), 13.24 vs 13.40 at 244M, and 10.90 vs 11.15 at 1B. Downstream gains are marginal (a tie at 319M).
  It also trains on The Stack (code) at 243M and 727M. Experts are reused across layers, often over a contiguous
  range; early-layer routing follows token identity and per-token expert diversity peaks mid-depth. Balancing is a
  **within-sequence entropy** loss (0.01), because batch-level balancing was unstable.
- **Looped-MoE, "Sparse Layers are Critical to Scaling Looped LMs" (2605.09165, 2026):** Base / Looped / MoE /
  Looped-MoE at matched *active* params; isoFLOP points 5e16 to 1e18; E = 8, k = 2, load-balance loss plus z-loss.
  Test loss: MoE < Looped-MoE < Base < Looped (looping alone hurts; experts recover the gap); downstream at 1e18:
  Looped-MoE 39.6, Base 38.7, Looped 37.4. **Routing divergence:** 25-53% of tokens get a disjoint expert set on
  pass 2, only 4-14% an identical one. No analysis of what experts specialise in.
- **Mixture-of-Recursions (2507.10524, in refs)** and **Mixture-of-Depths (2404.02258)** route *depth*, not experts:
  per-token recursion counts or layer skipping. MoR is the natural partner for adaptive loops, but a different axis.
- **Takeaway:** MoE in a shared-weight loop is the best-supported part of the proposal. Per-pass routing differs
  **without** an explicit loop-index input, because the hidden state changes between passes. Whether a loop index
  helps is unmeasured.

### 1.3 Small-scale MoE: collapse, balancing, and does it help at 10-30M?
- **Classical recipes:** Switch (2101.03961: top-1 routing, load-balance loss α·E·Σ f_e P_e, α = 0.01); ST-MoE
  (2202.08906: router z-loss); DeepSeek's auxiliary-loss-free bias balancing (2408.15664); expert-choice routing
  (2202.09368: balanced by construction, non-causal at token level).
- **Collapse:** X-MoE (2204.09179) describes representation collapse of the routing space. OpenMoE (2402.01739)
  reports that routing saturates early and that tokens late in a sequence get dropped under capacity limits ("drop
  towards the end").
- **Hash Layers (2106.04426):** fixed hash routing by token id is competitive with learned routing.
- **Fan, Messmer, Jaggi (2402.13089, GPT-2 small, 124M):** at equal total params, only token-level, layer-wise top-2
  routing beat the dense model, and a frozen random router was about as good as a learned one.
- **Scaling-law papers** (2202.01169 Clark et al.; 2402.07871 fine-grained MoE) fit MoE gains against active and
  total params down to about 1e8 params. My reading is that the relative routing gain shrinks with base size, which
  favours small models, but this was not re-read here.
- **σ-MoE (2310.10837):** competitive with dense Transformer-XL at matched params on WikiText-103 and enwik8, at two
  scales.
- **Takeaway:** no source tests below about 44M params or 5e16 FLOPs. At matched *active* compute MoE should win at
  our scale, because it adds params and C197 says params are the lever. At matched *total* params dense usually ties
  or wins at more FLOPs, but our binding cost is wall-clock (§3, item 9). Collapse is a known, cheap-to-detect risk.

### 1.4 Are experts semantically specialised?
- **Mixtral (2401.04088 §5):** no domain specialisation (math, biology, code) visible; consecutive tokens often share
  experts; routing tracks syntax.
- **OpenMoE (2402.01739):** context-independent, token-ID specialisation, learned early and then frozen.
- **OLMoE (2409.02060):** from-scratch training shows domain preferences (code vs papers) and vocabulary
  specialisation, stronger in later layers; experts are rarely co-activated.
- **The Expert Strikes Back (2604.02178):** experts are "fine-grained task experts" (linguistic or semantic
  operations), neither domain specialists nor plain token processors. Code concepts are probed (is_function_def,
  is_loop, ...), but the paper reports **no expert knockout**, only logit attribution.
- **Standing Committee (2601.03425, ACL 2026):** a domain-invariant coalition of experts takes most routing mass; it
  handles syntax and reasoning structure, while peripheral experts hold domain knowledge.
- **Takeaway:** at token level, expect **lexical/syntactic** specialisation, mostly explained by token identity;
  "types", "data flow" or "security" experts are not reported anywhere at any scale. Any specialisation metric must
  **condition on token identity**, e.g. I(expert; label | token id), or a hash router "specialises" too. The
  proposal's knockout test is the right causal check, and nobody has published it for code categories.

### 1.5 Supervised, domain-routed and modular experts
- **DEMix (2108.05036):** one FFN expert per domain, routed by the document's domain label at training time, and a
  posterior mixture at test time. It helps out-of-domain and allows adding or removing domains.
- **Branch-Train-Merge (2208.03306), c-BTM (2303.14177, unsupervised domain clusters), Branch-Train-MiX
  (2403.07816):** train domain experts separately, then merge into an MoE with a learned router. Specialisation is
  **imposed by data partition**, at document level.
- **Task-MoE (2110.03742):** task- or language-level routing for MT; it extracts small sub-networks for deployment.
- **Recent:** an expert-router coupling auxiliary loss (2512.23447) and teacher-guided routing in vision MoE
  (2604.21330). No LLM paper found that compares token-level *supervised* routing (labels from ast or a type checker)
  against learned routing in pretraining.
- **Takeaway:** supervised specialisation works when the label is **document-level and the domains really differ**
  (DEMix, BTX). Token-level "dimension" labels inside one Python corpus are untested. That makes it a genuine
  research question, but with a low prior at 14M.

### 1.6 Region- and segment-level routing
- **Lory (2405.03133):** fully differentiable expert merging, routed per fixed-length segment with **the previous
  segment deciding**, which keeps it causal. Similarity-based batching makes experts domain-specialised without
  labels. It reports +13.9% perplexity gain over dense at 150B tokens and 30B/1.5B-active.
- **Fan et al. (2402.13089):** sequence-level routing gives weak topic specialisation; token-level gives syntactic.
- **MultiPL-MoE (2508.19268):** segment-level expert choice for syntax (non-causal as specified).
- **Takeaway:** region or program-level routing is the only route to non-lexical specialisation in the literature.
  To stay causal it must be **lagged** (previous segment, or the prefix mean), never "this region".

## 2. What our data offers the eight dimensions (measured, CPU, code-py-v1 val, 4,000 docs)

Measured with a throwaway CPU script over `docs_val.jsonl`, not committed; SQ254 makes it a reproducible check.
Train is the same distribution (random split, K9).
- **Sources:** algorithmic_corpus 2,556 docs, synthetic_code_snippet 1,444. Mean doc length 984 chars.
- **Share of docs containing each marker:** `def` 100%, `assert` 64.2%, `->` 17.5%, `import` 17.6%, `typing` 6.8%,
  `thread` 0.03%, `eval(` 0.03%; `async`, `await`, `multiprocessing`, `Lock(`, `subprocess`, `hashlib`, `pickle`,
  `open(`: **0%**.
- **Parsing:** `ast.parse` succeeds on **36.1%** (all 1,444 synthetic snippets, **0 of 2,556** algorithmic docs,
  which are a prose task statement, then asserts, then code); `tokenize` on **97.4%**.
- **Consequences:**
  - **Memory & concurrency, Security and APIs have essentially no signal**: their experts would be starved, and their
    knockout tests would have no category tokens. **Types** is thin (annotations in about 17% of docs; no type
    checker runs on 64% of docs). **Testing** (asserts) and **Syntax** are well covered.
  - The corpus has a natural **three-region structure** (problem prose, tests, solution code) that region routing
    could find and that `tokenize` plus line prefixes can label.
  - A realistic label set has 4 categories: (1) lexical/syntax tokens (operators, brackets, indentation, keywords);
    (2) identifiers and literals (the "data flow" proxy); (3) test regions (assert lines); (4) prose, docstrings and
    comments. "Algorithms" and "data flow" in the human's sense are not token labels we can compute reliably here.

## 3. Critique of the proposal in light of C5, C6 and C197

1. **The MoE gain will be confounded with capacity unless the controls are exact.**
   - C197 (v0.4.80, exploratory n = 1): at E2's FLOPs, 2.3-2.9x more unique params give about 7% better clean80
     bpc.
   - An MoE adds unique params at about zero extra FLOPs (computed, §4 of the draft). So arm B (unsupervised MoE)
     beating arm A (dense) is **expected**, and says nothing about expertise.
   - The two tests that matter:
     - **B vs a dense control with the same total params** (SwiGLU 2048: 18.17M vs B's 18.18M non-embedding,
       computed). That bounds how much of the capacity gain MoE captures, at what FLOPs and wall-clock.
     - **B vs a frozen-router B.** Does learned routing matter at all? Fan et al. say often not.
2. **Arms C and D (semantic and recurrent specialisation) add two ingredients at once** (labels, then per-iteration
   routing); AGENTS §5 requires one at a time. The loop ingredient belongs on the *unsupervised* MoE first (the
   Looped-MoE replication), supervision on the unlooped MoE first, and D only after both.
3. **The loop doesn't pay (C6):** E2 is 2.6% worse than E9 and 6.2% worse than E1 (1 seed each, v0.4.79). Building
   Code-MoE inside the loop stacks a new mechanism on a component that currently costs 2.1x the FLOPs and loses.
   - Recommendation: test MoE on E1 first, where the C5 tie gives a tight, 7-seed control.
   - Treat the loop as a separate 2x2 whose estimand is the interaction "MoE helps looped more than unlooped" (the
     2605.09165 claim).
4. **Is "semantic specialisation" measurable at 14M? Partly, and only in a narrow form.**
   - Measurable: whether experts are causally selective for **lexical, identifier, test and prose** categories
     beyond token identity. Use knockout Δbpc per category, normalised, and conditional MI given token id.
   - Not measurable:
     - **types, security or concurrency expertise**, for lack of signal (§2);
     - **type-error detection and execution correctness**, which floor at 13M (quick review 21:37: pass@1 can't
       resolve 13M differences).
   - "State tracking" via K5 parity is not a code eval and depends on the mixer (C18, C43), not on the FFN.
   - Primary readout: **clean80 bpc**. Secondary: per-category bpc and the knockout selectivity index.
5. **The proposed capacity arithmetic is off by 2x.**
   - A SwiGLU expert of width 512 at d = 384 has 589,824 params (computed: 3·384·512). The dense h = 1024 SwiGLU has
     1,179,648.
   - So **8x512 top-2 is 4x the dense MLP's total params**, not 2x (active params are matched).
   - 2x total at matched active is **4x512 top-2** or 8x256 top-4. The draft uses 4x512 top-2 as the primary arm and
     keeps 8x512 as the "more experts" arm (2.92x E1's non-embedding params, computed).
6. **Causality hazards the proposal doesn't mention:**
   - **Capacity-factor token dropping** makes token t's output depend on how many *later* tokens in the batch chose
     the same expert. That is a future-token leak in a causal LM, and the source of OpenMoE's "drop towards the
     end". Use **dropless** dispatch.
   - **Expert-choice routing** and "route by this code region" leak for the same reason.
   - **Labels as router inputs leak:** "is this NAME a call?" depends on the next token. Labels may be *targets* of a
     router auxiliary loss, never inputs, unless computed from tokens ≤ t only.
   - SQ261 checks all three on CPU.
7. **"Task context" has no source in pretraining.** Our corpus has two sources and no task labels. The usable
   stand-ins are a causal prefix summary (program-level routing) or a lagged segment summary (region-level, Lory).
   Both are in the draft as routing variants, not as an extra input of unknown meaning.
8. **Template leakage (the human's own caution) is real here:** 64% of docs are problem + asserts + solution
   templates, and K9 found 41.7% near-duplicate val docs.
   - Supervised "testing" experts could learn the template rather than testing.
   - Score on **clean80**, and report the memorisation gap g. More unique params can also mean more memorisation, so
     MoE's gain on near-dups vs clean must be shown separately.
9. **Wall-clock, not FLOPs, is our binding cost** (design-rigor §2: E1's 0.94x FLOPs ran at 0.60x E0's speed).
   - At d = 384, a dropless top-2 dispatch over 4-8 small experts is launch-bound. Estimate 1.3-1.8x E1's
     wall-clock (unmeasured; SQ256 records it).
   - The dense total-param control costs 1.47x FLOPs (computed) and runs fast dense matmuls. It may win on loss per
     GPU-minute even if MoE wins on loss per FLOP. Report both.

## 4. Recommended minimal, cheap, falsifiable design (details and numbers in the draft)

- **Stage 1 (E60a-d, about 3.1 GPU-h; the a/b/c core alone about 2.5 GPU-h): MoE as capacity on E1.** On code-py-v1, 200M tokens, clean80, RTX 5090 only,
  6 seeds per new arm; the E1 control is Phase 2b-5090's 7 seeds.
  - **E60a** = E1 with every SwiGLU replaced by a dropless 4x512 top-2 MoE (learned router, Switch balance loss 0.01,
    z-loss 1e-3).
  - **E60b** = the same with a **frozen random router** (balance loss off).
  - **E60c** = E1 with SwiGLU 2048 (dense, total-param-matched, 1.47x FLOPs).
  - **E60d** = 8x512 top-2 (more experts, 2.92x params).
  - LR by the A3 G3 rule at 20M first.
  - Decisions:
    - "MoE pays at matched FLOPs" iff E60a < E1 by more than the MDE (about 0.8% at sigma 0.5%, n = 6);
    - "routing matters" iff E60a < E60b by more than the MDE;
    - Stage 3 (semantic supervision) runs only if routing matters, or if knockout shows category selectivity beyond
      token identity.
- **Stage 2 (E60f-h, about 3.5 GPU-h): does MoE rescue the loop?** E9, E2, E9-MoE, E2-MoE (router input = h only),
  and E2-MoE plus a loop-index bias; 4 seeds each.
  - The estimand is the interaction I = (E2MoE − E2) − (E9MoE − E9) on log clean80 bpc, plus the loop-to-loop
    routing-divergence curve.
  - The loop-index bias arm isolates the "router sees the loop index" ingredient.
- **Stage 3 (conditional): supervised routing on E60a's shape.** An auxiliary cross-entropy from the router to 4
  categories (§2) with λ in {0.03, 0.1} (pilot SQ258), vs E60a. The knockout selectivity test is SQ255.
- **Stage 4 (conditional): Byte Quest.** An MoE FFN in the patched global body (SQ260), only after E13 lands and
  Stage 1 is positive.
- **Not recommended now:**
  - the 8-dimension expert set (no data for 4 of 8);
  - mypy, pyright or execution supervision (0% of algorithmic docs parse; execution evals floor at 13M);
  - expert-choice routing;
  - capacity factors.

## 5. What would change my mind

- E60a beats **both** E60b and E60c at matched wall-clock: MoE routing is a real lever at 14M, so invest in Stage 3.
- I ≤ −1% in Stage 2 (MoE helps the looped model more): the loop thesis gets a second life as "loops + sparse
  experts" (2605.09165 replicated at 1/3 of its smallest compute).
- Knockout selectivity for test or prose regions that survives conditioning on token id: region-level routing is
  worth its own E-series stage.

## 6. References added (`refs/fetch_refs.py`, new section `15-moe`; PDFs not downloaded, see the hand-back)

2406.11931 DeepSeek-Coder-V2 · 2401.06066 DeepSeekMoE · 2407.10671 Qwen2 · 2508.19268 MultiPL-MoE · 2310.07096 SUT
· 2405.16039 MoEUT · 2605.09165 Looped-MoE · 2404.02258 MoD · 2101.03961 Switch · 2202.08906 ST-MoE · 2408.15664
aux-loss-free balancing · 2202.09368 Expert Choice · 2106.04426 Hash Layers · 2204.09179 X-MoE · 2402.13089 MoE design
choices (Fan) · 2202.01169 routed scaling laws · 2402.07871 fine-grained MoE scaling · 2310.10837 σ-MoE · 2401.04088
Mixtral · 2402.01739 OpenMoE · 2409.02060 OLMoE · 2604.02178 Expert Strikes Back · 2601.03425 Standing Committee ·
2108.05036 DEMix · 2208.03306 BTM · 2303.14177 c-BTM · 2403.07816 BTX · 2110.03742 Task-MoE · 2405.03133 Lory.
(MoR 2507.10524 and Qwen3 2505.09388 were already listed.)
