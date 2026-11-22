# Gogeta-0 devlog

Every entry is one commit and one annotated git tag (rules in [AGENTS.md](AGENTS.md)). Newest first.

## High-level goals

1. **FusionDance:** fuse three efficiency ideas into one causal LLM architecture, and show with **Gogeta-0** (its first model) that they stack rather than fight:
   - **Linear-time sequence mixing** (Gated DeltaNet / Mamba-style state + sliding-window attention): no quadratic cost inside the network, and only one layer whose memory grows with context.
   - **Looped depth**: one shared block applied r times, so reasoning depth comes from compute, not parameters.
   - **Ternary weights** (BitNet b1.58): ~1.6 bits per weight.
2. **Think in loops, not out loud.** Test whether latent loops can replace part of chain-of-thought, trading tokens (memory + quadratic cost) for compute.
3. **The Rival Loop.** Build critique → fix into the recurrence: per-loop supervision, a grounded critic head for halting and feedback, adversarial damage-and-repair training.
4. **Make the loops pay at inference.** Self-speculative decoding: draft with few loops, verify with many.
5. **Do it on a laptop.** Everything trains and runs locally (RTX Spark N1X via WSL). Small models, clean ablations, honest results, including failures.

**Headline metric:** quality per bit of weight memory and per FLOP, plus how memory and compute *scale with context length*.

---

## v0.4.15: Autonomy rule: never block on the human
*2026-11-22*

**What** (from the human: "don't ask me what to do, I want you to be autonomous; if you ask me while I'm sleeping we waste time"):
- **AGENTS.md §0a, "Be autonomous":** decide and proceed; thesis-direction calls are made by the agent with a written decision memo; never end a turn with a question while work remains; failures are diagnosed and fixed, not escalated.
- **The only actions not taken alone** are the safety list: downloads/installs, push/publish, deleting data, spending money. Even these **never block**: they're parked under "For the human" in `experiments/QUEUE.md` while other work continues.
- **§5b updated to match:** `needs_human` now covers the safety list only. Multi-hour GPU items and thesis decisions are autonomous.
- **Queue:** `needs_human` cleared on Q23 (loop-purpose decision memo) and E4 (full fusion, multi-hour); still parked: Q14 (eval-suite downloads/installs) and SQ7 (Mamba-3 package build). K7 and Q3 are marked running (launched from the queue at 17:25).

**Results**: `tools/queue.py validate`: ok (45 items).

---

## v0.4.14: Experiment queue + autonomous-queue rule
*2026-11-22*

**What**:
- **`experiments/queue.json`**: 45 prioritized items with dependencies, resources (GPU/CPU, minutes), `ready`/`missing`, `needs_human` and its reason, exact `launch` commands, and sources (queue subagent; reviewed by the main agent).
- **`tools/queue.py`** (stdlib): `list`, `next --gpu|--cpu [--max-minutes N]`, `show`, `set`, `add`, `move`, `validate`, `render`. It validates ids and cycles, never reorders silently, refuses a 2nd GPU or a 9th CPU item, and `next` skips `needs_human` items.
- **`experiments/QUEUE.md`**: the generated view.
- **AGENTS.md §5b:** the main agent works through the queue without asking the human, and asks only for `needs_human` items (download or install, > 3 GPU-hours in one item, deleting data, pushing, thesis-direction change) or when a result contradicts an item's premise.
- Q1 (predictions registered: v0.4.11, v0.4.13) and Q2 (`--lr` + campaign runner: v0.4.10) marked done.

**Results**: `validate`: ok (45 items, no unknown dependencies, no cycles). `next --gpu` → **K7** LR sweep (43 min); `next --cpu` → **Q3** K8 `--plan` (2 min).
**Ordering policy**: validity first (controls, tuning, seeds), then cheap inference-only diagnostics on existing checkpoints, then thesis-critical runs (E11, neg-eig in the loop, E3 at the tuned LR, loop fixes), then side quests, then long-term items.
**Needs the human (queued, not auto-run)**: Q14 eval suite tier 1 (downloads/installs), Q23 loop-purpose decision memo (thesis direction), E4 full fusion (> 3 GPU-h), SQ7 Mamba-3 kernels (install/build).
**Next**: run K7 (GPU) and Q3 (CPU) from the queue.

---

## v0.4.13: K7 and K8 registered: LR sweep and external reference
*2026-11-22*

**What**: registers the two checks that the Phase 2 campaign runs first and last. Predictions and decision rules are committed here, **before any run**.
- `experiments/checks/k7_lr_sweep.py` (applies the K7 rule and writes the next campaign; rule logic tested on 8 synthetic cases), `experiments/launch/campaigns/phase2a-k7-lr.campaign` (8 short runs) and `phase2b-replication.template`.
- `experiments/checks/k8_external_reference.py`: doc-level bits/char for our checkpoints and SmolLM2-135M. It loads only from the local cache, at the pinned revision `93efa2f`, and refuses on a mismatch.
- `experiments/checks/PREDICTIONS.md`: K7, K8, and the **replication protocol** with its comparison criteria. Cards K7 and K8.
- AGENTS.md: campaign runner and `--lr` (§5); controls are LR-tuned before use and E10 becomes the control (§6); docs table rows.
- Site regenerated (`docs/`) and `experiments/INDEX.md` regenerated from committed sources; version 0.4.13.

### K7: LR tuning for E10 (control) and E1 (best hybrid)
- **Question:** what's each model's best peak LR? Every earlier run used an untuned 3e-3.
- **Result:** not run. Phase 2a: lr ∈ {1e-3, 2e-3, 3e-3, 6e-3} × {E10, E1}, seed 0, 20M tokens, ~43 min GPU.
- **Verdict vs prediction:**
  - Prediction: E10's best is 3e-3 or 6e-3, E1's is 2e-3 or 3e-3, both curves flat near the optimum, and 1e-3 is 3–8% worse.
  - Rule: the lowest full-val loss wins (ties go to the smaller LR). If the winner is at the grid edge, extend one step (at most twice per side).
- **What it means:** the replication (E10 / E1 / E11 × 3 seeds, E9, conditional E0 and E2) then runs at the tuned LRs.
- **Card:** experiments/cards/K7.md

### K8: External sanity reference (SmolLM2-135M, bits/char)
- **Question:** are our loss numbers in a sane range against a well-known public model?
- **Result:** not run (runs after Phase 2b, ~5 min).
- **Verdict vs prediction:**
  - Prediction: SmolLM2 0.28–0.45 bits/char, ours 0.21–0.26.
  - K8a: ours ≥ 0.5 × SmolLM2, else suspect train/val leakage.
  - K8b: doc-level bits/char within 5% of the training-time value.
- **What it means:** a sanity ceiling only (10× larger, different data, possible contamination), not a control.
- **Card:** experiments/cards/K8.md

**Phase 2 GPU budget** (estimated from v0.3.0 wall times): ~6.5 h if both tuned LRs are 3e-3, up to ~8.5 h with conditional runs and grid extensions.

---

## v0.4.12: K7-PARITY: E10 construction matches E0; FusionDance unchanged
*2026-11-22*

### K7-PARITY: E10 construction parity and FusionDance unchanged
- **Question:** does the new `train.py` builder leave FusionDance bit-identical, and does E10 match E0 exactly in params and FLOPs?
- **Result** (CPU, `results/K7-parity.json`):
  - state_dicts, model config, RNG state and FLOPs are **bit-identical** to the old construction for E0, E1, E2, E3 and E9 (re-checked after the `gdn_neg_eigval` change);
  - E10: total 13,767,552 = E0, non-embedding 10,621,824 = E0, FLOPs per token 9.676e7 = E0;
  - E10 at init: loss 9.094, tied embeddings.
- **Verdict vs prediction:** PASS.
- **What it means:** E10 vs E0 is a clean implementation comparison. Caveat (ledger C91 PARTIAL): a 101-step GPU replay of E0 matched the v0.3.0 log at step 0 exactly (9.0900) but read 2.8370 vs 2.8460 at step 100. Likely GPU nondeterminism, but unproven; an old-vs-new replay is the first Phase 2 item.
- **Card:** experiments/cards/K7-parity.md

---

## v0.4.11: E10 registered: Hugging Face Llama reference control
*2026-11-22*

**What**: registers **E10**: Hugging Face's reference `LlamaForCausalLM` (transformers 5.24.0) at E0's exact size, trained by our own loop, data, eval and tracking. **Not run yet.**
- `gogeta/train.py`: a `"arch": "hf_llama"` config key (default `fusiondance`), an `HFLlama` adapter with loss computed identically, FLOPs by the same rule as FusionDance, and a `--lr` override. Run configs record `arch` and `cli_overrides`.
- `configs/e10_llama_ref.json`, `experiments/launch/E10.sh`, card, plain-English page, registry entry with its **prediction**; E9's card, and a note that E9 runs at E1's tuned LR.

### E10: Reference control: Hugging Face Llama at E0's size
- **Question:** what does a well-known reference implementation reach on our data with a tuned LR? Is our E0 baseline weak?
- **Result:** not run. It runs in the Phase 2 campaign: K7 LR sweep, then 3 seeds.
- **Verdict vs prediction:** prediction (registry): 3-seed mean 0.595–0.615, seed sd ≤ 0.004; E0 at the same LR within mean ± max(3 sd, 0.5%); E1 still beats it by 1.5–4%.
- **What it means:** E10 becomes **the** control for every comparison once it lands.
- **Card:** experiments/cards/E10.md

---

## v0.4.10: Campaign runner + launcher --lr
*2026-11-22*

**What**:
- **`tools/launch_campaign.sh <file>`** runs one launch per line via `launch_experiment.sh --foreground --wait-cool`, sequentially and detached (setsid nohup, PID and state files). It waits while the GPU slot is held, any GPU process exists, or the GPU is ≥ 85 °C; it retries transient refusals and aborts on persistent ones; it has `--status`, `--resume`, `--wait` and `--stop`.
- **`launch_experiment.sh --lr X`:** validated, passed to `gogeta.train --lr`, shown in the plan with a WARN, and recorded in the run config. Written by the cleanup agent as a patch outside its permissions; reviewed and applied by the main agent.

**Results**: the runner was tested end to end by the cleanup agent with a stub launcher and a fake `nvidia-smi` (two bugs fixed). The main agent re-tested `--lr` with dry runs (`--lr 2e-3` accepted and recorded; `--lr abc` rejected).
**Not verified**: a real campaign hasn't run yet.

---

## v0.4.9: Science gaps: wrong claims corrected everywhere, gap list
*2026-11-22*

**What** (control + science-cleanup agent, Phase 1; reviewed by the main agent):
- **Three wrong statements fixed everywhere they appeared**, now matching the ledger and K1/K2:
  - C12 (sliding window "O(n·w)": our dense-mask kernel is O(n²)): `docs/index.html`, `README.md`, `site-src/learn.md`;
  - C13 ("~10× smaller" → computed 7.8× for the body, 3.1× for the whole model): `site-src/learn.md`;
  - C14 ("ternary makes compute cheap" → only with ternary kernels; ours are simulated): `docs/index.html`, `README.md`, writeups E3/E4.
- **`notes/science-gaps.md`:** every known gap, its fix and its status. The six original gaps are fixed or designed. New ones found:
  - E1-vs-E0 "matched compute" is ambiguous (+4.4% params, −6% FLOPs; C95);
  - single-seed claims are stated as fact in the README and primer (left until the replication decides C5);
  - `val_bits_per_char` bias ~0.05%;
  - train/val near-duplicate leakage unchecked;
  - one compute-budget violation (two out-of-launcher GPU jobs at ~17:05);
  - detached-run survival unverified.
- No prediction was ever edited after the fact (checked against the registry's git history).

**Results**: n/a (docs). The site HTML is regenerated in v0.4.13.

---

## v0.4.8: Model architecture visualizer (local server)
*2026-11-22*

**What**: a local web app that **reads the real model code** and draws each config's architecture. `tools/viz/server.py` (stdlib, 127.0.0.1 only, CPU-only) builds `FusionDance` from `configs/*.json` via `gogeta/model.py` on every request, so new configs and code changes show up automatically. Start it with `bash tools/launch_viz.sh start` (WSL) or `tools\launch_viz.ps1 start -Open` (Windows) → http://127.0.0.1:7870. Guide: `tools/viz/README.md`.
- **Shapes carry meaning:**
  - full attention = a filled triangle (n² scores);
  - sliding window = a diagonal band;
  - Gated DeltaNet = a fixed-size square memory tile;
  - MLP = an expand/contract trapezoid;
  - block height ∝ params;
  - the core is drawn once with a "× r" loop arrow, plus an unroll toggle with shared-weight chips;
  - ternary = a −1/0/+1 texture with a 1.58-bit badge.
- **Controls:** a context slider 1k → 32k with live memory/compute growth, a loops slider 1–12, side-by-side compare, tooltips, a plain-English legend, and a DBZ scouter "power level" panel.

**Results**: no training. API numbers match the model code: E0 13,767,552 total; E1 14,237,796 / 91.0 MFLOPs per token; E2 non-embedding 9,460,760, depth 14 at r=4. The page structure was checked in the browser (7 configs, sliders, 21 diagrams, no console errors, no overflow at 1,156 px and 375 px).
**Findings**: the palette check (dataviz validator, dark surface) passes for blue/orange/aqua; yellow vs orange fails all-pairs, but that pair (MLP vs GDN) is always separated by shape, label and position. E10 (`hf_llama`) is drawn as the equivalent 6 × (attention + SwiGLU) stack, with a note.
**Not verified**: the visual look (the browser pane was too small to screenshot); touch tooltips on a real phone.
**Next**: unchanged (cleanup Phase 1 → E10/K7/K8 registered → Phase 2 campaign incl. E11).

---

## v0.4.7: E11 registered: negative-eigenvalue GDN for language modelling
*2026-11-22*

**What**: registers **E11**: E1's layout with GDN's write strength β ∈ (0, 2) (negative eigenvalues). It's one change vs E1, with the same params and FLOPs. **Not run yet.**
- `gogeta/model.py`: new `ModelConfig.gdn_neg_eigval` (default False; the default path is unchanged). When it's on, raw β logits go to fla with `use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True`.
- `tests/test_model.py`: 12/12 pass, including two new tests: causality, and that the computation differs from the default with identical weights (max logit diff 0.25). The config loop now skips non-FusionDance configs.
- `configs/e11_negeig.json`, `experiments/launch/E11.sh`, card, plain-English page; the registry entry with its **prediction** (committed here, before any run).
- `experiments/INDEX.md` generated (from committed sources only); `pyproject.toml` / README badge → 0.4.7.

### E11: Negative-eigenvalue Gated DeltaNet for language modelling
- **Question:** does β ∈ (0, 2) cost or help LM, given that SQ6 shows it solves state tracking in one pass?
- **Result:** not run. It runs at E1's tuned LR from K7, with 3 seeds, in the cleanup campaign right after E1's seeds.
- **Verdict vs prediction:** the prediction is no LM cost, within ±1% of E1's 3-seed mean. Criteria: free / helps / hurts by max(1%, 2× pooled seed std).
- **What it means:** if it's free or helps, neg-eig GDN becomes the default mixer and the loop question is re-asked with it.
- **Card:** experiments/cards/E11.md

---

## v0.4.6: SQ6 (+SQ2): rotation vs loops vs negative eigenvalues on parity
*2026-11-22*

**What**: side quest SQ6 (Mamba-3-style rotating state) with SQ2 (negative-eigenvalue GDN) folded in; prediction committed in `4a3e361` before it ran. This version also commits the **ledger** (`claims.md`: verdicts for C12, C14–C18, C20, C22, new C43–C44; the pending-review sections C30–C42, C50–C69 and C80–C84), the **side-quest log** (SQ outcomes; pending-review SQ14–SQ21, SQ30–SQ41, SQ50–SQ59) and the **lab notebook** entries for all checks and reviews.

### SQ6: Rotation vs loops vs negative eigenvalues on parity
- **Question:** can a single-pass layer track state (parity) as well as or better than looping? (C22; SQ2)
- **Result** (2 seeds; positions 32–63 / 64–127 at T=128):
  - GDN 1 pass 61.7 / 50.3;
  - looped ×4 97.9 / 61.2;
  - RotSSM 84.0 / 76.5;
  - RotSSM, θ=0: 50.2 / 50.1;
  - **GDN with negative eigenvalues: 100.0 / 99.99** (`results/sq_06.json`).
- **Verdict vs prediction:**
  - C22 REFUTED at our scale (84 < 90), although rotation beats looping on length generalisation;
  - rotation does not beat looping in-length;
  - **SQ2 strongly confirmed (+38 points)**.
- **What it means:** a one-flag change to GDN (β ∈ (0, 2)) solves state tracking in one pass and length-generalises, where looping doesn't generalise. This challenges what FusionDance's loop is for, and motivates E11.
- **Card:** experiments/cards/SQ6.md

**Findings**: single-pass GDN parity accuracy has high variance (K5 81.5% vs SQ6 73.3% / 50.1% with identical setup), so ≥ 2 seeds are needed (C44).

---

## v0.4.5: K6: Does E2's loop converge?
*2026-11-22*

**What**: check K6 (Does E2's loop converge?): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K6: Does E2's loop converge?
- **Question:** Does each extra pass refine E2's state? (C20)
- **Result:** Relative change k1 7.20 → k4 0.283 → k12 **0.085**; ‖h_k‖/‖h₀‖ 7.4 → 58.9; loss after the coda best at k=4 (0.722) (`results/K6.json`, inference on the E2 checkpoint).
- **Verdict vs prediction:** Hit (not convergent): C20 REFUTED for E2. The reviews read it as constant-size steps aligned with h, so the norm grows linearly.
- **What it means:** E2's loop inflates its state rather than refining it; loop fixes must bound the state (SQ14–SQ18).
- **Card:** experiments/cards/K6.md

---

## v0.4.4: K5: State tracking: single pass vs looped
*2026-11-22*

**What**: check K5 (State tracking: single pass vs looped): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K5: State tracking: single pass vs looped
- **Question:** Are linear mixers weak at state tracking, and does looping add that depth? (C18)
- **Result:** Running parity, positions 32–63: GDN 1 pass **81.5%**, looped ×4 **97.0%**, 4 unshared GDN 99.8%, attention 50.2% (`results/K5.json`).
- **Verdict vs prediction:** Hit: weakness supported (< 90%); C18 VERIFIED (+15.4 points). Looped ≈ unshared depth (−2.9 points).
- **What it means:** Looping buys state-tracking depth on parity, but so does plain depth. SQ6 then found a cheaper fix.
- **Card:** experiments/cards/K5.md

---

## v0.4.3: K4: Ternary gap vs model size
*2026-11-22*

**What**: check K4 (Ternary gap vs model size): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K4: Ternary gap vs model size
- **Question:** Does the ternary quality gap shrink with size at our scale? (C17)
- **Result:** Mean relative gap **15.32% / 13.76% / 10.55%** at d = 64/128/256 (2 seeds, decreasing within each seed; `results/K4.json`).
- **Verdict vs prediction:** Hit on the trend; the gaps are larger than predicted (12/8/5%). Local analogue VERIFIED, so C17 is PARTIAL (3B parity untestable here).
- **What it means:** Ternary costs ~10%+ at tiny sizes; E3 should run at E1's tuned LR and be judged against this trend.
- **Card:** experiments/cards/K4.md

---

## v0.4.2: K3: MQAR associative recall (invalid)
*2026-11-22*

**What**: check K3 (MQAR associative recall (invalid)): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K3: MQAR associative recall (invalid)
- **Question:** Does GDN recall better than plain linear attention? (C16)
- **Result:** attn 8.6%, gdn 3.3%, linear attention 4.9%, swa(32) 0.4% (2 seeds, `results/K3.json`).
- **Verdict vs prediction:** **Inconclusive**: the positive control (attention ≥ 90%) failed, so the task was too hard or undertrained. C16 stays UNVERIFIED.
- **What it means:** The check needs a redesign (fewer pairs, more steps, or Zoology's recipe) before any recall claim.
- **Card:** experiments/cards/K3.md

---

## v0.4.1: K2: BitLinear speed vs nn.Linear
*2026-11-22*

**What**: check K2 (BitLinear speed vs nn.Linear): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K2: BitLinear speed vs nn.Linear
- **Question:** Does ternary make compute cheaper in our code? (C14)
- **Result:** BitLinear forward is **5.58× / 4.06× slower** (384→1024 / 384→2048); fwd+bwd 1.92× / 1.59× (`results/K2.json`).
- **Verdict vs prediction:** Hit on direction; the slowdown slightly exceeds the predicted 1.5–5×. C14 confirmed WRONG FOR OUR CODE.
- **What it means:** Ternary is a memory win only until we have real ternary kernels; never claim speed from it.
- **Card:** experiments/cards/K2.md

---

## v0.4.0: K1: Memory and time vs context length
*2026-11-22*

**What**: check K1 (Memory and time vs context length): script, results JSON and card. Prediction committed in `f1b4ebd` before it ran.

### K1: Memory and time vs context length
- **Question:** Do our layers scale as claimed? (C12 SWA O(n·w); C15 GDN fixed memory)
- **Result:** Exponents over n = 1k..16k: attn time 1.26 / mem 0.98; **SWA 1.74 / 1.95** (2.9 GB at 16k); **GDN 0.66 / 0.98**. GDN streaming split at 2048: max diff **0.0**, state [1,6,64,64] at every n (`results/K1.json`, commit `ea3aa5b`).
- **Verdict vs prediction:** Hit: SWA ≈ quadratic, as predicted; GDN linear, as predicted. C12 confirmed WRONG FOR OUR CODE; **C15 VERIFIED** (kernel level).
- **What it means:** GDN's constant-memory streaming is real on this GPU; our sliding window needs a block-sparse kernel (SQ3) before any long-context claim.
- **Card:** experiments/cards/K1.md

---

## v0.3.6: Research reviews, experiment cards + index, process rules
*2026-11-22*

**What**
- **Rules (AGENTS.md):**
  - §0b side quests;
  - §0c status updates every 10 minutes;
  - §5 compute budget (1 GPU job + ≤ 8 CPU jobs, 85 °C cool-down), launcher usage, a TRL venv and sandbox rule;
  - §6 one card per experiment plus a generated `experiments/INDEX.md`;
  - §1 **every experiment of any kind is saved, version-tagged and gets its own devlog block**, with the E-series reserved for the main thesis;
  - §1 **no WIP commits: every commit is a tagged version**.
- **Experiment cards** (`experiments/cards/`, from `_TEMPLATE.md`) for E0–E3, and `tools/build_experiment_index.py` (also run by `build_site_data.py`; supports `--registry` / `--tracked-only`).
- **Research:**
  - `notes/research-looped.md`: 12 looped-transformer papers, 2025–26;
  - `notes/review-competitors.md`: fact-check of a pasted analysis on MixerLoop, Auron and TR-Mamba2-Attn;
  - `notes/smollm2-reference.md`: SmolLM2-135M downloaded (rev `93efa2f`, sha256 matches the Hub, Apache-2.0) as an external sanity reference;
  - refs: `10-qwen` (Qwen3.8-Next architecture, FlashNext-RL, Qwen3) and `11-looped-2026` (12 papers).

**Results**: no training.
**Findings**:
- "Inkling 2" is Thinking Machines' Inkling 2 Small: a mixture-of-experts transformer, not looped.
- There's no Qwen4 paper on arXiv; Qwen3.8-Flash-Next is the official Qwen4 preview.
- LT2 (2605.20670) already loops GDN hybrids, but no one combines loops with ternary weights.
- The pasted analysis overstated MixerLoop (41.5% of the gain at 110M, not "much").
- At matched FLOPs, loops lose on LM loss in every paper that checks.

**Notes**: the ledger and side-quest rows proposed by these reviews are committed with v0.4.6. The site isn't regenerated in this version, because its sources are mid-edit by the cleanup agent.
**History**: unpushed untagged WIP commits made before the no-WIP rule: `f1b4ebd` (K1–K6 predictions), `4a3e361` (SQ6 prediction, compute rules), `21fb049` (launch scripts), `ea3aa5b` (TRL scaffold). Check results cite their hashes as provenance, so they stay.
**Next**: one tagged version per check (v0.4.0–v0.4.6), then E11 registered.

---

## v0.3.5: Trust but verify: a local check for every claim
*2026-11-22*

**What**
- **New AGENTS.md §0 "Trust but verify"**, binding on everything else. Every claim we rely on gets the smallest local experiment that could prove it wrong (≤ ~15 min, tiny models, a seeded script in `experiments/checks/`, its pass criterion written first). Paper results motivate experiments but are never adopted as facts. Nothing is called "measured" without on-disk evidence.
- **Claims ledger** `experiments/claims.md`: 20 claims from the README, site, primer and write-ups, each with source, status and evidence, plus six proposed checks (K1–K6).

**Results**: no training. Audit outcome: 5 verified (incl. 1 computed), 6 partial (mostly single seed), 6 unverified paper/theory claims, and **3 that are wrong as currently stated**:
- the "~10× smaller" ternary claim (computed: 7.8× body, 3.1× whole);
- sliding-window "O(n·w)": our dense-mask kernel is O(n²);
- "ternary makes compute cheap": it's simulated, so no speedup.

Plus 1 refuted assumption: fla's recurrent kernel NaNs on this GPU.
**Findings**: on-disk check confirmed **E3 has never been run**; only E0/E1/E2 (seed 0) have results.
**Next**: run checks K1–K6 (~50 min GPU total, all tiny) and fix the three wrong claims everywhere they appear.

---

## v0.3.4: E3 designed: ternary weights on the E1 stack
*2026-11-22*

**What**
- **E3 fully designed** (it was a placeholder): `configs/e3_ternary.json` is E1's exact layout and recipe with `ternary: true`, so the only change is weight precision. Ternary × loops moves to E4.
- **Prediction committed before any run:** val loss 0.605–0.645 (+3.5% to +10% vs E1), most likely ~0.62; stable; 10–25% slower per token (simulated quantisation). Decision rules: ≤3% nearly free, 3–10% expected, >10% → try BitNet's full recipe.
- **E3 page rewritten:** plain-English summary, the one-change design table, how ternary works step by step, the memory payoff, what the BitNet papers found, how it will be judged, risks, and status.

**Results**: no training. Measured on CPU: **33** projection layers become ternary, covering **11.06M of 11.09M** non-embedding params. Weights shrink from 28.5 MB to **9.1 MB** (2-bit packed); the model body from 22.2 MB to **2.8 MB (7.8×)**. From the papers: BitNet b1.58 only matches full precision from 3B params (700M: perplexity +4.4%).
**Findings**: after ternarisation the **word table (6.3 MB) is ~2/3 of the model**, so at our size the next memory win is the embeddings, not the layers.
**Next**: unchanged: training stays paused; when it resumes: E9 → seeds → E3.

---

## v0.3.3: Live on GitHub Pages; version tags published
*2026-11-22*

**What**
- **GitHub Pages enabled**: legacy build from `main` / `docs`, HTTPS enforced. Live at **https://monstercameron.github.io/Gogeta-0/** and set as the repo's homepage link.
- **All 7 version tags pushed** (`v0.1.0` → `v0.3.2`). The repo now has `push.followTags = true`, so future pushes carry their annotated tags automatically.
- README: "Project site" and "Start here" links point at the live site; new tags/site badges; the publishing section is now "Project site". `pyproject.toml` Homepage → the site, plus a Repository URL. AGENTS.md documents the remote, follow-tags and the post-push Pages check.

**Results**: first Pages build `built` in ~20 s. All key paths return HTTP 200 (landing, devlog, primer, E0/E2/E9 pages, CSS/JS, site data, banner). In the browser: version badge v0.3.2, 10 roadmap links resolve, E2's 2 charts render, no console errors.
**Next**: unchanged. Training stays paused until the human asks; then E9 and the seed replications (detached).

---

## v0.3.2: Repo cleanup for GitHub: README, MIT license, packaging
*2026-11-22*

**What**
- **README rewritten** with branding: an SVG banner (`docs/assets/banner.svg`), badges, the pitch, a Mermaid architecture diagram, goals, a results table with a plain-English summary, quick start, repo layout, how the research is run, publishing instructions, credits (including the dataset's ODC-By attribution) and the fan-homage disclaimer.
- **MIT `LICENSE`** (chosen by the human). **`pyproject.toml`**: `pip install -e .` installs the `gogeta` package with its tokenizer asset (torch stays a separate CUDA-index install).
- **Organised:** `sketch.md` → `notes/sketch.md`, next to the lab notebook. References updated; older devlog entries keep their original wording as history.
- **`.gitignore` hardened:** raw/parquet data, tracker DBs, logs, partial downloads, packaging/test caches, keys, and `.claude/*` except the shared `launch.json`. No tracked file is caught by the new rules.
- **Scrubbed** the one machine-specific path in AGENTS.md. AGENTS.md documents the new files and adds a rule to bump `pyproject.toml` + the README badge with every version.

**Results**: no training. Checks: every `.py` compiles; the package installs and imports in WSL and finds its tokenizer; full-history scan found **no tokens/keys/passwords**; the largest committed file is the 536 KB tokenizer; the Mermaid diagram renders.
**Findings**: the Windows username still appears in **2 historical diff lines** (an old AGENTS.md path). It's harmless, and removing it would mean rewriting history and moving tags, which AGENTS.md forbids, so it stays.
**Next**: push to GitHub and enable Pages (`main` / `docs`) when the human is ready; training stays paused until asked.

---

## v0.3.1: Plain-English experiment pages and a newcomer primer
*2026-11-22*

**What**
- **One page per experiment** on the site (`docs/experiments/E0–E9.html`), generated from `experiments/writeups/E<n>.md`. Each opens with **"What it means, in plain English"**, then the setup, charts, a breakdown of the findings, the prediction scorecard, caveats and next steps. Planned experiments get a short "what it will test" page.
- **"Start here" primer** (`docs/learn.html` from `site-src/learn.md`): how a model learns, every number on the site decoded (loss, perplexity, bits/char, FLOPs, seeds), the three tricks as analogies, and what we've found so far.
- **Charts:** interactive learning curves and loss-vs-loops (hover crosshair/tooltips, legend, direct labels, "show the numbers" tables) from `experiments/curves/E<n>.json`. The palette was validated for colour-blind safety on the site's dark surfaces.
- **AGENTS.md rule:** every finished experiment gets its write-up, curve file and generated page before its version is cut.
- Devlog cards and roadmap items link to the experiment pages; "Start here" is in every nav.

**Results**: no training (paused by request). Yardsticks measured on CPU for the explainers: the models compress held-out Python to ~3% of its size vs 15.4% for `xz -9` and 25.1% for `gzip -9`; E1 reached E0's final quality at 84% of the training budget.
**Findings / surprises**
- ⚠ **The replication queue was killed** when the previous agent session ended (E9 at step 600, E2-s1 at step 200). There are no new results, E9 is back to `planned`, and the replication predictions stay open. Future queues should run detached (`nohup`/`setsid`).
- **Correction to v0.3.0:** E2's scorecard is **5 of 7** predicted claims, not "5 of 8".
**Next**: when training resumes: E9, then seeds 1–2 (detached), then the functional-correctness eval.

---

## v0.3.0: First training arcs: E0 dense, E1 hybrid, E2 looped (initial arch)
*2026-11-22*

**What**
- **Data** (`tools/prepare_data.py`): OpenCoder `opc-annealing-corpus` (ODC-By), algorithmic + synthetic snippets, **Python only**, exact-deduped → 991k docs → 8k byte-level BPE → **262.4M train / 1.06M val tokens** (3.72 chars/token). Tokenizer is versioned in `gogeta/assets/`.
- **Model** (`gogeta/model.py`): `FusionDance`, prelude → core × r → coda with `attn` / `swa` / `gdn` blocks (GDN on fla's chunk kernel, verified against a reference), random-r training with truncated BPTT, BitLinear switch. 10/10 correctness tests.
- **Training** (`gogeta/train.py`, `tools/run_queue.sh`): compiled bf16, AdamW, cosine schedule, tracked runs with eval, samples and checkpoints.
- **Notes:** `notes/LABNOTEBOOK.md` (timestamped evidence) plus a new AGENTS.md rule to keep it.
- **Site:** E0–E2 results and verdicts show on the devlog page. The power-level chart's axis now starts near the best result and labels the % gaps (zero-based bars hid a 6% difference). E9 registered.

**Results**: 200M tokens each, seed 0, commit `0dc7491`, full val split

| | Non-emb params | Val loss | Δ vs E0 | Train FLOPs | Wall |
|---|---|---|---|---|---|
| E0 dense (6× attn) | 10.62M | 0.6063 | n/a | 1.9e16 | 23 min |
| **E1 hybrid** (1 attn + 3 GDN + 2 SWA) | 11.09M | **0.5846** | **−3.6%** | 1.8e16 | 39 min |
| E2 looped, r=4 (the initial arch) | 9.46M | 0.6186 | +2.0% | 3.9e16 | 80 min |

E2 loss vs eval loops: r1 0.669 · r2 0.627 · **r4 0.619** · r6 0.624 · r8 0.633 · r12 0.654.

**Findings**
- The **hybrid beats dense** at every checkpoint with only one full-attention layer. It's the first positive result for the linear-time-mixer fighter.
- **Looping doesn't pay yet:** E2 is 5.8% worse than E1 at 2.1× the FLOPs.
- **The loop isn't convergent:** loss bottoms out at r=4 (the mean training r) and degrades when looping more, so it learned "about 4 passes", not "each pass improves". That's a measured motivation for the Rival Loop's per-loop supervision.
- Prediction scorecard: E0 loss range ✗ (data far easier than assumed), E1 direction ✓ / size ✗, E2 5 of 8 claims ✓. Details in the lab notebook.
- ⚠ fla's recurrent GDN kernel returns NaN on this GPU (training kernel fine). SWA still uses a dense mask (O(n²) kernel). Triton needs `python3.12-dev`.

**Decisions**: E1's layout is the current best mixer stack. Added **E9** (unlooped control) to separate looping from depth. Single seed so far: gaps aren't claims until 3 seeds.

**Next**: E9 control; 2 more seeds for E0–E2; a functional-correctness eval (execute generated code against held-out asserts); per-loop diagnostics (‖h‖, per-loop loss) before E3 ternary.

---

## v0.2.0: Local experiment tracking (Trackio) and research hygiene
*2026-11-22*

**What**
- WSL training environment: `~/.venvs/gogeta` (Python 3.12.3) with **Trackio 0.45.1**, pinned in `requirements.txt`.
- `gogeta/tracking.py`: a tracker-agnostic `Run` (`GOGETA_TRACKER=trackio|wandb|none`). It logs git commit / describe / dirty flag, config and seed, writes `meta.json` and `summary.json` per run, records crashes as `failed`, and has a `require_clean` guard for runs that feed tagged versions.
- `tools/record_result.py` (run summary → registry), `tools/dashboard.py` (local dashboard, localhost only, sharing off), `tools/smoke_tracking.py`.
- Every experiment in the registry gets a `prediction` field, shown on the site's devlog page. It's written before the run.
- AGENTS.md §6: tracking and hygiene rules (predictions first, ≥3 seeds for headline claims, tokens/FLOPs x-axes).

**Results**: smoke run `smoke-20261122-115609-s0` logged 200 steps to `~/fusiondance-data/trackio`. The dashboard renders from Windows at 127.0.0.1:7861, and every network request stayed on 127.0.0.1. `require_clean` refuses on a dirty tree.
**Decisions**: Trackio over W&B local: pip-only (no Docker or license; the W&B server image's ARM64 support is unconfirmed), local SQLite storage, and a wandb-compatible API so switching later is one variable. Checked: no analytics/telemetry code in the package.
**Findings**: Trackio reserves config keys starting with `_`, so provenance uses flat `meta_*` keys. From WSL, this repo needs `safe.directory` (WSL git config) and `core.filemode=false` (repo config); otherwise every file looks changed.
**Next**: PyTorch (CUDA, aarch64) in the venv and check `flash-linear-attention`; write E0's prediction; build the E0 baseline.

---

## v0.1.2: Project site (GitHub Pages) and experiment registry
*2026-11-22*

**What**
- `docs/`: a static GitHub Pages site, DBZ-flavoured but scientific. The landing page covers the goals, the three fighters, the FusionDance diagram, the three budgets, the big ideas and the training-arc roadmap. The devlog page has the experiment log with status filters, a val-loss "power level" chart, and the version timeline.
- `experiments/experiments.json`: a registry of the nine planned arcs (E0–E8), from the baseline to self-speculative decoding.
- `tools/build_site_data.py`: generates `docs/data/site-data.js` from `DEVLOG.md` plus the registry, so the site can't drift from the devlog.
- `.gitignore`: `data/` → `/data/` (root only), because it was also hiding `docs/data/`.

**Results**: n/a. Site checked in the browser at desktop and phone widths: no console errors, no horizontal scroll.
**Decisions**: The site reads generated JS data (not fetched JSON), so it also works when opened from disk. Experiment status is tracked in the registry, not in prose.
**Next**: push to GitHub and enable Pages (main, `/docs`) when ready; then WSL setup and E0.

---

## v0.1.1: Naming: repo Gogeta-0, architecture FusionDance
*2026-11-22*

**What**: renamed the repo/project to **Gogeta-0**. **FusionDance** is now the name of the architecture (linear-time mixers + loops + ternary); Gogeta-0 is its first model. Updated README, DEVLOG, AGENTS, sketch and the refs index.
**Results**: n/a
**Decisions**: Repo = Gogeta-0, architecture = FusionDance.
**Next**: unchanged from v0.1.0.

---

## v0.1.0: Project kickoff: design, references, ground rules
*2026-11-22*

**What**
- `sketch.md`: the full design scratchpad: core thesis, v0 architecture, sub-quadratic parts bin, speculative decoding, the Rival Loop, data & evals, local hardware feasibility, decisions log.
- `refs/`: 55 papers across 9 topics, with an index (`refs/README.md`) and a re-runnable fetch script. PDFs are git-ignored.
- `README.md` (pitch), `AGENTS.md` (repo rules), this devlog.

**Decisions**
- Target is a **regular causal LLM**: loops add per-token latent depth.
- Device: the laptop. No hard memory target: we explore the design space with a configurable sweep harness.
- v0 layout: prelude (SWA) → core × r (GDN, GDN, SWA) → coda (one full-attention layer, run once). No quadratic layer inside the loop.
- Intermediate loops decode through a cheap exit head, never the coda.

**Findings**
- Native Windows ARM64 has CPU-only PyTorch. **WSL Ubuntu sees the GPU** and Linux aarch64 CUDA wheels exist, so we train in WSL.
- Closest prior art: *Tiny Recursive Reasoning with Mamba-2 Attention Hybrid* (arXiv 2602.12078). Must read before building.

**Next**
- Read the closest prior art, then lock the novelty story.
- Set up the WSL environment (PyTorch CUDA, check `flash-linear-attention` on aarch64 + Blackwell).
- Smoke-test harness: TinyStories, ~10M params, full precision, loop vs no-loop.

---

<!-- ENTRY TEMPLATE: copy above the previous entry

## vX.Y.Z: <short title>
*YYYY-MM-DD*

**What**: what changed (code, configs, docs)
**Results**: numbers + the config/seed that produced them (or "n/a")
**Decisions**: what we chose and why
**Findings / surprises**: incl. failures
**Next**: the next 1–3 steps

One block per experiment recorded in this version (E-series, checks K, side quests SQ, sweeps, evals…):

### <ID>: <title>
- **Question:** what it tested (and which claim/side quest)
- **Result:** the numbers, with run id / result file and commit
- **Verdict vs prediction:** hit / miss / inconclusive, against the committed criterion
- **What it means:** one or two sentences for the project
- **Card:** experiments/cards/<ID>.md

-->
