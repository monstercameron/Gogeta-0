# Gogeta-0: FusionDance architecture sketch 🕺🕺 → 💥

> LLM + SSM + looped transformers + BitNet... *fu-sion-HA!*

A scratchpad for ideas and feasibility checks. Nothing here is settled.

---

## The core idea

Each of the three techniques attacks a **different cost axis**, so in principle they stack instead of competing:

| Fighter | What it shrinks | Trade-off |
|---|---|---|
| **SSM** (Mamba / Mamba-2 style) | Cost of **sequence length**: linear time, constant-size state instead of a growing KV cache | Weaker at exact recall / copying than attention |
| **Looped transformer** (shared block applied N times) | **Parameter count**: depth comes from repetition, not new weights | More compute per token; training stability is harder |
| **BitNet b1.58** (ternary weights {-1, 0, +1}) | **Bits per parameter**: memory and bandwidth; matmuls become adds/subtracts | Usually needs training from scratch (quantization-aware), not post-hoc |

**Thesis:** a small, ternary, looped, hybrid SSM/attention block could give "big model" reasoning depth on a tiny memory footprint, and it would scale to long contexts.

The nice part is that looping costs extra compute and BitNet makes compute cheap. They cover each other's weak spots.

---

## Decisions log
- **2026-11-22:** Target is a **regular causal LLM**: loops add latent depth *per token*. Not TRM-style whole-answer refinement (maybe later as a second mode).
- **2026-11-22:** Device is **the laptop** (RTX Spark N1X, via WSL). **No hard memory target.** Memory efficiency is a design value, not a constraint, so we're exploring.
- **2026-11-22:** Build it as a **configurable sweep harness**, not one fixed model. Each axis gets toggled and measured.
- **2026-11-22:** Intermediate loops get a **cheap exit head** (norm + tied unembed), not the coda. Decoding every loop through the coda's full attention would bring back r × O(n²).
- **2026-11-22:** Naming: the **repo/project is Gogeta-0**, and the **architecture is FusionDance**. Gogeta-0 is the first FusionDance model.
- **2026-11-22 (v0.3.0):** First results: the E1 hybrid mixer stack beats dense by 3.6%. Looping (E2) does not pay yet and is non-convergent (best at the mean training r). Next comes the unlooped control E9 and per-loop supervision. See DEVLOG v0.3.0 and notes/LABNOTEBOOK.md.
- **2026-11-22 (v0.3.1):** Every finished experiment gets a site page with a plain-English explainer (now an AGENTS.md rule). Training is paused by request; E9 and the seed replications are pending and must run detached.
- **2026-11-22 (v0.3.5):** Trust but verify: every claim we rely on needs its own small local check (AGENTS.md §0, ledger in experiments/claims.md). Paper results motivate experiments but are never adopted as facts.
- **2026-11-22 (v0.4.33):** K9 found 41.7% of val docs are near-duplicates of train. All absolute loss numbers stay provisional until a clean-val re-score; the primary metric for Phase 2b is doc-level bits/char on clean val (< 80% 13-gram overlap).
- **2026-11-22 (v0.4.37):** Phase 2b analysis plan registered: H1 E1 vs E10 and H2 E11 vs E1, Holm at 0.05. The memorisation gap is a caveat flag, never a verdict change.
- **2026-11-22 (v0.4.38-40):** Ledger status is the stricter of the registered verdict and the statistics policy. Hourly claim audits run with a 6-hour cool-down per claim. Rigor design (A3) registered: fair HPO, matched arms, E1 decomposition, and a 2x2x2 factorial with prediction I = +0.5% ± 1.5%.
- **2026-11-22 (K7):** Tuned LRs: E10 3e-3, E1 6e-3 (20M-token sweep, seed 0). Every earlier run used an untuned 3e-3, so E0 and E1 numbers are not yet at tuned LR.
- **2026-11-22 (SQ70 / C110):** The "NaN" in the recurrent GDN kernel was a call-site bug (positional args), not a GPU defect. Keyword args work (C2 verified), so a GDN decode path exists. Decode for SWA and attention is still to build (SQ53).
- **2026-11-22 (C105):** The `post_training/rewards.py` sandbox can write to the repo under WSL, so use `evals/sandbox.py` for untrusted code until SQ61 moves its hardening in.
- **2026-11-22 (v0.4.58):** Claims ledger v2: one file per claim (`experiments/claims/ledger/C<n>.jsonl`), an immutable definition plus append-only events. `experiments/claims.md` is a frozen archive; `experiments/claims-status.md` is generated.
- **2026-11-22 (v0.4.59):** Phase 2b's confirmatory comparisons use E10 (HF Llama at E0's shapes) as the baseline, not E0. E10 gets 0.5822 / 0.5854 on seeds 0/1 vs E0's 0.6063, and E0's scaled residual init may be a handicap (SQ200 tests it).
- **2026-11-22 (v0.4.61):** First hourly claim audit: 48 claims checked; VERIFIED fell 23 to 18. C15 is superseded by C190. Audited claims get a 6-hour cool-down.
- **2026-11-22 (v0.4.63, corrected v0.4.65):** SQ99 out-of-corpus: E1's in-corpus advantage does not transfer (+0.42% worse, exploratory). The cluster-robust bootstrap includes 0, so "E1 is worse out of corpus" is withdrawn; the non-transfer verdict stands.
- **2026-11-22 (v0.4.65):** Byte Quest becomes a main-quest arc: E12-E16 (patch granularity x refinement loops), with predictions registered before any run.
- **2026-11-22 (v0.4.66):** SQ99 length re-analysis: the out-of-corpus pattern tracks document length (E1 ahead on short files, behind on files over ~1,200 tokens). Recorded as C193 (PARTIAL, exploratory, n=1, length confounded with file type). C192 stays REFUTED under its registered pooled test.
- **2026-11-22 (v0.4.68-69):** Byte Quest E12-E16 predictions registered before any byte run (`experiments/checks/PREDICTIONS.md`); the 30 staged configs are hashed into that file.
- **2026-11-22 (v0.4.69):** Vast.ai CLI installed; no instance rented. Each rental needs the human's explicit go-ahead (AGENTS §0a). SQ220 stayed parked until the human approved the pilot later that evening ("get 1 working first", $30 / 12 h cap).
- **2026-11-22 (v0.4.70):** Outside review (20:37): E1's lead over E0 may be largely a baseline artifact (E0 trails E10 at identical shapes). Recorded as a C5 limitation event. Phase 2b's comparison stays E1 vs E10.
- **2026-11-22 (v0.4.72):** AGENTS §0h, kill-on-empty (the human: "kill them unless we have no work queued up"). Rented GPUs live only while work is queued: rent just in time, batch runs per box, pull results and destroy the box when its last job ends, and destroy any box idle for more than 10 min.
- **2026-11-22 (v0.4.73):** Cloud card choice: the RTX 5090 is the reference rental (fastest and cheapest per token vs the 3090 and N1X; SQ220, C195 PARTIAL).
- **2026-11-22 (v0.4.73):** Operator hold on Phase 2b. After the N1X thermal kill (86.6 °C) and an unwanted auto-resume, exit 143/130 is classified as operator and never auto-resumed. `cockpit.sh --hold` blocks every resume; only the human releases it. Phase 2b stays at 6 of 11 until then.
- **2026-11-22 (v0.4.74):** Expense tracking: an append-only ledger in `experiments/expenses.json` (purchases, estimated and reconciled compute). Public-repo rule: amounts and purposes only, with no card digits, receipts, addresses or e-mails.
- **2026-11-22 (v0.4.74):** Phase 2b-5090 is a single-SKU replication: E10 and E1 at 7 seeds each (E11 ×3, E9 and E2 ×1), 200M tokens. The two SKUs are never pooled, and the N1X 3 + 3-seed comparison stays the registered confirmatory test (Q12).
- **2026-11-22 (v0.4.75):** Fleet on 5090s only: 8 RTX 5090s, at most 8 concurrent, at the human's request ("the cheapest 5090s"). The N1X is limited to light work after its thermal kill. Wave 2 is pinned to one SKU. The cockpit spend approval is human-only (TTY); until the human types it, rentals run under the chat go-ahead.

- **2026-11-22 (v0.4.79):** Phase 2b-5090 is in: E1 ties E10 at 7 seeds per arm (+0.08%, CI spans 0, p 0.64), so the hybrid is tied with the Llama control, not better. E11 is equivalent to E1 (C196, TOST within ±1%). Loops don't pay: E2 is 2.6% worse than E9. The same-SKU replay floor is about 0.002 bpc (SQ232), so cross-SKU deltas below that are noise.
- **2026-11-22 (v0.4.80):** Fleet waves 1–3b are done (106 runs, 0 failed, $6.54). C197 (exploratory, n=1): at E2's FLOPs, unshared blocks with about 2.3× E1's unique parameters beat E1 by about 7%. Scale is the lever to test next, not loops.
- **2026-11-22 (v0.4.80, AGENTS §0h):** Pipeline the waves. Queue the next wave before the current one ends, so boxes move straight on with env, data and compile cache in place. Kill-on-empty still applies, and "empty" means no current or next-wave job for that box.
- **2026-11-22 (wave 4, registered before dispatch):** SQ243, the two-size check (E1 and E10 at d = 256 and 512, seeds 0–3, LRs not re-tuned: a stated limitation), and SQ241, param-matched E1 (mlp_hidden 952: 10,622,052 non-embedding params against E10's 10,621,824, seeds 0–3). SQ253 gives an 80%-power detectable effect of about 0.9% at 4 seeds per arm. Predictions S1–S3 and P1–P2 are in `experiments/checks/PREDICTIONS.md`.
- **2026-11-22 (wave 5, registered before dispatch):** The F1 pilot's pooled log-sd is 0.009, outside both cases the design allowed for (0.004, N = 46; 0.0067, N = 127). So F2 is re-sized by the registered sigma² rule to N = 246 (E20 = 82 seeds, the others 41 each; 231 new runs at 20M tokens on code-py-v2). The analysis is unchanged. The sample size changed after the pilot's variance was seen, as the design intended, and no pilot means entered the sizing.
- **2026-11-22 (v0.4.85, the human):** New goal 6: end the experiments with one final model that runs batch-1 inference with a KV/state cache and lossless speculative decoding on a single 24 GB RTX 3090/4090. Plan in `notes/final-model-plan.md`: training compute and unique tokens bind, not VRAM. Primary target G-1B (~1.2B, E1-pattern hybrid with GQA, ~$500–650 to train); stretch G-3B (~$2.8k, needs Stack v3 data). The ingredients (loops, ternary, MoE, spec-decode scheme) are decided by gates; the final train needs the human's approval. Registered as E61 (planned), side quests SQ285–SQ298. Open question for the human: does v1.0.0 still require the full fusion?
- **2026-11-22 (v0.4.85, SQ188 rule applied):** The byte LR minimum is lr 3e-3, inside the grid after the registered 1.5e-3 extension. All E13 arms run at 3e-3.
- **2026-11-23 (v0.4.89):** C199 downgraded VERIFIED -> PARTIAL. "E1 is insensitive to init" is a no-effect claim: a CI spanning zero means no difference detected, not equivalence, unless a TOST was registered (statistics-policy §2.6).
- **2026-11-23 (v0.4.89):** C198's "they tie" wording is replaced by CI-based wording. SQ307 (fresh-seed confirmation of SQ301, prediction W3) is registered.
- **2026-11-23 (v0.4.89):** F2 stage-1 blinding was breached in display: the fleet monitor showed per-run F2 scores by cell until 01:00. Disclosed in the devlog; the monitor now redacts F2 scores until the blinded re-estimation.
- **2026-11-23 (v0.4.90):** The init confound on C197 is registered as SQ308 (E1 and E32 under gogeta and flat init, seeds 0-2). Rule: a flat-init gap within ±1% means C197's lever is an init artifact.
- **2026-11-23 (v0.4.91):** F2 stage-2 size is set by the registered blinded-sigma rule: N_final = min(246, ⌈235·(σ/0.009)²⌉) = 221, from seed-blocked σ = 0.00873 (stage 1 = 135 runs). No cell means, contrasts or interactions were computed. Under the seed-blocked model, a seed run by one cell adds nothing to the contrasts, so the primary effective N is the 37 seeds shared by all five cells; the analysis must report this with the power.

- **2026-11-23 (v0.4.94):** SQ282 (LR-fair) shows the wide E34 ties the deep E32 at best LRs, so C197's lever is unique parameters at matched FLOPs, not distinct depth; E60 (MoE, width-like capacity) regains its prior and is ungated (SQ238 also done). SQ238: dedup alone is worth 1.6% (C200 VERIFIED).
- **2026-11-23 (v0.4.95):** Claim audit downgrades C200, C1 and C131 from VERIFIED to PARTIAL (narrower scope than the claim). C5 stays REFUTED by its registered rule ("REFUTED if mean(E1) >= mean(E10)"): changing a registered rule after the fact would be a forking path. Its qualifier now says both CIs include 0, i.e. no difference detected.
- **2026-11-23 (v0.4.96):** Code-MoE FFN (`gogeta/moe.py`) is added, off by default and bit-identical with it off (`tests/test_moe.py`). E60 stage 1 is registered: a G3 LR grid at 20M, then 3 arms x 6 seeds at 200M (predictions M1-M3). E60 is ungated by SQ282 and SQ238.
- **2026-11-23 (v0.4.97):** F2 result: the GDN x loops x ternary interaction is -3.54% (seed-blocked, SYNERGY; C201, VERIFIED with limitations). The registered prediction (+0.5% +/- 1.5%, worst cell = full fusion) was wrong on both counts. Read as overlapping penalties, not a gain over dense: the full fusion stays 11% worse than dense at 20M. SQ263 (200M) is now unblinded.
- **2026-11-23 (v0.4.98):** Docs refresh and F2 audits: inputs reconciled, estimator validated, seed-robust. The GDN cell's 1.6x wall time is a C201 limitation: FLOPs-matched is not wall-clock-matched.
- **2026-11-23 (v0.4.99):** Outside review accepted: init-matched F2 check (SQ326) and context truncation (SQ327). The data-order pairing doubt is rejected: same-seed arms see identical batches, so the paired CIs are paired.
- **2026-11-23 (v0.4.100):** `/var/tmp` joins the private tmpfs mounts in the eval sandbox after a leak was found (self-test 12/12). E60 takes lr 6e-3 for all arms (registered G3 rule). At the human's request ("tear down any vps ... if we don't use it") w4-1 was destroyed.
- **2026-11-23 (v0.4.102):** SQ222 (fp32 vs bf16 recurrence) is promoted, since C197 and C201 have no precision check. The log scale is the registered metric; a raw-scale I of about −4.2% is a sensitivity number, not the test.
- **2026-11-23 (v0.4.103):** SQ263, the registered 200M gate, is analysed: the interaction is +6.21 log-points (anti-synergy), H2' FAIL. C201 is REFUTED by its own registered criterion; C202 (PARTIAL) replaces it as the live claim. At the human's "kill the fleet" the last boxes were destroyed; total compute spend is $30.13 against the approved $30. Lesson: destroy each box as its last job is pulled, and check instances at the start of every session.
- **2026-11-23 (v0.4.104):** SQ337 and SQ338 are run from runs already on disk (descriptive): the 200M interaction rises through training, and GDN loses at equal wall-clock (+8.2%). The F2 write-up's "not an LR artifact" was overstated: cells 000 and 100 ran only at 3e-3 at 200M, so the LR confound is open (SQ342, GPU, needs budget approval). SQ339 to SQ341 are CPU side quests; SQ343 is GPU and also needs approval.

- **2026-11-23 (v0.4.108):** The statistics policy gains an INCONCLUSIVE status row, matching the ledger schema: a check that ran but failed its own validity gate is INCONCLUSIVE, not UNVERIFIED (C16 stays INCONCLUSIVE). C201 stays REFUTED under its own criterion, registered at creation, and its 20M measurement is noted as standing.

- **2026-11-23 (v0.5.0), human goal:** "start putting the model together ... solve the unintuitive parts and find a way to compensate for its challenges, use the new dataset ... use the 5090 or greater VPS ... make sure all the components work well together". Plan:
  - a Sonnet worker clears the CPU side quests on disjoint files;
  - the main agent assembles the fusion on code-py-v2 at 200M tokens: A1 screens one compensation at a time on E27 (2 seeds), then A2 stacks the winners (3 seeds, with dense and GDN references);
  - compensation hooks in `gogeta/model.py` are off by default and bit-identical;
  - rental: the human authorised 5090-or-better boxes without naming an amount, so the agent caps itself at $25 for A1+A2 and at most 8 concurrent boxes, destroys each box as soon as its last job is pulled, and reports spend.
  - code-py-v3 (Stack v3) needs a separate download approval (45-120 GB), so v2 is "the new dataset" for now.

- **2026-11-23 (v0.5.10), human decision:** "I want all the parts assembled and their issues designed around or fixed!" Every ingredient (GDN mixer, shared loops, ternary, MoE) stays in the model, and each cost is designed around instead of dropping the ingredient. This supersedes A2 amendment 2's loop go/no-go default. A3 tests the fixes: a ternary MoE core for the loops' unique-parameter deficit, and full-precision mixers or FFN to locate and remove ternary's cost.

- **2026-11-23 (v0.5.12), human decision:** "I mean everything, ssm + sliding windows + loops + coarse to fine + bitnet + MoE + speculative decode". The target is ONE model with every component:
  - GDN (the SSM-family mixer) + SWA + full-attention blocks;
  - a shared loop with contractive injection;
  - BitNet ternary with full-precision ends/mixers as fixed by A3;
  - a ternary MoE core;
  - a coarse-to-fine byte front/back end (bytes -> patches -> this backbone -> shared-weight byte refinement loop);
  - lossless self-speculative decoding (draft with fewer loop passes, verify with full).
  Build plan: two builder subagents in parallel on disjoint new files (`gogeta/decode.py` for cached inference and self-speculative decoding; `gogeta/c2f.py` for the byte front/back end around `FusionDance`), using new `inputs_embeds` / `return_hidden` hooks in `FusionDance.forward`. A4 trains the full stack once A3 picks the backbone.

- **2026-11-23 (v0.5.10):** Human: keep every part (ternary, loops, MoE, coarse-to-fine bytes, speculative decoding). Costs are designed around or fixed, not dropped. The v0.5.9 loop go/no-go default is superseded.
- **2026-11-23 (v0.5.15):** Tokens as patches. The global model stays on BPE tokens and a byte decoder spells each next token. Fixed P-byte patches are rejected (SQ374: +36% to +63%).
- **2026-11-23 (v0.5.16):** A4 amendment 1, registered before any A4 run: R2 is split into a second pass (A4d vs A4b) and deep supervision (A4a vs A4d), Holm-adjusted.
- **2026-11-23 (v0.5.18):** A3 LR rule: 6e-3 within 1% of the best LR restores C206-C208 to VERIFIED; otherwise the affected comparisons re-run at the better LR. C204 is superseded by C209, with the compute difference (about 10%) stated.
- **2026-11-23 (v0.5.19):** Erratum: "A3mx ties dense" is retracted (TOST at +-1%: p = 0.58). Rule: "ties" or "equivalent" needs a TOST result in the ledger; a point estimate is not a tie.
- **2026-11-23 (v0.5.20):** The A4 viability rule (within 3% of A3mx) failed at +14.4%. The byte decoder stays in the model (human directive); the design-around SQ394 is the top pick (wider decoder, hybrid head keeping the BPE loss on the trunk, frozen-trunk control).
- **2026-11-23 (v0.5.20):** Self-speculative decoding is not claimed lossless. A lossless claim needs batch-invariant matmuls in the verify path and a GPU wall-clock measurement.
- **2026-11-23 (v0.5.22):** Erratum: the A2 headline was never "at matched FLOPs". A2s uses about 10% more training compute than E27 (C209). A2 amendment 3 (E27 at 221.6M tokens, seeds 2-4) and A3 amendment 2 (A2f at 3e-3 and 1.2e-2) are registered. Expenses reconciled: $42.75 total.
- **2026-11-23 (v0.5.23):** A3 amendment 1 passed: A2s and A3mx are LR-robust at 6e-3 (within 1% of the best). C206-C208 are back to VERIFIED. A2s's 3e-3 is 0.58% better, a stated limitation.
- **2026-11-23 (v0.5.24):** Fix: `gogeta/hf.py` could not build a model from v0.5.0 to v0.5.23 (missing assembly fields). `hidden()` now delegates to `FusionDance.forward`. C80 re-verified with saved output (11/11, 6/6). Process: the release lint should re-run implementation claims' tests (SQ274 extension).
- **2026-11-23 (v0.5.25):** Claim audit: C210 and C211 go to UNVERIFIED until the clean80 score (SQ387). C209's LR objection is rejected (the A3 bracket is the evidence).
- **2026-11-23 (v0.5.26):** Out of corpus (SQ379, 330 SQ99 files): the ternary and loop costs nearly vanish while the assembly gain survives. Recorded as hypothesis C213 (UNVERIFIED); SQ400 is its registered replication. The coda's full attention carries long-range copy (SQ383), so the design keeps it. Ternary loops are not contractive (SQ353), which supports the full-precision mixer fix (C207).
- **2026-11-23 (v0.5.27):** Reading rules agreed after the outside review: ternary arms are counted at full-precision FLOPs; "matched compute" means equal analytic FLOPs; assembly headlines are FLOP-matched, not param-matched, and each claim's scope states its params. SQ401 (dense 2048-hidden core FFN vs the MoE core), SQ402 (GDN beta) and SQ403 (eval-time pass count) are registered. The init-depth concern was checked: no confound.
- **2026-11-23 (v0.5.28):** Docs refresh. A3 amendment 2: A2f is LR-robust at 6e-3, so the ternary cost is +9.5% at each arm's best LR (recorded on C207).
- **2026-11-23 (v0.5.29):** Outside review: SQ404 (optimizer confound of weight sharing: shared weights step larger than unshared) and SQ405 (temporal holdout) registered. Statistics policy: criterion 4 (CI excludes 0) applies to difference claims, and TOST (criterion 6) to equivalence claims.
- **2026-11-23 (v0.5.30):** The A2 headline holds at matched compute: A2s vs FLOP-matched E27 is -16.18% [-16.87, -15.48] (C209). The assembly fleet is torn down ($13.09).
- **2026-11-23 (v0.5.31), human decision:** "Yes, rent one". GPU batch B1 is registered before dispatch (SQ401, SQ402, SQ404), 23 runs on one RTX 5090; three off-by-default switches are added and tested.
- **2026-11-23 (v0.5.32):** Claim audit: C213 gains a scale-compression confound and stays UNVERIFIED.
- **2026-11-23 (v0.5.33):** Outside review: SQ350 (parameter-matched loop control) added to B1 as amendment 1, registered before those runs. C212 selection-seed limitation recorded. SQ408 and SQ409 added.
- **2026-11-23 (v0.5.34):** B1 amendment 2: the gating contrasts G1-G3 form one Holm family (alpha 0.05, m = 3), registered before any B1 result. Everything else in B1 is an exploratory readout, and no B1 contrast can promote a claim.
- **2026-11-23 (v0.5.35):** SQ404's "larger step" premise is withdrawn: AdamW normalises each parameter's update, and clipping never fires after warm-up, so the loop cost has no clipping component. B1 amendment 3 (registered before any SQ404 result) reads SQ404 as "does the shared core's best LR differ from the global LR?".
- **2026-11-23 (v0.5.36):** B1 readout (20M screens): a fixed GDN write strength is a candidate simplification, pending a 200M check; the loop cost is not an LR artifact; unique parameters beat a second pass about 2.4x per FLOP.
- **2026-11-23 (v0.5.38), design decision (C213's registered rule fires):** the final model (E61) is sized and judged on out-of-corpus quality next to clean80. Ternary and loops are no longer treated as costs on that metric. SQ403 raises the priority of draft-aware or variable-r training (SQ381).
- **2026-11-23 (v0.5.39):** every batch registration names its readout script's sha256-16 before dispatch, and in-flight amendments state what had been seen. Erratum: "the ternary cost vanishes" becomes "not distinguishable from 0 out of corpus".
- **2026-11-23 (v0.5.40):** SQ411 closes the `__eq__` reward-forgery route (tests are AST-rewritten in the grader; strict checks over exact builtin types). The route no longer blocks GRPO. Honest solutions returning custom classes now fail closed, by design.
- **2026-11-23 (v0.5.45):** Analysis scripts use the exact Student t quantile (`experiments/checks/tstat.py`), never a series approximation: the Cornish-Fisher expansion was 10% low at df 2, so n=3 intervals were too narrow (no decision changed). SQ419 (F-6, the parameter-matched control for C209) runs at 200M tokens on seeds 2-4 against the existing E27 and A2s runs, not as a 20M screen, because 20M keeps sign but not size and interactions can flip (outside review 18:08).
- **2026-11-23 (v0.5.47-48):** The sandbox setsid escape (SQ61) re-gated GRPO. The fix (on the human's go-ahead) gives each sandbox child its own PID namespace and a bounded wait, which lifts that gate again.
- **2026-11-23 (v0.5.50-51):** GPU batch C1 registered before dispatch: SQ419 (F-6 = A1g on seeds 2-4 at 200M, plus a seed-5 LR bracket) is the parameter-matched control for C209. Amendment 1 (before any result) adds a FLOP-matched E27 comparator, a 1.5e-3 LR edge and an interval-based S rule.
- **2026-11-23 (v0.5.53):** Wording rule: "a tie" becomes "no difference detected", and "not needed" becomes "no cost detected (2 seeds)". A NO-DIFFERENCE ledger status was rejected, since it would rewrite registered verdicts. C19 now records that the self-speculative decoder is not lossless under int8 activations.
- **2026-11-23 (v0.5.54):** SQ435 registered before any scoring: a loss-matched corruption control for C213 (script hash 9a53278250e80782). Its result is diagnostic and changes no status.
- **2026-11-23 (v0.5.55, at the human's request):** The data-flow atlas (SQ439) is descriptive. Its eval-only swap (SQ441) is recorded as internal evidence on C213, with no status change. Five technique side quests (SQ442-SQ446) follow.
- **2026-11-23 (v0.5.57):** the human paused the research. The recurring review jobs are cancelled, and the next GPU batch waits for a go-ahead ("not yet"). The continuation guide is `notes/CONTINUATION.md`, and the session's fleet and version tooling is saved in `tools/fleet/`.
- **2026-11-23 (v0.5.57, outside review 20:52):** C215 reads as "the A2 bundle, which spends more weight bits on the ends, improves the recipe" until the bit-matched control SQ447 runs, not as "fixed r and contractive injection help". SQ441's recall reading is scoped to A2s and E27, since E24's 0.00% is a consistency check.

## Status (2026-11-23, v0.5.57)

- Data-flow atlas (SQ439, v0.5.55, descriptive): the first SWA layer acts as a causal moving average. Its eval-only swap (SQ441) costs E24 0.00% and at most 0.4% out of corpus, but E27 loses 6.5-7.3% in corpus. E27's ternary prelude is nearly inert (RMS 0.09 to 0.14). Loop passes past 2 change the state by at most about 0.3x. The coda carries the largest share out of corpus. Recorded as evidence on C213 with no status change (`experiments/devlogs/SQ439.md`).
- Phase: full-stack assembly. Every component of the full stack is built (GDN, sliding-window and full attention, a shared loop with contractive injection, BitNet ternary, a ternary MoE core, coarse-to-fine bytes, self-speculative decoding). The costs are designed around or fixed, not dropped (human directive, v0.5.10).
- A3 (E64, all parts, seeds 5-7, final in v0.5.17): every fix counts: ternary MoE core -3.29%, full-precision mixers -5.02%, full-precision FFN -5.16%, all parts -7.67% vs A2s. A3mx vs E24 +4.35%; vs E20 +1.10% (90% CI [-0.13, +2.33]), which is not an equivalence to dense: the v0.5.17 "ties dense" wording was retracted in the v0.5.19 erratum (TOST p 0.58). C206-C208 are VERIFIED: the learning-rate bracket passed in v0.5.23 (A3 amendment 1).
- A4 (E65, full stack, tokens as patches), final in v0.5.20: the byte decoder costs +14.39% against A3mx (C210, UNVERIFIED since v0.5.25, pending clean80, SQ387). The registered viability rule (within 3%) fails; EOT accounts for 0.072 nats per token of the gap. A second shared byte pass gives -1.68% (C211, UNVERIFIED); per-pass deep supervision adds nothing. The design-around is registered as SQ394 (top pick).
- A2 (E63; C209, VERIFIED at recipe level): the assembled model is -16.7% vs E27 and +8.6% vs dense E20. Its compute is about 10% above E27's, and that difference is stated in the claim. The headline is not at matched FLOPs (erratum v0.5.22); the compute-matched control holds: against E27 at 221.6M tokens (A2 amendment 3, v0.5.30, seeds 2-4), A2s is -16.18% [-16.87, -15.48]. C209's LR objection was rejected in v0.5.25. A2f is -0.9% vs E20 (n = 3, not a tie) and +3.2% vs E24 (3 seeds). C204 is SUPERSEDED by C209.
- A1 amendment 2 (final in v0.5.20): fixed r = 4 beats random r with full backprop by 3.17% (C212, PARTIAL: the LR is not bracketed for A1a and A1i). Seeds 0-1 both chose and tested A1a (v0.5.33), so the CI is not a fresh-seed interval; the fresh seed alone gives -3.11%.
- Self-speculative decoding (E8; `gogeta/decode.py`, v0.5.20): built with exact KV, SWA and GDN-state caches and rollback. CPU bench on the A2s s2 checkpoint: r_draft = 1 accepts 70% at k = 4 (3.7 tokens per verify call) and 57% at k = 8. NOT lossless: 16 of 120 runs differ at near-tie steps under int8 activation rounding; with activation quantisation off, 30 of 30 runs are lossless. Exact losslessness needs batch-invariant matmuls in the verify path, which is not built. GPU wall-clock speedup is not measured. C19 stays UNVERIFIED.
- Byte Quest (SQ374, v0.5.15, clean80, one seed each): fixed P-byte patches are far too costly (E13b, P = 4: +63%; E13a, P = 2: +36% vs E1). Decision: tokens as patches.
- Wrapper: `gogeta/hf.py` could not build any model from v0.5.0 to v0.5.23 (missing assembly fields). Fixed in v0.5.24; C80 VERIFIED on saved output (11/11, 6/6).
- Reward harness: the forgery is fixed for every probed route (SQ373). SQ411 (v0.5.40) closes the `__eq__` route, so the GRPO block is lifted. C83 stays PARTIAL: solution-defined helpers, `.equals`-style methods and any/all over forged iterables remain (`post_training/SETUP.md` section 7). The setsid descendant escape is fixed (v0.5.48: a PID namespace per child and a bounded wait); SQ61's resource-budget, world-writable-path and /proc tests are still open.
- Data gate (SQ376, passed in v0.5.54, descriptive: val loss still falls 22.0% for E1 and 20.1% for E10 from 100M to 197M tokens; part of the late gain is cosine annealing): code-py-v2 has about 0.2B tokens, and the 1.2B final model needs about 20B+. code-py-v3 needs human download approval before E61.
- Claims (`experiments/claims-status.md`, generated): 27 VERIFIED, 27 PARTIAL, 9 REFUTED, 2 WRONG FOR OUR CODE, 1 INCONCLUSIVE, 74 UNVERIFIED, 3 SUPERSEDED (143 total, as of v0.5.57).
- Results so far: E1 is +0.08% vs E10 (7 seeds per arm on RTX 5090s; no difference detected). E11 is equivalent to E1 (C196, VERIFIED). Loops do not pay yet (C6, PARTIAL). At 200M, 14 unshared blocks beat E1 (C197, VERIFIED). The F2 interaction at 200M is +6.21 log-points (C202, PARTIAL); C201 is REFUTED by its 200M check.
- B1 (v0.5.36; 20M screens, exploratory, no ledger change): the MoE core matches dense-2048 at 31% fewer FLOPs (a tie, SQ401); a fixed GDN write strength costs under 0.5% (SQ402); a core LR multiplier closes 9% of the loop cost (SQ404); unique parameters beat a second pass about 2.4x per FLOP (SQ350). No Holm gate opens.
- Out of corpus (SQ400, v0.5.38; registered, 330 SQ99 files, paired-by-seed CIs): the ternary cost is not distinguishable from 0 (A2s vs A2f +0.42% [-0.92, +1.77]), and the loop cost reverses sign (A2f vs E24 -1.91% [-3.40, -0.40]; C214, VERIFIED on that one set). The assembly gain survives (A2s vs E27 -11.3%). Recorded as hypothesis C213 (PARTIAL). The earlier SQ379 read (v0.5.26) said both costs nearly vanish; that was descriptive. A scale-compression confound is recorded (v0.5.32): out-of-corpus losses are 6-7x higher, which compresses every relative effect, but the attenuation is differential, so compression alone does not explain it. SQ400 must also report absolute bits/char differences.
- Queue: GPU batch C1 (SQ419) finished in v0.5.56 and its box is destroyed ($0.84). F-6 vs E27 -10.54% [-10.83, -10.25]; A2s vs F-6 -6.85% [-8.02, -5.67] (C215, VERIFIED, but probably mostly the precision scope: the full-precision ends spend more bits; SQ447 is the bit-matched control); S = 0.61 [0.57, 0.65], so C209's gap is mostly shape and parameters. F-6's checkpoints were not pulled (date-filter bug, now fixed). Research is paused (v0.5.57): nothing runs on a GPU. The next batch waits for the human ("not yet", 20:57); the suggested batch is SQ447 first, then SQ433, SQ442, SQ444 and SQ448. E8 waits for a 24 GB card.
- Fleet: A3 amendment 2 (v0.5.28) and A2 amendment 3 (v0.5.30) are finished and summarised in the devlog. Every A3/A4 box is destroyed (v0.5.30); the assembly fleet cost $13.09. Total spend is $44.85 (v0.5.56). B1's one rented RTX 5090 was destroyed at 16:55 (v0.5.36); 0 instances.
- Open: the clean80 score for C210 and C211 (SQ387); SQ405 (temporal holdout; any download needs the human's OK); SQ415 and SQ416 (a training set filtered against the SQ99 files, and a prose slice, for C214's style-shift limitation); SQ417 (WSD vs cosine schedule); E8's batch-1 decode frontier (needs a 24 GB card); SQ408 and SQ409 (v0.5.33); SQ381 (draft-aware or variable-r training); SQ394 (the A4 design-around); SQ435 finished after the pause (v0.5.61, UNMATCHED and descriptive: renaming identifiers moves the ternary cost +9.5% to +3.7% and the loop cost +3.1% to -1.8%); SQ265 (written, not run); SQ447 and SQ448 (new at v0.5.57); SQ442-SQ446 (technique ideas from the atlas).

---

## The fusion dance metaphor (it maps surprisingly well)

- **Matching power levels.** Fusion only works if both fighters are matched. Here that means the hidden dims, norms and activation scales of the SSM and attention paths have to line up so neither one dominates the residual stream.
- **The dance must be perfectly synced.** One wrong finger and you get fat or skinny Gotenks. The interfaces matter: where the SSM and attention layers interleave, how norms sit around the looped block (Huginn-style sandwich norm?), and how ternary quantization interacts with the recurrent state.
- **30-minute time limit.** Fusion is temporary, so think adaptive loop count / halting: easy tokens exit early and hard tokens loop longer. See Mixture-of-Recursions and adaptive computation time.
- **Dance vs Potara earrings.**
  - *Dance* = fusion at runtime: merge or route between separately trained models, or toggle the number of loops at inference.
  - *Potara* = permanent fusion: one architecture trained end to end as a hybrid.
  - Potara is the realistic path. The Dance is the fun experiment.
- **Failed fusion.** Training divergence. Looped models and ternary models are both known to be finicky, so combining them doubles the chance of a skinny Gotenks.

---

## Gogeta-0: the first FusionDance model

```
tokens
  │
[Prelude]       1 unshared layer: embed → latent          (ternary linear)
  │
┌─[Core block] × r loops ────────────────────────────────────┐
│   Gated DeltaNet ×2   ← linear-time mixing, good recall     │
│                         (ablate vs Mamba-2 / Mamba-3)       │
│   Sliding-window attn ← local precision, O(n·w) not O(n²)   │
│   MLP (ternary)                                             │
│   + input re-injection each loop (Huginn-style)             │
└─────────────────────────────────────────────────────────────┘
  │   r sampled randomly in training, adaptive at inference
[Coda]          1 FULL attention layer (runs once, not r times) + MLP → logits
```

**Rule of thumb: never put a quadratic layer inside the loop.** Anything quadratic in the core block costs r × O(n²). Full attention only goes in the coda, where it runs once. (Revised 2026-11-22; v0 had Mamba-2 plus one maybe-windowed attention layer in the loop.)

Open design choices:
- Should the SSM state **carry across loops**, or reset each loop? Carrying it gives a 2D recurrence (time × depth). Interesting, but maybe cursed.
- Ternary everywhere, or keep the SSM's A/Δ/state params at higher precision? (Probably the latter: the scan is sensitive.)
- Attention every loop, or only every k loops? Zamba reuses one *shared* attention block across depth, which is already halfway to looping.

---

## Sub-quadratic parts bin (and where each fits in Gogeta-0)

Here n = sequence length. "Decode" means the cost per generated token.

### A. Recurrent / linear-time state: O(n) training, O(1) per token at decode
| Part | Idea | Gogeta-0 fit |
|---|---|---|
| **Gated DeltaNet** (ICLR 2025) / **Gated DeltaNet-2** (NVIDIA, May 2026) / **KDA** (Kimi Linear) | Linear attention with a delta-rule update: it *overwrites* memory instead of just adding to it, so recall is much better. GDN-2 splits erase and write into separate channel-wise gates. | ⭐ **Core-loop mixer, first choice.** Used in production hybrids (Qwen3-Next, Kimi Linear). |
| **Mamba-2 / Mamba-3** | Selective SSM scan | Core-loop mixer, **ablation baseline** |
| RWKV-7, GLA, RetNet, mLSTM (xLSTM) | Older members of the same family | Extra ablations only if time allows |
| **Titans / TTT layers** | Memory that updates itself with gradient steps at test time | 🃏 Wild card: "loop = more memory-update steps" is a cute match with looping. Expensive. Later. |
| Hyena / long convolutions | FFT-based, O(n log n) | Skip: weak at recall |

### B. Restricted softmax attention: fewer token pairs
| Part | Idea | Gogeta-0 fit |
|---|---|---|
| **Sliding-window attention** | Each token attends to only the last w tokens: O(n·w) | ⭐ **The attention inside the loop** |
| **DSA** (DeepSeek Sparse Attention; also in GLM-5) | A tiny learned "indexer" picks the top-k tokens for each query. The indexer is technically O(n²) but very cheap. | **Coda global layer, v1** for long context |
| HISA / MISA (2026) | Hierarchical / mixture-of-indexer versions that make the DSA indexer cheaper | Same slot as DSA, later |
| NSA, MoBA | Block-level sparse selection. Reported to lag DSA on retrieval. | Fallback options for the same slot |
| Longformer/BigBird, Reformer (LSH), Linformer/Performer/Nyströmformer | Older global-token / hashing / low-rank approximations | Mostly superseded; skip |

### C. In-between / mixed
| Part | Idea | Gogeta-0 fit |
|---|---|---|
| **Log-linear attention** (2025) | Hierarchical states, O(n log n): sits between a fixed-size RNN state and full KV | Alternative coda layer if SWA + GDN can't recall well enough |
| **NAtS-L** (ICML 2026) | Each *token* picks linear or softmax attention within the same layer | 🃏 Fits the adaptive-compute theme; per-token loop count plus per-token attention type. Later. |

### Not actually sub-quadratic (but often mistaken for it)
- **FlashAttention:** linear *memory*, still quadratic *compute*. We'll still use it for the SWA and coda layers.
- **Ring attention:** spreads the quadratic cost across GPUs. **MLA / KV compression:** shrinks the cache, not the compute.

### Notes for the fusion
- **Ternary:** the q/k/v/out projections of GDN, Mamba and SWA can go ternary. Gates, decays and the recurrent state stay in higher precision.
- **Speculative decoding:** rolling back SWA is trivial (truncate the cache). GDN/Mamba state needs per-position checkpoints (see the inference section).
- **Kernels:** `flash-linear-attention` (fla) has Triton kernels for GDN, Mamba-2, GLA, RWKV and more. **Check that it runs on aarch64 + Blackwell**; if not, fall back to pure-PyTorch chunked forms.
- **Smoke-test honesty:** at 512–1k tokens, quadratic attention costs almost nothing. These choices show their value only in long-context tests: add an **8k–32k needle-in-a-haystack / RULER-style eval** to the ladder.

---

## Prior art to read / check

- **Hybrids:** Jamba (AI21), Zamba / Zamba2 (Zyphra, shared attention block), Samba (Microsoft), Mamba-2 (SSD).
- **Looped / recurrent depth:** Universal Transformers; "Looped Transformers as Programmable Computers"; Geiping et al. 2025 "recurrent depth" (Huginn, 3.5B); Mixture-of-Recursions (2025).
- **BitNet:** BitNet (1-bit), BitNet b1.58, BitNet b1.58 2B4T (open weights), bitnet.cpp.
- **Crossovers:** Bi-Mamba (binarized Mamba). Is anyone doing ternary + looped? → **search this first; it's the novelty check.**
- **⚠️ Closest prior art found so far:** "Tiny Recursive Reasoning with Mamba-2 Attention Hybrid" (arXiv 2602.12078), a TRM-style recursion with a Mamba-2/attention hybrid. That's 2 of our 3 fighters, so **read it before building anything.** Also: TRM, "Less is More: Recursive Reasoning with Tiny Networks" (arXiv 2510.04871); HRM; Probabilistic TRM (arXiv 2605.19943).

---

## Feasibility

### What's plausible at hobby scale
- Train **~20–150M param** models from scratch on TinyStories, a FineWeb-Edu subset, or synthetic reasoning tasks (arithmetic, copying, graph search). Looped models shine on algorithmic tasks, so they're a good place for quick, cheap signal.
- Ternary QAT is just a custom `Linear` with straight-through estimator (STE) and absmean quantization. Easy to implement in PyTorch.
- Mamba-2 has reference kernels (`mamba_ssm`). A pure-PyTorch fallback is slow but fine for small runs.

### Hard parts / risks
- **Stability squared.** Looping and ternary weights are both unstable on their own. Expect a lot of LR / norm / init tuning.
- **No speedup without kernels.** Ternary weights only pay off with custom kernels. In plain PyTorch they're fake-quantized fp16, so they'll train *slower*. Fine for research, but don't expect wall-clock wins.
- **Backprop through loops.** Memory grows with r. Use truncated backprop through the last k loops, like Huginn.
- **The SSM scan in low precision.** Likely needs fp16/fp32 state, so only the projections go ternary.
- **Evaluation.** Small models score at the noise floor on standard benchmarks. Use synthetic tasks plus perplexity-per-bit / per-FLOP comparisons.

### Local setup (checked 2026-11-22)
- **GPU:** RTX Spark N1X (Blackwell, 6144 cores). nvidia-smi reports ~96 GB, so **memory is not the constraint; compute and bandwidth are.**
- **Native Windows (ARM64):** only CPU builds of PyTorch exist, so it's no good for training.
- **WSL Ubuntu 24.04 (aarch64):** sees the GPU ✅, and CUDA PyTorch wheels exist for Linux aarch64 (cu128/cu130) ✅. **Train here.** It still needs pip/venv installed, and WSL only has ~14 GB of system RAM (can be raised in `.wslconfig`).
- Start with a **pure-PyTorch Mamba-2** (SSD chunked scan). The official `mamba_ssm` CUDA kernels on aarch64 + Blackwell may not compile, so try them later.

**Size tiers** (rough guesses; measure real throughput first):
| Tier | Unique params | Data | Wall-clock |
|---|---|---|---|
| Smoke test | ~10M (acts like ~25M at 4 loops) | TinyStories (~0.5B tok) + synthetic tasks | ~1–3 h |
| Real test | ~50M | FineWeb-Edu ~1B tok | ~overnight to ~1 day |
| Ceiling-ish | ~150M+ | 3B+ tok | days to weeks: compute-bound |

Keep the **vocab small** (~8k BPE). At ~10M params, a GPT-2 50k vocab would make the embedding bigger than the rest of the model.

### Hardware tie-in? 🤔
Ternary math is adds and subtracts only, which suits FPGAs well. Possible crossover with the RISC-V FPGA research later.

---

## Inference side: speculative decoding (DFlash / DSpark)

- **DFlash** (ICML 2026): a small *block-diffusion* drafter proposes a whole block of tokens in parallel, conditioned on the target model's hidden features. The target verifies the block in one pass. The authors report up to ~6× lossless speedup.
- **DSpark** (2026): confidence-scheduled speculative decoding with semi-autoregressive drafting.
- **Why it helps in general:** generating text is limited by memory bandwidth, not compute. Verifying k drafted tokens costs about the same as generating 1, so speculative decoding turns idle compute into throughput.

**How it fits with FusionDance:**
- 🟡 **Overlap.** Ternary weights mean ~10× fewer bytes per token, and looping reuses the same weights r times. Both already make generation less bandwidth-bound, so speculative decoding has less to win. At ~10M params the whole model fits in GPU cache anyway, so it's pointless for the smoke test. It only starts to matter at much larger scale.
- 🔴 **SSM rollback.** If drafted tokens get rejected, attention just truncates its cache. The SSM's recurrent state has to be checkpointed at each drafted position, or recomputed. Looping multiplies that: one state per loop per position.
- 🟢 **Free drafter.** A looped model can draft for itself. Run 1–2 loops to draft and the full r loops to verify, with the same weights and no extra model (self-speculation, like LayerSkip-style early exit). Training with a random loop count (already planned) should make low-loop drafts decent for free.
- 🟢 **DFlash-style head.** Attach a small block-diffusion drafter to early-loop features instead of the final layer. This needs our own drafter training, and DFlash's training recipe wasn't public at last check.

**Plan:** don't build it yet; design for it now.
1. Make the SSM state checkpointable per position.
2. Keep random-loop training.
3. Once the model trains, measure **draft acceptance rate for r=1/2 vs full r**. That number decides whether self-speculation is worth building.

---

## The Rival Loop: critique → fix inside the recurrence

Goku and Vegeta are rivals, so build the rivalry in. In the agent version, a generator drafts, a critic finds flaws, a fixer revises, and this repeats until the critic is happy. The latent version: each loop iteration is a refinement step, so make some loops *critique* and others *fix*.

**The catch.** Agent critique loops work because the critic gets *outside* information: tests, compiler errors, a fresh context. A critic that shares the same weights and the same latent has no new information, and can learn to rubber-stamp. (LLMs mostly fail at self-correction without external feedback; Huang et al. 2023.) So the critic needs a **training signal grounded in the truth**, and the value at inference is mostly in **knowing when to stop** and **keeping each fix actually better**.

### Options, cheapest/most stable first
1. **Deep supervision + "fixes must fix" penalty.** Decode every loop through the coda and compute the loss at each loop. Add a penalty when loop k+1 is *worse* than loop k. This forces monotonic refinement. Almost free.
2. **Critic head.** A tiny head reads the latent at loop k and predicts *how wrong the current prediction is* (the per-token loss, or "will the next loop change the answer?"). It's trained against the real loss, so it's grounded. It gets used three ways:
   - **Halting:** stop looping when the critic says "good enough". This is the 30-minute fusion timer, learned (PonderNet / ACT style).
   - **Feedback:** inject the critic's error vector into the next loop's input, so the fixer gets told *what* is wrong, not just to try again.
   - **Speculative decoding:** the critic's confidence also predicts whether a low-loop draft will be accepted. One head with three jobs.
3. **Role-alternating loops.** Same shared weights, but add a learned role embedding (propose / critique) on alternate loops. Optionally add tiny role-specific adapters: two fighters sharing one body. Compare against a plain loop at the **same total r**: is an explicit critic step worth giving up a refinement step?
4. **Damage-and-repair training.** Corrupt the latent between loops and train later loops to recover. Start with random noise, then use *adversarial* noise: a PGD/FGSM-style worst-case perturbation of the hidden state. This is the genuinely adversarial version, and it makes each loop a learned denoiser, close to diffusion, which links up with the DFlash drafter idea. It costs about 1.5–2× training compute for the extra backward pass.
5. ~~GAN-style discriminator on outputs.~~ Discrete-text GANs are notoriously unstable. Ternary + looping + GAN would be three instabilities stacked: a guaranteed skinny Gotenks. Skip.

### What to measure
- Does loss go down **monotonically** as r increases (1 → 2 → … → 8), and keep improving past the training r?
- With critic halting: **average loops used at equal loss** compared with fixed r.
- On synthetic tasks with injected errors (e.g. arithmetic with a corrupted carry): does the loop *repair* them?
- How well critic confidence predicts self-speculation acceptance.

### Outside-the-model version (for later)
For code specifically, the real critic is a test runner, which is the agent loop itself. A tiny model won't write code, but a FusionDance-style model at scale could use **adaptive loop depth as the inner critic** and **tests as the outer critic**.

---

## Training data & evals

### Data, by experiment stage
| Stage | Data | Size / notes |
|---|---|---|
| **Smoke test (~10M)** | **TinyStories** (`roneneldan/TinyStories`) | ~0.5B tokens, simple English. Tiny models produce coherent text on it. |
| **Loop / recall probes** | **Generated on the fly** (no download): multi-digit arithmetic, copying, selective copy, induction, parity, graph reachability, **MQAR** (multi-query associative recall) | Infinite and controllable. Train short, test long for length generalization. Corrupted-carry arithmetic for the Rival Loop. |
| | **Reasoning Gym** | ~100 procedurally generated reasoning tasks with verifiers |
| **Puzzles (looping showcase)** | **Sudoku-Extreme, Maze-Hard, ARC-AGI-1/2** (the HRM / TRM datasets) | Small training sets plus heavy augmentation. **Published tiny baselines to beat:** TRM (7M) gets 87.4% on Sudoku-Extreme, 85.3% on Maze-Hard, 45% on ARC-AGI-1. |
| **Real text (~50M)** | **FineWeb-Edu** `sample-10BT` (stream ~1B tokens) | Default choice. Educational web text hits above its weight for small models. |
| | **SmolLM-Corpus** (Cosmopedia v2 synthetic textbooks + Python-Edu + FineWeb-Edu dedup) | A ready-made mix that includes code |
| | **FineMath** | For a math-leaning mix |
| | Alternatives: DCLM-baseline, Zyda-2, **Common Corpus** (~2T, fully open licenses), **MixtureVitae** (422B, low copyright risk), **FinePhrase** (486B synthetic rephrased web) | Swap in if licensing matters or for data ablations |
| **Sample-efficiency angle** | **BabyLM** (10M / 100M word tracks) | Tests whether looping helps when data is scarce, which is the actual claim of looping |

Tokenizer: train our own **~8k BPE** on whatever mix we use.

### Evals
| What it tests | Eval | Notes |
|---|---|---|
| Language modeling | Held-out loss / perplexity, LAMBADA | The primary metric at small scale |
| Grammar | **BLiMP** (minimal pairs) | Gives real signal even for tiny models |
| Story quality | LLM-as-judge on TinyStories completions | Same idea as the original TinyStories paper |
| Zero-shot commonsense | lm-evaluation-harness: PIQA, ARC-Easy, HellaSwag, SciQ, WinoGrande | ⚠️ Near chance at ≤50M. Compare **differences between variants**, not absolute scores. |
| **Recall (linear vs attention)** | MQAR; SWDE / FDA / SQuAD-completion | The standard recall suite from the Based / Zoology / Gated DeltaNet papers. This is where GDN vs Mamba vs SWA shows up. |
| **Long context** | RULER, BABILong, passkey / needle-in-a-haystack at 8k–32k | Tests the sub-quadratic design |
| **Looping / Rival Loop** | Accuracy vs r; length generalization; error-repair rate; Sudoku/Maze | Should look like a curve that keeps improving as loops increase |
| **Efficiency** | Train/decode tokens/sec, peak memory, **loss vs FLOPs and vs weight bits**, average loops used, speculative acceptance rate | The FusionDance headline metric |

**Skip for now:** HumanEval, MBPP, GSM8K, MMLU. These sit at chance level below a few hundred million parameters.

---

## Experiment ladder (each step is one fusion at a time)

0. **Baseline:** tiny dense transformer at a fixed param budget.
1. **+SSM:** hybrid Mamba/attention, same params. Does long context get cheaper without losing recall?
2. **+Loop:** share the core block, loop r times. Does a smaller param count match the baseline on algorithmic tasks?
3. **+Ternary:** BitNet the linears. How much quality do we lose? Plot memory vs loss.
4. **Full fusion:** all three. Compare loss vs params, vs bits, and vs FLOPs.
5. **Bonus, the Dance:** adaptive loop count at inference. Does "thinking longer" help on harder inputs?
6. **Rival Loop:** deep supervision + critic head (halting/feedback), then role-alternating loops, then adversarial damage-and-repair. Each is compared at equal compute.

Metric to beat: **quality per bit of weight memory** at matched compute.

---

## Questions for me

- What GPU / compute budget do I have? (Sets the max model size.)
- Goal: a paper-style ablation, a toy that runs on a laptop/FPGA, or just learning?
- Dance (runtime merge) or Potara (train-from-scratch hybrid) first?

## Scratch / random ideas

-
