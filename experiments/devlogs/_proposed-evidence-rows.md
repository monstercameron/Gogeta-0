# Proposed evidence-pointer rows for `experiments/claims.md`

Pointer backfill, audit of v0.4.30 ledger (latest row per ID wins, as `experiments/claims_list.py` computes it). Rows below are **proposed append-only rows**: same ID, same status, no status change. Append them under a new section "Evidence pointer backfill (v0.4.31)". Nothing in `claims.md` has been edited. Every path below was checked to exist on disk at audit time.

## Counts

| Item | Count |
|---|---|
| Claims in the ledger (latest status) | 116 (72 UNVERIFIED, 21 VERIFIED, 16 PARTIAL, 5 REFUTED, 2 WRONG FOR OUR CODE) |
| Settled claims (latest status not UNVERIFIED) | 44 |
| Already had a concrete, existing pointer | 16 (C1, C2, C3, C4, C10, C80, C83, C90, C91, C92, C100, C101, C103, C106, C107, C128) |
| Fixed by a proposed row below | 27 |
| Nothing findable (listed separately) | 1 (C28) |
| Settled claims whose evidence cell fails the lint regex `(experiments\|evals\|notes\|repro\|gogeta\|tools)/\S+\|run [A-Z0-9-]+\|v\d+\.\d+\.\d+` today | 33 of 44 (tests/, post_training/, refs/ pointers are real but not in the regex; see the lint proposal in the report) |

## Proposed rows

| ID | Claim | Source | Status | Evidence / notes |
|---|---|---|---|---|
| C5 | The hybrid stack (E1) beats dense attention (E0) at 14M params / 200M tokens of Python | pointer backfill | `PARTIAL` | Runs `E0-20261122-122059-s0` and `E1-20261122-124440-s0` (v0.3.0, commit 0dc7491): `experiments/experiments.json` (E0 val 0.6063, E1 0.5846), curves `experiments/curves/E0.json` / `E1.json`, `experiments/devlogs/E0.md`, `experiments/devlogs/E1.md`. Note: val split near-duplicated with train (`experiments/checks/results/K9.json`, v0.4.33). (Source: pointer backfill v0.4.31) |
| C6 | Looping E2's core does not beat E0/E1 at ~2x the compute | pointer backfill | `PARTIAL` | Run `E2-20261122-132340-s0` (v0.3.0): `experiments/experiments.json` E2 (0.6186 vs E0 0.6063 / E1 0.5846), `experiments/devlogs/E2.md`; control `experiments/devlogs/E9.md` (not completed). (Source: pointer backfill v0.4.31) |
| C7 | E2's loss gets worse when looping beyond the trained range | pointer backfill | `PARTIAL` | `experiments/experiments.json` E2 `val_loss_r4/r8/r12` (0.6186 / 0.6328 / 0.6540), `experiments/curves/E2.json`, run `E2-20261122-132340-s0` v0.3.0; consistent with `experiments/checks/results/K6.json`. (Source: pointer backfill v0.4.31) |
| C8 | Our models compress held-out Python to ~3% vs xz/gzip | pointer backfill | `VERIFIED` | `notes/LABNOTEBOOK.md:161-165` (yardsticks measured on the 4,000 val docs: gzip -9 25.1%, xz -9 15.4%), DEVLOG v0.3.1; `experiments/devlogs/E0.md`. The one-off measurement script is not recorded; model side single seed. (Source: pointer backfill v0.4.31) |
| C9 | Ternary shrinks the E1 body 22.2 to 2.8 MB, whole model 28.5 to 9.1 MB | pointer backfill | `PARTIAL` | Computed: `gogeta/model.py:278` (`param_counts`), arithmetic recorded in `notes/LABNOTEBOOK.md:198`, `experiments/devlogs/E3.md`, DEVLOG v0.3.4. No script persists the packing arithmetic. (Source: pointer backfill v0.4.31) |
| C11 | Triton on this WSL setup needs `python3.12-dev` | pointer backfill | `VERIFIED` | `notes/LABNOTEBOOK.md:40` (12:00 entry, Python.h missing, fixed by installing python3.12-dev), DEVLOG v0.3.0. (Source: pointer backfill v0.4.31) |
| C12 | SWA is O(n*w) | pointer backfill | `WRONG FOR OUR CODE` | `experiments/checks/results/K1.json` (SWA time exponent 1.74, memory 1.95), `experiments/cards/K1.md`, `experiments/devlogs/K1.md`, v0.4.0. (Source: pointer backfill v0.4.31) |
| C13 | Ternary weights make the model "~10x smaller" | pointer backfill | `REFUTED` | Computed 7.8x body / 3.1x whole: see C9 pointers `gogeta/model.py:278`, `notes/LABNOTEBOOK.md:198`, `experiments/devlogs/E3.md`, v0.3.4. (Source: pointer backfill v0.4.31) |
| C14 | Ternary makes compute cheaper | pointer backfill | `WRONG FOR OUR CODE` | `experiments/checks/results/K2.json` (forward 5.58x / 4.06x slower), `experiments/cards/K2.md`, `experiments/devlogs/K2.md`, v0.4.1. (Source: pointer backfill v0.4.31) |
| C15 | GDN fixed memory / linear time | pointer backfill | `VERIFIED` | `experiments/checks/results/K1.json` (GDN time exponent 0.66, memory 0.98, streaming diff 0.0), `experiments/cards/K1.md`, `experiments/devlogs/K1.md`, v0.4.0. (Source: pointer backfill v0.4.31) |
| C17 | Ternary gap shrinks with size | pointer backfill | `PARTIAL` | `experiments/checks/results/K4.json` (15.3 / 13.8 / 10.5% at d=64/128/256), `experiments/cards/K4.md`, `experiments/devlogs/K4.md`, v0.4.3. (Source: pointer backfill v0.4.31) |
| C18 | Looping adds state-tracking depth | pointer backfill | `VERIFIED` | `experiments/checks/results/K5.json` and `experiments/checks/results/sq_06.json`, `experiments/devlogs/K5.md`, `experiments/devlogs/SQ6.md`, v0.4.4 / v0.4.6. (Source: pointer backfill v0.4.31) |
| C20 | E2's loop converges | pointer backfill | `REFUTED` | `experiments/checks/results/K6.json` (relative change 0.085 at k=12), `experiments/cards/K6.md`, `experiments/devlogs/K6.md`, v0.4.5. (Source: pointer backfill v0.4.31) |
| C22 | A rotating state solves parity in one pass | pointer backfill | `REFUTED` | `experiments/checks/results/sq_06.json` (RotSSM 84.0%), `experiments/cards/SQ6.md`, `experiments/devlogs/SQ6.md`, v0.4.6. (Source: pointer backfill v0.4.31) |
| C43 | GDN with negative eigenvalues solves state tracking in one pass and length-generalises | pointer backfill | `VERIFIED` | `experiments/checks/results/sq_06.json` arm (e): 100.0% / 99.99%, `experiments/devlogs/SQ2.md`, `experiments/devlogs/SQ6.md`, v0.4.6. (Source: pointer backfill v0.4.31) |
| C44 | Single-pass GDN parity accuracy has high run-to-run variance | pointer backfill | `VERIFIED` | `experiments/checks/results/K5.json` (81.5%) vs `experiments/checks/results/sq_06.json` arm (a) (73.3% / 50.1%), `experiments/devlogs/K5.md`, v0.4.6. (Source: pointer backfill v0.4.31) |
| C81 | TRL's fused LM-head log-probs match log_softmax of our logits | pointer backfill | `VERIFIED` | `post_training/test_hf.py` (10/10 at v0.4.25), `experiments/devlogs/EVAL-SUITE.md`. (Source: pointer backfill v0.4.31) |
| C82 | GRPO, SFT and DPO run end to end through TRL with our wrapper | pointer backfill | `PARTIAL` | `post_training/grpo_smoke.py`, `post_training/sft_dpo_smoke.py`, numbers recorded in `post_training/SETUP.md:118` (SFT 8.947 to 8.210); CPU only; v0.3.x WIP `ea3aa5b`. (Source: pointer backfill v0.4.31) |
| C84 | Without a decode path, RL sampling is ~340x more work than cached decoding | pointer backfill | `PARTIAL` | Computed: `post_training/SETUP.md:201`. Note: the decode blocker was removed by SQ70, see `experiments/checks/results/sq_hw_fla_recurrent.json` (v0.4.29). (Source: pointer backfill v0.4.31) |
| C95 | E1 vs E0 is matched tokens, near-matched params, not matched compute | pointer backfill | `VERIFIED` | Computed: `gogeta/model.py:278` (`param_counts`), `gogeta/model.py:289` (`flops_per_token`); registry `experiments/experiments.json` (E0/E1 flops 1.935e16 / 1.821e16); `experiments/devlogs/E1.md`. (Source: pointer backfill v0.4.31) |
| C104 | OpenCoder's annealing corpus was not decontaminated against HumanEval/MBPP | pointer backfill | `PARTIAL` | Consistent with C103: `evals/results/contamination-2026-11-22.json`, `notes/benchmark-suite.md` (OpenCoder section); the paper itself read only partly. (Source: pointer backfill v0.4.31) |
| C105 | rewards.py's sandbox child cannot write to the repo | pointer backfill | `REFUTED` | Control write succeeded via `post_training/rewards.py:175` (`run_tests`) and failed via `evals/sandbox.py`; DEVLOG v0.4.19 "Safety finding"; `experiments/devlogs/EVAL-SUITE.md`. (Source: pointer backfill v0.4.31) |
| C108 | A 14M all-attention model does in-context key-value recall (`gogeta_recall_d*`) | pointer backfill | `REFUTED` | `evals/results/check-E0-s0-tier1ll-tier3-tier1-20261122-172441.json` (tier 3 recall), `notes/benchmark-suite.md:109`, v0.4.19. (Source: pointer backfill v0.4.31) |
| C109 | lm-eval's `mbpp_plus` scores only MBPP's 3 base asserts, not EvalPlus's extra tests | pointer backfill | `VERIFIED` | `evals/tasks/tier2/gogeta_mbpp_plus_exec.yaml`, `evals/tasks/gogeta_utils.py`, v0.4.19. (Source: pointer backfill v0.4.31) |
| C110 | The C2 NaN is a call-site bug in our tests | pointer backfill | `VERIFIED` | `experiments/checks/results/sq_hw_fla_recurrent.json` (positional inf, keyword 2.98e-08), `experiments/devlogs/SQ70.md`, v0.4.29. (Source: pointer backfill v0.4.31) |
| C111 | fused_recurrent_gated_delta_rule is usable for decoding on this GPU | pointer backfill | `VERIFIED` | `experiments/checks/results/sq_hw_fla_recurrent.json` (decode 5.97e-05 / 4.91e-04, 0.036 ms per step), `experiments/devlogs/SQ70.md`, v0.4.29. (Source: pointer backfill v0.4.31) |
| C130 | The 3D page Play trace for E2 is the model's real computation | pointer backfill | `PARTIAL` | `docs/data/token-trace.json`, `experiments/viz/export_trace.py`, checkpoint run `E2-20261122-132340-s0` (v0.3.0), DEVLOG v0.4.22 (traced forward equals `model.forward` to 9.5e-6; CPU val loss 0.6423 vs 0.6186). (Source: pointer backfill v0.4.31) |

## Settled claims with no findable evidence

| ID | Latest status | Why nothing was found | Recommended downgrade |
|---|---|---|---|
| C28 | `PARTIAL` | The evidence cell says only "SQ4 (functional eval)". SQ4 never ran (`experiments/side-quests.md`: `idea`); the Tier-1 eval-suite run on E1/E2 (SQ60 / Q14) is still queued. The only local observation is anecdotal: E1 had the best loss but the least coherent samples (`notes/LABNOTEBOOK.md:127-131`, `experiments/devlogs/E1.md`), and the claim's main content (a larger n-gram vocabulary lowers loss while benchmarks saturate) is a paper result. | `UNVERIFIED` until SQ60 runs, or keep `PARTIAL` worded as "observed anecdotally in E1 samples only" with the LABNOTEBOOK pointer. |

## Notes on already-valid rows that a stricter reader may want tightened (not proposed as rows)
- C8 and C128 name "notebook" in prose; the audit counted C128 valid (a `refs/12-hardware/` PDF path exists) and C8 fixed above.
- C106 cites `python -m evals.sandbox` (module `evals/sandbox.py` exists); C107 cites a glob `evals/results/check-E0-s0-exec-limit8-*.json` that resolves to one file.
- Rows citing `tests/`, `post_training/` or `refs/` are concrete but fail the proposed lint regex unless those prefixes are added to it.
