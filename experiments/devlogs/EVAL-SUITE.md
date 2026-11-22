# EVAL-SUITE: standard benchmark suite (lm-evaluation-harness) and its E0 probe runs

**Status:** tooling done; E0 CPU probes run; GPU tier-1 run on E0/E1/E2 queued (Q14 / SQ60) · **Verdict:** wiring 6/6; E0 per-document val 0.2208 bits/char vs 0.2351 packed (-6.1%, protocol gap); HumanEval/MBPP training-data contamination found; sandbox gap found · **Links:** `evals/` · notes `notes/benchmark-suite.md` · results `evals/results/*.json` · ledger C100-C109 · side quests SQ60-SQ69 · queue Q14, Q30, SQ61

## Log (oldest first)

### 2026-11-22 17:24 · v0.4.14 · Queued
Q14 (benchmark suite tier 1) is queued needing the human for downloads and installs; later covered by the "eval-suite request" approval. (Source: DEVLOG v0.4.14, v0.4.19; queue.json Q14)

### 2026-11-22 (time not recorded) · v0.4.19 · Environment
Venv `~/.venvs/gogeta-eval`, pinned in `requirements-eval.txt`, lm-eval 0.4.13 (install logs `gogeta-eval-pip.log`, `gogeta-eval-torch.log`); about 14 MB of eval datasets in `~/fusiondance-data/eval-cache`, no model weights. A subagent built the tooling; the main agent checked its key claims. (Source: DEVLOG v0.4.19; logs directory)

### 2026-11-22 12:20 · v0.4.19 · Packed vs per-document protocol check
`evals/results/check-packed-vs-doc-E0-20261122-122059-s0.json` (via `evals/checks/packed_vs_doc.py`): train.py's protocol re-run on CPU fp32 gives 0.606254 vs 0.606341 recorded (0.014%); first 64 window positions loss 1.060 vs 0.576 after. The filename's timestamp is the E0 run id, not a run time; the run time is not recorded. (Source: C101 evidence; file name)

### 2026-11-22 17:18 · v0.4.19 · E0 CPU runs (times from result file names)
Results written to `evals/results/`: `check-E0-s0-valbpb-full-tier1-20261122-171816.json` (17:18:16), `check-E0-s0-tier1ll-tier3-tier1-20261122-172441.json` (17:24:41), `check-E0-s0-exec-limit8-tier1-20261122-172825.json` (17:28:25), `check-E0-s0-cruxeval-samples-tier1-20261122-172837.json` (17:28:37), `smoke-random-tiny_attn-tier1+3-20261122-173426.json` (17:34:26) and `smoke-random-tiny_attn-tier2+2s-20261122-173751.json` (17:37:51). Command lines not recorded (entry point `python -m evals.run_suite`). `evals/results/contamination-2026-11-22.json` from `evals/contamination.py`, time not recorded. (Source: file names; DEVLOG v0.4.19)

### 2026-11-22 17:44 · v0.4.19 · Results
Wiring tests 6/6. E0 val 0.2208 bits/char per document vs 0.2351 packed (-6.1%, missed the +-3% prediction; protocol, C101). HumanEval reference-solution bpb 0.5106 (clean 0.5174); MBPP 0.6398 (clean 0.6544); CRUXEval-O exact match 8.9% (always "False" scores 3.6%); recall at chance (C108 REFUTED, probe needs redesign). Contamination: 32/164 HumanEval problems and 43/500 MBPP statements verbatim in training data (C103). Version cut 17:44:49. (Source: DEVLOG v0.4.19; evals results)

### 2026-11-22 17:44 · v0.4.19 · Incident: sandbox gap (C105 REFUTED)
Main agent verification: `evals/sandbox.self_test()` 11/11; a control write into the repo is blocked by `evals/sandbox.py` but succeeds through `post_training/rewards.run_tests` (drvfs mode 777 under /mnt/c). Also `GogetaForCausalLM` failed to build (missing `gdn_neg_eigval`). `post_training/SETUP.md` and AGENTS section 6 corrected; SQ61 queued. (Source: DEVLOG v0.4.19)

### 2026-11-22 18:05 · v0.4.25 · Fix
Q30 done 18:05:29: `GogetaConfig` gained `gdn_neg_eigval`; `post_training/test_hf.py` 10/10, `evals/test_wiring.py` 6/6 (CPU). Ledger C80 VERIFIED again. (Source: queue.json Q30; DEVLOG v0.4.25)

### 2026-11-22 (time not recorded) · v0.4.33 · Caveat from K9
Per-document val and solution-likelihood numbers for E0 inherit the near-duplicate val split that K9 found (41.7% of val docs). C107 (functional pass@1 on val docs) already noted that near-duplicates were not ruled out. See `experiments/checks/results/K9.json`. (Source: coordinator message; claims.md C107)
