# K7-stage1: the 8-run LR sweep (E10 and E1 at 20M tokens)

**Status:** done (7 of 8 planned runs in stage 1 plus item 1; see log) · **Verdict:** E10 best 3e-3 (1.0286); E1 best at the grid edge 6e-3 (0.9925) so extend · **Links:** parent `K7.md` · results `experiments/checks/results/K7.json` · campaigns `phase2a-k7-lr`, `phase2a-k7-lr-resume` · ledger C92

## Log (oldest first)

### 2026-11-22 17:25 · v0.4.14 · Item 1/8: E10 lr 1e-3
`E10 --seed 0 --lr 1e-3 --max-tokens 20000000`, run `E10-20261122-172534-s0`, 17:25:37-17:27:37, commit e46abd2 (v0.4.14-dirty), val 1.2385, 0.4802 bits/char, 1.86 min. (Source: run summary; campaign log)

### 2026-11-22 17:28 · v0.4.16 · Campaign aborted
Preflight refused item 2 because of untracked `tools/viz/export_arch.py`. See `K7.md`. (Source: campaign log 17:28:48)

### 2026-11-22 17:35 · v0.4.18 · Resume, items 2-8 (campaign `phase2a-k7-lr-resume`, commit 48862f9 / v0.4.18 onward)
| Item | Run id | Start - end | Commit | Val |
|---|---|---|---|---|
| E10 2e-3 | `E10-20261122-173601-s0` | 17:36:03 - 17:37:59 | 48862f9 | 1.0650 |
| E10 3e-3 | `E10-20261122-173914-s0` | 17:39:16 - 17:41:16 | 48862f9 | 1.0286 |
| E10 6e-3 | `E10-20261122-174231-s0` | 17:42:33 - 17:44:31 | 48862f9 | 1.0889 |
| E1 1e-3 | `E1-20261122-174549-s0` | 17:45:55 - 17:49:09 | a22ac48 | 1.1625 |
| E1 2e-3 | `E1-20261122-175030-s0` | 17:50:33 - 17:53:22 | 0d5ca31 | 1.0367 |
| E1 3e-3 | `E1-20261122-175440-s0` | 17:54:43 - 17:57:25 | f8d4ed5 | 1.0071 |
| E1 6e-3 | `E1-20261122-175844-s0` | 17:58:47 - 18:01:25 | 2d06d95 | 0.9925 |

Commits differ only by non-code-path changes (the check "code paths identical" is stated in the card). (Source: run summaries; campaign resume log; card K7)

### 2026-11-22 18:01 · v0.4.25 · Rule applied
`k7_lr_sweep.py` applied the decision rule: E10 bracketed at 3e-3 (2e-3 +3.54%, 6e-3 +5.86%, 1e-3 +20.4%); E1 best at 6e-3 with 3e-3 +1.47%, 2e-3 +4.46%, 1e-3 +17.1%: extend to 1.2e-2. Scorecard: 3 hits, 3 misses (E1 best LR higher than predicted; curves sharper than "flat within 1%"; 1e-3 was +17-20%, not +3-8%). Exact time the analysis ran not recorded (K7.json holds the final run at 18:09:10). (Source: card K7; DEVLOG v0.4.25)

### 2026-11-22 18:06 · v0.4.25 · Version cut
v0.4.25 recorded stage 1 and the hf.py fix (Q30). C92 -> PARTIAL, C80 -> VERIFIED. (Source: DEVLOG v0.4.25; git log 18:06:02)
