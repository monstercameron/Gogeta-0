# K7-analysis: applying the LR decision rule and writing the Phase 2b campaign

**Status:** done · **Verdict:** tuned LR_E10 = 3e-3, LR_E1 = 6e-3; wrote `phase2b-replication.campaign` · **Links:** parent `K7.md` · script `experiments/checks/k7_lr_sweep.py` · result `experiments/checks/results/K7.json` · queue Q5

## Log (oldest first)

### 2026-11-22 17:22 · v0.4.13 · Script registered
`k7_lr_sweep.py` committed with its rule logic tested on 8 synthetic cases; it also writes the next campaign (extension or `phase2b-replication`). (Source: DEVLOG v0.4.13)

### 2026-11-22 18:01 · v0.4.25 · First application
Rule applied after the 8 sweep runs: E10 3e-3 tuned, E1 edge-winner so extend. Time not recorded exactly. (Source: DEVLOG v0.4.25)

### 2026-11-22 18:09 · v0.4.26 · Final application
`k7_lr_sweep.py --write-campaign` re-run after the extension (K7.json ran_at 18:09:10, 0.2 s, CPU, commit 1290f8a v0.4.25-dirty). Queue Q5 done 18:09:32: "phase2b-replication.campaign written". Campaign: E10 x3 at 3e-3; E1 x3, E11 x3, E9 s0, E2 s0 at 6e-3; E0 skipped (LR_E10 = 3e-3). (Source: K7.json; queue.json Q5; DEVLOG v0.4.26)
