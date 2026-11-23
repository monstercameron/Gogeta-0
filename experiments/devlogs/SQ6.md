# SQ6: rotation vs loops vs negative eigenvalues on parity (includes SQ2)

**Status:** done · **Verdict:** RotSSM 84.0% (below the 90% bar, C22 REFUTED at our scale) but generalises better than looping (76.5% vs 61.2% on positions 64-127); GDN with negative eigenvalues 100.0% / 99.99% in one pass (SQ2) · **Links:** card `experiments/cards/SQ6.md` · results `experiments/checks/results/sq_06.json` · script `experiments/checks/sq_06_rotation_parity.py` · SQ2 devlog `SQ2.md` · ledger C22, C43, C44 · follow-up E11

## Log (oldest first)

### 2026-11-22 (time not recorded) · WIP 4a3e361 · Mamba-3 reading and registration
Mamba-3 (arXiv 2603.15569) reportedly solves parity in one pass with a complex/rotary state; queued as SQ6 after K1-K6, with SQ2 (negative-eigenvalue GDN) folded in as arm (e). Prediction and criteria committed in `4a3e361` ("SQ6/SQ2 prediction (before running)", 16:37:43): (c) RotSSM >= 95% and largely holds on 64-127; (d) <= 60%; (a) < 75%; (e) >= 90%. C22 VERIFIED if (c) >= 90% and (c) >= (d) + 25; SQ2 promising if (e) >= (a) + 20. (Source: LABNOTEBOOK "Reading: Mamba-3"; git log; sq_06.json)

### 2026-11-22 (time not recorded) · v0.4.6 · Incident: first run invalid
The first SQ6 output (all arms about 50%) was invalid and was superseded; the agent fixed its setup and re-ran. What exactly was wrong in the first run is not recorded beyond "setup". (Source: LABNOTEBOOK "Variance" bullet)

### 2026-11-22 17:10 · v0.4.6 · Valid run
`~/.venvs/gogeta/bin/python experiments/checks/sq_06_rotation_parity.py`, GPU, 661 s (10 runs, 2 seeds), finished 17:10:55 (start about 16:59:54, derived), HEAD ea3aa5b (dirty), script uncommitted; GPU 55 C before, 58 C after. For about the first minute another agent's GPU job (`sq_kda_kernel.py`, SQ50) ran concurrently, affecting only timings. Deviation: RotSSM trained in the masked parallel form (the sequential complex scan measured about 0.37 s/step, over budget). (Source: sq_06.json; card SQ6)

### 2026-11-22 17:11 · v0.4.6 · Result (positions 32-63 / 64-127 at T=128, 2 seeds)
(a) GDN 1 pass 61.7 / 50.3; (b) looped x4 97.9 / 61.2; (c) RotSSM 84.0 / 76.5; (d) RotSSM theta=0 50.2 / 50.1; (e) GDN negative eigenvalues 100.0 / 99.99. (Source: card SQ6; sq_06.json mtime 17:11:22)

### 2026-11-22 17:11 · v0.4.6 · Observation: variance
(a) did not reproduce K5 (i) (73.3% and 50.1% vs 81.5%) with identical init and data (identical step-0 loss); attributed to GPU nondeterminism amplified by an unstable task. (b) reproduced K5 (ii). Hence C44: at least 2 seeds needed for 1-pass GDN parity. (Source: card SQ6 caveats; LABNOTEBOOK)

### 2026-11-22 17:19 · v0.4.6 · Ledger and decisions
C22 REFUTED at our scale; C43 added VERIFIED (parity, 2 seeds): a one-flag change to GDN does what the loop was supposed to do for state tracking and length-generalises. Decision (proposed): test neg-eig GDN for language modelling (E11) and inside the looped core before more loop work. Commit 17:19:34. (Source: DEVLOG v0.4.6; LABNOTEBOOK; claims.md verdicts)

### 2026-11-22 19:30 · v0.4.59 · Claim audit triage
C18, C43 (n=2 seeds) and C44 (2 distinct seeds, two protocols) downgraded VERIFIED to PARTIAL under the statistics policy; C22 stays REFUTED, requalified as the local analogue [exploratory n=2]. (Source: notes/reviews/claims/2026-11-22-1902-b1/b5/b6/b7.md Triage)
