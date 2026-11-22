# SQ2: negative-eigenvalue Gated DeltaNet

**Status:** done (run as arm (e) of SQ6) · **Verdict:** 100.0% on positions 32-63 and 99.99% on 64-127 in one pass, vs looped GDN 97.9% / 61.2% (2 seeds): C43 VERIFIED for parity · **Links:** `SQ6.md` (full log) · card `experiments/cards/SQ6.md` · results `experiments/checks/results/sq_06.json` · ledger C43 · follow-ups E11, Q18/Q19, Q22

## Log (oldest first)

### 2026-11-22 (time not recorded) · pre-v0.4.0 · Idea logged
SQ2 listed as `idea`: `allow_neg_eigval` in fla's kernel (beta in (0,2)), theory says it enables state tracking. (Source: side-quests.md)

### 2026-11-22 17:10 · v0.4.6 · Run inside SQ6
Prediction (e) >= 90% and "SQ2 promising if (e) >= (a) + 20 points" registered in 4a3e361; run as part of SQ6 (see `SQ6.md` for command, commit and time). Result (e) beat (a) by +38.3 points. Standout result so far. (Source: sq_06.json; card SQ6)

### 2026-11-22 17:19 · v0.4.6 · Ledger and follow-ups
Side-quest table: SQ2 `done` ("promising, top result so far"); C43 VERIFIED. Follow-ups: E11 (LM cost, registered v0.4.7), looped neg-eig parity (Q18/Q19), neg-eig inside the looped core (Q22), decision memo Q23. (Source: side-quests.md outcomes; DEVLOG v0.4.6, v0.4.7; queue.json)
