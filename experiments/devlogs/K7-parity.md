# K7-parity: E10 construction matches E0, FusionDance unchanged

**Status:** done · **Verdict:** PASS: FusionDance bit-identical for E0/E1/E2/E3/E9; E10 equals E0 in params (13,767,552 / 10,621,824) and FLOPs/token (9.676e7) · **Links:** card `experiments/cards/K7-parity.md` · results `experiments/checks/results/K7-parity.json` · script `experiments/checks/k7_e10_parity.py` · ledger C90, C91

## Log (oldest first)

### 2026-11-22 (time not recorded) · v0.4.12 · Prediction
Prediction and pass criteria are inside the result JSON: (a) bit-identical state_dicts, configs, RNG state; (b) E10 params and FLOPs within 0.5% of E0; (c) init loss within 0.5 of ln(8192). Where it was committed before the run: not recorded. (Source: K7-parity.json)

### 2026-11-22 17:18 · v0.4.12 · Run
CPU, 5.4 s, commit ea3aa5b (v0.3.5-4-gea3aa5b-dirty). Result: all (a)-(c) pass; E10 init loss 9.094. (Source: K7-parity.json ran_at 17:18:03; DEVLOG v0.4.12)

### 2026-11-22 (time not recorded) · v0.4.12 · GPU replay caveat
A 101-step GPU replay of E0 matched the v0.3.0 log at step 0 (9.0900) but read 2.8370 vs 2.8460 at step 100; probably GPU nondeterminism, unproven. C91 PARTIAL; an old-vs-new replay was meant to be the first Phase 2 item (not recorded as run). (Source: LABNOTEBOOK Phase 1; DEVLOG v0.4.12)

### 2026-11-22 17:22 · v0.4.12 · Version cut
Commit 17:22:55. Ledger C90 VERIFIED (computed), C91 PARTIAL. (Source: git log; claims.md Science-gaps section)
