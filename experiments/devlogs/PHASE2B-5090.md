# Phase 2b-5090: an independent single-SKU replication of Phase 2b

**Status:** done · **Verdict:**
- E1 vs E10 is a tie: E1 is +0.08% worse, paired 95% CI spans 0 (n = 7). Same sign as the N1X test, where C5 is REFUTED.
- E11 is equivalent to E1 (C196 VERIFIED, TOST).
- The loop doesn't pay (E2 is 2.6% worse than E9; 1 seed).
- Cross-SKU loss matches within 0.0010 (R2 pass).

**Links:** results `experiments/checks/results/phase2b_5090.json`, `REPLICATION-stats.json` (N1X) · predictions `experiments/checks/PREDICTIONS.md` (fleet wave 1 plus amendment) · ledger C5, C6, C195, C196

## Log (oldest first)

### 2026-11-22 ~21:20 · v0.4.74 · Registered
- **Design:** the same 11 Phase 2b items on RTX 5090s, with E10 and E1 raised to 7 seeds (the power argument, outside review 21:10).
- **Predictions:** R1 (E1-vs-E10 sign matches the N1X), R2 (per-item val within ±0.004 of the N1X), R3 (E11 within ±1% of E1). Holm over R1 and R3.
- The N1X 3 + 3 test stays the confirmatory Q12.

### 2026-11-22 21:26–21:48 · v0.4.75 · Launch
- **Fleet:** 6 rented RTX 5090s (`gogeta-cockpit-w1-3`…`8`) at commit `8441ad0` (training code identical to the N1X campaign's), clean clones, the dataset rebuilt on each box and verified byte-identical, `--require-clean`.
- **Jobs:** 19 runs of 200M tokens (about 3.5–5.6 min each; E2 10.3 min) plus clean80 scoring of every checkpoint on the box. The six N1X checkpoints were also scored there.
- All rc = 0. (Source: each box's `worker.status`, pulled to `~/fusiondance-data/cloud/fleet/`.)

### 2026-11-22 21:55 · v0.4.79 · Result
Clean80 bpc on the 5090:

| arm | seeds | mean | values |
|---|---|---|---|
| E10 | 0–6 | 0.26772 | .2660 .2672 .2676 .2696 .2670 .2677 .2690 |
| E1 | 0–6 | 0.26795 | .2667 .2677 .2679 .2684 .2686 .2689 .2674 |
| E11 | 0–2 | 0.26710 | .2662 .2672 .2679 |
| E9 | 0 | 0.2773 | |
| E2 | 0 | 0.2845 | (r = 4 is best) |

- **R1:** E1 − E10 paired, +0.00022 (+0.08%), 95% CI [−0.00087, +0.00132], p 0.64 (Welch p 0.69).
  - The sign matches the N1X (E1 worse), but the CI spans 0, so this is a **tie** and not a confirmation of anything.
  - With the N1X's 3 + 3 (+0.18%), 10 seeds per arm say E1 is within about ±0.4% of E10 and not better.
- **R3:** E11 − E1, −0.00034 (−0.13%). TOST p 0.002 within the margin 0.00267 (1% of E1), so the two are **equivalent**: C196 VERIFIED.
- **Holm (m = 2):** neither comparison rejects equality.
- **R2:** **pass**. 6/6 items within 0.0010 val_loss of the N1X, and clean80 within 0.0008.
- **Descriptive:** E2 (looped) is 2.6% worse than its unlooped twin E9, and 6.2% worse than E1.

### 2026-11-22 21:55 · v0.4.79 · Ledger and decisions
- **C5:** evidence added; it stays REFUTED.
- **C6:** evidence added (the loop doesn't pay; 1 seed).
- **C195:** evidence added (200M, 5090 half), plus a limitation from the SQ232 replay noise.
- **C196:** new, VERIFIED (E11 ≈ E1).
- **Decision:** the mixer question at 13.8M is settled as a tie with E10. The next questions are scale (SQ243 two-size check), init/implementation parity (SQ234), and whether the hybrid's *efficiency* (SQ244) or loops (SQ207) can still earn their place.
