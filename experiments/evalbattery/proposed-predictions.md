# Proposed predictions (A4, evaluation battery). Written BEFORE the runs they cover.

Proposed text for `experiments/checks/PREDICTIONS.md` (the main agent integrates it; this file is not the registered copy).
Written 2026-11-22. None of the GPU runs below exist yet. The CPU numbers quoted in `notes/eval-battery.md` were
produced before this file was written and are labelled as diagnostics there; they are not what is predicted here.

## B1. MQAR v2 validity gate (queue QB1; then QB2 = the Q16 replacement)
- **Prediction.** With the v2 recipe (mastery curriculum, `mqar.py --mastery`), 2-layer d=128 attention reaches
  >= 90% on the K3 geometry (P=64, T=256, 256 keys x 256 values) within 30,000 steps on 2 of 2 seeds (confidence 60%;
  the plateau length at P=64 is extrapolated from P <= 16). swa(w=4) <= 1/64 + 5 = 6.6%.
  If attention misses 90% at 30k steps the gate fails, and the fallback geometry (P=16, T=64) becomes the registered
  K3 v2 geometry (confidence that attention clears it: 85%).
- **Criterion.** Task VALID iff attn mean over 2 seeds >= 90% AND swa(4) <= 1/P + 5 points. Otherwise the family is
  `unvalidated` and C16/SQ59 stay UNVERIFIED. Arms are interpreted only after the gate passes.
- **Arm predictions (meaningful only if valid).** Plain linear attention 10-60% at P=16 (a 32x32 state cannot hold 16
  exact bindings), GDN 40-95%, GDN neg-eig within +-10 points of GDN.

## B2. Code-native state-tracking probe on real checkpoints (queue QB4)
- **Prediction.** E0, E1, E2, E10 are all at floor on all 18 cells: lower 95% CI bound < 35% on at least 16 of 18
  cells for every checkpoint (confidence 85%); `incr` cells fall below 25% for at least one checkpoint (stale-value
  bias, as E0 shows: 15.0% and 17.3%). No ordering between hybrid and attention is expected.
- **Criterion.** A (checkpoint, cell) pair is `interpretable` iff its lower CI bound >= 35%. Claims use only
  interpretable pairs. If none exists the probe is reported as `floor` and no memory claim cites it.

## B3. Language-modelling controls (queue QB4, all checkpoints)
- **Prediction.** bpb(line-shuffled) - bpb(original) >= 0.15 for every trained checkpoint (confidence 95%; E0 on 150
  docs: +0.547). Fresh GitHub Python bits/byte exceeds in-distribution val by 0.3-1.0 for all models.
- **Criterion.** Gate = gap >= 0.15 bits/byte. A checkpoint failing it has a broken evaluation path and is excluded.

## B4. Functional family on real checkpoints (queue QB5)
- **Prediction.** Greedy pass@1 on HumanEval+ and MBPP+ is < 5% for E0 and E1 (confidence 80%: the smallest published
  models, 12M-160M, sit at 2-4% with 100x our tokens). The canonical-solution harness gate passes 100%.
- **Criterion.** `floor` if pass@1 < 5%; then F is reported but excluded from claim tables.

## B5. Iterative-computation suite, from-scratch tiny models (queue QB3)
- **Prediction (d=128, 20k steps, 2 seeds, GPU).**
  - parity and modadd5: GDN neg-eig 1-pass >= 90% on id and len2x (SQ6: 100.0% / 99.99% on parity); attention 1-pass
    < 60% on len2x; looped GDN (not neg-eig, x4) id >= 90% but len4x < 70% (SQ6: 61.2% at 2x).
  - graph_hops: 4-layer unlooped attention clears 90% on id (k <= 4) only after the plateau; if it does not within 20k
    steps the task is `unvalidated`. Looped x4 1-layer attention beats the 1-pass 1-layer by >= 10 points on id
    (confidence 55%) and degrades more slowly on gen (k=5..8) than the 4-layer stack (confidence 35%).
  - program: id >= 90% for 4-layer attention; gen (L=7..12) lower than id by >= 15 points for every arm.
  - label-shuffled controls: within chance +- 3 points (confidence 95%; the CPU runs already show 10.0% / 49.8%).
- **Criterion.** The validity gate of `notes/eval-battery.md` section 3.3. "Looping adds useful computation" needs
  looped > unlooped at matched compute (4-layer stack vs 1-layer x4) on >= 2 tasks with paired CIs excluding 0,
  and a smaller id-to-gen drop.
