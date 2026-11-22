# Proposed PREDICTIONS: SQ140 (same-seed determinism) and SQ141

Append to `experiments/checks/PREDICTIONS.md` and commit BEFORE any SQ140 run. Written 2026-11-22; no SQ140 run exists.

## SQ140: same-seed determinism, E0 and E1, default vs --deterministic (claims C183-C186)

**Design.** Seed 0, `--max-tokens 10000000` (305 steps), configs `e0_dense.json` and `e1_hybrid.json` (each at its
recorded LR), `--compile` on. Four arms, each run twice with the identical command: E0-default, E0-det, E1-default,
E1-det; plus one E0-default pair with `--compile` off. Per pair record: train loss at steps 0/10/50/100/300, final
full-val loss, checkpoint sha256, number of weight tensors that differ and their max abs delta.

**Prediction (numbers fixed now).**
1. Step-0 loss is identical to 4 decimals in every pair (seed, init and data order are deterministic).
2. E0-default pair: checkpoints NOT bitwise equal (85%); abs(d val_loss) in 0.0005-0.004; step-100 train-loss gap 0.001-0.02.
3. E0-det pair: bitwise equal checkpoints (65%). If not, the residual comes from SDPA backward or Inductor, and the
   warn_only warnings in the log name the op. Throughput cost of det mode: 5-30%.
4. E1-default pair: not bitwise (90%); abs(d val_loss) in 0.0005-0.006 (larger than E0's).
5. E1-det pair: still NOT bitwise (65%), because the fla Triton kernels are outside `use_deterministic_algorithms`.
6. `--compile` off, E0 default: the gap is not larger than with compile (the nondeterminism sits in CUDA kernels, not Inductor): 60%.

**Criteria (fixed now).**
* C183 VERIFIED iff at least one default-mode pair has unequal checkpoint sha256.
* C185 VERIFIED iff the E0-det pair has equal sha256 (and equal final val loss to all printed digits); REFUTED otherwise.
* C186 VERIFIED iff the E1-det pair has unequal sha256; REFUTED if equal.
* C184 holds iff EVERY default-mode pair has abs(d val_loss) <= 0.004. If any pair exceeds it, the 0.004 floor in
  `notes/reproducibility.md` is too tight and must be raised (new version) before any `reproduce --compare` verdict is cited.
* The same-seed spread is only a lower bound for the seed sd (a new seed also changes init and data order).

## SQ141: C91 closed (claim C187)

Replay E0's first 100 steps (`--max-tokens 3276800`) twice with current code and once with v0.3.0 code.
**Prediction:** all three step-100 train losses fall within 0.02 of each other (80%), i.e. 2.8370 vs 2.8460 is inside the
same-code spread. **Criterion:** C187 VERIFIED iff the same-code pair gap >= 0.5 x the old-vs-new gap (same order as
nondeterminism); REFUTED (the code changed behaviour) iff the same-code gap < 0.002 while old-vs-new > 0.005.
