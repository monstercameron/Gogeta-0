---
id: SQ6
type: side-quest
title: Single-pass state tracking via rotation (RotSSM) and negative eigenvalues (SQ2), vs loops
status: done
headline: parity pos 32-63 (2 seeds): GDN-negeig 100.0%, looped GDN 97.9%, RotSSM 84.0%, RotSSM no-rot 50.2%, GDN 1-pass 61.7%; C22 REFUTED at our scale, rotation-beats-looping NOT supported, SQ2 PROMISING
reuse: GatedDeltaNetNegEig (class swap, fla allow_neg_eigval) solves parity in one pass and length-generalises to 128 -> prime candidate for the core mixer; RotSSM mixer + scan-equivalence check are reusable
---

# SQ6 (+ SQ2): single-pass state tracking via rotation, vs loops

## Question and prediction
- **Question:** can a rotating complex state (Mamba-3-style) or a GDN with negative eigenvalues solve running parity in one pass, and does that beat looping?
- **Prediction (committed before running):** `experiments/checks/PREDICTIONS.md`, section "SQ6 (+ SQ2)", commit `4a3e361`. (c) ≥ 95% and holds on 64–127; (d) ≤ 60%; (a) < 75%; (e) ≥ 90%; (b) below (c).
- **Pass criteria:** C22 (local) is VERIFIED if mean (c) ≥ 90% and (c) ≥ (d) + 25, otherwise REFUTED at our scale. "Rotation beats looping" is supported if (c) ≥ (b) + 10. SQ2 is promising if (e) ≥ (a) + 20.

## How to run it
- `wsl -d Ubuntu-24.04 -- bash -c "cd /mnt/c/Users/mreca/Desktop/Gogeta-0 && ~/.venvs/gogeta/bin/python experiments/checks/sq_06_rotation_parity.py"` (env `SQ6_STEPS` overrides the steps)
- GPU, about 11 min (661 s), 10 runs. GPU temperature was 55 °C before and 58 °C after. Script uncommitted at run time, HEAD `ea3aa5b` (dirty).

## Runs
Task and training are identical to K5: parity over 64 bits, 3000 steps, batch 128, lr 1e-3, warm-up 100, cosine, d=128, 2 heads, MLP 256, one block. Seeds 0 and 1. Eval uses 20 held-out batches of 128 (generator 10000+seed) on positions 32–63 at T=64, and on positions 64–127 at T=128 (length generalisation).

| model | s0 32–63 | s1 32–63 | **mean 32–63** | s0 64–127 | s1 64–127 | **mean 64–127** |
|---|---|---|---|---|---|---|
| (a) GDN 1 pass | 73.31 | 50.08 | **61.69** | 50.40 | 50.10 | **50.25** |
| (b) GDN looped ×4 | 97.15 | 98.74 | **97.94** | 57.49 | 64.99 | **61.24** |
| (c) RotSSM 1 pass | 83.73 | 84.27 | **84.00** | 76.62 | 76.33 | **76.48** |
| (d) RotSSM, θ ≡ 0 | 49.22 | 51.17 | **50.19** | 50.05 | 50.09 | **50.07** |
| (e) GDN neg. eigval 1 pass | 100.00 | 100.00 | **100.00** | 99.99 | 100.00 | **99.99** |

## Results
- **C22 (local):** `REFUTED` at our scale. (c) 84.0% < 90%, although (c) − (d) = +33.8 points.
- **Rotation beats looping:** NOT SUPPORTED. (c) 84.0% vs (b) 97.9%. On length generalisation, however, (c) 76.5% beats (b) 61.2%.
- **SQ2:** PROMISING. (e) 100.0% vs (a) 61.7% (+38.3 points). It is also perfect at positions 64–127, which it never saw in training.

## Artifacts (where things live)
- Result JSON: `experiments/checks/results/sq_06.json` (per-seed accuracies, including positions 32–63 at T=128, loss trajectories, sanity checks)

## How to reuse this
- **`GatedDeltaNetNegEig`**: a subclass of `gogeta.model.GatedDeltaNet` whose forward passes raw `b(x)` logits to `chunk_gated_delta_rule(..., use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)`, so β = 2·sigmoid ∈ (0, 2). It is installed by `block.mixer.__class__ = GatedDeltaNetNegEig`, which keeps the params and init. Sanity check: it matches passing 2·sigmoid explicitly to within 7.6e-5. This is the cheapest candidate for a looped or 1-pass core with state tracking. Next step: an LM-loss A/B on code-py-v1.
- **`RotSSM`**: a plain-PyTorch complex-diagonal SSM mixer with a sequential `scan()` reference and a masked `parallel()` form, checked equal (max |diff| 6.3e-6 with rotation, 2.1e-6 without).
- The 5-arm harness is the template for further state-tracking A/Bs (e.g. negeig GDN looped, or harder groups such as S3/S5 word problems).

## Caveats
- Deviation: RotSSM was trained in the masked parallel form, which is the same function as the sequential complex scan. The scan measured about 0.37 s/step (~75 min for the 4 RotSSM runs), over budget. W_a and W_θ are affine (with bias); the decay bias starts so that softplus ∈ [1e-3, 1e-1]. λ is per head and per state dim, broadcast over dv.
- **(a)/(b) vs K5:** (b) reproduces K5 (ii) (97.2% vs 97.0% on seed 0). (a) does not: 73.3% on seed 0 vs K5 (i) 81.5%, with identical config, init (identical step-0 loss) and data. The cause is GPU nondeterminism in training, amplified by an unstable task for 1-pass GDN. Seed 1 stayed at chance (50.1%). The 1-pass GDN number has high variance.
- RotSSM is untuned (one LR, N=32, tanh-bounded θ that can reach ±π only asymptotically). 84% may understate what a tuned rotation SSM can do.
- For about the first minute, another agent's GPU job (`sq_kda_kernel.py`) ran at the same time. It affects timings only.
- 2 seeds; tiny models; parity only.

## Links
Claims C22, C18 (`experiments/claims.md`) · side quests SQ6, SQ2 (`experiments/side-quests.md`) · K5 (`experiments/cards/K5.md`) · predictions `experiments/checks/PREDICTIONS.md` (SQ6 (+ SQ2))
