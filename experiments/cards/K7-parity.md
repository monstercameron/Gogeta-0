---
id: K7-PARITY
type: check
title: E10 construction and FusionDance-unchanged check
status: done
headline: PASS - FusionDance construction bit-identical for E0/E1/E2/E3/E9 after the arch switch; E10 params and FLOPs identical to E0
reuse: run after any change to model construction in gogeta/train.py; template for "a refactor left behaviour unchanged" checks
---

# K7-PARITY: E10 construction and FusionDance-unchanged check

## Question and prediction
- **Question:** does adding `"arch": "hf_llama"` to `gogeta/train.py` leave FusionDance unchanged, and does E10 match E0's size?
- **Prediction / criterion:** the script's `PREDICTION` and `CRITERION` constants, and `experiments/checks/PREDICTIONS.md` → "K7-parity". Written with the code change, before the result.

## How to run it
- `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/checks/k7_e10_parity.py` (CPU, ~5 s)

## Runs
| Run id | Seed | Commit | Status | Key metrics |
|---|---|---|---|---|
| `results/K7-parity.json` | 0 | v0.3.5 + uncommitted train.py change | done | 5/5 bit-identical; E10 13,767,552 / 10,621,824 params; 96,758,784 FLOPs/token = E0 |

## Results
- **(a) Old vs new construction** for E0, E1, E2, E3, E9: identical state_dict (all tensors), model-config dict, global RNG state after construction, and FLOPs/token.
- **(b) E10 vs E0:**
  - total params 13,767,552 = 13,767,552;
  - non-embedding 10,621,824 = 10,621,824;
  - train FLOPs/token 96,758,784 = 96,758,784.
- **(c) E10 at init:** tied head; init std 0.020 everywhere (E0's residual outputs: 0.0058); loss 9.094 vs ln 8192 = 9.011.
- **GPU replay** (bench, 101 steps, compiled): the new `train.py` reproduces the v0.3.0 E0 log at step 0 exactly (loss 9.0900, grad-norm 5.72).
  - At step 100 it gave 2.8370 vs the log's 2.8460.
  - A determinism baseline (HEAD code, same command) is still pending, so this difference isn't attributed yet.

## Artifacts (where things live)
`experiments/checks/results/K7-parity.json`

## How to reuse this
Re-run before any version that touches model construction in `gogeta/train.py`.

## Caveats
This checks construction only. The training loop is unchanged by inspection (diff), not by a bit-exact GPU replay.

## Links
Claims C90, C91 · E10 card · `notes/science-gaps.md`
