# What is paired in our runs (read from `gogeta/train.py`, checked on CPU 2026-11-22)

One integer, `train.seed` (CLI `--seed`), drives everything random in a run. Reading `gogeta/train.py`:

| Source of randomness | Seeded by | Depends on architecture? | Consequence |
|---|---|---|---|
| **Data order**: `TokenData.g = torch.Generator().manual_seed(seed)`, consumed only by `batch()` (`torch.randint(len(train) - T - 1, (B,))`) | `seed` | **No** (depends only on seed, batch size, seq_len, dataset) | Seed s sees the **identical sequence of 6,104 batches** in E10, E1, E11, E9, E2, E0. Data order is paired across every architecture. |
| **Init**: `torch.manual_seed(seed)` before `build_model` (global RNG; `nn.init.normal_` in `FusionDance`, HF's own init in `HFLlama`) | `seed` | **Yes** (consumption order = parameter order) | Different architectures draw different tensors from the same seed. **Not paired** across architectures with different parameter lists (E10 vs E1, E9 vs E1). |
| Loop-count sampling (looped models only): `random.Random(seed + 1)` | `seed + 1` | E2 only | Separate Python RNG; does not touch data or init. |
| Validation set | none (fixed, deterministic, non-overlapping first N / all sequences) | No | Same val tokens for every run: the val loss difference has no val-sampling component between runs. |
| Dropout, other stochastic layers | none exist (grep: none in `model.py`) | - | - |
| GPU non-determinism (bf16, fused AdamW, atomics, `torch.compile`) | not controlled | - | Even the same config + seed is not bitwise reproducible; this is a noise floor inside every "paired" difference. |

## Verified by instantiation (CPU, `torch.manual_seed(s)` then `build_model`)
- E1 seed 0 vs **E11 seed 0: every tensor identical** (same parameter list; `gdn_neg_eigval` only changes the kernel call). E11 vs E1 is therefore paired on **init and data order**; the only differences are the neg-eig flag and GPU noise.
- E1 seed 0 vs E1 seed 1: different (as expected).
- E10 vs E1: different state_dict keys, so different init streams. E0 vs E10: different keys/implementation, different init.

## So, per comparison
| Comparison | Paired on | Not paired on | Analysis |
|---|---|---|---|
| **E10 seed s vs E1 seed s** | data order (all 6,104 batches), val set | init | Paired-seed t (df = n-1) is justified by design, but pairing only helps if the data-order variance is a real share of seed variance; unknown. Welch shown as sensitivity. |
| **E11 seed s vs E1 seed s** | init + data order + val set | nothing but the flag | Strongest pairing we have. Paired t / paired TOST. |
| E9 vs E1 / E2 vs E9 | data order (seed 0) | init, depth | Single seed: descriptive. |
| **E10 s0 vs v0.3.0 E1 s0** (old run, lr 3e-3) | data order | init; LR; code version | Not a paired comparison (E1's tuned LR is now 6e-3). |
| E0 s0 vs E10 s0 | data order | init, implementation | Descriptive (prediction interval from E10's seeds). |

## What a paired difference does and does not buy
`Var(d) = Var(a) + Var(b) - 2 Cov(a, b)`. The covariance is the shared-seed part: data order for E10/E1, data order and init for E1/E11. If it is ~0, pairing costs df (df 2 instead of Welch's ~4 at n=3) and gains nothing. `core.variance_components` on the seed x model matrix estimates this (`icc_rows`); `replication_stats.py` reports it for E10/E1 (data order only) and for E10/E1/E11. With 2 df it is only indicative.

## What cannot be separated with the current code
`train.seed` sets init and data order together, so "init variance" vs "data-order variance" cannot be separated from existing runs. A crossed init x data-order design needs a `--data-seed` option in `gogeta/train.py` (proposed as SQ110; deliberately not made now because the GPU campaign requires `gogeta/` to stay clean):

```python
# gogeta/train.py (proposed; absent today)
ap.add_argument("--data-seed", type=int, help="override the data-order seed (default: train.seed)")
...
data = TokenData(tc["dataset"], tc["seq_len"], args.data_seed if args.data_seed is not None else tc["seed"])
# and record cli_overrides["data_seed"] in the run config
```
