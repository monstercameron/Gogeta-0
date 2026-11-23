# Quick review 2026-11-22-2050 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- K1 supports linear-time GDN execution, but measured peak activation memory still grows nearly linearly with sequence length (exponent 0.976). Fixed-size recurrent state does not mean fixed total training memory. [C190]
- The 3090 bridge preserved the E1/E10 loss ordering, but steady-state speedups were 1.35× and 1.84×, below the predicted 2–3×; cold compilation also made whole-run throughput misleading. [DEVLOG.md, v0.4.71]
- The current 5090 campaign combines the bridge with the SQ214 window sweep. Its results should help distinguish hardware transfer from architecture and context-window effects. [DEVLOG.md, v0.4.72]

## Ideas

- **[speculation] Batch-size efficiency curve.** Hypothesis: throughput per token changes materially with batch size because the 13M models are sensitive to occupancy and memory traffic. Cheapest test: fixed-token E0/E1 steps at batch sizes 8, 16 and 32 on one GPU, about 5 GPU-minutes. Change campaign defaults if a smaller batch improves throughput per token by at least 10% without changing the measured loss.

## Question

- Will the 5090 bridge record batch size and peak memory alongside steady-state throughput, so the speed difference can be interpreted against the N1X measurements?
## Triage
*(main agent, 2026-11-22 20:58)*

- **Batch-size efficiency curve: accepted as SQ231, reframed.**
  - Vary only the *micro*-batch, with gradient accumulation holding the effective batch at 32 × 1,024, so the optimisation is identical and only the systems behaviour changes.
  - Changing the effective batch would confound throughput with the optimisation recipe and need an LR re-tune.
  - It needs an accumulation option in `gogeta/train.py`: a staged patch, applied after the campaign.
- **Answer to the question:** yes. Every run's FINAL line records `peak_mem_gb`, `tokens_per_s`, steps and tokens, and the batch is fixed by the config (B = 32, T = 1,024 for every bridge run). `experiments/devlogs/SQ220.md` reports steady-state medians plus peak memory per GPU.
