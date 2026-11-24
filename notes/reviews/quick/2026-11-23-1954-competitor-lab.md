# Quick review 2026-11-23-1954 - competitor lab (Codex GPT-6 Luna high)

## Observations

- C209 reports A2s ahead of E27 by 16.67% on clean80, but with about 10% more training compute and about 15.1M vs 9.5M unique non-embedding parameters. [C209]
- The interaction result changes with training horizon: synergy at 20M, anti-synergy at 200M. That makes short-run component rankings a risky guide to the final recipe. [C201, C202]
- SQ419 is running; no GPU item is pickable, and Q15’s CPU analysis is next. The queue also flags Q14’s benchmark work as needing human input. [experiments/QUEUE.md]

## Ideas

- **[speculation] Measure serving throughput across batch sizes.** Hypothesis: A2s’s repeated core passes may change its latency-throughput tradeoff versus E24. **Cheapest test:** on existing 13M-scale checkpoints, compare batch 1/4/16 with a fixed prompt and 128 generated tokens; record throughput and peak memory (about 10 GPU-minutes). **Change the plan if:** A2s’s advantage disappears at the intended serving batch size.
- **[speculation] Check whether model rankings hold on a second, already-local eval slice.** Hypothesis: clean80 results may depend on that split’s composition. **Cheapest test:** score A2s, E27, and E24 on the existing prose slice, using the same seeds (about 20 CPU-minutes). **Change the plan if:** A2s loses its lead; qualify C209 as slice-specific.

## Question

- Before Q14 is unblocked, what evidence supports interpreting the clean80 gain as a code-assistant advantage?

## Triage
- Idea 1 (serving throughput across batch sizes): accepted as an SQ288 extension, the decode harness. It was batch 1 only and now adds batch 4 and 16 for A2s, E24 and E27, with throughput, latency and peak memory. Energy per token was added at 19:49.
- Idea 2 (rankings on a second local slice, prose): already SQ416 (a prose slice from the corpus's docstrings and comments, for A2s, E27, E24 and E20, CPU about 20 min). It does not exist yet; SQ416 builds it. No new SQ. It goes to the worker after SQ435 and SQ265.
- Question (what supports reading the clean80 gain as a code-assistant advantage?): today, nothing beyond likelihood. C209 is a clean80 bits-per-char result, and the README calls it recipe-level. The 13M-scale proxies are queued: syntax validity (SQ349, now with the C209 pair), patch preference (SQ210) and API-symbol grounding (SQ432). The functional test is SQ4 at E61 scale, behind Q14's human-gated benchmark work. Until one of them runs, no code-assistant claim is made.
- Observation 2 (short-run rankings flip): agreed; that is why SQ419 runs at 200M (SQ414: 20M keeps sign for named contrasts but not for interactions).
