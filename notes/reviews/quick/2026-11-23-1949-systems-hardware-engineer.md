# Quick review 2026-11-23-1949 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- The GDN recurrent kernel has a narrow hardware check on sm_121: one-token decode is reported at 0.036 ms, but that does not establish end-to-end generation speed. [C111; experiments/claims-status.md]
- The latest data gate reports gains from 100M to 197M tokens across 7 seeds, though the final-token gains are partly confounded by cosine annealing. [v0.5.54; DEVLOG.md]
- SQ419 is the only listed running GPU job; the next pick is a 60-minute CPU analysis, so there is no queued GPU follow-on yet. [experiments/QUEUE.md]

## Idea

- **[speculation] Measure energy per generated token alongside latency.** Hypothesis: loop count may change the energy cost of serving even when quality per FLOP looks favorable. **Cheapest test:** on the N1X, decode 256 tokens at batch 1 for two existing checkpoints; record supported board power and clock telemetry, then integrate joules/token (about 5 GPU-minutes). **Change the plan if:** the preferred quality arm uses over 20% more joules/token than its nearest alternative; include energy in the serving trade-off.

## Question

- Will the E61 deployment gate track joules per token as well as latency and quality?

## Triage
- Idea (energy per generated token): accepted as an extension of SQ288, the decode tok/s harness. Power and SM clock are sampled every 100 ms during batch-1 decode and integrated to joules per token, for every arm. No existing SQ covered energy (grep). It will not run on the N1X now: that machine is under an operator hold (light work only since the thermal kill at 86.6 C, DEVLOG v0.4.73), so the first measurement goes on a rented card, where nvidia-smi reports power.draw.
- Question (will E61's deployment gate track joules per token?): yes, as a reported trade-off column next to latency and quality, with the review's rule (over 20% more J/token than the nearest alternative makes it decision-relevant). It is not a pass/fail gate, because no energy budget has been set; that budget would be the human's call.
- Observation 3: right, no GPU item is pickable while C1 runs (one GPU job at a time). SQ75's profile already ran on the C1 box; SQ433 (width-matched) is the next registered GPU work after C1's readout.
