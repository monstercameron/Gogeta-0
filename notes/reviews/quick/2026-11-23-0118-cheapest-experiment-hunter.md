# Quick review 2026-11-23-0118 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- **71 of 127 claims are UNVERIFIED**, so a cheap, falsifiable check may have more value than another broad claim (`experiments/claims-status.md`).
- **Q7 is running; Q8 is ready** for three 200M-token seeds. The queue estimates 117 GPU minutes for Q8, with planning estimates based on compiled throughputs (`experiments/QUEUE.md`).
- The latest devlog entries record new controls for init and LR confounds, including SQ307 and SQ308; those checks target specific threats to interpretation (`DEVLOG.md`, v0.4.89–v0.4.90).

## Ideas

- **Q8 launch preflight [speculation]** — Hypothesis: a short run using the exact Q8 config can catch setup, non-finite loss, or gradient problems before the full campaign. Cheapest test: 50 steps on one 13M-parameter seed, about 1 GPU-minute. If loss or gradients are non-finite, or the observed LR/token count disagrees with the config, pause Q8 for a fix; otherwise proceed.

## Question

- Does the Q8 launcher already perform an equivalent short preflight, or would this be a new gate?
## Triage
- Q8 launch preflight: rejected. Q7/Q8 were stale queue entries: both ran on the N1X in Phase 2b and are superseded by the 7-seed 5090 arms (v0.4.79). They are now marked done in `experiments/queue.json` (v0.4.91). The local launcher also runs its own preflight (`cockpit/core.py`, `--no-preflight` is a forbidden flag), and fleet boxes run a compile smoke test in `bootstrap.sh`.
- Question: yes, the launcher has a preflight; see above.
- Follow-up: `experiments/QUEUE.md` still lags the fleet work (it lists E9 as "next"); a full reconcile is pending.
