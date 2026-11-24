# Quick review 2026-11-23-1934 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- The claims ledger has 142 claims, including 75 UNVERIFIED; the status labels make the evidence gap visible, but the volume leaves many claims without local checks. [experiments/claims-status.md]
- C1’s streaming evidence is PARTIAL because the saved N1X measurements are notebook transcriptions, while the machine-readable rerun used an RTX 5090. This is a useful example of hardware and artifact provenance limiting reproducibility. [experiments/claims-status.md, C1]
- SQ419 is running with a frozen readout, and its registration and LR amendment were recorded before results existed. That sequence supports auditing predictions against outcomes without hindsight. [experiments/QUEUE.md; DEVLOG.md, v0.5.50–v0.5.51]

## Ideas

- **Replay the frozen SQ419 readout from saved rows.** Hypothesis: the result can be regenerated from the recorded per-seed inputs without manual edits or hidden state. **Cheapest test:** CPU, about 5 minutes at 13M-parameter scale; run the readout twice against the same saved rows and compare normalized output hashes. **Change the plan if:** outputs differ or depend on unrecorded inputs; block interpretation until the readout is reproducible.
- **Check run-artifact provenance completeness.** Hypothesis: an otherwise valid score may be difficult to reproduce if its run summary omits the exact code revision, resolved config, seed, and data identity. **Cheapest test:** CPU, about 10 minutes at 13M-parameter scale; audit one existing comparison’s run summaries against its registered config and score artifact. **Change the plan if:** any field needed to reconstruct the run is missing; require that field in future summaries before claim status changes.

## Question

- Should a claim’s evidence be considered reproducible only when its saved score can be regenerated from the recorded run artifacts?

## Triage
- Idea 1 (replay the frozen SQ419 readout from saved rows): accepted as part of C1's readout procedure. `sq419_f6.py --gather` writes every input row, including each run's doclevel aggregates, into `results/sq419_runs.json`, which is committed. `main()` reads only that file, and it is run twice with the two outputs' sha256 compared before anything is recorded. The readout is deterministic (no sampling), so a mismatch would mean hidden state.
- Idea 2 (run-artifact provenance): checked on an A2s run's `summary.json`. It records the commit (with dirty and code_dirty flags), the fully resolved config and the seed, but the dataset only by name: no data hash, no library versions, no GPU name. Accepted as an SQ418 extension. Fleet runs' data identity currently rests on the dispatch-time manifest check (`V2 OK` in the driver logs), which is not stored with the run.
- Question (reproducible only if the score can be regenerated from recorded artifacts?): yes for the analysis step. Every readout must regenerate its numbers from committed inputs (the `*_runs.json` files), and that is the check above. For the training step, the standard is AGENTS section 5: a config plus a seed plus the commit. Bit-exact retraining is not promised (GPU nondeterminism), only reproduction within the reported seed spread.
