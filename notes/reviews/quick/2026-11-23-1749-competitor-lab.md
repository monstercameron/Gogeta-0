# Quick review 2026-11-23-1749 - competitor lab (Codex GPT-6 Luna high)

## Observations

- The A2 headline still lacks a parameter-matched comparison: A2s has about 15.1M unique parameters versus E27’s 9.5M. The planned F-6 control is well targeted, but its 20M-token result should be treated as a screen. [DEVLOG.md, v0.5.42]
- The out-of-corpus result is limited to one SQ99 set that also differs in library style. That supports a result on this set, but not yet broad generalization. [DEVLOG.md, v0.5.41; C214]
- The queue’s generated “NEXT GPU” item is SQ75, while the latest devlog names SQ419 as the next GPU pick. That mismatch could dispatch work against an outdated priority. [experiments/QUEUE.md; DEVLOG.md, v0.5.42]

## Ideas

- **Cluster-aware uncertainty for C214.** **Hypothesis:** treating documents from the same repository as independent may overstate confidence. **Cheapest test:** CPU bootstrap the existing per-document scores by repository cluster, about 5–10 minutes. **Change the plan if:** the clustered interval includes zero; report the result as inconclusive on this set until more independent sources are scored.
- **Check whether the loss gain transfers to generated code.** **Hypothesis:** A2s’s lower clean80 loss may not improve executable completions. **Cheapest test:** paired sandboxed pass@1 on a small held-out set using existing A2s and E27 checkpoints, about 30–60 GPU minutes at this scale. **Change the plan if:** A2s does not improve pass@1; keep its claim scoped to likelihood until a capability gain is shown.

## Question

- Which outcome is the final-model gate: quality per compute, or measured executable-code performance at the intended inference budget?
## Triage
- Observation 3 (the queue's "NEXT GPU" is SQ75 while the devlog names SQ419): confirmed and fixed. SQ419 is now a queue item placed ahead of SQ75. It is marked HUMAN because GPU spend waits for the human's go-ahead (asked 17:45), and NOT READY until its F-6 config and its registration with the readout hash exist. "NEXT GPU" still shows SQ75 only because it is the first item that needs neither.
- Cluster-aware uncertainty for C214: already accepted at 17:14 as an SQ213 extension (two-way seed x package bootstrap from SQ400's per-file scores; C214 drops to PARTIAL if the combined interval crosses 0). Not added again.
- Pass@1 on A2s vs E27 at this scale: rejected again (same reason as 13:24, 16:39 and 17:39). Pass@1 sits at the ~2% floor at 13-35M params, so a small held-out set cannot show a capability gain or its absence. It stays the final-model gate (Q14).
- Question (is the final-model gate quality per compute, or executable-code performance at the intended inference budget?): both, in sequence. The architecture is chosen on quality per training FLOP and per weight byte, on clean80 plus the out-of-corpus set (v0.5.38). The final model (E61) is then gated on goal 6's deployment numbers: functional pass@1 with the forgery-proof harness (Q14) and measured batch-1 decode speed and memory on a 24 GB card (E8's frontier).
