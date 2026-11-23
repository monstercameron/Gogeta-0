# Quick review 2026-11-23-1754 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- The claims summary lists 142 claims, including 75 `UNVERIFIED`; several scope or refutation fields are still `TBD`. That leaves substantial room for claims whose boundaries are not yet reproducible from their definitions. [experiments/claims-status.md]
- C3 is marked verified, but its listed limitations say the “exactly at the window” result has no saved test output and its asserted tolerances concern past-token differences. The evidence may not fully support the exact cutoff wording. [experiments/claims-status.md; C3]
- SQ419 is named as the next GPU pick, while the latest quick review says its config and readout hash registration are still missing and it is not ready. The comparison is not yet fully reproducible from a registered run specification. [DEVLOG.md, v0.5.42; notes/reviews/quick/2026-11-23-1749-competitor-lab.md]

## Idea

- **Persist a causal-mask witness for C3.** **Hypothesis:** the current test evidence may not distinguish a strict sliding-window cutoff from a merely small effect on older tokens. **Cheapest test:** on CPU, perturb future and out-of-window tokens in a seeded ~13M-parameter model and save the measured maximum logit differences plus the expected pass thresholds; about 5–10 minutes. **Change the plan if:** any forbidden perturbation exceeds tolerance; narrow C3’s wording and add the affected block type to the follow-up checks.

## Question

- Will C3’s “exactly at the window” wording stay in use until the missing saved output verifies that cutoff?
## Triage
- Persist a causal-mask witness for C3: done now (`experiments/checks/c3_window_witness.py` -> `results/c3_window_witness.json`, CPU; GDN through the torch reference recurrence, since `tests/test_model.py` needs CUDA). Every block type shows max |diff| exactly 0.0 before the perturbed token (attn, swa, gdn, looped hybrid). In a window-8 SWA layer, token 0 moves positions < 8 by up to 1.37 and positions >= 8 by exactly 0.0. Recorded as C3 evidence, which closes its "no saved test output" limitation for the cutoff wording.
- Observation 3 (SQ419 is not yet reproducible from a registered spec): correct and intended. It is queued as NOT READY until its F-6 config, its prediction and its readout script's sha256-16 are committed (v0.5.39 rule), and it waits for the human's go-ahead on GPU spend.
- Question (will C3's "exactly at the window" wording stay in use until a saved output verifies it?): it is now verified by a saved output with exact zeros, so the wording stands for the torch path. The fla GPU path's last check (12/12 on the B1 box) was recorded only as a pass count; SQ274's test re-run extension covers it at the next GPU batch.
