# Quick review 2026-11-23-1804 - skeptical peer reviewer (Codex GPT-6 Luna high)

## Observations

- **C3’s exact-zero witness is encouraging, but its stated scope needs care:** the new check uses a CPU torch reference recurrence. That supports the reference path; it does not by itself establish identical behavior in the production CUDA kernels. [DEVLOG.md, v0.5.43; C3]
- **C106’s evidence is explicitly bounded:** 16/16 sandbox checks and 6/6 gate tests support the tested process-sandbox behaviors, while the entry says this is not a security boundary. Keep claims and product wording within that scope. [DEVLOG.md, v0.5.43; C106]
- **The GPU queue has a clear executable next pick:** SQ75 is listed as NEXT GPU; SQ419 is human-gated and not ready pending its config, prediction, and readout hash. The headline about SQ419 being “ahead” should be read as priority once those gates clear. [experiments/QUEUE.md; DEVLOG.md, v0.5.43]

## Ideas

- **[speculation] Check C3’s perturbation witness on the production CUDA paths.** **Hypothesis:** a future-token perturbation could expose a causal leak or nonzero SWA cutoff in a fused kernel that the reference recurrence cannot catch. **Cheapest test:** run one seeded ~13M-parameter model on GPU, perturb tokens beyond the tested boundary, and compare earlier logits for each production block type; about 5 GPU minutes. **Change the plan if:** any pre-boundary logits move beyond the registered tolerance; narrow C3’s wording and add a kernel-specific check before relying on that claim.

## Question

- Should C3’s general wording remain explicitly limited to the reference path until a production-kernel perturbation check passes?
## Triage
- C3 witness on the production CUDA path: accepted. `experiments/checks/c3_window_witness.py` gains a `--gpu` mode that runs the production fla kernels unpatched and writes `results/c3_window_witness_gpu.json`; it runs at the next GPU batch. The CPU result was re-checked after the edit and is unchanged.
- Question (should C3's wording stay limited to the reference path until a production-kernel check passes?): yes, for the "exactly 0" part. C3 now carries a path-scope limitation: the exact-zero witness is the torch reference path, while the CUDA path's last check (tests/test_model.py 12/12 on the B1 box, v0.5.31) used a 1e-4 past-token tolerance and saved no output.
- Observation 3: agreed. SQ419's place ahead of SQ75 is priority once its gates clear (config, registration with readout hash, the human's go-ahead), not executability.
