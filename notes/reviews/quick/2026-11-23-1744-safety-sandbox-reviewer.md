# Quick review 2026-11-23-1744 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- C106 is marked VERIFIED for tested process-sandbox behavior on this machine, with limits recorded in the claim. Keep that scope: the sandbox source says it is not a security boundary for adversarial code. [experiments/claims-status.md; C106; evals/sandbox.py]
- The current sandbox source defines 16 self-test checks, while the saved 2026-11-23 result contains 12. That artifact may predate the current checks, so it does not evidence all current probes. [evals/sandbox.py; experiments/checks/results/test-runs/2026-11-23-sandbox-selftest.json]
- The queue’s next picks are a GPU profiler and an MQAR redesign; neither addresses the sandbox evidence gap. [experiments/QUEUE.md]

## Ideas

- **Reconcile the saved self-test evidence.** Hypothesis: the 12-check artifact is stale relative to the current 16-check source. **Cheapest test:** run the self-test in WSL and compare its check names and results with the saved JSON; about 1 CPU minute. **Change the plan if:** any current check fails or the artifact differs; refresh the evidence and C106’s supporting record before relying on it as current.
- **Exercise the execution-point fail-closed gate.** Hypothesis: `run_tests()` refuses to start a child when its self-test fails, including when called directly; about 2 CPU minutes for a focused harness check at 13M scale. **Cheapest test:** force a failed self-test and assert the direct call raises before a harmless child-side sentinel can run. **Change the plan if:** the child starts or the call returns a score; hold code-execution evaluations until fixed.

## Question

- Should C106 remain scoped to the tested process-sandbox probes until the saved evidence is reconciled and direct-call gate behavior is checked?
## Triage
- Reconcile the saved self-test evidence: done now. The saved artifact had 12 checks and the current source has 16. A re-run at HEAD passes all 16 (the four newer ones: home_listdir_raises, home_venvs_invisible, negative_control_fails, positive_control_proc_visible). Saved as `experiments/checks/results/test-runs/2026-11-23-sandbox-selftest-1745.txt` and recorded as C106 evidence.
- Exercise the fail-closed gate: already a test, re-run now. `tests.test_sandbox_gate` passes 6/6, including `test_failing_self_test_blocks_every_child` (a forced self-test failure blocks every child start) and the two entry-path gate tests (`test_exec_metrics_path_is_gated`, `test_lm_eval_taskmanager_path_is_gated`). Not added.
- Question (should C106 stay scoped to the tested probes?): yes, and it is. C106 says "process sandbox, not a security boundary", with world-readable files and /proc host metadata as recorded limitations (SQ61's PID-namespace and fixture-audit extensions). The evidence is now current: 16/16 checks plus 6/6 gate tests at HEAD.
