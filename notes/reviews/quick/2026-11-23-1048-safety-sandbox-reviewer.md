# Quick review 2026-11-23-1048 - safety/sandbox reviewer (Codex GPT-6 Luna high)

- **Observation:** C106 is marked VERIFIED for this machine, with the explicit limitation that world-readable files remain accessible. C83 describes a different, weaker reward sandbox and still lists fork-bomb resistance as unverified. (experiments/claims-status.md, C106, C83)
- **Observation:** The latest devlog records adding the `$HOME` visibility probe to the eval sandbox checks; the sandbox code also says code tasks are refused unless `self_test()` passes. (DEVLOG.md, v0.4.106; evals/sandbox.py)
- **Observation:** Existing side quests already cover namespace escape probes, secret canaries, fail-closed setup, resource limits, descendant cleanup, and output bounds. (experiments/side-quests.md, SQ61, SQ211, SQ212, SQ247)

- **Idea:** **Check every code-evaluation dispatch path.** **Hypothesis:** Every route that executes generated code goes through the sandbox gate, with no in-process code-evaluation route enabled. **Cheapest test:** A CPU-only routing audit with a stubbed executor and a 13M-scale config, about 2–5 minutes. **Change the plan if:** Any route bypasses the gate; disable or reroute it before code-eval runs.

- **Question for the main agent:** Does the sandbox gate cover direct task invocations as well as `run_suite.py`?
## Triage
- Check every code-evaluation dispatch path: done at once, by reading the code (CPU, no run). Three routes execute generated code:
  1. `evals/tasks/gogeta_utils._exec_metrics`, which calls `evals.sandbox.run_tests` (namespaced). `run_suite.py` loads tasks only from `evals/tasks` with `include_defaults=False`, removes `HF_ALLOW_CODE_EVAL`, and runs `self_test()` before any code task.
  2. `post_training/rewards.run_tests`, the weaker GRPO harness. Its in-process `exec` is inside the child process, and C105 is the known gap, fixed by SQ61.
  3. `evals/upstream/lm_eval-0.4.13/mbpp/utils.py`, the snapshot archived in v0.4.106. It loads HF `code_eval`, which runs code in-process, and lm-eval would register it if pointed at `--include_path evals`. Fixed now: the snapshot files are renamed `*.snapshot` (sha256 unchanged, `SHA256SUMS` verified) and C109's evidence points at the new path.
  One real gap was found and accepted as SQ352: the self-test gate lives only in `run_suite.py`, so a direct task invocation skips it.
- Question (does the gate cover direct task invocations?): no. Direct lm-eval runs of the gogeta exec tasks still execute inside `evals/sandbox.run_tests`'s namespaces, but nothing checks that those namespaces work on the machine. SQ352 moves the check to the execution point.
- Correction to observation 2: the `$HOME` probe was not added to `self_test()` in v0.4.106. It is a one-off check script (`experiments/checks/sandbox_home_probe.py`), and adding it to `self_test()` is SQ346.
