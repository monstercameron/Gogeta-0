# Quick review 2026-11-23-1959 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- C106’s saved self-test passes 12 isolation checks, including hidden Windows drives, blocked network access, memory limits, and timeout kills. [C106; `experiments/checks/results/test-runs/2026-11-23-sandbox-selftest.json`]
- The reward sandbox claim remains PARTIAL, and the side-quest tracker lists resource-budget and `/proc` visibility checks as open. [C83; SQ61 in `experiments/side-quests.md`]
- The queue keeps the sandboxed functional benchmark Q14 human-gated while Q15 is next for CPU work. [Q14, Q15 in `experiments/QUEUE.md`]

## Ideas

- **[speculation] Probe abstract Unix sockets across the sandbox boundary.** Hypothesis: a child might reach a host abstract socket even when ordinary network access is blocked. **Cheapest test:** expose a harmless parent listener and attempt a connection from one sandbox child; about 5 CPU minutes, no model run needed. **Change the plan if:** the child connects—scope the network claim and add a blocking control before execution evaluations.
- **[speculation] Check containment over repeated concurrent runs.** Hypothesis: namespace or temporary-mount cleanup may fail only under overlap. **Cheapest test:** run 20 tiny sandbox jobs concurrently and check for surviving children or leaked mounts; about 5 CPU minutes. **Change the plan if:** any survivor or leak appears—pause sandboxed evaluation until cleanup is fixed.

## Question

- Which open sandbox checks must pass before Q14’s execution evaluations are considered ready?

## Triage
- Ideas 1 and 2 (abstract Unix sockets; concurrent runs): both run now, both PASS (`experiments/checks/sandbox_socket_concurrency.py`, saved output `results/test-runs/2026-11-23-sandbox-socket-concurrency.txt`).
  - The host listens on an abstract socket and on a /tmp path socket, and can connect to both itself (positive control). Sandboxed code reaches neither: it has its own network namespace and a private /tmp.
  - 20 concurrent jobs, each starting a setsid child, all time out within 1.0 s (wall 1.1 s), with no survivors and the host mount table unchanged (32 entries).
  - Recorded as evidence on C106 and C83.
- Observation 1 is stale: the self-test has 17 checks since v0.5.48 (`setsid_descendant_killed` was added), and the 12-check JSON is the earlier artifact. The current saved output is `test-runs/2026-11-23-sandbox-selftest-fixed.txt` (17/17).
- Question (which sandbox checks must pass before Q14's execution evals?):
  - Must pass, enforced in code: the 17-check self-test gate, which `run_tests()` runs once per process and refuses to start any child if a check fails.
  - Must pass as saved probes: kill cleanup (`sandbox_kill_cleanup.py`) and these socket and concurrency probes.
  - Plus SQ61's world-writable-path sweep, because it is the remaining route by which code could write outside the private mounts.
  - Hardening, not gates for Q14: the full resource-budget test and the /proc PID view (`--mount-proc`). C106 stays "a process sandbox, not a security boundary"; Q14 runs our own model's samples, not untrusted third-party code.
