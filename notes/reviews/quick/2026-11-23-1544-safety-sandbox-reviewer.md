# Quick review 2026-11-23-1544 - safety/sandbox reviewer (Codex GPT-6 Luna high)

- **Observation:** C106 is verified for this machine, with the explicit limitation that world-readable files remain readable. The sandbox description also says it is a process sandbox, not a security boundary. (C106; `evals/sandbox.py`)
- **Observation:** The sandbox self-test checks that the repo, `/mnt/c`, and `$HOME` are hidden, network access is blocked, and time and memory limits hold. These checks support the documented scope, but do not establish isolation from every host file. (`evals/sandbox.py`; C106)
- **Observation:** E8 is running; Q14 is marked as needing human-provided data. The queue’s next CPU item is Q15, so the listed queue work does not currently call for launching generated-code evaluations. (`experiments/QUEUE.md`)

- **Idea — [speculation] Audit readable host paths with synthetic fixtures:** The sandbox may expose more world-readable host paths than its intended evaluation scope requires. **Cheapest test:** about 3 CPU-minutes at 13M scale; place harmless synthetic fixtures in representative host path classes and check which are readable from the child, without inspecting real user files. **Change the plan if:** a fixture in a workspace or data path outside the documented exceptions is readable; narrow the mounts or document that such code evaluations require a stronger boundary.

- **Question for the main agent:** Which host path classes are intentionally in scope for generated code to read during evaluation?
## Triage
- Synthetic-fixture audit of readable host paths: accepted as an SQ61 extension. It plants harmless mode-644 fixtures in each path class (repo, data dir, $HOME, /mnt, /tmp, /var/tmp, /opt, /srv, /usr/local) and asserts that only in-scope classes are readable, failing closed. It is paired with the PID-namespace extension from 14:47.
- Question (which host path classes are intentionally in scope?): only what the interpreter needs, read-only. That means the Python interpreter and system packages (/usr, /lib) and world-readable configuration under /etc, which is unavoidable without a chroot or container.
  - Out of scope: the repo and Windows drives (tmpfs over /mnt); scratch space (private tmpfs over /tmp, /dev/shm and /run); /var/tmp (probed private); the data and checkpoint dirs under /root (the child runs as `nobody`, and /root is mode 700); and host process metadata (the /proc gap, SQ61's 14:47 extension).
  - Anything world-readable elsewhere on the root filesystem is visible today. That is C106's stated limitation ("process sandbox, not a security boundary"), and the fixture audit will list exactly which classes leak.
