# Quick review 2026-11-22-2010 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- **No remote boundary has been exercised yet:** the Vast CLI authenticated and found offers, but no instance was rented; the cloud pilot is parked pending human approval. [DEVLOG.md, v0.4.69; SQ220]
- **The proposed upload scope needs tighter controls:** the rental note says the corpus and repo are fine to upload, while also flagging corpus license terms for verification. It doesn’t describe an allowlist, secret scan, or remote deletion check. [notes/compute-rental.md]
- **Cockpit gates don’t yet cover cloud spending or access:** its documented confirmation gate applies to local queue launchers, and the devlog says a remote-runner extension is still needed. [cockpit/README.md; DEVLOG.md, v0.4.68]

## Ideas

- **Set a hard spend ceiling before enabling rentals.** Hypothesis: an instance-level hourly cap alone may not bound total spend if several instances or long jobs are started. Cheapest test: 10 minutes of doc and CLI review to define and verify per-instance, total, and timeout limits. Change the plan if the CLI cannot enforce a cap; require a cockpit-side blocking check before any rental action.
- **Test the upload path for secret exposure.** Hypothesis: staging a broad repo or data directory could include credentials or local config. Cheapest test: 5 CPU minutes using a synthetic canary in a disposable fixture and the exact staging/archive path. If the canary appears in the staged output, switch to an explicit allowlist and block staging until it passes.
- **Define the unencrypted SSH key lifecycle.** Hypothesis: a passphrase-free key increases the impact of local disclosure unless its file access, use, and revocation are constrained. Cheapest test: 10 minutes of doc/config review covering local ACLs, key scope, instance teardown, and revocation. If the key must be copied off the local machine or remains authorized after teardown, redesign the flow before enabling rentals. [speculation]

## Question for the main agent

Will the cockpit keep the private SSH key local and enforce a total spend ceiling before it can create any instance?
## Triage (main agent, 20:19)
- **Idea 1 (hard total spend ceiling): already in the design; made explicit.** Human-created approvals carry max_usd_total, max_usd_per_hour and max_instances, checked **before** any `create instance` (no MCP path creates approvals), plus watchdog auto-destroy at the cap, at max_hours and at 5 min idle. The cockpit subagent was asked to confirm there's no rental code path that skips the check. The human can also set an account-level limit in the Vast console.
- **Idea 2 (upload-path secret exposure): accepted.** Code reaches the remote only by `git clone` of the pushed public commit; data by sha256 allowlist; job env by allowlist. A canary test is added to the cockpit work (SQ223).
- **Idea 3 (SSH key lifecycle): accepted.**
  - The private key never leaves this machine (Windows `~/.ssh/gogeta_vast`, ACL user/SYSTEM/Administrators only, checked; WSL copy mode 600).
  - `IdentitiesOnly=yes`, no agent forwarding.
  - A human-run `cockpit.sh --revoke-ssh-key` deletes Vast account key 2014984.
  - Instances are destroyed after their jobs, so no authorised key persists remotely.
- **Question: yes to both.** The key stays local, and the total ceiling is enforced before any instance can be created.
