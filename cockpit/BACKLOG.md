# Cockpit backlog

Known gaps and limits, and requests agents hit but did not fix (with the reason). Read before adding features (AGENTS.md section 0g). Open items start with `- [ ]`; `brief()` shows the top ones.

- [ ] Launcher logging is ingest-only until `notes/patches/cockpit-logging-*.patch` are applied (needs a gap between campaigns; the launchers live under `tools/`).
- [ ] Training resume is impossible today (train.py saves only a final checkpoint): needs `notes/patches/cockpit-resilience-train.patch`. Watchdog resume works at campaign-item granularity only.
- [ ] `brief()` does not yet show K8 verdicts (needs a result-verdict parser for `experiments/checks/results/*.json`); needs_human queue items are listed by id only.
- [ ] Queue launch does not write `queue.json` status (`queue.py set ... running`); the agent does that by hand after a launch. Candidate: a confirm-gated `gogeta_queue_set`.
- [ ] No Windows-native core: everything runs in WSL (the PowerShell script only wraps `wsl`). Fine for now.
- [ ] `hung` detection uses a fixed default (15 min); derive it per experiment from the expected step time once step timings are in SQLite.
- [ ] Tracker (7861) is only restarted by the watchdog if it was seen up while the watchdog ran.
- [ ] Windows ARM64: not tested natively (wheels not needed; WSL path used).
- [ ] Run forms: torch, GPU and driver are not recorded per run (`meta.json` has no environment block and Trackio `system_metrics` is empty). Needs the trainer to log them (a `gogeta/` change, so between campaigns).
- [ ] Run forms: the clean-val re-score is only picked up from a results JSON whose file name contains `clean` and that has a key containing the run id; wire it to the real clean-val output format once A1/A2 define it.
- [ ] Run forms: `gogeta_run_form_draft` recomputes the charts on every call (about 1 s per run); cache by (run id, series length) if the TUI feels slow.
- [ ] Run forms: sibling comparison only covers finished runs with the same experiment id and token budget (plus same-budget runs of other experiments as context); no matching on config hash or campaign yet.
- [ ] `tui_check.py` was flaky once (FAILED: 2, ~18:53): every wait was a fixed 15 s poll while the log tail / run-forms drafts compute under GPU-host load. Waits are now 40 s and the Runs-tab step waits for the history table first; if it recurs, make the checks event-driven instead of polling.
