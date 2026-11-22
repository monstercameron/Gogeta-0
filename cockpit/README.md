# Gogeta-0 cockpit

One terminal UI + one MCP server over the repo's existing scripts, with a SQLite log store. Rules: [AGENTS.md §0e](../AGENTS.md).

```
cockpit/core.py        stdlib only: one function per question (status, queue, claims, runs, search, brief, servers, guarded launch)
cockpit/logdb.py       stdlib sqlite3 log store: runs, events, errors (full tracebacks), campaigns, servers, tool_calls
cockpit/mcp_server.py  MCP stdio server (official `mcp` SDK, 22 tools + 3 resources)
cockpit/tui.py         Textual TUI (9 tabs)
cockpit/cockpit.sh|ps1 launchers (WSL / PowerShell)
```

## Install (once, WSL)
```
bash cockpit/cockpit.sh --setup      # creates ~/.venvs/gogeta-cockpit from requirements-cockpit.txt (mcp 1.30.0, textual 8.2.8)
```
Only that venv is touched; `gogeta`, `gogeta-rl` and `gogeta-eval` are never used. `core.py` and `logdb.py` need no packages.

## Run
- TUI: `bash cockpit/cockpit.sh` (WSL) or `.\cockpit\cockpit.ps1` (PowerShell). `?` shows keys. Works at 80x24 and larger.
- MCP: `.mcp.json` at the repo root registers `gogeta-cockpit` for Claude Code (`wsl -d Ubuntu-24.04 -- /root/.venvs/gogeta-cockpit/bin/python .../cockpit/mcp_server.py`).
- Tests: `bash cockpit/cockpit.sh --test` (unittest + stdio probe + headless TUI). Backfill the DB: `--ingest`.

## TUI tabs
Dashboard (GPU card with temperature band to 85 C, campaign progress and per-item val_loss, CPU slots n/8, server tiles with Start/Stop/Open, power level, unresolved-error badge, refresh every 5 s), Queue (sortable, details, next GPU/CPU, confirm-gated Launch, Dry run), Experiments (index + card + val_loss bars), Claims (claims or side quests, status filter, text filter), Devlog, Logs (live redacted tail), Errors (unresolved first, full traceback, Resolve with note), Runs (SQLite history), Search (global; `/`).

## MCP tools
Read-only: `gogeta_brief` (**call first**), `gogeta_status`, `gogeta_queue` (filter | next_kind | item), `gogeta_experiment`, `gogeta_index`, `gogeta_claims`, `gogeta_side_quests`, `gogeta_search`, `gogeta_runs`, `gogeta_results`, `gogeta_devlog`, `gogeta_notes`, `gogeta_log_tail`, `gogeta_servers`, `gogeta_errors`, `gogeta_events`, `gogeta_run_history`, `gogeta_sql` (single SELECT only).
**STATE-CHANGING:** `gogeta_start_server`, `gogeta_stop_server`, `gogeta_launch_queue_item` (`confirm: true`), `gogeta_resolve_error` (`confirm: true`, note required).
Resources: `gogeta://agents`, `gogeta://queue-md`, `gogeta://index`.

Examples: `gogeta_queue {"next_kind":"gpu"}`, `gogeta_claims {"status":"REFUTED"}`, `gogeta_errors {"since":"2h","unresolved":true}`, `gogeta_run_history {"experiment":"E10","order":"val_loss"}`, `gogeta_launch_queue_item {"id":"K8","confirm":true,"dry_run":true}`.

## Safety
- Everything binds to 127.0.0.1. Share/Space settings are never exposed. No delete operations exist.
- Log text and DB writes are redacted (`hf_...`, `token=`, `write_token`, bearer/auth headers).
- Launch goes only through `tools/launch_{experiment,campaign,cpu_job}.sh` from a queue item; refused unless `confirm` is exactly true, the item is launchable, the command has no shell syntax or skip/force/foreground flags and never `gogeta.train`; for GPU items also refused while the GPU slot is held, a campaign is alive, a GPU process exists, the GPU is >= 85 C or `gogeta/ configs/ tools/ tests/ requirements.txt` are dirty. The launcher then runs its own preflight. `queue.json` is not edited: run `queue.py set` afterwards.
- Stop only kills processes recorded in the launchers' or the cockpit's own PID files (`~/fusiondance-data/run/cockpit/`).

## SQLite store
`~/fusiondance-data/cockpit/gogeta.sqlite` (WAL; data, not committed). `logdb.ingest()` is incremental (file offsets in `ingest_state`) and runs on every TUI/MCP refresh. Direct logging from launchers and trainer: `notes/patches/cockpit-logging-*.patch` (not applied; apply between campaigns, then `git apply notes/patches/cockpit-logging-*.patch`).

## Incidents and the watchdog (AGENTS 0f)
`bash cockpit/cockpit.sh --watchdog` starts the daemon (one instance, polls every ~10 s). Incidents land in SQLite `incidents`, `~/fusiondance-data/cockpit/incidents/<id>.json` and the stream `~/fusiondance-data/cockpit/incidents.log` (`tail -F` it). Detectors: dead/hung/failed runs, NaN/OOM/Triton/traceback in training logs, aborted campaigns (writes `experiments/launch/campaigns/<name>-resume-<n>.campaign` with exactly the unfinished items; auto-launched only for transient/thermal causes), failed CPU jobs, servers down (auto-restart, max 3/h), GPU >= 85 C (pauses auto-resume until < 75 C), low disk. MCP: `gogeta_incidents`, `gogeta_incident` (bundle + agent brief), `gogeta_incident_update` (confirm), `gogeta_resume_campaign` (confirm), `gogeta_watchdog`, `gogeta_backlog`. TUI: Incidents tab (`i`) and a badge on every screen.

## Adding a feature (AGENTS 0g)
1. Write one stdlib function in `core.py` (typed args, docstring, returns a JSON-able dict). Check `BACKLOG.md` first.
2. Expose it in `mcp_server.py` (schema from the type hints, description from the docstring, calls are logged to SQLite):
```python
# core.py
def disk_free(path: str = "") -> dict:
    """Free GB on the data drive (or on `path`)."""
    import shutil
    return {"free_gb": round(shutil.disk_usage(path or core.DATA).free / 2**30, 1)}
# mcp_server.py
expose(core.disk_free, "gogeta_disk_free")            # read-only; pass readonly=False for state-changing + confirm gate
```
3. TUI (if humans need it): add a table/Static in `tui.py` and fill it with `self.bg("name", core.fn, done_callback)`.
4. Test in `test_core.py`, one line in this README, PATCH version + push. Unfixable gap now? Add a `- [ ]` line to `BACKLOG.md` with the reason.
