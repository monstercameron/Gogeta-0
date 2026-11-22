# Gogeta-0 cockpit

One terminal UI + one MCP server over the repo's existing scripts, with a SQLite log store. Rules: [AGENTS.md §0e](../AGENTS.md).

```
cockpit/core.py        stdlib only: one function per question (status, queue, claims, runs, search, brief, servers, guarded launch)
cockpit/logdb.py       stdlib sqlite3 log store: runs, events, errors (full tracebacks), campaigns, servers, tool_calls
cockpit/runform.py     stdlib only: run report forms (draft / submit / render / list) + pure-Python SVG charts
cockpit/mcp_forms.py   stdlib MCP client used to fill forms from a shell (draft / submit / list)
cockpit/mcp_server.py  MCP stdio server (official `mcp` SDK, 31 tools + 3 resources)
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
Dashboard (GPU card with temperature band to 85 C, campaign progress and per-item val_loss, CPU slots n/8, server tiles with Start/Stop/Open, power level, unresolved-error badge, refresh every 5 s), Queue (sortable, details, next GPU/CPU, confirm-gated Launch, Dry run), Experiments (index + card + val_loss bars), Claims (claims or side quests, status filter, text filter), Devlog, Logs (live redacted tail), Errors (unresolved first, full traceback, Resolve with note), Runs (SQLite history, or **Run forms**: none / draft / filled per finished run, with a Draft button), Search (global; `/`).

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

## Run forms (AGENTS 5: every finished run gets one)
One report per finished run (a folder in `~/fusiondance-data/runs/` with `summary.json`). The cockpit fills the facts and the graphs; the agent writes five narrative fields. Every field is tagged `[machine]` or `[agent]` in the output. Smoke runs are marked "smoke run: no science".

**Flow (MCP):** `gogeta_brief` (lists finished runs without a form) -> `gogeta_run_form_draft {"run_id": ...}` -> read the draft and the card, micro devlog, DEVLOG and ledger it links -> write the narrative -> `gogeta_run_form_submit {"run_id": ..., "fields": {...}}`. `gogeta_run_forms` lists every run with its form status. From a shell: `python3 cockpit/mcp_forms.py list | draft RUN | submit FILE.json` (same tools over stdio; every call is logged in `tool_calls`).

**Auto-filled (machine):** run id, experiment and title, arc, question and registered prediction, architecture (layout, params total and non-embedding, FLOPs per token, loops r), hyperparameters (lr, schedule, warmup, batch, seq len, tokens, precision, compile, seed), commit and git describe, CLI overrides, campaign and item, environment (torch/GPU/driver are **not recorded per run**; the draft says so), start, end, wall, tokens/s, status, exit, final metrics (val loss, bits/char, train loss at the end, best periodic val, tokens to within 10% of the last periodic val, max grad norm, per-loop val loss for looped runs), siblings, related ledger claims with their latest status, links, and the **K9 data caveat** (always shown; a clean-val score is shown beside it once a `*clean*` results JSON names the run). Params and FLOPs come from what `gogeta.train` recorded in `meta.json` (from `param_counts()` and `flops_per_token()`); the form never imports torch.

**Narrative (agent, all required):** `purpose`, `what_happened`, `siblings`, `verdict`, `followups` (non-empty, at least 20 chars; 8 for smoke runs), plus `claims`: a list of `{id: "C92", effect: supports|weakens|informs|neutral, evidence: <pointer>}`. `gogeta_run_form_submit` **refuses and writes nothing** if a narrative field is empty, if a `C<n>` is mentioned in the text without a `claims` entry, if a claim has no evidence pointer (a file path such as `experiments/checks/results/K7.json` or a card/devlog, or a run id plus a version tag), if the id is not in `experiments/claims.md`, or if the verdict names no claim and does not say "counts toward no claim".

**Outputs (reports, committed by the main agent; never raw run artifacts):** `experiments/run-forms/<run_id>.md`, `experiments/run-forms/<run_id>/*.svg`, `experiments/run-forms/INDEX.md`. Form state (JSON: status, narrative) lives in `~/fusiondance-data/cockpit/run-forms/` (not committed). `python3 cockpit/runform.py render --all` re-renders every filled form from that state.

**Graphs** (pure-Python SVG, <= 60 KB each, transparent background, mid-tone ink so they read in light and dark): training loss vs tokens (log-log) with val-loss eval points on the same axis; LR schedule vs step; gradient norm vs step; throughput vs step (step 0, compile warm-up, omitted); sibling comparison (val loss vs LR on a log-x axis for sweeps, otherwise labelled dots for the runs at the same token budget); val loss vs inference loops for looped runs. A chart whose data is not recorded is skipped and the form says why; nothing is invented. Series come from the Trackio SQLite (read-only), falling back to the training log.

**TUI:** Runs tab (`h`) -> **Run forms** button -> table of every finished run with NONE / DRAFT / filled; highlight a row to read its fields (machine vs agent); **Draft form** stores a draft state for the highlighted run (no repo files).
