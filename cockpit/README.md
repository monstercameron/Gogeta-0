# Gogeta-0 cockpit

One terminal UI + one MCP server over the repo's existing scripts, with a SQLite log store. Rules: [AGENTS.md §0e](../AGENTS.md).

```
cockpit/core.py        stdlib only: one function per question (status, queue, claims, runs, search, brief, servers, guarded launch)
cockpit/logdb.py       stdlib sqlite3 log store: runs, events, errors (full tracebacks), campaigns, servers, tool_calls
cockpit/runform.py     stdlib only: run report forms (draft / submit / render / list) + pure-Python SVG charts
cockpit/mcp_forms.py   stdlib MCP client used to fill forms from a shell (draft / submit / list)
cockpit/mcp_server.py  MCP stdio server (official `mcp` SDK, 57 tools + 3 resources)
cockpit/cloud.py, fleet.py, providers/  cloud GPUs (Vast.ai + plugins): guardrails, lifecycle, fleet, SSH monitoring
cockpit/tui.py         Textual TUI (13 tabs)
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

## Cloud GPUs (Vast.ai and other providers)
Files: `cockpit/providers/` (plugins: `vast.py`, `ssh.py` bring-your-own box; add one file per new provider), `cockpit/cloud.py` (spend guardrails, registry, redaction), `cockpit/fleet.py` (lifecycle, fleet, SSH monitoring, watchdog guards), `cockpit/remote/bootstrap.sh` + `fetch_data.sh` (remote side: needs only ssh, root and an NVIDIA GPU).
- **CLI:** the Windows `vastai.exe` through WSL interop, so the API key stays only in the Windows vastai config. The cockpit never reads, passes or logs it; `--api-key` arguments are refused and every log/SQLite row/incident/tool call masks 64-hex strings and `--api-key` values.
- **Spending needs the human (AGENTS 0a).** A standing budget ("approval": total $, per-instance $/h, max instances, GPU list, <= 24 h) is created ONLY by the human: TUI Cloud tab > **Approve budget** (typed `APPROVE <total>`), or `bash cockpit/cockpit.sh --approve-spend --max-usd 30 --max-usd-per-hour 1.00 --max-instances 1 --gpus "RTX 3090" --hours 12 --purpose "..."` on a terminal (refused without a TTY or from an agent shell). Revoke: `--revoke-spend APR-...`. No MCP tool creates or edits approvals (tested). File: `~/fusiondance-data/cockpit/spend-approvals.json`.
- **Every rent** checks, BEFORE the provider's create call (and again under a lock): approval valid and unexpired, GPU allowed, dph <= per-hour cap, active + 1 <= max_instances, dph x max_hours <= remaining budget. It then reserves that amount; destroy settles it to the actual cost. Rents and destroys are logged with their price (events source `cloud`).
- **Only cockpit instances are managed:** registry + `gogeta-cockpit-*` label + matching live label. Other `gogeta-cockpit-*` instances are shown read-only (status, logs, pull); anything else is listed as "not managed" with its $/h.
- **Lifecycle** (MCP `gogeta_cloud_*`, TUI Fleet tab, `cockpit/fleet.py`): `up` (cheapest exact-GPU offer inside the budget; dry run first) -> `setup` (pipes `remote/bootstrap.sh <pushed sha>`, detached, `~/setup.log`) -> `data` (`remote/fetch_data.sh` rebuild + manifest check, fallback: allowlisted upload of the manifest files + meta.json, `sha256sum -c` on arrival; never a directory sync, never `~/.ssh`/`.env`/SQLite) -> `run` (`E10 --seed 0` or `configs/x.json --lr 3e-3 ...`, nohup under a wrapper that refuses if any `gogeta.train` already runs) -> `status` / `logs` -> `pull` (run folder, sha256-verified, never overwriting, plus `repro-manifest.json` with provider/GPU/driver/dph/cost, no IPs) -> `down`. Code on the box comes only from `git clone` of the public repo at a pushed commit.
- **One shot:** `gogeta_cloud_session(config_list, gpu, max_price)` = up -> setup -> data -> run each -> pull each -> down on one box; the watchdog drives it, progress in events (source `fleet`) and `gogeta_fleet_status`.
- **Fleet (just-in-time):** `gogeta_fleet_up(n, gpu, approval)` opens a session (rents nothing, one GPU SKU per session); `gogeta_fleet_dispatch(jobs)` rents one instance per ready job (fleet size = min(ready jobs, max_instances, budget)), reuses an instance when a job waits, destroys it at once when none does.
- **Watchdog `cloud` detector:** fleet instances: no job for 5 min -> destroy; setup > 15 min -> destroy, retry the job once on another machine; all cockpit instances: over max_hours, over the budget cap, approval expired/revoked, unreachable 20 min (flagged at 5) -> destroy. **AGENTS 0h** (kill boxes unless work is queued): when a box's queue empties its runs are pulled and it is destroyed at once; a manual/attached box with no running or queued job for 10 min (`GOGETA_IDLE_KILL_MIN`) is pulled, then destroyed; `cloud_session` tears down in a try/finally even on failure. Status flags idle boxes. Offer choice (`rank_offers`, `gpu="auto"`): measured $ per token where a synced benchmark exists, else $/h above the VRAM floor (E1 7.1 GB, E2 12.0 GB, else 12 GB). Remote NaN/OOM/traceback/hung/dead become incidents tagged `vast:<label>/<experiment>`.
- **Status / metadata** (`gogeta_cloud_status`, TUI Fleet tab `j`): GPU, VRAM, driver, CPU/RAM, location, reliability, network, IP:port (display only, never committed), uptime, $/h (GPU + disk; bandwidth $/GB), accrued $, cost to finish, budget bar, nvidia-smi util/memory/temperature/power/clocks, run name/config/seed/commit, step/total, loss, val/loss, tok/s, ETA, cost per run, tok/s ratio and val delta vs the matching N1X run, loss sparkline, $ per 1M tokens, tokens per cent, a temperature mood, fleet power level, lifetime spend and GPU-hours.
- **MCP:** read-only `gogeta_cloud_providers`, `gogeta_cloud_offers`/`gogeta_vast_offers`, `gogeta_vast_instances`, `gogeta_vast_cost`, `gogeta_vast_balance` (sufficient/low/unknown only), `gogeta_vast_approvals`, `gogeta_cloud_status`, `gogeta_cloud_logs`, `gogeta_fleet_status`, `gogeta_fleet_logs`. Stop spending (confirm): `gogeta_cloud_down`, `gogeta_vast_destroy`, `gogeta_fleet_down`. Control (confirm): `gogeta_cloud_setup/data/run/stop/pull/attach`. Spend (budget-gated + confirm, dry run by default): `gogeta_cloud_up`, `gogeta_cloud_session`, `gogeta_vast_rent`, `gogeta_vast_run`, `gogeta_fleet_up`, `gogeta_fleet_dispatch`.
- **TUI:** Cloud tab (`k`): offers (GPU / max $/h), approvals, account instances, Approve budget (human), Revoke, Up (dry run, then confirm), Destroy (confirm). Fleet tab (`j`): one row per GPU with the metadata above; Refresh (ssh poll), Tail, Setup, Push data, Run job, Stop job, Pull, Attach, Destroy and Fleet down (both confirm).
- **SSH:** `ssh -F /dev/null -i ~/.ssh/gogeta_vast -o IdentitiesOnly=yes -o ForwardAgent=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new`, with an allowlisted environment (nothing local is inherited). The private key never leaves this machine; instances only get the public key via the provider, and a destroyed instance takes it with it. Remove the key from the Vast account: `bash cockpit/cockpit.sh --revoke-ssh-key` (human, TTY).
- **Tests:** `cockpit/test_core.py` Cloud* classes with a fake vastai shim and a fake ssh (`cockpit/cloud_fakes.py`; remote scripts run in a temp "remote home"), plus a live read-only smoke (offers + instances) when the Windows exe exists.
