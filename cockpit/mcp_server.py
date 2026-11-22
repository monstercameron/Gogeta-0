#!/usr/bin/env python3
"""Gogeta-0 cockpit MCP server (stdio). Thin layer over cockpit/core.py + cockpit/logdb.py.

Run (WSL):  ~/.venvs/gogeta-cockpit/bin/python cockpit/mcp_server.py
Every tool call is recorded in the SQLite store (tool_calls) with redacted args, duration and ok/error;
uncaught exceptions also go to `errors` with the full traceback.
Tools that CHANGE STATE are marked "STATE-CHANGING" in their description:
  gogeta_start_server, gogeta_stop_server, gogeta_launch_queue_item (confirm=true), gogeta_resolve_error, gogeta_incident_update, gogeta_resume_campaign (confirm=true), gogeta_watchdog (start/stop), gogeta_run_form_submit (writes experiments/run-forms/ only).
There are no delete operations.
"""
from __future__ import annotations

import functools
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import logdb  # noqa: E402
import runform  # noqa: E402
import watchdog  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

mcp = FastMCP("gogeta-cockpit", instructions=(
    "Gogeta-0 research cockpit. Call gogeta_brief FIRST for a one-call project overview. Read-only tools are safe; "
    "tools marked STATE-CHANGING need care: launches require confirm=true and are refused while the GPU slot is held. "
    "Log text is redacted. Unresolved failures are in gogeta_errors."))

RO = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
RW = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def tool(name: str, readonly: bool = True):
    """Register a logged tool: records timing/args in tool_calls and exceptions in errors."""
    def deco(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            t0 = time.time()
            try:
                r = fn(*a, **k)
                ok = not (isinstance(r, dict) and r.get("error"))
                logdb.tool_call("mcp", name, k or list(a), (time.time() - t0) * 1000, ok, None if ok else str(r.get("error"))[:300])
                return r
            except Exception as ex:  # noqa: BLE001
                logdb.tool_call("mcp", name, k, (time.time() - t0) * 1000, False, "%s: %s" % (type(ex).__name__, ex))
                logdb.error("mcp", str(ex), type(ex).__name__, traceback.format_exc(), context={"tool": name, "args": k})
                return {"error": "%s: %s" % (type(ex).__name__, ex)}
        return mcp.tool(name=name, annotations=RO if readonly else RW)(inner)
    return deco


def expose(fn, name: str, readonly: bool = True):
    """Extension pattern: turn ANY core function (typed args, docstring, JSON-able dict result) into a logged MCP tool.
    The schema comes from its type hints, the description from its docstring."""
    return tool(name, readonly)(fn)


@tool("gogeta_brief")
def gogeta_brief() -> dict:
    """START HERE. One-call project overview (~4 KB): version, GPU/campaign state, queue top 5, needs-human items, latest results and runs, latest review, claims counts, unresolved errors."""
    return core.brief()


@tool("gogeta_status")
def gogeta_status() -> dict:
    """Live status: GPU slot holder, temperature (85 C cool-down limit), utilisation, running/ended runs, CPU jobs (n/8), campaigns with per-item val_loss, server ports up/down."""
    return core.status()


@tool("gogeta_queue")
def gogeta_queue(filter: str = "", next_kind: str = "", item: str = "", include_done: bool = False) -> dict:
    """Experiment queue in priority order. filter: free text or 'state:READY', 'res:gpu', 'kind:train', 'status:running'. next_kind='gpu'|'cpu' returns what `queue.py next` would pick (and why nothing, if blocked). item='K7' returns one full item."""
    if item:
        return core.queue_item(item)
    if next_kind:
        return core.queue_next(next_kind)
    return core.queue(filter, include_done)


@tool("gogeta_experiment")
def gogeta_experiment(id: str) -> dict:
    """Everything about one experiment/check/side quest id (E10, K7, SQ6...): registry entry, documentation card, latest runs, queue item, devlog '### <ID>' blocks."""
    return core.experiment(id)


@tool("gogeta_index")
def gogeta_index() -> dict:
    """The generated experiment index (id, title, status, headline result, reuse, config). Read before designing a new experiment."""
    return core.index()


@tool("gogeta_claims")
def gogeta_claims(query: str = "", status: str = "", id: str = "") -> dict:
    """Claims ledger rows. Filter by free text, status (VERIFIED, PARTIAL, REFUTED, UNVERIFIED, WRONG FOR OUR CODE) or id (C12)."""
    return core.claims(query, status, id)


@tool("gogeta_side_quests")
def gogeta_side_quests(query: str = "", status: str = "") -> dict:
    """Side-quest ledger rows (idea/running/done/dropped), filterable by text and status."""
    return core.side_quests(query, status)


@tool("gogeta_search")
def gogeta_search(text: str, limit: int = 40) -> dict:
    """Case-insensitive search across devlog, AGENTS, README, claims, side quests, queue, index, cards, notes and reviews. Returns file:line hits."""
    return core.search(text, limit)


@tool("gogeta_runs")
def gogeta_runs(experiment: str = "", limit: int = 15) -> dict:
    """Run folders (summary.json): val_loss, bits/char, tokens/s, lr, seed, git. Optional experiment filter (E10). For SQL-backed history with filters use gogeta_run_history."""
    return core.runs(experiment, limit)


@tool("gogeta_results")
def gogeta_results(name: str = "") -> dict:
    """Check/eval result JSON files. No name: list them. With a name (K7-parity, K4, ...): the (size-capped, redacted) content."""
    return core.results(name) if name else core.list_results()


@tool("gogeta_devlog")
def gogeta_devlog(version: str = "", latest: int = 0, tags: bool = False) -> dict:
    """Devlog: version='v0.4.21' returns that entry; latest=N the newest N (truncated); tags=true lists git version tags with titles."""
    if tags:
        return core.versions()
    return core.devlog(version, latest)


@tool("gogeta_notes")
def gogeta_notes(name: str = "", search_notebook: str = "", reviews: int = 0) -> dict:
    """Notes: no args lists notes/*.md; name reads one; search_notebook greps the lab notebook; reviews=N returns the latest N outside reviews."""
    if search_notebook:
        return core.notebook(search_notebook)
    if reviews:
        return core.reviews(reviews)
    return core.notes(name)


@tool("gogeta_log_tail")
def gogeta_log_tail(name: str = "", n: int = 40) -> dict:
    """Redacted tail of a training run, campaign or CPU-job log. name: run id prefix (E10-s0-2026...), campaign name, or 'cpu:<job>'. Empty name lists available logs."""
    return core.log_tail(name, n) if name else core.list_logs()


@tool("gogeta_servers")
def gogeta_servers() -> dict:
    """Local servers (tracker 7861, viz 7870, site 8791, 3D page) with up/down probes and URLs. All bind 127.0.0.1."""
    return core.servers()


@tool("gogeta_start_server", readonly=False)
def gogeta_start_server(name: str, port: int = 0) -> dict:
    """STATE-CHANGING: start tracker|viz (via tools/launch_tracker.sh / launch_viz.sh) or site (static docs server), localhost only. Idempotent: does nothing if the port already answers."""
    return core.start_server(name, port or None)


@tool("gogeta_stop_server", readonly=False)
def gogeta_stop_server(name: str, port: int = 0) -> dict:
    """STATE-CHANGING: stop a tracker|viz|site server that the project launchers or this cockpit started. Never kills foreign processes."""
    return core.stop_server(name, port or None)


@tool("gogeta_launch_queue_item", readonly=False)
def gogeta_launch_queue_item(id: str, confirm: bool = False, dry_run: bool = False) -> dict:
    """STATE-CHANGING (needs confirm=true): launch a queue item through its recorded launcher (tools/launch_experiment.sh|launch_campaign.sh|launch_cpu_job.sh only; never gogeta.train, never skip-preflight flags). Refused if the item is blocked, needs a human, the GPU slot is held, a campaign is alive, GPU >= 85 C or code paths are dirty. dry_run=true runs the launcher's own --dry-run (or just validates). Does not edit queue.json."""
    return core.launch_queue_item(id, confirm, dry_run)


@tool("gogeta_errors")
def gogeta_errors(since: str = "", source: str = "", run: str = "", unresolved: bool = True, limit: int = 20, full: bool = False) -> dict:
    """Errors from the SQLite store (tracebacks, NaN/OOM, preflight refusals, failed launches). since: '2h','1d' or ISO date. source: train|launcher|campaign|cpu-job|server|mcp|tui|cockpit. full=true returns whole tracebacks. Unresolved first."""
    logdb.ingest(max_seconds=6)
    return logdb.errors(since, source, "", run, unresolved, limit, full)


@tool("gogeta_events")
def gogeta_events(since: str = "", source: str = "", level: str = "", run: str = "", kind: str = "", limit: int = 40) -> dict:
    """Events from the SQLite store (launches, campaign progress, server actions, run finishes). Filters: since ('2h'), source, level (debug/info/warn/error), run id prefix, kind."""
    logdb.ingest(max_seconds=6)
    return logdb.events(since, source, level, run, kind, limit)


@tool("gogeta_run_history")
def gogeta_run_history(experiment: str = "", status: str = "", campaign: str = "", since: str = "", limit: int = 30, order: str = "start") -> dict:
    """SQL-backed run history (id, experiment, seed, lr, status, val_loss, bpc, tokens/s, git, campaign). order: start|val_loss|tokens_per_s."""
    logdb.ingest(max_seconds=6)
    return logdb.run_history(experiment, status, campaign, since, limit, order)


@tool("gogeta_resolve_error", readonly=False)
def gogeta_resolve_error(id: int, note: str, confirm: bool = False) -> dict:
    """STATE-CHANGING (needs confirm=true): mark error #id resolved with a required note saying what was done. Never deletes anything."""
    if confirm is not True:
        return {"resolved": False, "refused": "confirm must be exactly true"}
    r = logdb.resolve_error(id, note)
    logdb.event("mcp", "info", "error_resolved", "error #%s resolved: %s" % (id, note[:200]))
    return r


@tool("gogeta_sql")
def gogeta_sql(query: str, limit: int = 100) -> dict:
    """Read-only SQL (single SELECT/WITH only, enforced by statement check + read-only connection + authorizer; rows capped at 200) over the cockpit SQLite store. Tables: runs, events, errors, campaigns, campaign_items, servers, tool_calls."""
    logdb.ingest(max_seconds=6)
    return logdb.sql(query, min(int(limit), 200))


@tool("gogeta_incidents")
def gogeta_incidents(open_only: bool = True, status: str = "", kind: str = "", category: str = "", limit: int = 20) -> dict:
    """Watchdog incidents (crashes, hangs, NaN/OOM, aborted campaigns, servers down, GPU hot, disk low), open first. Each has a category (code-bug/env/transient/thermal/data/dirty-tree/resource), auto-actions taken and a resume file if any. Use gogeta_incident for the full bundle and agent brief."""
    return watchdog.incidents(status, open_only, kind, category, limit)


@tool("gogeta_incident")
def gogeta_incident(id: str) -> dict:
    """Full incident: redacted log tail, traceback, git/dirty paths, GPU state, config, CLI args, last metrics, classification, auto-actions, notes and a ready-to-paste `agent_brief` (what failed, evidence, suspected cause, files, repro command, what 'fixed' means)."""
    return watchdog.incident(id)


@tool("gogeta_incident_update", readonly=False)
def gogeta_incident_update(id: str, status: str = "", note: str = "", confirm: bool = False) -> dict:
    """STATE-CHANGING (needs confirm=true): move an incident along open -> triaged -> fixing -> fixed -> verified (or wontfix) and/or add a note. 'fixed'/'verified' require a note (what was changed, the version tag, the post-mortem if it cost > 10 GPU-min)."""
    if confirm is not True:
        return {"refused": "confirm must be exactly true"}
    return watchdog.update_incident(id, status, note)


@tool("gogeta_resume_campaign", readonly=False)
def gogeta_resume_campaign(file: str, confirm: bool = False, dry_run: bool = False) -> dict:
    """STATE-CHANGING (needs confirm=true): launch a watchdog-written *-resume-N.campaign (exactly the unfinished items) through tools/launch_campaign.sh. Refused if code paths are dirty, the GPU slot is held, a campaign is alive or the GPU is >= 85 C. dry_run=true runs the launcher's --dry-run."""
    return watchdog.resume_campaign(file, confirm, dry_run)


@tool("gogeta_watchdog", readonly=False)
def gogeta_watchdog(action: str = "status") -> dict:
    """Watchdog daemon: action=status|start|stop|scan. start is idempotent (one instance); scan runs one detection pass now. It only does safe recovery (restart servers, write/launch resume campaigns for transient/thermal causes); it never edits code."""
    if action == "start":
        return watchdog.start()
    if action == "stop":
        return watchdog.stop()
    if action == "scan":
        r = watchdog.scan_once()
        return {"new": r["new"], "findings": r["findings"], "detector_errors": r["detector_errors"]}
    return watchdog.status()


expose(core.backlog, "gogeta_backlog")  # cockpit backlog: read before adding features (AGENTS 0g)
# Run report forms (cockpit/runform.py): one form per finished run. draft = read-only; submit writes experiments/run-forms/ only.
expose(runform.run_form_draft, "gogeta_run_form_draft")
expose(runform.run_form_submit, "gogeta_run_form_submit", readonly=False)
expose(runform.run_forms_list, "gogeta_run_forms")


@mcp.resource("gogeta://agents", name="AGENTS.md", description="Binding repo rules for agents", mime_type="text/markdown")
def res_agents() -> str:
    return core._read(core.REPO / "AGENTS.md")


@mcp.resource("gogeta://queue-md", name="QUEUE.md", description="Generated queue view", mime_type="text/markdown")
def res_queue() -> str:
    return core._read(core.REPO / "experiments/QUEUE.md")


@mcp.resource("gogeta://index", name="INDEX.md", description="Generated experiment index", mime_type="text/markdown")
def res_index() -> str:
    return core._read(core.REPO / "experiments/INDEX.md")


if __name__ == "__main__":
    mcp.run(transport="stdio")
