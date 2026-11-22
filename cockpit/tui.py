#!/usr/bin/env python3
"""Gogeta-0 cockpit TUI (Textual). Run in WSL: ~/.venvs/gogeta-cockpit/bin/python cockpit/tui.py

Tabs: Dashboard, Queue, Experiments, Claims, Devlog, Logs, Errors, Runs, Search.  `?` = key bindings.
All data comes from cockpit/core.py and cockpit/logdb.py (the same code the MCP server uses).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import logdb  # noqa: E402
import runform  # noqa: E402
import watchdog  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.text import Text  # noqa: E402
from textual import on  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, ProgressBar, RichLog,  # noqa: E402
                             Select, Static, TabbedContent, TabPane)

GOLD = "#f5a623"
TABS = [("dashboard", "Dashboard", "d"), ("queue", "Queue", "u"), ("experiments", "Experiments", "e"),
        ("claims", "Claims", "c"), ("devlog", "Devlog", "v"), ("logs", "Logs", "l"), ("errors", "Errors", "x"), ("incidents", "Incidents", "i"),
        ("runs", "Runs", "h"), ("search", "Search", "s")]
HELP = """[b #f5a623]Gogeta-0 cockpit: key bindings[/]

  [b]d u e c v l x h s[/]   jump to Dashboard / Queue / Experiments / Claims / Devlog / Logs / Errors / Runs / Search
  [b]r[/]            refresh now (the Dashboard also refreshes every 5 s)
  [b]/[/]            global search
  [b]?[/]            this help          [b]q[/]  quit          [b]escape[/]  close dialogs
  Tables: arrows move, [b]enter[/] selects, click a column header to sort.
  Queue: [b]L[/] launch selected (asks first), [b]D[/] dry run, [b]g[/] next GPU, [b]n[/] next CPU.
  Errors: [b]R[/] resolve selected (note required).
  Logs: [b]f[/] toggle follow.
  Runs: [b]Run forms[/] button lists every finished run with its form status (NONE / DRAFT / filled); [b]Draft form[/] drafts the selected run.

State-changing buttons: server Start/Stop and queue Launch. Launch only calls the project launchers and is refused
while the GPU slot is held. Everything shown is token-redacted. Nothing here deletes data."""


class Confirm(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.text, id="dialog-text")
            with Horizontal(id="dialog-buttons"):
                yield Button("Confirm", id="yes", variant="warning")
                yield Button("Cancel", id="no")

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id == "yes")


class Prompt(ModalScreen[str]):
    BINDINGS = [Binding("escape", "dismiss('')", "Cancel")]

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.text, id="dialog-text")
            yield Input(placeholder="resolution note (required)", id="note")
            with Horizontal(id="dialog-buttons"):
                yield Button("Save", id="yes", variant="warning")
                yield Button("Cancel", id="no")

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(self.query_one("#note", Input).value if ev.button.id == "yes" else "")

    @on(Input.Submitted)
    def submitted(self, ev: Input.Submitted) -> None:
        self.dismiss(ev.value)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("question_mark", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(HELP, id="dialog-text")
            yield Button("Close", id="close")

    @on(Button.Pressed)
    def pressed(self) -> None:
        self.dismiss()


def bar(value, lo, hi, width=20) -> str:
    try:
        f = max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))
    except (TypeError, ValueError):
        f = 0.0
    n = int(round(f * width))
    return "#" * n + "-" * (width - n)


class CockpitApp(App):
    TITLE = "Gogeta-0 cockpit"
    SUB_TITLE = "FusionDance lab"
    CSS = """
    $gold: #f5a623;
    Screen { background: $surface; }
    Header { background: #7a3e00; color: white; }
    TabbedContent > ContentTabs { background: $panel; }
    Tabs .-active { color: $gold; text-style: bold; }
    .card { border: round $gold; padding: 0 1; margin: 0 0 1 0; height: auto; }
    .card-title { color: $gold; text-style: bold; }
    #dash { height: 1fr; padding: 0 1; }
    DataTable { height: auto; max-height: 14; }
    #queue-table, #exp-table, #claims-table, #errors-table, #runs-table, #forms-table, #search-table, #dev-table { height: 1fr; max-height: 100%; }
    .detail { height: 1fr; border: round $gold; padding: 0 1; }
    RichLog { border: round $gold; height: 1fr; }
    .srv { height: 3; }
    .srv Static { width: 1fr; content-align: left middle; }
    .srv Button { min-width: 9; margin-left: 1; }
    .toolbar { height: 3; }
    .toolbar Button { margin-right: 1; }
    #dialog { width: 70; max-width: 95%; height: auto; border: heavy $gold; background: $panel; padding: 1 2; align: center middle; }
    #dialog-buttons { height: 3; align-horizontal: right; }
    ModalScreen { align: center middle; }
    #errbadge { dock: top; height: 1; background: $panel; padding: 0 1; }
    """
    BINDINGS = [Binding(k, "tab('%s')" % tid, name, show=False) for tid, name, k in TABS] + [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("slash", "tab('search')", "Search"),
        Binding("L", "launch(False)", "Launch", show=False),
        Binding("D", "launch(True)", "Dry-run", show=False),
        Binding("g", "next('gpu')", "Next GPU", show=False),
        Binding("n", "next('cpu')", "Next CPU", show=False),
        Binding("R", "resolve", "Resolve", show=False),
        Binding("f", "follow", "Follow", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.status: dict = {}
        self.follow = True
        self._queue_rows: list = []
        self._sort = {}

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="errbadge")
        with TabbedContent(initial="dashboard", id="tabs"):
            with TabPane("Dashboard", id="dashboard"):
                with VerticalScroll(id="dash"):
                    yield Static("loading...", id="gpu", classes="card")
                    with Vertical(classes="card", id="camp-card"):
                        yield Static("Campaign", classes="card-title", id="camp-title")
                        yield ProgressBar(total=1, show_eta=False, id="camp-bar")
                        yield DataTable(id="camp-table", cursor_type="none", zebra_stripes=True)
                    yield Static("", id="cpu", classes="card")
                    with Vertical(classes="card"):
                        yield Static("Servers (127.0.0.1 only)", classes="card-title")
                        for name, label in (("tracker", "Tracker :7861"), ("viz", "Visualiser :7870"),
                                            ("site", "Site :8791"), ("arch3d", "3D page /architecture.html")):
                            with Horizontal(classes="srv"):
                                yield Static(label, id="srv-%s" % name)
                                yield Button("Start", id="start-%s" % name, variant="success")
                                yield Button("Stop", id="stop-%s" % name, variant="error")
                                yield Button("Open", id="open-%s" % name)
                    yield Static("", id="power", classes="card")
            with TabPane("Queue", id="queue"):
                with Horizontal(classes="toolbar"):
                    yield Button("Next GPU", id="q-next-gpu")
                    yield Button("Next CPU", id="q-next-cpu")
                    yield Button("Dry run", id="q-dry")
                    yield Button("Launch...", id="q-launch", variant="warning")
                yield DataTable(id="queue-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="queue-detail", wrap=True, markup=False)
            with TabPane("Experiments", id="experiments"):
                yield DataTable(id="exp-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="exp-detail", wrap=True, markup=False)
            with TabPane("Claims", id="claims"):
                with Horizontal(classes="toolbar"):
                    yield Select([("Claims", "claims"), ("Side quests", "sq")], value="claims", allow_blank=False, id="c-kind")
                    yield Select([("any status", ""), ("VERIFIED", "VERIFIED"), ("PARTIAL", "PARTIAL"), ("REFUTED", "REFUTED"),
                                  ("UNVERIFIED", "UNVERIFIED"), ("WRONG FOR OUR CODE", "WRONG FOR OUR CODE"),
                                  ("idea", "idea"), ("running", "running"), ("done", "done"), ("dropped", "dropped")],
                                 value="", allow_blank=False, id="c-status")
                    yield Input(placeholder="filter text", id="c-query")
                yield DataTable(id="claims-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Devlog", id="devlog"):
                yield DataTable(id="dev-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="dev-detail", wrap=True, markup=False)
            with TabPane("Logs", id="logs"):
                with Horizontal(classes="toolbar"):
                    yield Select([("(loading)", "")], value="", allow_blank=False, id="log-pick")
                    yield Button("Follow: on", id="log-follow")
                yield RichLog(id="log-view", wrap=True, markup=False, max_lines=2000)
            with TabPane("Errors", id="errors"):
                with Horizontal(classes="toolbar"):
                    yield Button("Resolve...", id="err-resolve", variant="warning")
                    yield Button("Show resolved: off", id="err-all")
                yield DataTable(id="errors-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="err-detail", wrap=True, markup=False)
            with TabPane("Incidents", id="incidents"):
                with Horizontal(classes="toolbar"):
                    yield Button("Resume campaign", id="inc-resume", variant="warning")
                    yield Button("Restart server", id="inc-restart")
                    yield Select([(s_, s_) for s_ in watchdog.STATUSES + ["wontfix"]], value="triaged", allow_blank=False, id="inc-status")
                    yield Button("Set status...", id="inc-set")
                    yield Button("Copy brief", id="inc-copy")
                    yield Button("Watchdog", id="inc-wd")
                yield DataTable(id="incidents-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="inc-detail", wrap=True, markup=False)
            with TabPane("Runs", id="runs"):
                with Horizontal(classes="toolbar"):
                    yield Button("Run forms: off", id="runs-forms")
                    yield Button("Draft form", id="form-draft", variant="primary")
                yield DataTable(id="runs-table", cursor_type="row", zebra_stripes=True)
                yield DataTable(id="forms-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="form-detail", wrap=True, markup=False)
            with TabPane("Search", id="search"):
                yield Input(placeholder="search devlog, claims, cards, notes... (enter)", id="search-input")
                yield DataTable(id="search-table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    # ------------------------------------------------------------------ plumbing
    def on_mount(self) -> None:
        self.theme = "textual-dark"
        for tid, cols in {
            "camp-table": ["#", "item", "rc", "val_loss", "bpc", "min"],
            "queue-table": ["pos", "id", "state", "res", "min", "title"],
            "exp-table": ["id", "status", "title", "headline"],
            "claims-table": ["id", "status", "text"],
            "dev-table": ["version", "title"],
            "errors-table": ["id", "state", "ts", "source", "type", "message"],
            "incidents-table": ["id", "status", "sev", "kind", "category", "subject", "summary"],
            "runs-table": ["run", "exp", "seed", "lr", "status", "val_loss", "bpc", "tok/s", "start"],
            "forms-table": ["run", "exp", "seed", "lr", "val_loss", "form"],
            "search-table": ["file:line", "text"],
        }.items():
            t = self.query_one("#" + tid, DataTable)
            t.add_columns(*cols)
        self.query_one("#forms-table").display = False
        self.query_one("#form-detail").display = False
        self.refresh_dashboard()
        self.set_interval(5, self.refresh_dashboard)
        self.set_interval(2, self.tail_log)
        self.load_tab("dashboard")
        self.load_claims()

    def bg(self, name: str, fn, done, *args) -> None:
        """Run fn(*args) in a thread, log it as a tui tool call, then done(result) on the UI thread."""
        def job():
            t0 = time.time()
            try:
                r = fn(*args)
                ok = not (isinstance(r, dict) and r.get("error"))
                logdb.tool_call("tui", name, list(args), (time.time() - t0) * 1000, ok)
            except Exception as ex:  # noqa: BLE001
                logdb.tool_call("tui", name, list(args), (time.time() - t0) * 1000, False, repr(ex))
                logdb.error("tui", str(ex), type(ex).__name__, traceback.format_exc(), context={"action": name})
                r = {"error": "%s: %s" % (type(ex).__name__, ex)}
            self.call_from_thread(done, r)
        self.run_worker(job, thread=True, group=name, exit_on_error=False)

    def action_tab(self, tid: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tid

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_refresh(self) -> None:
        self.refresh_dashboard()
        self.load_tab(self.query_one("#tabs", TabbedContent).active)

    @on(TabbedContent.TabActivated)
    def tab_activated(self, ev: TabbedContent.TabActivated) -> None:
        self.load_tab(ev.pane.id)

    def load_tab(self, tid: str) -> None:
        {"queue": self.load_queue, "experiments": self.load_experiments, "claims": self.load_claims,
         "devlog": self.load_devlog, "logs": self.load_logs, "errors": self.load_errors, "incidents": self.load_incidents, "runs": self.load_runs}.get(tid, lambda: None)()

    # ------------------------------------------------------------------ dashboard
    def refresh_dashboard(self) -> None:
        def work():
            logdb.ingest(max_seconds=6)
            return {"status": core.status(), "unresolved": logdb.unresolved_count(), "inc_open": watchdog.open_count(),
                    "wd": watchdog.status(),
                    "best": logdb.run_history(order="val_loss", limit=1)["runs"]}
        self.bg("dashboard_refresh", work, self.apply_dashboard)

    def apply_dashboard(self, d: dict) -> None:
        if d.get("error"):
            self.query_one("#gpu", Static).update("status error: " + escape(str(d["error"])))
            return
        s = self.status = d["status"]
        n, ni, wd = d["unresolved"], d["inc_open"], d["wd"]
        wd_txt = ("OK, beat %ss ago" % wd["heartbeat_age_s"]) if wd["healthy"] else (
            "RUNNING but stale" if wd["running"] else "NOT RUNNING (Incidents tab > Watchdog)")
        self.query_one("#errbadge", Static).update(
            Text("INCIDENTS OPEN: %d (i)  |  errors unresolved: %d (x)  |  watchdog: %s" % (max(ni, 0), max(n, 0), wd_txt),
                 style="bold white on #a02020" if (ni > 0 or n > 0 or not wd["healthy"]) else "green"))
        g = s["gpu"]
        slot = g["slot"]
        t = g.get("temp_c")
        band = {"ok": "OK", "warm": "WARM (launches wait above 75 C after a cool-down)", "hot": "HOT: cool-down, launches refused"}.get(g.get("band"), "?")
        holder = ("HELD by [b]%s[/] seed %s, pid %s, up %s" % (escape(str(slot.get("experiment"))), slot.get("seed"), slot.get("pid"),
                                                              slot.get("running_for"))) if slot["held"] else (
            "free" + (" (stale lock)" if slot["stale_lock"] else ""))
        procs = "; ".join(g.get("compute_procs", [])[:2])
        self.query_one("#gpu", Static).update(
            "[b #f5a623]GPU[/]  %s\n slot : %s\n temp : [b]%s C[/]  [%s] %s  (limit %d C)\n util : %s%%  [%s]   mem %s MiB\n procs: %s"
            % (escape(str(g.get("name", ""))), holder, t, bar(t, 40, g["hot_limit_c"]), band, g["hot_limit_c"], g.get("util_pct"),
               bar(g.get("util_pct"), 0, 100, 12), g.get("mem_used_mib"), escape(procs or "none")))
        camps = [c for c in s["campaigns"] if c["state"] == "running"] or s["campaigns"][:1]
        bar_w, tbl = self.query_one("#camp-bar", ProgressBar), self.query_one("#camp-table", DataTable)
        tbl.clear()
        if camps:
            c = camps[0]
            cur = c["current"]
            self.query_one("#camp-title", Static).update("[b #f5a623]Campaign[/] %s  [%s]  %d/%d done, %d failed%s" % (
                escape(c["name"]), c["state"].upper(), c["done"], c["total"], c["failed"],
                ("; now: " + escape(cur["args"])) if cur else ""))
            bar_w.update(total=max(c["total"], 1), progress=c["done"])
            for it in c["items"]:
                tbl.add_row(str(it["idx"]), it["args"].replace("--max-tokens", "tok"), str(it["rc"]),
                            "%.4f" % it["val_loss"] if it.get("val_loss") else "-",
                            "%.4f" % it["bpc"] if it.get("bpc") else "-", str(it.get("wall_min") or "-"))
        else:
            self.query_one("#camp-title", Static).update("[b #f5a623]Campaign[/] none")
            bar_w.update(total=1, progress=0)
        cpu = s["cpu"]
        self.query_one("#cpu", Static).update("[b #f5a623]CPU jobs[/] %d/%d slots  %s" % (
            cpu["alive"], cpu["max"], escape("; ".join("%s (%s)" % (j["name"], j["running_for"]) for j in cpu["jobs"]) or "(idle)")))
        for name, v in s["servers"].items():
            self.query_one("#srv-%s" % name, Static).update("%s  [b]%s[/]  %s" % (
                {"tracker": "Tracker :7861", "viz": "Visualiser :7870", "site": "Site :8791",
                 "arch3d": "3D page"}[name], "UP  " if v["up"] else "DOWN", v["url"]))
        best = d["best"][0] if d["best"] else None
        run = s["runs"][0] if s["runs"] else None
        pl = int(1000 / best["val_loss"]) if best and best.get("val_loss") else 0
        self.query_one("#power", Static).update(
            "[b #f5a623]POWER LEVEL[/]  [b]%s[/]%s\n (1000 / best val_loss; higher is better)   last run: %s" % (
                "{:,}".format(pl), (" (best %s %.4f)" % (escape(str(best["experiment"])), best["val_loss"])) if best else "",
                escape("%s %s" % (run["name"], run["state"])) if run else "none"))

    @on(Button.Pressed)
    def button(self, ev: Button.Pressed) -> None:
        bid = ev.button.id or ""
        if bid.startswith(("start-", "stop-", "open-")):
            act, name = bid.split("-", 1)
            if name == "arch3d":
                if act == "open":
                    self.open_url(self.status.get("servers", {}).get("arch3d", {}).get("url", "http://127.0.0.1:8791/architecture.html"))
                else:
                    self.notify("The 3D page is served by the site server: use the Site buttons.")
                return
            if act == "open":
                url = "http://127.0.0.1:%d/" % core.SERVERS[name]["port"]
                self.open_url(url)
                self.notify("Opening " + url)
                return
            fn = core.start_server if act == "start" else core.stop_server
            self.notify("%s %s ..." % (act, name))
            self.bg("%s_server" % act, fn, lambda r: (self.notify(
                "%s %s: %s" % (act, name, "ok" if (r.get("up") if act == "start" else not r.get("up")) else str(r)[:120]),
                severity="information" if "error" not in r else "error"), self.refresh_dashboard()), name)
        elif bid == "runs-forms":
            self.show_forms = not getattr(self, "show_forms", False)
            self.query_one("#runs-forms", Button).label = "Run forms: %s" % ("on" if self.show_forms else "off")
            self.query_one("#runs-table").display = not self.show_forms
            self.query_one("#forms-table").display = self.show_forms
            self.query_one("#form-detail").display = self.show_forms
            self.load_runs()
        elif bid == "form-draft":
            self.draft_form()
        elif bid == "q-next-gpu":
            self.action_next("gpu")
        elif bid == "q-next-cpu":
            self.action_next("cpu")
        elif bid == "q-dry":
            self.action_launch(True)
        elif bid == "q-launch":
            self.action_launch(False)
        elif bid == "log-follow":
            self.action_follow()
        elif bid.startswith("inc-"):
            self.incident_button(bid)
        elif bid == "err-resolve":
            self.action_resolve()
        elif bid == "err-all":
            self.show_resolved = not getattr(self, "show_resolved", False)
            ev.button.label = "Show resolved: %s" % ("on" if self.show_resolved else "off")
            self.load_errors()

    # ------------------------------------------------------------------ queue
    def load_queue(self) -> None:
        self.bg("queue", lambda: core.queue(include_done=True, limit=200), self.apply_queue)

    def apply_queue(self, d: dict) -> None:
        self._queue_rows = d.get("rows", [])
        self.fill_queue()
        self.notify_queue_next(d)

    def notify_queue_next(self, d):
        self.sub_title = "next GPU: %s | next CPU: %s" % (d.get("next_gpu") or "-", d.get("next_cpu") or "-")

    def fill_queue(self) -> None:
        t = self.query_one("#queue-table", DataTable)
        t.clear()
        for r in self._queue_rows:
            t.add_row(str(r["pos"]), r["id"], r["state"], r["res"], str(r["est_min"]), r["title"][:90], key=r["id"])

    def selected_queue_id(self):
        t = self.query_one("#queue-table", DataTable)
        if t.row_count and t.cursor_row is not None:
            try:
                return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
            except Exception:  # noqa: BLE001
                return None
        return None

    @on(DataTable.RowHighlighted, "#queue-table")
    def queue_row(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None or ev.row_key.value is None:
            return
        it = core.queue_item(ev.row_key.value)
        log = self.query_one("#queue-detail", RichLog)
        log.clear()
        log.write(core.redact(json.dumps({k: v for k, v in it.items() if k not in ("history",)}, indent=1, ensure_ascii=False)))

    @on(DataTable.HeaderSelected, "#queue-table")
    def queue_sort(self, ev: DataTable.HeaderSelected) -> None:
        key = ["pos", "id", "state", "res", "est_min", "title"][ev.column_index]
        rev = self._sort.get("queue") == key
        self._sort["queue"] = None if rev else key
        self._queue_rows.sort(key=lambda r: (r[key] is None, r[key]), reverse=rev)
        self.fill_queue()

    def action_next(self, kind: str) -> None:
        def done(r):
            it = r.get("item")
            self.query_one("#queue-detail", RichLog).clear()
            self.query_one("#queue-detail", RichLog).write(
                ("NEXT %s: %s  %s\nwhy: %s\nlaunch: %s" % (kind.upper(), it["id"], it["title"], it.get("why"), it.get("launch"))) if it
                else "Nothing pickable for %s.\n%s" % (kind, json.dumps(r, indent=1)))
        self.bg("queue_next", core.queue_next, done, kind)

    def action_launch(self, dry: bool) -> None:
        qid = self.selected_queue_id()
        if not qid:
            self.notify("Select a queue row first (Queue tab).", severity="warning")
            return
        item = core.queue_item(qid)

        def run(confirm: bool):
            def done(r):
                self.query_one("#queue-detail", RichLog).clear()
                self.query_one("#queue-detail", RichLog).write(json.dumps(r, indent=1, ensure_ascii=False))
                self.notify(("launched" if r.get("launched") else "not launched: %s" % str(r.get("refused", r.get("output", "")))[:100]),
                            severity="information" if r.get("launched") or dry else "warning")
                self.refresh_dashboard()
            self.bg("launch_queue_item", core.launch_queue_item, done, qid, confirm, dry)
        if dry:
            run(True)
            return
        self.push_screen(Confirm("Launch queue item [b]%s[/]?\n\n%s\n\ncommand: %s\n\nRefused automatically if the GPU slot is held, a campaign "
                                 "is alive, the GPU is >= 85 C or code paths are dirty." % (
                                     escape(qid), escape(item.get("title", "")), escape(str(item.get("launch")))))
                         , lambda ok: run(True) if ok else None)

    # ------------------------------------------------------------------ experiments
    def load_experiments(self) -> None:
        self.bg("index", core.index, self.apply_index)

    def apply_index(self, d: dict) -> None:
        t = self.query_one("#exp-table", DataTable)
        t.clear()
        for r in d.get("rows", []):
            t.add_row(r["id"], r["status"], r["title"][:50], r["headline"][:70], key=r["id"])

    @on(DataTable.RowHighlighted, "#exp-table")
    def exp_row(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None or ev.row_key.value is None:
            return
        eid = ev.row_key.value

        def done(e):
            log = self.query_one("#exp-detail", RichLog)
            log.clear()
            rr = [r for r in e.get("latest_runs", []) if r.get("val_loss")]
            if rr:
                lo, hi = min(r["val_loss"] for r in rr), max(r["val_loss"] for r in rr)
                log.write("val_loss by run (shorter bar = lower = better)")
                for r in rr:
                    log.write("  seed %s lr %-7s %s %.4f  %s" % (r["seed"], r["lr"], bar(r["val_loss"], lo - 1e-9, hi + 1e-9 if hi > lo else lo + 1, 20),
                                                              r["val_loss"], r["run"]))
                log.write("")
            log.write(e.get("card") or "(no card)")
        self.bg("experiment", core.experiment, done, eid)

    # ------------------------------------------------------------------ claims / side quests
    def load_claims(self) -> None:
        kind = self.query_one("#c-kind", Select).value
        status = self.query_one("#c-status", Select).value or ""
        q = self.query_one("#c-query", Input).value
        fn = (lambda: core.claims(q, status)) if kind == "claims" else (lambda: core.side_quests(q, status))

        def done(d):
            t = self.query_one("#claims-table", DataTable)
            t.clear()
            for r in d.get("rows", []):
                t.add_row(r["id"], r["status"] or "?", (r.get("claim") or r.get("idea") or "")[:110])
        self.bg("claims", fn, done)

    @on(Select.Changed, "#c-kind")
    @on(Select.Changed, "#c-status")
    @on(Input.Changed, "#c-query")
    def claims_changed(self) -> None:
        if self.is_mounted if hasattr(self, "is_mounted") else True:
            self.load_claims()

    # ------------------------------------------------------------------ devlog
    def load_devlog(self) -> None:
        self.bg("devlog", lambda: core.devlog(latest=60), self.apply_devlog)

    def apply_devlog(self, d: dict) -> None:
        t = self.query_one("#dev-table", DataTable)
        t.clear()
        for e in d.get("entries", []):
            t.add_row(e["version"], e["title"][:90], key=e["version"])

    @on(DataTable.RowHighlighted, "#dev-table")
    def dev_row(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None or ev.row_key.value is None:
            return

        def done(e):
            log = self.query_one("#dev-detail", RichLog)
            log.clear()
            log.write(e.get("text") or str(e))
        self.bg("devlog_entry", core.devlog, done, ev.row_key.value)

    # ------------------------------------------------------------------ logs
    def load_logs(self) -> None:
        def done(d):
            sel = self.query_one("#log-pick", Select)
            opts = [(("%s  [%s]" % (l["name"], l["kind"]))[:80], l["name"]) for l in d.get("logs", [])] or [("(no logs)", "")]
            cur = sel.value
            sel.set_options(opts)
            sel.value = cur if cur in [o[1] for o in opts] else opts[0][1]
        self.bg("list_logs", core.list_logs, done)

    @on(Select.Changed, "#log-pick")
    def log_changed(self) -> None:
        self.query_one("#log-view", RichLog).clear()
        self.tail_log()

    def tail_log(self) -> None:
        if self.query_one("#tabs", TabbedContent).active != "logs" or not self.follow:
            return
        name = self.query_one("#log-pick", Select).value
        if not name or name is Select.BLANK:
            return

        def done(d):
            v = self.query_one("#log-view", RichLog)
            v.clear()
            v.write(d.get("text") or d.get("error", ""))
        self.bg("log_tail", core.log_tail, done, name, 60)

    def action_follow(self) -> None:
        self.follow = not self.follow
        self.query_one("#log-follow", Button).label = "Follow: %s" % ("on" if self.follow else "off")

    # ------------------------------------------------------------------ errors / runs
    def load_errors(self) -> None:
        unresolved = not getattr(self, "show_resolved", False)
        self.bg("errors", lambda: (logdb.ingest(max_seconds=6), logdb.errors(unresolved=unresolved, limit=200, full=True))[1], self.apply_errors)

    def apply_errors(self, d: dict) -> None:
        self._errors = {str(e["id"]): e for e in d.get("errors", [])}
        t = self.query_one("#errors-table", DataTable)
        t.clear()
        for e in d.get("errors", []):
            t.add_row(str(e["id"]), "resolved" if e["resolved"] else "OPEN", e["ts"][:19], e["source"], str(e["exc_type"] or "")[:24],
                      (e["message"] or "").replace("\n", " ")[:90], key=str(e["id"]))
        if not d.get("errors"):
            self.query_one("#err-detail", RichLog).clear()
            self.query_one("#err-detail", RichLog).write("No errors in this view.")

    @on(DataTable.RowHighlighted, "#errors-table")
    def err_row(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None or ev.row_key.value is None:
            return
        e = getattr(self, "_errors", {}).get(ev.row_key.value)
        if not e:
            return
        log = self.query_one("#err-detail", RichLog)
        log.clear()
        log.write("#%s  %s  source=%s  run=%s\n%s: %s\n\nContext: %s\n%s\n\n%s" % (
            e["id"], e["ts"], e["source"], e["run_id"], e["exc_type"], e["message"], e["context"],
            ("RESOLVED: " + str(e["resolution_note"])) if e["resolved"] else "OPEN", e["traceback"] or "(no traceback)"))

    def action_resolve(self) -> None:
        t = self.query_one("#errors-table", DataTable)
        if not t.row_count:
            return
        eid = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value

        def got(note):
            if not note or not note.strip():
                self.notify("A resolution note is required.", severity="warning")
                return
            self.bg("resolve_error", logdb.resolve_error, lambda r: (self.notify("resolved #%s" % eid), self.load_errors(),
                                                                    self.refresh_dashboard()), int(eid), note)
        self.push_screen(Prompt("Resolve error #%s: say what fixed or explains it." % eid), got)

    # ------------------------------------------------------------------ incidents
    def load_incidents(self) -> None:
        self.bg("incidents", lambda: (logdb.ingest(max_seconds=4), watchdog.incidents(open_only=False, limit=100))[1], self.apply_incidents)

    def apply_incidents(self, d: dict) -> None:
        self._incs = {i["id"]: i for i in d.get("incidents", [])}
        t = self.query_one("#incidents-table", DataTable)
        t.clear()
        for i in d.get("incidents", []):
            t.add_row(i["id"], i["status"].upper() if i["status"] == "open" else i["status"], i["severity"], i["kind"],
                      i["category"] or "", str(i["subject"])[:28], i["summary"][:80], key=i["id"])
        if not d.get("incidents"):
            log = self.query_one("#inc-detail", RichLog)
            log.clear()
            log.write("No incidents. The watchdog records crashes, hangs, aborted campaigns, servers down, hot GPU and low disk here.")

    def selected_incident(self):
        t = self.query_one("#incidents-table", DataTable)
        if not t.row_count:
            return None
        try:
            return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001
            return None

    @on(DataTable.RowHighlighted, "#incidents-table")
    def inc_row(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None or ev.row_key.value is None:
            return

        def done(i):
            self._inc_full = i
            log = self.query_one("#inc-detail", RichLog)
            log.clear()
            log.write(i.get("agent_brief") or str(i))
            log.write("\nACTIONS: " + "; ".join(i.get("actions", [])) + "\nNOTES: " + json.dumps(i.get("notes", [])))
        self.bg("incident", watchdog.incident, done, ev.row_key.value)

    def incident_button(self, bid: str) -> None:
        iid = self.selected_incident()
        if bid == "inc-wd":
            self.bg("watchdog_start", watchdog.start, lambda r: (
                self.notify("watchdog: %s" % ("running pid %s" % r.get("pid") if r.get("running") else "not running")),
                self.refresh_dashboard()))
            return
        if not iid:
            self.notify("Select an incident first.", severity="warning")
            return
        inc = getattr(self, "_incs", {}).get(iid, {})
        if bid == "inc-copy":
            txt = (getattr(self, "_inc_full", None) or {}).get("agent_brief") or ""
            if txt:
                self.copy_to_clipboard(txt)
            self.notify("Agent brief copied (%d chars). Paste it to an agent, or it can call gogeta_incident." % len(txt))
        elif bid == "inc-restart":
            if inc.get("subject_type") != "server":
                self.notify("Not a server incident.", severity="warning")
                return
            self.bg("restart_server", core.start_server, lambda r: (
                self.notify("restart %s: %s" % (inc["subject"], "up" if r.get("up") else str(r)[:100])), self.load_incidents()), inc["subject"])
        elif bid == "inc-resume":
            rf = inc.get("resume_file")
            if not rf:
                self.notify("This incident has no resume file.", severity="warning")
                return

            def go(ok):
                if ok:
                    self.bg("resume_campaign", watchdog.resume_campaign, lambda r: (
                        self.notify("launched" if r.get("launched") else "refused: %s" % str(r.get("refused", r.get("output", "")))[:120],
                                    severity="information" if r.get("launched") else "warning"), self.load_incidents()), rf, True)
            self.push_screen(Confirm("Launch resume campaign [b]%s[/] for %s?\n\nIt holds exactly the unfinished items and runs through "
                                     "tools/launch_campaign.sh. Refused if code paths are dirty, the GPU slot is held or the GPU is hot." % (
                                         escape(rf), escape(iid))), go)
        elif bid == "inc-set":
            st = self.query_one("#inc-status", Select).value

            def got(note):
                self.bg("incident_update", watchdog.update_incident, lambda r: (
                    self.notify(str(r.get("error") or "%s -> %s" % (iid, r.get("status"))), severity="error" if r.get("error") else "information"),
                    self.load_incidents(), self.refresh_dashboard()), iid, st, note or "")
            self.push_screen(Prompt("Set %s to %s. Note (required for fixed/verified):" % (iid, st)), got)

    def load_runs(self) -> None:
        if getattr(self, "show_forms", False):
            self.bg("run_forms", runform.run_forms_list, self.apply_forms)
            return
        self.bg("run_history", lambda: (logdb.ingest(max_seconds=6), logdb.run_history(limit=150))[1], self.apply_runs)

    # ---- run report forms (cockpit/runform.py): none / draft / filled
    def draft_form(self) -> None:
        rid = getattr(self, "_form_sel", None)
        if not getattr(self, "show_forms", False) or not rid:
            self.notify("Turn on 'Run forms' and select a run first.", severity="warning")
            return
        self.bg("run_form_draft_save", lambda r: runform.run_form_draft(r, True), lambda d: (
            self.notify(d["error"] if d.get("error") else "Drafted %s (status %s). An agent fills the narrative via gogeta_run_form_submit." % (rid, d.get("form_status")),
                        severity="error" if d.get("error") else "information"), self.load_runs()), rid)

    def apply_forms(self, d: dict) -> None:
        t = self.query_one("#forms-table", DataTable)
        t.clear()
        for r in d.get("runs", []):
            t.add_row(r["run"][:30], r["experiment"] or "", str(r["seed"]), str(r["lr"] or ""), "%.4f" % r["val_loss"] if r["val_loss"] is not None else "-",
                      r["form"].upper() if r["form"] != "filled" else "filled", key=r["run"])
        log = self.query_one("#form-detail", RichLog)
        if not d.get("runs"):
            log.clear()
            log.write("No finished runs.")

    @on(DataTable.RowHighlighted, "#forms-table")
    def form_row(self, ev: DataTable.RowHighlighted) -> None:
        if ev.row_key is None or ev.row_key.value is None:
            return
        rid = ev.row_key.value
        self._form_sel = rid

        def show(d: dict) -> None:
            if getattr(self, "_form_sel", None) != rid:
                return  # stale response
            log = self.query_one("#form-detail", RichLog)
            log.clear()
            if d.get("error"):
                log.write(str(d["error"]))
                return
            i, h, r, tm = d["identity"], d["hyperparameters"], d["results"], d["timing"]
            nar = d.get("existing_narrative") or {}
            lines = ["%s   form: %s   [machine = cockpit, agent = written by the agent]" % (rid, d["form_status"].upper()),
                     "[machine] %s %s (%s) seed %s | lr %s | %s tokens | status %s, exit %s | %s min" % (
                         i["experiment"], i.get("experiment_title") or "", i.get("arc") or "-", i["seed"], h["lr_peak"], h["tokens_trained"], tm["status"], tm["exit_code"],
                         tm["wall_min"] and round(tm["wall_min"], 1)),
                     "[machine] val loss %s | bits/char %s | train loss last %s | best periodic val %s" % (r["val_loss"], r["val_bits_per_char"], r["train_loss_last"], r["best_val_loss"]),
                     "[machine] DATA CAVEAT: " + d["data_caveat"]["text"],
                     "[machine] charts: %s | skipped: %s" % (", ".join(d["charts"]["available"]) or "none", "; ".join(d["charts"]["skipped"]) or "none"),
                     "[machine] related claims: " + ", ".join("%s %s" % (c["id"], c["status"][:14]) for c in d["related_claims"][:8]), ""]
            for k in runform.NARRATIVE:
                lines.append("[agent] %s: %s" % (k, nar.get(k) or "(not written yet)"))
            log.write("\n".join(lines))
        self.bg("run_form_view", runform.run_form_draft, show, rid)

    def apply_runs(self, d: dict) -> None:
        t = self.query_one("#runs-table", DataTable)
        t.clear()
        for r in d.get("runs", []):
            t.add_row(r["id"][:28], r["experiment"] or "", str(r["seed"]), str(r["lr"] or ""), r["status"] or "",
                      "%.4f" % r["val_loss"] if r["val_loss"] else "-", "%.4f" % r["val_bpc"] if r["val_bpc"] else "-",
                      "%d" % r["tokens_per_s"] if r["tokens_per_s"] else "-", (r["start"] or "")[:19])

    # ------------------------------------------------------------------ search
    @on(Input.Submitted, "#search-input")
    def do_search(self, ev: Input.Submitted) -> None:
        def done(d):
            t = self.query_one("#search-table", DataTable)
            t.clear()
            for h in d.get("hits", []):
                t.add_row("%s:%s" % (h["file"], h["line"]), h["text"][:150])
            if d.get("error"):
                self.notify(d["error"], severity="warning")
        self.bg("search", core.search, done, ev.value)

    def action_tab_search(self) -> None:
        self.action_tab("search")
        self.query_one("#search-input", Input).focus()


def main() -> int:
    CockpitApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
