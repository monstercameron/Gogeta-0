"""Gogeta-0 cockpit SQLite log store (stdlib only).

One DB for runs, events, errors (full tracebacks), campaigns, servers and tool calls.
  Location : $GOGETA_COCKPIT_DB or $GOGETA_DATA/cockpit/gogeta.sqlite  (WSL filesystem; data, never committed)
  Safety   : every write redacts tokens first (core.redact); every writer swallows its own failures, so
             logging can never crash a launcher, a training run or the cockpit.
  Ingest   : ingest() is incremental (file mtime/size/offset in ingest_state) and idempotent.

CLI (used by the launcher patches in notes/patches/):
  python -m cockpit.logdb event --source launcher --level error --kind preflight --message "..." [--run-id R] [--data '{}']
  python -m cockpit.logdb error --source launcher --message "..." [--exc-type T] [--traceback TXT] [--run-id R]
  python -m cockpit.logdb ingest | stats | init
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sqlite3
import sys
import time
import traceback as _tb
from pathlib import Path

try:  # package import (python -m cockpit.logdb) or script import (cockpit/ on sys.path)
    from . import core
except ImportError:  # pragma: no cover
    import core  # type: ignore

SCHEMA_VERSION = 4
HOST = socket.gethostname()


def db_path() -> Path:
    p = os.environ.get("GOGETA_COCKPIT_DB")
    return Path(p) if p else core.DATA / "cockpit" / "gogeta.sqlite"


def now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


MIGRATIONS = {
    1: """
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, experiment TEXT, seed INTEGER, config TEXT, lr REAL, max_tokens INTEGER,
  git_describe TEXT, git_dirty INTEGER, start TEXT, end TEXT, status TEXT, exit_code INTEGER,
  val_loss REAL, val_bpc REAL, tokens_per_s REAL, flops REAL, log_path TEXT, summary_path TEXT,
  campaign TEXT, host TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, source TEXT NOT NULL, level TEXT NOT NULL, run_id TEXT,
  kind TEXT, message TEXT, data TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_events ON events(ts, source, kind, message);
CREATE INDEX IF NOT EXISTS ix_events_run ON events(run_id);
CREATE TABLE IF NOT EXISTS errors(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, source TEXT NOT NULL, run_id TEXT, exc_type TEXT, message TEXT,
  traceback TEXT, context TEXT, resolved INTEGER NOT NULL DEFAULT 0, resolved_ts TEXT, resolution_note TEXT,
  fingerprint TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS ix_errors_open ON errors(resolved, ts);
CREATE TABLE IF NOT EXISTS campaigns(
  name TEXT PRIMARY KEY, state TEXT, pid INTEGER, started TEXT, git TEXT, total_items INTEGER,
  done_items INTEGER, failed_items INTEGER, log_path TEXT, exit_code TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS campaign_items(
  campaign TEXT, idx INTEGER, args TEXT, experiment TEXT, seed TEXT, lr TEXT, rc INTEGER, start TEXT, end TEXT,
  val_loss REAL, val_bpc REAL, wall_min REAL, PRIMARY KEY(campaign, idx));
CREATE TABLE IF NOT EXISTS servers(
  id INTEGER PRIMARY KEY, ts TEXT, name TEXT, action TEXT, port INTEGER, pid INTEGER, ok INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS tool_calls(
  id INTEGER PRIMARY KEY, ts TEXT, interface TEXT, tool TEXT, args TEXT, duration_ms REAL, ok INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS ingest_state(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, offset INTEGER);
""",
    2: """
CREATE INDEX IF NOT EXISTS ix_runs_exp ON runs(experiment, start);
CREATE INDEX IF NOT EXISTS ix_tool_calls_ts ON tool_calls(ts);
""",
    3: """
CREATE TABLE IF NOT EXISTS incidents(
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, severity TEXT, subject_type TEXT, subject TEXT,
  campaign TEXT, category TEXT, status TEXT NOT NULL DEFAULT 'open', summary TEXT, bundle TEXT, actions TEXT,
  notes TEXT, resume_file TEXT, agent_brief TEXT, updated TEXT, fingerprint TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS ix_incidents_status ON incidents(status, ts);
""",
    4: """
CREATE TABLE IF NOT EXISTS cloud_instances(
  id TEXT PRIMARY KEY, label TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'vast', approval_id TEXT, session_id TEXT, offer_id TEXT, machine_id TEXT, gpu_name TEXT,
  num_gpus INTEGER, dph REAL, max_hours REAL, reserved_usd REAL, purpose TEXT, image TEXT, disk_gb INTEGER, created TEXT,
  created_epoch REAL, state TEXT NOT NULL, state_since REAL, mode TEXT NOT NULL DEFAULT 'manual', driver TEXT, job_id TEXT,
  live_status TEXT, ssh_host TEXT, ssh_port INTEGER, meta TEXT, setup_state TEXT, setup_commit TEXT, setup_s REAL, ready_epoch REAL,
  remote_env TEXT, data_ok TEXT, push_dataset TEXT, push_state TEXT, last_poll_epoch REAL, last_poll_ok_epoch REAL,
  last_heartbeat_epoch REAL, last_busy_epoch REAL, poll TEXT, jobs_done INTEGER DEFAULT 0, destroyed_epoch REAL,
  destroy_reason TEXT, cost_usd REAL);
CREATE INDEX IF NOT EXISTS ix_cloud_instances_state ON cloud_instances(state);
CREATE TABLE IF NOT EXISTS cloud_sessions(
  id TEXT PRIMARY KEY, approval_id TEXT, gpu_name TEXT, image TEXT, max_instances INTEGER, max_hours REAL, disk_gb INTEGER,
  min_cuda REAL, commit_sha TEXT, purpose TEXT, created TEXT, status TEXT, exclude_machines TEXT, provider TEXT DEFAULT 'vast', max_dph REAL);
CREATE TABLE IF NOT EXISTS cloud_jobs(
  id TEXT PRIMARY KEY, session_id TEXT, launch TEXT, experiment TEXT, dataset TEXT, status TEXT, instance_id TEXT, attempts INTEGER,
  setup_retries INTEGER, created TEXT, queued_epoch REAL, started_epoch REAL, ended_epoch REAL, remote_run TEXT, exit_code INTEGER,
  result_dir TEXT, verified INTEGER, checkpoint INTEGER, checkpoint_ok INTEGER, error TEXT, setup_usd REAL, run_usd REAL,
  teardown_usd REAL, manifest_sha256 TEXT, metrics TEXT, after_job TEXT);
CREATE INDEX IF NOT EXISTS ix_cloud_jobs_status ON cloud_jobs(status);
""",
}


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=8000")
    con.execute("PRAGMA synchronous=NORMAL")
    migrate(con)
    return con


def migrate(con: sqlite3.Connection) -> int:
    """Idempotent: applies only versions above the recorded one."""
    con.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL, applied TEXT)")
    cur = con.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
    for v in sorted(MIGRATIONS):
        if v > cur:
            con.executescript(MIGRATIONS[v])
            con.execute("INSERT INTO schema_version(version, applied) VALUES(?, ?)", (v, now()))
            con.commit()
    return con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]


def _j(x) -> str | None:
    if x is None:
        return None
    return core.redact(x if isinstance(x, str) else json.dumps(x, default=str, ensure_ascii=False))


def _r(s):
    return core.redact(s) if isinstance(s, str) else s


def _safe(fn):
    """Logging must never raise into callers."""
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as ex:  # noqa: BLE001
            try:
                sys.stderr.write("cockpit.logdb: %s failed: %r\n" % (fn.__name__, ex))
            except Exception:  # noqa: BLE001
                pass
            return None
    w.__name__ = fn.__name__
    w.__doc__ = fn.__doc__
    return w


# --------------------------------------------------------------------------- writers
@_safe
def event(source: str, level: str, kind: str, message: str, run_id: str | None = None, data=None,
          ts: str | None = None, con=None) -> int | None:
    own = con is None
    con = con or connect()
    try:
        cur = con.execute("INSERT OR IGNORE INTO events(ts,source,level,run_id,kind,message,data) VALUES(?,?,?,?,?,?,?)",
                          (ts or now(), source, level, run_id, kind, _r(message), _j(data)))
        con.commit()
        return cur.lastrowid
    finally:
        if own:
            con.close()


def _fp(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8", "replace")).hexdigest()


@_safe
def error(source: str, message: str, exc_type: str | None = None, traceback: str | None = None,
          run_id: str | None = None, context=None, ts: str | None = None, con=None) -> int | None:
    own = con is None
    con = con or connect()
    try:
        msg, tb = _r(message), _r(traceback)
        cur = con.execute(
            "INSERT OR IGNORE INTO errors(ts,source,run_id,exc_type,message,traceback,context,fingerprint) VALUES(?,?,?,?,?,?,?,?)",
            (ts or now(), source, run_id, exc_type, msg, tb, _j(context), _fp(source, run_id, exc_type, msg, tb)))
        con.commit()
        return cur.lastrowid
    finally:
        if own:
            con.close()


@_safe
def error_from_exception(source: str, ex: BaseException, context=None, run_id=None) -> int | None:
    return error(source, str(ex), type(ex).__name__, "".join(_tb.format_exception(type(ex), ex, ex.__traceback__)),
                 run_id=run_id, context=context)


@_safe
def server_event(name: str, action: str, port=None, pid=None, ok=True, detail="") -> None:
    con = connect()
    try:
        con.execute("INSERT INTO servers(ts,name,action,port,pid,ok,detail) VALUES(?,?,?,?,?,?,?)",
                    (now(), name, action, port, pid, 1 if ok else 0, _r(str(detail))[:1000]))
        con.commit()
    finally:
        con.close()
    event("server", "info" if ok else "error", "server_" + action, "%s %s port %s" % (action, name, port),
          data={"pid": pid, "ok": ok})


@_safe
def tool_call(interface: str, tool: str, args, duration_ms: float, ok: bool, err: str | None = None) -> None:
    con = connect()
    try:
        con.execute("INSERT INTO tool_calls(ts,interface,tool,args,duration_ms,ok,error) VALUES(?,?,?,?,?,?,?)",
                    (now(), interface, tool, (_j(args) or "")[:2000], round(duration_ms, 1), 1 if ok else 0, _r(err)))
        con.commit()
    finally:
        con.close()


@_safe
def upsert_run(con, **f) -> None:
    cols = ["id", "experiment", "seed", "config", "lr", "max_tokens", "git_describe", "git_dirty", "start", "end",
            "status", "exit_code", "val_loss", "val_bpc", "tokens_per_s", "flops", "log_path", "summary_path",
            "campaign", "host"]
    f.setdefault("host", HOST)
    have = [c for c in cols if c in f and f[c] is not None]
    sets = ",".join("%s=excluded.%s" % (c, c) for c in have if c != "id")
    con.execute("INSERT INTO runs(%s) VALUES(%s) ON CONFLICT(id) DO UPDATE SET %s" % (
        ",".join(have), ",".join("?" * len(have)), sets), [_r(f[c]) if isinstance(f[c], str) else f[c] for c in have])


# --------------------------------------------------------------------------- readers
def _rows(con, sql, args=()):
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def errors(since: str = "", source: str = "", level: str = "", run: str = "", unresolved: bool = False,
           limit: int = 30, full: bool = False) -> dict:
    con = connect()
    try:
        w, a = [], []
        if since:
            w.append("ts >= ?"); a.append(_since(since))
        if source:
            w.append("source = ?"); a.append(source)
        if run:
            w.append("run_id LIKE ?"); a.append(run + "%")
        if unresolved:
            w.append("resolved = 0")
        sql = "SELECT id,ts,source,run_id,exc_type,message,%s AS traceback,context,resolved,resolution_note FROM errors" % (
            "traceback" if full else "substr(traceback,1,600)")
        if w:
            sql += " WHERE " + " AND ".join(w)
        sql += " ORDER BY resolved ASC, ts DESC LIMIT ?"
        rows = _rows(con, sql, (*a, limit))
        total_open = con.execute("SELECT COUNT(*) FROM errors WHERE resolved=0").fetchone()[0]
        return {"unresolved_total": total_open, "count": len(rows), "errors": rows}
    finally:
        con.close()


def _since(s: str) -> str:
    """'2h', '30m', '1d' or an ISO prefix."""
    m = re.fullmatch(r"(\d+)\s*([smhd])", s.strip())
    if m:
        secs = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - secs))
    return s


def events(since: str = "", source: str = "", level: str = "", run: str = "", kind: str = "", limit: int = 50) -> dict:
    con = connect()
    try:
        w, a = [], []
        for col, val, op in (("ts", _since(since) if since else "", ">="), ("source", source, "="), ("level", level, "="),
                             ("kind", kind, "="), ("run_id", run + "%" if run else "", "LIKE")):
            if val:
                w.append("%s %s ?" % (col, op)); a.append(val)
        sql = "SELECT id,ts,source,level,run_id,kind,message,data FROM events"
        if w:
            sql += " WHERE " + " AND ".join(w)
        rows = _rows(con, sql + " ORDER BY ts DESC, id DESC LIMIT ?", (*a, limit))
        return {"count": len(rows), "events": rows}
    finally:
        con.close()


def run_history(experiment: str = "", status: str = "", campaign: str = "", since: str = "", limit: int = 30,
                order: str = "start") -> dict:
    con = connect()
    try:
        w, a = [], []
        if experiment:
            w.append("experiment = ?"); a.append(experiment.upper())
        if status:
            w.append("status = ?"); a.append(status)
        if campaign:
            w.append("campaign = ?"); a.append(campaign)
        if since:
            w.append("start >= ?"); a.append(_since(since))
        order = order if order in ("start", "val_loss", "tokens_per_s") else "start"
        sql = ("SELECT id,experiment,seed,lr,status,exit_code,val_loss,val_bpc,tokens_per_s,max_tokens,git_describe,start,end,"
               "campaign,log_path FROM runs")
        if w:
            sql += " WHERE " + " AND ".join(w)
        sql += " ORDER BY %s %s LIMIT ?" % (order, "ASC" if order == "val_loss" else "DESC")
        rows = _rows(con, sql, (*a, limit))
        return {"count": len(rows), "runs": rows}
    finally:
        con.close()


def resolve_error(error_id: int, note: str) -> dict:
    if not note or not note.strip():
        return {"resolved": False, "error": "a resolution note is required"}
    con = connect()
    try:
        cur = con.execute("UPDATE errors SET resolved=1, resolved_ts=?, resolution_note=? WHERE id=?",
                          (now(), _r(note.strip()), int(error_id)))
        con.commit()
        return {"resolved": cur.rowcount == 1, "id": int(error_id)}
    finally:
        con.close()


def unresolved_count() -> int:
    try:
        con = connect()
        try:
            return con.execute("SELECT COUNT(*) FROM errors WHERE resolved=0").fetchone()[0]
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return -1


def last_errors(n: int = 3) -> list:
    try:
        return errors(unresolved=True, limit=n)["errors"]
    except Exception:  # noqa: BLE001
        return []


def sql(query: str, limit: int = 200) -> dict:
    """Read-only SELECT. Enforced three ways: statement check, read-only connection, SQLite authorizer."""
    q = query.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", q) or ";" in q:
        return {"error": "only a single SELECT (or WITH ... SELECT) statement is allowed"}
    p = db_path()
    connect().close()  # make sure the file/schema exist
    con = sqlite3.connect("file:%s?mode=ro" % p, uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def auth(action, *_):
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    con.set_authorizer(auth)
    try:
        cur = con.execute(q)
        rows = cur.fetchmany(limit + 1)
        cols = [d[0] for d in cur.description or []]
        data = [[core.redact(x) if isinstance(x, str) else x for x in r] for r in rows[:limit]]
        return {"columns": cols, "rows": data, "truncated": len(rows) > limit}
    except sqlite3.Error as ex:
        return {"error": "%s: %s" % (type(ex).__name__, ex)}
    finally:
        con.close()


def stats() -> dict:
    con = connect()
    try:
        out = {t: con.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0] for t in
               ("runs", "events", "errors", "campaigns", "campaign_items", "servers", "tool_calls", "ingest_state")}
        out["errors_unresolved"] = con.execute("SELECT COUNT(*) FROM errors WHERE resolved=0").fetchone()[0]
        out["schema_version"] = con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        out["db"] = str(db_path())
        return out
    finally:
        con.close()


# --------------------------------------------------------------------------- scanning text for problems
_PATTERNS = [
    ("oom", re.compile(r"(?i)out of memory|OutOfMemoryError|CUDA error: .*memory")),
    ("nan", re.compile(r"(?i)(loss|grad\w*)\W{0,6}(is\s+)?(nan|inf)\b|non-finite|\bNaN loss")),
    ("preflight", re.compile(r"(?i)\bREFUSED\b|preflight.*(fail|refus)|^\S*launch\w*:.*(refus|error)|\bFAIL(ED)?\b.*(?:preflight|dirty|slot)")),
]
_TB_START = "Traceback (most recent call last):"


def scan_text(text: str):
    """Yield (kind, exc_type, message, traceback, line_no) problems found in log text."""
    lines = text.splitlines()
    i, n = 0, len(lines)
    consumed_to = 0
    while i < n:
        ln = lines[i]
        if ln.startswith(_TB_START) or ln.strip().startswith(_TB_START):
            j = i + 1
            while j < n and (lines[j].startswith((" ", "\t")) or lines[j].startswith(("During handling", "The above exception", "Traceback"))
                             or not lines[j].strip()):
                j += 1
            if j >= n:  # exception line not yet written: leave for the next pass
                return
            exc = lines[j].strip()
            m = re.match(r"([\w.]+(?:Error|Exception|Exit|Interrupt|Warning)?)\s*(?::\s*(.*))?$", exc)
            yield ("traceback", m.group(1) if m else exc.split(":")[0][:80], exc[:500], "\n".join(lines[i:j + 1]), i + 1)
            i = j + 1
            consumed_to = i
            continue
        for kind, pat in _PATTERNS:
            if pat.search(ln):
                yield (kind, None, ln.strip()[:500], None, i + 1)
                break
        i += 1
        consumed_to = i


def _scan_complete_prefix(text: str):
    """Scan only up to an unterminated traceback; return (problems, consumed_bytes_chars)."""
    probs = list(scan_text(text))
    idx = text.rfind(_TB_START)
    if idx >= 0:
        tail = text[idx:]
        if not any(p[0] == "traceback" and p[3] and p[3].startswith(tail[:60]) for p in probs[-3:]):
            return probs, idx
    cut = text.rfind("\n") + 1
    return probs, cut


# --------------------------------------------------------------------------- ingestion
def _state(con, path: Path):
    st = path.stat()
    row = con.execute("SELECT mtime,size,offset FROM ingest_state WHERE path=?", (str(path),)).fetchone()
    return st, row


def _save_state(con, path, st, offset):
    con.execute("INSERT INTO ingest_state(path,mtime,size,offset) VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "mtime=excluded.mtime,size=excluded.size,offset=excluded.offset", (str(path), st.st_mtime, st.st_size, offset))


def _ingest_log(con, path: Path, source: str, run_id: str | None, counts: dict, events_from_lines=None) -> None:
    try:
        st, row = _state(con, path)
    except OSError:
        return
    off = row["offset"] if row else 0
    if row and row["mtime"] == st.st_mtime and row["size"] == st.st_size:
        return
    if st.st_size < off:  # truncated / rotated
        off = 0
    with open(path, "rb") as fh:
        fh.seek(off)
        raw = fh.read(4_000_000)
    text = raw.decode("utf-8", "replace").replace("\r", "\n")
    probs, used = _scan_complete_prefix(text)
    used_bytes = len(text[:used].encode("utf-8", "replace"))
    for kind, exc, msg, tb, ln in probs:
        error(source, msg, exc or kind, tb, run_id=run_id,
              context={"log": str(path), "kind": kind, "line_in_chunk": ln, "offset": off}, con=con)
        counts["errors_seen"] = counts.get("errors_seen", 0) + 1
    if events_from_lines:
        events_from_lines(con, text[:used], counts)
    _save_state(con, path, st, off + used_bytes if len(raw) < 4_000_000 else off + used_bytes)
    con.commit()


_CAMP_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)\]\s+(.*)$")


def _campaign_events(campaign: str):
    def f(con, text, counts):
        for ln in text.splitlines():
            m = _CAMP_LINE.match(ln)
            if not m:
                continue
            ts, msg = "%sT%s.000" % (m.group(1), m.group(2)), m.group(3)
            low = msg.lower()
            kind = ("failed" if "failed with exit" in low else "aborted" if ("abort" in low and "persistent problem" in low or "--stop-on-fail" in low) else "done" if " done:" in low else
                    "launch" if " launch:" in low else "wait" if low.startswith(("waiting", "gpu busy")) else "note")
            level = "error" if kind in ("failed", "aborted") else "info"
            if kind == "wait":
                continue
            event("campaign", level, "campaign_" + kind, "[%s] %s" % (campaign, msg[:400]), ts=ts, con=con,
                  data={"campaign": campaign})
            if kind in ("failed", "aborted"):
                error("campaign", "[%s] %s" % (campaign, msg[:400]), "campaign_" + kind, None, ts=ts, con=con,
                      context={"campaign": campaign})
            counts["campaign_events"] = counts.get("campaign_events", 0) + 1
    return f


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ingest_summaries(con, counts):
    d = core.RUNS_DIR
    if not d.exists():
        return
    for sd in sorted(d.iterdir()):
        sj = sd / "summary.json"
        try:
            st, row = _state(con, sj)
        except OSError:
            continue
        if row and row["mtime"] == st.st_mtime and row["size"] == st.st_size:
            continue
        try:
            s = json.loads(sj.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        m, cfg = s.get("metrics", {}) or {}, s.get("config", {}) or {}
        tr = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        g = s.get("git", {}) or {}
        run_id = _match_launcher_run(con, s) or s.get("name", sd.name)
        upsert_run(con, id=run_id, experiment=s.get("experiment"), seed=s.get("seed"),
                   config=json.dumps({"config_file": cfg.get("config_file"), "cli_overrides": cfg.get("cli_overrides"),
                                      "train": tr}, default=str) if isinstance(cfg, dict) else None,
                   lr=_fnum(tr.get("lr")), max_tokens=tr.get("max_tokens"), git_describe=g.get("describe"),
                   git_dirty=int(bool(g.get("dirty"))), start=s.get("started"), end=s.get("finished"),
                   status=s.get("status"), val_loss=_fnum(m.get("val_loss")), val_bpc=_fnum(m.get("val_bits_per_char")),
                   tokens_per_s=_fnum(m.get("tokens_per_s")), flops=_fnum(m.get("flops")), summary_path=str(sj))
        _save_state(con, sj, st, 0)
        counts["summaries"] = counts.get("summaries", 0) + 1
    con.commit()


def _match_launcher_run(con, s) -> str | None:
    """A summary.json folder name differs from the launcher's run name; match on experiment + seed + start time."""
    try:
        t0 = time.mktime(time.strptime(s.get("started", "")[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None
    for r in con.execute("SELECT id,start FROM runs WHERE experiment=? AND seed=? AND summary_path IS NULL", (s.get("experiment"), s.get("seed"))):
        try:
            t1 = time.mktime(time.strptime((r["start"] or "")[:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError):
            continue
        if abs(t1 - t0) <= 180:
            return r["id"]
    return None


def _ingest_train_pids(con, counts):
    for f in sorted(core.RUN_DIR.glob("train-*.pid")) if core.RUN_DIR.exists() else []:
        base = f.with_suffix("")
        st = f.stat()
        ex = base.with_suffix(".exit")
        stamp = (st.st_mtime + (ex.stat().st_mtime if ex.exists() else 0))
        row = con.execute("SELECT mtime FROM ingest_state WHERE path=?", (str(f),)).fetchone()
        alive = core._alive_with(core._pid(f), "gogeta-gpu-slot")
        if row and row["mtime"] == stamp and not alive:
            continue
        name = base.name[len("train-"):]
        info = core._kv(base.with_suffix(".info"))
        cmd = info.get("cmd", "")
        m_lr = re.search(r"--lr\s+(\S+)", cmd)
        m_seed = re.search(r"--seed\s+(\d+)", cmd)
        mm = re.match(r"(E\d+)-s(\d+)-", name)
        status = "running" if alive else ("done" if ex.exists() and core._read(ex).strip() == "0" else "failed" if ex.exists() else "killed")
        code = int(core._read(ex).strip()) if ex.exists() and core._read(ex).strip().lstrip("-").isdigit() else None
        row_exists = con.execute("SELECT 1 FROM runs WHERE id=?", (name,)).fetchone()
        fields = dict(id=name, experiment=info.get("experiment") or (mm.group(1) if mm else None),
                      seed=int(info["seed"]) if info.get("seed", "").isdigit() else (int(m_seed.group(1)) if m_seed else None),
                      config=info.get("config"), lr=_fnum(m_lr.group(1)) if m_lr else None,
                      start=info.get("started"), log_path=info.get("log"), exit_code=code)
        if status != "running" or not row_exists:
            fields["status"] = status
        upsert_run(con, **fields)
        if ex.exists() and code not in (0, None):
            error("launcher", "training run %s exited with code %s" % (name, code), "nonzero_exit", None, run_id=name,
                  context={"log": info.get("log")}, con=con)
        con.execute("INSERT INTO ingest_state(path,mtime,size,offset) VALUES(?,?,?,0) ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime",
                    (str(f), stamp, st.st_size))
        counts["train_pids"] = counts.get("train_pids", 0) + 1
    con.commit()


def _ingest_campaigns(con, counts):
    for c in core.campaigns():
        con.execute("INSERT INTO campaigns(name,state,pid,started,git,total_items,done_items,failed_items,log_path,exit_code,updated) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET state=excluded.state,pid=excluded.pid,"
                    "started=excluded.started,git=excluded.git,total_items=excluded.total_items,done_items=excluded.done_items,"
                    "failed_items=excluded.failed_items,log_path=excluded.log_path,exit_code=excluded.exit_code,updated=excluded.updated",
                    (c["name"], c["state"], c["pid"], c["started"], c["git"], c["total"], c["done"], c["failed"], c["log"],
                     c["exit"], now()))
        for it in c["items"]:
            con.execute("INSERT INTO campaign_items(campaign,idx,args,experiment,seed,lr,rc,val_loss,val_bpc,wall_min) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(campaign,idx) DO UPDATE SET args=excluded.args,rc=excluded.rc,"
                        "val_loss=excluded.val_loss,val_bpc=excluded.val_bpc,wall_min=excluded.wall_min",
                        (c["name"], it["idx"], it["args"], it.get("experiment"), it.get("seed"), it.get("lr"), it["rc"],
                         it.get("val_loss"), it.get("bpc"), it.get("wall_min")))
        if c["log"] and Path(c["log"]).exists():
            _ingest_log(con, Path(c["log"]), "campaign", None, counts, _campaign_events(c["name"]))
        counts["campaigns"] = counts.get("campaigns", 0) + 1
    # item launcher outputs: preflight refusals etc.
    for f in sorted(core.LOG_DIR.glob("campaign-*-*.out")) if core.LOG_DIR.exists() else []:
        _ingest_log(con, f, "launcher", None, counts)
    con.commit()


def _ingest_cpu(con, counts):
    d = core.CPU_DIR
    for ex in sorted(d.glob("*.exit")) if d.exists() else []:
        name = ex.stem
        st = ex.stat()
        row = con.execute("SELECT 1 FROM ingest_state WHERE path=?", (str(ex),)).fetchone()
        code = core._read(ex).strip()
        if not row:
            info = core._kv(ex.with_suffix(".info"))
            event("cpu-job", "info" if code == "0" else "error", "cpu_job_end", "cpu job %s exited %s" % (name, code),
                  ts=time.strftime("%Y-%m-%dT%H:%M:%S.000", time.localtime(st.st_mtime)), con=con,
                  data={"cmd": info.get("cmd", "")[:300], "log": info.get("log")})
            if code not in ("0", ""):
                error("cpu-job", "cpu job %s exited with code %s" % (name, code), "nonzero_exit", None, context=info, con=con)
            _save_state(con, ex, st, 0)
            counts["cpu_jobs"] = counts.get("cpu_jobs", 0) + 1
        lg = ex.with_suffix(".log")
        if lg.exists():
            _ingest_log(con, lg, "cpu-job", None, counts)
    con.commit()


def _ingest_train_logs(con, counts):
    if not core.LOG_DIR.exists():
        return
    run_by_log = {r["log_path"]: r["id"] for r in con.execute("SELECT id,log_path FROM runs WHERE log_path IS NOT NULL")}
    for f in sorted(core.LOG_DIR.glob("E*-s*-*.log")):
        _ingest_log(con, f, "train", run_by_log.get(str(f)), counts)
    for f in sorted(core.LOG_DIR.glob("e*.log")):
        pass  # legacy hand-launched logs are not tailed
    con.commit()


def ingest(max_seconds: float = 20.0) -> dict:
    """Incremental, idempotent ingest of run folders, PID files, campaign state/logs, CPU-job logs and training logs."""
    counts: dict = {}
    t0 = time.time()
    try:
        con = connect()
    except Exception as ex:  # noqa: BLE001
        return {"error": "db unavailable: %r" % ex}
    try:
        for step in (_ingest_train_pids, _ingest_summaries, _ingest_campaigns, _ingest_cpu, _ingest_train_logs):
            if time.time() - t0 > max_seconds:
                counts["timeout_before"] = step.__name__
                break
            try:
                step(con, counts)
            except Exception as ex:  # noqa: BLE001
                error_from_exception("cockpit", ex, context={"step": step.__name__})
                counts.setdefault("step_errors", []).append("%s: %r" % (step.__name__, ex))
        # link runs to campaigns by args
        con.execute("UPDATE runs SET campaign=(SELECT ci.campaign FROM campaign_items ci WHERE ci.experiment=runs.experiment "
                    "AND CAST(ci.seed AS INTEGER)=runs.seed AND ci.val_loss IS runs.val_loss LIMIT 1) WHERE campaign IS NULL AND val_loss IS NOT NULL")
        con.commit()
    finally:
        con.close()
    counts["seconds"] = round(time.time() - t0, 2)
    return counts


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cockpit.logdb")
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("event")
    e.add_argument("--source", required=True); e.add_argument("--level", default="info"); e.add_argument("--kind", default="note")
    e.add_argument("--message", required=True); e.add_argument("--run-id"); e.add_argument("--data")
    r = sp.add_parser("error")
    r.add_argument("--source", required=True); r.add_argument("--message", required=True); r.add_argument("--exc-type")
    r.add_argument("--traceback"); r.add_argument("--run-id"); r.add_argument("--context")
    for n in ("ingest", "stats", "init"):
        sp.add_parser(n)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "event":
            event(a.source, a.level, a.kind, a.message, a.run_id, _loads(a.data))
        elif a.cmd == "error":
            error(a.source, a.message, a.exc_type, a.traceback, a.run_id, _loads(a.context))
        elif a.cmd == "ingest":
            print(json.dumps(ingest()))
        elif a.cmd == "stats":
            print(json.dumps(stats(), indent=1))
        else:
            connect().close()
    except Exception as ex:  # noqa: BLE001  logging must never fail a caller
        sys.stderr.write("cockpit.logdb: %r\n" % ex)
    return 0


def _loads(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except ValueError:
        return {"raw": s}


if __name__ == "__main__":
    sys.exit(main())
