#!/usr/bin/env python3
"""Gogeta-0 watchdog: detect failures, record incidents with context bundles, do SAFE recovery only. Stdlib only.

  python3 cockpit/watchdog.py --start | --stop | --status | --once | --run [--interval 10] [--hung-minutes 15]

Detects (polling existing PID files, campaign state/logs and status files; never touches code):
  run_failed / run_died / run_stopped, run_hung, log_traceback / log_nan / log_oom / log_triton,
  campaign_aborted (incl. preflight REFUSED) with a resume campaign file, cpu_job_failed,
  server_down (tracker 7861, viz 7870, site 8791), gpu_hot (>= 85 C), disk_low,
  cloud (cockpit/fleet.py: cloud_destroy_* / cloud_idle_suggest; remote_traceback/nan/oom/hung/dead on rented GPUs).
Each detection becomes: a row in SQLite `incidents`, a JSON file in ~/fusiondance-data/cockpit/incidents/,
and one line appended to ~/fusiondance-data/cockpit/incidents.log (stable, greppable).
Automatic actions are limited to containment/retry: restart a down server (max 3/h, backoff), write a resume campaign
file holding exactly the unfinished items, launch it automatically ONLY for transient/thermal causes (never code bug or
dirty tree), pause auto-resume while hot. It never edits code, deletes data, resets git or kills a healthy process.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback as _tb
from pathlib import Path

try:
    from . import core, logdb
except ImportError:  # script mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import core  # type: ignore
    import logdb  # type: ignore

STATUSES = ["open", "triaged", "fixing", "fixed", "verified"]
CATEGORIES = ["code-bug", "env", "transient", "thermal", "data", "dirty-tree", "resource", "operator", "unknown"]
CAMPAIGN_DIR = core.REPO / "experiments" / "launch" / "campaigns"
DEFAULTS = {"interval": 10.0, "lookback_h": 3.0, "hung_minutes": 15.0, "disk_min_gb": 20.0, "resume_cool_c": 75.0}
MAX_AUTO_RESTARTS_PER_H = 3


def cdir() -> Path:
    p = Path(os.environ.get("GOGETA_COCKPIT_DIR") or core.DATA / "cockpit")
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_s() -> float:
    return time.time()


def _iso(t=None) -> str:
    return logdb.now() if t is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


# --------------------------------------------------------------------------- classification
_CLASS = [
    ("dirty-tree", re.compile(r"(?i)(dirty|uncommitted|untracked)[^\n]{0,80}(path|tree|code|tools|gogeta)|code paths? (are )?dirty|--require-clean|working tree")),
    ("thermal", re.compile(r"(?i)\b(8[5-9]|9\d) ?C\b|thermal|too hot|\bHOT\b|cool-?down")),
    ("transient", re.compile(r"(?i)triton[^\n]{0,120}(cache|autotun|compil)|autotun|inductor[^\n]{0,60}(cache|worker)|connection reset|temporarily unavailable|broken pipe|wsl|unspecified launch failure|stale file handle|Resource temporarily")),
    ("resource", re.compile(r"(?i)out of memory|OutOfMemoryError|No space left|gpu slot|slot is held|slots? (alive|full)|GPU is busy|compute process")),
    ("data", re.compile(r"(?i)(FileNotFoundError|No such file)[^\n]{0,160}(dataset|shard|meta\.json|\.bin|tokenizer)|tokenized|corrupt")),
    ("env", re.compile(r"(?i)ModuleNotFoundError|ImportError|libcuda|CUDA driver|torch\.cuda.*not available|pip install|No module named")),
    ("code-bug", re.compile(r"Traceback \(most recent call last\)|\b(AttributeError|TypeError|ValueError|KeyError|IndexError|AssertionError|RuntimeError|NameError|NotImplementedError)\b|\b(loss|grad\w*)\W{0,6}(is\s+)?(nan|inf)\b|non-finite")),
]
_SUGGEST = {
    "code-bug": "Reproduce on CPU or a tiny config (--max-tokens small), fix the root cause in the gap between campaigns, add a test, PATCH version + devlog.",
    "env": "Check the venv/driver/package named in the error; fix the environment (not the experiment); record versions.",
    "transient": "Safe to retry once after a cool-down; if it recurs it is not transient: treat as code-bug.",
    "thermal": "Wait for the GPU to drop below 75 C; campaigns resume by themselves. Check airflow/power if recurrent.",
    "data": "Verify dataset files and meta.json under ~/fusiondance-data/datasets; re-run tools/prepare_data.py only if truly missing.",
    "dirty-tree": "Commit or stash the stray files under gogeta/ configs/ tools/ tests/ requirements.txt (git status), then resume the campaign.",
    "resource": "Free the resource (GPU slot, CPU slots, disk, memory) or shrink the job (batch size); then resume.",
    "operator": "Stopped on purpose (SIGTERM/SIGINT). Nothing to fix; resume if the work is still wanted.",
    "unknown": "Read the evidence tail; classify manually.",
}


def classify(text: str, hint: str = "") -> dict:
    t = text or ""
    if hint:
        return {"category": hint, "reason": "fixed by detector", "suggestion": _SUGGEST[hint]}
    for cat, pat in _CLASS:
        m = pat.search(t)
        if m:
            return {"category": cat, "reason": "matched %r" % m.group(0)[:80], "suggestion": _SUGGEST[cat]}
    return {"category": "unknown", "reason": "no pattern matched", "suggestion": _SUGGEST["unknown"]}


# --------------------------------------------------------------------------- helpers
def _tail(path, nbytes=60_000, lines=60) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            sz = fh.tell()
            fh.seek(max(0, sz - nbytes))
            txt = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    txt = txt.replace("\r", "\n")
    return core.redact("\n".join([l for l in txt.splitlines() if l.strip()][-lines:]))


def _last_metrics(text: str) -> dict:
    m = {}
    for ln in reversed(text.splitlines()):
        if "step" in ln and "loss" in ln and "train_loss" not in m:
            mm = re.search(r"step\s+(\d+)/(\d+)\s+loss\s+(\S+)", ln)
            if mm:
                m.update(step=int(mm.group(1)), total=int(mm.group(2)), train_loss=mm.group(3))
        if "val/loss" in ln and "val_loss" not in m:
            mm = re.search(r"val/loss\s+(\S+)", ln)
            if mm:
                m["val_loss"] = mm.group(1)
        if len(m) >= 4:
            break
    return m


def _files_in(text: str) -> list:
    out = []
    for mm in re.finditer(r'File "([^"]+)"', text or ""):
        p = mm.group(1)
        for root in (str(core.REPO), "/mnt/c/Users/mreca/Desktop/Gogeta-0"):
            if p.startswith(root):
                p = p[len(root):].lstrip("/")
                break
        if "site-packages" not in p and p not in out:
            out.append(p)
    return out[-6:]


def _state_file() -> Path:
    return cdir() / "watchdog-state.json"


def _load_state() -> dict:
    try:
        return json.loads(_state_file().read_text())
    except (OSError, ValueError):
        return {}


def _save_state(st: dict) -> None:
    tmp = _state_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str))
    os.replace(tmp, _state_file())


# --------------------------------------------------------------------------- incidents store
def _fp(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()


def _new_id(con) -> str:
    n = con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] + 1
    return "INC-%s-%03d" % (time.strftime("%Y%m%d"), n)


def _write_json(inc: dict) -> None:
    d = cdir() / "incidents"
    d.mkdir(exist_ok=True)
    (d / (inc["id"] + ".json")).write_text(core.redact(json.dumps(inc, indent=1, default=str, ensure_ascii=False)))


def _log_line(inc_id, sev, kind, subject, category, status, summary) -> None:
    line = "%s %s sev=%s kind=%s subject=%s category=%s status=%s :: %s\n" % (
        _iso(), inc_id, sev, kind, subject or "-", category, status, core.redact(summary).replace("\n", " ")[:220])
    with open(cdir() / "incidents.log", "a", encoding="utf-8") as f:
        f.write(line)


def build_bundle(f: dict) -> dict:
    log = f.get("log")
    tail = _tail(log) if log else f.get("tail", "")
    tbs = [p[3] for p in logdb.scan_text(tail) if p[0] == "traceback" and p[3]] if tail else []
    return {
        "log_path": log, "log_tail": core.redact((tail or "")[-6000:]),
        "traceback": core.redact(tbs[-1]) if tbs else (core.redact(f.get("traceback")) if f.get("traceback") else None),
        "git": {"describe": core._git("describe", "--tags", "--always"), "dirty_code_paths": core.code_paths_dirty()[:10]},
        "gpu": core._gpu_hw(), "gpu_slot": core._gpu_slot(),
        "config": f.get("config"), "cli_args": f.get("cli_args"), "last_metrics": _last_metrics(tail or ""),
        "evidence": core.redact(json.dumps(f.get("evidence", {}), default=str))[:3000],
    }


def agent_brief(inc: dict) -> str:
    b, c = inc["bundle"], inc["classification"]
    files = _files_in((b.get("traceback") or "") + (b.get("log_tail") or ""))
    repro = inc.get("repro") or "bash experiments/launch/status.sh"
    lines = [
        "INCIDENT %s [%s] %s: %s" % (inc["id"], inc["severity"], inc["kind"], inc["summary"]),
        "WHAT FAILED: %s %s%s" % (inc["subject_type"], inc["subject"], (" in campaign " + inc["campaign"]) if inc.get("campaign") else ""),
        "EVIDENCE:\n" + core.redact(((b.get("traceback") or b.get("log_tail") or inc.get("evidence_text") or "")[-1800:])),
        "SUSPECTED CAUSE: %s (%s)" % (c["category"], c["reason"]),
        "SUGGESTED FIX: " + c["suggestion"],
        "FILES INVOLVED: %s" % (", ".join(files) or "(none identified; read the evidence)"),
        "REPRODUCE: %s" % repro,
        "GIT: %s, dirty code paths: %s" % (b["git"]["describe"], b["git"]["dirty_code_paths"] or "none"),
        "RESUME FILE: %s" % (inc.get("resume_file") or "none"),
        "FIXED MEANS: the repro passes (or the resumed campaign finishes); if a result's validity was affected, devlog + claims ledger say so "
        "(AGENTS 0f); PATCH version cut and pushed; then gogeta_incident_update status=verified with a note.",
        "RULES: edit code only between campaigns (dirty tree aborts them); never retry a deterministic failure more than once; "
        "use gogeta_incident_update (confirm) to move status open -> triaged -> fixing -> fixed -> verified.",
    ]
    return "\n".join(lines)


def record(f: dict, auto_actions: list | None = None) -> dict | None:
    """Create an incident from a finding (idempotent per fingerprint). Returns the incident or None if already known."""
    con = logdb.connect()
    try:
        fp = _fp(f["kind"], f["subject"], f.get("key", ""))
        if con.execute("SELECT 1 FROM incidents WHERE fingerprint=?", (fp,)).fetchone():
            return None
        text = (f.get("tail") or "") + "\n" + (_tail(f["log"]) if f.get("log") else "") + "\n" + f.get("summary", "")
        cls = classify(text, f.get("hint", ""))
        inc = {"id": _new_id(con), "ts": _iso(), "kind": f["kind"], "severity": f.get("severity", "high"),
               "subject_type": f.get("subject_type", "run"), "subject": f["subject"], "campaign": f.get("campaign"),
               "status": "open", "summary": core.redact(f["summary"])[:300], "classification": cls,
               "actions": list(auto_actions or []) + list(f.get("actions", [])), "notes": [], "resume_file": f.get("resume_file"),
               "repro": f.get("repro"), "evidence_text": core.redact(f.get("tail", ""))[-1500:]}
        inc["bundle"] = build_bundle(f)
        inc["agent_brief"] = agent_brief(inc)
        con.execute("INSERT INTO incidents(id,ts,kind,severity,subject_type,subject,campaign,category,status,summary,bundle,actions,notes,"
                    "resume_file,agent_brief,updated,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (inc["id"], inc["ts"], inc["kind"], inc["severity"], inc["subject_type"], inc["subject"], inc["campaign"],
                     cls["category"], "open", inc["summary"], core.redact(json.dumps(inc["bundle"], default=str)), core.redact(json.dumps(inc["actions"])),
                     "[]", inc["resume_file"], inc["agent_brief"], _iso(), fp))
        con.commit()
    finally:
        con.close()
    _write_json(inc)
    _log_line(inc["id"], inc["severity"], inc["kind"], inc["subject"], cls["category"], "open", inc["summary"])
    logdb.event("watchdog", "error" if inc["severity"] in ("high", "critical") else "warn", "incident_" + inc["kind"],
                "%s %s: %s" % (inc["id"], inc["subject"], inc["summary"]), run_id=inc["subject"] if inc["subject_type"] == "run" else None)
    return inc


def _row(r, full=False) -> dict:
    d = dict(r)
    for k in ("actions", "notes"):
        d[k] = json.loads(d[k] or "[]")
    d["bundle"] = json.loads(d["bundle"] or "{}") if full else None
    if not full:
        d.pop("bundle"), d.pop("agent_brief", None)
    return d


def incidents(status: str = "", open_only: bool = False, kind: str = "", category: str = "", limit: int = 30) -> dict:
    con = logdb.connect()
    try:
        w, a = [], []
        if open_only:
            w.append("status NOT IN ('verified','wontfix')")
        if status:
            w.append("status=?"); a.append(status)
        if kind:
            w.append("kind=?"); a.append(kind)
        if category:
            w.append("category=?"); a.append(category)
        sql = ("SELECT id,ts,kind,severity,subject_type,subject,campaign,category,status,summary,actions,notes,resume_file,updated FROM incidents"
               + (" WHERE " + " AND ".join(w) if w else "")
               + " ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'triaged' THEN 1 WHEN 'fixing' THEN 2 WHEN 'fixed' THEN 3 ELSE 4 END, ts DESC LIMIT ?")
        rows = [_row(r) for r in con.execute(sql, (*a, limit))]
        nopen = con.execute("SELECT COUNT(*) FROM incidents WHERE status NOT IN ('verified','wontfix')").fetchone()[0]
        return {"open_total": nopen, "count": len(rows), "incidents": rows}
    finally:
        con.close()


def open_count() -> int:
    try:
        return incidents(open_only=True, limit=1)["open_total"]
    except Exception:  # noqa: BLE001
        return -1


def incident(id: str) -> dict:
    con = logdb.connect()
    try:
        r = con.execute("SELECT * FROM incidents WHERE id=?", (id,)).fetchone()
        return _row(r, True) if r else {"error": "unknown incident %s" % id}
    finally:
        con.close()


def update_incident(id: str, status: str = "", note: str = "") -> dict:
    """Lifecycle: open -> triaged -> fixing -> fixed -> verified (forward only), 'open' reopens, 'wontfix' from anywhere not verified."""
    con = logdb.connect()
    try:
        r = con.execute("SELECT status,notes,kind,subject,severity,category,summary FROM incidents WHERE id=?", (id,)).fetchone()
        if not r:
            return {"error": "unknown incident %s" % id}
        cur = r["status"]
        if status:
            ok = (status in STATUSES and cur in STATUSES and (STATUSES.index(status) > STATUSES.index(cur) or status == "open")) or \
                 (status == "wontfix" and cur != "verified") or (status == "open" and cur in ("fixed", "wontfix"))
            if status in ("fixed", "verified") and not note.strip():
                return {"error": "a note is required when moving to %s" % status}
            if not ok:
                return {"error": "invalid transition %s -> %s" % (cur, status)}
        notes = json.loads(r["notes"] or "[]")
        if note.strip():
            notes.append({"ts": _iso(), "status": status or cur, "note": core.redact(note.strip())})
        new = status or cur
        con.execute("UPDATE incidents SET status=?, notes=?, updated=? WHERE id=?", (new, json.dumps(notes), _iso(), id))
        con.commit()
    finally:
        con.close()
    inc = incident(id)
    _write_json(inc)
    _log_line(id, r["severity"], r["kind"], r["subject"], r["category"], new, "status %s -> %s %s" % (cur, new, note[:100]))
    return {"id": id, "status": new, "previous": cur}


# --------------------------------------------------------------------------- detectors
def _recent(p: Path, ctx) -> bool:
    try:
        return ctx["now"] - p.stat().st_mtime < ctx["lookback_h"] * 3600
    except OSError:
        return False


def d_runs(ctx):
    out = []
    for f in sorted(core.RUN_DIR.glob("train-*.pid")) if core.RUN_DIR.exists() else []:
        base = f.with_suffix("")
        ex = base.with_suffix(".exit")
        if not (_recent(f, ctx) or _recent(ex, ctx)):
            continue
        name, pid, info = base.name[len("train-"):], core._pid(f), core._kv(base.with_suffix(".info"))
        log, cmd = info.get("log"), info.get("cmd", "").strip()
        m = re.match(r"(E\d+)-s(\d+)", name)
        repro = ("bash tools/launch_experiment.sh %s --seed %s --dry-run" % (m.group(1), m.group(2))) if m else None
        common = {"subject": name, "subject_type": "run", "log": log, "cli_args": cmd, "config": info.get("config"), "repro": repro,
                  "evidence": {"info": info}}
        if core._alive_with(pid, "gogeta-gpu-slot"):
            age_min = (ctx["now"] - Path(log).stat().st_mtime) / 60 if log and Path(log).exists() else 0
            if age_min > ctx["hung_minutes"]:
                out.append({**common, "kind": "run_hung", "key": "hung", "severity": "high", "hint": "unknown",
                            "summary": "run %s: no new log line for %.0f min (limit %.0f); process alive pid %s" % (name, age_min, ctx["hung_minutes"], pid)})
            continue
        if ex.exists():
            code = core._read(ex).strip()
            if code == "0":
                continue
            if code in ("130", "143"):
                out.append({**common, "kind": "run_stopped", "key": "stopped", "severity": "info", "hint": "operator",
                            "summary": "run %s stopped by signal (exit %s)" % (name, code)})
            else:
                out.append({**common, "kind": "run_failed", "key": "exit" + code, "severity": "high",
                            "summary": "run %s exited with code %s" % (name, code), "tail": _tail(log) if log else ""})
        else:
            out.append({**common, "kind": "run_died", "key": "died", "severity": "high",
                        "summary": "run %s: process gone, no exit file (killed, WSL restart or OOM-kill?)" % name,
                        "tail": _tail(log) if log else ""})
    return out


_TRITON = re.compile(r"(?i)triton[^\n]*(error|exception|failed)|CompilationError|autotun[^\n]*(error|fail)")


def d_logs(ctx):
    out = []
    if not core.LOG_DIR.exists():
        return out
    covered = set()
    for f in core.RUN_DIR.glob("train-*.info") if core.RUN_DIR.exists() else []:
        base = f.with_suffix("")
        ex = base.with_suffix(".exit")
        if ex.exists() and core._read(ex).strip() not in ("0", ""):
            covered.add(core._kv(f).get("log"))
    for lf in sorted(core.LOG_DIR.glob("E*-s*-*.log")):
        if str(lf) in covered or not _recent(lf, ctx):
            continue
        tail = _tail(lf, 200_000, 400)
        probs = [p for p in logdb.scan_text(tail) if p[0] in ("traceback", "nan", "oom")]
        probs += [("triton", None, m.group(0)[:200], None, 0) for m in _TRITON.finditer(tail)]
        seen = set()
        for kind, exc, msg, tb, _ in probs:
            key = kind + hashlib.md5(msg.encode()).hexdigest()[:8]
            if key in seen:
                continue
            seen.add(key)
            hint = {"nan": "code-bug", "oom": "resource", "triton": "transient"}.get(kind, "")
            out.append({"kind": "log_" + kind, "subject": lf.stem, "subject_type": "run", "key": key, "severity": "high" if kind != "triton" else "medium",
                        "hint": hint, "log": str(lf), "summary": "%s in %s: %s" % (kind, lf.name, msg[:160]), "traceback": tb,
                        "cli_args": None, "repro": "tail -n 80 %s" % lf})
    return out


def _snapshot_items(c) -> list:
    snap = core._kv(core.RUN_DIR / ("campaign-%s.info" % c["name"])).get("snapshot", "")
    return [l.rstrip("\n") for l in core._read(Path(snap)).splitlines() if l.strip() and not l.lstrip().startswith("#")] if snap else []


def unfinished_items(c) -> list:
    """Exactly the snapshot items that are not finished with exit 0 (failed and never-started both count)."""
    items = _snapshot_items(c)
    done = {(it["idx"], it["args"]) for it in c["items"] if it["rc"] == 0}
    return [it for i, it in enumerate(items, 1) if (i, it) not in done]


def _next_resume_path(name: str) -> Path:
    base = re.sub(r"-resume(-\d+)?$", "", name)
    n = 1
    while (CAMPAIGN_DIR / ("%s-resume-%d.campaign" % (base, n))).exists():
        n += 1
    return CAMPAIGN_DIR / ("%s-resume-%d.campaign" % (base, n))


def write_resume_file(c, items: list, cause: str) -> Path:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    p = _next_resume_path(c["name"])
    head = "# Resume of campaign %s (written by cockpit watchdog %s). Cause: %s\n# Exactly the %d unfinished item(s); finished ones are not repeated.\n" % (
        c["name"], _iso(), cause, len(items))
    p.write_text(head + "\n".join(items) + "\n", encoding="utf-8")
    return p


def _resume_exists(name: str) -> bool:
    base = re.sub(r"-resume(-\d+)?$", "", name)
    return any(CAMPAIGN_DIR.glob(base + "-resume*.campaign")) if CAMPAIGN_DIR.exists() else False


def d_campaigns(ctx):
    out = []
    for c in core.campaigns():
        if c["state"] != "ended" or not _recent(Path(c["log"] or "/nonexistent"), ctx):
            continue
        if _resume_exists(c["name"]) and "-resume" not in c["name"]:
            continue
        left = unfinished_items(c)
        if c["exit"] == "0" or not left:
            continue  # exit 0 = every item finished (items skipped by --resume count as done)
        text = core._tail_text(c["log"]) if hasattr(core, "_tail_text") else _tail(c["log"], 80_000, 80)
        outs = sorted(core.LOG_DIR.glob("campaign-%s-*-%d.out" % (c["name"], c["done"] + 1))) if core.LOG_DIR.exists() else []
        if outs:
            text += "\n" + _tail(outs[-1], 30_000, 40)
        # A runner ended by SIGTERM/SIGINT (exit 143/130) was stopped on purpose by the human or an agent: classify it
        # "operator" so it is never auto-resumed (2026-11-22: a human-ordered thermal kill was auto-resumed as "transient").
        stopped = {"hint": "operator"} if str(c["exit"]) in ("130", "143") else {}
        out.append({**stopped, "kind": "campaign_aborted", "subject": c["name"], "subject_type": "campaign", "campaign": c["name"],
                    "key": "%s-exit%s-%d" % (c["name"], c["exit"], c["done"]), "severity": "high", "log": c["log"], "tail": text,
                    "summary": "campaign %s ended (exit %s) with %d of %d items unfinished" % (c["name"], c["exit"], len(left), c["total"]),
                    "repro": "bash tools/launch_campaign.sh --status %s" % c["name"], "_left": left, "_c": c,
                    "evidence": {"exit": c["exit"], "done": c["done"], "failed": c["failed"], "unfinished": left}})
    return out


def d_cpu(ctx):
    out = []
    for ex in sorted(core.CPU_DIR.glob("*.exit")) if core.CPU_DIR.exists() else []:
        code = core._read(ex).strip()
        if code in ("0", "") or not _recent(ex, ctx):
            continue
        info = core._kv(ex.with_suffix(".info"))
        out.append({"kind": "cpu_job_failed", "subject": ex.stem, "subject_type": "cpu-job", "key": "exit" + code, "severity": "medium",
                    "log": str(ex.with_suffix(".log")), "summary": "cpu job %s exited %s" % (ex.stem, code), "cli_args": info.get("cmd"),
                    "repro": info.get("cmd"), "evidence": {"info": info}})
    return out


def _server_state(ctx):
    return ctx["state"].setdefault("servers", {})


def d_servers(ctx):
    out = []
    for name, s in core.SERVERS.items():
        st = _server_state(ctx).setdefault(name, {"seen_up": False, "down_since": None, "restarts": []})
        up = core.port_up(s["port"])
        if up:
            if st["down_since"]:
                st["recovered"] = st["down_since"]
            st.update(seen_up=True, down_since=None)
            continue
        if not st["seen_up"]:
            continue  # never seen up: nobody asked for it, do not start it
        if not st["down_since"]:
            st["down_since"] = ctx["now"]
        out.append({"kind": "server_down", "subject": name, "subject_type": "server", "key": "%s@%d" % (name, st["down_since"]),
                    "severity": "medium", "hint": "transient", "summary": "server %s (port %s) is down" % (name, s["port"]),
                    "repro": "bash cockpit/cockpit.sh then Dashboard > Start %s (or gogeta_start_server)" % name,
                    "evidence": {"port": s["port"]}, "_server": name})
    return out


def d_thermal(ctx):
    hw = ctx.get("gpu") if "gpu" in ctx else core._gpu_hw()
    t = hw.get("temp_c")
    flag = cdir() / "pause.flag"
    if t is None:
        return []
    if t >= core.HOT_C:
        ctx["state"]["hot_since"] = ctx["state"].get("hot_since") or ctx["now"]
        flag.write_text("paused by watchdog at %s: GPU %s C; auto-resume waits for < %s C\n" % (_iso(), t, ctx["resume_cool_c"]))
        return [{"kind": "gpu_hot", "subject": "gpu", "subject_type": "gpu", "key": "hot@%d" % ctx["state"]["hot_since"], "severity": "high",
                 "hint": "thermal", "summary": "GPU at %s C (limit %d); campaign waits, auto-resume paused until < %s C" % (t, core.HOT_C, ctx["resume_cool_c"]),
                 "actions": ["wrote pause flag %s" % flag], "repro": "nvidia-smi; bash experiments/launch/status.sh", "evidence": hw}]
    if t < ctx["resume_cool_c"]:
        ctx["state"]["hot_since"] = None
        flag.unlink(missing_ok=True)  # the watchdog's own flag file
    return []


def d_disk(ctx):
    try:
        u = shutil.disk_usage(core.DATA if core.DATA.exists() else core.DATA.parent)
    except OSError:
        return []
    free_gb = u.free / 2**30
    if free_gb >= ctx["disk_min_gb"]:
        return []
    return [{"kind": "disk_low", "subject": str(core.DATA), "subject_type": "disk", "key": time.strftime("%Y%m%d", time.localtime(ctx["now"])),
             "severity": "high", "hint": "resource", "summary": "only %.1f GB free on the data drive (min %.0f GB)" % (free_gb, ctx["disk_min_gb"]),
             "repro": "df -h %s; du -sh %s/*" % (core.DATA, core.DATA), "evidence": {"free_gb": round(free_gb, 1)}}]


def d_cloud(ctx):
    """Cloud guards + dispatcher (cockpit/fleet.py): destroys over-time / idle / stuck / over-budget cockpit instances (each an incident),
    SSH-polls remote runs (NaN/OOM/traceback/hung/dead incidents tagged with the instance), rents just-in-time for queued jobs
    within the human approval. No network at all when nothing is running or queued."""
    try:
        from . import fleet
    except ImportError:
        import fleet  # type: ignore
    if not fleet.enabled():
        return []
    out = fleet.watch(ctx["now"])
    t = fleet.tick()
    return out + list(t.get("findings") or [])


DETECTORS = [d_runs, d_logs, d_campaigns, d_cpu, d_servers, d_thermal, d_disk, d_cloud]


# --------------------------------------------------------------------------- safe recovery
def _gate_resume(auto: bool) -> str | None:
    """Reason NOT to launch a resume campaign now, or None."""
    hold = cdir() / "hold.flag"   # operator hold: only a human removes it (cockpit.sh --release-hold); blocks manual AND auto
    if hold.exists():
        return "operator hold: %s" % core._read(hold).strip()[:160]
    if core.code_paths_dirty():
        return "launch-relevant code paths are dirty: %s" % core.code_paths_dirty()[:3]
    if core._gpu_slot()["held"]:
        return "GPU slot is held"
    if any(x["state"] == "running" for x in core.campaigns()):
        return "a campaign runner is alive"
    hw = core._gpu_hw()
    if hw.get("available") and (hw.get("temp_c") or 0) >= (DEFAULTS["resume_cool_c"] if auto else core.HOT_C):
        return "GPU is %s C" % hw.get("temp_c")
    if auto and (cdir() / "pause.flag").exists():
        return "auto-resume paused by thermal flag"
    return None


def resume_campaign(file: str, confirm: bool = False, dry_run: bool = False, _auto: bool = False) -> dict:
    """Launch a resume campaign written by the watchdog, through tools/launch_campaign.sh only."""
    if confirm is not True and not _auto:
        return {"launched": False, "refused": "confirm must be exactly true"}
    p = Path(file)
    if not p.is_absolute():
        p = core.REPO / p
    p = p.resolve()
    if CAMPAIGN_DIR.resolve() not in p.parents or not re.search(r"-resume-\d+\.campaign$", p.name) and not p.name.endswith("-resume.campaign"):
        return {"launched": False, "refused": "only *-resume*.campaign files under experiments/launch/campaigns/ may be resumed here"}
    if not p.exists():
        return {"launched": False, "refused": "no such file"}
    why = _gate_resume(_auto)
    if why:
        return {"launched": False, "refused": why}
    argv = ["bash", str(core.REPO / "tools/launch_campaign.sh"), str(p)] + (["--dry-run"] if dry_run else [])
    rc, out = core._sh(argv, timeout=120)
    r = {"launched": rc == 0 and not dry_run, "dry_run": dry_run, "rc": rc, "file": str(p), "output": core.redact(core._cap(out, 1500))}
    logdb.event("watchdog" if _auto else "mcp", "info" if rc == 0 else "error", "resume_campaign", "resume %s rc=%s dry=%s" % (p.name, rc, dry_run))
    return r


def _auto_restart_server(ctx, name) -> str:
    st = _server_state(ctx)[name]
    hist = [t for t in st["restarts"] if ctx["now"] - t < 3600]
    st["restarts"] = hist
    if len(hist) >= MAX_AUTO_RESTARTS_PER_H:
        return "restart skipped: %d attempts in the last hour (max %d)" % (len(hist), MAX_AUTO_RESTARTS_PER_H)
    if hist and ctx["now"] - hist[-1] < 30 * 2 ** (len(hist) - 1):
        return "restart deferred: backoff"
    st["restarts"].append(ctx["now"])
    try:
        r = core.start_server(name, core.SERVERS[name]["port"])
    except Exception as ex:  # noqa: BLE001
        return "restart attempt raised %r" % ex
    return "restart attempt %d/%d: %s" % (len(st["restarts"]), MAX_AUTO_RESTARTS_PER_H, "up" if r.get("up") else "FAILED " + str(r.get("output", r))[:120])


def _retry_budget(ctx, family: str) -> bool:
    r = ctx["state"].setdefault("retries", {})
    if r.get(family, 0) >= 1:
        return False
    r[family] = r.get(family, 0) + 1
    return True


def handle_findings(ctx, findings: list) -> list:
    new = []
    for f in findings:
        acts = []
        if f["kind"] == "server_down":
            fp = _fp(f["kind"], f["subject"], f.get("key", ""))
            first = not _known(fp)
            if first or True:
                acts.append(_auto_restart_server(ctx, f["_server"]))
        if f["kind"] == "campaign_aborted":
            left, c = f.pop("_left"), f.pop("_c")
            if not _known(_fp(f["kind"], f["subject"], f.get("key", ""))):
                cls = classify(f.get("tail", ""), f.get("hint", ""))
                p = write_resume_file(c, left, cls["category"] + ": " + f["summary"])
                f["resume_file"] = str(p.relative_to(core.REPO)) if core.REPO in p.parents else str(p)
                acts.append("wrote %s with %d unfinished item(s)" % (f["resume_file"], len(left)))
                if cls["category"] in ("transient", "thermal"):
                    if _retry_budget(ctx, "campaign:" + re.sub(r"-resume(-\d+)?$", "", c["name"])):
                        r = resume_campaign(str(p), _auto=True)
                        acts.append("auto-resume (%s): %s" % (cls["category"], "launched" if r.get("launched") else "not launched: " + str(r.get("refused", r.get("output", ""))[:150])))
                    else:
                        acts.append("auto-resume skipped: retry budget (1) already used for this campaign")
                else:
                    acts.append("auto-resume NOT done: cause is %s (agent must fix first, then gogeta_resume_campaign)" % cls["category"])
        if f["kind"] == "run_failed" and not f.get("campaign"):
            cls = classify(f.get("tail", ""))
            if cls["category"] == "transient":
                acts.append("transient: one retry allowed after cool-down" if _retry_budget(ctx, "run:" + str(f.get("cli_args"))) else "retry budget used; not retrying")
        inc = record(f, acts)
        if inc:
            new.append(inc)
    return new


def _known(fp: str) -> bool:
    con = logdb.connect()
    try:
        return bool(con.execute("SELECT 1 FROM incidents WHERE fingerprint=?", (fp,)).fetchone())
    finally:
        con.close()


def _close_recovered(ctx) -> None:
    """Servers that came back: mark their open incident fixed (auto)."""
    for name, st in _server_state(ctx).items():
        if st.get("recovered"):
            con = logdb.connect()
            rows = con.execute("SELECT id FROM incidents WHERE kind='server_down' AND subject=? AND status IN ('open','triaged','fixing')", (name,)).fetchall()
            con.close()
            for r in rows:
                update_incident(r["id"], "fixed", "auto: server %s answers again (watchdog)" % name)
            st["recovered"] = None


def scan_once(overrides: dict | None = None) -> dict:
    ctx = {**DEFAULTS, **(overrides or {}), "now": now_s(), "state": _load_state()}
    findings, errors = [], []
    for det in DETECTORS:
        try:
            findings += det(ctx)
        except Exception as ex:  # noqa: BLE001
            errors.append("%s: %r" % (det.__name__, ex))
            logdb.error("watchdog", "detector %s crashed: %r" % (det.__name__, ex), type(ex).__name__, _tb.format_exc())
    new = handle_findings(ctx, findings)
    _close_recovered(ctx)
    _save_state(ctx["state"])
    return {"new": [i["id"] for i in new], "incidents": new, "detector_errors": errors, "findings": len(findings)}


# --------------------------------------------------------------------------- daemon
def _lock_path() -> Path:
    return cdir() / "watchdog.lock"


def acquire_lock():
    """Single instance: returns an open fd holding an exclusive flock, or None if another instance holds it."""
    fd = open(_lock_path(), "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    fd.seek(0)
    fd.truncate()
    fd.write(str(os.getpid()))
    fd.flush()
    return fd


def status() -> dict:
    pf = cdir() / "watchdog.pid"
    pid = core._pid(pf)
    alive = bool(pid) and "watchdog" in core._cmdline(pid)
    hb = {}
    try:
        hb = json.loads((cdir() / "watchdog-heartbeat.json").read_text())
    except (OSError, ValueError):
        pass
    age = round(time.time() - hb["epoch"], 1) if hb.get("epoch") else None
    return {"running": alive, "pid": pid if alive else None, "heartbeat_age_s": age, "healthy": bool(alive and age is not None and age < 60),
            "cycles": hb.get("cycles"), "last_new": hb.get("last_new"), "log": str(cdir() / "incidents.log"), "open_incidents": open_count()}


def run_forever(interval: float, overrides: dict) -> int:
    fd = acquire_lock()
    if fd is None:
        print("watchdog already running", file=sys.stderr)
        return 0
    (cdir() / "watchdog.pid").write_text(str(os.getpid()) + "\n")
    stop = []
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, lambda *_: stop.append(1))
    cycles, last_new = 0, []
    logdb.event("watchdog", "info", "watchdog_start", "watchdog started pid %d interval %.0fs" % (os.getpid(), interval))
    while not stop:
        try:
            r = scan_once(overrides)
            last_new = r["new"] or last_new
        except Exception as ex:  # noqa: BLE001  the watchdog itself must survive anything
            logdb.error("watchdog", "scan crashed: %r" % ex, type(ex).__name__, _tb.format_exc())
        cycles += 1
        (cdir() / "watchdog-heartbeat.json").write_text(json.dumps({"epoch": time.time(), "ts": _iso(), "pid": os.getpid(), "cycles": cycles, "last_new": last_new}))
        for _ in range(int(interval * 5)):
            if stop:
                break
            time.sleep(0.2)
    logdb.event("watchdog", "info", "watchdog_stop", "watchdog stopped")
    (cdir() / "watchdog.pid").unlink(missing_ok=True)
    return 0


def start(interval: float = DEFAULTS["interval"], hung_minutes: float = DEFAULTS["hung_minutes"]) -> dict:
    s = status()
    if s["running"]:
        return {**s, "started": False, "note": "already running"}
    out = open(cdir() / "watchdog.out", "ab")
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--run", "--interval", str(interval), "--hung-minutes", str(hung_minutes)],
                     stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True, cwd=str(core.DATA if core.DATA.exists() else "/"))
    for _ in range(40):
        time.sleep(0.25)
        if status()["running"]:
            break
    return {**status(), "started": True}


def stop() -> dict:
    s = status()
    if not s["running"]:
        return {**s, "stopped": False, "note": "not running"}
    os.kill(s["pid"], signal.SIGTERM)  # our own daemon only (cmdline verified in status())
    for _ in range(40):
        time.sleep(0.25)
        if not status()["running"]:
            break
    return {**status(), "stopped": not status()["running"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="watchdog")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--once", action="store_true")
    ap.add_argument("--start", action="store_true"); ap.add_argument("--stop", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--interval", type=float, default=DEFAULTS["interval"]); ap.add_argument("--hung-minutes", type=float, default=DEFAULTS["hung_minutes"])
    a = ap.parse_args(argv)
    ov = {"hung_minutes": a.hung_minutes}
    if a.run:
        return run_forever(a.interval, ov)
    if a.once:
        r = scan_once(ov)
        print(json.dumps({k: r[k] for k in ("new", "findings", "detector_errors")}))
    elif a.start:
        print(json.dumps(start(a.interval, a.hung_minutes), default=str))
    elif a.stop:
        print(json.dumps(stop(), default=str))
    else:
        print(json.dumps(status(), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
