"""Gogeta-0 cockpit core: stdlib only. One function per question, each returns a JSON-able dict.

Used by the TUI (tui.py) and the MCP server (mcp_server.py); testable without any library
(python3 -m unittest cockpit.test_core). Designed to run inside WSL (the launchers are bash scripts
and the run data lives on the WSL filesystem). READ-ONLY except start_server / stop_server /
launch_queue_item, which only call the project's existing launchers (or a static docs server) and
never delete anything.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("GOGETA_REPO") or Path(__file__).resolve().parent.parent)
DATA = Path(os.environ.get("GOGETA_DATA") or Path.home() / "fusiondance-data")
RUN_DIR = DATA / "run"
CPU_DIR = RUN_DIR / "cpu"
LOG_DIR = DATA / "logs"
RUNS_DIR = DATA / "runs"
COCKPIT_RUN = RUN_DIR / "cockpit"
GPU_LOCK = RUN_DIR / "gpu.lock"
MAX_CPU = 8
HOT_C = 85
LAUNCH_PATHS = ["gogeta", "configs", "tools", "tests", "requirements.txt"]

SERVERS = {
    "tracker": {"port": 7861, "desc": "Trackio dashboard (launch_tracker.sh)"},
    "viz": {"port": 7870, "desc": "2D model visualiser (launch_viz.sh)"},
    "site": {"port": 8791, "desc": "static docs site (python http.server on docs/)"},
}
SITE_PAGES = {"site": "/", "arch3d": "/architecture.html"}

# --------------------------------------------------------------------------- helpers
_REDACT = [
    (re.compile(r"hf_[A-Za-z0-9]{6,}"), "hf_[REDACTED]"),
    (re.compile(r"(?i)(\"?(?:write[_-]?token|access[_-]?token|auth[_-]?token|api[_-]?key|secret|password|hf[_-]?token|(?<![A-Za-z])token)\"?\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&\"')}]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)([?&;]\s*(?:__sign|token|write_token|access_token)=)[^\s&\"')]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\btoken=[^\s&\"')]+"), "token=[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization:\s*)\S+(\s+\S+)?"), r"\1[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-[REDACTED]"),
]


def redact(text: str) -> str:
    """Strip secrets (HF tokens, token=..., write_token, bearer/auth headers) from any text we return."""
    if not text:
        return text
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text


def _read(p: Path, limit: int | None = None) -> str:
    try:
        t = Path(p).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return t if limit is None else t[:limit]


def _cap(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 20] + "\n...[truncated]"


def _sh(args, timeout=60, cwd=None, env=None) -> tuple[int, str]:
    e = dict(os.environ)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        e.update(env)
    try:
        r = subprocess.run(args, cwd=str(cwd or REPO), capture_output=True, text=True, timeout=timeout,
                           env=e, errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as ex:
        out = ex.stdout if isinstance(ex.stdout, str) else (ex.stdout or b"").decode("utf-8", "replace")
        return 124, (out or "") + "\n[timeout]"
    except OSError as ex:
        return 127, str(ex)


_GIT_CACHE: dict = {}


def _git(*a, timeout=20, ttl=45) -> str:
    key = a
    hit = _GIT_CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    v = _git_raw(*a, timeout=timeout)
    _GIT_CACHE[key] = (time.time(), v)
    return v


def _git_raw(*a, timeout=20) -> str:
    rc, out = _sh(["git", "-c", "safe.directory=*", "-C", str(REPO), *a], timeout=timeout)
    return out.strip() if rc == 0 else ""


def _cmdline(pid) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, ValueError, TypeError):
        return ""


def _alive_with(pid, marker: str) -> bool:
    return bool(pid) and marker in _cmdline(pid)


def _kv(path: Path) -> dict:
    d = {}
    for ln in _read(path).splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def _pid(path: Path):
    m = re.search(r"\d+", _read(path))
    return int(m.group()) if m else None


def _etime(pid) -> str:
    rc, out = _sh(["ps", "-o", "etime=", "-p", str(pid)], timeout=5)
    return out.strip() if rc == 0 else ""


def port_up(port: int, host="127.0.0.1", timeout=0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _f(x, nd=4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- status
def _gpu_slot() -> dict:
    kv = _kv(GPU_LOCK) if GPU_LOCK.exists() else {}
    pid = int(kv["pid"]) if kv.get("pid", "").isdigit() else None
    held = bool(pid) and _alive_with(pid, "gogeta-gpu-slot")
    out = {"held": held, "stale_lock": bool(kv) and not held}
    if held:
        out.update(experiment=kv.get("experiment"), seed=kv.get("seed"), pid=pid, log=kv.get("log"),
                   running_for=_etime(pid))
    return out


def _gpu_hw() -> dict:
    nv = "nvidia-smi"
    if not any(os.access(os.path.join(p, nv), os.X_OK) for p in os.environ.get("PATH", "").split(":")):
        nv = "/usr/lib/wsl/lib/nvidia-smi"
    rc, out = _sh([nv, "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,name",
                   "--format=csv,noheader,nounits"], timeout=10)
    if rc != 0 or not out.strip():
        return {"available": False}
    p = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    d = {"available": True, "temp_c": _f(p[0], 1), "util_pct": _f(p[1], 0), "mem_used_mib": _f(p[2], 0),
         "mem_total_mib": _f(p[3], 0) if len(p) > 3 else None, "name": p[4] if len(p) > 4 else ""}
    t = d["temp_c"]
    d["band"] = "unknown" if t is None else "hot" if t >= HOT_C else "warm" if t >= 75 else "ok"
    rc, apps = _sh([nv, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], timeout=10)
    d["compute_procs"] = [a.strip() for a in apps.strip().splitlines() if a.strip()] if rc == 0 else []
    return d


def _train_runs() -> list:
    runs = []
    for f in sorted(RUN_DIR.glob("train-*.pid"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        base = f.with_suffix("")
        name = base.name[len("train-"):]
        pid = _pid(f)
        info = _kv(base.with_suffix(".info"))
        r = {"name": name, "log": info.get("log"), "started": info.get("started")}
        if _alive_with(pid, "gogeta-gpu-slot"):
            r.update(state="running", pid=pid, running_for=_etime(pid))
        elif base.with_suffix(".exit").exists():
            r.update(state="ended", exit=_read(base.with_suffix(".exit")).strip())
        else:
            r.update(state="killed")
        runs.append(r)
    return runs


def _cpu_jobs() -> dict:
    jobs, ended = [], []
    for f in sorted(CPU_DIR.glob("*.pid"), key=lambda p: p.stat().st_mtime, reverse=True):
        base = f.with_suffix("")
        pid = _pid(f)
        info = _kv(base.with_suffix(".info"))
        if _alive_with(pid, "gogeta-cpu-job"):
            jobs.append({"name": base.name, "pid": pid, "running_for": _etime(pid), "cmd": info.get("cmd", "")[:140],
                         "log": info.get("log")})
    for e in sorted(CPU_DIR.glob("*.exit"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        ended.append({"name": e.stem, "exit": _read(e).strip()})
    return {"alive": len(jobs), "max": MAX_CPU, "jobs": jobs, "recent_ended": ended}


def parse_final(text: str) -> dict:
    """Numbers from a campaign 'FINAL {json}' fragment (the log line is often truncated, so regex, not json)."""
    d = {}
    for k in ("val_loss", "val_bits_per_char", "tokens", "flops", "steps", "wall_min", "tokens_per_s"):
        m = re.search(r'"%s":\s*(-?[0-9.]+(?:e[-+]?\d+)?|NaN|Infinity)' % k, text)
        if m:
            try:
                d[k] = float(m.group(1))
            except ValueError:
                pass
    return d


_ITEM_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(\w+)[: ]\s*(.*)")


def _parse_args(a: str) -> dict:
    t = a.split()
    d = {"experiment": t[0] if t else ""}
    for i, x in enumerate(t):
        if x in ("--seed", "--lr", "--max-tokens") and i + 1 < len(t):
            d[x.lstrip("-").replace("-", "_")] = t[i + 1]
    return d


def campaigns() -> list:
    out = []
    for pidf in sorted(RUN_DIR.glob("campaign-*.pid"), key=lambda p: p.stat().st_mtime, reverse=True):
        base = pidf.with_suffix("")
        name = base.name[len("campaign-"):]
        info = _kv(base.with_suffix(".info"))
        pid = _pid(pidf)
        alive = _alive_with(pid, "launch_campaign")
        total = int(info.get("items", "0") or 0)
        items = []
        for ln in _read(base.with_suffix(".state")).splitlines():
            p = ln.split("\t")
            if len(p) >= 5:
                items.append({"idx": int(p[0]), "rc": int(p[1]), "args": p[4], **_parse_args(p[4])})
        finals, current, last_event = {}, None, ""
        for ln in _read(Path(info.get("log", "/nonexistent"))).splitlines():
            m = _ITEM_RE.search(ln)
            if not m:
                if ln.strip():
                    last_event = ln.strip()
                continue
            last_event = ln.strip()
            idx, kind, rest = int(m.group(1)), m.group(3).lower(), m.group(4)
            if kind == "launch":
                current = {"idx": idx, "args": rest.split("  (output")[0].strip(), "started": ln[1:20]}
            elif kind in ("done", "failed"):
                current = None
                if "FINAL" in rest:
                    finals[idx] = parse_final(rest)
        for it in items:
            fin = finals.get(it["idx"], {})
            it["val_loss"] = _f(fin.get("val_loss"))
            it["bpc"] = _f(fin.get("val_bits_per_char"))
            it["wall_min"] = _f(fin.get("wall_min"), 2)
            it["tokens_per_s"] = _f(fin.get("tokens_per_s"), 0)
        if current:
            current.update(_parse_args(current["args"]))
        ex = base.with_suffix(".exit")
        out.append({"name": name, "state": "running" if alive else "ended", "pid": pid if alive else None,
                    "exit": _read(ex).strip() if ex.exists() and not alive else None,
                    "running_for": _etime(pid) if alive else "", "total": total, "done": len(items),
                    "failed": sum(1 for i in items if i["rc"] != 0), "started": info.get("started"),
                    "git": info.get("git"), "log": info.get("log"), "current": current if alive else None,
                    "items": items, "last_event": redact(last_event)[:200]})
    return out


def servers() -> dict:
    out = {}
    for n, s in SERVERS.items():
        out[n] = {"port": s["port"], "up": port_up(s["port"]), "url": f"http://127.0.0.1:{s['port']}/", "desc": s["desc"]}
    out["arch3d"] = {"port": 8791, "up": out["site"]["up"], "url": "http://127.0.0.1:8791/architecture.html",
                     "desc": "3D architecture page (served by the site server)"}
    own = []
    for f in COCKPIT_RUN.glob("*.pid") if COCKPIT_RUN.exists() else []:
        pid = _pid(f)
        own.append({"file": f.name, "pid": pid, "alive": _alive_with(pid, "http.server")})
    return {"servers": out, "cockpit_started": own}


def status() -> dict:
    """Parsed status.sh (GPU slot, temperature, runs, CPU jobs) + campaigns + servers."""
    rc, raw = _sh(["bash", str(REPO / "experiments/launch/status.sh"), "-n", "0"], timeout=60)
    slot = _gpu_slot()
    hw = _gpu_hw()
    m = re.search(r"temp\s*:\s*([\d.]+)\s*C\s+util\s+(\d+)%\s+mem\s+(\d+)", raw)
    if m and not hw.get("available"):
        hw = {"available": True, "temp_c": float(m.group(1)), "util_pct": int(m.group(2)), "mem_used_mib": int(m.group(3))}
    camps = campaigns()
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": {"slot": slot, **hw, "hot_limit_c": HOT_C},
        "runs": _train_runs(),
        "cpu": _cpu_jobs(),
        "campaigns": camps,
        "campaign_running": any(c["state"] == "running" for c in camps),
        "servers": servers()["servers"],
        "status_sh_exit": rc,
        "status_sh_hint": "; ".join(l.strip() for l in raw.splitlines() if "outside the launcher" in l or "NOT via the launcher" in l)[:300],
    }


# --------------------------------------------------------------------------- queue
def _qload() -> dict:
    return json.loads(_read(REPO / "experiments/queue.json") or "{}")


def _blockers(q, it) -> list:
    m = {i["id"]: i for i in q["items"]}
    if it["status"] != "queued":
        return ["status is %s" % it["status"]]
    why = []
    if not it.get("ready"):
        why.append("not ready: " + "; ".join(it.get("missing") or ["unspecified"]))
    for d in it.get("depends_on", []):
        if m.get(d, {}).get("status") != "done":
            why.append("waits for %s (%s)" % (d, m.get(d, {}).get("status", "?")))
    if it.get("needs_human"):
        why.append("needs human: " + it.get("reason", ""))
    return why


def _label(q, it) -> str:
    b = _blockers(q, it)
    if it["status"] != "queued":
        return it["status"].upper()
    if not b:
        return "READY"
    if any(x.startswith("needs human") for x in b):
        return "HUMAN"
    if any(x.startswith("not ready") for x in b):
        return "NOT-READY"
    return "WAITING"


def _pickable(q, kind=None):
    running = [i for i in q["items"] if i["status"] == "running"]
    if kind == "gpu" and any(i["resources"]["type"] == "gpu" for i in running):
        return []
    if kind == "cpu" and sum(1 for i in running if i["resources"]["type"] == "cpu") >= MAX_CPU:
        return []
    return [i for i in q["items"] if not _blockers(q, i) and (not kind or i["resources"]["type"] == kind)]


def _qrow(q, n, it) -> dict:
    r = it["resources"]
    return {"pos": n, "id": it["id"], "state": _label(q, it), "status": it["status"], "kind": it["kind"],
            "res": r["type"], "est_min": r.get("est_minutes"), "tokens_M": (r.get("tokens") or 0) // 1_000_000,
            "title": it["title"], "depends_on": it.get("depends_on", []), "blockers": _blockers(q, it),
            "needs_human": bool(it.get("needs_human")), "launch": it.get("launch")}


def queue(filter: str = "", include_done: bool = False, limit: int = 60) -> dict:
    """Queue rows in priority order. filter: text, or 'state:READY', 'res:gpu', 'kind:train', 'status:running'."""
    q = _qload()
    rows = [_qrow(q, n, it) for n, it in enumerate(q.get("items", []), 1)]
    if not include_done and "status:" not in filter:
        rows = [r for r in rows if r["status"] not in ("done", "dropped")]
    for tok in filter.split():
        if ":" in tok:
            k, v = tok.split(":", 1)
            key = {"state": "state", "res": "res", "kind": "kind", "status": "status", "id": "id"}.get(k)
            if key:
                rows = [r for r in rows if str(r[key]).lower() == v.lower()]
                continue
        rows = [r for r in rows if tok.lower() in (r["id"] + " " + r["title"]).lower()]
    return {"updated": q.get("updated"), "count": len(rows), "rows": rows[:limit],
            "next_gpu": (_pickable(q, "gpu") or [{}])[0].get("id"), "next_cpu": (_pickable(q, "cpu") or [{}])[0].get("id")}


def queue_next(kind: str = "gpu") -> dict:
    q = _qload()
    kind = kind if kind in ("gpu", "cpu") else None
    c = _pickable(q, kind)
    if c:
        it = c[0]
        return {"kind": kind, "item": {**_qrow(q, 0, it), "why": it.get("why"), "note": it.get("note")}}
    skipped = []
    for it in q["items"]:
        if it["status"] == "queued" and (not kind or it["resources"]["type"] == kind):
            skipped.append({"id": it["id"], "blockers": _blockers(q, it)})
            if len(skipped) >= 6:
                break
    return {"kind": kind, "item": None, "running": [i["id"] for i in q["items"] if i["status"] == "running"],
            "skipped": skipped}


def queue_item(item_id: str) -> dict:
    q = _qload()
    for n, it in enumerate(q["items"], 1):
        if it["id"].lower() == item_id.lower():
            return {**it, "state": _label(q, it), "blockers": _blockers(q, it)}
    return {"error": "unknown queue id %s" % item_id}


# --------------------------------------------------------------------------- experiments
def _registry() -> list:
    try:
        return json.loads(_read(REPO / "experiments/experiments.json"))
    except ValueError:
        return []


def index() -> dict:
    txt = _read(REPO / "experiments/INDEX.md")
    rows = []
    for ln in txt.splitlines():
        m = re.match(r"\|\s*\[?([A-Z]+\d+)\]?(?:\([^)]*\))?\s*\|(.*)\|\s*$", ln)
        if m:
            c = [x.strip() for x in m.group(2).split("|")]
            if len(c) >= 3:
                rows.append({"id": m.group(1), "title": c[0], "status": c[1], "headline": c[2],
                             "reuse": c[3] if len(c) > 3 else "", "config": c[4] if len(c) > 4 else ""})
    return {"count": len(rows), "rows": rows, "text": _cap(txt, 12000)}


def _block(text: str, heading_re: str, level_stop=r"^#{1,3} ") -> str:
    lines, out, on = text.splitlines(), [], False
    for ln in lines:
        if on and re.match(level_stop, ln) and not re.match(heading_re, ln):
            break
        if re.match(heading_re, ln):
            on = True
        if on:
            out.append(ln)
    return "\n".join(out)


def runs(experiment: str = "", limit: int = 15) -> dict:
    rows = []
    for d in RUNS_DIR.iterdir() if RUNS_DIR.exists() else []:
        sj = d / "summary.json"
        if not sj.exists():
            continue
        if experiment and not d.name.lower().startswith(experiment.lower() + "-"):
            continue
        try:
            s = json.loads(sj.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        m, cfg = s.get("metrics", {}), s.get("config", {})
        tr = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        g = s.get("git", {})
        rows.append({"run": s.get("name", d.name), "exp": s.get("experiment"), "seed": s.get("seed"),
                     "status": s.get("status"), "val_loss": _f(m.get("val_loss")), "bpc": _f(m.get("val_bits_per_char")),
                     "tok_s": _f(m.get("tokens_per_s"), 0), "tokens_M": _f((m.get("tokens") or 0) / 1e6, 1),
                     "lr": tr.get("lr"), "wall_min": _f(m.get("wall_min"), 1), "git": g.get("describe"),
                     "started": s.get("started")})
    rows.sort(key=lambda r: r.get("started") or "", reverse=True)
    return {"count": len(rows), "rows": rows[:limit]}


def experiment(id: str) -> dict:
    """Registry entry + card + latest runs + devlog '### <ID>' blocks."""
    eid = id.strip()
    reg = next((e for e in _registry() if e.get("id", "").lower() == eid.lower()), None)
    card = _read(REPO / "experiments/cards" / f"{eid.upper()}.md")
    chk = None
    for cand in (REPO / "experiments/checks/results").glob("*.json"):
        if cand.stem.lower() == eid.lower() or cand.stem.lower().replace("-", "") == eid.lower().replace("-", ""):
            chk = cand.name
    dl = _read(REPO / "DEVLOG.md")
    blocks = []
    for m in re.finditer(r"(?m)^###\s+" + re.escape(eid) + r"(?![A-Za-z0-9])[^\n]*$", dl):
        blk = _block(dl[m.start():], r"^###\s+" + re.escape(eid) + r"(?![A-Za-z0-9])", r"^#{2,3} ")
        ver = re.findall(r"(?m)^## (v\d+\.\d+\.\d+)", dl[: m.start()])
        blocks.append({"version": ver[-1] if ver else None, "text": _cap(blk, 1800)})
    qi = [i["id"] for i in _qload().get("items", []) if i["id"].lower() == eid.lower()]
    if not (reg or card or blocks or chk or qi):
        return {"error": "no registry entry, card, devlog block or check result for %s" % eid}
    return {"id": eid, "registry": reg, "card": _cap(redact(card), 7000), "card_exists": bool(card),
            "check_result_file": chk, "queue_item": qi[0] if qi else None,
            "latest_runs": runs(eid, 6)["rows"], "devlog_blocks": blocks[:4]}


# --------------------------------------------------------------------------- ledgers
def _table_rows(path: Path, idpat: str) -> list:
    rows = []
    for n, ln in enumerate(_read(path).splitlines(), 1):
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and re.fullmatch(idpat, cells[0]):
            rows.append((n, cells))
    return rows


def _status_of(cells, pat=r"`([A-Za-z][A-Za-z ]*)`") -> str:
    for x in cells[1:]:
        m = re.match(pat, x)
        if m:
            return m.group(1)
    return ""


def claims(query: str = "", status: str = "", id: str = "") -> dict:
    out = []
    for n, c in _table_rows(REPO / "experiments/claims.md", r"C\d+"):
        st = _status_of(c)
        row = {"id": c[0], "claim": c[1] if len(c) > 1 else "", "status": st, "line": n,
               "evidence": c[-1] if len(c) > 4 else ""}
        if id and row["id"].lower() != id.lower():
            continue
        if status and status.lower() not in st.lower():
            continue
        if query and query.lower() not in " ".join(c).lower():
            continue
        out.append(row)
    allrows = _table_rows(REPO / "experiments/claims.md", r"C\d+")
    counts = {}
    for _, c in allrows:
        st = _status_of(c) or "?"
        counts[st] = counts.get(st, 0) + 1
    return {"total": len(allrows), "by_status": counts, "count": len(out), "rows": [
        {**r, "claim": r["claim"][:260], "evidence": r["evidence"][:260]} for r in out[:60]]}


def side_quests(query: str = "", status: str = "") -> dict:
    out = []
    for n, c in _table_rows(REPO / "experiments/side-quests.md", r"SQ\d+"):
        st = _status_of(c)
        row = {"id": c[0], "idea": c[1] if len(c) > 1 else "", "status": st, "line": n,
               "outcome": c[-1] if len(c) > 4 else ""}
        if status and status.lower() != st.lower():
            continue
        if query and query.lower() not in " ".join(c).lower():
            continue
        out.append({**row, "idea": row["idea"][:220], "outcome": row["outcome"][:260]})
    return {"count": len(out), "rows": out[:80]}


def list_results() -> dict:
    out = []
    for d, kind in ((REPO / "experiments/checks/results", "check"), (REPO / "evals/results", "eval")):
        for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []:
            out.append({"name": f.stem, "kind": kind, "size": f.stat().st_size,
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))})
    return {"count": len(out), "results": out}


def results(name: str, max_chars: int = 6000) -> dict:
    cands = []
    for d in (REPO / "experiments/checks/results", REPO / "evals/results"):
        if d.exists():
            cands += [f for f in d.glob("*.json") if f.stem == name or f.stem.lower() == name.lower()]
    if not cands:
        cands = [f for d in (REPO / "experiments/checks/results", REPO / "evals/results") if d.exists()
                 for f in d.glob("*.json") if name.lower() in f.stem.lower()]
    if not cands:
        return {"error": "no result named %s; see list_results" % name}
    f = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    txt = _read(f)
    try:
        data = json.loads(txt)
        body = json.dumps(data, indent=1, ensure_ascii=False)
    except ValueError:
        data, body = None, txt
    return {"name": f.stem, "path": str(f.relative_to(REPO)), "keys": list(data)[:30] if isinstance(data, dict) else None,
            "text": redact(_cap(body, max_chars))}


# --------------------------------------------------------------------------- devlog / notes
def versions(limit: int = 30) -> dict:
    out = _git("for-each-ref", "refs/tags/v*", "--sort=-v:refname", "--count=%d" % limit, "--format=%(refname:short)	%(contents:subject)")
    rows = []
    for ln in out.splitlines()[:limit]:
        t, _, s = ln.partition("\t")
        rows.append({"tag": t, "title": re.sub(r"^v[\d.]+:\s*", "", s)})
    return {"head": _git("describe", "--tags", "--always"), "count": len(rows), "tags": rows}


def _devlog_sections() -> list:
    txt = _read(REPO / "DEVLOG.md")
    parts = re.split(r"(?m)^(?=## v\d+\.\d+\.\d+)", txt)
    secs = []
    for p in parts:
        m = re.match(r"## (v\d+\.\d+\.\d+):\s*(.*)", p)
        if m:
            secs.append({"version": m.group(1), "title": m.group(2).strip(), "text": p.rstrip()})
    return secs


def devlog(version: str = "", latest: int = 0) -> dict:
    secs = _devlog_sections()
    if version:
        v = version if version.startswith("v") else "v" + version
        for s in secs:
            if s["version"] == v:
                return {**s, "text": _cap(s["text"], 9000)}
        return {"error": "no devlog entry for %s" % v}
    n = latest or 5
    return {"entries": [{"version": s["version"], "title": s["title"], "text": _cap(s["text"], 900)} for s in secs[:n]],
            "total": len(secs)}


def notebook(search: str = "", tail: int = 40) -> dict:
    lines = _read(REPO / "notes/LABNOTEBOOK.md").splitlines()
    if search:
        hits = [{"line": i + 1, "text": redact(l.strip())[:300]} for i, l in enumerate(lines) if search.lower() in l.lower()]
        return {"hits": hits[:40], "count": len(hits)}
    return {"tail": redact("\n".join(lines[-tail:])), "total_lines": len(lines)}


def notes(name: str = "") -> dict:
    d = REPO / "notes"
    if not name:
        return {"notes": [{"name": f.name, "size": f.stat().st_size} for f in sorted(d.glob("*.md"))] +
                         [{"name": "reviews/" + f.name, "size": f.stat().st_size} for f in sorted(d.glob("reviews/*.md"))]}
    p = (d / name).resolve()
    if d.resolve() not in p.parents or p.suffix != ".md" or not p.exists():
        return {"error": "no such note (use names from notes(); .md files under notes/ only)"}
    return {"name": name, "text": redact(_cap(_read(p), 12000))}


def _review_files() -> list:
    fs = list((REPO / "notes/reviews").glob("*.md")) + list((REPO / "notes").glob("review*.md"))
    return sorted(fs, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def reviews(latest: int = 1) -> dict:
    fs = _review_files()
    out = []
    for f in fs[:latest]:
        out.append({"file": str(f.relative_to(REPO)), "text": redact(_cap(_read(f), 6000))})
    return {"count": len(fs), "reviews": out}


def search(text: str, limit: int = 40) -> dict:
    if not text or len(text) < 2:
        return {"error": "query too short"}
    files = [REPO / "DEVLOG.md", REPO / "AGENTS.md", REPO / "README.md", REPO / "experiments/claims.md",
             REPO / "experiments/side-quests.md", REPO / "experiments/QUEUE.md", REPO / "experiments/INDEX.md"]
    files += sorted((REPO / "experiments/cards").glob("*.md")) + sorted((REPO / "notes").glob("*.md"))
    files += sorted((REPO / "notes/reviews").glob("*.md"))
    hits, needle = [], text.lower()
    for f in files:
        for i, ln in enumerate(_read(f).splitlines(), 1):
            if needle in ln.lower():
                hits.append({"file": str(f.relative_to(REPO)), "line": i, "text": redact(ln.strip())[:240]})
    return {"query": text, "count": len(hits), "hits": hits[:limit]}


# --------------------------------------------------------------------------- logs
def _log_sources() -> dict:
    d = {}
    for f in sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if LOG_DIR.exists() else []:
        kind = "campaign" if f.name.startswith("campaign-") else "tracker" if f.name.startswith(("tracker", "viz")) else "run"
        d[f.name] = (kind, f)
    for f in sorted(CPU_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if CPU_DIR.exists() else []:
        d["cpu:" + f.stem] = ("cpu-job", f)
    return d


def list_logs(limit: int = 40) -> dict:
    src = _log_sources()
    return {"logs": [{"name": n, "kind": k, "size": f.stat().st_size, "age_s": int(time.time() - f.stat().st_mtime)}
                     for n, (k, f) in list(src.items())[:limit]]}


def log_tail(name: str, n: int = 40) -> dict:
    """Tail of a training/campaign/cpu-job log, redacted. name: a name from list_logs, a run id prefix
    (e.g. 'E10-s0-20261122-1742'), a campaign name, or 'cpu:<job>'."""
    n = max(1, min(int(n), 400))
    src = _log_sources()
    key = name if name in src else name + ".log" if name + ".log" in src else None
    if not key:
        if name.startswith("cpu:"):
            key = next((k for k in src if k.startswith(name)), None)
        else:
            key = next((k for k, (kd, _) in src.items() if k.lower().startswith(name.lower()) or name.lower() in k.lower()), None)
    if not key:
        return {"error": "no log matches %r; see list_logs" % name}
    kind, f = src[key]
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            sz = fh.tell()
            fh.seek(max(0, sz - 200_000))
            txt = fh.read().decode("utf-8", "replace")
    except OSError as ex:
        return {"error": str(ex)}
    txt = txt.replace("\r", "\n")
    lines = [l for l in txt.splitlines() if l.strip()][-n:]
    return {"name": key, "kind": kind, "lines": len(lines), "text": redact("\n".join(lines))}


# --------------------------------------------------------------------------- servers
def _port_ok(port) -> int:
    p = int(port)
    if not 1024 <= p <= 65535:
        raise ValueError("port must be 1024-65535")
    return p


def start_server(name: str, port: int | None = None) -> dict:
    """STATE-CHANGING. Start tracker / viz (via the project launchers) or site (static docs, 127.0.0.1 only)."""
    if name not in SERVERS:
        return {"error": "unknown server %r; one of %s" % (name, sorted(SERVERS))}
    p = _port_ok(port or SERVERS[name]["port"])
    if port_up(p):
        return {"name": name, "port": p, "started": False, "up": True, "note": "port already answering; nothing changed"}
    if name == "site":
        COCKPIT_RUN.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        lf = open(LOG_DIR / f"cockpit-site-{p}.log", "ab")
        proc = subprocess.Popen([sys.executable if sys.executable else "python3", "-m", "http.server", str(p), "--bind", "127.0.0.1",
                                 "--directory", str(REPO / "docs")], cwd=str(DATA), stdout=lf, stderr=lf,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        (COCKPIT_RUN / f"site-{p}.pid").write_text(str(proc.pid) + "\n")
        out = f"pid {proc.pid}"
    else:
        script = {"tracker": "tools/launch_tracker.sh", "viz": "tools/launch_viz.sh"}[name]
        rc, out = _sh(["bash", str(REPO / script), "start", "--port", str(p)], timeout=150)
        if rc != 0:
            return {"name": name, "port": p, "started": False, "up": port_up(p), "output": redact(_cap(out, 1500))}
    for _ in range(40):
        if port_up(p):
            break
        time.sleep(0.25)
    return {"name": name, "port": p, "started": True, "up": port_up(p), "url": f"http://127.0.0.1:{p}/",
            "output": redact(_cap(out, 600))}


def stop_server(name: str, port: int | None = None) -> dict:
    """STATE-CHANGING. Stop a server this tooling started (launcher PID files / cockpit PID file). Never kills foreign processes."""
    if name not in SERVERS:
        return {"error": "unknown server %r; one of %s" % (name, sorted(SERVERS))}
    p = _port_ok(port or SERVERS[name]["port"])
    if name == "site":
        pf = COCKPIT_RUN / f"site-{p}.pid"
        pid = _pid(pf)
        cl = _cmdline(pid)
        if not pid or "http.server" not in cl or str(p) not in cl:
            return {"name": name, "port": p, "stopped": False, "up": port_up(p),
                    "note": "no cockpit-started site server recorded for this port; refusing to kill anything else"}
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass
        for _ in range(30):
            if not port_up(p):
                break
            time.sleep(0.2)
        if not port_up(p):
            pf.unlink(missing_ok=True)  # our own pid file, not a deletion of user data
        return {"name": name, "port": p, "stopped": not port_up(p), "up": port_up(p)}
    script = {"tracker": "tools/launch_tracker.sh", "viz": "tools/launch_viz.sh"}[name]
    rc, out = _sh(["bash", str(REPO / script), "stop", "--port", str(p)], timeout=60)
    time.sleep(0.5)
    return {"name": name, "port": p, "stopped": not port_up(p), "up": port_up(p), "output": redact(_cap(out, 600))}


# --------------------------------------------------------------------------- guarded launch
_LAUNCHERS = ("tools/launch_experiment.sh", "tools/launch_campaign.sh", "tools/launch_cpu_job.sh")
_FORBIDDEN_FLAG = re.compile(r"(?i)^--?(no-?preflight|skip\S*|force\S*|no-?require-clean|ignore\S*|unsafe\S*|yes|foreground|no-?check\S*)$")


def code_paths_dirty() -> list:
    out = _git("status", "--porcelain", "--", *LAUNCH_PATHS)
    return [l for l in out.splitlines() if l.strip()]


def launch_queue_item(id: str, confirm: bool = False, dry_run: bool = False) -> dict:
    """STATE-CHANGING (unless dry_run). Run a queue item's recorded launcher command, nothing else.

    Gates: confirm is True; item is queued with all deps done and no human needed; the command is exactly
    `bash tools/launch_{experiment,campaign,cpu_job}.sh ...` (no shell syntax, no placeholders, no skip/force flags,
    never gogeta.train); for gpu items the GPU slot is free, no campaign is running, no gpu item runs, GPU < 85 C.
    The launcher then runs its own preflight. The queue file is NOT edited: call queue.py set afterwards."""
    if confirm is not True:
        return {"launched": False, "refused": "confirm must be exactly true"}
    q = _qload()
    it = next((i for i in q.get("items", []) if i["id"].lower() == id.lower()), None)
    if not it:
        return {"launched": False, "refused": "unknown queue id %s" % id}
    b = _blockers(q, it)
    if b:
        return {"launched": False, "refused": "item not launchable: " + " | ".join(b)}
    cmd = it.get("launch") or ""
    if not cmd:
        return {"launched": False, "refused": "item has no launcher command (it is an agent task)"}
    if re.search(r"[;&|`$<>(){}\n\\]|<[^>]*>", cmd.split(" -- ")[0]) or "<" in cmd:
        return {"launched": False, "refused": "launch command has shell syntax or a placeholder: " + cmd[:120]}
    try:
        argv = shlex.split(cmd)
    except ValueError as ex:
        return {"launched": False, "refused": "unparsable launch command: %s" % ex}
    if len(argv) < 2 or argv[0] != "bash" or argv[1] not in _LAUNCHERS:
        return {"launched": False, "refused": "only `bash %s` launchers are allowed; got: %s" % (" | ".join(_LAUNCHERS), cmd[:120])}
    pre = argv[:argv.index("--")] if "--" in argv else argv
    bad = [a for a in pre[2:] if a.startswith("-") and _FORBIDDEN_FLAG.match(a)]
    if bad or "gogeta.train" in " ".join(pre):
        return {"launched": False, "refused": "forbidden flag/target in launch command: %s" % (bad or "gogeta.train")}
    argv = [os.path.expanduser(a) for a in argv]
    argv[1] = str(REPO / argv[1])
    if dry_run:
        if argv[1].endswith("launch_experiment.sh") or argv[1].endswith("launch_campaign.sh"):
            argv.append("--dry-run")
            rc, out = _sh(argv, timeout=120)
            return {"launched": False, "dry_run": True, "rc": rc, "command": cmd, "output": redact(_cap(out, 3000))}
        return {"launched": False, "dry_run": True, "command": cmd, "note": "this launcher has no --dry-run; nothing executed"}
    if it["resources"]["type"] == "gpu":
        slot = _gpu_slot()
        if slot["held"]:
            return {"launched": False, "refused": "GPU slot is held by %s seed %s (pid %s)" % (slot["experiment"], slot["seed"], slot["pid"])}
        if any(c["state"] == "running" for c in campaigns()):
            return {"launched": False, "refused": "a campaign runner is alive (it owns the GPU between items)"}
        if any(i["status"] == "running" and i["resources"]["type"] == "gpu" for i in q["items"]):
            return {"launched": False, "refused": "another GPU queue item is marked running"}
        hw = _gpu_hw()
        if hw.get("available") and (hw.get("temp_c") or 0) >= HOT_C:
            return {"launched": False, "refused": "GPU is %s C (>= %d)" % (hw["temp_c"], HOT_C)}
        if hw.get("compute_procs"):
            return {"launched": False, "refused": "a GPU compute process is running: %s" % hw["compute_procs"][:2]}
        dirty = code_paths_dirty()
        if dirty:
            return {"launched": False, "refused": "launch-relevant code paths are dirty (launcher would refuse): %s" % dirty[:4]}
    else:
        if _cpu_jobs()["alive"] >= MAX_CPU:
            return {"launched": False, "refused": "8/8 CPU job slots in use"}
    rc, out = _sh(argv, timeout=180)
    return {"launched": rc == 0, "rc": rc, "command": cmd, "output": redact(_cap(out, 2500)),
            "reminder": "queue.json was not edited; record the start with: python tools/queue.py set %s running --note ..." % it["id"]}


# --------------------------------------------------------------------------- brief
def brief(max_chars: int = 4000) -> dict:
    """One-call project overview (<= ~4 KB of text)."""
    L = []
    v = versions(3)
    dirty = code_paths_dirty()
    L.append("VERSION %s | latest tags: %s" % (v["head"], ", ".join(t["tag"] for t in v["tags"])))
    L.append("code paths dirty (launches would be refused): %s" % ("YES (%d)" % len(dirty) if dirty else "no"))
    st = status()
    g = st["gpu"]
    slot = g["slot"]
    L.append("GPU: slot %s | %s C | util %s%% | mem %s MiB" % (
        "HELD by %s s%s (%s)" % (slot.get("experiment"), slot.get("seed"), slot.get("running_for")) if slot["held"] else "free",
        g.get("temp_c"), g.get("util_pct"), g.get("mem_used_mib")))
    for c in st["campaigns"]:
        if c["state"] == "running":
            cur = c["current"]
            L.append("CAMPAIGN %s: %d/%d done%s; now %s" % (c["name"], c["done"], c["total"],
                     (", %d failed" % c["failed"]) if c["failed"] else "", cur["args"] if cur else "(between items)"))
            fin = [i for i in c["items"] if i.get("val_loss")]
            if fin:
                L.append("  finished: " + "; ".join("%s lr%s=%.4f" % (i["experiment"], i.get("lr", "?"), i["val_loss"]) for i in fin[-6:]))
    L.append("CPU jobs %d/%d | servers: %s" % (st["cpu"]["alive"], MAX_CPU, ", ".join(
        "%s:%s" % (n, "up" if s["up"] else "down") for n, s in st["servers"].items() if n != "arch3d")))
    qq = queue(limit=200)
    L.append("QUEUE next gpu=%s cpu=%s; top 5 open:" % (qq["next_gpu"], qq["next_cpu"]))
    for r in qq["rows"][:5]:
        L.append("  %-5s %-9s %s %3sm %s" % (r["id"], r["state"], r["res"], r["est_min"], r["title"][:70]))
    hum = [r["id"] for r in queue("state:HUMAN", limit=50)["rows"]]
    L.append("NEEDS HUMAN: %s" % (", ".join(hum) if hum else "none"))
    rs = list_results()["results"][:4]
    L.append("LATEST RESULTS: " + ", ".join(r["name"] for r in rs))
    rr = runs(limit=3)["rows"]
    L.append("LATEST RUNS: " + "; ".join("%s val %s" % (r["run"], r["val_loss"]) for r in rr))
    rev = _review_files()
    if rev:
        head = next((l.strip("# ").strip() for l in _read(rev[0]).splitlines() if l.strip()), "")
        L.append("LATEST REVIEW: %s (%s)" % (head[:100], rev[0].name))
    else:
        L.append("LATEST REVIEW: none yet (notes/reviews/ absent)")
    cl = claims()
    L.append("CLAIMS %d: %s" % (cl["total"], ", ".join("%s=%d" % kv for kv in cl["by_status"].items())))
    ch = _git("log", "-3", "--format=%h %s", "--", "experiments/claims.md")
    if ch:
        L.append("claims recent commits: " + " | ".join(x[:70] for x in ch.splitlines()))
    L.append("Tools: gogeta_status/queue/experiment/claims/search/runs/results/devlog/log_tail; launch only via gogeta_launch_queue_item(confirm=true).")
    text = redact("\n".join(L))
    return {"text": _cap(text, max_chars), "bytes": len(text.encode())}


# --------------------------------------------------------------------------- SQLite logging hooks
def _lg():
    """The log store, or None if it cannot be loaded. Logging never breaks a core function."""
    try:
        try:
            from . import logdb
        except ImportError:
            import logdb  # type: ignore
        return logdb
    except Exception:  # noqa: BLE001
        return None


def _wrap_server(fn, action):
    def w(name, port=None):
        try:
            r = fn(name, port)
        except Exception as ex:  # noqa: BLE001
            lg = _lg()
            if lg:
                lg.error_from_exception("server", ex, context={"name": name, "port": port, "action": action})
            raise
        lg = _lg()
        if lg:
            ok = "error" not in r and (r.get("up") if action == "start" else r.get("stopped", False) or not r.get("up"))
            lg.server_event(name, action, r.get("port"), None, bool(ok), r.get("output") or r.get("note") or "")
            if not ok:
                lg.error("server", "%s %s failed" % (action, name), "server_" + action + "_failed", None, context=r)
        return r
    w.__name__, w.__doc__ = fn.__name__, fn.__doc__
    return w


def _wrap_launch(fn):
    def w(id, confirm=False, dry_run=False):
        r = fn(id, confirm, dry_run)
        lg = _lg()
        if lg:
            if r.get("launched"):
                lg.event("mcp", "info", "queue_launch", "launched queue item %s" % id, data={"command": r.get("command")})
            elif r.get("refused"):
                lg.event("mcp", "warn", "queue_launch_refused", "refused %s: %s" % (id, r["refused"]), data={"dry_run": dry_run})
            elif not r.get("dry_run"):
                lg.event("mcp", "error", "queue_launch_failed", "launch of %s rc=%s" % (id, r.get("rc")), data={"command": r.get("command")})
                lg.error("launcher", "launch of queue item %s exited rc=%s" % (id, r.get("rc")), "launch_failed", None,
                         context={"output": r.get("output"), "command": r.get("command")})
        return r
    w.__name__, w.__doc__ = fn.__name__, fn.__doc__
    return w


start_server = _wrap_server(start_server, "start")
stop_server = _wrap_server(stop_server, "stop")
launch_queue_item = _wrap_launch(launch_queue_item)
_brief_base = brief


def brief(max_chars: int = 4000) -> dict:
    """One-call overview, now including the SQLite unresolved-error count and last 3 errors."""
    lg = _lg()
    extra = ""
    if lg:
        try:
            lg.ingest(max_seconds=6)
            n = lg.unresolved_count()
            last = lg.last_errors(3)
            extra = "\nERRORS unresolved: %d%s" % (n, "".join("\n  #%s %s %s: %s" % (e["id"], e["source"], e["exc_type"], (e["message"] or "")[:80]) for e in last))
        except Exception as ex:  # noqa: BLE001
            extra = "\nERRORS: log store unavailable (%r)" % ex
    inc = ""
    try:
        try:
            from . import watchdog as _wd
        except ImportError:
            import watchdog as _wd  # type: ignore
        ops = _wd.incidents(open_only=True, limit=4)
        w = _wd.status()
        inc = "OPEN INCIDENTS: %d (watchdog %s)%s\n" % (
            ops["open_total"], "healthy" if w["healthy"] else "NOT RUNNING: start with gogeta_watchdog",
            "".join("\n  %s %s %s [%s] %s" % (i["id"], i["severity"], i["kind"], i["category"], i["summary"][:70]) for i in ops["incidents"]))
    except Exception as ex:  # noqa: BLE001
        inc = "INCIDENTS: unavailable (%r)\n" % ex
    try:
        bl = backlog_top(3)
        if bl:
            inc += "COCKPIT BACKLOG: " + " | ".join(bl) + "\n"
    except Exception:  # noqa: BLE001
        pass
    b = _brief_base(max_chars=max_chars - len(extra) - len(inc) - 5)
    text = redact(inc + b["text"] + extra)
    return {"text": _cap(text, max_chars), "bytes": len(text.encode())}


_brief_prev = brief


def brief(max_chars: int = 4000) -> dict:
    """One-call overview, now also listing finished runs that have no filled run report form (every finished run gets one)."""
    line = ""
    try:
        try:
            from . import runform as _rf
        except ImportError:
            import runform as _rf  # type: ignore
        miss = _rf.missing_forms()
        if miss:
            line = "\nRUN FORMS missing for %d finished run(s): %s. Fill with gogeta_run_form_draft -> write narrative -> gogeta_run_form_submit." % (
                len(miss), ", ".join(m[:28] for m in miss[:8]) + (" ..." if len(miss) > 8 else ""))
    except Exception as ex:  # noqa: BLE001
        line = "\nRUN FORMS: unavailable (%r)" % ex
    b = _brief_prev(max_chars=max_chars - len(line) - 5)
    text = redact(b["text"] + line)
    return {"text": _cap(text, max_chars), "bytes": len(text.encode())}


def backlog_top(n: int = 5) -> list:
    """Top open items from cockpit/BACKLOG.md (lines starting with '- [ ]')."""
    items = [l[5:].strip() for l in _read(REPO / "cockpit/BACKLOG.md").splitlines() if l.startswith("- [ ]")]
    return [i[:90] for i in items[:n]]


def backlog() -> dict:
    return {"text": _cap(_read(REPO / "cockpit/BACKLOG.md"), 8000), "top": backlog_top(10)}
