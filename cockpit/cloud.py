"""Gogeta-0 cockpit: Vast.ai cloud GPUs with spending guardrails. Stdlib only.

Rules (AGENTS 0a: spending money needs the human):
  * Read-only, free to call: offers(), instances(), cost_estimate(), balance_ok(), approvals().
  * rent() SPENDS MONEY. It is refused unless a human-created, unexpired approval in
    ~/fusiondance-data/cockpit/spend-approvals.json covers it: the GPU is allowed, dph <= max_usd_per_hour, the active
    instance count stays <= max_instances and dph x max_hours fits the approval's remaining budget. All of this is
    checked BEFORE any `vastai create instance` call (and re-checked under the approvals lock); rent() is the only
    function in the cockpit that builds a `create instance` command.
  * Approvals are created ONLY by the human: the TUI Cloud tab form (typed confirmation) or
    `bash cockpit/cockpit.sh --approve-spend ...` on a terminal (refused without a TTY). The MCP server exposes NO
    tool that creates, extends or edits an approval (asserted in test_core.py). The cockpit itself only writes its
    own bookkeeping into an approval (spend reservations / settlements).
  * destroy() is always allowed (stopping spending is safe) but only for instances the cockpit launched: the
    instance must be in the cockpit registry with a `gogeta-cockpit-*` label, and the live label must match.
  * Every state-changing call has dry_run, which prints the exact command and its cost and executes nothing.
The API key is never read, printed, passed or stored here: vastai reads it from its own config. The CLI used is the
Windows exe through WSL interop (so the key stays in the Windows config only); override with $GOGETA_VASTAI (tests).
Instance ids, ssh hosts and billing data live only in the cockpit SQLite (data, never committed).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    from . import core, logdb, providers
except ImportError:  # script mode (cockpit/ on sys.path)
    import core  # type: ignore
    import logdb  # type: ignore
    import providers  # type: ignore
_pv = providers.vast

PROVIDER = "vast.ai"
WIN_EXE = _pv.WIN_EXE
LABEL_PREFIX = "gogeta-cockpit"
MAX_APPROVAL_H = 24.0
APPROVAL_SANITY = {"max_usd_total": 500.0, "max_usd_per_hour": 50.0, "max_instances": 16}
# Official PyTorch image, pinned by tag (CUDA 13.0 to match the tested torch 2.14.1+cu130; devel = has gcc for Triton).
# NOT verified to exist on Docker Hub from here: check the tag before the first real rent.
DEFAULT_IMAGE = "pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel"
DEFAULT_DISK_GB = 40
ACTIVE_STATES = ("provisioning", "setup", "ready", "busy", "pulling", "idle", "destroying")
_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9._-]+$")
_GPU_RE = re.compile(r"^[A-Za-z0-9 _.-]{2,40}$")
_ID_RE = re.compile(r"^\d{1,12}$")


def _now() -> float:
    return time.time()


def _iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_now() if t is None else t))


def _epoch(iso: str) -> float:
    return time.mktime(time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S"))


def cdir() -> Path:
    p = Path(os.environ.get("GOGETA_COCKPIT_DIR") or core.DATA / "cockpit")
    p.mkdir(parents=True, exist_ok=True)
    return p


def approvals_path() -> Path:
    return cdir() / "spend-approvals.json"


@contextmanager
def _lock(name: str):
    fd = open(cdir() / ("%s.lock" % name), "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _event(level: str, kind: str, msg: str, data=None) -> None:
    logdb.event("cloud", level, kind, msg, data=data)


# --------------------------------------------------------------------------- providers (cockpit/providers/: vast, ssh, ...)
vast_argv = _pv.vast_argv
_vast = _pv.run  # the raw vastai runner (refuses --api-key; redacts output)


def cli_available() -> bool:
    return providers.get("vast").available()


def _P(name: str = "vast"):
    return providers.get(name or "vast")


# --------------------------------------------------------------------------- approvals (human-created only)
def _load_approvals() -> list:
    try:
        d = json.loads(approvals_path().read_text(encoding="utf-8"))
        return d.get("approvals", []) if isinstance(d, dict) else []
    except (OSError, ValueError):
        return []


def _save_approvals(lst: list) -> None:
    p = approvals_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": 1, "note": "Spend approvals. Created ONLY by the human (TUI Cloud tab or "
                               "cockpit.sh --approve-spend). The cockpit writes only reservations/used_usd.",
                               "approvals": lst}, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def _used(a: dict) -> float:
    return round(sum(float(r.get("usd") or 0) for r in a.get("reservations", [])), 4)


def _approval_problems(a: dict, now: float | None = None) -> list:
    now = _now() if now is None else now
    bad = []
    if a.get("created_by") != "human" or a.get("created_via") not in ("tui", "cli"):
        bad.append("approval %s was not created by the human (TUI/CLI)" % a.get("id"))
    try:
        c, e = _epoch(a["created_at"]), _epoch(a["expires_at"])
    except (KeyError, ValueError, TypeError):
        return bad + ["approval %s has unreadable dates" % a.get("id")]
    if e - c > MAX_APPROVAL_H * 3600 + 60:
        bad.append("approval %s lasts more than %d h" % (a.get("id"), MAX_APPROVAL_H))
    if now >= e:
        bad.append("approval %s expired at %s" % (a.get("id"), a["expires_at"]))
    if a.get("revoked_at"):
        bad.append("approval %s was revoked at %s" % (a.get("id"), a["revoked_at"]))
    return bad


def create_approval(max_usd_total: float, max_usd_per_hour: float, max_instances: int, allowed_gpus: list,
                    purpose: str, hours: float = 12.0, created_via: str = "cli") -> dict:
    """HUMAN ONLY. Called by the TUI 'Approve budget' form and `cockpit.sh --approve-spend` (TTY). Never exposed via MCP."""
    if created_via not in ("tui", "cli"):
        return {"created": False, "refused": "approvals are created only from the TUI or the CLI by the human"}
    errs = []
    try:
        tot, ph, n, h = float(max_usd_total), float(max_usd_per_hour), int(max_instances), float(hours)
    except (TypeError, ValueError):
        return {"created": False, "refused": "numbers required for max_usd_total, max_usd_per_hour, max_instances, hours"}
    gpus = [g.strip() for g in (allowed_gpus.split(",") if isinstance(allowed_gpus, str) else allowed_gpus or []) if g and g.strip()]
    if not 0 < tot <= APPROVAL_SANITY["max_usd_total"]:
        errs.append("max_usd_total must be in (0, %s]" % APPROVAL_SANITY["max_usd_total"])
    if not 0 < ph <= APPROVAL_SANITY["max_usd_per_hour"]:
        errs.append("max_usd_per_hour must be in (0, %s]" % APPROVAL_SANITY["max_usd_per_hour"])
    if not 1 <= n <= APPROVAL_SANITY["max_instances"]:
        errs.append("max_instances must be 1..%d" % APPROVAL_SANITY["max_instances"])
    if not 0 < h <= MAX_APPROVAL_H:
        errs.append("hours must be in (0, 24]")
    if not gpus or not all(_GPU_RE.match(g) for g in gpus):
        errs.append("allowed_gpus: one or more GPU names such as 'RTX 3090'")
    if len((purpose or "").strip()) < 8:
        errs.append("purpose: say what the money is for (>= 8 chars)")
    if errs:
        return {"created": False, "refused": "; ".join(errs)}
    now = _now()
    a = {"id": "APR-%s-%s" % (time.strftime("%Y%m%d-%H%M", time.localtime(now)), uuid.uuid4().hex[:4]),
         "created_by": "human", "created_via": created_via, "created_at": _iso(now), "expires_at": _iso(now + h * 3600),
         "max_usd_total": round(tot, 2), "max_usd_per_hour": round(ph, 4), "max_instances": n, "allowed_gpus": gpus,
         "purpose": core.redact(purpose.strip())[:300], "used_usd": 0.0, "reservations": [], "revoked_at": None}
    with _lock("approvals"):
        lst = _load_approvals()
        lst.append(a)
        _save_approvals(lst)
    _event("info", "approval_created", "human approved %s: $%.2f total, $%.3f/h, %d instance(s), %s, %.1f h (%s)" % (
        a["id"], tot, ph, n, ",".join(gpus), h, created_via))
    return {"created": True, "approval": a}


def revoke_approval(approval_id: str, created_via: str = "cli") -> dict:
    """HUMAN ONLY (TUI/CLI). Revoking only reduces what may be spent; running instances under it are then destroyed by the watchdog."""
    if created_via not in ("tui", "cli"):
        return {"revoked": False, "refused": "human only (TUI/CLI)"}
    with _lock("approvals"):
        lst = _load_approvals()
        a = next((x for x in lst if x.get("id") == approval_id), None)
        if not a:
            return {"revoked": False, "refused": "unknown approval %s" % approval_id}
        a["revoked_at"] = _iso()
        _save_approvals(lst)
    _event("warn", "approval_revoked", "human revoked %s" % approval_id)
    return {"revoked": True, "id": approval_id}


def _approval_view(a: dict, now: float) -> dict:
    act = [r for r in _inst_rows(active_only=True) if r["approval_id"] == a["id"]]
    spent = round(sum(inst_cost(r, now) for r in _inst_rows() if r["approval_id"] == a["id"]), 4)
    probs = _approval_problems(a, now)
    used = _used(a)
    return {"id": a["id"], "valid": not probs, "problems": probs, "purpose": a.get("purpose"), "created_at": a.get("created_at"),
            "expires_at": a.get("expires_at"), "expires_in_h": round(max(0.0, (_epoch(a["expires_at"]) - now) / 3600), 2) if a.get("expires_at") else None,
            "max_usd_total": a.get("max_usd_total"), "max_usd_per_hour": a.get("max_usd_per_hour"), "max_instances": a.get("max_instances"),
            "allowed_gpus": a.get("allowed_gpus"), "reserved_usd": used, "remaining_usd": round(float(a.get("max_usd_total") or 0) - used, 4),
            "spent_actual_usd": spent, "active_instances": len(act), "created_via": a.get("created_via")}


def approvals(include_inactive: bool = False) -> dict:
    """READ-ONLY. Spend approvals (human-created): caps, reserved and actually spent $, remaining budget, expiry, active instances."""
    now = _now()
    rows = [_approval_view(a, now) for a in _load_approvals()]
    if not include_inactive:
        rows = [r for r in rows if r["valid"] or r["active_instances"]]
    return {"count": len(rows), "approvals": rows, "file": str(approvals_path()),
            "how_to_create": "human only: TUI Cloud tab > Approve budget, or bash cockpit/cockpit.sh --approve-spend (no MCP tool can)"}


def check_spend(approval_id: str, gpu_name: str, dph: float, max_hours: float, n_new: int = 1, max_usd: float | None = None) -> dict:
    """The rent guard. ok only if a valid human approval covers gpu, dph, instance count and dph*max_hours*n_new."""
    now = _now()
    lst = _load_approvals()
    cands = [a for a in lst if a.get("id") == approval_id] if approval_id else lst
    if approval_id and not cands:
        return {"ok": False, "reasons": ["unknown approval %s" % approval_id], "approval": None}
    if not cands:
        return {"ok": False, "reasons": ["no spend approval exists: the human must create one (TUI Cloud tab or cockpit.sh --approve-spend)"], "approval": None}
    projected = round(float(dph) * float(max_hours) * int(n_new), 4)
    all_reasons = []
    for a in cands:
        r = list(_approval_problems(a, now))
        if not (0 < float(max_hours) <= MAX_APPROVAL_H):
            r.append("max_hours must be in (0, 24]")
        if gpu_name.strip().lower() not in [g.lower() for g in a.get("allowed_gpus", [])]:
            r.append("GPU %r not allowed by %s (allowed: %s)" % (gpu_name, a["id"], ", ".join(a.get("allowed_gpus", []))))
        if float(dph) > float(a.get("max_usd_per_hour") or 0) + 1e-9:
            r.append("dph $%.4f > per-hour cap $%.4f of %s" % (float(dph), float(a.get("max_usd_per_hour") or 0), a["id"]))
        active = len([x for x in _inst_rows(active_only=True) if x["approval_id"] == a["id"]])
        if active + int(n_new) > int(a.get("max_instances") or 0):
            r.append("%d active + %d new > max_instances %s of %s" % (active, int(n_new), a.get("max_instances"), a["id"]))
        remaining = float(a.get("max_usd_total") or 0) - _used(a)
        if projected > remaining + 1e-9:
            r.append("projected $%.2f (dph x max_hours x n) > remaining $%.2f of %s" % (projected, remaining, a["id"]))
        if max_usd is not None and float(max_usd) > 0 and projected > float(max_usd) + 1e-9:
            r.append("projected $%.2f > the call's own max_usd $%.2f" % (projected, float(max_usd)))
        if not r:
            return {"ok": True, "reasons": [], "approval": a["id"], "projected_usd": projected, "remaining_after": round(remaining - projected, 4)}
        all_reasons += r
    return {"ok": False, "reasons": all_reasons, "approval": cands[0].get("id") if len(cands) == 1 else None, "projected_usd": projected}


def _reserve(approval_id: str, key: str, usd: float, meta: dict) -> None:
    """Internal bookkeeping only (spend reservation). Caller holds the approvals lock."""
    lst = _load_approvals()
    a = next(x for x in lst if x["id"] == approval_id)
    a.setdefault("reservations", []).append({"key": key, "usd": round(usd, 4), "ts": _iso(), "settled": False, **meta})
    a["used_usd"] = _used(a)
    _save_approvals(lst)


def _rekey(approval_id: str, old: str, new: str) -> None:
    with _lock("approvals"):
        lst = _load_approvals()
        for a in lst:
            if a.get("id") == approval_id:
                for r in a.get("reservations", []):
                    if r.get("key") == old:
                        r["key"] = new
        _save_approvals(lst)


def _settle(approval_id: str, key: str, actual_usd: float | None) -> None:
    """Internal bookkeeping: replace a reservation by the actual cost (None = drop it: nothing was rented)."""
    with _lock("approvals"):
        lst = _load_approvals()
        for a in lst:
            if a.get("id") != approval_id:
                continue
            res = a.get("reservations", [])
            for r in list(res):
                if r.get("key") == key and not r.get("settled"):
                    if actual_usd is None:
                        res.remove(r)
                    else:
                        r.update(usd=round(max(0.0, actual_usd), 4), settled=True, settled_ts=_iso())
            a["used_usd"] = _used(a)
        _save_approvals(lst)


# --------------------------------------------------------------------------- registry (SQLite, never committed)
def _db():
    return logdb.connect()


def _inst_rows(active_only: bool = False, session_id: str = "") -> list:
    con = _db()
    try:
        q, a = "SELECT * FROM cloud_instances", []
        w = []
        if active_only:
            w.append("state IN (%s)" % ",".join("?" * len(ACTIVE_STATES)))
            a += list(ACTIVE_STATES)
        if session_id:
            w.append("session_id=?")
            a.append(session_id)
        if w:
            q += " WHERE " + " AND ".join(w)
        return [dict(r) for r in con.execute(q + " ORDER BY created_epoch", a)]
    finally:
        con.close()


def inst(instance_id) -> dict | None:
    con = _db()
    try:
        r = con.execute("SELECT * FROM cloud_instances WHERE id=?", (str(instance_id),)).fetchone()
        return dict(r) if r else None
    finally:
        con.close()


def inst_update(instance_id, **f) -> None:
    if not f:
        return
    con = _db()
    try:
        con.execute("UPDATE cloud_instances SET %s WHERE id=?" % ",".join("%s=?" % k for k in f), (*f.values(), str(instance_id)))
        con.commit()
    finally:
        con.close()


def set_state(instance_id, state: str, **f) -> None:
    inst_update(instance_id, state=state, state_since=_now(), **f)


def _inst_insert(**f) -> None:
    con = _db()
    try:
        con.execute("INSERT INTO cloud_instances(%s) VALUES(%s)" % (",".join(f), ",".join("?" * len(f))), tuple(f.values()))
        con.commit()
    finally:
        con.close()


def inst_cost(row: dict, now: float | None = None) -> float:
    """Billed estimate: dph x hours from the create call to destroy (or now)."""
    if not row.get("created_epoch") or row.get("state") == "dry":
        return 0.0
    end = row.get("destroyed_epoch") or (_now() if now is None else now)
    return round(float(row.get("dph") or 0) * max(0.0, end - row["created_epoch"]) / 3600, 4)


# --------------------------------------------------------------------------- read-only tools
def offers(gpu: str = "RTX 3090", num_gpus: int = 1, min_reliability: float = 0.98, verified: bool = True, max_dph: float = 0.0,
           sort: str = "dph_total", limit: int = 20, disk_gb: int = DEFAULT_DISK_GB, min_cuda: float = 0.0, provider: str = "vast") -> dict:
    """READ-ONLY. Live on-demand offers (provider: vast | ssh | ...) for one GPU name (prices include disk_gb of storage). Returns cheapest first plus count/min/median $/h."""
    if not _GPU_RE.match(gpu or ""):
        return {"error": "bad gpu name %r" % gpu}
    if sort not in ("dph_total", "dph_total-", "reliability2-", "dlperf_per_dphtotal-", "inet_down-", "score-"):
        return {"error": "sort must be one of dph_total, dph_total-, reliability2-, dlperf_per_dphtotal-, inet_down-, score-"}
    spec = {"gpu": gpu, "num_gpus": int(num_gpus), "min_reliability": float(min_reliability), "verified": bool(verified), "max_dph": float(max_dph or 0),
            "sort": sort, "limit": int(limit), "disk_gb": int(disk_gb), "min_cuda": float(min_cuda or 0)}
    try:
        res = _P(provider).search_offers(spec)
    except ValueError as ex:
        return {"error": str(ex)}
    if isinstance(res, dict):
        return res
    rows = [o for o in res if str(o.get("gpu_name", "")).lower() == gpu.strip().lower()]
    prices = [r["dph_total"] for r in rows if r["dph_total"] is not None]
    return {"provider": provider, "query": spec, "count": len(rows), "min_dph": min(prices) if prices else None,
            "median_dph": round(statistics.median(prices), 4) if prices else None, "disk_gb_priced": int(disk_gb), "offers": rows,
            "note": "on-demand prices; interruptible is cheaper but stays off until mid-run resume lands (notes/compute-rental.md)"}


def _offer_by_id(offer_id, disk_gb: int, provider: str = "vast") -> dict | None:
    return _P(provider).offer(offer_id, int(disk_gb))


def cost_estimate(gpu: str = "RTX 3090", hours: float = 1.0, count: int = 1, disk_gb: int = DEFAULT_DISK_GB, plan: dict | None = None) -> dict:
    """READ-ONLY. hours x dph x count at the live MIN and MEDIAN on-demand price. plan={"gpu","hours","count","disk_gb"} overrides."""
    if plan:
        gpu, hours, count, disk_gb = plan.get("gpu", gpu), plan.get("hours", hours), plan.get("count", count), plan.get("disk_gb", disk_gb)
    o = offers(gpu, 1, 0.98, True, 0.0, "dph_total", 300, int(disk_gb))
    if o.get("error"):
        return o
    if not o["count"]:
        return {"gpu": gpu, "error": "no live offers for %s" % gpu}
    h, n = float(hours), int(count)
    return {"gpu": gpu, "hours": h, "count": n, "disk_gb": int(disk_gb), "offers_seen": o["count"],
            "min_dph": o["min_dph"], "median_dph": o["median_dph"],
            "total_at_min_usd": round(o["min_dph"] * h * n, 2), "total_at_median_usd": round(o["median_dph"] * h * n, 2),
            "available_at_or_below_median": sum(1 for x in o["offers"] if x["dph_total"] <= o["median_dph"]),
            "formula": "hours x dph x count (live on-demand prices incl. %d GB storage); setup time is billed too" % int(disk_gb)}


def _live_instances(provider: str = "vast") -> list | None:
    """Normalised live listing of one provider (see cockpit/providers/__init__.py), tagged with the provider name."""
    try:
        P = _P(provider)
    except ValueError:
        return None
    if not P.available():
        return [] if not P.spends_money else None
    data = P.instances()
    if data is None:
        return None
    for x in data:
        x["provider"] = P.name
    if P.name == "vast":
        _write_cache(data)
    return data


def live_all() -> list | None:
    """Every provider's live listing combined (None only if the Vast listing is unavailable)."""
    out, ok = [], True
    for n in providers.names():
        L = _live_instances(n)
        if L is None:
            ok = ok and n != "vast"
            continue
        out += L
    return out if ok else (out or None)


def _write_cache(live: list) -> None:
    mine = {r["id"] for r in _inst_rows()}
    foreign = [x for x in live if str(x.get("id")) not in mine]
    c = {"epoch": _now(), "total": len(live), "foreign": len(foreign),
         "foreign_dph": round(sum(float(x.get("dph_total") or 0) for x in foreign), 4)}
    try:
        (cdir() / "cloud-live-cache.json").write_text(json.dumps(c))
    except OSError:
        pass


def _read_cache() -> dict:
    try:
        return json.loads((cdir() / "cloud-live-cache.json").read_text())
    except (OSError, ValueError):
        return {}


def sync_live(live: list | None = None) -> list | None:
    """Fold the live listings into the registry: ssh host/port, status, driver, display metadata; mark vanished instances destroyed."""
    live = live_all() if live is None else live
    if live is None:
        return None
    by = {(x.get("provider", "vast"), str(x.get("id"))): x for x in live}
    seen = {x.get("provider", "vast") for x in live} | {"vast"}
    now = _now()
    for r in _inst_rows(active_only=True):
        prov = r.get("provider") or "vast"
        L = by.get((prov, r["id"]))
        if L is None:
            if prov in seen and now - (r["created_epoch"] or now) > 120:
                _mark_destroyed(r, "vanished from the account listing")
            continue
        if (L.get("label") or "") != r["label"]:
            continue  # label mismatch: never act on it
        f = {"live_status": str(L.get("actual_status") or "")[:40], "meta": json.dumps(live_meta(L))}
        h, p = live_endpoint(L)
        if h and p:
            f.update(ssh_host=h, ssh_port=p)
        if L.get("driver_version"):
            f["driver"] = str(L["driver_version"])
        inst_update(r["id"], **f)
    return live


def live_endpoint(L: dict) -> tuple:
    """(host, port) from a normalised instance (providers choose the right endpoint; Vast: direct public IP + mapped port 22)."""
    try:
        return (str(L["ssh_host"]), int(L["ssh_port"])) if L.get("ssh_host") and L.get("ssh_port") else (None, None)
    except (TypeError, ValueError):
        return None, None


def live_meta(L: dict) -> dict:
    """Display metadata (allowlisted by the provider). Holds the IP/port for the TUI and MCP status only: SQLite, never committed."""
    return dict(L.get("meta") or {})


def instances() -> dict:
    """READ-ONLY. Live account instances. Cockpit-launched ones (label gogeta-cockpit-*) with state, job, cost so far; others are listed as NOT MANAGED (they still cost money)."""
    live = sync_live()
    now = _now()
    mine = {(r.get("provider") or "vast", r["id"]): r for r in _inst_rows()}
    rows, foreign = [], []
    for x in live or []:
        iid = str(x.get("id"))
        r = mine.get((x.get("provider", "vast"), iid))
        if r and (x.get("label") or "") == r["label"]:
            rows.append({"id": iid, "provider": r.get("provider") or "vast", "label": r["label"], "state": r["state"], "live_status": x.get("actual_status"), "gpu": r["gpu_name"],
                         "dph": r["dph"], "hours": round((now - r["created_epoch"]) / 3600, 3), "cost_usd": inst_cost(r, now),
                         "max_hours": r["max_hours"], "job": r["job_id"], "approval": r["approval_id"],
                         "ssh": "%s:%s" % (r["ssh_host"], r["ssh_port"]) if r["ssh_host"] else None})
        else:
            foreign.append({"id": iid, "provider": x.get("provider", "vast"), "label": x.get("label"), "gpu": x.get("gpu_name"),
                            "dph": core._f(x.get("dph_total"), 4), "status": x.get("actual_status"), "managed": False})
    return {"available": live is not None, "cockpit": rows, "not_managed": foreign,
            "burn_usd_per_h": round(sum(r["dph"] or 0 for r in rows) + sum(f["dph"] or 0 for f in foreign), 4),
            "note": "the cockpit only ever manages/destroys its own (gogeta-cockpit-*) instances" if live is not None else
                    "vastai listing unavailable; registry only", "registry_active": len(_inst_rows(active_only=True))}


def balance_ok(low_usd: float = 5.0, provider: str = "vast") -> dict:
    """READ-ONLY. Account credit as sufficient / low / unknown only (never the amount, never the account e-mail)."""
    return {"credit": _P(provider).balance(float(low_usd)), "threshold_usd": float(low_usd), "provider": provider}


# --------------------------------------------------------------------------- state-changing: rent / destroy
def _validate_rent(offer_id, max_hours, purpose, image, disk_gb, provider: str = "vast") -> list:
    e = []
    if provider == "vast" and not _ID_RE.match(str(offer_id)) or not re.match(r"^[A-Za-z0-9_.:-]{1,64}$", str(offer_id)):
        e.append("offer_id must be a number")
    try:
        if not 0 < float(max_hours) <= MAX_APPROVAL_H:
            e.append("max_hours must be in (0, 24]")
    except (TypeError, ValueError):
        e.append("max_hours must be a number")
    if len((purpose or "").strip()) < 8:
        e.append("purpose is required (>= 8 chars); it is logged with the call")
    if not _IMAGE_RE.match(image or "") or (image or "").endswith(":latest"):
        e.append("image must be pinned by an explicit tag (not :latest), e.g. %s" % DEFAULT_IMAGE)
    if not 10 <= int(disk_gb) <= 500:
        e.append("disk_gb must be 10..500")
    return e


def rent(offer_id: int, max_hours: float, max_usd: float, purpose: str, image: str = DEFAULT_IMAGE, disk_gb: int = DEFAULT_DISK_GB,
         approval_id: str = "", confirm: bool = False, dry_run: bool = False, session_id: str = "", job_id: str = "", provider: str = "vast") -> dict:
    """STATE-CHANGING, SPENDS MONEY (provider vast | ssh | ...; ssh = bring-your-own, no spend). Rent one on-demand offer. Refused unless a human approval covers the GPU, dph <= per-hour cap,
    active+1 <= max_instances and dph*max_hours fits the remaining budget (and max_usd). All checked BEFORE `create instance`.
    dry_run=true prints the exact command and cost. Needs confirm=true. Every call is logged with its purpose."""
    errs = _validate_rent(offer_id, max_hours, purpose, image, disk_gb, provider)
    if errs:
        _event("warn", "rent_refused", "rent offer %s refused: %s" % (offer_id, "; ".join(errs)), {"purpose": purpose})
        return {"rented": False, "refused": "; ".join(errs)}
    try:
        P = _P(provider)
    except ValueError as ex:
        return {"rented": False, "refused": str(ex)}
    off = _offer_by_id(offer_id, int(disk_gb), P.name)
    if not off:
        return {"rented": False, "refused": "offer %s not found or no longer rentable" % offer_id}
    gpu, dph = str(off.get("gpu_name")), float(off.get("dph_total") or 0)
    chk = check_spend(approval_id, gpu, dph, float(max_hours), 1, max_usd) if P.spends_money else \
        {"ok": True, "reasons": [], "approval": approval_id or None, "projected_usd": 0.0, "remaining_after": None}
    label = "%s-%s" % (LABEL_PREFIX, uuid.uuid4().hex[:10])
    argv = P.rent_command(off, label, int(disk_gb), image)
    plan = {"offer_id": int(offer_id), "gpu": gpu, "dph": round(dph, 4), "max_hours": float(max_hours),
            "projected_usd": round(dph * float(max_hours), 4), "approval": chk.get("approval"), "purpose": purpose,
            "command": ("vastai " if P.name == "vast" else P.name + " ") + " ".join(shlex.quote(a) for a in argv), "on_demand": True, "provider": P.name,
            "machine_id": off.get("machine_id"), "driver": off.get("driver_version")}
    if not chk["ok"]:
        _event("warn", "rent_refused", "rent offer %s (%s $%.3f/h) refused: %s" % (offer_id, gpu, dph, "; ".join(chk["reasons"])),
               {"purpose": purpose, "dry_run": dry_run})
        return {"rented": False, "refused": "; ".join(chk["reasons"]), **plan}
    if dry_run:
        return {"rented": False, "dry_run": True, **plan, "remaining_after": chk.get("remaining_after"),
                "would": "reserve $%.2f on %s, then run the command above" % (plan["projected_usd"], chk["approval"])}
    if confirm is not True:
        return {"rented": False, "refused": "confirm must be exactly true (renting spends money)", **plan}
    aid = chk["approval"]
    if P.spends_money:
        with _lock("approvals"):  # re-check under the lock, then reserve BEFORE the create call
            chk2 = check_spend(aid, gpu, dph, float(max_hours), 1, max_usd)
            if not chk2["ok"]:
                return {"rented": False, "refused": "; ".join(chk2["reasons"]), **plan}
            _reserve(aid, label, plan["projected_usd"], {"gpu": gpu, "dph": round(dph, 4), "max_hours": float(max_hours)})
    res = P.rent(off, label, int(disk_gb), image)  # the ONLY call that creates an instance; it is reached only after the checks above
    iid = str(res["id"]) if res.get("ok") and res.get("id") else None
    if not iid:
        if P.spends_money:
            _settle(aid, label, None)
        logdb.error("cloud", "%s rent failed for offer %s: %s" % (P.name, offer_id, str(res.get("error"))[:300]), "RentFailed", None,
                    context={"offer": offer_id, "purpose": purpose})
        return {"rented": False, "error": str(res.get("error"))[:300], **plan}
    if P.spends_money:
        _rekey(aid, label, iid)
    now = _now()
    _inst_insert(id=iid, label=label, approval_id=aid, session_id=session_id or None, offer_id=str(offer_id),
                 machine_id=str(off.get("machine_id") or ""), gpu_name=gpu, num_gpus=int(off.get("num_gpus") or 1), dph=round(dph, 6),
                 max_hours=float(max_hours), reserved_usd=plan["projected_usd"], purpose=core.redact(purpose)[:300], image=image,
                 disk_gb=int(disk_gb), created=_iso(now), created_epoch=now, state="provisioning", state_since=now,
                 driver=str(off.get("driver_version") or ""), job_id=job_id or None, provider=P.name)
    _event("warn", "rent", ("RENTED %s on " + P.name + " (%s, $%.4f/h, max %.1f h, reserved $%.2f on %s) purpose: %s") % (
        label, gpu, dph, float(max_hours), plan["projected_usd"], aid, purpose[:160]), {"instance": iid, "session": session_id, "job": job_id})
    return {"rented": True, "instance_id": iid, "label": label, **plan}


def _mark_destroyed(row: dict, reason: str) -> dict:
    now = _now()
    row = dict(row, destroyed_epoch=now)
    cost = inst_cost(row, now)
    inst_update(row["id"], state="destroyed", state_since=now, destroyed_epoch=now, destroy_reason=reason[:200], cost_usd=cost)
    if row.get("approval_id"):
        _settle(row["approval_id"], row["id"], cost)
    return {"cost_usd": cost}


def destroy(instance_id: str, confirm: bool = False, dry_run: bool = False, reason: str = "manual") -> dict:
    """STATE-CHANGING (stops spending; always allowed). Destroy ONE cockpit-launched instance (registry + gogeta-cockpit-* label,
    live label must match). Foreign instances are never touched. dry_run=true shows the command and cost so far. Needs confirm=true."""
    iid = str(instance_id).strip()
    r = inst(iid)
    if not r or not str(r.get("label") or "").startswith(LABEL_PREFIX):
        _event("warn", "destroy_refused", "destroy %s refused: not launched by the cockpit" % iid)
        return {"destroyed": False, "refused": "instance %s was not launched by the cockpit: never touched" % iid}
    if r["state"] == "destroyed":
        return {"destroyed": True, "note": "already destroyed", "cost_usd": r["cost_usd"]}
    P = _P(r.get("provider") or "vast")
    live = _live_instances(P.name)
    if live is not None:
        L = next((x for x in live if str(x.get("id")) == iid), None)
        if L is None:
            c = _mark_destroyed(r, "gone before destroy (%s)" % reason)
            return {"destroyed": True, "note": "no longer on the account; registry updated", **c}
        if (L.get("label") or "") != r["label"]:
            return {"destroyed": False, "refused": "live label %r does not match the cockpit's %r: never touched" % (L.get("label"), r["label"])}
    plan = {"instance_id": iid, "label": r["label"], "provider": P.name, "cost_so_far_usd": inst_cost(r), "dph": r.get("dph"), "reason": reason,
            "command": "vastai destroy instance %s -y --raw" % iid if P.name == "vast" else "%s release %s" % (P.name, iid)}
    if dry_run:
        return {"destroyed": False, "dry_run": True, **plan}
    if confirm is not True:
        return {"destroyed": False, "refused": "confirm must be exactly true", **plan}
    inst_update(iid, state="destroying", state_since=_now())
    res = P.destroy(iid)
    if not res.get("ok"):
        inst_update(iid, state=r["state"])
        logdb.error("cloud", "destroy %s failed: %s" % (r["label"], str(res.get("error"))[:300]), "DestroyFailed", None, context={"reason": reason})
        return {"destroyed": False, "error": str(res.get("error"))[:300], **plan}
    c = _mark_destroyed(r, reason)
    _event("warn", "destroy", "DESTROYED %s on %s (%s, $%.4f/h): cost $%.4f; reason: %s" % (r["label"], P.name, r["gpu_name"], float(r.get("dph") or 0),
                                                                              c["cost_usd"], reason[:160]), {"instance": iid, "dph": r.get("dph"), "cost_usd": c["cost_usd"]})
    return {"destroyed": True, **plan, **c}


# --------------------------------------------------------------------------- brief
def brief_lines(live_refresh: bool = True) -> list:
    """Cloud status for the top of brief(): spending must never be silent."""
    now = _now()
    act = _inst_rows(active_only=True)
    cache = _read_cache()
    if live_refresh and cli_available() and not os.environ.get("GOGETA_CLOUD_OFFLINE") and now - cache.get("epoch", 0) > 120:
        try:
            if sync_live() is not None:
                act = _inst_rows(active_only=True)
                cache = _read_cache()
        except Exception:  # noqa: BLE001
            pass
    L = []
    vals = [_approval_view(a, now) for a in _load_approvals()]
    vals = [v for v in vals if v["valid"] or v["active_instances"]]
    if act:
        cost = sum(inst_cost(r, now) for r in act)
        L.append("CLOUD SPENDING: %d cockpit instance(s) up (%s), $%.2f so far, burning $%.3f/h. Stop: gogeta_fleet_down / gogeta_vast_destroy" % (
            len(act), ", ".join(sorted({r["gpu_name"] for r in act})), cost, sum(r["dph"] or 0 for r in act)))
    if cache.get("foreign"):
        L.append("CLOUD: %d NON-cockpit instance(s) on the account, $%.3f/h (not managed by the cockpit; the human must stop them)" % (
            cache["foreign"], cache.get("foreign_dph", 0)))
    for v in vals[:3]:
        L.append("APPROVAL %s: $%.2f reserved / $%.2f cap, $%.2f actually spent, %d/%s instances, expires in %.1f h%s" % (
            v["id"], v["reserved_usd"], v["max_usd_total"], v["spent_actual_usd"], v["active_instances"], v["max_instances"],
            v["expires_in_h"] or 0, "" if v["valid"] else " (INVALID: %s)" % "; ".join(v["problems"])[:80]))
    if not act and not cache.get("foreign"):
        L.append("CLOUD: no instances running%s" % ("" if vals else "; no valid spend approval (renting refused)"))
    return L


# --------------------------------------------------------------------------- human CLI
def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _human_confirm(phrase: str) -> bool:
    print("Type exactly:  %s" % phrase)
    try:
        return input("> ").strip() == phrase
    except EOFError:
        return False


def revoke_ssh_key(key_id: str = "") -> dict:
    """HUMAN ONLY (cockpit.sh --revoke-ssh-key, TTY). Deletes the cockpit's SSH key from the Vast account."""
    pub = Path.home() / ".ssh" / "gogeta_vast.pub"
    want = pub.read_text().split()[1] if pub.exists() and len(pub.read_text().split()) > 1 else ""
    rc, data, text = _vast(["show", "ssh-keys"])
    match = None
    if isinstance(data, list):
        for k in data:
            body = str(k.get("public_key") or k.get("ssh_key") or k.get("key") or "")
            if (want and want in body) or (key_id and str(k.get("id")) == str(key_id)):
                match = str(k.get("id"))
                break
    if not match:
        return {"revoked": False, "refused": "no account SSH key matches ~/.ssh/gogeta_vast.pub%s" % (" or id %s" % key_id if key_id else "")}
    if key_id and str(key_id) != match:
        return {"revoked": False, "refused": "key id %s does not match the cockpit key (%s)" % (key_id, match)}
    print("Vast account SSH key %s matches ~/.ssh/gogeta_vast.pub. Deleting it stops new instances from getting it." % match)
    if not _human_confirm("REVOKE %s" % match):
        return {"revoked": False, "refused": "confirmation not typed"}
    rc, data, text = _vast(["delete", "ssh-key", match])
    ok = rc == 0
    _event("warn" if ok else "error", "ssh_key_revoked" if ok else "ssh_key_revoke_failed", "human revoked Vast ssh key %s rc=%s" % (match, rc))
    return {"revoked": ok, "key_id": match, "output": text[:300]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cockpit.cloud", description="Vast.ai cloud (human CLI). Approvals only from a terminal.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("approve", help="HUMAN: create a spend approval (typed confirmation)")
    a.add_argument("--max-usd", type=float, required=True, help="total budget (USD)")
    a.add_argument("--max-usd-per-hour", type=float, required=True, help="per-instance $/h cap")
    a.add_argument("--max-instances", type=int, default=1)
    a.add_argument("--gpus", default="RTX 3090", help="comma-separated GPU names allowed")
    a.add_argument("--hours", type=float, default=12.0, help="validity, <= 24")
    a.add_argument("--purpose", required=True)
    r = sub.add_parser("revoke", help="HUMAN: revoke an approval")
    r.add_argument("id")
    k = sub.add_parser("revoke-ssh-key", help="HUMAN: delete the cockpit SSH key from the Vast account")
    k.add_argument("--key-id", default="")
    sub.add_parser("approvals")
    o = sub.add_parser("offers")
    o.add_argument("--gpu", default="RTX 3090")
    o.add_argument("--limit", type=int, default=10)
    sub.add_parser("instances")
    sub.add_parser("balance")
    sub.add_parser("brief")
    ns = ap.parse_args(argv)
    if ns.cmd in ("approve", "revoke", "revoke-ssh-key"):
        if not _tty() or os.environ.get("CLAUDECODE") or os.environ.get("GOGETA_AGENT"):
            print("refused: %s is for the human at an interactive terminal (no TTY / agent environment detected)" % ns.cmd, file=sys.stderr)
            return 3
        if ns.cmd == "approve":
            gp = [g.strip() for g in ns.gpus.split(",") if g.strip()]
            print("SPEND APPROVAL (Vast.ai, on-demand)\n  total cap      $%.2f\n  per-hour cap   $%.4f per instance\n  max instances  %d\n"
                  "  GPUs           %s\n  valid for      %.1f h\n  purpose        %s" % (ns.max_usd, ns.max_usd_per_hour, ns.max_instances, ", ".join(gp), ns.hours, ns.purpose))
            if not _human_confirm("APPROVE %.2f" % ns.max_usd):
                print("not approved")
                return 1
            res = create_approval(ns.max_usd, ns.max_usd_per_hour, ns.max_instances, gp, ns.purpose, ns.hours, "cli")
        elif ns.cmd == "revoke":
            res = revoke_approval(ns.id, "cli") if _human_confirm("REVOKE %s" % ns.id) else {"revoked": False}
        else:
            res = revoke_ssh_key(ns.key_id)
    elif ns.cmd == "approvals":
        res = approvals(True)
    elif ns.cmd == "offers":
        res = offers(ns.gpu, limit=ns.limit)
    elif ns.cmd == "instances":
        res = instances()
    elif ns.cmd == "balance":
        res = balance_ok()
    else:
        res = {"lines": brief_lines()}
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
