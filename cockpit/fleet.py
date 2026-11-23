"""Gogeta-0 cockpit: Vast.ai fleet, remote runner, SSH monitoring and control. Stdlib only. Guardrails live in cockpit/cloud.py.

Instance modes (cockpit registry, SQLite; never committed):
  fleet     rented just-in-time by dispatch() for one ready job; reused while jobs wait (same SKU), destroyed at once when none
            do; the watchdog destroys it after 5 min without a job or 15 min stuck in setup (job retried once elsewhere).
  manual    rented by cloud_up()/vast_up() for the step-by-step flow (setup -> data -> run -> pull -> down). AGENTS 0h: when its job
            queue empties it is pulled and destroyed at once; no running or queued job for 10 min -> pulled, then destroyed.
  attached  an instance rented outside the cockpit but labelled gogeta-cockpit-*, registered by vast_attach (confirm). Same rules as manual.
  unmanaged a gogeta-cockpit-* instance on the account but not in the registry (e.g. the pilot): READ-ONLY status, tail and pull.
Every rent goes through cloud.rent() (approval checked before `vastai create instance`). One GPU SKU per fleet session.
Remote runner (per job): rent -> setup (cockpit/remote/bootstrap.sh, detached on the remote, log ~/setup.log: gcc, uv, Python 3.12 venv,
pinned torch/triton/fla/trackio/transformers/einops, clone of the PUBLIC repo at the exact pushed commit, CUDA compile smoke test) ->
upload the dataset by explicit allowlist (manifest files + meta.json) and sha256-verify on arrival -> run the job under a
cockpit wrapper (nohup setsid; ~/fusiondance-data/run/cockpit-job.{info,pid,exit}) -> poll over SSH (nvidia-smi, processes,
redacted log tail; errors/incidents in SQLite) -> pull the run folder (sha256-verified) + a repro/manifest.py manifest with
provider/GPU/driver/dph/cost -> reuse or destroy.
Never sent: local code, ~/.ssh, ~/.config, ~/.vast*, .env*, the cockpit SQLite, the local environment. The private key
~/.ssh/gogeta_vast never leaves this machine (ssh -i ... -o IdentitiesOnly=yes -o ForwardAgent=no). Destroyed instances take
their authorised public key with them (the container is deleted); `cockpit.sh --revoke-ssh-key` removes it from the account.
Test hooks: $GOGETA_SSH (fake ssh), $GOGETA_FLEET_SYNC=1 (run long phases inline instead of detached workers).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    from . import cloud, core, logdb
except ImportError:  # script mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cloud  # type: ignore
    import core  # type: ignore
    import logdb  # type: ignore

REPO_URL = "https://github.com/monstercameron/Gogeta-0.git"  # public; the only code source for remote runs
SSH_KEY = Path.home() / ".ssh" / "gogeta_vast"
R_REPO, R_DATA = "$HOME/Gogeta-0", "$HOME/fusiondance-data"  # /root/... on Vast
IDLE_S = 300                    # fleet: no job for 5 min -> destroy
IDLE_KILL_S = float(os.environ.get("GOGETA_IDLE_KILL_MIN", "10")) * 60  # AGENTS 0h: manual/attached box idle 10 min -> pull, then destroy
SUGGEST_IDLE_S = IDLE_KILL_S
UNREACH_FLAG_S = 300           # no successful poll for 5 min -> incident (destroy at UNREACH_S)
VRAM_NEED_GB = {"E1": 7.1, "E2": 12.0}  # measured peak memory (main agent, 2026-11-22); unknown experiments use DEFAULT_VRAM_GB
DEFAULT_VRAM_GB = 12.0
STUCK_S = 900                   # provisioning / setup > 15 min -> bad host
UNREACH_S = 1200                # busy without a successful poll for 20 min -> destroy (fleet/manual)
POLL_S = float(os.environ.get("GOGETA_FLEET_POLL_S", "60"))
LIVE_SYNC_S = float(os.environ.get("GOGETA_FLEET_LIVE_S", "30"))
HUNG_MIN = 15.0
DATASET_MANIFESTS = {"code-py-v1": "experiments/data_integrity/dataset.manifest.json",
                     "code-py-v1-bytes": "experiments/byte_model/dataset-bytes.manifest.json"}
EXTRA_ALLOWED = ("meta.json",)  # read by gogeta/train.py, not hashed in the manifests: verified against the local sha256
_FORBIDDEN = re.compile(r"(^|/)(\.ssh|\.config|\.vast[^/]*|\.env[^/]*|cockpit|[^/]*\.sqlite[^/]*)(/|$)", re.I)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
_TOKEN = re.compile(r"^[A-Za-z0-9._/=+-]{1,120}$")
_HOST = re.compile(r"^[A-Za-z0-9.-]{1,253}$")
_FLAGS = {"--seed": r"^\d{1,6}$", "--max-tokens": r"^\d{1,13}$", "--lr": r"^\d*\.?\d+(e-?\d+)?$"}
JOB_LIVE = ("provisioning", "setup", "staging", "running", "pulling")


def _now() -> float:
    return cloud._now()


def _event(level, kind, msg, data=None):
    logdb.event("fleet", level, kind, msg, data=data)


def _sync_mode() -> bool:
    return os.environ.get("GOGETA_FLEET_SYNC") == "1"


# --------------------------------------------------------------------------- DB helpers
def _q(sql, args=()):
    con = logdb.connect()
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


def _x(sql, args=()):
    con = logdb.connect()
    try:
        con.execute(sql, args)
        con.commit()
    finally:
        con.close()


def _upd(table, key, **f):
    if f:
        _x("UPDATE %s SET %s WHERE id=?" % (table, ",".join("%s=?" % k for k in f)), (*f.values(), key))


def job(jid):
    r = _q("SELECT * FROM cloud_jobs WHERE id=?", (jid,)) if jid else []
    return r[0] if r else None


def session(sid):
    r = _q("SELECT * FROM cloud_sessions WHERE id=?", (sid,)) if sid else []
    return r[0] if r else None


def jobs(session_id="", status=""):
    w, a = [], []
    if session_id:
        w.append("session_id=?"); a.append(session_id)
    if status:
        w.append("status=?"); a.append(status)
    return _q("SELECT * FROM cloud_jobs" + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY queued_epoch, id", a)


def open_sessions():
    return _q("SELECT * FROM cloud_sessions WHERE status='open' ORDER BY created")


def enabled() -> bool:
    """Anything for the cloud detector to do? (no network when nothing runs or waits)"""
    try:
        return bool(cloud._inst_rows(active_only=True) or _q("SELECT 1 FROM cloud_jobs WHERE status='queued' LIMIT 1"))
    except Exception:  # noqa: BLE001
        return False


def _new_job(launch: str, experiment: str, dataset: str, session_id=None, instance_id=None, status="queued", checkpoint=False) -> str:
    jid = "JOB-%s" % uuid.uuid4().hex[:8]
    _x("INSERT INTO cloud_jobs(id,session_id,launch,experiment,dataset,status,instance_id,attempts,setup_retries,created,queued_epoch,checkpoint)"
       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (jid, session_id, launch, experiment, dataset, status, instance_id, 0, 0, cloud._iso(), _now(), 1 if checkpoint else 0))
    return jid


# --------------------------------------------------------------------------- SSH layer
def _ssh_env() -> dict:
    """Explicit allowlist: nothing from the local environment reaches ssh/tar (no agent socket, tokens or canaries)."""
    return {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "FAKE_SSH_ROOT", "FAKE_SSH_LOG") if k in os.environ}


def ssh_argv(host: str, port) -> list:
    if not _HOST.match(str(host or "")) or not 0 < int(port) < 65536:
        raise ValueError("bad ssh host/port")
    pre = os.environ.get("GOGETA_SSH")
    base = shlex.split(pre) if pre else ["ssh"]
    return base + ["-F", "/dev/null", "-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
                   "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
                   "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-o", "LogLevel=ERROR",
                   "-p", str(int(port)), "root@%s" % host]


def ssh(row: dict, script: str, timeout: int = 120, retries: int = 0) -> tuple[int, str]:
    """Run a script on the instance via `bash -s` (stdin), retrying on rc 255 while sshd comes up (vssh.sh)."""
    try:
        argv = ssh_argv(row.get("ssh_host"), row.get("ssh_port") or 0) + ["bash -s"]
    except (ValueError, TypeError) as ex:
        return 255, "no ssh endpoint yet (%s)" % ex
    rc, out = 255, ""
    for i in range(retries + 1):
        try:
            r = subprocess.run(argv, input=script.encode(), capture_output=True, timeout=timeout, env=_ssh_env())
            rc, out = r.returncode, (r.stdout + r.stderr).decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            rc, out = 124, "ssh timed out after %ss" % timeout
        except OSError as ex:
            return 127, str(ex)
        if rc != 255 or i == retries:
            break
        time.sleep(15)
    return rc, out


def _pipe(src: list, dst: list, timeout: int) -> tuple[int, str]:
    """src | dst without a shell (tar | ssh, ssh | tar)."""
    p1 = p2 = None
    try:
        p1 = subprocess.Popen(src, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_ssh_env())
        p2 = subprocess.Popen(dst, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_ssh_env())
        p1.stdout.close()
        o2, e2 = p2.communicate(timeout=timeout)
        p1.wait(timeout=60)
        e1 = p1.stderr.read()
        p1.stderr.close()
        return (p1.returncode or p2.returncode), (e1 + o2 + e2).decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        for p in (p1, p2):
            if p:
                p.kill()
        return 124, "pipe timed out after %ss" % timeout
    except OSError as ex:
        return 127, str(ex)


# --------------------------------------------------------------------------- instance resolution (registry or live, read-only)
endpoint = cloud.live_endpoint


def resolve(ref: str, live: list | None = None):
    """(row, managed). Registry rows first; else a live gogeta-cockpit-* instance as a read-only 'unmanaged' row; else (None, False)."""
    ref = str(ref or "").strip()
    r = next((x for x in cloud._inst_rows() if ref in (x["id"], x["label"])), None)
    if r and r["state"] != "destroyed":
        return r, True
    live = cloud.live_all() if live is None else live
    L = next((x for x in live or [] if ref in (str(x.get("id")), str(x.get("label") or ""))), None)
    if L and str(L.get("label") or "").startswith(cloud.LABEL_PREFIX):
        h, p = endpoint(L)
        return {"id": str(L.get("id")), "label": L.get("label"), "gpu_name": L.get("gpu_name"), "num_gpus": L.get("num_gpus"),
                "dph": float(L.get("dph_total") or 0), "created_epoch": float(L.get("start_date") or _now()), "state": "unmanaged",
                "mode": "unmanaged", "ssh_host": h, "ssh_port": p, "approval_id": None, "max_hours": None, "driver": L.get("driver_version"),
                "image": L.get("image"), "live_status": L.get("actual_status"), "poll": None, "job_id": None, "session_id": None,
                "meta": json.dumps(cloud.live_meta(L)), "provider": L.get("provider", "vast")}, False
    return (r, True) if r else (None, False)


# --------------------------------------------------------------------------- jobs and the data allowlist
def _dataset_of(config_path: str) -> str:
    try:
        cfg = json.loads(core._read(core.REPO / config_path) or "{}")
    except ValueError:
        cfg = {}
    t = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    return str(t.get("dataset") or cfg.get("dataset") or "code-py-v1")


def parse_job(spec: str) -> dict:
    """Accepted forms (validated token by token; nothing else runs remotely):
      E10 [--seed N] [--max-tokens N] [--lr X] [--no-compile]          -> tools/launch_experiment.sh ... --foreground
      bash tools/launch_experiment.sh E10 [...]  | a queue id (K7)      -> same
      configs/e10_llama_ref.json [--seed N] [--max-tokens N] [--lr X] [--no-compile] -> python -m gogeta.train --config ... --require-clean"""
    s = (spec or "").strip()
    if re.fullmatch(r"[A-Za-z]{1,4}\d{1,4}[a-z]?", s) and not re.fullmatch(r"[A-Z]{1,3}\d{1,4}[a-z]?", s) or re.fullmatch(r"[KS]Q?\d{1,4}", s):
        it = core.queue_item(s)
        if it.get("error"):
            return {"error": it["error"]}
        s = it.get("launch") or ""
    if re.search(r"[;&|`$<>(){}\n\\*?~'\"]", s):
        return {"error": "job has shell syntax: %s" % s[:100]}
    a = s.split()
    if a[:2] == ["bash", "tools/launch_experiment.sh"]:
        a = a[2:]
    if not a:
        return {"error": "empty job"}
    head, rest = a[0], a[1:]
    args, compile_ = [], True
    i = 0
    while i < len(rest):
        f = rest[i]
        if f == "--no-compile":
            compile_ = False; i += 1
        elif f in _FLAGS and i + 1 < len(rest) and re.fullmatch(_FLAGS[f], rest[i + 1]):
            args += [f, rest[i + 1]]; i += 2
        else:
            return {"error": "flag %r not allowed remotely (allowed: --seed --max-tokens --lr --no-compile)" % f}
    if re.fullmatch(r"[A-Z]{1,3}\d{1,4}[a-z]?", head):
        reg = next((e for e in core._registry() if e.get("id") == head), None)
        if not reg or not reg.get("config"):
            return {"error": "experiment %s has no registered config" % head}
        ds = _dataset_of(reg["config"])
        cmd = ["env", 'PY="$HOME/.venvs/gogeta/bin/python"', "bash", "tools/launch_experiment.sh", head] + args + \
              ([] if compile_ else ["--no-compile"]) + ["--foreground"]
        kind, exp = "launcher", head
    elif re.fullmatch(r"configs/[A-Za-z0-9_.-]+\.json", head) and (core.REPO / head).is_file():
        ds = _dataset_of(head)
        exp = next((e["id"] for e in core._registry() if e.get("config") == head), Path(head).stem)
        cmd = ["python", "-m", "gogeta.train", "--config", head] + (["--compile"] if compile_ else []) + args + ["--require-clean"]
        kind = "config"
    else:
        return {"error": "first token must be a registered experiment id (E10) or an existing configs/*.json; got %r" % head}
    if ds not in DATASET_MANIFESTS:
        return {"error": "dataset %r has no committed manifest; only %s can be uploaded" % (ds, sorted(DATASET_MANIFESTS))}
    if not all(_TOKEN.match(t) or t.startswith('PY="') for t in cmd):
        return {"error": "unsafe token in job"}
    return {"kind": kind, "experiment": exp, "dataset": ds, "command": " ".join(cmd), "launch": " ".join([head] + args + ([] if compile_ else ["--no-compile"]))}


def data_allowlist(dataset: str) -> dict:
    """EXPLICIT allowlist: the files hashed in the committed manifest (minus docs_*.jsonl, never read by training) + meta.json.
    Plain files directly in ~/fusiondance-data/datasets/<name>/ only. Never a directory sync."""
    mf = DATASET_MANIFESTS.get(dataset)
    if not mf:
        return {"files": [], "errors": ["no manifest for dataset %r" % dataset]}
    try:
        m = json.loads(core._read(core.REPO / mf))
    except ValueError:
        return {"files": [], "errors": ["unreadable manifest %s" % mf]}
    want = {}
    for split in (m.get("splits") or {}).values():
        for name, info in (split.get("files") or {}).items():
            if not name.endswith(".jsonl"):
                want[name] = (info.get("sha256"), mf)
    tok = m.get("tokenizer") or {}
    if tok.get("file"):
        want[Path(tok["file"]).name] = (tok.get("sha256"), mf)
    for n in EXTRA_ALLOWED:
        want.setdefault(n, (None, "local sha256 (not in the manifest)"))
    root = (core.DATA / "datasets" / dataset).resolve()
    files, errs = [], []
    for name, (sha, source) in sorted(want.items()):
        p = root / name
        if not _SAFE_NAME.match(name) or _FORBIDDEN.search(name):
            errs.append("refused file name %r" % name)
        elif not p.is_file() or p.is_symlink() or p.resolve().parent != root:
            errs.append("missing (or not a plain file): %s" % p)
        else:
            files.append({"name": name, "path": str(p), "bytes": p.stat().st_size, "sha256": sha, "source": source})
    return {"dataset": dataset, "manifest": mf, "root": str(root), "files": files, "errors": errs, "total_bytes": sum(f["bytes"] for f in files)}


_HASHES: dict = {}


def sha256_file(p) -> str:
    p = Path(p)
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime)
    if k not in _HASHES:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        _HASHES[k] = h.hexdigest()
    return _HASHES[k]


def _parse_sums(text: str) -> dict:
    d = {}
    for ln in (text or "").splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?(\S+)$", ln.strip())
        if m:
            d[Path(m.group(2)).name] = m.group(1)
    return d


# --------------------------------------------------------------------------- remote scripts (ported from the pilot)
BOOTSTRAP = Path(__file__).resolve().parent / "remote" / "bootstrap.sh"  # the single source of the remote env + pins
SETUP_DONE = r"== (bootstrap|setup) done"     # bootstrap.sh ends with "== bootstrap done"; the pilot's script with "== setup done"


def setup_script(commit: str) -> str:
    """Write cockpit/remote/bootstrap.sh to the remote and run it DETACHED (survives ssh drops); progress in ~/setup.log."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise ValueError("setup needs a full 40-hex commit")
    body = BOOTSTRAP.read_text(encoding="utf-8")
    assert "GOGETA_BOOT_EOF" not in body
    if not body.endswith("\n"):
        body += "\n"
    return ("cat > \"$HOME/gogeta-bootstrap.sh\" <<'GOGETA_BOOT_EOF'\n" + body + "GOGETA_BOOT_EOF\n"
            "bash \"$HOME/gogeta-bootstrap.sh\" " + commit + " --detach\n")


SETUP_STATUS = r"""L="$HOME/setup.log"
if grep -qE "== (bootstrap|setup) done" "$L" 2>/dev/null; then echo SETUP=done; grep -E "^ENV OK" "$L" | tail -1
elif grep -q "SETUP FAILED" "$L" 2>/dev/null; then echo SETUP=failed; tail -n 15 "$L"
elif pgrep -f "[g]ogeta-bootstrap.sh" >/dev/null || pgrep -f "[s]etup_env.sh" >/dev/null; then echo SETUP=running; tail -n 2 "$L" 2>/dev/null
elif [ -f "$L" ]; then echo SETUP=failed; tail -n 15 "$L"
else echo SETUP=none; fi
"""


def job_script(jid: str, command: str, tracker: str = "none") -> str:
    """The cockpit job wrapper: refuses if any gogeta.train runs (one GPU job per instance), then nohup setsid; writes pid/info/exit."""
    assert re.fullmatch(r"JOB-[0-9a-f]{8}", jid) and tracker in ("none", "trackio")
    return """D="$HOME/fusiondance-data"; mkdir -p "$D/run" "$D/logs"
if pgrep -f "[g]ogeta[.]train" >/dev/null; then echo "BUSY: a gogeta.train process already runs"; exit 3; fi
if [ -f "$D/run/cockpit-job.pid" ] && kill -0 "$(cat "$D/run/cockpit-job.pid")" 2>/dev/null; then echo "BUSY: cockpit job alive"; exit 3; fi
grep -qE "== (bootstrap|setup) done" "$HOME/setup.log" 2>/dev/null || {{ echo "NOT SET UP (no done marker in ~/setup.log)"; exit 4; }}
cat > "$D/run/cockpit-job.sh" <<'GOGETA_EOF'
. "$HOME/.venvs/gogeta/bin/activate"
export GOGETA_DATA="$HOME/fusiondance-data" PYTHONUNBUFFERED=1 GOGETA_TRACKER={tracker}
cd "$HOME/Gogeta-0"
{command}
echo $? > "$HOME/fusiondance-data/run/cockpit-job.exit"
GOGETA_EOF
rm -f "$D/run/cockpit-job.exit"
cat > "$D/run/cockpit-job.info" <<GOGETA_EOF
job={jid}
log=$D/logs/cockpit-{jid}.log
started=$(date +%s)
GOGETA_EOF
nohup setsid bash "$D/run/cockpit-job.sh" > "$D/logs/cockpit-{jid}.log" 2>&1 < /dev/null &
echo $! > "$D/run/cockpit-job.pid"
echo "LAUNCHED $!"
""".format(jid=jid, command=command, tracker=tracker)


STOP_SCRIPT = r"""D="$HOME/fusiondance-data"; p=$(cat "$D/run/cockpit-job.pid" 2>/dev/null)
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; sleep 3; kill -0 "$p" 2>/dev/null && kill -KILL -- "-$p" 2>/dev/null; echo "STOPPED $p"; echo 143 > "$D/run/cockpit-job.exit"
else echo "NO COCKPIT JOB RUNNING"; fi
"""

POLL_SCRIPT = r"""D="$HOME/fusiondance-data"
echo '@@GPU'; nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,name,driver_version,power.draw,power.limit,clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null | head -1
echo '@@APPS'; nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true
echo '@@RUN'; R=$(ls -td "$D"/runs/*/ 2>/dev/null | head -1)
if [ -n "$R" ] && [ -f "$R/meta.json" ]; then python3 - "$R" <<'GOGETA_PY' 2>/dev/null
import json, os, sys
d = sys.argv[1]; m = json.load(open(os.path.join(d, "meta.json"))); c = m.get("config") or {}; t = c.get("train") or {}; g = m.get("git") or {}
print(json.dumps({"name": m.get("name"), "experiment": m.get("experiment"), "seed": m.get("seed"), "config": c.get("config_file"), "lr": t.get("lr"),
                  "max_tokens": t.get("max_tokens"), "commit": (g.get("commit") or "")[:12], "describe": g.get("describe"), "started": m.get("started"),
                  "finished": os.path.exists(os.path.join(d, "summary.json"))}))
GOGETA_PY
fi
echo '@@JOB'; if [ -f "$D/run/cockpit-job.info" ]; then cat "$D/run/cockpit-job.info"; p=$(cat "$D/run/cockpit-job.pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo alive=1; else echo alive=0; fi; [ -f "$D/run/cockpit-job.exit" ] && echo "exit=$(cat "$D/run/cockpit-job.exit")"; fi
echo '@@PROCS'; pgrep -af "[g]ogeta[.]train" 2>/dev/null | cut -c1-220
echo '@@SETUP'; L="$HOME/setup.log"; if grep -qE "== (bootstrap|setup) done" "$L" 2>/dev/null; then echo done; elif grep -q "SETUP FAILED" "$L" 2>/dev/null; then echo failed; elif pgrep -f "[g]ogeta-bootstrap.sh|[s]etup_env.sh" >/dev/null; then echo running; elif [ -f "$L" ]; then echo failed; else echo none; fi
echo '@@LOG'; L=$(sed -n 's/^log=//p' "$D/run/cockpit-job.info" 2>/dev/null); [ -f "$L" ] || L=$(ls -t "$D"/logs/*.log 2>/dev/null | grep -vE '/(extras|setup|pip|bench)[^/]*$' | head -1)
if [ -n "$L" ] && [ -f "$L" ]; then echo "path=$(basename "$L")"; echo "mtime=$(stat -c %Y "$L")"; echo '@@HEAD'; head -n 8 "$L"; echo '@@TAIL'; tail -n 80 "$L"; fi
echo '@@RUNS'; ls -t "$D/runs" 2>/dev/null | head -8
echo '@@NOW'; date +%s
"""


def parse_poll(text: str) -> dict:
    sec, cur = {}, None
    for ln in text.replace("\r", "").splitlines():
        if re.fullmatch(r"@@[A-Z]+", ln.strip()):
            cur = ln.strip()[2:]
            sec[cur] = []
        elif cur:
            sec[cur].append(ln)
    out = {"ok": "NOW" in sec}
    g = [x.strip() for x in ("".join(sec.get("GPU", [""])[:1])).split(",")]
    if len(g) >= 4:
        gi = lambda i, nd=0: core._f(g[i], nd) if len(g) > i else None  # noqa: E731
        out["gpu"] = {"temp_c": gi(0), "util_pct": gi(1), "mem_used_mib": gi(2), "mem_total_mib": gi(3),
                      "name": g[4] if len(g) > 4 else "", "driver": g[5] if len(g) > 5 else "", "power_w": gi(6, 1), "power_limit_w": gi(7, 1),
                      "clock_sm_mhz": gi(8), "clock_mem_mhz": gi(9)}
    try:
        out["run"] = json.loads("".join(sec.get("RUN", [])).strip() or "null")
    except ValueError:
        out["run"] = None
    try:
        out["gpu_procs"] = int(("".join(sec.get("APPS", ["0"])).strip() or "0"))
    except ValueError:
        out["gpu_procs"] = 0
    out["job"] = dict(l.split("=", 1) for l in sec.get("JOB", []) if "=" in l)
    out["procs"] = [core.redact(l.strip()) for l in sec.get("PROCS", []) if l.strip()]
    out["setup"] = ("".join(sec.get("SETUP", [""])).strip() or "unknown")
    lg = dict(l.split("=", 1) for l in sec.get("LOG", []) if "=" in l)
    out["log_name"], out["log_mtime"] = lg.get("path"), core._f(lg.get("mtime"), 0)
    head, tail = "\n".join(sec.get("HEAD", [])), "\n".join(sec.get("TAIL", []))
    out["tail"] = core.redact(tail)
    out["runs"] = [x.strip() for x in sec.get("RUNS", []) if x.strip()]
    out["remote_now"] = core._f("".join(sec.get("NOW", [""])).strip(), 0)
    out["metrics"] = log_metrics(head, tail)
    return out


def log_metrics(head: str, tail: str) -> dict:
    m = {}
    lines = [l for l in (tail or "").splitlines() if l.strip()]
    if lines:
        m["last_line"] = core.redact(lines[-1].strip())[:200]
    losses = [float(x.group(1)) for x in (re.search(r"^step\s+\d+/\d+\s+loss\s+([0-9.]+)", l.strip()) for l in lines) if x]
    if losses:
        m["losses"] = losses[-40:]
    for ln in reversed(lines):
        if "step" not in m:
            mm = re.search(r"^step\s+(\d+)/(\d+)\s+loss\s+(\S+)", ln.strip())
            if mm:
                m.update(step=int(mm.group(1)), total=int(mm.group(2)), loss=core._f(mm.group(3)))
                t = re.search(r"([\d.]+)([kM]?)\s*tok/s", ln)
                if t:
                    m["tok_s"] = round(float(t.group(1)) * {"": 1, "k": 1e3, "M": 1e6}[t.group(2)])
        if "val_loss" not in m:
            mm = re.search(r"val/loss\s+(\S+)", ln)
            if mm:
                m["val_loss"] = core._f(mm.group(1))
        if "final" not in m and ln.startswith("FINAL "):
            m["final"] = core.parse_final(ln)
        if "run_dir" not in m:
            mm = re.match(r"^done:\s+(\S+)", ln.strip())
            if mm:
                m["run_dir"] = mm.group(1)
    tps = re.search(r"steps\s+\d+\s+x\s+(\d+)\s+tok", head or "")
    if tps and m.get("tok_s") and m.get("total") is not None and m.get("step") is not None:
        m["eta_min"] = round((m["total"] - m["step"]) * int(tps.group(1)) / max(1, m["tok_s"]) / 60, 1)
    return m


# --------------------------------------------------------------------------- monitoring (read-only on the remote)
def poll_instance(r: dict) -> tuple[dict, list]:
    """One SSH poll: GPU (nvidia-smi), cockpit job, gogeta.train processes, setup state, redacted log tail.
    Registry rows: stored in SQLite, errors/incidents raised; unmanaged rows: returned only."""
    rc, out = ssh(r, POLL_SCRIPT, timeout=60)
    now = _now()
    managed = r.get("mode") != "unmanaged"
    if rc != 0:
        if managed:
            cloud.inst_update(r["id"], last_poll_epoch=now)
        return {"ok": False, "error": core.redact(out)[-200:]}, []
    p = parse_poll(out)
    if not managed:
        return p, []
    f = {"last_poll_epoch": now, "poll": json.dumps({k: p.get(k) for k in ("gpu", "gpu_procs", "job", "procs", "setup", "metrics", "log_name",
                                                                              "log_mtime", "runs", "remote_now", "run")})}
    if p["ok"]:
        f["last_poll_ok_epoch"] = now
        if p.get("log_mtime") and p.get("remote_now") and p["remote_now"] - p["log_mtime"] < 120:
            f["last_heartbeat_epoch"] = now
        if p["procs"] or p["job"].get("alive") == "1":
            f["last_busy_epoch"] = now
    cloud.inst_update(r["id"], **f)
    mirror = cloud.cdir() / "remote" / ("%s.log" % r["label"])
    mirror.parent.mkdir(exist_ok=True)
    mirror.write_text(p["tail"] + "\n", encoding="utf-8")
    j = job(r.get("job_id"))
    subj = "vast:%s/%s" % (r["label"], (j or {}).get("experiment") or "-")
    findings = []
    for kind, exc, msg, tb, _ in logdb.scan_text(p["tail"]):
        if kind not in ("traceback", "nan", "oom"):
            continue
        logdb.error("remote", msg, exc, tb, run_id=(j or {}).get("remote_run"), context={"instance": r["label"], "gpu": r["gpu_name"], "job": (j or {}).get("id")})
        findings.append({"kind": "remote_" + kind, "subject": subj, "subject_type": "remote-run", "key": hashlib.md5(msg.encode()).hexdigest()[:10],
                         "severity": "high", "hint": {"nan": "code-bug", "oom": "resource"}.get(kind, ""), "tail": p["tail"][-3000:], "traceback": tb,
                         "summary": "%s on %s (%s): %s" % (kind, r["label"], r["gpu_name"], msg[:150]),
                         "repro": "gogeta_vast_tail %s" % r["label"], "evidence": {"instance": r["label"], "gpu": r["gpu_name"], "job": (j or {}).get("launch")}})
    if j and j["status"] == "running":
        run = (Path(p["metrics"]["run_dir"]).name if p["metrics"].get("run_dir") else None)
        if run and run != j.get("remote_run"):
            _upd("cloud_jobs", j["id"], remote_run=run)
        jb = p["job"]
        if jb.get("job") == j["id"] and jb.get("exit") not in (None, ""):
            _job_exited(r, job(j["id"]), jb["exit"], p["tail"])
        elif jb.get("job") == j["id"] and jb.get("alive") == "0":
            findings.append({"kind": "remote_dead", "subject": subj, "subject_type": "remote-run", "key": "dead-%s" % j["id"], "severity": "high",
                             "tail": p["tail"][-3000:], "summary": "remote job %s on %s: process gone, no exit file" % (j["launch"], r["label"])})
            _upd("cloud_jobs", j["id"], status="failed", error="process died without an exit file", ended_epoch=now)
            _after_job(cloud.inst(r["id"]))
        elif p.get("log_mtime") and p.get("remote_now") and (p["remote_now"] - p["log_mtime"]) / 60 > HUNG_MIN:
            findings.append({"kind": "remote_hung", "subject": subj, "subject_type": "remote-run", "key": "hung-%s-%d" % (j["id"], int(p["log_mtime"])),
                             "severity": "high", "tail": p["tail"][-2000:], "summary": "remote job %s on %s: no log line for %.0f min" % (
                                 j["launch"], r["label"], (p["remote_now"] - p["log_mtime"]) / 60)})
    return p, findings


def _approval_for(r: dict):
    if r.get("approval_id"):
        return next((a for a in cloud._load_approvals() if a.get("id") == r["approval_id"]), None)
    v = [a for a in cloud._load_approvals() if not cloud._approval_problems(a)]
    return v[0] if len(v) == 1 else None


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(vals: list, width: int = 24) -> str:
    v = [x for x in vals or [] if isinstance(x, (int, float))][-width:]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * 7) if hi > lo else 3] for x in v)


def temp_mood(t) -> str:
    if t is None:
        return "(?_?) no reading"
    return "(-_-) chill" if t < 55 else "(^_^) warm" if t < 70 else "(o_o) hot" if t < 80 else "(x_x) melting"


def text_bar(frac, width: int = 20) -> str:
    try:
        f = max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        return "-" * width
    n = int(round(f * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def n1x_match(run: dict | None) -> dict | None:
    """The matching LOCAL (N1X) run: same experiment, seed, lr and max_tokens, not itself a cloud run."""
    if not run or not run.get("experiment"):
        return None
    best = None
    for d in core.RUNS_DIR.glob("%s-*" % run["experiment"]) if core.RUNS_DIR.exists() else []:
        sj = d / "summary.json"
        if not sj.exists() or d.name == run.get("name"):
            continue
        try:
            man = json.loads(core._read(d / "repro-manifest.json") or "{}")
            if (man.get("compute") or {}).get("provider"):
                continue
            s = json.loads(sj.read_text())
        except ValueError:
            continue
        t = ((s.get("config") or {}).get("train") or {})
        if str(s.get("seed")) == str(run.get("seed")) and core._f(t.get("lr"), 8) == core._f(run.get("lr"), 8) and t.get("max_tokens") == run.get("max_tokens"):
            m = s.get("metrics") or {}
            best = {"run": s.get("name", d.name), "tokens_per_s": m.get("tokens_per_s"), "val_loss": m.get("val_loss")}
    return best


def status_row(r: dict, p: dict | None = None, now: float | None = None) -> dict:
    now = _now() if now is None else now
    p = p if p is not None else json.loads(r.get("poll") or "{}")
    p = p or {}
    m, g = p.get("metrics") or {}, p.get("gpu") or {}
    meta = json.loads(r.get("meta") or "{}")
    a = _approval_for(r)
    dph = float(r.get("dph") or 0)
    end = r.get("destroyed_epoch") or now
    start = r.get("created_epoch") or now
    accrued = round(dph * max(0.0, end - start) / 3600, 4)
    spent_appr = round(sum(cloud.inst_cost(x, now) for x in cloud._inst_rows() if a and x["approval_id"] == a["id"]), 4) if a else None
    if a and not spent_appr:
        spent_appr = accrued  # unmanaged instance under the only valid approval
    cap = float(a.get("max_usd_total") or 0) if a else None
    j = job(r.get("job_id"))
    procs = p.get("procs") or []
    running = bool(procs) or (p.get("job") or {}).get("alive") == "1"
    run = p.get("run") or {}
    run_started = None
    if j and j.get("started_epoch"):
        run_started = j["started_epoch"]
    elif run.get("started"):
        try:
            run_started = cloud._epoch(run["started"])
        except (ValueError, TypeError):
            run_started = None
    tok_s = m.get("tok_s")
    run_cost = round(dph * max(0.0, now - run_started) / 3600, 4) if run_started and running else None
    eta = m.get("eta_min")
    usd_per_mtok = round(dph / (tok_s * 3600 / 1e6), 4) if tok_s else None
    n1x = n1x_match(run) if run else None
    cmp_ = None
    if n1x:
        cmp_ = {"local_run": n1x["run"], "tok_s_ratio": round(tok_s / n1x["tokens_per_s"], 2) if tok_s and n1x.get("tokens_per_s") else None,
                "val_loss_delta": round(m["val_loss"] - n1x["val_loss"], 4) if m.get("val_loss") is not None and n1x.get("val_loss") is not None else None,
                "note": "remote latest val/loss minus the local run's final val_loss" if not run.get("finished") else "both final"}
    return {"instance": r["id"], "label": r["label"], "mode": r.get("mode") or "fleet", "state": r["state"], "live_status": r.get("live_status"),
            "gpu": r["gpu_name"], "vram_gb": meta.get("vram_gb"), "driver": r.get("driver") or meta.get("driver") or g.get("driver"),
            "cpu_cores": meta.get("cpu_cores"), "ram_gb": meta.get("ram_gb"), "location": meta.get("location"), "hosting_type": meta.get("hosting_type"),
            "verification": meta.get("verification"), "reliability": meta.get("reliability"), "inet_up_mbps": meta.get("inet_up_mbps"),
            "inet_down_mbps": meta.get("inet_down_mbps"), "ip": r.get("ssh_host"), "ssh_port": r.get("ssh_port"),
            "uptime_h": round((end - start) / 3600, 3), "dph": dph, "dph_parts": {"gpu": meta.get("dph_gpu"), "disk": meta.get("dph_disk"),
                                                                              "bandwidth_usd_per_gb_up_down": [meta.get("bw_up_usd_per_gb"), meta.get("bw_down_usd_per_gb")]},
            "accrued_usd": accrued, "projected_to_finish_usd": round(dph * eta / 60, 4) if eta is not None and running else None,
            "approval": a and a["id"], "approval_cap_usd": cap, "approval_spent_usd": spent_appr, "approval_expires": a and a.get("expires_at"),
            "budget_remaining_usd": round(cap - spent_appr, 4) if a else None, "budget_bar": text_bar((spent_appr or 0) / cap) if a and cap else None,
            "util_pct": g.get("util_pct"), "mem_used_mib": g.get("mem_used_mib"), "mem_total_mib": g.get("mem_total_mib"), "temp_c": g.get("temp_c"),
            "power_w": g.get("power_w"), "power_limit_w": g.get("power_limit_w"), "clock_sm_mhz": g.get("clock_sm_mhz"), "clock_mem_mhz": g.get("clock_mem_mhz"),
            "job": (j or {}).get("launch") or (procs[0][:120] if procs else None), "job_running": running, "setup": p.get("setup"),
            "run": {"name": run.get("name"), "config": run.get("config"), "seed": run.get("seed"), "lr": run.get("lr"), "max_tokens": run.get("max_tokens"),
                    "commit": run.get("commit"), "finished": run.get("finished")} if run else None,
            "step": m.get("step"), "total": m.get("total"), "loss": m.get("loss"), "val_loss": m.get("val_loss"), "tok_s": tok_s, "eta_min": eta,
            "run_cost_usd": run_cost, "last_line": m.get("last_line"), "log": p.get("log_name"), "vs_n1x": cmp_,
            "fun": {"loss_sparkline": sparkline(m.get("losses")), "usd_per_1M_tokens": usd_per_mtok,
                    "tokens_per_cent": int(tok_s * 3600 / (dph * 100)) if tok_s and dph else None, "mood": temp_mood(g.get("temp_c"))},
            "polled_s_ago": round(now - r["last_poll_epoch"]) if r.get("last_poll_epoch") else None, "setup_s": r.get("setup_s"),
            "idle_min": 0.0 if running else round((now - (r.get("last_busy_epoch") or r.get("state_since") or now)) / 60, 1)}


def lifetime() -> dict:
    """Lifetime cloud spend and GPU-hours over every instance the cockpit registry has seen (unmanaged ones excluded)."""
    rows = cloud._inst_rows()
    now = _now()
    return {"instances": len(rows), "spend_usd": round(sum(cloud.inst_cost(r, now) for r in rows), 4),
            "gpu_hours": round(sum(((r.get("destroyed_epoch") or now) - (r.get("created_epoch") or now)) / 3600 * (r.get("num_gpus") or 1)
                                   for r in rows if r.get("created_epoch")), 3),
            "jobs_done": len(_q("SELECT 1 FROM cloud_jobs WHERE status='done'"))}


def vast_status(instance: str = "", refresh: bool = True) -> dict:
    """READ-ONLY. Monitor one instance (id or label) or all gogeta-cockpit-* instances: state, $/h, accrued $ vs the approval cap,
    GPU util/memory/temp (nvidia-smi over ssh), current job, last log line, latest val/loss, ETA, idle minutes. Works for
    unmanaged gogeta-cockpit-* instances too (read-only). refresh=false uses the last poll."""
    live = cloud.sync_live()
    now = _now()
    if instance:
        r, _ = resolve(instance, live)
        targets = [r] if r else []
        if not r:
            return {"error": "no cockpit instance %s (registry or live gogeta-cockpit-* label)" % instance}
    else:
        targets = [r for r in cloud._inst_rows(active_only=True)]
        known = {r["id"] for r in targets}
        for L in live or []:
            if str(L.get("label") or "").startswith(cloud.LABEL_PREFIX) and str(L.get("id")) not in known:
                targets.append(resolve(str(L.get("id")), live)[0])
    rows = []
    for r in targets:
        p = None
        if refresh and r.get("ssh_host") and r["state"] not in ("provisioning", "destroyed"):
            p, f = poll_instance(r)
            if not p.get("ok"):
                p = {"error": p.get("error")}
            r = cloud.inst(r["id"]) or r
        row = status_row(r, p if p and not p.get("error") else None, now)
        if p and p.get("error"):
            row["poll_error"] = p["error"]
        if not row["job_running"] and row["state"] in ("ready", "idle", "unmanaged"):
            lim = IDLE_S if row["mode"] == "fleet" else IDLE_KILL_S
            row["idle_flag"] = True
            row["suggest"] = ("IDLE %.0f min: auto pull + destroy at %d min (AGENTS 0h)" % (row["idle_min"], lim // 60)) if row["mode"] != "unmanaged" else \
                "IDLE (no gogeta.train process): not managed by the cockpit, destroy it by hand if no work is queued (AGENTS 0h)"
        rows.append(row)
    return {"instances": rows, "count": len(rows), "burn_usd_per_h": round(sum(x["dph"] or 0 for x in rows if x["state"] != "destroyed"), 4),
            "fleet_power_level_tok_s": sum(x["tok_s"] or 0 for x in rows if x["job_running"]), "lifetime": lifetime(), "live_listing": live is not None,
            "note": "ip/ssh_port are for display only: never commit or paste them"}


def vast_tail(instance: str, n: int = 60, refresh: bool = True) -> dict:
    """READ-ONLY. Redacted tail of the current training log on an instance (registry or unmanaged gogeta-cockpit-*)."""
    r, managed = resolve(instance)
    if not r:
        return {"error": "unknown cockpit instance %s" % instance}
    n = max(1, min(int(n), 400))
    if refresh:
        p, _ = poll_instance(r)
        if p.get("ok"):
            lines = [l for l in p["tail"].splitlines() if l.strip()][-n:]
            return {"instance": r["label"], "log": p.get("log_name"), "text": "\n".join(lines), "lines": len(lines), "metrics": p["metrics"]}
        if not managed:
            return {"error": "poll failed: %s" % p.get("error")}
    txt = core._read(cloud.cdir() / "remote" / ("%s.log" % r["label"]))
    lines = [l for l in txt.splitlines() if l.strip()][-n:]
    return {"instance": r["label"], "text": core.redact("\n".join(lines)), "lines": len(lines), "cached": True}


fleet_logs = vast_tail


# --------------------------------------------------------------------------- data staging, results sync
def upload_dataset(r: dict, dataset: str) -> dict:
    """Allowlist-only upload (tar of named files, --no-recursion) + sha256sum -c on arrival. Skips files already verified there."""
    al = data_allowlist(dataset)
    if al["errors"]:
        return {"ok": False, "error": "; ".join(al["errors"])}
    for f in al["files"]:
        local = sha256_file(f["path"])
        if f["sha256"] and local != f["sha256"]:
            return {"ok": False, "error": "local %s does not match its manifest sha256: refusing to upload" % f["name"]}
        f["expect"] = f["sha256"] or local
    names = [f["name"] for f in al["files"]]
    dst = "%s/datasets/%s" % (R_DATA, dataset)
    sums = "".join("%s  %s\n" % (f["expect"], f["name"]) for f in al["files"])
    check = 'mkdir -p "%s" && cd "%s" && cat > .gogeta-expected.sha256 <<GOGETA_EOF\n%sGOGETA_EOF\nsha256sum -c .gogeta-expected.sha256 2>&1 || true\n' % (dst, dst, sums)
    rc, out = ssh(r, check, timeout=900)
    ok_now = {m.group(1) for m in re.finditer(r"^(\S+): OK$", out, re.M)}
    todo = [f for f in al["files"] if f["name"] not in ok_now]
    t0 = _now()
    if todo:
        tar = ["tar", "-C", al["root"], "-cf", "-", "--no-recursion", "--"] + [f["name"] for f in todo]
        rc, out = _pipe(tar, ssh_argv(r["ssh_host"], r["ssh_port"]) + ['mkdir -p "%s" && tar -C "%s" -xf -' % (dst, dst)], timeout=7200)
        if rc != 0:
            return {"ok": False, "error": "upload failed (rc %s): %s" % (rc, core.redact(out)[-300:])}
        rc, out = ssh(r, check, timeout=900)
        ok_now = {m.group(1) for m in re.finditer(r"^(\S+): OK$", out, re.M)}
    bad = [n for n in names if n not in ok_now]
    if bad:
        return {"ok": False, "error": "sha256 mismatch on arrival: %s" % ", ".join(bad)}
    return {"ok": True, "uploaded": [f["name"] for f in todo], "verified": names, "bytes": sum(f["bytes"] for f in todo),
            "upload_s": round(_now() - t0, 1)}


FETCH_DATA = Path(__file__).resolve().parent / "remote" / "fetch_data.sh"
FETCHABLE = ("code-py-v1",)  # datasets fetch_data.sh can rebuild on the box from the public HF source


def stage_dataset(r: dict, dataset: str) -> dict:
    """Data step: rebuild on the box with cockpit/remote/fetch_data.sh (verified against the committed manifest) when possible;
    on DATA MISMATCH / failure fall back to the allowlisted, sha256-checked upload. Either way the result is verified."""
    al = data_allowlist(dataset)
    if al["errors"]:
        return {"ok": False, "error": "; ".join(al["errors"])}
    fetched = None
    if dataset in FETCHABLE and FETCH_DATA.exists():
        t0 = _now()
        rc, out = ssh(r, FETCH_DATA.read_text(encoding="utf-8"), timeout=5400)
        last = [l for l in out.strip().splitlines() if l.startswith("DATA ")]
        fetched = {"rc": rc, "result": core.redact(last[-1] if last else out.strip()[-200:])[:200], "s": round(_now() - t0, 1)}
        _event("info", "remote_fetch_data", "fetch_data on %s: %s (%.0fs)" % (r["label"], fetched["result"], fetched["s"]))
    up = upload_dataset(r, dataset)  # verifies everything; uploads only what fetch_data did not produce byte-identically
    up["fetch_data"] = fetched
    return up


def _manifest_hash(m: dict) -> str:  # identical to repro/common.py manifest_self_hash
    body = json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def job_costs(j: dict, r: dict, now: float | None = None) -> dict:
    now = _now() if now is None else now
    dph = float(r.get("dph") or 0)
    first = not _q("SELECT 1 FROM cloud_jobs WHERE instance_id=? AND id!=? AND started_epoch < ? LIMIT 1", (r["id"], j["id"], j.get("started_epoch") or now))
    setup = dph * float(r.get("setup_s") or 0) / 3600 if first else 0.0
    run = dph * max(0.0, (j.get("ended_epoch") or now) - (j.get("started_epoch") or now)) / 3600
    td = float(j.get("teardown_usd") or 0)
    return {"setup_usd": round(setup, 4), "run_usd": round(run, 4), "teardown_usd": round(td, 4), "total_usd": round(setup + run + td, 4),
            "dph": dph, "setup_s": r.get("setup_s")}


def pull_run(r: dict, run: str, checkpoint: bool = False, j: dict | None = None) -> dict:
    """Pull ~/fusiondance-data/runs/<run>/ into local runs/ (staged, sha256-verified, never overwriting) + repro-manifest.json
    with provider/GPU/driver/dph/cost. Read-only on the remote."""
    if not run or not _SAFE_NAME.match(run):
        return {"ok": False, "error": "bad run name %r" % run}
    stage = cloud.cdir() / "incoming" / run
    final = core.RUNS_DIR / run
    if final.exists():
        return {"ok": False, "error": "local run folder %s already exists: not overwritten" % final}
    shutil.rmtree(stage, ignore_errors=True)  # the cockpit's own staging dir
    stage.parent.mkdir(parents=True, exist_ok=True)
    rc, out = _pipe(ssh_argv(r["ssh_host"], r["ssh_port"]) + ['tar -C "%s/runs" -cf - -- %s' % (R_DATA, shlex.quote(run))],
                    ["tar", "-C", str(stage.parent), "--no-same-owner", "-xf", "-"], timeout=3600)
    files = [n for n in ("summary.json", "meta.json", "samples.txt") if (stage / n).exists()]
    if rc != 0 or "meta.json" not in files:
        return {"ok": False, "error": "pull failed rc %s: %s" % (rc, core.redact(out)[-200:])}
    if "summary.json" not in files:
        return {"ok": False, "error": "run %s has no summary.json yet (still running?)" % run}
    rc, out = ssh(r, 'cd "%s/runs/%s" && sha256sum -- %s' % (R_DATA, run, " ".join(files)), timeout=120)
    remote = _parse_sums(out)
    bad = [n for n in files if remote.get(n) != sha256_file(stage / n)]
    if bad:
        return {"ok": False, "error": "sha256 mismatch after pull: %s" % bad}
    rc, out = ssh(r, 'cd "%s" && . "$HOME/.venvs/gogeta/bin/activate" 2>/dev/null; GOGETA_DATA="%s" python repro/manifest.py "%s/runs/%s" --stdout-only --no-checkpoint-hash'
                  % (R_REPO, R_DATA, R_DATA, run), timeout=600)
    try:
        man, msrc = json.loads(out[out.index("{"):]), "repro/manifest.py on the remote"
    except ValueError:
        man, msrc = {"schema": 1, "kind": "gogeta-run-manifest", "run": run, "note": "remote manifest.py failed: %s" % core.redact(out)[-200:]}, "minimal (remote manifest.py failed)"
    now = _now()
    p = json.loads(r.get("poll") or "{}")
    man["compute"] = {"provider": cloud.PROVIDER, "gpu": r.get("gpu_name"), "num_gpus": r.get("num_gpus"),
                      "driver": r.get("driver") or ((p.get("gpu") or {}).get("driver")), "dph_usd": r.get("dph"), "on_demand": True,
                      "image": r.get("image"), "remote_env": r.get("remote_env"), "instance_mode": r.get("mode"),
                      "cost_usd": job_costs(j, r, now) if j and r.get("mode") != "unmanaged" else None, "manifest_source": msrc,
                      "note": "instance ids, hosts and billing records stay in the cockpit SQLite (never committed)"}
    man["manifest_sha256"] = _manifest_hash(man)
    (stage / "repro-manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    man_sha = sha256_file(stage / "repro-manifest.json")
    ck = None
    if checkpoint:
        ckd = core.DATA / "checkpoints"
        ckd.mkdir(parents=True, exist_ok=True)
        _pipe(ssh_argv(r["ssh_host"], r["ssh_port"]) + ['tar -C "%s/checkpoints" -cf - -- %s' % (R_DATA, shlex.quote(run))],
              ["tar", "-C", str(ckd), "--no-same-owner", "-xf", "-"], timeout=3600)
        rc2, out2 = ssh(r, 'sha256sum -- "%s/checkpoints/%s/model.pt"' % (R_DATA, run), timeout=300)
        pk = ckd / run / "model.pt"
        ck = pk.exists() and _parse_sums(out2).get("model.pt") == sha256_file(pk)
    os.replace(stage, final)
    try:
        summ = json.loads((final / "summary.json").read_text())
        metrics = {k: (summ.get("metrics") or {}).get(k) for k in ("val_loss", "val_bits_per_char", "tokens_per_s", "wall_min")}
    except ValueError:
        metrics = {}
    _event("info", "remote_synced", "pulled %s from %s: summary/meta/manifest sha256 verified, val_loss %s" % (run, r["label"], metrics.get("val_loss")))
    return {"ok": True, "run": run, "result_dir": str(final), "verified": files, "manifest_sha256": man_sha, "checkpoint_ok": ck, "metrics": metrics}


# --------------------------------------------------------------------------- phases (detached workers, or inline in sync mode)
def _workers() -> Path:
    d = cloud.cdir() / "fleet-workers"
    d.mkdir(exist_ok=True)
    return d


def _worker_alive(iid) -> bool:
    pid = core._pid(_workers() / ("%s.pid" % iid))
    return bool(pid) and "fleet.py" in core._cmdline(pid) and "--worker" in core._cmdline(pid)


def _start_phase(phase: str, iid: str) -> None:
    if _sync_mode():
        run_phase(phase, iid)
        return
    if _worker_alive(iid):
        return
    out = open(_workers() / ("%s.out" % iid), "ab")
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", phase, iid], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=out, start_new_session=True, cwd=str(core.REPO), env=dict(_ssh_env(), **{
                             k: os.environ[k] for k in ("GOGETA_DATA", "GOGETA_REPO", "GOGETA_COCKPIT_DB", "GOGETA_COCKPIT_DIR", "GOGETA_VASTAI", "GOGETA_SSH",
                                                        "WSL_INTEROP", "WSL_DISTRO_NAME") if k in os.environ}))
    (_workers() / ("%s.pid" % iid)).write_text(str(p.pid))


def run_phase(phase: str, iid: str) -> dict:
    r = cloud.inst(iid)
    if not r or r["state"] in ("destroyed", "destroying"):
        return {"error": "instance not active"}
    try:
        if phase == "push":
            return _phase_push(r)
        if phase == "pull":
            return _phase_pull(r)
        return {"error": "unknown phase"}
    except Exception as ex:  # noqa: BLE001
        logdb.error_from_exception("fleet", ex, context={"phase": phase, "instance": r["label"]})
        return {"error": repr(ex)}


def start_setup(r: dict, commit: str) -> dict:
    rc, out = ssh(r, setup_script(commit), timeout=120, retries=0 if _sync_mode() else 9)
    ok = rc == 0 and "BOOTSTRAP_DETACHED" in out
    if ok:
        cloud.inst_update(r["id"], setup_state="running", setup_commit=commit)
        _event("info", "remote_setup", "setup started on %s (commit %s)" % (r["label"], commit[:9]))
    return {"ok": ok, "output": core.redact(out)[-300:]}


def check_setup(r: dict) -> dict:
    rc, out = ssh(r, SETUP_STATUS, timeout=60)
    st = (re.search(r"SETUP=(\w+)", out) or [None, "unreachable" if rc else "unknown"])[1]
    f = {"setup_state": st}
    if st == "done":
        env = re.search(r"^ENV OK (.+)$", out, re.M)
        if env:
            f["remote_env"] = env.group(1).strip()[:240]
    cloud.inst_update(r["id"], **f)
    return {"state": st, "detail": core.redact(out)[-600:]}


def _phase_push(r: dict) -> dict:
    j = job(r.get("job_id"))
    ds = (j or {}).get("dataset") or r.get("push_dataset") or "code-py-v1"
    if j:
        _upd("cloud_jobs", j["id"], status="staging")
    up = stage_dataset(r, ds)
    if not up["ok"]:
        logdb.error("fleet", "staging %s on %s failed: %s" % (ds, r["label"], up["error"]), "RemoteStageFailed", None, context={"instance": r["label"]})
        if r.get("mode") == "fleet":
            return replace_bad_host(r, "data staging failed: %s" % up["error"][:120])
        cloud.set_state(r["id"], "ready", push_state="failed")
        return up
    have = set(json.loads(r.get("data_ok") or "[]")) | {ds}
    f = {"data_ok": json.dumps(sorted(have)), "push_state": "done"}
    if not r.get("setup_s"):
        now = _now()
        f.update(setup_s=round(now - r["created_epoch"], 1), ready_epoch=now)  # rent -> env ready -> data verified
    cloud.inst_update(r["id"], **f)
    _event("info", "remote_data", "%s verified on %s (%d files, %d bytes uploaded)" % (ds, r["label"], len(up["verified"]), up["bytes"]))
    r = cloud.inst(r["id"])
    if j:
        return _launch(r, job(j["id"]))
    cloud.set_state(r["id"], "ready")
    return up


def _launch(r: dict, j: dict) -> dict:
    p = parse_job(j["launch"])
    if p.get("error"):
        _upd("cloud_jobs", j["id"], status="failed", error=p["error"], ended_epoch=_now())
        return {"error": p["error"]}
    tracker = "trackio" if p["kind"] == "launcher" else "none"
    rc, out = ssh(r, job_script(j["id"], p["command"], tracker), timeout=120)
    out = core.redact(out)
    if rc != 0 or "LAUNCHED" not in out:
        logdb.error("fleet", "remote launch of %s on %s exited %s: %s" % (j["launch"], r["label"], rc, out[-300:]), "RemoteLaunchFailed", None,
                    context={"job": j["id"], "instance": r["label"]})
        _upd("cloud_jobs", j["id"], status="failed", error="launch rc %s: %s" % (rc, out[-200:]), ended_epoch=_now())
        _after_job(cloud.inst(r["id"]))
        return {"error": "launch failed: %s" % out[-200:]}
    now = _now()
    _upd("cloud_jobs", j["id"], status="running", started_epoch=now, attempts=(j.get("attempts") or 0) + 1, instance_id=r["id"])
    cloud.set_state(r["id"], "busy", job_id=j["id"], last_poll_ok_epoch=now, last_heartbeat_epoch=now, last_busy_epoch=now, last_poll_epoch=0)
    _event("info", "remote_launch", "%s launched on %s (%s)" % (j["launch"], r["label"], r["gpu_name"]), {"job": j["id"]})
    return {"ok": True, "job": j["id"], "launched": j["launch"]}


def _phase_pull(r: dict) -> dict:
    j = job(r.get("job_id"))
    if not j:
        cloud.set_state(r["id"], "idle")
        return {"error": "no job"}
    run = j.get("remote_run")
    now = _now()
    res = pull_run(r, run, bool(j.get("checkpoint")), j) if run else {"ok": False, "error": "no `done: <dir>` line: run folder unknown"}
    if res.get("ok"):
        c = job_costs(job(j["id"]), r, now)
        _upd("cloud_jobs", j["id"], status="done", verified=1, result_dir=res["result_dir"], manifest_sha256=res["manifest_sha256"],
             metrics=json.dumps(res["metrics"]), setup_usd=c["setup_usd"], run_usd=c["run_usd"],
             checkpoint_ok=None if res["checkpoint_ok"] is None else int(bool(res["checkpoint_ok"])))
        cloud.inst_update(r["id"], jobs_done=(r.get("jobs_done") or 0) + 1)
    else:
        _upd("cloud_jobs", j["id"], status="failed", error=res.get("error"))
    _after_job(cloud.inst(r["id"]))
    return res


# --------------------------------------------------------------------------- reuse / destroy decisions
def _ready_jobs(sid: str) -> list:
    out = []
    for j in jobs(sid, "queued"):
        if j.get("instance_id"):
            continue
        if j.get("after_job") and (job(j["after_job"]) or {}).get("status") != "done":
            continue
        out.append(j)
    return out


def expected_gap_s(sid: str):
    """Seconds until a waiting job becomes ready (it waits on a running job with a known ETA); None = unknown."""
    best = None
    for j in jobs(sid, "queued"):
        dep = job(j.get("after_job"))
        if not dep or dep["status"] != "running" or not dep.get("instance_id"):
            continue
        m = json.loads((cloud.inst(dep["instance_id"]) or {}).get("poll") or "{}").get("metrics") or {}
        if m.get("eta_min") is not None:
            best = m["eta_min"] * 60 if best is None else min(best, m["eta_min"] * 60)
    return best


def _pull_pending(r: dict) -> list:
    """Pull every finished-but-not-synced cockpit run on r (before any automatic destroy: 'pull whatever exists')."""
    out = []
    for j in _q("SELECT * FROM cloud_jobs WHERE instance_id=? AND remote_run IS NOT NULL AND (verified IS NULL OR verified=0) "
                "AND status NOT IN ('running','staging','provisioning','setup')", (r["id"],)):
        if (core.RUNS_DIR / j["remote_run"]).exists():
            continue
        res = pull_run(r, j["remote_run"], bool(j.get("checkpoint")), j)
        if res.get("ok"):
            _upd("cloud_jobs", j["id"], verified=1, result_dir=res["result_dir"], manifest_sha256=res["manifest_sha256"], metrics=json.dumps(res["metrics"]))
        out.append({"run": j["remote_run"], "ok": bool(res.get("ok")), "error": res.get("error")})
    return out


def _queued_for(r: dict) -> list:
    return _q("SELECT * FROM cloud_jobs WHERE instance_id=? AND status='queued' ORDER BY queued_epoch", (r["id"],))


def _after_job(r: dict | None) -> dict:
    """Right after a job ends (AGENTS 0h: kill boxes unless work is queued): run the next job queued for this box or (fleet) its session;
    otherwise pull whatever exists and destroy the box at once. Only gogeta-cockpit-* registry instances ever get here."""
    if not r or r["state"] in ("destroyed", "destroying"):
        return {}
    cloud.set_state(r["id"], "idle", job_id=None, last_busy_epoch=_now())
    _close_session_if_done(r.get("session_id"))
    mine = _queued_for(r)
    if mine:
        return _assign(cloud.inst(r["id"]), mine[0])
    if r.get("mode") != "fleet":
        return destroy_instance(cloud.inst(r["id"]), "AGENTS 0h: job queue empty -> pull down", "fleet")
    nxt = _ready_jobs(r["session_id"]) if r.get("session_id") else []
    if nxt:
        return _assign(cloud.inst(r["id"]), nxt[0])
    gap = expected_gap_s(r["session_id"]) if r.get("session_id") else None
    setup = r.get("setup_s")
    if gap is not None and setup and gap < min(float(setup), IDLE_S):
        _event("info", "fleet_keep", "%s kept: next job expected in %.0f s < setup %.0f s" % (r["label"], gap, setup))
        return {"kept": True, "gap_s": gap}
    return destroy_instance(r, "job finished, no job waiting (just-in-time teardown)" if gap is None else
                            "next job in %.0f s >= setup %.0f s: cheaper to re-rent" % (gap, setup or 0), "fleet")


def _assign(r: dict, j: dict) -> dict:
    _upd("cloud_jobs", j["id"], instance_id=r["id"], status="staging")
    cloud.inst_update(r["id"], job_id=j["id"], state="staging", state_since=_now())
    _event("info", "fleet_reuse", "%s reused for %s (setup skipped)" % (r["label"], j["launch"]), {"job": j["id"]})
    _start_phase("push", r["id"])
    return {"reused": True, "job": j["id"]}


def destroy_instance(r: dict, reason: str, by: str) -> dict:
    pulled = []
    if r.get("ssh_host") and r["state"] not in ("provisioning",):
        try:
            pulled = _pull_pending(r)
        except Exception as ex:  # noqa: BLE001  pulling must never block stopping the spend
            pulled = [{"error": repr(ex)}]
    res = cloud.destroy(r["id"], confirm=True, reason="%s: %s" % (by, reason))
    if pulled:
        res["pulled_first"] = pulled
    if res.get("destroyed"):
        last = _q("SELECT * FROM cloud_jobs WHERE instance_id=? AND ended_epoch IS NOT NULL ORDER BY ended_epoch DESC LIMIT 1", (r["id"],))
        rr = cloud.inst(r["id"])
        if last and rr and rr.get("destroyed_epoch"):
            _upd("cloud_jobs", last[0]["id"], teardown_usd=round(float(r["dph"] or 0) * max(0.0, rr["destroyed_epoch"] - last[0]["ended_epoch"]) / 3600, 4))
        for j in _q("SELECT * FROM cloud_jobs WHERE instance_id=? AND status IN ('provisioning','setup','staging','running','pulling')", (r["id"],)):
            _upd("cloud_jobs", j["id"], status="failed", error="instance destroyed: %s" % reason[:150], ended_epoch=_now())
        _close_session_if_done(r.get("session_id"))
    return res


def replace_bad_host(r: dict, reason: str) -> dict:
    """Stuck/failed setup: destroy, exclude that machine, retry the job ONCE on another offer."""
    s = session(r.get("session_id"))
    j = job(r.get("job_id"))
    if j:
        _upd("cloud_jobs", j["id"], status="queued" if (j.get("setup_retries") or 0) < 1 else "failed")
    res = destroy_instance(r, "bad host: " + reason, "fleet")
    if s and r.get("machine_id"):
        ex = set(json.loads(s.get("exclude_machines") or "[]")) | {r["machine_id"]}
        _upd("cloud_sessions", s["id"], exclude_machines=json.dumps(sorted(ex)))
    retried = False
    if j:
        if (j.get("setup_retries") or 0) < 1:
            _upd("cloud_jobs", j["id"], status="queued", instance_id=None, setup_retries=(j.get("setup_retries") or 0) + 1, error=None, ended_epoch=None)
            retried = True
        else:
            _upd("cloud_jobs", j["id"], status="failed", error="setup failed on two hosts: %s" % reason[:150], ended_epoch=_now())
    return {"replaced": True, "retried": retried, "destroy": res}


def _close_session_if_done(sid):
    if not sid:
        return
    if not _q("SELECT 1 FROM cloud_jobs WHERE session_id=? AND status IN ('queued','provisioning','setup','staging','running','pulling') LIMIT 1", (sid,)) \
            and not cloud._inst_rows(active_only=True, session_id=sid) and _q("SELECT 1 FROM cloud_jobs WHERE session_id=? LIMIT 1", (sid,)):
        if (session(sid) or {}).get("status") == "open":
            _upd("cloud_sessions", sid, status="done")
            _event("info", "fleet_done", "session %s: queue empty, no instances left" % sid)


def _job_exited(r, j, code: str, tail: str) -> None:
    now = _now()
    code = str(code).strip()
    rc = int(code) if code.lstrip("-").isdigit() else None
    if code == "0":
        _upd("cloud_jobs", j["id"], status="pulling", exit_code=0, ended_epoch=now)
        cloud.set_state(r["id"], "pulling")
        _start_phase("pull", r["id"])
        return
    try:
        from . import watchdog as _wd
    except ImportError:
        import watchdog as _wd  # type: ignore
    cat = _wd.classify(tail)["category"]
    if cat == "transient" and (j.get("attempts") or 0) < 2 and r.get("mode") == "fleet":
        _upd("cloud_jobs", j["id"], status="queued", exit_code=rc, error="transient failure (exit %s), retrying once" % code, instance_id=None)
        _event("warn", "remote_retry", "%s on %s failed (transient); retrying once" % (j["launch"], r["label"]), {"job": j["id"]})
    else:
        _upd("cloud_jobs", j["id"], status="failed", exit_code=rc, error="exit %s (%s)" % (code, cat), ended_epoch=now)
    _after_job(cloud.inst(r["id"]))


# --------------------------------------------------------------------------- fleet sessions + dispatch (just-in-time)
def _commit_on_origin(commit: str) -> bool:
    out = core._git_raw("branch", "-r", "--contains", commit)
    return any(l.strip().startswith("origin/") for l in out.splitlines())


def _head_commit() -> tuple[str, bool]:
    c = core._git_raw("rev-parse", "HEAD")
    return c, bool(c) and _commit_on_origin(c)


def fleet_up(n: int, gpu: str = "RTX 3090", approval_id: str = "", image: str = cloud.DEFAULT_IMAGE, max_hours: float = 4.0,
             disk_gb: int = cloud.DEFAULT_DISK_GB, purpose: str = "", min_cuda: float = 13.0, confirm: bool = False, dry_run: bool = True,
             provider: str = "vast", max_dph: float = 0.0) -> dict:
    """APPROVAL-GATED. Open a fleet session: up to n instances of ONE GPU SKU under a human approval. Rents NOTHING (just-in-time:
    dispatch rents only when a job is ready). Pins the current pushed commit. dry_run=true (default) only plans."""
    errs = []
    approval = next((a for a in cloud._load_approvals() if a.get("id") == approval_id), None)
    if not approval:
        errs.append("unknown approval %r (the human creates approvals; MCP cannot)" % approval_id)
    else:
        errs += cloud._approval_problems(approval)
        if gpu.strip().lower() not in [g.lower() for g in approval.get("allowed_gpus", [])]:
            errs.append("GPU %r not allowed by %s" % (gpu, approval_id))
        if not 1 <= int(n) <= int(approval.get("max_instances") or 0):
            errs.append("n=%s outside 1..max_instances %s" % (n, approval.get("max_instances")))
    opened = [s for s in open_sessions() if s["approval_id"] == approval_id]
    for s in opened:
        if s["gpu_name"].lower() != gpu.strip().lower():
            errs.append("one GPU SKU per fleet: session %s already uses %s" % (s["id"], s["gpu_name"]))
    errs += [e for e in cloud._validate_rent(1, max_hours, purpose, image, disk_gb) if not e.startswith("offer_id")]
    commit, on_origin = _head_commit()
    if not on_origin:
        errs.append("HEAD %s is not on origin: push first (remote code comes only from the public repo at a pushed commit)" % commit[:9])
    plan = {"n_max": int(n), "gpu": gpu, "approval": approval_id, "image": image, "max_hours_per_instance": float(max_hours), "disk_gb": int(disk_gb),
            "commit": commit, "commit_on_origin": on_origin, "min_cuda": float(min_cuda),
            "policy": "just-in-time: nothing is rented now; dispatch rents one instance per ready job and destroys it when no job waits"}
    if errs:
        return {"opened": False, "refused": "; ".join(errs), **plan}
    if dry_run:
        est = cloud.offers(gpu, 1, 0.98, True, float(approval.get("max_usd_per_hour") or 0), "dph_total", 50, int(disk_gb), float(min_cuda))
        plan["live"] = {k: est.get(k) for k in ("count", "min_dph", "median_dph", "error")}
        if est.get("min_dph"):
            plan["max_cost_if_all_run_max_hours_usd"] = round(est["min_dph"] * float(max_hours) * int(n), 2)
        return {"opened": False, "dry_run": True, **plan}
    if confirm is not True:
        return {"opened": False, "refused": "confirm must be exactly true", **plan}
    same = [s for s in opened if s["gpu_name"].lower() == gpu.strip().lower()]
    if same:
        _upd("cloud_sessions", same[0]["id"], max_instances=int(n), max_hours=float(max_hours), commit_sha=commit, image=image)
        return {"opened": True, "session_id": same[0]["id"], "reused_session": True, **plan}
    sid = "FLT-%s-%s" % (time.strftime("%Y%m%d-%H%M"), uuid.uuid4().hex[:4])
    _x("INSERT INTO cloud_sessions(id,approval_id,gpu_name,image,max_instances,max_hours,disk_gb,min_cuda,commit_sha,purpose,created,status,exclude_machines,provider,max_dph)"
       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (sid, approval_id, gpu.strip(), image, int(n), float(max_hours), int(disk_gb), float(min_cuda), commit,
                                                 core.redact(purpose)[:300], cloud._iso(), "open", "[]", provider, float(max_dph or 0) or None))
    _event("info", "fleet_up", "session %s: up to %d x %s under %s (commit %s); rents just-in-time" % (sid, int(n), gpu, approval_id, commit[:9]))
    return {"opened": True, "session_id": sid, **plan}


def _resolve_session(session_id: str, approval_id: str):
    if session_id:
        s = session(session_id)
        return s if s and s["status"] == "open" else None
    c = [s for s in open_sessions() if not approval_id or s["approval_id"] == approval_id]
    return c[0] if len(c) == 1 else None


def plan_steps(p: dict, s: dict, offer: dict | None) -> list:
    """The remote runner's 7 steps for one job, as exact commands (every dry run shows them)."""
    al = data_allowlist(p["dataset"])
    commit_ok = bool(re.fullmatch(r"[0-9a-f]{40}", s.get("commit_sha") or ""))
    return [
        {"step": 1, "what": "pick offer (exact SKU, within the per-hour cap)", "query": "gpu_name=%s num_gpus=1 reliability>=0.98 verified=true rentable=true cuda_max_good>=%s"
         % (s["gpu_name"].replace(" ", "_"), s["min_cuda"]), "chosen": offer and {k: offer.get(k) for k in ("id", "gpu_name", "dph_total", "driver_version", "reliability")}},
        {"step": 2, "what": "rent: approval checked (cap, per-hour, max_instances, GPU) BEFORE create", "command":
         "vastai create instance %s --image %s --disk %d --label gogeta-cockpit-<id> --ssh --direct --cancel-unavail --raw" % (offer["id"] if offer else "<offer>", s["image"], s["disk_gb"])},
        {"step": 3, "what": "setup: cockpit/remote/bootstrap.sh %s --detach (gcc, uv, py3.12 venv, pins, clone of the public repo, compile smoke test; ~/setup.log)"
         % (s.get("commit_sha") or "")[:9], "command": "ssh ... 'bash -s' < (cat > ~/gogeta-bootstrap.sh <<EOF <bootstrap.sh> EOF; bash ~/gogeta-bootstrap.sh <sha> --detach)",
         "ready": commit_ok},
        {"step": 4, "what": "upload by allowlist, sha256sum -c on arrival", "files": [{k: f[k] for k in ("name", "bytes", "sha256", "source")} for f in al["files"]],
         "errors": al["errors"], "total_bytes": al.get("total_bytes")},
        {"step": 5, "what": "launch under the cockpit wrapper (nohup setsid; refuses if any gogeta.train runs)", "command": p["command"]},
        {"step": 6, "what": "poll over ssh every %ds: nvidia-smi, processes, redacted log tail -> SQLite + incidents" % POLL_S},
        {"step": 7, "what": "pull run folder (sha256 verified) + manifest (provider/GPU/driver/dph/cost); reuse if a job waits, else destroy at once",
         "command": "ssh ... 'tar -C ~/fusiondance-data/runs -cf - -- <run>' | tar -xf - (staged); vastai destroy instance <id> -y --raw"},
    ]


def dispatch(jobs: list, session_id: str = "", approval_id: str = "", dry_run: bool = True, confirm: bool = False, checkpoint: bool = False) -> dict:
    """APPROVAL-GATED. Queue jobs (E10 --seed 0 | configs/x.json ... | queue ids) on a fleet session; instances are rented just-in-time
    (one per ready job, one GPU job per instance), reused while jobs wait, destroyed when none do. dry_run=true (default): plan + cost."""
    s = _resolve_session(session_id, approval_id)
    if not s:
        return {"dispatched": 0, "refused": "no single open fleet session%s: call fleet_up first" % (" for %s" % approval_id if approval_id else "")}
    a = next((x for x in cloud._load_approvals() if x.get("id") == s["approval_id"]), None)
    probs = cloud._approval_problems(a) if a else ["approval %s missing" % s["approval_id"]]
    parsed = [parse_job(x) for x in (jobs or [])]
    bad = ["%s: %s" % (x, p["error"]) for x, p in zip(jobs or [], parsed) if p.get("error")] or ([] if parsed else ["no jobs given"])
    for p in parsed:
        if not p.get("error"):
            al = data_allowlist(p["dataset"])
            bad += ["data allowlist for %s: %s" % (p["dataset"], e) for e in al["errors"]]
    if probs or bad:
        return {"dispatched": 0, "refused": "; ".join(probs + bad)}
    if dry_run:
        o = cloud.offers(s["gpu_name"], 1, 0.98, True, float(a.get("max_usd_per_hour") or 0), "dph_total", 20, int(s["disk_gb"]), float(s["min_cuda"]))
        top = (o.get("offers") or [None])[0]
        active = len([r for r in cloud._inst_rows(active_only=True) if r["approval_id"] == s["approval_id"]])
        remaining = float(a["max_usd_total"]) - cloud._used(a)
        per = (top["dph_total"] * float(s["max_hours"])) if top else None
        fit = int(remaining // per) if per else 0
        n_now = max(0, min(len(parsed), int(s["max_instances"]) - active, int(a["max_instances"]) - active, fit))
        return {"dispatched": 0, "dry_run": True, "session": s["id"], "jobs": [p["launch"] for p in parsed], "would_rent_now": n_now,
                "fleet_size_rule": "min(ready jobs, max_instances, budget left / (dph x max_hours))", "reservation_per_instance_usd": per and round(per, 4),
                "remaining_budget_usd": round(remaining, 2), "offers_live": {k: o.get(k) for k in ("count", "min_dph", "median_dph", "error")},
                "steps": plan_steps(parsed[0], s, top)}
    if confirm is not True:
        return {"dispatched": 0, "refused": "confirm must be exactly true (dispatch rents instances under %s)" % s["approval_id"]}
    ids = [_new_job(p["launch"], p["experiment"], p["dataset"], s["id"], checkpoint=checkpoint) for p in parsed]
    _event("info", "dispatch", "%d job(s) queued on %s: %s" % (len(ids), s["id"], "; ".join(p["launch"] for p in parsed)[:300]))
    return {"dispatched": len(ids), "jobs": ids, "session": s["id"], "tick": tick()}


def vast_run(experiment_args: str, approval_id: str, dry_run: bool = True, confirm: bool = False, checkpoint: bool = False,
             gpu: str = "RTX 3090", max_hours: float = 4.0, purpose: str = "") -> dict:
    """APPROVAL-GATED. One job on one rented GPU end to end (= fleet_up(1) + dispatch([args])): rent, setup, upload, run, poll, pull, destroy.
    dry_run=true (default) returns the 7 steps with exact commands and the max cost."""
    purpose = purpose or "vast_run %s" % experiment_args[:80]
    p = parse_job(experiment_args)
    if p.get("error"):
        return {"ran": False, "refused": p["error"]}
    up = fleet_up(1, gpu, approval_id, cloud.DEFAULT_IMAGE, max_hours, cloud.DEFAULT_DISK_GB, purpose, 13.0, confirm, dry_run)
    if not up.get("opened") and not up.get("dry_run"):
        return {"ran": False, "refused": up.get("refused"), "fleet_up": up}
    if dry_run:
        s = {"image": up["image"], "disk_gb": up["disk_gb"], "gpu_name": gpu, "min_cuda": 13.0, "commit_sha": up["commit"], "max_hours": max_hours}
        o = cloud.offers(gpu, 1, 0.98, True, 0.0, "dph_total", 5, cloud.DEFAULT_DISK_GB, 13.0)
        top = (o.get("offers") or [None])[0]
        return {"ran": False, "dry_run": True, "fleet_up": up, "steps": plan_steps(p, s, top), "max_cost_usd": top and round(top["dph_total"] * float(max_hours), 2)}
    d = dispatch([experiment_args], up["session_id"], approval_id, False, True, checkpoint)
    return {"ran": bool(d.get("dispatched")), "session": up["session_id"], "dispatch": d}


_LAST_LIVE = [0.0]


def gpu_bench(experiment: str) -> dict:
    """Measured tokens/s per GPU name for an experiment, from cloud runs the cockpit synced (median)."""
    import statistics as _st
    by = {}
    for j in _q("SELECT j.metrics, i.gpu_name FROM cloud_jobs j JOIN cloud_instances i ON i.id=j.instance_id WHERE j.status='done' AND j.experiment=?", (experiment,)):
        try:
            t = (json.loads(j["metrics"] or "{}") or {}).get("tokens_per_s")
        except ValueError:
            t = None
        if t:
            by.setdefault(j["gpu_name"], []).append(float(t))
    return {g: _st.median(v) for g, v in by.items()}


def rank_offers(experiment: str = "E1", max_dph: float = 0.0, provider: str = "vast", min_vram_gb: float = 0.0, limit: int = 200) -> dict:
    """READ-ONLY. Rank live single-GPU offers (Ampere+) for an experiment: by measured $ per 1M tokens where the cockpit has a benchmark for that
    GPU (synced cloud runs), else by $/h, among offers with VRAM >= the experiment's floor (E1 7.1 GB, E2 12.0 GB, else 12 GB)."""
    floor = float(min_vram_gb or VRAM_NEED_GB.get(experiment, DEFAULT_VRAM_GB))
    spec = {"gpu": "", "num_gpus": 1, "min_reliability": 0.98, "verified": True, "max_dph": float(max_dph or 0), "sort": "dph_total", "limit": limit,
            "disk_gb": cloud.DEFAULT_DISK_GB, "min_cuda": 13.0}
    res = cloud.providers.get(provider).search_offers(spec)
    if isinstance(res, dict):
        return res
    bench = gpu_bench(experiment)
    rows = []
    for o in res:
        if (o.get("gpu_ram_gb") or 0) < floor or (o.get("compute_cap") or 800) < 800:
            continue
        tps = bench.get(o["gpu_name"])
        o = dict(o, bench_tok_s=tps, usd_per_mtok=round(o["dph_total"] / (tps * 3600 / 1e6), 4) if tps else None)
        rows.append(o)
    rows.sort(key=lambda o: (o["usd_per_mtok"] is None, o["usd_per_mtok"] if o["usd_per_mtok"] is not None else o["dph_total"]))
    return {"experiment": experiment, "vram_floor_gb": floor, "benchmarked_gpus": bench, "count": len(rows), "offers": rows[:30],
            "best_gpu": rows[0]["gpu_name"] if rows else None,
            "rule": "measured $/token first (benchmarked GPUs), then $/h; VRAM >= floor; one SKU per comparison (the session pins the chosen GPU)"}


def _cap_dph(s: dict, a: dict) -> float:
    caps = [float(x) for x in (a.get("max_usd_per_hour"), s.get("max_dph")) if x]
    return min(caps) if caps else 0.0


def _pick_offer(s: dict, a: dict, exclude: set) -> dict | None:
    o = cloud.offers(s["gpu_name"], 1, 0.98, True, _cap_dph(s, a), "dph_total", 30, int(s["disk_gb"]), float(s["min_cuda"]), s.get("provider") or "vast")
    for x in o.get("offers") or []:
        if str(x.get("gpu_name")).lower() == s["gpu_name"].lower() and str(x.get("machine_id")) not in exclude:
            return x
    return None


def tick() -> dict:
    """One dispatcher pass (dispatch(), the watchdog and the TUI call it). Never rents beyond the approval; never leaves a fleet instance jobless."""
    import fcntl
    fd = open(cloud.cdir() / "fleet-tick.lock", "a+")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"skipped": "another tick is running"}
        return _tick()
    finally:
        fd.close()


def _tick() -> dict:
    now = _now()
    acts, findings = [], []
    act = cloud._inst_rows(active_only=True)
    if act and now - _LAST_LIVE[0] >= LIVE_SYNC_S:
        _LAST_LIVE[0] = now
        cloud.sync_live()
        act = cloud._inst_rows(active_only=True)
    for r in act:
        st = r["state"]
        if st == "provisioning" and r.get("ssh_host") and (r.get("live_status") or "").lower() == "running":
            if r.get("mode") == "fleet":
                s = session(r.get("session_id"))
                res = start_setup(r, (s or {}).get("commit_sha") or "")
                if res["ok"]:
                    cloud.set_state(r["id"], "setup")
                acts.append("setup %s: %s" % (r["label"], "started" if res["ok"] else res["output"][-80:]))
            else:
                cloud.set_state(r["id"], "ready")
        elif st == "setup" and now - (r.get("last_poll_epoch") or 0) >= min(POLL_S, 30):
            cloud.inst_update(r["id"], last_poll_epoch=now)
            c = check_setup(r)
            if c["state"] == "done":
                if r.get("mode") == "fleet" and r.get("job_id"):
                    cloud.set_state(r["id"], "staging")
                    _start_phase("push", r["id"])
                else:
                    cloud.set_state(r["id"], "ready")
                acts.append("setup done on %s" % r["label"])
            elif c["state"] == "failed":
                acts.append(str(replace_bad_host(r, "setup failed: %s" % c["detail"][-100:]) if r.get("mode") == "fleet" else
                                cloud.set_state(r["id"], "ready", setup_state="failed")))
        elif st == "busy" and now - (r.get("last_poll_epoch") or 0) >= POLL_S:
            _, f = poll_instance(r)
            findings += f
        elif st == "pulling" and not _sync_mode() and not _worker_alive(r["id"]):
            _start_phase("pull", r["id"])
        elif st == "staging" and not _sync_mode() and not _worker_alive(r["id"]) and now - (r.get("state_since") or now) > 60:
            _start_phase("push", r["id"])
        elif st in ("idle", "ready") and not r.get("job_id") and r.get("mode") == "fleet":
            acts.append(str(_after_job(r)))
        elif st in ("idle", "ready") and r.get("ssh_host") and now - (r.get("last_poll_epoch") or 0) >= POLL_S:
            _, f = poll_instance(r)
            findings += f
    for s in open_sessions():
        a = next((x for x in cloud._load_approvals() if x.get("id") == s["approval_id"]), None)
        ready = _ready_jobs(s["id"])
        for r in [x for x in cloud._inst_rows(active_only=True, session_id=s["id"]) if x["state"] in ("idle", "ready") and not x.get("job_id")]:
            if ready:
                acts.append(str(_assign(r, ready.pop(0))))
        if not ready or not a or cloud._approval_problems(a):
            _close_session_if_done(s["id"])
            continue
        exclude = set(json.loads(s.get("exclude_machines") or "[]")) | {str(x.get("machine_id")) for x in cloud._inst_rows(active_only=True)}
        for j in ready:
            if len(cloud._inst_rows(active_only=True, session_id=s["id"])) >= int(s["max_instances"]):
                break
            off = _pick_offer(s, a, exclude)
            if not off:
                acts.append("no %s offer within the per-hour cap" % s["gpu_name"])
                break
            res = cloud.rent(off["id"], float(s["max_hours"]), 0, "%s | %s" % (s["purpose"], j["launch"]), s["image"], int(s["disk_gb"]),
                             s["approval_id"], confirm=True, session_id=s["id"], job_id=j["id"], provider=s.get("provider") or "vast")
            if not res.get("rented"):
                acts.append("rent refused: %s" % str(res.get("refused") or res.get("error"))[:160])
                break  # budget / count exhausted: the job stays queued
            cloud.inst_update(res["instance_id"], mode="fleet")
            exclude.add(str(off.get("machine_id")))
            _upd("cloud_jobs", j["id"], instance_id=res["instance_id"], status="provisioning")
            acts.append("rented %s for %s ($%.4f/h)" % (res["label"], j["launch"], res["dph"]))
    return {"actions": acts, "findings": findings}


# --------------------------------------------------------------------------- watchdog guards (always allowed: they stop spending)
def watch(now: float | None = None) -> list:
    """Watchdog `cloud` detector. fleet: destroy idle > 5 min, stuck setup > 15 min (job retried once elsewhere), unreachable 20 min.
    fleet+manual: destroy over max_hours, over the approval cap, or approval invalid. manual/attached idle > N min: SUGGEST destroying.
    attached: suggestions only. Each action is an incident finding. Never touches non-cockpit instances."""
    now = _now() if now is None else now
    out = []
    apprs = {a["id"]: a for a in cloud._load_approvals()}
    rows = cloud._inst_rows(active_only=True)
    burn = {}
    for r in rows:
        burn[r["approval_id"]] = burn.get(r["approval_id"], 0.0) + float(r["dph"] or 0)
    over = set()
    for aid, a in apprs.items():
        spent = sum(cloud.inst_cost(x, now) for x in cloud._inst_rows() if x["approval_id"] == aid)
        if spent + burn.get(aid, 0.0) * 120 / 3600 >= float(a.get("max_usd_total") or 0):
            over.add(aid)
    for r in rows:
        if not str(r.get("label") or "").startswith(cloud.LABEL_PREFIX):
            continue
        mode = r.get("mode") or "fleet"
        a = apprs.get(r["approval_id"])
        age = now - (r["created_epoch"] or now)
        idle_since = max(r.get("last_busy_epoch") or 0, r.get("state_since") or 0) or r["created_epoch"]
        jobless = r["state"] in ("ready", "idle") and not r.get("job_id")
        why = None
        if r.get("max_hours") and age > float(r["max_hours"]) * 3600:
            why = ("overtime", "ran %.2f h > max_hours %.2f" % (age / 3600, r["max_hours"]))
        elif r["approval_id"] and r["approval_id"] in over:
            why = ("budget", "approval %s spend would pass its cap $%.2f" % (r["approval_id"], float(a.get("max_usd_total") or 0)))
        elif mode != "attached" and (not a or cloud._approval_problems(a, now)):
            why = ("unapproved", "approval %s no longer valid: %s" % (r["approval_id"], "; ".join(cloud._approval_problems(a, now)) if a else "missing"))
        elif mode == "fleet" and jobless and now - idle_since > IDLE_S:
            why = ("idle", "no running job for %.1f min (limit %d)" % ((now - idle_since) / 60, IDLE_S // 60))
        elif mode != "fleet" and jobless and not _queued_for(r) and now - idle_since > IDLE_KILL_S:
            why = ("idle", "AGENTS 0h: no running or queued job for %.1f min (limit %d): pulled, then destroyed" % ((now - idle_since) / 60, IDLE_KILL_S // 60))
        elif mode == "fleet" and r["state"] in ("provisioning", "setup") and age > STUCK_S:
            why = ("stuck", "still %s after %.1f min (limit %d): bad host" % (r["state"], age / 60, STUCK_S // 60))
        elif r["state"] in ("busy", "ready", "idle") and r.get("ssh_host") and now - (r.get("last_poll_ok_epoch") or r["created_epoch"]) > UNREACH_S:
            why = ("unreachable", "no successful ssh poll for %.0f min" % ((now - (r.get("last_poll_ok_epoch") or r["created_epoch"])) / 60))
        elif r.get("last_poll_epoch") and r.get("ssh_host") and r["last_poll_epoch"] - (r.get("last_poll_ok_epoch") or r["created_epoch"]) > UNREACH_FLAG_S:
            out.append({"kind": "cloud_unreachable", "subject": "vast:%s" % r["label"], "subject_type": "cloud-instance", "severity": "high",
                        "key": "%s-unreach-%d" % (r["id"], int(r.get("last_poll_ok_epoch") or r["created_epoch"])), "hint": "transient",
                        "summary": "%s (%s, $%.3f/h): ssh polls failing for %.0f min (destroyed at %d min)" % (
                            r["label"], r["gpu_name"], r["dph"], (r["last_poll_epoch"] - (r.get("last_poll_ok_epoch") or r["created_epoch"])) / 60, UNREACH_S // 60),
                        "actions": ["flagged"], "repro": "gogeta_cloud_status %s" % r["label"]})
            continue
        if not why:
            continue
        if why[0] == "stuck":
            res = replace_bad_host(r, why[1])
            acts = ["destroyed %s: %s" % (r["label"], "ok" if res["destroy"].get("destroyed") else res["destroy"]),
                    "job retried on another offer" if res["retried"] else "no retry left (or no job)"]
        else:
            res = destroy_instance(r, why[1], "watchdog")
            acts = ["destroyed %s: %s" % (r["label"], ("ok, cost $%.4f" % res.get("cost_usd", 0)) if res.get("destroyed") else str(res)[:150])]
        out.append({"kind": "cloud_destroy_%s" % why[0], "subject": "vast:%s" % r["label"], "subject_type": "cloud-instance", "key": "%s-%s" % (r["id"], why[0]),
                    "severity": "high" if why[0] in ("budget", "unapproved", "overtime") else "medium", "hint": "resource",
                    "summary": "watchdog destroyed %s (%s, $%.4f/h): %s" % (r["label"], r["gpu_name"], r["dph"], why[1]), "actions": acts,
                    "repro": "gogeta_fleet_status", "evidence": {"instance": r["label"], "approval": r["approval_id"], "state": r["state"]}})
    for s_ in open_sessions():  # a session whose work is over never keeps a box (AGENTS 0h)
        if not _q("SELECT 1 FROM cloud_jobs WHERE session_id=? AND status IN ('queued','provisioning','setup','staging','running','pulling') LIMIT 1", (s_["id"],)) \
                and _q("SELECT 1 FROM cloud_jobs WHERE session_id=? LIMIT 1", (s_["id"],)) and cloud._inst_rows(active_only=True, session_id=s_["id"]):
            fin = finish_session(s_["id"], "no work left")
            out.append({"kind": "cloud_session_finished", "subject": "fleet:%s" % s_["id"], "subject_type": "cloud-session", "severity": "info",
                        "key": s_["id"], "hint": "resource", "summary": "session %s: no work left; pulled and destroyed %d box(es)" % (s_["id"], len(fin["destroyed"])),
                        "actions": [str(x)[:120] for x in fin["destroyed"]]})
    return out


def finish_session(session_id: str, reason: str) -> dict:
    """Pull whatever exists, then destroy every box of the session and close it (used on completion AND on failure)."""
    out = []
    for r in cloud._inst_rows(active_only=True, session_id=session_id):
        out.append({"label": r["label"], **{k: v for k, v in destroy_instance(r, "session end: %s" % reason, "fleet").items() if k in ("destroyed", "cost_usd", "error", "pulled_first")}})
    for j in jobs(session_id, "queued"):
        _upd("cloud_jobs", j["id"], status="cancelled", error="session ended: %s" % reason[:100])
    if (session(session_id) or {}).get("status") == "open":
        _upd("cloud_sessions", session_id, status="done")
    _event("info", "fleet_session_end", "session %s ended (%s): %d box(es) destroyed" % (session_id, reason[:80], len(out)))
    return {"destroyed": out}


# --------------------------------------------------------------------------- manual control: vast up / setup / push-data / run / stop / pull / down / attach
def _managed(ref: str):
    r, managed = resolve(ref)
    if not r:
        return None, {"refused": "unknown cockpit instance %s (never touched)" % ref}
    if not managed:
        return None, {"refused": "%s is not in the cockpit registry (read-only: status, tail, pull). Register it with gogeta_vast_attach first." % r["label"]}
    return r, None


def vast_up(gpu: str = "RTX 3090", max_hours: float = 4.0, approval_id: str = "", purpose: str = "", image: str = cloud.DEFAULT_IMAGE,
            disk_gb: int = cloud.DEFAULT_DISK_GB, min_cuda: float = 13.0, confirm: bool = False, dry_run: bool = True, provider: str = "vast",
            max_dph: float = 0.0) -> dict:
    """APPROVAL-GATED, SPENDS MONEY. `vast up`: rent the cheapest exact-SKU offer within the approval's per-hour cap (mode manual).
    dry_run=true (default) shows the offer, command and cost. Then: vast_setup -> vast_push_data -> vast_run_job -> vast_pull -> vast_down."""
    a = next((x for x in cloud._load_approvals() if x.get("id") == approval_id), None) if approval_id else _approval_for({})
    if not a:
        return {"rented": False, "refused": "no (single valid) approval%s: the human creates one (TUI Cloud tab or cockpit.sh --approve-spend)" % (
            " %s" % approval_id if approval_id else "")}
    s = {"gpu_name": gpu, "disk_gb": int(disk_gb), "min_cuda": float(min_cuda), "provider": provider, "max_dph": max_dph}
    off = _pick_offer(s, a, {str(x.get("machine_id")) for x in cloud._inst_rows(active_only=True)})
    if not off:
        return {"rented": False, "refused": "no %s offer within $%.3f/h (min cuda %s)" % (gpu, float(a.get("max_usd_per_hour") or 0), min_cuda)}
    res = cloud.rent(off["id"], max_hours, 0, purpose or "cloud up %s" % gpu, image, disk_gb, a["id"], confirm=confirm, dry_run=dry_run, provider=provider)
    if res.get("rented"):
        cloud.inst_update(res["instance_id"], mode="manual")
        res["next"] = "vast_setup %s (after it is running), then vast_push_data, vast_run_job" % res["label"]
    return res


def vast_attach(instance_id: str, approval_id: str = "", confirm: bool = False) -> dict:
    """STATE-CHANGING (registry only, spends nothing). Register a live gogeta-cockpit-* instance rented outside the cockpit, so it can be
    controlled (setup/push/run/stop/destroy). Mode 'attached': the watchdog only SUGGESTS destroying it. Needs confirm=true."""
    live = cloud.live_all()
    if live is None:
        return {"attached": False, "error": "provider listing unavailable"}
    L = next((x for x in live if str(x.get("id")) == str(instance_id)), None)
    if not L or not str(L.get("label") or "").startswith(cloud.LABEL_PREFIX):
        return {"attached": False, "refused": "only live instances labelled %s-* can be attached" % cloud.LABEL_PREFIX}
    if cloud.inst(str(instance_id)):
        return {"attached": False, "refused": "already in the registry"}
    if confirm is not True:
        return {"attached": False, "refused": "confirm must be exactly true", "label": L.get("label"), "dph": L.get("dph_total")}
    h, p = endpoint(L)
    st = float(L.get("start_date") or _now())
    cloud._inst_insert(id=str(instance_id), label=L["label"], approval_id=approval_id or None, gpu_name=str(L.get("gpu_name")), num_gpus=int(L.get("num_gpus") or 1),
                       dph=float(L.get("dph_total") or 0), max_hours=None, reserved_usd=0, purpose="attached", image=str(L.get("image") or ""),
                       disk_gb=int(float(L.get("disk_gb") or 0)), created=cloud._iso(st), created_epoch=st, state="ready", state_since=_now(),
                       driver=str(L.get("driver_version") or ""), ssh_host=h, ssh_port=p, mode="attached", live_status=str(L.get("actual_status") or ""),
                       last_busy_epoch=_now(), provider=L.get("provider", "vast"), meta=json.dumps(cloud.live_meta(L)))
    _event("info", "attach", "attached %s (%s, $%.4f/h)" % (L["label"], L.get("gpu_name"), float(L.get("dph_total") or 0)))
    return {"attached": True, "label": L["label"], "mode": "attached"}


def vast_setup(instance: str, confirm: bool = False) -> dict:
    """STATE-CHANGING on the remote (no spend). `vast setup`: start the background environment setup at the current PUSHED commit.
    Progress: gogeta_vast_status (setup: running/done/failed; log ~/setup.log on the remote). Needs confirm=true."""
    r, err = _managed(instance)
    if err:
        return err
    commit, ok = _head_commit()
    if not ok:
        return {"started": False, "refused": "HEAD %s is not on origin: push first" % commit[:9]}
    if confirm is not True:
        return {"started": False, "refused": "confirm must be exactly true", "commit": commit, "script": setup_script(commit)}
    res = start_setup(r, commit)
    if res["ok"]:
        cloud.set_state(r["id"], "setup")
    return {"started": res["ok"], "commit": commit, "output": res["output"]}


def vast_push_data(instance: str, dataset: str = "code-py-v1", confirm: bool = False, wait: bool = False) -> dict:
    """STATE-CHANGING on the remote (no spend). `vast push-data`: upload the dataset by allowlist (manifest files + meta.json) and verify
    sha256 on arrival. Runs as a background worker unless wait=true. Needs confirm=true."""
    r, err = _managed(instance)
    if err:
        return err
    al = data_allowlist(dataset)
    if al["errors"]:
        return {"pushed": False, "refused": "; ".join(al["errors"])}
    if confirm is not True:
        return {"pushed": False, "refused": "confirm must be exactly true", "files": [{k: f[k] for k in ("name", "bytes", "sha256")} for f in al["files"]]}
    cloud.inst_update(r["id"], push_dataset=dataset, push_state="running")
    cloud.set_state(r["id"], "staging")
    if wait or _sync_mode():
        return {"pushed": True, **_phase_push(cloud.inst(r["id"]))}
    _start_phase("push", r["id"])
    return {"pushed": "started", "note": "background upload; watch gogeta_vast_status (push: running/done/failed)"}


def vast_run_job(instance: str, spec: str, confirm: bool = False, checkpoint: bool = False) -> dict:
    """STATE-CHANGING on the remote (no new spend). `vast run <config|E-id> [--seed N --lr X --max-tokens N --no-compile]`: launch ONE job
    (nohup, cockpit wrapper). Refused if any gogeta.train already runs there, setup is not done, or its dataset is not verified there."""
    r, err = _managed(instance)
    if err:
        return err
    p = parse_job(spec)
    if p.get("error"):
        return {"launched": False, "refused": p["error"]}
    if p["dataset"] not in json.loads(r.get("data_ok") or "[]"):
        return {"launched": False, "refused": "dataset %s not verified on %s yet: vast_push_data first" % (p["dataset"], r["label"])}
    if confirm is not True:
        return {"launched": False, "refused": "confirm must be exactly true", "command": p["command"]}
    if r.get("job_id") and (job(r["job_id"]) or {}).get("status") in JOB_LIVE:
        jid = _new_job(p["launch"], p["experiment"], p["dataset"], None, r["id"], "queued", checkpoint)
        return {"launched": False, "queued": jid, "note": "a cockpit job runs there; this one starts when it ends (then the box is pulled down when the queue is empty)"}
    jid = _new_job(p["launch"], p["experiment"], p["dataset"], None, r["id"], "staging", checkpoint)
    return _launch(r, job(jid))


def vast_stop_job(instance: str, confirm: bool = False) -> dict:
    """STATE-CHANGING on the remote. Stop the cockpit job (SIGTERM to its process group). Never kills jobs the cockpit did not start. Needs confirm=true."""
    r, err = _managed(instance)
    if err:
        return err
    if confirm is not True:
        return {"stopped": False, "refused": "confirm must be exactly true"}
    rc, out = ssh(r, STOP_SCRIPT, timeout=60)
    j = job(r.get("job_id"))
    if j and "STOPPED" in out:
        _upd("cloud_jobs", j["id"], status="stopped", exit_code=143, ended_epoch=_now(), error="stopped by operator")
        cloud.set_state(r["id"], "idle", job_id=None, last_busy_epoch=_now())
    _event("warn", "remote_stop", "stop on %s: %s" % (r["label"], core.redact(out).strip()[:120]))
    return {"stopped": "STOPPED" in out, "output": core.redact(out).strip()[:300]}


def vast_pull(instance: str, run: str = "", checkpoint: bool = False) -> dict:
    """Writes local runs/ only (READ-ONLY on the remote; works for unmanaged gogeta-cockpit-* instances too). `vast pull`: copy a finished
    run folder (default: the one named by the current log's `done:` line) into ~/fusiondance-data/runs/, sha256-verified, with a manifest."""
    r, managed = resolve(instance)
    if not r:
        return {"error": "unknown cockpit instance %s" % instance}
    if not run:
        p, _ = poll_instance(r)
        run = Path(p.get("metrics", {}).get("run_dir") or "").name if p.get("ok") else ""
        if not run:
            return {"pulled": False, "refused": "no finished run found in the current log (pass run=<name>); recent: %s" % (p.get("runs") or [])[:5]}
    j = job(r.get("job_id")) if managed else None
    res = pull_run(r, run, checkpoint, j)
    return {"pulled": bool(res.get("ok")), **res}


def vast_down(instance: str, confirm: bool = False, dry_run: bool = False) -> dict:
    """STATE-CHANGING (stops spending). `vast down`: destroy ONE cockpit instance (registry: launched or attached). Unmanaged instances
    are never touched. dry_run shows the command and cost so far. Needs confirm=true (the TUI asks first)."""
    r, err = _managed(instance)
    if err:
        return err
    if dry_run:
        return cloud.destroy(r["id"], dry_run=True, reason="vast down")
    if confirm is not True:
        return {"destroyed": False, "refused": "confirm must be exactly true", "cost_so_far_usd": cloud.inst_cost(r)}
    return destroy_instance(r, "vast down", "operator")


# --------------------------------------------------------------------------- provider-agnostic lifecycle: up -> setup -> data -> run -> status -> pull -> down
def cloud_up(gpu: str = "RTX 3090", provider: str = "vast", max_hours: float = 4.0, max_dph: float = 0.0, purpose: str = "", approval_id: str = "",
             image: str = cloud.DEFAULT_IMAGE, disk_gb: int = cloud.DEFAULT_DISK_GB, confirm: bool = False, dry_run: bool = True) -> dict:
    """SPENDS MONEY, BUDGET-GATED. Rent the cheapest exact-GPU offer on `provider` within the human's standing budget (the single valid
    approval, or approval_id) and below max_dph. Refused with the reason when projected spend would leave the budget. dry_run=true (default)."""
    if gpu.strip().lower() == "auto":
        rk = rank_offers("E1", max_dph, provider)
        if not rk.get("best_gpu"):
            return {"rented": False, "refused": "no offer above the VRAM floor within the price cap", "ranking": rk}
        gpu = rk["best_gpu"]
    return vast_up(gpu, max_hours, approval_id, purpose, image, disk_gb, 13.0, confirm, dry_run, provider, max_dph)


def cloud_setup(instance: str, confirm: bool = False) -> dict:
    """STATE-CHANGING on the remote. Pipe cockpit/remote/bootstrap.sh <pushed sha> (detached; ~/setup.log; idempotent). Needs confirm=true."""
    return vast_setup(instance, confirm)


def cloud_data(instance: str, dataset: str = "code-py-v1", confirm: bool = False, wait: bool = False) -> dict:
    """STATE-CHANGING on the remote. Stage a dataset: fetch_data.sh rebuild + manifest check (code-py-v1), else/fallback the allowlisted
    sha256-checked upload. Background unless wait=true. Needs confirm=true."""
    return vast_push_data(instance, dataset, confirm, wait)


def cloud_run(instance: str, spec: str, confirm: bool = False, checkpoint: bool = False) -> dict:
    """STATE-CHANGING on the remote. Launch one job ('E10 --seed 0' or 'configs/x.json --lr 3e-3 ...') under the cockpit wrapper. Needs confirm=true."""
    return vast_run_job(instance, spec, confirm, checkpoint)


def cloud_status(instance: str = "", refresh: bool = True) -> dict:
    """READ-ONLY. Monitor one or all gogeta-cockpit-* instances on every provider (metadata, GPU, job, val/loss, ETA, $, budget bar, vs N1X)."""
    return vast_status(instance, refresh)


def cloud_logs(instance: str, n: int = 60, refresh: bool = True) -> dict:
    """READ-ONLY. Redacted tail of the current training log on an instance."""
    return vast_tail(instance, n, refresh)


def cloud_pull(instance: str, run: str = "", checkpoint: bool = False) -> dict:
    """Pull a finished run folder (sha256-verified) + manifest into local runs/. Read-only on the remote."""
    return vast_pull(instance, run, checkpoint)


def cloud_stop(instance: str, confirm: bool = False) -> dict:
    """STATE-CHANGING on the remote. Stop the cockpit job there (never other processes). Needs confirm=true."""
    return vast_stop_job(instance, confirm)


def cloud_down(instance: str, confirm: bool = False, dry_run: bool = False) -> dict:
    """STOPS SPENDING; always allowed for cockpit-labelled registry instances. Destroy one instance (logged with its price). Needs confirm=true."""
    return vast_down(instance, confirm, dry_run)


def cloud_session(config_list: list, gpu: str = "RTX 3090", max_price: float = 0.0, provider: str = "vast", max_hours: float = 4.0,
                  approval_id: str = "", checkpoint: bool = False, confirm: bool = False, dry_run: bool = True, wait: bool = False) -> dict:
    """SPENDS MONEY, BUDGET-GATED. One shot: up -> setup -> data -> run each job in config_list ('E10 --seed 0', 'configs/x.json --lr 3e-3', ...)
    -> pull each -> down, on ONE instance of one GPU SKU (jobs run one after another; the box is destroyed when the list is done).
    max_price caps $/h. Progress goes to the logdb (gogeta_events source=fleet) and gogeta_fleet_status; the watchdog drives it.
    dry_run=true (default) returns the plan and the cost bound."""
    a = next((x for x in cloud._load_approvals() if x.get("id") == approval_id), None) if approval_id else _approval_for({})
    if not a:
        return {"started": False, "refused": "no (single valid) standing budget/approval: the human sets one (TUI Cloud tab or cockpit.sh --approve-spend)"}
    parsed = [parse_job(x) for x in config_list or []]
    bad = ["%s: %s" % (x, q["error"]) for x, q in zip(config_list or [], parsed) if q.get("error")]
    if bad or not parsed:
        return {"started": False, "refused": "; ".join(bad) or "config_list is empty"}
    purpose = "cloud_session: %s" % "; ".join(q["launch"] for q in parsed)[:200]
    if gpu.strip().lower() == "auto":
        need = max(VRAM_NEED_GB.get(q["experiment"], DEFAULT_VRAM_GB) for q in parsed)
        rk = rank_offers(parsed[0]["experiment"], max_price, provider, need)
        if not rk.get("best_gpu"):
            return {"started": False, "refused": "no offer with >= %.1f GB VRAM within $%s/h" % (need, max_price or "any"), "ranking": rk}
        gpu = rk["best_gpu"]
    up = fleet_up(1, gpu, a["id"], cloud.DEFAULT_IMAGE, max_hours, cloud.DEFAULT_DISK_GB, purpose, 13.0, confirm, dry_run, provider, max_price)
    if not up.get("opened") and not up.get("dry_run"):
        return {"started": False, "refused": up.get("refused")}
    if dry_run:
        sess = {"image": up["image"], "disk_gb": up["disk_gb"], "gpu_name": gpu, "min_cuda": 13.0, "commit_sha": up["commit"], "max_hours": max_hours,
                "provider": provider, "max_dph": max_price}
        off = _pick_offer(sess, a, set())
        return {"started": False, "dry_run": True, "approval": a["id"], "jobs": [q["launch"] for q in parsed], "offer": off,
                "max_cost_usd": off and round(off["dph_total"] * float(max_hours), 2), "steps": plan_steps(parsed[0], sess, off),
                "note": "one instance, jobs sequential (reuse, no second setup); destroyed right after the last pull"}
    sid = up["session_id"]
    if wait:
        try:  # AGENTS 0h: whatever happens, pull what exists and destroy the box at the end
            d = dispatch(config_list, sid, a["id"], False, True, checkpoint)
            deadline = _now() + float(max_hours) * 3600 + 600
            while _now() < deadline and _q("SELECT 1 FROM cloud_jobs WHERE session_id=? AND status IN ('queued','provisioning','setup','staging','running','pulling') LIMIT 1", (sid,)):
                tick()
                watch()
                time.sleep(2 if _sync_mode() else 20)
        finally:
            fin = finish_session(sid, "cloud_session finished")
        return {"started": True, "session": sid, "jobs": [dict(id=j["id"], launch=j["launch"], status=j["status"], result_dir=j.get("result_dir"))
                                                          for j in jobs(sid)], "teardown": fin}
    d = dispatch(config_list, sid, a["id"], False, True, checkpoint)
    try:
        try:
            from . import watchdog as _wd
        except ImportError:
            import watchdog as _wd  # type: ignore
        wd = _wd.start() if not _wd.status()["running"] else {"running": True}
    except Exception as ex:  # noqa: BLE001
        wd = {"error": repr(ex)}
    return {"started": bool(d.get("dispatched")), "session": sid, "dispatch": d, "watchdog_running": bool(wd.get("running")),
            "progress": "gogeta_fleet_status / gogeta_cloud_status; events source=fleet"}


# --------------------------------------------------------------------------- fleet views
def fleet_status(refresh: bool = False) -> dict:
    """READ-ONLY. One row per cockpit GPU (mode, state, job, step/total, loss, val/loss, tok/s, ETA, temp, util, $ so far), job costs
    (setup/run/teardown), sessions, and spend vs the approval cap. refresh=true polls busy instances over SSH first."""
    now = _now()
    if refresh:
        for r in cloud._inst_rows(active_only=True):
            if r["state"] == "busy" or r.get("mode") in ("manual", "attached"):
                poll_instance(r)
    rows = [status_row(r, None, now) for r in cloud._inst_rows() if r["state"] in cloud.ACTIVE_STATES or now - (r.get("destroyed_epoch") or 0) < 7200]
    js = []
    for j in _q("SELECT * FROM cloud_jobs ORDER BY queued_epoch DESC LIMIT 60"):
        r = cloud.inst(j["instance_id"]) if j.get("instance_id") else None
        js.append({"id": j["id"], "session": j["session_id"], "launch": j["launch"], "status": j["status"], "attempts": j["attempts"], "instance": r and r["label"],
                   "remote_run": j.get("remote_run"), "verified": bool(j.get("verified")), "error": j.get("error"),
                   "cost": job_costs(j, r, now) if r and j.get("started_epoch") else None})
    sess = []
    for s in _q("SELECT * FROM cloud_sessions ORDER BY created DESC LIMIT 10"):
        insts = [x for x in cloud._inst_rows() if x.get("session_id") == s["id"]]
        sess.append({"id": s["id"], "status": s["status"], "gpu": s["gpu_name"], "approval": s["approval_id"], "max_instances": s["max_instances"],
                     "cost_usd": round(sum(cloud.inst_cost(x, now) for x in insts), 4), "instances_used": len(insts), "jobs": _counts(s["id"])})
    act = [r for r in rows if r["state"] in cloud.ACTIVE_STATES]
    return {"instances": rows, "running": len(act), "burn_usd_per_h": round(sum(r["dph"] or 0 for r in act), 4),
            "cost_so_far_usd": round(sum(r["accrued_usd"] for r in rows), 4), "jobs": js, "sessions": sess, "approvals": cloud.approvals()["approvals"],
            "fleet_power_level_tok_s": sum(r["tok_s"] or 0 for r in act if r["job_running"]), "lifetime": lifetime(),
            "policy": "fleet: just-in-time, idle %d min / stuck %d min guards; manual/attached: idle > %d min -> suggest destroy" % (
                IDLE_S // 60, STUCK_S // 60, SUGGEST_IDLE_S // 60)}


def _counts(sid):
    d = {}
    for j in jobs(sid):
        d[j["status"]] = d.get(j["status"], 0) + 1
    return d


def fleet_down(confirm: bool = False, dry_run: bool = False, session_id: str = "") -> dict:
    """STATE-CHANGING (stops spending). Cancel queued jobs and destroy every cockpit-REGISTRY instance (one session, or all; attached
    ones included). Running jobs are lost: dry_run lists them. Needs confirm=true. Never touches unmanaged/foreign instances."""
    rows = [r for r in cloud._inst_rows(active_only=True) if not session_id or r.get("session_id") == session_id]
    q = [j for j in jobs(session_id) if j["status"] == "queued"]
    run = [j for j in jobs(session_id) if j["status"] in JOB_LIVE]
    plan = {"destroy": [{"instance": r["id"], "label": r["label"], "mode": r.get("mode"), "cost_usd": cloud.inst_cost(r)} for r in rows],
            "cancel_queued": [j["id"] for j in q], "running_jobs_lost": [j["launch"] for j in run]}
    if dry_run:
        return {"down": False, "dry_run": True, **plan}
    if confirm is not True:
        return {"down": False, "refused": "confirm must be exactly true", **plan}
    for j in q:
        _upd("cloud_jobs", j["id"], status="cancelled", error="fleet_down")
    res = [destroy_instance(r, "fleet_down", "operator") for r in rows]
    for s in open_sessions():
        if not session_id or s["id"] == session_id:
            _upd("cloud_sessions", s["id"], status="down")
    return {"down": True, **plan, "results": [{"label": r["label"], "destroyed": x.get("destroyed"), "note": x.get("refused") or x.get("error")} for r, x in zip(rows, res)]}


def brief_lines() -> list:
    L = []
    now = _now()
    for r in cloud._inst_rows(active_only=True):
        s = status_row(r, None, now)
        L.append("  %s %s [%s/%s] $%.3f/h, $%.2f so far%s%s%s" % (
            s["label"], s["gpu"], s["mode"], s["state"], s["dph"] or 0, s["accrued_usd"], (" | %s" % s["job"][:50]) if s["job"] else "",
            (" step %s/%s val %s ETA %smin" % (s["step"], s["total"], s["val_loss"], s["eta_min"])) if s.get("step") is not None else "",
            (" | IDLE %.0f min (auto-down at %d)" % (s["idle_min"], (IDLE_S if s["mode"] == "fleet" else IDLE_KILL_S) // 60)) if (not s["job_running"] and s["state"] in ("ready", "idle")) else ""))
    for s in open_sessions():
        c = _counts(s["id"])
        L.append("FLEET %s (%s, max %s): jobs %s" % (s["id"], s["gpu_name"], s["max_instances"], ", ".join("%s=%d" % kv for kv in sorted(c.items())) or "none"))
    return L


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 3 and a[0] == "--worker":
        try:
            print(json.dumps(run_phase(a[1], a[2]), default=str))
        finally:
            (_workers() / ("%s.pid" % a[2])).unlink(missing_ok=True)  # our own pid file
        return 0
    if a and a[0] == "--status":
        print(json.dumps(vast_status(a[1] if len(a) > 1 else ""), indent=1, default=str))
        return 0
    print("usage: fleet.py --status [instance] | --worker push|pull <instance>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
