"""Code execution for the eval suite: `post_training/rewards.py`'s sandbox plus a private filesystem view.

Every generated program the eval suite runs goes through `run_tests()` here. Nothing in
evals/ calls exec/eval on model output, and lm-eval's own in-process code tasks
(`humaneval`, `mbpp`, ... which need HF_ALLOW_CODE_EVAL=1) are never used.

What this reuses from post_training/rewards.py (unchanged): the in-child harness (`_HARNESS`:
nonce-tagged result line, per-assert scoring), the rlimits (`_limits`: CPU time, address
space, file size, no core, 64 fds, NPROC), the privilege drop to `nobody` when we are root,
the base interpreter in isolated mode (`python3 -I`, stdlib + system packages only), the
PATH/LANG-only environment, and the wall-clock kill of the whole process group.

What it adds, because of a gap measured on this machine (2026-11-22): rewards.py documents
that the filesystem is not isolated, and on WSL that is worse than it sounds: the Windows
drive is mounted at /mnt/c as 9p/drvfs with every file shown as mode 777, so a child running
as `nobody` CAN WRITE to /mnt/c, i.e. to this repo and the whole Windows profile (checked
with `setpriv --reuid=65534 ... sh -c "echo hi > /mnt/c/.../scratchpad/x"`: it succeeded).
So here the child runs in a new user + network + MOUNT namespace (`unshare -rnm`) and, before
the harness starts, tmpfs is mounted over /mnt, /tmp, /dev/shm and /run (hides /mnt/c and
/mnt/wsl, the WSL interop socket under /run, and makes the scratch dirs private). If any
mount fails, the child exits before running anything (fail closed). The harness source is
passed with `-c`, so nothing is written to disk.

Still NOT isolated: the rest of the root filesystem is visible read-only-by-permission (the
child is `nobody`; it can read world-readable files such as /etc/passwd and system Python
packages, but cannot write outside the private tmpfs mounts). It is a process sandbox, not a
security boundary. Use a container / nsjail for anything adversarial.

`self_test()` probes all of this and run_suite.py refuses to run code tasks unless every probe
passes.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
from functools import lru_cache

from post_training.rewards import (IS_LINUX, NOBODY, _HARNESS, _child_python, _drop_privileges,  # noqa: F401
                                   _limits, extract_code, score)

HIDE = ("/mnt", "/tmp", "/dev/shm", "/run")
# Runs as root-in-a-new-userns (outer `unshare -rnm`), mounts the private tmpfs views, then
# enters a NESTED user + mount namespace as uid/gid 65534 before exec'ing Python. Without the
# nested step the child would be ns-root with CAP_SYS_ADMIN and could simply `umount /mnt`
# (verified here: the umount works and /mnt/c reappears). After the nested step, the execve of
# a non-root uid clears every capability (CapEff = 0, verified) and the inherited mounts are
# locked, so the private view cannot be undone.
_PRELUDE = ("set -e; for d in " + " ".join(HIDE) +
            "; do mount -t tmpfs -o size=64m,mode=1777 tmpfs \"$d\" || exit 97; done; "
            "cd /tmp; exec unshare -U -m --map-user=65534 --map-group=65534 \"$0\" -I -c \"$1\"")


@lru_cache(maxsize=1)
def _unshare() -> str | None:
    return shutil.which("unshare") if IS_LINUX else None


def run_tests(code: str, tests: list[str], timeout: float = 5.0, mem_mb: int = 1024) -> dict:
    """Run `code`, then each assert in `tests`, in the hardened sandbox.

    Returns rewards.run_tests' dict: {passed, total, compiled, ran, error, timed_out,
    network_isolated} (+ `fs_private`). Raises RuntimeError if the sandbox cannot be built.
    """
    exe = _unshare()
    if not exe:
        raise RuntimeError("code execution needs Linux with util-linux `unshare` (run inside WSL)")
    nonce = secrets.token_hex(16)
    kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []} if _drop_privileges() else {}
    cmd = [exe, "-rnm", "sh", "-c", _PRELUDE, _child_python(), _HARNESS]
    stdin = nonce + "\n" + json.dumps({"code": code, "tests": list(tests),
                                       "limits": _limits(timeout, mem_mb, None)})
    proc = subprocess.Popen(cmd, cwd="/", env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", start_new_session=True, **kw)
    timed_out = False
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
    if proc.returncode == 97:
        raise RuntimeError(f"sandbox mount setup failed: {err.strip()[:300]}")
    res = None
    for line in reversed(out.splitlines()):
        if line.startswith(nonce):
            try:
                res = json.loads(line[len(nonce):])
            except json.JSONDecodeError:
                res = None
            break
    if res is None:
        res = {"compiled": None, "ran": False, "passed": 0, "total": len(tests),
               "error": "timeout" if timed_out else f"no result (exit code {proc.returncode})"}
    res.update(timed_out=timed_out, network_isolated=True, fs_private=True)
    return res


def self_test(repo_root: str) -> dict:
    """Probe the sandbox. Every entry must be True before code tasks may run."""
    tag = secrets.token_hex(4)
    probe_file = os.path.join(repo_root, f".sandbox_probe_{tag}")
    home_probe = os.path.join(os.path.expanduser("~"), f".sandbox_probe_{tag}")
    checks = {}
    try:
        checks["linux_and_unshare"] = bool(_unshare())
        if not checks["linux_and_unshare"]:
            return {"ok": False, "checks": checks}
        r = run_tests("def f(x):\n    return x + 1\n", ["assert f(1) == 2", "assert f(2) == 4"])
        checks["pass_and_fail_scored"] = (r["passed"], r["total"]) == (1, 2)
        r = run_tests("import os\nok = not os.path.exists(%r)\n" % repo_root, ["assert ok"])
        checks["repo_not_visible"] = r["passed"] == 1
        r = run_tests("open(%r, 'w').write('x')\n" % probe_file, ["assert True"])
        checks["repo_not_writable"] = not r["ran"] and not os.path.exists(probe_file)
        r = run_tests("import os\nok = not os.path.exists('/mnt/c')\n", ["assert ok"])
        checks["windows_drive_hidden"] = r["passed"] == 1
        r = run_tests("open(%r, 'w').write('x')\n" % home_probe, ["assert True"])
        checks["home_not_writable"] = not r["ran"] and not os.path.exists(home_probe)
        r = run_tests("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
                      "rc = libc.umount2(b'/mnt', 2)\nimport os\n"
                      "ok = rc != 0 and not os.path.exists('/mnt/c')\n", ["assert ok"])
        checks["cannot_unmount_private_view"] = r["passed"] == 1
        r = run_tests("import socket\ns = socket.socket(); s.settimeout(2)\ns.connect(('1.1.1.1', 80))\n",
                      ["assert True"], timeout=10)
        checks["network_blocked"] = not r["ran"]
        r = run_tests("while True:\n    pass\n", ["assert True"], timeout=2)
        checks["timeout_kills"] = r["timed_out"] and not r["ran"]
        r = run_tests("x = bytearray(2 * 1024**3)\n", ["assert True"], mem_mb=256)
        checks["memory_capped"] = not r["ran"]
        checks["runs_as_nobody"] = _drop_privileges()
    except Exception as e:  # any failure to build the sandbox = not ok
        checks["error"] = f"{type(e).__name__}: {e}"
    finally:
        for f, key in ((probe_file, "repo_not_writable"), (home_probe, "home_not_writable")):
            if os.path.exists(f):
                os.remove(f)
                checks[key] = False
    ok = all(v is True for k, v in checks.items() if k != "error") and "error" not in checks
    return {"ok": ok, "checks": checks, "child_python": _child_python() if IS_LINUX else None,
            "platform": sys.platform}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(self_test(root), indent=2))
