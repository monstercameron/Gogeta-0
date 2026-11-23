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

`self_test()` probes all of this (incl. that $HOME is unlistable and ~/.venvs invisible, with a negative control).
run_suite.py refuses to run code tasks unless every probe passes, and `run_tests()` itself runs the self-test once per
process and refuses to start any child if a probe fails (SQ352).
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

from post_training import rewards
from post_training.rewards import (HIDE, IS_LINUX, NOBODY, _HARNESS, _NS_PRELUDE as _PRELUDE,  # noqa: F401
                                   _child_python, _drop_privileges, _limits, extract_code, score)

# The mount-namespace hardening itself (HIDE, the prelude, exit code 97 = fail closed) now lives in
# post_training/rewards.py (SQ61); this module keeps the strict policy (network + filesystem isolation both REQUIRED)
# and the self-test gate.


@lru_cache(maxsize=1)
def _unshare() -> str | None:
    return shutil.which("unshare") if IS_LINUX else None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATE: dict | None = None  # self_test() report of this process, filled once (SQ352)


def _gate() -> None:
    """Fail closed at the execution point (SQ352): the first run_tests() of a process runs self_test() once and
    caches the report; if any probe failed, every run_tests() raises and no child process is started. This does not
    depend on run_suite.py, so a direct lm-eval invocation (`--include_path evals/tasks`) is covered too."""
    global _GATE
    if _GATE is None:
        _GATE = self_test(REPO_ROOT)
    if not _GATE.get("ok"):
        bad = sorted(k for k, v in _GATE.get("checks", {}).items() if v is not True)
        raise RuntimeError(f"sandbox self-test failed (refusing to execute generated code): {bad}")


def run_tests(code: str, tests: list[str], timeout: float = 5.0, mem_mb: int = 1024) -> dict:
    """Run `code`, then each assert in `tests`, in the hardened sandbox, after the once-per-process self-test gate.

    Returns rewards.run_tests' dict: {passed, total, compiled, ran, error, timed_out,
    network_isolated} (+ `fs_private`). Raises RuntimeError if the sandbox cannot be built or the self-test failed.
    """
    _gate()
    return _run_tests_unchecked(code, tests, timeout, mem_mb)


def _run_tests_unchecked(code: str, tests: list[str], timeout: float = 5.0, mem_mb: int = 1024) -> dict:
    """run_tests without the gate; only self_test() itself may call this (it is the gate's probe)."""
    if not _unshare():
        raise RuntimeError("code execution needs Linux with util-linux `unshare` (run inside WSL)")
    res = rewards.run_tests(code, tests, timeout=timeout, mem_mb=mem_mb, network="require_off", fs_isolation="require")
    assert res["fs_private"] and res["network_isolated"]
    return res


def self_test(repo_root: str) -> dict:
    """Probe the sandbox. Every entry must be True before code tasks may run."""
    global _GATE
    tag = secrets.token_hex(4)
    probe_file = os.path.join(repo_root, f".sandbox_probe_{tag}")
    home_probe = os.path.join(os.path.expanduser("~"), f".sandbox_probe_{tag}")
    checks = {}
    try:
        checks["linux_and_unshare"] = bool(_unshare())
        if not checks["linux_and_unshare"]:
            return {"ok": False, "checks": checks}
        r = _run_tests_unchecked("def f(x):\n    return x + 1\n", ["assert f(1) == 2", "assert f(2) == 4"])
        checks["pass_and_fail_scored"] = (r["passed"], r["total"]) == (1, 2)
        r = _run_tests_unchecked("import os\nok = not os.path.exists(%r)\n" % repo_root, ["assert ok"])
        checks["repo_not_visible"] = r["passed"] == 1
        r = _run_tests_unchecked("open(%r, 'w').write('x')\n" % probe_file, ["assert True"])
        checks["repo_not_writable"] = not r["ran"] and not os.path.exists(probe_file)
        r = _run_tests_unchecked("import os\nok = not os.path.exists('/mnt/c')\n", ["assert ok"])
        checks["windows_drive_hidden"] = r["passed"] == 1
        r = _run_tests_unchecked("open(%r, 'w').write('x')\n" % home_probe, ["assert True"])
        checks["home_not_writable"] = not r["ran"] and not os.path.exists(home_probe)
        home = os.path.expanduser("~")
        listdir_probe = ("import os\ntry:\n    os.listdir(%r)\n    ok = False\nexcept OSError:\n    ok = True\nassert ok" % home)
        r = _run_tests_unchecked("x = 1\n", [listdir_probe])
        checks["home_listdir_raises"] = r["ran"] and r["passed"] == 1
        r = _run_tests_unchecked("x = 1\n", ["import os\nassert not os.path.exists(%r)" % os.path.join(home, ".venvs")])
        checks["home_venvs_invisible"] = r["ran"] and r["passed"] == 1
        # controls for the probes above: a failing assert must score 0 (so "passed == 1" can mean something) and
        # /proc must be visible (the sandbox is not just an empty box)
        r = _run_tests_unchecked("x = 1\n", ["assert False"])
        checks["negative_control_fails"] = r["ran"] and r["passed"] == 0 and r["total"] == 1
        r = _run_tests_unchecked("x = 1\n", ["import os\nassert os.path.exists('/proc')"])
        checks["positive_control_proc_visible"] = r["ran"] and r["passed"] == 1
        var_probe = os.path.join("/var/tmp", f".sandbox_probe_{tag}")
        r = _run_tests_unchecked("open(%r, 'w').write('x')\n" % var_probe, ["assert True"])
        checks["shared_tmp_private"] = not os.path.exists(var_probe)  # a write inside must not reach the host
        r = _run_tests_unchecked("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
                      "rc = libc.umount2(b'/mnt', 2)\nimport os\n"
                      "ok = rc != 0 and not os.path.exists('/mnt/c')\n", ["assert ok"])
        checks["cannot_unmount_private_view"] = r["passed"] == 1
        r = _run_tests_unchecked("import socket\ns = socket.socket(); s.settimeout(2)\ns.connect(('1.1.1.1', 80))\n",
                      ["assert True"], timeout=10)
        checks["network_blocked"] = not r["ran"]
        r = _run_tests_unchecked("while True:\n    pass\n", ["assert True"], timeout=2)
        checks["timeout_kills"] = r["timed_out"] and not r["ran"]
        r = _run_tests_unchecked("x = bytearray(2 * 1024**3)\n", ["assert True"], mem_mb=256)
        checks["memory_capped"] = not r["ran"]
        checks["runs_as_nobody"] = _drop_privileges()
    except Exception as e:  # any failure to build the sandbox = not ok
        checks["error"] = f"{type(e).__name__}: {e}"
    finally:
        for f, key in ((probe_file, "repo_not_writable"), (home_probe, "home_not_writable"),
                       (os.path.join("/var/tmp", f".sandbox_probe_{tag}"), "shared_tmp_private")):
            if os.path.exists(f):
                os.remove(f)
                checks[key] = False
    ok = all(v is True for k, v in checks.items() if k != "error") and "error" not in checks
    rep = {"ok": ok, "checks": checks, "child_python": _child_python() if IS_LINUX else None,
           "platform": sys.platform}
    if os.path.abspath(repo_root) == REPO_ROOT:  # run_suite.py's own call also satisfies the gate for this process
        _GATE = rep
    return rep


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(self_test(root), indent=2))
