"""Code-execution rewards for RL with verifiable rewards (GRPO-style).

A completion is scored by running the generated solution plus the problem's `assert`
lines in a separate Python process, with a wall-clock timeout and (on Linux) resource
limits. Score = fraction of asserts that pass (partial credit) or 1.0/0.0 (all-or-nothing).

Usable directly as a TRL reward function: `code_exec_reward(completions, tests, **kwargs)`
returns one float per completion. TRL passes every dataset column except `prompt` to reward
functions as keyword arguments, so the dataset needs a `tests` column (list of assert strings).

What IS isolated / limited on Linux (each point exercised by post_training/test_rewards.py):
  * separate process in its own session; the whole process group is SIGKILLed on timeout
  * wall-clock timeout (`timeout` s) plus a CPU-time limit (RLIMIT_CPU)
  * address-space limit (RLIMIT_AS = `mem_mb`), max written file size (RLIMIT_FSIZE, 16 MB),
    no core dumps, max 64 open files. The harness sets these on itself with soft == hard
    before running the solution, so the solution cannot raise them again.
  * if we are root (the WSL default user here is root!), the child runs as `nobody`
    (uid/gid 65534, no supplementary groups), so it cannot write to /root, the repo, etc.,
    and RLIMIT_NPROC (`nproc`, default 256 for nobody) stops fork bombs.
  * the child uses the BASE interpreter (`/usr/bin/python3.x`, stdlib only, no venv packages)
    in isolated mode (`-I`: no PYTHON* env vars, no user site, cwd not on sys.path), in a fresh
    temp dir that is deleted afterwards, with an environment of only PATH and LANG.
  * network: with `network="off"` (default) and a working `unshare -rn`, the child runs in a
    new user + network namespace (only a down loopback, so no network at all). If unshare is
    missing or blocked (e.g. Ubuntu's AppArmor userns restriction), it falls back to running
    WITH network; `sandbox_info()` and the per-run `network_isolated` field say which.
    `network="require_off"` raises instead of falling back.

What is NOT isolated (read this before running untrusted model output at scale):
  * the filesystem is NOT private: the child can read anything world-readable and write to
    world-writable places (/tmp, /dev/shm). If we are NOT root it runs as your own user and
    can read/write all your files. For real RL runs use a container / bubblewrap / nsjail /
    a throwaway VM.
  * processes of the same uid can be signalled (as nobody: other reward children only).
  * reward hacking: the solution runs in the same interpreter as the test harness, so it can
    e.g. monkeypatch builtins, or try to forge the result line. The result line carries a
    random nonce sent on stdin (and deleted from the harness globals before the solution
    runs), which defeats accidental collisions and naive forgery, but a determined policy
    could still find ways around an in-process harness. Watch for suspicious reward jumps.
  * on non-Linux hosts there are no resource limits, no privilege drop and no network
    isolation; only the timeout applies.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

IS_LINUX = sys.platform.startswith("linux")
NOBODY = 65534

# Runs inside the child: read nonce + payload from stdin, apply rlimits, exec the solution,
# then every assert separately, and print one result line prefixed with the nonce.
_HARNESS = r'''
import sys, json
_nonce = sys.stdin.readline().strip()
_payload = json.loads(sys.stdin.read())
_lim = _payload["limits"]
if _lim is not None:
    import resource
    for _name, _v in _lim.items():
        resource.setrlimit(getattr(resource, _name), (_v, _v))
_out = sys.stdout
_res = {"compiled": False, "ran": False, "passed": 0, "total": len(_payload["tests"]), "error": None}
def _report(_nonce=_nonce, _out=_out, _res=_res, _dumps=json.dumps):
    _out.write("\n" + _nonce + _dumps(_res) + "\n")
    _out.flush()
del _nonce
_ns = {"__name__": "__main__"}
try:
    _code = compile(_payload["code"], "<solution>", "exec")
    _res["compiled"] = True
    exec(_code, _ns)
    _res["ran"] = True
except BaseException as _e:
    _res["error"] = type(_e).__name__ + ": " + str(_e)[:200]
if _res["ran"]:
    for _t in _payload["tests"]:
        try:
            exec(compile(_t, "<test>", "exec"), _ns)
            _res["passed"] += 1
        except BaseException as _e:
            if _res["error"] is None:
                _res["error"] = type(_e).__name__ + ": " + str(_e)[:200]
_report()
'''

_FENCE = re.compile(r"```[ \t]*(?:python3|python|py)?[ \t]*\n(.*?)```", re.DOTALL)


# ----------------------------------------------------------------------------- parsing
def extract_code(completion: str) -> str:
    """Pull the solution out of a completion.

    1. the first fenced block ```python ... ``` if there is one;
    2. else, if the prompt already opened the fence (the completion has only a closing ```),
       the text before that closing fence;
    3. else the whole completion.
    """
    m = _FENCE.search(completion)
    if m:
        return m.group(1)
    if "```" in completion:
        return completion.split("```", 1)[0]
    return completion


def extract_asserts(doc: str) -> list[str]:
    """Top-level `assert ...` lines of a document (e.g. an OpenCoder doc with tests)."""
    return [ln.rstrip() for ln in doc.splitlines() if ln.startswith("assert ")]


def _completion_text(c) -> str:
    """TRL passes a str (standard format) or a list of messages (conversational format)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


# ----------------------------------------------------------------------------- sandbox
def _drop_privileges() -> bool:
    return IS_LINUX and os.geteuid() == 0


def _child_python() -> str:
    """The base interpreter (stdlib only). Readable/executable by `nobody`, unlike a venv
    under /root."""
    return os.path.realpath(getattr(sys, "_base_executable", None) or sys.executable)


@lru_cache(maxsize=1)
def _unshare_works() -> bool:
    exe = shutil.which("unshare")
    if not (IS_LINUX and exe):
        return False
    kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []} if _drop_privileges() else {}
    try:
        r = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=10, **kw)
        return r.returncode == 0
    except Exception:
        return False


def sandbox_info(network: str = "off") -> dict:
    """What the sandbox actually does on this machine."""
    return {"platform": sys.platform, "rlimits": IS_LINUX,
            "runs_as_nobody": _drop_privileges(),
            "network_isolated": network != "on" and _unshare_works(),
            "child_python": _child_python()}


def _limits(timeout: float, mem_mb: int, nproc: int | None) -> dict | None:
    if not IS_LINUX:
        return None
    cpu = max(1, int(timeout) + 1)
    lim = {"RLIMIT_CPU": cpu, "RLIMIT_AS": mem_mb * 1024 * 1024,
           "RLIMIT_FSIZE": 16 * 1024 * 1024, "RLIMIT_CORE": 0, "RLIMIT_NOFILE": 64}
    if nproc is None and _drop_privileges():
        nproc = 256  # counts all processes of `nobody` (i.e. all concurrent reward children)
    if nproc is not None:
        lim["RLIMIT_NPROC"] = nproc
    return lim


def run_tests(code: str, tests: list[str], timeout: float = 5.0, mem_mb: int = 1024,
              network: str = "off", nproc: int | None = None) -> dict:
    """Run `code`, then each assert in `tests`, in a sandboxed subprocess.

    Returns {passed, total, compiled, ran, error, timed_out, network_isolated}.
    network: "off" (isolate if possible), "require_off" (raise if not possible), "on".
    nproc: RLIMIT_NPROC. It counts ALL processes of the child's uid; when we run as root the
    child is `nobody` and the default is 256. As a normal user the default is no limit
    (set it above `ps -u $USER | wc -l` if you want one); fork bombs are still cut off by the
    wall-clock kill of the process group.
    """
    nonce = secrets.token_hex(16)
    use_ns = network != "on" and _unshare_works()
    if network == "require_off" and not use_ns:
        raise RuntimeError("network isolation requested but `unshare -rn` is not usable here")
    drop = _drop_privileges()
    work = tempfile.mkdtemp(prefix="gogeta-reward-")
    try:
        harness = os.path.join(work, "harness.py")
        with open(harness, "w") as f:
            f.write(_HARNESS)
        os.chmod(harness, 0o444)
        if drop:
            os.chown(work, NOBODY, NOBODY)
        cmd = [_child_python(), "-I", harness]
        if use_ns:
            cmd = [shutil.which("unshare"), "-rn"] + cmd
        kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []} if drop else {}
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        stdin = nonce + "\n" + json.dumps({"code": code, "tests": list(tests),
                                           "limits": _limits(timeout, mem_mb, nproc)})
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                errors="replace", start_new_session=True, **kw)
        timed_out = False
        try:
            out, _err = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()
            out, _err = proc.communicate()
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
        res["timed_out"] = timed_out
        res["network_isolated"] = use_ns
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def score(res: dict, partial: bool = True) -> float:
    """Fraction of asserts passed (partial=True) or 1.0 only if all passed."""
    if res["total"] == 0:
        return 0.0
    frac = res["passed"] / res["total"]
    return frac if partial else float(frac == 1.0)


# ----------------------------------------------------------------------------- TRL reward
def code_exec_reward(completions, tests, partial: bool = True, timeout: float = 5.0,
                     mem_mb: int = 1024, max_workers: int = 8, **kwargs) -> list[float]:
    """TRL-compatible reward function: one float per completion.

    `tests[i]` is the list of assert strings for completion i (TRL repeats each dataset row's
    columns once per generation). To change options, wrap it and give the wrapper a
    `__name__` (TRL logs rewards under the function's name), e.g.

        def pass_all(completions, tests, **kw):
            return code_exec_reward(completions, tests, partial=False, **kw)
    """
    codes = [extract_code(_completion_text(c)) for c in completions]

    def one(i):
        return score(run_tests(codes[i], tests[i], timeout=timeout, mem_mb=mem_mb), partial)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, range(len(codes))))
