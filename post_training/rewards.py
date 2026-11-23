"""Code-execution rewards for RL with verifiable rewards (GRPO-style).

A completion is scored by running the generated solution plus the problem's `assert`
lines in a separate Python process, with a wall-clock timeout and (on Linux) resource
limits. Score = fraction of asserts that pass (partial credit) or 1.0/0.0 (all-or-nothing).

Usable directly as a TRL reward function: `code_exec_reward(completions, tests, **kwargs)`
returns one float per completion. TRL passes every dataset column except `prompt` to reward
functions as keyword arguments, so the dataset needs a `tests` column (list of assert strings).

What IS isolated / limited on Linux (each point exercised by post_training/test_rewards.py):
  * separate process in its own session and its own PID namespace (PID 1 of it); on timeout PID 1 and the process
    group are SIGKILLed, and the kernel then kills every process in the namespace, including setsid descendants
    (SQ61); the wait for output after the kill is bounded and raises if anything survived
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

Filesystem isolation (SQ61, C83; moved here from evals/sandbox.py): `run_tests(..., fs_isolation="auto"|"require"|"off")`.
With "auto" (the default of run_tests) the child runs in a new user + network + MOUNT namespace with tmpfs over /mnt,
/tmp, /dev/shm, /run and /var/tmp, then drops into a nested namespace as nobody, so it can neither see the repo or the
Windows drive (/mnt/c on WSL is world-writable 9p) nor write outside private tmpfs; if the namespace cannot be built it
falls back to the weaker mode above and the result's `fs_private` is False. With "require" (the default of
`code_exec_reward`, i.e. the GRPO reward path) it FAILS CLOSED: RuntimeError, and no generated code is run, when the
namespace cannot be built or a mount fails. See the comment above `HIDE` for the mechanism.

What is NOT isolated (read this before running untrusted model output at scale):
  * without fs isolation the filesystem is NOT private: the child can read anything world-readable and write to
    world-writable places (/tmp, /dev/shm). If we are NOT root it runs as your own user and
    can read/write all your files. With it, the rest of the root filesystem is still readable (world-readable files such
    as /etc/passwd and system Python packages). It is a process sandbox, not a security boundary: for real RL runs
    use a container / bubblewrap / nsjail / a throwaway VM.
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

# Runs inside the child. Two processes (SQ373, reward forgery): the GRADER (this process) reads the nonce + payload from
# stdin, applies rlimits, then forks a WORKER that runs the solution and the tests. Hardening against a solution that
# tampers with the interpreter it shares with the tests:
#   * every function used after the solution has run is bound BEFORE it runs (exec, compile, BaseException, os.read/write,
#     json.dumps, ...), so patching builtins / os / json / sys.modules / __main__ cannot change what the tests do;
#   * each test runs in its own fresh namespace (a shallow copy of the solution's) with a pristine copy of the builtins;
#   * no secret exists in the worker while the solution runs: the stdin nonce is dropped in the worker, and the per-run
#     inner nonce is generated by the grader and sent only AFTER the worker announces that the solution has finished;
#   * the result travels over a pipe to the grader, which checks the inner nonce and structure and is the only process
#     that writes the final nonce-tagged line; a worker that dies (os._exit, sys.exit, kill) yields "no result", and
#     print()-forged lines in the worker's stdout are discarded;
#   * a solution that leaves extra threads running or a trace/profile hook set is scored as failed;
#   * (SQ411) the tests are AST-rewritten before the fork: every non-identity comparison and the truthiness of an assert / `not`
#     goes through a harness-bound check that requires BOTH sides to be exact builtin value types (or containers of them), so
#     a solution-defined __eq__/__lt__/__contains__/__bool__ is never called and a non-builtin result fails the assertion.
# Residual risk (documented, not closed): code running in the worker can still imitate the worker's side of the pipe
# protocol (it can enumerate /proc/self/fd) or tamper with state the tests read through other routes than the rewritten
# comparisons (e.g. a test that calls a solution-defined helper whose own code lies, or compares via a method such as
# `.equals`). Solutions that return non-builtin objects (custom classes, numpy arrays) fail equality assertions by design.
# This is a process sandbox, not a security boundary.
_HARNESS = r"""
import sys, os, json, builtins, threading
def _main():
    _read, _write, _fork, _exit, _urandom = os.read, os.write, os.fork, os._exit, os.urandom
    _waitpid, _pipe, _close = os.waitpid, os.pipe, os.close
    _dumps, _loads = json.dumps, json.loads
    _exec, _compile = builtins.exec, builtins.compile
    _BE, _str, _len, _type, _dict = BaseException, str, len, type, dict
    _settrace, _setprofile, _tcount = sys.settrace, sys.setprofile, threading.active_count
    _stdout = sys.stdout
    _pristine = _dict(builtins.__dict__)
    _outer = sys.stdin.readline().strip()
    _payload = _loads(sys.stdin.read())
    _lim = _payload["limits"]
    if _lim is not None:
        import resource
        for _name, _v in _lim.items():
            resource.setrlimit(getattr(resource, _name), (_v, _v))
    _tests = _payload["tests"]
    # ---- SQ411: tests are AST-rewritten BEFORE the fork (so nothing the solution does can touch the rewriter): every non-identity
    # comparison (==, !=, <, <=, >, >=, in, not in) and the truthiness of an assert / of `not` goes through a harness-bound strict
    # check that first requires BOTH sides to be builtin value types (exact type, no subclasses) or containers of them, so a
    # solution-defined __eq__ / __lt__ / __contains__ / __bool__ is never called; anything else makes the assertion fail.
    import ast, operator, collections, fractions, decimal, datetime
    _ast = ast
    _ATOMS = {int, float, complex, bool, str, bytes, bytearray, type(None), range, fractions.Fraction, decimal.Decimal,
              datetime.date, datetime.datetime, datetime.time, datetime.timedelta}
    _SEQ = {list, tuple, set, frozenset, collections.deque}
    _MAP = {dict, collections.OrderedDict, collections.defaultdict, collections.Counter}
    _bool, _int = bool, int
    _OPS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge,
            lambda a, b: operator.contains(b, a), lambda a, b: not operator.contains(b, a)]
    _CODE = {_ast.Eq: 0, _ast.NotEq: 1, _ast.Lt: 2, _ast.LtE: 3, _ast.Gt: 4, _ast.GtE: 5, _ast.In: 6, _ast.NotIn: 7}

    def _ok(x, d=0):
        t = _type(x)
        if t in _ATOMS:
            return True
        if d > 40:
            return False
        if t in _SEQ:
            for e in x:
                if not _ok(e, d + 1):
                    return False
            return True
        if t in _MAP:
            for k, v in x.items():
                if not (_ok(k, d + 1) and _ok(v, d + 1)):
                    return False
            return True
        return False

    def _sc(code, a, b):
        if _ok(a) and _ok(b):
            return _bool(_OPS[code](a, b))
        return False

    def _st(x):
        if _type(x) is _bool:
            return x
        return _bool(x) if _ok(x) else False

    def _sn(x):
        if _type(x) is _bool:
            return not x
        return (not _bool(x)) if _ok(x) else False

    def _nm(n):
        return _ast.Name(id=n, ctx=_ast.Load())

    class _Rw(_ast.NodeTransformer):
        n = 0

        def visit_Compare(self, node):
            self.generic_visit(node)
            ops = node.ops
            if all(_type(o) in (_ast.Is, _ast.IsNot) for o in ops):
                return node
            operands = [node.left] + list(node.comparators)
            vals, prev = [], operands[0]
            for i, o in enumerate(ops):
                nxt = operands[i + 1]
                if i + 1 < len(operands) - 1:      # a middle operand is evaluated once: bind it with a walrus
                    _Rw.n += 1
                    tmp = "__rw_t%d" % _Rw.n
                    right = _ast.NamedExpr(target=_ast.Name(id=tmp, ctx=_ast.Store()), value=nxt)
                    nxt_ref = _nm(tmp)
                else:
                    right, nxt_ref = nxt, nxt
                if _type(o) in (_ast.Is, _ast.IsNot):
                    vals.append(_ast.Compare(left=prev, ops=[o], comparators=[right]))
                else:
                    vals.append(_ast.Call(func=_nm("__sc__"), args=[_ast.Constant(value=_CODE[_type(o)]), prev, right], keywords=[]))
                prev = nxt_ref
            return vals[0] if _len(vals) == 1 else _ast.BoolOp(op=_ast.And(), values=vals)

        def test(self, node):
            if _type(node) is _ast.BoolOp:
                return _ast.BoolOp(op=node.op, values=[self.test(v) for v in node.values])
            if _type(node) is _ast.UnaryOp and _type(node.op) is _ast.Not:
                o = node.operand
                inner = self.test(o) if (_type(o) is _ast.BoolOp or (_type(o) is _ast.UnaryOp and _type(o.op) is _ast.Not)) else self.visit(o)
                return _ast.Call(func=_nm("__sn__"), args=[inner], keywords=[])
            return _ast.Call(func=_nm("__st__"), args=[self.visit(node)], keywords=[])

        def visit_Assert(self, node):
            node.test = self.test(node.test)
            if node.msg is not None:
                node.msg = self.visit(node.msg)
            return node

    _tcodes = []
    for _t in _payload["tests"]:
        try:
            _tree = _Rw().visit(_ast.parse(_t, "<test>", "exec"))
            _tcodes.append(_compile(_ast.fix_missing_locations(_tree), "<test>", "exec"))
        except _BE as _e:
            _tcodes.append(_type(_e).__name__ + ": " + _str(_e)[:200])
    _r1, _w1 = _pipe()
    _r2, _w2 = _pipe()
    _pid = _fork()
    if _pid == 0:                                   # ---- worker
        _outer = None                               # no secret in the worker while the solution runs
        _close(_r1)
        _close(_w2)
        _res = {"compiled": False, "ran": False, "passed": 0, "total": _len(_tests), "error": None}
        _ns = {"__name__": "__main__"}
        try:
            _code = _compile(_payload["code"], "<solution>", "exec")
            _res["compiled"] = True
            _exec(_code, _ns)
            _res["ran"] = True
        except _BE as _e:
            _res["error"] = _type(_e).__name__ + ": " + _str(_e)[:200]
        _settrace(None)
        _setprofile(None)
        if _res["ran"] and _tcount() > 1:
            _res["ran"] = False
            _res["error"] = "TamperingError: solution left threads running"
        _write(_w1, b"READY\n")
        _inner = b""
        while not _inner.endswith(b"\n"):
            _c = _read(_r2, 64)
            if not _c:
                _exit(0)
            _inner += _c
        if _res["ran"]:
            for _c in _tcodes:
                if _type(_c) is _str:                   # the test did not compile / rewrite: it fails
                    if _res["error"] is None:
                        _res["error"] = _c
                    continue
                _tns = _dict(_ns)
                _tns["__builtins__"] = _dict(_pristine)
                _tns["__sc__"], _tns["__st__"], _tns["__sn__"] = _sc, _st, _sn
                try:
                    _exec(_c, _tns)
                    _res["passed"] += 1
                except _BE as _e:
                    if _res["error"] is None:
                        _res["error"] = _type(_e).__name__ + ": " + _str(_e)[:200]
        _write(_w1, _dumps({"n": _inner.decode().strip(), "res": _res}).encode() + b"\n")
        _exit(0)
    # ---- grader
    _close(_w1)
    _close(_r2)
    _buf = b""
    while b"\n" not in _buf and _len(_buf) < 4096:
        _c = _read(_r1, 4096)
        if not _c:
            break
        _buf += _c
    _final = None
    if _buf.split(b"\n", 1)[0] == b"READY":
        _inner = _urandom(16).hex()
        _write(_w2, (_inner + "\n").encode())
        _rest = _buf.split(b"\n", 1)[1]
        while _len(_rest) < 1000000:
            _c = _read(_r1, 65536)
            if not _c:
                break
            _rest += _c
        _lines = [_l for _l in _rest.split(b"\n") if _l.strip()]
        if _lines:
            try:
                _m = _loads(_lines[-1])
                _r = _m["res"]
                if (_m["n"] == _inner and _type(_r) is _dict and _type(_r["passed"]) is int and _type(_r["total"]) is int
                        and 0 <= _r["passed"] <= _r["total"] == _len(_tests)):
                    _final = _r
            except _BE:
                _final = None
    try:
        _waitpid(_pid, 0)
    except _BE:
        pass
    if _final is not None:
        _stdout.write("\n" + _outer + _dumps(_final) + "\n")
        _stdout.flush()
_main()
"""

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


# Descendant cleanup (SQ61; escape demonstrated by experiments/checks/sandbox_kill_cleanup.py, v0.5.47): every sandbox
# child runs as PID 1 of its own PID namespace (`-p -f`). When PID 1 dies the kernel SIGKILLs every process in that
# namespace, including descendants that called setsid to leave the process group, which a killpg alone misses.
# --kill-child also kills PID 1 if the unshare launcher itself dies. The child cannot signal the launcher: it sits
# outside the child's PID namespace.
_PIDNS = ("-p", "-f", "--kill-child")
_CLEANUP_S = 5.0  # bounded wait for the pipes after the kill; past it, fail loudly instead of hanging


def _kill_sandbox(proc: subprocess.Popen) -> None:
    """SIGKILL a timed-out sandbox: first PID 1 of its PID namespace (the launcher's children), which takes the whole
    namespace down even if the harness itself called setsid, then the process group, then the launcher."""
    kids = []
    try:
        with open(f"/proc/{proc.pid}/task/{proc.pid}/children") as f:
            kids = [int(p) for p in f.read().split()]
    except (OSError, ValueError):
        pass
    for pid in kids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()


def _reap(proc: subprocess.Popen) -> tuple[str, str]:
    """Collect output after _kill_sandbox, but never wait unbounded: a survivor that still holds the pipes raises."""
    try:
        return proc.communicate(timeout=_CLEANUP_S)
    except subprocess.TimeoutExpired:
        for s in (proc.stdout, proc.stderr, proc.stdin):
            try:
                if s:
                    s.close()
            except OSError:
                pass
        raise RuntimeError("sandbox cleanup failed: a process survived the timeout kill and still holds the output "
                           "pipes (fail closed; see experiments/checks/sandbox_kill_cleanup.py)")


@lru_cache(maxsize=1)
def _unshare_works() -> bool:
    exe = shutil.which("unshare")
    if not (IS_LINUX and exe):
        return False
    kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []} if _drop_privileges() else {}
    try:
        r = subprocess.run([exe, "-rn", *_PIDNS, "true"], capture_output=True, timeout=10, **kw)
        return r.returncode == 0
    except Exception:
        return False


def sandbox_info(network: str = "off") -> dict:
    """What the sandbox actually does on this machine."""
    return {"platform": sys.platform, "rlimits": IS_LINUX,
            "runs_as_nobody": _drop_privileges(),
            "network_isolated": network != "on" and _unshare_works(),
            "fs_private_available": _fs_ns_works(),
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


# Filesystem isolation (SQ61, moved here from evals/sandbox.py, where it is the reference for C106). Why: on WSL the
# Windows drive is mounted at /mnt/c as 9p/drvfs with every file mode 777, so a child running as `nobody` CAN WRITE to the
# repo and the whole Windows profile (measured 2026-11-22). The child therefore runs in a new user + network + MOUNT
# namespace (`unshare -rnm`); tmpfs is mounted over HIDE (hides /mnt/c, the WSL interop socket under /run, makes the
# scratch dirs private); then a NESTED user+mount namespace as uid/gid 65534 is entered before exec'ing Python (without
# it the child would be ns-root with CAP_SYS_ADMIN and could `umount /mnt`); after the nested step the execve of a
# non-root uid clears every capability and the inherited mounts are locked. Exit code 97 = a mount failed (fail closed).
HIDE = ("/mnt", "/tmp", "/dev/shm", "/run", "/var/tmp")
_NS_PRELUDE = ("set -e; for d in " + " ".join(HIDE) +
               "; do mount -t tmpfs -o size=64m,mode=1777 tmpfs \"$d\" || exit 97; done; "
               "cd /tmp; exec unshare -U -m --map-user=65534 --map-group=65534 \"$0\" -I -c \"$1\"")
FS_MODES = ("auto", "require", "off")


def _ns_cmd(python: str, harness_src: str) -> list[str]:
    return [shutil.which("unshare"), "-rnm", *_PIDNS, "sh", "-c", _NS_PRELUDE, python, harness_src]


@lru_cache(maxsize=1)
def _fs_ns_works() -> bool:
    """Can the full private-filesystem sandbox be built here? Runs the real prelude once with a trivial program."""
    if not (IS_LINUX and shutil.which("unshare")):
        return False
    kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []} if _drop_privileges() else {}
    try:
        r = subprocess.run(_ns_cmd(_child_python(), "pass"), capture_output=True, timeout=15, cwd="/",
                           env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, **kw)
        return r.returncode == 0
    except Exception:
        return False


def run_tests(code: str, tests: list[str], timeout: float = 5.0, mem_mb: int = 1024,
              network: str = "off", nproc: int | None = None, fs_isolation: str = "auto") -> dict:
    """Run `code`, then each assert in `tests`, in a sandboxed subprocess.

    Returns {passed, total, compiled, ran, error, timed_out, network_isolated, fs_private}.
    network: "off" (isolate if possible), "require_off" (raise if not possible), "on".
    fs_isolation: "auto" (private mount namespace if it can be built, else the old shared-filesystem behaviour, and
    `fs_private` says which), "require" (raise RuntimeError and start NO child if it cannot be built or a mount fails:
    this is what the GRPO reward path uses), "off". The namespace path always has no network (needs network != "on").
    nproc: RLIMIT_NPROC. It counts ALL processes of the child's uid; when we run as root the
    child is `nobody` and the default is 256. As a normal user the default is no limit
    (set it above `ps -u $USER | wc -l` if you want one); fork bombs are still cut off by the
    wall-clock kill of the process group.
    """
    if fs_isolation not in FS_MODES:
        raise ValueError(f"fs_isolation must be one of {FS_MODES}")
    nonce = secrets.token_hex(16)
    if fs_isolation == "require" and network == "on":
        raise ValueError("fs_isolation='require' uses a network namespace; it cannot be combined with network='on'")
    use_fs = fs_isolation != "off" and network != "on" and _fs_ns_works()
    if fs_isolation == "require" and not use_fs:
        raise RuntimeError("filesystem isolation required but a private mount namespace cannot be built here "
                           "(needs Linux with util-linux `unshare` and permission to create user namespaces)")
    if use_fs:
        return _run_in_fs_ns(code, tests, timeout, mem_mb, nproc, nonce)
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
            cmd = [shutil.which("unshare"), "-rn", *_PIDNS] + cmd
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
            _kill_sandbox(proc)
            out, _err = _reap(proc)
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
        res["fs_private"] = False
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_in_fs_ns(code, tests, timeout, mem_mb, nproc, nonce) -> dict:
    """The private-filesystem path of run_tests (harness passed with -c: nothing is written to disk)."""
    kw = {"user": NOBODY, "group": NOBODY, "extra_groups": []} if _drop_privileges() else {}
    stdin = nonce + "\n" + json.dumps({"code": code, "tests": list(tests), "limits": _limits(timeout, mem_mb, nproc)})
    proc = subprocess.Popen(_ns_cmd(_child_python(), _HARNESS), cwd="/", env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            errors="replace", start_new_session=True, **kw)
    timed_out = False
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_sandbox(proc)
        out, err = _reap(proc)
    if proc.returncode == 97:
        raise RuntimeError(f"sandbox mount setup failed (fail closed): {err.strip()[:300]}")
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


def score(res: dict, partial: bool = True) -> float:
    """Fraction of asserts passed (partial=True) or 1.0 only if all passed."""
    if res["total"] == 0:
        return 0.0
    frac = res["passed"] / res["total"]
    return frac if partial else float(frac == 1.0)


# ----------------------------------------------------------------------------- TRL reward
def code_exec_reward(completions, tests, partial: bool = True, timeout: float = 5.0,
                     mem_mb: int = 1024, max_workers: int = 8, fs_isolation: str = "require", **kwargs) -> list[float]:
    """TRL-compatible reward function: one float per completion.

    `tests[i]` is the list of assert strings for completion i (TRL repeats each dataset row's
    columns once per generation). To change options, wrap it and give the wrapper a
    `__name__` (TRL logs rewards under the function's name), e.g.

        def pass_all(completions, tests, **kw):
            return code_exec_reward(completions, tests, partial=False, **kw)
    """
    codes = [extract_code(_completion_text(c)) for c in completions]

    def one(i):
        return score(run_tests(codes[i], tests[i], timeout=timeout, mem_mb=mem_mb, fs_isolation=fs_isolation), partial)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, range(len(codes))))
