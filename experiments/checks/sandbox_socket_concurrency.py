"""Sandbox probes from quick review 2026-11-23 19:59 (safety/sandbox reviewer). WSL/Linux only, CPU, no model.

(1) Abstract Unix socket: the host process listens on an abstract-namespace socket ("\\0gogeta_probe_<tag>"); sandboxed code
    tries to connect. Abstract sockets are scoped to the network namespace, and the child has its own (unshare -n), so the
    connect must fail. A filesystem-path Unix socket in the host's /tmp is tried too (the child's /tmp is a private tmpfs).
(2) Concurrency: 20 sandboxed jobs at once, each starting a setsid child (`sleep 77`) and timing out after 1 s. Every call
    must return within its bounded cleanup, no sandbox process may survive, and the host mount table must be unchanged.
    python3 experiments/checks/sandbox_socket_concurrency.py [repo_root]   (venv gogeta-eval)
"""
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from evals import sandbox  # noqa: E402

N_CONC = 20


def mounts():
    return sorted(l.split()[4] + " " + l.split(" - ")[1].split()[0] for l in open("/proc/self/mountinfo"))


def strays(marker):
    out = subprocess.run(["ps", "-eo", "pid,user,args"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if marker in l and "ps -eo" not in l]


ok = True
sandbox._gate()
tag = secrets.token_hex(4)

# (1) abstract and path Unix sockets
abstract = "\0gogeta_probe_" + tag
path = f"/tmp/gogeta_probe_{tag}.sock"
srv_a = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv_a.bind(abstract); srv_a.listen(1)
srv_p = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv_p.bind(path); os.chmod(path, 0o777); srv_p.listen(1)
# positive control: the same connects succeed from the host side (the probe can detect a reachable socket)
ctl = []
for addr in (abstract, path):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        c.connect(addr); ctl.append(True)
    except OSError:
        ctl.append(False)
    finally:
        c.close()
print(f"positive control (host connects): abstract {ctl[0]}, path {ctl[1]}")
ok &= all(ctl)
probe = ("import socket\n"
         "def tryc(addr):\n"
         "    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
         "    try:\n"
         "        s.connect(addr); return True\n"
         "    except OSError:\n"
         "        return False\n"
         "reached_abstract = tryc(%r)\n"
         "reached_path = tryc(%r)\n" % (abstract, path))
r = sandbox.run_tests(probe, ["assert not reached_abstract", "assert not reached_path"], timeout=5)
srv_a.close(); srv_p.close(); os.remove(path)
sock_ok = r["ran"] and r["passed"] == 2
ok &= sock_ok
print(f"unix sockets: {'PASS' if sock_ok else 'FAIL'}  (abstract and host-/tmp path both unreachable: passed {r['passed']}/2, ran {r['ran']})")

# (2) concurrency
marker = "sleep 77"
code = "import subprocess\nsubprocess.Popen(['sleep', '77'], start_new_session=True)\nimport time\ntime.sleep(30)\n"
before = mounts()
t0 = time.monotonic()


def one(_):
    t = time.monotonic()
    try:
        res = sandbox.run_tests(code, ["assert True"], timeout=1.0)
        return res.get("timed_out"), time.monotonic() - t, None
    except RuntimeError as e:
        return None, time.monotonic() - t, str(e)[:80]


with ThreadPoolExecutor(N_CONC) as ex:
    res = list(ex.map(one, range(N_CONC)))
wall = time.monotonic() - t0
time.sleep(1)
left = strays(marker)
for l in left:
    try:
        os.kill(int(l.split()[0]), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError):
        pass
after = mounts()
n_timed = sum(1 for t, _, e in res if t is True)
errors = [e for _, _, e in res if e]
max_call = max(d for _, d, _ in res)
conc_ok = n_timed == N_CONC and not errors and not left and before == after
ok &= conc_ok
print(f"concurrency x{N_CONC}: {'PASS' if conc_ok else 'FAIL'}  timed_out {n_timed}/{N_CONC}, errors {len(errors)}, "
      f"slowest call {max_call:.1f}s, wall {wall:.1f}s, survivors {left or 'none'}, mounts identical {before == after} "
      f"({len(before)} -> {len(after)})")
if errors:
    print("errors:", errors[:3])
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
