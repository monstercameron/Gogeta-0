"""Sandbox kill cleanup probe (quick review 2026-11-23 18:34, safety/sandbox reviewer). WSL/Linux only.

Three sandboxed snippets that the harness must kill on its 1 s timeout:
  busy_loop       an infinite loop;
  child           spawns a `sleep 77` child in the same process group;
  setsid_child    spawns a `sleep 77` child that calls setsid (start_new_session) to leave the process group
                  (SQ61's descendant-cleanup case).
Each run_tests() call runs in a thread. If it has not returned DEADLINE s after the timeout, the probe records the
harness as blocked, lists the surviving sandbox processes, SIGKILLs them (which unblocks the harness) and moves on,
so a failing case cannot hang the probe.
Pass: every call returns within the deadline, no sandbox process survives, and the host mount table
(/proc/self/mountinfo: mount point + fs type) is identical before and after (no tmpfs leaks from the child's private
mount namespace).
    python3 experiments/checks/sandbox_kill_cleanup.py [repo_root]   (venv gogeta-eval)
"""
import os
import signal
import subprocess
import sys
import threading
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from evals import sandbox  # noqa: E402

TIMEOUT = 1.0
DEADLINE = 5.0
CASES = {
    "busy_loop": "import time\nwhile True:\n    pass\n",
    "child": "import subprocess\nsubprocess.Popen(['sleep', '77'])\nimport time\ntime.sleep(30)\n",
    "setsid_child": "import subprocess\nsubprocess.Popen(['sleep', '77'], start_new_session=True)\nimport time\ntime.sleep(30)\n",
}


def mounts():
    return sorted(l.split()[4] + " " + l.split(" - ")[1].split()[0] for l in open("/proc/self/mountinfo"))


def strays():
    out = subprocess.run(["ps", "-eo", "pid,user,args"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if "while True" in l or "sleep 77" in l or "unshare -rnm" in l]


def kill(lines):
    for l in lines:
        try:
            os.kill(int(l.split()[0]), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


sandbox._gate()  # run the self-test first so its time is not counted against the cases
before = mounts()
ok = True
for name, code in CASES.items():
    box = {}
    t0 = time.monotonic()
    th = threading.Thread(target=lambda: box.update(r=sandbox.run_tests(code, ["assert True"], timeout=TIMEOUT)), daemon=True)
    th.start()
    th.join(TIMEOUT + DEADLINE)
    blocked = th.is_alive()
    survivors = strays()
    if blocked or survivors:
        kill(survivors)
        th.join(30)
    elapsed = time.monotonic() - t0
    r = box.get("r", {})
    good = not blocked and not survivors and r.get("timed_out") is True
    ok &= good
    print(f"{name:13s} {'PASS' if good else 'FAIL'}  returned_within_deadline={not blocked}  elapsed={elapsed:.1f}s  "
          f"timed_out={r.get('timed_out')}  survivors_at_deadline={survivors or 'none'}")
time.sleep(1)
after = mounts()
same = before == after
ok &= same
print(f"host mount table identical before/after: {same} ({len(before)} -> {len(after)} entries); added: {sorted(set(after) - set(before))}")
left = strays()
kill(left)
ok &= not left
print("stray sandbox processes at the end:", left or "none")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
