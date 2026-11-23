"""C106 follow-up (claim audit 2026-11-23-1037 b3): does evals/sandbox.py hide $HOME, not just block writes to it?
Probes run inside run_tests, plus a negative control (assert False must fail) and a positive control (/proc visible).
Run (WSL, repo root): python experiments/checks/sandbox_home_probe.py > experiments/checks/results/sandbox_home_probe_2026-11-23.txt
"""
import os,sys
sys.path.insert(0,".")
from evals.sandbox import run_tests
home=os.path.expanduser("~")
for name,t in [("listdir_home_fails","import os\ntry:\n    os.listdir(%r); ok=False\nexcept OSError:\n    ok=True\nassert ok" % home),
               ("home_venvs_invisible","import os\nassert not os.path.exists(%r)" % os.path.join(home,".venvs")),
               ("control_true","assert True")]:
    r=run_tests("x=1\n",[t]); print(name, {k:r.get(k) for k in ("ran","passed","n_passed","error") if k in r} or r)
r=run_tests("x=1\n",["assert False"]); print("neg_control_assert_false", r.get("ran"), r.get("passed"))
r=run_tests("x=1\n",["import os\nassert os.path.exists('/proc')"]); print("pos_control_proc_visible", r.get("ran"), r.get("passed"))
print("host_home_listable", bool(os.listdir(home)), "host_venvs_exists", os.path.exists(os.path.join(home,".venvs")))
