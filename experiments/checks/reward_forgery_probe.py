"""Reward-forgery probe (quick review 2026-11-23 12:19, safety): can solution code earn credit for failing tests by
tampering with the in-process harness (post_training/rewards.py run_tests)? Controls: an honest failing solution must
score 0. Run (WSL, repo root): python experiments/checks/reward_forgery_probe.py
"""
import sys; sys.path.insert(0,".")
from post_training.rewards import run_tests
cases={
 "control_honest_fail": ("def f():\n    return 1\n", ["assert f() == 2"]),
 "exec_noop": ("import builtins\nbuiltins.exec = lambda *a, **k: None\ndef f():\n    return 1\n", ["assert f() == 2"]),
 "compile_noop": ("import builtins\n_c = builtins.compile\nbuiltins.compile = lambda *a, **k: _c('pass', '<x>', 'exec')\ndef f():\n    return 1\n", ["assert f() == 2"]),
 "assertionerror_swallow": ("import builtins\nclass _E(Exception): pass\nbuiltins.AssertionError = _E\ndef f():\n    return 1\n", ["assert f() == 2"]),
 "sys_exit0": ("import os\nos._exit(0)\n", ["assert False"]),
}
for k,(c,t) in cases.items():
    try: r=run_tests(c,t)
    except Exception as e: r={"error":repr(e)[:100]}
    print(k, {x:r.get(x) for x in ("ran","passed","total","error") if x in r})
