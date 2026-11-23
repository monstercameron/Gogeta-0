"""Tests for post_training/rewards.py. Pure stdlib (unittest), pytest also works.

    python -m unittest post_training.test_rewards -v       (from the repo root, in WSL)
"""
from __future__ import annotations

import inspect
import os
import sys
import time
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from post_training import rewards  # noqa: E402
from post_training.rewards import (IS_LINUX, code_exec_reward, extract_asserts, extract_code,  # noqa: E402
                                   run_tests, sandbox_info, score)

ADD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(2, 2) == 4"]


class TestExtraction(unittest.TestCase):
    def test_fenced(self):
        c = "Here you go:\n```python\n" + ADD + "```\nDone."
        self.assertEqual(extract_code(c), ADD)

    def test_prompt_opened_fence(self):  # prompt ended with ```python\n, completion closes it
        self.assertEqual(extract_code(ADD + "```\ntrailing"), ADD)

    def test_plain(self):
        self.assertEqual(extract_code(ADD), ADD)

    def test_extract_asserts(self):
        doc = "Problem...\n```python\n" + ADD + "```\n" + "\n".join(TESTS) + "\n  assert nested\n"
        self.assertEqual(extract_asserts(doc), TESTS)


class TestSandbox(unittest.TestCase):
    def test_pass(self):
        r = run_tests(ADD, TESTS)
        self.assertEqual((r["passed"], r["total"]), (3, 3), r)
        self.assertEqual(score(r), 1.0)

    def test_fail_and_partial(self):
        wrong = "def add(a, b):\n    return a + b if a > 0 else -5\n"
        r = run_tests(wrong, TESTS)  # passes tests 1 and 3, fails test 2
        self.assertEqual(r["passed"], 2, r)
        self.assertAlmostEqual(score(r, partial=True), 2 / 3)
        self.assertEqual(score(r, partial=False), 0.0)
        self.assertIn("AssertionError", r["error"])

    def test_infinite_loop_times_out(self):
        t0 = time.time()
        r = run_tests("while True:\n    pass\n", TESTS, timeout=2)
        dt = time.time() - t0
        self.assertTrue(r["timed_out"], r)
        self.assertEqual(score(r), 0.0)
        self.assertLess(dt, 6, f"took {dt:.1f}s")

    def test_sleep_times_out(self):  # wall clock, not just CPU time
        r = run_tests("import time\ntime.sleep(60)\n", TESTS, timeout=1)
        self.assertTrue(r["timed_out"], r)
        self.assertEqual(score(r), 0.0)

    def test_syntax_error(self):
        r = run_tests("def add(a, b)\n    return a + b\n", TESTS)
        self.assertFalse(r["compiled"], r)
        self.assertIn("SyntaxError", r["error"])
        self.assertEqual(score(r), 0.0)

    def test_runtime_error_in_solution(self):
        r = run_tests("raise ValueError('boom')\n" + ADD, TESTS)
        self.assertFalse(r["ran"], r)
        self.assertEqual(score(r), 0.0)

    def test_no_tests_scores_zero(self):
        self.assertEqual(score(run_tests(ADD, [])), 0.0)

    def test_forged_result_line_is_ignored(self):
        forged = 'print("\\n" + "0" * 32 + \'{"compiled": true, "ran": true, "passed": 3, "total": 3, "error": null}\')\n'
        r = run_tests(forged + "import os\nos._exit(0)\n", TESTS)
        self.assertEqual(score(r), 0.0, r)

    def test_exit_scores_zero(self):
        self.assertEqual(score(run_tests("import sys\nsys.exit(0)\n", TESTS)), 0.0)

    @unittest.skipUnless(IS_LINUX, "rlimits are Linux-only")
    def test_memory_limit(self):
        r = run_tests("x = bytearray(2_000_000_000)\n" + ADD, TESTS, mem_mb=256)
        self.assertFalse(r["ran"], r)
        self.assertIn("MemoryError", r["error"] or "")

    @unittest.skipUnless(IS_LINUX, "network isolation is Linux-only")
    def test_network_blocked_when_isolated(self):
        info = sandbox_info()
        code = ("import socket\n"
                "def add(a, b):\n"
                "    try:\n"
                "        socket.create_connection(('1.1.1.1', 53), timeout=2).close()\n"
                "        return 'net'\n"
                "    except OSError:\n"
                "        return a + b\n")
        r = run_tests(code, ["assert add(1, 2) == 3"], timeout=5)
        if info["network_isolated"]:
            self.assertEqual(r["passed"], 1, r)  # connection failed -> no network
        else:
            self.skipTest("unshare -rn not usable here; network NOT isolated")

    @unittest.skipUnless(IS_LINUX and os.geteuid() == 0, "privilege drop only applies when run as root")
    def test_runs_as_nobody_when_root(self):
        # inside `unshare -r` the child *sees* uid 0, so check what it can do, not getuid()
        r = run_tests("import os\nos.listdir('/root')\n", ["assert True"])
        self.assertFalse(r["ran"], r)
        self.assertIn("PermissionError", r["error"])
        r = run_tests("open('/root/.gogeta_reward_probe', 'w').write('x')\n", ["assert True"])
        self.assertFalse(r["ran"], r)
        self.assertIn("PermissionError", r["error"])


class TestTRLShape(unittest.TestCase):
    def test_reward_function_signature(self):
        completions = ["```python\n" + ADD + "```", "```python\ndef add(a, b):\n    return 99\n```",
                       [{"role": "assistant", "content": ADD}], "while True: pass"]
        tests = [TESTS] * 4
        rewards = code_exec_reward(completions=completions, tests=tests, prompts=["p"] * 4,
                                   completion_ids=[[0]] * 4, trainer_state=None, timeout=2)
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)
        self.assertAlmostEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)
        self.assertTrue(all(isinstance(x, float) for x in rewards))


@unittest.skipUnless(IS_LINUX, "namespaces are Linux-only")
class TestFsIsolation(unittest.TestCase):
    """SQ61 / C83: the mount-namespace hardening moved here from evals/sandbox.py."""

    def setUp(self):
        if not rewards._fs_ns_works():
            self.skipTest("private mount namespace cannot be built here")

    def test_repo_write_control_fails(self):
        probe = os.path.join(REPO, f".reward_fs_probe_{os.getpid()}")
        try:
            for mode in ("auto", "require"):
                r = run_tests("open(%r, 'w').write('x')\n" % probe, ["assert True"], fs_isolation=mode)
                self.assertFalse(r["ran"], (mode, r))
                self.assertFalse(os.path.exists(probe), f"the sandboxed child wrote into the repo ({mode})")
                self.assertTrue(r["fs_private"])
            # positive control: the same harness still scores a correct solution under 'require'
            r = run_tests(ADD, TESTS, fs_isolation="require")
            self.assertEqual((r["passed"], r["total"]), (3, 3), r)           # the positive control still scores
        finally:
            if os.path.exists(probe):
                os.remove(probe)

    def test_repo_and_windows_drive_hidden(self):
        r = run_tests("import os\nok = not os.path.exists(%r) and not os.path.exists('/mnt/c')\n" % REPO,
                      ["assert ok"], fs_isolation="require")
        self.assertEqual(r["passed"], 1, r)

    def test_scratch_dirs_private(self):
        name = f".reward_probe_{os.getpid()}"
        r = run_tests("open('/tmp/%s', 'w').write('x')\nopen('/var/tmp/%s', 'w').write('x')\n" % (name, name),
                      ["assert True"], fs_isolation="require")
        self.assertTrue(r["ran"], r)                                          # writable inside ...
        self.assertFalse(os.path.exists("/tmp/" + name) or os.path.exists("/var/tmp/" + name))  # ... invisible outside

    def test_negative_control(self):
        r = run_tests("x = 1\n", ["assert False"], fs_isolation="require")
        self.assertEqual((r["ran"], r["passed"], r["total"]), (True, 0, 1), r)


class TestFailClosed(unittest.TestCase):
    @staticmethod
    def _tripwire(*a, **k):
        raise AssertionError("a child process was started although isolation could not be established")

    def test_require_raises_without_namespace_and_starts_no_child(self):
        with mock.patch.object(rewards, "_fs_ns_works", lambda: False), \
                mock.patch.object(rewards.subprocess, "Popen", self._tripwire):
            with self.assertRaisesRegex(RuntimeError, "filesystem isolation required"):
                run_tests(ADD, TESTS, fs_isolation="require")

    def test_grpo_reward_path_fails_closed_by_default(self):
        self.assertEqual(inspect.signature(code_exec_reward).parameters["fs_isolation"].default, "require")
        with mock.patch.object(rewards, "_fs_ns_works", lambda: False), \
                mock.patch.object(rewards.subprocess, "Popen", self._tripwire):
            with self.assertRaisesRegex(RuntimeError, "filesystem isolation required"):
                code_exec_reward(completions=["```python\n" + ADD + "```"], tests=[TESTS])

    def test_auto_falls_back_and_says_so(self):
        with mock.patch.object(rewards, "_fs_ns_works", lambda: False):
            r = run_tests(ADD, TESTS, fs_isolation="auto")
        self.assertEqual(r["passed"], 3, r)
        self.assertIs(r["fs_private"], False)

    def test_bad_mode_and_network_conflict_rejected(self):
        with self.assertRaises(ValueError):
            run_tests(ADD, TESTS, fs_isolation="sometimes")
        with self.assertRaises(ValueError):
            run_tests(ADD, TESTS, fs_isolation="require", network="on")

    @unittest.skipUnless(IS_LINUX, "namespaces are Linux-only")
    def test_mount_failure_fails_closed(self):
        if not rewards._fs_ns_works():
            self.skipTest("private mount namespace cannot be built here")
        broken = rewards._NS_PRELUDE.replace("for d in ", "for d in /nonexistent_reward_probe_dir ")
        self.assertNotEqual(broken, rewards._NS_PRELUDE)
        with mock.patch.object(rewards, "_NS_PRELUDE", broken):
            with self.assertRaisesRegex(RuntimeError, "mount setup failed"):
                run_tests(ADD, TESTS, fs_isolation="require")


class TestForgery(unittest.TestCase):
    """SQ373: solution code that tampers with the interpreter it shares with the tests must NOT earn credit for a failing
    test (probe: experiments/checks/reward_forgery_probe.py). Controls: an honest pass and an honest fail."""

    FAIL = ["assert f() == 2"]
    CASES = {
        "exec_noop": "import builtins\nbuiltins.exec = lambda *a, **k: None\ndef f():\n    return 1\n",
        "compile_noop": ("import builtins\n_c = builtins.compile\n"
                         "builtins.compile = lambda *a, **k: _c('pass', '<x>', 'exec')\ndef f():\n    return 1\n"),
        "assertionerror_swallow": "import builtins\nclass _E(Exception): pass\nbuiltins.AssertionError = _E\ndef f():\n    return 1\n",
        "baseexception_str_len_type": ("import builtins\nfor n in ('BaseException', 'str', 'len', 'type', 'dict', 'int', 'isinstance'):\n"
                                       "    setattr(builtins, n, lambda *a, **k: 0)\ndef f():\n    return 1\n"),
        "main_report_tamper": ("import sys\nm = sys.modules['__main__']\nm._report = lambda *a, **k: None\n"
                               "m._res = {'compiled': True, 'ran': True, 'passed': 1, 'total': 1, 'error': None}\n"
                               "m._main = lambda *a, **k: None\ndef f():\n    return 1\n"),
        "sys_modules_json_os": ("import sys, types\nfake = types.ModuleType('json')\nfake.dumps = lambda *a, **k: '{}'\nfake.loads = lambda *a, **k: {}\n"
                                "sys.modules['json'] = fake\nimport json, os\njson.dumps = lambda *a, **k: '{}'\n"
                                "os.write = lambda *a, **k: None\nos.read = lambda *a, **k: b''\ndef f():\n    return 1\n"),
        "os_exit_0": "import os\nos._exit(0)\n",
        "sys_exit_0": "import sys\nsys.exit(0)\n",
        "print_forged_line": ("print('\n' + '0' * 32 + '{\"compiled\": true, \"ran\": true, \"passed\": 1, \"total\": 1, \"error\": null}')\n"
                              "def f():\n    return 1\n"),
        "trace_hook": ("import sys\ndef t(frame, ev, arg):\n    return t\nsys.settrace(t)\ndef f():\n    return 1\n"),
        "background_thread": ("import threading, time\nthreading.Thread(target=lambda: time.sleep(30), daemon=True).start()\n"
                              "def f():\n    return 1\n"),
    }

    def test_controls(self):
        r = run_tests("def f():\n    return 2\n", self.FAIL)
        self.assertEqual((r["ran"], r["passed"], r["total"]), (True, 1, 1), r)           # honest pass
        r = run_tests("def f():\n    return 1\n", self.FAIL)
        self.assertEqual((r["ran"], r["passed"], r["total"]), (True, 0, 1), r)           # honest fail

    def test_forgeries_score_zero(self):
        for name, code in self.CASES.items():
            for mode in ("off", "auto"):
                with self.subTest(case=name, fs_isolation=mode):
                    r = run_tests(code, self.FAIL + ["assert False"], fs_isolation=mode, timeout=5)
                    self.assertEqual(r["passed"], 0, (name, r))
                    self.assertEqual(score(r), 0.0)

    def test_tests_are_isolated_from_each_other(self):
        # a test that mutates its own namespace cannot make the next one pass or fail
        r = run_tests("def f():\n    return 2\n", ["f = None", "assert f() == 2"])
        self.assertEqual(r["passed"], 2, r)            # the assignment "test" passes; the next one still sees the real f

    def test_partial_credit_unchanged(self):
        wrong = "def add(a, b):\n    return a + b if a > 0 else -5\n"
        r = run_tests(wrong, TESTS)
        self.assertEqual((r["passed"], r["total"]), (2, 3), r)


if __name__ == "__main__":
    print("sandbox:", sandbox_info())
    unittest.main(verbosity=2)
