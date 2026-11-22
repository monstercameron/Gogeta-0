"""Tests for post_training/rewards.py. Pure stdlib (unittest), pytest also works.

    python -m unittest post_training.test_rewards -v       (from the repo root, in WSL)
"""
from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


if __name__ == "__main__":
    print("sandbox:", sandbox_info())
    unittest.main(verbosity=2)
