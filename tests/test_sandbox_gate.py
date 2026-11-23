"""SQ346 / SQ352: sandbox self-test has the HOME probes; run_tests fails closed at the execution point.

    wsl, repo root: python -m unittest tests.test_sandbox_gate -v      (stdlib only; pytest also works)
The probe tests need Linux + util-linux `unshare` (skipped otherwise). The gate tests are stubs: they assert that no
child process is started.
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evals import sandbox  # noqa: E402

CAN_RUN = sys.platform.startswith("linux") and bool(sandbox._unshare())


def _no_child(*a, **k):
    raise AssertionError("a child process was started although the self-test failed")


class _Gate(unittest.TestCase):
    def setUp(self):
        self._saved = sandbox._GATE
        sandbox._GATE = None

    def tearDown(self):
        sandbox._GATE = self._saved


@unittest.skipUnless(CAN_RUN, "needs Linux with unshare")
class TestHomeProbes(unittest.TestCase):
    def test_self_test_includes_home_probes_and_passes(self):
        rep = sandbox.self_test(ROOT)
        self.assertTrue(rep["ok"], rep)
        for k in ("home_listdir_raises", "home_venvs_invisible", "negative_control_fails",
                  "positive_control_proc_visible"):
            self.assertIs(rep["checks"].get(k), True, (k, rep["checks"]))

    def test_probes_discriminate(self):
        home = os.path.expanduser("~")
        r = sandbox._run_tests_unchecked("x = 1\n", ["import os\nassert os.path.exists('/proc')"])
        self.assertTrue(r["ran"] and r["passed"] == 1)               # positive control
        r = sandbox._run_tests_unchecked("x = 1\n", ["assert False"])
        self.assertTrue(r["ran"] and r["passed"] == 0)               # negative control
        r = sandbox._run_tests_unchecked("x = 1\n", ["import os\nos.listdir(%r)" % home])
        self.assertTrue(r["ran"] and r["passed"] == 0, "listdir(HOME) succeeded inside the sandbox")


class TestFailClosed(_Gate):
    def test_failing_self_test_blocks_every_child(self):
        with mock.patch.object(sandbox, "self_test", lambda root: {"ok": False, "checks": {"home_listdir_raises": False}}), \
                mock.patch.object(sandbox.subprocess, "Popen", _no_child):
            for _ in range(2):  # stays closed on every later call, not only the first
                with self.assertRaisesRegex(RuntimeError, "self-test failed"):
                    sandbox.run_tests("x = 1\n", ["assert True"])

    def test_self_test_runs_once_per_process(self):
        calls = []
        with mock.patch.object(sandbox, "self_test", lambda root: calls.append(1) or {"ok": True, "checks": {}}), \
                mock.patch.object(sandbox, "_run_tests_unchecked", lambda *a, **k: {"passed": 1, "total": 1}):
            self.assertEqual(sandbox.run_tests("x = 1\n", ["assert True"])["passed"], 1)
            self.assertEqual(sandbox.run_tests("x = 1\n", ["assert True"])["passed"], 1)
        self.assertEqual(len(calls), 1)

    def test_exec_metrics_path_is_gated(self):
        """gogeta_utils._exec_metrics (what lm-eval's exec tasks call) uses sandbox.run_tests: a failed self-test
        stops it with no run_suite.py in the loop."""
        try:
            from evals.tasks import gogeta_utils as gu
        except Exception as e:  # lm-eval / datasets not installed in this venv
            self.skipTest(f"cannot import gogeta_utils: {type(e).__name__}: {e}")
        doc = {"_tests": ["assert True"], "_setup": "", "_prompt": ""}
        with mock.patch.object(sandbox, "self_test", lambda root: {"ok": False, "checks": {"x": False}}), \
                mock.patch.object(sandbox.subprocess, "Popen", _no_child):
            with self.assertRaisesRegex(RuntimeError, "self-test failed"):
                gu.exec_native_pass1(doc, [["pass"]])

    def test_lm_eval_taskmanager_path_is_gated(self):
        """Load the gogeta exec tasks directly through lm-eval's TaskManager (no run_suite.py), score one doc with
        the self-test stubbed to fail: no child may start."""
        try:
            from lm_eval import tasks as lt
        except Exception as e:
            self.skipTest(f"lm_eval not importable here: {e}")
        try:
            manager = lt.TaskManager(include_path=os.path.join(ROOT, "evals", "tasks"))
            names = sorted(n for n in manager.all_tasks if n.startswith("gogeta") and ("exec" in n or "pass" in n))
            if not names:
                self.skipTest("no gogeta exec task names registered in this lm-eval version")
            task = list(lt.get_task_dict(names[:1], manager).values())[0]
            doc = next(iter(task.eval_docs))
        except unittest.SkipTest:
            raise
        except Exception as e:  # task data (HF datasets / val docs) may be unavailable offline
            self.skipTest(f"cannot materialise a gogeta exec task here: {type(e).__name__}: {e}")
        with mock.patch.object(sandbox, "self_test", lambda root: {"ok": False, "checks": {"x": False}}), \
                mock.patch.object(sandbox.subprocess, "Popen", _no_child):
            with self.assertRaisesRegex(RuntimeError, "self-test failed"):
                task.process_results(doc, [["pass"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
