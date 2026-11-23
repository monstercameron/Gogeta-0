"""Watchdog tests (stdlib unittest): python3 -m unittest cockpit.test_watchdog -v
Everything runs against fixtures in a temp dir (fake PID files, fake logs, a fake `gogeta-gpu-slot` process).
It never touches the live run directory, never stops a real job, and only starts a static server on test port 8892.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import logdb  # noqa: E402
import watchdog  # noqa: E402

FAKE_HF = "hf_" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2"
HOT = {"available": True, "temp_c": 50.0, "util_pct": 0, "mem_used_mib": 0, "compute_procs": []}


class Fx(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        for sub in ("run/cpu", "logs", "runs", "campaigns", "cockpit"):
            (self.d / sub).mkdir(parents=True)
        self.patches = [
            mock.patch.dict(os.environ, {"GOGETA_COCKPIT_DB": str(self.d / "t.sqlite"), "GOGETA_COCKPIT_DIR": str(self.d / "cockpit")}),
            mock.patch.object(core, "DATA", self.d), mock.patch.object(core, "RUN_DIR", self.d / "run"),
            mock.patch.object(core, "CPU_DIR", self.d / "run/cpu"), mock.patch.object(core, "LOG_DIR", self.d / "logs"),
            mock.patch.object(core, "RUNS_DIR", self.d / "runs"), mock.patch.object(core, "COCKPIT_RUN", self.d / "run/cockpit"),
            mock.patch.object(core, "GPU_LOCK", self.d / "run/gpu.lock"), mock.patch.object(core, "SERVERS", {}),
            mock.patch.object(core, "code_paths_dirty", return_value=[]), mock.patch.object(core, "_git", return_value="vTEST"),
            mock.patch.object(core, "_gpu_hw", return_value=dict(HOT)), mock.patch.object(watchdog, "CAMPAIGN_DIR", self.d / "campaigns"),
        ]
        for p in self.patches:
            p.start()
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.kill()
            p.wait()
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def kinds(self, r):
        return sorted(i["kind"] for i in r["incidents"])

    def train_run(self, name, exit_code=None, log_text="", pid=999999):
        log = self.d / "logs" / (name + ".log")
        log.write_text(log_text)
        base = self.d / "run" / ("train-" + name)
        base.with_suffix(".pid").write_text(str(pid))
        base.with_suffix(".info").write_text("experiment=%s\nseed=0\nconfig=configs/x.json\nlog=%s\nstarted=2026-01-01T00:00:00\ncmd= python -m gogeta.train --config configs/x.json --seed 0\n" % (name.split("-")[0], log))
        if exit_code is not None:
            base.with_suffix(".exit").write_text(str(exit_code))
        return log

    def campaign(self, name, n_items=3, done=1, exit_code="3", log_text=""):
        items = ["E10 --seed 0 --lr %de-3" % (i + 1) for i in range(n_items)]
        snap = self.d / "run" / ("campaign-%s-STAMP.items" % name)
        snap.write_text("\n".join(items) + "\n")
        log = self.d / "logs" / ("campaign-%s-STAMP.log" % name)
        log.write_text(log_text)
        r = self.d / "run" / ("campaign-" + name)
        r.with_suffix(".pid").write_text("999999")
        r.with_suffix(".info").write_text("campaign=%s\nsnapshot=%s\nlog=%s\nstarted=x\ngit=vT\nitems=%d\n" % (name, snap, log, n_items))
        r.with_suffix(".state").write_text("".join("%d\t0\ts\te\t%s\n" % (i + 1, items[i]) for i in range(done)))
        if exit_code is not None:
            r.with_suffix(".exit").write_text(exit_code)
        return items


class DetectorTests(Fx):
    def test_dead_pid_without_exit(self):
        self.train_run("E1-s0-20260101-0000", None, "step 0/600 loss 5.0\n")
        r = watchdog.scan_once()
        self.assertEqual(self.kinds(r), ["run_died"])
        self.assertEqual(r["incidents"][0]["status"], "open")

    def test_nonzero_exit_and_stop_signal(self):
        self.train_run("E1-s0-A", 1, "step 100/600 loss 1.2\nstep 200/600 loss nan\n")
        self.train_run("E1-s0-B", 143, "step 1/600 loss 5\n")
        r = watchdog.scan_once()
        self.assertEqual(self.kinds(r), ["run_failed", "run_stopped"])
        failed = [i for i in r["incidents"] if i["kind"] == "run_failed"][0]
        self.assertEqual(failed["classification"]["category"], "code-bug")
        stopped = [i for i in r["incidents"] if i["kind"] == "run_stopped"][0]
        self.assertEqual(stopped["classification"]["category"], "operator")

    def test_log_nan_oom_traceback_triton(self):
        (self.d / "logs" / "E2-s0-N.log").write_text("step 10/100 loss 2.0\nstep 20/100 loss nan\n")
        (self.d / "logs" / "E3-s0-O.log").write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n")
        (self.d / "logs" / "E4-s0-T.log").write_text('Traceback (most recent call last):\n  File "/x/gogeta/model.py", line 3, in f\n    x()\nAttributeError: boom\n')
        (self.d / "logs" / "E5-s0-X.log").write_text("triton.runtime.errors.OutOfResources: autotune error: cache write failed\n")
        r = watchdog.scan_once()
        k = {i["subject"]: i for i in r["incidents"]}
        self.assertEqual(k["E2-s0-N"]["kind"], "log_nan")
        self.assertEqual(k["E3-s0-O"]["classification"]["category"], "resource")
        self.assertEqual(k["E4-s0-T"]["kind"], "log_traceback")
        self.assertEqual(k["E4-s0-T"]["classification"]["category"], "code-bug")
        self.assertIn("gogeta/model.py", k["E4-s0-T"]["agent_brief"])
        self.assertEqual(k["E5-s0-X"]["classification"]["category"], "transient")

    def test_hung_run_uses_fake_process(self):
        p = subprocess.Popen(["bash", "-c", "sleep 60", "gogeta-gpu-slot"])
        self.procs.append(p)
        log = self.train_run("E1-s0-H", None, "step 0/600 loss 5.0\n", pid=p.pid)
        old = time.time() - 40 * 60
        os.utime(log, (old, old))
        r = watchdog.scan_once({"hung_minutes": 15})
        self.assertEqual(self.kinds(r), ["run_hung"])
        self.assertIsNone(p.poll())  # the healthy process was NOT killed

    def test_healthy_run_not_flagged(self):
        p = subprocess.Popen(["bash", "-c", "sleep 60", "gogeta-gpu-slot"])
        self.procs.append(p)
        self.train_run("E1-s0-OK", None, "step 0/600 loss 5.0\n", pid=p.pid)
        self.assertEqual(watchdog.scan_once()["new"], [])

    def test_cpu_job_failed(self):
        (self.d / "run/cpu/k4-1.exit").write_text("1")
        (self.d / "run/cpu/k4-1.info").write_text("name=k4\ncmd=python x.py\n")
        (self.d / "run/cpu/k4-1.log").write_text("boom\n")
        self.assertEqual(self.kinds(watchdog.scan_once()), ["cpu_job_failed"])

    def test_gpu_hot_and_pause_flag(self):
        with mock.patch.object(core, "_gpu_hw", return_value={**HOT, "temp_c": 90.0}):
            r = watchdog.scan_once()
        self.assertEqual(self.kinds(r), ["gpu_hot"])
        flag = self.d / "cockpit/pause.flag"
        self.assertTrue(flag.exists())
        self.assertEqual(r["incidents"][0]["classification"]["category"], "thermal")
        with mock.patch.object(core, "_gpu_hw", return_value={**HOT, "temp_c": 60.0}):
            watchdog.scan_once()
        self.assertFalse(flag.exists())

    def test_disk_low(self):
        fake = shutil.disk_usage(self.d)._replace(free=1 * 2**30)
        with mock.patch.object(watchdog.shutil, "disk_usage", return_value=fake):
            self.assertEqual(self.kinds(watchdog.scan_once()), ["disk_low"])

    def test_server_down_restart_on_alt_port(self):
        port = 8892
        with mock.patch.object(core, "SERVERS", {"site": {"port": port, "desc": "test"}}):
            try:
                core.start_server("site", port)
                self.assertTrue(core.port_up(port))
                self.assertEqual(watchdog.scan_once()["new"], [])  # seen up
                core.stop_server("site", port)
                self.assertFalse(core.port_up(port))
                r = watchdog.scan_once()
                self.assertEqual(self.kinds(r), ["server_down"])
                self.assertTrue(any("restart attempt 1/3" in a for a in r["incidents"][0]["actions"]), r["incidents"][0]["actions"])
                self.assertTrue(core.port_up(port))
                watchdog.scan_once()  # sees it up again -> auto-closes
                self.assertEqual(watchdog.incident(r["new"][0])["status"], "fixed")
            finally:
                core.stop_server("site", port)
            self.assertFalse(core.port_up(port))

    def test_server_never_seen_up_is_not_started(self):
        with mock.patch.object(core, "SERVERS", {"site": {"port": 8893, "desc": "t"}}), mock.patch.object(core, "start_server") as st:
            self.assertEqual(watchdog.scan_once()["new"], [])
            st.assert_not_called()


class CampaignTests(Fx):
    def test_aborted_dirty_tree_writes_exact_resume_file_no_autolaunch(self):
        items = self.campaign("k7", 3, 1, "3", "[t] [2/3] preflight REFUSED and the GPU is free. Failed checks:\n  [FAIL] code paths have uncommitted changes: tools/x.py (dirty tree)\n")
        with mock.patch.object(watchdog, "resume_campaign") as rc:
            r = watchdog.scan_once()
        self.assertEqual(self.kinds(r), ["campaign_aborted"])
        inc = r["incidents"][0]
        self.assertEqual(inc["classification"]["category"], "dirty-tree")
        rc.assert_not_called()
        f = self.d / "campaigns/k7-resume-1.campaign"
        body = [l for l in f.read_text().splitlines() if not l.startswith("#")]
        self.assertEqual(body, items[1:])  # exactly the unfinished items
        self.assertIn("k7-resume-1.campaign", inc["resume_file"])
        self.assertTrue(any("auto-resume NOT done" in a for a in inc["actions"]))
        self.assertEqual(watchdog.scan_once()["new"], [])  # no duplicate incident or second resume file
        self.assertFalse((self.d / "campaigns/k7-resume-2.campaign").exists())

    def test_transient_cause_auto_resumes_once(self):
        self.campaign("k8", 2, 0, "5", "[t] [1/2] FAILED with exit 1: E1 --seed 0\n  triton autotune cache error: stale file handle\n")
        with mock.patch.object(watchdog, "resume_campaign", return_value={"launched": True}) as rc:
            r = watchdog.scan_once()
        self.assertEqual(r["incidents"][0]["classification"]["category"], "transient")
        rc.assert_called_once()
        self.assertTrue(any("auto-resume (transient): launched" in a for a in r["incidents"][0]["actions"]))

    def test_exit_zero_campaign_is_not_an_incident(self):
        self.campaign("ok", 3, 1, "0", "[t] campaign finished: 3 item(s), 0 failed\n")  # exit 0 with --resume-skipped items
        self.assertEqual(watchdog.scan_once()["new"], [])

    def test_resume_refused_when_dirty_or_unconfirmed(self):
        f = self.d / "campaigns/x-resume-1.campaign"
        f.write_text("E1 --seed 0\n")
        self.assertIn("confirm", watchdog.resume_campaign(str(f), False)["refused"])
        with mock.patch.object(core, "code_paths_dirty", return_value=["?? tools/x.py"]), mock.patch.object(core, "_sh") as sh:
            self.assertIn("dirty", watchdog.resume_campaign(str(f), True)["refused"])
            sh.assert_not_called()
        with mock.patch.object(core, "_gpu_slot", return_value={"held": True}), mock.patch.object(core, "_sh") as sh:
            self.assertIn("slot", watchdog.resume_campaign(str(f), True)["refused"])
            sh.assert_not_called()
        self.assertIn("only", watchdog.resume_campaign(str(self.d / "x.campaign"), True)["refused"])

    def test_sigterm_stopped_campaign_is_operator_never_auto_resumed(self):
        # 2026-11-22: the human ordered a GPU kill (overheat); the runner exited 143 and the watchdog auto-resumed it as "transient"
        self.campaign("p2b", 3, 1, "143", "[t] [2/3] E11 --seed 0\n  Terminated\n  connection reset by peer\n")
        with mock.patch.object(watchdog, "resume_campaign") as rc:
            r = watchdog.scan_once()
        self.assertEqual(r["incidents"][0]["classification"]["category"], "operator")
        rc.assert_not_called()

    def test_hold_flag_blocks_manual_and_auto_resume(self):
        f = self.d / "campaigns/h-resume-1.campaign"
        f.write_text("E1 --seed 0\n")
        hold = watchdog.cdir() / "hold.flag"
        hold.parent.mkdir(parents=True, exist_ok=True)
        hold.write_text("HOLD by the human\n")
        with mock.patch.object(core, "_sh") as sh:
            self.assertIn("operator hold", watchdog.resume_campaign(str(f), True)["refused"])
            self.assertIn("operator hold", watchdog.resume_campaign(str(f), _auto=True)["refused"])
            sh.assert_not_called()
        hold.unlink()
        with mock.patch.object(core, "_gpu_slot", return_value={"held": True}):
            self.assertNotIn("operator hold", watchdog.resume_campaign(str(f), True)["refused"])


class IncidentTests(Fx):
    def make(self):
        self.train_run("E1-s0-LC", 1, "step 1/6 loss 5\nValueError: x\n")
        return watchdog.scan_once()["new"][0]

    def test_lifecycle(self):
        iid = self.make()
        u = watchdog.update_incident
        self.assertEqual(u(iid, "triaged", "looked")["status"], "triaged")
        self.assertIn("error", u(iid, "bogus"))
        self.assertIn("error", u(iid, "fixed"))  # note required
        self.assertEqual(u(iid, "fixing")["status"], "fixing")
        self.assertEqual(u(iid, "fixed", "patched in v0.9.9")["status"], "fixed")
        self.assertIn("error", u(iid, "triaged"))  # no going back
        self.assertEqual(u(iid, "verified", "resume passed")["status"], "verified")
        self.assertIn("error", u(iid, "wontfix"))
        self.assertEqual(watchdog.incidents(open_only=True)["open_total"], 0)
        self.assertEqual(len(watchdog.incident(iid)["notes"]), 3)

    def test_wontfix_and_filters(self):
        iid = self.make()
        self.assertEqual(watchdog.update_incident(iid, "wontfix", "n/a")["status"], "wontfix")
        self.assertEqual(watchdog.incidents(kind="run_failed")["count"], 1)

    def test_storage_files_and_log_line(self):
        iid = self.make()
        self.assertTrue((self.d / "cockpit/incidents" / (iid + ".json")).exists())
        log = (self.d / "cockpit/incidents.log").read_text()
        self.assertIn(iid, log)
        self.assertIn("kind=run_failed", log)
        watchdog.update_incident(iid, "triaged", "x")
        self.assertIn("status=triaged", (self.d / "cockpit/incidents.log").read_text())

    def test_redaction_everywhere(self):
        self.train_run("E1-s0-RD", 1, "step 1/6 loss 5\nauth token=SECRET99 hf_%s\nValueError: bad hf_%s\n" % (FAKE_HF, FAKE_HF))
        iid = watchdog.scan_once()["new"][0]
        blob = json.dumps(watchdog.incident(iid)) + (self.d / "cockpit/incidents" / (iid + ".json")).read_text() + (self.d / "cockpit/incidents.log").read_text()
        con = logdb.connect()
        blob += json.dumps([list(r) for r in con.execute("SELECT * FROM incidents")]) + json.dumps([list(r) for r in con.execute("SELECT * FROM events")])
        self.assertNotIn(FAKE_HF, blob)
        self.assertNotIn("SECRET99", blob)

    def test_agent_brief_has_required_parts(self):
        b = watchdog.incident(self.make())["agent_brief"]
        for s in ("WHAT FAILED", "EVIDENCE", "SUSPECTED CAUSE", "FILES INVOLVED", "REPRODUCE", "FIXED MEANS"):
            self.assertIn(s, b)

    def test_detection_is_idempotent(self):
        self.train_run("E1-s0-ID", 1, "ValueError: x\n")
        self.assertEqual(len(watchdog.scan_once()["new"]), 1)
        self.assertEqual(watchdog.scan_once()["new"], [])


class DaemonTests(Fx):
    def test_single_instance_lock(self):
        a = watchdog.acquire_lock()
        self.assertIsNotNone(a)
        self.assertIsNone(watchdog.acquire_lock())  # second instance refused while the first holds it
        a.close()
        b = watchdog.acquire_lock()
        self.assertIsNotNone(b)
        b.close()

    def test_status_not_running(self):
        s = watchdog.status()
        self.assertFalse(s["running"])
        self.assertFalse(s["healthy"])


if __name__ == "__main__":
    unittest.main()
