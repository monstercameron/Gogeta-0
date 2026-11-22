"""Cockpit tests (stdlib unittest; run in WSL from the repo root):
    python3 -m unittest cockpit.test_core -v
Reads the live repo read-only. Uses a temporary SQLite file (GOGETA_COCKPIT_DB), never the real one.
Nothing here starts, stops or launches anything real.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import logdb  # noqa: E402

FAKE_HF = "hf_" + "A1b2C3d4E5f6G7h8I9j0KlMnOp"
TB_LOG = """step 100 loss 1.2
Traceback (most recent call last):
  File "/x/train.py", line 10, in main
    loss = model(x)
  File "/x/model.py", line 5, in forward
    raise ValueError("boom hf_%s")
ValueError: boom hf_%s
step 101 loss nan
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB
launch_experiment: preflight REFUSED: GPU slot held
""" % (FAKE_HF, FAKE_HF)


class TmpDB(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"GOGETA_COCKPIT_DB": os.path.join(self.d, "t.sqlite")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.d, ignore_errors=True)


class RedactionTests(unittest.TestCase):
    def test_tokens_removed(self):
        txt = "x hf_%s y token=abc123SECRET z write_token: 'qq11ww22' Authorization: Bearer abcdefgh12345678 ?__sign=deadbeef" % FAKE_HF
        out = core.redact(txt)
        for s in (FAKE_HF, "abc123SECRET", "qq11ww22", "abcdefgh12345678", "deadbeef"):
            self.assertNotIn(s, out)
        self.assertIn("REDACTED", out)

    def test_plain_text_untouched(self):
        self.assertEqual(core.redact("val_loss 1.0286 at step 611"), "val_loss 1.0286 at step 611")


class ParsingTests(unittest.TestCase):
    def test_final_truncated(self):
        d = core.parse_final('done: E1 ...; FINAL {"val_loss": 1.0367, "val_bits_per_char": 0.402, "tokens": 20021248.0, "tokens_per_s": 1263')
        self.assertEqual(d["val_loss"], 1.0367)
        self.assertEqual(d["val_bits_per_char"], 0.402)

    def test_table_rows_fixture(self):
        p = Path(tempfile.mkdtemp()) / "c.md"
        p.write_text("| ID | Claim | Source | Status | Evidence |\n|---|---|---|---|---|\n| C1 | a claim | src | `VERIFIED` | ev |\n| C2 | b | s | `UNVERIFIED` (x) | e |\n")
        rows = core._table_rows(p, r"C\d+")
        self.assertEqual([r[1][0] for r in rows], ["C1", "C2"])
        self.assertEqual(core._status_of(rows[1][1]), "UNVERIFIED")

    def test_queue_semantics_match_queue_py(self):
        q = core._qload()
        mine = core.queue_next("gpu")
        import subprocess
        out = subprocess.run([sys.executable, "-B", str(core.REPO / "tools/queue.py"), "next", "--gpu"], capture_output=True, text=True,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).stdout
        if mine["item"]:
            self.assertTrue(out.startswith(mine["item"]["id"]))
        else:
            self.assertIn("nothing pickable", out)
        mine = core.queue_next("cpu")
        out = subprocess.run([sys.executable, "-B", str(core.REPO / "tools/queue.py"), "next", "--cpu"], capture_output=True, text=True,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).stdout
        self.assertTrue(out.startswith(mine["item"]["id"]) if mine["item"] else "nothing pickable" in out)
        self.assertGreater(len(q["items"]), 10)


class LiveRepoTests(unittest.TestCase):
    def test_queue(self):
        r = core.queue(include_done=True, limit=500)
        self.assertGreater(r["count"], 10)
        self.assertEqual(core.queue("id:K7", include_done=True)["rows"][0]["id"], "K7")

    def test_claims_side_quests(self):
        c = core.claims()
        self.assertGreater(c["total"], 50)
        self.assertTrue(all(r["id"] == "C2" for r in core.claims(id="C2")["rows"]))
        self.assertGreater(core.side_quests()["count"], 10)

    def test_devlog_versions(self):
        d = core.devlog(latest=2)
        self.assertEqual(len(d["entries"]), 2)
        self.assertTrue(core.devlog(version=d["entries"][0]["version"])["text"].startswith("## v"))
        self.assertGreater(core.versions(5)["count"], 0)

    def test_results_and_experiment(self):
        names = core.list_results()["results"]
        self.assertGreater(len(names), 3)
        self.assertNotIn("error", core.results(names[0]["name"]))
        self.assertNotIn("error", core.experiment("E10"))
        self.assertIn("error", core.experiment("NOPE999"))

    def test_search_index_runs(self):
        self.assertGreater(core.search("Gated DeltaNet")["count"], 0)
        self.assertGreater(core.index()["count"], 5)
        self.assertGreaterEqual(core.runs()["count"], 1)

    def test_brief_cap_and_status(self):
        b = core.brief()
        self.assertLessEqual(b["bytes"], 4200)
        self.assertIn("QUEUE", b["text"])
        s = core.status()
        self.assertIn("slot", s["gpu"])
        self.assertEqual(s["cpu"]["max"], 8)

    def test_log_tail_redacts(self):
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "fake.log"
            lf.write_text("line1\napi token=SECRETVALUE99 hf_%s\n" % FAKE_HF)
            with mock.patch.object(core, "LOG_DIR", Path(td)):
                r = core.log_tail("fake", 5)
            self.assertNotIn(FAKE_HF, r["text"])
            self.assertNotIn("SECRETVALUE99", r["text"])

    def test_notes_path_traversal(self):
        self.assertIn("error", core.notes("../AGENTS.md"))


class LaunchGuardTests(unittest.TestCase):
    """The launch tool must refuse without ever executing anything."""

    def q(self, **over):
        it = {"id": "T1", "kind": "train", "title": "t", "why": "", "depends_on": [], "ready": True, "needs_human": False,
              "status": "queued", "resources": {"type": "gpu", "est_minutes": 1}, "launch": "bash tools/launch_experiment.sh E10 --seed 0"}
        it.update(over)
        return {"items": [it]}

    def run_launch(self, **over):
        with mock.patch.object(core, "_qload", return_value=self.q(**over)), mock.patch.object(core, "_sh") as sh:
            r = core.launch_queue_item.__wrapped__("T1", True) if hasattr(core.launch_queue_item, "__wrapped__") else core.launch_queue_item("T1", True)
            return r, sh

    def test_needs_confirm(self):
        with mock.patch.object(core, "_sh") as sh:
            r = core.launch_queue_item("K7", False)
        self.assertFalse(r["launched"])
        sh.assert_not_called()

    def test_refuses_when_slot_held(self):
        held = {"held": True, "experiment": "E1", "seed": "0", "pid": 1, "stale_lock": False}
        with mock.patch.object(core, "_gpu_slot", return_value=held), mock.patch.object(core, "_qload", return_value=self.q()), \
                mock.patch.object(core, "_sh") as sh, mock.patch.object(core, "_lg", return_value=None):
            r = core.launch_queue_item("T1", True)
        self.assertIn("slot is held", r["refused"])
        sh.assert_not_called()

    def test_refuses_bad_commands(self):
        for cmd in ("python -m gogeta.train --config x", "bash tools/launch_experiment.sh E1 --skip-preflight",
                    "bash tools/launch_experiment.sh E1; rm -rf x", "bash tools/launch_experiment.sh <placeholder>",
                    "bash tools/launch_experiment.sh E1 --foreground", "bash experiments/launch/status.sh && echo hi", ""):
            with mock.patch.object(core, "_qload", return_value=self.q(launch=cmd)), mock.patch.object(core, "_sh") as sh, \
                    mock.patch.object(core, "_lg", return_value=None):
                r = core.launch_queue_item("T1", True)
            self.assertFalse(r["launched"], cmd)
            sh.assert_not_called()

    def test_refuses_blocked_item(self):
        for over in ({"needs_human": True}, {"status": "running"}, {"ready": False}):
            with mock.patch.object(core, "_qload", return_value=self.q(**over)), mock.patch.object(core, "_sh") as sh, \
                    mock.patch.object(core, "_lg", return_value=None):
                self.assertFalse(core.launch_queue_item("T1", True)["launched"])
            sh.assert_not_called()


class LogDbTests(TmpDB):
    def test_schema_and_idempotent_migrations(self):
        con = logdb.connect()
        v1 = logdb.migrate(con)
        v2 = logdb.migrate(con)
        self.assertEqual(v1, v2)
        self.assertEqual(v1, logdb.SCHEMA_VERSION)
        n = con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        self.assertEqual(n, len(logdb.MIGRATIONS))
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("runs", "events", "errors", "campaigns", "campaign_items", "servers", "tool_calls", "ingest_state"):
            self.assertIn(t, names)
        self.assertEqual(con.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        con.close()

    def test_redaction_on_write(self):
        logdb.event("test", "info", "k", "token=SUPERSECRET and hf_%s" % FAKE_HF, data={"write_token": "abc123xyz"})
        logdb.error("test", "bad hf_%s" % FAKE_HF, "E", "tb token=ZZZSECRET")
        logdb.tool_call("mcp", "t", {"token": "abcSECRET9"}, 1.0, True)
        con = logdb.connect()
        blob = json.dumps([list(r) for r in con.execute("SELECT * FROM events")] + [list(r) for r in con.execute("SELECT * FROM errors")]
                          + [list(r) for r in con.execute("SELECT * FROM tool_calls")])
        for s in (FAKE_HF, "SUPERSECRET", "abc123xyz", "ZZZSECRET", "abcSECRET9"):
            self.assertNotIn(s, blob)

    def test_scan_text_tracebacks(self):
        probs = list(logdb.scan_text(TB_LOG))
        kinds = [p[0] for p in probs]
        self.assertEqual(kinds.count("traceback"), 1)
        self.assertIn("nan", kinds)
        self.assertIn("oom", kinds)
        self.assertIn("preflight", kinds)
        tb = [p for p in probs if p[0] == "traceback"][0]
        self.assertEqual(tb[1], "ValueError")
        self.assertIn("forward", tb[3])

    def test_incremental_ingest_log_no_duplicates(self):
        lf = Path(self.d) / "E9-s0-2026.log"
        lf.write_text(TB_LOG)
        con = logdb.connect()
        c = {}
        logdb._ingest_log(con, lf, "train", "E9-s0", c)
        n1 = con.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
        logdb._ingest_log(con, lf, "train", "E9-s0", c)
        self.assertEqual(n1, con.execute("SELECT COUNT(*) FROM errors").fetchone()[0])
        self.assertGreaterEqual(n1, 4)
        with open(lf, "a") as f:  # growth: a new traceback appends one error only
            f.write("Traceback (most recent call last):\n  File \"a.py\", line 1\nKeyError: 'x'\n")
        logdb._ingest_log(con, lf, "train", "E9-s0", c)
        self.assertEqual(n1 + 1, con.execute("SELECT COUNT(*) FROM errors").fetchone()[0])
        tbs = [r[0] for r in con.execute("SELECT traceback FROM errors WHERE traceback IS NOT NULL")]
        self.assertTrue(all(FAKE_HF not in (t or "") for t in tbs))

    def test_partial_traceback_waits(self):
        lf = Path(self.d) / "E9-s1-2026.log"
        lf.write_text("ok\nTraceback (most recent call last):\n  File \"a.py\", line 1\n")
        con = logdb.connect()
        logdb._ingest_log(con, lf, "train", None, {})
        self.assertEqual(con.execute("SELECT COUNT(*) FROM errors").fetchone()[0], 0)
        with open(lf, "a") as f:
            f.write("RuntimeError: late\n")
        logdb._ingest_log(con, lf, "train", None, {})
        self.assertEqual(con.execute("SELECT COUNT(*) FROM errors").fetchone()[0], 1)

    def test_live_ingest_twice_is_stable(self):
        a = logdb.ingest()
        s1 = logdb.stats()
        b = logdb.ingest()
        s2 = logdb.stats()
        for t in ("runs", "events", "errors", "campaigns", "campaign_items"):
            self.assertEqual(s1[t], s2[t], t)
        print("\nLIVE INGEST counts:", json.dumps({k: v for k, v in s2.items() if k != "db"}), "first-pass:", json.dumps(a))

    def test_resolve_and_filters(self):
        eid = logdb.error("launcher", "x failed", "T", "tb")
        self.assertEqual(logdb.errors(unresolved=True)["unresolved_total"], 1)
        self.assertFalse(logdb.resolve_error(eid, "  ")["resolved"])
        self.assertTrue(logdb.resolve_error(eid, "fixed by restart")["resolved"])
        self.assertEqual(logdb.errors(unresolved=True)["count"], 0)
        self.assertEqual(logdb.errors(source="launcher")["count"], 1)

    def test_sql_select_only(self):
        logdb.event("t", "info", "k", "m")
        self.assertEqual(logdb.sql("select count(*) from events")["rows"][0][0], 1)
        for bad in ("delete from events", "drop table events", "select 1; select 2", "insert into events(ts) values('x')",
                    "update errors set resolved=1", "pragma writable_schema=1", "attach database 'x' as y",
                    "with a as (select 1) delete from events"):
            self.assertIn("error", logdb.sql(bad), bad)
        self.assertEqual(logdb.stats()["events"], 1)
        self.assertTrue(logdb.sql("select * from events union all select * from events", limit=1)["truncated"])

    def test_writers_never_raise_when_db_unavailable(self):
        with mock.patch.dict(os.environ, {"GOGETA_COCKPIT_DB": "/proc/nonexistent/x.sqlite"}):
            self.assertIsNone(logdb.event("t", "info", "k", "m"))
            self.assertIsNone(logdb.error("t", "m"))
            self.assertEqual(logdb.main(["event", "--source", "t", "--message", "m"]), 0)


if __name__ == "__main__":
    unittest.main()
