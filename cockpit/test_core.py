"""Cockpit tests (stdlib unittest; run in WSL from the repo root):
    python3 -m unittest cockpit.test_core -v
Reads the live repo read-only. Uses a temporary SQLite file (GOGETA_COCKPIT_DB), never the real one.
Nothing here starts, stops or launches anything real.
"""
import json
import os
import re
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
import runform  # noqa: E402

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

    @unittest.skipUnless(core.RUNS_DIR.exists() and any(core.RUNS_DIR.glob("*/summary.json")), "live run data exists only in WSL")
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

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs a Linux path that cannot exist (/proc)")
    def test_writers_never_raise_when_db_unavailable(self):
        with mock.patch.dict(os.environ, {"GOGETA_COCKPIT_DB": "/proc/nonexistent/x.sqlite"}):
            self.assertIsNone(logdb.event("t", "info", "k", "m"))
            self.assertIsNone(logdb.error("t", "m"))
            self.assertEqual(logdb.main(["event", "--source", "t", "--message", "m"]), 0)


def _make_runform_fixture(root: Path) -> tuple:
    """A fake repo + data dir with one finished experiment run (E5 s0, with a Trackio DB) and a smoke run."""
    repo, data = root / "repo", root / "data"
    rid = "E5-20260101-000000-s0"
    (repo / "experiments/cards").mkdir(parents=True)
    (repo / "experiments/devlogs").mkdir(parents=True)
    (repo / "experiments/cards/E5.md").write_text("# E5 card\n")
    (repo / "experiments/cards/K9.md").write_text("# K9\n")
    (repo / "experiments/experiments.json").write_text(json.dumps([{"id": "E5", "arc": "Test Arc", "title": "Fixture experiment",
        "question": "Does the fixture work?", "prediction": "Yes.", "status": "done", "version": "v0.0.1", "config": "configs/e5.json"}]))
    (repo / "experiments/claims.md").write_text("| ID | Claim | Source | Status | Evidence |\n|---|---|---|---|---|\n"
        "| C5 | E5 beats E0 in the fixture | test | `UNVERIFIED` | none yet |\n"
        "| C94 | plausible bits/char | test | `REFUTED` (leak) | experiments/cards/K9.md |\n"
        "| C107 | val split is exact-dedup only | test | `VERIFIED` | experiments/cards/K9.md |\n"
        "| C131 | corpus is self-duplicated | test | `VERIFIED` | experiments/cards/K9.md |\n")
    d = data / "runs" / rid
    d.mkdir(parents=True)
    cfg = {"config_file": "configs/e5.json", "arch": "fusiondance", "cli_overrides": {"lr": 0.003, "max_tokens": 100000},
           "model": {"d_model": 8, "n_heads": 2, "mlp_hidden": 16, "prelude": ["attn", "gdn"], "core": [], "coda": [], "loops": 1, "window": 8, "ternary": False},
           "train": {"seq_len": 16, "batch_size": 2, "max_tokens": 100000, "lr": 0.003, "min_lr_frac": 0.1, "warmup_steps": 5, "weight_decay": 0.1,
                     "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0, "eval_every": 10, "eval_seqs": 4, "dataset": "toy", "seed": 0},
           "params": {"total": 1000, "embedding": 400, "non_embedding": 600}, "flops_per_token_at_default_loops": 12345.0, "compile": False}
    meta = {"name": rid, "experiment": "E5", "seed": 0, "config": cfg, "git": {"commit": "abc123", "describe": "v0.0.1", "dirty": False, "code_dirty": False},
            "tracker": "trackio", "started": "2026-01-01T00:00:00"}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "summary.json").write_text(json.dumps({**meta, "status": "done", "finished": "2026-01-01T00:05:00", "metrics": {
        "val_loss": 1.5, "val_bits_per_char": 0.5, "tokens": 100352.0, "steps": 3136, "wall_min": 5.0, "tokens_per_s": 1000.0, "params_total": 1000}}))
    s = data / "runs" / "smoke-20260101-000001-s0"
    s.mkdir()
    (s / "summary.json").write_text(json.dumps({"name": s.name, "experiment": "smoke", "seed": 0, "config": {"lr": 0.003}, "git": {}, "status": "done",
        "started": "2026-01-01T00:10:00", "metrics": {"val_loss": 1.7}}))
    (data / "runs" / "E5-20260101-000002-s0").mkdir()  # interrupted: no summary.json
    (data / "runs" / "E5-20260101-000002-s0" / "meta.json").write_text("{}")
    (data / "trackio").mkdir()
    con = sqlite3.connect(str(data / "trackio" / "gogeta-0.db"))
    con.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, timestamp TEXT, run_name TEXT, step INTEGER, metrics TEXT, log_id TEXT, space_id TEXT)")
    for i in range(0, 3136, 20):
        m = {"train/loss": 9.0 / (1 + i / 50.0) + 1.4, "train/lr": 0.003 * min(1, (i + 1) / 5), "train/grad_norm": 2.0 / (1 + i / 100.0),
             "tokens": float((i + 1) * 32), "perf/tokens_per_s": 1000.0 + (i % 7)}
        con.execute("INSERT INTO metrics(run_id,timestamp,run_name,step,metrics) VALUES('x','t',?,?,?)", (rid, i, json.dumps(m)))
        if i % 400 == 0:
            con.execute("INSERT INTO metrics(run_id,timestamp,run_name,step,metrics) VALUES('x','t',?,?,?)",
                        (rid, i, json.dumps({"val/loss": 9.0 / (1 + i / 50.0) + 1.5, "tokens": float((i + 1) * 32)})))
    con.commit()
    con.close()
    return repo, data, rid


GOOD = {"purpose": "Check that the fixture run produces a complete form for the tests.",
        "what_happened": "Nothing unusual: the run finished with exit 0 and no restarts (checked summary.json).",
        "siblings": "No sibling runs of E5 exist in the fixture, so there is nothing to compare against.",
        "verdict": "Counts toward C5 as a weak supporting data point; single seed only.",
        "followups": "None: the fixture has no follow-ups.",
        "claims": [{"id": "C5", "effect": "informs", "evidence": "experiments/cards/E5.md"}]}


class RunFormTests(TmpDB):
    def setUp(self):
        super().setUp()
        self.repo, self.data, self.rid = _make_runform_fixture(Path(self.d))
        self.patches = [mock.patch.object(core, "REPO", self.repo), mock.patch.object(core, "DATA", self.data),
                        mock.patch.object(core, "RUNS_DIR", self.data / "runs"), mock.patch.object(core, "LOG_DIR", self.data / "logs"),
                        mock.patch.object(runform, "_host_gpu", return_value="")]
        for p in self.patches:
            p.start()
        runform._LOG_CACHE.clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_draft_fixture(self):
        d = runform.run_form_draft(self.rid)
        self.assertNotIn("error", d)
        self.assertEqual(d["identity"]["experiment_title"], "Fixture experiment")
        self.assertEqual(d["experiment"]["question"], "Does the fixture work?")
        self.assertEqual(d["architecture"]["params_non_embedding"], 600)
        self.assertEqual(d["hyperparameters"]["lr_peak"], 0.003)
        self.assertEqual(d["provenance"]["commit"], "abc123")
        self.assertIn("41.7%", d["data_caveat"]["text"])
        self.assertFalse(d["data_caveat"]["clean_val"]["available"])
        for c in ("loss", "lr", "grad_norm", "throughput"):
            self.assertIn(c, d["charts"]["available"])
        self.assertIn("siblings", d["charts"]["skipped"])  # a single run has no siblings: skipped, not faked
        self.assertIn("loops", d["charts"]["skipped"])
        self.assertEqual(set(d["narrative_todo"]), set(runform.NARRATIVE))
        ids = [c["id"] for c in d["related_claims"]]
        self.assertIn("C5", ids)
        self.assertTrue({"C94", "C107", "C131"} <= set(ids))
        self.assertEqual(d["form_status"], "none")
        self.assertFalse((self.repo / "experiments/run-forms").exists(), "draft must not write repo files")

    def test_unfinished_and_unknown_runs_refused(self):
        self.assertIn("error", runform.run_form_draft("E5-20260101-000002-s0"))
        self.assertIn("error", runform.run_form_draft("nope"))
        self.assertIn("error", runform.run_form_draft("../etc"))
        self.assertFalse(runform.run_form_submit("E5-20260101-000002-s0", GOOD)["filed"])

    def test_validation_refusals(self):
        def refuse(**over):
            f = dict(GOOD, **over)
            r = runform.run_form_submit(self.rid, f)
            self.assertFalse(r["filed"], over)
            self.assertFalse((self.repo / "experiments/run-forms").exists(), "a refused form writes nothing")
            return " ".join(r["refused"])
        self.assertIn("purpose", refuse(purpose=""))
        self.assertIn("followups", refuse(followups="   "))
        self.assertIn("too short", refuse(verdict="ok"))
        self.assertIn("evidence pointer", refuse(claims=[{"id": "C5", "effect": "informs", "evidence": "see above"}]))
        self.assertIn("evidence pointer", refuse(claims=[{"id": "C5", "effect": "informs", "evidence": ""}]))
        self.assertIn("C94", refuse(siblings="This leans on the K9 finding behind C94 without listing it among the claims."))
        self.assertIn("not in experiments/claims.md", refuse(claims=[{"id": "C999", "effect": "informs", "evidence": "experiments/cards/E5.md"}]))
        self.assertIn("effect", refuse(claims=[{"id": "C5", "effect": "wins", "evidence": "experiments/cards/E5.md"}]))
        self.assertIn("counts toward no claim", refuse(verdict="This run is informative for the arc as a whole.", claims=[]))
        ok = dict(GOOD, claims=[{"id": "C5", "effect": "informs", "evidence": "run %s at v0.0.1" % self.rid}])
        self.assertTrue(runform.run_form_submit(self.rid, ok)["filed"])

    def test_submit_renders_outputs(self):
        r = runform.run_form_submit(self.rid, GOOD)
        self.assertTrue(r["filed"], r)
        out = self.repo / "experiments/run-forms"
        md = (out / ("%s.md" % self.rid)).read_text()
        for must in ("Data caveat (K9)", "[machine]", "[agent]", "Purpose", "Verdict", "![", "not yet available"):
            self.assertIn(must, md)
        self.assertTrue((out / self.rid / "loss.svg").exists())
        self.assertIn(self.rid, (out / "INDEX.md").read_text())
        lst = runform.run_forms_list()
        self.assertEqual(lst["by_form_status"], {"none": 1, "draft": 0, "filled": 1})
        self.assertIn("smoke-20260101-000001-s0", lst["missing"])
        self.assertIn("E5-20260101-000002-s0", lst["unfinished_run_folders"])
        self.assertEqual(runform.run_forms_list("filled")["count"], 1)
        # smoke run: minimal narrative is accepted
        sm = "smoke-20260101-000001-s0"
        fields = {k: "smoke run: no science" for k in runform.NARRATIVE}
        fields["claims"] = []
        fields["verdict"] = "smoke run: no science, counts toward no claim"
        r2 = runform.run_form_submit(sm, fields)
        self.assertTrue(r2["filed"], r2)
        self.assertEqual(runform.run_forms_list()["by_form_status"]["filled"], 2)

    def test_brief_lists_missing_forms(self):
        self.assertEqual(sorted(runform.missing_forms()), sorted([self.rid, "smoke-20260101-000001-s0"]))
        runform.run_form_submit(self.rid, GOOD)
        self.assertNotIn(self.rid, runform.missing_forms())

    def test_tui_draft_saves_state_only(self):
        d = runform.run_form_draft(self.rid, save=True)
        self.assertEqual(d["form_status"], "draft")
        self.assertEqual(runform.run_forms_list()["by_form_status"]["draft"], 1)
        self.assertFalse((self.repo / "experiments/run-forms").exists())


class SvgTests(unittest.TestCase):
    def check(self, svg):
        import xml.etree.ElementTree as ET
        ET.fromstring(svg)
        self.assertLessEqual(len(svg.encode()), runform.SVG_CAP)
        self.assertNotRegex(svg, r"(?i)\bnan\b|\binf\b")

    def test_valid_xml_size_and_no_nan(self):
        n = 6000
        tr = [(i, (i + 1) * 32768, 9.0 / (1 + i / 30.0) + 1.0, 0.003, 1.0 / (1 + i), 1e5) for i in range(n)]
        tr[10] = (10, 360448, float("nan"), float("inf"), None, float("nan"))
        va = [(i, (i + 1) * 32768, 2.0) for i in range(0, n, 500)]
        series = {"train": tr, "val": va, "source": "test"}
        s = {"metrics": {"val_loss": 1.234}, "config": {}, "experiment": "T"}
        sib = {"same_experiment_same_budget": [], "same_experiment_other_budget": [], "other_experiments_same_budget": []}
        ch = runform._build_charts("T-1", series, sib, s)
        self.assertEqual(set(ch["made"]), {"loss", "lr", "grad_norm", "throughput"})
        for svg in ch["made"].values():
            self.check(svg)
            self.assertLess(svg.count("L"), 700, "long series must be downsampled")

    def test_sibling_and_loop_charts(self):
        rows = [{"run": "r%d" % i, "experiment": "T", "seed": 0, "lr": lr, "max_tokens": 1.0, "val_loss": v, "val_bpc": 0.4, "this": i == 1}
                for i, (lr, v) in enumerate([(1e-3, 1.3), (2e-3, 1.1), (3e-3, 1.05), (6e-3, 1.12)])]
        sib = {"same_experiment_same_budget": rows, "same_experiment_other_budget": [], "other_experiments_same_budget": []}
        s = {"metrics": {"val_loss": 1.1, "val_loss_r1": 1.3, "val_loss_r2": 1.2, "val_loss_r4": 1.1}, "config": {"train": {"max_tokens": 1.0}}, "experiment": "T"}
        ch = runform._build_charts("r1", {"train": [], "val": [], "source": None}, sib, s)
        self.assertEqual(set(ch["made"]), {"siblings", "loops"})
        self.assertEqual(set(ch["skipped"]), {"loss", "lr", "grad_norm", "throughput"})
        for svg in ch["made"].values():
            self.check(svg)


@unittest.skipUnless(core.RUNS_DIR.exists() and any(core.RUNS_DIR.glob("*/summary.json")), "no finished runs on this machine")
class RunFormLiveTests(TmpDB):
    def test_real_runs_draft_and_charts(self):
        finished = runform._finished()
        self.assertTrue(finished)
        for rid in finished:
            d = runform.run_form_draft(rid)
            self.assertNotIn("error", d, rid)
            self.assertIn("41.7%", d["data_caveat"]["text"])
            self.assertTrue(d["identity"]["experiment"])
            self.assertIsInstance(d["charts"]["skipped"], dict)
        e = [r for r in finished if not r.startswith("smoke")][0]
        r = runform.run_form_submit(e, {k: "" for k in runform.NARRATIVE})  # refusal path on real data writes nothing
        self.assertFalse(r["filed"])
        print("\nLIVE RUN FORMS: %d finished runs drafted OK" % len(finished))


# =============================================================================== cloud (Vast.ai): fake vastai + fake ssh, never the real service
import hashlib  # noqa: E402
import time  # noqa: E402

import cloud  # noqa: E402
import cloud_fakes  # noqa: E402
import fleet  # noqa: E402
import watchdog  # noqa: E402

FAKE_VAST_KEY = "3f" * 32  # a FAKE 64-hex key (Vast keys are 64 lowercase hex)
CANARY = "FAKE_SECRET_CANARY_7f3a9c"
SHA40 = "a" * 40


def _offer(i, gpu="RTX 3090", dph=0.20, machine=None, vram_mb=24576):
    return {"id": 9000 + i, "gpu_name": gpu, "num_gpus": 1, "dph_total": dph, "reliability2": 0.99, "cuda_max_good": 13.0,
            "driver_version": "580.1", "gpu_ram": vram_mb, "inet_down": 900, "disk_space": 100, "geolocation": "Testland", "machine_id": machine or 500 + i}


@unittest.skipUnless(sys.platform.startswith("linux"), "cloud tests need bash/tar/sha256sum (run in WSL)")
class CloudFx(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        d = self.d
        self.repo, self.data, self.ck = d / "repo", d / "data", d / "cockpit"
        for p in (self.repo / "experiments/data_integrity", self.repo / "configs", self.data / "datasets/code-py-v1", self.data / "runs", self.ck, d / "remote"):
            p.mkdir(parents=True, exist_ok=True)
        ds = self.data / "datasets/code-py-v1"
        files = {"train.bin": b"T" * 5000, "val.bin": b"V" * 700, "tokenizer.json": b'{"tok": 1}', "meta.json": b'{"vocab_size": 8192}',
                 "docs_train.jsonl": b'{"text": "never uploaded"}\n'}
        for n, b in files.items():
            (ds / n).write_bytes(b)
        (ds / "secret.txt").write_text(CANARY + "_FILE")  # canary next to the data: must never be staged
        (ds / ".env").write_text("TOKEN=" + CANARY + "_DOTENV")
        h = lambda n: hashlib.sha256(files[n]).hexdigest()  # noqa: E731
        (self.repo / "experiments/data_integrity/dataset.manifest.json").write_text(json.dumps({"dataset": "code-py-v1",
            "tokenizer": {"file": "code-py-v1/tokenizer.json", "sha256": h("tokenizer.json")},
            "splits": {"train": {"files": {"docs_train.jsonl": {"sha256": h("docs_train.jsonl")}, "train.bin": {"sha256": h("train.bin")}}},
                       "val": {"files": {"val.bin": {"sha256": h("val.bin")}}}}}))
        (self.repo / "configs/e10.json").write_text(json.dumps({"train": {"dataset": "code-py-v1"}}))
        (self.repo / "experiments/experiments.json").write_text(json.dumps([{"id": "E10", "config": "configs/e10.json"}, {"id": "E1", "config": "configs/e10.json"}]))
        for name, body in (("fake_vast.py", cloud_fakes.FAKE_VAST), ("fake_ssh.py", cloud_fakes.FAKE_SSH)):
            (d / name).write_text(body)
        cloud_fakes.make_remote_home(d / "remote", "_template")
        self.state, self.calls, self.sshlog = d / "vast.json", d / "vast-calls.log", d / "ssh.log"
        self.set_state(offers=[_offer(1, dph=0.20), _offer(2, dph=0.25), _offer(3, dph=0.30), _offer(4, "RTX 4090", 0.40)])
        env = {"GOGETA_COCKPIT_DB": str(d / "t.sqlite"), "GOGETA_COCKPIT_DIR": str(self.ck), "GOGETA_VASTAI": "%s %s" % (sys.executable, d / "fake_vast.py"),
               "GOGETA_SSH": "%s %s" % (sys.executable, d / "fake_ssh.py"), "FAKE_VAST_STATE": str(self.state), "FAKE_VAST_CALLS": str(self.calls),
               "FAKE_SSH_ROOT": str(d / "remote"), "FAKE_SSH_LOG": str(self.sshlog), "GOGETA_FLEET_SYNC": "1", "GOGETA_CLOUD_OFFLINE": "1",
               "FAKE_SECRET_CANARY_ENV": CANARY + "_ENV"}
        self.patches = [mock.patch.dict(os.environ, env), mock.patch.object(core, "REPO", self.repo), mock.patch.object(core, "DATA", self.data),
                        mock.patch.object(core, "RUNS_DIR", self.data / "runs"), mock.patch.object(fleet, "POLL_S", 0), mock.patch.object(fleet, "LIVE_SYNC_S", 0),
                        mock.patch.object(fleet, "_head_commit", return_value=(SHA40, True))]
        for p in self.patches:
            p.start()
        fleet._LAST_LIVE[0] = 0.0

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    # helpers
    def set_state(self, **kw):
        st = {"offers": [], "instances": [], "next": 1000, "destroyed": [], "key": FAKE_VAST_KEY}
        if self.state.exists():
            st.update(json.loads(self.state.read_text()))
        st.update(kw)
        self.state.write_text(json.dumps(st))

    def vstate(self):
        return json.loads(self.state.read_text())

    def calls_of(self, word):
        return [json.loads(l) for l in self.calls.read_text().splitlines() if word in l] if self.calls.exists() else []

    def approve(self, **over):
        kw = dict(max_usd_total=10.0, max_usd_per_hour=0.30, max_instances=2, allowed_gpus=["RTX 3090"], purpose="unit test budget", hours=12, created_via="cli")
        kw.update(over)
        r = cloud.create_approval(**kw)
        self.assertTrue(r["created"], r)
        return r["approval"]["id"]

    def wait_exit(self, port, timeout=20):
        f = self.d / "remote" / str(port) / "fusiondance-data/run/cockpit-job.exit"
        for _ in range(int(timeout * 10)):
            if f.exists():
                return True
            time.sleep(0.1)
        return False

    def drive(self, rounds=12):
        """Tick until nothing changes: waits for each background remote job to write its exit file."""
        for _ in range(rounds):
            fleet.tick()
            busy = [r for r in cloud._inst_rows(active_only=True) if r["state"] == "busy"]
            for r in busy:
                self.wait_exit(r["ssh_port"])
            if not cloud._inst_rows(active_only=True) and not fleet.jobs(status="queued"):
                break


class CloudRedactionTests(CloudFx):
    def test_fake_key_masked_everywhere(self):
        for txt in ("vastai --api-key %s show user" % FAKE_VAST_KEY, "vastai --api-key=%s" % FAKE_VAST_KEY, "key %s end" % FAKE_VAST_KEY,
                    '{"api_key": "%s"}' % FAKE_VAST_KEY, "vast_api_key=%s" % FAKE_VAST_KEY):
            self.assertNotIn(FAKE_VAST_KEY, core.redact(txt), txt)
        self.assertIn("[REDACTED]", core.redact("--api-key sekret123"))
        dig = "b" * 64
        self.assertIn(dig, core.redact('"sha256": "%s"' % dig))  # an explicitly labelled digest stays readable
        logdb.event("cloud", "info", "k", "vastai --api-key %s" % FAKE_VAST_KEY, data={"k": FAKE_VAST_KEY})
        logdb.error("cloud", "boom %s" % FAKE_VAST_KEY, "E", "tb --api-key %s" % FAKE_VAST_KEY)
        logdb.tool_call("mcp", "gogeta_vast_rent", {"note": FAKE_VAST_KEY}, 1.0, True)
        inc = watchdog.record({"kind": "cloud_test", "subject": "vast:x", "subject_type": "cloud-instance", "summary": "leak %s" % FAKE_VAST_KEY,
                               "tail": "--api-key %s" % FAKE_VAST_KEY, "evidence": {"key": FAKE_VAST_KEY}})
        self.assertIsNotNone(inc)
        con = logdb.connect()
        blob = json.dumps([[list(r) for r in con.execute("SELECT * FROM %s" % t)] for t in ("events", "errors", "tool_calls", "incidents")], default=str)
        con.close()
        self.assertNotIn(FAKE_VAST_KEY, blob)
        self.assertNotIn(FAKE_VAST_KEY, "".join(p.read_text() for p in (self.ck / "incidents").glob("*.json")))
        self.assertNotIn(FAKE_VAST_KEY, (self.ck / "incidents.log").read_text())

    def test_balance_never_returns_amount_or_email(self):
        r = cloud.balance_ok()
        self.assertEqual(r["credit"], "sufficient")
        self.assertNotIn("example.com", json.dumps(r))
        self.assertNotIn(FAKE_VAST_KEY, json.dumps(r))
        self.set_state(credit=1.0)
        self.assertEqual(cloud.balance_ok()["credit"], "low")

    def test_never_passes_api_key(self):
        with self.assertRaises(ValueError):
            cloud._vast(["show", "user", "--api-key", "x"])


class CloudGuardTests(CloudFx):
    def rent(self, offer=9001, **kw):
        a = dict(max_hours=4, max_usd=0, purpose="unit test rent", confirm=True)
        a.update(kw)
        return cloud.rent(offer, **a)

    def test_refused_without_approval_and_no_create_call(self):
        r = self.rent()
        self.assertFalse(r["rented"])
        self.assertIn("no spend approval", r["refused"])
        self.assertEqual(self.calls_of('"create"'), [])

    def test_refusals_budget_gpu_per_hour_instances_expired_forged(self):
        aid = self.approve(max_usd_total=1.0, max_instances=1)
        self.assertIn("projected", self.rent(9001, max_hours=10)["refused"])         # 0.20 x 10 h = $2 > $1
        self.assertIn("not allowed", self.rent(9004, max_hours=1)["refused"])       # RTX 4090 not in the list
        aid2 = self.approve(max_usd_per_hour=0.22, max_usd_total=5)
        self.assertIn("per-hour cap", self.rent(9002, max_hours=1, approval_id=aid2)["refused"])  # 0.25 > 0.22
        ok = self.rent(9001, max_hours=2, approval_id=aid)
        self.assertTrue(ok["rented"], ok)
        self.assertIn("max_instances", self.rent(9003, max_hours=1, approval_id=aid)["refused"])
        self.assertEqual(len(self.calls_of('"create"')), 1, "create runs only after every check passed")
        a = next(x for x in cloud._load_approvals() if x["id"] == aid)
        self.assertAlmostEqual(a["used_usd"], 0.40, places=3)   # reservation dph x max_hours recorded before create
        lst = cloud._load_approvals()                            # an agent hand-editing the file does not get a valid approval
        lst.append(dict(lst[0], id="APR-FORGED", created_by="agent", reservations=[]))
        lst.append(dict(lst[0], id="APR-OLD", created_at="2026-01-01T00:00:00", expires_at="2026-01-01T10:00:00", reservations=[]))
        lst.append(dict(lst[0], id="APR-LONG", expires_at=cloud._iso(time.time() + 48 * 3600), reservations=[]))
        cloud._save_approvals(lst)
        self.assertIn("not created by the human", self.rent(9003, max_hours=1, approval_id="APR-FORGED")["refused"])
        self.assertIn("expired", self.rent(9003, max_hours=1, approval_id="APR-OLD")["refused"])
        self.assertIn("more than 24 h", self.rent(9003, max_hours=1, approval_id="APR-LONG")["refused"])
        self.assertEqual(len(self.calls_of('"create"')), 1)

    def test_dry_run_prints_command_and_cost_only(self):
        self.approve()
        r = cloud.rent(9001, 3, 0, "unit test dry run", dry_run=True)
        self.assertTrue(r["dry_run"])
        self.assertIn("vastai create instance 9001", r["command"])
        self.assertAlmostEqual(r["projected_usd"], 0.60, places=3)
        self.assertEqual(self.calls_of('"create"'), [])
        r = cloud.rent(9001, 3, 0, "unit test no confirm")
        self.assertIn("confirm", r["refused"])
        self.assertEqual(self.calls_of('"create"'), [])
        ok = cloud.rent(9001, 3, 0, "unit test real", confirm=True)
        d = cloud.destroy(ok["instance_id"], dry_run=True)
        self.assertTrue(d["dry_run"])
        self.assertIn("destroy instance", d["command"])
        self.assertEqual(self.calls_of('"destroy"'), [])
        p = fleet.fleet_down(dry_run=True)
        self.assertEqual(len(p["destroy"]), 1)
        self.assertEqual(self.calls_of('"destroy"'), [])

    def test_never_touches_foreign_instances(self):
        self.set_state(instances=[{"id": 77, "label": "someone-else", "gpu_name": "A100", "dph_total": 1.0, "actual_status": "running"},
                                  {"id": 78, "label": "gogeta-cockpit-pilot", "gpu_name": "RTX 3090", "dph_total": 0.23, "actual_status": "running",
                                   "public_ipaddr": "127.0.0.1", "ports": {"22/tcp": [{"HostPort": "20078"}]}, "start_date": time.time() - 3600}])
        for iid in ("77", "78"):
            self.assertFalse(cloud.destroy(iid, confirm=True).get("destroyed"))
            self.assertIn("refused", fleet.vast_down(iid, confirm=True))
            self.assertIn("refused", fleet.vast_run_job(iid, "E10 --seed 0", confirm=True))
        aid = self.approve()
        ok = cloud.rent(9001, 1, 0, "unit test mine", approval_id=aid, confirm=True)
        st = self.vstate()
        for i in st["instances"]:
            if str(i["id"]) == ok["instance_id"]:
                i["label"] = "relabelled-by-someone"
        self.state.write_text(json.dumps(st))
        self.assertIn("does not match", cloud.destroy(ok["instance_id"], confirm=True)["refused"])
        for r in cloud._inst_rows(active_only=True):  # even over-time, the watchdog destroys only registry+label matches
            cloud.inst_update(r["id"], created_epoch=time.time() - 99 * 3600)
        fleet.watch()
        self.assertEqual(self.calls_of('"destroy"'), [])
        inst = cloud.instances()
        self.assertEqual({x["id"] for x in inst["not_managed"]} >= {"77", "78"}, True)
        self.assertGreater(inst["burn_usd_per_h"], 1.0)  # foreign instances still cost money and are reported

    def test_no_mcp_path_creates_or_edits_approvals(self):
        src = (Path(__file__).resolve().parent / "mcp_server.py").read_text()
        for bad in ("create_approval", "revoke_approval", "_reserve", "_settle", "_save_approvals", "_rekey", "approve_spend", "revoke_ssh_key"):
            self.assertNotIn(bad, src, bad)
        names = re.findall(r'(?:expose\([\w.]+,\s*|@tool\()"(gogeta_\w+)"', src)
        self.assertIn("gogeta_vast_approvals", names)
        for n in names:
            self.assertFalse(re.search(r"approve|create_appr|grant|budget", n), n)
        try:
            import mcp_server  # noqa: F401  (needs the cockpit venv: mcp SDK)
        except ImportError:
            return
        import asyncio
        tools = asyncio.run(mcp_server.mcp.list_tools())
        forbidden = {"max_usd_total", "max_usd_per_hour", "max_instances", "allowed_gpus", "created_by", "created_via", "expires_at", "revoked_at"}
        for t in tools:
            props = set((t.inputSchema or {}).get("properties", {}))
            self.assertFalse(props & forbidden, (t.name, props & forbidden))
            self.assertFalse(re.search(r"approv", t.name) and t.name != "gogeta_vast_approvals", t.name)
        appr = next(t for t in tools if t.name == "gogeta_vast_approvals")
        self.assertTrue(appr.annotations.readOnlyHint)

    def test_cli_approve_refused_without_tty(self):
        with mock.patch.object(cloud, "_tty", return_value=False):
            self.assertEqual(cloud.main(["approve", "--max-usd", "5", "--max-usd-per-hour", "0.3", "--purpose", "agent attempt"]), 3)
        self.assertEqual(cloud._load_approvals(), [])
        self.assertFalse(cloud.create_approval(5, 0.3, 1, ["RTX 3090"], "via mcp attempt", 1, created_via="mcp")["created"])


class CloudWatchdogTests(CloudFx):
    def rent_mode(self, mode, aid, offer=9001, max_hours=4):
        r = cloud.rent(offer, max_hours, 0, "unit test watchdog", approval_id=aid, confirm=True)
        self.assertTrue(r["rented"], r)
        cloud.inst_update(r["instance_id"], mode=mode, state="ready", state_since=time.time())
        return r["instance_id"]

    def test_overtime_idle_budget_and_suggestions(self):
        aid = self.approve(max_instances=4, max_usd_total=20)
        over = self.rent_mode("fleet", aid, 9001)
        idle = self.rent_mode("fleet", aid, 9002)
        man = self.rent_mode("manual", aid, 9003)
        fresh = self.rent_mode("manual", aid, 9002)
        cloud.inst_update(over, created_epoch=time.time() - 5 * 3600)          # > max_hours 4
        cloud.inst_update(idle, state_since=time.time() - 6 * 60)              # fleet: 6 min without a job (limit 5)
        cloud.inst_update(man, state_since=time.time() - 11 * 60, last_busy_epoch=time.time() - 11 * 60)  # manual: AGENTS 0h, 11 min > 10
        cloud.inst_update(fresh, state_since=time.time() - 9 * 60, last_busy_epoch=time.time() - 9 * 60)  # manual, 9 min: kept (flagged idle)
        f = fleet.watch()
        kinds = sorted(x["kind"] for x in f)
        self.assertEqual(kinds, ["cloud_destroy_idle", "cloud_destroy_idle", "cloud_destroy_overtime"])
        self.assertEqual(sorted(c[2] for c in self.calls_of('"destroy"')), sorted([over, idle, man]))
        self.assertEqual(cloud.inst(fresh)["state"], "ready")
        row = fleet.vast_status(fresh, refresh=False)["instances"][0]
        self.assertTrue(row["idle_flag"])
        self.assertIn("AGENTS 0h", row["suggest"])
        for x in f:  # each destroy becomes an incident
            self.assertIsNotNone(watchdog.record(x, x.get("actions")))
        self.assertEqual(watchdog.incidents(open_only=True)["open_total"], 3)
        # spend cap: an old busy instance that already burnt the budget
        cloud.inst_update(fresh, state="busy", job_id="JOB-00000000", last_poll_ok_epoch=time.time(), created_epoch=time.time() - 3.9 * 3600, dph=5.0)
        f = fleet.watch()
        self.assertEqual([x["kind"] for x in f], ["cloud_destroy_budget"])
        self.assertEqual(cloud._inst_rows(active_only=True), [])
        a = next(x for x in cloud._load_approvals() if x["id"] == aid)
        self.assertTrue(all(r["settled"] for r in a["reservations"]), "destroy settles the reservation with the actual cost")

    def test_stuck_setup_replaced_once_then_fails(self):
        aid = self.approve(max_instances=2)
        up = fleet.fleet_up(1, "RTX 3090", aid, max_hours=2, purpose="unit test stuck", confirm=True, dry_run=False)
        jid = fleet.dispatch(["E10 --seed 0"], up["session_id"], dry_run=False, confirm=True)["jobs"][0]
        rows = cloud._inst_rows(active_only=True)
        self.assertEqual(len(rows), 1)
        first_machine = rows[0]["machine_id"]
        for attempt in (1, 2):
            r = cloud._inst_rows(active_only=True)
            if not r:
                fleet.tick()
                r = cloud._inst_rows(active_only=True)
            cloud.inst_update(r[0]["id"], state="setup", created_epoch=time.time() - 16 * 60)
            f = fleet.watch()
            self.assertEqual([x["kind"] for x in f], ["cloud_destroy_stuck"])
            if attempt == 1:
                self.assertEqual(fleet.job(jid)["status"], "queued")
                fleet.tick()  # re-rent on ANOTHER machine
                r2 = cloud._inst_rows(active_only=True)
                self.assertEqual(len(r2), 1)
                self.assertNotEqual(r2[0]["machine_id"], first_machine)
        self.assertEqual(fleet.job(jid)["status"], "failed")
        self.assertEqual(len(self.calls_of('"create"')), 2)

    def test_remote_log_problems_become_incidents(self):
        aid = self.approve()
        iid = self.rent_mode("manual", aid)
        cloud.sync_live()
        r = cloud.inst(iid)
        home = self.d / "remote" / str(r["ssh_port"])
        fleet.poll_instance(r)  # creates the fake remote home
        (home / "fusiondance-data/logs").mkdir(parents=True, exist_ok=True)
        (home / "fusiondance-data/logs/E10-s0-1.log").write_text("step 10/100 loss 2.0 lr 1e-3 gn 1 100k tok/s r=None\nTraceback (most recent call last):\n"
                                                                 "  File \"/root/Gogeta-0/gogeta/train.py\", line 9\nRuntimeError: CUDA error: boom\n"
                                                                 "step 11/100 loss nan lr 1e-3\n")
        p, f = fleet.poll_instance(cloud.inst(iid))
        kinds = sorted(x["kind"] for x in f)
        self.assertIn("remote_traceback", kinds)
        self.assertIn("remote_nan", kinds)
        self.assertTrue(all(x["subject"].startswith("vast:gogeta-cockpit-") for x in f))
        errs = logdb.errors(source="remote")["errors"]
        self.assertTrue(errs and "instance" in (errs[0]["context"] or ""))
        for x in f:
            watchdog.record(x)
        self.assertGreaterEqual(watchdog.incidents(kind="remote_traceback")["count"], 1)


class CloudFleetTests(CloudFx):
    def test_jit_fleet_reuse_teardown_results_and_costs(self):
        aid = self.approve(max_instances=2, max_usd_total=10)
        up = fleet.fleet_up(2, "RTX 3090", aid, max_hours=2, purpose="unit test fleet", confirm=True, dry_run=False)
        self.assertTrue(up["opened"], up)
        self.assertEqual(self.calls_of('"create"'), [], "fleet_up never pre-provisions")
        self.assertIn("one GPU SKU", fleet.fleet_up(1, "RTX 4090", aid, purpose="unit test other sku", confirm=True, dry_run=False)["refused"])
        plan = fleet.dispatch(["E10 --seed 0", "E10 --seed 1", "E1 --seed 0"], up["session_id"])
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["would_rent_now"], 2)
        self.assertEqual([s["step"] for s in plan["steps"]], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(self.calls_of('"create"'), [])
        d = fleet.dispatch(["E10 --seed 0", "E10 --seed 1", "E1 --seed 0"], up["session_id"], dry_run=False, confirm=True)
        self.assertEqual(d["dispatched"], 3)
        self.assertEqual(len(self.calls_of('"create"')), 2, "fleet size = min(ready jobs 3, max_instances 2)")
        self.drive(20)
        js = fleet.jobs(up["session_id"])
        self.assertEqual([j["status"] for j in js], ["done"] * 3, [(j["status"], j["error"]) for j in js])
        self.assertTrue(all(j["verified"] for j in js))
        self.assertEqual(len(self.calls_of('"create"')), 2, "the third job reused an instance (no new rent, no new setup)")
        self.assertEqual(len(self.calls_of('"destroy"')), 2, "both destroyed when the queue emptied")
        self.assertEqual(cloud._inst_rows(active_only=True), [])
        self.assertEqual(fleet.session(up["session_id"])["status"], "done")
        for j in js:
            man = json.loads((Path(j["result_dir"]) / "repro-manifest.json").read_text())
            self.assertEqual(man["compute"]["provider"], "vast.ai")
            self.assertEqual(man["compute"]["gpu"], "RTX 3090")
            self.assertIsNotNone(man["compute"]["dph_usd"])
            self.assertNotIn("127.0.0.1", json.dumps(man), "no IPs in a manifest that may be committed")
            self.assertEqual(man["manifest_sha256"], fleet._manifest_hash(man))
            self.assertTrue((Path(j["result_dir"]) / "summary.json").exists())
        reused = [j for j in js if j["setup_usd"] == 0]
        self.assertGreaterEqual(len(reused), 1)
        st = fleet.fleet_status()
        self.assertEqual(st["running"], 0)
        self.assertEqual(st["lifetime"]["jobs_done"], 3)
        self.assertTrue(all(x["cost"] and "teardown_usd" in x["cost"] for x in st["jobs"]))

    def test_canary_never_staged_or_sent(self):
        aid = self.approve(max_instances=1)
        up = fleet.fleet_up(1, "RTX 3090", aid, max_hours=2, purpose="unit test canary", confirm=True, dry_run=False)
        fleet.dispatch(["E10 --seed 0"], up["session_id"], dry_run=False, confirm=True)
        self.drive(10)
        self.assertEqual(fleet.jobs(up["session_id"])[0]["status"], "done")
        sent = self.sshlog.read_bytes()
        self.assertGreater(len(sent), 1000)
        for blob, where in ((sent, "ssh argv/env/stdin"), (self.calls.read_bytes(), "vastai calls")):
            self.assertNotIn(CANARY.encode(), blob, where)
            self.assertNotIn(b"PRIVATE KEY", blob)
        self.assertNotIn(b"never uploaded", sent, "docs_*.jsonl is not in the allowlist")
        self.assertIn(b"IdentitiesOnly=yes", sent)
        self.assertIn(b"ForwardAgent=no", sent)
        for p in (self.d / "remote").rglob("*"):
            if p.is_file():
                self.assertNotIn(CANARY.encode(), p.read_bytes(), str(p))
        ups = list((self.d / "remote").glob("2*/fusiondance-data/datasets/code-py-v1"))
        self.assertTrue(ups)
        self.assertEqual(sorted(x.name for x in ups[0].iterdir() if not x.name.startswith(".gogeta")),
                         ["meta.json", "tokenizer.json", "train.bin", "val.bin"])

    def test_transient_failure_retried_once(self):
        aid = self.approve(max_instances=1)
        up = fleet.fleet_up(1, "RTX 3090", aid, max_hours=2, purpose="unit test retry", confirm=True, dry_run=False)
        (self.d / "remote/_template/EXIT").write_text("1")
        (self.d / "remote/_template/EXTRA_LOG").write_text("triton autotuning cache error: connection reset\n")
        fleet.dispatch(["E10 --seed 0"], up["session_id"], dry_run=False, confirm=True)
        self.drive(16)
        j = fleet.jobs(up["session_id"])[0]
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["attempts"], 2, "one retry for a transient failure, no more")
        self.assertEqual(cloud._inst_rows(active_only=True), [])

    def test_manual_flow_and_unmanaged_readonly(self):
        aid = self.approve(max_instances=2)
        dry = fleet.vast_up("RTX 3090", 2, aid, "unit test manual up")
        self.assertTrue(dry["dry_run"])
        self.assertEqual(self.calls_of('"create"'), [])
        up = fleet.vast_up("RTX 3090", 2, aid, "unit test manual up", confirm=True, dry_run=False)
        self.assertTrue(up["rented"], up)
        iid = up["instance_id"]
        cloud.sync_live()
        self.assertIn("refused", fleet.vast_run_job(iid, "E10 --seed 0", confirm=True))          # data not pushed yet
        self.assertTrue(fleet.vast_setup(iid, confirm=True)["started"])
        self.assertTrue(fleet.vast_push_data(iid, "code-py-v1", confirm=True)["pushed"])
        self.assertIn("refused", fleet.vast_run_job(iid, "E10 --seed 0; rm -rf /", confirm=True))
        r = fleet.vast_run_job(iid, "E10 --seed 3 --max-tokens 3276800", confirm=True)
        self.assertTrue(r.get("ok"), r)
        self.assertTrue(self.wait_exit(cloud.inst(iid)["ssh_port"]))
        s = fleet.vast_status(iid)["instances"][0]
        for k in ("gpu", "dph", "accrued_usd", "util_pct", "mem_used_mib", "power_w", "val_loss", "last_line", "run", "fun", "budget_bar", "ip"):
            self.assertIn(k, s)
        self.assertEqual(s["util_pct"], 97)
        self.assertEqual(s["val_loss"], 1.2345)
        self.assertTrue(s["fun"]["loss_sparkline"])
        self.assertIn(s["fun"]["mood"].split()[0], ("(^_^)", "(-_-)", "(o_o)", "(x_x)"))
        self.assertGreater(s["fun"]["tokens_per_cent"], 0)
        # AGENTS 0h: that poll saw the exit -> pulled the run -> nothing queued -> destroyed at once
        self.assertEqual(cloud.inst(iid)["state"], "destroyed")
        self.assertEqual([c[2] for c in self.calls_of('"destroy"')], [iid])
        done = fleet.jobs(status="done")
        self.assertEqual(done[0]["experiment"], "E10")
        self.assertTrue((Path(done[0]["result_dir"]) / "repro-manifest.json").exists())
        self.assertTrue(fleet.vast_down(iid, confirm=True)["destroyed"])  # idempotent
        # a second box: two jobs, the second queued behind the first; the box goes down only after BOTH (then nothing is queued)
        up2 = fleet.vast_up("RTX 3090", 2, aid, "unit test manual queue", confirm=True, dry_run=False)
        i2 = up2["instance_id"]
        cloud.sync_live()
        fleet.vast_setup(i2, confirm=True)
        fleet.vast_push_data(i2, "code-py-v1", confirm=True)
        self.assertTrue(fleet.vast_run_job(i2, "E10 --seed 1", confirm=True).get("ok"))
        q = fleet.vast_run_job(i2, "E1 --seed 2", confirm=True)
        self.assertTrue(q.get("queued"), q)
        for _ in range(3):
            self.assertTrue(self.wait_exit(cloud.inst(i2)["ssh_port"]))
            fleet.tick()
            if cloud.inst(i2)["state"] == "destroyed":
                break
        self.assertEqual(cloud.inst(i2)["state"], "destroyed")
        self.assertEqual(len(fleet.jobs(status="done")), 3)
        # an unmanaged gogeta-cockpit-* instance (like the pilot): status/tail/pull work, control is refused
        st = self.vstate()
        st["instances"].append({"id": 4242, "label": "gogeta-cockpit-pilot", "gpu_name": "RTX 3090", "dph_total": 0.232, "actual_status": "running",
                                "public_ipaddr": "127.0.0.1", "ports": {"22/tcp": [{"HostPort": "24242"}]}, "start_date": time.time() - 1800, "gpu_totalram": 24576,
                                "geolocation": "United States, US", "cpu_cores_effective": 16, "cpu_ram": 257757, "reliability2": 0.99})
        self.state.write_text(json.dumps(st))
        s = fleet.vast_status("4242")["instances"][0]
        self.assertEqual(s["mode"], "unmanaged")
        self.assertAlmostEqual(s["accrued_usd"], 0.116, places=2)
        self.assertEqual(s["vram_gb"], 24.0)
        for fn in (lambda: fleet.vast_down("4242", confirm=True), lambda: fleet.vast_stop_job("4242", confirm=True),
                   lambda: fleet.vast_setup("4242", confirm=True), lambda: fleet.vast_push_data("4242", confirm=True)):
            self.assertIn("refused", fn())
        self.assertNotIn("4242", [c[2] for c in self.calls_of('"destroy"')])

    def test_cloud_session_destroys_even_on_failure(self):
        aid = self.approve(max_instances=1)
        with mock.patch.object(fleet, "watch", side_effect=RuntimeError("boom mid-session")):
            with self.assertRaises(RuntimeError):
                fleet.cloud_session(["E10 --seed 0"], "RTX 3090", 0.3, confirm=True, dry_run=False, wait=True)
        self.assertEqual(cloud._inst_rows(active_only=True), [], "try/finally destroyed the box")
        self.assertEqual(len(self.calls_of('"destroy"')), 1)
        r = fleet.cloud_session(["E10 --seed 0", "E1 --seed 0"], "RTX 3090", 0.3, confirm=True, dry_run=False, wait=True)
        self.assertEqual([j["status"] for j in r["jobs"]], ["done", "done"])
        self.assertEqual(len(self.calls_of('"create"')), 2, "one box per session; the 2nd job reused it")
        self.assertEqual(cloud._inst_rows(active_only=True), [])

    def test_rank_offers_vram_floor_and_measured_cost(self):
        self.set_state(offers=[_offer(1, "RTX 3060", 0.048, vram_mb=12288), _offer(2, "RTX A4000", 0.081, vram_mb=16384),
                               _offer(3, "RTX 3090", 0.179), _offer(4, "RTX 5090", 0.494, vram_mb=32607), _offer(5, "RTX 3050", 0.03, vram_mb=8192)])
        r = fleet.rank_offers("E2")
        self.assertEqual(r["vram_floor_gb"], 12.0)
        self.assertNotIn("RTX 3050", [o["gpu_name"] for o in r["offers"]])
        self.assertEqual(r["best_gpu"], "RTX 3060")          # no benchmarks: cheapest above the floor
        now = time.time()
        for gpu, tps in (("RTX 3090", 400000.0), ("RTX 3060", 20000.0)):
            iid = "b-%s" % gpu[-4:]
            cloud._inst_insert(id=iid, label="gogeta-cockpit-bench", gpu_name=gpu, dph=0.1, created_epoch=now, state="destroyed", destroyed_epoch=now)
            fleet._x("INSERT INTO cloud_jobs(id,experiment,status,instance_id,metrics) VALUES(?,?,?,?,?)",
                     ("JOB-%s" % iid, "E2", "done", iid, json.dumps({"tokens_per_s": tps})))
        r = fleet.rank_offers("E2")
        self.assertEqual(r["best_gpu"], "RTX 3090")          # measured $/token wins: 0.179/(400k tok/s) < 0.048/(20k tok/s)
        self.assertIsNotNone(r["offers"][0]["usd_per_mtok"])

    def test_poll_parse(self):
        txt = ("@@GPU\n70, 99, 20000, 24576, NVIDIA GeForce RTX 3090, 580.1, 340.2, 350.0, 1900, 9750\n@@APPS\n1\n@@RUN\n"
               '{"name": "E1-x-s0", "experiment": "E1", "seed": 0, "lr": 0.003, "max_tokens": 20000000, "started": "2026-11-22T00:00:00", "finished": false}\n'
               "@@JOB\njob=JOB-0000abcd\nalive=1\n@@PROCS\n123 python -m gogeta.train --config configs/e1.json\n@@SETUP\ndone\n@@LOG\npath=bridge-E1.log\nmtime=100\n"
               "@@HEAD\nparams: total 14.24M non-emb 11.09M | steps 6104 x 32768 tok | looped=False\n@@TAIL\n"
               "step   5100/6104 loss 0.5861 lr 9.89e-04 gn 0.11 128k tok/s r=None\n  eval step 5250: val/loss 0.6289\n"
               "step   5300/6104 loss 0.5575 lr 8.52e-04 gn 0.11 123k tok/s r=None\n@@RUNS\nE1-x-s0\n@@NOW\n160\n")
        p = fleet.parse_poll(txt)
        m = p["metrics"]
        self.assertEqual((m["step"], m["total"], m["loss"], m["tok_s"], m["val_loss"]), (5300, 6104, 0.5575, 123000, 0.6289))
        self.assertAlmostEqual(m["eta_min"], (6104 - 5300) * 32768 / 123000 / 60, places=1)
        self.assertEqual(p["gpu"]["power_w"], 340.2)
        self.assertEqual(p["run"]["experiment"], "E1")
        self.assertEqual(p["procs"][0][:3], "123")
        self.assertEqual(len(fleet.sparkline([3, 2, 1, 0.5])), 4)


@unittest.skipUnless(cloud.WIN_EXE.exists() and not os.environ.get("GOGETA_VASTAI"), "live smoke needs the Windows vastai.exe (WSL interop)")
class CloudLiveReadOnlySmoke(TmpDB):
    """READ-ONLY against the real account: search offers and list instances. Rents nothing, changes nothing."""
    def test_offers_and_instances(self):
        o = cloud.offers("RTX 3090", limit=50)
        self.assertNotIn("error", o, o.get("error"))
        self.assertGreater(o["count"], 0)
        i = cloud.instances()
        self.assertTrue(i["available"])
        s = fleet.vast_status(refresh=False)
        self.assertNotIn("error", s)
        print("\nLIVE VAST (read-only): %d RTX 3090 offers, cheapest $%s/h, median $%s/h; %d instance(s) on the account (%d gogeta-cockpit-*)" % (
            o["count"], o["min_dph"], o["median_dph"], len(i["cockpit"]) + len(i["not_managed"]),
            sum(1 for x in i["not_managed"] if str(x.get("label") or "").startswith(cloud.LABEL_PREFIX)) + len(i["cockpit"])))


if __name__ == "__main__":
    unittest.main()


class ExpensesTests(unittest.TestCase):
    def test_summary_and_reconcile_keep_only_two_numbers(self):
        import expenses
        d = Path(tempfile.mkdtemp())
        led = d / "expenses.json"
        led.write_text(json.dumps({"currency": "USD", "entries": [
            {"id": "X1", "date": "d", "type": "credit_purchase", "amount": 100.0},
            {"id": "X2", "date": "d", "type": "compute", "amount": 0.3, "experiments": ["A", "B"]}], "reconciliations": []}))
        fake = {"credit": 99.5, "total_spend": -0.5, "email": "someone@example.com", "billaddress_line1": "1 Road"}
        with mock.patch.object(expenses, "LEDGER", led):
            s = expenses.expenses()
            self.assertEqual((s["credit_bought"], s["compute_estimated"], s["per_experiment_estimated"]), (100.0, 0.3, {"A": 0.15, "B": 0.15}))
            with mock.patch("providers.vast.run", return_value=(0, dict(fake), "")):
                r = expenses.expenses_reconcile("t")
            self.assertEqual((r["appended"]["credit_left"], r["appended"]["provider_total_spend"]), (99.5, 0.5))
            txt = led.read_text()
            self.assertNotIn("example.com", txt)
            self.assertNotIn("Road", txt)
            self.assertAlmostEqual(expenses.expenses()["unattributed_gap"], 0.2)
        shutil.rmtree(d)
