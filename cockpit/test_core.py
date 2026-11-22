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


if __name__ == "__main__":
    unittest.main()
