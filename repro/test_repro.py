"""Selftests for manifest.py, reproduce.py and lint_science.py, on throw-away fixtures (temp dirs, temp git repos).

    python3 repro/test_repro.py          (stdlib only; no GPU; touches nothing outside a temp directory)
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lint_science  # noqa: E402
import manifest as M  # noqa: E402
import reproduce as R  # noqa: E402

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def sh(cwd, *cmd):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, (cmd, r.stderr)
    return r.stdout.strip()


def commit(repo, msg, tag=None):
    sh(repo, "git", "add", "-A")
    sh(repo, "git", "commit", "-qm", msg)
    if tag:
        sh(repo, "git", "tag", "-a", tag, "-m", tag)
    return sh(repo, "git", "rev-parse", "HEAD")


def w(p: pathlib.Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class ManifestFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="repro-test-"))
        self.repo, self.data = self.tmp / "repo", self.tmp / "data"
        self.repo.mkdir()
        sh(self.repo, "git", "init", "-q")
        w(self.repo / "gogeta" / "train.py",
          "import torch\nwith torch.autocast('cuda', dtype=torch.bfloat16): pass\ntorch.backends.cuda.matmul.allow_tf32 = True\n")
        w(self.repo / "gogeta" / "model.py", "X = 1\n")
        w(self.repo / "configs" / "t.json", '{"experiment": "T"}')
        w(self.repo / "requirements.txt", "numpy\n")
        self.c1 = commit(self.repo, "code", "v0.0.1")
        # dataset + run
        ds = self.data / "datasets" / "toy"
        w(ds / "meta.json", json.dumps({"source": "x", "splits": {"train": {"sha256_16": "abc"}}}))
        (ds / "train.bin").write_bytes(b"\x01\x02" * 1000)
        (ds / "val.bin").write_bytes(b"\x03\x04" * 100)
        w(ds / "tokenizer.json", "{}")
        run = self.data / "runs" / "T-20260101-000000-s0"
        cfg = {"config_file": "configs/t.json", "arch": None, "cli_overrides": {"seed": 0}, "model": {"d": 1},
               "train": {"dataset": "toy", "seed": 0}, "compile": True}
        summ = {"name": run.name, "experiment": "T", "seed": 0, "config": cfg,
                "git": {"commit": self.c1, "branch": "main", "describe": "v0.0.1", "dirty": False, "code_dirty": False},
                "tracker": "none", "started": "2999-01-01T00:00:00", "status": "done", "finished": "2999-01-01T00:10:00",
                "metrics": {"val_loss": 1.0, "steps": 10, "tokens": 100.0, "wall_min": 1.0, "checkpoint": "/x/model.pt"}}
        w(run / "meta.json", json.dumps({k: summ[k] for k in ("name", "experiment", "seed", "config", "git")}))
        w(run / "summary.json", json.dumps(summ))
        ck = self.data / "checkpoints" / run.name
        ck.mkdir(parents=True)
        (ck / "model.pt").write_bytes(b"weights")
        self.run = run
        self.cache = common.HashCache(self.tmp / "cache.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self):
        return M.build_manifest(self.run, repo=self.repo, data=self.data, cache=self.cache)

    # ---------------------------------------------------------------- manifest
    def test_manifest_fields_and_unknowns(self):
        m = self.build()
        self.assertEqual(m["code"]["commit"], self.c1)
        self.assertTrue(m["code"]["code_exactly_recoverable"])
        self.assertEqual(m["precision"]["autocast_dtype"], ["bfloat16"])
        self.assertIs(m["precision"]["tf32_matmul"], True)
        self.assertFalse(m["deterministic_mode"]["enabled"])
        self.assertEqual(m["config"]["file_sha256_at_commit"], common.sha256_bytes(b'{"experiment": "T"}'))
        self.assertEqual(m["dataset"]["files"]["train.bin"]["sha256"], common.sha256_bytes(b"\x01\x02" * 1000))
        self.assertEqual(m["artifacts"]["checkpoint"]["sha256"], common.sha256_bytes(b"weights"))
        # never guess: run-time env was not recorded
        rt = m["environment"]["recorded_at_run_time"]
        for k in ("torch", "cuda", "nvidia_driver", "gpu", "pip_freeze_sha256"):
            self.assertEqual(rt[k], common.UNKNOWN)
        self.assertEqual(m["launch"]["argv"][-3:], ["--seed", "0", "--compile"])
        self.assertNotIn("checkpoint", m["metrics"])
        self.assertEqual(m["manifest_sha256"], common.manifest_self_hash(m))

    def test_dirty_code_is_flagged_not_recoverable(self):
        s = json.loads((self.run / "summary.json").read_text())
        s["git"]["code_dirty"] = True
        w(self.run / "summary.json", json.dumps(s))
        m = self.build()
        self.assertFalse(m["code"]["code_exactly_recoverable"])
        res = {c: v for c, v, _ in R.verify(m, self.repo, self.data, cache=self.cache)}
        self.assertEqual(res["code provenance"], R.BAD)

    def test_missing_commit_gives_unknown(self):
        s = json.loads((self.run / "summary.json").read_text())
        s["git"]["commit"] = "0" * 40
        w(self.run / "summary.json", json.dumps(s))
        m = self.build()
        self.assertEqual(m["code"]["gogeta_code_sha256"], common.UNKNOWN)
        self.assertEqual(m["config"]["file_sha256_at_commit"], common.UNKNOWN)

    # ---------------------------------------------------------------- reproduce.verify
    def test_verify_match_then_detect_changes(self):
        m = self.build()
        res = {c: v for c, v, _ in R.verify(m, self.repo, self.data, cache=self.cache)}
        self.assertEqual(res["manifest integrity"], R.OK)
        self.assertEqual(res["code (gogeta/**/*.py)"], R.OK)
        self.assertEqual(res["config file"], R.OK)
        self.assertEqual(res["data train.bin"], R.OK)
        # edit code in the WORKING TREE (uncommitted): must be caught
        w(self.repo / "gogeta" / "model.py", "X = 2\n")
        w(self.repo / "configs" / "t.json", '{"experiment": "T", "x": 1}')
        (self.data / "datasets" / "toy" / "val.bin").write_bytes(b"corrupt")
        res = {c: v for c, v, _ in R.verify(m, self.repo, self.data, cache=self.cache)}
        self.assertEqual(res["code (gogeta/**/*.py)"], R.BAD)
        self.assertEqual(res["config file"], R.BAD)
        self.assertEqual(res["data val.bin"], R.BAD)
        self.assertEqual(res["data train.bin"], R.OK)

    def test_tampered_manifest_detected(self):
        m = self.build()
        m["metrics"]["val_loss"] = 0.5
        res = {c: v for c, v, _ in R.verify(m, self.repo, self.data, cache=self.cache)}
        self.assertEqual(res["manifest integrity"], R.BAD)

    # ---------------------------------------------------------------- compare
    def _new_run(self, name, val_loss, seed_cfg=None, steps=10):
        d = self.data / "runs" / name
        s = json.loads((self.run / "summary.json").read_text())
        s["name"] = name
        s["metrics"] = {**s["metrics"], "val_loss": val_loss, "steps": steps}
        if seed_cfg:
            s["config"]["train"]["seed"] = seed_cfg
        w(d / "summary.json", json.dumps(s))
        return name

    def test_compare_verdicts(self):
        m = self.build()
        tol = R.load_tolerances()
        old_data = R.DATA
        R.DATA = self.data
        try:
            ok = R.compare(m, self._new_run("N1", 1.003), tol, data=self.data)
            self.assertTrue(ok["verdict"].startswith("STATISTICAL"), ok["verdict"])
            bad = R.compare(m, self._new_run("N2", 1.02), tol, data=self.data)
            self.assertTrue(bad["verdict"].startswith("NOT REPRODUCED"), bad["verdict"])
            exact = R.compare(m, self._new_run("N3", 1.0, steps=11), tol, data=self.data)
            self.assertTrue(exact["verdict"].startswith("NOT REPRODUCED"), "steps are exact-match")
            other = R.compare(m, self._new_run("N4", 1.0, seed_cfg=7), tol, data=self.data)
            self.assertTrue(other["verdict"].startswith("NOT COMPARABLE"), other["verdict"])
            # bitwise: same checkpoint bytes
            ck = self.data / "checkpoints" / "N5"
            ck.mkdir(parents=True)
            (ck / "model.pt").write_bytes(b"weights")
            same = R.compare(m, self._new_run("N5", 1.0), tol, data=self.data)
            self.assertTrue(same["verdict"].startswith("BITWISE"), same["verdict"])
        finally:
            R.DATA = old_data

    def test_tolerance_uses_seed_sd_when_larger(self):
        m = self.build()
        tol = {**R.load_tolerances(), "seed_sd_val_loss": 0.01}
        old = R.DATA
        R.DATA = self.data
        try:
            r = R.compare(m, self._new_run("N6", 1.015), tol, data=self.data)
            self.assertAlmostEqual(r["val_loss_tolerance"], 0.02)
            self.assertTrue(r["verdict"].startswith("STATISTICAL"), r["verdict"])
        finally:
            R.DATA = old

    def test_tolerance_block_parses_in_repo_notes(self):
        t = R.load_tolerances()
        self.assertEqual(t["val_loss_abs_floor"], 0.004)
        self.assertIn("steps", t["exact"])


class LintFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="lint-test-"))
        self.r = self.tmp / "r"
        self.r.mkdir()
        sh(self.r, "git", "init", "-q")
        w(self.r / "experiments" / "claims_list.py", (REPO / "experiments" / "claims_list.py").read_text(encoding="utf-8"))
        w(self.r / "DEVLOG.md", "# devlog\n")
        w(self.r / "experiments" / "claims.md", "| ID | Claim | Source | Status | Evidence |\n|---|---|---|---|---|\n")
        w(self.r / "experiments" / "checks" / "PREDICTIONS.md", "# predictions\n")
        w(self.r / "experiments" / "queue.json", json.dumps({"items": []}))
        w(self.r / "experiments" / "experiments.json", "[]")
        commit(self.r, "base", "v0.0.1")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lint(self):
        return lint_science.Lint(self.r, self.tmp / "nodata")

    def levels(self, rep, check):
        return {f["subject"]: f["level"] for f in rep["findings"] if f["check"] == check}

    def test_done_experiment_checks(self):
        exps = [{"id": "X1", "status": "done", "version": "v0.0.1", "results": {"val_loss": 1.0, "run": "R1"}},
                {"id": "X2", "status": "done", "version": "v9.9.9", "results": {}},
                {"id": "X3", "status": "planned"}]
        w(self.r / "experiments" / "experiments.json", json.dumps(exps))
        w(self.r / "experiments" / "cards" / "X1.md", "card")
        w(self.r / "DEVLOG.md", "# d\n### X1: thing\n")
        w(self.r / "experiments" / "claims.md",
          "| ID | Claim | Source | Status | Evidence |\n|---|---|---|---|---|\n| C1 | X1 beats X0 | us | `PARTIAL` | n |\n")
        l = self.lint()
        l.check_experiments()
        rep = l.report()
        lv = self.levels(rep, "E")
        self.assertEqual(lv["X1"], "fail")  # only the manifest is missing
        self.assertIn("X1", lv)
        self.assertNotIn("X3", lv)
        msgs = {f["subject"]: f["message"] for f in rep["findings"]}
        self.assertEqual(msgs["X1"], "no repro manifest for run R1")
        self.assertIn("no card", msgs["X2"])
        self.assertIn("not a git tag", msgs["X2"])
        self.assertIn("empty results", msgs["X2"])
        self.assertIn("no ledger row", msgs["X2"])

    def test_verified_claim_evidence(self):
        w(self.r / "experiments" / "claims.md",
          "| ID | Claim | Source | Status | Evidence |\n|---|---|---|---|---|\n"
          "| C1 | good | us | `VERIFIED` | `experiments/checks/results/K1.json` |\n"
          "| C2 | missing | us | `VERIFIED` | `experiments/checks/results/NOPE.json` |\n"
          "| C3 | prose | us | `VERIFIED` | measured in the notebook |\n"
          "| C4 | later demoted | us | `VERIFIED` | `nope.json` |\n"
          "| C4 | later demoted | us | `REFUTED` | x |\n")
        w(self.r / "experiments" / "checks" / "results" / "K1.json", "{}")
        commit(self.r, "x")
        l = self.lint()
        l.check_claims()
        lv = self.levels(l.report(), "C")
        self.assertEqual(lv, {"C1": "pass", "C2": "fail", "C3": "warn"})

    def test_preregistration_order(self):
        res = self.r / "experiments" / "checks" / "results"
        # K1: prediction first, results later -> pass
        w(self.r / "experiments" / "checks" / "PREDICTIONS.md", "# p\n## K1: a thing\nPrediction: x\n")
        commit(self.r, "register K1")
        w(res / "K1.json", json.dumps({"check": "K1"}))
        commit(self.r, "results K1")
        # K2: prediction and results in the same commit -> fail
        w(self.r / "experiments" / "checks" / "PREDICTIONS.md", "# p\n## K1: a thing\nPrediction: x\n## K2: b\n")
        w(res / "K2.json", json.dumps({"check": "K2"}))
        commit(self.r, "K2 both")
        # K3: results, prediction later -> fail
        w(res / "K3.json", json.dumps({"check": "K3"}))
        commit(self.r, "results K3")
        w(self.r / "experiments" / "checks" / "PREDICTIONS.md", "# p\n## K1: a\n## K2: b\n## K3: late\n")
        commit(self.r, "late prediction K3")
        # K4: results with no prediction at all -> fail; K5 uncommitted -> warn
        w(res / "K4.json", json.dumps({"check": "K4"}))
        commit(self.r, "results K4")
        w(res / "K5.json", json.dumps({"check": "K5"}))
        l = self.lint()
        l.check_preregistration()
        lv = self.levels(l.report(), "P")
        self.assertEqual(lv, {"K1": "pass", "K2": "fail", "K3": "fail", "K4": "fail", "K5": "warn"})

    def test_queue_done_needs_version(self):
        q = {"items": [{"id": "Q1", "status": "done", "version": "v0.0.1"},
                       {"id": "Q2", "status": "done", "note": "shipped in v0.0.1"},
                       {"id": "Q3", "status": "done", "note": "no version here"},
                       {"id": "Q4", "status": "done", "version": "v7.7.7"},
                       {"id": "Q5", "status": "queued"}]}
        w(self.r / "experiments" / "queue.json", json.dumps(q))
        l = self.lint()
        l.check_queue()
        lv = self.levels(l.report(), "Q")
        self.assertEqual(lv, {"Q1": "pass", "Q2": "pass", "Q3": "fail", "Q4": "fail"})

    def test_lint_never_modifies_repo(self):
        before = sh(self.r, "git", "status", "--porcelain")
        self.lint().run()
        self.assertEqual(before, sh(self.r, "git", "status", "--porcelain"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
