"""Tests for the v2 claims ledger (experiments/claims_list.py + experiments/claims/migrate_v1.py). Stdlib unittest.

    python -m unittest experiments/claims/test_ledger.py -v      (from the repo root; CPU only, ~10 s)

The real-repo tests only READ (claims.md, the ledger, git). Everything that writes runs in a temporary git repo.
"""
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import claims_list as CL  # noqa: E402
import migrate_v1 as M  # noqa: E402

V = "v0.9.1"
MINI = """# Claims ledger

Intro text.

## Verified or measured here

| ID | Claim | Source | Status | Evidence |
|---|---|---|---|---|
| C1 | Kernel A matches the reference on this GPU | our code | `VERIFIED` | `experiments/checks/results/K1.json`: diff 0 |
| C2 | Model B beats model C at 14M params | our hypothesis | `PARTIAL` (single seed) | `experiments/cards/E1.md`; run `E1-20260101-000000-s0` |
| C3 | Paper D says ternary is free | Paper D, arXiv 1234.5678 | `UNVERIFIED` | **K4**: tiny check |

## Status updates

| ID | Claim | Source | Status | Evidence / notes |
|---|---|---|---|---|
| C2 | Model B beats C | later | `REFUTED` (2 seeds) | `experiments/checks/results/K1.json` second look, v0.9.0 |
| C3 | Paper D | later | `UNVERIFIED` (still) | nothing yet |
"""


ANN = {"C2": {"dependencies": [{"claim": "C1", **M.Q("Model B beats model C", 10)}]}}


def git(root, *a):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                           "core.autocrlf=false", *a], capture_output=True, text=True, check=True).stdout


class RealRepo(unittest.TestCase):
    """Migration round trip against the repo's own claims.md (read-only)."""

    def test_every_row_is_an_event_and_latest_status_matches(self):
        legacy = CL.parse_legacy(REPO / "experiments/claims.md")
        claims, counts, _, _ = M.build(REPO, V, dt.datetime.now().astimezone())
        self.assertEqual(set(claims), set(legacy))
        self.assertEqual(counts["rows"], sum(len(c["history"]) for c in legacy.values()))
        for cid, c in legacy.items():
            mig = [e for e in claims[cid] if (e.get("provenance") or {}).get("line")]
            self.assertEqual([e["provenance"]["line"] for e in mig], [h["line"] for h in c["history"]], cid)
            self.assertEqual([e["status"] for e in mig], [h["status"] for h in c["history"]], cid)
            self.assertEqual(CL.Store.derive({"events": claims[cid]})["status"], c["history"][-1]["status"], cid)
            self.assertEqual(CL.clean(claims[cid][0]["definition"]["proposition"]), c["text"], cid)

    def test_committed_ledger_matches_claims_md_and_parse_shape(self):
        st = CL.Store(REPO)
        if not st.exists():
            self.skipTest("no v2 ledger in this checkout yet")
        probs, n_ids, n_rows, n_mig = M.verify(REPO)
        self.assertEqual(probs, [])
        self.assertEqual(n_rows, n_mig)
        legacy, new = CL.parse_legacy(st.src), st.parse()
        keys = ["status", "qual", "section", "line", "evidence", "raw_evidence"]
        for cid, c in legacy.items():
            mig = [h for h in new[cid]["history"] if h["line"] < 10 ** 6]
            self.assertEqual([[h[k] for k in keys] for h in c["history"]], [[h[k] for k in keys] for h in mig], cid)
            self.assertEqual(c["text"], new[cid]["text"])

    def test_annotations_are_verbatim(self):
        cache = {}
        for cid, ann in M.ANNOTATIONS.items():
            for a in list(ann.get("scope", {}).values()) + ann.get("refutation", []) + ann.get("limitations", []) \
                    + ann.get("dependencies", []):
                M.resolve(a, REPO, cache)   # raises SystemExit if not found verbatim


class TempLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="ledger-test-"))
        r = self.root = self.tmp / "repo"
        (r / "experiments/checks/results").mkdir(parents=True)
        (r / "experiments/cards").mkdir(parents=True)
        (r / "experiments/checks/results/K1.json").write_text("{}", encoding="utf-8")
        (r / "experiments/cards/E1.md").write_text("card", encoding="utf-8")
        (r / "experiments/claims.md").write_text(MINI, encoding="utf-8", newline="\n")
        git(r, "init", "-q")
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", "base")
        git(r, "tag", "-a", "v0.9.0", "-m", "v0.9.0")
        M.migrate(r, V, annotations=ANN, quiet=True)
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", "migrate")
        self.st = CL.Store(r)
        self._saved = CL._STORE
        CL._STORE = self.st

    def tearDown(self):
        CL._STORE = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fails(self, rule=None):
        return [f for f in self.st.lint() if f["level"] == "fail" and (rule is None or f["rule"] == rule)]

    # --- migration
    def test_mini_migration(self):
        legacy = CL.parse_legacy(self.st.src)
        claims = self.st.load()
        self.assertEqual(sum(len(c["history"]) for c in legacy.values()),
                         sum(1 for c in claims.values() for e in c["events"] if e.get("provenance")))
        self.assertEqual(self.st.derive(claims["C2"])["status"], "REFUTED")
        self.assertEqual([e["event"] for e in claims["C2"]["events"]], ["created", "status_change"])
        self.assertEqual([e["event"] for e in claims["C3"]["events"]], ["created", "evidence_added"])
        self.assertEqual(claims["C3"]["definition"]["type"], "external finding")
        self.assertEqual(claims["C1"]["definition"]["scope"]["dataset"], CL.TBD)
        self.assertEqual(claims["C2"]["events"][0]["version"], "v0.9.0")   # from git blame + tag
        self.assertEqual(self.fails(), [])
        self.assertEqual(self.st.stale(), [])
        p = CL.parse()
        self.assertEqual(p["C2"]["history"][-1]["status"], "REFUTED")
        self.assertEqual(p["C2"]["history"][-1]["line"], 17)

    def test_rerun_refused_after_post_migration_event(self):
        M.migrate(self.root, V, annotations=ANN, quiet=True)   # idempotent while only migrated events exist
        self.st.add_event("C3", V, status="PARTIAL", reason="tiny check", evidence=["experiments/cards/E1.md"])
        with self.assertRaises(SystemExit):
            M.migrate(self.root, V, annotations=ANN, quiet=True)

    def test_freeze_and_frozen_lint(self):
        shutil.rmtree(self.st.ledger)
        counts = M.migrate(self.root, V, freeze=True, annotations=ANN, quiet=True)
        self.assertTrue(self.st.src.read_text(encoding="utf-8").startswith(M.FROZEN_MARK))
        self.assertEqual(CL.parse_legacy(self.st.src)["C2"]["history"][-1]["line"], 17)   # line numbers kept
        self.assertEqual(counts["rows"], 5)
        self.assertEqual(self.fails("frozen"), [])
        with open(self.st.src, "a", encoding="utf-8") as fh:
            fh.write("| C9 | sneaky row | x | `VERIFIED` | v0.1.0 |\n")
        self.assertTrue(self.fails("frozen"))
        with self.assertRaises(SystemExit):
            M.migrate(self.root, V, annotations=ANN, quiet=True)

    # --- immutability / append-only
    def test_mutated_definition_fails(self):
        p = self.st.path("C1")
        lines = p.read_text(encoding="utf-8").splitlines()
        e = json.loads(lines[0])
        e["definition"]["proposition"] = "Kernel A matches the reference on EVERY GPU"
        lines[0] = CL.dump_line(e)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        f = self.fails("immutability")
        self.assertEqual(len(f), 1)
        self.assertIn("erratum", f[0]["msg"])

    def test_edited_or_removed_event_fails(self):
        p = self.st.path("C2")
        lines = p.read_text(encoding="utf-8").splitlines()
        e = json.loads(lines[1])
        e["status"] = "VERIFIED"
        p.write_text(lines[0] + "\n" + CL.dump_line(e) + "\n", encoding="utf-8")
        self.assertTrue(self.fails("append-only"))
        p.write_text(lines[0] + "\n", encoding="utf-8")                 # removal
        self.assertTrue(self.fails("append-only"))
        self.st.path("C3").unlink()                                       # deleted claim
        self.assertTrue([f for f in self.fails("append-only") if f["id"] == "C3"])

    def test_appending_is_fine_and_erratum_rules(self):
        self.st.add_event("C1", V, evidence=["experiments/cards/E1.md"], reason="re-run")
        self.st.erratum("C1", "proposition", "Kernel A matches the reference on this GPU",
                        "Kernel A matches the reference on this GPU.", "typo: missing full stop", V)
        self.assertEqual(self.fails(), [])
        self.assertTrue(self.st.claim("C1")["definition"]["proposition"].endswith("GPU."))
        with self.assertRaises(CL.LedgerError):   # not a typo: needs a new claim with supersedes
            self.st.erratum("C1", "proposition", "Kernel A matches the reference on this GPU.",
                            "Kernel Z is faster than everything on all hardware", "rewrite", V)
        with self.assertRaises(CL.LedgerError):   # old text must match the current definition
            self.st.erratum("C1", "proposition", "wrong old", "wrong olde", "typo", V)
        self.st.erratum("C3", "scope.dataset", CL.TBD, "tiny synthetic", "fill", V, kind="tbd_fill")
        self.assertEqual(self.st.claim("C3")["definition"]["scope"]["dataset"], "tiny synthetic")
        with self.assertRaises(CL.LedgerError):
            self.st.erratum("C3", "scope.dataset", "tiny synthetic", "other", "fill", V, kind="tbd_fill")

    # --- dependencies
    def test_dependency_cycle_and_unknown(self):
        e = json.loads(self.st.path("C1").read_text(encoding="utf-8").splitlines()[0])
        e["definition"]["dependencies"] = [{"claim": "C2"}]   # C2 already depends on C1
        self.st.path("C1").write_text(CL.dump_line(e) + "\n", encoding="utf-8")
        self.assertTrue([f for f in self.fails("dependencies") if "cycle" in f["msg"]])
        self.assertEqual(CL.find_cycles({"C1": ["C2"], "C2": ["C3"], "C3": ["C1"], "C4": ["C1"]}), [["C1", "C2", "C3"]])
        with self.assertRaises(CL.LedgerError):
            self.st.new_claim("x depends on nothing real", "hypothesis", "us", V, dependencies=[{"claim": "C99"}])

    def test_refuted_dependency_flags_dependents(self):
        cid = self.st.new_claim("D builds on C1", "hypothesis", "us", V, dependencies=[{"claim": "C1"}])
        self.assertNotIn(cid, self.st.review_flags())
        self.st.add_event("C1", V, status="REFUTED", reason="rerun failed", evidence=["experiments/checks/results/K1.json"])
        self.assertIn(cid, self.st.review_flags())
        self.assertTrue([f for f in self.st.lint() if f["rule"] == "review" and f["id"] == cid])
        self.st.add_event(cid, V, limitations=["rests on refuted C1"], reason="reviewed")
        self.assertNotIn(cid, self.st.review_flags())

    # --- evidence pointer rule
    def test_evidence_pointer_rule(self):
        with self.assertRaises(CL.LedgerError):   # settled status without any pointer
            self.st.add_event("C3", V, status="PARTIAL", reason="no pointer")
        with self.assertRaises(CL.LedgerError):   # pointer to a file that does not exist
            self.st.add_event("C3", V, status="PARTIAL", reason="x", evidence=["experiments/checks/results/NOPE.json"])
        with self.assertRaises(CL.LedgerError):
            self.st.add_event("C3", V, status="PARTIAL", reason="x", evidence=["just prose"])
        self.st.add_event("C3", V, status="PARTIAL", reason="run only", evidence=["K4-20260101-000000-s0"])
        self.assertEqual(self.fails(), [])
        # a hand-made file that violates the rule is caught by the lint
        bad = {"seq": 1, "ts": CL.now_iso(), "version": V, "actor": "main-agent", "event": "created",
               "status": "VERIFIED", "qual": "", "reason": "x", "evidence": [], "limitations": [],
               "evidence_class": "deterministic", "verified_at": {"commit": "abc1234", "date": "2026-01-01"},
               "definition": {"id": "C50", "proposition": "p", "scope": {k: CL.TBD for k in CL.SCOPE_FIELDS},
                              "type": "empirical", "refutation_criteria": CL.TBD, "dependencies": CL.TBD,
                              "created_version": V, "origin": {"source": "us"}}}
        self.st.path("C50").write_text(CL.dump_line(bad) + "\n", encoding="utf-8")
        self.assertTrue([f for f in self.fails("evidence") if f["id"] == "C50"])

    def test_statistics_policy_and_last_verified(self):
        with self.assertRaises(CL.LedgerError):   # training comparison, 1 seed
            self.st.add_event("C3", V, status="VERIFIED", reason="x", evidence=["experiments/cards/E1.md"],
                              evidence_class="training", n_seeds=1)
        with self.assertRaises(CL.LedgerError):   # no evidence class at all
            self.st.add_event("C3", V, status="VERIFIED", reason="x", evidence=["experiments/cards/E1.md"])
        e = self.st.add_event("C3", V, status="VERIFIED", reason="3 seeds, CI excludes 0",
                              evidence=["experiments/cards/E1.md"], evidence_class="training", n_seeds=3)
        self.assertEqual(e["verified_at"]["date"], dt.date.today().isoformat())
        self.assertTrue(self.st.derive(self.st.claim("C3"))["last_verified"]["commit"])
        self.assertEqual(self.fails(), [])

    # --- API: new-claim, add-event, supersede, audit verdicts, CLI
    def test_new_claim_add_event_supersede_cli(self):
        cid = self.st.new_claim("E beats F at d=128", "empirical", "SQ9 result", V,
                                scope={"model_size": "d=128"}, refutation="F >= E over 3 seeds")
        self.assertEqual(cid, "C4")
        d = self.st.claim(cid)["definition"]
        self.assertEqual(d["scope"]["model_size"], "d=128")
        self.assertEqual(d["scope"]["dataset"], CL.TBD)
        self.assertEqual(d["dependencies"], CL.TBD)
        rc = CL.main(["add-event", cid, "--status", "PARTIAL", "--reason", "one seed", "--evidence",
                      "experiments/cards/E1.md,v0.9.0", "--version", V, "--limitation", "single seed"])
        self.assertEqual(rc, 0)
        s = self.st.summary(cid)
        self.assertEqual((s["status"], s["n_events"], len(s["limitations"])), ("PARTIAL", 2, 1))
        self.assertEqual(CL.main(["add-event", cid, "--status", "PARTIAL", "--reason", "x", "--evidence",
                                  "nope/missing.json", "--version", V, "--no-render"]), 3)
        h = CL.parse()[cid]["history"]
        self.assertEqual([x["status"] for x in h], ["UNVERIFIED", "PARTIAL"])
        self.assertGreater(h[-1]["line"], 10 ** 6)   # synthetic, time-ordered line for new events
        new = self.st.new_claim("E beats F at d=128 and d=256", "empirical", "SQ9 rerun", V, supersedes=cid,
                                reason="scope widened: a changed proposition gets a new ID")
        self.assertEqual(self.st.derive(self.st.claim(cid))["status"], "SUPERSEDED")
        self.assertEqual(self.st.claim(new)["definition"]["supersedes"], cid)
        self.assertEqual(self.fails(), [])
        self.st.audit_verdict("C1", "weak", "missing seed list", V, batch_file="experiments/cards/E1.md")
        a = self.st.derive(self.st.claim("C1"))["audits"]
        self.assertEqual(a[-1]["verdict"], "WEAK")
        self.assertEqual(self.st.derive(self.st.claim("C1"))["status"], "VERIFIED")   # a verdict changes nothing
        self.assertEqual(CL.main(["--check"]), 1)   # library writes don't regenerate views
        self.st.write_outputs()
        self.assertEqual(CL.main(["--check"]), 0)
        self.assertEqual(CL.main(["lint"]), 0)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "more")
        self.assertEqual(self.fails(), [])

    def test_concurrent_appends_get_distinct_seqs(self):
        import threading
        errs = []

        def w(i):
            try:
                self.st.add_event("C3", V, limitations=[f"note {i}"], reason="n")
            except Exception as ex:  # noqa: BLE001
                errs.append(ex)
        ts = [threading.Thread(target=w, args=(i,)) for i in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])
        seqs = [e["seq"] for e in self.st.claim("C3")["events"]]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))
        self.assertEqual(self.fails(), [])


if __name__ == "__main__":
    unittest.main()
