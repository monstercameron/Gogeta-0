"""Registry consistency linter. REPORTS, never fixes. Read-only on the repo (git log/show/tag only).

    python3 repro/lint_science.py [--root REPO] [--out report.json] [--data $GOGETA_DATA] [--quiet]

Checks (each finding is pass / warn / fail):
  E  every experiment in experiments/experiments.json with status "done" has: a card (experiments/cards/<ID>.md), a
     DEVLOG `### <ID>` block, a version that is an existing git tag, non-empty results, at least one ledger row
     (experiments/claims.md) that mentions it, and a manifest in repro/manifests for results.run.
  C  every ledger claim whose LATEST status is VERIFIED cites evidence that exists: a file path, a check/card ID, or a
     run folder. No citable artifact at all (prose only) is a warn; a cited artifact that does not exist is a fail.
  P  every check result (experiments/checks/results/*.json) was pre-registered: its PREDICTION section in
     experiments/checks/PREDICTIONS.md (or a side-quests.md row naming a prediction) was committed in a commit
     strictly EARLIER than the commit that first added the results file. Same commit, no section, or an uncommitted
     results file are findings.
  Q  every queue item with status "done" references a version (a `version` field or a vX.Y.Z in its notes) that is
     an existing git tag.
Exit code: 0 no fails (warns allowed), 1 at least one fail, 2 usage error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

sys.dont_write_bytecode = True  # importing claims_list must not write __pycache__ into the audited repo
SELF = pathlib.Path(__file__).resolve()
ART_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:json|md|py|sh|txt|jsonl|html|patch))\b")
RUN_RE = re.compile(r"\b((?:E|K|SQ|smoke)\d*-\d{8}-\d{6}-s\d+)\b")
CARD_RE = re.compile(r"\b(K\d+(?:-[Pp]arity)?|SQ\d+|E\d+)\b")


def git(root, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", errors="replace").strip() if r.returncode == 0 else ""


class Lint:
    def __init__(self, root, data=None):
        self.root = pathlib.Path(root)
        self.data = pathlib.Path(data) if data else pathlib.Path.home() / "fusiondance-data"
        self.findings: list[dict] = []
        self._order = None
        self._names = None
        self.tags = set(git(self.root, "tag", "-l").split())

    def add(self, check, subject, level, msg):
        self.findings.append({"check": check, "subject": subject, "level": level, "message": msg})

    # -- git helpers
    def order(self) -> dict:
        if self._order is None:
            self._order = {c: i for i, c in enumerate(git(self.root, "rev-list", "--reverse", "HEAD").split())}
        return self._order

    def first_commit(self, *args) -> str | None:
        out = git(self.root, "log", "--reverse", "--format=%H", *args).split()
        return out[0] if out else None

    # -- E: experiments
    def check_experiments(self):
        reg = self.root / "experiments" / "experiments.json"
        exps = json.loads(reg.read_text(encoding="utf-8"))
        devlog = (self.root / "DEVLOG.md").read_text(encoding="utf-8") if (self.root / "DEVLOG.md").exists() else ""
        ledger = (self.root / "experiments" / "claims.md").read_text(encoding="utf-8") \
            if (self.root / "experiments" / "claims.md").exists() else ""
        ledger_rows = [l for l in ledger.splitlines() if l.startswith("|")]
        for e in exps:
            if e.get("status") != "done":
                continue
            i, probs = e["id"], []
            if not (self.root / "experiments" / "cards" / f"{i}.md").exists():
                probs.append("no card experiments/cards/%s.md" % i)
            if not re.search(rf"^###\s+{re.escape(i)}\b", devlog, re.M):
                probs.append(f"no DEVLOG '### {i}' block")
            v = e.get("version")
            if not v:
                probs.append("no version recorded")
            elif v not in self.tags:
                probs.append(f"version {v} is not a git tag")
            res = e.get("results")
            if not res:
                probs.append("empty results")
            if not any(re.search(rf"(?<![A-Za-z0-9]){re.escape(i)}(?![0-9])", l) for l in ledger_rows):
                probs.append("no ledger row (claims.md) mentions it")
            run = (res or {}).get("run")
            if run and not (self.root / "repro" / "manifests" / f"{run}.json").exists():
                probs.append(f"no repro manifest for run {run}")
            self.add("E", i, "fail" if probs else "pass", "; ".join(probs) if probs else "card, devlog block, tag, results, ledger row, manifest all present")

    def _exists(self, a: str) -> bool:
        if (self.root / a).exists() or (self.root / "experiments" / "checks" / a).exists():  # checks/ = common relative cite
            return True
        if "/" in a:
            return False
        if self._names is None:   # basename index from git (tracked + untracked, not ignored): fast on /mnt/c
            self._names = {n.rsplit("/", 1)[-1] for n in git(self.root, "ls-files", "-co", "--exclude-standard").splitlines()}
        return a in self._names

    # -- C: VERIFIED claims
    def check_claims(self):
        spec = importlib.util.spec_from_file_location("claims_list", self.root / "experiments" / "claims_list.py")
        cl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cl)
        lines = (self.root / "experiments" / "claims.md").read_text(encoding="utf-8").splitlines()
        claims = cl.parse()
        for cid, c in sorted(claims.items(), key=lambda kv: int(kv[0][1:])):
            h = c["history"][-1]
            if h["status"] != "VERIFIED":
                continue
            row = lines[h["line"] - 1]
            # evidence = the row text after the status cell
            ev = row.split("VERIFIED", 1)[-1]
            arts, missing, found, unver = set(ART_RE.findall(ev)), [], [], []
            for a in sorted(arts):
                a = a.strip(".")
                if "*" in a:
                    continue
                ok = self._exists(a)
                (found if ok else missing).append(a)
            for r in sorted(set(RUN_RE.findall(ev))):
                if (self.data / "runs" / r).is_dir() or (self.root / "repro" / "manifests" / f"{r}.json").exists():
                    found.append(r)
                elif not (self.data / "runs").is_dir():
                    unver.append(r)
                else:
                    missing.append(r)
            for cid_ in sorted(set(CARD_RE.findall(ev))):
                card = self.root / "experiments" / "cards" / f"{cid_.replace('-Parity', '-parity')}.md"
                if card.exists():
                    found.append(f"card:{cid_}")
            if missing:
                self.add("C", cid, "fail", f"VERIFIED but cites missing artifact(s): {', '.join(missing)}")
            elif found:
                self.add("C", cid, "pass", f"{len(found)} cited artifact(s) exist")
            elif unver:
                self.add("C", cid, "warn", f"cites run(s) {', '.join(unver)}; run folders not visible from here")
            else:
                self.add("C", cid, "warn", "VERIFIED with prose-only evidence (no file, card or run cited)")

    # -- P: pre-registration
    def check_preregistration(self):
        pred = "experiments/checks/PREDICTIONS.md"
        resdir = self.root / "experiments" / "checks" / "results"
        order = self.order()
        for rp in sorted(resdir.glob("*.json")):
            rel = rp.relative_to(self.root).as_posix()
            try:
                cid = json.loads(rp.read_text(encoding="utf-8"))["check"]
            except Exception as ex:
                self.add("P", rp.name, "fail", f"unreadable results JSON: {ex}")
                continue
            res_c = self.first_commit("--diff-filter=A", "--", rel)
            if not res_c:
                self.add("P", cid, "warn", f"{rel} is not committed: pre-registration order cannot be verified yet")
                continue
            esc = re.escape(cid)
            cand = []
            c1 = self.first_commit("-G", rf"^#{{2,3}} {esc}([^A-Za-z0-9]|$)", "--", pred)
            if c1:
                cand.append(("PREDICTIONS.md section", c1))
            c2 = self.first_commit("-G", rf"{esc}\b.*[Pp]redict", "--", "experiments/side-quests.md")
            if c2:
                cand.append(("side-quests.md row naming a prediction", c2))
            if not cand:
                self.add("P", cid, "fail", f"no PREDICTION section for {cid} found in {pred} or side-quests.md history "
                         f"(results first committed {res_c[:7]})")
                continue
            src, c = min(cand, key=lambda x: order.get(x[1], 1 << 30))
            if order.get(c, 1 << 30) < order.get(res_c, -1):
                self.add("P", cid, "pass", f"{src} committed {c[:7]} before results {res_c[:7]}")
            elif c == res_c:
                self.add("P", cid, "fail", f"prediction and results first appear in the SAME commit {c[:7]}: not pre-registered")
            else:
                self.add("P", cid, "fail", f"earliest prediction ({src}, {c[:7]}) is AFTER the results commit {res_c[:7]}")

    # -- Q: queue
    def check_queue(self):
        qp = self.root / "experiments" / "queue.json"
        q = json.loads(qp.read_text(encoding="utf-8"))
        for it in q.get("items", []):
            if it.get("status") != "done":
                continue
            texts = [str(it.get(k, "")) for k in ("version", "note", "last_note")] + \
                    [str(h.get("note", "")) for h in it.get("history", []) if isinstance(h, dict)]
            vs = set(re.findall(r"\bv\d+\.\d+\.\d+\b", " ".join(texts)))
            ok = [v for v in vs if v in self.tags]
            if ok:
                self.add("Q", it["id"], "pass", f"references {', '.join(sorted(ok))}")
            elif vs:
                self.add("Q", it["id"], "fail", f"references {', '.join(sorted(vs))}, not an existing tag")
            else:
                self.add("Q", it["id"], "fail", "done but references no version (no `version` field, no vX.Y.Z in notes)")

    def run(self):
        for fn in (self.check_experiments, self.check_claims, self.check_preregistration, self.check_queue):
            try:
                fn()
            except Exception as ex:  # a linter bug must be visible, not silent
                self.add("LINT", fn.__name__, "fail", f"linter error: {type(ex).__name__}: {ex}")
        return self.report()

    def report(self) -> dict:
        s = {}
        for f in self.findings:
            s.setdefault(f["check"], {"pass": 0, "warn": 0, "fail": 0})[f["level"]] += 1
        return {"tool": "repro/lint_science.py", "root": str(self.root), "head": git(self.root, "rev-parse", "HEAD"),
                "summary": s, "total_fail": sum(v["fail"] for v in s.values()),
                "total_warn": sum(v["warn"] for v in s.values()), "findings": self.findings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SELF.parent.parent))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    a = ap.parse_args(argv)
    rep = Lint(a.root, a.data).run()
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    names = {"E": "done experiments", "C": "VERIFIED claims", "P": "pre-registration", "Q": "queue done items", "LINT": "linter errors"}
    print("== lint_science summary (reports only; nothing was changed)")
    for k, v in rep["summary"].items():
        print(f"  {k:4s} {names.get(k, k):20s} pass {v['pass']:3d}  warn {v['warn']:3d}  FAIL {v['fail']:3d}")
    if not a.quiet:
        for f in rep["findings"]:
            if f["level"] != "pass":
                print(f"  [{f['level'].upper():4s}] {f['check']} {f['subject']}: {f['message']}")
    print(f"total: {rep['total_fail']} fail, {rep['total_warn']} warn")
    return 1 if rep["total_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
