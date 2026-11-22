"""Claims ledger v2: immutable claim definitions + append-only evidence events. Stdlib only.

Source of truth: experiments/claims/ledger/C<n>.jsonl, one file per claim, one JSON object per line. Line 1 is the
`created` event, which carries the claim's immutable definition; every later line is one appended event. Current
status, evidence, limitations and "last verified" are DERIVED from the events, never hand-edited. Schema, status
definitions, event types and lint rules: experiments/claims/README.md.

experiments/claims.md is the FROZEN v1 table ledger (migrated by experiments/claims/migrate_v1.py). parse() still
returns its old shape ({id: {"id", "text", "history": [{"status", "qual", "section", "line", "evidence",
"raw_evidence"}]}}), now built from the events, and falls back to reading claims.md when no ledger exists.

    python experiments/claims_list.py                      # regenerate claims-status.md + claims/views/, then lint
    python experiments/claims_list.py --check              # no writes: exit 1 if a view is stale or the lint fails
    python experiments/claims_list.py lint [--base REF]    # lint only (immutability/append-only vs REF, default HEAD)
    python experiments/claims_list.py show C5 [--json]
    python experiments/claims_list.py add-event C5 --status PARTIAL --reason "..." \
        --evidence experiments/checks/results/K9.json --version v0.4.45 [--actor main-agent] [--limitation "..."]
    python experiments/claims_list.py new-claim --proposition "..." --type empirical --source "..." --version v0.4.45 \
        [--id C150] [--scope-dataset ...] [--refutation "..."] [--depends C5,C9] [--supersedes C5 --reason "..."]
    python experiments/claims_list.py erratum C5 --field proposition --old "..." --new "..." --reason "..." --version ...
    python experiments/claims_list.py audit-verdict C5 --verdict WEAK --note "..." --batch-file <audit.md> --version ...
Exit codes: 0 ok; 1 lint failure (--check, lint); 2 lint failure after regenerating (default mode); 3 refused write.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import glob as _glob
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments/claims.md"            # v1 ledger: frozen archive once migrated
OUT = ROOT / "experiments/claims-status.md"     # generated current list (same path as v1)

STATUSES = ["VERIFIED", "PARTIAL", "REFUTED", "WRONG FOR OUR CODE", "INCONCLUSIVE", "UNVERIFIED"]  # v1 order
ALL_STATUSES = STATUSES + ["SUPERSEDED"]
SETTLED = {"VERIFIED", "PARTIAL", "REFUTED", "WRONG FOR OUR CODE", "INCONCLUSIVE"}
NEGATIVE = {"REFUTED", "WRONG FOR OUR CODE"}
TYPES = ["empirical", "theoretical", "implementation", "external finding", "hypothesis"]
EVENT_TYPES = ["created", "evidence_added", "status_change", "limitation_added", "audit_verdict", "erratum",
               "superseded"]
STATUS_EVENTS = {"created", "status_change", "evidence_added", "superseded"}   # events that may carry a status
VERDICTS = ["SOLID", "WEAK", "WRONG", "NEEDS-EVIDENCE", "UNCHECKABLE"]
SCOPE_FIELDS = ["dataset", "architecture", "model_size", "precision", "hardware"]
DEF_FIELDS = ["id", "proposition", "scope", "type", "refutation_criteria", "dependencies", "created_version",
              "origin"]
ERRATUM_FIELDS = {"proposition", "refutation_criteria", "type", "dependencies"} | {f"scope.{f}" for f in SCOPE_FIELDS}
EVIDENCE_CLASSES = ["deterministic", "training", "benchmark", "timing"]   # notes/statistics-policy.md section 1
TBD = "TBD (not stated at creation)"
ACTOR_RE = re.compile(r"^(main-agent|human|migration|subagent(:[\w.-]+)?|codex-auditor(:[\w.-]+)?)$")
VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
ID_FILE_RE = re.compile(r"^C(\d+)\.jsonl$")
TYPO_RATIO = 0.9   # an erratum to the proposition must be this similar (difflib) to count as a typo fix


class LedgerError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- v1 (claims.md)
ID_RE = re.compile(r"^\|\s*(?:\*\*)?(C\d+)\b(?:\s*\(new\))?(?:\*\*)?\s*([^|]*)\|(.*)$")
ST_RE = re.compile(r"`(VERIFIED|PARTIAL|REFUTED|WRONG FOR OUR CODE|INCONCLUSIVE|UNVERIFIED)([^`]*)`")


def clean(s, n=None):
    s = re.sub(r"[`*]", "", s).strip()
    return (s[: n - 1] + "…") if n and len(s) > n else s


def parse_legacy(path=None):
    """The v1 parser of the table ledger experiments/claims.md (latest row per ID wins)."""
    claims, section = {}, ""
    for i, line in enumerate(pathlib.Path(path or SRC).read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = ID_RE.match(line)
        if not m:
            continue
        cid, inline, rest = m.group(1), m.group(2).strip(), m.group(3)
        cells = [c.strip() for c in rest.split("|")]
        st = ST_RE.search(rest)
        if not st:
            continue  # e.g. the "proposed checks" table, which maps checks to claims
        text = inline or (cells[0] if cells else "")
        evidence = cells[-2] if len(cells) >= 2 and cells[-1] == "" else (cells[-1] if cells else "")
        c = claims.setdefault(cid, {"id": cid, "text": clean(text), "history": []})
        if inline and not c["text"]:
            c["text"] = clean(inline)
        c["history"].append({"status": st.group(1), "qual": st.group(2).strip(" ()"), "section": section,
                             "line": i, "evidence": clean(evidence, 220), "raw_evidence": evidence})
    return claims


# v1 rule (AGENTS 5): every settled claim's latest row cites a concrete pointer a reader can open.
PTR_RE = re.compile(r"(experiments|evals|notes|repro|gogeta|tools|tests|post_training|refs|docs|cockpit)/\S+"
                    r"|\b(?:E|K|SQ)\d+[A-Za-z0-9-]*-\d{8}-\d{6}-s\d+\b|v\d+\.\d+\.\d+")


def lint(claims):
    """v1-compatible pointer lint on parse()-shaped data: [(id, status, line)] of settled claims whose latest
    history entry has no pointer. The full v2 lint is lint_ledger()."""
    bad = []
    for c in sorted(claims.values(), key=lambda c: int(c["id"][1:])):
        h = c["history"][-1]
        if h["status"] not in ("UNVERIFIED", "SUPERSEDED") and not PTR_RE.search(h["raw_evidence"]):
            bad.append((c["id"], h["status"], h["line"]))
    return bad


def render(claims):
    """v1 view of parse()-shaped data (kept for callers; the generated file now uses Store.render_status())."""
    key = lambda c: int(c["id"][1:])
    rows = sorted(claims.values(), key=key)
    sts = ALL_STATUSES
    by = {s: [c for c in rows if c["history"][-1]["status"] == s] for s in sts}
    out = ["# Claims: current status", "", "| Status | Count |", "|---|---|"]
    out += [f"| {s} | {len(by[s])} |" for s in sts if by[s]] + [f"| **Total** | {len(rows)} |", ""]
    for s in sts:
        if by[s]:
            out += [f"## {s} ({len(by[s])})", "", "| ID | Claim | Latest evidence |", "|---|---|---|"]
            out += [f"| {c['id']} | {c['text'].replace('|', '/')} | {c['history'][-1]['evidence'].replace('|', '/')} |"
                    for c in by[s]] + [""]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------- evidence pointers
_TOP = r"(?:experiments|evals|notes|repro|gogeta|tools|tests|post_training|refs|docs|cockpit|configs|site-src)"
FILE_RE = re.compile(rf"(?<![\w/.~-])({_TOP}/[\w./*<>+-]*[\w*>])(?::(\d+(?:-\d+)?))?")
RUN_RE = re.compile(r"\b((?:E|K|SQ)\d+[A-Za-z0-9-]*-\d{8}-\d{6}-s\d+)\b")
VER_RE = re.compile(r"\bv\d+\.\d+\.\d+\b")
COMMIT_RE = re.compile(r"(?:\bcommit\s+`?|`)([0-9a-f]{7,40})`?(?![\w])")
EXP_RE = re.compile(r"(?<![\w-])((?:E|K|SQ)\d+(?:-[Pp]arity)?)(?![\w])")
CLAIM_RE = re.compile(r"^C\d+$")


def extract_pointers(text, cell=None):
    """Structured evidence pointers found in free text (used by the migration; order of appearance, deduped)."""
    out, seen = [], set()

    def add(p):
        k = (p["kind"], p["ref"], p.get("lines"))
        if k not in seen:
            seen.add(k)
            if cell:
                p["cell"] = cell
            out.append(p)

    runs = RUN_RE.findall(text)
    rest = RUN_RE.sub(" ", text)
    for m in FILE_RE.finditer(rest):
        p = {"kind": "file", "ref": m.group(1).rstrip(".")}
        if m.group(2):
            p["lines"] = m.group(2)
        add(p)
    for r in runs:
        add({"kind": "run", "ref": r})
    for v in VER_RE.findall(text):
        add({"kind": "version", "ref": v})
    for c in COMMIT_RE.findall(text):
        add({"kind": "commit", "ref": c})
    for e in EXP_RE.findall(FILE_RE.sub(" ", rest)):
        add({"kind": "experiment", "ref": e})
    return out


def classify_pointer(tok):
    """One CLI --evidence token -> pointer dict (raises LedgerError if unrecognised)."""
    tok = tok.strip().strip("`")
    if RUN_RE.fullmatch(tok):
        return {"kind": "run", "ref": tok}
    if VER_RE.fullmatch(tok):
        return {"kind": "version", "ref": tok}
    if CLAIM_RE.match(tok):
        return {"kind": "claim", "ref": tok}
    if EXP_RE.fullmatch(tok):
        return {"kind": "experiment", "ref": tok}
    if re.fullmatch(r"(?:commit:)?[0-9a-f]{7,40}", tok):
        return {"kind": "commit", "ref": tok.split(":")[-1]}
    m = re.fullmatch(r"([^:]+?)(?::(\d+(?:-\d+)?))?", tok)
    if m and "/" in m.group(1):
        p = {"kind": "file", "ref": m.group(1)}
        if m.group(2):
            p["lines"] = m.group(2)
        return p
    raise LedgerError(f"unrecognised evidence pointer {tok!r}: use a repo path (dir/file[:line]), a run id, "
                      "a version tag vX.Y.Z, a commit hash, an experiment id (E1, K9, SQ70) or a claim id")


def pointer_text(p):
    return p["ref"] + (f":{p['lines']}" if p.get("lines") else "")


# ----------------------------------------------------------------------------------------------- helpers
def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git(root, *args, inp=None):
    try:
        r = subprocess.run(["git", "-C", str(root), *args], input=inp, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def cid_num(cid):
    return int(cid[1:])


def esc(s):
    return str(s).replace("|", "/").replace("\n", " ").strip()


def dump_line(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def scope_text(scope):
    if not isinstance(scope, dict):
        return str(scope)
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in scope.items() if v and v != TBD]
    return "; ".join(parts) if parts else TBD


def deps_list(defn):
    d = defn.get("dependencies")
    return [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []


def get_field(defn, field):
    if field.startswith("scope."):
        return (defn.get("scope") or {}).get(field[6:])
    return defn.get(field)


def set_field(defn, field, value):
    if field.startswith("scope."):
        defn.setdefault("scope", {})[field[6:]] = value
    else:
        defn[field] = value


# ----------------------------------------------------------------------------------------------- the store
class Store:
    """The ledger under <root>/experiments/claims/. Every path derives from root, so tests can use a temp repo."""

    def __init__(self, root=ROOT):
        self.root = pathlib.Path(root)
        self.dir = self.root / "experiments/claims"
        self.ledger = self.dir / "ledger"
        self.views = self.dir / "views"
        self.manifest = self.dir / "MIGRATION.json"
        self.src = self.root / "experiments/claims.md"
        self.out = self.root / "experiments/claims-status.md"

    # ---- reading
    def exists(self):
        return self.ledger.is_dir() and any(self.ledger.glob("C*.jsonl"))

    def path(self, cid):
        return self.ledger / f"{cid}.jsonl"

    def rel(self, p):
        return pathlib.Path(p).relative_to(self.root).as_posix()

    def files(self):
        fs = [p for p in self.ledger.glob("C*.jsonl") if ID_FILE_RE.match(p.name)] if self.ledger.is_dir() else []
        return sorted(fs, key=lambda p: int(ID_FILE_RE.match(p.name).group(1)))

    @staticmethod
    def parse_lines(text, name="?"):
        events, errors = [], []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if not isinstance(e, dict):
                    raise ValueError("not an object")
                events.append(e)
            except ValueError as ex:
                errors.append(f"{name}:{n}: bad JSON line ({ex})")
        return events, errors

    def read(self, cid):
        p = self.path(cid)
        if not p.exists():
            raise LedgerError(f"no such claim {cid} ({self.rel(p)} missing)")
        return self.parse_lines(p.read_text(encoding="utf-8"), self.rel(p))

    def build(self, cid, events, errors=()):
        """Claim dict from its events: original + effective definition (errata applied in order)."""
        created = events[0] if events and events[0].get("event") == "created" else None
        orig = json.loads(json.dumps(created.get("definition") or {})) if created else {}
        eff = json.loads(json.dumps(orig))
        for e in events[1:]:
            if e.get("event") == "erratum" and isinstance(e.get("erratum"), dict):
                set_field(eff, e["erratum"].get("field", ""), e["erratum"].get("new"))
        return {"id": cid, "file": self.rel(self.path(cid)), "events": events, "definition": eff,
                "definition_original": orig, "errors": list(errors)}

    def load(self):
        out = {}
        for p in self.files():
            cid = p.stem
            ev, err = self.parse_lines(p.read_text(encoding="utf-8"), self.rel(p))
            out[cid] = self.build(cid, ev, err)
        return out

    def claim(self, cid):
        ev, err = self.read(cid)
        return self.build(cid, ev, err)

    # ---- derived state
    @staticmethod
    def derive(claim):
        st = {"status": None, "qual": "", "reason": "", "status_event": None, "evidence_current": [],
              "evidence_all": [], "limitations": [], "last_verified": None, "superseded_by": None, "audits": [],
              "n_events": len(claim["events"]), "last_event": None}
        for e in claim["events"]:
            ptrs = [p for p in (e.get("evidence") or []) if isinstance(p, dict)]
            st["evidence_all"] += ptrs
            if e.get("status"):
                st.update(status=e["status"], qual=e.get("qual", ""), reason=e.get("reason", ""), status_event=e,
                          evidence_current=list(ptrs))
            elif e.get("event") == "evidence_added":
                st["evidence_current"] += ptrs
            for lim in e.get("limitations") or []:
                lim = lim if isinstance(lim, dict) else {"text": str(lim)}
                st["limitations"].append({**lim, "seq": e.get("seq"), "version": e.get("version")})
            if e.get("verified_at"):
                st["last_verified"] = {**e["verified_at"], "seq": e.get("seq")}
            if e.get("event") == "audit_verdict":
                st["audits"].append({"seq": e.get("seq"), "ts": e.get("ts"), "verdict": e.get("verdict"),
                                     "actor": e.get("actor"), "reason": e.get("reason", "")})
            if e.get("event") == "superseded":
                st["superseded_by"] = e.get("superseded_by")
            st["last_event"] = e
        return st

    def summary(self, cid=None, claim=None):
        """JSON-able current view of one claim (definition + derived state): the MCP-friendly shape."""
        c = claim or self.claim(cid)
        d, s = c["definition"], self.derive(c)
        return {"id": c["id"], "file": c["file"], "proposition": d.get("proposition"), "type": d.get("type"),
                "type_inferred": d.get("type_inferred", False), "scope": d.get("scope"),
                "refutation_criteria": d.get("refutation_criteria"), "dependencies": d.get("dependencies"),
                "supersedes": d.get("supersedes"), "created_version": d.get("created_version"),
                "origin": d.get("origin"), "status": s["status"], "qual": s["qual"], "reason": s["reason"],
                "evidence": s["evidence_current"], "limitations": s["limitations"],
                "last_verified": s["last_verified"], "superseded_by": s["superseded_by"],
                "audits": s["audits"], "n_events": s["n_events"]}

    # ---- v1-shaped view (parse())
    def parse(self):
        out = {}
        for cid, c in self.load().items():
            hist, cur = [], None
            for e in c["events"]:
                if not (e.get("status") or e.get("event") == "evidence_added"):
                    continue
                stt = e.get("status") or cur
                if not stt:
                    continue
                cur = stt
                prov = e.get("provenance") or {}
                extra = {"file": c["file"], "seq": e.get("seq"), "event": e.get("event"), "version": e.get("version"),
                         "ts": e.get("ts")}
                if prov.get("line"):   # migrated row: keep its claims.md line and verbatim evidence cell
                    raw, section, line = prov.get("row_evidence", ""), prov.get("section", ""), prov["line"]
                else:                  # post-migration event: synthetic, time-ordered line (NOT a claims.md line)
                    raw = "; ".join([e.get("reason", "")] + [pointer_text(p) for p in e.get("evidence") or []]).strip("; ")
                    section = f"{c['file']} event {e.get('seq')} ({e.get('event')}, {e.get('version')})"
                    try:
                        line = int(dt.datetime.fromisoformat(e.get("ts")).timestamp())
                    except (TypeError, ValueError):
                        line = 10 ** 10
                hist.append({"status": stt, "qual": e.get("qual", "") if e.get("status") else "", "section": section,
                             "line": line, "evidence": clean(raw, 220), "raw_evidence": raw, **extra})
            if hist:
                out[cid] = {"id": cid, "text": clean(c["definition"].get("proposition", "")), "history": hist}
        return out

    # ---- pointers
    def pointer_ok(self, p):
        """Does this pointer satisfy the evidence rule (AGENTS 5)? files must exist; run ids and version tags count."""
        k, ref = p.get("kind"), str(p.get("ref", ""))
        if k == "file":
            if any(ch in ref for ch in "*?["):
                return bool(_glob.glob(str(self.root / ref)))
            return "<" not in ref and (self.root / ref).exists()
        if k == "run":
            return bool(RUN_RE.fullmatch(ref))
        if k == "version":
            return bool(VER_RE.fullmatch(ref))
        return False

    # ---- lint
    def check_claim(self, c, ids, legacy_ids=frozenset()):
        """Per-claim structural + definition + evidence rules. Returns findings [(level, id, rule, msg)]."""
        F, cid, ev = [], c["id"], c["events"]
        add = lambda lv, rule, msg: F.append((lv, cid, rule, msg))
        for err in c["errors"]:
            add("fail", "structure", err)
        if not ev:
            add("fail", "structure", "no events")
            return F
        if ev[0].get("event") != "created":
            add("fail", "structure", "first event must be `created`")
        if sum(e.get("event") == "created" for e in ev) != 1:
            add("fail", "structure", "exactly one `created` event allowed")
        for i, e in enumerate(ev, 1):
            where = f"event {i}"
            if e.get("seq") != i:
                add("fail", "structure", f"{where}: seq {e.get('seq')!r} (expected {i}; events are numbered 1..n)")
            if e.get("event") not in EVENT_TYPES:
                add("fail", "structure", f"{where}: unknown event type {e.get('event')!r}")
            if not ACTOR_RE.match(str(e.get("actor", ""))):
                add("fail", "structure", f"{where}: bad actor {e.get('actor')!r}")
            try:
                dt.datetime.fromisoformat(str(e.get("ts")))
            except ValueError:
                add("fail", "structure", f"{where}: bad ts {e.get('ts')!r}")
            if not VERSION_RE.match(str(e.get("version", ""))):
                add("fail", "structure", f"{where}: bad version {e.get('version')!r} (vX.Y.Z)")
            s = e.get("status")
            if s is not None and s not in ALL_STATUSES:
                add("fail", "status", f"{where}: unknown status {s!r}")
            if s and e.get("event") not in STATUS_EVENTS:
                add("fail", "status", f"{where}: a {e.get('event')} event cannot set a status")
            if e.get("event") == "status_change" and not s:
                add("fail", "status", f"{where}: status_change without a status")
            if s == "SUPERSEDED" and e.get("event") != "superseded":
                add("fail", "status", f"{where}: SUPERSEDED is set only by a `superseded` event")
            if e.get("event") == "superseded" and (s != "SUPERSEDED" or not e.get("superseded_by")):
                add("fail", "status", f"{where}: a superseded event needs status SUPERSEDED and superseded_by")
            if e.get("event") == "audit_verdict" and e.get("verdict") not in VERDICTS:
                add("fail", "structure", f"{where}: audit verdict {e.get('verdict')!r} not in {VERDICTS}")
            migrated = e.get("actor") == "migration"
            if not migrated and e.get("event") in ("status_change", "superseded", "erratum") and not e.get("reason"):
                add("fail", "structure", f"{where}: {e.get('event')} needs a reason")
            if not migrated and s == "VERIFIED":   # notes/statistics-policy.md
                ec = e.get("evidence_class")
                if ec not in EVIDENCE_CLASSES:
                    add("fail", "policy", f"{where}: VERIFIED needs evidence_class in {EVIDENCE_CLASSES} "
                                          "(notes/statistics-policy.md section 1)")
                elif ec == "training" and int(e.get("n_seeds") or 0) < 3:
                    add("fail", "policy", f"{where}: a training comparison is VERIFIED only with >= 3 seeds per arm "
                                          "(notes/statistics-policy.md section 2); use PARTIAL")
            for p in e.get("evidence") or []:
                if not isinstance(p, dict) or "kind" not in p or "ref" not in p:
                    add("fail", "evidence", f"{where}: malformed evidence pointer {p!r}")
        # definition
        d0 = c["definition_original"]
        for f in DEF_FIELDS:
            if f not in d0 or d0[f] in (None, "", {}):
                add("fail", "definition", f"definition lacks required field `{f}`")
        if d0.get("id") and d0.get("id") != cid:
            add("fail", "definition", f"definition id {d0.get('id')} does not match file {c['file']}")
        d = c["definition"]
        if not isinstance(d.get("scope"), dict) or any(not (d["scope"].get(k) or "").strip() for k in SCOPE_FIELDS):
            add("fail", "definition", f"scope must give all of {SCOPE_FIELDS} (use {TBD!r} when not stated)")
        if d.get("type") not in TYPES:
            add("fail", "definition", f"type {d.get('type')!r} not in {TYPES}")
        if not (isinstance(d.get("dependencies"), list) or d.get("dependencies") == TBD):
            add("fail", "definition", f"dependencies must be a list or {TBD!r}")
        if not VERSION_RE.match(str(d0.get("created_version", ""))):
            add("fail", "definition", f"created_version {d0.get('created_version')!r} is not vX.Y.Z")
        for dep in deps_list(d):
            if dep.get("claim") and dep["claim"] not in ids:
                add("fail", "dependencies", f"depends on unknown claim {dep['claim']}")
        sup = d.get("supersedes")
        if sup and sup not in ids:
            add("fail", "supersede", f"supersedes unknown claim {sup}")
        # errata
        shadow = json.loads(json.dumps(d0))
        evidence_seen = False
        for i, e in enumerate(ev, 1):
            if e.get("event") in ("evidence_added", "status_change") and e.get("evidence"):
                evidence_seen = True
            if e.get("event") != "erratum":
                continue
            er = e.get("erratum") or {}
            fld, old, new = er.get("field"), er.get("old"), er.get("new")
            if fld not in ERRATUM_FIELDS:
                add("fail", "erratum", f"event {i}: field {fld!r} cannot take an erratum (allowed: {sorted(ERRATUM_FIELDS)})")
                continue
            if get_field(shadow, fld) != old:
                add("fail", "erratum", f"event {i}: `old` does not match the current {fld}")
            kind = er.get("kind", "typo")
            if kind == "tbd_fill":
                if old != TBD:
                    add("fail", "erratum", f"event {i}: tbd_fill only replaces {TBD!r}")
                elif fld == "refutation_criteria" and evidence_seen:
                    add("warn", "erratum", f"event {i}: refutation criteria filled after evidence was recorded (post hoc)")
            elif kind == "typo":
                if fld == "proposition" and difflib.SequenceMatcher(None, str(old), str(new)).ratio() < TYPO_RATIO:
                    add("fail", "erratum", f"event {i}: not a typo fix (similarity < {TYPO_RATIO}); a changed "
                                           "proposition needs a new claim with supersedes")
            else:
                add("fail", "erratum", f"event {i}: erratum kind must be typo or tbd_fill")
            set_field(shadow, fld, new)
        # current status and evidence
        s = self.derive(c)
        if s["status"] is None:
            add("fail", "status", "no status (the created event must set one, e.g. UNVERIFIED)")
        elif s["status"] == "SUPERSEDED":
            pass
        elif s["status"] != "UNVERIFIED":
            ok = [p for p in s["evidence_current"] if self.pointer_ok(p)]
            if not ok:
                add("fail", "evidence", f"{s['status']} but the event that set it (and later evidence_added events) "
                                        "cite no pointer that exists: a repo file, a run id or a version tag")
            for p in s["evidence_current"]:
                if p.get("kind") == "file" and not self.pointer_ok(p):
                    add("warn", "evidence", f"cited file not found: {pointer_text(p)}")
            if s["status"] in SETTLED and not (s["last_verified"] or {}).get("commit") \
                    or s["status"] in SETTLED and not (s["last_verified"] or {}).get("date"):
                add("fail", "last_verified", f"{s['status']} but no verified_at {{commit, date}} in its events")
        return F

    def lint(self, base="HEAD", include_git=True):
        """Full v2 lint. Returns findings [{"level", "id", "rule", "msg"}]; any "fail" blocks a version."""
        claims, F = self.load(), []
        ids = set(claims)
        for c in claims.values():
            F += self.check_claim(c, ids)
        # dependency cycles
        graph = {cid: [d["claim"] for d in deps_list(c["definition"]) if d.get("claim") in ids]
                 for cid, c in claims.items()}
        for cyc in find_cycles(graph):
            F.append(("fail", cyc[0], "dependencies", "dependency cycle: " + " -> ".join(cyc + [cyc[0]])))
        # dependents of REFUTED / WRONG FOR OUR CODE claims are flagged for review
        for cid, flags in self.review_flags(claims).items():
            for msg in flags:
                F.append(("warn", cid, "review", msg))
        # supersede consistency
        for cid, c in claims.items():
            s = self.derive(c)
            if s["status"] == "SUPERSEDED":
                nb = s["superseded_by"]
                if nb not in claims or claims[nb]["definition"].get("supersedes") != cid:
                    F.append(("fail", cid, "supersede", f"SUPERSEDED by {nb}, whose definition does not say supersedes: {cid}"))
            sup = c["definition"].get("supersedes")
            if sup in claims and self.derive(claims[sup])["superseded_by"] != cid:
                F.append(("fail", cid, "supersede", f"supersedes {sup}, but {sup} has no superseded event pointing here"))
        # frozen v1 ledger
        F += self.check_frozen()
        if include_git:
            F += self.check_append_only(base)
        return [{"level": lv, "id": i, "rule": r, "msg": m} for lv, i, r, m in F]

    def review_flags(self, claims=None):
        claims = claims or self.load()
        out = {}
        for cid, c in claims.items():
            mine = c["events"]
            for dep in deps_list(c["definition"]):
                dc = claims.get(dep.get("claim"))
                if not dc:
                    continue
                ds = self.derive(dc)
                if ds["status"] in NEGATIVE:
                    t_bad = str((ds["status_event"] or {}).get("ts", ""))
                    reviewed = any(str(e.get("ts", "")) > t_bad and e.get("actor") != "migration" for e in mine)
                    if not reviewed:
                        out.setdefault(cid, []).append(
                            f"depends on {dep['claim']}, which is {ds['status']}: review this claim and append an event")
        return out

    def check_frozen(self):
        if not self.manifest.exists():
            return []
        try:
            m = json.loads(self.manifest.read_text(encoding="utf-8"))
        except ValueError as ex:
            return [("fail", "-", "frozen", f"MIGRATION.json unreadable: {ex}")]
        sha = m.get("frozen_sha256")
        if not sha:
            return []
        if not self.src.exists():
            return [("fail", "-", "frozen", "experiments/claims.md (frozen archive) is missing: never delete it")]
        now = hashlib.sha256(self.src.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if now != sha:
            return [("fail", "-", "frozen", "experiments/claims.md changed after the freeze: it is a read-only archive. "
                     "Revert it and record the change with `python experiments/claims_list.py add-event ...`")]
        return []

    def check_append_only(self, base="HEAD"):
        """Compare every ledger file with its version in `base`: base lines must be an exact prefix."""
        rel = self.rel(self.ledger)
        names = _git(self.root, "ls-tree", "-r", "--name-only", base, "--", rel + "/")
        if names is None:
            return []   # no git or no such base (e.g. first commit): nothing to compare
        paths = [p for p in names.decode("utf-8").splitlines() if p.endswith(".jsonl")]
        if not paths:
            return []
        blobs = git_cat_files(self.root, base, paths)
        F = []
        for p in paths:
            cid = pathlib.Path(p).stem
            cur_p = self.root / p
            if not cur_p.exists():
                F.append(("fail", cid, "append-only", f"{p} exists in {base} but was deleted: claims are never removed"))
                continue
            old = [l for l in (blobs.get(p) or "").splitlines() if l.strip()]
            new = [l for l in cur_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            if old and (not new or new[0] != old[0]):
                F.append(("fail", cid, "immutability", f"the definition (created event, line 1 of {p}) differs from "
                          f"{base}: definitions never change; append an `erratum` event or create a new claim with "
                          "supersedes"))
            bad = [i for i, l in enumerate(old[1:], 2) if i > len(new) or new[i - 1] != l]
            if bad:
                F.append(("fail", cid, "append-only", f"{p}: event line(s) {bad[:5]} edited or removed vs {base}: "
                          "events are append-only; restore them and append a correcting event"))
        return F

    # ---- writing
    def _lock(self, path, timeout=20.0):
        lp = path.with_name(path.name + ".lock")
        t0 = time.time()
        while True:
            try:
                fd = os.open(str(lp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lp
            except FileExistsError:
                try:
                    if time.time() - lp.stat().st_mtime > 60:
                        lp.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > timeout:
                    raise LedgerError(f"{self.rel(lp)} is held by another writer")
                time.sleep(0.05)

    def append(self, cid, event, validate=True):
        """Append one event (seq assigned here) under a per-claim lock. Refuses if the result would fail the lint."""
        p = self.path(cid)
        if not p.exists():
            raise LedgerError(f"no such claim {cid}")
        lp = self._lock(p)
        try:
            events, errors = self.read(cid)
            if errors:
                raise LedgerError("; ".join(errors))
            event = {"seq": len(events) + 1, **{k: v for k, v in event.items() if k != "seq"}}
            if validate:
                ids = {f.stem for f in self.files()}
                fails = [f for f in self.check_claim(self.build(cid, events + [event]), ids) if f[0] == "fail"]
                pre = {(f[2], f[3]) for f in self.check_claim(self.build(cid, events), ids) if f[0] == "fail"}
                fails = [f for f in fails if (f[2], f[3]) not in pre]   # only refuse what THIS event breaks
                if fails:
                    raise LedgerError("refused: " + "; ".join(f"[{f[2]}] {f[3]}" for f in fails))
            data = p.read_bytes()
            with open(p, "ab") as fh:
                if data and not data.endswith(b"\n"):
                    fh.write(b"\n")
                fh.write((dump_line(event) + "\n").encode("utf-8"))
            return event
        finally:
            try:
                lp.unlink()
            except FileNotFoundError:
                pass

    def resolve_pointers(self, tokens, must_exist=True):
        ptrs = [classify_pointer(t) for t in tokens if t.strip()]
        if must_exist:
            for p in ptrs:
                if p["kind"] == "file" and not self.pointer_ok(p):
                    raise LedgerError(f"evidence file not found: {pointer_text(p)} (never cite evidence that isn't on disk)")
                if p["kind"] == "claim" and not self.path(p["ref"]).exists():
                    raise LedgerError(f"evidence claim {p['ref']} does not exist")
        return ptrs

    def head(self, short=True):
        out = _git(self.root, "rev-parse", "--short" if short else "--verify", "HEAD")
        return out.decode().strip() if out else "unknown"

    def add_event(self, cid, version, actor="main-agent", event=None, status=None, qual="", reason="",
                  evidence=(), limitations=(), verified_commit=None, verified_date=None, evidence_class=None,
                  n_seeds=None, verdict=None, extra=None):
        """Append an evidence_added / status_change / limitation_added / audit_verdict event."""
        c = self.claim(cid)
        cur = self.derive(c)["status"]
        if event is None:
            event = ("status_change" if status and status != cur else "evidence_added" if evidence or status
                     else "limitation_added" if limitations else None)
            if event is None:
                raise LedgerError("nothing to record: give --status, --evidence or --limitation")
        if event in ("created", "erratum", "superseded"):
            raise LedgerError(f"use new-claim / erratum / new-claim --supersedes for a {event} event")
        if status == "SUPERSEDED":
            raise LedgerError("SUPERSEDED is set by `new-claim --supersedes`")
        e = {"ts": now_iso(), "version": version, "actor": actor, "event": event, "status": status or None,
             "qual": qual or "", "reason": reason or "", "evidence": self.resolve_pointers(evidence),
             "limitations": [{"text": l} for l in limitations if l.strip()]}
        if status in SETTLED:
            e["verified_at"] = {"commit": verified_commit or self.head(), "date": verified_date or dt.date.today().isoformat(),
                                "basis": "given" if verified_commit else "HEAD of the checkout when the event was recorded"}
        if evidence_class:
            e["evidence_class"] = evidence_class
        if n_seeds is not None:
            e["n_seeds"] = int(n_seeds)
        if verdict:
            e["verdict"] = verdict
        if extra:
            e.update(extra)
        return self.append(cid, e)

    def next_id(self):
        used = {cid_num(p.stem) for p in self.files()}
        if self.src.exists():
            used |= {cid_num(k) for k in parse_legacy(self.src)}
        for rp in ("experiments/stats/proposed-claims.md", "repro/proposed-claims.md", "experiments/checks/PREDICTIONS.md"):
            f = self.root / rp
            if f.exists():   # IDs already reserved by proposals are skipped
                used |= {int(x) for x in re.findall(r"\bC(\d+)\b", f.read_text(encoding="utf-8"))}
        return f"C{max(used | {0}) + 1}"

    def new_claim(self, proposition, ctype, source, version, actor="main-agent", cid=None, scope=None,
                  refutation=None, dependencies=None, supersedes=None, reason="", status="UNVERIFIED", qual="",
                  evidence=(), type_inferred=False, origin_extra=None):
        if ctype not in TYPES:
            raise LedgerError(f"type must be one of {TYPES}")
        if not proposition.strip():
            raise LedgerError("empty proposition")
        if not VERSION_RE.match(version):
            raise LedgerError("version must be vX.Y.Z")
        if supersedes:
            if not self.path(supersedes).exists():
                raise LedgerError(f"cannot supersede unknown claim {supersedes}")
            if not reason:
                raise LedgerError("superseding needs --reason")
        if status not in STATUSES:
            raise LedgerError(f"initial status must be one of {STATUSES}")
        scope = {k: (scope or {}).get(k) or TBD for k in SCOPE_FIELDS}
        deps = TBD if dependencies is None else list(dependencies)
        ids = {p.stem for p in self.files()}
        for dep in deps if isinstance(deps, list) else []:
            if dep.get("claim") and dep["claim"] not in ids:
                raise LedgerError(f"dependency {dep['claim']} does not exist")
        self.ledger.mkdir(parents=True, exist_ok=True)
        auto = cid is None
        for _ in range(50):
            cid = cid or self.next_id()
            defn = {"id": cid, "proposition": proposition, "scope": scope, "type": ctype, "type_inferred": bool(type_inferred),
                    "refutation_criteria": refutation or TBD, "dependencies": deps, "created_version": version,
                    "origin": {"source": source, **(origin_extra or {})}, "supersedes": supersedes}
            e = {"seq": 1, "ts": now_iso(), "version": version, "actor": actor, "event": "created", "status": status,
                 "qual": qual, "reason": reason or "new claim", "evidence": self.resolve_pointers(evidence),
                 "limitations": [], "definition": defn}
            if status in SETTLED:
                e["verified_at"] = {"commit": self.head(), "date": dt.date.today().isoformat(),
                                    "basis": "HEAD of the checkout when the claim was created"}
            c = self.build(cid, [e])
            fails = [f for f in self.check_claim(c, ids | {cid}) if f[0] == "fail" and f[2] != "supersede"]
            if fails:
                raise LedgerError("refused: " + "; ".join(f"[{f[2]}] {f[3]}" for f in fails))
            try:
                fd = os.open(str(self.path(cid)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not auto:
                    raise LedgerError(f"{cid} already exists")
                cid = None
                continue
            with os.fdopen(fd, "wb") as fh:
                fh.write((dump_line(e) + "\n").encode("utf-8"))
            break
        else:
            raise LedgerError("could not allocate a claim id")
        if supersedes:
            self.append(supersedes, {"ts": now_iso(), "version": version, "actor": actor, "event": "superseded",
                                     "status": "SUPERSEDED", "qual": f"by {cid}", "reason": reason,
                                     "evidence": [{"kind": "claim", "ref": cid}], "limitations": [],
                                     "superseded_by": cid})
        return cid

    def erratum(self, cid, field, old, new, reason, version, actor="main-agent", kind="typo"):
        if not reason:
            raise LedgerError("an erratum needs --reason")
        return self.append(cid, {"ts": now_iso(), "version": version, "actor": actor, "event": "erratum",
                                 "status": None, "qual": "", "reason": reason, "evidence": [], "limitations": [],
                                 "erratum": {"field": field, "kind": kind, "old": old, "new": new}})

    def audit_verdict(self, cid, verdict, note, version, batch_file=None, actor="codex-auditor"):
        verdict = verdict.strip().upper()
        if verdict not in VERDICTS:
            raise LedgerError(f"verdict must be one of {VERDICTS}")
        ev = [batch_file] if batch_file and (self.root / batch_file).exists() else []
        return self.add_event(cid, version, actor=actor, event="audit_verdict", reason=note or "", evidence=ev,
                              verdict=verdict, extra={"batch_file": batch_file} if batch_file else None)

    # ---- views
    def render_status(self, claims=None):
        claims = claims or self.load()
        rows = sorted(claims.values(), key=lambda c: cid_num(c["id"]))
        der = {c["id"]: self.derive(c) for c in rows}
        by = {s: [c for c in rows if der[c["id"]]["status"] == s] for s in ALL_STATUSES}
        types = {t: sum(c["definition"].get("type") == t for c in rows) for t in TYPES}
        tbd = tbd_counts(rows)
        flags = self.review_flags(claims)
        out = ["# Claims: current status", "",
               "Generated by `python experiments/claims_list.py` from the claims ledger `experiments/claims/ledger/` "
               "(immutable definitions + append-only events; schema in `experiments/claims/README.md`). Status, "
               "evidence, limitations and last-verified are derived from the events. Don't edit by hand. The v1 table "
               "ledger `experiments/claims.md` is a frozen archive.", "",
               "| Status | Count |", "|---|---|"]
        out += [f"| {s} | {len(by[s])} |" for s in ALL_STATUSES if by[s]] + [f"| **Total** | {len(rows)} |", ""]
        out += ["| Type | Count |", "|---|---|"] + [f"| {t} | {n} |" for t, n in types.items() if n]
        out += ["", "Definition fields still `TBD (not stated at creation)`: " +
                ", ".join(f"{k} {v}" for k, v in tbd.items()) + ".", ""]
        if flags:
            out += ["## Flagged for review (a dependency is REFUTED or WRONG FOR OUR CODE)", ""]
            out += [f"- **{cid}**: " + "; ".join(m) for cid, m in sorted(flags.items(), key=lambda kv: cid_num(kv[0]))]
            out.append("")
        for s in ALL_STATUSES:
            if not by[s]:
                continue
            out += [f"## {s} ({len(by[s])})", "",
                    "| ID | Type | Claim | Scope | Latest evidence | Limitations | Last verified | History |",
                    "|---|---|---|---|---|---|---|---|"]
            for c in by[s]:
                d, x = c["definition"], der[c["id"]]
                first = next((e.get("status") for e in c["events"] if e.get("status")), None)
                qual = f" ({x['qual']})" if x["qual"] else ""
                was = f" ⟵ was {first}" if first and first != x["status"] else ""
                ty = d.get("type", "?") + ("?" if d.get("type_inferred") else "")
                ev = self.evidence_cell(c, x)
                lims = x["limitations"]
                lim = (f"{len(lims)}: " + clean(lims[-1]["text"], 110)) if lims else "none recorded"
                lv = x["last_verified"]
                lvs = f"{lv.get('commit')} {lv.get('date')}" if lv else "-"
                out.append(f"| {c['id']} | {esc(ty)} | {esc(clean(d.get('proposition', '')))}{esc(qual)}{was} | "
                           f"{esc(clean(scope_text(d.get('scope')), 140))} | {esc(ev)} | {esc(lim)} | {esc(lvs)} | "
                           f"[{x['n_events']} events](claims/views/{c['id']}.md) |")
            out.append("")
        out += ["Type `?` = inferred at migration (`type_inferred`). Limitations: count, then the newest.", ""]
        return "\n".join(out) + "\n"

    @staticmethod
    def evidence_cell(c, x):
        e = x["status_event"] or {}
        prov = e.get("provenance") or {}
        if prov.get("row_evidence") and not [ev for ev in c["events"][e.get("seq", 0):] if ev.get("event") == "evidence_added"]:
            return clean(prov["row_evidence"], 220)
        txt = "; ".join([e.get("reason", "")] + [pointer_text(p) for p in x["evidence_current"]])
        return clean(txt.strip("; "), 220)

    def render_view(self, c):
        d, x = c["definition"], self.derive(c)
        L = [f"# {c['id']}: {esc(clean(d.get('proposition', '')))}", "",
             f"Generated from `{c['file']}` by `python experiments/claims_list.py`. Don't edit: append events with "
             "`python experiments/claims_list.py add-event` (see `experiments/claims/README.md`).", "",
             "## Definition (immutable)", "", "| Field | Value |", "|---|---|",
             f"| Proposition | {esc(d.get('proposition', ''))} |",
             f"| Type | {esc(d.get('type'))}{' (inferred at migration)' if d.get('type_inferred') else ''} |"]
        sc = d.get("scope") or {}
        L += [f"| Scope: {k.replace('_', ' ')} | {esc(sc.get(k, ''))} |" for k in SCOPE_FIELDS]
        L.append(f"| Refutation criteria | {esc(d.get('refutation_criteria', ''))} |")
        deps = d.get("dependencies")
        dtxt = deps if isinstance(deps, str) else ("none" if not deps else "; ".join(
            (f"{x_.get('claim')}" if x_.get("claim") else esc(x_.get("assumption", ""))) +
            (f" ({esc(x_.get('why'))})" if x_.get("why") else "") for x_ in deps))
        L.append(f"| Dependencies | {esc(dtxt)} |")
        o = d.get("origin") or {}
        otxt = "; ".join(f"{k}: {v}" for k, v in o.items() if v not in (None, ""))
        L += [f"| Created | {esc(d.get('created_version'))} |", f"| Origin | {esc(otxt)} |"]
        if d.get("supersedes"):
            L.append(f"| Supersedes | [{d['supersedes']}]({d['supersedes']}.md) |")
        if c["definition"] != c["definition_original"]:
            L.append("| Errata | applied; see the erratum events below |")
        L += ["", "## Current state (derived from the events)", "", "| Field | Value |", "|---|---|",
              f"| Status | **{x['status']}**{(' (' + esc(x['qual']) + ')') if x['qual'] else ''} |",
              f"| Reason | {esc(x['reason'])} |"]
        if x["superseded_by"]:
            L.append(f"| Superseded by | [{x['superseded_by']}]({x['superseded_by']}.md) |")
        lv = x["last_verified"]
        L.append(f"| Last verified | {esc(lv.get('commit'))}, {esc(lv.get('date'))} ({esc(lv.get('basis', ''))}) |"
                 if lv else "| Last verified | - |")
        L.append("| Evidence | " + (esc(", ".join(pointer_text(p) for p in x["evidence_current"])) or "-") + " |")
        L += ["", "## Limitations", ""]
        L += [f"- {esc(l['text'])}" + (f" (source: {esc(l['source'])})" if l.get("source") else "") +
              f" [event {l['seq']}, {l['version']}]" for l in x["limitations"]] or ["- none recorded"]
        L += ["", "## Events (append-only)", "", "| # | Time | Version | Actor | Event | Status | Reason / evidence |",
              "|---|---|---|---|---|---|---|"]
        for e in c["events"]:
            prov = e.get("provenance") or {}
            bits = [e.get("reason", "")]
            if e.get("verdict"):
                bits.insert(0, f"verdict {e['verdict']}")
            if e.get("erratum"):
                er = e["erratum"]
                bits.append(f"erratum {er.get('field')} ({er.get('kind')}): {er.get('old')!r} -> {er.get('new')!r}")
            if prov.get("line"):
                bits.append(f"claims.md:L{prov['line']} [{prov.get('section', '')}]")
                if prov.get("row_evidence"):
                    bits.append("evidence: " + prov["row_evidence"])
            elif e.get("evidence"):
                bits.append("evidence: " + ", ".join(pointer_text(p) for p in e["evidence"]))
            if e.get("limitations"):
                bits.append("limitations: " + " / ".join((l["text"] if isinstance(l, dict) else str(l)) for l in e["limitations"]))
            st = (e.get("status") or "") + (f" ({e.get('qual')})" if e.get("status") and e.get("qual") else "")
            L.append(f"| {e.get('seq')} | {esc(e.get('ts', ''))} | {esc(e.get('version', ''))} | {esc(e.get('actor', ''))} | "
                     f"{esc(e.get('event', ''))} | {esc(st)} | {esc(' · '.join(b for b in bits if b))} |")
        return "\n".join(L) + "\n"

    def outputs(self):
        """{path: text} of every generated file."""
        claims = self.load()
        outs = {self.out: self.render_status(claims)}
        for c in claims.values():
            outs[self.views / f"{c['id']}.md"] = self.render_view(c)
        return outs

    def stale(self):
        outs = self.outputs()
        bad = [p for p, t in outs.items() if not p.exists() or p.read_text(encoding="utf-8").replace("\r\n", "\n") != t]
        extra = [p for p in self.views.glob("C*.md") if p not in outs] if self.views.is_dir() else []
        return bad + extra

    def write_outputs(self):
        if self.out.exists() and self.out.samefile(self.src) or str(self.out).lower() == str(self.src).lower():
            raise LedgerError(f"refusing: output {self.out} is the v1 ledger itself")
        outs = self.outputs()
        self.views.mkdir(parents=True, exist_ok=True)
        for p, t in outs.items():
            if not p.exists() or p.read_text(encoding="utf-8").replace("\r\n", "\n") != t:
                p.write_text(t, encoding="utf-8", newline="\n")
        for p in self.views.glob("C*.md"):
            if p not in outs:
                p.unlink()
        return len(outs)


def git_cat_files(root, base, paths):
    inp = "".join(f"{base}:{p}\n" for p in paths).encode("utf-8")
    raw = _git(root, "cat-file", "--batch", inp=inp)
    out, i = {}, 0
    if raw is None:
        return out
    for p in paths:
        nl = raw.index(b"\n", i)
        head = raw[i:nl].decode()
        i = nl + 1
        if head.endswith("missing"):
            continue
        size = int(head.split()[2])
        out[p] = raw[i:i + size].decode("utf-8")
        i += size + 1
    return out


def find_cycles(graph):
    """Simple cycles reachable in a dependency graph {id: [ids]} (each reported once, by its smallest member)."""
    seen_cycles, out = set(), []
    color = {}

    def dfs(n, stack):
        color[n] = 1
        stack.append(n)
        for m in graph.get(n, []):
            if color.get(m) == 1:
                cyc = stack[stack.index(m):]
                k = frozenset(cyc)
                if k not in seen_cycles:
                    seen_cycles.add(k)
                    i = cyc.index(min(cyc, key=cid_num))
                    out.append(cyc[i:] + cyc[:i])
            elif not color.get(m):
                dfs(m, stack)
        stack.pop()
        color[n] = 2

    for n in sorted(graph, key=cid_num):
        if not color.get(n):
            dfs(n, [])
    return out


def tbd_counts(claims):
    rows = claims.values() if isinstance(claims, dict) else claims
    t = {f"scope.{k}": 0 for k in SCOPE_FIELDS}
    t.update(refutation_criteria=0, dependencies=0)
    for c in rows:
        d = c["definition"]
        for k in SCOPE_FIELDS:
            t[f"scope.{k}"] += (d.get("scope") or {}).get(k) == TBD
        t["refutation_criteria"] += d.get("refutation_criteria") == TBD
        t["dependencies"] += d.get("dependencies") == TBD
    return t


# ----------------------------------------------------------------------------------------------- module API
_STORE = Store(ROOT)


def parse():
    """v1-shaped {id: {"id", "text", "history": [...]}} from the v2 ledger (falls back to claims.md)."""
    return _STORE.parse() if _STORE.exists() else parse_legacy(_STORE.src)


def load():
    return _STORE.load()


def claim_summary(cid):
    return _STORE.summary(cid)


def status_table():
    """[{id, status, qual, type, proposition, last_verified, n_events}] for every claim (MCP-friendly)."""
    out = []
    for cid, c in sorted(_STORE.load().items(), key=lambda kv: cid_num(kv[0])):
        s = _STORE.derive(c)
        out.append({"id": cid, "status": s["status"], "qual": s["qual"], "type": c["definition"].get("type"),
                    "proposition": clean(c["definition"].get("proposition", "")), "last_verified": s["last_verified"],
                    "n_events": s["n_events"]})
    return out


def lint_ledger(base="HEAD"):
    return _STORE.lint(base)


def add_event(cid, **kw):
    return _STORE.add_event(cid, **kw)


def new_claim(**kw):
    return _STORE.new_claim(**kw)


def record_audit_verdict(cid, verdict, note, version, batch_file=None, actor="codex-auditor"):
    return _STORE.audit_verdict(cid, verdict, note, version, batch_file, actor)


# ----------------------------------------------------------------------------------------------- CLI
def _print_findings(F, quiet_warn=False):
    for f in F:
        if f["level"] == "fail" or not quiet_warn:
            print(f"[{f['level'].upper()}] {f['id']} {f['rule']}: {f['msg']}")
    nf, nw = sum(f["level"] == "fail" for f in F), sum(f["level"] == "warn" for f in F)
    print(f"lint: {nf} fail, {nw} warn")
    return nf


def _split(vals):
    out = []
    for v in vals or []:
        out += [x.strip() for x in v.split(",") if x.strip()]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    st = _STORE
    cmds = {"add-event", "new-claim", "erratum", "audit-verdict", "lint", "show", "status"}
    if not argv or argv[0] not in cmds:
        check = "--check" in argv
        base = argv[argv.index("--base") + 1] if "--base" in argv else "HEAD"
        if not st.exists():   # pre-migration fallback: the v1 behaviour on claims.md
            claims = parse_legacy(st.src)
            bad = lint(claims)
            for cid, s, ln in bad:
                print(f"no evidence pointer: {cid} ({s}) claims.md:L{ln}")
            print("no v2 ledger found (experiments/claims/ledger/); run experiments/claims/migrate_v1.py")
            return 1 if bad else 0
        F = st.lint(base)
        if check:
            stale = st.stale()
            for p in stale:
                print(f"stale: {st.rel(p)} (run python experiments/claims_list.py)")
            nf = _print_findings(F, quiet_warn=True)
            return 1 if nf or stale else 0
        n = st.write_outputs()
        nf = _print_findings(F, quiet_warn=True)
        print(f"wrote {st.rel(st.out)} and {n - 1} views in {st.rel(st.views)}")
        return 2 if nf else 0

    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"claims_list.py {cmd}")
    if cmd in ("add-event", "erratum", "audit-verdict", "show"):
        ap.add_argument("id")
    if cmd in ("add-event", "new-claim", "erratum", "audit-verdict"):
        ap.add_argument("--version", required=True, help="the version tag this event will be committed in")
        ap.add_argument("--actor", default="main-agent", help="main-agent | subagent:<name> | codex-auditor:<batch> | human")
        ap.add_argument("--reason", default="")
        ap.add_argument("--no-render", action="store_true", help="don't regenerate the views after writing")
    if cmd in ("add-event", "new-claim"):
        ap.add_argument("--status", default=None, choices=STATUSES)
        ap.add_argument("--qual", default="", help="short qualifier shown next to the status, e.g. 'CPU only'")
        ap.add_argument("--evidence", action="append", help="pointer(s), comma-separated or repeated")
    if cmd == "add-event":
        ap.add_argument("--event", choices=["evidence_added", "status_change", "limitation_added", "audit_verdict"])
        ap.add_argument("--limitation", action="append", default=[])
        ap.add_argument("--verified-commit")
        ap.add_argument("--verified-date")
        ap.add_argument("--evidence-class", choices=EVIDENCE_CLASSES)
        ap.add_argument("--n-seeds", type=int)
    if cmd == "new-claim":
        ap.add_argument("--id")
        ap.add_argument("--proposition", required=True)
        ap.add_argument("--type", required=True, choices=TYPES)
        ap.add_argument("--source", required=True, help="where the claim comes from (paper, experiment, our code ...)")
        for k in SCOPE_FIELDS:
            ap.add_argument(f"--scope-{k.replace('_', '-')}")
        ap.add_argument("--refutation")
        ap.add_argument("--depends", action="append", help="claim ids this claim depends on")
        ap.add_argument("--assumption", action="append", default=[], help="a non-claim assumption that must hold")
        ap.add_argument("--no-dependencies", action="store_true", help="state explicitly that there are none")
        ap.add_argument("--supersedes")
    if cmd == "erratum":
        ap.add_argument("--field", required=True, choices=sorted(ERRATUM_FIELDS))
        ap.add_argument("--old", required=True)
        ap.add_argument("--new", required=True)
        ap.add_argument("--kind", default="typo", choices=["typo", "tbd_fill"])
    if cmd == "audit-verdict":
        ap.add_argument("--verdict", required=True, choices=VERDICTS)
        ap.add_argument("--note", default="")
        ap.add_argument("--batch-file")
    if cmd == "show":
        ap.add_argument("--json", action="store_true")
    if cmd == "lint":
        ap.add_argument("--base", default="HEAD")
    a = ap.parse_args(rest)
    try:
        if cmd == "lint":
            return 1 if _print_findings(st.lint(a.base)) else 0
        if cmd == "status":
            t = status_table()
            from collections import Counter
            print(dict(Counter(r["status"] for r in t)), f"total {len(t)}")
            return 0
        if cmd == "show":
            s = st.summary(a.id)
            if a.json:
                print(json.dumps(s, indent=1, ensure_ascii=False))
            else:
                print(st.render_view(st.claim(a.id)))
            return 0
        if cmd == "add-event":
            e = st.add_event(a.id, a.version, actor=a.actor, event=a.event, status=a.status, qual=a.qual,
                             reason=a.reason, evidence=_split(a.evidence), limitations=a.limitation,
                             verified_commit=a.verified_commit, verified_date=a.verified_date,
                             evidence_class=a.evidence_class, n_seeds=a.n_seeds)
            print(f"appended {a.id} event {e['seq']} ({e['event']})")
        elif cmd == "new-claim":
            deps = None
            if a.depends or a.assumption or a.no_dependencies:
                deps = [{"claim": d} for d in _split(a.depends)] + [{"assumption": x} for x in a.assumption]
            scope = {k: getattr(a, f"scope_{k}") for k in SCOPE_FIELDS}
            cid = st.new_claim(a.proposition, a.type, a.source, a.version, actor=a.actor, cid=a.id, scope=scope,
                               refutation=a.refutation, dependencies=deps, supersedes=a.supersedes, reason=a.reason,
                               status=a.status or "UNVERIFIED", qual=a.qual, evidence=_split(a.evidence))
            print(f"created {cid}" + (f" (supersedes {a.supersedes})" if a.supersedes else ""))
        elif cmd == "erratum":
            e = st.erratum(a.id, a.field, a.old, a.new, a.reason, a.version, a.actor, a.kind)
            print(f"appended {a.id} erratum event {e['seq']}")
        elif cmd == "audit-verdict":
            actor = a.actor if a.actor != "main-agent" else "codex-auditor"
            e = st.audit_verdict(a.id, a.verdict, a.note, a.version, a.batch_file, actor)
            print(f"appended {a.id} audit_verdict event {e['seq']}")
        if not a.no_render:
            st.write_outputs()
        return 0
    except LedgerError as ex:
        print(f"refused: {ex}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
