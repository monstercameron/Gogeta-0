"""Hourly claim audit batching (AGENTS.md section 0d-ter).

Picks the claims due for an independent review and splits them into N batches (one per Codex GPT-6 Luna auditor),
honouring a per-claim cool-down (default 6 h after its last review). Review history lives in
experiments/claim-reviews.json: {"reviews": [{"id", "ts", "batch_file", "verdict", "note"}...]}, and (since the v2
ledger) as `audit_verdict` events in experiments/claims/ledger/C<n>.jsonl; both count for the cool-down.

Priority: settled claims (VERIFIED, PARTIAL, REFUTED, WRONG FOR OUR CODE, INCONCLUSIVE) before UNVERIFIED, SUPERSEDED
last; within a tier, never reviewed first, then least recently reviewed, then claims whose latest status event is
newest (recently changed).

    python experiments/claim_review_batches.py plan  [--n 8] [--per 6] [--cooldown-h 6]   # JSON: batches to run now
    python experiments/claim_review_batches.py record <batch_file> <id>=<VERDICT>[:note] ... [--version vX.Y.Z]
    python experiments/claim_review_batches.py status                                       # due / cooling counts
Stdlib only. VERDICT is one of SOLID, WEAK, WRONG, NEEDS-EVIDENCE, UNCHECKABLE.
`record` also appends one `audit_verdict` event per claim to the v2 ledger (see experiments/claims/README.md). Its
version defaults to the next patch after the latest tag (the audit's PATCH version, AGENTS 0d-ter). A verdict never
changes a status: the main agent verifies it and appends its own event.
"""
import datetime as dt
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import claims_list  # noqa: E402
from claims_list import parse  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments/claim-reviews.json"
VERDICTS = {"SOLID", "WEAK", "WRONG", "NEEDS-EVIDENCE", "UNCHECKABLE"}
SETTLED = {"VERIFIED", "PARTIAL", "REFUTED", "WRONG FOR OUR CODE", "INCONCLUSIVE"}


def now():
    return dt.datetime.now().astimezone()


def load():
    if LOG.exists():
        return json.loads(LOG.read_text(encoding="utf-8"))
    return {"cooldown_hours": 6, "reviews": []}


def last_reviewed(log):
    stamps = [(r["id"], r["ts"]) for r in log["reviews"]]
    if claims_list._STORE.exists():   # audit_verdict events in the v2 ledger count too
        stamps += [(cid, a["ts"]) for cid, c in claims_list.load().items()
                   for a in claims_list.Store.derive(c)["audits"] if a.get("ts")]
    out = {}
    for cid, ts in stamps:
        t = dt.datetime.fromisoformat(ts)
        if cid not in out or t > out[cid]:
            out[cid] = t
    return out


def plan(n=8, per=6, cooldown_h=6.0):
    claims, log, t = parse(), load(), now()
    last = last_reviewed(log)
    due, cooling = [], []
    for c in claims.values():
        lr = last.get(c["id"])
        if lr and (t - lr).total_seconds() < cooldown_h * 3600:
            cooling.append(c["id"])
            continue
        h = c["history"][-1]
        tier = 0 if h["status"] in SETTLED else 2 if h["status"] == "SUPERSEDED" else 1
        due.append((tier, lr is not None, lr.timestamp() if lr else 0, -h["line"], c["id"], h["status"]))
    due.sort()
    picked = due[: n * per]
    batches = [[] for _ in range(n)]
    for i, d in enumerate(picked):  # round-robin so every auditor gets a mix
        batches[i % n].append({"id": d[4], "status": d[5]})
    return {"ts": t.strftime("%Y-%m-%d-%H%M"), "cooldown_hours": cooldown_h, "n_due": len(due),
            "n_cooling": len(cooling), "batches": [b for b in batches if b]}


def next_patch():
    r = subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--abbrev=0", "--match", "v*"],
                       capture_output=True, text=True)
    tag = r.stdout.strip() if r.returncode == 0 else "v0.0.0"
    a, b, c = (int(x) for x in tag.lstrip("v").split("."))
    return f"v{a}.{b}.{c + 1}"


def record(batch_file, pairs, version=None):
    log, t = load(), now().isoformat(timespec="seconds")
    parsed = []
    for p in pairs:
        cid, _, verdict = p.partition("=")
        verdict, _, note = verdict.partition(":")
        verdict = verdict.strip().upper()
        if verdict not in VERDICTS:
            sys.exit(f"bad verdict {verdict!r} for {cid}; use one of {sorted(VERDICTS)}")
        parsed.append((cid.strip(), verdict, note.strip()))
    for cid, verdict, note in parsed:
        log["reviews"].append({"id": cid, "ts": t, "batch_file": batch_file, "verdict": verdict, "note": note})
    LOG.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if claims_list._STORE.exists():
        v = version or next_patch()
        for cid, verdict, note in parsed:
            try:
                claims_list.record_audit_verdict(cid, verdict, note, v, batch_file=batch_file, actor="codex-auditor")
            except claims_list.LedgerError as ex:
                print(f"warning: {cid}: audit_verdict event not appended ({ex})", file=sys.stderr)
        claims_list._STORE.write_outputs()
    return len(parsed)


if __name__ == "__main__":
    a = sys.argv[1:] or ["status"]
    if a[0] == "plan":
        kw = {"n": 8, "per": 6, "cooldown_h": 6.0}
        for k, flag, cast in (("n", "--n", int), ("per", "--per", int), ("cooldown_h", "--cooldown-h", float)):
            if flag in a:
                kw[k] = cast(a[a.index(flag) + 1])
        print(json.dumps(plan(**kw), indent=1))
    elif a[0] == "record":
        ver = a[a.index("--version") + 1] if "--version" in a else None
        pairs = [x for i, x in enumerate(a) if i >= 2 and x != "--version" and a[i - 1] != "--version"]
        print(f"recorded {record(a[1], pairs, ver)} verdict(s) in {LOG.name} and the claims ledger")
    else:
        p = plan()
        print(f"due {p['n_due']}, cooling {p['n_cooling']}, next batch sizes {[len(b) for b in p['batches']]}")
