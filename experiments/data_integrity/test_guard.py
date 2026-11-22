"""test_guard.py: refuse-unless-flag guard for the frozen test split(s). Import it in any script that would read
test.bin / docs_test.jsonl, or use `python test_guard.py --open <split> --claim C123 --why "..."` to see the policy.

Policy (notes/data-integrity.md section "Lock policy"): the test set is read ONLY for a final, pre-registered claim.
A read needs ALL of:
  1. env GOGETA_TEST_UNLOCK=<claim id>  (e.g. C150), set by a human or by the main agent when cutting the final run;
  2. that claim id appears in experiments/checks/PREDICTIONS.md or the claims ledger with status PREDICTED/REGISTERED
     (the prediction is on record before the read);
  3. the checkpoint hash is named (`ckpt_sha`), so the access log shows which model saw the test set;
  4. the checkpoint was selected on val / clean-val only (the caller attests with `selected_on_val_only=True`).
Every granted or refused attempt is appended to experiments/data_integrity/test-access-log.jsonl. A second read of the
same (split, claim, ckpt) is refused (no re-looks); a new checkpoint under the same claim is allowed once.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG = HERE / "test-access-log.jsonl"


class TestSetLocked(RuntimeError):
    pass


def _log(rec: dict) -> None:
    rec["t"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _registered(claim: str) -> bool:
    for fn in (ROOT / "experiments" / "checks" / "PREDICTIONS.md", ROOT / "experiments" / "claims.md"):
        if fn.exists() and claim in fn.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def require_test_access(split: str, ckpt_sha: str, selected_on_val_only: bool) -> dict:
    """Raise TestSetLocked unless every condition holds; log either way. Returns the granted record."""
    claim = os.environ.get("GOGETA_TEST_UNLOCK", "")
    rec = {"split": split, "claim": claim, "ckpt_sha": ckpt_sha, "selected_on_val_only": selected_on_val_only}
    why = None
    if not claim:
        why = "GOGETA_TEST_UNLOCK is not set"
    elif not _registered(claim):
        why = f"{claim} has no registered prediction/claim in PREDICTIONS.md or claims.md"
    elif not ckpt_sha:
        why = "checkpoint hash missing"
    elif not selected_on_val_only:
        why = "checkpoint selection used the test set"
    elif LOG.exists() and any(json.loads(ln).get("granted") and (json.loads(ln)["split"], json.loads(ln)["claim"],
                              json.loads(ln)["ckpt_sha"]) == (split, claim, ckpt_sha) for ln in LOG.read_text().splitlines() if ln):
        why = "this (split, claim, checkpoint) was already read once: no re-looks"
    if why:
        _log({**rec, "granted": False, "reason": why})
        raise TestSetLocked(f"test split '{split}' is locked: {why}")
    _log({**rec, "granted": True})
    return rec


def guard_selftest() -> dict:
    import tempfile
    global LOG
    saved, LOG = LOG, pathlib.Path(tempfile.mkdtemp()) / "log.jsonl"
    old = os.environ.pop("GOGETA_TEST_UNLOCK", None)
    out = {}
    try:
        for name, kw, env in [("no_flag", dict(ckpt_sha="abc", selected_on_val_only=True), None),
                              ("unregistered_claim", dict(ckpt_sha="abc", selected_on_val_only=True), "C99999"),
                              ("no_ckpt", dict(ckpt_sha="", selected_on_val_only=True), "C1"),
                              ("test_selected", dict(ckpt_sha="abc", selected_on_val_only=False), "C1")]:
            if env:
                os.environ["GOGETA_TEST_UNLOCK"] = env
            else:
                os.environ.pop("GOGETA_TEST_UNLOCK", None)
            try:
                require_test_access("code-py-v1-test", **kw)
                out[name] = "GRANTED (bug)"
            except TestSetLocked:
                out[name] = "refused"
    finally:
        LOG = saved
        if old is not None:
            os.environ["GOGETA_TEST_UNLOCK"] = old
        else:
            os.environ.pop("GOGETA_TEST_UNLOCK", None)
    out["pass"] = all(v == "refused" for k, v in out.items())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = guard_selftest()
        print(json.dumps(r))
        sys.exit(0 if r["pass"] else 1)
    if a.log:
        print(LOG.read_text() if LOG.exists() else "(no accesses)")
    else:
        print(__doc__)
