"""Expense tracking (the human's request, 2026-11-22: "lets track our expenses").

Ledger: experiments/expenses.json (committed, PUBLIC: amounts and purposes only; never card digits, receipt numbers or
links, billing addresses or e-mails). Entry types: credit_purchase, compute (per rental, may be estimated), other.
Reconciliation reads the provider's own credit and total spend (two numbers; everything else in the account record is
discarded) and appends a dated row, so estimates can be checked against what was actually charged.

    bash cockpit/cockpit.sh --expenses              summary
    bash cockpit/cockpit.sh --expenses reconcile    append a reconciliation from the provider's live numbers
MCP: gogeta_expenses (read-only), gogeta_expenses_reconcile (writes the ledger file only; no money moves).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

LEDGER = core.REPO / "experiments" / "expenses.json"


def _load() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _save(d: dict) -> None:
    LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def expenses() -> dict:
    """READ-ONLY. Expense summary from experiments/expenses.json: credit bought, compute spent (estimated vs reconciled
    with the provider), remaining credit, spend per experiment and per provider, and the latest reconciliation."""
    d = _load()
    ent = d.get("entries", [])
    bought = sum(e["amount"] for e in ent if e["type"] == "credit_purchase")
    spent = [e for e in ent if e["type"] == "compute"]
    est = sum(e["amount"] for e in spent)
    per_exp: dict = {}
    for e in spent:
        exps = e.get("experiments") or ["(unattributed)"]
        for x in exps:  # split a shared rental evenly across its experiments
            per_exp[x] = round(per_exp.get(x, 0) + e["amount"] / len(exps), 4)
    rec = (d.get("reconciliations") or [None])[-1]
    actual = rec["provider_total_spend"] if rec else None
    return {"currency": d.get("currency", "USD"), "credit_bought": round(bought, 2), "compute_estimated": round(est, 4),
            "compute_actual_latest": actual, "unattributed_gap": round(actual - est, 4) if actual is not None else None,
            "credit_left_latest": rec["credit_left"] if rec else None, "reconciled_at": rec["ts"] if rec else None,
            "per_experiment_estimated": per_exp, "rentals": len(spent),
            "lines": ["EXPENSES: bought $%.2f credit; compute $%s actual (provider, %s) vs $%.2f estimated; $%s credit left" % (
                bought, actual if actual is not None else "?", rec["ts"][:16] if rec else "never", est,
                rec["credit_left"] if rec else "?")]}


def expenses_reconcile(note: str = "") -> dict:
    """WRITES experiments/expenses.json only (no money moves). Read the provider's live credit and total spend (vast.ai,
    via the vastai CLI's own config; no API key passed) and append a dated reconciliation row."""
    from providers import vast  # local import: tests and offline use need no CLI
    rc, data, text = vast.run(["show", "user"], timeout=60)
    if rc != 0 or not isinstance(data, dict):
        return {"error": "vastai show user failed: %s" % text[:200]}
    credit, total = data.get("credit"), data.get("total_spend")
    del data  # discard everything else in the account record (address, e-mail, ...)
    if not isinstance(credit, (int, float)) or not isinstance(total, (int, float)):
        return {"error": "provider did not return numeric credit / total_spend"}
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "provider": "vast.ai", "credit_left": round(float(credit), 4),
           "provider_total_spend": round(abs(float(total)), 4), "note": (note or "")[:300], "by": "cockpit"}
    d = _load()
    d.setdefault("reconciliations", []).append(row)
    _save(d)
    return {"appended": row, "summary": expenses()["lines"]}


if __name__ == "__main__":
    out = expenses_reconcile(" ".join(sys.argv[2:])) if len(sys.argv) > 1 and sys.argv[1] == "reconcile" else expenses()
    print(json.dumps(out, indent=1))
