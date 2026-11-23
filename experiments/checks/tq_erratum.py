"""Erratum check (v0.5.45): Cornish-Fisher t quantile -> exact Student t.

Six analysis scripts used a Cornish-Fisher expansion of the two-sided 95% t quantile. It is about 10% low at df 2
(3.852 vs exact 4.303), so n=3 intervals were too narrow. Outside review 2026-11-23 18:08 flagged that C209's CI
did not reproduce; this is the cause. The scripts now call tstat.t_quantile (exact, stdlib).

This check (1) tabulates old vs exact quantiles at the df values the affected analyses use, and (2) recomputes the F2
intervals from the stored (I, se, df) in results/f2_analysis*.json (their inputs are pulled run dirs, not in git),
re-applying f2_analysis.decide's rule (DELTA 0.01) to see whether any decision changes.
    python experiments/checks/tq_erratum.py   -> results/tq_erratum.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tstat import t_quantile  # noqa: E402

RES = os.path.join(HERE, "results")
DELTA = 0.01


def cf2(df):  # the two-term expansion used by a2_assembled, e60_stage1, sq263, sq282_sq283
    z = 1.959963984540054
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def decide(lo, hi):
    if hi < 0:
        return "SYNERGY"
    if lo > 0:
        return "ANTI-SYNERGY"
    if lo > -DELTA and hi < DELTA:
        return "NO INTERACTION (equivalence)"
    return "INCONCLUSIVE"


def main():
    out = {"quantiles": [], "f2": {}, "decisions_changed": []}
    for df in (2, 3, 4, 5, 8, 10, 20, 60, 140):
        e, c = t_quantile(0.05, df), cf2(df)
        out["quantiles"].append({"df": df, "exact": e, "cornish_fisher": c, "cf_low_by_pct": 100 * (1 - c / e)})
    for f in sorted(os.listdir(RES)):
        if not (f.startswith("f2_analysis") and f.endswith(".json")):
            continue
        d = json.load(open(os.path.join(RES, f)))
        for key in ("primary_seed_blocked", "sensitivity_unblocked"):
            r = d.get(key)
            if not r:
                continue
            t = t_quantile(0.05, r["df"])
            ci = [r["I"] - t * r["se"], r["I"] + t * r["se"]]
            row = {"stored_ci95": r["ci95"], "exact_ci95": ci, "df": r["df"], "stored_decision": r["decision"],
                   "exact_decision": decide(*ci)}
            out["f2"][f"{f}:{key}"] = row
            if row["exact_decision"] != row["stored_decision"].split(" (")[0] and \
                    not row["stored_decision"].startswith(row["exact_decision"].split(" (")[0]):
                out["decisions_changed"].append(f"{f}:{key}")
    json.dump(out, open(os.path.join(RES, "tq_erratum.json"), "w", newline="\n"), indent=1)
    for q in out["quantiles"]:
        print(f"df {q['df']:>4}: exact {q['exact']:.5f}  CF {q['cornish_fisher']:.5f}  CF low by {q['cf_low_by_pct']:.3f}%")
    for k, r in out["f2"].items():
        print(f"{k:55s} {r['stored_ci95'][0]:+.5f},{r['stored_ci95'][1]:+.5f} -> {r['exact_ci95'][0]:+.5f},{r['exact_ci95'][1]:+.5f}  {r['exact_decision']}")
    print("decisions changed:", out["decisions_changed"] or "none")


if __name__ == "__main__":
    main()
