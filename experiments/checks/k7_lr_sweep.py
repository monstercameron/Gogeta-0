"""K7: learning-rate tuning for the control (E10) and the best hybrid (E1). CPU-only ANALYSIS script.

The runs themselves are training runs and go through the launcher (campaign
experiments/launch/campaigns/phase2a-k7-lr.campaign): 20M tokens, seed 0, lr in {1e-3, 2e-3, 3e-3,
6e-3}, everything else = the registered config. This script reads their summary.json files, applies
the decision rule that was committed before the runs (experiments/checks/PREDICTIONS.md, K7), and
either asks for a grid extension or writes the replication campaign with the tuned LRs filled in.

    ~/.venvs/gogeta/bin/python experiments/checks/k7_lr_sweep.py                    # report + verdicts
    ~/.venvs/gogeta/bin/python experiments/checks/k7_lr_sweep.py --write-campaign   # + phase2b campaign

Decision rule (pre-registered):
  1. Per model, the LR with the lowest final full-val loss wins (a run with a non-finite loss, or that
     ended 'failed', counts as diverged = worst). An exact tie (equal to 4 decimals) goes to the
     smaller LR.
  2. If the winner is at an edge of the grid, extend the grid by one step on that side
     (below 1e-3: 5e-4; above 6e-3: 1.2e-2; then 2.5e-4 / 2.4e-2) and re-apply 1. At most two
     extensions per side; if the winner is still at the edge, take it and flag "not bracketed".
  3. Phase 2b runs: E10 seeds 0,1,2 at LR_E10; E1 and E11 seeds 0,1,2 at LR_E1; E9 seed 0 at LR_E1;
     E0 seed 0 at LR_E10 only if LR_E10 != 3e-3; E2 seed 0 at LR_E1 only if LR_E1 != 3e-3
     (otherwise the existing v0.3.0 seed-0 runs are already at the comparison LR).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, save  # noqa: E402

from gogeta.tracking import DATA  # noqa: E402

GRID = [1e-3, 2e-3, 3e-3, 6e-3]
LOWER_EXT = [5e-4, 2.5e-4]
UPPER_EXT = [1.2e-2, 2.4e-2]
SWEEP_TOKENS = 20_000_000
MODELS = ["E10", "E1"]
CAMPAIGNS = ROOT / "experiments" / "launch" / "campaigns"

PREDICTION = (
    "Written 2026-11-22 before any K7 run. E10 (HF Llama control): best LR 3e-3 or 6e-3 (6e-3 slightly "
    "more likely at this short horizon); E1 (hybrid): best 2e-3 or 3e-3. Both curves are flat near the "
    "optimum (best vs second best < 1%), and lr 1e-3 is clearly worse (+3% to +8% vs best) at 20M tokens. "
    "No run diverges at 6e-3 (grad clip 1.0). Neither winner is at the low edge.")
CRITERION = ("Decision rule in the docstring / PREDICTIONS.md K7: lowest final full-val loss wins; extend the "
             "grid by one step if the winner is at an edge.")


def fmt(lr: float) -> str:
    return f"{lr:g}"   # 0.001, 0.0005, 0.012: exact, and accepted by the launcher's --lr


def find_runs(commit: str | None):
    """Latest finished 20M-token sweep run per (experiment, lr) at seed 0."""
    rows = {}
    for meta_p in sorted((DATA / "runs").glob("E*/meta.json")):
        meta = json.loads(meta_p.read_text())
        cfg = meta.get("config", {})
        ov = cfg.get("cli_overrides") or {}
        exp = meta.get("experiment")
        if exp not in MODELS or ov.get("max_tokens") != SWEEP_TOKENS or "lr" not in ov or meta.get("seed") != 0:
            continue
        if commit and not str(meta.get("git", {}).get("commit", "")).startswith(commit):
            continue
        summ_p = meta_p.parent / "summary.json"
        status, vl = "running/killed", float("nan")
        if summ_p.exists():
            s = json.loads(summ_p.read_text())
            status = s.get("status", "?")
            vl = s.get("metrics", {}).get("val_loss", float("nan"))
        key = (exp, float(ov["lr"]))
        rows[key] = {"run": meta["name"], "lr": float(ov["lr"]), "status": status, "val_loss": vl,
                     "commit": meta.get("git", {}).get("commit", "")[:7],
                     "code_dirty": meta.get("git", {}).get("code_dirty")}   # later names win (sorted by time)
    return rows


def decide(rows: dict, exp: str):
    """Apply the pre-registered rule. Returns (best_lr or None, needed_runs, flags, table)."""
    tried = sorted(lr for (e, lr) in rows if e == exp)
    table = [rows[(exp, lr)] for lr in tried]
    missing = [lr for lr in GRID if lr not in tried]
    if missing:
        return None, missing, ["grid incomplete"], table

    def loss(lr):
        r = rows[(exp, lr)]
        v = r["val_loss"]
        return math.inf if (r["status"] != "done" or not math.isfinite(v)) else round(v, 4)

    flags = []
    while True:
        tried = sorted(lr for (e, lr) in rows if e == exp)
        best = min(tried, key=lambda lr: (loss(lr), lr))
        if best == tried[0] and best != min(LOWER_EXT + GRID):
            nxt = [lr for lr in LOWER_EXT if lr < best][:1]
            if nxt and nxt[0] not in tried:
                return None, nxt, flags + [f"best {fmt(best)} at the low edge: extend down"], table
        if best == tried[-1] and best != max(UPPER_EXT + GRID):
            nxt = [lr for lr in UPPER_EXT if lr > best][:1]
            if nxt and nxt[0] not in tried:
                return None, nxt, flags + [f"best {fmt(best)} at the high edge: extend up"], table
        if best in (min(LOWER_EXT), max(UPPER_EXT)):
            flags.append("not bracketed (still at the edge after two extensions)")
        if loss(best) == math.inf:
            return None, [], flags + ["every run diverged or failed"], table
        return best, [], flags, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", help="only use runs whose git commit starts with this")
    ap.add_argument("--write-campaign", action="store_true", help="write the phase2b (or extension) campaign")
    args = ap.parse_args()
    t0 = time.time()
    rows = find_runs(args.commit)
    data, verdicts, picks, ext = {}, {}, {}, []
    for exp in MODELS:
        best, need, flags, table = decide(rows, exp)
        data[exp] = {"runs": table, "best_lr": best, "needed": need, "flags": flags}
        print(f"\n== {exp}")
        losses = [r["val_loss"] for r in table if r["status"] == "done" and math.isfinite(r["val_loss"])]
        lo = min(losses) if losses else float("nan")
        for r in table:
            rel = (r["val_loss"] - lo) / lo * 100 if math.isfinite(r["val_loss"]) and losses else float("nan")
            print(f"  lr {fmt(r['lr']):>6s}  val {r['val_loss']:.4f}  ({rel:+.2f}% vs best)  {r['status']:8s}  "
                  f"{r['run']}  @{r['commit']}{'  CODE-DIRTY' if r['code_dirty'] else ''}")
        if best is None:
            verdicts[exp] = f"UNDECIDED: need runs at lr {', '.join(fmt(x) for x in need) or '-'} ({'; '.join(flags)})"
            ext += [(exp, lr) for lr in need]
        else:
            picks[exp] = best
            verdicts[exp] = f"tuned LR {fmt(best)}" + (f" ({'; '.join(flags)})" if flags else "")
        print(f"  -> {verdicts[exp]}")

    if args.write_campaign:
        CAMPAIGNS.mkdir(parents=True, exist_ok=True)
        if ext:
            out = CAMPAIGNS / "phase2a-k7-lr-ext.campaign"
            lines = ["# K7 grid extension, written by experiments/checks/k7_lr_sweep.py (decision rule step 2)"]
            lines += [f"{e} --seed 0 --lr {fmt(lr)} --max-tokens {SWEEP_TOKENS}" for e, lr in ext]
        else:
            out = CAMPAIGNS / "phase2b-replication.campaign"
            tpl = (CAMPAIGNS / "phase2b-replication.template").read_text(encoding="utf-8")
            e10, e1 = fmt(picks["E10"]), fmt(picks["E1"])
            text = tpl.replace("{LR_E10}", e10).replace("{LR_E1}", e1)
            keep = []
            for line in text.splitlines():
                if line.startswith("#IF_E10_NOT_3e-3 "):
                    line = line.split(" ", 1)[1] if picks["E10"] != 3e-3 else "# (skipped: LR_E10 == 3e-3) " + line.split(" ", 1)[1]
                elif line.startswith("#IF_E1_NOT_3e-3 "):
                    line = line.split(" ", 1)[1] if picks["E1"] != 3e-3 else "# (skipped: LR_E1 == 3e-3) " + line.split(" ", 1)[1]
                keep.append(line)
            lines = [f"# Generated by experiments/checks/k7_lr_sweep.py from phase2b-replication.template: "
                     f"LR_E10={e10}, LR_E1={e1}"] + keep
        out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"\nwrote {out.relative_to(ROOT)}")
        data["campaign_written"] = str(out.relative_to(ROOT))
    save("K7", ["E10 control", "C5"], PREDICTION, CRITERION, verdicts, data, t0)


if __name__ == "__main__":
    main()
