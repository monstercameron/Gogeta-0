"""REPLICATION: the pre-registered Phase 2 verdicts, computed mechanically. CPU-only ANALYSIS script (queue item Q6).

Reads finished run folders ($GOGETA_DATA/runs/<name>/{meta,summary}.json, default ~/fusiondance-data), applies
the criteria committed in experiments/checks/PREDICTIONS.md ("K7" and "Replication protocol (Phase 2b)") and
the E10 / E11 registry predictions (experiments/experiments.json), and prints/saves the verdicts. No judgement
is applied after the results: if a required run is missing the affected verdict is INCOMPLETE.

    ~/.venvs/gogeta/bin/python experiments/checks/replication_analysis.py              # real runs, writes results/REPLICATION.json
    ~/.venvs/gogeta/bin/python experiments/checks/replication_analysis.py --no-save    # same, writes nothing
    ~/.venvs/gogeta/bin/python experiments/checks/replication_analysis.py --selftest   # synthetic runs, all verdict branches

Which runs count: status `done`; full budget (train.max_tokens == 200M, which drops K7's 20M sweep runs);
effective LR (train.lr) equal to the model's tuned LR; code not dirty (git.code_dirty; falls back to git.dirty
if the former was not recorded). LR_E10 / LR_E1 come from results/K7.json (data.<exp>.best_lr) or --lr-e10 /
--lr-e1. LR_E10 applies to E10 and E0; LR_E1 to E1, E11, E9 and E2. If several runs share (experiment, seed) the
latest one by name (= start time) counts. Needed: E10, E1, E11 seeds 0,1,2; E9, E0, E2 seed 0.

Only stdlib + numpy. common.py (which imports torch) is imported lazily, only when a results file is written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
REGISTRY = ROOT / "experiments" / "experiments.json"
K7_JSON = ROOT / "experiments" / "checks" / "results" / "K7.json"

FULL_TOKENS = 200_000_000
SEEDS3 = (0, 1, 2)
NEEDED = {"E10": SEEDS3, "E1": SEEDS3, "E11": SEEDS3, "E9": (0,), "E0": (0,), "E2": (0,)}
LR_GROUP = {"E10": "E10", "E0": "E10", "E1": "E1", "E11": "E1", "E9": "E1", "E2": "E1"}
ORDER = ["E0", "E10", "E1", "E11", "E9", "E2"]
OLD_E1_VALUE, OLD_E1_TOL = 0.5846, 0.002   # criterion 3 (fallbacks; the registry values are used when present)

PREDICTION = (
    "Written before the Phase 2b runs (PREDICTIONS.md 'Replication protocol'; E10/E11 registry predictions). "
    "E10 (3 seeds, tuned LR): mean val loss 0.595-0.615, seed std <= 0.004; E0 at the same LR within E10's mean "
    "+- max(3 std, 0.5%); E1 beats E10 by 1.5-4%; E10 throughput within +-20% of E0; stable. E1 seed std <= 0.004. "
    "E11 (neg-eig GDN) within +-1.0% of E1's mean, 'free for LM'.")
CRITERION = (
    "C5: VERIFIED if mean(E10) - mean(E1) > 2 x pooled sd AND max(E1 seeds) < min(E10 seeds); PARTIAL if "
    "mean(E1) < mean(E10) but either fails; REFUTED if mean(E1) >= mean(E10). Parity: E0 within mean(E10) +- "
    "max(3 s_E10, 0.5% of mean(E10)). Loop: E9 - E2 > 2 sqrt(2) s_E1 -> looping helps; E2 - E9 > 2 sqrt(2) s_E1 "
    "-> looping hurts; else unresolved. E11: free if |mean(E11) - mean(E1)| <= max(1%, 2 x pooled sd), else "
    "helps/hurts by sign. Pooled sd = sqrt((s_A^2 + s_B^2)/2); s = sample std (ddof 1).")
CLAIMS = ["C5", "C6", "C92", "C93", "E10 control", "E11"]


# ----------------------------------------------------------------------------- discovery
def _jload(p: pathlib.Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover(runs_dir: pathlib.Path) -> list[dict]:
    """One record per run folder with a meta.json (any experiment), with the fields the filters need."""
    out = []
    for meta_p in sorted(runs_dir.glob("*/meta.json")):
        meta = _jload(meta_p)
        if not isinstance(meta, dict):
            continue
        cfg = meta.get("config") or {}
        tr = cfg.get("train") or {}
        ov = cfg.get("cli_overrides") or {}
        summ = _jload(meta_p.parent / "summary.json")
        met = (summ or {}).get("metrics") or {}
        git = meta.get("git") or {}
        dirty = git.get("code_dirty") if git.get("code_dirty") is not None else git.get("dirty")
        flops = met.get("flops_per_token_default", cfg.get("flops_per_token_at_default_loops"))
        pc = cfg.get("params") or {}
        out.append({
            "run": meta.get("name", meta_p.parent.name), "exp": meta.get("experiment"), "seed": meta.get("seed"),
            "lr": tr.get("lr", ov.get("lr")), "max_tokens": tr.get("max_tokens", ov.get("max_tokens")),
            "status": (summ or {}).get("status", "no summary (running/killed)"),
            "val_loss": met.get("val_loss"), "val_bpc": met.get("val_bits_per_char"),
            "tokens_per_s": met.get("tokens_per_s"), "flops_per_token": flops,
            "params_ne": met.get("params_non_embedding", pc.get("non_embedding")),
            "code_dirty": dirty, "describe": git.get("describe", ""),
        })
    return out


def lr_eq(a, b) -> bool:
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-9)


def why_excluded(r: dict, lrs: dict) -> str | None:
    """None if the run counts; otherwise the first reason it does not."""
    if r["exp"] not in NEEDED:
        return "experiment not part of the replication"
    if r["status"] != "done":
        return f"status {r['status']}"
    if r["max_tokens"] != FULL_TOKENS:
        return f"not full budget (max_tokens {r['max_tokens']})"
    if r["val_loss"] is None or not math.isfinite(float(r["val_loss"])):
        return "no finite val_loss"
    if r["code_dirty"]:
        return "dirty code"
    want = lrs.get(LR_GROUP[r["exp"]])
    if want is None:
        return f"tuned LR_{LR_GROUP[r['exp']]} unknown"
    if not lr_eq(r["lr"], want):
        return f"lr {r['lr']:g} != tuned {want:g}"
    return None


def select(runs: list[dict], lrs: dict):
    """-> (chosen {(exp, seed): run}, excluded [(run, reason)]). Latest run by name wins a (exp, seed) slot."""
    chosen, excluded = {}, []
    for r in sorted(runs, key=lambda r: r["run"]):
        reason = why_excluded(r, lrs)
        if reason:
            excluded.append((r, reason))
        else:
            k = (r["exp"], r["seed"])
            if k in chosen:
                excluded.append((chosen[k], f"superseded by later run {r['run']}"))
            chosen[k] = r
    return chosen, excluded


# ----------------------------------------------------------------------------- statistics
def stats(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"n": len(a), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
            "min": float(a.min()), "max": float(a.max()), "values": [float(x) for x in a]}


def pooled(sa: float, sb: float) -> float:
    return math.sqrt((sa ** 2 + sb ** 2) / 2)


def missing_for(chosen: dict, exps: list[str]) -> list[str]:
    miss = []
    for e in exps:
        for s in NEEDED[e]:
            if (e, s) not in chosen:
                miss.append(f"{e} seed {s}")
    return miss


def inc(miss: list[str], lrs_missing: list[str]) -> str:
    parts = [f"tuned LR_{g} unknown (K7 undecided)" for g in lrs_missing] + (
        [f"missing/non-qualifying runs: {', '.join(miss)}"] if miss else [])
    return "INCOMPLETE: " + "; ".join(parts)


# ----------------------------------------------------------------------------- the analysis
def analyse(runs: list[dict], lrs: dict, ref: dict) -> dict:
    """runs: discovered records; lrs: {'E10': float|None, 'E1': float|None}; ref: registry reference values."""
    chosen, excluded = select(runs, lrs)
    lr_unknown = [g for g in ("E10", "E1") if lrs.get(g) is None]
    notes: list[str] = []
    V: dict[str, dict] = {}

    def vals(e):
        return [chosen[(e, s)]["val_loss"] for s in NEEDED[e] if (e, s) in chosen]

    S = {e: stats(vals(e)) for e in ORDER if vals(e)}
    f10 = (chosen.get(("E10", 0)) or {}).get("flops_per_token")
    table = []
    for e in ORDER:
        rs = [chosen[(e, s)] for s in NEEDED[e] if (e, s) in chosen]
        fl = next((r["flops_per_token"] for r in rs if r["flops_per_token"]), None)
        pe = next((r["params_ne"] for r in rs if r["params_ne"]), None)
        table.append({"exp": e, "lr": lrs.get(LR_GROUP[e]), "n_needed": len(NEEDED[e]), **(S.get(e) or {"n": 0}),
                      "flops_per_token": fl, "flops_x_E10": (fl / f10) if (fl and f10) else None,
                      "params_non_embedding": pe})

    def verdict(name, exps, fn, extra_lr=()):
        miss = missing_for(chosen, exps)
        need = [g for g in dict.fromkeys(list(extra_lr) + [LR_GROUP[e] for e in exps]) if g in lr_unknown]
        if miss or need:
            V[name] = {"verdict": "INCOMPLETE", "text": inc(miss, need), "missing": miss}
        else:
            V[name] = fn()

    # C5 -----------------------------------------------------------------------------------------
    def c5():
        a, b = S["E1"], S["E10"]                     # a = hybrid, b = control
        sp = pooled(a["sd"], b["sd"])
        diff = b["mean"] - a["mean"]
        c1, c2 = diff > 2 * sp, a["max"] < b["min"]
        v = "REFUTED" if a["mean"] >= b["mean"] else ("VERIFIED" if (c1 and c2) else "PARTIAL")
        return {"verdict": v, "mean_E10": b["mean"], "mean_E1": a["mean"], "diff_E10_minus_E1": diff,
                "gap_pct": diff / b["mean"] * 100, "pooled_sd": sp, "two_pooled_sd": 2 * sp,
                "cond_diff_gt_2sd": bool(c1), "max_E1": a["max"], "min_E10": b["min"], "cond_separated": bool(c2),
                "text": f"{v}: E10 {b['mean']:.4f} - E1 {a['mean']:.4f} = {diff:+.4f} ({diff / b['mean'] * 100:+.2f}%) vs "
                        f"2 x pooled sd {2 * sp:.4f} [{'ok' if c1 else 'fail'}]; max(E1) {a['max']:.4f} < min(E10) "
                        f"{b['min']:.4f} [{'ok' if c2 else 'fail'}]"}
    verdict("C5", ["E10", "E1"], c5)

    # seed noise (reported, not a criterion) ---------------------------------------------------------
    sn = {}
    for e in ("E10", "E1"):
        if e in S and S[e]["n"] >= 2:
            sn[e] = {"sd": S[e]["sd"], "le_0.004": bool(S[e]["sd"] <= 0.004)}
    V["seed_noise"] = {"verdict": "REPORTED" if sn else "INCOMPLETE", "per_model": sn,
                       "text": ("; ".join(f"{e} s={d['sd']:.4f} ({'<=' if d['le_0.004'] else '>'} 0.004)" for e, d in sn.items())
                                or "INCOMPLETE: need >= 2 seeds of E10 / E1")}

    # old-E1 reproducibility (criterion 3) ------------------------------------------------------------
    if lrs.get("E1") is not None and not lr_eq(lrs["E1"], 3e-3):
        V["E1_reproducibility"] = {"verdict": "N/A", "text": "N/A: only applies when LR_E1 = 3e-3"}
    elif lrs.get("E1") is None:
        V["E1_reproducibility"] = {"verdict": "INCOMPLETE", "text": inc([], ["E1"])}
    else:
        r0 = chosen.get(("E1", 0))
        if r0 is None or r0["run"] == ref["E1_run"]:
            V["E1_reproducibility"] = {"verdict": "INCOMPLETE", "text": (
                "INCOMPLETE: needs a NEW E1 seed-0 run at 3e-3 (the only one found is the v0.3.0 run itself)"
                if r0 else "INCOMPLETE: no qualifying E1 seed-0 run")}
        else:
            d = abs(r0["val_loss"] - ref["E1_val"])
            v = "PASS" if d <= ref["E1_tol"] else "FAIL"
            V["E1_reproducibility"] = {"verdict": v, "new": r0["val_loss"], "old": ref["E1_val"], "abs_diff": d,
                                       "text": f"{v}: new E1 s0 {r0['val_loss']:.4f} vs v0.3.0 {ref['E1_val']:.4f}, |diff| {d:.4f} "
                                               f"(tolerance {ref['E1_tol']})"}

    # E0 implementation parity -----------------------------------------------------------------------------
    def parity():
        b, e0 = S["E10"], S["E0"]["mean"]
        half = max(3 * b["sd"], 0.005 * b["mean"])
        lo, hi = b["mean"] - half, b["mean"] + half
        v = "PASS" if lo <= e0 <= hi else "FAIL"
        return {"verdict": v, "E0": e0, "mean_E10": b["mean"], "half_width": half, "lo": lo, "hi": hi,
                "three_sd": 3 * b["sd"], "half_pct": 0.005 * b["mean"],
                "text": f"{v}: E0 {e0:.4f} vs E10 mean {b['mean']:.4f} +- max(3 s_E10 = {3 * b['sd']:.4f}, 0.5% = "
                        f"{0.005 * b['mean']:.4f}) = [{lo:.4f}, {hi:.4f}]"}
    verdict("E0_parity", ["E10", "E0"], parity)

    # the loop question (E9 vs E2) ------------------------------------------------------------------------------
    def loop():
        e9, e2, s1 = S["E9"]["mean"], S["E2"]["mean"], S["E1"]["sd"]
        thr = 2 * math.sqrt(2) * s1
        v = "helps" if e9 - e2 > thr else ("hurts" if e2 - e9 > thr else "unresolved")
        f9 = (chosen.get(("E9", 0)) or {}).get("flops_per_token")
        f2 = (chosen.get(("E2", 0)) or {}).get("flops_per_token")
        fr = f2 / f9 if (f9 and f2) else None
        return {"verdict": v, "E9": e9, "E2": e2, "E9_minus_E2": e9 - e2, "threshold_2sqrt2_sE1": thr, "s_E1": s1,
                "E2_flops_x_E9": fr,
                "text": f"{v}: E9 {e9:.4f} - E2 {e2:.4f} = {e9 - e2:+.4f} vs 2*sqrt(2)*s_E1 = {thr:.4f} "
                        f"(helps if > +thr, hurts if < -thr); E2 spends {fr:.2f}x E9's FLOPs/token" if fr else
                        f"{v}: E9 {e9:.4f} - E2 {e2:.4f} = {e9 - e2:+.4f} vs thr {thr:.4f}"}
    verdict("loop_E9_vs_E2", ["E9", "E2", "E1"], loop)

    # E11 --------------------------------------------------------------------------------------------------------
    def e11():
        a, b = S["E11"], S["E1"]
        sp = pooled(a["sd"], b["sd"])
        diff = a["mean"] - b["mean"]
        band = max(0.01 * b["mean"], 2 * sp)
        v = "free" if abs(diff) <= band else ("helps" if diff < 0 else "hurts")
        return {"verdict": v, "mean_E11": a["mean"], "mean_E1": b["mean"], "diff": diff, "diff_pct": diff / b["mean"] * 100,
                "pooled_sd": sp, "band": band, "band_is": "1%" if 0.01 * b["mean"] >= 2 * sp else "2 x pooled sd",
                "within_1pct_prediction": bool(abs(diff) <= 0.01 * b["mean"]),
                "text": f"{v}: E11 {a['mean']:.4f} - E1 {b['mean']:.4f} = {diff:+.4f} ({diff / b['mean'] * 100:+.2f}%) vs band "
                        f"max(1% = {0.01 * b['mean']:.4f}, 2 x pooled sd = {2 * sp:.4f}) = {band:.4f}"}
    verdict("E11", ["E11", "E1"], e11)

    # E10 scorecard (registry prediction, items 1-5) ---------------------------------------------------------------
    sc: dict[str, dict] = {}
    m10 = missing_for(chosen, ["E10"])
    need10 = [g for g in ("E10",) if g in lr_unknown]

    def item(key, desc, fn, exps):
        miss = missing_for(chosen, exps)
        need = [g for g in dict.fromkeys(LR_GROUP[e] for e in exps) if g in lr_unknown]
        sc[key] = ({"verdict": "INCOMPLETE", "desc": desc, "text": inc(miss, need)} if (miss or need)
                   else {"desc": desc, **fn()})

    def i1():
        b = S["E10"]
        ok = 0.595 <= b["mean"] <= 0.615 and b["sd"] <= 0.004
        return {"verdict": "HIT" if ok else "MISSED",
                "text": f"mean {b['mean']:.4f} (predicted 0.595-0.615), seed sd {b['sd']:.4f} (predicted <= 0.004)"}
    item("1_loss_and_seed_sd", "mean full-val loss 0.595-0.615 and seed sd <= 0.004", i1, ["E10"])
    sc["2_parity_with_E0"] = {"desc": "E0 at the same LR within mean(E10) +- max(3 sd, 0.5%)", "verdict":
                              {"PASS": "HIT", "FAIL": "MISSED"}.get(V["E0_parity"]["verdict"], "INCOMPLETE"),
                              "text": V["E0_parity"]["text"]}

    def i3():
        g = V["C5"]["gap_pct"]
        return {"verdict": "HIT" if 1.5 <= g <= 4.0 else "MISSED",
                "text": f"E1 beats E10 by {g:+.2f}% (predicted 1.5-4%)"}
    item("3_E1_beats_E10_by_1.5_to_4pct", "mean(E1) < mean(E10) by 1.5-4%", i3, ["E10", "E1"])

    def i4():
        tp = float(np.mean([chosen[("E10", s)]["tokens_per_s"] for s in SEEDS3]))
        e0 = (chosen.get(("E0", 0)) or {}).get("tokens_per_s") or ref.get("E0_tps")
        r = tp / e0 - 1
        return {"verdict": "HIT" if abs(r) <= 0.20 else "MISSED", "E10_tokens_per_s": tp, "E0_tokens_per_s": e0,
                "text": f"E10 {tp:,.0f} tok/s vs E0 {e0:,.0f} tok/s ({r * 100:+.1f}%; predicted within +-20%). "
                        f"Informational: registry also quotes '170-250k steady' ({'inside' if 170e3 <= tp <= 250e3 else 'outside'})"}
    if not all(chosen.get(("E10", s), {}).get("tokens_per_s") for s in SEEDS3) and not m10 and not need10:
        sc["4_throughput_within_20pct_of_E0"] = {"desc": "tokens/s within +-20% of E0", "verdict": "INCOMPLETE",
                                                 "text": "INCOMPLETE: tokens_per_s missing from a summary.json"}
    else:
        item("4_throughput_within_20pct_of_E0", "compiled tokens/s within +-20% of E0", i4, ["E10"])
    sc["5_stability"] = {"desc": "no train-loss jumps > 0.5 after warmup", "verdict": "NOT_CHECKED",
                         "text": "NOT_CHECKED: needs per-step train loss (Trackio/logs); summary.json does not carry it. "
                                 "Excluded from the scorecard count."}
    scored = [v["verdict"] for k, v in sc.items() if k != "5_stability"]
    n_hit = scored.count("HIT")
    overall = ("INCOMPLETE" if "INCOMPLETE" in scored else f"{n_hit}/{len(scored)} HIT")
    V["E10_scorecard"] = {"verdict": overall, "items": sc,
                          "text": overall + " (items 1-4; item 5 not checkable here)"}

    notes += [
        "C5/E11 'sample std' is ddof=1 over seeds; pooled sd = sqrt((sA^2+sB^2)/2).",
        "E11 reading: 'free' if |diff| <= max(1% of mean(E1), 2 x pooled sd); outside that band 'helps' (E11 lower) or "
        "'hurts' (E11 higher). Since the band is >= 2 x pooled sd, this equals the registry's helps/hurts wording and "
        "removes its overlap with 'free'.",
        "E10 prediction item 4: scored as +-20% of the E0 run's tokens_per_s (E0 summary if selected, else the registry "
        "value); the '170-250k' band in the same sentence is reported but not scored (it contradicts E0's 143k).",
        "Latest run (by name) wins when several qualifying runs share (experiment, seed).",
        "The loop verdict uses each run's val_loss at its default loops (E2 r=4), the registered headline number.",
    ]
    return {"table": table, "verdicts": V, "stats": S, "excluded": [{"run": r["run"], "reason": w} for r, w in excluded],
            "included": {f"{e}/s{s}": r["run"] for (e, s), r in sorted(chosen.items())}, "notes": notes}


# ----------------------------------------------------------------------------- I/O
def load_registry() -> dict:
    ref = {"E1_val": OLD_E1_VALUE, "E1_tol": OLD_E1_TOL, "E1_run": None, "E0_tps": None}
    reg = _jload(REGISTRY)
    for e in reg if isinstance(reg, list) else []:
        if e.get("id") == "E1":
            ref["E1_val"] = e.get("results", {}).get("val_loss", OLD_E1_VALUE)
            ref["E1_run"] = e.get("results", {}).get("run")
        if e.get("id") == "E0":
            ref["E0_tps"] = e.get("results", {}).get("tokens_per_s")
    return ref


def tuned_lrs(args) -> dict:
    lrs = {"E10": args.lr_e10, "E1": args.lr_e1}
    k7 = _jload(pathlib.Path(args.k7))
    for g in lrs:
        if lrs[g] is None and isinstance(k7, dict):
            b = (k7.get("data", {}).get(g) or {}).get("best_lr")
            lrs[g] = float(b) if b is not None else None
    return lrs


def f4(x):
    return "   -   " if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.4f}"


def report(res: dict, lrs: dict, runs: list[dict]) -> None:
    print(f"tuned LRs: E10/E0 = {lrs['E10'] if lrs['E10'] is not None else 'UNKNOWN'}, "
          f"E1/E11/E9/E2 = {lrs['E1'] if lrs['E1'] is not None else 'UNKNOWN'}")
    n20 = sum(1 for r in runs if r["exp"] in NEEDED and r["max_tokens"] != FULL_TOKENS)
    print(f"{len(runs)} run folders; {len(res['included'])} qualifying; {len(res['excluded'])} excluded "
          f"({n20} not at the 200M budget, e.g. the K7 20M sweep)")
    print("\nExcluded runs:" if res["excluded"] else "\nNo excluded runs.")
    for x in res["excluded"]:
        print(f"  {x['run']:34s} {x['reason']}")
    print("\nQualifying runs:", ", ".join(f"{k}={v}" for k, v in res["included"].items()) or "(none)")
    print("\nTable (final full-val loss, nats/token, 200M tokens)")
    print(f"  {'exp':4s} {'lr':>7s} {'n':>3s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'max':>8s} {'FLOPs xE10':>11s} {'params(non-emb)':>16s}")
    for t in res["table"]:
        lr = f"{t['lr']:g}" if t["lr"] is not None else "?"
        fx = f"{t['flops_x_E10']:.2f}" if t["flops_x_E10"] else "-"
        pe = f"{t['params_non_embedding']:,}" if t["params_non_embedding"] else "-"
        print(f"  {t['exp']:4s} {lr:>7s} {t['n']:>1d}/{t['n_needed']} {f4(t.get('mean')):>8s} {f4(t.get('sd')):>8s} "
              f"{f4(t.get('min')):>8s} {f4(t.get('max')):>8s} {fx:>11s} {pe:>16s}")
    print("\nVerdicts")
    for k, v in res["verdicts"].items():
        print(f"  {k}: {v['text']}")
        if k == "E10_scorecard":
            for ik, iv in v["items"].items():
                print(f"      {ik}: " + (iv["text"] if iv["text"].startswith(iv["verdict"]) else f"{iv['verdict']}: {iv['text']}"))
    print("\nReadings of ambiguous wording:")
    for n in res["notes"]:
        print(f"  - {n}")


# ----------------------------------------------------------------------------- self-test
def _mk(d: pathlib.Path, exp, seed, lr, loss, *, tokens=FULL_TOKENS, status="done", dirty=False, stamp="20261201-000000",
        tps=150000.0, flops=96758784.0, pne=10621824, summary=True):
    name = f"{exp}-{stamp}-s{seed}"
    p = d / name
    p.mkdir(parents=True)
    meta = {"name": name, "experiment": exp, "seed": seed,
            "config": {"train": {"lr": lr, "max_tokens": tokens}, "params": {"non_embedding": pne},
                       "flops_per_token_at_default_loops": flops, "cli_overrides": {"lr": lr}},
            "git": {"commit": "abc", "describe": "v0.4.0", "dirty": False, "code_dirty": dirty}}
    (p / "meta.json").write_text(json.dumps(meta))
    if summary:
        (p / "summary.json").write_text(json.dumps({**meta, "status": status, "metrics": {
            "val_loss": loss, "val_bits_per_char": loss * 0.39, "tokens_per_s": tps, "flops_per_token_default": flops,
            "params_non_embedding": pne}}))
    return name


def selftest() -> int:
    ref = {"E1_val": 0.5846, "E1_tol": 0.002, "E1_run": "E1-OLD-s0", "E0_tps": 143083.0}
    fails = []

    def check(label, got, want):
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {got}" + ("" if ok else f"   (expected {want})"))
        if not ok:
            fails.append(label)

    def scenario(title, spec, lrs=None, extra=None, *, lr=3e-3):
        """spec: {exp: [losses for seed 0,1,...]}; returns the analysis."""
        print(f"\n-- {title}")
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            for e, ls in spec.items():
                for s, v in enumerate(ls):
                    fl = {"E10": 96758784.0, "E0": 96758784.0, "E1": 91e6, "E11": 91e6, "E9": 79170960.0, "E2": 192793152.0}[e]
                    _mk(d, e, s, lr, v, flops=fl, tps=143000.0 if e == "E0" else 160000.0)
            if extra:
                extra(d)
            runs = discover(d)
            return analyse(runs, lrs or {"E10": lr, "E1": lr}, ref)

    base = {"E10": [0.6050, 0.6060, 0.6040], "E0": [0.6055], "E1": [0.5850, 0.5860, 0.5840],
            "E11": [0.5855, 0.5865, 0.5845], "E9": [0.5900], "E2": [0.6190]}

    r = scenario("all pass: C5 VERIFIED, parity PASS, loop hurts, E11 free", base)
    V = r["verdicts"]
    check("C5", V["C5"]["verdict"], "VERIFIED")
    check("E0 parity", V["E0_parity"]["verdict"], "PASS")
    check("loop (E2 - E9 = 0.029 > 2sqrt2*0.001)", V["loop_E9_vs_E2"]["verdict"], "hurts")
    check("E11", V["E11"]["verdict"], "free")
    check("scorecard", V["E10_scorecard"]["verdict"], "3/4 HIT" if V["C5"]["gap_pct"] < 1.5 else "4/4 HIT")
    check("table has E10 FLOPs x1.00 / E2 x1.99", (round(r["table"][1]["flops_x_E10"], 2), round(r["table"][5]["flops_x_E10"], 2)), (1.0, 1.99))

    r = scenario("C5 PARTIAL: E1 mean better but seeds overlap", {**base, "E1": [0.5990, 0.6010, 0.6075]})
    check("C5", r["verdicts"]["C5"]["verdict"], "PARTIAL")
    r = scenario("C5 PARTIAL: separated but gap < 2 pooled sd", {**base, "E10": [0.610, 0.600, 0.620], "E1": [0.595, 0.585, 0.605]})
    check("C5", (r["verdicts"]["C5"]["verdict"], r["verdicts"]["C5"]["cond_separated"]), ("PARTIAL", False))
    r = scenario("C5 REFUTED: E1 not better", {**base, "E1": [0.6060, 0.6070, 0.6050]})
    check("C5", r["verdicts"]["C5"]["verdict"], "REFUTED")
    r = scenario("C5 INCOMPLETE: E1 has only 2 seeds", {**base, "E1": [0.5850, 0.5860]})
    check("C5", r["verdicts"]["C5"]["verdict"], "INCOMPLETE")
    check("E11 also INCOMPLETE (needs E1 x3)", r["verdicts"]["E11"]["verdict"], "INCOMPLETE")
    check("loop INCOMPLETE (needs E1 sd)", r["verdicts"]["loop_E9_vs_E2"]["verdict"], "INCOMPLETE")
    check("parity unaffected (E10 x3, E0 present)", r["verdicts"]["E0_parity"]["verdict"], "PASS")

    r = scenario("parity FAIL: E0 far from E10", {**base, "E0": [0.6300]})
    check("E0 parity", r["verdicts"]["E0_parity"]["verdict"], "FAIL")
    r = scenario("E11 hurts (+3%)", {**base, "E11": [0.6030, 0.6040, 0.6020]})
    check("E11", r["verdicts"]["E11"]["verdict"], "hurts")
    r = scenario("E11 helps (-3%)", {**base, "E11": [0.5670, 0.5680, 0.5660]})
    check("E11", r["verdicts"]["E11"]["verdict"], "helps")
    r = scenario("E11 free via wide noise (2 pooled sd > 1%)", {**base, "E1": [0.575, 0.585, 0.595], "E11": [0.595, 0.605, 0.615]})
    check("E11 (diff 0.020 = 3.4%, band 2*0.010=0.020)", r["verdicts"]["E11"]["verdict"], "free")
    r = scenario("loop helps: E9 much worse than E2", {**base, "E9": [0.6400], "E2": [0.6190]})
    check("loop", r["verdicts"]["loop_E9_vs_E2"]["verdict"], "helps")
    r = scenario("loop unresolved: |E9 - E2| under threshold", {**base, "E1": [0.575, 0.585, 0.595], "E9": [0.6100], "E2": [0.6190]})
    check("loop (diff 0.009 < 2sqrt2*0.010)", r["verdicts"]["loop_E9_vs_E2"]["verdict"], "unresolved")

    def junk(d):   # none of these may count
        _mk(d, "E10", 0, 3e-3, 0.1, tokens=20_000_000, stamp="20261201-000100")        # K7 sweep run
        _mk(d, "E1", 0, 6e-3, 0.1, stamp="20261201-000101")                            # other LR (E1 tuned = 3e-3 here)
        _mk(d, "E1", 1, 3e-3, 0.1, dirty=True, stamp="20261201-000102")                # dirty code
        _mk(d, "E10", 1, 3e-3, 0.1, status="failed", stamp="20261201-000103")          # failed
        _mk(d, "E10", 2, 3e-3, 0.1, summary=False, stamp="20261201-000104")            # still running
    r = scenario("filters: 20M, wrong LR, dirty, failed and running runs are excluded", base, extra=junk)
    check("C5 unchanged", r["verdicts"]["C5"]["verdict"], "VERIFIED")
    check("5 excluded", len(r["excluded"]), 5)
    check("E10 mean unchanged", round(r["stats"]["E10"]["mean"], 4), 0.6050)

    r = scenario("LR unknown (K7 undecided): everything INCOMPLETE", base, lrs={"E10": None, "E1": None})
    check("all INCOMPLETE", sorted({v["verdict"] for k, v in r["verdicts"].items() if k not in ("seed_noise", "E1_reproducibility")}), ["INCOMPLETE"])

    r = scenario("tuned LR 2e-3 != 3e-3 (E1 reproducibility N/A)", base, lrs={"E10": 2e-3, "E1": 2e-3}, lr=2e-3)
    check("repro", r["verdicts"]["E1_reproducibility"]["verdict"], "N/A")
    check("C5 at the new LR", r["verdicts"]["C5"]["verdict"], "VERIFIED")

    def newer_e1(d):
        _mk(d, "E1", 0, 3e-3, 0.5855, stamp="20270101-000000")
    r = scenario("old-E1 reproduction, LR 3e-3: later E1 s0 wins and is within 0.002 of 0.5846", base, extra=newer_e1)
    check("repro", r["verdicts"]["E1_reproducibility"]["verdict"], "PASS")
    r = scenario("old-E1 reproduction FAIL", base, extra=lambda d: _mk(d, "E1", 0, 3e-3, 0.5900, stamp="20270101-000000"))
    check("repro", r["verdicts"]["E1_reproducibility"]["verdict"], "FAIL")
    r = scenario("scorecard misses: E10 mean outside 0.595-0.615", {**base, "E10": [0.700, 0.701, 0.699], "E0": [0.700], "E1": [0.585, 0.586, 0.584]})
    check("item 1", r["verdicts"]["E10_scorecard"]["items"]["1_loss_and_seed_sd"]["verdict"], "MISSED")

    print(f"\nselftest: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    return 0 if not fails else 1


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", "--dry-run", action="store_true", dest="selftest",
                    help="run on synthetic run summaries covering every verdict branch; writes nothing")
    ap.add_argument("--no-save", action="store_true", help="do not write results/REPLICATION.json")
    ap.add_argument("--runs-dir", default=str(DATA / "runs"))
    ap.add_argument("--k7", default=str(K7_JSON), help="K7.json with the tuned LRs")
    ap.add_argument("--lr-e10", type=float, help="override the tuned LR of E10/E0")
    ap.add_argument("--lr-e1", type=float, help="override the tuned LR of E1/E11/E9/E2")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    t0 = time.time()
    lrs = tuned_lrs(args)
    runs = discover(pathlib.Path(args.runs_dir))
    res = analyse(runs, lrs, load_registry())
    report(res, lrs, runs)
    if args.no_save:
        print("\n(--no-save: results/REPLICATION.json not written)")
        return 0
    from common import save   # lazy: common.py imports torch (CPU-only here)
    verdicts = {k: v["text"] for k, v in res["verdicts"].items()}
    data = {"lrs": lrs, "table": res["table"], "verdicts_detail": res["verdicts"], "included": res["included"],
            "excluded": res["excluded"], "notes": res["notes"]}
    save("REPLICATION", CLAIMS, PREDICTION, CRITERION, verdicts, data, t0)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    sys.exit(main())
