"""REPLICATION-stats: the pre-registered SECONDARY statistical layer for Phase 2b (workstream A2). CPU only.

It does NOT replace experiments/checks/replication_analysis.py; it imports it (run discovery, filters, tuned LRs,
the registered decision rules) and adds, on the same qualifying runs:
  * the registered decision rules re-applied, unchanged, to the PRIMARY METRIC (clean-val doc-level bits/char,
    docs with < 80% 13-gram overlap with train, see K9) and, as sensitivity / secondary, to the < 50% subset and to
    the training-time val_loss (contaminated val). The rules are not modified: only the metric they are applied to
    is stated (see experiments/stats/phase2b-analysis-plan.md);
  * paired-seed differences with 95% CIs (df stated), Welch sensitivity CIs, dz / Cohen d, relative gap %;
  * TOST for E11 vs E1 with the registered margin max(1% of mean(E1), 2 x pooled sd);
  * E2 vs E9 and E0 vs E10 as descriptive single-run comparisons (prediction interval, no p-value);
  * the memorisation gap (bits/char on near-dup docs minus on clean docs) per model, E1 vs E10;
  * Holm over the confirmatory family, Benjamini-Hochberg over the exploratory family;
  * variance decomposition across (seed x model) and the minimum detectable effect at n=3.

Per-checkpoint doc-level numbers come from experiments/stats/doclevel/<run>.json (written by score_doclevel.py,
which needs the GPU). If they are missing the clean-metric parts are reported INCOMPLETE and nothing is guessed.

    ~/.venvs/gogeta/bin/python experiments/stats/replication_stats.py              # writes results/REPLICATION-stats.json
    ~/.venvs/gogeta/bin/python experiments/stats/replication_stats.py --no-save
    ~/.venvs/gogeta/bin/python experiments/stats/replication_stats.py --selftest   # synthetic runs only
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "checks"))
import core  # noqa: E402
import replication_analysis as RA  # noqa: E402

DOCLEVEL = HERE / "doclevel"
PRIMARY = "clean80_bpc"
SENS = "clean50_bpc"
SECONDARY_METRIC = "val_loss"
METRICS = (PRIMARY, SENS, SECONDARY_METRIC)
SEEDS = RA.SEEDS3
ALPHA = 0.05
Q_EXPLORATORY = 0.10
PLANNING_SD_VAL_LOSS = 0.004      # the registered "predicted seed sd <= 0.004" (val_loss units)

CLAIMS = ["C5", "C6", "C92", "C93", "E10 control", "E11", "C150", "C151"]
PREDICTION = (
    "Written before Phase 2b results (addendum in PREDICTIONS.md). The registered decision rules are unchanged; "
    "this adds paired CIs, TOST and Holm. Expected, not scored: E1 beats E10 on clean-val bits/char with a paired "
    "95% CI excluding 0 if the true gap is >= ~2%; E11 vs E1 TOST equivalent if the seed sd is <= ~0.002 (clean bpc "
    "units scale accordingly); the memorisation gap (near-dup minus clean bits/char) is negative for every model.")
CRITERION = (
    "Primary decision rule: PREDICTIONS.md 'Replication protocol' criteria 1-5, applied to clean-val bits/char "
    "(< 80% overlap). Confirmatory family (Holm, alpha 0.05): H1 E1 vs E10 paired-seed t (two-sided); H2 E11 vs E1 "
    "TOST (margin max(1%, 2 x pooled sd)). Everything else is exploratory (BH q 0.10) or descriptive. A hypothesis "
    "is 'supported' only if its Holm-adjusted p < 0.05; the original rule decides the claim status either way.")


# ----------------------------------------------------------------------------- doc-level inputs
def load_doclevel(dirpath: pathlib.Path) -> dict:
    out = {}
    for p in sorted(dirpath.glob("*.json")) if dirpath.exists() else []:
        d = RA._jload(p)
        if isinstance(d, dict) and d.get("run") and isinstance(d.get("bpc"), dict):
            out[d["run"]] = d
    return out


def metric_value(run: dict, doclevel: dict, metric: str):
    if metric == "val_loss":
        return run.get("val_loss")
    d = doclevel.get(run["run"])
    if not d:
        return None
    b = d["bpc"]
    return {"clean80_bpc": b.get("clean80"), "clean50_bpc": b.get("clean50"),
            "dup80_bpc": b.get("dup80"), "dup50_bpc": b.get("dup50"), "all_bpc": b.get("all")}.get(metric)


def fin(x):
    return x is not None and math.isfinite(float(x))


# ----------------------------------------------------------------------------- the analysis
def analyse_stats(runs: list[dict], lrs: dict, ref: dict, doclevel: dict) -> dict:
    chosen, excluded = RA.select(runs, lrs)
    lr_unknown = [g for g in ("E10", "E1") if lrs.get(g) is None]

    def series(exp, metric, seeds=None):
        seeds = seeds if seeds is not None else RA.NEEDED[exp]
        out = {}
        for s in seeds:
            r = chosen.get((exp, s))
            v = metric_value(r, doclevel, metric) if r else None
            if fin(v):
                out[s] = float(v)
        return out

    def paired_arrays(ea, eb, metric):
        a, b = series(ea, metric), series(eb, metric)
        common = sorted(set(a) & set(b))
        return common, np.array([a[s] for s in common]), np.array([b[s] for s in common])

    res: dict = {"metrics": {}, "lrs": lrs, "included": {f"{e}/s{s}": r["run"] for (e, s), r in sorted(chosen.items())},
                 "excluded": [{"run": r["run"], "reason": w} for r, w in excluded]}
    if lr_unknown:
        res["status"] = f"INCOMPLETE: tuned LR_{'/'.join(lr_unknown)} unknown (K7 undecided)"

    # ---- A. registered decision rules, unchanged, on each metric ------------------------------------------------
    rule_runs = {}
    for metric in METRICS:
        recs = []
        for r in runs:
            v = metric_value(r, doclevel, metric)
            recs.append({**r, "val_loss": v})
        # runs without a value for this metric drop out of the rule (-> INCOMPLETE) rather than being guessed
        recs = [r for r in recs if fin(r["val_loss"]) or r["exp"] not in RA.NEEDED or r["status"] != "done"]
        a = RA.analyse(recs, lrs, ref)["verdicts"]
        rule_runs[metric] = {k: {"verdict": a[k]["verdict"], "text": a[k]["text"]}
                             for k in ("C5", "E0_parity", "loop_E9_vs_E2", "E11")}
    res["registered_rules_by_metric"] = rule_runs
    res["registered_rules_note"] = (f"Same code and thresholds as replication_analysis.py; metric = {PRIMARY} (PRIMARY), "
                                    f"{SENS} (sensitivity), {SECONDARY_METRIC} (training-time, contaminated val: "
                                    "secondary/descriptive). Registry-value checks (E10 scorecard, E1 reproducibility) "
                                    "are val_loss-specific and stay in REPLICATION.json.")

    # ---- B. paired / Welch / TOST layer, per metric ------------------------------------------------------------------
    for metric in METRICS:
        M: dict = {"role": {PRIMARY: "primary", SENS: "sensitivity", SECONDARY_METRIC: "secondary (contaminated val)"}[metric]}
        # E1 vs E10 (H1)
        seeds, a, b = paired_arrays("E1", "E10", metric)
        if len(seeds) >= 2:
            pd_ = core.paired_diff(a, b)
            M["E1_vs_E10"] = {"seeds": seeds, "paired": pd_, "welch": core.welch(a, b),
                              "sign_flip": core.sign_flip_test(a - b),
                              "protocol": "E1 at LR_E1 vs E10 at LR_E10, 200M tokens, same seed index = same data order "
                                          "(init differs by architecture); a - b < 0 means E1 better (lower loss/bpc)"}
        else:
            M["E1_vs_E10"] = {"verdict": "INCOMPLETE", "note": f"need >= 2 paired seeds, have {len(seeds)}"}
        # E11 vs E1 (H2, TOST)
        seeds, a, b = paired_arrays("E11", "E1", metric)
        if len(seeds) >= 2:
            sp = RA.pooled(float(a.std(ddof=1)), float(b.std(ddof=1)))
            margin = max(0.01 * float(b.mean()), 2 * sp)
            M["E11_vs_E1"] = {"seeds": seeds, "paired": core.paired_diff(a, b),
                              "margin": margin, "margin_is": "1% of mean(E1)" if 0.01 * b.mean() >= 2 * sp else "2 x pooled sd",
                              "tost_paired": core.tost_paired(a, b, margin), "tost_welch_sensitivity": core.tost_welch(a, b, margin),
                              "welch": core.welch(a, b),
                              "protocol": "E11 and E1 share init AND data order at the same seed (same parameter shapes; "
                                          "verified on CPU); only the neg-eig flag differs"}
        else:
            M["E11_vs_E1"] = {"verdict": "INCOMPLETE", "note": f"need >= 2 paired seeds, have {len(seeds)}"}
        # E2 vs E9: single seed each -> descriptive
        e2, e9, e1 = series("E2", metric), series("E9", metric), series("E1", metric)
        if 0 in e2 and 0 in e9:
            d = e2[0] - e9[0]
            item = {"E2": e2[0], "E9": e9[0], "diff_E2_minus_E9": d, "rel_gap_pct": 100 * d / e9[0], "n": 1, "inferential": False,
                    "label": "DESCRIPTIVE ONLY: one run per arm, no CI, no p-value"}
            if len(e1) >= 2:
                s1 = float(np.std(list(e1.values()), ddof=1))
                item.update(noise_sd_of_difference_sqrt2_sE1=math.sqrt(2) * s1, z_vs_E1_seed_noise=d / (math.sqrt(2) * s1) if s1 > 0 else None,
                            note="the noise unit is E1's seed sd (closest architecture), as in the registered criterion 5; "
                                 "FLOPs differ (E2 ~2.4x E9 per token)")
            M["E2_vs_E9"] = item
        else:
            M["E2_vs_E9"] = {"verdict": "INCOMPLETE", "note": "need E2 and E9 seed 0"}
        # E0 vs E10: single run vs 3-seed distribution -> prediction interval
        e0, e10 = series("E0", metric), series("E10", metric)
        if 0 in e0 and len(e10) >= 2:
            v = np.array(list(e10.values()))
            n = len(v)
            tc = core.t_ppf(0.975, n - 1)
            half = tc * float(v.std(ddof=1)) * math.sqrt(1 + 1 / n)
            M["E0_vs_E10"] = {"E0": e0[0], "E10_mean": float(v.mean()), "E10_sd": float(v.std(ddof=1)), "n_E10": n, "df": n - 1,
                              "prediction_interval_95": [float(v.mean()) - half, float(v.mean()) + half],
                              "inside_prediction_interval": bool(abs(e0[0] - v.mean()) <= half),
                              "note": "95% prediction interval for ONE new run from E10's seeds (descriptive; the registered parity band is unchanged)"}
        else:
            M["E0_vs_E10"] = {"verdict": "INCOMPLETE", "note": "need E0 seed 0 and >= 2 E10 seeds (E0 only runs if LR_E10 != 3e-3)"}
        # variance decomposition: seeds x models sharing the data order
        for key, models, reading in (
                ("variance_decomposition_seed_x_model", ("E10", "E1", "E11"),
                 "var_rows = variance shared by all models at one seed; icc_rows near 0 => pairing buys nothing. E1 and E11 share "
                 "init AND data order, E10 shares only data order with them, so this mixed design overstates the data-order share"),
                ("variance_decomposition_E10_E1_data_order_only", ("E10", "E1"),
                 "E10 and E1 share ONLY the data order at a seed (different init): var_rows = data-order variance, "
                 "var_resid = init + architecture-specific response + GPU noise (2 columns, 2 df: indicative only)")):
            cols = [series(e, metric) for e in models]
            common = sorted(set.intersection(*[set(c) for c in cols])) if all(cols) else []
            if len(common) >= 2:
                mat = np.array([[c[s] for c in cols] for s in common])
                M[key] = {"models": list(models), "seeds": common, **core.variance_components(mat), "reading": reading}
            else:
                M[key] = {"verdict": "INCOMPLETE"}
        # MDE at n=3 from the observed sds
        sds = [float(np.std(list(series(e, metric).values()), ddof=1)) for e in ("E10", "E1") if len(series(e, metric)) >= 2]
        if sds:
            sd_pool = math.sqrt(sum(s * s for s in sds) / len(sds))
            ref_mean = float(np.mean(list(series("E10", metric).values()))) if series("E10", metric) else None
            M["mde_n3"] = {"observed_pooled_sd": sd_pool,
                           "two_sample_alpha_0.05": core.min_detectable_effect(sd_pool, 3, "two_sample", ref_mean=ref_mean),
                           "two_sample_alpha_0.025_holm_m2": core.min_detectable_effect(sd_pool, 3, "two_sample", alpha=0.025, ref_mean=ref_mean)}
        res["metrics"][metric] = M

    # ---- C. memorisation gap (needs doc-level numbers) ----------------------------------------------------------------
    gap: dict = {}
    for thr, (dup, clean) in {"80": ("dup80_bpc", "clean80_bpc"), "50": ("dup50_bpc", "clean50_bpc")}.items():
        per = {}
        for e in ("E10", "E1", "E11", "E9", "E2", "E0"):
            d_, c_ = series(e, dup), series(e, clean)
            per[e] = {s: {"gap": d_[s] - c_[s], "ratio": d_[s] / c_[s], "dup_bpc": d_[s], "clean_bpc": c_[s]} for s in d_ if s in c_}
        g = {"per_model": {e: v for e, v in per.items() if v},
             "definition": f"gap = bits/char on docs with >= {thr}% overlap minus bits/char on docs with < {thr}% overlap "
                           "(more negative = repeated docs are much easier = more memorisation/recall benefit)"}
        s10, s1 = per["E10"], per["E1"]
        common = sorted(set(s10) & set(s1))
        if len(common) >= 2:
            a = np.array([s1[s]["gap"] for s in common])
            b = np.array([s10[s]["gap"] for s in common])
            g["E1_vs_E10_gap"] = {"seeds": common, "paired": core.paired_diff(a, b),
                                  "paired_log_ratio": core.paired_diff([math.log(s1[s]["ratio"]) for s in common],
                                                                       [math.log(s10[s]["ratio"]) for s in common], ) if len(common) >= 2 else None}
        else:
            g["E1_vs_E10_gap"] = {"verdict": "INCOMPLETE"}
        gap[thr] = g
    res["memorisation_gap"] = gap
    gci = gap["80"].get("E1_vs_E10_gap", {}).get("paired", {}) if gap["80"].get("E1_vs_E10_gap") else {}
    res["memorisation_confound_flag"] = (None if not gci.get("ci") else bool(gci["ci_excludes_0"]))
    res["memorisation_confound_note"] = ("If True (paired 95% CI of the gap difference excludes 0), the E1-vs-E10 result "
                                         "must be reported with the caveat 'differs in recall of repeated docs' and the < 50% "
                                         "sensitivity subset next to it. It never changes the registered verdict mechanically.")

    # ---- D. multiplicity -----------------------------------------------------------------------------------------------
    P = res["metrics"][PRIMARY]
    h1 = P["E1_vs_E10"].get("paired", {})
    h2 = P["E11_vs_E1"].get("tost_paired", {})
    conf = []
    if h1.get("p") is not None:
        conf.append(("H1 E1 vs E10 (paired t, two-sided)", h1["p"]))
    if h2.get("p_tost") is not None:
        conf.append(("H2 E11 vs E1 equivalence (TOST)", h2["p_tost"]))
    fam = {"alpha": ALPHA, "m_planned": 2, "members_planned": ["H1 E1 vs E10", "H2 E11 vs E1 (TOST)"]}
    if len(conf) == 2:
        hm = core.holm([p for _, p in conf], ALPHA)
        fam["holm"] = [{"hypothesis": n, "p_raw": p, "p_holm": q, "supported": r}
                       for (n, p), q, r in zip(conf, hm["p_adj"], hm["reject"])]
    else:
        fam["holm"] = "INCOMPLETE: need both H1 and H2 p-values (planned m=2 is fixed; a missing member is NOT dropped)"
    res["confirmatory_family"] = fam
    expl = []

    def add(name, p):
        if p is not None:
            expl.append((name, float(p)))
    add("E1 vs E10, memorisation gap (clean<80)", (gap["80"].get("E1_vs_E10_gap", {}).get("paired") or {}).get("p"))
    add("E1 vs E10, clean<50 sensitivity", (res["metrics"][SENS]["E1_vs_E10"].get("paired") or {}).get("p"))
    add("E1 vs E10, val_loss (contaminated)", (res["metrics"][SECONDARY_METRIC]["E1_vs_E10"].get("paired") or {}).get("p"))
    add("E11 vs E1 TOST, clean<50 sensitivity", (res["metrics"][SENS]["E11_vs_E1"].get("tost_paired") or {}).get("p_tost"))
    add("E11 vs E1 TOST, val_loss (contaminated)", (res["metrics"][SECONDARY_METRIC]["E11_vs_E1"].get("tost_paired") or {}).get("p_tost"))
    if expl:
        bh = core.benjamini_hochberg([p for _, p in expl], Q_EXPLORATORY)
        res["exploratory_family"] = {"q": Q_EXPLORATORY, "label": "exploratory: hypothesis-generating, no claim status changes",
                                     "bh": [{"hypothesis": n, "p_raw": p, "p_bh": q, "flag": r}
                                            for (n, p), q, r in zip(expl, bh["p_adj"], bh["reject"])]}
    else:
        res["exploratory_family"] = {"q": Q_EXPLORATORY, "bh": "INCOMPLETE"}

    # ---- E. n = 3 power statement (design-level, independent of results) --------------------------------------------------
    res["power_design"] = {
        "planning_sd_val_loss": PLANNING_SD_VAL_LOSS,
        "two_sample_n3_alpha_0.05": core.min_detectable_effect(PLANNING_SD_VAL_LOSS, 3, "two_sample", ref_mean=0.60),
        "two_sample_n3_alpha_0.025": core.min_detectable_effect(PLANNING_SD_VAL_LOSS, 3, "two_sample", alpha=0.025, ref_mean=0.60),
        "power_at_registered_gap_range_1.5_to_4pct_alpha_0.05": {
            f"{g}%": core.power_t(0.60 * g / 100, PLANNING_SD_VAL_LOSS, 3, "two_sample")["power"] for g in (1.5, 2.0, 3.0, 4.0)},
        "note": "val_loss units, sd = registered 0.004; recompute with the observed sd in metrics.*.mde_n3"}
    res["notes"] = [
        "The original criteria (REPLICATION.json / PREDICTIONS.md) remain the primary decision rule; this file adds estimation "
        "and multiplicity context only.",
        "With n=3 the paired t has df=2 (t_crit 4.30) and the sign-flip test cannot go below p=0.25; both are reported, the CI "
        "is the interpretable object.",
        "Welch (df~4) is shown as a sensitivity analysis: if pairing buys nothing it is the more powerful test.",
        "A single-seed E2 vs E9 is never given a p-value.",
    ]
    return res


# ----------------------------------------------------------------------------- report / IO
def report(res: dict) -> None:
    print(f"tuned LRs: {res['lrs']}   qualifying runs: {len(res['included'])}   excluded: {len(res['excluded'])}")
    if "status" in res:
        print(res["status"])
    for m, d in res["registered_rules_by_metric"].items():
        print(f"\n[registered rules on {m}]")
        for k, v in d.items():
            print(f"  {k:14s} {v['text']}")
    for m, M in res["metrics"].items():
        print(f"\n[{m}: {M['role']}]")
        h = M["E1_vs_E10"]
        if "paired" in h:
            p = h["paired"]
            print(f"  E1-E10 paired n={p['n']} df={p['df']}: {p['mean_diff']:+.5f} ({p['rel_gap_pct']:+.2f}%) CI "
                  f"[{p['ci'][0]:+.5f}, {p['ci'][1]:+.5f}] p={p['p']:.3f} dz={p['dz']:.2f}; Welch df={h['welch']['df']:.1f} "
                  f"CI [{h['welch']['ci'][0]:+.5f}, {h['welch']['ci'][1]:+.5f}]")
        else:
            print("  E1-E10:", h.get("note"))
        h = M["E11_vs_E1"]
        if "tost_paired" in h:
            t = h["tost_paired"]
            print(f"  E11-E1 diff {t['mean_diff']:+.5f} margin {h['margin']:.5f} ({h['margin_is']}) TOST p={t['p_tost']:.3f} "
                  f"equivalent={t['equivalent']} 90% CI [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}]")
        else:
            print("  E11-E1:", h.get("note"))
    f = res["confirmatory_family"]
    print("\n[confirmatory family, Holm]", f["holm"] if isinstance(f["holm"], str) else "")
    if not isinstance(f["holm"], str):
        for r in f["holm"]:
            print(f"  {r['hypothesis']}: p={r['p_raw']:.4f} Holm p={r['p_holm']:.4f} -> {'supported' if r['supported'] else 'not supported'}")
    print("\nmemorisation confound flag:", res["memorisation_confound_flag"])


def _write(check_id, claims, prediction, criterion, verdicts, data, t0):
    try:
        sys.path.insert(0, str(ROOT / "experiments" / "checks"))
        from common import save   # imports torch; present in the main venv
        return save(check_id, claims, prediction, criterion, verdicts, data, t0)
    except ImportError:
        import datetime
        out = RA.ROOT / "experiments" / "checks" / "results" / f"{check_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rec = {"check": check_id, "claims": claims, "prediction": prediction, "criterion": criterion, "verdicts": verdicts,
               "data": data, "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "runtime_s": round(time.time() - t0, 1), "device": "cpu", "note": "written without torch (common.save unavailable)"}
        out.write_text(json.dumps(core._j(rec), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"  -> {out}")
        return out


def summary_verdicts(res: dict) -> dict:
    v = {}
    for m in METRICS:
        r = res["registered_rules_by_metric"][m]
        v[f"C5[{m}]"] = r["C5"]["text"]
        v[f"E11[{m}]"] = r["E11"]["text"]
    f = res["confirmatory_family"]["holm"]
    v["confirmatory_holm"] = f if isinstance(f, str) else "; ".join(
        f"{r['hypothesis']}: Holm p {r['p_holm']:.4f} -> {'supported' if r['supported'] else 'not supported'}" for r in f)
    v["memorisation_confound_flag"] = str(res["memorisation_confound_flag"])
    return v


# ----------------------------------------------------------------------------- selftest (synthetic runs only)
def selftest() -> int:
    fails = []

    def check(label, ok, info=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {label} {info}")
        if not ok:
            fails.append(label)

    ref = {"E1_val": 0.5846, "E1_tol": 0.002, "E1_run": "E1-OLD-s0", "E0_tps": 143083.0}
    rng = np.random.default_rng(7)

    def build(spec, lr=3e-3, doc=True, drop=()):
        """spec: {exp: [clean80 bpc per seed]}; val_loss = bpc*2.5, dup80 = 0.45*clean80 (E1: 0.40 -> memorises more)."""
        td = tempfile.TemporaryDirectory()
        d = pathlib.Path(td.name)
        (d / "doc").mkdir()
        (d / "runs").mkdir()
        for e, vals in spec.items():
            for s, v in enumerate(vals):
                if (e, s) in drop:
                    continue
                name = RA._mk(d / "runs", e, s, lr, v * 2.5,
                              flops={"E10": 96758784.0, "E0": 96758784.0, "E1": 91e6, "E11": 91e6, "E9": 79170960.0, "E2": 192793152.0}[e])
                if doc:
                    k = 0.40 if e in ("E1", "E11") else 0.45
                    (d / "doc" / f"{name}.json").write_text(json.dumps({"run": name, "bpc": {
                        "all": v * 0.7, "clean80": v, "dup80": v * k, "clean50": v * 1.1, "dup50": v * 0.8}}))
        return td, RA.discover(d / "runs"), load_doclevel(d / "doc")

    base = {"E10": [0.2500, 0.2510, 0.2490], "E0": [0.2505], "E1": [0.2400, 0.2410, 0.2390], "E11": [0.2402, 0.2412, 0.2392],
            "E9": [0.2430], "E2": [0.2520]}
    td, runs, doc = build(base)
    r = analyse_stats(runs, {"E10": 3e-3, "E1": 3e-3}, ref, doc)
    P = r["metrics"][PRIMARY]
    p = P["E1_vs_E10"]["paired"]
    exp_d = np.array(base["E1"]) - np.array(base["E10"])
    check("paired mean diff == -0.0100", abs(p["mean_diff"] + 0.01) < 1e-12 and p["n"] == 3 and p["df"] == 2)
    tc = core.t_ppf(0.975, 2)
    check("paired CI by hand", abs(p["ci"][0] - (exp_d.mean() - tc * exp_d.std(ddof=1) / math.sqrt(3))) < 1e-12)
    check("rel gap % about -4%", abs(p["rel_gap_pct"] + 100 * 0.01 / 0.25) < 1e-9)
    check("registered C5 on clean80 VERIFIED (rule unchanged)", r["registered_rules_by_metric"][PRIMARY]["C5"]["verdict"] == "VERIFIED")
    check("same rule on val_loss (= 2.5 x bpc) also VERIFIED", r["registered_rules_by_metric"][SECONDARY_METRIC]["C5"]["verdict"] == "VERIFIED")
    check("E11 'free' by registered rule", r["registered_rules_by_metric"][PRIMARY]["E11"]["verdict"] == "free")
    t = P["E11_vs_E1"]["tost_paired"]
    check("TOST: E11 equivalent to E1 (diff 2e-4, tiny paired sd)", t["equivalent"] and t["df"] == 2, f"p={t['p_tost']:.4f} margin={P['E11_vs_E1']['margin']:.5f}")
    check("margin = max(1%, 2 x pooled sd) = 1% here", P["E11_vs_E1"]["margin_is"] == "1% of mean(E1)")
    check("E2 vs E9 descriptive, no p", P["E2_vs_E9"]["inferential"] is False and "p" not in P["E2_vs_E9"])
    check("E0 inside E10 prediction interval", P["E0_vs_E10"]["inside_prediction_interval"])
    check("sign-flip p = 0.25 at n=3 (cannot reach 0.05)", P["E1_vs_E10"]["sign_flip"]["p"] == 0.25)
    hm = r["confirmatory_family"]["holm"]
    check("Holm m=2 applied", isinstance(hm, list) and len(hm) == 2 and all(x["p_holm"] >= x["p_raw"] for x in hm))
    g = r["memorisation_gap"]["80"]["E1_vs_E10_gap"]["paired"]
    check("memorisation gap: E1 more negative than E10, CI excludes 0 -> confound flag True",
          g["mean_diff"] < 0 and r["memorisation_confound_flag"] is True, f"{g['mean_diff']:+.4f}")
    check("variance decomposition ran (3 seeds x 3 models)", P["variance_decomposition_seed_x_model"]["rows"] == 3)
    check("MDE n=3 reported in sd units ~3.07", abs(P["mde_n3"]["two_sample_alpha_0.05"]["mde_in_sd"] - 3.07) < 0.05)
    check("power design: 1.5% gap at sd 0.004 has power < 0.7, 4% > 0.95", r["power_design"]["power_at_registered_gap_range_1.5_to_4pct_alpha_0.05"]["1.5%"] < 0.7 < 0.95 < r["power_design"]["power_at_registered_gap_range_1.5_to_4pct_alpha_0.05"]["4.0%"],
          str({k: round(v, 2) for k, v in r["power_design"]["power_at_registered_gap_range_1.5_to_4pct_alpha_0.05"].items()}))
    json.dumps(core._j(r), allow_nan=False)
    td.cleanup()

    td, runs, doc = build(base, doc=False)
    r = analyse_stats(runs, {"E10": 3e-3, "E1": 3e-3}, ref, doc)
    check("no doc-level files: primary INCOMPLETE, val_loss still computed",
          r["registered_rules_by_metric"][PRIMARY]["C5"]["verdict"] == "INCOMPLETE" and
          r["registered_rules_by_metric"][SECONDARY_METRIC]["C5"]["verdict"] == "VERIFIED" and
          isinstance(r["confirmatory_family"]["holm"], str))
    td.cleanup()

    td, runs, doc = build(base, drop=[("E1", 2)])
    r = analyse_stats(runs, {"E10": 3e-3, "E1": 3e-3}, ref, doc)
    check("E1 with 2 seeds: paired n=2 df=1 (CI wide), registered C5 INCOMPLETE",
          r["metrics"][PRIMARY]["E1_vs_E10"]["paired"]["df"] == 1 and r["registered_rules_by_metric"][PRIMARY]["C5"]["verdict"] == "INCOMPLETE")
    td.cleanup()

    td, runs, doc = build({**base, "E1": [0.2500, 0.2510, 0.2490]})
    r = analyse_stats(runs, {"E10": 3e-3, "E1": 3e-3}, ref, doc)
    check("identical E1 and E10: registered C5 REFUTED, CI includes 0, E1_vs_E10 not supported by Holm",
          r["registered_rules_by_metric"][PRIMARY]["C5"]["verdict"] == "REFUTED" and not r["metrics"][PRIMARY]["E1_vs_E10"]["paired"]["ci_excludes_0"]
          and r["confirmatory_family"]["holm"][0]["supported"] is False)
    td.cleanup()

    td, runs, doc = build(base)
    r = analyse_stats(runs, {"E10": None, "E1": None}, ref, doc)
    check("LR unknown -> flagged INCOMPLETE", "status" in r and r["registered_rules_by_metric"][PRIMARY]["C5"]["verdict"] == "INCOMPLETE")
    td.cleanup()

    print(f"\nselftest: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="synthetic runs only; writes nothing")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--runs-dir", default=str(RA.DATA / "runs"))
    ap.add_argument("--doclevel-dir", default=str(DOCLEVEL))
    ap.add_argument("--k7", default=str(RA.K7_JSON))
    ap.add_argument("--lr-e10", type=float)
    ap.add_argument("--lr-e1", type=float)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    t0 = time.time()
    lrs = RA.tuned_lrs(args)
    runs = RA.discover(pathlib.Path(args.runs_dir))
    res = analyse_stats(runs, lrs, RA.load_registry(), load_doclevel(pathlib.Path(args.doclevel_dir)))
    report(res)
    if args.no_save:
        print("\n(--no-save: REPLICATION-stats.json not written)")
        return 0
    _write("REPLICATION-stats", CLAIMS, PREDICTION, CRITERION, summary_verdicts(res), core._j(res), t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
