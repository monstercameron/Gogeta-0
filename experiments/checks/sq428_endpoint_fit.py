"""SQ428 (diagnostic, not confirmatory; zero GPU): is the 200M endpoint of the A2s-vs-E27 gap near its asymptote?

Each arm's in-training eval curve (launcher logs of the 200M runs on the fleet pulls: `eval step N: val/loss`, 128 val sequences of code-py-v2, every 250 steps, 32,768 tokens
per step) for A2s and E27, seeds 2, 3, 4 (paired by seed), is fitted per seed to L(N) = a + b * N^(-c) (N in millions of tokens; scipy curve_fit from several starting points, bounds a > 0,
b > 0, 0.02 <= c <= 3). Primary fit range: evals from step 750 (24.6M tokens) to the end; sensitivity: from step 2000 (65.5M tokens). Reported per seed and pooled (mean and sd of
the per-seed gaps; also the fit to the seed-mean curves): the fitted A2s-vs-E27 gap at N = 200M tokens and at the asymptote a, as relative change expm1(log a_A2s - log a_E27) in %;
fit residuals (RMS in nats and relative to the loss); the observed gap at the last eval.
Decision rule (fixed in the request): if the fitted asymptote gap differs from -16.7% (C209, clean80) by more than 1.2 points (the C209 95% CI half-width), report "C209 needs a
horizon qualifier"; otherwise "200M endpoint near its limit". Caveat for the reading: this is the in-training val loss (128 sequences), whose observed 200M gap (SQ395: -12 to -13.5%) is
smaller than the clean80 gap, so the fitted asymptote is also compared with the fitted 200M gap of the same metric (reported as an extra, not part of the rule). Both runs follow a cosine schedule
to 200M tokens, so the curve's shape is partly the schedule, not a pure data-scaling law: the asymptote is an extrapolation of the schedule-shaped curve.

Run (WSL, venv gogeta-eval): python experiments/checks/sq428_endpoint_fit.py -> results/sq428.json
"""
import glob
import json
import math
import os
import re
import subprocess
import warnings

import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOME = os.path.expanduser("~/fusiondance-data")
TOK = 32768
LR = {"A2s": 0.006, "E27": 0.006}
SEEDS = (2, 3, 4)
C209 = -16.7
HALF_WIDTH = 1.2


def parse(path):
    txt = open(path, errors="replace").read()
    run = re.search(r"^run: (([A-Za-z0-9]+)-\d+-\d+-s(\d+))", txt, re.M)
    steps = re.search(r"steps (\d+) x (\d+) tok", txt)
    if not (run and steps) or "done:" not in txt or int(steps.group(1)) != 6104:
        return None
    ev = {int(a): float(b) for a, b in re.findall(r"eval step (\d+): val/loss ([0-9.]+)", txt)}
    return run.group(1), run.group(2), int(run.group(3)), ev if len(ev) >= 20 else None


def lr_of(run):
    for mp in glob.glob(f"{HOME}/cloud/fleet/*/fusiondance-data/runs/{run}/meta.json") + glob.glob(f"{HOME}/runs/{run}/meta.json"):
        return json.load(open(mp))["config"]["train"]["lr"]
    return None


def law(N, a, b, c):
    return a + b * np.power(N, -c)


def fit(steps, loss):
    N = np.array(steps, float) * TOK / 1e6
    y = np.array(loss, float)
    best = None
    for c0 in (0.2, 0.5, 1.0):
        for a0 in (0.5 * y.min(), 0.9 * y.min()):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, _ = curve_fit(law, N, y, p0=[a0, (y[0] - a0) * N[0] ** c0, c0], bounds=([1e-6, 1e-9, 0.02], [10, 1e9, 3.0]), maxfev=20000)
            except Exception:
                continue
            r = y - law(N, *p)
            sse = float((r ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, p, r)
    if best is None:
        return None
    sse, p, r = best
    return {"a": float(p[0]), "b": float(p[1]), "c": float(p[2]), "rms_nats": float(math.sqrt(sse / len(y))), "rms_rel": float(math.sqrt(np.mean((r / y) ** 2))),
            "L200": float(law(200.0 * TOK / 32768 * 32768 / 1e6 * 1.0, *p)) if False else float(law(6103 * TOK / 1e6, *p)), "n_points": len(y)}


def pct(a, b):
    return 100 * math.expm1(math.log(a) - math.log(b))


def main():
    paths = sorted(set(glob.glob(f"{HOME}/logs/*.log") + glob.glob(f"{HOME}/cloud/fleet/*/fusiondance-data/logs/*.log")))
    curves = {}
    for p in paths:
        r = parse(p)
        if r and r[3] and r[1] in LR and r[2] in SEEDS and "lrhi" not in p and "lrlo" not in p:
            lr = lr_of(r[0])
            if lr is not None and abs(lr - LR[r[1]]) < 1e-12:
                curves.setdefault((r[1], r[2]), r[3])
    missing = [(a, s) for a in LR for s in SEEDS if (a, s) not in curves]
    out = {"check": "SQ428", "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "status": "diagnostic, not confirmatory", "law": "L(N) = a + b N^-c, N in millions of tokens", "seeds": list(SEEDS), "missing_curves": [list(m) for m in missing], "fits": {}}
    if missing:
        out["error"] = f"missing in-training eval curves: {missing}"
        json.dump(out, open(os.path.join(ROOT, "experiments/checks/results/sq428.json"), "w"), indent=1)
        print(out["error"])
        return
    for name, lo in (("from_step_750", 750), ("from_step_2000", 2000)):
        res = {"per_seed": {}}
        gaps_inf, gaps_200, gaps_obs = [], [], []
        for s in SEEDS:
            f = {}
            for a in LR:
                ev = curves[(a, s)]
                steps = [t for t in sorted(ev) if t >= lo]
                f[a] = fit(steps, [ev[t] for t in steps])
                f[a]["last_eval_loss"] = ev[max(ev)]
            g_inf, g_200, g_obs = pct(f["A2s"]["a"], f["E27"]["a"]), pct(f["A2s"]["L200"], f["E27"]["L200"]), pct(f["A2s"]["last_eval_loss"], f["E27"]["last_eval_loss"])
            res["per_seed"][str(s)] = {"A2s": f["A2s"], "E27": f["E27"], "gap_asymptote_pct": g_inf, "gap_at_200M_fitted_pct": g_200, "gap_at_200M_observed_pct": g_obs}
            gaps_inf.append(g_inf), gaps_200.append(g_200), gaps_obs.append(g_obs)
        # fit to the seed-mean curves
        mean = {}
        for a in LR:
            steps = [t for t in sorted(curves[(a, SEEDS[0])]) if t >= lo]
            mean[a] = fit(steps, [float(np.mean([curves[(a, s)][t] for s in SEEDS])) for t in steps])
        res["mean_curve_fit"] = {"A2s": mean["A2s"], "E27": mean["E27"], "gap_asymptote_pct": pct(mean["A2s"]["a"], mean["E27"]["a"]), "gap_at_200M_fitted_pct": pct(mean["A2s"]["L200"], mean["E27"]["L200"])}
        sd = lambda v: float(np.std(v, ddof=1))
        res["pooled"] = {"gap_asymptote_pct_mean": float(np.mean(gaps_inf)), "gap_asymptote_pct_sd": sd(gaps_inf), "gap_at_200M_fitted_pct_mean": float(np.mean(gaps_200)),
                         "gap_at_200M_fitted_pct_sd": sd(gaps_200), "gap_at_200M_observed_pct_mean": float(np.mean(gaps_obs)), "per_seed_asymptote": gaps_inf}
        d = float(np.mean(gaps_inf)) - C209
        at_bound = [res["per_seed"][str(sd_)][a]["a"] < 1e-3 for sd_ in SEEDS for a in LR]
        res["asymptote_identified"] = not any(at_bound)
        res["n_fits_with_a_at_lower_bound"] = int(sum(at_bound))
        res["decision_rule"] = {"asymptote_gap_minus_C209_points": d, "result_mechanical": "C209 needs a horizon qualifier" if abs(d) > HALF_WIDTH else "200M endpoint near its limit",
                                "result": ("INCONCLUSIVE: the asymptote is not identified (a sits at its lower bound in %d of 6 fits: the curve is fitted by a pure power law, so a -> 0 and the asymptote gap is 0 by construction); "
                                           "the mechanical rule output is therefore not evidence" % sum(at_bound)) if any(at_bound) else ("C209 needs a horizon qualifier" if abs(d) > HALF_WIDTH else "200M endpoint near its limit")}
        # horizon trend without an asymptote: fitted gap at larger N (mean over seeds)
        ext = {}
        for NM in (200, 400, 1000, 10000, 100000):
            g = []
            for sd_ in SEEDS:
                r_ = res["per_seed"][str(sd_)]
                la, le = [r_[a]["a"] + r_[a]["b"] * NM ** (-r_[a]["c"]) for a in ("A2s", "E27")]
                g.append(pct(la, le))
            ext[str(NM)] = float(np.mean(g))
        res["extrapolated_gap_pct_by_Mtokens"] = ext
        res["extra_same_metric"] = {"asymptote_minus_fitted_200M_points": float(np.mean(gaps_inf) - np.mean(gaps_200))}
        out["fits"][name] = res
        print(name, json.dumps({k: res[k] for k in ("pooled", "decision_rule", "extra_same_metric")}, indent=1))
        for s in SEEDS:
            r = res["per_seed"][str(s)]
            print("  seed", s, "asym %.2f%% fitted200 %.2f%% observed %.2f%%" % (r["gap_asymptote_pct"], r["gap_at_200M_fitted_pct"], r["gap_at_200M_observed_pct"]),
                  "A2s a=%.3f c=%.2f rms=%.4f" % (r["A2s"]["a"], r["A2s"]["c"], r["A2s"]["rms_nats"]), "E27 a=%.3f c=%.2f rms=%.4f" % (r["E27"]["a"], r["E27"]["c"], r["E27"]["rms_nats"]))
    json.dump(out, open(os.path.join(ROOT, "experiments/checks/results/sq428.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
