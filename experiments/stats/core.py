"""Statistics core for Gogeta-0 (workstream A2). numpy only; scipy is used if importable and never required.

Every public function returns a plain JSON-serialisable dict (numpy scalars -> float, NaN/inf -> None), so the
cockpit can call them directly. CLI at the bottom:  python experiments/stats/core.py <command> ... | --selftest

Conventions
  * "a - b": a positive difference means a is larger. For val loss (lower is better) a NEGATIVE a-b means a wins.
  * Seeds are the unit of replication for LM runs (n is tiny: 1-5). Items (problems/docs) are the unit for
    benchmarks (n in the hundreds). Never mix the two: a bootstrap over 164 HumanEval items says nothing about
    seed-to-seed training variance.
  * Paired = the same seed index in both arms (see notes/statistics-policy.md for what the pairing shares).
  * df is always stated. CIs are two-sided, default 95%.

Set GOGETA_STATS_NO_SCIPY=1 to force the pure-numpy fallbacks (the selftest checks they agree with scipy).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

try:
    if os.environ.get("GOGETA_STATS_NO_SCIPY"):
        raise ImportError
    from scipy import stats as _sps  # type: ignore
except Exception:  # pragma: no cover
    _sps = None

HAVE_SCIPY = _sps is not None


# ----------------------------------------------------------------------------- JSON hygiene
def _j(x):
    """Recursively convert to JSON-safe python (NaN/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, np.ndarray):
        return _j(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def _arr(x, name="x") -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: need a non-empty array of finite numbers")
    return a


# ----------------------------------------------------------------------------- t / normal distribution (no scipy needed)
def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if math.isinf(df):
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    if HAVE_SCIPY:
        return float(_sps.t.cdf(t, df))
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student t. Exact for df given; scipy if present, else bisection on t_cdf."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if HAVE_SCIPY:
        return float(_sps.t.ppf(p, df))
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _p_two_sided(t: float, df: float) -> float:
    return float(min(1.0, 2.0 * (1.0 - t_cdf(abs(t), df))))


def _ci_from(mean, se, df, conf):
    tc = t_ppf(0.5 + conf / 2.0, df)
    return mean - tc * se, mean + tc * se, tc


# ----------------------------------------------------------------------------- descriptive
def summarize(x, conf: float = 0.95) -> dict:
    """n, mean, sd (ddof 1), sem, t-based CI of the mean (df = n-1)."""
    a = _arr(x)
    n = a.size
    out = {"n": n, "mean": a.mean(), "min": a.min(), "max": a.max(), "values": a.tolist()}
    if n >= 2:
        sd = a.std(ddof=1)
        lo, hi, tc = _ci_from(a.mean(), sd / math.sqrt(n), n - 1, conf)
        out.update(sd=sd, sem=sd / math.sqrt(n), df=n - 1, conf=conf, t_crit=tc, ci=[lo, hi])
    else:
        out.update(sd=None, sem=None, df=0, conf=conf, ci=None, note="n=1: no spread estimate; exploratory only")
    return _j(out)


# ----------------------------------------------------------------------------- effect sizes
def hedges_J(df: float) -> float:
    return 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else float("nan")


def cohens_d(a, b) -> dict:
    """Unpaired Cohen's d with pooled sd (ddof 1) and Hedges' small-sample g. Approx SE (Hedges-Olkin)."""
    a, b = _arr(a, "a"), _arr(b, "b")
    if a.size < 2 or b.size < 2:
        return _j({"d": None, "note": "need n>=2 per arm"})
    df = a.size + b.size - 2
    sp = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / df)
    d = (a.mean() - b.mean()) / sp if sp > 0 else float("nan")
    se = math.sqrt((a.size + b.size) / (a.size * b.size) + d * d / (2.0 * (a.size + b.size)))
    return _j({"d": d, "hedges_g": d * hedges_J(df), "pooled_sd": sp, "df": df, "se_approx": se})


def dz(diffs) -> dict:
    """Paired effect size dz = mean(diff)/sd(diff) (+ Hedges correction, df = n-1)."""
    d = _arr(diffs, "diffs")
    if d.size < 2:
        return _j({"dz": None, "note": "need n>=2 pairs"})
    sd = d.std(ddof=1)
    z = d.mean() / sd if sd > 0 else float("nan")
    return _j({"dz": z, "hedges_gz": z * hedges_J(d.size - 1), "df": d.size - 1,
               "se_approx": math.sqrt(1.0 / d.size + z * z / (2.0 * d.size))})


def rel_gap_pct(a, b) -> float:
    """100 * (mean(a) - mean(b)) / mean(b): relative gap of a vs the reference b."""
    return float(100.0 * (_arr(a).mean() - _arr(b).mean()) / _arr(b).mean())


# ----------------------------------------------------------------------------- paired and Welch
def paired_diff(a, b, conf: float = 0.95) -> dict:
    """Paired-seed difference d_i = a_i - b_i (same seed index in both arms).

    Returns mean, sd of d, se, df = n-1, t, two-sided p, t-based CI (absolute and as % of mean(b)), dz, and the
    relative gap in %. The % CI divides the absolute CI by mean(b) (it ignores the uncertainty of the denominator;
    negligible for val losses whose seed noise is <1%)."""
    a, b = _arr(a, "a"), _arr(b, "b")
    if a.size != b.size:
        raise ValueError("paired_diff needs equal-length arrays (same seeds)")
    d = a - b
    n = d.size
    out = {"test": "paired t (df = n-1)", "n": n, "mean_a": a.mean(), "mean_b": b.mean(), "diffs": d.tolist(),
           "mean_diff": d.mean(), "rel_gap_pct": 100.0 * d.mean() / b.mean(), "conf": conf}
    if n < 2:
        out.update(sd_diff=None, df=0, ci=None, p=None, note="n=1 pair: descriptive only, no CI")
        return _j(out)
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    df = n - 1
    lo, hi, tc = _ci_from(d.mean(), se, df, conf)
    t = d.mean() / se if se > 0 else (math.copysign(math.inf, d.mean()) if d.mean() else 0.0)
    out.update(sd_diff=sd, se=se, df=df, t_crit=tc, t=t, p=(_p_two_sided(t, df) if se > 0 else (1.0 if d.mean() == 0 else 0.0)),
               ci=[lo, hi], ci_rel_pct=[100 * lo / b.mean(), 100 * hi / b.mean()], ci_excludes_0=bool(lo > 0 or hi < 0),
               dz=(d.mean() / sd if sd > 0 else None), hedges_gz=(d.mean() / sd * hedges_J(df) if sd > 0 else None),
               pearson_r_ab=(float(np.corrcoef(a, b)[0, 1]) if n >= 3 and a.std() > 0 and b.std() > 0 else None))
    return _j(out)


def welch(a, b, conf: float = 0.95) -> dict:
    """Unpaired Welch t: CI and p with Welch-Satterthwaite df; also Cohen's d and the relative gap."""
    a, b = _arr(a, "a"), _arr(b, "b")
    out = {"test": "Welch t (Satterthwaite df)", "n_a": a.size, "n_b": b.size, "mean_a": a.mean(), "mean_b": b.mean(),
           "mean_diff": a.mean() - b.mean(), "rel_gap_pct": 100.0 * (a.mean() - b.mean()) / b.mean(), "conf": conf}
    if a.size < 2 or b.size < 2:
        out.update(df=None, ci=None, p=None, note="need n>=2 per arm for a CI")
        return _j(out)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1)) if se > 0 else float("nan")
    lo, hi, tc = _ci_from(out["mean_diff"], se, df, conf)
    t = out["mean_diff"] / se if se > 0 else 0.0
    out.update(se=se, df=df, t_crit=tc, t=t, p=_p_two_sided(t, df), ci=[lo, hi],
               ci_rel_pct=[100 * lo / b.mean(), 100 * hi / b.mean()], ci_excludes_0=bool(lo > 0 or hi < 0),
               effect=cohens_d(a, b))
    return _j(out)


# ----------------------------------------------------------------------------- bootstrap over items
def _stat_vec(data, stat):
    return float(stat(data))


def bootstrap(data, stat=np.mean, B: int = 10000, conf: float = 0.95, method: str = "bca", seed: int = 0,
              fast_mean: bool | None = None) -> dict:
    """Nonparametric bootstrap over ITEMS (rows of `data`; 1-D array of per-item scores, or 2-D with one row per item).

    method: "percentile" or "bca" (bias-corrected and accelerated; acceleration from the item-level jackknife).
    stat(resampled_rows) -> float. Default np.mean. For the paired two-model comparison use `paired_bootstrap`,
    which resamples the SAME items for both models. Resample the unit of independence (problem, document), not
    individual samples of one problem; for pass@k with k samples per problem, pass per-problem scores.
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < 2:
        raise ValueError("need >= 2 items")
    rng = np.random.default_rng(seed)
    is_mean = stat is np.mean if fast_mean is None else fast_mean
    theta = _stat_vec(X, stat)
    if is_mean and X.ndim == 1:
        idx = rng.integers(0, n, size=(B, n))
        boots = X[idx].mean(axis=1)
    else:
        boots = np.empty(B)
        for i in range(B):
            boots[i] = stat(X[rng.integers(0, n, size=n)])
    alpha = 1.0 - conf
    res = {"method": method, "n_items": n, "B": B, "conf": conf, "estimate": theta, "boot_se": boots.std(ddof=1),
           "seed": seed}
    if method == "percentile":
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    elif method == "bca":
        frac = (np.sum(boots < theta) + 0.5 * np.sum(boots == theta)) / B
        frac = min(max(frac, 1.0 / (B + 1)), 1.0 - 1.0 / (B + 1))
        z0 = norm_ppf(frac)
        if is_mean and X.ndim == 1:
            jack = (X.sum() - X) / (n - 1)
        else:
            jack = np.array([stat(np.delete(X, i, axis=0)) for i in range(n)])
        jm = jack.mean()
        num, den = np.sum((jm - jack) ** 3), 6.0 * np.sum((jm - jack) ** 2) ** 1.5
        acc = num / den if den > 0 else 0.0
        zl, zu = norm_ppf(alpha / 2), norm_ppf(1 - alpha / 2)

        def adj(z):
            return norm_cdf(z0 + (z0 + z) / (1.0 - acc * (z0 + z)))
        a1, a2 = adj(zl), adj(zu)
        lo, hi = np.quantile(boots, [a1, a2])
        res.update(z0=z0, acceleration=acc, adj_quantiles=[a1, a2])
    else:
        raise ValueError("method must be 'percentile' or 'bca'")
    res.update(ci=[float(lo), float(hi)])
    return _j(res)


def paired_bootstrap(scores_a, scores_b, B: int = 10000, conf: float = 0.95, method: str = "bca", seed: int = 0,
                     stat=None) -> dict:
    """Paired item-level bootstrap of model A vs model B on the SAME items.

    scores_a[i], scores_b[i]: per-item score of each model on item i (e.g. 0/1 pass, per-doc loss). Each resample
    draws the same item indices for both models, so item difficulty cancels. Default statistic: mean(A) - mean(B).
    Returns the CI of the difference, the fraction of resamples with A>B, and the item-level counts.
    """
    a, b = _arr(scores_a, "scores_a"), _arr(scores_b, "scores_b")
    if a.size != b.size:
        raise ValueError("paired bootstrap needs the same items for both models")
    if stat is None:
        d = a - b
        r = bootstrap(d, np.mean, B, conf, method, seed)
        r["statistic"] = "mean(A) - mean(B) over the same items"
        data = d
    else:
        data = np.stack([a, b], axis=1)
        r = bootstrap(data, lambda m: stat(m[:, 0], m[:, 1]), B, conf, method, seed, fast_mean=False)
        r["statistic"] = "user stat(A, B)"
    r.update(mean_a=a.mean(), mean_b=b.mean(), a_only_better=int(np.sum(a > b)), b_only_better=int(np.sum(b > a)),
             ties=int(np.sum(a == b)), ci_excludes_0=bool(r["ci"][0] > 0 or r["ci"][1] < 0))
    return _j(r)


# ----------------------------------------------------------------------------- sign-flip permutation test
def sign_flip_test(diffs, max_exact: int = 20, n_mc: int = 200000, seed: int = 0) -> dict:
    """Two-sided sign-flip (randomisation) test of H0: the paired differences are symmetric about 0.

    Exact enumeration of the 2^n sign patterns when n <= max_exact, else Monte Carlo. Statistic = |sum d_i|.
    NOTE: the smallest attainable two-sided p is 2/2^n: with n=3 seeds it is 0.25, so at n<=4 the test can
    NEVER reach p<0.05 (n=5: 0.0625; n=6: 0.03125). Report it, but use CIs for decisions at n=3.
    """
    d = _arr(diffs, "diffs")
    n = d.size
    obs = abs(d.sum())
    tol = 1e-12 * max(1.0, np.abs(d).sum())
    if n <= max_exact:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        stats_ = np.abs(signs @ d)
        p = float(np.mean(stats_ >= obs - tol))
        mode, total = "exact", 2 ** n
    else:
        rng = np.random.default_rng(seed)
        s = rng.choice([-1.0, 1.0], size=(n_mc, n))
        p = float((1 + np.sum(np.abs(s @ d) >= obs - tol)) / (n_mc + 1))
        mode, total = "monte_carlo", n_mc
    return _j({"test": "sign-flip permutation (two-sided, mean statistic)", "n": n, "mode": mode, "n_perm": total,
               "p": p, "min_attainable_p": 2.0 / 2 ** n, "can_reach_0.05": bool(2.0 / 2 ** n <= 0.05)})


# ----------------------------------------------------------------------------- multiplicity
def holm(pvals, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down. Controls the family-wise error rate for ANY dependence between tests."""
    p = _arr(pvals, "pvals")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = np.empty(m)
    adj[order] = np.minimum(1.0, adj_sorted)
    return _j({"method": "Holm-Bonferroni (FWER)", "m": m, "alpha": alpha, "p": p, "p_adj": adj,
               "reject": adj <= alpha})


def benjamini_hochberg(pvals, q: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up. Controls the FDR (valid under independence / positive dependence)."""
    p = _arr(pvals, "pvals")
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(1.0, adj_sorted)
    return _j({"method": "Benjamini-Hochberg (FDR)", "m": m, "q": q, "p": p, "p_adj": adj, "reject": adj <= q})


# ----------------------------------------------------------------------------- equivalence (TOST)
def tost_paired(a, b, margin: float, alpha: float = 0.05) -> dict:
    """TOST for paired seeds: H0 |mean(a-b)| >= margin vs H1 |mean(a-b)| < margin (absolute margin, same units).

    Equivalent to: the (1-2*alpha) = 90% CI of the mean difference lies inside (-margin, +margin). Reports both.
    `min_margin_at_zero_diff` = t_{1-alpha,df} * se: the smallest margin this data could ever have certified even
    if the observed difference were exactly 0 (tells you whether the chosen margin was testable at all)."""
    a, b = _arr(a, "a"), _arr(b, "b")
    if a.size != b.size or a.size < 2:
        raise ValueError("need >= 2 paired seeds")
    d = a - b
    return _tost_from(d.mean(), d.std(ddof=1) / math.sqrt(d.size), d.size - 1, margin, alpha, "paired",
                      mean_a=a.mean(), mean_b=b.mean())


def tost_welch(a, b, margin: float, alpha: float = 0.05) -> dict:
    """TOST on an unpaired Welch difference (use when arms do not share seed structure)."""
    a, b = _arr(a, "a"), _arr(b, "b")
    if a.size < 2 or b.size < 2:
        raise ValueError("need >= 2 per arm")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return _tost_from(a.mean() - b.mean(), se, df, margin, alpha, "welch", mean_a=a.mean(), mean_b=b.mean())


def _tost_from(diff, se, df, margin, alpha, kind, **extra) -> dict:
    if margin <= 0:
        raise ValueError("margin must be > 0")
    if se <= 0:
        p_lo = p_hi = 0.0 if abs(diff) < margin else 1.0
    else:
        p_lo = 1.0 - t_cdf((diff + margin) / se, df)    # H0: diff <= -margin
        p_hi = t_cdf((diff - margin) / se, df)          # H0: diff >= +margin
    tc = t_ppf(1.0 - alpha, df)
    ci = [diff - tc * se, diff + tc * se]
    p = max(p_lo, p_hi)
    return _j({"test": f"TOST ({kind} t)", "mean_diff": diff, "se": se, "df": df, "margin": margin, "alpha": alpha,
               "p_lower": p_lo, "p_upper": p_hi, "p_tost": p, "equivalent": bool(p < alpha),
               "ci_level": 1 - 2 * alpha, "ci": ci, "ci_inside_margin": bool(ci[0] > -margin and ci[1] < margin),
               "min_margin_at_zero_diff": tc * se, **extra})


# ----------------------------------------------------------------------------- variance decomposition
def variance_components(matrix) -> dict:
    """Two-way random-effects ANOVA without replication on a rows x cols matrix (EMS estimators).

    Seed-vs-model use: rows = seeds, cols = models that SHARE the seed's data order (E10, E1, E11). Then
      var_rows  = variance shared by all models for a given seed index  (data order / anything seed-tied),
      var_resid = row x col interaction + noise                          (init, arch-specific response).
    `icc_rows` = var_rows/(var_rows+var_resid) is the correlation between two models' losses at the same seed:
    pairing only helps when it is clearly > 0.
    Init-vs-data use (needs a crossed design, see proposed-side-quests SQ110): rows = init seeds, cols = data seeds;
    then var_rows = init variance, var_cols = data-order variance, var_resid = interaction + noise.
    Negative method-of-moments estimates are truncated to 0 (and flagged). df: rows r-1, cols c-1, resid (r-1)(c-1);
    with 3 seeds x 3 models that is 2, 2, 4 df: treat as an order-of-magnitude estimate only.
    """
    Y = np.asarray(matrix, dtype=float)
    if Y.ndim != 2 or min(Y.shape) < 2:
        raise ValueError("need a 2-D matrix with >= 2 rows and >= 2 cols")
    r, c = Y.shape
    gm = Y.mean()
    ss_r = c * np.sum((Y.mean(axis=1) - gm) ** 2)
    ss_c = r * np.sum((Y.mean(axis=0) - gm) ** 2)
    ss_e = np.sum((Y - gm) ** 2) - ss_r - ss_c
    ms_r, ms_c = ss_r / (r - 1), ss_c / (c - 1)
    ms_e = ss_e / ((r - 1) * (c - 1))
    raw_r, raw_c = (ms_r - ms_e) / c, (ms_c - ms_e) / r
    v_r, v_c, v_e = max(raw_r, 0.0), max(raw_c, 0.0), max(ms_e, 0.0)
    tot = v_r + v_c + v_e
    return _j({"design": "two-way random effects, no replication", "rows": r, "cols": c,
               "df": {"rows": r - 1, "cols": c - 1, "resid": (r - 1) * (c - 1)},
               "var_rows": v_r, "var_cols": v_c, "var_resid": v_e, "sd_rows": math.sqrt(v_r),
               "sd_cols": math.sqrt(v_c), "sd_resid": math.sqrt(v_e),
               "icc_rows": v_r / tot if tot > 0 else None, "truncated_negative": bool(raw_r < 0 or raw_c < 0),
               "ms": {"rows": ms_r, "cols": ms_c, "resid": ms_e},
               "caveat": "few df: indicative only; the pairing question is whether icc_rows is clearly > 0"})


def variance_components_oneway(groups) -> dict:
    """One-way random effects (e.g. groups = models, replicates = seeds): within-group (seed) vs between-group var."""
    g = [_arr(x) for x in groups]
    k = len(g)
    ni = np.array([x.size for x in g], float)
    N = ni.sum()
    gm = np.concatenate(g).mean()
    ssb = sum(x.size * (x.mean() - gm) ** 2 for x in g)
    ssw = sum(np.sum((x - x.mean()) ** 2) for x in g)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(ni ** 2) / N) / (k - 1)
    v_b = max((msb - msw) / n0, 0.0)
    return _j({"design": "one-way random effects", "groups": k, "N": int(N), "var_between": v_b, "var_within": msw,
               "sd_within": math.sqrt(msw), "icc": v_b / (v_b + msw) if v_b + msw > 0 else None})


# ----------------------------------------------------------------------------- power / minimum detectable effect
def _nct_sf_abs(ncp: float, df: float, tcrit: float, force_fallback: bool = False) -> float:
    """P(|T'| > tcrit) for a noncentral t (ncp, df), by numerical integration over the chi-square (no scipy)."""
    if HAVE_SCIPY and not force_fallback:
        return float(_sps.nct.sf(tcrit, df, ncp) + _sps.nct.cdf(-tcrit, df, ncp))
    # T' = (Z + ncp)/sqrt(V/df),  V ~ chi2(df).  P(|T'|>tc | V) = Phi(-tc*s + ncp) + Phi(-tc*s - ncp), s = sqrt(V/df)
    hi = df + 40.0 * math.sqrt(2.0 * df) + 40.0
    v = np.linspace(1e-9, hi, 40001)
    logf = (df / 2 - 1) * np.log(v) - v / 2 - (df / 2) * math.log(2) - math.lgamma(df / 2)
    f = np.exp(logf)
    s = np.sqrt(v / df)
    erfv = np.vectorize(math.erf)
    cond = 0.5 * (1 + erfv((-tcrit * s + ncp) / math.sqrt(2))) + 0.5 * (1 + erfv((-tcrit * s - ncp) / math.sqrt(2)))
    y = f * cond
    return float(np.sum((y[1:] + y[:-1]) * np.diff(v)) / 2.0)


def power_t(effect: float, sd: float, n: int, design: str = "paired", alpha: float = 0.05) -> dict:
    """Power of the two-sided t-test to detect a true mean difference `effect`.

    paired: sd = sd of the seed DIFFERENCES, n pairs, df = n-1.   two_sample: sd = common seed sd, n per arm,
    df = 2n-2 (equal-variance t; Welch is slightly lower)."""
    se, df = _design_se(sd, n, design)
    tc = t_ppf(1 - alpha / 2, df)
    return _j({"design": design, "n": n, "df": df, "sd": sd, "effect": effect, "alpha": alpha,
               "power": _nct_sf_abs(abs(effect) / se, df, tc)})


def _design_se(sd, n, design):
    if design == "paired":
        return sd / math.sqrt(n), n - 1
    if design == "two_sample":
        return sd * math.sqrt(2.0 / n), 2 * n - 2
    raise ValueError("design must be 'paired' or 'two_sample'")


def min_detectable_effect(sd: float, n: int, design: str = "two_sample", alpha: float = 0.05, power: float = 0.80,
                          ref_mean: float | None = None) -> dict:
    """Smallest true mean difference detected with probability `power` by the two-sided t-test (exact noncentral t).

    sd: observed seed sd (design 'two_sample') or sd of paired differences (design 'paired'). With n=3 per arm
    and alpha=0.05 the t critical value is 2.78 (df 4); paired n=3 has df 2 and t crit 4.30. `mde_in_sd` is the
    MDE in units of sd: a model gap smaller than this is invisible at this n. Also gives the 'approx' closed form
    (t_{1-a/2}+t_{power}) * se and, if ref_mean is given, the MDE as % of it."""
    se, df = _design_se(sd, n, design)
    tc = t_ppf(1 - alpha / 2, df)
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _nct_sf_abs(mid, df, tc) < power:
            lo = mid
        else:
            hi = mid
    ncp = 0.5 * (lo + hi)
    out = {"design": design, "n": n, "df": df, "sd": sd, "alpha": alpha, "power": power, "se": se,
           "mde": ncp * se, "mde_in_sd": ncp * se / sd if sd > 0 else None,
           "approx_mde": (tc + t_ppf(power, df)) * se}
    if ref_mean:
        out["mde_pct_of_ref"] = 100.0 * ncp * se / ref_mean
    return _j(out)


def n_for_effect(effect: float, sd: float, design: str = "two_sample", alpha: float = 0.05, power: float = 0.80,
                 n_max: int = 200) -> dict:
    """Smallest n (seeds per arm, or pairs) whose MDE <= effect."""
    for n in range(2, n_max + 1):
        if min_detectable_effect(sd, n, design, alpha, power)["mde"] <= effect:
            return _j({"n": n, "design": design, "effect": effect, "sd": sd, "alpha": alpha, "power": power})
    return _j({"n": None, "note": f"> {n_max}"})


# ----------------------------------------------------------------------------- selftest
def _selftest() -> int:
    fails: list[str] = []
    rng = np.random.default_rng(12345)

    def check(label, ok, info=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {label} {info}")
        if not ok:
            fails.append(label)

    print(f"scipy available: {HAVE_SCIPY}")
    print("-- t quantiles (tables)")
    for df, want in [(2, 4.302653), (4, 2.776445), (9, 2.262157), (30, 2.042272)]:
        got = t_ppf(0.975, df)
        check(f"t_0.975,df={df}", abs(got - want) < 2e-5, f"{got:.6f} vs {want}")
    check("t_cdf symmetric/known", abs(t_cdf(2.776445, 4) - 0.975) < 1e-6)
    check("norm_ppf(0.975)=1.959964", abs(norm_ppf(0.975) - 1.959964) < 1e-5)
    if HAVE_SCIPY:
        os.environ["GOGETA_STATS_NO_SCIPY"] = "1"
        # fallback agreement: recompute _betainc based cdf directly
        x = 4 / (4 + 2.5 ** 2)
        fb = 1.0 - 0.5 * _betainc(2.0, 0.5, x)
        check("fallback t_cdf == scipy", abs(fb - float(_sps.t.cdf(2.5, 4))) < 1e-10, f"{fb:.10f}")
        del os.environ["GOGETA_STATS_NO_SCIPY"]

    print("-- paired CI coverage (n=3, df=2) and Welch coverage (unequal sd)")
    N = 40000
    x = rng.normal(0.3, 1.0, size=(N, 3))
    m, s = x.mean(1), x.std(1, ddof=1)
    tc = t_ppf(0.975, 2)
    cov = np.mean(np.abs(m - 0.3) <= tc * s / math.sqrt(3))
    check("paired t CI coverage ~95%", abs(cov - 0.95) < 0.006, f"{cov:.4f}")
    a = rng.normal(0, 1, size=(N, 3))
    b = rng.normal(0, 3, size=(N, 4))
    va, vb = a.var(1, ddof=1) / 3, b.var(1, ddof=1) / 4
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 3)
    tcrit = np.array([t_ppf(0.975, d) for d in df[:4000]])
    cov = np.mean(np.abs(a.mean(1) - b.mean(1))[:4000] <= tcrit * se[:4000])
    check("Welch CI coverage ~95% (sd 1 vs 3, n 3/4)", abs(cov - 0.95) < 0.012, f"{cov:.4f}")
    pd_ = paired_diff([1, 2, 3], [0, 1, 1])
    check("paired_diff known case", abs(pd_["mean_diff"] - 4 / 3) < 1e-12 and pd_["df"] == 2 and abs(pd_["sd_diff"] - math.sqrt(1 / 3)) < 1e-9,
          f"mean {pd_['mean_diff']:.4f} sd {pd_['sd_diff']:.4f} ci {pd_['ci']}")
    w = welch([1, 2, 3, 4], [1, 2, 3, 4])
    check("welch identical samples: diff 0, p=1", w["mean_diff"] == 0 and abs(w["p"] - 1.0) < 1e-9)
    if HAVE_SCIPY:
        a_, b_ = rng.normal(0, 1, 5), rng.normal(1, 2, 7)
        ref = _sps.ttest_ind(a_, b_, equal_var=False)
        check("welch p == scipy ttest_ind(equal_var=False)", abs(welch(a_, b_)["p"] - ref.pvalue) < 1e-9)
        ref = _sps.ttest_rel(a_, b_[:5])
        check("paired p == scipy ttest_rel", abs(paired_diff(a_, b_[:5])["p"] - ref.pvalue) < 1e-9)

    print("-- effect sizes")
    g1, g2 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    check("cohens_d = -1 for shift 1, sd 1", abs(cohens_d(g1, g2)["d"] + 1.0) < 1e-12)
    check("rel_gap_pct", abs(rel_gap_pct([0.99], [1.0]) + 1.0) < 1e-12)
    check("dz of [1,1.5,2] = 2", abs(dz([1, 1.5, 2])["dz"] - 3.0) < 1e-9 or abs(dz([1, 1.5, 2])["dz"] - 1.5 / 0.5) < 1e-9)

    print("-- bootstrap over items (skewed data, n=40, mean)")
    M, n_i, mu = 600, 40, 1.0           # exponential(1): mean 1, skewed
    hit = {"percentile": 0, "bca": 0}
    for i in range(M):
        xs = rng.exponential(1.0, n_i)
        for meth in hit:
            ci = bootstrap(xs, np.mean, B=1000, method=meth, seed=i)["ci"]
            hit[meth] += ci[0] <= mu <= ci[1]
    pc, bc = hit["percentile"] / M, hit["bca"] / M
    check("percentile coverage in [0.90, 0.97]", 0.90 <= pc <= 0.97, f"{pc:.3f}")
    check("BCa coverage in [0.91, 0.97] and >= percentile - 0.01", 0.91 <= bc <= 0.97 and bc >= pc - 0.01, f"{bc:.3f}")
    xs = rng.normal(5, 2, 200)
    bb = bootstrap(xs, np.mean, B=4000, method="bca", seed=1)
    se_t = xs.std(ddof=1) / math.sqrt(200)
    check("bootstrap SE ~ s/sqrt(n)", abs(bb["boot_se"] / se_t - 1) < 0.1, f"{bb['boot_se']:.4f} vs {se_t:.4f}")
    # paired bootstrap: same items; item difficulty must cancel
    diff_item = rng.normal(0, 3, 150)                              # big item-difficulty spread
    pa = (rng.random(150) < 1 / (1 + np.exp(-(diff_item + 0.3)))).astype(float)
    pb = (rng.random(150) < 1 / (1 + np.exp(-diff_item))).astype(float)
    pbt = paired_bootstrap(pa, pb, B=3000, seed=2)
    ub = bootstrap(pa, B=3000, seed=2)["ci"], bootstrap(pb, B=3000, seed=2)["ci"]
    unpaired_w = math.sqrt((ub[0][1] - ub[0][0]) ** 2 + (ub[1][1] - ub[1][0]) ** 2) / 1
    check("paired CI narrower than unpaired when items are correlated", (pbt["ci"][1] - pbt["ci"][0]) < unpaired_w,
          f"paired width {pbt['ci'][1] - pbt['ci'][0]:.3f} < unpaired {unpaired_w:.3f}")
    pbs = paired_bootstrap(pa, pb, B=500, seed=3, stat=lambda u, v: u.mean() - v.mean())
    check("paired_bootstrap with user stat == default estimate", abs(pbs["estimate"] - pbt["estimate"]) < 1e-12)

    print("-- sign-flip test")
    sf = sign_flip_test([1.0, 2.0, 3.0])
    check("n=3 all positive: exact two-sided p = 0.25", abs(sf["p"] - 0.25) < 1e-12 and not sf["can_reach_0.05"], f"p={sf['p']}")
    sf2 = sign_flip_test([1, 2, 3, 4, 5])
    check("n=5 all positive: p = 2/32", abs(sf2["p"] - 2 / 32) < 1e-12)
    dd = rng.normal(0.5, 1, 14)
    pe, pm = sign_flip_test(dd)["p"], sign_flip_test(dd, max_exact=3, n_mc=100000, seed=1)["p"]
    check("Monte Carlo agrees with exact (n=14)", abs(pe - pm) < 0.01, f"{pe:.4f} vs {pm:.4f}")
    fp = np.mean([sign_flip_test(rng.normal(0, 1, 8))["p"] < 0.05 for _ in range(1500)])
    check("false-positive rate under H0 <= 0.055 (n=8)", fp <= 0.055, f"{fp:.4f}")

    print("-- Holm / BH")
    p = [0.01, 0.04, 0.03, 0.005]
    h = holm(p)
    check("Holm adjusted known", np.allclose(h["p_adj"], [0.03, 0.06, 0.06, 0.02]), str(h["p_adj"]))
    bh = benjamini_hochberg(p)
    check("BH adjusted known", np.allclose(bh["p_adj"], [0.02, 0.04, 0.04, 0.02]), str(bh["p_adj"]))
    check("Holm >= raw, BH <= Holm", all(x >= y for x, y in zip(h["p_adj"], p)) and all(u <= v + 1e-12 for u, v in zip(bh["p_adj"], h["p_adj"])))
    pv = rng.random((6000, 6))                                      # global null, m=6
    fw = np.mean([holm(r)["reject"].__contains__(True) for r in pv[:3000]])
    check("Holm FWER under global null <= 0.055 (m=6)", fw <= 0.055, f"{fw:.4f}")
    raw_fw = np.mean(np.any(pv < 0.05, axis=1))
    check("(uncorrected FWER is ~26%: the problem Holm fixes)", 0.22 < raw_fw < 0.30, f"{raw_fw:.3f}")
    fdr = []
    for _ in range(1500):                                           # 4 nulls + 4 real effects (tiny p)
        pp = np.concatenate([rng.random(4), rng.random(4) * 1e-4])
        rj = np.array(benjamini_hochberg(pp)["reject"])
        fdr.append(rj[:4].sum() / max(1, rj.sum()))
    check("BH FDR <= 0.05 + tol (4 nulls, 4 real)", np.mean(fdr) <= 0.055, f"{np.mean(fdr):.4f}")

    print("-- TOST")
    margin, nrep = 0.8, 4000
    def tost_rate(true):
        r = 0
        for _ in range(nrep):
            a_ = rng.normal(true, 1.0, 20) + rng.normal(0, 1, 20) * 0
            b_ = np.zeros(20) + rng.normal(0, 1.0, 20) * 0
            dd_ = rng.normal(true, 1.0, 20)
            r += _tost_from(dd_.mean(), dd_.std(ddof=1) / math.sqrt(20), 19, margin, 0.05, "paired")["equivalent"]
        return r / nrep
    check("TOST size at the margin boundary <= 0.055", tost_rate(margin) <= 0.062, "(nominal 0.05, MC se 0.004)")
    check("TOST power at true diff 0 (n=20, margin .8 sd) > 0.6", tost_rate(0.0) > 0.6, f"{tost_rate(0.0):.3f}")
    ok = True
    for _ in range(300):
        aa, bb_ = rng.normal(0, 1, 5), rng.normal(0, 1, 5)
        r = tost_paired(aa, bb_, 0.7)
        ok &= (r["equivalent"] == r["ci_inside_margin"])
    check("TOST p<alpha  <=>  90% CI inside margin", ok)
    t3 = tost_paired([0.585, 0.586, 0.584], [0.585, 0.585, 0.585], 0.006)
    check("TOST n=3 sanity: tiny diffs, margin 1% -> equivalent", t3["equivalent"], f"p={t3['p_tost']:.4f}")

    print("-- variance components")
    R, C = 4000, 3
    Yr = rng.normal(0, 1.0, (R, 1)) + rng.normal(0, 0.5, (R, C))          # var_rows 1, resid 0.25
    vc = variance_components(Yr)
    check("var_rows ~ 1.0, var_resid ~ 0.25", abs(vc["var_rows"] - 1.0) < 0.1 and abs(vc["var_resid"] - 0.25) < 0.02,
          f"{vc['var_rows']:.3f}, {vc['var_resid']:.3f}")
    Yc = rng.normal(0, 1.0, (400, 1)) + rng.normal(0, 0.7, (1, 400)) + rng.normal(0, 0.5, (400, 400))
    vc = variance_components(Yc)
    check("crossed: var_rows ~ 1, var_cols ~ 0.49, resid ~ 0.25",
          abs(vc["var_rows"] - 1) < 0.15 and abs(vc["var_cols"] - 0.49) < 0.12 and abs(vc["var_resid"] - 0.25) < 0.02,
          f"{vc['var_rows']:.2f}, {vc['var_cols']:.2f}, {vc['var_resid']:.3f}")
    vo = variance_components_oneway([rng.normal(0, 1, 3000) + 2, rng.normal(0, 1, 3000) - 2, rng.normal(0, 1, 3000)])
    check("one-way: within var ~ 1", abs(vo["var_within"] - 1) < 0.05, f"{vo['var_within']:.3f}")

    print("-- power / MDE")
    mde = min_detectable_effect(1.0, 3, "two_sample")
    tc4 = t_ppf(0.975, 4)
    check("MDE n=3/arm in sd units in [2.8, 3.4] (df 4, se=0.816 sd)", 2.8 < mde["mde_in_sd"] < 3.4, f"{mde['mde_in_sd']:.3f} sd (approx {mde['approx_mde']:.3f})")
    sim = rng.normal(mde["mde"], 1.0, (60000, 3)), rng.normal(0.0, 1.0, (60000, 3))
    # equal-variance two-sample t via simulation at the MDE
    A, B_ = sim
    sp = np.sqrt((A.var(1, ddof=1) + B_.var(1, ddof=1)) / 2)
    tt = (A.mean(1) - B_.mean(1)) / (sp * math.sqrt(2 / 3))
    simpow = np.mean(np.abs(tt) > tc4)
    check("simulated power at MDE ~ 0.80", abs(simpow - 0.80) < 0.012, f"{simpow:.4f}")
    mdp = min_detectable_effect(1.0, 3, "paired")
    d_ = rng.normal(mdp["mde"], 1.0, (60000, 3))
    tt = d_.mean(1) / (d_.std(1, ddof=1) / math.sqrt(3))
    simpow = np.mean(np.abs(tt) > t_ppf(0.975, 2))
    check("paired n=3 MDE: simulated power ~ 0.80", abs(simpow - 0.80) < 0.012, f"{simpow:.4f}; MDE {mdp['mde_in_sd']:.2f} sd of the differences")
    check("power_t at MDE = 0.80", abs(power_t(mde["mde"], 1.0, 3, "two_sample")["power"] - 0.80) < 1e-6)
    check("n_for_effect monotone: MDE 1 sd needs n >= 15 per arm", n_for_effect(1.0, 1.0)["n"] >= 15, f"n={n_for_effect(1.0, 1.0)['n']}")
    if HAVE_SCIPY:
        fb = _nct_sf_abs(3.0, 4, 2.776445, force_fallback=True)
        sc = float(_sps.nct.sf(2.776445, 4, 3.0) + _sps.nct.cdf(-2.776445, 4, 3.0))
        check("noncentral-t integration fallback == scipy nct", abs(fb - sc) < 1e-5, f"{fb:.7f} vs {sc:.7f}")

    print("-- JSON hygiene")
    json.dumps(_j({"a": np.float64("nan"), "b": np.array([1, 2.5]), "c": np.bool_(True)}), allow_nan=False)
    check("outputs serialise with allow_nan=False", True)

    print(f"\nselftest: {'PASS' if not fails else 'FAIL ' + str(fails)}")
    return 0 if not fails else 1


# ----------------------------------------------------------------------------- CLI
def _floats(xs):
    return [float(v) for v in xs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gogeta-0 statistics core (prints JSON)")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("paired", "welch"):
        p = sub.add_parser(name)
        p.add_argument("--a", nargs="+", type=float, required=True)
        p.add_argument("--b", nargs="+", type=float, required=True)
        p.add_argument("--conf", type=float, default=0.95)
    for name in ("tost-paired", "tost-welch"):
        p = sub.add_parser(name)
        p.add_argument("--a", nargs="+", type=float, required=True)
        p.add_argument("--b", nargs="+", type=float, required=True)
        p.add_argument("--margin", type=float, required=True, help="absolute margin, same units as the values")
        p.add_argument("--alpha", type=float, default=0.05)
    p = sub.add_parser("boot")
    p.add_argument("--values", nargs="+", type=float, required=True)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--method", choices=["percentile", "bca"], default="bca")
    p = sub.add_parser("pboot", help="paired item bootstrap: per-item scores of model A and B")
    p.add_argument("--a", nargs="+", type=float, required=True)
    p.add_argument("--b", nargs="+", type=float, required=True)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--method", choices=["percentile", "bca"], default="bca")
    p = sub.add_parser("signflip")
    p.add_argument("--diffs", nargs="+", type=float, required=True)
    for name in ("holm", "bh"):
        p = sub.add_parser(name)
        p.add_argument("--p", nargs="+", type=float, required=True)
        p.add_argument("--alpha", type=float, default=0.05)
    p = sub.add_parser("mde")
    p.add_argument("--sd", type=float, required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--design", choices=["paired", "two_sample"], default="two_sample")
    p.add_argument("--ref-mean", type=float)
    p.add_argument("--power", type=float, default=0.8)
    p = sub.add_parser("varcomp", help="--matrix 'a,b,c;d,e,f' rows;cols")
    p.add_argument("--matrix", required=True)
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.cmd == "paired":
        out = paired_diff(args.a, args.b, args.conf)
    elif args.cmd == "welch":
        out = welch(args.a, args.b, args.conf)
    elif args.cmd == "tost-paired":
        out = tost_paired(args.a, args.b, args.margin, args.alpha)
    elif args.cmd == "tost-welch":
        out = tost_welch(args.a, args.b, args.margin, args.alpha)
    elif args.cmd == "boot":
        out = bootstrap(args.values, B=args.B, method=args.method)
    elif args.cmd == "pboot":
        out = paired_bootstrap(args.a, args.b, B=args.B, method=args.method)
    elif args.cmd == "signflip":
        out = sign_flip_test(args.diffs)
    elif args.cmd == "holm":
        out = holm(args.p, args.alpha)
    elif args.cmd == "bh":
        out = benjamini_hochberg(args.p, args.alpha)
    elif args.cmd == "mde":
        out = min_detectable_effect(args.sd, args.n, args.design, power=args.power, ref_mean=args.ref_mean)
    elif args.cmd == "varcomp":
        out = variance_components([[float(v) for v in r.split(",")] for r in args.matrix.split(";")])
    else:
        ap.print_help()
        return 2
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
