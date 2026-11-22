# Cockpit integration (A2)

All functions return JSON-safe dicts (NaN/inf become null). Import: `sys.path.insert(0, "experiments/stats"); import core`. CLI: `python experiments/stats/core.py <cmd> ...` prints JSON. No GPU, numpy only (scipy optional; `GOGETA_STATS_NO_SCIPY=1` forces the fallbacks).

| Function (CLI) | Returns | Use in the cockpit |
|---|---|---|
| `core.summarize(x)` | n, mean, sd, sem, df, t-CI | run-group card (n and CI always shown) |
| `core.paired_diff(a, b)` (`paired`) | mean/sd of differences, df, t, p, 95% CI (abs and %), dz, relative gap % | E1 vs E10, E11 vs E1 panels |
| `core.welch(a, b)` (`welch`) | Welch df, CI, p, Cohen d, relative gap % | sensitivity row |
| `core.tost_paired / tost_welch(a, b, margin)` (`tost-paired`, `tost-welch`) | p_tost, 90% CI, equivalent, min_margin_at_zero_diff | "free / no loss" badges |
| `core.bootstrap(x)` / `core.paired_bootstrap(a, b)` (`boot`, `pboot`) | percentile or BCa CI over items | benchmark and doc-level panels |
| `core.sign_flip_test(d)` (`signflip`) | exact p, minimum attainable p | shows why n=3 cannot reach 0.05 |
| `core.holm(p)` / `core.benjamini_hochberg(p)` (`holm`, `bh`) | adjusted p, reject flags | arc "family" table |
| `core.variance_components(matrix)` (`varcomp`) | var_rows / var_cols / var_resid, icc_rows | "does pairing help" |
| `core.min_detectable_effect(sd, n, design)` (`mde`), `power_t`, `n_for_effect` | MDE (absolute, in sd, % of ref), power | "what could this n have seen" annotation |
| `replication_stats.analyse_stats(runs, lrs, ref, doclevel)` | the full secondary layer (dict) | Phase 2b statistics page |

Files: `experiments/checks/results/REPLICATION-stats.json` (written by `replication_stats.py`; schema = `common.save`: `verdicts` = text lines, `data` = the full dict) and `experiments/stats/doclevel/<run>.json` (written by `score_doclevel.py`, GPU).
Suggested rules: never show a difference without n, df and CI; show an `[exploratory n=1]` badge on any LM comparison with one seed; link each claim row to its family and Holm-adjusted p.
Self-tests (CPU, ~10 s): `python experiments/stats/core.py --selftest`, `python experiments/stats/replication_stats.py --selftest`, `python experiments/stats/score_doclevel.py --selftest`.
