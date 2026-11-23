"""F2 seed-influence audit (quick review 2026-11-23 02:29): leave-one-seed-out on the registered seed-blocked
interaction I, using f2_analysis.py's inclusion rules, plus per-cell wall-clock from each run's summary.json.
    python3 experiments/design/f2_loso.py --doclevel /root/f2all --runs-glob "/root/fusiondance-data/cloud/fleet/*/fusiondance-data/runs"
"""
import argparse, glob, json, math, os, statistics as st, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import f2_analysis as A  # noqa: E402
import factorial_analysis as fa  # noqa: E402


def blocked_I(rows):
    cells = sorted({c for (c, _), _ in rows}); seeds = sorted({s for (_, s), _ in rows})
    X = np.zeros((len(rows), len(cells) + len(seeds) - 1)); y = np.array([v for _, v in rows])
    for i, ((c, s), _) in enumerate(rows):
        X[i, cells.index(c)] = 1
        if seeds.index(s) > 0:
            X[i, len(cells) + seeds.index(s) - 1] = 1
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return sum(fa.W5[c] * beta[cells.index(c)] for c in fa.W5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--doclevel", required=True); ap.add_argument("--runs-glob", required=True)
    a = ap.parse_args()
    meta, wall = {}, {}
    for d in glob.glob(a.runs_glob):
        for m in glob.glob(os.path.join(d, "E2[0-7]-*", "meta.json")):
            j = json.load(open(m)); cfg = j.get("config") or {}; tr = cfg.get("train", {}); co = j.get("cli_overrides") or {}
            lr = co.get("lr", tr.get("lr")) if isinstance(co, dict) else tr.get("lr")
            mt = (co.get("max_tokens") if isinstance(co, dict) else None) or tr.get("max_tokens")
            run = os.path.basename(os.path.dirname(m)); meta[run] = (lr, mt, tr.get("dataset"))
            sp = os.path.join(os.path.dirname(m), "summary.json")
            if os.path.exists(sp):
                wall[run] = (json.load(open(sp)).get("metrics") or {}).get("wall_min")
    rows, runs_by = {}, {}
    for f in glob.glob(os.path.join(a.doclevel, "E2[0-7]-2026*.json")):
        d = json.load(open(f)); run, arm, seed = d["run"], d["run"].split("-")[0], int(d["seed"])
        if arm not in A.CELL or run not in meta or seed == 0: continue
        lr, mt, ds = meta[run]
        if lr is None or abs(lr - A.LR[arm]) > 1e-12 or mt != 20000000 or ds != "code-py-v2": continue
        rows[(A.CELL[arm], seed)] = math.log(d["bpc"]["clean80"]); runs_by[(A.CELL[arm], seed)] = run
    rows = sorted(rows.items()); full = blocked_I(rows)
    seeds = sorted({s for (_, s), _ in rows}); loso = {s: blocked_I([r for r in rows if r[0][1] != s]) for s in seeds}
    vals = list(loso.values())
    out = {"I_full": full, "loso_min": min(vals), "loso_max": max(vals), "n_seeds": len(seeds),
           "most_influential_seed": max(loso, key=lambda s: abs(loso[s] - full)), "any_sign_flip": any(v >= 0 for v in vals)}
    wc = {}
    for (c, s), run in runs_by.items():
        if wall.get(run) is not None: wc.setdefault(c, []).append(wall[run])
    out["wall_min_median_per_cell"] = {c: st.median(v) for c, v in sorted(wc.items())}
    out["wall_n_per_cell"] = {c: len(v) for c, v in sorted(wc.items())}
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "checks", "results", "f2_loso_wall.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
