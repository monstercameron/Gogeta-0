"""SQ414 (descriptive, not registered; no GPU): do 20M-token screens predict 200M outcomes?

(a) From the in-training eval curves (launcher logs: `eval step N: val/loss`, 128 val sequences, every 250 steps, 32,768 tokens/step) of the 200M runs on the fleet pulls, per arm
(mean over the seeds found; best LR only: 6e-3 for A-family/E27, 3e-3 for E20/E24, via the run's meta.json): val loss at the step nearest 20M tokens (610; linear interpolation
between the evals at steps 500 and 750) and at the end (step 6103). NOTE the cosine schedule: the 20M point of a 200M run is mid-warmdown-free (about peak LR), a different
state from a run annealed over 20M tokens. (b) B1's direct 20M runs (results/b1_runs.json; schedule annealed over 20M tokens): final val_loss and clean80 per arm (mean over seeds) versus the 200M
end (in-training val loss). For arms present in both.
Metrics: Spearman and Kendall rank correlation of the arm orderings (20M point vs 200M end), the share of pairwise orderings that flip, and the effect (% difference of val loss) of named
contrasts at each horizon. Per the row: if the correlation is below 0.5 or more than a quarter of orderings flip, 20M screens may only reject candidates, never select them.

Run (WSL): python experiments/checks/sq414_screen_predicts.py -> results/sq414.json
"""
import glob
import itertools
import json
import math
import os
import re
import statistics as st
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOME = os.path.expanduser("~/fusiondance-data")
BEST_LR = {"A2s": 0.006, "A2f": 0.006, "A3m": 0.006, "A3mx": 0.006, "A3x": 0.006, "A3y": 0.006, "E20": 0.003, "E24": 0.003, "E27": 0.006, "E22": 0.006, "E21": 0.006}
CONTRASTS = [("A3m", "A2s"), ("A2f", "E24"), ("A2s", "A2f"), ("A2s", "E27"), ("A3mx", "A2s"), ("E27", "E20"), ("E24", "E20"), ("A2f", "E20")]


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


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def compare(names, early, late):
    pairs = list(itertools.combinations(names, 2))
    flips = [p for p in pairs if (early[p[0]] < early[p[1]]) != (late[p[0]] < late[p[1]])]
    conc = len(pairs) - len(flips)
    return {"n_arms": len(names), "spearman": spearman([early[n] for n in names], [late[n] for n in names]), "kendall_tau": (conc - len(flips)) / len(pairs),
            "pairs": len(pairs), "flipped_pairs": len(flips), "flip_share": len(flips) / len(pairs), "flipped": [list(p) for p in flips]}


def main():
    paths = sorted(set(glob.glob(f"{HOME}/logs/*.log") + glob.glob(f"{HOME}/cloud/fleet/*/fusiondance-data/logs/*.log")))
    curves = {}
    for p in paths:
        r = parse(p)
        if r and r[3] and r[1] in BEST_LR and "lrhi" not in p and "lrlo" not in p:
            lr = lr_of(r[0])
            if lr is not None and abs(lr - BEST_LR[r[1]]) < 1e-12:
                curves.setdefault((r[1], r[2]), r[3])
    arms = {}
    for (a, s), ev in curves.items():
        e20 = ev[500] + (ev[750] - ev[500]) * (610 - 500) / 250
        arms.setdefault(a, []).append((s, e20, ev[max(ev)]))
    early = {a: st.mean(x[1] for x in v) for a, v in arms.items()}
    late = {a: st.mean(x[2] for x in v) for a, v in arms.items()}
    names = sorted(arms)
    out = {"check": "SQ414", "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "status": "descriptive, not registered", "arms_200M": {a: {"seeds": sorted(x[0] for x in v), "val20M_point": early[a], "val_end": late[a]} for a, v in arms.items()}}
    out["inrun_20M_point_vs_200M_end"] = compare(names, early, late)

    def eff(d, a, b):
        return 100 * (d[a] / d[b] - 1)
    out["contrasts_inrun"] = {f"{a}_vs_{b}": {"at_20M_point_pct": eff(early, a, b), "at_end_pct": eff(late, a, b), "same_sign": (eff(early, a, b) > 0) == (eff(late, a, b) > 0)}
                              for a, b in CONTRASTS if a in early and b in early}
    # B1 direct 20M runs
    b1 = {}
    for x in json.load(open(os.path.join(ROOT, "experiments/checks/results/b1_runs.json")))["runs"]:
        if x.get("core_lr_mult", 1.0) == 1.0 and x["arm"] in late and abs(x["lr"] - BEST_LR.get(x["arm"], -1)) < 1e-12:
            b1.setdefault(x["arm"], []).append((x["val_loss"], x["clean80"], x["seed"]))
    bn = sorted(b1)
    b1_val = {a: st.mean(v[0] for v in b1[a]) for a in bn}
    b1_c80 = {a: st.mean(v[1] for v in b1[a]) for a in bn}
    out["b1_direct_20M"] = {a: {"seeds": [v[2] for v in b1[a]], "val_loss": b1_val[a], "clean80": b1_c80[a]} for a in bn}
    if len(bn) >= 3:
        out["b1_20M_val_vs_200M_end_val"] = compare(bn, b1_val, late)
        out["b1_20M_clean80_vs_200M_end_val"] = compare(bn, b1_c80, late)
        out["inrun_20M_point_vs_200M_end_same_arms_as_b1"] = compare(bn, early, late)
        out["contrasts_b1"] = {f"{a}_vs_{b}": {"b1_20M_val_pct": eff(b1_val, a, b), "b1_20M_clean80_pct": eff(b1_c80, a, b), "inrun_20M_point_pct": eff(early, a, b), "end_200M_pct": eff(late, a, b),
                                               "same_sign_b1_vs_end": (eff(b1_val, a, b) > 0) == (eff(late, a, b) > 0)}
                               for a, b in CONTRASTS if a in b1_val and b in b1_val}
    json.dump(out, open(os.path.join(ROOT, "experiments/checks/results/sq414.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ("arms_200M",)}, indent=1)[:4000])
    print({a: (round(early[a], 3), round(late[a], 3), len(arms[a])) for a in names})


if __name__ == "__main__":
    main()
