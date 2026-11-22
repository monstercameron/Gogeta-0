"""A3: power and GPU-hour calculations for the factorial, decomposition and HPO additions (pure python, CPU, instant).

    python3 experiments/design/power_cost.py        # reads experiments/design/staged-table.json (run build_staged.py first)

Estimand (log-loss scale; negative = the combination beats the additive prediction):
    I = y111 - y100 - y010 - y001 + 2*y000          (y = log val loss, cell mean over seeds)
Contrast weights w = (+1, -1, -1, -1, +2) on cells (111, 100, 010, 001, 000), sum(w) = 0.
Var(I-hat) = sigma^2 * sum(w_i^2 / n_i). Optimal allocation n_i ~ |w_i|  ->  Var = sigma^2 * (sum|w|)^2 / N = 36 sigma^2 / N.
Equal allocation: 8 sigma^2 / n_cell = 40 sigma^2 / N.  Two-sided alpha 0.05, power 0.80: MDE = 2.80 * SE.
"""
import json
import math
import pathlib

D = pathlib.Path(__file__).parent
T = json.load(open(D / "staged-table.json"))
est = {r["id"]: r["est_wall_min"] for r in T["staged"] + T["reference"]}
Z = 1.96 + 0.8416
W = {"E27": 1, "E24": -1, "E22": -1, "E21": -1, "E20": 2}   # cell 111,100,010,001,000 -> ids (id = 20+4g+2l+t)


def n_for(delta, sigma, opt=True):
    k = 36 if opt else 40
    return math.ceil(k * (Z * sigma / delta) ** 2)


def mde(N, sigma, opt=True):
    return Z * sigma * math.sqrt((36 if opt else 40) / N)


out = []
P = out.append
P("## Power for the full-model interaction I (5 cells, optimal allocation 2:1:1:1:1)\n")
P("sigma = seed sd of log val loss. The task's assumption is sd 0.004 in loss; at a cell loss of L that is sigma = 0.004/L "
  "(0.004 at L=1.0, 0.0067 at L=0.6). Runs needed N for 80% power (alpha 0.05, two-sided); per-cell n in brackets (000 gets double).\n")
P("| sigma | delta 1.0% | delta 1.5% | delta 2.0% | delta 3.0% |")
P("|---|---|---|---|---|")
for s in (0.003, 0.004, 0.005, 0.0067, 0.010):
    row = [f"{s}"]
    for d in (0.010, 0.015, 0.020, 0.030):
        N = n_for(d, s)
        per = math.ceil(N / 6)
        row.append(f"{N} ({per}/{2*per})")
    P("| " + " | ".join(row) + " |")
P("\nMDE (log units, 80% power) at fixed N, sigma 0.004: " + ", ".join(f"N={N}: {100*mde(N,0.004):.2f}%" for N in (10, 20, 30, 48, 60, 90)))
P("If seeds share the data order (common random numbers) with within-pair correlation rho between cells, Var scales by (1-rho): "
  "rho 0.3 cuts N by 30%. Not assumed; measured in the pilot.\n")

# plan costs at 20M tokens
cell = {i: est[i] for i in ("E20", "E21", "E22", "E23", "E24", "E25", "E26", "E27", "E35", "E36")}
prim = ("E20", "E21", "E22", "E24", "E27")
rate_tune = sum(cell[i] for i in prim)           # one run of each primary cell
N_sigma = {s: n_for(0.010, s) for s in (0.004, 0.0067)}
P("## Factorial GPU-hours (20M tokens per run, d=384, depth budget 14; est. wall min per run incl. 1 min gap)\n")
P("| cell (id) | est. min / run |\n|---|---|")
for i in cell:
    P(f"| {i} | {cell[i]:.1f} |")
P("")
G3 = 3.5   # 3 LR points + average 0.5 extension run
F0 = G3 * rate_tune / 60
F1 = 2 * rate_tune / 60
def confirm(sigma, pilot=2):
    N = n_for(0.010, sigma); per = math.ceil(N / 6)
    nn = {"E20": 2 * per, "E21": per, "E22": per, "E24": per, "E27": per}
    extra = sum(max(0, nn[i] - pilot) * cell[i] for i in prim) / 60
    return N, nn, extra
F2a = confirm(0.004); F2b = confirm(0.0067)
rem = ("E23", "E25", "E26")
F3 = (G3 * sum(cell[i] for i in rem) + 3 * sum(cell[i] for i in rem)) / 60
F4 = 3 * (cell["E35"] + cell["E36"]) / 60
plans = [("F0  LR tune: 5 primary cells x G3 {3e-3,6e-3,1.2e-2} + ~0.5 extension, seed 0", F0),
         ("F1  pilot: 5 cells x seeds 1,2 at their LR (estimates sigma, rho; MDE ~3-4%)", F1),
         (f"F2  confirm to N={F2a[0]} (sigma .004; n/cell {F2a[1]}): top-up", F2a[2]),
         (f"F2' confirm to N={F2b[0]} (sigma .0067) instead of F2: top-up", F2b[2]),
         ("F3  remaining 3 cells (E23,E25,E26): G3 tune + 3 seeds (pairwise interactions; not powered for 1%)", F3),
         ("F4  params-matched loop partners E35,E36: 3 seeds each, LR borrowed", F4)]
P("| stage | GPU-h |\n|---|---|")
for n, h in plans:
    P(f"| {n} | {h:.2f} |")
P(f"\n**Cheapest 1%-powered version (sigma 0.004): F0+F1+F2 = {F0+F1+F2a[2]:.1f} GPU-h.** With sigma 0.0067 (F2'): {F0+F1+F2b[2]:.1f} GPU-h. "
  f"Full 8-cell table adds F3 (+{F3:.1f}) and the matching-rule check F4 (+{F4:.1f}).")
# 200M-token scale-up check: 1 seed per primary cell + LR from F0
rows = {r["id"]: r for r in T["staged"]}
def wall200(i):
    r = rows[i]
    tf = 7.9 if i in ("E24", "E27", "E26", "E25") else 13.8
    if i in ("E21", "E27", "E25", "E23"): tf *= 0.75
    return 200e6 / (tf * 1e12 / r["flops_train"]) / 60 + 1
P("\n200M-token scale-up check (one seed per primary cell, same LRs; not powered): " +
  ", ".join(f"{i} {wall200(i):.0f} min" for i in prim) + f"; total {sum(wall200(i) for i in prim)/60:.1f} GPU-h.\n")

# decomposition
P("## E1 decomposition cost (20M tokens, deduplicated set)\n")
dec = ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "E47", "E48"]
e1_20 = est["E51"]            # E1 at 20M (same cost as E51)
e10_20 = est["E50"]
minimal = (sum(est[i] for i in dec) + e1_20) / 60
rigorous = (sum(4 * est[i] for i in dec) + 4 * e1_20 + 4 * e10_20) / 60
P(f"- Minimal tier (the outside review's version): 9 arms + E1 control, seed 0, LR 6e-3 = 10 runs = **{minimal:.2f} GPU-h** ({60*minimal:.0f} min). Reads only effects larger than about 2 x sqrt(2) x sigma (1.1% at sigma .004).")
P(f"- Rigorous tier: each arm and the E1 and E10 controls get LR {{3e-3, 6e-3}} at seed 0 (equal tuning budget), then 2 more seeds at the better LR: 11 x 4 = 44 runs = **{rigorous:.2f} GPU-h**. Arm-vs-control SE = 0.004 x sqrt(1/3+1/3) = 0.33% (MDE 80% power = 0.92%).")
P("")
# HPO additions
P("## HPO additions (GPU minutes)\n")
P("| item | est. min |\n|---|---|")
P(f"| H0 E10 at 1.2e-2, 20M, seed 0 (equalises K7 to 5 LR points per model) | {est['E50']:.1f} |")
P(f"| H1 second seed at the top-2 LRs, 20M: E10 {{2e-3,3e-3}}, E1 {{3e-3,6e-3}} (4 runs) | {2*est['E50']+2*est['E51']:.1f} |")
e1_200 = est["E53"]; e10_200 = est["E52"]
P(f"| H2 horizon check stage A, 200M: E1@3e-3, E10@6e-3, E10@2e-3 (v1, 1 seed each) | {e1_200+2*e10_200:.1f} |")
P(f"| H3 conditional stage B: E1@1.2e-2 200M (if E1@6e-3 beats E1@3e-3 by >1%) | {e1_200:.1f} |")
P(f"| H4 warmup sensitivity, 20M, warmup 30: E10 {{3e-3,6e-3}}, E1 {{6e-3,1.2e-2}} (E50,E51) | {2*est['E50']+2*est['E51']:.1f} |")
P(f"| H5 dedup bridge: E10 and E1 seed 0 at 200M on code-py-v2 (E52, E53) | {e1_200+e10_200:.1f} |")
P(f"| H5b conditional LR re-check on v2: E10 {{2e-3,3e-3,6e-3}} + E1 {{3e-3,6e-3,1.2e-2}} at 20M | {3*est['E50']+3*est['E51']:.1f} |")
# matched arms
P("\n## Matched-arm cost (200M, v1, 1 seed each; est.)\n")
P("| id | est. min |\n|---|---|")
for i in ("E30", "E31", "E32", "E33", "E34"):
    P(f"| {i} | {est[i]:.0f} |")
P(f"| **total** | **{sum(est[i] for i in ('E30','E31','E32','E33','E34'))/60:.1f} GPU-h** |")
(D / "power-cost.md").write_text("# Power and cost tables (generated by power_cost.py)\n\n" + "\n".join(out) + "\n")
print("\n".join(out))
