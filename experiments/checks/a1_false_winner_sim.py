"""A1 screen false-winner calibration (quick review 2026-11-23 11:29, statistician). Descriptive simulation, not a result.

Null model: every arm has zero true effect. Per-run log-bpc noise sd = 0.0053 (E27 3-seed sd, SQ263). Each arm's paired
effect per seed = (arm_s - E27_s); all arms share the SAME E27 reference runs (cross-arm correlation). Winner rule (A1
registration): 2-seed mean <= log(0.99) and both seeds < 0. Reports P(at least one false winner among k arms) and the
expected number, for k = 8 (original) and 11 (with amendment 1), plus a fresh-seed A2 confirmation (3 seeds, mean <= log 0.99).
Run: python experiments/checks/a1_false_winner_sim.py -> experiments/checks/results/a1_false_winner_sim.json
"""
import json
import math
import os
import random

SD, THR, N = 0.0053, math.log(0.99), 200000
random.seed(0)


def sim(k, a2_seeds=0):
    any_fw = tot = any_after = 0
    for _ in range(N):
        ref = [random.gauss(0, SD) for _ in range(2)]
        wins = 0
        survive = 0
        for _a in range(k):
            d = [random.gauss(0, SD) - ref[s] for s in range(2)]
            if sum(d) / 2 <= THR and all(x < 0 for x in d):
                wins += 1
                if a2_seeds:
                    ref2 = [random.gauss(0, SD) for _ in range(a2_seeds)]
                    d2 = [random.gauss(0, SD) - ref2[s] for s in range(a2_seeds)]
                    survive += sum(d2) / a2_seeds <= THR
        any_fw += wins > 0
        tot += wins
        any_after += survive > 0
    return {"k_arms": k, "P_at_least_one_false_winner": any_fw / N, "expected_false_winners": tot / N,
            **({"P_false_winner_survives_A2_3_fresh_seeds": any_after / N} if a2_seeds else {})}


out = {"model": "null effects, per-run log sd 0.0053, shared E27 reference, A1 winner rule", "n_sim": N,
       "k8": sim(8), "k11_with_A2": sim(11, a2_seeds=3)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "a1_false_winner_sim.json"), "w",
               newline="\n"), indent=1)
print(json.dumps(out, indent=1))
