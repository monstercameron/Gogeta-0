"""Code-native state-tracking probe (outside review idea A): inference only, no training, no code execution.

A short module-level Python snippet reassigns ONE target variable k times (mode `assign`: `n = 47`; mode `incr`:
first `n = 47`, then k-1 updates `n += 3`), interleaved with assignments to OTHER variables, and then `distance`
distractor lines (assignments to other variables only) separate the last update from the query line
`assert n ==`. The model must give the highest likelihood to the CURRENT value among 4 choices:
  correct value  +  stale values of the same variable (earlier states)  +  current values of other variables
  (+ random 2-digit values only if fewer than 3 distinct distractors exist).
Chance = 25% (4 choices, correct position irrelevant: scoring is per-choice likelihood, not position).

Difficulty axes: k (how many updates must be tracked: k=1 is plain recall, k>1 needs 'latest wins', and in `incr`
mode, arithmetic on the state) and distance (lines of distractors between last update and query).

Everything here is pure python and seeded: item i of cell (mode,k,distance,seed) depends only on those four numbers.
"""
from __future__ import annotations

import random

NAMES = ["count", "total", "idx", "n", "acc", "size", "limit", "offset", "step", "level", "depth", "width",
         "height", "score", "rate", "base", "mask", "shift", "head", "tail", "pos", "left", "right", "mid"]
N_CHOICES = 4


def make_item(mode: str, k: int, distance: int, seed: int, i: int, gap: int = 2) -> dict:
    """One probe item. gap = distractor lines between consecutive updates of the target."""
    r = random.Random(f"state-{mode}-{k}-{distance}-{seed}-{i}")
    names = r.sample(NAMES, 7)
    target, others = names[0], names[1:]
    other_val = {o: r.randint(10, 99) for o in others}
    lines: list[str] = []

    def distractor():
        o = r.choice(others)
        other_val[o] = r.randint(10, 99)
        lines.append(f"{o} = {other_val[o]}")

    # make sure every other variable is bound once up front (so their values are valid distractor choices)
    for o in others[:3]:
        lines.append(f"{o} = {other_val[o]}")
    states: list[int] = []
    for j in range(k):
        for _ in range(gap if j else 1):
            distractor()
        if mode == "assign" or j == 0:
            v = r.randint(10, 99)
            while states and v == states[-1]:
                v = r.randint(10, 99)
            lines.append(f"{target} = {v}")
        else:
            c = r.randint(1, 9)
            v = states[-1] + c
            lines.append(f"{target} += {c}")
        states.append(v)
    for _ in range(distance):
        distractor()
    cur = states[-1]
    pool: list[int] = []
    for v in reversed(states[:-1]):                       # most recent stale value first
        if v != cur and v not in pool:
            pool.append(v)
    n_stale = len(pool[: N_CHOICES - 1])
    for o in others:                                      # then the other variables' current values
        v = other_val[o]
        if v != cur and v not in pool:
            pool.append(v)
    while len(pool) < N_CHOICES - 1:
        v = r.randint(10, 99)
        if v != cur and v not in pool:
            pool.append(v)
    distract = pool[: N_CHOICES - 1]
    choices = [cur] + distract
    r.shuffle(choices)
    return {"prompt": "\n".join(lines) + f"\nassert {target} ==", "choices": [f" {c}" for c in choices],
            "gold": choices.index(cur), "k": k, "distance": distance, "mode": mode, "n_stale_distractors": n_stale, "stale": [f" {v}" for v in pool[:n_stale]],
            "lines": len(lines) + 1}


def make_cell(mode: str, k: int, distance: int, n: int = 1100, seed: int = 0) -> list[dict]:
    return [make_item(mode, k, distance, seed, i) for i in range(n)]


# the frozen grid (battery v1): 18 cells x 1100 items. Worst-case 95% CI half-width per cell
# 1.96*sqrt(.25/1100) = 2.95 points (binary accuracy, p = .5); aggregated rows are tighter.
GRID = {"assign": [(k, d) for k in (1, 4, 8) for d in (0, 16, 64)],
        "incr": [(k, d) for k in (2, 4, 8) for d in (0, 16, 64)]}
N_PER_CELL = 1100

if __name__ == "__main__":
    for m, k, d in (("assign", 3, 2), ("incr", 3, 2)):
        it = make_item(m, k, d, 0, 0)
        print(it["prompt"], "\n  choices:", it["choices"], "gold:", it["gold"], "\n")
