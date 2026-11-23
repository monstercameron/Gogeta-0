"""SQ225-extension parseability census (feeds SQ239): does each v1 JSONL doc parse as Python?

A doc "parses" if `ast.parse(text)` succeeds on the whole text, or, when the text contains fenced code blocks (```python ... ```), every fenced block parses (the algorithmic
corpus docs are problem statement + fenced solution + asserts, so the whole text usually does not parse). Census by source config and by split: all 4,000 val docs (and the
clean80 subset, K9 overlap < 0.8), and a seeded random sample of 20,000 train docs (stream over docs_train.jsonl; every line is read, 20,000 are parsed). The per-val-doc
flag is saved for the stratified re-score: results/sq239_parse_flags.npy (bool, indexed like docs_val.jsonl). Rule from the SQ225 row: a source gap above 10 points or a
train/val gap above 5 points adds a parseability stratum to SQ239.

Run (WSL): python experiments/checks/sq239_parseability.py -> results/sq239_parseability.json
"""
import ast
import json
import os
import random
import re
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
FENCE = re.compile(r"```[ \t]*(?:python3|python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def parses(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        pass
    blocks = FENCE.findall(text)
    if not blocks:
        return False
    for b in blocks:
        try:
            ast.parse(b)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return False
    return True


def main():
    rows = [json.loads(l) for l in open(os.path.join(D, "docs_val.jsonl"), encoding="utf-8")]
    ov = json.load(open(os.path.join(ROOT, "experiments/checks/results/K9.json")))["data"]["val_per_doc_overlap"]
    clean80 = np.array([o is None or o < 0.8 for o in ov])
    flags = np.array([parses(r["text"]) for r in rows])
    src = np.array([r["src"] for r in rows])
    out = {"check": "SQ239-parseability", "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "status": "descriptive census", "val": {}, "clean80": {}, "train_sample": {}}
    for name, mask in (("val", np.ones(len(flags), bool)), ("clean80", clean80)):
        out[name]["all"] = {"n": int(mask.sum()), "parse_rate": float(flags[mask].mean())}
        for c in sorted(set(src)):
            m = mask & (src == c)
            out[name][c] = {"n": int(m.sum()), "parse_rate": float(flags[m].mean())}
    # train sample
    rng = random.Random(0)
    n_lines = sum(1 for _ in open(os.path.join(D, "docs_train.jsonl"), "rb"))
    pick = set(rng.sample(range(n_lines), 20000))
    res = {}
    with open(os.path.join(D, "docs_train.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in pick:
                d = json.loads(line)
                res.setdefault(d["src"], []).append(parses(d["text"]))
    allp = [x for v in res.values() for x in v]
    out["train_sample"]["all"] = {"n": len(allp), "parse_rate": float(np.mean(allp))}
    for c, v in res.items():
        out["train_sample"][c] = {"n": len(v), "parse_rate": float(np.mean(v))}
    cfgs = sorted(set(src))
    out["source_gap_points_val"] = 100 * abs(out["val"][cfgs[0]]["parse_rate"] - out["val"][cfgs[1]]["parse_rate"])
    out["train_val_gap_points"] = 100 * abs(out["train_sample"]["all"]["parse_rate"] - out["val"]["all"]["parse_rate"])
    out["adds_parseability_stratum"] = bool(out["source_gap_points_val"] > 10 or out["train_val_gap_points"] > 5)
    json.dump(out, open(os.path.join(ROOT, "experiments/checks/results/sq239_parseability.json"), "w"), indent=1)
    np.save(os.path.join(ROOT, "experiments/checks/results/sq239_parse_flags.npy"), flags)
    print(json.dumps({k: out[k] for k in ("val", "clean80", "train_sample", "source_gap_points_val", "train_val_gap_points", "adds_parseability_stratum")}, indent=1))


if __name__ == "__main__":
    main()
