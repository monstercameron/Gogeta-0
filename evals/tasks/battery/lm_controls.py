"""Negative control for the language-modelling family: the same val docs with their LINES shuffled (seeded).
A model that has learned code structure (not just byte statistics) must score clearly worse on it."""
from __future__ import annotations

import random
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_val_lineshuffled(n: int = 1000, seed: int = 0, **kw):
    from evals.tasks import gogeta_utils as U
    docs = U._read_jsonl(U.VAL_DOCS)[: int(n)]
    rows = []
    for i, d in enumerate(docs):
        lines = d["text"].split("\n")
        random.Random(f"lineshuf-{seed}-{i}").shuffle(lines)
        rows.append({"text": "\n".join(lines), "src": d["src"]})
    return U._dsdict(rows)
