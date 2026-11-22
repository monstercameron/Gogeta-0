"""lm-eval glue for the code-native state-tracking probe (generator: evals/battery/state_track.py)."""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_state_track(mode: str = "assign", k: int = 1, distance: int = 0, n: int = 1100, seed: int = 0, **kw):
    import datasets
    from evals.battery.state_track import make_cell
    rows = make_cell(mode, int(k), int(distance), int(n), int(seed))
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def doc_prompt(doc):
    return doc["prompt"]


def doc_choices(doc):
    return doc["choices"]


def doc_gold(doc):
    return doc["gold"]
