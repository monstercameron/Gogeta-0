"""First-pass contamination screen: do HumanEval / MBPP problems appear in our training docs?

    ~/.venvs/gogeta-eval/bin/python -m evals.contamination        (WSL, repo root; ~2-4 min, 1 core)

Why: our corpus is OpenCoder's annealing data (algorithmic_corpus + synthetic_code_snippet).
The OpenCoder paper (arXiv 2411.04905 §4.4) describes 10-gram decontamination for its SFT data;
we found no statement that the annealing corpus was decontaminated. Its algorithmic_corpus
docs use MBPP's exact layout (statement, 3 asserts, fenced solution), so leakage is plausible.

Method (line-level exact match, whitespace-insensitive, case-sensitive), streamed over
docs_train.jsonl (987k docs):
  * MBPP (full/test, 500): the problem statement, and each of the 3 test asserts.
    A problem is flagged if its statement OR >= 2 of its asserts occur as a line in a train doc.
  * HumanEval (164): docstring lines and canonical-solution lines of >= 40 characters.
    A problem is flagged if >= 2 such lines occur in train docs.
This is a screen, not proof: it misses paraphrases and reformatting (false negatives) and can
flag generic lines (false positives; the 40-char / 2-line thresholds limit that).
Writes evals/results/contamination-<date>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
CACHE = pathlib.Path(os.environ.get("GOGETA_EVAL_CACHE", DATA / "eval-cache"))
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
TRAIN = DATA / "datasets" / "code-py-v1" / "docs_train.jsonl"
WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub("", s)


def main() -> int:
    import datasets
    mbpp = datasets.load_dataset("google-research-datasets/mbpp", "full", split="test")
    he = datasets.load_dataset("openai/openai_humaneval", split="test")

    keys: dict[str, set] = defaultdict(set)  # normalized line -> {(bench, id, kind)}
    for d in mbpp:
        keys[norm(d["text"])].add(("mbpp", d["task_id"], "statement"))
        for t in d["test_list"]:
            keys[norm(t)].add(("mbpp", d["task_id"], "assert"))
    he_lines = {}
    for d in he:
        doc = d["prompt"].split('"""', 1)[-1] if '"""' in d["prompt"] else d["prompt"]
        lines = [ln for ln in (doc + "\n" + d["canonical_solution"]).splitlines() if len(ln.strip()) >= 40]
        he_lines[d["task_id"]] = len(lines)
        for ln in lines:
            keys[norm(ln)].add(("humaneval", d["task_id"], "line"))

    hits: dict[tuple, set] = defaultdict(set)  # (bench, id, kind) -> set of matched normalized lines
    docs_with_hit = 0
    t0 = time.time()
    with open(TRAIN, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            text = json.loads(line)["text"]
            found = False
            for ln in text.splitlines():
                k = norm(ln)
                if k in keys:
                    found = True
                    for tag in keys[k]:
                        hits[tag].add(k)
            docs_with_hit += found
            if n % 200_000 == 0:
                print(f"  {n} docs, {time.time() - t0:.0f}s", flush=True)

    mbpp_flag, he_flag = [], []
    for d in mbpp:
        tid = d["task_id"]
        st = bool(hits.get(("mbpp", tid, "statement")))
        na = len(hits.get(("mbpp", tid, "assert"), ()))
        if st or na >= 2:
            mbpp_flag.append({"task_id": tid, "statement_match": st, "assert_lines_matched": na})
    for d in he:
        tid = d["task_id"]
        nl = len(hits.get(("humaneval", tid, "line"), ()))
        if nl >= 2:
            he_flag.append({"task_id": tid, "long_lines_matched": nl, "long_lines_total": he_lines[tid]})

    out = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "train_docs_scanned": n, "train_docs_with_any_match": docs_with_hit,
        "method": "line-level exact match, whitespace removed; see evals/contamination.py docstring",
        "mbpp": {"split": "full/test", "problems": len(mbpp), "flagged": len(mbpp_flag),
                 "flagged_statement": sum(x["statement_match"] for x in mbpp_flag), "items": mbpp_flag},
        "humaneval": {"problems": len(he), "flagged": len(he_flag), "items": he_flag},
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = ROOT / "evals" / "results" / f"contamination-{dt.date.today().isoformat()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "items"})
                      for k, v in out.items()}, indent=2))
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
