"""Fetch the small external eval datasets into the eval cache (never into the repo).

    python -m evals.fetch_data uncheatable [--snapshot 2026-10] [--category github_python]
    python -m evals.fetch_data hf          # HumanEval, MBPP (+ EvalPlus) via `datasets` into HF_HOME

Cache root: $GOGETA_EVAL_CACHE, default ~/fusiondance-data/eval-cache. run_suite.py points
HF_HOME / HF_DATASETS_CACHE there too.

UncheatableEval (Jellyfish042/UncheatableEval-<snapshot>) is one 231 MB parquet with 15
categories x 500 docs (incl. an `untruncated_content` column we do not need). To stay small we
pull only one category through the Hugging Face datasets-server `/filter` API (100 rows per
page) and keep the `content` column, which is what the upstream evaluator scores
(uncheatable_eval/evaluator.py: `content = row["content"]`). A manifest records the row count,
bytes downloaded and a sha256 of the written file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
import urllib.parse
import urllib.request

CACHE = pathlib.Path(os.environ.get("GOGETA_EVAL_CACHE",
                                    pathlib.Path.home() / "fusiondance-data" / "eval-cache"))
UA = {"User-Agent": "gogeta-0-evals/0.1"}
DS_SERVER = "https://datasets-server.huggingface.co"


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def uncheatable_path(snapshot: str, category: str) -> pathlib.Path:
    return CACHE / "uncheatable" / snapshot / f"{category}.jsonl"


def fetch_uncheatable(snapshot: str = "2026-10", category: str = "github_python") -> dict:
    out = uncheatable_path(snapshot, category)
    if out.exists() and out.with_suffix(".manifest.json").exists():
        return json.loads(out.with_suffix(".manifest.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset = f"Jellyfish042/UncheatableEval-{snapshot}"
    where = urllib.parse.quote(f"\"category\"='{category}'")
    rows, downloaded, offset = [], 0, 0
    while True:
        url = (f"{DS_SERVER}/filter?dataset={urllib.parse.quote(dataset)}&config=default&split=test"
               f"&where={where}&offset={offset}&length=100")
        raw = _get(url)
        downloaded += len(raw)
        page = json.loads(raw)
        got = [r["row"] for r in page.get("rows", [])]
        for r in got:
            rows.append({"content": r["content"], "category": r["category"], "date": r["date"], "url": r["url"]})
        offset += len(got)
        total = page.get("num_rows_total", offset)
        if not got or offset >= total:
            break
    tmp = out.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    man = {"dataset": dataset, "category": category, "rows": len(rows),
           "bytes_downloaded": downloaded, "bytes_written": out.stat().st_size,
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "dates": [min(r["date"] for r in rows), max(r["date"] for r in rows)] if rows else None,
           "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "datasets-server /filter"}
    out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2))
    return man


HF_SETS = [  # (path, name, split) used by tier-1/2 tasks; all are small parquet files
    ("openai/openai_humaneval", None, "test"),
    ("google-research-datasets/mbpp", "full", "test"),
    ("evalplus/humanevalplus", None, "test"),
    ("evalplus/mbppplus", None, "test"),
]


def fetch_hf(which: list[tuple] = HF_SETS) -> list[dict]:
    import datasets
    out = []
    for path, name, split in which:
        ds = datasets.load_dataset(path, name, split=split)
        out.append({"dataset": path, "name": name, "split": split, "rows": len(ds),
                    "fingerprint": ds._fingerprint})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["uncheatable", "hf"])
    ap.add_argument("--snapshot", default="2026-10")
    ap.add_argument("--category", default="github_python")
    a = ap.parse_args()
    if a.what == "uncheatable":
        print(json.dumps(fetch_uncheatable(a.snapshot, a.category), indent=2))
    else:
        os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
        print(json.dumps(fetch_hf(), indent=2))


if __name__ == "__main__":
    main()
