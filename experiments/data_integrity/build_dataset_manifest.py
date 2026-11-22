"""build_dataset_manifest.py: dataset.manifest.json for train / val / test (sha256 per file and per 64M-token block of the
.bin shards, doc and token counts, tokenizer hash, link to the raw-source files). Read-only on the data.

    python build_dataset_manifest.py            # writes experiments/data_integrity/dataset.manifest.json (~1-2 CPU-min)
    python build_dataset_manifest.py --check    # re-hash and compare with the committed manifest
Cockpit functions: build(), check().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DS = DATA / "datasets" / "code-py-v1"
OUT = HERE / "dataset.manifest.json"
BLOCK_TOK = 64 * 1024 * 1024


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bin_info(p: pathlib.Path) -> dict:
    arr = np.memmap(p, dtype=np.uint16, mode="r")
    blocks = []
    for i in range(0, len(arr), BLOCK_TOK):
        blocks.append(hashlib.sha256(np.asarray(arr[i:i + BLOCK_TOK]).tobytes()).hexdigest())
    return {"tokens": int(len(arr)), "docs": int((np.asarray(arr) == 0).sum()), "max_token_id": int(np.asarray(arr).max()),
            "block_tokens": BLOCK_TOK, "block_sha256": blocks}


def build() -> dict:
    meta = json.loads((DS / "meta.json").read_text())
    tok_sha = sha(DS / "tokenizer.json")
    man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": "code-py-v1", "source": meta["source"],
           "license": meta["license"], "tokenizer": {"file": "code-py-v1/tokenizer.json", "sha256": tok_sha,
                                                      "vocab_size": meta["vocab_size"], "eos_id": meta["eos_id"]},
           "prepare_seed": meta["seed"], "filter": meta["filter"], "splits": {}, "known_issues": []}
    for split, (docs_f, bin_f) in {"train": ("docs_train.jsonl", "train.bin"), "val": ("docs_val.jsonl", "val.bin")}.items():
        info = bin_info(DS / bin_f)
        assert info["docs"] == meta["splits"][split]["docs"] and info["tokens"] == meta["splits"][split]["tokens"], split
        man["splits"][split] = {"docs": info["docs"], "tokens": info["tokens"], "chars": meta["splits"][split]["chars"],
                                "files": {docs_f: {"sha256": sha(DS / docs_f), "bytes": (DS / docs_f).stat().st_size},
                                          bin_f: {"sha256": sha(DS / bin_f), "bytes": (DS / bin_f).stat().st_size,
                                                  "block_tokens": info["block_tokens"], "block_sha256": info["block_sha256"]}}}
    tm = HERE / "test-split.manifest.json"
    if tm.exists():
        man["splits"]["test"] = {k: {kk: vv for kk, vv in v.items() if kk in ("docs", "tokens", "mode", "files", "seed")}
                                 for k, v in json.loads(tm.read_text())["splits"].items()}
    man["known_issues"] = [
        "K9 (v0.4.32 era): 41.7% of val docs have >= 80% of their 13-grams in train; val is not held out in any useful sense. "
        "Use clean-val.manifest.json subsets and the frozen test split.",
        "Train is massively self-duplicated (see results/families.json)."]
    OUT.write_text(json.dumps(man, indent=1))
    return {k: (v["docs"], v["tokens"]) for k, v in man["splits"].items() if "docs" in v}


def check() -> dict:
    man = json.loads(OUT.read_text())
    res = {}
    for split, s in man["splits"].items():
        groups = {split: (DS, s)} if "files" in s else {k: (DATA / "datasets" / k, v) for k, v in s.items()}
        for gname, (d, g) in groups.items():
            for fn, v in g["files"].items():
                res[f"{gname}/{fn}"] = sha(d / fn) == v["sha256"]
    res["tokenizer"] = sha(DS / "tokenizer.json") == man["tokenizer"]["sha256"]
    res["all_ok"] = all(res.values())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    print(json.dumps(check() if a.check else build(), indent=1))
    sys.exit(0)
