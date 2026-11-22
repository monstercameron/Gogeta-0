"""group_proxy.py: the raw OpenCoder parquet has NO source/repo/project id (columns: type, lang, text only; `type` is
'qa' for algorithmic_corpus and 'snippet' for synthetic_code_snippet, already kept as docs_*.jsonl `src`). This script
measures a PROXY group, the first top-level function/class name of a doc ("problem identity" for LeetCode-style
items), and how often it crosses a naive doc-level split (val vs train), against the MinHash families.

    python group_proxy.py --run     # reads docs_train.jsonl/docs_val.jsonl (~2 min); writes results/group_proxy.json
Cockpit function: run().
"""
import collections, json, os, pathlib, re, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DS = DATA / "datasets" / "code-py-v1"
DEF = re.compile(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.M)
GENERIC = {"main", "solution", "Solution", "solve", "test", "run", "__init__", "f", "foo", "func", "function"}


def first_name(t):
    m = DEF.search(t)
    return m.group(1) if m else None


def run():
    train, srcs = [], collections.Counter()
    for line in open(DS / "docs_train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        train.append(first_name(r["text"]))
        srcs[r["src"]] += 1
    val = []
    vsrc = collections.Counter()
    for line in open(DS / "docs_val.jsonl", encoding="utf-8"):
        r = json.loads(line)
        val.append(first_name(r["text"]))
        vsrc[r["src"]] += 1
    cnt = collections.Counter(n for n in train if n)
    spec = [n for n in val if n and n not in GENERIC]
    out = {"raw_columns": ["type", "lang", "text"], "source_id_available": False,
           "src_counts_train": dict(srcs), "src_counts_val": dict(vsrc),
           "train_docs_with_def": int(sum(1 for n in train if n)), "distinct_first_names_train": len(cnt),
           "val_docs_with_specific_name": len(spec),
           "val_frac_name_in_train": round(float(np.mean([cnt[n] > 0 for n in spec])), 4),
           "val_frac_name_in_train_ge5": round(float(np.mean([cnt[n] >= 5 for n in spec])), 4),
           "train_top_names": cnt.most_common(8),
           "train_docs_in_names_with_ge2_docs": int(sum(c for c in cnt.values() if c >= 2)),
           "note": "name groups are a coarse proxy: a shared function name is not proof of a shared source"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "group_proxy.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--run" in sys.argv:
        print(json.dumps(run(), indent=1))
    else:
        print(__doc__)
