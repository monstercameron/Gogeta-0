"""benchmark_contamination.py: per-item contamination labels for HumanEval(+), MBPP(+), CRUXEval and the UncheatableEval
GitHub-Python slice against code-py-v1 train (+ val), using token n-gram presence, MinHash/char-5-gram Jaccard and
the >= 50-character normalised substring check. Upgrades evals/contamination.py (line-level screen; C103).

Method per item component (prompt / solution / tests / input+output; CRUXEval: code, io; uncheatable: whole file):
  1. tokenise with the code-py-v1 tokenizer; 13-gram and 8-gram PRESENCE in train.bin+val.bin (dedup.presence_in_streams:
     fraction of the component's unique n-grams that occur anywhere; no boilerplate drop);
  2. candidate train docs: dedup.scan_token_stream (8-grams, boilerplate-capped) -> the 3 docs sharing the most 8-grams
     with the component; their text is fetched from docs_train.jsonl and verified with dedup.compare_texts
     (char-5 Jaccard, containment, longest common whitespace-normalised substring >= 50 chars);
  3. labels (thresholds fixed here before the run, see PREDICTION/CRITERION):
        contaminated : any prompt/solution/code/whole-file component has 13-gram presence >= 0.50, OR a verified common
                       substring >= 100 chars with a train doc
        suspect      : not contaminated, and any such component has 8-gram presence >= 0.25 or 13-gram >= 0.25 or a
                       verified substring >= 50 chars
        clean        : otherwise
     `exclude_for_clean_twin` = contaminated or suspect (conservative). Tests/IO components are reported, never gate.

    python benchmark_contamination.py --plan
    python benchmark_contamination.py --run         # real data, ~5 CPU-min, < 2 GB
Output: results/benchmark-contamination.json, proposed-contaminated_ids.json (same keys as evals/tasks/contaminated_ids.json
plus the new benchmarks; PROPOSED: evals/ is not edited here).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dedup  # noqa: E402

DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
CACHE = pathlib.Path(os.environ.get("GOGETA_EVAL_CACHE", DATA / "eval-cache"))
DS = DATA / "datasets" / "code-py-v1"
RESULTS = HERE / "results"
GATE_COMPONENTS = ("prompt", "solution", "code", "file")

PREDICTION = (
    "Written 2026-11-22 before the first real-data run. (1) The existing screen under-counts: the new strict "
    "'contaminated' count is >= the old 32/164 HumanEval and 43/500 MBPP (token n-gram presence also catches reformatted "
    "copies), i.e. HumanEval strict 32-70 of 164, MBPP strict 43-150 of 500. (2) CRUXEval (800 functions) has far less: "
    "< 10% contaminated (its functions are LLM-written, not the algorithmic corpus). (3) HumanEval+/MBPP+ labels agree "
    "with HumanEval/MBPP on > 95% of shared task ids. (4) The UncheatableEval GitHub slice (post-cutoff) has < 5% "
    "contaminated files and < 10% suspect.")
CRITERION = (
    "Descriptive; the thresholds are fixed above. Prediction (1) is judged by the strict counts; (2)-(4) by the stated "
    "shares. Any clause that fails is reported as a miss. The labels, not the clauses, are the deliverable: the "
    "`*_clean` eval twins should exclude `exclude_for_clean_twin` items.")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_items() -> dict:
    os.environ["HF_HOME"] = str(CACHE / "hf")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets
    out: dict = {}
    he = datasets.load_dataset("openai/openai_humaneval", split="test")
    out["humaneval"] = {d["task_id"]: {"prompt": d["prompt"], "solution": d["canonical_solution"], "tests": d["test"]} for d in he}
    hp = datasets.load_dataset("evalplus/humanevalplus", split="test")
    out["humaneval_plus"] = {d["task_id"]: {"prompt": d["prompt"], "solution": d["canonical_solution"], "tests": d["test"]} for d in hp}
    mb = datasets.load_dataset("google-research-datasets/mbpp", "full", split="test")
    out["mbpp"] = {str(d["task_id"]): {"prompt": d["text"], "solution": d["code"], "tests": "\n".join(d["test_list"])} for d in mb}
    mp = datasets.load_dataset("evalplus/mbppplus", split="test")
    out["mbpp_plus"] = {str(d["task_id"]): {"prompt": d["prompt"], "solution": d["code"], "tests": str(d["test_list"])} for d in mp}
    cx = datasets.load_dataset("cruxeval-org/cruxeval", split="test")
    out["cruxeval"] = {d["id"]: {"code": d["code"], "io": f"f({d['input']}) == {d['output']}"} for d in cx}
    p = CACHE / "uncheatable" / "2026-10" / "github_python.jsonl"
    if p.exists():
        rows = [json.loads(ln) for ln in open(p, encoding="utf-8")]
        out["uncheatable_github"] = {str(i): {"file": (r.get("content") or r.get("text") or "")} for i, r in enumerate(rows)}
    return out


def run(limit: int | None = None) -> dict:
    from tokenizers import Tokenizer
    t0 = time.time()
    tok = Tokenizer.from_file(str(DS / "tokenizer.json"))
    items = load_items()
    comps: dict = {}   # (bench, id, comp) -> (text, ids)
    for b, d in items.items():
        for k, (iid, cs) in enumerate(d.items()):
            if limit and k >= limit:
                break
            for c, t in cs.items():
                if t and t.strip():
                    comps[(b, iid, c)] = t
    keys = list(comps)
    enc = tok.encode_batch([comps[k] for k in keys], add_special_tokens=False)
    ids = {k: np.asarray(e.ids, dtype=np.int64) for k, e in zip(keys, enc)}
    streams = [("train", DS / "train.bin"), ("val", DS / "val.bin")]
    log(f"{len(keys)} components; presence scan n=13 ...")
    p13 = dedup.presence_in_streams({k: ids[k] for k in keys}, streams, n=13)
    log("presence scan n=8 ...")
    p8 = dedup.presence_in_streams({k: ids[k] for k in keys}, streams, n=8)
    log("candidate-doc scan (8-grams, boilerplate-capped) ...")
    sc = dedup.scan_token_stream({k: ids[k] for k in keys if p8[k]["overlap"]}, streams, n=8, max_occ=2000)
    top = {}
    for k, pr in sc["pairs"].items():
        top[k] = sorted(pr.items(), key=lambda kv: -kv[1])[:3]
    # fetch the candidate docs' text
    need = {"train": set(), "val": set()}
    for v in top.values():
        for (nm, d), _ in v:
            need[nm].add(d)
    texts = {}
    for nm, fn in (("train", "docs_train.jsonl"), ("val", "docs_val.jsonl")):
        with open(DS / fn, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in need[nm]:
                    texts[(nm, i)] = json.loads(line)["text"]
    log(f"fetched {len(texts)} candidate docs")
    per_comp = {}
    for k in keys:
        o13, o8 = p13[k]["overlap"], p8[k]["overlap"]
        best = {"substring": 0, "char5_containment": 0.0, "char5_jaccard": 0.0, "doc": None}
        for (nm, d), shared in top.get(k, []):
            cmp = dedup.compare_texts(comps[k], texts[(nm, d)])
            if cmp["substring"]["length"] > best["substring"] or (best["doc"] is None):
                best = {"substring": cmp["substring"]["length"], "char5_containment": cmp["char5"]["a_in_b"],
                        "char5_jaccard": cmp["char5"]["jaccard"], "doc": [nm, d], "shared_8grams": shared}
        per_comp[k] = {"ngram13_presence": None if o13 is None else round(o13, 3),
                       "ngram8_presence": None if o8 is None else round(o8, 3), "n_tokens": int(len(ids[k])), **best}
    res = {"created": dt.datetime.now().isoformat(timespec="seconds"), "prediction": PREDICTION, "criterion": CRITERION,
           "reference": "code-py-v1 train.bin + val.bin (n-gram presence); docs_train.jsonl/docs_val.jsonl (verification)",
           "benchmarks": {}}
    flat_ids: dict = {}
    for b, d in items.items():
        bres, n_c, n_s = {}, 0, 0
        for iid in d:
            cs = {c: per_comp[(b, iid, c)] for c in d[iid] if (b, iid, c) in per_comp}
            if not cs:
                continue
            gate = [v for c, v in cs.items() if c in GATE_COMPONENTS]
            strict = any((v["ngram13_presence"] or 0) >= 0.5 or v["substring"] >= 100 for v in gate)
            suspect = (not strict) and any((v["ngram8_presence"] or 0) >= 0.25 or (v["ngram13_presence"] or 0) >= 0.25
                                           or v["substring"] >= 50 for v in gate)
            label = "contaminated" if strict else ("suspect" if suspect else "clean")
            n_c += strict
            n_s += suspect
            bres[iid] = {"label": label, "exclude_for_clean_twin": bool(strict or suspect), "components": cs}
        res["benchmarks"][b] = {"items": len(bres), "contaminated": int(n_c), "suspect": int(n_s),
                                "clean": len(bres) - int(n_c) - int(n_s), "per_item": bres}
        flat_ids[b] = [i for i, v in bres.items() if v["exclude_for_clean_twin"]]
    res["old_screen_C103"] = {"humaneval_flagged": 32, "mbpp_flagged": 43}
    res["seconds"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    tag = f"_limit{limit}" if limit else ""
    (RESULTS / f"benchmark-contamination{tag}.json").write_text(json.dumps(res, indent=1))
    prop = {"source": "experiments/data_integrity/benchmark_contamination.py", "created": res["created"],
            "rule": "exclude items labelled contaminated or suspect (see script docstring)",
            "humaneval": flat_ids.get("humaneval", []), "mbpp": [int(x) for x in flat_ids.get("mbpp", [])],
            "humaneval_plus": flat_ids.get("humaneval_plus", []), "mbpp_plus": [int(x) for x in flat_ids.get("mbpp_plus", [])],
            "cruxeval": flat_ids.get("cruxeval", []), "uncheatable_github_file_idx": flat_ids.get("uncheatable_github", [])}
    (RESULTS / f"proposed-contaminated_ids{tag}.json").write_text(json.dumps(prop, indent=1))
    return {b: {k: v for k, v in r.items() if k != "per_item"} for b, r in res["benchmarks"].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N items per benchmark (smoke test)")
    a = ap.parse_args()
    if a.plan:
        print(json.dumps({"prediction": PREDICTION, "criterion": CRITERION, "reads": [str(DS), str(CACHE)],
                          "writes": [str(RESULTS)]}, indent=1))
    if a.run:
        print(json.dumps(run(a.limit), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
