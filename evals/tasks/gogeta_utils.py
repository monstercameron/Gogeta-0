"""Dataset loaders, prompt builders and metrics for the Gogeta-0 lm-eval tasks (evals/tasks/*.yaml).

Conventions
  * bits-per-byte (bpb) uses lm-eval's own `bits_per_byte` aggregation:
    -sum(loglikelihood) / sum(utf-8 bytes) / ln 2, i.e. a byte-weighted corpus average.
    bits-per-char (bpc) is the same with Python characters as the weight; train.py reports bpc.
  * Execution-based metrics never run code in this process. Programs go to
    evals/sandbox.run_tests (rewards.py's harness + private mount/network/user namespaces).
  * pass@k uses the unbiased estimator of Chen et al. 2021 (arXiv 2107.03374, Fig. 3):
    1 - C(n-c, k) / C(n, k), averaged over problems.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
CACHE = pathlib.Path(os.environ.get("GOGETA_EVAL_CACHE", DATA / "eval-cache"))
VAL_DOCS = DATA / "datasets" / "code-py-v1" / "docs_val.jsonl"


def _dsdict(rows: list[dict]):
    import datasets
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def _read_jsonl(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing (see evals/README.md, 'Data')")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------- tier 1: bpb
def load_val_docs(**kwargs):
    """The 4,000 held-out docs of code-py-v1 (tools/prepare_data.py), in file order."""
    return _dsdict([{"text": d["text"], "src": d["src"]} for d in _read_jsonl(VAL_DOCS)])


def load_uncheatable(snapshot: str = "2026-10", category: str = "github_python", **kwargs):
    """Fresh GitHub code from UncheatableEval (fetched by `python -m evals.fetch_data uncheatable`)."""
    p = CACHE / "uncheatable" / snapshot / f"{category}.jsonl"
    return _dsdict([{"text": d["content"], "date": d["date"]} for d in _read_jsonl(p)])


def process_bpb_rolling(doc, results):
    (ll,) = results
    text = doc["text"]
    return {"bits_per_byte": (ll, len(text.encode("utf-8"))), "bits_per_char": (ll, len(text))}


def process_bpb_target(doc, results):
    """Conditional bpb of the reference solution given the prompt (loglikelihood requests)."""
    ll, _is_greedy = results[0] if isinstance(results[0], (tuple, list)) else results
    return _bpb_items(ll, doc)


def _scored_text(doc) -> str:
    """The text lm-eval actually scores for (prompt, target): HFLM._encode_pair moves the
    prompt's trailing whitespace (here the newline after ```python / the docstring) onto the
    continuation, so it is scored too and must count in the denominator."""
    p = doc["_prompt"]
    return p[len(p.rstrip()):] + doc["_target"]


def _bpb_items(ll, doc) -> dict:
    s = _scored_text(doc)
    return {"bits_per_byte": (ll, len(s.encode("utf-8"))), "bits_per_char": (ll, len(s))}


# ----------------------------------------------------------------------------- prompt formats
def _lf(s: str) -> str:
    return s.replace("\r\n", "\n")


def native_prompt(statement: str, asserts: list[str]) -> str:
    """Our pretraining format (OpenCoder algorithmic_corpus): statement, asserts, fenced solution."""
    return statement.strip() + "\n\n" + "\n".join(a.strip() for a in asserts) + "\n\n```python\n"


def process_mbpp(dataset):
    def f(d):
        tests = [_lf(t) for t in d["test_list"]]
        setup = _lf(d.get("test_setup_code") or "")
        return {"_prompt": native_prompt(d["text"], tests), "_target": _lf(d["code"]).strip() + "\n```",
                "_tests": tests, "_setup": setup}
    return dataset.map(f)


def process_mbpp_plus(dataset):
    def f(d):
        base = [_lf(t) for t in d["test_list"]]
        imports = "\n".join(d.get("test_imports") or [])
        return {"_prompt": native_prompt(d["prompt"], base), "_target": _lf(d["code"]).strip() + "\n```",
                "_tests": [imports + "\n" + _lf(d["test"])], "_setup": imports}
    return dataset.map(f)


def process_humaneval(dataset):
    def f(d):
        return {"_prompt": d["prompt"], "_target": d["canonical_solution"],
                "_tests": [d["test"] + f"\ncheck({d['entry_point']})\n"], "_setup": ""}
    return dataset.map(f)


_FENCE = re.compile(r"```[ \t]*(?:python3|python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def load_heldout_exec(**kwargs):
    """Held-out functional set: val docs from algorithmic_corpus with top-level asserts and a
    fenced solution. Prompt = statement + asserts + "```python\\n"; tests = the asserts."""
    from post_training.rewards import extract_asserts
    rows = []
    for i, d in enumerate(_read_jsonl(VAL_DOCS)):
        if d["src"] != "algorithmic_corpus":
            continue
        text = d["text"]
        asserts = extract_asserts(text)
        m = _FENCE.search(text)
        if not asserts or not m:
            continue
        cut = min([text.find("```")] + [text.find(a) for a in asserts if text.find(a) >= 0])
        statement = text[:cut].strip()
        if not statement:
            continue
        rows.append({"val_index": i, "_prompt": native_prompt(statement, asserts),
                     "_target": m.group(1).rstrip("\n") + "\n```", "_tests": asserts, "_setup": ""})
    return _dsdict(rows)


# ----------------------------------------------------------------------------- execution metrics
def keep_all(resps, docs):
    """Filter: keep every sampled completion (lm-eval's default filter keeps only the first)."""
    return resps


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _exec_metrics(doc, results, ks, program_fn):
    from evals.sandbox import run_tests
    gens = results[0] if results and isinstance(results[0], list) else list(results)
    timeout = float(os.environ.get("GOGETA_EXEC_TIMEOUT", 10))
    passed = []
    for g in gens:
        r = run_tests(program_fn(doc, g), doc["_tests"], timeout=timeout)
        passed.append(r["total"] > 0 and r["passed"] == r["total"])
    n, c = len(passed), sum(passed)
    out = {f"pass@{k}": pass_at_k(n, c, k) for k in ks if k <= n}
    out["frac_correct"] = c / n if n else 0.0
    return out


def _completion_program(doc, gen):  # native format: the completion IS the solution code
    return (doc["_setup"] + "\n" if doc["_setup"] else "") + gen.split("```")[0]


def _humaneval_program(doc, gen):  # HumanEval: prompt (signature + docstring) + completion
    return doc["_prompt"] + gen.split("```")[0]


def exec_native_pass1(doc, results):
    return _exec_metrics(doc, results, [1], _completion_program)


def exec_native_passk(doc, results):
    return _exec_metrics(doc, results, [1, 5, 10, 20], _completion_program)


def exec_humaneval_pass1(doc, results):
    return _exec_metrics(doc, results, [1], _humaneval_program)


def exec_humaneval_passk(doc, results):
    return _exec_metrics(doc, results, [1, 5, 10, 20], _humaneval_program)


# ----------------------------------------------------------------------------- tier 3: code recall
def load_code_recall(distance: int = 256, n: int = 200, seed: int = 0, **kwargs):
    """Synthetic Python key-value recall (a code-native MQAR / RULER-NIAH analogue).

    A file assigns the needle `k_xxxxx = <4-digit int>` first, then distractor assignments
    (other keys, other values) until `distance` tokens (our tokenizer) separate the needle from
    the query line `assert k_xxxxx == `. The target is the value. Scored by loglikelihood:
    exact greedy match (`acc`) and bpb of the value. The final context must fit max_seq_len.
    """
    from gogeta.hf import load_tokenizer
    tok = load_tokenizer()
    rng = random.Random(f"{seed}-{distance}")
    letters = "abcdefghijklmnopqrstuvwxyz"

    def name():
        return "k_" + "".join(rng.choice(letters) for _ in range(5))

    rows = []
    for _ in range(n):
        key, val = name(), str(rng.randint(1000, 9999))
        lines, used = [f"{key} = {val}"], {key}
        query = f"assert {key} == "
        while len(tok.encode("\n".join(lines[1:]) + "\n" + query)) < distance:
            k = name()
            if k in used:
                continue
            used.add(k)
            lines.append(f"{k} = {rng.randint(1000, 9999)}")
        ctx = "\n".join(lines) + "\n" + query
        rows.append({"_prompt": ctx, "_target": val, "distance": distance,
                     "ctx_tokens": len(tok.encode(ctx))})
    return _dsdict(rows)


def process_recall(doc, results):
    ll, is_greedy = results[0] if isinstance(results[0], (tuple, list)) else results
    return {"acc": float(is_greedy), "bits_per_byte": _bpb_items(ll, doc)["bits_per_byte"]}


# ----------------------------------------------------------------------------- doc accessors
# (functions, not Jinja templates: Jinja drops a trailing newline, and our prompts end in one)
def doc_prompt(doc):
    return doc["_prompt"]


def doc_target(doc):
    return doc["_target"]


def doc_text(doc):
    return doc["text"]


# ----------------------------------------------------------------------------- decontaminated subsets
def _flagged(bench: str) -> set:
    p = pathlib.Path(__file__).with_name("contaminated_ids.json")
    return set(json.loads(p.read_text())[bench])


def process_humaneval_clean(dataset):
    bad = _flagged("humaneval")
    return process_humaneval(dataset.filter(lambda d: d["task_id"] not in bad))


def process_mbpp_clean(dataset):
    bad = _flagged("mbpp")
    return process_mbpp(dataset.filter(lambda d: d["task_id"] not in bad))


# ----------------------------------------------------------------------------- CRUXEval-O (loglikelihood)
def process_cruxeval_o(dataset):
    """CRUXEval output prediction as a loglikelihood probe, in our assert format:
    context = the function + "\nassert f(<input>) == ", target = the true output literal."""
    def f(d):
        return {"_prompt": _lf(d["code"]).rstrip() + f"\nassert f({d['input']}) == ", "_target": d["output"]}
    return dataset.map(f)


def process_cruxeval(doc, results):
    ll, is_greedy = results[0] if isinstance(results[0], (tuple, list)) else results
    return {"acc": float(is_greedy), **_bpb_items(ll, doc)}
