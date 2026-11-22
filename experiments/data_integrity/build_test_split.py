"""build_test_split.py: the frozen, untouched final test set for code-py-v1 (data, not a result; no model is run).

FINDING (see notes/data-integrity.md): there are NO unused Python source documents. tools/prepare_data.py kept all
991,154 deduplicated Python docs and split them into train (987,154) and val (4,000); the other ~7.4M raw rows are bash
(and other languages), the 5,889 dropped rows are exact copies of kept docs. A test set that is disjoint from train and
val must therefore come from somewhere else. Two modes:

  external  (default, valid for EVERY existing checkpoint): the UncheatableEval GitHub-Python slice already in the eval
            cache (500 files from 2026-10, ~0.8M tokens), after (a) exact + near-duplicate removal inside the pool,
            (b) removal of any doc with >= OVERLAP_MAX of its 13-grams in train/val, (c) removal of any doc in a MinHash
            family (est. Jaccard >= FAMILY_MIN over token 8-grams) with a train/val doc. Distribution: real GitHub
            code, different from the synthetic/algorithmic training corpus: it measures generalisation, not
            in-distribution fit. It is ~0.8M tokens, not 1M: reaching 1M needs another UncheatableEval month (a
            download, not allowed to this workstream).
  carve     (in-distribution; valid ONLY for models trained after the exclusion list is applied, i.e. a future code-py-v2
            retrain; every checkpoint trained so far has SEEN these docs): seeded sample of train docs that are
            singletons in the family clustering AND have < OVERLAP_MAX of their 13-grams present elsewhere in
            train/val. Writes the test docs plus train_exclude_doc_ids.json.

Outputs go ONLY under $GOGETA_DATA/datasets/ (never the repo); the repo gets a manifest:
  experiments/data_integrity/test-split.manifest.json  (sha256 of every file, docs, tokens, builder git commit, seed)
Lock: files are chmod a-w; access policy and guard in notes/data-integrity.md and test_guard.py.

    python build_test_split.py --plan
    python build_test_split.py --selftest
    python build_test_split.py --mode external            # real data; <= 3 GB RAM, ~5 CPU-min
    python build_test_split.py --mode carve               # needs corpus_families.py --run first
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dedup  # noqa: E402

ROOT = HERE.parents[1]
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DS = DATA / "datasets" / "code-py-v1"
POOL = DATA / "eval-cache" / "uncheatable" / "2026-10" / "github_python.jsonl"
SEED = 20261122
TARGET_TOKENS = 1_000_000
MIN_TOKENS, MAX_CHARS = 48, 20_000
OVERLAP_MAX = 0.30          # drop docs with >= 30% of their 13-grams in train/val (K9 used 50%/80%; stricter on purpose)
FAMILY_MIN = 0.40           # drop docs whose signature-estimated Jaccard with any train/val doc is >= this
POOL_DEDUP_J = 0.7
NAME = {"external": "code-py-v1-test", "carve": "code-py-v1-test-indist"}

PREDICTION = (
    "Written 2026-11-22 before the first real-data build. (1) Of the 500 UncheatableEval files, 90-100% survive near-dup "
    "removal against train/val (fresh Oct-2026 GitHub code vs a synthetic-leaning 2024 corpus), leaving 400-500 docs and "
    "0.5-0.9M tokens. (2) Fewer than 5% of pool docs have >= 30% of their 13-grams in train (K9: 70% of val docs do, "
    "because val is from the corpus itself). (3) Mean 13-gram overlap of pool docs with train is < 0.10.")
CRITERION = (
    "Descriptive; no pass/fail on the data. If (2) is false (>= 5% of fresh GitHub files overlap train at >= 30%), the "
    "pool is itself contaminated by the training corpus: the report says so and the removal rule still applies. The "
    "build is accepted when the manifest's disjointness checks (13-gram overlap < 0.30 for every kept doc, no family "
    "link to train/val at est. Jaccard >= 0.4, no exact copies) are re-verified by --verify.")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "HEAD"), "describe": g("describe", "--tags", "--always", "--dirty"),
            "dirty_paths": [ln for ln in g("status", "--porcelain").splitlines() if "data_integrity" in ln]}


def tokenize_docs(texts: list[str]) -> list[np.ndarray]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DS / "tokenizer.json"))
    eos = tok.token_to_id("<|eos|>")
    assert eos == 0
    return [np.asarray(e.ids, dtype=np.uint16) for e in tok.encode_batch(texts, add_special_tokens=False)]


def load_pool() -> list[dict]:
    docs = []
    with open(POOL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            t = (r.get("content") or r.get("text") or "").strip()
            docs.append({"pool_id": i, "text": t, "date": r.get("date") or r.get("created_at"),
                         "sha1": hashlib.sha1(t.encode()).hexdigest()})
    return docs


def seeded_order(docs: list[dict], seed: int) -> list[dict]:
    return sorted(docs, key=lambda d: hashlib.sha256(f"{seed}|{d['sha1']}".encode()).hexdigest())


def build_external(seed: int = SEED, out_name: str | None = None, write: bool = True) -> dict:
    rep = {"mode": "external", "pool": str(POOL), "seed": seed}
    docs = load_pool()
    rep["pool_docs"] = len(docs)
    # (a) hygiene + exact dup + within-pool near-dup
    d1 = [d for d in docs if d["text"] and len(d["text"]) <= MAX_CHARS]
    rep["dropped_empty_or_long"] = len(docs) - len(d1)
    seen, d2 = set(), []
    for d in seeded_order(d1, seed):
        if d["sha1"] in seen:
            continue
        seen.add(d["sha1"])
        d2.append(d)
    rep["dropped_exact_dup"] = len(d1) - len(d2)
    nd = dedup.find_near_duplicates({d["pool_id"]: d["text"] for d in d2}, threshold=POOL_DEDUP_J, mode="char5")
    drop = set()
    order = {d["pool_id"]: k for k, d in enumerate(d2)}
    for p in nd["pairs"]:
        drop.add(max(p["a"], p["b"], key=lambda i: order[i]))     # keep the earlier in seeded order
    d3 = [d for d in d2 if d["pool_id"] not in drop]
    rep["dropped_pool_near_dup"] = len(drop)
    ids = tokenize_docs([d["text"] for d in d3])
    keep = [k for k, a in enumerate(ids) if len(a) >= MIN_TOKENS]
    rep["dropped_too_short"] = len(d3) - len(keep)
    d3, ids = [d3[k] for k in keep], [ids[k] for k in keep]
    scratch = DATA / "datasets" / (out_name or NAME["external"]) / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cand_stream = scratch / "candidates.bin"
    np.concatenate([np.append(a, 0) for a in ids]).astype(np.uint16).tofile(cand_stream)
    # (b) 13-gram overlap with train+val
    log("presence scan vs train+val ...")
    pres13 = dedup.presence_in_streams({i: a for i, a in enumerate(ids)}, [("train", DS / "train.bin"), ("val", DS / "val.bin")], n=13)
    ov = np.array([pres13[i]["overlap"] if pres13[i]["overlap"] is not None else 0.0 for i in range(len(ids))])
    rep["overlap13_pool"] = {"mean": float(ov.mean()), "median": float(np.median(ov)), "frac_ge_0.30": float((ov >= OVERLAP_MAX).mean()),
                             "frac_ge_0.50": float((ov >= 0.5).mean()), "frac_ge_0.80": float((ov >= 0.8).mean()), "n": len(ov)}
    # (c) MinHash family link with any train/val doc
    fam_link = np.zeros(len(ids), dtype=bool)
    best_est = np.zeros(len(ids))
    sg_c, ct_c = dedup.stream_signatures(str(cand_stream), num_perm=64, n=8)
    cache = sorted((DS / "derived").glob("sig_train_p64_n8.npz")) + sorted((DS / "derived").glob("sig_val_p64_n8.npz"))
    if len(cache) == 2:
        sg_ref = np.concatenate([np.load(c)["sigs"] for c in cache])
        ct_ref = np.concatenate([np.load(c)["cnt"] for c in cache])
        bnd, rw = dedup.lsh_params(FAMILY_MIN, 64, 0.2, 0.8)
        best_est, _ = dedup.lsh_query_against_ref(sg_ref, ct_ref > 0, sg_c, bnd, rw)
        fam_link = best_est >= FAMILY_MIN
        rep["family_link_checked"] = True
    else:
        rep["family_link_checked"] = False
        rep["note"] = "signature caches missing: run corpus_families.py --run first; family check skipped"
    rep["frac_family_link"] = float(fam_link.mean())
    ok = (ov < OVERLAP_MAX) & ~fam_link
    rep["dropped_overlap13"] = int((ov >= OVERLAP_MAX).sum())
    rep["dropped_family_link"] = int((fam_link & (ov < OVERLAP_MAX)).sum())
    sel = [k for k in np.flatnonzero(ok)]
    sel_sorted = sorted(sel, key=lambda k: hashlib.sha256(f"{seed}|{d3[k]['sha1']}".encode()).hexdigest())
    total, final = 0, []
    for k in sel_sorted:
        if total >= TARGET_TOKENS:
            break
        final.append(k)
        total += len(ids[k]) + 1
    rep["candidates_after_filters"] = len(sel)
    rep["kept_docs"], rep["kept_tokens"] = len(final), int(total)
    rep["reached_target_tokens"] = bool(total >= TARGET_TOKENS)
    if write:
        _write_split(NAME["external"] if out_name is None else out_name, [d3[k] for k in final], [ids[k] for k in final], rep, seed,
                     extra={"overlap13": {str(d3[k]["pool_id"]): round(float(ov[k]), 4) for k in final},
                            "best_family_est": {str(d3[k]["pool_id"]): round(float(best_est[k]), 3) for k in final}})
    return rep


def build_carve(seed: int = SEED, out_name: str | None = None, write: bool = True) -> dict:
    """In-distribution test docs from train; valid only for a retrain that applies train_exclude_doc_ids.json."""
    derived = DS / "derived"
    lab = np.load(derived / "family_labels_th0.5.npy")
    gid = np.load(derived / "family_group.npy")
    ntok = np.load(derived / "sig_train_p64_n8.npz")["ntok"]
    tr = np.flatnonzero(gid == 0)
    sizes = np.bincount(lab)
    single = tr[(sizes[lab[tr]] == 1) & (ntok >= MIN_TOKENS)]
    rep = {"mode": "carve", "seed": seed, "train_docs": int(len(tr)), "singleton_family_docs": int(len(single))}
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.permutation(single)[:int(TARGET_TOKENS / 270 * 6)])   # ~6x oversample (only ~19% pass the overlap filter), then filter
    arr = np.memmap(DS / "train.bin", dtype=np.uint16, mode="r")
    st, en = dedup._doc_bounds(np.asarray(arr), 0)
    ids = [np.asarray(arr[st[i]:en[i]]) for i in pick]
    pres = dedup.presence_in_streams({int(i): a for i, a in zip(pick, ids)}, [("train", DS / "train.bin"), ("val", DS / "val.bin")], n=13, min_count=2)
    ov = np.array([pres[int(i)]["overlap"] or 0.0 for i in pick])
    rep["overlap13_elsewhere"] = {"mean": float(ov.mean()), "frac_ge_0.30": float((ov >= OVERLAP_MAX).mean()), "n": len(ov)}
    ok = np.flatnonzero(ov < OVERLAP_MAX)
    total, final = 0, []
    for k in ok:
        if total >= TARGET_TOKENS:
            break
        final.append(int(k))
        total += len(ids[k]) + 1
    rep["kept_docs"], rep["kept_tokens"] = len(final), int(total)
    if write:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(DS / "tokenizer.json"))
        texts = [tok.decode(ids[k].tolist()) for k in final]       # token-faithful text (byte-level BPE round trips)
        docs = [{"pool_id": int(pick[k]), "text": t, "sha1": hashlib.sha1(t.encode()).hexdigest()} for k, t in zip(final, texts)]
        _write_split(out_name or NAME["carve"], docs, [ids[k] for k in final], rep, seed,
                     extra={"train_exclude_doc_ids": [int(pick[k]) for k in final],
                            "WARNING": "These docs are in train.bin. Valid only for models trained with them excluded."})
    return rep


def _write_split(name: str, docs: list[dict], ids: list[np.ndarray], rep: dict, seed: int, extra: dict) -> None:
    out = DATA / "datasets" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file() and f.name in ("docs_test.jsonl", "test.bin", "meta.json"):
            os.chmod(f, 0o644)
    with open(out / "docs_test.jsonl", "w", encoding="utf-8") as w:
        for d in docs:
            w.write(json.dumps({"src": "uncheatable-2026-10" if rep["mode"] == "external" else "train-carve",
                                "pool_id": d["pool_id"], "text": d["text"]}, ensure_ascii=False) + "\n")
    np.concatenate([np.append(a, 0) for a in ids]).astype(np.uint16).tofile(out / "test.bin")
    tok_sha = sha256_file(DS / "tokenizer.json")
    meta = {"name": name, "mode": rep["mode"], "tokenizer": "code-py-v1/tokenizer.json", "tokenizer_sha256": tok_sha,
            "vocab_size": 8192, "eos_id": 0, "dtype": "uint16", "docs": len(docs), "tokens": int(sum(len(a) + 1 for a in ids)),
            "seed": seed, "LOCKED": True, "access": "final pre-registered claims only: see notes/data-integrity.md"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "extra.json").write_text(json.dumps(extra))
    files = {}
    for fn in ("docs_test.jsonl", "test.bin", "meta.json", "extra.json"):
        os.chmod(out / fn, 0o444)
        files[fn] = {"sha256": sha256_file(out / fn), "bytes": (out / fn).stat().st_size}
    mf_path = HERE / "test-split.manifest.json"
    mf = json.loads(mf_path.read_text()) if mf_path.exists() else {"splits": {}}
    mf["splits"][name] = {"created": dt.datetime.now().isoformat(timespec="seconds"), "mode": rep["mode"], "seed": seed,
                          "docs": len(docs), "tokens": meta["tokens"], "tokenizer_sha256": tok_sha, "files": files,
                          "builder": "experiments/data_integrity/build_test_split.py", "builder_git": git_info(),
                          "thresholds": {"overlap13_max": OVERLAP_MAX, "family_min_jaccard": FAMILY_MIN, "min_tokens": MIN_TOKENS},
                          "build_report": {k: v for k, v in rep.items()},
                          "prediction": PREDICTION, "criterion": CRITERION,
                          "location_note": "files live under ~/fusiondance-data/datasets/<name>/ (WSL), not in the repo"}
    mf["note"] = "builder_git.commit is the repo HEAD when the builder ran; dirty_paths lists uncommitted data_integrity files."
    mf_path.write_text(json.dumps(mf, indent=1))
    log(f"wrote {out} ({len(docs)} docs, {meta['tokens']:,} tokens)")


def verify(name: str) -> dict:
    """Re-hash files against the manifest and re-check disjointness from train/val (13-gram overlap per kept doc)."""
    mf = json.loads((HERE / "test-split.manifest.json").read_text())["splits"][name]
    out = DATA / "datasets" / name
    res = {"hash_ok": {fn: sha256_file(out / fn) == v["sha256"] for fn, v in mf["files"].items()}}
    arr = np.memmap(out / "test.bin", dtype=np.uint16, mode="r")
    st, en = dedup._doc_bounds(np.asarray(arr), 0)
    mincount = 2 if mf["mode"] == "carve" else 1
    pres = dedup.presence_in_streams({i: np.asarray(arr[st[i]:en[i]]) for i in range(len(st))},
                                     [("train", DS / "train.bin"), ("val", DS / "val.bin")], n=13, min_count=mincount)
    ov = np.array([pres[i]["overlap"] or 0.0 for i in range(len(st))])
    res["max_overlap13"], res["mean_overlap13"] = float(ov.max()), float(ov.mean())
    res["disjoint_ok"] = bool(ov.max() < OVERLAP_MAX)
    return res


def selftest() -> dict:
    """Determinism and the ordering rule on a tiny synthetic pool; no real data."""
    d = [{"sha1": hashlib.sha1(str(i).encode()).hexdigest(), "pool_id": i} for i in range(50)]
    a = [x["pool_id"] for x in seeded_order(d, 7)]
    b = [x["pool_id"] for x in seeded_order(list(reversed(d)), 7)]
    c = [x["pool_id"] for x in seeded_order(d, 8)]
    return {"deterministic_across_input_order": a == b, "seed_changes_order": a != c, "pass": a == b and a != c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=["external", "carve"])
    ap.add_argument("--verify", metavar="NAME")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.plan:
        print(json.dumps({"prediction": PREDICTION, "criterion": CRITERION, "pool": str(POOL),
                          "writes": [str(DATA / "datasets" / n) for n in NAME.values()] + [str(HERE / "test-split.manifest.json")],
                          "seed": SEED, "target_tokens": TARGET_TOKENS}, indent=1))
    if a.selftest:
        r = selftest()
        print(json.dumps(r))
        return 0 if r["pass"] else 1
    if a.mode == "external":
        print(json.dumps(build_external(a.seed), indent=1))
    if a.mode == "carve":
        print(json.dumps(build_carve(a.seed), indent=1))
    if a.verify:
        print(json.dumps(verify(a.verify), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
