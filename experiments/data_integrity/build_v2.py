"""build_v2.py: deterministic builder for the deduplicated, leakage-checked dataset `code-py-v2` (SQ102).

Definition (notes/data-integrity.md section 3.5; notes/data-v2.md has the full text):
  1. every v1 doc (train + val) is tokenized with the v1 tokenizer and MinHashed (64 perms, token 8-grams);
     near-duplicate FAMILIES = connected components of est. Jaccard >= 0.5 (single linkage, as corpus_families.py);
  2. val candidates = the v1 val docs, then a seeded draw of singleton-family train docs; the whole family of every
     candidate (and every train doc sharing a rare first def/class name with an accepted val doc) leaves train;
  3. train = ONE doc per remaining family (seeded hash rank), in v1 file order; val = candidates whose 13-grams are
     present in train for < 0.30 (K9's metric), at most one per family, up to the v1 val size.
The result is audited at the end (no val doc >= 0.30; no family shared by train and val) or the build fails.

    python experiments/data_integrity/build_v2.py                       # real build, ~$GOGETA_DATA/datasets/code-py-v2
    python experiments/data_integrity/build_v2.py --manifest out.json   # + sha256 manifest (dataset.manifest.json schema)

Inputs (v1 intermediates): $GOGETA_DATA/datasets/code-py-v1/{docs_train.jsonl,docs_val.jsonl}; tokenizer: the committed
gogeta/assets/tokenizer-code-py-v1.json. Output layout = v1 (train.bin, val.bin uint16, <|eos|> after every doc,
docs_val.jsonl, meta.json, tokenizer.json) + train_keep_ids.npy (v1 train line index of each kept doc, in order).
docs_train.jsonl is NOT written. Stdlib + numpy + tokenizers (pyarrow not needed). CPU only.
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # workers are the parallelism; keeps forks warning-free

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DEFAULT_TOKENIZER = ROOT / "gogeta" / "assets" / "tokenizer-code-py-v1.json"

U64 = np.uint64
BASE = U64(0x9E3779B97F4A7C15)
FIN1 = U64(0xBF58476D1CE4E5B9)
FIN2 = U64(0x94D049BB133111EB)
BIG = U64(np.iinfo(np.uint64).max)
DEF = re.compile(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.M)          # as group_proxy.py
GENERIC = {"main", "solution", "Solution", "solve", "test", "run", "__init__", "f", "foo", "func", "function"}
SIG_SEED = 99                      # MinHash coefficients: same as dedup.stream_signatures
BLOCK_DOCS = 4096
BLOCK_TOK = 64 * 1024 * 1024       # manifest block size, as build_dataset_manifest.py
G: dict = {}                       # fork-shared read-only state for workers


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ hashing (same functions as dedup.py)
def mix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(U64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> U64(30)
        x *= FIN1
        x ^= x >> U64(27)
        x *= FIN2
        x ^= x >> U64(31)
    return x


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    ids = np.asarray(ids)
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=U64)
    a = ids.astype(U64) + U64(1)
    h = np.zeros(m, dtype=U64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * BASE + a[j:j + m]
    return mix64(h)


def _coeffs(num_perm: int):
    rng = np.random.default_rng(SIG_SEED)
    a = (rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) << U64(1)) | U64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
    return a, b


def lsh_params(threshold: float, num_perm: int, w_fp: float = 0.2, w_fn: float = 0.8):
    trap = getattr(np, "trapezoid", None) or np.trapz
    best, arg = 1e9, (num_perm, 1)
    xs = np.linspace(0, 1, 201)
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b == 0:
            continue
        p = 1 - (1 - xs ** r) ** b
        fp = trap(p[xs < threshold], xs[xs < threshold])
        fn = trap(1 - p[xs >= threshold], xs[xs >= threshold])
        cost = w_fp * fp + w_fn * fn
        if cost < best:
            best, arg = cost, (b, r)
    return arg


# ------------------------------------------------------------------ parallel map
def pmap(fn, items, workers: int):
    items = list(items)
    if workers > 1 and len(items) > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(min(workers, len(items))) as p:
            return p.map(fn, items, chunksize=1)
    return [fn(i) for i in items]


# ------------------------------------------------------------------ pass A: tokenize + names + MinHash (per block)
def _signatures(seg: np.ndarray, lens: np.ndarray, n: int, num_perm: int):
    nb = len(lens)
    ends = np.cumsum(lens) - 1                 # position of each doc's eos
    starts = ends - lens + 1
    a, b = _coeffs(num_perm)
    sig = np.full((nb, num_perm), BIG, dtype=U64)
    cnt = np.zeros(nb, dtype=np.int64)
    m = len(seg) - n + 1
    if m <= 0:
        return sig.astype(np.uint32), cnt
    h = ngram_hashes(seg, n)
    pos = np.arange(m)
    rel = np.searchsorted(starts, pos, side="right") - 1
    valid = (pos + n) <= ends[rel]             # window stays inside the doc content (never covers its eos)
    cnt = np.bincount(rel[valid], minlength=nb).astype(np.int64)
    first = np.minimum(np.searchsorted(rel, np.arange(nb), side="left"), m - 1)
    has = cnt > 0
    with np.errstate(over="ignore"):
        for p in range(num_perm):
            x = np.where(valid, h * a[p] + b[p], BIG)
            r = np.minimum.reduceat(x, first)
            r[~has] = BIG
            sig[:, p] = r
    return (sig & U64(0xFFFFFFFF)).astype(np.uint32), cnt


def _first_name(t: str):
    m = DEF.search(t)
    return m.group(1) if m else None


def _tok():
    if "tok" not in G:
        from tokenizers import Tokenizer
        G["tok"] = Tokenizer.from_file(G["tokenizer_path"])
    return G["tok"]


def _pass_a(span):
    d0, d1 = span
    tok, eos = _tok(), G["eos"]
    texts = [json.loads(l)["text"] for l in G["lines"][d0:d1]]
    ids, lens = [], np.empty(len(texts), dtype=np.int64)
    for i, e in enumerate(tok.encode_batch(texts, add_special_tokens=False)):
        ids.extend(e.ids)
        ids.append(eos)
        lens[i] = len(e.ids) + 1
    seg = np.asarray(ids, dtype=np.uint16)
    sig, cnt = _signatures(seg.astype(np.int64), lens, G["ngram"], G["num_perm"])
    names = [_first_name(t) for t in texts]
    return seg, lens, sig, cnt, names, np.array([len(t) for t in texts], dtype=np.int64)


# ------------------------------------------------------------------ families (LSH + connected components)
def _band(b: int) -> np.ndarray:
    s32, idx, r, thr_cnt, maxb, N = G["s32"], G["valid_idx"], G["rows"], G["thr_cnt"], G["max_bucket"], G["N"]
    key = np.zeros(len(idx), dtype=U64)
    with np.errstate(over="ignore"):
        for c in range(r):
            key = key * BASE + s32[idx, b * r + c].astype(U64)
    key = mix64(key)
    order = np.argsort(key, kind="stable")
    ks, ids = key[order], idx[order]
    brk = np.flatnonzero(ks[1:] != ks[:-1]) + 1
    gs = np.concatenate([[0], brk])
    ge = np.concatenate([brk, [len(ks)]])
    sz = np.repeat(ge - gs, ge - gs)
    out = []
    for k in range(1, maxb):
        m = ks[:-k] == ks[k:]
        if k > 1:
            m &= sz[:-k] <= maxb               # buckets larger than max_bucket are only chained (k == 1)
        if not m.any():
            break
        u, v = ids[:-k][m], ids[k:][m]
        for i in range(0, len(u), 250_000):
            uu, vv = u[i:i + 250_000], v[i:i + 250_000]
            ok = (s32[uu] == s32[vv]).sum(1) >= thr_cnt
            out.append(np.minimum(uu[ok], vv[ok]).astype(np.int64) * N + np.maximum(uu[ok], vv[ok]).astype(np.int64))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def components(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Connected components by min-label hooking + pointer jumping; label = smallest doc index of the component."""
    lab = np.arange(n, dtype=np.int64)
    if len(u) == 0:
        return lab
    while True:
        lu, lv = lab[u], lab[v]
        if not (lu != lv).any():
            return lab
        m = np.minimum(lu, lv)
        np.minimum.at(lab, lu, m)
        np.minimum.at(lab, lv, m)
        while True:
            nl = lab[lab]
            if (nl == lab).all():
                break
            lab = nl


# ------------------------------------------------------------------ 13-gram presence (K9 metric)
def _presence_block(span):
    d0, d1 = span
    tok, U, n, eos = G["tok_all"], G["U"], G["pn"], G["eos"]
    mask = G["pool"][d0:d1]
    if not mask.any():
        return np.zeros(0, np.int64)
    off = G["off"]
    seg = tok[off[d0]:off[d1]][np.repeat(mask, G["lens"][d0:d1])]
    if len(seg) < n:
        return np.zeros(0, np.int64)
    ceos = np.concatenate([[0], np.cumsum(seg == eos)])
    h = ngram_hashes(seg.astype(np.int64), n)
    m = len(h)
    h = h[(ceos[np.arange(m) + n] - ceos[np.arange(m)]) == 0]
    idx = np.minimum(np.searchsorted(U, h), len(U) - 1)
    return np.unique(idx[U[idx] == h])


def presence_overlap(qdocs: list, pool: np.ndarray, workers: int, pn: int) -> np.ndarray:
    """Per query doc (index into the combined doc space): fraction of its unique pn-grams occurring in any pool doc
    (pool = bool mask over combined doc space). NaN for docs with no pn-gram."""
    off, tok = G["off"], G["tok_all"]
    per = []
    for d in qdocs:
        t = tok[off[d]:off[d + 1] - 1].astype(np.int64)             # content only (no eos)
        per.append(np.unique(ngram_hashes(t, pn)))
    U = np.unique(np.concatenate(per)) if per else np.zeros(0, U64)
    out = np.full(len(qdocs), np.nan)
    if len(U) == 0:
        return out
    G.update(U=U, pool=pool, pn=pn)
    spans = [(d, min(d + BLOCK_DOCS, len(pool))) for d in range(0, len(pool), BLOCK_DOCS)]
    present = np.zeros(len(U), dtype=bool)
    for hit in pmap(_presence_block, spans, workers):
        present[hit] = True
    for i, h in enumerate(per):
        if len(h):
            out[i] = present[np.searchsorted(U, h)].mean()
    return out


# ------------------------------------------------------------------ helpers
def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(p: pathlib.Path) -> list:
    out = []
    with open(p, "rb") as f:
        for l in f:
            if l.strip():
                out.append(l if l.endswith(b"\n") else l + b"\n")
    return out


def bin_blocks(p: pathlib.Path) -> list:
    arr = np.memmap(p, dtype=np.uint16, mode="r")
    return [hashlib.sha256(np.asarray(arr[i:i + BLOCK_TOK]).tobytes()).hexdigest() for i in range(0, len(arr), BLOCK_TOK)]


def rank_in_family(lab: np.ndarray, seed: int) -> np.ndarray:
    """0-based rank of each doc inside its family, ordered by a seeded hash of the doc index (ties impossible)."""
    h = mix64(np.arange(len(lab), dtype=np.int64).astype(U64) ^ mix64(np.array([seed], dtype=np.int64).astype(U64))[0])
    order = np.lexsort((h, lab))
    sl = lab[order]
    first = np.concatenate([[True], sl[1:] != sl[:-1]])
    start = np.maximum.accumulate(np.where(first, np.arange(len(sl)), 0))
    rank = np.empty(len(lab), dtype=np.int64)
    rank[order] = np.arange(len(sl)) - start
    return rank


# ------------------------------------------------------------------ the build
def build(src: pathlib.Path, out: pathlib.Path, tokenizer: pathlib.Path = DEFAULT_TOKENIZER, workers: int | None = None,
          seed: int = 1234, jaccard: float = 0.5, num_perm: int = 64, ngram: int = 8, overlap_n: int = 13,
          max_overlap: float = 0.30, val_docs: int | None = None, val_extra: int = 20000, name_link_max: int = 50,
          big_family_min: int = 300, big_family_cap: int = 1, max_bucket: int = 60, manifest: pathlib.Path | None = None) -> dict:
    t00 = time.time()
    workers = workers or min(64, os.cpu_count() or 1)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(tokenizer))
    eos = tk.token_to_id("<|eos|>")
    pad = tk.token_to_id("<|pad|>")
    vocab = tk.get_vocab_size()
    assert vocab <= 65536
    del tk
    log(f"build v2: src={src} out={out} workers={workers} seed={seed}")

    # ---- read + tokenize + signatures
    t0 = time.time()
    tr_lines, va_lines = read_lines(src / "docs_train.jsonl"), read_lines(src / "docs_val.jsonl")
    Ntr, Nva = len(tr_lines), len(va_lines)
    N = Ntr + Nva
    lines = tr_lines + va_lines
    del tr_lines
    log(f"read {Ntr} train + {Nva} val docs in {time.time()-t0:.0f}s")
    G.update(lines=lines, eos=eos, tokenizer_path=str(tokenizer), ngram=ngram, num_perm=num_perm, N=N)
    t0 = time.time()
    res = pmap(_pass_a, [(d, min(d + BLOCK_DOCS, N)) for d in range(0, N, BLOCK_DOCS)], workers)
    tok_all = np.concatenate([r[0] for r in res])
    lens = np.concatenate([r[1] for r in res])
    s32 = np.concatenate([r[2] for r in res])
    cnt = np.concatenate([r[3] for r in res])
    names = [x for r in res for x in r[4]]
    nchars = np.concatenate([r[5] for r in res])
    del res
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    assert len(tok_all) == off[-1] and len(lens) == N
    log(f"tokenize+minhash: {len(tok_all):,} tokens in {time.time()-t0:.0f}s")
    G.pop("lines", None)

    # ---- families
    t0 = time.time()
    bands, rows = lsh_params(jaccard, num_perm)
    thr_cnt = int(np.ceil(jaccard * num_perm - 1e-9))
    G.update(s32=s32, valid_idx=np.flatnonzero(cnt > 0), rows=rows, thr_cnt=thr_cnt, max_bucket=max_bucket)
    keys = np.unique(np.concatenate(pmap(_band, range(bands), workers)))
    lab = components(N, keys // N, keys % N)
    del keys
    fam_size = np.bincount(lab, minlength=N)
    log(f"families: LSH {bands}x{rows}, {int((fam_size > 0).sum())} families, {int((fam_size[lab] > 1).sum())} docs in "
        f"a family >= 2, largest {int(fam_size.max())} in {time.time()-t0:.0f}s")
    G.pop("s32")

    # ---- rare first-name proxy
    name_cnt = Counter(n for n in names if n and n not in GENERIC)
    rare = sorted(n for n, c in name_cnt.items() if c <= name_link_max) if name_link_max > 0 else []
    rare_id = {n: i for i, n in enumerate(rare)}
    name_id = np.array([rare_id.get(n, -1) if n else -1 for n in names], dtype=np.int64)

    # ---- val candidates: v1 val docs, then seeded singleton-family train docs
    n_val_target = val_docs if val_docs is not None else Nva
    singles = np.flatnonzero((fam_size[lab][:Ntr] == 1) & (cnt[:Ntr] > 0))
    perm = np.random.default_rng(seed).permutation(len(singles))
    extras = singles[perm][:max(0, min(val_extra, len(singles)))]
    cand = np.concatenate([np.arange(Ntr, N), extras]).astype(np.int64)
    fam_c = np.zeros(N, dtype=bool)
    fam_c[lab[cand]] = True

    # ---- train pool: one doc (or capped) per family, families of val candidates excluded
    rank = rank_in_family(lab, seed)
    cap = np.where(fam_size[lab] > big_family_min, max(1, big_family_cap), 1)
    pool0 = (rank < cap) & ~fam_c[lab]
    pool0[Ntr:] = False
    log(f"train pool before val check: {int(pool0.sum())} docs ({len(extras)} extra val candidates, "
        f"{int(fam_c[lab[:Ntr]].sum())} train docs in candidate families)")

    # ---- val selection (pass 1: against pool0)
    t0 = time.time()
    G.update(tok_all=tok_all, off=off, lens=lens)
    ov1 = presence_overlap(cand.tolist(), pool0, workers, overlap_n)
    val_ids, used = [], set()
    for d, o in zip(cand.tolist(), ov1):
        if len(val_ids) >= n_val_target:
            break
        if np.isnan(o) or o >= max_overlap or int(lab[d]) in used:
            continue
        val_ids.append(d)
        used.add(int(lab[d]))
    if len(val_ids) < n_val_target:
        log(f"WARNING: only {len(val_ids)} of {n_val_target} val docs pass the overlap rule (raise --val-extra)")
    n_v1_val_kept = sum(1 for d in val_ids if d >= Ntr)
    log(f"val: {len(val_ids)} docs ({n_v1_val_kept} from v1 val), pass-1 overlap check in {time.time()-t0:.0f}s")

    # ---- name-link exclusion + final train
    vnames = np.unique(name_id[val_ids])
    vnames = vnames[vnames >= 0]
    name_hit = np.isin(name_id, vnames)
    pool = pool0 & ~name_hit
    keep_ids = np.flatnonzero(pool[:Ntr])
    log(f"train: {len(keep_ids)} docs (name-link dropped {int((pool0 & name_hit).sum())})")

    # ---- audit (pass 2): must hold by construction (pool is a subset of pool0)
    ov2 = presence_overlap(val_ids, pool, workers, overlap_n)
    assert (ov2 < max_overlap).all(), "audit failed: val doc with 13-gram overlap >= threshold"
    assert not np.isin(lab[val_ids], lab[keep_ids]).any(), "audit failed: family shared by train and val"
    assert not np.isin(np.array(val_ids), keep_ids).any(), "audit failed: val doc in train"

    # ---- write
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    mask = np.repeat(pool, lens)
    n_tr_tok = int(lens[pool].sum())
    with open(out / "train.bin", "wb") as f:
        tok_all[mask].tofile(f)
    del mask
    va_tok = np.concatenate([tok_all[off[d]:off[d + 1]] for d in val_ids])
    va_tok.tofile(out / "val.bin")
    with open(out / "docs_val.jsonl", "wb") as f:
        for d in val_ids:
            f.write(lines[d])
    np.save(out / "train_keep_ids.npy", keep_ids.astype(np.int64))
    shutil.copyfile(tokenizer, out / "tokenizer.json")
    tr_chars, va_chars = int(nchars[keep_ids].sum()), int(nchars[val_ids].sum())
    n_va_tok = int(len(va_tok))
    v1_meta = {}
    if (src / "meta.json").exists():
        v1_meta = json.loads((src / "meta.json").read_text())
    meta = {"name": "code-py-v2", "source": v1_meta.get("source", "OpenCoder-LLM/opc-annealing-corpus"),
            "configs": v1_meta.get("configs", ["algorithmic_corpus", "synthetic_code_snippet"]),
            "license": v1_meta.get("license", "ODC-By"), "vocab_size": vocab, "eos_id": eos, "pad_id": pad,
            "dtype": "uint16", "seed": seed, "derived_from": "code-py-v1",
            "splits": {"val": {"docs": len(val_ids), "tokens": n_va_tok, "chars": va_chars,
                               "chars_per_token": round(va_chars / n_va_tok, 3), "sha256_16": sha256_file(out / "val.bin")[:16]},
                       "train": {"docs": len(keep_ids), "tokens": n_tr_tok, "chars": tr_chars,
                                 "chars_per_token": round(tr_chars / n_tr_tok, 3), "sha256_16": sha256_file(out / "train.bin")[:16]}},
            "filter": v1_meta.get("filter", {}),
            "v2": {"v1_train_docs": Ntr, "v1_val_docs": Nva, "v1_train_tokens": int(lens[:Ntr].sum()),
                   "jaccard": jaccard, "num_perm": num_perm, "ngram": ngram, "lsh": [bands, rows], "max_bucket": max_bucket,
                   "overlap_n": overlap_n, "max_overlap": max_overlap, "name_link_max": name_link_max,
                   "big_family_min": big_family_min, "big_family_cap": big_family_cap, "val_extra_candidates": int(len(extras)),
                   "families": int((fam_size > 0).sum()), "docs_in_dup_families": int((fam_size[lab] > 1).sum()),
                   "train_dropped_in_val_candidate_families": int(fam_c[lab[:Ntr]].sum()),
                   "train_dropped_name_link": int((pool0 & name_hit).sum()), "val_from_v1_val": n_v1_val_kept,
                   "val_from_train_draw": len(val_ids) - n_v1_val_kept,
                   "val_max_overlap_audited": float(np.nanmax(ov2)) if len(ov2) else 0.0}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "val_ids.json").write_text(json.dumps([int(d) for d in val_ids]))   # combined numbering: >= v1 train docs = v1 val line
    log(f"wrote {out} in {time.time()-t0:.0f}s")

    if manifest:
        man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": "code-py-v2", "source": meta["source"],
               "license": meta["license"],
               "tokenizer": {"file": "code-py-v2/tokenizer.json", "sha256": sha256_file(out / "tokenizer.json"),
                             "vocab_size": vocab, "eos_id": eos},
               "prepare_seed": v1_meta.get("seed"), "filter": meta["filter"], "derived_from": "code-py-v1",
               "build": {"script": "experiments/data_integrity/build_v2.py", "seed": seed, "params": meta["v2"]},
               "splits": {
                   "train": {"docs": len(keep_ids), "tokens": n_tr_tok, "chars": tr_chars, "files": {
                       "train.bin": {"sha256": sha256_file(out / "train.bin"), "bytes": (out / "train.bin").stat().st_size,
                                     "block_tokens": BLOCK_TOK, "block_sha256": bin_blocks(out / "train.bin")},
                       "train_keep_ids.npy": {"sha256": sha256_file(out / "train_keep_ids.npy"),
                                              "bytes": (out / "train_keep_ids.npy").stat().st_size}}},
                   "val": {"docs": len(val_ids), "tokens": n_va_tok, "chars": va_chars, "files": {
                       "docs_val.jsonl": {"sha256": sha256_file(out / "docs_val.jsonl"), "bytes": (out / "docs_val.jsonl").stat().st_size},
                       "val.bin": {"sha256": sha256_file(out / "val.bin"), "bytes": (out / "val.bin").stat().st_size,
                                   "block_tokens": BLOCK_TOK, "block_sha256": bin_blocks(out / "val.bin")}}}},
               "known_issues": ["Template redundancy below family level remains (13-gram overlap inside train is not removed).",
                                "Val is redrawn (v1 val docs + singleton train docs that pass the 13-gram rule); it is not comparable "
                                "with code-py-v1 val numbers."]}
        manifest.write_text(json.dumps(man, indent=1))
        log(f"manifest -> {manifest}")
    log(f"done in {time.time()-t00:.0f}s")
    try:
        import resource
        log(f"peak RSS (parent) {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB")
    except ImportError:
        pass
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=pathlib.Path, default=DATA / "datasets" / "code-py-v1")
    ap.add_argument("--out", type=pathlib.Path, default=DATA / "datasets" / "code-py-v2")
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--max-overlap", type=float, default=0.30)
    ap.add_argument("--val-docs", type=int, default=None, help="target val size (default: v1 val size)")
    ap.add_argument("--val-extra", type=int, default=20000, help="extra val candidates drawn from singleton-family train docs")
    ap.add_argument("--name-link-max", type=int, default=50)
    ap.add_argument("--big-family-cap", type=int, default=1, help="docs kept from families > --big-family-min (notes: 300; default 1)")
    ap.add_argument("--big-family-min", type=int, default=300)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print(f"refusing to write into non-empty {a.out}", file=sys.stderr)
        return 2
    build(a.src, a.out, a.tokenizer, a.workers, a.seed, a.jaccard, max_overlap=a.max_overlap, val_docs=a.val_docs,
          val_extra=a.val_extra, name_link_max=a.name_link_max, big_family_min=a.big_family_min,
          big_family_cap=a.big_family_cap, manifest=a.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
