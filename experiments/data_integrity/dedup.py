"""dedup.py: contamination / near-duplicate toolkit (numpy only; CPU; cockpit-ready: every function returns JSON-able dicts).

Three engines, all deterministic:

  1. DOC ENGINE (corpora that fit in memory: val vs test, benchmarks vs each other, planted self-tests).
     MinHash (128 permutations, numpy uint64 universal hashing) over char 5-gram shingles of whitespace-normalised
     text, or over "code-structure" word 5-grams (identifiers/numbers canonicalised, so renamed variables still
     match), + LSH banding (b bands x r rows, picked for a Jaccard threshold) -> candidate pairs -> verified with
     exact Jaccard, containment and the longest common normalised substring.
  2. SUBSTRING ENGINE: longest common substring >= min_len (default 50) characters after whitespace normalisation
     (rolling 64-bit hash of every window, verified by character comparison).
  3. STREAM ENGINE (the 262M-token train.bin, which does not fit a doc index): hash every token n-gram (default 8)
     of a memory-mapped uint16 token stream in chunks, look the hashes up in the (small) query table, and attribute
     matches to documents (doc = run of tokens ending in <|eos|>). Very common n-grams (boilerplate, > max_occ
     occurrences) are dropped as evidence. Candidate (query, train-doc) pairs are then verified with engines 1/2
     on the real text.

CLI:   python dedup.py --selftest | --plan | --compare A.txt B.txt
Functions meant for the cockpit: selftest, compare_texts, find_near_duplicates, scan_token_stream_report.
"""
from __future__ import annotations

import argparse
import builtins
import json
import keyword
import pathlib
import re
import sys
import time

import numpy as np

U64 = np.uint64
BASE = U64(0x9E3779B97F4A7C15)
FIN1 = U64(0xBF58476D1CE4E5B9)
FIN2 = U64(0x94D049BB133111EB)
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|\"\"\"|'''|\S")
_KEEP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


# ------------------------------------------------------------------ hashing helpers
def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised (uint64 wraparound)."""
    x = x.astype(U64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> U64(30)
        x *= FIN1
        x ^= x >> U64(27)
        x *= FIN2
        x ^= x >> U64(31)
    return x


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Hash of every length-n window of an integer array (Horner, uint64 wraparound). Shape (len-n+1,)."""
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


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def char_shingles(text: str, k: int = 5) -> np.ndarray:
    """Unique hashes of the character k-grams of whitespace-normalised text."""
    b = np.frombuffer(normalize_ws(text).encode("utf-8", "ignore"), dtype=np.uint8)
    return np.unique(ngram_hashes(b, k))


_VOCAB: dict[str, int] = {}


def canonical_words(text: str) -> list[int]:
    """Identifier-blind view: identifiers that are not keywords/builtins -> <ID>; numbers, operators and punctuation kept.
    (Docstring words also become <ID>; only structure + literals are compared.)"""
    out = []
    for t in _TOK.findall(text):
        if t[0].isalpha() or t[0] == "_":
            t = t if t in _KEEP else "<ID>"
        # numbers are kept (a renamed copy keeps its constants); only identifiers are blinded
        out.append(_VOCAB.setdefault(t, len(_VOCAB) + 1))
    return out


def struct_shingles(text: str, n: int = 5) -> np.ndarray:
    return np.unique(ngram_hashes(np.asarray(canonical_words(text), dtype=np.int64), n))


def shingles(text: str, mode: str = "char5") -> np.ndarray:
    if mode == "char5":
        return char_shingles(text, 5)
    if mode == "struct5":
        return struct_shingles(text, 5)
    raise ValueError(mode)


# ------------------------------------------------------------------ MinHash + LSH
class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 12345):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = (rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) << U64(1)) | U64(1)  # odd
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)

    def signature(self, sh: np.ndarray, block: int = 4096) -> np.ndarray:
        sig = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=U64)
        with np.errstate(over="ignore"):
            for i in range(0, len(sh), block):
                x = sh[i:i + block, None] * self.a[None, :] + self.b[None, :]
                sig = np.minimum(sig, x.min(axis=0))
        return sig


def est_jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of a's shingles found in b."""
    return len(np.intersect1d(a, b, assume_unique=True)) / len(a) if len(a) else 0.0


def lsh_params(threshold: float, num_perm: int = 128, w_fp: float = 0.5, w_fn: float = 0.5) -> tuple[int, int]:
    """(bands, rows) with bands*rows <= num_perm minimising weighted false-positive + false-negative area
    around `threshold` (datasketch's recipe, numerically)."""
    best, arg = 1e9, (num_perm, 1)
    xs = np.linspace(0, 1, 201)
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b == 0:
            continue
        p = 1 - (1 - xs ** r) ** b
        fp = np.trapezoid(p[xs < threshold], xs[xs < threshold]) if threshold > 0 else 0.0
        fn = np.trapezoid(1 - p[xs >= threshold], xs[xs >= threshold])
        cost = w_fp * fp + w_fn * fn
        if cost < best:
            best, arg = cost, (b, r)
    return arg


class LSHIndex:
    def __init__(self, bands: int, rows: int):
        self.bands, self.rows = bands, rows
        self.tables: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]

    def _keys(self, sig):
        r = self.rows
        return [sig[i * r:(i + 1) * r].tobytes() for i in range(self.bands)]

    def add(self, idx: int, sig: np.ndarray) -> None:
        for t, k in zip(self.tables, self._keys(sig)):
            t.setdefault(k, []).append(idx)

    def candidates(self, sig: np.ndarray) -> set[int]:
        out: set[int] = set()
        for t, k in zip(self.tables, self._keys(sig)):
            out.update(t.get(k, ()))
        return out


# ------------------------------------------------------------------ substring engine
def longest_common_substring(a: str, b: str, min_len: int = 50) -> dict:
    """Longest common substring of whitespace-normalised a, b, found only if >= min_len chars (else length 0).
    Rolling 64-bit hash of all min_len windows; each matching window is extended left/right by comparison."""
    a, b = normalize_ws(a), normalize_ws(b)
    if len(a) < min_len or len(b) < min_len:
        return {"length": 0, "min_len": min_len}
    ba, bb = np.frombuffer(a.encode("utf-8", "ignore"), np.uint8), np.frombuffer(b.encode("utf-8", "ignore"), np.uint8)
    a, b = ba.tobytes().decode("utf-8", "ignore"), bb.tobytes().decode("utf-8", "ignore")  # keep indices byte-aligned
    ha, hb = ngram_hashes(ba, min_len), ngram_hashes(bb, min_len)
    pos_b: dict[int, int] = {}
    for i, h in enumerate(hb.tolist()):
        pos_b.setdefault(h, i)
    best, best_i = 0, -1
    ra, rb = ba.tobytes(), bb.tobytes()
    i = 0
    hal = ha.tolist()
    while i < len(hal):
        j = pos_b.get(hal[i])
        if j is not None and ra[i:i + min_len] == rb[j:j + min_len]:
            l, r = 0, min_len
            while i - l - 1 >= 0 and j - l - 1 >= 0 and ra[i - l - 1] == rb[j - l - 1]:
                l += 1
            while i + r < len(ra) and j + r < len(rb) and ra[i + r] == rb[j + r]:
                r += 1
            if l + r > best:
                best, best_i = l + r, i - l
            i += max(1, r - min_len + 1)  # skip past the run just measured
        else:
            i += 1
    return {"length": best, "start_a": best_i, "min_len": min_len}


# ------------------------------------------------------------------ pairwise report
def compare_texts(a: str, b: str, min_len: int = 50) -> dict:
    """Document-level report for one pair of texts (JSON-able)."""
    out = {}
    for mode in ("char5", "struct5"):
        sa, sb = shingles(a, mode), shingles(b, mode)
        out[mode] = {"jaccard": round(exact_jaccard(sa, sb), 4), "a_in_b": round(containment(sa, sb), 4),
                     "b_in_a": round(containment(sb, sa), 4), "n_a": int(len(sa)), "n_b": int(len(sb))}
    out["substring"] = longest_common_substring(a, b, min_len)
    return out


def find_near_duplicates(docs: dict, threshold: float = 0.7, mode: str = "char5", num_perm: int = 128,
                         min_len: int = 50, ref: dict | None = None, seed: int = 12345,
                         contain: float = 0.8, min_shingles: int = 30) -> dict:
    """All near-duplicate pairs among `docs` (id -> text), or between docs and `ref` (id -> text) when given.
    LSH candidates (tuned for `threshold`) are verified: reported if exact Jaccard >= threshold, OR the smaller doc
    (>= min_shingles shingles) is >= `contain` contained in the other, OR a verbatim normalised substring >= min_len
    covers >= 50% of the shorter doc. (LSH is tuned for Jaccard, so low-Jaccard containment cases can be missed:
    partial copies of a much larger doc are the known recall gap; the stream engine covers them by containment.) Returns {"pairs": [...], "n_docs", "lsh": (b, r)}."""
    mh = MinHasher(num_perm, seed)
    b, r = lsh_params(threshold, num_perm)
    pool = ref if ref is not None else docs
    ids = list(pool)
    sets = {i: shingles(pool[i], mode) for i in ids}
    index = LSHIndex(b, r)
    sigs = {}
    for k, i in enumerate(ids):
        sigs[i] = mh.signature(sets[i])
        index.add(k, sigs[i])
    pairs, seen = [], set()
    for qid, qtext in docs.items():
        qs = sets[qid] if ref is None else shingles(qtext, mode)
        qsig = sigs[qid] if ref is None else mh.signature(qs)
        for k in sorted(index.candidates(qsig)):
            cid = ids[k]
            if ref is None and (cid == qid or (cid, qid) in seen):
                continue
            seen.add((qid, cid))
            ej = exact_jaccard(qs, sets[cid])
            ssub = longest_common_substring(qtext, pool[cid], min_len)["length"]
            shorter = min(len(normalize_ws(qtext)), len(normalize_ws(pool[cid])))
            sc = sets[cid]
            small, big = (qs, sc) if len(qs) <= len(sc) else (sc, qs)
            cont = containment(small, big) if len(small) >= min_shingles else 0.0
            if ej >= threshold or cont >= contain or (ssub >= min_len and ssub >= 0.5 * shorter):
                pairs.append({"a": qid, "b": cid, "jaccard_est": round(est_jaccard(sigs[qid] if ref is None else qsig, sigs[cid]), 3),
                              "jaccard_exact": round(ej, 3), "containment": round(cont, 3), "substring_chars": ssub})
    return {"pairs": pairs, "n_docs": len(docs), "lsh": [b, r], "threshold": threshold, "mode": mode}


# ------------------------------------------------------------------ stream engine
def scan_token_stream(queries: dict, streams: list, n: int = 8, eos: int = 0, max_occ: int = 300,
                      chunk: int = 4_000_000, bloom_bits: int = 27, exclude_docs: dict | None = None,
                      max_tokens: int | None = None, log=None) -> dict:
    """Find which docs of the token streams share n-grams with the queries.

    queries: qid -> np.ndarray of token ids (one query = one benchmark item / candidate doc).
    streams: list of (name, path-to-uint16-.bin) scanned in order; doc numbering restarts at 0 per stream and a doc is
             a run of tokens ended by `eos` (as tools/prepare_data.py wrote them).
    exclude_docs: qid -> {(stream_name, doc_idx)} ignored (a query that is itself a stream doc).
    Returns {"q_total": {qid: #unique n-grams}, "pairs": {qid: {(stream, doc): shared_unique_ngrams}},
             "dropped_boilerplate": int, "tokens_scanned": int}. n-grams seen > max_occ times anywhere in the streams
    are dropped (counted once per occurrence; see report)."""
    qh = []
    q_total = {}
    for qid, ids in queries.items():
        h = np.unique(ngram_hashes(np.asarray(ids), n))
        q_total[qid] = int(len(h))
        qh.append(h)
    qids = list(queries)
    allh = np.concatenate(qh) if qh else np.zeros(0, U64)
    allq = np.concatenate([np.full(len(h), k, dtype=np.int32) for k, h in enumerate(qh)]) if qh else np.zeros(0, np.int32)
    order = np.argsort(allh, kind="stable")
    allh, allq = allh[order], allq[order]
    uh, first = np.unique(allh, return_index=True)          # unique hashes; rows first[i]..first[i+1] hold their queries
    bounds = np.append(first, len(allh))
    alive = np.ones(len(uh), dtype=bool)
    occ = np.zeros(len(uh), dtype=np.int64)
    bloom = np.zeros(1 << bloom_bits, dtype=bool)
    mask = U64((1 << bloom_bits) - 1)
    bloom[(uh & mask).astype(np.int64)] = True
    pairs_chunks = []  # (hash idx, stream id, doc idx) as int64 triplets
    scanned = 0
    for sid, (sname, path) in enumerate(streams):
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        total = len(arr) if max_tokens is None else min(len(arr), max_tokens)
        doc_base = 0
        pos = 0
        while pos < total:
            end = min(total, pos + chunk)
            seg = np.asarray(arr[pos:min(total, end + n - 1)])
            is_eos = seg == eos
            ceos = np.concatenate([[0], np.cumsum(is_eos)])           # eos count before position i
            m = len(seg) - n + 1
            if m > 0:
                m = min(m, end - pos)
                h = ngram_hashes(seg.astype(np.int64), n)[:m]
                # window i spans seg[i:i+n]; eos count inside = ceos[i+n]-ceos[i]
                ok = (ceos[np.arange(m) + n] - ceos[np.arange(m)]) == 0
                cand = np.flatnonzero(ok & bloom[(h & mask).astype(np.int64)])
                if len(cand):
                    idx = np.searchsorted(uh, h[cand])
                    idx = np.minimum(idx, len(uh) - 1)
                    hit = uh[idx] == h[cand]
                    cand, idx = cand[hit], idx[hit]
                    keep = alive[idx]
                    cand, idx = cand[keep], idx[keep]
                    if len(cand):
                        occ += np.bincount(idx, minlength=len(uh))
                        doc = doc_base + ceos[cand]
                        trip = np.unique((idx.astype(np.int64) << 32 | doc.astype(np.int64)))
                        pairs_chunks.append((sid, trip))
                        alive &= occ <= max_occ
            doc_base += int(ceos[min(len(seg), end - pos)])
            scanned += end - pos
            pos = end
            if log:
                log(f"  {sname}: {pos:,}/{total:,} tokens")
    dropped = int((~alive).sum())
    pairs: dict = {q: {} for q in qids}
    for sid, trip in pairs_chunks:
        idx, doc = trip >> 32, trip & 0xFFFFFFFF
        good = alive[idx]
        for hi, d in zip(idx[good].tolist(), doc[good].tolist()):
            for qk in allq[bounds[hi]:bounds[hi + 1]].tolist():
                qid = qids[qk]
                if exclude_docs and (streams[sid][0], d) in exclude_docs.get(qid, ()):
                    continue
                key = (streams[sid][0], d)
                pairs[qid][key] = pairs[qid].get(key, 0) + 1
    return {"q_total": q_total, "pairs": pairs, "dropped_boilerplate": dropped, "unique_query_ngrams": int(len(uh)),
            "tokens_scanned": scanned, "n": n, "max_occ": max_occ}


def scan_token_stream_report(scan: dict, min_shared: int = 1) -> dict:
    """JSON-able digest of scan_token_stream: per query, best doc, shared/total (containment on n-grams)."""
    out = {}
    for qid, pr in scan["pairs"].items():
        tot = scan["q_total"][qid]
        if not pr or not tot:
            out[str(qid)] = {"containment": 0.0, "best": None, "n_candidate_docs": 0}
            continue
        (name, d), sh = max(pr.items(), key=lambda kv: kv[1])
        # union over docs: n-grams of the query covered by ANY doc is not tracked per n-gram; best single doc is reported
        out[str(qid)] = {"containment": round(sh / tot, 4), "best": [name, d], "n_candidate_docs": len(pr)}
    return out



# ------------------------------------------------------------------ whole-corpus engines (token streams)
def _doc_bounds(arr: np.ndarray, eos: int) -> tuple[np.ndarray, np.ndarray]:
    """Doc start offsets and content end offsets (the eos position) for an eos-terminated uint16 stream."""
    e = np.flatnonzero(arr == eos)
    starts = np.concatenate([[0], e[:-1] + 1]) if len(e) else np.zeros(0, np.int64)
    return starts.astype(np.int64), e.astype(np.int64)


def presence_in_streams(query_ids: dict, streams: list, n: int = 13, eos: int = 0, chunk: int = 4_000_000,
                        bloom_bits: int = 27, max_tokens: int | None = None, min_count: int = 1) -> dict:
    """(min_count=2 counts an n-gram as present only if it occurs at least twice in the streams: use it when the query
    docs are themselves inside the streams, so a doc's own copy does not count.)
    Per query: fraction of its unique token n-grams that occur anywhere in the streams (K9's overlap metric, but
    with no boilerplate drop and no per-doc attribution, so duplicated families cost nothing). Returns
    {qid: {"overlap": f, "n_ngrams": k}}."""
    qids = list(query_ids)
    hs = [np.unique(ngram_hashes(np.asarray(query_ids[q], dtype=np.int64), n)) for q in qids]
    uh = np.unique(np.concatenate(hs)) if hs else np.zeros(0, U64)
    counts = np.zeros(len(uh), dtype=np.int64)
    bloom = np.zeros(1 << bloom_bits, dtype=bool)
    mask = U64((1 << bloom_bits) - 1)
    bloom[(uh & mask).astype(np.int64)] = True
    for _name, path in streams:
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        total = len(arr) if max_tokens is None else min(len(arr), max_tokens)
        for pos in range(0, total, chunk):
            end = min(total, pos + chunk)
            seg = np.asarray(arr[pos:min(total, end + n - 1)])
            m = min(len(seg) - n + 1, end - pos)
            if m <= 0:
                continue
            ceos = np.concatenate([[0], np.cumsum(seg == eos)])
            h = ngram_hashes(seg.astype(np.int64), n)[:m]
            ok = (ceos[np.arange(m) + n] - ceos[np.arange(m)]) == 0
            cand = np.flatnonzero(ok & bloom[(h & mask).astype(np.int64)])
            if len(cand):
                hc, cc = np.unique(h[cand], return_counts=True)
                idx = np.minimum(np.searchsorted(uh, hc), len(uh) - 1)
                okk = uh[idx] == hc
                counts[idx[okk]] += cc[okk]
    present = counts >= min_count
    out = {}
    for q, h in zip(qids, hs):
        if len(h) == 0:
            out[q] = {"overlap": None, "n_ngrams": 0}
        else:
            out[q] = {"overlap": float(present[np.searchsorted(uh, h)].mean()), "n_ngrams": int(len(h))}
    return out


def stream_signatures(path: str, num_perm: int = 32, n: int = 8, eos: int = 0, block_docs: int = 20000,
                      seed: int = 99, max_docs: int | None = None, log=None) -> tuple[np.ndarray, np.ndarray]:
    """MinHash signature (num_perm x uint64) of every doc of an eos-terminated token stream, over its token n-grams.
    Returns (sigs [n_docs, num_perm], n_shingles [n_docs]). Docs with no n-gram have sig = uint64 max and count 0."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    starts, ends = _doc_bounds(np.asarray(arr), eos)
    nd = len(starts) if max_docs is None else min(len(starts), max_docs)
    rng = np.random.default_rng(seed)
    a = (rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) << U64(1)) | U64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
    big = U64(np.iinfo(np.uint64).max)
    sigs = np.full((nd, num_perm), big, dtype=U64)
    cnt = np.zeros(nd, dtype=np.int64)
    for d0 in range(0, nd, block_docs):
        d1 = min(nd, d0 + block_docs)
        nb = d1 - d0
        t0, t1 = int(starts[d0]), int(ends[d1 - 1])          # token range of docs d0..d1-1; t1 = the last eos
        seg = np.asarray(arr[t0:t1 + 1]).astype(np.int64)
        m = len(seg) - n + 1
        if m <= 0:
            continue
        h = ngram_hashes(seg, n)
        pos = np.arange(m) + t0
        rel = np.searchsorted(starts[d0:d1], pos, side="right") - 1      # doc (relative) containing each window start
        valid = (pos + n <= ends[d0:d1][rel])                           # the window stays inside the doc's content
        h = np.where(valid, h, big)
        cnt[d0:d1] = np.bincount(rel[valid], minlength=nb)
        first = np.searchsorted(rel, np.arange(nb), side="left")         # first window of each doc (rel is sorted)
        has = cnt[d0:d1] > 0
        first_c = np.minimum(first, m - 1)
        with np.errstate(over="ignore"):
            for p in range(num_perm):
                x = np.where(valid, h * a[p] + b[p], big)
                r = np.minimum.reduceat(x, first_c)
                r[~has] = big
                sigs[d0:d1, p] = r
        if log:
            log(f"  sigs {d1}/{nd} docs")
    return sigs, cnt


def lsh_candidate_pairs(sigs: np.ndarray, bands: int, rows: int, valid: np.ndarray | None = None,
                        max_bucket: int = 60) -> np.ndarray:
    """Candidate doc pairs (int64 [k,2], i<j) sharing at least one LSH band bucket. Buckets larger than max_bucket
    are chained (neighbour pairs) rather than expanded to all pairs, which keeps the output linear (connected
    components still join the whole bucket)."""
    n = len(sigs)
    idx_all = np.arange(n) if valid is None else np.flatnonzero(valid)
    out = []
    for bnd in range(bands):
        cols = sigs[idx_all, bnd * rows:(bnd + 1) * rows]
        key = np.zeros(len(idx_all), dtype=U64)
        with np.errstate(over="ignore"):
            for c in range(rows):
                key = key * BASE + cols[:, c]
        key = mix64(key)
        order = np.argsort(key, kind="stable")
        ks, ids = key[order], idx_all[order]
        brk = np.flatnonzero(ks[1:] != ks[:-1]) + 1
        gs = np.concatenate([[0], brk])
        ge = np.concatenate([brk, [len(ks)]])
        size = ge - gs
        multi = size > 1
        for s, e, z in zip(gs[multi].tolist(), ge[multi].tolist(), size[multi].tolist()):
            g = ids[s:e]
            if z <= max_bucket:
                iu = np.triu_indices(z, 1)
                out.append(np.stack([g[iu[0]], g[iu[1]]], 1))
            else:
                out.append(np.stack([g[:-1], g[1:]], 1))
        if len(out) > 20000:  # compact
            k = np.unique(np.concatenate([np.minimum(o[:, 0], o[:, 1]) * n + np.maximum(o[:, 0], o[:, 1]) for o in out]))
            out = [np.stack([k // n, k % n], 1)]
    if not out:
        return np.zeros((0, 2), np.int64)
    keys = np.unique(np.concatenate([np.minimum(o[:, 0], o[:, 1]) * n + np.maximum(o[:, 0], o[:, 1]) for o in out]))
    return np.stack([keys // n, keys % n], 1)


def lsh_query_against_ref(ref_sigs: np.ndarray, ref_valid: np.ndarray, q_sigs: np.ndarray, bands: int, rows: int,
                          max_per_bucket: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """For each query signature: best estimated Jaccard over LSH-candidate reference docs (vectorised per band).
    Returns (best_est [nq], best_ref_index [nq], -1 when none)."""
    best = np.zeros(len(q_sigs))
    arg = -np.ones(len(q_sigs), dtype=np.int64)
    idx_ref = np.flatnonzero(ref_valid)
    for b in range(bands):
        def key(sg):
            k = np.zeros(len(sg), dtype=U64)
            with np.errstate(over="ignore"):
                for c in range(rows):
                    k = k * BASE + sg[:, b * rows + c]
            return mix64(k)
        kr = key(ref_sigs[idx_ref])
        order = np.argsort(kr, kind="stable")
        kr_s, ids_s = kr[order], idx_ref[order]
        kq = key(q_sigs)
        lo = np.searchsorted(kr_s, kq, side="left")
        hi = np.searchsorted(kr_s, kq, side="right")
        for qi in np.flatnonzero(hi > lo):
            cand = ids_s[lo[qi]:min(hi[qi], lo[qi] + max_per_bucket)]
            est = (ref_sigs[cand] == q_sigs[qi]).mean(1)
            j = int(est.argmax())
            if est[j] > best[qi]:
                best[qi], arg[qi] = est[j], int(cand[j])
    return best, arg


def cluster_families(sigs: np.ndarray, counts: np.ndarray, threshold: float = 0.5, bands: int | None = None,
                     rows: int | None = None) -> dict:
    """Near-duplicate families: connected components of the graph with an edge when the signature-estimated Jaccard of
    two docs is >= threshold (LSH candidates verified on the full signature). Docs without shingles are singletons."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if bands is None or rows is None:
        bands, rows = lsh_params(threshold, sigs.shape[1], w_fp=0.2, w_fn=0.8)  # recall matters more than precision: candidates are re-verified
    cand = lsh_candidate_pairs(sigs, bands, rows, valid=counts > 0)
    if len(cand):
        est = np.concatenate([(sigs[cand[i:i + 2_000_000, 0]] == sigs[cand[i:i + 2_000_000, 1]]).mean(1)
                              for i in range(0, len(cand), 2_000_000)])
        edges = cand[est >= threshold]
    else:
        edges = cand
    n = len(sigs)
    g = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    return {"labels": labels, "n_components": int(ncomp), "n_candidates": int(len(cand)), "n_edges": int(len(edges)),
            "lsh": [bands, rows], "threshold": threshold}


def family_report(labels: np.ndarray, token_counts: np.ndarray, groups: dict | None = None) -> dict:
    """Cluster-size distribution and token share in duplicate families. groups: name -> bool mask over docs
    (e.g. train/val/test) to also report how many families span groups."""
    sizes = np.bincount(labels)
    sz = sizes[labels]
    tok_total = float(token_counts.sum())
    bins = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 10**9]
    hist = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        fam = (sizes >= lo) & (sizes < hi)
        hist[f"{lo}-{hi - 1}" if hi < 10**9 else f">={lo}"] = {"families": int(fam.sum()), "docs": int(sizes[fam].sum())}
    # unique "effective" docs = number of families; keep-one-per-family corpus
    rep = {"n_docs": int(len(labels)), "n_families": int(len(sizes)), "singleton_families": int((sizes == 1).sum()),
           "largest_family": int(sizes.max()), "top10_family_sizes": sorted(sizes.tolist(), reverse=True)[:10],
           "size_histogram": hist,
           "docs_in_duplicate_families": int((sz > 1).sum()), "frac_docs_in_duplicate_families": float((sz > 1).mean()),
           "frac_tokens_in_duplicate_families": float(token_counts[sz > 1].sum() / tok_total),
           "effective_unique_docs": int(len(sizes)),
           "median_family_size_of_a_doc": float(np.median(sz)), "mean_family_size_of_a_doc": float(sz.mean())}
    first = np.zeros(len(sizes), dtype=np.int64) - 1
    order = np.arange(len(labels))
    first[labels[::-1]] = order[::-1]            # first doc index per family
    keep = np.zeros(len(labels), dtype=bool)
    keep[first[first >= 0]] = True
    rep["keep_one_per_family_docs"] = int(keep.sum())
    rep["keep_one_per_family_tokens"] = int(token_counts[keep].sum())
    rep["keep_one_per_family_token_frac"] = float(token_counts[keep].sum() / tok_total)
    if groups:
        span = {}
        for name, msk in groups.items():
            fam_in = np.zeros(len(sizes), dtype=bool)
            fam_in[labels[msk]] = True
            span[name] = fam_in
        names = list(groups)
        rep["families_with_members_in"] = {nm: int(span[nm].sum()) for nm in names}
        rep["families_spanning_more_than_one_group"] = int((sum(span[nm].astype(int) for nm in names) > 1).sum())
    return rep


# ------------------------------------------------------------------ self-test
def _mutate_rename(code: str, rng) -> str:
    names = sorted(set(re.findall(r"\b[a-z_][a-z_0-9]{2,}\b", code)) - _KEEP)
    mp = {n: f"r{rng.integers(100, 999)}_{n[::-1]}" for n in names[: max(1, len(names) // 2)]}
    return re.sub(r"\b[a-z_][a-z_0-9]*\b", lambda m: mp.get(m.group(0), m.group(0)), code)


def _mutate_reformat(code: str, rng) -> str:
    lines = [ln.rstrip() for ln in code.splitlines() if ln.strip()]
    return "\n\n".join(lines).replace("    ", "  ").replace(" = ", "=")


def _synthetic_corpus(rng, n_docs: int = 400) -> dict:
    """Random but code-like documents (vocabulary of statements); distinct docs share boilerplate only."""
    verbs = ["compute", "load", "merge", "filter", "parse", "update", "scan", "build", "count", "split"]
    nouns = ["items", "rows", "graph", "cache", "queue", "total", "index", "bucket", "tokens", "state"]
    docs = {}
    for d in range(n_docs):
        lines = ["import os", "import sys", ""]
        for f in range(rng.integers(3, 7)):
            fn = f"{verbs[rng.integers(10)]}_{nouns[rng.integers(10)]}_{rng.integers(10000)}"
            a, b = nouns[rng.integers(10)] + str(rng.integers(1000)), nouns[rng.integers(10)] + str(rng.integers(1000))
            lines += [f"def {fn}({a}, {b}={rng.integers(100)}):", f'    """{verbs[rng.integers(10)]} the {nouns[rng.integers(10)]} {rng.integers(10**6)}."""']
            for _ in range(rng.integers(4, 10)):
                x = rng.integers(10**5)
                t = rng.integers(4)
                if t == 0:
                    lines.append(f"    {a} = [{b} * {x} + k for k in range({rng.integers(50)}) if k % {rng.integers(2, 9)} == {x % 3}]")
                elif t == 1:
                    lines += [f"    for i in range({x % 90}):", f"        if {a}[i] > {x}:", f"            {b} += {rng.integers(999)}"]
                elif t == 2:
                    lines.append(f"    {b} = {{k: v ** {x % 7} for k, v in {a}.items() if v != {x}}}")
                else:
                    lines += [f"    while {a} < {x}:", f"        {a} = {a} * {rng.integers(2, 9)} - {rng.integers(99)}"]
            lines += [f"    return {a}", ""]
        docs[f"d{d}"] = "\n".join(lines)
    return docs


def selftest(seed: int = 0, verbose: bool = True) -> dict:
    """Planted near-duplicates: measured precision and recall per mutation type, plus MinHash estimation error,
    LSH recall and the substring check. Deterministic. Pass criteria are in the returned dict (`pass`)."""
    rng = np.random.default_rng(seed)
    base = _synthetic_corpus(rng, 300)
    ids = list(base)
    planted, truth = {}, {}
    kinds = {"exact": lambda t: t, "reformat": _mutate_reformat, "rename": _mutate_rename,
             "partial60": lambda t: "\n".join(t.splitlines()[: max(6, int(len(t.splitlines()) * 0.6))]),
             "partial30": lambda t: "\n".join(t.splitlines()[: max(4, int(len(t.splitlines()) * 0.3))]),
             "embedded": lambda t: _synthetic_corpus(rng, 1)["d0"] + "\n" + t}
    for k, fn in kinds.items():
        for j in range(20):
            src = ids[(j * 13 + len(k)) % len(ids)]
            name = f"plant_{k}_{j}"
            planted[name] = fn(base[src], rng) if k in ("rename", "reformat") else fn(base[src])
            truth[name] = src
    # negatives: fresh docs from the same generator (share only boilerplate)
    fresh = _synthetic_corpus(np.random.default_rng(seed + 1), 100)
    for k, t in fresh.items():
        planted[f"neg_{k}"] = t
    res = {"seed": seed, "per_kind": {}}
    for mode in ("char5", "struct5"):
        t0 = time.time()
        out = find_near_duplicates(planted, ref=base, threshold=0.5, mode=mode, min_len=50)
        found = {}
        for p in out["pairs"]:
            found.setdefault(p["a"], set()).add(p["b"])
        kind_res = {}
        for k in kinds:
            names = [f"plant_{k}_{j}" for j in range(20)]
            tp = sum(truth[n] in found.get(n, ()) for n in names)
            kind_res[k] = round(tp / len(names), 3)
        # precision over everything reported: a pair is correct iff it is the planted source
        rep = [(p["a"], p["b"]) for p in out["pairs"]]
        correct = sum(1 for a, b in rep if truth.get(a) == b)
        false_pos_neg = sum(1 for a, b in rep if a.startswith("neg_"))
        # reported pairs for a planted doc that point to a DIFFERENT base doc are false positives
        res["per_kind"][mode] = {"recall": kind_res, "reported_pairs": len(rep), "correct": correct,
                                 "precision": round(correct / len(rep), 3) if rep else 1.0,
                                 "negatives_flagged": false_pos_neg, "negatives": 100, "lsh": out["lsh"],
                                 "seconds": round(time.time() - t0, 1)}
    # MinHash estimation error against exact Jaccard over random doc pairs of mixed similarity
    mh = MinHasher(128, 7)
    errs = []
    names = list(planted)
    for i in range(0, 200, 2):
        a, b = names[i % len(names)], names[(i * 7 + 3) % len(names)]
        for x, y in ((planted[a], planted[a]), (planted[a], base.get(truth.get(a, ""), planted[b]))):
            sa, sb = shingles(x), shingles(y)
            errs.append(abs(est_jaccard(mh.signature(sa), mh.signature(sb)) - exact_jaccard(sa, sb)))
    res["minhash_abs_error"] = {"mean": round(float(np.mean(errs)), 4), "max": round(float(np.max(errs)), 4),
                                "n": len(errs), "expected_sd_at_J0.5": round(0.5 / np.sqrt(128), 4)}
    # substring engine
    s1 = "alpha " * 5 + "def foo(x):\n    return x * 2 + helper(x, 17) - other(x)\n" + "beta " * 3
    s2 = "gamma " + "def foo(x):\n return x * 2 + helper(x, 17) - other(x)\n" + "delta" * 4
    ls = longest_common_substring(s1, s2, 30)["length"]
    res["substring"] = {"found": ls, "expected_at_least": len(normalize_ws("def foo(x): return x * 2 + helper(x, 17) - other(x)")),
                        "none_when_short": longest_common_substring("abc def", "abc def", 50)["length"] == 0}
    # stream engine on a tiny synthetic token stream with a planted copy
    import tempfile
    r2 = np.random.default_rng(seed + 5)
    docs = [r2.integers(2, 500, size=int(r2.integers(40, 120))) for _ in range(300)]
    stream = np.concatenate([np.append(d, 0) for d in docs]).astype(np.uint16)
    tmp = pathlib.Path(tempfile.mkdtemp()) / "s.bin"
    stream.tofile(tmp)
    q = {"copy": docs[17][5:45], "half": np.concatenate([docs[200][:20], r2.integers(2, 500, 20)]),
         "none": r2.integers(2, 500, 40), "cross_eos": np.concatenate([docs[3][-6:], [0], docs[4][:6]])}
    sc = scan_token_stream(q, [("s", str(tmp))], n=8, max_occ=50, chunk=1000, bloom_bits=16)
    rep = scan_token_stream_report(sc)
    res["stream"] = {k: rep[k] for k in q}
    ok_stream = (rep["copy"]["containment"] == 1.0 and rep["copy"]["best"] == ["s", 17]
                 and 0.3 <= rep["half"]["containment"] <= 0.7 and rep["half"]["best"] == ["s", 200]
                 and rep["none"]["containment"] == 0.0 and rep["cross_eos"]["containment"] == 0.0)
    res["stream"]["ok"] = bool(ok_stream)
    # whole-corpus engines on a planted stream: 60 families (sizes 1-4, ~1.2% token mutations (J ~0.8)) among 400 singles
    r3 = np.random.default_rng(seed + 9)
    sd, fam_of = [], []
    for f in range(60):
        base_doc = r3.integers(2, 4000, size=int(r3.integers(80, 200)))
        for c in range(int(r3.integers(1, 5))):
            d = base_doc.copy()
            mut = r3.random(len(d)) < 0.012
            d[mut] = r3.integers(2, 4000, size=int(mut.sum()))
            sd.append(d)
            fam_of.append(f)
    for _ in range(400):
        sd.append(r3.integers(2, 4000, size=int(r3.integers(80, 200))))
        fam_of.append(-1)
    perm = r3.permutation(len(sd))
    sd = [sd[i] for i in perm]
    fam_of = np.array(fam_of)[perm]
    tmp2 = pathlib.Path(tempfile.mkdtemp()) / "f.bin"
    np.concatenate([np.append(d, 0) for d in sd]).astype(np.uint16).tofile(tmp2)
    sg, ct = stream_signatures(str(tmp2), num_perm=64, n=8, block_docs=100)
    cl = cluster_families(sg, ct, threshold=0.5)
    lab = cl["labels"]

    def pairs_of(lbl):
        out = set()
        for c in np.unique(lbl):
            m = np.flatnonzero(lbl == c)
            out |= {(int(i), int(j)) for i in m for j in m if i < j}
        return out
    truth_lbl = np.where(fam_of >= 0, fam_of, 10**6 + np.arange(len(fam_of)))
    same_truth, found = pairs_of(truth_lbl), pairs_of(lab)
    tp = len(same_truth & found)
    res["families"] = {"true_pairs": len(same_truth), "found_pairs": len(found),
                       "recall": round(tp / max(1, len(same_truth)), 3), "precision": round(tp / max(1, len(found)), 3),
                       "lsh": cl["lsh"], "n_edges": cl["n_edges"]}
    pres = presence_in_streams({"in": sd[0][10:60], "out": r3.integers(5000, 6000, 50)}, [("f", str(tmp2))], n=13, bloom_bits=18)
    res["presence"] = {"in": pres["in"]["overlap"], "out": pres["out"]["overlap"]}
    pk = res["per_kind"]
    res["pass"] = bool(
        pk["char5"]["recall"]["exact"] == 1.0 and pk["char5"]["recall"]["reformat"] >= 0.95
        and pk["char5"]["recall"]["partial60"] >= 0.9 and pk["char5"]["recall"]["embedded"] >= 0.9
        and pk["struct5"]["recall"]["rename"] >= 0.9
        and min(pk["char5"]["precision"], pk["struct5"]["precision"]) >= 0.95
        and pk["char5"]["negatives_flagged"] == 0
        and res["minhash_abs_error"]["mean"] <= 0.03 and res["substring"]["found"] >= res["substring"]["expected_at_least"]
        and res["substring"]["none_when_short"] and ok_stream
        and res["families"]["recall"] >= 0.95 and res["families"]["precision"] >= 0.95
        and res["presence"]["in"] == 1.0 and res["presence"]["out"] == 0.0)
    if verbose:
        print(json.dumps(res, indent=1))
    return res


PLAN = {"tool": "dedup.py", "engines": ["minhash+lsh doc engine", "substring engine", "token-stream engine"],
        "selftest": "planted exact/reformat/rename/partial/embedded dups + fresh negatives; MinHash error; stream scan"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.plan:
        print(json.dumps(PLAN, indent=1))
    if a.compare:
        print(json.dumps(compare_texts(pathlib.Path(a.compare[0]).read_text(), pathlib.Path(a.compare[1]).read_text()), indent=1))
    if a.selftest:
        r = selftest()
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
        print("SELFTEST", "PASS" if r["pass"] else "FAIL")
        return 0 if r["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
