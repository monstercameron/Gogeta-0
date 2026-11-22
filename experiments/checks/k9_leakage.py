"""K9: train/val near-duplicate leakage check for code-py-v1 (CPU only, numpy; ~1-3 min + optional CPU scoring).

Why: K8 FAILED its plausibility gate (our 14M models score 0.2126-0.2209 bits/char on the val docs, SmolLM2-135M
scores 0.4490: ratio 0.47-0.49 < 0.5) and registered the consequence "check train/val near-duplicate leakage before
trusting any loss number". The val split is exact-dedup only (ledger C107). This check measures how many val docs
are near-copies of train text.

Method (token level, on the real tokenized streams train.bin / val.bin; <|eos|> = id 0 ends every doc):
  1. every 13-gram of content tokens (a window never spans an <|eos|>, so never spans docs) is hashed with a
     64-bit polynomial hash (Horner, odd multiplier, xor-shift-multiply finaliser), vectorised in numpy;
  2. query sets: the 4,000 val docs; a PSEUDO-VAL of 4,000 train docs held out of the reference; and a SHUFFLED
     control (val docs with their tokens permuted inside each doc: same unigrams, no n-gram structure);
  3. the train stream is read in chunks; a bloom-style byte table prefilters, then searchsorted against the sorted
     unique query hashes counts, for every query 13-gram, how often it occurs in the reference. The reference is
     train MINUS the held-out pseudo-val docs (so the pseudo-val is a fair "a val doc as seen from train" baseline);
  4. per doc: overlap = (# 13-grams present in the reference) / (# 13-grams of the doc). Near-dup docs: >= 80% (and
     >= 50%). Docs with < 13 content tokens have no 13-grams and are excluded from all fractions of docs;
  5. boilerplate: the same numbers after excluding the 1,000 query 13-grams with the highest reference count
     (imports, `if __name__ ...`). Approximation: ranked among the 13-grams that occur in the query sets (a train
     13-gram that never occurs in any query doc cannot change any overlap), so this excludes at least the part of
     the global top-1000 that matters;
  6. optional loss effect: if near-dups exist, E0's doc-level bits/char (K8's scorer, CPU, attention-only model)
     on the near-dup docs, and the full-set bits/char with them removed. The full-set number is E0's K8.json value
     (GPU, bf16) unless --rescore-all; the near-dup subset is scored on CPU (fp32).

    ~/.venvs/gogeta/bin/python experiments/checks/k9_leakage.py --selftest
    ~/.venvs/gogeta/bin/python experiments/checks/k9_leakage.py --plan
    ~/.venvs/gogeta/bin/python experiments/checks/k9_leakage.py --max-train-tokens 20000000   # partial (K9_partial.json)
    bash tools/launch_cpu_job.sh --name k9 -- ~/.venvs/gogeta/bin/python experiments/checks/k9_leakage.py

Ledger claims: C94 (our bits/char is plausible, not leakage), C107 (the val split is exact-dedup only).
A partial run (--max-train-tokens) only UNDER-estimates overlap: it is a smoke test, never a verdict.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import RESULTS, ROOT, save  # noqa: E402

import numpy as np  # noqa: E402

DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))  # as gogeta.tracking
DATASET = "code-py-v1"
EOS = 0
W = 13                                   # n-gram length
BASE = np.uint64(0x9E3779B97F4A7C15)     # odd multiplier
FIN = np.uint64(0xD6E8FEB86659FD93)
CHUNK = 4_000_000                        # train tokens per chunk (~200 MB of temporaries)
BLOOM_BITS = 28                          # 256 MB byte table; ~1% false positives for ~3M query hashes
N_PSEUDO = 4000
BP_K = 1000
E0_CKPT = "~/fusiondance-data/checkpoints/E0-20261122-122059-s0/model.pt"
E0_RUN = "E0-20261122-122059-s0"

PREDICTION = (
    "Written 2026-11-22 before any real-data K9 run. (1) Fewer than 1% of val docs are near-duplicates of train "
    "text (>= 80% of their 13-grams occur in train). (2) The val per-doc overlap distribution is within +-5 points "
    "of the pseudo-val baseline (mean per-doc overlap and the share of docs >= 50%); both are well above the "
    "shuffled control (~0) because code has boilerplate. (3) Excluding the near-dup docs changes E0's doc-level "
    "bits/char by less than 2%: the 0.47-0.49 ratio in K8 comes from in-distribution training, not leakage.")
CRITERION = (
    "K9 PASS (no meaningful leakage) if near-dup val docs (>= 80% 13-gram overlap, among val docs with at least "
    "one 13-gram) are < 1% AND, if computed (only when near-dups exist), excluding them changes E0's doc-level "
    "bits/char by < 2%. FAIL otherwise; on FAIL, rebuild the val split with near-dup removal and re-score. The "
    "val-vs-pseudo-val comparison (+-5 points), the shuffled control and the boilerplate-excluded numbers are "
    "reported as context and do not gate the verdict. Maps to ledger C94 (bits/char plausible, not leakage) and "
    "C107 (val split is exact-dedup only). A --max-train-tokens run is PARTIAL and never a verdict.")


# ----------------------------------------------------------------------------- hashing
def _fin(h):
    """Finaliser (in place on a uint64 array): xor-shift, multiply, xor-shift."""
    h ^= h >> np.uint64(32)
    h *= FIN
    h ^= h >> np.uint64(32)
    return h


def ngram_hashes(tokens) -> np.ndarray:
    """64-bit hash of every W-gram window of `tokens` (any int dtype), shape [len-W+1] (empty if len < W)."""
    t = np.asarray(tokens)
    n = len(t) - W + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    t = t.astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        h = t[:n].copy()
        for j in range(1, W):
            h *= BASE
            h += t[j:j + n]
        return _fin(h)


def py_hash(window) -> int:
    """Pure-python reference of ngram_hashes for one window (selftest)."""
    M = (1 << 64) - 1
    h = window[0] + 1
    for x in window[1:]:
        h = (h * int(BASE) + x + 1) & M
    h ^= h >> 32
    h = (h * int(FIN)) & M
    h ^= h >> 32
    return h


def window_valid(t) -> np.ndarray:
    """valid[i] = tokens i..i+W-1 contain no EOS (so the window lies inside one doc)."""
    n = len(t) - W + 1
    if n <= 0:
        return np.zeros(0, dtype=bool)
    cs = np.zeros(len(t) + 1, dtype=np.int32)
    np.cumsum(np.asarray(t) == EOS, out=cs[1:])
    return (cs[W:W + n] - cs[:n]) == 0


def find_docs(stream):
    """Content spans of EOS-terminated docs: (starts, ends), tokens [start, end) are the doc."""
    eos = np.flatnonzero(np.asarray(stream) == EOS).astype(np.int64)
    starts = np.concatenate([[0], eos[:-1] + 1]).astype(np.int64) if len(eos) else np.zeros(0, np.int64)
    return starts, eos


def find_docs_chunked(stream, chunk=1 << 24):
    parts = [np.flatnonzero(np.asarray(stream[i:i + chunk]) == EOS) + i for i in range(0, len(stream), chunk)]
    eos = np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)
    starts = np.concatenate([[0], eos[:-1] + 1]).astype(np.int64) if len(eos) else np.zeros(0, np.int64)
    return starts, eos


# ----------------------------------------------------------------------------- query sets
class QuerySet:
    """All 13-grams of an EOS-terminated token stream, with the doc each belongs to."""

    def __init__(self, stream):
        stream = np.asarray(stream)
        self.n_docs = int((stream == EOS).sum())
        self.h = ngram_hashes(stream)
        self.valid = window_valid(stream)
        is_eos = (stream == EOS).astype(np.int64)
        self.doc = (np.cumsum(is_eos) - is_eos)[:len(self.h)]       # number of EOS strictly before the window

    def hashes(self):
        return self.h[self.valid]


def shuffled_stream(stream, seed):
    """Permute the tokens inside each doc (EOS stay): same unigrams per doc, no 13-gram structure."""
    stream = np.asarray(stream)
    rng = np.random.default_rng(seed)
    out = stream.copy()
    s, e = find_docs(stream)
    for a, b in zip(s, e):
        if b - a > 1:
            out[a:b] = stream[a:b][rng.permutation(b - a)]
    return out


def pseudo_stream(train, starts, ends, idx):
    parts = []
    for i in idx:
        parts.append(np.asarray(train[starts[i]:ends[i] + 1]))      # doc + its EOS
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


# ----------------------------------------------------------------------------- the streaming pass
def count_in_reference(train, n_tok, Q, held_spans, chunk=CHUNK, bloom_bits=BLOOM_BITS, log=None):
    """For each sorted-unique query hash in Q: number of occurrences among 13-grams of train[:n_tok] whose window
    contains no EOS and does not start inside a held-out doc span (start, end) (content ranges, sorted)."""
    shift = np.uint64(64 - bloom_bits)
    table = np.zeros(1 << bloom_bits, dtype=np.uint8)
    table[(Q >> shift).astype(np.intp)] = 1
    cnt = np.zeros(len(Q), dtype=np.int64)
    hs_start = np.asarray([s for s, _ in held_spans], dtype=np.int64)
    hs_end = np.asarray([e for _, e in held_spans], dtype=np.int64)
    t0 = time.time()
    for c0 in range(0, n_tok, chunk):
        c1 = min(c0 + chunk, n_tok)
        e = min(c1 + W - 1, n_tok)
        t = np.asarray(train[c0:e])
        h = ngram_hashes(t)
        nwin = len(h)
        if nwin == 0:
            continue
        valid = window_valid(t)
        i0 = int(np.searchsorted(hs_end, c0, side="right"))          # first span ending after c0
        i1 = int(np.searchsorted(hs_start, c0 + nwin, side="left"))  # spans starting before the chunk's last start
        for k in range(i0, i1):
            valid[max(hs_start[k], c0) - c0:min(hs_end[k], c0 + nwin) - c0] = False
        sel = np.flatnonzero(valid & (table[(h >> shift).astype(np.intp)] == 1))
        if len(sel):
            hsel = h[sel]
            pos = np.searchsorted(Q, hsel)
            pos[pos == len(Q)] = len(Q) - 1
            m = Q[pos] == hsel
            cnt += np.bincount(pos[m], minlength=len(Q))
        if log:
            done = c1 / n_tok
            log(f"  chunk {c0 // chunk + 1}/{-(-n_tok // chunk)}  {done * 100:5.1f}%  "
                f"{time.time() - t0:6.1f}s  survivors {len(sel):,}")
    return cnt


# ----------------------------------------------------------------------------- per-doc statistics
def doc_overlap(qs: QuerySet, Q, cnt, excl_flag=None):
    """Per doc: (# 13-grams, # present in reference). excl_flag: bool over Q of 13-grams to drop (boilerplate)."""
    idx = np.searchsorted(Q, qs.h)
    idx[idx == len(Q)] = len(Q) - 1 if len(Q) else 0
    ok = qs.valid.copy()
    if excl_flag is not None and len(Q):
        ok &= ~excl_flag[idx]
    hit = ok & (cnt[idx] > 0) if len(Q) else ok & False
    n13 = np.bincount(qs.doc[ok], minlength=qs.n_docs)
    hits = np.bincount(qs.doc[hit], minlength=qs.n_docs)
    return n13.astype(np.int64), hits.astype(np.int64)


def summarize(n13, hits) -> dict:
    ev = n13 > 0
    frac = np.where(ev, hits / np.maximum(n13, 1), np.nan)
    f = frac[ev]
    out = {
        "n_docs": int(len(n13)), "n_docs_with_13grams": int(ev.sum()),
        "overall_13gram_overlap": float(hits.sum() / max(1, n13.sum())), "n_13grams": int(n13.sum()),
        "per_doc_mean": float(f.mean()) if len(f) else None,
        "per_doc_median": float(np.median(f)) if len(f) else None,
        "per_doc_p90": float(np.percentile(f, 90)) if len(f) else None,
        "per_doc_p99": float(np.percentile(f, 99)) if len(f) else None,
        "docs_ge80": int((f >= 0.8).sum()), "docs_ge50": int((f >= 0.5).sum()), "docs_ge95": int((f >= 0.95).sum()),
        "docs_eq0": int((f == 0).sum()),
        "frac_docs_ge80": float((f >= 0.8).mean()) if len(f) else None,
        "frac_docs_ge50": float((f >= 0.5).mean()) if len(f) else None,
        "hist_10bins": np.histogram(f, bins=np.linspace(0, 1, 11))[0].tolist() if len(f) else [],
    }
    return out


def leakage(train, n_tok, val_stream, heldout_idx=None, n_pseudo=N_PSEUDO, seed=0, chunk=CHUNK,
            bloom_bits=BLOOM_BITS, bp_k=BP_K, log=None):
    """The whole measurement on in-memory/memmapped streams. Returns (results dict, per-doc arrays dict)."""
    starts, ends = (find_docs_chunked(train[:n_tok]) if n_tok > (1 << 24) else find_docs(train[:n_tok]))
    n_tok = int(ends[-1]) + 1 if len(ends) else 0                   # drop a trailing partial doc
    if heldout_idx is None:
        rng = np.random.default_rng(seed)
        heldout_idx = np.sort(rng.choice(len(starts), size=min(n_pseudo, len(starts)), replace=False))
    heldout_idx = np.asarray(sorted(heldout_idx), dtype=np.int64)
    spans = [(int(starts[i]), int(ends[i])) for i in heldout_idx]
    if log:
        log(f"train: {n_tok:,} tokens, {len(starts):,} docs; held out {len(heldout_idx)} docs as pseudo-val")

    val_stream = np.asarray(val_stream)
    sets = {
        "val": QuerySet(val_stream),
        "pseudo_val": QuerySet(pseudo_stream(train, starts, ends, heldout_idx)),
        "shuffled_val": QuerySet(shuffled_stream(val_stream, seed + 1)),
    }
    Q = np.unique(np.concatenate([s.hashes() for s in sets.values()]))
    if log:
        log(f"query 13-grams: {sum(int(s.valid.sum()) for s in sets.values()):,} total, {len(Q):,} unique")
    cnt = count_in_reference(train, n_tok, Q, spans, chunk=chunk, bloom_bits=bloom_bits, log=log)

    bp = np.zeros(len(Q), dtype=bool)
    k = min(bp_k, len(Q))
    if k:
        bp[np.argpartition(-cnt, k - 1)[:k]] = True
    bp_counts = np.sort(cnt[bp])[::-1]
    res = {"reference_tokens": n_tok - sum(e - s for s, e in spans), "n_train_docs": int(len(starts)),
           "heldout_docs": int(len(heldout_idx)), "unique_query_13grams": int(len(Q)),
           "boilerplate_excluded": int(k), "boilerplate_min_reference_count": int(bp_counts[-1]) if k else 0,
           "boilerplate_max_reference_count": int(bp_counts[0]) if k else 0}
    arrays = {}
    for name, qs in sets.items():
        n13, hits = doc_overlap(qs, Q, cnt)
        n13x, hitsx = doc_overlap(qs, Q, cnt, excl_flag=bp)
        res[name] = summarize(n13, hits)
        res[name + "_excl_boilerplate"] = summarize(n13x, hitsx)
        arrays[name] = (n13, hits)
        arrays[name + "_excl"] = (n13x, hitsx)
    return res, arrays


# ----------------------------------------------------------------------------- reporting helpers
def snippet(text: str, n=120) -> str:
    return text[:n].replace("\r", "").replace("\n", "\\n").encode("ascii", "replace").decode()


def top_docs(arrays_val, docs, k=20):
    n13, hits = arrays_val
    frac = np.where(n13 > 0, hits / np.maximum(n13, 1), -1.0)
    order = np.lexsort((-n13, -frac))[:k]
    return [{"doc": int(i), "overlap": round(float(frac[i]), 4), "n_13grams": int(n13[i]),
             "snippet": snippet(docs[i]) if docs is not None else None} for i in order if frac[i] >= 0]


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss     # KB on Linux
    return r / 1e6


# ----------------------------------------------------------------------------- loss effect (optional)
def loss_effect(docs, near80, band50, ckpt, rescore_all=False, log=print):
    """E0 doc-level bits/char with near-dup docs removed (K8's scorer, CPU)."""
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from k8_external_reference import score_checkpoint

    ck = os.path.expanduser(ckpt)
    if not os.path.exists(ck):
        return {"skipped": f"checkpoint missing: {ck}"}
    chars_all = sum(len(t) for t in docs)
    full_src = None
    if not rescore_all:
        k8 = RESULTS / "K8.json"
        if k8.exists():
            for r in json.loads(k8.read_text())["data"]["ours"]:
                if r["run"] == E0_RUN:
                    full = (r["bits_per_char"] * r["chars"], r["chars"])
                    full_src = "K8.json (GPU bf16)"
    if full_src is None:
        r = score_checkpoint(ck, docs, "cpu", 1024)
        full, full_src = (r["bits_per_char"] * r["chars"], r["chars"]), "CPU fp32 rescore of all docs"
    assert full[1] == chars_all, (full[1], chars_all)

    def sub(idx, label):
        if not len(idx):
            return (0.0, 0)
        t0 = time.time()
        r = score_checkpoint(ck, [docs[i] for i in idx], "cpu", 1024)
        log(f"  CPU-scored {label}: {len(idx)} docs, {r['bits_per_char']:.4f} bits/char ({time.time() - t0:.0f}s)")
        return (r["bits_per_char"] * r["chars"], r["chars"])

    out = {"checkpoint": ckpt, "full_set_source": full_src, "full_bits_per_char": full[0] / full[1]}
    b80 = sub(near80, ">=80% near-dups")
    out["near80_docs"] = len(near80)
    out["near80_bits_per_char"] = b80[0] / b80[1] if b80[1] else None
    out["bpc_without_near80"] = (full[0] - b80[0]) / (full[1] - b80[1])
    out["change_without_near80_pct"] = (out["bpc_without_near80"] / out["full_bits_per_char"] - 1) * 100
    if len(band50) <= 1000:
        bb = sub(band50, "50-80% band")
        out["band50_docs"] = len(band50)
        out["bpc_without_near50"] = (full[0] - b80[0] - bb[0]) / (full[1] - b80[1] - bb[1])
        out["change_without_near50_pct"] = (out["bpc_without_near50"] / out["full_bits_per_char"] - 1) * 100
    else:
        out["band50_skipped"] = f"{len(band50)} docs in the 50-80% band (> 1000), not scored"
    return out


# ----------------------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(0)
    V = 8192
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")
        ok &= bool(cond)

    # 1. hash: vectorised == pure python; equal windows equal, different windows different
    s = rng.integers(0, V, 5000)
    s[::97] = EOS
    h = ngram_hashes(s)
    idx = rng.integers(0, len(h), 300)
    check("vectorised hash == pure-python reference (300 windows)",
          all(int(h[i]) == py_hash([int(x) for x in s[i:i + W]]) for i in idx))
    s2 = rng.integers(2, V, 100_000)
    h2 = ngram_hashes(s2)
    check("no hash collisions among 100k distinct windows", len(set(h2.tolist())) == len(h2))
    rep = np.concatenate([s2[:50], s2[:50]])
    hr = ngram_hashes(rep)
    check("identical windows hash identically", hr[0] == hr[50] and hr[1] == hr[51])
    check("window_valid excludes windows containing EOS",
          not window_valid(np.array([5] * 20 + [EOS] + [6] * 20))[8:21].any()
          and window_valid(np.array([5] * 20 + [EOS] + [6] * 20))[:8].all())

    # 2. exact per-doc overlap, held-out masking, boilerplate, chunk invariance
    def rnd(n):
        return rng.integers(2, V, n)

    boiler = rnd(20)
    tdocs = []
    for i in range(300):
        d = rnd(int(rng.integers(100, 160)))
        if i < 60:
            d[:20] = boiler                                        # boilerplate in 60 train docs
        tdocs.append(d)
    tdocs[250] = tdocs[201].copy()                                 # a copy of held-out doc 201 elsewhere in train
    train = np.concatenate([np.concatenate([d, [EOS]]) for d in tdocs]).astype(np.uint16)
    vdocs = [
        tdocs[150].copy(),                                         # 0: exact copy               -> 1.0
        np.concatenate([tdocs[160][:60], rnd(40)]),                # 1: 60 copied of 100          -> 48/88
        rnd(80),                                                   # 2: random                    -> 0
        tdocs[170][:10].copy(),                                    # 3: 10 tokens: no 13-grams    -> n/a
        np.concatenate([boiler, rnd(60)]),                         # 4: boilerplate only          -> 8/68, excl 0/60
        np.concatenate([tdocs[180][:85], rnd(15)]),                # 5: 85 copied of 100          -> 73/88 (near-dup)
    ]
    val = np.concatenate([np.concatenate([d, [EOS]]) for d in vdocs]).astype(np.uint16)
    held = list(range(200, 220))                                   # random docs: own spans must NOT count (doc 201 matches 250)
    exp = {0: (len(tdocs[150]) - 12, len(tdocs[150]) - 12), 1: (88, 48), 2: (68, 0), 3: (0, 0), 4: (68, 8), 5: (88, 73)}
    exp[2] = (80 - 12, 0)
    runs = []
    for chunk in (CHUNK, 64, 1000):
        res, arr = leakage(train, len(train), val, heldout_idx=held, chunk=chunk, bloom_bits=20, bp_k=8)
        runs.append((res, arr))
    res, arr = runs[0]
    n13, hits = arr["val"]
    check("val per-doc 13-gram counts exact", all(n13[i] == exp[i][0] for i in exp), f"{n13.tolist()}")
    check("val per-doc hits exact", all(hits[i] == exp[i][1] for i in exp), f"{hits.tolist()}")
    s_ = res["val"]
    check("doc with < 13 tokens excluded from doc fractions", s_["n_docs_with_13grams"] == 5)
    check("near-dup counts: >= 80% is docs 0 and 5 (2); >= 50% adds doc 1 at 0.545 (3)",
          s_["docs_ge80"] == 2 and s_["docs_ge50"] == 3, f"ge80={s_['docs_ge80']} ge50={s_['docs_ge50']}")
    n13x, hitsx = arr["val_excl"]
    check("boilerplate exclusion: doc 4 -> 0/60, copies untouched",
          n13x[4] == 60 and hitsx[4] == 0 and hitsx[0] == hits[0] and hitsx[5] == hits[5], f"{n13x[4]}/{hitsx[4]}")
    pn, ph = arr["pseudo_val"]
    check("pseudo-val: held-out docs do not match themselves; only doc 201 (copy at 250) overlaps",
          ph[1] == pn[1] and ph.sum() == pn[1], f"hits {ph.tolist()}")
    sn, sh = arr["shuffled_val"]
    check("shuffled control ~ 0 overlap", sh.sum() == 0, f"hits {sh.sum()}")
    same = all(np.array_equal(runs[0][1][k][j], runs[r][1][k][j]) for r in (1, 2) for k in runs[0][1] for j in (0, 1))
    check("results identical for chunk sizes 4M / 64 / 1000 (windows straddling chunks)", same)
    top = top_docs(arr["val"], [f"doc {i}" for i in range(6)], k=3)
    check("top_docs order (docs 0, 5, 1)", [t["doc"] for t in top] == [0, 5, 1], f"{[t['doc'] for t in top]}")
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------- plan
def plan(args):
    d = DATA / "datasets" / DATASET
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else None
    print(f"dataset: {d}")
    for f in ("train.bin", "val.bin", "docs_val.jsonl", "meta.json"):
        p = d / f
        print(f"  {f:15s} {p.stat().st_size / 1e6:10.1f} MB" if p.exists() else f"  {f:15s} MISSING")
    n_train = (d / "train.bin").stat().st_size // 2 if (d / "train.bin").exists() else 0
    if args.max_train_tokens:
        n_train = min(n_train, args.max_train_tokens)
    n_val = (d / "val.bin").stat().st_size // 2 if (d / "val.bin").exists() else 0
    print(f"tokens: train {n_train:,} (uint16, vocab 8192, eos 0), val {n_val:,}; docs val "
          f"{meta['splits']['val']['docs'] if meta else '?'}, train {meta['splits']['train']['docs'] if meta else '?'}")
    # micro-benchmark the hot loop on synthetic tokens (no real data touched)
    rng = np.random.default_rng(0)
    Q = np.unique(ngram_hashes(rng.integers(2, 8192, 3_300_000).astype(np.uint16)))
    shift = np.uint64(64 - BLOOM_BITS)
    table = np.zeros(1 << BLOOM_BITS, np.uint8)
    table[(Q >> shift).astype(np.intp)] = 1
    t = rng.integers(2, 8192, 4_000_000).astype(np.uint16)
    t0 = time.time()
    h = ngram_hashes(t)
    valid = window_valid(t)
    sel = np.flatnonzero(valid & (table[(h >> shift).astype(np.intp)] == 1))
    np.searchsorted(Q, h[sel])
    per_m = (time.time() - t0) / 4.0
    est = per_m * n_train / 1e6
    print(f"hot loop measured on 4M synthetic tokens: {per_m:.2f} s per M tokens "
          f"(+ ~15 s pre-pass for EOS positions and disk reads)")
    print(f"estimated runtime, counting pass: ~{est / 60:.1f} min (x1.5 safety: ~{est * 1.5 / 60:.1f} min); "
          f"plus ~1 min to build query sets/analysis")
    print(f"estimated memory: byte table {(1 << BLOOM_BITS) / 1e6:.0f} MB + chunk temporaries "
          f"~{CHUNK * 8 * 6 / 1e6:.0f} MB + query arrays ~150 MB = ~{((1 << BLOOM_BITS) + CHUNK * 48 + 150e6) / 1e9:.1f} GB "
          f"(limit 6 GB; train.bin is memory-mapped, page cache not counted); measured peak RSS in this plan: {rss_gb():.2f} GB")
    print("optional loss effect: E0 on CPU, 2 threads, only the near-dup docs (and the 50-80% band if <= 1000 docs); "
          "~1k tokens/s => ~20 min per 1M tokens scored; --no-loss skips it. Full-set number comes from K8.json "
          "(use --rescore-all to rescore all 4000 docs on CPU: ~1 h)")
    print(f"checkpoint: {E0_CKPT} {'(present)' if os.path.exists(os.path.expanduser(E0_CKPT)) else '(MISSING)'}")
    print(f"result: {RESULTS / ('K9_partial.json' if args.max_train_tokens else 'K9.json')}")
    print(f"PREDICTION: {PREDICTION}\nCRITERION: {CRITERION}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--max-train-tokens", type=int, default=None, help="quick partial run (writes K9_partial.json)")
    ap.add_argument("--no-loss", action="store_true", help="skip the optional E0 CPU scoring")
    ap.add_argument("--rescore-all", action="store_true", help="rescore all val docs on CPU instead of K8.json")
    ap.add_argument("--checkpoint", default=E0_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.plan:
        return plan(args)

    t0 = time.time()
    d = DATA / "datasets" / DATASET
    train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    val = np.fromfile(d / "val.bin", dtype=np.uint16)
    n_tok = len(train) if not args.max_train_tokens else min(len(train), args.max_train_tokens)
    partial = n_tok < len(train)
    log = lambda s: print(s, flush=True)  # noqa: E731
    log(f"{DATASET}: train {len(train):,} tokens (using {n_tok:,}), val {len(val):,} tokens")
    res, arrays = leakage(train, n_tok, val, seed=args.seed, log=log)

    from k8_external_reference import load_docs        # K8's loader: the identical 4,000 val docs
    docs = load_docs(None)
    vs, ve = find_docs(val)
    assert len(docs) == len(vs) == res["val"]["n_docs"], (len(docs), len(vs))
    n13, hits = arrays["val"]
    frac = np.where(n13 > 0, hits / np.maximum(n13, 1), np.nan)
    res["top20_val_docs"] = top_docs(arrays["val"], docs, 20)
    res["val_per_doc_overlap"] = [None if np.isnan(x) else round(float(x), 4) for x in frac]
    near80 = np.flatnonzero(frac >= 0.8).tolist()
    band50 = np.flatnonzero((frac >= 0.5) & (frac < 0.8)).tolist()
    res["near_dup_docs_ge80"] = near80
    res["near_dup_docs_50_80"] = band50

    for name in ("val", "pseudo_val", "shuffled_val", "val_excl_boilerplate", "pseudo_val_excl_boilerplate",
                 "shuffled_val_excl_boilerplate"):
        s = res[name]
        print(f"{name:30s} overlap {s['overall_13gram_overlap'] * 100:6.2f}%  per-doc mean {s['per_doc_mean']:.3f} "
              f"median {s['per_doc_median']:.3f}  docs>=80%: {s['docs_ge80']} ({s['frac_docs_ge80'] * 100:.2f}%)  "
              f">=50%: {s['docs_ge50']} ({s['frac_docs_ge50'] * 100:.2f}%)")
    for t in res["top20_val_docs"]:
        print(f"  doc {t['doc']:4d} overlap {t['overlap']:.3f} ({t['n_13grams']} 13-grams)  {t['snippet']}")

    v, p = res["val"], res["pseudo_val"]
    dmean = abs(v["per_doc_mean"] - p["per_doc_mean"])
    d50 = abs(v["frac_docs_ge50"] - p["frac_docs_ge50"])
    verdicts = {}
    nd_frac = v["frac_docs_ge80"]
    a_ok = nd_frac < 0.01
    verdicts["K9a_near_dup_docs"] = (f"{'PASS' if a_ok else 'FAIL'}: {v['docs_ge80']}/{v['n_docs_with_13grams']} val docs "
                                     f"({nd_frac * 100:.2f}%) have >= 80% of 13-grams in train (need < 1%); "
                                     f">=50%: {v['docs_ge50']} ({v['frac_docs_ge50'] * 100:.2f}%)")
    verdicts["K9_context_vs_pseudo_val"] = (
        f"{'within' if dmean <= 0.05 and d50 <= 0.05 else 'OUTSIDE'} +-5 points of pseudo-val: per-doc mean "
        f"val {v['per_doc_mean']:.3f} vs pseudo {p['per_doc_mean']:.3f}; docs>=50% val "
        f"{v['frac_docs_ge50'] * 100:.2f}% vs pseudo {p['frac_docs_ge50'] * 100:.2f}%; shuffled control overlap "
        f"{res['shuffled_val']['overall_13gram_overlap'] * 100:.2f}% (context only, not gating)")
    c_ok = True
    if near80 and not args.no_loss and not partial:
        res["loss_effect"] = loss_effect(docs, near80, band50, args.checkpoint, args.rescore_all, log=log)
        le = res["loss_effect"]
        if "change_without_near80_pct" in le:
            c_ok = abs(le["change_without_near80_pct"]) < 2.0
            verdicts["K9b_loss_effect"] = (f"{'PASS' if c_ok else 'FAIL'}: E0 bits/char {le['full_bits_per_char']:.4f} -> "
                                           f"{le['bpc_without_near80']:.4f} without the {le['near80_docs']} near-dups "
                                           f"({le['change_without_near80_pct']:+.2f}%; need < 2%)")
        else:
            verdicts["K9b_loss_effect"] = f"NOT COMPUTED: {le.get('skipped')}"
    elif not near80:
        verdicts["K9b_loss_effect"] = "PASS (vacuous): no val doc has >= 80% overlap, nothing to exclude"
    else:
        verdicts["K9b_loss_effect"] = "NOT COMPUTED (--no-loss or partial run)"
        c_ok = None
    if partial:
        verdicts["K9"] = f"PARTIAL ({n_tok:,} of {len(train):,} train tokens): smoke test, no verdict"
    else:
        overall = a_ok and c_ok is not False
        verdicts["K9"] = ("PASS" if overall and c_ok is not None else "FAIL" if not overall else "INCOMPLETE") + \
            (": no meaningful leakage" if overall and c_ok is not None else "; rebuild the val split with near-dup removal and re-score" if not overall else "")
    res["peak_rss_gb"] = round(rss_gb(), 2)
    res["notes"] = ("overlap = fraction of a doc's 13-grams (content tokens, no EOS inside) occurring in the "
                    "reference = train minus the 4,000 held-out pseudo-val docs; boilerplate = 1,000 query 13-grams "
                    "with the highest reference count; near-dup fractions are over docs with >= 1 13-gram.")
    save("K9", ["C94", "C107"], PREDICTION, CRITERION, verdicts, res, t0,
         filename="K9_partial" if partial else "K9")


if __name__ == "__main__":
    main()
