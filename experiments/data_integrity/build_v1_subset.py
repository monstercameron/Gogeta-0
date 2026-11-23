"""build_v1_subset.py: `code-py-v1sub`, the dedup-only control for code-py-v2 (SQ238).

v2 differs from v1 in two ways at once: near-duplicates are removed AND coverage changes (v2 train = 698,714 docs /
202.5M tokens vs v1 987,154 / 262.4M; a 200M-token run sees ~99% of v2 but ~76% of v1). To attribute a v1-vs-v2 gap to
dedup, v1sub is a uniform random subset of v1 train with v2's doc count and NO dedup.

  1. Exclusion (so v1sub is no more contaminated against v1 val than v2 is): the SAME families as build_v2 are recomputed
     (same code, same params, train+val together); every train doc whose family contains a v1 val doc, and every train doc
     sharing a rare first def/class name (<= --name-link-max docs, not generic) with any v1 val doc, is excluded.
  2. Selection: a seeded uniform draw (numpy default_rng(--seed), 1234) of exactly --count (default: v2 train docs, read
     from dataset-v2.manifest.json) docs from the remaining train docs, kept in v1 file order.
  3. Output = v1 layout: train.bin (uint16, <|eos|> after every doc, same encode path as tools/prepare_data.py), val.bin and
     docs_val.jsonl = v1's (byte-identical), meta.json, tokenizer.json, plus train_keep_ids.npy (v1 train line index).

    python experiments/data_integrity/build_v1_subset.py --manifest v1sub.manifest.json
    # options: --src --out --tokenizer --workers --seed --count --v2-manifest --name-link-max

Needs $GOGETA_DATA/datasets/code-py-v1/{docs_train.jsonl,docs_val.jsonl} (cloud boxes: build v1 with KEEP=1). Refuses a
non-empty --out. CPU only; deterministic for any worker count. See notes/data-v2.md, section "code-py-v1sub".
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_v2 as b2  # noqa: E402

DEFAULT_V2_MANIFEST = HERE / "dataset-v2.manifest.json"
NAME = "code-py-v1sub"
log = b2.log


def v2_train_docs(manifest: pathlib.Path = DEFAULT_V2_MANIFEST) -> int:
    return int(json.loads(manifest.read_text())["splits"]["train"]["docs"])


def build(src: pathlib.Path, out: pathlib.Path, tokenizer: pathlib.Path = b2.DEFAULT_TOKENIZER, workers: int | None = None,
          seed: int = 1234, count: int | None = None, v2_manifest: pathlib.Path = DEFAULT_V2_MANIFEST,
          jaccard: float = 0.5, num_perm: int = 64, ngram: int = 8, name_link_max: int = 50, max_bucket: int = 60,
          manifest: pathlib.Path | None = None) -> dict:
    t00 = time.time()
    workers = workers or min(64, os.cpu_count() or 1)
    count = count if count is not None else v2_train_docs(v2_manifest)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(tokenizer))
    eos, pad, vocab = tk.token_to_id("<|eos|>"), tk.token_to_id("<|pad|>"), tk.get_vocab_size()
    assert vocab <= 65536
    del tk
    log(f"build v1sub: src={src} out={out} count={count} workers={workers} seed={seed}")

    t0 = time.time()
    tr_lines, va_lines = b2.read_lines(src / "docs_train.jsonl"), b2.read_lines(src / "docs_val.jsonl")
    Ntr, Nva = len(tr_lines), len(va_lines)
    N = Ntr + Nva
    lines = tr_lines + va_lines
    del tr_lines
    log(f"read {Ntr} train + {Nva} val docs in {time.time()-t0:.0f}s")
    b2.G.update(lines=lines, eos=eos, tokenizer_path=str(tokenizer), ngram=ngram, num_perm=num_perm, N=N)
    t0 = time.time()
    res = b2.pmap(b2._pass_a, [(d, min(d + b2.BLOCK_DOCS, N)) for d in range(0, N, b2.BLOCK_DOCS)], workers)
    tok_all = np.concatenate([r[0] for r in res])
    lens = np.concatenate([r[1] for r in res])
    s32 = np.concatenate([r[2] for r in res])
    cnt = np.concatenate([r[3] for r in res])
    names = [x for r in res for x in r[4]]
    nchars = np.concatenate([r[5] for r in res])
    del res
    b2.G.pop("lines", None)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    assert len(tok_all) == off[-1] and len(lens) == N
    log(f"tokenize+minhash: {len(tok_all):,} tokens in {time.time()-t0:.0f}s")

    # ---- families: identical procedure and parameters to build_v2 (so identical families)
    t0 = time.time()
    bands, rows = b2.lsh_params(jaccard, num_perm)
    thr_cnt = int(np.ceil(jaccard * num_perm - 1e-9))
    b2.G.update(s32=s32, valid_idx=np.flatnonzero(cnt > 0), rows=rows, thr_cnt=thr_cnt, max_bucket=max_bucket)
    keys = np.unique(np.concatenate(b2.pmap(b2._band, range(bands), workers)))
    lab = b2.components(N, keys // N, keys % N)
    del keys
    b2.G.pop("s32")
    del s32
    fam_size = np.bincount(lab, minlength=N)
    log(f"families: {int((fam_size > 0).sum())}, largest {int(fam_size.max())} in {time.time()-t0:.0f}s")

    # ---- exclusion: train docs tied to v1 val (family, then one-hop rare first-name)
    fam_val = np.zeros(N, dtype=bool)
    fam_val[lab[Ntr:]] = True
    in_val_family = fam_val[lab[:Ntr]]
    name_cnt = Counter(n for n in names if n and n not in b2.GENERIC)
    rare = sorted(n for n, c in name_cnt.items() if c <= name_link_max) if name_link_max > 0 else []
    rare_id = {n: i for i, n in enumerate(rare)}
    name_id = np.array([rare_id.get(n, -1) if n else -1 for n in names], dtype=np.int64)
    vnames = np.unique(name_id[Ntr:])
    vnames = vnames[vnames >= 0]
    name_hit = np.isin(name_id[:Ntr], vnames)
    excluded = in_val_family | name_hit
    eligible = np.flatnonzero(~excluded)
    log(f"excluded {int(excluded.sum())} train docs ({int(in_val_family.sum())} in v1-val families, "
        f"{int((name_hit & ~in_val_family).sum())} more by rare-name link); eligible {len(eligible)}")
    if len(eligible) < count:
        raise SystemExit(f"only {len(eligible)} eligible docs, need {count}")

    # ---- selection
    pick = np.random.default_rng(seed).choice(len(eligible), size=count, replace=False)
    keep_ids = np.sort(eligible[pick]).astype(np.int64)
    assert len(keep_ids) == count and len(np.unique(keep_ids)) == count and keep_ids[-1] < Ntr
    pool = np.zeros(N, dtype=bool)
    pool[keep_ids] = True

    # ---- write
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.bin", "wb") as f:
        tok_all[np.repeat(pool, lens)].tofile(f)
    n_tr_tok = int(lens[keep_ids].sum())
    va_tok = tok_all[off[Ntr]:off[N]]
    va_tok.tofile(out / "val.bin")
    with open(out / "docs_val.jsonl", "wb") as f:
        for l in lines[Ntr:]:
            f.write(l)
    if (src / "val.bin").exists():
        assert (src / "val.bin").read_bytes() == (out / "val.bin").read_bytes(), "val.bin differs from v1's"
    if (src / "train.bin").exists():           # same tokenisation path as prepare_data.py: v1 stream must match
        v1t = np.memmap(src / "train.bin", dtype=np.uint16, mode="r")
        k = min(1 << 24, int(off[Ntr]))
        assert len(v1t) == int(off[Ntr]) and np.array_equal(v1t[:k], tok_all[:k]) \
            and np.array_equal(v1t[-k:], tok_all[int(off[Ntr]) - k:int(off[Ntr])]), "tokenisation differs from v1 train.bin"
    np.save(out / "train_keep_ids.npy", keep_ids)
    shutil.copyfile(tokenizer, out / "tokenizer.json")
    tr_chars, va_chars, n_va_tok = int(nchars[keep_ids].sum()), int(nchars[Ntr:].sum()), int(len(va_tok))
    v1_meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    meta = {"name": NAME, "source": v1_meta.get("source", "OpenCoder-LLM/opc-annealing-corpus"),
            "configs": v1_meta.get("configs", ["algorithmic_corpus", "synthetic_code_snippet"]),
            "license": v1_meta.get("license", "ODC-By"), "vocab_size": vocab, "eos_id": eos, "pad_id": pad,
            "dtype": "uint16", "seed": seed, "derived_from": "code-py-v1",
            "splits": {"val": {"docs": Nva, "tokens": n_va_tok, "chars": va_chars,
                               "chars_per_token": round(va_chars / n_va_tok, 3), "sha256_16": b2.sha256_file(out / "val.bin")[:16]},
                       "train": {"docs": count, "tokens": n_tr_tok, "chars": tr_chars,
                                 "chars_per_token": round(tr_chars / n_tr_tok, 3), "sha256_16": b2.sha256_file(out / "train.bin")[:16]}},
            "filter": v1_meta.get("filter", {}),
            "v1sub": {"v1_train_docs": Ntr, "v1_val_docs": Nva, "count": count, "selection_seed": seed,
                      "families_jaccard": jaccard, "num_perm": num_perm, "ngram": ngram, "max_bucket": max_bucket,
                      "name_link_max": name_link_max, "excluded_in_v1_val_families": int(in_val_family.sum()),
                      "excluded_name_link_only": int((name_hit & ~in_val_family).sum()),
                      "excluded_total": int(excluded.sum()), "eligible": int(len(eligible)), "dedup": False}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {out} in {time.time()-t0:.0f}s")

    if manifest:
        sf = b2.sha256_file
        man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": NAME, "source": meta["source"],
               "license": meta["license"],
               "tokenizer": {"file": f"{NAME}/tokenizer.json", "sha256": sf(out / "tokenizer.json"),
                             "vocab_size": vocab, "eos_id": eos},
               "prepare_seed": v1_meta.get("seed"), "filter": meta["filter"], "derived_from": "code-py-v1",
               "build": {"script": "experiments/data_integrity/build_v1_subset.py", "seed": seed, "params": meta["v1sub"]},
               "splits": {
                   "train": {"docs": count, "tokens": n_tr_tok, "chars": tr_chars, "files": {
                       "train.bin": {"sha256": sf(out / "train.bin"), "bytes": (out / "train.bin").stat().st_size,
                                     "block_tokens": b2.BLOCK_TOK, "block_sha256": b2.bin_blocks(out / "train.bin")},
                       "train_keep_ids.npy": {"sha256": sf(out / "train_keep_ids.npy"),
                                              "bytes": (out / "train_keep_ids.npy").stat().st_size}}},
                   "val": {"docs": Nva, "tokens": n_va_tok, "chars": va_chars, "files": {
                       "docs_val.jsonl": {"sha256": sf(out / "docs_val.jsonl"), "bytes": (out / "docs_val.jsonl").stat().st_size},
                       "val.bin": {"sha256": sf(out / "val.bin"), "bytes": (out / "val.bin").stat().st_size,
                                   "block_tokens": b2.BLOCK_TOK, "block_sha256": b2.bin_blocks(out / "val.bin")}}}},
               "known_issues": ["NOT deduplicated: a uniform random subset of v1 train (near-duplicate families kept), the "
                                "dedup-only control for code-py-v2 (same doc count). Val is v1's val unchanged, so v1 val "
                                "numbers apply; docs sharing a family or rare first name with a v1 val doc are excluded."]}
        manifest.write_text(json.dumps(man, indent=1))
        log(f"manifest -> {manifest}")
    log(f"done in {time.time()-t00:.0f}s")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=pathlib.Path, default=b2.DATA / "datasets" / "code-py-v1")
    ap.add_argument("--out", type=pathlib.Path, default=b2.DATA / "datasets" / NAME)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=b2.DEFAULT_TOKENIZER)
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--v2-manifest", type=pathlib.Path, default=DEFAULT_V2_MANIFEST)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--count", type=int, default=None, help="train docs (default: v2 train docs from the v2 manifest)")
    ap.add_argument("--name-link-max", type=int, default=50)
    a = ap.parse_args()
    if not (a.src / "docs_train.jsonl").exists():
        print(f"missing {a.src / 'docs_train.jsonl'} (build v1 with KEEP=1)", file=sys.stderr)
        return 2
    if a.out.exists() and any(a.out.iterdir()):
        print(f"refusing to write into non-empty {a.out}", file=sys.stderr)
        return 2
    build(a.src, a.out, a.tokenizer, a.workers, a.seed, a.count, a.v2_manifest, name_link_max=a.name_link_max,
          manifest=a.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
