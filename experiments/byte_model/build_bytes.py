"""Build the byte-level twin of code-py-v1: `code-py-v1-bytes` (CPU, streaming, < 1 GB RAM; run inside WSL).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python -m experiments.byte_model.build_bytes
    ... --verify-only          # re-run the checks on an existing build
    ... --tiny 300 --out DIR   # first 300 docs of each split (smoke tests); writes no repo manifest

Same documents, same order, same train/val split as code-py-v1: it reads code-py-v1's own `docs_{train,val}.jsonl`
(the exact texts that were BPE-tokenized; sha256-checked against experiments/data_integrity/dataset.manifest.json),
so doc i of val.bin is line i of docs_val.jsonl, and the K9 clean-val id lists (line indices) carry over unchanged.
The frozen test splits (code-py-v1-test*) are NOT read here (lock policy, notes/data-integrity.md section 4): their
byte form is the same pure function (utf-8 + EOS) of docs_test.jsonl, applied at the guarded read.

Encoding: each doc's text -> UTF-8 bytes, then one EOS byte. **EOS = 0x00** (byte value 0), the doc separator with the
same semantics as code-py-v1's <|eos|> (appended after every doc; training windows are packed random windows that
cross it). 0x00 never occurs inside a doc (checked: the build fails otherwise), so the stream decodes losslessly and
the vocab is exactly 256 (a 256-way head; no extra special ids; no PAD is ever written). Stored as uint8.

Outputs under $GOGETA_DATA/datasets/code-py-v1-bytes/: train.bin, val.bin (uint8), {split}_doc_offsets.npy (int64,
n_docs + 1 boundaries; doc i = bin[off[i]:off[i+1]-1], the last byte is EOS), meta.json.
Checks: (1) decoding val.bin reproduces docs_val.jsonl exactly; (2) per-doc byte counts match utf-8 lengths, for both
splits (train is re-streamed and compared doc by doc); (3) the BPE val.bin decodes, doc by doc, to the same texts
(so the clean-val ids index the same docs in both datasets); BPE train.bin has the same doc count; (4) the clean-val
manifest's val_docs_sha256 matches docs_val.jsonl and its ids are valid; (5) no 0x00 inside any doc.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
SRC = DATA / "datasets" / "code-py-v1"
NAME = "code-py-v1-bytes"
EOS = 0
MANIFEST = ROOT / "experiments" / "byte_model" / "dataset-bytes.manifest.json"
DS_MANIFEST = ROOT / "experiments" / "data_integrity" / "dataset.manifest.json"
CLEAN_MANIFEST = ROOT / "experiments" / "data_integrity" / "clean-val.manifest.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: pathlib.Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def iter_docs(path: pathlib.Path, limit: int | None = None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)["text"]


def build_split(src: pathlib.Path, out: pathlib.Path, split: str, limit: int | None) -> dict:
    t0 = time.time()
    offsets, n_chars, n_nonascii_docs, nul_docs = [0], 0, 0, []
    with open(out / f"{split}.bin", "wb") as w:
        pos = 0
        for i, text in enumerate(iter_docs(src / f"docs_{split}.jsonl", limit)):
            b = text.encode("utf-8")
            if b"\x00" in b:
                nul_docs.append(i)
            n_chars += len(text)
            n_nonascii_docs += len(b) != len(text)
            w.write(b)
            w.write(b"\x00")
            pos += len(b) + 1
            offsets.append(pos)
    if nul_docs:
        raise RuntimeError(f"{split}: {len(nul_docs)} docs contain 0x00 (first ids {nul_docs[:5]}); EOS=0x00 would "
                           "be ambiguous: switch to uint16 with EOS=256")
    off = np.asarray(offsets, dtype=np.int64)
    np.save(out / f"{split}_doc_offsets.npy", off)
    n_docs = len(off) - 1
    positions = int(off[-1])
    content = positions - n_docs
    rec = {"docs": n_docs, "bytes_content": content, "positions": positions, "chars": n_chars,
           "bytes_per_char": round(content / n_chars, 6), "chars_per_token": round(n_chars / positions, 6),
           "nonascii_docs": n_nonascii_docs, "doc_bytes_p50": int(np.median(np.diff(off) - 1)),
           "doc_bytes_max": int((np.diff(off) - 1).max()), "build_s": round(time.time() - t0, 1)}
    log(f"{split}: {n_docs:,} docs, {content:,} content bytes + {n_docs:,} EOS = {positions:,} positions "
        f"({rec['bytes_per_char']} bytes/char) in {rec['build_s']} s")
    return rec


def verify(src: pathlib.Path, out: pathlib.Path, limit: int | None) -> dict:
    res = {}
    t0 = time.time()
    # (1) + (2) val: exact decode round trip, per-doc byte counts
    val = np.fromfile(out / "val.bin", dtype=np.uint8)
    off = np.load(out / "val_doc_offsets.npy")
    assert len(val) == off[-1], "val.bin length != last offset"
    eos_pos = np.flatnonzero(val == EOS)
    assert np.array_equal(eos_pos + 1, off[1:]), "val: EOS positions != doc boundaries (0x00 inside a doc?)"
    texts = list(iter_docs(src / "docs_val.jsonl", limit))
    raw = val.tobytes()
    dec = [raw[off[i]:off[i + 1] - 1].decode("utf-8") for i in range(len(off) - 1)]
    res["val_docs"] = len(dec)
    res["val_roundtrip_exact"] = dec == texts
    res["val_bytecounts_match"] = all(off[i + 1] - off[i] - 1 == len(t.encode("utf-8")) for i, t in enumerate(texts))
    # (2) train: stream docs again and compare every doc's bytes with its slice
    tr = np.memmap(out / "train.bin", dtype=np.uint8, mode="r")
    toff = np.load(out / "train_doc_offsets.npy")
    assert len(tr) == toff[-1], "train.bin length != last offset"
    bad, n = 0, 0
    for i, text in enumerate(iter_docs(src / "docs_train.jsonl", limit)):
        b = text.encode("utf-8")
        s, e = int(toff[i]), int(toff[i + 1])
        if e - s - 1 != len(b) or tr[e - 1] != EOS or bytes(tr[s:e - 1]) != b:
            bad += 1
        n += 1
    res["train_docs"] = n
    res["train_roundtrip_mismatches"] = bad
    n_eos = sum(int(np.count_nonzero(tr[i:i + (1 << 26)] == EOS)) for i in range(0, len(tr), 1 << 26))
    res["train_eos_count_matches_docs"] = n_eos == n
    # (3) BPE twin: same docs in the same order
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(src / "tokenizer.json"))
        bpe_eos = json.loads((src / "meta.json").read_text())["eos_id"]
        bv = np.fromfile(src / "val.bin", dtype=np.uint16)
        cuts = np.flatnonzero(bv == bpe_eos)
        starts = np.concatenate([[0], cuts[:-1] + 1])
        k = len(dec) if limit is None else min(limit, len(cuts))
        bpe_docs = tok.decode_batch([bv[s:c].tolist() for s, c in zip(starts[:k], cuts[:k])], skip_special_tokens=False)
        res["bpe_val_docs"] = int(len(cuts))
        res["bpe_val_decodes_to_same_docs"] = bpe_docs == dec[:k]
        if bpe_docs != dec[:k]:
            res["bpe_val_first_mismatch"] = next(i for i, (a, b) in enumerate(zip(bpe_docs, dec)) if a != b)
        bt = np.memmap(src / "train.bin", dtype=np.uint16, mode="r")
        cnt = sum(int(np.count_nonzero(bt[i:i + (1 << 26)] == bpe_eos)) for i in range(0, len(bt), 1 << 26))
        res["bpe_train_docs"] = cnt
        res["bpe_train_doc_count_matches"] = cnt == n if limit is None else None
    except ImportError:
        res["bpe_check"] = "skipped: tokenizers not installed"
    # (4) clean-val ids
    if limit is None and CLEAN_MANIFEST.exists():
        cm = json.loads(CLEAN_MANIFEST.read_text())
        res["clean_val_manifest_docs_sha_matches"] = cm["val_docs_sha256"] == sha256_file(src / "docs_val.jsonl")
        res["clean_val_n_val_docs_matches"] = cm["n_val_docs"] == len(dec)
        lens = np.diff(off) - 1
        res["clean_val_lists"] = {}
        for name, lst in cm["lists"].items():
            ids = lst["ids"]
            ok = all(0 <= i < len(dec) for i in ids) and len(set(ids)) == len(ids) == lst["n"]
            res["clean_val_lists"][name] = {"n": len(ids), "ids_valid_unique": ok,
                                            "bytes_content": int(lens[ids].sum()),
                                            "chars": int(sum(len(texts[i]) for i in ids))}
    res["verify_s"] = round(time.time() - t0, 1)
    flags = [v for k, v in res.items() if isinstance(v, bool)]
    flags += [v["ids_valid_unique"] for v in res.get("clean_val_lists", {}).values()]
    res["all_pass"] = all(flags) and res["train_roundtrip_mismatches"] == 0
    return res


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "describe": g("describe", "--tags", "--always")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(DATA / "datasets" / NAME))
    ap.add_argument("--tiny", type=int, default=None, help="first N docs per split (smoke tests; no manifest)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src_sha = {}
    if a.tiny is None:
        dm = json.loads(DS_MANIFEST.read_text())
        for split in ("train", "val"):
            src_sha[split] = sha256_file(src / f"docs_{split}.jsonl")
            want = dm["splits"][split]["files"][f"docs_{split}.jsonl"]["sha256"]
            assert src_sha[split] == want, f"docs_{split}.jsonl sha256 {src_sha[split]} != dataset manifest {want}"
        log("source docs_*.jsonl match experiments/data_integrity/dataset.manifest.json")
    if a.verify_only:
        meta = json.loads((out / "meta.json").read_text())
    else:
        meta = {"name": NAME if a.tiny is None else f"{NAME}-tiny{a.tiny}", "source": "code-py-v1 docs_{train,val}.jsonl",
                "license": "ODC-By", "vocab_size": 256, "eos_id": EOS, "pad_id": None, "dtype": "uint8",
                "encoding": "utf-8 bytes of each doc text, then EOS = 0x00 (never inside a doc); ids = byte values",
                "unit": "bytes", "splits": {}}
        for split in ("val", "train"):
            meta["splits"][split] = build_split(src, out, split, a.tiny)
        for split in ("val", "train"):
            meta["splits"][split]["sha256"] = sha256_file(out / f"{split}.bin")
            meta["splits"][split]["offsets_sha256"] = sha256_file(out / f"{split}_doc_offsets.npy")
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    checks = verify(src, out, a.tiny)
    log("verify: " + json.dumps(checks if a.tiny else {k: v for k, v in checks.items() if k != "clean_val_lists"}))
    if a.tiny is None:
        man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": NAME,
               "builder": "experiments/byte_model/build_bytes.py", "builder_git": git_info(),
               "location": f"$GOGETA_DATA/datasets/{NAME}/ (WSL filesystem; data are never committed)",
               "source": {"dataset": "code-py-v1", "docs_sha256": src_sha,
                          "pointer": "experiments/data_integrity/dataset.manifest.json"},
               "encoding": meta["encoding"], "vocab_size": 256, "eos_id": EOS, "dtype": "uint8",
               "splits": {s: {**meta["splits"][s], "files": {
                   f"{s}.bin": {"sha256": meta["splits"][s]["sha256"], "bytes": (out / f"{s}.bin").stat().st_size},
                   f"{s}_doc_offsets.npy": {"sha256": meta["splits"][s]["offsets_sha256"],
                                            "bytes": (out / f"{s}_doc_offsets.npy").stat().st_size}}}
                          for s in ("train", "val")},
               "test_splits": "not built (locked; byte form = utf-8 + EOS of docs_test.jsonl at the guarded read)",
               "verification": checks, "wall_s": round(time.time() - t0, 1)}
        MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
        log(f"manifest -> {MANIFEST.relative_to(ROOT)}")
    log(f"{'PASS' if checks['all_pass'] else 'FAIL'} in {time.time() - t0:.0f} s")
    sys.exit(0 if checks["all_pass"] else 1)


if __name__ == "__main__":
    main()
