"""Build the Gogeta-0 code dataset (run inside WSL with the project venv).

    ~/.venvs/gogeta/bin/python tools/prepare_data.py            # everything
    ~/.venvs/gogeta/bin/python tools/prepare_data.py --steps tokenize   # just one step

Steps (each skips work already done):
  download  OpenCoder-LLM/opc-annealing-corpus, configs algorithmic_corpus +
            synthetic_code_snippet (ODC-By), parquet from the HF datasets-server
  filter    keep lang == python; drop empty/huge docs; dedupe exact copies;
            shuffle with a fixed seed; hold out validation docs
  tokenizer byte-level BPE, 8192 vocab, trained on training docs only
            -> also copied into the repo (gogeta/assets/) so it is versioned
  tokenize  docs -> uint16 token stream, <|eos|> after every doc
            -> $GOGETA_DATA/datasets/code-py-v1/{train,val}.bin + meta.json

Output layout is described in meta.json, which also records counts and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
DATASET = "OpenCoder-LLM/opc-annealing-corpus"
CONFIGS = ("algorithmic_corpus", "synthetic_code_snippet")
NAME = "code-py-v1"
RAW = DATA / "raw" / "opc-annealing"
OUT = DATA / "datasets" / NAME
VOCAB = 8192
SEED = 1234
VAL_DOCS = 4000
MAX_CHARS = 20_000
SPECIALS = ["<|eos|>", "<|pad|>"]
UA = {"User-Agent": "gogeta-0-data/0.1"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
def step_download() -> None:
    files = _get_json(f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}")["parquet_files"]
    files = [f for f in files if f["config"] in CONFIGS]
    log(f"download: {len(files)} parquet files, {sum(f['size'] for f in files)/1e9:.2f} GB")
    for f in files:
        dest = RAW / f["config"] / f["filename"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(f["url"], headers=UA), timeout=300) as r, open(tmp, "wb") as w:
            shutil.copyfileobj(r, w, length=1 << 20)
        if tmp.stat().st_size != f["size"]:
            raise RuntimeError(f"size mismatch for {dest}")
        tmp.rename(dest)
        log(f"  got {f['config']}/{f['filename']} ({f['size']/1e6:.0f} MB)")


def step_filter() -> None:
    import pyarrow.parquet as pq

    if (OUT / "docs_train.jsonl").exists():
        log("filter: already done")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    seen, docs, stats = set(), [], {"rows": 0, "python": 0, "empty": 0, "too_long": 0, "dupes": 0}
    for cfg in CONFIGS:
        for path in sorted((RAW / cfg).glob("*.parquet")):
            t = pq.read_table(path, columns=["lang", "text"])
            for lang, text in zip(t.column("lang").to_pylist(), t.column("text").to_pylist()):
                stats["rows"] += 1
                if (lang or "").lower() != "python":
                    continue
                stats["python"] += 1
                text = (text or "").strip()
                if not text:
                    stats["empty"] += 1
                    continue
                if len(text) > MAX_CHARS:
                    stats["too_long"] += 1
                    continue
                h = hashlib.sha1(text.encode("utf-8")).digest()
                if h in seen:
                    stats["dupes"] += 1
                    continue
                seen.add(h)
                docs.append({"src": cfg, "text": text})
        log(f"filter: after {cfg}: {len(docs)} docs")
    random.Random(SEED).shuffle(docs)
    val, train = docs[:VAL_DOCS], docs[VAL_DOCS:]
    for name, part in (("val", val), ("train", train)):
        with open(OUT / f"docs_{name}.jsonl", "w", encoding="utf-8") as w:
            for d in part:
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats.update(kept=len(docs), train_docs=len(train), val_docs=len(val),
                 train_chars=sum(len(d["text"]) for d in train), val_chars=sum(len(d["text"]) for d in val))
    (OUT / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"filter: {json.dumps(stats)}")


def _iter_texts(split: str):
    with open(OUT / f"docs_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def step_tokenizer() -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok_path = OUT / "tokenizer.json"
    if tok_path.exists():
        log("tokenizer: already trained")
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS, min_frequency=2,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        t0 = time.time()
        tok.train_from_iterator(_iter_texts("train"), trainer=trainer)
        tok.save(str(tok_path))
        log(f"tokenizer: trained {tok.get_vocab_size()} vocab in {time.time()-t0:.0f}s")
    repo_copy = ROOT / "gogeta" / "assets" / f"tokenizer-{NAME}.json"
    repo_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tok_path, repo_copy)


def step_tokenize() -> None:
    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    eos = tok.token_to_id("<|eos|>")
    meta = {"name": NAME, "source": DATASET, "configs": list(CONFIGS), "license": "ODC-By",
            "vocab_size": tok.get_vocab_size(), "eos_id": eos, "pad_id": tok.token_to_id("<|pad|>"),
            "dtype": "uint16", "seed": SEED, "splits": {}}
    for split in ("val", "train"):
        out = OUT / f"{split}.bin"
        texts = list(_iter_texts(split))
        n_tok, chars, t0 = 0, 0, time.time()
        with open(out, "wb") as w:
            for i in range(0, len(texts), 4096):
                batch = texts[i:i + 4096]
                for enc in tok.encode_batch(batch, add_special_tokens=False):
                    ids = enc.ids + [eos]
                    np.asarray(ids, dtype=np.uint16).tofile(w)
                    n_tok += len(ids)
                chars += sum(len(t) for t in batch)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        meta["splits"][split] = {"docs": len(texts), "tokens": n_tok, "chars": chars,
                                 "chars_per_token": round(chars / n_tok, 3), "sha256_16": sha}
        log(f"tokenize: {split}: {len(texts)} docs -> {n_tok:,} tokens ({chars/n_tok:.2f} chars/tok) "
            f"in {time.time()-t0:.0f}s")
    meta["filter"] = json.loads((OUT / "filter_stats.json").read_text())
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    (ROOT / "gogeta" / "assets" / f"dataset-{NAME}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"done: {OUT}")


STEPS = {"download": step_download, "filter": step_filter, "tokenizer": step_tokenizer, "tokenize": step_tokenize}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="download,filter,tokenizer,tokenize")
    for s in ap.parse_args().steps.split(","):
        log(f"== {s}")
        STEPS[s.strip()]()
