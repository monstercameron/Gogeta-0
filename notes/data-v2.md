# code-py-v2: deduplicated, leakage-checked train/val (SQ102)

Builder: `experiments/data_integrity/build_v2.py`. Test: `experiments/data_integrity/test_build_v2.py`.
Spec source: `notes/data-integrity.md` section 3.5 (+ `proposed-claims.md` C148, `proposed-side-quests.md` SQ102).
Not yet built on the real corpus: sizes below are estimates, not results.

## Definition

1. **Input**: v1 `docs_train.jsonl` (987,154 docs) + `docs_val.jsonl` (4,000), tokenized with the committed v1 tokenizer
   (`gogeta/assets/tokenizer-code-py-v1.json`, copied unchanged to `tokenizer.json`; vocab 8192, `<|eos|>` = 0, same
   encoding as v1: `add_special_tokens=False`, eos appended per doc).
2. **Families**: 64-perm MinHash over token 8-grams (as `corpus_families.py`), LSH banding tuned for est. Jaccard >= 0.5
   (candidates verified on the full signature), single linkage = connected components, over train + val together.
3. **Val** (redrawn at group level): candidates = the 4,000 v1 val docs, then up to 20,000 singleton-family train docs in a
   seeded order. Every candidate's whole family leaves train. A candidate becomes val if >= 13 tokens, < 0.30 of its unique
   13-grams occur in the v2 train pool (K9's metric, no boilerplate drop), and no earlier val doc is in its family; stop at
   4,000 docs (v1 size; `--val-docs`). v1 val docs have priority.
4. **Train**: keep ONE doc per remaining family (the one with the lowest seeded hash rank), then drop every train doc that
   shares a rare first top-level def/class name (<= 50 docs corpus-wide, not in the generic list of `group_proxy.py`) with a
   val doc. Kept docs stay in v1 file order. Final audit (the build fails otherwise): every val doc < 0.30 against the
   final train, no family shared by train and val, no val doc in train.

## Run

```
GOGETA_DATA=/root/fusiondance-data python experiments/data_integrity/build_v2.py --manifest v2.manifest.json
# options: --src --out --workers N --seed 1234 --val-docs --val-extra --name-link-max 50 --big-family-cap 1
```
Needs `docs_train.jsonl` in `code-py-v1/` (fresh box: build v1 with `KEEP=1`). Refuses a non-empty `--out`. Output
`datasets/code-py-v2/`: `train.bin`, `val.bin` (uint16), `docs_val.jsonl`, `meta.json`, `tokenizer.json`, plus
`train_keep_ids.npy` (v1 train line index of each kept doc) and `val_ids.json` (combined index; >= 987,154 means v1 val
line `id - 987154`). No `docs_train.jsonl`. `meta.json` has v1's keys (`splits.*.chars_per_token`, `vocab_size`, `eos_id`) plus a
`v2` block (params, drop counts, audited max overlap). Deterministic: same inputs and seed give byte-identical files for
any worker count (tested 1 vs 3 workers). `--manifest` follows `dataset.manifest.json` (sha256 per file and per 64M-token block).

## Expected cost (estimate, not measured on the real corpus)

Pipeline: read + tokenize + MinHash (parallel by 4,096-doc block), LSH bands in parallel, numpy connected components, two
13-gram scans of ~260M tokens. Extrapolated from the ~10-15 CPU-min of the families run plus tokenization: **about 3-6 min
wall on 24 cores, 2-4 min on 64**. Peak RAM roughly 6-9 GB (raw lines 1.1 GB, tokens 0.5 GB, signatures 0.25 GB, workers fork
copy-on-write). Output about 0.4 GB. The synthetic test (3,200 docs) takes ~3 s per build.

## Differences from v1

- Train ~74% of docs / ~80% of tokens expected (C148: 737.6k docs, 214.7M tokens for keep-one with val families dropped; the
  20k extra val candidates, name-link drops and the one-per-family rule at a slightly different linkage move it a few points).
- Val is not v1 val: 13-gram-overlap-filtered v1 val docs + docs drawn from train, so v2 val numbers (bits/char) are NOT
  comparable with v1 val or `val_clean80`. Tokenizer and doc format are identical.
- Template redundancy below family level is NOT removed (singleton docs still share ~56% of 13-grams with others).

## Ambiguities resolved (most conservative reading)
- "Cap huge families at 300 instead of one": default keeps **1** per family everywhere (matches C148 and SQ102);
  `--big-family-cap 300` reproduces the 300 reading (families > `--big-family-min` 300). Use the default for claims.
- "Group = family union first-def-name": a full union risks one giant connected component, so names are applied **one hop**
  (train docs sharing a rare name with a val doc are dropped), never transitively.
- Rejected val candidates are not returned to train (keeps the overlap guarantee monotone: final train is a subset of the
  pool the check ran against); the cost is up to ~20k extra docs lost. Docs with < 8 tokens have no shingles and stay singletons.
- Test split (`code-py-v1-test*`) is untouched; `-indist` docs are not excluded here (apply `train_exclude_doc_ids` separately).
- Signatures keep the low 32 bits of each MinHash value (half the RAM; collision odds 2^-32 per perm), so families can
  differ at the margin from the 64-bit `corpus_families.py` labels.
