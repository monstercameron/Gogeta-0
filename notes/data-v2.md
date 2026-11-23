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

## code-py-v1sub: the dedup-only control (SQ238)

Builder: `experiments/data_integrity/build_v1_subset.py`; test: `test_build_v1_subset.py`.

**Why.** v2 changes two things relative to v1: it removes near-duplicates, and it shrinks train (698,714 docs / 202.5M tokens vs
987,154 / 262.4M), so a 200M-token run covers ~99% of v2 but ~76% of v1. A v1-vs-v2 gap cannot be attributed to dedup alone.
v1sub is **a uniform random subset of v1 train with v2's doc count and no dedup**; v1sub-vs-v2 isolates dedup at matched
doc count, v1-vs-v1sub isolates coverage/size.

**Definition.** (1) Recompute build_v2's families (same functions and params, train + val together). (2) Exclude every v1 train
doc whose family contains a v1 val doc, and every train doc sharing a rare first def/class name (<= 50 docs, non-generic) with a
v1 val doc, so v1sub is not more contaminated against v1 val than v2 is. (3) Draw exactly 698,714 (read from
`dataset-v2.manifest.json`) of the remaining docs with `numpy.default_rng(1234).choice(replace=False)`, keep v1 file order.
Output `datasets/code-py-v1sub/`: `train.bin`, `val.bin` (v1's, byte-identical), `docs_val.jsonl` (v1's), `meta.json`
(with a `v1sub` block: exclusion counts), `tokenizer.json`, `train_keep_ids.npy`; `--manifest` uses the v2 manifest schema.
The build asserts that its token stream matches v1 `train.bin`/`val.bin` when those exist (same tokenisation path as
`tools/prepare_data.py`).

**Exclusion is a recomputation, not read from v2's outputs.** `train_keep_ids.npy` only lists kept docs, so "removed because of
v1 val" cannot be separated from "removed as a duplicate". The builder therefore re-runs the family computation (deterministic,
identical to v2's). Choice: use all 4,000 v1 val docs (v2 applied the name link only to the 614 accepted ones, and also dropped
families of 20k extra train-drawn candidates, which are not v1-val related and are NOT excluded here). This is a slight superset
of v2's v1-val-driven exclusions: v1sub is at least as clean against v1 val as v2. Counts are in `meta.json` `v1sub`.

**Run** (box with v1 `docs_train.jsonl`, KEEP=1): `python experiments/data_integrity/build_v1_subset.py --manifest v1sub.manifest.json`.
Expected cost: same pipeline as build_v2 minus the two 13-gram scans, about 3-5 min on 24 cores, peak RAM ~6-9 GB, output ~0.4 GB.

**Built and verified (2026-11-23, v0.4.91):** two RTX 5090 boxes (w4-1 and w4-5) built it independently in 43 s and 37 s, with byte-identical output.
- train.bin sha256 `7e0f15f1…c77ee4`: 698,714 docs, 196,510,203 tokens.
- Exclusion: 179,069 v1 train docs excluded (165,732 in v1-val families, 13,337 more by the rare-name link), leaving 808,085 eligible.
- val.bin and docs_val.jsonl are byte-identical to v1's.
- Manifest: `experiments/data_integrity/dataset-v1sub.manifest.json`.
