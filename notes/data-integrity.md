# Data integrity (workstream A1)

Written 2026-11-22. Code and results: `experiments/data_integrity/`. Claims C140-C149 (proposed in `experiments/data_integrity/proposed-claims.md`). Trigger: the audit's P0 list, then K9 failing (41.7% of val docs have >= 80% of their 13-grams in train; 70.7% have >= 50%; the same for a pseudo-val made of train docs; shuffled control 0%).

## 1. What the data is

- Source: OpenCoder `opc-annealing-corpus`, configs `algorithmic_corpus` (qa, 7 shards) and `synthetic_code_snippet` (snippet, 10 shards). The raw parquet has exactly three columns, `type`, `lang`, `text`. **There is no source, repo, project or problem id** (checked on every shard's schema). `tools/prepare_data.py` keeps `lang == python`: 997,044 rows, minus 1 empty and 5,889 exact copies = 991,154 docs, shuffled with seed 1234, 4,000 to val, 987,154 to train.
- **No unused Python documents exist.** Every kept doc is in train or val; the other ~7.4M rows are bash / other languages; the 5,889 dropped rows are exact copies of kept docs. (There is no `gogeta/data.py`; the loader is `TokenData` in `gogeta/train.py`, reading `train.bin`/`val.bin`.) A test set disjoint from train and val has to come from outside this corpus, or from a retrain with docs held out.

## 2. Toolkit (`dedup.py`, numpy only, every function returns JSON-able dicts)

| Engine | What | Used for |
|---|---|---|
| Doc engine | MinHash (128 perms) over char 5-gram or identifier-blind word 5-gram shingles, LSH banding (b, r tuned to a threshold), verified by exact Jaccard, containment, and the longest common whitespace-normalised substring >= 50 chars | val vs test, pool dedup, planted self-tests |
| Substring engine | rolling 64-bit hash of every 50-char window, verified by comparing characters | verification of candidates, benchmark items vs docs |
| Stream engine | token n-gram presence (any-occurrence, `min_count` option), per-doc MinHash signatures straight from `train.bin` (64 perms over token 8-grams), LSH candidate pairs, connected components = families, family report | the 991k-doc corpus, benchmarks, test split |

Self-test on planted duplicates (`results/dedup_selftest.json`, 300 base docs, 20 plants per mutation, 100 fresh negatives, reference = the 300): recall at estimated Jaccard threshold 0.5.

| Mutation | char-5 view | identifier-blind view |
|---|---|---|
| exact copy | 1.00 | 1.00 |
| reformatted (blank lines, indentation, spacing) | 1.00 | 1.00 |
| half the identifiers renamed | 0.85 | **1.00** |
| first 60% of the doc | 0.95 | 0.95 |
| first 30% of the doc | 0.15 | 0.10 (Jaccard too low by design; containment catches it in the stream engine) |
| a foreign doc prepended | 0.90 | 0.70 |

Precision 1.00 in both views (97 and 95 reported pairs, all the planted source; 0 of 100 negatives flagged). MinHash estimate vs exact Jaccard: mean absolute error 0.012 (max 0.11; sd 0.044 expected at J=0.5 with 128 perms). The substring engine finds the planted 54-char run; the stream engine finds a planted token copy (containment 1.0, right doc), a half copy (0.39), nothing for random queries and nothing across an `<|eos|>`. Planted token families among singletons (64 perms, 1.2% token mutations): pair recall 0.963, precision 1.000. Caveat: the plants are synthetic; real near-duplicates are messier. The identifier-blind view is meant for renamed copies, not as a primary detector (on templated code it over-merges; the self-test generator had to be made structurally varied for it to be meaningful).

## 3. Findings

### 3.1 The corpus is self-duplicated (`results/families.json`, `corpus_families.py`)

64-perm MinHash over token 8-grams, LSH tuned for recall, candidates verified on the full signature, single-linkage families. Estimate sd 0.06 at J=0.5, and LSH recall at the threshold is below 1, so these are lower bounds on duplication at that Jaccard.

| Estimated Jaccard >= | 0.5 | 0.7 |
|---|---|---|
| families (of 991,154 docs) | 741,056 | 894,809 |
| docs in a family of size >= 2 | 268,633 (27.1%) | 112,505 (11.4%) |
| tokens in such docs | 19.6% | 7.7% |
| largest family | 5,095 | 2,644 |
| 38 families with >= 1000 docs hold | 78,232 docs | (4 families, 6,535 docs) |
| val docs with a train family member | 1,084 (27.1%) | 472 (11.8%) |
| families spanning train and val | 555 | 372 |

Family-size histogram at 0.5: 722,521 singletons; 11,167 pairs; 3,920 of size 3-4; 1,689 of 5-9; 956 of 10-29; 423 of 30-99; 209 of 100-299; 133 of 300-999; 38 of >= 1000.

**This under-counts what K9 saw**: the pre-registered prediction (> 40% of train docs in a family) missed (27%). K9's measure is one-sided 13-gram containment (70.7% of val docs >= 50%), which also fires on docs assembled from templates shared by many different docs (no single partner has Jaccard >= 0.5). Both are real: families are the right unit for "no copy across splits"; containment is the right unit for "is this text memorised". Even **singleton-family** train docs have a mean 56% of their 13-grams elsewhere in train/val (81% of them >= 30%): the unit of redundancy is the template, not the document.

Source proxy (`group_proxy.py`, `results/group_proxy.json`): with no source id, the first top-level `def`/`class` name is a coarse "problem identity". 93.5% of val docs (87% with a name occurring >= 5 times in train) share it with a train doc; 927k of 987k train docs share their first name with another doc. A naive doc-level split therefore crosses name groups almost always, and crosses MinHash families for 27% of val docs (so ~72% of val docs have no near-twin at J >= 0.5 yet still overlap heavily by templates).

### 3.2 Clean val (`clean-val.manifest.json`)

From K9's per-doc 13-gram overlap (ids are line numbers in `docs_val.jsonl`; sha256 of the id lists and of `docs_val.jsonl` are in the manifest). `val_clean80`: 2,333 docs (overlap < 0.8, primary); `val_clean50`: 1,174 docs (strict). `val_clean50_nofamily` (also no train doc in the same MinHash family): 1,174, i.e. identical: every doc under 50% overlap already has no train family member (prediction clause 5 held). They are still selected from val, so they are a cleaner val, not a test set. Re-score every checkpoint on both lists (SQ100). Expect numbers well above 0.2209 bits/char for E0 (the near-dup docs are scored at 0.101, the rest at 0.278).

### 3.3 Benchmark contamination (`benchmark_contamination.py`, `results/benchmark-contamination.json`)

Items are split into components (prompt / solution / tests; CRUXEval code / io). Labels use thresholds fixed in the script before the run: **contaminated** = a prompt, solution, code or file component has >= 50% of its 13-grams anywhere in train+val, or a verified common normalised substring >= 100 chars with a train doc; **suspect** = >= 25% of 13-grams or 8-grams, or a verified substring >= 50 chars; else **clean**. Tests/IO components never gate.

| Benchmark | items | contaminated | suspect | clean | old screen (C103) |
|---|---|---|---|---|---|
| HumanEval | 164 | 74 (45%) | 81 | 9 | 32 flagged; 30 of them are `contaminated` now, 2 `suspect` |
| HumanEval+ | 164 | 78 | 75 | 11 | none |
| MBPP (full test) | 500 | 103 (21%) | 228 | 169 | 43 flagged; 37 `contaminated`, 6 `suspect` |
| MBPP+ | 378 | 65 | 228 | 85 | none |
| CRUXEval | 800 | 9 (1%) | 186 | 605 | none |
| UncheatableEval GitHub slice (2026-10) | 500 | 1 | 44 (8.8%) | 455 | none |

Reading: the old line screen caught the strong cases but missed more than half of the HumanEval contamination (74 vs 32). The 8.8% "suspect" rate on post-cutoff GitHub code is the floor of that label (generic idioms), so "suspect" is a loose conservative flag, and a `_clean` twin built from `contaminated or suspect` leaves only 9/164 HumanEval items: use the strict label for the headline `_clean` twin (HumanEval 90, MBPP 397 left) and report the lenient twin as a sensitivity check. HumanEval/HumanEval+ agree on the exclude label for 160/164 items; MBPP/MBPP+ for 178/224 shared ids (MBPP+ rewrote the prompts, so prompt-level labels differ). Proposed `*_clean` id file: `results/proposed-contaminated_ids.json` (not applied: `evals/` is out of scope; SQ101).

The causal reading is limited to what was measured: presence of the text in the training corpus, not that the model memorised it.

### 3.4 Frozen test split (`build_test_split.py`, `test-split.manifest.json`)

- **`code-py-v1-test`** (external mode, valid for every checkpoint so far): the 500 files of the UncheatableEval GitHub-Python slice (2026-10, post-dating the corpus). All 500 passed hygiene, pool near-dup removal (0 dropped), the 13-gram rule (< 0.30 of a doc's 13-grams in train/val; the pool's mean overlap is 0.002, max 0.06, 0 of 500 >= 0.30) and the family rule (est. Jaccard >= 0.4 with any train/val doc; none). A deterministic seeded order (`seed 20261122`) takes docs until 1M tokens: **462 docs, 1,000,269 tokens** (`reached_target_tokens`). Pre-registered prediction clause (1) was too pessimistic on size (0.5-0.9M; the slice is 1.08M tokens); clauses (2) and (3) held (0% >= 30%; mean 0.002). `--verify` re-hashes the files and re-checks overlap (max 0.060, `disjoint_ok`).
  Distribution: real GitHub code, not the OpenCoder synthetic/algorithmic mix. It measures generalisation to real code, not in-distribution fit, and its files are long (2.2k tokens each), so 462 docs is a small number of independent units for confidence intervals (A2's bootstrap should resample docs).
- **`code-py-v1-test-indist`** (carve mode, in-distribution): 3,062 train docs (1,000,308 tokens) that are singletons in the 0.5 family clustering and have < 0.30 of their 13-grams elsewhere in train/val (`min_count=2` excludes the doc itself). **Valid only for a retrain that applies `train_exclude_doc_ids` in `extra.json`**: every checkpoint trained so far has seen these docs. Only 19% of the sampled singleton-family docs pass the overlap filter, so the set is biased toward atypical docs (this is the cost of the corpus's template redundancy) and is the reason the v2 retrain should be decided before E4-class runs.
- Files live under `~/fusiondance-data/datasets/<name>/` as read-only (`chmod 444`) `docs_test.jsonl`, `test.bin`, `meta.json`, `extra.json`; nothing is in the repo except the manifest (sha256 of each, docs, tokens, tokenizer sha256, seed, thresholds, the builder's `git rev-parse HEAD`, and the `dirty_paths` of `experiments/data_integrity/`, which were uncommitted when the builder ran: re-run the builder from the committed tree to refresh `builder_git`; the sha256 values do not change). No model metric was computed on either split.

### 3.5 Group-aware v2 split

No source id exists, so the group key must be derived. Proposal for `code-py-v2`, built from one pass over `docs_*.jsonl` (not applied; SQ102):

1. group = MinHash family at 0.5 (this report) **union** the first-def-name proxy for names with <= 50 docs (specific problems), and assign every group wholly to train, val or test;
2. keep one doc per family in train, drop every family with a val/test member, and cap huge families (> 300 docs) at 300 docs by seeded sampling instead of one (to keep the style variety, not the copies);
3. val/test = group-level seeded draw, then re-check with the 13-gram presence rule (< 0.30; K9's metric).

Size estimate (keep-one-per-family, dropping families touching val): 737,585 docs (74.7%), 214.7M tokens (81.8% of 262.4M). At Jaccard 0.7: 890,909 docs, 245.0M tokens (93.4%). The template redundancy (singleton docs with 56% elsewhere) is not removed by any family rule; a 13-gram-level filter would cut far deeper and should be a separate ablation (what the model learns from templated data is the point of the corpus).

## 4. Lock policy for the test set

1. Evaluate `code-py-v1-test` (and `-indist` for a v2 retrain) **only for final, pre-registered claims**: the claim and its prediction are in `PREDICTIONS.md`/the ledger first; the checkpoint was chosen on val/clean-val only; one read per (claim, checkpoint).
2. Never use the test set for tuning, early stopping, LR or architecture choice, seed selection, or in `*_clean` twin construction. If a test look changes a decision, that test set is spent: declare it, and build the next one (a later UncheatableEval month) rather than reuse.
3. `experiments/data_integrity/test_guard.py::require_test_access(split, ckpt_sha, selected_on_val_only)` refuses unless `GOGETA_TEST_UNLOCK=<claim id>` is set, the claim id exists in `PREDICTIONS.md` or `claims.md`, the checkpoint hash is given and selection was val-only, and the (split, claim, checkpoint) was not read before. Every attempt, granted or refused, is appended to `test-access-log.jsonl` (commit it). The guard is cooperative: it protects against accidents and documents intent; the real lock is the read-only files plus the log being reviewed at each version cut.
4. Any eval script that reads `datasets/code-py-v1-test*` must call the guard first (proposal for `evals/run_suite.py`, not applied here).
5. The manifest's sha256 values are verified (`build_test_split.py --verify`, `build_dataset_manifest.py --check`) before every final evaluation.
6. The audit's warning stands for val too: repeated selection on the 4,000-doc val is indirect overfitting. Use `val_clean80` for model selection, the test split once.

## 5. How the pieces fit with K9

K9 (`experiments/checks/k9_leakage.py`) asks one question of val: how much of each val doc's 13-grams are in train. It stays the validity check and its per-doc list seeds `clean-val.manifest.json`. This workstream adds: the corpus-wide family structure (K9 cannot say how much of train is redundant), the fuzzy/renamed/partial-copy view, benchmark labels, and the split design. `dedup.presence_in_streams` reproduces K9's metric (13-gram, no boilerplate drop, token streams) and is what the builder uses to verify the test split. Differences to keep in mind: K9 drops the top-1,000 boilerplate 13-grams in a second view; the presence function does not.

## 6. Not done / limits

- Families use 64-perm estimates (sd 0.06) and 8-gram token Jaccard; LSH recall near the threshold is below 1; borderline edges are not verified exactly (SQ103).
- The identifier-blind (rename) view was not run on the real corpus (cost); renamed copies of benchmark items would be missed by the token scan except through docstring/test text.
- The `suspect` label is loose (8.8% on fresh GitHub); only `contaminated` should gate a headline.
- Test pool is a single month and 462 independent docs; ~0.5 doc-bootstrap sd is large for small effects. SQ104 proposes fetching more months.
- No model was scored on any split (no GPU, no metric on test). The claim that the test split is out-of-distribution relative to training is by construction (source and date), not measured with a model.
