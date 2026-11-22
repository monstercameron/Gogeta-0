Functions to register in the cockpit (all return JSON-able dicts; modules in `experiments/data_integrity/`).

| Cockpit action | Function | Cost | Notes |
|---|---|---|---|
| Dedup self-test | `dedup.selftest()` | 5 s | planted-duplicate precision/recall |
| Compare two texts | `dedup.compare_texts(a, b)` | ms | char-5 / identifier-blind Jaccard, containment, longest common substring |
| Near-dups in a doc set | `dedup.find_near_duplicates(docs, threshold, mode, ref=None)` | seconds per 1k docs | MinHash/LSH + verification |
| Overlap with train | `dedup.presence_in_streams(queries, streams, n=13)` | ~1 min vs train | K9's metric for any queries |
| Corpus families | `corpus_families.run_families()` | ~5 CPU-min, 2 GB | writes results/families.json |
| Clean val | `corpus_families.clean_val()` | seconds | writes clean-val.manifest.json |
| Benchmark labels | `benchmark_contamination.run(limit=None)` | ~2 CPU-min | writes results/benchmark-contamination.json |
| Build / verify test split | `build_test_split.build_external()`, `.build_carve()`, `.verify(name)` | 1 min | writes read-only files under ~/fusiondance-data/datasets |
| Dataset manifest | `build_dataset_manifest.build()`, `.check()` | 10 s warm | sha256 per file and 64M-token block |
| Test-access guard | `test_guard.require_test_access(split, ckpt_sha, selected_on_val_only)` | ms | any eval reading the test split must call it |
| Group proxy | `group_proxy.run()` | 2 CPU-min | first-def-name proxy |

Code that should eventually move (paths were off-limits during Phase 2b): the guard call into `evals/run_suite.py` (before loading any `*-test*` dataset), the clean-val scorer into `evals/`, regeneration of `evals/tasks/contaminated_ids.json` from `results/proposed-contaminated_ids.json`.
