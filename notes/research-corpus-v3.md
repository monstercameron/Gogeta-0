# Research: a real-world Python corpus for code-py-v3

*Research subagent, 2026-11-22. Scope: find real-world Python corpora to replace or extend code-py-v1/v2, and plan a
reproducible `code-py-v3`. Every dataset fact below was read on 2026-11-22 from the Hugging Face dataset card, the Hub API
(`/api/datasets/<id>`: `gated`, `sha`, file listing with LFS sha256) or the datasets-server (`/size`, `/statistics`).
**No data shard was downloaded.** Labels (AGENTS.md §0): **card** = stated on the dataset card, **API** = Hub/datasets-server
metadata, **computed** = arithmetic on those numbers, **predicted**, **unverified** = could not be checked.
Token estimates use our 3.7 chars/token unless marked; predicted real-code compression is worse (section 3.6).*

## 0. Why (recap of on-disk evidence)

- v1 = OpenCoder `opc-annealing-corpus` (`algorithmic_corpus` + `synthetic_code_snippet`, ODC-By), 991,154 Python docs,
  262.4M tokens, BPE 8192 (`tools/prepare_data.py`). No source/repo id exists (`notes/data-integrity.md` §1).
- K9: 41.7% of val docs have >= 80% of their 13-grams in train; 27.1% of docs sit in a MinHash family at J >= 0.5;
  singleton docs still share 56% of their 13-grams (template redundancy). v2 dedup keeps 698,714 docs / 202.5M tokens.
- Coverage (`notes/research-code-moe.md` §2, 4,000 val docs): `async`, `await`, `subprocess`, `open(`, `pickle`: 0% of
  docs; `ast.parse` fails on all 2,556 algorithmic docs. SQ225: 63% algorithmic / 37% snippets.
- C197 (PARTIAL, n=1): unique params look like our lever. A 100M-param model needs >= 1-2B unique tokens to avoid
  many epochs; v2 has 0.2B. E60 (Code-MoE) needs categories that do not exist in v1.

## 1. Candidates (checked 2026-11-22)

All rows need **download approval** from the human (AGENTS §0a: downloading is human-only). The last column lists what
is needed *beyond* that.

| # | Dataset (HF id, revision `sha` pinned) | Python size | Source | Licence / gated | Dedup, filters, PII, decontam | Format, practicality on a 60-100 GB box | Needs human beyond download approval |
|---|---|---|---|---|---|---|---|
| A | `HuggingFaceCode/stack-v3-train` @ `159792e7` (lastModified 2026-11-16) | Python: 159,360,122 files, 595.5 GB, 154.0B tokens by the card's own estimate (`stats/train/stats_by_language.json`); ~161B at 3.7 c/t (computed). Median file 1,218 B, p95 12.6 KB | GitHub default-branch HEAD, crawl finished 2025-08-07 (card); 173M repos, 713 languages | Dataset ODC-By 1.0 (card); files tagged `license_type` `permissive` or `no_license`, non-permissive excluded (card). **`gated: false`** (API) | Global near-dedup: MinHash 256 perms, 5-grams, LSH, J >= 0.7, one file per cluster kept (card); v3.1 removed leaked exact dups. StarCoder2 quality filters (alpha >= 25%, avg line <= 100, max line <= 1000, ...). StarPII redaction with placeholders (`<EMAIL>`, `<KEY>`, ...). **No decontamination** stated | Parquet, 3,545,642,520,539 B total (API `/size`), shards ~300-450 MB (a few 0.6-1.3 GB), each with LFS sha256 (API). **One row = one repository**, all languages mixed, no per-language config: Python needs filtering per shard (~5% of bytes, computed 595 GB / 11.74 TB in-memory). Shard order/grouping not documented | Nothing (no login, no terms click). Opt-out removals happen at patch releases, so pin the revision |
| B | `codeparrot/codeparrot-clean` @ `35a59fb0` (2022-10-10) | 5,361,373 files, ~50 GB (card) -> ~13.5B tokens (computed). Partial stats (477,028 rows): ~60% permissive (MIT, Apache, BSD, ISC, CC0, Unlicense), ~40% GPL/AGPL/LGPL/MPL/EPL (computed from `/statistics`) -> ~8B permissive tokens (computed, extrapolated) | GitHub Python files; snapshot date not on the card (unverified; believed BigQuery `github_repos`) | **No dataset licence tag**; per-file `license` column. **`gated: false`** (API) | Exact dedup only; avg line < 100, max line < 1000, alnum > 0.25, keyword auto-gen removal (card). No near-dedup, no PII removal, no decontam stated | 54 `json.gz` shards, **12.81 GB total** with sha256 (API). Python-only, `repo_name` column for repo grouping. Easiest download of all | Nothing beyond download (no terms). Licence of the compilation itself unstated: keep permissive files only |
| C | `common-pile/stackv2_edu_filtered` @ `c354dbe8` (2025-06-06) | 69,588,607 docs, 255 GB UTF-8 across 15 languages incl. Markdown (card, paper 2506.05209); **Python share unverified** (`metadata.language` exists; the 8 preview rows are all Markdown) | Stack v2, openly licensed subset, educational classifier ("lower threshold than SmolLM2"; value unverified) | Openly licensed only: every repo licence on the Blue Oak list, per-doc `metadata.license` (card). No licence tag. **`gated: false`** (API) | Common Pile-wide Dolma Bloom-filter dedup (> 90% shared 20-grams), regex PII (email/phone/IP), RedPajama code heuristics (paper). No decontam stated | 95 `json.gz` shards, ~83 GB (card). Mixed languages; preview suggests language-sorted shards (unverified) | Nothing beyond download. Cleanest licensing of all candidates |
| D | `hongliu9903/stack_edu_python` @ `76eeb3eb` (2025-07-31) | 25,286,012 rows (API), parquet 31.6 GB, 72.8 GB in memory -> ~17-19B tokens (computed); row count equals Stack-Edu Python's 25.3M (card of E) | Unofficial re-host of Stack-Edu Python **with text**; build undocumented | No licence stated; `license_type`, `detected_licenses` columns. **`gated: false`** | Inherits Stack-Edu (edu score >= 3) and Stack v2 near-dedup (unverified for this copy) | 49 parquet shards, 31.6 GB. Python-only, easy | Nothing formally; **provenance risk**: verify `blob_id`s against E's id list before use |
| E | `HuggingFaceTB/stack-edu` (Python) @ `eeec5caa` | 25.3M files, 21.8B tokens (SmolLM2 tokenizer) (card) | Stack v2 filtered by edu classifier, threshold 3 (card) | Stack v2 terms; ids + licence columns only. `gated: false` for the id list | As Stack v2 + edu filter | 5 parquet id shards (~2.5 GB); **no content**: files must be fetched from the Software Heritage S3 bucket with AWS credentials | AWS account/credentials (possible spend), SWH bulk-access terms |
| F | `bigcode/the-stack-v2-dedup` @ `611ef38f` | Python size not on the card; 32.1 TB dedup, all languages | GitHub via Software Heritage | `license_type` permissive / no_license; **`gated: auto`** (share contact info, accept terms) | Exact + near-dedup; no PII removal (card warns of keys/emails); no decontam | **Ids only**; contents from SWH S3 with AWS credentials; bulk needs an SWH/INRIA agreement (card) | Login + accept terms + AWS credentials + SWH agreement. Impractical |
| G | `bigcode/starcoderdata` (python/) @ `9fc30b57` | 59 parquet shards under `python/` (API); Python bytes not on the card (unverified); 783 GB / ~250B tokens all languages | The Stack v1.2 | Stack terms; **`gated: auto`** | Near-dedup, PII removed, decontaminated (benchmarks not named on the card) | Parquet, Python-only folder, convenient | Login + accept Stack terms (share email/username) |
| H | `bigcode/the-stack-dedup` (v1.2, data/python) @ `17cad72c` | Python size not on the card (unverified) | GitHub 2015-2022 | 193 permissive licences only; **`gated: auto`** | Exact + MinHash 256 / J 0.85 near-dedup; PII "work in progress"; no decontam | Parquet per language, content included | Login + accept terms |
| I | `codeparrot/github-code-clean` @ `c48d40f9` | Python share unverified (card subset list shows 645k "Python-all" rows, likely partial) | GitHub (BigQuery) | Dataset tag apache-2.0; per-file licence configs (`Python-mit`, ...); `gated: false` | Line-length/alnum/auto-gen filters; **no dedup mentioned** | 880 mixed-language parquet shards, 389 GB storage, loading script | Nothing; weaker than B for our use |
| J | `nvidia/Nemotron-Pretraining-Code-v1` / `-v2` | Not stated per language; v2: 835.8M records, ~897 GB, mixed real (cutoff 2025-04-15) + synthetic | GitHub + LLM-generated (Qwen3, DeepSeek-R1, Mixtral) | NVIDIA Data Agreement for Model Training; **gated** | Exact + fuzzy dedup (v1); no decontam stated | Parquet; content vs metadata-only ambiguous on the card | Login + accept NVIDIA agreement; synthetic share conflicts with "realistic code" |

Also checked and rejected: `OpenCoder-LLM/RefineCode-code-corpus-meta` (MIT, **metadata only**, 336.8M rows; "unable to
provide the complete source code"); `OpenCoder-LLM/opc-fineweb-code-corpus` (MIT, ungated, 100.9M web pages: code-related
prose, not code files); `bigcode/commitpackft` (Python 56,025 commits, 133 MB: far too small); `bigcode/the-stack-smol`
(gated, 10,000 Python files); `togethercomputer/RedPajama-Data-1T` GitHub (59B tokens all languages, MIT/BSD/Apache only,
file-level dedup, hosted on data.together.xyz with SHA256SUMS; Python share not reported); SlimPajama (card returned 401,
unverified). `HuggingFaceCode/stack-v3-full` (language-partitioned contents, 113.7 TB, unfiltered) is a Storage Bucket; its
page returned 401 (access unverified).

## 2. Recommendation

**Primary: A, The Stack v3 train (`HuggingFaceCode/stack-v3-train`), Python files with `license_type == "permissive"`.**
- Realistic: real GitHub repositories, 2025 crawl, all of Python's long tail (async, subprocess, I/O, tests, packaging).
- Size: 154B Python tokens upstream; we need 2-3B after filtering, i.e. ~2% of it. No epochs even at 100M params.
- Licence: dataset ODC-By (same family as v1's ODC-By); we keep only `permissive` files (drop `no_license`, which carries
  no grant). **Ungated** (`gated: false`): no login, no terms click; only the download approval.
- Reproducible: Hub revision sha + LFS sha256 per shard; deterministic shard subset; already near-deduped (J >= 0.7) and
  PII-redacted upstream, which reduces our work.
- Structural win: **one row = one repository** (`repo_id`, `commit_id`). That is the source/group id v1 never had
  (data-integrity §1), so val/test can be split at repo level and leakage via sibling files is controllable.
- Costs and risks: mixed-language shards (download ~110-300 shards, 45-120 GB, streamed and deleted; section 3.8); shard
  order undocumented (SQ276 census first); no decontamination upstream; dedup leaves "holes" in repos (maintainer,
  discussion #4); copyright-complaint threads on the Hub (#5, #16, #23, #27) concern `no_license` repos, which we drop;
  the repo is still being patched (lastModified 2026-11-16): never float the revision.

**Fallback: B, `codeparrot/codeparrot-clean`, permissive-licence files only.**
- Python-only, ungated, a single 12.81 GB download (54 sha256-pinned shards), `repo_name` for repo grouping, ~8B
  permissive tokens (computed from partial stats) before our near-dedup. Older snapshot and no upstream near-dedup/PII
  removal, so our MinHash pass and a PII regex pass do more work; overlap with A's repos is expected (do not mix them
  without a cross-corpus dedup).
- Choose B if A's census (SQ276) shows Python is spread so thinly that > 150 GB must be fetched, or the human prefers the
  smallest download.

**Quality arm (not a replacement):** C (`common-pile/stackv2_edu_filtered`, openly licensed + edu-filtered, ungated) as an
SQ281 ablation once its Python share is known. D is not recommended as a primary: an undocumented re-host.

## 3. Build plan for `code-py-v3`

Builder `experiments/data_integrity/build_v3.py` (to write), importing `build_v2.py`'s hashing, MinHash, LSH and
connected-components functions unchanged; tests in `test_build_v3.py` on a synthetic repo set (as `test_build_v2.py`).

### 3.1 Revision pinning and deterministic shard selection
1. Resolve the revision once: `HfApi().dataset_info("HuggingFaceCode/stack-v3-train", revision=REV)` with
   `REV = 159792e76c98d74843d3be3fb003aad925e3a8f0`; list `data/*.parquet` with `lfs.sha256` and size at that revision.
2. Census (SQ276, human-approved small transfer): read only the leaf columns `files.language`, `files.license_type`,
   `files.size_bytes`, `files.is_vendor` of every shard by HTTP range reads (pyarrow over `hf://...@REV`), giving the
   permissive-Python bytes per shard without fetching contents. Output `census.parquet` + its sha256.
3. Order shards by `sha256(f"{SEED}:{filename}")` (SEED = 20261122) and take shards in that order until the census says
   the permissive-Python bytes reach 1.3x the target (headroom for filters/dedup). Write the list (filename, size,
   sha256) to the manifest **before** downloading; verify every download's sha256; delete each shard after extraction.
4. Target: **train 2.5B tokens** (enough for a 100M model at ~25 tokens/param, Chinchilla-style ratio reported by
   Hoffmann et al., not verified here), val 2M, test 2M. If the census shows uniform ~10-14M usable tokens/shard
   (predicted), that is ~200-260 shards, ~80-105 GB transferred, never more than ~4 shards on disk.

### 3.2 Filters (in this order; counts per step go to `meta.json`)
- `language == "Python"` and extension `.py` (drop `.pyi`, `.pyx`, notebooks); `license_type == "permissive"`;
  `is_vendor == False`; path not under `site-packages/`, `vendor/`, `third_party/`, `node_modules/`, `.venv/`, `build/`.
- Size: 200 <= chars <= 100,000 (v1 used <= 20,000; real modules are longer, p95 12.6 KB upstream; the loader chunks).
- Auto-generated: drop `*_pb2.py`, `*_pb2_grpc.py`, Django `migrations/NNNN_*.py`, and files whose first 20 lines match
  `(?i)auto-?generated|automatically generated|generated by|do not edit` (Qt `pyuic`, SWIG, Thrift, protoc).
- Data blobs: drop if any run of `[A-Za-z0-9+/=]` > 1,000 chars or > 50% of lines are numeric/string literals only.
- Re-apply StarCoder2-style line filters (avg line <= 100, max <= 1000, alnum >= 0.25) to be robust to upstream changes.
- Parseability: keep only files that `ast.parse` under Python 3.12 (drops Python 2 code; also makes SQ254-style AST
  labels possible). Record the drop rate; predicted 5-15% of files.
- PII: keep upstream placeholders, count them; add a regex pass for leftover emails, IPs and `AKIA...`/`ghp_...`/
  private-key headers (drop the file if a secret pattern remains).
- Exact dedup: sha1 of whitespace-normalised text (upstream already removed most).

### 3.3 Near-dedup and grouping (reuse `build_v2.py`)
- Tokenize with the chosen tokenizer (3.6), 64-perm MinHash over token 8-grams, LSH tuned for J >= 0.5, single-linkage
  families (exactly v2's parameters: `num_perm 64`, `ngram 8`, `lsh [16, 4]`, `max_bucket 60`). Keep one doc per family
  (seeded hash rank). Upstream dedup was J >= 0.7 on 5-grams, so this pass removes the 0.5-0.7 band (predicted < 10%).
- Group key for splits = **repo_id ∪ MinHash family** (union-find over both). A whole group goes to one split; this
  closes both sibling-file and copied-file leakage. Cap groups at 2,000 files by seeded sampling (huge monorepos).

### 3.4 Decontamination (train side; thresholds fixed before the run)
1. **Eval battery** (`notes/eval-battery.md` family F; items from `experiments/data_integrity/benchmark_contamination.py`):
   HumanEval, HumanEval+, MBPP (full), MBPP+, CRUXEval. Drop a train doc if, for any item, a prompt/solution/code
   component has >= 50% of its 13-grams in the doc, or a verified normalised common substring >= 100 chars (the script's
   `contaminated` rule, applied doc-side). Also drop whole repos whose name matches
   `(?i)human-?eval|mbpp|evalplus|cruxeval` (benchmark mirrors). Afterwards re-run `benchmark_contamination.py` against
   v3: predicted `contaminated` HumanEval <= 5/164 (v1: 74/164), MBPP <= 10/500 (v1: 103/500).
2. **Frozen test `code-py-v1-test`** (UncheatableEval GitHub-Python 2026-10, 462 docs): drop any v3 doc with >= 0.30 of
   the test doc's 13-grams, or in a family with a test doc at J >= 0.4, plus any repo of the same name. The crawl
   (2025-08-07) predates the slice, so predicted ~0 drops; the check proves it.
3. Memory trick: build the 13-gram set of the *small* side (benchmarks, test, val: a few M entries) and stream train
   against it; never materialise train's n-gram set (3B entries would need ~24 GB).
4. Self-test before the real run: SQ278 (planted HumanEval/MBPP solutions: verbatim, reformatted, renamed, embedded).

### 3.5 Splits and the new frozen test `code-py-v3-test`
- Seeded group draw (`sha256(f"{SEED}:{group_id}")` order): test groups until **2M tokens**, then val groups until 2M,
  rest train. Then audit (build fails otherwise): every val/test doc < 0.30 of its 13-grams in train (K9's metric), no
  family or repo shared across splits, no test doc with >= 0.30 overlap with **v1 train** either (so existing
  v1/v2 checkpoints can be scored on v3-test fairly).
- Lock as in data-integrity §4: read-only files, sha256 in `test-split.manifest.json`-style manifest, `test_guard.py`
  access log; CIs resample **repos** (cluster bootstrap), not files.

### 3.6 Tokenizer decision: keep BPE 8192 v1 (`gogeta/assets/tokenizer-code-py-v1.json`) for v3.0
- **Keep** (recommended): same vocab and embedding shapes, so params/FLOPs per config are unchanged; every v1/v2
  checkpoint can be scored on v3 val/test (transfer); data is the only ingredient that changes (AGENTS §5). Bits/char
  metrics stay comparable across corpora.
- **Cost (predicted):** the v1 tokenizer was fitted to algorithmic snippets (3.73 chars/token on v1 val,
  `notes/research-tokenizer.md`); on real code expect ~3.0-3.4 (long identifiers, comments, strings, non-ASCII), so a
  200M-token run reads ~10-20% fewer characters than on v1 and token-based horizons are not char-matched across corpora.
- **Retrain** (v3.1 option): BPE 8192 on a 1 GB seeded sample of v3 train. Better compression, but every number is a new
  series and old checkpoints cannot be scored on its token streams. Decide by SQ277: retrain only if the v3-fitted 8192
  BPE is >= 10% better in chars/token on v3 val; then cut `code-py-v3t` as a separate dataset, not a replacement.

### 3.7 Manifest (`experiments/data_integrity/dataset-v3.manifest.json`, schema of `dataset-v2.manifest.json` plus)
`source` + `revision` sha; shard list (filename, bytes, LFS sha256, selection rank) and the census sha256; seed; filter
parameters and per-step drop counts; dedup params, families, docs/tokens dropped; decontam params and per-benchmark
drop counts; split rule, group counts, audited max overlaps; tokenizer sha256; output sha256 per file and per 64M-token
block; licence histogram of kept files; `attribution.jsonl.gz` sha256 (repo_path, commit_id, file_path,
detected_licenses for every kept file: ODC-By and the per-file licences require attribution; stays on the data disk);
builder `git rev-parse HEAD`. `--verify` re-hashes everything (as `build_test_split.py --verify`).

### 3.8 CPU, disk and time estimate (predicted, not measured; scaled from v2's ~10-15 CPU-min families run on 262M tokens)

| Step | 24 cores | 64 cores | Notes |
|---|---|---|---|
| Census (column chunks only) | 5-10 min | 3-5 min | network-bound; a few GB |
| Download + verify ~230 shards (~90 GB) | 8-15 min | 8-15 min | at 100-200 MB/s aggregate; overlaps with extraction |
| Parquet decode + filters incl. `ast.parse` (~6-8M files) | 6-10 min | 3-5 min | ast.parse ~1-2 ms/file |
| Tokenize ~10 GB of text | 3-6 min | 2-3 min | Rust tokenizers, batch encode |
| MinHash + LSH + components (~3B tokens) | 8-15 min | 4-7 min | ~12x v2's input |
| Decontam + split audits (13-gram streams) | 10-20 min | 5-10 min | small-side hash sets |
| **Total wall** | **~45-75 min** | **~25-45 min** | peak RAM ~40-60 GB (tokens 6 GB, signatures ~2 GB, workers) |

Disk: <= 4 shards in flight (~2 GB) + extracted text ~12 GB + tokens ~6 GB + attribution ~1 GB: **< 40 GB**. Output
`train.bin` ~5 GB (uint16). A CPU box at ~$0.3-1/hr makes this < $1; check the host's per-TB ingress price before
approving (~100 GB transfer).

## 4. Re-baseline plan on v3

Rule: v3 is a new series; v1/v2 numbers are never mixed with v3 numbers in one table (eval-battery §1 "frozen" rule).

| Step | Runs | GPU time (5090, ~5 min per 200M-token 14M run) | Cost at ~$0.5/hr |
|---|---|---|---|
| R0 LR check (SQ280): E10 and E1 at 3 LRs x 1 seed, 20M tokens | 6 | ~3 min | ~$0.03 |
| **R1 confirmatory: E1 vs E10, 7 seeds each, 200M tokens**, pre-registered paired test (as Phase 2b-5090) | 14 | ~70 min | ~$0.60 |
| R2 scale lever (C197): E32 and E2 at E2's FLOPs, 3 seeds each | 6 | ~60 min (~10 min each) | ~$0.50 |
| R3 loop pair: E9 vs E2, 3 seeds each | 6 | ~45 min | ~$0.40 |
| R4 unique-token payoff: E10-shaped models at ~50M (1B tokens) and ~100M (2.5B tokens), 2 seeds | 4 | ~15-25 h (FLOPs scale ~18x and ~90x a 14M/200M run, computed) | ~$8-13 |
| R5 re-score: all v1/v2 checkpoints on v3 val + v3-test (guarded, once) | 0 train | ~10 min | ~$0.10 |
| **Total** | 36 | **~3 h without R4; ~20-28 h with R4** | **~$1.6; ~$10-15 with R4** |

Spread over 8 boxes (AGENTS §0h pipelining), R0-R3 finish in < 30 min wall. Report on v3 val (clean by construction),
then one guarded read of `code-py-v3-test` per claim.

**What would change the project's direction** (thresholds fixed now; v1 references: E1 vs E10 +0.08%, 95% CI
[-0.00087, +0.00132] as quoted in README; E32 -7.2% vs E1; E2 +2.6% vs E9):
1. **E1 beats E10 on v3 by > 1% with the paired CI excluding 0** -> the linear-mixer advantage is a real-code /
   long-file effect (consistent with C193's long-file signal), C5's refutation was corpus-specific: revive the hybrid
   thesis on v3 and make v3 the default corpus. A tie again (|gap| < 0.5%) -> C5 holds on realistic code too.
2. **C197's gain shrinks below 2% on v3** -> the v1 "unique params" lever was template memorisation; demote the scale
   arc. **It holds or grows (>= 5%)** -> scale on v3 (R4) becomes the main line and loops are retired unless (3).
3. **E2 vs E9 gap reverses (E2 <= E9) on v3** -> loops help on real code; the loop thesis gets a second life.
4. v1-trained checkpoints within 2% of v3-trained ones on v3-test -> v1 was less harmful than feared (keep v1 results
   as valid but narrow); > 10% worse -> treat every v1 number as in-distribution only, and say so everywhere.
5. SQ279 shows category coverage (async, subprocess, I/O, typing) >= 1% of docs each -> E60 Code-MoE is runnable on v3.

## 5. Open questions and limits
- Python share per shard and permissive vs `no_license` share of Python in A are **unknown** until SQ276.
- Upstream token counts use the card's own tokenizer (595.5 GB / 154.0B = 3.87 bytes/token); ours will differ.
- B's licence split is from partial statistics (477,028 of 5.36M rows) and B's snapshot date is unverified.
- C's Python share and edu threshold are unverified; D's provenance is undocumented.
- No dataset here states decontamination against our battery except G ("decontaminated", benchmarks unnamed): §3.4 is required regardless.

## 6. Draft side quests (added to `experiments/side-quests.md` by the main agent, v0.4.85)

| ID | Idea | Why it might help | Status | Outcome / what we did |
|---|---|---|---|---|
| SQ276 | **Stack v3 shard census**: at revision `159792e7`, read only the `files.language`, `files.license_type`, `files.size_bytes`, `files.is_vendor` column chunks of every `data/*.parquet` shard (HTTP range reads); per-shard permissive-Python bytes, shard-order structure, and the shard count K for 2.5B tokens | Sets code-py-v3's download size before any content is fetched and decides primary (Stack v3) vs fallback (codeparrot-clean) (research subagent, notes/research-corpus-v3.md) | `idea` (`needs_human`: download approval, a few GB) | ~10 CPU-min |
| SQ277 | **Tokenizer fit on real code**: chars/token of the v1 BPE-8192 vs a BPE-8192 and a 16384 fitted on a 1 GB seeded v3 sample, on v3 val and code-py-v1-test | Decides keep-vs-retrain (retrain only if >= 10% better); the v1 tokenizer was fitted to algorithmic snippets (research subagent) | `idea` (after the v3 build) | ~10 CPU-min |
| SQ278 | **Decontamination self-test**: plant HumanEval(+)/MBPP(+)/CRUXEval solutions into v3-like files (verbatim, reformatted, half renamed, embedded in a 5 KB module) and measure the doc-side filter's recall and false-positive rate on 10k clean files | The v3 decon filter reverses `benchmark_contamination.py`'s direction; pass if recall >= 0.95 verbatim/reformatted, >= 0.8 renamed, FP <= 0.1% (research subagent) | `idea` (do before the real build) | ~3 CPU-min |
| SQ279 | **Category coverage on v3**: re-run research-code-moe §2's marker table and `ast.parse` rate on 4,000 seeded v3 val docs vs v1 val | v1 had 0% async/subprocess/I/O and 0/2,556 parseable algorithmic docs; predicted >= 1% per category and >= 95% parseable on v3, which would unblock E60 (research subagent) | `idea` (after the v3 build) | ~5 CPU-min |
| SQ280 | **v3 pilot before the 7-seed confirmatory**: E10 and E1 at LR {1.5e-3, 3e-3, 6e-3}, 1 seed, 20M tokens on code-py-v3; plus E1/E10 seed 0 at 200M for a timing and loss sanity check | Real code may shift the tuned LRs (K7 tuned on v1); a wrong LR would bias R1 (research subagent) | `idea` (after the v3 build) | ~15 GPU-min (~$0.15) |
| SQ281 | **Edu-quality arm**: the same v3 pipeline on `common-pile/stackv2_edu_filtered` Python (openly licensed, edu-filtered), E10 at 200M tokens, 2 seeds per corpus, scored on both v3-test and code-py-v1-test | SmolLM2 reports educational filtering helps small models; tells whether "realistic" or "curated realistic" code is the better lever at 14M (research subagent) | `idea` (`needs_human`: download approval, ~83 GB, Python share unknown) | ~20 GPU-min + ~1 CPU-h build |
