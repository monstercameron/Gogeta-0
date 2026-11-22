# Benchmark suite for Gogeta-0: survey, recommendation, first checks

*2026-11-22. Research + scaffold by a subagent; nothing here is committed yet. Scaffold: [`evals/`](../evals/README.md).*

Labels (AGENTS.md §0): **[verified here]** = run or read on this machine, evidence on disk ·
**[source]** = read in the installed package source · **[primary]** = read in the primary source
(repo, PyPI JSON, HF API, paper text) today · **[reported]** = someone else's claim, not checked ·
**[predicted]** / **[computed]**.

## 0. TL;DR

- **Standard functional benchmarks floor at our size.** The smallest published HumanEval data
  points are Codex-12M at 2.00% pass@1, PolyCoder-160M at 2.13% and CodeParrot-110M at 3.80%,
  all trained on 26–100B tokens [primary: paper PDFs]. We have ~13–14M params and 200M tokens.
  At 164 problems, one problem is 0.61 points, so variant-vs-variant gaps of a few problems
  can't be resolved. Report them (tier 2), but don't decide anything with them.
- **What discriminates now (tier 1):** compression, i.e. bits per byte, and conditional
  likelihoods, all from lm-eval loglikelihood requests. Two kinds:
  - corpus bpb: in-distribution val, and **fresh GitHub Python** from UncheatableEval 2026-10;
  - likelihood probes: **reference-solution bpb** for HumanEval and MBPP, and **CRUXEval-O output
    likelihood**.

  Tier 1 also has one execution metric, **held-out functional pass@1** (our own val problems,
  sandboxed). On E0 it scores 7/8 on its first 8 problems [verified here, subset].
- **Contamination is real, not hypothetical** [verified here]:
  - 32 of 164 HumanEval problems have ≥ 50% of their long docstring/solution lines verbatim in
    our training docs;
  - 43 of 500 MBPP test statements appear verbatim, though none of their asserts do.

  Every HumanEval/MBPP number gets a `_clean` twin.
- **Harness:** EleutherAI lm-evaluation-harness **0.4.13** (PyPI, 2026-08-31) with a registered
  `gogeta` model (an `HFLM` subclass handed our `GogetaForCausalLM` object). Wiring verified:
  harness log-probs equal a hand-written forward pass (`evals/test_wiring.py`, 6/6).
- **Safety:**
  - lm-eval's built-in code tasks execute in-process, so they are never loaded.
  - Our exec tasks go through `evals/sandbox.py`, which wraps `post_training/rewards.py`.
  - **Found and fixed for evals:** rewards.py's child *can write to /mnt/c* (the repo, all of
    C:) on this WSL setup [verified here].

## 1. Constraints that decide what is informative

| Fact about Gogeta-0 | Consequence for evals |
|---|---|
| ~13–14M params, 200M tokens | Generative benchmarks floor (§3). Use likelihood-based metrics with many tokens per item. |
| Python only, curated OpenCoder annealing data: problem statement + asserts + fenced solution; documented functions | English-instruction suites (RULER, BABILong, LongBench, instruct variants) are out of distribution. Our own prompt format matters: MBPP-style "statement, asserts, ```python" is literally the `algorithmic_corpus` layout [verified here: val docs]. |
| 8k byte-level BPE, no BOS, `<|eos|>` between docs | Compare to other models only in **bits per byte**. Score docs from an `<|eos|>` prefix. |
| 1,024-token context (RoPE table) | RULER's default 4K and long-context suites can't run yet. Tier 3 is scaffolded with code-native recall ≤ 1K. |
| No KV cache in `gogeta/hf.py` | Generation recomputes the whole prefix every token. Tier 2 and exec tasks are slow; batches with mixed left-padding run row-group by row-group. |
| `gdn` blocks need CUDA + fla | CPU checks use attn/swa-only models (E0, random tiny). |

## 2. Survey of candidates

| Candidate | Current version / status (as checked 2026-11-22) | Integration with our model | Discriminates at ~13M Python-only? |
|---|---|---|---|
| **EleutherAI lm-evaluation-harness** | **0.4.13**, PyPI 2026-08-31; previous 0.4.12 (2026-05-11); a 0.5.0.dev1 pre-release exists [primary: PyPI JSON]. Installed and used here. | Two documented routes [primary: `docs/model_guide.md`; source]: (a) subclass `lm_eval.api.model.LM` and register it with `@register_model`, implementing `loglikelihood`, `loglikelihood_rolling` and `generate_until`; (b) hand `HFLM(pretrained=<PreTrainedModel instance>, tokenizer=...)`. HFLM's `__init__` takes a non-str `pretrained` as an already-initialised model [source]. **We use (b) inside (a)**: `evals/gogeta_lm.py` registers `gogeta` as an HFLM subclass. | The harness is neutral. What matters is the task choice (rows below). |
| lm-eval **code tasks** | Installed tasks [source]: `humaneval`, `humaneval_64`, `humaneval_plus`, `humaneval_instruct`, `mbpp`, `mbpp_plus`, `mbpp_instruct`, `mbpp_plus_instruct`, `cruxeval_{input,output}[_08,_cot]`, `humaneval_infilling`, `code_x_glue`. Details [source]: `humaneval` is greedy pass@1; `humaneval_64` is T=0.2, k=2..64; `mbpp` is 3-shot with a `[BEGIN]`/`[DONE]` template. All set `unsafe_code: true` and execute via HF `evaluate`'s `code_eval` in the eval process; `humaneval/utils.py` even runs a test program at import time. **lm-eval's `mbpp_plus` scores only the 3 base `test_list` asserts, not EvalPlus's extra tests** [source: mbpp.yaml `doc_to_target` uses `test_list`, inherited by mbpp_plus.yaml]. | Not used directly (in-process exec; English templates). We re-implemented HumanEval, MBPP, HumanEval+ and MBPP+ as `gogeta_*` tasks: our prompt format, sandboxed execution, real plus tests. | Floors (§3). `humaneval_infilling` needs FIM training: N/A. |
| lm-eval **perplexity / bpb tasks** | `wikitext`, `paloma_*` (16 domains incl. `paloma_dolma_100_programing_languages`), `pile_*`, `c4`, `lambada` [source]. The `loglikelihood_rolling` output type supports `bits_per_byte` with the `bits_per_byte` aggregation, `-Σll / Σbytes / ln 2` [source: `api/metrics.py`]. Rolling windows are disjoint, `context_len=1` [source]. | Same machinery for our tier-1 bpb tasks. | Yes. wikitext is English, though: out of domain and not useful. |
| **HumanEval** (openai/openai_humaneval) | 164 problems, test split, 84 KB parquet [primary: HF API; verified here: 164 rows] | `gogeta_humaneval_exec` (greedy pass@1), `_n20` (pass@k), `gogeta_humaneval_sol_bpb` (likelihood). | Execution: floors. Solution bpb: yes. |
| **MBPP** (google-research-datasets/mbpp) | `full` config: test 500 / train 374 / validation 90 / prompt 10; `sanitized` config also exists [primary: HF API; verified here: split sizes] | `gogeta_mbpp_exec` in OUR native format (statement + 3 asserts + "```python"); `gogeta_mbpp_sol_bpb`. | Execution: near floor (E0: 1/8 on a first-8 subset). Solution bpb: yes. |
| **EvalPlus** (HumanEval+, MBPP+) | PyPI **0.3.1** (2024-10-20, current); last repo commit 2025-10-02 [primary: PyPI JSON, GitHub API]. HF datasets `evalplus/humanevalplus` (164 rows) and `evalplus/mbppplus` (378 rows) [verified here]. Their `test` fields carry thousands of extra inputs and `import numpy` [primary: dataset viewer]. | `gogeta_humaneval_plus_exec`, `gogeta_mbpp_plus_exec` run the real plus tests in our sandbox. The sandbox's base interpreter has numpy 1.26.4 [verified here]. | Floors even harder than the base sets. Report for completeness. |
| **bigcode-evaluation-harness** | No releases. Last commits 2025-07-22 and 2025-07-01, before that 2024-10-31 [primary: GitHub API]: low activity. Has HumanEval(+), MBPP(+), MultiPL-E, DS-1000, APPS, HumanEvalPack. Exec is off by default (`--allow_code_execution`); the README recommends Docker [primary: README]. | Would need a third model integration. | Adds nothing for Python-only at our size. **Not recommended.** |
| **UncheatableEval** (fresh data) | Monthly-ish snapshots on the HF Hub. **Latest `UncheatableEval-2026-10`** (data 2026-10-01..30, updated 2026-11-17): 15 categories × 500 docs incl. `github_python`; one 231 MB parquet [primary: HF API + datasets-server]. Upstream metric [primary: `evaluator.py`]: summed NLL over 4,000-token chunks (BOS per chunk, bf16), reported as bpb/bpc and "compression rate". | `gogeta_uncheatable_py_bpb`: `github_python` only, fetched through the datasets-server filter API (9.6 MB downloaded, 3.0 MB kept), `content` column (the one upstream scores), lm-eval rolling bpb. | **Yes**, and it is the only tokenizer-free comparison to public models on code written after our data. Protocol caveat in §5. |
| **Paloma** (allenai/paloma) | Has a `dolma_100_programing_languages` config (Stack-derived, top-100 languages) [primary: dataset card via search; source: lm-eval task]. **Gated**: needs the AI2 ImpACT licence accepted on the HF account. | lm-eval task exists: `paloma_dolma_100_programing_languages`, bpb. | Would discriminate, but it is mostly non-Python and gated. Deferred: the human must accept the licence. |
| **The Stack v2 held-out** | Gated, and the content sits in Software Heritage buckets [reported; not checked today] | Custom task. | Possible later. UncheatableEval covers the "fresh, held-out code" need now. |
| **CRUXEval** (cruxeval-org/cruxeval) | 800 functions, 187 KB jsonl [primary: HF API]. Generative in lm-eval (`cruxeval_output`, 2-shot, `[ANSWER]` tags, T=0.2, 10 repeats) [source]. | **Re-cast as a likelihood probe**, `gogeta_cruxeval_o_ll`: context `code + "assert f(<input>) == "`, target = the true output literal; greedy exact match + bpb. | Yes as a probe (E0 8.9%; the trivial "always False" baseline is 3.6%) [verified here]. The generative version floors. |
| **LiveCodeBench** | Last commits 2025-06/07 [primary: GitHub API]. Competition problems, instruction prompts. | Generative. | Floors. Not recommended now. |
| **BigCodeBench** | PyPI 0.2.5 (2025-03-31) [primary]. Library-heavy tasks (139 libraries [reported]). | Generative + heavy deps. | Floors. Not recommended. |
| **Zoology** (MQAR) | Active, last commits 2026-03-22 [primary: GitHub API]. Paper arXiv 2312.04927. Pydantic configs, W&B logging, mixers referenced by import path; GPU-oriented, CPU not documented [primary: README]. | Mixers plug in as `ModuleConfig(name="pkg.Class")`. Our GDN/SWA/attn blocks would need thin adapters. | Synthetic and trained from scratch, so it compares architectures, not checkpoints. **Recommended for redesigning K3** (our MQAR check was inconclusive: attn 8.6%). |
| **MAD** (mad-lab) | Paper arXiv 2403.17844. Tasks: in-context / fuzzy / noisy recall, selective copying, compression, memorization. README says CUDA is required [primary: README]. | `make_model_fn(task, vocab_size, max_length)` → nn.Module. | Architecture comparison at tiny scale. Same role as Zoology. Optional. |
| **RULER** | NVIDIA repo. NIAH variants, variable tracking, CWE/FWE, QA; lengths 4K–128K; per-model chat templates [primary: README]. In lm-eval: `ruler` group, `DEFAULT_SEQ_LENGTHS = [4096]`, override via `--metadata max_seq_lengths`; needs an HF tokenizer name [source]. | Possible in principle. | **Floors**: English haystacks and instruction-style queries, and ≥ 4K by default > our 1K context. Use once a longer-context, more general model exists. |
| **State tracking** (parity, S5 word problems) | Our K5 (parity) already runs. S5 permutation composition: "The Illusion of State in State-Space Models" (arXiv 2404.08819) [reported]. Negative-eigenvalue GDN: arXiv 2411.12537 [reported; used by SQ2/SQ6]. | Synthetic, trained from scratch (`experiments/checks/`). | Architecture-level. Keep it in checks, not in the checkpoint suite. |
| **2026 practice for tiny code models** | A web search found no established sub-100M code benchmark. The nearest recent study covers 0.4–10B code models (arXiv 2507.03160) [reported via search]. | n/a | Confirms the tier-1 approach: likelihood metrics + fresh-data bpb. |

## 3. What floors, and why (with numbers)

**HumanEval pass@k reference points.** All from the papers' tables [primary: pdftotext of
arXiv 2107.03374 Table 1 and arXiv 2202.13169's results table]:

| Model | Params | Training | pass@1 | pass@10 | pass@100 |
|---|---|---|---|---|---|
| Codex-12M | 12M | fine-tuned from GPT-3, 100B tokens of GitHub Python | 2.00% | 3.62% | 8.58% |
| Codex-25M | 25M | same | 3.21% | 7.1% | 12.89% |
| GPT-Neo 125M | 125M | the Pile | 0.75% | 1.88% | 2.97% |
| PolyCoder | 160M | 39B code tokens (2.5B Python) | 2.13% | 3.35% | 4.88% |
| CodeParrot | 110M | 26B Python tokens | 3.80% | 6.57% | 12.78% |
| phi-1-small | 350M | ~7B tokens of "textbook" data | 45% | n/a | n/a [reported: abstract] |

What follows from these numbers:

- **Data budget.** We have 130–500× fewer tokens than any of these, at Codex-12M's size. The
  phi-1 row says curated, textbook-like data moves pass@1 a lot. Our corpus is that kind of data,
  so pass@1 above 0 is possible.
- **Resolution** [computed]: one HumanEval problem = 0.61 points. At p = 2%, the binomial SE of
  greedy pass@1 is √(0.02·0.98/164) = 1.1 points, so two variants need about a 3-point gap
  (≈ 5 problems) to separate. n=20 sampled pass@k with the unbiased estimator lowers the
  per-problem variance, but the problem-to-problem variance stays.
- **Contamination inflates it** [verified here, §7]: 20% of HumanEval is lexically in our
  training data.
- **Prediction for E0–E2 greedy HumanEval pass@1 on the full set:** 0–5%, with clean ≤ full
  [predicted]. Tier 2 stays "report, don't decide".

**Also floors** [predicted; reasons]:
- MBPP+/HumanEval+: stricter tests.
- LiveCodeBench, BigCodeBench: competition- or library-level difficulty, instruction prompts.
- RULER, BABILong, LongBench: English, ≥ 4K tokens.
- Generative CRUXEval: 2-shot `[ANSWER]` format.
- Instruct variants: no chat template.

**Our own recall probe floored too** (unexpected: P4 missed, §8). E0 scores 0–0.5% exact
recall, and at d=64 its value bpb (2.60) equals the uniform-chance 2.63 [computed: log2(9000)
bits over 5 bytes]. At longer distances bpb rises above chance, 3.41 at d=960. One reading is
that the model confidently copies a wrong, probably nearby, value; this is not checked. The probe
needs a redesign before it can rank architectures (SQ63).

## 4. Recommended suite

All tasks live in `evals/tasks/` and run through `python -m evals.run_suite`.
Data versions are pinned by snapshot/config name plus the sha256 in the manifest; task YAML
hashes go into every results JSON.

### Tier 1: informative now, run on every checkpoint

| Task | Data (version, split) | Metric | Why |
|---|---|---|---|
| `gogeta_val_bpb` | code-py-v1 val: the 4,000 held-out docs (`meta.json` sha256_16 `32e4b0ede9211f61`) | bits/byte and bits/char, per doc from `<|eos|>`, disjoint 1,024-token windows | In-distribution fit, comparable with train.py's val loss (see §7 for the protocol offset). |
| `gogeta_uncheatable_py_bpb` | UncheatableEval-2026-10 `github_python`, 500 docs (2026-10-01..22), `content` column; manifest sha256 `81fa43e8…` | bits/byte | Fresh real-world code, contamination-resistant and tokenizer-free. Use the same protocol for public baselines (§5). Refresh with each new snapshot, keeping the old one for continuity. |
| `gogeta_humaneval_sol_bpb` (+ `_clean`, 132 problems) | openai_humaneval test (164) | conditional bits/byte of the canonical solution given the prompt | A smooth HumanEval proxy that never floors. |
| `gogeta_mbpp_sol_bpb` (+ `_clean`, 457) | mbpp `full`/test (500), native prompt | conditional bits/byte of the reference code | Same, for MBPP. |
| `gogeta_cruxeval_o_ll` | cruxeval test (800) | greedy exact-match acc + bpb of the output literal | Execution reasoning without generation. |
| `gogeta_heldout_exec` | the 2,555 val docs from `algorithmic_corpus` with asserts + a fenced solution | greedy pass@1, sandboxed | Functional correctness in distribution. Caveat: the val split is only exact-deduplicated against train, so near-duplicates are likely. |

### Tier 2: standard code benchmarks, reported anyway (expected near zero)

| Task | Data | Protocol |
|---|---|---|
| `gogeta_humaneval_exec` (+ `_clean`) | openai_humaneval test (164) | Greedy pass@1. Program = prompt + completion. Stop on `\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint` (lm-eval's list) and `` ``` ``. Test = `test` + `check(entry_point)`. 10 s per program. |
| `gogeta_humaneval_plus_exec` | evalplus/humanevalplus (164) | Same, with EvalPlus tests (numpy). |
| `gogeta_mbpp_exec` (+ `_clean`) | mbpp full/test (500) | Greedy pass@1, our native prompt (statement + the 3 asserts + "```python"), stop at `` ``` ``, pass = all 3 asserts. |
| `gogeta_mbpp_plus_exec` | evalplus/mbppplus (378) | Same prompt; pass = the EvalPlus `test` program. |
| `gogeta_{humaneval,mbpp}_exec_n20` (`--tier 2s`) | as above | n = 20 samples per problem at T = 0.8, top-p 0.95. Unbiased pass@k (Chen et al. 2021, Fig. 3) for k = 1, 5, 10, 20, averaged over problems. |

**pass@k protocol notes:**
- Codex used n = 200 with the best temperature per k: 0.2 for pass@1, 0.8 for pass@100
  [primary: arXiv 2107.03374 §2.3 and Fig. 5].
- We use n = 20 at T = 0.8 to bound cost (no KV cache), and the greedy task for pass@1.
- Always state n, T and top-p next to the numbers. Never compare pass@k across different n or T.
- Report full and `_clean` side by side.

**MBPP prompt choice.** MBPP's original/lm-eval prompt ("You are an expert Python programmer…
[BEGIN]", 3-shot) is English instruction text our model never saw. Our native format is the
`algorithmic_corpus` layout, so for MBPP we measure the model, not its template-following.
Published MBPP numbers use other prompts, so ours are **not directly comparable**. Say so.

### Tier 3: long-context and recall, for when we test scaling

| Task | Now | Later |
|---|---|---|
| `gogeta_recall_d{64,256,512,960}` | Code-native key-value recall (needle assignment, distractor assignments, `assert key == `); 200 items per distance, seeded. **Floors on E0: redesign first (SQ63).** | Extend the distances past 1K once max_seq_len grows. Add multi-query and a no-distractor control. |
| Zoology MQAR (architecture-level) | n/a | Replace K3's homemade MQAR (inconclusive). Run our mixers inside Zoology at its published settings. |
| RULER (lm-eval `ruler`, `--metadata max_seq_lengths`) | Floors (English, ≥ 4K) | Only for a general-text, long-context successor. |
| State tracking (parity K5, S5) | `experiments/checks/` | Unchanged: architecture-level, trained from scratch. |

## 5. Comparing with public models (and K8)

- **Use bits per byte, the same documents, and the same window in characters.** A 1,024-token
  window is ~3.7k characters for us [computed: 3.719 chars/token on val] and a different span for
  a 49k-vocab model. To compare fairly:
  - either cap every model's window at the same number of characters,
  - or score short docs only (`content` excerpts are ≤ 17.3k chars, so they still span several
    of our windows).

  Upstream UncheatableEval uses 4,000-token chunks per model [primary], which favours big-vocab
  models.
- **K8.** The concurrent `experiments/checks/k8_*` check measures SmolLM2-135M bits/char on our
  val set. **This suite doesn't duplicate it.** Once K8 exists, running SmolLM2-135M through
  `gogeta_uncheatable_py_bpb` with lm-eval's stock `hf` model needs no weights download beyond
  K8's.
  - That is a separate run in a separate venv or process, with `--model hf`.
  - `run_suite.py` deliberately only runs `gogeta`.

  It would give the first public-model anchor on fresh code.

## 6. Integration and safety design

- `evals/gogeta_lm.py`, `@register_model("gogeta")` with `class GogetaLM(HFLM)`:
  - builds `GogetaForCausalLM` from a train.py checkpoint, an HF dir or a random preset;
  - calls `HFLM.__init__(pretrained=model, tokenizer=load_tokenizer(), backend="causal",
    max_length=config.max_seq_len, add_bos_token=False, prefix_token_id=<|eos|>)`.

  Every request type uses HFLM's own code. CLI use: `lm-eval --model gogeta --include_path
  evals/tasks --plugins evals.gogeta_lm` (the plugin form is from lm-eval's model guide; we only
  tested the Python API).
- **Shim:** `gogeta/model.py` gained `gdn_neg_eigval` today (concurrent work), but
  `gogeta/hf.py`'s `GogetaConfig` doesn't declare it, so `GogetaForCausalLM` construction raises
  [verified here]. `gogeta_lm.py` fills missing ModelConfig fields with their defaults and records
  them in each results JSON. **`gogeta/hf.py` needs the field** (not my file to edit).
- **Safety:**
  1. Only `gogeta_*` tasks; `include_defaults=False`.
  2. `HF_ALLOW_CODE_EVAL` is removed from the environment.
  3. `confirm_run_unsafe_code=True` is passed only after `sandbox.self_test()` passes (11
     probes).
  4. Programs run via `evals/sandbox.run_tests`, which keeps rewards.py's harness, rlimits,
     privilege drop and isolated interpreter, and adds the following.
     - **New user + net + mount namespaces**, with tmpfs over `/mnt`, `/tmp`, `/dev/shm` and
       `/run`. This hides C:, the WSL interop socket and the shared scratch dirs.
     - **A nested user namespace as uid 65534**, so the child holds no capabilities and can't
       unmount its private view.
       - [verified here] With a single namespace, `umount /mnt` succeeds and `/mnt/c` reappears.
       - With the nested one, `CapEff=0` and `umount2` fails.
     - **Control write** [verified here]: writing to a `/mnt/c` path
       - through `rewards.run_tests`: succeeds;
       - through `evals.sandbox.run_tests`: fails (FileNotFoundError).

## 7. Verified here (2026-11-22, CPU only, ≤ 2 threads, `CUDA_VISIBLE_DEVICES=""`, via `tools/launch_cpu_job.sh`)

| What | Result | Evidence |
|---|---|---|
| Wiring tests | 6/6 OK. Rolling and pair log-likelihoods equal a manual forward pass (incl. a 3,112-token doc over 4 windows). pass@k values. Built-in task names refused. Sandbox self-test. | `evals/test_wiring.py` |
| Smoke: random `tiny_attn`, tiers 1+3, `--limit 8` | All 12 tasks ran in 61.0 s. Random-weight numbers are meaningless (val bpb 3.62, heldout pass@1 0). | `evals/results/smoke-random-tiny_attn-tier1+3-20261122-173426.json` |
| Smoke: random `tiny_attn`, tiers 2+2s, `--limit 1` | All 8 exec tasks ran through the sandbox in 222.9 s (46 generations = 6 greedy + 2 × 20 sampled); pass 0 as expected. | `evals/results/smoke-random-tiny_attn-tier2+2s-20261122-173751.json` |
| E0 val, all 4,000 docs | lm-eval **0.2208 bits/char** (0.2207 bits/byte) vs train.py 0.2351. | `evals/results/check-E0-s0-valbpb-full-tier1-20261122-171816.json` (442 s) |
| Why the 6.1% gap | train.py's protocol re-run on CPU in fp32 gives val loss **0.606254 vs 0.606341** recorded (Δ 0.014%), so the wiring is right. The gap is protocol: packed 1,024-token windows start mid-document with no context. Loss over the first 64 positions of a window is 1.060 vs 0.576 after; excluding `<|eos|>` targets changes almost nothing (0.6072). | `evals/checks/packed_vs_doc.py` → `evals/results/check-packed-vs-doc-E0-20261122-122059-s0.json` |
| E0 likelihood probes (full sets) | HumanEval solution bpb 0.5106 (clean 132: 0.5174) · MBPP 0.6398 (clean 457: 0.6544) · CRUXEval-O acc 8.9%, bpb 1.92 · recall acc 0.5/0/0/0% at d = 64/256/512/960, bpb 2.60/2.79/3.29/3.41 | `evals/results/check-E0-s0-tier1ll-tier3-tier1-20261122-172441.json` (186 s) |
| CRUXEval-O breakdown | acc by output type: bool 26.5% (n = 49; below the 59% majority "False"), int 10.3%, list 9.6%, str 7.0%, other 3.5%. "Always False" would score 3.6% overall. | per-sample log in `~/fusiondance-data/eval-runs/` |
| E0 exec path, first 8 problems each | `gogeta_heldout_exec` **7/8**; `gogeta_mbpp_exec` **1/8** (88 s). Generations inspected: genuine solutions (e.g. `zip(*matrix)` transpose, Kadane). **Subset, not a result.** | `evals/results/check-E0-s0-exec-limit8-tier1-20261122-172825.json` |
| Contamination screen | MBPP: **43/500** statements verbatim in train, 0 assert lines. HumanEval: **43/164** with ≥ 2 long lines matched, **32/164 with ≥ 50%** (e.g. HumanEval/78 13/13 lines). 2,234 of 987,154 train docs contain a match. 28 s. | `evals/contamination.py` → `evals/results/contamination-2026-11-22.json`; flags in `evals/tasks/contaminated_ids.json` |
| Sandbox gap in rewards.py | As `nobody`, a child can write `/mnt/c/...` (drvfs shows 777). | §6 controls; `python -m evals.sandbox` |
| Env | `~/.venvs/gogeta-eval`: torch 2.14.1+cu130 kept after lm-eval install (checked). Pins in `requirements-eval.txt`. | |

Downloads: UncheatableEval slice 9.6 MB; HF parquet files for HumanEval, MBPP, HumanEval+,
MBPP+ and CRUXEval ≈ 4.3 MB. The cache is now 26 MB including Arrow copies. No model weights.

## 8. Predictions (verbatim, written before running) and scorecard

P1. **lm-eval reproduces our own val metric.** `gogeta_val_bpb` on the E0 checkpoint
(`~/fusiondance-data/checkpoints/E0-20261122-122059-s0/model.pt`, all-attention, so CPU-safe),
all 4,000 val docs, gives bits/char within ±3% of train.py's `val_bits_per_char` = 0.2351.
Expected slightly different, not equal: lm-eval scores each doc from an `<|eos|>` prefix in
disjoint 1,024-token windows and does not score the trailing `<|eos|>`; train.py scores a packed
token stream. Pass: |Δ| ≤ 3%.

P2. **Contamination screen.** MBPP's algorithmic-corpus look-alikes make leakage plausible:
I expect ≥ 5% of MBPP test problems (≥ 25 of 500) to be flagged by the line-match screen
(`evals/contamination.py`), and fewer for HumanEval (≤ 5%, its format is not the corpus's).
No pass/fail; it decides whether MBPP numbers need a "possibly contaminated" label.

Added 17:25, before the second E0 job (tier-1 loglikelihood tasks + tier-3 recall, full sets):

P3. **Contamination shows up in the numbers.** E0's HumanEval reference-solution bpb on all 164
problems is lower (better) than on the 132-problem `_clean` subset (the 32 flagged problems are
easier for it because it has seen their lines). Same direction for MBPP (500 vs 457), smaller gap.

P4. **Recall works at every distance for an all-attention model.** E0 `gogeta_recall_d64`
greedy accuracy ≥ 40%, and d960 within 20 points of d64 (full attention has no window).

P5. **CRUXEval-O as loglikelihood is above zero.** E0 greedy exact-match accuracy 5–25%.

Added 17:29, before the E0 generation job (first 8 problems each; wiring of the exec path with a
trained model, not a result):

P6. E0 `gogeta_heldout_exec` (in-distribution, greedy): at least 1 of the first 8 passes.
P7. E0 `gogeta_mbpp_exec` (native prompt, greedy): 0–3 of the first 8 pass.

(The `packed_vs_doc.py` prediction, "reproduces train.py within 0.5%, and the first-64-position
loss is clearly higher", is in that script's docstring, written before it ran.)

**Scorecard:**

| # | Outcome |
|---|---|
| P1 | ✗ **MISSED** (−6.1%, outside ±3%). The cause was found and verified: packing cold starts. The wiring is exact (packed re-run Δ 0.014%). |
| P2 | MBPP ✓ (43/500 = 8.6%). HumanEval ✗ (26% flagged, 20% heavily; far above 5%). |
| P3 | Direction ✓ for both. "MBPP gap smaller" ✗: MBPP 2.3% vs HumanEval 1.3%. |
| P4 | ✗ **MISSED badly**: ~0% at every distance. |
| P5 | ✓ 8.9%. |
| P6 | ✓ 7/8. |
| P7 | ✓ 1/8. |
| packed_vs_doc | ✓ both parts. |

## 9. Open items and blockers

1. **`gogeta/hf.py` lacks `gdn_neg_eigval`.** It breaks `GogetaForCausalLM`, and
   post_training's tests too. The eval shim covers it for now.
2. **GDN checkpoints (E1, E2, …) need CUDA + `flash-linear-attention==0.6.0`** in gogeta-eval.
   Not installed; GPU not used here.
3. **Generation cost.** There is no KV cache, and mixed left-padding runs row-group by
   row-group. E0 took ~5.5 s per problem on CPU (88 s / 16). Full tier 1 exec (2,555) plus
   tier 2 (≈1,700 programs greedy, ×20 for 2s) should run on the GPU slot when free.
4. **rewards.py's /mnt/c exposure** also affects post_training (GRPO rewards). Proposed: upstream
   the mount-namespace step into rewards.py.
5. **Paloma (gated)**: needs the human to accept the licence and set `HF_TOKEN`.
6. **Recall probe redesign** (SQ63) before using tier 3.

## 10. Sources (checked 2026-11-22 unless marked)

- lm-eval: https://pypi.org/pypi/lm-eval/json ·
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md ·
  installed source `lm_eval/models/huggingface.py`, `api/metrics.py`, `api/task.py`,
  `utils.py:get_rolling_token_windows`, `tasks/{humaneval,mbpp,paloma,ruler,cruxeval}/`
- EvalPlus: https://pypi.org/pypi/evalplus/json ·
  https://api.github.com/repos/evalplus/evalplus/commits ·
  https://huggingface.co/datasets/evalplus/humanevalplus · https://huggingface.co/datasets/evalplus/mbppplus
- bigcode-evaluation-harness: https://github.com/bigcode-project/bigcode-evaluation-harness ·
  https://api.github.com/repos/bigcode-project/bigcode-evaluation-harness/commits
- HumanEval / Codex: arXiv 2107.03374 (Table 1, §2.3, Fig. 3, Fig. 5; PDF text extracted here)
- PolyCoder / CodeParrot: arXiv 2202.13169 (results table; PDF text extracted here)
- phi-1: arXiv 2306.11644 (abstract page)
- OpenCoder: arXiv 2411.04905 (annealing-data composition; decontamination described for SFT data, §4.4)
- MBPP: https://huggingface.co/datasets/google-research-datasets/mbpp · HumanEval: https://huggingface.co/datasets/openai/openai_humaneval
- UncheatableEval: https://github.com/Jellyfish042/uncheatable_eval (README, `evaluator.py`) ·
  https://huggingface.co/datasets/Jellyfish042/UncheatableEval-2026-10 · HF datasets-server `/statistics`, `/filter`
- Paloma: arXiv 2312.10523 · https://huggingface.co/datasets/allenai/paloma (gated)
- CRUXEval: arXiv 2401.03065 · https://huggingface.co/datasets/cruxeval-org/cruxeval
- Zoology: https://github.com/HazyResearch/zoology (+ commits API), arXiv 2312.04927
- MAD: https://github.com/athms/mad-lab, arXiv 2403.17844
- RULER: https://github.com/NVIDIA/RULER
- LiveCodeBench commits API; BigCodeBench PyPI JSON
- Small-model study: arXiv 2507.03160 (via search result only)
