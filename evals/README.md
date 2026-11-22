# evals/: the Gogeta-0 benchmark suite (lm-evaluation-harness)

Why these benchmarks, what floors at our size, and the sources: [`notes/benchmark-suite.md`](../notes/benchmark-suite.md).
Status (2026-11-22): wiring verified on CPU with a random tiny model and with the E0 checkpoint
(see "Verified here"). No GPU run yet; no GDN checkpoint through it yet.

## Layout

| Path | What |
|---|---|
| `gogeta_lm.py` | lm-eval model `gogeta`: an `HFLM` subclass that builds `gogeta.hf.GogetaForCausalLM` from a train.py checkpoint (or an HF dir, or a random preset) and passes the model object to HFLM |
| `tasks/tier{1,2,3}/*.yaml`, `tasks/gogeta_utils.py` | Our task configs (loaded with `TaskManager(include_path="evals/tasks", include_defaults=False)`) |
| `tasks/contaminated_ids.json` | HumanEval/MBPP problems flagged by `contamination.py`; the `*_clean` tasks drop them |
| `run_suite.py` | The runner: tiers → lm-eval → `evals/results/<run>.json` |
| `sandbox.py` | The only place generated code runs (see "Code execution") |
| `fetch_data.py` | Downloads the UncheatableEval GitHub-Python slice into the eval cache |
| `contamination.py` | Line-match screen of HumanEval/MBPP against `docs_train.jsonl` |
| `test_wiring.py` | Harness numbers == hand-computed log-probs; pass@k; guards; sandbox self-test |
| `results/` | Results JSON (small; per-sample logs go to `~/fusiondance-data/eval-runs/`) |

## Environment (WSL)

A separate venv, `~/.venvs/gogeta-eval` (never install lm-eval into `gogeta` or `gogeta-rl`):

```bash
python3 -m venv ~/.venvs/gogeta-eval
~/.venvs/gogeta-eval/bin/pip install --upgrade pip
~/.venvs/gogeta-eval/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # FIRST
~/.venvs/gogeta-eval/bin/pip install -r requirements-eval.txt
~/.venvs/gogeta-eval/bin/python -c "import torch, lm_eval; print(torch.__version__, lm_eval.__version__)"   # 2.14.1+cu130 0.4.13
```
For checkpoints with `gdn` blocks (CUDA only) also `pip install flash-linear-attention==0.6.0` (not done yet).

## Data

Caches live under `~/fusiondance-data/eval-cache` (`$GOGETA_EVAL_CACHE`); `run_suite.py` sets
`HF_HOME`/`HF_DATASETS_CACHE` there. Nothing is downloaded into the repo, and no model weights are
ever downloaded.

| Task data | Source | Size |
|---|---|---|
| in-distribution val | `~/fusiondance-data/datasets/code-py-v1/docs_val.jsonl` (tools/prepare_data.py) | local, 4.2 MB |
| fresh GitHub Python | `python -m evals.fetch_data uncheatable` → `uncheatable/2026-10/github_python.jsonl` | 9.6 MB downloaded, 3.0 MB kept |
| HumanEval / MBPP / CRUXEval | fetched by `datasets` on first use | 84 KB / 116 KB / 187 KB |
| HumanEval+ / MBPP+ (tier 2 only) | `evalplus/humanevalplus`, `evalplus/mbppplus` | 2.9 MB / 1.1 MB parquet |

## Running

```bash
cd /mnt/c/Users/mreca/Desktop/Gogeta-0
P=~/.venvs/gogeta-eval/bin/python
# wiring tests (CPU, ~5 s)
CUDA_VISIBLE_DEVICES= $P -m unittest evals.test_wiring -v
# CPU smoke (random weights; proves wiring, not quality), through the CPU-slot launcher
bash tools/launch_cpu_job.sh --name eval-smoke -- $P -m evals.run_suite --random-init tiny_attn --tier 1,3 --limit 8
# a real checkpoint, tier 1 (GPU jobs only when the GPU slot is free: bash experiments/launch/status.sh)
$P -m evals.run_suite --checkpoint ~/fusiondance-data/checkpoints/<run>/model.pt --tier 1 --device cuda
```

Options: `--tier 1,2,2s,3` · `--tasks a,b` (only `gogeta_*`) · `--limit N` (testing only; recorded
in the JSON) · `--loops R` (looped configs) · `--no-exec` · `--log-samples` · `--threads` (default 2).

Tiers (details and rationale in the notes):

| Tier | Tasks | Needs |
|---|---|---|
| 1 | `gogeta_val_bpb`, `gogeta_uncheatable_py_bpb`, `gogeta_humaneval_sol_bpb[_clean]`, `gogeta_mbpp_sol_bpb[_clean]`, `gogeta_cruxeval_o_ll`, `gogeta_heldout_exec` | loglikelihood only, except `heldout_exec` (generation + sandbox) |
| 2 | `gogeta_humaneval_exec[_clean]`, `gogeta_humaneval_plus_exec`, `gogeta_mbpp_exec[_clean]`, `gogeta_mbpp_plus_exec` (greedy pass@1) | generation + sandbox |
| 2s | `gogeta_humaneval_exec_n20`, `gogeta_mbpp_exec_n20` (n=20, T=0.8, top-p 0.95, unbiased pass@1/5/10/20) | generation + sandbox, ~20× tier 2 |
| 3 | `gogeta_recall_d{64,256,512,960}` (code-native key-value recall) | loglikelihood only |

Generation has **no KV cache** (gogeta/hf.py), so generation tasks are slow, on CPU very slow.

## Code execution (safety)

- lm-eval's own `humaneval`/`mbpp`/`cruxeval` tasks run model code **in-process** via HF
  `evaluate`'s `code_eval` (needs `HF_ALLOW_CODE_EVAL=1`; `humaneval/utils.py` even executes a test
  program at import). `run_suite.py` never loads lm-eval's built-in tasks (`include_defaults=False`,
  only `gogeta_*` names), removes `HF_ALLOW_CODE_EVAL` from the environment, and only passes
  `confirm_run_unsafe_code=True` for our own tasks **after** `sandbox.self_test()` passes.
- Our exec tasks run each program through `evals/sandbox.run_tests`, which reuses
  `post_training/rewards.py`'s harness and limits and adds a private filesystem view (below).
- **Gap found in `post_training/rewards.py` (verified here):** on WSL the Windows drive is mounted
  at `/mnt/c` with every file shown as mode 777, so its child, even as `nobody`, **can write to
  `/mnt/c`**, i.e. this repo and the whole Windows profile (a control write through
  `rewards.run_tests` succeeded). `evals/sandbox.py` therefore runs the child in new user +
  network + mount namespaces, mounts tmpfs over `/mnt`, `/tmp`, `/dev/shm`, `/run`, and then drops
  into a nested user namespace as uid 65534 so the private view cannot be unmounted (with a
  single namespace the child is ns-root and `umount /mnt` works: verified). Same control write: blocked.
- `python -m evals.sandbox` prints the self-test: pass/fail scoring, repo invisible and
  unwritable, home unwritable, `/mnt/c` hidden, unmount blocked, network blocked, timeout, memory
  cap, runs as nobody. All must be true or exec tasks are refused (exit code 2).
- Still a process sandbox, not a security boundary: the child can read world-readable system files.
  For adversarial settings use a container / nsjail.

## Verified here (CPU, 2 threads, 2026-11-22)

- `test_wiring.py`: 6/6 OK. Rolling loglikelihood (short doc, and a 3,112-token doc over 4
  windows) and pair loglikelihood equal a hand-written forward pass (≤ 1e-3 nats / rel 1e-4).
- Smoke: random `tiny_attn` (697K-param family, 1,024 context), tiers 1+3, `--limit 8`: all 12
  tasks ran in 61.0 s; result `results/smoke-random-tiny_attn-tier1+3-20261122-173426.json`.
- Smoke, tiers 2+2s, `--limit 1`: all 8 exec tasks ran (46 generations: 6 greedy + 2 × 20 sampled)
  in 222.9 s; `results/smoke-random-tiny_attn-tier2+2s-20261122-173751.json`.
- E0 checkpoint numbers: see `notes/benchmark-suite.md` §7.

## Caveats

- `--limit` results are subsets; never compare them with full runs.
- bpb across models is only comparable under the same protocol (document set, window in
  characters, scoring of the first tokens). Our window is 1,024 of OUR tokens (~3.7k chars).
- Each doc is scored from an `<|eos|>` prefix (what pretraining sees); lm-eval moves a prompt's
  trailing whitespace into the scored continuation, and our bpb denominators count it.
- Contamination: see `tasks/contaminated_ids.json`; report `_clean` variants next to full ones.
- `gogeta_lm.py` contains a temporary shim: if `gogeta/model.py`'s `ModelConfig` has a field that
  `gogeta/hf.py`'s `GogetaConfig` lacks (today: `gdn_neg_eigval`), it is filled with the
  ModelConfig default and listed in the results' `model.config_fields_filled_with_defaults`.
  Remove it once `gogeta/hf.py` declares the field.
