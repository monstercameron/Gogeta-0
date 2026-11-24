# Post-training with Hugging Face TRL: setup guide (prepared, NOT activated)

Status (2026-11-23, v0.5.55): the wrapper, reward sandbox and smoke tests exist and pass **on CPU with a
tiny random attn/swa model**. Nothing here has been run on the GPU, on a trained checkpoint, on
GDN blocks, or on real data. Real SFT / GRPO / DPO runs are blocked on the items in
[Blockers](#8-prerequisites-and-blockers).

Labels used below (AGENTS.md §0):
- **[verified here]**: run on this machine, the output is quoted in section 5.
- **[source]**: read in the installed package source (`~/.venvs/gogeta-rl/lib/python3.12/site-packages/...`), not executed separately.
- **[docs]**: from the TRL docs (main branch, which can be ahead of the 1.18.2 release). Not checked here.
- **[reported]**: someone else's claim (paper, coordinator). Not checked here.

## 1. Versions

| Package | Version | How we know |
|---|---|---|
| trl | 1.18.2 (latest on PyPI at install time) | [verified here] `pip install trl` + PyPI JSON |
| transformers | 5.24.0 | [verified here] pulled by trl (`transformers>=4.56.2` per PyPI metadata) |
| accelerate / datasets | 1.16.0 / 5.2.0 | [verified here] |
| torch | 2.14.1+cu130 | [verified here] installed first from the cu130 index, `torch.__version__` checked after trl |
| Python | 3.12.3 (Ubuntu 24.04 aarch64, WSL2) | [verified here] |

The full pinned set is `requirements-rl.txt`. vLLM is not installed (see §8). Note: the main
training venv `~/.venvs/gogeta` already contains transformers 5.24.0 (pulled in by
flash-linear-attention; seen in its `pip freeze`), so a later merge of the two venvs may be
cheap, but check with `pip install --dry-run trl==1.18.2` first. Nothing was installed there.

## 2. Environment

```bash
# in WSL, never inside ~/.venvs/gogeta (the training venv)
python3 -m venv ~/.venvs/gogeta-rl
~/.venvs/gogeta-rl/bin/pip install --upgrade pip
~/.venvs/gogeta-rl/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # FIRST
~/.venvs/gogeta-rl/bin/pip install -r requirements-rl.txt
~/.venvs/gogeta-rl/bin/python -c "import torch, trl, transformers; print(torch.__version__, trl.__version__, transformers.__version__)"
# expect: 2.14.1+cu130 1.18.2 5.24.0
# for "gdn" blocks also: pip install flash-linear-attention==0.6.0
#   (a --dry-run says it only adds einops, fla-core, fla; not installed or tested in gogeta-rl)
```

Run everything from the repo root (`/mnt/c/Users/mreca/Desktop/Gogeta-0`) so `gogeta` and
`post_training` import without `pip install -e .`.

## 3. What TRL needs from a custom model, and how `gogeta/hf.py` provides it

| Requirement | Where it comes from | Our answer |
|---|---|---|
| A `PreTrainedModel` instance (or a path loadable with `from_pretrained`); "only causal language models" | [docs] GRPO/DPO trainer pages; [source] `GRPOTrainer.__init__` | `GogetaForCausalLM(PreTrainedModel, GenerationMixin)` |
| `PreTrainedConfig` (in transformers 5 configs are dataclasses: fields are class annotations, `@dataclass` is applied by `__init_subclass__`) | [source] `transformers/configuration_utils.py` | `GogetaConfig`: every `ModelConfig` field + `eos_token_id=0`, `pad_token_id=1`, `use_cache=False`, `tie_word_embeddings` |
| `model.base_model(...)` returning `.last_hidden_state`, and `get_output_embeddings()` | [source] `trl/trainer/utils.py: add_fused_lm_head`. GRPO, DPO, KTO **and SFT** patch the forward so that `model(..., labels=..., fused_lm_head=True)` returns per-token log-probs computed from the backbone's hidden states and the head weight, without full logits | `base_model_prefix="model"`; `.model` is a `FusionDanceBackbone` (FusionDance subclass whose forward stops at `norm_out`); `get_output_embeddings()` = `model.head` |
| `generate()` via `GenerationMixin` (used when `use_vllm=False`, the default) | [source] `GRPOTrainer._generate_single_turn` calls `unwrapped_model.generate(input_ids, attention_mask, generation_config=...)` | full-sequence recompute every token; `prepare_inputs_for_generation` returns the whole sequence; `_supports_default_dynamic_cache()` is False so no `Cache` is built even though TRL's `GenerationConfig` leaves `use_cache=True` |
| Left padding for prompts | [source] GRPO pads prompts left; [docs] DPO "The padding side must be set to left" | wrapper honours left padding exactly (groups rows by pad count, runs each un-padded) |
| A tokenizer with `pad_token` (TRL falls back to `eos_token`) | [source] GRPO/SFT `__init__` | `load_tokenizer()`: `PreTrainedTokenizerFast`, eos `<\|eos\|>`=0, pad `<\|pad\|>`=1, left padding, no bos, **no chat template** (so use standard text formats, not conversational) |
| Reference model by path when `beta > 0` (GRPO) or by default (DPO) | [source] both call `create_model_from_path(model.config._name_or_path)`, which reads `config.auto_map` and uses `AutoModelForCausalLM` | `register_auto_classes()` runs on `import gogeta.hf`; `GogetaConfig.auto_map = {}`; load the policy with `from_pretrained(dir)` so `_name_or_path` is set |
| Gradient/activation checkpointing | [source] `_BaseConfig.__post_init__`: TRL turns it ON by default (`activation_checkpointing=True` on transformers >= 5.23) | not supported (`supports_gradient_checkpointing=False`); pass `activation_checkpointing=False` |
| bf16 | [source] TRL configs default `bf16=True` unless `fp16` | smoke tests use `bf16=False` on CPU; bf16 on GPU untested |

Other wrapper facts:
- **v0.5.0 to v0.5.23 broke the wrapper; fixed in v0.5.24 (C80).** `GogetaConfig` lacked the eight assembly fields that `ModelConfig` gained in v0.5.0, so no model could be built (the guard failed loudly; no training or evaluation result used the wrapper). `FusionDanceBackbone.hidden()` now delegates to `FusionDance.forward(..., return_hidden=True)` instead of a hand-written mirror, so the assembly switches (loop_embed, contractive injection, adapters, ternary scope) reach the wrapper. Test: `TestAssembledConfig` (exact logit equality).
- Non-persistent buffers: transformers 5 re-creates them **empty** in `from_pretrained`
  [source]. Our RoPE tables are non-persistent, so `GogetaForCausalLM._init_weights` recomputes
  them. [verified here]: with that branch disabled, a save/load round trip changes the
  logits by 0.157 (max abs); with it, by 0.0.
- `loops`: fixed at `config.loops` (or `loops=` on forward/generate). In train mode
  FusionDance's truncated BPTT applies (`bptt_loops`). TRL never passes `loops`; set it in the config.
- Masks with holes (packing / `padding_free=True`) raise `ValueError`. Keep `packing=False`.
- `max_seq_len` (RoPE table, 1024 in our configs) bounds prompt + completion.

## 4. Wrapping a trained checkpoint

```python
from gogeta.hf import from_gogeta_checkpoint, load_tokenizer, to_gogeta_checkpoint
model = from_gogeta_checkpoint("~/fusiondance-data/runs/<run>/ckpt/model.pt", loops=4)  # {"model", "model_config"}
model.save_pretrained("~/fusiondance-data/hf/<run>")      # so TRL can rebuild the reference model by path
tok = load_tokenizer()
# ... train with TRL ..., then back to our format:
import torch; torch.save(to_gogeta_checkpoint(model), "model_rl.pt")
```
`from_gogeta_checkpoint` strips a `_orig_mod.` (torch.compile) prefix and loads with `strict=True`.
Keep HF exports under `~/fusiondance-data/` (not in the repo).

## 5. Tests and smoke runs (how to run, and what they printed here)

All CPU-only (`CUDA_VISIBLE_DEVICES=` is set inside the scripts), from the repo root, in WSL:

```bash
P=~/.venvs/gogeta-rl/bin/python
CUDA_VISIBLE_DEVICES= $P -m unittest post_training.test_hf post_training.test_rewards -v   # 48 tests at v0.5.48 (test_hf 11 at v0.5.24, test_rewards 37 at v0.5.48)
CUDA_VISIBLE_DEVICES= $P post_training/grpo_smoke.py       # ~6-18 s
CUDA_VISIBLE_DEVICES= $P post_training/sft_dpo_smoke.py    # ~5 s
```

[verified here] results on 2026-11-22:
- `test_hf` (10 tests) OK: wrapper logits equal FusionDance logits exactly (max diff 0.0); loss
  equals FusionDance's shifted loss; left-padded row equals the un-padded run (< 1e-5); TRL's
  fused-LM-head log-probs equal `log_softmax` of our logits with left padding (< 1e-4);
  greedy `generate` equals a manual greedy loop; `save_pretrained`/`from_pretrained` round
  trip is exact and keeps the head/embedding tie; tokenizer decode(encode(s)) == s.
- v0.5.24 (2026-11-23): `test_hf` 11/11 OK (adds `TestAssembledConfig`) and `evals/test_wiring.py` 6/6 OK; saved output `experiments/checks/results/test-runs/2026-11-23-C80.txt`.
- `test_rewards` (17 tests at v0.4.104; 30 since v0.5.11; 34 since v0.5.40; 37 since v0.5.48, three of them descendant tests) OK: pass, fail, partial credit, infinite loop (timeout), sleep
  (wall-clock timeout), syntax error, runtime error, `sys.exit`, forged result line, 2 GB
  allocation under a 256 MB cap (MemoryError), network blocked, runs as `nobody` (cannot list
  `/root`), TRL-shaped call. `sandbox_info()` here: `runs_as_nobody=True, network_isolated=True,
  child_python=/usr/bin/python3.12`. Extra one-off controls: the same socket test with
  `network="on"` connects (so "blocked" is meaningful); a fork bomb hits RLIMIT_NPROC
  (`BlockingIOError`), is killed at the timeout, and leaves 0 processes behind.
- `grpo_smoke.py`: PASS. 697,024 params, 3 steps, 4 prompts, 4 generations/prompt, 16-token
  completions, beta=0.04 (reference model built by path), toy reward + code reward (weight 0):

  ```
  step=1  loss=-2.086e-07  reward=0.08757  reward_std=0.02423  kl=0         completions/mean_length=16
  step=2  loss=5.296e-05   reward=0.1141   reward_std=0.1221   kl=0.001324  completions/mean_length=16
  step=3  loss=7.899e-05   reward=0.07685  reward_std=0.01556  kl=0.00197   completions/mean_length=16
  code_reward calls (completions scored): 24
  parameter tensors changed: 29/29
  SMOKE: PASS
  ```
- `sft_dpo_smoke.py`: SFT PASS (loss 8.947 -> 8.427 -> 8.210 on one repeated example; TRL's
  tokenization ends with `<|eos|>` and sets the 18 prompt labels to -100), DPO PASS (loss
  0.6931 -> 0.5735 -> 0.5299, reward margin 0 -> 0.256 -> 0.358).

These prove the plumbing only. Nothing about quality, speed or GPU behaviour.

## 6. Recipes (outlines, not run on real data)

### 6.1 SFT
- Data: standard (non-conversational) prompt-completion rows, e.g. prompt = problem statement
  + "\n```python\n", completion = solution + "```". TRL appends the eos token and computes the
  loss on the completion only by default for that format ([source] `completion_only_loss=None`
  → completion-only for prompt-completion data; [verified here] in the SFT smoke).
- Or pre-tokenize ourselves into `input_ids` (+ optional `labels`) [source]; useful to reuse
  `tools/prepare_data.py` outputs.
- Config sketch: `SFTConfig(max_length=1024, packing=False, padding_free=False,
  activation_checkpointing=False, bf16=True, learning_rate=..., per_device_train_batch_size=...)`.
  SFT's default LR is 2e-5 [source]; ours should come from a small sweep.
- Pass `model=from_gogeta_checkpoint(...)`, `processing_class=load_tokenizer()`.
- Do not use `packing=True`: with the default `bfd` strategy TRL turns on padding-free
  batches [source/docs], which our wrapper rejects.

### 6.2 GRPO with assert rewards
- Dataset rows: `{"prompt": <problem>\n```python\n, "tests": [<assert lines>]}`. Build from the
  OpenCoder docs that contain a fenced solution plus `assert` lines
  (`post_training.rewards.extract_asserts`); drop the solution from the prompt; keep held-out
  problems for eval; deduplicate against the SFT data.
- Reward function [docs + source]: called with keyword args `prompts`, `completions`,
  `completion_ids`, `trainer_state`, `log_extra`, `log_metric` and every dataset column except
  `prompt`; returns one float per completion (or `None` to skip). Multiple functions are summed,
  weighted by `reward_weights`. Async functions are allowed. `remove_unused_columns` defaults to
  False in GRPOConfig [source], so `tests` reaches the reward.
  Ours: `post_training.rewards.code_exec_reward(completions, tests, partial=True, timeout=5,
  mem_mb=1024)`; use `partial=False` for strict pass/fail.
- Key GRPOConfig defaults in trl 1.18.2 [source]: `num_generations=8`,
  `max_completion_length=512`, `temperature=1.0`, `top_p=1.0`, `top_k=0`, `beta=0.0` (no
  reference model), `loss_type="dapo"` (also `grpo`, `dr_grpo`, `bnpo`, `cispo`, `sapo`, `luspo`,
  `vespo`), `scale_rewards="group"`, `epsilon=0.2`, `num_iterations=1`, `learning_rate=1e-6`,
  `use_vllm=False` (then `model.generate` is used), `generation_batch_size=None`
  (= per-device batch x processes x steps_per_generation). The effective batch must be divisible
  by `num_generations`.
- Generation alternatives [source]: `use_vllm=True` (`vllm_mode="colocate"` or `"server"`),
  `use_transformers_continuous_batching=True` (calls `model.generate_batch`; whether it works
  without a KV cache is unverified, probably not), or a custom `rollout_func(prompts, trainer) -> {"prompt_ids", "completion_ids", "logprobs", ...}`
  (experimental). `rollout_func` is the natural place for our own fast decode path.
- Start small: `num_generations=8`, `max_completion_length=256`, `beta=0.0`, a few hundred
  problems that the SFT model solves sometimes (all-0 or all-1 groups give zero advantage; watch
  `frac_reward_zero_std`).

### 6.3 DPO (maybe later)
- Rows `{"prompt", "chosen", "rejected"}` (standard format) [docs]. Defaults [source]:
  `beta=0.1`, `loss_type=["sigmoid"]`, `precompute_ref_log_probs=False`, `max_length=1024`.
  Reference model = the initial policy, rebuilt by path (needs the Auto registration).
- Preference pairs for code can come from the GRPO sandbox: two samples of the same problem, one
  passing all asserts and one failing.

## 7. Sandboxing (what `post_training/rewards.py` does, honestly)

Isolated / limited on Linux [verified here, test_rewards.py]: own process group killed on
timeout; wall-clock timeout + RLIMIT_CPU; RLIMIT_AS (memory), RLIMIT_FSIZE (16 MB),
no core files, 64 open files; when run as root (the WSL default user here **is root**) the child
runs as `nobody` with RLIMIT_NPROC=256; base interpreter in isolated mode (`python3.12 -I`,
stdlib only, no venv packages), fresh temp cwd, env reduced to PATH/LANG; network removed with
`unshare -rn` (new user + net namespace) when available, else it runs with network and says so
(`network="require_off"` makes that an error).

**Status at v0.5.11 (SQ373 and SQ61; committed devlogs `experiments/devlogs/SQ373.md`, `SQ61.md`).**

- **Reward forgery: fixed for the probe's cases (SQ373, C83).** The child is now a grader process that forks a worker. The worker runs the solution and the tests inside the same namespaced sandbox and rlimits. The grader binds every function it uses before the solution runs, each test runs in a fresh shallow copy of the solution's namespace with pristine builtins, the stdin nonce is dropped in the worker, and the result comes back over a pipe that only the grader turns into the final line. Every forgery probe (exec/compile no-op, AssertionError swallowing, `__main__` and `sys.modules` tampering, `os._exit`/`sys.exit`, printed forged result lines, trace hooks, background threads) now scores 0 on a failing test. Honest pass, honest fail and partial credit are unchanged. `python -m unittest post_training.test_rewards`: 34 tests OK at v0.5.40 (the 30 of v0.5.11 plus four added with SQ411); 37 tests OK at v0.5.48 (saved output `experiments/checks/results/test-runs/2026-11-23-test-rewards-fixed.txt`).
- **SQ411 closes the `__eq__` route (v0.5.40; DEVLOG v0.5.40).** The grader AST-rewrites each test before the fork, so every non-identity comparison and every assert/`not` truthiness goes through a strict check over exact builtin value types (or containers of them). The probe's 14 forgery arms go from 1/1 to 0/1, and honest solutions that return custom classes now fail closed, by design.
- **Residual forgery routes (not closed).** Solution code can still imitate the worker's side of the pipe (find the fds via `/proc/self/fd` and write a forged result), defeat a test through solution-defined helpers the test calls, use `.equals`-style methods, or pass any/all over forged iterables (C83). Closing these needs a process the solution cannot touch. Not attempted.
- **Filesystem and network (SQ61, C105/C106).** `run_tests(..., fs_isolation="auto"|"require"|"off")` builds a private mount namespace (tmpfs over /mnt, /tmp, /dev/shm, /run, /var/tmp; nested uid 65534) with no network. `code_exec_reward`, the GRPO path, defaults to `"require"` and fails closed: it refuses to score rather than run unisolated. The repo-write control and the fail-closed tests are in `post_training/test_rewards.py`. `evals/sandbox.py` delegates to the same harness and keeps its self-test.
- **Descendant cleanup (SQ61, v0.5.48): fixed.** Each sandbox child is PID 1 of its own PID namespace (`unshare -p -f --kill-child`). On timeout the harness SIGKILLs PID 1, which kills every process in that namespace, including `setsid` descendants. The wait for the pipes is bounded (5 s), and the harness fails closed if anything survives. The probe `experiments/checks/sandbox_kill_cleanup.py` kills a setsid child in 1.0 s with no survivors (saved output `experiments/checks/results/test-runs/2026-11-23-sandbox-kill-cleanup-fixed.txt`). The sandbox self-test passes 17 of 17 checks, including the fail-closed probe `setsid_descendant_killed` (`experiments/checks/results/test-runs/2026-11-23-sandbox-selftest-fixed.txt`).
- **Q14 gate (SQ61, v0.5.55):** before Q14's execution evals, the world-writable-path sweep must pass, together with the 17-check self-test and the kill-cleanup probe. The socket/concurrency probe (`experiments/checks/sandbox_socket_concurrency.py`, PASS) checks that abstract and path Unix sockets are unreachable from sandboxed code (with a positive control) and that 20 concurrent setsid jobs leave no survivors and no mount leaks (evidence on C83 and C106).
- **Still open (SQ61 row):** the resource-budget test, the world-writable-path sweep, and a /proc PID view.
- **Not a security boundary.** Inside the namespace the rest of the root filesystem stays world-readable. With `fs_isolation="off"`, or "auto" on a host without a working namespace, the child can still write to /tmp, /dev/shm and, under WSL, `/mnt/c` (C105, verified 2026-11-22 on the old path). For real RL, wrap the same harness in a container or bubblewrap/nsjail with a read-only root and a tmpfs work dir, and log/inspect high-reward completions. Throughput
[verified here, trivial solution, CPU shared with another job]: 29.5 ms per completion serially,
64 completions in 0.47 s with the default 8 threads.

## 8. Prerequisites and blockers

1. **No incremental decoding.** `generate` recomputes the whole sequence for every new token
   (computed: a 256-token prompt + 512-token completion processes ~262k token positions instead
   of 768 with a cache, about 340x). GRPO needs `num_generations` samples per problem, so
   generation will dominate. Needed: a cached decode path (KV cache for attn/swa, recurrent
   state for GDN, per-loop state for the looped core), exposed either through `generate` or a
   `rollout_func`.
2. **GDN recurrent kernel (resolved for GDN, ledger C2/C110/C111).** The NaN was our call-site
   bug: fla 0.6.0 takes `gk` as its 5th positional argument, so our positional call passed `beta`
   as a decay. With keyword args, `fused_recurrent_gated_delta_rule` matches the fp32 reference
   (chunk prefill + 1-token steps, and the neg-eig mode; `experiments/checks/results/sq_hw_fla_recurrent.json`).
   Still open: the decode path is not yet wired into `generate` (SQ53), and SWA/attention have no
   cache yet. The chunk kernel is what training and the current no-cache
   generation would use. Not verified by this work.
3. **vLLM cannot serve FusionDance** without writing a vLLM model implementation (custom
   architecture, looped core, GDN). Not attempted. Also unknown whether vLLM wheels exist for
   aarch64 + cu130 + sm_121 [unverified].
4. **GDN on CUDA through the wrapper is untested.** All tests use attn/swa on CPU. First GPU
   step: rerun `test_hf.py` with a GDN tiny config on CUDA (when the GPU is free).
5. **No chat template.** Use standard text formats. Conversational datasets would need a template.
6. **Activation checkpointing** must be off (TRL defaults it on). Memory at 1k context should be
   fine for our sizes but is unmeasured.
7. **Ternary (`ternary=True`)** works through the wrapper in principle (BitLinear is an
   nn.Linear), but RL of a ternary model is untested here; the fused LM head uses the head
   weight directly (the head is full precision in our model, so that matches).
8. **Reward sandbox** is a process sandbox, not a security boundary (§7).

## 9. Lessons from FlashNext-RL (reported, not verified here)

Source: `refs/10-qwen/2610.28998 - FlashNext-RL.pdf` (Snowflake AI Research, arXiv 2610.28998,
Oct 2026), RL for Qwen3.8-Flash-Next (36 of 48 layers Gated DeltaNet, sparse attention, MoE,
gated residual), built on prime-rl with vLLM rollouts. What it reports, and what transfers:

- **General-purpose RL stacks were not enough for an unusual architecture.** The paper reports it
  needed model-specific kernels (GDN, sparse attention backward, MoE, gated residual), host
  offload and memory planning, and concludes RL should be tailored to the architecture rather
  than rely only on general-purpose RL infrastructure. For us: FusionDance (GDN + looped core
  + ternary) is also non-standard, which is why the plan is TRL with plain `generate` first,
  where the training and sampling code paths are the same PyTorch module (no second
  implementation to keep in sync), and only then a custom decode path, plugged in through
  `rollout_func` rather than vLLM.
- **Trainer–sampler mismatch KL.** The paper monitors the KL between the rollout engine's and
  the trainer's token distributions (reported: 3.6–4.0e-3 per step with "router replay"; it
  attributes ~96% of the mismatch to BF16 MoE routing flips between the two implementations).
  For us: with plain `generate` the sampler is the trainer's own forward, so the mismatch is
  only nondeterminism. As soon as generation runs through a different path (a cached/recurrent
  GDN decode, a fused recurrent kernel, vLLM), we must log it. Note [source]: TRL 1.18.2 only
  computes the sampler-vs-trainer log-prob difference and importance-sampling correction when
  `use_vllm=True` (`sampling/sampling_logp_difference/*`, `vllm_importance_sampling_correction`);
  with a `rollout_func` and `use_vllm=False` it does not, so we would have to return sampler
  log-probs from the rollout and compute the KL ourselves (e.g. in a small `GRPOTrainer`
  subclass or callback). A cheap pre-RL check: sample with the decode path, re-score with the
  full forward, compare per-token log-probs.
- **Freezing parts of the model.** The paper freezes the first two layers, the n-gram table, the
  sparse-attention indexer and the MTP head; it reports ~10% throughput and 37 GiB host memory
  saved from the two frozen layers, and a bug where re-wrapping parameters silently made frozen
  experts trainable again. For us: transformers' Trainer only optimises parameters with
  `requires_grad=True` [source: `create_optimizer`], so freezing is `p.requires_grad_(False)`
  before building the trainer. Candidates: the prelude block(s), or the embedding — but our
  head is **tied** to the embedding, so freezing one freezes both. Verify after trainer
  construction that the frozen tensors are still frozen (the paper's re-freezing bug).
- **Kernel autotune keys.** The paper reports fla's Triton kernels autotune on the token count
  (rounded to 1,024), causing multi-second stalls when a batch lands in a new bucket. Relevant
  once GDN runs under GRPO with variable-length batches: expect first-step slowness per new
  length bucket.
- Scale differences: their setup is 4 x 8xB200 nodes, 131K-token contexts, 125B params; most of
  their memory engineering (CPU offload, context/expert parallelism) does not apply to a
  sub-100M model on one GPU.

## 10. Sources

- TRL on PyPI (version, deps): https://pypi.org/project/trl/
- GRPO trainer docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
- DPO trainer docs: https://huggingface.co/docs/trl/main/en/dpo_trainer
- SFT trainer docs: https://huggingface.co/docs/trl/main/en/sft_trainer (not fetched; facts above come from the installed source)
- Dataset formats: https://huggingface.co/docs/trl/main/en/dataset_formats (not fetched)
- TRL source (installed 1.18.2): `trl/trainer/grpo_trainer.py` (`_generate_single_turn`,
  `_get_per_token_logps_and_entropies`, `rollout_func`), `trl/trainer/utils.py`
  (`add_fused_lm_head`, `create_model_from_path`), `trl/trainer/{grpo,sft,dpo}_config.py`,
  `trl/trainer/base_config.py`. Upstream: https://github.com/huggingface/trl/tree/main/trl/trainer
- transformers source (installed 5.24.0): `generation/utils.py` (`_prepare_cache_for_generation`,
  `_supports_default_dynamic_cache`), `modeling_utils.py` (non-persistent buffer handling,
  `_init_weights`, tied weights), `configuration_utils.py` (dataclass configs).
