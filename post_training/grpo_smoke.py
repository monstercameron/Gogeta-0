"""Tiny CPU-only end-to-end GRPO run through TRL. Proves the WIRING, not quality.

    cd /mnt/c/Users/mreca/Desktop/Gogeta-0
    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta-rl/bin/python post_training/grpo_smoke.py

Random tiny FusionDance (attn/swa only, no GDN, so no CUDA needed) -> gogeta.hf wrapper ->
save_pretrained (so TRL can build the beta>0 reference model by path) -> GRPOTrainer with
transformers `generate` (no vLLM), 4 generations per prompt, a toy reward plus the real
code-execution reward (weight 0: a random model never passes, but it exercises the sandbox
and the `tests` column plumbing). Checks: training steps run, losses are finite, weights
change, and the code reward was called once per completion.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never touch the GPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from gogeta.hf import GogetaConfig, GogetaForCausalLM, load_tokenizer  # noqa: E402
from post_training.rewards import code_exec_reward, extract_asserts  # noqa: E402

TINY = dict(d_model=64, n_heads=4, mlp_hidden=128, prelude=["swa"], core=["attn", "swa"], coda=["attn"],
            loops=2, loops_min=1, loops_max=3, bptt_loops=2, window=16, max_seq_len=128)

DOC = '''Write a function add(a, b) that returns the sum of a and b.
```python
def add(a, b):
    return a + b
```
assert add(1, 2) == 3
assert add(-1, 1) == 0
'''
PROBLEMS = [
    ("Write a function add(a, b) that returns the sum.\n```python\n", extract_asserts(DOC)),
    ("Write a function sq(x) that returns x squared.\n```python\n", ["assert sq(3) == 9", "assert sq(-2) == 4"]),
    ("Write a function neg(x) that returns -x.\n```python\n", ["assert neg(1) == -1"]),
    ("Write a function first(xs) that returns xs[0].\n```python\n", ["assert first([5, 6]) == 5"]),
]

CALLS = {"code": 0}


def toy_reward(completions, **kwargs) -> list[float]:
    """Fraction of characters that are spaces or newlines: varies across random samples, so
    group-normalized advantages are non-zero and the policy gradient is non-trivial."""
    return [sum(ch in " \n" for ch in c) / max(1, len(c)) for c in completions]


def code_reward(completions, tests, **kwargs) -> list[float]:
    CALLS["code"] += len(completions)
    return code_exec_reward(completions, tests, timeout=2.0, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.04, help=">0 exercises the reference-model path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    tok = load_tokenizer()
    out_root = tempfile.mkdtemp(prefix="gogeta-grpo-smoke-")
    model_dir = os.path.join(out_root, "init")
    GogetaForCausalLM(GogetaConfig(**TINY)).save_pretrained(model_dir)
    model = GogetaForCausalLM.from_pretrained(model_dir)  # sets config._name_or_path for the ref model
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    n_params = sum(p.numel() for p in model.parameters())

    ds = Dataset.from_list([{"prompt": p, "tests": t} for p, t in PROBLEMS])
    cfg = GRPOConfig(
        output_dir=os.path.join(out_root, "out"),
        use_cpu=True, bf16=False, fp16=False,
        activation_checkpointing=False,         # TRL turns it ON by default (transformers>=5.23 name); unsupported here
        per_device_train_batch_size=8, num_generations=4,   # 2 prompts x 4 generations per step
        max_completion_length=16, temperature=1.0,
        max_steps=args.steps, learning_rate=1e-3, beta=args.beta,
        reward_weights=[1.0, 0.0],
        logging_steps=1, report_to="none", save_strategy="no",
        seed=args.seed, dataloader_num_workers=0,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[toy_reward, code_reward], args=cfg,
                          train_dataset=ds, processing_class=tok)
    result = trainer.train()

    logs = [h for h in trainer.state.log_history if "loss" in h]
    changed = sum(not torch.equal(before[k], v.detach()) for k, v in trainer.model.named_parameters())
    print("\n==== GRPO smoke summary ====")
    print(f"params: {n_params:,}  steps: {trainer.state.global_step}  wall: {time.time() - t0:.1f}s  beta: {args.beta}")
    for h in logs:
        keys = ["step", "loss", "reward", "reward_std", "rewards/toy_reward/mean", "rewards/code_reward/mean",
                "kl", "completions/mean_length"]
        print("  " + "  ".join(f"{k}={h[k]:.4g}" if isinstance(h.get(k), float) else f"{k}={h.get(k)}"
                               for k in keys if k in h))
    print(f"code_reward calls (completions scored): {CALLS['code']}")
    print(f"parameter tensors changed: {changed}/{len(before)}")
    ok = (trainer.state.global_step == args.steps and logs and all(math.isfinite(h["loss"]) for h in logs)
          and changed > 0 and CALLS["code"] == args.steps * 8)
    print("SMOKE:", "PASS" if ok else "FAIL")
    print(f"(scratch output in {out_root}; train_runtime={result.metrics.get('train_runtime')})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
