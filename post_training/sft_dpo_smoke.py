"""Tiny CPU-only SFT and DPO runs through TRL with the gogeta.hf wrapper. Wiring only.

    cd /mnt/c/Users/mreca/Desktop/Gogeta-0
    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta-rl/bin/python post_training/sft_dpo_smoke.py

SFT: prompt-completion dataset (loss on the completion only, TRL's default for that format).
DPO: prompt / chosen / rejected, reference model loaded by path (needs register_auto_classes,
done on `import gogeta.hf`). Checks: steps run, losses finite, weights change, and for SFT that
the trainer's tokenization ends with <|eos|> and sets prompt labels to -100.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

from gogeta.hf import GogetaConfig, GogetaForCausalLM, load_tokenizer  # noqa: E402

TINY = dict(d_model=64, n_heads=4, mlp_hidden=128, prelude=["swa"], core=["attn", "swa"], coda=["attn"],
            loops=2, loops_min=1, loops_max=3, bptt_loops=2, window=16, max_seq_len=128)
COMMON = dict(use_cpu=True, bf16=False, fp16=False, activation_checkpointing=False, max_steps=3,
              per_device_train_batch_size=4, learning_rate=1e-3, logging_steps=1, report_to="none",
              save_strategy="no", seed=0, dataloader_num_workers=0)
P = "Write a function add(a, b) that returns the sum.\n```python\n"
GOOD = "def add(a, b):\n    return a + b\n```"
BAD = "def add(a, b):\n    return a - b\n```"


def fresh_model(root):
    d = os.path.join(root, "init")
    GogetaForCausalLM(GogetaConfig(**TINY)).save_pretrained(d)
    return GogetaForCausalLM.from_pretrained(d)


def run(trainer, name):
    before = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    changed = sum(not torch.equal(before[k], v.detach()) for k, v in trainer.model.named_parameters())
    print(f"\n==== {name} ====")
    for h in logs:
        print("  " + "  ".join(f"{k}={h[k]}" for k in ("step", "loss", "rewards/margins", "rewards/accuracies",
                                                      "mean_token_accuracy") if k in h))
    print(f"  parameter tensors changed: {changed}/{len(before)}")
    return trainer.state.global_step == 3 and logs and all(math.isfinite(float(h["loss"])) for h in logs) and changed > 0


def main():
    torch.manual_seed(0)
    tok = load_tokenizer()
    root = tempfile.mkdtemp(prefix="gogeta-sft-dpo-smoke-")

    # ---- SFT
    sft_ds = Dataset.from_list([{"prompt": P, "completion": GOOD}] * 8)
    sft = SFTTrainer(model=fresh_model(root), args=SFTConfig(output_dir=os.path.join(root, "sft"), max_length=128,
                                                             **COMMON),
                     train_dataset=sft_ds, processing_class=tok)
    ex = sft.train_dataset[0]
    ids, labels = ex["input_ids"], ex["labels"]  # TRL folds completion_mask into labels (-100 = no loss)
    n_prompt = len(tok(P).input_ids)
    eos_ok = ids[-1] == tok.eos_token_id
    mask_ok = all(l == -100 for l in labels[:n_prompt]) and labels[n_prompt:] == ids[n_prompt:]
    print("SFT tokenized example: len", len(ids), "| ends with <|eos|>:", eos_ok,
          "| prompt tokens:", n_prompt, "| loss tokens:", sum(l != -100 for l in labels),
          "| prompt masked, completion kept:", mask_ok)
    ok_sft = run(sft, "SFT") and eos_ok and mask_ok

    # ---- DPO
    dpo_ds = Dataset.from_list([{"prompt": P, "chosen": GOOD, "rejected": BAD}] * 8)
    dpo = DPOTrainer(model=fresh_model(root), args=DPOConfig(output_dir=os.path.join(root, "dpo"), max_length=128,
                                                             **COMMON),
                     train_dataset=dpo_ds, processing_class=tok)
    ok_dpo = run(dpo, "DPO")

    print("\nSFT:", "PASS" if ok_sft else "FAIL", "| DPO:", "PASS" if ok_dpo else "FAIL")
    sys.exit(0 if (ok_sft and ok_dpo) else 1)


if __name__ == "__main__":
    main()
