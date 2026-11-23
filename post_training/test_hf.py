"""Wiring tests for gogeta/hf.py (CPU only, tiny attn/swa model, a few seconds).

    CUDA_VISIBLE_DEVICES= python -m unittest post_training.test_hf -v     (repo root, WSL, gogeta-rl venv)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from gogeta.hf import (GogetaConfig, GogetaForCausalLM, from_gogeta_checkpoint, load_tokenizer,  # noqa: E402
                       to_gogeta_checkpoint)
from gogeta.model import FusionDance, ModelConfig  # noqa: E402

TINY = dict(vocab_size=8192, d_model=64, n_heads=4, mlp_hidden=128, prelude=["swa"], core=["attn", "swa"],
            coda=["attn"], loops=2, loops_min=1, loops_max=3, bptt_loops=2, window=8, max_seq_len=128)


def tiny_checkpoint(seed=0) -> dict:
    torch.manual_seed(seed)
    fd = FusionDance(ModelConfig(**TINY))
    return {"model": fd.state_dict(), "model_config": fd.cfg.to_dict(), "train": {}}


class TestWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt = tiny_checkpoint()
        cls.fd = FusionDance(ModelConfig(**cls.ckpt["model_config"])).eval()
        cls.fd.load_state_dict(cls.ckpt["model"])
        cls.m = from_gogeta_checkpoint(cls.ckpt).eval()
        cls.tok = load_tokenizer()

    def test_config_roundtrip(self):
        self.assertEqual(self.m.config.to_model_config(), ModelConfig(**TINY))

    def test_logits_match_fusiondance(self):
        x = torch.randint(2, 8192, (3, 40))
        with torch.no_grad():
            ref, _ = self.fd(x)
            out = self.m(input_ids=x).logits
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual((out - ref).abs().max().item(), 0.0)

    def test_loss_matches_shifted_fusiondance(self):
        x = torch.randint(2, 8192, (2, 33))
        with torch.no_grad():
            _, ref = self.fd(x[:, :-1], x[:, 1:])
            loss = self.m(input_ids=x, labels=x).loss
        self.assertAlmostEqual(loss.item(), ref.item(), places=5)

    def test_left_padding_is_exact(self):
        a = torch.randint(2, 8192, (1, 20))
        b = torch.randint(2, 8192, (1, 12))
        pad = self.tok.pad_token_id
        batch = torch.cat([a, torch.cat([torch.full((1, 8), pad), b], 1)])
        mask = torch.cat([torch.ones(1, 20), torch.cat([torch.zeros(1, 8), torch.ones(1, 12)], 1)]).long()
        with torch.no_grad():
            out = self.m(input_ids=batch, attention_mask=mask).logits
            ref_b, _ = self.fd(b)
        self.assertLess((out[1, 8:] - ref_b[0]).abs().max().item(), 1e-5)

    def test_mask_with_hole_rejected(self):
        x = torch.randint(2, 8192, (1, 10))
        mask = torch.ones(1, 10, dtype=torch.long)
        mask[0, 4] = 0
        with self.assertRaises(ValueError):
            self.m(input_ids=x, attention_mask=mask)

    def test_generate_greedy_matches_manual_loop(self):
        enc = self.tok(["def add(a, b):", "x ="], return_tensors="pt", padding=True)
        with torch.no_grad():
            g = self.m.generate(**enc, max_new_tokens=6, do_sample=False, use_cache=False,
                                eos_token_id=None, pad_token_id=self.tok.pad_token_id)
        # manual greedy on the un-padded second prompt
        ids = self.tok("x =", return_tensors="pt").input_ids
        with torch.no_grad():
            for _ in range(6):
                nxt = self.fd(ids)[0][:, -1].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], 1)
        self.assertEqual(g[1, -6:].tolist(), ids[0, -6:].tolist())

    def test_save_pretrained_roundtrip(self):
        x = torch.randint(2, 8192, (2, 16))
        with tempfile.TemporaryDirectory() as d:
            self.m.save_pretrained(d)
            m2 = GogetaForCausalLM.from_pretrained(d).eval()
        self.assertIs(m2.model.head.weight, m2.model.embed.weight)  # tie survives
        with torch.no_grad():
            self.assertEqual((m2(input_ids=x).logits - self.m(input_ids=x).logits).abs().max().item(), 0.0)

    def test_trl_fused_lm_head_logprobs(self):
        """TRL's GRPO/DPO compute log-probs through add_fused_lm_head (base_model + output
        embeddings). They must equal log_softmax of our own logits, with left padding."""
        try:
            from trl.trainer.utils import add_fused_lm_head
        except ImportError:
            self.skipTest("trl not installed")
        m = from_gogeta_checkpoint(self.ckpt).eval()
        add_fused_lm_head(m, outputs=("log_probs", "entropy"))
        pad = self.tok.pad_token_id
        x = torch.randint(2, 8192, (2, 24))
        x[1, :5] = pad
        mask = torch.ones_like(x)
        mask[1, :5] = 0
        labels = x.masked_fill(mask == 0, -100)
        labels[:, :10] = -100  # score only the "completion" part, as GRPO does
        with torch.no_grad():
            out = m(input_ids=x, attention_mask=mask, labels=labels, fused_lm_head=True)
            ref = torch.log_softmax(self.m(input_ids=x, attention_mask=mask).logits.float(), -1)
        ref_lp = ref[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1) * (labels[:, 1:] != -100)
        self.assertLess((out.log_probs - ref_lp).abs().max().item(), 1e-4)
        self.assertIsNotNone(out.entropy)

    def test_export_checkpoint_roundtrip(self):
        ck = to_gogeta_checkpoint(self.m)
        fd = FusionDance(ModelConfig(**ck["model_config"]))
        fd.load_state_dict(ck["model"])

    def test_tokenizer(self):
        self.assertEqual((self.tok.eos_token, self.tok.pad_token), ("<|eos|>", "<|pad|>"))
        self.assertEqual(self.tok.padding_side, "left")
        s = "def f(x):\n    return x + 1\n"
        self.assertEqual(self.tok.decode(self.tok(s).input_ids), s)


class TestAssembledConfig(unittest.TestCase):
    """v0.5.24: the wrapper must honour the assembly switches (loop_embed, contractive inject, per-pass adapters,
    ternary scope). Before v0.5.24 GogetaConfig lacked these fields and hidden() was a hand-written mirror of
    FusionDance.forward that ignored them."""

    def test_assembled_logits_match_fusiondance(self):
        cfg = dict(TINY, loop_embed=True, inject_contractive=True, loop_adapter_rank=4, ternary=True,
                   ternary_fp_ends=True, ternary_mixer=False)
        torch.manual_seed(1)
        fd = FusionDance(ModelConfig(**cfg)).eval()
        ckpt = {"model": fd.state_dict(), "model_config": fd.cfg.to_dict(), "train": {}}
        m = from_gogeta_checkpoint(ckpt).eval()
        self.assertEqual(m.config.to_model_config(), ModelConfig(**cfg))
        x = torch.randint(2, 8192, (2, 24))
        with torch.no_grad():
            ref, _ = fd(x)
            out = m(input_ids=x).logits
        self.assertEqual((out - ref).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
