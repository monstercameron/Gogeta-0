"""CPU tests for the opt-in training options: model.init, train.schedule, train.micro_batch.

    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python -m pytest tests/test_options.py -q
    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python tests/test_options.py

Tiny SWA/attn models only. GDN blocks need the fla Triton kernel (GPU), so no GDN case runs here;
`test_gdn_needs_gpu` is a documented skip. Init and accumulation are mixer-agnostic, but verify on a GPU box.

(a) Bit-identical defaults: with no new keys set, the initial state_dict hash and the first 8 training-step
    losses equal (1) PINNED values captured by running the PRE-CHANGE code (valid for the torch version they
    were captured with; skipped otherwise) and (2) an inline copy of the legacy loop/lr in this file.
(b) init=flat changes the residual-output std as intended. (c) wsd LR curve. (d) micro_batch=B/2 grads == full batch.
"""
import hashlib
import math
import os
import pathlib
import random
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gogeta.model import FusionDance, ModelConfig  # noqa: E402
from gogeta.train import forward_backward, lr_at, micro_batches  # noqa: E402

try:
    torch.use_deterministic_algorithms(True)
except Exception:  # pragma: no cover
    pass
torch.set_num_threads(1)

MC = dict(vocab_size=64, d_model=32, n_heads=2, mlp_hidden=64, prelude=["swa"], core=["swa", "attn"], coda=["attn"],
          loops=2, loops_min=1, loops_max=3, bptt_loops=2, window=8, max_seq_len=16)
TC = dict(lr=1e-3, warmup_steps=3, min_lr_frac=0.1, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
          batch_size=4, seq_len=16, seed=0)
N = 8

# Captured by running the pre-change gogeta/model.py + gogeta/train.py (torch 2.14.1+cu130, CPU, 1 thread).
PINNED_TORCH = "2.14.1+cu130"
PINNED_INIT_SHA256 = "61a98074fee2d968a4243720db3b20e32c113f15d6e0ee487c3e65fbc7c9afd6"
PINNED_LOSSES_HEX = ["0x1.0913500000000p+2", "0x1.0b8b220000000p+2", "0x1.0a4a020000000p+2", "0x1.0b80980000000p+2",
                     "0x1.08a4a60000000p+2", "0x1.0ac6b20000000p+2", "0x1.097f3c0000000p+2", "0x1.0a64cc0000000p+2"]


def state_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def make_opt(m, tc):
    decay = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]}, {"params": nod, "weight_decay": 0.0}],
                             lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), fused=False)


def legacy_lr_at(step, total, tc):  # verbatim copy of the pre-change lr_at
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    return tc["lr"] * (tc["min_lr_frac"] + (1 - tc["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def run_steps(tc, mc, new_path: bool):
    """Initial-state hash + N losses. new_path=False runs the pre-change loop body inline."""
    torch.manual_seed(tc["seed"])
    m = FusionDance(ModelConfig(**mc))
    h0 = state_hash(m)
    opt = make_opt(m, tc)
    g = torch.Generator().manual_seed(tc["seed"])
    rng = random.Random(tc["seed"] + 1)
    losses = []
    for step in range(N):
        lr = (lr_at if new_path else legacy_lr_at)(step, N, tc)
        for gp in opt.param_groups:
            gp["lr"] = lr
        r = rng.randint(mc["loops_min"], mc["loops_max"])
        t = torch.randint(mc["vocab_size"], (tc["batch_size"], tc["seq_len"] + 1), generator=g)
        x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
        if new_path:
            loss = forward_backward(m, x, y, r, micro_batches(tc)[1])
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = m(x, y, loops=r)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return h0, losses


class TestDefaultsBitIdentical(unittest.TestCase):
    def test_new_path_equals_legacy_path(self):
        h_new, l_new = run_steps(TC, MC, True)
        h_old, l_old = run_steps(TC, MC, False)
        self.assertEqual(h_new, h_old)
        self.assertEqual(l_new, l_old)  # exact float equality

    def test_equals_pinned_prechange_reference(self):
        if torch.__version__ != PINNED_TORCH:
            self.skipTest(f"pinned values were captured on torch {PINNED_TORCH}")
        h, losses = run_steps(TC, MC, True)
        self.assertEqual(h, PINNED_INIT_SHA256)
        self.assertEqual([float(l).hex() for l in losses], PINNED_LOSSES_HEX)

    def test_explicit_defaults_equal_unset(self):
        h, l = run_steps({**TC, "schedule": "cosine", "micro_batch": TC["batch_size"]}, {**MC, "init": "gogeta"}, True)
        self.assertEqual((h, l), run_steps(TC, MC, True))

    def test_default_cosine_lr_unchanged(self):
        for total in (50, 200):
            for s in range(total):
                self.assertEqual(lr_at(s, total, TC), legacy_lr_at(s, total, TC))


class TestInit(unittest.TestCase):
    BIG = dict(MC, d_model=256, n_heads=4, mlp_hidden=512, vocab_size=128, max_seq_len=16)

    def build(self, init):
        torch.manual_seed(1)
        m = FusionDance(ModelConfig(**{**self.BIG, "init": init}))
        lins = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
        res = [mod.weight for mod in lins if getattr(mod, "is_residual_out", False)]
        oth = [mod.weight for mod in lins if not getattr(mod, "is_residual_out", False)]
        return m, res, oth

    def test_flat_vs_gogeta(self):
        mg, rg, og = self.build("gogeta")
        mf, rf, of = self.build("flat")
        want = 0.02 / math.sqrt(2 * mg.effective_depth())
        for w in rg:
            self.assertAlmostEqual(w.std().item() / want, 1.0, delta=0.06)
        for w in rf + of + og:
            self.assertAlmostEqual(w.std().item() / 0.02, 1.0, delta=0.06)
        # same seed => same RNG draws: non-residual weights bit-identical, residual ones differ by the scale only
        for a, b in zip(og, of):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(rg, rf):
            torch.testing.assert_close(a * (0.02 / want), b, rtol=1e-5, atol=0)

    def test_bad_init_rejected(self):
        with self.assertRaises(ValueError):
            FusionDance(ModelConfig(**{**MC, "init": "nope"}))

    def test_hf_llama_gogeta_init(self):
        try:
            from gogeta.train import HFLlama
            import transformers  # noqa: F401
        except Exception as e:
            self.skipTest(f"transformers unavailable: {e}")
        base = dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=4, max_position_embeddings=64, tie_word_embeddings=True)
        torch.manual_seed(0)
        hf = HFLlama(base, 512, 64)  # default: HF's own init
        torch.manual_seed(0)
        sc = HFLlama({**base, "init": "gogeta"}, 512, 64)
        want = 0.02 / math.sqrt(2 * 4)
        self.assertNotIn("init", hf.config_dict())
        self.assertEqual(sc.config_dict()["init"], "gogeta")
        for n, p in sc.lm.named_parameters():
            if n.endswith(("o_proj.weight", "down_proj.weight")):
                self.assertAlmostEqual(p.std().item() / want, 1.0, delta=0.1, msg=n)
        for n, p in hf.lm.named_parameters():
            if n.endswith(("o_proj.weight", "down_proj.weight")):
                self.assertAlmostEqual(p.std().item() / 0.02, 1.0, delta=0.1, msg=n)

    def test_hf_wrapper_config_roundtrip(self):
        try:
            from gogeta.hf import GogetaConfig
        except Exception as e:
            self.skipTest(f"transformers/hf wrapper unavailable: {e}")
        mc = ModelConfig(**{**MC, "init": "flat"})
        self.assertEqual(GogetaConfig.from_model_config(mc).to_model_config(), mc)


class TestWSD(unittest.TestCase):
    def test_curve(self):
        tc = dict(lr=1.0, warmup_steps=10, min_lr_frac=0.1, schedule="wsd", decay_frac=0.2)
        T = 100
        lrs = [lr_at(s, T, tc) for s in range(T)]
        for s in range(10):
            self.assertAlmostEqual(lrs[s], (s + 1) / 10)  # warmup
        for s in range(10, 80):
            self.assertEqual(lrs[s], 1.0)  # stable until the last 20%
        for s in range(80, T):
            self.assertAlmostEqual(lrs[s], 1 - 0.9 * (s - 80) / 20)  # linear from the peak to min_lr_frac
        self.assertTrue(all(a >= b for a, b in zip(lrs[80:], lrs[81:])))
        self.assertAlmostEqual(lrs[-1], 0.1 + 0.9 / 20)  # last step is one tick above min
        tc2 = {**tc, "decay_frac": 0.5}
        self.assertEqual(lr_at(49, T, tc2), 1.0)
        self.assertLess(lr_at(60, T, tc2), 1.0)

    def test_default_decay_frac_and_bad_name(self):
        tc = dict(lr=1.0, warmup_steps=1, min_lr_frac=0.0, schedule="wsd")
        self.assertEqual(lr_at(79, 100, tc), 1.0)
        self.assertLess(lr_at(85, 100, tc), 1.0)
        with self.assertRaises(ValueError):
            lr_at(5, 100, {**tc, "schedule": "linear"})


class TestMicroBatch(unittest.TestCase):
    def grads(self, micro, B=4):
        torch.manual_seed(3)
        m = FusionDance(ModelConfig(**MC))
        g = torch.Generator().manual_seed(5)
        t = torch.randint(MC["vocab_size"], (B, 17), generator=g)
        x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
        loss = forward_backward(m, x, y, 2, micro_batches({"batch_size": B, "micro_batch": micro})[1])
        return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    def test_half_batch_matches_full(self):
        l1, g1 = self.grads(None)
        for micro in (2, 1):
            l2, g2 = self.grads(micro)
            self.assertAlmostEqual(l1, l2, places=5)
            self.assertEqual(g1.keys(), g2.keys())
            for n in g1:
                rel = ((g1[n] - g2[n]).abs().max() / g1[n].abs().max().clamp(min=1e-12)).item()
                self.assertLess(rel, 1e-5, msg=f"{n} micro={micro} rel={rel:.2e}")

    def test_divisibility_asserted(self):
        with self.assertRaises(AssertionError):
            micro_batches({"batch_size": 4, "micro_batch": 3})
        self.assertEqual(micro_batches({"batch_size": 4}), (4, 1))
        self.assertEqual(micro_batches({"batch_size": 4, "micro_batch": 2}), (2, 2))


class TestGDN(unittest.TestCase):
    @unittest.skip("GDN needs the fla Triton kernel (GPU); verify init and --micro-batch on a GPU box with a gdn config.")
    def test_gdn_needs_gpu(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
