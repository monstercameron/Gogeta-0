"""CPU tests for the opt-in Code-MoE FFN (gogeta/moe.py, ModelConfig.moe_*; E60).

    wsl: CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python -m unittest tests.test_moe -v
    capture the pinned reference (run on the PRE-CHANGE code only): python tests/test_moe.py --capture

(a) Bit-identical defaults: E0/E1/E2/E3/E9/E11 configs at default (dense) fields give the same initial
    state_dict sha256, RNG state after construction, to_dict(), param_counts(), flops_per_token(1024) and the
    first CPU training losses as PINNED values captured by running the unmodified gogeta/model.py + train.py
    (torch 2.14.1+cu130, CPU, 1 thread). GDN blocks run on a plain-PyTorch reference recurrence on CPU.
"""
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gogeta.model import FusionDance, GatedDeltaNet, ModelConfig  # noqa: E402
from gogeta.train import forward_backward  # noqa: E402

try:
    torch.use_deterministic_algorithms(True)
except Exception:  # pragma: no cover
    pass
torch.set_num_threads(1)

REF_CONFIGS = ["e0_dense", "e1_hybrid", "e2_looped", "e3_ternary", "e9_unlooped", "e11_negeig"]
LOSS_CONFIGS = ["e0_dense", "e1_hybrid", "e2_looped"]
TC = dict(lr=6e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0, batch_size=2, seq_len=16, seed=0)
N_STEPS = 3


# ----------------------------------------------------------------------------- CPU GDN
def gated_delta_rule_ref(q, k, v, g, beta, scale=None, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, **_):
    """Recurrent gated delta rule in plain PyTorch (copy of experiments/checks/sq_99_out_of_corpus.py)."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** -0.5 if scale is None else scale
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    S = q.new_zeros(B, H, dk, dv)
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t] - pred)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
    return torch.stack(outs, 1), S


def cpu_gdn(m):
    for mod in m.modules():
        if isinstance(mod, GatedDeltaNet):
            mod._kernel = gated_delta_rule_ref
    return m


# ----------------------------------------------------------------------------- fingerprints
def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def state_hash(m) -> str:
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_cfg(name: str, **extra) -> dict:
    return {**json.loads((ROOT / "configs" / f"{name}.json").read_text())["model"], **extra}


def fingerprint(name: str, **extra) -> dict:
    torch.manual_seed(0)
    mc = ModelConfig(**model_cfg(name, **extra))
    m = FusionDance(mc)
    return {"state": state_hash(m), "rng_after": sha(torch.get_rng_state()),
            "to_dict": hashlib.sha256(json.dumps(mc.to_dict(), sort_keys=True).encode()).hexdigest(),
            "param_counts": m.param_counts(), "flops_1024": float(m.flops_per_token(1024)).hex()}


def train_losses(name: str, **extra) -> list:
    torch.manual_seed(TC["seed"])
    mc = ModelConfig(**model_cfg(name, **extra))
    m = cpu_gdn(FusionDance(mc))
    decay = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": TC["weight_decay"]}, {"params": nod, "weight_decay": 0.0}],
                            lr=TC["lr"], betas=(TC["beta1"], TC["beta2"]), fused=False)
    g = torch.Generator().manual_seed(TC["seed"])
    rng = random.Random(TC["seed"] + 1)
    looped = len(m.core) > 0
    out = []
    for _ in range(N_STEPS):
        r = rng.randint(mc.loops_min, mc.loops_max) if looped else None
        t = torch.randint(mc.vocab_size, (TC["batch_size"], TC["seq_len"] + 1), generator=g)
        x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
        loss = forward_backward(m, x, y, r, 1)
        torch.nn.utils.clip_grad_norm_(m.parameters(), TC["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        out.append(float(loss.item()).hex())
    return out


def capture() -> dict:
    return {"torch": torch.__version__,
            "fingerprints": {n: fingerprint(n) for n in REF_CONFIGS},
            "losses": {n: train_losses(n) for n in LOSS_CONFIGS}}


if __name__ == "__main__" and "--capture" in sys.argv:
    print(json.dumps(capture(), indent=1))
    sys.exit(0)


# Captured with `python tests/test_moe.py --capture` on the UNMODIFIED gogeta/model.py + gogeta/train.py at commit
# 5e72d80 (v0.4.94), before gogeta/moe.py existed (torch 2.14.1+cu130, CPU, 1 thread, WSL Ubuntu 24.04).
PINNED_TORCH = "2.14.1+cu130"
_E1_PC = {"total": 14237796, "embedding": 3145728, "prelude": 11091684, "core": 0, "coda": 0, "other": 384,
          "non_embedding": 11092068}
_E1_STATE = "af109624044138ac80dffbe5cceb09e4f0fae1195cd7d6cb91a83be104f9c9cc"
_E1_RNG = "99448948294a818cc8d29f8059d54458fe088615605c4041bf7df1c985a47278"
PINNED = {
    "e0_dense": {"state": "5d9d34631a9b1e1d8dcb3bd94fe2264c03cc3c2160aa5a8fbd6e60824578c926",
                 "rng_after": "d9e256f89f17d31e91437ebdd94672258d79239be3389e424f7f0b8fa0241eb1",
                 "to_dict": "6aa4f93cdfd3059d39b485e61ee581950fa33d01f03f7cee56b9d7078546e47c",
                 "param_counts": {"total": 13767552, "embedding": 3145728, "prelude": 10621440, "core": 0, "coda": 0,
                                  "other": 384, "non_embedding": 10621824},
                 "flops_1024": "0x1.711b000000000p+26"},
    "e1_hybrid": {"state": _E1_STATE, "rng_after": _E1_RNG,
                  "to_dict": "14b6518ae5d0b75a8a5efea26947e1df97d92e15b53181b26f60da9005c76302",
                  "param_counts": _E1_PC, "flops_1024": "0x1.5b3e560000000p+26"},
    "e2_looped": {"state": "38b80959806556d4a20e5943fae1310cf6a48ccb858f9957b901d171c91b0836",
                  "rng_after": "6c6b46f2467f9c371502cb0dc4e5429210ea7e678dcc2ea8f83eb28b87a5ed01",
                  "to_dict": "0945e78bc4dc6d2195bf439ee42224a803b9cea7f91290e09cb3f5a20c261eb4",
                  "param_counts": {"total": 12606488, "embedding": 3145728, "prelude": 1770240, "core": 5624216,
                                   "coda": 1770240, "other": 296064, "non_embedding": 9460760},
                  "flops_1024": "0x1.6fb9480000000p+27"},
    "e3_ternary": {"state": _E1_STATE, "rng_after": _E1_RNG,
                   "to_dict": "9e7fb4146dbc499956f78dae161d70cb72d5c73fbe965e5cf19f1317f9b94942",
                   "param_counts": _E1_PC, "flops_1024": "0x1.5b3e560000000p+26"},
    "e9_unlooped": {"state": "6925176c20f82eefb107776778613277f382fef2be03807d2f11cb7ad5088def",
                    "rng_after": "286fee8eafbcde3bdb9df37769420e9589f7d378301f40e3dd5991b792c45c1d",
                    "to_dict": "bc6f658af51aba9a0d94af8388cacde4cc55cec77d7882f1c8b9ca0646a65d7b",
                    "param_counts": {"total": 12310808, "embedding": 3145728, "prelude": 9164696, "core": 0, "coda": 0,
                                     "other": 384, "non_embedding": 9165080},
                    "flops_1024": "0x1.2e03640000000p+26"},
    "e11_negeig": {"state": _E1_STATE, "rng_after": _E1_RNG,
                   "to_dict": "a389cd6f5ad754b8470fbdd0389af0ac8a3d97911be0f3be89772c3939b1bf38",
                   "param_counts": _E1_PC, "flops_1024": "0x1.5b3e560000000p+26"},
}
PINNED_LOSSES = {
    "e0_dense": ["0x1.278a620000000p+3", "0x1.247e680000000p+3", "0x1.1fa1a80000000p+3"],
    "e1_hybrid": ["0x1.24f66e0000000p+3", "0x1.21781e0000000p+3", "0x1.2060e60000000p+3"],
    "e2_looped": ["0x1.245f240000000p+3", "0x1.1ecb6e0000000p+3", "0x1.22d9220000000p+3"],
}
MOE_DEFAULTS = dict(moe_experts=0, moe_top_k=2, moe_hidden=512, moe_balance_coef=0.01, moe_zloss_coef=1e-3,
                    moe_router_frozen=False, moe_impl="gather")


class TestDefaultsBitIdentical(unittest.TestCase):
    def setUp(self):
        if torch.__version__ != PINNED_TORCH:
            self.skipTest(f"pinned values were captured on torch {PINNED_TORCH}")

    def test_construction_matches_prechange(self):
        for name in REF_CONFIGS:
            with self.subTest(name):
                self.assertEqual(fingerprint(name), PINNED[name])

    def test_explicit_dense_moe_fields_match_prechange(self):
        for name in REF_CONFIGS:
            with self.subTest(name):
                self.assertEqual(fingerprint(name, **MOE_DEFAULTS), PINNED[name])

    def test_training_losses_match_prechange(self):
        for name in LOSS_CONFIGS:
            with self.subTest(name):
                self.assertEqual(train_losses(name), PINNED_LOSSES[name])  # exact float equality

    def test_dense_to_dict_has_no_moe_keys(self):
        d = ModelConfig(**model_cfg("e1_hybrid")).to_dict()
        self.assertFalse([k for k in d if k.startswith("moe_")])
        self.assertTrue(all(k in ModelConfig(**model_cfg("e1_hybrid", moe_experts=4)).to_dict() for k in MOE_DEFAULTS))

    def test_dense_forward_signature_unchanged(self):
        m = build(TINY)
        x = torch.randint(64, (2, 8))
        out = m(x, x)
        self.assertEqual(len(out), 2)
        self.assertIsNone(m(x, x, return_aux=True)[2])


# ----------------------------------------------------------------------------- MoE behaviour (tiny CPU models)
TINY = dict(vocab_size=64, d_model=32, n_heads=2, mlp_hidden=64, prelude=["swa", "gdn", "attn"], core=[], coda=[],
            window=8, max_seq_len=32)
TINY_MOE = dict(TINY, moe_experts=4, moe_top_k=2, moe_hidden=16)
TINY_LOOPED_MOE = dict(TINY_MOE, prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=3, loops_min=1,
                       loops_max=4, bptt_loops=2)


def build(mc: dict, seed: int = 0):
    torch.manual_seed(seed)
    return cpu_gdn(FusionDance(ModelConfig(**mc)))


class TestMoE(unittest.TestCase):
    def test_gather_equals_dense_impl(self):
        for mc in (TINY_MOE, TINY_LOOPED_MOE):
            a, b = build(mc), build({**mc, "moe_impl": "dense"})
            x = torch.randint(64, (3, 20), generator=torch.Generator().manual_seed(1))
            la, lossa, auxa = a(x, x, return_aux=True)
            lb, lossb, auxb = b(x, x, return_aux=True)
            torch.testing.assert_close(la, lb, rtol=0, atol=1e-5)
            (lossa + auxa["loss"]).backward()
            (lossb + auxb["loss"]).backward()
            for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                if pa.grad is not None:
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=1e-5, msg=n)

    def test_moe_layer_matches_manual_topk_mixture(self):
        m = build(TINY_MOE)
        moe = m.prelude[0].mlp
        x = torch.randn(5, 32)
        y = moe(x)
        p = (x @ moe.router.weight.T).softmax(-1)
        for t in range(5):
            tv, ti = p[t].topk(2)
            want = sum((tv[j] / tv.sum()) * moe._expert(int(ti[j]), x[t:t + 1])[0] for j in range(2))
            torch.testing.assert_close(y[t], want, rtol=0, atol=1e-6)

    def test_causality(self):
        """Perturbing tokens t+1..T-1 leaves logits at positions <= t unchanged (MoE on, dropless token routing)."""
        T = 24
        for mc in (TINY_MOE, {**TINY_MOE, "moe_impl": "dense"}, TINY_LOOPED_MOE):
            m = build(mc).eval()
            g = torch.Generator().manual_seed(2)
            x = torch.randint(64, (2, T), generator=g)
            with torch.no_grad():
                base = m(x)[0]
                for t in (0, 5, 11, T - 2):
                    x2 = x.clone()
                    x2[:, t + 1:] = torch.randint(64, (2, T - t - 1), generator=g)
                    out = m(x2)[0]
                    torch.testing.assert_close(out[:, :t + 1], base[:, :t + 1], rtol=0, atol=1e-6)
                    self.assertGreater((out[:, t + 1:] - base[:, t + 1:]).abs().max().item(), 1e-3)  # test has teeth

    def test_balance_and_z_loss_normalisation(self):
        moe = build(TINY_MOE).prelude[0].mlp
        E, N = 4, 8
        # uniform: every expert gets N*k/E assignments, mean prob 1/E -> 1.0
        top_i = torch.tensor([[0, 1], [2, 3]] * (N // 2))
        probs = torch.full((N, E), 1 / E)
        logits = torch.zeros(N, E)
        s = moe.stats(logits, probs, top_i)
        self.assertAlmostEqual(s["balance"].item(), 1.0, places=6)
        self.assertAlmostEqual(s["z"].item(), math.log(E) ** 2, places=5)
        # collapsed onto experts {0, 1} with all mass there -> E * (0.5*0.5 + 0.5*0.5) = 2.0
        probs = torch.tensor([[0.5, 0.5, 0.0, 0.0]]).repeat(N, 1)
        s = moe.stats(logits, probs, torch.tensor([[0, 1]]).repeat(N, 1))
        self.assertAlmostEqual(s["balance"].item(), 2.0, places=6)
        self.assertEqual(s["load"].tolist(), [0.5, 0.5, 0.0, 0.0])

    def test_aux_loss_is_added_and_logged(self):
        x = torch.randint(64, (2, 16), generator=torch.Generator().manual_seed(3))
        y = torch.roll(x, -1, 1)
        m = build(TINY_MOE)
        stats = {}
        loss = forward_backward(m, x, y, None, 1, stats)
        g_fb = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        self.assertEqual(set(stats), {"aux", "balance", "z", "max_load", "min_load", "entropy"})
        m2 = build(TINY_MOE)
        _, lm, aux = m2(x, y, return_aux=True)
        self.assertEqual(loss.item(), lm.item())                     # returned/logged loss is the LM loss only
        want = 0.01 * 3 * aux["balance"] + 1e-3 * 3 * aux["z"]      # 3 MoE layers, 1 call each (sum of layer terms)
        torch.testing.assert_close(aux["loss"], want, rtol=1e-6, atol=0)
        self.assertAlmostEqual(stats["aux"].item(), aux["loss"].item(), places=6)
        (lm + aux["loss"]).backward()
        for n, p in m2.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(g_fb[n], p.grad, rtol=0, atol=0, msg=n)
        # the aux term actually changes the router gradient
        m3 = build({**TINY_MOE, "moe_balance_coef": 0.0, "moe_zloss_coef": 0.0})
        forward_backward(m3, x, y, None, 1, {})
        r = "prelude.0.mlp.router.weight"
        self.assertFalse(torch.equal(dict(m3.named_parameters())[r].grad, g_fb[r]))

    def test_looped_aux_uses_grad_calls_only(self):
        m = build(TINY_LOOPED_MOE).train()
        x = torch.randint(64, (2, 12), generator=torch.Generator().manual_seed(4))
        _, lm, aux = m(x, x, loops=4, return_aux=True)               # 2 no-grad + 2 grad loops
        self.assertEqual(tuple(aux["load"].shape), (4, 4))           # 4 MoE layers (swa, gdn, swa, attn)
        (lm + aux["loss"]).backward()
        self.assertIsNotNone(m.core[0].mlp.router.weight.grad)

    def test_frozen_router(self):
        m = build({**TINY_MOE, "moe_router_frozen": True})
        routers = [b.mlp.router.weight for b in m.prelude]
        self.assertTrue(all(not w.requires_grad for w in routers))
        before = [w.clone() for w in routers]
        other = m.prelude[0].mlp.w_up.detach().clone()
        decay = [p for p in m.parameters() if p.dim() >= 2]
        nod = [p for p in m.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}], lr=1e-2)
        x = torch.randint(64, (2, 16), generator=torch.Generator().manual_seed(5))
        for _ in range(2):
            forward_backward(m, x, torch.roll(x, -1, 1), None, 1, {})
            self.assertTrue(all(w.grad is None for w in routers))
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for a, b in zip(before, routers):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(other, m.prelude[0].mlp.w_up))

    def test_init_and_guards(self):
        big = dict(TINY_MOE, d_model=256, n_heads=4, moe_hidden=256, prelude=["swa", "attn"])
        m = build(big)
        std_res = 0.02 / math.sqrt(2 * m.effective_depth())
        moe = m.prelude[0].mlp
        self.assertAlmostEqual(moe.w_up.std().item() / 0.02, 1.0, delta=0.02)
        self.assertAlmostEqual(moe.w_down.std().item() / std_res, 1.0, delta=0.02)
        self.assertAlmostEqual(moe.router.weight.std().item() / 0.02, 1.0, delta=0.1)
        # v0.5.10 (A3): MoE with ternary is supported (BitLinear-style fake-quantised experts); it used to raise
        self.assertTrue(all(b.mlp.ternary for b in FusionDance(ModelConfig(**{**TINY_MOE, "ternary": True})).prelude if b.moe))
        with self.assertRaises(ValueError):
            FusionDance(ModelConfig(**{**TINY_MOE, "moe_top_k": 5}))
        with self.assertRaises(ValueError):
            FusionDance(ModelConfig(**{**TINY_MOE, "moe_impl": "nope"}))


# ----------------------------------------------------------------------------- E60 configs: params and FLOPs
E60_DIR = ROOT / "experiments" / "staged-configs" / "e60"


class TestE60Accounting(unittest.TestCase):
    """Reproduces experiments/design/code-moe-draft.md §2 (computed) from param_counts() / flops_per_token()."""

    def build_cfg(self, path):
        torch.manual_seed(0)
        return FusionDance(ModelConfig(**json.loads(path.read_text())["model"]))

    def test_design_table(self):
        e1 = self.build_cfg(ROOT / "configs" / "e1_hybrid.json")
        a = self.build_cfg(E60_DIR / "e60a_moe.json")
        b = self.build_cfg(E60_DIR / "e60b_frozen_router.json")
        c = self.build_cfg(E60_DIR / "e60c_dense2048.json")
        self.assertEqual(e1.param_counts()["non_embedding"], 11_092_068)
        self.assertEqual(e1.flops_per_token(1024), 91_027_800)
        for m in (a, b):
            pc = m.param_counts()
            self.assertEqual(pc["non_embedding"], 18_179_172)
            self.assertEqual(pc["active_non_embedding"], 11_092_068 + 6 * 384 * 4)   # E1 + 6 routers
            self.assertEqual(pc["moe_inactive"], 6 * 2 * 3 * 384 * 512)
            self.assertEqual(m.flops_per_token(1024), 91_083_096)
        self.assertEqual(c.param_counts()["non_embedding"], 18_169_956)
        self.assertNotIn("active_non_embedding", c.param_counts())
        self.assertEqual(c.flops_per_token(1024), 133_495_128)
        self.assertTrue(all(not blk.mlp.router.weight.requires_grad for blk in b.prelude))
        self.assertEqual(b.cfg.moe_balance_coef, 0.0)

    def test_configs_derive_from_e1(self):
        e1 = json.loads((ROOT / "configs" / "e1_hybrid.json").read_text())
        for f in ("e60a_moe", "e60b_frozen_router", "e60c_dense2048"):
            cfg = json.loads((E60_DIR / f"{f}.json").read_text())
            self.assertTrue(cfg["description"])
            self.assertEqual({**e1["train"], "lr": 0.006}, cfg["train"])
            for k, v in e1["model"].items():
                if not (f == "e60c_dense2048" and k == "mlp_hidden"):
                    self.assertEqual(cfg["model"][k], v, msg=f"{f}.{k}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
