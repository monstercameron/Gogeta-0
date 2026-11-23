"""Unit tests for the byte model family (CPU only; GDN through cpu_ref's recurrent reference).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python -m unittest experiments.byte_model.test_byte_model -v

Covers: SQ206 causality for every variant and P in {2, 4, 8} (+16) with the broken (no-shift) control failing;
parallel_refine masking inside the target patch; default ModelConfig keeps token models bit-identical to
gogeta.model (weights, outputs, config dict); per-pass loss shapes and weighting; every staged byte config builds at
full width with RoPE tables long enough and runs a forward at its full T (tiny width); dataset round trip (when the
built dataset is present); param / FLOP numbers vs notes/analysis-coarse-to-fine.md section 3 and the table.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.byte_model import cpu_ref  # noqa: E402

cpu_ref.install()
import gogeta.model as G  # noqa: E402  (reads only; PYTHONDONTWRITEBYTECODE=1 keeps gogeta/ untouched)
from experiments.byte_model import causality  # noqa: E402
from experiments.byte_model import model_bytes as M  # noqa: E402

torch.set_num_threads(2)
CFG_DIR = ROOT / "experiments" / "staged-configs" / "byte"
DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))
TINY_G = {"d_model": 32, "n_heads": 2, "mlp_hidden": 64}
TINY_L = {"d_local": 16, "local_heads": 2, "local_mlp_hidden": 32}


def tiny_patched(**kw):
    base = dict(vocab_size=256, max_seq_len=32, patch_size=4, prelude=["swa", "gdn", "attn"], core=[], coda=[],
                window=4, **TINY_G, **TINY_L)
    base.update(kw)
    return M.make_model(M.ModelConfig(**base))


class TestCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = causality.run_all(Ps=(2, 4, 8), T=64, extra_P16=True)

    def test_all_variants_bitwise_causal(self):
        bad = [(r["variant"], r["P"]) for r in self.out["results"] if not r["ok"]]
        self.assertEqual(bad, [], f"leaks in {bad}")

    def test_non_vacuous(self):
        vac = [(r["variant"], r["P"]) for r in self.out["results"] if not r["non_vacuous"]]
        self.assertEqual(vac, [])

    def test_every_variant_and_P_covered(self):
        got = {(r["variant"], r["P"]) for r in self.out["results"]}
        for P in (2, 4, 8):
            for v in causality.VARIANTS:
                self.assertIn((v, P), got)
        self.assertIn(("flat", 1), got)

    def test_broken_variant_fails(self):
        for r in self.out["broken_controls"]:
            self.assertFalse(r["ok"], f"{r['variant']} P={r['P']} should leak")

    def test_training_mode_and_sampled_loops_causal(self):
        """Training mode (bptt no_grad, sampled global loops) must not change causality."""
        torch.manual_seed(0)
        m = tiny_patched(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=3, bptt_loops=1,
                         refine_passes=2).train()
        x = torch.randint(0, 256, (1, 32))
        with torch.no_grad():
            a = m(x, loops=3)[0]
            xp = x.clone()
            xp[0, 13] = (xp[0, 13] + 7) % 256
            b = m(xp, loops=3)[0]
        self.assertTrue(torch.equal(a[:, :13], b[:, :13]))
        self.assertFalse(torch.equal(a[:, 13], b[:, 13]))


class TestParallelMasking(unittest.TestCase):
    def test_target_patch_bytes_invisible(self):
        """parallel_refine: rows predicting target patch q see no real byte of patch q, in any pass."""
        torch.manual_seed(1)
        P, T = 4, 32
        m = tiny_patched(decoder="parallel_refine", refine_passes=3).eval()
        x = torch.randint(0, 256, (2, T))
        base = [lg for _, lg in m(x, return_all=True)[0]]
        for s in range(T):
            xp = x.clone()
            xp[:, s] = (xp[:, s] + 5) % 256
            pert = [lg for _, lg in m(xp, return_all=True)[0]]
            q = s // P
            rows = [j for j in range(T) if (j + 1) // P == q]     # rows whose target patch contains byte s
            for a, b in zip(base, pert):
                self.assertTrue(torch.equal(a[:, rows], b[:, rows]), f"s={s} leaks into its own target patch")

    def test_ar_mode_does_see_own_patch(self):
        """Sanity: in ar mode, the next row after s inside the same patch does change (bytes flow causally)."""
        torch.manual_seed(1)
        m = tiny_patched(decoder="ar", refine_passes=1).eval()
        x = torch.randint(0, 256, (1, 32))
        xp = x.clone()
        xp[0, 9] = (xp[0, 9] + 5) % 256
        self.assertFalse(torch.equal(m(x)[0][:, 10], m(xp)[0][:, 10]))

    def test_right_padding_for_generation(self):
        """A prompt that is not a multiple of P gives the same logits as the first rows of a longer input."""
        torch.manual_seed(2)
        m = tiny_patched(refine_passes=2).eval()
        x = torch.randint(0, 256, (1, 32))
        full, part = m(x)[0], m(x[:, :19])[0]
        self.assertEqual(part.shape[1], 19)
        torch.testing.assert_close(part, full[:, :19], rtol=1e-5, atol=1e-5)


class TestDefaultsIdentical(unittest.TestCase):
    CASES = {
        "E1": dict(prelude=["swa", "gdn", "gdn", "swa", "gdn", "attn"], core=[], coda=[]),
        "E2": dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=2),
        "E3": dict(prelude=["swa", "gdn", "attn"], core=[], coda=[], ternary=True),
        "E11": dict(prelude=["gdn", "attn"], core=[], coda=[], gdn_neg_eigval=True),
    }

    def test_config_dict_identical(self):
        self.assertEqual(M.ModelConfig().to_dict(), G.ModelConfig().to_dict())
        for kw in self.CASES.values():
            self.assertEqual(M.ModelConfig(**kw).to_dict(), G.ModelConfig(**kw).to_dict())

    def test_weights_and_outputs_bit_identical(self):
        for name, kw in self.CASES.items():
            kw = dict(vocab_size=512, d_model=32, n_heads=2, mlp_hidden=64, window=8, max_seq_len=48, **kw)
            torch.manual_seed(0)
            a = G.FusionDance(G.ModelConfig(**kw)).eval()
            torch.manual_seed(0)
            b = M.make_model(M.ModelConfig(**kw)).eval()
            self.assertIs(type(b), M.FusionDance)
            sa, sb = a.state_dict(), b.state_dict()
            self.assertEqual(list(sa), list(sb), name)
            for k in sa:
                self.assertTrue(torch.equal(sa[k], sb[k]), f"{name}: {k}")
            x = torch.randint(0, 512, (2, 48))
            with torch.no_grad():
                la, lossa = a(x, x)
                lb, lossb = b(x, x)
            self.assertTrue(torch.equal(la, lb), name)
            self.assertEqual(lossa.item(), lossb.item())
            self.assertEqual(a.flops_per_token(48), b.flops_per_token(48))
            self.assertEqual(a.param_counts(), b.param_counts())

    def test_unknown_field_still_rejected(self):
        with self.assertRaises(TypeError):
            M.ModelConfig(not_a_field=1)


class TestPerPassLoss(unittest.TestCase):
    def test_shapes_and_weighting(self):
        for dec in ("ar", "parallel_refine"):
            for r in (1, 2, 4):
                torch.manual_seed(0)
                m = tiny_patched(decoder=dec, refine_passes=r, refine_loss="linear").train()
                x = torch.randint(0, 256, (2, 32))
                y = torch.randint(0, 256, (2, 32))
                outs, loss = m(x, y, return_all=True)
                self.assertEqual(len(outs), r)
                for _, lg in outs:
                    self.assertEqual(tuple(lg.shape), (2, 32, 256))
                pl = m.last_aux["pass_loss"]
                self.assertEqual(len(pl), r)
                self.assertEqual(tuple(m.last_aux["slot_loss"].shape), (4,))
                w = m.loss_weights(r)
                self.assertAlmostEqual(sum(w), 1.0, places=6)
                self.assertTrue(all(b > a for a, b in zip(w, w[1:])))
                self.assertAlmostEqual(loss.item(), sum(wk * p.item() for wk, p in zip(w, pl)), places=4)
                loss.backward()
                self.assertTrue(all(p.grad is not None for n, p in m.named_parameters()
                                    if not (r == 1 and n.startswith("feedback"))), dec)
                m.eval()
                _, l_eval = m(x, y)
                self.assertAlmostEqual(l_eval.item(), pl[-1].item(), places=4)  # eval loss = final pass

    def test_loss_specs(self):
        m = tiny_patched(refine_passes=3, refine_loss="last")
        self.assertEqual(m.loss_weights(3), [0.0, 0.0, 1.0])
        m = tiny_patched(refine_passes=3, refine_loss=[1, 1, 2])
        self.assertEqual(m.loss_weights(3), [0.25, 0.25, 0.5])

    def test_unshared_has_separate_weights_equal_flops(self):
        a = tiny_patched(refine_passes=2, refine_shared=True)
        b = tiny_patched(refine_passes=2, refine_shared=False)
        self.assertEqual(len(a.passes), 1)
        self.assertEqual(len(b.passes), 2)
        self.assertEqual(a.flops_per_token(32), b.flops_per_token(32))
        self.assertGreater(b.param_counts()["total"], a.param_counts()["total"])

    def test_gated_update_bounded(self):
        """contractive: alpha <= alpha_max; the update is a convex step toward f(u)."""
        m = tiny_patched(refine_passes=2, refine_update="contractive")
        with torch.no_grad():
            m.gate.fill_(50.0)
        self.assertLessEqual(torch.sigmoid(m.gate).max().item() * m.cfg.refine_alpha_max, 0.5 + 1e-6)


def staged_configs():
    return sorted(CFG_DIR.glob("*.json"))


class TestStagedConfigs(unittest.TestCase):
    def test_configs_exist_and_name_budget(self):
        files = staged_configs()
        ids = {json.loads(p.read_text())["experiment"] for p in files}
        for want in ("E12", "E13a", "E13b", "E13c", "E13d", "E14a", "E14b", "E14c", "E14d", "E15a", "E15b", "E15c",
                     "E16a", "E16b"):
            self.assertIn(want, ids)
        for p in files:
            c = json.loads(p.read_text())
            tr, m = c["train"], c["model"]
            self.assertIn("max_bytes", tr, p.name)
            self.assertNotIn("max_tokens", tr, p.name)
            self.assertEqual(tr["seq_len_unit"], "bytes")
            self.assertEqual(m["max_seq_len"], tr["seq_len"], f"{p.name}: max_seq_len must be the byte context")
            self.assertEqual(tr["seq_len"], 4096)
            self.assertEqual(m["vocab_size"], 256)
            self.assertIn(tr["max_bytes"], (200_000_000, 20_000_000))

    def test_full_width_build_rope_lengths(self):
        """Real widths: RoPE tables cover the byte context (flat) / T+P bytes and T/P+1 patches (patched)."""
        table = {r["file"]: r for r in json.loads((ROOT / "experiments/byte_model/flops-table.json").read_text())["rows"]}
        for p in staged_configs():
            m = json.loads(p.read_text())["model"]
            mm = {k: v for k, v in m.items() if k != "swa_impl"}
            model = M.make_model(M.ModelConfig(**mm))
            T = m["max_seq_len"]
            if m.get("patch_size"):
                self.assertGreaterEqual(model.lrope.cos.shape[2], T + m["patch_size"])
                self.assertGreaterEqual(model.grope.cos.shape[2], T // m["patch_size"] + 1)
                self.assertTrue(all(not isinstance(x, M.BitLinear) for x in model.encoder.modules()))
                self.assertTrue(all(not isinstance(x, M.BitLinear) for x in model.passes.modules()))
                if m.get("ternary"):
                    self.assertTrue(any(isinstance(x, M.BitLinear) for x in model.body.modules()))
            else:
                self.assertGreaterEqual(model.rope.cos.shape[2], T)
            rel = f"experiments/staged-configs/byte/{p.name}"
            self.assertEqual(model.param_counts()["total"], table[rel]["params_total"], p.name)

    def test_forward_at_full_T_tiny_width(self):
        """Each staged model family runs a forward at its full T = 4,096 bytes (tiny widths, B = 1)."""
        seen = set()
        for p in staged_configs():
            m = json.loads(p.read_text())["model"]
            key = json.dumps({k: v for k, v in m.items() if k not in ("d_model", "n_heads", "mlp_hidden", "d_local",
                                                                      "local_heads", "local_mlp_hidden")}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            mm = {k: v for k, v in m.items() if k != "swa_impl"}
            mm.update(TINY_G)
            if mm.get("patch_size"):
                mm.update(TINY_L)
            torch.manual_seed(0)
            model = M.make_model(M.ModelConfig(**mm)).eval()
            T = m["max_seq_len"]
            x = torch.randint(0, 256, (1, T))
            with torch.no_grad():
                lg, loss = model(x, x)
            self.assertEqual(tuple(lg.shape), (1, T, 256), p.name)
            self.assertTrue(math.isfinite(loss.item()), p.name)


class TestDataset(unittest.TestCase):
    D = DATA / "datasets" / "code-py-v1-bytes"
    MAN = ROOT / "experiments" / "byte_model" / "dataset-bytes.manifest.json"

    @unittest.skipUnless((DATA / "datasets" / "code-py-v1-bytes" / "meta.json").exists(), "byte dataset not built")
    def test_val_roundtrip_and_manifest(self):
        man = json.loads(self.MAN.read_text())
        self.assertTrue(man["verification"]["all_pass"])
        val = np.fromfile(self.D / "val.bin", dtype=np.uint8)
        off = np.load(self.D / "val_doc_offsets.npy")
        self.assertEqual(len(val), man["splits"]["val"]["positions"])
        with open(DATA / "datasets/code-py-v1/docs_val.jsonl", encoding="utf-8") as f:
            texts = [json.loads(ln)["text"] for ln in f]
        raw = val.tobytes()
        self.assertEqual([raw[off[i]:off[i + 1] - 1].decode("utf-8") for i in range(len(texts))], texts)
        self.assertTrue(np.array_equal(np.flatnonzero(val == 0) + 1, off[1:]))
        cm = json.loads((ROOT / "experiments/data_integrity/clean-val.manifest.json").read_text())
        for name, lst in cm["lists"].items():
            self.assertEqual(man["verification"]["clean_val_lists"][name]["n"], lst["n"])
            self.assertEqual(man["verification"]["clean_val_lists"][name]["bytes_content"],
                             int((np.diff(off) - 1)[lst["ids"]].sum()))
        import hashlib
        self.assertEqual(hashlib.sha256(raw).hexdigest(), man["splits"]["val"]["files"]["val.bin"]["sha256"])

    @unittest.skipUnless((DATA / "datasets" / "code-py-v1-bytes" / "meta.json").exists(), "byte dataset not built")
    def test_meta_trainer_contract(self):
        meta = json.loads((self.D / "meta.json").read_text())
        self.assertEqual((meta["unit"], meta["dtype"], meta["vocab_size"], meta["eos_id"]), ("bytes", "uint8", 256, 0))
        sp = meta["splits"]["train"]
        self.assertAlmostEqual(sp["chars_per_token"], sp["chars"] / sp["positions"], places=5)


class TestAccounting(unittest.TestCase):
    def test_e1_reference_unchanged(self):
        e1 = M.FusionDance(M.ModelConfig(prelude=["swa", "gdn", "gdn", "swa", "gdn", "attn"], core=[], coda=[]))
        self.assertEqual(e1.flops_per_token(1024), 91_027_800)  # experiments/design/staged-table.md

    def test_against_analysis_section3(self):
        from experiments.byte_model.build_configs import T, model_of, patched
        flat = M.FusionDance(M.ModelConfig(vocab_size=256, prelude=["swa", "gdn", "gdn", "swa", "gdn", "attn"],
                                           core=[], coda=[], window=1024, max_seq_len=T))
        self.assertAlmostEqual(flat.flops_per_token(T) * M.BYTES_PER_TOKEN / 1e6, 325, delta=325 * 0.02)
        for r, ref in ((1, 95), (2, 108), (4, 135)):
            got = model_of(patched(4, refine_passes=r)).flops_per_token(T) * M.BYTES_PER_TOKEN / 1e6
            self.assertAlmostEqual(got, ref, delta=ref * 0.03, msg=f"r={r}")

    def test_table_matches_recomputation(self):
        from experiments.byte_model.build_configs import model_of
        tab = json.loads((ROOT / "experiments/byte_model/flops-table.json").read_text())
        for row in tab["rows"]:
            m = json.loads((ROOT / row["file"]).read_text())["model"]
            if m.get("refine_max", 0) > m.get("refine_min", 0):
                continue
            model = model_of(m)
            self.assertAlmostEqual(model.flops_per_token(4096), row["flops_per_byte_train"], delta=1.0)
            self.assertAlmostEqual(row["x_E1"], row["flops_per_token_equiv_train"] / 91_027_800, places=6)

    def test_patch_scaling(self):
        """Global cost per byte falls as 1/P; local cost per byte is independent of P (except patch_in/up)."""
        from experiments.byte_model.build_configs import T, model_of, patched
        b = {P: model_of(patched(P)).flops_breakdown(T) for P in (2, 4, 8)}
        self.assertAlmostEqual(b[2]["decoder"], b[8]["decoder"], delta=1.0)
        self.assertGreater(b[2]["global"] / b[4]["global"], 1.9)


if __name__ == "__main__":
    unittest.main()
