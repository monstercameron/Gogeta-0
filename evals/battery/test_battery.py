"""CPU unit tests for the frozen-battery generators (no checkpoint, no GPU, < 30 s).

    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta-eval/bin/python -m unittest evals.battery.test_battery -v
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "evalbattery"))

from evals.battery.state_track import GRID, N_CHOICES, make_cell, make_item  # noqa: E402


class StateTrack(unittest.TestCase):
    def test_gold_matches_python_semantics(self):
        """exec() the generated snippet (our own text) and compare the target variable with the gold choice."""
        for mode, cells in GRID.items():
            for k, d in cells:
                for it in make_cell(mode, k, d, 25, 0):
                    src, query = it["prompt"].rsplit("\n", 1)
                    target = query.split()[1]
                    ns = {}
                    exec(src, {}, ns)
                    self.assertEqual(ns[target], int(it["choices"][it["gold"]]))

    def test_choices(self):
        for it in make_cell("assign", 4, 16, 200, 1) + make_cell("incr", 4, 16, 200, 1):
            self.assertEqual(len(it["choices"]), N_CHOICES)
            self.assertEqual(len(set(it["choices"])), N_CHOICES)
            self.assertTrue(it["n_stale_distractors"] >= 1)

    def test_deterministic_and_seeded(self):
        self.assertEqual(make_item("incr", 4, 16, 0, 7), make_item("incr", 4, 16, 0, 7))
        self.assertNotEqual(make_item("incr", 4, 16, 0, 7)["prompt"], make_item("incr", 4, 16, 1, 7)["prompt"])

    def test_gold_position_balanced(self):
        g = np.bincount([it["gold"] for it in make_cell("assign", 4, 16, 2000, 0)], minlength=4) / 2000
        self.assertTrue(((g > 0.21) & (g < 0.29)).all(), g)


class Mqar(unittest.TestCase):
    def test_batch_semantics(self):
        import torch, mqar
        g = torch.Generator().manual_seed(0)
        P, T, K, V = 8, 48, 20, 20
        x, y = mqar.mqar_batch(32, g, P, T, K, V)
        for b in range(32):
            ctx = {int(x[b, 2 * i]): int(x[b, 2 * i + 1]) for i in range(P)}
            self.assertEqual(len(ctx), P)                                    # distinct keys
            pos = (y[b] != -100).nonzero().flatten().tolist()
            self.assertEqual(len(pos), P)                                    # every key queried exactly once
            self.assertEqual(sorted(int(x[b, p]) for p in pos), sorted(ctx))
            for p in pos:
                self.assertEqual(int(y[b, p]), ctx[int(x[b, p])])            # target = the key's bound value
                self.assertEqual(int(x[b, p + 1]), int(y[b, p]))             # value follows the query key
                self.assertGreater(p, 2 * P - 1)                             # queries only after the context
            self.assertTrue(((x[b] <= K) | (x[b] > K)).all() and int(x.max()) <= K + V)

    def test_eval_stream_disjoint(self):
        import torch, mqar
        a = mqar.mqar_batch(4, torch.Generator().manual_seed(0), 4, 16, 16, 16)[0]
        b = mqar.mqar_batch(4, torch.Generator().manual_seed(10_000), 4, 16, 16, 16)[0]
        self.assertFalse(bool((a == b).all()))

    def test_bootstrap_ci(self):
        import mqar
        rng = np.random.default_rng(0)
        tot = np.full(2048, 16); corr = rng.binomial(16, 0.5, 2048)
        acc, lo, hi = mqar.cluster_bootstrap_ci(corr, tot)
        self.assertLess(hi - lo, 0.03 * 2)                                   # half-width < 3 points at n=2048 seqs


class IterComp(unittest.TestCase):
    def test_selftest(self):
        import itercomp
        self.assertTrue(itercomp.selftest(verbose=False))

    def test_shapes_and_determinism(self):
        import itercomp
        for task in itercomp.SPLITS:
            for split in ("train", "id") + (("gen",) if "gen" in itercomp.SPLITS[task] else ("len2x", "len4x")):
                x1, y1 = itercomp.make_batch(task, 8, 3, split, 5)
                x2, y2 = itercomp.make_batch(task, 8, 3, split, 5)
                self.assertTrue((x1 == x2).all() and (y1 == y2).all())
                self.assertLess(int(x1.max()), itercomp.vocab_size(task))
                self.assertLessEqual(x1.shape[1], itercomp.max_len(task))
                self.assertTrue((y1 != -100).any())

    def test_train_eval_streams_differ(self):
        import itercomp
        a = itercomp.make_batch("graph_hops", 16, 0, "id", 0)[0]
        b = itercomp.make_batch("graph_hops", 16, 10_000, "id", 0)[0]
        self.assertFalse(bool((a == b).all()))


if __name__ == "__main__":
    unittest.main()
