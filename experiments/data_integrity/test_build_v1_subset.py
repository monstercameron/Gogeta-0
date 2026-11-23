"""Tests for build_v1_subset.py on the small synthetic corpus of test_build_v2 (temp dir, CPU only).

    CUDA_VISIBLE_DEVICES= wsl -d Ubuntu-24.04 -- /root/.venvs/gogeta/bin/python -m unittest -v experiments/data_integrity/test_build_v1_subset.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import build_v1_subset as bs  # noqa: E402
import test_build_v2 as t2  # noqa: E402

COUNT = 1500


class BuildV1SubTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = pathlib.Path(cls.tmp.name)
        cls.src = t / "code-py-v1"
        cls.info = t2.make_v1(cls.src)
        cls.o1, cls.o2 = t / "a" / "datasets" / bs.NAME, t / "b" / bs.NAME
        bs.build(cls.src, cls.o1, workers=1, count=COUNT, manifest=t / "m.json")
        bs.build(cls.src, cls.o2, workers=3, count=COUNT)
        cls.t = t

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deterministic_byte_identical(self):
        a, b = t2.sha_dir(self.o1), t2.sha_dir(self.o2)
        self.assertEqual(set(a), {"train.bin", "val.bin", "docs_val.jsonl", "meta.json", "tokenizer.json", "train_keep_ids.npy"})
        self.assertEqual(a, b)

    def test_count_exact_and_unique(self):
        keep = np.load(self.o1 / "train_keep_ids.npy")
        self.assertEqual(len(keep), COUNT)
        self.assertEqual(len(np.unique(keep)), COUNT)
        self.assertTrue((np.diff(keep) > 0).all())
        self.assertEqual(len(t2.read_bin_docs(self.o1 / "train.bin")), COUNT)
        meta = json.loads((self.o1 / "meta.json").read_text())
        self.assertEqual(meta["splits"]["train"]["docs"], COUNT)

    def test_default_count_reads_v2_manifest(self):
        self.assertEqual(bs.v2_train_docs(), 698714)

    def test_not_deduplicated_and_val_excluded(self):
        keep = np.load(self.o1 / "train_keep_ids.npy")
        kept = np.array(self.info["train_group"])[keep]
        # planted families (3 docs each) are not collapsed: some family keeps >= 2 docs
        self.assertTrue(any(int((kept == g).sum()) >= 2 for g in range(t2.N_PLANT)))
        # families that val copies (leaks) are excluded
        leaks = {int(g) for g in self.info["val_group"] if g >= t2.N_PLANT}
        self.assertTrue(leaks)
        self.assertFalse(np.isin(kept, list(leaks)).any())

    def test_docs_match_v1_text_and_val_is_v1(self):
        tr = t2.read_bin_docs(self.o1 / "train.bin")
        keep = np.load(self.o1 / "train_keep_ids.npy")
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(bs.b2.DEFAULT_TOKENIZER))
        for i in (0, 1, COUNT // 2, COUNT - 1):
            self.assertEqual(tr[i].tolist(), tk.encode(self.info["texts_train"][int(keep[i])], add_special_tokens=False).ids)
        self.assertEqual((self.o1 / "docs_val.jsonl").read_bytes(), (self.src / "docs_val.jsonl").read_bytes())
        self.assertEqual(len(t2.read_bin_docs(self.o1 / "val.bin")), len(self.info["texts_val"]))

    def test_readable_by_gogeta_train_loader(self):
        import gogeta.train as gt
        old = gt.DATA
        gt.DATA = self.t / "a"
        try:
            d = gt.TokenData(bs.NAME, 64, 0)
            self.assertEqual(d.train.dtype, np.uint16)
            self.assertEqual(len(d.train), json.loads((self.o1 / "meta.json").read_text())["splits"]["train"]["tokens"])
            self.assertGreater(len(d.val), 64)
            x, y = next(iter(d.val_batches(2, 2, "cpu")))
            self.assertEqual(tuple(x.shape), (2, 64))
            self.assertLess(int(d.train.max()), 8192)
        finally:
            gt.DATA = old

    def test_manifest_schema(self):
        v1 = json.loads((HERE / "dataset.manifest.json").read_text())
        m = json.loads((self.t / "m.json").read_text())
        self.assertEqual(set(m["tokenizer"]), set(v1["tokenizer"]))
        for k in ("created", "dataset", "source", "license", "tokenizer", "prepare_seed", "filter", "splits", "known_issues"):
            self.assertIn(k, m)
        for sp in ("train", "val"):
            self.assertTrue({"docs", "tokens", "chars", "files"} <= set(m["splits"][sp]))


if __name__ == "__main__":
    unittest.main()
