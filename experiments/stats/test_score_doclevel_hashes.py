"""SQ345/SQ366: score_doclevel's hash pinning, compatibility check and optional per-doc NLL storage.

    python -m unittest experiments.stats.test_score_doclevel_hashes -v      (repo root; stdlib only, no torch)
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_doclevel as S  # noqa: E402


class TestHashes(unittest.TestCase):
    def test_overlap_hash_is_stable_and_order_sensitive(self):
        a = [0.0, None, 0.5, 1.0]
        self.assertEqual(S.overlap_sha256(a), S.overlap_sha256(list(a)))
        self.assertNotEqual(S.overlap_sha256(a), S.overlap_sha256([0.0, 0.5, None, 1.0]))   # a reorder is detected
        self.assertNotEqual(S.overlap_sha256(a), S.overlap_sha256(a[:-1]))

    def test_file_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"abc")
        try:
            self.assertEqual(S.sha256_file(f.name), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        finally:
            os.unlink(f.name)

    def test_compatible_ok(self):
        r = {"docs_val_sha256": "a" * 64, "overlap_sha256": "b" * 64}
        out = S.check_compatible([dict(r, run="r1"), dict(r, run="r2")], require_hash=True)
        self.assertEqual(out["unhashed"], [])
        self.assertEqual(out["hashes"]["docs_val_sha256"], "a" * 64)

    def test_mismatch_refused(self):
        a = {"run": "r1", "docs_val_sha256": "a" * 64, "overlap_sha256": "b" * 64}
        for key in S.HASH_KEYS:
            b = dict(a, run="r2")
            b[key] = "c" * 64
            with self.assertRaises(S.HashMismatch):
                S.check_compatible([a, b])

    def test_old_files_still_readable(self):
        old = {"run": "old", "bpc": {"clean80": 0.25}}          # pre-SQ345 file: no hashes
        new = {"run": "new", "docs_val_sha256": "a" * 64, "overlap_sha256": "b" * 64}
        out = S.check_compatible([old, new])
        self.assertEqual(out["unhashed"], ["old"])
        self.assertEqual(S.check_compatible([old, dict(old, run="old2")])["hashes"], {})
        with self.assertRaises(S.HashMismatch):
            S.check_compatible([old, new], require_hash=True)

    def test_selftest_still_passes(self):
        self.assertEqual(S.selftest(), 0)


class TestStoreNll(unittest.TestCase):
    def _fake(self):
        import torch
        torch.manual_seed(0)
        emb = torch.nn.Embedding(64, 64)
        return (lambda x: emb(x)), (lambda t: [ord(c) % 50 + 2 for c in t])

    def test_perdoc_matches_k8_totals_and_sums(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "checks"))
            sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
            import k8_external_reference as K8
        except Exception as e:
            self.skipTest(f"K8 not importable here: {e}")
        import math
        fwd, enc = self._fake()
        docs = ["abc" * 5, "hello world " * 40, "x", "def f(x):\n    return x\n" * 30]
        for window in (1024, 64):
            a = K8.score(fwd, enc, 0, docs, window, "cpu")
            b = S.score_perdoc(fwd, enc, 0, docs, window, "cpu")
            for k in ("bits_per_char", "nats_per_token", "tokens", "chars"):
                self.assertAlmostEqual(a[k], b[k], places=9, msg=k)
            self.assertEqual(len(b["nll_doc"]), len(docs))
            self.assertEqual(sum(b["tok_doc"]), b["tokens"])
            self.assertAlmostEqual(sum(b["nll_doc"]) / math.log(2) / b["chars"], b["bits_per_char"], places=9)

    def test_npz_roundtrip(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy missing")
        d = {"clean80": {"idx": [0, 2, 5], "nll_nats": [1.5, 2.25, 3.0], "tok": [10, 20, 30]}}
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "r.nll.npz")
            S.save_nll_npz(p, d)
            r = S.load_nll_npz(p)
        self.assertEqual(list(r["clean80"]["idx"]), [0, 2, 5])
        self.assertEqual(list(r["clean80"]["nll_nats"]), [1.5, 2.25, 3.0])
        self.assertEqual(list(r["clean80"]["tok"]), [10, 20, 30])


if __name__ == "__main__":
    unittest.main(verbosity=2)
