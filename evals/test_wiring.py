"""Wiring tests for the eval suite (CPU only, random tiny attn/swa model, < 1 min).

    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta-eval/bin/python -m unittest evals.test_wiring -v   (repo root, WSL)

Checks that lm-eval's requests, run through evals/gogeta_lm.GogetaLM, give exactly the
numbers a hand-written forward pass gives (so bpb from the harness means what we think),
plus the pass@k estimator, the task-name guard and the sandbox self-test.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lm_eval.api.instance import Instance  # noqa: E402

from evals.gogeta_lm import GogetaLM  # noqa: E402
from evals.tasks.gogeta_utils import pass_at_k  # noqa: E402

torch.set_num_threads(2)


class TestHarnessNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm = GogetaLM(random_init="tiny_attn", seed=0, device="cpu", batch_size=4)
        cls.model, cls.tok = cls.lm.model, cls.lm.tokenizer

    def manual_ll(self, ctx_ids: list[int], tgt_ids: list[int]) -> float:
        ids = torch.tensor([ctx_ids + tgt_ids])
        with torch.no_grad():
            logp = F.log_softmax(self.model(ids[:, :-1]).logits.float(), -1)[0]
        tgt = torch.tensor(tgt_ids)
        return float(logp[len(ctx_ids) - 1:].gather(-1, tgt[:, None]).sum())

    def test_rolling_short_doc(self):
        text = "def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3\n"
        ids = self.tok.encode(text)
        (ll,) = self.lm.loglikelihood_rolling([Instance("loglikelihood_rolling", {}, (text,), 0)])
        ll = ll[0] if isinstance(ll, tuple) else ll
        self.assertAlmostEqual(ll, self.manual_ll([self.tok.eos_token_id], ids), places=3)

    def test_rolling_long_doc_two_windows(self):
        text = "".join(f"x_{i} = {i * 7}\n" for i in range(400))  # > 1,024 tokens
        ids = self.tok.encode(text)
        self.assertGreater(len(ids), 1024)
        T = 1024
        # lm_eval.utils.get_rolling_token_windows(context_len=1): first window predicts ids[:T]
        # from [eos] + ids[:T-1]; then disjoint windows of up to T predicted tokens.
        manual = self.manual_ll([self.tok.eos_token_id], ids[:T])
        done = T
        while done < len(ids):
            end = min(len(ids), done + T)
            inp = ids[end - T - 1:end - 1]
            n_pred = end - done
            with torch.no_grad():
                logp = F.log_softmax(self.model(torch.tensor([inp])).logits.float(), -1)[0]
            manual += float(logp[-n_pred:].gather(-1, torch.tensor(ids[done:end])[:, None]).sum())
            done = end
        (ll,) = self.lm.loglikelihood_rolling([Instance("loglikelihood_rolling", {}, (text,), 0)])
        ll = ll[0] if isinstance(ll, tuple) else ll
        self.assertTrue(math.isclose(ll, manual, rel_tol=1e-4), (ll, manual))

    def test_loglikelihood_pair(self):
        ctx, cont = "Write a function.\n\n```python\n", "def f(x):\n    return x\n```"
        ((ll, _greedy),) = self.lm.loglikelihood([Instance("loglikelihood", {}, (ctx, cont), 0)])
        # HFLM._encode_pair: the context's trailing whitespace moves to the continuation, then
        # whole = enc(context + continuation), split at len(enc(stripped context)).
        whole = self.tok.encode(ctx + cont)
        n_ctx = len(self.tok.encode(ctx.rstrip()))
        self.assertAlmostEqual(ll, self.manual_ll(whole[:n_ctx], whole[n_ctx:]), places=3)
        from evals.tasks.gogeta_utils import _scored_text
        self.assertEqual(_scored_text({"_prompt": ctx, "_target": cont}), "\n" + cont)


class TestPieces(unittest.TestCase):
    def test_pass_at_k(self):
        self.assertAlmostEqual(pass_at_k(10, 1, 1), 0.1)
        self.assertEqual(pass_at_k(10, 0, 5), 0.0)
        self.assertEqual(pass_at_k(5, 5, 5), 1.0)
        self.assertAlmostEqual(pass_at_k(20, 2, 10), 1 - math.comb(18, 10) / math.comb(20, 10))

    def test_builtin_tasks_refused(self):
        from evals.run_suite import main
        with self.assertRaises(SystemExit):
            main(["--random-init", "tiny_attn", "--tasks", "humaneval", "--limit", "1"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "sandbox needs Linux")
    def test_sandbox_self_test(self):
        from evals.sandbox import self_test
        rep = self_test(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertTrue(rep["ok"], rep)


if __name__ == "__main__":
    unittest.main()
