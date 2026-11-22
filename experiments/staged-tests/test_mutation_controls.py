"""Negative controls (A6): a test suite that cannot fail proves nothing. Each test here injects a deliberate bug of the kind
this project has actually had or could plausibly have, and asserts that the staged numerical tests CATCH it.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/staged-tests/test_mutation_controls.py
"""
from __future__ import annotations

from _harness import run_module

import ast
from unittest import mock

import torch

import numref as R
import test_api_contract as AC
import test_reference_blocks as TB
from gogeta import model as gm


def _caught(fn, *a, **k):
    try:
        fn(*a, **k)
    except (AssertionError, RuntimeError):
        return True
    return False


def test_scanner_flags_the_c2_call():
    """The exact C2 bug: beta as the 5th positional argument of fused_recurrent_gated_delta_rule."""
    src = ("from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as frg\n"
           "def f(q,k,v,g,beta):\n    return frg(q, k, v, g, beta, scale=1.0)\n"
           "def ok(q,k,v,g,beta):\n    return frg(q, k, v, g=g, beta=beta, scale=1.0)\n"
           "def typo(q,k,v,g,beta):\n    return frg(q, k, v, g=g, beta=beta, use_beta_sigmoid_in_kernal=True)\n")
    sc = AC._Scan()
    sc.visit(ast.parse(src))
    over = [(ln, n) for ln, short, n, kws, star, ds in sc.calls if n > AC._safe_n(short)]
    assert len(over) == 1 and over[0][1] == 5, f"scanner missed the C2 call: {sc.calls}"
    named = AC._named_params("fused_recurrent_gated_delta_rule")
    assert [k for _, _, _, kws, _, _ in sc.calls for k in kws if k not in named] == ["use_beta_sigmoid_in_kernal"]


def test_signature_pin_catches_reordering():
    real = AC._params

    def swapped(fn):
        ps = real(fn)
        if getattr(fn, "__name__", "") == "fused_recurrent_gated_delta_rule":
            ps = list(ps)
            i, j = [p.name for p in ps].index("gk"), [p.name for p in ps].index("beta")
            ps[i], ps[j] = ps[j], ps[i]
        return ps
    with mock.patch.object(AC, "_params", swapped):
        assert _caught(AC.test_signatures_pinned), "pin did not notice gk/beta swapped"


def test_model_vs_reference_catches_kernel_misuse():
    """C2-style bug inside the model: beta handed to the kernel in the decay slot."""
    idx = torch.randint(0, 48, (1, 16))
    m = TB.tiny_model(prelude=["gdn"], core=[], coda=[], patch_kernel=False)
    good = R.make_cpu_kernel()
    for mod in m.modules():
        if isinstance(mod, gm.GatedDeltaNet):
            mod._kernel = lambda q, k, v, g, beta, **kw: good(q, k, v, torch.log(beta), torch.ones_like(g), **kw)
    assert _caught(TB._compare_model, m, idx, 1e-5, label="mutant"), "swapped g/beta not detected"


def test_swa_mask_off_by_one_is_caught():
    orig = gm.Attention._mask

    def mutant(self, T, device):
        i = torch.arange(T, device=device)
        d = i[:, None] - i[None, :]
        return (d >= 0) & (d <= self.window)     # window+1 keys
    with mock.patch.object(gm.Attention, "_mask", mutant):
        assert _caught(TB.test_known_model_swa_mask_and_influence_support)
        assert _caught(TB.test_model_matches_reference_every_block_kind)
    assert orig is gm.Attention._mask


def test_missing_ste_is_caught():
    def no_ste(self, x):
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        w = (w / scale).round().clamp(-1, 1) * scale          # forgot  w + (w_q - w).detach()
        return torch.nn.functional.linear(x, w, self.bias)
    with mock.patch.object(gm.BitLinear, "forward", no_ste):
        assert _caught(TB.test_bitlinear_ste_gradient_is_exactly_the_identity_ste)


def test_wrong_rope_is_caught():
    def mutant(self, x):
        T = x.shape[2]
        c, s = self.cos[:, :, :T].to(x.dtype), self.sin[:, :, :T].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * c + x2 * s, -x1 * s + x2 * c), dim=-1).flatten(-2)   # rotates the other way
    with mock.patch.object(gm.Rotary, "forward", mutant):
        assert _caught(TB.test_model_matches_reference_every_block_kind)


def test_bptt_off_by_one_is_caught():
    idx = torch.randint(0, 48, (1, 16))
    W = torch.randn(1, 16, 48, dtype=torch.float64)
    m = TB.tiny_model(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=5, bptt_loops=3).train()   # reference says 2
    (m(idx, loops=5)[0] * W).sum().backward()
    g_model = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    (R.model_ref(m, idx, loops=5, bptt=2) * W).sum().backward()
    rel = max(TB.maxerr(g_model[n], p.grad) / max(p.grad.abs().max().item(), 1e-12) for n, p in m.named_parameters())
    assert rel > 1e-3, "gradient comparison cannot distinguish bptt=3 from bptt=2"


def test_decay_sign_error_is_caught_by_known_answers():
    q, k, v, g, beta = TB.seq(T=30, K=5, V=3, seed=9)
    o_bad, _ = R.delta_rule_chunked(q, k, v, -g, beta, 1.0, chunk=8)
    o, _ = R.delta_rule_chunked(q, k, v, g, beta, 1.0, chunk=8)
    o_rec, _ = R.delta_rule_recurrent(q, k, v, g, beta, 1.0)
    assert TB.maxerr(o_bad, o_rec) > 1e-3 and TB.maxerr(o, o_rec) < 1e-12


def test_quantisation_precision_loss_is_visible_to_the_budget_check():
    """Rounding a perfect fp64 result through bf16 must exceed the fp32 budget but respect the bf16 one (budget is not vacuous)."""
    import tolerances as TL
    q, k, v, g, beta = TB.seq(T=256, K=8, V=8, seed=1)
    o64, _ = R.delta_rule_chunked(q, k, v, g, beta, 8 ** -0.5, chunk=64)
    err = TL.scaled_err(o64.to(torch.bfloat16), o64)
    assert TL.tol("gdn_recurrent", "fp32", 256, 8) < err < TL.tol("gdn_chunk", "bf16", 256, 8), err


def test_fingerprint_detects_version_and_signature_drift():
    import copy

    import env_fingerprint as EF
    fp = EF.current()
    assert EF.diff(fp, fp) == []
    bumped = copy.deepcopy(fp)
    bumped["packages"]["flash-linear-attention"] = "0.6.1"
    bumped["signatures"]["fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule"] += ", extra=None"
    bumped["source_sha256"]["fla.ops.kda.chunk_kda"] = "0" * 16
    d = EF.diff(fp, bumped)
    assert len(d) == 3, d
    assert EF.digest(fp) != EF.digest(bumped)


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
