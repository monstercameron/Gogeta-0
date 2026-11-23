"""Assembly compensations (v0.4.109): defaults are bit-identical, and each new switch does what it says.

CPU-only: attention/SWA blocks (GDN needs fla's GPU kernel; the GPU numerical suite covers it)."""
import copy

import torch

from gogeta.model import BitLinear, FusionDance, ModelConfig, set_ternary_quant

BASE = dict(vocab_size=97, d_model=32, n_heads=4, mlp_hidden=64, prelude=["swa"], core=["attn", "swa"], coda=["attn"],
            loops=3, loops_min=1, loops_max=4, window=8, max_seq_len=32)


def build(**kw):
    torch.manual_seed(0)
    return FusionDance(ModelConfig(**{**BASE, **kw})).eval()


def out(m, loops=None):
    torch.manual_seed(1)
    x = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        return m(x, loops=loops)[0]


def test_defaults_omitted_from_config_dict():
    d = ModelConfig(**BASE).to_dict()
    for k in ("loop_embed", "inject_contractive", "ternary_acts", "ternary_fp_ends"):
        assert k not in d
    assert ModelConfig(**BASE, loop_embed=True).to_dict()["loop_embed"] is True


def test_loop_embed_zero_init_is_identity_and_adds_params():
    a, b = build(), build(loop_embed=True)
    assert torch.equal(out(a), out(b))
    assert sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters()) == 4 * 32
    with torch.no_grad():
        b.loop_emb[1].fill_(0.5)
    assert not torch.equal(out(a), out(b))
    assert out(b, loops=9).shape == out(a, loops=9).shape  # passes beyond loops_max reuse the last vector


def test_inject_contractive_changes_recurrence_and_is_bounded_scalar():
    a, b = build(), build(inject_contractive=True)
    lam = torch.sigmoid(b.lam_logit)
    assert torch.allclose(lam, torch.full_like(lam, 0.9))
    assert not torch.allclose(out(a), out(b))
    with torch.no_grad():
        b.lam_logit.fill_(30.0)  # lam -> 1 recovers the additive update
    assert torch.allclose(out(a), out(b), atol=1e-5)


def test_ternary_acts_off_is_weight_only():
    m = build(ternary=True, ternary_acts=False)
    lins = [x for x in m.modules() if isinstance(x, BitLinear)]
    assert lins and all(not x.quant_acts for x in lins)
    full = build(ternary=True)
    assert all(x.quant_acts for x in full.modules() if isinstance(x, BitLinear))
    assert not torch.allclose(out(m), out(full))


def test_ternary_fp_ends_keeps_prelude_coda_inject_full_precision():
    m = build(ternary=True, ternary_fp_ends=True)
    assert not any(isinstance(x, BitLinear) for x in m.prelude.modules())
    assert not any(isinstance(x, BitLinear) for x in m.coda.modules())
    assert not isinstance(m.inject, BitLinear)
    assert any(isinstance(x, BitLinear) for x in m.core.modules())


def test_set_ternary_quant_off_equals_full_precision_model():
    t, f = build(ternary=True), build()
    f.load_state_dict(t.state_dict())
    assert set_ternary_quant(t, False) > 0
    assert torch.equal(out(t), out(f))
    assert set_ternary_quant(t, True) > 0 and not torch.equal(out(t), out(f))
    assert set_ternary_quant(t, True) == 0


def test_contractive_loop_embed_train_step_backprops():
    m = build(loop_embed=True, inject_contractive=True, ternary=True, ternary_acts=False).train()
    x = torch.randint(0, 97, (2, 16))
    _, loss = m(x, targets=x, loops=3)
    loss.backward()
    assert m.loop_emb.grad is not None and m.lam_logit.grad is not None
