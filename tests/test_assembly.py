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


def test_loop_adapter_zero_init_identity_params_and_flops():
    a, b = build(), build(loop_adapter_rank=4)
    assert torch.equal(out(a), out(b))
    extra = sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters())
    assert extra == 4 * 2 * 2 * 32 * 4 + 32  # loops_max x core blocks x (down + up) x d x rank + adapter norm
    assert b.flops_per_token(16) > a.flops_per_token(16)
    with torch.no_grad():
        b.adapt_up[0, 0].fill_(0.1)
    assert not torch.equal(out(a), out(b))
    assert out(b, loops=9).shape == out(a, loops=9).shape  # passes beyond loops_max reuse the last adapter set
    assert "loop_adapter_rank" not in ModelConfig(**BASE).to_dict()


def test_loop_adapter_backprops():
    m = build(loop_adapter_rank=4).train()
    x = torch.randint(0, 97, (2, 16))
    _, loss = m(x, targets=x, loops=3)
    loss.backward()
    assert m.adapt_up.grad is not None and m.adapt_up.grad.abs().sum() > 0


def test_moe_with_ternary_experts_builds_quantises_and_backprops():
    from gogeta.moe import MoEFFN
    m = build(ternary=True, moe_experts=4, moe_top_k=2, moe_hidden=32).train()
    moes = [x for x in m.modules() if isinstance(x, MoEFFN)]
    assert moes and all(x.ternary for x in moes)
    x = torch.randint(0, 97, (2, 16))
    _, loss = m(x, targets=x, loops=2)
    loss.backward()
    assert moes[0].w_up.grad is not None and moes[0].w_up.grad.abs().sum() > 0
    # quantisation changes the forward pass relative to the same weights in full precision
    m.eval(); a = out(m); set_ternary_quant(m, False); b = out(m)
    assert not torch.allclose(a, b)
    assert set_ternary_quant(m, True) >= len(moes)


def test_moe_core_only_keeps_swiglu_in_prelude_and_coda():
    from gogeta.model import SwiGLU
    from gogeta.moe import MoEFFN
    m = build(moe_experts=4, moe_top_k=2, moe_hidden=32, moe_core_only=True)
    assert all(isinstance(b.mlp, SwiGLU) for b in [*m.prelude, *m.coda])
    assert all(isinstance(b.mlp, MoEFFN) for b in m.core)
    assert "moe_core_only" in ModelConfig(**BASE, moe_experts=4, moe_core_only=True).to_dict()


def test_ternary_scope_mixer_vs_ffn():
    m = build(ternary=True, ternary_mixer=False)
    for b in m.core:
        assert not any(isinstance(x, BitLinear) for x in b.mixer.modules())
        assert any(isinstance(x, BitLinear) for x in b.mlp.modules())
    m = build(ternary=True, ternary_ffn=False)
    for b in m.core:
        assert any(isinstance(x, BitLinear) for x in b.mixer.modules())
        assert not any(isinstance(x, BitLinear) for x in b.mlp.modules())
    d = ModelConfig(**BASE).to_dict()
    assert "ternary_mixer" not in d and "ternary_ffn" not in d and "moe_core_only" not in d


def test_inputs_embeds_and_return_hidden_hooks():
    m = build()
    x = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        a = m(x)[0]
        b = m(None, inputs_embeds=m.embed(x))[0]
        h = m(x, return_hidden=True)
    assert torch.equal(a, b)
    assert h.shape == (2, 16, 32) and torch.allclose(m.head(h), a)


def test_core_mlp_hidden_widens_only_the_core_ffn():  # SQ401
    assert "core_mlp_hidden" not in ModelConfig(**BASE).to_dict()
    a, same = build(), build(core_mlp_hidden=64)        # equal to mlp_hidden: identical model and output
    assert torch.equal(out(a), out(same))
    m = build(core_mlp_hidden=128)
    for b in m.core:
        assert b.mlp.down.in_features == 128
    for b in [*m.prelude, *m.coda]:
        assert b.mlp.down.in_features == 64
    assert m.param_counts()["core"] > a.param_counts()["core"]
    assert m.flops_per_token(16) > a.flops_per_token(16)
    assert ModelConfig(**BASE, core_mlp_hidden=128).to_dict()["core_mlp_hidden"] == 128


def test_gdn_beta_fixed_is_constant_and_leaves_b_without_grad():  # SQ402 (CPU torch path of GatedDeltaNet)
    kw = dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"])
    assert "gdn_beta_fixed" not in ModelConfig(**{**BASE, **kw}).to_dict()
    try:
        build(**kw, gdn_beta_fixed=0.5, gdn_neg_eigval=True)
        raise AssertionError("gdn_beta_fixed + gdn_neg_eigval must raise")
    except ValueError:
        pass
    from gogeta.decode import _gdn_recurrence_torch

    def ref_kernel(q, k, v, g, beta, scale=None, **_):  # torch reference recurrence (fla's kernel needs a GPU)
        B, T, H, K = q.shape
        S = torch.zeros(B, H, K, v.shape[-1])
        o, *_rest = _gdn_recurrence_torch(q.float(), k.float(), v.float(), g.float(), beta.float(), scale, S, T)
        return o.to(q.dtype), None

    def with_ref(m):
        for b in m.core:
            if b.kind == "gdn":
                b.mixer._kernel = ref_kernel
        return m
    a, f = with_ref(build(**kw)), with_ref(build(**kw, gdn_beta_fixed=0.5))
    assert not torch.equal(out(a), out(f))
    f.train()
    x = torch.randint(0, 97, (2, 16))
    _, loss = f(x[:, :-1], x[:, 1:], loops=2)
    loss.backward()
    gdn = [b.mixer for b in f.core if b.kind == "gdn"][0]
    assert gdn.b.weight.grad is None or float(gdn.b.weight.grad.abs().sum()) == 0.0
    assert gdn.q.weight.grad is not None and float(gdn.q.weight.grad.abs().sum()) > 0


def test_core_lr_mult_param_groups():  # SQ404
    from gogeta.train import param_groups
    m = build()
    tc = {"weight_decay": 0.1}
    g1 = param_groups(m, tc)
    assert len(g1) == 2 and all("lr_mult" not in g for g in g1)
    g2 = param_groups(m, {**tc, "core_lr_mult": 2.0})
    n_core = sum(p.numel() for n, p in m.named_parameters() if n.startswith("core."))
    assert sum(p.numel() for g in g2 if g["lr_mult"] == 2.0 for p in g["params"]) == n_core
    assert sum(p.numel() for g in g2 for p in g["params"]) == sum(p.numel() for p in m.parameters())
