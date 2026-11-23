"""CPU tests for the coarse-to-fine byte model (gogeta/c2f.py). Plain functions (pytest is not installed):

    wsl: CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python tests/test_c2f.py

GDN runs on the token-by-token CPU reference (experiments/byte_model/cpu_ref.py), fp32, so causality is bit-exact.
Causality criterion (notes/analysis-coarse-to-fine.md section 2, written before running): row j of the logits predicts
byte j+1; perturbing input byte s must leave every row j < s BITWISE unchanged in every refinement pass, for P in
{2, 4} and r_b in {1, 2}, with perturbations at patch starts, middles and ends; row s itself must change (test has
teeth); the deliberately broken no-shift variant must leak.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.byte_model import cpu_ref  # noqa: E402

cpu_ref.install()
from gogeta.c2f import BlockLocalAttention, C2FConfig, C2FModel  # noqa: E402
from gogeta.model import BitLinear, FusionDance  # noqa: E402

torch.set_num_threads(1)

G_DENSE = dict(d_model=32, n_heads=2, mlp_hidden=64, prelude=["swa"], core=[], coda=["gdn", "attn"], window=4)
G_ALL = dict(d_model=32, n_heads=2, mlp_hidden=64, window=4, prelude=["swa"], core=["gdn", "swa"], coda=["attn"],
             loops=2, loops_min=2, loops_max=2, bptt_loops=2, ternary=True, ternary_fp_ends=True,
             inject_contractive=True, moe_experts=4, moe_top_k=2, moe_hidden=32, moe_core_only=True,
             ternary_mixer=False)
LOCAL = dict(d_local=16, local_heads=2, local_mlp_hidden=32, local_window=6, local_encoder=["conv", "gdn"],
             refine_blocks=["conv", "lswa"])


def build(P=4, r=2, glob=None, T=48, seed=0, **kw) -> C2FModel:
    torch.manual_seed(seed)
    cfg = C2FConfig(max_seq_len=T, patch_size=P, refine_passes=r, global_model=dict(glob or G_DENSE), **{**LOCAL, **kw})
    return C2FModel(cfg).eval()


def all_logits(m, x):
    with torch.no_grad():
        return m(x, return_all=True)[0]


def positions(P, T):
    ps = {0, P // 2, P - 1, P, P + P // 2, 2 * P - 1, 3 * P, T - P, T - P // 2, T - 1}
    return sorted(s for s in ps if 0 <= s < T)


def perturbation_check(m, T=48, batch=2, seed=0):
    """Returns (n_leaks, n_vacuous, n_perturbations)."""
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randint(0, 256, (batch, T), generator=g)
    base = all_logits(m, x)
    leaks = vacuous = 0
    ps = positions(m.P, T)
    for s in ps:
        xp = x.clone()
        xp[:, s] = (xp[:, s] + 1 + torch.randint(0, 255, (batch,), generator=g)) % 256
        pert = all_logits(m, xp)
        for a, b in zip(base, pert):
            if not torch.equal(a[:, :s], b[:, :s]):
                leaks += 1
        if torch.equal(base[-1][:, s], pert[-1][:, s]):
            vacuous += 1
    return leaks, vacuous, len(ps)


# ----------------------------------------------------------------------------- causality
def test_causality_perturbation():
    rows = []
    for glob_name, glob in (("dense", G_DENSE), ("all_parts", {**G_ALL, "moe_impl": "dense"}), ("all_parts_gather", G_ALL)):
        for P in (2, 4):
            for r in (1, 2):
                for contractive in ((False, True) if r == 2 else (False,)):
                    m = build(P=P, r=r, glob=glob, refine_contractive=contractive, refine_aux_weight=0.5)
                    leaks, vac, n = perturbation_check(m)
                    rows.append((glob_name, P, r, contractive, leaks, vac, n))
                    assert leaks == 0, f"LEAK {glob_name} P={P} r={r} contractive={contractive}: {leaks}"
                    assert vac == 0, f"VACUOUS {glob_name} P={P} r={r}: row s unchanged in {vac}/{n}"
    for row in rows:
        print("   causal OK  global=%-17s P=%d r_b=%d contractive=%-5s perturbations=%d" % (row[0], row[1], row[2],
                                                                                         row[3], row[6]))


def test_broken_noshift_control_leaks():
    for P in (2, 4):
        m = build(P=P, r=2)
        m._broken_noshift = True
        leaks, _, n = perturbation_check(m)
        assert leaks > 0, f"the no-shift control did not leak at P={P}: the perturbation test has no teeth"
        print(f"   broken no-shift control P={P}: {leaks} leaking (pass, perturbation) pairs of {2 * n}, as required")


def test_block_local_swa_matches_dense_window():
    """lswa (block-local) equals gogeta.model's dense-mask SWA with the same weights."""
    m = build(P=4, r=1)
    blk = m.passes[0][1]
    assert isinstance(blk.mixer, BlockLocalAttention)
    from gogeta.model import Attention
    dense = Attention(m.rcfg, m.rcfg.window, m.lrope)
    dense.load_state_dict(blk.mixer.state_dict())
    x = torch.randn(2, 37, 16)
    with torch.no_grad():
        torch.testing.assert_close(blk.mixer(x), dense(x), rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------- shapes / reductions
def test_shapes_and_padding():
    m = build(P=4, r=2)
    for T in (48, 45, 3):
        x = torch.randint(0, 256, (3, T))
        with torch.no_grad():
            lg, loss = m(x, torch.randint(0, 256, (3, T)))
            outs, _ = m(x, return_all=True)
        assert lg.shape == (3, T, 256) and loss.ndim == 0 and torch.isfinite(loss)
        assert len(outs) == 2 and all(o.shape == (3, T, 256) for o in outs)
    # a padded prompt gives the same logits as the prefix of the full sequence (generation path)
    x = torch.randint(0, 256, (2, 48))
    with torch.no_grad():
        full = m(x)[0]
        part = m(x[:, :45])[0]
    assert torch.equal(full[:, :44], part[:, :44])
    # L(r) curve: a shared decoder runs any pass count
    with torch.no_grad():
        assert m(x, passes=4)[0].shape == (2, 48, 256)


def test_rb1_is_plain_single_pass_decoder():
    """r_b = 1 (contractive off) builds no inject / lam and computes exactly head(norm(blocks(ctx + e)))."""
    m = build(P=4, r=1)
    assert not hasattr(m, "inject") and not hasattr(m, "lam_logit") and len(m.passes) == 1
    x = torch.randint(0, 256, (2, 48))
    with torch.no_grad():
        got = m(x)[0]
        B, T, P = 2, 48, 4
        e = m.embed(x)
        h = e
        for b in m.encoder:
            h = b(h)
        g = m.glob(None, inputs_embeds=m.patch_in(h.reshape(B, T // P, P * 16)), return_hidden=True)
        gs = torch.cat([m.patch_start.expand(B, 1, -1), g], dim=1)
        t = torch.arange(T)
        y = m.up(gs)[:, (t + 1) // P] + m.slot_emb((t + 1) % P) + e
        for b in m.passes[0]:
            y = b(y)
        want = m.head(m.norm_out(y))
    assert torch.equal(got, want)
    # contractive is a pass >= 1 option only: with r_b = 1 it changes nothing (same params, same outputs)
    m2 = build(P=4, r=1, refine_contractive=True)
    assert sum(p.numel() for p in m2.parameters()) == sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        assert torch.equal(m2(x)[0], got)


# ----------------------------------------------------------------------------- training path
def test_backward_through_all_parts():
    """Loops + ternary + MoE core-only global, contractive r_b = 2 decoder, per-pass loss: every used parameter
    gets a finite, non-zero gradient; the MoE aux loss passes through; ternary layers exist where configured."""
    m = build(P=4, r=2, glob=G_ALL, refine_contractive=True, refine_aux_weight=0.5).train()
    x, y = torch.randint(0, 256, (2, 48)), torch.randint(0, 256, (2, 48))
    logits, loss, aux = m(x, y, return_aux=True)
    assert aux is not None and aux["loss"].ndim == 0 and aux["loss"].requires_grad
    assert set(m.last_aux["pass_ids"]) == {0, 1} and m.last_aux["slot_loss"].shape == (4,)
    (loss + aux["loss"]).backward()
    unused = {"glob.embed.weight", "glob.head.weight"}
    missing = [n for n, p in m.named_parameters() if p.requires_grad and n not in unused
               and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0)]
    assert not missing, f"no / bad gradient: {missing}"
    # precision scopes: global core ternary (mixers full precision: ternary_mixer False, FFN = ternary MoE experts),
    # global prelude/coda/inject full precision (ternary_fp_ends), byte layers full precision (defaults)
    assert any(mod.ternary for mod in m.glob.core.modules() if hasattr(mod, "w_up"))
    assert not any(isinstance(mod, BitLinear) for mod in m.glob.prelude.modules())
    assert not any(isinstance(mod, BitLinear) for mod in [*m.encoder.modules(), *m.passes.modules()])
    # explicit byte-decoder precision switch
    mt = build(P=4, r=2, decoder_ternary=True)
    assert any(isinstance(mod, BitLinear) for mod in mt.passes.modules()) and isinstance(mt.inject, BitLinear)
    assert not any(isinstance(mod, BitLinear) for mod in mt.encoder.modules())
    # dense global: no aux
    assert build(P=4, r=1)(x, y, return_aux=True)[2] is None


def test_train_loss_weights():
    m = build(P=4, r=2, refine_aux_weight=0.5).train()
    x, y = torch.randint(0, 256, (2, 48)), torch.randint(0, 256, (2, 48))
    torch.manual_seed(0)
    _, loss = m(x, y)
    p0, p1 = (v.item() for v in m.last_aux["pass_loss"])
    assert abs(loss.item() - (p1 + 0.5 * p0)) < 1e-5
    m.eval()
    with torch.no_grad():
        _, ev = m(x, y)
    assert abs(ev.item() - m.last_aux["ce_last"].item()) < 1e-6


# ----------------------------------------------------------------------------- init scales (review I3)
def _res_std(mods):
    ws = [mod.weight.detach().flatten() for mod in mods.modules()
          if isinstance(mod, torch.nn.Linear) and getattr(mod, "is_residual_out", False)]
    return torch.cat(ws).std().item(), sum(w.numel() for w in ws)


def test_decoder_residual_init_uses_its_own_depth():
    """Decoder residual outputs: 0.02/sqrt(2 * r_b * len(refine_blocks)); encoder: 0.02/sqrt(2 * len(encoder));
    the shared r_b = 2 and the unshared 2-pass controls (incl. the parameter-matched one) get the same scale; the
    global model keeps FusionDance's own scale (its effective depth)."""
    big = dict(d_local=128, local_heads=2, local_mlp_hidden=512, refine_blocks=["lswa"], local_encoder=["conv", "lswa"])
    gbig = dict(d_model=128, n_heads=2, mlp_hidden=512, prelude=["swa"], core=["swa"], coda=["attn"], loops=3,
                loops_min=1, loops_max=3, bptt_loops=3, window=8)
    arms = {"r1": dict(r=1), "r2_shared": dict(r=2), "r2_unshared": dict(r=2, refine_shared=False),
            "r2_unshared_parammatched": dict(r=2, refine_shared=False, refine_mlp_hidden=170)}
    for name, kw in arms.items():
        m = build(P=4, glob=gbig, T=64, **{**big, **kw})
        depth = kw["r"] * 1
        want = 0.02 / math.sqrt(2 * depth)
        got, n = _res_std(m.passes)
        if kw["r"] > 1:
            got_inj = m.inject.weight.std().item()
            assert abs(got_inj - want) / want < 0.05, (name, "inject", got_inj, want)
        assert abs(got - want) / want < 0.05, (name, got, want, n)
        enc_got, _ = _res_std(m.encoder)
        enc_want = 0.02 / math.sqrt(2 * 2)
        assert abs(enc_got - enc_want) / enc_want < 0.05, (name, "encoder", enc_got, enc_want)
        g_got, _ = _res_std(m.glob)
        g_want = 0.02 / math.sqrt(2 * m.glob.effective_depth())   # 1 + 1 + 3 * 1 = 5
        assert abs(g_got - g_want) / g_want < 0.05, (name, "global", g_got, g_want)
        print(f"   init {name:25s} decoder residual std {got:.5f} (want {want:.5f}, n={n}); "
              f"encoder {enc_got:.5f} (want {enc_want:.5f}); global {g_got:.5f} (want {g_want:.5f})")


# ----------------------------------------------------------------------------- accounting
def test_param_counts_add_up():
    m = build(P=4, r=2, glob=G_ALL)
    pc = m.param_counts()
    total = sum(p.numel() for p in m.parameters())
    assert pc["total"] == total
    assert pc["embedding"] + pc["encoder"] + pc["patch_io"] + pc["global"] + pc["global_unused_embed"] \
        + pc["decoder"] + pc["head_norm"] == total
    assert pc["decoder_blocks"] == pc["decoder_block_per_pass"]          # shared: one block set
    mu = build(P=4, r=2, glob=G_ALL, refine_shared=False)
    assert mu.param_counts()["decoder_blocks"] == 2 * pc["decoder_block_per_pass"]
    assert pc["active_total"] < pc["total"]                               # MoE: inactive experts excluded


def test_flops_consistent_with_byte_flops_table():
    """Encoder, patch_in, global and up FLOPs per byte equal model_bytes.PatchedByteLM's (the rule behind
    experiments/byte_model/flops-table.md) for E14b's shapes; decoder/head differ by design (documented)."""
    from experiments.byte_model.model_bytes import ModelConfig as OldCfg, make_model as old_make
    e1 = dict(d_model=384, n_heads=6, mlp_hidden=1024, prelude=["swa", "gdn", "gdn", "swa", "gdn", "attn"], core=[],
              coda=[], window=256)
    old = old_make(OldCfg(vocab_size=256, max_seq_len=4096, patch_size=4, d_local=192, local_heads=3,
                          local_mlp_hidden=512, local_window=64, local_encoder=["conv", "gdn"],
                          refine_blocks=["conv", "gdn"], refine_passes=2, **e1))
    torch.manual_seed(0)
    new = C2FModel(C2FConfig(max_seq_len=4096, patch_size=4, d_local=192, local_heads=3, local_mlp_hidden=512,
                             local_window=64, local_encoder=["conv", "gdn"], refine_blocks=["conv", "gdn"],
                             refine_passes=2, global_model=e1))
    fo, fn = old.flops_breakdown(4096), new.flops_breakdown(4096)
    for k in ("encoder", "patch_in", "global", "up"):
        assert abs(fo[k] - fn[k]) < 1e-6 * fo[k], (k, fo[k], fn[k])
    assert abs(fn["global"] - 18.04e6) < 0.01e6 and abs(fn["encoder"] - 3.05e6) < 0.01e6   # flops-table.md E14b row
    print("   E14b shapes, train FLOPs/byte: " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in fn.items()))
    # bptt rule: global loops beyond bptt_loops count forward only for the prefix (as gogeta.train does)
    m = build(P=4, r=2, glob={**G_ALL, "loops": 3, "loops_max": 3, "bptt_loops": 1})
    fwd3 = m._global_fwd_per_patch(12, 3)
    fwd1 = m._global_fwd_per_patch(12, 1)
    assert abs(m.flops_breakdown(48, loops=3)["global"] - (fwd3 + 2 * fwd1) / 4) < 1e-6


def test_config_dict_roundtrip():
    m = build(P=4, r=2, glob=G_ALL)
    d = m.cfg.to_dict()
    assert d["arch"] == "c2f" and d["global_model"]["moe_experts"] == 4 and d["global_model"]["vocab_size"] == 1
    from gogeta.c2f import build_c2f
    d2 = {k: v for k, v in d.items()}
    d2["global_model"] = {k: v for k, v in d["global_model"].items()}
    m2 = build_c2f(d2)
    assert sum(p.numel() for p in m2.parameters()) == sum(p.numel() for p in m.parameters())
    assert isinstance(m2.glob, FusionDance)


# ============================================================================= tokens as patches
TOK_V = 40
TLOCAL = dict(d_local=16, local_heads=2, local_mlp_hidden=32, local_window=6, local_encoder=[],
              refine_blocks=["conv", "lswa"])


def tiny_table(V=TOK_V, seed=0):
    g = torch.Generator().manual_seed(seed)
    tab = [b"\x00"]
    for _ in range(V - 1):
        n = int(torch.randint(1, 6, (1,), generator=g))
        tab.append(bytes(torch.randint(1, 256, (n,), generator=g).tolist()))
    return tab


def tbuild(r=2, glob=None, N=16, seed=0, table=None, **kw):
    from gogeta.c2f import C2FTokenModel
    torch.manual_seed(seed)
    g = {**(glob or G_DENSE), "vocab_size": TOK_V}
    cfg = C2FConfig(patching="tokens", max_seq_len=N, refine_passes=r, global_model=g, **{**TLOCAL, **kw})
    return C2FTokenModel(cfg, table or tiny_table()).eval()


def test_token_byte_table_roundtrip():
    """The project tokenizer's table: 8192 ids, specials -> 0x00, and concat(table[ids]) == text bytes."""
    from gogeta.c2f import DEFAULT_TOKENIZER, token_byte_table
    tab = token_byte_table(DEFAULT_TOKENIZER)
    assert len(tab) == 8192 and tab[0] == b"\x00" and tab[1] == b"\x00"
    lens = [len(b) for b in tab]
    print(f"   tokenizer table: {len(tab)} ids, bytes/token max {max(lens)}, mean over vocab {sum(lens) / len(lens):.2f}")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("   (tokenizers not installed: roundtrip skipped)")
        return
    tok = Tokenizer.from_file(str(ROOT / DEFAULT_TOKENIZER))
    for text in ("def f(x):\n    return x**2  # café\n", "class Stack:\n\tdef push(self, v): ...\n", "naïve ✓ 日本\n"):
        ids = tok.encode(text).ids
        assert b"".join(tab[i] for i in ids) == text.encode("utf-8"), text


def test_token_spans_layout():
    tab = tiny_table()
    m = tbuild(table=tab)
    y = torch.tensor([[3, 7, 0], [5, 5, 9]])
    inp, tgt, span, slot = m.spans(y)
    for b in range(2):
        want_inp, want_tgt, want_span, want_slot = [], [], [], []
        for i, t in enumerate(y[b].tolist()):
            bs = list(tab[t])
            want_inp += [256] + bs
            want_tgt += bs + [256]
            want_span += [i] * (len(bs) + 1)
            want_slot += list(range(len(bs) + 1))
        n = len(want_inp)
        assert inp[b, :n].tolist() == want_inp and tgt[b, :n].tolist() == want_tgt
        assert span[b, :n].tolist() == want_span and slot[b, :n].tolist() == want_slot
        assert (tgt[b, n:] == -100).all()


def _span_start(span, i):
    return int((span[0] < i).sum())


def token_perturbation_check(m, N=16, seed=0):
    """Perturb token s of the sequence (input x[s] = target y[s-1]); rows of spans < s-1 and span s-1's BOT row must
    be bitwise unchanged in every pass; span s's BOT row must change. Returns (leaks, vacuous, n)."""
    g = torch.Generator().manual_seed(1000 + seed)
    seq = torch.randint(1, TOK_V, (1, N + 1), generator=g)
    leaks = vac = 0
    ss = [1, 2, 5, N // 2, N - 1]
    for s in ss:
        sp = seq.clone()
        sp[0, s] = 1 + (seq[0, s] + int(torch.randint(1, TOK_V - 2, (1,), generator=g))) % (TOK_V - 1)
        outs = []
        for q in (seq, sp):
            with torch.no_grad():
                lg = m(q[:, :-1], q[:, 1:], return_all=True)[0]
            outs.append((lg, m.last_aux["layout"][1]))
        (a, span_a), (b, span_b) = outs
        keep = _span_start(span_a, s - 1) + 1                  # spans < s-1, plus span s-1's BOT row
        assert keep == _span_start(span_b, s - 1) + 1
        for la, lb in zip(a, b):
            if not torch.equal(la[:, :keep], lb[:, :keep]):
                leaks += 1
        if s < N:
            ia, ib = _span_start(span_a, s), _span_start(span_b, s)
            if torch.equal(a[-1][:, ia], b[-1][:, ib]):
                vac += 1
    return leaks, vac, len(ss)


def test_token_causality_perturbation():
    for gname, glob in (("dense", G_DENSE), ("all_parts_gather", G_ALL)):
        for r in (1, 2):
            for frozen in (False, True):
                m = tbuild(r=r, glob=glob, refine_contractive=True, refine_aux_weight=0.5, freeze_global=frozen)
                leaks, vac, n = token_perturbation_check(m)
                assert leaks == 0, f"LEAK tokens {gname} r={r} frozen={frozen}: {leaks}"
                assert vac == 0, f"VACUOUS tokens {gname} r={r}: {vac}/{n}"
                print(f"   causal OK  tokens global={gname:17s} r_b={r} frozen={frozen!s:5s} perturbations={n}")
    m = tbuild(r=2)
    m._broken_lookahead = True
    leaks, _, n = token_perturbation_check(m)
    assert leaks > 0, "the lookahead control did not leak: the token perturbation test has no teeth"
    print(f"   broken lookahead control: {leaks} leaking (pass, perturbation) pairs of {2 * n}, as required")


def test_token_loss_is_nats_per_token_and_generation_matches():
    tab = tiny_table()
    m = tbuild(r=2, table=tab)
    g = torch.Generator().manual_seed(5)
    seq = torch.randint(1, TOK_V, (2, 17), generator=g)
    x, y = seq[:, :-1], seq[:, 1:]
    with torch.no_grad():
        lg, loss = m(x, y)
    tgt = m.last_aux["layout"][0]
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, 257), tgt.reshape(-1), ignore_index=-100, reduction="sum")
    assert abs(loss.item() - ce.item() / y.numel()) < 1e-5
    assert m.last_aux["rows"].item() == sum(len(tab[t]) + 1 for t in y.flatten().tolist())
    # generation path == teacher-forced rows (prefix of the same causal stream)
    n = 9
    full = m(seq[:1, :n], seq[:1, 1:n + 1])[0]
    span = m.last_aux["layout"][1]
    nxt = list(tab[int(seq[0, n])])
    for j in range(len(nxt) + 1):
        row = _span_start(span, n - 1) + j
        got = m.next_byte_logits(seq[:1, :n], nxt[:j])
        torch.testing.assert_close(got, full[0, row], rtol=1e-5, atol=1e-5)


def test_token_backward_frozen_and_unfrozen():
    x, y = torch.randint(1, TOK_V, (2, 16)), torch.randint(1, TOK_V, (2, 16))
    # unfrozen all-parts: every parameter gets a gradient (the token embedding is used here)
    m = tbuild(r=2, glob=G_ALL, refine_contractive=True, refine_aux_weight=0.5).train()
    _, loss, aux = m(x, y, return_aux=True)
    (loss + aux["loss"]).backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not bad, bad
    # frozen trunk: no global gradients, global unchanged by a step, stays in eval mode; decoder trains
    m = tbuild(r=2, glob=G_ALL, freeze_global=True).train()
    assert not m.glob.training and all(not p.requires_grad for p in m.glob.parameters())
    before = {k: v.clone() for k, v in m.glob.state_dict().items()}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    _, loss, aux = m(x, y, return_aux=True)
    assert aux is None
    loss.backward()
    assert all(p.grad is None for p in m.glob.parameters())
    assert all(p.grad is not None for n, p in m.named_parameters() if p.requires_grad)
    opt.step()
    assert all(torch.equal(before[k], v) for k, v in m.glob.state_dict().items())
    pc = m.param_counts()
    assert pc["trainable"] == pc["local"]


def test_token_flops_and_init():
    m = tbuild(r=2, glob=G_ALL)
    f, ff = m.flops_breakdown(16), tbuild(r=2, glob=G_ALL, freeze_global=True).flops_breakdown(16)
    fwd = m.glob.flops_per_token(16, training=False) - 2 * 32 * TOK_V
    assert abs(f["global"] - 3 * fwd) < 1e-6 and abs(ff["global"] - fwd) < 1e-6
    assert abs(f["decoder"] - ff["decoder"]) < 1e-6
    for kw, depth in ((dict(r=1), 2), (dict(r=2), 4), (dict(r=2, refine_shared=False), 4)):
        mm = tbuild(**kw, d_local=128, local_mlp_hidden=512)
        got, _ = _res_std(mm.passes)
        want = 0.02 / math.sqrt(2 * depth)
        assert abs(got - want) / want < 0.05, (kw, got, want)


if __name__ == "__main__":
    import time
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
            except Exception as ex:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {ex}", flush=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
