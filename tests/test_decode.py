"""CPU tests for gogeta/decode.py: cached inference and lossless self-speculative decoding.

    python tests/test_decode.py          (from the repo root, in the gogeta venv; no pytest needed)

(a) prefill + step logits == the model's own full forward at every position (fp32, the full forward's GDN runs the
    torch reference kernel `decode.gated_delta_rule_ref`, as tests/test_moe.cpu_gdn does);
(b) one step over k tokens == k single steps (logits and the following step);
(c) rollback after a k-token step == never having seen the rolled-back tokens;
(d) speculative_generate == plain cached greedy == greedy from the full re-forward, for several seeds and configs.
"""
import pathlib
import sys
import traceback

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gogeta.decode import greedy_generate, prefill, speculative_generate, step, use_reference_gdn  # noqa: E402
from gogeta.model import FusionDance, ModelConfig  # noqa: E402

V = 97
BASE = dict(vocab_size=V, d_model=32, n_heads=4, mlp_hidden=64, window=6, max_seq_len=96, conv_size=4,
            loops=2, loops_min=1, loops_max=4)
CONFIGS = {
    "attn": dict(prelude=["attn"], core=[], coda=["attn"]),
    "swa": dict(prelude=["swa"], core=[], coda=["swa", "attn"]),
    "gdn": dict(prelude=["gdn"], core=[], coda=["gdn", "attn"]),
    "gdn_negeig": dict(prelude=["gdn"], core=[], coda=["attn"], gdn_neg_eigval=True),
    "loops": dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"]),
    "contractive_embed_adapter": dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"],
                                      inject_contractive=True, loop_embed=True, loop_adapter_rank=4),
    "ternary": dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], ternary=True),
    "ternary_fp_ends": dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], ternary=True, ternary_fp_ends=True,
                            inject_contractive=True),
    "moe_core_ternary": dict(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], ternary=True, moe_experts=4,
                             moe_top_k=2, moe_hidden=32, moe_core_only=True),
}
TOL = 1e-4


def build(name, seed=0, **kw):
    torch.manual_seed(seed)
    m = FusionDance(ModelConfig(**{**BASE, **CONFIGS[name], **kw})).eval()
    with torch.no_grad():
        m.embed.weight.mul_(20.0)          # sharpen the random-init logits so greedy argmaxes are not near-ties
        if m.core:                         # make every core pass move h, so r_draft != r_full drafts get rejected
            m.inject.weight.mul_(60.0)
            for b in m.core:
                b.mixer.o.weight.mul_(30.0)
                if hasattr(b.mlp, "down"):
                    b.mlp.down.weight.mul_(30.0)
                else:
                    b.mlp.w_down.mul_(30.0)
    return use_reference_gdn(m)


def toks(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (1, n), generator=g)


def full_logits(m, idx, r):
    with torch.no_grad():
        return m(idx, loops=r)[0]


def loops_for(name):
    return [1, 2, 3, 4] if CONFIGS[name]["core"] else [None]


# ----------------------------------------------------------------------------- (a) prefill + step == full forward
def check_cached_equals_full(name):
    worst = 0.0
    for r in loops_for(name):
        m = build(name)
        idx = toks(20)
        ref = full_logits(m, idx, r)
        for split, mr in ((1, 16), (7, 3), (13, 0)):   # max_rollback 3 / 0 exercise the log trimming paths
            lp, cache = prefill(m, idx[:, :split], r, return_all=True, max_rollback=mr)
            got = [lp]
            pos = split
            for kstep in (1, 3, 2, 1, 4, 1, 2, 5, 1):
                if pos >= idx.shape[1]:
                    break
                kk = min(kstep, idx.shape[1] - pos)
                got.append(step(m, cache, idx[:, pos:pos + kk], loops=r))
                pos += kk
            err = (torch.cat(got, 1) - ref).abs().max().item()
            worst = max(worst, err)
            assert err < TOL, f"{name} r={r} split={split}: max|cached - full| = {err:.2e}"
    return worst


# ----------------------------------------------------------------------------- (b) k-token step == k single steps
def check_multi_equals_single(name):
    worst = 0.0
    for r in loops_for(name):
        m = build(name, seed=3)
        idx = toks(18, seed=4)
        _, c1 = prefill(m, idx[:, :8], r)
        _, ck = prefill(m, idx[:, :8], r)
        singles = torch.cat([step(m, c1, idx[:, i:i + 1]) for i in range(8, 14)], 1)
        multi = step(m, ck, idx[:, 8:14])
        nxt1, nxtk = step(m, c1, idx[:, 14:18]), step(m, ck, idx[:, 14:18])
        err = max((singles - multi).abs().max().item(), (nxt1 - nxtk).abs().max().item())
        worst = max(worst, err)
        assert err < TOL, f"{name} r={r}: max|multi - single| = {err:.2e}"
    return worst


# ----------------------------------------------------------------------------- (c) rollback
def check_rollback(name):
    worst = 0.0
    for r in loops_for(name):
        m = build(name, seed=5)
        idx = toks(20, seed=6)
        junk = toks(6, seed=7)
        _, ca = prefill(m, idx[:, :10], r, max_rollback=6)
        _, cb = prefill(m, idx[:, :10], r, max_rollback=6)
        _, cc = prefill(m, idx[:, :10], r, max_rollback=6)
        step(m, ca, idx[:, 10:12])
        step(m, cb, torch.cat([idx[:, 10:12], junk[:, :4]], 1))   # one step: 2 good + 4 junk tokens
        cb.rollback(4)
        step(m, cc, idx[:, 10:11])                                 # several steps: 2 good + 4 junk, then roll back
        step(m, cc, torch.cat([idx[:, 11:12], junk[:, :1]], 1))
        step(m, cc, junk[:, 1:2])
        step(m, cc, junk[:, 2:4])
        cc.rollback(3)
        cc.rollback(1)                                             # two rollbacks in a row
        a, b, c = (step(m, x, idx[:, 12:20]) for x in (ca, cb, cc))
        err = max((a - b).abs().max().item(), (a - c).abs().max().item())
        worst = max(worst, err)
        assert ca.pos == cb.pos == cc.pos and err < TOL, f"{name} r={r}: rollback mismatch {err:.2e}"
        try:                                                       # beyond the log: must refuse, not corrupt
            cc.rollback(7)
            raise AssertionError("rollback beyond max_rollback was accepted")
        except ValueError:
            pass
    return worst


# ----------------------------------------------------------------------------- (d) speculative == greedy
def greedy_full_reforward(m, idx, n, r):
    seq = idx.clone()
    out = []
    for _ in range(n):
        t = int(full_logits(m, seq, r)[0, -1].argmax())
        out.append(t)
        seq = torch.cat([seq, torch.tensor([[t]])], 1)
    return out


def check_speculative(name):
    rows = []
    for seed in (0, 1, 2):
        m = build(name, seed=10 + seed)
        idx = toks(9, seed=20 + seed)
        n = 24
        has_core = bool(CONFIGS[name]["core"])
        pairs = [(1, 2), (1, 4), (2, 3), (3, 3), (4, 1)] if has_core else [(None, None)]
        ref_by_r = {}
        for rd, rf in pairs:
            if rf not in ref_by_r:
                g = greedy_generate(m, idx, n, rf)
                ref = greedy_full_reforward(m, idx, n, rf) if seed == 0 else g
                assert g == ref, f"{name} seed={seed} r={rf}: cached greedy != full re-forward greedy"
                ref_by_r[rf] = g
            for k in (1, 2, 4, 7):
                out, st = speculative_generate(m, idx, n, rd if has_core else 2, rf if has_core else 2, k)
                assert out == ref_by_r[rf], f"{name} seed={seed} rd={rd} rf={rf} k={k}: speculative != greedy"
                if rd == rf:   # same model as draft: every draft must be accepted
                    assert st["accepted"] == st["proposed"], f"{name}: rd == rf but acceptance {st['acceptance_rate']}"
                else:
                    rows.append(st["acceptance_rate"])
    if CONFIGS[name]["core"]:   # the rejection + rollback path must actually have been exercised
        assert min(rows) < 1.0, f"{name}: no draft was ever rejected; the test does not exercise rollback"
        return sum(rows) / len(rows)
    return 1.0


def main():
    torch.set_num_threads(4)
    ok = True
    for name in CONFIGS:
        for fn in (check_cached_equals_full, check_multi_equals_single, check_rollback, check_speculative):
            try:
                v = fn(name)
                print(f"PASS {fn.__name__:28s} {name:28s} {'mean acc' if fn is check_speculative else 'max err'} {v:.3g}")
            except Exception:  # noqa: BLE001
                ok = False
                print(f"FAIL {fn.__name__:28s} {name}")
                traceback.print_exc()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
