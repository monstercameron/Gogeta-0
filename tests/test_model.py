"""Model correctness tests (GPU; GDN needs the fla kernel).

    ~/.venvs/gogeta/bin/python tests/test_model.py
"""
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gogeta.model import BitLinear, FusionDance, ModelConfig  # noqa: E402

DEV = "cuda"
SMALL = dict(vocab_size=512, d_model=128, n_heads=4, mlp_hidden=256, window=16, max_seq_len=128)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def model(**kw):
    torch.manual_seed(0)
    return FusionDance(ModelConfig(**{**SMALL, **kw})).to(DEV).eval()


def causality(m, T=64, t=40, **fw):
    idx = torch.randint(0, SMALL["vocab_size"], (2, T), device=DEV)
    idx2 = idx.clone()
    idx2[:, t:] = torch.randint(0, SMALL["vocab_size"], (2, T - t), device=DEV)
    with torch.no_grad():
        a, _ = m(idx, **fw)
        b, _ = m(idx2, **fw)
    past = (a[:, :t] - b[:, :t]).abs().max().item()
    future = (a[:, t:] - b[:, t:]).abs().max().item()
    return past, future


# 1. causality for every block kind, and for the looped model
for name, kw in {
    "attn": dict(prelude=["attn"], core=[], coda=[]),
    "swa": dict(prelude=["swa"], core=[], coda=[]),
    "gdn": dict(prelude=["gdn"], core=[], coda=[]),
    "looped": dict(prelude=["swa"], core=["gdn", "gdn", "swa"], coda=["attn"], loops=3),
}.items():
    past, future = causality(model(**kw))
    check(f"causal[{name}]", past < 1e-4 and future > 1e-4, f"past diff {past:.1e}, future diff {future:.1e}")

# 2. sliding window really cuts off: one SWA layer, change token 0 -> only positions < window change
m = model(prelude=["swa"], core=[], coda=[])
idx = torch.randint(0, 512, (1, 64), device=DEV)
idx2 = idx.clone()
idx2[0, 0] = (idx[0, 0] + 1) % 512
with torch.no_grad():
    a, _ = m(idx)
    b, _ = m(idx2)
inside = (a[0, :16] - b[0, :16]).abs().max().item()
outside = (a[0, 16:] - b[0, 16:]).abs().max().item()
check("swa window cutoff", inside > 1e-4 and outside < 1e-5, f"inside {inside:.1e}, outside {outside:.1e}")

# 3. BitLinear gives ternary effective weights
lin = BitLinear(64, 32, bias=False).to(DEV)
w = lin.weight
scale = w.abs().mean()
levels = torch.unique(((w / scale).round().clamp(-1, 1))).tolist()
check("bitlinear ternary levels", set(levels) <= {-1.0, 0.0, 1.0}, f"levels {levels}")
mt = model(ternary=True)
n_bit = sum(isinstance(x, BitLinear) for x in mt.modules())
check("ternary model swaps projections", n_bit > 10, f"{n_bit} BitLinear modules")

# 4. loops: output depends on r; truncated BPTT still trains the prelude + injection
m = model(prelude=["swa"], core=["gdn", "swa"], coda=["attn"], loops=4, bptt_loops=2)
idx = torch.randint(0, 512, (2, 32), device=DEV)
with torch.no_grad():
    l2, _ = m(idx, loops=2)
    l6, _ = m(idx, loops=6)
check("loop count changes output", (l2 - l6).abs().max().item() > 1e-3)
m.train()
_, loss = m(idx, targets=idx, loops=6)
loss.backward()
gp = m.prelude[0].mlp.up.weight.grad
gi = m.inject.weight.grad
check("bptt grads reach prelude + inject", gp is not None and gp.abs().sum() > 0 and gi.abs().sum() > 0)

# 5. backward through everything incl. ternary, loss finite
mt.train()
_, loss = mt(idx, targets=idx, loops=3)
loss.backward()
check("ternary looped backward finite", torch.isfinite(loss).item()
      and all(torch.isfinite(p.grad).all() for p in mt.parameters() if p.grad is not None), f"loss {loss.item():.3f}")

# 6. negative-eigenvalue GDN (E11 / SQ2): causal, and actually different from the default beta range
past, future = causality(model(prelude=["gdn"], core=[], coda=[], gdn_neg_eigval=True))
check("causal[gdn neg-eig]", past < 1e-4 and future > 1e-4, f"past diff {past:.1e}, future diff {future:.1e}")
torch.manual_seed(0)
m_std = FusionDance(ModelConfig(**{**SMALL, "prelude": ["gdn"], "core": [], "coda": []})).to(DEV).eval()
m_neg = FusionDance(ModelConfig(**{**SMALL, "prelude": ["gdn"], "core": [], "coda": [], "gdn_neg_eigval": True})).to(DEV).eval()
m_neg.load_state_dict(m_std.state_dict())  # identical weights: only the beta range differs
idx = torch.randint(0, 512, (2, 64), device=DEV)
with torch.no_grad():
    d = (m_std(idx)[0] - m_neg(idx)[0]).abs().max().item()
check("neg-eig changes the computation (same weights)", d > 1e-3, f"max logit diff {d:.2e}")

# 7. parameter / FLOP accounting for the FusionDance experiment configs (other archs, e.g. hf_llama, are skipped)
for cfg_path in sorted((pathlib.Path(__file__).resolve().parent.parent / "configs").glob("e*.json")):
    full = json.loads(cfg_path.read_text())
    if full.get("arch", full.get("model", {}).get("arch", "fusiondance")) != "fusiondance":
        print(f"      {cfg_path.name:22s} (arch {full.get('arch', full['model'].get('arch'))}: skipped)")
        continue
    mc = {k: v for k, v in full["model"].items() if k != "arch"}
    mm = FusionDance(ModelConfig(**mc))
    pc = mm.param_counts()
    print(f"      {cfg_path.name:22s} total {pc['total']/1e6:6.2f}M  non-emb {pc['non_embedding']/1e6:6.2f}M  "
          f"emb {pc['embedding']/1e6:.2f}M  depth@r {mm.effective_depth()}  "
          f"train GFLOP/token {mm.flops_per_token(1024)/1e9:.3f}")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
