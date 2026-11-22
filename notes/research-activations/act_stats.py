"""CPU-only activation-outlier statistics for candidate MLP activations (research note: notes/research-activations.md).

Takes the REAL pre-activations (a, b = up(x).chunk(2)) of the E0 checkpoint (6 attention blocks, SwiGLU, fp32, CPU)
on val tokens, and applies each candidate activation to them. CAVEAT: the weights were trained for SwiGLU, so this
shows the *shape* each activation imposes on the down-projection input, not what a model trained with it would learn.
Metrics at the input of the down projection (what BitLinear's per-token absmax int8 quantiser sees):
  sparsity   fraction with |h| < 0.01 * rms(h)   (exact zeros for ReLU-type)
  kurt       excess kurtosis over all entries
  crest      mean over tokens of max_c|h| / rms_c(h)  (per-token crest factor; int8 absmax step = max/127)
  int8_relerr  ||q(h) - h|| / ||h|| with per-token absmax int8 (what BitLinear does)
Usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -I notes/research-activations/act_stats.py  (reads, never writes the repo
except results/act_stats.json)
"""
import json, math, os, pathlib, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(0)
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from gogeta.model import FusionDance, ModelConfig  # noqa: E402

CK = pathlib.Path(os.path.expanduser("~/fusiondance-data/checkpoints/E0-20261122-122059-s0/model.pt"))
VAL = pathlib.Path(os.path.expanduser("~/fusiondance-data/datasets/code-py-v1/val.bin"))
ck = torch.load(CK, map_location="cpu", weights_only=False)
cfg = ModelConfig(**ck["model_config"])
model = FusionDance(cfg); model.load_state_dict(ck["model"]); model.eval()
val = np.memmap(VAL, dtype=np.uint16, mode="r")
T, N = 512, 8
idx = torch.tensor(np.stack([val[i * 40000: i * 40000 + T].astype(np.int64) for i in range(N)]))

def xielu(x, ap=0.8, an=0.8, beta=0.5):
    sp = lambda v: F.softplus(torch.tensor(v)).item()
    p, n = sp(ap), sp(an) + beta
    return torch.where(x > 0, p * x * x + beta * x, n * torch.expm1(torch.clamp(x, max=0)) - n * x + beta * x)

CANDS = {  # name -> f(a, b) giving the down-projection input
    "swiglu  silu(a)*b": lambda a, b: F.silu(a) * b,
    "geglu   gelu(a)*b": lambda a, b: F.gelu(a) * b,
    "reglu   relu(a)*b": lambda a, b: F.relu(a) * b,
    "relu2glu relu(a)^2*b": lambda a, b: F.relu(a) ** 2 * b,
    "drelu   relu(a)*relu(b)": lambda a, b: F.relu(a) * F.relu(b),
    "relu2   relu(a)^2 (ungated)": lambda a, b: F.relu(a) ** 2,
    "relu    relu(a) (ungated)": lambda a, b: F.relu(a),
    "gelu    gelu(a) (ungated)": lambda a, b: F.gelu(a),
    "silu    silu(a) (ungated)": lambda a, b: F.silu(a),
    "xielu   xIELU(a) (ungated, init)": lambda a, b: xielu(a),
}

def stats(h):
    h = h.reshape(-1, h.shape[-1]).double()
    rms = h.pow(2).mean().sqrt()
    mu = h.mean(); sd = h.std()
    kurt = (((h - mu) / sd) ** 4).mean() - 3
    sparse = (h.abs() < 0.01 * rms).double().mean()
    tok_rms = h.pow(2).mean(-1).sqrt().clamp(min=1e-12)
    crest = (h.abs().amax(-1) / tok_rms).mean()
    s = 127.0 / h.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    q = (h * s).round().clamp(-128, 127) / s
    rel = ((q - h).norm() / h.norm())
    return dict(sparsity=sparse.item(), kurt=kurt.item(), crest=crest.item(), max_over_rms=(h.abs().max() / rms).item(), int8_relerr=rel.item())

acc = {k: [] for k in CANDS}; acc["_input_to_up (norm2 out)"] = []; acc["_gaussian_ref"] = []
caps = {}
def mk(i):
    def hook(mod, inp, out):
        caps[i] = (inp[0].detach(), out.detach())
    return hook
for i, blk in enumerate(model.prelude):
    blk.mlp.up.register_forward_hook(mk(i))
with torch.no_grad():
    model(idx)
    for i, (x, u) in caps.items():
        a, b = u.chunk(2, -1)
        for k, f in CANDS.items():
            acc[k].append(stats(f(a, b)))
        acc["_input_to_up (norm2 out)"].append(stats(x))
        g = torch.randn_like(a); acc["_gaussian_ref"].append(stats(g))
out = {}
for k, v in acc.items():
    out[k] = {m: float(np.mean([d[m] for d in v])) for m in v[0]}
    out[k]["per_layer_max_over_rms"] = [round(d["max_over_rms"], 1) for d in v]
res = ROOT / "notes/research-activations/results/act_stats.json"
res.write_text(json.dumps(dict(ckpt="E0-20261122-122059-s0", tokens=N * T, layers=len(model.prelude), stats=out), indent=1))
print(f"{'candidate':34s} {'sparse':>7s} {'kurt':>8s} {'crest':>7s} {'max/rms':>8s} {'int8err':>8s}")
for k, d in out.items():
    print(f"{k:34s} {d['sparsity']:7.3f} {d['kurt']:8.1f} {d['crest']:7.2f} {d['max_over_rms']:8.1f} {d['int8_relerr']:8.4f}")
