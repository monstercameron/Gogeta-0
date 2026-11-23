"""Shared CPU helpers for the 200M F2 post-hoc side quests (SQ339/340/341/347/348/351).

CPU inference on the SQ263 checkpoints (code-py-v2-trained, vocab 8192). Two facts that matter:
  * the scoring docs are the v1 `docs_val.jsonl` clean80 subset (K9 `val_per_doc_overlap` < 0.8): this is exactly the
    set the SQ263 fleet scorer used (`experiments/stats/doclevel/*.json`: overlap_source K9, clean80 = 2,333 docs). The
    code-py-v2 dataset itself is NOT present locally, and v2 train excludes docs overlapping v1 val, so these are valid
    held-out docs for v2-trained models. The v2 tokenizer is byte-identical to v1's (sha256 0e75f39f...).
  * the GDN mixer calls fla's Triton chunk kernel (GPU only). On CPU we replace `GatedDeltaNet._kernel` with the fp32
    loop `reference()` of experiments/checks/sq_hw_fla_recurrent.py (verified against the chunk kernel there).

Run from WSL with PYTHONPATH = repo root. Not a runnable check by itself.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pathlib
import random
import subprocess
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

torch.set_num_threads(4)

FLEET = os.path.expanduser("~/fusiondance-data/cloud/fleet")
DATASET_DIR = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1")
RES = HERE / "results"
CELL_ID = {"000": "E20", "001": "E21", "010": "E22", "100": "E24", "111": "E27"}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def best_runs() -> dict:
    """{cell_code: {seed: run_name}} at each cell's best LR (sq263.json best_lr), seeds 0-2."""
    best = json.load(open(RES / "sq263.json"))["best_lr"]
    runs = json.load(open(RES / "sq263_runs.json"))["runs"]
    out: dict = {}
    for code, e in CELL_ID.items():
        out[code] = {r["seed"]: r["run"] for r in runs if r["cell"] == e and abs(r["lr"] - best[e]) < 1e-12}
    return out


def ckpt_path(run: str) -> str:
    p = glob.glob(f"{FLEET}/*/fusiondance-data/checkpoints/{run}/model.pt") or glob.glob(os.path.expanduser(f"~/fusiondance-data/checkpoints/{run}/model.pt"))
    assert len(p) == 1, (run, p)
    return p[0]


def gdn_chunk_cpu(q, k, v, g, beta, scale, C: int = 64):
    """fp32 chunkwise (WY) gated delta rule in plain torch, ~5-12x faster than the per-token loop `reference()`.
    Same math as fla's naive chunk reference; validated against `reference()` by `kernel_selfcheck()` (max|err| ~5e-7
    on the output, 3e-6 on the state, T = 17, 300, 1024). Right-padding with g = 0, beta = 0 is an exact no-op."""
    import torch.nn.functional as F

    B, T, H, K = q.shape
    V = v.shape[-1]
    pad = (-T) % C
    tr = lambda x: F.pad(x.float().transpose(1, 2), (0, 0, 0, pad)) if x.dim() == 4 else F.pad(x.float().transpose(1, 2), (0, pad))  # noqa: E731
    q, k, v, g, beta = tr(q), tr(k), tr(v), tr(g), tr(beta)
    N = (T + pad) // C
    q, k, v = [x.reshape(B, H, N, C, -1) for x in (q, k, v)]
    g = g.reshape(B, H, N, C)
    beta = beta.reshape(B, H, N, C)
    q = q * scale
    kb, vb = k * beta[..., None], v * beta[..., None]
    d = g.cumsum(-1)
    Lm = (d[..., :, None] - d[..., None, :]).tril().exp().tril()
    up = torch.triu(torch.ones(C, C, dtype=torch.bool), 0)
    attn = -((kb @ k.transpose(-1, -2)) * Lm).masked_fill(up, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(C)
    u, w = attn @ vb, attn @ (kb * d.exp()[..., None])
    S = torch.zeros(B, H, K, V)
    outs = []
    st = torch.triu(torch.ones(C, C, dtype=torch.bool), 1)
    for i in range(N):
        qi, ki, ui, wi = q[:, :, i], k[:, :, i], u[:, :, i], w[:, :, i]
        a = ((qi @ ki.transpose(-1, -2)) * Lm[:, :, i]).masked_fill(st, 0)
        u2 = ui - wi @ S
        outs.append((qi * d[:, :, i, :, None].exp()) @ S + a @ u2)
        S = S * d[:, :, i, -1, None, None].exp() + (ki * (d[:, :, i, -1, None] - d[:, :, i]).exp()[..., None]).transpose(-1, -2) @ u2
    return torch.cat(outs, 2)[:, :, :T].transpose(1, 2).contiguous(), S


def kernel_selfcheck() -> dict:
    import torch.nn.functional as F

    import sq_hw_fla_recurrent as R

    torch.manual_seed(0)
    out = {}
    for T in (17, 300, 1024):
        B, H, K = 3, 6, 64
        q = F.normalize(torch.randn(B, T, H, K), dim=-1)
        k = F.normalize(torch.randn(B, T, H, K), dim=-1)
        v = torch.randn(B, T, H, K)
        g = -torch.rand(B, T, H) * 3
        beta = torch.sigmoid(torch.randn(B, T, H))
        a, sa = R.reference(q, k, v, g, beta, 0.125)
        b, sb = gdn_chunk_cpu(q, k, v, g, beta, 0.125)
        out[T] = {"max_err_out": (a - b).abs().max().item(), "max_err_state": (sa - sb).abs().max().item()}
    return out


def patch_gdn_cpu():
    """Replace the fla kernel by the fp32 chunkwise torch implementation (same signature/return as chunk_gated_delta_rule)."""
    from gogeta.model import GatedDeltaNet

    def kern(q, k, v, g, beta, scale=None, **kw):
        assert not kw, kw  # neg_eigval path not used by the F2 cells
        return gdn_chunk_cpu(q, k, v, g, beta, scale)

    orig_init = GatedDeltaNet.__init__

    def init(self, cfg):
        orig_init(self, cfg)
        self._kernel = kern

    GatedDeltaNet.__init__ = init


_PATCHED = False


def load_model(run: str):
    global _PATCHED
    if not _PATCHED:
        patch_gdn_cpu()
        _PATCHED = True
    from gogeta.model import FusionDance, ModelConfig

    ck = torch.load(ckpt_path(run), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**dict(ck["model_config"]))
    model = FusionDance(cfg)
    model.load_state_dict(ck["model"])
    return model.float().eval()


def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(DATASET_DIR, "tokenizer.json"))


def clean80_docs() -> list[tuple[int, str]]:
    ov = json.load(open(ROOT / "experiments" / "checks" / "results" / "K9.json"))["data"]["val_per_doc_overlap"]
    texts = [json.loads(l)["text"] for l in open(os.path.join(DATASET_DIR, "docs_val.jsonl"), encoding="utf-8")]
    assert len(texts) == len(ov)
    return [(i, texts[i]) for i, o in enumerate(ov) if o is None or o < 0.8]


def subset(n: int = 300, seed: int = 0) -> list[tuple[int, str]]:
    allc = clean80_docs()
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(allc)), n))
    return [allc[i] for i in idx]


def chunk_doc(ids: list[int], window: int):
    """K8's windowing: (ids chunk, #leading targets not scored) list for one doc."""
    L, stride = len(ids), window // 2
    if L <= window:
        return [(ids, 0)]
    out, prev_end, begin = [], 0, 0
    while True:
        end = min(begin + window, L)
        first_target = max(prev_end, begin + 1)
        out.append((ids[begin:end], first_target - begin - 1))
        prev_end = end
        if end == L:
            return out
        begin += stride


@torch.no_grad()
def score_docs_multi(fwd, docs, window: int = 1024, budget: int = 8192, log=None):
    """fwd(ids[B,T] long) -> {name: logits [B,T,V]}. One pass over the docs; returns {name: {nll_nats, n_tok, chars}}
    with per-doc nll (nats; K8's windowing, window/2 stride). Several readouts share one trunk forward."""
    tok = tokenizer()
    bos = tok.token_to_id("<|eos|>")
    chunks, ntok = [], [0] * len(docs)
    for di, (_, text) in enumerate(docs):
        ids = [bos] + tok.encode(text).ids
        ntok[di] = len(ids) - 1
        for c, skip in chunk_doc(ids, window):
            chunks.append((di, c, skip))
    chunks.sort(key=lambda c: len(c[1]))
    nll: dict = {}
    i, done = 0, 0
    while i < len(chunks):
        T = len(chunks[i][1])
        j = i
        while j < len(chunks) and (j - i + 1) * len(chunks[j][1]) <= max(budget, T):
            j += 1
        batch = chunks[i:j]
        T = max(len(c[1]) for c in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        mask = torch.zeros(len(batch), T - 1, dtype=torch.bool)
        for b, (_, ids, skip) in enumerate(batch):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, skip:len(ids) - 1] = True
        outs = fwd(x)
        if isinstance(outs, torch.Tensor):
            outs = {"_": outs}
        for name, logits in outs.items():
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            l = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
            l = (l * mask).sum(-1).double()
            acc = nll.setdefault(name, [0.0] * len(docs))
            for b, (di, _, _) in enumerate(batch):
                acc[di] += float(l[b])
        i = j
        done += len(batch)
        if log:
            log(f"  chunks {done}/{len(chunks)}")
    chars = [len(t) for _, t in docs]
    return {name: {"nll_nats": v, "n_tok": ntok, "chars": chars} for name, v in nll.items()}


def score_docs(fwd, docs, window: int = 1024, budget: int = 8192, log=None):
    """Single-readout wrapper: per-doc nll (nats), n_tok, chars (list order = docs)."""
    return score_docs_multi(fwd, docs, window, budget, log)["_"]


def bpc(sc: dict) -> float:
    return sum(sc["nll_nats"]) / math.log(2) / sum(sc["chars"])


def fwd_default(model, loops=None):
    return lambda x: model(x, loops=loops)[0]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs):
    xs = list(xs)
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def mean_ci(xs):
    """mean, sd, 95% CI half-width (t, df=n-1) of a small sample."""
    xs = list(xs)
    n = len(xs)
    m, s = mean(xs), sd(xs)
    return {"mean": m, "sd": s, "n": n, "ci95": [m - T975[n - 1] * s / math.sqrt(n), m + T975[n - 1] * s / math.sqrt(n)]}


def make_log(path):
    f = open(path, "a", buffering=1)

    def log(msg):
        import time

        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    return log
