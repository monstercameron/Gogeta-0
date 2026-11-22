#!/usr/bin/env python
"""Export a REAL forward-pass trace of the prompt "# write fizzbuzz in python" to docs/data/token-trace.json.

Measured on CPU (no GPU): the GDN layers need fla's Triton kernels, which are GPU-only, so this script
substitutes a plain-PyTorch reference of the gated delta rule (recurrent form) for `chunk_gated_delta_rule`
(the stub is injected into sys.modules; gogeta/ is NOT modified). Fidelity is checked, not assumed:
  1. our traced forward must reproduce model.forward() logits (max abs diff printed, asserted tiny);
  2. the CPU model's val loss on a few val sequences is compared with the run's reported val_loss.
The "check" block in the JSON records both numbers.

Records: real tokenizer ids/pieces; per attention/SWA layer and loop, top-k attended sources per query
(attention weights averaged over heads); per-loop ||dh||/||h|| per token; next-token top-5; a greedy continuation.

    bash tools/launch_cpu_job.sh --name viz-trace -- ~/.venvs/gogeta/bin/python experiments/viz/export_trace.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get("GOGETA_DATA", Path.home() / "fusiondance-data"))
OUT = REPO / "docs" / "data" / "token-trace.json"
RAW = "write fizzbuzz in python"
PROMPT = "# " + RAW + "\n"
TOPK = 3
N_NEW = 40
CANDIDATES = ["E2", "E1", "E0"]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)


# ---------------------------------------------------------------- CPU reference for the GDN kernel
def gated_delta_rule_ref(q, k, v, g, beta, scale=None, **_):
    """Recurrent gated delta rule. q,k,v: [B,T,H,d]; g (log decay) and beta: [B,T,H].
    S_t = exp(g_t) S_{t-1};  S_t += beta_t k_t (v_t - S_t^T k_t)^T;  o_t = S_t^T (scale q_t)."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** -0.5 if scale is None else scale
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    S = q.new_zeros(B, H, dk, dv)
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t] - pred)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
    return torch.stack(outs, 1), S


def install_fla_stub():
    fla = types.ModuleType("fla")
    ops = types.ModuleType("fla.ops")
    gdr = types.ModuleType("fla.ops.gated_delta_rule")
    gdr.chunk_gated_delta_rule = gated_delta_rule_ref
    fla.ops, ops.gated_delta_rule = ops, gdr
    sys.modules.update({"fla": fla, "fla.ops": ops, "fla.ops.gated_delta_rule": gdr})


install_fla_stub()
from gogeta import model as M  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


# ---------------------------------------------------------------- tracing
class Rec:
    on = False
    attn = []  # filled by traced attention


def attn_forward(self, x):
    """Same math as Attention.forward, but explicit softmax so the weights can be read."""
    B, T, D = x.shape
    H, hd = self.cfg.n_heads, self.cfg.head_dim
    q, k, v = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k = self.rope(q), self.rope(k)
    if not Rec.on:
        if self.window is None or self.window >= T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(T, x.device))
    else:
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        i = torch.arange(T)
        d = i[:, None] - i[None, :]
        ok = (d >= 0) if self.window is None else ((d >= 0) & (d < self.window))
        w = torch.softmax(s.masked_fill(~ok, float("-inf")), dim=-1)
        y = w @ v
        Rec.attn.append(w[0].mean(0))  # [T,T] averaged over heads
    return self.o(y.transpose(1, 2).reshape(B, T, D))


M.Attention.forward = attn_forward


def vis(s: str) -> str:
    return s.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␣")


def top5(logits_row, tok):
    p = torch.softmax(logits_row.float(), -1)
    v, i = torch.topk(p, 5)
    return [{"id": int(a), "piece": vis(tok.decode([int(a)])), "p": round(float(b), 4)} for b, a in zip(v, i)]


@torch.no_grad()
def traced_forward(model, ids, loops):
    """Replicates FusionDance.forward (eval) and records attention + per-loop dh. Returns logits, record."""
    T = ids.shape[1]
    layers, delta = [], []
    Rec.on = True

    def run(blocks, x, stage, loop):
        for li, b in enumerate(blocks):
            Rec.attn = []
            x = b(x)
            ent = {"stage": stage, "layer": li, "loop": loop, "kind": b.kind}
            if Rec.attn:
                w = Rec.attn[0]
                tv, ti = torch.topk(w, min(TOPK, T), dim=-1)
                ent["top"] = [[[int(j), round(float(a), 3)] for a, j in zip(tv[q], ti[q]) if a > 1e-4] for q in range(T)]
            layers.append(ent)
        return x

    x = model.embed(ids)
    x = run(model.prelude, x, "prelude", None)
    if model.core:
        e = x
        h = x
        for l in range(loops):
            h0 = h
            h = h + model.inject(torch.cat([model.inj_norm_h(h), model.inj_norm_e(e)], dim=-1))
            h = run(model.core, h, "core", l)
            delta.append([round(float((h[0, t] - h0[0, t]).norm() / h0[0, t].norm().clamp(min=1e-9)), 4) for t in range(T)])
        x = h
    x = run(model.coda, x, "coda", None)
    Rec.on = False
    return model.head(model.norm_out(x)), {"layers": layers, "delta": delta}


def find_checkpoint():
    for exp in CANDIDATES:
        cks = sorted((DATA / "checkpoints").glob(f"{exp}-*/model.pt"))
        if cks:
            return exp, cks[-1]
    raise SystemExit("no E2/E1/E0 checkpoint found")


def main():
    exp, ck = find_checkpoint()
    state = torch.load(ck, map_location="cpu", weights_only=False)
    mc = state["model_config"]
    model = M.build_model(mc)
    model.load_state_dict(state["model"])
    model.eval()
    run_dir = DATA / "runs" / ck.parent.name
    summ = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    met = summ.get("metrics", {})
    ds = state["train"]["dataset"]
    tok = Tokenizer.from_file(str(DATA / "datasets" / ds / "tokenizer.json"))
    enc = tok.encode(PROMPT)
    ids = torch.tensor([enc.ids])
    loops = model.cfg.loops if model.core else 0
    print(f"model {ck.parent.name}  loops={loops}  prompt tokens={len(enc.ids)}  {enc.tokens}", flush=True)

    # --- fidelity checks
    logits, rec = traced_forward(model, ids, loops)
    with torch.no_grad():
        ref, _ = model(ids, loops=loops if model.core else None)
    diff = float((logits - ref).abs().max())
    print("traced vs forward max|dlogits| =", diff, flush=True)
    assert diff < 1e-3, "traced forward diverges from model.forward"
    val = np.memmap(DATA / "datasets" / ds / "val.bin", dtype=np.uint16, mode="r")
    T = 1024
    n_seq = int(os.environ.get("VIZ_EVAL_SEQS", "128"))  # the run reported 128 val seqs
    tot = 0.0
    for s in range(n_seq):
        rows = torch.from_numpy(val[s * T:s * T + T + 1].astype(np.int64))[None]
        with torch.no_grad():
            _, loss = model(rows[:, :-1], rows[:, 1:], loops=loops if model.core else None)
        tot += float(loss)
    cpu_loss = tot / n_seq
    ref_loss = met.get(f"val_loss_r{loops}", met.get("val_loss"))
    print(f"CPU val loss ({n_seq} seqs, fp32, reference GDN) = {cpu_loss:.4f}; run reported val_loss(r={loops}, 128 seqs, bf16, fla) = {ref_loss}", flush=True)

    # --- greedy continuation
    cur = ids.clone()
    cont = []
    eos = tok.token_to_id("<|eos|>")
    with torch.no_grad():
        for _ in range(N_NEW):
            lg, _ = model(cur[:, -model.cfg.max_seq_len:], loops=loops if model.core else None)
            row = lg[0, -1]
            t5 = top5(row, tok)
            nxt = int(row.argmax())
            if nxt == eos:
                break
            cont.append({"id": nxt, "piece": vis(tok.decode([nxt])), "top5": t5})
            cur = torch.cat([cur, torch.tensor([[nxt]])], 1)
    cont_text = tok.decode(cur[0, ids.shape[1]:].tolist())
    print("continuation:", repr(cont_text), flush=True)

    toks = []
    for tid, off in zip(enc.ids, enc.offsets):
        piece = PROMPT[off[0]:off[1]] if off[1] > off[0] else tok.decode([tid])
        toks.append({"id": tid, "piece": piece, "vis": vis(piece)})
    out = {
        "measured": True,
        "prompt": {"raw": RAW, "text": PROMPT, "text_visible": vis(PROMPT),
                   "note": "The model is a Python code model with no chat format, so the request is written as a leading comment."},
        "tokenizer": {"dataset": ds, "vocab_size": mc["vocab_size"], "file": f"datasets/{ds}/tokenizer.json"},
        "tokens": toks,
        "model": {"experiment": exp, "run": ck.parent.name, "config_file": summ.get("config", {}).get("config_file", ""),
                  "checkpoint": str(ck), "step": met.get("steps"), "params_total": met.get("params_total"),
                  "prelude": mc["prelude"], "core": mc["core"], "coda": mc["coda"], "loops": loops,
                  "window": mc["window"], "val_loss_reported": ref_loss},
        "check": {"traced_vs_forward_max_abs_logit_diff": diff, "cpu_val_loss_fp32": round(cpu_loss, 4), "cpu_val_seqs": n_seq,
                  "reported_val_loss": ref_loss,
                  "note": "GDN on CPU uses a plain-PyTorch reference of the gated delta rule instead of fla's kernel; fp32 not bf16."},
        "top_k": TOPK,
        "layers": rec["layers"],
        "delta": rec["delta"],
        "next_top5": top5(logits[0, -1], tok),
        "continuation": {"text": cont_text, "tokens": cont, "decoding": "greedy (argmax), r=%d loops" % loops},
        "how_regenerated": "bash tools/launch_cpu_job.sh --name viz-trace -- ~/.venvs/gogeta/bin/python experiments/viz/export_trace.py (CPU; with a GPU the real fla kernels give the same trace up to bf16 noise).",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KiB)", flush=True)


if __name__ == "__main__":
    main()
