#!/usr/bin/env python
"""Gogeta-0 architecture visualiser: a tiny local web server (stdlib only).

Builds the *real* model from gogeta/model.py + configs/*.json on every request (cached by file
mtime) and serves a JSON description plus the static frontend in tools/viz/static/.

    python tools/viz/server.py [--port 7870]      # binds 127.0.0.1 only

Run it CPU-only (tools/launch_viz.sh does this): nothing here touches CUDA.
"""
from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATIC = HERE / "static"
CONFIGS = REPO / "configs"
REGISTRY = REPO / "experiments" / "experiments.json"
MODEL_PY = REPO / "gogeta" / "model.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CTX_STOPS = [1024, 2048, 4096, 8192, 16384, 32768]
BYTES_BF16 = 2
LOCK = threading.Lock()
_model_mod = {"mtime": None, "mod": None}
_cache: dict = {}


# ----------------------------------------------------------------------------- code + config loading
def _torch():
    import torch
    torch.set_num_threads(2)
    return torch


def model_module():
    """gogeta.model, reloaded whenever model.py changes on disk."""
    _torch()
    mt = MODEL_PY.stat().st_mtime_ns
    if _model_mod["mod"] is None:
        _model_mod["mod"] = importlib.import_module("gogeta.model")
        _model_mod["mtime"] = mt
    elif _model_mod["mtime"] != mt:
        _model_mod["mod"] = importlib.reload(_model_mod["mod"])
        _model_mod["mtime"] = mt
        _cache.clear()
    return _model_mod["mod"]


def _exp_sort(p: Path):
    try:
        e = json.loads(p.read_text(encoding="utf-8")).get("experiment", "")
        return (int(e[1:]) if e[1:].isdigit() else 999, p.name)
    except Exception:
        return (999, p.name)


def load_registry() -> dict:
    try:
        return {e["id"]: e for e in json.loads(REGISTRY.read_text(encoding="utf-8"))}
    except Exception:
        return {}


def list_configs() -> list[dict]:
    reg = load_registry()
    out = []
    for p in sorted(CONFIGS.glob("*.json"), key=_exp_sort):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # keep listing even if one file is broken
            out.append({"name": p.stem, "error": str(e)})
            continue
        eid = c.get("experiment")
        r = reg.get(eid, {})
        out.append({"name": p.stem, "experiment": eid, "arch": c.get("arch", "fusiondance"),
                    "description": c.get("description", ""), "title": r.get("title"),
                    "status": r.get("status"), "val_loss": (r.get("results") or {}).get("val_loss")})
    return out


def model_cfg_dict(raw: dict) -> tuple[dict, list[str]]:
    """Return (ModelConfig kwargs, notes). hf_llama configs are mapped onto an equivalent stack."""
    notes: list[str] = []
    m = dict(raw["model"])
    if raw.get("arch") == "hf_llama":
        notes.append("arch hf_llama: Hugging Face's unmodified LlamaForCausalLM, not FusionDance. "
                     "Drawn as its equivalent stack of full-attention + SwiGLU blocks "
                     "(HF's init, RoPE layout and norm eps differ, but shapes and param counts match).")
        n = m["num_hidden_layers"]
        if m.get("num_key_value_heads", m["num_attention_heads"]) != m["num_attention_heads"]:
            notes.append("GQA in the config is not modelled here (counted as full multi-head).")
        vocab = m.get("vocab_size")
        if vocab is None:
            vocab = 8192
            notes.append("vocab_size is not in the config; assumed 8192 (same tokenizer as E0).")
        m = {"vocab_size": vocab, "d_model": m["hidden_size"], "n_heads": m["num_attention_heads"],
             "mlp_hidden": m["intermediate_size"], "prelude": ["attn"] * n, "core": [], "coda": [],
             "max_seq_len": m.get("max_position_embeddings", 1024), "ternary": False,
             "tie_embeddings": bool(m.get("tie_word_embeddings", True))}
    return m, notes


# ----------------------------------------------------------------------------- description
def _np(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


def _bitparams(mod, BitLinear) -> tuple[int, int]:
    """(# BitLinear modules, params inside them) under mod."""
    n = p = 0
    for m in mod.modules():
        if isinstance(m, BitLinear):
            n += 1
            p += m.weight.numel()
    return n, p


def _parts(mod, BitLinear) -> list[dict]:
    """Direct-child parameter breakdown, for tooltips."""
    out = []
    own = sum(p.numel() for p in mod.parameters(recurse=False))
    if own:
        out.append({"name": "own tensors (A_log, dt_bias)", "params": own, "ternary": False})
    for name, ch in mod.named_children():
        n = _np(ch)
        if n:
            out.append({"name": name, "params": n, "ternary": isinstance(ch, BitLinear)})
    return out


def describe(cfg_name: str, loops: int | None, mt_key) -> dict:
    mm = model_module()
    raw = json.loads((CONFIGS / f"{cfg_name}.json").read_text(encoding="utf-8"))
    mcfg_d, notes = model_cfg_dict(raw)
    cfg = mm.ModelConfig(**mcfg_d)
    ckey = (cfg_name, mt_key)
    model = _cache.get(ckey)
    if model is None:
        _torch().manual_seed(0)
        model = mm.FusionDance(cfg).eval()
        _cache.clear()
        _cache[ckey] = model
    BL = mm.BitLinear
    looped = bool(model.core)
    r = int(loops) if (loops and looped) else (cfg.loops if looped else 1)
    r = max(1, min(r, 64))
    d, H, hd, W = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.window

    def block(b, section, idx):
        nb, pb = _bitparams(b, BL)
        mixer_p, mlp_p = _np(b.mixer), _np(b.mlp)
        norm_p = _np(b.norm1) + _np(b.norm2)
        mb, mpb = _bitparams(b.mixer, BL)
        lb, lpb = _bitparams(b.mlp, BL)
        info = {"section": section, "index": idx, "kind": b.kind,
                "mixer_type": {"attn": "full causal softmax attention (RoPE)",
                               "swa": f"sliding-window attention (w={W}, RoPE)",
                               "gdn": "Gated DeltaNet (fixed-size state)"}.get(b.kind, b.kind),
                "mlp": {"type": "SwiGLU", "d_in": d, "hidden": cfg.mlp_hidden},
                "params": {"mixer": mixer_p, "mlp": mlp_p, "norms": norm_p, "total": mixer_p + mlp_p + norm_p},
                "mixer_parts": _parts(b.mixer, BL), "mlp_parts": _parts(b.mlp, BL),
                "bitlinear": {"count": nb, "params": pb, "mixer_params": mpb, "mlp_params": lpb,
                              "mixer_count": mb, "mlp_count": lb}}
        if b.kind == "gdn":
            info["state_elems"] = H * hd * hd
            info["state_note"] = f"{H} heads x {hd}x{hd}"
        else:
            info["kv_elems_per_token"] = 2 * d
            if b.kind == "swa":
                info["kv_cap_tokens"] = W
        return info

    sections = []
    for name, mods in (("prelude", model.prelude), ("core", model.core), ("coda", model.coda)):
        if not len(mods):
            continue
        s = {"name": name, "blocks": [block(b, name, i) for i, b in enumerate(mods)],
             "repeat": r if name == "core" else 1}
        if name == "core":
            lin = _np(model.inject)
            ip = _np(model.inj_norm_h) + _np(model.inj_norm_e) + lin
            _, ibp = _bitparams(model.inject, BL)
            s["injection"] = {"params": ip, "linear_params": lin, "norms": ip - lin,
                              "bitlinear_params": ibp, "shape": f"{2 * d} -> {d}",
                              "formula": "h = h + inject([norm(h), norm(e)])"}
        sections.append(s)

    pc = model.param_counts()
    bit_n = bit_p = 0
    for m in model.modules():
        if isinstance(m, BL):
            bit_n += 1
            bit_p += m.weight.numel()
    total = pc["total"]
    bf16_bytes = total * BYTES_BF16
    packed_bytes = (total - bit_p) * BYTES_BF16 + bit_p * 2 / 8  # ternary stored as 2 bits/weight

    # per-application counts (the core runs r times)
    apps = {"attn": 0, "swa": 0, "gdn": 0}
    for k in list(cfg.prelude) + list(cfg.coda) + r * list(cfg.core if looped else []):
        apps[k] += 1
    state_elems = H * hd * hd
    fl_loops = r if looped else None

    def at_ctx(n: int) -> dict:
        w = min(W, n)
        return {"ctx": n, "train_flops_per_token": model.flops_per_token(n, loops=fl_loops, training=True),
                "attn_kv_bytes": apps["attn"] * 2 * d * n * BYTES_BF16,
                "swa_kv_bytes": apps["swa"] * 2 * d * w * BYTES_BF16,
                "gdn_state_bytes": apps["gdn"] * state_elems * BYTES_BF16,
                "attn_score_cells": apps["attn"] * n * (n + 1) // 2,
                "swa_score_cells": apps["swa"] * (n * w - w * (w - 1) // 2),
                "gdn_state_cells": apps["gdn"] * state_elems}

    reg = load_registry().get(raw.get("experiment"), {})
    all_kinds = list(cfg.prelude) + list(cfg.core) + list(cfg.coda)
    return {
        "name": cfg_name, "experiment": raw.get("experiment"), "arch": raw.get("arch", "fusiondance"),
        "description": raw.get("description", ""), "notes": notes,
        "config": cfg.to_dict(),
        "looped": looped, "loops": r,
        "loops_range": {"default": cfg.loops, "train_min": cfg.loops_min, "train_max": cfg.loops_max,
                        "bptt": cfg.bptt_loops} if looped else None,
        "sections": sections,
        "embedding": {"params": pc["embedding"], "vocab": cfg.vocab_size, "d_model": d, "tied": cfg.tie_embeddings},
        "ternary": {"enabled": cfg.ternary, "bitlinear_count": bit_n, "bitlinear_params": bit_p,
                    "fraction_of_total": bit_p / total, "fraction_of_non_embedding": bit_p / pc["non_embedding"]},
        "params": pc,
        "distinct_blocks": len(all_kinds),
        "effective_depth": model.effective_depth(r) if looped else model.effective_depth(),
        "applications": apps,
        "flops": {"seq_len": 1024,
                  "train_per_token": model.flops_per_token(1024, loops=fl_loops),
                  "infer_per_token": model.flops_per_token(1024, loops=fl_loops, training=False),
                  "formula": "model.flops_per_token (6N rule + attention/GDN terms)"},
        "by_ctx": [at_ctx(n) for n in CTX_STOPS],
        "memory": {"bf16_bytes": bf16_bytes, "packed_bytes": packed_bytes, "ternary_packed": cfg.ternary},
        "gdn": ({"state_elems_per_layer": state_elems, "heads": H, "head_dim": hd}
                if "gdn" in all_kinds else None),
        "attention": {"kv_elems_per_token_per_layer": 2 * d, "window": W},
        "registry": ({"status": reg.get("status"), "version": reg.get("version"), "title": reg.get("title"),
                      "val_loss": (reg.get("results") or {}).get("val_loss"), "arc": reg.get("arc")}
                     if reg else None),
    }


def api_model(name: str, loops: int | None):
    path = CONFIGS / f"{name}.json"
    if "/" in name or "\\" in name or ".." in name or not path.is_file():
        return 404, {"error": f"no such config: {name}"}
    with LOCK:
        try:
            return 200, describe(name, loops, (path.stat().st_mtime_ns, MODEL_PY.stat().st_mtime_ns))
        except Exception as e:  # report build failures to the page instead of a blank screen
            import traceback
            return 500, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


# ----------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "GogetaViz/1"

    def log_message(self, fmt, *a):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)
        if p == "/api/models":
            return self._json(200, {"models": list_configs()})
        if p.startswith("/api/model/"):
            name = p[len("/api/model/"):]
            try:
                loops = int(q["loops"][0]) if "loops" in q else None
            except ValueError:
                return self._json(400, {"error": "loops must be an integer"})
            code, obj = api_model(name, loops)
            return self._json(code, obj)
        if p == "/api/health":
            return self._json(200, {"ok": True, "app": "gogeta-viz"})
        rel = "index.html" if p in ("/", "") else p.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC not in f.parents or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7870)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Gogeta-0 viz on http://127.0.0.1:{a.port}  (repo {REPO})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
