"""A3 design: generate and verify the staged configs, the patched-model copy, the cost/power tables.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/design/build_staged.py

CPU only, construction and counting only (no GDN forward: the fla kernels are GPU-only).
Writes (all under experiments/design/ or experiments/staged-configs/, never under configs/ or gogeta/):
  experiments/staged-configs/e<NN>_*.json            the staged configs
  experiments/staged-configs/patches/gdn_ablation_flags.patch   proposed gogeta/model.py change (3 flags)
  experiments/design/model_gdn_flags.py              patched copy of gogeta/model.py used only to verify
  experiments/design/staged-table.json / .md         params, FLOPs, estimated and measured cost per config
  experiments/design/proposed-registry.json          registry entries for the staged ids
IDs E20-E59 are reserved for A3 (the main agent renumbers on integration if needed).
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gogeta.model import FusionDance, ModelConfig  # noqa: E402

OUT = ROOT / "experiments" / "staged-configs"
DES = ROOT / "experiments" / "design"
V, SEQ = 8192, 1024

# ------------------------------------------------------------------ patched model (flags for GDN ablations)
def make_patched() -> str:
    src = (ROOT / "gogeta" / "model.py").read_text()
    reps = [
        ("    gdn_neg_eigval: bool = False  # GDN write strength beta in (0, 2) instead of (0, 1): negative eigenvalues (SQ2/E11)\n",
         "    gdn_neg_eigval: bool = False  # GDN write strength beta in (0, 2) instead of (0, 1): negative eigenvalues (SQ2/E11)\n"
         "    gdn_conv: bool = True         # GDN short causal convs on q/k/v (False: identity)   [E1 decomposition]\n"
         "    gdn_gate: bool = True         # GDN SiLU output gate (False: no gate projection)    [E1 decomposition]\n"
         "    gdn_out_norm: bool = True     # GDN per-head output RMSNorm (False: identity)       [E1 decomposition]\n"),
        ("        self.gate = make_linear(cfg, D, H * hd)\n",
         "        self.gate = make_linear(cfg, D, H * hd) if cfg.gdn_gate else None\n"),
        ("        self.conv_q, self.conv_k, self.conv_v = (ShortConv(H * hd, cfg.conv_size) for _ in range(3))\n",
         "        self.conv_q, self.conv_k, self.conv_v = (\n"
         "            ShortConv(H * hd, cfg.conv_size) if cfg.gdn_conv else nn.Identity() for _ in range(3))\n"),
        ("        self.out_norm = nn.RMSNorm(hd)\n",
         "        self.out_norm = nn.RMSNorm(hd) if cfg.gdn_out_norm else nn.Identity()\n"),
        ("        o = self.out_norm(o.to(x.dtype)) * F.silu(self.gate(x).view(B, T, H, hd))\n",
         "        o = self.out_norm(o.to(x.dtype))\n"
         "        if self.gate is not None:\n"
         "            o = o * F.silu(self.gate(x).view(B, T, H, hd))\n"),
    ]
    for a, b in reps:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    return src


def write_patch() -> None:
    old = (ROOT / "gogeta" / "model.py").read_text().splitlines(keepends=True)
    new = make_patched().splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(old, new, "a/gogeta/model.py", "b/gogeta/model.py"))
    (OUT / "patches").mkdir(parents=True, exist_ok=True)
    (OUT / "patches" / "gdn_ablation_flags.patch").write_text(diff)
    (DES / "model_gdn_flags.py").write_text(make_patched())


def patched_module():
    spec = importlib.util.spec_from_file_location("model_gdn_flags", DES / "model_gdn_flags.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["model_gdn_flags"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ layouts
S_HYB14 = ["swa"] + ["gdn", "gdn", "swa"] * 4 + ["attn"]       # E2's executed mixer order (r=4), 14 blocks
S_DEN14 = ["attn"] * 14
E1_LAYOUT = ["swa", "gdn", "gdn", "swa", "gdn", "attn"]
BASE_MODEL = dict(vocab_size=V, d_model=384, n_heads=6, mlp_hidden=1024, window=256, max_seq_len=SEQ, ternary=False)


def unlooped(seq):
    return dict(prelude=list(seq), core=[], coda=[])


def looped(seq):  # prelude 1, core 3, coda 1, r=4 (E2's loop settings)
    assert len(seq) == 14
    return dict(prelude=[seq[0]], core=list(seq[1:4]), coda=[seq[13]], loops=4, loops_min=1, loops_max=7, bptt_loops=4)


def train(max_tokens, lr, dataset, warmup=300, extra=None):
    t = dict(dataset=dataset, seq_len=SEQ, batch_size=32, max_tokens=max_tokens, lr=lr, min_lr_frac=0.1,
             warmup_steps=warmup, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0, eval_every=250,
             eval_seqs=128, seed=0)
    t.update(extra or {})
    return t


V1, V2 = "code-py-v1", "code-py-v2"   # v2 = deduplicated train set (A1); does not exist yet
M200, M20 = 200_000_000, 20_000_000

# id -> dict(file, desc, model, train, group, arch (opt), lr_note)
SPECS: dict[str, dict] = {}


def add(eid, slug, desc, model, tr, group, arch=None, patched=False):
    SPECS[eid] = dict(file=f"{eid.lower()}_{slug}.json", desc=desc, model=model, train=tr, group=group,
                      arch=arch, patched=patched)


def mdl(**kw):
    m = dict(BASE_MODEL)
    m.update(kw)
    return m


# ---- (3) 2x2x2 factorial on the deduplicated set, 20M tokens, depth budget = 14 block applications
cells = {}
for g in (0, 1):
    for lp in (0, 1):
        for t in (0, 1):
            seq = S_HYB14 if g else S_DEN14
            layout = looped(seq) if lp else unlooped(seq)
            eid = f"E{20 + g * 4 + lp * 2 + t}"
            name = f"{g}{lp}{t}"
            cells[name] = eid
            bits = [("hybrid GDN/SWA/attn mixers" if g else "all full-attention mixers"),
                    ("shared loop (prelude 1, core 3 x r=4, coda 1, re-injection)" if lp else "unlooped, 14 unique blocks"),
                    ("ternary BitLinear" if t else "bf16/fp weights")]
            add(eid, f"fact{name}", f"Factorial cell {name} (GDN={g}, loops={lp}, ternary={t}): " + "; ".join(bits)
                + ". Depth budget 14 block applications, d=384, 20M tokens, deduplicated train set. LR comes from the cell's own "
                  "G3 sweep (--lr), not from this file.",
                mdl(**layout, ternary=bool(t)), train(M20, 0.006, V2), "factorial")

# ---- (4) E1 decomposition (one factor at a time vs E1), deduplicated set, 20M tokens
dec = [
    ("E40", "gdn2attn", "GDN blocks replaced by full attention (mixer-type test); SWA slots kept",
     dict(prelude=["swa", "attn", "attn", "swa", "attn", "attn"]), False),
    ("E41", "swaonly", "SWA-only stack: GDN blocks replaced by SWA (no recurrent state, window 256, one full-attention last)",
     dict(prelude=["swa", "swa", "swa", "swa", "swa", "attn"]), False),
    ("E42", "gdn_noconv", "E1 with the GDN short causal convs removed (identity)", dict(gdn_conv=False), True),
    ("E43", "gdn_nogate", "E1 with the GDN SiLU output gate removed (no gate projection)", dict(gdn_gate=False), True),
    ("E44", "gdn_noonorm", "E1 with the GDN per-head output RMSNorm removed", dict(gdn_out_norm=False), True),
    ("E45", "gdn_bare", "E1 with conv, gate and output norm all removed (bare delta rule + q/k/v/o)",
     dict(gdn_conv=False, gdn_gate=False, gdn_out_norm=False), True),
    ("E46", "attnfirst", "Attention placement: the single full-attention layer first, final slot becomes SWA",
     dict(prelude=["attn", "gdn", "gdn", "swa", "gdn", "swa"]), False),
    ("E47", "attnmid", "Attention placement: full-attention layer in the middle (slot 4), final slot SWA",
     dict(prelude=["swa", "gdn", "gdn", "attn", "gdn", "swa"]), False),
    ("E48", "gdn_noswa", "All SWA replaced by GDN (pure-GDN stack + one full-attention last; Qwen3.8-style, SQ9)",
     dict(prelude=["gdn", "gdn", "gdn", "gdn", "gdn", "attn"]), False),
]
for eid, slug, desc, over, patched in dec:
    add(eid, slug, desc + ". Everything else identical to E1 (6 blocks, d=384). 20M tokens, deduplicated set; LR from its own sweep (--lr).",
        mdl(**{**dict(prelude=list(E1_LAYOUT), core=[], coda=[]), **over}), train(M20, 0.006, V2), "decomposition",
        patched=patched)

# ---- (5) HPO side checks: warmup sensitivity (20M, 5% warmup) and the dedup bridge (200M)
e10 = json.load(open(ROOT / "configs" / "e10_llama_ref.json"))
add("E50", "e10_warm30", "E10 (HF Llama control) with warmup_steps 30 (about 5% of a 20M-token run) instead of 300 (49%): "
    "does the 20M proxy's long warmup distort the K7 LR ranking?", e10["model"], train(M20, 0.003, V1, warmup=30),
    "hpo", arch="hf_llama")
add("E51", "e1_warm30", "E1 with warmup_steps 30 (about 5% of a 20M-token run) instead of 300 (49%).",
    mdl(prelude=list(E1_LAYOUT), core=[], coda=[]), train(M20, 0.006, V1, warmup=30), "hpo")
add("E52", "e10_v2", "Dedup bridge: E10 exactly as configs/e10_llama_ref.json but trained on the deduplicated set code-py-v2.",
    e10["model"], train(M200, 0.003, V2), "bridge", arch="hf_llama")
add("E53", "e1_v2", "Dedup bridge: E1 exactly as configs/e1_hybrid.json but trained on the deduplicated set code-py-v2.",
    mdl(prelude=list(E1_LAYOUT), core=[], coda=[]), train(M200, 0.006, V2), "bridge")


# ---- (2) matched-compute / matched-parameter arms (v1 so they sit next to Phase 2b's E10/E1/E2/E9)
def count(model_dict, arch=None, patched=False):
    if arch == "hf_llama":
        from gogeta.train import build_model
        m, _ = build_model({"arch": "hf_llama", "model": model_dict}, V, SEQ)
    else:
        mod = patched_module() if patched else sys.modules["gogeta.model"]
        m = mod.FusionDance(mod.ModelConfig(**model_dict))
    return m


def find_mlp(base_model, target, key, lo=256, hi=4096, step=8, **extra):
    """Params and FLOPs are linear in mlp_hidden: evaluate at two points, solve, round to a multiple of
    `step`, then pick the best of the neighbours."""
    def val(h):
        m = FusionDance(ModelConfig(**{**base_model, "mlp_hidden": h, **extra}))
        return m.param_counts()["non_embedding"] if key == "params" else m.flops_per_token(SEQ)
    v1, v2 = val(512), val(1024)
    h0 = 512 + (target - v1) * 512 / (v2 - v1)
    cands = [max(step, int(round(h0 / step)) * step + k * step) for k in (-1, 0, 1)]
    return min(cands, key=lambda h: abs(val(h) - target))


E1_cfg = {**dict(BASE_MODEL), **dict(prelude=list(E1_LAYOUT), core=[], coda=[])}
E1_m = FusionDance(ModelConfig(**E1_cfg))
E2_cfg = json.load(open(ROOT / "configs" / "e2_looped.json"))["model"]
E2_m = FusionDance(ModelConfig(**E2_cfg))
E1_PARAMS, E1_FLOPS = E1_m.param_counts()["non_embedding"], E1_m.flops_per_token(SEQ)
E2_FLOPS = E2_m.flops_per_token(SEQ)

den6 = mdl(prelude=["attn"] * 6, core=[], coda=[])
h30 = find_mlp(den6, E1_PARAMS, "params")
h31 = find_mlp(den6, E1_FLOPS, "flops")
add("E30", "dense_parammatch", f"Dense (6 full-attention blocks, FusionDance code) with mlp_hidden {h30} so non-embedding params match E1 "
    f"({E1_PARAMS:,}). Control for 'E1 wins because it has 4.4% more params'. LR_E10.",
    {**den6, "mlp_hidden": h30}, train(M200, 0.003, V1), "matched")
add("E31", "dense_flopmatch", f"Dense (6 full-attention blocks) with mlp_hidden {h31} so train FLOPs/token match E1 "
    f"({E1_FLOPS:,.0f}). Control for 'E1 wins at fewer FLOPs'. LR_E10.",
    {**den6, "mlp_hidden": h31}, train(M200, 0.003, V1), "matched")
unl14 = mdl(**unlooped(S_HYB14))
add("E32", "unshared14_e2flops", "14 UNSHARED blocks in E2's executed mixer order (swa, 4x[gdn,gdn,swa], attn), no loop, no re-injection: "
    "the unshared control at E2's FLOPs (outside review idea F). LR_E1.", unl14, train(M200, 0.006, V1), "matched")
S_HYB12 = ["swa", "gdn", "gdn", "swa", "gdn", "gdn", "swa", "gdn", "gdn", "swa", "gdn", "attn"]
b12 = mdl(**unlooped(S_HYB12))
h33 = find_mlp(b12, E2_FLOPS, "flops")
add("E33", "unshared12_e2flops", f"12 unshared blocks total (the review's wording), same mixer family, mlp_hidden {h33} so FLOPs match E2's "
    f"({E2_FLOPS:,.0f}). LR_E1.", {**b12, "mlp_hidden": h33}, train(M200, 0.006, V1), "matched")
b34 = mdl(d_model=576, n_heads=9, **unlooped(["swa", "gdn", "gdn", "swa", "attn"]))
h34 = find_mlp(b34, E2_FLOPS, "flops", lo=512, hi=6000)
add("E34", "e9_wide_e2flops", f"E9's 5-block unlooped layout widened (d=576, 9 heads of 64, mlp_hidden {h34}) to E2's FLOPs: the width way "
    "of giving E9 the compute that looping gave E2. LR_E1 (assumption: not retuned at the wider width).",
    {**b34, "mlp_hidden": h34}, train(M200, 0.006, V1), "matched")


# params-matched (not FLOPs-matched) unlooped partners for the loop factor: sensitivity to the matching rule
add("E35", "dense5_unlooped", "Factorial matching-rule sensitivity (M2, params-matched): 5 unshared full-attention blocks, no loop (the dense "
    "analogue of E9). Compare with E22 (looped dense, same 5 unique blocks + injection). 20M tokens, deduplicated set.",
    mdl(**unlooped(["attn"] * 5)), train(M20, 0.006, V2), "factorial")
add("E36", "e9layout_v2", "Factorial matching-rule sensitivity (M2, params-matched): E9's layout (5 unshared hybrid blocks), no loop. "
    "Compare with E26. 20M tokens, deduplicated set.",
    mdl(**unlooped(["swa", "gdn", "gdn", "swa", "attn"])), train(M20, 0.006, V2), "factorial")

# ------------------------------------------------------------------ cost model
# Achieved TFLOP/s by class, from MEASURED runs (see staged-table.md): short (20M, K7, compiled) and long (200M) runs.
# A looped hybrid (E2, 200M) achieves the same TFLOP/s as unlooped E1 (8.0 vs 7.9), so the class is "has GDN?".
TF = {"20M": {"dense": 17.4, "gdn": 12.2}, "200M": {"dense": 13.8, "gdn": 7.9}}
TERNARY_SPEED = 0.75   # ASSUMPTION: BitLinear is slower (K2, eager); not measured under compile (SQ76 pending)
GAP_MIN = 1.0          # campaign gap between runs


def est_wall_min(row, model, arch):
    kinds = list(model.get("prelude", [])) + list(model.get("core", [])) + list(model.get("coda", []))
    cls = "gdn" if "gdn" in kinds else "dense"
    if not model:      # reference configs: classify by id
        cls = "gdn" if row["id"] in ("E1", "E2", "E3", "E9") else "dense"
    tf = TF["20M" if row["max_tokens"] <= 20_000_000 else "200M"][cls]
    if row.get("ternary"):
        tf *= TERNARY_SPEED
    return row["max_tokens"] / (tf * 1e12 / row["flops_train"]) / 60 + GAP_MIN


def measured_runs():
    import glob
    import os
    out = []
    for d in sorted(glob.glob(os.path.expanduser("~/fusiondance-data/runs/*"))):
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        m = j["metrics"]
        if j["experiment"] == "smoke" or "flops_per_token_default" not in m:
            continue
        ov = j["config"].get("cli_overrides", {})
        out.append(dict(run=j["name"], lr=ov.get("lr", j["config"]["train"]["lr"]), tokens=m["tokens"],
                        tok_s=round(m["tokens_per_s"]), tflops=round(m["tokens_per_s"] * m["flops_per_token_default"] / 1e12, 2),
                        wall_min=round(m["wall_min"], 2), peak_gb=round(m["peak_mem_gb"], 2), val_loss=round(m["val_loss"], 4)))
    return out


def write_md(rows, refs, measured):
    L = ["# Staged configs: params, FLOPs, cost (generated by build_staged.py; do not edit)", "",
         "FLOPs = `flops_per_token(1024)` training (6N rule + attention/GDN terms, core counted r=4 times). It does not credit ternary "
         "with any saving and counts SWA as O(window) although our kernel is dense-masked (C12), so measured cost differs.", "",
         "| id | file | dataset | tokens | LR in file | unique blocks | non-emb params | train FLOPs/tok | x E1 | x E2 | est. wall min |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    e1 = next(r for r in refs if r["id"] == "E1")["flops_train"]
    e2 = next(r for r in refs if r["id"] == "E2")["flops_train"]
    for r in refs + rows:
        L.append(f"| {r['id']} | {r['file']} | {r['dataset']} | {r['max_tokens']/1e6:.0f}M | {r['lr_file']} | {r['unique_blocks'] or ''} | "
                 f"{r['params_non_embedding']:,} | {r['flops_train']:,.0f} | {r['flops_train']/e1:.3f} | {r['flops_train']/e2:.3f} | {r['est_wall_min']:.1f} |")
    L += ["", "## Measured cost (from ~/fusiondance-data/runs/*/summary.json)", "",
          "tok/s = tokens / wall (compile and evals included). TFLOP/s = tok/s x analytic train FLOPs/token (our accounting).", "",
          "| run | LR | tokens | tok/s | TFLOP/s | wall min | peak GB | val loss |", "|---|---|---|---|---|---|---|---|"]
    for m in measured:
        L.append(f"| {m['run']} | {m['lr']} | {m['tokens']/1e6:.0f}M | {m['tok_s']:,} | {m['tflops']} | {m['wall_min']} | {m['peak_gb']} | {m['val_loss']} |")
    L += ["", f"Cost model for the estimates: achieved TFLOP/s = {TF}, ternary x{TERNARY_SPEED} (assumed), +{GAP_MIN:.0f} min campaign gap. "
          "Refresh `TF` from Phase 2b's 200M measurements when they land."]
    (DES / "staged-table.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ write + verify
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DES.mkdir(parents=True, exist_ok=True)
    write_patch()
    # patch parity: with the three flags at their defaults, E1/E2/E3 construct bit-identically under the patched model
    import torch
    mod = patched_module()
    for name in ["e1_hybrid", "e2_looped", "e3_ternary"]:
        c = json.load(open(ROOT / "configs" / f"{name}.json"))["model"]
        torch.manual_seed(0)
        a = FusionDance(ModelConfig(**c))
        ra = torch.get_rng_state()
        torch.manual_seed(0)
        b = mod.FusionDance(mod.ModelConfig(**c))
        same = (list(a.state_dict()) == list(b.state_dict()) and all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
                and torch.equal(ra, torch.get_rng_state()))
        print(f"patch parity {name}: {'IDENTICAL' if same else 'DIFFERENT'}")
        assert same
    rows = []
    for eid, s in sorted(SPECS.items(), key=lambda kv: int(kv[0][1:])):
        cfg = {"experiment": eid, "description": s["desc"]}
        if s["arch"]:
            cfg["arch"] = s["arch"]
        cfg["model"] = s["model"]
        cfg["train"] = s["train"]
        if s["patched"]:
            cfg["requires_patch"] = "experiments/staged-configs/patches/gdn_ablation_flags.patch (gogeta/model.py flags)"
        (OUT / s["file"]).write_text(json.dumps(cfg, indent=2) + "\n")
        # verify: rebuild from the file we just wrote
        back = json.loads((OUT / s["file"]).read_text())
        m = count(back["model"], back.get("arch"), s["patched"])
        pc = m.param_counts()
        fl_tr = m.flops_per_token(SEQ)
        fl_fw = m.flops_per_token(SEQ, training=False)
        rows.append(dict(id=eid, file=s["file"], group=s["group"], params_total=pc["total"],
                         params_non_embedding=pc["non_embedding"], flops_train=fl_tr, flops_fwd=fl_fw,
                         dataset=back["train"]["dataset"], max_tokens=back["train"]["max_tokens"],
                         lr_file=back["train"]["lr"], ternary=back["model"].get("ternary", False),
                         unique_blocks=len(back["model"].get("prelude", [])) + len(back["model"].get("core", [])) + len(back["model"].get("coda", [])),
                         desc=back["description"][:110]))
    # reference rows from the committed configs
    refs = []
    for name in ["e0_dense", "e1_hybrid", "e2_looped", "e3_ternary", "e9_unlooped"]:
        c = json.load(open(ROOT / "configs" / f"{name}.json"))
        m = FusionDance(ModelConfig(**c["model"]))
        pc = m.param_counts()
        refs.append(dict(id=c["experiment"], file=f"configs/{name}.json", group="reference", params_total=pc["total"],
                         params_non_embedding=pc["non_embedding"], flops_train=m.flops_per_token(SEQ),
                         flops_fwd=m.flops_per_token(SEQ, training=False), dataset="code-py-v1", max_tokens=M200,
                         lr_file=c["train"]["lr"], ternary=c["model"]["ternary"], unique_blocks=0, desc=c["description"][:110]))
    from gogeta.train import build_model
    m, _ = build_model({"arch": "hf_llama", "model": e10["model"]}, V, SEQ)
    pc = m.param_counts()
    refs.append(dict(id="E10", file="configs/e10_llama_ref.json", group="reference", params_total=pc["total"],
                     params_non_embedding=pc["non_embedding"], flops_train=m.flops_per_token(SEQ),
                     flops_fwd=m.flops_per_token(SEQ, training=False), dataset="code-py-v1", max_tokens=M200,
                     lr_file=0.003, ternary=False, unique_blocks=0, desc="HF Llama control"))
    for r in rows + refs:
        r["est_wall_min"] = est_wall_min(r, SPECS.get(r["id"], {}).get("model") or {}, SPECS.get(r["id"], {}).get("arch"))
    measured = measured_runs()
    json.dump({"staged": rows, "reference": refs, "measured": measured}, open(DES / "staged-table.json", "w"), indent=1)
    write_md(rows, refs, measured)
    print(f"{'id':5} {'params non-emb':>15} {'train FLOPs/tok':>16} {'x E1':>6} {'x E2':>6}  file")
    for r in refs + rows:
        print(f"{r['id']:5} {r['params_non_embedding']:>15,} {r['flops_train']:>16,.0f} {r['flops_train']/E1_FLOPS:>6.3f} "
              f"{r['flops_train']/E2_FLOPS:>6.3f}  {r['file']}")
    print(f"mlp: E30 {h30}  E31 {h31}  E33 {h33}  E34 {h34}")


if __name__ == "__main__":
    main()
