"""Anchored edits that turn gogeta/train.py into the byte-capable trainer (the working copy train_bytes.py).

    python experiments/byte_model/train_edits.py [--base gogeta/train.py] [--out experiments/byte_model/train_bytes.py]

Each edit is (exact old text, new text) and must match exactly once, so re-running this on a newer gogeta/train.py
(e.g. after cockpit-resilience-train.patch or repro-manifest-at-run-end.patch land, which conflict textually with
byte-train.patch) either rebuilds the mirror or names the anchor that moved. Then re-run make_patches.py.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

EDITS = [('''--lr overrides train.lr (used by the K7 LR sweep); the effective value is what the run config
records under "train", and every CLI override is listed under "cli_overrides".
"""''', '''--lr overrides train.lr (used by the K7 LR sweep); the effective value is what the run config
records under "train", and every CLI override is listed under "cli_overrides".

Byte datasets (meta.json "unit": "bytes", e.g. code-py-v1-bytes, uint8, EOS = 0x00): the budget is train.max_bytes
(--max-bytes), and max_tokens is refused, so a byte budget is never read as tokens (or vice versa). A "token" in the
loop is then one byte position; runs also log bytes and token_equiv (bytes / 3.735). Configs with
model.patch_size > 0 build gogeta.model.PatchedByteLM (coarse-to-fine); refinement passes are sampled per step when
refine_max > refine_min, and per-pass / per-slot losses are logged. --device cpu runs a CPU smoke test (no autocast,
no fused AdamW, no CUDA calls).
"""'''),
('from gogeta.model import FusionDance, ModelConfig\n',
 'from experiments.byte_model.model_bytes import BYTES_PER_TOKEN, FusionDance, ModelConfig, make_model  # MIRROR\n'),
('''        self.train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
        self.T = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.chars_per_token = self.meta["splits"]["train"]["chars_per_token"]

    def batch(self, B: int, device: str):
        ix = torch.randint(len(self.train) - self.T - 1, (B,), generator=self.g).tolist()
        buf = np.stack([self.train[i:i + self.T + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(buf).pin_memory().to(device, non_blocking=True)''',
'''        dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.train = np.memmap(d / "train.bin", dtype=dtype, mode="r")
        self.val = np.memmap(d / "val.bin", dtype=dtype, mode="r")
        self.unit = self.meta.get("unit", "tokens")   # "bytes" for code-py-v1-bytes
        self.T = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.chars_per_token = self.meta["splits"]["train"]["chars_per_token"]   # chars per position (incl. EOS)

    def batch(self, B: int, device: str):
        ix = torch.randint(len(self.train) - self.T - 1, (B,), generator=self.g).tolist()
        buf = np.stack([self.train[i:i + self.T + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(buf)
        t = t.pin_memory().to(device, non_blocking=True) if device == "cuda" else t.to(device)'''),
('''class HFLlama(torch.nn.Module):''', '''class ByteTokenizer:
    """The tokenizer interface sample() uses, for byte datasets: ids are byte values, EOS = meta eos_id (0x00)."""

    def __init__(self, eos_id: int = 0):
        self.eos_id = eos_id

    def encode(self, text: str):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i != self.eos_id and 0 <= i < 256).decode("utf-8", errors="replace")

    def token_to_id(self, name: str):
        return self.eos_id if name == "<|eos|>" else None


class HFLlama(torch.nn.Module):'''),
('''        mc = ModelConfig(**{**cfg["model"], "vocab_size": vocab_size, "max_seq_len": seq_len})
        return FusionDance(mc), mc''', '''        mc = ModelConfig(**{**cfg["model"], "vocab_size": vocab_size, "max_seq_len": seq_len})
        return make_model(mc), mc   # FusionDance, or PatchedByteLM when model.patch_size > 0'''),
('''@torch.no_grad()
def evaluate(model, data, n_seqs, B, device, loops=None) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for x, y in data.val_batches(n_seqs, B, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, loops=loops)
        tot += loss.item() * x.numel()
        cnt += x.numel()
    model.train()
    return tot / cnt''', '''def autocast(device: str):
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")


def is_patched(model) -> bool:
    return getattr(getattr(model, "cfg", None), "patch_size", 0) > 0


@torch.no_grad()
def evaluate(model, data, n_seqs, B, device, loops=None, passes=None) -> float:
    model.eval()
    kw = {"passes": passes} if passes is not None else {}
    tot, cnt = 0.0, 0
    for x, y in data.val_batches(n_seqs, B, device):
        with autocast(device):
            _, loss = model(x, y, loops=loops, **kw)
        tot += loss.item() * x.numel()
        cnt += x.numel()
    model.train()
    return tot / cnt


@torch.no_grad()
def evaluate_detail(model, data, n_seqs, B, device) -> dict:
    """Patched byte models: val loss of every refinement pass and of every within-patch slot (patch lag)."""
    model.eval()
    P, r = model.cfg.patch_size, model.cfg.refine_passes
    passes, slots, cnt = torch.zeros(r), torch.zeros(P), 0
    for x, y in data.val_batches(n_seqs, B, device):
        with autocast(device):
            model(x, y, return_all=True)
        passes += torch.stack([v.float().cpu() for v in model.last_aux["pass_loss"]]) * x.numel()
        slots += model.last_aux["slot_loss"].float().cpu() * x.numel()
        cnt += x.numel()
    model.train()
    out = {f"val_loss_pass{k + 1}": v for k, v in enumerate((passes / cnt).tolist())}
    out.update({f"val_loss_slot{j}": v for j, v in enumerate((slots / cnt).tolist())})
    return out'''),
('''    x = torch.tensor([ids], device=device)
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):''', '''    x = torch.tensor([ids], device=device)
    for _ in range(n_new):
        with autocast(torch.device(device).type):'''),
('''    ap.add_argument("--max-tokens", type=int, help="override train.max_tokens")''', '''    ap.add_argument("--max-tokens", type=int, help="override train.max_tokens (token datasets)")
    ap.add_argument("--max-bytes", type=int, help="override train.max_bytes (byte datasets)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: smoke tests only")'''),
('''    if args.max_tokens:
        tc["max_tokens"] = args.max_tokens''', '''    if args.max_tokens:
        tc["max_tokens"] = args.max_tokens
    if args.max_bytes:
        tc["max_bytes"] = args.max_bytes'''),
('''    overrides = {k: v for k, v in (("max_tokens", args.max_tokens or None), ("seed", args.seed),
                                   ("lr", args.lr)) if v is not None}
    device = "cuda"
    torch.manual_seed(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    data = TokenData(tc["dataset"], tc["seq_len"], tc["seed"])''', '''    overrides = {k: v for k, v in (("max_tokens", args.max_tokens or None), ("max_bytes", args.max_bytes or None),
                                   ("seed", args.seed), ("lr", args.lr)) if v is not None}
    if args.device != "cuda":
        overrides["device"] = args.device
    device = args.device
    cuda = device == "cuda"
    torch.manual_seed(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    data = TokenData(tc["dataset"], tc["seq_len"], tc["seed"])
    budget_key = "max_bytes" if data.unit == "bytes" else "max_tokens"
    other_key = "max_tokens" if budget_key == "max_bytes" else "max_bytes"
    if other_key in tc or budget_key not in tc:
        raise SystemExit(f"dataset {tc['dataset']} counts {data.unit}: the budget must be train.{budget_key} "
                         f"(found {sorted(k for k in tc if k.startswith('max_'))})")'''),
('''    pc = model.param_counts()
    looped = len(model.core) > 0
    loop_rng = random.Random(tc["seed"] + 1)''', '''    pc = model.param_counts()
    looped = len(model.core) > 0
    loop_rng = random.Random(tc["seed"] + 1)
    patched = is_patched(model)
    sample_passes = patched and mc.refine_max > mc.refine_min
    pass_rng = random.Random(tc["seed"] + 2)'''),
('''                            lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), fused=True)

    tokens_per_step = tc["batch_size"] * tc["seq_len"]
    total_steps = args.bench or math.ceil(tc["max_tokens"] / tokens_per_step)''', '''                            lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), fused=cuda)

    tokens_per_step = tc["batch_size"] * tc["seq_len"]   # positions: tokens, or bytes for byte datasets
    total_steps = args.bench or math.ceil(tc[budget_key] / tokens_per_step)'''),
('''    print(f"params: total {pc['total']/1e6:.2f}M non-emb {pc['non_embedding']/1e6:.2f}M | "
          f"steps {total_steps} x {tokens_per_step} tok | looped={looped}", flush=True)''', '''    unit = "bytes" if data.unit == "bytes" else "tok"
    extra = f" | patched P={mc.patch_size} r={mc.refine_passes} {mc.decoder}" if patched else ""
    print(f"params: total {pc['total']/1e6:.2f}M non-emb {pc['non_embedding']/1e6:.2f}M | "
          f"steps {total_steps} x {tokens_per_step} {unit} | looped={looped}{extra}", flush=True)'''),
('''"compile": args.compile, "dataset_meta": {k: data.meta[k] for k in ("name", "source", "vocab_size")}}''',
 '''"compile": args.compile, "dataset_meta": {k: data.meta.get(k) for k in ("name", "source", "vocab_size", "unit")},
                   "position_unit": data.unit}'''),
('''    t_log, tok_log = time.time(), 0
    torch.cuda.reset_peak_memory_stats()''', '''    t_log, tok_log = time.time(), 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()'''),
('''            r = loop_rng.randint(mc.loops_min, mc.loops_max) if looped else None
            x, y = data.batch(tc["batch_size"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd_model(x, y, loops=r)''', '''            r = loop_rng.randint(mc.loops_min, mc.loops_max) if looped else None
            npass = pass_rng.randint(mc.refine_min, mc.refine_max) if sample_passes else None
            kw = {"passes": npass} if npass is not None else {}
            x, y = data.batch(tc["batch_size"], device)
            with autocast(device):
                _, loss = fwd_model(x, y, loops=r, **kw)'''),
('''            flops += model.flops_per_token(tc["seq_len"], loops=r) * tokens_per_step
            tok_log += tokens_per_step
            if step % 10 == 0 or step == total_steps - 1:
                torch.cuda.synchronize()''', '''            flops += model.flops_per_token(tc["seq_len"], loops=r, **kw) * tokens_per_step
            tok_log += tokens_per_step
            if step % 10 == 0 or step == total_steps - 1:
                if cuda:
                    torch.cuda.synchronize()'''),
('''                if looped:
                    m["train/loops"] = r
                if run:''', '''                if looped:
                    m["train/loops"] = r
                if data.unit == "bytes":
                    m["bytes"] = tokens
                    m["token_equiv"] = tokens / BYTES_PER_TOKEN
                if patched and model.last_aux:
                    for k, v in zip(model.last_aux["pass_ids"], model.last_aux["pass_loss"]):
                        m[f"train/loss_pass{k + 1}"] = v.item()
                    if npass is not None:
                        m["train/passes"] = npass
                if run:'''),
('''        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**30''', '''        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0'''),
('''        if looped:
            for r_eval in tc.get("eval_loops", [mc.loops]):
                summary[f"val_loss_r{r_eval}"] = evaluate(model, data, None, tc["batch_size"], device, loops=r_eval)''',
 '''        if looped:
            for r_eval in tc.get("eval_loops", [mc.loops]):
                summary[f"val_loss_r{r_eval}"] = evaluate(model, data, None, tc["batch_size"], device, loops=r_eval)
        if patched:
            summary.update(evaluate_detail(model, data, None, tc["batch_size"], device))
            if mc.refine_shared:
                for p_eval in tc.get("eval_passes", []):
                    summary[f"val_loss_passes{p_eval}"] = evaluate(model, data, None, tc["batch_size"], device,
                                                                   passes=p_eval)
        if data.unit == "bytes":
            summary["bytes"] = tokens
            summary["token_equiv"] = tokens / BYTES_PER_TOKEN'''),
('''        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(DATA / "datasets" / tc["dataset"] / "tokenizer.json"))
        torch.manual_seed(0)
        texts = [sample(model, tok, p, 160, device) for p in PROMPTS]''', '''        if data.unit == "bytes":
            tok = ByteTokenizer(data.meta.get("eos_id", 0))
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(DATA / "datasets" / tc["dataset"] / "tokenizer.json"))
        torch.manual_seed(0)
        n_new = 160 * (4 if data.unit == "bytes" else 1)   # about the same text length in bytes
        texts = [sample(model, tok, p, n_new, device) for p in PROMPTS]'''),

# later edits (CPU GDN hook for the mirror, ROOT, configs outside the repo)
('from experiments.byte_model.model_bytes import BYTES_PER_TOKEN, FusionDance, ModelConfig, make_model  # MIRROR\n',
 'if os.environ.get("GOGETA_CPU_GDN"):  # MIRROR-ONLY\n'
 '    from experiments.byte_model import cpu_ref  # MIRROR-ONLY\n'
 '    cpu_ref.install()  # MIRROR-ONLY\n'
 'from experiments.byte_model.model_bytes import BYTES_PER_TOKEN, FusionDance, ModelConfig, make_model  # MIRROR\n'),
('ROOT = pathlib.Path(__file__).resolve().parent.parent\n', 'ROOT = pathlib.Path(__file__).resolve().parents[2]  # MIRROR\n'),
('''        run_cfg = {"config_file": str(cfg_path.relative_to(ROOT)), "arch": arch, "cli_overrides": overrides,''',
 '''        cfg_name = str(cfg_path.relative_to(ROOT) if cfg_path.is_relative_to(ROOT) else cfg_path)
        run_cfg = {"config_file": cfg_name, "arch": arch, "cli_overrides": overrides,'''),
]


def apply(text: str) -> tuple[str, list[str]]:
    missing = []
    for a, b in EDITS:
        n = text.count(a)
        if n != 1:
            missing.append(f"anchor found {n}x: {a.strip().splitlines()[0][:90]}")
            continue
        text = text.replace(a, b)
    return text, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="file to edit (default: HEAD:gogeta/train.py)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "byte_model" / "train_bytes.py"))
    a = ap.parse_args()
    base = (pathlib.Path(a.base).read_text(encoding="utf-8") if a.base else
            subprocess.run(["git", "show", "HEAD:gogeta/train.py"], cwd=ROOT, capture_output=True, text=True,
                           check=True, encoding="utf-8").stdout)
    out, missing = apply(base)
    if missing:
        raise SystemExit("anchors moved:\n  " + "\n  ".join(missing))
    pathlib.Path(a.out).write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
