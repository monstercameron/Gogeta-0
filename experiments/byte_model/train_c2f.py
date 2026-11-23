"""Train the coarse-to-fine byte model (gogeta/c2f.py), tokens-as-patches (default) or fixed byte patches.

    python -m experiments.byte_model.train_c2f --config experiments/staged-configs/a4/a4a_tok_r2_shared.json \
        --seed 0 --lr 0.006 --max-tokens 200000000 --compile --require-clean
    CUDA_VISIBLE_DEVICES= python -m experiments.byte_model.train_c2f --config <tiny.json> --device cpu   # CPU smoke

Same CLI and run layout as gogeta/train.py (--config, --seed, --lr, --max-tokens, --micro-batch, --compile, --bench,
--require-clean; gogeta.tracking.Run -> $GOGETA_DATA/runs/<name>/{meta,summary}.json, samples.txt, and
$GOGETA_DATA/checkpoints/<name>/model.pt). The config's "arch" is "c2f" and "model" holds gogeta.c2f.C2FConfig
fields, with the global FusionDance config under model.global_model.

Training positions follow the dataset's unit (meta.json "unit"):
  * patching "tokens" on a TOKEN dataset (code-py-v1 / code-py-v2): a position is a BPE token; the budget is
    train.max_tokens (tokens) and --max-tokens counts tokens, exactly as gogeta.train. The decoder spells every target
    token's bytes; val_loss is nats per token on the same val tokens as a BPE model, so val_bits_per_char =
    val_loss / ln 2 / chars_per_token compares directly with E1 / A2s / A3 runs on that dataset;
  * patching "fixed" on a BYTE dataset (code-py-v1-bytes): a position is a byte; the budget is train.max_bytes and
    --max-tokens counts BYTES (sets max_bytes); summaries add bytes and token_equiv = bytes / 3.735.
Frozen trunk (model.freeze_global + model.global_init = a FusionDance checkpoint, e.g. an A2s model.pt): the global
model's config and weights come from the checkpoint (path, sha256 and its dataset are recorded in the run config); it
runs in eval mode at its default loops under no_grad, and only the byte decoder trains (SQ207 / X1).
MoE in a trained global model: LM loss + moe aux["loss"] is backpropagated; train/loss stays the last-pass LM loss.
Optional train keys: final_eval_seqs (default None = full val split, as gogeta.train), sample_bytes (default 640),
eval_passes (extra L(r_b) evals, shared decoder only), eval_loops, ternary_start_frac, print_every.
--device cpu: smoke tests only (no autocast, GDN on the CPU reference recurrence, no fused AdamW).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


@torch.no_grad()
def sample_tokens(model, tok, prompt: str, n_bytes: int, device: str, autocast, temperature=0.8, top_k=40) -> str:
    """Token-patch generation: the global model reads tokens, the decoder spells the next token byte by byte until
    EOT; a spelled byte string that is a vocab token is appended as that id, otherwise the text is re-tokenized."""
    model.eval()
    lookup: dict = {}
    for i, (row, n) in enumerate(zip(model.tok_bytes.tolist(), model.tok_len.tolist())):
        lookup.setdefault(bytes(row[:n]), i)
    ids = tok.encode(prompt).ids
    out = bytearray(prompt.encode("utf-8"))
    made = 0
    while made < n_bytes:
        x = torch.tensor([ids[-model.cfg.max_seq_len:]], device=device)
        prefix: list[int] = []
        while len(prefix) < model.max_token_bytes:
            with autocast(device):
                lg = model.next_byte_logits(x, prefix).float() / temperature
            v, _ = torch.topk(lg, top_k)
            lg[lg < v[-1]] = -float("inf")
            b = int(torch.multinomial(torch.softmax(lg, -1), 1))
            if b == model.EOT:
                break
            prefix.append(b)
        if not prefix:
            break
        bs = bytes(prefix)
        if bs == b"\x00":   # end of document
            break
        out += bs
        made += len(bs)
        if bs in lookup:
            ids.append(lookup[bs])
        else:
            ids = tok.encode(out.decode("utf-8", errors="replace")).ids
    model.train()
    return out.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-tokens", type=int, help="override the budget in dataset positions: tokens (token "
                    "datasets, train.max_tokens) or BYTES (byte datasets, train.max_bytes)")
    ap.add_argument("--max-bytes", type=int, help="alias of --max-tokens for byte datasets")
    ap.add_argument("--seed", type=int, help="override train.seed")
    ap.add_argument("--lr", type=float, help="override train.lr (peak LR; recorded in the run config)")
    ap.add_argument("--micro-batch", type=int, help="override train.micro_batch (gradient accumulation)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: smoke tests only")
    ap.add_argument("--require-clean", action="store_true", help="refuse to run with uncommitted changes")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--bench", type=int, default=0, help="only time N steps, no tracking/eval/ckpt")
    args = ap.parse_args()
    device = args.device
    cuda = device == "cuda"
    if not cuda:   # GDN's fla kernel is GPU-only: CPU smoke runs use the token-by-token reference recurrence
        from experiments.byte_model import cpu_ref
        cpu_ref.install()

    from experiments.byte_model.train_bytes import PROMPTS, ByteTokenizer, TokenData, autocast, sample
    from gogeta.c2f import BYTES_PER_TOKEN, build_c2f, load_global_checkpoint
    from gogeta.model import set_ternary_quant
    from gogeta.tracking import DATA, Run
    from gogeta.train import MOE_STATS, lr_at, micro_batches

    cfg_path = (ROOT / args.config).resolve() if not os.path.isabs(args.config) else pathlib.Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("arch", "c2f") != "c2f":
        raise SystemExit(f"train_c2f trains arch c2f configs only (got {cfg.get('arch')!r})")
    tc = dict(cfg["train"])
    mcfg = {k: v for k, v in cfg["model"].items() if k != "arch"}
    tokens_mode = mcfg.get("patching", "fixed") == "tokens"
    torch.manual_seed(tc.get("seed", 0) if args.seed is None else args.seed)
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.lr is not None:
        tc["lr"] = args.lr
    if args.micro_batch is not None:
        tc["micro_batch"] = args.micro_batch
    mb, n_micro = micro_batches(tc)
    torch.backends.cuda.matmul.allow_tf32 = True

    data = TokenData(tc["dataset"], tc["seq_len"], tc["seed"])
    if tokens_mode and data.unit != "tokens":
        raise SystemExit(f"patching 'tokens' needs a token dataset ({tc['dataset']} counts {data.unit})")
    if not tokens_mode and data.unit != "bytes":
        raise SystemExit(f"patching 'fixed' needs a byte dataset ({tc['dataset']} counts {data.unit})")
    budget_key = "max_tokens" if tokens_mode else "max_bytes"
    if any(k.startswith("max_") and k != budget_key for k in tc):
        raise SystemExit(f"dataset {tc['dataset']} counts {data.unit}: the budget is train.{budget_key}")
    budget = args.max_tokens or args.max_bytes
    if budget:
        tc[budget_key] = budget
    overrides = {k: v for k, v in ((budget_key, budget or None), ("seed", args.seed), ("lr", args.lr),
                                   ("micro_batch", args.micro_batch)) if v is not None}
    if not cuda:
        overrides["device"] = device

    # ---- model (frozen-trunk: the global config + weights come from the checkpoint)
    gl_info, gl_state = None, None
    mcfg["max_seq_len"] = tc["seq_len"]
    if tokens_mode:
        mcfg.setdefault("global_model", {})
        mcfg["global_model"] = {**mcfg["global_model"], "vocab_size": data.meta["vocab_size"]}
        if mcfg.get("global_init"):
            gl_state, ck_cfg, gl_info = load_global_checkpoint(mcfg["global_init"])
            given = {k: v for k, v in mcfg["global_model"].items() if k not in ("vocab_size", "max_seq_len")}
            ck = {k: v for k, v in ck_cfg.items() if k != "max_seq_len"}
            clash = {k: (v, ck.get(k)) for k, v in given.items() if k in ck and ck[k] != v}
            if clash:
                raise SystemExit(f"model.global_model disagrees with the checkpoint's model_config: {clash}")
            mcfg["global_model"] = ck
            if gl_info["dataset"] != tc["dataset"]:
                print(f"WARNING: the global checkpoint was trained on {gl_info['dataset']!r}, this run uses "
                      f"{tc['dataset']!r} (same tokenizer required; the val numbers are not that trunk's own val)",
                      flush=True)
            print(f"global init: {gl_info['path']} (sha256 {gl_info['sha256'][:16]}, {gl_info['n_matches']} match(es))",
                  flush=True)
    model = build_c2f(mcfg)
    if tokens_mode:   # the spelling table must be the dataset's own tokenizer
        from gogeta.c2f import token_byte_table
        ds_tab = token_byte_table(str(DATA / "datasets" / tc["dataset"] / "tokenizer.json"))
        if ds_tab != [bytes(r[:n]) for r, n in zip(model.tok_bytes.tolist(), model.tok_len.tolist())]:
            raise SystemExit(f"model.tokenizer {mc.tokenizer} differs from {tc['dataset']}'s tokenizer.json")
    if gl_state is not None:
        model.load_global(gl_state)
    model = model.to(device)
    mc = model.cfg
    fwd_model = torch.compile(model) if args.compile else model
    gc = model.gcfg
    pc = model.param_counts()
    frozen = mc.freeze_global
    looped = len(model.core) > 0 and not frozen
    moe = bool(gc.moe_experts) and not frozen
    loop_rng = random.Random(tc["seed"] + 1)
    T = tc["seq_len"]

    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]), fused=cuda)

    pos_per_step = tc["batch_size"] * T
    total_steps = args.bench or math.ceil(tc[budget_key] / pos_per_step)
    unit = "tok" if tokens_mode else "bytes"
    fb = model.flops_breakdown(T)
    print(f"params: total {pc['total']/1e6:.2f}M global {pc['global']/1e6:.2f}M local {pc['local']/1e6:.2f}M "
          f"trainable {sum(p.numel() for p in params)/1e6:.2f}M | steps {total_steps} x {pos_per_step} {unit} | "
          f"patching={mc.patching} r_b={mc.refine_passes} shared={mc.refine_shared} frozen_global={frozen} | "
          f"global looped={len(model.core) > 0} ternary={gc.ternary} moe={gc.moe_experts}", flush=True)
    print(f"train FLOPs/{'token' if tokens_mode else 'byte'}: " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in fb.items()),
          flush=True)

    run = None
    if not args.bench:
        cfg_name = str(cfg_path.relative_to(ROOT) if cfg_path.is_relative_to(ROOT) else cfg_path)
        run_cfg = {"config_file": cfg_name, "arch": "c2f", "cli_overrides": overrides, "model": mc.to_dict(),
                   "train": tc, "params": pc, "flops_per_token_at_default_loops": model.flops_per_token(T),
                   "flops_breakdown": fb, "flops_unit": "per token" if tokens_mode else "per byte",
                   "compile": args.compile,
                   "dataset_meta": {k: data.meta.get(k) for k in ("name", "source", "vocab_size", "unit")},
                   "position_unit": data.unit, "trainer": "experiments/byte_model/train_c2f.py",
                   "global_init": {k: v for k, v in (gl_info or {}).items() if k != "train"} or None}
        run = Run(cfg["experiment"], config=run_cfg, seed=tc["seed"], require_clean=args.require_clean)
        print(f"run: {run.name}", flush=True)

    @torch.no_grad()
    def evaluate(n_seqs, loops=None, passes=None, detail=False) -> dict:
        model.eval()
        acc: dict = {}
        cnt = 0.0
        for x, y in data.val_batches(n_seqs, tc["batch_size"], device):
            with autocast(device):
                _, loss = model(x, y, loops=loops, passes=passes, return_all=detail)
            n = float(x.numel())
            la = model.last_aux
            vals = {"loss": loss.float().cpu() * n}
            if detail:
                vals["pass"] = torch.stack([v.float().cpu() for v in la["pass_loss"]]) * n
                vals["slot"] = la["slot_loss"].float().cpu() * n
                if tokens_mode:
                    vals["row"] = la["nats_per_row"].float().cpu() * la["rows"].float().cpu()
                    vals["rows"] = la["rows"].float().cpu()
                    vals["eot"] = la["eot_nats_per_token"].float().cpu() * n
            for k, v in vals.items():
                acc[k] = acc.get(k, 0) + v
            cnt += n
        model.train()
        out = {"loss": acc["loss"].item() / cnt}
        if detail:
            out.update({f"val_loss_pass{k + 1}": v for k, v in enumerate((acc["pass"] / cnt).tolist())})
            out.update({f"val_loss_slot{j}": v for j, v in enumerate((acc["slot"] / cnt).tolist()[:8])})
            if tokens_mode:
                out["val_nats_per_row"] = (acc["row"] / acc["rows"]).item()
                out["val_rows_per_token"] = (acc["rows"] / cnt).item()
                out["val_eot_nats_per_token"] = (acc["eot"] / cnt).item()
        return out

    t_start = time.time()
    pos = flops = 0.0
    t_log, pos_log = time.time(), 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(total_steps):
            lr = lr_at(step, total_steps, tc)
            for gp in opt.param_groups:
                gp["lr"] = lr
            r = loop_rng.randint(gc.loops_min, gc.loops_max) if looped else None
            if tc.get("ternary_start_frac"):
                if set_ternary_quant(model, step >= tc["ternary_start_frac"] * total_steps):
                    print(f"step {step}: ternary quantisation {'on' if step else 'off'}", flush=True)
            x, y = data.batch(tc["batch_size"], device)
            lm = obj = 0.0
            acc = {k: 0.0 for k in MOE_STATS} if moe else None
            pass_acc = None
            for xs, ys in (zip(x.chunk(n_micro), y.chunk(n_micro)) if n_micro > 1 else [(x, y)]):
                with autocast(device):
                    _, loss, aux = fwd_model(xs, ys, loops=r, return_aux=True)
                total = loss + aux["loss"] if (moe and aux is not None) else loss
                (total / n_micro).backward()
                la = model.last_aux
                lm = lm + la["ce_last"] / n_micro
                obj = obj + total.detach() / n_micro
                pl = torch.stack(la["pass_loss"]) / n_micro
                pass_acc = pl if pass_acc is None else pass_acc + pl
                if moe:
                    for k in MOE_STATS:
                        acc[k] = acc[k] + (aux["loss"] if k == "aux" else aux[k]).detach().float() / n_micro
            gnorm = torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)

            pos += pos_per_step
            flops += model.flops_per_token(T, loops=r) * pos_per_step
            pos_log += pos_per_step
            if step % 10 == 0 or step == total_steps - 1:
                if cuda:
                    torch.cuda.synchronize()
                dt = time.time() - t_log
                tps = pos_log / dt if dt > 0 else 0.0
                t_log, pos_log = time.time(), 0
                m = {"train/loss": lm.item(), "train/loss_total": obj.item(), "train/lr": lr,
                     "train/grad_norm": gnorm.item(), "tokens": pos, "flops": flops, "perf/tokens_per_s": tps}
                if not tokens_mode:
                    m.update({"bytes": pos, "token_equiv": pos / BYTES_PER_TOKEN})
                else:
                    m["train/nats_per_row"] = model.last_aux["nats_per_row"].item()
                for k, v in zip(model.last_aux["pass_ids"], pass_acc.tolist()):
                    m[f"train/loss_pass{k + 1}"] = v
                if looped:
                    m["train/loops"] = r
                if moe:
                    m.update({f"moe/{k}": v.item() for k, v in acc.items()})
                if run:
                    run.log(m, step=step)
                if step % tc.get("print_every", 100) == 0 or step == total_steps - 1:
                    print(f"step {step:6d}/{total_steps} loss {m['train/loss']:.4f} lr {lr:.2e} "
                          f"gn {m['train/grad_norm']:.2f} {tps/1e3:.1f}k {unit}/s r={r}"
                          + "".join(f" p{k + 1} {m[f'train/loss_pass{k + 1}']:.4f}" for k in model.last_aux["pass_ids"])
                          + (f" moe_aux {m['moe/aux']:.4f} max_load {m['moe/max_load']:.3f}" if moe else ""),
                          flush=True)
            if run and (step % tc["eval_every"] == 0 or step == total_steps - 1) and step > 0:
                vl = evaluate(tc["eval_seqs"])["loss"]
                run.log({"val/loss": vl, "tokens": pos, "flops": flops}, step=step)
                print(f"  eval step {step}: val/loss {vl:.4f}", flush=True)

        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        if args.bench:
            print(f"BENCH {cfg['experiment']}: {total_steps} steps, {pos/wall/1e3:.1f}k {unit}/s overall "
                  f"(incl. warmup), peak mem {peak:.2f} GB", flush=True)
            return

        # ---- final evaluation (full validation split unless train.final_eval_seqs)
        n_final = tc.get("final_eval_seqs")
        det = evaluate(n_final, detail=True)
        summary = {"val_loss": det.pop("loss")}
        summary["val_bits_per_char"] = summary["val_loss"] / math.log(2) / data.chars_per_token
        summary.update(det)
        if looped:
            for r_eval in tc.get("eval_loops", [gc.loops]):
                summary[f"val_loss_r{r_eval}"] = evaluate(n_final, loops=r_eval)["loss"]
        if mc.refine_shared:
            for p_eval in tc.get("eval_passes", []):
                summary[f"val_loss_passes{p_eval}"] = evaluate(n_final, passes=p_eval)["loss"]
        if not tokens_mode:
            summary.update({"bytes": pos, "token_equiv": pos / BYTES_PER_TOKEN})
        summary.update({
            "tokens": pos, "flops": flops, "steps": total_steps, "wall_min": wall / 60,
            "tokens_per_s": pos / wall, "peak_mem_gb": peak,
            "params_total": pc["total"], "params_non_embedding": pc["non_embedding"],
            "params_global": pc["global"], "params_local": pc["local"],
            "params_trainable": sum(p.numel() for p in params),
            "flops_per_token_default": model.flops_per_token(T),
            "final_eval_seqs": n_final, "position_unit": data.unit,
        })
        run.summary.update(summary)
        run.log({f"final/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}, step=total_steps)
        print("FINAL " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}),
              flush=True)

        # ---- samples + checkpoint
        torch.manual_seed(0)
        n_new = tc.get("sample_bytes", 640)
        if tokens_mode:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(DATA / "datasets" / tc["dataset"] / "tokenizer.json"))
            texts = [sample_tokens(model, tok, p, n_new, device, autocast) for p in PROMPTS]
        else:
            tok = ByteTokenizer(data.meta.get("eos_id", 0))
            texts = [sample(model, tok, p, n_new, device) for p in PROMPTS]
        (run.dir / "samples.txt").write_text("\n\n" + "\n\n=====\n\n".join(texts) + "\n", encoding="utf-8")
        ck = DATA / "checkpoints" / run.name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "model_config": mc.to_dict(), "train": tc,
                    "global_init": run_cfg["global_init"]}, ck / "model.pt")
        run.summary["checkpoint"] = str(ck / "model.pt")
        run.finish("done")
        print(f"done: {run.dir}", flush=True)
    except BaseException as e:
        if run and run._t is not None:
            run.finish("failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
