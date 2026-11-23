"""Self-speculative decoding bench on a real A2s checkpoint (CPU, fp32, torch GDN recurrence).

    python experiments/checks/spec_decode_bench.py [--ckpt <model.pt>] [--n-prompts 20] [--prompt-len 128]
                                                   [--n-new 64] [--workers 5] [--threads 4]

Protocol (fixed before the run):
  * checkpoint: the first A2s model.pt (sorted) under ~/fusiondance-data/cloud/fleet/gogeta-cockpit-a2-*/; its config
    from the checkpoint's model_config, cross-checked against runs/<run>/meta.json;
  * prompts: the first n docs of val_clean50 (experiments/data_integrity/clean-val.manifest.json) whose tokenisation
    has >= prompt_len tokens; prompt = <|eos|> + the doc's first prompt_len - 1 tokens;
  * reference = plain cached greedy at r_full = 2 (A2s was trained at a FIXED r = 2), n_new = 64 tokens;
  * speculative: r_draft in {1, 3}, k in {2, 4, 8}; output must equal the reference on every prompt (lossless).
    r_draft = 1 is an untrained operating point for this model; r_draft = 3 is costlier than the target (reported
    only as an agreement measurement, not as a useful draft);
  * r-agreement: teacher-forced per-position argmax agreement of r = 1 and r = 3 with r = 2 on the r = 2 greedy
    continuation, and the free-running matched-prefix length of greedy r = 1 / 3 vs r = 2;
  * numerics, per generated step: the cached path's logits vs ONE full forward of the model itself
    (FusionDance.forward, torch reference GDN kernel) over prompt + reference tokens; the full model's top-1 minus
    top-2 margin; and the full forward's OWN noise floor (the same forward at 1 thread vs --threads threads).
    Steps whose margin is below these differences are the steps where greedy could flip under numerics.
  * path exactness without activation rounding: the same cached-vs-full comparison with every BitLinear's int8
    activation quantisation switched off on the loaded instance (quant_acts=False; diagnostic only).

PASS criteria. Original (written before the 1-prompt smoke run): lossless everywhere AND max|cached - full| < 1e-3.
The smoke run gave max|cached - full| = 0.087 with lossless output; a diagnostic then showed the full forward itself
moves by 0.08-0.16 between 8 and 1 CPU threads (int8 activation rounding amplifies fp32 reordering noise) and the
cached path matches to 1.2e-5 with activation quantisation off. Revised criteria (fixed before this full run):
  (1) lossless on every (prompt, r_draft, k);
  (2) activation quant off: max|cached - full| < 1e-4 (the decode path computes the same function);
  (3) activation quant on: max|cached - full| <= max over prompts of the full forward's 1-vs-N-thread difference.
The original criterion is still evaluated and reported. Acceptance rates are measured (no prediction registered).
CPU wall-clock seconds are informational only (GPU timing is out of scope).
Output: experiments/checks/results/spec_decode_bench.json
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import pathlib
import platform
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gogeta.decode import _FLA_FAILED, prefill, speculative_generate, step, use_reference_gdn  # noqa: E402
from gogeta.model import BitLinear, FusionDance, ModelConfig  # noqa: E402
from gogeta.moe import MoEFFN  # noqa: E402

DATA = pathlib.Path(os.path.expanduser("~/fusiondance-data"))
DATASET = DATA / "datasets" / "code-py-v1"
MANIFEST = ROOT / "experiments" / "data_integrity" / "clean-val.manifest.json"
OUT = ROOT / "experiments" / "checks" / "results" / "spec_decode_bench.json"
CKPT_GLOB = str(DATA / "cloud" / "fleet" / "gogeta-cockpit-a2-*" / "fusiondance-data" / "checkpoints" / "A2s-*" / "model.pt")
R_FULL, R_DRAFTS, KS = 2, [1, 3], [2, 4, 8]


def load(ckpt: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_config"])
    run = pathlib.Path(ckpt).parent.name
    meta_p = pathlib.Path(ckpt).parents[2] / "runs" / run / "meta.json"
    meta_match = None
    if meta_p.exists():
        meta_model = json.loads(meta_p.read_text())["config"]["model"]
        meta_match = all(mcfg.get(k) == v for k, v in meta_model.items())
    m = FusionDance(ModelConfig(**mcfg))
    m.load_state_dict(ck["model"])
    m.eval()
    return use_reference_gdn(m), mcfg, run, (str(meta_p) if meta_p.exists() else None), meta_match


def prompts(n: int, plen: int, min_total: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(DATASET / "tokenizer.json"))
    eos = tok.token_to_id("<|eos|>")
    ids = json.loads(MANIFEST.read_text())["lists"]["val_clean50"]["ids"]
    lines = (DATASET / "docs_val.jsonl").read_text(encoding="utf-8").splitlines()
    out = []
    for i in ids:
        t = [eos] + tok.encode(json.loads(lines[i])["text"]).ids
        if len(t) >= min_total:
            out.append((i, t[:plen]))
        if len(out) == n:
            break
    return out


@torch.no_grad()
def greedy_with_logits(m, idx, n_new, r):
    lp, cache = prefill(m, idx, r, max_rollback=0)
    logits = [lp[0]]
    out = [int(lp[0].argmax())]
    while len(out) < n_new:
        lg = step(m, cache, torch.tensor([[out[-1]]]))[0, -1]
        logits.append(lg)
        out.append(int(lg.argmax()))
    return out, torch.stack(logits)


@torch.no_grad()
def teacher_forced_cached(m, ptoks, cont, r):
    """Cached logits at the positions predicting cont[0..n-1] (prefill prompt, then 1-token steps over cont[:-1])."""
    lp, cache = prefill(m, torch.tensor([ptoks]), r, max_rollback=0)
    rows = [lp[0]] + [step(m, cache, torch.tensor([[t]]))[0, -1] for t in cont[:-1]]
    return torch.stack(rows)


def set_act_quant(m, on: bool):
    for mod in m.modules():
        if isinstance(mod, (BitLinear, MoEFFN)):
            mod.quant_acts = on


def matched_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


_W = {}


def _init(ckpt, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    _W["m"] = load(ckpt)[0]
    _W["threads"] = threads


def run_prompt(args):
    doc_id, ptoks, n_new = args
    m, threads = _W["m"], _W["threads"]
    idx = torch.tensor([ptoks])
    t0 = time.perf_counter()
    ref, cl = greedy_with_logits(m, idx, n_new, R_FULL)
    t_greedy = time.perf_counter() - t0
    seq = torch.tensor([ptoks + ref[:-1]])
    sl = slice(len(ptoks) - 1, None)               # positions whose logits predict the generated tokens
    with torch.no_grad():
        fl = {r: m(seq, loops=r)[0][0, sl] for r in (1, 2, 3)}
        torch.set_num_threads(1)
        fl_1t = m(seq, loops=R_FULL)[0][0, sl]
        torch.set_num_threads(threads)
    diff = (cl - fl[R_FULL]).abs().amax(-1)        # per step max |cached - full|
    floor = (fl_1t - fl[R_FULL]).abs().amax(-1)    # per step: the full forward's own 1-vs-N-thread difference
    top2 = fl[R_FULL].topk(2, -1).values
    margin = top2[:, 0] - top2[:, 1]
    set_act_quant(m, False)
    try:
        cl_q = teacher_forced_cached(m, ptoks, ref, R_FULL)
        with torch.no_grad():
            fl_q = m(seq, loops=R_FULL)[0][0, sl]
        diff_noact = (cl_q - fl_q).abs().max().item()
    finally:
        set_act_quant(m, True)
    row = {"doc_id": doc_id, "prompt_tokens": len(ptoks), "greedy_r2_cpu_seconds": t_greedy,
           "max_abs_logit_diff_cached_vs_full": diff.max().item(),
           "max_abs_logit_diff_full_1_vs_n_threads": floor.max().item(),
           "max_abs_logit_diff_cached_vs_full_act_quant_off": diff_noact,
           "min_margin": margin.min().item(),
           "steps_margin_below_cached_diff": int((margin < diff).sum()),
           "steps_margin_below_thread_floor": int((margin < floor).sum()),
           "steps_margin_below_0.2": int((margin < 0.2).sum()),
           "full_forward_argmax_equals_cached_greedy": fl[R_FULL].argmax(-1).tolist() == ref,
           "teacher_forced_agreement": {f"r{r}_vs_r2": (fl[r].argmax(-1) == fl[R_FULL].argmax(-1)).float().mean().item()
                                        for r in (1, 3)},
           "spec": {}}
    for r in (1, 3):
        g, _ = greedy_with_logits(m, idx, n_new, r)
        row[f"free_running_matched_prefix_r{r}_vs_r2"] = matched_prefix(g, ref)
    for rd in R_DRAFTS:
        for k in KS:
            out, st = speculative_generate(m, idx, n_new, rd, R_FULL, k)
            st["lossless"] = out == ref
            if not st["lossless"]:
                st["first_divergence"] = matched_prefix(out, ref)
            row["spec"][f"rd{rd}_k{k}"] = st
    row["fla_failures"] = list(_FLA_FAILED)
    return row


def run_diag(args):
    """Lossy prompt: margin and cached-vs-full gap at each divergence step (quant on), then the full speculative
    sweep with int8 activation quantisation OFF (decides whether the flips come from activation rounding)."""
    doc_id, ptoks, n_new, divs = args
    m = _W["m"]
    idx = torch.tensor([ptoks])
    ref, cl = greedy_with_logits(m, idx, n_new, R_FULL)
    with torch.no_grad():
        fl = m(torch.tensor([ptoks + ref[:-1]]), loops=R_FULL)[0][0, len(ptoks) - 1:]
    top2 = fl.topk(2, -1).values
    at = {}
    for cfg, j in divs.items():
        at[cfg] = {"step": j, "full_margin": (top2[j, 0] - top2[j, 1]).item(),
                   "cached_vs_full_diff": (cl[j] - fl[j]).abs().max().item()}
    set_act_quant(m, False)
    try:
        ref_q, _ = greedy_with_logits(m, idx, n_new, R_FULL)
        off = {}
        for rd in R_DRAFTS:
            for k in KS:
                out, st = speculative_generate(m, idx, n_new, rd, R_FULL, k)
                off[f"rd{rd}_k{k}"] = {"lossless": out == ref_q, "acceptance_rate": st["acceptance_rate"]}
    finally:
        set_act_quant(m, True)
    return {"doc_id": doc_id, "divergence_steps_act_quant_on": at, "act_quant_off_spec": off}


def diag(a):
    res = json.loads(pathlib.Path(a.out).read_text())
    P = dict(prompts(res["protocol"]["n_prompts"], res["protocol"]["prompt_len"], res["protocol"]["prompt_len"]))
    jobs = []
    for p in res["per_prompt"]:
        divs = {k: s["first_divergence"] for k, s in p["spec"].items() if not s["lossless"]}
        if divs:
            jobs.append((p["doc_id"], P[p["doc_id"]], res["protocol"]["n_new"], divs))
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(max(1, min(a.workers, len(jobs))), initializer=_init,
                                      initargs=(res["checkpoint"], a.threads)) as pool:
        rows = pool.map(run_diag, jobs)
    for r in rows:
        print(json.dumps(r))
    res["diagnostic_lossy_prompts"] = {
        "what": "for every prompt with a lossy (prompt, r_draft, k): the full model's top-1 minus top-2 margin and the "
                "max|cached - full| logit gap at the first divergence step (act quant on), then greedy + speculative "
                "with every BitLinear's int8 activation quantisation OFF on the loaded instance",
        "rows": rows,
        "all_lossless_with_act_quant_off": all(v["lossless"] for r in rows for v in r["act_quant_off_spec"].values()),
        "wall_seconds": time.perf_counter() - t0}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("all lossless with act quant off:", res["diagnostic_lossy_prompts"]["all_lossless_with_act_quant_off"])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-lossy", action="store_true",
                    help="after a run: diagnose its lossy prompts and add the result to the same JSON")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.diag_lossy:
        return diag(a)
    ckpt = a.ckpt or sorted(glob.glob(CKPT_GLOB))[0]
    m, mcfg, run, meta_p, meta_match = load(ckpt)
    n_params = sum(p.numel() for p in m.parameters())
    del m
    P = prompts(a.n_prompts, a.prompt_len, a.prompt_len)
    print(f"ckpt {ckpt}\nparams {n_params:,} (computed)  prompts {len(P)}  workers {a.workers} x {a.threads} threads",
          flush=True)
    t_all = time.perf_counter()
    ctx = mp.get_context("spawn")
    per = []
    with ctx.Pool(a.workers, initializer=_init, initargs=(ckpt, a.threads)) as pool:
        for row in pool.imap(run_prompt, [(d, t, a.n_new) for d, t in P]):
            per.append(row)
            print(f"[{len(per):2d}/{len(P)}] doc {row['doc_id']:4d} cached-full {row['max_abs_logit_diff_cached_vs_full']:.1e}"
                  f" floor {row['max_abs_logit_diff_full_1_vs_n_threads']:.1e} noact {row['max_abs_logit_diff_cached_vs_full_act_quant_off']:.1e}"
                  f" minmargin {row['min_margin']:.3f} tf r1 {row['teacher_forced_agreement']['r1_vs_r2']:.2f} | " +
                  " ".join(f"{key}:{v['acceptance_rate']:.2f}{'' if v['lossless'] else '!LOSSY'}"
                           for key, v in row["spec"].items()), flush=True)

    agg = {}
    for rd in R_DRAFTS:
        for k in KS:
            key = f"rd{rd}_k{k}"
            sts = [p["spec"][key] for p in per]
            prop, acc = sum(s["proposed"] for s in sts), sum(s["accepted"] for s in sts)
            vc = sum(s["verify_calls"] for s in sts)
            gen = sum(s["n_new"] for s in sts)
            agg[key] = {"r_draft": rd, "r_full": R_FULL, "k": k,
                        "lossless_prompts": sum(s["lossless"] for s in sts), "n_prompts": len(sts),
                        "acceptance_rate_pooled": acc / prop,
                        "acceptance_rate_mean_over_prompts": sum(s["acceptance_rate"] for s in sts) / len(sts),
                        "tokens_per_verify_call": (gen - len(sts)) / vc,
                        "full_calls_per_token_incl_prefill": (vc + len(sts)) / gen,
                        "draft_calls_per_token": sum(s["draft_calls"] for s in sts) / gen,
                        "cpu_seconds_total": sum(s["seconds"] for s in sts)}
    n_steps = len(per) * a.n_new
    mx = lambda key: max(p[key] for p in per)  # noqa: E731
    sm = lambda key: sum(p[key] for p in per)  # noqa: E731
    numer = {"max_abs_logit_diff_cached_vs_full": mx("max_abs_logit_diff_cached_vs_full"),
             "max_abs_logit_diff_full_1_vs_n_threads": mx("max_abs_logit_diff_full_1_vs_n_threads"),
             "max_abs_logit_diff_cached_vs_full_act_quant_off": mx("max_abs_logit_diff_cached_vs_full_act_quant_off"),
             "min_margin": min(p["min_margin"] for p in per),
             "steps_margin_below_cached_diff": sm("steps_margin_below_cached_diff"),
             "steps_margin_below_thread_floor": sm("steps_margin_below_thread_floor"),
             "steps_margin_below_0.2": sm("steps_margin_below_0.2"),
             "n_steps": n_steps,
             "full_forward_argmax_equals_cached_greedy_prompts": sm("full_forward_argmax_equals_cached_greedy")}
    agree = {f"r{r}_vs_r2_teacher_forced_mean": sum(p["teacher_forced_agreement"][f"r{r}_vs_r2"] for p in per) / len(per)
             for r in (1, 3)}
    agree.update({f"r{r}_vs_r2_free_running_matched_prefix_mean":
                  sum(p[f"free_running_matched_prefix_r{r}_vs_r2"] for p in per) / len(per) for r in (1, 3)})
    c1 = all(v["lossless_prompts"] == v["n_prompts"] for v in agg.values())
    c2 = numer["max_abs_logit_diff_cached_vs_full_act_quant_off"] < 1e-4
    c3 = numer["max_abs_logit_diff_cached_vs_full"] <= numer["max_abs_logit_diff_full_1_vs_n_threads"]
    orig = c1 and numer["max_abs_logit_diff_cached_vs_full"] < 1e-3
    res = {"check": "spec_decode_bench", "checkpoint": ckpt, "run": run, "run_meta": meta_p,
           "meta_config_matches_checkpoint": meta_match, "model_config": mcfg, "params_computed": n_params,
           "kernel_path": "CPU, fp32, torch gated-delta recurrence (gogeta/decode.py); reference full forward uses "
                          "decode.gated_delta_rule_ref. bf16 / fla fused_recurrent on GPU NOT checked.",
           "device": "cpu", "torch": torch.__version__, "workers": a.workers, "threads_per_worker": a.threads,
           "platform": platform.platform(),
           "protocol": {"prompts": "first val_clean50 docs with >= prompt_len tokens; <|eos|> + doc prefix",
                        "n_prompts": len(P), "prompt_len": a.prompt_len, "n_new": a.n_new, "r_full": R_FULL,
                        "r_drafts": R_DRAFTS, "ks": KS, "decoding": "greedy"},
           "criteria": {"original_lossless_and_cached_diff_lt_1e-3": orig,
                        "revised_1_lossless_all": c1, "revised_2_act_quant_off_diff_lt_1e-4": c2,
                        "revised_3_cached_diff_le_thread_noise_floor": c3},
           "verdict": "PASS" if (c1 and c2 and c3) else "FAIL",
           "verdict_original_criterion": "PASS" if orig else "FAIL",
           "aggregate": agg, "numerics": numer, "r_agreement": agree,
           "fla_failures": sorted({f for p in per for f in p.pop("fla_failures")}),
           "wall_seconds_total": time.perf_counter() - t_all, "per_prompt": per}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"criteria": res["criteria"], "aggregate": agg, "numerics": numer, "r_agreement": agree}, indent=1))
    print(f"verdict (revised) {res['verdict']}; original criterion {res['verdict_original_criterion']}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
