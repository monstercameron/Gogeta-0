"""SQ374 (descriptive, not registered): clean80 bits/char of the existing byte-model runs E13a-d (20M bytes), E13a/E13b (200M bytes) and the
SQ188 LR grid of E13b (20M bytes: lr 1.5e-3, 3e-3, 6e-3, 1.2e-2), scored with a byte scorer on CPU, compared with the token models'
clean80 at matched training FLOPs, plus the loss by position within the patch (the "patch lag").

Runs (fleet copy gogeta-cockpit-w4-1, all seed 0): runs/E13*/meta.json + checkpoints/E13*/model.pt, found by glob.
Scorer: the same K8 protocol as the token models, in bytes: doc ids = [EOS(0) as BOS] + utf-8 bytes of the doc text; window 4,096 bytes
(the training sequence length) with stride 2,048 for the few docs that are longer; targets = the doc's own bytes (no EOS target); bits/char =
total NLL in bits / number of characters of the docs (bytes per char = 1.0005), exactly as for the token models, so numbers are comparable.
Docs: the 2,333 clean80 docs (K9 overlap < 0.8; the byte val set is built from the same docs_val.jsonl in the same order, asserted by length).
Byte models run in the byte_model module (experiments/byte_model/model_bytes.py); the GDN kernel is replaced by the fp32 chunkwise torch
kernel of sq339_cpu.py (chunk size min(64, T); validated against the per-token reference there).
Patch lag: NLL of the target byte at position u of the window, grouped by slot = u mod P (windows start on patch boundaries, so slot 0 is the
first byte of a patch, predicted from the previous patches only; slot P-1 sees P-1 bytes of its own patch through the local decoder).
Matched FLOPs: train FLOPs of a run = FLOPs/unit (flops_per_token_at_default_loops in meta.json) x its budget (bytes or tokens), from the run's
own meta; the token runs are every fleet score file (experiments/stats/doclevel/*.json) whose run meta is on this machine. NOTE the byte
and token FLOPs both come from the repo's own accounting (model.flops_per_token), and a 200M-byte run sees 53.5M token-equivalents of text.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq374_byte_scoring.py -> results/sq374.json
"""
import glob
import json
import math
import os
import sys
import time

import torch

import sq339_cpu as C

sys.path.insert(0, str(C.ROOT))
from experiments.byte_model import cpu_ref  # noqa: E402

WIN, STRIDE, BUDGET = 4096, 2048, 16384
W4 = os.path.join(C.FLEET, "gogeta-cockpit-w4-1", "fusiondance-data")


def install_kernel():
    cpu_ref.install(force=True)

    def kern(q, k, v, g, beta, scale=None, **kw):
        assert not kw, kw
        return C.gdn_chunk_cpu(q, k, v, g, beta, scale, C=min(64, q.shape[1]))

    sys.modules["fla.ops.gated_delta_rule"].chunk_gated_delta_rule = kern


def load_byte_model(path):
    from experiments.byte_model.model_bytes import ModelConfig, make_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**dict(ck["model_config"]))
    m = make_model(cfg)
    m.load_state_dict(ck["model"])
    return m.float().eval(), cfg


def byte_docs():
    ov = json.load(open(C.ROOT / "experiments" / "checks" / "results" / "K9.json"))["data"]["val_per_doc_overlap"]
    texts = [json.loads(l)["text"] for l in open(os.path.join(C.DATASET_DIR, "docs_val.jsonl"), encoding="utf-8")]
    assert len(texts) == len(ov)
    return [texts[i] for i, o in enumerate(ov) if o is None or o < 0.8]


@torch.no_grad()
def score_bytes(model, P, texts, log=None):
    chunks, nbytes = [], []
    for di, t in enumerate(texts):
        ids = [0] + list(t.encode("utf-8"))
        nbytes.append(len(ids) - 1)
        for c, skip in C.chunk_doc(ids, WIN):   # K8 windowing: stride WIN // 2 = 2048
            chunks.append((di, c, skip))
    chunks.sort(key=lambda c: len(c[1]))
    nll = [0.0] * len(texts)
    slot_sum, slot_n = torch.zeros(P, dtype=torch.float64), torch.zeros(P, dtype=torch.float64)
    i, done = 0, 0
    while i < len(chunks):
        T = len(chunks[i][1])
        j = i
        while j < len(chunks) and (j - i + 1) * len(chunks[j][1]) <= max(BUDGET, T):
            j += 1
        batch = chunks[i:j]
        T = max(len(c[1]) for c in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        mask = torch.zeros(len(batch), T - 1, dtype=torch.bool)
        for b, (_, ids, skip) in enumerate(batch):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, skip:len(ids) - 1] = True
        logits = model(x)[0]
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        l = -lp.gather(-1, x[:, 1:, None]).squeeze(-1) * mask
        ls = l.sum(-1).double()
        slot = (torch.arange(T - 1) + 1) % P
        for b, (di, _, _) in enumerate(batch):
            nll[di] += float(ls[b])
        slot_sum.index_add_(0, slot, (l.double()).sum(0))
        slot_n.index_add_(0, slot, mask.double().sum(0))
        i = j
        done += len(batch)
        if log and done % 200 < len(batch):
            log(f"  chunks {done}/{len(chunks)}")
    chars = sum(len(t) for t in texts)
    return {"bits_per_char": sum(nll) / math.log(2) / chars, "bytes": sum(nbytes), "chars": chars, "nll_doc": nll,
            "slot_nats_per_byte": (slot_sum / slot_n).tolist(), "slot_n": slot_n.tolist()}


def byte_runs():
    out = []
    for d in sorted(glob.glob(W4 + "/runs/E13*")):
        m = json.load(open(d + "/meta.json"))
        c = m["config"]
        run = os.path.basename(d)
        sm = json.load(open(d + "/summary.json")).get("metrics", {})
        out.append({"run": run, "id": c["config_file"].split("/")[-1].replace(".json", ""), "exp": m["experiment"], "lr": c["train"]["lr"],
                    "seed": c["train"]["seed"], "P": c["model"]["patch_size"], "max_bytes": c["train"]["max_bytes"],
                    "flops_per_byte": c["flops_per_token_at_default_loops"], "ckpt": f"{W4}/checkpoints/{run}/model.pt",
                    "train_val_bits_per_char_all_val": sm.get("val_bits_per_char"), "train_val_slot_loss": {k: v for k, v in sm.items() if "slot" in k}})
    return out


def token_runs():
    rows = []
    for f in glob.glob(f"{C.FLEET}/*/Gogeta-0/experiments/stats/doclevel/*.json"):
        d = json.load(open(f))
        run = d["run"]
        mp = glob.glob(f"{C.FLEET}/*/fusiondance-data/runs/{run}/meta.json")
        if not mp or "clean80" not in d.get("bpc", {}):
            continue
        c = json.load(open(mp[0]))["config"]
        t = c["train"]
        budget = t.get("max_tokens")
        if not budget or "flops_per_token_at_default_loops" not in c:
            continue
        rows.append({"run": run, "exp": d.get("experiment"), "seed": d.get("seed"), "lr": t["lr"], "tokens": budget,
                     "flops_per_token": c["flops_per_token_at_default_loops"], "train_flops": c["flops_per_token_at_default_loops"] * budget,
                     "params": c.get("params", {}).get("total"), "clean80_bpc": d["bpc"]["clean80"]})
    return sorted({r["run"]: r for r in rows}.values(), key=lambda r: r["train_flops"])


def main():
    log = C.make_log(str(C.RES / "sq374.log"))
    install_kernel()
    texts = byte_docs()
    log(f"clean80 docs {len(texts)}; bytes {sum(len(t.encode('utf-8')) for t in texts)}")
    runs = byte_runs()
    res = {"check": "SQ374", "git": C.git_commit(), "status": "descriptive, not registered", "n_docs": len(texts),
           "protocol": "K8 windowing in bytes, window 4096 stride 2048, fp32 CPU, EOS as BOS, clean80 of v1 val", "byte_runs": [], "token_runs": token_runs()}
    log(f"byte runs: {[r['run'] for r in runs]}; token runs with FLOPs: {len(res['token_runs'])}")
    out_path = C.RES / "sq374.json"
    for r in runs:
        t = time.time()
        model, cfg = load_byte_model(r["ckpt"])
        sc = score_bytes(model, cfg.patch_size, texts, log)
        r = {k: v for k, v in r.items() if k != "ckpt"}
        r.update(clean80_bpc=sc["bits_per_char"], chars=sc["chars"], bytes=sc["bytes"], slot_nats_per_byte=sc["slot_nats_per_byte"],
                 slot_rel_to_slot0=[x / sc["slot_nats_per_byte"][0] for x in sc["slot_nats_per_byte"]], train_flops=r["flops_per_byte"] * r["max_bytes"],
                 token_equiv=r["max_bytes"] / 3.735, sec=time.time() - t)
        res["byte_runs"].append(r)
        log(f"{r['run']} {r['id']} lr {r['lr']} {r['max_bytes'] / 1e6:.0f}M bytes: clean80 {r['clean80_bpc']:.4f} bpc (train-time all-val {r['train_val_bits_per_char_all_val']:.4f}); "
            f"slots {[round(x, 3) for x in sc['slot_nats_per_byte']]} ({r['sec']:.0f}s)")
        json.dump(res, open(out_path, "w"), indent=1)
    # matched-FLOPs comparison: for every byte run, the token runs bracketing / nearest in train FLOPs
    cmp = []
    for r in res["byte_runs"]:
        near = sorted(res["token_runs"], key=lambda t: abs(math.log(t["train_flops"] / r["train_flops"])))[:3]
        cmp.append({"byte_run": r["run"], "id": r["id"], "train_flops": r["train_flops"], "clean80_bpc": r["clean80_bpc"],
                    "nearest_token_runs": [{"run": t["run"], "exp": t["exp"], "train_flops": t["train_flops"], "flops_ratio_vs_byte": t["train_flops"] / r["train_flops"],
                                            "clean80_bpc": t["clean80_bpc"], "byte_over_token_bpc": r["clean80_bpc"] / t["clean80_bpc"]} for t in near]})
    res["matched_flops"] = cmp
    json.dump(res, open(out_path, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
