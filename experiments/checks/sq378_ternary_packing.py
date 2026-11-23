"""SQ378 (descriptive, not registered): a real packed ternary artifact for the assembled model, and reload parity.

BitLinear keeps fp32 latent weights and fake-quantises them in the forward, so the 'ternary saves bytes' numbers (C9) were computed, never measured.
Here, for the first A2s checkpoint found (seeds 2-4; pass --all for all): every BitLinear layer (the ternary core; prelude, coda and the inject are full
precision in A2s) is packed to 2-bit codes (code + 1 in {0,1,2}, 4 per byte, little-endian within the byte) plus one fp32 scale (absmean of the latent weight,
as BitLinear computes it); everything else (embeddings, norms, full-precision layers, GDN a/b, convs, ...) stays fp32 in the same file. Reported: bytes of
model.pt (the training checkpoint), bytes of the packed file on disk (torch.save), the ternary-layer bytes before and after, resident bytes of the packed
tensors, the information-theoretic size at log2(3) bits/weight, and a bf16 equivalent. Unpack -> a model whose BitLinear layers use the stored codes x scale
directly (the weight is NOT re-quantised from the dequantised weights: absmean of codes x scale differs from the scale, so a plain fp reload through the stock
BitLinear re-rounds differently; that naive reload is measured too) with the same per-token int8 activation quantisation; parity vs the original fake-quant
model on 20 fixed clean80 docs (max abs logit diff, argmax agreement) and clean80 300-doc bits/char.
The packed file is written under ~/fusiondance-data/scratch (not the repo).

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq378_ternary_packing.py -> results/sq378.json
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

import sq339_cpu as C
from sq379_a2_out_of_corpus import find_runs


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int64) + 1
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.ones(pad, dtype=torch.int64)])  # pad with code 0 (stored as 1)
    f = flat.view(-1, 4)
    return (f[:, 0] | (f[:, 1] << 2) | (f[:, 2] << 4) | (f[:, 3] << 6)).to(torch.uint8)


def unpack_codes(packed: torch.Tensor, n: int) -> torch.Tensor:
    p = packed.to(torch.int64)
    f = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=1).flatten()[:n]
    return (f - 1).to(torch.float32)


def bitlinears(model):
    from gogeta.model import BitLinear

    return [(n, m) for n, m in model.named_modules() if isinstance(m, BitLinear)]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def main():
    log = C.make_log(str(C.RES / "sq378.log"))
    from gogeta.model import FusionDance, ModelConfig

    allruns = find_runs()
    runs = sorted((k for k in allruns if k[0] == "A2s"), key=lambda k: k[1])
    if "--all" not in sys.argv:
        runs = runs[:1]
    docs = C.subset(300, 0)
    parity_docs = docs[:20]
    res = {"check": "SQ378", "git": C.git_commit(), "status": "descriptive, not registered", "runs": {}}
    scratch = os.path.expanduser("~/fusiondance-data/scratch")
    os.makedirs(scratch, exist_ok=True)
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    for k in runs:
        e, seed = k
        run = allruns[k]
        ck_path = C.ckpt_path(run)
        model = C.load_model(run)
        layers = bitlinears(model)
        packed, scales, shapes, n_w = {}, {}, {}, 0
        for n, m in layers:
            w = m.weight.detach()
            s = w.abs().mean().clamp(min=1e-5)
            codes = (w / s).round().clamp(-1, 1)
            packed[n], scales[n], shapes[n] = pack_codes(codes), s.clone(), list(w.shape)
            n_w += w.numel()
        names = {n + ".weight" for n, _ in layers}
        rest = {kk: v for kk, v in model.state_dict().items() if kk not in names}
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        art = {"format": "sq378-ternary-2bit-v1", "model_config": ck["model_config"], "packed": packed, "scale": scales, "shape": shapes, "rest": rest}
        out = os.path.join(scratch, f"sq378_{run}.packed.pt")
        torch.save(art, out)
        rest_bytes = sum(v.numel() * v.element_size() for v in rest.values())
        packed_bytes = sum(v.numel() for v in packed.values())
        info = {"run": run, "n_bitlinear_layers": len(layers), "n_ternary_weights": n_w, "model_pt_bytes": os.path.getsize(ck_path),
                "packed_file_bytes": os.path.getsize(out), "ternary_fp32_bytes": n_w * 4, "ternary_packed_bytes": packed_bytes + 4 * len(layers),
                "ternary_ratio_fp32_over_packed": n_w * 4 / (packed_bytes + 4 * len(layers)), "ternary_bf16_bytes": n_w * 2,
                "ternary_entropy_bytes_log2_3": n_w * math.log2(3) / 8, "other_params_fp32_bytes": rest_bytes,
                "total_params_fp32_bytes": sum(p.numel() for p in model.parameters()) * 4, "resident_packed_tensors_bytes": packed_bytes}
        info["file_over_model_pt"] = info["packed_file_bytes"] / info["model_pt_bytes"]
        info["file_over_total_params_fp32"] = info["packed_file_bytes"] / info["total_params_fp32_bytes"]
        art2 = torch.load(out, map_location="cpu", weights_only=False)
        fresh = FusionDance(ModelConfig(**dict(art2["model_config"])))
        missing = fresh.load_state_dict(art2["rest"], strict=False)
        assert all(kk in names for kk in missing.missing_keys) and not missing.unexpected_keys, (missing.missing_keys[:3], missing.unexpected_keys)
        for n, m in bitlinears(fresh):
            shape = art2["shape"][n]
            wq = unpack_codes(art2["packed"][n], numel(shape)).view(shape) * art2["scale"][n]
            m.weight.data.copy_(wq)

            def fwd(x, m=m, wq=wq):
                if m.quant_acts:
                    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                    x = ((x * s).round().clamp(-128, 127)) / s
                return F.linear(x, wq, m.bias)

            m.forward = fwd
        fresh = fresh.float().eval()
        ids = torch.nn.utils.rnn.pad_sequence([torch.tensor([bos] + tok.encode(t).ids[:255]) for _, t in parity_docs], batch_first=True)
        with torch.no_grad():
            a, b = model(ids)[0], fresh(ids)[0]
        info["max_abs_logit_diff"] = float((a - b).abs().max())
        info["argmax_agreement"] = float((a.argmax(-1) == b.argmax(-1)).float().mean())
        info["clean80_300_bpc_fakequant"] = C.bpc(C.score_docs(C.fwd_default(model), docs))
        info["clean80_300_bpc_packed_reload"] = C.bpc(C.score_docs(C.fwd_default(fresh), docs))
        naive = C.load_model(run)
        for n, m in bitlinears(naive):
            shape = art2["shape"][n]
            m.weight.data.copy_(unpack_codes(art2["packed"][n], numel(shape)).view(shape) * art2["scale"][n])
        info["clean80_300_bpc_naive_dequant_reload_through_stock_BitLinear"] = C.bpc(C.score_docs(C.fwd_default(naive), docs))
        res["runs"][f"{e}_s{seed}"] = info
        log(json.dumps({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in info.items()}))
        json.dump(res, open(C.RES / "sq378.json", "w"), indent=1)


if __name__ == "__main__":
    main()
