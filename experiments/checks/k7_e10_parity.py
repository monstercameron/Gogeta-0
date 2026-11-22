"""K7 prerequisite (CPU, ~1 min): the "arch" switch in gogeta/train.py leaves FusionDance unchanged,
and E10 (HF LlamaForCausalLM) matches E0's parameter count and FLOP accounting.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python experiments/checks/k7_e10_parity.py

Checks
  (a) For every FusionDance config (E0, E1, E2, E3, E9): gogeta.train.build_model() gives a state_dict
      bit-identical to the pre-change construction (ModelConfig(**{**model, vocab, max_seq_len}) then
      FusionDance(mc), after the same torch.manual_seed), the same model-config dict, and leaves the
      global torch RNG in the same state (so everything after construction sees the same random stream).
  (b) E10 vs E0: total and non-embedding params, and analytic train FLOPs/token at seq 1024.
  (c) E10 sanity on CPU: tied head, HF init statistics, loss at init close to ln(vocab).
"""
from __future__ import annotations

import json
import math
import time

import torch

from common import ROOT, save  # noqa: F401  (common puts the repo root on sys.path)
from gogeta.model import FusionDance, ModelConfig
from gogeta.train import HFLlama, build_model

PREDICTION = ("(a) bit-identical state_dicts, configs and RNG state for all FusionDance configs; "
              "(b) E10 total 13,767,552 / non-emb 10,621,824 params, identical to E0, and identical "
              "train FLOPs/token; (c) loss at init within 0.1 nats of ln(8192) = 9.011.")
CRITERION = ("PASS if (a) holds exactly for every FusionDance config, (b) E10 and E0 param counts differ by "
             "< 0.5% (total and non-embedding) and FLOPs/token by < 0.5%, and (c) |loss_init - ln V| < 0.5.")

VOCAB, SEQ = 8192, 1024


def old_construction(cfg: dict) -> FusionDance:
    """Exactly what gogeta/train.py did before the arch switch (commit 083f222 and earlier)."""
    mc = ModelConfig(**{**cfg["model"], "vocab_size": VOCAB, "max_seq_len": SEQ})
    return FusionDance(mc)


def main():
    t0 = time.time()
    torch.set_num_threads(2)
    data, ok_a = {"fusiondance": {}}, True
    for name in ["e0_dense", "e1_hybrid", "e2_looped", "e3_ternary", "e9_unlooped"]:
        cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        assert cfg.get("arch", "fusiondance") == "fusiondance"
        torch.manual_seed(cfg["train"]["seed"])
        old = old_construction(cfg)
        rng_old = torch.get_rng_state()
        torch.manual_seed(cfg["train"]["seed"])
        new, mc = build_model(cfg, VOCAB, SEQ)
        rng_new = torch.get_rng_state()
        so, sn = old.state_dict(), new.state_dict()
        same_keys = list(so) == list(sn)
        same_vals = same_keys and all(torch.equal(so[k], sn[k]) for k in so)
        same_cfg = old.cfg.to_dict() == mc.to_dict()
        same_rng = torch.equal(rng_old, rng_new)
        same_flops = old.flops_per_token(SEQ) == new.flops_per_token(SEQ)
        row = {"tensors": len(so), "same_keys": same_keys, "bit_identical": same_vals, "same_config": same_cfg,
               "same_rng_after": same_rng, "same_flops": same_flops, "params": new.param_counts()}
        data["fusiondance"][name] = row
        ok = same_vals and same_cfg and same_rng and same_flops
        ok_a &= ok
        print(f"(a) {name:12s} tensors {len(so):3d}  bit-identical {same_vals}  config {same_cfg}  "
              f"rng {same_rng}  flops {same_flops}  -> {'OK' if ok else 'MISMATCH'}")

    # (b) E10 vs E0
    e0 = json.loads((ROOT / "configs" / "e0_dense.json").read_text())
    e10 = json.loads((ROOT / "configs" / "e10_llama_ref.json").read_text())
    torch.manual_seed(0)
    m0, _ = build_model(e0, VOCAB, SEQ)
    torch.manual_seed(0)
    m10, mc10 = build_model(e10, VOCAB, SEQ)
    assert isinstance(m10, HFLlama)
    p0, p10 = m0.param_counts(), m10.param_counts()
    f0, f10 = m0.flops_per_token(SEQ), m10.flops_per_token(SEQ)
    rel = lambda a, b: abs(a - b) / b
    ok_b = rel(p10["total"], p0["total"]) < 5e-3 and rel(p10["non_embedding"], p0["non_embedding"]) < 5e-3 \
        and rel(f10, f0) < 5e-3
    print(f"(b) E0  params total {p0['total']:,}  non-emb {p0['non_embedding']:,}  train FLOPs/token {f0:,.0f}")
    print(f"    E10 params total {p10['total']:,}  non-emb {p10['non_embedding']:,}  train FLOPs/token {f10:,.0f}")
    data["e0_vs_e10"] = {"e0_params": p0, "e10_params": p10, "e0_flops_per_token": f0, "e10_flops_per_token": f10,
                         "total_diff": p10["total"] - p0["total"],
                         "non_emb_diff": p10["non_embedding"] - p0["non_embedding"]}

    # (c) E10 sanity
    lm = m10.lm
    tied = lm.lm_head.weight is lm.model.embed_tokens.weight
    stds = {k: float(v.detach().std()) for k, v in {
        "embed": lm.model.embed_tokens.weight, "q_proj": lm.model.layers[0].self_attn.q_proj.weight,
        "o_proj": lm.model.layers[0].self_attn.o_proj.weight, "down_proj": lm.model.layers[0].mlp.down_proj.weight,
    }.items()}
    stds_e0 = {"embed": float(m0.embed.weight.detach().std()), "qkv": float(m0.prelude[0].mixer.qkv.weight.detach().std()),
               "o (residual-scaled)": float(m0.prelude[0].mixer.o.weight.detach().std())}
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (2, 128), generator=g)
    y = torch.randint(0, VOCAB, (2, 128), generator=g)
    m10.eval()
    with torch.no_grad():
        logits, loss = m10(x, y)
    ok_c = tied and abs(loss.item() - math.log(VOCAB)) < 0.5 and logits.shape == (2, 128, VOCAB)
    print(f"(c) E10 tied head {tied}; init std {stds}; E0 init std {stds_e0}")
    print(f"    loss at init {loss.item():.4f} (ln V = {math.log(VOCAB):.4f}); model_config keys "
          f"{len(mc10.to_dict())}, arch={mc10.to_dict()['arch']}, rope={mc10.to_dict().get('rope_parameters')}, "
          f"eps={mc10.to_dict()['rms_norm_eps']}")
    data["e10_sanity"] = {"tied": tied, "init_std": stds, "e0_init_std": stds_e0, "loss_init": loss.item(),
                          "ln_vocab": math.log(VOCAB)}

    verdicts = {
        "fusiondance_unchanged": f"{'PASS' if ok_a else 'FAIL'}: bit-identical construction for 5 configs",
        "e10_matches_e0": (f"{'PASS' if ok_b else 'FAIL'}: total {p10['total']:,} vs {p0['total']:,}, non-emb "
                           f"{p10['non_embedding']:,} vs {p0['non_embedding']:,}, FLOPs/token {f10:.4g} vs {f0:.4g}"),
        "e10_sane": f"{'PASS' if ok_c else 'FAIL'}: loss at init {loss.item():.3f}, tied {tied}",
        "pass": bool(ok_a and ok_b and ok_c),
    }
    save("K7-parity", ["E10 control"], PREDICTION, CRITERION, verdicts, data, t0)
    print("PASS" if verdicts["pass"] else "FAIL")


if __name__ == "__main__":
    main()
