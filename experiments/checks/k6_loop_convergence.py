"""K6: does E2's loop converge? (claim C20). Design: experiments/checks/PREDICTIONS.md."""
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import save, seed_all  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gogeta.model import FusionDance, ModelConfig  # noqa: E402

PREDICTION = ("Not convergent. The relative change stays > 0.1 out to k=12 and the norm keeps growing; the loss is "
              "lowest near k=4.")
CRITERION = ("C20 'each pass refines (convergent loop)' is VERIFIED only if the relative change is non-increasing "
             "for k >= 4 and < 0.05 at k=12. Otherwise it's REFUTED for E2.")

CKPT = os.path.expanduser("~/fusiondance-data/checkpoints/E2-20261122-132340-s0/model.pt")
VAL = os.path.expanduser("~/fusiondance-data/datasets/code-py-v1/val.bin")
N_SEQ, T, K_MAX = 16, 1024, 12


def main():
    started = time.time()
    seed_all(0)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = FusionDance(cfg)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()

    val = np.memmap(VAL, dtype=np.uint16, mode="r")
    seqs = np.stack([np.asarray(val[i * T:(i + 1) * T + 1], dtype=np.int64) for i in range(N_SEQ)])
    idx = torch.from_numpy(seqs[:, :-1]).cuda()
    tgt = torch.from_numpy(seqs[:, 1:]).cuda()

    def coda_loss(h):
        x = h
        for b in model.coda:
            x = b(x)
        logits = model.head(model.norm_out(x))
        return float(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1)))

    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = model.embed(idx)
        for b in model.prelude:
            x = b(x)
        e = h = x
        h0 = h.float()
        h0_norm = float(h0.norm())
        rows.append({"k": 0, "norm_ratio": 1.0, "rel_change": None, "loss": coda_loss(h)})
        prev = h0
        for k in range(1, K_MAX + 1):
            h = h + model.inject(torch.cat([model.inj_norm_h(h), model.inj_norm_e(e)], dim=-1))
            for b in model.core:
                h = b(h)
            hf = h.float()
            # per-token relative change, averaged (secondary) and global Frobenius (primary)
            tok_rc = ((hf - prev).norm(dim=-1) / prev.norm(dim=-1).clamp(min=1e-6)).mean()
            rows.append({
                "k": k,
                "rel_change": float((hf - prev).norm() / prev.norm()),
                "rel_change_per_token_mean": float(tok_rc),
                "norm_ratio": float(hf.norm()) / h0_norm,
                "loss": coda_loss(h),
            })
            print(rows[-1], flush=True)
            prev = hf
        # sanity: our replica matches model.forward at the configured loop count
        _, ref_loss = model(idx, tgt, loops=cfg.loops)
    ref_loss = float(ref_loss)

    rc = {r["k"]: r["rel_change"] for r in rows if r["k"] >= 1}
    nonincr = all(rc[k + 1] <= rc[k] for k in range(4, K_MAX))
    final_small = rc[K_MAX] < 0.05
    ok = nonincr and final_small
    best = min(rows[1:], key=lambda r: r["loss"])
    verdicts = {
        "C20": ("VERIFIED (convergent loop)" if ok else
                f"REFUTED for E2 (rel change non-increasing for k>=4: {nonincr}; rel change at k=12 = "
                f"{rc[K_MAX]:.4f}, need < 0.05)"),
        "criterion_pass": bool(ok),
        "rel_change_nonincreasing_k_ge_4": bool(nonincr),
        "rel_change_k12": rc[K_MAX],
        "best_loss_k": best["k"],
    }
    data = {
        "checkpoint": CKPT, "model_config": ck["model_config"], "n_seq": N_SEQ, "seq_len": T,
        "val_sequences": "first 16 non-overlapping 1024-token windows of val.bin (+1 token for targets)",
        "norms": "Frobenius norm over the whole [16,1024,d] tensor (primary); per-token mean also given",
        "rows": rows,
        "sanity_model_forward_loss_at_cfg_loops": {"loops": cfg.loops, "loss": ref_loss,
                                                   "replica_loss": rows[cfg.loops]["loss"]},
    }
    print("sanity forward loss", ref_loss, "replica", rows[cfg.loops]["loss"])
    save("K6", ["C20"], PREDICTION, CRITERION, verdicts, data, started)


if __name__ == "__main__":
    main()
