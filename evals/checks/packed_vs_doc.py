"""Why does lm-eval's per-document bpc differ from train.py's packed val bpc? (CPU, ~7 min for E0)

    CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta-eval/bin/python -m evals.checks.packed_vs_doc \
        --checkpoint ~/fusiondance-data/checkpoints/E0-20261122-122059-s0/model.pt

Recomputes train.py's protocol (gogeta/train.py: evaluate + val_batches) on CPU in fp32 with
the HF wrapper: the val token stream (val.bin, <|eos|> after every doc) cut into disjoint
1,024-token windows, mean cross-entropy over every position, bpc = loss / ln2 / chars_per_token.
Also reports the same stream with <|eos|> targets excluded, and the loss on the first 64
positions of each window vs the rest (windows start mid-document with no context).

Prediction (written before running): the packed fp32 CPU value reproduces train.py's recorded
val_loss (0.6063) within 0.5%, so the gap to lm-eval (0.2208 vs 0.2351 bpc) is protocol, not a
wiring bug; and the first-64-position loss is clearly higher than the rest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from evals.gogeta_lm import build_model  # noqa: E402

DATA = pathlib.Path(os.environ.get("GOGETA_DATA", pathlib.Path.home() / "fusiondance-data"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = DATA / "datasets" / "code-py-v1"
    meta = json.loads((d / "meta.json").read_text())
    cpt = meta["splits"]["val"]["chars_per_token"]
    val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
    model, prov = build_model(checkpoint=a.checkpoint)
    T = model.config.max_seq_len
    n = (len(val) - 1) // T
    tot = tot_n = noeos = noeos_n = head = head_n = 0.0
    with torch.no_grad():
        for s in range(0, n, a.batch):
            rows = np.stack([val[i * T:i * T + T + 1] for i in range(s, min(s + a.batch, n))]).astype(np.int64)
            x, y = torch.from_numpy(rows[:, :-1]), torch.from_numpy(rows[:, 1:])
            ce = F.cross_entropy(model(x).logits.float().transpose(1, 2), y, reduction="none")
            tot += float(ce.sum()); tot_n += ce.numel()
            m = y != 0
            noeos += float(ce[m].sum()); noeos_n += int(m.sum())
            head += float(ce[:, :64].sum()); head_n += ce[:, :64].numel()
    loss = tot / tot_n
    out = {"checkpoint": prov["path"], "windows": n, "T": T, "chars_per_token": cpt,
           "packed_val_loss": loss, "packed_bpc": loss / math.log(2) / cpt,
           "packed_loss_excl_eos_targets": noeos / noeos_n,
           "loss_first64_positions": head / head_n,
           "loss_after_first64": (tot - head) / (tot_n - head_n)}
    print(json.dumps(out, indent=2))
    p = ROOT / "evals" / "results" / f"check-packed-vs-doc-{pathlib.Path(a.checkpoint).parent.name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
