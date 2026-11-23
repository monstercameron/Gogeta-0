"""SQ347: copy-deficit probe on the 200M F2 cells (registered in experiments/checks/PREDICTIONS.md, section SQ347).

Question: is the 200M anti-synergy (I_bpc ~ +6.2 log-points) a recall deficit? Synthetic sequences, all five cells at their best LR,
seeds 0-2, fp32 CPU (GDN via the chunkwise torch kernel):
  [BOS] pre-filler (64 tokens) | span S (32 tokens) | gap filler (G tokens) | span S again (32 tokens),  G in {128, 256, 512}.
Filler and spans are random token windows cut from the clean80 val docs (the same docs/subset source as the other SQs; the
token stream is the concatenation of all clean80 docs without separators), drawn with random.Random(0): 200 sequences per
gap, the SAME sequences for every cell and seed. y(cell, seed, gap) = log of the mean NLL (nats/token) on the second copy's 32
tokens, averaged over the 200 sequences. We also record the NLL of the first occurrence of the span (same tokens, no copy
available) so the "copy gain" is visible.
I_copy = y111 - y100 - y010 - y001 + 2*y000 with the seed-blocked estimator of experiments/checks/sq263.py (cell + seed effects, seeds
0-2 shared), in log-points; reference: I_bpc seed-blocked from sq263.json.
Registered prediction: at gaps 256 and 512, I_copy > 0 with the seed-blocked 95% CI excluding 0, and |I_copy| within 0.5x to 2x of I_bpc;
falsifier: |I_copy| < 1 point or the opposite sign at both gaps.
Caveat: sequences are synthetic and the spans are natural-text windows, not identifiers; 200 sequences x 32 tokens per cell/gap.

Run (WSL): PYTHONPATH=.:experiments/checks python experiments/checks/sq347_copy_deficit.py -> results/sq347.json
"""
import json
import math
import random
import time

import torch

import sq339_cpu as C
from sq340_bf16_emulation import seed_blocked_I

GAPS = (128, 256, 512)
N = 200
PRE, SPAN = 64, 32
CELLS = ("000", "001", "010", "100", "111")


def build_sequences():
    tok = C.tokenizer()
    bos = tok.token_to_id("<|eos|>")
    stream = []
    for _, t in C.clean80_docs():
        stream.extend(tok.encode(t).ids)
    rng = random.Random(0)
    seqs = {}
    for g in GAPS:
        lst = []
        for _ in range(N):
            def cut(n):
                a = rng.randrange(0, len(stream) - n)
                return stream[a:a + n]
            pre, span, gap = cut(PRE), cut(SPAN), cut(g)
            lst.append([bos] + pre + span + gap + span)
        seqs[g] = lst
    return seqs, len(stream)


@torch.no_grad()
def score(model, seqs, g, bs=16):
    first, second = [], []
    s1, s2 = 1 + PRE, 1 + PRE + SPAN + g          # index of the first token of the first / second span in ids
    for i in range(0, len(seqs), bs):
        x = torch.tensor(seqs[i:i + bs])
        lp = torch.log_softmax(model(x)[0][:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)                     # nll[:, t-1] is the loss of token t
        first.extend(nll[:, s1 - 1:s1 - 1 + SPAN].mean(1).tolist())
        second.extend(nll[:, s2 - 1:s2 - 1 + SPAN].mean(1).tolist())
    return sum(first) / len(first), sum(second) / len(second)


def main():
    log = C.make_log(str(C.RES / "sq347.log"))
    runs = C.best_runs()
    seqs, nstream = build_sequences()
    log(f"stream tokens {nstream}; sequences {N} per gap, lengths {[len(seqs[g][0]) for g in GAPS]}")
    res = {"check": "SQ347", "git": C.git_commit(), "status": "registered prediction in PREDICTIONS.md (section SQ347)", "n_seq_per_gap": N,
           "gaps": list(GAPS), "span": SPAN, "seeds": [0, 1, 2], "metric": "mean NLL nats/token on the second copy (log scale for I_copy)",
           "second_copy_nll": {}, "first_occurrence_nll": {}}
    for cell in CELLS:
        res["second_copy_nll"][cell], res["first_occurrence_nll"][cell] = {}, {}
        for seed in (0, 1, 2):
            model = C.load_model(runs[cell][seed])
            t = time.time()
            r2, r1 = {}, {}
            for g in GAPS:
                r1[g], r2[g] = score(model, seqs[g], g)
            res["second_copy_nll"][cell][str(seed)] = r2
            res["first_occurrence_nll"][cell][str(seed)] = r1
            log(f"{cell} s{seed} ({time.time() - t:.0f}s) second-copy NLL " + " ".join(f"g{g}={r2[g]:.4f}" for g in GAPS)
                + " | first-occurrence " + " ".join(f"g{g}={r1[g]:.4f}" for g in GAPS))
            json.dump(res, open(C.RES / "sq347.json", "w"), indent=1)
    sq = json.load(open(C.RES / "sq263.json"))["H3prime_seed_blocked"]
    res["I_bpc_reference_points"] = 100 * sq["I"]
    res["I_copy"] = {}
    for g in GAPS:
        y = {c: {int(s): math.log(res["second_copy_nll"][c][s][g]) for s in res["second_copy_nll"][c]} for c in CELLS}
        I, ci, df = seed_blocked_I(y)
        yc = {c: sum(y[c].values()) / 3 for c in CELLS}
        res["I_copy"][str(g)] = {"I_points": 100 * I, "ci95_points": [100 * ci[0], 100 * ci[1]], "df": df,
                                 "ratio_to_I_bpc": I / sq["I"],
                                 "mean_second_copy_nll": {c: math.exp(yc[c]) for c in CELLS},
                                 "vs_dense_pct": {c: 100 * math.expm1(yc[c] - yc["000"]) for c in CELLS},
                                 "I_per_seed_points": [100 * (y["111"][s] - y["100"][s] - y["010"][s] - y["001"][s] + 2 * y["000"][s]) for s in (0, 1, 2)]}
        log(f"gap {g}: I_copy {100 * I:+.2f} points, CI [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] (df {df}), ratio to I_bpc {I / sq['I']:.2f}")
    d = res["I_copy"]
    ok = {g: (d[str(g)]["ci95_points"][0] > 0 and 0.5 <= d[str(g)]["ratio_to_I_bpc"] <= 2.0) for g in (256, 512)}
    fals = all(abs(d[str(g)]["I_points"]) < 1 or d[str(g)]["I_points"] < 0 for g in (256, 512))
    res["prediction_check"] = {"pass_at_256_and_512_ci_excludes0_and_within_0.5_2x": ok, "falsifier_met_both_gaps": fals}
    json.dump(res, open(C.RES / "sq347.json", "w"), indent=1)
    log(json.dumps(res["prediction_check"]))


if __name__ == "__main__":
    main()
