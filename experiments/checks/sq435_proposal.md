# SQ435 registration proposal: loss-matched corruption control for C213

Script: `experiments/checks/sq435_loss_matched_corruption.py` (complete; modes `--selftest`, `--calibrate`, `--score --part 0|1`, `--analyze`; nothing has been scored or calibrated, only the model-free `--selftest` was run). Freeze its sha256 before running (the hash of the file at proposal time is printed by `sha256sum experiments/checks/sq435_loss_matched_corruption.py`; main agent: re-hash at registration).

## Question
Out of corpus (SQ99) the ternary cost (A2s vs A2f: +8.9% in corpus, +0.4% out) and the loop cost (A2f vs E24: +2.5% in corpus, -1.9% out) shrink (C213, C214), but a higher loss level compresses every relative gap. SQ435 asks whether the shrinkage follows the LOSS LEVEL or the corpus membership: make an in-corpus set as hard as SQ99 for the same reference model, by a corruption that keeps the documents in-corpus in every other respect, and measure the same costs there.

## Method (all fixed by the script)
- Documents: 1,000 clean80 docs of the code-py-v1 val set (K9 overlap < 0.8), drawn by `random.Random(435)`; a disjoint 200-doc calibration sample from the same shuffle. (v2 val is not local; SQ379/SQ400 used the same v1 clean80 docs.)
- Corruption: rename distinct user identifiers (not keywords, builtins, attributes after `.`, import names) with probability p, consistently per doc, to random lowercase strings of the same length (character count, the bits/char denominator, unchanged). Fenced code blocks in algorithmic docs, the whole text of pure-code docs. Tested: deterministic, same length (300/300), 299/300 docs still parse.
- Calibration: bisection on p (<= 8 iterations, tolerance 1%) so that E24 seed 2's bits/char on the corrupted calibration docs equals E24's mean bits/char on SQ99 (seeds 2-7; about 1.70).
- Scoring: A2s, A2f, E27, E24, E20 at seeds 2-4 (CPU scorer of `sq339_cpu.py`, fp32, K8 windowing) on the 1,000 test docs uncorrupted and corrupted at p*; plus p*/2 and min(1, 2p*) for seeds 2-3 (dose-response). Per-doc NLL stored.
- Effects: paired-by-seed 100 x (bpc_a / bpc_b - 1), t-CI (df = n - 1): ternary cost A2s vs A2f, loop cost A2f vs E24, assembly gain A2s vs E27.

## Ladder if the target is unreachable
If E24 at p = 1 stays more than 1% below the target (the corruption saturates before the SQ99 loss level), the calibration records `unmatched_p1_below_target`, the run is labelled UNMATCHED, the achieved E24 loss level is reported, and the result is descriptive only (the dose-response at p*/2 and p* still shows how the costs move with corruption level). No second corruption type is introduced after the fact.

## Prediction (written before any score)
Neither pure hypothesis; the costs at matched loss are expected to fall roughly half way to their out-of-corpus values: ternary cost +3% (range 0% to +6%; in-corpus +8.9%, out of corpus +0.4%), loop cost +0.5% (range -1.5% to +2%; in-corpus +2.5%, out -1.9%). Assembly gain A2s vs E27 stays between -12% and -16% (in corpus -16.5%, out -11.3%). Reason: a part of the out-of-corpus shrinkage is the loss-level compression the corruption reproduces, a part is the loss of memorised content the corruption does not touch (control flow and docstring content of the original docs stay in-corpus).

## Decision rule (fixed)
For each of the two costs, f = (cost at matched loss - cost out of corpus) / (cost in corpus - cost out of corpus), the fraction of the in-corpus cost that survives at matched loss; the in-corpus value is measured on the SAME 1,000 uncorrupted docs, the out-of-corpus value is the SQ400 paired-by-seed mean (+0.42% ternary, -1.90% loop).
- f < 0.25: "scale artifact" (the cost falls to its out-of-corpus value at matched loss). f > 0.75: "memorisation stands". Otherwise "mixed".
- Overall: "scale artifact" only if BOTH costs are < 0.25; "memorisation stands" only if BOTH are > 0.75; otherwise "mixed". The overall label requires E24's mean bits/char at p* within 3% of the target; otherwise the run is UNMATCHED and descriptive only.
- A "scale" result is the weaker conclusion (renaming matches the loss level, not the kind of errors library code induces); "memorisation stands" would be strong (matched loss without the memorisation lever). The overall result is exploratory/diagnostic and does not by itself change a claim status; it qualifies C213's mechanism.

## Caveats stated in advance
- Renamed identifiers are a different kind of unpredictability from unseen library idioms; loss-matching on E24 does not match every arm's loss (the other arms are scored, not matched): the loss levels reached by A2s/A2f/E27/E20 are reported.
- n = 3 seeds per arm (E24 seeds 2-4), t-intervals with df 2; the classification uses point estimates and the intervals are context.
- Cost: calibration about 15 CPU-min (E24 s2, 200 docs x <= 8 iterations), scoring about 1.5 CPU-h for 15 models x 2-4 levels on 1,000 docs; run as two parts.
