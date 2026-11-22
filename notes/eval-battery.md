# Frozen evaluation battery v1 (workstream A4)

*2026-11-22. Spec + generators + CPU validation. Nothing here is committed; no GPU was used. Labels (AGENTS.md):
**[verified here]** = run on this machine, file named · **[computed]** = arithmetic shown · **[proposed]** = not run.*

The external audit's complaint: all evidence rests on one metric (next-token loss on one curated Python set), and
our one non-loss check that tried to be independent (K3 MQAR) failed its own positive control (attention 8.6%).
This note freezes **four independent measurement families**, gives each a statistic with an item-level bootstrap CI,
a chance baseline, a **required positive control** and a **pass gate for that control**: *a family is only allowed to
support a claim if a reference model clears its gate*. A task that everything scores at chance (as K3 did, and as
E0 does on the code-native probes below) is reported as `floor / unvalidated`, never as evidence for or against a claim.

Related: `notes/benchmark-suite.md` (survey, tier 1-3 tasks), `evals/README.md`, A2's `experiments/stats/core.py`
(`bootstrap`, `paired_bootstrap`: the CI machinery named below), `experiments/evalbattery/` (this workstream's code).

## 0. TL;DR

1. **Battery v1 = 4 families, 30 task cells**, all seeded and deterministic; item counts are chosen so the 95% CI
   half-width is <= 3 points wherever the task is not floored (arithmetic in section 6).
2. **K3 diagnosed (section 3.2).** The MQAR *task* was fine; the *training recipe* was the problem. Tiny attention
   models solve P=4 pairs to 100% in 3,000 steps at any sane lr, but at P=8 reach 38% and at P=16 only 15-24% in
   3,000 steps, for every lr/curriculum variant tried. This is the induction-circuit plateau: the steps needed grow
   steeply with the number of pairs, and K3 asked for P=64 in 4,000 steps. Fix: [MQAR-FIX].
3. **Floors are the main finding for LM-level probes.** The 13.8M attention-only E0 sits at chance on every
   code-native probe: state tracking 31% vs 25% chance (and *below* chance, 15-17%, on increments), `gogeta_recall_d64`
   0.5-0.7%. These families therefore **cannot discriminate our current checkpoints**; they are kept in the battery
   because they are the only ones that can *ever* test the memory claim at the checkpoint level, but their verdict is
   `floor` until a checkpoint clears the gate. The synthetic from-scratch families (MQAR v2, iterative suite) are the
   ones that can decide architecture claims today.
4. **Blocker (human):** lm-eval-based tasks cannot load GDN checkpoints because `~/.venvs/gogeta-eval` has no
   `flash-linear-attention`. Exact command in section 7. **Not a blocker for my standalone scorers**: they run in
   `~/.venvs/gogeta`, which already has fla 0.6.0 + transformers + torch (checked: `fla 0.6.0`).

## 1. Principles (what "frozen" means)

- **Frozen** = item sets, seeds, prompts, splits and scoring code are fixed under the version tag `battery-v1.0`.
  Anything that changes an item or a score bumps the version; old and new numbers are never mixed in one table.
- **Item-level CI**: every headline number carries a 95% bootstrap CI over *items* (documents, problems, sequences,
  or snippets: the unit named per family). **Items are not seeds**: item CIs say how well the *benchmark* resolves
  one trained model; run-to-run variance needs >= 3 training seeds (A2's `statistics-policy`). Comparisons between
  two checkpoints use the *paired* bootstrap on shared items.
- **Required positive control + gate** per family (below). A claim may cite a family only if its gate was cleared
  *in the same battery version*; otherwise write "family X unvalidated at our scale".
- **Chance baselines are stated and verified by a no-memory / shuffled control**, because "above 1/V chance" is not
  the right bar when a model can exploit a cheaper shortcut (e.g. MQAR: 1/P if it knows which values are in context).
- **Hardware split.** CPU = attention/SWA-only checkpoints (E0, E10) and tiny from-scratch models;
  GPU = anything with `gdn` blocks (needs CUDA + fla). Marked in the tables.

## 2. The battery (v1.0)

| Family | Task IDs (cells) | Items per cell | Unit for CI | Chance | Runs on |
|---|---|---|---|---|---|
| **L** Language modelling | `gogeta_val_bpb` (4,000 val docs), `gogeta_uncheatable_py_bpb` (fresh 2026-10 GitHub Python), + control `gogeta_battery_val_lineshuf_bpb` | 4,000 / ~(see run) / 1,000 docs | document | 8.0 bits/byte (uniform bytes) | CPU for E0/E10; GPU for GDN |
| **F** Functional coding | `gogeta_humaneval_plus_exec`, `gogeta_mbpp_plus_exec` (greedy pass@1; `_n20` variants for pass@k), `gogeta_cruxeval_o_ll`; `_clean` twins | 164 / 378 / 800 | problem | ~0% (pass), 3.6% ("always False") for CRUX-O | GPU (generation has no KV cache) |
| **M** Memory and retrieval | **MQAR v2** (`experiments/evalbattery/mqar.py`); **state-tracking probe** `gogeta_state_{assign,incr}_k{k}_d{d}` (18 cells); **code recall** `gogeta_battery_recall_d{64,256,512,960}` (n raised 200 -> 1,100) | 2,048 seq x P queries / 1,100 / 1,100 | sequence (MQAR), snippet | 1/P (MQAR; 1/V if value set unknown), 25% (probe), ~0.01% (recall, 4-digit exact) | MQAR: tiny from-scratch, CPU or GPU; probe: CPU (attn) / GPU (GDN) |
| **I** Iterative computation | `parity`, `modadd5`, `modadd7`, `graph_hops`, `graph_reach`, `program` x {in-distribution, length/depth-generalisation} (`experiments/evalbattery/itercomp.py`) | 2,048 per split | item (sequence) | 1/2, 1/5, 1/7, 1/10, 1/2, 1/10 | tiny from-scratch, CPU for attention, GPU for GDN |

"BABILong-style" is covered by the state-tracking probe and code recall (needle facts among distractor lines, then a
question), in-distribution for a Python-only model; English BABILong itself is out of distribution (benchmark-suite.md
section 1). "Controlled key-value recall" = MQAR v2 (synthetic) + code recall (code-native).

## 3. Per-family spec

### 3.1 L: language modelling (doc-level bits/byte)

- **Statistic.** Corpus bits/byte = `-sum(ll) / sum(bytes) / ln 2` over documents (lm-eval `bits_per_byte`
  aggregation; each doc scored from an `<|eos|>` prefix in disjoint 1,024-token windows). **CI**: percentile/BCa
  bootstrap resampling *documents* and recomputing the ratio of sums (`stats.core.bootstrap`, per-doc `(ll, bytes)`
  pairs from `--log-samples`). **Model comparisons use the paired document bootstrap** (same docs, per-doc
  bits/byte difference): per-doc scores are strongly correlated across models, so the paired CI is far tighter than
  two unpaired ones (A2's policy; to be quantified on E0 vs E1 logs, SQ135).
- **Chance**: 8.0 bits/byte (uniform over bytes). Cheap-baseline references on val: unigram bytes ~5, xz -9 ~1.2
  (C8: model 3% vs xz 15.4%).
- **Positive control (required).** (a) *Wiring*: lm-eval == hand-written forward pass, <= 1e-3 nats
  (`evals/test_wiring.py`, 6/6 [verified here]). (b) *Trained vs random*: random-init `tiny_attn` 3.62 vs trained E0
  0.22 bits/byte [verified here, notes section 7]. (c) **Line-shuffle negative control** (new,
  `gogeta_battery_val_lineshuf_bpb`): the same docs with lines permuted (seed 0). A model that models code structure
  must lose clearly. **Gate: bpb(lineshuf) - bpb(same docs, original order) >= 0.15.**
  [verified here, n=150 docs, E0, CPU] shuffled 0.787 vs original 0.240 bits/byte: gap **+0.547** (gate passed by 3.6x).
- **Known caveats:** the val set is in-distribution to training (curation overlap), the fresh GitHub set is the
  out-of-distribution check; always report both. Protocol gap vs train.py (0.2208 vs 0.2351 bits/char) is explained
  in benchmark-suite.md section 7.

### 3.2 M: memory and retrieval

#### MQAR v2 (the K3 redesign, Q15/Q16)

**Why K3's attention control reached only 8.6% [verified here, CPU diagnostics, `experiments/evalbattery/results/diag/`].**
2-layer attention, d=64, loss at query positions, AdamW (0.9, 0.95), warm-up 200 + cosine, wd 0.1, batch 64:

| Config (CPU, seed 0, 1,024 eval seqs) | steps | acc | 95% CI (seq bootstrap) | 1/P |
|---|---|---|---|---|
| P=4, T=16, V=16/16, lr 1e-3 | 3,000 | **100.0%** | [100.0, 100.0] | 25% |
| P=4, lr 3e-3 | 3,000 | **100.0%** | [100.0, 100.0] | 25% |
| P=4, lr 1e-2 | 3,000 | 56.9% | [55.7, 58.1] | 25% |
| P=8, T=32, V=32/32, lr 1e-3 | 3,000 | 38.0% | [37.2, 38.8] | 12.5% |
| P=16, T=64, V=64/64, lr 1e-3 (K3-style recipe) | 3,000 | 24.1% | [23.6, 24.6] | 6.3% |
| P=16, lr 3e-3 | 3,000 | 20.9% | [20.4, 21.3] | 6.3% |
| P=16, lr 1e-3 + doubling-P curriculum (4->16) | 3,000 | 23.7% | [23.1, 24.2] | 6.3% |
| P=16, lr 3e-3 + curriculum | 3,000 | 14.6% | [14.2, 15.0] | 6.3% |
| [MQAR-ROWS] | | | | |

Reading: (i) the task and generator are fine (P=4 is solved exactly, and `evals/battery/test_battery.py` checks
every query's target equals its context binding); (ii) accuracy at fixed steps falls steeply with P and is
insensitive to lr (1e-3 to 3e-3) and to a fixed-ramp curriculum: this is the **induction-circuit plateau** (the loss
sits near ln P, "I know which values are in context", until a previous-token head + lookup head form), and its
length grows with P. K3 (P=64, 4,000 steps) was stopped on the plateau: its eval loss 4.23 ~ ln 64 = 4.16.
(iii) it is **not** too-few-heads/too-small-d, vocabulary size, or sequence length per se; it is steps-at-P.
[MQAR-FIX]

**Generator** (`mqar.py`): Zoology-style. Tokens: 0 = filler, keys 1..K, values K+1..K+V (disjoint). Context
`k1 v1 ... kP vP`, then each key queried once in random order followed by its value, filler in unused slots when
`seq_len` > 4P. Loss and accuracy at query-key positions only. Train stream seed s, eval stream seed 10,000+s
(never overlaps). Metric: query-level accuracy with a **sequence-clustered bootstrap** (queries inside a sequence
share the context, so resampling queries would understate the CI), >= 2,048 eval sequences.

- **Chance**: 1/V if the model has no idea which values occur; **1/P** if it knows the context's value set but not
  the binding (this is what K3's attention reached). The gate uses 1/P.
- **Required positive control**: 2-layer attention. **Gate: mean accuracy over 2 seeds >= 90%.**
- **Required negative control (no-memory baseline)**: SWA with window 4 (cannot see the context pairs). **Gate:
  <= 1/P + 5 points.** (K3's swa(32) check used a window equal to half the context at P=16-scale geometry; w=4 is a
  true no-memory control for any P >= 4.)
- **Arms once the gate passes**: attn, swa(4), plain linear attention, GDN, GDN neg-eig (and later KDA, SQ59).
  Pass/fail verdict language for C16 stays "vs plain linear attention" (Mamba-2 not tested).

#### Code-native state-tracking probe (outside review's idea A) [new, `evals/battery/state_track.py`]

A module-level Python snippet reassigns one target variable k times (`assign`: `n = 47`; `incr`: `n = 47` then
`n += 3` ...), interleaved with assignments to other variables, then `distance` distractor lines, then
`assert n ==`. Score = log-likelihood of each of 4 two-digit choices: the correct current value, **stale values of the
same variable** (the signature of failed tracking) and other variables' values. Chance 25%. 18 cells
(assign k in {1,4,8}, incr k in {2,4,8}; distance in {0,16,64} lines), 1,100 items each, seeds fixed per cell.
Inference only; tasks `gogeta_state_<mode>_k<k>_d<d>` (lm-eval) and a standalone scorer
`python -m evals.battery.score_state_track` (also reports the share of errors that chose a stale value).
**Item correctness is checked by exec()-ing the generated snippet** (our own text, never model output) and comparing
the variable with the gold choice (`test_battery.py`, 10/10 [verified here]).

- **E0 result [verified here, CPU, n=300/cell]** (`results/state_track_E0_cpu_n300.json`): assign k=1 d=0 31.0%
  [25.7, 36.3]; k=1 d=16 30.7% [25.7, 36.0]; k=4 d=0 31.3% [26.3, 36.3]; incr k=2 d=0 15.0% [11.0, 19.0]; incr k=4 d=0
  17.3% [13.0, 22.0]. Flat in k and d and *below* chance on increments: E0 has a bias towards the stale/earlier
  value, no tracking. **Verdict: floor.** `gogeta_recall_d64` agrees (0.7%, n=150).
- **Required positive control**: (1) *scorer oracle/pipeline control*: a tiny attention LM trained **on probe-style
  snippets only** (train seeds 1000+, eval seed 0) must reach >= 90% on assign k=1; this proves the probe is solvable
  and the scoring path (eos prefix, tokenisation boundary, choice log-likelihood) works end to end:
  `evals/battery/train_state_control.py` [CONTROL-RESULT]. (2) *Checkpoint validity gate*: a pretrained checkpoint
  is interpretable on a cell only if its lower CI bound is >= 35% (chance + 10); otherwise the cell is `floor`
  for that checkpoint and **excluded from the claim tables** (not counted as "no difference").

#### Code recall (existing tier-3 task, n raised)

`gogeta_battery_recall_d{64,256,512,960}`: same generator as tier 3, `n: 1100` instead of 200 (CI 6.9 -> 2.95 points
worst case). E0 is at floor (0.7% at d=64, n=150). Same validity rule as the probe. Proposed redesign (SQ134):
4-way choice scoring of a 2-digit value instead of 4-digit exact generation, so the floor is 25% not ~0.01%.

### 3.3 I: iterative computation ("do extra loops do useful work?")

`experiments/evalbattery/itercomp.py`, pure-numpy generators, seeded, every item a function of (task, split, seed,
stream) only.

| Task | Input / target | Depth axis | Splits | Chance |
|---|---|---|---|---|
| `parity` | bits; running XOR at every position (K5/SQ6) | sequence length | train T=32, id pos 16-31, **len2x** T=64 pos 32-63, **len4x** T=128 pos 64-127 | 50% |
| `modadd5`, `modadd7` | digits mod p; running sum mod p | sequence length | same | 20%, 14.3% |
| `graph_hops` | 10-node **single-cycle** graph as shuffled `src dst` pairs, query `(s, k)`; answer `f^k(s)` | hops k | train k in 1..4, id 1..4, **gen 5..8** | 10% |
| `graph_reach` | same graph, query `(s, t, k)`; is `t` on `s, f(s), ..., f^k(s)`? (balanced) | hops k | same | 50% |
| `program` | 4 registers (values mod 10), 4 random init SETs + L statements `SET/ADD/MOV/ADDR`; query a register | statements L | train L 1..6, id 1..6, **gen 7..12** | 10% |

- **Why a single cycle** [verified here, `itercomp.py --baselines`]: with a random function graph the shortcut
  "answer = most frequent destination" scored 28.7-33.2%, and with a random permutation "answer = f(s)" scored
  27.7% at k=5..8 (short cycles). A single N-cycle makes both 10.0-10.7% / 0.0% at gen.
  Program shortcuts (heuristic "last constant SET to that register"): id 54.9%, gen 27.0% (chance 10%), reported as
  the reference a model must beat; SET is rare (10% of statements) and initial values are random, so
  "always 0" is 11-12%.
- **Reference checkers** (`itercomp.py --selftest`, 3 s): parity/modadd recomputed by a running loop; graph answers
  recomputed *from the token stream* by walking the edge list; programs **decoded from tokens, rendered as Python
  source and exec'd** (our own text) and compared with the label; YES-rate of `graph_reach` = 0.50. All pass
  [verified here].
- **Statistic**: accuracy on target positions, sequence-clustered bootstrap CI (parity/modadd have 16-64 scored
  positions per sequence), 2,048 items per split. Report the **length/depth-generalisation drop** (id minus gen),
  because "extra loops add useful computation" predicts the looped model degrades more slowly with depth, not just
  scores higher in-distribution.
- **Required positive control + gate.** The task is valid for the loops claim only if (i) the self-test passes;
  (ii) a reference model reaches >= 90% in-distribution (parity/modadd: GDN neg-eig 1 pass, already 100.0%/99.99% on
  SQ6; graph/program: a reference with enough sequential depth, to be established on GPU, SQ131-SQ133); (iii) a
  **label-shuffled control** is within chance +- 3 points; (iv) the best trivial heuristic on the generalisation split is
  <= chance + 10 (graph) / + 20 (program).
- **CPU tiny-model sanity [verified here, d=64, 3,000 steps, seed 0]**: [ITER-ROWS]

### 3.4 F: functional coding

- **Statistic**: greedy pass@1 per problem (binary), paired problem bootstrap between checkpoints; `_n20` variants
  (T=0.8, top-p 0.95) give unbiased pass@1/5/10/20 and a lower-variance estimate; always report the `_clean` twin
  (32/164 HumanEval and 43/500 MBPP problems overlap training text).
- **Chance/floor**: pass ~0%; CRUXEval-O always-"False" = 3.6%.
- **Required positive control.** (a) *Harness gate* (new, SQ138): for every problem, the canonical solution must pass
  100% and an empty/`pass` stub 0% through `evals/sandbox.py`. This validates tests, sandbox and prompt plumbing
  without needing a strong model. (b) *Discrimination gate*: a checkpoint is interpretable on F only if
  pass@1 >= 5% (>= 9/164 HE+; the published smallest data points are Codex-12M 2.0%, PolyCoder-160M 2.1%,
  CodeParrot-110M 3.8% [benchmark-suite.md section 0]); below that F is `floor` and not used in decisions.
- **Resolution (computed)**: HE+ n=164: half-width 1.96 sqrt(p(1-p)/164) = 3.3 points at p=5%, 4.6 at p=10%.
  So **F cannot resolve <3-point differences for HumanEval+** at any plausible p; MBPP+ (n=378) gets 2.2 / 3.0 at
  p = 5% / 10%. F is a *sanity family*, not a deciding one, at this scale.

## 4. Mapping to the paper's four claims

| Claim | L | F | M | I | Deciding evidence |
|---|---|---|---|---|---|
| **P1: hybrid (linear mixer + attn) > attention** at matched compute | paired doc bpb, E1 vs E10/E0 (>= 3 seeds), val + fresh | pass@1 if above floor, else reported as floor | MQAR v2: GDN vs attn and vs plain linear attention (attn is the ceiling); recall/probe if any checkpoint clears the gate | parity/modadd/program: GDN vs attn (1-pass), the K5 "linear mixers are weak at state tracking" claim | L + M(MQAR) |
| **P2: loops add useful computation** | E2 vs E9 (unlooped control) bpb vs r | idem | state-track `incr` cells vs r (loops override) | **graph_hops / program / parity: looped vs unlooped at matched params AND at matched compute, length/depth generalisation slope** | **I** (+ L at matched compute) |
| **P3: ternary is efficient** | bpb penalty E3 vs E1 at same tokens; bytes/bit | pass@1 drop | recall/MQAR degradation under ternary (does ternary hit retrieval first?) | I: does ternary break depth extrapolation? | L (size-quality) + M |
| **P4: synergy (fusion > sum of parts)** | 2x2 interaction: (E4 - E3) - (E1 - E0-or-E10) in bpb, paired bootstrap | | same contrast on MQAR, probe | same contrast on I | **interaction contrast reported per family**; a synergy claim needs the interaction CI to exclude 0 in >= 2 families |

## 5. CPU vs GPU

| Item | Needs | Where |
|---|---|---|
| L, F, probe, recall on **attention-only** checkpoints (E0, E10) | torch only | CPU, 2 threads (E0 val 4,000 docs: 442 s; probe cell 1,100 items k=8 d=64: ~2.5 min est.) |
| same on **GDN checkpoints** (E1, E2, E9, E11) | CUDA + fla | GPU: lm-eval route blocked on fla in `gogeta-eval` (section 7); **standalone probe scorer works today in `~/.venvs/gogeta`** |
| MQAR v2, iterative suite, attention/SWA arms | torch | CPU (tiny) or GPU |
| MQAR v2, iterative suite, GDN / GDN neg-eig / loop-GDN arms | CUDA + fla (`~/.venvs/gogeta`) | GPU only |

## 6. Item counts and the CI arithmetic

Binary accuracy, normal approximation: half-width `h = 1.96 sqrt(p(1-p)/n)`, worst case p = 0.5.
`n >= 1.96^2 * 0.25 / h^2`: **h=3 points -> n >= 1,068; h=2 -> n >= 2,401.**

| Cell | n | worst-case h | note |
|---|---|---|---|
| MQAR eval | 2,048 sequences (x P queries) | 1.96 * 0.5 / sqrt(2048) = **2.17** | sequence as the cluster; true h lower because per-sequence SD < 0.5 near the extremes |
| iterative, per split | 2,048 | **2.17** | |
| state probe, per cell | 1,100 | 1.96 * 0.5 / sqrt(1100) = **2.95** | pooling the 3 distances of a (mode,k) row: 3,300 -> 1.7 |
| code recall (battery copy) | 1,100 | **2.95** (was 6.9 at n=200) | |
| CRUXEval-O | 800 | 3.5 at p=.5, **2.0 at p=.09** | |
| HumanEval+ | 164 | 7.7 at p=.5; **3.3 at p=.05** | not resolvable to 3 points: sanity family |
| MBPP+ | 378 | 5.0 at p=.5; **2.2 at p=.05, 3.0 at p=.10** | |
| LM bpb | 4,000 docs | = 1.96 SD_doc / sqrt(4000); SD_doc from the E0 logs (SQ135) | the paired difference CI is what is used |

Training-seed variance is *not* in these intervals (K5/SQ6: the same 1-pass GDN arm scored 81.5%, 73.3% and 50.1% on
three runs, C44).

## 7. Blockers and exact commands

- **`flash-linear-attention` is not in `~/.venvs/gogeta-eval`** (lm-eval 0.4.13, scipy, pandas present; checked).
  Needed only for the lm-eval route on GDN checkpoints (E1, E2, E9, E11). **Human action** (a network install into
  a venv, so it is parked as AGENTS.md requires):
  `~/.venvs/gogeta-eval/bin/pip install flash-linear-attention==0.6.0` (the pin in `evals/README.md`; torch 2.14.1
  must stay: check `python -c "import torch,lm_eval,fla; print(torch.__version__)"` afterwards).
- **Workaround that needs no install**: `python -m evals.battery.score_state_track` and the tiny-model scripts run in
  `~/.venvs/gogeta` (fla 0.6.0, transformers 5.24.0, torch 2.14.1). Only the lm-eval tasks (L, F, recall) wait on the
  install. HumanEval+/MBPP+/fresh GitHub data still need the cached datasets (benchmark-suite.md section 2; already
  fetched for tier 1).
- E10 (HF Llama control) loads through the HF-dir path of `evals/gogeta_lm.py`; not tested here.
- The standalone scorers import `gogeta.hf.from_gogeta_checkpoint`; it needs `GogetaConfig.gdn_neg_eigval`
  (fixed in queue item Q-hf; E0 loaded fine).

## 8. What I verified (all CPU, <= 2 threads, no GPU) and where

| What | Result | File |
|---|---|---|
| Unit tests (state-probe exec check, MQAR semantics, bootstrap width, iterative shapes/determinism) | 10/10 OK, 3 s | `evals/battery/test_battery.py` |
| Iterative self-test + baselines | PASS; shortcut table in section 3.3 | `experiments/evalbattery/itercomp.py` |
| lm-eval route on E0 CPU, `--limit 150`: val bpb 0.2398, **lineshuf 0.7867**, recall_d64 0.7%, state_assign_k1_d0 32.7% | all 4 battery tasks run end to end | `evals/results/battery-smoke-E0-limit150-tier1-20261122-183529.json` |
| State probe on E0 (n=300 x 5 cells) | floor, section 3.2 | `experiments/evalbattery/results/state_track_E0_cpu_n300.json` |
| MQAR K3 diagnosis grid | section 3.2 | `experiments/evalbattery/results/diag/*.json` |
| Iterative tiny-model sanity | section 3.3 | `experiments/evalbattery/results/itercomp/*.json` |

## 9. Next steps

See `experiments/evalbattery/proposed-queue.json` (GPU campaign after Phase 2b), `proposed-claims.md` (C170-C179),
`proposed-side-quests.md` (SQ130-SQ139), `proposed-predictions.md`.
