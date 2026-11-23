# Side-quest worker log (CPU items SQ339...SQ351, SQ61)

Append-only, newest last. Files listed are new or edited by the worker; nothing is committed.

## Shared CPU infrastructure (used by SQ339/340/341/347/348/351)
- `experiments/checks/sq339_cpu.py`: CPU loader/scorer for the 200M F2 checkpoints. GDN kernel replaced by `gdn_chunk_cpu` (fp32 chunkwise torch, matches the per-token fp32 `reference()` to 5e-7 on the output, 3e-6 on the state; 5-12x faster). Scoring docs = clean80 subset of the v1 `docs_val.jsonl` (K9 overlap < 0.8): the code-py-v2 dataset is NOT local, but this is exactly the set the SQ263 fleet scorer used (doclevel JSONs: overlap_source K9, clean80 = 2,333 docs); tokenizer sha256 is identical for v1 and v2.

## SQ346 (done): HOME-hidden probe in `evals/sandbox.py` self_test
Four new probes (home_listdir_raises, home_venvs_invisible, negative_control_fails, positive_control_proc_visible); self-test 16/16 pass. Files: `evals/sandbox.py`, `tests/test_sandbox_gate.py`, `experiments/devlogs/SQ346.md`.

## SQ352 (done): fail closed at the execution point
`run_tests` = once-per-process cached self-test gate + `_run_tests_unchecked`; failing self-test raises, no child started (tested with a Popen tripwire, incl. gogeta_utils exec path and lm-eval TaskManager path). `run_suite.py` untouched. 6 unittest OK (`python -m unittest tests.test_sandbox_gate`, gogeta-eval venv; pytest is not installed in any venv). Files: `evals/sandbox.py`, `tests/test_sandbox_gate.py`, `experiments/devlogs/SQ352.md`.

## SQ345 (done): hashes in score_doclevel
Writer adds docs_val_sha256 / overlap_sha256 / overlap_file_sha256; `check_compatible()` + `HashMismatch`; old files accepted (or refused with require_hash). Analysis scripts NOT edited (not mine): main agent should call `check_compatible` where score files are combined; the 24 SQ263 files have no hashes. Files: `experiments/stats/score_doclevel.py`, `experiments/stats/test_score_doclevel_hashes.py`, `experiments/devlogs/SQ345.md`.

## SQ61 (core done): namespace hardening upstreamed to `post_training/rewards.py`
`run_tests(fs_isolation="auto"|"require"|"off")`, `fs_private` in results; `code_exec_reward` defaults to "require" (fail closed, no child). evals/sandbox.py now delegates (self-test 17/17 incl. SQ346 probes). `python -m unittest post_training.test_rewards`: 26 OK (13 new: repo-write control fails, require+no-namespace raises with Popen tripwire, mount failure -> raise). NOT done: resource-budget, world-writable sweep, descendant-cleanup tests from the row. Note: the default behaviour of `run_tests` under "auto" now uses namespaces when available. Files: `post_training/rewards.py`, `post_training/test_rewards.py`, `evals/sandbox.py`, `experiments/devlogs/SQ61.md`.

## SQ339 (done): post-hoc split of the ternary cost
CPU path validated vs GPU: seed-0 full clean80 gaps -0.015% (000), -0.011% (100), -0.013% (010), +0.002% (111). Eval-time fake quant, 300 clean80 docs, seeds 0-2: act-int8 only +0.11% (000, CI +0.06..+0.16), +0.09% (100); weight-ternary only +769% (000), +197% (100) (no QAT, catastrophic, not comparable to the trained +15.75%); both ~= weights-only. Files: `experiments/checks/sq339_cpu.py`, `sq339_cpu_validate.py`, `sq339_ternary_split.py`, results `sq339.json`, `sq339_cpu_validation.json`, devlog `experiments/devlogs/SQ339.md`.

## SQ366 (done): `score_doclevel.py --store-nll`
Per-doc NLL + token counts to `<run>.nll.npz` (default off; JSON unchanged otherwise). `score_perdoc` = K8.score copy with per-doc outputs; totals verified equal to K8.score in a CPU test; npz round-trip test. Not exercised on GPU. Files: `experiments/stats/score_doclevel.py`, `experiments/stats/test_score_doclevel_hashes.py`, `experiments/devlogs/SQ366.md`.

## SQ341 + SQ348 (done): loop-count curve and per-pass logit lens (one shared pass, r = 12)
Shared-trunk readout verified identical to model(loops=k) (max dNLL 0.0). With coda, 3 seeds, 300 clean80 docs: both loop cells are best at r=4 (010 0.2865, 111 0.3387 bpc); r=1 +8.3%/+7.3%, r=12 +5.6%/+4.3% vs r=4; vs dense (0.2614) at r=4: +9.6% (010), +29.6% (111). No-coda lens: 010 0.497->0.391 (min k=8), 111 0.422->0.373 (min k=5-6); coda worth 23% (010) / 9% (111) at k=4. Files: `experiments/checks/sq341_sq348_loops.py`, results `sq341.json`, `sq348.json`, `sq341_sq348_raw.json`, devlogs `SQ341.md`, `SQ348.md`.

## SQ351 (done): length extrapolation, cells 000 / 100
Only 3 clean80 val docs exceed 1,200 tokens (set A, anecdotal), so added set B = 153 sha-verified SQ99 out-of-corpus files > 1,200 tokens. Nats/token by position bin, 3 seeds: dense 000 on B: 3.95 / 3.03 / 3.99 (bin3 +160% over bin2), GDN 100: 3.95 / 3.15 / 3.05 (bin3 -10%); GDN vs dense -60.7% [-69.8,-51.5] in 1024-2048, +13% in 512-1024. Set A same direction (-29%). RoPE-extension control matches the 1024 window to 1.6e-5. Files: `experiments/checks/sq351_length_extrap.py`, `results/sq351.json`, `experiments/devlogs/SQ351.md`.

## SQ340 (done; PREDICTIONS section SQ340 was committed in v0.5.3 before running): fp32 vs bf16 emulation
Max |shift| 0.0035% (cell 111, beta also rounded), 111-vs-100 shift diff 0.0004 / 0.0038 points; I (300 docs, seed-blocked) fp32 +6.36 [+4.67,+8.06] vs bf16 arm +6.36 / +6.37. Prediction held (< 1 point), falsifier not met. Emulation is partial (mixer cast points only). Files: `experiments/checks/sq340_bf16_emulation.py`, `results/sq340.json`, `experiments/devlogs/SQ340.md`.

## SQ373 (done; high priority): reward-forgery fix in `post_training/rewards.py`
Harness is now grader + forked worker: pre-bound exec/compile/BaseException/os/json, fresh namespace + pristine builtins per test, inner nonce sent only after the solution finished, result via pipe checked by the grader, threads/trace hooks scored as failure. The probe's exec_noop/compile_noop now 0/1 (were 1/1); 11 forgery cases + honest controls in `TestForgery`; `post_training.test_rewards` 30 OK; `evals/sandbox.py` self-test ok (17), `tests.test_sandbox_gate` 6 OK. Residual: a worker-side attacker imitating the pipe protocol or an always-equal return object is not stopped. Files: `post_training/rewards.py`, `post_training/test_rewards.py`, `experiments/devlogs/SQ373.md`.

## SQ347 (done; registered): copy-deficit probe
Prediction NOT met at gaps 256/512: I_copy +3.51 [-2.99,+10.01] and +3.27 [-3.20,+9.74] points (seed-blocked, df 8; I_bpc ref +6.21); falsifier not met either (sign positive everywhere). Gap 128: +8.95 [+1.98,+15.92]. Descriptive: GDN cells (100/111) copy far worse than dense at gap 256 (+32%/+37% second-copy NLL vs dense); loops/ternary alone ~ dense. Files: `experiments/checks/sq347_copy_deficit.py`, `results/sq347.json`, `experiments/devlogs/SQ347.md`.

## SQ365 (done): ternary threshold sensitivity
Gaussian jitter of 5% of absmean scale, 3 draws, 300 clean80 docs, seeds 0-2: penalty 001 +1.39% (sd 0.38), 111 +2.63% (sd 0.74) vs 000 +0.23%, 100 +0.21%. Near-threshold share (|abs(w)/s-0.5|<0.05) 10.5% (001), 14.1% (111) vs 6.0% (fp cells); code flips per draw 4.1% / 5.5% vs 2.4%. Files: `experiments/checks/sq365_ternary_jitter.py`, `results/sq365.json`, `experiments/devlogs/SQ365.md`.
