# lm-eval 0.4.13 MBPP task files (snapshot)

These are unmodified copies of `lm_eval/tasks/mbpp/{mbpp.yaml, mbpp_plus.yaml, utils.py}`, stored with a `.snapshot` suffix so that lm-eval can never register them as tasks (the upstream `utils.py` loads HF `code_eval`, which runs code in-process) from the installed
`lm_eval==0.4.13` wheel (EleutherAI lm-evaluation-harness, MIT licence: `LICENSE.md`). They were taken from the WSL
eval venv on 2026-11-23, and their sha256 values match the wheel's `RECORD` (`SHA256SUMS`). They are archived as
primary evidence for ledger claim C109 (claim audit 2026-11-23-1037).

What they show: `mbpp_plus.yaml` includes `mbpp.yaml` and only swaps the dataset (`evalplus/mbppplus`). The target is
`doc_to_target = test_list[0..2]` (non-few-shot), and `utils.pass_at_1` scores predictions against those references.
So lm-eval's `mbpp_plus` runs MBPP's 3 base asserts, not EvalPlus's extra tests.
