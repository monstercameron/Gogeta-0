# Side quests

Small exploratory experiments and research detours outside the main experiment ladder: things that aren't obvious but might help us or give us an edge. Rules: [AGENTS.md §0b](../AGENTS.md).

Statuses: `idea` · `running` · `done` · `dropped`. Ideas are cheap, so log them even if they aren't run yet.

| ID | Idea | Why it might help | Status | Outcome / what we did |
|---|---|---|---|---|
| SQ1 | Shrink or quantise the **embedding table** (smaller vocab, factorised or ternary embeddings) | Computed for E3: after ternarising the layers, the 6.3 MB embedding is ~2/3 of the model | `idea` | n/a |
| SQ2 | **Negative-eigenvalue Gated DeltaNet** (`allow_neg_eigval` in fla's kernel, β ∈ (0, 2)) | Theory says it enables state tracking (parity) that standard linear RNNs can't do, which could matter for check K5 and for loops | `idea` | n/a |
| SQ3 | **Block-sparse / flex-attention sliding window** instead of the dense mask | Makes our SWA actually O(n·w) (claim C12); needed before any long-context test | `idea` | n/a |
| SQ4 | **Functional-correctness eval**: run generated code against the held-out asserts | Loss and sample quality disagreed in E0–E2; this measures working code directly | `idea` | n/a |
| SQ5 | **Loop-count curriculum or per-loop supervision** (cheap version of E6) | E2's loss gets worse beyond the trained loop count (C7); a tiny model could show whether per-loop loss fixes it | `idea` | n/a |
| SQ6 | **Mamba-3-style complex/rotary state update** on the K5 parity task: a single-pass layer with a data-dependent RoPE on q/k vs GDN, vs GDN looped ×4 | Mamba-3 (arXiv 2603.15569) reports its complex-valued update near-perfectly solves parity/modular arithmetic in **one pass**, where Mamba-2 is at chance. If true at our scale, it challenges FusionDance's "loops add state tracking" bet and could change what the loop is for | `idea` | n/a |
| SQ7 | **Get Mamba-3 kernels running** on this GPU (state-spaces/mamba release; aarch64 + Blackwell build unknown), then a Mamba-2/Mamba-3 block kind for an E1-style swap | Needed for any real comparison of Mamba-3 vs our Gated DeltaNet choice (ledger C16, C21) | `idea` | n/a |
