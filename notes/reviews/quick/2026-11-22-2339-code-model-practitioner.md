# Quick review 2026-11-22-2339 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The headline E1 comparison is refuted under the registered clean80 metric: E1 was 0.18% worse than E10. That leaves no verified evidence here that the hybrid stack improves practical code generation (C5; `experiments/claims-status.md`).
- The measured validation split has substantial near-duplicate overlap with training, and benchmark material appears in the corpus. Scores need clean, contamination-aware evaluation before they can guide code-model choices (C103, C191; `experiments/claims-status.md`).
- The project has a queued gaming-GPU target, but decode caching and generation are still planned work. Practical completion quality and latency are therefore still open questions (`DEVLOG.md`; SQ286, SQ288 in `experiments/side-quests.md`).

## Ideas

- **Whitespace robustness probe [speculation]:** Hypothesis: small code models may change predictions sharply under semantically neutral formatting changes. Cheapest test: about 3 GPU-minutes at 13M scale; score next-token agreement and syntax validity on paired, locally available Python snippets with normalized versus perturbed indentation and spacing. If agreement drops over 10 percentage points or syntax validity falls, add formatting robustness as a model-selection metric.
- **Identifier-use probe [speculation]:** Hypothesis: clean80 bpc can hide failures to consistently reuse names across a short function. Cheapest test: about 5 GPU-minutes at 13M scale; teacher-force locally available, repo-disjoint snippets and compare identifier-token loss with other code tokens, stratified by identifier reuse distance. If reused identifiers have over 20% higher loss, prioritize a small controlled name-reuse benchmark before architecture claims.
- **Patch-completion pilot [speculation]:** Hypothesis: continuation scores alone may not predict whether the model can make a localized, executable repair. Cheapest test: about 10 GPU-minutes at 13M scale, using existing local code and a small set of seeded syntax or type-error mutations; generate completions and report repair rate plus parse rate. If parse rate is below 50% or repairs rarely pass the available checks, treat task-level repair as a separate capability gate for the final model.

## Question

- Which local checkpoint and inference path can currently generate completions for a small practitioner-facing evaluation?

## Triage
- Whitespace robustness probe: rejected as a duplicate of SQ194 (formatting robustness, already accepted) and SQ187 (tab/space perturbations for byte vs BPE).
- Identifier-use probe: merged into SQ209 (exact-span eval on identifiers, seen-in-context vs novel); the reuse-distance stratification is added there. SQ254 supplies the identifier token labels.
- Patch-completion pilot: accepted as SQ300, gated on SQ286 (`generate()`); at 13M it will likely read ~0, so it targets the M3 pilot and E61 (goal 6).
- Question (which checkpoint and path can generate today): any trained checkpoint can sample via the training script's sample path (full re-forward per token, no cache; `samples.txt` in each run dir). A cached `generate()` is SQ286, M1 of notes/final-model-plan.md.
