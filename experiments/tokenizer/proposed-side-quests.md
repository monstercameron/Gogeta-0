# Proposed side quests (tokenizer), SQ170-SQ179
Proposals only; not registered in `experiments/side-quests.md`. Each follows AGENTS.md §0b (small, expectation first).

- **SQ170 Vocab-size trade at matched non-embedding params (the decisive experiment).** E1 at 4k/8k/16k, matched non-embedding params and matched characters, clean-val bits/char. Design in `notes/research-tokenizer.md` §7. Status: idea. GPU, not small.
- **SQ171 Line-level / superword tokenizer proxy at 8k.** Same as above arm C; also report clean vs duplicated val gap to separate memorization from real gain. Status: idea. GPU.
- **SQ172 Proper SuperBPE (two-stage) on our corpus.** Needs the SuperBPE training code (not installed, not downloaded). Train stage 1 whitespace-BPE to ~4k then continue without pre-tokenization to 8k; compare chars/token and rare-token tail to the line-level proxy. CPU. Status: idea, `needs_human` (download).
- **SQ173 Full-corpus 16k/24k/32k BPE.** Retrain on all 976 MB (single CPU job) to replace the sample-limited 16k/24k rows. CPU, ~minutes. Status: idea.
- **SQ174 Single-form space-agnostic words.** Pre-tokenizer that makes `word` and ` word` share one entry (e.g. SentencePiece-style marker or separate whitespace tokens); measure chars/token, vocab slots saved (~1.6k of 8.2k), keyword fragmentation at line start. CPU measurement first. Status: idea.
- **SQ175 Tokenizer-agnostic bits/char of a dummy: unigram/bigram token LM baselines per tokenizer** (cheap CPU) to see how much of the bits/char difference between tokenizers is explained by a count-based model (a floor for the effect size). Status: idea.
- **SQ176 Embedding precision.** Ternary / 4-bit embedding+head at 8k vs fp16 on E1 (BitNet keeps embeddings fp). Quantifies the storage story. Status: idea. GPU, small.
- **SQ177 Digit splitting cost on a model.** Cost +5.1% tokens (measured) vs any bits/char effect; only if arithmetic evals are added. Status: idea, low priority.
- **SQ178 Prune never-used tail and reuse slots (LiteToken-like).** Measure: 206 entries unused in 30M tokens; expected bits/char effect <0.5% (predicted). CPU/GPU small. Status: idea, low priority.
- **SQ179 Sequence-length-in-chars equivalence for GDN/looped cost.** Computed table of FLOPs/char for each tokenizer at fixed core, including loops (cost per char = loops x tokens/char + head). Pure arithmetic. Status: idea.
