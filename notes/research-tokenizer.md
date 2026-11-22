# Research: the tokenizer (audit, alternatives, fit, cheapest experiment)

Status: research + cheap CPU measurement only. Nothing here was trained on GPU. Labels follow AGENTS.md §0: **measured** (output on disk), **computed**, **reported by <paper>** (not verified here), **predicted**.
Scripts: `experiments/tokenizer/audit.py`, `experiments/tokenizer/train_variants.py`, `experiments/tokenizer/run.sh`. Outputs: `experiments/tokenizer/results/audit.json`, `variants.json`. Total CPU about 10 min, 2 threads, `CUDA_VISIBLE_DEVICES=` (empty), peak RAM well under 2 GB (not separately logged).

## 1. What we ship (read from `tools/prepare_data.py` and the asset)
- Byte-level BPE, 8192 vocab incl. `<|eos|>`=0, `<|pad|>`=1, trained with HF `tokenizers` on the **entire train split** (976M chars), `min_frequency=2`, `ByteLevel(add_prefix_space=False)` with the default **GPT-2 regex** pre-tokenizer (use_regex=True), no normalizer, no unk token (all 256 bytes are in the alphabet, so nothing is ever unknown).
- Pre-tokenization consequences of the GPT-2 regex: digits are **not** split (a number is one pre-token); whitespace runs are one pre-token except the last space, which sticks to the next word (`\n        x` becomes `"\n       "` + `" x"`); merges never cross a pre-token boundary, so no multi-word tokens.
- Tied embedding/head: 8192 x 384 = 3.146M params of E0's 13.768M total (22.8%); 10.622M non-embedding (computed, `gogeta/model.py` param_counts; matches C90). The embedding is **not** ternary (`gogeta/model.py` docstring: embeddings, norms, gate projections, convs stay full precision; BitLinear covers the projections only).

## 2. Audit of our tokenizer (measured, 4,000 val docs; "clean" = 2,333 docs after removing K9's 1,667 near-duplicates)
| Metric | ours 8k, all val | ours 8k, clean val | SmolLM2 49k, all | SmolLM2 49k, clean |
|---|---|---|---|---|
| chars/token | **3.733** (dataset meta 3.719) | **3.811** | 3.461 (K8: 3.457) | 3.573 |
| bytes/token | 3.735 | 3.813 | 3.463 | 3.576 |
| whitespace-only tokens, share of tokens | 9.0% | 8.8% | 12.2% | 11.7% |
| ... of which contain a newline | 8.9% of all tokens | 8.7% | 8.3% | 8.2% |
| chars covered by whitespace-only tokens | 13.6% of text | 13.7% | 14.7% | 14.7% |
| tokens per `snake_case` identifier | 3.88 | 3.91 | 4.21 | 4.16 |
| tokens per `camelCase` identifier | 2.24 | 2.47 | 2.82 | 2.81 |
| tokens per single-word identifier | 1.03 | 1.04 | 1.05 | 1.05 |
| tokens per number, 1 / 2 / 3 / 4 / 6+ digits | 1.0 / 1.01 / 1.20 / 1.45 / 2.56 | similar | 1.0 / 2 / 3 / 4 / 7.9 (digits split) | similar |
| non-ASCII chars (share of text / tokens per char) | 0.05% / 2.12 | | 0.05% / 1.98 | |
| exact round-trip decode(encode(doc)) | 4000/4000 | 2333/2333 | 4000/4000 | |

Findings:
1. **Our 8k tokenizer is already more compact than SmolLM2's 49k on this corpus** (+7.9% chars/token on all val, +6.6% clean) because it is in-domain; the gap to a general vocab is not a bottleneck.
2. **Whitespace/indentation.** About one token in eleven is a newline+indent token (the regex glues "\n" and all but one of the indent spaces into one token; `_` the last space is carried by the next word). The indent costs about 1 token per line regardless of depth (4-20 spaces all about 1.9 "span tokens", i.e. 1 ws token + the shared word token). So indentation is already cheap; ~9% of tokens are line breaks and they carry little information after the first few tokens of a line (they are predictable from syntax). That is the target SuperBPE/line-level approaches exploit (section 4).
3. **Identifier splitting.** Single-word identifiers are essentially one token (1.03). `snake_case` costs 3.9 tokens (about words + underscores; `_` often separate). camelCase 2.2-2.5. These are reasonable; no obvious pathology.
4. **Numbers.** 1-2 digit numbers are 1 token, 3+ digits start fragmenting irregularly (3 digits 1.2, 6+ digits 2.6). Digits are merged by frequency, not by place value; for Python code (small constants, indices) this is a minor issue, for arithmetic evals it would matter (Llama/SmolLM-style digit splitting is the usual fix; our measured cost of forcing digit splitting is -5.1% chars/token at 8k, section 3).
5. **Vocab usage (30M train tokens, measured).** 206 entries (2.5%) are **never** used in 30M tokens; 773 (9.4%) fire <1 per million tokens; 4,682 (57%) fire <10 per million. The rarest 10% (819 entries) account for **0.029%** of tokens (max count 32 in 30M). Example rarest entries are Latin-1 accented single characters (`À Á Æ ...`). The top 100 entries are 63.4% of tokens, top 1,000 91.4%. On val, 1,327 entries (16%) never appear. By the LiteToken definition (arXiv 2602.04706, reported: ~10% of tokens in popular tokenizers are "intermediate merge residue") we would have a comparable ~10% dead tail, but at our scale that is only 819 x 384 = 0.31M params (2.3% of the model), so pruning is a low-value move *for us*, unlike for 100k+ vocabularies.
6. **Single-character entries carry 25.2% of tokens** (punctuation, brackets, operators): 25% of token positions are single characters, consistent with Python's symbol density.
7. **Byte fallback / unknown.** No unk exists; all 4,000 docs round-trip exactly; non-ASCII is 0.05% of chars and costs ~2.1 tokens/char (UTF-8 bytes merged only for frequent accents). Safe.
8. **Leading-space consistency.** 3,553 alphabetic entries without space, 3,357 with a leading `Ġ`; only 1,578 words have **both** forms (so ~38% of word entries are space-form only and ~56% no-space-form only, partly sub-word pieces). The space forms hold 35.3% of token mass vs 10.8% for the no-space twins of the same words. Of the 3,000 most frequent val words, **922 (31%) take a different number of tokens at line start than after a space** (e.g. `else`, `while`, `through`, `returns` 1 token with a space, 2 without; `Initialize` 1 vs 3). It is the standard GPT-2 behaviour (the same word has two unrelated embeddings), costs vocab slots (1,578 duplicated forms = 19% of the vocab) and makes the first word of a line (statement keyword) systematically more fragmented. That is the best-evidenced inefficiency we found. Mitigations: SentencePiece-style `▁` prefix with a single form, or a pre-tokenizer that attaches the preceding whitespace as a feature (not tested).

## 3. Candidates trained on our corpus (measured; `variants.json`)
Setup: `tokenizers` 0.23.3 (installed; SentencePiece is **not** installed, so Unigram/WordPiece were trained with `tokenizers`' own trainers, same byte-level alphabet and regex). Training sample = every 50th train doc (about 20 MB, 19.8k docs); "small" = every 150th (about 6.6 MB). Scores on the same val. Embedding params computed for tied d=384.

| Variant (trained on 20 MB unless noted) | vocab | chars/tok all val | chars/tok clean | embed params (tied, d=384) | vocab entries unused on val |
|---|---|---|---|---|---|
| BPE regex, shipped (full 976 MB train) | 8192 | 3.733 | 3.811 | 3.15M (22.8% of E0) | 1,327 |
| BPE regex, 20 MB sample (control for sample size) | 8192 | 3.727 | 3.803 | 3.15M | 1,430 |
| BPE regex 4k | 4096 | 3.610 (-3.1% vs 8k) | 3.672 | 1.57M | 285 |
| BPE regex 16k | 16384 | 3.771 (+1.2%) | 3.857 | 6.29M | 7,178 |
| BPE regex "32k" (ran out of merges at 24,152; min_frequency 2 on 20 MB) | 24152 | 3.783 (+1.5%) | 3.872 | 9.27M | 14,261 |
| BPE regex + individual-digit split, 8k | 8192 | 3.536 (-5.1%) | 3.629 | 3.15M | 1,575 |
| BPE "indentmerge" pre-tokenizer (my regex; flawed: `[^\s]+` glues punctuation to words) | 8192 | 3.154 | 3.091 | 3.15M | 546 |
| BPE **line-level pre-tokenizer** (merges may cross spaces inside a line; superword-style approximation, not SuperBPE proper) | 8192 | **6.248 (+67%)** | 6.002 | 3.15M | 241 |
| BPE line-level, 16k | 16384 | **7.435 (+99%)** | 6.997 | 6.29M | 864 |
| 6.6 MB sample: BPE / Unigram / WordPiece, 8k | 8192 | 3.717 / 2.861 / 3.702 | 3.792 / 2.914 / 3.773 | 3.15M | 1,591 / 3,126 / 1,608 |

Reading it:
- **Vocab size is a weak lever for compression here**: 4k to 8k +3.3%, 8k to 16k +1.2%, beyond that the 20 MB sample saturates (the 16k/24k rows are sample-limited: a full-corpus 16k would be somewhat better and have fewer unused entries; their numbers are lower bounds). Each doubling costs +3.15M / +6.3M embedding params (+23% / +46% of E0) and doubles head FLOPs (computed: head = 2·V·d per token forward; at V=8192 it is ~19% of E0's 96.8M train FLOPs/token, ~37% at 16k).
- **BPE vs Unigram vs WordPiece**: at matched sample and vocab, WordPiece is -0.4% and Unigram is -23% (2.86 vs 3.72). The Unigram figure is a weakness of `tokenizers`' trainer under byte-level pre-tokenization and `max_piece_length=16` with a tiny sample; it should not be read as "Unigram is 23% worse in general" (Kudo reports Unigram is on par with BPE; arXiv 1804.10959, reported). Practical conclusion: BPE stays; no evidence for switching (measured), but a proper SentencePiece run was not possible (not installed).
- **Digit splitting costs 5.1% chars/token** on this corpus, in exchange for place-value-regular numbers; relevant only if we ever eval arithmetic.
- **The superword / line-level result is by far the largest effect (+67% chars/token at 8k, +99% at 16k).** Caveats that must be understood before believing it: (a) the corpus is self-duplicated (K9: 41.7% of val docs are >=80% 13-gram near-duplicates of train, and the clean-val gain is smaller but still +57%/+84%); repeated lines become single tokens; (b) a token that is a whole line `return a + b` is a *memorized unit*: fewer tokens does not mean less information per character, only that the information moves into the prediction of fewer, higher-entropy tokens; (c) 9.7% of the 8k line-vocab fires <10 times on val, and any such vocab is heavily dependent on the train distribution (brittle OOD). It is the same idea as SuperBPE (reported: -33% tokens at 200k vocab, +4.0 pts on 30 tasks at 8B, arXiv 2503.13423), but SuperBPE has two stages (subword merges first, superwords second), so my line-level BPE is an extreme version, not a replication.
- Compression is **not** a loss-quality measure; token count falling 40-50% will raise nats/token, so only bits/char on a fixed val (the K8 protocol) can compare.

## 4. Literature (primary sources; numbers are "reported by", not verified here)
| Topic | Source (arXiv) | Reported gain / finding and scale |
|---|---|---|
| Vocab size scaling | Tao et al., Scaling Laws with Vocabulary, **2407.13623** | Optimal vocab grows with compute; 33M-3B models, 500B chars; Llama2-70B should have ~7x larger vocab; ARC-C 29.1 to 32.0 for 32K to 43K vocab at equal FLOPs. Their smallest models (33M) have optimum well below 32K, consistent with an 8k-16k choice for us. |
| Tokenizer choice for code | Dagan et al., Getting the most out of your tokenizer, **2402.01035** | Vocab size, pre-tokenization regex, and training data change HumanEval/MBPP, effective context, generation speed; retraining a tokenizer for a domain helps after >50B tokens. |
| Tokenizer choice (general) | Ali et al., Tokenizer Choice for LLM Training: Negligible or Crucial?, **2310.08754** (id from memory) | 24 1B-param models; tokenizer choice moves downstream scores and cost materially; fertility is a poor sole predictor. |
| Tokenization != compression | Schmidt et al., **2402.18376** (id from memory) | Compression alone does not predict downstream; PathPiece etc.; digit/whitespace handling matters. |
| Code LLM tokenizers | StarCoder **2305.06161**, StarCoder2 **2402.19173**, CodeLlama **2308.12950**, DeepSeek-Coder **2401.14196**, Qwen2.5-Coder **2409.12186** (ids from memory) | Byte-level BPE with 49k (StarCoder, digits split individually, whitespace-run tokens/indentation merges, `<fim_*>` and repo/file tokens), 32k (CodeLlama, SentencePiece), 32k (DeepSeek-Coder 1.3-33B), 151k (Qwen2.5-Coder, shared with Qwen2.5). Common choices: individual-digit splitting and merging long whitespace runs. None isolates the tokenizer's effect at our scale. |
| SuperBPE | Liu et al., SuperBPE: Space Travel for Language Models, **2503.13423** (fetched) | Two-stage BPE letting tokens cross whitespace; up to 33% fewer tokens at 200k vocab, +4.0 avg over 30 tasks (+8.2 MMLU), 27% less inference compute; 8B models, equal size/vocab/train compute. |
| BoundlessBPE | Schmidt et al., **2504.00178** (fetched) | Merges whole pre-tokens into "superwords"; up to +15% bytes/token and a more uniform token-frequency distribution. |
| LiteToken | Sun et al., **2602.04706** (fetched via search snippet) | ~10% of tokens in widely used BPE vocabularies are "intermediate merge residues" rarely used; removal post hoc, no retraining; fewer params, more robustness, performance preserved (I did not read its tables). |
| Scaffold-BPE | **2404.17808** (id from memory) | Removes scaffold tokens (same idea as LiteToken, during training). |
| Tokenizer transfer | Minixhofer et al., Zero-Shot Tokenizer Transfer (ZeTT), **2405.07883** (fetched) | Hypernetwork predicts embeddings for any new tokenizer; near-original performance on cross-lingual and code tasks (Mistral-7B, XLM-R) with shorter sequences; gap closed with <1B tokens continued training. Irrelevant at our scale (retraining a 14M model is cheaper than a hypernetwork), useful only if we later want to graft a code tokenizer onto a pretrained checkpoint. |
| Unigram / SentencePiece | Kudo **1804.10959**, **1808.06226** (ids from memory) | Unigram LM tokenization with subword regularization; comparable to BPE in compression. |
| Tokenizer-free | Byte Latent Transformer **2412.09871**, MambaByte **2401.13660**, H-Net dynamic chunking **2507.07955** (ids from memory) | Entropy-patched bytes match BPE Llama at 8B in compute-matched settings (reported); learned chunking removes the tokenizer. At our scale, bytes would multiply sequence length 3.7x and the linear-time mixers' state-update count with it; not a near-term option. |
| Embeddings in small models | MobileLLM **2402.14905** (id from memory) | At 125M-350M, input-output embedding sharing saves 11.8M params and is kept; small-vocab+deeper/thinner designs favoured. |
| Over-tokenized transformers | **2501.16975** (id from memory) | Larger *input* vocabularies (n-gram embeddings) decoupled from the output head give log-linear loss gains; relevant: input-vocab cost is lookup-only. |
| Digits | Singh & Strouse, Tokenization counts, **2402.14903** (id from memory) | Right-to-left/place-value digit tokenization changes arithmetic accuracy up to 20 pts. |

## 5. Fit to FusionDance
- **Ternary / tied embeddings.** BitLinear does not cover the embedding/head, so a bigger vocab adds *full-precision* parameters that dominate the storage story: at 16k, embedding 6.3M fp16 = 12.6 MB vs ternary body 10.6M x 1.58 bit ~ 2.1 MB. If the project's pitch is memory/energy at ternary, the vocabulary is the first-order cost (at 8k already: 6.3 MB embed vs ~2.1 MB body, computed, assuming fp16 embed). This argues for a *small* vocabulary or a ternary/low-bit embedding (BitNet keeps embeddings full precision, reported). Untested here.
- **Matched-parameter accounting.** Any vocab comparison must hold *non-embedding* params fixed (and report total), otherwise 16k just adds 3M params. Per C95, E1 vs E0 is already a +4.4%-params comparison; do not repeat that sloppiness.
- **Linear-time mixers and compression.** GDN state updates, FLOPs and KV-less decode scale with tokens, so +X% chars/token is +X% context and -X% compute per char. 16k gives only +1.2% (not worth 2x head FLOPs); a superword/line-level scheme gives +67% to +99% (large) but with a rare-token tail and a harder per-token prediction (see caveats). Under a fixed *seq length in tokens* the model sees up to 1.7x more characters of context per window; under fixed *bytes seen* it runs 40% fewer steps.
- **Looped depth.** Each loop re-runs the full core per token, so token-count reduction multiplies the loop savings (cost per char = loops x tokens/char). This is where superwords help most (predicted, not measured). The head/embedding costs are paid once, not per loop, so large vocab penalises looped models *less* than dense ones in parameter terms (the shared core is small, so the embedding share is even larger: E2's embedding share is higher than E0's).
- **bits/char comparability.** K8's protocol (doc-level bits/char = total nats / chars / ln 2) is tokenizer-independent by construction, so all of the above can be compared. Care: (i) a model must be scored on the same val docs and the same *clean val* (K9); (ii) the tokenizer must not be trained on val; (iii) token-level loss/perplexity are not comparable across vocabs; (iv) with merges across docs impossible, the eos accounting must be the same (count eos bits or not, for both).
- **Self-duplication interacts with the tokenizer.** Vocab and superword gains measured on a 41%-duplicated corpus are inflated relative to clean code; report clean-val numbers.

## 6. Recommendations (confidence)
1. **Keep the 8k BPE for the current campaign (high confidence).** It is in-domain, round-trips exactly, outperforms SmolLM2's 49k on chars/token, and changing it mid-campaign would invalidate every comparison. Nothing measured says it is badly wrong.
2. **Do not raise to 16k/32k for compression reasons (high confidence on compression, low on loss).** +1.2% chars/token for +23% params or 2x head FLOPs. Possible loss benefit is unmeasured; Tao et al. suggest optimal vocab at ~30M-scale compute is small.
3. **Single-form (space-agnostic) words are the best "cheap fix" candidate (medium-low confidence it matters):** 19% of vocab are duplicate space/no-space forms and line-start keywords fragment (31% of top-3000 words differ). Test only as a side quest.
4. **Superword/line-level tokenization is the high-upside direction (medium confidence it is worth testing, low confidence it wins on bits/char).** The compression is enormous, but fewer, harder tokens and corpus duplication may hide a bits/char loss. It uniquely suits linear-time and looped designs. Test before adopting.
5. **Digit splitting: skip unless we add arithmetic evals (medium).**
6. **Prune never-used tail (LiteToken-like): skip (high);** only 0.3M params.
7. **Unigram/WordPiece switch: no evidence (measured weak; my trainer is crude). Tokenizer-free/ZeTT: not at this scale (high).**

## 7. Cheapest decisive experiment (designed, **not run**)
**Question:** does a different tokenizer change clean-val bits/char of E1 at matched non-embedding params and matched bytes seen?
- **Arms** (all E1 config, d=384, same seed(s), same lr/schedule in *characters of data*):
  - A: shipped 8k (control).
  - B: 16k BPE, trained on the full train split (CPU, one pass, several min).
  - C: 8k with the line-level pre-tokenizer (superword proxy).
  - (optional D: 4k, for the trend.)
- **Matching:** non-embedding params identical (10.62M-ish; B and C therefore have different total, reported); train for the same **characters** (about 745M chars = 200M 8k-tokens), so B sees ~198M tokens, C ~119M tokens at 8k-line vocab; context length in characters matched (about 3.8k chars: 1,024 tokens for A, ~1,013 for B, ~610 for C). Report FLOPs per char (computed) too: C is cheaper per char, B is dearer.
- **Score:** doc-level bits/char, K8 protocol, **clean val** (K9, 2,333 docs) and all val; >=2 seeds if budget allows (seed noise is unknown, see C92 and statistics-policy).
- **Predictions (written before running):**
  - B vs A: bpc within -2% to +1% (not significant); I expect 16k not to beat 8k by >=1.5% at 13.8M-scale compute.
  - C vs A: bpc in -3% to +6% (uncertain direction): fewer steps (-40%) but a richer unit; I expect C to be *worse* on clean val and *better* on duplicated val, i.e. the gain is memorization.
  - D vs A: +1% to +4% worse.
- **Criteria:** adopt B only if bpc(B) <= bpc(A) x 0.985 on clean val for both seeds and FLOPs/char not worse than +25%. Adopt C only if bpc(C) <= bpc(A) on clean val (the compute saving per char then is free). Otherwise keep A. A failure of C on clean val and success on duplicated val is itself the result (memorization gain).
- Cost: 3 runs of E1's 200M-token budget (the same GPU time as E1 each, scaled by chars); tokenizer training and tokenizing are CPU.

## 8. Papers to add to `refs/` (via `refs/fetch_refs.py`, owned by someone else, so just listed)
- 2503.13423 SuperBPE (superword tokens, two-stage BPE; -33% tokens at 200k).
- 2504.00178 BoundlessBPE (superwords by merging pre-tokens; +15% bytes/token).
- 2602.04706 LiteToken (remove residue tokens, ~10% dead vocab).
- 2407.13623 Scaling Laws with Vocabulary (optimal vocab vs compute).
- 2402.01035 Getting the most out of your tokenizer (code tokenizer ablations, regex, vocab size).
- 2405.07883 Zero-Shot Tokenizer Transfer.
- 2310.08754 Tokenizer Choice for LLM Training (id from memory, verify).
- 2402.18376 Tokenization Is More Than Compression (id from memory, verify).
- 2402.19173 StarCoder 2 (tokenizer and whitespace choices; verify id).
- 2412.09871 Byte Latent Transformer; 2507.07955 H-Net (tokenizer-free; verify ids).
- 2501.16975 Over-Tokenized Transformer (verify id).
