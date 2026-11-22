# Proposed claims (tokenizer), C210-C219
Propositions only (ledger is being redesigned). Type: M = measured here, C = computed, R = reported by a paper (unverified), P = prediction to test.

| id | Proposition | Type | Evidence / status |
|---|---|---|---|
| C210 | Our shipped 8k BPE gets 3.733 chars/token (3.811 on clean val) vs SmolLM2's 3.461 (3.573) on code-py-v1 val | M | `experiments/tokenizer/results/audit.json` (K8 reports 3.457 for SmolLM2) |
| C211 | The shipped tokenizer round-trips every val doc exactly and has no unk (byte-level, 256 bytes in alphabet) | M | audit.json roundtrip 4000/4000 |
| C212 | 9% of tokens are whitespace-only tokens (8.9% newline+indent), covering 13.6% of chars | M | audit.json |
| C213 | 206 of 8,190 vocab entries are never used in 30M train tokens; the rarest 10% carry 0.029% of tokens | M | audit.json vocab_usage_train30M (pruning would save only ~0.3M params, C) |
| C214 | 31% of the 3,000 most frequent val words tokenize into a different number of tokens at line start vs after a space; 1,578 words have both forms (19% of vocab) | M | audit.json leading_space |
| C215 | Doubling vocab 8k to 16k gives only +1.2% chars/token (sample-limited lower bound), 4k costs -3.1%; individual-digit splitting costs -5.1% | M | variants.json (20 MB train sample) |
| C216 | A line-level pre-tokenizer (superword proxy) gives +67% chars/token at 8k and +99% at 16k (clean val +57%/+84%) | M | variants.json; inflated by corpus self-duplication (K9) |
| C217 | A 16k vocab does not improve clean-val bits/char of E1 by >=1.5% at matched non-embedding params and matched characters | P | experiment in research-tokenizer.md §7, not run |
| C218 | The line-level superword tokenizer is worse on clean-val bits/char than the 8k baseline but better on duplicated val (the gain is memorization) | P | not run |
| C219 | Reported by others: SuperBPE gives up to -33% tokens and +4.0 pts on 30 tasks at 8B (2503.13423); vocab optimum grows with compute (2407.13623); ~10% of BPE vocab is residue (2602.04706). None is verified at our scale | R | papers listed in research-tokenizer.md §4 |
