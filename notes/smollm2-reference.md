# SmolLM2-135M: external sanity reference (setup note)

- Repo id: `HuggingFaceTB/SmolLM2-135M` (base model, public, no token used)
- Resolved revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Local snapshot (WSL, outside the repo): `~/fusiondance-data/hf-cache/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- License (model card): **Apache-2.0**

## Files (exact bytes)

| File | Bytes |
|---|---|
| config.json | 704 |
| generation_config.json | 111 |
| merges.txt | 466391 |
| model.safetensors | 269060552 |
| special_tokens_map.json | 831 |
| tokenizer.json | 2104556 |
| tokenizer_config.json | 3658 |
| vocab.json | 800662 |

Total: 272,437,465 bytes (~272 MB), plus `README.md` (model card, fetched for the license).

`model.safetensors` sha256 = `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
It **matches** the Hub LFS sha256 (`HfApi().model_info(files_metadata=True)`).

## CPU load test (fp32, CUDA hidden, 2 threads, via `tools/launch_cpu_job.sh --name smollm2-setup`)

- Class `LlamaForCausalLM`; **134,515,008 params** (tied embeddings counted once).
- hidden 576, 30 layers, 9 heads (3 KV heads), intermediate 1536, vocab 49152, max positions 8192, `tie_word_embeddings=True`.
- Loss on a 5-line snippet (`add_squares`, 29 tokens, 92 chars): 1.389 nats/token = 0.610 bits/char (a single tiny sample: only shows it runs).
- Greedy, 20 new tokens from `def fibonacci(n):`:
  `'def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return'`
- Whole test took ~6.5 s. The formal evaluation (check K8) was NOT run here.

## Caveats as a reference

- About 10x our parameter count (~135M vs our model), so it is a ceiling-ish sanity anchor, not a matched baseline.
- Different training data (general web/educational mix, not our OpenCoder data), so it is not a controlled comparison.
- Possible contamination: its corpus may overlap with our OpenCoder data, and our eval text could have been seen in training.
- Different tokenizer (49152 BPE vocab vs ours): per-token loss is not comparable. **Compare in bits per character only.**
