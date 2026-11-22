# Research note: audio input for Gogeta-0 / FusionDance

Status: research only, nothing run. Date: 2026-11-22. Nice-to-have for after the core fusion (E0-E9) is settled.
Labels: **[verified]** = seen in a primary source or search result this session; **[recalled]** = from memory, check before citing; **[speculation]** = my inference.
Caveat: web search returned snippets, not full papers. Numbers marked [recalled] were not re-checked.

## 0. Target model, in one paragraph

~13M params total (about 11M non-embedding + 8k x 384 embeddings), prelude -> shared looped core x r -> coda, mixers = Gated DeltaNet (GDN, O(1) state per layer) + sliding-window attention (SWA, window 256) + 1 full-attention coda layer, ternary weights (BitNet b1.58), context T=1024, 8k BPE vocab, ~200M-token budgets, one laptop GPU. Two consequences drive everything below:

1. **T=1024 is a hard budget in positions.** Audio at 50 Hz fills it in 20 s; at 12.5 Hz in 82 s. The only full-attention layer is O(T^2) in memory, so audio length is capped by T unless that layer is changed to SWA or dropped for audio. GDN and SWA layers have constant cost per token, so only that one layer and the training context cap length.
2. **13M params is about the size of Whisper-tiny's encoder alone** (Whisper-tiny is ~39M total [recalled]; Moonshine-tiny is 27.1M vs 37.8M for Whisper-tiny [verified, arXiv 2509.02523 snippet]). Any approach that bolts a real audio encoder onto Gogeta makes the encoder bigger than the LM. This is the central design tension.

## 1. Approaches

### 1a. Encoder + projector (LLaVA-style)
Frozen or lightly tuned audio encoder (Whisper encoder, BEATs, AST) -> small projector (MLP / Q-Former / conv downsampler) -> embeddings spliced into the LM's input stream. Examples: SALMONN (Whisper + BEATs, 2310.13289), Qwen-Audio (Whisper encoder, 2311.07919), LTU (AST, 2305.10790) [verified from search, projector details secondary-sourced]. DuplexMamba (2502.11123) is notable: ConMamba speech encoder + adapter + Mamba LM, i.e. an SSM-LM audio model [verified].
- Pros: no need to extend the 8k vocab; continuous features keep paralinguistic info; adapter-only training is cheap.
- Cons: the encoder is the expensive part; the LM never "writes" audio (input only, which is all we want); the projector must map into a d=384 space. Whisper-tiny's encoder emits 50 Hz features and (as released) expects 30 s padded input [recalled], which is wasteful for a streaming model.

### 1b. Discrete audio tokens (codecs / SSL units)
Quantize audio to ids, append ids to the vocab, train the LM on them as ordinary tokens.
- **Acoustic codecs (RVQ, multi-codebook):** SoundStream (2107.03312), EnCodec (2210.13438; 75 Hz at 24 kHz, 8 codebooks at 6 kbps [recalled/partially verified]), DAC (2306.06546; ~86 Hz at 44.1 kHz, 9+ codebooks [recalled]; "700+ tokens/s" claim secondary-sourced [verified-ish]).
- **Single-codebook / low-rate codecs:** WavTokenizer (2408.16532): 75 or 40 tokens/s, one quantizer, speech + music + audio, ICLR 2025 [verified]. Mimi (in Moshi, 2410.00037): 12.5 Hz, 1.1 kbps, 80 ms frames, streaming, 8 codebooks (so 100 ids/s if all are kept [recalled]), first codebook distilled from WavLM [verified]. GLM-4-Voice tokenizer (2412.02612): 12.5 Hz, single codebook, 175 bps, VQ bottleneck on a Whisper encoder (ASR-supervised, so semantic) [verified].
- **Semantic/SSL units:** HuBERT k-means units (~50 Hz, 100-2000 clusters [recalled]), GSLM/textless NLP; SpeechTokenizer (2308.16692): RVQ with first layer semantic, rest acoustic [verified]. For *understanding* (ASR-like input), semantic units are the better fit than acoustic codec tokens.
- Pros: reuses the existing LM, loss and sampler unchanged; discrete ids make the "audio is just another language" idea trivial; no continuous-feature pipeline in the training loop.
- Cons: tokenizer is a separate (frozen) model; vocab grows (8k text + e.g. 1k-4k audio ids is fine at d=384, since embeddings are 3M at 8k); multi-codebook RVQ multiplies the rate.

### 1c. Interleaved speech-text LMs
SpiRit-LM (2402.05755; 7B base, HuBERT-unit speech tokens, interleaved with text at word boundaries, TACL 2025) [verified]; Moshi (2410.00037; 7B Temporal Transformer + small depth transformer, parallel user/assistant audio streams, text "inner monologue", Mimi codec, ~160-200 ms latency) [verified]; GLM-4-Voice (9B, 1T speech-text tokens) [verified]. These are training *recipes* on top of 7-9B LMs and 100B-1T tokens; useful as designs (interleaving, text-aligned tokens), not as budgets. Our "interleave" would be: `<aud> u1 u2 ... <txt> def foo ( ...`.

### 1d. Direct waveform / mel patches
Feed log-mel frames (80 bins, 100 Hz) or raw-waveform patches through a learned strided conv stem straight into the LM, trained end to end. Whisper-style stems and Moonshine's raw-waveform conv stem (strided convs, ~384x downsample, i.e. ~41.7 Hz [recalled]) are the reference. Pros: no external model, tiny stem (<1M params), streamable. Cons: from-scratch audio representation learning from a 13M code LM is hopeless without lots of paired audio; ternary conv stems are untested [speculation]. Best seen as the *front end of 1a*, not an alternative.

## 2. Token rates and what they mean for T=1024

Seconds of audio that fit in 1024 positions (leaving ~0 for text; subtract text length in practice):

| Representation | Rate (positions/s) | Audio per 1024 positions |
|---|---|---|
| Whisper encoder frames | 50 | 20 s |
| HuBERT / wav2vec2 units | 50 | 20 s (dedup of repeated units shortens, ~30-40% [recalled]) |
| EnCodec 24 kHz, 1 codebook / 8 codebooks | 75 / 600 | 13.6 s / 1.7 s |
| DAC 44 kHz, 1 codebook / 9 codebooks | ~86 / ~775 | 12 s / 1.3 s |
| WavTokenizer | 75 or 40 | 13.6 s or 25.6 s |
| Mimi, 1 codebook / 8 codebooks | 12.5 / 100 | 82 s / 10 s |
| GLM-4-Voice tokenizer | 12.5 | 82 s |
| Mel frames raw / after 4x stride / 8x stride | 100 / 25 / 12.5 | 10 s / 41 s / 82 s |
| Text for comparison (spoken English ~3 words/s, ~4 BPE tok/s) | ~3-4 | n/a |

Takeaways:
- Anything with multiple codebooks per frame is out for a T=1024 model unless codebooks are summed into one embedding per frame (as Moshi/MusicGen-style "delay"/sum tricks do) [speculation as applied here]. Use **one id per frame**.
- Target rate **12.5-25 Hz** for input-only understanding: an utterance of a code command (3-10 s) is 40-250 positions. Mimi's first (semantic) codebook or GLM-4-Voice's tokenizer (12.5 Hz) hits this natively; for mel/Whisper-style features add a 2-4x conv downsample in the projector.
- The ASR-supervised tokenizers (GLM-4-Voice) already carry mostly lexical content, which is what a code model needs; acoustic detail (speaker, music timbre) is irrelevant for voice-to-code.

### Why linear-time mixers fit audio
- Cost per token: **GDN** O(d * d_head) state update, constant memory per layer regardless of length, trivially streamable (carry the state across chunks). **SWA** O(window) per token, constant memory if a real ring-buffer kernel is used (README notes our SWA kernel currently uses a dense mask, so it is O(n^2) for now). **Full attention** O(n) per token and O(n) KV cache; it is the only component that makes audio length cost memory. Audio is long and high-rate, so this ordering matters more for audio than for code.
- Mamba-family speech evidence: Mamba in Speech (2405.12609): BiMamba beats vanilla (causal) Mamba for ASR, needs extra nonlinearity [verified]. Speech Slytherin (2407.09732): ConMamba matches or beats self-attention for ASR, but Mamba's advantage is "minor or none" at low-resolution tasks like ASR and large at high-resolution ones (separation); more efficient only above a duration threshold [verified]. Samba-ASR (2501.02832): Mamba encoder and decoder, claims SOTA on Gigaspeech/SPGISpeech with lower latency, single short preprint, unreplicated [verified-as-claimed]. UMA-Mamba (2410.00070) streaming ASR with lookahead + early termination [verified]. DuplexMamba (2502.11123) streaming duplex speech with a Mamba LM [verified]. TC-BiMamba (2602.11546) argues causal Mamba alone is weak for unified streaming/non-streaming ASR [verified from search snippet].
- Gated DeltaNet specifically: I found **no GDN benchmark on standard ASR sets (LibriSpeech)**. Adjacent: Fish Audio Drama 3 report (2610.18010) reports a GDN + full-attention hybrid for long-form speech *generation*; an independent report (OLMo-Hybrid codec LM, ~21.9M params, GDN blocks on LJ Speech codec tokens) exists; E-Gaborformer (2611.08092) is a causal ASR with delta-rule memory. All [verified to exist, contents only from snippets; read before citing]. So "GDN works for audio" is plausible but thinly evidenced.
- Caveat for us **[speculation]**: causal-only mixers (what a decoder LM needs) lose the bidirectional context that helps ASR encoders (Mamba in Speech, TC-BiMamba). Mitigation: treat the audio as a *prefix* the model reads before emitting text, with the loop (r passes) giving a second look; or run the audio prefix through the core with a bidirectional/non-causal GDN pass (extra engineering).
- Delta rule's strength (associative recall/overwrite) suits "remember keyword X from 10 s ago"; its weakness is state tracking, which the loop is meant to cover **[speculation]**.

## 3. Tiny audio models (calibration for 13M)

| Model | Params | Data | Notes / licence |
|---|---|---|---|
| Whisper tiny / base | 39M / 74M [recalled] (tiny encoder ~8M [recalled]) | 680k h weak supervision [recalled] | 30 s fixed window, 50 Hz encoder output. MIT (code + weights) [recalled] |
| Distil-Whisper | decoder cut to 2 layers; sizes from small up [recalled] | pseudo-labelled ~22k h [recalled] | 2311.00430 [recalled]; no tiny variant I can confirm |
| Moonshine tiny | 27.1M (vs 37.8M Whisper-tiny), 0.7x FLOPs [verified, 2509.02523]; ~5x less compute on 10 s clips with no WER increase vs Whisper-tiny.en [verified, 2410.15608] | open data + pseudo-labels + synthetic; hours not confirmed (I recall ~200k h [recalled]) | RoPE, variable-length input (no 30 s pad), English; MIT [verified, model card] |
| Moonshine "Flavors" | tiny, language-specialised | see 2509.02523 | tiny-edge ASR [verified] |
| Mimi codec | ~80M [recalled] | large speech corpora | streaming; weights reported CC-BY 4.0 on HF, paper page CC BY-NC-SA 4.0: check model card [verified conflict] |
| WavTokenizer | ~ 100M-class [recalled] | 8k-9k h speech/music/audio [recalled] | MIT code [recalled] |
| Small GDN codec-token LM (OLMo-Hybrid report) | 21.9M | LJ Speech (~24 h) | single-speaker toy; independent report [verified existence] |

Take-away: the smallest *competent* ASR is ~25-40M params trained on 10^5 hours. A 13M code LM cannot also learn ASR from scratch on a laptop budget. The audio front end must be **borrowed** (frozen), and the LM only needs to learn a mapping from borrowed units to its existing text/code space.

## 4. Ternary / low-bit audio models

- **Conformer ASR binarization/ternary QAT** (arXiv 2505.21245, Interspeech 2025): 2-bit and 1-bit QAT of Conformer ASR with scaled {-1,0,+1} / {-1,+1} weights [verified, snippet].
- **Whisper PTQ**: Edge-ASR benchmark of low-bit PTQ for edge ASR (2507.07877) [verified]; Quantizing Whisper-small (2511.08093) [verified]; ultra-low-bit PTQ for Whisper-large-v3 ~2.12 bits via k-means + mixed precision (Interspeech 2025) [verified]; Mobius Labs blog: at 2 and 1.58 bits Whisper needs group size 16 to stay usable [verified]. Community ternary Whisper-small (HF `armanibadboy/whisper-small-ternary`): hybrid, ~82% of params ternary, not peer-reviewed [verified].
- I found **no BitNet-b1.58-from-scratch audio model** and no ternary audio *codec* in this search; a "VibeVoice-ASR-BitNet" mention had no primary source (ignore) .
- Implication: QAT-from-scratch ternary audio is demonstrated on Conformer scale (tens of M), PTQ-ternary Whisper needs fine groups. **Keep the audio front end full precision and frozen (it is small and offline); keep the ternary LM as is.** Do not ternarize the encoder in the PoC. Whether a ternary LM can ingest continuous audio embeddings (activations are 8-bit absmax in BitNet) is untested **[speculation]**; discrete ids sidestep this entirely because they are just embedding rows.

## 5. Code-relevant audio tasks and data

Tasks that would justify audio for a *code* model **[speculation on value]**:
1. **Voice-to-code / dictated commands** ("define function add taking a and b..."): the clearest fit. Related: Speech-to-SQL (AU-Harness, 2509.08031) [verified], multilingual spoken code queries and ASR errors on code vocabulary (2601.15339) [verified], DoWhatISay spoken instruction set (2603.09881) [verified, not code-specific], CodeVaani (2511.20654) [verified, snippet only].
2. **Spoken programming recognition** (SpokenJava, Nowrin and Vertanen, Michigan Tech): transcripts on OSF (osf.io/h6nk4); **audio was not released** [verified]. Useful as text prompts to synthesize.
3. **Code-review / pair-programming audio** (meeting-style): no suitable open dataset found; privacy-sensitive [speculation].
4. **Streaming "voice cursor"** (dictate while the model edits): the use case where streaming GDN state actually pays off.
Note the core obstacle: **ASR for code is hard because of identifiers and symbols**; generic ASR transcripts garble `snake_case`, `len(x)`, `->`. A code LM that conditions on units *and* has strong Python priors could in principle fix transcripts, which is the real research hook **[speculation]**.

Datasets and licences (check each before use):
| Dataset | Size | Licence |
|---|---|---|
| LibriSpeech (OpenSLR 12) | 960 h read English, 16 kHz | CC BY 4.0 [recalled] |
| Common Voice | 20k+ h, many langs | CC0 [recalled] |
| Multilingual LibriSpeech | ~44k h English | CC BY 4.0 [recalled] |
| People's Speech | ~30k h | CC-BY / CC-BY-SA mix [recalled] |
| GigaSpeech | 10k h | mixed, some non-commercial terms [recalled]; avoid |
| SpokenJava transcripts | text only | see OSF page [unverified] |
| Synthetic: TTS of own Python docstrings/identifiers/code lines | unlimited | depends on the TTS voice licence; choose permissive-licensed TTS [speculation]. Our data licence is ODC-By 1.0 (OpenCoder), so derived audio keeps attribution |

## 6. Comparison table

"Params added" = new parameters inside/around Gogeta, not counting the frozen external model. "Fits 13M streaming?" is my judgement. Data/compute are for the proposed adaptation, not for training the external model.

| Approach | Example(s) | Params added (trainable / frozen external) | Positions per second of audio | Fits 13M streaming model? | Data and compute to adapt | Licence of the borrowed part |
|---|---|---|---|---|---|---|
| Encoder + MLP projector, Whisper-tiny encoder | SALMONN-style, Qwen-Audio | ~0.3-1M proj / ~8M frozen encoder [recalled] | 50 (25 with 2x pool) | Meh: encoder ~2/3 the LM; fixed 30 s window hurts streaming; continuous features vs ternary LM untested | LibriSpeech 960 h, ~hours on 1 GPU [speculation] | MIT [recalled] |
| Encoder + projector, Moonshine-tiny encoder | Moonshine 2410.15608 | ~0.3-1M / ~14M frozen [speculation] | ~42, variable length | Better than Whisper (no pad, RoPE); still an extra encoder as big as the LM | same | MIT [verified] |
| Single-codebook semantic tokens (ASR-derived) | GLM-4-Voice tokenizer | +1-4k embedding rows (0.4-1.5M) / ~tokenizer encoder frozen (Whisper-based, size not checked) | 12.5 | Yes on the LM side: shortest sequences, just ids | tokenize ASR set once offline, then plain LM training | check HF card; (code Apache-2.0 [recalled]) |
| Mimi codebook-0 only | Moshi 2410.00037 | +2k rows (0.8M) / ~80M frozen [recalled] | 12.5 | Yes: streaming codec, 80 ms frames, WavLM-distilled first codebook is semantic | tokenize offline, small | CC-BY 4.0 weights per HF [verified, conflicting page text] |
| WavTokenizer | 2408.16532 | +4k rows / frozen | 40 or 75 | Possible: 40 Hz variant = 25 s/1024; acoustic not semantic, so harder to read content | tokenize offline | MIT [recalled] |
| HuBERT/k-means units | GSLM, SpiRit-LM | +0.1-2k rows / ~95M frozen HuBERT-base [recalled] | 50 (dedup ~30) | Yes (units cheap once extracted); encoder is offline, but at inference it is a 95M streaming front end | tokenize offline; k-means once | check (fairseq repo licences vary) |
| Multi-codebook RVQ (EnCodec, DAC, Mimi all 8) | MusicGen/Moshi-style | +8 x embedding tables (summed) | 75-100 post-sum | No for T=1024; fine only for generation research | large | EnCodec weights CC-BY-NC [recalled]; DAC MIT [recalled] |
| Interleaved speech-text pretraining | SpiRit-LM 2402.05755, GLM-4-Voice | vocab only | 12.5-50 | Recipe yes, scale no: needs aligned speech-text at 10^10+ tokens | far beyond our budget | n/a |
| Raw wave / mel conv stem trained in-model | Moonshine-style stem | ~0.2-1M trainable | 25-42 | Architecturally yes (streamable, tiny), but unsupervised audio representation learning is infeasible at 13M | needs 10^4+ h paired data [speculation] | n/a |

## 7. Recommended path (confidence labels)

**Primary: discrete semantic tokens, one id per frame at ~12.5-25 Hz, added to the vocab, trained with an interleaved `<audio> ids <text> transcript/code` format.** Confidence: **medium (~60%)** that this is the best cost/benefit route for this repo; **high (~85%)** that it beats "add Whisper encoder + projector" on engineering cost and fit with the ternary/streaming story.

Why: (a) zero changes to the LM, loss, kernels; (b) 12.5 Hz makes 10 s command audio ~125 positions; (c) GDN's streaming state carries over; (d) ternary issue disappears (ids are embedding rows); (e) the tokenizer is offline/frozen so it costs no training compute; (f) the interleaved format is what SpiRit-LM/Moshi/GLM-4-Voice converge on.

**Fallback: frozen encoder (Moonshine-tiny or Whisper-tiny) + 2x conv pool + linear projector to d=384, input-only.** Confidence **medium-low (~40%)**: simplest to get a first nonzero WER, but fights ternary/streaming goals and makes the encoder larger than the LM.

**Not recommended now:** multi-codebook RVQ audio, audio generation, from-scratch encoder, ternary audio encoder. Confidence high (~80%) these are premature.

**Prerequisites before any audio work [speculation]:** (1) the K9 val-leakage cleanup and E1/E2 replication settled; (2) the full-attention coda layer made optional/SWA-only for audio positions, or the audio span kept inside its window; (3) real SWA kernel (not dense mask). Audio should be a **new arc after v1.0.0**, not mixed into the fusion ablations ("one ingredient at a time", AGENTS.md section 5).

### Cheapest proof-of-concept (described, not run)
Question: *can the existing 13M FusionDance LM, with ~2k extra embedding rows, learn to transcribe (and later code-ify) speech from frozen discrete units?*
1. **Tokenize once, offline, CPU/GPU:** LibriSpeech train-clean-100 (100 h, CC BY 4.0 [recalled]) with a frozen semantic tokenizer (GLM-4-Voice tokenizer at 12.5 Hz, or Mimi codebook-0 at 12.5 Hz, or HuBERT-base k-means-500 at 50 Hz with run-length dedup). 100 h x 12.5 Hz = 4.5M audio tokens; transcripts ~1M text tokens (after re-tokenizing with the 8k BPE; LibriSpeech is out of domain for our Python-only tokenizer, so expect poor BPE efficiency, say ~2x more tokens/word; **[speculation]** consider a small tokenizer extension).
2. **Model:** same E1 config; vocab 8192 -> 8192 + K (K = 1024-4096 audio ids) + 3 specials (`<audio>`, `</audio>`, `<text>`). Initialize new rows small-random. Keep T=1024 (fits 60-80 s audio + text at 12.5 Hz).
3. **Training:** Stage A (optional): audio-id-only next-unit LM, to learn unit statistics. Stage B: sequences `<audio> u_1..u_n </audio> <text> words </text>`, loss masked to the text span (ASR) -- or full-sequence loss with both directions. Start from the Python-pretrained checkpoint and from scratch as two arms. Use the chosen config + seed per AGENTS.md section 5.
4. **Control (matched compute, one ingredient at a time):** same data and tokens for E0 (6x attention) so the claim is about *GDN vs attention on audio positions*, per ledger style. Also a shuffle control (audio ids permuted within batch) to prove the model uses the audio.
5. **Metric:** WER on dev-clean (and a few hundred synthetic spoken-Python lines, TTS'd locally, as a code-flavoured probe). Success bar: WER clearly below the shuffle control and below a unigram-prior baseline. Expect high WER (tens of %) at this scale; the point is a nonzero, attributable signal, and the code-probe WER delta with vs without Python pretraining **[speculation]**.
6. **Budget guess [speculation]:** 4.5M audio + ~1-2M text tokens per epoch is ~1% of a 200M-token run; tens of epochs is a few hours on the target GPU. Tokenization is the dominant one-off cost (hours of encoder inference on 100 h, 5-10x faster than real time on a laptop GPU [speculation]).

Variant PoC (if discrete tokens fail): precompute Moonshine-tiny or Whisper-tiny encoder features, 2x average-pool, train only a linear projector + LoRA-free LM with frozen core for stage 1. Cheaper to debug (no vocab change) but needs storing features (100 h x 25 Hz x 384 x 2 bytes ~ 7 GB fp16 [speculation]).

## 8. Risks

1. **Capacity:** 11M non-embedding params must serve Python and audio units; interference/forgetting. Audio positions may crowd out code capacity (loss on a mixed run could regress the code val loss). Mitigate with mixture ratio ablation and with two-stage (freeze core, train embeddings/LoRA-style first).
2. **Causal-only is a handicap for ASR** (BiMamba > Mamba, 2405.12609, TC-BiMamba 2602.11546). Look-ahead isn't free in a decoder LM.
3. **Tokenizer mismatch:** semantic tokenizers trained on speech in large models; at 12.5 Hz some code-relevant detail (spelling letters, symbols) may be lost. Spoken identifiers are inherently ambiguous ("dunder init").
4. **Ternary x continuous features:** BitNet activations are 8-bit; audio embeddings have different statistics. Untested **[speculation]**. Discrete ids avoid this.
5. **Context budget:** T=1024 and a full-attention coda layer cap audio length and cost O(T^2) memory at train time; streaming beyond T needs state-carry across chunks, which the training loop does not do today (and SWA/looped-state carry is unspecified).
6. **Dependence on a big frozen tokenizer:** the "13M model" at inference needs an 80-100M+ front end, which weakens the "tiny" story. Honest framing: tiny *LM*, borrowed front end.
7. **Data licences and privacy:** avoid non-commercial weights (EnCodec CC-BY-NC, GigaSpeech terms) [recalled]; Mimi's licence shows conflicting information (CC BY-NC-SA on arXiv page vs CC-BY 4.0 on HF) [verified conflict]; SpokenJava has no audio; synthetic TTS voice licences vary.
8. **Evaluation validity:** read-speech LibriSpeech is not spoken code; WER on it says little about voice-to-code. We would need our own small spoken-Python set (TTS-generated or self-recorded). Also repeat K9: dedupe synthetic spoken prompts from train.
9. **Evidence base is thin for GDN-on-audio:** no LibriSpeech GDN result found. This could simply fail.
10. **Scope creep:** it competes with E3-E9 for GPU time and the single-ingredient discipline.

## 9. Papers to add to `refs/fetch_refs.py` (arXiv id: one line)

Core approaches:
- 2210.13438: EnCodec, high-fidelity neural audio codec, RVQ, 75 Hz at 24 kHz.
- 2107.03312: SoundStream, the original neural audio codec with RVQ.
- 2306.06546: DAC (Descript Audio Codec), high-fidelity 44.1 kHz codec, ~8 kbps.
- 2308.16692: SpeechTokenizer, semantic + acoustic RVQ hierarchy for speech LMs.
- 2408.16532: WavTokenizer, single-codebook codec at 40/75 tokens/s.
- 2410.00037: Moshi + Mimi, streaming 12.5 Hz codec and full-duplex speech-text LM.
- 2402.05755: SpiRit-LM, interleaved speech-text LM with HuBERT units.
- 2412.02612: GLM-4-Voice, 12.5 Hz single-codebook ASR-derived tokenizer.
- 2310.13289: SALMONN, Whisper + BEATs audio LLM.
- 2311.07919: Qwen-Audio, Whisper-encoder audio LLM with multitask tags.
- 2305.10790: LTU, Listen Think and Understand, AST-based audio LLM.
Tiny / linear-time audio:
- 2410.15608: Moonshine, tiny variable-length-input ASR (27M).
- 2509.02523: Flavors of Moonshine, tiny specialised ASR models, param/FLOP comparison to Whisper-tiny.
- 2407.09732: Speech Slytherin, Mamba for separation/ASR/TTS, honest efficiency study.
- 2405.12609: Mamba in Speech, BiMamba vs vanilla Mamba for ASR.
- 2501.02832: Samba-ASR, Mamba encoder-decoder ASR (single preprint, treat as claim).
- 2410.00070: UMA-Mamba, streaming ASR with Mamba plus lookahead.
- 2502.11123: DuplexMamba, ConMamba encoder + adapter + Mamba LM, streaming duplex speech.
- 2602.11546: TC-BiMamba, unified streaming/non-streaming Mamba ASR (verify).
- 2610.18010: Fish Audio Drama 3 report, GDN + attention hybrid for long-form speech (verify contents).
- 2611.08092: E-Gaborformer, strictly causal ASR with delta-rule memory (verify contents).
Low-bit:
- 2505.21245: 1-/2-bit (binary/ternary) Conformer ASR QAT, Interspeech 2025.
- 2507.07877: Edge-ASR, low-bit PTQ benchmark for edge ASR incl. Whisper.
- 2511.08093: Quantizing Whisper-small, design choices vs WER.
Code-audio tasks/data:
- 2509.08031: AU-Harness, audio-LLM eval toolkit with a Speech-to-SQL task.
- 2601.15339: spoken code queries, ASR errors hurt code understanding (verify).
- 2511.20654: CodeVaani, voice-based multilingual code learning assistant (verify).
Not in this list but worth a later look: Whisper (2212.04356 [recalled]), Distil-Whisper (2311.00430 [recalled]), HuBERT (2106.07447 [recalled]), GSLM (2102.01192 [recalled]), BEATs (2212.09058 [recalled]), BitNet b1.58 (already in refs). Verify these ids before adding.

## 10. Source URLs seen this session
- Moonshine: https://arxiv.org/abs/2410.15608 ; https://huggingface.co/UsefulSensors/moonshine-tiny ; https://arxiv.org/abs/2509.02523
- Moshi/Mimi: https://arxiv.org/abs/2410.00037 ; WavTokenizer: https://arxiv.org/abs/2408.16532 ; SpeechTokenizer: https://arxiv.org/abs/2308.16692
- SpiRit-LM: https://arxiv.org/abs/2402.05755 ; GLM-4-Voice: https://arxiv.org/abs/2412.02612 ; SALMONN: https://arxiv.org/abs/2310.13289 ; Qwen-Audio: https://arxiv.org/abs/2311.07919
- Samba-ASR: https://arxiv.org/abs/2501.02832 ; UMA-Mamba: https://arxiv.org/abs/2410.00070 ; DuplexMamba: https://arxiv.org/abs/2502.11123 ; Speech Slytherin: https://arxiv.org/abs/2407.09732 ; Mamba in Speech: https://arxiv.org/abs/2405.12609
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Low-bit: https://arxiv.org/abs/2505.21245 ; https://arxiv.org/abs/2507.07877 ; https://arxiv.org/abs/2511.08093
- Spoken code: https://osf.io/h6nk4 (SpokenJava transcripts); https://arxiv.org/abs/2509.08031
