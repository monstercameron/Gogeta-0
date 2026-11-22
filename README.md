# FusionDance

**Three efficiency tricks, one tiny language model. Can they fuse into something stronger than any of them alone?**

> Status: **v0.1.0, design phase.** Experimental research repo. Expect sharp edges, failed runs, and honest write-ups of both.

---

## The pitch

Modern LLMs are expensive along three separate axes:

1. **Context:** attention cost grows *quadratically* with sequence length, and the KV cache grows with every token.
2. **Parameters:** more reasoning depth usually means more layers, which means more weights.
3. **Bits:** every one of those weights is stored and moved in 16-bit precision.

The research community has a strong answer to *each* of these, on its own:

| Fighter | Attacks | Idea |
|---|---|---|
| **Linear-time mixers** (Gated DeltaNet, Mamba) + sliding-window attention | Context cost | A fixed-size recurrent state instead of an ever-growing cache. Linear time, constant memory. |
| **Looped transformers** | Parameter count | Apply one shared block *r* times. Depth comes from compute, not weights, and you can think *longer* on harder inputs. |
| **BitNet b1.58** | Bits per weight | Ternary weights {-1, 0, +1}: ~1.6 bits each, and matrix multiplies become additions. |

**FusionDance asks what happens when you fuse all three.**

They should stack instead of fighting. Looping costs extra compute, and ternary weights make compute cheap. Linear-time mixers are weak at multi-step state tracking, and looping adds exactly that depth. Each fighter covers another's weak spot.

## The model (v0 sketch)

```
tokens → Embed
       → Prelude    sliding-window attention
       → Core × r   [Gated DeltaNet → Gated DeltaNet → sliding-window attn]
                    one shared block, looped; input re-injected every pass
       → Coda       a single full-attention layer, run once
       → next token
```

The single design rule: **nothing quadratic lives inside the loop.** Anything inside the loop is paid *r* times. So the only quadratic, cache-growing layer sits outside it and runs once.

## The ideas worth testing

- **Think in loops, not out loud.** Chain-of-thought reasons by emitting tokens, and every token grows memory and attention cost. Latent loops reason *silently*: compute, but no tokens. How much chain-of-thought can loops replace?
- **The Rival Loop.** Agent workflows improve code with a critique → fix cycle. We build that rivalry *into the recurrence*: every loop is scored, a critic head grounded in the true loss decides when to stop and tells the next pass what's wrong, and adversarial damage-and-repair training teaches loops to fix their own mistakes.
- **Loops that pay for themselves at inference.** A looped model can draft for itself: run 1–2 loops to guess several tokens, verify with the full loop count. Self-speculative decoding with no extra drafter model.
- **Scaling shape over raw numbers.** The headline isn't "X MB". It's *how* memory and compute grow with context, and how much quality you get per bit and per FLOP.

## Why this is a laptop project

Every experiment here is designed to run locally: small models (10M–50M parameters), clean one-ingredient-at-a-time ablations, synthetic reasoning probes plus small real-text corpora. If the fusion works at small scale, the scaling story is the interesting part. If it fails, we want to know exactly which ingredient broke it.

## Roadmap

0. Baseline: tiny dense transformer
1. \+ linear-time mixers (hybrid with sliding-window attention)
2. \+ loops (shared core block, random loop counts in training)
3. \+ ternary weights
4. **Full fusion** → `v1.0.0`
5. Adaptive loop count per token: "thinking longer" on hard tokens
6. The Rival Loop, self-speculative decoding, loops vs chain-of-thought

## Repo map

| Path | What |
|---|---|
| [`DEVLOG.md`](DEVLOG.md) | Versioned history: goals, results, decisions. Every entry is a tagged release. |
| [`sketch.md`](sketch.md) | The design scratchpad: architecture, parts bin, feasibility, decisions log |
| [`refs/`](refs/) | Reading list (~55 papers) with an index; `python refs/fetch_refs.py` downloads the PDFs |
| [`AGENTS.md`](AGENTS.md) | Working rules for humans and AI agents in this repo |

## Standing on the shoulders of

BitNet b1.58, Mamba / Mamba-2, Gated DeltaNet, Samba, Zamba, Universal Transformers, Recurrent Depth (Huginn), Mixture-of-Recursions, TRM / HRM, LayerSkip, DFlash, and more. Full list in [`refs/README.md`](refs/README.md). The closest prior work we know of (a TRM-style recursive Mamba-2/attention hybrid) is flagged there too. FusionDance's bet is the *full* fusion, ternary included, aimed at a general causal LLM.

---

*Named for a certain two-person fusion dance. Two fighters become one stronger one, but only if the dance is perfectly in sync. Get it wrong and things go badly. Same with architectures.*
