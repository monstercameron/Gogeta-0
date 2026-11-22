## The project in one paragraph

> We're trying to build a **small AI model that writes Python code** while using as little memory and computation as possible. To do that we're combining three tricks researchers have found separately: **cheaper "memory" layers** instead of expensive attention, **reusing the same layers in a loop** to get more thinking out of fewer parameters, and **storing numbers as just −1, 0 or +1**. Each experiment adds one trick at a time and measures what it changes. Everything runs on one laptop.

## How a model like this learns

1. **Text becomes tokens.** Code is chopped into small pieces called *tokens*: a word, part of a word, a bracket, a few spaces of indentation. Our tokenizer has 8,192 different tokens, and one token is about 3.7 characters of Python on average.
2. **The model guesses the next token.** Given some code, it outputs a probability for every one of the 8,192 possible next tokens.
3. **Training nudges it to guess better.** We show it real code, check how much probability it gave to the token that actually came next, and adjust its numbers a tiny bit so it would do better next time. Then we repeat **hundreds of millions of times**.
4. **That's it.** Everything the model "knows" about Python (syntax, common patterns, what a palindrome checker looks like) comes out of getting better at that one guessing game.

## The numbers on our pages, decoded

| Term | What it means | Our numbers |
|---|---|---|
| **Parameters** | The adjustable numbers inside the model, the "knobs" training turns. More parameters usually means more capacity. | ~12–14 million (large commercial chat models are thousands of times bigger) |
| **Tokens (training)** | How much text the model read while learning. | 200 million per experiment, about 750 MB of Python |
| **Loss** | How surprised the model is by the real next token, averaged over a lot of text. **Lower is better.** 0 would mean it's always certain and always right. | 0.585–0.619 |
| **Validation loss** | Loss measured on code the model **never saw during training**. This is the honest score, because it can't just be memorised. | That's the number we compare |
| **Perplexity** | Loss turned into "how many equally likely options it's choosing between". Perplexity = e raised to the power of the loss. | ≈ 1.8: like choosing between fewer than 2 options per token |
| **Bits per character** | Another way to say the same thing: how many bits of information each character costs the model. | ≈ 0.23. Plain text uses 8 bits per character, so the model "compresses" code to ~3% (the `xz` compressor manages 15%, `gzip` 25%) |
| **FLOPs** | The number of arithmetic operations (multiplies and adds) training took. It's the fairest measure of cost, independent of how fast the hardware is. | 1.8–3.9 × 10¹⁶ per experiment |
| **Seed** | The starting point of the randomness (initial numbers, data order). Running again with a different seed tells you how much results wiggle by chance. | So far one seed per experiment; replication is pending |
| **Prediction** | What we expected *before* running. Writing it down first stops us from explaining any result after the fact. | Shown in a purple box on every experiment page |

**How big is a difference?** On this project, a 1% change in validation loss is small but can be real; 3–6% is a clear effect. With only one seed per experiment, we treat everything as *probable*, not *proven*.

## The three tricks, with analogies

### 1. Cheaper memory layers (instead of attention everywhere)
Standard **attention** is like rereading the *entire* document every time you write a word. It's very thorough, but the cost grows with the *square* of the document's length: a 10× longer document costs 100× more.
**Gated DeltaNet** keeps a fixed-size set of **running notes** instead, updating them as it reads. **Sliding-window attention** only glances at the **last few lines**. In principle both cost the same per word, however long the document gets. (Our current sliding-window code still builds a full-size mask, so it slows down on long documents until we swap in a better kernel.)

### 2. Loops
Instead of 14 different layers, keep a block of 3 and **run it several times**, like re-reading your answer before handing it in. You store fewer parameters, and in principle you can "think longer" on hard problems by looping more.

### 3. Ternary weights
Store every parameter as just **−1, 0 or +1** instead of a precise 16-bit number. The model's layers become about **7.8× smaller** to store (the whole model about **3.1×**, because the word table stays 16-bit; computed, not yet measured on a packed file). With special ternary kernels, multiplications could become additions, but our code only simulates the rounding, so it isn't faster yet. The open question is how much quality that costs.

## What we've found so far

> **1. The cheaper memory layers work, and actually beat the standard design.** Swapping 5 of 6 attention layers for running-notes and recent-lines layers made the model **3.6% better** with slightly less computation ([E1](experiments/E1.html) vs [E0](experiments/E0.html)).
>
> **2. Loops don't pay off yet.** The looped model was **2% worse** than the standard one while doing **twice the arithmetic** ([E2](experiments/E2.html)).
>
> **3. We know *why* the loop disappoints.** The looped model does best with exactly the number of passes it got used to in training (about 4), and gets *worse* with more. It learned a routine, not "more thinking = better answer". Fixing that is the next big step.
>
> **4. Low loss ≠ good code.** All models wrote a correct palindrome checker, but the one with the best score also wrote some of the least sensible code. We need a test that actually *runs* the generated code.

**What's still uncertain:** each result comes from a single run, the fair comparison for the loop ([E9](experiments/E9.html)) was interrupted, and everything so far uses short inputs (1,024 tokens), where the memory layers' biggest advantage, long documents, doesn't show yet.

## How to read an experiment page

- **The question** at the top is what the experiment tests.
- **The purple prediction box** is what we expected, written *before* running.
- **"What it means, in plain English"** is the short version, so start there.
- **Results** show the charts. Hover over a chart to see exact numbers, or open "Show the numbers" for a table.
- **Breakdown of the findings** is the detailed version, with every claim tied to a measured number.
- **Prediction scorecard** shows which predictions came true.
- **Caveats** cover what could make the result misleading.

Experiments so far: [E0 baseline](experiments/E0.html) · [E1 cheaper layers](experiments/E1.html) · [E2 loops](experiments/E2.html) · planned: [E3](experiments/E3.html) · [E4](experiments/E4.html) · [E5](experiments/E5.html) · [E6](experiments/E6.html) · [E7](experiments/E7.html) · [E8](experiments/E8.html) · [E9](experiments/E9.html)
