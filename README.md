<p align="center">
  <img src="docs/assets/banner.svg" alt="Gogeta-0: the first FusionDance language model" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-ffd23f"></a>
  <a href="https://github.com/monstercameron/Gogeta-0/tags"><img alt="Version" src="https://img.shields.io/badge/version-v0.4.17-ff8a1f"></a>
  <a href="https://monstercameron.github.io/Gogeta-0/"><img alt="Project site" src="https://img.shields.io/badge/site-live-3cff9a"></a>
  <img alt="Status: research" src="https://img.shields.io/badge/status-experimental%20research-4a90ff">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3cff9a">
  <img alt="PyTorch 2.14" src="https://img.shields.io/badge/pytorch-2.14-b07cff">
  <a href="https://monstercameron.github.io/Gogeta-0/"><img alt="Site on GitHub Pages" src="https://img.shields.io/badge/site-GitHub%20Pages-ff8a1f"></a>
</p>

<p align="center">
  <b>Quick links</b><br>
  <a href="https://monstercameron.github.io/Gogeta-0/">Site</a> ·
  <a href="https://monstercameron.github.io/Gogeta-0/devlog.html">Devlog page</a> ·
  <a href="https://monstercameron.github.io/Gogeta-0/learn.html">Learn</a> ·
  <a href="https://monstercameron.github.io/Gogeta-0/architecture.html">3D architecture</a> ·
  <a href="experiments/INDEX.md">Experiment index</a> ·
  <a href="experiments/QUEUE.md">Queue</a> ·
  <a href="experiments/claims.md">Claims ledger</a> ·
  <a href="DEVLOG.md">DEVLOG.md</a> ·
  <a href="https://github.com/monstercameron/Gogeta-0/tags">Releases / tags</a>
</p>

<p align="center">
  <b>Three efficiency tricks. One tiny language model. Fused, or it falls apart.</b><br>
  <a href="https://monstercameron.github.io/Gogeta-0/"><b>Project site</b></a> ·
  <a href="https://monstercameron.github.io/Gogeta-0/learn.html">Start here (plain English)</a> ·
  <a href="DEVLOG.md">Devlog</a> ·
  <a href="experiments/experiments.json">Experiments</a> ·
  <a href="notes/LABNOTEBOOK.md">Lab notebook</a> ·
  <a href="refs/README.md">Reading list</a>
</p>

---

**Gogeta-0** is a small causal language model for Python code, and the first model built on **FusionDance**: an architecture that fuses three separate answers to the three big costs of modern LLMs. It's an open, laptop-scale research project, and every experiment is written up honestly, failures included.

> **Status: v0.4.17. First models trained.** On 200M tokens of Python, swapping five of six attention layers for linear-time layers **beat the standard transformer by 3.6%**. The looped version **doesn't pay off yet**, and we've measured why. Each result comes from a single seed so far; see [Results](#results-so-far).

## The pitch

Modern LLMs are expensive along three separate axes, and the research community has a strong answer to each one *on its own*:

| | The cost | The fighter | The idea |
|---|---|---|---|
| ⚡ | **Context**: attention cost grows with the *square* of the text length | **Linear-time mixers** (Gated DeltaNet) + sliding-window attention | A fixed-size running memory instead of re-reading everything: linear time, constant memory (in principle; our sliding-window kernel still uses a dense mask, so it is O(n²) for now) |
| 🌀 | **Parameters**: more reasoning depth means more layers | **Looped depth** | Run one shared block *r* times: depth comes from compute, not weights |
| 🪶 | **Bits**: every weight stored in 16-bit precision | **Ternary weights** (BitNet b1.58) | Every weight is −1, 0 or +1: ~1.58 bits. With dedicated ternary kernels, matrix multiplies could become additions (ours are simulated, so no speedup yet) |

**FusionDance fuses all three, and Gogeta-0 is the first model to try it.** The bet is that they cover each other's weak spots. Linear-time mixers are weak at multi-step state tracking, and looping adds exactly that depth. Looping costs compute, and ternary weights could make compute cheap once real ternary kernels replace our floating-point simulation.

## The architecture

```mermaid
flowchart LR
    T([tokens]) --> E[Embed<br/>8k vocab]
    E --> P[Prelude<br/>sliding-window attn]
    P --> G1
    subgraph CORE ["Core × r (shared weights)"]
        G1[Gated DeltaNet] --> G2[Gated DeltaNet] --> S[Sliding-window attn]
    end
    S -. loop r times .-> G1
    S --> D[Coda<br/>full attention · runs once]
    D --> N([next token])
```

**The one rule: nothing quadratic lives inside the loop.** Anything inside the loop is paid *r* times, so the only full-attention layer sits outside it and runs once. It's also the only layer whose memory grows with the length of the input.

## Goals

1. **Fuse the three:** show that linear-time mixing, looped depth and ternary weights *stack* in one causal LLM.
2. **Think in loops, not out loud:** replace part of chain-of-thought with silent latent loops, spending compute instead of tokens and memory.
3. **The Rival Loop:** build critique → fix into the recurrence, with per-loop scoring, a grounded critic head, and adversarial damage-and-repair.
4. **Make loops pay at inference:** self-speculative decoding (draft with a few loops, verify with all of them, no extra model).
5. **Do it on a laptop:** small models, one-ingredient-at-a-time ablations, and results reported as *scaling shapes*: quality per bit, per FLOP, and cost vs context length.

## Results so far

200M tokens of Python each, seed 0, full validation split. Lower loss is better.

| Experiment | Layout | Params (non-emb) | Val loss | vs E0 | Compute |
|---|---|---|---|---|---|
| [E0: dense baseline](experiments/writeups/E0.md) | 6 × full attention | 10.6M | 0.606 | n/a | 1.0× |
| [**E1: linear-time mixers**](experiments/writeups/E1.md) | 1 attention + 3 Gated DeltaNet + 2 sliding-window | 11.1M | **0.585** | **−3.6%** | 0.94× |
| [E2: looped core](experiments/writeups/E2.md) | `[SWA] → [GDN, GDN, SWA] × 4 → [attn]` | 9.5M | 0.619 | +2.0% | 2.0× |

**In plain English:**
- **The cheaper layers win.** Running notes plus a glance at recent lines beat re-reading everything, at slightly less compute. E1 matched E0's final quality after 84% of the training.
- **Loops don't pay yet.** The looped model was 2% worse at twice the cost.
- **We know why.** Tested with 1–12 loops, it does best at exactly 4, the average it was trained with, and gets *worse* with more. It learned a routine, not "more thinking = better", which is what the Rival Loop is designed to fix.
- **Not confirmed yet:** each result is a single seed, and the unlooped control ([E9](experiments/writeups/E9.md)) is still pending.

Every experiment has a page with interactive charts and a plain-English explainer on the **[project site](https://monstercameron.github.io/Gogeta-0/)**, e.g. [E2: why the loop disappoints](https://monstercameron.github.io/Gogeta-0/experiments/E2.html). The **[full roadmap](experiments/experiments.json)** runs E0 → E9: ternary weights, the full fusion (→ `v1.0.0`), adaptive loops, the Rival Loop, loops vs chain-of-thought, and self-speculative decoding.

## Quick start

Tested on Ubuntu 24.04 under WSL2 with an NVIDIA Blackwell GPU (ARM64). Any recent CUDA GPU that runs Triton should work.

```bash
# 1. Environment (Triton needs the Python headers)
sudo apt-get install -y python3.12-venv python3.12-dev
python3 -m venv ~/.venvs/gogeta && source ~/.venvs/gogeta/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e .

# 2. Check the GPU kernels and the model
python tests/test_gdn_kernels.py
python tests/test_model.py

# 3. Build the dataset (downloads ~2.8 GB; Python-only subset, ~262M tokens)
python tools/prepare_data.py

# 4. Train (the configs are E0/E1/E2/E9) and watch it in the local dashboard
python -m gogeta.train --config configs/e1_hybrid.json --compile
python tools/dashboard.py          # http://127.0.0.1:7861

# 5. Rebuild the project site after recording results
python tools/build_site_data.py
python -m http.server 8791 --directory docs
```

Data, checkpoints and run logs go to `~/fusiondance-data/` (set `GOGETA_DATA` to change it), never into the repo. Tracking is fully local ([Trackio](https://huggingface.co/docs/trackio)); nothing is uploaded.

## Repository layout

```
gogeta/              the package: model.py (FusionDance), train.py, tracking.py, assets/ (tokenizer)
configs/             one JSON config per experiment
tests/               GPU kernel check against a reference + model correctness tests
tools/               data pipeline, run queue, result recorder, dashboard, site builder
experiments/         experiments.json (registry: predictions, status, results),
                     writeups/ (per-experiment explainers), curves/ (eval curves)
docs/                the GitHub Pages site (landing, devlog, primer, a page per experiment)
site-src/            source for the "Start here" primer
notes/               LABNOTEBOOK.md (timestamped evidence) + sketch.md (design scratchpad)
refs/                55-paper reading list + fetch script (PDFs are not committed)
DEVLOG.md            versioned history: every entry is a commit and a git tag
AGENTS.md            working rules for humans and AI agents in this repo
```

## How the research is run

- **Predictions first.** Every experiment's prediction is committed *before* it runs, then scored hit/miss afterwards.
- **One ingredient at a time**, so a failed fusion can be traced to a specific change.
- **Provenance.** Every run records its git commit, config and seed, and runs for a release refuse to start on uncommitted code.
- **Devlog = commits = versions.** Each entry in [`DEVLOG.md`](DEVLOG.md) is exactly one commit and one semver tag (`0.y.z` until the full fusion works).
- **Everything explained twice:** a detailed breakdown for researchers, and a plain-English section for everyone else.

Full rules are in [`AGENTS.md`](AGENTS.md).

## Project site

Live at **[https://monstercameron.github.io/Gogeta-0/](https://monstercameron.github.io/Gogeta-0/)**. GitHub Pages deploys it straight from `main` / `docs` on every push; it's static, so there's no build step on GitHub. Regenerate it locally with `python tools/build_site_data.py` before committing. Every version is a git tag ([all tags](https://github.com/monstercameron/Gogeta-0/tags)).

## Standing on the shoulders of

BitNet b1.58, Mamba / Mamba-2, Gated DeltaNet, Kimi Linear, Samba, Zamba, Universal Transformers, Recurrent Depth (Huginn), Mixture-of-Recursions, TRM / HRM, LayerSkip, DFlash, and more: 55 papers with notes in [`refs/README.md`](refs/README.md). The closest prior work we know of, a TRM-style recursive Mamba-2/attention hybrid, is flagged there. FusionDance's bet is the *full* fusion, ternary included, aimed at a general causal LLM.

**Credits**
- **Training data:** [OpenCoder-LLM/opc-annealing-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus) (`algorithmic_corpus` + `synthetic_code_snippet`, Python only), licensed under **ODC-By 1.0**. The tokenizer in `gogeta/assets/` was trained on it. The data itself is not redistributed here.
- **Kernels:** [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (Gated DeltaNet chunk kernel). **Tracking:** [Trackio](https://huggingface.co/docs/trackio). **Framework:** PyTorch.

## License

Code and documentation: [MIT](LICENSE). The dataset keeps its own license (ODC-By 1.0); paper PDFs are fetched from arXiv and not redistributed.

---

<sub>*The name is a fan homage to a certain two-person fusion dance: two fighters become one stronger one, but only if the dance is perfectly in sync. This is an independent research project, not affiliated with or endorsed by the Dragon Ball rights holders.*</sub>
