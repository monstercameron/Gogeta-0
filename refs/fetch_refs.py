"""Download the Gogeta-0 / FusionDance reading list from arXiv and rebuild refs/README.md.

Usage:  python refs/fetch_refs.py [section-folder ...]   (no argument = all sections)
Add a paper: append (arxiv_id, short_name, why) to the right section in PAPERS, re-run.
Already-downloaded PDFs are skipped. Waits 3 s between requests (arXiv etiquette).
"""
import pathlib, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
UA = {"User-Agent": "paper-fetch/0.1"}

PAPERS = {
    "01-bitnet": ("BitNet / low-bit weights", [
        ("2310.11453", "BitNet", "Original 1-bit transformer (BitLinear)"),
        ("2402.17764", "BitNet b1.58", "Ternary {-1,0,1} weights, absmean quantization"),
        ("2504.12285", "BitNet b1.58 2B4T", "Open-weights ternary model; training recipe details"),
        ("2410.16144", "bitnet.cpp", "Fast ternary inference kernels (CPU)"),
        ("2411.11843", "Bi-Mamba", "1-bit SSM: the closest crossover of BitNet and SSMs"),
    ]),
    "02-linear-time-mixers": ("SSMs & linear attention (core-loop mixer candidates)", [
        ("2312.00752", "Mamba", "Selective state spaces"),
        ("2405.21060", "Mamba-2 (SSD)", "Structured state space duality; chunked scan"),
        ("2603.15569", "Mamba-3", "Latest Mamba; ablation baseline"),
        ("2406.06484", "DeltaNet (parallel)", "Delta-rule linear attention, chunkwise parallel training"),
        ("2412.06464", "Gated DeltaNet", "First-choice core-loop mixer"),
        ("2605.22791", "Gated DeltaNet-2", "Decoupled erase/write gates"),
        ("2510.26692", "Kimi Linear (KDA)", "Production delta-rule hybrid"),
        ("2506.04761", "Log-Linear Attention", "O(n log n) middle ground; coda alternative"),
        ("2407.04620", "TTT layers", "Test-time-training hidden state (wild card)"),
        ("2501.00663", "Titans", "Neural memory that learns at test time (wild card)"),
    ]),
    "03-hybrids": ("Hybrid SSM/attention architectures", [
        ("2403.19887", "Jamba", "Interleaved Mamba + attention + MoE"),
        ("2405.16712", "Zamba", "Shared attention block reused across depth (loop-like!)"),
        ("2406.07522", "Samba", "Mamba + sliding-window attention"),
        ("2602.03681", "NAtS-L", "Per-token choice of linear vs softmax attention"),
    ]),
    "04-sparse-window-attention": ("Sliding-window & sparse attention", [
        ("2004.05150", "Longformer", "Sliding-window + global attention (classic)"),
        ("2502.11089", "NSA", "Native Sparse Attention"),
        ("2502.13189", "MoBA", "Mixture of Block Attention"),
        ("2512.02556", "DeepSeek-V3.2 (DSA)", "DeepSeek Sparse Attention with lightning indexer"),
        ("2603.28458", "HISA", "Hierarchical (cheaper) DSA indexer"),
        ("2605.07363", "MISA", "Mixture-of-indexer sparse attention"),
    ]),
    "05-looped-recursive": ("Looped / recursive-depth models", [
        ("1807.03819", "Universal Transformers", "Weight-shared depth + ACT"),
        ("2301.13196", "Looped Transformers as Programmable Computers", "Theory of looping"),
        ("2502.05171", "Recurrent Depth (Huginn)", "Prelude/recurrent/coda, latent reasoning at scale"),
        ("2507.10524", "Mixture-of-Recursions", "Per-token adaptive recursion depth"),
        ("2506.21734", "HRM", "Hierarchical Reasoning Model"),
        ("2510.04871", "TRM", "Tiny Recursive Model: 7M baseline to beat"),
        ("2602.12078", "TRM + Mamba-2 Hybrid", "CLOSEST PRIOR ART: read first"),
        ("2605.19943", "Probabilistic TRM", "TRM follow-up"),
    ]),
    "06-speculative-decoding": ("Inference: speculative decoding", [
        ("2211.17192", "Speculative Decoding", "The original method"),
        ("2404.16710", "LayerSkip", "Early exit + self-speculative decoding"),
        ("2503.01840", "EAGLE-3", "Strong feature-level drafter baseline"),
        ("2602.06036", "DFlash", "Block-diffusion drafter"),
        ("2607.05147", "DSpark", "Confidence-scheduled semi-autoregressive drafting"),
    ]),
    "07-rival-loop-halting": ("Rival Loop: halting & self-correction", [
        ("1603.08983", "Adaptive Computation Time", "Learned halting (Graves)"),
        ("2107.05407", "PonderNet", "Probabilistic halting"),
        ("2310.01798", "LLMs Cannot Self-Correct Reasoning Yet", "Why the critic needs grounded signal"),
    ]),
    "08-data": ("Training data", [
        ("2305.07759", "TinyStories", "Smoke-test corpus"),
        ("2406.17557", "FineWeb (+ FineWeb-Edu)", "Main real-text corpus"),
        ("2502.02737", "SmolLM2", "SmolLM-Corpus mix + small-model data recipe"),
        ("2506.01732", "Common Corpus", "~2T tokens, open licenses"),
        ("2509.25531", "MixtureVitae", "422B tokens, permissive-first"),
        ("2604.13977", "FinePhrase", "486B synthetic rephrased web"),
        ("2411.06068", "Zyda-2", "5T-token distilled mix"),
    ]),
    "09-evals": ("Evals & benchmarks", [
        ("1912.00582", "BLiMP", "Grammar minimal pairs"),
        ("2312.04927", "Zoology (MQAR)", "Associative recall in efficient LMs"),
        ("2402.18668", "Based", "Recall-throughput tradeoff; recall eval suite"),
        ("2404.06654", "RULER", "Synthetic long-context eval"),
        ("2406.10149", "BABILong", "Reasoning-in-a-haystack"),
        ("2505.24760", "Reasoning Gym", "Procedural reasoning tasks with verifiers"),
        ("2505.11831", "ARC-AGI-2", "Puzzle benchmark"),
    ]),
    "10-qwen": ("Qwen family (the Qwen4 architecture preview)", [
        ("2608.30320", "Qwen3.8-Next architecture",
         "Official Qwen design report for Qwen3.8-Flash-Next, the Qwen4 preview: 3:1 GDN/sparse-attention hybrid, "
         "gated residuals, n-gram embeddings, Muon; ablations + training stability"),
        ("2610.28998", "FlashNext-RL",
         "Third-party (2 authors): long-horizon agentic RL for Qwen3.8-Flash-Next on prime-rl + vLLM; RL systems lessons"),
        ("2505.09388", "Qwen3 Technical Report", "Baseline the Qwen3.x line builds on"),
    ]),
    "11-looped-2026": ("Looped transformers 2025–2026 (notes: notes/research-looped.md)", [
        ("2510.25741", "Ouro (LoopLM)",
         "ByteDance looped LMs (1.4B/2.6B, 7.7T tokens), entropy-regularised exit gate; also peaks at the trained T=4 "
         "and degrades at T=5-8, like our E2"),
        ("2605.20670", "LT2 Linear-Time Looped Transformers",
         "CLOSEST PRIOR ART for loop + GDN + sparse/window attention (0.6B/1.3B); no ternary, no prelude/coda split"),
        ("2607.10110", "CHASE (Looped Mamba)",
         "Looped Mamba-2 / hybrid LMs (50M-370M); iso-FLOP deeper non-looped still wins on perplexity"),
        ("2604.12946", "Parcae",
         "Loop instability = injection with spectral radius >= 1; contractive diagonal injection + prelude norm; "
         "test-time looping saturates near the trained mean"),
        ("2606.18524", "Loop-muP",
         "Shared-weight loop increments add coherently (norm ~ N); scale residual branches by 1/N for LR transfer"),
        ("2605.18797", "Fully Looped Transformer",
         "Residual explosion + gradient oscillation in loops; prev-loop state as attention query; stable to 12 loops"),
        ("2604.21254", "Hyperloop Transformers",
         "Loop-level hyper-connections (diagonal sigmoid mixing, loop embedding); looped models lose more under INT4 PTQ"),
        ("2602.11451", "LoopFormer",
         "Budget-conditioned elastic depth: time/step-size conditioning + shortcut-consistency loss (M <= trained L)"),
        ("2606.29983", "Learned Stochastic Stopping",
         "Loop extrapolation is high-variance across seeds; randomised / RL-learned loop counts stabilise it"),
        ("2604.21106", "Iso-Depth Scaling Laws (loops)",
         "One recurrence worth r^0.46 unique blocks; truncated BPTT lowers it to 0.38, hyper-connections raise it to 0.65"),
        ("2610.31943", "Agreement not Normalisation (loops)",
         "Removing Ouro's loop-boundary norm: from-scratch 19M models extrapolate better without it (16 seeds)"),
        ("2603.08391", "Adaptive Loops and Memory",
         "Per-layer PonderNet-style halting + learnable loop scales (init ~identity) + memory banks; ~200M, 14B tokens"),
    ]),
    "13-activations": ("Activation functions (notes: notes/research-activations.md)", [
        ("2002.05202", "GLU Variants Improve Transformer",
         "Shazeer: GEGLU/SwiGLU/ReGLU/Bilinear; hidden width scaled to 2/3 to match params (T5-base, 524k steps)"),
        ("2109.08668", "Primer (squared ReLU)",
         "Squared ReLU + depthwise conv after QKV found by search; 4x training-cost cut at 500M; origin of ReLU^2"),
        ("2411.13010", "xIELU",
         "Trainable piecewise activation (quadratic positive side, ELU-like negative side); beats SwiGLU/ReLU^2 at 1.1B/3B, 125B tokens"),
        ("2411.03884", "PolyCom (PolyReLU / PolyNorm)",
         "Order-3 polynomial composition activations, trainable coefficients; 1.3B dense / OLMoE; PolyNorm designed for bf16 stability"),
        ("2603.13347", "PolyGLU",
         "Per-neuron routed choice among ReLU/Tanh/SiLU/GELU in a GLU; 597M, 10B tokens, no SwiGLU control; deep layers pick Tanh"),
        ("2605.26647", "Mixture of Activations (MoA)",
         "Token-adaptive mixing of activations (+ input-independent learnable-activation baseline), 0.12B-2B dense/MoE"),
        ("2310.04564", "ReLU Strikes Back",
         "ReLU in place of GELU/SiLU costs ~nothing in quality and gives exploitable activation sparsity"),
        ("2406.05955", "TurboSparse (dReLU)",
         "dReLU (ReLU on both gate and up branches) for high GLU sparsity; Mistral/Mixtral sparsified"),
        ("2402.13516", "ProSparse",
         "Progressive sparsity regularisation + ReLU for ~89% FFN sparsity without quality loss"),
        ("2411.04965", "BitNet a4.8",
         "Ternary model with 4-bit inputs to attention/FFN, sparsified 8-bit intermediate states; squared-ReLU GLU sparsity"),
        ("2504.18415", "BitNet v2",
         "Hadamard transform before activation quantisation tames outliers in down-proj/o-proj inputs (4-bit activations)"),
        ("2402.16819", "Nemotron-4 15B",
         "Squared ReLU (ungated) in a 15B production LM, 8T tokens"),
        ("2402.17762", "Massive Activations",
         "A few huge fixed-position activations in LLM residual streams; outlier source relevant to int8 activations"),
        ("2505.06708", "Gated Attention (Qwen)",
         "Sigmoid output gate after SDPA: non-linearity + sparsity, fewer loss spikes, no attention sink; gate-activation ablation"),
        ("2404.19756", "KAN",
         "Learnable univariate edge functions (splines); the 'learnable activation' extreme; parameter/compute heavy"),
    ]),
    "14-byte-level": ("Byte-level / tokenizer-free LMs (notes: notes/research-byte-level.md, notes/research-byte-level-2025-26.md)", [
        # --- foundational (pre-2025) ---
        ("2105.13626", "ByT5", "2021: T5 on raw bytes; the 'robust but slow' baseline every later byte paper cites"),
        ("2103.06874", "CANINE", "2021: char-level encoder, downsample -> deep stack -> upsample; the pooled-hierarchy idea"),
        ("2106.12672", "Charformer", "2021: GBST, learned soft block pooling over bytes; early learned-chunking precedent"),
        ("2305.07185", "MegaByte", "2023: fixed patches, global/local split, FLOP formulas, init std 0.006; basis of our BLT-lite (SQ181)"),
        ("2401.13660", "MambaByte", "2024: Mamba SSM on raw bytes, Code bpb 0.396; closest precedent for GDN-on-bytes (SQ180); byte speculative decoding"),
        ("2404.14408", "SpaceByte", "2024: rule-based patches after space-like bytes; fixed-patch ablation; cleanest matched-FLOPs byte vs subword table (SQ184)"),
        ("2412.09871", "BLT (Byte Latent Transformer)", "2024-12: entropy patching, local enc/dec + latent core, hash n-grams; 8B/4T bytes; HumanEval/MBPP"),
        ("2402.19155", "bGPT", "2024: next-byte prediction across file types; low priority"),
        # --- 2025 ---
        ("2502.14553", "MBLM (Multiscale Byte LM)", "2025-02: model-agnostic hierarchical byte decoder stack with Transformer and Mamba blocks; hybrids best on long byte sequences"),
        ("2506.14761", "AU-Net", "2025-06: autoregressive U-Net, bytes -> words -> word pairs -> 4 words; shallow hierarchies tie BPE"),
        ("2506.14123", "Byte-wise sampling from a BPE LM", "2025-06: BPE LM -> byte distribution; token-boundary misalignment (matters for code)"),
        ("2507.07955", "H-Net (Dynamic Chunking)", "2025-07: learned chunking in a recursive U-Net with Mamba-2 outer layers; GitHub 0.316 vs 0.338 bpb; ICLR 2026"),
        ("2508.05628", "H-Net++", "2025-08: H-Net for Persian; 1.9M-param cross-chunk mixer, staged sequence-length curriculum; small-data regime (1.4B tokens)"),
        ("2511.11346", "MTPC (multi-byte prediction, prob. circuits)", "2025-11: retrofits EvaByte / byteified Llama for multi-byte speculative decoding; byte decode-speed lever"),
        ("2512.15586", "Bolmo", "2025-12: byteify pretrained subword LMs (1B/7B) in <1% of pretraining cost; byte models do not have to be trained from scratch"),
        # --- 2026 ---
        ("2601.22805", "SOMBRERO", "2026-01: boundary-enrichment metric + confidence-alignment loss for learned patch boundaries (1B, code+math+text)"),
        ("2602.04289", "Proxy Compression", "2026-02: joint raw-byte + compressed-view training, raw bytes at inference; code LM, 0.5B-14B; tokenizer leads below ~4B"),
        ("2603.03583", "ByteFlow Net", "2026-03: coding-rate chunking with static-graph Top-K boundaries; sliding-window attn + Canon layers, 0.6B/1.3B; beats BPE Llama"),
        ("2603.15953", "T-Free HAT", "2026-03: byte enc -> word backbone -> byte dec, 7B from scratch + Llama retrofits; word-boundary patching"),
        ("2604.27263", "Decoupling Subword Benefits (byte simulation)", "2026-04: why subword beats bytes: training throughput (patch-4 warm-up) + boundary priors, not vocab size; 68M and 1.7B"),
        ("2605.01188", "Compute Optimal Tokenization", "2026-05: 988 BLT models 50M-7B; params scale with bytes not tokens; optimal compression rate falls with compute"),
        ("2605.08044", "Fast BLT", "2026-05: BLT-D diffusion / self-speculation, >50% lower decode memory-bandwidth cost"),
        ("2605.09630", "Scratchpad Patching", "2026-05: fixes patch lag with entropy-triggered scratchpads; fixed-p vs entropy vs SpaceByte vs H-Net on code, ~2B/400B bytes"),
        ("2605.12928", "Efficiency Gap in Byte Modeling", "2026-05: compute-matched byte vs BPE, 48M-1.2B, AR gap shrinks with scale (7.9x at 1.0 bpb, 2.3x at 0.8); counter-evidence at small scale"),
        ("2605.30080", "ATDC (Adaptive Targeted Dynamic Chunking)", "2026-05: curriculum on H-Net compression ratio (low -> high) stabilises chunk learning; FineWeb-Edu 100B"),
        ("2608.15454", "Dynamic Multi-Byte Prediction (MBP)", "2026-08: variable-length multi-byte decode aligned to latent segments of a hierarchical LM; no extra params"),
        ("2609.12303", "Breaking the Token Ceiling", "2026-09: 1B byte vs token overtraining to 1T bytes; token wins at low FLOPs, byte overtakes later; distillation via byte logits"),
        ("2610.05978", "Byte LMs: Scaling, Emergent Abstractions", "2026-10: plain byte Transformers + token-superposition + hash embeddings beat subword with scale; 3.4x speculative acceptance"),
        ("2610.11790", "Boundary dependence vs entropy patching", "2026-10: BLT entropy rule misses computed values at tight budgets; loss-change boundary signal; from-scratch 1M-50M byte models"),
        ("2610.22100", "Entropy patching skips computed values", "2026-10: companion; two-byte lookup-table patcher beats entropy at 50M; tiny-scale (1M-50M) patching evidence"),
        ("2611.04215", "Tokenizer Myths", "2026-11: BPB is NOT tokenizer-agnostic for untrained/real models; caution for our byte-vs-BPE bits/char comparisons"),
        ("2611.13607", "One Byte, One Token (27B byteified)", "2026-11: Qwen3.8-27B (GDN + full attn hybrid) to vocab-384 pure byte model, no patching; 3-5x less output per step"),
    ]),
    "15-moe": ("Mixture-of-Experts: code MoE, looped MoE, specialisation, routing (notes: notes/research-code-moe.md)", [
        # --- code models built on MoE ---
        ("2406.11931", "DeepSeek-Coder-V2", "MoE code LM (16B/2.4B-active, 236B/21B-active); generic DeepSeekMoE experts, not code-dimension experts"),
        ("2401.06066", "DeepSeekMoE", "Fine-grained + shared experts; the framework under DeepSeek-Coder-V2"),
        ("2407.10671", "Qwen2 Technical Report", "Includes Qwen2-57B-A14B MoE; generic experts"),
        ("2508.19268", "MultiPL-MoE", "Token-level + segment-level (expert-choice) experts for programming-language syntax; segment EC is non-causal for pretraining"),
        # --- MoE x recurrent / universal / looped depth ---
        ("2310.07096", "Sparse Universal Transformer (SUT)", "UT + MoE FFN/attention + stick-breaking halting"),
        ("2405.16039", "MoEUT", "Shared-layer MoE UT; beats param-matched dense from 44M (C4 18.30 vs 18.97 ppl); experts reused across layers; also on The Stack"),
        ("2605.09165", "Sparse Layers for Looped LMs (Looped-MoE)", "MoE fixes looped-model scaling; 25-53% of tokens get disjoint experts on loop 2; E=8 k=2; 5e16-1e18 FLOPs"),
        ("2404.02258", "Mixture-of-Depths", "Per-token layer skipping (routing depth, not experts)"),
        # --- routing, balancing, collapse ---
        ("2101.03961", "Switch Transformer", "Top-1 routing + load-balance loss (alpha 0.01); capacity factor"),
        ("2202.08906", "ST-MoE", "Router z-loss and stability recipe"),
        ("2408.15664", "Auxiliary-Loss-Free Load Balancing", "DeepSeek bias-based balancing without an aux loss"),
        ("2202.09368", "Expert Choice Routing", "Experts pick tokens: balanced by construction, non-causal at token level"),
        ("2106.04426", "Hash Layers", "Fixed token-id hash routing competitive with learned routing: the 'is routing doing anything' control"),
        ("2204.09179", "X-MoE (representation collapse)", "Routing representation collapse and a fix"),
        ("2402.13089", "Empirical MoE Design Choices (Fan et al.)", "GPT-2 small: token vs sequence routing; frozen random router ~ learned; only token-level top-2 beats dense at equal total params"),
        ("2202.01169", "Unified Scaling Laws for Routed LMs", "Routed-model scaling with active vs total params across 5 orders of magnitude"),
        ("2402.07871", "Scaling Laws for Fine-Grained MoE", "Expert granularity as a scaling axis; FFN-sized experts are rarely optimal"),
        ("2310.10837", "sigma-MoE (Approximating 2-layer FFNs)", "Parameter-equal MoE vs dense Transformer-XL at small scale (WikiText-103, enwik8)"),
        # --- is specialisation semantic? ---
        ("2401.04088", "Mixtral of Experts", "Routing analysis: no domain specialisation; routing follows syntax"),
        ("2402.01739", "OpenMoE", "Context-independent (token-ID) specialisation, early routing saturation, drop-towards-the-end"),
        ("2409.02060", "OLMoE", "From-scratch MoE: domain + vocabulary specialisation, rare co-activation, early router saturation"),
        ("2604.02178", "The Expert Strikes Back", "Experts as fine-grained task experts; code concepts probed; attribution, no knockout"),
        ("2601.03425", "Standing Committee (Illusion of Specialization)", "Domain-invariant coalition of experts takes most routing mass"),
        # --- supervised / domain-routed / modular experts ---
        ("2108.05036", "DEMix Layers", "Domain-label-routed FFN experts; modular add/remove of domains"),
        ("2208.03306", "Branch-Train-Merge", "Separately trained domain expert LMs"),
        ("2303.14177", "c-BTM", "Unsupervised domain clusters -> expert LMs, sparse ensemble"),
        ("2403.07816", "Branch-Train-MiX", "Domain experts merged into an MoE with a learned router"),
        ("2110.03742", "Task-level MoE (Beyond Distillation)", "Token vs sentence vs task-level routing granularity"),
        # --- region / segment-level routing ---
        ("2405.03133", "Lory", "Fully differentiable MoE; causal segment routing (previous segment decides); domain specialisation without labels"),
    ]),
}
# Note (2026-11-22): no Qwen4 paper exists on arXiv (title search "Qwen4" returned nothing); Qwen3.8-Flash-Next is
# Alibaba's announced Qwen4 architecture preview. Re-check and add Qwen4 here when an official report appears.


def safe(name):
    return "".join(c if c.isalnum() or c in " .-+()" else "_" for c in name).strip()


def main(only=None):
    """only: optional set of section folder names (e.g. {"13-activations"}); default is all sections."""
    ok, skipped, failed = 0, 0, []
    for folder, (_, papers) in PAPERS.items():
        if only and folder not in only:
            continue
        d = ROOT / folder
        d.mkdir(exist_ok=True)
        for aid, name, _ in papers:
            out = d / f"{aid} - {safe(name)}.pdf"
            if out.exists() and out.stat().st_size > 10_000:
                skipped += 1
                continue
            try:
                req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers=UA)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if not data.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                out.write_bytes(data)
                ok += 1
                print(f"ok   {out.relative_to(ROOT)}  ({len(data)//1024} KB)", flush=True)
            except Exception as e:
                failed.append((aid, name, str(e)))
                print(f"FAIL {aid} {name}: {e}", flush=True)
            time.sleep(3)
    write_readme()
    print(f"\ndownloaded {ok}, already had {skipped}, failed {len(failed)}")
    for f in failed:
        print("  failed:", *f)


def write_readme():
    lines = ["# Gogeta-0 references (FusionDance architecture)", "",
             "Generated by `fetch_refs.py`: edit the PAPERS list there, not this file.", ""]
    for folder, (title, papers) in PAPERS.items():
        lines += [f"## {title}  (`{folder}/`)", "", "| arXiv | Paper | Why it's here |", "|---|---|---|"]
        for aid, name, why in papers:
            pdf = f"{folder}/{aid} - {safe(name)}.pdf".replace(" ", "%20")
            lines.append(f"| [{aid}](https://arxiv.org/abs/{aid}) | [{name}]({pdf}) | {why} |")
        lines.append("")
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(set(sys.argv[1:]) or None)
