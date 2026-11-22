"""Download the FusionDance reading list from arXiv and rebuild refs/README.md.

Usage:  python refs/fetch_refs.py
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
}


def safe(name):
    return "".join(c if c.isalnum() or c in " .-+()" else "_" for c in name).strip()


def main():
    ok, skipped, failed = 0, 0, []
    for folder, (_, papers) in PAPERS.items():
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
    lines = ["# FusionDance references", "",
             "Generated by `fetch_refs.py`: edit the PAPERS list there, not this file.", ""]
    for folder, (title, papers) in PAPERS.items():
        lines += [f"## {title}  (`{folder}/`)", "", "| arXiv | Paper | Why it's here |", "|---|---|---|"]
        for aid, name, why in papers:
            pdf = f"{folder}/{aid} - {safe(name)}.pdf".replace(" ", "%20")
            lines.append(f"| [{aid}](https://arxiv.org/abs/{aid}) | [{name}]({pdf}) | {why} |")
        lines.append("")
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
