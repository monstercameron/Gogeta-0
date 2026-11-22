# Proposed claims C220-C229 (byte-level models)

Proposals only; the ledger is being redesigned, so no ledger file was touched. Type: **R** = reported by a paper (UNVERIFIED here, quote only), **C** = computed from our code/data (reproducible), **M** = measured locally, **H** = hypothesis to test. All paper numbers were read through a page-summarising fetch tool, not the PDFs: re-check before use.

| ID | Proposition | Type | Evidence / how to settle | Proposed status |
|---|---|---|---|---|
| C220 | On our 4,000 val docs the 8k BPE averages 3.735 bytes (3.733 chars) per token; bytes per char is 1.0005, so bits/byte equals K8's bits/char to 0.05% | M | `experiments/byte_level/seq_stats.py` -> `seq_stats.json` (CPU). Train-split ratio not measured | VERIFIED on val (script and output on disk); train split unmeasured |
| C221 | A byte-level E1 (vocab 256, T=4,096, window 1,024) costs 86.9 MFLOP/byte train = 3.57x E1's FLOPs per unit of text; matched FLOPs means 28% of the text (209.6M bytes) | C | `experiments/byte_level/flop_arith.py` (same formula as `flops_per_token`; its E1 reproduces the registry's 91M). Not the model code itself | computed |
| C222 | A fixed-patch-4 BLT-lite (local d=192, E1 body on 1,024 patches) has 12.92M params (-9.3% vs E1) and 1.14x E1's FLOPs per unit of text | C | `flop_arith.py`; param count not yet checked against an instantiated module | computed, check on build |
| C223 | At matched training FLOPs, plain byte Transformers are worse than subword Transformers (SpaceByte: PG-19 1.138 vs 0.989; GitHub 0.655 vs 0.508 bpb), and patching/SSM/hierarchy closes the gap | R | SpaceByte (2404.14408) Table 1; MambaByte (2401.13660); H-Net (2507.07955) | UNVERIFIED (reported); our check is SQ180 |
| C224 | MambaByte (SSM on bytes) beats MegaByte and a byte Transformer on bpb at 353M / 30B bytes (Code 0.396 vs 0.411 vs 0.575) and is 29 s vs 132 s for 8,192 bytes of generation | R | MambaByte Tables 2, 7. Scale 25x above ours | UNVERIFIED (reported) |
| C225 | At 8B/4T bytes BLT matches or beats Llama 3 (avg 61.1 vs 60.0; HumanEval 35.4 vs 31.1) and H-Net beats a BPE Transformer on GitHub bpb at 1.3B (0.3161 vs 0.3376) | R | BLT (2412.09871) Table 1; H-Net (2507.07955) | UNVERIFIED (reported); no paper tests below ~300M |
| C226 | Byte-level interfaces cost quality at 14M params / 0.75 GB: byte-E1 ratio vs BPE-E1 on clean-val bits/char is 1.10-1.35 at matched FLOPs and 0.97-1.10 at matched bytes | H | SQ180; refute/confirm per the criteria in the note | UNVERIFIED (prediction) |
| C227 | Fixed-patch-4 BLT-lite reaches within 3% of BPE-E1 clean-val bits/char at 1.14x its FLOPs | H | SQ181 | UNVERIFIED (prediction: 1.00-1.12, central 1.05) |
| C228 | The fla chunk GDN kernel handles variable-length patch rows (cu_seqlens) correctly on sm_121, and T=4,096 aligns with its chunk size | H | SQ185 | UNVERIFIED |
| C229 | Byte models degrade less than BPE models on perturbed code (typos, rare identifiers), which clean-val bits/char cannot show because 41.7% of val is near-dup of train (C131) | H (R for general text: MambaByte noise table, BLT noised HellaSwag, H-Net) | SQ187 | UNVERIFIED |
