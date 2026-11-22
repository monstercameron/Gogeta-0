# Gogeta-0 architecture viz

A tiny local web page that draws every model in `configs/*.json` **from the real code**: the server
imports `gogeta/model.py`, builds `FusionDance` from the config on CPU, and reports what it actually
contains. Nothing is hand-copied, so the picture cannot drift from the model.

- URL: **http://127.0.0.1:7870** (localhost only)
- Stdlib `http.server` plus the venv's torch/fla. No pip installs, no build step, no CDN except Google Fonts.
- CPU only: `CUDA_VISIBLE_DEVICES=""`, 2 threads. It never touches the GPU, so it can run next to training.

## Start / stop

PowerShell (Windows):

```powershell
.\tools\launch_viz.ps1 start -Open     # start (idempotent) and open the browser
.\tools\launch_viz.ps1 status
.\tools\launch_viz.ps1 url
.\tools\launch_viz.ps1 stop
.\tools\launch_viz.ps1 start -Port 7871   # another port
```

WSL:

```bash
bash tools/launch_viz.sh start [--port N]   # default 7870; waits for HTTP, then prints the URL
bash tools/launch_viz.sh status|url|stop [--port N]
```

It runs detached (`setsid nohup`), so closing the terminal does not stop it. PID file
`~/fusiondance-data/run/viz-<port>.pid`, log `~/fusiondance-data/logs/viz-<port>.log`. `stop` only kills
the process in its own PID file. Ports already taken in this project: 7861 (Trackio), 8791 (docs preview).

Without the launcher: `CUDA_VISIBLE_DEVICES= ~/.venvs/gogeta/bin/python tools/viz/server.py --port 7870`.

## How to read it

Pick a model (top left), optionally tick **side by side** to compare two. The sliders and toggles apply
to every panel. Hover (or tap) any block for exact numbers; the state is kept in the URL hash, so a view
can be bookmarked.

| Shape | Meaning |
|---|---|
| **Filled lower triangle** (blue) | Full causal attention. The n x n score matrix: every token looks at every earlier token, so cost grows with the **square** of the length. The triangle's side is proportional to the context (the dashed frame is 32k). |
| **Diagonal band** (green) | Sliding-window attention, window w tokens (256). Each token looks back w tokens only, so cost and cache grow **linearly**, then the cache caps at w. Same scale as the triangle, so the band thins as context grows. |
| **Fixed square tile + circling arrow** (orange) | Gated DeltaNet. A small fixed-size memory per head (heads x head_dim x head_dim) rewritten token by token. It **does not grow** with context, whatever the slider says. |
| **Expand-then-contract trapezoids** (yellow) | The SwiGLU MLP, d -> hidden -> d. Per-token compute, no look-back. |
| **Dotted -1 / 0 / +1 texture + gold "1.58-bit" badge** | Ternary (BitLinear) projections. Embeddings, norms and the GDN gate/decay projections stay full precision, so those parts have no dots. |
| **Gold dashed box with a loop arrow "x r"** | The looped core: stored once, run r times, output fed back to input. |
| **"Unroll the loop"** | Draws the core r times. Loop 1 is the real weights; loops 2..r are compact chips with a **chain icon**: same weights, extra compute. Effective depth grows, parameters do not. |
| **Re-inject** (grey-violet) | The layer that mixes the original input e back into h at the start of every loop. |
| **Striped bar on the embedding** | A scale break: the embedding is capped in height, its true count is in the label. |
| **Block size** | Height is proportional to parameters: 1,000,000 params = 100 px. Parts smaller than 56 px are drawn at that minimum so the glyph fits. |

Colours (blue attn, green SWA, orange GDN, yellow MLP) come from the dataviz dark palette. They are never
the only cue: each kind has its own glyph and a text label.

**Green scouter panel** ("POWER LEVEL"): total params, non-embedding / embedding split, effective depth,
train FLOPs per token (`FusionDance.flops_per_token`, seq 1024), weight memory in bf16 and (ternary models)
2-bit packed, the ternary share, and the val loss from `experiments/experiments.json`.
Its **"AT n TOKENS" readout** follows the context slider (1k to 32k) and shows how each cost grows versus
1k: full-attention KV cache and score cells grow with n and n^2, SWA KV caps at the window, GDN state is flat.
Memory assumes bf16 (2 bytes/elem); KV = 2 x d per attention layer per token; each pass through a looped core
counts as its own layer. **Loops slider** (1 to 12) changes r for looped models: effective depth and FLOPs update.

E10 (`"arch": "hf_llama"`) is Hugging Face's stock Llama, not FusionDance. It is drawn as its equivalent
6 x (full attention + SwiGLU) stack (shapes and param counts match) with a note saying so.

## How it reads the code

`GET /api/models` lists `configs/*.json` joined with `experiments/experiments.json`. `GET
/api/model/<config-name>?loops=r` constructs `FusionDance(ModelConfig(**cfg["model"]))`, walks its
prelude/core/coda blocks and sums real `parameters()` per sub-module, counts `BitLinear` modules, and calls
`param_counts()`, `effective_depth()` and `flops_per_token()`. `gogeta/model.py` is **reloaded when its mtime
changes** and models are rebuilt when a config changes, so edits show up on the next page action. A **new
config file appears in the dropdown automatically**, no restart. If a build fails, the panel shows the error.

## Troubleshooting

- **Blank panel / red error text**: the config or `model.py` failed to build. Read the error; the log
  (`~/fusiondance-data/logs/viz-7870.log`) has request lines and any traceback.
- **"port in use ... not the viz"**: pick another `--port`, or see who owns it: `ss -ltnp | grep 7870`.
- **Start times out**: the first request imports torch + fla (several seconds). `status` shows the pid;
  check the log. A harmless "Triton is not supported ... roll back to CPU" warning is expected.
- **Fonts look plain**: Google Fonts did not load (offline). Everything else still works.
- **Stale after editing `model.py`**: reload the page and move any control; the module reloads on the next API call.
- Windows can't reach it: use the PowerShell wrapper; `127.0.0.1` forwarding from WSL is automatic on current WSL.

## Adding a glyph for a new block kind

1. In `gogeta/model.py`, add the kind to `Block`. The server needs no change: it reports `kind` as-is
   (add a label in `mixer_type` inside `describe()` in `tools/viz/server.py`, and cache terms in `at_ctx()` if it has KV/state).
2. In `tools/viz/static/app.js`: add an entry to `KIND` (name, 3-4 letter short, css class `cls`), write a
   `glyphMyKind(ctx, window)` function returning a 56 x 56 SVG string (encode the **cost shape**: how it grows
   with n), and add it to the glyph ternary in `blockHtml()`.
3. In `style.css`: add `--c-<cls>` plus a `.part.mixer.k-<cls>` rule. Pick a colour validated against the
   existing set: `node validate_palette.js "<hexes>" --mode dark --surface "#161c3a" --pairs all`.
4. Add a legend entry in `buildLegend()` and a row to the table above.

## 3D site page

`docs/architecture.html` (+ `docs/assets/arch3d.js`, `arch3d.css`) is an interactive three.js view of every
config, for the static GitHub Pages site. Pages has no Python, so the page loads a pre-computed snapshot,
`docs/data/architectures.json`, written by `tools/viz/export_arch.py`. The exporter imports `describe()` from
`tools/viz/server.py` (nothing is re-counted), so the 3D numbers equal the 2D viz. It also adds per-loop FLOPs
(r = 1..12), the registry title/status/val_loss, and a `generated_at` + `git describe` stamp.

three.js is pinned to `three@0.160.0` (plus `OrbitControls` and `CSS2DRenderer` from the same version) via an
import map from jsDelivr. Shapes: plinth height = mixer params; full attention = triangular wedge whose side
grows with the context slider (area ~ n^2); sliding window = fixed-width diagonal band; Gated DeltaNet = fixed
cube; MLP = hexagonal bicone (height = params); looped core = blocks on a glowing ring, unrolled by the loops
slider into ghost copies (same weights); ternary = -1/0/+1 speckles; gold particles = data flow.

**Regenerate after model or config changes** (CPU only, in WSL):

```bash
bash tools/launch_cpu_job.sh --name arch-export -- ~/.venvs/gogeta/bin/python tools/viz/export_arch.py
# or directly: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 ~/.venvs/gogeta/bin/python tools/viz/export_arch.py
```

**Preview:** `python -m http.server 8792 --bind 127.0.0.1 --directory docs`, then open
http://127.0.0.1:8792/architecture.html. `window.__arch3d` exposes scene stats and setters for testing.
