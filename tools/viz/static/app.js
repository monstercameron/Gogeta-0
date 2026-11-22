/* Gogeta-0 architecture viz: frontend. No build step, no dependencies. Data comes from /api/model/<config>. */
(() => {
"use strict";

const PX_PER_M = 100;            // block-area scale: 1,000,000 params = 100 px of height
const MIN_PART = 56;             // floor so a glyph always fits (tiny parts are drawn at the floor)
const GLYPH = 52;                // attention glyph frame = the 32k-token square
const KIND = {
  attn: { name: "Full attention", short: "ATTN", cls: "attn" },
  swa:  { name: "Sliding-window attention", short: "SWA", cls: "swa" },
  gdn:  { name: "Gated DeltaNet", short: "GDN", cls: "gdn" },
  inj:  { name: "Re-inject", short: "INJ", cls: "inj" },
};
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ---------------------------------------------------------------- formatting
const fmtFull = (n) => Math.round(n).toLocaleString("en-US");
function fmtN(n) {
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(Math.round(n));
}
function fmtBytes(b) {
  if (b >= 2 ** 30) return (b / 2 ** 30).toFixed(2) + " GiB";
  if (b >= 2 ** 20) return (b / 2 ** 20).toFixed(2) + " MiB";
  if (b >= 2 ** 10) return (b / 2 ** 10).toFixed(1) + " KiB";
  return Math.round(b) + " B";
}
function fmtFlops(f) {
  if (f >= 1e9) return (f / 1e9).toFixed(2) + " GFLOPs";
  if (f >= 1e6) return (f / 1e6).toFixed(1) + " MFLOPs";
  return fmtN(f) + " FLOPs";
}
const fmtCtx = (n) => (n >= 1024 ? n / 1024 + "k" : String(n));

// ---------------------------------------------------------------- glyphs (inline SVG strings)
function glyphAttn(ctx, window_, S = GLYPH) {
  // filled lower triangle: causal n x n score matrix. Side is proportional to ctx (32k = full frame).
  const s = clamp(S * ctx / 32768, 8, S);
  return `<svg width="${S + 4}" height="${S + 4}" viewBox="0 0 ${S + 4} ${S + 4}" aria-hidden="true">
    <rect class="glyph-frame" x="2" y="2" width="${S}" height="${S}"/>
    <polygon points="2,2 2,${2 + s} ${2 + s},${2 + s}" fill="var(--c-attn)" stroke="#fff" stroke-opacity=".5"/>
    <polygon points="2,2 ${2 + s},2 ${2 + s},${2 + s}" fill="none" stroke="var(--dim)" stroke-dasharray="2 2" stroke-opacity=".7" ${s < 10 ? 'display="none"' : ""}/>
  </svg>`;
}
function glyphSwa(ctx, window_, S = GLYPH) {
  // diagonal band of width w tokens; same scale as the attention glyph, so it thins as ctx grows.
  const s = clamp(S * ctx / 32768, 8, S);
  const bw = Math.min(s, Math.max(3, window_ / ctx * s));
  const x0 = Math.max(0, s - bw);
  return `<svg width="${S + 4}" height="${S + 4}" viewBox="0 0 ${S + 4} ${S + 4}" aria-hidden="true">
    <rect class="glyph-frame" x="2" y="2" width="${S}" height="${S}"/>
    <rect x="2" y="2" width="${s}" height="${s}" fill="none" stroke="var(--dim)" stroke-opacity=".6"/>
    <polygon points="2,2 ${2 + s},${2 + s} ${2 + x0},${2 + s} 2,${2 + Math.min(bw, s)}" fill="var(--c-swa)" stroke="#fff" stroke-opacity=".5"/>
  </svg>`;
}
function glyphGdn(S = GLYPH) {
  // fixed-size memory tile; the arc arrow shows it being rewritten token by token. Never scales with ctx.
  const c = (S + 4) / 2, t = 26, x = c - t / 2, y = c - t / 2 - 3;
  let g = "";
  for (let i = 1; i < 4; i++) {
    g += `<line x1="${x + i * t / 4}" y1="${y}" x2="${x + i * t / 4}" y2="${y + t}" stroke="#fff" stroke-opacity=".35"/>`;
    g += `<line x1="${x}" y1="${y + i * t / 4}" x2="${x + t}" y2="${y + i * t / 4}" stroke="#fff" stroke-opacity=".35"/>`;
  }
  const R = t / 2 + 8, cx = c, cy = y + t / 2;
  const arc = `M ${cx - R} ${cy} A ${R} ${R} 0 1 1 ${cx} ${cy - R}`;
  let toks = "";
  for (let i = 0; i < 6; i++) toks += `<circle cx="${6 + i * 8.5}" cy="${S + 1}" r="2.2" fill="var(--dim)" opacity="${0.35 + i * 0.12}"/>`;
  return `<svg width="${S + 4}" height="${S + 4}" viewBox="0 0 ${S + 4} ${S + 4}" aria-hidden="true">
    <rect x="${x}" y="${y}" width="${t}" height="${t}" fill="var(--c-gdn)" stroke="#fff" stroke-opacity=".6" class="pulse"/>${g}
    <g class="upd"><path d="${arc}" fill="none" stroke="var(--gold)" stroke-width="2"/>
    <polygon points="${cx - 4},${cy - R - 4} ${cx + 4},${cy - R} ${cx - 4},${cy - R + 4}" fill="var(--gold)"/></g>
    ${toks}
  </svg>`;
}
function glyphMlp(d, hidden, h) {
  // expand-then-contract trapezoid pair: d -> hidden -> d
  const W = 150, c = h / 2, a = Math.max(5, h * 0.15);
  const b = clamp(a * hidden / d, a + 4, h / 2 - 2);
  return `<svg width="${W}" height="${h}" viewBox="0 0 ${W} ${h}" aria-hidden="true">
    <polygon points="2,${c - a} ${W / 2},${c - b} ${W - 2},${c - a} ${W - 2},${c + a} ${W / 2},${c + b} 2,${c + a}" fill="var(--c-mlp)" fill-opacity=".85" stroke="#fff" stroke-opacity=".5"/>
    <g font-family="JetBrains Mono, monospace" font-size="10" fill="#0a0d1c" font-weight="700" text-anchor="middle">
    <text x="16" y="${c + 3.5}">${d}</text><text x="${W / 2}" y="${c + 3.5}">${hidden}</text><text x="${W - 16}" y="${c + 3.5}">${d}</text></g>
  </svg>`;
}
const chainSvg = `<svg class="chain" width="16" height="14" viewBox="0 0 16 14" aria-label="shared weights"><title>shared weights</title>
  <rect x="1" y="4" width="9" height="6" rx="3" fill="none" stroke="var(--gold)" stroke-width="1.6"/>
  <rect x="6" y="4" width="9" height="6" rx="3" fill="none" stroke="var(--gold)" stroke-width="1.6"/></svg>`;

// ---------------------------------------------------------------- tooltips
const tips = [];
function tip(html) { tips.push(html); return tips.length - 1; }
const tipEl = $("#tip");
function showTip(html, x, y) {
  tipEl.innerHTML = html; tipEl.style.display = "block";
  const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  tipEl.style.left = clamp(x + 14, 6, window.innerWidth - w - 6) + "px";
  tipEl.style.top = clamp(y + 14, 6, window.innerHeight - h - 6) + "px";
}
const hideTip = () => { tipEl.style.display = "none"; };
document.addEventListener("mouseover", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(tips[+t.dataset.tip], e.clientX, e.clientY); else hideTip(); });
document.addEventListener("mousemove", (e) => { if (tipEl.style.display === "block") { const t = e.target.closest("[data-tip]"); if (t) showTip(tips[+t.dataset.tip], e.clientX, e.clientY); } });
document.addEventListener("click", (e) => { // touch: tap a block to toggle
  const t = e.target.closest("[data-tip]");
  if (t && tipEl.dataset.for !== t.dataset.tip) { tipEl.dataset.for = t.dataset.tip; const r = t.getBoundingClientRect(); showTip(tips[+t.dataset.tip], r.left + 20, r.top + 20); }
  else { tipEl.dataset.for = ""; hideTip(); }
});
document.addEventListener("focusin", (e) => { const t = e.target.closest && e.target.closest("[data-tip]"); if (t) { const r = t.getBoundingClientRect(); showTip(tips[+t.dataset.tip], r.left + 20, r.top + 10); } });
document.addEventListener("focusout", hideTip);
document.addEventListener("scroll", hideTip, { passive: true });

const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
function blockTip(b, D, ctx) {
  const bc = ctx, p = b.params, cfg = D.config, bl = b.bitlinear;
  const k = KIND[b.kind] || { name: b.kind };
  let h = `<h4>${esc(k.name)} block</h4><table>`;
  h += row(esc(b.section) + " #" + (b.index + 1), D.looped && b.section === "core" ? `runs ${D.loops}x, params counted once` : "");
  h += `<tr><td colspan="2" class="sep">${esc(b.mixer_type)}</td></tr>`;
  h += row("mixer params", fmtFull(p.mixer));
  for (const q of b.mixer_parts) h += row("&nbsp;&nbsp;" + esc(q.name) + (q.ternary ? " (1.58b)" : ""), fmtFull(q.params));
  h += `<tr><td colspan="2" class="sep">SwiGLU MLP ${b.mlp.d_in} &rarr; ${b.mlp.hidden} &rarr; ${b.mlp.d_in}</td></tr>`;
  h += row("MLP params", fmtFull(p.mlp));
  for (const q of b.mlp_parts) h += row("&nbsp;&nbsp;" + esc(q.name) + (q.ternary ? " (1.58b)" : ""), fmtFull(q.params));
  h += row("norms (2 x RMSNorm)", fmtFull(p.norms));
  h += row("block total", fmtFull(p.total));
  if (bl.count) h += row("BitLinear (ternary)", `${bl.count} mats, ${fmtFull(bl.params)} (${(100 * bl.params / p.total).toFixed(1)}%)`);
  h += `<tr><td colspan="2" class="sep">at ${fmtCtx(bc.ctx)} tokens</td></tr>`;
  if (b.kind === "attn") {
    h += row("score cells / layer", fmtFull(bc.ctx * (bc.ctx + 1) / 2) + " (n&sup2;/2)");
    h += row("KV cache / layer", fmtBytes(b.kv_elems_per_token * bc.ctx * 2) + ` (2d=${b.kv_elems_per_token}/tok)`);
  } else if (b.kind === "swa") {
    const w = Math.min(cfg.window, bc.ctx);
    h += row("score cells / layer", fmtFull(bc.ctx * w - w * (w - 1) / 2) + ` (~n&middot;w, w=${cfg.window})`);
    h += row("KV cache / layer", fmtBytes(b.kv_elems_per_token * w * 2) + " (capped at window)");
  } else if (b.kind === "gdn") {
    h += row("state / layer", fmtFull(b.state_elems) + " elems = " + fmtBytes(b.state_elems * 2));
    h += row("", esc(b.state_note) + ", flat in n");
  }
  return h + "</table>";
}

// ---------------------------------------------------------------- state
const S = { models: [], a: null, b: null, cmp: false, ctx: 0, loops: null, unroll: false, data: {}, tok: 0 };

// ---------------------------------------------------------------- panel rendering
function blockHtml(b, D, ctxRow, opts = {}) {
  const cfg = D.config, tern = D.ternary.enabled;
  const mh = Math.max(MIN_PART, Math.round(b.params.mixer / 1e6 * PX_PER_M));
  const lh = Math.max(MIN_PART, Math.round(b.params.mlp / 1e6 * PX_PER_M));
  const k = KIND[b.kind] || { name: b.kind, short: b.kind, cls: "attn" };
  const g = b.kind === "attn" ? glyphAttn(ctxRow.ctx, cfg.window) : b.kind === "swa" ? glyphSwa(ctxRow.ctx, cfg.window) : glyphGdn();
  const tipId = tip(blockTip(b, D, ctxRow));
  const mixTern = b.bitlinear.mixer_params > 0, mlpTern = b.bitlinear.mlp_params > 0;
  const gdnNote = b.kind === "gdn" ? `<span>state ${esc(b.state_note)} (fixed)</span>` :
    b.kind === "swa" ? `<span>window ${cfg.window} tokens</span>` : `<span>every token sees every earlier token</span>`;
  return `<div class="block ${opts.shared ? "shared" : ""}" tabindex="0" data-tip="${tipId}">
    <div class="bhead"><span class="dot" style="background:var(--c-${k.cls})"></span>${esc(k.short)}
      <span class="sub">${esc(b.section)} ${b.index + 1}${opts.shared ? "" : ""}</span>
      ${opts.shared ? `<span class="shr">${chainSvg}shared</span>` : ""}
      ${b.bitlinear.count ? `<span class="tbadge" title="ternary weights">1.58-bit</span>` : ""}</div>
    <div class="part mixer k-${k.cls} ${mixTern ? "tern" : ""}" style="min-height:${mh}px">${g}
      <div class="txt"><b>${esc(k.name)}</b>${gdnNote}<br><em>${fmtN(b.params.mixer)} params</em></div></div>
    <div class="normline" title="RMSNorm (${fmtFull(b.params.norms)} params)"></div>
    <div class="part mlp ${mlpTern ? "tern" : ""}" style="min-height:${lh}px">${glyphMlp(b.mlp.d_in, b.mlp.hidden, lh - 10)}
      <div class="txt"><b>SwiGLU MLP</b><span>${b.mlp.d_in} &rarr; ${b.mlp.hidden} &rarr; ${b.mlp.d_in}</span><br><em>${fmtN(b.params.mlp)} params</em></div></div>
  </div>`;
}

function injectHtml(inj, D) {
  const h = Math.max(MIN_PART - 14, Math.round(inj.params / 1e6 * PX_PER_M));
  const t = tip(`<h4>Re-injection</h4><table>${row("formula", esc(inj.formula))}${row("linear " + esc(inj.shape), fmtFull(inj.linear_params))}${row("2 x RMSNorm", fmtFull(inj.norms))}${row("total", fmtFull(inj.params))}${inj.bitlinear_params ? row("BitLinear (ternary)", fmtFull(inj.bitlinear_params)) : ""}</table>`);
  return `<div class="block" tabindex="0" data-tip="${t}"><div class="bhead"><span class="dot" style="background:var(--c-inj)"></span>RE-INJECT
    ${inj.bitlinear_params ? `<span class="tbadge">1.58-bit</span>` : ""}</div>
    <div class="part inj ${inj.bitlinear_params ? "tern" : ""}" style="min-height:${h}px"><div class="txt"><b>h + inject([norm h, norm e])</b><span>${esc(inj.shape)}: feeds the original input e back in each loop</span><br><em>${fmtN(inj.params)} params</em></div></div></div>`;
}

function chipHtml(b, i) {
  const k = KIND[b.kind] || { name: b.kind, cls: "attn" };
  return `<div class="chip"><i style="background:var(--c-${k.cls})"></i>${esc(k.name)} ${chainSvg} <span>same weights as loop 1</span><span class="nn">+0 params</span></div>`;
}

function sectionHtml(sec, D, ctxRow) {
  const label = { prelude: "Prelude: runs once", core: "Core: shared weights", coda: "Coda: runs once" }[sec.name] || sec.name;
  const blocks = sec.blocks;
  if (sec.name !== "core") {
    return `<div class="section"><div class="secttl"><span>${label}</span></div>${blocks.map((b) => blockHtml(b, D, ctxRow)).join("")}</div>`;
  }
  const r = D.loops;
  const inj = sec.injection ? injectHtml(sec.injection, D) : "";
  const coreTip = `${fmtN(blocks.reduce((s, b) => s + b.params.total, 0) + (sec.injection ? sec.injection.params : 0))} params stored once, applied ${r}x`;
  if (!S.unroll) {
    return `<div class="section core"><div class="secttl"><span>${label}</span><span>${coreTip}</span></div>
      <div class="coreRow"><div class="coreCol">${inj}${blocks.map((b) => blockHtml(b, D, ctxRow)).join("")}</div>
      <div class="loopbar spin" title="the core's output is fed back into its input r times"><b>&times; ${r}</b></div></div></div>`;
  }
  let out = `<div class="section core"><div class="secttl"><span>${label} (unrolled)</span><span>${coreTip}</span></div>`;
  for (let i = 0; i < r; i++) {
    out += `<div class="loopcopy"><div class="lc">${chainSvg} loop ${i + 1} of ${r} ${i ? "· shared weights, compute only" : "· the real weights"}</div>`;
    out += i === 0 ? inj + blocks.map((b) => blockHtml(b, D, ctxRow, { shared: true })).join("")
      : (sec.injection ? chipHtml({ kind: "inj" }, i) : "") + blocks.map((b) => chipHtml(b, i)).join("");
    out += `</div>`;
  }
  return out + `</div>`;
}

function liveRows(D, ci, c1) {
  const c = D.by_ctx[ci], z = D.by_ctx[0];
  const ap = D.applications;
  const rows = [];
  const add = (label, v, v0, fmt, show) => { if (show) rows.push({ label, v: fmt(v), x: v0 > 0 ? v / v0 : 1 }); };
  add("full-attn KV cache", c.attn_kv_bytes, z.attn_kv_bytes, fmtBytes, ap.attn > 0);
  add("SWA KV cache", c.swa_kv_bytes, z.swa_kv_bytes, fmtBytes, ap.swa > 0);
  add("GDN state", c.gdn_state_bytes, z.gdn_state_bytes, fmtBytes, ap.gdn > 0);
  add("attn score cells", c.attn_score_cells, z.attn_score_cells, fmtN, ap.attn > 0);
  add("train FLOPs/token", c.train_flops_per_token, z.train_flops_per_token, fmtFlops, true);
  return rows.map((r) => `<div class="row"><span>${r.label}</span><b>${r.v}</b><span class="${r.x > 1.05 ? "grow" : "flat"}">${r.x > 1.05 ? "x" + (r.x >= 10 ? r.x.toFixed(0) : r.x.toFixed(1)) + " vs 1k" : "flat"}</span></div>`).join("");
}

function panelHtml(D, label) {
  if (D.error) return `<div class="panel"><div class="ph"><h2>${esc(label)}</h2></div><div class="err">${esc(D.error)}\n${esc(D.trace || "")}</div></div>`;
  const ci = S.ctx, cr = D.by_ctx[ci];
  const reg = D.registry, P = D.params, T = D.ternary;
  const embTip = tip(`<h4>Embedding</h4><table>${row("vocab x d_model", `${D.embedding.vocab} x ${D.embedding.d_model}`)}${row("params", fmtFull(D.embedding.params))}${row("output head", D.embedding.tied ? "tied (same weights, counted once)" : "separate")}</table>`);
  const secs = D.sections.map((s) => sectionHtml(s, D, cr)).join(`<div class="arrow"></div>`);
  const kinds = [...new Set(D.sections.flatMap((s) => s.blocks.map((b) => b.kind)))];
  const mem = D.memory;
  return `<article class="panel">
    <div class="ph"><h2>${esc(D.experiment || D.name)}<small>${esc(reg && reg.title ? reg.title : D.name)}</small></h2>
      <span>${reg && reg.status ? `<span class="badge ${esc(reg.status)}">${esc(reg.status)}${reg.version ? " " + esc(reg.version) : ""}</span>` : ""}
      ${D.arch !== "fusiondance" ? `<span class="badge">${esc(D.arch)}</span>` : ""}</span></div>
    <p class="desc">${esc(D.description)}</p>
    ${D.notes.map((n) => `<div class="note">${esc(n)}</div>`).join("")}
    <div class="scouter">
      <div class="pl">POWER LEVEL</div>
      <div class="big">${fmtFull(P.total)}<small>params</small></div>
      <div class="sgrid">
        <div><span>non-embedding</span><b>${fmtFull(P.non_embedding)}</b></div>
        <div><span>embedding</span><b>${fmtFull(P.embedding)}</b></div>
        <div><span>eff. depth${D.looped ? " @r=" + D.loops : ""}</span><b>${D.effective_depth} blocks${D.looped ? ` (${D.distinct_blocks} distinct)` : ""}</b></div>
        <div><span>train FLOPs/tok @1k</span><b>${fmtFlops(D.flops.train_per_token)}</b></div>
        <div><span>weights bf16</span><b>${fmtBytes(mem.bf16_bytes)}</b></div>
        <div><span>weights ${mem.ternary_packed ? "2-bit packed" : "(no ternary)"}</span><b>${mem.ternary_packed ? fmtBytes(mem.packed_bytes) : "n/a"}</b></div>
        <div><span>ternary share</span><b>${T.enabled ? (100 * T.fraction_of_total).toFixed(1) + "% of total, " + T.bitlinear_count + " BitLinear" : "none (full precision)"}</b></div>
        ${reg && reg.val_loss != null ? `<div><span>val loss</span><b>${reg.val_loss}</b></div>` : `<div><span>val loss</span><b>not run yet</b></div>`}
      </div>
      <div class="live"><div class="row head"><span>AT ${fmtCtx(cr.ctx)} TOKENS</span><span></span><span></span></div>${liveRows(D, ci)}</div>
    </div>
    <div class="stack">
      <div class="endcap"><span>tokens</span><span>${kinds.map((k) => KIND[k] ? KIND[k].short : k).join(" · ")}</span></div><div class="arrow"></div>
      <div class="endcap emb" tabindex="0" data-tip="${embTip}"><span><b>Embedding</b> ${D.embedding.vocab} &times; ${D.embedding.d_model}</span><span>${fmtN(D.embedding.params)} params (drawn capped: scale break)</span></div>
      <div class="arrow"></div>
      ${secs}
      <div class="arrow"></div>
      <div class="endcap"><span><b>RMSNorm &rarr; output head</b></span><span>${D.embedding.tied ? "tied to embedding" : "own weights"}</span></div>
    </div>
  </article>`;
}

// ---------------------------------------------------------------- data + render
async function fetchModel(name) {
  const q = S.loops != null ? `?loops=${S.loops}` : "";
  try {
    const r = await fetch(`/api/model/${encodeURIComponent(name)}${q}`, { cache: "no-store" });
    return await r.json();
  } catch (e) { return { error: "request failed: " + e }; }
}
async function refresh() {
  const tok = ++S.tok;
  const names = [S.a, S.cmp ? S.b : null].filter(Boolean);
  const res = await Promise.all(names.map(fetchModel));
  if (tok !== S.tok) return;
  names.forEach((n, i) => { S.data[n] = res[i]; });
  if (S.loops == null) {
    const lp = res.find((d) => d && d.looped);
    if (lp) { S.loops = lp.loops_range.default; $("#loops").value = S.loops; $("#loopsv").textContent = S.loops; if (res.some((d) => d.looped && d.loops !== S.loops)) return refresh(); }
  }
  render();
}
function render() {
  tips.length = 0;
  const names = [S.a, S.cmp ? S.b : null].filter(Boolean);
  const ds = names.map((n) => S.data[n]).filter(Boolean);
  const anyLoop = ds.some((d) => d.looped);
  $("#ctlLoops").classList.toggle("dis", !anyLoop);
  $("#ctlUnroll").classList.toggle("dis", !anyLoop);
  $("#ctlB").style.display = S.cmp ? "" : "none";
  const el = $("#panels");
  el.className = "panels" + (S.cmp ? " two" : "");
  el.innerHTML = ds.map((d, i) => panelHtml(d, names[i])).join("");
  location.hash = `a=${S.a}&b=${S.cmp ? S.b : ""}&ctx=${S.ctx}${S.loops != null ? "&loops=" + S.loops : ""}${S.unroll ? "&unroll=1" : ""}`;
}

// ---------------------------------------------------------------- legend
function buildLegend() {
  const items = [
    [glyphAttn(16384, 256), "Full attention: a filled triangle", "Every token looks at every earlier token, so the score matrix is n x n (lower half). Double the context and the work and the KV cache go up 4x and 2x. That is why it gets expensive. Drag the context slider: the triangle grows with n."],
    [glyphSwa(16384, 256), "Sliding window: a diagonal band", "Each token only looks back w tokens (here 256). The band has fixed width, so cost grows linearly with length and the cache stops growing once it hits w."],
    [glyphGdn(), "Gated DeltaNet: a fixed memory tile", "One small square 'memory' per head that never grows with the text. The circling arrow means it is rewritten token by token. Cost is linear; state size is flat (heads x head_dim x head_dim)."],
    [glyphMlp(384, 1024, 56), "SwiGLU MLP: expand then contract", "Each token's vector is widened (d to hidden), mixed, and squeezed back (hidden to d). Pure per-token compute, no look-back."],
    [`<svg width="56" height="56"><rect x="2" y="2" width="52" height="52" rx="6" fill="var(--c-mlp)" fill-opacity=".25" stroke="var(--c-mlp)"/><rect x="2" y="2" width="52" height="52" rx="6" fill="url(#t)"/><defs><pattern id="t" width="24" height="16" patternUnits="userSpaceOnUse"><circle cx="4" cy="4" r="2.2" fill="#ebeeff"/><circle cx="15" cy="4" r="2" fill="none" stroke="#ebeeff"/><path d="M2 12h6" stroke="#ebeeff" stroke-width="1.6"/><circle cx="17" cy="12" r="2.2" fill="#ebeeff"/></pattern></defs></svg>`,
      "Ternary: -1 / 0 / +1 dots", "Weights restricted to three values (1.58 bits each), drawn as a dot texture over the affected projections, plus a gold '1.58-bit' badge. Embeddings, norms and the tiny GDN gates stay full precision."],
    [`<svg width="56" height="56"><path d="M12 46 L12 16 Q12 8 20 8 L44 8" fill="none" stroke="var(--gold)" stroke-width="3"/><polygon points="40,2 50,8 40,14" fill="var(--gold)"/><text x="20" y="40" font-family="Bangers" font-size="18" fill="var(--gold)">x r</text></svg>`,
      "Loop arrow: x r", "The core is stored once and run r times, feeding its output back to its input. Same weights, more compute. Move the loops slider."],
    [`<svg width="56" height="56">${"<g/>"}<rect x="4" y="6" width="48" height="16" rx="5" fill="none" stroke="var(--gold)" stroke-dasharray="4 3"/><rect x="4" y="34" width="48" height="16" rx="5" fill="none" stroke="var(--gold)" stroke-dasharray="4 3"/><g transform="translate(20 26)">${chainSvg.replace(/<svg[^>]*>|<\/svg>/g, "").replace("<title>shared weights</title>", "")}</g></svg>`,
      "Unroll + chain icon", "Tick 'unroll the loop' to draw the core r times. Gold dashed outline and the chain icon mean 'same weights as loop 1': depth comes from compute, not new parameters."],
    [`<svg width="56" height="56"><rect x="6" y="8" width="44" height="14" fill="var(--c-attn)"/><rect x="6" y="26" width="44" height="22" fill="var(--c-mlp)"/><text x="8" y="58" font-size="9" fill="var(--muted)" font-family="JetBrains Mono">size~params</text></svg>`,
      "Size = parameter count", `Heights are proportional to parameters: 1,000,000 params = ${PX_PER_M}px. Parts under ${MIN_PART}px are drawn at the minimum so the glyph fits. The embedding is capped (striped scale break); its true count is in its label.`],
    [`<svg width="56" height="56"><rect x="6" y="6" width="44" height="12" fill="var(--c-attn)"/><rect x="6" y="22" width="44" height="12" fill="var(--c-swa)"/><rect x="6" y="38" width="44" height="12" fill="var(--c-gdn)"/></svg>`,
      "Colour = block kind", "Blue: full attention. Green: sliding window. Orange: Gated DeltaNet. Yellow: MLP. Grey-violet: re-injection. Shape and label always say the same thing, so colour is never the only cue."],
  ];
  $("#legend").innerHTML = items.map(([svg, t, d]) => `<div class="li">${svg}<div><b>${t}</b>${d}</div></div>`).join("");
}

// ---------------------------------------------------------------- wiring
function opt(sel, models, val) {
  sel.innerHTML = models.map((m) => `<option value="${esc(m.name)}">${esc((m.experiment || "?") + "  " + (m.title || m.name) + (m.status ? "  [" + m.status + "]" : ""))}</option>`).join("");
  sel.value = val;
}
let loopTimer = 0;
async function init() {
  buildLegend();
  let list = [];
  try { list = (await (await fetch("/api/models", { cache: "no-store" })).json()).models.filter((m) => !m.error); } catch (e) { $("#panels").innerHTML = `<div class="panel err">cannot reach /api/models: ${esc(e)}</div>`; return; }
  S.models = list;
  const h = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  const has = (n) => list.some((m) => m.name === n);
  S.a = has(h.a) ? h.a : (list.find((m) => m.experiment === "E1") || list[0]).name;
  S.b = has(h.b) ? h.b : (list.find((m) => m.experiment === "E2") || list[list.length - 1]).name;
  S.cmp = !!h.b && has(h.b);
  S.ctx = clamp(parseInt(h.ctx || "0", 10) || 0, 0, 5);
  S.loops = h.loops ? clamp(parseInt(h.loops, 10), 1, 12) : null;
  S.unroll = h.unroll === "1";
  opt($("#selA"), list, S.a); opt($("#selB"), list, S.b);
  $("#cmp").checked = S.cmp; $("#ctx").value = S.ctx; $("#unroll").checked = S.unroll;
  $("#ctxv").textContent = fmtCtx(1024 * 2 ** S.ctx);
  if (S.loops != null) { $("#loops").value = S.loops; $("#loopsv").textContent = S.loops; }
  $("#selA").onchange = (e) => { S.a = e.target.value; refresh(); };
  $("#selB").onchange = (e) => { S.b = e.target.value; refresh(); };
  $("#cmp").onchange = (e) => { S.cmp = e.target.checked; refresh(); };
  $("#ctx").oninput = (e) => { S.ctx = +e.target.value; $("#ctxv").textContent = fmtCtx(1024 * 2 ** S.ctx); render(); };
  $("#unroll").onchange = (e) => { S.unroll = e.target.checked; render(); };
  $("#loops").oninput = (e) => { S.loops = +e.target.value; $("#loopsv").textContent = S.loops; clearTimeout(loopTimer); loopTimer = setTimeout(refresh, 120); };
  refresh();
}
init();
})();
