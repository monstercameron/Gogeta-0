// Gogeta-0 · 3D architecture viewer.
// Data: data/architectures.json (exported from the real model code by tools/viz/export_arch.py).
// three.js is pinned to 0.160.0 via the import map in architecture.html. No other network dependencies.
//
// What each shape means (all sizes come from the exported numbers):
//   plinth under a mixer ........ height = mixer parameters (1M params = U_PARAM units)
//   full-attention wedge ........ lower-triangle n x n score matrix; side = S_CTX * n / 1024, so area ~ n^2
//   sliding-window band ......... diagonal band, fixed width (window w), only its length grows with n
//   Gated DeltaNet cube ......... fixed-size memory (heads x head_dim^2); never grows with n
//   MLP bicone .................. hexagonal d -> hidden -> d; height = MLP parameters
//   ring + ghost pillars ........ looped core stored once, run r times
//   speckles .................... ternary BitLinear weights (-1 / 0 / +1)
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";

// ------------------------------------------------------------------ constants
const U_PARAM = 1.0;      // scene units of height per 1M parameters
const S_CTX = 0.4;        // wedge side length (units) per 1024 tokens of context
const SLAB_T = 0.09;      // thickness of the attention slabs (cost shapes carry no parameter meaning)
const PLINTH = 1.25;      // plinth footprint
const CUBE_REF = 0.9;     // GDN cube edge for the reference state size (24,576 values)
const CUBE_REF_ELEMS = 24576;
const BICONE_R0 = 0.28;   // bicone radius at d; widens linearly with hidden/d
const GAP = 0.35;
const BYTES = 2;          // bf16
const COL = { attn: 0x3987e5, swa: 0x199e70, gdn: 0xd95926, mlp: 0xc98500, inj: 0x9a8fc8, emb: 0x6f7bb8, gold: 0xffd23f, orange: 0xff8a1f };
const KIND = {
  attn: { name: "Full attention", short: "ATTN", css: "#3987e5" },
  swa: { name: "Sliding-window attention", short: "SWA", css: "#199e70" },
  gdn: { name: "Gated DeltaNet", short: "GDN", css: "#d95926" },
};
const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("en-US");
const fmtB = (b) => (b >= 1048576 ? (b / 1048576).toFixed(2) + " MiB" : b >= 1024 ? (b / 1024).toFixed(1) + " KiB" : fmt(b) + " B");
const fmtF = (f) => (f >= 1e9 ? (f / 1e9).toFixed(2) + " GFLOPs" : (f / 1e6).toFixed(1) + " MFLOPs");
const ctxLabel = (n) => (n >= 1024 ? n / 1024 + "k" : String(n));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ state
const S = { a: null, b: null, compare: false, ctxIdx: 2, loops: 4, unroll: true, motion: !REDUCED };
let DATA = null, CTX = [1024, 2048, 4096, 8192, 16384, 32768];
let renderer, labelRenderer, scene, camera, controls, stage, raycaster;
let models = [];           // built model objects currently in the scene
let pickables = [];
let stars, dirty = true, hoverTip = null;
const geoCache = new Map(), edgeCache = new Map(), matCache = new Map(), texCache = new Map();
let glowTex = null;
const ctxN = () => CTX[S.ctxIdx];
const wedgeSide = (n) => (S_CTX * n) / 1024;

// ------------------------------------------------------------------ boot
init().catch((e) => {
  console.error(e);
  showNoGL("Something went wrong starting the 3D view: " + (e && e.message ? e.message : e));
});

function showNoGL(msg) {
  const n = $("nogl");
  if (n) { n.hidden = false; if (msg && $("noglMsg")) $("noglMsg").textContent = msg + " The numbers are still on the site:"; }
  document.querySelectorAll("#controls select, #controls input, #controls button").forEach((e) => (e.disabled = true));
}

function webglOK() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (e) { return false; }
}

async function init() {
  stage = $("stage");
  if (!webglOK()) { showNoGL("This browser cannot create a WebGL context."); window.__arch3dReady = true; return; }
  const res = await fetch("data/architectures.json", { cache: "no-cache" });
  if (!res.ok) throw new Error("architectures.json: HTTP " + res.status);
  DATA = await res.json();
  CTX = DATA.ctx_stops || CTX;
  $("stamp").textContent = `${DATA.generated_at} · ${DATA.git}`;

  // renderer
  const canvas = $("gl");
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch (e) { showNoGL("WebGL could not be started."); window.__arch3dReady = true; return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "labels";
  stage.appendChild(labelRenderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 1, 0.1, 1200);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = !REDUCED;
  controls.dampingFactor = 0.08;
  controls.minDistance = 3;
  controls.maxDistance = 220;
  controls.addEventListener("change", () => (dirty = true));
  raycaster = new THREE.Raycaster();

  scene.add(new THREE.HemisphereLight(0xb8c4ff, 0x2a1a10, 0.9));
  const sun = new THREE.DirectionalLight(0xfff0d0, 1.2);
  sun.position.set(6, 12, 8);
  scene.add(sun);
  glowTex = makeGlowTexture();
  stars = makeStars();
  scene.add(stars);

  // UI
  const names = DATA.models.map((m) => m.name);
  for (const sel of [$("selA"), $("selB")]) {
    sel.innerHTML = DATA.models
      .map((m) => `<option value="${esc(m.name)}">${esc(m.experiment || "")} · ${esc(m.name)}${m.title ? " · " + esc(m.title) : ""}</option>`)
      .join("");
  }
  const want = new URLSearchParams(location.search).get("model");
  S.a = names.includes(want) ? want : names.includes("e2_looped") ? "e2_looped" : names[0];
  S.b = names.includes("e1_hybrid") && S.a !== "e1_hybrid" ? "e1_hybrid" : names.find((n) => n !== S.a) || S.a;
  $("selA").value = S.a;
  $("selB").value = S.b;
  const first = DATA.models.find((m) => m.name === S.a);
  S.loops = first.looped ? first.loops : 4;
  $("rngLoops").value = S.loops;
  $("rngCtx").value = S.ctxIdx;

  $("selA").addEventListener("change", () => { S.a = $("selA").value; syncLoopsDefault(S.a); rebuild(true); });
  $("selB").addEventListener("change", () => { S.b = $("selB").value; rebuild(true); });
  $("chkCompare").addEventListener("change", () => { S.compare = $("chkCompare").checked; $("ctlB").hidden = !S.compare; rebuild(true); });
  $("rngCtx").addEventListener("input", () => { S.ctxIdx = +$("rngCtx").value; applyCtx(); });
  $("rngLoops").addEventListener("input", () => { S.loops = +$("rngLoops").value; rebuild(false); });
  $("chkUnroll").addEventListener("change", () => { S.unroll = $("chkUnroll").checked; rebuild(false); });
  $("btnReset").addEventListener("click", () => fitCamera());
  const bm = $("btnMotion");
  const paintMotion = () => { bm.textContent = "Motion: " + (S.motion ? "on" : "off"); bm.setAttribute("aria-pressed", String(S.motion)); controls.enableDamping = S.motion; };
  bm.addEventListener("click", () => { S.motion = !S.motion; paintMotion(); dirty = true; });
  paintMotion();

  // picking
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointerleave", () => { if (lastPointerType === "mouse") hideTip(); });

  new ResizeObserver(resize).observe(stage);
  resize();
  rebuild(true);
  requestAnimationFrame(frame);
  window.__arch3dReady = true;
  exposeTestHook();
}

function syncLoopsDefault(name) {
  const m = DATA.models.find((x) => x.name === name);
  if (m && m.looped) { S.loops = m.loops; $("rngLoops").value = S.loops; }
}

function resize() {
  const w = Math.max(1, stage.clientWidth), h = Math.max(1, stage.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  dirty = true;
}

// ------------------------------------------------------------------ textures / materials / geometry cache
function makeGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.25, "rgba(255,255,255,.55)");
  grd.addColorStop(0.6, "rgba(255,255,255,.12)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function ternaryTexture(hex) {
  const key = "t" + hex;
  if (texCache.has(key)) return texCache.get(key);
  const cells = 8, px = 12, c = document.createElement("canvas");
  c.width = c.height = cells * px;
  const g = c.getContext("2d");
  const base = "#" + new THREE.Color(hex).multiplyScalar(0.55).getHexString();
  g.fillStyle = base;
  g.fillRect(0, 0, c.width, c.height);
  let seed = 1234567; // deterministic speckle
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const v = rnd();
    g.fillStyle = v < 0.34 ? "#4a90ff" : v < 0.68 ? "#10142a" : "#ff8a1f"; // -1 blue, 0 dark, +1 orange
    g.fillRect(x * px + 2, y * px + 2, px - 4, px - 4);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.NearestFilter;
  t.repeat.set(1, 1);
  texCache.set(key, t);
  return t;
}

function mat(hex, { ternary = false, ghost = false, opacity = 1 } = {}) {
  const key = `${hex}|${ternary}|${ghost}|${opacity}`;
  if (matCache.has(key)) return matCache.get(key);
  let m;
  if (ghost) {
    m = new THREE.MeshBasicMaterial({ color: hex, transparent: true, opacity: 0.17, depthWrite: false });
  } else if (ternary) {
    const t = ternaryTexture(hex);
    m = new THREE.MeshStandardMaterial({ color: 0xffffff, map: t, emissive: 0xffffff, emissiveMap: t, emissiveIntensity: 0.45, roughness: 0.55, metalness: 0.05 });
  } else {
    m = new THREE.MeshStandardMaterial({ color: hex, emissive: hex, emissiveIntensity: 0.28, roughness: 0.5, metalness: 0.1, transparent: opacity < 1, opacity });
  }
  matCache.set(key, m);
  return m;
}

function geo(key, make) {
  let g = geoCache.get(key);
  if (!g) { g = make(); geoCache.set(key, g); }
  return g;
}
const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28 });
const edgeMatGhost = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12 });
function withEdges(mesh, ghost) {
  const g = mesh.geometry;
  let e = edgeCache.get(g.uuid);
  if (!e) { e = new THREE.EdgesGeometry(g, 20); edgeCache.set(g.uuid, e); }
  mesh.add(new THREE.LineSegments(e, ghost ? edgeMatGhost : edgeMat));
  return mesh;
}

const unitBox = () => geo("box1", () => new THREE.BoxGeometry(1, 1, 1));
const wedgeGeo = () => geo("wedge", () => {
  const sh = new THREE.Shape();
  sh.moveTo(-0.5, -0.5); sh.lineTo(0.5, -0.5); sh.lineTo(0.5, 0.5); sh.closePath();
  const g = new THREE.ExtrudeGeometry(sh, { depth: SLAB_T, bevelEnabled: false });
  g.rotateX(-Math.PI / 2); // lie in the XZ plane, extrude upward
  return g;
});
const bandGeo = () => geo("band", () => { const g = new THREE.BoxGeometry(1, SLAB_T, 1); g.translate(0, SLAB_T / 2, 0); return g; });
const biconeGeo = (h, ratio) => geo(`bi${h.toFixed(3)}|${ratio.toFixed(3)}`, () => {
  const r0 = BICONE_R0, r1 = BICONE_R0 * Math.min(ratio, 4);
  const pts = [[0, 0], [r0, 0], [r1, h / 2], [r0, h], [0, h]].map((p) => new THREE.Vector2(p[0], p[1]));
  const g = new THREE.LatheGeometry(pts, 6); // 6 segments: hexagonal
  g.rotateY(Math.PI / 6);
  return g;
});

function sprite(color, scale, opacity) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
  s.scale.set(scale, scale, 1);
  s.renderOrder = 1;
  return s;
}

function makeStars() {
  const n = 900, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let seed = 99;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < n; i++) {
    const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, r = 260 + rnd() * 200, s = Math.sqrt(1 - u * u);
    pos.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
    const t = rnd();
    const c = t < 0.2 ? [1, 0.82, 0.4] : t < 0.4 ? [0.6, 0.75, 1] : [1, 1, 1];
    col.set(c, i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const p = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.8, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false }));
  p.userData.isStars = true;
  return p;
}

function label(text, cls, pos, css) {
  const el = document.createElement("div");
  el.className = "tag " + (cls || "");
  el.textContent = text;
  if (css) el.style.setProperty("--c", css);
  const o = new CSS2DObject(el);
  o.position.copy(pos);
  return o;
}

// ------------------------------------------------------------------ tooltips (HTML strings)
const row = (k, v) => `<span class="k">${k}</span> ${v}<br>`;
const partsStr = (parts) => parts.map((p) => `${esc(p.name)} ${fmt(p.params)}${p.ternary ? "*" : ""}`).join(" · ");
function blockTitle(b, d, copy) {
  const loc = b.section === "core" ? `core block ${b.index + 1}` : `${b.section} block ${b.index + 1}`;
  return `<b>${KIND[b.kind].name}</b> · ${loc}${copy ? ` · copy ${copy.k + 1} of ${copy.r}` : ""}<br>`;
}
function tipWeights(d, b, copy) {
  let h = blockTitle(b, d, copy);
  if (copy && copy.k > 0) h += `<span class="sub">A ghost copy: it reuses the weights of the real block, so it adds 0 parameters and costs another pass of compute.</span>`;
  h += row("mixer params", fmt(b.params.mixer));
  h += `<span class="sub">${partsStr(b.mixer_parts)}${b.bitlinear.mixer_params ? "<br>* ternary (BitLinear)" : ""}</span>`;
  h += row("norms", fmt(b.params.norms)) + row("block total", fmt(b.params.total));
  if (b.bitlinear.params) h += row("ternary", `${fmt(b.bitlinear.params)} weights (${((100 * b.bitlinear.params) / b.params.total).toFixed(1)}% of block), 1.58 bit`);
  h += `<span class="sub">Plinth height = mixer parameters (${(b.params.mixer / 1e6).toFixed(3)}M = ${(b.params.mixer / 1e6 * U_PARAM).toFixed(2)} units)</span>`;
  return h;
}
function tipShape(d, b, copy) {
  const n = ctxN(), c = d.config, W = c.window, dm = c.d_model;
  let h = blockTitle(b, d, copy);
  if (b.kind === "attn") {
    const cells = (n * (n + 1)) / 2;
    h += row("context n", fmt(n)) + row("score cells n(n+1)/2", fmt(cells)) + row("KV cache", fmtB(2 * dm * n * BYTES) + " (2·d·n·2B)");
    h += `<span class="sub">Footprint side ∝ n, so area ∝ n². At 32k it is 1024× the 1k area.</span>`;
  } else if (b.kind === "swa") {
    const w = Math.min(W, n), cells = n * w - (w * (w - 1)) / 2;
    h += row("window w", fmt(W)) + row("context n", fmt(n)) + row("score cells", fmt(cells)) + row("KV cache", fmtB(2 * dm * w * BYTES) + " (capped at w)");
    h += `<span class="sub">Fixed width, only the length grows: cost ∝ n·w.</span>`;
  } else {
    h += row("state", `${b.state_note} = ${fmt(b.state_elems)} values`) + row("state memory", fmtB(b.state_elems * BYTES));
    h += `<span class="sub">Constant at any context length (now ${fmt(n)}): the cube does not grow.</span>`;
  }
  return h;
}
function tipMLP(d, b, copy) {
  let h = blockTitle(b, d, copy).replace("<br>", "") + `<br><b>SwiGLU MLP</b> ${b.mlp.d_in} → ${b.mlp.hidden} → ${b.mlp.d_in}<br>`;
  h += row("mlp params", fmt(b.params.mlp));
  h += `<span class="sub">${partsStr(b.mlp_parts)}${b.bitlinear.mlp_params ? "<br>* ternary (BitLinear)" : ""}</span>`;
  h += `<span class="sub">Height = MLP parameters (${(b.params.mlp / 1e6 * U_PARAM).toFixed(2)} units). Per token, no look-back.</span>`;
  return h;
}

// ------------------------------------------------------------------ building one model
function buildTile(d, b, ghost, copy) {
  const g = new THREE.Group();
  const tern = d.ternary.enabled;
  const mixTern = tern && b.bitlinear.mixer_params > 0, mlpTern = tern && b.bitlinear.mlp_params > 0;
  const hP = Math.max(0.12, (b.params.mixer / 1e6) * U_PARAM);
  const hM = Math.max(0.3, (b.params.mlp / 1e6) * U_PARAM);
  const ratio = b.mlp.hidden / b.mlp.d_in;
  const kcol = COL[b.kind];
  const out = { group: g, wedges: [], pick: [], gdn: null };
  const add = (mesh, tip, colorCss) => {
    mesh.userData.tip = tip;
    mesh.userData.color = colorCss;
    g.add(mesh);
    out.pick.push(mesh);
    return mesh;
  };
  // weights plinth
  const plinth = new THREE.Mesh(geo(`pl${hP.toFixed(3)}`, () => new THREE.BoxGeometry(PLINTH, hP, PLINTH)),
    ghost ? mat(kcol, { ghost: true }) : mat(COL.emb, { ternary: mixTern }));
  plinth.position.y = hP / 2;
  if (!ghost && !mixTern) plinth.material = mat(kcol, { opacity: 1 });
  withEdges(plinth, ghost);
  add(plinth, () => tipWeights(d, b, copy), KIND[b.kind].css);

  // cost shape on top of the plinth
  let shapeH;
  if (b.kind === "gdn") {
    const edge = CUBE_REF * Math.cbrt(b.state_elems / CUBE_REF_ELEMS);
    shapeH = edge + 0.04;
    const cube = new THREE.Mesh(unitBox(), ghost ? mat(kcol, { ghost: true }) : mat(kcol));
    cube.scale.set(edge, edge, edge);
    cube.position.y = hP + 0.02 + edge / 2;
    withEdges(cube, ghost);
    add(cube, () => tipShape(d, b, copy), KIND.gdn.css);
    out.gdn = { y: hP + 0.02 + edge / 2, edge };
    if (!ghost) { const sp = sprite(COL.orange, edge * 2.6, 0.45); sp.position.copy(cube.position); g.add(sp); }
  } else if (b.kind === "attn") {
    shapeH = SLAB_T + 0.04;
    const m = new THREE.Mesh(wedgeGeo(), ghost ? mat(kcol, { ghost: true }) : mat(kcol, { opacity: 0.88 }));
    m.position.y = hP + 0.02;
    withEdges(m, ghost);
    add(m, () => tipShape(d, b, copy), KIND.attn.css);
    out.wedges.push({ mesh: m, type: "attn", ghost });
  } else {
    shapeH = SLAB_T + 0.04;
    const pivot = new THREE.Group();
    pivot.rotation.y = (3 * Math.PI) / 4; // local +z runs along the footprint diagonal
    pivot.position.y = hP + 0.02;
    const m = new THREE.Mesh(bandGeo(), ghost ? mat(kcol, { ghost: true }) : mat(kcol, { opacity: 0.92 }));
    withEdges(m, ghost);
    pivot.add(m);
    g.add(pivot);
    add(m, () => tipShape(d, b, copy), KIND.swa.css);
    out.wedges.push({ mesh: m, type: "swa", ghost, window: d.config.window });
  }
  // MLP bicone
  const base = hP + shapeH + 0.18;
  const bi = new THREE.Mesh(biconeGeo(hM, ratio), ghost ? mat(COL.mlp, { ghost: true }) : mat(COL.mlp, { ternary: mlpTern }));
  bi.material.flatShading = true;
  bi.position.y = base;
  withEdges(bi, ghost);
  add(bi, () => tipMLP(d, b, copy), "#c98500");
  out.height = base + hM;
  out.hP = hP;
  if (!ghost) {
    const lab = label(KIND[b.kind].short, "kind", new THREE.Vector3(PLINTH * 0.62, hP / 2, 0), KIND[b.kind].css);
    g.add(lab);
  }
  return out;
}

function buildModel(d, slotLoops) {
  const M = { d, grp: new THREE.Group(), wedges: [], pick: [], gdnSites: [], r: 1, ringR: 0, height: 0, ghostTiles: 0, realTiles: 0 };
  const g = M.grp;
  const cfg = d.config, dm = cfg.d_model;
  const looped = d.looped;
  const r = looped ? slotLoops : 1;
  M.r = r;
  const unroll = looped && S.unroll && r > 1;
  const secs = Object.fromEntries(d.sections.map((s) => [s.name, s]));
  const pre = secs.prelude?.blocks || [], core = secs.core?.blocks || [], coda = secs.coda?.blocks || [];
  const place = (blocks, y0) => {
    let y = y0;
    for (const b of blocks) {
      const t = buildTile(d, b, false, null);
      t.group.position.set(0, y, 0);
      g.add(t.group);
      M.wedges.push(...t.wedges); M.pick.push(...t.pick);
      if (t.gdn) M.gdnSites.push(new THREE.Vector3(0, y + t.gdn.y, 0));
      M.realTiles++;
      y += t.height + GAP;
    }
    return y;
  };
  const note = (txt, cls, x, y, z, css) => { const l = label(txt, cls, new THREE.Vector3(x, y, z), css); g.add(l); return l; };

  // embedding (height capped at 1.6 with scale-break rings; the tooltip has the true count)
  const hE = Math.min(1.6, Math.max(0.4, (d.embedding.params / 1e6) * U_PARAM));
  const emb = new THREE.Mesh(geo(`emb${hE.toFixed(2)}`, () => new THREE.CylinderGeometry(1.15, 1.15, hE, 40)), mat(COL.emb));
  emb.position.y = hE / 2;
  withEdges(emb, false);
  emb.userData.tip = () => `<b>Embedding</b> (token → vector)<br>${row("params", fmt(d.embedding.params))}${row("shape", `${fmt(d.embedding.vocab)} × ${d.embedding.d_model}`)}<span class="sub">Full precision${d.embedding.tied ? "; tied with the output head (counted once)" : ""}.<br>Drawn capped at ${(1.6 / U_PARAM).toFixed(1)}M-equivalent height (scale-break rings).</span>`;
  emb.userData.color = "#6f7bb8";
  g.add(emb); M.pick.push(emb);
  if (d.embedding.params / 1e6 * U_PARAM > 1.6) {
    for (const f of [0.3, 0.7]) {
      const ring = new THREE.Mesh(geo("brk", () => new THREE.TorusGeometry(1.16, 0.045, 8, 40)), new THREE.MeshBasicMaterial({ color: 0x0a0d1c }));
      ring.rotation.x = Math.PI / 2; ring.position.y = hE * f; g.add(ring);
    }
  }
  note("EMBED", "sect", 0, -0.45, 1.5);
  let y = hE + 0.7;

  // prelude
  const yPre0 = y;
  if (pre.length) { y = place(pre, y); note("PRELUDE", "sect", -(PLINTH + 1.15), yPre0 + 0.4, 0); }

  // core: the Time Chamber
  let entryY = null, exitY = null, loopPts = null;
  const tilesMax = core.length ? 0 : 0;
  if (core.length) {
    const n = core.length;
    const R = Math.max(3, 1.1 * n + 1);
    M.ringR = R;
    const base = y + 0.5;
    // tile heights (all tiles share the ring's base)
    const built = core.map((b) => buildTile(d, b, false, null));
    const tH = Math.max(...built.map((t) => t.height));
    const pitch = tH + 0.55;
    const phi = (i) => Math.PI / n + (2 * Math.PI * i) / n;
    built.forEach((t, i) => {
      const a = phi(i);
      t.group.position.set(R * Math.cos(a), base, R * Math.sin(a));
      t.group.rotation.y = -a;
      g.add(t.group);
      M.wedges.push(...t.wedges); M.pick.push(...t.pick);
      if (t.gdn) M.gdnSites.push(new THREE.Vector3(R * Math.cos(a), base + t.gdn.y, R * Math.sin(a)));
      M.realTiles++;
    });
    // ring, floor, aura
    const ring = new THREE.Mesh(geo(`ring${R.toFixed(2)}`, () => new THREE.TorusGeometry(R, 0.07, 10, 96)), new THREE.MeshBasicMaterial({ color: COL.gold }));
    ring.rotation.x = Math.PI / 2; ring.position.y = base - 0.02;
    ring.userData.tip = () => `<b>The Time Chamber: looped core</b><br>${row("blocks stored", `${n} (${core.map((b) => KIND[b.kind].short).join(" · ")})`)}${row("loops r", r)}${row("core params", fmt(core.reduce((a, b) => a + b.params.total, 0)) + " (stored once)")}${row("effective depth", `${d.by_loops ? d.by_loops[r - 1].effective_depth : d.effective_depth} layers`)}<span class="sub">The same weights run r times: depth comes from compute, not new parameters.</span>`;
    ring.userData.color = "#ffd23f";
    g.add(ring); M.pick.push(ring);
    const floor = new THREE.Mesh(geo(`floor${R.toFixed(2)}`, () => new THREE.CircleGeometry(R, 64)), new THREE.MeshBasicMaterial({ color: COL.gold, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = base - 0.03; g.add(floor);
    const aura = sprite(COL.orange, R * 4.4, 0.32); aura.position.set(0, base + tH * 0.4, 0); g.add(aura); M.aura = aura;
    // re-inject disc at the centre
    const inj = d.sections.find((s) => s.name === "core").injection;
    const hI = Math.max(0.14, (inj.params / 1e6) * U_PARAM);
    const injM = new THREE.Mesh(geo(`inj${hI.toFixed(2)}`, () => new THREE.CylinderGeometry(0.3, 0.6, hI, 24)), mat(COL.inj));
    injM.position.set(0, base + hI / 2, 0); withEdges(injM, false);
    injM.userData.tip = () => `<b>Re-inject</b> (every loop)<br>${row("params", fmt(inj.params))}${row("shape", inj.shape)}<span class="sub">${esc(inj.formula)}<br>The original input e is mixed back in at the start of each pass, so the loop never forgets the prompt.</span>`;
    injM.userData.color = "#9a8fc8";
    g.add(injM); M.pick.push(injM);
    note("RE-INJECT", "kind", 0, base + hI + 0.4, 0, "#9a8fc8");
    // e-beam from the embedding to the disc (dashed violet)
    const bx = PLINTH * 0.5 + 0.9;
    const beam = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(bx, hE * 0.5, 0), new THREE.Vector3(bx, base + 0.2, 0), new THREE.Vector3(0.45, base + 0.2, 0)]),
      new THREE.LineDashedMaterial({ color: COL.inj, dashSize: 0.22, gapSize: 0.14 }));
    beam.computeLineDistances(); g.add(beam);

    // unrolled helix of ghost copies (same angles, stacked one pitch apart)
    let top = base + tH;
    if (unroll) {
      for (let k = 1; k < r; k++) {
        core.forEach((b, i) => {
          const t = buildTile(d, b, true, { k, r });
          const a = phi(i);
          t.group.position.set(R * Math.cos(a), base + k * pitch, R * Math.sin(a));
          t.group.rotation.y = -a;
          g.add(t.group);
          M.wedges.push(...t.wedges); M.pick.push(...t.pick);
          M.ghostTiles++;
        });
      }
      top = base + (r - 1) * pitch + tH;
      // shared-weights markers: dashed gold lines linking each real block to its copies
      core.forEach((b, i) => {
        const a = phi(i), rr = R + 1.0;
        const ln = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(rr * Math.cos(a), base, rr * Math.sin(a)), new THREE.Vector3(rr * Math.cos(a), top, rr * Math.sin(a))]),
          new THREE.LineDashedMaterial({ color: COL.gold, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.8 }));
        ln.computeLineDistances(); g.add(ln);
      });
      const a0 = phi(0), rr = R + 1.0;
      note(`SAME WEIGHTS × ${r}  ·  0 new params`, "shared", rr * Math.cos(a0) * 1.0, (base + top) / 2, rr * Math.sin(a0) * 1.0);
      // helix tube = the path of the computation
      const hp = [];
      const M2 = 48;
      for (let k = 0; k < r; k++) for (let j = 0; j <= M2; j++) {
        const t = (k * M2 + j) / (r * M2);
        const ang = (2 * Math.PI * (k * M2 + j)) / M2;
        hp.push(new THREE.Vector3(R * Math.cos(ang), base + 0.35 + (k + j / M2) * pitch, R * Math.sin(ang)));
      }
      const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(hp), r * 64, 0.03, 6, false), new THREE.MeshBasicMaterial({ color: COL.orange, transparent: true, opacity: 0.55 }));
      g.add(tube);
    }
    note(`CORE × ${r}`, "loop", -(R + 1.4), base + tH / 2, 0);
    entryY = base + 0.35;
    exitY = unroll ? base + 0.35 + (r - 1) * pitch + pitch * 1 : base + 0.35;
    loopPts = { R, base, pitch, unroll, r };
    y = top + 0.7;
  }

  // coda
  const yCoda0 = y;
  if (coda.length) { y = place(coda, y); note("CODA", "sect", -(PLINTH + 1.15), yCoda0 + 0.4, 0); }

  // head
  const hH = 0.5;
  const head = new THREE.Mesh(geo("head", () => new THREE.CylinderGeometry(0.3, 1.0, hH, 24)), mat(COL.emb));
  head.position.y = y + hH / 2; withEdges(head, false);
  head.userData.tip = () => `<b>Output head</b> (vector → next-token scores)<br>${d.embedding.tied ? "Tied to the embedding: 0 extra parameters." : "Separate output matrix."}`;
  head.userData.color = "#6f7bb8";
  g.add(head); M.pick.push(head);
  note("NEXT TOKEN", "sect", 0, y + hH + 0.45, 0);
  y += hH;
  M.height = y;

  // ternary badge + title
  const titleY = y + 1.5;
  note(`${d.experiment || ""} · ${d.name}`, "title", 0, titleY, 0);
  if (d.ternary.enabled) note(`1.58-bit ternary · ${(100 * d.ternary.fraction_of_total).toFixed(0)}% of weights`, "badge", 0, titleY - 0.8, 0);

  // overall aura
  const big = sprite(COL.orange, Math.max(y * 1.1, 8), 0.14); big.position.set(0, y * 0.5, 0); g.add(big);

  // ki particle path
  const pts = [new THREE.Vector3(0, hE * 0.5, 0)];
  if (loopPts) {
    const { R, base, pitch, unroll: un, r: rr } = loopPts;
    pts.push(new THREE.Vector3(0, base + 0.35, 0), new THREE.Vector3(R, base + 0.35, 0));
    const Mseg = 48;
    for (let k = 0; k < rr; k++) for (let j = 1; j <= Mseg; j++) {
      const ang = (2 * Math.PI * j) / Mseg;
      pts.push(new THREE.Vector3(R * Math.cos(ang), base + 0.35 + (un ? (k + j / Mseg) * pitch : 0), R * Math.sin(ang)));
    }
    const yEnd = base + 0.35 + (un ? rr * pitch : 0);
    pts.push(new THREE.Vector3(0, yEnd, 0));
  }
  pts.push(new THREE.Vector3(0, y + 0.2, 0));
  M.path = makePath(pts);
  M.trip = 6 + 1.1 * r;

  // particles
  const N = 70;
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  M.ki = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.32, map: glowTex, color: 0xffe27a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, sizeAttenuation: true }));
  M.ki.renderOrder = 3; M.ki.frustumCulled = false;
  g.add(M.ki); M.kiN = N;
  if (M.gdnSites.length) {
    const per = 8, fg = new THREE.BufferGeometry();
    fg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(M.gdnSites.length * per * 3), 3));
    M.feed = new THREE.Points(fg, new THREE.PointsMaterial({ size: 0.2, map: glowTex, color: 0xff9a50, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
    M.feed.renderOrder = 3; M.feed.frustumCulled = false; M.feedPer = per;
    g.add(M.feed);
  }
  M.hasCtx = M.wedges.some((w) => !w.ghost);
  M.halfW = Math.max(1.4, M.ringR ? M.ringR + 1.5 : 0, PLINTH);
  tickParticles(M, 0);
  return M;
}

function makePath(pts) {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  const len = cum[cum.length - 1];
  return {
    len,
    at(u, out) {
      const s = Math.min(Math.max(u, 0), 1) * len;
      let lo = 0, hi = cum.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m; }
      const seg = cum[hi] - cum[lo] || 1, f = (s - cum[lo]) / seg;
      return out.set(pts[lo].x + (pts[hi].x - pts[lo].x) * f, pts[lo].y + (pts[hi].y - pts[lo].y) * f, pts[lo].z + (pts[hi].z - pts[lo].z) * f);
    },
  };
}

// ------------------------------------------------------------------ scene assembly
function visibleNames() { return S.compare ? [S.a, S.b] : [S.a]; }

function rebuild(refit) {
  for (const M of models) { scene.remove(M.grp); }
  models = [];
  pickables = [];
  for (const name of visibleNames()) {
    const d = DATA.models.find((m) => m.name === name);
    const M = buildModel(d, S.loops);
    scene.add(M.grp);
    models.push(M);
    pickables.push(...M.pick);
  }
  const anyLoop = models.some((m) => m.d.looped);
  $("rngLoops").disabled = !anyLoop;
  $("chkUnroll").disabled = !anyLoop;
  $("loopVal").textContent = anyLoop ? String(S.loops) : "n/a";
  $("scaleParam").textContent = `1M params = ${U_PARAM.toFixed(1)} unit`;
  $("scaleNote").textContent = `Scale: plinth and MLP height = ${U_PARAM.toFixed(1)} unit per 1M parameters. Wedge side = ${S_CTX} units per 1k tokens of context (so area grows with the square). The memory cube is a fixed ${CUBE_REF} units for a ${fmt(CUBE_REF_ELEMS)}-value state. The embedding is drawn capped.`;
  applyCtx();
  if (refit) fitCamera();
  hideTip();
  renderPower();
  renderSummary();
  dirty = true;
}

function layoutX() {
  const n = ctxN();
  const hw = models.map((M) => Math.max(M.halfW, M.hasCtx ? wedgeSide(n) / 2 + 0.6 : 0));
  if (models.length === 1) models[0].grp.position.x = 0;
  else { models[0].grp.position.x = -(hw[0] + 0.8); models[1].grp.position.x = hw[1] + 0.8; }
  return hw;
}

function applyCtx() {
  const n = ctxN(), L = wedgeSide(n);
  $("ctxVal").textContent = ctxLabel(n);
  for (const M of models) for (const w of M.wedges) {
    if (w.type === "attn") w.mesh.scale.set(L, 1, L);
    else {
      const bw = Math.max(0.07, wedgeSide(w.window)); // band width: fixed (window), independent of n
      w.mesh.scale.set(bw, 1, L * Math.SQRT2);
    }
  }
  layoutX();
  renderPower(true);
  dirty = true;
}

function fitCamera() {
  if (!models.length) return;
  const hw = layoutX();
  let minX = Infinity, maxX = -Infinity, H = 0;
  models.forEach((M, i) => { minX = Math.min(minX, M.grp.position.x - hw[i]); maxX = Math.max(maxX, M.grp.position.x + hw[i]); H = Math.max(H, M.height + 2.2); });
  const W = maxX - minX, cx = (minX + maxX) / 2;
  const tanH = Math.tan((camera.fov * Math.PI) / 360);
  const dist = Math.max(H / 2 / tanH, W / 2 / (tanH * camera.aspect)) * 1.18 + 3;
  controls.target.set(cx, H / 2 - 0.5, 0);
  camera.position.set(cx + dist * 0.5, H / 2 + dist * 0.18, dist * 0.84);
  camera.near = 0.1; camera.far = 1200; camera.updateProjectionMatrix();
  controls.update();
  dirty = true;
}

// ------------------------------------------------------------------ animation
let t0 = performance.now(), lastT = t0;
function tickParticles(M, time) {
  const a = M.ki.geometry.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < M.kiN; i++) {
    const u = (((i / M.kiN) + time / M.trip) % 1 + 1) % 1;
    M.path.at(u, v);
    a.setXYZ(i, v.x, v.y, v.z);
  }
  a.needsUpdate = true;
  if (M.feed) {
    const f = M.feed.geometry.attributes.position, per = M.feedPer;
    M.gdnSites.forEach((s, si) => {
      for (let j = 0; j < per; j++) {
        const ph = (((j / per) + time * 0.5) % 1 + 1) % 1, ang = (j / per) * Math.PI * 2 + si, rad = 1.9 * (1 - ph) + 0.05;
        f.setXYZ(si * per + j, s.x + rad * Math.cos(ang), s.y + (Math.sin(ang * 3) * 0.18) * (1 - ph), s.z + rad * Math.sin(ang));
      }
    });
    f.needsUpdate = true;
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const time = (now - t0) / 1000;
  if (S.motion) {
    for (const M of models) {
      tickParticles(M, time);
      if (M.aura) M.aura.material.opacity = 0.28 + 0.07 * Math.sin(time * 2.2);
    }
    stars.rotation.y = time * 0.004;
    dirty = true;
  }
  if (controls.update()) dirty = true;
  if (dirty) {
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    dirty = false;
  }
  lastT = now;
}

// ------------------------------------------------------------------ picking + tooltips
let lastPointerType = "mouse", downAt = null;
function ndc(ev) {
  const r = $("gl").getBoundingClientRect();
  return { x: ((ev.clientX - r.left) / r.width) * 2 - 1, y: -((ev.clientY - r.top) / r.height) * 2 + 1, px: ev.clientX - r.left, py: ev.clientY - r.top, w: r.width, h: r.height };
}
function pick(ev) {
  const p = ndc(ev);
  raycaster.setFromCamera({ x: p.x, y: p.y }, camera);
  const hit = raycaster.intersectObjects(pickables, false)[0];
  return { hit, p };
}
function showTip(hit, p) {
  const tip = $("tip");
  const u = hit.object.userData;
  tip.innerHTML = u.tip();
  tip.style.borderColor = u.color || "#ffd23f";
  tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = p.px + 14, y = p.py + 14;
  if (x + tw > p.w - 6) x = Math.max(6, p.px - tw - 14);
  if (y + th > p.h - 6) y = Math.max(6, p.h - th - 6);
  tip.style.left = x + "px"; tip.style.top = y + "px";
  hoverTip = hit.object;
  $("hint").style.visibility = "hidden";
}
function hideTip() { $("tip").hidden = true; hoverTip = null; $("hint").style.visibility = ""; }
function onMove(ev) {
  lastPointerType = ev.pointerType || "mouse";
  if (lastPointerType !== "mouse" || ev.buttons) return;
  const { hit, p } = pick(ev);
  if (hit) { showTip(hit, p); $("gl").style.cursor = "pointer"; } else { hideTip(); $("gl").style.cursor = ""; }
}
function onDown(ev) { lastPointerType = ev.pointerType || "mouse"; downAt = { x: ev.clientX, y: ev.clientY, t: performance.now() }; if (lastPointerType === "mouse") hideTip(); }
function onUp(ev) {
  if (!downAt) return;
  const moved = Math.hypot(ev.clientX - downAt.x, ev.clientY - downAt.y), dt = performance.now() - downAt.t;
  downAt = null;
  if (moved > 6 || dt > 500) return;
  const { hit, p } = pick(ev);
  if (hit) showTip(hit, p); else hideTip();
}

// ------------------------------------------------------------------ scouter readout
const tweens = new WeakMap();
function countTo(el, to) {
  const from = tweens.get(el)?.value ?? to;
  if (!S.motion || from === to) { el.textContent = fmt(to); tweens.set(el, { value: to }); return; }
  const st = performance.now();
  const rec = { value: to };
  tweens.set(el, rec);
  const step = (now) => {
    if (tweens.get(el) !== rec) return;
    const k = Math.min(1, (now - st) / 450);
    el.textContent = fmt(from + (to - from) * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function stats(M) {
  const d = M.d, n = ctxN(), c = d.config;
  const r = d.looped ? M.r : 1;
  const bl = d.by_loops ? d.by_loops[r - 1] : null;
  const apps = bl ? bl.applications : d.applications;
  const flops = bl ? bl.train_per_token : d.flops.train_per_token;
  const w = Math.min(c.window, n);
  return {
    total: d.params.total, nonEmb: d.params.non_embedding, emb: d.params.embedding,
    depth: bl ? bl.effective_depth : d.effective_depth, flops, apps,
    attnKV: apps.attn * 2 * c.d_model * n * BYTES, swaKV: apps.swa * 2 * c.d_model * w * BYTES,
    gdnState: apps.gdn * (d.gdn ? d.gdn.state_elems_per_layer : 0) * BYTES,
    kvPerTok: apps.attn * 2 * c.d_model * BYTES, cells: apps.attn * n * (n + 1) / 2,
  };
}

function renderPower(light) {
  const host = $("power");
  if (!light || host.children.length !== models.length) {
    host.innerHTML = models.map((M, i) => `<div class="m" data-i="${i}"></div>`).join("");
    light = false;
  }
  models.forEach((M, i) => {
    const d = M.d, s = stats(M), el = host.children[i];
    const body = `
      <div class="name"><span>${esc(d.experiment || "")} · ${esc(d.name)}</span>${d.status ? `<span class="status">${esc(d.status)}</span>` : ""}</div>
      <div class="big" data-big>${fmt(s.total)}</div>
      <div class="gag">PARAMETERS · over 9,000? yes, ×${(s.total / 9000).toFixed(0)}</div>
      <dl>
        <dt>non-embedding</dt><dd>${fmt(s.nonEmb)}</dd>
        <dt>embedding</dt><dd>${fmt(s.emb)}</dd>
        <dt>effective depth</dt><dd>${s.depth} layers${d.looped ? ` (r=${M.r})` : ""}</dd>
        <dt>train FLOPs/token</dt><dd>${fmtF(s.flops)}</dd>
        <dt>weights, bf16</dt><dd>${fmtB(d.memory.bf16_bytes)}</dd>
        ${d.memory.ternary_packed ? `<dt>weights, 2-bit packed</dt><dd>${fmtB(d.memory.packed_bytes)}</dd>` : ""}
        <dt>val loss</dt><dd>${d.val_loss != null ? d.val_loss.toFixed(4) : "not run yet"}</dd>
      </dl>
      <div class="reading"><span>AT ${fmt(ctxN())} TOKENS</span></div>
      <dl style="margin-top:8px">
        <dt>full-attn KV cache</dt><dd>${s.apps.attn ? fmtB(s.attnKV) : "none"}</dd>
        <dt>SWA KV cache</dt><dd>${s.apps.swa ? fmtB(s.swaKV) : "none"}</dd>
        <dt>GDN state (flat)</dt><dd>${s.apps.gdn ? fmtB(s.gdnState) : "none"}</dd>
        <dt>KV growth / token</dt><dd>${fmtB(s.kvPerTok)}</dd>
        <dt>attn score cells</dt><dd>${fmt(s.cells)}</dd>
      </dl>
      ${(d.notes || []).map((t) => `<div class="note">${esc(t)}</div>`).join("")}`;
    if (light && el.dataset.sig === d.name) {
      // context slider: refresh everything but keep the counting number stable
      const big = el.querySelector("[data-big]").textContent;
      el.innerHTML = body;
      el.querySelector("[data-big]").textContent = big;
    } else {
      el.innerHTML = body;
      el.dataset.sig = d.name;
      countTo(el.querySelector("[data-big]"), s.total);
    }
  });
}

function renderSummary() {
  $("a11y-summary").innerHTML = models.map((M) => {
    const d = M.d;
    const parts = d.sections.map((s) => `${s.name}${s.name === "core" ? ` (run ${M.r} times)` : ""}: ${s.blocks.map((b) => KIND[b.kind].name).join(", ")}`).join("; ");
    return `<li>${esc(d.name)}: ${fmt(d.params.total)} parameters. ${esc(parts)}.${d.ternary.enabled ? " Ternary weights." : ""}</li>`;
  }).join("");
}

// ------------------------------------------------------------------ test hook
function exposeTestHook() {
  const fire = (id, v, ev) => { const e = $(id); if (e.type === "checkbox") e.checked = !!v; else e.value = v; e.dispatchEvent(new Event(ev, { bubbles: true })); };
  window.__arch3d = {
    threeRevision: THREE.REVISION,
    get state() { return { ...S, ctx: ctxN() }; },
    names: DATA.models.map((m) => m.name),
    stats() {
      let objects = 0, meshes = 0, points = 0, lines = 0, sprites = 0;
      scene.traverse((o) => { objects++; if (o.isMesh) meshes++; else if (o.isPoints) points++; else if (o.isLine) lines++; else if (o.isSprite) sprites++; });
      const gl = renderer.getContext();
      return {
        objects, meshes, points, lines, sprites, pickables: pickables.length,
        ghostTiles: models.map((m) => m.ghostTiles), realTiles: models.map((m) => m.realTiles),
        wedges: models.flatMap((m) => m.wedges.filter((w) => !w.ghost).map((w) => ({ type: w.type, scale: [w.mesh.scale.x, w.mesh.scale.y, w.mesh.scale.z] }))),
        webgl: !!gl && !gl.isContextLost(), webglVersion: renderer.capabilities.isWebGL2 ? 2 : 1,
        pixelRatio: renderer.getPixelRatio(), canvas: [$("gl").width, $("gl").height],
        labels: document.querySelectorAll(".labels .tag").length,
        docWidth: document.documentElement.scrollWidth, viewport: window.innerWidth,
      };
    },
    setCtxIndex: (i) => fire("rngCtx", i, "input"),
    setLoops: (r) => fire("rngLoops", r, "input"),
    setModel: (name) => fire("selA", name, "change"),
    setModelB: (name) => fire("selB", name, "change"),
    setCompare: (on) => fire("chkCompare", on, "change"),
    setUnroll: (on) => fire("chkUnroll", on, "change"),
    reset: () => fitCamera(),
    // returns the tooltip text of the first pickable with a tip whose title matches `re`
    tipFor: (re) => { const m = pickables.find((p) => re.test(p.userData.tip().replace(/<[^>]+>/g, " "))); return m ? m.userData.tip().replace(/<[^>]+>/g, " ") : null; },
    render() { renderer.render(scene, camera); labelRenderer.render(scene, camera); },
  };
}
