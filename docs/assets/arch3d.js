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
const GAP = 0.5;
const GROUND_Y = -2.6;
const BYTES = 2;          // bf16
const COL = { attn: 0x3987e5, swa: 0x199e70, gdn: 0xd95926, mlp: 0xc98500, inj: 0x9a8fc8, emb: 0x6f7bb8, gold: 0xffd23f, orange: 0xff8a1f };
const KIND = {
  attn: { name: "Full attention", short: "ATTN", css: "#3987e5" },
  swa: { name: "Sliding-window attention", short: "SWA", css: "#199e70" },
  gdn: { name: "Gated DeltaNet", short: "GDN", css: "#d95926" },
};
// One plain-English line per component (shown when a block is clicked, and in the legend).
const PLAIN = {
  attn: "Every token looks back at every earlier token. Cost grows with the square of the context.",
  swa: "Each token only looks back a fixed window (256). Cost grows only linearly.",
  gdn: "A small fixed-size memory that is rewritten token by token. It never grows with context.",
  mlp: "A per-token feed-forward network: it thinks about each token on its own, with no look-back.",
  plinth: "The weights of this block's mixer. Taller means more parameters.",
  emb: "Turns each token id into a vector. Also reused as the output head (tied weights).",
  ring: "The looped core: one set of blocks stored once and run r times. Same weights, extra compute.",
  inj: "Mixes the original input e back in at the start of every loop, so the loop never forgets the prompt.",
  head: "Turns the final vector into scores for the next token.",
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
  try { const tr = await fetch("data/token-trace.json", { cache: "no-cache" }); if (tr.ok) TRACE = await tr.json(); } catch (e) { TRACE = null; }
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

  scene.add(new THREE.HemisphereLight(0xc8d4ff, 0x3a2414, 1.05));
  const sun = new THREE.DirectionalLight(0xfff0d0, 1.25);
  sun.position.set(6, 12, 8);
  scene.add(sun);
  const rim = new THREE.DirectionalLight(0x6aa8ff, 0.9); // cool rim light from behind: separates shapes from the backdrop
  rim.position.set(-9, 6, -10);
  scene.add(rim);
  const warm = new THREE.PointLight(0xff9a40, 0.8, 40); // warm "ki" glow from below
  warm.position.set(0, -1, 3);
  scene.add(warm);
  // subtle ground grid + glow disc so the model stands on something
  const grid = new THREE.GridHelper(60, 40, 0x3a4690, 0x1d2552);
  grid.position.y = GROUND_Y;
  grid.material.transparent = true; grid.material.opacity = 0.35; grid.material.depthWrite = false;
  scene.add(grid);
  const gdisc = new THREE.Mesh(new THREE.CircleGeometry(14, 64), new THREE.MeshBasicMaterial({ color: 0x4a60ff, transparent: true, opacity: 0.06, depthWrite: false }));
  gdisc.rotation.x = -Math.PI / 2; gdisc.position.y = GROUND_Y + 0.01;
  scene.add(gdisc);
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

  $("selA").addEventListener("change", () => { S.a = $("selA").value; syncLoopsDefault(S.a); rebuild(true); fusion(); });
  $("selB").addEventListener("change", () => { S.b = $("selB").value; rebuild(true); fusion(); });
  $("chkCompare").addEventListener("change", () => { S.compare = $("chkCompare").checked; $("ctlB").hidden = !S.compare; rebuild(true); if (S.compare) fusion(); });
  $("rngCtx").addEventListener("input", () => { S.ctxIdx = +$("rngCtx").value; applyCtx(); });
  $("rngLoops").addEventListener("input", () => { S.loops = +$("rngLoops").value; rebuild(false); });
  $("chkUnroll").addEventListener("change", () => { S.unroll = $("chkUnroll").checked; rebuild(false); });
  $("btnReset").addEventListener("click", () => fitCamera(true));
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
  PLAY_BOOTED = true;
  initUI2();
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
  if (typeof layoutChips === "function") layoutChips();
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
  t.userData.cached = true;
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
  m.userData.cached = true;
  matCache.set(key, m);
  return m;
}

function geo(key, make) {
  let g = geoCache.get(key);
  if (!g) { g = make(); g.userData.cached = true; geoCache.set(key, g); }
  return g;
}
const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28 });
const edgeMatGhost = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12 });
edgeMat.userData.cached = edgeMatGhost.userData.cached = true;
function withEdges(mesh, ghost) {
  const g = mesh.geometry;
  let e = edgeCache.get(g.uuid);
  if (!e) { e = new THREE.EdgesGeometry(g, 20); e.userData.cached = true; edgeCache.set(g.uuid, e); }
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
  const n = 320, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
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
  const p = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.5, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
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
  const add = (mesh, tip, colorCss, leg) => {
    mesh.userData.tip = tip;
    mesh.userData.color = colorCss;
    mesh.userData.leg = ghost ? "ghost" : leg;
    mesh.userData.sec = b.section;
    mesh.userData.plain = PLAIN[leg] || "";
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
  add(plinth, () => tipWeights(d, b, copy), KIND[b.kind].css, mixTern ? "plinth tern" : "plinth");

  // cost shape on top of the plinth
  let shapeH;
  if (b.kind === "gdn") {
    const edge = CUBE_REF * Math.cbrt(b.state_elems / CUBE_REF_ELEMS);
    shapeH = edge + 0.04;
    const cube = new THREE.Mesh(unitBox(), ghost ? mat(kcol, { ghost: true }) : mat(kcol));
    cube.scale.set(edge, edge, edge);
    cube.position.y = hP + 0.02 + edge / 2;
    withEdges(cube, ghost);
    add(cube, () => tipShape(d, b, copy), KIND.gdn.css, "gdn");
    out.gdn = { y: hP + 0.02 + edge / 2, edge };
    if (!ghost) { const sp = sprite(COL.orange, edge * 2.6, 0.45); sp.position.copy(cube.position); g.add(sp); }
  } else if (b.kind === "attn") {
    shapeH = SLAB_T + 0.04;
    const m = new THREE.Mesh(wedgeGeo(), ghost ? mat(kcol, { ghost: true }) : mat(kcol, { opacity: 0.88 }));
    m.position.y = hP + 0.02;
    withEdges(m, ghost);
    add(m, () => tipShape(d, b, copy), KIND.attn.css, "attn");
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
    add(m, () => tipShape(d, b, copy), KIND.swa.css, "swa");
    out.wedges.push({ mesh: m, type: "swa", ghost, window: d.config.window });
  }
  // MLP bicone
  const base = hP + shapeH + 0.18;
  const bi = new THREE.Mesh(biconeGeo(hM, ratio), ghost ? mat(COL.mlp, { ghost: true }) : mat(COL.mlp, { ternary: mlpTern }));
  bi.material.flatShading = true;
  bi.position.y = base;
  withEdges(bi, ghost);
  add(bi, () => tipMLP(d, b, copy), "#c98500", mlpTern ? "mlp tern" : "mlp");
  out.height = base + hM;
  out.hP = hP;
  if (!ghost) {
    const lab = label(KIND[b.kind].short, "kind", new THREE.Vector3(PLINTH * 0.62, hP / 2, 0), KIND[b.kind].css);
    g.add(lab);
  }
  return out;
}

function buildModel(d, slotLoops) {
  const M = { d, grp: new THREE.Group(), wedges: [], pick: [], gdnSites: [], r: 1, ringR: 0, height: 0, ghostTiles: 0, realTiles: 0, pl: { pre: [], core: [], coda: [] }, spin: [] };
  const g = M.grp;
  const cfg = d.config, dm = cfg.d_model;
  const looped = d.looped;
  const r = looped ? slotLoops : 1;
  M.r = r;
  const unroll = looped && S.unroll && r > 1;
  const secs = Object.fromEntries(d.sections.map((s) => [s.name, s]));
  const pre = secs.prelude?.blocks || [], core = secs.core?.blocks || [], coda = secs.coda?.blocks || [];
  const place = (blocks, y0, store) => {
    let y = y0;
    for (const b of blocks) {
      const t = buildTile(d, b, false, null);
      t.group.position.set(0, y, 0);
      g.add(t.group);
      M.wedges.push(...t.wedges); M.pick.push(...t.pick);
      if (t.gdn) M.gdnSites.push(new THREE.Vector3(0, y + t.gdn.y, 0));
      M.realTiles++;
      if (store) store.push({ y, kind: b.kind, hP: t.hP, gdn: t.gdn });
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
  emb.userData.color = "#6f7bb8"; emb.userData.leg = "emb"; emb.userData.plain = PLAIN.emb;
  g.add(emb); M.pick.push(emb);
  if (d.embedding.params / 1e6 * U_PARAM > 1.6) {
    for (const f of [0.3, 0.7]) {
      const ring = new THREE.Mesh(geo("brk", () => new THREE.TorusGeometry(1.16, 0.045, 8, 40)), new THREE.MeshBasicMaterial({ color: 0x0a0d1c }));
      ring.rotation.x = Math.PI / 2; ring.position.y = hE * f; ring.userData.leg = "emb"; g.add(ring);
    }
  }
  note("EMBED", "sect", 0, -0.45, 1.5);
  let y = hE + 0.7;

  // prelude
  const yPre0 = y;
  if (pre.length) { y = place(pre, y, M.pl.pre); note("PRELUDE", "sect", -(PLINTH + 1.15), yPre0 + 0.4, 0); }

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
    ring.userData.color = "#ffd23f"; ring.userData.leg = "ring"; ring.userData.plain = PLAIN.ring;
    g.add(ring); M.pick.push(ring);
    M.pl.core = built.map((t, i) => ({ kind: core[i].kind, hP: t.hP, gdn: t.gdn }));
    Object.assign(M.pl, { base, R, pitch, unroll, r, n, tH });
    const floor = new THREE.Mesh(geo(`floor${R.toFixed(2)}`, () => new THREE.CircleGeometry(R, 64)), new THREE.MeshBasicMaterial({ color: COL.gold, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = base - 0.03; floor.userData.leg = "ring"; g.add(floor);
    const aura = sprite(COL.orange, R * 4.4, 0.32); aura.position.set(0, base + tH * 0.4, 0); g.add(aura); M.aura = aura; aura.userData.leg = "aura";
    // re-inject disc at the centre
    const inj = d.sections.find((s) => s.name === "core").injection;
    const hI = Math.max(0.14, (inj.params / 1e6) * U_PARAM);
    const injM = new THREE.Mesh(geo(`inj${hI.toFixed(2)}`, () => new THREE.CylinderGeometry(0.3, 0.6, hI, 24)), mat(COL.inj));
    injM.position.set(0, base + hI / 2, 0); withEdges(injM, false);
    injM.userData.tip = () => `<b>Re-inject</b> (every loop)<br>${row("params", fmt(inj.params))}${row("shape", inj.shape)}<span class="sub">${esc(inj.formula)}<br>The original input e is mixed back in at the start of each pass, so the loop never forgets the prompt.</span>`;
    injM.userData.color = "#9a8fc8"; injM.userData.leg = "inj"; injM.userData.plain = PLAIN.inj;
    g.add(injM); M.pick.push(injM);
    note("RE-INJECT", "kind", 0, base + hI + 0.4, 0, "#9a8fc8");
    // e-beam from the embedding to the disc (dashed violet)
    const bx = PLINTH * 0.5 + 0.9;
    const beam = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(bx, hE * 0.5, 0), new THREE.Vector3(bx, base + 0.2, 0), new THREE.Vector3(0.45, base + 0.2, 0)]),
      new THREE.LineDashedMaterial({ color: COL.inj, dashSize: 0.22, gapSize: 0.14 }));
    beam.computeLineDistances(); beam.userData.leg = "beam"; g.add(beam);

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
        ln.computeLineDistances(); ln.userData.leg = "same"; g.add(ln);
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
      tube.userData.leg = "helix"; g.add(tube);
    }
    note(`CORE × ${r}`, "loop", -(R + 1.4), base + tH / 2, 0);
    for (let q = 0; q < 3; q++) { const sp = sprite(COL.gold, 0.9, 0.9); sp.userData.leg = "ring"; g.add(sp); M.spin.push({ sp, R, y: base - 0.02, off: (q * 2 * Math.PI) / 3 }); }
    entryY = base + 0.35;
    exitY = unroll ? base + 0.35 + (r - 1) * pitch + pitch * 1 : base + 0.35;
    loopPts = { R, base, pitch, unroll, r };
    y = top + 0.7;
  }

  // coda
  const yCoda0 = y;
  M.pl.codaY0 = y;
  if (coda.length) { y = place(coda, y, M.pl.coda); note("CODA", "sect", -(PLINTH + 1.15), yCoda0 + 0.4, 0); }

  // head
  const hH = 0.5;
  const head = new THREE.Mesh(geo("head", () => new THREE.CylinderGeometry(0.3, 1.0, hH, 24)), mat(COL.emb));
  head.position.y = y + hH / 2; withEdges(head, false);
  head.userData.tip = () => `<b>Output head</b> (vector → next-token scores)<br>${d.embedding.tied ? "Tied to the embedding: 0 extra parameters." : "Separate output matrix."}`;
  head.userData.color = "#6f7bb8"; head.userData.leg = "head"; head.userData.plain = PLAIN.head;
  g.add(head); M.pick.push(head);
  note("NEXT TOKEN", "sect", 0, y + hH + 0.45, 0);
  M.pl.headBase = y; M.pl.headTop = y + hH; M.pl.hE = hE;
  y += hH;
  M.height = y;

  // ternary badge + title
  const titleY = y + 1.5;
  note(`${d.experiment || ""} · ${d.name}`, "title", 0, titleY, 0);
  if (d.ternary.enabled) note(`1.58-bit ternary · ${(100 * d.ternary.fraction_of_total).toFixed(0)}% of weights`, "badge", 0, titleY - 0.8, 0);

  // overall aura
  const big = sprite(COL.orange, Math.max(y * 1.1, 8), 0.14); big.position.set(0, y * 0.5, 0); big.userData.leg = "aura"; g.add(big); M.big = big;

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
  M.ki.renderOrder = 3; M.ki.frustumCulled = false; M.ki.userData.leg = "ki";
  g.add(M.ki); M.kiN = N;
  if (M.gdnSites.length) {
    const per = 8, fg = new THREE.BufferGeometry();
    fg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(M.gdnSites.length * per * 3), 3));
    M.feed = new THREE.Points(fg, new THREE.PointsMaterial({ size: 0.2, map: glowTex, color: 0xff9a50, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
    M.feed.renderOrder = 3; M.feed.frustumCulled = false; M.feedPer = per; M.feed.userData.leg = "feed";
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

// ------------------------------------------------------------------ disposal (CSS2D labels and GPU resources never leak)
const TEX_KEYS = ["map", "alphaMap", "emissiveMap", "normalMap", "roughnessMap", "metalnessMap", "envMap", "lightMap", "aoMap"];
function disposeMaterial(m) {
  if (!m || (m.userData && m.userData.cached)) return;
  for (const k of TEX_KEYS) { const t = m[k]; if (t && t !== glowTex && !t.userData.cached) t.dispose(); }
  m.dispose();
}
function disposeGroup(grp) {
  if (!grp) return;
  const labels = [];
  grp.traverse((o) => {
    if (o.isCSS2DObject || (o.element && o.element instanceof HTMLElement)) labels.push(o);
    if (o.geometry && !(o.geometry.userData && o.geometry.userData.cached)) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(disposeMaterial);
  });
  for (const o of labels) o.element.remove();
  if (grp.parent) grp.parent.remove(grp);
}
function clearModels() {
  hideTip();
  clearHalos();
  for (const M of models) disposeGroup(M.grp);
  models = []; pickables = []; PLAY = null;
}

// ------------------------------------------------------------------ scene assembly
function visibleNames() { return S.compare ? [S.a, S.b] : [S.a]; }

function rebuild(refit) {
  clearModels();
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
  $("scaleNote").textContent = `Scale: plinth and MLP height = ${U_PARAM.toFixed(1)} unit per 1M parameters. Wedge side = ${S_CTX} units per 1k tokens of context (so area grows with the square). The memory cube is a fixed ${CUBE_REF} units for a ${fmt(CUBE_REF_ELEMS)}-value state. The embedding is drawn capped.`;
  applyCtx();
  if (refit) fitCamera(!!PLAY_BOOTED);
  hideTip();
  playReset();
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

let fly = null, fitDist = 20, focused = null;
function flyTo(toT, toP, dur) {
  if (!S.motion || !dur) { controls.target.copy(toT); camera.position.copy(toP); controls.update(); fly = null; dirty = true; return; }
  fly = { t0: performance.now(), dur: dur * 1000, fT: controls.target.clone(), fP: camera.position.clone(), tT: toT.clone(), tP: toP.clone() };
}
function fitCamera(smooth) {
  if (!models.length) return;
  focused = null; $("focusNote").hidden = true;
  const hw = layoutX();
  let minX = Infinity, maxX = -Infinity, top = 0;
  models.forEach((M, i) => { minX = Math.min(minX, M.grp.position.x - hw[i]); maxX = Math.max(maxX, M.grp.position.x + hw[i]); top = Math.max(top, M.height + 3.1); });
  const bottom = -2.4, H = top - bottom, cy = (top + bottom) / 2;
  const W = maxX - minX, cx = (minX + maxX) / 2;
  const tanH = Math.tan((camera.fov * Math.PI) / 360);
  const dist = Math.max(H / 2 / tanH, W / 2 / (tanH * camera.aspect)) * 1.04 + 1.5;
  fitDist = dist;
  camera.near = 0.1; camera.far = 1200; camera.updateProjectionMatrix();
  flyTo(new THREE.Vector3(cx, cy, 0), new THREE.Vector3(cx + dist * 0.5, cy + dist * 0.22, dist * 0.82), smooth ? 0.9 : 0);
  if (typeof layoutChips === "function") layoutChips();
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

let burst = null;
function fusion() {
  if (!S.motion || !models.length) return;
  const fl = $("flash"); fl.classList.remove("go"); void fl.offsetWidth; fl.classList.add("go");
  const N = 140, pos = new Float32Array(N * 3), vel = [];
  for (let i = 0; i < N; i++) { const u = Math.random() * 2 - 1, th = Math.random() * 6.283, sq = Math.sqrt(1 - u * u), sp = 3 + Math.random() * 6; vel.push(sq * Math.cos(th) * sp, u * sp, sq * Math.sin(th) * sp); }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.4, map: glowTex, color: 0xffe27a, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
  pts.frustumCulled = false; pts.renderOrder = 4;
  const c = controls.target.clone();
  pts.position.copy(c);
  if (burst) disposeGroup(burst.pts);
  scene.add(pts);
  burst = { pts, vel, t0: performance.now() };
}
function tickBurst(now) {
  if (!burst) return;
  const k = (now - burst.t0) / 1000;
  if (k > 1.1) { disposeGroup(burst.pts); burst = null; return; }
  const a = burst.pts.geometry.attributes.position, e = 1 - Math.pow(1 - k / 1.1, 2);
  for (let i = 0; i < burst.vel.length / 3; i++) a.setXYZ(i, burst.vel[i * 3] * e * 0.6, burst.vel[i * 3 + 1] * e * 0.6, burst.vel[i * 3 + 2] * e * 0.6);
  a.needsUpdate = true; burst.pts.material.opacity = 1 - k / 1.1; dirty = true;
}
let farLabels = null;
function frame(now) {
  requestAnimationFrame(frame);
  const time = (now - t0) / 1000;
  const dtp = Math.min(0.1, (now - lastT) / 1000);
  playTick(dtp);
  if (fly) {
    const k = Math.min(1, (now - fly.t0) / fly.dur), e = k * k * (3 - 2 * k);
    controls.target.lerpVectors(fly.fT, fly.tT, e); camera.position.lerpVectors(fly.fP, fly.tP, e);
    if (k >= 1) fly = null;
    dirty = true;
  }
  tickBurst(now);
  if (PLAY && PLAY.cenW && PLAY.t > 0 && $("chkFollow").checked && !fly && !focused) {
    const want = Math.max(8, fitDist * 0.5), cur = camera.position.distanceTo(controls.target);
    const dir = camera.position.clone().sub(controls.target).normalize();
    const kk = S.motion ? 0.08 : 1;
    controls.target.lerp(PLAY.cenW, kk);
    camera.position.copy(controls.target).addScaledVector(dir, cur + (want - cur) * kk);
    dirty = true;
  }
  const far = camera.position.distanceTo(controls.target) > fitDist * 0.8;
  if (far !== farLabels) { farLabels = far; labelRenderer.domElement.classList.toggle("far", far); }
  if (S.motion) {
    for (const M of models) {
      tickParticles(M, time);
      if (M.aura) M.aura.material.opacity = 0.28 + 0.07 * Math.sin(time * 2.2);
      if (M.big) M.big.material.opacity = 0.14 + 0.05 * Math.sin(time * 1.3 + 1);
      for (const q of M.spin) { const a = q.off + time * 0.35 * M.r; q.sp.position.set(q.R * Math.cos(a), q.y, q.R * Math.sin(a)); }
    }
    stars.rotation.y = time * 0.004;
    dirty = true;
  }
  if (halos.length) { for (const h of halos) h.sp.material.opacity = 0.5 + 0.3 * Math.sin(time * 6); dirty = true; }
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
  tip.innerHTML = u.tip() + (u.plain ? `<span class="sub plain">${esc(u.plain)}</span>` : "");
  tip.style.borderColor = u.color || "#ffd23f";
  tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = p.px + 14, y = p.py + 14;
  if (x + tw > p.w - 6) x = Math.max(6, p.px - tw - 14);
  if (y + th > p.h - 6) y = Math.max(6, p.h - th - 6);
  tip.style.left = x + "px"; tip.style.top = y + "px";
  setHover(hit.object);
  $("hint").style.visibility = "hidden";
}
function setHover(o) {
  if (hoverTip === o) return;
  if (hoverTip && hoverTip.userData.base) { hoverTip.scale.copy(hoverTip.userData.base); hoverTip.userData.base = null; }
  hoverTip = o;
  if (o) { o.userData.base = o.scale.clone(); o.scale.multiplyScalar(1.06); }
  dirty = true;
}
function hideTip() { $("tip").hidden = true; setHover(null); $("hint").style.visibility = ""; }
function focusOn(obj) {
  const b = new THREE.Box3().setFromObject(obj), c = b.getCenter(new THREE.Vector3()), sz = b.getSize(new THREE.Vector3()).length();
  const dir = camera.position.clone().sub(controls.target).normalize();
  const d = Math.max(4, sz * 1.6 + 2.5);
  flyTo(c, c.clone().add(dir.multiplyScalar(d)), 0.8);
  focused = obj; $("focusNote").hidden = false;
}
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
  if (hit) { showTip(hit, p); focusOn(hit.object); } else hideTip();
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
    reset: () => fitCamera(false),
    play: {
      get P() { return PLAY; },
      seek(t) { PLAY.playing = false; PLAY.target = null; PLAY.t = t; applyPlay(); dirty = true; renderer.render(scene, camera); labelRenderer.render(scene, camera); },
      toggle: playToggle, step: playStep, back: playBack, restart: playRestart,
    },
    legendCount: () => document.querySelectorAll(".lrow").length,
    // returns the tooltip text of the first pickable with a tip whose title matches `re`
    tipFor: (re) => { const m = pickables.find((p) => re.test(p.userData.tip().replace(/<[^>]+>/g, " "))); return m ? m.userData.tip().replace(/<[^>]+>/g, " ") : null; },
    render() { renderer.render(scene, camera); labelRenderer.render(scene, camera); },
  };
}

// ================================================================== Play mode (token trace)
// Real tokens (data/token-trace.json, from experiments/viz/export_trace.py) flow through the selected model's real structure.
// Attention arcs / loop deltas / continuation are MEASURED only when the trace was recorded for exactly this model and r;
// otherwise the same animation runs on synthetic arcs and is labelled ILLUSTRATIVE.
var PLAY_BOOTED = false;
const halos = [];
let TRACE = null, PLAY = null;
const ROW_Y = -1.15, ROW_Z = 2.5, SP_DOT = 0.17, PLAY_V = 7, ARC_SEG = 10;
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const ease = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };
const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
const tokColor = (i, n) => new THREE.Color().setHSL(0.03 + 0.6 * (i / Math.max(1, n - 1)), 0.9, 0.62);
const unvis = (s) => String(s).replace(/↵/g, "\n").replace(/␣/g, " ").replace(/⇥/g, "\t");

function traceValid(d) {
  const t = TRACE;
  if (!t || t.measured !== true || !t.model || !t.model.config_file) return false;
  if (!t.model.config_file.endsWith("/" + d.name + ".json")) return false;
  const kinds = (n) => ((d.sections.find((s) => s.name === n) || {}).blocks || []).map((b) => b.kind);
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  if (!same(kinds("prelude"), t.model.prelude) || !same(kinds("core"), t.model.core) || !same(kinds("coda"), t.model.coda)) return false;
  if (d.looped && S.loops !== t.model.loops) return false;
  return true;
}
function whyIllustrative(d) {
  const t = TRACE;
  if (!t.model.config_file.endsWith("/" + d.name + ".json")) return `The measured trace was recorded on ${t.model.experiment} (${t.model.config_file.split("/").pop()}), not on ${d.name}. Tokens are real and the layout is this model's real layout; the arcs and outputs here are synthetic.`;
  return `The measured trace used r = ${t.model.loops}; you set r = ${S.loops}. Set r back to ${t.model.loops} to see measured data. Until then the arcs are synthetic.`;
}

function hash01(str) { let h = 2166136261; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return ((h >>> 0) % 100000) / 100000; }
function synthTop(P, key) {
  if (P.synth.has(key)) return P.synth.get(key);
  const out = [];
  for (let q = 0; q < P.T; q++) {
    const c = new Map();
    c.set(q, 0.5 + hash01(key + "s" + q));
    if (q > 0) c.set(q - 1, 0.7 + hash01(key + "p" + q));
    c.set(0, (c.get(0) || 0) + 0.3 + hash01(key + "z" + q));
    const j = Math.floor(hash01(key + "r" + q) * (q + 1)); c.set(j, (c.get(j) || 0) + 0.4 + hash01(key + "w" + q));
    const arr = [...c.entries()], tot = arr.reduce((a, b) => a + b[1], 0);
    out.push(arr.map(([s, w]) => [s, w / tot]).sort((a, b) => b[1] - a[1]).slice(0, 3));
  }
  P.synth.set(key, out);
  return out;
}

function buildPlay(M) {
  const d = M.d, pl = M.pl, toks = TRACE.tokens, T = toks.length, valid = traceValid(d);
  const P = { M, d, T, toks, valid, t: 0, playing: false, target: null, speed: S.speed || 1, outShown: 0, chipObjs: [], chipPos: toks.map(() => V3(0, ROW_Y, ROW_Z)), pos: toks.map(() => V3()), synth: new Map(), byKey: new Map(), cen: V3(), lc: -2 };
  if (valid) for (const L of TRACE.layers) P.byKey.set(`${L.stage}|${L.layer}|${L.loop}`, L);
  const pts = [], sts = [];
  const station = (st, v) => { st.pt = pts.length; pts.push(v); sts.push(st); };
  const yS = (t) => t.y + (t.kind === "gdn" && t.gdn ? t.gdn.y : t.hP + 0.12);
  station({ type: "tokens", stage: "tokens", dwell: 0.6 }, V3(0, ROW_Y, ROW_Z));
  station({ type: "embed", stage: "embed", dwell: 0.8 }, V3(0, pl.hE * 0.5, 0));
  pl.pre.forEach((t, i) => station({ type: "layer", stage: "prelude", idx: i, kind: t.kind, dwell: t.kind === "gdn" ? 1.5 : 1.8, key: `prelude|${i}|null` }, V3(0, yS(t), 0)));
  P.r = 0;
  if (pl.core.length) {
    const { base, R, pitch, unroll, r, n } = pl;
    P.r = r;
    station({ type: "inject", stage: "core", loop: 0, dwell: 0.5 }, V3(0, base + 0.35, 0));
    let ang = 0, y = base + 0.35;
    pts.push(V3(R, y, 0));
    const arc = (a, yy) => { const st = Math.max(2, Math.ceil(Math.abs(a - ang) / (Math.PI / 14))); for (let s = 1; s < st; s++) { const f = s / st, aa = ang + (a - ang) * f; pts.push(V3(R * Math.cos(aa), y + (yy - y) * f, R * Math.sin(aa))); } };
    for (let k = 0; k < r; k++) for (let i = 0; i < n; i++) {
      const a = 2 * Math.PI * k + Math.PI / n + (2 * Math.PI * i) / n, t = pl.core[i];
      const yy = base + (unroll ? k * pitch : 0) + (t.kind === "gdn" && t.gdn ? t.gdn.y : t.hP + 0.12);
      arc(a, yy);
      station({ type: "layer", stage: "core", loop: k, idx: i, kind: t.kind, dwell: t.kind === "gdn" ? 1.5 : 1.8, key: `core|${i}|${k}`, loopFirst: i === 0 }, V3(R * Math.cos(a), yy, R * Math.sin(a)));
      ang = a; y = yy;
    }
    arc(2 * Math.PI * r, y);
    pts.push(V3(0, pl.codaY0 - 0.25, 0));
  }
  pl.coda.forEach((t, i) => station({ type: "layer", stage: "coda", idx: i, kind: t.kind, dwell: t.kind === "gdn" ? 1.5 : 1.8, key: `coda|${i}|null` }, V3(0, yS(t), 0)));
  station({ type: "head", stage: "out", dwell: 1.1 }, V3(0, (pl.headBase + pl.headTop) / 2, 0));
  P.path = makePath(pts);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  let tt = 0, prevS = 0;
  P.steps = sts.map((st, j) => {
    const s = cum[st.pt], travel = j === 0 ? 0 : Math.max(0.35, (s - prevS) / PLAY_V);
    const o = { st, tStart: tt, travel, dwell: st.dwell, s, sPrev: prevS };
    tt += travel + st.dwell; prevS = s; return o;
  });
  P.headIdx = P.steps.length - 1;
  P.cont = valid ? TRACE.continuation.tokens.slice(0, 40) : [];
  P.nOut = valid ? P.cont.length : 3;
  for (let k = 0; k < P.nOut; k++) { P.steps.push({ st: { type: "out", stage: "out", k }, tStart: tt, travel: 0, dwell: 0.3, s: prevS, sPrev: prevS }); tt += 0.3; }
  P.total = tt + 0.4;

  const g = M.grp;
  const dg = new THREE.BufferGeometry();
  dg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(T * 3), 3));
  const cols = new Float32Array(T * 3); toks.forEach((_, i) => tokColor(i, T).toArray(cols, i * 3));
  dg.setAttribute("color", new THREE.BufferAttribute(cols, 3));
  P.dots = new THREE.Points(dg, new THREE.PointsMaterial({ size: 0.5, map: glowTex, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
  P.dots.renderOrder = 5; P.dots.frustumCulled = false; P.dots.visible = false; g.add(P.dots);
  const ag = new THREE.BufferGeometry(), AM = T * 3 * ARC_SEG * 2;
  ag.setAttribute("position", new THREE.BufferAttribute(new Float32Array(AM * 3), 3));
  ag.setAttribute("color", new THREE.BufferAttribute(new Float32Array(AM * 3), 3));
  ag.setDrawRange(0, 0);
  P.arcs = new THREE.LineSegments(ag, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false }));
  P.arcs.renderOrder = 4; P.arcs.frustumCulled = false; P.arcs.userData.leg = "arcs"; g.add(P.arcs);
  P.flash = sprite(0xb07cff, 1.8, 0); g.add(P.flash);
  P.emit = sprite(0xffe27a, 0.9, 0); g.add(P.emit);
  toks.forEach((tk, i) => {
    const el = document.createElement("div");
    el.className = "tkn"; el.title = `token ${i + 1}: id ${tk.id}, piece "${tk.vis}"`;
    el.innerHTML = `<i style="background:#${tokColor(i, T).getHexString()}"></i><span></span>`;
    el.lastChild.textContent = tk.vis;
    const o = new CSS2DObject(el); o.userData.leg = "chips"; g.add(o); P.chipObjs.push(o);
  });
  const lab = label("PROMPT TOKENS", "sect promptlab", V3(0, ROW_Y, ROW_Z), null); g.add(lab); P.promptLab = lab;
  return P;
}

function layoutChips() {
  const P = PLAY; if (!P) return;
  const W = Math.max(200, stage.clientWidth), H = Math.max(200, stage.clientHeight);
  const camP = fly ? fly.tP : camera.position, camT = fly ? fly.tT : controls.target;
  const tc = camera.clone(); tc.aspect = W / H; tc.position.copy(camP); tc.lookAt(camT); tc.updateProjectionMatrix(); tc.updateMatrixWorld(true);
  const dist = camP.distanceTo(camT) * 0.97, g0 = P.M.grp.position;
  const perRow = Math.max(3, Math.min(P.T, Math.floor((W * 0.94) / 72)));
  const spPx = Math.min(96, (W * 0.94) / perRow), gapPx = 30, base = 58;
  const place = (px, py) => { const v = V3((px / (W / 2)), -1 + (2 * py) / H, 0.5).unproject(tc); const dir = v.sub(tc.position).normalize(); return tc.position.clone().addScaledVector(dir, dist).sub(g0); };
  const rows = Math.ceil(P.T / perRow);
  for (let i = 0; i < P.T; i++) {
    const row = Math.floor(i / perRow), inRow = Math.min(perRow, P.T - row * perRow), col = i - row * perRow;
    P.chipPos[i].copy(place((col - (inRow - 1) / 2) * spPx, base + (rows - 1 - row) * gapPx));
    P.chipObjs[i].position.copy(P.chipPos[i]);
  }
  P.promptLab.position.copy(place(0, base + rows * gapPx));
  applyPlay();
}

function setPlayEnabled(on, msg) {
  for (const id of ["btnPlay", "btnStep", "btnBack", "btnRestart", "rngSpeed"]) $(id).disabled = !on;
  if (msg) $("playDetail").textContent = msg;
}

function playReset() {
  PLAY = null; $("outRow").hidden = true; $("outStrip").innerHTML = ""; $("ptip").hidden = true;
  const M = models[0];
  if (!TRACE || !M) { setPlayEnabled(false, "Play needs data/token-trace.json (generate it with experiments/viz/export_trace.py)."); renderPlayInfo(null); renderLegend(); return; }
  PLAY = buildPlay(M);
  setPlayEnabled(true);
  paintPlayBtn();
  renderPlayInfo(PLAY);
  buildStageReadout();
  layoutChips();
  renderLegend();
}

function paintPlayBtn() {
  const P = PLAY, b = $("btnPlay");
  if (!P) return;
  if (!S.motion) { b.textContent = "Step ▶"; b.title = "Motion is off (or reduced-motion is set): step through one stage at a time"; }
  else { b.title = ""; b.textContent = P.playing ? "⏸ Pause" : P.t >= P.total - 1e-3 ? "↻ Replay" : P.t > 0 ? "▶ Resume" : "▶ Play"; }
  b.setAttribute("aria-pressed", String(!!P.playing));
}
const stepTarget = (s) => s.tStart + s.travel + s.dwell * 0.9;
function playToggle() {
  const P = PLAY; if (!P) return;
  if (!S.motion) { playStep(); return; }
  if (P.t >= P.total - 1e-3) P.t = 0;
  P.playing = !P.playing; P.target = null; paintPlayBtn(); dirty = true;
}
function playStep() {
  const P = PLAY; if (!P) return;
  P.playing = false;
  const n = P.steps.findIndex((s) => stepTarget(s) > P.t + 1e-3);
  if (n < 0) { paintPlayBtn(); return; }
  const tg = stepTarget(P.steps[n]);
  if (!S.motion) { P.t = tg; applyPlay(); } else P.target = tg;
  paintPlayBtn(); dirty = true;
}
function playBack() {
  const P = PLAY; if (!P) return;
  P.playing = false; P.target = null;
  let tg = 0; for (const s of P.steps) { const x = stepTarget(s); if (x < P.t - 0.05) tg = x; }
  P.t = tg; applyPlay(); paintPlayBtn(); dirty = true;
}
function playRestart() {
  const P = PLAY; if (!P) return;
  fitCamera(true);
  P.playing = false; P.target = null; P.t = 0; applyPlay(); paintPlayBtn(); dirty = true;
}
function playTick(dt) {
  const P = PLAY; if (!P) return;
  if (P.playing) {
    P.t += dt * P.speed;
    if (P.t >= P.total) { P.t = P.total; P.playing = false; paintPlayBtn(); }
    applyPlay(); dirty = true;
  } else if (P.target != null) {
    P.t = Math.min(P.target, P.t + dt * P.speed);
    if (P.t >= P.target) P.target = null;
    applyPlay(); dirty = true;
  }
}

function locateStep(P, t) { let j = 0; while (j + 1 < P.steps.length && P.steps[j + 1].tStart <= t) j++; return j; }
function layerTop(P, st) {
  if (P.valid) { const L = P.byKey.get(st.key); if (L && L.top) return L.top; }
  return synthTop(P, st.key);
}

const STAGE_KEYS = [["embed", "Embedding"], ["prelude", "Prelude"], ["core", "Core loop"], ["coda", "Coda"], ["out", "Output"]];
function buildStageReadout() {
  const P = PLAY, has = { embed: 1, prelude: P.M.pl.pre.length, core: P.M.pl.core.length, coda: P.M.pl.coda.length, out: 1 };
  $("playStage").innerHTML = STAGE_KEYS.filter((k) => has[k[0]]).map((k) => `<span data-s="${k[0]}">${k[1]}</span>`).join('<i aria-hidden="true">→</i>');
}

function applyPlay() {
  const P = PLAY; if (!P) return;
  const T = P.T, j = locateStep(P, P.t), S0 = P.steps[j], st = S0.st, local = P.t - S0.tStart;
  const inTravel = local < S0.travel;
  const ph = inTravel ? -1 : clamp01((local - S0.travel) / S0.dwell);
  const k = inTravel ? ease(local / S0.travel) : 1;
  const afterHead = j > P.headIdx;
  const started = P.t > 0;
  P.path.at((inTravel ? S0.sPrev + (S0.s - S0.sPrev) * k : S0.s) / P.path.len, P.cen);
  const cen = P.cen, hr = Math.hypot(cen.x, cen.z);
  const dx = hr > 1.5 ? cen.x / hr : 1, dz = hr > 1.5 ? cen.z / hr : 0;
  const gdn = st.kind === "gdn" && ph >= 0;
  const spread = gdn ? 0.22 : SP_DOT;
  const rowBlend = j === 0 ? 1 : j === 1 && inTravel ? 1 - k : 0;
  let conv = 1;
  if (st.type === "head" && ph >= 0) conv = 1 - 0.9 * ease(ph * 1.4);
  const dp = P.dots.geometry.attributes.position;
  for (let i = 0; i < T; i++) {
    let lat = (i - (T - 1) / 2) * spread * conv;
    if (gdn) lat *= 1 - 0.92 * Math.sin(Math.PI * clamp01(ph * 1.35 - i * 0.085));
    const p = P.pos[i].set(cen.x + dx * lat, cen.y, cen.z + dz * lat);
    if (rowBlend > 0) p.lerp(P.chipPos[i], rowBlend);
    dp.setXYZ(i, p.x, p.y, p.z);
  }
  dp.needsUpdate = true;
  P.dots.visible = started && !afterHead;
  P.cenW = cen.clone().add(P.M.grp.position);
  P.M.ki.visible = !started;
  for (const o of P.chipObjs) o.element.classList.toggle("used", j >= 1);

  let nv = 0, strongest = null;
  const isAtt = st.type === "layer" && (st.kind === "attn" || st.kind === "swa") && ph >= 0;
  if (isAtt) {
    const top = layerTop(P, st), pa = P.arcs.geometry.attributes.position, ca = P.arcs.geometry.attributes.color;
    const kc = new THREE.Color(st.kind === "attn" ? COL.attn : COL.swa).lerp(new THREE.Color(0xffffff), 0.35), win = P.d.config.window;
    for (let q = 0; q < T; q++) {
      const aq = clamp01((ph - 0.05 - q * 0.07) / 0.15);
      if (aq <= 0) continue;
      for (const [s, w] of top[q]) {
        if (s === q || (st.kind === "swa" && q - s >= win)) continue;
        if (!strongest || w > strongest.w) strongest = { q, s, w };
        const a = P.pos[q], b = P.pos[s], mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 + 0.14 + 0.07 * Math.abs(q - s), mz = (a.z + b.z) / 2;
        const f = aq * (0.12 + 0.88 * Math.min(1, w * 1.6));
        let px = a.x, py = a.y, pz = a.z;
        for (let sg = 1; sg <= ARC_SEG; sg++) {
          const u = sg / ARC_SEG, iu = 1 - u;
          const x = iu * iu * a.x + 2 * iu * u * mx + u * u * b.x, y = iu * iu * a.y + 2 * iu * u * my + u * u * b.y, z = iu * iu * a.z + 2 * iu * u * mz + u * u * b.z;
          pa.setXYZ(nv, px, py, pz); pa.setXYZ(nv + 1, x, y, z);
          ca.setXYZ(nv, kc.r * f, kc.g * f, kc.b * f); ca.setXYZ(nv + 1, kc.r * f, kc.g * f, kc.b * f);
          nv += 2; px = x; py = y; pz = z;
        }
      }
    }
    pa.needsUpdate = true; ca.needsUpdate = true;
  }
  P.arcs.geometry.setDrawRange(0, nv);

  const lf = st.loopFirst && inTravel ? Math.sin(Math.PI * k) : 0;
  P.flash.material.opacity = 0.85 * lf; P.flash.position.copy(cen);
  const outK = afterHead ? j - P.headIdx : 0;
  if (afterHead) { const u = clamp01(local / S0.dwell); P.emit.position.set(0, P.M.pl.headTop + 0.15 + 1.1 * u, 0); P.emit.material.opacity = 0.9 * (1 - u); } else P.emit.material.opacity = 0;
  setOutCount(outK);

  const cur = afterHead || st.type === "head" ? "out" : st.stage === "tokens" ? "" : st.stage;
  const loopTxt = P.r ? `Core loop ${Math.min(P.r, (st.loop == null ? P.r - 1 : st.loop) + 1)}/${P.r}` : "Core";
  for (const sp of $("playStage").querySelectorAll("span")) {
    const s = sp.dataset.s, on = s === cur;
    sp.classList.toggle("on", on);
    sp.textContent = s === "core" ? (cur === "core" ? loopTxt : cur === "coda" || cur === "out" ? `Core ×${P.r} done` : `Core ×${P.r}`) : STAGE_KEYS.find((q) => q[0] === s)[1];
  }
  let txt = "";
  const dm = P.d.config.d_model, ill = P.valid ? "" : " (illustrative)";
  if (st.type === "tokens") txt = `Prompt: ${T} real tokens from the model's own tokenizer. Press Play.`;
  else if (st.type === "embed") txt = `Embedding: each token id becomes a ${dm}-number vector.`;
  else if (st.type === "inject") txt = "Re-inject: the original embeddings e are mixed back into h (violet beam).";
  else if (st.type === "head") txt = `Output head: vector → scores over ${fmt(P.d.embedding.vocab)} tokens; the best one is the next token.`;
  else if (st.type === "out") txt = P.valid ? `Output token ${st.k + 1}/${P.nOut}: "${P.cont[st.k].piece}" (p = ${P.cont[st.k].top5[0].p})` : `Output token ${st.k + 1}: ? (no measured output for this model)`;
  else {
    const nm = KIND[st.kind].name, where = st.stage === "core" ? `core block ${st.idx + 1}, loop ${st.loop + 1}/${P.r}` : `${st.stage} block ${st.idx + 1}`;
    if (st.kind === "gdn") txt = `${nm}, ${where}: tokens stream into a fixed-size state; it never grows with length.`;
    else {
      txt = `${nm}, ${where}: arcs go from each token to the tokens it attends to; brighter = more weight${ill}.`;
      if (strongest) txt += ` Strongest: "${P.toks[strongest.q].vis}" → "${P.toks[strongest.s].vis}" (${strongest.w.toFixed(2)}).`;
      if (st.kind === "swa") txt += ` Window ${P.d.config.window} ≥ ${T}, so nothing is cut off.`;
    }
  }
  $("playDetail").textContent = txt;
  const lc = st.stage === "core" ? (st.type === "inject" ? -1 : st.loop) : (cur === "coda" || cur === "out") ? P.r - 1 : -1;
  if (lc !== P.lc) { P.lc = lc; renderDelta(P); }
}

function renderDelta(P) {
  const el = $("pDelta"); if (!el) return;
  if (!P.valid || !P.r) { el.innerHTML = P.r ? `<div class="pnote">Per-loop change ‖Δh‖/‖h‖ needs a measured trace.</div>` : ""; return; }
  const means = TRACE.delta.map((a) => a.reduce((x, y) => x + y, 0) / a.length), mx = Math.max(...means);
  el.innerHTML = `<div class="pnote">Per-loop change ‖Δh‖/‖h‖ (mean over tokens, measured)</div>` + means.map((v, l) => `<div class="drow${l <= P.lc ? " on" : ""}"><span>r${l + 1}</span><b style="width:${(12 + 88 * Math.sqrt(v / mx)).toFixed(0)}%"></b><em>${v.toFixed(2)}</em></div>`).join("");
}

function setOutCount(n) {
  const P = PLAY, strip = $("outStrip");
  if (n === P.outShown) return;
  while (P.outShown < n) {
    const k = P.outShown, b = document.createElement("button");
    b.type = "button"; b.className = "tkn out"; b.textContent = P.valid ? P.cont[k].piece : "?";
    b.dataset.k = k; strip.appendChild(b); P.outShown++;
  }
  while (P.outShown > n) { strip.lastChild.remove(); P.outShown--; }
  $("outRow").hidden = n === 0;
  strip.scrollLeft = strip.scrollWidth;
  const po = $("pOut"); if (po) po.textContent = P.valid ? unvis(P.cont.slice(0, n).map((c) => c.piece).join("")) : "";
}
function outTip(b) {
  const P = PLAY; if (!P) return;
  const tip = $("ptip"), k = +b.dataset.k;
  if (P.valid) {
    const c = P.cont[k];
    tip.innerHTML = `<b>step ${k + 1}</b> top-5 next-token probabilities<br>` + c.top5.map((t) => `<span class="${t.id === c.id ? "k" : ""}">${esc(t.piece)}</span> ${(t.p * 100).toFixed(1)}%`).join("<br>") + `<span class="sub">greedy: the top one is chosen</span>`;
  } else tip.innerHTML = `<b>?</b><span class="sub">Illustrative: no measured output for this model. Run experiments/viz/export_trace.py on a checkpoint of it.</span>`;
  tip.hidden = false;
  const r = b.getBoundingClientRect(), sr = stage.getBoundingClientRect();
  tip.style.left = Math.max(6, Math.min(sr.width - tip.offsetWidth - 6, r.left - sr.left)) + "px";
  tip.style.top = (r.bottom - sr.top + 6) + "px";
}

function renderPlayInfo(P) {
  const el = $("playInfo");
  if (!P) { el.innerHTML = ""; return; }
  const t = TRACE, m = t.model, c = t.check || {};
  const ids = t.tokens.map((x) => x.id).join(", ");
  const why = P.valid
    ? `Measured on the <b>${esc(m.experiment)}</b> checkpoint <code>${esc(m.run)}</code> (step ${fmt(m.step || 0)}), r = ${m.loops}. Arcs are real softmax attention weights (top ${t.top_k} per token, mean over heads). The continuation is greedy. Run on CPU in fp32 with a plain-PyTorch GDN (fla's kernel is GPU-only): CPU val loss ${c.cpu_val_loss_fp32} over ${c.cpu_val_seqs} seqs vs ${c.reported_val_loss != null ? (+c.reported_val_loss).toFixed(4) : "n/a"} reported (bf16, fla).`
    : esc(whyIllustrative(P.d));
  el.innerHTML = `<h3>PLAY · TOKEN TRACE</h3>
    <div class="pbadge ${P.valid ? "ok" : "ill"}">${P.valid ? "MEASURED" : "ILLUSTRATIVE"}</div>
    <p class="pwhy">${why}</p>
    <div class="pnote">Prompt <code>${esc(t.prompt.text_visible)}</code> → ${t.tokens.length} tokens, ids ${esc(ids)}</div>
    <div id="pDelta"></div>
    <div class="pnote">Output so far${P.valid ? "" : " (placeholders only)"}</div><pre id="pOut" class="pout"></pre>
    <details class="pnote"><summary>How to regenerate the real trace</summary><code>${esc(t.how_regenerated)}</code></details>`;
  renderDelta(P);
}

// ================================================================== Legend (generated from the same COL/KIND/PLAIN tables as the scene)
const hexCss = (n) => "#" + n.toString(16).padStart(6, "0");
const hasKind = (d, k) => d.sections.some((s) => s.blocks.some((b) => b.kind === k));
const hasSec = (d, n) => (d.sections.find((s) => s.name === n) || { blocks: [] }).blocks.length > 0;
const LEGEND = [
  { key: "attn", icon: "tri", c: COL.attn, name: "Full attention: wedge", meaning: "Footprint side ∝ context n, so area ∝ n². " + PLAIN.attn, has: (d) => hasKind(d, "attn") },
  { key: "swa", icon: "band", c: COL.swa, name: "Sliding window: band", meaning: "Band width = window w (fixed, 256); only its length grows with n. " + PLAIN.swa, has: (d) => hasKind(d, "swa") },
  { key: "gdn", icon: "cube", c: COL.gdn, name: "Gated DeltaNet: state cube", meaning: "Edge ∝ cube root of the state size (heads × d × d); constant at any context. " + PLAIN.gdn, has: (d) => hasKind(d, "gdn") },
  { key: "mlp", icon: "hex", c: COL.mlp, name: "MLP: hexagonal bicone", meaning: "Height = MLP parameters; width ratio = hidden/d. " + PLAIN.mlp, has: () => true },
  { key: "plinth", icon: "plinth", c: COL.emb, name: "Plinth: mixer weights", meaning: "Height = mixer parameters (1M = 1 unit). Coloured by block kind.", has: () => true },
  { key: "emb", icon: "cyl", c: COL.emb, name: "Embedding + scale-break rings", meaning: "Height = embedding parameters, capped at 1.6 units; the dark rings mean the true height is cut. Tied with the output head.", has: () => true },
  { key: "inj", icon: "disc", c: COL.inj, name: "Re-inject disc", meaning: "Height = injection parameters (2d → d). " + PLAIN.inj, has: (d) => d.looped },
  { key: "beam", icon: "dash", c: COL.inj, name: "Violet dashed beam", meaning: "Carries the original input e from the embedding to the re-inject disc.", has: (d) => d.looped },
  { key: "prelude", icon: "tag", c: COL.gold, name: "Prelude (PRELUDE tag)", meaning: "Blocks that run once, before the loop. Stack height = their parameters.", has: (d) => hasSec(d, "prelude") },
  { key: "ring", icon: "ring", c: COL.gold, name: "Core ring: CORE × r", meaning: "The looped blocks stored once; r on the label = loop count; the orbiting glows spin faster as r grows. " + PLAIN.ring, has: (d) => d.looped },
  { key: "ghost", icon: "ghost", c: COL.gold, name: "Ghost loop pillars", meaning: "Translucent copies, one per extra loop: 0 new parameters, one more pass of compute each.", has: (d) => d.looped && S.unroll && S.loops > 1 },
  { key: "helix", icon: "helix", c: COL.orange, name: "Helix path", meaning: "The route of the computation: one turn per loop, climbing one pitch each time.", has: (d) => d.looped && S.unroll && S.loops > 1 },
  { key: "same", icon: "dash", c: COL.gold, name: "“same weights” lines", meaning: "Dashed gold lines tie each real block to its ghost copies (shared weights).", has: (d) => d.looped && S.unroll && S.loops > 1 },
  { key: "coda", icon: "tag", c: COL.gold, name: "Coda (CODA tag)", meaning: "Blocks that run once, after the loop.", has: (d) => hasSec(d, "coda") },
  { key: "head", icon: "head", c: COL.emb, name: "Output head", meaning: "Vector → next-token scores. Tied to the embedding, so 0 extra parameters. " + PLAIN.head, has: () => true },
  { key: "tern", icon: "tern", c: 0x4a90ff, name: "Ternary speckle + 1.58-bit badge", meaning: "A −1/0/+1 texture marks BitLinear layers; the gold badge gives the % of weights that are ternary.", has: (d) => d.ternary.enabled },
  { key: "ki", icon: "ki", c: COL.gold, name: "Gold ki particles", meaning: "Data flow: up the prelude, around the core r times, out the coda. Speed is decorative; the order is real.", has: () => true },
  { key: "feed", icon: "ki2", c: COL.orange, name: "Orange feed particles", meaning: "Tokens being written into a GDN memory cube.", has: (d) => hasKind(d, "gdn") },
  { key: "chips", icon: "tkn", c: 0xcfd6ff, name: "Prompt token chips", meaning: "The real tokens of the prompt, in order. Each has a coloured dot: the token's colour in the animation.", has: () => !!TRACE, play: true },
  { key: "arcs", icon: "arc", c: COL.attn, name: "Attention arcs", meaning: "From a token to a token it attends to. Opacity = attention weight (mean over heads, top 3). Blue = full attention, green = sliding window.", has: (d) => hasKind(d, "attn") || hasKind(d, "swa"), play: true },
  { key: "outrow", icon: "out", c: 0xffe27a, name: "Output row", meaning: "Next tokens appear one by one. Hover or tap a chip for its top-5 probabilities.", has: () => !!TRACE, play: true },
  { key: "aura", icon: "glow", c: COL.orange, name: "Aura and glow", meaning: "Decorative: a soft pulse around the model and ring. It encodes nothing.", has: () => true, deco: true },
  { key: "stars", icon: "dots", c: 0xffffff, name: "Starfield and ground grid", meaning: "Decorative backdrop. It encodes nothing.", has: () => true, deco: true },
];
function legendTargets(key) {
  const out = [];
  if (key === "chips") return PLAY ? [...PLAY.chipObjs] : [];
  if (key === "outrow" || key === "stars") return [];
  const hit = (o) => {
    const u = o.userData || {};
    if (key === "prelude" || key === "coda") return u.sec === key && /(^| )plinth/.test(u.leg || "");
    return (u.leg || "").split(" ").includes(key);
  };
  for (const M of models) M.grp.traverse((o) => { if (out.length < 14 && hit(o)) out.push(o); });
  return out;
}
function clearHalos() { for (const h of halos) { scene.remove(h.sp); h.sp.material.dispose(); } if (halos.length) dirty = true; halos.length = 0; }
function highlight(key) {
  clearHalos();
  $("outRow").classList.toggle("hl", key === "outrow");
  const row = LEGEND.find((l) => l.key === key);
  for (const o of legendTargets(key)) {
    let c, sz;
    if (o.isCSS2DObject) { c = o.getWorldPosition(V3()); sz = 0.6; }
    else { const b = new THREE.Box3().setFromObject(o); if (b.isEmpty()) continue; c = b.getCenter(V3()); sz = b.getSize(V3()).length(); }
    const sp = sprite(0xffffff, Math.min(7, sz * 1.5 + 0.9), 0.6); sp.position.copy(c); sp.material.color.set(row ? hexCss(row.c) : "#ffd23f");
    scene.add(sp); halos.push({ sp });
  }
  dirty = true;
}
function focusLegend(key) {
  const t = legendTargets(key)[0]; if (!t) return;
  if (t.isCSS2DObject) { const w = t.getWorldPosition(V3()); flyTo(w, w.clone().add(V3(2, 1.5, 5)), 0.8); $("focusNote").hidden = false; return; }
  focusOn(t);
}
function renderLegend() {
  const d = models[0] && models[0].d; if (!d) return;
  $("legendList").innerHTML = LEGEND.map((l) => {
    const on = l.has(d), css = hexCss(l.c);
    return `<button type="button" class="lcard lrow${on ? "" : " off"}" style="--c:${css}" data-key="${l.key}"><i class="sw ${l.icon}" aria-hidden="true"></i><div><b>${esc(l.name)}</b>${l.play ? ' <span class="lt">Play mode</span>' : ""}${l.deco ? ' <span class="lt">decorative</span>' : ""}${on ? "" : ' <span class="lt no">not in this model</span>'}<p>${esc(l.meaning)}</p></div></button>`;
  }).join("");
}

// ================================================================== extra UI: keyboard, screenshot, legend wiring
function initUI2() {
  const L = $("legendList");
  L.addEventListener("pointerover", (e) => { const r = e.target.closest(".lrow"); if (r && !r.classList.contains("off")) highlight(r.dataset.key); });
  L.addEventListener("pointerleave", () => { clearHalos(); $("outRow").classList.remove("hl"); });
  L.addEventListener("focusin", (e) => { const r = e.target.closest(".lrow"); if (r && !r.classList.contains("off")) highlight(r.dataset.key); });
  L.addEventListener("focusout", () => { clearHalos(); $("outRow").classList.remove("hl"); });
  L.addEventListener("click", (e) => { const r = e.target.closest(".lrow"); if (r && !r.classList.contains("off")) { stage.scrollIntoView({ block: "nearest", behavior: S.motion ? "smooth" : "auto" }); focusLegend(r.dataset.key); } });
  const lp = $("legendPanel");
  try { lp.open = window.matchMedia("(min-width: 521px)").matches; } catch (e) { lp.open = true; }

  $("btnPlay").addEventListener("click", playToggle);
  $("btnStep").addEventListener("click", playStep);
  $("btnBack").addEventListener("click", playBack);
  $("chkFollow").addEventListener("change", () => { if (!$("chkFollow").checked) fitCamera(true); });
  $("btnRestart").addEventListener("click", playRestart);
  $("rngSpeed").addEventListener("input", () => { S.speed = +$("rngSpeed").value; $("speedVal").textContent = S.speed + "×"; if (PLAY) PLAY.speed = S.speed; });
  S.speed = +$("rngSpeed").value;
  $("btnMotion").addEventListener("click", paintPlayBtn);
  $("outStrip").addEventListener("pointerover", (e) => { const b = e.target.closest(".tkn.out"); if (b) outTip(b); });
  $("outStrip").addEventListener("click", (e) => { const b = e.target.closest(".tkn.out"); if (b) outTip(b); });
  $("outStrip").addEventListener("focusin", (e) => { const b = e.target.closest(".tkn.out"); if (b) outTip(b); });
  $("outRow").addEventListener("pointerleave", () => { $("ptip").hidden = true; });
  $("btnShot").addEventListener("click", () => {
    renderer.render(scene, camera);
    $("gl").toBlob((b) => { if (!b) return; const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `gogeta-${S.a}-r${S.loops}.png`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); });
  });
  const KEYS = $("keysPanel");
  window.addEventListener("keydown", (e) => {
    const tg = e.target; if (tg && /^(INPUT|SELECT|TEXTAREA)$/.test(tg.tagName)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === " " && !(tg && tg.tagName === "BUTTON")) { e.preventDefault(); playToggle(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); playStep(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); playBack(); }
    else if (e.key === "r" || e.key === "R") { playRestart(); fitCamera(true); }
    else if (e.key === "Escape") { fitCamera(true); clearHalos(); }
    else if (e.key === "?") { KEYS.open = !KEYS.open; }
    else if (/^[1-7]$/.test(e.key)) { const n = DATA.models[+e.key - 1]; if (n) { $("selA").value = n.name; $("selA").dispatchEvent(new Event("change", { bubbles: true })); } }
  });
}

// ================================================================== self test (leak regression)
window.__arch3dSelfTest = async function (rounds = 20) {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const snap = () => ({ labels: labelRenderer.domElement.children.length, geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures, sceneChildren: scene.children.length, halos: halos.length });
  const A = window.__arch3d, start = { model: S.a, loops: S.loops, ctx: S.ctxIdx, compare: S.compare, unroll: S.unroll };
  clearHalos(); A.setCompare(false); A.setUnroll(true); await wait(1300);
  const initial = snap();
  for (let i = 0; i < rounds; i++) {
    for (const n of A.names) { A.setModel(n); renderer.render(scene, camera); }
    A.setModel(start.model); A.setCompare(true); A.setModelB(A.names[(i + 1) % A.names.length]); A.setCompare(false);
    for (let r = 1; r <= 12; r++) A.setLoops(r);
    for (let r = 11; r >= start.loops; r--) A.setLoops(r);
    for (let c = 0; c <= 5; c++) A.setCtxIndex(c);
    A.setCtxIndex(start.ctx); A.setUnroll(false); A.setUnroll(true);
    if (A.play.P) { A.play.seek(A.play.P.total); A.play.restart(); }
    renderer.render(scene, camera);
  }
  A.setModel(start.model); A.setLoops(start.loops); A.setCtxIndex(start.ctx);
  await wait(1400); if (burst) { disposeGroup(burst.pts); burst = null; } // a burst is removed by the frame loop; force it when the tab is throttled
  renderer.render(scene, camera);
  const final = snap();
  const same = final.labels === initial.labels && final.sceneChildren === initial.sceneChildren && final.halos === initial.halos;
  return { rounds, initial, final, geometryDelta: final.geometries - initial.geometries, textureDelta: final.textures - initial.textures, note: "geometry/texture deltas are shared caches by design (e.g. the ternary speckle textures, once E3 has been shown); labels and scene children must match exactly", pass: same && final.textures - initial.textures <= 6 && Math.abs(final.geometries - initial.geometries) <= 40 };
};
