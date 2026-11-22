// Hover layer for the generated experiment charts (tools/site_pages.py).
(function () {
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function place(tip, wrap, clientX, clientY) {
    const r = wrap.getBoundingClientRect();
    let x = clientX - r.left + 14, y = clientY - r.top + 14;
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > r.width) x = clientX - r.left - tw - 14;
    if (y + th > r.height) y = Math.max(0, clientY - r.top - th - 14);
    tip.style.left = Math.max(0, x) + "px";
    tip.style.top = y + "px";
  }

  function svgPoint(svg, clientX, clientY) {
    const pt = svg.createSVGPoint();
    pt.x = clientX; pt.y = clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  // ---- learning curves: crosshair + multi-series tooltip
  document.querySelectorAll('figure.viz-root[data-kind="curves"]').forEach((fig) => {
    const series = JSON.parse(fig.dataset.series);
    const svg = fig.querySelector("svg"), wrap = fig.querySelector(".viz-wrap");
    const tip = fig.querySelector(".tooltip"), hair = fig.querySelector(".crosshair"), hit = fig.querySelector(".hit");
    const x0 = +fig.dataset.x0, x1 = +fig.dataset.x1, xmax = +fig.dataset.xmax;
    const xs = series[0].pts.map((p) => p[0]);
    const move = (ev) => {
      const p = svgPoint(svg, ev.clientX, ev.clientY);
      const xv = ((p.x - x0) / (x1 - x0)) * xmax;
      let k = 0;
      xs.forEach((x, i) => { if (Math.abs(x - xv) < Math.abs(xs[k] - xv)) k = i; });
      const sx = x0 + (xs[k] / xmax) * (x1 - x0);
      hair.setAttribute("x1", sx); hair.setAttribute("x2", sx); hair.style.display = "";
      tip.innerHTML = `<b>${xs[k].toFixed(0)}M tokens</b>` + series
        .map((s) => `<div><i style="background:${esc(s.color)}"></i>${esc(s.id)} <span>${s.pts[k][1].toFixed(4)}</span></div>`).join("");
      place(tip, wrap, ev.clientX, ev.clientY);
    };
    const leave = () => { tip.hidden = true; hair.style.display = "none"; };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", leave);
  });

  // ---- dot charts: per-point tooltip
  document.querySelectorAll('figure.viz-root[data-kind="dots"]').forEach((fig) => {
    const wrap = fig.querySelector(".viz-wrap"), tip = fig.querySelector(".tooltip");
    fig.querySelectorAll(".dot-hit").forEach((h) => {
      const show = (ev) => { tip.textContent = h.dataset.tip; place(tip, wrap, ev.clientX, ev.clientY); };
      h.addEventListener("pointerenter", show);
      h.addEventListener("pointermove", show);
      h.addEventListener("pointerdown", show);
      h.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
  });
})();
