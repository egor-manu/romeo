// Interactive results: mask controls, noise performance and the explorer of held-out experimental fields.
// Data are exported from the paper's repository by tools/export_paper_data.py.

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const NS = "http://www.w3.org/2000/svg";
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { s1: css("--s1"), s2: css("--s2"), s3: css("--s3"), neutral: css("--neutral"), surface: css("--panel"), ink3: css("--ink-3"), good: css("--good") };

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) k === "text" ? (node.textContent = v) : node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const pow10 = (e) => "10" + String(e).split("").map((c) => SUP[c]).join("");
function sci(v) {
  const e = Math.floor(Math.log10(v)), m = v / 10 ** e;
  return `${m.toFixed(1)}×${pow10(e)}`;
}
const pct = (v, digits = 1) => `${(100 * v).toFixed(digits)}%`;

// ---------- tooltip ----------

const tip = $("#tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
const hideTip = () => tip.classList.remove("on");
addEventListener("scroll", hideTip, { passive: true });

function frame(host, margin) {
  host.innerHTML = "";
  const W = host.clientWidth, H = host.clientHeight;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, host);
  return { svg, W, H, x0: margin.l, x1: W - margin.r, y0: margin.t, y1: H - margin.b };
}

// ---------- data ----------

let paper, buffer;
try {
  [paper, buffer] = await Promise.all([
    fetch("data/paper.json").then((r) => r.json()),
    fetch("data/paper.bin").then((r) => r.arrayBuffer()),
  ]);
} catch (err) {
  $$(".chart").forEach((c) => { c.innerHTML = '<p class="loading">Could not load the results data.</p>'; });
  throw err;
}
const f32 = (e) => new Float32Array(buffer, e.offset, e.length);

// ---------- mask controls: condition number against light capture ----------

const masks = { zoom: "all" };
const GRAINS = [1, 2, 4, 8, 16, 32];

function drawMasks() {
  const host = $("#chart-masks"), f = frame(host, { l: 54, r: 18, t: 30, b: 44 });
  const { svg, x0, x1, y0, y1 } = f;
  const [xa, xb] = masks.zoom === "all" ? [38, 100.4] : [96, 100.1];
  const [la, lb] = [Math.log10(5), Math.log10(2e4)];
  const X = (v) => x0 + ((v - xa) / (xb - xa)) * (x1 - x0);
  const Y = (k) => y1 - ((Math.log10(k) - la) / (lb - la)) * (y1 - y0);

  for (const k of [10, 100, 1000, 10000]) {
    el("line", { class: "grid", x1: x0, x2: x1, y1: Y(k), y2: Y(k) }, svg);
    el("text", { x: x0 - 8, y: Y(k) + 4, "text-anchor": "end", text: pow10(Math.log10(k)) }, svg);
  }
  const xticks = masks.zoom === "all" ? [40, 50, 60, 70, 80, 90, 100] : [96, 97, 98, 99, 100];
  for (const t of xticks) el("text", { x: X(t), y: y1 + 18, "text-anchor": "middle", text: `${t}%` }, svg);
  el("text", { class: "axis-title", x: (x0 + x1) / 2, y: y1 + 38, "text-anchor": "middle", text: "Mean light captured on the detector region" }, svg);
  el("text", { class: "axis-title", x: x0 - 40, y: y0 - 14, text: "PMP condition number κ (lower is better)" }, svg);

  // target region: at least 98% capture and κ below 25
  el("rect", { x: X(98), y: Y(25), width: x1 - X(98), height: y1 - Y(25), fill: C.good, opacity: 0.09 }, svg);
  el("line", { class: "target", x1: X(98), x2: X(98), y1: y0, y2: y1 }, svg);
  el("line", { class: "target", x1: x0, x2: x1, y1: Y(25), y2: Y(25) }, svg);
  el("text", { class: "lbl", x: X(98) - 6, y: y0 + 10, "text-anchor": "end", text: "98% capture" }, svg);
  el("text", { class: "lbl", x: x0 + 6, y: Y(25) - 6, text: "κ = 25" }, svg);

  const clip = el("clipPath", { id: "clip-masks" }, svg);
  el("rect", { x: x0, y: y0 - 8, width: x1 - x0 + 10, height: y1 - y0 + 16 }, clip);
  const layer = el("g", { "clip-path": "url(#clip-masks)" }, svg);
  const points = [];
  const order = { random: 0, flat: 1, optimised: 2 };
  for (const p of [...paper.controls].sort((a, b) => order[a.family] - order[b.family])) {
    const cx = X(100 * p.capture), cy = Y(p.kappa);
    let node;
    if (p.family === "flat") node = el("rect", { x: cx - 5.5, y: cy - 5.5, width: 11, height: 11, rx: 2, fill: C.neutral, stroke: C.surface, "stroke-width": 2 }, layer);
    else node = el("circle", { cx, cy, r: p.family === "optimised" ? 6 : 4.2, fill: p.family === "optimised" ? C.s2 : C.s1, stroke: C.surface, "stroke-width": 2 }, layer);
    points.push({ p, cx, cy, node });
  }

  // direct labels for the random-mask clusters
  for (const g of GRAINS) {
    const group = paper.controls.filter((p) => p.family === "random" && p.grain === g);
    const cap = (100 * group.reduce((s, p) => s + p.capture, 0)) / group.length;
    const kap = 10 ** (group.reduce((s, p) => s + Math.log10(p.kappa), 0) / group.length);
    // on narrow screens only the two extreme clusters are labelled; the tooltip names the rest
    if (cap < xa || (x1 - x0 < 520 && masks.zoom === "all" && g !== 1 && g !== 32)) continue;
    // keep the label clear of the κ = 25 target line
    const below = Math.abs(Y(kap) - 10 - Y(25)) < 14;
    const left = X(cap) - x0 < 110; // too close to the axis to fit a label on its left
    el("text", { class: "lbl", x: X(cap) + (left ? 12 : -12), y: Y(kap) + (below ? 18 : -10), "text-anchor": left ? "start" : "end", text: `${g} px cells` }, svg);
  }
  const best = paper.controls.filter((p) => p.family === "optimised");
  const bx = X((100 * best.reduce((s, p) => s + p.capture, 0)) / best.length), by = Y(Math.min(...best.map((p) => p.kappa)));
  el("text", { class: "lbl", x: bx - 12, y: by + 22, "text-anchor": "end", text: "optimised" }, svg);

  const ring = el("circle", { r: 10, fill: "none", stroke: "#fff", "stroke-width": 2, visibility: "hidden" }, svg);
  const hit = el("rect", { class: "hit", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, svg);
  const names = { random: (p) => `Random mask, ${p.grain} px cells`, optimised: () => "Optimised mask", flat: () => "Flat-phase SLM" };
  const move = (ev) => {
    const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let near = null, dist = 28;
    for (const pt of points) {
      if (pt.cx < x0 - 1 || pt.cx > x1 + 10) continue;
      const d = Math.hypot(pt.cx - mx, pt.cy - my);
      if (d < dist) { dist = d; near = pt; }
    }
    if (!near) { ring.setAttribute("visibility", "hidden"); return hideTip(); }
    ring.setAttribute("cx", near.cx); ring.setAttribute("cy", near.cy); ring.setAttribute("visibility", "visible");
    const p = near.p;
    showTip(`<b>${names[p.family](p)}</b><br>κ = ${p.kappa < 100 ? p.kappa.toFixed(1) : Math.round(p.kappa).toLocaleString("en-GB")}<br>mean capture ${pct(p.capture)}<br>worst-case capture ${pct(p.worst)}`, ev.clientX, ev.clientY);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { ring.setAttribute("visibility", "hidden"); hideTip(); });
}

$$("#mask-zoom button").forEach((b) => b.addEventListener("click", () => {
  $$("#mask-zoom button").forEach((x) => x.classList.toggle("on", x === b));
  masks.zoom = b.dataset.z;
  drawMasks();
}));

// ---------- noise performance: error against SNR for each mode count ----------

const PEAK_OFFSET_DB = 24.9; // 20 log10(I_max/σ) − full-sensor SNR, median over all simulated trials
const modeCounts = [...new Set(paper.simulation.map((r) => r.modes))].sort((a, b) => a - b);
const series = new Map(modeCounts.map((m) => [m, paper.simulation.filter((r) => r.modes === m).sort((a, b) => a.snr - b.snr)]));
const noise = { modes: 42 };

function drawNoise() {
  const host = $("#chart-noise"), f = frame(host, { l: 54, r: 74, t: 30, b: 44 });
  const { svg, x0, x1, y0, y1 } = f;
  const [xa, xb] = [-15, 32], [la, lb] = [-9, -2];
  const X = (v) => x0 + ((v - xa) / (xb - xa)) * (x1 - x0);
  const Y = (v) => y1 - ((Math.log10(v) - la) / (lb - la)) * (y1 - y0);

  for (let e = la; e <= lb; e++) {
    el("line", { class: "grid", x1: x0, x2: x1, y1: Y(10 ** e), y2: Y(10 ** e) }, svg);
    el("text", { x: x0 - 8, y: Y(10 ** e) + 4, "text-anchor": "end", text: pow10(e) }, svg);
  }
  for (const t of x1 - x0 < 420 ? [-10, 0, 10, 20, 30] : [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]) el("text", { x: X(t), y: y1 + 18, "text-anchor": "middle", text: t }, svg);
  el("text", { class: "axis-title", x: (x0 + x1) / 2, y: y1 + 38, "text-anchor": "middle", text: "Full-sensor signal-to-noise ratio (dB)" }, svg);
  el("text", { class: "axis-title", x: x0 - 40, y: y0 - 14, text: "Median coefficient error (NMSE)" }, svg);

  const path = (rows, key) => rows.map((r, i) => `${i ? "L" : "M"}${X(r.snr).toFixed(1)},${Y(r[key]).toFixed(1)}`).join("");
  const selectedEnd = Y(series.get(noise.modes).at(-1).median);
  for (const m of modeCounts) {
    if (m === noise.modes) continue;
    const rows = series.get(m);
    el("path", { d: path(rows, "median"), fill: "none", stroke: C.s1, "stroke-width": 1.5, opacity: 0.55 }, svg);
    const last = rows.at(-1);
    if ((m === modeCounts[0] || m === modeCounts.at(-1)) && Math.abs(Y(last.median) - selectedEnd) > 16) {
      el("text", { x: X(last.snr) + 8, y: Y(last.median) + 4, text: `${m} modes` }, svg);
    }
  }
  const rows = series.get(noise.modes);
  const band = rows.map((r, i) => `${i ? "L" : "M"}${X(r.snr).toFixed(1)},${Y(r.p95).toFixed(1)}`).join("")
    + [...rows].reverse().map((r) => `L${X(r.snr).toFixed(1)},${Y(r.p05).toFixed(1)}`).join("") + "Z";
  el("path", { d: band, fill: C.s2, opacity: 0.2 }, svg);
  el("path", { d: path(rows, "median"), fill: "none", stroke: C.s2, "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);
  for (const r of rows) el("circle", { cx: X(r.snr), cy: Y(r.median), r: 3.5, fill: C.s2, stroke: C.surface, "stroke-width": 1.5 }, svg);
  const last = rows.at(-1);
  el("text", { class: "lbl", x: X(last.snr) + 8, y: Y(last.median) + 4, "font-weight": 600, text: `${noise.modes} modes` }, svg);

  const cross = el("line", { stroke: C.ink3, "stroke-width": 1, y1: y0, y2: y1, visibility: "hidden" }, svg);
  const dot = el("circle", { r: 6, fill: C.s2, stroke: "#fff", "stroke-width": 2, visibility: "hidden" }, svg);
  const hit = el("rect", { class: "hit", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, svg);
  const move = (ev) => {
    const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left;
    const row = rows.reduce((a, b) => (Math.abs(X(b.snr) - mx) < Math.abs(X(a.snr) - mx) ? b : a));
    cross.setAttribute("x1", X(row.snr)); cross.setAttribute("x2", X(row.snr)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(row.snr)); dot.setAttribute("cy", Y(row.median)); dot.setAttribute("visibility", "visible");
    const peak = 10 ** ((row.snr + PEAK_OFFSET_DB) / 20);
    showTip(`<b>${noise.modes} modes · ${row.snr} dB</b><br>brightest pixel ≈ ${peak < 10 ? peak.toFixed(1) : Math.round(peak)}× the noise<br>median error ${sci(row.median)}<br>relative error ${pct(Math.sqrt(row.median), 2)}<br>5th–95th percentile ${sci(row.p05)} – ${sci(row.p95)}`, ev.clientX, ev.clientY);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); });

  const at13 = rows.find((r) => r.snr === 13);
  $("#noise-note").textContent = `At 13 dB, where the brightest pixel is about 80 times the noise, the median error for ${noise.modes} modes is ${sci(at13.median)}: a relative error of ${pct(Math.sqrt(at13.median), 2)}. The error falls in proportion to the noise power and grows smoothly with the number of modes. A separate inverse is computed for each fibre.`;
}

const chipHost = $("#noise-modes");
for (const m of modeCounts) {
  const b = document.createElement("button");
  b.textContent = m; b.classList.toggle("on", m === noise.modes); b.setAttribute("aria-label", `${m} modes`);
  b.addEventListener("click", () => {
    noise.modes = m;
    [...chipHost.children].forEach((x) => x.classList.toggle("on", x === b));
    drawNoise();
  });
  chipHost.append(b);
}

// ---------- explorer of held-out experimental fields ----------

const basis = f32(paper.basis.fields), NPIX = paper.basis.pixels ** 2, M = paper.basis.modes;
const cases = {};
for (const [key, c] of Object.entries(paper.experiment)) {
  const nmse = f32(c.nmse);
  cases[key] = {
    ...c, nmse, nmseLp: f32(c.nmseLp), coherence: f32(c.coherence),
    holographic: f32(c.holographic), recovered: f32(c.recovered),
    order: [...nmse.keys()].sort((a, b) => nmse[a] - nmse[b]),
    names: c.labels.map((l) => l.replace("-1", " a").replace("-2", " b")),
  };
}
const ex = { key: "diffuser", rank: 0 };

function hsv(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p]; case 1: return [q, v, p]; case 2: return [p, v, t];
    case 3: return [p, q, v]; case 4: return [t, p, v]; default: return [v, p, q];
  }
}

function synthesise(re, im) {
  const er = new Float32Array(NPIX), ei = new Float32Array(NPIX);
  for (let k = 0; k < M; k++) {
    const a = re[k], b = im[k], o = k * NPIX;
    for (let p = 0; p < NPIX; p++) { const v = basis[o + p]; er[p] += a * v; ei[p] += b * v; }
  }
  return { er, ei };
}

function paint(canvas, field, max) {
  const ctx = canvas.getContext("2d"), n = paper.basis.pixels, img = ctx.createImageData(n, n);
  for (let p = 0; p < NPIX; p++) {
    const level = Math.min(1, (Math.hypot(field.er[p], field.ei[p]) / max) ** 2); // brightness = intensity
    const [r, g, b] = hsv((Math.atan2(field.ei[p], field.er[p]) / (2 * Math.PI) + 1) % 1, 0.9, level);
    img.data[4 * p] = r * 255; img.data[4 * p + 1] = g * 255; img.data[4 * p + 2] = b * 255; img.data[4 * p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function coefficients(c, i) {
  const h = { re: new Float64Array(M), im: new Float64Array(M) }, r = { re: new Float64Array(M), im: new Float64Array(M) };
  for (let k = 0; k < M; k++) {
    h.re[k] = c.holographic[2 * (i * M + k)]; h.im[k] = c.holographic[2 * (i * M + k) + 1];
    r.re[k] = c.recovered[2 * (i * M + k)]; r.im[k] = c.recovered[2 * (i * M + k) + 1];
  }
  // align the reconstruction to the reference with one complex scalar, α = (rᴴh)/(rᴴr)
  let nr = 0, ni = 0, d = 0;
  for (let k = 0; k < M; k++) {
    nr += r.re[k] * h.re[k] + r.im[k] * h.im[k];
    ni += r.re[k] * h.im[k] - r.im[k] * h.re[k];
    d += r.re[k] ** 2 + r.im[k] ** 2;
  }
  const ar = nr / d, ai = ni / d, a = { re: new Float64Array(M), im: new Float64Array(M) };
  for (let k = 0; k < M; k++) { a.re[k] = ar * r.re[k] - ai * r.im[k]; a.im[k] = ar * r.im[k] + ai * r.re[k]; }
  return { h, a };
}

function drawWeights(c, h, a) {
  const host = $("#chart-weights"), f = frame(host, { l: 40, r: 8, t: 12, b: 34 });
  const { svg, x0, x1, y0, y1 } = f;
  const power = (v) => { const p = Array.from(v.re, (x, k) => x * x + v.im[k] ** 2), s = p.reduce((q, w) => q + w, 0); return p.map((w) => w / s); };
  const ph = power(h), pa = power(a);
  const top = Math.max(0.1, Math.ceil(Math.max(...ph, ...pa) * 10) / 10);
  const step = (x1 - x0) / M, X = (k) => x0 + (k + 0.5) * step, Y = (v) => y1 - (v / top) * (y1 - y0);
  for (const t of [0, top / 2, top]) {
    el("line", { class: "grid", x1: x0, x2: x1, y1: Y(t), y2: Y(t) }, svg);
    el("text", { x: x0 - 6, y: Y(t) + 4, "text-anchor": "end", text: `${Math.round(100 * t)}%` }, svg);
  }
  // group ticks by azimuthal index l
  let startIndex = 0;
  const azimuth = (name) => +name[2];
  for (let k = 1; k <= M; k++) {
    if (k === M || azimuth(c.names[k]) !== azimuth(c.names[startIndex])) {
      el("text", { x: (X(startIndex) + X(k - 1)) / 2, y: y1 + 16, "text-anchor": "middle", text: step * 2 > 26 ? `l=${azimuth(c.names[startIndex])}` : azimuth(c.names[startIndex]) }, svg);
      if (k < M) el("line", { class: "grid", x1: x0 + k * step, x2: x0 + k * step, y1: y1, y2: y1 + 6 }, svg);
      startIndex = k;
    }
  }
  el("text", { class: "axis-title", x: (x0 + x1) / 2, y: y1 + 31, "text-anchor": "middle", text: "LP modes, grouped by azimuthal index l" }, svg);
  const w = Math.max(2, Math.min(10, step - 3));
  for (let k = 0; k < M; k++) {
    const yv = Y(ph[k]);
    if (y1 - yv > 0.5) el("path", { d: `M${X(k) - w / 2},${y1}V${Math.min(y1, yv + 2)}Q${X(k) - w / 2},${yv} ${X(k)},${yv}Q${X(k) + w / 2},${yv} ${X(k) + w / 2},${Math.min(y1, yv + 2)}V${y1}Z`, fill: C.s1 }, svg);
  }
  for (let k = 0; k < M; k++) el("circle", { cx: X(k), cy: Y(pa[k]), r: 3.6, fill: C.s3, stroke: C.surface, "stroke-width": 1.5 }, svg);
  const marker = el("rect", { y: y0, width: step, height: y1 - y0, fill: "#fff", opacity: 0.07, visibility: "hidden" }, svg);
  const hit = el("rect", { class: "hit", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, svg);
  const move = (ev) => {
    const r = svg.getBoundingClientRect(), k = Math.max(0, Math.min(M - 1, Math.floor((ev.clientX - r.left - x0) / step)));
    marker.setAttribute("x", x0 + k * step); marker.setAttribute("visibility", "visible");
    showTip(`<b>${c.names[k]}</b><br>reference ${pct(ph[k])}<br>from intensity ${pct(pa[k])}`, ev.clientX, ev.clientY);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { marker.setAttribute("visibility", "hidden"); hideTip(); });
}

function drawHistogram(c, index) {
  const host = $("#chart-hist"), f = frame(host, { l: 34, r: 10, t: 12, b: 34 });
  const { svg, x0, x1, y0, y1 } = f;
  const values = Array.from(c.nmse, (v) => 100 * v), sorted = [...values].sort((a, b) => a - b);
  const hi = Math.ceil(sorted[Math.floor(0.98 * (sorted.length - 1))]), BINS = 24, bw = hi / BINS;
  const bins = Array.from({ length: BINS }, () => []);
  values.forEach((v, i) => bins[Math.min(BINS - 1, Math.floor(v / bw))].push(i));
  const top = Math.max(...bins.map((b) => b.length));
  const X = (v) => x0 + (Math.min(v, hi) / hi) * (x1 - x0), Y = (n) => y1 - (n / top) * (y1 - y0);
  for (const t of [0, top]) el("text", { x: x0 - 6, y: Y(t) + 4, "text-anchor": "end", text: t }, svg);
  el("line", { class: "grid", x1: x0, x2: x1, y1: y1, y2: y1 }, svg);
  const ticks = hi <= 8 ? 2 : 5;
  for (let t = 0; t <= hi; t += ticks) el("text", { x: X(t), y: y1 + 16, "text-anchor": "middle", text: t === hi ? `${t}%+` : `${t}%` }, svg);
  el("text", { class: "axis-title", x: (x0 + x1) / 2, y: y1 + 31, "text-anchor": "middle", text: "Complex-field error against the hologram" }, svg);
  const bars = bins.map((b, i) => {
    const xa = X(i * bw) + 1, w = X((i + 1) * bw) - X(i * bw) - 2, yv = Y(b.length);
    if (!b.length) return null;
    const r = Math.min(3, w / 2, (y1 - yv) / 2);
    return el("path", { d: `M${xa},${y1}V${yv + r}Q${xa},${yv} ${xa + r},${yv}H${xa + w - r}Q${xa + w},${yv} ${xa + w},${yv + r}V${y1}Z`, fill: C.s1 }, svg);
  });
  const median = 100 * c.medianNmse;
  el("line", { x1: X(median), x2: X(median), y1: y0, y2: y1, stroke: C.ink3, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, svg);
  const here = X(values[index]);
  el("line", { x1: here, x2: here, y1: y0, y2: y1, stroke: C.s2, "stroke-width": 2.5 }, svg);
  el("path", { d: `M${here - 5},${y0 - 2}h10l-5,7z`, fill: C.s2 }, svg);
  const hit = el("rect", { class: "hit", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, svg);
  const binAt = (ev) => Math.max(0, Math.min(BINS - 1, Math.floor(((ev.clientX - svg.getBoundingClientRect().left - x0) / (x1 - x0)) * BINS)));
  const move = (ev) => {
    const i = binAt(ev);
    bars.forEach((b, j) => b?.setAttribute("opacity", j === i ? 1 : 0.6));
    const range = i === BINS - 1 ? `${(i * bw).toFixed(1)}% and above` : `${(i * bw).toFixed(1)}–${((i + 1) * bw).toFixed(1)}%`;
    showTip(`<b>${range}</b><br>${bins[i].length} field${bins[i].length === 1 ? "" : "s"}${bins[i].length ? "<br>click to open one" : ""}`, ev.clientX, ev.clientY);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", () => { bars.forEach((b) => b?.setAttribute("opacity", 1)); hideTip(); });
  hit.addEventListener("click", (ev) => {
    const members = bins[binAt(ev)];
    if (members.length) { ex.rank = c.order.indexOf(members[Math.floor(Math.random() * members.length)]); drawExplorer(); }
  });
}

function drawExplorer() {
  const c = cases[ex.key], i = c.order[ex.rank];
  const { h, a } = coefficients(c, i);
  const ref = synthesise(h.re, h.im), rec = synthesise(a.re, a.im);
  const res = { er: rec.er.map((v, p) => v - ref.er[p]), ei: rec.ei.map((v, p) => v - ref.ei[p]) };
  let max = 0;
  for (let p = 0; p < NPIX; p++) max = Math.max(max, Math.hypot(ref.er[p], ref.ei[p]));
  paint($("#ex-ref"), ref, max); paint($("#ex-rec"), rec, max); paint($("#ex-res"), res, max);
  $("#ex-id").textContent = `${c.launch[i]} · ${c.state[i]}`;
  $("#ex-nmse").textContent = pct(c.nmse[i]);
  $("#ex-nmse-lp").textContent = pct(c.nmseLp[i], 2);
  $("#ex-coh").textContent = c.coherence[i].toFixed(3);
  $("#ex-rank").textContent = `${ex.rank + 1} of ${c.count} · error ${pct(c.nmse[i])}`;
  const range = $("#ex-range");
  range.max = c.count - 1; range.value = ex.rank;
  drawWeights(c, h, a);
  drawHistogram(c, i);
}

$$("#ex-case button").forEach((b) => b.addEventListener("click", () => {
  $$("#ex-case button").forEach((x) => x.classList.toggle("on", x === b));
  ex.key = b.dataset.c;
  ex.rank = Math.floor(cases[ex.key].count / 2);
  drawExplorer();
}));
$("#ex-range").addEventListener("input", (e) => { ex.rank = +e.target.value; drawExplorer(); });
$$("#explorer [data-jump]").forEach((b) => b.addEventListener("click", () => {
  const n = cases[ex.key].count;
  ex.rank = { best: 0, median: Math.floor((n - 1) / 2), worst: n - 1, random: Math.floor(Math.random() * n) }[b.dataset.jump];
  drawExplorer();
}));

// ---------- first draw, and redraw when the layout width changes ----------

ex.rank = Math.floor((cases[ex.key].count - 1) / 2);
const redraw = () => { drawMasks(); drawNoise(); drawExplorer(); };
let lastWidth = 0;
new ResizeObserver(() => {
  const w = $("#chart-masks").clientWidth;
  if (w && w !== lastWidth) { lastWidth = w; redraw(); }
}).observe($("#chart-masks"));
