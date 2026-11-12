// ROMEO interactive lab: forward model, direct PMP inversion and rank-one projection.
// Detector-plane mode fields and PMP pseudo-inverses are precomputed by tools/export_demo_data.py in the paper's
// simulated optical system with its optimised (Figure 1) mask; everything per frame runs here.

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

renderMath();

const MEAS = {
  near: { label: "near field", color: "#3987e5", element: "4f image of the fibre facet, no mask" },
  flat: { label: "no mask", color: "#d95926", element: "flat SLM: lens and propagation only" },
  random: { label: "random mask", color: "#9085e9", element: "random phase mask, 16 µm cells" },
  optimised: { label: "optimised mask", color: "#199e70", element: "the paper's optimised mask" },
};

let lab;
try {
  lab = await fetch("data/lab/index.json").then((r) => r.json());
} catch (err) {
  const note = location.protocol === "file:"
    ? "The simulation data cannot load when this page is opened as a file. Open it through a web server instead (for example python3 -m http.server)."
    : "Could not load the simulation data.";
  $("#phasors").innerHTML = `<p class="loading">${note}</p>`;
  Object.assign($("#lab-loading"), { hidden: false, textContent: note });
  throw err;
}

// float16 -> float32 lookup table for the stored arrays
const HALF = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  HALF[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
}

function decode(buffer, e) {
  if (e.type === "uint8") return new Uint8Array(buffer, e.offset, e.length);
  if (e.type === "float32") return new Float32Array(buffer, e.offset, e.length);
  const raw = new Uint16Array(buffer, e.offset, e.length), out = new Float32Array(e.length);
  for (let i = 0; i < e.length; i++) out[i] = HALF[raw[i]] * e.scale;
  return out;
}

const fibres = lab.fibres.map((f) => ({ ...f, M: f.modes, N: f.modes * f.modes, ready: null }));

// Load one fibre's fields and inverses on first use (the 42-mode file is large).
function loadFibre(i) {
  const fib = fibres[i];
  fib.ready ??= Promise.all([
    fetch(`data/lab/${fib.file}.json`).then((r) => r.json()),
    fetch(`data/lab/${fib.file}.bin`).then((r) => r.arrayBuffer()),
  ]).then(([meta, buffer]) => {
    const M = fib.M;
    fib.names = meta.names;
    fib.near = decode(buffer, meta.nearField);
    fib.nearGrid = lab.near.pixels;
    fib.meas = {};
    for (const [key, m] of Object.entries(meta.measurements)) {
      const raw = decode(buffer, m.fields), P = m.grid * m.grid;
      const re = new Float32Array(M * P), im = new Float32Array(M * P);
      for (let q = 0; q < M * P; q++) { re[q] = raw[2 * q]; im[q] = raw[2 * q + 1]; }
      fib.meas[key] = { re, im, grid: m.grid, P, inverse: decode(buffer, m.inverse), singular: decode(buffer, m.singular),
        kappa: m.kappa ?? Infinity, capture: m.capture };
    }
    fib.masks = Object.fromEntries(Object.entries(meta.masks).map(([k, e]) => [k, decode(buffer, e)]));
    return fib;
  });
  return fib.ready;
}

const state = {
  fibre: Math.min(1, fibres.length - 1),
  meas: "optimised",
  snr: 20,
  animate: false,
  C: null,           // {re, im} unnormalised truth coefficients
  rec: null,         // reconstructed coefficients
  omega: null,       // per-mode angular rates for phase rotation
  drift: null,       // per-mode velocities {re, im} for random drift
  evolve: "rotate",
  speed: 1,
};

// ---------- numerics ----------

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss() {
  let u = 0; while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// Symmetric eigendecomposition (Householder tridiagonalisation + implicit QL, after JAMA).
// A is n*n row-major; returns ascending eigenvalues d and eigenvectors as columns of V.
function eigSym(A, n) {
  const V = Float64Array.from(A), d = new Float64Array(n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) d[j] = V[(n - 1) * n + j];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0, h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) { d[j] = V[(i - 1) * n + j]; V[i * n + j] = 0; V[j * n + i] = 0; }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
      let f = d[i - 1], g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j]; V[j * n + i] = f; g = e[j] + V[j * n + j] * f;
        for (let k = j + 1; k <= i - 1; k++) { g += V[k * n + j] * d[k]; e[k] += V[k * n + j] * f; }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j]; g = e[j];
        for (let k = j; k <= i - 1; k++) V[k * n + j] -= f * e[k] + g * d[k];
        d[j] = V[(i - 1) * n + j]; V[i * n + j] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[(n - 1) * n + i] = V[i * n + i]; V[i * n + i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k * n + i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[k * n + i + 1] * V[k * n + j];
        for (let k = 0; k <= i; k++) V[k * n + j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k * n + i + 1] = 0;
  }
  for (let j = 0; j < n; j++) { d[j] = V[(n - 1) * n + j]; V[(n - 1) * n + j] = 0; }
  V[(n - 1) * n + n - 1] = 1;
  e[0] = 0;

  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0, tst1 = 0;
  const eps = 2 ** -52;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
    if (m > l) {
      let iter = 0;
      do {
        if (++iter > 60) break;
        let g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1, c2 = c, c3 = c, s = 0, s2 = 0;
        const el1 = e[l + 1];
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
          e[i + 1] = s * r; s = e[i] / r; c = p / r;
          p = c * d[i] - s * g; d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            const row = k * n;
            h = V[row + i + 1];
            V[row + i + 1] = s * V[row + i] + c * h;
            V[row + i] = c * V[row + i] - s * h;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1;
        e[l] = s * p; d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] += f; e[l] = 0;
  }
  return { d, V };
}

function normalised(C) {
  let s = 0;
  for (let k = 0; k < C.re.length; k++) s += C.re[k] ** 2 + C.im[k] ** 2;
  s = Math.sqrt(s) || 1;
  return { re: C.re.map((v) => v / s), im: C.im.map((v) => v / s), norm: s };
}

function intensity(fib, key, C, out) {
  const { re, im, P } = fib.meas[key];
  const M = fib.M;
  out ??= new Float64Array(P);
  out.fill(0);
  const er = new Float64Array(P), ei = new Float64Array(P);
  for (let k = 0; k < M; k++) {
    const cr = C.re[k], ci = C.im[k], o = k * P;
    for (let p = 0; p < P; p++) {
      const a = re[o + p], b = im[o + p];
      er[p] += cr * a - ci * b; ei[p] += cr * b + ci * a;
    }
  }
  for (let p = 0; p < P; p++) out[p] = er[p] * er[p] + ei[p] * ei[p];
  return out;
}

// Products S = Υ⁺ I. The PMP matrix is never formed: Υᵀ I is read off the Hermitian matrix Σ_p I_p f_p f_pᴴ
// (column order |F_k|², then 2Re, −2Im of F_k F_j* for k < j), and Υ⁺ = G⁺Υᵀ uses the stored packed G⁺.
function reconstruct(fib, key, I) {
  const { re, im, P, inverse } = fib.meas[key];
  const M = fib.M, N = fib.N;
  const Hr = new Float64Array(M * M), Hi = new Float64Array(M * M), fr = new Float64Array(M), fi = new Float64Array(M);
  for (let p = 0; p < P; p++) {
    const v = I[p];
    for (let k = 0; k < M; k++) { fr[k] = re[k * P + p] * v; fi[k] = im[k * P + p] * v; }
    for (let k = 0; k < M; k++) {
      const a = re[k * P + p], b = im[k * P + p], row = k * M;
      for (let j = k; j < M; j++) { // (I f f^H)_kj = I F_k conj(F_j)
        Hr[row + j] += fr[j] * a + fi[j] * b;
        Hi[row + j] += fr[j] * b - fi[j] * a;
      }
    }
  }
  const b = new Float64Array(N);
  let col = 0;
  for (let k = 0; k < M; k++) b[col++] = Hr[k * M + k];
  for (let k = 0; k < M; k++) for (let j = k + 1; j < M; j++) { b[col++] = 2 * Hr[k * M + j]; b[col++] = -2 * Hi[k * M + j]; }
  // S = G⁺ b with G⁺ stored as its packed upper triangle (row-major)
  const S = new Float64Array(N);
  let q = 0;
  for (let i = 0; i < N; i++) {
    const bi = b[i];
    let s = inverse[q++] * bi;
    for (let j = i + 1; j < N; j++, q++) { const r = inverse[q]; s += r * b[j]; S[j] += r * bi; }
    S[i] += s;
  }
  // Hermitian H from S, embedded as the real symmetric [[Re, -Im], [Im, Re]].
  const n = 2 * M, H = new Float64Array(n * n);
  let c = 0;
  for (let k = 0; k < M; k++) { const v = S[c++]; H[k * n + k] = v; H[(k + M) * n + k + M] = v; }
  for (let k = 0; k < M; k++) for (let j = k + 1; j < M; j++) {
    const hr = S[c++], hi = S[c++]; // H_kj = C_k C_j*
    H[k * n + j] = hr; H[j * n + k] = hr; H[(k + M) * n + j + M] = hr; H[(j + M) * n + k + M] = hr;
    H[(k + M) * n + j] = hi; H[k * n + j + M] = -hi; H[(j + M) * n + k] = -hi; H[j * n + k + M] = hi;
  }
  const { d, V } = eigSym(H, n);
  const lam = Math.max(d[n - 1], 0), scale = Math.sqrt(lam);
  const cr = new Float64Array(M), ci = new Float64Array(M);
  for (let k = 0; k < M; k++) { cr[k] = V[k * n + n - 1] * scale; ci[k] = V[(k + M) * n + n - 1] * scale; }
  return { re: cr, im: ci };
}

function alignPhase(rec, truth) {
  let zr = 0, zi = 0;
  for (let k = 0; k < truth.re.length; k++) {
    zr += rec.re[k] * truth.re[k] + rec.im[k] * truth.im[k];
    zi += rec.re[k] * truth.im[k] - rec.im[k] * truth.re[k];
  }
  const z = Math.hypot(zr, zi) || 1, cr = zr / z, ci = zi / z;
  return { re: rec.re.map((a, k) => a * cr - rec.im[k] * ci), im: rec.re.map((a, k) => a * ci + rec.im[k] * cr) };
}

// ---------- rendering ----------

function hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p]; case 1: return [q, v, p]; case 2: return [p, v, t];
    case 3: return [p, q, v]; case 4: return [t, p, v]; default: return [v, p, q];
  }
}

// Fit a canvas's pixel grid to the image it shows (the CSS size stays fixed).
function sized(canvas, n) {
  if (canvas.width !== n) { canvas.width = n; canvas.height = n; }
  return canvas.getContext("2d");
}

function drawComplex(canvas, fib, C, ampMax) {
  const n = fib.nearGrid, P = n * n;
  const ctx = sized(canvas, n), img = ctx.createImageData(n, n);
  const M = fib.M, er = new Float64Array(P), ei = new Float64Array(P);
  for (let k = 0; k < M; k++) {
    const o = k * P;
    for (let p = 0; p < P; p++) { const a = fib.near[o + p]; er[p] += C.re[k] * a; ei[p] += C.im[k] * a; }
  }
  let max = ampMax || 0;
  if (!max) for (let p = 0; p < P; p++) max = Math.max(max, Math.hypot(er[p], ei[p]));
  for (let p = 0; p < P; p++) {
    const level = Math.min(1, (Math.hypot(er[p], ei[p]) / (max || 1)) ** 2); // brightness = intensity
    const [r, g, b] = hsv2rgb((Math.atan2(ei[p], er[p]) / (2 * Math.PI) + 1) % 1, 0.9, level);
    img.data.set([r * 255, g * 255, b * 255, 255], 4 * p);
  }
  ctx.putImageData(img, 0, 0);
  return max;
}

const MAGMA = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function cmap(t) {
  t = Math.max(0, Math.min(1, t)) * (MAGMA.length - 1);
  const i = Math.min(MAGMA.length - 2, Math.floor(t)), f = t - i, a = MAGMA[i], b = MAGMA[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function drawIntensity(canvas, I, n) {
  const P = n * n, ctx = sized(canvas, n), img = ctx.createImageData(n, n);
  let max = 0;
  for (let p = 0; p < P; p++) max = Math.max(max, I[p]);
  for (let p = 0; p < P; p++) img.data.set([...cmap(Math.sqrt(Math.max(0, I[p]) / (max || 1))), 255], 4 * p);
  ctx.putImageData(img, 0, 0);
}

function drawMask(fib, key) {
  const cv = $("#cv-mask"), n = lab.mask.pixels, ctx = sized(cv, n);
  const img = ctx.createImageData(n, n);
  const mask = fib.masks[key]; // phase as uint8, 0..255 -> 0..2π, over the central 5.12 mm of the SLM
  for (let p = 0; p < n * n; p++) {
    let rgb;
    // cyclic hue for phase, with zero phase dark (the flat exterior of the optimised mask)
    const phaseColour = (t) => hsv2rgb(t, 0.6, 0.18 + 0.72 * Math.sin(Math.PI * t) ** 2);
    if (mask) rgb = phaseColour(mask[p] / 256);
    else if (key === "flat") rgb = phaseColour(0);
    else rgb = [0.25, 0.32, 0.4];
    img.data.set([rgb[0] * 255, rgb[1] * 255, rgb[2] * 255, 255], 4 * p);
  }
  ctx.putImageData(img, 0, 0);
  let cap = document.querySelector(".mask-caption");
  if (!cap) { cap = document.createElement("p"); cap.className = "mask-caption"; cv.after(cap); }
  cap.textContent = MEAS[key].element;
}

const fmtKappa = (k) => (!isFinite(k) ? "∞ (degenerate)" : k >= 1e4 ? k.toExponential(1).replace("e+", "×10^") : k.toFixed(k < 10 ? 2 : 1));

// ---------- phasor pads ----------

let pads = [];
function buildPads() {
  const fib = fibres[state.fibre], host = $("#phasors");
  host.innerHTML = "";
  host.classList.toggle("many", fib.M > 20);
  pads = fib.names.map((name, k) => {
    const wrap = document.createElement("div");
    wrap.className = "pad";
    const cv = document.createElement("canvas");
    cv.width = cv.height = 160;
    wrap.append(cv, Object.assign(document.createElement("div"), { textContent: name }));
    host.append(wrap);
    const setFrom = (ev) => {
      const r = cv.getBoundingClientRect(), R = padRadius();
      let x = ((ev.clientX - r.left) / r.width) * 2 - 1, y = -(((ev.clientY - r.top) / r.height) * 2 - 1);
      const m = Math.hypot(x, y) / 0.86;
      if (m > 1) { x /= m; y /= m; }
      state.C.re[k] = (x / 0.86) * R; state.C.im[k] = (y / 0.86) * R;
      schedule();
    };
    cv.addEventListener("pointerdown", (ev) => { cv.setPointerCapture(ev.pointerId); setFrom(ev); });
    cv.addEventListener("pointermove", (ev) => { if (cv.hasPointerCapture(ev.pointerId)) setFrom(ev); });
    cv.addEventListener("dblclick", () => { state.C.re[k] = 0; state.C.im[k] = 0; schedule(); });
    return cv;
  });
}

const padRadius = () => Math.min(1, 2.2 / Math.sqrt(fibres[state.fibre].M));

function drawPads() {
  const R = padRadius();
  pads.forEach((cv, k) => {
    const ctx = cv.getContext("2d"), w = cv.width, c = w / 2, r = w * 0.43;
    ctx.clearRect(0, 0, w, w);
    ctx.strokeStyle = "#2b4560"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(c, c, r, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(c - r, c); ctx.lineTo(c + r, c); ctx.moveTo(c, c - r); ctx.lineTo(c, c + r); ctx.stroke();
    const tx = c + (state.C.re[k] / R) * r, ty = c - (state.C.im[k] / R) * r;
    const hue = (Math.atan2(state.C.im[k], state.C.re[k]) / (2 * Math.PI) + 1) % 1;
    const [cr, cg, cb] = hsv2rgb(hue, 0.85, 1);
    ctx.strokeStyle = `rgb(${cr * 255},${cg * 255},${cb * 255})`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(c, c); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath(); ctx.arc(tx, ty, 10, 0, 2 * Math.PI); ctx.fill();
    if (state.rec) {
      const rx = c + (state.rec.re[k] / R) * r, ry = c - (state.rec.im[k] / R) * r;
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(rx, ry, 15, 0, 2 * Math.PI); ctx.stroke();
    }
  });
}

// ---------- spectrum ----------

function drawSpectrum() {
  const fib = fibres[state.fibre], cv = $("#cv-spectrum"), dpr = window.devicePixelRatio || 1;
  if (!fib.meas) return; // still loading
  const W = cv.clientWidth, H = 260;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const L = 54, Rm = 14, T = 10, B = 34, floor = -16;
  const X = (t) => L + t * (W - L - Rm), Y = (v) => T + (-Math.max(v, floor) / -floor) * (H - T - B);
  ctx.font = "11px Inter, sans-serif"; ctx.fillStyle = "#8fa3b8"; ctx.strokeStyle = "#1f3449"; ctx.lineWidth = 1;
  for (let e = 0; e >= floor; e -= 4) {
    ctx.beginPath(); ctx.moveTo(L, Y(e)); ctx.lineTo(W - Rm, Y(e)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(e === 0 ? "1" : `1e${e}`, L - 6, Y(e) + 4);
  }
  ctx.textAlign = "center";
  ctx.fillText(`singular-value index (1 … ${fib.N})`, L + (W - L - Rm) / 2, H - 8);
  ctx.save(); ctx.translate(12, T + (H - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("σᵢ / σ₁", 0, 0); ctx.restore();
  const legend = [];
  for (const [key, m] of Object.entries(fib.meas)) {
    const s = m.singular, on = key === state.meas;
    ctx.strokeStyle = MEAS[key].color; ctx.lineWidth = on ? 3.2 : 1.6; ctx.globalAlpha = on ? 1 : 0.55;
    ctx.beginPath();
    for (let i = 0; i < s.length; i++) {
      const x = X(s.length > 1 ? i / (s.length - 1) : 0), y = Y(Math.log10(Math.max(s[i], 1e-17)));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke(); ctx.globalAlpha = 1;
    legend.push(`<span style="--c:${MEAS[key].color}">${MEAS[key].label}: κ = ${fmtKappa(m.kappa ?? Infinity)}, light kept ${(100 * m.capture).toFixed(0)}%</span>`);
  }
  $("#spectrum-legend").innerHTML = legend.join("");
}

// ---------- main update ----------

let truthMax = 0, pending = false, lastI = null;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(update); } }

function update() {
  pending = false;
  const fib = fibres[state.fibre], key = state.meas;
  if (!fib.meas) return; // still loading
  const { P, grid } = fib.meas[key];
  const Cn = normalised(state.C);
  const I = intensity(fib, key, Cn);
  if (state.snr <= 60) {
    // SNR = 20 log10(mean intensity of a fully captured beam / noise sigma): Σ_p I_p is the captured power.
    const sigma = 1 / P / Math.pow(10, state.snr / 20);
    for (let p = 0; p < P; p++) I[p] += sigma * gauss();
  }
  const rec = alignPhase(reconstruct(fib, key, I), Cn);
  state.rec = { re: rec.re.map((v) => v * Cn.norm), im: rec.im.map((v) => v * Cn.norm) };
  lastI = I;

  truthMax = drawComplex($("#cv-true"), fib, Cn);
  drawComplex($("#cv-rec"), fib, rec, truthMax);
  drawIntensity($("#cv-cam"), I, grid);
  drawPads();

  let err = 0;
  for (let k = 0; k < fib.M; k++) err += (rec.re[k] - Cn.re[k]) ** 2 + (rec.im[k] - Cn.im[k]) ** 2;
  const twin = intensity(fib, key, { re: Cn.re, im: Cn.im.map((v) => -v) });
  const clean = intensity(fib, key, Cn);
  let dn = 0, nn = 0;
  for (let p = 0; p < P; p++) { dn += (twin[p] - clean[p]) ** 2; nn += clean[p] ** 2; }

  $("#st-kappa").textContent = fmtKappa(fib.meas[key].kappa ?? Infinity);
  $("#st-floor").textContent = Math.sqrt(fib.M + 1).toFixed(2);
  $("#st-capture").textContent = `${(100 * fib.meas[key].capture).toFixed(0)}%`;
  $("#st-nmse").textContent = err < 1e-6 ? "< 10⁻⁴ %" : err > 10 ? "> 1000%" : `${err < 0.01 ? (100 * err).toPrecision(2) : (100 * err).toFixed(1)}%`;
  $("#st-twin").textContent = `${(100 * Math.sqrt(dn / nn)).toFixed(dn / nn < 1e-8 ? 4 : 1)}%`;
  const tag = $("#tag-ok");
  tag.textContent = err < 0.01 ? "recovered" : err < 0.1 ? "approximate" : "failed";
  tag.className = "tag " + (err < 0.01 ? "good" : err < 0.1 ? "" : "bad");
  $("#tag-meas").textContent = MEAS[key].label;
  $("#hint-cam").textContent = key === "near" ? `Intensity only · ${grid}×${grid} image of the facet`
    : `Intensity only · ${grid}×${grid} camera, central ${lab.camera.spanMm.toFixed(1)} mm`;
  $("#note-k").textContent = `${fib.N.toLocaleString("en-GB")} unknown real products (M² with M = ${fib.M}) from ${P.toLocaleString("en-GB")} pixels.`;
  $("#out-snr").textContent = state.snr > 60 ? "noiseless" : `${state.snr} dB`;
}

function randomField() {
  const M = fibres[state.fibre].M, R = padRadius();
  const re = new Float64Array(M), im = new Float64Array(M);
  for (let k = 0; k < M; k++) {
    const a = Math.random() ** 1.5 * R * 0.95, ph = 2 * Math.PI * Math.random();
    re[k] = a * Math.cos(ph); im[k] = a * Math.sin(ph);
  }
  re[0] = Math.max(re[0], 0.3 * R);
  state.C = { re, im };
  state.omega = Float64Array.from({ length: M }, () => (Math.random() - 0.5) * 1.6);
  state.drift = { re: new Float64Array(M), im: new Float64Array(M) };
}

let selecting = 0;
async function selectFibre(i) {
  const ticket = ++selecting, fib = fibres[i];
  if (!fib.meas) {
    $("#lab-loading").hidden = false;
    $("#lab-loading").textContent = `Loading the ${fib.label} fibre (${(fib.bytes / 1e6).toFixed(0)} MB)…`;
  }
  await loadFibre(i);
  if (ticket !== selecting) return; // a later choice overtook this one
  $("#lab-loading").hidden = true;
  state.fibre = i;
  selFibre.value = i;
  randomField();
  buildPads();
  drawSpectrum();
  drawMask(fib, state.meas);
  if (state.evolve === "pictures") {
    if (fib.M === PICTURE_MODES) await startTour(); else setEvolve("rotate"); // the pictures exist for 42 modes only
  }
  schedule();
}

// ---------- wiring ----------

const selFibre = $("#sel-fibre");
fibres.forEach((f, i) => selFibre.append(new Option(
  `${f.label} (core radius ${f.coreRadiusUm} µm)${f.bytes > 1e7 ? ` · ${(f.bytes / 1e6).toFixed(0)} MB` : ""}`, i)));
selFibre.value = state.fibre;
selFibre.addEventListener("change", () => selectFibre(+selFibre.value));

$$("#seg-meas button").forEach((b) => b.addEventListener("click", () => {
  $$("#seg-meas button").forEach((x) => x.classList.toggle("on", x === b));
  state.meas = b.dataset.m;
  drawMask(fibres[state.fibre], state.meas);
  drawSpectrum();
  schedule();
}));
$("#rng-snr").addEventListener("input", (e) => { state.snr = +e.target.value; schedule(); });
$("#btn-random").addEventListener("click", () => { randomField(); schedule(); });
$("#btn-twin").addEventListener("click", () => { state.C.im = state.C.im.map((v) => -v); schedule(); });

// Phase rotation: every mode keeps its amplitude and turns at its own rate.
function rotate(dt) {
  const { re, im } = state.C;
  for (let k = 0; k < re.length; k++) {
    const w = state.omega[k] * dt, c = Math.cos(w), s = Math.sin(w), a = re[k];
    re[k] = a * c - im[k] * s; im[k] = a * s + im[k] * c;
  }
}

// Random drift: each coefficient moves with a smoothly varying random velocity (Ornstein-Uhlenbeck),
// so amplitude and phase both wander; it bounces off the edge of its pad circle and never leaves it.
function drift(dt) {
  if (!dt) return;
  const { re, im } = state.C, v = state.drift, R = padRadius();
  const tau = 0.8, sigma = 0.9 * R, edge = 0.97 * R;
  const decay = Math.exp(-dt / tau), kick = sigma * Math.sqrt((1 - decay * decay) / 2);
  for (let k = 0; k < re.length; k++) {
    v.re[k] = v.re[k] * decay + kick * gauss();
    v.im[k] = v.im[k] * decay + kick * gauss();
    re[k] += v.re[k] * dt; im[k] += v.im[k] * dt;
    const r = Math.hypot(re[k], im[k]);
    if (r > edge) {
      const nx = re[k] / r, ny = im[k] / r, out = v.re[k] * nx + v.im[k] * ny;
      re[k] = nx * edge; im[k] = ny * edge;
      if (out > 0) { v.re[k] -= 2 * out * nx; v.im[k] -= 2 * out * ny; }
    }
  }
}

// Picture tour: pairs of 42-mode fields that draw the same picture in the near field (the construction of the
// paper's Figure 1). It holds each field, then moves to the next along a sigmoid: field A, its twin B, the next
// picture's A, and so on, in a loop.
const PICTURE_MODES = 42, HOLD = 2.5, MOVE = 2.5; // seconds at 1x speed
let picturesData = null;
const tour = { keys: null, i: 0, t: 0, from: null, to: null };

const logistic = (x) => 1 / (1 + Math.exp(-x));
function sigmoid(u, k = 10) { // logistic step rescaled to run exactly from 0 to 1 over u in [0, 1]
  const lo = logistic(-k / 2), hi = logistic(k / 2);
  return (logistic(k * (u - 0.5)) - lo) / (hi - lo);
}

// The same field with its global phase chosen to be closest to `ref` (the global phase is unobservable).
function alignedTo(key, ref) {
  let zr = 0, zi = 0;
  for (let k = 0; k < key.re.length; k++) { zr += ref.re[k] * key.re[k] + ref.im[k] * key.im[k]; zi += ref.re[k] * key.im[k] - ref.im[k] * key.re[k]; }
  const z = Math.hypot(zr, zi) || 1, c = zr / z, s = -zi / z;
  return { ...key, re: key.re.map((a, k) => a * c - key.im[k] * s), im: key.re.map((a, k) => a * s + key.im[k] * c) };
}

async function startTour() {
  picturesData ??= fetch("data/lab/pictures.json").then((r) => r.json());
  const { pictures } = await picturesData;
  const keys = [];
  for (const p of pictures) for (const twin of ["A", "B"]) {
    const v = p[twin], M = v.length / 2;
    keys.push({ label: p.label, twin, count: pictures.length, index: keys.length / 2 | 0,
      re: Float64Array.from({ length: M }, (_, k) => v[2 * k]), im: Float64Array.from({ length: M }, (_, k) => v[2 * k + 1]) });
  }
  // one scale for all keyframes so that every coefficient stays inside its pad circle
  let peak = 0;
  for (const key of keys) for (let k = 0; k < key.re.length; k++) peak = Math.max(peak, Math.hypot(key.re[k], key.im[k]));
  const scale = (0.95 * padRadius()) / peak;
  for (const key of keys) { key.re = key.re.map((v) => v * scale); key.im = key.im.map((v) => v * scale); }
  Object.assign(tour, { keys, i: 0, t: 0, from: keys[0], to: alignedTo(keys[1 % keys.length], keys[0]) });
  state.C = { re: Float64Array.from(keys[0].re), im: Float64Array.from(keys[0].im) };
  showTour();
}

function tourStep(dt) {
  if (!tour.keys) return;
  tour.t += dt;
  while (tour.t >= HOLD + MOVE) { // arrived: the next keyframe becomes the one held
    tour.t -= HOLD + MOVE;
    tour.i = (tour.i + 1) % tour.keys.length;
    tour.from = tour.to;
    tour.to = alignedTo(tour.keys[(tour.i + 1) % tour.keys.length], tour.from);
  }
  const s = tour.t < HOLD ? 0 : sigmoid((tour.t - HOLD) / MOVE), { re, im } = state.C;
  for (let k = 0; k < re.length; k++) {
    re[k] = (1 - s) * tour.from.re[k] + s * tour.to.re[k];
    im[k] = (1 - s) * tour.from.im[k] + s * tour.to.im[k];
  }
  showTour();
}

function showTour() {
  const el = $("#tour-status");
  el.hidden = state.evolve !== "pictures" || !tour.keys;
  if (el.hidden) return;
  const from = tour.from, to = tour.to, field = (k) => `<b>${k.label}</b>, field ${k.twin}`;
  el.innerHTML = tour.t < HOLD
    ? `${field(from)} · picture ${from.index + 1} of ${from.count}. Fields A and B have different modes but the same near-field image; switch to Near field to compare.`
    : `${field(from)} → ${field(to)}`;
}

function setEvolve(mode) {
  state.evolve = mode;
  $$("#seg-evolve button").forEach((x) => x.classList.toggle("on", x.dataset.e === mode));
  showTour();
}

let lastT = 0;
function tick(t) {
  if (!state.animate) return;
  const dt = (lastT ? Math.min(0.05, (t - lastT) / 1000) : 0) * state.speed;
  lastT = t;
  if (state.evolve === "rotate") rotate(dt); else if (state.evolve === "drift") drift(dt); else tourStep(dt);
  update();
  requestAnimationFrame(tick);
}
$$("#seg-evolve button").forEach((b) => b.addEventListener("click", async () => {
  setEvolve(b.dataset.e);
  if (b.dataset.e !== "pictures") return;
  const i = fibres.findIndex((f) => f.M === PICTURE_MODES);
  if (state.fibre !== i) await selectFibre(i); else await startTour(); // selectFibre starts the tour itself
  if (!state.animate) $("#btn-animate").click();
  schedule();
}));
// speed slider: 0..100 maps logarithmically onto 0.1x..10x, with 1x in the middle
$("#rng-speed").addEventListener("input", (e) => {
  state.speed = 10 ** ((+e.target.value - 50) / 50);
  $("#out-speed").textContent = `${state.speed < 1 ? state.speed.toFixed(2) : state.speed.toFixed(1)}×`;
});
$("#btn-animate").addEventListener("click", (e) => {
  state.animate = !state.animate;
  e.currentTarget.setAttribute("aria-pressed", state.animate);
  e.currentTarget.textContent = state.animate ? "❚❚ Pause" : "▶ Evolve";
  lastT = 0;
  if (state.animate) requestAnimationFrame(tick);
});

$("#btn-bench").addEventListener("click", (e) => {
  const fib = fibres[state.fibre], key = state.meas, { P } = fib.meas[key];
  const runs = fib.M > 20 ? 100 : 1000, button = e.currentTarget;
  button.disabled = true;
  $("#out-bench").textContent = "Timing…";
  setTimeout(() => { // let the status paint before the loop blocks the page
    reconstruct(fib, key, lastI);
    const t0 = performance.now();
    for (let i = 0; i < runs; i++) reconstruct(fib, key, lastI);
    const us = ((performance.now() - t0) / runs) * 1000;
    const per = us < 1000 ? `${us.toFixed(0)} µs` : `${(us / 1000).toFixed(1)} ms`;
    $("#out-bench").textContent = `${per} per frame (${Math.round(1e6 / us).toLocaleString("en-GB")} frames/s) over ${runs} frames: ${P.toLocaleString("en-GB")} pixels, ${fib.N.toLocaleString("en-GB")} products, rank-one projection included, single-threaded JavaScript.`;
    button.disabled = false;
  }, 30);
});
$("#btn-bench").textContent = "Time reconstructions in this browser";

new ResizeObserver(() => drawSpectrum()).observe($("#cv-spectrum"));

selectFibre(state.fibre);
drawPipelineDecor();
const HERO_MODES = 17;
loadFibre(Math.max(0, fibres.findIndex((f) => f.M === HERO_MODES))).then(startHero);

// ---------- decoration ----------

function renderMath() {
  const go = () => $$("[data-tex]").forEach((el) => {
    window.katex.render(el.dataset.tex, el, { displayMode: el.classList.contains("eq"), throwOnError: false });
  });
  if (window.katex) go(); else window.addEventListener("load", () => window.katex && go());
}

function drawPipelineDecor() {
  const ns = "http://www.w3.org/2000/svg", rnd = mulberry32(5);
  const mask = $(".pl-mask");
  if (mask) {
    mask.innerHTML = "";
    for (let y = 0; y < 6; y++) for (let x = 0; x < 7; x++) {
      const r = document.createElementNS(ns, "rect");
      const [cr, cg, cb] = hsv2rgb(rnd(), 0.5, 0.85);
      Object.entries({ x: 248 + x * 10, y: 55 + y * 10, width: 10, height: 10, fill: `rgb(${cr * 255},${cg * 255},${cb * 255})` })
        .forEach(([k, v]) => r.setAttribute(k, v));
      mask.append(r);
    }
  }
  const camera = $(".pl-dots");
  if (camera) for (let y = 0; y < 6; y++) for (let x = 0; x < 7; x++) {
    const r = document.createElementNS(ns, "rect");
    const [cr, cg, cb] = cmap(rnd() ** 1.5);
    Object.entries({ x: 664 + x * 10, y: 55 + y * 10, width: 10, height: 10, fill: `rgb(${cr},${cg},${cb})` })
      .forEach(([k, v]) => r.setAttribute(k, v));
    camera.append(r);
  }
}

function startHero(fib) {
  // Background: a slowly evolving field. Foreground: a live reconstruction loop with the optimised mask.
  const cv = $("#hero-canvas"), big = fib;
  const small = document.createElement("canvas");
  const rnd = mulberry32(42);
  const phasors = (M, speed) => Array.from({ length: M }, () => ({ a: 0.3 + rnd(), p: 2 * Math.PI * rnd(), w: (rnd() - 0.5) * speed }));
  const at = (list, t) => ({
    re: Float64Array.from(list, (m) => m.a * Math.cos(m.p + m.w * t)),
    im: Float64Array.from(list, (m) => m.a * Math.sin(m.p + m.w * t)),
  });
  const bigModes = phasors(big.M, 0.9);

  const liveModes = phasors(fib.M, 1.6);
  const tiles = { truth: $("#hero-true"), cam: $("#hero-cam"), rec: $("#hero-rec") }, out = $("#hero-err");
  const { P, grid } = fib.meas.optimised;
  const I = new Float64Array(P), sigma = 1 / P / 10; // 20 dB pixel SNR
  let smoothed = null, lastLive = -1e9;

  function live(t) {
    const C = normalised(at(liveModes, t / 1000));
    intensity(fib, "optimised", C, I);
    for (let p = 0; p < P; p++) I[p] += sigma * gauss();
    const rec = alignPhase(reconstruct(fib, "optimised", I), C);
    const max = drawComplex(tiles.truth, fib, C);
    drawIntensity(tiles.cam, I, grid);
    drawComplex(tiles.rec, fib, rec, max);
    let err = 0;
    for (let k = 0; k < fib.M; k++) err += (rec.re[k] - C.re[k]) ** 2 + (rec.im[k] - C.im[k]) ** 2;
    smoothed = smoothed === null ? err : 0.9 * smoothed + 0.1 * err;
    out.textContent = `${(100 * smoothed).toPrecision(2)}%`;
  }

  function background(t) {
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    drawComplex(small, big, at(bigModes, t / 1000));
    const ctx = cv.getContext("2d"), s = Math.min(h * 1.2, w * 0.8);
    ctx.fillStyle = "#060c14"; ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, w - s * 0.95, (h - s) / 2, s, s);
  }

  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let visible = true, running = false;
  function frame(t) {
    if (t - lastLive > 45) { lastLive = t; live(t); background(t); }
    if (visible && !still) requestAnimationFrame(frame); else running = false;
  }
  const start = () => { if (!running) { running = true; requestAnimationFrame(frame); } };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible && !still) start(); }).observe($(".hero"));
  live(0); background(0);
  if (!still) start();
}
