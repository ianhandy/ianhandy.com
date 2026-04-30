/* =========================================================================
   STRINGS (i18n)
   All user-facing text is loaded from strings.json and applied to every node
   carrying data-i18n / data-i18n-html / data-i18n-title. Math content in
   .formula blocks is left alone (those are rendered by KaTeX as static
   LaTeX source, not translatable copy).
   ========================================================================= */
let STRINGS = {};

async function loadStrings() {
  try {
    const res = await fetch('strings.json');
    if (!res.ok) throw new Error(res.statusText);
    STRINGS = await res.json();
  } catch (err) {
    console.warn('[celestial-sim] strings.json failed to load:', err);
    STRINGS = {};
  }
}

function t(dotted, fallback = '') {
  const v = dotted.split('.').reduce((o, k) => (o == null ? o : o[k]), STRINGS);
  return (typeof v === 'string') ? v : fallback;
}

function applyStrings() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const text = t(key, el.textContent);
    if (el.tagName === 'TITLE') document.title = text;
    else el.textContent = text;
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const key = el.getAttribute('data-i18n-html');
    el.innerHTML = t(key, el.innerHTML);
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const key = el.getAttribute('data-i18n-title');
    el.title = t(key, el.title);
  });
}

/* =========================================================================
   N-BODY PHYSICS
   Softened 1/r^n force law + velocity-Verlet integration.
   Flat Float64Arrays for cache-friendly inner loops.
     pos[2i], pos[2i+1]  = x, y of body i
     F_ij ∝ m_i m_j (r_j - r_i) / (|r|² + ε²)^((n+1)/2)
   ========================================================================= */
class NBody {
  constructor(positions, velocities, masses) {
    this.N = masses.length;
    this.pos    = new Float64Array(this.N * 2);
    this.vel    = new Float64Array(this.N * 2);
    this.acc    = new Float64Array(this.N * 2);
    this.accOld = new Float64Array(this.N * 2);
    this.mass   = new Float64Array(this.N);
    for (let i = 0; i < this.N; i++) {
      this.pos[2*i]     = positions[i][0];
      this.pos[2*i + 1] = positions[i][1];
      this.vel[2*i]     = velocities[i][0];
      this.vel[2*i + 1] = velocities[i][1];
      this.mass[i]      = masses[i];
    }
    this.G = 1.0; this.eps = 0.05; this.n = 2.0; this.dt = 0.01;
    this.t = 0;
    this.computeAccel();
    this.E0 = this.energy();
  }

  computeAccel() {
    const N = this.N, pos = this.pos, m = this.mass, acc = this.acc;
    const eps2 = this.eps * this.eps;
    const power = (this.n + 1) * 0.5;
    for (let k = 0; k < 2*N; k++) acc[k] = 0;
    for (let i = 0; i < N; i++) {
      const xi = pos[2*i], yi = pos[2*i + 1];
      let ax = 0, ay = 0;
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const dx = pos[2*j]     - xi;
        const dy = pos[2*j + 1] - yi;
        const r2 = dx*dx + dy*dy + eps2;
        const f = this.G * m[j] * Math.pow(r2, -power);
        ax += f * dx;
        ay += f * dy;
      }
      acc[2*i]     = ax;
      acc[2*i + 1] = ay;
    }
  }

  step() {
    const N = this.N, dt = this.dt;
    const pos = this.pos, vel = this.vel, acc = this.acc, accOld = this.accOld;
    for (let k = 0; k < 2*N; k++) accOld[k] = acc[k];
    for (let k = 0; k < 2*N; k++) pos[k] += vel[k] * dt + 0.5 * acc[k] * dt * dt;
    this.computeAccel();
    for (let k = 0; k < 2*N; k++) vel[k] += 0.5 * (accOld[k] + acc[k]) * dt;
    this.t += dt;
  }

  stepN(n) { for (let i = 0; i < n; i++) this.step(); }

  energy() {
    const N = this.N, pos = this.pos, vel = this.vel, m = this.mass;
    let KE = 0;
    for (let i = 0; i < N; i++) {
      const vx = vel[2*i], vy = vel[2*i + 1];
      KE += 0.5 * m[i] * (vx*vx + vy*vy);
    }
    let PE = 0;
    const eps2 = this.eps * this.eps;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = pos[2*j]     - pos[2*i];
        const dy = pos[2*j + 1] - pos[2*i + 1];
        const r  = Math.sqrt(dx*dx + dy*dy + eps2);
        PE -= this.G * m[i] * m[j] / r;
      }
    }
    return KE + PE;
  }

  setParams(G, eps, n, dt) {
    this.G = G; this.eps = eps; this.n = n; this.dt = dt;
    this.computeAccel();
    this.E0 = this.energy();
  }

  setMasses(masses) {
    for (let i = 0; i < this.N; i++) this.mass[i] = masses[i];
    this.computeAccel();
    this.E0 = this.energy();
  }
}

/* =========================================================================
   PRESETS
   ========================================================================= */
/* All presets use G = 1 sim units. Velocities are tuned for circular orbits
   accounting for the default softening ε = 0.05; small drift over many orbits
   is expected and shows up in the diagnostics drift meter. */
const PRESETS = {
  binary: {
    bodies: [
      { pos: [-1, 0], vel: [0, -0.5], mass: 1.0 },
      { pos: [ 1, 0], vel: [0,  0.5], mass: 1.0 },
    ],
    colors: ['#f6c667', '#ff7ab6'],
  },
  figure8: {
    bodies: [
      { pos: [-0.97000436,  0.24308753], vel: [ 0.466203685,  0.43236573 ], mass: 1.0 },
      { pos: [ 0.97000436, -0.24308753], vel: [ 0.466203685,  0.43236573 ], mass: 1.0 },
      { pos: [ 0,           0         ], vel: [-0.93240737,  -0.86473146 ], mass: 1.0 },
    ],
    colors: ['#89c9ff', '#f6c667', '#ff7ab6'],
  },
  solar: {
    bodies: [
      { pos: [ 0,   0  ], vel: [ 0,     0    ], mass: 20.0 },
      { pos: [ 1.5, 0  ], vel: [ 0,     3.651], mass:  0.3 },
      { pos: [-2.5, 0  ], vel: [ 0,    -2.828], mass:  0.6 },
      { pos: [ 0,   4  ], vel: [-2.236, 0    ], mass:  0.2 },
    ],
    colors: ['#f6c667', '#6be1c7', '#ff7ab6', '#c89bf5'],
  },
  lagrange: {
    bodies: [
      { pos: [ 1.000,  0.000], vel: [ 0.000,  0.760], mass: 1.0 },
      { pos: [-0.500,  0.866], vel: [-0.658, -0.380], mass: 1.0 },
      { pos: [-0.500, -0.866], vel: [ 0.658, -0.380], mass: 1.0 },
    ],
    colors: ['#f6c667', '#89c9ff', '#ff7ab6'],
  },
  pinwheel: {
    bodies: [
      { pos: [ 0,  1], vel: [-0.978,  0    ], mass: 1.0 },
      { pos: [ 1,  0], vel: [ 0,      0.978], mass: 1.0 },
      { pos: [ 0, -1], vel: [ 0.978,  0    ], mass: 1.0 },
      { pos: [-1,  0], vel: [ 0,     -0.978], mass: 1.0 },
    ],
    colors: ['#f6c667', '#89c9ff', '#ff7ab6', '#80e0a3'],
  },
};

/* =========================================================================
   STATE
   ========================================================================= */
let sim;
let currentPreset = 'binary';
let paused = false;
let trail = [];
let lastFrame = 0;
let fps = 60;
let currentMasses = [];
const params = { G: 1.0, eps: 0.05, n: 2.0, dt: 0.01, substeps: 4, trail: 220, zoom: 1 };
const view = { panX: 0, panY: 0 };

async function init() {
  await loadStrings();
  applyStrings();
  loadPreset('binary');
  renderAllMath();
  wireUpLatexInteractions();
  updateLegend();
  requestAnimationFrame(animate);
}

function renderAllMath() {
  if (typeof renderMathInElement !== 'function') {
    // KaTeX auto-render loads async; try again once it has arrived.
    setTimeout(renderAllMath, 50);
    return;
  }
  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }],
    trust: (ctx) => ctx.command === '\\htmlClass',
    strict: false,
    throwOnError: false,
  });
  // Wire LaTeX interactions again after render replaces the DOM nodes.
  wireUpLatexInteractions();
}

function wireUpLatexInteractions() {
  const classToParam = { 'tw-G': 'G', 'tw-eps': 'eps', 'tw-n': 'n', 'tw-dt': 'dt' };
  for (const cls in classToParam) {
    const param = classToParam[cls];
    document.querySelectorAll('.katex .' + cls).forEach(el => {
      if (el.dataset.wired === '1') return;
      el.dataset.wired = '1';
      el.addEventListener('mouseenter', () => highlightRow(param, true));
      el.addEventListener('mouseleave', () => highlightRow(param, false));
      el.addEventListener('click', () => focusRow(param));
    });
  }
}

function highlightRow(param, on) {
  document.querySelectorAll(`[data-param="${param}"]`).forEach(r => {
    r.classList.toggle('highlight', on);
  });
}

function focusRow(param) {
  const row = document.querySelector(`[data-param="${param}"]`);
  if (!row) return;
  row.scrollIntoView({ behavior: 'smooth', block: 'center' });
  row.classList.remove('pulse'); void row.offsetWidth; row.classList.add('pulse');
  const inp = row.querySelector('input');
  if (inp) inp.focus();
}

function loadPreset(name) {
  currentPreset = name;
  const p = PRESETS[name];
  const positions  = p.bodies.map(b => b.pos);
  const velocities = p.bodies.map(b => b.vel);
  const masses     = p.bodies.map(b => b.mass);
  currentMasses = masses.slice();
  sim = new NBody(positions, velocities, masses);
  sim.setParams(params.G, params.eps, params.n, params.dt);
  trail = p.bodies.map(() => []);
  view.panX = 0;
  view.panY = 0;
  rebuildMassSliders();
}

function rebuildMassSliders() {
  const container = document.getElementById('mass-sliders');
  container.innerHTML = '';
  const preset = PRESETS[currentPreset];
  const hintTemplate = t('sections.masses.hintBody', 'body {n}');
  preset.bodies.forEach((b, i) => {
    const hint = hintTemplate.replace('{n}', String(i + 1));
    const row = document.createElement('div');
    row.className = 'slider-row';
    row.dataset.param = 'mass' + i;
    row.style.setProperty('--slider-color', preset.colors[i]);
    row.innerHTML = `
      <label style="color: ${preset.colors[i]}">m<span class="hint">${hint}</span></label>
      <input type="range" min="0.05" max="25" step="0.05" value="${b.mass}">
      <span class="value">${b.mass.toFixed(2)}</span>`;
    const input = row.querySelector('input');
    const valEl = row.querySelector('.value');
    input.addEventListener('input', e => {
      const v = parseFloat(e.target.value);
      valEl.textContent = v.toFixed(2);
      currentMasses[i] = v;
      sim.setMasses(currentMasses);
    });
    container.appendChild(row);
  });
}

function updateLegend() {
  const el = document.getElementById('legend-force');
  if (!el) return;
  el.innerHTML = `
    <span><span style="color:var(--col-G)">G</span> = <span class="val">${params.G.toFixed(2)}</span></span>
    <span><span style="color:var(--col-eps)">ε</span> = <span class="val">${params.eps.toFixed(2)}</span></span>
    <span><span style="color:var(--col-n)">n</span> = <span class="val">${params.n.toFixed(2)}</span></span>
    <span><span style="color:var(--col-dt)">Δt</span> = <span class="val">${params.dt.toFixed(3)}</span></span>`;
}

document.querySelectorAll('.slider-row input[type=range]').forEach(input => {
  const row = input.closest('.slider-row');
  const param = row.dataset.param;
  if (!param || param.startsWith('mass')) return;
  input.addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    const dp = (param === 'dt') ? 3
             : (param === 'substeps' || param === 'trail') ? 0 : 2;
    row.querySelector('.value').textContent = v.toFixed(dp);
    if      (param === 'substeps') params.substeps = v;
    else if (param === 'trail')    params.trail = v;
    else if (param === 'zoom')     params.zoom = v;
    else {
      params[param] = v;
      if (sim) sim.setParams(params.G, params.eps, params.n, params.dt);
    }
    updateLegend();
  });
});

document.getElementById('preset').addEventListener('change', e => loadPreset(e.target.value));
document.getElementById('reset').addEventListener('click', () => loadPreset(currentPreset));
document.getElementById('playPause').addEventListener('click', () => {
  paused = !paused;
  document.getElementById('playPause').textContent =
    paused ? t('controls.play', 'Play') : t('controls.pause', 'Pause');
});
document.getElementById('minimize').addEventListener('click', () => {
  const panel = document.getElementById('panel');
  const btn = document.getElementById('minimize');
  panel.classList.toggle('collapsed');
  const collapsed = panel.classList.contains('collapsed');
  btn.textContent = collapsed ? '+' : '−';
  btn.title = collapsed ? t('controls.expandPanel', 'Expand panel')
                        : t('controls.collapsePanel', 'Collapse panel');
});

document.getElementById('settings-toggle').addEventListener('click', () => {
  document.getElementById('stats').classList.toggle('visible');
});

/* =========================================================================
   CANVAS RENDERING
   ========================================================================= */
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = window.innerWidth  * dpr;
  canvas.height = window.innerHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

/* Pan: click-and-drag (or single-finger drag) on the canvas shifts the view.
   Pan is stored in screen pixels so zooming doesn't change how far you've
   moved. Pan resets whenever a preset loads. */
let panActive = false;
let panLastX = 0, panLastY = 0;

canvas.addEventListener('pointerdown', e => {
  panActive = true;
  panLastX = e.clientX;
  panLastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add('grabbing');
});
canvas.addEventListener('pointermove', e => {
  if (!panActive) return;
  view.panX += e.clientX - panLastX;
  view.panY += e.clientY - panLastY;
  panLastX = e.clientX;
  panLastY = e.clientY;
});
canvas.addEventListener('pointerup', e => {
  panActive = false;
  canvas.releasePointerCapture(e.pointerId);
  canvas.classList.remove('grabbing');
});
canvas.addEventListener('pointercancel', () => {
  panActive = false;
  canvas.classList.remove('grabbing');
});

/* Gravity-warped grid: each world-aligned grid vertex is pulled toward every
   body by an amount proportional to mass / distance^4. The grid is anchored
   to world coordinates so it slides with the camera when the user pans. */
function drawGravityGrid() {
  if (!sim) return;
  const W = window.innerWidth, H = window.innerHeight;
  const scale = Math.min(W, H) / 8 * params.zoom;

  // World bounds visible on screen, accounting for pan.
  const wxMin = (-W / 2 - view.panX) / scale;
  const wxMax = ( W / 2 - view.panX) / scale;
  const wyMin = (view.panY - H / 2) / scale;
  const wyMax = (view.panY + H / 2) / scale;

  // Aim for ~38px screen spacing, snapped to integer multiples in world space.
  const spacingWorld = 38 / scale;
  const x0 = Math.floor(wxMin / spacingWorld) - 1;
  const x1 = Math.ceil (wxMax / spacingWorld) + 1;
  const y0 = Math.floor(wyMin / spacingWorld) - 1;
  const y1 = Math.ceil (wyMax / spacingWorld) + 1;
  const cols = x1 - x0 + 1;
  const rows = y1 - y0 + 1;
  if (cols * rows > 6000) return;

  const N = sim.N;
  const eps2 = params.eps * params.eps;
  const xs = new Float32Array(rows * cols);
  const ys = new Float32Array(rows * cols);

  for (let ri = 0; ri < rows; ri++) {
    for (let ci = 0; ci < cols; ci++) {
      const wx = (x0 + ci) * spacingWorld;
      const wy = (y0 + ri) * spacingWorld;

      let dx = 0, dy = 0;
      for (let i = 0; i < N; i++) {
        const rx = sim.pos[2 * i]     - wx;
        const ry = sim.pos[2 * i + 1] - wy;
        const r2 = rx * rx + ry * ry + eps2 + 0.05;
        const m = currentMasses[i] || 1;
        const pull = m * 0.18 / (r2 * r2);
        dx += rx * pull;
        dy += ry * pull;
      }
      const mag2 = dx * dx + dy * dy;
      const maxPull = 0.7;
      if (mag2 > maxPull * maxPull) {
        const f = maxPull / Math.sqrt(mag2);
        dx *= f; dy *= f;
      }

      const idx = ri * cols + ci;
      xs[idx] = W / 2 + (wx + dx) * scale + view.panX;
      ys[idx] = H / 2 - (wy + dy) * scale + view.panY;
    }
  }

  ctx.strokeStyle = 'rgba(120, 160, 220, 0.22)';
  ctx.lineWidth = 0.9;

  for (let ri = 0; ri < rows; ri++) {
    ctx.beginPath();
    const base = ri * cols;
    ctx.moveTo(xs[base], ys[base]);
    for (let ci = 1; ci < cols; ci++) {
      ctx.lineTo(xs[base + ci], ys[base + ci]);
    }
    ctx.stroke();
  }
  for (let ci = 0; ci < cols; ci++) {
    ctx.beginPath();
    ctx.moveTo(xs[ci], ys[ci]);
    for (let ri = 1; ri < rows; ri++) {
      const idx = ri * cols + ci;
      ctx.lineTo(xs[idx], ys[idx]);
    }
    ctx.stroke();
  }
}

function worldToScreen(x, y) {
  const W = window.innerWidth, H = window.innerHeight;
  const scale = Math.min(W, H) / 8 * params.zoom;
  return [W / 2 + x * scale + view.panX, H / 2 - y * scale + view.panY];
}

function draw() {
  const W = window.innerWidth, H = window.innerHeight;
  ctx.clearRect(0, 0, W, H);

  drawGravityGrid();

  const preset = PRESETS[currentPreset];
  const N = sim.N;

  for (let i = 0; i < N; i++) {
    trail[i].push([sim.pos[2*i], sim.pos[2*i + 1]]);
    while (trail[i].length > params.trail) trail[i].shift();
  }
  for (let i = 0; i < N; i++) {
    const color = preset.colors[i];
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.1;
    const tr = trail[i];
    for (let j = 1; j < tr.length; j++) {
      ctx.globalAlpha = (j / tr.length) * 0.55;
      ctx.beginPath();
      const [x1, y1] = worldToScreen(tr[j - 1][0], tr[j - 1][1]);
      const [x2, y2] = worldToScreen(tr[j    ][0], tr[j    ][1]);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  for (let i = 0; i < N; i++) {
    const [x, y] = worldToScreen(sim.pos[2*i], sim.pos[2*i + 1]);
    const m = currentMasses[i] || preset.bodies[i].mass;
    const r = Math.max(3, Math.pow(m, 1 / 3) * 5.5);
    const color = preset.colors[i];

    const glow = ctx.createRadialGradient(x, y, 0, x, y, r * 5);
    glow.addColorStop(0, color + 'cc');
    glow.addColorStop(0.35, color + '55');
    glow.addColorStop(1, color + '00');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, r * 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.arc(x, y, r * 0.35, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

function animate(now) {
  if (!paused && sim) sim.stepN(params.substeps);
  if (sim) {
    draw();
    document.getElementById('stat-time').textContent = sim.t.toFixed(2);
    const E = sim.energy();
    document.getElementById('stat-energy').textContent = E.toFixed(4);
    if (Math.abs(sim.E0) > 1e-12) {
      const drift = (E - sim.E0) / Math.abs(sim.E0);
      const sign = drift >= 0 ? '+' : '';
      document.getElementById('stat-drift').textContent = sign + (drift * 100).toFixed(3) + '%';
    }
  }
  if (lastFrame && now - lastFrame > 0) {
    fps = 0.9 * fps + 0.1 * (1000 / (now - lastFrame));
    document.getElementById('stat-fps').textContent = fps.toFixed(0);
  }
  lastFrame = now;
  requestAnimationFrame(animate);
}

init();
