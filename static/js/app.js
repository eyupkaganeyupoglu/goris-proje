'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let selectedOp = null;           // currently selected operation slug
let activeBtn  = null;           // DOM button currently active

// ── DOM refs ───────────────────────────────────────────────────────────────
const fileInput      = document.getElementById('fileInput');
const fileInput2     = document.getElementById('fileInput2');
const fileDrop       = document.getElementById('fileDrop');
const previewImg     = document.getElementById('previewImg');
const previewImg2    = document.getElementById('previewImg2');
const origEmpty      = document.getElementById('origEmpty');
const resultImg      = document.getElementById('resultImg');
const resultEmpty    = document.getElementById('resultEmpty');
const spinner        = document.getElementById('spinner');
const paramsBox      = document.getElementById('paramsBox');
const statusMsg      = document.getElementById('statusMsg');
const secondFileWrap = document.getElementById('secondFileWrap');
const histogramArea  = document.getElementById('histogramArea');
const histCanvas     = document.getElementById('histCanvas');

// ── Parameter definitions per operation ───────────────────────────────────
const PARAM_DEFS = {
  binary:            [{ name: 'threshold',     label: 'Threshold (0-255)', type: 'number', default: 128, min: 0, max: 255 }],
  rotation:          [{ name: 'angle',         label: 'Angle (°)',         type: 'number', default: 45 }],
  crop:              [
                       { name: 'x1', label: 'x1', type: 'number', default: 0, min: 0 },
                       { name: 'y1', label: 'y1', type: 'number', default: 0, min: 0 },
                       { name: 'x2', label: 'x2', type: 'number', default: 200, min: 1 },
                       { name: 'y2', label: 'y2', type: 'number', default: 200, min: 1 },
                     ],
  zoom:              [{ name: 'scale',         label: 'Scale factor',          type: 'number', default: 2.0,  step: 0.1,  min: 0.1 }],
  'contrast-multiply': [{ name: 'alpha',       label: 'Alpha (>1 = brighter)', type: 'number', default: 1.5,  step: 0.1,  min: 0.1 }],
  'mean-filter':     [{ name: 'size',          label: 'Kernel size (odd)',     type: 'number', default: 3,    min: 3,     step: 2 }],
  threshold:         [{ name: 'threshold_val', label: 'Threshold (0-255)',     type: 'number', default: 128,  min: 0,     max: 255 }],
  'noise-add':       [{ name: 'amount',        label: 'Noise % (0.0-1.0)',     type: 'number', default: 0.05, step: 0.01, min: 0, max: 1 }],
};

// ── Image preview on file select ───────────────────────────────────────────
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.remove('hidden');
  origEmpty.classList.add('hidden');
  fileDrop.querySelector('.drop-text').textContent = file.name;
});

fileInput2.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  previewImg2.src = URL.createObjectURL(file);
  previewImg2.classList.remove('hidden');
});

// ── Drag & drop ────────────────────────────────────────────────────────────
fileDrop.addEventListener('dragover', (e) => { e.preventDefault(); fileDrop.classList.add('active'); });
fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('active'));
fileDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  fileDrop.classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

// ── Operation button clicks ────────────────────────────────────────────────
document.getElementById('opsGrid').addEventListener('click', (e) => {
  const btn = e.target.closest('.op-btn');
  if (!btn) return;

  const op = btn.dataset.op;
  const needsSecond = btn.dataset.needsSecond === 'true';

  // Toggle active button style
  if (activeBtn) activeBtn.classList.remove('active');
  btn.classList.add('active');
  activeBtn = btn;
  selectedOp = op;

  // Show/hide second image upload
  secondFileWrap.classList.toggle('hidden', !needsSecond);

  // Build param inputs
  renderParams(op);

  // Clear status
  hideStatus();

  // Auto-run operation
  runOperation(op, needsSecond);
});

// ── Render parameter inputs ────────────────────────────────────────────────
function renderParams(op) {
  paramsBox.innerHTML = '';
  const defs = PARAM_DEFS[op];
  if (!defs) return;
  defs.forEach(p => {
    const row = document.createElement('div');
    row.className = 'param-row';
    const label = document.createElement('label');
    label.setAttribute('for', `param-${p.name}`);
    label.textContent = p.label;
    const input = document.createElement('input');
    input.type  = p.type || 'number';
    input.id    = `param-${p.name}`;
    input.name  = p.name;
    input.value = p.default;
    if (p.min  !== undefined) input.min  = p.min;
    if (p.max  !== undefined) input.max  = p.max;
    if (p.step !== undefined) input.step = p.step;
    row.appendChild(label);
    row.appendChild(input);
    paramsBox.appendChild(row);
  });
}

// ── Run the operation ──────────────────────────────────────────────────────
async function runOperation(op, needsSecond = false) {
  if (!fileInput.files.length) {
    showStatus('Please select an image first.', 'error');
    return;
  }
  if (needsSecond && !fileInput2.files.length) {
    showStatus('Please also select a second image.', 'error');
    return;
  }

  // Collect FormData
  const formData = new FormData();
  if (needsSecond) {
    formData.append('file1', fileInput.files[0]);
    formData.append('file2', fileInput2.files[0]);
  } else {
    formData.append('file', fileInput.files[0]);
  }

  // Append params
  const defs = PARAM_DEFS[op] || [];
  defs.forEach(p => {
    const el = document.getElementById(`param-${p.name}`);
    if (el) formData.append(p.name, el.value);
  });

  // UI: loading state
  setLoading(true);
  histogramArea.classList.add('hidden');

  try {
    const res = await fetch(`/process/${op}`, { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    if (data.result_url) {
      resultImg.src = data.result_url + '?t=' + Date.now();
      resultImg.classList.remove('hidden');
      resultEmpty.classList.add('hidden');
    } else if (data.result_b64) {
      resultImg.src = 'data:image/png;base64,' + data.result_b64;
      resultImg.classList.remove('hidden');
      resultEmpty.classList.add('hidden');
    } else if (data.histogram) {
      drawHistogram(data.histogram);
      histogramArea.classList.remove('hidden');
      resultEmpty.classList.add('hidden');
    }

    showStatus('Done ✓', 'success');

  } catch (err) {
    showStatus('Error: ' + err.message, 'error');
  } finally {
    setLoading(false);
  }
}

// ── Draw histogram on canvas ───────────────────────────────────────────────
function drawHistogram(hist) {
  const ctx = histCanvas.getContext('2d');
  const W = histCanvas.width;
  const H = histCanvas.height;
  const maxVal = Math.max(...hist);
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#1a1e2b';
  ctx.fillRect(0, 0, W, H);

  const barW = W / 256;
  for (let i = 0; i < 256; i++) {
    const barH = (hist[i] / maxVal) * (H - 10);
    const shade = Math.round((i / 255) * 200 + 55);
    ctx.fillStyle = `rgb(${shade},${Math.round(shade * 0.7)},255)`;
    ctx.fillRect(i * barW, H - barH, barW, barH);
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function setLoading(on) {
  spinner.classList.toggle('hidden', !on);
  resultImg.classList.toggle('faded', on);
  if (activeBtn) activeBtn.disabled = on;
}

function showStatus(msg, type = 'error') {
  statusMsg.textContent = msg;
  statusMsg.className = 'status-msg' + (type === 'success' ? ' success' : '');
  statusMsg.classList.remove('hidden');
  if (type === 'success') setTimeout(hideStatus, 3000);
}

function hideStatus() {
  statusMsg.classList.add('hidden');
}
