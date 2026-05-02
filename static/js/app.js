'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let selectedOp = null;           // currently selected operation slug
let activeBtn  = null;           // DOM button currently active
let selectedNeedsSecond = false; // whether the current op needs a second image

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
const applyBtn       = document.getElementById('applyBtn');
const secondFileWrap = document.getElementById('secondFileWrap');
const origHistArea   = document.getElementById('origHistArea');
const origHistCanvas = document.getElementById('origHistCanvas');
const procHistArea   = document.getElementById('procHistArea');
const procHistCanvas = document.getElementById('procHistCanvas');

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
  zoom:              [
                       { name: 'scale',  label: 'Scale factor', type: 'number', default: 2.0, step: 0.1, min: 0.1 },
                       { name: 'method', label: 'Yöntem',       type: 'select', default: 'nearest', options: [
                         { value: 'nearest',  text: 'Nearest (En Yakın Komşu)' },
                         { value: 'bilinear', text: 'Bilinear (Çift Doğrusal)' },
                         { value: 'bicubic',  text: 'Bicubic (Bikübik)' },
                       ]},
                     ],
  'contrast-multiply': [{ name: 'alpha',       label: 'Alpha (>1 = brighter)', type: 'number', default: 1.5,  step: 0.1,  min: 0.1 }],
  'mean-filter':     [{ name: 'size',          label: 'Kernel size (odd)',     type: 'number', default: 3,    min: 3,     step: 2 }],
  threshold:         [{ name: 'threshold_val', label: 'Threshold (0-255)',     type: 'number', default: 128,  min: 0,     max: 255 }],
  'noise-add':       [{ name: 'amount',        label: 'Noise % (0.0-1.0)',     type: 'number', default: 0.05, step: 0.01, min: 0, max: 1 }],
  'histogram-expand': [
                       { name: 'a', label: 'Alt Sınır (0.0-1.0)',  type: 'number', default: 0.3,  step: 0.05, min: 0, max: 1 },
                       { name: 'b', label: 'Üst Sınır (0.0-1.0)',  type: 'number', default: 0.7,  step: 0.05, min: 0, max: 1 },
                     ],
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

previewImg.addEventListener('load', () => {
  if (selectedOp === 'crop') {
    renderParams('crop');
  }
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

// ── Apply button click ─────────────────────────────────────────────────────
applyBtn.addEventListener('click', () => {
  if (selectedOp) runOperation(selectedOp, selectedNeedsSecond);
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
  selectedNeedsSecond = needsSecond;
  secondFileWrap.classList.toggle('hidden', !needsSecond);

  // Build param inputs
  renderParams(op);

  // Clear status
  hideStatus();

  // Auto-run only for single-image operations
  if (needsSecond) {
    applyBtn.classList.remove('hidden');
  } else {
    runOperation(op, needsSecond);
  }
});

// ── Render parameter inputs ────────────────────────────────────────────────
function renderParams(op) {
  paramsBox.innerHTML = '';
  const defs = PARAM_DEFS[op];
  
  if (!defs) {
    applyBtn.classList.add('hidden');
    return;
  }
  defs.forEach(p => {
    const row = document.createElement('div');
    row.className = 'param-row';
    const label = document.createElement('label');
    label.setAttribute('for', `param-${p.name}`);
    label.textContent = p.label;

    let control;
    if (p.type === 'select') {
      control = document.createElement('select');
      control.id   = `param-${p.name}`;
      control.name = p.name;
      (p.options || []).forEach(opt => {
        const o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.text;
        if (opt.value === p.default) o.selected = true;
        control.appendChild(o);
      });
    } else {
      control = document.createElement('input');
      control.type  = p.type || 'number';
      control.id    = `param-${p.name}`;
      control.name  = p.name;
      control.value = p.default;
      if (p.min  !== undefined) control.min  = p.min;
      if (p.max  !== undefined) control.max  = p.max;
      if (p.step !== undefined) control.step = p.step;

      // Dynamic max for crop based on current image
      if (op === 'crop' && previewImg && previewImg.naturalWidth) {
        if (p.name === 'x1' || p.name === 'x2') control.max = previewImg.naturalWidth;
        if (p.name === 'y1' || p.name === 'y2') control.max = previewImg.naturalHeight;
        if (p.name === 'x2' && p.default === 200 && previewImg.naturalWidth < 200) control.value = previewImg.naturalWidth;
        if (p.name === 'y2' && p.default === 200 && previewImg.naturalHeight < 200) control.value = previewImg.naturalHeight;
      }
    }

    row.appendChild(label);
    row.appendChild(control);
    paramsBox.appendChild(row);
  });

  // Show/hide main Apply button
  applyBtn.classList.remove('hidden');
}

// ── Run the operation ──────────────────────────────────────────────────────
async function runOperation(op, needsSecond = false) {
  hideStatus();
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

  // Client-side crop validation before sending request
  if (op === 'crop') {
    const x1 = parseInt(formData.get('x1'), 10);
    const x2 = parseInt(formData.get('x2'), 10);
    const y1 = parseInt(formData.get('y1'), 10);
    const y2 = parseInt(formData.get('y2'), 10);

    if (x1 >= x2) {
      showStatus('X1 değeri X2 değerinden küçük olmalıdır.', 'error');
      return;
    }
    if (y1 >= y2) {
      showStatus('Y1 değeri Y2 değerinden küçük olmalıdır.', 'error');
      return;
    }
  }

  // UI: loading state
  setLoading(true);
  origHistArea.classList.add('hidden');
  procHistArea.classList.add('hidden');

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
    }

    // Histogram Germe, Genişletme veya Eşitleme sonrası çift histogram görüntüle
    if (data.original_histogram && data.processed_histogram) {
      drawInlineHistogram(origHistCanvas, data.original_histogram, '#818cf8');
      origHistArea.classList.remove('hidden');
      drawInlineHistogram(procHistCanvas, data.processed_histogram, '#22c55e');
      procHistArea.classList.remove('hidden');
    }

    if (data.warning) {
      showStatus(data.warning, 'warning');
    } else {
      showStatus('İşlem Başarılı ✓', 'success');
    }

  } catch (err) {
    showStatus('Error: ' + err.message, 'error');
  } finally {
    setLoading(false);
  }
}

// ── Draw inline histogram (under preview boxes) ─────────────────────────────
function drawInlineHistogram(canvas, hist, accentColor) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const maxVal = Math.max(...hist);
  ctx.clearRect(0, 0, W, H);

  // Koyu arka plan
  ctx.fillStyle = '#111420';
  ctx.fillRect(0, 0, W, H);

  // Yatay referans çizgileri
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let y = 0; y < H; y += Math.round(H / 4)) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  if (maxVal === 0) return;

  const barW = W / 256;
  const padding = 6;

  for (let i = 0; i < 256; i++) {
    const barH = (hist[i] / maxVal) * (H - padding * 2);
    // Parlaklık ve aksana göre renk gradyanı
    const t = i / 255;
    ctx.fillStyle = accentColor;
    ctx.globalAlpha = 0.35 + t * 0.65;
    ctx.fillRect(i * barW, H - padding - barH, Math.max(barW - 0.5, 1), barH);
  }
  ctx.globalAlpha = 1.0;
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function setLoading(on) {
  spinner.classList.toggle('hidden', !on);
  resultImg.classList.toggle('faded', on);
  if (activeBtn) activeBtn.disabled = on;
}

function showStatus(msg, type = 'error') {
  statusMsg.textContent = msg;
  statusMsg.className = 'status-msg' + (type === 'error' ? '' : ' ' + type);
  statusMsg.classList.remove('hidden');
  
  // Sadece başarı mesajları otomatik kaybolur. Hata ve uyarılar kalıcıdır.
  if (type === 'success') {
    setTimeout(hideStatus, 4000);
  }
}

function hideStatus() {
  statusMsg.classList.add('hidden');
}
