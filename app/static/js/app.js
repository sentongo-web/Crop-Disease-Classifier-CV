/* ── Crop Disease Classifier — Frontend ─────────────────────────────── */

const $ = (sel) => document.querySelector(sel);

const dropZone       = $('#dropZone');
const fileInput      = $('#fileInput');
const previewWrap    = $('#previewWrap');
const previewImg     = $('#previewImg');
const analyzeBtn     = $('#analyzeBtn');
const clearBtn       = $('#clearBtn');
const uploadSection  = $('#uploadSection');
const resultSection  = $('#resultSection');
const loadingOverlay = $('#loadingOverlay');
const statusBadge    = $('#statusBadge');
const toast          = $('#toast');

let currentFile = null;

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (data.model_loaded) {
      statusBadge.textContent = 'Model Ready';
      statusBadge.className   = 'status-badge status-ready';
    } else {
      statusBadge.textContent = 'Demo Mode';
      statusBadge.className   = 'status-badge status-demo';
    }
  } catch {
    statusBadge.textContent = 'API Offline';
    statusBadge.className   = 'status-badge status-error';
  }
}

// ── File handling ─────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('Please select a valid image file (JPG, PNG, WEBP).', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File exceeds 10 MB limit.', 'error');
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);
  dropZone.hidden    = true;
  previewWrap.hidden = false;
  resultSection.hidden = true;
}

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & drop
dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', () => fileInput.click());

clearBtn.addEventListener('click', () => {
  currentFile    = null;
  fileInput.value = '';
  previewImg.src  = '';
  previewWrap.hidden  = true;
  dropZone.hidden     = false;
  resultSection.hidden = true;
});

// Paste support
document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      handleFile(item.getAsFile());
      break;
    }
  }
});

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', analyzeImage);

async function analyzeImage() {
  if (!currentFile) return;

  loadingOverlay.hidden = false;

  try {
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('top_k', '5');

    const res = await fetch('/predict', { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    showToast(`Analysis failed: ${err.message}`, 'error');
  } finally {
    loadingOverlay.hidden = true;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const top      = data.top_prediction;
  const preds    = data.all_predictions || [];
  const info     = data.disease_info    || {};
  const isDemo   = data.status === 'demo';
  const isHealthy = top.is_healthy;

  // Top result box
  const stateClass = isDemo ? 'demo' : isHealthy ? 'healthy' : 'diseased';
  const icon = isHealthy ? '✅' : isDemo ? '⚠️' : '⚠️';
  const confClass = top.confidence >= 80 ? 'conf-high' : top.confidence >= 50 ? 'conf-medium' : 'conf-low';

  $('#topResult').className = `top-result ${stateClass}`;
  $('#topResult').innerHTML = `
    <div class="top-result-icon">${icon}</div>
    <div class="top-result-body">
      <div class="top-result-plant">${escHtml(top.plant)}</div>
      <div class="top-result-cond">${escHtml(top.condition)}</div>
      <div class="top-result-conf">${isDemo ? '⚠️ Demo mode — train the model for real predictions' : isHealthy ? 'No disease detected' : 'Disease detected'}</div>
    </div>
    <span class="confidence-pill ${confClass}">${top.confidence.toFixed(1)}%</span>
  `;

  // Confidence bars
  const listEl = $('#predictionsList');
  listEl.innerHTML = '';
  preds.forEach((p, i) => {
    const rankClass = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : 'rank-other';
    const item = document.createElement('div');
    item.className = 'pred-item';
    item.innerHTML = `
      <div class="pred-header">
        <span class="pred-name">${escHtml(p.plant)} — ${escHtml(p.condition)}</span>
        <span class="pred-pct">${p.confidence.toFixed(1)}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${rankClass}" style="width:0%" data-width="${p.confidence}%"></div>
      </div>
    `;
    listEl.appendChild(item);
  });

  // Animate bars after paint
  requestAnimationFrame(() => requestAnimationFrame(() => {
    listEl.querySelectorAll('.bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  }));

  // Disease info
  const diseaseInfoEl = $('#diseaseInfo');
  if (info && info.common_name) {
    const sevClass = `sev-${(info.severity || 'unknown').toLowerCase()}`;
    $('#diseaseContent').innerHTML = `
      <div class="info-grid">
        <div class="info-item">
          <span class="info-label">Common Name</span>
          <span class="info-value">${escHtml(info.common_name)}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Severity</span>
          <span class="info-value"><span class="severity-badge ${sevClass}">${escHtml(info.severity || 'unknown')}</span></span>
        </div>
        <div class="info-item full">
          <span class="info-label">Description</span>
          <span class="info-value">${escHtml(info.description || '—')}</span>
        </div>
        <div class="info-item full">
          <span class="info-label">Treatment / Management</span>
          <span class="info-value">${escHtml(info.treatment || '—')}</span>
        </div>
      </div>
    `;
    diseaseInfoEl.hidden = false;
  } else {
    diseaseInfoEl.hidden = true;
  }
}

// ── Toast notification ────────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = 'info') {
  toast.textContent = msg;
  toast.style.background = type === 'error' ? '#dc2626' : '#1f2937';
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
checkHealth();
