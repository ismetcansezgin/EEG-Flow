/* ════════════════════════════════════════════════
   EEGFlow — Frontend Application Script
   ════════════════════════════════════════════════ */
'use strict';

const API_BASE = 'http://127.0.0.1:8000';

// ── DOM ─────────────────────────────────────────
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const progressWrap   = document.getElementById('progressWrap');
const progressFill   = document.getElementById('progressFill');
const progressLabel  = document.getElementById('progressLabel');
const resultsSection = document.getElementById('resultsSection');
const uploadAgainBtn = document.getElementById('uploadAgainBtn');
const statusDot      = document.getElementById('statusDot');
const statusLabel    = document.getElementById('statusLabel');

const bannerFilename = document.getElementById('bannerFilename');
const numSamplesEl   = document.getElementById('numSamples');
const numChannelsEl  = document.getElementById('numChannels');
const samplingRateEl = document.getElementById('samplingRate');
const durationSecEl  = document.getElementById('durationSec');
const channelPillsEl = document.getElementById('channelPills');
const subjectListEl  = document.getElementById('subjectList');
const eventListEl    = document.getElementById('eventList');
const hasSubjectEl   = document.getElementById('hasSubject');
const hasEventsEl    = document.getElementById('hasEvents');

// ════════════════════════════════════════════════
// PAGE ROUTER  (Data Loader ↔ Guidelines)
// ════════════════════════════════════════════════
function showPage(pageId) {
    document.querySelectorAll('.page-view').forEach(p => p.classList.add('hidden'));
    document.getElementById(pageId).classList.remove('hidden');

    document.querySelectorAll('.nav-item[data-page]').forEach(a => {
        a.classList.remove('active');
        const badge = a.querySelector('.nav-badge');
        if (badge && badge.classList.contains('active-badge')) badge.style.display = '';
    });

    const activeLink = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

document.querySelectorAll('.nav-item[data-page]').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        showPage(link.dataset.page);
    });
});

// "How to format?" inline link inside the drop zone
const guidelinesLink = document.getElementById('goToGuidelinesLink');
if (guidelinesLink) {
    guidelinesLink.addEventListener('click', e => {
        e.preventDefault();
        showPage('page-guidelines');
    });
}

// "Back to Data Loader" button
const backBtn = document.getElementById('backToLoaderBtn');
if (backBtn) backBtn.addEventListener('click', () => showPage('page-loader'));

// ════════════════════════════════════════════════
// CONVERTER TABS
// ════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.remove('hidden');
    });
});

// ════════════════════════════════════════════════
// COPY BUTTONS
// ════════════════════════════════════════════════
document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.dataset.target || 'csvExample';
        const text = document.getElementById(targetId)?.innerText || '';
        navigator.clipboard.writeText(text).then(() => {
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
        });
    });
});

// ════════════════════════════════════════════════
// API HEALTH CHECK
// ════════════════════════════════════════════════
async function checkAPIHealth() {
    statusDot.className = 'status-dot checking';
    statusLabel.textContent = 'Checking API…';
    try {
        const resp = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(4000) });
        if (resp.ok) {
            statusDot.className = 'status-dot online';
            statusLabel.textContent = 'API Online';
        } else throw new Error();
    } catch {
        statusDot.className = 'status-dot offline';
        statusLabel.textContent = 'API Offline';
    }
}

// ════════════════════════════════════════════════
// DRAG & DROP + FILE INPUT
// ════════════════════════════════════════════════
['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length) handleUpload(files[0]);
});
dropZone.addEventListener('click', e => {
    if (e.target !== fileInput) fileInput.click();
});
dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleUpload(e.target.files[0]);
});
uploadAgainBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    fileInput.value = '';
});

// ════════════════════════════════════════════════
// UPLOAD HANDLER
// ════════════════════════════════════════════════
async function handleUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Only CSV files (.csv) are accepted. Please check the Dataset Guidelines page for formatting help.');
        return;
    }

    resultsSection.classList.add('hidden');
    progressWrap.classList.remove('hidden');
    setProgress(10, `Uploading ${file.name}…`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        setProgress(35, 'Sending to EEGFlow API…');
        const resp = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
        setProgress(70, 'Validating and cleaning data…');
        const json = await resp.json();
        if (!resp.ok) throw new Error(json.detail || `HTTP ${resp.status}`);
        if (!json.success) throw new Error(json.detail || 'Upload failed.');
        setProgress(100, 'Complete!');
        await sleep(350);
        progressWrap.classList.add('hidden');
        renderResults(file.name, json.metadata);
    } catch (err) {
        progressWrap.classList.add('hidden');
        const msg = err.message.includes('fetch')
            ? 'Cannot reach EEGFlow API. Make sure the backend is running:\n\nuvicorn backend.main:app --reload'
            : err.message;
        alert(`⚠️ ${msg}`);
    }
}

// ════════════════════════════════════════════════
// RENDER RESULTS
// ════════════════════════════════════════════════
function renderResults(filename, meta) {
    bannerFilename.textContent  = filename;
    numSamplesEl.textContent    = meta.num_samples.toLocaleString();
    numChannelsEl.textContent   = meta.num_channels;
    samplingRateEl.textContent  = `${meta.sampling_rate_hz} Hz`;
    durationSecEl.textContent   = `${meta.time_duration_sec} s`;

    channelPillsEl.innerHTML = '';
    (meta.channels || []).forEach(ch => {
        const el = document.createElement('span');
        el.className = 'pill';
        el.textContent = ch;
        channelPillsEl.appendChild(el);
    });

    subjectListEl.textContent = (meta.subject_ids?.length) ? meta.subject_ids.join(', ') : 'None';
    eventListEl.textContent   = (meta.event_types?.length) ? meta.event_types.join(', ') : 'None';
    hasSubjectEl.textContent  = meta.has_subject_id ? '✅ Detected' : '❌ Not found';
    hasEventsEl.textContent   = meta.has_events     ? '✅ Detected' : '❌ Not found';

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Helpers ────────────────────────────────────
function setProgress(pct, label) {
    progressFill.style.width = `${pct}%`;
    progressLabel.textContent = label;
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Init ───────────────────────────────────────
checkAPIHealth();
setInterval(checkAPIHealth, 30_000);
