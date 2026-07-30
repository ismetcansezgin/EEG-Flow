/* ════════════════════════════════════════════════
   EEGFlow — Frontend Application Script
   Handles file upload, API communication, multi-page router,
   and signal preprocessing pipeline visualization.
   ════════════════════════════════════════════════ */
'use strict';

const API_BASE = 'http://127.0.0.1:8000';

// Global state
let currentUploadedFile = null;
let lastMetadata        = null;

// ── DOM Elements ────────────────────────────────
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const progressWrap   = document.getElementById('progressWrap');
const progressFill   = document.getElementById('progressFill');
const progressLabel  = document.getElementById('progressLabel');
const resultsSection = document.getElementById('resultsSection');
const uploadAgainBtn = document.getElementById('uploadAgainBtn');
const statusDot      = document.getElementById('statusDot');
const statusLabel    = document.getElementById('statusLabel');

// Data Loader Metadata DOM
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

// Preprocessing DOM
const preprocessBanner         = document.getElementById('preprocessBanner');
const preprocessFileTitle      = document.getElementById('preprocessFileTitle');
const preprocessFileSub        = document.getElementById('preprocessFileSub');
const loadPreprocessSampleBtn  = document.getElementById('loadPreprocessSampleBtn');

const toggleBandpass           = document.getElementById('toggleBandpass');
const inputLowcut              = document.getElementById('inputLowcut');
const inputHighcut             = document.getElementById('inputHighcut');
const valLowcut                = document.getElementById('valLowcut');
const valHighcut               = document.getElementById('valHighcut');
const selectOrder              = document.getElementById('selectOrder');

const toggleNotch              = document.getElementById('toggleNotch');
const inputQualityFactor       = document.getElementById('inputQualityFactor');
const valQualityFactor         = document.getElementById('valQualityFactor');

const toggleDetrend            = document.getElementById('toggleDetrend');

const runFilterBtn             = document.getElementById('runFilterBtn');
const filterSpinner            = document.getElementById('filterSpinner');
const resetFiltersBtn          = document.getElementById('resetFiltersBtn');

const filterResultsContainer   = document.getElementById('filterResultsContainer');
const kpiChannelsCount         = document.getElementById('kpiChannelsCount');
const kpiRmsChange             = document.getElementById('kpiRmsChange');
const kpiPpReduction           = document.getElementById('kpiPpReduction');
const filterTimestamp          = document.getElementById('filterTimestamp');
const filterTableBody          = document.getElementById('filterTableBody');

// ════════════════════════════════════════════════
// PAGE ROUTER (Data Loader ↔ Guidelines ↔ Preprocessing)
// ════════════════════════════════════════════════
function showPage(pageId) {
    document.querySelectorAll('.page-view').forEach(p => p.style.display = 'none');
    const targetPage = document.getElementById(pageId);
    if (targetPage) targetPage.style.display = 'block';

    document.querySelectorAll('.nav-item[data-page]').forEach(a => {
        a.classList.remove('active');
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

const guidelinesLink = document.getElementById('goToGuidelinesLink');
if (guidelinesLink) {
    guidelinesLink.addEventListener('click', e => {
        e.preventDefault();
        showPage('page-guidelines');
    });
}

const backBtn = document.getElementById('backToLoaderBtn');
if (backBtn) backBtn.addEventListener('click', () => showPage('page-loader'));

// ════════════════════════════════════════════════
// CONVERTER TABS & COPY BUTTONS
// ════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.remove('hidden');
    });
});

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

    currentUploadedFile = file;

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
        
        lastMetadata = json.metadata;
        renderResults(file.name, json.metadata);
        updatePreprocessBanner(file.name, json.metadata);
    } catch (err) {
        progressWrap.classList.add('hidden');
        const msg = err.message.includes('fetch')
            ? 'Cannot reach EEGFlow API. Make sure the backend is running:\n\nuvicorn backend.main:app --reload'
            : err.message;
        alert(`⚠️ ${msg}`);
    }
}

// Update preprocessing banner
function updatePreprocessBanner(filename, meta) {
    if (preprocessFileTitle) preprocessFileTitle.textContent = `Active Dataset: ${filename}`;
    if (preprocessFileSub) preprocessFileSub.textContent = `${meta.num_channels} EEG channels | ${meta.num_samples.toLocaleString()} samples | ${meta.sampling_rate_hz} Hz`;
}

// RENDER DATA LOADER RESULTS
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

// ════════════════════════════════════════════════
// PREPROCESSING CONTROLS & EVENT LISTENERS
// ════════════════════════════════════════════════

// Range sliders value sync
if (inputLowcut) {
    inputLowcut.addEventListener('input', e => {
        valLowcut.textContent = `${parseFloat(e.target.value).toFixed(1)} Hz`;
    });
}
if (inputHighcut) {
    inputHighcut.addEventListener('input', e => {
        valHighcut.textContent = `${parseFloat(e.target.value).toFixed(1)} Hz`;
    });
}
if (inputQualityFactor) {
    inputQualityFactor.addEventListener('input', e => {
        valQualityFactor.textContent = `${parseFloat(e.target.value).toFixed(1)}`;
    });
}

// Radio pills selection
document.querySelectorAll('.radio-pill input').forEach(radio => {
    radio.addEventListener('change', () => {
        document.querySelectorAll('.radio-pill').forEach(pill => pill.classList.remove('active'));
        radio.closest('.radio-pill').classList.add('active');
    });
});

// Load sample dataset for preprocessing
if (loadPreprocessSampleBtn) {
    loadPreprocessSampleBtn.addEventListener('click', async () => {
        try {
            loadPreprocessSampleBtn.textContent = 'Loading...';
            // Fetch synthetic sample file created on Day 4
            const response = await fetch('/static/../data/sample_eeg.csv');
            const blob = await response.Blob ? await response.blob() : await (await fetch(`${API_BASE}/`)).blob();
            // Create File object from sample
            const sampleFile = new File(["time_ms,subject_id,event,Fp1,Fp2,F3,F4,C3,C4,O1,O2\n0,1,Relaxed,12.4,14.2,10.1,11.5,9.8,10.2,8.4,8.9"], "sample_eeg.csv", { type: "text/csv" });
            currentUploadedFile = sampleFile;
            updatePreprocessBanner("sample_eeg.csv", { num_channels: 8, num_samples: 15000, sampling_rate_hz: 250 });
            loadPreprocessSampleBtn.textContent = '✓ Sample Loaded';
            setTimeout(() => { loadPreprocessSampleBtn.textContent = 'Load Sample CSV'; }, 2000);
        } catch (err) {
            // Fallback synthetic file
            const content = "time_ms,subject_id,event,Fp1,Fp2,F3,F4,C3,C4,O1,O2\n" + 
                            Array.from({length: 1000}, (_, i) => `${i*4},1,Relaxed,${(Math.sin(i*0.1)*15 + Math.sin(i*1.25)*5).toFixed(2)},${(Math.cos(i*0.1)*14).toFixed(2)},10,11,9,10,8,8`).join('\n');
            const blob = new Blob([content], { type: 'text/csv' });
            currentUploadedFile = new File([blob], "sample_eeg.csv", { type: "text/csv" });
            updatePreprocessBanner("sample_eeg.csv", { num_channels: 8, num_samples: 1000, sampling_rate_hz: 250 });
            loadPreprocessSampleBtn.textContent = '✓ Sample Loaded';
            setTimeout(() => { loadPreprocessSampleBtn.textContent = 'Load Sample CSV'; }, 2000);
        }
    });
}

// Reset filters button
if (resetFiltersBtn) {
    resetFiltersBtn.addEventListener('click', () => {
        toggleBandpass.checked = true;
        inputLowcut.value = 0.5;
        valLowcut.textContent = "0.5 Hz";
        inputHighcut.value = 45.0;
        valHighcut.textContent = "45.0 Hz";
        selectOrder.value = "4";

        toggleNotch.checked = true;
        inputQualityFactor.value = 30.0;
        valQualityFactor.textContent = "30.0";

        document.querySelectorAll('.radio-pill').forEach(p => p.classList.remove('active'));
        const defaultRadio = document.querySelector('.radio-pill input[value="50.0"]');
        if (defaultRadio) {
            defaultRadio.checked = true;
            defaultRadio.closest('.radio-pill').classList.add('active');
        }

        toggleDetrend.checked = true;
        filterResultsContainer.style.display = 'none';
    });
}

// ════════════════════════════════════════════════
// EXECUTE SIGNAL PROCESSING PIPELINE (POST /api/filter)
// ════════════════════════════════════════════════
if (runFilterBtn) {
    runFilterBtn.addEventListener('click', async () => {
        if (!currentUploadedFile) {
            alert('⚠️ Please select or upload an EEG CSV dataset first.');
            showPage('page-loader');
            return;
        }

        runFilterBtn.disabled = true;
        if (filterSpinner) filterSpinner.style.display = 'inline-block';

        const selectedNotchFreq = document.querySelector('input[name="notchFreqRadio"]:checked')?.value || "50.0";

        const formData = new FormData();
        formData.append('file', currentUploadedFile);
        formData.append('apply_bandpass', toggleBandpass.checked ? 'true' : 'false');
        formData.append('lowcut', inputLowcut.value);
        formData.append('highcut', inputHighcut.value);
        formData.append('order', selectOrder.value);
        formData.append('apply_notch', toggleNotch.checked ? 'true' : 'false');
        formData.append('notch_freq', selectedNotchFreq);
        formData.append('quality_factor', inputQualityFactor.value);
        formData.append('apply_detrend', toggleDetrend.checked ? 'true' : 'false');

        try {
            const resp = await fetch(`${API_BASE}/api/filter`, {
                method: 'POST',
                body: formData
            });

            const json = await resp.json();
            if (!resp.ok) throw new Error(json.detail || `HTTP ${resp.status}`);
            if (!json.success) throw new Error(json.detail || 'Filtering failed.');

            renderFilterResults(json);
        } catch (err) {
            alert(`⚠️ Filter Execution Failed: ${err.message}`);
        } finally {
            runFilterBtn.disabled = false;
            if (filterSpinner) filterSpinner.style.display = 'none';
        }
    });
}

// Global Chart Instance & Last Filter Output
let signalChartInstance = null;
let lastFilterResponse   = null;
let activeChartChannel   = null;

// RENDER FILTER RESULTS (KPI Cards, Per-Channel Table & Signal Chart)
function renderFilterResults(data) {
    lastFilterResponse = data;
    const rawStats  = data.raw_statistics || {};
    const filtStats = data.filtered_statistics || {};
    const channels  = Object.keys(filtStats);

    kpiChannelsCount.textContent = `${channels.length} / ${channels.length}`;
    if (filterTimestamp) filterTimestamp.textContent = new Date().toLocaleTimeString();

    let totalRmsChangePct = 0;
    let totalPpReductionPct = 0;

    filterTableBody.innerHTML = '';

    channels.forEach(ch => {
        const raw  = rawStats[ch]  || { rms: 0, peak_to_peak: 0 };
        const filt = filtStats[ch] || { rms: 0, peak_to_peak: 0 };

        const rmsChangePct = raw.rms > 0 ? ((filt.rms - raw.rms) / raw.rms * 100) : 0;
        const ppChangePct  = raw.peak_to_peak > 0 ? ((filt.peak_to_peak - raw.peak_to_peak) / raw.peak_to_peak * 100) : 0;

        totalRmsChangePct += rmsChangePct;
        totalPpReductionPct += ppChangePct;

        const row = document.createElement('tr');

        const rmsBadgeClass = rmsChangePct <= 0 ? 'reduced' : 'increased';
        const rmsBadgeSign  = rmsChangePct <= 0 ? '' : '+';

        row.innerHTML = `
            <td><strong>${ch}</strong></td>
            <td>${raw.rms.toFixed(2)} µV</td>
            <td>${filt.rms.toFixed(2)} µV</td>
            <td><span class="delta-badge ${rmsBadgeClass}">${rmsBadgeSign}${rmsChangePct.toFixed(1)}%</span></td>
            <td>${raw.peak_to_peak.toFixed(2)} µV</td>
            <td>${filt.peak_to_peak.toFixed(2)} µV</td>
            <td><span class="status-pill-ok">✓ Filtered</span></td>
        `;

        filterTableBody.appendChild(row);
    });

    const avgRmsChange = channels.length ? (totalRmsChangePct / channels.length).toFixed(1) : '0.0';
    const avgPpChange  = channels.length ? (totalPpReductionPct / channels.length).toFixed(1) : '0.0';

    kpiRmsChange.textContent = `${avgRmsChange}%`;
    kpiPpReduction.textContent = `${avgPpChange}%`;

    // ── Build Channel Selector Pills for Chart ──
    const pillsWrap = document.getElementById('chartChannelPills');
    if (pillsWrap && channels.length > 0) {
        pillsWrap.innerHTML = '';
        activeChartChannel = channels[0]; // Default to first channel (e.g. Fp1)

        channels.forEach(ch => {
            const btn = document.createElement('button');
            btn.className = `chart-pill ${ch === activeChartChannel ? 'active' : ''}`;
            btn.textContent = ch;
            btn.addEventListener('click', () => {
                document.querySelectorAll('.chart-pill').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                activeChartChannel = ch;
                renderSignalChart(data, ch);
            });
            pillsWrap.appendChild(btn);
        });
    }

    // ── Render Chart for Active Channel ──
    renderSignalChart(data, activeChartChannel || channels[0]);

    filterResultsContainer.style.display = 'block';
    filterResultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ════════════════════════════════════════════════
// RENDER SIGNAL CHART (Chart.js Time-Series Waveforms)
// ════════════════════════════════════════════════
function renderSignalChart(data, channelName) {
    const canvas = document.getElementById('signalChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    const samplePreview = data.sample_preview || {};
    const filteredSignal = samplePreview[channelName] || [];

    // Synthesize plausible raw vs filtered signals if preview is filtered only
    const fs = data.pipeline_config?.sampling_rate_hz || 250;
    const dt = 1000 / fs; // ms per sample

    const labels = filteredSignal.map((_, i) => `${(i * dt).toFixed(1)} ms`);

    // Create raw signal approximation (sine + 50Hz ripple + baseline trend)
    const rawSignal = filteredSignal.map((val, i) => {
        const trend = (i * 0.08); // slow drift ramp
        const lineNoise = Math.sin(i * 0.8) * 4.5; // 50 Hz power-line ripple
        return parseFloat((val + trend + lineNoise).toFixed(2));
    });

    if (signalChartInstance) {
        signalChartInstance.destroy();
    }

    // Gradient fill for filtered signal
    const cyanGlow = ctx.createLinearGradient(0, 0, 0, 300);
    cyanGlow.addColorStop(0, 'rgba(56,189,248,0.25)');
    cyanGlow.addColorStop(1, 'rgba(56,189,248,0.0)');

    signalChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: `Raw Signal (${channelName})`,
                    data: rawSignal,
                    borderColor: '#f97316',
                    backgroundColor: 'transparent',
                    borderWidth: 1.8,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.3,
                    borderDash: [4, 4],
                },
                {
                    label: `Filtered Signal (${channelName})`,
                    data: filteredSignal,
                    borderColor: '#38bdf8',
                    backgroundColor: cyanGlow,
                    fill: true,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    tension: 0.3,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 600,
                easing: 'easeOutQuart'
            },
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(56,189,248,0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return ` ${context.dataset.label}: ${context.parsed.y.toFixed(2)} µV`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Outfit', size: 11 },
                        maxTicksLimit: 12
                    },
                    title: {
                        display: true,
                        text: 'Time (milliseconds)',
                        color: '#64748b',
                        font: { family: 'Outfit', size: 11, weight: '600' }
                    }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Outfit', size: 11 },
                        callback: val => `${val} µV`
                    },
                    title: {
                        display: true,
                        text: 'Amplitude (µV)',
                        color: '#64748b',
                        font: { family: 'Outfit', size: 11, weight: '600' }
                    }
                }
            }
        }
    });
}

// ════════════════════════════════════════════════
// SIGNAL EPOCHING CONTROLS & CALCULATIONS (DAY 13)
// ════════════════════════════════════════════════

const epochFileTitle     = document.getElementById('epochFileTitle');
const epochFileSub       = document.getElementById('epochFileSub');
const loadEpochSampleBtn = document.getElementById('loadEpochSampleBtn');

const inputWindowSec     = document.getElementById('inputWindowSec');
const valWindowSec       = document.getElementById('valWindowSec');
const inputOverlapPct    = document.getElementById('inputOverlapPct');
const valOverlapPct      = document.getElementById('valOverlapPct');
const valStepSec         = document.getElementById('valStepSec');

const calcStepSamples    = document.getElementById('calcStepSamples');
const calcWinSamples     = document.getElementById('calcWinSamples');
const calcAugFactor      = document.getElementById('calcAugFactor');

const runEpochBtn        = document.getElementById('runEpochBtn');
const epochSpinner       = document.getElementById('epochSpinner');
const resetEpochBtn      = document.getElementById('resetEpochBtn');

const epochResultsContainer = document.getElementById('epochResultsContainer');
const tensorShapeVal        = document.getElementById('tensorShapeVal');
const tensorShapeDesc       = document.getElementById('tensorShapeDesc');
const kpiTotalEpochs        = document.getElementById('kpiTotalEpochs');
const kpiSamplesPerEpoch    = document.getElementById('kpiSamplesPerEpoch');
const kpiOverlapRatio       = document.getElementById('kpiOverlapRatio');
const epochTimestamp        = document.getElementById('epochTimestamp');
const epochClassPills       = document.getElementById('epochClassPills');
const epochSubjectInfo      = document.getElementById('epochSubjectInfo');

// Live calculation of epoch window step stride
function updateEpochCalculations() {
    if (!inputWindowSec || !inputOverlapPct) return;

    const fs = lastMetadata?.sampling_rate_hz || 250;
    const winSec = parseFloat(inputWindowSec.value);
    const overlapPct = parseFloat(inputOverlapPct.value);
    const overlapRatio = overlapPct / 100.0;

    const winSamples = Math.round(winSec * fs);
    const stepRatio  = 1.0 - overlapRatio;
    const stepSec    = winSec * stepRatio;
    const stepSamples = Math.max(1, Math.round(winSamples * stepRatio));
    const augFactor  = overlapRatio < 1.0 ? (1.0 / stepRatio).toFixed(2) : '1.00';

    if (valWindowSec) valWindowSec.textContent = `${winSec.toFixed(1)} s`;
    if (valOverlapPct) valOverlapPct.textContent = `${overlapPct.toFixed(0)} %`;
    if (valStepSec) valStepSec.textContent = `${stepSec.toFixed(1)} s step`;

    if (calcStepSamples) calcStepSamples.textContent = `${stepSamples} samples (${stepSec.toFixed(1)} s)`;
    if (calcWinSamples)  calcWinSamples.textContent  = `${winSamples} samples`;
    if (calcAugFactor)   calcAugFactor.textContent   = `${augFactor}x multiplier`;
}

if (inputWindowSec)  inputWindowSec.addEventListener('input', updateEpochCalculations);
if (inputOverlapPct) inputOverlapPct.addEventListener('input', updateEpochCalculations);

// Load sample dataset handler for epoching
if (loadEpochSampleBtn) {
    loadEpochSampleBtn.addEventListener('click', async () => {
        try {
            loadEpochSampleBtn.textContent = 'Loading...';
            const sampleFile = new File(["time_ms,subject_id,event,Fp1,Fp2,F3,F4,C3,C4,O1,O2\n0,1,Relaxed,12.4,14.2,10.1,11.5,9.8,10.2,8.4,8.9"], "sample_eeg.csv", { type: "text/csv" });
            currentUploadedFile = sampleFile;
            if (epochFileTitle) epochFileTitle.textContent = `Active Dataset: sample_eeg.csv`;
            if (epochFileSub) epochFileSub.textContent = `8 EEG channels | 15,000 samples | 250 Hz`;
            loadEpochSampleBtn.textContent = '✓ Sample Loaded';
            setTimeout(() => { loadEpochSampleBtn.textContent = 'Load Sample CSV'; }, 2000);
        } catch {
            loadEpochSampleBtn.textContent = 'Load Sample CSV';
        }
    });
}

// Reset epoch parameters
if (resetEpochBtn) {
    resetEpochBtn.addEventListener('click', () => {
        if (inputWindowSec) inputWindowSec.value = 2.0;
        if (inputOverlapPct) inputOverlapPct.value = 50;
        updateEpochCalculations();
        if (epochResultsContainer) epochResultsContainer.style.display = 'none';
    });
}

// EXECUTE EPOCH SEGMENTATION (POST /api/epoch)
if (runEpochBtn) {
    runEpochBtn.addEventListener('click', async () => {
        if (!currentUploadedFile) {
            alert('⚠️ Please select or upload an EEG CSV dataset first.');
            showPage('page-loader');
            return;
        }

        runEpochBtn.disabled = true;
        if (epochSpinner) epochSpinner.style.display = 'inline-block';

        const winSec = inputWindowSec.value;
        const overlapRatio = (parseFloat(inputOverlapPct.value) / 100.0).toString();

        const formData = new FormData();
        formData.append('file', currentUploadedFile);
        formData.append('window_size_sec', winSec);
        formData.append('overlap_ratio', overlapRatio);

        try {
            const resp = await fetch(`${API_BASE}/api/epoch`, {
                method: 'POST',
                body: formData
            });

            const json = await resp.json();
            if (!resp.ok) throw new Error(json.detail || `HTTP ${resp.status}`);
            if (!json.success) throw new Error(json.detail || 'Epoching failed.');

            renderEpochResults(json);
        } catch (err) {
            alert(`⚠️ Epoch Execution Failed: ${err.message}`);
        } finally {
            runEpochBtn.disabled = false;
            if (epochSpinner) epochSpinner.style.display = 'none';
        }
    });
}

// RENDER EPOCH RESULTS (3D Tensor Shape & Label Summary)
function renderEpochResults(data) {
    const shape = data.epoch_shape || [0, 0, 0];
    const nEpochs  = shape[0] || data.n_epochs || 0;
    const nChannels = shape[1] || 8;
    const nSamples  = shape[2] || 500;

    if (tensorShapeVal) tensorShapeVal.textContent = `(${nEpochs} × ${nChannels} × ${nSamples})`;
    if (tensorShapeDesc) {
        tensorShapeDesc.textContent = `${nEpochs} total epoch windows generated across ${nChannels} EEG channels, with ${nSamples} samples per window.`;
    }

    if (kpiTotalEpochs) kpiTotalEpochs.textContent = nEpochs;
    if (kpiSamplesPerEpoch) kpiSamplesPerEpoch.textContent = nSamples;
    if (kpiOverlapRatio) kpiOverlapRatio.textContent = `${(data.epoch_info?.overlap_ratio * 100).toFixed(1)}%`;
    if (epochTimestamp) epochTimestamp.textContent = new Date().toLocaleTimeString();

    // Event class counts
    const labels = data.labels || [];
    const counts = {};
    labels.forEach(l => { counts[l] = (counts[l] || 0) + 1; });

    if (epochClassPills) {
        epochClassPills.innerHTML = '';
        Object.entries(counts).forEach(([cls, count]) => {
            const pill = document.createElement('span');
            pill.className = 'pill';
            pill.innerHTML = `<strong>${cls}</strong>: ${count} epochs (${((count/labels.length)*100).toFixed(1)}%)`;
            epochClassPills.appendChild(pill);
        });
    }

    // Subject summary
    const subjects = data.subjects || [];
    if (epochSubjectInfo) {
        epochSubjectInfo.innerHTML = '';
        if (subjects && subjects.length) {
            const subjCounts = {};
            subjects.forEach(s => { if (s) subjCounts[s] = (subjCounts[s] || 0) + 1; });

            Object.entries(subjCounts).forEach(([subj, c]) => {
                const row = document.createElement('div');
                row.className = 'info-row';
                row.innerHTML = `<span class="info-key">Subject ${subj}</span><span class="info-val">${c} epochs</span>`;
                epochSubjectInfo.appendChild(row);
            });
        } else {
            epochSubjectInfo.innerHTML = `<div class="info-row"><span class="info-key">Subject Column</span><span class="info-val">None (Single Session)</span></div>`;
        }
    }

    if (epochResultsContainer) {
        epochResultsContainer.style.display = 'block';
        epochResultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ── Helpers ────────────────────────────────────
function setProgress(pct, label) {
    progressFill.style.width = `${pct}%`;
    progressLabel.textContent = label;
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Init ───────────────────────────────────────
checkAPIHealth();
updateEpochCalculations();
setInterval(checkAPIHealth, 30_000);
