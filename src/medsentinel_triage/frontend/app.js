/**
 * MedSentinel Triage Frontend Controller
 * Connects file uploads (CSV, PNG/JPG, TXT), preprocessing endpoints (`prepare_patient_input.py`),
 * individual model endpoints, and the stateful LangGraph HITL triage pipeline.
 */

const appState = {
  tabular_input: null,
  vitals_24h: null,
  xray_28x28: null,
  clinical_note: "",
  activeThreadId: null,
  lastPreparedPayload: null,
};

// DOM Elements
const el = {
  healthBadge: document.getElementById("health-badge"),
  deviceBadge: document.getElementById("device-badge"),
  alertBanner: document.getElementById("alert-banner"),
  alertBannerText: document.getElementById("alert-banner-text"),
  alertBannerClose: document.getElementById("alert-banner-close"),

  patientIdInput: document.getElementById("patient-id-input"),
  padMethodSelect: document.getElementById("pad-method-select"),
  tabularRowIndex: document.getElementById("tabular-row-index"),

  fileTabular: document.getElementById("file-tabular"),
  fileVitals: document.getElementById("file-vitals"),
  fileXray: document.getElementById("file-xray"),
  fileNote: document.getElementById("file-note"),
  textareaNote: document.getElementById("textarea-note"),

  statusTabular: document.getElementById("status-tabular"),
  statusVitals: document.getElementById("status-vitals"),
  statusXray: document.getElementById("status-xray"),
  statusNote: document.getElementById("status-note"),

  previewTabular: document.getElementById("preview-tabular"),
  previewVitals: document.getElementById("preview-vitals"),
  canvasXrayInput: document.getElementById("canvas-xray-input"),
  xrayMetaText: document.getElementById("xray-meta-text"),

  payloadReadyIndicator: document.getElementById("payload-ready-indicator"),
  btnViewPayload: document.getElementById("btn-view-payload"),
  payloadJsonDrawer: document.getElementById("payload-json-drawer"),
  payloadJsonPre: document.getElementById("payload-json-pre"),

  triageResultsSection: document.getElementById("triage-results-section"),
  latencyBadge: document.getElementById("latency-badge"),
  threadBadge: document.getElementById("thread-badge"),
  esiBanner: document.getElementById("esi-banner"),
  esiTitle: document.getElementById("esi-title"),
  phenotypeSubtitle: document.getElementById("phenotype-subtitle"),
  compositeScoreVal: document.getElementById("composite-score-val"),
  hitlStatusBadge: document.getElementById("hitl-status-badge"),
  hitlPanel: document.getElementById("hitl-panel"),
  hitlClinicianName: document.getElementById("hitl-clinician-name"),
  hitlClinicianNotes: document.getElementById("hitl-clinician-notes"),
  alertsList: document.getElementById("alerts-list"),
  ordersList: document.getElementById("orders-list"),

  tabRiskBadge: document.getElementById("tab-risk-badge"),
  tabModelSummary: document.getElementById("tab-model-summary"),
  shapBarsContainer: document.getElementById("shap-bars-container"),

  vitalsRiskBadge: document.getElementById("vitals-risk-badge"),
  vitalsModelSummary: document.getElementById("vitals-model-summary"),
  vitalsChartContainer: document.getElementById("vitals-chart-container"),
  vitalsProjTable: document.getElementById("vitals-proj-table"),

  xrayRiskBadge: document.getElementById("xray-risk-badge"),
  xrayModelSummary: document.getElementById("xray-model-summary"),
  canvasXrayRaw: document.getElementById("canvas-xray-raw"),
  canvasXrayGradcam: document.getElementById("canvas-xray-gradcam"),

  nlpStatusBadge: document.getElementById("nlp-status-badge"),
  nlpEntitiesContainer: document.getElementById("nlp-entities-container"),

  ragCustomQuery: document.getElementById("rag-custom-query"),
  ragResultsContainer: document.getElementById("rag-results-container"),
  consoleSummaryPre: document.getElementById("console-summary-pre"),
};

function showBanner(msg, type = "error") {
  el.alertBannerText.textContent = msg;
  el.alertBanner.className = `alert-banner ${type === "info" ? "info" : ""}`;
}

function hideBanner() {
  el.alertBanner.className = "alert-banner hidden";
}

el.alertBannerClose.addEventListener("click", hideBanner);

function updatePayloadReadiness() {
  el.textareaNote.value = el.textareaNote.value || appState.clinical_note;
  appState.clinical_note = el.textareaNote.value.trim();

  const readyCount = [
    Boolean(appState.tabular_input),
    Boolean(appState.vitals_24h),
    Boolean(appState.xray_28x28),
    Boolean(appState.clinical_note),
  ].filter(Boolean).length;

  if (readyCount === 4) {
    el.payloadReadyIndicator.textContent = "All 4 modalities preprocessed and ready for LangGraph clinical triage.";
    appState.lastPreparedPayload = {
      patient_id: el.patientIdInput.value.trim() || "PATIENT-001",
      tabular_input: appState.tabular_input,
      vitals_24h: appState.vitals_24h,
      xray_28x28: appState.xray_28x28,
      clinical_note: appState.clinical_note,
    };
    el.btnViewPayload.classList.remove("hidden");
    el.payloadJsonPre.textContent = JSON.stringify(appState.lastPreparedPayload, null, 2);
  } else {
    el.payloadReadyIndicator.textContent = `${readyCount}/4 modalities preprocessed in state (or select all 4 files to upload & run directly).`;
  }
}

el.textareaNote.addEventListener("input", () => {
  appState.clinical_note = el.textareaNote.value.trim();
  if (appState.clinical_note) {
    el.statusNote.textContent = `Ready (${appState.clinical_note.length} chars)`;
    el.statusNote.className = "badge badge-ready";
  }
  updatePayloadReadiness();
});

// Render 28x28 grayscale matrix + optional Grad-CAM heatmap onto a <canvas>
function drawXrayCanvas(canvas, img28x28, heatmap28x28 = null, hotspotRC = null) {
  if (!canvas || !img28x28 || !img28x28.length) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const scaleX = w / 28;
  const scaleY = h / 28;

  ctx.clearRect(0, 0, w, h);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      const gray = Number(img28x28[r][c]) || 0;
      if (heatmap28x28 && heatmap28x28[r] && heatmap28x28[r][c] !== undefined) {
        const cam = Math.max(0, Math.min(1, Number(heatmap28x28[r][c])));
        // Blend grayscale with thermal jet/hot colormap
        const red = Math.min(255, Math.round(gray * 0.45 + 255 * Math.min(1, cam * 1.6)));
        const green = Math.min(255, Math.round(gray * 0.45 + 255 * Math.max(0, Math.min(1, (cam - 0.35) * 1.5))));
        const blue = Math.min(255, Math.round(gray * 0.45 + 180 * Math.max(0, 0.5 - cam)));
        ctx.fillStyle = `rgb(${red},${green},${blue})`;
      } else {
        ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
      }
      ctx.fillRect(Math.floor(c * scaleX), Math.floor(r * scaleY), Math.ceil(scaleX), Math.ceil(scaleY));
    }
  }

  if (hotspotRC && hotspotRC.length === 2) {
    const [hr, hc] = hotspotRC;
    const cx = (hc + 0.5) * scaleX;
    const cy = (hr + 0.5) * scaleY;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// Render preprocessed Tabular preview
function renderTabularPreview(tabularData, filename = "tabular.csv") {
  appState.tabular_input = tabularData;
  el.statusTabular.textContent = `Preprocessed (${filename})`;
  el.statusTabular.className = "badge badge-ready";
  el.previewTabular.className = "modality-preview";
  const items = Object.entries(tabularData)
    .map(([k, v]) => `<div class="mini-kv"><span>${k}</span><strong>${v}</strong></div>`)
    .join("");
  el.previewTabular.innerHTML = `<div class="mini-grid">${items}</div>`;
  updatePayloadReadiness();
}

// Render preprocessed Vitals preview
function renderVitalsPreview(vitals24h, filename = "vitals.csv") {
  appState.vitals_24h = vitals24h;
  el.statusVitals.textContent = `24x5 Ready (${filename})`;
  el.statusVitals.className = "badge badge-ready";
  el.previewVitals.className = "modality-preview";
  const h0 = vitals24h[0];
  const h23 = vitals24h[vitals24h.length - 1];
  el.previewVitals.innerHTML = `
    <div class="mini-grid">
      <div class="mini-kv"><span>HR (h0 &rarr; h23)</span><strong>${h0[0]} &rarr; ${h23[0]} bpm</strong></div>
      <div class="mini-kv"><span>SpO2 (h0 &rarr; h23)</span><strong>${h0[1]} &rarr; ${h23[1]}%</strong></div>
      <div class="mini-kv"><span>SBP (h0 &rarr; h23)</span><strong>${h0[2]} &rarr; ${h23[2]} mmHg</strong></div>
      <div class="mini-kv"><span>RespRate (h0 &rarr; h23)</span><strong>${h0[3]} &rarr; ${h23[3]} bpm</strong></div>
      <div class="mini-kv"><span>TempC (h0 &rarr; h23)</span><strong>${h0[4]} &rarr; ${h23[4]} &deg;C</strong></div>
      <div class="mini-kv"><span>Tensor Shape</span><strong>24 &times; 5 float32</strong></div>
    </div>
  `;
  updatePayloadReadiness();
}

// Render preprocessed X-Ray preview
function renderXrayPreview(xray28x28, filename = "chest_xray.png") {
  appState.xray_28x28 = xray28x28;
  el.statusXray.textContent = `28x28 Ready (${filename})`;
  el.statusXray.className = "badge badge-ready";
  el.canvasXrayInput.classList.remove("hidden");
  drawXrayCanvas(el.canvasXrayInput, xray28x28);
  drawXrayCanvas(el.canvasXrayRaw, xray28x28);
  el.xrayMetaText.className = "";
  el.xrayMetaText.innerHTML = `
    <div><strong>File:</strong> ${filename}</div>
    <div><strong>Preprocessed Shape:</strong> 28 &times; 28 uint8 grayscale</div>
    <div><strong>Model Input:</strong> Bilinear 3 &times; 64 &times; 64 tensor</div>
  `;
  updatePayloadReadiness();
}

// Render preprocessed Note preview
function renderNotePreview(noteText, filename = "note.txt") {
  appState.clinical_note = noteText;
  el.textareaNote.value = noteText;
  el.statusNote.textContent = `Ready (${filename})`;
  el.statusNote.className = "badge badge-ready";
  updatePayloadReadiness();
}

// Render Model 1 & 2: Tabular Phenotyper + LightGBM & TreeSHAP
function renderTabularPrediction(unsupReport, shapReport) {
  const prob = (shapReport.readmission_or_icu_risk_prob * 100).toFixed(1);
  el.tabRiskBadge.textContent = `Risk: ${prob}% (${shapReport.risk_tier})`;
  el.tabRiskBadge.className = `badge ${
    shapReport.risk_tier === "HIGH"
      ? "badge-danger"
      : shapReport.risk_tier === "MODERATE"
      ? "badge-warning"
      : "badge-ready"
  }`;

  el.tabModelSummary.innerHTML = `
    <div><strong>Assigned Cohort:</strong> ${unsupReport.phenotype_name}</div>
    <div><strong>Isolation Forest Anomaly Score:</strong> ${unsupReport.isolation_forest_anomaly_score} (${
      unsupReport.is_outlier_flagged ? "OUTLIER FLAGGED" : "Normal Distribution"
    })</div>
  `;

  const drivers = shapReport.top_treeshap_drivers || [];
  const maxImpact = Math.max(...drivers.map((d) => Math.abs(d.shap_log_odds_impact)), 0.01);
  el.shapBarsContainer.innerHTML = drivers
    .map((d) => {
      const pct = Math.min(100, Math.round((Math.abs(d.shap_log_odds_impact) / maxImpact) * 100));
      const isPos = d.shap_log_odds_impact > 0;
      const sign = isPos ? "+" : "";
      return `
        <div class="shap-row">
          <div class="shap-row-top">
            <span><strong>${d.feature}</strong> = ${d.value}</span>
            <span>${sign}${d.shap_log_odds_impact.toFixed(4)} (${d.direction})</span>
          </div>
          <div class="shap-bar-track">
            <div class="shap-bar-fill ${isPos ? "shap-pos" : "shap-neg"}" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

// Render Model 3: PyTorch Bi-LSTM Vitals Trajectory
function renderVitalsPrediction(vitalsReport) {
  const prob = (vitalsReport.deterioration_6h_prob * 100).toFixed(1);
  el.vitalsRiskBadge.textContent = `6h Deterioration: ${prob}%`;
  el.vitalsRiskBadge.className = `badge ${
    vitalsReport.deterioration_6h_prob >= 0.5 ? "badge-danger" : "badge-ready"
  }`;

  const peakStr = (vitalsReport.peak_attention_hours || []).map((h) => `Hour ${h}`).join(", ");
  el.vitalsModelSummary.innerHTML = `
    <div><strong>Peak MultiheadAttention Hours:</strong> ${peakStr}</div>
  `;

  const proj = vitalsReport.projected_vitals_6h || {};
  el.vitalsProjTable.innerHTML = `
    <div class="mini-grid">
      <div class="mini-kv"><span>Projected 6h HR</span><strong>${proj.heart_rate_bpm} bpm</strong></div>
      <div class="mini-kv"><span>Projected 6h SpO2</span><strong>${proj.spo2_pct}%</strong></div>
      <div class="mini-kv"><span>Projected 6h SBP</span><strong>${proj.systolic_bp_mmhg} mmHg</strong></div>
      <div class="mini-kv"><span>Projected 6h RR</span><strong>${proj.resp_rate_bpm} bpm</strong></div>
      <div class="mini-kv"><span>Projected 6h Temp</span><strong>${proj.temp_c} &deg;C</strong></div>
    </div>
  `;

  // Draw SVG sparkline for HR and SpO2 over 24 hours if vitals_24h is loaded
  if (appState.vitals_24h && appState.vitals_24h.length === 24) {
    const w = 440;
    const h = 90;
    const hrVals = appState.vitals_24h.map((r) => r[0]);
    const spo2Vals = appState.vitals_24h.map((r) => r[1]);
    const minVal = Math.min(...hrVals, ...spo2Vals) - 4;
    const maxVal = Math.max(...hrVals, ...spo2Vals) + 4;

    const toPts = (vals) =>
      vals
        .map((v, i) => {
          const x = 16 + (i / 23) * (w - 32);
          const y = h - 14 - ((v - minVal) / Math.max(1, maxVal - minVal)) * (h - 28);
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");

    el.vitalsChartContainer.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;">
        <polyline fill="none" stroke="#ef4444" stroke-width="2" points="${toPts(hrVals)}" />
        <polyline fill="none" stroke="#0284c7" stroke-width="2" points="${toPts(spo2Vals)}" />
        <text x="20" y="14" font-size="10" fill="#ef4444" font-weight="600">HR (bpm)</text>
        <text x="85" y="14" font-size="10" fill="#0284c7" font-weight="600">SpO2 (%)</text>
      </svg>
    `;
  }
}

// Render Model 4: ResNet-18 Chest X-Ray & Grad-CAM
function renderXrayPrediction(xrayReport) {
  const prob = (xrayReport.pneumonia_probability * 100).toFixed(1);
  el.xrayRiskBadge.textContent = `Pneumonia: ${prob}%`;
  el.xrayRiskBadge.className = `badge ${
    xrayReport.pneumonia_probability >= 0.5 ? "badge-danger" : "badge-ready"
  }`;

  el.xrayModelSummary.innerHTML = `
    <div><strong>Radiology Impression:</strong> ${xrayReport.radiology_impression}</div>
    <div><strong>Anatomical Zone:</strong> ${xrayReport.gradcam_anatomical_zone} (Hotspot r=${xrayReport.gradcam_hotspot_rc[0]}, c=${xrayReport.gradcam_hotspot_rc[1]})</div>
  `;

  if (appState.xray_28x28) {
    drawXrayCanvas(el.canvasXrayRaw, appState.xray_28x28);
    drawXrayCanvas(
      el.canvasXrayGradcam,
      appState.xray_28x28,
      xrayReport.gradcam_heatmap_28x28 || null,
      xrayReport.gradcam_hotspot_rc || null
    );
  }
}

// Render Model 5: Hugging Face Biomedical NER + NegEx
function renderNLPPrediction(nlpReport) {
  el.nlpStatusBadge.textContent = "Entities Extracted";
  el.nlpStatusBadge.className = "badge badge-ready";

  const makePills = (items, cls) =>
    items && items.length
      ? items.map((x) => `<span class="entity-pill ${cls}">${x}</span>`).join("")
      : `<span class="empty-preview">None detected</span>`;

  el.nlpEntitiesContainer.innerHTML = `
    <div class="nlp-group">
      <div class="nlp-group-title">Documented Allergies</div>
      <div class="pill-list">${makePills(nlpReport.documented_allergies, "pill-allergy")}</div>
    </div>
    <div class="nlp-group">
      <div class="nlp-group-title">Active Home Medications</div>
      <div class="pill-list">${makePills(nlpReport.extracted_medications, "pill-med")}</div>
    </div>
    <div class="nlp-group">
      <div class="nlp-group-title">Affirmed Clinical Findings</div>
      <div class="pill-list">${makePills(nlpReport.affirmed_clinical_findings, "pill-affirmed")}</div>
    </div>
    <div class="nlp-group">
      <div class="nlp-group-title">Negated Findings (45-char NegEx Scope)</div>
      <div class="pill-list">${makePills(nlpReport.negated_clinical_findings, "pill-negated")}</div>
    </div>
  `;
}

// Render Model 6: LangChain Hybrid RAG Citations
function renderRAGCitations(citations, extractedEvidence = null) {
  const list = extractedEvidence && extractedEvidence.length ? extractedEvidence : citations || [];
  if (!list.length) {
    el.ragResultsContainer.innerHTML = `<div class="empty-preview">No matching documents returned.</div>`;
    return;
  }

  el.ragResultsContainer.innerHTML = list
    .map((doc) => {
      const conclusionText = doc.extracted_conclusion || doc.conclusion || doc.snippet || "";
      return `
        <div class="rag-doc-card">
          <div class="rag-doc-top">
            <span>${doc.doc_id}</span>
            <span>CrossEncoder: ${doc.cross_encoder_score}</span>
          </div>
          <div class="rag-doc-title">${doc.title}</div>
          <div class="rag-doc-conclusion"><strong>Extracted Conclusion:</strong> ${conclusionText}</div>
        </div>
      `;
    })
    .join("");
}

// Render Full LangGraph Triage State
function renderFullTriageResponse(data) {
  const state = data.state;
  appState.activeThreadId = data.thread_id;

  el.triageResultsSection.classList.remove("hidden");
  el.latencyBadge.textContent = `Execution: ${data.elapsed_ms || "--"} ms`;
  el.threadBadge.textContent = `Thread: ${data.thread_id}`;

  el.esiTitle.textContent = state.esi_triage_level;
  el.phenotypeSubtitle.textContent = state.unsupervised_phenotype_report?.phenotype_name || "";
  el.compositeScoreVal.textContent = `${(state.composite_acuity_score * 100).toFixed(1)}%`;

  if (state.composite_acuity_score >= 0.6) {
    el.esiBanner.className = "esi-banner";
  } else if (state.composite_acuity_score >= 0.35) {
    el.esiBanner.className = "esi-banner esi-moderate";
  } else {
    el.esiBanner.className = "esi-banner esi-low";
  }

  el.hitlStatusBadge.textContent = state.hitl_clinician_status;
  if (state.hitl_clinician_status === "APPROVED_BY_ATTENDING_PHYSICIAN") {
    el.hitlStatusBadge.className = "badge badge-ready";
    el.hitlPanel.className = "hitl-panel approved";
  } else {
    el.hitlStatusBadge.className = "badge badge-warning";
    el.hitlPanel.className = "hitl-panel";
  }

  const alerts = state.safety_guardrail_alerts || [];
  el.alertsList.innerHTML = alerts.length
    ? alerts.map((a) => `<li>${a}</li>`).join("")
    : `<li>No critical safety contraindications triggered.</li>`;

  const orders = state.recommended_clinical_orders || [];
  el.ordersList.innerHTML = orders.map((o) => `<li>${o}</li>`).join("");

  // Populate individual model inspector cards
  if (state.unsupervised_phenotype_report && state.tabular_risk_shap_report) {
    renderTabularPrediction(state.unsupervised_phenotype_report, state.tabular_risk_shap_report);
  }
  if (state.vitals_trajectory_report) {
    renderVitalsPrediction(state.vitals_trajectory_report);
  }
  if (state.chest_xray_gradcam_report) {
    renderXrayPrediction(state.chest_xray_gradcam_report);
  }
  if (state.clinical_nlp_report) {
    renderNLPPrediction(state.clinical_nlp_report);
  }
  renderRAGCitations(state.hybrid_rag_citations, state.extracted_rag_evidence);

  el.consoleSummaryPre.textContent = data.formatted_summary || JSON.stringify(state, null, 2);
}

// API Helpers
async function checkBackendHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    el.healthBadge.textContent = data.status === "healthy" ? "6/6 Checkpoints Ready" : "Checkpoints Missing";
    el.healthBadge.className = `status-pill ${data.status === "healthy" ? "status-ok" : "badge-warning"}`;
    el.deviceBadge.textContent = `Device: ${String(data.device).toUpperCase()}`;
  } catch (err) {
    el.healthBadge.textContent = "Backend Offline";
    el.healthBadge.className = "status-pill badge-danger";
  }
}

// 1. Load Sample test_data/ Files
document.getElementById("btn-load-sample").addEventListener("click", async () => {
  hideBanner();
  try {
    const res = await fetch("/api/sample-files/load");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load sample files.");

    const p = data.payload;
    el.patientIdInput.value = p.patient_id;
    renderTabularPreview(p.tabular_input, "tabular.csv");
    renderVitalsPreview(p.vitals_24h, "vitals_8h.csv");
    renderXrayPreview(p.xray_28x28, "chest_xray.png");
    renderNotePreview(p.clinical_note, "note.txt");
    showBanner("Loaded and preprocessed test_data/ (tabular.csv, vitals_8h.csv, chest_xray.png, note.txt).", "info");
  } catch (err) {
    showBanner(err.message);
  }
});

// 2. Individual Modality Preprocessing Buttons
document.getElementById("btn-preprocess-tabular").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileTabular.files[0];
  if (!file) return showBanner("Please select a tabular .csv, .tsv, or .json file first.");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("row_index", el.tabularRowIndex.value || "0");
  try {
    const res = await fetch("/api/preprocess/tabular", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderTabularPreview(data.tabular_input, data.filename);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-preprocess-vitals").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileVitals.files[0];
  if (!file) return showBanner("Please select a vitals .csv, .tsv, or .json file first.");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("pad_method", el.padMethodSelect.value);
  try {
    const res = await fetch("/api/preprocess/vitals", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderVitalsPreview(data.vitals_24h, data.filename);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-preprocess-xray").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileXray.files[0];
  if (!file) return showBanner("Please select a Chest X-Ray .png or .jpg file first.");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/preprocess/xray", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderXrayPreview(data.xray_28x28, data.filename);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-preprocess-note").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileNote.files[0];
  const fd = new FormData();
  if (file) {
    fd.append("file", file);
  } else if (el.textareaNote.value.trim()) {
    fd.append("clinical_note", el.textareaNote.value.trim());
  } else {
    return showBanner("Please select a .txt file or enter clinical note text first.");
  }
  try {
    const res = await fetch("/api/preprocess/note", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderNotePreview(data.clinical_note, data.filename);
  } catch (err) {
    showBanner(err.message);
  }
});

// Auto-preprocess when a user picks a file
el.fileTabular.addEventListener("change", () => document.getElementById("btn-preprocess-tabular").click());
el.fileVitals.addEventListener("change", () => document.getElementById("btn-preprocess-vitals").click());
el.fileXray.addEventListener("change", () => document.getElementById("btn-preprocess-xray").click());
el.fileNote.addEventListener("change", () => document.getElementById("btn-preprocess-note").click());

// 3. Individual Model Prediction Buttons
document.getElementById("btn-predict-tabular").addEventListener("click", async () => {
  hideBanner();
  if (!appState.tabular_input) return showBanner("Preprocess or load tabular data first.");
  try {
    const res = await fetch("/api/predict/tabular", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tabular_input: appState.tabular_input }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderTabularPrediction(data.unsupervised_phenotype_report, data.tabular_risk_shap_report);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-predict-vitals").addEventListener("click", async () => {
  hideBanner();
  if (!appState.vitals_24h) return showBanner("Preprocess or load vitals data first.");
  try {
    const res = await fetch("/api/predict/vitals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vitals_24h: appState.vitals_24h }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderVitalsPrediction(data.vitals_trajectory_report);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-predict-xray").addEventListener("click", async () => {
  hideBanner();
  if (!appState.xray_28x28) return showBanner("Preprocess or load a Chest X-Ray image first.");
  try {
    const res = await fetch("/api/predict/xray", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ xray_28x28: appState.xray_28x28 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderXrayPrediction(data.chest_xray_gradcam_report);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-predict-nlp").addEventListener("click", async () => {
  hideBanner();
  const note = el.textareaNote.value.trim() || appState.clinical_note;
  if (!note) return showBanner("Provide or upload a clinical note first.");
  try {
    const res = await fetch("/api/predict/nlp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clinical_note: note }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderNLPPrediction(data.clinical_nlp_report);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-predict-rag").addEventListener("click", async () => {
  hideBanner();
  const query = el.ragCustomQuery.value.trim();
  if (!query) return showBanner("Enter a clinical search query for RAG retrieval.");
  try {
    const res = await fetch("/api/predict/rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clinical_query: query, top_k: 3 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderRAGCitations(data.hybrid_rag_citations);
  } catch (err) {
    showBanner(err.message);
  }
});

// 4. Combined Patient Preparation (/api/prepare-patient)
document.getElementById("btn-prepare-payload").addEventListener("click", async () => {
  hideBanner();
  const hasAllFiles =
    el.fileTabular.files[0] &&
    el.fileVitals.files[0] &&
    el.fileXray.files[0] &&
    (el.fileNote.files[0] || el.textareaNote.value.trim());

  if (hasAllFiles) {
    const fd = new FormData();
    fd.append("patient_id", el.patientIdInput.value.trim() || "PATIENT-001");
    fd.append("pad_method", el.padMethodSelect.value);
    fd.append("row_index", el.tabularRowIndex.value || "0");
    fd.append("tabular_file", el.fileTabular.files[0]);
    fd.append("vitals_file", el.fileVitals.files[0]);
    fd.append("xray_file", el.fileXray.files[0]);
    if (el.fileNote.files[0]) {
      fd.append("note_file", el.fileNote.files[0]);
    } else {
      fd.append("clinical_note", el.textareaNote.value.trim());
    }

    try {
      const res = await fetch("/api/prepare-patient", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      const p = data.payload;
      renderTabularPreview(p.tabular_input, data.uploaded_filenames.tabular_file);
      renderVitalsPreview(p.vitals_24h, data.uploaded_filenames.vitals_file);
      renderXrayPreview(p.xray_28x28, data.uploaded_filenames.xray_file);
      renderNotePreview(p.clinical_note, data.uploaded_filenames.note_file);
      el.payloadJsonDrawer.classList.remove("hidden");
      showBanner("Prepared validated patient JSON payload via /api/prepare-patient.", "info");
    } catch (err) {
      showBanner(err.message);
    }
  } else if (appState.lastPreparedPayload) {
    updatePayloadReadiness();
    el.payloadJsonDrawer.classList.remove("hidden");
    showBanner("Prepared patient payload is ready in state.", "info");
  } else {
    showBanner("Select all 4 modality files (CSV, CSV, PNG, TXT) or click 'Load test_data/ Sample' first.");
  }
});

el.btnViewPayload.addEventListener("click", () => {
  el.payloadJsonDrawer.classList.toggle("hidden");
});

document.getElementById("btn-close-drawer").addEventListener("click", () => {
  el.payloadJsonDrawer.classList.add("hidden");
});

document.getElementById("btn-download-payload").addEventListener("click", () => {
  if (!appState.lastPreparedPayload) return;
  const blob = new Blob([JSON.stringify(appState.lastPreparedPayload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${appState.lastPreparedPayload.patient_id}_input.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// 5. Run Full LangGraph Clinical Triage (/api/triage/run or /api/triage/upload-and-run)
const btnRunTriage = document.getElementById("btn-run-triage");
btnRunTriage.addEventListener("click", async () => {
  hideBanner();
  updatePayloadReadiness();

  btnRunTriage.disabled = true;
  const origText = btnRunTriage.textContent;
  btnRunTriage.textContent = "Running LangGraph Models...";

  try {
    // If all 4 modalities are already preprocessed in state, call /api/triage/run
    if (appState.tabular_input && appState.vitals_24h && appState.xray_28x28 && appState.clinical_note) {
      const payload = {
        patient_id: el.patientIdInput.value.trim() || "PATIENT-001",
        tabular_input: appState.tabular_input,
        vitals_24h: appState.vitals_24h,
        xray_28x28: appState.xray_28x28,
        clinical_note: appState.clinical_note,
        auto_approve_hitl: false,
      };
      const res = await fetch("/api/triage/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      renderFullTriageResponse(data);
    } else if (
      el.fileTabular.files[0] &&
      el.fileVitals.files[0] &&
      el.fileXray.files[0] &&
      (el.fileNote.files[0] || el.textareaNote.value.trim())
    ) {
      // Otherwise upload raw files and run via /api/triage/upload-and-run
      const fd = new FormData();
      fd.append("patient_id", el.patientIdInput.value.trim() || "PATIENT-001");
      fd.append("pad_method", el.padMethodSelect.value);
      fd.append("row_index", el.tabularRowIndex.value || "0");
      fd.append("tabular_file", el.fileTabular.files[0]);
      fd.append("vitals_file", el.fileVitals.files[0]);
      fd.append("xray_file", el.fileXray.files[0]);
      if (el.fileNote.files[0]) {
        fd.append("note_file", el.fileNote.files[0]);
      } else {
        fd.append("clinical_note", el.textareaNote.value.trim());
      }
      const res = await fetch("/api/triage/upload-and-run", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      const p = data.prepared_payload;
      renderTabularPreview(p.tabular_input, data.uploaded_filenames.tabular_file);
      renderVitalsPreview(p.vitals_24h, data.uploaded_filenames.vitals_file);
      renderXrayPreview(p.xray_28x28, data.uploaded_filenames.xray_file);
      renderNotePreview(p.clinical_note, data.uploaded_filenames.note_file);
      renderFullTriageResponse(data);
    } else {
      throw new Error(
        "Please upload all 4 modality files (Tabular CSV, Vitals CSV, Chest X-Ray PNG, Clinical Note TXT) or click 'Load test_data/ Sample' first."
      );
    }
  } catch (err) {
    showBanner(err.message);
  } finally {
    btnRunTriage.disabled = false;
    btnRunTriage.textContent = origText;
  }
});

// 6. Attending Physician HITL Sign-Off (/api/triage/approve-hitl)
async function submitHITLDecision(action) {
  hideBanner();
  if (!appState.activeThreadId) return showBanner("No active LangGraph triage thread to sign off.");
  try {
    const res = await fetch("/api/triage/approve-hitl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: appState.activeThreadId,
        clinician_name: el.hitlClinicianName.value.trim() || "Attending Physician",
        clinician_notes: el.hitlClinicianNotes.value.trim() || null,
        action,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    el.hitlStatusBadge.textContent = data.hitl_clinician_status;
    if (action === "approve") {
      el.hitlStatusBadge.className = "badge badge-ready";
      el.hitlPanel.className = "hitl-panel approved";
      if (data.formatted_summary) {
        el.consoleSummaryPre.textContent = data.formatted_summary;
      }
      showBanner(`Orders approved and released by ${data.clinician_name} (Thread: ${data.thread_id}).`, "info");
    } else {
      el.hitlStatusBadge.className = "badge badge-danger";
      showBanner(`Clinical orders placed on hold by ${data.clinician_name}.`);
    }
  } catch (err) {
    showBanner(err.message);
  }
}

document.getElementById("btn-approve-hitl").addEventListener("click", () => submitHITLDecision("approve"));
document.getElementById("btn-reject-hitl").addEventListener("click", () => submitHITLDecision("reject"));

// Reset Button
document.getElementById("btn-reset").addEventListener("click", () => {
  window.location.reload();
});

// Initial Health Check
checkBackendHealth();
