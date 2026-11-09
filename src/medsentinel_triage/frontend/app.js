/**
 * MedSentinel Triage Frontend Controller
 * Connects single-patient and batch clinical data uploads (Two-CSV or JSON),
 * preprocessing, individual diagnostic models, and physician sign-off.
 */

const appState = {
  mode: "single",
  tabular_input: null,
  vitals_24h: null,
  xray_28x28: null,
  clinical_note: "",
  activeThreadId: null,
  activePatientId: null,
  activeBatchIndex: null,
  lastPreparedPayload: null,
  batchPatients: null,
  batchResults: null,
  batchElapsedMs: null,
};

// DOM Elements
const el = {
  healthBadge: document.getElementById("health-badge"),
  deviceBadge: document.getElementById("device-badge"),
  alertBanner: document.getElementById("alert-banner"),
  alertBannerText: document.getElementById("alert-banner-text"),
  alertBannerClose: document.getElementById("alert-banner-close"),

  tabBtnSingle: document.getElementById("tab-btn-single"),
  tabBtnBatch: document.getElementById("tab-btn-batch"),
  viewSingle: document.getElementById("view-single"),
  viewBatch: document.getElementById("view-batch"),
  patientInspectorWrapper: document.getElementById("patient-inspector-wrapper"),
  inspectorHeaderTitle: document.getElementById("inspector-header-title"),

  patientIdInput: document.getElementById("patient-id-input"),
  padMethodSelect: document.getElementById("pad-method-select"),
  batchPadMethodSelect: document.getElementById("batch-pad-method-select"),
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

  // Batch Elements (Two-CSV or JSON)
  fileBatchManifest: document.getElementById("file-batch-manifest"),
  fileBatchVitals: document.getElementById("file-batch-vitals"),
  fileBatchJson: document.getElementById("file-batch-json"),
  batchMaxRows: document.getElementById("batch-max-rows"),
  batchAutoApprove: document.getElementById("batch-auto-approve"),
  batchStatusBadge: document.getElementById("batch-status-badge"),
  batchPreviewBox: document.getElementById("batch-preview-box"),
  batchReadyIndicator: document.getElementById("batch-ready-indicator"),
  btnDownloadBatchResults: document.getElementById("btn-download-batch-results"),
  batchResultsWrapper: document.getElementById("batch-results-wrapper"),
  batchKpiStrip: document.getElementById("batch-kpi-strip"),
  batchTableBody: document.getElementById("batch-table-body"),

  triageResultsSection: document.getElementById("triage-results-section"),
  latencyBadge: document.getElementById("latency-badge"),
  threadBadge: document.getElementById("thread-badge"),
  esiBanner: document.getElementById("esi-banner"),
  esiTitle: document.getElementById("esi-title"),
  phenotypeSubtitle: document.getElementById("phenotype-subtitle"),
  compositeScoreVal: document.getElementById("composite-score-val"),
  hitlStatusBadge: document.getElementById("hitl-status-badge"),
  hitlPanel: document.getElementById("hitl-panel"),
  hitlPanelTitle: document.getElementById("hitl-panel-title"),
  hitlPanelDesc: document.getElementById("hitl-panel-desc"),
  hitlControls: document.getElementById("hitl-controls"),
  hitlClinicianName: document.getElementById("hitl-clinician-name"),
  hitlClinicianNotes: document.getElementById("hitl-clinician-notes"),
  btnApproveHitl: document.getElementById("btn-approve-hitl"),
  btnRejectHitl: document.getElementById("btn-reject-hitl"),
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

// Sync Vitals Baseline Padding selectors between Single and Batch tabs
el.padMethodSelect.addEventListener("change", () => {
  if (el.batchPadMethodSelect) {
    el.batchPadMethodSelect.value = el.padMethodSelect.value;
  }
});

if (el.batchPadMethodSelect) {
  el.batchPadMethodSelect.addEventListener("change", () => {
    el.padMethodSelect.value = el.batchPadMethodSelect.value;
  });
}

// Mode Tab Switching
window.switchModeTab = function (mode) {
  hideBanner();
  appState.mode = mode;
  if (mode === "batch") {
    el.tabBtnBatch.classList.add("active");
    el.tabBtnSingle.classList.remove("active");
    el.viewBatch.classList.remove("hidden");
    el.viewSingle.classList.add("hidden");
    if (!appState.batchResults || appState.batchResults.length === 0) {
      el.patientInspectorWrapper.classList.add("hidden");
    }
  } else {
    el.tabBtnSingle.classList.add("active");
    el.tabBtnBatch.classList.remove("active");
    el.viewSingle.classList.remove("hidden");
    el.viewBatch.classList.add("hidden");
  }
};

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
    el.payloadReadyIndicator.textContent = "All 4 clinical inputs are validated and ready for triage.";
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
    el.payloadReadyIndicator.textContent = `${readyCount} of 4 clinical modalities validated.`;
  }
}

el.patientIdInput.addEventListener("input", updatePayloadReadiness);

el.textareaNote.addEventListener("input", () => {
  appState.clinical_note = el.textareaNote.value.trim();
  if (appState.clinical_note) {
    el.statusNote.textContent = `Validated (${appState.clinical_note.length} chars)`;
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
function renderTabularPreview(tabularData) {
  appState.tabular_input = tabularData;
  el.statusTabular.textContent = "Validated";
  el.statusTabular.className = "badge badge-ready";
  el.previewTabular.className = "modality-preview";
  const items = Object.entries(tabularData)
    .map(([k, v]) => `<div class="mini-kv"><span>${k}</span><strong>${v}</strong></div>`)
    .join("");
  el.previewTabular.innerHTML = `<div class="mini-grid">${items}</div>`;
  updatePayloadReadiness();
}

// Render preprocessed Vitals preview
function renderVitalsPreview(vitals24h) {
  appState.vitals_24h = vitals24h;
  el.statusVitals.textContent = "24h Stream Ready";
  el.statusVitals.className = "badge badge-ready";
  el.previewVitals.className = "modality-preview";
  const h0 = vitals24h[0];
  const h23 = vitals24h[vitals24h.length - 1];
  el.previewVitals.innerHTML = `
    <div class="mini-grid">
      <div class="mini-kv"><span>Heart Rate (Start &rarr; Latest)</span><strong>${h0[0]} &rarr; ${h23[0]} bpm</strong></div>
      <div class="mini-kv"><span>SpO2 (Start &rarr; Latest)</span><strong>${h0[1]} &rarr; ${h23[1]}%</strong></div>
      <div class="mini-kv"><span>Systolic BP (Start &rarr; Latest)</span><strong>${h0[2]} &rarr; ${h23[2]} mmHg</strong></div>
      <div class="mini-kv"><span>Resp Rate (Start &rarr; Latest)</span><strong>${h0[3]} &rarr; ${h23[3]} bpm</strong></div>
      <div class="mini-kv"><span>Temperature (Start &rarr; Latest)</span><strong>${h0[4]} &rarr; ${h23[4]} &deg;C</strong></div>
      <div class="mini-kv"><span>Stream Window</span><strong>24 Hours &times; 5 Channels</strong></div>
    </div>
  `;
  updatePayloadReadiness();
}

// Render preprocessed X-Ray preview
function renderXrayPreview(xray28x28) {
  appState.xray_28x28 = xray28x28;
  el.statusXray.textContent = "Radiograph Ready";
  el.statusXray.className = "badge badge-ready";
  el.canvasXrayInput.classList.remove("hidden");
  drawXrayCanvas(el.canvasXrayInput, xray28x28);
  drawXrayCanvas(el.canvasXrayRaw, xray28x28);
  el.xrayMetaText.className = "";
  el.xrayMetaText.innerHTML = `
    <div><strong>Status:</strong> Validated Grayscale Radiograph</div>
    <div><strong>Resolution:</strong> 28 &times; 28 Normalized Matrix</div>
  `;
  updatePayloadReadiness();
}

// Render preprocessed Note preview
function renderNotePreview(noteText) {
  appState.clinical_note = noteText;
  el.textareaNote.value = noteText;
  el.statusNote.textContent = `Validated (${noteText.length} chars)`;
  el.statusNote.className = "badge badge-ready";
  updatePayloadReadiness();
}

// Render Model 1 & 2: Tabular Phenotyper + LightGBM & TreeSHAP
function renderTabularPrediction(unsupReport, shapReport) {
  el.patientInspectorWrapper.classList.remove("hidden");
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
    <div><strong>Anomaly Score:</strong> ${unsupReport.isolation_forest_anomaly_score} (${
      unsupReport.is_outlier_flagged ? "Outlier Profile Flagged" : "Within Normal Cohort Distribution"
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
function renderVitalsPrediction(vitalsReport, vitals24hOverride = null) {
  el.patientInspectorWrapper.classList.remove("hidden");
  const prob = (vitalsReport.deterioration_6h_prob * 100).toFixed(1);
  el.vitalsRiskBadge.textContent = `6h Deterioration: ${prob}%`;
  el.vitalsRiskBadge.className = `badge ${
    vitalsReport.deterioration_6h_prob >= 0.5 ? "badge-danger" : "badge-ready"
  }`;

  const peakStr = (vitalsReport.peak_attention_hours || []).map((h) => `Hour ${h}`).join(", ");
  el.vitalsModelSummary.innerHTML = `
    <div><strong>Peak Temporal Attention Hours:</strong> ${peakStr}</div>
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

  const series = vitals24hOverride || appState.vitals_24h;
  if (series && series.length === 24) {
    const w = 440;
    const h = 90;
    const hrVals = series.map((r) => r[0]);
    const spo2Vals = series.map((r) => r[1]);
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
function renderXrayPrediction(xrayReport, xray28x28Override = null) {
  el.patientInspectorWrapper.classList.remove("hidden");
  const prob = (xrayReport.pneumonia_probability * 100).toFixed(1);
  el.xrayRiskBadge.textContent = `Pneumonia: ${prob}%`;
  el.xrayRiskBadge.className = `badge ${
    xrayReport.pneumonia_probability >= 0.5 ? "badge-danger" : "badge-ready"
  }`;

  el.xrayModelSummary.innerHTML = `
    <div><strong>Radiology Impression:</strong> ${xrayReport.radiology_impression}</div>
    <div><strong>Anatomical Region:</strong> ${xrayReport.gradcam_anatomical_zone} (Focal Hotspot r=${xrayReport.gradcam_hotspot_rc[0]}, c=${xrayReport.gradcam_hotspot_rc[1]})</div>
  `;

  const img = xray28x28Override || appState.xray_28x28;
  if (img) {
    drawXrayCanvas(el.canvasXrayRaw, img);
    drawXrayCanvas(
      el.canvasXrayGradcam,
      img,
      xrayReport.gradcam_heatmap_28x28 || null,
      xrayReport.gradcam_hotspot_rc || null
    );
  }
}

// Render Model 5: Biomedical NER + NegEx
function renderNLPPrediction(nlpReport) {
  el.patientInspectorWrapper.classList.remove("hidden");
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
      <div class="nlp-group-title">Negated Symptoms</div>
      <div class="pill-list">${makePills(nlpReport.negated_clinical_findings, "pill-negated")}</div>
    </div>
  `;
}

// Render Model 6: Clinical Literature Citations
function renderRAGCitations(citations, extractedEvidence = null) {
  el.patientInspectorWrapper.classList.remove("hidden");
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
            <span>Relevance Score: ${doc.cross_encoder_score}</span>
          </div>
          <div class="rag-doc-title">${doc.title}</div>
          <div class="rag-doc-conclusion"><strong>Conclusion:</strong> ${conclusionText}</div>
        </div>
      `;
    })
    .join("");
}

// Synchronize the Physician Sign-Off Panel (#hitl-panel) so buttons are hidden/locked once signed off
function syncHitlPanelState(status) {
  el.hitlStatusBadge.textContent = status;
  const controlsEl = el.hitlControls || document.querySelector("#hitl-panel .hitl-controls");
  const titleEl = el.hitlPanelTitle || document.querySelector("#hitl-panel h3");
  const descEl = el.hitlPanelDesc || document.querySelector("#hitl-panel p");

  if (status === "APPROVED_BY_ATTENDING_PHYSICIAN") {
    el.hitlStatusBadge.className = "badge badge-ready";
    el.hitlPanel.className = "hitl-panel approved";
    if (titleEl) titleEl.textContent = "Attending Physician Sign-Off Completed";
    if (descEl) descEl.textContent = "Clinical orders have been approved and released by the Attending Physician.";
    if (controlsEl) controlsEl.classList.add("hidden");
    if (el.btnApproveHitl) el.btnApproveHitl.disabled = true;
    if (el.btnRejectHitl) el.btnRejectHitl.disabled = true;
  } else if (status === "HOLD_BY_ATTENDING_PHYSICIAN") {
    el.hitlStatusBadge.className = "badge badge-danger";
    el.hitlPanel.className = "hitl-panel held";
    if (titleEl) titleEl.textContent = "Clinical Orders Placed on Hold";
    if (descEl) descEl.textContent = "Orders have been placed on hold by the Attending Physician pending review.";
    if (controlsEl) controlsEl.classList.add("hidden");
    if (el.btnApproveHitl) el.btnApproveHitl.disabled = true;
    if (el.btnRejectHitl) el.btnRejectHitl.disabled = true;
  } else if (status === "AUTO_CLEARED_LOW_RISK") {
    el.hitlStatusBadge.className = "badge badge-ready";
    el.hitlPanel.className = "hitl-panel approved";
    if (titleEl) titleEl.textContent = "Auto-Cleared Low-Acuity Profile";
    if (descEl) {
      descEl.textContent =
        "Low-acuity observation patient with zero clinical safety alerts — no physician sign-off required.";
    }
    if (controlsEl) controlsEl.classList.add("hidden");
    if (el.btnApproveHitl) el.btnApproveHitl.disabled = true;
    if (el.btnRejectHitl) el.btnRejectHitl.disabled = true;
  } else {
    // PENDING_PHYSICIAN_APPROVAL
    el.hitlStatusBadge.className = "badge badge-warning";
    el.hitlPanel.className = "hitl-panel";
    if (titleEl) titleEl.textContent = "Attending Physician Verification & Sign-Off";
    if (descEl) {
      descEl.textContent =
        "Review the clinical safety alerts and recommended orders below before releasing orders.";
    }
    if (controlsEl) controlsEl.classList.remove("hidden");
    if (el.btnApproveHitl) el.btnApproveHitl.disabled = false;
    if (el.btnRejectHitl) el.btnRejectHitl.disabled = false;
  }
}

// Render Full Patient Triage State into the Inspector (stays on Batch tab when in Batch mode)
function renderFullTriageResponse(data) {
  const state = data.state;
  appState.activeThreadId = data.thread_id;
  appState.activePatientId = state.patient_id;

  el.patientInspectorWrapper.classList.remove("hidden");
  el.triageResultsSection.classList.remove("hidden");
  el.inspectorHeaderTitle.textContent = `Patient Report: ${state.patient_id} — Triage Synthesis & Physician Sign-Off`;
  el.latencyBadge.textContent = `Execution: ${data.elapsed_ms || "--"} ms`;
  el.threadBadge.textContent = `Session: ${data.thread_id}`;

  el.esiTitle.textContent = `${state.patient_id} — ${state.esi_triage_level}`;
  el.phenotypeSubtitle.textContent = state.unsupervised_phenotype_report?.phenotype_name || "";
  el.compositeScoreVal.textContent = `${(state.composite_acuity_score * 100).toFixed(1)}%`;

  if (state.composite_acuity_score >= 0.6) {
    el.esiBanner.className = "esi-banner";
  } else if (state.composite_acuity_score >= 0.35) {
    el.esiBanner.className = "esi-banner esi-moderate";
  } else {
    el.esiBanner.className = "esi-banner esi-low";
  }

  syncHitlPanelState(state.hitl_clinician_status);

  const alerts = state.safety_guardrail_alerts || [];
  el.alertsList.innerHTML = alerts.length
    ? alerts.map((a) => `<li>${a}</li>`).join("")
    : `<li>No critical safety contraindications triggered.</li>`;

  const orders = state.recommended_clinical_orders || [];
  el.ordersList.innerHTML = orders.map((o) => `<li>${o}</li>`).join("");

  if (state.unsupervised_phenotype_report && state.tabular_risk_shap_report) {
    renderTabularPrediction(state.unsupervised_phenotype_report, state.tabular_risk_shap_report);
  }
  if (state.vitals_trajectory_report) {
    renderVitalsPrediction(state.vitals_trajectory_report, state.vitals_24h_list);
  }
  if (state.chest_xray_gradcam_report) {
    renderXrayPrediction(state.chest_xray_gradcam_report, state.xray_28x28_list);
  }
  if (state.clinical_nlp_report) {
    renderNLPPrediction(state.clinical_nlp_report);
  }
  renderRAGCitations(state.hybrid_rag_citations, state.extracted_rag_evidence);

  el.consoleSummaryPre.textContent = data.formatted_summary || JSON.stringify(state, null, 2);
}

// Render Batch Preview List
function renderBatchCohortPreview(patients) {
  appState.batchPatients = patients;
  el.batchStatusBadge.textContent = `Validated (${patients.length} Patients)`;
  el.batchStatusBadge.className = "badge badge-ready";
  el.batchReadyIndicator.textContent = `${patients.length} patient record(s) validated (all paths verified) and ready for batch triage.`;
  el.batchPreviewBox.className = "modality-preview";
  el.batchPreviewBox.innerHTML = patients
    .map(
      (p, i) =>
        `<div><strong>#${i + 1} ${p.patient_id}</strong> &mdash; Age: ${p.tabular_input.age_years}, Medications: ${p.tabular_input.num_medications}, Vitals: ${p.vitals_24h.length}h (h0 HR=${p.vitals_24h[0][0]} &rarr; h23 HR=${p.vitals_24h[p.vitals_24h.length - 1][0]}), Radiograph &amp; Note: Verified</div>`
    )
    .join("");
}

// Live-compute and render the Batch KPI Strip from appState.batchResults
function renderBatchKpiStrip() {
  if (!appState.batchResults || !appState.batchResults.length) return;
  const results = appState.batchResults;
  const totalPatients = results.length;

  let esi1 = 0;
  let esi2 = 0;
  let esi4 = 0;
  let sumAcuity = 0;
  let totalAlerts = 0;
  let pendingCount = 0;

  for (const item of results) {
    const s = item.state || {};
    const esiStr = String(s.esi_triage_level || "");
    if (esiStr.includes("ESI Level 1")) esi1 += 1;
    else if (esiStr.includes("ESI Level 2")) esi2 += 1;
    else esi4 += 1;

    sumAcuity += Number(s.composite_acuity_score || 0);
    totalAlerts += (s.safety_guardrail_alerts || []).length;
    if (s.hitl_clinician_status === "PENDING_PHYSICIAN_APPROVAL") {
      pendingCount += 1;
    }
  }

  const meanAcuityPct = ((sumAcuity / totalPatients) * 100).toFixed(1);
  const pendingColor = pendingCount === 0 ? "#15803d" : "#b45309";

  el.batchKpiStrip.innerHTML = `
    <div class="batch-kpi-card"><span>Total Patients</span><strong>${totalPatients}</strong></div>
    <div class="batch-kpi-card"><span>ESI Level 1 (Immediate ICU)</span><strong style="color:#b91c1c">${esi1}</strong></div>
    <div class="batch-kpi-card"><span>ESI Level 2 (Step-Down)</span><strong style="color:#b45309">${esi2}</strong></div>
    <div class="batch-kpi-card"><span>ESI Level 4 (Observation)</span><strong style="color:#15803d">${esi4}</strong></div>
    <div class="batch-kpi-card"><span>Mean Composite Acuity</span><strong>${meanAcuityPct}%</strong></div>
    <div class="batch-kpi-card"><span>Safety Alerts</span><strong style="color:#b91c1c">${totalAlerts}</strong></div>
    <div class="batch-kpi-card"><span>Pending Sign-Off</span><strong id="batch-pending-hitl-count" style="color:${pendingColor}">${pendingCount}</strong></div>
  `;
}

// Render the Batch Table rows from appState.batchResults
function renderBatchTableRows() {
  if (!appState.batchResults) return;
  el.batchTableBody.innerHTML = appState.batchResults
    .map((item, idx) => {
      const s = item.state;
      const compPct = ((s.composite_acuity_score || 0) * 100).toFixed(1);
      const tabPct = ((s.tabular_risk_shap_report?.readmission_or_icu_risk_prob || 0) * 100).toFixed(1);
      const vitPct = ((s.vitals_trajectory_report?.deterioration_6h_prob || 0) * 100).toFixed(1);
      const xrayPct = ((s.chest_xray_gradcam_report?.pneumonia_probability || 0) * 100).toFixed(1);
      const allergies = (s.clinical_nlp_report?.documented_allergies || []).join(", ");
      const alertCount = (s.safety_guardrail_alerts || []).length;

      const esiClass =
        s.composite_acuity_score >= 0.6
          ? "badge-danger"
          : s.composite_acuity_score >= 0.35
          ? "badge-warning"
          : "badge-ready";

      const isPending = s.hitl_clinician_status === "PENDING_PHYSICIAN_APPROVAL";
      const isClearedOrApproved =
        s.hitl_clinician_status === "APPROVED_BY_ATTENDING_PHYSICIAN" ||
        s.hitl_clinician_status === "AUTO_CLEARED_LOW_RISK";
      const hitlBadgeClass = isClearedOrApproved
        ? "badge-ready"
        : isPending
        ? "badge-warning"
        : "badge-danger";

      return `
        <tr>
          <td><strong>${s.patient_id}</strong></td>
          <td><span class="badge ${esiClass}">${s.esi_triage_level.split("—")[0].trim()}</span></td>
          <td><strong>${compPct}%</strong></td>
          <td>${s.unsupervised_phenotype_report?.phenotype_name?.split(":")[0] || "--"}</td>
          <td>${tabPct}% (${s.tabular_risk_shap_report?.risk_tier || "--"})</td>
          <td>${vitPct}%</td>
          <td>${xrayPct}% (${s.chest_xray_gradcam_report?.gradcam_anatomical_zone || "--"})</td>
          <td>Allergy: <strong>${allergies}</strong> | Alerts: <strong>${alertCount}</strong></td>
          <td><span id="batch-hitl-badge-${idx}" class="badge ${hitlBadgeClass}">${s.hitl_clinician_status}</span></td>
          <td>
            <div class="inline-row">
              <button class="btn btn-sm btn-outline" type="button" onclick="window.inspectBatchPatient(${idx})">
                View Full Report
              </button>
              ${
                isPending
                  ? `<button id="batch-approve-btn-${idx}" class="btn btn-sm btn-success" type="button" onclick="window.approveBatchPatient(${idx})">Approve</button>`
                  : ""
              }
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

// Render Batch Results Table & Cohort KPI Strip
function renderBatchResults(batchData) {
  appState.batchResults = batchData.results || [];
  appState.batchElapsedMs = batchData.elapsed_ms;
  el.batchResultsWrapper.classList.remove("hidden");
  el.btnDownloadBatchResults.classList.remove("hidden");
  el.batchStatusBadge.textContent = `Completed (${batchData.patient_count} Patients in ${batchData.elapsed_ms} ms)`;
  el.batchStatusBadge.className = "badge badge-ready";

  renderBatchKpiStrip();
  renderBatchTableRows();

  if (appState.batchResults.length > 0) {
    appState.activeBatchIndex = 0;
    renderFullTriageResponse(appState.batchResults[0]);
  }
}

// Stay on Batch Tab and scroll smoothly down to the Selected Patient Report
window.inspectBatchPatient = function (idx) {
  if (!appState.batchResults || !appState.batchResults[idx]) return;
  appState.activeBatchIndex = idx;
  const item = appState.batchResults[idx];
  el.patientIdInput.value = item.state.patient_id;
  renderFullTriageResponse(item);
  el.patientInspectorWrapper.scrollIntoView({ behavior: "smooth" });
};

// Helper to merge approved/held status into a batch item while preserving Grad-CAM heatmap
function applySignOffToBatchItem(idx, data, fallbackStatus) {
  if (!appState.batchResults || !appState.batchResults[idx]) return;
  const item = appState.batchResults[idx];
  const prevHeatmap = item.state?.chest_xray_gradcam_report?.gradcam_heatmap_28x28;
  const newStatus = data.hitl_clinician_status || fallbackStatus;

  if (data.state) {
    item.state = data.state;
    if (prevHeatmap && item.state.chest_xray_gradcam_report) {
      item.state.chest_xray_gradcam_report.gradcam_heatmap_28x28 = prevHeatmap;
    }
  }
  if (item.state) {
    item.state.hitl_clinician_status = newStatus;
  }
  if (data.formatted_summary) {
    item.formatted_summary = data.formatted_summary;
  }
}

window.approveBatchPatient = async function (idx) {
  if (!appState.batchResults || !appState.batchResults[idx]) return;
  appState.activeBatchIndex = idx;
  const item = appState.batchResults[idx];
  try {
    const res = await fetch("/api/triage/approve-hitl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: item.thread_id,
        clinician_name: el.hitlClinicianName.value.trim() || "Attending Physician",
        action: "approve",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    applySignOffToBatchItem(idx, data, "APPROVED_BY_ATTENDING_PHYSICIAN");
    renderBatchKpiStrip();
    renderBatchTableRows();
    renderFullTriageResponse(appState.batchResults[idx]);
    showBanner(`Approved clinical orders for ${item.state.patient_id}.`, "info");
  } catch (err) {
    showBanner(err.message);
  }
};

// API Helpers
async function checkBackendHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    el.healthBadge.textContent = data.status === "healthy" ? "All Models Ready" : "Models Degraded";
    el.healthBadge.className = `status-pill ${data.status === "healthy" ? "status-ok" : "badge-warning"}`;
    el.deviceBadge.textContent = `Device: ${String(data.device).toUpperCase()}`;
  } catch (err) {
    el.healthBadge.textContent = "Server Offline";
    el.healthBadge.className = "status-pill badge-danger";
  }
}

// 1a. Load Sample Single Patient
document.getElementById("btn-load-sample").addEventListener("click", async () => {
  hideBanner();
  try {
    const res = await fetch("/api/sample-files/load");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load sample patient data.");

    window.switchModeTab("single");
    const p = data.payload;
    el.patientIdInput.value = p.patient_id;
    renderTabularPreview(p.tabular_input);
    renderVitalsPreview(p.vitals_24h);
    renderXrayPreview(p.xray_28x28);
    renderNotePreview(p.clinical_note);
    showBanner("Sample patient clinical data loaded and validated.", "info");
  } catch (err) {
    showBanner(err.message);
  }
});

// 1b. Load Sample Batch Cohort (validates all paths in batch_tabular_manifest.csv + batch_vitals.csv)
document.getElementById("btn-load-sample-batch").addEventListener("click", async () => {
  hideBanner();
  try {
    const padMethod = el.batchPadMethodSelect ? el.batchPadMethodSelect.value : "carry_first";
    const maxRows = el.batchMaxRows.value || "50";
    const res = await fetch(
      `/api/sample-files/load-batch?pad_method=${encodeURIComponent(padMethod)}&max_rows=${encodeURIComponent(maxRows)}`
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load sample batch cohort.");

    window.switchModeTab("batch");
    renderBatchCohortPreview(data.patients);
    showBanner(
      `Validated ${data.patient_count} patients from the Two-CSV Cohort (all radiograph, note, and vitals paths verified). Click 'Run Batch Triage' to execute.`,
      "info"
    );
  } catch (err) {
    showBanner(err.message);
  }
});

// 2. Individual Modality Preprocessing Buttons
document.getElementById("btn-preprocess-tabular").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileTabular.files[0];
  if (!file) return showBanner("Please select an admission history CSV or JSON file first.");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("row_index", el.tabularRowIndex.value || "0");
  try {
    const res = await fetch("/api/preprocess/tabular", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderTabularPreview(data.tabular_input);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-preprocess-vitals").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileVitals.files[0];
  if (!file) return showBanner("Please select a vital signs CSV or JSON file first.");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("pad_method", el.padMethodSelect.value);
  try {
    const res = await fetch("/api/preprocess/vitals", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderVitalsPreview(data.vitals_24h);
  } catch (err) {
    showBanner(err.message);
  }
});

document.getElementById("btn-preprocess-xray").addEventListener("click", async () => {
  hideBanner();
  const file = el.fileXray.files[0];
  if (!file) return showBanner("Please select a Chest X-Ray image file first.");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/preprocess/xray", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderXrayPreview(data.xray_28x28);
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
    return showBanner("Please select a text file or enter clinical note text first.");
  }
  try {
    const res = await fetch("/api/preprocess/note", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderNotePreview(data.clinical_note);
  } catch (err) {
    showBanner(err.message);
  }
});

// Auto-validate when a file is chosen
el.fileTabular.addEventListener("change", () => document.getElementById("btn-preprocess-tabular").click());
el.fileVitals.addEventListener("change", () => document.getElementById("btn-preprocess-vitals").click());
el.fileXray.addEventListener("change", () => document.getElementById("btn-preprocess-xray").click());
el.fileNote.addEventListener("change", () => document.getElementById("btn-preprocess-note").click());

// Re-validate vitals automatically when the user changes Vitals Baseline Padding
el.padMethodSelect.addEventListener("change", () => {
  if (el.fileVitals.files[0]) {
    document.getElementById("btn-preprocess-vitals").click();
  }
});

// 3. Individual Model Prediction Buttons
document.getElementById("btn-predict-tabular").addEventListener("click", async () => {
  hideBanner();
  if (!appState.tabular_input) return showBanner("Validate or load admission history data first.");
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
  if (!appState.vitals_24h) return showBanner("Validate or load vital signs data first.");
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
  if (!appState.xray_28x28) return showBanner("Validate or load a Chest X-Ray image first.");
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
  if (!query) return showBanner("Enter a clinical search query first.");
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

// 4. Combined Patient Data Preparation
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
      renderTabularPreview(p.tabular_input);
      renderVitalsPreview(p.vitals_24h);
      renderXrayPreview(p.xray_28x28);
      renderNotePreview(p.clinical_note);
      el.payloadJsonDrawer.classList.remove("hidden");
      showBanner("Patient clinical data prepared and validated.", "info");
    } catch (err) {
      showBanner(err.message);
    }
  } else if (appState.lastPreparedPayload) {
    updatePayloadReadiness();
    el.payloadJsonDrawer.classList.remove("hidden");
    showBanner("Prepared patient clinical data is ready.", "info");
  } else {
    showBanner("Please upload all 4 clinical modalities or click 'Load Sample Patient' first.");
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
  a.download = `${appState.lastPreparedPayload.patient_id}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// 5. Run Full Single-Patient Triage
const btnRunTriage = document.getElementById("btn-run-triage");
btnRunTriage.addEventListener("click", async () => {
  hideBanner();
  updatePayloadReadiness();

  btnRunTriage.disabled = true;
  const origText = btnRunTriage.textContent;
  btnRunTriage.textContent = "Running Clinical Triage...";

  try {
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
      appState.activeBatchIndex = null;
      renderFullTriageResponse(data);
    } else if (
      el.fileTabular.files[0] &&
      el.fileVitals.files[0] &&
      el.fileXray.files[0] &&
      (el.fileNote.files[0] || el.textareaNote.value.trim())
    ) {
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
      renderTabularPreview(p.tabular_input);
      renderVitalsPreview(p.vitals_24h);
      renderXrayPreview(p.xray_28x28);
      renderNotePreview(p.clinical_note);
      appState.activeBatchIndex = null;
      renderFullTriageResponse(data);
    } else {
      throw new Error(
        "Please upload all 4 clinical modalities or click 'Load Sample Patient' first."
      );
    }
  } catch (err) {
    showBanner(err.message);
  } finally {
    btnRunTriage.disabled = false;
    btnRunTriage.textContent = origText;
  }
});

// 6. Batch Preprocessing & Batch Triage (Two-CSV or JSON Only)
function buildBatchFormData() {
  const manifestFile = el.fileBatchManifest.files[0];
  const vitalsFile = el.fileBatchVitals.files[0];
  const jsonFile = el.fileBatchJson.files[0];

  const activePadMethod = el.batchPadMethodSelect ? el.batchPadMethodSelect.value : el.padMethodSelect.value;
  const fd = new FormData();
  fd.append("max_rows", el.batchMaxRows.value || "50");
  fd.append("pad_method", activePadMethod);
  fd.append("auto_approve_hitl", el.batchAutoApprove.checked ? "true" : "false");

  if (manifestFile && vitalsFile) {
    fd.append("manifest_csv_file", manifestFile);
    fd.append("vitals_csv_file", vitalsFile);
    return fd;
  }
  if (jsonFile) {
    fd.append("batch_json_file", jsonFile);
    return fd;
  }
  if (manifestFile || vitalsFile) {
    throw new Error(
      "For Two-CSV Batch Processing, please select BOTH the Cohort Tabular Manifest CSV and the Cohort Hourly Vitals CSV."
    );
  }
  throw new Error(
    "Please upload either the Two-CSV Cohort (Manifest CSV + Vitals CSV) or a Batch JSON file (.json), or click 'Load Sample Batch Cohort'."
  );
}

document.getElementById("btn-preprocess-batch").addEventListener("click", async () => {
  hideBanner();
  try {
    const fd = buildBatchFormData();
    const res = await fetch("/api/preprocess/batch", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    renderBatchCohortPreview(data.patients);
    showBanner(
      `Validated ${data.patient_count} patients. All referenced radiograph, clinical note, and vitals paths exist and are verified.`,
      "info"
    );
  } catch (err) {
    showBanner(err.message);
  }
});

function handleBatchInputSelectionChange() {
  appState.batchPatients = null;
  el.batchStatusBadge.textContent = "Files Selected";
  el.batchStatusBadge.className = "badge badge-neutral";
  el.batchReadyIndicator.textContent =
    "Click 'Validate Cohort Paths & Data' or 'Run Batch Triage' to process the cohort.";
}

el.fileBatchManifest.addEventListener("change", () => {
  if (el.fileBatchManifest.files[0]) el.fileBatchJson.value = "";
  handleBatchInputSelectionChange();
});

el.fileBatchVitals.addEventListener("change", () => {
  if (el.fileBatchVitals.files[0]) el.fileBatchJson.value = "";
  handleBatchInputSelectionChange();
});

el.fileBatchJson.addEventListener("change", () => {
  if (el.fileBatchJson.files[0]) {
    el.fileBatchManifest.value = "";
    el.fileBatchVitals.value = "";
  }
  handleBatchInputSelectionChange();
});

if (el.batchPadMethodSelect) {
  el.batchPadMethodSelect.addEventListener("change", () => {
    appState.batchPatients = null;
    if (
      (el.fileBatchManifest.files[0] && el.fileBatchVitals.files[0]) ||
      el.fileBatchJson.files[0]
    ) {
      document.getElementById("btn-preprocess-batch").click();
    }
  });
}

const btnRunBatch = document.getElementById("btn-run-batch");
btnRunBatch.addEventListener("click", async () => {
  hideBanner();
  btnRunBatch.disabled = true;
  const origText = btnRunBatch.textContent;
  btnRunBatch.textContent = "Running Batch Triage...";

  try {
    if (appState.batchPatients && appState.batchPatients.length > 0) {
      const res = await fetch("/api/triage/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patients: appState.batchPatients,
          auto_approve_hitl: el.batchAutoApprove.checked,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      renderBatchResults(data);
    } else {
      const fd = buildBatchFormData();
      const res = await fetch("/api/triage/batch-upload-and-run", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      renderBatchResults(data);
    }
  } catch (err) {
    showBanner(err.message);
  } finally {
    btnRunBatch.disabled = false;
    btnRunBatch.textContent = origText;
  }
});

el.btnDownloadBatchResults.addEventListener("click", () => {
  if (!appState.batchResults) return;
  const blob = new Blob([JSON.stringify(appState.batchResults, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "batch_triage_results.json";
  a.click();
  URL.revokeObjectURL(url);
});

// 7. Attending Physician Sign-Off (from the Single Patient Overview / Patient Triage Report below)
async function submitHITLDecision(action) {
  hideBanner();
  if (!appState.activeThreadId) return showBanner("No active patient triage session to sign off.");
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

    const newStatus =
      data.hitl_clinician_status ||
      (action === "approve" ? "APPROVED_BY_ATTENDING_PHYSICIAN" : "HOLD_BY_ATTENDING_PHYSICIAN");

    // Lock and hide the sign-off controls in the Single Patient Overview immediately
    syncHitlPanelState(newStatus);
    if (data.formatted_summary) {
      el.consoleSummaryPre.textContent = data.formatted_summary;
    }

    // If this patient is part of the Batch Cohort, sync the Batch Table row & decrement Pending Sign-Off
    if (appState.batchResults && appState.batchResults.length > 0) {
      let batchIdx = -1;
      if (
        appState.activeBatchIndex !== null &&
        appState.activeBatchIndex !== undefined &&
        appState.batchResults[appState.activeBatchIndex]
      ) {
        batchIdx = appState.activeBatchIndex;
      } else {
        batchIdx = appState.batchResults.findIndex(
          (r) =>
            r.thread_id === appState.activeThreadId ||
            (r.state && r.state.patient_id === appState.activePatientId)
        );
      }

      if (batchIdx !== -1) {
        applySignOffToBatchItem(batchIdx, data, newStatus);
        renderBatchKpiStrip();
        renderBatchTableRows();
      }
    }

    if (action === "approve") {
      showBanner(`Clinical orders approved and released by ${data.clinician_name}.`, "info");
    } else {
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
