"""Deterministic clinical safety guardrails, physiological rule checks, and extractive RAG evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Contraindication & alternative regimen rules for all supported allergy classes
ALLERGY_GUARDRAILS: Dict[str, Tuple[str, str]] = {
    "Penicillin": (
        "ALLERGY GUARDRAIL: Documented Penicillin allergy — Beta-Lactam/Ampicillin/Piperacillin orders blocked.",
        "Use respiratory fluoroquinolone (Levofloxacin 750mg IV) or Aztreonam + Vancomycin for empiric coverage.",
    ),
    "Cephalosporin": (
        "ALLERGY GUARDRAIL: Documented Cephalosporin allergy — Ceftriaxone/Cefepime orders blocked.",
        "Use respiratory fluoroquinolone (Levofloxacin 750mg IV) or Aztreonam for gram-negative coverage.",
    ),
    "Sulfa": (
        "ALLERGY GUARDRAIL: Documented Sulfa allergy — Trimethoprim-Sulfamethoxazole (Bactrim) orders blocked.",
        "Use non-sulfonamide antimicrobial or diuretic alternatives.",
    ),
    "Vancomycin": (
        "ALLERGY GUARDRAIL: Documented Vancomycin allergy — Glycopeptide orders blocked.",
        "Use Linezolid 600mg IV/PO q12h or Daptomycin (if non-pulmonary source) for MRSA coverage.",
    ),
    "Aspirin": (
        "ALLERGY GUARDRAIL: Documented Aspirin allergy — NSAID/Salicylate orders blocked.",
        "Use Acetaminophen for antipyretic/analgesic management; use Clopidogrel if antiplatelet therapy is required.",
    ),
}


def extract_rag_evidence(rag_citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts the clinical conclusion or primary evidence takeaway from each retrieved QA document."""
    evidence_list: List[Dict[str, Any]] = []
    for doc in rag_citations:
        doc_id = str(doc.get("doc_id", "UNKNOWN"))
        title = str(doc.get("title", "")).strip()
        snippet = str(doc.get("snippet", "")).strip()
        score = float(doc.get("cross_encoder_score", 0.0))

        # Prefer pre-extracted full-text conclusion from rag.py; fall back to parsing snippet
        explicit_conclusion = str(doc.get("conclusion", "")).strip()
        if explicit_conclusion:
            conclusion = explicit_conclusion
        elif "Conclusion:" in snippet:
            conclusion = snippet.split("Conclusion:", 1)[1].strip()
        else:
            sentences = [s.strip() for s in snippet.split(". ") if s.strip()]
            conclusion = ". ".join(sentences[-2:]) if len(sentences) >= 2 else snippet

        evidence_list.append(
            {
                "doc_id": doc_id,
                "title": title,
                "extracted_conclusion": conclusion,
                "cross_encoder_score": score,
            }
        )
    return evidence_list


def compute_synthesis_and_safety_guardrails(
    unsupervised_report: Dict[str, Any],
    tabular_report: Dict[str, Any],
    vitals_report: Dict[str, Any],
    xray_report: Dict[str, Any],
    nlp_report: Dict[str, Any],
    rag_citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes composite acuity, ESI triage level, multi-scenario safety alerts, orders, and RAG evidence."""
    p_tab = float(tabular_report["readmission_or_icu_risk_prob"])
    p_seq = float(vitals_report["deterioration_6h_prob"])
    p_vis = float(xray_report["pneumonia_probability"])
    composite = float(0.40 * p_tab + 0.35 * p_seq + 0.25 * p_vis)

    esi = (
        "ESI Level 1 — Immediate ICU / Resuscitation Triage"
        if composite >= 0.60
        else (
            "ESI Level 2 — High-Risk Inpatient Step-Down"
            if composite >= 0.35
            else "ESI Level 4 — Low-Acuity Observation"
        )
    )

    alerts: List[str] = []
    orders: List[str] = []

    extracted_evidence = extract_rag_evidence(rag_citations)
    cited_ids = [ev["doc_id"] for ev in extracted_evidence]

    # 1. Unsupervised Outlier Check
    if unsupervised_report.get("is_outlier_flagged"):
        alerts.append(
            "ISOLATION FOREST OUTLIER: Extreme clinical/sensor profile detected; verify bedside vitals manually."
        )

    # 2. Multi-Allergy Contraindication Guardrails
    allergies: List[str] = nlp_report.get("documented_allergies", [])
    has_beta_lactam_allergy = False
    for allergy in allergies:
        if allergy in ALLERGY_GUARDRAILS:
            alert_msg, alt_order = ALLERGY_GUARDRAILS[allergy]
            alerts.append(alert_msg)
            if allergy in {"Penicillin", "Cephalosporin"}:
                has_beta_lactam_allergy = True
            if p_vis >= 0.50 or allergy not in {"Penicillin", "Cephalosporin"}:
                orders.append(f"{alt_order} (Grounded in {cited_ids}).")

    # 3. Chest X-Ray & Respiratory Infection Orders
    zone = xray_report.get("gradcam_anatomical_zone", "lung field")
    if p_vis >= 0.50:
        if not has_beta_lactam_allergy:
            orders.append(
                f"Initiate empiric CAP antimicrobial regimen (Ceftriaxone + Azithromycin) for {zone} consolidation "
                f"and obtain sputum/blood cultures (Grounded in {cited_ids})."
            )
        else:
            orders.append(
                f"Obtain STAT blood/sputum cultures and procalcitonin for {zone} pulmonary consolidation (Grounded in {cited_ids})."
            )

    # 4. Dynamic 6-Hour Vitals Trajectory Guardrails
    proj_vitals: Dict[str, float] = vitals_report.get("projected_vitals_6h", {})
    spo2 = float(proj_vitals.get("spo2_pct", 98.0))
    sbp = float(proj_vitals.get("systolic_bp_mmhg", 120.0))
    hr = float(proj_vitals.get("heart_rate_bpm", 80.0))
    rr = float(proj_vitals.get("resp_rate_bpm", 16.0))
    temp_c = float(proj_vitals.get("temp_c", 37.0))

    if spo2 < 92.0 or rr >= 24.0:
        alerts.append(
            f"RESPIRATORY DETERIORATION ALERT: Projected 6h SpO2={spo2:.1f}% / RR={rr:.1f} bpm indicates acute hypoxia/tachypnea."
        )
        orders.append(
            "Titrate supplemental O2 to maintain SpO2 >= 94% and obtain STAT arterial blood gas (ABG)."
        )

    if sbp < 95.0 or (hr >= 105.0 and temp_c >= 38.3):
        alerts.append(
            f"HEMODYNAMIC / SEPSIS ALERT: Projected 6h SBP={sbp:.1f} mmHg, HR={hr:.1f} bpm, Temp={temp_c:.1f}°C."
        )
        orders.append(
            "Initiate sepsis fluid resuscitation bundle (30 mL/kg IV crystalloid), hourly MAP monitoring, and STAT serum lactate."
        )

    # 5. Active Home Medication Interaction Guardrails
    meds_lower = {m.lower() for m in nlp_report.get("extracted_medications", [])}
    if any("metformin" in m for m in meds_lower) and (spo2 < 92.0 or composite >= 0.50):
        alerts.append(
            "MEDICATION HOLD GUARDRAIL: Hold home Metformin due to acute hypoxia/sepsis lactic acidosis risk."
        )
    if any(m in meds_lower for m in ("lisinopril", "losartan", "enalapril")) and sbp < 100.0:
        alerts.append(
            "MEDICATION HOLD GUARDRAIL: Hold home ACE-inhibitor/ARB (e.g., Lisinopril) due to borderline/low systolic BP."
        )
    if any(m in meds_lower for m in ("warfarin", "apixaban", "rivaroxaban")):
        alerts.append(
            "ANTICOAGULATION ALERT: Active anticoagulant therapy documented; check baseline PT/INR and platelet count."
        )

    # 6. TreeSHAP Tabular Risk Driver Orders
    shap_drivers: List[Dict[str, Any]] = tabular_report.get("top_treeshap_drivers", [])
    positive_drivers = [d["feature"] for d in shap_drivers if d.get("direction") == "increases_risk"]
    if "num_medications" in positive_drivers[:2]:
        orders.append("Order clinical pharmacist medication reconciliation for high polypharmacy readmission risk.")
    if "number_inpatient" in positive_drivers[:2] or "number_emergency" in positive_drivers[:2]:
        orders.append("Schedule early transitional care management and 48-hour post-discharge follow-up.")

    # 7. Baseline Acuity Monitoring
    if composite >= 0.50:
        orders.append(
            f"Continuous telemetry + hourly SpO2/MAP trend checks; repeat STAT serum lactate (Grounded in {cited_ids})."
        )
    elif not orders:
        orders.append(
            f"Standard observation protocol and 48h outpatient follow-up (Grounded in {cited_ids})."
        )

    hitl_status = (
        "PENDING_PHYSICIAN_APPROVAL"
        if (composite >= 0.35 or alerts)
        else "AUTO_CLEARED_LOW_RISK"
    )

    return {
        "composite_acuity_score": round(composite, 4),
        "esi_triage_level": esi,
        "safety_guardrail_alerts": alerts,
        "recommended_clinical_orders": orders,
        "extracted_rag_evidence": extracted_evidence,
        "hitl_clinician_status": hitl_status,
    }
