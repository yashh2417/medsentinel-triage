"""LangGraph clinical triage workflow with safety guardrails and Human-in-the-Loop checkpointing."""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from medsentinel_triage.models.chest_xray import TorchvisionChestXRayGradCAM
from medsentinel_triage.models.clinical_nlp import HuggingFaceClinicalNLP
from medsentinel_triage.models.phenotyper import SklearnUnsupervisedPhenotyper
from medsentinel_triage.rag import ProductionClinicalHybridRAG
from medsentinel_triage.models.readmission import LightGBMReadmissionExplainer
from medsentinel_triage.models.vitals import PyTorchVitalsTrainer


class TriageAgentState(TypedDict):
    """Typed state dictionary flowing through the LangGraph clinical triage workflow."""

    patient_id: str
    tabular_row_json: str
    vitals_24h_list: List[List[float]]
    xray_28x28_list: List[List[int]]
    clinical_note: str
    unsupervised_phenotype_report: Dict[str, Any]
    tabular_risk_shap_report: Dict[str, Any]
    vitals_trajectory_report: Dict[str, Any]
    chest_xray_gradcam_report: Dict[str, Any]
    clinical_nlp_report: Dict[str, Any]
    hybrid_rag_citations: List[Dict[str, Any]]
    composite_acuity_score: float
    esi_triage_level: str
    safety_guardrail_alerts: List[str]
    recommended_clinical_orders: List[str]
    hitl_clinician_status: str
    execution_trace: List[str]


def compute_synthesis_and_safety_guardrails(
    unsupervised_report: Dict[str, Any],
    tabular_report: Dict[str, Any],
    vitals_report: Dict[str, Any],
    xray_report: Dict[str, Any],
    nlp_report: Dict[str, Any],
    rag_citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes composite acuity score, ESI triage level, clinical safety alerts, and orders.

    Preserves the exact formula, thresholds, and rule guardrails from Notebook 06.
    """
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
    cited_ids = [c["doc_id"] for c in rag_citations]

    if unsupervised_report.get("is_outlier_flagged"):
        alerts.append(
            "ISOLATION FOREST OUTLIER: Extreme clinical/sensor profile detected; verify bedside vitals manually."
        )

    if "Penicillin" in nlp_report.get("documented_allergies", []) and p_vis >= 0.50:
        alerts.append(
            "ALLERGY GUARDRAIL: Documented Penicillin allergy — Beta-Lactam/Ceftriaxone orders blocked."
        )
        orders.append(
            f"Administer respiratory fluoroquinolone (Levofloxacin 750mg IV) or Aztreonam + Vancomycin (Grounded in {cited_ids})."
        )
    elif p_vis >= 0.50:
        orders.append(
            f"Initiate empiric CAP antimicrobial regimen and obtain sputum/blood cultures (Grounded in {cited_ids})."
        )

    if composite >= 0.50:
        orders.append(
            f"Continuous telemetry + hourly SpO2/MAP trend checks; repeat STAT serum lactate (Grounded in {cited_ids})."
        )
    else:
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
        "hitl_clinician_status": hitl_status,
    }


def build_langgraph_triage_workflow(
    unsup_model: SklearnUnsupervisedPhenotyper,
    tabular_model: LightGBMReadmissionExplainer,
    vitals_model: PyTorchVitalsTrainer,
    xray_model: TorchvisionChestXRayGradCAM,
    nlp_model: HuggingFaceClinicalNLP,
    rag_model: ProductionClinicalHybridRAG,
    interrupt_before_hitl: bool = True,
):
    """Builds and compiles the stateful LangGraph clinical triage workflow from Notebook 06."""

    @tool
    def analyze_patient_phenotype_and_readmission_risk(tabular_row_json: str) -> Dict[str, Any]:
        """Runs scikit-learn Phenotyper and LightGBM + TreeSHAP on patient tabular features."""
        row_df = pd.DataFrame([json.loads(tabular_row_json)])
        x_imp, unsup_rep = unsup_model.analyze_patient(row_df)
        risk_rep = tabular_model.predict_and_explain(x_imp, unsup_model.expected_features)
        return {"unsupervised": unsup_rep, "tabular_shap": risk_rep}

    @tool
    def forecast_icu_vitals_trajectory(vitals_24h_list: List[List[float]]) -> Dict[str, Any]:
        """Runs PyTorch Bi-LSTM + MultiheadAttention over 24-hour ICU vital streams."""
        arr = np.array(vitals_24h_list, dtype=np.float32)
        return vitals_model.forecast_trajectory(arr)

    @tool
    def analyze_chest_xray_and_clinical_note(
        xray_28x28_list: List[List[int]],
        clinical_note: str,
    ) -> Dict[str, Any]:
        """Runs Torchvision ResNet-18 + Grad-CAM on chest X-ray and HuggingFace BioNER on clinical note."""
        img = np.array(xray_28x28_list, dtype=np.uint8)
        vis_rep = xray_model.predict_with_gradcam(img)
        nlp_rep = nlp_model.extract_clinical_entities(clinical_note)
        return {"vision_gradcam": vis_rep, "clinical_nlp": nlp_rep}

    @tool
    def retrieve_clinical_evidence_rag(clinical_query: str) -> List[Dict[str, Any]]:
        """Searches ChromaDB (PubMedBERT) + BM25Okapi via LangChain and reranks biomedical evidence with CrossEncoder."""
        return rag_model.hybrid_search_with_rerank(clinical_query, top_k=3)

    def node_tabular_phenotyping_and_shap(state: TriageAgentState) -> Dict[str, Any]:
        res = analyze_patient_phenotype_and_readmission_risk.invoke(
            {"tabular_row_json": state["tabular_row_json"]}
        )
        trace = state.get("execution_trace", []) + [
            f"[Tabular & Phenotyping Node] Phenotype='{res['unsupervised']['phenotype_name']}' | "
            f"Readmission Risk={res['tabular_shap']['readmission_or_icu_risk_prob']:.1%}"
        ]
        return {
            "unsupervised_phenotype_report": res["unsupervised"],
            "tabular_risk_shap_report": res["tabular_shap"],
            "execution_trace": trace,
        }

    def node_temporal_vitals_dl(state: TriageAgentState) -> Dict[str, Any]:
        res = forecast_icu_vitals_trajectory.invoke({"vitals_24h_list": state["vitals_24h_list"]})
        trace = state["execution_trace"] + [
            f"[Vitals Bi-LSTM Node] 6h Deterioration={res['deterioration_6h_prob']:.1%} | "
            f"Peak Hours={res['peak_attention_hours']}"
        ]
        return {"vitals_trajectory_report": res, "execution_trace": trace}

    def node_multimodal_vision_and_nlp(state: TriageAgentState) -> Dict[str, Any]:
        res = analyze_chest_xray_and_clinical_note.invoke(
            {
                "xray_28x28_list": state["xray_28x28_list"],
                "clinical_note": state["clinical_note"],
            }
        )
        trace = state["execution_trace"] + [
            f"[Vision & BioNER Node] X-Ray Pneumonia={res['vision_gradcam']['pneumonia_probability']:.1%} | "
            f"Allergies={res['clinical_nlp']['documented_allergies']}"
        ]
        return {
            "chest_xray_gradcam_report": res["vision_gradcam"],
            "clinical_nlp_report": res["clinical_nlp"],
            "execution_trace": trace,
        }

    def node_hybrid_rag_retrieval(state: TriageAgentState) -> Dict[str, Any]:
        nlp_rep = state["clinical_nlp_report"]
        vis_rep = state["chest_xray_gradcam_report"]
        findings = " ".join(nlp_rep["affirmed_clinical_findings"][:4])
        meds = " ".join(nlp_rep["extracted_medications"][:3])
        allergies = " ".join(nlp_rep["documented_allergies"])
        query = f"clinical management {vis_rep['radiology_impression']} {findings} {meds} allergy {allergies}".strip()

        citations = retrieve_clinical_evidence_rag.invoke({"clinical_query": query})
        trace = state["execution_trace"] + [
            f"[Hybrid RAG Node] Retrieved & CrossEncoder-reranked {len(citations)} PubMedQA documents"
        ]
        return {"hybrid_rag_citations": citations, "execution_trace": trace}

    def node_synthesis_and_safety_guardrails(state: TriageAgentState) -> Dict[str, Any]:
        synthesis = compute_synthesis_and_safety_guardrails(
            unsupervised_report=state["unsupervised_phenotype_report"],
            tabular_report=state["tabular_risk_shap_report"],
            vitals_report=state["vitals_trajectory_report"],
            xray_report=state["chest_xray_gradcam_report"],
            nlp_report=state["clinical_nlp_report"],
            rag_citations=state["hybrid_rag_citations"],
        )
        trace = state["execution_trace"] + [
            f"[Guardrail Synthesis Node] Composite Risk={synthesis['composite_acuity_score']:.1%} "
            f"({synthesis['esi_triage_level']}) with {len(synthesis['safety_guardrail_alerts'])} Safety Alerts"
        ]
        return {**synthesis, "execution_trace": trace}

    def node_clinician_hitl_signoff(state: TriageAgentState) -> Dict[str, Any]:
        trace = state["execution_trace"] + [
            "[Clinician HITL Node] Attending Physician Sign-Off Completed -> Orders Released"
        ]
        return {
            "hitl_clinician_status": "APPROVED_BY_ATTENDING_PHYSICIAN",
            "execution_trace": trace,
        }

    builder = StateGraph(TriageAgentState)
    builder.add_node("tabular_ml_node", node_tabular_phenotyping_and_shap)
    builder.add_node("temporal_dl_node", node_temporal_vitals_dl)
    builder.add_node("vision_nlp_node", node_multimodal_vision_and_nlp)
    builder.add_node("hybrid_rag_node", node_hybrid_rag_retrieval)
    builder.add_node("guardrail_synthesis_node", node_synthesis_and_safety_guardrails)
    builder.add_node("clinician_hitl_signoff", node_clinician_hitl_signoff)

    builder.add_edge(START, "tabular_ml_node")
    builder.add_edge("tabular_ml_node", "temporal_dl_node")
    builder.add_edge("temporal_dl_node", "vision_nlp_node")
    builder.add_edge("vision_nlp_node", "hybrid_rag_node")
    builder.add_edge("hybrid_rag_node", "guardrail_synthesis_node")
    builder.add_edge("guardrail_synthesis_node", "clinician_hitl_signoff")
    builder.add_edge("clinician_hitl_signoff", END)

    memory = MemorySaver()
    interrupt_nodes = ["clinician_hitl_signoff"] if interrupt_before_hitl else []
    return builder.compile(checkpointer=memory, interrupt_before=interrupt_nodes)
