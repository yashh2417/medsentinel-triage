"""Central inference orchestrator and CLI entry point for the Healthcare AI Clinical Triage platform."""

from __future__ import annotations

import os

# Prevent macOS OpenMP collision between PyTorch and SHAP TreeExplainer
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from a_project.base import InputValidationError
from a_project.config import InferenceConfig, configure_logging
from a_project.loader import LoadedModelBundle, ModelLoader
from a_project.preprocessing import (
    TabularInputType,
    VitalsInputType,
    XRayInputType,
    validate_and_preprocess_tabular_row,
    validate_and_preprocess_vitals,
    validate_non_empty_text,
    validate_xray_image,
)
from a_project.workflow import (
    TriageAgentState,
    build_langgraph_triage_workflow,
)

logger = logging.getLogger("a_project.engine")


class ClinicalTriageInferenceEngine:
    """Central inference orchestrator unifying individual model predictions and LangGraph execution."""

    def __init__(
        self,
        bundle: LoadedModelBundle,
        interrupt_before_hitl: bool = True,
    ) -> None:
        self.bundle = bundle
        self.config = bundle.config
        self.unsup_model = bundle.unsup_model
        self.tabular_model = bundle.tabular_model
        self.vitals_model = bundle.vitals_model
        self.xray_model = bundle.xray_model
        self.nlp_model = bundle.nlp_model
        self.rag_model = bundle.rag_model
        self.langgraph_app = build_langgraph_triage_workflow(
            unsup_model=self.unsup_model,
            tabular_model=self.tabular_model,
            vitals_model=self.vitals_model,
            xray_model=self.xray_model,
            nlp_model=self.nlp_model,
            rag_model=self.rag_model,
            interrupt_before_hitl=interrupt_before_hitl,
        )

    @classmethod
    def from_config(
        cls,
        config: Optional[InferenceConfig] = None,
        corpus: Optional[List[Dict[str, str]]] = None,
        corpus_path: Optional[Union[str, Path]] = None,
        interrupt_before_hitl: bool = True,
        force_reindex_rag: bool = False,
    ) -> "ClinicalTriageInferenceEngine":
        """Loads all pre-trained models from disk and constructs the central inference engine."""
        loader = ModelLoader(config=config)
        bundle = loader.load_all(
            corpus=corpus,
            corpus_path=corpus_path,
            force_reindex_rag=force_reindex_rag,
        )
        return cls(bundle=bundle, interrupt_before_hitl=interrupt_before_hitl)

    def predict_phenotype_and_readmission(
        self,
        tabular_input: TabularInputType,
    ) -> Dict[str, Any]:
        """Runs Unsupervised Phenotyping + LightGBM Readmission Risk & TreeSHAP on patient tabular data."""
        row_df = validate_and_preprocess_tabular_row(
            tabular_input,
            feature_names=self.unsup_model.expected_features,
        )
        x_imp, unsup_rep = self.unsup_model.run_inference(row_df)
        risk_rep = self.tabular_model.run_inference(x_imp)
        return {"unsupervised": unsup_rep, "tabular_shap": risk_rep}

    def predict_vitals_trajectory(self, vitals_24h: VitalsInputType) -> Dict[str, Any]:
        """Runs PyTorch 1D-CNN + Bi-LSTM + MultiheadAttention over 24-hour ICU vital streams."""
        return self.vitals_model.run_inference(vitals_24h)

    def predict_chest_xray(self, xray_28x28: XRayInputType) -> Dict[str, Any]:
        """Runs Torchvision ResNet-18 + autograd Grad-CAM on a 28x28 Chest X-Ray image."""
        return self.xray_model.run_inference(xray_28x28)

    def extract_clinical_entities(self, clinical_note: str) -> Dict[str, Any]:
        """Runs Hugging Face Biomedical NER + NegEx negation/allergy extraction on a clinical note."""
        return self.nlp_model.run_inference(clinical_note)

    def retrieve_clinical_evidence(
        self,
        clinical_query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Runs LangChain Hybrid RAG (`EnsembleRetriever` + `CrossEncoder`)."""
        self.rag_model.validate_input(clinical_query)
        return self.rag_model.hybrid_search_with_rerank(clinical_query, top_k=top_k)

    def build_initial_state(
        self,
        patient_id: str,
        tabular_input: TabularInputType,
        vitals_24h: VitalsInputType,
        xray_28x28: XRayInputType,
        clinical_note: str,
    ) -> TriageAgentState:
        """Validates all multimodal inputs and constructs the `TriageAgentState` dictionary."""
        ordered_df = validate_and_preprocess_tabular_row(
            tabular_input,
            feature_names=self.unsup_model.expected_features,
        )
        vitals_arr = validate_and_preprocess_vitals(vitals_24h)
        xray_arr = validate_xray_image(xray_28x28)
        cleaned_note = validate_non_empty_text(clinical_note, field_name="clinical_note")

        tabular_json = ordered_df.iloc[0].to_json()
        return {
            "patient_id": str(patient_id),
            "tabular_row_json": tabular_json,
            "vitals_24h_list": vitals_arr.tolist(),
            "xray_28x28_list": xray_arr.tolist(),
            "clinical_note": cleaned_note,
            "unsupervised_phenotype_report": {},
            "tabular_risk_shap_report": {},
            "vitals_trajectory_report": {},
            "chest_xray_gradcam_report": {},
            "clinical_nlp_report": {},
            "hybrid_rag_citations": [],
            "composite_acuity_score": 0.0,
            "esi_triage_level": "",
            "safety_guardrail_alerts": [],
            "recommended_clinical_orders": [],
            "hitl_clinician_status": "",
            "execution_trace": [],
        }

    def run_triage_until_hitl(
        self,
        patient_id: str,
        tabular_input: TabularInputType,
        vitals_24h: VitalsInputType,
        xray_28x28: XRayInputType,
        clinical_note: str,
        thread_id: str = "clinical-triage-thread-1",
    ) -> TriageAgentState:
        """Executes the LangGraph workflow up to the `clinician_hitl_signoff` checkpoint."""
        initial_state = self.build_initial_state(
            patient_id=patient_id,
            tabular_input=tabular_input,
            vitals_24h=vitals_24h,
            xray_28x28=xray_28x28,
            clinical_note=clinical_note,
        )
        thread_cfg = {"configurable": {"thread_id": thread_id}}
        paused_state: TriageAgentState = self.langgraph_app.invoke(initial_state, config=thread_cfg)
        return paused_state

    def approve_hitl(self, thread_id: str = "clinical-triage-thread-1") -> TriageAgentState:
        """Resumes the LangGraph workflow from the HITL checkpoint after clinician approval."""
        thread_cfg = {"configurable": {"thread_id": thread_id}}
        final_state: TriageAgentState = self.langgraph_app.invoke(None, config=thread_cfg)
        return final_state

    def run_full_triage(
        self,
        patient_id: str,
        tabular_input: TabularInputType,
        vitals_24h: VitalsInputType,
        xray_28x28: XRayInputType,
        clinical_note: str,
        thread_id: str = "clinical-triage-thread-1",
        auto_approve_hitl: bool = True,
    ) -> TriageAgentState:
        """Runs the complete multimodal clinical triage pipeline (with optional HITL auto-resume)."""
        state = self.run_triage_until_hitl(
            patient_id=patient_id,
            tabular_input=tabular_input,
            vitals_24h=vitals_24h,
            xray_28x28=xray_28x28,
            clinical_note=clinical_note,
            thread_id=thread_id,
        )
        if auto_approve_hitl:
            state = self.approve_hitl(thread_id=thread_id)
        return state


def load_patient_payload(input_json_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a patient multimodal payload from a JSON file.

    Expected keys in JSON file:
    - `patient_id` (str)
    - `tabular_input` (dict of patient tabular features)
    - `vitals_24h` (24x5 list of floats)
    - `xray_28x28` (28x28 list of uint8 ints)
    - `clinical_note` (str)
    """
    path = Path(input_json_path)
    if not path.exists():
        raise InputValidationError(f"Patient input JSON file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required_keys = {"patient_id", "tabular_input", "vitals_24h", "xray_28x28", "clinical_note"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise InputValidationError(
            f"Patient JSON payload must contain keys {sorted(required_keys)}."
        )
    return payload


def format_triage_summary(paused_state: TriageAgentState, final_state: Optional[TriageAgentState] = None) -> str:
    """Formats a human-readable clinical triage report."""
    lines = [
        "=" * 88,
        f"[LangGraph HITL Pause] Patient: {paused_state['patient_id']} | Status: {paused_state['hitl_clinician_status']}",
        f"Acuity Classification : {paused_state['esi_triage_level']} (Composite Risk: {paused_state['composite_acuity_score']:.1%})",
        f"Patient Phenotype     : {paused_state['unsupervised_phenotype_report']['phenotype_name']}",
        (
            f"LightGBM + TreeSHAP   : Risk={paused_state['tabular_risk_shap_report']['readmission_or_icu_risk_prob']:.1%} | "
            f"Top SHAP={paused_state['tabular_risk_shap_report']['top_treeshap_drivers'][:3]}"
        ),
        (
            f"PyTorch Bi-LSTM Vitals: 6h Deterioration={paused_state['vitals_trajectory_report']['deterioration_6h_prob']:.1%} | "
            f"Peak Hours={paused_state['vitals_trajectory_report']['peak_attention_hours']}"
        ),
        (
            f"ResNet-18 Grad-CAM    : {paused_state['chest_xray_gradcam_report']['radiology_impression']} "
            f"(p={paused_state['chest_xray_gradcam_report']['pneumonia_probability']:.1%})"
        ),
        (
            f"Biomedical NER        : Findings={paused_state['clinical_nlp_report']['affirmed_clinical_findings'][:5]} | "
            f"Allergies={paused_state['clinical_nlp_report']['documented_allergies']}"
        ),
    ]
    for alert in paused_state["safety_guardrail_alerts"]:
        lines.append(f"  [!] SAFETY ALERT    : {alert}")
    for order in paused_state["recommended_clinical_orders"]:
        lines.append(f"  [+] PROPOSED ORDER  : {order}")
    if final_state is not None:
        lines.append(f"\n[LangGraph Resumed] Final HITL Status: {final_state['hitl_clinician_status']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point to run multimodal clinical triage inference on user-supplied data."""
    parser = argparse.ArgumentParser(
        description="Run the Healthcare AI Multimodal Clinical Triage Inference Engine."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to patient multimodal input JSON file (`patient_id`, `tabular_input`, `vitals_24h`, `xray_28x28`, `clinical_note`).",
    )
    parser.add_argument(
        "--rag-corpus-file",
        type=str,
        default=None,
        help="Optional path to custom RAG corpus file (.json, .jsonl, or .csv) when building/reindexing the vector store.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to directory containing saved model checkpoints (defaults to ./models).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Preferred computation device override (cuda, mps, or cpu).",
    )
    parser.add_argument(
        "--reindex-rag",
        action="store_true",
        help="Force rebuild of the persistent ChromaDB vector store and BM25 index.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the final TriageAgentState as formatted JSON.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    cfg_kwargs: Dict[str, Any] = {}
    if args.models_dir:
        cfg_kwargs["models_dir"] = Path(args.models_dir)
    if args.device:
        cfg_kwargs["preferred_device"] = args.device

    config = InferenceConfig(**cfg_kwargs)
    engine = ClinicalTriageInferenceEngine.from_config(
        config=config,
        corpus_path=args.rag_corpus_file,
        force_reindex_rag=args.reindex_rag,
    )

    patient_payload = load_patient_payload(args.input_json)
    thread_id = f"clinical-triage-{patient_payload['patient_id']}"
    paused_state = engine.run_triage_until_hitl(
        patient_id=patient_payload["patient_id"],
        tabular_input=patient_payload["tabular_input"],
        vitals_24h=patient_payload["vitals_24h"],
        xray_28x28=patient_payload["xray_28x28"],
        clinical_note=patient_payload["clinical_note"],
        thread_id=thread_id,
    )
    final_state = engine.approve_hitl(thread_id=thread_id)

    if args.json:
        print(json.dumps(final_state, indent=2))
    else:
        print(format_triage_summary(paused_state, final_state))


if __name__ == "__main__":
    main()
