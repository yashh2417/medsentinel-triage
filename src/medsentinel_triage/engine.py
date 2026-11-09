"""Central inference orchestrator and CLI entry point for the MedSentinel Clinical Triage platform."""

from __future__ import annotations

import os

# Prevent macOS OpenMP collision between PyTorch and SHAP TreeExplainer
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from medsentinel_triage.base import InputValidationError
from medsentinel_triage.config import InferenceConfig, configure_logging
from medsentinel_triage.input_processor import (
    process_batch_csv_files,
    process_batch_json_file,
)
from medsentinel_triage.loader import LoadedModelBundle, ModelLoader
from medsentinel_triage.preprocessing import (
    TabularInputType,
    VitalsInputType,
    XRayInputType,
    validate_and_preprocess_tabular_row,
    validate_and_preprocess_vitals,
    validate_non_empty_text,
    validate_xray_image,
)
from medsentinel_triage.workflow import (
    TriageAgentState,
    build_langgraph_triage_workflow,
)

logger = logging.getLogger("medsentinel_triage.engine")


class ClinicalTriageInferenceEngine:
    """Central inference orchestrator unifying individual model predictions, batch inference, and LangGraph execution."""

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
        """Loads all pre-trained models from `trained_models/` and constructs the central inference engine."""
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
            "extracted_rag_evidence": [],
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

    def run_batch_triage(
        self,
        patient_payloads: Sequence[Dict[str, Any]],
        auto_approve_hitl: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[TriageAgentState]:
        """Runs multimodal clinical triage across a list of validated patient payload dictionaries."""
        records = [validate_patient_payload_dict(p, idx=i) for i, p in enumerate(patient_payloads)]
        results: List[TriageAgentState] = []
        for idx, payload in enumerate(records):
            pid = str(payload["patient_id"])
            thread_id = f"batch-triage-{pid}-{idx}"
            logger.info("Running batch triage [%d/%d] for patient_id=%s", idx + 1, len(records), pid)
            state = self.run_full_triage(
                patient_id=pid,
                tabular_input=payload["tabular_input"],
                vitals_24h=payload["vitals_24h"],
                xray_28x28=payload["xray_28x28"],
                clinical_note=payload["clinical_note"],
                thread_id=thread_id,
                auto_approve_hitl=auto_approve_hitl,
            )
            results.append(state)

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved %d batch triage results to %s", len(results), out_file)

        return results


def validate_patient_payload_dict(payload: Any, idx: Optional[int] = None) -> Dict[str, Any]:
    """Validates that a patient payload dict contains all required multimodal keys."""
    required_keys = {"patient_id", "tabular_input", "vitals_24h", "xray_28x28", "clinical_note"}
    prefix = f"Patient payload at index {idx}" if idx is not None else "Patient JSON payload"
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise InputValidationError(
            f"{prefix} must contain keys {sorted(required_keys)}."
        )
    return payload


def load_patient_payload(input_json_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a single patient multimodal payload from a JSON file."""
    path = Path(input_json_path)
    if not path.exists():
        raise InputValidationError(f"Patient input JSON file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return validate_patient_payload_dict(payload)


def load_batch_patient_payloads(
    manifest_or_json_path: Union[str, Path],
    vitals_csv_path: Optional[Union[str, Path]] = None,
    pad_method: str = "carry_first",
    max_rows: int = 100,
) -> List[Dict[str, Any]]:
    """Loads batch patient payloads from either two CSV files (`manifest_csv` + `vitals_csv`) or a `.json` file."""
    path = Path(manifest_or_json_path)
    if not path.exists():
        raise InputValidationError(f"Batch input file not found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return process_batch_json_file(path, pad_method=pad_method, max_rows=max_rows)
    if suffix == ".csv":
        if vitals_csv_path is None:
            raise InputValidationError(
                "Batch CSV processing requires both 'batch_tabular_manifest.csv' and 'batch_vitals.csv'."
            )
        return process_batch_csv_files(
            manifest_csv_path=path,
            vitals_csv_path=vitals_csv_path,
            pad_method=pad_method,
            max_rows=max_rows,
        )
    raise InputValidationError(
        f"Unsupported batch input format '{suffix}'. Only two CSV files (.csv) or a JSON file (.json) are supported."
    )


def format_triage_summary(paused_state: TriageAgentState, final_state: Optional[TriageAgentState] = None) -> str:
    """Formats a human-readable clinical triage report including extracted QA evidence."""
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
            f"Peak Hours={paused_state['vitals_trajectory_report']['peak_attention_hours']} | "
            f"Projected 6h={paused_state['vitals_trajectory_report']['projected_vitals_6h']}"
        ),
        (
            f"ResNet-18 Grad-CAM    : {paused_state['chest_xray_gradcam_report']['radiology_impression']} "
            f"(p={paused_state['chest_xray_gradcam_report']['pneumonia_probability']:.1%})"
        ),
        (
            f"Biomedical NER        : Findings={paused_state['clinical_nlp_report']['affirmed_clinical_findings'][:5]} | "
            f"Meds={paused_state['clinical_nlp_report']['extracted_medications']} | "
            f"Allergies={paused_state['clinical_nlp_report']['documented_allergies']}"
        ),
    ]
    for ev in paused_state.get("extracted_rag_evidence", []):
        lines.append(
            f"  [?] QA EVIDENCE [{ev['doc_id']}] (score={ev['cross_encoder_score']}): {ev['title']} -> {ev['extracted_conclusion']}"
        )
    for alert in paused_state["safety_guardrail_alerts"]:
        lines.append(f"  [!] SAFETY ALERT    : {alert}")
    for order in paused_state["recommended_clinical_orders"]:
        lines.append(f"  [+] PROPOSED ORDER  : {order}")
    if final_state is not None:
        lines.append(f"\n[LangGraph Resumed] Final HITL Status: {final_state['hitl_clinician_status']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point to run single-patient or batch multimodal clinical triage inference."""
    parser = argparse.ArgumentParser(
        description="Run the MedSentinel Multimodal Clinical Triage Inference Engine."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Path to a single patient multimodal input JSON file.",
    )
    parser.add_argument(
        "--batch-manifest-csv",
        type=str,
        default=None,
        help="Path to batch_tabular_manifest.csv (requires --batch-vitals-csv).",
    )
    parser.add_argument(
        "--batch-vitals-csv",
        type=str,
        default=None,
        help="Path to batch_vitals.csv (used with --batch-manifest-csv).",
    )
    parser.add_argument(
        "--batch-json",
        type=str,
        default=None,
        help="Path to a batch patient JSON file (.json).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save output predictions (.json).",
    )
    parser.add_argument(
        "--rag-corpus-file",
        type=str,
        default=None,
        help="Optional path to custom RAG corpus file when building/reindexing the vector store.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to directory containing saved model checkpoints (defaults to ./trained_models).",
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
        help="Output the final TriageAgentState(s) as formatted JSON to stdout.",
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

    if args.batch_manifest_csv or args.batch_json:
        if args.batch_manifest_csv:
            if not args.batch_vitals_csv:
                raise InputValidationError("--batch-vitals-csv is required when using --batch-manifest-csv.")
            payloads = load_batch_patient_payloads(args.batch_manifest_csv, args.batch_vitals_csv)
        else:
            payloads = load_batch_patient_payloads(args.batch_json)

        batch_states = engine.run_batch_triage(
            patient_payloads=payloads,
            auto_approve_hitl=True,
            output_path=args.output_file,
        )
        if args.json:
            print(json.dumps(batch_states, indent=2))
        else:
            for state in batch_states:
                print(format_triage_summary(state, state))
    elif args.input_json:
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

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(final_state, f, indent=2)

        if args.json:
            print(json.dumps(final_state, indent=2))
        else:
            print(format_triage_summary(paused_state, final_state))
    else:
        raise InputValidationError(
            "Provide either --input-json, (--batch-manifest-csv and --batch-vitals-csv), or --batch-json."
        )


if __name__ == "__main__":
    main()
