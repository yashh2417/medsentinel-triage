"""FastAPI router for individual model predictions, full LangGraph triage execution, and HITL approval."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import torch
import torch.nn.functional as F

from medsentinel_triage.api.routes_preprocess import prepare_patient_endpoint
from medsentinel_triage.api.schemas import (
    HITLApprovalRequest,
    NLPPredictRequest,
    RAGPredictRequest,
    TabularPredictRequest,
    TriageRunRequest,
    VitalsPredictRequest,
    XRayPredictRequest,
)
from medsentinel_triage.config import InferenceConfig, XRAY_RAW_SIZE
from medsentinel_triage.engine import ClinicalTriageInferenceEngine, format_triage_summary

logger = logging.getLogger("medsentinel_triage.api.inference")

router = APIRouter(prefix="/api", tags=["Model Inference & LangGraph Triage"])


class EngineSingleton:
    """Thread-safe singleton managing the loaded `ClinicalTriageInferenceEngine` instance."""

    _lock = threading.Lock()
    _engine: Optional[ClinicalTriageInferenceEngine] = None
    _config: InferenceConfig = InferenceConfig()

    @classmethod
    def get_engine(cls) -> ClinicalTriageInferenceEngine:
        with cls._lock:
            if cls._engine is None:
                logger.info("Initializing ClinicalTriageInferenceEngine from %s...", cls._config.models_dir)
                cls._engine = ClinicalTriageInferenceEngine.from_config(config=cls._config)
            return cls._engine

    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        cfg = cls._config
        checkpoints = {
            "unsupervised_phenotyper": cfg.phenotyper_path.exists(),
            "lightgbm_readmission": cfg.lightgbm_path.exists(),
            "vitals_bilstm_attention": cfg.vitals_path.exists(),
            "chest_xray_resnet18_gradcam": cfg.xray_path.exists(),
            "biomedical_ner_model": cfg.ner_model_path.exists(),
            "rag_vector_store": cfg.rag_persist_dir.exists() and cfg.rag_bm25_cache_path.exists(),
        }
        return {
            "status": "healthy" if all(checkpoints.values()) else "degraded",
            "device": str(cfg.device),
            "models_dir": str(cfg.models_dir),
            "engine_initialized": cls._engine is not None,
            "checkpoints_ready": checkpoints,
        }


def compute_xray_gradcam_heatmap(
    engine: ClinicalTriageInferenceEngine,
    xray_28x28: Any,
) -> Dict[str, Any]:
    """Runs `TorchvisionChestXRayGradCAM` and includes the 28x28 normalized Grad-CAM heatmap grid."""
    xray_model = engine.xray_model
    img_arr = xray_model.preprocess(xray_28x28)
    vis_report = xray_model.predict_with_gradcam(img_arr)

    heatmap_28x28 = [[0.0] * 28 for _ in range(28)]
    if xray_model._gradients is not None and xray_model._feature_maps is not None:
        grads = xray_model._gradients[0]
        fmaps = xray_model._feature_maps[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu(torch.sum(weights[:, None, None] * fmaps, dim=0))
        cam_up = F.interpolate(
            cam[None, None, ...],
            size=XRAY_RAW_SIZE,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam_np = cam_up.detach().cpu().numpy()
        if cam_np.max() > 1e-8:
            cam_np = cam_np / cam_np.max()
        heatmap_28x28 = [[round(float(val), 4) for val in row] for row in cam_np]

    return {**vis_report, "gradcam_heatmap_28x28": heatmap_28x28}


@router.get("/health")
def health_endpoint() -> Dict[str, Any]:
    """Returns system health, active PyTorch device, and model checkpoint status."""
    return EngineSingleton.get_health_status()


@router.post("/predict/tabular")
def predict_tabular_endpoint(req: TabularPredictRequest) -> Dict[str, Any]:
    """Runs `SklearnUnsupervisedPhenotyper` and `LightGBMReadmissionExplainer` (`TreeSHAP`)."""
    engine = EngineSingleton.get_engine()
    res = engine.predict_phenotype_and_readmission(req.tabular_input)
    return {
        "status": "ok",
        "unsupervised_phenotype_report": res["unsupervised"],
        "tabular_risk_shap_report": res["tabular_shap"],
    }


@router.post("/predict/vitals")
def predict_vitals_endpoint(req: VitalsPredictRequest) -> Dict[str, Any]:
    """Runs `PyTorchVitalsTrainer` (`VitalSignsBiLSTMAttentionNet`) on 24h ICU vitals."""
    engine = EngineSingleton.get_engine()
    return {
        "status": "ok",
        "vitals_trajectory_report": engine.predict_vitals_trajectory(req.vitals_24h),
    }


@router.post("/predict/xray")
def predict_xray_endpoint(req: XRayPredictRequest) -> Dict[str, Any]:
    """Runs `TorchvisionChestXRayGradCAM` (`ResNet-18` + `Grad-CAM`)."""
    engine = EngineSingleton.get_engine()
    return {
        "status": "ok",
        "chest_xray_gradcam_report": compute_xray_gradcam_heatmap(engine, req.xray_28x28),
    }


@router.post("/predict/nlp")
def predict_nlp_endpoint(req: NLPPredictRequest) -> Dict[str, Any]:
    """Runs `HuggingFaceClinicalNLP` (`BioNER` + `NegEx`)."""
    engine = EngineSingleton.get_engine()
    return {
        "status": "ok",
        "clinical_nlp_report": engine.extract_clinical_entities(req.clinical_note),
    }


@router.post("/predict/rag")
def predict_rag_endpoint(req: RAGPredictRequest) -> Dict[str, Any]:
    """Runs `ProductionClinicalHybridRAG` (`EnsembleRetriever` + `CrossEncoder`)."""
    engine = EngineSingleton.get_engine()
    return {
        "status": "ok",
        "query": req.clinical_query,
        "hybrid_rag_citations": engine.retrieve_clinical_evidence(req.clinical_query, top_k=req.top_k),
    }


@router.post("/triage/run")
def run_triage_endpoint(req: TriageRunRequest) -> Dict[str, Any]:
    """Executes the full multimodal LangGraph triage workflow on a prepared patient payload."""
    engine = EngineSingleton.get_engine()
    thread_id = req.thread_id or f"triage-{req.patient_id}-{int(time.time() * 1000)}"

    start_ts = time.perf_counter()
    paused_state = engine.run_triage_until_hitl(
        patient_id=req.patient_id,
        tabular_input=req.tabular_input,
        vitals_24h=req.vitals_24h,
        xray_28x28=req.xray_28x28,
        clinical_note=req.clinical_note,
        thread_id=thread_id,
    )

    xray_with_heatmap = compute_xray_gradcam_heatmap(engine, req.xray_28x28)
    paused_state["chest_xray_gradcam_report"]["gradcam_heatmap_28x28"] = xray_with_heatmap[
        "gradcam_heatmap_28x28"
    ]

    final_state: Optional[Dict[str, Any]] = None
    if req.auto_approve_hitl:
        final_state = dict(engine.approve_hitl(thread_id=thread_id))
        final_state["chest_xray_gradcam_report"]["gradcam_heatmap_28x28"] = xray_with_heatmap[
            "gradcam_heatmap_28x28"
        ]

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)
    active_state = final_state if final_state is not None else paused_state

    return {
        "status": "ok",
        "thread_id": thread_id,
        "elapsed_ms": elapsed_ms,
        "state": active_state,
        "formatted_summary": format_triage_summary(paused_state, final_state),
    }


@router.post("/triage/upload-and-run")
def upload_and_run_triage_endpoint(
    tabular_file: UploadFile = File(...),
    vitals_file: UploadFile = File(...),
    xray_file: UploadFile = File(...),
    note_file: Optional[UploadFile] = File(None),
    clinical_note: Optional[str] = Form(None),
    patient_id: str = Form("PATIENT-001"),
    pad_method: str = Form("carry_first"),
    row_index: int = Form(0),
    auto_approve_hitl: bool = Form(False),
) -> Dict[str, Any]:
    """Uploads raw CSV/PNG/TXT files, prepares the patient payload, and runs LangGraph triage."""
    prep_result = prepare_patient_endpoint(
        tabular_file=tabular_file,
        vitals_file=vitals_file,
        xray_file=xray_file,
        note_file=note_file,
        clinical_note=clinical_note,
        patient_id=patient_id,
        pad_method=pad_method,
        row_index=row_index,
    )
    payload = prep_result["payload"]
    triage_req = TriageRunRequest(**payload, auto_approve_hitl=auto_approve_hitl)
    triage_res = run_triage_endpoint(triage_req)
    return {
        **triage_res,
        "uploaded_filenames": prep_result["uploaded_filenames"],
        "prepared_payload": payload,
    }


@router.post("/triage/approve-hitl")
def approve_hitl_endpoint(req: HITLApprovalRequest) -> Dict[str, Any]:
    """Resumes the paused LangGraph workflow after Attending Physician sign-off."""
    engine = EngineSingleton.get_engine()
    if req.action == "approve":
        final_state = dict(engine.approve_hitl(thread_id=req.thread_id))
        if req.clinician_notes:
            final_state["execution_trace"] = list(final_state.get("execution_trace", [])) + [
                f"[Clinician Addendum ({req.clinician_name})] {req.clinician_notes}"
            ]
        return {
            "status": "ok",
            "thread_id": req.thread_id,
            "hitl_clinician_status": final_state["hitl_clinician_status"],
            "clinician_name": req.clinician_name,
            "clinician_notes": req.clinician_notes,
            "state": final_state,
            "formatted_summary": format_triage_summary(final_state, final_state),
        }
    return {
        "status": "ok",
        "thread_id": req.thread_id,
        "hitl_clinician_status": "HOLD_BY_ATTENDING_PHYSICIAN",
        "clinician_name": req.clinician_name,
        "clinician_notes": req.clinician_notes,
    }
