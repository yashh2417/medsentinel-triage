"""Centralized configuration, model paths, clinical constants, and device selection."""

from __future__ import annotations

import logging
import os

# Prevent macOS OpenMP / libomp multi-runtime collision between PyTorch and SHAP TreeExplainer
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

# Project root and default models directory
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR: Path = Path(os.getenv("HEALTHCARE_MODELS_DIR", PROJECT_ROOT / "models"))

# Exact feature ordering preserved from training pipelines
TABULAR_FEATURE_NAMES: List[str] = [
    "age_years",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Cluster phenotype labels from training (ordered by ascending mean acuity score)
DEFAULT_CLUSTER_LABELS_MAP: Dict[int, str] = {
    0: "Phenotype 1: Low-Utilization Baseline Cohort",
    2: "Phenotype 2: Moderate Chronic Multi-Medication Cohort",
    1: "Phenotype 3: High-Acuity Frequent-Inpatient Cohort",
}

# ICU vital sign channel ordering and MIMIC-IV itemid mapping from training
# Channels: [HR, SpO2, SBP, RespRate, TempC]
VITAL_CHANNEL_NAMES: List[str] = [
    "heart_rate_bpm",
    "spo2_pct",
    "systolic_bp_mmhg",
    "resp_rate_bpm",
    "temp_c",
]
MIMIC_VITAL_ITEMIDS: Dict[int, int] = {
    220045: 0,  # Heart Rate (bpm)
    220277: 1,  # SpO2 (%)
    220179: 2,  # Non-Invasive Blood Pressure systolic (mmHg)
    220210: 3,  # Respiratory Rate (insp/min)
    223761: 4,  # Temperature Fahrenheit (converted to Celsius during preprocessing)
}
DEFAULT_ICU_VITALS: Tuple[float, float, float, float, float] = (82.0, 97.0, 120.0, 17.0, 37.0)
ICU_SEQUENCE_LENGTH: int = 24
ICU_NUM_VITAL_CHANNELS: int = 5

# Chest X-Ray ResNet-18 Grad-CAM constants from training
XRAY_RAW_SIZE: Tuple[int, int] = (28, 28)
XRAY_MODEL_INPUT_SIZE: Tuple[int, int] = (64, 64)
XRAY_PNEUMONIA_THRESHOLD: float = 0.50

# Clinical NER & NegEx constants from training
NEGATION_PATTERNS: Tuple[str, ...] = (
    "denies",
    "denied",
    "no evidence of",
    "without",
    "negative for",
    "unremarkable for",
)
DOCUMENTED_ALLERGY_KEYWORDS: Tuple[str, ...] = (
    "penicillin",
    "sulfa",
    "cephalosporin",
    "aspirin",
    "vancomycin",
)
NEGATION_WINDOW_CHARS: int = 45


def resolve_torch_device(preferred_device: Optional[str] = None) -> torch.device:
    """Selects the optimal PyTorch device (CUDA -> MPS -> CPU) or honors an explicit override."""
    if preferred_device:
        normalized = preferred_device.strip().lower()
        if normalized == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if normalized == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if normalized == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_hf_pipeline_device(device: torch.device) -> Union[int, str]:
    """Maps a torch.device to the device identifier expected by Hugging Face `pipeline`."""
    if device.type == "cuda":
        return 0
    if device.type == "mps":
        return "mps"
    return -1


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration container for model checkpoints, device settings, and clinical thresholds."""

    models_dir: Path = DEFAULT_MODELS_DIR
    preferred_device: Optional[str] = None

    # Model checkpoint filenames / Hugging Face identifiers
    phenotyper_checkpoint_name: str = "healthcare_unsupervised_phenotyper.pkl"
    lightgbm_checkpoint_name: str = "lightgbm_model.txt"
    vitals_checkpoint_name: str = "vitals_bilstm_attention.pth"
    xray_checkpoint_name: str = "pneumonia_resnet18_gradcam.pth"
    ner_model_dir_name: str = "biomedical_ner_model"
    ner_fallback_hub_id: str = "d4data/biomedical-ner-all"

    # Hybrid RAG models & persistent vector/lexical index settings
    rag_embed_model_name: str = "NeuML/pubmedbert-base-embeddings"
    rag_reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_collection_name: str = "pubmedqa_clinical_guidelines"
    rag_persist_dir_name: str = "rag_vector_store"
    rag_bm25_cache_name: str = "rag_bm25_corpus.pkl"
    rag_use_persistent_store: bool = True
    rag_top_k: int = 3
    rag_dense_candidates: int = 10
    rag_bm25_candidates: int = 10
    rag_rrf_candidates: int = 8
    rag_rrf_k: float = 60.0

    # Tabular readmission risk thresholds
    readmission_moderate_threshold: float = 0.25
    readmission_high_threshold: float = 0.40
    shap_top_k: int = 5

    # Composite triage acuity weights & ESI thresholds
    weight_tabular_risk: float = 0.40
    weight_vitals_risk: float = 0.35
    weight_vision_risk: float = 0.25
    esi_level_1_threshold: float = 0.60
    esi_level_2_threshold: float = 0.35
    high_acuity_order_threshold: float = 0.50

    tabular_feature_names: List[str] = field(default_factory=lambda: list(TABULAR_FEATURE_NAMES))

    @property
    def device(self) -> torch.device:
        return resolve_torch_device(self.preferred_device)

    @property
    def hf_device(self) -> Union[int, str]:
        return resolve_hf_pipeline_device(self.device)

    @property
    def phenotyper_path(self) -> Path:
        return self.models_dir / self.phenotyper_checkpoint_name

    @property
    def lightgbm_path(self) -> Path:
        return self.models_dir / self.lightgbm_checkpoint_name

    @property
    def vitals_path(self) -> Path:
        return self.models_dir / self.vitals_checkpoint_name

    @property
    def xray_path(self) -> Path:
        return self.models_dir / self.xray_checkpoint_name

    @property
    def ner_model_path(self) -> Path:
        return self.models_dir / self.ner_model_dir_name

    @property
    def rag_persist_dir(self) -> Path:
        return self.models_dir / self.rag_persist_dir_name

    @property
    def rag_bm25_cache_path(self) -> Path:
        return self.rag_persist_dir / self.rag_bm25_cache_name


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures structured package-level logging."""
    logger = logging.getLogger("a_project")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
