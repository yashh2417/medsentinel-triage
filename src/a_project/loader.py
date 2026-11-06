"""Dedicated model and checkpoint loader for all pre-trained healthcare AI models."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import torch

from a_project.base import InputValidationError, ModelLoadError
from a_project.config import InferenceConfig
from a_project.models.chest_xray import TorchvisionChestXRayGradCAM
from a_project.models.clinical_nlp import HuggingFaceClinicalNLP
from a_project.models.phenotyper import SklearnUnsupervisedPhenotyper
from a_project.rag import ProductionClinicalHybridRAG
from a_project.models.readmission import LightGBMReadmissionExplainer
from a_project.models.vitals import PyTorchVitalsTrainer

logger = logging.getLogger("a_project.loader")


@dataclass
class LoadedModelBundle:
    """Container holding all initialized and checkpoint-restored inference models."""

    unsup_model: SklearnUnsupervisedPhenotyper
    tabular_model: LightGBMReadmissionExplainer
    vitals_model: PyTorchVitalsTrainer
    xray_model: TorchvisionChestXRayGradCAM
    nlp_model: HuggingFaceClinicalNLP
    rag_model: ProductionClinicalHybridRAG
    config: InferenceConfig


class ModelLoader:
    """Centralized loader responsible for restoring serialized model weights and pipelines."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()

    def load_unsupervised_phenotyper(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> SklearnUnsupervisedPhenotyper:
        """Loads `healthcare_unsupervised_phenotyper.pkl` and binds `SklearnUnsupervisedPhenotyper` methods."""
        path = Path(checkpoint_path) if checkpoint_path else self.config.phenotyper_path
        if not path.exists():
            raise ModelLoadError(f"Unsupervised phenotyper checkpoint not found at: {path}")

        try:
            # Notebook 01 pickled the class under `__main__.SklearnUnsupervisedPhenotyper`.
            # Registering on `sys.modules["__main__"]` ensures safe unpickling from any entry point.
            main_mod = sys.modules.get("__main__")
            if main_mod is not None and not hasattr(main_mod, "SklearnUnsupervisedPhenotyper"):
                setattr(main_mod, "SklearnUnsupervisedPhenotyper", SklearnUnsupervisedPhenotyper)

            loaded = joblib.load(path)
            if not isinstance(loaded, SklearnUnsupervisedPhenotyper):
                loaded.__class__ = SklearnUnsupervisedPhenotyper
            loaded.logger = logging.getLogger(f"a_project.models.{SklearnUnsupervisedPhenotyper.model_name}")
            logger.info(
                "Loaded %s | Features: %s",
                path.name,
                loaded.feature_names_,
            )
            return loaded
        except Exception as exc:
            raise ModelLoadError(f"Failed to load unsupervised phenotyper from {path}: {exc}") from exc

    def load_lightgbm_explainer(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        feature_names: Optional[Sequence[str]] = None,
        phenotyper: Optional[SklearnUnsupervisedPhenotyper] = None,
    ) -> LightGBMReadmissionExplainer:
        """Loads `lightgbm_model.txt` into `LightGBMReadmissionExplainer` with TreeSHAP."""
        path = Path(checkpoint_path) if checkpoint_path else self.config.lightgbm_path
        if not path.exists():
            raise ModelLoadError(f"LightGBM model file not found at: {path}")

        resolved_features = (
            list(feature_names)
            if feature_names is not None
            else (
                list(phenotyper.feature_names_)
                if phenotyper is not None and getattr(phenotyper, "feature_names_", None)
                else list(self.config.tabular_feature_names)
            )
        )
        imputer = phenotyper.imputer if phenotyper is not None else None

        try:
            explainer = LightGBMReadmissionExplainer(
                model_path=path,
                feature_names=resolved_features,
                imputer=imputer,
                moderate_threshold=self.config.readmission_moderate_threshold,
                high_threshold=self.config.readmission_high_threshold,
                top_k_drivers=self.config.shap_top_k,
            )
            logger.info("Loaded %s into LightGBMReadmissionExplainer", path.name)
            return explainer
        except Exception as exc:
            raise ModelLoadError(f"Failed to load LightGBM model from {path}: {exc}") from exc

    def load_vitals_bilstm_model(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> PyTorchVitalsTrainer:
        """Loads `vitals_bilstm_attention.pth` and restores weights, mean, and std tensors."""
        path = Path(checkpoint_path) if checkpoint_path else self.config.vitals_path
        if not path.exists():
            raise ModelLoadError(f"Vitals Bi-LSTM checkpoint not found at: {path}")

        try:
            vitals_ckpt = torch.load(path, map_location="cpu", weights_only=False)
            trainer = PyTorchVitalsTrainer(
                epochs=int(vitals_ckpt.get("epochs", 25)),
                lr=float(vitals_ckpt.get("lr", 2e-3)),
                in_channels=int(vitals_ckpt.get("in_channels", 5)),
                hidden_dim=int(vitals_ckpt.get("hidden_dim", 32)),
                num_heads=int(vitals_ckpt.get("num_heads", 4)),
                device=self.config.device,
            )
            trainer.net.load_state_dict(vitals_ckpt["model_state_dict"])
            trainer.mean_ = np.asarray(vitals_ckpt["mean"], dtype=np.float32)
            trainer.std_ = np.asarray(vitals_ckpt["std"], dtype=np.float32)
            trainer.net.eval()
            logger.info("Loaded %s on device=%s", path.name, self.config.device)
            return trainer
        except Exception as exc:
            raise ModelLoadError(f"Failed to load Vitals Bi-LSTM checkpoint from {path}: {exc}") from exc

    def load_chest_xray_gradcam_model(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> TorchvisionChestXRayGradCAM:
        """Loads `pneumonia_resnet18_gradcam.pth` into `TorchvisionChestXRayGradCAM`."""
        path = Path(checkpoint_path) if checkpoint_path else self.config.xray_path
        if not path.exists():
            raise ModelLoadError(f"Chest X-Ray ResNet-18 checkpoint not found at: {path}")

        try:
            xray_ckpt = torch.load(path, map_location="cpu", weights_only=True)
            xray_engine = TorchvisionChestXRayGradCAM(device=self.config.device)
            xray_engine.model.load_state_dict(xray_ckpt["model_state_dict"])
            xray_engine.model.eval()
            logger.info("Loaded %s on device=%s", path.name, self.config.device)
            return xray_engine
        except Exception as exc:
            raise ModelLoadError(f"Failed to load Chest X-Ray checkpoint from {path}: {exc}") from exc

    def load_clinical_nlp_model(
        self,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> HuggingFaceClinicalNLP:
        """Loads the local saved Hugging Face BioNER pipeline (`models/biomedical_ner_model`)."""
        target_dir = Path(model_dir) if model_dir else self.config.ner_model_path
        try:
            return HuggingFaceClinicalNLP(
                model_name=target_dir,
                fallback_hub_id=self.config.ner_fallback_hub_id,
                device=self.config.hf_device,
            )
        except Exception as exc:
            raise ModelLoadError(f"Failed to load Biomedical NER model from {target_dir}: {exc}") from exc

    @staticmethod
    def load_corpus_from_file(corpus_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Loads a custom RAG corpus from a local JSON, JSONL, or CSV file.

        Expected fields per record: `doc_id`, `title`, `source`, `text`.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise InputValidationError(f"RAG corpus file not found at: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".csv":
            df = pd.read_csv(path)
            records = df.to_dict(orient="records")
        else:
            raise InputValidationError(
                f"Unsupported RAG corpus file extension '{suffix}'. Expected .json, .jsonl, or .csv."
            )

        required_keys = {"doc_id", "title", "source", "text"}
        validated_corpus: List[Dict[str, str]] = []
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict) or not required_keys.issubset(rec.keys()):
                raise InputValidationError(
                    f"Corpus record at index {idx} is missing required keys {sorted(required_keys)}."
                )
            validated_corpus.append(
                {
                    "doc_id": str(rec["doc_id"]),
                    "title": str(rec["title"]),
                    "source": str(rec["source"]),
                    "text": str(rec["text"]),
                }
            )
        return validated_corpus

    def load_hybrid_rag_model(
        self,
        corpus: Optional[List[Dict[str, str]]] = None,
        corpus_path: Optional[Union[str, Path]] = None,
        force_reindex: bool = False,
    ) -> ProductionClinicalHybridRAG:
        """Initializes `ProductionClinicalHybridRAG` from persistent disk storage or a user-supplied corpus."""
        resolved_corpus = (
            corpus
            if corpus is not None
            else (self.load_corpus_from_file(corpus_path) if corpus_path is not None else None)
        )
        persist_dir = self.config.rag_persist_dir if self.config.rag_use_persistent_store else None
        try:
            return ProductionClinicalHybridRAG(
                corpus=resolved_corpus,
                embed_model_name=self.config.rag_embed_model_name,
                reranker_model_name=self.config.rag_reranker_model_name,
                collection_name=self.config.rag_collection_name,
                top_k=self.config.rag_top_k,
                persist_dir=persist_dir,
                force_reindex=force_reindex,
            )
        except Exception as exc:
            raise ModelLoadError(f"Failed to initialize ProductionClinicalHybridRAG: {exc}") from exc

    def load_all(
        self,
        corpus: Optional[List[Dict[str, str]]] = None,
        corpus_path: Optional[Union[str, Path]] = None,
        force_reindex_rag: bool = False,
    ) -> LoadedModelBundle:
        """Loads all 6 pre-trained models and returns a unified `LoadedModelBundle`."""
        unsup_model = self.load_unsupervised_phenotyper()
        tabular_model = self.load_lightgbm_explainer(phenotyper=unsup_model)
        vitals_model = self.load_vitals_bilstm_model()
        xray_model = self.load_chest_xray_gradcam_model()
        nlp_model = self.load_clinical_nlp_model()
        rag_model = self.load_hybrid_rag_model(
            corpus=corpus,
            corpus_path=corpus_path,
            force_reindex=force_reindex_rag,
        )

        return LoadedModelBundle(
            unsup_model=unsup_model,
            tabular_model=tabular_model,
            vitals_model=vitals_model,
            xray_model=xray_model,
            nlp_model=nlp_model,
            rag_model=rag_model,
            config=self.config,
        )
