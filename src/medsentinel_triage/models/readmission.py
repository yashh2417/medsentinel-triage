"""Tabular readmission risk prediction and TreeSHAP explainability module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.impute import KNNImputer

from medsentinel_triage.base import BaseInferenceModel, InputValidationError
from medsentinel_triage.config import TABULAR_FEATURE_NAMES
from medsentinel_triage.preprocessing import TabularInputType, validate_and_preprocess_tabular_row

ReadmissionInputType = Union[np.ndarray, TabularInputType]


class LightGBMReadmissionExplainer(
    BaseInferenceModel[ReadmissionInputType, np.ndarray, Dict[str, Any]]
):
    """Wraps the Optuna-tuned LightGBM Booster (`lightgbm_model.txt`) and TreeSHAP explainer for patient inference."""

    model_name: str = "lightgbm_readmission_explainer"

    def __init__(
        self,
        model_path: Union[str, Path] = "lightgbm_model.txt",
        feature_names: Optional[Sequence[str]] = None,
        imputer: Optional[KNNImputer] = None,
        moderate_threshold: float = 0.25,
        high_threshold: float = 0.40,
        top_k_drivers: int = 5,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.booster = lgb.Booster(model_file=str(self.model_path))
        self.explainer = shap.TreeExplainer(self.booster)
        self.feature_names_: List[str] = (
            list(feature_names) if feature_names is not None else list(TABULAR_FEATURE_NAMES)
        )
        self.imputer = imputer
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.top_k_drivers = top_k_drivers

    def validate_input(self, raw_input: ReadmissionInputType) -> None:
        if raw_input is None:
            raise InputValidationError("Readmission model input cannot be None.")
        if isinstance(raw_input, np.ndarray):
            arr = np.asarray(raw_input, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_names_):
                raise InputValidationError(
                    f"Expected 2D array with {len(self.feature_names_)} features, got shape {arr.shape}."
                )
            if np.isnan(arr).any():
                raise InputValidationError(
                    "Imputed numpy array passed to LightGBMReadmissionExplainer contains NaN values."
                )
        else:
            validate_and_preprocess_tabular_row(raw_input, feature_names=self.feature_names_)

    def preprocess(self, raw_input: ReadmissionInputType) -> np.ndarray:
        if isinstance(raw_input, np.ndarray):
            arr = np.asarray(raw_input, dtype=np.float64)
            return arr.reshape(1, -1) if arr.ndim == 1 else arr

        ordered_df = validate_and_preprocess_tabular_row(raw_input, feature_names=self.feature_names_)
        if self.imputer is not None:
            return self.imputer.transform(ordered_df)
        arr = ordered_df.to_numpy(dtype=np.float64)
        if np.isnan(arr).any():
            raise InputValidationError(
                "Tabular input contains missing values and no KNNImputer is attached to LightGBMReadmissionExplainer."
            )
        return arr

    def predict_and_explain(
        self,
        x_imp_row: ReadmissionInputType,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Runs single-patient inference with the loaded LightGBM Booster and computes exact TreeSHAP drivers."""
        arr = self.preprocess(x_imp_row) if not isinstance(x_imp_row, np.ndarray) else (
            x_imp_row.reshape(1, -1) if x_imp_row.ndim == 1 else x_imp_row
        )
        feat_cols = list(feature_names) if feature_names is not None else self.feature_names_

        prob = float(self.booster.predict(arr)[0])
        shap_vals = self.explainer.shap_values(arr)
        if isinstance(shap_vals, list):
            sv_row = shap_vals[1][0]
        elif np.asarray(shap_vals).ndim == 3:
            sv_row = shap_vals[0, :, 1]
        else:
            sv_row = shap_vals[0]

        ranked = sorted(
            zip(feat_cols, arr[0], sv_row),
            key=lambda item: abs(float(item[2])),
            reverse=True,
        )
        top_drivers = [
            {
                "feature": fname,
                "value": round(float(val), 2),
                "shap_log_odds_impact": round(float(sv), 4),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
            }
            for fname, val, sv in ranked[: self.top_k_drivers]
        ]

        if prob >= self.high_threshold:
            risk_tier = "HIGH"
        elif prob >= self.moderate_threshold:
            risk_tier = "MODERATE"
        else:
            risk_tier = "LOW"

        return {
            "readmission_or_icu_risk_prob": round(prob, 4),
            "risk_tier": risk_tier,
            "top_treeshap_drivers": top_drivers,
        }

    def predict(self, preprocessed_input: np.ndarray) -> Dict[str, Any]:
        return self.predict_and_explain(preprocessed_input, feature_names=self.feature_names_)
