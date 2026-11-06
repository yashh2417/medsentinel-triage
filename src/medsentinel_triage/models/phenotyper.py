"""Unsupervised patient phenotyping and anomaly detection inference module."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from medsentinel_triage.base import BaseInferenceModel, InputValidationError
from medsentinel_triage.config import DEFAULT_CLUSTER_LABELS_MAP, TABULAR_FEATURE_NAMES
from medsentinel_triage.preprocessing import TabularInputType, validate_and_preprocess_tabular_row


class SklearnUnsupervisedPhenotyper(
    BaseInferenceModel[TabularInputType, pd.DataFrame, Tuple[np.ndarray, Dict[str, Any]]]
):
    """Production Unsupervised ML pipeline using scikit-learn (`KNNImputer` + `StandardScaler` + `PCA` + `KMeans` + `IsolationForest`).

    Preserves exact class/attribute structure from Notebook 01 and Notebook 06 so
    `healthcare_unsupervised_phenotyper.pkl` unpickles directly into this class.
    """

    model_name: str = "unsupervised_phenotyper"

    def __init__(
        self,
        n_pca_components: int = 5,
        n_clusters: int = 3,
        contamination: float = 0.03,
    ) -> None:
        super().__init__()
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
        self.iforest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names_: List[str] = list(TABULAR_FEATURE_NAMES)
        self.cluster_labels_map_: Dict[int, str] = dict(DEFAULT_CLUSTER_LABELS_MAP)

    @property
    def _log(self) -> logging.Logger:
        """Safe logger property for instances restored via `joblib.load` without calling `__init__`."""
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(f"medsentinel_triage.models.{self.model_name}")
        return self.logger

    @property
    def expected_features(self) -> List[str]:
        return list(self.feature_names_) if getattr(self, "feature_names_", None) else list(TABULAR_FEATURE_NAMES)

    def validate_input(self, raw_input: TabularInputType) -> None:
        if raw_input is None:
            raise InputValidationError("Patient tabular input cannot be None.")
        validate_and_preprocess_tabular_row(raw_input, feature_names=self.expected_features)

    def preprocess(self, raw_input: TabularInputType) -> pd.DataFrame:
        return validate_and_preprocess_tabular_row(raw_input, feature_names=self.expected_features)

    def analyze_patient(self, patient_row: TabularInputType) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Runs imputation, scaling, PCA projection, KMeans phenotyping, and IsolationForest anomaly scoring."""
        if isinstance(patient_row, pd.DataFrame) and set(self.expected_features).issubset(patient_row.columns):
            ordered_df = patient_row[self.expected_features].apply(pd.to_numeric, errors="coerce")
        else:
            ordered_df = self.preprocess(patient_row)

        X_imp = self.imputer.transform(ordered_df)
        X_scaled = self.scaler.transform(X_imp)
        X_pca = self.pca.transform(X_scaled)
        cid = int(self.kmeans.predict(X_pca)[0])
        anom_pred = int(self.iforest.predict(X_scaled)[0])
        anom_score = float(-self.iforest.score_samples(X_scaled)[0])

        labels_map = getattr(self, "cluster_labels_map_", None) or DEFAULT_CLUSTER_LABELS_MAP
        report = {
            "cluster_id": cid,
            "phenotype_name": labels_map.get(cid, f"Cluster {cid}"),
            "pca_coordinates": [round(float(x), 3) for x in X_pca[0]],
            "isolation_forest_anomaly_score": round(anom_score, 4),
            "is_outlier_flagged": bool(anom_pred == -1),
        }
        return X_imp, report

    def predict(self, preprocessed_input: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.analyze_patient(preprocessed_input)
