"""Shared input validation and preprocessing utilities matching the training notebooks."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import torch
import torch.nn.functional as F

from a_project.base import InputValidationError
from a_project.config import (
    DEFAULT_ICU_VITALS,
    ICU_NUM_VITAL_CHANNELS,
    ICU_SEQUENCE_LENGTH,
    MIMIC_VITAL_ITEMIDS,
    TABULAR_FEATURE_NAMES,
    XRAY_MODEL_INPUT_SIZE,
    XRAY_RAW_SIZE,
)

TabularInputType = Union[str, Mapping[str, Any], pd.Series, pd.DataFrame]
VitalsInputType = Union[Sequence[Sequence[float]], np.ndarray, pd.DataFrame]
XRayInputType = Union[Sequence[Sequence[Union[int, float]]], np.ndarray]


def parse_age_to_years(age_val: Any) -> float:
    """Converts raw age bracket strings (e.g., '[80-90)') to numeric midpoint age_years (+5.0) as in Notebook 01/02."""
    if age_val is None or (isinstance(age_val, float) and np.isnan(age_val)):
        return np.nan
    if isinstance(age_val, (int, float, np.number)) and not isinstance(age_val, bool):
        return float(age_val)
    age_str = str(age_val).strip()
    match = re.search(r"(\d+)", age_str)
    if not match:
        return np.nan
    # If formatted as an interval like "[80-90)", add 5.0 to match training preprocessing
    if "[" in age_str or "-" in age_str or ")" in age_str:
        return float(match.group(1)) + 5.0
    return float(match.group(1))


def validate_and_preprocess_tabular_row(
    raw_input: TabularInputType,
    feature_names: Sequence[str] = TABULAR_FEATURE_NAMES,
) -> pd.DataFrame:
    """Validates and normalizes a patient tabular input into a 1-row DataFrame with exact feature ordering.

    Supports:
    - JSON string (as passed inside `TriageAgentState["tabular_row_json"]`)
    - Python dictionary / Mapping
    - `pd.Series` or `pd.DataFrame`
    - Automatic conversion of raw `age` intervals (e.g. `"[70-80)"`) into `age_years`
    """
    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise InputValidationError(f"Invalid tabular JSON string: {exc}") from exc
        if not isinstance(parsed, dict):
            raise InputValidationError("Tabular JSON string must decode to a JSON object/dictionary.")
        df = pd.DataFrame([parsed])
    elif isinstance(raw_input, pd.Series):
        df = pd.DataFrame([raw_input.to_dict()])
    elif isinstance(raw_input, Mapping):
        df = pd.DataFrame([dict(raw_input)])
    elif isinstance(raw_input, pd.DataFrame):
        if raw_input.empty:
            raise InputValidationError("Tabular input DataFrame cannot be empty.")
        df = raw_input.copy().reset_index(drop=True)
    else:
        raise InputValidationError(
            f"Unsupported tabular input type: {type(raw_input).__name__}. "
            "Expected JSON string, dict, pd.Series, or pd.DataFrame."
        )

    # Derive `age_years` from `age` if needed, matching Notebook 01 & Notebook 02
    if "age_years" not in df.columns and "age" in df.columns:
        if not is_numeric_dtype(df["age"]):
            df["age_years"] = df["age"].apply(parse_age_to_years)
        else:
            df["age_years"] = pd.to_numeric(df["age"], errors="coerce")

    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise InputValidationError(
            f"Missing required tabular features: {missing_cols}. Expected features: {list(feature_names)}"
        )

    ordered_df = df[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    return ordered_df


def build_vitals_tensor_from_chartevents(
    chartevents_df: pd.DataFrame,
    seq_len: int = ICU_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Converts a single-stay MIMIC-IV `chartevents` DataFrame into a `(seq_len, 5)` float32 array.

    Preserves exact Notebook 03 logic:
    - Filters `itemid` in `{220045: 0, 220277: 1, 220179: 2, 220210: 3, 223761: 4}`
    - Converts Fahrenheit (`itemid == 223761`) to Celsius: `(F - 32.0) * (5.0 / 9.0)`
    - Bins timestamps into hourly buckets relative to first charttime `t0`
    - Forward-fills missing hourly channel values starting from `DEFAULT_ICU_VITALS`
    """
    required_cols = {"charttime", "itemid", "valuenum"}
    if not required_cols.issubset(chartevents_df.columns):
        raise InputValidationError(
            f"chartevents DataFrame must contain columns {sorted(required_cols)}."
        )

    sub = chartevents_df[chartevents_df["itemid"].isin(MIMIC_VITAL_ITEMIDS.keys())].dropna(
        subset=["valuenum"]
    ).copy()
    if sub.empty:
        raise InputValidationError("No valid ICU vital sign events found in chartevents DataFrame.")

    sub["charttime"] = pd.to_datetime(sub["charttime"])
    sub = sub.sort_values("charttime")

    temp_mask = sub["itemid"] == 223761
    sub.loc[temp_mask, "valuenum"] = (sub.loc[temp_mask, "valuenum"] - 32.0) * (5.0 / 9.0)

    t0 = sub["charttime"].iloc[0]
    sub["hour_bin"] = ((sub["charttime"] - t0).dt.total_seconds() // 3600).astype(int)
    sub = sub[(sub["hour_bin"] >= 0) & (sub["hour_bin"] < seq_len)]

    default_vitals = np.array(DEFAULT_ICU_VITALS, dtype=np.float32)
    seq = np.tile(default_vitals, (seq_len, 1))
    mask = np.zeros((seq_len, ICU_NUM_VITAL_CHANNELS), dtype=bool)

    for _, ev in sub.iterrows():
        h = int(ev["hour_bin"])
        c = MIMIC_VITAL_ITEMIDS[int(ev["itemid"])]
        seq[h, c] = float(ev["valuenum"])
        mask[h, c] = True

    for c in range(ICU_NUM_VITAL_CHANNELS):
        last_val = default_vitals[c]
        for h in range(seq_len):
            if mask[h, c]:
                last_val = seq[h, c]
            else:
                seq[h, c] = last_val

    return seq.astype(np.float32)


def validate_and_preprocess_vitals(
    raw_vitals: VitalsInputType,
    seq_len: int = ICU_SEQUENCE_LENGTH,
    num_channels: int = ICU_NUM_VITAL_CHANNELS,
) -> np.ndarray:
    """Validates and converts 24-hour ICU vitals input into a `(24, 5)` float32 numpy array."""
    if isinstance(raw_vitals, pd.DataFrame):
        if {"charttime", "itemid", "valuenum"}.issubset(raw_vitals.columns):
            return build_vitals_tensor_from_chartevents(raw_vitals, seq_len=seq_len)
        arr = raw_vitals.to_numpy(dtype=np.float32)
    else:
        try:
            arr = np.asarray(raw_vitals, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise InputValidationError(f"Could not convert vitals input to float32 array: {exc}") from exc

    if arr.shape != (seq_len, num_channels):
        raise InputValidationError(
            f"Invalid ICU vitals shape {arr.shape}; expected ({seq_len}, {num_channels}) "
            "representing 24 hours x 5 channels [HR, SpO2, SBP, RespRate, TempC]."
        )
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise InputValidationError("ICU vitals tensor contains NaN or Inf values.")

    return arr


def validate_xray_image(raw_xray: XRayInputType) -> np.ndarray:
    """Validates a 28x28 grayscale Chest X-Ray image array."""
    try:
        arr = np.asarray(raw_xray)
    except (ValueError, TypeError) as exc:
        raise InputValidationError(f"Could not convert X-ray input to numpy array: {exc}") from exc

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    if arr.shape != XRAY_RAW_SIZE:
        raise InputValidationError(
            f"Invalid Chest X-Ray image shape {arr.shape}; expected {XRAY_RAW_SIZE}."
        )
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise InputValidationError("Chest X-Ray image contains NaN or Inf values.")

    return arr.astype(np.uint8)


def prepare_xray_batch_tensor(imgs_uint8: np.ndarray) -> torch.Tensor:
    """Applies exact Notebook 04 & 06 preprocessing: scale [0, 1], bilinear upsample to 64x64, 3-channel repeat."""
    t = torch.tensor(imgs_uint8, dtype=torch.float32) / 255.0
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[-1] == 1:
        t = t.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=XRAY_MODEL_INPUT_SIZE, mode="bilinear", align_corners=False)
    return t.repeat(1, 3, 1, 1)


def validate_non_empty_text(text: Any, field_name: str = "text") -> str:
    """Validates that a clinical note or search query is a non-empty string."""
    if not isinstance(text, str) or not text.strip():
        raise InputValidationError(f"'{field_name}' must be a non-empty string.")
    return text.strip()
