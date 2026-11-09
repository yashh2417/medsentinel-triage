"""Core input-processing module converting raw clinical modality files (CSV, PNG/JPG, TXT, JSON) into validated patient payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from medsentinel_triage.base import InputValidationError
from medsentinel_triage.config import (
    DEFAULT_ICU_VITALS,
    ICU_SEQUENCE_LENGTH,
    PROJECT_ROOT,
    TABULAR_FEATURE_NAMES,
    VITAL_CHANNEL_NAMES,
    XRAY_RAW_SIZE,
)
from medsentinel_triage.preprocessing import (
    build_vitals_tensor_from_chartevents,
    parse_age_to_years,
)

NORMAL_BASELINE_VITALS: List[float] = list(DEFAULT_ICU_VITALS)
VITAL_COLUMNS: List[str] = list(VITAL_CHANNEL_NAMES)


def _resolve_and_validate_path(raw_path_str: Any, field_name: str) -> Path:
    """Resolves a file path (absolute or relative to PROJECT_ROOT) and validates that it exists on disk."""
    if raw_path_str is None or (isinstance(raw_path_str, float) and np.isnan(raw_path_str)):
        raise InputValidationError(f"Missing required path for '{field_name}'.")
    cleaned = str(raw_path_str).strip()
    if not cleaned or cleaned.lower() == "nan":
        raise InputValidationError(f"Empty path provided for '{field_name}'.")

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    if not candidate.exists() or not candidate.is_file():
        raise InputValidationError(
            f"Referenced file path for '{field_name}' does not exist on disk: {candidate}"
        )
    return candidate


def _extract_tabular_row_dict(row: pd.Series) -> Dict[str, float]:
    """Validates and extracts the 9 required tabular features from a pandas Series row."""
    row_copy = row.copy()
    if "age_years" not in row_copy and "age" in row_copy:
        row_copy["age_years"] = parse_age_to_years(row_copy["age"])

    missing = [c for c in TABULAR_FEATURE_NAMES if c not in row_copy]
    if missing:
        raise InputValidationError(
            f"Tabular data is missing required columns: {missing}. Found: {list(row_copy.index)}"
        )

    result: Dict[str, float] = {}
    for col in TABULAR_FEATURE_NAMES:
        val = float(pd.to_numeric(row_copy[col], errors="coerce"))
        if np.isnan(val) or np.isinf(val):
            raise InputValidationError(f"Tabular column '{col}' contains non-numeric or NaN value.")
        result[col] = val
    return result


def _pad_or_truncate_vitals_array(
    arr: np.ndarray,
    target_hours: int = ICU_SEQUENCE_LENGTH,
    pad_method: str = "carry_first",
) -> List[List[float]]:
    """Pads or truncates an `(N, 5)` float32 vitals array into a `24x5` list of floats."""
    if arr.ndim != 2 or arr.shape[1] != 5 or arr.shape[0] == 0:
        raise InputValidationError(f"Expected non-empty (N, 5) vitals array, got {arr.shape}.")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise InputValidationError("Vitals data contains NaN or infinite values.")

    n_rows = arr.shape[0]
    if n_rows == target_hours:
        padded_arr = arr
    elif n_rows > target_hours:
        padded_arr = arr[-target_hours:]
    else:
        pad_len = target_hours - n_rows
        if pad_method == "carry_first":
            prefix = np.tile(arr[0], (pad_len, 1))
        elif pad_method == "normal_baseline":
            prefix = np.tile(np.array(NORMAL_BASELINE_VITALS, dtype=np.float32), (pad_len, 1))
        else:
            raise InputValidationError(f"Unknown pad_method: {pad_method}")
        padded_arr = np.vstack([prefix, arr])

    return [[round(float(v), 2) for v in row] for row in padded_arr]


def _normalize_vitals_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes vitals column names to `VITAL_COLUMNS`."""
    col_map = {
        "hr": "heart_rate_bpm",
        "heart_rate": "heart_rate_bpm",
        "heart_rate_bpm": "heart_rate_bpm",
        "spo2": "spo2_pct",
        "spo2_pct": "spo2_pct",
        "sbp": "systolic_bp_mmhg",
        "systolic_bp": "systolic_bp_mmhg",
        "systolic_bp_mmhg": "systolic_bp_mmhg",
        "rr": "resp_rate_bpm",
        "resp_rate": "resp_rate_bpm",
        "resp_rate_bpm": "resp_rate_bpm",
        "temp": "temp_c",
        "temp_c": "temp_c",
    }
    out = df.copy()
    out.columns = [col_map.get(str(c).lower().strip(), str(c).strip()) for c in out.columns]
    return out


def process_tabular_file(
    file_path: Union[str, Path],
    row_index: int = 0,
) -> Dict[str, float]:
    """Reads a CSV or JSON tabular file and returns a dictionary with the required 9 features."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tabular input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise InputValidationError(
            f"Unsupported tabular file format '{path.suffix}'. Expected .csv or .json"
        )

    if df.empty:
        raise InputValidationError(f"Tabular file {path.name} contains no data rows.")
    if row_index < 0 or row_index >= len(df):
        raise InputValidationError(
            f"row_index {row_index} is out of bounds for tabular file with {len(df)} row(s)."
        )

    return _extract_tabular_row_dict(df.iloc[row_index])


def process_vitals_file(
    file_path: Union[str, Path],
    target_hours: int = ICU_SEQUENCE_LENGTH,
    pad_method: str = "carry_first",
) -> List[List[float]]:
    """Reads a vitals CSV or JSON file and returns a `24x5` list of floats."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Vitals input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            arr = np.asarray(json.load(f), dtype=np.float32)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)

        if {"charttime", "itemid", "valuenum"}.issubset(df.columns):
            return build_vitals_tensor_from_chartevents(df, seq_len=target_hours).tolist()

        df = _normalize_vitals_columns(df)
        if set(VITAL_COLUMNS).issubset(df.columns):
            arr = df[VITAL_COLUMNS].to_numpy(dtype=np.float32)
        elif df.shape[1] == 5:
            arr = df.to_numpy(dtype=np.float32)
        else:
            raise InputValidationError(
                f"Vitals file must have columns {VITAL_COLUMNS} or 5 columns. Found: {list(df.columns)}"
            )
    else:
        raise InputValidationError(
            f"Unsupported vitals file format '{path.suffix}'. Expected .csv or .json"
        )

    return _pad_or_truncate_vitals_array(arr, target_hours=target_hours, pad_method=pad_method)


def process_xray_image(
    file_path: Union[str, Path],
    target_size: Tuple[int, int] = XRAY_RAW_SIZE,
) -> List[List[int]]:
    """Opens an image (PNG, JPG, BMP, TIFF), converts to grayscale, resizes to 28x28, and returns uint8 2D list."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Chest X-Ray image file not found: {path}")

    with Image.open(path) as img:
        img_gray = img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(img_gray, dtype=np.uint8)
    return arr.tolist()


def process_clinical_note(file_path: Union[str, Path]) -> str:
    """Reads a clinical note text file and returns a validated non-empty string."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Clinical note text file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise InputValidationError(f"Clinical note file {path.name} is empty.")
    return text


def create_patient_payload_from_files(
    patient_id: str,
    tabular_file: Union[str, Path],
    vitals_file: Union[str, Path],
    xray_file: Union[str, Path],
    note_file: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
    vitals_pad_method: str = "carry_first",
    tabular_row_index: int = 0,
) -> Dict[str, Any]:
    """Converts individual raw modality files into a validated patient payload dictionary and optional JSON file."""
    payload: Dict[str, Any] = {
        "patient_id": str(patient_id).strip() or "PATIENT-001",
        "tabular_input": process_tabular_file(tabular_file, row_index=tabular_row_index),
        "vitals_24h": process_vitals_file(vitals_file, pad_method=vitals_pad_method),
        "xray_28x28": process_xray_image(xray_file),
        "clinical_note": process_clinical_note(note_file),
    }

    if output_json_path:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def process_batch_csv_files(
    manifest_csv_path: Union[str, Path],
    vitals_csv_path: Union[str, Path],
    pad_method: str = "carry_first",
    max_rows: int = 100,
) -> List[Dict[str, Any]]:
    """Processes a batch from two CSV files (`batch_tabular_manifest.csv` and `batch_vitals.csv`).

    Strictly validates that:
    1. Both `manifest_csv_path` and `vitals_csv_path` exist on disk.
    2. Every per-patient file path referenced in `manifest_csv_path` (`xray_image_path`, `note_path`,
       and optional `vitals_csv_path`) exists on disk; raises `InputValidationError` if any path is missing.
    3. Every `patient_id` in `manifest_csv_path` has matching hourly vital sign rows in `vitals_csv_path`.
    """
    manifest_path = Path(manifest_csv_path)
    vitals_path = Path(vitals_csv_path)
    if not manifest_path.exists():
        raise InputValidationError(f"Batch tabular manifest CSV not found at: {manifest_path}")
    if not vitals_path.exists():
        raise InputValidationError(f"Batch vitals CSV not found at: {vitals_path}")

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise InputValidationError(f"Batch manifest CSV '{manifest_path.name}' contains 0 rows.")
    if "patient_id" not in manifest_df.columns:
        raise InputValidationError(
            f"Batch manifest CSV '{manifest_path.name}' must contain a 'patient_id' column."
        )
    if "xray_image_path" not in manifest_df.columns:
        raise InputValidationError(
            f"Batch manifest CSV '{manifest_path.name}' must contain an 'xray_image_path' column."
        )

    vitals_df = _normalize_vitals_columns(pd.read_csv(vitals_path))
    if "patient_id" not in vitals_df.columns or not set(VITAL_COLUMNS).issubset(vitals_df.columns):
        raise InputValidationError(
            f"Batch vitals CSV '{vitals_path.name}' must contain 'patient_id' and {VITAL_COLUMNS}. "
            f"Found: {list(vitals_df.columns)}"
        )

    payloads: List[Dict[str, Any]] = []
    for idx, row in manifest_df.head(max_rows).iterrows():
        raw_pid = row.get("patient_id")
        if raw_pid is None or pd.isna(raw_pid) or not str(raw_pid).strip():
            raise InputValidationError(f"Row {idx} in '{manifest_path.name}' has an empty 'patient_id'.")
        patient_id = str(raw_pid).strip()

        # 1. Validate & extract the 9 tabular features
        tabular_input = _extract_tabular_row_dict(row)

        # 2. Validate that referenced paths in the manifest exist on disk
        xray_file = _resolve_and_validate_path(
            row.get("xray_image_path"),
            field_name=f"xray_image_path ({patient_id})",
        )
        xray_28x28 = process_xray_image(xray_file)

        if "vitals_csv_path" in row and pd.notna(row.get("vitals_csv_path")) and str(row.get("vitals_csv_path")).strip():
            _resolve_and_validate_path(
                row.get("vitals_csv_path"),
                field_name=f"vitals_csv_path ({patient_id})",
            )

        if "note_path" in row and pd.notna(row.get("note_path")) and str(row.get("note_path")).strip():
            note_file = _resolve_and_validate_path(
                row.get("note_path"),
                field_name=f"note_path ({patient_id})",
            )
            clinical_note = process_clinical_note(note_file)
        elif "clinical_note" in row and pd.notna(row.get("clinical_note")) and str(row.get("clinical_note")).strip():
            clinical_note = str(row["clinical_note"]).strip()
        else:
            raise InputValidationError(
                f"Row for '{patient_id}' must provide a valid 'note_path' or 'clinical_note'."
            )

        # 3. Extract patient's 24h vitals from batch_vitals.csv by patient_id
        patient_vitals_df = vitals_df[vitals_df["patient_id"].astype(str).str.strip() == patient_id].copy()
        if patient_vitals_df.empty:
            raise InputValidationError(
                f"No vital sign records found for patient_id '{patient_id}' in '{vitals_path.name}'."
            )
        if "hour" in patient_vitals_df.columns:
            patient_vitals_df = patient_vitals_df.sort_values("hour")

        vitals_arr = patient_vitals_df[VITAL_COLUMNS].to_numpy(dtype=np.float32)
        vitals_24h = _pad_or_truncate_vitals_array(
            vitals_arr,
            target_hours=ICU_SEQUENCE_LENGTH,
            pad_method=pad_method,
        )

        payloads.append(
            {
                "patient_id": patient_id,
                "tabular_input": tabular_input,
                "vitals_24h": vitals_24h,
                "xray_28x28": xray_28x28,
                "clinical_note": clinical_note,
            }
        )

    return payloads


def process_batch_json_file(
    batch_json_path: Union[str, Path],
    pad_method: str = "carry_first",
    max_rows: int = 100,
) -> List[Dict[str, Any]]:
    """Processes a batch `.json` file containing a list (or single object) of patient records."""
    path = Path(batch_json_path)
    if not path.exists():
        raise InputValidationError(f"Batch JSON file not found at: {path}")
    if path.suffix.lower() != ".json":
        raise InputValidationError(
            f"Unsupported batch JSON format '{path.suffix}'. Only '.json' is supported."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_items = data if isinstance(data, list) else [data]
    if not raw_items:
        raise InputValidationError(f"Batch JSON file '{path.name}' contains no patient records.")

    payloads: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items[:max_rows]):
        if not isinstance(item, dict):
            raise InputValidationError(f"Batch JSON item at index {idx} must be a JSON object.")

        patient_id = str(item.get("patient_id") or f"BATCH-PATIENT-{idx + 1:03d}").strip()
        tab_raw = item.get("tabular_input", item)
        tabular_input = _extract_tabular_row_dict(pd.Series(tab_raw))

        if "vitals_24h" in item and item["vitals_24h"] is not None:
            vitals_arr = np.asarray(item["vitals_24h"], dtype=np.float32)
            vitals_24h = _pad_or_truncate_vitals_array(vitals_arr, pad_method=pad_method)
        elif "vitals_csv_path" in item:
            v_path = _resolve_and_validate_path(item["vitals_csv_path"], f"vitals_csv_path ({patient_id})")
            vitals_24h = process_vitals_file(v_path, pad_method=pad_method)
        else:
            raise InputValidationError(f"Patient '{patient_id}' in JSON is missing 'vitals_24h'.")

        if "xray_28x28" in item and item["xray_28x28"] is not None:
            xray_28x28 = np.asarray(item["xray_28x28"], dtype=np.uint8).tolist()
        elif "xray_image_path" in item:
            x_path = _resolve_and_validate_path(item["xray_image_path"], f"xray_image_path ({patient_id})")
            xray_28x28 = process_xray_image(x_path)
        else:
            raise InputValidationError(f"Patient '{patient_id}' in JSON is missing 'xray_28x28'.")

        if "note_path" in item and item["note_path"]:
            n_path = _resolve_and_validate_path(item["note_path"], f"note_path ({patient_id})")
            clinical_note = process_clinical_note(n_path)
        elif isinstance(item.get("clinical_note"), str) and item["clinical_note"].strip():
            clinical_note = item["clinical_note"].strip()
        else:
            raise InputValidationError(f"Patient '{patient_id}' in JSON is missing 'clinical_note'.")

        payloads.append(
            {
                "patient_id": patient_id,
                "tabular_input": tabular_input,
                "vitals_24h": vitals_24h,
                "xray_28x28": xray_28x28,
                "clinical_note": clinical_note,
            }
        )

    return payloads


def main(argv: Optional[List[str]] = None) -> None:
    """CLI to generate patient_input.json from raw modality files."""
    default_dir = Path(__file__).resolve().parents[2] / "test_data"

    parser = argparse.ArgumentParser(
        description="Convert raw clinical files (CSV, PNG/JPG, TXT) into a validated patient input JSON."
    )
    parser.add_argument("--patient-id", type=str, default="PATIENT-TEST-001")
    parser.add_argument("--tabular", type=str, default=str(default_dir / "tabular.csv"))
    parser.add_argument("--vitals", type=str, default=str(default_dir / "vitals_8h.csv"))
    parser.add_argument("--xray", type=str, default=str(default_dir / "chest_xray.png"))
    parser.add_argument("--note", type=str, default=str(default_dir / "note.txt"))
    parser.add_argument("--output", type=str, default=str(default_dir / "patient_input.json"))
    parser.add_argument("--pad-method", type=str, choices=["carry_first", "normal_baseline"], default="carry_first")
    args = parser.parse_args(argv)

    payload = create_patient_payload_from_files(
        patient_id=args.patient_id,
        tabular_file=args.tabular,
        vitals_file=args.vitals,
        xray_file=args.xray,
        note_file=args.note,
        output_json_path=args.output,
        vitals_pad_method=args.pad_method,
    )
    print(f"Created patient payload for {payload['patient_id']} at {args.output}")


if __name__ == "__main__":
    main()
