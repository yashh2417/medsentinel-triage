"""Core input-processing module converting raw clinical modality files (CSV, PNG/JPG, TXT) into validated patient JSON payloads."""

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


def process_tabular_file(
    file_path: Union[str, Path],
    row_index: int = 0,
) -> Dict[str, float]:
    """Reads a CSV, TSV, or JSON tabular file and returns a dictionary with the required 9 features."""
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
            f"Unsupported tabular file format '{path.suffix}'. Expected .csv, .tsv, or .json"
        )

    if df.empty:
        raise InputValidationError(f"Tabular file {path.name} contains no data rows.")
    if row_index < 0 or row_index >= len(df):
        raise InputValidationError(
            f"row_index {row_index} is out of bounds for tabular file with {len(df)} row(s)."
        )

    row = df.iloc[row_index].copy()
    if "age_years" not in row and "age" in row:
        row["age_years"] = parse_age_to_years(row["age"])

    missing = [c for c in TABULAR_FEATURE_NAMES if c not in row]
    if missing:
        raise InputValidationError(
            f"Tabular file is missing required columns: {missing}. Found: {list(row.index)}"
        )

    return {col: float(pd.to_numeric(row[col], errors="raise")) for col in TABULAR_FEATURE_NAMES}


def process_vitals_file(
    file_path: Union[str, Path],
    target_hours: int = ICU_SEQUENCE_LENGTH,
    pad_method: str = "carry_first",
) -> List[List[float]]:
    """Reads a vitals CSV, TSV, or JSON file and returns a `24x5` list of floats."""
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
        df.columns = [col_map.get(str(c).lower().strip(), str(c).strip()) for c in df.columns]

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
            f"Unsupported vitals file format '{path.suffix}'. Expected .csv, .tsv, or .json"
        )

    if arr.ndim != 2 or arr.shape[1] != 5 or arr.shape[0] == 0:
        raise InputValidationError(f"Expected non-empty (N, 5) vitals array, got {arr.shape}.")

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
