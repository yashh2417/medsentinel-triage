"""FastAPI router for file upload, individual modality preprocessing, and patient payload preparation."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np

from medsentinel_triage.base import InputValidationError
from medsentinel_triage.config import PROJECT_ROOT
from medsentinel_triage.input_processor import (
    TABULAR_FEATURE_NAMES,
    VITAL_COLUMNS,
    create_patient_payload_from_files,
    process_clinical_note,
    process_tabular_file,
    process_vitals_file,
    process_xray_image,
)

router = APIRouter(prefix="/api", tags=["File Upload & Preprocessing"])


def _save_temp_upload(upload: UploadFile) -> Path:
    """Writes a FastAPI `UploadFile` to a temporary file on disk and returns its Path."""
    suffix = Path(upload.filename or "upload.tmp").suffix.lower()
    fd, raw_path = tempfile.mkstemp(suffix=suffix, prefix="medsentinel_")
    path = Path(raw_path)
    content = upload.file.read()
    with open(fd, "wb") as f:
        f.write(content)
    if not content:
        path.unlink(missing_ok=True)
        raise InputValidationError(f"Uploaded file '{upload.filename}' is empty.")
    return path


@router.post("/preprocess/tabular")
def preprocess_tabular_endpoint(
    file: UploadFile = File(...),
    row_index: int = Form(0),
) -> Dict[str, Any]:
    """Uploads and preprocesses a tabular `.csv`, `.tsv`, or `.json` file."""
    tmp_path = _save_temp_upload(file)
    try:
        tabular_data = process_tabular_file(tmp_path, row_index=row_index)
        return {
            "status": "ok",
            "filename": file.filename,
            "feature_names": TABULAR_FEATURE_NAMES,
            "tabular_input": tabular_data,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/preprocess/vitals")
def preprocess_vitals_endpoint(
    file: UploadFile = File(...),
    pad_method: str = Form("carry_first"),
) -> Dict[str, Any]:
    """Uploads and preprocesses an 8h/24h vitals `.csv`, `.tsv`, or `.json` file into a 24x5 tensor."""
    tmp_path = _save_temp_upload(file)
    try:
        vitals_24h = process_vitals_file(tmp_path, target_hours=24, pad_method=pad_method)
        return {
            "status": "ok",
            "filename": file.filename,
            "pad_method": pad_method,
            "shape": [len(vitals_24h), len(vitals_24h[0])],
            "channel_names": VITAL_COLUMNS,
            "vitals_24h": vitals_24h,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/preprocess/xray")
def preprocess_xray_endpoint(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Uploads and converts a `.png` or `.jpg` Chest X-Ray image into a 28x28 grayscale matrix."""
    tmp_path = _save_temp_upload(file)
    try:
        xray_28x28 = process_xray_image(tmp_path, target_size=(28, 28))
        arr = np.asarray(xray_28x28, dtype=np.uint8)
        return {
            "status": "ok",
            "filename": file.filename,
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "xray_28x28": xray_28x28,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/preprocess/note")
def preprocess_note_endpoint(
    file: Optional[UploadFile] = File(None),
    clinical_note: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Uploads and reads a `.txt` clinical note file (or raw note text)."""
    if file is not None and file.filename:
        tmp_path = _save_temp_upload(file)
        try:
            note_text = process_clinical_note(tmp_path)
            filename = file.filename
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
    elif clinical_note and clinical_note.strip():
        note_text = clinical_note.strip()
        filename = "inline_note.txt"
    else:
        raise HTTPException(status_code=422, detail="Upload a .txt file or provide clinical_note text.")

    return {
        "status": "ok",
        "filename": filename,
        "char_count": len(note_text),
        "clinical_note": note_text,
    }


@router.post("/prepare-patient")
def prepare_patient_endpoint(
    tabular_file: UploadFile = File(...),
    vitals_file: UploadFile = File(...),
    xray_file: UploadFile = File(...),
    note_file: Optional[UploadFile] = File(None),
    clinical_note: Optional[str] = Form(None),
    patient_id: str = Form("PATIENT-001"),
    pad_method: str = Form("carry_first"),
    row_index: int = Form(0),
) -> Dict[str, Any]:
    """Combines uploaded CSV, PNG, and TXT files into a validated patient JSON payload."""
    tab_path = _save_temp_upload(tabular_file)
    vit_path = _save_temp_upload(vitals_file)
    xray_path = _save_temp_upload(xray_file)
    note_path: Optional[Path] = None

    try:
        if note_file is not None and note_file.filename:
            note_path = _save_temp_upload(note_file)
            note_filename = note_file.filename
        elif clinical_note and clinical_note.strip():
            fd, raw_note = tempfile.mkstemp(suffix=".txt", prefix="medsentinel_note_")
            note_path = Path(raw_note)
            with open(fd, "w", encoding="utf-8") as f:
                f.write(clinical_note.strip())
            note_filename = "inline_note.txt"
        else:
            raise InputValidationError("Provide either 'note_file' (.txt) or 'clinical_note'.")

        payload = create_patient_payload_from_files(
            patient_id=patient_id,
            tabular_file=tab_path,
            vitals_file=vit_path,
            xray_file=xray_path,
            note_file=note_path,
            vitals_pad_method=pad_method,
            tabular_row_index=row_index,
        )
        return {
            "status": "ok",
            "uploaded_filenames": {
                "tabular_file": tabular_file.filename,
                "vitals_file": vitals_file.filename,
                "xray_file": xray_file.filename,
                "note_file": note_filename,
            },
            "payload": payload,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        for p in (tab_path, vit_path, xray_path, note_path):
            if p is not None:
                p.unlink(missing_ok=True)


@router.get("/sample-files/load")
def load_sample_files_endpoint() -> Dict[str, Any]:
    """Loads and preprocesses sample patient files from `test_data/`."""
    test_dir = PROJECT_ROOT / "test_data"
    try:
        payload = create_patient_payload_from_files(
            patient_id="PATIENT-TEST-001",
            tabular_file=test_dir / "tabular.csv",
            vitals_file=test_dir / "vitals_8h.csv",
            xray_file=test_dir / "chest_xray.png",
            note_file=test_dir / "note.txt",
            vitals_pad_method="carry_first",
        )
        return {
            "status": "ok",
            "uploaded_filenames": {
                "tabular_file": "tabular.csv",
                "vitals_file": "vitals_8h.csv",
                "xray_file": "chest_xray.png",
                "note_file": "note.txt",
            },
            "payload": payload,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
