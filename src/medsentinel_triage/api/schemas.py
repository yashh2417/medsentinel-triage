"""Pydantic request and response schemas for the MedSentinel Triage REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TabularPredictRequest(BaseModel):
    """Request schema for standalone tabular phenotyping and readmission risk prediction."""

    tabular_input: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing the 9 inpatient tabular features (or raw 'age' bracket).",
    )


class VitalsPredictRequest(BaseModel):
    """Request schema for standalone 24-hour ICU vitals trajectory forecasting."""

    vitals_24h: List[List[float]] = Field(
        ...,
        description="24x5 matrix of hourly ICU vital signs [HR, SpO2, SBP, RespRate, TempC].",
    )


class XRayPredictRequest(BaseModel):
    """Request schema for standalone Chest X-Ray ResNet-18 + Grad-CAM inference."""

    xray_28x28: List[List[int]] = Field(
        ...,
        description="28x28 grayscale Chest X-Ray pixel matrix (uint8 values 0..255).",
    )


class NLPPredictRequest(BaseModel):
    """Request schema for standalone Biomedical NER and NegEx entity extraction."""

    clinical_note: str = Field(
        ...,
        min_length=3,
        description="Unstructured clinical admission or triage note text.",
    )


class RAGPredictRequest(BaseModel):
    """Request schema for standalone Hybrid Clinical RAG retrieval and Cross-Encoder reranking."""

    clinical_query: str = Field(
        ...,
        min_length=3,
        description="Clinical search query for retrieving and reranking biomedical evidence.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Number of top reranked documents to return.",
    )


class TriageRunRequest(BaseModel):
    """Request schema for executing the full multimodal LangGraph triage workflow."""

    patient_id: str = Field(
        default="PATIENT-001",
        min_length=1,
        description="Unique patient identifier.",
    )
    tabular_input: Dict[str, float] = Field(
        ...,
        description="Validated 9-feature tabular dictionary.",
    )
    vitals_24h: List[List[float]] = Field(
        ...,
        description="Validated 24x5 hourly ICU vitals array.",
    )
    xray_28x28: List[List[int]] = Field(
        ...,
        description="Validated 28x28 grayscale Chest X-Ray array.",
    )
    clinical_note: str = Field(
        ...,
        min_length=3,
        description="Validated clinical note text.",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional LangGraph checkpoint thread ID.",
    )
    auto_approve_hitl: bool = Field(
        default=False,
        description="If False, pauses at the Attending Physician HITL checkpoint (`PENDING_PHYSICIAN_APPROVAL`).",
    )


class HITLApprovalRequest(BaseModel):
    """Request schema for Attending Physician Human-in-the-Loop sign-off."""

    thread_id: str = Field(
        ...,
        min_length=1,
        description="LangGraph thread ID of the paused patient triage session.",
    )
    clinician_name: Optional[str] = Field(
        default="Attending Physician",
        description="Name or role of the approving clinician.",
    )
    clinician_notes: Optional[str] = Field(
        default=None,
        description="Optional attending physician addendum or sign-off comment.",
    )
    action: Literal["approve", "reject"] = Field(
        default="approve",
        description="Clinician sign-off action ('approve' releases orders; 'reject' halts orders).",
    )
