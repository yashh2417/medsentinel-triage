"""Biomedical Transformer NER and NegEx clinical negation/allergy extraction module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import pipeline as hf_pipeline

from medsentinel_triage.base import BaseInferenceModel
from medsentinel_triage.config import (
    DOCUMENTED_ALLERGY_KEYWORDS,
    NEGATION_PATTERNS,
    NEGATION_WINDOW_CHARS,
    resolve_hf_pipeline_device,
    resolve_torch_device,
)
from medsentinel_triage.preprocessing import validate_non_empty_text


def _expand_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int, str]:
    """Expands subword character offsets `[start, end)` to full alphanumeric word boundaries."""
    n = len(text)
    s = max(0, min(start, n))
    e = max(s, min(end, n))
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] == "-"):
        s -= 1
    while e < n and (text[e].isalnum() or text[e] == "-"):
        e += 1
    phrase = " ".join(text[s:e].strip(" ,.;:()[]{}").split())
    return s, e, phrase


class HuggingFaceClinicalNLP(BaseInferenceModel[str, str, Dict[str, Any]]):
    """Biomedical Transformer NER (`d4data/biomedical-ner-all`) + NegEx Clinical Scope Analyzer."""

    model_name: str = "biomedical_clinical_ner"
    NEGATION_PATTERNS: Tuple[str, ...] = NEGATION_PATTERNS

    def __init__(
        self,
        model_name: Union[str, Path] = "./biomedical_ner_model",
        fallback_hub_id: str = "d4data/biomedical-ner-all",
        device: Optional[Union[int, str]] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device_id = resolve_hf_pipeline_device(resolve_torch_device())
        else:
            device_id = device

        model_path = Path(model_name)
        target_model = str(model_path) if model_path.exists() else fallback_hub_id
        self.logger.info("Loading Biomedical NER model from: %s (device=%s)", target_model, device_id)
        self.ner_pipe = hf_pipeline(
            "ner",
            model=target_model,
            aggregation_strategy="simple",
            device=device_id,
        )

    def validate_input(self, raw_input: str) -> None:
        validate_non_empty_text(raw_input, field_name="clinical_note")

    def preprocess(self, raw_input: str) -> str:
        return validate_non_empty_text(raw_input, field_name="clinical_note")

    def extract_clinical_entities(self, note_text: str) -> Dict[str, Any]:
        """Extracts affirmed/negated clinical findings, medications, diagnostic procedures, and allergies."""
        cleaned_note = self.preprocess(note_text)
        raw_entities = self.ner_pipe(cleaned_note)

        affirmed_problems: List[str] = []
        negated_problems: List[str] = []
        medications: List[str] = []
        diagnostic_procedures: List[str] = []

        note_lower = cleaned_note.lower()
        allergies = [
            alg.capitalize()
            for alg in DOCUMENTED_ALLERGY_KEYWORDS
            if alg in note_lower and ("allerg" in note_lower or "anaphylaxis" in note_lower)
        ]
        allergy_lower_set = {a.lower() for a in allergies}

        seen_spans: set[Tuple[str, int, int]] = set()
        for ent in raw_entities:
            label = str(ent.get("entity_group", "")).upper()
            raw_start = int(ent.get("start", 0))
            raw_end = int(ent.get("end", raw_start))

            if raw_end > raw_start:
                start, end, word = _expand_to_word_boundaries(cleaned_note, raw_start, raw_end)
            else:
                word = str(ent.get("word", "")).replace("##", "").strip()
                start, end = raw_start, raw_end

            word_clean = word.lower()
            if len(word_clean) < 3:
                continue

            span_key = (label, start, end)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            context_window = note_lower[max(0, start - NEGATION_WINDOW_CHARS) : start]
            is_negated = any(pat in context_window for pat in self.NEGATION_PATTERNS)

            if label in {"SIGN_SYMPTOM", "DISEASE_DISORDER"}:
                if is_negated:
                    negated_problems.append(word_clean)
                else:
                    affirmed_problems.append(word_clean)
            elif label in {"MEDICATION", "THERAPEUTIC_PROCEDURE"}:
                # Skip substances that are documented allergies rather than active medications
                if word_clean in allergy_lower_set and (
                    "allerg" in context_window or "anaphylaxis" in note_lower[max(0, start - 30) : min(len(note_lower), end + 30)]
                ):
                    continue
                medications.append(word_clean)
            elif label in {"DIAGNOSTIC_PROCEDURE", "LAB_VALUE"}:
                diagnostic_procedures.append(word_clean)

        return {
            "affirmed_clinical_findings": sorted(list(set(affirmed_problems))),
            "negated_clinical_findings": sorted(list(set(negated_problems))),
            "extracted_medications": sorted(list(set(medications))),
            "diagnostic_procedures": sorted(list(set(diagnostic_procedures))),
            "documented_allergies": allergies if allergies else ["NKDA"],
        }

    def predict(self, preprocessed_input: str) -> Dict[str, Any]:
        return self.extract_clinical_entities(preprocessed_input)
