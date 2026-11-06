"""Common abstract interface and custom exceptions for all healthcare AI inference models."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
PreprocessedT = TypeVar("PreprocessedT")
OutputT = TypeVar("OutputT")


class InferenceError(Exception):
    """Base exception for inference pipeline errors."""


class InputValidationError(InferenceError, ValueError):
    """Raised when model input fails schema, shape, or domain validation."""


class ModelLoadError(InferenceError, RuntimeError):
    """Raised when a model checkpoint or resource fails to load."""


class BaseInferenceModel(ABC, Generic[InputT, PreprocessedT, OutputT]):
    """Abstract base class enforcing a consistent validation, preprocessing, and inference contract."""

    model_name: str = "base_inference_model"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"a_project.models.{self.model_name}")

    @abstractmethod
    def validate_input(self, raw_input: InputT) -> None:
        """Validates raw input types, shapes, and domain constraints before preprocessing."""

    @abstractmethod
    def preprocess(self, raw_input: InputT) -> PreprocessedT:
        """Transforms validated raw input into the exact tensor/dataframe format used in training."""

    @abstractmethod
    def predict(self, preprocessed_input: PreprocessedT) -> OutputT:
        """Runs core model forward pass / inference and formats the structured prediction dict."""

    def run_inference(self, raw_input: InputT) -> OutputT:
        """Executes the full validate -> preprocess -> predict pipeline with logging and error handling."""
        start_ts = time.perf_counter()
        try:
            self.validate_input(raw_input)
            processed = self.preprocess(raw_input)
            result = self.predict(processed)
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self.logger.debug("Completed %s inference in %.2f ms", self.model_name, elapsed_ms)
            return result
        except InferenceError:
            raise
        except Exception as exc:
            self.logger.exception("Unexpected failure during %s inference", self.model_name)
            raise InferenceError(f"Inference failed in {self.model_name}: {exc}") from exc

    def __call__(self, raw_input: InputT) -> OutputT:
        """Callable alias for `run_inference`."""
        return self.run_inference(raw_input)
