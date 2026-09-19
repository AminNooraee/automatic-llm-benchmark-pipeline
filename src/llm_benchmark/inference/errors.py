"""Inference-layer errors."""

from llm_benchmark.exceptions import BenchmarkError


class InferenceError(BenchmarkError):
    """Base class for fatal inference orchestration errors."""


class InferenceArtifactError(InferenceError):
    """Raised when normalized inputs or inference artifacts are invalid."""

