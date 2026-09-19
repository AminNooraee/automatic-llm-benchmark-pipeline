"""Evaluation-layer errors."""

from llm_benchmark.exceptions import BenchmarkError


class JudgeError(BenchmarkError):
    """Base class for judge evaluation failures."""


class JudgeArtifactError(JudgeError):
    """Raised when inference inputs or judge artifacts are invalid."""


class JudgePromptError(JudgeError):
    """Raised when a judge prompt template cannot be loaded or rendered."""


class JudgeOutputError(JudgeError):
    """Raised when a judge response is not valid structured JSON."""
