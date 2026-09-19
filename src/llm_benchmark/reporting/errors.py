"""Report generation errors."""

from llm_benchmark.exceptions import BenchmarkError


class ReportGenerationError(BenchmarkError):
    """Raised when required report metadata or artifacts are unavailable."""
