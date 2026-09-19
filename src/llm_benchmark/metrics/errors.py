"""Metric calculation errors."""

from llm_benchmark.exceptions import BenchmarkError


class MetricsError(BenchmarkError):
    """Raised when benchmark artifacts cannot be mapped consistently."""
