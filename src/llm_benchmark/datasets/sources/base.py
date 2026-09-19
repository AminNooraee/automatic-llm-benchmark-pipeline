"""Dataset source abstraction for local and future remote sources."""

from __future__ import annotations

from typing import Protocol

from llm_benchmark.config.models import DatasetConfig
from llm_benchmark.datasets.models import DatasetStream


class DatasetSourceAdapter(Protocol):
    """Open a configured source as a stream of raw records.

    A future Hugging Face adapter can implement this contract without changing
    schema detection or normalization.
    """

    source_type: str

    def open(self, config: DatasetConfig) -> DatasetStream:
        """Open the source and return its format plus raw record stream."""

