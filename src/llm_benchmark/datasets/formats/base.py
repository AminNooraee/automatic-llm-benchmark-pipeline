"""Interface implemented by every local file-format adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from llm_benchmark.config.models import DatasetFormat
from llm_benchmark.datasets.models import RawDatasetRecord


class FileFormatAdapter(Protocol):
    format: DatasetFormat
    extensions: tuple[str, ...]

    def read(self, path: Path) -> Iterator[RawDatasetRecord]:
        """Yield source records with human-readable row numbers."""

