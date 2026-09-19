"""Interface for detecting and extracting prompts from record schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_benchmark.datasets.models import DatasetSchema


class SchemaAdapter(ABC):
    schema: DatasetSchema

    @abstractmethod
    def match_score(self, record: dict[str, Any]) -> float:
        """Return zero when unsupported or a confidence score in (0, 1]."""

    @abstractmethod
    def extract_prompt(self, record: dict[str, Any]) -> str:
        """Extract one prompt or raise ValueError with a clear reason."""

