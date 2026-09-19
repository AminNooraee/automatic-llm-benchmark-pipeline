"""Adapter for records containing a direct prompt field."""

from __future__ import annotations

from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import require_text


class PromptSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.PROMPT

    def match_score(self, record: dict[str, Any]) -> float:
        return 1.0 if "prompt" in record else 0.0

    def extract_prompt(self, record: dict[str, Any]) -> str:
        return require_text(record.get("prompt"), "prompt")

