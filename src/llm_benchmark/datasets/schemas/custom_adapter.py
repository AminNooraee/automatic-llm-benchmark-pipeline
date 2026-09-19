"""Configuration-driven prompt column adapter."""

from __future__ import annotations

from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import get_dotted_value, require_text


class CustomColumnSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.CUSTOM

    def __init__(self, prompt_path: str) -> None:
        self.prompt_path = prompt_path

    def match_score(self, record: dict[str, Any]) -> float:
        found, _ = get_dotted_value(record, self.prompt_path)
        return 1.0 if found else 0.0

    def extract_prompt(self, record: dict[str, Any]) -> str:
        found, value = get_dotted_value(record, self.prompt_path)
        if not found:
            raise ValueError(
                f"Configured prompt column '{self.prompt_path}' is missing"
            )
        return require_text(value, self.prompt_path)

