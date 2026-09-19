"""Adapter for instruction records with optional input context."""

from __future__ import annotations

from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import require_text


class InstructionSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.INSTRUCTION

    def match_score(self, record: dict[str, Any]) -> float:
        if "instruction" not in record:
            return 0.0
        return 1.0 if "input" in record else 0.9

    def extract_prompt(self, record: dict[str, Any]) -> str:
        instruction = require_text(record.get("instruction"), "instruction")
        input_value = record.get("input")
        if input_value is None or input_value == "":
            return instruction
        input_text = require_text(input_value, "input")
        return f"{instruction}\n\n{input_text}"

