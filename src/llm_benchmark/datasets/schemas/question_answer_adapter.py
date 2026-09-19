"""Adapter for question and optional reference-answer records."""

from __future__ import annotations

from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import require_text


class QuestionAnswerSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.QUESTION_ANSWER

    def match_score(self, record: dict[str, Any]) -> float:
        if "question" not in record:
            return 0.0
        return 1.0 if "answer" in record else 0.9

    def extract_prompt(self, record: dict[str, Any]) -> str:
        return require_text(record.get("question"), "question")

