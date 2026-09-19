"""Adapter for ShareGPT from/value conversation records."""

from __future__ import annotations

from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import extract_last_user_message


class ShareGPTSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.SHAREGPT

    def match_score(self, record: dict[str, Any]) -> float:
        return 1.0 if "conversations" in record else 0.0

    def extract_prompt(self, record: dict[str, Any]) -> str:
        return extract_last_user_message(
            record.get("conversations"),
            role_field="from",
            content_field="value",
            user_roles={"human", "user"},
            field_name="conversations",
        )

