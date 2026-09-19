"""Adapter for OpenAI-style messages arrays."""

from __future__ import annotations

from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import extract_last_user_message


class OpenAIMessagesSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.OPENAI_MESSAGES

    def match_score(self, record: dict[str, Any]) -> float:
        return 1.0 if "messages" in record else 0.0

    def extract_prompt(self, record: dict[str, Any]) -> str:
        return extract_last_user_message(
            record.get("messages"),
            role_field="role",
            content_field="content",
            user_roles={"user"},
            field_name="messages",
        )

