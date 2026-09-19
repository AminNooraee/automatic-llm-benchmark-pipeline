"""Adapter for structured or serialized ChatML conversations."""

from __future__ import annotations

import re
from typing import Any

from llm_benchmark.datasets.models import DatasetSchema
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import extract_last_user_message, require_text


_CHATML_USER_PATTERN = re.compile(
    r"<\|im_start\|>\s*user\s*\n?(.*?)<\|im_end\|>",
    flags=re.DOTALL | re.IGNORECASE,
)


class ChatMLSchemaAdapter(SchemaAdapter):
    schema = DatasetSchema.CHATML

    def match_score(self, record: dict[str, Any]) -> float:
        if "chatml" in record or "chat" in record:
            return 1.0
        text = record.get("text")
        if isinstance(text, str) and "<|im_start|>" in text:
            return 0.95
        return 0.0

    def extract_prompt(self, record: dict[str, Any]) -> str:
        if "chat" in record:
            return extract_last_user_message(
                record.get("chat"),
                role_field="role",
                content_field="content",
                user_roles={"user"},
                field_name="chat",
            )

        field_name = "chatml" if "chatml" in record else "text"
        serialized = require_text(record.get(field_name), field_name)
        matches = [item.strip() for item in _CHATML_USER_PATTERN.findall(serialized)]
        matches = [item for item in matches if item]
        if not matches:
            raise ValueError(
                f"Field '{field_name}' contains no serialized ChatML user message"
            )
        return matches[-1]

