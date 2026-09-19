"""Shared prompt extraction helpers for schema adapters."""

from __future__ import annotations

from typing import Any


def get_dotted_value(record: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = record
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Field '{field_name}' must contain text")
    prompt = value.strip()
    if not prompt:
        raise ValueError(f"Field '{field_name}' contains an empty prompt")
    return prompt


def extract_text_content(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return require_text(value, field_name)
    if isinstance(value, list):
        text_parts: list[str] = []
        for part in value:
            if (
                isinstance(part, dict)
                and part.get("type") in {"text", "input_text"}
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ):
                text_parts.append(part["text"].strip())
        if text_parts:
            return "\n".join(text_parts)
    raise ValueError(f"Field '{field_name}' does not contain supported text content")


def extract_last_user_message(
    messages: Any,
    *,
    role_field: str,
    content_field: str,
    user_roles: set[str],
    field_name: str,
) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Field '{field_name}' must be a non-empty message list")

    prompts: list[str] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index} in '{field_name}' must be an object")
        role = message.get(role_field)
        if isinstance(role, str) and role.lower() in user_roles:
            prompts.append(
                extract_text_content(
                    message.get(content_field),
                    f"{field_name}[{index}].{content_field}",
                )
            )
    if not prompts:
        expected = ", ".join(sorted(user_roles))
        raise ValueError(
            f"Field '{field_name}' contains no user message with role: {expected}"
        )
    return prompts[-1]

