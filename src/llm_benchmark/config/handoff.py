"""Adapter for Project #1's provider-neutral endpoint handoff contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from llm_benchmark.exceptions import ConfigurationError


class _HandoffModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class _ModelIdentity(_HandoffModel):
    name: str = Field(min_length=1)


class _Models(_HandoffModel):
    base: _ModelIdentity
    fine_tuned: _ModelIdentity


class Project1Handoff(_HandoffModel):
    schema_version: Literal[1]
    status: Literal["ready"]
    api: Literal["openai-compatible"]
    base_url: AnyHttpUrl
    models: _Models
    provider: str | None = None
    container_name: str | None = None
    restart_policy: str | None = None

    @field_validator("base_url")
    @classmethod
    def reject_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        return value

    @model_validator(mode="after")
    def validate_current_project1_shape(self) -> "Project1Handoff":
        direct_fields = (self.container_name, self.restart_policy)
        if self.provider is not None and any(item is not None for item in direct_fields):
            raise ValueError("handoff must be either gateway or direct-serving shape")
        if self.provider is None and not all(item is not None for item in direct_fields):
            raise ValueError(
                "handoff must include provider or direct-serving lifecycle fields"
            )
        return self


def load_project1_handoff(path: str | Path) -> Project1Handoff:
    handoff_path = Path(path).expanduser().resolve()
    if not handoff_path.is_file():
        raise ConfigurationError(
            f"Project #1 handoff must be a readable regular JSON file: {handoff_path}"
        )
    try:
        raw = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Unable to parse Project #1 handoff: {handoff_path}"
        ) from exc
    try:
        return Project1Handoff.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ConfigurationError(
            f"Project #1 handoff validation failed: {details}"
        ) from exc


def overlay_handoff(raw: dict[str, Any], handoff: Project1Handoff) -> None:
    models = raw.get("models")
    if not isinstance(models, dict):
        return
    for role, name in (
        ("base", handoff.models.base.name),
        ("fine_tuned", handoff.models.fine_tuned.name),
    ):
        endpoint = models.get(role)
        if isinstance(endpoint, dict):
            endpoint["base_url"] = str(handoff.base_url)
            endpoint["name"] = name
            endpoint.pop("model_name", None)
