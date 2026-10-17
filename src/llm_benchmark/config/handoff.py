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


class ReportLoraContext(_HandoffModel):
    rank: int = Field(ge=0)
    alpha: int = Field(ge=0)
    dropout: float = Field(ge=0, le=1)


class ReportTrainingContext(_HandoffModel):
    model_name: str = Field(min_length=1)
    epochs: float = Field(ge=0)
    learning_rate: float = Field(ge=0)
    batch_size: int = Field(ge=1)
    cutoff_len: int = Field(ge=1)
    lora: ReportLoraContext | None = None


class ReportContext(_HandoffModel):
    """Sanitized, provider-neutral training context from Project #1.

    Carries only user-editable training facts and the measured training
    duration. It is a separate payload from the endpoint handoff and is never
    required: older pipelines that do not produce it must still run.
    """

    schema_version: Literal[1]
    training_duration_seconds: float | None = Field(default=None, ge=0)
    training: ReportTrainingContext


def load_report_context(path: str | Path) -> ReportContext:
    context_path = Path(path).expanduser().resolve()
    if not context_path.is_file():
        raise ConfigurationError(
            f"Project #1 report context must be a readable regular JSON file: "
            f"{context_path}"
        )
    try:
        raw = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Unable to parse Project #1 report context: {context_path}"
        ) from exc
    try:
        return ReportContext.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ConfigurationError(
            f"Project #1 report context validation failed: {details}"
        ) from exc


def load_optional_report_context(
    path: str | Path | None,
) -> ReportContext | None:
    """Load the report context when a path is supplied, otherwise return None."""

    if path is None:
        return None
    return load_report_context(path)
