"""Typed configuration models for the benchmark pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)


class StrictConfigModel(BaseModel):
    """Base configuration model that rejects unknown settings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class DatasetFormat(str, Enum):
    AUTO = "auto"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class DatasetColumnsConfig(StrictConfigModel):
    prompt: str = Field(min_length=1)
    id: str | None = None

    @field_validator("prompt", "id")
    @classmethod
    def validate_column_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        segments = value.split(".")
        if any(not segment.strip() for segment in segments):
            raise ValueError("column mappings must use non-empty dotted path segments")
        return ".".join(segment.strip() for segment in segments)


class DatasetConfig(StrictConfigModel):
    path: Path
    format: DatasetFormat = DatasetFormat.AUTO
    columns: DatasetColumnsConfig | None = None


class EndpointConfig(StrictConfigModel):
    base_url: AnyHttpUrl
    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "model_name"),
    )
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(
        gt=0,
        le=3600,
        validation_alias=AliasChoices("timeout_seconds", "timeout"),
    )
    generation_parameters: dict[str, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "generation_parameters", "request_parameters"
        ),
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.query or value.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        if value.username or value.password:
            raise ValueError("base_url must not contain embedded credentials")
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("generation_parameters")
    @classmethod
    def reject_reserved_generation_parameters(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        reserved = {"messages", "model", "stream"}
        invalid = sorted(reserved.intersection(value))
        if invalid:
            names = ", ".join(invalid)
            raise ValueError(f"generation_parameters contains reserved fields: {names}")
        if "n" in value and value["n"] != 1:
            raise ValueError("generation parameter 'n' must be 1")
        return value

    @property
    def model_name(self) -> str:
        """Compatibility accessor for the Phase 1 configuration name."""

        return self.name

    @property
    def request_parameters(self) -> dict[str, JsonValue]:
        """Compatibility accessor for the Phase 1 configuration name."""

        return self.generation_parameters


class ModelsConfig(StrictConfigModel):
    base: EndpointConfig
    fine_tuned: EndpointConfig
    judge: EndpointConfig


class EvaluationCriterion(StrictConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    weight: float = Field(gt=0, le=1)
    minimum: float = 1
    maximum: float = 5

    @model_validator(mode="after")
    def validate_score_range(self) -> "EvaluationCriterion":
        if self.maximum <= self.minimum:
            raise ValueError("maximum must be greater than minimum")
        return self


class EvaluationConfig(StrictConfigModel):
    prompt_template: Path
    criteria: list[EvaluationCriterion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_criteria(self) -> "EvaluationConfig":
        names = [criterion.name for criterion in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("evaluation criterion names must be unique")

        total_weight = sum(criterion.weight for criterion in self.criteria)
        if not math.isclose(total_weight, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("evaluation criterion weights must sum to 1.0")
        return self


class RuntimeConfig(StrictConfigModel):
    concurrency: int = Field(default=1, ge=1, le=256)
    max_retries: int = Field(default=2, ge=0, le=20)
    retry_backoff_seconds: float = Field(default=0.5, ge=0, le=300)
    random_seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class OutputConfig(StrictConfigModel):
    runs_dir: Path = Path("runs")


class AppConfig(StrictConfigModel):
    version: Literal[1] = 1
    dataset: DatasetConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def sanitized_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration with all API keys redacted."""

        data = self.model_dump(mode="json")
        model_data = data["models"]
        if isinstance(model_data, dict):
            for role in ("base", "fine_tuned", "judge"):
                endpoint = model_data.get(role)
                if isinstance(endpoint, dict) and endpoint.get("api_key") is not None:
                    endpoint["api_key"] = "***REDACTED***"
        return data

    def secret_values(self) -> tuple[str, ...]:
        """Return configured secret values for log redaction."""

        secrets: list[str] = []
        for endpoint in (
            self.models.base,
            self.models.fine_tuned,
            self.models.judge,
        ):
            if endpoint.api_key is not None:
                value = endpoint.api_key.get_secret_value()
                if value:
                    secrets.append(value)
        return tuple(secrets)

    def fingerprint(self) -> str:
        """Return a stable short hash of the sanitized resolved configuration."""

        serialized = json.dumps(
            self.sanitized_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
