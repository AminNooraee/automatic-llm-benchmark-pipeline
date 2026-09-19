"""Data contracts shared by dataset sources, schema adapters, and consumers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_benchmark.config.models import DatasetFormat


class DatasetSchema(str, Enum):
    PROMPT = "prompt"
    QUESTION_ANSWER = "question_answer"
    INSTRUCTION = "instruction_input"
    OPENAI_MESSAGES = "openai_messages"
    CHATML = "chatml"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom_columns"


@dataclass(frozen=True, slots=True)
class RawDatasetRecord:
    row_number: int
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetStream:
    source_path: Path
    source_format: DatasetFormat
    records: Iterable[RawDatasetRecord]


class BenchmarkSample(BaseModel):
    """The only dataset representation consumed by later pipeline phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)

    @field_validator("id", "prompt")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty or whitespace")
        return normalized


@dataclass(frozen=True, slots=True)
class DatasetDetection:
    schema: DatasetSchema
    confidence: float
    detected_fields: tuple[str, ...]
    sampled_records: int


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    detection: DatasetDetection
    samples: Iterable[BenchmarkSample]


class DatasetIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: Path
    source_format: DatasetFormat
    schema_format: DatasetSchema
    detection_confidence: float = Field(ge=0, le=1)
    detected_fields: list[str]
    sample_count: int = Field(ge=0)
    source_sha256: str
    original_artifact: str
    normalized_artifact: str

