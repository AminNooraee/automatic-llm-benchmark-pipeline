"""Normalized inference response and run summary contracts."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_benchmark.clients.contracts import GenerationUsage


class InferenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class SampleInferenceStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ModelResponseMetadata(InferenceContract):
    status: ModelResponseStatus
    model_name: str
    latency_ms: float | None = Field(default=None, ge=0)
    attempts: int = Field(default=1, ge=1)
    finish_reason: str | None = None
    usage: GenerationUsage | None = None
    error_type: str | None = None
    error_message: str | None = None
    reused: bool = False

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ModelResponseMetadata":
        if self.status == ModelResponseStatus.SUCCESS and self.error_message is not None:
            raise ValueError("successful model metadata cannot contain an error")
        if self.status == ModelResponseStatus.ERROR and not self.error_message:
            raise ValueError("failed model metadata must contain an error message")
        return self


class InferenceMetadata(InferenceContract):
    status: SampleInferenceStatus
    base: ModelResponseMetadata
    fine_tuned: ModelResponseMetadata

    @model_validator(mode="after")
    def validate_aggregate_status(self) -> "InferenceMetadata":
        successes = sum(
            item.status == ModelResponseStatus.SUCCESS
            for item in (self.base, self.fine_tuned)
        )
        expected = (
            SampleInferenceStatus.SUCCESS
            if successes == 2
            else SampleInferenceStatus.PARTIAL
            if successes == 1
            else SampleInferenceStatus.FAILED
        )
        if self.status != expected:
            raise ValueError(f"sample status must be '{expected.value}'")
        return self


class InferenceResponse(InferenceContract):
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    base_response: str = ""
    fine_tuned_response: str = ""
    metadata: InferenceMetadata

    @model_validator(mode="after")
    def validate_response_statuses(self) -> "InferenceResponse":
        if self.metadata.base.status == ModelResponseStatus.SUCCESS:
            if not self.base_response.strip():
                raise ValueError("successful base response must not be empty")
        elif self.base_response:
            raise ValueError("failed base response must be empty")

        if self.metadata.fine_tuned.status == ModelResponseStatus.SUCCESS:
            if not self.fine_tuned_response.strip():
                raise ValueError("successful fine-tuned response must not be empty")
        elif self.fine_tuned_response:
            raise ValueError("failed fine-tuned response must be empty")
        return self


class InferenceRunResult(InferenceContract):
    artifact_path: Path
    checkpoint_path: Path
    total_samples: int = Field(ge=0)
    successful_samples: int = Field(ge=0)
    partial_samples: int = Field(ge=0)
    failed_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    attempted_samples: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "InferenceRunResult":
        if (
            self.successful_samples + self.partial_samples + self.failed_samples
            != self.total_samples
        ):
            raise ValueError("inference outcome counts must equal total_samples")
        if self.skipped_samples + self.attempted_samples != self.total_samples:
            raise ValueError("skipped plus attempted samples must equal total_samples")
        return self

