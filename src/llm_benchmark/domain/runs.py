"""Run-management domain contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RunStatus(str, Enum):
    INITIALIZED = "initialized"
    DATASET_PREPARED = "dataset_prepared"
    INFERENCE_RUNNING = "inference_running"
    INFERENCE_COMPLETED = "inference_completed"
    INFERENCE_PARTIAL = "inference_partial"
    EVALUATION_RUNNING = "evaluation_running"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_PARTIAL = "evaluation_partial"
    REPORTING = "reporting"
    COMPLETED = "completed"


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str
    source_format: str
    schema_format: str
    detection_confidence: float
    detected_fields: list[str]
    sample_count: int
    source_sha256: str
    original_artifact: str
    normalized_artifact: str


class InferenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    checkpoint: str
    total_samples: int
    successful_samples: int
    partial_samples: int
    failed_samples: int
    skipped_samples: int
    attempted_samples: int


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    skipped_samples: int
    reused_samples: int
    attempted_samples: int


class ReportingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_json: str
    report_markdown: str
    samples_csv: str
    artifact_validation: str
    reproducibility: str
    generated_at: datetime


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    run_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    pipeline_version: str
    config_source: str
    config_fingerprint: str
    artifacts: dict[str, str]
    dataset: DatasetManifest | None = None
    inference: InferenceManifest | None = None
    evaluation: EvaluationManifest | None = None
    reporting: ReportingManifest | None = None


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    manifest_path: Path
    config_snapshot_path: Path
    log_path: Path
