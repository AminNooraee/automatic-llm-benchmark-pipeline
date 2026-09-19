"""Typed benchmark report contracts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llm_benchmark.metrics.models import AggregateMetrics, ResolvedSampleResult


class ReportContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkMetadata(ReportContract):
    run_id: str
    generated_at: datetime
    pipeline_version: str
    config_fingerprint: str


class ModelInformation(ReportContract):
    name: str
    base_url: str
    timeout_seconds: float
    generation_parameters: dict[str, object]


class ComparedModels(ReportContract):
    base: ModelInformation
    fine_tuned: ModelInformation
    judge: ModelInformation


class DatasetInformation(ReportContract):
    source_path: str
    source_format: str
    schema_format: str
    sample_count: int
    source_sha256: str
    detection_confidence: float
    detected_fields: list[str]


class CriterionInformation(ReportContract):
    name: str
    description: str
    weight: float
    minimum: float
    maximum: float


class BenchmarkReport(ReportContract):
    benchmark_metadata: BenchmarkMetadata
    model_information: ComparedModels
    dataset_information: DatasetInformation
    evaluation_criteria: list[CriterionInformation]
    aggregate_metrics: AggregateMetrics
    per_sample_results: list[ResolvedSampleResult]


class ReportGenerationResult(ReportContract):
    report_json_path: Path
    report_markdown_path: Path
    samples_csv_path: Path
    report: BenchmarkReport
