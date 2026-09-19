"""Cross-artifact consistency validation before a run is completed."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_benchmark.config.models import AppConfig
from llm_benchmark.domain.runs import RunContext, RunManifest, RunStatus
from llm_benchmark.evaluation.artifact_store import JudgeArtifactStore
from llm_benchmark.evaluation.reader import InferenceResponseReader
from llm_benchmark.exceptions import BenchmarkError
from llm_benchmark.inference.dataset_reader import NormalizedDatasetReader
from llm_benchmark.reporting.models import BenchmarkReport, ReportGenerationResult
from llm_benchmark.runs.artifact_store import LocalArtifactStore


class ArtifactValidationError(BenchmarkError):
    """Raised when run artifacts are missing, malformed, or inconsistent."""


class ArtifactValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    validated_at: datetime
    status: str
    run_id: str
    sample_count: int = Field(ge=0)
    checked_artifacts: list[str]
    checks: dict[str, str]


class ArtifactValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_path: Path
    summary: ArtifactValidationSummary


class RunArtifactValidator:
    RELATIVE_PATH = Path("metadata/artifact_validation.json")

    def validate_and_write(
        self,
        *,
        config: AppConfig,
        context: RunContext,
        manifest: RunManifest,
        report_result: ReportGenerationResult,
    ) -> ArtifactValidationResult:
        if manifest.status != RunStatus.REPORTING:
            raise ArtifactValidationError(
                f"Artifact validation requires reporting status, found "
                f"'{manifest.status.value}'"
            )
        if manifest.dataset is None:
            raise ArtifactValidationError("Run manifest has no dataset metadata")
        if manifest.inference is None:
            raise ArtifactValidationError("Run manifest has no inference metadata")
        if manifest.evaluation is None:
            raise ArtifactValidationError("Run manifest has no evaluation metadata")

        checked: list[str] = []
        config_snapshot = self._require_file(
            context, context.config_snapshot_path, checked
        )
        self._require_file(
            context, context.run_dir / "metadata" / "environment.json", checked
        )
        self._require_file(context, context.log_path, checked)
        self._require_file(
            context,
            context.run_dir / manifest.dataset.original_artifact,
            checked,
        )
        normalized_path = self._require_file(
            context,
            context.run_dir / manifest.dataset.normalized_artifact,
            checked,
        )
        inference_path = self._require_file(
            context,
            context.run_dir / manifest.inference.artifact,
            checked,
        )
        self._require_file(
            context,
            context.run_dir / manifest.inference.checkpoint,
            checked,
        )
        evaluation_path = self._require_file(
            context,
            context.run_dir / manifest.evaluation.artifact,
            checked,
        )
        report_json_path = self._require_file(
            context, report_result.report_json_path, checked
        )
        report_markdown_path = self._require_file(
            context, report_result.report_markdown_path, checked
        )
        samples_csv_path = self._require_file(
            context, report_result.samples_csv_path, checked
        )

        samples = NormalizedDatasetReader().read(normalized_path)
        inference = InferenceResponseReader().read(inference_path)
        judge_results = list(
            JudgeArtifactStore(context.run_dir).load_results().values()
        )
        if evaluation_path != (context.run_dir / "evaluation/judge_results.jsonl").resolve():
            raise ArtifactValidationError(
                "Evaluation manifest does not reference the canonical judge artifact"
            )
        try:
            report = BenchmarkReport.model_validate_json(
                report_json_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError) as exc:
            raise ArtifactValidationError(
                f"Invalid JSON report artifact: {report_json_path}"
            ) from exc

        dataset_ids = [sample.id for sample in samples]
        inference_ids = [sample.id for sample in inference]
        judge_ids = [result.id for result in judge_results]
        report_ids = [sample.id for sample in report.per_sample_results]
        if not dataset_ids == inference_ids == judge_ids == report_ids:
            raise ArtifactValidationError(
                "Sample IDs or ordering differ across dataset, inference, "
                "evaluation, and report artifacts"
            )

        count = len(dataset_ids)
        expected_counts = {
            "dataset manifest": manifest.dataset.sample_count,
            "inference manifest": manifest.inference.total_samples,
            "evaluation manifest": manifest.evaluation.total_samples,
            "report metrics": report.aggregate_metrics.total_samples,
            "report samples": len(report.per_sample_results),
        }
        mismatches = {
            name: value for name, value in expected_counts.items() if value != count
        }
        if mismatches:
            details = ", ".join(f"{name}={value}" for name, value in mismatches.items())
            raise ArtifactValidationError(
                f"Artifact sample counts disagree with dataset count {count}: {details}"
            )

        if report.benchmark_metadata.run_id != context.run_id:
            raise ArtifactValidationError("Report run ID does not match run directory")
        if report.benchmark_metadata.config_fingerprint != config.fingerprint():
            raise ArtifactValidationError(
                "Report configuration fingerprint does not match current configuration"
            )
        if report.model_information.base.name != config.models.base.name:
            raise ArtifactValidationError("Report base model does not match configuration")
        if report.model_information.fine_tuned.name != config.models.fine_tuned.name:
            raise ArtifactValidationError(
                "Report fine-tuned model does not match configuration"
            )
        if report.model_information.judge.name != config.models.judge.name:
            raise ArtifactValidationError("Report judge model does not match configuration")

        try:
            markdown = report_markdown_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ArtifactValidationError(
                f"Unable to read Markdown report: {report_markdown_path}"
            ) from exc
        if not markdown.startswith("# Benchmark Report"):
            raise ArtifactValidationError("Markdown report is missing its title")

        try:
            with samples_csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ArtifactValidationError(
                f"Unable to parse sample CSV: {samples_csv_path}"
            ) from exc
        if len(csv_rows) != count:
            raise ArtifactValidationError(
                f"CSV sample count {len(csv_rows)} does not match dataset count {count}"
            )

        snapshot_text = config_snapshot.read_text(encoding="utf-8")
        for secret in config.secret_values():
            if secret in snapshot_text:
                raise ArtifactValidationError(
                    "Configuration snapshot contains an unredacted API key"
                )

        summary = ArtifactValidationSummary(
            validated_at=datetime.now(UTC),
            status="passed",
            run_id=context.run_id,
            sample_count=count,
            checked_artifacts=sorted(checked),
            checks={
                "artifact_files": "passed",
                "configuration_redaction": "passed",
                "cross_artifact_ids": "passed",
                "cross_artifact_counts": "passed",
                "report_metadata": "passed",
                "report_formats": "passed",
            },
        )
        path = LocalArtifactStore(context.run_dir).write_json(
            self.RELATIVE_PATH,
            summary.model_dump(mode="json"),
        )
        return ArtifactValidationResult(artifact_path=path, summary=summary)

    @staticmethod
    def _require_file(
        context: RunContext,
        path: Path,
        checked: list[str],
    ) -> Path:
        root = context.run_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ArtifactValidationError(f"Artifact escapes run directory: {path}")
        if not resolved.is_file():
            raise ArtifactValidationError(f"Required run artifact is missing: {path}")
        checked.append(resolved.relative_to(root).as_posix())
        return resolved
