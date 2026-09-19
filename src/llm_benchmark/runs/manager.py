"""Creation of isolated, non-overwriting benchmark run folders."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from llm_benchmark import __version__
from llm_benchmark.config.models import AppConfig
from llm_benchmark.datasets.models import DatasetIngestionResult
from llm_benchmark.domain.runs import (
    DatasetManifest,
    EvaluationManifest,
    InferenceManifest,
    ReportingManifest,
    RunContext,
    RunManifest,
    RunStatus,
)
from llm_benchmark.exceptions import RunInitializationError, RunLifecycleError
from llm_benchmark.runs.artifact_store import LocalArtifactStore
from llm_benchmark.runs.manifest import manifest_payload
from llm_benchmark.runs.metadata import write_environment_metadata

if TYPE_CHECKING:
    from llm_benchmark.evaluation.models import JudgeRunResult
    from llm_benchmark.inference.models import InferenceRunResult
    from llm_benchmark.reporting.models import ReportGenerationResult


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RunManager:
    """Initialize and update local benchmark run metadata."""

    def initialize(
        self,
        config: AppConfig,
        config_source: str | Path,
        run_id: str | None = None,
    ) -> RunContext:
        selected_run_id = run_id or self._generate_run_id(config)
        self._validate_run_id(selected_run_id)

        runs_root = config.output.runs_dir.resolve()
        run_dir = (runs_root / selected_run_id).resolve()
        if run_dir.parent != runs_root:
            raise RunInitializationError("Resolved run directory escapes runs_dir")

        try:
            runs_root.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(exist_ok=False)
            (run_dir / "responses").mkdir()
            (run_dir / "logs").mkdir()
            (run_dir / "dataset").mkdir()
            (run_dir / "inference").mkdir()
            (run_dir / "evaluation").mkdir()
            (run_dir / "reports").mkdir()
            (run_dir / "metadata").mkdir()
        except FileExistsError as exc:
            raise RunInitializationError(
                f"Run directory already exists and will not be overwritten: {run_dir}"
            ) from exc
        except OSError as exc:
            raise RunInitializationError(
                f"Unable to initialize run directory: {run_dir}"
            ) from exc

        context = RunContext(
            run_id=selected_run_id,
            run_dir=run_dir,
            manifest_path=run_dir / "manifest.json",
            config_snapshot_path=run_dir / "config.snapshot.json",
            log_path=run_dir / "logs" / "run.jsonl",
        )
        artifacts = {
            "config_snapshot": "config.snapshot.json",
            "dataset": "dataset",
            "evaluation": "evaluation",
            "inference": "inference",
            "log": "logs/run.jsonl",
            "manifest": "manifest.json",
            "metadata": "metadata",
            "reports": "reports",
            "responses": "responses",
        }
        created_at = datetime.now(UTC)
        manifest = RunManifest(
            run_id=selected_run_id,
            status=RunStatus.INITIALIZED,
            created_at=created_at,
            updated_at=created_at,
            pipeline_version=__version__,
            config_source=str(Path(config_source).expanduser().resolve()),
            config_fingerprint=config.fingerprint(),
            artifacts=artifacts,
        )

        store = LocalArtifactStore(run_dir)
        store.write_json("config.snapshot.json", config.sanitized_dict())
        store.write_text("logs/run.jsonl", "")
        write_environment_metadata(run_dir, config)
        store.write_json("manifest.json", manifest_payload(manifest))
        return context

    def open_existing(self, config: AppConfig, run_dir: str | Path) -> RunContext:
        runs_root = config.output.runs_dir.resolve()
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        if resolved_run_dir.parent != runs_root:
            raise RunInitializationError(
                f"Resume run must be a direct child of configured runs_dir: {runs_root}"
            )
        manifest_path = resolved_run_dir / "manifest.json"
        manifest = self._read_manifest(manifest_path)
        if manifest.run_id != resolved_run_dir.name:
            raise RunInitializationError(
                "Run manifest id does not match the resume directory name"
            )
        if manifest.config_fingerprint != config.fingerprint():
            raise RunInitializationError(
                "Current configuration does not match the resumed run fingerprint"
            )
        self._validate_resume_layout(resolved_run_dir, manifest)
        return RunContext(
            run_id=manifest.run_id,
            run_dir=resolved_run_dir,
            manifest_path=manifest_path,
            config_snapshot_path=resolved_run_dir / "config.snapshot.json",
            log_path=resolved_run_dir / "logs" / "run.jsonl",
        )

    def mark_dataset_prepared(
        self, context: RunContext, result: DatasetIngestionResult
    ) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(current, {RunStatus.INITIALIZED}, "prepare dataset")

        dataset_manifest = DatasetManifest(
            source_path=str(result.source_path),
            source_format=result.source_format.value,
            schema_format=result.schema_format.value,
            detection_confidence=result.detection_confidence,
            detected_fields=result.detected_fields,
            sample_count=result.sample_count,
            source_sha256=result.source_sha256,
            original_artifact=result.original_artifact,
            normalized_artifact=result.normalized_artifact,
        )
        updated = current.model_copy(
            update={
                "status": RunStatus.DATASET_PREPARED,
                "updated_at": datetime.now(UTC),
                "dataset": dataset_manifest,
            }
        )
        LocalArtifactStore(context.run_dir).write_json(
            "manifest.json", manifest_payload(updated)
        )
        return updated

    def assert_can_prepare_dataset(self, context: RunContext) -> None:
        current = self._read_manifest(context.manifest_path)
        self._require_status(current, {RunStatus.INITIALIZED}, "prepare dataset")

    def mark_inference_started(self, context: RunContext) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(
            current,
            {
                RunStatus.DATASET_PREPARED,
                RunStatus.INFERENCE_RUNNING,
                RunStatus.INFERENCE_COMPLETED,
                RunStatus.INFERENCE_PARTIAL,
                RunStatus.EVALUATION_RUNNING,
                RunStatus.EVALUATION_COMPLETED,
                RunStatus.EVALUATION_PARTIAL,
                RunStatus.REPORTING,
                RunStatus.COMPLETED,
            },
            "start inference",
        )
        if current.dataset is None:
            raise RunLifecycleError("Cannot start inference without dataset metadata")
        updated = current.model_copy(
            update={
                "status": RunStatus.INFERENCE_RUNNING,
                "updated_at": datetime.now(UTC),
            }
        )
        self._write_manifest(context, updated)
        return updated

    def mark_inference_finished(
        self, context: RunContext, result: "InferenceRunResult"
    ) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(
            current, {RunStatus.INFERENCE_RUNNING}, "finish inference"
        )
        status = (
            RunStatus.INFERENCE_COMPLETED
            if result.partial_samples == 0 and result.failed_samples == 0
            else RunStatus.INFERENCE_PARTIAL
        )
        inference = InferenceManifest(
            artifact=self._relative_artifact(context, result.artifact_path),
            checkpoint=self._relative_artifact(context, result.checkpoint_path),
            total_samples=result.total_samples,
            successful_samples=result.successful_samples,
            partial_samples=result.partial_samples,
            failed_samples=result.failed_samples,
            skipped_samples=result.skipped_samples,
            attempted_samples=result.attempted_samples,
        )
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(UTC),
                "inference": inference,
            }
        )
        self._write_manifest(context, updated)
        return updated

    def mark_evaluation_started(self, context: RunContext) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(
            current,
            {
                RunStatus.INFERENCE_COMPLETED,
                RunStatus.INFERENCE_PARTIAL,
                RunStatus.EVALUATION_RUNNING,
                RunStatus.EVALUATION_COMPLETED,
                RunStatus.EVALUATION_PARTIAL,
                RunStatus.COMPLETED,
            },
            "start evaluation",
        )
        if current.inference is None:
            raise RunLifecycleError(
                "Cannot start evaluation without inference metadata"
            )
        updated = current.model_copy(
            update={
                "status": RunStatus.EVALUATION_RUNNING,
                "updated_at": datetime.now(UTC),
            }
        )
        self._write_manifest(context, updated)
        return updated

    def mark_evaluation_finished(
        self, context: RunContext, result: "JudgeRunResult"
    ) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(
            current, {RunStatus.EVALUATION_RUNNING}, "finish evaluation"
        )
        status = (
            RunStatus.EVALUATION_COMPLETED
            if result.failed_samples == 0 and result.skipped_samples == 0
            else RunStatus.EVALUATION_PARTIAL
        )
        evaluation = EvaluationManifest(
            artifact=self._relative_artifact(context, result.artifact_path),
            total_samples=result.total_samples,
            successful_samples=result.successful_samples,
            failed_samples=result.failed_samples,
            skipped_samples=result.skipped_samples,
            reused_samples=result.reused_samples,
            attempted_samples=result.attempted_samples,
        )
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(UTC),
                "evaluation": evaluation,
            }
        )
        self._write_manifest(context, updated)
        return updated

    def mark_reporting_started(self, context: RunContext) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(
            current,
            {
                RunStatus.EVALUATION_COMPLETED,
                RunStatus.EVALUATION_PARTIAL,
                RunStatus.REPORTING,
                RunStatus.COMPLETED,
            },
            "start reporting",
        )
        if current.evaluation is None:
            raise RunLifecycleError(
                "Cannot start reporting without evaluation metadata"
            )
        updated = current.model_copy(
            update={
                "status": RunStatus.REPORTING,
                "updated_at": datetime.now(UTC),
            }
        )
        self._write_manifest(context, updated)
        return updated

    def mark_reporting_finished(
        self,
        context: RunContext,
        result: "ReportGenerationResult",
        *,
        validation_path: Path,
        reproducibility_path: Path,
    ) -> RunManifest:
        current = self._read_manifest(context.manifest_path)
        self._require_status(current, {RunStatus.REPORTING}, "finish reporting")
        reporting = ReportingManifest(
            report_json=self._relative_artifact(context, result.report_json_path),
            report_markdown=self._relative_artifact(
                context, result.report_markdown_path
            ),
            samples_csv=self._relative_artifact(context, result.samples_csv_path),
            artifact_validation=self._relative_artifact(
                context, validation_path
            ),
            reproducibility=self._relative_artifact(
                context, reproducibility_path
            ),
            generated_at=result.report.benchmark_metadata.generated_at,
        )
        updated = current.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "updated_at": datetime.now(UTC),
                "reporting": reporting,
            }
        )
        self._write_manifest(context, updated)
        return updated

    def get_manifest(self, context: RunContext) -> RunManifest:
        return self._read_manifest(context.manifest_path)

    @staticmethod
    def _read_manifest(path: Path) -> RunManifest:
        try:
            return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RunInitializationError(f"Unable to read run manifest: {path}") from exc

    @staticmethod
    def _write_manifest(context: RunContext, manifest: RunManifest) -> None:
        LocalArtifactStore(context.run_dir).write_json(
            "manifest.json", manifest_payload(manifest)
        )

    @staticmethod
    def _relative_artifact(context: RunContext, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(context.run_dir.resolve())
        except ValueError as exc:
            raise RunInitializationError(
                f"Run artifact escapes the run directory: {path}"
            ) from exc
        return relative.as_posix()

    @staticmethod
    def _require_status(
        manifest: RunManifest,
        allowed: set[RunStatus],
        action: str,
    ) -> None:
        if manifest.status not in allowed:
            expected = ", ".join(sorted(status.value for status in allowed))
            raise RunLifecycleError(
                f"Cannot {action} while run status is '{manifest.status.value}'; "
                f"expected one of: {expected}"
            )

    @staticmethod
    def _validate_resume_layout(run_dir: Path, manifest: RunManifest) -> None:
        required = [
            run_dir / "config.snapshot.json",
            run_dir / "logs" / "run.jsonl",
            run_dir / "metadata" / "environment.json",
        ]
        if manifest.status != RunStatus.INITIALIZED:
            if manifest.dataset is None:
                raise RunInitializationError(
                    "Resumed run is missing dataset manifest metadata"
                )
            required.append(run_dir / manifest.dataset.normalized_artifact)
        if manifest.status in {
            RunStatus.INFERENCE_COMPLETED,
            RunStatus.INFERENCE_PARTIAL,
            RunStatus.EVALUATION_RUNNING,
            RunStatus.EVALUATION_COMPLETED,
            RunStatus.EVALUATION_PARTIAL,
            RunStatus.REPORTING,
            RunStatus.COMPLETED,
        }:
            if manifest.inference is None:
                raise RunInitializationError(
                    "Resumed run is missing inference manifest metadata"
                )
            required.append(run_dir / manifest.inference.artifact)
        if manifest.status in {
            RunStatus.EVALUATION_COMPLETED,
            RunStatus.EVALUATION_PARTIAL,
            RunStatus.REPORTING,
            RunStatus.COMPLETED,
        }:
            if manifest.evaluation is None:
                raise RunInitializationError(
                    "Resumed run is missing evaluation manifest metadata"
                )
            required.append(run_dir / manifest.evaluation.artifact)
        if manifest.status == RunStatus.COMPLETED:
            if manifest.reporting is None:
                raise RunInitializationError(
                    "Completed run is missing reporting manifest metadata"
                )
            required.extend(
                [
                    run_dir / manifest.reporting.report_json,
                    run_dir / manifest.reporting.report_markdown,
                    run_dir / manifest.reporting.samples_csv,
                    run_dir / manifest.reporting.artifact_validation,
                    run_dir / manifest.reporting.reproducibility,
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RunInitializationError(
                "Resumed run is missing required artifacts: " + ", ".join(missing)
            )

    @staticmethod
    def _generate_run_id(config: AppConfig) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}-{config.fingerprint()[:8]}"

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
            raise RunInitializationError(
                "run_id must contain only letters, numbers, dots, underscores, and hyphens"
            )
