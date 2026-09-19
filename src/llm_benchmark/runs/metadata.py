"""Environment capture and reproducibility metadata for benchmark runs."""

from __future__ import annotations

import hashlib
import os
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from llm_benchmark import __version__
from llm_benchmark.config.models import AppConfig
from llm_benchmark.domain.runs import RunContext, RunManifest
from llm_benchmark.exceptions import RunInitializationError
from llm_benchmark.reporting.models import ReportGenerationResult
from llm_benchmark.runs.artifact_store import LocalArtifactStore


class MetadataContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentMetadata(MetadataContract):
    schema_version: int = 1
    captured_at: datetime
    pipeline_version: str
    config_fingerprint: str
    python_version: str
    python_implementation: str
    operating_system: str
    operating_system_release: str
    machine: str
    processor: str
    cpu_count: int | None = Field(default=None, ge=1)
    timezone: str
    dependencies: dict[str, str]


class ArtifactDigest(MetadataContract):
    sha256: str
    size_bytes: int = Field(ge=0)


class ReproducibilityMetadata(MetadataContract):
    schema_version: int = 1
    generated_at: datetime
    run_id: str
    pipeline_version: str
    config_fingerprint: str
    random_seed: int
    concurrency: int
    max_retries: int
    retry_backoff_seconds: float
    dataset_source_sha256: str
    prompt_template_path: str
    prompt_template_sha256: str
    models: dict[str, object]
    artifacts: dict[str, ArtifactDigest]


def write_environment_metadata(run_dir: Path, config: AppConfig) -> Path:
    dependencies: dict[str, str] = {}
    for distribution in (
        "automatic-llm-benchmark-pipeline",
        "httpx",
        "pydantic",
        "pyarrow",
        "PyYAML",
    ):
        try:
            dependencies[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependencies[distribution] = "not-installed"

    captured = EnvironmentMetadata(
        captured_at=datetime.now(UTC),
        pipeline_version=__version__,
        config_fingerprint=config.fingerprint(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        machine=platform.machine() or "unknown",
        processor=platform.processor() or "unknown",
        cpu_count=os.cpu_count(),
        timezone=datetime.now().astimezone().tzname() or "unknown",
        dependencies=dependencies,
    )
    return LocalArtifactStore(run_dir).write_json(
        "metadata/environment.json",
        captured.model_dump(mode="json"),
    )


class ReproducibilityRecorder:
    RELATIVE_PATH = Path("metadata/reproducibility.json")

    def write(
        self,
        *,
        config: AppConfig,
        context: RunContext,
        manifest: RunManifest,
        report_result: ReportGenerationResult,
        validation_path: Path,
    ) -> Path:
        if manifest.dataset is None or manifest.inference is None:
            raise RunInitializationError(
                "Cannot record reproducibility metadata before dataset and inference"
            )
        if manifest.evaluation is None:
            raise RunInitializationError(
                "Cannot record reproducibility metadata before evaluation"
            )

        candidates = [
            context.config_snapshot_path,
            context.run_dir / "metadata" / "environment.json",
            context.run_dir / manifest.dataset.original_artifact,
            context.run_dir / manifest.dataset.normalized_artifact,
            context.run_dir / manifest.inference.artifact,
            context.run_dir / manifest.inference.checkpoint,
            context.run_dir / manifest.evaluation.artifact,
            report_result.report_json_path,
            report_result.report_markdown_path,
            report_result.samples_csv_path,
            validation_path,
        ]
        artifacts: dict[str, ArtifactDigest] = {}
        root = context.run_dir.resolve()
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise RunInitializationError(
                    f"Cannot fingerprint missing or unsafe artifact: {candidate}"
                )
            relative = resolved.relative_to(root).as_posix()
            artifacts[relative] = ArtifactDigest(
                sha256=_sha256(resolved),
                size_bytes=resolved.stat().st_size,
            )

        prompt_path = config.evaluation.prompt_template.resolve()
        sanitized = config.sanitized_dict()
        model_information = sanitized.get("models")
        if not isinstance(model_information, dict):
            raise RunInitializationError("Sanitized model configuration is invalid")
        reproducibility = ReproducibilityMetadata(
            generated_at=datetime.now(UTC),
            run_id=context.run_id,
            pipeline_version=__version__,
            config_fingerprint=config.fingerprint(),
            random_seed=config.runtime.random_seed,
            concurrency=config.runtime.concurrency,
            max_retries=config.runtime.max_retries,
            retry_backoff_seconds=config.runtime.retry_backoff_seconds,
            dataset_source_sha256=manifest.dataset.source_sha256,
            prompt_template_path=str(prompt_path),
            prompt_template_sha256=_sha256(prompt_path),
            models=model_information,
            artifacts=artifacts,
        )
        return LocalArtifactStore(context.run_dir).write_json(
            self.RELATIVE_PATH,
            reproducibility.model_dump(mode="json"),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunInitializationError(f"Unable to fingerprint artifact: {path}") from exc
    return digest.hexdigest()
