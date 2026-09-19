"""Dataset source-to-artifact orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_benchmark.config.models import DatasetConfig
from llm_benchmark.datasets.models import DatasetIngestionResult
from llm_benchmark.datasets.normalizer import DatasetNormalizer
from llm_benchmark.datasets.sources.base import DatasetSourceAdapter
from llm_benchmark.datasets.sources.local_file import LocalFileDatasetSource
from llm_benchmark.runs.artifact_store import LocalArtifactStore


class DatasetIngestionService:
    """Adapt one configured source into normalized, durable run artifacts."""

    def __init__(
        self,
        source: DatasetSourceAdapter | None = None,
        normalizer: DatasetNormalizer | None = None,
    ) -> None:
        self._source = source or LocalFileDatasetSource()
        self._normalizer = normalizer or DatasetNormalizer()

    def ingest(self, config: DatasetConfig, run_dir: Path) -> DatasetIngestionResult:
        stream = self._source.open(config)
        store = LocalArtifactStore(run_dir)

        original_relative = f"dataset/original_dataset.{stream.source_format.value}"
        normalized_relative = "dataset/normalized_dataset.jsonl"
        original_path = store.copy_file(stream.source_path, original_relative)

        prepared = self._normalizer.prepare(
            stream.records,
            source_path=stream.source_path,
            columns=config.columns,
        )
        write_result = store.write_jsonl(
            normalized_relative,
            (sample.model_dump(mode="json") for sample in prepared.samples),
        )

        return DatasetIngestionResult(
            source_path=stream.source_path,
            source_format=stream.source_format,
            schema_format=prepared.detection.schema,
            detection_confidence=prepared.detection.confidence,
            detected_fields=list(prepared.detection.detected_fields),
            sample_count=write_result.record_count,
            source_sha256=_sha256_file(original_path),
            original_artifact=original_relative,
            normalized_artifact=normalized_relative,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
