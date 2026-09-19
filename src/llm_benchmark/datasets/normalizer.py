"""Streaming normalization into the internal id/prompt contract."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain, islice
from pathlib import Path
from typing import Any

from llm_benchmark.config.models import DatasetColumnsConfig
from llm_benchmark.datasets.detector import DatasetSchemaDetector
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.models import (
    BenchmarkSample,
    PreparedDataset,
    RawDatasetRecord,
)
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.utils import get_dotted_value
from llm_benchmark.exceptions import DatasetValidationError


class DatasetNormalizer:
    def __init__(
        self,
        detector: DatasetSchemaDetector | None = None,
        detection_sample_size: int = 100,
    ) -> None:
        if detection_sample_size < 1:
            raise ValueError("detection_sample_size must be positive")
        self._detector = detector or DatasetSchemaDetector()
        self._detection_sample_size = detection_sample_size

    def prepare(
        self,
        records: Iterable[RawDatasetRecord],
        *,
        source_path: Path,
        columns: DatasetColumnsConfig | None,
    ) -> PreparedDataset:
        iterator = iter(records)
        buffered = list(islice(iterator, self._detection_sample_size))
        detection, adapter = self._detector.detect(
            buffered,
            source_path=source_path,
            columns=columns,
        )
        samples = self._normalize_records(
            chain(buffered, iterator),
            adapter=adapter,
            source_path=source_path,
            columns=columns,
        )
        return PreparedDataset(detection=detection, samples=samples)

    def _normalize_records(
        self,
        records: Iterable[RawDatasetRecord],
        *,
        adapter: SchemaAdapter,
        source_path: Path,
        columns: DatasetColumnsConfig | None,
    ) -> Iterator[BenchmarkSample]:
        seen_ids: set[str] = set()
        for ordinal, record in enumerate(records, start=1):
            try:
                prompt = adapter.extract_prompt(record.data)
            except ValueError as exc:
                raise self._record_error(
                    str(exc), record, source_path, (adapter.schema.value,)
                ) from exc

            try:
                sample_id = self._extract_id(record.data, columns, ordinal)
            except ValueError as exc:
                raise self._record_error(
                    str(exc),
                    record,
                    source_path,
                    (adapter.schema.value,),
                ) from exc
            if sample_id in seen_ids:
                raise self._record_error(
                    f"Duplicate id '{sample_id}'",
                    record,
                    source_path,
                    (adapter.schema.value,),
                )
            seen_ids.add(sample_id)

            try:
                yield BenchmarkSample(id=sample_id, prompt=prompt)
            except ValueError as exc:
                raise self._record_error(
                    "Normalized sample is invalid",
                    record,
                    source_path,
                    (adapter.schema.value,),
                ) from exc

    @staticmethod
    def _extract_id(
        record: dict[str, Any],
        columns: DatasetColumnsConfig | None,
        ordinal: int,
    ) -> str:
        if columns is not None and columns.id is not None:
            found, value = get_dotted_value(record, columns.id)
            if not found:
                raise ValueError(f"Configured id column '{columns.id}' is missing")
        elif "id" in record:
            value = record["id"]
        else:
            return f"sample-{ordinal:06d}"

        if value is None or isinstance(value, (dict, list)):
            raise ValueError("Record id must be a non-empty scalar value")
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Record id is empty")
        return normalized

    @staticmethod
    def _record_error(
        reason: str,
        record: RawDatasetRecord,
        source_path: Path,
        possible_formats: Iterable[str],
    ) -> DatasetValidationError:
        return DatasetValidationError(
            dataset_error_message(
                reason,
                file_path=source_path,
                row_number=record.row_number,
                fields=record.data.keys(),
                possible_formats=possible_formats,
            )
        )
