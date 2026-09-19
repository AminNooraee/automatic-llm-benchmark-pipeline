"""Automatic, ambiguity-intolerant dataset schema detection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_benchmark.config.models import DatasetColumnsConfig
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.models import DatasetDetection, RawDatasetRecord
from llm_benchmark.datasets.schemas.base import SchemaAdapter
from llm_benchmark.datasets.schemas.custom_adapter import CustomColumnSchemaAdapter
from llm_benchmark.datasets.schemas.registry import SchemaAdapterRegistry
from llm_benchmark.exceptions import DatasetDetectionError


class DatasetSchemaDetector:
    def __init__(self, registry: SchemaAdapterRegistry | None = None) -> None:
        self._registry = registry or SchemaAdapterRegistry()

    def detect(
        self,
        records: Sequence[RawDatasetRecord],
        *,
        source_path: Path,
        columns: DatasetColumnsConfig | None,
    ) -> tuple[DatasetDetection, SchemaAdapter]:
        if not records:
            raise DatasetDetectionError(
                dataset_error_message(
                    "Dataset is empty",
                    file_path=source_path,
                    possible_formats=self._registry.possible_formats,
                )
            )

        fields = tuple(sorted({key for record in records for key in record.data}))
        if columns is not None:
            adapter = CustomColumnSchemaAdapter(columns.prompt)
            missing = [
                record
                for record in records
                if adapter.match_score(record.data) == 0
            ]
            if missing:
                record = missing[0]
                raise DatasetDetectionError(
                    dataset_error_message(
                        f"Configured prompt column '{columns.prompt}' is missing",
                        file_path=source_path,
                        row_number=record.row_number,
                        fields=record.data.keys(),
                        possible_formats=(adapter.schema.value,),
                    )
                )
            return (
                DatasetDetection(
                    schema=adapter.schema,
                    confidence=1.0,
                    detected_fields=fields,
                    sampled_records=len(records),
                ),
                adapter,
            )

        candidates: list[tuple[SchemaAdapter, list[float]]] = []
        partial_formats: set[str] = set()
        for adapter in self._registry.adapters:
            scores = [adapter.match_score(record.data) for record in records]
            if any(score > 0 for score in scores):
                partial_formats.add(adapter.schema.value)
            if all(score > 0 for score in scores):
                candidates.append((adapter, scores))

        first_record = records[0]
        if len(candidates) > 1:
            possible = [adapter.schema.value for adapter, _ in candidates]
            raise DatasetDetectionError(
                dataset_error_message(
                    "Ambiguous dataset schema; configure dataset.columns.prompt",
                    file_path=source_path,
                    row_number=first_record.row_number,
                    fields=fields,
                    possible_formats=possible,
                )
            )
        if not candidates:
            possible = partial_formats or set(self._registry.possible_formats)
            raise DatasetDetectionError(
                dataset_error_message(
                    "No single supported schema matches all sampled records",
                    file_path=source_path,
                    row_number=first_record.row_number,
                    fields=fields,
                    possible_formats=possible,
                )
            )

        adapter, scores = candidates[0]
        confidence = sum(scores) / len(scores)
        return (
            DatasetDetection(
                schema=adapter.schema,
                confidence=round(confidence, 3),
                detected_fields=fields,
                sampled_records=len(records),
            ),
            adapter,
        )

