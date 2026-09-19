"""Registry for built-in local file-format adapters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from llm_benchmark.config.models import DatasetFormat
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.formats.base import FileFormatAdapter
from llm_benchmark.datasets.formats.csv_adapter import CsvFileAdapter
from llm_benchmark.datasets.formats.json_adapter import JsonFileAdapter
from llm_benchmark.datasets.formats.jsonl_adapter import JsonlFileAdapter
from llm_benchmark.datasets.formats.parquet_adapter import ParquetFileAdapter
from llm_benchmark.exceptions import DatasetSourceError


class FileFormatAdapterRegistry:
    """Resolve a format adapter without coupling readers to orchestration."""

    def __init__(self, adapters: Iterable[FileFormatAdapter] | None = None) -> None:
        configured = adapters or (
            JsonFileAdapter(),
            JsonlFileAdapter(),
            CsvFileAdapter(),
            ParquetFileAdapter(),
        )
        self._adapters = {adapter.format: adapter for adapter in configured}

    @property
    def supported_formats(self) -> tuple[str, ...]:
        return tuple(sorted(item.value for item in self._adapters))

    def resolve(
        self, path: Path, requested_format: DatasetFormat
    ) -> FileFormatAdapter:
        selected = requested_format
        if requested_format == DatasetFormat.AUTO:
            suffix = path.suffix.lower()
            matches = [
                adapter
                for adapter in self._adapters.values()
                if suffix in adapter.extensions
            ]
            if len(matches) != 1:
                raise DatasetSourceError(
                    dataset_error_message(
                        f"Unsupported or ambiguous file extension '{suffix}'",
                        file_path=path,
                        possible_formats=self.supported_formats,
                    )
                )
            return matches[0]

        adapter = self._adapters.get(selected)
        if adapter is None:
            raise DatasetSourceError(
                dataset_error_message(
                    f"Unsupported configured dataset format '{selected.value}'",
                    file_path=path,
                    possible_formats=self.supported_formats,
                )
            )
        return adapter

