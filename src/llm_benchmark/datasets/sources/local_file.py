"""Local-file dataset source using independent format adapters."""

from __future__ import annotations

from llm_benchmark.config.models import DatasetConfig
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.formats.registry import FileFormatAdapterRegistry
from llm_benchmark.datasets.models import DatasetStream
from llm_benchmark.exceptions import DatasetSourceError


class LocalFileDatasetSource:
    source_type = "local_file"

    def __init__(self, registry: FileFormatAdapterRegistry | None = None) -> None:
        self._registry = registry or FileFormatAdapterRegistry()

    def open(self, config: DatasetConfig) -> DatasetStream:
        path = config.path
        if not path.is_file():
            raise DatasetSourceError(
                dataset_error_message(
                    "Dataset file does not exist or is not a regular file",
                    file_path=path,
                    possible_formats=self._registry.supported_formats,
                )
            )
        adapter = self._registry.resolve(path, config.format)
        return DatasetStream(
            source_path=path,
            source_format=adapter.format,
            records=adapter.read(path),
        )

