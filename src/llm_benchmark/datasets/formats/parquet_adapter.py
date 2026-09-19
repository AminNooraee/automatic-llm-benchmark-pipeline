"""Apache Parquet source adapter backed by PyArrow."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from llm_benchmark.config.models import DatasetFormat
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.models import DatasetSchema, RawDatasetRecord
from llm_benchmark.exceptions import DatasetSourceError


_SCHEMAS = tuple(schema.value for schema in DatasetSchema)


class ParquetFileAdapter:
    format = DatasetFormat.PARQUET
    extensions = (".parquet",)

    def read(self, path: Path) -> Iterator[RawDatasetRecord]:
        try:
            import pyarrow.parquet as parquet

            parquet_file = parquet.ParquetFile(path)
            row_number = 0
            for batch in parquet_file.iter_batches():
                for record in batch.to_pylist():
                    row_number += 1
                    if not isinstance(record, dict):
                        raise DatasetSourceError(
                            dataset_error_message(
                                "Malformed Parquet record; expected an object",
                                file_path=path,
                                row_number=row_number,
                                possible_formats=_SCHEMAS,
                            )
                        )
                    yield RawDatasetRecord(row_number=row_number, data=record)
        except DatasetSourceError:
            raise
        except Exception as exc:
            raise DatasetSourceError(
                dataset_error_message(
                    f"Unable to read Parquet file: {type(exc).__name__}",
                    file_path=path,
                    possible_formats=(self.format.value,),
                )
            ) from exc

