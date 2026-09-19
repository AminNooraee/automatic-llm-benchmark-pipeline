"""CSV source adapter with header validation."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from llm_benchmark.config.models import DatasetFormat
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.models import DatasetSchema, RawDatasetRecord
from llm_benchmark.exceptions import DatasetSourceError


_SCHEMAS = tuple(schema.value for schema in DatasetSchema)


class CsvFileAdapter:
    format = DatasetFormat.CSV
    extensions = (".csv",)

    def read(self, path: Path) -> Iterator[RawDatasetRecord]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                if not fields:
                    raise DatasetSourceError(
                        dataset_error_message(
                            "CSV file is missing a header row",
                            file_path=path,
                            row_number=1,
                            possible_formats=_SCHEMAS,
                        )
                    )
                normalized_fields = [field.strip() for field in fields]
                if any(not field for field in normalized_fields):
                    raise DatasetSourceError(
                        dataset_error_message(
                            "CSV contains an empty column name",
                            file_path=path,
                            row_number=1,
                            fields=normalized_fields,
                            possible_formats=_SCHEMAS,
                        )
                    )
                if len(normalized_fields) != len(set(normalized_fields)):
                    raise DatasetSourceError(
                        dataset_error_message(
                            "CSV contains duplicate column names",
                            file_path=path,
                            row_number=1,
                            fields=normalized_fields,
                            possible_formats=_SCHEMAS,
                        )
                    )

                for row_number, row in enumerate(reader, start=2):
                    if None in row:
                        raise DatasetSourceError(
                            dataset_error_message(
                                "CSV row has more values than the header",
                                file_path=path,
                                row_number=row_number,
                                fields=normalized_fields,
                                possible_formats=_SCHEMAS,
                            )
                        )
                    normalized = {
                        field.strip(): value for field, value in row.items()
                    }
                    yield RawDatasetRecord(row_number=row_number, data=normalized)
        except DatasetSourceError:
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            raise DatasetSourceError(
                dataset_error_message(
                    "Unable to read CSV file",
                    file_path=path,
                    possible_formats=(self.format.value,),
                )
            ) from exc

