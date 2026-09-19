"""JSON array, single-record, and single-conversation source adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llm_benchmark.config.models import DatasetFormat
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.models import DatasetSchema, RawDatasetRecord
from llm_benchmark.exceptions import DatasetSourceError


_SCHEMAS = tuple(schema.value for schema in DatasetSchema)


def is_role_content_sequence(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, dict) and {"role", "content"}.issubset(item)
            for item in value
        )
    )


class JsonFileAdapter:
    format = DatasetFormat.JSON
    extensions = (".json",)

    def read(self, path: Path) -> Iterator[RawDatasetRecord]:
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                payload: Any = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DatasetSourceError(
                dataset_error_message(
                    f"Malformed JSON: {exc.msg}",
                    file_path=path,
                    row_number=exc.lineno,
                    possible_formats=(self.format.value,),
                )
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise DatasetSourceError(
                dataset_error_message(
                    "Unable to read JSON file",
                    file_path=path,
                    possible_formats=(self.format.value,),
                )
            ) from exc

        if isinstance(payload, dict):
            records: list[Any] = [payload]
        elif is_role_content_sequence(payload):
            records = [{"messages": payload}]
        elif isinstance(payload, list):
            records = payload
        else:
            raise DatasetSourceError(
                dataset_error_message(
                    "JSON root must be an object or an array of objects",
                    file_path=path,
                    row_number=1,
                    possible_formats=_SCHEMAS,
                )
            )

        for row_number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise DatasetSourceError(
                    dataset_error_message(
                        "Malformed record; expected a JSON object",
                        file_path=path,
                        row_number=row_number,
                        possible_formats=_SCHEMAS,
                    )
                )
            yield RawDatasetRecord(row_number=row_number, data=record)

