"""Newline-delimited JSON source adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from llm_benchmark.config.models import DatasetFormat
from llm_benchmark.datasets.errors import dataset_error_message
from llm_benchmark.datasets.formats.json_adapter import is_role_content_sequence
from llm_benchmark.datasets.models import DatasetSchema, RawDatasetRecord
from llm_benchmark.exceptions import DatasetSourceError


_SCHEMAS = tuple(schema.value for schema in DatasetSchema)


class JsonlFileAdapter:
    format = DatasetFormat.JSONL
    extensions = (".jsonl",)

    def read(self, path: Path) -> Iterator[RawDatasetRecord]:
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DatasetSourceError(
                            dataset_error_message(
                                f"Malformed JSONL record: {exc.msg}",
                                file_path=path,
                                row_number=line_number,
                                possible_formats=_SCHEMAS,
                            )
                        ) from exc
                    if is_role_content_sequence(record):
                        record = {"messages": record}
                    if not isinstance(record, dict):
                        raise DatasetSourceError(
                            dataset_error_message(
                                "Malformed record; expected a JSON object",
                                file_path=path,
                                row_number=line_number,
                                possible_formats=_SCHEMAS,
                            )
                        )
                    yield RawDatasetRecord(row_number=line_number, data=record)
        except DatasetSourceError:
            raise
        except (OSError, UnicodeError) as exc:
            raise DatasetSourceError(
                dataset_error_message(
                    "Unable to read JSONL file",
                    file_path=path,
                    possible_formats=(self.format.value,),
                )
            ) from exc

