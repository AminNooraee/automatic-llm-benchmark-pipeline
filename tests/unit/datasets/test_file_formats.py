from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_benchmark.config.models import DatasetConfig
from llm_benchmark.datasets.sources.local_file import LocalFileDatasetSource
from llm_benchmark.exceptions import DatasetSourceError


def _read(path: Path, configured_format: str = "auto"):
    config = DatasetConfig(path=path, format=configured_format)
    stream = LocalFileDatasetSource().open(config)
    return stream, list(stream.records)


def test_json_adapter_reads_array_records(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps([{"id": "one", "prompt": "First"}, {"prompt": "Second"}]),
        encoding="utf-8",
    )

    stream, records = _read(path)

    assert stream.source_format.value == "json"
    assert [record.row_number for record in records] == [1, 2]
    assert records[0].data["prompt"] == "First"


def test_json_adapter_treats_top_level_messages_as_one_conversation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps(
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Explain AI"},
            ]
        ),
        encoding="utf-8",
    )

    _, records = _read(path)

    assert len(records) == 1
    assert records[0].data["messages"][1]["content"] == "Explain AI"


def test_jsonl_adapter_preserves_physical_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"prompt":"First"}\n\n{"prompt":"Second"}\n', encoding="utf-8"
    )

    stream, records = _read(path)

    assert stream.source_format.value == "jsonl"
    assert [record.row_number for record in records] == [1, 3]


def test_csv_adapter_reads_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    path.write_text("id,prompt\none,First\ntwo,Second\n", encoding="utf-8")

    stream, records = _read(path)

    assert stream.source_format.value == "csv"
    assert records[0].row_number == 2
    assert records[1].data == {"id": "two", "prompt": "Second"}


def test_parquet_adapter_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "samples.parquet"
    table = pa.Table.from_pylist(
        [{"id": "one", "prompt": "First"}, {"id": "two", "prompt": "Second"}]
    )
    pq.write_table(table, path)

    stream, records = _read(path)

    assert stream.source_format.value == "parquet"
    assert [record.data["id"] for record in records] == ["one", "two"]


def test_explicit_format_supports_nonstandard_extension(tmp_path: Path) -> None:
    path = tmp_path / "samples.data"
    path.write_text('{"prompt":"Hello"}\n', encoding="utf-8")

    stream, records = _read(path, configured_format="jsonl")

    assert stream.source_format.value == "jsonl"
    assert records[0].data["prompt"] == "Hello"


def test_malformed_json_has_contextual_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('[{"prompt": "missing close"}', encoding="utf-8")

    stream = LocalFileDatasetSource().open(DatasetConfig(path=path, format="auto"))
    with pytest.raises(DatasetSourceError) as captured:
        list(stream.records)

    message = str(captured.value)
    assert "broken.json" in message
    assert "row 1" in message
    assert "Detected fields" in message
    assert "Possible formats" in message


def test_jsonl_non_object_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('"not an object"\n', encoding="utf-8")

    stream = LocalFileDatasetSource().open(DatasetConfig(path=path))
    with pytest.raises(DatasetSourceError, match="expected a JSON object"):
        list(stream.records)

