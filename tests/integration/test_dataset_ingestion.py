from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_benchmark.config.models import DatasetConfig
from llm_benchmark.datasets.service import DatasetIngestionService


def _write_source(path: Path, file_format: str) -> None:
    records = [
        {"id": "one", "question": "First question", "answer": "First answer"},
        {"id": "two", "question": "Second question", "answer": "Second answer"},
    ]
    if file_format == "json":
        path.write_text(json.dumps(records), encoding="utf-8")
    elif file_format == "jsonl":
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    elif file_format == "csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "question", "answer"])
            writer.writeheader()
            writer.writerows(records)
    elif file_format == "parquet":
        pq.write_table(pa.Table.from_pylist(records), path)
    else:
        raise AssertionError(file_format)


@pytest.mark.parametrize("file_format", ["json", "jsonl", "csv", "parquet"])
def test_ingestion_creates_original_and_normalized_artifacts(
    tmp_path: Path, file_format: str
) -> None:
    source = tmp_path / f"source.{file_format}"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_source(source, file_format)

    result = DatasetIngestionService().ingest(
        DatasetConfig(path=source, format="auto"), run_dir
    )

    assert result.source_format.value == file_format
    assert result.schema_format.value == "question_answer"
    assert result.detection_confidence == 1
    assert result.sample_count == 2
    assert len(result.source_sha256) == 64
    assert (run_dir / result.original_artifact).is_file()
    normalized_path = run_dir / result.normalized_artifact
    normalized = [
        json.loads(line)
        for line in normalized_path.read_text(encoding="utf-8").splitlines()
    ]
    assert normalized == [
        {"id": "one", "prompt": "First question"},
        {"id": "two", "prompt": "Second question"},
    ]

