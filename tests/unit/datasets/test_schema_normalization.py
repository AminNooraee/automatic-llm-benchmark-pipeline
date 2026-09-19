from __future__ import annotations

from pathlib import Path

import pytest

from llm_benchmark.config.models import DatasetColumnsConfig
from llm_benchmark.datasets.models import DatasetSchema, RawDatasetRecord
from llm_benchmark.datasets.normalizer import DatasetNormalizer
from llm_benchmark.exceptions import DatasetDetectionError, DatasetValidationError


@pytest.mark.parametrize(
    ("record", "expected_schema", "expected_prompt"),
    [
        ({"prompt": "Explain AI"}, DatasetSchema.PROMPT, "Explain AI"),
        (
            {"question": "What is AI?", "answer": "A field of study"},
            DatasetSchema.QUESTION_ANSWER,
            "What is AI?",
        ),
        (
            {"instruction": "Explain machine learning", "input": "for beginners"},
            DatasetSchema.INSTRUCTION,
            "Explain machine learning\n\nfor beginners",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "First request"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": "Latest request"},
                ]
            },
            DatasetSchema.OPENAI_MESSAGES,
            "Latest request",
        ),
        (
            {
                "chatml": (
                    "<|im_start|>system\nBe concise<|im_end|>"
                    "<|im_start|>user\nExplain AI<|im_end|>"
                )
            },
            DatasetSchema.CHATML,
            "Explain AI",
        ),
        (
            {
                "conversations": [
                    {"from": "human", "value": "Explain AI"},
                    {"from": "gpt", "value": "AI is..."},
                ]
            },
            DatasetSchema.SHAREGPT,
            "Explain AI",
        ),
    ],
)
def test_common_schemas_are_detected_and_normalized(
    tmp_path: Path,
    record: dict,
    expected_schema: DatasetSchema,
    expected_prompt: str,
) -> None:
    source = tmp_path / "samples.jsonl"
    prepared = DatasetNormalizer().prepare(
        [RawDatasetRecord(row_number=1, data=record)],
        source_path=source,
        columns=None,
    )

    samples = list(prepared.samples)

    assert prepared.detection.schema == expected_schema
    assert 0 < prepared.detection.confidence <= 1
    assert samples[0].id == "sample-000001"
    assert samples[0].prompt == expected_prompt


def test_openai_multimodal_text_parts_are_supported(tmp_path: Path) -> None:
    record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "input_text", "text": "Part two"},
                ],
            }
        ]
    }
    prepared = DatasetNormalizer().prepare(
        [RawDatasetRecord(row_number=1, data=record)],
        source_path=tmp_path / "messages.jsonl",
        columns=None,
    )

    assert list(prepared.samples)[0].prompt == "Part one\nPart two"


def test_custom_prompt_and_id_columns_override_detection(tmp_path: Path) -> None:
    records = [
        RawDatasetRecord(
            row_number=1,
            data={"keys": {"record_id": "custom-1"}, "question_text": "Why?"},
        )
    ]
    columns = DatasetColumnsConfig(
        prompt="question_text", id="keys.record_id"
    )

    prepared = DatasetNormalizer().prepare(
        records,
        source_path=tmp_path / "custom.jsonl",
        columns=columns,
    )
    sample = list(prepared.samples)[0]

    assert prepared.detection.schema == DatasetSchema.CUSTOM
    assert prepared.detection.confidence == 1
    assert sample.id == "custom-1"
    assert sample.prompt == "Why?"


def test_empty_prompt_error_has_required_context(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    prepared = DatasetNormalizer().prepare(
        [RawDatasetRecord(row_number=7, data={"id": "one", "prompt": "  "})],
        source_path=source,
        columns=None,
    )

    with pytest.raises(DatasetValidationError) as captured:
        list(prepared.samples)

    message = str(captured.value)
    assert "empty.jsonl" in message
    assert "row 7" in message
    assert "Detected fields: [id, prompt]" in message
    assert "Possible formats: [prompt]" in message


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    records = [
        RawDatasetRecord(row_number=1, data={"id": "same", "prompt": "One"}),
        RawDatasetRecord(row_number=2, data={"id": "same", "prompt": "Two"}),
    ]
    prepared = DatasetNormalizer().prepare(
        records, source_path=tmp_path / "duplicate.jsonl", columns=None
    )

    with pytest.raises(DatasetValidationError, match="Duplicate id 'same'"):
        list(prepared.samples)


def test_missing_supported_fields_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetDetectionError) as captured:
        DatasetNormalizer().prepare(
            [RawDatasetRecord(row_number=1, data={"title": "No prompt here"})],
            source_path=tmp_path / "missing.jsonl",
            columns=None,
        )

    message = str(captured.value)
    assert "No single supported schema" in message
    assert "Detected fields: [title]" in message
    assert "Possible formats" in message


def test_ambiguous_dataset_requires_mapping(tmp_path: Path) -> None:
    with pytest.raises(DatasetDetectionError) as captured:
        DatasetNormalizer().prepare(
            [
                RawDatasetRecord(
                    row_number=1,
                    data={"prompt": "Prompt value", "question": "Question value"},
                )
            ],
            source_path=tmp_path / "ambiguous.jsonl",
            columns=None,
        )

    message = str(captured.value)
    assert "Ambiguous dataset schema" in message
    assert "prompt" in message
    assert "question_answer" in message


def test_missing_custom_column_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetDetectionError, match="question_text"):
        DatasetNormalizer().prepare(
            [RawDatasetRecord(row_number=3, data={"question": "Why?"})],
            source_path=tmp_path / "custom.jsonl",
            columns=DatasetColumnsConfig(prompt="question_text"),
        )


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetDetectionError, match="Dataset is empty"):
        DatasetNormalizer().prepare(
            [], source_path=tmp_path / "empty.jsonl", columns=None
        )


def test_missing_custom_id_column_has_clear_error(tmp_path: Path) -> None:
    prepared = DatasetNormalizer().prepare(
        [RawDatasetRecord(row_number=4, data={"question_text": "Why?"})],
        source_path=tmp_path / "custom.jsonl",
        columns=DatasetColumnsConfig(prompt="question_text", id="record_id"),
    )

    with pytest.raises(DatasetValidationError, match="id column 'record_id'"):
        list(prepared.samples)
