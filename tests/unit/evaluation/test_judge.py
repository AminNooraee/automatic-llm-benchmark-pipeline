from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from llm_benchmark.clients.contracts import ChatMessage, GenerationResult
from llm_benchmark.clients.errors import ModelConnectionError
from llm_benchmark.config.models import EndpointConfig, EvaluationCriterion
from llm_benchmark.evaluation.engine import JudgeEngine
from llm_benchmark.evaluation.errors import JudgeOutputError
from llm_benchmark.evaluation.models import (
    EvaluatedModel,
    JudgeEvaluationStatus,
    JudgeResult,
    ResolvedWinner,
)
from llm_benchmark.evaluation.parser import JudgeOutputParser
from llm_benchmark.evaluation.randomization import AnswerRandomizer


TEMPLATE = """Question: {{question}}
Candidate A: {{answer_a}}
Candidate B: {{answer_b}}
Criteria: {{criteria}}
Schema: {{output_schema}}
"""

VALID_DECISION = {
    "winner": "A",
    "scores": {"A": 4, "B": 3},
    "criteria_scores": {"correctness": {"A": 4, "B": 3}},
    "reason": "Candidate A is more accurate.",
}


def _judge_model() -> EndpointConfig:
    return EndpointConfig.model_validate(
        {
            "base_url": "https://judge.example/v1",
            "name": "judge-model",
            "api_key": None,
            "timeout_seconds": 2,
            "generation_parameters": {"temperature": 0},
        }
    )


def _criteria() -> list[EvaluationCriterion]:
    return [
        EvaluationCriterion(
            name="correctness",
            description="Factual and logical correctness.",
            weight=1,
            minimum=1,
            maximum=5,
        )
    ]


def _write_inputs(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class FakeJudgeClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[str] = []

    async def generate(
        self,
        model_config: EndpointConfig,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
    ) -> GenerationResult:
        message = messages[0]
        content = (
            message.content
            if isinstance(message, ChatMessage)
            else message["content"]
        )
        assert isinstance(content, str)
        self.messages.append(content)
        if self.fail:
            raise ModelConnectionError(model_config.name, attempts=2)
        return GenerationResult(
            content=json.dumps(VALID_DECISION),
            model_name=model_config.name,
            finish_reason="stop",
            latency_ms=3,
            attempts=1,
        )

    async def aclose(self) -> None:
        return None


def test_answer_randomization_supports_both_blind_orders() -> None:
    base_first = AnswerRandomizer(random_value=lambda: 0.1).arrange(
        "base answer", "fine answer"
    )
    fine_first = AnswerRandomizer(random_value=lambda: 0.9).arrange(
        "base answer", "fine answer"
    )

    assert base_first.answer_a == "base answer"
    assert base_first.answer_b == "fine answer"
    assert base_first.order.A == EvaluatedModel.BASE
    assert base_first.order.B == EvaluatedModel.FINE_TUNED
    assert fine_first.answer_a == "fine answer"
    assert fine_first.answer_b == "base answer"
    assert fine_first.order.A == EvaluatedModel.FINE_TUNED
    assert fine_first.order.B == EvaluatedModel.BASE


def test_judge_engine_sends_only_anonymous_labels_and_resolves_winner(
    tmp_path: Path,
) -> None:
    inference_path = tmp_path / "inference" / "responses.jsonl"
    _write_inputs(
        inference_path,
        [
            {
                "id": "one",
                "prompt": "Which response is better?",
                "base_response": "First response text",
                "fine_tuned_response": "Second response text",
            }
        ],
    )
    template_path = tmp_path / "judge.txt"
    template_path.write_text(TEMPLATE, encoding="utf-8")
    client = FakeJudgeClient()
    engine = JudgeEngine(
        client,
        randomizer=AnswerRandomizer(random_value=lambda: 0.9),
    )

    result = asyncio.run(
        engine.run(
            inference_path=inference_path,
            run_dir=tmp_path,
            judge_model=_judge_model(),
            prompt_template_path=template_path,
            criteria=_criteria(),
            resume=False,
        )
    )

    assert result.successful_samples == 1
    assert "Candidate A: Second response text" in client.messages[0]
    assert "Candidate B: First response text" in client.messages[0]
    assert "base-model" not in client.messages[0]
    assert "fine-tuned-model" not in client.messages[0]
    stored = JudgeResult.model_validate_json(
        result.artifact_path.read_text(encoding="utf-8").strip()
    )
    assert stored.winner is not None and stored.winner.value == "A"
    assert stored.metadata.answer_order.A == EvaluatedModel.FINE_TUNED
    assert stored.metadata.resolved_winner == ResolvedWinner.FINE_TUNED


def test_judge_parser_accepts_required_structured_output() -> None:
    decision = JudgeOutputParser().parse(json.dumps(VALID_DECISION))

    assert decision.winner.value == "A"
    assert decision.scores.A == 4
    assert decision.criteria_scores["correctness"].B == 3


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "not valid JSON"),
        (
            json.dumps(
                {
                    "winner": "A",
                    "scores": {"A": 4, "B": 3},
                    "criteria_scores": {},
                }
            ),
            "reason",
        ),
        (
            json.dumps(
                {
                    **VALID_DECISION,
                    "winner": "base",
                }
            ),
            "winner",
        ),
    ],
)
def test_judge_parser_rejects_invalid_outputs(content: str, message: str) -> None:
    with pytest.raises(JudgeOutputError, match=message):
        JudgeOutputParser().parse(content)


def test_judge_api_failure_is_persisted_without_aborting(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference" / "responses.jsonl"
    _write_inputs(
        inference_path,
        [
            {
                "id": "one",
                "prompt": "Question",
                "base_response": "First response",
                "fine_tuned_response": "Second response",
            }
        ],
    )
    template_path = tmp_path / "judge.txt"
    template_path.write_text(TEMPLATE, encoding="utf-8")

    result = asyncio.run(
        JudgeEngine(FakeJudgeClient(fail=True)).run(
            inference_path=inference_path,
            run_dir=tmp_path,
            judge_model=_judge_model(),
            prompt_template_path=template_path,
            criteria=_criteria(),
            resume=False,
        )
    )

    assert result.failed_samples == 1
    stored = JudgeResult.model_validate_json(
        result.artifact_path.read_text(encoding="utf-8").strip()
    )
    assert stored.metadata.status == JudgeEvaluationStatus.ERROR
    assert stored.metadata.error_type == "ModelConnectionError"
    assert stored.metadata.attempts == 2
    assert stored.winner is None


def test_incomplete_inference_response_is_skipped_without_judge_call(
    tmp_path: Path,
) -> None:
    inference_path = tmp_path / "inference" / "responses.jsonl"
    _write_inputs(
        inference_path,
        [
            {
                "id": "one",
                "prompt": "Question",
                "base_response": "",
                "fine_tuned_response": "Second response",
            }
        ],
    )
    template_path = tmp_path / "judge.txt"
    template_path.write_text(TEMPLATE, encoding="utf-8")
    client = FakeJudgeClient()

    result = asyncio.run(
        JudgeEngine(client).run(
            inference_path=inference_path,
            run_dir=tmp_path,
            judge_model=_judge_model(),
            prompt_template_path=template_path,
            criteria=_criteria(),
            resume=False,
        )
    )

    assert result.skipped_samples == 1
    assert result.attempted_samples == 0
    assert client.messages == []
