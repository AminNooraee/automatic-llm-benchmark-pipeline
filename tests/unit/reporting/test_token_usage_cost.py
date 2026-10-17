from __future__ import annotations

from pathlib import Path

import pytest

from llm_benchmark.clients.contracts import GenerationUsage
from llm_benchmark.evaluation.models import (
    AnswerOrder,
    AnswerScores,
    EvaluatedModel,
    JudgeEvaluationStatus,
    JudgeResult,
    JudgeResultMetadata,
    JudgeWinner,
)
from llm_benchmark.inference.models import (
    InferenceMetadata,
    InferenceResponse,
    ModelResponseMetadata,
    ModelResponseStatus,
    SampleInferenceStatus,
)
from llm_benchmark.reporting.cost import estimate_judge_cost, load_pricing
from llm_benchmark.reporting.token_usage import RoleUsage, summarize_token_usage


ROOT = Path(__file__).resolve().parents[3]


def _usage(prompt: int | None, completion: int | None, total: int | None):
    return GenerationUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _response(
    sample_id: str,
    base_usage: GenerationUsage | None,
    fine_usage: GenerationUsage | None,
) -> InferenceResponse:
    return InferenceResponse(
        id=sample_id,
        prompt="prompt",
        base_response="base",
        fine_tuned_response="fine",
        metadata=InferenceMetadata(
            status=SampleInferenceStatus.SUCCESS,
            base=ModelResponseMetadata(
                status=ModelResponseStatus.SUCCESS,
                model_name="base",
                usage=base_usage,
            ),
            fine_tuned=ModelResponseMetadata(
                status=ModelResponseStatus.SUCCESS,
                model_name="fine",
                usage=fine_usage,
            ),
        ),
    )


def _judge(sample_id: str, usage: GenerationUsage | None) -> JudgeResult:
    order = AnswerOrder(A=EvaluatedModel.BASE, B=EvaluatedModel.FINE_TUNED)
    return JudgeResult(
        id=sample_id,
        prompt="prompt",
        winner=JudgeWinner.A,
        scores=AnswerScores(A=5, B=4),
        criteria_scores={"quality": AnswerScores(A=5, B=4)},
        reason="reason",
        metadata=JudgeResultMetadata(
            status=JudgeEvaluationStatus.SUCCESS,
            answer_order=order,
            resolved_winner=order.resolve(JudgeWinner.A),
            judge_model="judge",
            attempts=1,
            usage=usage,
        ),
    )


def test_base_fine_tuned_and_judge_usage_are_aggregated_separately() -> None:
    summary = summarize_token_usage(
        [
            _response("1", _usage(10, 2, 12), _usage(20, 3, 23)),
            _response("2", _usage(11, 4, 15), _usage(21, 5, 26)),
        ],
        [_judge("1", _usage(30, 6, 36)), _judge("2", _usage(31, 7, 38))],
    )

    assert summary.base.complete
    assert (summary.base.prompt_tokens, summary.base.completion_tokens, summary.base.total_tokens) == (21, 6, 27)
    assert (summary.fine_tuned.prompt_tokens, summary.fine_tuned.completion_tokens, summary.fine_tuned.total_tokens) == (41, 8, 49)
    assert (summary.judge.prompt_tokens, summary.judge.completion_tokens, summary.judge.total_tokens) == (61, 13, 74)


def test_partial_and_missing_usage_never_become_complete_or_zero() -> None:
    summary = summarize_token_usage(
        [
            _response("1", _usage(10, 2, 12), None),
            _response("2", None, _usage(20, None, None)),
        ],
        [_judge("1", _usage(30, 6, 36)), _judge("2", None)],
    )

    assert summary.base.partial
    assert summary.base.prompt_tokens == 10
    assert summary.fine_tuned.partial
    assert summary.fine_tuned.prompt_tokens == 20
    assert summary.fine_tuned.completion_tokens is None
    assert summary.judge.partial
    assert not summary.judge_complete
    fully_missing = summarize_token_usage([_response("3", None, None)], [_judge("3", None)])
    for role in (fully_missing.base, fully_missing.fine_tuned, fully_missing.judge):
        assert role.missing
        assert role.prompt_tokens is None
        assert role.completion_tokens is None
        assert role.total_tokens is None


def _complete_usage() -> RoleUsage:
    return RoleUsage(
        expected_successful=1,
        reported=1,
        available=1,
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        total_tokens=1_500_000,
    )


@pytest.mark.parametrize(
    ("model", "expected", "low", "high"),
    [
        ("gpt-5.6-sol", 14.0, None, None),
        ("claude-sonnet-5-5", 7.0, None, None),
        ("deepseek-v4-pro", None, 1.65, 3.30),
    ],
)
def test_static_pricing_calculations(model, expected, low, high) -> None:
    pricing = load_pricing(ROOT / "configs" / "judge_pricing.yaml")
    cost = estimate_judge_cost(
        judge_name=model,
        judge_usage=_complete_usage(),
        pricing=pricing,
    )

    assert cost.priced
    assert cost.snapshot_date == "2026-10-14"
    if expected is not None:
        assert cost.kind == "flat"
        assert cost.cost == pytest.approx(expected)
    else:
        assert cost.kind == "range"
        assert cost.cost_low == pytest.approx(low)
        assert cost.cost_high == pytest.approx(high)


def test_incomplete_usage_and_invalid_pricing_are_unavailable() -> None:
    pricing = load_pricing(ROOT / "configs" / "judge_pricing.yaml")
    incomplete = RoleUsage(
        expected_successful=2,
        reported=1,
        available=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    assert not estimate_judge_cost(
        judge_name="gpt-5.6-sol", judge_usage=incomplete, pricing=pricing
    ).priced

    invalid = {
        "snapshot_date": "2026-10-14",
        "currency": "USD",
        "models": [
            {
                "name": "broken",
                "pricing": {
                    "input_per_million": "not-a-number",
                    "output_per_million": 1,
                },
            }
        ],
    }
    unavailable = estimate_judge_cost(
        judge_name="broken", judge_usage=_complete_usage(), pricing=invalid
    )
    assert not unavailable.priced
    assert unavailable.snapshot_date == "2026-10-14"

