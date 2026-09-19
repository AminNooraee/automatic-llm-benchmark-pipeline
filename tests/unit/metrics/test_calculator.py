from __future__ import annotations

import pytest

from llm_benchmark.evaluation.models import (
    AnswerOrder,
    AnswerScores,
    EvaluatedModel,
    JudgeEvaluationStatus,
    JudgeInput,
    JudgeResult,
    JudgeResultMetadata,
    JudgeWinner,
    ResolvedWinner,
)
from llm_benchmark.metrics.calculator import BenchmarkMetricsCalculator


def _input(sample_id: str) -> JudgeInput:
    return JudgeInput(
        id=sample_id,
        prompt=f"Prompt {sample_id}",
        base_response=f"Base {sample_id}",
        fine_tuned_response=f"Fine {sample_id}",
    )


def _success(
    sample_id: str,
    *,
    a_model: EvaluatedModel,
    winner: JudgeWinner,
    score_a: float,
    score_b: float,
) -> JudgeResult:
    b_model = (
        EvaluatedModel.FINE_TUNED
        if a_model == EvaluatedModel.BASE
        else EvaluatedModel.BASE
    )
    order = AnswerOrder(A=a_model, B=b_model)
    return JudgeResult(
        id=sample_id,
        prompt=f"Prompt {sample_id}",
        winner=winner,
        scores=AnswerScores(A=score_a, B=score_b),
        criteria_scores={
            "correctness": AnswerScores(A=score_a, B=score_b)
        },
        reason=f"Reason {sample_id}",
        metadata=JudgeResultMetadata(
            status=JudgeEvaluationStatus.SUCCESS,
            answer_order=order,
            resolved_winner=order.resolve(winner),
            judge_model="judge-model",
            attempts=1,
        ),
    )


def _unsuccessful(
    sample_id: str,
    status: JudgeEvaluationStatus,
) -> JudgeResult:
    return JudgeResult(
        id=sample_id,
        prompt=f"Prompt {sample_id}",
        metadata=JudgeResultMetadata(
            status=status,
            answer_order=AnswerOrder(
                A=EvaluatedModel.BASE,
                B=EvaluatedModel.FINE_TUNED,
            ),
            judge_model="judge-model",
            attempts=0 if status == JudgeEvaluationStatus.SKIPPED else 1,
            error_type="TestError",
            error_message="Evaluation unavailable",
        ),
    )


def test_metric_calculation_counts_outcomes_rates_and_average_scores() -> None:
    responses = [_input(str(index)) for index in range(1, 6)]
    results = [
        _success(
            "1",
            a_model=EvaluatedModel.BASE,
            winner=JudgeWinner.A,
            score_a=4,
            score_b=3,
        ),
        _success(
            "2",
            a_model=EvaluatedModel.FINE_TUNED,
            winner=JudgeWinner.A,
            score_a=5,
            score_b=2,
        ),
        _success(
            "3",
            a_model=EvaluatedModel.FINE_TUNED,
            winner=JudgeWinner.TIE,
            score_a=4,
            score_b=4,
        ),
        _unsuccessful("4", JudgeEvaluationStatus.ERROR),
        _unsuccessful("5", JudgeEvaluationStatus.SKIPPED),
    ]

    analysis = BenchmarkMetricsCalculator().calculate(results, responses)
    metrics = analysis.aggregate_metrics

    assert metrics.total_samples == 5
    assert metrics.successful_evaluations == 3
    assert metrics.failed_evaluations == 1
    assert metrics.skipped_evaluations == 1
    assert metrics.base_model_wins == 1
    assert metrics.fine_tuned_model_wins == 1
    assert metrics.ties == 1
    assert metrics.win_rate_percentage.base_model == pytest.approx(33.3333)
    assert metrics.win_rate_percentage.fine_tuned_model == pytest.approx(33.3333)
    assert metrics.win_rate_percentage.ties == pytest.approx(33.3333)
    assert metrics.average_scores_per_model.base_model == pytest.approx(3.3333)
    assert metrics.average_scores_per_model.fine_tuned_model == 4


def test_ab_mapping_is_resolved_to_base_and_fine_tuned_scores() -> None:
    response = _input("one")
    result = _success(
        "one",
        a_model=EvaluatedModel.FINE_TUNED,
        winner=JudgeWinner.A,
        score_a=5,
        score_b=2,
    )

    sample = BenchmarkMetricsCalculator().calculate(
        [result], [response]
    ).samples[0]

    assert sample.winner == ResolvedWinner.FINE_TUNED
    assert sample.judge_winner_label == JudgeWinner.A
    assert sample.base_score == 2
    assert sample.fine_tuned_score == 5
    assert sample.criteria_scores is not None
    assert sample.criteria_scores["correctness"].base_model == 2
    assert sample.criteria_scores["correctness"].fine_tuned_model == 5
