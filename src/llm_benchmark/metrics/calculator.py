"""Aggregate metrics and resolve blind A/B decisions to model roles."""

from __future__ import annotations

from collections.abc import Sequence

from llm_benchmark.evaluation.models import (
    EvaluatedModel,
    JudgeEvaluationStatus,
    JudgeInput,
    JudgeResult,
    ResolvedWinner,
)
from llm_benchmark.metrics.errors import MetricsError
from llm_benchmark.metrics.models import (
    AggregateMetrics,
    BenchmarkAnalysis,
    ModelAverageScores,
    ModelWinRates,
    ResolvedCriteriaScores,
    ResolvedSampleResult,
)


class BenchmarkMetricsCalculator:
    def calculate(
        self,
        judge_results: Sequence[JudgeResult],
        inference_responses: Sequence[JudgeInput],
    ) -> BenchmarkAnalysis:
        responses = self._index_responses(inference_responses)
        seen_result_ids: set[str] = set()
        samples: list[ResolvedSampleResult] = []
        base_score_total = 0.0
        fine_score_total = 0.0
        successful = 0
        failed = 0
        skipped = 0
        base_wins = 0
        fine_wins = 0
        ties = 0

        for result in judge_results:
            if result.id in seen_result_ids:
                raise MetricsError(f"Duplicate judge result id: {result.id}")
            seen_result_ids.add(result.id)
            response = responses.get(result.id)
            if response is None:
                raise MetricsError(
                    f"Judge result '{result.id}' has no matching inference response"
                )
            if response.prompt != result.prompt:
                raise MetricsError(
                    f"Prompt mismatch between judge and inference artifacts for "
                    f"sample '{result.id}'"
                )

            if result.metadata.status == JudgeEvaluationStatus.SUCCESS:
                mapped = self._map_success(result, response)
                samples.append(mapped)
                successful += 1
                assert mapped.base_score is not None
                assert mapped.fine_tuned_score is not None
                assert mapped.winner is not None
                base_score_total += mapped.base_score
                fine_score_total += mapped.fine_tuned_score
                if mapped.winner == ResolvedWinner.BASE:
                    base_wins += 1
                elif mapped.winner == ResolvedWinner.FINE_TUNED:
                    fine_wins += 1
                else:
                    ties += 1
            else:
                samples.append(self._map_unsuccessful(result, response))
                if result.metadata.status == JudgeEvaluationStatus.ERROR:
                    failed += 1
                else:
                    skipped += 1

        missing_results = sorted(set(responses).difference(seen_result_ids))
        if missing_results:
            raise MetricsError(
                "Inference responses are missing judge results for sample IDs: "
                + ", ".join(missing_results)
            )

        win_rates = ModelWinRates(
            base_model=self._percentage(base_wins, successful),
            fine_tuned_model=self._percentage(fine_wins, successful),
            ties=self._percentage(ties, successful),
        )
        averages = ModelAverageScores(
            base_model=(
                round(base_score_total / successful, 4) if successful else None
            ),
            fine_tuned_model=(
                round(fine_score_total / successful, 4) if successful else None
            ),
        )
        metrics = AggregateMetrics(
            total_samples=len(judge_results),
            successful_evaluations=successful,
            failed_evaluations=failed,
            skipped_evaluations=skipped,
            base_model_wins=base_wins,
            fine_tuned_model_wins=fine_wins,
            ties=ties,
            win_rate_percentage=win_rates,
            average_scores_per_model=averages,
        )
        return BenchmarkAnalysis(aggregate_metrics=metrics, samples=samples)

    @staticmethod
    def _index_responses(
        responses: Sequence[JudgeInput],
    ) -> dict[str, JudgeInput]:
        indexed: dict[str, JudgeInput] = {}
        for response in responses:
            if response.id in indexed:
                raise MetricsError(f"Duplicate inference response id: {response.id}")
            indexed[response.id] = response
        return indexed

    @staticmethod
    def _map_success(
        result: JudgeResult,
        response: JudgeInput,
    ) -> ResolvedSampleResult:
        assert result.winner is not None
        assert result.scores is not None
        assert result.criteria_scores is not None
        assert result.reason is not None
        order = result.metadata.answer_order
        winner = order.resolve(result.winner)
        if order.A == EvaluatedModel.BASE:
            base_score = result.scores.A
            fine_score = result.scores.B
        else:
            base_score = result.scores.B
            fine_score = result.scores.A

        criteria_scores: dict[str, ResolvedCriteriaScores] = {}
        for name, scores in result.criteria_scores.items():
            if order.A == EvaluatedModel.BASE:
                base_criterion = scores.A
                fine_criterion = scores.B
            else:
                base_criterion = scores.B
                fine_criterion = scores.A
            criteria_scores[name] = ResolvedCriteriaScores(
                base_model=base_criterion,
                fine_tuned_model=fine_criterion,
            )

        return ResolvedSampleResult(
            id=result.id,
            prompt=result.prompt,
            base_response=response.base_response,
            fine_tuned_response=response.fine_tuned_response,
            evaluation_status=result.metadata.status,
            winner=winner,
            judge_winner_label=result.winner,
            answer_order=order,
            base_score=base_score,
            fine_tuned_score=fine_score,
            criteria_scores=criteria_scores,
            judge_reason=result.reason,
        )

    @staticmethod
    def _map_unsuccessful(
        result: JudgeResult,
        response: JudgeInput,
    ) -> ResolvedSampleResult:
        return ResolvedSampleResult(
            id=result.id,
            prompt=result.prompt,
            base_response=response.base_response,
            fine_tuned_response=response.fine_tuned_response,
            evaluation_status=result.metadata.status,
            answer_order=result.metadata.answer_order,
            error_type=result.metadata.error_type,
            error_message=result.metadata.error_message,
        )

    @staticmethod
    def _percentage(count: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return round((count / denominator) * 100, 4)
