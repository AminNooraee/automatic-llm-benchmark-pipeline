"""Normalized aggregate and per-sample benchmark metrics."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_benchmark.evaluation.models import (
    AnswerOrder,
    JudgeEvaluationStatus,
    JudgeWinner,
    ResolvedWinner,
)


class MetricsContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelWinRates(MetricsContract):
    base_model: float = Field(ge=0, le=100)
    fine_tuned_model: float = Field(ge=0, le=100)
    ties: float = Field(ge=0, le=100)


class ModelAverageScores(MetricsContract):
    base_model: float | None = None
    fine_tuned_model: float | None = None


class AggregateMetrics(MetricsContract):
    total_samples: int = Field(ge=0)
    successful_evaluations: int = Field(ge=0)
    failed_evaluations: int = Field(ge=0)
    skipped_evaluations: int = Field(ge=0)
    base_model_wins: int = Field(ge=0)
    fine_tuned_model_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    win_rate_percentage: ModelWinRates
    average_scores_per_model: ModelAverageScores

    @model_validator(mode="after")
    def validate_counts(self) -> "AggregateMetrics":
        evaluated = (
            self.successful_evaluations
            + self.failed_evaluations
            + self.skipped_evaluations
        )
        if evaluated != self.total_samples:
            raise ValueError("evaluation counts must equal total_samples")
        outcomes = self.base_model_wins + self.fine_tuned_model_wins + self.ties
        if outcomes != self.successful_evaluations:
            raise ValueError("win outcomes must equal successful_evaluations")
        return self


class ResolvedCriteriaScores(MetricsContract):
    base_model: float
    fine_tuned_model: float


class ResolvedSampleResult(MetricsContract):
    id: str
    prompt: str
    base_response: str
    fine_tuned_response: str
    evaluation_status: JudgeEvaluationStatus
    winner: ResolvedWinner | None = None
    judge_winner_label: JudgeWinner | None = None
    answer_order: AnswerOrder
    base_score: float | None = None
    fine_tuned_score: float | None = None
    criteria_scores: dict[str, ResolvedCriteriaScores] | None = None
    judge_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class BenchmarkAnalysis(MetricsContract):
    aggregate_metrics: AggregateMetrics
    samples: list[ResolvedSampleResult]
