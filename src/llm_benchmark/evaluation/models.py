"""Contracts for anonymous pairwise judge evaluation."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_benchmark.clients.contracts import GenerationUsage


class EvaluationContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EvaluatedModel(str, Enum):
    BASE = "base"
    FINE_TUNED = "fine_tuned"


class JudgeWinner(str, Enum):
    A = "A"
    B = "B"
    TIE = "tie"


class ResolvedWinner(str, Enum):
    BASE = "base"
    FINE_TUNED = "fine_tuned"
    TIE = "tie"


class JudgeEvaluationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class AnswerScores(EvaluationContract):
    A: float = Field(ge=0, allow_inf_nan=False)
    B: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("A", "B", mode="before")
    @classmethod
    def reject_non_numeric_scores(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be a finite non-negative number")
        if not math.isfinite(float(value)):
            raise ValueError("score must be a finite non-negative number")
        return value


class JudgeDecision(EvaluationContract):
    winner: JudgeWinner
    scores: AnswerScores
    criteria_scores: dict[str, AnswerScores]
    reason: str = Field(min_length=1)


class AnswerOrder(EvaluationContract):
    A: EvaluatedModel
    B: EvaluatedModel

    @model_validator(mode="after")
    def validate_distinct_models(self) -> "AnswerOrder":
        if self.A == self.B:
            raise ValueError("answer order must contain each evaluated model once")
        return self

    def resolve(self, winner: JudgeWinner) -> ResolvedWinner:
        if winner == JudgeWinner.TIE:
            return ResolvedWinner.TIE
        selected = self.A if winner == JudgeWinner.A else self.B
        return ResolvedWinner(selected.value)


class JudgeResultMetadata(EvaluationContract):
    status: JudgeEvaluationStatus
    answer_order: AnswerOrder
    resolved_winner: ResolvedWinner | None = None
    judge_model: str
    latency_ms: float | None = Field(default=None, ge=0)
    attempts: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    usage: GenerationUsage | None = None
    error_type: str | None = None
    error_message: str | None = None
    reused: bool = False

    @model_validator(mode="after")
    def validate_status_fields(self) -> "JudgeResultMetadata":
        if self.status == JudgeEvaluationStatus.SUCCESS:
            if self.resolved_winner is None:
                raise ValueError("successful evaluation requires a resolved winner")
            if self.error_message is not None:
                raise ValueError("successful evaluation cannot contain an error")
            if self.attempts < 1:
                raise ValueError("successful evaluation requires at least one attempt")
        else:
            if self.resolved_winner is not None:
                raise ValueError("unsuccessful evaluation cannot resolve a winner")
            if not self.error_message:
                raise ValueError("unsuccessful evaluation requires an error message")
        if self.status == JudgeEvaluationStatus.SKIPPED and self.attempts != 0:
            raise ValueError("skipped evaluation cannot contain judge attempts")
        return self


class JudgeResult(EvaluationContract):
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    winner: JudgeWinner | None = None
    scores: AnswerScores | None = None
    criteria_scores: dict[str, AnswerScores] | None = None
    reason: str | None = None
    metadata: JudgeResultMetadata

    @model_validator(mode="after")
    def validate_decision_presence(self) -> "JudgeResult":
        decision_fields = (
            self.winner,
            self.scores,
            self.criteria_scores,
            self.reason,
        )
        if self.metadata.status == JudgeEvaluationStatus.SUCCESS:
            if any(value is None for value in decision_fields):
                raise ValueError("successful evaluation requires all decision fields")
            assert self.winner is not None
            expected = self.metadata.answer_order.resolve(self.winner)
            if self.metadata.resolved_winner != expected:
                raise ValueError("resolved winner does not match the blind answer order")
        elif any(value is not None for value in decision_fields):
            raise ValueError("unsuccessful evaluation cannot contain a decision")
        return self


class JudgeRunResult(EvaluationContract):
    artifact_path: Path
    total_samples: int = Field(ge=0)
    successful_samples: int = Field(ge=0)
    failed_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    reused_samples: int = Field(ge=0)
    attempted_samples: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "JudgeRunResult":
        if (
            self.successful_samples + self.failed_samples + self.skipped_samples
            != self.total_samples
        ):
            raise ValueError("evaluation outcome counts must equal total_samples")
        accounted = (
            self.reused_samples + self.attempted_samples + self.skipped_samples
        )
        if accounted != self.total_samples:
            raise ValueError(
                "reused, attempted, and skipped samples must equal total_samples"
            )
        return self


class JudgeInput(EvaluationContract):
    """Minimal contract read from the Phase 4 response artifact."""

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    base_response: str
    fine_tuned_response: str
    metadata: dict[str, object] | None = None
