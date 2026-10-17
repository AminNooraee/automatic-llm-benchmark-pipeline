"""Per-role token-usage aggregation with explicit coverage accounting.

Token totals are only reported as complete when usage was present for every
successful request of that role. A missing usage object is never treated as
zero, and a partial sum is never presented as the full total; incomplete
coverage is surfaced so the report can render ``ناقص``/``ناموجود`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_benchmark.evaluation.models import (
    JudgeEvaluationStatus,
    JudgeResult,
)
from llm_benchmark.inference.models import (
    InferenceResponse,
    ModelResponseStatus,
)


@dataclass(frozen=True)
class RoleUsage:
    """Aggregated token usage for one role with its coverage status."""

    expected_successful: int
    reported: int
    available: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def complete(self) -> bool:
        """Usage was reported for every successful request of this role."""

        return (
            self.expected_successful > 0
            and self.reported == self.expected_successful
            and self.available == self.expected_successful
        )

    @property
    def missing(self) -> bool:
        """No successful request of this role reported any usage."""

        return self.reported == 0

    @property
    def partial(self) -> bool:
        """Some, but not all, successful requests reported usage."""

        return not self.complete and not self.missing


@dataclass(frozen=True)
class TokenUsageSummary:
    base: RoleUsage
    fine_tuned: RoleUsage
    judge: RoleUsage

    @property
    def any_incomplete(self) -> bool:
        return any(role.partial for role in (self.base, self.fine_tuned, self.judge))

    @property
    def judge_complete(self) -> bool:
        return self.judge.complete


def _sum_field(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _aggregate_from_usages(
    statuses: list[bool], usages: list["object"]
) -> RoleUsage:
    """Aggregate a role's usage from per-request success flags and usage objects."""

    successful = [i for i, ok in enumerate(statuses) if ok]
    expected = len(successful)
    reported_indices = [i for i in successful if usages[i] is not None]
    available_indices = [
        i
        for i in reported_indices
        if all(
            getattr(usages[i], field, None) is not None
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    ]
    reported = len(reported_indices)
    available = len(available_indices)

    def _collect(attribute: str) -> list[int | None]:
        out: list[int | None] = []
        for index in reported_indices:
            usage = usages[index]
            out.append(getattr(usage, attribute, None))
        return out

    return RoleUsage(
        expected_successful=expected,
        reported=reported,
        available=available,
        prompt_tokens=_sum_field(_collect("prompt_tokens")),
        completion_tokens=_sum_field(_collect("completion_tokens")),
        total_tokens=_sum_field(_collect("total_tokens")),
    )


def summarize_token_usage(
    inference_responses: list[InferenceResponse],
    judge_results: list[JudgeResult],
) -> TokenUsageSummary:
    base_status = [
        response.metadata.base.status == ModelResponseStatus.SUCCESS
        for response in inference_responses
    ]
    base_usage = [response.metadata.base.usage for response in inference_responses]
    fine_status = [
        response.metadata.fine_tuned.status == ModelResponseStatus.SUCCESS
        for response in inference_responses
    ]
    fine_usage = [response.metadata.fine_tuned.usage for response in inference_responses]

    judge_status = [
        result.metadata.status == JudgeEvaluationStatus.SUCCESS
        for result in judge_results
    ]
    judge_usage = [result.metadata.usage for result in judge_results]

    return TokenUsageSummary(
        base=_aggregate_from_usages(base_status, base_usage),
        fine_tuned=_aggregate_from_usages(fine_status, fine_usage),
        judge=_aggregate_from_usages(judge_status, judge_usage),
    )
