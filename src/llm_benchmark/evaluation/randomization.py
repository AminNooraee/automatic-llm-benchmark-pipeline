"""Blind answer-order randomization."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from llm_benchmark.evaluation.models import AnswerOrder, EvaluatedModel


@dataclass(frozen=True, slots=True)
class AnonymizedAnswers:
    answer_a: str
    answer_b: str
    order: AnswerOrder


class AnswerRandomizer:
    def __init__(
        self,
        seed: int | None = None,
        *,
        random_value: Callable[[], float] | None = None,
    ) -> None:
        self._random_value = random_value or random.Random(seed).random

    def arrange(self, base_response: str, fine_tuned_response: str) -> AnonymizedAnswers:
        value = self._random_value()
        if not 0 <= value < 1:
            raise ValueError("random source must return a value in [0, 1)")
        if value < 0.5:
            return AnonymizedAnswers(
                answer_a=base_response,
                answer_b=fine_tuned_response,
                order=AnswerOrder(
                    A=EvaluatedModel.BASE,
                    B=EvaluatedModel.FINE_TUNED,
                ),
            )
        return AnonymizedAnswers(
            answer_a=fine_tuned_response,
            answer_b=base_response,
            order=AnswerOrder(
                A=EvaluatedModel.FINE_TUNED,
                B=EvaluatedModel.BASE,
            ),
        )
