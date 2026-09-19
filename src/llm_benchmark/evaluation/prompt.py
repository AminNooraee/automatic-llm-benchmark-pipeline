"""Configurable judge prompt loading and safe one-pass rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from llm_benchmark.config.models import EvaluationCriterion
from llm_benchmark.config.validation import REQUIRED_PROMPT_PLACEHOLDERS
from llm_benchmark.evaluation.errors import JudgePromptError


_PLACEHOLDER_PATTERN = re.compile(
    "|".join(re.escape(item) for item in sorted(REQUIRED_PROMPT_PLACEHOLDERS))
)


class JudgePromptTemplate:
    def __init__(self, template: str) -> None:
        missing = sorted(
            placeholder
            for placeholder in REQUIRED_PROMPT_PLACEHOLDERS
            if placeholder not in template
        )
        if missing:
            raise JudgePromptError(
                "Judge prompt template is missing placeholders: " + ", ".join(missing)
            )
        self._template = template

    @classmethod
    def from_file(cls, path: Path) -> "JudgePromptTemplate":
        if not path.is_file():
            raise JudgePromptError(f"Judge prompt template does not exist: {path}")
        try:
            return cls(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            raise JudgePromptError(
                f"Unable to read judge prompt template: {path}"
            ) from exc

    def render(
        self,
        *,
        question: str,
        answer_a: str,
        answer_b: str,
        criteria: Sequence[EvaluationCriterion],
    ) -> str:
        criterion_payload = [
            {
                "name": criterion.name,
                "description": criterion.description,
                "weight": criterion.weight,
                "minimum": criterion.minimum,
                "maximum": criterion.maximum,
            }
            for criterion in criteria
        ]
        criteria_scores = {
            criterion.name: {"A": 0, "B": 0} for criterion in criteria
        }
        output_schema = {
            "winner": "A|B|tie",
            "scores": {"A": 0, "B": 0},
            "criteria_scores": criteria_scores,
            "reason": "Concise explanation",
        }
        values = {
            "{{question}}": question,
            "{{answer_a}}": answer_a,
            "{{answer_b}}": answer_b,
            "{{criteria}}": json.dumps(
                criterion_payload, ensure_ascii=False, indent=2
            ),
            "{{output_schema}}": json.dumps(
                output_schema, ensure_ascii=False, indent=2
            ),
        }
        return _PLACEHOLDER_PATTERN.sub(
            lambda match: values[match.group(0)], self._template
        )
