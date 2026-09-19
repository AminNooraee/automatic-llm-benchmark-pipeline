"""Strict parser for structured judge decisions."""

from __future__ import annotations

import json

from pydantic import ValidationError

from llm_benchmark.evaluation.errors import JudgeOutputError
from llm_benchmark.evaluation.models import JudgeDecision


class JudgeOutputParser:
    def parse(self, content: str) -> JudgeDecision:
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise JudgeOutputError("Judge response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise JudgeOutputError("Judge response JSON must be an object")
        try:
            return JudgeDecision.model_validate(payload)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False)
            )
            raise JudgeOutputError(f"Invalid judge response: {details}") from exc
