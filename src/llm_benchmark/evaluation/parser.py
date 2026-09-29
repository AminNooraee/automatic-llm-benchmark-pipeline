"""Strict parser for structured judge decisions."""

from __future__ import annotations

import json

from pydantic import ValidationError

from llm_benchmark.evaluation.errors import JudgeOutputError
from llm_benchmark.evaluation.models import JudgeDecision


class JudgeOutputParser:
    def parse(self, content: str) -> JudgeDecision:
        if not isinstance(content, str) or not content.strip():
            raise JudgeOutputError("Judge response contains no valid JudgeDecision object")

        decoder = json.JSONDecoder()
        valid: list[JudgeDecision] = []
        first_validation_error: ValidationError | None = None
        for position, character in enumerate(content):
            if character != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(content, position)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                decision = JudgeDecision.model_validate(payload)
            except ValidationError as exc:
                if first_validation_error is None:
                    first_validation_error = exc
                continue
            valid.append(decision)

        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            raise JudgeOutputError(
                "Judge response is ambiguous: multiple valid JudgeDecision objects"
            )
        if first_validation_error is not None:
            details = "; ".join(
                f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
                for error in first_validation_error.errors(include_input=False)
            )
            raise JudgeOutputError(f"Invalid judge response: {details}")
        if "{" not in content:
            raise JudgeOutputError("Judge response is not valid JSON")
        raise JudgeOutputError("Judge response contains no valid JudgeDecision object")
