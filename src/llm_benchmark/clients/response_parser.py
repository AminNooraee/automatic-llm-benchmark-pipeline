"""Strict parsing of the common Chat Completions response surface."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_benchmark.clients.contracts import GenerationResult, GenerationUsage
from llm_benchmark.clients.errors import ModelResponseError


class _CompatibleResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class _ResponseMessage(_CompatibleResponseModel):
    content: str | None = None
    reasoning: object | None = Field(default=None, exclude=True)
    reasoning_content: object | None = Field(default=None, exclude=True)


class _ResponseChoice(_CompatibleResponseModel):
    index: int = 0
    message: _ResponseMessage
    finish_reason: str | None = None


class _ResponseUsage(_CompatibleResponseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class _ChatCompletionResponse(_CompatibleResponseModel):
    id: str | None = None
    model: str | None = None
    choices: list[_ResponseChoice] = Field(min_length=1)
    usage: _ResponseUsage | None = None


def _contains_reasoning_field(value: object, depth: int = 4) -> bool:
    if depth <= 0:
        return False
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if key != "content" and (
                "reason" in normalized or "think" in normalized
            ):
                return True
            if _contains_reasoning_field(item, depth - 1):
                return True
    elif isinstance(value, list):
        return any(_contains_reasoning_field(item, depth - 1) for item in value)
    return False


def parse_chat_completion(
    response: httpx.Response,
    *,
    requested_model: str,
    attempts: int,
    latency_ms: float,
) -> GenerationResult:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelResponseError(
            requested_model, "response body is not valid JSON", attempts
        ) from exc

    try:
        parsed = _ChatCompletionResponse.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ModelResponseError(requested_model, details, attempts) from exc

    choice = parsed.choices[0]
    content = choice.message.content
    if content is None or not content.strip():
        has_reasoning = _contains_reasoning_field(payload)
        reason = (
            "Endpoint returned reasoning output but no final assistant content"
            if has_reasoning
            else "assistant content is null or empty"
        )
        raise ModelResponseError(requested_model, reason, attempts)
    usage = None
    if parsed.usage is not None:
        usage = GenerationUsage(
            prompt_tokens=parsed.usage.prompt_tokens,
            completion_tokens=parsed.usage.completion_tokens,
            total_tokens=parsed.usage.total_tokens,
        )
    return GenerationResult(
        content=content,
        model_name=parsed.model or requested_model,
        finish_reason=choice.finish_reason,
        usage=usage,
        completion_id=parsed.id,
        request_id=(
            response.headers.get("x-request-id")
            or response.headers.get("request-id")
        ),
        latency_ms=latency_ms,
        attempts=attempts,
    )

