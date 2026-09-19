"""Provider-independent OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from llm_benchmark import __version__
from llm_benchmark.clients.contracts import ChatMessage, GenerationResult, RetryPolicy
from llm_benchmark.clients.errors import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRequestValidationError,
    ModelTimeoutError,
)
from llm_benchmark.clients.response_parser import parse_chat_completion
from llm_benchmark.config.models import EndpointConfig


_MESSAGE_LIST_ADAPTER = TypeAdapter(list[ChatMessage])
_RETRYABLE_STATUS_CODES = {408, 409, 429}
_LOGGER = logging.getLogger("llm_benchmark.clients.openai_compatible")


class OpenAICompatibleClient:
    """Call the common non-streaming Chat Completions API surface."""

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self._retry_policy = retry_policy or RetryPolicy()
        self._http_client = http_client or httpx.AsyncClient(follow_redirects=False)
        self._sleep = sleep
        self._random_value = random_value

    async def __aenter__(self) -> "OpenAICompatibleClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def generate(
        self,
        model_config: EndpointConfig,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
    ) -> GenerationResult:
        validated_messages = self._validate_messages(messages)
        url = f"{str(model_config.base_url).rstrip('/')}/chat/completions"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"automatic-llm-benchmark-pipeline/{__version__}",
        }
        api_key = None
        if model_config.api_key is not None:
            api_key = model_config.api_key.get_secret_value()
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model_config.name,
            "messages": [message.model_dump(exclude_none=True) for message in validated_messages],
            **model_config.generation_parameters,
        }
        start = time.perf_counter()
        maximum_attempts = self._retry_policy.max_retries + 1

        for attempt in range(1, maximum_attempts + 1):
            _LOGGER.debug(
                "Sending Chat Completions request",
                extra={
                    "attempt": attempt,
                    "event": "model_request_attempt",
                    "model_name": model_config.name,
                },
            )
            try:
                response = await self._http_client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(model_config.timeout_seconds),
                )
            except httpx.TimeoutException as exc:
                if attempt < maximum_attempts:
                    await self._wait_before_retry(attempt, None, model_config.name)
                    continue
                _LOGGER.error(
                    "Model request timed out",
                    extra={
                        "attempt": attempt,
                        "event": "model_request_failed",
                        "model_name": model_config.name,
                    },
                )
                raise ModelTimeoutError(
                    model_config.name, model_config.timeout_seconds, attempt
                ) from exc
            except httpx.RequestError as exc:
                if attempt < maximum_attempts:
                    await self._wait_before_retry(attempt, None, model_config.name)
                    continue
                _LOGGER.error(
                    "Model endpoint connection failed",
                    extra={
                        "attempt": attempt,
                        "event": "model_request_failed",
                        "model_name": model_config.name,
                    },
                )
                raise ModelConnectionError(model_config.name, attempt) from exc

            if not 200 <= response.status_code < 300:
                error_message, error_type, error_code = self._parse_api_error(
                    response, api_key
                )
                if (
                    self._is_retryable_status(response.status_code)
                    and attempt < maximum_attempts
                ):
                    await self._wait_before_retry(
                        attempt, response, model_config.name
                    )
                    continue

                error_class = (
                    ModelAuthenticationError
                    if response.status_code in {401, 403}
                    else ModelAPIError
                )
                _LOGGER.error(
                    "Model API returned an error",
                    extra={
                        "attempt": attempt,
                        "event": "model_request_failed",
                        "model_name": model_config.name,
                        "status_code": response.status_code,
                    },
                )
                raise error_class(
                    model_name=model_config.name,
                    status_code=response.status_code,
                    message=error_message,
                    attempts=attempt,
                    error_type=error_type,
                    error_code=error_code,
                )

            latency_ms = (time.perf_counter() - start) * 1000
            result = parse_chat_completion(
                response,
                requested_model=model_config.name,
                attempts=attempt,
                latency_ms=latency_ms,
            )
            _LOGGER.info(
                "Chat Completions request succeeded",
                extra={
                    "attempt": attempt,
                    "event": "model_request_succeeded",
                    "model_name": model_config.name,
                    "status_code": response.status_code,
                },
            )
            return result

        raise AssertionError("retry loop exited unexpectedly")

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code < 600

    @staticmethod
    def _validate_messages(
        messages: Sequence[ChatMessage | Mapping[str, Any]],
    ) -> list[ChatMessage]:
        if isinstance(messages, (str, bytes)) or not messages:
            raise ModelRequestValidationError(
                "messages must be a non-empty sequence of chat messages"
            )
        try:
            validated = _MESSAGE_LIST_ADAPTER.validate_python(list(messages))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False)
            )
            raise ModelRequestValidationError(
                f"Invalid chat messages: {details}"
            ) from exc
        if not any(message.role == "user" for message in validated):
            raise ModelRequestValidationError(
                "messages must contain at least one user message"
            )
        return validated

    async def _wait_before_retry(
        self,
        attempt: int,
        response: httpx.Response | None,
        model_name: str,
    ) -> None:
        delay = self._retry_delay(attempt, response)
        _LOGGER.warning(
            "Retrying model request after transient failure",
            extra={
                "attempt": attempt,
                "event": "model_request_retry",
                "model_name": model_name,
                "retry_delay_seconds": delay,
                "status_code": response.status_code if response is not None else None,
            },
        )
        await self._sleep(delay)

    def _retry_delay(
        self, attempt: int, response: httpx.Response | None
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    parsed = float(retry_after)
                except ValueError:
                    pass
                else:
                    return min(
                        max(parsed, 0), self._retry_policy.maximum_backoff_seconds
                    )

        base = min(
            self._retry_policy.initial_backoff_seconds * (2 ** (attempt - 1)),
            self._retry_policy.maximum_backoff_seconds,
        )
        jitter = (
            (self._random_value() * 2) - 1
        ) * self._retry_policy.jitter_ratio
        return max(0, base * (1 + jitter))

    @staticmethod
    def _parse_api_error(
        response: httpx.Response, api_key: str | None
    ) -> tuple[str, str | None, str | None]:
        message = response.reason_phrase or "API request failed"
        error_type = None
        error_code = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                if isinstance(error.get("message"), str):
                    message = error["message"]
                if isinstance(error.get("type"), str):
                    error_type = error["type"]
                if error.get("code") is not None:
                    error_code = str(error["code"])

        if api_key:
            message = message.replace(api_key, "***REDACTED***")
        return message[:500], error_type, error_code
