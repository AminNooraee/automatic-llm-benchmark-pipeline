from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable

import httpx
import pytest

from llm_benchmark.clients.contracts import RetryPolicy
from llm_benchmark.clients.errors import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRequestValidationError,
    ModelResponseError,
    ModelTimeoutError,
)
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.models import EndpointConfig
from llm_benchmark.observability.logging import configure_logging


def _model_config(**overrides) -> EndpointConfig:
    values = {
        "base_url": "https://gateway.example/v1",
        "name": "test-model",
        "api_key": "test-api-key",
        "timeout_seconds": 2,
        "generation_parameters": {"temperature": 0, "max_tokens": 32},
    }
    values.update(overrides)
    return EndpointConfig.model_validate(values)


def _success_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        headers={"x-request-id": "request-123"},
        json={
            "id": "chatcmpl-123",
            "model": "served-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello back"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
    )


def _run_client(
    handler: Callable[[httpx.Request], httpx.Response],
    operation,
):
    async def scenario():
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = OpenAICompatibleClient(
            http_client=http_client,
            retry_policy=RetryPolicy(
                max_retries=0,
                initial_backoff_seconds=0,
                maximum_backoff_seconds=0,
                jitter_ratio=0,
            ),
        )
        try:
            return await operation(client)
        finally:
            await client.aclose()

    return asyncio.run(scenario())


def test_successful_generation_sends_common_chat_completions_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-api-key"
        assert request.extensions["timeout"]["read"] == 2
        body = json.loads(request.content)
        assert body == {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 32,
        }
        return _success_response(request)

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(), [{"role": "user", "content": "Hello"}]
        )

    result = _run_client(handler, operation)

    assert result.content == "Hello back"
    assert result.model_name == "served-model"
    assert result.finish_reason == "stop"
    assert result.completion_id == "chatcmpl-123"
    assert result.request_id == "request-123"
    assert result.attempts == 1
    assert result.usage is not None
    assert result.usage.total_tokens == 6


def test_api_key_is_optional_and_authorization_header_is_omitted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return _success_response(request)

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(api_key=None),
            [{"role": "user", "content": "Hello"}],
        )

    result = _run_client(handler, operation)
    assert result.content == "Hello back"


def test_invalid_endpoint_connection_is_normalized() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("unreachable", request=request)

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(), [{"role": "user", "content": "Hello"}]
        )

    with pytest.raises(ModelConnectionError) as captured:
        _run_client(handler, operation)

    assert captured.value.attempts == 1
    assert calls == 1
    assert "test-api-key" not in str(captured.value)


def test_authentication_failure_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            request=request,
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )

    async def scenario():
        client = OpenAICompatibleClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            retry_policy=RetryPolicy(max_retries=3),
        )
        try:
            await client.generate(
                _model_config(), [{"role": "user", "content": "Hello"}]
            )
        finally:
            await client.aclose()

    with pytest.raises(ModelAuthenticationError) as captured:
        asyncio.run(scenario())

    assert captured.value.status_code == 401
    assert captured.value.error_code == "invalid_api_key"
    assert calls == 1


def test_timeout_is_retried_then_normalized() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("too slow", request=request)

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario():
        client = OpenAICompatibleClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            retry_policy=RetryPolicy(
                max_retries=1,
                initial_backoff_seconds=0,
                maximum_backoff_seconds=0,
                jitter_ratio=0,
            ),
            sleep=no_sleep,
        )
        try:
            await client.generate(
                _model_config(timeout_seconds=0.01),
                [{"role": "user", "content": "Hello"}],
            )
        finally:
            await client.aclose()

    with pytest.raises(ModelTimeoutError) as captured:
        asyncio.run(scenario())

    assert captured.value.attempts == 2
    assert calls == 2
    assert delays == [0]


def test_malformed_success_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"id": "chatcmpl-broken", "model": "test-model", "choices": []},
        )

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(), [{"role": "user", "content": "Hello"}]
        )

    with pytest.raises(ModelResponseError, match="choices"):
        _run_client(handler, operation)


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_metadata_is_tolerated_when_final_content_exists(field: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = _success_response(request)
        payload = response.json()
        payload["choices"][0]["message"][field] = "private reasoning"
        payload["choices"][0]["message"]["provider_extension"] = {"opaque": True}
        return httpx.Response(200, request=request, json=payload)

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(), [{"role": "user", "content": "Hello"}]
        )

    assert _run_client(handler, operation).content == "Hello back"


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_only_response_has_normalized_non_sensitive_error(field: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": None, field: "do not persist"}}],
            },
        )

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(), [{"role": "user", "content": "Hello"}]
        )

    with pytest.raises(
        ModelResponseError,
        match="Endpoint returned reasoning output but no final assistant content",
    ) as captured:
        _run_client(handler, operation)
    assert "do not persist" not in str(captured.value)


def test_top_level_reasoning_metadata_with_null_content_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "test-model",
                "provider_reasoning_metadata": {"opaque": True},
                "choices": [{"message": {"content": None}}],
            },
        )

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(), [{"role": "user", "content": "Hello"}]
        )

    with pytest.raises(ModelResponseError, match="reasoning output"):
        _run_client(handler, operation)


def test_retryable_status_uses_retry_after_then_succeeds() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                request=request,
                headers={"retry-after": "0"},
                json={"error": {"message": "Rate limited"}},
            )
        return _success_response(request)

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario():
        client = OpenAICompatibleClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            retry_policy=RetryPolicy(
                max_retries=2,
                initial_backoff_seconds=1,
                maximum_backoff_seconds=4,
                jitter_ratio=0,
            ),
            sleep=no_sleep,
        )
        try:
            return await client.generate(
                _model_config(), [{"role": "user", "content": "Hello"}]
            )
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert result.attempts == 2
    assert calls == 2
    assert delays == [0]


def test_non_retryable_api_error_and_logs_redact_secret() -> None:
    secret = "top-secret-key"
    stream = io.StringIO()
    configure_logging(stream=stream, sensitive_values=(secret,))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            json={"error": {"message": f"Request included {secret}"}},
        )

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(
            _model_config(api_key=secret),
            [{"role": "user", "content": "Hello"}],
        )

    with pytest.raises(ModelAPIError) as captured:
        _run_client(handler, operation)

    assert secret not in str(captured.value)
    assert secret not in stream.getvalue()


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "No user message"}],
        [{"role": "user", "content": "   "}],
    ],
)
def test_invalid_messages_are_rejected_before_network(messages) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response(request)

    async def operation(client: OpenAICompatibleClient):
        return await client.generate(_model_config(), messages)

    with pytest.raises(ModelRequestValidationError):
        _run_client(handler, operation)
    assert calls == 0
