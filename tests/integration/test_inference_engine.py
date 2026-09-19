from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from llm_benchmark.clients.contracts import RetryPolicy
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.models import EndpointConfig
from llm_benchmark.inference.engine import InferenceEngine


def _endpoint(base_url: str, name: str) -> EndpointConfig:
    return EndpointConfig.model_validate(
        {
            "base_url": base_url,
            "name": name,
            "api_key": None,
            "timeout_seconds": 2,
            "generation_parameters": {"temperature": 0, "max_tokens": 24},
        }
    )


def test_inference_engine_calls_mocked_openai_compatible_endpoints(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset" / "normalized_dataset.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text(
        '{"id":"one","prompt":"Explain AI"}\n'
        '{"id":"two","prompt":"Explain ML"}\n',
        encoding="utf-8",
    )
    requests: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["temperature"] == 0
        assert body["max_tokens"] == 24
        prompt = body["messages"][0]["content"]
        requests.append((request.url.host, body["model"], prompt))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"completion-{len(requests)}",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"{body['model']} response to {prompt}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 4,
                    "total_tokens": 6,
                },
            },
        )

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
            return await InferenceEngine(client).run(
                normalized_dataset_path=dataset_path,
                run_dir=tmp_path,
                base_model=_endpoint("https://base.example/v1", "base-model"),
                fine_tuned_model=_endpoint(
                    "https://fine.example/v1", "fine-tuned-model"
                ),
                concurrency=1,
                resume=False,
            )
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert requests == [
        ("base.example", "base-model", "Explain AI"),
        ("fine.example", "fine-tuned-model", "Explain AI"),
        ("base.example", "base-model", "Explain ML"),
        ("fine.example", "fine-tuned-model", "Explain ML"),
    ]
    assert result.successful_samples == 2
    serialized = [
        json.loads(line)
        for line in result.artifact_path.read_text(encoding="utf-8").splitlines()
    ]
    assert serialized[0]["base_response"] == "base-model response to Explain AI"
    assert (
        serialized[0]["fine_tuned_response"]
        == "fine-tuned-model response to Explain AI"
    )
