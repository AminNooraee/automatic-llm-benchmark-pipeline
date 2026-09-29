from __future__ import annotations

import asyncio
import json

import pytest

from llm_benchmark.clients.contracts import GenerationResult
from llm_benchmark.clients.errors import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelResponseError,
    ModelTimeoutError,
)
from llm_benchmark.config.loader import load_config
from llm_benchmark.preflight import EndpointPreflight, PreflightError


DECISION = json.dumps({
    "winner": "A",
    "scores": {"A": 4, "B": 3},
    "criteria_scores": {
        "correctness": {"A": 4, "B": 3},
        "clarity": {"A": 4, "B": 3},
    },
    "reason": "A follows the instruction.",
})


class FakeClient:
    def __init__(self, failure_role=None, judge_content=DECISION, failure=None):
        self.failure_role = failure_role
        self.judge_content = judge_content
        self.failure = failure
        self.calls = []

    async def generate(self, endpoint, messages):
        role = {"base-model": "base", "fine-model": "fine_tuned", "judge-model": "judge"}[endpoint.name]
        self.calls.append(role)
        if role == self.failure_role:
            if self.failure is not None:
                raise self.failure
            if role == "base":
                raise ModelAuthenticationError(
                    model_name=endpoint.name, status_code=401, message="denied",
                    attempts=1,
                )
            raise ModelTimeoutError(endpoint.name, endpoint.timeout_seconds, 1)
        return GenerationResult(
            content=self.judge_content if role == "judge" else "ready",
            model_name=endpoint.name,
            latency_ms=1,
            attempts=1,
        )

    async def aclose(self):
        pass


def test_preflight_checks_each_role_once(config_factory) -> None:
    client = FakeClient()
    asyncio.run(EndpointPreflight().run(load_config(config_factory()), model_client=client))
    assert client.calls == ["base", "fine_tuned", "judge"]


@pytest.mark.parametrize(
    ("role", "expected_calls"),
    [("base", ["base"]), ("fine_tuned", ["base", "fine_tuned"]), ("judge", ["base", "fine_tuned", "judge"])],
)
def test_preflight_failure_is_role_specific_and_stops(config_factory, role, expected_calls) -> None:
    client = FakeClient(failure_role=role)
    with pytest.raises(PreflightError, match=role):
        asyncio.run(EndpointPreflight().run(load_config(config_factory()), model_client=client))
    assert client.calls == expected_calls


def test_preflight_rejects_invalid_or_ambiguous_judge_output(config_factory) -> None:
    client = FakeClient(judge_content="{}\n{}")
    with pytest.raises(PreflightError, match="judge"):
        asyncio.run(EndpointPreflight().run(load_config(config_factory()), model_client=client))


@pytest.mark.parametrize(
    "failure",
    [
        ModelAPIError(
            model_name="judge-model", status_code=400,
            message="bad request", attempts=1,
        ),
        ModelResponseError(
            "judge-model",
            "Endpoint returned reasoning output but no final assistant content",
            1,
        ),
    ],
)
def test_judge_preflight_classifies_api_and_reasoning_only_failures(
    config_factory, failure
) -> None:
    client = FakeClient(failure_role="judge", failure=failure)
    with pytest.raises(PreflightError, match="judge"):
        asyncio.run(
            EndpointPreflight().run(
                load_config(config_factory()), model_client=client
            )
        )
