from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_benchmark.clients.contracts import ChatMessage, GenerationResult
from llm_benchmark.clients.errors import ModelConnectionError
from llm_benchmark.config.models import EndpointConfig
from llm_benchmark.inference.engine import InferenceEngine
from llm_benchmark.inference.models import (
    InferenceMetadata,
    InferenceResponse,
    ModelResponseMetadata,
    ModelResponseStatus,
    SampleInferenceStatus,
)


def _model(name: str) -> EndpointConfig:
    return EndpointConfig.model_validate(
        {
            "base_url": f"https://{name}.example/v1",
            "name": name,
            "api_key": None,
            "timeout_seconds": 2,
            "generation_parameters": {"temperature": 0},
        }
    )


def _write_dataset(path: Path, samples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )


class FakeModelClient:
    def __init__(
        self,
        failures: set[tuple[str, str]] | None = None,
        *,
        delay: float = 0,
    ) -> None:
        self.failures = failures or set()
        self.delay = delay
        self.calls: list[tuple[str, str]] = []
        self.active = 0
        self.maximum_active = 0

    async def generate(
        self,
        model_config: EndpointConfig,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
    ) -> GenerationResult:
        message = messages[0]
        prompt = message.content if isinstance(message, ChatMessage) else message["content"]
        assert isinstance(prompt, str)
        self.calls.append((model_config.name, prompt))
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if (model_config.name, prompt) in self.failures:
                raise ModelConnectionError(model_config.name, attempts=2)
            return GenerationResult(
                content=f"{model_config.name} answered: {prompt}",
                model_name=model_config.name,
                finish_reason="stop",
                latency_ms=1,
                attempts=1,
            )
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        return None


def test_successful_sequential_inference_uses_identical_prompts_and_serializes(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset" / "normalized_dataset.jsonl"
    _write_dataset(
        dataset_path,
        [
            {"id": "one", "prompt": "First prompt"},
            {"id": "two", "prompt": "Second prompt"},
        ],
    )
    client = FakeModelClient()

    result = asyncio.run(
        InferenceEngine(client).run(
            normalized_dataset_path=dataset_path,
            run_dir=tmp_path,
            base_model=_model("base-model"),
            fine_tuned_model=_model("fine-model"),
            concurrency=1,
            resume=False,
        )
    )

    assert client.calls == [
        ("base-model", "First prompt"),
        ("fine-model", "First prompt"),
        ("base-model", "Second prompt"),
        ("fine-model", "Second prompt"),
    ]
    assert result.total_samples == 2
    assert result.successful_samples == 2
    assert result.partial_samples == 0
    assert result.failed_samples == 0
    assert result.attempted_samples == 2
    assert result.skipped_samples == 0
    assert result.artifact_path == tmp_path / "inference" / "responses.jsonl"

    records = [
        json.loads(line)
        for line in result.artifact_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in records] == ["one", "two"]
    assert records[0]["prompt"] == "First prompt"
    assert records[0]["base_response"] == "base-model answered: First prompt"
    assert records[0]["fine_tuned_response"] == "fine-model answered: First prompt"
    assert records[0]["metadata"]["status"] == "success"


def test_both_model_failures_are_collected_without_aborting_the_run(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset" / "normalized_dataset.jsonl"
    _write_dataset(dataset_path, [{"id": "one", "prompt": "Fails"}])
    client = FakeModelClient(
        failures={("base-model", "Fails"), ("fine-model", "Fails")}
    )

    result = asyncio.run(
        InferenceEngine(client).run(
            normalized_dataset_path=dataset_path,
            run_dir=tmp_path,
            base_model=_model("base-model"),
            fine_tuned_model=_model("fine-model"),
            resume=False,
        )
    )

    assert result.failed_samples == 1
    record = InferenceResponse.model_validate_json(
        result.artifact_path.read_text(encoding="utf-8").strip()
    )
    assert record.metadata.status == SampleInferenceStatus.FAILED
    assert record.base_response == ""
    assert record.fine_tuned_response == ""
    assert record.metadata.base.error_type == "ModelConnectionError"
    assert record.metadata.base.attempts == 2


def test_resume_retries_only_failed_role_and_skips_completed_samples(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset" / "normalized_dataset.jsonl"
    _write_dataset(
        dataset_path,
        [
            {"id": "one", "prompt": "Retry me"},
            {"id": "two", "prompt": "Already done"},
        ],
    )
    first_client = FakeModelClient(failures={("base-model", "Retry me")})
    first = asyncio.run(
        InferenceEngine(first_client).run(
            normalized_dataset_path=dataset_path,
            run_dir=tmp_path,
            base_model=_model("base-model"),
            fine_tuned_model=_model("fine-model"),
            concurrency=1,
            resume=False,
        )
    )
    assert first.partial_samples == 1
    assert first.successful_samples == 1

    resumed_client = FakeModelClient()
    resumed = asyncio.run(
        InferenceEngine(resumed_client).run(
            normalized_dataset_path=dataset_path,
            run_dir=tmp_path,
            base_model=_model("base-model"),
            fine_tuned_model=_model("fine-model"),
            concurrency=1,
            resume=True,
        )
    )

    assert resumed_client.calls == [("base-model", "Retry me")]
    assert resumed.successful_samples == 2
    assert resumed.skipped_samples == 1
    assert resumed.attempted_samples == 1
    responses = [
        InferenceResponse.model_validate_json(line)
        for line in resumed.artifact_path.read_text(encoding="utf-8").splitlines()
    ]
    assert responses[0].metadata.fine_tuned.reused is True
    assert responses[1].metadata.base.reused is True
    assert responses[1].metadata.fine_tuned.reused is True


def test_configured_concurrency_limits_parallel_samples(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset" / "normalized_dataset.jsonl"
    _write_dataset(
        dataset_path,
        [{"id": str(index), "prompt": f"Prompt {index}"} for index in range(4)],
    )
    client = FakeModelClient(delay=0.01)

    asyncio.run(
        InferenceEngine(client).run(
            normalized_dataset_path=dataset_path,
            run_dir=tmp_path,
            base_model=_model("base-model"),
            fine_tuned_model=_model("fine-model"),
            concurrency=2,
            resume=False,
        )
    )

    assert client.maximum_active == 2


def test_response_contract_round_trips_through_json() -> None:
    response = InferenceResponse(
        id="sample-1",
        prompt="Explain testing",
        base_response="Base answer",
        fine_tuned_response="Fine-tuned answer",
        metadata=InferenceMetadata(
            status=SampleInferenceStatus.SUCCESS,
            base=ModelResponseMetadata(
                status=ModelResponseStatus.SUCCESS,
                model_name="base-model",
            ),
            fine_tuned=ModelResponseMetadata(
                status=ModelResponseStatus.SUCCESS,
                model_name="fine-model",
            ),
        ),
    )

    payload = response.model_dump_json()
    restored = InferenceResponse.model_validate_json(payload)

    assert restored == response
    assert set(json.loads(payload)) == {
        "id",
        "prompt",
        "base_response",
        "fine_tuned_response",
        "metadata",
    }
