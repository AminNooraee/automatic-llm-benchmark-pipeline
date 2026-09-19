from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from llm_benchmark.clients.contracts import RetryPolicy
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.models import EndpointConfig, EvaluationCriterion
from llm_benchmark.evaluation.engine import JudgeEngine
from llm_benchmark.evaluation.models import JudgeResult, ResolvedWinner
from llm_benchmark.evaluation.randomization import AnswerRandomizer


def test_judge_engine_uses_mocked_openai_compatible_endpoint(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference" / "responses.jsonl"
    inference_path.parent.mkdir()
    inference_path.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt": "Explain the result",
                "base_response": "Response one",
                "fine_tuned_response": "Response two",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    template_path = tmp_path / "judge.txt"
    template_path.write_text(
        "Question={{question}}\nA={{answer_a}}\nB={{answer_b}}\n"
        "Criteria={{criteria}}\nSchema={{output_schema}}",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.update(body)
        assert str(request.url) == "https://judge.example/v1/chat/completions"
        assert body["model"] == "judge-model"
        assert body["temperature"] == 0
        rendered = body["messages"][0]["content"]
        assert "A=Response two" in rendered
        assert "B=Response one" in rendered
        assert "base-model" not in rendered
        assert "fine-tuned-model" not in rendered
        decision = {
            "winner": "A",
            "scores": {"A": 5, "B": 3},
            "criteria_scores": {"correctness": {"A": 5, "B": 3}},
            "reason": "A is more accurate.",
        }
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "judge-completion-1",
                "model": "served-judge-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(decision),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 15,
                    "total_tokens": 35,
                },
            },
        )

    async def scenario():
        client = OpenAICompatibleClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            retry_policy=RetryPolicy(max_retries=0),
        )
        try:
            return await JudgeEngine(
                client,
                randomizer=AnswerRandomizer(random_value=lambda: 0.9),
            ).run(
                inference_path=inference_path,
                run_dir=tmp_path,
                judge_model=EndpointConfig.model_validate(
                    {
                        "base_url": "https://judge.example/v1",
                        "name": "judge-model",
                        "api_key": None,
                        "timeout_seconds": 2,
                        "generation_parameters": {"temperature": 0},
                    }
                ),
                prompt_template_path=template_path,
                criteria=[
                    EvaluationCriterion(
                        name="correctness",
                        description="Correctness",
                        weight=1,
                    )
                ],
                resume=False,
            )
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert captured["model"] == "judge-model"
    assert result.successful_samples == 1
    stored = JudgeResult.model_validate_json(
        result.artifact_path.read_text(encoding="utf-8").strip()
    )
    assert stored.metadata.judge_model == "served-judge-model"
    assert stored.metadata.resolved_winner == ResolvedWinner.FINE_TUNED
    assert stored.metadata.usage is not None
    assert stored.metadata.usage.total_tokens == 35
