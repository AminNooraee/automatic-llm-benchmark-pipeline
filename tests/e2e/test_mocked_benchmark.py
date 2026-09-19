from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
from pathlib import Path

import httpx

from llm_benchmark.clients.contracts import RetryPolicy
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.config.loader import load_config
from llm_benchmark.observability.logging import configure_logging
from llm_benchmark.orchestrator import BenchmarkPipeline


def test_complete_mocked_benchmark_run_generates_valid_release_artifacts(
    config_factory,
) -> None:
    secret = "end-to-end-secret"
    config_path = config_factory(api_key=secret)
    dataset_path = config_path.parent / "benchmark.jsonl"
    dataset_path.write_text(
        '{"id":"sample-1","prompt":"Explain AI"}\n'
        '{"id":"sample-2","prompt":"Explain ML"}\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        rendered_prompt = body["messages"][0]["content"]
        requests.append((model, rendered_prompt))
        if model == "base-model":
            assert request.url.host == "base.example"
            assert request.headers["authorization"] == f"Bearer {secret}"
            content = f"base-model answer: {rendered_prompt}"
        elif model == "fine-model":
            assert request.url.host == "fine.example"
            assert "authorization" not in request.headers
            content = f"fine-model answer: {rendered_prompt}"
        elif model == "judge-model":
            assert request.url.host == "judge.example"
            assert "Candidate A:" in rendered_prompt
            assert "Candidate B:" in rendered_prompt
            fine_is_a = "Candidate A: fine-model answer:" in rendered_prompt
            winner = "A" if fine_is_a else "B"
            score_a, score_b = ((5, 3) if fine_is_a else (3, 5))
            content = json.dumps(
                {
                    "winner": winner,
                    "scores": {"A": score_a, "B": score_b},
                    "criteria_scores": {
                        "correctness": {"A": score_a, "B": score_b},
                        "clarity": {"A": score_a, "B": score_b},
                    },
                    "reason": "The fine-tuned response is stronger.",
                }
            )
        else:
            raise AssertionError(f"Unexpected model: {model}")
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"completion-{len(requests)}",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    pipeline = BenchmarkPipeline()
    context = pipeline.initialize_run(config, config_path)
    configure_logging(
        level="INFO",
        log_file=context.log_path,
        sensitive_values=config.secret_values(),
        stream=io.StringIO(),
    )
    dataset_result = pipeline.prepare_dataset(config, context)

    async def execute_model_stages():
        client = OpenAICompatibleClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            retry_policy=RetryPolicy(max_retries=0),
        )
        try:
            inference_result = await pipeline.run_inference(
                config,
                context,
                resume=False,
                model_client=client,
            )
            evaluation_result = await pipeline.run_evaluation(
                config,
                context,
                resume=False,
                model_client=client,
            )
            return inference_result, evaluation_result
        finally:
            await client.aclose()

    inference_result, evaluation_result = asyncio.run(execute_model_stages())
    report_result = pipeline.generate_reports(config, context)

    assert dataset_result.sample_count == 2
    assert inference_result.successful_samples == 2
    assert evaluation_result.successful_samples == 2
    assert [model for model, _ in requests].count("base-model") == 2
    assert [model for model, _ in requests].count("fine-model") == 2
    assert [model for model, _ in requests].count("judge-model") == 2

    required_artifacts = [
        "config.snapshot.json",
        "manifest.json",
        "logs/run.jsonl",
        "metadata/environment.json",
        "metadata/artifact_validation.json",
        "metadata/reproducibility.json",
        "dataset/original_dataset.jsonl",
        "dataset/normalized_dataset.jsonl",
        "inference/checkpoints.jsonl",
        "inference/responses.jsonl",
        "evaluation/judge_results.jsonl",
        "reports/report.json",
        "reports/report.md",
        "reports/samples.csv",
    ]
    for relative in required_artifacts:
        assert (context.run_dir / relative).is_file(), relative

    report = json.loads(report_result.report_json_path.read_text(encoding="utf-8"))
    metrics = report["aggregate_metrics"]
    assert metrics["total_samples"] == 2
    assert metrics["successful_evaluations"] == 2
    assert metrics["fine_tuned_model_wins"] == 2
    assert metrics["base_model_wins"] == 0
    assert metrics["average_scores_per_model"] == {
        "base_model": 3.0,
        "fine_tuned_model": 5.0,
    }
    assert all(
        sample["winner"] == "fine_tuned"
        for sample in report["per_sample_results"]
    )

    with report_result.samples_csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert all(row["winner"] == "fine_tuned" for row in csv_rows)

    validation = json.loads(
        (context.run_dir / "metadata/artifact_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["status"] == "passed"
    assert validation["sample_count"] == 2
    assert all(value == "passed" for value in validation["checks"].values())

    reproducibility_path = context.run_dir / "metadata/reproducibility.json"
    reproducibility_text = reproducibility_path.read_text(encoding="utf-8")
    assert secret not in reproducibility_text
    reproducibility = json.loads(reproducibility_text)
    assert reproducibility["pipeline_version"] == "1.0.0"
    assert reproducibility["random_seed"] == 7
    assert reproducibility["config_fingerprint"] == config.fingerprint()
    report_digest = hashlib.sha256(
        report_result.report_json_path.read_bytes()
    ).hexdigest()
    assert (
        reproducibility["artifacts"]["reports/report.json"]["sha256"]
        == report_digest
    )

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["pipeline_version"] == "1.0.0"
    assert manifest["reporting"]["artifact_validation"] == (
        "metadata/artifact_validation.json"
    )
    assert manifest["reporting"]["reproducibility"] == (
        "metadata/reproducibility.json"
    )

    snapshot = context.config_snapshot_path.read_text(encoding="utf-8")
    environment = (context.run_dir / "metadata/environment.json").read_text(
        encoding="utf-8"
    )
    log_text = context.log_path.read_text(encoding="utf-8")
    assert secret not in snapshot
    assert secret not in environment
    assert secret not in log_text
