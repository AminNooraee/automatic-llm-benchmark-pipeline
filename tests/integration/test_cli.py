from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_benchmark.clients.contracts import GenerationResult
from llm_benchmark.clients.openai_compatible import OpenAICompatibleClient
from llm_benchmark.cli import main
from llm_benchmark.orchestrator import BenchmarkPipeline


def test_validate_command_does_not_create_run(config_factory, capsys) -> None:
    config_path = config_factory()

    exit_code = main(["validate", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Configuration is valid" in captured.out
    assert not (config_path.parent / "runs").exists()


def test_run_command_executes_pipeline_and_generates_reports(
    config_factory, capsys, monkeypatch
) -> None:
    config_path = config_factory()

    async def fake_generate(self, model_config, messages):
        if model_config.name == "judge-model":
            content = json.dumps(
                {
                    "winner": "B",
                    "scores": {"A": 3, "B": 4},
                    "criteria_scores": {
                        "correctness": {"A": 3, "B": 4},
                        "clarity": {"A": 3, "B": 4},
                    },
                    "reason": "Candidate B is stronger.",
                }
            )
        else:
            content = f"{model_config.name}: {messages[0].content}"
        return GenerationResult(
            content=content,
            model_name=model_config.name,
            finish_reason="stop",
            latency_ms=1,
            attempts=1,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fake_generate)

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Benchmark run completed successfully" in captured.out
    assert "schema=prompt" in captured.out
    assert "Inference completed: success=1, partial=0, failed=0" in captured.out
    assert "Judge evaluation completed: success=1, failed=0, skipped=0" in captured.out

    run_dirs = list((config_path.parent / "runs").iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads(
        (run_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["dataset"]["sample_count"] == 1
    assert manifest["inference"]["successful_samples"] == 1
    assert manifest["evaluation"]["successful_samples"] == 1
    assert manifest["reporting"]["report_json"] == "reports/report.json"
    assert (run_dirs[0] / "dataset" / "original_dataset.jsonl").is_file()
    assert (run_dirs[0] / "dataset" / "normalized_dataset.jsonl").is_file()
    assert (run_dirs[0] / "inference" / "responses.jsonl").is_file()
    assert (run_dirs[0] / "evaluation" / "judge_results.jsonl").is_file()
    assert (run_dirs[0] / "reports" / "report.json").is_file()
    assert (run_dirs[0] / "reports" / "report.md").is_file()
    assert (run_dirs[0] / "reports" / "samples.csv").is_file()
    assert (run_dirs[0] / "metadata" / "environment.json").is_file()
    assert (run_dirs[0] / "metadata" / "artifact_validation.json").is_file()
    assert (run_dirs[0] / "metadata" / "reproducibility.json").is_file()
    log_events = [
        json.loads(line)["event"]
        for line in (run_dirs[0] / "logs" / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "inference_started" in log_events
    assert "inference_sample_completed" in log_events
    assert "inference_completed" in log_events
    assert "evaluation_started" in log_events
    assert "evaluation_sample_completed" in log_events
    assert "evaluation_completed" in log_events
    assert "reporting_started" in log_events
    assert "reporting_completed" in log_events


def test_cli_returns_error_for_invalid_configuration(
    tmp_path: Path, capsys
) -> None:
    missing_path = tmp_path / "missing.yaml"

    exit_code = main(["validate", "--config", str(missing_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration file does not exist" in captured.err


def test_cli_hides_unexpected_internal_error_details(
    config_factory, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = config_factory()

    def fail_initialization(self, config, config_source):
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(BenchmarkPipeline, "initialize_run", fail_initialization)

    exit_code = main(["run", "--config", str(config_path), "--skip-preflight"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "Unexpected internal error" in captured.err
    assert "sensitive internal detail" not in captured.err


def test_failed_preflight_creates_no_run_or_inference(
    config_factory, capsys, monkeypatch
) -> None:
    config_path = config_factory()
    calls = 0

    async def fail_generate(self, model_config, messages):
        nonlocal calls
        calls += 1
        from llm_benchmark.clients.errors import ModelConnectionError
        raise ModelConnectionError(model_config.name, 1)

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fail_generate)
    exit_code = main(["run", "--config", str(config_path)])

    assert exit_code == 2
    assert calls == 1
    assert not (config_path.parent / "runs").exists()
    assert "base endpoint" in capsys.readouterr().err


def test_preflight_command_checks_three_roles_without_creating_run(
    config_factory, capsys, monkeypatch
) -> None:
    config_path = config_factory()
    calls = []

    async def fake_generate(self, model_config, messages):
        calls.append(model_config.name)
        content = (
            json.dumps({
                "winner": "A", "scores": {"A": 4, "B": 3},
                "criteria_scores": {
                    "correctness": {"A": 4, "B": 3},
                    "clarity": {"A": 4, "B": 3},
                },
                "reason": "A follows the instruction.",
            })
            if model_config.name == "judge-model" else "ready"
        )
        return GenerationResult(
            content=content, model_name=model_config.name,
            latency_ms=1, attempts=1,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fake_generate)
    assert main(["preflight", "--config", str(config_path)]) == 0
    assert calls == ["base-model", "fine-model", "judge-model"]
    assert "Preflight succeeded" in capsys.readouterr().out
    assert not (config_path.parent / "runs").exists()


def test_skip_preflight_preserves_direct_run_path(
    config_factory, monkeypatch
) -> None:
    config_path = config_factory()
    calls = []

    async def fake_generate(self, model_config, messages):
        calls.append(model_config.name)
        if model_config.name == "judge-model":
            content = json.dumps({
                "winner": "A", "scores": {"A": 4, "B": 3},
                "criteria_scores": {
                    "correctness": {"A": 4, "B": 3},
                    "clarity": {"A": 4, "B": 3},
                },
                "reason": "A is better.",
            })
        else:
            content = "answer"
        return GenerationResult(
            content=content, model_name=model_config.name,
            latency_ms=1, attempts=1,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "generate", fake_generate)
    assert main(["run", "--config", str(config_path), "--skip-preflight"]) == 0
    assert calls.count("base-model") == 1
    assert calls.count("fine-model") == 1
    assert calls.count("judge-model") == 1
