from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_phase1_ci_config_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "ci_phase1.yaml").read_text(encoding="utf-8")
    )

    assert config["dataset"] == {
        "path": "/datasets/phase1_benchmark.json",
        "format": "auto",
    }
    for role in ("base", "fine_tuned"):
        endpoint = config["models"][role]
        assert endpoint["base_url"] == "http://project1-handoff.invalid/v1"
        assert "placeholder" in endpoint["name"]
        assert endpoint["api_key"] == "${LITELLM_API_KEY}"
        assert endpoint["timeout_seconds"] == 60
        assert endpoint["generation_parameters"] == {
            "temperature": 0,
            "max_tokens": 512,
        }

    judge = config["models"]["judge"]
    assert judge == {
        "base_url": "http://172.20.1.52:4000/v1",
        "name": "qwen38-sglang",
        "api_key": "${LITELLM_API_KEY}",
        "timeout_seconds": 120,
        "generation_parameters": {
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    assert {
        criterion["name"]: criterion["weight"]
        for criterion in config["evaluation"]["criteria"]
    } == {"correctness": 0.5, "relevance": 0.3, "clarity": 0.2}
    assert config["runtime"] == {
        "concurrency": 1,
        "max_retries": 2,
        "retry_backoff_seconds": 0.5,
        "random_seed": 42,
        "log_level": "INFO",
    }


def test_gitlab_job_is_downstream_only_and_publishes_exact_run() -> None:
    path = ROOT / ".gitlab-ci.yml"
    pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = pipeline["phase1_benchmark"]
    script = "\n".join(job["script"])

    assert pipeline["variables"] == {
        "BENCHMARK_DATASETS_DIR": "/srv/automatic-llm-benchmark/datasets",
        "BENCHMARK_RUNS_DIR": "/srv/automatic-llm-benchmark/runs",
    }
    assert job["stage"] == "benchmark"
    assert job["resource_group"] == "phase1-benchmark"
    assert job["rules"] == [
        {"if": '$CI_PIPELINE_SOURCE == "pipeline"'},
        {"when": "never"},
    ]
    assert "PROJECT1_HANDOFF_B64 is required" in script
    assert "LITELLM_API_KEY is required" in script
    assert '[ ! -r "$dataset_path" ]' in script
    assert "base64 --decode" in script
    assert "--project1-handoff" in script
    assert "--skip-preflight" not in script
    assert "Run initialized: /workspace/runs/" in script
    assert 'run_dir="$BENCHMARK_RUNS_DIR/$run_id"' in script
    assert "latest" not in script.lower()

    expected_artifacts = {
        "ci_artifacts/benchmark/report.md",
        "ci_artifacts/benchmark/report.json",
        "ci_artifacts/benchmark/samples.csv",
        "ci_artifacts/benchmark/artifact_validation.json",
        "ci_artifacts/benchmark/reproducibility.json",
        "ci_artifacts/benchmark/manifest.json",
    }
    assert set(job["artifacts"]["paths"]) == expected_artifacts
    assert job["artifacts"]["expire_in"] == "30 days"
    assert "image" not in job
