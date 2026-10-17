from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_phase1_ci_config_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "ci_phase1.yaml").read_text(encoding="utf-8")
    )

    assert config["dataset"] == {
        "path": "${BENCHMARK_DATASET_PATH}",
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

    assert pipeline["stages"] == ["benchmark"]
    assert pipeline["variables"] == {
        "BENCHMARK_RUNS_DIR": "/srv/automatic-llm-benchmark/runs",
        "GITLAB_INTERNAL_API_URL": "http://172.20.1.52:81/api/v4",
    }
    assert job["stage"] == "benchmark"
    assert job["resource_group"] == "phase1-benchmark"
    assert job["rules"] == [
        {"if": '$CI_PIPELINE_SOURCE == "pipeline"'},
        {"when": "never"},
    ]
    for variable in (
        "PROJECT1_HANDOFF_B64",
        "PROJECT1_UPSTREAM_PROJECT_ID",
        "PROJECT1_UPSTREAM_JOB_ID",
        "PROJECT1_BENCHMARK_FILE",
        "PROJECT1_ARTIFACT_READ_TOKEN",
        "LITELLM_API_KEY",
    ):
        assert f'"${{{variable}:?{variable} is required}}"' in script
    for variable in ("PROJECT1_UPSTREAM_PROJECT_ID", "PROJECT1_UPSTREAM_JOB_ID"):
        assert (
            f'printf \'%s\' "${variable}" | grep -Eq '
            "'^[0-9]*[1-9][0-9]*$'"
        ) in script
        assert f"{variable} must be a positive numeric ID" in script
    assert (
        'case "$PROJECT1_BENCHMARK_FILE" in\n'
        '  benchmark.json|benchmark.jsonl|benchmark.csv|benchmark.parquet) ;;\n'
        '  *) echo "Unsupported benchmark filename: $PROJECT1_BENCHMARK_FILE" >&2; exit 65 ;;\n'
        'esac'
    ) in script
    assert 'command -v curl >/dev/null 2>&1' in script
    assert "curl is required to download the benchmark dataset" in script
    assert 'dataset_dir="$CI_PROJECT_DIR/ci_artifacts/input"' in script
    assert 'rm -rf -- "$dataset_dir"' in script
    assert 'install -d -m 700 "$dataset_dir"' in script
    assert (
        '$GITLAB_INTERNAL_API_URL/projects/$PROJECT1_UPSTREAM_PROJECT_ID/'
        'jobs/$PROJECT1_UPSTREAM_JOB_ID/artifacts/ci_artifacts/job/'
        '$PROJECT1_BENCHMARK_FILE'
    ) in script
    assert "curl --fail --silent --show-error --location" in script
    assert '--header "PRIVATE-TOKEN: $PROJECT1_ARTIFACT_READ_TOKEN"' in script
    assert '--output "$dataset_path" "$artifact_url"' in script
    assert '[ ! -f "$dataset_path" ] || [ ! -s "$dataset_path" ]' in script
    assert 'BENCHMARK_DATASETS_DIR="$dataset_dir"' in script
    assert 'BENCHMARK_DATASET_PATH="/datasets/$PROJECT1_BENCHMARK_FILE"' in script
    assert "export BENCHMARK_DATASETS_DIR BENCHMARK_DATASET_PATH" in script
    assert "/srv/automatic-llm-benchmark/datasets/phase1_benchmark.json" not in script
    assert "set -x" not in script
    assert "BENCHMARK_FORCE_BUILD=1" in script
    assert "export BENCHMARK_FORCE_BUILD" in script
    assert script.index("export BENCHMARK_FORCE_BUILD") < script.index(
        "sh scripts/run_benchmark.sh"
    )
    assert 'echo "$PROJECT1_ARTIFACT_READ_TOKEN"' not in script
    assert 'echo "$LITELLM_API_KEY"' not in script
    assert "base64 --decode" in script
    assert "--project1-handoff" in script
    assert 'rm -f -- "$handoff_file" "$benchmark_log"' in script
    assert "--skip-preflight" not in script
    assert "Run initialized: /workspace/runs/" in script
    assert 'run_dir="$BENCHMARK_RUNS_DIR/$run_id"' in script
    assert "latest" not in script.lower()

    expected_artifacts = {
        "ci_artifacts/benchmark/report.md",
        "ci_artifacts/benchmark/report.json",
        "ci_artifacts/benchmark/samples.csv",
        "ci_artifacts/benchmark/benchmark_summary.html",
        "ci_artifacts/benchmark/benchmark_summary_en.html",
        "ci_artifacts/benchmark/artifact_validation.json",
        "ci_artifacts/benchmark/reproducibility.json",
        "ci_artifacts/benchmark/manifest.json",
    }
    assert set(job["artifacts"]["paths"]) == expected_artifacts
    assert job["artifacts"]["expire_in"] == "30 days"
    assert "image" not in job
