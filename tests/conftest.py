from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml


DEFAULT_TEMPLATE = """You are an impartial evaluator.
Criteria:
{{criteria}}
Question: {{question}}
Candidate A: {{answer_a}}
Candidate B: {{answer_b}}
Output: {{output_schema}}
"""


@pytest.fixture
def config_factory(tmp_path: Path) -> Callable[..., Path]:
    def create_config(
        *,
        api_key: str | None = "unit-test-secret",
        dataset_name: str = "benchmark.jsonl",
        prompt_text: str = DEFAULT_TEMPLATE,
        mutate: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        dataset_path = tmp_path / dataset_name
        if dataset_path.suffix == ".jsonl":
            dataset_path.write_text(
                '{"id":"example-1","prompt":"Example prompt"}\n',
                encoding="utf-8",
            )
        template_path = tmp_path / "judge_prompt.txt"
        template_path.write_text(prompt_text, encoding="utf-8")

        data: dict[str, Any] = {
            "version": 1,
            "dataset": {"path": dataset_path.name, "format": "auto"},
            "models": {
                "base": {
                    "base_url": "http://base.example/v1",
                    "name": "base-model",
                    "api_key": api_key,
                    "timeout_seconds": 10,
                    "generation_parameters": {"temperature": 0, "max_tokens": 32},
                },
                "fine_tuned": {
                    "base_url": "http://fine.example/v1",
                    "name": "fine-model",
                    "api_key": None,
                    "timeout_seconds": 10,
                    "generation_parameters": {"temperature": 0, "max_tokens": 32},
                },
                "judge": {
                    "base_url": "https://judge.example/v1",
                    "name": "judge-model",
                    "api_key": None,
                    "timeout_seconds": 10,
                    "generation_parameters": {"temperature": 0, "max_tokens": 64},
                },
            },
            "evaluation": {
                "prompt_template": template_path.name,
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer is correct.",
                        "weight": 0.6,
                        "minimum": 1,
                        "maximum": 5,
                    },
                    {
                        "name": "clarity",
                        "description": "The answer is clear.",
                        "weight": 0.4,
                        "minimum": 1,
                        "maximum": 5,
                    },
                ],
            },
            "runtime": {
                "concurrency": 2,
                "max_retries": 1,
                "retry_backoff_seconds": 0,
                "random_seed": 7,
                "log_level": "INFO",
            },
            "output": {"runs_dir": "runs"},
        }
        if mutate is not None:
            mutate(data)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        return config_path

    return create_config
