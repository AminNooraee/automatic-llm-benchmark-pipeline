from __future__ import annotations

import json

import pytest

from llm_benchmark.config.handoff import load_optional_report_context, load_report_context
from llm_benchmark.exceptions import ConfigurationError


def test_optional_report_context_supports_old_runs_and_strict_new_payload(tmp_path) -> None:
    assert load_optional_report_context(None) is None
    path = tmp_path / "report_context.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_duration_seconds": 12.5,
                "training": {
                    "model_name": "org/model",
                    "epochs": 2,
                    "learning_rate": 0.0001,
                    "batch_size": 4,
                    "cutoff_len": 2048,
                    "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
                },
            }
        ),
        encoding="utf-8",
    )
    context = load_report_context(path)
    assert context.training.model_name == "org/model"
    assert context.training_duration_seconds == 12.5

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["api_key"] = "must-not-be-accepted"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="validation failed"):
        load_report_context(path)
