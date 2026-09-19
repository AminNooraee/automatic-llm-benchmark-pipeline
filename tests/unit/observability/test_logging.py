from __future__ import annotations

import io
import json

from llm_benchmark.observability.logging import configure_logging


def test_structured_logging_emits_json_and_redacts_secrets() -> None:
    stream = io.StringIO()
    logger = configure_logging(
        stream=stream,
        sensitive_values=("secret-value",),
    )

    logger.info(
        "request used secret-value",
        extra={"run_id": "run-1", "stage": "initialization"},
    )

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "INFO"
    assert payload["run_id"] == "run-1"
    assert payload["stage"] == "initialization"
    assert "secret-value" not in payload["message"]
    assert "***REDACTED***" in payload["message"]

