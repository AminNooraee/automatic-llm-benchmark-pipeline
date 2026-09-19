"""Structured JSON logging for CLI and pipeline execution."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, TextIO


_CONTEXT_FIELDS = (
    "attempt",
    "completed_samples",
    "confidence",
    "dataset_format",
    "event",
    "model_role",
    "model_name",
    "retry_delay_seconds",
    "run_id",
    "sample_count",
    "sample_id",
    "sample_status",
    "schema_format",
    "stage",
    "status_code",
    "total_samples",
)


class JsonFormatter(logging.Formatter):
    def __init__(self, sensitive_values: Iterable[str] = ()) -> None:
        super().__init__()
        self._sensitive_values = tuple(
            sorted((value for value in sensitive_values if value), key=len, reverse=True)
        )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact(record.getMessage()),
        }
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self._redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _redact(self, value: str) -> str:
        redacted = value
        for secret in self._sensitive_values:
            redacted = redacted.replace(secret, "***REDACTED***")
        return redacted


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    sensitive_values: Iterable[str] = (),
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure and return the package logger without touching the root logger."""

    logger = logging.getLogger("llm_benchmark")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = JsonFormatter(sensitive_values)
    stream_handler = logging.StreamHandler(stream or sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
