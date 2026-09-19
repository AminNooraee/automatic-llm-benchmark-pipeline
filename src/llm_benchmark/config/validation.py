"""Semantic validation beyond the typed configuration schema."""

from __future__ import annotations

import os

from llm_benchmark.config.models import AppConfig, DatasetFormat
from llm_benchmark.exceptions import ConfigurationError


SUPPORTED_DATASET_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}
REQUIRED_PROMPT_PLACEHOLDERS = {
    "{{answer_a}}",
    "{{answer_b}}",
    "{{criteria}}",
    "{{output_schema}}",
    "{{question}}",
}


def validate_config_semantics(config: AppConfig) -> None:
    """Validate cross-field and filesystem-related configuration constraints."""

    dataset_suffix = config.dataset.path.suffix.lower()
    if (
        config.dataset.format == DatasetFormat.AUTO
        and dataset_suffix not in SUPPORTED_DATASET_EXTENSIONS
    ):
        supported = ", ".join(sorted(SUPPORTED_DATASET_EXTENSIONS))
        raise ConfigurationError(
            f"Dataset error in '{config.dataset.path.name}' at row n/a: "
            f"Unsupported dataset extension '{dataset_suffix}'. "
            f"Detected fields: [none]. Possible formats: [{supported}]"
        )
    if config.dataset.format != DatasetFormat.AUTO:
        expected_suffix = f".{config.dataset.format.value}"
        if dataset_suffix != expected_suffix:
            raise ConfigurationError(
                f"Dataset format '{config.dataset.format.value}' does not match "
                f"file extension '{dataset_suffix}' for: {config.dataset.path}"
            )
    if not config.dataset.path.is_file():
        raise ConfigurationError(
            f"Dataset file does not exist or is not a regular file: "
            f"{config.dataset.path}"
        )

    template_path = config.evaluation.prompt_template
    if template_path.suffix.lower() != ".txt":
        raise ConfigurationError(
            f"Evaluation prompt template must use .txt: {template_path}"
        )
    if not template_path.is_file():
        raise ConfigurationError(
            f"Evaluation prompt template does not exist: {template_path}"
        )

    try:
        template = template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(
            f"Unable to read evaluation prompt template: {template_path}"
        ) from exc

    missing = sorted(
        placeholder
        for placeholder in REQUIRED_PROMPT_PLACEHOLDERS
        if placeholder not in template
    )
    if missing:
        raise ConfigurationError(
            "Evaluation prompt template is missing placeholders: " + ", ".join(missing)
        )

    runs_dir = config.output.runs_dir
    if runs_dir.exists() and not runs_dir.is_dir():
        raise ConfigurationError(f"Configured runs_dir is not a directory: {runs_dir}")
    if runs_dir.exists() and not os.access(runs_dir, os.W_OK):
        raise ConfigurationError(f"Configured runs_dir is not writable: {runs_dir}")
