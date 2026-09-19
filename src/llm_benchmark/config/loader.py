"""YAML configuration loading with environment and path resolution."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm_benchmark.config.models import AppConfig
from llm_benchmark.config.validation import validate_config_semantics
from llm_benchmark.exceptions import ConfigurationError


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_config(path: str | Path) -> AppConfig:
    """Load, resolve, and validate a benchmark YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Configuration file must use .yaml or .yml")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to parse configuration: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    expanded = _expand_environment(copy.deepcopy(raw))
    resolved = _resolve_paths(expanded, config_path.parent)
    secret_values = _extract_raw_secrets(resolved)

    try:
        config = AppConfig.model_validate(resolved)
    except ValidationError as exc:
        message = _redact_text(str(exc), secret_values)
        raise ConfigurationError(f"Configuration validation failed:\n{message}") from exc

    validate_config_semantics(config)
    return config


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable not in os.environ:
            raise ConfigurationError(
                f"Environment variable referenced by configuration is not set: {variable}"
            )
        return os.environ[variable]

    return _ENV_PATTERN.sub(replace, value)


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    path_locations = (
        ("dataset", "path"),
        ("evaluation", "prompt_template"),
        ("output", "runs_dir"),
    )
    for section, field in path_locations:
        section_value = raw.get(section)
        if not isinstance(section_value, dict) or field not in section_value:
            continue
        raw_path = section_value[field]
        if not isinstance(raw_path, (str, Path)):
            continue
        resolved = Path(raw_path).expanduser()
        if not resolved.is_absolute():
            resolved = base_dir / resolved
        section_value[field] = str(resolved.resolve(strict=False))
    return raw


def _extract_raw_secrets(raw: dict[str, Any]) -> tuple[str, ...]:
    secrets: list[str] = []
    models = raw.get("models")
    if not isinstance(models, dict):
        return ()
    for endpoint in models.values():
        if not isinstance(endpoint, dict):
            continue
        api_key = endpoint.get("api_key")
        if isinstance(api_key, str) and api_key:
            secrets.append(api_key)
    return tuple(secrets)


def _redact_text(message: str, secrets: tuple[str, ...]) -> str:
    redacted = message
    for secret in secrets:
        redacted = redacted.replace(secret, "***REDACTED***")
    return redacted

