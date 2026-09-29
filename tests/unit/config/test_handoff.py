from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_benchmark.config.loader import load_config
from llm_benchmark.exceptions import ConfigurationError


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _write_handoff(tmp_path, mutate=None):
    payload = {
        "schema_version": 1,
        "status": "ready",
        "provider": "litellm",
        "api": "openai-compatible",
        "base_url": "https://gateway.example/v1",
        "models": {
            "base": {"name": "gateway-base"},
            "fine_tuned": {"name": "gateway-fine"},
        },
    }
    if mutate:
        mutate(payload)
    path = tmp_path / "gateway_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_project1_gateway_manifest_overlays_only_base_and_fine(config_factory, tmp_path) -> None:
    config = load_config(
        config_factory(),
        project1_handoff=FIXTURES / "project1_gateway_manifest.json",
    )
    assert str(config.models.base.base_url) == "https://gateway.example/v1"
    assert config.models.base.name == "gateway-base"
    assert config.models.base.api_key is not None
    assert config.models.fine_tuned.name == "gateway-fine"
    assert str(config.models.judge.base_url) == "https://judge.example/v1"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["models"].pop("base"),
        lambda value: value["models"].pop("fine_tuned"),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(base_url="not-a-url"),
        lambda value: value.update(api_key="must-not-be-accepted"),
    ],
)
def test_invalid_handoff_fails_closed(config_factory, tmp_path, mutate) -> None:
    with pytest.raises(ConfigurationError, match="handoff"):
        load_config(config_factory(), project1_handoff=_write_handoff(tmp_path, mutate))


def test_malformed_handoff_fails_clearly(config_factory, tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unable to parse"):
        load_config(config_factory(), project1_handoff=path)


def test_manual_mode_remains_unchanged_without_handoff(config_factory) -> None:
    config = load_config(config_factory())
    assert config.models.base.name == "base-model"
    assert config.models.fine_tuned.name == "fine-model"
    assert config.models.judge.name == "judge-model"


def test_documented_direct_serving_manifest_is_accepted(config_factory, tmp_path) -> None:
    path = _write_handoff(
        tmp_path,
        lambda value: (
            value.pop("provider"),
            value.update(container_name="model-server", restart_policy="unless-stopped"),
        ),
    )
    config = load_config(config_factory(), project1_handoff=path)
    assert config.models.base.name == "gateway-base"
