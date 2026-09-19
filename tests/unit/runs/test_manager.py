from __future__ import annotations

import json

import pytest

from llm_benchmark.config.loader import load_config
from llm_benchmark.exceptions import RunInitializationError, RunLifecycleError
from llm_benchmark.runs.manager import RunManager


def test_initialize_creates_isolated_run_with_sanitized_artifacts(
    config_factory,
) -> None:
    secret = "unit-test-secret"
    config_path = config_factory(api_key=secret)
    config = load_config(config_path)

    context = RunManager().initialize(config, config_path, run_id="test-run")

    assert context.run_dir.is_dir()
    assert (context.run_dir / "responses").is_dir()
    assert (context.run_dir / "logs").is_dir()
    assert (context.run_dir / "dataset").is_dir()
    assert (context.run_dir / "inference").is_dir()
    assert (context.run_dir / "evaluation").is_dir()
    assert (context.run_dir / "reports").is_dir()
    assert (context.run_dir / "metadata").is_dir()

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "test-run"
    assert manifest["status"] == "initialized"
    assert manifest["config_fingerprint"] == config.fingerprint()

    snapshot_text = context.config_snapshot_path.read_text(encoding="utf-8")
    assert secret not in snapshot_text
    assert "***REDACTED***" in snapshot_text
    environment_text = (context.run_dir / "metadata" / "environment.json").read_text(
        encoding="utf-8"
    )
    assert secret not in environment_text
    environment = json.loads(environment_text)
    assert environment["pipeline_version"]
    assert environment["python_version"]
    assert "pydantic" in environment["dependencies"]


def test_initialize_never_overwrites_existing_run(config_factory) -> None:
    config_path = config_factory()
    config = load_config(config_path)
    manager = RunManager()
    manager.initialize(config, config_path, run_id="same-run")

    with pytest.raises(RunInitializationError, match="will not be overwritten"):
        manager.initialize(config, config_path, run_id="same-run")


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "..", " space"])
def test_initialize_rejects_unsafe_run_id(config_factory, run_id: str) -> None:
    config_path = config_factory()
    config = load_config(config_path)

    with pytest.raises(RunInitializationError, match="run_id"):
        RunManager().initialize(config, config_path, run_id=run_id)


def test_run_lifecycle_rejects_out_of_order_stage(config_factory) -> None:
    config_path = config_factory()
    config = load_config(config_path)
    manager = RunManager()
    context = manager.initialize(config, config_path, run_id="lifecycle-test")

    with pytest.raises(RunLifecycleError, match="start evaluation"):
        manager.mark_evaluation_started(context)
