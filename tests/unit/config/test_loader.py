from __future__ import annotations

from pathlib import Path

import pytest

from llm_benchmark.config.loader import load_config
from llm_benchmark.exceptions import ConfigurationError


def test_load_config_resolves_paths_and_environment_secret(
    config_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_TEST_API_KEY", "resolved-secret")
    config_path = config_factory(api_key="${BENCHMARK_TEST_API_KEY}")

    config = load_config(config_path)

    assert config.dataset.path == config_path.parent / "benchmark.jsonl"
    assert config.evaluation.prompt_template == config_path.parent / "judge_prompt.txt"
    assert config.output.runs_dir == config_path.parent / "runs"
    assert config.models.base.api_key is not None
    assert config.models.base.api_key.get_secret_value() == "resolved-secret"


def test_missing_environment_variable_is_rejected(
    config_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_BENCHMARK_KEY", raising=False)
    config_path = config_factory(api_key="${MISSING_BENCHMARK_KEY}")

    with pytest.raises(ConfigurationError, match="MISSING_BENCHMARK_KEY"):
        load_config(config_path)


def test_reserved_request_parameter_is_rejected(config_factory) -> None:
    def mutate(data) -> None:
        data["models"]["base"]["generation_parameters"]["model"] = "override"

    config_path = config_factory(mutate=mutate)

    with pytest.raises(ConfigurationError, match="reserved fields: model"):
        load_config(config_path)


def test_criterion_weights_must_sum_to_one(config_factory) -> None:
    def mutate(data) -> None:
        data["evaluation"]["criteria"][0]["weight"] = 0.7

    config_path = config_factory(mutate=mutate)

    with pytest.raises(ConfigurationError, match="weights must sum to 1.0"):
        load_config(config_path)


def test_missing_prompt_placeholder_is_rejected(config_factory) -> None:
    config_path = config_factory(
        prompt_text=(
            "{{criteria}} {{question}} {{answer_a}} {{answer_b}}"
        )
    )

    with pytest.raises(ConfigurationError, match="output_schema"):
        load_config(config_path)


def test_unsupported_dataset_extension_is_rejected(config_factory) -> None:
    config_path = config_factory(dataset_name="benchmark.txt")

    with pytest.raises(ConfigurationError, match="Unsupported dataset extension"):
        load_config(config_path)


def test_configuration_must_be_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=".yaml or .yml"):
        load_config(config_path)


def test_validation_error_does_not_expose_api_key(config_factory) -> None:
    secret = "highly-sensitive-key"

    def mutate(data) -> None:
        data["models"]["base"]["name"] = ""

    config_path = config_factory(api_key=secret, mutate=mutate)

    with pytest.raises(ConfigurationError) as captured:
        load_config(config_path)

    assert secret not in str(captured.value)


def test_custom_dataset_columns_are_loaded(config_factory) -> None:
    def mutate(data) -> None:
        data["dataset"]["columns"] = {
            "prompt": "payload.question_text",
            "id": "payload.record_id",
        }

    config = load_config(config_factory(mutate=mutate))

    assert config.dataset.columns is not None
    assert config.dataset.columns.prompt == "payload.question_text"
    assert config.dataset.columns.id == "payload.record_id"


def test_model_endpoint_has_name_timeout_and_generation_parameters(
    config_factory,
) -> None:
    config = load_config(config_factory())

    assert config.models.base.name == "base-model"
    assert config.models.base.timeout_seconds == 10
    assert config.models.base.generation_parameters["temperature"] == 0


def test_legacy_model_configuration_names_remain_accepted(config_factory) -> None:
    def mutate(data) -> None:
        endpoint = data["models"]["base"]
        endpoint["model_name"] = endpoint.pop("name")
        endpoint["request_parameters"] = endpoint.pop("generation_parameters")

    config = load_config(config_factory(mutate=mutate))

    assert config.models.base.name == "base-model"
    assert config.models.base.model_name == "base-model"
    assert config.models.base.request_parameters["max_tokens"] == 32


def test_invalid_model_endpoint_is_rejected(config_factory) -> None:
    def mutate(data) -> None:
        data["models"]["base"]["base_url"] = "not-a-valid-url"

    with pytest.raises(ConfigurationError, match="base_url"):
        load_config(config_factory(mutate=mutate))


def test_multiple_completions_are_rejected(config_factory) -> None:
    def mutate(data) -> None:
        data["models"]["base"]["generation_parameters"]["n"] = 2

    with pytest.raises(ConfigurationError, match="must be 1"):
        load_config(config_factory(mutate=mutate))


def test_missing_dataset_file_is_rejected(config_factory) -> None:
    config_path = config_factory()
    (config_path.parent / "benchmark.jsonl").unlink()

    with pytest.raises(ConfigurationError, match="Dataset file does not exist"):
        load_config(config_path)


def test_explicit_dataset_format_must_match_extension(config_factory) -> None:
    def mutate(data) -> None:
        data["dataset"]["format"] = "csv"

    with pytest.raises(ConfigurationError, match="does not match file extension"):
        load_config(config_factory(mutate=mutate))


def test_prompt_template_must_be_a_text_file(config_factory) -> None:
    config_path = config_factory()
    original = config_path.parent / "judge_prompt.txt"
    renamed = config_path.parent / "judge_prompt.md"
    original.rename(renamed)

    def rewrite_prompt_path(data) -> None:
        data["evaluation"]["prompt_template"] = renamed.name

    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rewrite_prompt_path(data)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must use .txt"):
        load_config(config_path)


def test_runtime_concurrency_has_production_upper_bound(config_factory) -> None:
    def mutate(data) -> None:
        data["runtime"]["concurrency"] = 257

    with pytest.raises(ConfigurationError, match="concurrency"):
        load_config(config_factory(mutate=mutate))
