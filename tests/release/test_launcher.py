from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_launcher_has_safe_mount_and_runtime_contract() -> None:
    text = (ROOT / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8")
    assert "docker.sock" not in text
    assert "--privileged" not in text
    assert '--user "$(id -u):$(id -g)"' in text
    assert "dst=/workspace,readonly" in text
    assert "dst=/workspace/runs" in text
    assert "dst=/datasets,readonly" in text
    assert "dst=/handoff/project1.json,readonly" in text
    assert '"$@"' in text


def test_launcher_forwards_only_named_environment_references_and_options() -> None:
    text = (ROOT / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8")
    assert "grep -o" in text
    assert 'printenv "$variable_name" >/dev/null' in text
    assert 'set -- "$@" -e "$variable_name"' in text
    assert "--project1-handoff /handoff/project1.json" in text
    assert '"$@" --skip-preflight' in text
    assert "env |" not in text
    assert text.index('printenv "$variable_name"') < text.index("docker image inspect")
    assert text.rstrip().endswith('"$@"')


def test_launcher_reuses_an_existing_image_by_default() -> None:
    text = (ROOT / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8")

    assert 'if [ "${BENCHMARK_FORCE_BUILD:-0}" = "1" ] || \\' in text
    assert '! docker image inspect "$image" >/dev/null 2>&1; then' in text
    assert text.index("docker image inspect") < text.index("docker build --tag")


def test_launcher_force_build_uses_the_stable_configured_image_tag() -> None:
    text = (ROOT / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8")

    assert '${BENCHMARK_FORCE_BUILD:-0}' in text
    assert (
        'image=${BENCHMARK_IMAGE:-automatic-llm-benchmark-pipeline:local}' in text
    )
    assert 'docker build --tag "$image" "$repo_dir"' in text
