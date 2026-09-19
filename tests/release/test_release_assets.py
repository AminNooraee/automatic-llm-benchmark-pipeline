from __future__ import annotations

import tomllib
from pathlib import Path

from llm_benchmark import __version__


ROOT = Path(__file__).resolve().parents[2]


def test_release_versions_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == "1.0.0"
    assert pyproject["project"]["version"] == __version__


def test_docker_image_is_a_non_root_external_endpoint_runner() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "FROM python:" in dockerfile
    assert "USER benchmark" in dockerfile
    assert 'ENTRYPOINT ["llm-benchmark"]' in dockerfile
    assert "EXPOSE" not in dockerfile
    assert "tests" in dockerignore
    assert "runs" in dockerignore
    assert ".env" in dockerignore


def test_release_documentation_set_is_present() -> None:
    required = [
        "README.md",
        "CHANGELOG.md",
        "RELEASE_CHECKLIST.md",
        "docs/architecture.md",
        "docs/configuration.md",
        "docs/usage.md",
        "docs/benchmark-workflow.md",
        "docs/docker.md",
    ]

    for relative in required:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.read_text(encoding="utf-8").strip(), relative
