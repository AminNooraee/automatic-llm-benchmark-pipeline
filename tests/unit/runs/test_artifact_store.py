from pathlib import Path

import pytest

from llm_benchmark.exceptions import RunInitializationError
from llm_benchmark.runs.artifact_store import LocalArtifactStore


def test_artifact_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "run")

    with pytest.raises(RunInitializationError, match="escapes"):
        store.write_json("../outside.json", {"unsafe": True})


def test_artifact_store_writes_text_atomically(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "run")

    path = store.write_text("reports/report.md", "# Report\n")

    assert path.read_text(encoding="utf-8") == "# Report\n"
