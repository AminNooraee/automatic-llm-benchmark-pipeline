"""Safe local filesystem operations for run artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_benchmark.exceptions import BenchmarkError, RunInitializationError


@dataclass(frozen=True, slots=True)
class JsonlWriteResult:
    path: Path
    record_count: int


class LocalArtifactStore:
    """Write artifacts beneath one run directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        target = self._safe_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RunInitializationError(f"Unable to write run artifact: {target}") from exc
        return target

    def write_jsonl(
        self, relative_path: str | Path, records: Iterable[dict[str, Any]]
    ) -> JsonlWriteResult:
        target = self._safe_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        count = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                for record in records:
                    json.dump(record, temporary, ensure_ascii=False, separators=(",", ":"))
                    temporary.write("\n")
                    count += 1
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if isinstance(exc, BenchmarkError):
                raise
            raise RunInitializationError(f"Unable to write JSONL artifact: {target}") from exc
        return JsonlWriteResult(path=target, record_count=count)

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        target = self._safe_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RunInitializationError(
                f"Unable to write text artifact: {target}"
            ) from exc
        return target

    def copy_file(self, source: Path, relative_path: str | Path) -> Path:
        target = self._safe_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            raise RunInitializationError(f"Artifact source file does not exist: {source}")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            shutil.copy2(source, temporary_path)
            os.replace(temporary_path, target)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RunInitializationError(
                f"Unable to copy dataset artifact to: {target}"
            ) from exc
        return target

    def _safe_target(self, relative_path: str | Path) -> Path:
        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise RunInitializationError("Artifact path must be relative to the run folder")
        target = (self.root / supplied).resolve()
        if not target.is_relative_to(self.root):
            raise RunInitializationError("Artifact path escapes the run folder")
        return target
