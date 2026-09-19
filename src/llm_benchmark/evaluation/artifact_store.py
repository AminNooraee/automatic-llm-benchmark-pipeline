"""Canonical storage for judge evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from llm_benchmark.evaluation.errors import JudgeArtifactError
from llm_benchmark.evaluation.models import JudgeResult
from llm_benchmark.runs.artifact_store import LocalArtifactStore


class JudgeArtifactStore:
    RESULTS_RELATIVE = Path("evaluation/judge_results.jsonl")

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.results_path = self.run_dir / self.RESULTS_RELATIVE

    def ensure_empty_for_new_run(self) -> None:
        if self.results_path.exists():
            raise JudgeArtifactError(
                "Judge results already exist; enable resume to reuse them"
            )

    def load_results(self) -> dict[str, JudgeResult]:
        if not self.results_path.exists():
            return {}
        results: dict[str, JudgeResult] = {}
        try:
            with self.results_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise JudgeArtifactError(
                            f"Malformed judge artifact '{self.results_path.name}' at "
                            f"line {line_number}: {exc.msg}"
                        ) from exc
                    try:
                        result = JudgeResult.model_validate(payload)
                    except ValidationError as exc:
                        details = "; ".join(
                            f"{'.'.join(str(item) for item in error['loc'])}: "
                            f"{error['msg']}"
                            for error in exc.errors(include_input=False)
                        )
                        raise JudgeArtifactError(
                            f"Invalid judge artifact '{self.results_path.name}' at "
                            f"line {line_number}: {details}"
                        ) from exc
                    if result.id in results:
                        raise JudgeArtifactError(
                            f"Duplicate sample id '{result.id}' in judge artifact "
                            f"'{self.results_path.name}' at line {line_number}"
                        )
                    results[result.id] = result
        except JudgeArtifactError:
            raise
        except (OSError, UnicodeError) as exc:
            raise JudgeArtifactError(
                f"Unable to read judge artifact: {self.results_path}"
            ) from exc
        return results

    def write_results(self, results: list[JudgeResult]) -> Path:
        written = LocalArtifactStore(self.run_dir).write_jsonl(
            self.RESULTS_RELATIVE,
            (result.model_dump(mode="json") for result in results),
        )
        return written.path
