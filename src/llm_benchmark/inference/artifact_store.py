"""Checkpoint and canonical response storage for resumable inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import ValidationError

from llm_benchmark.inference.errors import InferenceArtifactError
from llm_benchmark.inference.models import InferenceResponse
from llm_benchmark.runs.artifact_store import LocalArtifactStore


class InferenceArtifactStore:
    RESPONSES_RELATIVE = Path("inference/responses.jsonl")
    CHECKPOINT_RELATIVE = Path("inference/checkpoints.jsonl")

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.responses_path = self.run_dir / self.RESPONSES_RELATIVE
        self.checkpoint_path = self.run_dir / self.CHECKPOINT_RELATIVE

    def load_resume_records(self) -> dict[str, InferenceResponse]:
        responses: dict[str, InferenceResponse] = {}
        if self.responses_path.exists():
            self._load_file(self.responses_path, responses, allow_duplicates=False)
        if self.checkpoint_path.exists():
            self._load_file(self.checkpoint_path, responses, allow_duplicates=True)
        return responses

    def ensure_empty_for_new_run(self) -> None:
        existing = [
            path
            for path in (self.responses_path, self.checkpoint_path)
            if path.exists()
        ]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise InferenceArtifactError(
                f"Inference artifacts already exist ({names}); enable resume to reuse them"
            )

    def append_checkpoint(self, response: InferenceResponse) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.checkpoint_path.open("a", encoding="utf-8") as handle:
                json.dump(
                    response.model_dump(mode="json"),
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise InferenceArtifactError(
                f"Unable to append inference checkpoint: {self.checkpoint_path}"
            ) from exc

    def write_responses(self, responses: list[InferenceResponse]) -> Path:
        result = LocalArtifactStore(self.run_dir).write_jsonl(
            self.RESPONSES_RELATIVE,
            (response.model_dump(mode="json") for response in responses),
        )
        return result.path

    @staticmethod
    def _load_file(
        path: Path,
        target: dict[str, InferenceResponse],
        *,
        allow_duplicates: bool,
    ) -> None:
        seen_in_file: set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise InferenceArtifactError(
                            f"Malformed inference artifact '{path.name}' at line "
                            f"{line_number}: {exc.msg}"
                        ) from exc
                    try:
                        response = InferenceResponse.model_validate(payload)
                    except ValidationError as exc:
                        details = "; ".join(
                            f"{'.'.join(str(item) for item in error['loc'])}: "
                            f"{error['msg']}"
                            for error in exc.errors(include_input=False)
                        )
                        raise InferenceArtifactError(
                            f"Invalid inference artifact '{path.name}' at line "
                            f"{line_number}: {details}"
                        ) from exc
                    if not allow_duplicates and response.id in seen_in_file:
                        raise InferenceArtifactError(
                            f"Duplicate sample id '{response.id}' in inference artifact "
                            f"'{path.name}' at line {line_number}"
                        )
                    seen_in_file.add(response.id)
                    target[response.id] = response
        except InferenceArtifactError:
            raise
        except (OSError, UnicodeError) as exc:
            raise InferenceArtifactError(
                f"Unable to read inference artifact: {path}"
            ) from exc

