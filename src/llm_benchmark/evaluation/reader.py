"""Reader for normalized inference response artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from llm_benchmark.evaluation.errors import JudgeArtifactError
from llm_benchmark.evaluation.models import JudgeInput


class InferenceResponseReader:
    def read(self, path: Path) -> list[JudgeInput]:
        if not path.is_file():
            raise JudgeArtifactError(
                f"Inference response artifact does not exist: {path}"
            )

        records: list[JudgeInput] = []
        seen_ids: set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise JudgeArtifactError(
                            f"Malformed inference response '{path.name}' at line "
                            f"{line_number}: {exc.msg}"
                        ) from exc
                    try:
                        record = JudgeInput.model_validate(payload)
                    except ValidationError as exc:
                        details = "; ".join(
                            f"{'.'.join(str(item) for item in error['loc'])}: "
                            f"{error['msg']}"
                            for error in exc.errors(include_input=False)
                        )
                        raise JudgeArtifactError(
                            f"Invalid inference response '{path.name}' at line "
                            f"{line_number}: {details}"
                        ) from exc
                    if record.id in seen_ids:
                        raise JudgeArtifactError(
                            f"Duplicate sample id '{record.id}' in inference response "
                            f"'{path.name}' at line {line_number}"
                        )
                    seen_ids.add(record.id)
                    records.append(record)
        except JudgeArtifactError:
            raise
        except (OSError, UnicodeError) as exc:
            raise JudgeArtifactError(
                f"Unable to read inference response artifact: {path}"
            ) from exc

        if not records:
            raise JudgeArtifactError(f"Inference response artifact is empty: {path}")
        return records
