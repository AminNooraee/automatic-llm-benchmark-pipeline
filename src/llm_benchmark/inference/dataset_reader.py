"""Reader for the normalized dataset contract produced by Phase 2."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from llm_benchmark.datasets.models import BenchmarkSample
from llm_benchmark.inference.errors import InferenceArtifactError


class NormalizedDatasetReader:
    def read(self, path: Path) -> list[BenchmarkSample]:
        if not path.is_file():
            raise InferenceArtifactError(
                f"Normalized dataset artifact does not exist: {path}"
            )

        samples: list[BenchmarkSample] = []
        seen_ids: set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise InferenceArtifactError(
                            f"Malformed normalized dataset '{path.name}' at line "
                            f"{line_number}: {exc.msg}"
                        ) from exc
                    try:
                        sample = BenchmarkSample.model_validate(payload)
                    except ValidationError as exc:
                        details = "; ".join(
                            f"{'.'.join(str(item) for item in error['loc'])}: "
                            f"{error['msg']}"
                            for error in exc.errors(include_input=False)
                        )
                        raise InferenceArtifactError(
                            f"Invalid normalized dataset '{path.name}' at line "
                            f"{line_number}: {details}"
                        ) from exc
                    if sample.id in seen_ids:
                        raise InferenceArtifactError(
                            f"Duplicate sample id '{sample.id}' in normalized dataset "
                            f"'{path.name}' at line {line_number}"
                        )
                    seen_ids.add(sample.id)
                    samples.append(sample)
        except InferenceArtifactError:
            raise
        except (OSError, UnicodeError) as exc:
            raise InferenceArtifactError(
                f"Unable to read normalized dataset artifact: {path}"
            ) from exc

        if not samples:
            raise InferenceArtifactError(
                f"Normalized dataset artifact is empty: {path}"
            )
        return samples

