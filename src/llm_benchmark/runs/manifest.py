"""Manifest serialization helpers."""

from llm_benchmark.domain.runs import RunManifest


def manifest_payload(manifest: RunManifest) -> dict[str, object]:
    """Return a JSON-compatible manifest payload."""

    return manifest.model_dump(mode="json")

