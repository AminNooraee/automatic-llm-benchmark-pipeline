"""Consistent, contextual error messages for dataset failures."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def dataset_error_message(
    reason: str,
    *,
    file_path: Path,
    row_number: int | None = None,
    fields: Iterable[str] = (),
    possible_formats: Iterable[str] = (),
) -> str:
    row = "n/a" if row_number is None else str(row_number)
    field_list = sorted({str(field) for field in fields})
    format_list = sorted({str(item) for item in possible_formats})
    detected = ", ".join(field_list) if field_list else "none"
    possible = ", ".join(format_list) if format_list else "none"
    return (
        f"Dataset error in '{file_path.name}' at row {row}: {reason}. "
        f"Detected fields: [{detected}]. Possible formats: [{possible}]"
    )

