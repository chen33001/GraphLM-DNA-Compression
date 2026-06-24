"""Versioned JSON archive serialization."""

import json
from pathlib import Path
from typing import Any

MAGIC = "GHDNA_1"


def _validate_archive(archive: dict[str, Any]) -> None:
    if archive.get("magic") != MAGIC:
        raise ValueError("invalid archive magic")
    if not isinstance(archive.get("blocks"), list):
        raise ValueError("archive blocks must be a list")


def write_archive(archive: dict[str, Any], path: str | Path) -> None:
    _validate_archive(archive)
    Path(path).write_text(
        json.dumps(archive, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )


def read_archive(path: str | Path) -> dict[str, Any]:
    try:
        archive = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid archive JSON") from error
    if not isinstance(archive, dict):
        raise ValueError("archive root must be an object")
    _validate_archive(archive)
    return archive
