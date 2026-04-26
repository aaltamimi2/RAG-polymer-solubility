"""Shared path handling for typed planner runtime artifacts."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from strap.tools._helpers import normalize_wsl_path


RUNTIME_ARTIFACT_SCHEMA_VERSION = "1.0"
DEFAULT_TYPED_RUNTIME_ROOT = Path("architecture") / "test_results" / "typed_planner_runtime"


def normalize_runtime_path(path: str | Path) -> str:
    """Normalize user-facing paths, including Windows/WSL UNC paths, for local use."""
    return normalize_wsl_path(path)


def resolve_runtime_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve and create a runtime artifact output directory."""
    root = Path(normalize_runtime_path(output_dir)) if output_dir else DEFAULT_TYPED_RUNTIME_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def slugify_run_component(value: str, *, fallback: str = "run") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:80] or fallback


def unique_child_path(directory: Path, filename: str) -> Path:
    """Return a non-conflicting path under directory for filename."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        next_candidate = directory / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically enough for local diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)
