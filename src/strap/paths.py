"""Runtime-safe path helpers for data and model assets.

These helpers avoid relying on package-relative paths like ``site-packages/../data``,
which break in deployed containers where the application code is installed as a wheel
but the runtime assets live under ``/app``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _dedupe(candidates: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _candidate_dirs(explicit_env: str | None, dirname: str) -> list[Path]:
    candidates: list[Path] = []
    if explicit_env:
        candidates.append(Path(explicit_env))

    cwd = Path.cwd().resolve()
    candidates.append(cwd / dirname)
    candidates.append(Path("/app") / dirname)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / dirname)

    return _dedupe(candidates)


def _resolve_existing_dir(explicit_env: str | None, dirname: str) -> Path:
    candidates = _candidate_dirs(explicit_env, dirname)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if explicit_env:
        return Path(explicit_env).resolve()
    return (Path("/app") / dirname).resolve()


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    return _resolve_existing_dir(os.environ.get("DATA_DIR"), "data")


@lru_cache(maxsize=1)
def get_models_dir() -> Path:
    return _resolve_existing_dir(os.environ.get("ML_MODEL_DIR"), "models")


def get_data_path(*parts: str) -> Path:
    return get_data_dir().joinpath(*parts)


def get_models_path(*parts: str) -> Path:
    return get_models_dir().joinpath(*parts)
