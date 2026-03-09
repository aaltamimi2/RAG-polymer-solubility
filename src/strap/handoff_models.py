"""Shared handoff data models and artifact helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_ARTIFACT_HINT_KEYS = {
    "artifacts",
    "charts",
    "filepath",
    "files",
    "path",
    "pdf_paths",
    "plot_path",
    "plot_paths",
    "saved_files",
}
_ARTIFACT_EXTENSIONS = {
    ".csv",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".pdf",
    ".png",
    ".svg",
    ".txt",
}


def _slugify(value: str, default: str) -> str:
    value = _SAFE_NAME_RE.sub("-", value or "").strip("-")
    return value[:80] or default


def _looks_like_artifact(value: str) -> bool:
    cleaned = value.strip().strip("`")
    if not cleaned:
        return False
    if cleaned.startswith(("/", "./")):
        return True
    return Path(cleaned).suffix.lower() in _ARTIFACT_EXTENSIONS and ("/" in cleaned or "\\" in cleaned)


def extract_artifacts_from_payload(payload: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    seen: set[str] = set()

    def visit(value: Any, *, hinted: bool = False) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, hinted=hinted or key in _ARTIFACT_HINT_KEYS)
            return
        if isinstance(value, list):
            for item in value:
                visit(item, hinted=hinted)
            return
        if hinted and isinstance(value, str) and _looks_like_artifact(value):
            cleaned = value.strip().strip("`")
            if cleaned not in seen:
                seen.add(cleaned)
                artifacts.append(cleaned)

    visit(payload)
    return artifacts


@dataclass(frozen=True)
class HandoffScope:
    """Execution scope for one orchestrator run."""

    invocation_id: str
    run_id: str
    thread_id: str

    @property
    def scope_id(self) -> str:
        return f"{self.thread_id}::{self.run_id}"


@dataclass
class HandoffRecord:
    """Stored subagent result or derived inter-agent handoff."""

    handoff_id: str
    scope: HandoffScope
    producer: str
    consumer: str
    contract: str
    status: str
    payload: dict[str, Any]
    created_at: str
    source_tool_call_id: str | None = None
    parent_handoff_id: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    task_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scope"] = asdict(self.scope)
        return data


@dataclass
class _ScopeState:
    scope: HandoffScope
    artifact_root: Path
    user_query: str | None = None
    handoffs: list[HandoffRecord] = field(default_factory=list)
