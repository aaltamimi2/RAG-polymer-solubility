"""Scoped, versioned sidecar artifact tools for multi-agent handoffs."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..handoffs import get_scope_artifact_dir, set_handoff_root

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def set_scratch_dir(path: Path) -> None:
    """Backward-compatible wrapper for configuring the handoff artifact root."""
    set_handoff_root(path)


def _json_error(message: str, **details) -> str:
    payload = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    return json.dumps(payload, indent=2)


def _matching_versions(key: str) -> list[Path]:
    artifact_dir = get_scope_artifact_dir()
    return sorted(artifact_dir.glob(f"*__{key}.json"))


def write_sidecar(key: str, data: str) -> str:
    """Write one versioned JSON artifact for the current handoff scope."""
    if not _KEY_RE.match(key):
        return _json_error(
            f"key '{key}' is invalid",
            key_rules="letters, digits, hyphens, underscores; max 64 chars",
        )
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        return _json_error("data is not valid JSON", parse_error=str(exc))

    artifact_dir = get_scope_artifact_dir()
    versions = _matching_versions(key)
    version = len(versions) + 1
    target = artifact_dir / f"{version:04d}__{key}.json"
    target.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "ok": True,
        "key": key,
        "version": version,
        "path": str(target),
        "bytes": target.stat().st_size,
    }
    return json.dumps(payload, indent=2)


def read_sidecar(key: str) -> str:
    """Read versioned artifacts for the current handoff scope."""
    artifact_dir = get_scope_artifact_dir()
    if key == "list":
        grouped: dict[str, list[dict[str, object]]] = {}
        for path in sorted(artifact_dir.glob("*__*.json")):
            prefix, logical_key = path.stem.split("__", 1)
            grouped.setdefault(logical_key, []).append(
                {
                    "version": int(prefix),
                    "path": str(path),
                }
            )
        return json.dumps({"ok": True, "keys": grouped}, indent=2)

    if not _KEY_RE.match(key):
        return _json_error(
            f"key '{key}' is invalid",
            key_rules="letters, digits, hyphens, underscores; max 64 chars",
        )

    versions = _matching_versions(key)
    if not versions:
        available = sorted(
            {
                path.stem.split("__", 1)[1]
                for path in artifact_dir.glob("*__*.json")
                if "__" in path.stem
            }
        )
        return _json_error(
            f"No sidecar artifact found for '{key}'",
            available_keys=available,
        )

    latest = versions[-1]
    version = int(latest.stem.split("__", 1)[0])
    content = json.loads(latest.read_text(encoding="utf-8"))
    payload = {
        "ok": True,
        "key": key,
        "version": version,
        "path": str(latest),
        "data": content,
    }
    return json.dumps(payload, indent=2)
