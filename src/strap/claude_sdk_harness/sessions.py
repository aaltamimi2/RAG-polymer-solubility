"""Bridge DISSOLVE sessions to Claude Agent SDK session IDs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strap.session_state import session_paths

from .tool_catalog import ToolNameMap, fingerprint_allowed_tools

BRIDGE_FILENAME = "claude_sdk_session.json"
BRIDGE_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bridge_path(thread_id: str) -> Path:
    return session_paths(thread_id)["dir"] / BRIDGE_FILENAME


def load_bridge(thread_id: str) -> dict[str, Any]:
    path = bridge_path(thread_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_bridge(thread_id: str, bridge: dict[str, Any]) -> dict[str, Any]:
    path = bridge_path(thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(bridge)
    data["schema_version"] = BRIDGE_SCHEMA_VERSION
    data["thread_id"] = thread_id
    data["updated_at"] = utc_now()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def bridge_is_resumable(bridge: dict[str, Any], *, cwd: str | Path) -> bool:
    session_id = str(bridge.get("claude_session_id") or "")
    stored_cwd = str(bridge.get("cwd") or "")
    return bool(session_id and stored_cwd and Path(stored_cwd).resolve() == Path(cwd).resolve())


def build_bridge_update(
    *,
    thread_id: str,
    cwd: str | Path,
    harness_profile: str,
    model_alias: str,
    sdk_model: str,
    permission_mode: str,
    allowed_tools: list[str],
    claude_session_id: str | None = None,
    last_result_subtype: str | None = None,
    last_cost_usd: float | None = None,
    last_usage: dict[str, Any] | None = None,
    last_error_code: str | None = None,
    previous_model_alias: str | None = None,
    clear_claude_session_id: bool = False,
) -> dict[str, Any]:
    tool_map = ToolNameMap()
    bridge = load_bridge(thread_id)
    bridge.update(
        {
            "cwd": str(Path(cwd).resolve()),
            "harness": "claude_sdk",
            "harness_profile": harness_profile,
            "model_alias": model_alias,
            "model_id": f"anthropic:{sdk_model}",
            "permission_mode": permission_mode,
            "allowed_tools_fingerprint": fingerprint_allowed_tools(allowed_tools),
            "tool_name_map_version": tool_map.version,
        }
    )
    if previous_model_alias:
        bridge["previous_model_alias"] = previous_model_alias
    if clear_claude_session_id:
        bridge["claude_session_id"] = None
    elif claude_session_id:
        bridge["claude_session_id"] = claude_session_id
    if last_result_subtype is not None:
        bridge["last_result_subtype"] = last_result_subtype
    if last_cost_usd is not None:
        bridge["last_cost_usd"] = last_cost_usd
    if last_usage is not None:
        bridge["last_usage"] = dict(last_usage)
    if last_error_code is not None:
        bridge["last_error_code"] = last_error_code
    return save_bridge(thread_id, bridge)
