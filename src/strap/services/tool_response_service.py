"""Shared JSON response helpers for agent-facing tools."""

from __future__ import annotations

import json
from typing import Any, Mapping


def _infer_success(payload: dict[str, Any]) -> bool:
    if "success" in payload:
        return bool(payload["success"])
    if payload.get("found") is False:
        return False
    if payload.get("error"):
        return False
    return True


def json_tool_response(
    display: str,
    data: Mapping[str, Any] | None = None,
    *,
    tool_name: str | None = None,
    success: bool | None = None,
) -> str:
    """Return the standard tool envelope used across agent-facing tools."""
    payload = dict(data or {})
    if tool_name and "tool_name" not in payload:
        payload["tool_name"] = tool_name
    if success is None:
        payload["success"] = _infer_success(payload)
    else:
        payload["success"] = success
    return json.dumps({"display": display, "data": payload}, indent=2, ensure_ascii=False)


def json_tool_success(display: str, *, tool_name: str, **data: Any) -> str:
    """Return a successful tool envelope with a stable shape."""
    return json_tool_response(display, data, tool_name=tool_name, success=True)


def json_tool_error(
    message: str,
    *,
    tool_name: str,
    error_code: str = "invalid_input",
    **data: Any,
) -> str:
    """Return a predictable structured error envelope."""
    payload = {"error": message, "error_code": error_code}
    payload.update(data)
    return json_tool_response(message, payload, tool_name=tool_name, success=False)
