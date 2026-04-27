"""Message extraction helpers for Claude Agent SDK streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClaudeSdkTurnResult:
    content: str
    origin: str
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    result_subtype: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    num_turns: int | None = None
    mcp_tool_calls: list[str] = field(default_factory=list)
    legacy_tool_calls: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result_subtype not in {"error", "error_max_turns", "error_max_budget_usd"}


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def message_kind(message: Any) -> str:
    return message.__class__.__name__


def extract_session_id(message: Any) -> str | None:
    if message_kind(message) == "SystemMessage" and getattr(message, "subtype", None) == "init":
        data = getattr(message, "data", None)
        if isinstance(data, dict):
            value = data.get("session_id")
            return str(value) if value else None
    value = getattr(message, "session_id", None)
    return str(value) if value else None


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    if message_kind(message) not in {"AssistantMessage"} and not hasattr(message, "content"):
        return []
    calls: list[dict[str, Any]] = []
    for block in getattr(message, "content", []) or []:
        name = getattr(block, "name", None)
        if not name and isinstance(block, dict):
            name = block.get("name")
        if not name:
            continue
        block_input = getattr(block, "input", None)
        if block_input is None and isinstance(block, dict):
            block_input = block.get("input")
        calls.append({"name": str(name), "input": block_input or {}})
    return calls


def is_result_message(message: Any) -> bool:
    if message_kind(message) == "ResultMessage":
        return True
    return hasattr(message, "result") and (
        hasattr(message, "subtype")
        or hasattr(message, "session_id")
        or hasattr(message, "usage")
        or hasattr(message, "total_cost_usd")
    )


def result_text(message: Any) -> str:
    return str(getattr(message, "result", "") or "")
