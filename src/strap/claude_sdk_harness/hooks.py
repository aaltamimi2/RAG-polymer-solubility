"""Guard and diagnostic helpers for Claude SDK hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tool_catalog import ToolNameMap


DOCUMENTED_HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "UserPromptSubmit",
    "Stop",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "Notification",
    "PermissionRequest",
)


@dataclass
class HookDiagnostics:
    blocked_tools: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    notifications: list[dict[str, Any]] = field(default_factory=list)


def guard_tool_name(
    tool_name: str,
    *,
    allowed_tools: list[str],
    tool_map: ToolNameMap | None = None,
    diagnostics: HookDiagnostics | None = None,
    active_intent: str | None = None,
    active_plan_step: str | None = None,
) -> bool:
    """Return whether an SDK tool name is allowed for this turn."""
    tool_map = tool_map or ToolNameMap()
    allowed = set(allowed_tools)
    ok = tool_name in allowed or any(item.endswith("__*") and tool_name.startswith(item[:-1]) for item in allowed)
    if not ok and diagnostics is not None:
        try:
            legacy = tool_map.legacy_name(tool_name)
        except KeyError:
            legacy = tool_name
        diagnostics.blocked_tools.append(
            {
                "tool_name": tool_name,
                "legacy_name": legacy,
                "active_intent": active_intent,
                "active_plan_step": active_plan_step,
            }
        )
    return ok


def build_hooks(*, diagnostics: HookDiagnostics | None = None) -> dict[str, list[Any]] | None:
    """Return a minimal hook mapping when the SDK is available.

    Permission enforcement is primarily handled by ``allowed_tools`` plus
    ``permission_mode='dontAsk'``. PreToolUse mirrors the allowlist so blocked
    calls are observable before the SDK permission layer denies them.
    """
    return build_guarded_hooks(diagnostics=diagnostics)


def build_guarded_hooks(
    *,
    diagnostics: HookDiagnostics | None = None,
    allowed_tools: list[str] | None = None,
    tool_map: ToolNameMap | None = None,
    active_intent: str | None = None,
    active_plan_step: str | None = None,
) -> dict[str, list[Any]] | None:
    """Build hooks with explicit PreToolUse guard diagnostics."""
    try:
        from claude_agent_sdk import HookMatcher
    except Exception:
        return None

    diagnostics = diagnostics or HookDiagnostics()
    allowed_tools = list(allowed_tools or [])
    tool_map = tool_map or ToolNameMap()

    async def _pre_tool_use(input_data, tool_use_id=None, context=None):  # noqa: ANN001
        tool_name = str(input_data.get("tool_name") or "")
        if guard_tool_name(
            tool_name,
            allowed_tools=allowed_tools,
            tool_map=tool_map,
            diagnostics=diagnostics,
            active_intent=active_intent,
            active_plan_step=active_plan_step,
        ):
            return {}
        reason = f"DISSOLVE blocked unapproved tool for intent {active_intent or 'unknown'}: {tool_name}"
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
            "reason": reason,
        }

    async def _post_tool_use(input_data, tool_use_id=None, context=None):  # noqa: ANN001
        diagnostics.tool_outputs.append(
            {"tool_use_id": tool_use_id, "tool_name": input_data.get("tool_name"), "input": input_data}
        )
        return {}

    async def _notification(input_data, tool_use_id=None, context=None):  # noqa: ANN001
        diagnostics.notifications.append({"tool_use_id": tool_use_id, "input": input_data})
        return {}

    return {
        "PreToolUse": [HookMatcher(matcher=None, hooks=[_pre_tool_use])],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_use])],
        "Notification": [HookMatcher(matcher=None, hooks=[_notification])],
    }
