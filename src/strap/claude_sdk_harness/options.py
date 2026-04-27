"""Build ClaudeAgentOptions for DISSOLVE SDK turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import build_agent_definitions
from .hooks import HookDiagnostics, build_guarded_hooks
from .mcp_server import ClaudeSdkUnavailableError, build_mcp_servers
from .tool_catalog import ToolNameMap

VALID_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"}
DISSOLVE_SCIENCE_PERMISSION_MODE = "dontAsk"


def import_sdk_options() -> Any:
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except Exception as exc:
        raise ClaudeSdkUnavailableError(
            "claude-agent-sdk is not installed. Install with `pip install -e .[claude]` "
            "or run without `--harness claude_sdk`."
        ) from exc
    return ClaudeAgentOptions


def load_harness_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "dissolve_sdk.md").read_text(encoding="utf-8")


def build_options(
    *,
    sdk_model: str,
    allowed_tools: list[str],
    resume: str | None = None,
    cwd: str | Path | None = None,
    max_turns: int = 8,
    max_budget_usd: float = 0.25,
    permission_mode: str = DISSOLVE_SCIENCE_PERMISSION_MODE,
    include_agents: bool = False,
    diagnostics: HookDiagnostics | None = None,
    tool_map: ToolNameMap | None = None,
    active_intent: str | None = None,
    active_plan_step: str | None = None,
) -> Any:
    if permission_mode not in VALID_PERMISSION_MODES:
        raise ValueError(f"Unsupported Claude SDK permission mode: {permission_mode}")
    ClaudeAgentOptions = import_sdk_options()
    tool_map = tool_map or ToolNameMap()
    hooks = build_guarded_hooks(
        diagnostics=diagnostics,
        allowed_tools=allowed_tools,
        tool_map=tool_map,
        active_intent=active_intent,
        active_plan_step=active_plan_step,
    )
    kwargs: dict[str, Any] = {
        "tools": [],
        "allowed_tools": allowed_tools,
        "disallowed_tools": ["Bash", "Edit", "Write"],
        "permission_mode": permission_mode,
        "mcp_servers": build_mcp_servers(tool_map),
        "system_prompt": load_harness_prompt(),
        "model": sdk_model,
        "max_turns": max_turns,
        "max_budget_usd": max_budget_usd,
        "include_partial_messages": True,
        "setting_sources": [],
    }
    if cwd is not None:
        kwargs["cwd"] = Path(cwd)
    if resume:
        kwargs["resume"] = resume
    if hooks:
        kwargs["hooks"] = hooks
    if include_agents:
        kwargs["agents"] = build_agent_definitions(tool_map)
        if "Agent" not in kwargs["allowed_tools"]:
            kwargs["allowed_tools"] = list(kwargs["allowed_tools"]) + ["Agent"]
    return ClaudeAgentOptions(**kwargs)
