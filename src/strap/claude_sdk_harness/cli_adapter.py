"""Small CLI-facing helpers for Claude SDK harness state."""

from __future__ import annotations

from pathlib import Path

from .sessions import bridge_path, load_bridge


def format_harness_status(
    *,
    harness: str,
    thread_id: str,
    model_alias: str,
    model_name: str,
    cwd: str | Path,
) -> str:
    bridge = load_bridge(thread_id)
    lines = [
        f"Harness: {harness}",
        f"DISSOLVE thread: {thread_id}",
        f"SDK model: {model_alias} ({model_name})" if harness == "claude_sdk" else f"Model: {model_alias} ({model_name})",
    ]
    if harness == "claude_sdk":
        last_cost = bridge.get("last_cost_usd")
        if isinstance(last_cost, (int, float)):
            cost_line = f"Last cost: ${last_cost:.6f}"
        else:
            cost_line = "Last cost: <none>"
        lines.extend(
            [
                f"Claude session: {bridge.get('claude_session_id', '<none>')}",
                f"Bridge file: {bridge_path(thread_id)}",
                f"CWD: {Path(cwd).resolve()}",
                "Tool search: disabled (intent-scoped allowlists)",
                cost_line,
                "Switching: restart with dissolve --harness langchain|claude_sdk",
            ]
        )
    return "\n".join(lines)
