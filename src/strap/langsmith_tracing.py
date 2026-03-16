"""LangSmith tracing helpers for frontend-friendly workflow metadata."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_helpers import trace as langsmith_trace
except Exception:  # pragma: no cover - optional dependency/import path
    LangSmithClient = None
    langsmith_trace = None

_LANGSMITH_CLIENT: Any | None = None
_SUBAGENT_TRACE_CAPTURE: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "_SUBAGENT_TRACE_CAPTURE",
    default=None,
)


def tracing_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY")) and langsmith_trace is not None and LangSmithClient is not None


def _workspace_id() -> str | None:
    value = (
        os.getenv("LANGSMITH_WORKSPACE_ID")
        or os.getenv("LANGCHAIN_WORKSPACE_ID")
    )
    if value is None:
        return None
    value = value.strip()
    return value or None


def _friendly_langsmith_error(exc: Exception) -> str:
    text = str(exc)
    if "X-Tenant-ID" in text or "workspace specification" in text:
        return (
            "LangSmith workspace ID is required for this org-scoped API key. "
            "Set LANGSMITH_WORKSPACE_ID on the deployment."
        )
    return text


def get_langsmith_client() -> Any | None:
    global _LANGSMITH_CLIENT
    if not tracing_enabled():
        return None
    if _LANGSMITH_CLIENT is None:
        try:
            _LANGSMITH_CLIENT = LangSmithClient(workspace_id=_workspace_id())
        except Exception:
            logger.exception("Failed to initialize LangSmith client")
            return None
    return _LANGSMITH_CLIENT


def resolve_run_links(run_tree: Any) -> dict[str, str | None]:
    client = get_langsmith_client()
    if client is None or run_tree is None:
        return {"run_id": None, "trace_url": None, "shared_trace_url": None}

    project_name = os.getenv("LANGSMITH_PROJECT", "strap-agent")
    run_id = str(getattr(run_tree, "id", "") or "") or None
    trace_url = None
    shared_trace_url = None

    try:
        trace_url = client.get_run_url(run=run_tree, project_name=project_name)
    except Exception:
        logger.exception("Failed to resolve LangSmith run URL for run %s", run_id)

    share_enabled = os.getenv("LANGSMITH_SHARE_RUNS", "").strip().lower() in {"1", "true", "yes"}
    if share_enabled and run_id:
        try:
            shared_trace_url = client.share_run(run_id)
        except Exception:
            logger.exception("Failed to create shared LangSmith run URL for run %s", run_id)

    return {
        "run_id": run_id,
        "trace_url": trace_url,
        "shared_trace_url": shared_trace_url,
    }


def start_subagent_trace_capture() -> Token:
    return _SUBAGENT_TRACE_CAPTURE.set({})


def stop_subagent_trace_capture(token: Token) -> None:
    _SUBAGENT_TRACE_CAPTURE.reset(token)


def get_captured_subagent_traces() -> dict[str, dict[str, Any]]:
    return dict(_SUBAGENT_TRACE_CAPTURE.get() or {})


def record_subagent_trace(
    *,
    tool_call_id: str | None,
    subagent_type: str,
    description: str,
    run_tree: Any,
) -> dict[str, Any] | None:
    if not tool_call_id:
        return None

    links = resolve_run_links(run_tree)
    record = {
        "tool_call_id": tool_call_id,
        "subagent": subagent_type,
        "description": description,
        **links,
    }

    capture = _SUBAGENT_TRACE_CAPTURE.get()
    if capture is not None:
        capture[tool_call_id] = record
    return record


def summarize_subagent_tool_runs(run_id: str | None) -> dict[str, Any]:
    client = get_langsmith_client()
    if client is None or not run_id:
        return {"tools": [], "tool_count": 0, "error": None}

    try:
        run = client.read_run(run_id, load_child_runs=True)
    except Exception as exc:
        logger.exception("Failed to read LangSmith subagent run %s", run_id)
        return {
            "tools": [],
            "tool_count": 0,
            "error": _friendly_langsmith_error(exc),
        }

    tools: list[dict[str, Any]] = []

    def _duration_ms(node: Any) -> int | None:
        start_time = getattr(node, "start_time", None)
        end_time = getattr(node, "end_time", None)
        if start_time is None or end_time is None:
            return None
        try:
            return max(0, int((end_time - start_time).total_seconds() * 1000))
        except Exception:
            return None

    def visit(node: Any, depth: int = 0) -> None:
        children = list(getattr(node, "child_runs", None) or [])
        children.sort(key=lambda item: getattr(item, "dotted_order", "") or "")
        for child in children:
            if getattr(child, "run_type", None) == "tool":
                links = resolve_run_links(child)
                status = str(getattr(child, "status", "") or "")
                if not status:
                    status = "error" if getattr(child, "error", None) else "completed"
                tools.append(
                    {
                        "order": len(tools) + 1,
                        "name": str(getattr(child, "name", "tool")),
                        "run_id": links["run_id"],
                        "trace_url": links["trace_url"],
                        "shared_trace_url": links["shared_trace_url"],
                        "status": status,
                        "error": getattr(child, "error", None),
                        "depth": depth,
                        "started_at": getattr(child, "start_time", None).isoformat() if getattr(child, "start_time", None) else None,
                        "ended_at": getattr(child, "end_time", None).isoformat() if getattr(child, "end_time", None) else None,
                        "duration_ms": _duration_ms(child),
                    }
                )
            visit(child, depth + 1)

    visit(run, 0)
    total_duration_ms = sum(tool["duration_ms"] or 0 for tool in tools)
    return {
        "tools": tools,
        "tool_count": len(tools),
        "total_duration_ms": total_duration_ms if tools else None,
        "error": None,
    }
