"""Claude Agent SDK turn runner."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterable, Callable

from strap.direct_fast_path import try_direct_tool_fast_path
from strap.planning.config import get_planner_config
from strap.planning.typed_runtime_integration import (
    format_typed_runtime_failure,
    format_typed_runtime_success,
    maybe_run_typed_runtime,
    summarize_typed_runtime_progress,
)
from strap.session_state import build_session_context_block

from .hooks import HookDiagnostics
from .messages import (
    ClaudeSdkTurnResult,
    extract_session_id,
    extract_tool_calls,
    is_result_message,
    result_text,
)
from .models import resolve_claude_model_selection
from .mcp_server import ClaudeSdkUnavailableError
from .options import DISSOLVE_SCIENCE_PERMISSION_MODE, build_options
from .sessions import bridge_is_resumable, build_bridge_update, load_bridge
from .tool_catalog import ToolNameMap, infer_intent, normalize_tool_call_names

QueryFunc = Callable[..., AsyncIterable[Any]]


@dataclass
class ClaudeSdkRunner:
    thread_id: str
    model_alias: str = "claude-sonnet"
    sdk_model: str = ""
    cwd: str | Path = "."
    harness_profile: str = "science"
    permission_mode: str = DISSOLVE_SCIENCE_PERMISSION_MODE
    max_turns: int = 8
    max_budget_usd: float = 0.25
    query_func: QueryFunc | None = None
    tool_map: ToolNameMap | None = None

    def __post_init__(self) -> None:
        if not self.sdk_model:
            selection = resolve_claude_model_selection(self.model_alias)
            self.model_alias = selection.alias
            self.sdk_model = selection.sdk_model
        self.cwd = Path(self.cwd).resolve()
        self.tool_map = self.tool_map or ToolNameMap()
        self.last_cost_usd: float | None = None
        self.last_session_id: str | None = None

    @classmethod
    def from_model_request(
        cls,
        *,
        thread_id: str,
        raw_model: str | None,
        cwd: str | Path,
        query_func: QueryFunc | None = None,
    ) -> "ClaudeSdkRunner":
        selection = resolve_claude_model_selection(raw_model)
        return cls(
            thread_id=thread_id,
            model_alias=selection.alias,
            sdk_model=selection.sdk_model,
            cwd=cwd,
            query_func=query_func,
        )

    def update_model(self, raw_model: str | None) -> str | None:
        selection = resolve_claude_model_selection(raw_model)
        self.model_alias = selection.alias
        self.sdk_model = selection.sdk_model
        return selection.notice

    def _contextual_prompt(self, prompt: str, session_context: dict[str, Any] | None) -> str:
        if "Session context" in prompt or session_context is None:
            return prompt
        block = build_session_context_block(session_context)
        if not block:
            return prompt
        return f"{block}\n\nUser request:\n{prompt}"

    def _resume_session_id(self) -> str | None:
        bridge = load_bridge(self.thread_id)
        if bridge_is_resumable(bridge, cwd=self.cwd):
            return str(bridge.get("claude_session_id") or "") or None
        return None

    def _bridge_update(
        self,
        *,
        allowed_tools: list[str],
        session_id: str | None = None,
        result_subtype: str | None = None,
        usage: dict[str, Any] | None = None,
        cost: float | None = None,
        last_error_code: str | None = None,
        previous_model_alias: str | None = None,
        clear_claude_session_id: bool = False,
    ) -> None:
        bridge = build_bridge_update(
            thread_id=self.thread_id,
            cwd=self.cwd,
            harness_profile=self.harness_profile,
            model_alias=self.model_alias,
            sdk_model=self.sdk_model,
            permission_mode=self.permission_mode,
            allowed_tools=allowed_tools,
            claude_session_id=session_id,
            last_result_subtype=result_subtype,
            last_cost_usd=cost,
            last_usage=usage,
            last_error_code=last_error_code,
            previous_model_alias=previous_model_alias,
            clear_claude_session_id=clear_claude_session_id,
        )
        self.last_session_id = bridge.get("claude_session_id")
        self.last_cost_usd = bridge.get("last_cost_usd")

    def _typed_runtime_tool_calls(self, result) -> list[str]:  # noqa: ANN001
        if result.ledger is None:
            return []
        tools: list[str] = []
        seen: set[str] = set()
        for record in result.ledger.step_records:
            name = str(record.callable_name or "").strip()
            if name and name not in seen:
                seen.add(name)
                tools.append(name)
        return tools

    def _direct_fast_path(self, prompt: str, allowed_tools: list[str]) -> ClaudeSdkTurnResult | None:
        result = try_direct_tool_fast_path(prompt)
        if result is None:
            return None
        self._bridge_update(allowed_tools=allowed_tools, result_subtype="direct_fast_path", cost=0.0, usage={})
        return ClaudeSdkTurnResult(
            content=result.display,
            origin="direct_tool_fast_path",
            additional_kwargs={
                "strap_origin": "direct_tool_fast_path",
                "strap_tool_name": result.tool_name,
                "strap_fast_path": True,
                "strap_route_decision": result.route_decision,
                "strap_artifacts": result.artifacts,
                "strap_run_ledger": result.run_ledger,
                "claude_model_calls": 0,
                "claude_mcp_tool_calls": [],
            },
            result_subtype="direct_fast_path",
            total_cost_usd=0.0,
            usage={},
            legacy_tool_calls=[result.tool_name],
        )

    def _typed_runtime(self, prompt: str, allowed_tools: list[str]) -> ClaudeSdkTurnResult | None:
        config = get_planner_config()
        if config.mode in {"off", "shadow"}:
            return None
        result = maybe_run_typed_runtime(prompt, config=config)
        if result is None:
            return None
        status = result.status
        legacy_tool_calls = self._typed_runtime_tool_calls(result)
        content = (
            format_typed_runtime_success(result, config=config)
            if status == "executed"
            else format_typed_runtime_failure(result)
        )
        self._bridge_update(allowed_tools=allowed_tools, result_subtype=status, cost=0.0, usage={})
        return ClaudeSdkTurnResult(
            content=content,
            origin="typed_runtime",
            additional_kwargs={
                "strap_origin": "typed_runtime",
                "strap_typed_runtime_status": status,
                "strap_typed_runtime_selected": result.selected,
                "strap_plan_id": result.plan.plan_id if result.plan else None,
                "strap_workflow_id": result.plan.workflow_id if result.plan else None,
                "strap_runtime_progress": summarize_typed_runtime_progress(result).model_dump(mode="json"),
                "strap_manifest": result.manifest.model_dump(mode="json") if result.manifest else None,
                "strap_compile_result": result.compile_result.model_dump(mode="json"),
                "strap_run_plan": result.plan.model_dump(mode="json") if result.plan else None,
                "strap_run_ledger": result.ledger.model_dump(mode="json") if result.ledger else None,
                "claude_model_calls": 0,
                "claude_mcp_tool_calls": [],
                "claude_legacy_tool_calls": legacy_tool_calls,
            },
            result_subtype=status,
            total_cost_usd=0.0,
            usage={},
            legacy_tool_calls=legacy_tool_calls,
        )

    async def arun_turn(self, prompt: str, *, session_context: dict[str, Any] | None = None) -> ClaudeSdkTurnResult:
        contextual_prompt = self._contextual_prompt(prompt, session_context)
        intent = infer_intent(prompt)
        allowed_tools = self.tool_map.allowed_for_intent(intent) if self.tool_map else []

        if typed := self._typed_runtime(contextual_prompt, allowed_tools):
            return typed
        if direct := self._direct_fast_path(contextual_prompt, allowed_tools):
            return direct

        if not os.getenv("ANTHROPIC_API_KEY"):
            message = "Claude SDK harness requires ANTHROPIC_API_KEY for non-fast-path turns."
            self._bridge_update(
                allowed_tools=allowed_tools,
                session_id=None,
                result_subtype="missing_key",
                cost=0.0,
                usage={},
                last_error_code="missing_anthropic_api_key",
                clear_claude_session_id=True,
            )
            return ClaudeSdkTurnResult(content=message, origin="claude_sdk", error=message, result_subtype="missing_key")

        try:
            if self.query_func is None:
                try:
                    from claude_agent_sdk import query as query_func
                except Exception as exc:
                    raise ClaudeSdkUnavailableError(
                        "claude-agent-sdk is not installed. Install with `pip install -e .[claude]` "
                        "or use the default LangChain harness."
                    ) from exc
            else:
                query_func = self.query_func
            diagnostics = HookDiagnostics()
            options = build_options(
                sdk_model=self.sdk_model,
                allowed_tools=allowed_tools,
                resume=self._resume_session_id(),
                cwd=self.cwd,
                max_turns=self.max_turns,
                max_budget_usd=self.max_budget_usd,
                permission_mode=self.permission_mode,
                diagnostics=diagnostics,
                tool_map=self.tool_map,
                active_intent=intent,
            )
        except ClaudeSdkUnavailableError as exc:
            self._bridge_update(
                allowed_tools=allowed_tools,
                session_id=None,
                result_subtype="sdk_unavailable",
                cost=0.0,
                usage={},
                last_error_code="claude_agent_sdk_unavailable",
                clear_claude_session_id=True,
            )
            return ClaudeSdkTurnResult(content=str(exc), origin="claude_sdk", error=str(exc), result_subtype="sdk_unavailable")

        t0 = perf_counter()
        session_id = None
        result_message = None
        mcp_tool_calls: list[str] = []
        try:
            async for message in query_func(prompt=contextual_prompt, options=options):
                session_id = extract_session_id(message) or session_id
                for call in extract_tool_calls(message):
                    mcp_tool_calls.append(call["name"])
                if is_result_message(message):
                    result_message = message
        except KeyboardInterrupt:
            self._bridge_update(allowed_tools=allowed_tools, session_id=session_id, result_subtype="cancelled")
            raise
        except Exception as exc:
            error = f"Claude SDK turn failed: {exc}"
            self._bridge_update(allowed_tools=allowed_tools, session_id=session_id, result_subtype="interrupted")
            return ClaudeSdkTurnResult(content=error, origin="claude_sdk", error=error, result_subtype="interrupted")

        if result_message is None:
            error = "Claude SDK did not return a final ResultMessage."
            self._bridge_update(allowed_tools=allowed_tools, session_id=session_id, result_subtype="missing_result")
            return ClaudeSdkTurnResult(content=error, origin="claude_sdk", error=error, result_subtype="missing_result")

        usage = getattr(result_message, "usage", None) or {}
        cost = getattr(result_message, "total_cost_usd", None)
        subtype = getattr(result_message, "subtype", None)
        session_id = str(getattr(result_message, "session_id", None) or session_id or "")
        self._bridge_update(
            allowed_tools=allowed_tools,
            session_id=session_id,
            result_subtype=subtype,
            usage=usage,
            cost=cost,
        )
        legacy_tool_calls = normalize_tool_call_names(mcp_tool_calls, self.tool_map)
        return ClaudeSdkTurnResult(
            content=result_text(result_message),
            origin="claude_sdk",
            additional_kwargs={
                "strap_origin": "claude_sdk",
                "claude_model_calls": 1,
                "claude_mcp_tool_calls": mcp_tool_calls,
                "claude_legacy_tool_calls": legacy_tool_calls,
                "claude_elapsed_s": perf_counter() - t0,
            },
            result_subtype=subtype,
            session_id=session_id,
            total_cost_usd=cost,
            usage=usage,
            stop_reason=getattr(result_message, "stop_reason", None),
            num_turns=getattr(result_message, "num_turns", None),
            mcp_tool_calls=mcp_tool_calls,
            legacy_tool_calls=legacy_tool_calls,
        )

    def run_turn(self, prompt: str, *, session_context: dict[str, Any] | None = None) -> ClaudeSdkTurnResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_turn(prompt, session_context=session_context))
        raise RuntimeError("ClaudeSdkRunner.run_turn cannot be called from an active event loop; use arun_turn().")
