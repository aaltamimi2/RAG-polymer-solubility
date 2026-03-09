"""Subagent guardrail middleware: iteration cap + token budget + tool-call
limit + synthesis injection + old tool-result truncation."""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .guardrail_checks import (
    get_separation_analysis_coverage_errors,
    get_separation_feasibility_errors,
    get_separation_selectivity_scope_errors,
    get_separation_support_scope_errors,
    get_separation_temperature_bound_errors,
    get_structured_result_errors,
)
from .guardrail_messages import (
    iteration_limit_message,
    separation_analysis_coverage_repair_message,
    separation_feasibility_repair_message,
    structured_result_repair_message,
    token_budget_message,
    tool_budget_repair_message,
    tool_budget_suffix,
)
from .guardrail_policy import (
    inject_separation_support_directive,
    inject_separation_temperature_bound_directive,
    inject_synthesis_directive,
    inject_visualization_tool_directive,
    maybe_block_late_separation_todos,
    maybe_block_duplicate_biosteam_batch,
    maybe_enforce_visualization_tool_directive,
    restrict_visualization_tools,
)
from .guardrail_utils import extract_completed_tool_names

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest

logger = logging.getLogger(__name__)


@dataclass
class _GuardState:
    iterations: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    synthesis_tool_seen: bool = False
    structured_result_repairs: int = 0
    tool_budget_repairs: int = 0
    separation_analysis_repairs: int = 0


_guard_state: contextvars.ContextVar[_GuardState] = contextvars.ContextVar(
    "_guard_state"
)
_MAX_STRUCTURED_RESULT_REPAIRS = 1
_SEPARATION_NON_ANALYSIS_TOOLS = {
    "think",
    "write_todos",
    "list_available_polymers",
    "list_available_solvents",
    "get_supported_polymers_and_solvents",
}


class SubagentGuardMiddleware(AgentMiddleware):
    """Middleware that enforces iteration, token, and tool-call limits on
    subagents and injects synthesis directives after key tools complete.

    Prevents runaway subagent loops by capping:
    - The number of model calls (iterations)
    - The cumulative prompt token usage
    - The total number of tool calls made

    Additionally:
    - After a "synthesis tool" (e.g. plan_sequential_separation) returns,
      injects a directive into the system prompt telling the LLM to
      synthesize immediately.
    - Truncates old ToolMessage content to limit quadratic context growth.

    When a limit is hit the middleware short-circuits with an AIMessage
    containing no tool calls, which causes the LangGraph agent loop to
    terminate gracefully.
    """

    def __init__(
        self,
        max_iterations: int = 25,
        token_budget: int = 200_000,
        max_tool_calls: int = 10,
        synthesis_tools: set[str] | None = None,
        truncate_tool_results_after: int | None = None,
        free_tools: set[str] | None = None,
        keep_recent: int = 4,
        agent_name: str | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._token_budget = token_budget
        self._max_tool_calls = max_tool_calls
        self._synthesis_tools: set[str] = synthesis_tools or set()
        self._truncate_tool_results_after = truncate_tool_results_after
        self._free_tools: set[str] = free_tools or set()
        self._keep_recent = keep_recent
        self._agent_name = agent_name

    # -- per-invocation state (ContextVar-isolated) ---------------------------

    @property
    def _state(self) -> _GuardState:
        try:
            return _guard_state.get()
        except LookupError:
            state = _GuardState()
            _guard_state.set(state)
            return state

    # -- lifecycle: reset counters at task() invocation start ----------------

    def before_agent(self, state, runtime):
        _guard_state.set(_GuardState())

    async def abefore_agent(self, state, runtime):
        _guard_state.set(_GuardState())

    # -- model-call wrapper: enforce limits ----------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        self._state.iterations += 1
        if self._state.iterations > self._max_iterations:
            logger.warning(
                "SubagentGuard: iteration limit (%d) reached",
                self._max_iterations,
            )
            return AIMessage(content=iteration_limit_message())

        # Detect synthesis tool results in the conversation
        self._detect_synthesis_tools(request.messages)

        # Inject synthesis directive if a key tool has already returned
        request = self._inject_synthesis_directive(request)
        request = self._inject_separation_temperature_bound_directive(request)
        request = self._inject_separation_support_directive(request)
        request = self._inject_visualization_tool_directive(request)
        request = self._restrict_visualization_tools(request)

        # Truncate old tool results to limit context growth
        request = self._truncate_old_tool_results(request)

        response = handler(request)

        # Track tokens
        self._track_tokens(response)
        total_tokens = self._state.total_prompt_tokens + self._state.total_output_tokens
        if total_tokens >= self._token_budget:
            logger.warning(
                "SubagentGuard: token budget (%d) exceeded — "
                "input=%d output=%d total=%d",
                self._token_budget,
                self._state.total_prompt_tokens,
                self._state.total_output_tokens,
                total_tokens,
            )
            return AIMessage(content=token_budget_message())

        # Track and enforce tool-call limit
        return self._enforce_tool_call_limit(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        self._state.iterations += 1
        if self._state.iterations > self._max_iterations:
            logger.warning(
                "SubagentGuard: iteration limit (%d) reached",
                self._max_iterations,
            )
            return AIMessage(content=iteration_limit_message())

        # Detect synthesis tool results in the conversation
        self._detect_synthesis_tools(request.messages)

        # Inject synthesis directive if a key tool has already returned
        request = self._inject_synthesis_directive(request)
        request = self._inject_separation_temperature_bound_directive(request)
        request = self._inject_separation_support_directive(request)
        request = self._inject_visualization_tool_directive(request)
        request = self._restrict_visualization_tools(request)

        # Truncate old tool results to limit context growth
        request = self._truncate_old_tool_results(request)

        response = await handler(request)

        # Track tokens
        self._track_tokens(response)
        total_tokens = self._state.total_prompt_tokens + self._state.total_output_tokens
        if total_tokens >= self._token_budget:
            logger.warning(
                "SubagentGuard: token budget (%d) exceeded — "
                "input=%d output=%d total=%d",
                self._token_budget,
                self._state.total_prompt_tokens,
                self._state.total_output_tokens,
                total_tokens,
            )
            return AIMessage(content=token_budget_message())

        # Track and enforce tool-call limit
        return self._enforce_tool_call_limit(response)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        """Force one repair turn when a synthesis-capable subagent omits its contract."""
        messages = state.get("messages", [])
        last_ai = self._get_last_ai_message(messages)
        if last_ai is None:
            return None
        if self._should_repair_tool_budget(last_ai):
            self._state.tool_budget_repairs += 1
            return {
                "messages": [
                    HumanMessage(
                        content=tool_budget_repair_message(self._agent_name)
                    )
                ],
                "jump_to": "model",
            }
        separation_analysis_errors = get_separation_analysis_coverage_errors(
            messages,
            last_ai,
            self._agent_name,
        )
        if self._should_repair_separation_analysis_coverage(messages, separation_analysis_errors):
            self._state.separation_analysis_repairs += 1
            detail = "; ".join(separation_analysis_errors[:2])
            return {
                "messages": [
                    HumanMessage(
                        content=separation_analysis_coverage_repair_message(detail)
                    )
                ],
                "jump_to": "model",
            }
        structured_errors = get_structured_result_errors(last_ai, self._agent_name)
        if self._should_repair_structured_result(messages, structured_errors):
            self._state.structured_result_repairs += 1
            detail = ""
            if structured_errors:
                detail = f" Validation errors: {'; '.join(structured_errors[:3])}."
            return {
                "messages": [
                    HumanMessage(
                        content=structured_result_repair_message(detail)
                    )
                ],
                "jump_to": "model",
            }

        separation_errors = get_separation_feasibility_errors(last_ai, self._agent_name)
        separation_errors.extend(
            get_separation_temperature_bound_errors(messages, last_ai, self._agent_name)
        )
        separation_errors.extend(
            get_separation_support_scope_errors(messages, last_ai, self._agent_name)
        )
        separation_errors.extend(
            get_separation_selectivity_scope_errors(messages, last_ai, self._agent_name)
        )
        if self._should_repair_separation_feasibility(messages, separation_errors):
            self._state.structured_result_repairs += 1
            detail = "; ".join(separation_errors[:3])
            return {
                "messages": [
                    HumanMessage(
                        content=separation_feasibility_repair_message(detail)
                    )
                ],
                "jump_to": "model",
            }

        return None

    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)

    def wrap_tool_call(self, request, handler):
        blocked = self._maybe_enforce_visualization_tool_directive(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_late_separation_todos(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_duplicate_biosteam_batch(request)
        if blocked is not None:
            return blocked
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        blocked = self._maybe_enforce_visualization_tool_directive(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_late_separation_todos(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_duplicate_biosteam_batch(request)
        if blocked is not None:
            return blocked
        return await handler(request)

    # -- helpers -------------------------------------------------------------

    def _track_tokens(self, response) -> None:
        """Accumulate prompt + output tokens from the model response."""
        if hasattr(response, "result"):
            if not response.result:
                return
            ai_msg = response.result[0]
        else:
            ai_msg = response

        usage = getattr(ai_msg, "usage_metadata", None)
        if usage:
            self._state.total_prompt_tokens += usage.get("input_tokens", 0)
            self._state.total_output_tokens += usage.get("output_tokens", 0)

    def _enforce_tool_call_limit(
        self, response: ModelResponse
    ) -> ModelResponse | AIMessage:
        """Count tool calls on the response and short-circuit if over budget.

        Tools listed in ``free_tools`` (e.g. ``think``) are excluded from the
        count so reflection doesn't eat into the analysis budget.

        When the limit is hit, the LLM's text content (if any) is preserved
        but tool calls are stripped — this ends the loop while keeping any
        partial synthesis the model already generated.
        """
        if not response.result:
            return response
        ai_msg = response.result[0]
        tool_calls = getattr(ai_msg, "tool_calls", None)
        if tool_calls:
            billable = [
                tc for tc in tool_calls
                if tc.get("name") not in self._free_tools
            ]
            self._state.total_tool_calls += len(billable)
        if self._state.total_tool_calls >= self._max_tool_calls:
            logger.warning(
                "SubagentGuard: tool call limit (%d) reached at %d calls",
                self._max_tool_calls,
                self._state.total_tool_calls,
            )
            # Preserve any text the LLM already generated
            existing_text = getattr(ai_msg, "content", "") or ""
            if isinstance(existing_text, list):
                # Gemini list-of-dicts format
                parts = []
                for item in existing_text:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item["text"])
                existing_text = "\n".join(parts)
            suffix = tool_budget_suffix()
            return AIMessage(content=existing_text + suffix)
        return response

    def _detect_synthesis_tools(self, messages: list) -> None:
        """Scan recent ToolMessages for synthesis tool results."""
        if self._state.synthesis_tool_seen or not self._synthesis_tools:
            return
        completed = [name for name in extract_completed_tool_names(messages) if name in self._synthesis_tools]
        if completed:
            self._state.synthesis_tool_seen = True
            logger.info(
                "SubagentGuard: synthesis tool(s) %s detected — will inject synthesis directive",
                completed,
            )
            return

    def _should_repair_tool_budget(self, last_ai: AIMessage) -> bool:
        if self._agent_name not in {"safety-analyst", "biosteam-analyst"}:
            return False
        if self._state.tool_budget_repairs >= 1:
            return False
        content = getattr(last_ai, "content", "") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)
        return "[LIMIT] Tool call budget exhausted" in str(content)

    def _inject_synthesis_directive(
        self, request: ModelRequest
    ) -> ModelRequest:
        return inject_synthesis_directive(
            request,
            synthesis_tool_seen=self._state.synthesis_tool_seen,
        )

    def _inject_visualization_tool_directive(
        self, request: ModelRequest
    ) -> ModelRequest:
        return inject_visualization_tool_directive(
            request,
            agent_name=self._agent_name,
        )

    def _inject_separation_support_directive(
        self,
        request: ModelRequest,
    ) -> ModelRequest:
        return inject_separation_support_directive(
            request,
            agent_name=self._agent_name,
        )

    def _inject_separation_temperature_bound_directive(
        self,
        request: ModelRequest,
    ) -> ModelRequest:
        return inject_separation_temperature_bound_directive(
            request,
            agent_name=self._agent_name,
        )

    def _restrict_visualization_tools(
        self, request: ModelRequest
    ) -> ModelRequest:
        return restrict_visualization_tools(
            request,
            agent_name=self._agent_name,
        )

    @staticmethod
    def _get_last_ai_message(messages: list) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _should_repair_structured_result(
        self,
        messages: list,
        errors: list[str] | None = None,
    ) -> bool:
        has_separation_analysis_evidence = False
        if self._agent_name == "separation-engineer":
            completed_tool_names = [
                name for name in extract_completed_tool_names(messages)
                if name not in _SEPARATION_NON_ANALYSIS_TOOLS
            ]
            has_separation_analysis_evidence = bool(completed_tool_names)

        if not self._state.synthesis_tool_seen and not has_separation_analysis_evidence:
            return False
        if self._state.structured_result_repairs >= _MAX_STRUCTURED_RESULT_REPAIRS:
            return False

        last_ai = self._get_last_ai_message(messages)
        if last_ai is None:
            return False
        if getattr(last_ai, "tool_calls", None):
            return False

        return bool(
            errors if errors is not None else get_structured_result_errors(last_ai, self._agent_name)
        )

    def _should_repair_separation_analysis_coverage(
        self,
        messages: list,
        errors: list[str] | None = None,
    ) -> bool:
        if self._agent_name != "separation-engineer":
            return False
        if self._state.separation_analysis_repairs >= 1:
            return False

        last_ai = self._get_last_ai_message(messages)
        if last_ai is None:
            return False
        if getattr(last_ai, "tool_calls", None):
            return False

        return bool(
            errors
            if errors is not None
            else get_separation_analysis_coverage_errors(messages, last_ai, self._agent_name)
        )

    def _should_repair_separation_feasibility(
        self,
        messages: list,
        errors: list[str] | None = None,
    ) -> bool:
        if self._agent_name != "separation-engineer":
            return False
        if not self._state.synthesis_tool_seen:
            return False
        if self._state.structured_result_repairs >= _MAX_STRUCTURED_RESULT_REPAIRS:
            return False

        last_ai = self._get_last_ai_message(messages)
        if last_ai is None:
            return False
        if getattr(last_ai, "tool_calls", None):
            return False

        return bool(
            errors
            if errors is not None
            else get_separation_feasibility_errors(last_ai, self._agent_name)
        )

    def _truncate_old_tool_results(
        self, request: ModelRequest
    ) -> ModelRequest:
        """Truncate ToolMessage content for older messages to reduce context
        growth.  Keeps the most recent messages intact so the LLM can still
        reference its latest tool results."""
        if (
            self._truncate_tool_results_after is None
            or self._state.iterations <= 1
        ):
            return request

        limit = self._truncate_tool_results_after
        messages = request.messages
        # Keep the last N messages untruncated (default: 4 = 2 AI+Tool pairs)
        cutoff = len(messages) - self._keep_recent

        truncated = False
        new_messages: list = []
        for i, msg in enumerate(messages):
            if (
                i < cutoff
                and isinstance(msg, ToolMessage)
                and isinstance(msg.content, str)
                and len(msg.content) > limit
            ):
                shortened = (
                    msg.content[:limit]
                    + f"\n\n... [truncated from {len(msg.content)} to "
                    f"{limit} chars]"
                )
                new_messages.append(msg.model_copy(update={"content": shortened}))
                truncated = True
            else:
                new_messages.append(msg)

        if truncated:
            return request.override(messages=new_messages)
        return request

    def _maybe_block_duplicate_biosteam_batch(self, request) -> ToolMessage | None:
        return maybe_block_duplicate_biosteam_batch(
            request,
            agent_name=self._agent_name,
        )

    def _maybe_block_late_separation_todos(self, request) -> ToolMessage | None:
        return maybe_block_late_separation_todos(
            request,
            agent_name=self._agent_name,
        )

    def _maybe_enforce_visualization_tool_directive(self, request) -> ToolMessage | None:
        return maybe_enforce_visualization_tool_directive(
            request,
            agent_name=self._agent_name,
        )
