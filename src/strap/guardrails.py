"""Subagent guardrail middleware: iteration cap + token budget + tool-call
limit + synthesis injection + old tool-result truncation."""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest

logger = logging.getLogger(__name__)


@dataclass
class _GuardState:
    iterations: int = 0
    total_prompt_tokens: int = 0
    total_tool_calls: int = 0
    synthesis_tool_seen: bool = False


_guard_state: contextvars.ContextVar[_GuardState] = contextvars.ContextVar(
    "_guard_state"
)


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
    ) -> None:
        self._max_iterations = max_iterations
        self._token_budget = token_budget
        self._max_tool_calls = max_tool_calls
        self._synthesis_tools: set[str] = synthesis_tools or set()
        self._truncate_tool_results_after = truncate_tool_results_after
        self._free_tools: set[str] = free_tools or set()
        self._keep_recent = keep_recent

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
            return AIMessage(
                content="[LIMIT] Max iterations reached. Synthesize your answer now.",
            )

        # Detect synthesis tool results in the conversation
        self._detect_synthesis_tools(request.messages)

        # Inject synthesis directive if a key tool has already returned
        request = self._inject_synthesis_directive(request)

        # Truncate old tool results to limit context growth
        request = self._truncate_old_tool_results(request)

        response = handler(request)

        # Track tokens
        self._track_tokens(response)
        if self._state.total_prompt_tokens > self._token_budget:
            logger.warning(
                "SubagentGuard: token budget (%d) exceeded at %d tokens",
                self._token_budget,
                self._state.total_prompt_tokens,
            )
            return AIMessage(
                content="[LIMIT] Token budget exceeded. Synthesize your answer now.",
            )

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
            return AIMessage(
                content="[LIMIT] Max iterations reached. Synthesize your answer now.",
            )

        # Detect synthesis tool results in the conversation
        self._detect_synthesis_tools(request.messages)

        # Inject synthesis directive if a key tool has already returned
        request = self._inject_synthesis_directive(request)

        # Truncate old tool results to limit context growth
        request = self._truncate_old_tool_results(request)

        response = await handler(request)

        # Track tokens
        self._track_tokens(response)
        if self._state.total_prompt_tokens > self._token_budget:
            logger.warning(
                "SubagentGuard: token budget (%d) exceeded at %d tokens",
                self._token_budget,
                self._state.total_prompt_tokens,
            )
            return AIMessage(
                content="[LIMIT] Token budget exceeded. Synthesize your answer now.",
            )

        # Track and enforce tool-call limit
        return self._enforce_tool_call_limit(response)

    # -- helpers -------------------------------------------------------------

    def _track_tokens(self, response: ModelResponse) -> None:
        """Accumulate prompt tokens from the model response."""
        if not response.result:
            return
        ai_msg = response.result[0]
        usage = getattr(ai_msg, "usage_metadata", None)
        if usage:
            self._state.total_prompt_tokens += usage.get("input_tokens", 0)

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
            suffix = (
                "\n\n[LIMIT] Tool call budget exhausted. Synthesize your "
                "findings into a clear, complete answer NOW. Do NOT call "
                "any more tools."
            )
            return AIMessage(content=existing_text + suffix)
        return response

    def _detect_synthesis_tools(self, messages: list) -> None:
        """Scan recent ToolMessages for synthesis tool results."""
        if self._state.synthesis_tool_seen or not self._synthesis_tools:
            return
        # Walk backwards from the end; stop at the first AIMessage
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None)
                if tool_name and tool_name in self._synthesis_tools:
                    self._state.synthesis_tool_seen = True
                    logger.info(
                        "SubagentGuard: synthesis tool '%s' detected — "
                        "will inject synthesis directive",
                        tool_name,
                    )
                    return
            elif isinstance(msg, AIMessage):
                break

    def _inject_synthesis_directive(
        self, request: ModelRequest
    ) -> ModelRequest:
        """Append a synthesis directive to the system prompt if a key tool
        has already returned results."""
        if not self._state.synthesis_tool_seen or request.system_message is None:
            return request
        directive = (
            "\n\n[NOTE] A comprehensive analysis tool has returned results. "
            "Consider synthesizing your findings now unless you need "
            "additional data for a complete answer."
        )
        new_system = append_to_system_message(
            request.system_message, directive
        )
        return request.override(system_message=new_system)

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
