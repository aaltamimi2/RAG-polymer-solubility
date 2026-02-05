"""Subagent guardrail middleware: iteration cap + token budget."""

from __future__ import annotations

import logging

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class SubagentGuardMiddleware(AgentMiddleware):
    """Middleware that enforces iteration and token limits on subagents.

    Prevents runaway subagent loops by capping:
    - The number of model calls (iterations)
    - The cumulative prompt token usage

    When a limit is hit the middleware short-circuits with an AIMessage
    containing no tool calls, which causes the LangGraph agent loop to
    terminate gracefully.
    """

    def __init__(
        self,
        max_iterations: int = 25,
        token_budget: int = 200_000,
    ) -> None:
        self._max_iterations = max_iterations
        self._token_budget = token_budget
        self._iterations = 0
        self._total_prompt_tokens = 0

    # -- lifecycle: reset counters at task() invocation start ----------------

    def before_agent(self, state, runtime):
        self._iterations = 0
        self._total_prompt_tokens = 0

    async def abefore_agent(self, state, runtime):
        self._iterations = 0
        self._total_prompt_tokens = 0

    # -- model-call wrapper: enforce limits ----------------------------------

    def wrap_model_call(self, request, handler):
        self._iterations += 1
        if self._iterations > self._max_iterations:
            logger.warning(
                "SubagentGuard: iteration limit (%d) reached",
                self._max_iterations,
            )
            return AIMessage(
                content="[LIMIT] Max iterations reached. Synthesize your answer now.",
            )

        response = handler(request)

        self._track_tokens(response)
        if self._total_prompt_tokens > self._token_budget:
            logger.warning(
                "SubagentGuard: token budget (%d) exceeded at %d tokens",
                self._token_budget,
                self._total_prompt_tokens,
            )
            return AIMessage(
                content="[LIMIT] Token budget exceeded. Synthesize your answer now.",
            )
        return response

    async def awrap_model_call(self, request, handler):
        self._iterations += 1
        if self._iterations > self._max_iterations:
            logger.warning(
                "SubagentGuard: iteration limit (%d) reached",
                self._max_iterations,
            )
            return AIMessage(
                content="[LIMIT] Max iterations reached. Synthesize your answer now.",
            )

        response = await handler(request)

        self._track_tokens(response)
        if self._total_prompt_tokens > self._token_budget:
            logger.warning(
                "SubagentGuard: token budget (%d) exceeded at %d tokens",
                self._token_budget,
                self._total_prompt_tokens,
            )
            return AIMessage(
                content="[LIMIT] Token budget exceeded. Synthesize your answer now.",
            )
        return response

    # -- helpers -------------------------------------------------------------

    def _track_tokens(self, response: ModelResponse) -> None:
        """Accumulate prompt tokens from the model response."""
        if not response.result:
            return
        ai_msg = response.result[0]
        usage = getattr(ai_msg, "usage_metadata", None)
        if usage:
            self._total_prompt_tokens += usage.get("input_tokens", 0)
