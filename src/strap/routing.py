"""Routing middleware for the DISSOLVE orchestrator agent.

Provides LLM-based semantic routing with keyword fallback. Advisory hints
are appended to the system prompt before each LLM call. The hints are
advisory — the LLM remains in control and can override them.

Single source of truth: ROUTING_RULES defines both the prompt routing table
(via generate_routing_table()), the runtime keyword matcher (via
classify_query_keywords()), and the LLM classifier prompt (via
_build_subagent_list()).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from .routing_classifier import (
    _build_hint_from_matches,
    _normalize_matched_rules,
    classify_query_keywords,
    classify_query_llm,
    generate_routing_table,
    order_workflow_rules,
    plan_workflow_rules,
    select_workflow_rules,
)

from .routing_progress import (
    _build_progress_directive,
    _build_write_todos_guard_messages,
    _extract_all_task_subagents,
    _extract_completed_subagent_calls,
    _extract_completed_subagents,
    _extract_failed_subagent_calls,
    _extract_returned_subagent_calls,
    _get_active_remaining_steps,
    _get_allowed_subagent_names,
    _get_effective_completed_task_ids,
    _get_effective_failed_task_ids,
    _get_last_human_message,
    _get_latest_dispatch_for_subagent,
    _get_ordered_plan,
    _get_pending_required_handoff,
    _get_ready_downstream_consumer,
    _get_ready_downstream_handoff,
    _get_task_handoff_statuses,
    _get_tool_message_registry,
    _has_active_progress,
    _has_built_handoff_since,
    _is_router_guard_message,
    _is_retry_of_failed_dispatch,
    _normalize_task_description,
    _parse_tool_json_content,
    _should_block_write_todos,
    _task_descriptions_match,
    _validate_write_todos_call,
    _workflow_is_active,
)

from .routing_guards import (
    _build_filesystem_guard_messages,
    _build_initial_route_task_response,
    _build_incomplete_route_retry_hint,
    _build_multi_specialist_completion_response,
    _build_pending_handoff_response,
    _build_ready_handoff_response,
    _build_single_specialist_completion_response,
    _build_workflow_guard_messages,
    _response_matches_pending_handoff,
    _response_matches_ready_handoff,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel
    from collections.abc import Callable
# ------------------------------------------------------------------
# Middleware helpers
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Middleware (class-based)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Middleware (class-based)
# ------------------------------------------------------------------

class RoutingMiddleware(AgentMiddleware):
    """LLM-based semantic routing with keyword fallback.

    - First call (no completed subagents): try LLM classifier → fall back
      to keywords → build advisory hint.
    - Subsequent calls: progress tracking only (keyword-based, no LLM needed).
    """

    def __init__(self, classifier_model: BaseChatModel | None = None) -> None:
        self._classifier_model = classifier_model
        self._route_cache: dict[str, list[dict]] = {}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        allowed_rules = self._get_allowed_rules(request.messages)
        short_circuit = _build_single_specialist_completion_response(request.messages, allowed_rules)
        if short_circuit is not None:
            ai_msg = short_circuit.result[0] if getattr(short_circuit, "result", None) else None
            logger.info(
                "routing_middleware: short-circuiting final synthesis from completed single-specialist result origin=%s subagent=%s",
                getattr(ai_msg, "additional_kwargs", {}).get("strap_origin"),
                getattr(ai_msg, "additional_kwargs", {}).get("strap_subagent"),
            )
            return short_circuit
        multi_specialist_short_circuit = _build_multi_specialist_completion_response(
            request.messages,
            allowed_rules,
        )
        if multi_specialist_short_circuit is not None:
            ai_msg = multi_specialist_short_circuit.result[0] if getattr(multi_specialist_short_circuit, "result", None) else None
            logger.info(
                "routing_middleware: short-circuiting final synthesis from completed multi-specialist result origin=%s",
                getattr(ai_msg, "additional_kwargs", {}).get("strap_origin"),
            )
            return multi_specialist_short_circuit
        response = handler(request)
        response = self._autobuild_pending_handoff(request, response, allowed_rules)
        response = self._autodispatch_ready_handoff(request, response, allowed_rules)
        response = self._retry_incomplete_route_once(request, response, handler, allowed_rules)
        self._log_decision(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        allowed_rules = self._get_allowed_rules(request.messages)
        short_circuit = _build_single_specialist_completion_response(request.messages, allowed_rules)
        if short_circuit is not None:
            ai_msg = short_circuit.result[0] if getattr(short_circuit, "result", None) else None
            logger.info(
                "routing_middleware: short-circuiting final synthesis from completed single-specialist result (async) origin=%s subagent=%s",
                getattr(ai_msg, "additional_kwargs", {}).get("strap_origin"),
                getattr(ai_msg, "additional_kwargs", {}).get("strap_subagent"),
            )
            return short_circuit
        multi_specialist_short_circuit = _build_multi_specialist_completion_response(
            request.messages,
            allowed_rules,
        )
        if multi_specialist_short_circuit is not None:
            ai_msg = multi_specialist_short_circuit.result[0] if getattr(multi_specialist_short_circuit, "result", None) else None
            logger.info(
                "routing_middleware: short-circuiting final synthesis from completed multi-specialist result (async) origin=%s",
                getattr(ai_msg, "additional_kwargs", {}).get("strap_origin"),
            )
            return multi_specialist_short_circuit
        response = await handler(request)
        response = self._autobuild_pending_handoff(request, response, allowed_rules)
        response = self._autodispatch_ready_handoff(request, response, allowed_rules)
        response = await self._aretry_incomplete_route_once(request, response, handler, allowed_rules)
        self._log_decision(response)
        return response

    def after_model(self, state, runtime) -> dict[str, list[ToolMessage]] | None:
        allowed_rules = self._get_allowed_rules(state["messages"])
        guard_messages = _build_write_todos_guard_messages(state["messages"], allowed_rules)
        guard_messages.extend(_build_workflow_guard_messages(state["messages"], allowed_rules))
        guard_messages.extend(_build_filesystem_guard_messages(state["messages"]))
        if guard_messages:
            logger.info(
                "routing_middleware: blocked guarded tool calls=%s",
                [msg.tool_call_id for msg in guard_messages],
            )
            return {"messages": guard_messages}
        return None

    async def aafter_model(self, state, runtime) -> dict[str, list[ToolMessage]] | None:
        return self.after_model(state, runtime)

    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call
        if tool_call.get("name") == "write_todos":
            state_messages = (request.state or {}).get("messages", [])
            allowed_rules = self._get_allowed_rules(state_messages)
            if _should_block_write_todos(state_messages, allowed_rules):
                allowed_names = sorted(_get_allowed_subagent_names(allowed_rules))
                if allowed_names:
                    specialists = ", ".join(allowed_names)
                    message = (
                        "Router guard: `write_todos` is disabled before the first specialist dispatch "
                        f"for this routed workflow. Dispatch `task(...)` for: {specialists}."
                    )
                else:
                    message = (
                        "Router guard: `write_todos` is disabled before the first specialist dispatch "
                        "for this routed workflow."
                    )
                return ToolMessage(
                    content=message,
                    tool_call_id=tool_call["id"],
                    status="error",
                )
            validation_error = _validate_write_todos_call(tool_call)
            if validation_error is not None:
                return ToolMessage(
                    content=f"Router guard: {validation_error}",
                    tool_call_id=tool_call["id"],
                    status="error",
                )
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        result = self.wrap_tool_call(request, lambda req: None)
        if isinstance(result, ToolMessage):
            return result
        return await handler(request)

    def _inject_hint(self, request: ModelRequest) -> ModelRequest:
        """Inject routing or progress hint into the system message."""
        returned_calls = _extract_returned_subagent_calls(request.messages)
        completed_calls = _extract_completed_subagent_calls(request.messages)
        failed_calls = _extract_failed_subagent_calls(request.messages)
        query_text = _get_last_human_message(request.messages)
        allowed_rules = self._get_allowed_rules(request.messages)

        if not returned_calls:
            hint = _build_hint_from_matches(allowed_rules)

            if hint:
                logger.info(
                    "routing_middleware: advisory hint injected for query=%s",
                    (query_text or "")[:80],
                )
            else:
                logger.info(
                    "routing_middleware: no routing hint for query=%s",
                    (query_text or "")[:80],
                )

            if hint and request.system_message is not None:
                new_system = append_to_system_message(request.system_message, hint)
                return request.override(system_message=new_system)

            return request

        # After task() calls: inject progress directive based on actual history.
        # _get_ordered_plan derives the plan from what the orchestrator actually
        # dispatched, not from re-predicting the original query.
        ordered_plan = _get_ordered_plan(request.messages, allowed_rules=allowed_rules)

        if ordered_plan:
            completed_ids = _get_effective_completed_task_ids(request.messages)
            failed_ids = _get_effective_failed_task_ids(request.messages)
            remaining = _get_active_remaining_steps(request.messages, ordered_plan)
            progress = _build_progress_directive(
                request.messages,
                completed_ids,
                ordered_plan,
                failed_ids=failed_ids,
            )
            if progress and request.system_message is not None:
                new_system = append_to_system_message(
                    request.system_message, progress
                )
                logger.info(
                    "routing_middleware: progress directive injected, "
                    "completed=%s failed=%s remaining=%s",
                    [r["subagent"] for r in ordered_plan if r["step_id"] in completed_ids],
                    [r["subagent"] for r in ordered_plan if r["step_id"] in failed_ids],
                    [r["subagent"] for r in remaining],
                )
                return request.override(system_message=new_system)

        return request

    def _log_decision(self, response) -> None:
        """Log what the LLM decided to do after routing."""
        result = getattr(response, "result", None)
        if result:
            ai_msg = result[0]
            tool_calls = getattr(ai_msg, "tool_calls", None)
            if tool_calls:
                tool_names = [tc.get("name", "?") for tc in tool_calls]
                logger.info(
                    "routing_middleware: LLM decided tool_calls=%s", tool_names,
                )

    def _get_allowed_rules(self, messages: list) -> list[dict]:
        """Return the classifier-selected specialist set for the active query."""
        query_text = _get_last_human_message(messages) or ""
        cache_key = query_text.strip()
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        matched = None
        keyword_matched = classify_query_keywords(messages)
        if self._classifier_model and cache_key:
            matched = classify_query_llm(cache_key, self._classifier_model)

        matched = select_workflow_rules(
            cache_key,
            llm_matched=matched,
            keyword_matched=keyword_matched,
        )
        matched = plan_workflow_rules(cache_key, matched)
        self._route_cache[cache_key] = matched or []
        return self._route_cache[cache_key]

    def _autodispatch_ready_handoff(self, request: ModelRequest, response, allowed_rules):
        """Synthesize the downstream task() once the next handoff is already validated."""
        ready_handoff = _get_ready_downstream_handoff(request.messages, allowed_rules)
        if ready_handoff is None:
            return response

        if _response_matches_ready_handoff(response, ready_handoff):
            return response

        logger.info(
            "routing_middleware: auto-dispatching ready handoff consumer=%s handoff_id=%s",
            ready_handoff.get("consumer"),
            ready_handoff.get("handoff_id"),
        )
        return _build_ready_handoff_response(ready_handoff, response=response)

    def _autobuild_pending_handoff(self, request: ModelRequest, response, allowed_rules):
        """Synthesize build_handoff() once the next sequential edge is known."""
        pending_handoff = _get_pending_required_handoff(request.messages, allowed_rules)
        if pending_handoff is None:
            return response

        if _response_matches_pending_handoff(response, pending_handoff):
            return response

        producer, consumer = pending_handoff
        logger.info(
            "routing_middleware: auto-building pending handoff producer=%s consumer=%s",
            producer,
            consumer,
        )
        return _build_pending_handoff_response(pending_handoff, response=response)

    def _retry_incomplete_route_once(self, request: ModelRequest, response, handler, allowed_rules=None):
        """Retry once when the model stops before the routed workflow is complete."""
        result = getattr(response, "result", None)
        if not result:
            return response

        ai_msg = result[0]
        if getattr(ai_msg, "tool_calls", None):
            return response

        if allowed_rules is None:
            allowed_rules = self._get_allowed_rules(request.messages)
        start_response = _build_initial_route_task_response(request.messages, allowed_rules)
        if start_response is not None:
            logger.info("routing_middleware: synthesizing initial task dispatch for routed workflow")
            return start_response
        retry_hint = _build_incomplete_route_retry_hint(request.messages, allowed_rules)
        if retry_hint is None or request.system_message is None:
            return response

        system_text = str(getattr(request.system_message, "content", request.system_message))
        if "ROUTER_RETRY:" in system_text:
            return response

        retry_request = request.override(
            system_message=append_to_system_message(request.system_message, retry_hint)
        )
        logger.info("routing_middleware: retrying incomplete route with hard directive")
        return handler(retry_request)

    async def _aretry_incomplete_route_once(self, request: ModelRequest, response, handler, allowed_rules=None):
        """Async variant of the incomplete-route retry."""
        result = getattr(response, "result", None)
        if not result:
            return response

        ai_msg = result[0]
        if getattr(ai_msg, "tool_calls", None):
            return response

        if allowed_rules is None:
            allowed_rules = self._get_allowed_rules(request.messages)
        start_response = _build_initial_route_task_response(request.messages, allowed_rules)
        if start_response is not None:
            logger.info("routing_middleware: synthesizing initial task dispatch for routed workflow (async)")
            return start_response
        retry_hint = _build_incomplete_route_retry_hint(request.messages, allowed_rules)
        if retry_hint is None or request.system_message is None:
            return response

        system_text = str(getattr(request.system_message, "content", request.system_message))
        if "ROUTER_RETRY:" in system_text:
            return response

        retry_request = request.override(
            system_message=append_to_system_message(request.system_message, retry_hint)
        )
        logger.info("routing_middleware: retrying incomplete route with hard directive (async)")
        return await handler(retry_request)


# Module-level instance for backward compat (keyword-only, no LLM)
routing_middleware = RoutingMiddleware(classifier_model=None)
