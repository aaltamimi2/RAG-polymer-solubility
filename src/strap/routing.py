"""Routing middleware for the DISSOLVE orchestrator agent.

Planner-first semantic routing: a single :class:`~strap.route_planner.RoutePlanner`
decision (LLM-backed, keyword fallback) drives the advisory hints appended to
the system prompt, the task()/tool guards, and the workflow progress
machinery. The hints remain advisory — the orchestrator LLM stays in control
within the planned specialist set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from .route_planner import (
    LLMRoutePlannerBackend,
    RoutePlan,
    RoutePlanner,
    build_session_digest,
)
from .routing_classifier import (
    _build_hint_from_matches,
    build_direct_answer_hint,
    generate_routing_table,
    is_direct_solubility_plot_query,
    is_direct_solubility_lookup_query,
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

_DIRECT_ANSWER_BLOCKED_TOOLS = {
    "rank_solvents_selectivity",
    "predict_solubility",
    "predict_solubility_range",
    "list_interpolation_coverage",
}
_GENERIC_DIRECT_HINT = (
    "\n\n[DIRECT_ANSWER: The route planner classified this as a direct core-tool "
    "lookup. Do not delegate to task(). Answer from the core database/lookup tools "
    "and keep the response compact. Do not invent constraints the user did not "
    "provide.]"
)

# Injected on the direct path when this turn has prior conversation. The direct
# hints tell the model to "resolve referents from the compact session context",
# but that context is only built for the planner — without it, a follow-up like
# "what solvents are good under 80C" (after "solubility of LDPE?") loses the
# polymer and the model calls a generic unscoped list tool. This makes the
# carried-over subject explicit so referents resolve and lookups stay scoped.
_DIRECT_FOLLOWUP_DIRECTIVE = (
    "\n\n[FOLLOW-UP CONTEXT: This turn may continue the prior conversation. "
    "Resolve elliptical/pronoun references in the current message (e.g. 'what "
    "solvents are good', 'plot those', 'at 80C') against the SUBJECT of the "
    "prior turn shown below — carry forward the polymer(s) and named solvents, "
    "and scope every tool call to them (pass the polymer argument). Apply any "
    "NEW constraint in the current message, such as a temperature range, on top "
    "of that carried-over subject. Never answer a scoped follow-up with a "
    "generic, unscoped catalog listing.]\n"
)

# -- mid-turn replanning triggers ----------------------------------
_MAX_PLAN_REVISIONS_PER_TURN = 2
_STEP_BUDGET_FAILURE_MARKER = "exhausting its step budget"
_STRUCTURED_RESULT_BLOCK_RE = None  # compiled lazily below


def _qualifying_step_failure(messages: list, plan: RoutePlan) -> tuple[str, str, str] | None:
    """Return (tool_call_id, subagent, reason) when the latest returned
    plan-step dispatch ended in a state that may invalidate the remaining plan:
    a task tool error, a step-budget exhaustion, or an infeasible structured
    result. None otherwise."""
    import json
    import re

    global _STRUCTURED_RESULT_BLOCK_RE
    if _STRUCTURED_RESULT_BLOCK_RE is None:
        _STRUCTURED_RESULT_BLOCK_RE = re.compile(
            r"<STRUCTURED_RESULT>\s*(\{.*?\})\s*</STRUCTURED_RESULT>", re.DOTALL
        )

    dispatches = _extract_returned_subagent_calls(messages)
    if not dispatches:
        return None
    plan_names = set(plan.subagent_names())
    latest = None
    for dispatch in dispatches:  # message order; keep the latest plan-step dispatch
        if dispatch.get("subagent") in plan_names:
            latest = dispatch
    if latest is None:
        return None

    tool_results = _get_tool_message_registry(messages)
    result = tool_results.get(latest["tool_call_id"])
    if result is None:
        return None
    message = result["message"]
    content = message.content if isinstance(message.content, str) else str(message.content)

    if getattr(message, "status", None) == "error":
        return latest["tool_call_id"], latest["subagent"], f"task error — {content[:200]}"
    if _STEP_BUDGET_FAILURE_MARKER in content:
        return latest["tool_call_id"], latest["subagent"], f"step budget exhausted — {content[:200]}"
    match = _STRUCTURED_RESULT_BLOCK_RE.search(content)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = {}
        if str(payload.get("analysis_type") or "").strip().lower() == "infeasible" or payload.get("no_data") is True:
            reason = str(
                payload.get("message") or payload.get("failure_reason") or "infeasible result"
            )
            return latest["tool_call_id"], latest["subagent"], f"infeasible — {reason[:200]}"
    return None


def _plan_step_statuses(
    messages: list,
    plan: RoutePlan,
    failed_subagent: str,
    failed_reason: str,
) -> dict[str, str]:
    """Annotate each prior-plan step for the [PLAN REVISION REQUEST] block."""
    tool_results = _get_tool_message_registry(messages)
    returned_status: dict[str, str] = {}
    for dispatch in _extract_returned_subagent_calls(messages):
        result = tool_results.get(dispatch["tool_call_id"])
        message = result["message"] if result else None
        is_error = getattr(message, "status", None) == "error"
        returned_status[dispatch["subagent"]] = "error" if is_error else "ok"
    statuses: dict[str, str] = {}
    for step in plan.steps:
        if step.subagent == failed_subagent:
            statuses[step.subagent] = f"FAILED: {failed_reason}"
        elif returned_status.get(step.subagent) == "ok":
            statuses[step.subagent] = "completed"
        elif step.subagent in returned_status:
            statuses[step.subagent] = "returned with errors"
        else:
            statuses[step.subagent] = "not started"
    return statuses


# ------------------------------------------------------------------
# Middleware helpers
# ------------------------------------------------------------------

def _latest_ai_origin(messages: list) -> str | None:
    """Return the origin tag on the latest AI message, if present."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            return getattr(msg, "additional_kwargs", {}).get("strap_origin")
    return None


# ------------------------------------------------------------------
# Middleware (class-based)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Middleware (class-based)
# ------------------------------------------------------------------

class RoutingMiddleware(AgentMiddleware):
    """Planner-first semantic routing.

    - First call (no completed subagents): compute the RoutePlan (one
      planner-model call, cached per query; keyword fallback offline) and
      inject the advisory hint derived from it.
    - Subsequent calls: progress tracking against the same plan.
    """

    def __init__(
        self,
        classifier_model: BaseChatModel | None = None,
        *,
        planner: RoutePlanner | None = None,
    ) -> None:
        if planner is None:
            backend = LLMRoutePlannerBackend(classifier_model) if classifier_model is not None else None
            planner = RoutePlanner(backend=backend)
        self._planner = planner

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        self._maybe_replan(request.messages)
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
        self._maybe_replan(request.messages)
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
        if _latest_ai_origin(state.get("messages", [])) == "direct_tool_fast_path":
            return None
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
        state_messages = (request.state or {}).get("messages", [])
        query_text = _get_last_human_message(state_messages) or ""
        plan = self._get_plan(query_text, state_messages)
        # Hard (execution-affecting) blocks require an authoritative plan:
        # planner-sourced, or fallback in a deliberate no-backend deployment.
        # A degraded planner leaves keyword output advisory-only.
        if not self._planner.is_authoritative(plan):
            return handler(request)
        if tool_call.get("name") == "task" and plan.is_direct:
            return ToolMessage(
                content=(
                    "Router guard: this is a direct core-tool lookup, not a specialist workflow. "
                    "Do not delegate to `task()`. Use the deterministic/direct tool path and "
                    "answer from the resulting structured lookup."
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )
        if tool_call.get("name") == "task" and plan.is_specialists:
            requested = str((tool_call.get("args") or {}).get("subagent_type") or "")
            planned_names = plan.subagent_names()
            if (
                requested
                and requested not in planned_names
                and plan.source == "planner"
                and plan.confidence == "high"
            ):
                specialists = ", ".join(planned_names)
                return ToolMessage(
                    content=(
                        f"Router guard: `{requested}` is not part of the planned workflow "
                        f"for this query. Dispatch task() for: {specialists}. If the user's "
                        "request genuinely needs another specialist, re-read the query and "
                        "explain the deviation in your final synthesis."
                    ),
                    tool_call_id=tool_call["id"],
                    status="error",
                )
        if (
            plan.is_direct
            and not is_direct_solubility_lookup_query(query_text)
            and not is_direct_solubility_plot_query(query_text)
            and tool_call.get("name") in _DIRECT_ANSWER_BLOCKED_TOOLS
        ):
            return ToolMessage(
                content=(
                    "Router guard: this is a direct solvent lookup. Do not run "
                    f"`{tool_call.get('name')}` unless the user explicitly asks for "
                    "temperature-dependent solubility, selectivity, ranking, or separation. "
                    "Use `list_available_solvents(polymer=<target polymer>)` and answer from that lookup."
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )
        if tool_call.get("name") == "write_todos":
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
            plan = self._get_plan(query_text or "", request.messages)
            if plan.is_direct:
                hint = build_direct_answer_hint(query_text or "") or _GENERIC_DIRECT_HINT
                # Carry the prior-turn subject onto the direct path so follow-up
                # referents resolve and lookups stay scoped to the polymer.
                digest = build_session_digest(request.messages)
                if digest:
                    hint = f"{hint}{_DIRECT_FOLLOWUP_DIRECTIVE}{digest}"
            elif plan.is_specialists:
                hint = _build_hint_from_matches(allowed_rules, query_text=query_text or "")
            else:
                hint = None

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

    def _get_plan(self, query_text: str, messages: list | None = None) -> RoutePlan:
        """Compute (or reuse) the session-aware RoutePlan for the active query.

        The session digest is derived from history before the current user
        turn, so every consumer sees the same plan for the whole turn; the
        planner's own cache is keyed by (query, digest).
        """
        return self._planner.plan(
            query_text,
            session_digest=build_session_digest(messages) if messages else None,
        )

    def _maybe_replan(self, messages: list) -> None:
        """Mid-turn plan revision after a qualifying step outcome.

        Runs before the allowed-rules computation so a revision, when applied,
        feeds every downstream guard in the same model call. The plan's
        structure is otherwise frozen for the turn; only a typed step failure
        (task error, step-budget exhaustion, infeasible structured result) can
        trigger a revision, each outcome can trigger at most one, and at most
        two revisions are applied per turn. A degraded planner never revises —
        the orchestrator model is left to adapt on its own.
        """
        query_text = (_get_last_human_message(messages) or "").strip()
        if not query_text:
            return
        plan = self._get_plan(query_text, messages)
        if not plan.steps or not self._planner.is_authoritative(plan):
            return
        revision_markers = [
            str(note) for note in plan.validation_notes
            if str(note).startswith("revised_after:")
        ]
        if len(revision_markers) >= _MAX_PLAN_REVISIONS_PER_TURN:
            return
        outcome = _qualifying_step_failure(messages, plan)
        if outcome is None:
            return
        tool_call_id, subagent, reason = outcome
        if f"revised_after:{tool_call_id}" in revision_markers:
            return
        revised = self._planner.revise(
            query_text,
            session_digest=build_session_digest(messages) if messages else None,
            prior_plan=plan,
            step_statuses=_plan_step_statuses(messages, plan, subagent, reason),
            outcome_key=tool_call_id,
        )
        if revised is not None:
            logger.info(
                "routing_middleware: mid-turn replan after %s outcome on %s — steps %s -> %s",
                reason.split(" — ", 1)[0],
                subagent,
                plan.subagent_names(),
                revised.subagent_names(),
            )

    def _get_allowed_rules(self, messages: list) -> list[dict]:
        """Return the planned specialist set for the active query as rule dicts."""
        query_text = _get_last_human_message(messages) or ""
        plan = self._get_plan(query_text.strip(), messages)
        logger.info(
            "routing_middleware: route_decision mode=%s planned=%s source=%s confidence=%s",
            plan.mode,
            plan.subagent_names(),
            plan.source,
            plan.confidence,
        )
        return plan.to_rules()

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
