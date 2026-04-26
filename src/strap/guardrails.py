"""Subagent guardrail middleware: iteration cap + token budget + tool-call
limit + synthesis injection + old tool-result truncation."""

from __future__ import annotations

import contextvars
import json
import logging
import re
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
    visualization_required_tool_repair_message,
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
from .guardrail_utils import (
    coerce_message_text,
    extract_completed_tool_names,
    extract_required_visualization_tool,
)
from .handoff_store import get_handoff
from .query_context import extract_query_context

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
    visualization_required_tool_repairs: int = 0


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
_SEPARATION_TOP_K_TOOL_ARGS = {
    "plan_sequential_separation": "top_k_solvents",
    "view_alternative_separation_sequence": "top_k_solvents",
    "analyze_integrated_separation": "top_k",
}
_BLOCKED_SUBAGENT_FILESYSTEM_TOOLS = {"grep", "glob", "execute", "edit_file"}


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
        missing_visualization_tool = self._missing_required_visualization_tool(messages, last_ai)
        if missing_visualization_tool is not None:
            self._state.visualization_required_tool_repairs += 1
            return {
                "messages": [
                    HumanMessage(
                        content=visualization_required_tool_repair_message(missing_visualization_tool)
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
        blocked = self._maybe_block_subagent_filesystem_tool(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_enforce_visualization_tool_directive(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_late_separation_todos(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_duplicate_biosteam_batch(request)
        if blocked is not None:
            return blocked
        prepared = self._maybe_prepare_separation_tool_call(request)
        if prepared is not request:
            return handler(prepared)
        prepared = self._maybe_prepare_optimization_tool_call(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return handler(prepared)

    async def awrap_tool_call(self, request, handler):
        blocked = self._maybe_block_subagent_filesystem_tool(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_enforce_visualization_tool_directive(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_late_separation_todos(request)
        if blocked is not None:
            return blocked
        blocked = self._maybe_block_duplicate_biosteam_batch(request)
        if blocked is not None:
            return blocked
        prepared = self._maybe_prepare_separation_tool_call(request)
        if prepared is not request:
            return await handler(prepared)
        prepared = self._maybe_prepare_optimization_tool_call(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return await handler(prepared)

    def _maybe_block_subagent_filesystem_tool(self, request):
        if not self._agent_name:
            return None
        tool_call = request.tool_call or {}
        tool_name = tool_call.get("name")
        if tool_name not in _BLOCKED_SUBAGENT_FILESYSTEM_TOOLS:
            return None
        return ToolMessage(
            content=(
                f"Subagent guard: `{tool_name}` is disabled for `{self._agent_name}`. "
                "Use the attached handoff payload, handoff/domain tools, or read_file only "
                "when an exact returned file path was provided."
            ),
            tool_call_id=tool_call.get("id"),
            status="error",
        )

    def _maybe_prepare_separation_tool_call(self, request):
        if self._agent_name != "separation-engineer":
            return request
        tool_call = request.tool_call or {}
        tool_name = tool_call.get("name")
        arg_name = _SEPARATION_TOP_K_TOOL_ARGS.get(str(tool_name or ""))
        if not arg_name:
            return request
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return request

        state_map = self._state_mapping(request.state)
        requested_top_k = self._infer_requested_separation_top_k(state_map)
        if requested_top_k is None:
            return request
        try:
            current_top_k = int(args.get(arg_name) or 0)
        except (TypeError, ValueError):
            current_top_k = 0
        if current_top_k >= requested_top_k:
            return request

        repaired_args = dict(args)
        repaired_args[arg_name] = requested_top_k
        new_tool_call = {**tool_call, "args": repaired_args}
        return request.override(tool_call=new_tool_call)

    def _infer_requested_separation_top_k(self, state_map: dict) -> int | None:
        text_parts: list[str] = []
        for message in state_map.get("messages") or []:
            if isinstance(message, HumanMessage):
                text = coerce_message_text(message.content)
                if text:
                    text_parts.append(text)
        text = "\n".join(text_parts).lower()
        if not text:
            return None
        patterns = (
            r"top\s+(\d+)\s+(?:unique\s+)?solvent\s+candidates?\s+per\s+polymer",
            r"top\s+(\d+)\s+(?:unique\s+)?solvent\s+choices?\s+per\s+polymer",
            r"top\s+(\d+)\s+(?:unique\s+)?(?:polymer-)?solvent\s+pairs?",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return min(value, 50)
        return None

    def _maybe_prepare_optimization_tool_call(self, request):
        if self._agent_name != "optimization-engineer":
            return request
        tool_call = request.tool_call or {}
        tool_name = tool_call.get("name")
        if tool_name not in {
            "run_waste_management_optimization",
            "run_waste_management_pareto",
            "run_waste_management_pareto_slices",
        }:
            return request
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return ToolMessage(
                content="Optimization guard: tool arguments must be a JSON object.",
                tool_call_id=tool_call.get("id"),
                status="error",
            )

        state_map = self._state_mapping(request.state)
        handoff_payload = self._get_optimization_handoff_payload(state_map, args)
        feed_composition, feed_capacity = self._infer_optimization_feed_inputs(state_map, handoff_payload)
        composition_slices = self._infer_optimization_composition_slices(state_map)
        if tool_name == "run_waste_management_pareto" and len(composition_slices) > 1:
            return ToolMessage(
                content=(
                    "Optimization guard: this is a multi-composition Pareto request. "
                    "Use `run_waste_management_pareto_slices` with "
                    f"`composition_slices_json={json.dumps(composition_slices)}` instead of solving only one slice."
                ),
                tool_call_id=tool_call.get("id"),
                status="error",
            )
        repaired: list[str] = []
        repaired_args = dict(args)

        stage_value = repaired_args.get("stage_candidates_json")
        stage_payload, stage_error = self._coerce_mapping_arg(stage_value, "stage_candidates_json")
        if stage_error:
            if handoff_payload is not None:
                repaired_args["stage_candidates_json"] = handoff_payload
                repaired.append("replaced malformed stage_candidates_json with attached typed handoff payload")
            else:
                return ToolMessage(
                    content=f"Optimization guard: {stage_error}. Retry using the exact attached typed handoff payload only.",
                    tool_call_id=tool_call.get("id"),
                    status="error",
                )
        elif stage_payload is None and handoff_payload is not None:
            repaired_args["stage_candidates_json"] = handoff_payload
            repaired.append("injected attached typed handoff payload as stage_candidates_json")

        feed_value = repaired_args.get("feed_composition_json")
        feed_payload, feed_error = self._coerce_mapping_arg(feed_value, "feed_composition_json")
        if feed_error:
            if feed_composition:
                repaired_args["feed_composition_json"] = feed_composition
                repaired.append("replaced malformed feed_composition_json with inferred feed composition")
            else:
                return ToolMessage(
                    content=f"Optimization guard: {feed_error}. Retry with an explicit feed_composition_json mapping or legacy fractions.",
                    tool_call_id=tool_call.get("id"),
                    status="error",
                )
        elif feed_payload is None and feed_composition:
            repaired_args["feed_composition_json"] = feed_composition
            repaired.append("injected inferred feed_composition_json")

        if repaired_args.get("feed") in (None, ""):
            if feed_capacity is not None:
                repaired_args["feed"] = feed_capacity
                repaired.append("injected inferred feed tonnes/year")
            else:
                return ToolMessage(
                    content="Optimization guard: missing `feed` in tonnes/year. Provide the total feed or include it in the routed handoff.",
                    tool_call_id=tool_call.get("id"),
                    status="error",
                )

        if tool_name == "run_waste_management_pareto_slices":
            slices_value = repaired_args.get("composition_slices_json")
            slices_payload, slices_error = self._coerce_sequence_or_mapping_arg(
                slices_value,
                "composition_slices_json",
            )
            if composition_slices:
                if slices_payload != composition_slices:
                    repaired_args["composition_slices_json"] = composition_slices
                    repaired.append("replaced composition_slices_json with inferred composition slices")
            elif slices_error:
                if composition_slices:
                    repaired_args["composition_slices_json"] = composition_slices
                    repaired.append("replaced malformed composition_slices_json with inferred composition slices")
                else:
                    return ToolMessage(
                        content=f"Optimization guard: {slices_error}. Retry with a list of fixed feed-composition mappings.",
                        tool_call_id=tool_call.get("id"),
                        status="error",
                    )
            elif slices_payload is None:
                if composition_slices:
                    repaired_args["composition_slices_json"] = composition_slices
                    repaired.append("injected inferred composition_slices_json")
                else:
                    return ToolMessage(
                        content="Optimization guard: missing `composition_slices_json` for the multi-slice Pareto tool.",
                        tool_call_id=tool_call.get("id"),
                        status="error",
                    )

            if not repaired:
                return request

            logger.info(
                "optimization_preflight: repaired %s with steps=%s",
                tool_name,
                repaired,
            )
            new_state = self._append_preflight_notes(state_map, repaired)
            new_tool_call = {**tool_call, "args": repaired_args}
            if new_state is not None:
                return request.override(tool_call=new_tool_call, state=new_state)
            return request.override(tool_call=new_tool_call)

        if repaired_args.get("feed_composition_json") in (None, "", {}):
            missing_legacy = [
                name
                for name in ("pe_fraction", "pet_fraction", "n6_fraction", "evoh_fraction")
                if repaired_args.get(name) is None
            ]
            if missing_legacy:
                return ToolMessage(
                    content=(
                        "Optimization guard: feed_composition_json is missing and the legacy feed fractions are incomplete. "
                        f"Missing: {', '.join(missing_legacy)}."
                    ),
                    tool_call_id=tool_call.get("id"),
                    status="error",
                )

        if not repaired:
            return request

        logger.info(
            "optimization_preflight: repaired %s with steps=%s",
            tool_name,
            repaired,
        )
        new_state = self._append_preflight_notes(state_map, repaired)
        new_tool_call = {**tool_call, "args": repaired_args}
        if new_state is not None:
            return request.override(tool_call=new_tool_call, state=new_state)
        return request.override(tool_call=new_tool_call)

    def _state_mapping(self, state) -> dict:
        if isinstance(state, dict):
            return dict(state)
        if hasattr(state, "model_dump"):
            dumped = state.model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        return {}

    def _append_preflight_notes(self, state_map: dict, repaired: list[str]) -> dict | None:
        if not state_map:
            return None
        notes = list(state_map.get("strap_optimization_preflight") or [])
        notes.append({"repairs": list(repaired)})
        new_state = dict(state_map)
        new_state["strap_optimization_preflight"] = notes
        return new_state

    def _coerce_mapping_arg(self, value, field_name: str) -> tuple[dict | None, str | None]:
        if value in (None, "", {}):
            return None, None
        if isinstance(value, dict):
            return dict(value), None
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                return None, f"{field_name} is not valid JSON ({exc})"
            if not isinstance(payload, dict):
                return None, f"{field_name} must decode to a JSON object"
            return dict(payload), None
        return None, f"{field_name} must be a JSON object or JSON string"

    def _coerce_sequence_or_mapping_arg(self, value, field_name: str) -> tuple[list | dict | None, str | None]:
        if value in (None, "", {}, []):
            return None, None
        if isinstance(value, (list, dict)):
            return value, None
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                return None, f"{field_name} is not valid JSON ({exc})"
            if not isinstance(payload, (list, dict)):
                return None, f"{field_name} must decode to a list or JSON object"
            return payload, None
        return None, f"{field_name} must be a JSON list, JSON object, or JSON string"

    def _iter_message_handoff_candidates(self, messages) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for message in messages or []:
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if not isinstance(additional_kwargs, dict):
                continue
            handoff_id = str(additional_kwargs.get("strap_handoff_id") or "").strip()
            contract = str(additional_kwargs.get("strap_handoff_contract") or "").strip()
            if handoff_id:
                candidates.append((handoff_id, contract))
        return candidates

    def _get_optimization_handoff_payload(self, state_map: dict, args: dict) -> dict | None:
        payload = state_map.get("strap_handoff_payload")
        contract = str(state_map.get("strap_handoff_contract") or "").strip()
        if isinstance(payload, dict) and contract == "optimization.stage_candidates.v1":
            return dict(payload)

        messages = state_map.get("messages") or []
        for handoff_id, message_contract in self._iter_message_handoff_candidates(messages):
            record = get_handoff(handoff_id)
            if record is None:
                continue
            if message_contract and record.contract != message_contract:
                continue
            if record.contract == "optimization.stage_candidates.v1":
                return dict(record.payload)

        handoff_id = str(args.get("handoff_id") or state_map.get("strap_handoff_id") or "").strip()
        if not handoff_id:
            return None
        record = get_handoff(handoff_id)
        if record is None or record.contract != "optimization.stage_candidates.v1":
            return None
        return dict(record.payload)

    def _infer_optimization_feed_inputs(self, state_map: dict, handoff_payload: dict | None) -> tuple[dict | None, float | None]:
        if isinstance(handoff_payload, dict):
            composition = handoff_payload.get("feed_composition")
            capacity = handoff_payload.get("feed_capacity_tpy")
            if isinstance(composition, dict) and composition:
                parsed_capacity = self._coerce_float(capacity)
                return dict(composition), parsed_capacity
            parsed_capacity = self._coerce_float(capacity)
            if parsed_capacity is not None:
                return None, parsed_capacity

        query_text = self._latest_human_text(state_map.get("messages") or [])
        if not query_text:
            return None, None
        context = extract_query_context(query_text)
        composition = context.feed_composition or None
        return composition, context.feed_capacity_tpy

    def _infer_optimization_composition_slices(self, state_map: dict) -> list[dict[str, float]]:
        query_text = self._latest_human_text(state_map.get("messages") or [])
        if not query_text:
            return []
        context = extract_query_context(query_text)
        return [dict(item) for item in context.feed_composition_slices]

    def _latest_human_text(self, messages: list) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return coerce_message_text(message.content)
        return ""

    def _coerce_float(self, value) -> float | None:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

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

    def _missing_required_visualization_tool(self, messages: list, last_ai: AIMessage) -> str | None:
        if self._agent_name != "visualization-specialist":
            return None
        if self._state.visualization_required_tool_repairs >= 1:
            return None
        if getattr(last_ai, "tool_calls", None):
            return None
        required_tool = extract_required_visualization_tool(messages)
        if not required_tool:
            return None
        if required_tool in extract_completed_tool_names(messages):
            return None
        return required_tool

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
