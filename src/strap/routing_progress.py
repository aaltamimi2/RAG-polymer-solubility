"""Routing progress directives plus compatibility imports for routing state helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from .routing_handoff_state import (
    _get_built_handoff_result_since,
    _get_built_handoff_since,
    _get_missing_required_handoffs_for_consumer,
    _get_pending_required_handoff,
    _get_ready_downstream_consumer,
    _get_ready_downstream_handoff,
    _has_built_handoff_since,
    _has_returned_subagent_since,
    _parse_tool_json_content,
)
from .routing_message_state import (
    _extract_all_task_subagents,
    _extract_completed_subagent_calls,
    _extract_completed_subagents,
    _extract_failed_subagent_calls,
    _extract_returned_subagent_calls,
    _extract_structured_payload_from_tool_message,
    _extract_task_dispatches,
    _extract_task_dispatches,
    _extract_user_temperature_limit_c,
    _get_active_remaining_steps,
    _get_allowed_subagent_names,
    _get_downstream_subagents,
    _get_step_dependencies,
    _get_effective_completed_task_ids,
    _get_effective_failed_task_ids,
    _get_last_human_message,
    _get_latest_dispatch_for_subagent,
    _get_ordered_plan,
    _get_superseded_failed_task_ids,
    _get_task_handoff_statuses,
    _get_tool_call_registry,
    _get_tool_message_registry,
    _is_retry_of_failed_dispatch,
    _is_router_guard_message,
    _normalize_task_description,
    _task_descriptions_match,
    _workflow_is_active,
)
from .solubility import get_boiling_point

_VALID_TODO_STATUSES = {"pending", "in_progress", "completed"}
_BOILING_POINT_CAVEAT_MARGIN_C = 10.0
_NEAR_BOILING_MARGIN_C = 5.0


def _get_latest_validated_task_payload(messages: list, subagent: str) -> dict | None:
    dispatch = _get_latest_dispatch_for_subagent(messages, subagent, status="ok")
    if dispatch is None:
        return None
    tool_messages = _get_tool_message_registry(messages)
    tool_result = tool_messages.get(dispatch["tool_call_id"])
    if tool_result is None:
        return None
    return _extract_structured_payload_from_tool_message(
        tool_result["message"],
        producer=subagent,
    )


def _build_completion_synthesis_anchor(messages: list, ordered_plan: list[dict]) -> str | None:
    completed_ids = _get_effective_completed_task_ids(messages)
    completed_names = {
        step["subagent"]
        for step in ordered_plan
        if step["step_id"] in completed_ids
    }
    if not completed_names:
        return None

    anchor_parts: list[str] = [
        "Use only validated subagent outputs in the final answer. "
        "Do not invent new solvents, temperatures, selectivity values, or purity claims."
    ]

    if "separation-engineer" in completed_names:
        payload = _get_latest_validated_task_payload(messages, "separation-engineer")
        if isinstance(payload, dict):
            supported = payload.get("supported_polymers")
            unsupported = payload.get("unsupported_polymers")
            if isinstance(supported, list) and isinstance(unsupported, list) and unsupported:
                supported_text = ", ".join(str(item) for item in supported if str(item).strip()) or "none"
                unsupported_text = ", ".join(str(item) for item in unsupported if str(item).strip())
                anchor_parts.append(
                    f"Scope separation conclusions to the supported subset ({supported_text}); "
                    f"treat these polymers as unsupported: {unsupported_text}."
                )
                anchor_parts.append(
                    f"For unsupported polymers ({unsupported_text}), state that their phase behavior is "
                    "unknown without additional data. Do not describe them as dissolved, insoluble, "
                    "remaining in the residue, or purified."
                )

            steps = payload.get("steps")
            user_max_temp_c = _extract_user_temperature_limit_c(messages)
            if isinstance(steps, list):
                for index, step in enumerate(steps, start=1):
                    if not isinstance(step, dict):
                        continue
                    solvent = str(step.get("solvent", "")).strip()
                    polymer = str(step.get("polymer", "")).strip()
                    if not solvent:
                        continue
                    temp_value = step.get("temperature_c", step.get("temp"))
                    try:
                        operating_temp_c = float(temp_value)
                    except (TypeError, ValueError):
                        continue

                    step_no = step.get("step", index)
                    anchor_parts.append(
                        f"Preserve validated Step {step_no}: {polymer or 'target polymer'} with "
                        f"{solvent} at {operating_temp_c:.1f}C."
                    )

                    boiling_point_c = get_boiling_point(solvent)
                    if boiling_point_c is None:
                        continue

                    if user_max_temp_c is not None and user_max_temp_c >= boiling_point_c - _BOILING_POINT_CAVEAT_MARGIN_C:
                        anchor_parts.append(
                            f"Because the user allows up to {user_max_temp_c:.1f}C but {solvent} boils at "
                            f"{boiling_point_c:.1f}C, explicitly say the actual operating temperature is "
                            f"{operating_temp_c:.1f}C and remains below the boiling point at 1 atm."
                        )

                    bp_margin_c = boiling_point_c - operating_temp_c
                    if 0 < bp_margin_c <= _NEAR_BOILING_MARGIN_C:
                        anchor_parts.append(
                            f"Explicitly state that {solvent} at {operating_temp_c:.1f}C is only "
                            f"{bp_margin_c:.1f}C below its {boiling_point_c:.1f}C boiling point, so this "
                            "is a narrow atmospheric-pressure operating margin that requires careful "
                            "temperature control."
                        )

    return " ".join(anchor_parts) if anchor_parts else None


def _build_progress_directive(
    messages: list,
    completed_ids: set[str],
    ordered_plan: list[dict],
    failed_ids: set[str] | None = None,
) -> str | None:
    """Build a progress-tracking directive for multi-agent sequential plans."""
    superseded_failed_ids = _get_superseded_failed_task_ids(messages)
    failed_ids = (failed_ids or set()) - superseded_failed_ids
    completed_ids = set(completed_ids) - set(failed_ids)
    resolved_ids = set(completed_ids) | superseded_failed_ids
    remaining = [step for step in ordered_plan if step["step_id"] not in resolved_ids]
    completed_names = [step["subagent"] for step in ordered_plan if step["step_id"] in completed_ids]
    failed_names = [step["subagent"] for step in ordered_plan if step["step_id"] in failed_ids]
    done_names = ", ".join(completed_names) if completed_names else "(none)"
    failed_text = ", ".join(failed_names) if failed_names else "(none)"
    failure_note = ""
    if failed_names:
        failure_note = (
            f" Failed subagents: {failed_text}. "
            "These steps returned without a usable structured handoff. "
            "Retry them or request a valid <STRUCTURED_RESULT> before treating them as complete."
        )

    if not remaining:
        synthesis_anchor = _build_completion_synthesis_anchor(messages, ordered_plan)
        anchor_text = f" {synthesis_anchor}" if synthesis_anchor else ""
        return (
            "\n\n[PROGRESS: All subagent steps are complete. "
            "Write the final answer now using the completed subagent outputs. "
            "Do not call any more tools unless the user explicitly asks for more analysis. "
            f"{anchor_text}"
            "Do not call additional task() functions unless the user explicitly requests "
            "a new specialist or the new task is materially different.]"
        )

    next_agent = remaining[0]
    next_name = next_agent["subagent"]
    progress_allowed_rules = [
        {
            "subagent": step["subagent"],
            "description": step.get("description", ""),
        }
        for step in ordered_plan
    ]
    ready_handoff = _get_ready_downstream_handoff(messages, progress_allowed_rules)
    if ready_handoff is not None:
        task_prompt = ready_handoff.get("task_prompt") or ""
        prompt_suffix = f', description="{task_prompt}"' if task_prompt else ""
        return (
            f"\n\n[PROGRESS: Completed subagents: {done_names}. "
            f"Next required step: call "
            f'task(subagent_type="{next_name}"{prompt_suffix}) now. '
            "Do not call other orchestrator tools first. "
            f"Remaining steps: {', '.join(step['subagent'] for step in remaining)}."
            f"{failure_note}]"
        )

    pending_handoff = _get_pending_required_handoff(messages, progress_allowed_rules)
    if pending_handoff is not None:
        producer, consumer = pending_handoff
        return (
            f"\n\n[PROGRESS: Completed subagents: {done_names}. "
            f"Next required step: call "
            f'build_handoff(consumer="{consumer}", producer="{producer}") '
            f"or build from a specific source_handoff_id. "
            f'Do not dispatch task(subagent_type="{consumer}") until all required upstream handoffs exist. '
            f"Remaining steps: {', '.join(step['subagent'] for step in remaining)}."
            f"{failure_note}]"
        )

    return (
        f"\n\n[PROGRESS: Completed subagents: {done_names}. "
        f'Suggested next: task(subagent_type="{next_name}") '
        f"for {next_agent['description']}. "
        f"Do not repeat a completed step unless the new task is materially different. "
        f"Remaining steps: {', '.join(step['subagent'] for step in remaining)}."
        f"{failure_note}]"
    )


def _has_active_progress(messages: list) -> bool:
    """Return True once task-return/progress tracking has started."""
    return bool(_extract_returned_subagent_calls(messages)) and bool(_get_ordered_plan(messages))


def _should_block_write_todos(messages: list, allowed_rules: list[dict] | None = None) -> bool:
    """Suppress todo rewrites once orchestrator progress tracking is active."""
    if _has_active_progress(messages):
        return True
    allowed_names = _get_allowed_subagent_names(allowed_rules or [])
    return len(allowed_names) > 1 and not _extract_task_dispatches(messages)


def _validate_write_todos_call(tool_call: dict) -> str | None:
    """Return a validation error message for malformed write_todos args."""
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return "`write_todos` arguments must be a JSON object."

    todos = args.get("todos")
    if not isinstance(todos, list):
        return "`write_todos` requires a `todos` list."

    for index, todo in enumerate(todos, start=1):
        if not isinstance(todo, dict):
            return f"`write_todos` item {index} must be an object with `content` and `status`."

        content = todo.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"`write_todos` item {index} is missing a non-empty `content` field."

        status = todo.get("status")
        if status not in _VALID_TODO_STATUSES:
            return (
                f"`write_todos` item {index} has invalid status `{status}`. "
                "Allowed values are pending, in_progress, completed."
            )

    return None


def _build_write_todos_guard_messages(
    messages: list,
    allowed_rules: list[dict] | None = None,
) -> list[ToolMessage]:
    """Create error ToolMessages for redundant write_todos calls."""
    last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if not last_ai_msg or not getattr(last_ai_msg, "tool_calls", None):
        return []

    write_todos_calls = [tc for tc in last_ai_msg.tool_calls if tc.get("name") == "write_todos"]
    if not write_todos_calls:
        return []

    guard_messages: list[ToolMessage] = []
    progress_active = _has_active_progress(messages)
    route_start_block = _should_block_write_todos(messages, allowed_rules) and not progress_active
    allowed_names = sorted(_get_allowed_subagent_names(allowed_rules or []))
    for tool_call in write_todos_calls:
        if progress_active:
            message = (
                "Router guard: `write_todos` is disabled once subagent progress tracking is active. "
                "Use the existing progress directive and continue with `task(...)`, "
                "`get_subagent_result(...)`, `list_handoffs(...)`, or direct analysis tools instead."
            )
        elif route_start_block:
            if allowed_names:
                specialists = ", ".join(allowed_names)
                message = (
                    "Router guard: `write_todos` is disabled before the first specialist dispatch "
                    f"for this routed multi-specialist workflow. Dispatch `task(...)` for the "
                    f"classifier-selected specialists now: {specialists}."
                )
            else:
                message = (
                    "Router guard: `write_todos` is disabled before the first specialist dispatch "
                    "for this routed multi-specialist workflow. Dispatch the required `task(...)` "
                    "calls now."
                )
        else:
            validation_error = _validate_write_todos_call(tool_call)
            if validation_error is None:
                continue
            message = (
                f"Router guard: {validation_error} "
                "Retry `write_todos` with valid todo items shaped as "
                '{"content": "...", "status": "pending|in_progress|completed"}.'
            )

        guard_messages.append(
            ToolMessage(
                content=message,
                tool_call_id=tool_call["id"],
                status="error",
            )
        )

    return guard_messages
