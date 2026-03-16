"""Routing message-state extraction and plan-tracking helpers."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .handoffs import get_tool_call_handoff_statuses, validate_agent_payload
from .routing_classifier import (
    ROUTING_RULES,
    classify_query_keywords,
    derive_workflow_dependencies,
    order_workflow_rules,
)

_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)
_TEMPERATURE_LIMIT_PATTERNS = (
    re.compile(
        r"\b(?:up to|max(?:imum)?|at most|no more than|under|below)\s*(\d+(?:\.\d+)?)\s*°?\s*C\b",
        re.IGNORECASE,
    ),
)
_TASK_DESC_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TASK_DESC_STOPWORDS = {
    "a", "an", "and", "answer", "analyze", "analysis", "assess", "build", "calculate",
    "create", "determine", "do", "evaluate", "find", "for", "from", "generate", "give",
    "help", "make", "of", "on", "or", "please", "provide", "run", "summarize", "the",
    "to", "use", "using", "with",
}


def _is_router_guard_message(message: ToolMessage) -> bool:
    content = message.content if isinstance(message.content, str) else str(message.content)
    return content.startswith("Router guard:")


def _extract_task_dispatches(messages: list) -> list[dict]:
    """Extract task() dispatches in the order they were emitted."""
    dispatches: list[dict] = []
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "task":
                    args = tc.get("args", {})
                    subagent = args.get("subagent_type", "")
                    tool_call_id = tc.get("id", "")
                    if subagent and tool_call_id:
                        dispatches.append(
                            {
                                "tool_call_id": tool_call_id,
                                "subagent": subagent,
                                "description": args.get("description", ""),
                                "message_index": index,
                            }
                        )
    blocked_task_ids = {
        getattr(msg, "tool_call_id", None)
        for msg in messages
        if isinstance(msg, ToolMessage) and _is_router_guard_message(msg)
    }
    return [dispatch for dispatch in dispatches if dispatch["tool_call_id"] not in blocked_task_ids]


def _extract_returned_subagent_calls(messages: list) -> list[dict]:
    """Extract task() dispatches that returned a ToolMessage."""
    dispatches = _extract_task_dispatches(messages)
    if not dispatches:
        return []

    completed_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if _is_router_guard_message(msg):
                continue
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                completed_ids.add(tool_call_id)

    return [dispatch for dispatch in dispatches if dispatch["tool_call_id"] in completed_ids]


def _get_tool_call_registry(messages: list) -> dict[str, dict]:
    """Return AI tool-call metadata keyed by tool_call_id."""
    registry: dict[str, dict] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_id = tc.get("id")
                if not tool_call_id:
                    continue
                registry[tool_call_id] = {
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                    "message_index": index,
                }
    return registry


def _get_tool_message_registry(messages: list) -> dict[str, dict]:
    """Return ToolMessages keyed by tool_call_id with their message index."""
    registry: dict[str, dict] = {}
    for index, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        tool_call_id = getattr(msg, "tool_call_id", None)
        if not tool_call_id:
            continue
        registry[tool_call_id] = {"message": msg, "message_index": index}
    return registry


def _infer_handoff_status_from_tool_message(
    message: ToolMessage,
    producer: str | None = None,
) -> str:
    """Infer handoff status directly from task ToolMessage content."""
    text = message.content if isinstance(message.content, str) else str(message.content)
    match = _STRUCTURED_RESULT_RE.search(text)
    if not match:
        return "missing"

    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return "invalid"

    if not isinstance(payload, dict):
        return "invalid"

    payload_producer = producer or payload.get("agent")
    if isinstance(payload_producer, str) and payload_producer:
        if validate_agent_payload(payload_producer, payload):
            return "invalid"
    return "ok"


def _extract_structured_payload_from_tool_message(
    message: ToolMessage,
    producer: str | None = None,
) -> dict | None:
    """Decode a validated structured-result payload from a task ToolMessage."""
    text = message.content if isinstance(message.content, str) else str(message.content)
    match = _STRUCTURED_RESULT_RE.search(text)
    if not match:
        return None

    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    payload_producer = producer or payload.get("agent")
    if isinstance(payload_producer, str) and payload_producer:
        if validate_agent_payload(payload_producer, payload):
            return None

    return payload


def _get_task_handoff_statuses(messages: list) -> dict[str, str]:
    """Return task() statuses using stored handoffs or ToolMessage fallback."""
    statuses = dict(get_tool_call_handoff_statuses())
    tool_results = _get_tool_message_registry(messages)

    for dispatch in _extract_returned_subagent_calls(messages):
        tool_call_id = dispatch["tool_call_id"]
        if tool_call_id in statuses:
            continue
        tool_result = tool_results.get(tool_call_id)
        if tool_result is None:
            continue
        statuses[tool_call_id] = _infer_handoff_status_from_tool_message(
            tool_result["message"],
            producer=dispatch["subagent"],
        )

    return statuses


def _extract_completed_subagent_calls(messages: list) -> list[dict]:
    """Extract task() dispatches that produced a usable structured handoff."""
    returned = _extract_returned_subagent_calls(messages)
    if not returned:
        return []

    statuses = _get_task_handoff_statuses(messages)
    return [dispatch for dispatch in returned if statuses.get(dispatch["tool_call_id"]) == "ok"]


def _extract_failed_subagent_calls(messages: list) -> list[dict]:
    """Extract task() dispatches that returned but did not produce a usable handoff."""
    returned = _extract_returned_subagent_calls(messages)
    if not returned:
        return []

    statuses = _get_task_handoff_statuses(messages)
    return [
        dispatch for dispatch in returned
        if (status := statuses.get(dispatch["tool_call_id"])) is not None and status != "ok"
    ]


def _extract_completed_subagents(messages: list) -> list[str]:
    """Return completed subagent names, preserving duplicate task() calls."""
    return [dispatch["subagent"] for dispatch in _extract_completed_subagent_calls(messages)]


def _normalize_task_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _task_description_tokens(text: str) -> set[str]:
    normalized = _normalize_task_description(text)
    if not normalized:
        return set()
    return {
        token
        for token in _TASK_DESC_TOKEN_RE.findall(normalized)
        if token not in _TASK_DESC_STOPWORDS
    }


def _task_descriptions_match(left: str, right: str) -> bool:
    left_normalized = _normalize_task_description(left)
    right_normalized = _normalize_task_description(right)
    if not left_normalized or not right_normalized:
        return False
    if (
        left_normalized == right_normalized
        or left_normalized in right_normalized
        or right_normalized in left_normalized
    ):
        return True

    left_tokens = _task_description_tokens(left_normalized)
    right_tokens = _task_description_tokens(right_normalized)
    if not left_tokens or not right_tokens:
        return False

    overlap = len(left_tokens & right_tokens)
    minimum = min(len(left_tokens), len(right_tokens))
    return minimum > 0 and (overlap / minimum) >= 0.75


def _is_retry_of_failed_dispatch(
    failed_dispatch: dict,
    later_dispatch: dict,
    dispatches: list[dict] | None = None,
) -> bool:
    """Return True when a later dispatch is a retry of the same logical step."""
    if later_dispatch["subagent"] != failed_dispatch["subagent"]:
        return False

    if _task_descriptions_match(
        failed_dispatch.get("description", ""),
        later_dispatch.get("description", ""),
    ):
        return True

    if not dispatches:
        return False

    failed_index = failed_dispatch.get("message_index", -1)
    later_index = later_dispatch.get("message_index", -1)
    if failed_index < 0 or later_index < 0 or later_index <= failed_index:
        return False

    return not any(
        failed_index < dispatch.get("message_index", -1) < later_index
        and dispatch["subagent"] != failed_dispatch["subagent"]
        for dispatch in dispatches
    )


def _get_superseded_failed_task_ids(messages: list) -> set[str]:
    """Return failed task IDs that were later superseded by another retry attempt."""
    dispatches = _extract_task_dispatches(messages)
    if not dispatches:
        return set()

    statuses = _get_task_handoff_statuses(messages)
    superseded: set[str] = set()
    for index, dispatch in enumerate(dispatches):
        dispatch_status = statuses.get(dispatch["tool_call_id"])
        if dispatch_status not in {"missing", "invalid"}:
            continue
        for later_dispatch in dispatches[index + 1:]:
            if _is_retry_of_failed_dispatch(dispatch, later_dispatch, dispatches):
                superseded.add(dispatch["tool_call_id"])
                break
    return superseded


def _get_effective_failed_task_ids(messages: list) -> set[str]:
    """Return failed task IDs after removing superseded retry attempts."""
    failed_ids = {call["tool_call_id"] for call in _extract_failed_subagent_calls(messages)}
    return failed_ids - _get_superseded_failed_task_ids(messages)


def _get_effective_completed_task_ids(messages: list) -> set[str]:
    """Return task() IDs considered complete after subtracting failed attempts."""
    completed_ids = {call["tool_call_id"] for call in _extract_completed_subagent_calls(messages)}
    failed_ids = {call["tool_call_id"] for call in _extract_failed_subagent_calls(messages)}
    return completed_ids - failed_ids


def _get_active_remaining_steps(messages: list, ordered_plan: list[dict]) -> list[dict]:
    """Return plan steps that still need attention after retry resolution."""
    completed_ids = _get_effective_completed_task_ids(messages)
    superseded_failed_ids = _get_superseded_failed_task_ids(messages)
    resolved_ids = completed_ids | superseded_failed_ids
    return [step for step in ordered_plan if step["step_id"] not in resolved_ids]


def _extract_all_task_subagents(messages: list) -> list[str]:
    """Extract all subagent names dispatched via task(), completed or in-flight."""
    return [dispatch["subagent"] for dispatch in _extract_task_dispatches(messages)]


def _extract_user_temperature_limit_c(messages: list) -> float | None:
    limits: list[float] = []
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        for pattern in _TEMPERATURE_LIMIT_PATTERNS:
            for match in pattern.finditer(content):
                try:
                    limits.append(float(match.group(1)))
                except (TypeError, ValueError):
                    continue
    return max(limits) if limits else None


def _get_allowed_subagent_names(allowed_rules: list[dict]) -> set[str]:
    return {rule["subagent"] for rule in allowed_rules}


def _get_workflow_dependency_map(
    query_text: str,
    advisory_rules: list[dict],
) -> dict[str, tuple[str, ...]]:
    if not advisory_rules:
        return {}

    ordered_names = [rule["subagent"] for rule in advisory_rules]
    dependency_sets = derive_workflow_dependencies(query_text, set(ordered_names))
    return {
        name: tuple(dep for dep in ordered_names if dep in dependency_sets.get(name, set()))
        for name in ordered_names
    }


def _get_step_dependencies(ordered_plan: list[dict], subagent: str) -> tuple[str, ...]:
    for step in ordered_plan:
        if step["subagent"] != subagent:
            continue
        depends_on = step.get("depends_on")
        if isinstance(depends_on, tuple):
            return depends_on
        if isinstance(depends_on, list):
            return tuple(str(item) for item in depends_on if str(item).strip())
    return ()


def _get_downstream_subagents(ordered_plan: list[dict], subagent: str) -> tuple[str, ...]:
    downstream: list[str] = []
    seen: set[str] = set()
    for step in ordered_plan:
        if step["subagent"] == subagent:
            continue
        depends_on = step.get("depends_on")
        deps = depends_on if isinstance(depends_on, (list, tuple)) else ()
        if subagent in deps and step["subagent"] not in seen:
            seen.add(step["subagent"])
            downstream.append(step["subagent"])
    return tuple(downstream)


def _get_ordered_plan(messages: list, allowed_rules: list[dict] | None = None) -> list[dict]:
    """Build the ordered execution plan from actual orchestrator history."""
    rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
    advisory_rules = allowed_rules if allowed_rules is not None else classify_query_keywords(messages)
    query_text = _get_last_human_message(messages) or ""
    advisory_rules = order_workflow_rules(query_text, advisory_rules)
    dependency_map = _get_workflow_dependency_map(query_text, advisory_rules)
    allowed_names = {rule["subagent"] for rule in advisory_rules} if advisory_rules else None

    dispatched = _extract_task_dispatches(messages)
    plan: list[dict] = []
    seen_advisory: set[str] = set()
    for dispatch in dispatched:
        name = dispatch["subagent"]
        if allowed_names is not None and name not in allowed_names:
            continue
        rule = rules_by_name.get(name)
        if rule:
            plan.append({
                **rule,
                "step_id": dispatch["tool_call_id"],
                "depends_on": dependency_map.get(name, ()),
            })
            seen_advisory.add(name)

    for rule in advisory_rules:
        name = rule["subagent"]
        if name not in seen_advisory:
            plan.append({
                **rule,
                "step_id": f"advisory:{name}",
                "depends_on": dependency_map.get(name, ()),
            })
            seen_advisory.add(name)

    return plan


def _get_last_human_message(messages: list) -> str | None:
    """Extract text from the last HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


def _workflow_is_active(messages: list, allowed_rules: list[dict]) -> bool:
    allowed_names = _get_allowed_subagent_names(allowed_rules)
    if not allowed_names:
        return False
    return len(allowed_names) > 1 or bool(_extract_task_dispatches(messages))


def _get_latest_dispatch_for_subagent(
    messages: list,
    subagent: str,
    *,
    status: str | None = None,
) -> dict | None:
    statuses = _get_task_handoff_statuses(messages)
    for dispatch in reversed(_extract_task_dispatches(messages)):
        if dispatch["subagent"] != subagent:
            continue
        if status is not None and statuses.get(dispatch["tool_call_id"]) != status:
            continue
        return dispatch
    return None
