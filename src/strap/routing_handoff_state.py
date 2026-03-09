"""Routing helpers for handoff readiness and sequential dispatch state."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from .routing_classifier import SEQUENTIAL_PAIRS
from .routing_message_state import (
    _extract_returned_subagent_calls,
    _get_active_remaining_steps,
    _get_latest_dispatch_for_subagent,
    _get_ordered_plan,
    _get_tool_call_registry,
    _get_tool_message_registry,
)


def _parse_tool_json_content(message: ToolMessage) -> dict | None:
    content = message.content if isinstance(message.content, str) else str(message.content)
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _has_built_handoff_since(
    messages: list,
    *,
    producer: str,
    consumer: str,
    after_task_call_id: str,
) -> bool:
    """Return True when a successful build_handoff exists after a producer result."""
    tool_calls = _get_tool_call_registry(messages)
    tool_messages = _get_tool_message_registry(messages)
    source_result = tool_messages.get(after_task_call_id)
    if source_result is None:
        return False

    source_index = source_result["message_index"]
    for tool_call_id, tool_call_meta in tool_calls.items():
        if tool_call_meta.get("name") != "build_handoff":
            continue
        tool_result = tool_messages.get(tool_call_id)
        if tool_result is None or tool_result["message_index"] <= source_index:
            continue
        payload = _parse_tool_json_content(tool_result["message"])
        if not payload or payload.get("ok") is not True:
            continue
        handoff = payload.get("handoff", {})
        if (
            handoff.get("producer") == producer
            and handoff.get("consumer") == consumer
            and handoff.get("status") == "ok"
        ):
            return True
    return False


def _get_built_handoff_result_since(
    messages: list,
    *,
    producer: str,
    consumer: str,
    after_task_call_id: str,
) -> tuple[dict, int] | None:
    """Return the latest successful build_handoff payload and its message index."""
    tool_calls = _get_tool_call_registry(messages)
    tool_messages = _get_tool_message_registry(messages)
    source_result = tool_messages.get(after_task_call_id)
    if source_result is None:
        return None

    source_index = source_result["message_index"]
    latest_handoff: dict | None = None
    latest_index = -1
    for tool_call_id, tool_call_meta in tool_calls.items():
        if tool_call_meta.get("name") != "build_handoff":
            continue
        tool_result = tool_messages.get(tool_call_id)
        if tool_result is None or tool_result["message_index"] <= source_index:
            continue
        payload = _parse_tool_json_content(tool_result["message"])
        if not payload or payload.get("ok") is not True:
            continue
        handoff = payload.get("handoff", {})
        if (
            handoff.get("producer") == producer
            and handoff.get("consumer") == consumer
            and handoff.get("status") == "ok"
        ):
            latest_handoff = handoff
            latest_index = tool_result["message_index"]

    if latest_handoff is None:
        return None
    return latest_handoff, latest_index


def _get_built_handoff_since(
    messages: list,
    *,
    producer: str,
    consumer: str,
    after_task_call_id: str,
) -> dict | None:
    """Return the most recent successful build_handoff payload after a producer result."""
    result = _get_built_handoff_result_since(
        messages,
        producer=producer,
        consumer=consumer,
        after_task_call_id=after_task_call_id,
    )
    if result is None:
        return None
    return result[0]


def _has_returned_subagent_since(
    messages: list,
    *,
    subagent: str,
    after_message_index: int,
) -> bool:
    """Return True when a subagent task has already returned after a handoff was built."""
    for dispatch in _extract_returned_subagent_calls(messages):
        if dispatch["subagent"] != subagent:
            continue
        if dispatch["message_index"] > after_message_index:
            return True
    return False


def _get_pending_required_handoff(
    messages: list,
    allowed_rules: list[dict],
) -> tuple[str, str] | None:
    """Return the next required producer->consumer handoff, if one is pending."""
    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if not ordered_plan:
        return None

    remaining = _get_active_remaining_steps(messages, ordered_plan)
    if not remaining:
        return None

    next_name = remaining[0]["subagent"]
    plan_names = {step["subagent"] for step in ordered_plan}
    predecessors = [
        producer for producer, consumer in SEQUENTIAL_PAIRS
        if consumer == next_name and producer in plan_names
    ]
    successful_predecessors = [
        dispatch
        for producer in predecessors
        if (dispatch := _get_latest_dispatch_for_subagent(messages, producer, status="ok")) is not None
    ]
    if not successful_predecessors:
        return None

    tool_messages = _get_tool_message_registry(messages)
    latest_predecessor = max(
        successful_predecessors,
        key=lambda dispatch: tool_messages.get(dispatch["tool_call_id"], {}).get("message_index", -1),
    )
    if _has_built_handoff_since(
        messages,
        producer=latest_predecessor["subagent"],
        consumer=next_name,
        after_task_call_id=latest_predecessor["tool_call_id"],
    ):
        return None

    return latest_predecessor["subagent"], next_name


def _get_ready_downstream_consumer(
    messages: list,
    allowed_rules: list[dict],
) -> str | None:
    """Return the next downstream consumer once its handoff has already been built."""
    handoff = _get_ready_downstream_handoff(messages, allowed_rules)
    if handoff is None:
        return None
    return handoff.get("consumer")


def _get_ready_downstream_handoff(
    messages: list,
    allowed_rules: list[dict],
) -> dict | None:
    """Return the next downstream handoff once it has already been built."""
    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if not ordered_plan:
        return None

    remaining = _get_active_remaining_steps(messages, ordered_plan)
    if not remaining:
        return None

    next_name = remaining[0]["subagent"]
    plan_names = {step["subagent"] for step in ordered_plan}
    predecessors = [
        producer for producer, consumer in SEQUENTIAL_PAIRS
        if consumer == next_name and producer in plan_names
    ]
    successful_predecessors = [
        dispatch
        for producer in predecessors
        if (dispatch := _get_latest_dispatch_for_subagent(messages, producer, status="ok")) is not None
    ]
    if not successful_predecessors:
        return None

    tool_messages = _get_tool_message_registry(messages)
    latest_predecessor = max(
        successful_predecessors,
        key=lambda dispatch: tool_messages.get(dispatch["tool_call_id"], {}).get("message_index", -1),
    )
    handoff_result = _get_built_handoff_result_since(
        messages,
        producer=latest_predecessor["subagent"],
        consumer=next_name,
        after_task_call_id=latest_predecessor["tool_call_id"],
    )
    if handoff_result is None:
        return None

    handoff, handoff_index = handoff_result
    if _has_returned_subagent_since(
        messages,
        subagent=next_name,
        after_message_index=handoff_index,
    ):
        return None
    return handoff
