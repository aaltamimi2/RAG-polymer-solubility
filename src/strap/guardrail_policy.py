"""Request-mutation and tool-blocking policy helpers for subagent guardrails."""

from __future__ import annotations

from deepagents.middleware._utils import append_to_system_message
from langchain_core.messages import ToolMessage

from .guardrail_messages import (
    duplicate_biosteam_batch_message,
    late_separation_todo_message,
    separation_support_directive,
    separation_temperature_bound_directive,
    synthesis_directive,
    visualization_tool_block_message,
    visualization_tool_directive,
)
from .guardrail_utils import (
    extract_prior_successful_biosteam_batches,
    extract_completed_tool_names,
    extract_requested_biosteam_batch,
    extract_required_visualization_tool,
    extract_user_temperature_limit_c,
    infer_requested_polymer_support,
    tool_name,
)

_SEPARATION_NON_DOMAIN_TOOLS = {"think", "write_todos"}


def inject_synthesis_directive(request, *, synthesis_tool_seen: bool):
    if not synthesis_tool_seen or request.system_message is None:
        return request
    directive = synthesis_directive()
    new_system = append_to_system_message(request.system_message, directive)
    return request.override(system_message=new_system)


def inject_visualization_tool_directive(request, *, agent_name: str | None):
    if agent_name != "visualization-specialist" or request.system_message is None:
        return request
    required_tool = extract_required_visualization_tool(request.messages)
    if not required_tool:
        return request
    directive = visualization_tool_directive(required_tool)
    new_system = append_to_system_message(request.system_message, directive)
    return request.override(system_message=new_system)


def inject_separation_support_directive(request, *, agent_name: str | None):
    if agent_name != "separation-engineer" or request.system_message is None:
        return request

    requested, supported, unsupported = infer_requested_polymer_support(request.messages)
    if not unsupported:
        return request

    directive = separation_support_directive(requested, supported, unsupported)
    new_system = append_to_system_message(request.system_message, directive)
    return request.override(system_message=new_system)


def inject_separation_temperature_bound_directive(request, *, agent_name: str | None):
    if agent_name != "separation-engineer" or request.system_message is None:
        return request

    max_temp_c = extract_user_temperature_limit_c(request.messages)
    if max_temp_c is None:
        return request

    directive = separation_temperature_bound_directive(max_temp_c)
    new_system = append_to_system_message(request.system_message, directive)
    return request.override(system_message=new_system)


def restrict_visualization_tools(request, *, agent_name: str | None):
    if agent_name != "visualization-specialist":
        return request
    required_tool = extract_required_visualization_tool(request.messages)
    if not required_tool:
        return request

    allowed_names = {required_tool, "think"}
    filtered_tools = [tool for tool in request.tools if tool_name(tool) in allowed_names]
    if not filtered_tools:
        return request
    if required_tool not in {tool_name(tool) for tool in filtered_tools}:
        return request
    return request.override(tools=filtered_tools)


def maybe_block_duplicate_biosteam_batch(request, *, agent_name: str | None) -> ToolMessage | None:
    tool_call = getattr(request, "tool_call", {}) or {}
    if agent_name != "biosteam-analyst":
        return None
    if tool_call.get("name") != "run_biosteam_multi_polymer":
        return None

    batch = extract_requested_biosteam_batch(tool_call.get("args", {}))
    if batch is None:
        return None

    state = getattr(request, "state", {}) or {}
    messages = state.get("messages", []) if isinstance(state, dict) else []
    for prior_batch in extract_prior_successful_biosteam_batches(messages):
        if prior_batch["energy_case"] != batch["energy_case"]:
            continue
        if prior_batch["allocation_method"] != batch["allocation_method"]:
            continue

        overlap = sorted(batch["solvents"] & prior_batch["solvents"])
        if not overlap:
            continue

        fresh = sorted(batch["solvents"] - prior_batch["solvents"])
        return ToolMessage(
            content=duplicate_biosteam_batch_message(
                energy_case=batch["energy_case"],
                allocation_method=batch["allocation_method"],
                overlap=overlap,
                fresh=fresh,
            ),
            tool_call_id=tool_call.get("id", ""),
            name="run_biosteam_multi_polymer",
            status="error",
        )

    return None


def maybe_enforce_visualization_tool_directive(request, *, agent_name: str | None) -> ToolMessage | None:
    tool_call = getattr(request, "tool_call", {}) or {}
    if agent_name != "visualization-specialist":
        return None

    called_tool_name = tool_call.get("name")
    if not called_tool_name or called_tool_name == "think":
        return None

    state = getattr(request, "state", {}) or {}
    messages = state.get("messages", []) if isinstance(state, dict) else []
    required_tool = extract_required_visualization_tool(messages)
    if not required_tool or called_tool_name == required_tool:
        return None

    return ToolMessage(
        content=visualization_tool_block_message(required_tool, called_tool_name),
        tool_call_id=tool_call.get("id", ""),
        name=called_tool_name,
        status="error",
    )


def maybe_block_late_separation_todos(request, *, agent_name: str | None) -> ToolMessage | None:
    tool_call = getattr(request, "tool_call", {}) or {}
    if agent_name != "separation-engineer":
        return None
    if tool_call.get("name") != "write_todos":
        return None

    state = getattr(request, "state", {}) or {}
    messages = state.get("messages", []) if isinstance(state, dict) else []
    prior_tools = [
        name for name in extract_completed_tool_names(messages)
        if name not in _SEPARATION_NON_DOMAIN_TOOLS
    ]
    if not prior_tools:
        return None

    return ToolMessage(
        content=late_separation_todo_message(),
        tool_call_id=tool_call.get("id", ""),
        name="write_todos",
        status="error",
    )
