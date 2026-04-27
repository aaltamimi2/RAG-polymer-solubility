"""Convert DISSOLVE YAML subagent specs to Claude SDK AgentDefinition objects."""

from __future__ import annotations

from typing import Any

from strap.prompts import FILE_IO_DIRECTIVE, THINK_DIRECTIVE
from strap.subagent_config import load_subagent_specs

from .tool_catalog import ToolNameMap

_GROUP_TO_LEGACY_TOOLS: dict[str, tuple[str, ...]] = {
    "database_query": ("list_available_solvents", "list_available_polymers"),
    "interpolation": ("predict_solubility", "predict_solubility_range"),
    "solvent_lookup": ("list_available_solvents", "list_available_polymers"),
    "visualization": ("plot_solubility_vs_temperature", "plot_optimization_pareto_front"),
    "safety_card": ("get_solvent_safety_card", "compare_solvent_safety_cards"),
    "safety_gsk": ("get_solvent_safety_card", "compare_solvent_safety_cards"),
    "safety_pubchem": ("get_solvent_safety_card", "compare_solvent_safety_cards"),
    # Real DISSOLVE groups that are intentionally not exposed in the first SDK slice.
    "adaptive_separation": (),
    "biosteam": (),
    "contaminant_removal": (),
    "ml_prediction": (),
    "patent": (),
    "rag_core": (),
    "rag_diagnostics": (),
    "reflection": (),
    "result_extractor": (),
    "scholar": (),
    "separation_core": (),
    "separation_plot": (),
    "sidecar_read": (),
    "sidecar_write": (),
    "statistical": (),
    "thermal_prediction": (),
    "waste_optimization": (
        "run_waste_management_optimization",
        "run_waste_management_pareto",
        "plot_optimization_pareto_front",
    ),
}


def _tools_for_groups(spec: dict[str, Any], tool_map: ToolNameMap) -> list[str]:
    tools: list[str] = []
    for name in spec.get("tools") or []:
        if name == "Agent":
            continue
        try:
            tools.append(tool_map.mcp_name(str(name)))
        except KeyError:
            continue
    for group in spec.get("tool_groups") or []:
        group_name = str(group)
        if group_name not in _GROUP_TO_LEGACY_TOOLS:
            raise ValueError(f"Unknown Claude SDK subagent tool group: {group_name}")
        for legacy in _GROUP_TO_LEGACY_TOOLS[group_name]:
            tools.append(tool_map.mcp_name(legacy))
    deduped: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        if tool in seen:
            continue
        seen.add(tool)
        deduped.append(tool)
    return deduped


def deferred_tool_groups(spec: dict[str, Any]) -> list[str]:
    """Return configured groups known but deferred in this SDK slice."""
    groups: list[str] = []
    for group in spec.get("tool_groups") or []:
        group_name = str(group)
        if group_name not in _GROUP_TO_LEGACY_TOOLS:
            raise ValueError(f"Unknown Claude SDK subagent tool group: {group_name}")
        if not _GROUP_TO_LEGACY_TOOLS[group_name]:
            groups.append(group_name)
    return groups


def build_agent_definitions(tool_map: ToolNameMap | None = None) -> dict[str, Any]:
    """Build SDK AgentDefinition objects, or plain dicts if SDK is absent."""
    tool_map = tool_map or ToolNameMap()
    try:
        from claude_agent_sdk import AgentDefinition
    except Exception:
        AgentDefinition = None  # type: ignore[assignment]

    agents: dict[str, Any] = {}
    for spec in load_subagent_specs():
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        deferred = deferred_tool_groups(spec)
        prompt = str(spec.get("system_prompt") or "").rstrip() + FILE_IO_DIRECTIVE + THINK_DIRECTIVE
        if deferred:
            prompt += (
                "\n\nClaude SDK migration note: the following DISSOLVE tool groups are "
                "explicitly deferred in this harness slice and are not available as MCP tools: "
                + ", ".join(sorted(deferred))
                + ". Do not claim you used those tools."
            )
        description = str(spec.get("description") or "").strip()
        if deferred:
            description += f" Deferred SDK groups: {', '.join(sorted(deferred))}."
        payload = {
            "description": description,
            "prompt": prompt,
            "tools": _tools_for_groups(spec, tool_map),
            "model": "inherit",
        }
        agents[name] = AgentDefinition(**payload) if AgentDefinition is not None else payload
    return agents
