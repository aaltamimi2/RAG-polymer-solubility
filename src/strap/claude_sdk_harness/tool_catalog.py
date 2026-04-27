"""Tool-name mapping between DISSOLVE callables and Claude MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    legacy_name: str
    server_name: str
    description: str
    read_only: bool = True

    @property
    def mcp_name(self) -> str:
        return f"mcp__{self.server_name}__{self.legacy_name}"


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("list_available_solvents", "dissolve_solubility", "List solvent records by polymer."),
    ToolSpec("list_available_polymers", "dissolve_solubility", "List available polymers."),
    ToolSpec("predict_solubility", "dissolve_solubility", "Predict one polymer-solvent solubility point."),
    ToolSpec("predict_solubility_range", "dissolve_solubility", "Predict solubility over a temperature range."),
    ToolSpec(
        "plot_solubility_vs_temperature",
        "dissolve_solubility",
        "Create a solubility-vs-temperature plot.",
        read_only=False,
    ),
    ToolSpec("get_solvent_safety_card", "dissolve_safety", "Render one solvent safety card."),
    ToolSpec("compare_solvent_safety_cards", "dissolve_safety", "Compare several solvent safety cards."),
    ToolSpec(
        "run_waste_management_optimization",
        "dissolve_optimization",
        "Run point-optimum waste-management optimization.",
    ),
    ToolSpec(
        "run_waste_management_pareto",
        "dissolve_optimization",
        "Run a waste-management Pareto sweep.",
        read_only=False,
    ),
    ToolSpec(
        "plot_optimization_pareto_front",
        "dissolve_optimization",
        "Create a Pareto-front plot from optimizer output.",
        read_only=False,
    ),
)


class ToolNameMap:
    """Central adapter between legacy callable names and SDK MCP names."""

    version = "1"

    def __init__(self, specs: tuple[ToolSpec, ...] = _TOOL_SPECS) -> None:
        self._by_legacy = {spec.legacy_name: spec for spec in specs}
        self._by_mcp = {spec.mcp_name: spec for spec in specs}

    def mcp_name(self, legacy_name: str) -> str:
        spec = self._by_legacy.get(legacy_name)
        if spec is None:
            raise KeyError(f"No Claude MCP mapping registered for DISSOLVE tool '{legacy_name}'.")
        return spec.mcp_name

    def legacy_name(self, mcp_name: str) -> str:
        spec = self._by_mcp.get(mcp_name)
        if spec is None:
            raise KeyError(f"No DISSOLVE tool mapping registered for Claude MCP tool '{mcp_name}'.")
        return spec.legacy_name

    def spec_for_legacy(self, legacy_name: str) -> ToolSpec:
        spec = self._by_legacy.get(legacy_name)
        if spec is None:
            raise KeyError(f"No Claude MCP mapping registered for DISSOLVE tool '{legacy_name}'.")
        return spec

    def allowed_for_legacy(self, legacy_name: str) -> list[str]:
        return [self.mcp_name(legacy_name)]

    def allowed_for_intent(self, intent: str, plan_step: object | None = None) -> list[str]:
        if plan_step is not None:
            names = list(getattr(plan_step, "allowed_tools", []) or [])
            if not names and isinstance(plan_step, dict):
                names = list(plan_step.get("allowed_tools") or [])
            return [self.mcp_name(name) for name in names]

        intent_key = (intent or "").strip().lower()
        if intent_key in {"solubility_lookup", "solvent_candidate_lookup", "direct_answer"}:
            return [
                self.mcp_name("list_available_solvents"),
                self.mcp_name("predict_solubility"),
                self.mcp_name("predict_solubility_range"),
            ]
        if intent_key in {"solubility_plot", "artifact_transform", "plot"}:
            return [
                self.mcp_name("predict_solubility_range"),
                self.mcp_name("plot_solubility_vs_temperature"),
            ]
        if intent_key == "safety_lookup":
            return [
                self.mcp_name("get_solvent_safety_card"),
                self.mcp_name("compare_solvent_safety_cards"),
            ]
        if intent_key == "optimization":
            return [
                self.mcp_name("run_waste_management_optimization"),
                self.mcp_name("run_waste_management_pareto"),
                self.mcp_name("plot_optimization_pareto_front"),
            ]
        if intent_key in {"complex_workflow", "subagent"}:
            return [spec.mcp_name for spec in self._by_legacy.values()] + ["Agent"]
        return [spec.mcp_name for spec in self._by_legacy.values() if spec.server_name in {"dissolve_solubility", "dissolve_safety"}]

    def specs_for_server(self, server_name: str) -> list[ToolSpec]:
        return [spec for spec in self._by_legacy.values() if spec.server_name == server_name]

    def legacy_names(self) -> list[str]:
        return sorted(self._by_legacy)

    def mcp_names(self) -> list[str]:
        return sorted(self._by_mcp)


def infer_intent(query: str) -> str:
    """Small deterministic intent hint for SDK tool scoping."""
    text = str(query or "").lower()
    if any(
        word in text
        for word in (
            "optimize",
            "optimization",
            "pareto",
            "profit",
            "circularity",
            "waste management",
            "wash step",
            "strap wash",
            "scenario a",
            "scenario b",
            "scenario c",
        )
    ):
        return "optimization"
    if any(word in text for word in ("safety", "toxicity", "flash point", "ld50", "peroxide")):
        return "safety_lookup"
    if any(word in text for word in ("plot", "chart", "graph", "visualiz")) and "solubility" in text:
        return "solubility_plot"
    if "solubility" in text or "dissolv" in text or "solvent" in text:
        return "solubility_lookup"
    return "direct_answer"


def fingerprint_allowed_tools(allowed_tools: list[str]) -> str:
    import hashlib

    payload = "\n".join(sorted(set(allowed_tools))).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def normalize_tool_call_names(tool_calls: list[dict[str, Any] | str], tool_map: ToolNameMap | None = None) -> list[str]:
    tool_map = tool_map or ToolNameMap()
    names: list[str] = []
    for call in tool_calls:
        if isinstance(call, str):
            name = call
        else:
            name = str(call.get("name") or "")
        if not name:
            continue
        try:
            names.append(tool_map.legacy_name(name))
        except KeyError:
            names.append(name)
    return names
