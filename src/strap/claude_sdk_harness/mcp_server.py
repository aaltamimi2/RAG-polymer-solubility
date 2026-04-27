"""In-process MCP wrappers for Claude Agent SDK mode."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from strap.services.tool_response_service import json_tool_error

from .tool_catalog import ToolNameMap


class ClaudeSdkUnavailableError(RuntimeError):
    """Raised when SDK-only functionality is requested without the dependency."""


def import_claude_sdk() -> dict[str, Any]:
    try:
        from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool
    except Exception as exc:  # pragma: no cover - exercised by import-guard tests
        raise ClaudeSdkUnavailableError(
            "claude-agent-sdk is not installed. Install with `pip install -e .[claude]` "
            "or use the default LangChain harness."
        ) from exc
    return {
        "ToolAnnotations": ToolAnnotations,
        "create_sdk_mcp_server": create_sdk_mcp_server,
        "tool": tool,
    }


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _json_like_arg(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            return json.loads(text)
    return value


def _mcp_text_result(raw: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": raw}]}
    if is_error:
        result["is_error"] = True
    return result


def _tool_success(raw: str) -> bool:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return True
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if isinstance(data, dict) and "success" in data:
        return bool(data["success"])
    return True


async def call_legacy_tool(legacy_name: str, args: dict[str, Any]) -> str:
    """Call one wrapped DISSOLVE tool and return its JSON envelope string."""
    args = dict(args or {})
    if legacy_name == "list_available_solvents":
        from strap.tools.listing import list_available_solvents

        return await _await_if_needed(
            list_available_solvents(
                include_properties=bool(args.get("include_properties", False)),
                polymer=args.get("polymer"),
                limit=int(args.get("limit", 12)),
            )
        )
    if legacy_name == "list_available_polymers":
        from strap.tools.listing import list_available_polymers

        return await _await_if_needed(list_available_polymers())
    if legacy_name == "predict_solubility":
        from strap.tools.interpolation import predict_solubility

        return await _await_if_needed(
            predict_solubility(
                polymer_name=str(args["polymer_name"]),
                solvent_name=str(args["solvent_name"]),
                temperature_c=float(args.get("temperature_c", 25.0)),
            )
        )
    if legacy_name == "predict_solubility_range":
        from strap.tools.interpolation import predict_solubility_range

        return await _await_if_needed(
            predict_solubility_range(
                polymer_name=str(args["polymer_name"]),
                solvent_name=str(args["solvent_name"]),
                t_start_c=float(args.get("t_start_c", 25.0)),
                t_end_c=float(args.get("t_end_c", 160.0)),
                t_step_c=float(args.get("t_step_c", 5.0)),
            )
        )
    if legacy_name == "plot_solubility_vs_temperature":
        from strap.tools.visualization import plot_solubility_vs_temperature

        return await _await_if_needed(
            plot_solubility_vs_temperature(
                table_name="solubility_data",
                polymer_column="polymer",
                solvent_column="solvent",
                temperature_column="temperature_c",
                solubility_column="solubility_percentage",
                polymers=str(args["polymers"]),
                solvents=str(args["solvents"]),
                plot_title=args.get("plot_title"),
                include_confidence_bands=bool(args.get("include_confidence_bands", True)),
                annotate_model_limits=bool(args.get("annotate_model_limits", False)),
                temperature_min=_optional_float(args.get("temperature_min")),
                temperature_max=_optional_float(args.get("temperature_max")),
                y_axis_max=_optional_float(args.get("y_axis_max")) if args.get("y_axis_max") is not None else 100.0,
                y_axis_ranges_json=args.get("y_axis_ranges_json"),
                output_dir=args.get("output_dir"),
                output_path=args.get("output_path"),
            )
        )
    if legacy_name == "get_solvent_safety_card":
        from strap.tools.safety_card import get_solvent_safety_card

        return await _await_if_needed(
            get_solvent_safety_card(
                solvent_name=str(args["solvent_name"]),
                operating_temp_c=args.get("operating_temp_c"),
                include_pubchem=bool(args.get("include_pubchem", True)),
            )
        )
    if legacy_name == "compare_solvent_safety_cards":
        from strap.tools.safety_card import compare_solvent_safety_cards

        return await _await_if_needed(
            compare_solvent_safety_cards(
                solvent_names=str(args["solvent_names"]),
                operating_temp_c=args.get("operating_temp_c"),
                include_pubchem=bool(args.get("include_pubchem", True)),
                limit=int(args.get("limit", 6)),
            )
        )
    if legacy_name == "run_waste_management_optimization":
        from strap.tools.waste_optimization import run_waste_management_optimization

        return await _await_if_needed(
            run_waste_management_optimization(
                feed=float(args["feed"]),
                pe_fraction=_optional_float(args.get("pe_fraction")),
                pet_fraction=_optional_float(args.get("pet_fraction")),
                n6_fraction=_optional_float(args.get("n6_fraction")),
                evoh_fraction=_optional_float(args.get("evoh_fraction")),
                feed_composition_json=_json_like_arg(args.get("feed_composition_json")),
                scenario=str(args.get("scenario", "A")),
                objective=str(args.get("objective", "max_profit")),
                candidate_solvents=_json_like_arg(args.get("candidate_solvents")),
                polymer_solvent_filters_json=_json_like_arg(args.get("polymer_solvent_filters_json")),
                stage_candidates_json=_json_like_arg(args.get("stage_candidates_json")),
                constraint_mode=args.get("constraint_mode"),
                fallback_policy=args.get("fallback_policy"),
                feed_mode=str(args.get("feed_mode", "fixed")),
                composition_constraints_json=_json_like_arg(args.get("composition_constraints_json")),
                composition_step=float(args.get("composition_step", 0.1)),
                min_active_washes=_optional_int(args.get("min_active_washes")),
                max_active_washes=_optional_int(args.get("max_active_washes")) if args.get("max_active_washes") is not None else 2,
            )
        )
    if legacy_name == "run_waste_management_pareto":
        from strap.tools.waste_optimization import run_waste_management_pareto

        return await _await_if_needed(
            run_waste_management_pareto(
                feed=float(args["feed"]),
                pe_fraction=_optional_float(args.get("pe_fraction")),
                pet_fraction=_optional_float(args.get("pet_fraction")),
                n6_fraction=_optional_float(args.get("n6_fraction")),
                evoh_fraction=_optional_float(args.get("evoh_fraction")),
                feed_composition_json=_json_like_arg(args.get("feed_composition_json")),
                scenario=str(args.get("scenario", "A")),
                x_metric=str(args.get("x_metric", "total_cost")),
                y_metric=str(args.get("y_metric", "circularity")),
                n_points=int(args.get("n_points", 100)),
                candidate_solvents=_json_like_arg(args.get("candidate_solvents")),
                polymer_solvent_filters_json=_json_like_arg(args.get("polymer_solvent_filters_json")),
                stage_candidates_json=_json_like_arg(args.get("stage_candidates_json")),
                constraint_mode=args.get("constraint_mode"),
                fallback_policy=args.get("fallback_policy"),
                route_pool_mode=args.get("route_pool_mode"),
                min_active_washes=_optional_int(args.get("min_active_washes")),
                max_active_washes=_optional_int(args.get("max_active_washes")),
            )
        )
    if legacy_name == "plot_optimization_pareto_front":
        from strap.tools.visualization import plot_optimization_pareto_front

        return await _await_if_needed(
            plot_optimization_pareto_front(
                pareto_result_json=_json_like_arg(args.get("pareto_result_json")),
                color_by=str(args.get("color_by", "auto")),
                plot_mode=str(args.get("plot_mode", "frontier_only")),
                plot_title=args.get("plot_title"),
                source_handoff_id=args.get("source_handoff_id"),
                output_stem=args.get("output_stem"),
                output_dir=args.get("output_dir"),
                output_path=args.get("output_path"),
            )
        )
    return json_tool_error(
        f"No Claude SDK wrapper registered for {legacy_name}.",
        tool_name=legacy_name,
        error_code="unmapped_tool",
    )


async def call_mcp_tool(legacy_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a wrapped tool and return an MCP CallToolResult-like dict."""
    try:
        raw = await call_legacy_tool(legacy_name, args)
        return _mcp_text_result(raw, is_error=not _tool_success(raw))
    except Exception as exc:
        raw = json_tool_error(
            f"{legacy_name} failed: {exc}",
            tool_name=legacy_name,
            error_code="wrapper_exception",
        )
        return _mcp_text_result(raw, is_error=True)


def _schema_for_tool(legacy_name: str) -> dict[str, Any]:
    """Return narrow schemas with explicit optional fields where useful."""
    if legacy_name == "list_available_solvents":
        return {
            "type": "object",
            "properties": {
                "polymer": {"type": "string", "description": "Optional polymer filter such as LDPE, PET, or EVOH."},
                "limit": {"type": "integer", "description": "Optional maximum number of polymer-specific solvents."},
                "include_properties": {"type": "boolean", "description": "Whether to include physical properties."},
            },
            "required": [],
        }
    if legacy_name == "list_available_polymers":
        return {}
    if legacy_name == "predict_solubility":
        return {
            "type": "object",
            "properties": {
                "polymer_name": {"type": "string"},
                "solvent_name": {"type": "string"},
                "temperature_c": {"type": "number"},
            },
            "required": ["polymer_name", "solvent_name"],
        }
    if legacy_name == "predict_solubility_range":
        return {
            "type": "object",
            "properties": {
                "polymer_name": {"type": "string"},
                "solvent_name": {"type": "string"},
                "t_start_c": {"type": "number"},
                "t_end_c": {"type": "number"},
                "t_step_c": {"type": "number"},
            },
            "required": ["polymer_name", "solvent_name"],
        }
    if legacy_name == "plot_solubility_vs_temperature":
        return {
            "type": "object",
            "properties": {
                "polymers": {"type": "string", "description": "Comma-separated polymer names."},
                "solvents": {"type": "string", "description": "Comma-separated solvent names."},
                "temperature_min": {"type": "number"},
                "temperature_max": {"type": "number"},
                "output_dir": {"type": "string"},
                "output_path": {"type": "string"},
                "plot_title": {"type": "string"},
                "annotate_model_limits": {"type": "boolean"},
                "y_axis_max": {"type": "number"},
                "y_axis_ranges_json": {"type": "string"},
            },
            "required": ["polymers", "solvents"],
        }
    if legacy_name == "get_solvent_safety_card":
        return {
            "type": "object",
            "properties": {
                "solvent_name": {"type": "string"},
                "operating_temp_c": {"type": "number"},
                "include_pubchem": {"type": "boolean"},
            },
            "required": ["solvent_name"],
        }
    if legacy_name == "compare_solvent_safety_cards":
        return {
            "type": "object",
            "properties": {
                "solvent_names": {"type": "string"},
                "operating_temp_c": {"type": "number"},
                "include_pubchem": {"type": "boolean"},
                "limit": {"type": "integer"},
            },
            "required": ["solvent_names"],
        }
    if legacy_name == "run_waste_management_optimization":
        return {
            "type": "object",
            "properties": {
                "feed": {"type": "number", "description": "Total mixed plastic feed in tonnes/year."},
                "feed_composition_json": {
                    "type": "object",
                    "description": "Polymer fraction mapping, e.g. {'PE': 0.6, 'EVOH': 0.4}.",
                },
                "pe_fraction": {"type": "number"},
                "pet_fraction": {"type": "number"},
                "n6_fraction": {"type": "number"},
                "evoh_fraction": {"type": "number"},
                "scenario": {"type": "string", "description": "Scenario A, B, or C."},
                "objective": {"type": "string", "description": "max_profit, min_emissions, or max_circularity."},
                "candidate_solvents": {"type": "array", "items": {"type": "string"}},
                "polymer_solvent_filters_json": {
                    "type": "object",
                    "description": "Per-polymer allowed solvents, e.g. {'PE': ['Toluene'], 'EVOH': ['Pyridazine']}.",
                },
                "constraint_mode": {"type": "string"},
                "fallback_policy": {"type": "string"},
                "feed_mode": {"type": "string"},
                "composition_constraints_json": {"type": "object"},
                "composition_step": {"type": "number"},
                "min_active_washes": {"type": "integer"},
                "max_active_washes": {"type": "integer"},
            },
            "required": ["feed"],
        }
    if legacy_name == "run_waste_management_pareto":
        return {
            "type": "object",
            "properties": {
                "feed": {"type": "number", "description": "Total mixed plastic feed in tonnes/year."},
                "feed_composition_json": {
                    "type": "object",
                    "description": "Polymer fraction mapping, e.g. {'PE': 0.6, 'EVOH': 0.4}.",
                },
                "pe_fraction": {"type": "number"},
                "pet_fraction": {"type": "number"},
                "n6_fraction": {"type": "number"},
                "evoh_fraction": {"type": "number"},
                "scenario": {"type": "string", "description": "Scenario A, B, or C."},
                "x_metric": {"type": "string", "description": "Currently total_cost."},
                "y_metric": {"type": "string", "description": "emissions or circularity."},
                "n_points": {"type": "integer"},
                "candidate_solvents": {"type": "array", "items": {"type": "string"}},
                "polymer_solvent_filters_json": {
                    "type": "object",
                    "description": "Per-polymer allowed solvents, e.g. {'PE': ['Toluene', 'Heptane']}.",
                },
                "constraint_mode": {"type": "string"},
                "fallback_policy": {"type": "string"},
                "route_pool_mode": {"type": "string"},
                "min_active_washes": {"type": "integer"},
                "max_active_washes": {"type": "integer"},
            },
            "required": ["feed"],
        }
    if legacy_name == "plot_optimization_pareto_front":
        return {
            "type": "object",
            "properties": {
                "pareto_result_json": {"type": "object", "description": "Data payload returned by run_waste_management_pareto."},
                "color_by": {"type": "string"},
                "plot_mode": {"type": "string", "description": "frontier_only or landscape."},
                "plot_title": {"type": "string"},
                "source_handoff_id": {"type": "string"},
                "output_stem": {"type": "string"},
                "output_dir": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": [],
        }
    return {}


def _description_for_tool(legacy_name: str) -> str:
    descriptions = {
        "list_available_solvents": (
            "List solvents in DISSOLVE. Optionally pass polymer and limit; optional "
            "arguments are read with args.get because dict-schema keys are required."
        ),
        "list_available_polymers": "List polymers available in DISSOLVE.",
        "predict_solubility": "Predict one solubility point. Required: polymer_name, solvent_name. Optional: temperature_c.",
        "predict_solubility_range": (
            "Predict solubility over a temperature range. Required: polymer_name, solvent_name. "
            "Optional: t_start_c, t_end_c, t_step_c."
        ),
        "plot_solubility_vs_temperature": (
            "Create a publication-ready solubility-vs-temperature plot. Required: polymers, solvents. "
            "Optional: temperature_min, temperature_max, output_dir, output_path."
        ),
        "get_solvent_safety_card": "Render a solvent safety card. Required: solvent_name. Optional: operating_temp_c, include_pubchem.",
        "compare_solvent_safety_cards": "Compare solvent safety cards. Required: solvent_names. Optional: operating_temp_c, include_pubchem, limit.",
        "run_waste_management_optimization": (
            "Run the DISSOLVE waste-management point optimizer. Use for max_profit, min_emissions, "
            "or max_circularity at a fixed feed/composition. Include feed_composition_json or legacy fractions."
        ),
        "run_waste_management_pareto": (
            "Run a DISSOLVE waste-management Pareto sweep. Use for Pareto/frontier requests over total_cost "
            "versus emissions or circularity. Include feed_composition_json or legacy fractions."
        ),
        "plot_optimization_pareto_front": (
            "Create and save a Pareto-front plot from run_waste_management_pareto output. "
            "Pass pareto_result_json and optional output_dir or output_path."
        ),
    }
    return descriptions.get(legacy_name, legacy_name)


def _make_handler(legacy_name: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        return await call_mcp_tool(legacy_name, args)

    return _handler


def build_mcp_servers(tool_map: ToolNameMap | None = None) -> dict[str, Any]:
    """Create in-process SDK MCP servers for the registered wrappers."""
    sdk = import_claude_sdk()
    tool_decorator = sdk["tool"]
    create_server = sdk["create_sdk_mcp_server"]
    annotations_cls = sdk["ToolAnnotations"]
    tool_map = tool_map or ToolNameMap()
    servers: dict[str, Any] = {}
    for server_name in ("dissolve_solubility", "dissolve_safety", "dissolve_optimization"):
        sdk_tools = []
        for spec in tool_map.specs_for_server(server_name):
            sdk_tools.append(
                tool_decorator(
                    spec.legacy_name,
                    _description_for_tool(spec.legacy_name),
                    _schema_for_tool(spec.legacy_name),
                    annotations=annotations_cls(readOnlyHint=spec.read_only),
                )(_make_handler(spec.legacy_name))
            )
        servers[server_name] = create_server(
            name=server_name,
            version="1.0.0",
            tools=sdk_tools,
        )
    return servers
