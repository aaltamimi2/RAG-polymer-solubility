"""Static capability registry for typed planning PR 1.

The registry is deliberately conservative: high-value P0 tools have explicit
artifact semantics, while every exported tool also gets legacy coverage so CI
can detect stale names without forcing complete semantics in the first PR.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache

from strap.planning.models import CapabilitySpec, PlanRole, RequestPlan
from strap.subagent_config import load_subagent_specs
from strap import tools as tool_module


CAPABILITY_REGISTRY_VERSION = "2026.04.pr1"

ARTIFACT_TYPES: frozenset[str] = frozenset({
    "legacy_tool_result",
    "solubility_curve",
    "solubility_table",
    "solvent_property_table",
    "solvent_safety_card",
    "solvent_safety_comparison",
    "solvent_gscore_visualization",
    "pubchem_safety_record",
    "hsp_single_pair_summary",
    "hsp_red_heatmap",
    "thermal_property_prediction",
    "statistical_summary",
    "statistical_model_result",
    "separation_sequence_plan",
    "separation_topk_sequences",
    "separation_dp_state_map",
    "separation_tree_plot",
    "separation_selectivity_heatmap",
    "separation_process_flow_diagram",
    "precipitation_curve_plot",
    "atmospheric_feasibility_plot",
    "optimization_stage_candidates",
    "optimization_point_result",
    "optimization_point_plot",
    "optimization_pareto_front",
    "optimization_pareto_landscape",
    "optimization_pareto_plot",
    "optimization_pareto_slices",
    "optimization_pareto_slices_plot",
    "biosteam_tea_lca_result",
    "biosteam_tea_lca_plot",
    "biosteam_sensitivity_result",
    "contaminant_removal_screen",
    "research_citation_bundle",
    "rag_answer",
    "rag_diagnostic_plot",
    "sidecar_file",
    "handoff_payload",
})

ADAPTER_CALLABLES: frozenset[str] = frozenset({"build_handoff"})

_TOOL_GROUP_GETTERS: dict[str, Callable[[], list]] = {
    "database_query": tool_module.get_database_query_tools,
    "adaptive_separation": tool_module.get_adaptive_separation_tools,
    "statistical": tool_module.get_statistical_tools,
    "visualization": tool_module.get_visualization_tools,
    "solvent_property": tool_module.get_solvent_property_tools,
    "safety_gsk": tool_module.get_safety_gsk_tools,
    "safety_pubchem": tool_module.get_safety_pubchem_tools,
    "safety_card": tool_module.get_safety_card_tools,
    "listing": tool_module.get_listing_tools,
    "interpolation": tool_module.get_interpolation_tools,
    "ml_prediction": tool_module.get_ml_prediction_tools,
    "thermal_prediction": tool_module.get_thermal_prediction_tools,
    "literature": tool_module.get_literature_tools,
    "scholar": tool_module.get_scholar_tools,
    "patent": tool_module.get_patent_tools,
    "rag_core": tool_module.get_rag_core_tools,
    "rag_diagnostics": tool_module.get_rag_diagnostics_tools,
    "advanced_separation": tool_module.get_advanced_separation_tools,
    "separation_core": tool_module.get_separation_core_tools,
    "biosteam": tool_module.get_biosteam_tools,
    "contaminant_removal": tool_module.get_contaminant_removal_tools,
    "waste_optimization": tool_module.get_waste_optimization_tools,
    "solvent_lookup": tool_module.get_solvent_lookup_tools,
    "reflection": tool_module.get_reflection_tools,
    "sidecar_write": tool_module.get_sidecar_write_tools,
    "sidecar_read": tool_module.get_sidecar_read_tools,
    "result_extractor": tool_module.get_result_extractor_tools,
    "separation_plot": tool_module.get_separation_plot_tools,
}


def _callable_name(tool: object) -> str:
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", tool))


@lru_cache(maxsize=1)
def exported_tools_by_group() -> dict[str, frozenset[str]]:
    exported: dict[str, frozenset[str]] = {}
    for group, getter in _TOOL_GROUP_GETTERS.items():
        exported[group] = frozenset(_callable_name(tool) for tool in getter())
    return exported


@lru_cache(maxsize=1)
def exported_tool_names() -> frozenset[str]:
    names: set[str] = set()
    for group_tools in exported_tools_by_group().values():
        names.update(group_tools)
    return frozenset(names)


@lru_cache(maxsize=1)
def subagent_names() -> frozenset[str]:
    return frozenset(spec["name"] for spec in load_subagent_specs())


@lru_cache(maxsize=1)
def role_allowed_tools() -> dict[str, frozenset[str]]:
    by_group = exported_tools_by_group()
    allowed: dict[str, set[str]] = {"direct_tool": set(exported_tool_names())}
    for spec in load_subagent_specs():
        role = spec["name"]
        names: set[str] = set()
        for group in spec.get("tool_groups") or []:
            names.update(by_group.get(group, frozenset()))
        allowed[role] = names
    allowed["handoff_adapter"] = set(ADAPTER_CALLABLES)
    return {role: frozenset(names) for role, names in allowed.items()}


def _cap(
    capability_id: str,
    owner: PlanRole,
    callable_name: str,
    produces: Iterable[str],
    *,
    consumes: Iterable[str] = (),
    required_inputs: Iterable[str] = (),
    optional_inputs: Iterable[str] = (),
    rejects: Iterable[str] = (),
    callable_kind: str = "tool",
    supports_batch: bool = False,
    supports_multislice: bool = False,
    deterministic: bool = False,
) -> CapabilitySpec:
    return CapabilitySpec(
        capability_id=capability_id,
        owner=owner,
        callable_name=callable_name,
        callable_kind=callable_kind,  # type: ignore[arg-type]
        produces=list(produces),
        consumes=list(consumes),
        required_inputs=list(required_inputs),
        optional_inputs=list(optional_inputs),
        rejects=list(rejects),
        supports_batch=supports_batch,
        supports_multislice=supports_multislice,
        deterministic=deterministic,
    )


def _explicit_capabilities() -> list[CapabilitySpec]:
    return [
        _cap(
            "solubility.curve_direct",
            "direct_tool",
            "plot_solubility_vs_temperature",
            ["solubility_curve"],
            required_inputs=["polymer", "solvent"],
            optional_inputs=["temperature_range"],
            rejects=["separation_dp_state_map", "separation_tree_plot", "optimization_pareto_landscape"],
            deterministic=True,
        ),
        _cap(
            "solubility.curve_visualization",
            "visualization-specialist",
            "plot_solubility_vs_temperature",
            ["solubility_curve"],
            required_inputs=["polymer", "solvent"],
            optional_inputs=["temperature_range"],
            rejects=["separation_dp_state_map", "separation_tree_plot", "optimization_pareto_landscape"],
            deterministic=True,
        ),
        _cap(
            "solubility.curve_interactive",
            "visualization-specialist",
            "plot_solubility_vs_temperature_interactive",
            ["solubility_curve"],
            required_inputs=["polymer", "solvent"],
            optional_inputs=["temperature_range"],
            rejects=["separation_dp_state_map", "separation_tree_plot", "optimization_pareto_landscape"],
            deterministic=True,
        ),
        _cap(
            "solubility.table_prediction",
            "direct_tool",
            "predict_solubility_range",
            ["solubility_table"],
            required_inputs=["polymer", "solvent"],
            optional_inputs=["temperature_range"],
            rejects=["separation_dp_state_map", "separation_tree_plot", "optimization_pareto_landscape"],
            deterministic=True,
        ),
        _cap(
            "safety.solvent_card",
            "safety-analyst",
            "get_solvent_safety_card",
            ["solvent_safety_card"],
            required_inputs=["solvent_name"],
            optional_inputs=["operating_temp_c", "include_pubchem"],
            deterministic=True,
        ),
        _cap(
            "safety.solvent_comparison",
            "safety-analyst",
            "compare_solvent_safety_cards",
            ["solvent_safety_comparison"],
            required_inputs=["solvents"],
            optional_inputs=["operating_temp_c"],
            deterministic=True,
        ),
        _cap(
            "stats.hsp_single_pair",
            "statistics-ml",
            "predict_solubility_ml",
            ["hsp_single_pair_summary"],
            required_inputs=["polymer", "solvent"],
            optional_inputs=["generate_visualizations"],
            deterministic=True,
        ),
        _cap(
            "stats.hsp_matrix",
            "statistics-ml",
            "screen_hsp_solubility_matrix",
            ["hsp_red_heatmap"],
            optional_inputs=["polymer_category", "solvent_category", "solvent_polarity"],
            supports_batch=True,
            deterministic=True,
        ),
        _cap(
            "separation.plan_multiple_schemes",
            "separation-engineer",
            "plan_multiple_separation_schemes",
            ["separation_topk_sequences", "optimization_stage_candidates"],
            required_inputs=["polymers"],
            optional_inputs=["top_k", "temperature_recommendations"],
            supports_batch=True,
            deterministic=True,
        ),
        _cap(
            "separation.plan_sequential",
            "separation-engineer",
            "plan_sequential_separation",
            ["separation_sequence_plan", "optimization_stage_candidates"],
            required_inputs=["polymers"],
            deterministic=True,
        ),
        _cap(
            "separation.dp_state_map_plot",
            "visualization-specialist",
            "plot_dynamic_programming_separation_options",
            ["separation_dp_state_map"],
            consumes=["separation_topk_sequences"],
            rejects=["solubility_curve"],
            deterministic=True,
        ),
        _cap(
            "separation.tree_plot",
            "visualization-specialist",
            "create_separation_tree_plot",
            ["separation_tree_plot"],
            consumes=["separation_topk_sequences"],
            deterministic=True,
        ),
        _cap(
            "separation.selectivity_heatmap",
            "visualization-specialist",
            "create_selectivity_heatmap",
            ["separation_selectivity_heatmap"],
            required_inputs=["polymers"],
            optional_inputs=["solvents", "temperature"],
            rejects=["solubility_curve", "separation_dp_state_map"],
            deterministic=True,
        ),
        _cap(
            "handoff.optimization_candidates",
            "handoff_adapter",
            "build_handoff",
            ["optimization_stage_candidates", "handoff_payload"],
            consumes=["separation_topk_sequences"],
            callable_kind="adapter",
            deterministic=True,
        ),
        _cap(
            "optimization.point",
            "optimization-engineer",
            "run_waste_management_optimization",
            ["optimization_point_result"],
            consumes=["optimization_stage_candidates"],
            required_inputs=["feed_capacity_tpy", "feed_composition_json", "objective"],
            deterministic=True,
        ),
        _cap(
            "optimization.pareto",
            "optimization-engineer",
            "run_waste_management_pareto",
            ["optimization_pareto_front", "optimization_pareto_landscape"],
            consumes=["optimization_stage_candidates"],
            required_inputs=["feed_capacity_tpy", "feed_composition_json", "x_metric", "y_metric"],
            deterministic=True,
        ),
        _cap(
            "optimization.pareto_slices",
            "optimization-engineer",
            "run_waste_management_pareto_slices",
            ["optimization_pareto_slices", "optimization_pareto_front", "optimization_pareto_landscape", "sidecar_file"],
            consumes=["optimization_stage_candidates"],
            required_inputs=["feed_capacity_tpy", "composition_slices_json", "x_metric", "y_metric"],
            supports_batch=True,
            supports_multislice=True,
            deterministic=True,
        ),
        _cap(
            "visualization.optimization_point",
            "visualization-specialist",
            "plot_optimization_point_result",
            ["optimization_point_plot"],
            consumes=["optimization_point_result"],
            deterministic=True,
        ),
        _cap(
            "visualization.optimization_pareto",
            "visualization-specialist",
            "plot_optimization_pareto_front",
            ["optimization_pareto_plot"],
            consumes=["optimization_pareto_front", "optimization_pareto_landscape"],
            deterministic=True,
        ),
        _cap(
            "visualization.optimization_pareto_slices",
            "visualization-specialist",
            "plot_optimization_pareto_slices",
            ["optimization_pareto_slices_plot"],
            consumes=["optimization_pareto_slices"],
            supports_multislice=True,
            deterministic=True,
        ),
        _cap(
            "biosteam.single_tea_lca",
            "biosteam-analyst",
            "run_biosteam_simulation",
            ["biosteam_tea_lca_result"],
            required_inputs=["solvent", "target_plastic", "energy_case"],
            deterministic=True,
        ),
        _cap(
            "biosteam.visualize",
            "visualization-specialist",
            "visualize_biosteam_results",
            ["biosteam_tea_lca_plot"],
            consumes=["biosteam_tea_lca_result"],
            deterministic=True,
        ),
        _cap(
            "contaminant.strap_removal",
            "contaminant-removal-analyst",
            "screen_contaminant_strap_removal",
            ["contaminant_removal_screen"],
            required_inputs=["contaminant", "polymer"],
            deterministic=True,
        ),
        _cap(
            "research.scholar_search",
            "scholar-researcher",
            "search_google_scholar",
            ["research_citation_bundle"],
            required_inputs=["query"],
        ),
        _cap(
            "research.patent_search",
            "patent-researcher",
            "search_google_patents",
            ["research_citation_bundle"],
            required_inputs=["query"],
        ),
        _cap(
            "rag.ask_literature",
            "rag-analyst",
            "ask_literature",
            ["rag_answer"],
            required_inputs=["question"],
        ),
    ]


def _first_owner_for_tool(tool_name: str) -> PlanRole:
    for role, allowed in role_allowed_tools().items():
        if role == "direct_tool":
            continue
        if tool_name in allowed:
            return role  # type: ignore[return-value]
    return "direct_tool"


def _legacy_capabilities(existing_callables: set[str]) -> list[CapabilitySpec]:
    capabilities: list[CapabilitySpec] = []
    for tool_name in sorted(exported_tool_names() - existing_callables):
        capabilities.append(
            CapabilitySpec(
                capability_id=f"legacy.{tool_name}",
                owner=_first_owner_for_tool(tool_name),
                callable_name=tool_name,
                callable_kind="tool",
                produces=["legacy_tool_result"],
                legacy_unplanned=True,
            )
        )
    return capabilities


@lru_cache(maxsize=1)
def get_default_capability_registry() -> dict[str, CapabilitySpec]:
    explicit = _explicit_capabilities()
    existing = {cap.callable_name for cap in explicit}
    all_capabilities = explicit + _legacy_capabilities(existing)
    return {cap.capability_id: cap for cap in all_capabilities}


def capabilities_for_artifact(
    artifact_type: str,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[CapabilitySpec]:
    caps = registry or get_default_capability_registry()
    return [cap for cap in caps.values() if artifact_type in cap.produces]


def capabilities_for_callable(
    callable_name: str,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[CapabilitySpec]:
    caps = registry or get_default_capability_registry()
    return [cap for cap in caps.values() if cap.callable_name == callable_name]


def validate_capability_registry(
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[str]:
    caps = registry or get_default_capability_registry()
    errors: list[str] = []
    ids_seen: set[str] = set()
    exported = exported_tool_names()
    subagents = subagent_names()
    allowed_by_role = role_allowed_tools()
    valid_owners = set(allowed_by_role) | {"handoff_adapter"}
    covered_tools: set[str] = set()

    for cap_id, cap in caps.items():
        if cap_id in ids_seen:
            errors.append(f"duplicate capability_id: {cap_id}")
        ids_seen.add(cap_id)
        if cap.capability_id != cap_id:
            errors.append(f"registry key {cap_id} does not match spec id {cap.capability_id}")
        if cap.owner not in valid_owners:
            errors.append(f"{cap_id}: unknown owner {cap.owner}")
        if cap.callable_kind == "tool":
            if cap.callable_name not in exported:
                errors.append(f"{cap_id}: missing exported tool {cap.callable_name}")
            covered_tools.add(cap.callable_name)
        elif cap.callable_kind == "adapter":
            if cap.callable_name not in ADAPTER_CALLABLES and cap.callable_name not in exported:
                errors.append(f"{cap_id}: missing adapter {cap.callable_name}")
            if cap.callable_name in exported:
                covered_tools.add(cap.callable_name)
        elif cap.callable_kind == "subagent":
            if cap.callable_name not in subagents:
                errors.append(f"{cap_id}: missing subagent {cap.callable_name}")

        for artifact_type in list(cap.produces) + list(cap.consumes) + list(cap.rejects):
            if artifact_type not in ARTIFACT_TYPES:
                errors.append(f"{cap_id}: unknown artifact type {artifact_type}")

        allowed_tools = allowed_by_role.get(cap.owner, frozenset())
        if cap.callable_kind == "tool" and cap.owner not in {"direct_tool", "handoff_adapter"}:
            if cap.callable_name not in allowed_tools:
                errors.append(f"{cap_id}: tool {cap.callable_name} is not allowed for role {cap.owner}")
        if cap.callable_kind == "adapter" and cap.owner != "handoff_adapter":
            errors.append(f"{cap_id}: adapter capabilities must be owned by handoff_adapter")

    missing_coverage = exported - covered_tools
    for tool_name in sorted(missing_coverage):
        errors.append(f"exported tool has no capability: {tool_name}")
    return errors


def assert_valid_capability_registry(
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> None:
    errors = validate_capability_registry(registry)
    if errors:
        raise ValueError("Invalid capability registry:\n" + "\n".join(errors))


def validate_plan_against_registry(
    plan: RequestPlan,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[str]:
    caps = registry or get_default_capability_registry()
    errors: list[str] = []
    exported = exported_tool_names()
    allowed_by_role = role_allowed_tools()

    for step in plan.steps:
        role_allowed = allowed_by_role.get(step.role, frozenset())
        if step.role not in allowed_by_role:
            errors.append(f"{step.step_id}: unknown role {step.role}")

        for input_contract in step.input_contracts:
            if input_contract.artifact_type not in ARTIFACT_TYPES:
                errors.append(f"{step.step_id}: unknown input artifact {input_contract.artifact_type}")

        output_artifacts: list[str] = []
        for output_contract in step.output_contracts:
            for artifact_contract in output_contract.artifact_contracts:
                artifact_type = artifact_contract.artifact_type
                output_artifacts.append(artifact_type)
                if artifact_type not in ARTIFACT_TYPES:
                    errors.append(f"{step.step_id}: unknown output artifact {artifact_type}")
                for forbidden in artifact_contract.forbidden_artifact_types:
                    if forbidden not in ARTIFACT_TYPES:
                        errors.append(f"{step.step_id}: unknown forbidden artifact {forbidden}")

        if step.execution_kind == "synthesis":
            continue

        for tool_name in step.allowed_tools:
            if step.execution_kind == "handoff_adapter":
                if tool_name not in ADAPTER_CALLABLES:
                    errors.append(f"{step.step_id}: unknown handoff adapter {tool_name}")
                continue
            if tool_name not in exported:
                errors.append(f"{step.step_id}: unknown tool {tool_name}")
                continue
            if step.role not in {"direct_tool", "handoff_adapter"} and tool_name not in role_allowed:
                errors.append(f"{step.step_id}: tool {tool_name} is not allowed for role {step.role}")

        for artifact_type in output_artifacts:
            if artifact_type not in ARTIFACT_TYPES:
                continue
            if artifact_type == "legacy_tool_result":
                continue
            matching = [
                cap for cap in caps.values()
                if cap.callable_name in step.allowed_tools and artifact_type in cap.produces
            ]
            if not matching:
                errors.append(
                    f"{step.step_id}: no allowed tool can produce artifact {artifact_type}"
                )
    return errors
