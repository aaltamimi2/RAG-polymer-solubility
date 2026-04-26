"""Compile user requests into typed plans without executing them."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError

from strap.planning.capability_registry import (
    CAPABILITY_REGISTRY_VERSION,
    CapabilitySpec,
    capabilities_for_artifact,
)
from strap.planning.extractors import ExtractedFacts, extract_facts
from strap.planning.models import (
    ArtifactContract,
    FinalResponseContract,
    InputContract,
    MissingInput,
    OutputContract,
    PlanStep,
    PlanningModel,
    RequestPlan,
)
from strap.planning.validators import validate_compiled_plan


COMPILER_VERSION = "2026.04.pr2"
CompileStatus = Literal["compiled", "clarification_required", "unsupported", "invalid"]


class PlannerBackend(Protocol):
    """Provider-agnostic planner backend interface.

    PR 2 tests use stub implementations only. Live model backends can implement
    this protocol later without changing compiler validation.
    """

    planner_model_id: str | None

    def propose_plan_payload(self, query: str, facts: ExtractedFacts) -> str | Mapping[str, Any] | None:
        """Return a raw plan payload, JSON string, mapping, or None to defer."""


class CompileDiagnostic(PlanningModel):
    level: Literal["info", "warning", "error"] = "info"
    code: str
    message: str
    step_id: str | None = None


class CompileResult(PlanningModel):
    status: CompileStatus
    plan: RequestPlan | None = None
    diagnostics: list[CompileDiagnostic] = Field(default_factory=list)
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_id(query: str) -> str:
    return "plan_" + hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]


def _artifact_output(
    contract_id: str,
    *artifact_types: str,
    validation_checks: list[str] | None = None,
    path_required: bool = False,
) -> OutputContract:
    return OutputContract(
        contract_id=contract_id,
        artifact_contracts=[
            ArtifactContract(
                artifact_type=artifact_type,
                path_policy="required" if path_required else "optional",
            )
            for artifact_type in artifact_types
        ],
        validation_checks=validation_checks or [],
    )


def _select_capability(
    artifact_type: str,
    *,
    owner: str | None = None,
    supports_multislice: bool | None = None,
) -> CapabilitySpec:
    caps = [cap for cap in capabilities_for_artifact(artifact_type) if not cap.legacy_unplanned]
    if owner is not None:
        caps = [cap for cap in caps if cap.owner == owner]
    if supports_multislice is not None:
        caps = [cap for cap in caps if cap.supports_multislice is supports_multislice]
    if not caps:
        raise ValueError(f"no capability can produce {artifact_type} for owner={owner!r}")
    return sorted(caps, key=lambda cap: cap.capability_id)[0]


def _base_payload(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> dict[str, Any]:
    return {
        "plan_id": _plan_id(query),
        "created_at": created_at or _now_iso(),
        "compiler_version": COMPILER_VERSION,
        "capability_registry_version": CAPABILITY_REGISTRY_VERSION,
        "planner_model_id": planner_model_id,
        "user_query": query,
        "global_constraints": {
            "scenario": facts.scenario,
            "energy_case": facts.energy_case,
            "feed_capacity_tpy": facts.feed_capacity_tpy,
            "feed_composition": facts.feed_composition,
            "composition_slices": facts.composition_slices,
            "top_k_per_polymer": facts.top_k_per_polymer,
            "n_points": facts.n_points,
            "min_washes": facts.min_washes,
            "max_washes": facts.max_washes,
            "metrics": facts.metrics,
            "requested_artifact_types": facts.requested_artifact_types,
            "forbidden_artifact_types": facts.forbidden_artifact_types,
            "output_dir": facts.output_dir,
        },
        "final_response_contract": FinalResponseContract(
            require_paths=any(
                artifact.endswith("_plot")
                or artifact.endswith("_heatmap")
                or artifact in {"optimization_pareto_landscape", "separation_tree_plot"}
                for artifact in facts.requested_artifact_types
            ),
            must_cite_artifacts=list(facts.requested_artifact_types),
            forbidden_claims=["upstream prose substituted for downstream structured output"],
        ),
    }


def _tool_args(facts: ExtractedFacts, **extra: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "polymers": facts.polymers,
        "solvents": facts.solvents,
        "temperature_c": facts.temperatures_c[0] if facts.temperatures_c else None,
        "feed_capacity_tpy": facts.feed_capacity_tpy,
        "feed_composition_json": facts.feed_composition or None,
        "composition_slices_json": facts.composition_slices or None,
        "scenario": facts.scenario,
        "top_k_per_polymer": facts.top_k_per_polymer,
        "n_points": facts.n_points,
        "min_washes": facts.min_washes,
        "max_washes": facts.max_washes,
    }
    args.update(extra)
    return {key: value for key, value in args.items() if value not in (None, [], {})}


def _plot_destination_args(facts: ExtractedFacts, *, default_stem: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if facts.output_dir:
        args["output_dir"] = facts.output_dir
    if facts.output_filename_hint:
        stem = facts.output_filename_hint.rsplit(".", 1)[0]
        args["output_stem"] = stem or default_stem
    return args


def _missing_inputs_for_optimization(facts: ExtractedFacts, *, slices: bool = False) -> list[MissingInput]:
    missing: list[MissingInput] = []
    if facts.feed_capacity_tpy is None:
        missing.append(MissingInput(name="feed_capacity_tpy", reason="Optimization requires feed capacity."))
    if slices:
        if not facts.composition_slices:
            missing.append(MissingInput(name="composition_slices_json", reason="Multi-slice optimization requires composition slices."))
    elif not facts.feed_composition:
        missing.append(MissingInput(name="feed_composition_json", reason="Optimization requires feed composition."))
    return missing


def _compile_safety(query: str, facts: ExtractedFacts, *, created_at: str | None, planner_model_id: str | None) -> RequestPlan:
    comparison = "solvent_safety_comparison" in facts.requested_artifact_types
    artifact = "solvent_safety_comparison" if comparison else "solvent_safety_card"
    cap = _select_capability(artifact, owner="safety-analyst")
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "single_tool_or_specialist",
        "intent_family": "safety",
        "complexity": "moderate" if comparison else "simple",
        "steps": [
            PlanStep(
                step_id="safety_assessment",
                label="Safety assessment",
                role=cap.owner,
                execution_kind="tool",
                allowed_tools=[cap.callable_name],
                output_contracts=[_artifact_output("safety_output", artifact)],
                tool_args_template=_tool_args(
                    facts,
                    solvent_name=facts.solvents[0] if facts.solvents and not comparison else None,
                    solvents=facts.solvents if comparison else None,
                    operating_temp_c=facts.temperatures_c[0] if facts.temperatures_c else None,
                    include_pubchem=True,
                ),
            )
        ],
    })
    return RequestPlan(**payload)


def _compile_hsp(query: str, facts: ExtractedFacts, *, created_at: str | None, planner_model_id: str | None) -> RequestPlan:
    artifact = "hsp_red_heatmap" if "hsp_red_heatmap" in facts.requested_artifact_types else "hsp_single_pair_summary"
    if artifact == "hsp_red_heatmap":
        missing: list[MissingInput] = []
        if not facts.polymers and not facts.hsp_polymer_category:
            missing.append(MissingInput(name="polymers_or_polymer_category", reason="HSP matrix screening requires polymers or a polymer category."))
        if not facts.solvents and not facts.hsp_solvent_category and not facts.hsp_solvent_polarity:
            missing.append(MissingInput(name="solvents_or_solvent_category", reason="HSP matrix screening requires solvents, a solvent category, or solvent polarity."))
        if missing:
            payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
            payload.update({
                "mode": "clarification_required",
                "intent_family": "statistics_ml",
                "complexity": "moderate",
                "missing_inputs": missing,
                "steps": [],
            })
            return RequestPlan(**payload)

    cap = _select_capability(artifact, owner="statistics-ml")
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "single_tool_or_specialist",
        "intent_family": "statistics_ml",
        "complexity": "moderate",
        "steps": [
            PlanStep(
                step_id="hsp_screen",
                label="HSP/RED screen",
                role=cap.owner,
                execution_kind="tool",
                allowed_tools=[cap.callable_name],
                output_contracts=[_artifact_output("hsp_output", artifact, path_required=artifact == "hsp_red_heatmap")],
                tool_args_template=_tool_args(
                    facts,
                    polymer=facts.polymers[0] if facts.polymers else None,
                    solvent=facts.solvents[0] if facts.solvents else None,
                    polymers=facts.polymers if artifact == "hsp_red_heatmap" and facts.polymers else None,
                    solvents=facts.solvents if artifact == "hsp_red_heatmap" and facts.solvents else None,
                    polymer_category=facts.hsp_polymer_category if artifact == "hsp_red_heatmap" else None,
                    solvent_category=facts.hsp_solvent_category if artifact == "hsp_red_heatmap" else None,
                    solvent_polarity=facts.hsp_solvent_polarity if artifact == "hsp_red_heatmap" else None,
                    output_dir=facts.output_dir,
                    generate_visualizations=True,
                ),
            )
        ],
    })
    return RequestPlan(**payload)


def _compile_dp_state_map(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    cap = _select_capability("separation_dp_state_map", owner="visualization-specialist")
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "single_tool_or_specialist",
        "intent_family": "visualization",
        "complexity": "moderate",
        "steps": [
            PlanStep(
                step_id="plot_state_map",
                label="Plot dynamic-programming state map",
                role=cap.owner,
                execution_kind="tool",
                allowed_tools=[cap.callable_name],
                output_contracts=[
                    _artifact_output("dp_state_map_output", "separation_dp_state_map", path_required=True)
                ],
                tool_args_template=_tool_args(
                    facts,
                    polymers=",".join(facts.polymers) if facts.polymers else None,
                    temperature=facts.temperatures_c[0] if facts.temperatures_c else None,
                    output_dir=facts.output_dir,
                ),
            )
        ],
    })
    return RequestPlan(**payload)


def _compile_separation_visualization(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    artifact = (
        "separation_selectivity_heatmap"
        if "separation_selectivity_heatmap" in facts.requested_artifact_types
        else "separation_tree_plot"
    )
    cap = _select_capability(artifact, owner="visualization-specialist")
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "single_tool_or_specialist",
        "intent_family": "visualization",
        "complexity": "moderate",
        "steps": [
            PlanStep(
                step_id="plot_separation_visualization",
                label="Plot separation visualization",
                role=cap.owner,
                execution_kind="tool",
                allowed_tools=[cap.callable_name],
                output_contracts=[
                    _artifact_output("separation_visualization_output", artifact, path_required=True)
                ],
                tool_args_template=_tool_args(
                    facts,
                    polymers=",".join(facts.polymers) if facts.polymers else None,
                    solvents=",".join(facts.solvents) if facts.solvents else None,
                    temperature=facts.temperatures_c[0] if facts.temperatures_c else None,
                    output_dir=facts.output_dir,
                ),
            )
        ],
    })
    return RequestPlan(**payload)


def _biosteam_target_plastic(facts: ExtractedFacts) -> str | None:
    for raw in ("LDPE", "HDPE", "PE", "EVOH", "PET", "PP", "PS", "PVC", "PC"):
        if raw in facts.polymer_aliases:
            return raw
        if raw in facts.polymers:
            return raw
    return facts.polymers[0] if facts.polymers else None


def _target_plastic_percent(facts: ExtractedFacts, target_plastic: str | None) -> float | None:
    if not target_plastic:
        return None
    canonical = facts.polymer_aliases.get(target_plastic, target_plastic)
    if canonical in {"LDPE", "HDPE"}:
        canonical = "PE"
    fraction = facts.feed_composition.get(canonical)
    return fraction * 100.0 if fraction is not None else None


def _compile_biosteam(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    target_plastic = _biosteam_target_plastic(facts)
    missing: list[MissingInput] = []
    if not facts.solvents:
        missing.append(MissingInput(name="solvent", reason="BioSTEAM TEA/LCA requires a solvent."))
    if target_plastic is None:
        missing.append(MissingInput(name="target_plastic", reason="BioSTEAM TEA/LCA requires a target polymer."))
    if not facts.energy_case:
        missing.append(MissingInput(name="energy_case", reason="BioSTEAM TEA/LCA requires an explicit energy case C1, C2, or C3."))
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if missing:
        payload.update({
            "mode": "clarification_required",
            "intent_family": "biosteam_tea_lca",
            "complexity": "moderate",
            "missing_inputs": missing,
            "steps": [],
        })
        return RequestPlan(**payload)

    result_cap = _select_capability("biosteam_tea_lca_result", owner="biosteam-analyst")
    plot_requested = "biosteam_tea_lca_plot" in facts.requested_artifact_types
    steps = [
        PlanStep(
            step_id="run_biosteam_tea_lca",
            label="Run BioSTEAM TEA/LCA",
            role=result_cap.owner,
            execution_kind="tool",
            allowed_tools=[result_cap.callable_name],
            output_contracts=[
                _artifact_output(
                    "biosteam_result_output",
                    "biosteam_tea_lca_result",
                    validation_checks=["source_is_biosteam_structured_output"],
                )
            ],
            tool_args_template=_tool_args(
                facts,
                solvent=facts.solvents[0],
                target_plastic=target_plastic,
                energy_case=facts.energy_case,
                processing_capacity=facts.feed_capacity_tpy,
                target_plastic_percent=_target_plastic_percent(facts, target_plastic),
                dissolution_temp_c=facts.temperatures_c[0] if facts.temperatures_c else None,
            ),
        )
    ]
    if plot_requested:
        plot_cap = _select_capability("biosteam_tea_lca_plot", owner="visualization-specialist")
        steps.append(
            PlanStep(
                step_id="plot_biosteam_tea_lca",
                label="Plot BioSTEAM TEA/LCA",
                role=plot_cap.owner,
                execution_kind="tool",
                allowed_tools=[plot_cap.callable_name],
                depends_on=["run_biosteam_tea_lca"],
                input_contracts=[InputContract(artifact_type="biosteam_tea_lca_result", source_step_id="run_biosteam_tea_lca")],
                output_contracts=[_artifact_output("biosteam_plot_output", "biosteam_tea_lca_plot", path_required=True)],
                tool_args_template={
                    "source_step_id": "run_biosteam_tea_lca",
                    "chart_types": "all",
                    **_plot_destination_args(facts, default_stem="biosteam_tea_lca"),
                },
            )
        )

    payload.update({
        "mode": "planned_workflow" if plot_requested else "single_agent",
        "intent_family": "biosteam_tea_lca",
        "complexity": "moderate",
        "workflow_id": "biosteam_tea_lca_with_plot" if plot_requested else None,
        "steps": steps,
    })
    return RequestPlan(**payload)


def _optimization_metrics(facts: ExtractedFacts) -> tuple[str, str]:
    metrics = facts.metrics
    if "emissions" in metrics:
        return "total_cost", "emissions"
    if "circularity" in metrics:
        return "total_cost", "circularity"
    return "total_cost", "circularity"


def _compile_direct_optimization(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan | CompileResult:
    missing = _missing_inputs_for_optimization(facts)
    if missing:
        payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
        payload.update({
            "mode": "clarification_required",
            "intent_family": "optimization",
            "complexity": "moderate",
            "missing_inputs": missing,
            "steps": [],
        })
        return RequestPlan(**payload)
    cap = _select_capability("optimization_point_result", owner="optimization-engineer")
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "single_agent",
        "intent_family": "optimization",
        "complexity": "moderate",
        "steps": [
            PlanStep(
                step_id="optimize_point",
                label="Optimize point objective",
                role=cap.owner,
                execution_kind="tool",
                allowed_tools=[cap.callable_name],
                output_contracts=[
                    _artifact_output(
                        "optimization_point_output",
                        "optimization_point_result",
                        validation_checks=["source_is_optimizer_structured_output"],
                    )
                ],
                tool_args_template=_tool_args(
                    facts,
                    objective=facts.objective or "max_profit",
                    solvent_shortlist=facts.solvents or None,
                ),
            )
        ],
    })
    return RequestPlan(**payload)


def _compile_direct_pareto(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    missing = _missing_inputs_for_optimization(facts)
    if missing:
        payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
        payload.update({
            "mode": "clarification_required",
            "intent_family": "optimization",
            "complexity": "moderate",
            "missing_inputs": missing,
            "steps": [],
        })
        return RequestPlan(**payload)

    cap = _select_capability("optimization_pareto_landscape", owner="optimization-engineer")
    x_metric, y_metric = _optimization_metrics(facts)
    steps: list[PlanStep] = [
        PlanStep(
            step_id="optimize_pareto",
            label="Optimize Pareto frontier",
            role=cap.owner,
            execution_kind="tool",
            allowed_tools=[cap.callable_name],
            output_contracts=[
                _artifact_output(
                    "optimization_pareto_output",
                    "optimization_pareto_front",
                    "optimization_pareto_landscape",
                    validation_checks=["source_is_optimizer_structured_output"],
                )
            ],
            tool_args_template=_tool_args(
                facts,
                x_metric=x_metric,
                y_metric=y_metric,
                objective="pareto",
                solvent_shortlist=facts.solvents or None,
            ),
        )
    ]

    if "optimization_pareto_plot" in facts.requested_artifact_types or "visualization" in facts.workflow_markers:
        plot_cap = _select_capability("optimization_pareto_plot", owner="visualization-specialist")
        steps.append(
            PlanStep(
                step_id="plot_optimization",
                label="Plot optimization output",
                role=plot_cap.owner,
                execution_kind="tool",
                allowed_tools=[plot_cap.callable_name],
                depends_on=["optimize_pareto"],
                input_contracts=[InputContract(artifact_type="optimization_pareto_landscape", source_step_id="optimize_pareto")],
                output_contracts=[
                    _artifact_output(
                        "optimization_plot_output",
                        "optimization_pareto_plot",
                        validation_checks=["visualization_from_authoritative_payload"],
                        path_required=True,
                    )
                ],
                tool_args_template={
                    "source_step_id": "optimize_pareto",
                    "plot_mode": "landscape",
                    **_plot_destination_args(facts, default_stem="optimization_pareto"),
                },
            )
        )

    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "single_agent",
        "intent_family": "optimization",
        "complexity": "moderate",
        "steps": steps,
    })
    return RequestPlan(**payload)


def _compile_routed_optimization(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    is_slices = "optimization_pareto_slices" in facts.requested_artifact_types
    is_pareto = is_slices or "optimization_pareto_landscape" in facts.requested_artifact_types
    opt_artifact = (
        "optimization_pareto_slices"
        if is_slices
        else "optimization_pareto_landscape"
        if is_pareto
        else "optimization_point_result"
    )
    opt_cap = _select_capability(
        opt_artifact,
        owner="optimization-engineer",
        supports_multislice=True if is_slices else None,
    )
    sep_cap = _select_capability("separation_topk_sequences", owner="separation-engineer")
    handoff_cap = _select_capability("handoff_payload", owner="handoff_adapter")
    steps: list[PlanStep] = [
        PlanStep(
            step_id="separation_candidates",
            label="Compile separation candidates",
            role=sep_cap.owner,
            execution_kind="tool",
            allowed_tools=[sep_cap.callable_name],
            output_contracts=[
                _artifact_output("separation_candidates_output", "separation_topk_sequences", "optimization_stage_candidates")
            ],
            tool_args_template=_tool_args(
                facts,
                temperature_recommendations="temperature" in query.lower(),
                source_user_query=query,
            ),
        ),
        PlanStep(
            step_id="build_optimization_handoff",
            label="Build optimization handoff",
            role=handoff_cap.owner,
            execution_kind="handoff_adapter",
            allowed_tools=[handoff_cap.callable_name],
            depends_on=["separation_candidates"],
            input_contracts=[InputContract(artifact_type="separation_topk_sequences", source_step_id="separation_candidates")],
            output_contracts=[_artifact_output("optimization_handoff_output", "optimization_stage_candidates", "handoff_payload")],
            tool_args_template={
                "source_step_id": "separation_candidates",
                "target_role": "optimization-engineer",
                "source_user_query": query,
            },
        ),
    ]

    x_metric, y_metric = _optimization_metrics(facts)
    opt_args = _tool_args(
        facts,
        x_metric=x_metric,
        y_metric=y_metric,
        objective=facts.objective or ("pareto" if is_pareto else "max_profit"),
    )
    steps.append(
        PlanStep(
            step_id="optimize_slices" if is_slices else "optimize_pareto" if is_pareto else "optimize_point",
            label="Optimize routed candidates",
            role=opt_cap.owner,
            execution_kind="tool",
            allowed_tools=[opt_cap.callable_name],
            depends_on=["build_optimization_handoff"],
            input_contracts=[InputContract(artifact_type="optimization_stage_candidates", source_step_id="build_optimization_handoff")],
            output_contracts=[
                _artifact_output(
                    "optimization_output",
                    *(
                        ["optimization_pareto_slices", "optimization_pareto_front", "optimization_pareto_landscape", "sidecar_file"]
                        if is_slices
                        else ["optimization_pareto_front", "optimization_pareto_landscape"]
                        if is_pareto
                        else [opt_artifact]
                    ),
                    validation_checks=["source_handoff_consumed", "no_upstream_prose_substitution"],
                )
            ],
            tool_args_template=opt_args,
        )
    )

    if "separation_tree_plot" in facts.requested_artifact_types:
        tree_cap = _select_capability("separation_tree_plot", owner="visualization-specialist")
        steps.append(
            PlanStep(
                step_id="plot_separation_tree",
                label="Plot separation tree",
                role=tree_cap.owner,
                execution_kind="tool",
                allowed_tools=[tree_cap.callable_name],
                depends_on=["separation_candidates"],
                input_contracts=[InputContract(artifact_type="separation_topk_sequences", source_step_id="separation_candidates")],
                output_contracts=[_artifact_output("separation_tree_output", "separation_tree_plot", path_required=True)],
                tool_args_template={"source_step_id": "separation_candidates"},
            )
        )

    plot_artifact = (
        "optimization_pareto_slices_plot"
        if is_slices
        else "optimization_pareto_plot"
        if is_pareto
        else "optimization_point_plot"
    )
    if plot_artifact in facts.requested_artifact_types or "visualization" in facts.workflow_markers:
        plot_cap = _select_capability(plot_artifact, owner="visualization-specialist")
        source_step = "optimize_slices" if is_slices else "optimize_pareto" if is_pareto else "optimize_point"
        source_artifact = opt_artifact
        steps.append(
            PlanStep(
                step_id="plot_optimization",
                label="Plot optimization output",
                role=plot_cap.owner,
                execution_kind="tool",
                allowed_tools=[plot_cap.callable_name],
                depends_on=[source_step],
                input_contracts=[InputContract(artifact_type=source_artifact, source_step_id=source_step)],
                output_contracts=[
                    _artifact_output(
                        "optimization_plot_output",
                        plot_artifact,
                        validation_checks=["visualization_from_authoritative_payload"],
                        path_required=True,
                    )
                ],
                tool_args_template={
                    "source_step_id": source_step,
                    "plot_mode": "landscape" if is_pareto else "point",
                    **_plot_destination_args(
                        facts,
                        default_stem="optimization_pareto_slices" if is_slices else "optimization_pareto",
                    ),
                },
            )
        )

    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "planned_workflow",
        "intent_family": "mixed_workflow",
        "complexity": "complex",
        "workflow_id": "routed_optimization_slices" if is_slices else "routed_optimization",
        "steps": steps,
    })
    return RequestPlan(**payload)


def _compile_deterministic(
    query: str,
    facts: ExtractedFacts,
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    requested = set(facts.requested_artifact_types)
    query_lc = query.lower()
    waste_optimization_intent = (
        "optimization_pareto_front" in requested
        or "optimization_pareto_landscape" in requested
        or "optimization_pareto_slices" in requested
        or "waste management" in query_lc
        or "waste-management" in query_lc
        or "circularity" in query_lc
    )
    if {"separation_topk_sequences", "optimization_stage_candidates"} & requested and (
        "optimization" in facts.workflow_markers or "optimization" in requested
    ):
        return _compile_routed_optimization(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "optimization_pareto_slices" in requested and "separation" in facts.workflow_markers:
        return _compile_routed_optimization(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "biosteam_tea_lca_result" in requested and not waste_optimization_intent:
        return _compile_biosteam(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "optimization_pareto_landscape" in requested:
        return _compile_direct_pareto(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "optimization_point_result" in requested or (
        "optimization" in facts.workflow_markers and "separation" not in facts.workflow_markers
    ):
        result = _compile_direct_optimization(query, facts, created_at=created_at, planner_model_id=planner_model_id)
        if isinstance(result, RequestPlan):
            return result
        raise ValueError("direct optimization returned non-plan")
    if "separation_dp_state_map" in requested:
        return _compile_dp_state_map(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "separation_tree_plot" in requested or "separation_selectivity_heatmap" in requested:
        return _compile_separation_visualization(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "solvent_safety_card" in requested or "solvent_safety_comparison" in requested:
        return _compile_safety(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    if "hsp_single_pair_summary" in requested or "hsp_red_heatmap" in requested:
        return _compile_hsp(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload = _base_payload(query, facts, created_at=created_at, planner_model_id=planner_model_id)
    payload.update({
        "mode": "unsupported",
        "intent_family": "mixed_workflow",
        "complexity": "simple",
        "unsupported_reason": "No deterministic PR 2 compiler path for this request.",
        "steps": [],
    })
    return RequestPlan(**payload)


def _parse_backend_payload(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("planner backend JSON payload must be an object")
        return parsed
    return dict(raw)


def _compile_backend_payload(
    query: str,
    facts: ExtractedFacts,
    raw_payload: str | Mapping[str, Any],
    *,
    created_at: str | None,
    planner_model_id: str | None,
) -> RequestPlan:
    payload = _parse_backend_payload(raw_payload)
    payload.setdefault("plan_id", _plan_id(query))
    payload.setdefault("created_at", created_at or _now_iso())
    payload.setdefault("compiler_version", COMPILER_VERSION)
    payload.setdefault("capability_registry_version", CAPABILITY_REGISTRY_VERSION)
    payload.setdefault("planner_model_id", planner_model_id)
    payload.setdefault("user_query", query)
    payload.setdefault("final_response_contract", FinalResponseContract())
    return RequestPlan(**payload)


def _result(
    status: CompileStatus,
    facts: ExtractedFacts,
    *,
    plan: RequestPlan | None = None,
    diagnostics: list[CompileDiagnostic] | None = None,
    validation_errors: list[str] | None = None,
) -> CompileResult:
    return CompileResult(
        status=status,
        plan=plan,
        diagnostics=diagnostics or [],
        extracted_facts=facts.model_dump(),
        validation_errors=validation_errors or [],
    )


def compile_request(
    query: str,
    *,
    context: dict[str, Any] | None = None,
    planner_backend: PlannerBackend | None = None,
    created_at: str | None = None,
) -> CompileResult:
    """Compile a query into a typed plan or a typed compile result failure."""
    facts = extract_facts(query, context)
    planner_model_id = getattr(planner_backend, "planner_model_id", None) if planner_backend else None
    diagnostics = [
        CompileDiagnostic(
            level="info",
            code="facts_extracted",
            message="Deterministic facts extracted; no execution attempted.",
        )
    ]
    try:
        backend_payload = planner_backend.propose_plan_payload(query, facts) if planner_backend else None
        plan = (
            _compile_backend_payload(
                query,
                facts,
                backend_payload,
                created_at=created_at,
                planner_model_id=planner_model_id,
            )
            if backend_payload is not None
            else _compile_deterministic(
                query,
                facts,
                created_at=created_at,
                planner_model_id=planner_model_id,
            )
        )
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return _result(
            "invalid",
            facts,
            diagnostics=diagnostics + [
                CompileDiagnostic(level="error", code="plan_payload_invalid", message=str(exc))
            ],
            validation_errors=[str(exc)],
        )

    validation_errors = validate_compiled_plan(plan)
    if validation_errors:
        return _result(
            "invalid",
            facts,
            plan=plan,
            diagnostics=diagnostics + [
                CompileDiagnostic(level="error", code="plan_validation_failed", message="Plan failed compile-time validation.")
            ],
            validation_errors=validation_errors,
        )
    if plan.mode == "clarification_required":
        return _result(
            "clarification_required",
            facts,
            plan=plan,
            diagnostics=diagnostics + [
                CompileDiagnostic(level="warning", code="missing_required_inputs", message="Required inputs are missing.")
            ],
        )
    if plan.mode == "unsupported":
        return _result(
            "unsupported",
            facts,
            plan=plan,
            diagnostics=diagnostics + [
                CompileDiagnostic(level="warning", code="unsupported_request", message=plan.unsupported_reason or "Unsupported request.")
            ],
        )
    return _result("compiled", facts, plan=plan, diagnostics=diagnostics)


def compile_shadow_diagnostics(result: CompileResult) -> dict[str, Any]:
    """Return passive shadow diagnostics suitable for logs/debug artifacts."""
    plan = result.plan
    return {
        "status": result.status,
        "diagnostics": [diagnostic.model_dump() for diagnostic in result.diagnostics],
        "validation_errors": result.validation_errors,
        "extracted_facts": result.extracted_facts,
        "plan": plan.model_dump(mode="json") if plan else None,
    }
