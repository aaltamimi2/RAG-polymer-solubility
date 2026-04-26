from __future__ import annotations

import pytest

from strap.planning.compiler import compile_request
from strap.planning.config import PlannerConfig, get_planner_config, get_typed_planner_mode
from strap.planning.guard import evaluate_plan_tool_call, validate_final_synthesis_sources


FIXED_TIME = "2026-04-26T00:00:00+00:00"


def _state_map_plan():
    result = compile_request("Plot the dynamic-programming state map for LDPE/EVOH/PET.", created_at=FIXED_TIME)
    assert result.plan is not None
    # Keep this guard fixture minimal and independent of compiler heuristics.
    from strap.planning.models import ArtifactContract, OutputContract, PlanStep, RequestPlan

    return RequestPlan(
        plan_id="plan_state_map",
        created_at=FIXED_TIME,
        compiler_version="test",
        capability_registry_version="test",
        user_query="Plot the dynamic-programming state map for LDPE/EVOH/PET.",
        mode="single_tool_or_specialist",
        intent_family="visualization",
        complexity="moderate",
        steps=[
            PlanStep(
                step_id="plot_state_map",
                label="Plot state map",
                role="visualization-specialist",
                execution_kind="tool",
                allowed_tools=["plot_dynamic_programming_separation_options"],
                output_contracts=[
                    OutputContract(
                        contract_id="state_map",
                        artifact_contracts=[ArtifactContract(artifact_type="separation_dp_state_map")],
                    )
                ],
            )
        ],
    )


def _routed_pareto_plan(query: str | None = None):
    result = compile_request(
        query
        or (
            "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year under scenario A, have the "
            "separation engineer propose the top 12 solvent candidates per polymer using the "
            "dynamic-programming planner with temperature recommendations. Then run cost-vs-circularity "
            "Pareto landscape optimization for five fixed feed compositions: 20/60/20, 34/33/33, "
            "60/20/20, 20/20/60, and 5/5/90. Require at least 1 STRAP wash and allow up to 2 washes. "
            "Save one PNG per composition and one combined comparison plot showing all feasible points."
        ),
        created_at=FIXED_TIME,
    )
    assert result.status == "compiled"
    assert result.plan is not None
    return result.plan


def test_config_parses_modes_and_selected_sets():
    cfg = get_planner_config({
        "DISSOLVE_TYPED_PLANNER": "enforce_selected",
        "DISSOLVE_TYPED_PLANNER_ENFORCE_ARTIFACTS": "separation_dp_state_map, optimization_pareto_slices",
        "DISSOLVE_TYPED_PLANNER_ENFORCE_WORKFLOWS": "routed_optimization",
    })

    assert cfg.mode == "enforce_selected"
    assert cfg.selected_enforcement_artifacts == {"separation_dp_state_map", "optimization_pareto_slices"}
    assert cfg.selected_enforcement_workflows == {"routed_optimization"}


def test_phase7_default_selected_artifacts_cover_new_domains():
    cfg = get_planner_config({"DISSOLVE_TYPED_PLANNER": "enforce_selected"})

    assert {
        "solvent_safety_card",
        "hsp_red_heatmap",
        "biosteam_tea_lca_result",
        "separation_selectivity_heatmap",
    } <= cfg.selected_enforcement_artifacts


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Invalid DISSOLVE_TYPED_PLANNER"):
        get_typed_planner_mode({"DISSOLVE_TYPED_PLANNER": "block_everything"})


def test_off_and_shadow_modes_are_noop_for_tool_guard():
    plan = _state_map_plan()

    for mode in ("off", "shadow"):
        decision = evaluate_plan_tool_call(
            plan,
            "plot_state_map",
            "plot_solubility_vs_temperature",
            {},
            config=PlannerConfig(mode=mode),
        )
        assert decision.outcome == "not_applicable"
        assert decision.selected is False
        assert decision.enforcement_scope == "none"


def test_enforce_mode_is_parsed_but_not_behaviorally_enabled_until_executor_phase():
    plan = _state_map_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "plot_state_map",
        "plot_solubility_vs_temperature",
        {},
        config=PlannerConfig(mode="enforce"),
    )
    validation = validate_final_synthesis_sources(
        _routed_pareto_plan(),
        [],
        {},
        config=PlannerConfig(mode="enforce"),
    )

    assert decision.outcome == "not_applicable"
    assert decision.selected is False
    assert decision.enforcement_scope == "none"
    assert validation.status == "not_applicable"
    assert validation.selected is False


def test_dp_state_map_blocks_solubility_plotter_when_selected():
    plan = _state_map_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "plot_state_map",
        "plot_solubility_vs_temperature",
        {},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"separation_dp_state_map"},
        ),
    )

    assert decision.outcome == "block_retry_same_step"
    assert decision.selected is True
    assert decision.enforcement_scope == "selected"
    assert "tool_not_allowed_for_step" in decision.failed_checks


def test_dp_state_map_allows_dp_plotter_when_selected():
    plan = _state_map_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "plot_state_map",
        "plot_dynamic_programming_separation_options",
        {"source_step_id": "separation_candidates"},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"separation_dp_state_map"},
        ),
    )

    assert decision.outcome == "allow"
    assert decision.required_artifacts == ["separation_dp_state_map"]
    assert "separation_dp_state_map" in decision.produced_artifacts


def test_pareto_slices_blocks_single_slice_tool_when_selected():
    plan = _routed_pareto_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "optimize_slices",
        "run_waste_management_pareto",
        {},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
    )

    assert decision.outcome == "block_retry_same_step"
    assert "tool_not_allowed_for_step" in decision.failed_checks


def test_pareto_slices_allows_slice_tool_when_selected():
    plan = _routed_pareto_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "optimize_slices",
        "run_waste_management_pareto_slices",
        {"source_step_id": "build_optimization_handoff"},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
    )

    assert decision.outcome == "allow"
    assert "optimization_pareto_slices" in decision.required_artifacts
    assert "optimization_pareto_slices" in decision.produced_artifacts


def test_visualization_authoritative_payload_check_blocks_missing_source():
    plan = _routed_pareto_plan(
        "For a mixed plastic feedstock of 8000 tonnes/year composed of 20% LDPE, 60% EVOH, and "
        "20% PET under scenario A, have the separation engineer propose the top 12 solvent "
        "candidates per polymer using the dynamic-programming planner. Then pass those candidates "
        "to the optimization engineer to run a cost-vs-circularity Pareto landscape with 100 points. "
        "Finally, plot all feasible points and highlight the frontier."
    )
    decision = evaluate_plan_tool_call(
        plan,
        "plot_optimization",
        "plot_optimization_pareto_front",
        {},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_plot"},
        ),
    )

    assert decision.outcome == "block_retry_same_step"
    assert "missing_authoritative_visualization_source" in decision.failed_checks


def test_visualization_authoritative_payload_check_allows_source():
    plan = _routed_pareto_plan(
        "For a mixed plastic feedstock of 8000 tonnes/year composed of 20% LDPE, 60% EVOH, and "
        "20% PET under scenario A, have the separation engineer propose the top 12 solvent "
        "candidates per polymer using the dynamic-programming planner. Then pass those candidates "
        "to the optimization engineer to run a cost-vs-circularity Pareto landscape with 100 points. "
        "Finally, plot all feasible points and highlight the frontier."
    )
    decision = evaluate_plan_tool_call(
        plan,
        "plot_optimization",
        "plot_optimization_pareto_front",
        {"source_payload_path": "architecture/test_results/example.json"},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_plot"},
        ),
    )

    assert decision.outcome == "allow"


def test_non_selected_artifact_returns_not_applicable():
    plan = _state_map_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "plot_state_map",
        "plot_solubility_vs_temperature",
        {},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
    )

    assert decision.outcome == "not_applicable"
    assert decision.selected is False


def test_missing_active_step_is_not_applicable_unless_workflow_selected():
    plan = _state_map_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "missing_step",
        "plot_dynamic_programming_separation_options",
        {},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"separation_dp_state_map"},
        ),
    )

    assert decision.outcome == "not_applicable"
    assert decision.selected is False
    assert decision.enforcement_scope == "none"


def test_missing_active_step_in_enforce_mode_has_no_scope_until_executor_phase():
    plan = _state_map_plan()
    decision = evaluate_plan_tool_call(
        plan,
        "missing_step",
        "plot_dynamic_programming_separation_options",
        {},
        config=PlannerConfig(mode="enforce"),
    )

    assert decision.outcome == "not_applicable"
    assert decision.selected is False
    assert decision.enforcement_scope == "none"


def test_final_synthesis_validator_fails_when_optimizer_artifacts_absent():
    plan = _routed_pareto_plan()
    validation = validate_final_synthesis_sources(
        plan,
        [{"artifact_type": "separation_topk_sequences"}],
        {"source_artifact_types": ["separation_topk_sequences"]},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
    )

    assert validation.status == "failed"
    assert "required_optimizer_artifacts_absent" in validation.failed_checks


def test_final_synthesis_validator_passes_with_optimizer_artifacts():
    plan = _routed_pareto_plan()
    validation = validate_final_synthesis_sources(
        plan,
        [{"artifact_type": "optimization_pareto_slices"}],
        {"source_artifact_types": ["optimization_pareto_slices"]},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
    )

    assert validation.status == "passed"


def test_final_synthesis_validator_requires_explicit_optimizer_artifact_citation():
    plan = _routed_pareto_plan()
    validation = validate_final_synthesis_sources(
        plan,
        [{"artifact_type": "optimization_pareto_slices"}],
        {},
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
    )

    assert validation.status == "failed"
    assert "optimizer_artifacts_not_cited" in validation.failed_checks
