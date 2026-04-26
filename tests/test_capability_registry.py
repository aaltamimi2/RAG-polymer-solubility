from __future__ import annotations

from strap.planning.capability_registry import (
    ARTIFACT_TYPES,
    assert_valid_capability_registry,
    capabilities_for_artifact,
    exported_tool_names,
    get_default_capability_registry,
    role_allowed_tools,
    validate_capability_registry,
    validate_plan_against_registry,
)
from strap.planning.models import ArtifactContract, CapabilitySpec, OutputContract, PlanStep, RequestPlan


def _output(artifact_type: str) -> OutputContract:
    return OutputContract(
        contract_id=f"{artifact_type}_contract",
        artifact_contracts=[ArtifactContract(artifact_type=artifact_type)],
    )


def _plan_for_step(step: PlanStep) -> RequestPlan:
    return RequestPlan(
        plan_id="plan_registry_test",
        created_at="2026-04-26T00:00:00Z",
        compiler_version="test",
        capability_registry_version="test",
        planner_model_id=None,
        user_query="test",
        mode="single_tool_or_specialist",
        intent_family="optimization",
        complexity="moderate",
        steps=[step],
    )


def test_default_registry_is_consistent_with_exported_tools():
    assert_valid_capability_registry()

    registry = get_default_capability_registry()
    exported = exported_tool_names()
    covered = {cap.callable_name for cap in registry.values() if cap.callable_name in exported}

    assert exported <= covered
    assert "run_waste_management_pareto_slices" in covered
    assert "get_solvent_safety_card" in covered


def test_p0_artifact_capabilities_are_explicit():
    assert "optimization_pareto_slices" in ARTIFACT_TYPES

    slice_caps = capabilities_for_artifact("optimization_pareto_slices")
    assert [cap.callable_name for cap in slice_caps] == ["run_waste_management_pareto_slices"]
    assert slice_caps[0].supports_multislice is True
    assert slice_caps[0].legacy_unplanned is False

    safety_caps = capabilities_for_artifact("solvent_safety_card")
    assert any(cap.callable_name == "get_solvent_safety_card" for cap in safety_caps)


def test_solubility_capabilities_are_explicit_and_reject_separation_artifacts():
    curve_caps = capabilities_for_artifact("solubility_curve")
    table_caps = capabilities_for_artifact("solubility_table")

    assert any(cap.callable_name == "plot_solubility_vs_temperature" for cap in curve_caps)
    assert any(cap.callable_name == "plot_solubility_vs_temperature_interactive" for cap in curve_caps)
    assert any(cap.callable_name == "predict_solubility_range" for cap in table_caps)
    assert all("separation_dp_state_map" in cap.rejects for cap in curve_caps)


def test_role_allowed_tools_are_derived_from_subagent_exports():
    allowed = role_allowed_tools()

    assert "run_waste_management_pareto" in allowed["optimization-engineer"]
    assert "run_waste_management_pareto" not in allowed["safety-analyst"]
    assert "build_handoff" in allowed["handoff_adapter"]


def test_plan_registry_validation_rejects_unknown_tool():
    step = PlanStep(
        step_id="bad_tool",
        label="Bad tool",
        role="optimization-engineer",
        execution_kind="tool",
        allowed_tools=["not_a_real_tool"],
        output_contracts=[_output("legacy_tool_result")],
    )
    errors = validate_plan_against_registry(_plan_for_step(step))

    assert any("unknown tool not_a_real_tool" in error for error in errors)


def test_plan_registry_validation_rejects_unknown_artifact_type():
    step = PlanStep(
        step_id="bad_artifact",
        label="Bad artifact",
        role="optimization-engineer",
        execution_kind="tool",
        allowed_tools=["run_waste_management_pareto"],
        output_contracts=[_output("unknown_artifact")],
    )
    errors = validate_plan_against_registry(_plan_for_step(step))

    assert any("unknown output artifact unknown_artifact" in error for error in errors)


def test_plan_registry_validation_rejects_role_tool_mismatch():
    step = PlanStep(
        step_id="mismatch",
        label="Mismatch",
        role="safety-analyst",
        execution_kind="tool",
        allowed_tools=["run_waste_management_pareto"],
        output_contracts=[_output("optimization_pareto_front")],
    )
    errors = validate_plan_against_registry(_plan_for_step(step))

    assert any("not allowed for role safety-analyst" in error for error in errors)


def test_plan_registry_validation_requires_allowed_tool_to_produce_artifact():
    step = PlanStep(
        step_id="wrong_producer",
        label="Wrong producer",
        role="optimization-engineer",
        execution_kind="tool",
        allowed_tools=["run_waste_management_optimization"],
        output_contracts=[_output("optimization_pareto_slices")],
    )
    errors = validate_plan_against_registry(_plan_for_step(step))

    assert any("no allowed tool can produce artifact optimization_pareto_slices" in error for error in errors)


def test_capability_registry_rejects_stale_capability_names():
    registry = {
        "bad.capability": CapabilitySpec(
            capability_id="bad.capability",
            owner="optimization-engineer",
            callable_name="missing_tool",
            callable_kind="tool",
            produces=["optimization_point_result"],
        )
    }
    errors = validate_capability_registry(registry)

    assert any("missing exported tool missing_tool" in error for error in errors)
    assert any("exported tool has no capability" in error for error in errors)
