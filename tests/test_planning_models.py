from __future__ import annotations

import pytest
from pydantic import ValidationError

from strap.planning.models import (
    ArtifactContract,
    DataRequirement,
    FinalResponseContract,
    OutputContract,
    PlanStep,
    RequestPlan,
    is_non_trivial_request,
)


def _artifact_output(artifact_type: str = "solvent_safety_card") -> OutputContract:
    return OutputContract(
        contract_id=f"{artifact_type}_contract",
        artifact_contracts=[ArtifactContract(artifact_type=artifact_type)],
    )


def _data_output() -> OutputContract:
    return OutputContract(
        contract_id="data_contract",
        data_requirements=[DataRequirement(field_path="data.success")],
    )


def _plan(**overrides) -> RequestPlan:
    payload = {
        "plan_id": "plan_test",
        "created_at": "2026-04-26T00:00:00Z",
        "compiler_version": "test",
        "capability_registry_version": "test",
        "planner_model_id": "test/model",
        "user_query": "Show a safety card for THF at 60 C.",
        "mode": "single_tool_or_specialist",
        "intent_family": "safety",
        "complexity": "moderate",
        "steps": [
            PlanStep(
                step_id="safety_card",
                label="Safety card",
                role="safety-analyst",
                execution_kind="tool",
                allowed_tools=["get_solvent_safety_card"],
                output_contracts=[_artifact_output()],
            )
        ],
        "final_response_contract": FinalResponseContract(require_paths=False),
    }
    payload.update(overrides)
    return RequestPlan(**payload)


def test_request_plan_accepts_single_step_artifact_contract():
    plan = _plan()

    assert plan.mode == "single_tool_or_specialist"
    assert plan.steps[0].output_contracts[0].artifact_contracts[0].artifact_type == "solvent_safety_card"
    assert plan.compiler_version == "test"
    assert plan.capability_registry_version == "test"
    assert plan.planner_model_id == "test/model"


def test_request_plan_requires_provenance_fields():
    with pytest.raises(ValidationError, match="compiler_version"):
        _plan(compiler_version="")

    with pytest.raises(ValidationError, match="capability_registry_version"):
        _plan(capability_registry_version="")


def test_plan_step_rejects_missing_output_contract_for_non_synthesis():
    with pytest.raises(ValidationError, match="non-synthesis steps"):
        PlanStep(
            step_id="bad",
            label="Bad",
            role="optimization-engineer",
            execution_kind="tool",
            allowed_tools=["run_waste_management_optimization"],
            output_contracts=[],
        )


def test_single_step_plan_rejects_empty_synthesis_only_shell():
    synthesis_only = PlanStep(
        step_id="synthesis",
        label="Synthesis",
        role="direct_tool",
        execution_kind="synthesis",
        allowed_tools=[],
        output_contracts=[],
    )

    with pytest.raises(ValidationError, match="at least one enforceable step"):
        _plan(steps=[synthesis_only])


def test_plan_rejects_unknown_future_dependency_as_cycle_guard():
    first = PlanStep(
        step_id="first",
        label="First",
        role="safety-analyst",
        execution_kind="tool",
        allowed_tools=["get_solvent_safety_card"],
        output_contracts=[_artifact_output()],
        depends_on=["second"],
    )
    second = PlanStep(
        step_id="second",
        label="Second",
        role="safety-analyst",
        execution_kind="tool",
        allowed_tools=["get_solvent_safety_card"],
        output_contracts=[_artifact_output()],
        depends_on=["first"],
    )

    with pytest.raises(ValidationError, match="unknown or future step"):
        _plan(mode="planned_workflow", steps=[first, second])


def test_plan_accepts_handoff_adapter_role_and_execution_kind():
    plan = _plan(
        mode="planned_workflow",
        intent_family="mixed_workflow",
        steps=[
            PlanStep(
                step_id="separation",
                label="Separation",
                role="separation-engineer",
                execution_kind="tool",
                allowed_tools=["plan_multiple_separation_schemes"],
                output_contracts=[_artifact_output("separation_topk_sequences")],
            ),
            PlanStep(
                step_id="handoff",
                label="Build handoff",
                role="handoff_adapter",
                execution_kind="handoff_adapter",
                allowed_tools=["build_handoff"],
                depends_on=["separation"],
                output_contracts=[_artifact_output("optimization_stage_candidates")],
            ),
        ],
    )

    assert plan.steps[1].role == "handoff_adapter"
    assert plan.steps[1].execution_kind == "handoff_adapter"


def test_clarification_and_unsupported_modes_require_explanatory_fields():
    with pytest.raises(ValidationError, match="missing_inputs"):
        _plan(mode="clarification_required", steps=[])

    with pytest.raises(ValidationError, match="unsupported_reason"):
        _plan(mode="unsupported", steps=[])


def test_non_trivial_request_detector_covers_known_contract_triggers():
    assert is_non_trivial_request("Plot the dynamic-programming state map for LDPE/EVOH/PET.")
    assert is_non_trivial_request("Run a cost-vs-circularity Pareto landscape.")
    assert is_non_trivial_request("Estimate BioSTEAM TEA/LCA for LDPE in cyclohexane.")
    assert not is_non_trivial_request("hello")


def test_tool_step_requires_single_tool_unless_tool_choice_enabled():
    with pytest.raises(ValidationError, match="exactly one allowed tool"):
        PlanStep(
            step_id="choice",
            label="Choice",
            role="safety-analyst",
            execution_kind="tool",
            allowed_tools=["get_solvent_safety_card", "compare_solvent_safety_cards"],
            output_contracts=[_data_output()],
        )

    step = PlanStep(
        step_id="choice",
        label="Choice",
        role="safety-analyst",
        execution_kind="tool",
        allowed_tools=["get_solvent_safety_card", "compare_solvent_safety_cards"],
        output_contracts=[_data_output()],
        allow_tool_choice=True,
    )
    assert step.allow_tool_choice is True
