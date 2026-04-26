"""Pure selected-enforcement guard functions for typed plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from strap.planning.capability_registry import CapabilitySpec, get_default_capability_registry
from strap.planning.config import PlannerConfig
from strap.planning.models import ArtifactFrame, PlanningModel, PlanStep, RequestPlan


GuardOutcome = Literal[
    "allow",
    "allow_with_arg_repair",
    "block_retry_same_step",
    "block_plan_failure",
    "not_applicable",
]


class PlanGuardDecision(PlanningModel):
    outcome: GuardOutcome
    reason: str
    active_step_id: str | None = None
    tool_name: str | None = None
    required_artifacts: list[str] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    repaired_args: dict[str, Any] | None = None
    enforcement_scope: Literal["none", "selected", "full"] = "none"
    selected: bool = False


class FinalSynthesisValidation(PlanningModel):
    status: Literal["passed", "failed", "not_applicable"]
    reason: str
    required_artifacts: list[str] = Field(default_factory=list)
    available_artifacts: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    enforcement_scope: Literal["none", "selected", "full"] = "none"
    selected: bool = False


def _default_config() -> PlannerConfig:
    return PlannerConfig(mode="off")


def _step_by_id(plan: RequestPlan, active_step_id: str) -> PlanStep | None:
    return next((step for step in plan.steps if step.step_id == active_step_id), None)


def _required_artifacts(step: PlanStep) -> list[str]:
    return sorted({
        artifact.artifact_type
        for contract in step.output_contracts
        for artifact in contract.artifact_contracts
        if artifact.required
    })


def _validation_checks(step: PlanStep) -> set[str]:
    return {
        check
        for contract in step.output_contracts
        for check in contract.validation_checks
    }


def _matching_capabilities(
    tool_name: str,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[CapabilitySpec]:
    caps = registry or get_default_capability_registry()
    return [
        cap
        for cap in caps.values()
        if cap.callable_name == tool_name
    ]


def _produced_artifacts(
    tool_name: str,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[str]:
    produced: set[str] = set()
    for cap in _matching_capabilities(tool_name, registry):
        produced.update(cap.produces)
    return sorted(produced)


def _tool_rejects_artifact(
    tool_name: str,
    artifact_type: str,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> bool:
    return any(
        artifact_type in cap.rejects
        for cap in _matching_capabilities(tool_name, registry)
    )


def _selected_required_artifacts(
    required_artifacts: list[str],
    plan: RequestPlan,
    config: PlannerConfig,
) -> list[str]:
    if config.mode == "enforce":
        # Full enforcement is intentionally parsed but not behaviorally enabled
        # until the deterministic executor lands in the next phase.
        return []
    if config.mode != "enforce_selected":
        return []
    selected = set(required_artifacts) & set(config.selected_enforcement_artifacts)
    if plan.workflow_id and plan.workflow_id in config.selected_enforcement_workflows:
        selected.update(required_artifacts)
    return sorted(selected)


def _scope(config: PlannerConfig, selected: bool) -> Literal["none", "selected", "full"]:
    if config.mode == "enforce":
        # Parsed only in PR 3; full enforcement starts with the executor phase.
        return "none"
    if config.mode == "enforce_selected" and selected:
        return "selected"
    return "none"


def _has_authoritative_source(tool_args: Mapping[str, Any]) -> bool:
    for key in (
        "source_handoff_id",
        "source_handoff_ids",
        "source_step_id",
        "source_payload_path",
        "payload_path",
        "authoritative_payload_path",
    ):
        value = tool_args.get(key)
        if value:
            return True
    return False


def evaluate_plan_tool_call(
    plan: RequestPlan,
    active_step_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    config: PlannerConfig | None = None,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> PlanGuardDecision:
    """Evaluate a proposed tool call against the active plan step.

    This is a pure function. It does not mutate state, execute tools, or
    determine the next step.
    """
    config = config or _default_config()
    step = _step_by_id(plan, active_step_id)
    if step is None:
        selected_workflow = (
            config.mode == "enforce_selected"
            and plan.workflow_id is not None
            and plan.workflow_id in config.selected_enforcement_workflows
        )
        return PlanGuardDecision(
            outcome="block_plan_failure" if selected_workflow else "not_applicable",
            reason=f"Active step {active_step_id!r} is not present in plan.",
            active_step_id=active_step_id,
            tool_name=tool_name,
            failed_checks=["active_step_missing"],
            enforcement_scope=_scope(config, selected=selected_workflow),
            selected=selected_workflow,
        )

    required = _required_artifacts(step)
    produced = _produced_artifacts(tool_name, registry)
    selected_required = _selected_required_artifacts(required, plan, config)
    selected = bool(selected_required) or config.mode == "enforce"
    scope = _scope(config, selected)

    if config.mode in {"off", "shadow", "enforce"}:
        return PlanGuardDecision(
            outcome="not_applicable",
            reason=(
                "Full typed planner enforcement is parsed but not implemented until the executor phase."
                if config.mode == "enforce"
                else f"Typed planner mode {config.mode!r} does not block tool calls."
            ),
            active_step_id=active_step_id,
            tool_name=tool_name,
            required_artifacts=required,
            produced_artifacts=produced,
            enforcement_scope="none",
            selected=False,
        )

    if not selected:
        return PlanGuardDecision(
            outcome="not_applicable",
            reason="Active step artifacts/workflow are not selected for enforcement.",
            active_step_id=active_step_id,
            tool_name=tool_name,
            required_artifacts=required,
            produced_artifacts=produced,
            enforcement_scope="none",
            selected=False,
        )

    if step.allowed_tools and tool_name not in step.allowed_tools:
        return PlanGuardDecision(
            outcome="block_retry_same_step",
            reason=f"Tool {tool_name!r} is not allowed for active step {active_step_id!r}.",
            active_step_id=active_step_id,
            tool_name=tool_name,
            required_artifacts=selected_required,
            produced_artifacts=produced,
            failed_checks=["tool_not_allowed_for_step"],
            enforcement_scope=scope,
            selected=True,
        )

    for artifact_type in selected_required:
        if artifact_type not in produced:
            return PlanGuardDecision(
                outcome="block_retry_same_step",
                reason=f"Tool {tool_name!r} cannot produce required artifact {artifact_type!r}.",
                active_step_id=active_step_id,
                tool_name=tool_name,
                required_artifacts=selected_required,
                produced_artifacts=produced,
                failed_checks=["required_artifact_not_produced"],
                enforcement_scope=scope,
                selected=True,
            )
        if _tool_rejects_artifact(tool_name, artifact_type, registry):
            return PlanGuardDecision(
                outcome="block_retry_same_step",
                reason=f"Tool {tool_name!r} explicitly rejects required artifact {artifact_type!r}.",
                active_step_id=active_step_id,
                tool_name=tool_name,
                required_artifacts=selected_required,
                produced_artifacts=produced,
                failed_checks=["tool_rejects_required_artifact"],
                enforcement_scope=scope,
                selected=True,
            )

    checks = _validation_checks(step)
    if "visualization_from_authoritative_payload" in checks and not _has_authoritative_source(tool_args):
        return PlanGuardDecision(
            outcome="block_retry_same_step",
            reason="Visualization step requires an authoritative source handoff, step id, or payload path.",
            active_step_id=active_step_id,
            tool_name=tool_name,
            required_artifacts=selected_required,
            produced_artifacts=produced,
            failed_checks=["missing_authoritative_visualization_source"],
            enforcement_scope=scope,
            selected=True,
        )

    return PlanGuardDecision(
        outcome="allow",
        reason="Tool call satisfies selected plan contract.",
        active_step_id=active_step_id,
        tool_name=tool_name,
        required_artifacts=selected_required,
        produced_artifacts=produced,
        enforcement_scope=scope,
        selected=True,
    )


def _available_artifact_types(ledger_or_artifacts: object) -> list[str]:
    if ledger_or_artifacts is None:
        return []
    if isinstance(ledger_or_artifacts, list):
        values = ledger_or_artifacts
    elif isinstance(ledger_or_artifacts, dict):
        values = ledger_or_artifacts.get("artifacts", [])
    else:
        values = getattr(ledger_or_artifacts, "artifacts", [])
    artifacts: set[str] = set()
    for item in values:
        if isinstance(item, ArtifactFrame):
            artifacts.add(item.artifact_type)
        elif isinstance(item, dict) and item.get("artifact_type"):
            artifacts.add(str(item["artifact_type"]))
        elif hasattr(item, "artifact_type"):
            artifacts.add(str(getattr(item, "artifact_type")))
    return sorted(artifacts)


def _required_optimization_artifacts(plan: RequestPlan) -> list[str]:
    return sorted({
        artifact.artifact_type
        for step in plan.steps
        if step.role == "optimization-engineer"
        for contract in step.output_contracts
        for artifact in contract.artifact_contracts
        if artifact.required and artifact.artifact_type.startswith("optimization_")
    })


def validate_final_synthesis_sources(
    plan: RequestPlan,
    ledger_or_artifacts: object,
    final_payload: object,
    *,
    config: PlannerConfig | None = None,
) -> FinalSynthesisValidation:
    """Conservatively validate final synthesis source availability.

    This does not judge prose quality. It checks whether required optimizer
    artifacts exist before final synthesis can claim optimization results.
    """
    config = config or _default_config()
    required = _required_optimization_artifacts(plan)
    selected_required = _selected_required_artifacts(required, plan, config)
    selected = bool(selected_required) or config.mode == "enforce"
    scope = _scope(config, selected)
    available = _available_artifact_types(ledger_or_artifacts)

    if config.mode in {"off", "shadow", "enforce"}:
        return FinalSynthesisValidation(
            status="not_applicable",
            reason=(
                "Full typed planner enforcement is parsed but not implemented until the executor phase."
                if config.mode == "enforce"
                else f"Typed planner mode {config.mode!r} does not block final synthesis."
            ),
            required_artifacts=required,
            available_artifacts=available,
            enforcement_scope="none",
            selected=False,
        )
    if not selected:
        return FinalSynthesisValidation(
            status="not_applicable",
            reason="No final synthesis artifacts/workflows are selected for enforcement.",
            required_artifacts=required,
            available_artifacts=available,
            enforcement_scope="none",
            selected=False,
        )

    missing = sorted(set(selected_required) - set(available))
    if missing:
        return FinalSynthesisValidation(
            status="failed",
            reason="Required optimization artifacts are absent from synthesis sources.",
            required_artifacts=selected_required,
            available_artifacts=available,
            failed_checks=["required_optimizer_artifacts_absent"],
            enforcement_scope=scope,
            selected=True,
        )

    if isinstance(final_payload, dict):
        cited = set(final_payload.get("artifact_types") or final_payload.get("source_artifact_types") or [])
        if not cited:
            return FinalSynthesisValidation(
                status="failed",
                reason="Final payload must cite optimizer artifact types under selected enforcement.",
                required_artifacts=selected_required,
                available_artifacts=available,
                failed_checks=["optimizer_artifacts_not_cited"],
                enforcement_scope=scope,
                selected=True,
            )
        if not (set(selected_required) & cited):
            return FinalSynthesisValidation(
                status="failed",
                reason="Final payload does not cite required optimizer artifacts.",
                required_artifacts=selected_required,
                available_artifacts=available,
                failed_checks=["optimizer_artifacts_not_cited"],
                enforcement_scope=scope,
                selected=True,
            )

    return FinalSynthesisValidation(
        status="passed",
        reason="Required optimization artifacts are available for synthesis.",
        required_artifacts=selected_required,
        available_artifacts=available,
        enforcement_scope=scope,
        selected=True,
    )
