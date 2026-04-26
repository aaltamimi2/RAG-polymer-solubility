"""Plan validation helpers for compile-only planning."""

from __future__ import annotations

from collections.abc import Mapping

from strap.planning.capability_registry import get_default_capability_registry, validate_plan_against_registry
from strap.planning.models import CapabilitySpec, RequestPlan


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return len(value) == 0
    return False


def validate_required_tool_inputs(
    plan: RequestPlan,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[str]:
    """Validate that planned tool args satisfy selected capability inputs."""
    caps = registry or get_default_capability_registry()
    errors: list[str] = []
    for step in plan.steps:
        if step.execution_kind == "synthesis":
            continue
        for tool_name in step.allowed_tools:
            matching_caps = [
                cap for cap in caps.values()
                if cap.callable_name == tool_name
                and any(
                    artifact.artifact_type in cap.produces
                    for contract in step.output_contracts
                    for artifact in contract.artifact_contracts
                )
            ]
            if not matching_caps:
                continue
            missing = [
                required
                for required in matching_caps[0].required_inputs
                if required not in step.tool_args_template
                or _is_missing_value(step.tool_args_template.get(required))
            ]
            for field in missing:
                errors.append(f"{step.step_id}: missing required input {field} for {tool_name}")
    return errors


def validate_compiled_plan(
    plan: RequestPlan,
    registry: Mapping[str, CapabilitySpec] | None = None,
) -> list[str]:
    """Run all PR 2 compile-time validators."""
    return validate_plan_against_registry(plan, registry) + validate_required_tool_inputs(plan, registry)
