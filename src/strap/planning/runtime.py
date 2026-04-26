"""Opt-in runtime bridge for typed planning.

This module does not replace DISSOLVE orchestration. It is a narrow PR 5 bridge
that can compile a query, check selected-enforcement activation, execute through
the deterministic PR 4 executor with explicit wrappers, and persist diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from strap.planning.compiler import CompileResult, PlannerBackend, compile_request
from strap.planning.config import PlannerConfig, get_planner_config
from strap.planning.executor import StepCallable, run_plan
from strap.planning.models import ExecutionLedger, PlanningModel, RequestPlan
from strap.planning.runtime_persistence import RuntimeArtifactManifest, persist_runtime_artifacts
from strap.planning.runtime_wrappers import get_runtime_callable_registry
from strap.planning.validators import validate_compiled_plan


TypedRuntimeStatus = Literal["executed", "legacy_fallback", "typed_failure"]


class TypedRuntimeResult(PlanningModel):
    schema_version: Literal["1.0"] = "1.0"
    status: TypedRuntimeStatus
    reason: str
    selected: bool = False
    compile_result: CompileResult
    plan: RequestPlan | None = None
    ledger: ExecutionLedger | None = None
    manifest: RuntimeArtifactManifest | None = None
    diagnostics: list[str] = Field(default_factory=list)


def _required_artifacts(plan: RequestPlan) -> set[str]:
    return {
        artifact.artifact_type
        for step in plan.steps
        for contract in step.output_contracts
        for artifact in contract.artifact_contracts
        if artifact.required
    }


def _selected_artifacts(plan: RequestPlan, config: PlannerConfig) -> set[str]:
    if config.mode != "enforce_selected":
        return set()
    required = _required_artifacts(plan)
    selected = required & config.selected_enforcement_artifacts
    if plan.workflow_id and plan.workflow_id in config.selected_enforcement_workflows:
        selected.update(required)
    return selected


def _compile_failure_selected(result: CompileResult, config: PlannerConfig) -> bool:
    if config.mode != "enforce_selected":
        return False
    if result.plan and result.plan.workflow_id and result.plan.workflow_id in config.selected_enforcement_workflows:
        return True
    requested = set(result.extracted_facts.get("requested_artifact_types") or [])
    markers = set(result.extracted_facts.get("workflow_markers") or [])
    if (
        "routed_optimization" in config.selected_enforcement_workflows
        and {"separation", "handoff", "optimization"} <= markers
    ):
        return True
    if (
        "routed_optimization_slices" in config.selected_enforcement_workflows
        and {"separation", "optimization", "multi_slice"} <= markers
    ):
        return True
    return bool(requested & config.selected_enforcement_artifacts)


def _required_callable_names(plan: RequestPlan) -> set[str]:
    return {
        step.allowed_tools[0]
        for step in plan.steps
        if step.allowed_tools
    }


def _missing_wrappers(plan: RequestPlan, callable_registry: Mapping[str, StepCallable]) -> list[str]:
    return sorted(_required_callable_names(plan) - set(callable_registry))


def _persist_if_requested(
    *,
    persist: bool,
    query: str,
    compile_result: CompileResult,
    config: PlannerConfig,
    plan: RequestPlan | None,
    ledger: ExecutionLedger | None,
    output_root: str | None,
) -> RuntimeArtifactManifest | None:
    if not persist:
        return None
    return persist_runtime_artifacts(
        query=query,
        compile_result=compile_result,
        config=config,
        plan=plan,
        ledger=ledger,
        output_root=output_root,
    )


def run_typed_runtime(
    query: str,
    *,
    config: PlannerConfig | None = None,
    context: dict | None = None,
    planner_backend: PlannerBackend | None = None,
    callable_registry: Mapping[str, StepCallable] | None = None,
    output_root: str | None = None,
    created_at: str | None = None,
    persist: bool = True,
) -> TypedRuntimeResult:
    """Compile and maybe execute a query through the typed runtime bridge."""
    config = config or get_planner_config()
    registry = dict(callable_registry) if callable_registry is not None else get_runtime_callable_registry()
    compile_result = compile_request(query, context=context, planner_backend=planner_backend, created_at=created_at)
    plan = compile_result.plan

    if config.mode in {"off", "shadow"}:
        return TypedRuntimeResult(
            status="legacy_fallback",
            reason=f"Typed runtime inactive in planner mode {config.mode!r}.",
            selected=False,
            compile_result=compile_result,
            plan=plan,
        )

    if config.mode == "enforce":
        manifest = _persist_if_requested(
            persist=persist,
            query=query,
            compile_result=compile_result,
            config=config,
            plan=plan,
            ledger=None,
            output_root=output_root,
        )
        return TypedRuntimeResult(
            status="typed_failure",
            reason="Full typed runtime enforcement is not implemented; use enforce_selected.",
            selected=True,
            compile_result=compile_result,
            plan=plan,
            manifest=manifest,
            diagnostics=["full_enforce_not_implemented"],
        )

    if compile_result.status != "compiled" or plan is None:
        selected = _compile_failure_selected(compile_result, config)
        manifest = _persist_if_requested(
            persist=persist and selected,
            query=query,
            compile_result=compile_result,
            config=config,
            plan=plan,
            ledger=None,
            output_root=output_root,
        )
        return TypedRuntimeResult(
            status="typed_failure" if selected else "legacy_fallback",
            reason=(
                f"Compile result {compile_result.status!r} for selected enforcement target."
                if selected
                else f"Compile result {compile_result.status!r}; falling back to legacy runtime."
            ),
            selected=selected,
            compile_result=compile_result,
            plan=plan,
            manifest=manifest,
            diagnostics=compile_result.validation_errors,
        )

    selected = bool(_selected_artifacts(plan, config))
    if not selected:
        return TypedRuntimeResult(
            status="legacy_fallback",
            reason="Compiled plan is not selected for typed enforcement.",
            selected=False,
            compile_result=compile_result,
            plan=plan,
        )

    validation_errors = validate_compiled_plan(plan)
    if validation_errors:
        manifest = _persist_if_requested(
            persist=persist,
            query=query,
            compile_result=compile_result,
            config=config,
            plan=plan,
            ledger=None,
            output_root=output_root,
        )
        return TypedRuntimeResult(
            status="typed_failure",
            reason="Compiled selected plan failed validation.",
            selected=True,
            compile_result=compile_result,
            plan=plan,
            manifest=manifest,
            diagnostics=validation_errors,
        )

    missing = _missing_wrappers(plan, registry)
    if missing:
        manifest = _persist_if_requested(
            persist=persist,
            query=query,
            compile_result=compile_result,
            config=config,
            plan=plan,
            ledger=None,
            output_root=output_root,
        )
        return TypedRuntimeResult(
            status="typed_failure",
            reason="Selected typed runtime target is missing registered wrappers.",
            selected=True,
            compile_result=compile_result,
            plan=plan,
            manifest=manifest,
            diagnostics=[f"missing_wrapper:{name}" for name in missing],
        )

    ledger = run_plan(plan, registry, config)
    manifest = _persist_if_requested(
        persist=persist,
        query=query,
        compile_result=compile_result,
        config=config,
        plan=plan,
        ledger=ledger,
        output_root=output_root,
    )
    return TypedRuntimeResult(
        status="executed" if ledger.status == "succeeded" else "typed_failure",
        reason="Typed runtime execution completed." if ledger.status == "succeeded" else "Typed runtime execution failed.",
        selected=True,
        compile_result=compile_result,
        plan=plan,
        ledger=ledger,
        manifest=manifest,
        diagnostics=[] if ledger.status == "succeeded" else [str(ledger.final_contract_status)],
    )
