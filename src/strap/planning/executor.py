"""Deterministic execution shell for compiled RequestPlan objects.

This module is intentionally isolated from LangGraph, model calls, prompt
generation, and live domain tools. It executes only injected callables that
return the canonical StepCallableResult envelope.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from strap.planning.config import PlannerConfig
from strap.planning.guard import PlanGuardDecision, evaluate_plan_tool_call
from strap.planning.models import (
    ArtifactFrame,
    ExecutionLedger,
    PlanStep,
    PlanningModel,
    RequestPlan,
    StepExecutionRecord,
)


VerificationStatus = Literal["passed", "failed"]
StepCallable = Callable[[PlanStep, ExecutionLedger], "StepCallableResult"]


class StepCallableResult(PlanningModel):
    success: bool = True
    artifacts: list[ArtifactFrame] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class VerificationResult(PlanningModel):
    status: VerificationStatus
    failed_checks: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_execution_ledger(
    plan: RequestPlan,
    *,
    run_id: str | None = None,
    started_at: str | None = None,
) -> ExecutionLedger:
    return ExecutionLedger(
        plan_id=plan.plan_id,
        run_id=run_id or f"run_{uuid.uuid4().hex[:12]}",
        status="running",
        started_at=started_at or _now_iso(),
    )


def _latest_records_by_step(ledger: ExecutionLedger) -> dict[str, StepExecutionRecord]:
    records: dict[str, StepExecutionRecord] = {}
    for record in ledger.step_records:
        records[record.step_id] = record
    return records


def _successful_steps(ledger: ExecutionLedger) -> set[str]:
    latest = _latest_records_by_step(ledger)
    return {step_id for step_id, record in latest.items() if record.status == "succeeded"}


def _failed_steps(ledger: ExecutionLedger) -> set[str]:
    latest = _latest_records_by_step(ledger)
    return {step_id for step_id, record in latest.items() if record.status == "failed"}


def _attempt_count(ledger: ExecutionLedger, step_id: str) -> int:
    return sum(1 for record in ledger.step_records if record.step_id == step_id and record.status != "running")


def validate_execution_plan(plan: RequestPlan) -> list[str]:
    """Runtime safety validation independent of Pydantic construction."""
    errors: list[str] = []
    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        errors.append("duplicate_step_id")
    known = set(step_ids)
    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in known:
                errors.append(f"{step.step_id}: unknown dependency {dep}")
            if dep == step.step_id:
                errors.append(f"{step.step_id}: self dependency")

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {step.step_id: step for step in plan.steps}

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            errors.append(f"cycle detected at {step_id}")
            return
        visiting.add(step_id)
        for dep in by_id[step_id].depends_on:
            if dep in by_id:
                visit(dep)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_ids:
        visit(step_id)
    return sorted(set(errors))


def next_runnable_step(plan: RequestPlan, ledger: ExecutionLedger) -> PlanStep | None:
    if ledger.status in {"failed", "succeeded"}:
        return None
    succeeded = _successful_steps(ledger)
    failed = _failed_steps(ledger)
    for step in plan.steps:
        if step.step_id in succeeded or step.step_id in failed:
            continue
        if all(dep in succeeded for dep in step.depends_on):
            return step
    return None


def _callable_name_for_step(step: PlanStep) -> str:
    return step.allowed_tools[0] if step.allowed_tools else step.step_id


def authorize_step(
    plan: RequestPlan,
    step: PlanStep,
    ledger: ExecutionLedger,
    config: PlannerConfig,
) -> PlanGuardDecision:
    return evaluate_plan_tool_call(
        plan,
        step.step_id,
        _callable_name_for_step(step),
        step.tool_args_template,
        config=config,
    )


def _artifacts_for_contract(
    ledger: ExecutionLedger,
    *,
    artifact_type: str,
    source_step_id: str | None,
) -> list[ArtifactFrame]:
    return [
        artifact
        for artifact in ledger.artifacts
        if artifact.artifact_type == artifact_type
        and (source_step_id is None or artifact.source_step_id == source_step_id)
    ]


def _all_artifacts_of_type(ledger: ExecutionLedger, artifact_type: str) -> list[ArtifactFrame]:
    return [artifact for artifact in ledger.artifacts if artifact.artifact_type == artifact_type]


def _input_artifact_ids(step: PlanStep, ledger: ExecutionLedger) -> list[str]:
    ids: list[str] = []
    for contract in step.input_contracts:
        ids.extend(
            artifact.artifact_id
            for artifact in _artifacts_for_contract(
                ledger,
                artifact_type=contract.artifact_type,
                source_step_id=contract.source_step_id,
            )
        )
    return sorted(set(ids))


def _artifact_type_counts(artifacts: list[ArtifactFrame]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        counts[artifact.artifact_type] = counts.get(artifact.artifact_type, 0) + 1
    return counts


def _normalize_output_artifacts(step: PlanStep, result: StepCallableResult) -> list[ArtifactFrame]:
    normalized: list[ArtifactFrame] = []
    for artifact in result.artifacts:
        if artifact.source_step_id is None:
            normalized.append(artifact.model_copy(update={"source_step_id": step.step_id}))
        else:
            normalized.append(artifact)
    return normalized


def verify_step_outputs(
    step: PlanStep,
    result: StepCallableResult,
    ledger: ExecutionLedger,
) -> VerificationResult:
    failed: list[str] = []
    if not result.success:
        failed.append("callable_reported_failure")

    for input_contract in step.input_contracts:
        matching = _artifacts_for_contract(
            ledger,
            artifact_type=input_contract.artifact_type,
            source_step_id=input_contract.source_step_id,
        )
        if input_contract.required and not matching:
            same_type = _all_artifacts_of_type(ledger, input_contract.artifact_type)
            failed.append("source_step_id_mismatch" if same_type and input_contract.source_step_id else "dependency_output_missing")

    artifacts = _normalize_output_artifacts(step, result)
    produced_types = sorted({artifact.artifact_type for artifact in artifacts})
    counts = _artifact_type_counts(artifacts)
    allowed_types: set[str] = set()
    required_types: list[str] = []
    for contract in step.output_contracts:
        for artifact_contract in contract.artifact_contracts:
            allowed_types.add(artifact_contract.artifact_type)
            if artifact_contract.required:
                required_types.append(artifact_contract.artifact_type)
                if counts.get(artifact_contract.artifact_type, 0) < artifact_contract.count.min:
                    failed.append("required_artifact_missing")
                if artifact_contract.count.max is not None and counts.get(artifact_contract.artifact_type, 0) > artifact_contract.count.max:
                    failed.append("artifact_count_exceeded")
            for forbidden in artifact_contract.forbidden_artifact_types:
                if counts.get(forbidden, 0) > 0:
                    failed.append("forbidden_artifact_present")
            if artifact_contract.path_policy == "required":
                matching = [artifact for artifact in artifacts if artifact.artifact_type == artifact_contract.artifact_type]
                for artifact in matching:
                    if not artifact.output_paths:
                        failed.append("output_path_missing")
                    for output_path in artifact.output_paths:
                        if not Path(output_path).exists():
                            failed.append("output_path_not_found")
            if artifact_contract.path_policy == "forbidden":
                matching = [artifact for artifact in artifacts if artifact.artifact_type == artifact_contract.artifact_type]
                for artifact in matching:
                    if artifact.output_paths:
                        failed.append("output_path_forbidden")

    if allowed_types:
        for artifact in artifacts:
            if artifact.artifact_type not in allowed_types:
                failed.append("artifact_type_mismatch")
            if artifact.source_step_id != step.step_id:
                failed.append("output_source_step_mismatch")

    failed = sorted(set(failed))
    return VerificationResult(
        status="failed" if failed else "passed",
        failed_checks=failed,
        required_artifacts=sorted(set(required_types)),
        produced_artifacts=produced_types,
        error=result.error,
    )


def _append_record(
    ledger: ExecutionLedger,
    record: StepExecutionRecord,
    artifacts: list[ArtifactFrame] | None = None,
    *,
    status: Literal["running", "succeeded", "failed", "partial"] | None = None,
    completed_at: str | None = None,
    repair: dict[str, Any] | None = None,
    final_contract_status: dict[str, Any] | None = None,
) -> ExecutionLedger:
    next_ledger = ledger.model_copy(deep=True)
    next_ledger.step_records.append(record)
    if artifacts:
        next_ledger.artifacts.extend(artifacts)
    if repair:
        next_ledger.repairs.append(repair)
    if final_contract_status:
        next_ledger.final_contract_status.update(final_contract_status)
    if status:
        next_ledger.status = status
    if completed_at:
        next_ledger.completed_at = completed_at
    return next_ledger


def record_step_result(
    ledger: ExecutionLedger,
    step: PlanStep,
    result: StepCallableResult,
    verification: VerificationResult,
    *,
    attempt: int,
    callable_name: str,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> ExecutionLedger:
    artifacts = _normalize_output_artifacts(step, result)
    output_ids = [artifact.artifact_id for artifact in artifacts]
    verified_artifacts = artifacts if verification.passed else []
    record = StepExecutionRecord(
        step_id=step.step_id,
        status="succeeded" if verification.passed else "failed",
        attempt=attempt,
        started_at=started_at,
        completed_at=completed_at,
        callable_name=callable_name,
        artifact_ids=output_ids,
        input_artifact_ids=_input_artifact_ids(step, ledger),
        output_artifact_ids=output_ids,
        verification_status=verification.status,
        failed_checks=verification.failed_checks,
        error=result.error or verification.error,
    )
    return _append_record(ledger, record, verified_artifacts)


def _record_plan_failure(
    ledger: ExecutionLedger,
    *,
    failed_checks: list[str],
    error: str,
) -> ExecutionLedger:
    next_ledger = ledger.model_copy(deep=True)
    next_ledger.status = "failed"
    next_ledger.completed_at = _now_iso()
    next_ledger.final_contract_status.update({
        "status": "failed",
        "failed_checks": failed_checks,
        "error": error,
    })
    return next_ledger


def _record_failed_step(
    ledger: ExecutionLedger,
    step: PlanStep,
    *,
    callable_name: str,
    attempt: int,
    failed_checks: list[str],
    error: str,
) -> ExecutionLedger:
    record = StepExecutionRecord(
        step_id=step.step_id,
        status="failed",
        attempt=attempt,
        started_at=_now_iso(),
        completed_at=_now_iso(),
        callable_name=callable_name,
        verification_status="failed",
        failed_checks=failed_checks,
        error=error,
    )
    return _append_record(ledger, record, status="failed", completed_at=_now_iso())


def run_plan(
    plan: RequestPlan,
    callable_registry: Mapping[str, StepCallable],
    config: PlannerConfig,
    *,
    ledger: ExecutionLedger | None = None,
) -> ExecutionLedger:
    """Run a compiled plan with injected callables only."""
    current = ledger or create_execution_ledger(plan)
    plan_errors = validate_execution_plan(plan)
    if plan_errors:
        return _record_plan_failure(current, failed_checks=["execution_plan_invalid"], error="; ".join(plan_errors))

    while current.status == "running":
        step = next_runnable_step(plan, current)
        if step is None:
            if len(_successful_steps(current)) == len(plan.steps):
                done = current.model_copy(deep=True)
                done.status = "succeeded"
                done.completed_at = _now_iso()
                return done
            return _record_plan_failure(
                current,
                failed_checks=["no_runnable_step"],
                error="No runnable step remains before all contracts succeeded.",
            )

        callable_name = _callable_name_for_step(step)
        decision = authorize_step(plan, step, current, config)
        if decision.outcome in {"block_retry_same_step", "block_plan_failure"}:
            return _record_failed_step(
                current,
                step,
                callable_name=callable_name,
                attempt=_attempt_count(current, step.step_id) + 1,
                failed_checks=decision.failed_checks or ["authorization_failed"],
                error=decision.reason,
            )

        callable_func = callable_registry.get(callable_name)
        if callable_func is None:
            return _record_failed_step(
                current,
                step,
                callable_name=callable_name,
                attempt=_attempt_count(current, step.step_id) + 1,
                failed_checks=["callable_missing"],
                error=f"No callable registered for {callable_name!r}.",
            )

        max_attempts = step.retry_policy.max_attempts
        if max_attempts <= 0:
            return _record_plan_failure(
                current,
                failed_checks=["attempt_budget_exhausted"],
                error=f"Step {step.step_id!r} has zero allowed attempts.",
            )

        step_succeeded = False
        while _attempt_count(current, step.step_id) < max_attempts:
            attempt = _attempt_count(current, step.step_id) + 1
            started_at = _now_iso()
            try:
                raw_result = callable_func(step, current)
                if not isinstance(raw_result, StepCallableResult):
                    result = StepCallableResult(success=False, error="Callable did not return StepCallableResult.")
                    verification = VerificationResult(status="failed", failed_checks=["invalid_callable_result"], error=result.error)
                else:
                    result = raw_result
                    verification = verify_step_outputs(step, result, current)
            except Exception as exc:  # pragma: no cover - caller-defined exception type
                result = StepCallableResult(success=False, error=str(exc))
                verification = VerificationResult(status="failed", failed_checks=["callable_exception"], error=str(exc))

            current = record_step_result(
                current,
                step,
                result,
                verification,
                attempt=attempt,
                callable_name=callable_name,
                started_at=started_at,
                completed_at=_now_iso(),
            )
            if verification.passed:
                step_succeeded = True
                break
            if attempt < max_attempts:
                current = _append_record(
                    current,
                    StepExecutionRecord(
                        step_id=step.step_id,
                        status="running",
                        attempt=attempt,
                        callable_name=callable_name,
                        verification_status="failed",
                        failed_checks=verification.failed_checks,
                        error="Retrying same callable after verification failure.",
                    ),
                    repair={
                        "step_id": step.step_id,
                        "attempt": attempt,
                        "failed_checks": verification.failed_checks,
                        "action": "retry_same_callable",
                    },
                )

        if not step_succeeded:
            failed = current.model_copy(deep=True)
            failed.status = "failed"
            failed.completed_at = _now_iso()
            failed.final_contract_status.update({"status": "failed", "failed_step_id": step.step_id})
            if step.retry_policy.allow_recompile:
                failed.repairs.append({"step_id": step.step_id, "action": "recompile_not_implemented"})
            return failed

    return current
