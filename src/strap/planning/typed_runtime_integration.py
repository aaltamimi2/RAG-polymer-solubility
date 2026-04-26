"""Thin integration point for opt-in typed runtime execution.

This module keeps the production decision boundary small. Normal orchestration
can call ``maybe_run_typed_runtime`` before legacy routing; off/shadow and
unselected requests return ``None`` so legacy behavior remains unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, Literal

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage
from pydantic import Field

from strap.planning.config import PlannerConfig, get_planner_config
from strap.planning.executor import StepCallable
from strap.planning.frontier_formatting import (
    pareto_frontier_count,
    pareto_frontier_table_lines,
    pareto_landscape_count,
    pareto_metric_labels,
)
from strap.planning.guard import validate_final_synthesis_sources
from strap.planning.models import ArtifactFrame, PlanningModel
from strap.planning.runtime import TypedRuntimeResult, run_typed_runtime
from strap.planning.runtime_production_wrappers import get_production_runtime_callable_registry
from strap.planning.typed_runtime_context import maybe_hydrate_context_for_selected_followup
from strap.planning.typed_runtime_followups import TypedRuntimeFollowupDecision, maybe_answer_typed_runtime_followup


LegacyRunner = Callable[[str, dict[str, Any] | None], Any]


class RuntimeProgressSummary(PlanningModel):
    schema_version: Literal["1.0"] = "1.0"
    status: str
    selected: bool = False
    plan_id: str | None = None
    workflow_id: str | None = None
    current_step_id: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    failed_step_id: str | None = None
    failed_checks: list[str] = Field(default_factory=list)
    produced_artifact_paths: list[str] = Field(default_factory=list)
    diagnostic_bundle_path: str | None = None


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _get_last_human_message(messages: list) -> str:
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None)
        if msg_type == "human" or msg.__class__.__name__ == "HumanMessage":
            return _extract_text(getattr(msg, "content", ""))
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _extract_text(msg.get("content", ""))
    return ""


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def maybe_run_typed_runtime(
    query: str,
    context: dict[str, Any] | None = None,
    legacy_runner: LegacyRunner | None = None,
    *,
    config: PlannerConfig | None = None,
    output_root: str | None = None,
    callable_registry: Mapping[str, StepCallable] | None = None,
    created_at: str | None = None,
) -> TypedRuntimeResult | None:
    """Run typed runtime only for selected enforcement targets.

    ``legacy_runner`` is accepted for the orchestration call-site shape but is
    intentionally not invoked here; returning ``None`` means the caller should
    continue on the existing legacy path.
    """
    del legacy_runner
    effective_config = config or get_planner_config()
    if effective_config.mode in {"off", "shadow"}:
        return None
    registry = (
        dict(callable_registry)
        if callable_registry is not None
        else get_production_runtime_callable_registry()
    )
    result = run_typed_runtime(
        query,
        context=context,
        config=effective_config,
        callable_registry=registry,
        output_root=output_root,
        created_at=created_at,
        persist=True,
    )
    if result.status == "legacy_fallback":
        return None
    return result


def _latest_failed_record(result: TypedRuntimeResult):
    if result.ledger is None:
        return None
    for record in reversed(result.ledger.step_records):
        if record.status == "failed":
            return record
    return None


def summarize_typed_runtime_progress(result: TypedRuntimeResult) -> RuntimeProgressSummary:
    completed_steps: list[str] = []
    failed_step_id: str | None = None
    failed_checks: list[str] = []
    if result.ledger is not None:
        for record in result.ledger.step_records:
            if record.status == "succeeded" and record.step_id not in completed_steps:
                completed_steps.append(record.step_id)
            if record.status == "failed":
                failed_step_id = record.step_id
                failed_checks = list(record.failed_checks)

    produced_paths: list[str] = []
    if result.ledger is not None:
        for artifact in result.ledger.artifacts:
            produced_paths.extend(artifact.output_paths)
    produced_paths = _dedupe_preserving_order(produced_paths)

    current_step_id = None
    if result.plan is not None:
        completed = set(completed_steps)
        failed = {failed_step_id} if failed_step_id else set()
        for step in result.plan.steps:
            if step.step_id not in completed and step.step_id not in failed:
                current_step_id = step.step_id
                break

    return RuntimeProgressSummary(
        status=result.status,
        selected=result.selected,
        plan_id=result.plan.plan_id if result.plan else None,
        workflow_id=result.plan.workflow_id if result.plan else None,
        current_step_id=current_step_id,
        completed_steps=completed_steps,
        failed_step_id=failed_step_id,
        failed_checks=failed_checks or list(result.diagnostics),
        produced_artifact_paths=produced_paths,
        diagnostic_bundle_path=result.manifest.run_dir if result.manifest else None,
    )


def _failure_phase(result: TypedRuntimeResult) -> str:
    if result.compile_result.status != "compiled":
        return "compile"
    if any(item.startswith("missing_wrapper:") for item in result.diagnostics):
        return "missing_wrapper"
    record = _latest_failed_record(result)
    if record is None:
        return "runtime"
    if "callable_exception" in record.failed_checks or "callable_missing" in record.failed_checks:
        return "callable"
    return "verification"


def format_typed_runtime_failure(result: TypedRuntimeResult) -> str:
    """Return a concise operator-facing failure summary."""
    record = _latest_failed_record(result)
    failed_step_id = record.step_id if record else None
    failed_checks = record.failed_checks if record else list(result.diagnostics)
    diagnostic_path = result.manifest.run_dir if result.manifest else None
    lines = [
        "Typed runtime failed.",
        f"Reason: {result.reason}",
        f"Failed phase: {_failure_phase(result)}",
    ]
    if failed_step_id:
        lines.append(f"Failed step: {failed_step_id}")
    if failed_checks:
        lines.append("Failed checks: " + ", ".join(str(item) for item in failed_checks))
    if diagnostic_path:
        lines.append(f"Diagnostic bundle: {diagnostic_path}")
    return "\n".join(lines)


def _artifact_payload(artifact: ArtifactFrame) -> dict[str, Any]:
    payload = artifact.validation_summary.get("payload")
    return payload if isinstance(payload, dict) else {}


def _latest_artifact(result: TypedRuntimeResult, artifact_types: set[str]) -> ArtifactFrame | None:
    if result.ledger is None:
        return None
    for artifact in reversed(result.ledger.artifacts):
        if artifact.artifact_type in artifact_types:
            return artifact
    return None


def _artifact_lines(result: TypedRuntimeResult) -> list[str]:
    if result.ledger is None:
        return []
    grouped: dict[str, list[str]] = {}
    for artifact in result.ledger.artifacts:
        if not artifact.output_paths:
            continue
        for path in artifact.output_paths:
            grouped.setdefault(path, [])
            if artifact.artifact_type not in grouped[path]:
                grouped[path].append(artifact.artifact_type)
    return [
        f"- {', '.join(artifact_types)}: {path}"
        for path, artifact_types in grouped.items()
    ]


def _optimization_summary_lines(result: TypedRuntimeResult) -> list[str]:
    slices_artifact = _latest_artifact(result, {"optimization_pareto_slices"})
    if slices_artifact is not None:
        payload = _artifact_payload(slices_artifact)
        return [
            f"Workflow: {result.plan.workflow_id if result.plan else 'typed_runtime'}",
            f"Solved slices: {payload.get('n_slices_solved', 0)}/{payload.get('n_slices_requested', 0)}",
            f"Metrics: {payload.get('x_metric', 'total_cost')} vs {payload.get('y_metric', 'circularity')}",
        ]

    pareto_artifact = _latest_artifact(result, {"optimization_pareto_front", "optimization_pareto_landscape"})
    if pareto_artifact is not None:
        payload = _artifact_payload(pareto_artifact)
        x_metric, y_metric = pareto_metric_labels(payload)
        lines = [
            f"Workflow: {result.plan.workflow_id if result.plan else 'typed_runtime'}",
            f"Frontier points: {pareto_frontier_count(payload)}",
            f"Feasible landscape points: {pareto_landscape_count(payload)}",
            f"Metrics: {x_metric} vs {y_metric}",
        ]
        lines.extend(pareto_frontier_table_lines(payload, max_points=6))
        return lines
    return []


def format_typed_runtime_success(
    result: TypedRuntimeResult,
    *,
    config: PlannerConfig | None = None,
) -> str:
    """Return a final response generated only from verified ledger artifacts."""
    if result.plan is None or result.ledger is None:
        return format_typed_runtime_failure(result)
    artifact_types = sorted({artifact.artifact_type for artifact in result.ledger.artifacts})
    validation = validate_final_synthesis_sources(
        result.plan,
        result.ledger,
        {"source_artifact_types": artifact_types},
        config=config,
    )
    if validation.status == "failed":
        failed = result.model_copy(update={"reason": validation.reason, "diagnostics": validation.failed_checks})
        return format_typed_runtime_failure(failed)

    lines = [
        "Typed runtime completed from verified structured artifacts.",
        f"Plan: {result.plan.plan_id}",
    ]
    lines.extend(_optimization_summary_lines(result))
    artifacts = _artifact_lines(result)
    if artifacts:
        lines.append("Artifacts:")
        lines.extend(artifacts)
    if result.manifest is not None:
        lines.append(f"Diagnostic bundle: {result.manifest.run_dir}")
    progress = summarize_typed_runtime_progress(result)
    if progress.completed_steps:
        lines.append("Completed steps: " + ", ".join(progress.completed_steps))
    return "\n".join(lines)


def _typed_runtime_message_status(
    result: TypedRuntimeResult,
    *,
    config: PlannerConfig,
) -> str:
    if result.status != "executed" or result.plan is None or result.ledger is None:
        return result.status
    artifact_types = sorted({artifact.artifact_type for artifact in result.ledger.artifacts})
    validation = validate_final_synthesis_sources(
        result.plan,
        result.ledger,
        {"source_artifact_types": artifact_types},
        config=config,
    )
    return "typed_failure" if validation.status == "failed" else result.status


class TypedRuntimeMiddleware(AgentMiddleware):
    """Short-circuit selected typed workflows before legacy routing."""

    def __init__(
        self,
        *,
        config: PlannerConfig | None = None,
        output_root: str | None = None,
        callable_registry: Mapping[str, StepCallable] | None = None,
    ) -> None:
        self._config = config
        self._output_root = output_root
        self._callable_registry = callable_registry

    def _to_model_response(self, result: TypedRuntimeResult) -> ModelResponse:
        config = self._config or get_planner_config()
        message_status = _typed_runtime_message_status(result, config=config)
        content = (
            format_typed_runtime_success(result, config=config)
            if message_status == "executed"
            else format_typed_runtime_failure(result)
        )
        return ModelResponse(
            result=[
                AIMessage(
                    content=content,
                    additional_kwargs={
                        "strap_origin": "typed_runtime",
                        "strap_typed_runtime_status": message_status,
                        "strap_typed_runtime_selected": result.selected,
                        "strap_plan_id": result.plan.plan_id if result.plan else None,
                        "strap_workflow_id": result.plan.workflow_id if result.plan else None,
                        "strap_runtime_progress": summarize_typed_runtime_progress(result).model_dump(mode="json"),
                        "strap_manifest": result.manifest.model_dump(mode="json") if result.manifest else None,
                        "strap_compile_result": result.compile_result.model_dump(mode="json"),
                        "strap_run_plan": result.plan.model_dump(mode="json") if result.plan else None,
                        "strap_run_ledger": result.ledger.model_dump(mode="json") if result.ledger else None,
                    },
                )
            ]
        )

    def _followup_to_model_response(self, decision: TypedRuntimeFollowupDecision) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content=decision.response_text or "",
                    additional_kwargs={
                        "strap_origin": "typed_runtime_followup",
                        "strap_typed_runtime_status": "answered_from_prior_artifacts",
                        "strap_runtime_progress": decision.progress,
                        "strap_followup_reason": decision.reason,
                        "strap_followup_artifacts": [
                            artifact.model_dump(mode="json") for artifact in decision.matched_artifacts
                        ],
                    },
                )
            ]
        )

    def wrap_model_call(self, request, handler):
        query = _get_last_human_message(request.messages)
        config = self._config or get_planner_config()
        if config.mode in {"off", "shadow"}:
            return handler(request)
        followup = maybe_answer_typed_runtime_followup(query, request.messages)
        if followup.should_answer:
            return self._followup_to_model_response(followup)
        hydration = maybe_hydrate_context_for_selected_followup(query, request.messages)
        result = maybe_run_typed_runtime(
            query,
            context=hydration.context if hydration is not None else None,
            config=self._config,
            output_root=self._output_root,
            callable_registry=self._callable_registry,
        )
        if result is None:
            return handler(request)
        return self._to_model_response(result)

    async def awrap_model_call(self, request, handler):
        query = _get_last_human_message(request.messages)
        config = self._config or get_planner_config()
        if config.mode in {"off", "shadow"}:
            return await handler(request)
        followup = maybe_answer_typed_runtime_followup(query, request.messages)
        if followup.should_answer:
            return self._followup_to_model_response(followup)
        hydration = maybe_hydrate_context_for_selected_followup(query, request.messages)
        result = await asyncio.to_thread(
            maybe_run_typed_runtime,
            query,
            context=hydration.context if hydration is not None else None,
            config=self._config,
            output_root=self._output_root,
            callable_registry=self._callable_registry,
        )
        if result is None:
            return await handler(request)
        return self._to_model_response(result)
