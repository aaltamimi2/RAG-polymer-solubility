"""Thin integration point for opt-in typed runtime execution.

This module keeps the production decision boundary small. Normal orchestration
can call ``maybe_run_typed_runtime`` before legacy routing; off/shadow and
unselected requests return ``None`` so legacy behavior remains unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal

logger = logging.getLogger(__name__)

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


# Specialists whose deliverables the typed runtime can produce end to end.
# Interception is only safe when the route plan stays inside this set;
# anything else (research/RAG/contaminant stages) must reach the orchestrator.
_TYPED_COVERED_SUBAGENTS = frozenset({
    "separation-engineer",
    "safety-analyst",
    "statistics-ml",
    "biosteam-analyst",
    "optimization-engineer",
    "visualization-specialist",
})


class TypedRuntimeMiddleware(AgentMiddleware):
    """Short-circuit selected typed workflows before legacy routing."""

    def __init__(
        self,
        *,
        config: PlannerConfig | None = None,
        output_root: str | None = None,
        callable_registry: Mapping[str, StepCallable] | None = None,
        route_planner=None,
    ) -> None:
        self._config = config
        self._output_root = output_root
        self._callable_registry = callable_registry
        self._route_planner = route_planner

    def _plan_permits_typed_runtime(self, query: str, session_digest: str | None = None) -> bool:
        """Gate token-triggered interception behind the route plan's intent."""
        if self._route_planner is None:
            return True
        plan = self._route_planner.plan(query, session_digest=session_digest)
        if plan.source != "planner":
            # Deliberate keyword-mode deployment (no backend) keeps legacy
            # behavior; a degraded planner must not intercept on keywords.
            permitted = not getattr(self._route_planner, "has_backend", False)
            if not permitted:
                logger.warning(
                    "typed_runtime: skipped — planner degraded; keyword intent is advisory only"
                )
            return permitted
        if plan.is_direct:
            return True
        if plan.mode == "orchestrator":
            logger.info("typed_runtime: skipped — route plan says orchestrator handles query")
            return False
        names = set(plan.subagent_names())
        if names and names <= _TYPED_COVERED_SUBAGENTS:
            return True
        logger.info(
            "typed_runtime: skipped — route plan requires uncovered specialists %s",
            sorted(names - _TYPED_COVERED_SUBAGENTS),
        )
        return False

    def _plan_deliverable_context(self, query: str, session_digest: str | None = None) -> dict[str, Any] | None:
        """Planner-declared intent as authoritative compile context.

        Deliverables are filtered against the artifact catalog; workflow
        markers are derived from the plan's step graph (structured data, not
        query tokens) so routed workflows are reachable without keyword
        phrasing. The compiler's keyword detection becomes fallback-only.
        """
        if self._route_planner is None:
            return None
        plan = self._route_planner.plan(query, session_digest=session_digest)
        if plan.source != "planner":
            return None
        from strap.planning.capability_registry import ARTIFACT_TYPES

        context: dict[str, Any] = {}
        valid = [name for name in plan.deliverables if name in ARTIFACT_TYPES]
        if valid:
            context["plan_requested_artifact_types"] = valid

        step_names = set(plan.subagent_names())
        markers: list[str] = []
        if "separation-engineer" in step_names:
            markers.append("separation")
        if "optimization-engineer" in step_names:
            markers.append("optimization")
        if any(step.depends_on for step in plan.steps):
            markers.append("handoff")
        if any(name.startswith("optimization_pareto_slices") for name in valid):
            markers.append("multi_slice")
        if markers:
            context["plan_workflow_markers"] = markers
        return context or None

    def _defer_typed_failure(self, query: str, result: TypedRuntimeResult, session_digest: str | None = None) -> bool:
        """Never dead-end a planner-routed query on a typed failure.

        When the route plan assigns capable specialists, a compile or
        execution failure in the typed lane falls through to the orchestrator
        so the specialist produces the deliverable instead of the user
        receiving a typed-failure/clarification message.
        """
        if result.status != "typed_failure" or self._route_planner is None:
            return False
        plan = self._route_planner.plan(query, session_digest=session_digest)
        if plan.source != "planner" or not plan.is_specialists:
            return False
        logger.warning(
            "typed_runtime: typed_failure deferred to planned specialists %s (%s)",
            plan.subagent_names(),
            (result.reason or "")[:120],
        )
        return True

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
        from strap.route_planner import build_session_digest

        session_digest = build_session_digest(request.messages)
        if not self._plan_permits_typed_runtime(query, session_digest):
            return handler(request)
        hydration = maybe_hydrate_context_for_selected_followup(query, request.messages)
        context = dict(hydration.context) if hydration is not None and hydration.context else {}
        context.update(self._plan_deliverable_context(query, session_digest) or {})
        result = maybe_run_typed_runtime(
            query,
            context=context or None,
            config=self._config,
            output_root=self._output_root,
            callable_registry=self._callable_registry,
        )
        if result is None or self._defer_typed_failure(query, result, session_digest):
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
        from strap.route_planner import build_session_digest

        session_digest = build_session_digest(request.messages)
        if not self._plan_permits_typed_runtime(query, session_digest):
            return await handler(request)
        hydration = maybe_hydrate_context_for_selected_followup(query, request.messages)
        context = dict(hydration.context) if hydration is not None and hydration.context else {}
        context.update(self._plan_deliverable_context(query, session_digest) or {})
        result = await asyncio.to_thread(
            maybe_run_typed_runtime,
            query,
            context=context or None,
            config=self._config,
            output_root=self._output_root,
            callable_registry=self._callable_registry,
        )
        if result is None or self._defer_typed_failure(query, result, session_digest):
            return await handler(request)
        return self._to_model_response(result)
