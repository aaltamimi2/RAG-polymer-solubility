"""Cross-turn typed-runtime context collection and conservative hydration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from strap.planning.models import ArtifactFrame, PlanningModel


_OPTIMIZATION_RE = re.compile(r"\b(?:pareto|optimization|optimisation|frontier|landscape)\b", re.I)
_RERUN_ACTION_RE = re.compile(
    r"\b(?:"
    r"rerun|re-run|repeat|recompute|regenerate|"
    r"run\s+(?:it|that|the\s+same|again)|"
    r"(?:generate|create|make|plot|build)\s+(?:the\s+)?same|"
    r"(?:generate|create|make|plot|build)\b.*\bagain"
    r")\b",
    re.I,
)
_NEGATED_RERUN_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|not|no|without)\s+"
    r"(?:rerun|re-run|run|regenerate|recompute|repeat)\b",
    re.I,
)


class TypedRuntimeContextSnapshot(PlanningModel):
    plan_id: str | None = None
    workflow_id: str | None = None
    user_query: str | None = None
    status: str | None = None
    plan: dict[str, Any] | None = None
    ledger: dict[str, Any] | None = None
    compile_result: dict[str, Any] | None = None
    artifacts: list[ArtifactFrame] = Field(default_factory=list)
    manifest: dict[str, Any] = Field(default_factory=dict)


class HydratedRuntimeContext(PlanningModel):
    context: dict[str, Any]
    source_plan_id: str | None = None
    source_workflow_id: str | None = None
    source_run_dir: str | None = None
    hydration_notes: list[str] = Field(default_factory=list)


def _message_kwargs(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        kwargs = message.get("additional_kwargs") or {}
        return kwargs if isinstance(kwargs, dict) else {}
    kwargs = getattr(message, "additional_kwargs", None)
    return kwargs if isinstance(kwargs, dict) else {}


def _load_json_file(path: str | None) -> Any:
    if not path:
        return None
    try:
        with Path(path).expanduser().open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _artifact_from_mapping(value: Any) -> ArtifactFrame | None:
    if not isinstance(value, dict):
        return None
    try:
        return ArtifactFrame.model_validate(value)
    except ValueError:
        return None


def _manifest_files(manifest: dict[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return {}
    return {str(key): str(value) for key, value in files.items() if value}


def _load_payload_from_manifest(
    manifest: dict[str, Any],
    key: str,
) -> Any:
    return _load_json_file(_manifest_files(manifest).get(key))


def _artifacts_from_sources(
    ledger: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> list[ArtifactFrame]:
    artifacts: list[ArtifactFrame] = []
    seen: set[str] = set()
    if isinstance(ledger, dict):
        for item in ledger.get("artifacts") or []:
            artifact = _artifact_from_mapping(item)
            if artifact is not None and artifact.artifact_id not in seen:
                artifacts.append(artifact)
                seen.add(artifact.artifact_id)
    loaded = _load_payload_from_manifest(manifest, "artifacts")
    if isinstance(loaded, list):
        for item in loaded:
            artifact = _artifact_from_mapping(item)
            if artifact is not None and artifact.artifact_id not in seen:
                artifacts.append(artifact)
                seen.add(artifact.artifact_id)
    return artifacts


def _snapshot_from_kwargs(kwargs: dict[str, Any]) -> TypedRuntimeContextSnapshot | None:
    if kwargs.get("strap_origin") != "typed_runtime":
        return None

    manifest = kwargs.get("strap_manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    plan = kwargs.get("strap_run_plan")
    plan = plan if isinstance(plan, dict) else _load_payload_from_manifest(manifest, "plan")
    compile_result = kwargs.get("strap_compile_result")
    compile_result = (
        compile_result if isinstance(compile_result, dict) else _load_payload_from_manifest(manifest, "compile_result")
    )
    if not isinstance(plan, dict) and isinstance(compile_result, dict) and isinstance(compile_result.get("plan"), dict):
        plan = compile_result["plan"]
    ledger = kwargs.get("strap_run_ledger")
    ledger = ledger if isinstance(ledger, dict) else _load_payload_from_manifest(manifest, "ledger")
    request_payload = _load_payload_from_manifest(manifest, "request")

    plan_id = kwargs.get("strap_plan_id") or (plan or {}).get("plan_id")
    workflow_id = kwargs.get("strap_workflow_id") or (plan or {}).get("workflow_id")
    user_query = (plan or {}).get("user_query") or (request_payload or {}).get("query")
    artifacts = _artifacts_from_sources(ledger if isinstance(ledger, dict) else None, manifest)
    if not any([plan, ledger, compile_result, artifacts, manifest]):
        return None
    return TypedRuntimeContextSnapshot(
        plan_id=str(plan_id) if plan_id else None,
        workflow_id=str(workflow_id) if workflow_id else None,
        user_query=str(user_query) if user_query else None,
        status=str(kwargs.get("strap_typed_runtime_status")) if kwargs.get("strap_typed_runtime_status") else None,
        plan=plan if isinstance(plan, dict) else None,
        ledger=ledger if isinstance(ledger, dict) else None,
        compile_result=compile_result if isinstance(compile_result, dict) else None,
        artifacts=artifacts,
        manifest=manifest,
    )


def collect_recent_typed_runtime_snapshots(
    messages: list[Any],
    *,
    limit: int = 4,
) -> list[TypedRuntimeContextSnapshot]:
    """Collect recent typed-runtime snapshots from assistant message metadata."""
    snapshots: list[TypedRuntimeContextSnapshot] = []
    for message in reversed(messages):
        snapshot = _snapshot_from_kwargs(_message_kwargs(message))
        if snapshot is None:
            continue
        snapshots.append(snapshot)
        if len(snapshots) >= limit:
            break
    return snapshots


def _is_optimization_snapshot(snapshot: TypedRuntimeContextSnapshot) -> bool:
    if snapshot.workflow_id and "optimization" in snapshot.workflow_id:
        return True
    return any(artifact.artifact_type.startswith("optimization_") for artifact in snapshot.artifacts)


def _ordered_unique(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _step_args(plan: dict[str, Any], step_id: str) -> dict[str, Any]:
    for step in plan.get("steps") or []:
        if isinstance(step, dict) and step.get("step_id") == step_id:
            args = step.get("tool_args_template")
            return args if isinstance(args, dict) else {}
    return {}


def _prior_artifact_types(snapshot: TypedRuntimeContextSnapshot) -> list[str]:
    types = [artifact.artifact_type for artifact in snapshot.artifacts]
    constraints = (snapshot.plan or {}).get("global_constraints")
    if isinstance(constraints, dict):
        requested = constraints.get("requested_artifact_types")
        if isinstance(requested, list):
            types.extend(str(item) for item in requested)
    return [str(item) for item in _ordered_unique(types)]


def _hydrated_optimization_context(snapshot: TypedRuntimeContextSnapshot) -> HydratedRuntimeContext:
    plan = snapshot.plan or {}
    constraints = plan.get("global_constraints")
    constraints = constraints if isinstance(constraints, dict) else {}
    optimize_args = (
        _step_args(plan, "optimize_pareto")
        or _step_args(plan, "optimize_slices")
        or _step_args(plan, "optimize_point")
    )

    context: dict[str, Any] = {
        "hydrated_from_typed_runtime": True,
        "prior_plan_id": snapshot.plan_id,
        "prior_workflow_id": snapshot.workflow_id,
        "prior_artifact_types": _prior_artifact_types(snapshot),
    }
    for key in (
        "feed_capacity_tpy",
        "feed_composition",
        "composition_slices",
        "scenario",
        "top_k_per_polymer",
        "n_points",
        "min_washes",
        "max_washes",
        "metrics",
    ):
        value = constraints.get(key)
        if value not in (None, "", [], {}):
            context[key] = value

    if "feed_capacity_tpy" not in context and optimize_args.get("feed_capacity_tpy") is not None:
        context["feed_capacity_tpy"] = optimize_args["feed_capacity_tpy"]
    if "feed_composition" not in context and optimize_args.get("feed_composition_json"):
        context["feed_composition"] = optimize_args["feed_composition_json"]
    if "scenario" not in context and optimize_args.get("scenario"):
        context["scenario"] = optimize_args["scenario"]
    if optimize_args.get("objective"):
        context["objective"] = optimize_args["objective"]

    metrics = list(context.get("metrics") or [])
    for metric_key in ("x_metric", "y_metric"):
        metric = optimize_args.get(metric_key)
        if metric and metric not in metrics:
            metrics.append(metric)
    if metrics:
        context["metrics"] = metrics

    requested = list(context.get("prior_artifact_types") or [])
    if snapshot.workflow_id == "routed_optimization":
        requested.extend([
            "separation_topk_sequences",
            "optimization_stage_candidates",
            "optimization_pareto_front",
            "optimization_pareto_landscape",
            "optimization_pareto_plot",
        ])
        context["workflow_markers"] = ["separation", "handoff", "optimization", "visualization"]
    elif snapshot.workflow_id == "routed_optimization_slices":
        requested.extend([
            "separation_topk_sequences",
            "optimization_stage_candidates",
            "optimization_pareto_slices",
            "optimization_pareto_slices_plot",
        ])
        context["workflow_markers"] = ["separation", "handoff", "optimization", "visualization", "multi_slice"]
    elif any(item in requested for item in ("optimization_pareto_front", "optimization_pareto_landscape")):
        requested.extend(["optimization_pareto_front", "optimization_pareto_landscape", "optimization_pareto_plot"])
        context["workflow_markers"] = ["optimization", "visualization"]
    context["requested_artifact_types"] = [str(item) for item in _ordered_unique(requested)]

    if isinstance(context.get("feed_composition"), dict):
        context["polymers"] = list(context["feed_composition"])

    return HydratedRuntimeContext(
        context=context,
        source_plan_id=snapshot.plan_id,
        source_workflow_id=snapshot.workflow_id,
        source_run_dir=str(snapshot.manifest.get("run_dir")) if snapshot.manifest.get("run_dir") else None,
        hydration_notes=[
            "hydrated_from_prior_typed_runtime",
            f"source_workflow={snapshot.workflow_id}" if snapshot.workflow_id else "source_workflow=unknown",
        ],
    )


def maybe_hydrate_context_for_selected_followup(
    query: str,
    messages: list[Any],
) -> HydratedRuntimeContext | None:
    """Return typed context only for explicit selected rerun/action requests."""
    if _NEGATED_RERUN_RE.search(query):
        return None
    if not _RERUN_ACTION_RE.search(query) or not _OPTIMIZATION_RE.search(query):
        return None
    for snapshot in collect_recent_typed_runtime_snapshots(messages):
        if _is_optimization_snapshot(snapshot):
            return _hydrated_optimization_context(snapshot)
    return None
