"""Follow-up answers over prior typed-runtime artifacts.

This module handles cross-turn questions that refer to artifacts already
produced by the typed runtime. It intentionally reads only ledger/manifest
metadata from prior assistant messages; it does not compile or execute new
plans.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from strap.planning.frontier_formatting import (
    pareto_frontier_count,
    pareto_frontier_table_lines,
    pareto_landscape_count,
    pareto_metric_labels,
    pareto_stage3_mentions,
)
from strap.planning.models import ArtifactFrame, PlanningModel
from strap.planning.typed_runtime_context import (
    TypedRuntimeContextSnapshot,
    collect_recent_typed_runtime_snapshots,
)


_PATH_STATUS_RE = re.compile(
    r"\b(?:where|saved|save|path|file|files|diagnostic|bundle|manifest|ledger|"
    r"typed\s+runtime|legacy|open|locat(?:e|ion)|which\s+folder)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(?:summari[sz]e|summary|recap|what\s+did\s+we\s+learn|what\s+plots?|"
    r"generated\s+so\s+far|outputs?\s+so\s+far|artifacts?\s+so\s+far)\b",
    re.IGNORECASE,
)
_RUNTIME_STATUS_RE = re.compile(
    r"\b(?:steps?|completed|ran|run|tools?|callables?|workflow|plan|routed|direct|"
    r"typed\s+runtime|legacy|fast[-\s]?path|solubility|safety)\b",
    re.IGNORECASE,
)
_REFERENTIAL_RE = re.compile(
    r"\b(?:that|this|those|these|previous|prior|last|recent|generated|above|"
    r"heatmap|plot|chart|artifact|file|bundle|result|output|summary)\b",
    re.IGNORECASE,
)
_OPTIMIZER_EXPLANATION_RE = re.compile(
    r"\b(?:why|explain|interpret|frontier\s+points?|pareto\s+points?|trade[-\s]?offs?|"
    r"choose|chosen|selected|selection|landfill|stage[-\s]?3)\b",
    re.IGNORECASE,
)
_OPTIMIZATION_ARTIFACT_TYPES = {
    "optimization_pareto_front",
    "optimization_pareto_landscape",
    "optimization_pareto_slices",
    "optimization_pareto_plot",
    "optimization_pareto_slices_plot",
}
_NEW_REQUEST_RE = re.compile(
    r"^\s*(?:generate|create|run|screen|plot|estimate|optimi[sz]e|compare|"
    r"make|build|calculate|compute|evaluate|simulate|use\s+\w+)\b",
    re.IGNORECASE,
)

_ARTIFACT_ALIASES: tuple[tuple[set[str], re.Pattern[str]], ...] = (
    (
        {"hsp_red_heatmap", "hsp_single_pair_summary"},
        re.compile(r"\b(?:hsp|hansen|red|heatmap|compatib(?:le|ility))\b", re.IGNORECASE),
    ),
    (
        {"solvent_safety_card", "solvent_safety_comparison"},
        re.compile(r"\b(?:safety|card|comparison|flash|boiling|toxicity|ld50)\b", re.IGNORECASE),
    ),
    (
        {"biosteam_tea_lca_result", "biosteam_tea_lca_plot"},
        re.compile(r"\b(?:biosteam|tea|lca|msp|tci|aoc|gwp)\b", re.IGNORECASE),
    ),
    (
        {"separation_dp_state_map"},
        re.compile(r"\b(?:dp|dynamic[-\s]*programming|state\s+map)\b", re.IGNORECASE),
    ),
    (
        {"separation_tree_plot"},
        re.compile(r"\b(?:tree|separation\s+tree)\b", re.IGNORECASE),
    ),
    (
        {"separation_selectivity_heatmap"},
        re.compile(r"\b(?:selectivity|selectivity\s+heatmap)\b", re.IGNORECASE),
    ),
    (
        {
            "optimization_pareto_front",
            "optimization_pareto_landscape",
            "optimization_pareto_slices",
            "optimization_pareto_plot",
            "optimization_pareto_slices_plot",
        },
        re.compile(r"\b(?:pareto|frontier|optimization|optimisation|slices?)\b", re.IGNORECASE),
    ),
    (
        {"sidecar_file"},
        re.compile(r"\b(?:sidecar|json|data\s+file)\b", re.IGNORECASE),
    ),
)


class TypedRuntimeArtifactReference(PlanningModel):
    """A compact, cross-turn reference to a prior typed-runtime artifact."""

    artifact_id: str | None = None
    artifact_type: str
    source_step_id: str | None = None
    output_paths: list[str] = Field(default_factory=list)
    diagnostic_copies: dict[str, str] = Field(default_factory=dict)
    diagnostic_bundle_path: str | None = None
    plan_id: str | None = None
    workflow_id: str | None = None
    status: str | None = None
    payload_summary: dict[str, Any] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)


class TypedRuntimeFollowupDecision(PlanningModel):
    """Decision returned by the follow-up resolver."""

    should_answer: bool = False
    reason: str = "not_followup"
    response_text: str | None = None
    matched_artifacts: list[TypedRuntimeArtifactReference] = Field(default_factory=list)
    progress: dict[str, Any] = Field(default_factory=dict)


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


def _infer_artifact_type_from_path(path: str) -> str:
    text = Path(path).name.lower()
    if any(token in text for token in ("hsp", "red", "heatmap")):
        return "hsp_red_heatmap"
    if "biosteam" in text or "tea_lca" in text:
        return "biosteam_tea_lca_plot"
    if "safety" in text:
        return "solvent_safety_comparison"
    if "state" in text and "map" in text:
        return "separation_dp_state_map"
    if "selectivity" in text:
        return "separation_selectivity_heatmap"
    if "tree" in text:
        return "separation_tree_plot"
    if "pareto" in text or "optimization" in text:
        return "optimization_pareto_plot"
    if text.endswith(".json"):
        return "sidecar_file"
    return "typed_runtime_artifact"


def _payload_summary(artifact: ArtifactFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if artifact.entities:
        summary["entities"] = artifact.entities
    if artifact.inputs_used:
        summary["inputs_used"] = artifact.inputs_used

    validation = artifact.validation_summary if isinstance(artifact.validation_summary, dict) else {}
    payload = validation.get("payload")
    if isinstance(payload, dict):
        for key in (
            "polymer_resolution",
            "solvent_resolution",
            "polymer_category",
            "solvent_category",
            "solvent_polarity",
            "solvent_names",
            "solvent_name",
            "operating_temp_c",
            "scenario",
            "target_plastic",
            "MSP",
            "msp",
            "TCI",
            "tci",
            "AOC",
            "aoc",
            "GWP",
            "gwp",
            "x_metric",
            "y_metric",
            "n_slices_requested",
            "n_slices_solved",
            "warnings",
            "warning",
            "error",
            "unsupported_reason",
        ):
            if key in payload and payload[key] not in (None, "", [], {}):
                summary[key] = payload[key]
        for key in ("results", "points", "frontier_points", "landscape_points", "all_feasible_points"):
            value = payload.get(key)
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
    elif validation:
        for key in ("warning", "error", "unsupported_reason"):
            if validation.get(key):
                summary[key] = validation[key]
    return summary


def _manifest_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    manifest = kwargs.get("strap_manifest")
    return manifest if isinstance(manifest, dict) else {}


def _ledger_artifacts_from_kwargs(kwargs: dict[str, Any]) -> list[ArtifactFrame]:
    artifacts: list[ArtifactFrame] = []
    ledger = kwargs.get("strap_run_ledger")
    if isinstance(ledger, dict):
        for item in ledger.get("artifacts") or []:
            artifact = _artifact_from_mapping(item)
            if artifact is not None:
                artifacts.append(artifact)
    manifest = _manifest_from_kwargs(kwargs)
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    loaded = _load_json_file(files.get("artifacts"))
    if isinstance(loaded, list):
        for item in loaded:
            artifact = _artifact_from_mapping(item)
            if artifact is not None and artifact.artifact_id not in {existing.artifact_id for existing in artifacts}:
                artifacts.append(artifact)
    return artifacts


def _progress_paths_from_kwargs(kwargs: dict[str, Any]) -> list[str]:
    progress = kwargs.get("strap_runtime_progress")
    if not isinstance(progress, dict):
        return []
    values = progress.get("produced_artifact_paths")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _failed_checks_from_kwargs(kwargs: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    progress = kwargs.get("strap_runtime_progress")
    if isinstance(progress, dict):
        failed = progress.get("failed_checks")
        if isinstance(failed, list):
            checks.extend(str(item) for item in failed if str(item))
    ledger = kwargs.get("strap_run_ledger")
    if isinstance(ledger, dict):
        for record in ledger.get("step_records") or []:
            if not isinstance(record, dict):
                continue
            failed = record.get("failed_checks")
            if isinstance(failed, list):
                checks.extend(str(item) for item in failed if str(item))
    return list(dict.fromkeys(checks))


def _references_from_typed_kwargs(kwargs: dict[str, Any]) -> list[TypedRuntimeArtifactReference]:
    status = kwargs.get("strap_typed_runtime_status")
    plan_id = kwargs.get("strap_plan_id")
    workflow_id = kwargs.get("strap_workflow_id")
    manifest = _manifest_from_kwargs(kwargs)
    diagnostic_bundle_path = str(manifest.get("run_dir")) if manifest.get("run_dir") else None
    copies = manifest.get("produced_file_copies")
    diagnostic_copies = {str(k): str(v) for k, v in copies.items()} if isinstance(copies, dict) else {}

    refs: list[TypedRuntimeArtifactReference] = []
    seen_paths: set[str] = set()
    for artifact in _ledger_artifacts_from_kwargs(kwargs):
        refs.append(
            TypedRuntimeArtifactReference(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                source_step_id=artifact.source_step_id,
                output_paths=list(artifact.output_paths),
                diagnostic_copies={
                    path: diagnostic_copies[path] for path in artifact.output_paths if path in diagnostic_copies
                },
                diagnostic_bundle_path=diagnostic_bundle_path,
                plan_id=str(plan_id) if plan_id else None,
                workflow_id=str(workflow_id) if workflow_id else None,
                status=str(status) if status else None,
                payload_summary=_payload_summary(artifact),
                failed_checks=_failed_checks_from_kwargs(kwargs),
            )
        )
        seen_paths.update(artifact.output_paths)

    for index, path in enumerate(_progress_paths_from_kwargs(kwargs)):
        if path in seen_paths:
            continue
        refs.append(
            TypedRuntimeArtifactReference(
                artifact_id=f"progress_path:{index}",
                artifact_type=_infer_artifact_type_from_path(path),
                output_paths=[path],
                diagnostic_copies={path: diagnostic_copies[path]} if path in diagnostic_copies else {},
                diagnostic_bundle_path=diagnostic_bundle_path,
                plan_id=str(plan_id) if plan_id else None,
                workflow_id=str(workflow_id) if workflow_id else None,
                status=str(status) if status else None,
                failed_checks=_failed_checks_from_kwargs(kwargs),
            )
        )

    if not refs and status and status != "executed":
        refs.append(
            TypedRuntimeArtifactReference(
                artifact_type="typed_runtime_failure",
                diagnostic_bundle_path=diagnostic_bundle_path,
                plan_id=str(plan_id) if plan_id else None,
                workflow_id=str(workflow_id) if workflow_id else None,
                status=str(status),
                failed_checks=_failed_checks_from_kwargs(kwargs),
            )
        )
    return refs


def _collect_recent_references(messages: list[Any], *, limit: int = 8) -> list[TypedRuntimeArtifactReference]:
    refs: list[TypedRuntimeArtifactReference] = []
    for message in reversed(messages):
        kwargs = _message_kwargs(message)
        if kwargs.get("strap_origin") != "typed_runtime":
            continue
        refs.extend(_references_from_typed_kwargs(kwargs))
        if len(refs) >= limit:
            break
    return refs[:limit]


def _artifact_type_hints(query: str) -> set[str]:
    hints: set[str] = set()
    for artifact_types, pattern in _ARTIFACT_ALIASES:
        if pattern.search(query):
            hints.update(artifact_types)
    return hints


def _is_followup_query(query: str) -> bool:
    text = query.strip()
    if not text:
        return False
    summary = bool(_SUMMARY_RE.search(text))
    runtime_status = bool(_RUNTIME_STATUS_RE.search(text) and _REFERENTIAL_RE.search(text))
    path_status = bool(_PATH_STATUS_RE.search(text))
    referential = bool(_REFERENTIAL_RE.search(text))
    optimizer_explanation = bool(_OPTIMIZER_EXPLANATION_RE.search(text) and (referential or _artifact_type_hints(text)))
    if runtime_status:
        return True
    if optimizer_explanation:
        return True
    if summary:
        return True
    if path_status and referential:
        return True
    if path_status and not _NEW_REQUEST_RE.search(text):
        return True
    return False


def _looks_like_new_request(query: str) -> bool:
    if _NEW_REQUEST_RE.search(query):
        if _OPTIMIZER_EXPLANATION_RE.search(query) and _REFERENTIAL_RE.search(query):
            return False
        return True
    if _SUMMARY_RE.search(query):
        return False
    if _PATH_STATUS_RE.search(query) and _REFERENTIAL_RE.search(query):
        return False
    return False


def _match_references(
    query: str,
    references: list[TypedRuntimeArtifactReference],
) -> list[TypedRuntimeArtifactReference]:
    hints = _artifact_type_hints(query)
    if not hints:
        return references
    matched = [ref for ref in references if ref.artifact_type in hints]
    if matched and _SUMMARY_RE.search(query):
        matched_ids = {id(ref) for ref in matched}
        matched.extend(
            ref
            for ref in references
            if ref.artifact_type == "typed_runtime_failure" and id(ref) not in matched_ids
        )
    return matched or references


def _format_payload_facts(summary: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    if not summary:
        return facts
    for key, value in summary.items():
        if isinstance(value, dict):
            facts.append(f"{key}: " + ", ".join(f"{k}={v}" for k, v in value.items()))
        elif isinstance(value, list):
            facts.append(f"{key}: {', '.join(str(item) for item in value[:6])}")
        else:
            facts.append(f"{key}: {value}")
    return facts


def format_typed_artifact_summary(
    artifacts: list[TypedRuntimeArtifactReference],
    *,
    include_payload_facts: bool = True,
) -> str:
    """Format a concise summary from prior typed-runtime artifact metadata."""
    if not artifacts:
        return "No prior typed-runtime artifacts were found in this conversation."

    lines = ["Typed runtime artifact summary:"]
    for ref in artifacts:
        status = f", status={ref.status}" if ref.status else ""
        lines.append(f"- {ref.artifact_type}{status}")
        if ref.source_step_id:
            lines.append(f"  Source step: {ref.source_step_id}")
        for path in ref.output_paths:
            lines.append(f"  Path: {path}")
            if ref.diagnostic_copies.get(path):
                lines.append(f"  Diagnostic copy: {ref.diagnostic_copies[path]}")
        if ref.diagnostic_bundle_path:
            lines.append(f"  Diagnostic bundle: {ref.diagnostic_bundle_path}")
        if ref.workflow_id:
            lines.append(f"  Workflow: {ref.workflow_id}")
        if ref.failed_checks:
            lines.append(f"  Failed checks: {', '.join(ref.failed_checks)}")
        if include_payload_facts:
            for fact in _format_payload_facts(ref.payload_summary):
                lines.append(f"  {fact}")
    return "\n".join(lines)


def _format_path_answer(artifacts: list[TypedRuntimeArtifactReference]) -> str:
    if not artifacts:
        return "I found prior typed-runtime metadata, but it does not include artifact paths."
    lines = ["Typed runtime artifact path lookup:"]
    for ref in artifacts:
        lines.append(f"- {ref.artifact_type}")
        if ref.status:
            lines.append(f"  Runtime status: {ref.status}")
        if ref.output_paths:
            for path in ref.output_paths:
                lines.append(f"  Path: {path}")
                if ref.diagnostic_copies.get(path):
                    lines.append(f"  Diagnostic copy: {ref.diagnostic_copies[path]}")
        else:
            lines.append("  Path: not recorded")
        if ref.diagnostic_bundle_path:
            lines.append(f"  Diagnostic bundle: {ref.diagnostic_bundle_path}")
        if ref.workflow_id:
            lines.append(f"  Workflow: {ref.workflow_id}")
    return "\n".join(lines)


def _snapshot_artifact_types(snapshot: TypedRuntimeContextSnapshot) -> list[str]:
    return list(dict.fromkeys(artifact.artifact_type for artifact in snapshot.artifacts))


def _snapshot_records(snapshot: TypedRuntimeContextSnapshot) -> list[dict[str, Any]]:
    ledger = snapshot.ledger if isinstance(snapshot.ledger, dict) else {}
    records = ledger.get("step_records") if isinstance(ledger.get("step_records"), list) else []
    return [record for record in records if isinstance(record, dict)]


def _format_runtime_status_answer(snapshot: TypedRuntimeContextSnapshot) -> str:
    records = _snapshot_records(snapshot)
    completed = [str(record.get("step_id")) for record in records if record.get("status") == "succeeded"]
    failed = [record for record in records if record.get("status") == "failed"]
    tools = [
        str(record.get("callable_name"))
        for record in records
        if record.get("callable_name")
    ]
    artifact_types = _snapshot_artifact_types(snapshot)
    run_dir = snapshot.manifest.get("run_dir") if isinstance(snapshot.manifest, dict) else None

    lines = ["Typed runtime status:"]
    if snapshot.status:
        lines.append(f"- Runtime status: {snapshot.status}")
    if snapshot.plan_id:
        lines.append(f"- Plan: {snapshot.plan_id}")
    if snapshot.workflow_id:
        lines.append(f"- Workflow: {snapshot.workflow_id}")
    if completed:
        lines.append("- Completed steps: " + ", ".join(completed))
    if tools:
        lines.append("- Tools/callables: " + ", ".join(list(dict.fromkeys(tools))))
    if artifact_types:
        lines.append("- Artifact types: " + ", ".join(artifact_types))
    for record in failed:
        checks = record.get("failed_checks")
        check_text = ", ".join(str(item) for item in checks) if isinstance(checks, list) else ""
        lines.append(f"- Failed step: {record.get('step_id')}" + (f" ({check_text})" if check_text else ""))
    if run_dir:
        lines.append(f"- Diagnostic bundle: {run_dir}")
    if snapshot.workflow_id == "routed_optimization":
        lines.append("- Provenance: this used the routed Pareto optimization workflow.")
    elif snapshot.workflow_id:
        lines.append(f"- Provenance: this used workflow `{snapshot.workflow_id}`.")
    direct_markers = {"solubility_curve", "solubility_table", "solvent_safety_card", "solvent_safety_comparison"}
    if not (set(artifact_types) & direct_markers) and not any(
        "solubility" in tool or "safety" in tool for tool in tools
    ):
        lines.append("- Direct fast paths: no direct solubility or safety fast-path artifact was involved.")
    return "\n".join(lines)


def _latest_optimizer_payload(snapshot: TypedRuntimeContextSnapshot) -> dict[str, Any] | None:
    for artifact in reversed(snapshot.artifacts):
        if artifact.artifact_type not in _OPTIMIZATION_ARTIFACT_TYPES:
            continue
        validation = artifact.validation_summary if isinstance(artifact.validation_summary, dict) else {}
        payload = validation.get("payload")
        if isinstance(payload, dict) and (
            isinstance(payload.get("points"), list)
            or isinstance(payload.get("frontier_points"), list)
            or isinstance(payload.get("pareto_points"), list)
        ):
            return payload
    return None


def _optimizer_artifact_path_lines(snapshot: TypedRuntimeContextSnapshot) -> list[str]:
    path_lines: list[str] = []
    for artifact in snapshot.artifacts:
        if artifact.artifact_type not in _OPTIMIZATION_ARTIFACT_TYPES:
            continue
        for path in artifact.output_paths:
            path_lines.append(f"- {artifact.artifact_type}: {path}")
    if not path_lines:
        return []
    lines = ["Artifact paths:"]
    lines.extend(path_lines)
    if snapshot.manifest.get("run_dir"):
        lines.append(f"- Diagnostic bundle: {snapshot.manifest['run_dir']}")
    return lines


def _format_optimizer_artifact_answer(
    snapshot: TypedRuntimeContextSnapshot,
    payload: dict[str, Any],
    query: str,
) -> str:
    x_metric, y_metric = pareto_metric_labels(payload)
    lines = [
        "Typed runtime optimizer artifact answer:",
        "- Source: prior verified typed-runtime optimizer artifact; no new run was started.",
    ]
    if snapshot.plan_id:
        lines.append(f"- Plan: {snapshot.plan_id}")
    if snapshot.workflow_id:
        lines.append(f"- Workflow: {snapshot.workflow_id}")
    lines.extend(
        [
            f"- Frontier points: {pareto_frontier_count(payload)}",
            f"- Feasible landscape points: {pareto_landscape_count(payload)}",
            f"- Metrics: {x_metric} vs {y_metric}",
        ]
    )
    if re.search(r"\b(?:why|choose|chosen|selected|selection|landfill|stage[-\s]?3)\b", query, re.IGNORECASE):
        stage3 = pareto_stage3_mentions(payload)
        if stage3:
            lines.append("- Frontier stage options: " + ", ".join(stage3))
        if re.search(r"\b(?:landfill|lf)\b", query, re.IGNORECASE):
            has_landfill = any(str(item).lower() in {"lf", "landfill"} for item in stage3)
            lines.append(
                "- Landfill check: "
                + (
                    "the verified frontier includes lf/landfill in a process stage."
                    if has_landfill
                    else "the verified frontier does not include lf/landfill in the recorded process stages."
                )
            )
        lines.append(
            "- Interpretation limit: the artifact records selected designs, objectives, constraints, and metric values; "
            "it does not contain a separate causal narrative beyond those fields."
        )
    table = pareto_frontier_table_lines(payload, max_points=6)
    if table:
        lines.extend(table)
    lines.extend(_optimizer_artifact_path_lines(snapshot))
    return "\n".join(lines)


def _optimizer_answer_from_snapshots(
    query: str,
    messages: list[Any],
) -> tuple[str | None, TypedRuntimeContextSnapshot | None]:
    for snapshot in collect_recent_typed_runtime_snapshots(messages, limit=4):
        payload = _latest_optimizer_payload(snapshot)
        if payload is not None:
            return _format_optimizer_artifact_answer(snapshot, payload, query), snapshot
    return None, None


def _progress_for_snapshot(snapshot: TypedRuntimeContextSnapshot) -> dict[str, Any]:
    records = _snapshot_records(snapshot)
    completed = [str(record.get("step_id")) for record in records if record.get("status") == "succeeded"]
    failed_records = [record for record in records if record.get("status") == "failed"]
    failed_checks: list[str] = []
    for record in failed_records:
        checks = record.get("failed_checks")
        if isinstance(checks, list):
            failed_checks.extend(str(item) for item in checks if str(item))
    paths: list[str] = []
    for artifact in snapshot.artifacts:
        paths.extend(artifact.output_paths)
    return {
        "schema_version": "1.0",
        "status": "answered_from_prior_artifacts",
        "plan_id": snapshot.plan_id,
        "workflow_id": snapshot.workflow_id,
        "completed_steps": completed,
        "tool_names": list(dict.fromkeys(str(record.get("callable_name")) for record in records if record.get("callable_name"))),
        "failed_checks": list(dict.fromkeys(failed_checks)),
        "produced_artifact_paths": list(dict.fromkeys(paths)),
        "diagnostic_bundle_paths": [str(snapshot.manifest.get("run_dir"))] if snapshot.manifest.get("run_dir") else [],
        "matched_artifact_types": _snapshot_artifact_types(snapshot),
    }


def _progress_for_answer(artifacts: list[TypedRuntimeArtifactReference]) -> dict[str, Any]:
    paths: list[str] = []
    bundle_paths: list[str] = []
    failed_checks: list[str] = []
    for ref in artifacts:
        paths.extend(ref.output_paths)
        if ref.diagnostic_bundle_path:
            bundle_paths.append(ref.diagnostic_bundle_path)
        failed_checks.extend(ref.failed_checks)
    return {
        "schema_version": "1.0",
        "status": "answered_from_prior_artifacts",
        "matched_artifact_types": [ref.artifact_type for ref in artifacts],
        "produced_artifact_paths": list(dict.fromkeys(paths)),
        "diagnostic_bundle_paths": list(dict.fromkeys(bundle_paths)),
        "failed_checks": list(dict.fromkeys(failed_checks)),
    }


def maybe_answer_typed_runtime_followup(
    query: str,
    messages: list[Any],
) -> TypedRuntimeFollowupDecision:
    """Answer path/status/summary follow-ups from prior typed-runtime metadata."""
    if not _is_followup_query(query) or _looks_like_new_request(query):
        return TypedRuntimeFollowupDecision()

    references = _collect_recent_references(messages)
    if not references:
        return TypedRuntimeFollowupDecision(reason="no_prior_typed_runtime_artifacts")

    optimizer_followup = bool(
        _OPTIMIZER_EXPLANATION_RE.search(query)
        or (_SUMMARY_RE.search(query) and (_artifact_type_hints(query) & _OPTIMIZATION_ARTIFACT_TYPES))
    )
    if optimizer_followup:
        response, snapshot = _optimizer_answer_from_snapshots(query, messages)
        if response is not None and snapshot is not None:
            matched = _match_references(query, references)
            return TypedRuntimeFollowupDecision(
                should_answer=True,
                reason="optimization_answer_from_prior_artifacts",
                response_text=response,
                matched_artifacts=matched,
                progress=_progress_for_snapshot(snapshot),
            )

    if _RUNTIME_STATUS_RE.search(query) and not _SUMMARY_RE.search(query):
        snapshots = collect_recent_typed_runtime_snapshots(messages, limit=1)
        if snapshots:
            snapshot = snapshots[0]
            return TypedRuntimeFollowupDecision(
                should_answer=True,
                reason="runtime_status_from_prior_plan",
                response_text=_format_runtime_status_answer(snapshot),
                matched_artifacts=_match_references(query, references),
                progress=_progress_for_snapshot(snapshot),
            )

    matched = _match_references(query, references)
    if _SUMMARY_RE.search(query):
        response = format_typed_artifact_summary(matched)
        reason = "summary_from_prior_artifacts"
    else:
        response = _format_path_answer(matched)
        reason = "path_status_from_prior_artifacts"

    return TypedRuntimeFollowupDecision(
        should_answer=True,
        reason=reason,
        response_text=response,
        matched_artifacts=matched,
        progress=_progress_for_answer(matched),
    )
