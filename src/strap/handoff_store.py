"""Handoff scope binding, validation, and append-only storage."""

from __future__ import annotations

import atexit
import contextvars
import logging
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .handoff_models import (
    HandoffRecord,
    HandoffScope,
    _ScopeState,
    _slugify,
    extract_artifacts_from_payload,
)

logger = logging.getLogger(__name__)

_DEFAULT_HANDOFF_ROOT: Path | None = None
_RESULT_CONTRACT_SUFFIX = ".result.v1"

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


_scope_key: contextvars.ContextVar[str] = contextvars.ContextVar("_handoff_scope_key")
_states_lock = threading.Lock()
_states: dict[str, _ScopeState] = {}


_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "separation-engineer": (
        "agent",
        "schema_version",
        "polymers",
        "best_sequence",
        "steps",
        "solvent_mapping",
        "top_k_sequences",
    ),
    "safety-analyst": (
        "agent",
        "schema_version",
        "solvents_assessed",
    ),
    "biosteam-analyst": (
        "agent",
        "schema_version",
        "target_plastic",
        "energy_case",
        "results",
        "n_simulations",
        "n_failed",
    ),
    "scholar-researcher": (
        "agent",
        "schema_version",
        "query",
        "n_results",
        "papers",
    ),
    "patent-researcher": (
        "agent",
        "schema_version",
        "query",
        "n_results",
        "patents",
    ),
    "rag-analyst": (
        "agent",
        "schema_version",
        "operation",
    ),
    "visualization-specialist": (
        "agent",
        "schema_version",
        "plot_type",
        "plot_paths",
        "format",
    ),
    "statistics-ml": (
        "agent",
        "schema_version",
        "analysis_type",
    ),
    "contaminant-removal-analyst": (
        "agent",
        "schema_version",
        "mode",
        "target_polymer",
        "contaminants",
        "candidate_solvents",
        "recommended_solvents",
    ),
}

# At least one field from each group must be present. Safety results may be
# card-style (schema 1.1 `safety_cards`) or legacy gscore/GHS lists.
_ANY_OF_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "safety-analyst": (("safety_cards", "gscore_results", "ghs_results"),),
}

_CONTRACT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "optimization.stage_candidates.v1": (
        "schema_version",
        "workflow_scope",
        "route_id",
        "constraint_mode",
        "fallback_policy",
        "operating_constraints",
        "stages",
        "candidate_pairs",
        "polymer_solvent_filters",
        "candidate_solvents",
    ),
    "optimization.infeasible.v1": (
        "schema_version",
        "analysis_type",
        "constraint_mode",
        "fallback_policy",
        "failure_reason",
        "message",
        "requested_candidate_pairs",
        "applied_candidate_pairs",
        "suggested_relaxation",
    ),
}


def set_handoff_root(path: Path) -> None:
    """Set the base directory used for namespaced handoff artifacts."""
    global _DEFAULT_HANDOFF_ROOT
    path.mkdir(parents=True, exist_ok=True)
    _DEFAULT_HANDOFF_ROOT = path
    logger.info("Handoff root directory: %s", path)


def _get_handoff_root() -> Path:
    global _DEFAULT_HANDOFF_ROOT
    if _DEFAULT_HANDOFF_ROOT is None:
        _DEFAULT_HANDOFF_ROOT = Path(tempfile.mkdtemp(prefix="strap_handoffs_"))
        atexit.register(shutil.rmtree, _DEFAULT_HANDOFF_ROOT, True)
        logger.info(
            "handoffs: created fallback artifact root %s",
            _DEFAULT_HANDOFF_ROOT,
        )
    return _DEFAULT_HANDOFF_ROOT


def initialize_handoff_scope(
    *,
    run_id: str | None = None,
    thread_id: str | None = None,
    invocation_id: str | None = None,
    artifact_root: Path | None = None,
    user_query: str | None = None,
) -> HandoffScope:
    """Create and bind a scope for the current execution context."""
    scope = HandoffScope(
        invocation_id=invocation_id or uuid.uuid4().hex,
        run_id=str(run_id or invocation_id or uuid.uuid4().hex),
        thread_id=str(thread_id or "threadless"),
    )
    _scope_key.set(scope.scope_id)
    root = artifact_root or _get_handoff_root()
    root.mkdir(parents=True, exist_ok=True)
    with _states_lock:
        existing = _states.get(scope.scope_id)
        if existing is not None:
            if existing.user_query is None and user_query:
                existing.user_query = user_query
            if existing.artifact_root == root:
                logger.debug("handoffs: rebound existing scope %s", scope.scope_id)
                return existing.scope
            logger.debug(
                "handoffs: resetting scope %s for new artifact root %s",
                scope.scope_id,
                root,
            )
        _states[scope.scope_id] = _ScopeState(
            scope=scope,
            artifact_root=root,
            user_query=user_query or (existing.user_query if existing else None),
        )
    logger.debug("handoffs: initialized scope %s", scope.scope_id)
    return scope


def bind_handoff_scope(
    scope: HandoffScope,
    *,
    artifact_root: Path | None = None,
    user_query: str | None = None,
) -> HandoffScope:
    """Re-bind a previously created scope in the current execution context."""
    _scope_key.set(scope.scope_id)
    with _states_lock:
        existing = _states.get(scope.scope_id)
        if existing is not None:
            if existing.user_query is None and user_query:
                existing.user_query = user_query
            logger.debug("handoffs: rebound cached scope %s", scope.scope_id)
            return existing.scope

    root = artifact_root or _get_handoff_root()
    root.mkdir(parents=True, exist_ok=True)
    with _states_lock:
        _states[scope.scope_id] = _ScopeState(
            scope=scope,
            artifact_root=root,
            user_query=user_query,
        )
    logger.debug("handoffs: restored missing scope state %s", scope.scope_id)
    return scope


def cleanup_handoff_scope() -> None:
    """Drop the current in-memory scope state."""
    try:
        scope_id = _scope_key.get()
    except LookupError:
        return
    with _states_lock:
        _states.pop(scope_id, None)
    logger.debug("handoffs: cleaned up scope %s", scope_id)


def _get_current_state() -> _ScopeState | None:
    try:
        scope_id = _scope_key.get()
    except LookupError:
        return None
    with _states_lock:
        return _states.get(scope_id)


def get_current_scope() -> HandoffScope | None:
    state = _get_current_state()
    return state.scope if state else None


def snapshot_handoff_scope_state() -> dict[str, Any]:
    """Return a serializable snapshot of the current in-memory handoff scope."""
    state = _get_current_state()
    if state is None:
        raise RuntimeError("No active handoff scope.")

    return {
        "scope": asdict(state.scope),
        "artifact_root": str(state.artifact_root),
        "user_query": state.user_query,
        "handoffs": [record.to_dict() for record in state.handoffs],
    }


def restore_handoff_scope_state(snapshot: dict[str, Any]) -> HandoffScope:
    """Restore a previously snapshotted handoff scope into the current process."""
    scope_data = snapshot.get("scope")
    if not isinstance(scope_data, dict):
        raise ValueError("restore_handoff_scope_state requires a mapping 'scope'")

    try:
        scope = HandoffScope(
            invocation_id=str(scope_data["invocation_id"]),
            run_id=str(scope_data["run_id"]),
            thread_id=str(scope_data["thread_id"]),
        )
    except KeyError as exc:
        raise ValueError(f"restore_handoff_scope_state missing scope field: {exc.args[0]}") from exc

    artifact_root = Path(str(snapshot.get("artifact_root") or _get_handoff_root()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    restored_records: list[HandoffRecord] = []
    raw_records = snapshot.get("handoffs")
    if raw_records is not None and not isinstance(raw_records, list):
        raise ValueError("restore_handoff_scope_state requires 'handoffs' to be a list")

    for raw_record in raw_records or []:
        if not isinstance(raw_record, dict):
            raise ValueError("restore_handoff_scope_state handoff records must be mappings")
        raw_scope = raw_record.get("scope")
        if not isinstance(raw_scope, dict):
            raise ValueError("restore_handoff_scope_state handoff record missing nested scope")
        record_scope = HandoffScope(
            invocation_id=str(raw_scope["invocation_id"]),
            run_id=str(raw_scope["run_id"]),
            thread_id=str(raw_scope["thread_id"]),
        )
        restored_records.append(
            HandoffRecord(
                handoff_id=str(raw_record["handoff_id"]),
                scope=record_scope,
                producer=str(raw_record["producer"]),
                consumer=str(raw_record["consumer"]),
                contract=str(raw_record["contract"]),
                status=str(raw_record["status"]),
                payload=dict(raw_record.get("payload") or {}),
                created_at=str(raw_record["created_at"]),
                source_tool_call_id=raw_record.get("source_tool_call_id"),
                parent_handoff_id=raw_record.get("parent_handoff_id"),
                parent_handoff_ids=list(raw_record.get("parent_handoff_ids") or []),
                validation_errors=list(raw_record.get("validation_errors") or []),
                artifacts=list(raw_record.get("artifacts") or []),
                task_prompt=raw_record.get("task_prompt"),
            )
        )

    _scope_key.set(scope.scope_id)
    with _states_lock:
        _states[scope.scope_id] = _ScopeState(
            scope=scope,
            artifact_root=artifact_root,
            user_query=str(snapshot.get("user_query")) if snapshot.get("user_query") is not None else None,
            handoffs=restored_records,
        )
    logger.debug("handoffs: restored snapshot for scope %s with %d records", scope.scope_id, len(restored_records))
    return scope


def get_scope_user_query() -> str | None:
    state = _get_current_state()
    return None if state is None else state.user_query


def get_scope_artifact_dir() -> Path:
    """Return the directory used for versioned sidecar artifacts for this scope."""
    state = _get_current_state()
    if state is None:
        raise RuntimeError("No active handoff scope.")
    scope = state.scope
    root = state.artifact_root
    path = (
        root
        / _slugify(scope.thread_id, "threadless")
        / _slugify(scope.run_id, scope.invocation_id)
        / "sidecars"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_agent_payload(producer: str, payload: dict[str, Any]) -> list[str]:
    """Validate a subagent structured result against the expected fields."""
    payload = normalize_agent_payload(producer, payload)
    if payload.get("no_data") is True:
        return []

    errors: list[str] = []
    expected_agent = payload.get("agent")
    if expected_agent != producer:
        errors.append(
            f"payload agent mismatch: expected '{producer}', got '{expected_agent}'"
        )

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        errors.append("missing schema_version")

    for field_name in _REQUIRED_FIELDS.get(producer, ()):
        if field_name not in payload:
            errors.append(f"missing required field '{field_name}'")

    for group in _ANY_OF_FIELDS.get(producer, ()):
        if not any(field_name in payload for field_name in group):
            errors.append(f"missing all of the alternative fields {group}; at least one is required")

    if producer == "separation-engineer" and "top_k_sequences" in payload:
        if not isinstance(payload["top_k_sequences"], list) or not payload["top_k_sequences"]:
            errors.append("top_k_sequences must be a non-empty list")

    return errors


def normalize_agent_payload(producer: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize producer-specific structured result variants before validation/storage."""
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)

    if (
        producer == "contaminant-removal-analyst"
        and normalized.get("mode") == "comparison"
    ):
        modes = normalized.get("modes")
        if not isinstance(modes, dict):
            inferred_modes: dict[str, Any] = {}
            for mode_name in ("leaching", "strap_contaminant_removal"):
                mode_payload = normalized.get(mode_name)
                if isinstance(mode_payload, dict):
                    inferred_modes[mode_name] = mode_payload
            if inferred_modes:
                modes = inferred_modes
                normalized["modes"] = inferred_modes
        if not isinstance(modes, dict):
            return normalized
        if "candidate_solvents" not in normalized:
            flattened: list[dict[str, Any]] = []
            for mode_name, mode_payload in modes.items():
                if not isinstance(mode_payload, dict):
                    continue
                for candidate in mode_payload.get("candidate_solvents", []) or []:
                    if not isinstance(candidate, dict):
                        continue
                    row = dict(candidate)
                    row.setdefault("screen_mode", mode_name)
                    flattened.append(row)
            normalized["candidate_solvents"] = flattened
        recommended_solvents = normalized.get("recommended_solvents")
        if isinstance(recommended_solvents, dict):
            recommended_mode = normalized.get("recommended_mode")
            selected = recommended_solvents.get(recommended_mode) if recommended_mode else None
            if isinstance(selected, list):
                normalized["recommended_solvents"] = [
                    str(item) for item in selected if str(item).strip()
                ]
            else:
                normalized.pop("recommended_solvents", None)
        if "recommended_solvents" not in normalized:
            recommended_mode = normalized.get("recommended_mode")
            recommended: list[str] = []
            selected_payload = modes.get(recommended_mode) if recommended_mode else None
            if isinstance(selected_payload, dict):
                selected_recommended = selected_payload.get("recommended_solvents")
                if isinstance(selected_recommended, list):
                    recommended = [str(item) for item in selected_recommended if str(item).strip()]
            if not recommended:
                recommended = [
                    str(row.get("solvent"))
                    for row in normalized.get("candidate_solvents", [])
                    if isinstance(row, dict)
                    and row.get("passes")
                    and row.get("screen_mode") == recommended_mode
                    and str(row.get("solvent", "")).strip()
                ]
            normalized["recommended_solvents"] = recommended

    if producer == "separation-engineer":
        requested_polymers = {
            str(polymer).strip().upper()
            for polymer in normalized.get("polymers", []) or []
            if str(polymer).strip()
        }
        declared_supported = {
            str(polymer).strip().upper()
            for polymer in normalized.get("supported_polymers", []) or []
            if str(polymer).strip()
        }
        if isinstance(normalized.get("unsupported_polymers"), list):
            normalized["unsupported_polymers"] = [
                str(polymer).strip()
                for polymer in normalized.get("unsupported_polymers", [])
                if str(polymer).strip().upper() in requested_polymers
                and str(polymer).strip().upper() not in declared_supported
            ]

    return normalized


def is_result_contract(contract: str) -> bool:
    return contract.endswith(_RESULT_CONTRACT_SUFFIX)


def validate_contract_payload(contract: str, payload: dict[str, Any]) -> list[str]:
    """Validate a derived typed handoff payload against its registered contract."""
    if not isinstance(payload, dict):
        return ["contract payload must be a mapping"]

    errors: list[str] = []
    for field_name in _CONTRACT_REQUIRED_FIELDS.get(contract, ()):
        if field_name not in payload:
            errors.append(f"missing required field '{field_name}'")

    if contract == "optimization.stage_candidates.v1":
        workflow_scope = payload.get("workflow_scope")
        if workflow_scope not in {"single_stage", "multi_stage", "full_process"}:
            errors.append("workflow_scope must be one of: single_stage, multi_stage, full_process")

        constraint_mode = payload.get("constraint_mode")
        if constraint_mode not in {"fixed", "hard", "soft", "ranked_soft"}:
            errors.append("constraint_mode must be one of: fixed, hard, soft, ranked_soft")

        fallback_policy = payload.get("fallback_policy")
        if fallback_policy not in {"fail_closed", "broaden_silent", "broaden_disclosed"}:
            errors.append("fallback_policy must be one of: fail_closed, broaden_silent, broaden_disclosed")

        operating_constraints = payload.get("operating_constraints")
        if not isinstance(operating_constraints, dict):
            errors.append("operating_constraints must be a mapping")

        stages = payload.get("stages")
        if not isinstance(stages, list) or not stages:
            errors.append("stages must be a non-empty list")
        else:
            for idx, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    errors.append(f"stages[{idx}] must be a mapping")
                    continue
                for field_name in ("stage_id", "stage_kind", "target_polymer", "candidate_pairs"):
                    if field_name not in stage:
                        errors.append(f"stages[{idx}] missing required field '{field_name}'")

        candidate_pairs = payload.get("candidate_pairs")
        if not isinstance(candidate_pairs, list):
            errors.append("candidate_pairs must be a list")

        polymer_solvent_filters = payload.get("polymer_solvent_filters")
        if not isinstance(polymer_solvent_filters, dict):
            errors.append("polymer_solvent_filters must be a mapping")

        candidate_solvents = payload.get("candidate_solvents")
        if not isinstance(candidate_solvents, list):
            errors.append("candidate_solvents must be a list")

    elif contract == "optimization.infeasible.v1":
        if payload.get("analysis_type") != "infeasible":
            errors.append("analysis_type must be 'infeasible'")
        if payload.get("constraint_mode") not in {"fixed", "hard", "soft", "ranked_soft"}:
            errors.append("constraint_mode must be one of: fixed, hard, soft, ranked_soft")
        if payload.get("fallback_policy") not in {"fail_closed", "broaden_silent", "broaden_disclosed"}:
            errors.append("fallback_policy must be one of: fail_closed, broaden_silent, broaden_disclosed")

    return errors


def store_agent_result(
    *,
    producer: str,
    payload: dict[str, Any],
    source_tool_call_id: str | None = None,
    task_prompt: str | None = None,
) -> HandoffRecord:
    """Store one extracted subagent result as an append-only handoff record."""
    state = _get_current_state()
    if state is None:
        raise RuntimeError("No active handoff scope.")

    payload = normalize_agent_payload(producer, payload)
    errors = validate_agent_payload(producer, payload)
    record = HandoffRecord(
        handoff_id=f"h_{uuid.uuid4().hex[:12]}",
        scope=state.scope,
        producer=producer,
        consumer="orchestrator",
        contract=f"{producer}.result.v1",
        status="ok" if not errors else "invalid",
        payload=payload,
        created_at=_utcnow(),
        source_tool_call_id=source_tool_call_id,
        validation_errors=errors,
        artifacts=extract_artifacts_from_payload(payload),
        task_prompt=task_prompt,
    )

    with _states_lock:
        state.handoffs.append(record)

    logger.info(
        "handoffs: stored %s for %s status=%s",
        record.handoff_id,
        producer,
        record.status,
    )
    return record


def store_agent_failure(
    *,
    producer: str,
    error_kind: str,
    message: str,
    source_tool_call_id: str | None = None,
    raw_text: str | None = None,
    task_prompt: str | None = None,
) -> HandoffRecord:
    """Store a failed or incomplete subagent handoff attempt."""
    state = _get_current_state()
    if state is None:
        raise RuntimeError("No active handoff scope.")

    payload: dict[str, Any] = {
        "agent": producer,
        "error_kind": error_kind,
        "message": message,
    }
    if raw_text:
        payload["raw_text_preview"] = raw_text[:1000]

    record = HandoffRecord(
        handoff_id=f"h_{uuid.uuid4().hex[:12]}",
        scope=state.scope,
        producer=producer,
        consumer="orchestrator",
        contract=f"{producer}.result.v1",
        status="missing",
        payload=payload,
        created_at=_utcnow(),
        source_tool_call_id=source_tool_call_id,
        validation_errors=[message],
        task_prompt=task_prompt,
    )

    with _states_lock:
        state.handoffs.append(record)

    logger.info(
        "handoffs: stored %s for %s status=%s (%s)",
        record.handoff_id,
        producer,
        record.status,
        error_kind,
    )
    return record


def store_derived_handoff(
    *,
    producer: str,
    consumer: str,
    contract: str,
    payload: dict[str, Any],
    parent_handoff_id: str,
    parent_handoff_ids: list[str] | None = None,
    task_prompt: str,
    artifacts: list[str] | None = None,
) -> HandoffRecord:
    """Store an adapter-produced handoff for a downstream consumer."""
    state = _get_current_state()
    if state is None:
        raise RuntimeError("No active handoff scope.")

    normalized_parent_ids: list[str] = []
    for handoff_id in parent_handoff_ids or [parent_handoff_id]:
        if not isinstance(handoff_id, str) or not handoff_id.strip():
            continue
        normalized = handoff_id.strip()
        if normalized not in normalized_parent_ids:
            normalized_parent_ids.append(normalized)
    primary_parent_id = parent_handoff_id.strip() if isinstance(parent_handoff_id, str) and parent_handoff_id.strip() else None
    if primary_parent_id is None and normalized_parent_ids:
        primary_parent_id = normalized_parent_ids[0]
    if primary_parent_id and primary_parent_id not in normalized_parent_ids:
        normalized_parent_ids.insert(0, primary_parent_id)

    contract_errors = validate_contract_payload(contract, payload)
    record = HandoffRecord(
        handoff_id=f"h_{uuid.uuid4().hex[:12]}",
        scope=state.scope,
        producer=producer,
        consumer=consumer,
        contract=contract,
        status="ok" if not contract_errors else "invalid",
        payload=payload,
        created_at=_utcnow(),
        parent_handoff_id=primary_parent_id,
        parent_handoff_ids=normalized_parent_ids,
        task_prompt=task_prompt,
        artifacts=artifacts or extract_artifacts_from_payload(payload),
        validation_errors=contract_errors,
    )
    with _states_lock:
        state.handoffs.append(record)
    logger.info(
        "handoffs: derived %s for %s -> %s (%s)",
        record.handoff_id,
        producer,
        consumer,
        contract,
    )
    return record


def _select_records(
    *,
    producer: str | None = None,
    consumer: str | None = None,
    contract: str | None = None,
    status: str | None = None,
) -> list[HandoffRecord]:
    state = _get_current_state()
    if state is None:
        return []

    records = list(state.handoffs)
    if producer is not None:
        records = [r for r in records if r.producer == producer]
    if consumer is not None:
        records = [r for r in records if r.consumer == consumer]
    if contract is not None:
        records = [r for r in records if r.contract == contract]
    if status is not None:
        records = [r for r in records if r.status == status]
    return records


def get_handoff(handoff_id: str) -> HandoffRecord | None:
    state = _get_current_state()
    if state is None:
        return None
    for record in state.handoffs:
        if record.handoff_id == handoff_id:
            return record
    return None


def list_handoff_records(
    *,
    producer: str | None = None,
    consumer: str | None = None,
    contract: str | None = None,
    status: str | None = None,
) -> list[HandoffRecord]:
    return _select_records(
        producer=producer,
        consumer=consumer,
        contract=contract,
        status=status,
    )


def get_latest_handoff(
    *,
    producer: str,
    contract: str | None = None,
    status: str | None = None,
) -> HandoffRecord | None:
    records = _select_records(producer=producer, contract=contract, status=status)
    return records[-1] if records else None


def list_result_records(
    *,
    producer: str | None = None,
    status: str | None = None,
) -> list[HandoffRecord]:
    records = _select_records(producer=producer, status=status)
    return [record for record in records if is_result_contract(record.contract)]


def get_latest_result_handoff(
    *,
    producer: str,
    status: str | None = None,
) -> HandoffRecord | None:
    records = list_result_records(producer=producer, status=status)
    return records[-1] if records else None


def get_tool_call_handoff_statuses() -> dict[str, str]:
    """Return the latest handoff status keyed by source task() tool call ID."""
    state = _get_current_state()
    if state is None:
        return {}

    statuses: dict[str, str] = {}
    for record in state.handoffs:
        if record.source_tool_call_id:
            statuses[record.source_tool_call_id] = record.status
    return statuses
