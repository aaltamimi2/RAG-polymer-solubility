"""Shared handoff adapter and JSON-envelope utilities."""

from __future__ import annotations

import json
import logging
from typing import Any

from .handoff_adapters import build_typed_handoff
from .handoff_models import (
    HandoffRecord,
    _slugify,
    extract_artifacts_from_payload,
)

from .handoff_store import (
    _scope_key,
    bind_handoff_scope,
    cleanup_handoff_scope,
    get_current_scope,
    get_handoff,
    get_latest_handoff,
    get_latest_result_handoff,
    get_scope_artifact_dir,
    get_scope_user_query,
    get_tool_call_handoff_statuses,
    initialize_handoff_scope,
    is_result_contract,
    list_handoff_records,
    list_result_records,
    normalize_agent_payload,
    set_handoff_root,
    store_agent_failure,
    store_agent_result,
    store_derived_handoff,
    validate_agent_payload,
)

logger = logging.getLogger(__name__)


def _consumer_guidance(consumer: str) -> str:
    return {
        "biosteam-analyst": "Translate the upstream context into simulation-ready scenarios and compare TEA/LCA implications.",
        "patent-researcher": "Use the upstream context to narrow patent search terms or extract the exact patent angles to investigate.",
        "rag-analyst": "Use the upstream context before any new retrieval and only search further if the payload is insufficient.",
        "safety-analyst": "Focus on the solvent/material safety implications contained in the upstream result.",
        "scholar-researcher": "Use the upstream context to narrow literature search terms or extract the exact questions to investigate.",
        "separation-engineer": "Use the upstream context to refine or extend the separation plan without re-running the upstream task unless necessary.",
        "statistics-ml": "Use the upstream context to perform the requested calculations or screening without repeating the upstream analysis.",
        "visualization-specialist": "Prefer plotting the provided numeric results or reusing upstream artifacts; do not re-run the upstream analysis.",
    }.get(
        consumer,
        "Use your domain tools only as needed; do not repeat the upstream task unless the payload is insufficient.",
    )


def _build_generic_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available_keys": sorted(payload.keys()),
    }
    for key in (
        "analysis_type",
        "best_sequence",
        "energy_case",
        "format",
        "kb_name",
        "operation",
        "plot_type",
        "polymers",
        "query",
        "safest_solvent",
        "schema_version",
        "solvents_assessed",
        "target_plastic",
        "top_sources",
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list) and len(value) > 5:
            summary[key] = value[:5]
            summary[f"{key}_count"] = len(value)
        else:
            summary[key] = value

    for key in (
        "gscore_results",
        "ghs_results",
        "papers",
        "patents",
        "plot_paths",
        "results",
        "steps",
        "top_k_sequences",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
    return summary


def _build_generic_prompt(
    *,
    source: HandoffRecord,
    consumer: str,
    summary: dict[str, Any],
    artifacts: list[str],
) -> str:
    lines = [
        f"Continue the task as {consumer} using the validated upstream handoff from {source.producer}.",
        f"Source handoff ID: {source.handoff_id}",
        f"Source contract: {source.contract}",
        "Treat `payload.source_payload` as authoritative upstream context.",
    ]

    if summary:
        lines.append("Key context:")
        for key, value in summary.items():
            lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")

    if artifacts:
        lines.append("Available artifacts:")
        for artifact in artifacts[:5]:
            lines.append(f"- {artifact}")

    lines.append(_consumer_guidance(consumer))
    return "\n".join(lines)


def _adapt_generic_to_consumer(
    source: HandoffRecord,
    consumer: str,
) -> tuple[str, dict[str, Any], str]:
    artifacts = source.artifacts or extract_artifacts_from_payload(source.payload)
    summary = _build_generic_summary(source.payload)
    contract = f"{_slugify(source.producer, 'producer')}.to.{_slugify(consumer, 'consumer')}.context.v1"
    payload = {
        "source_handoff_id": source.handoff_id,
        "source_contract": source.contract,
        "source_producer": source.producer,
        "source_consumer": source.consumer,
        "source_payload": source.payload,
        "source_summary": summary,
        "artifacts": artifacts,
    }
    task_prompt = _build_generic_prompt(
        source=source,
        consumer=consumer,
        summary=summary,
        artifacts=artifacts,
    )
    return (contract, payload, task_prompt)


def _build_multi_source_prompt(
    *,
    consumer: str,
    source_payloads: list[dict[str, Any]],
    artifacts: list[str],
) -> str:
    lines = [
        f"Continue the task as {consumer} using the validated multi-source upstream handoff.",
        "Treat `payload.source_handoffs` as authoritative upstream context.",
        "Required upstream sources:",
    ]
    for source_payload in source_payloads:
        lines.append(
            f"- {source_payload['producer']} | handoff_id={source_payload['handoff_id']} | contract={source_payload['contract']}"
        )
        summary = source_payload.get("summary") or {}
        if summary:
            lines.append(
                f"  Summary: {json.dumps(summary, ensure_ascii=False)}"
            )
        source_prompt = str(source_payload.get("task_prompt") or "").strip()
        if source_prompt:
            lines.append(f"  Guidance: {source_prompt}")

    if artifacts:
        lines.append("Available artifacts:")
        for artifact in artifacts[:8]:
            lines.append(f"- {artifact}")

    lines.append(_consumer_guidance(consumer))
    return "\n".join(lines)


def build_multi_source_handoff_for_consumer(
    *,
    consumer: str,
    source_handoff_ids: list[str],
    task_prompt: str | None = None,
) -> HandoffRecord:
    """Create or reuse a merged multi-source context handoff for one consumer."""
    unique_source_ids: list[str] = []
    for handoff_id in source_handoff_ids:
        if not isinstance(handoff_id, str) or not handoff_id.strip():
            continue
        normalized = handoff_id.strip()
        if normalized not in unique_source_ids:
            unique_source_ids.append(normalized)
    if len(unique_source_ids) < 2:
        raise ValueError("build_multi_source_handoff_for_consumer requires at least 2 source_handoff_ids")

    unordered_sources: list[HandoffRecord] = []
    for handoff_id in unique_source_ids:
        source = get_handoff(handoff_id)
        if source is None:
            raise ValueError(f"source handoff '{handoff_id}' not found")
        if source.status != "ok":
            raise ValueError(f"source handoff {handoff_id} is {source.status} and cannot be merged")
        unordered_sources.append(source)

    sources = sorted(
        unordered_sources,
        key=lambda source: (
            source.producer,
            source.contract,
            source.handoff_id,
        ),
    )
    canonical_source_ids = [source.handoff_id for source in sources]

    contract = f"multi-source.to.{_slugify(consumer, 'consumer')}.context.v1"
    existing_records = list_handoff_records(
        producer="multi-source",
        consumer=consumer,
        contract=contract,
        status="ok",
    )
    for record in reversed(existing_records):
        if record.parent_handoff_ids == canonical_source_ids:
            return record
        payload_ids = record.payload.get("source_handoff_ids")
        if isinstance(payload_ids, list) and payload_ids == canonical_source_ids:
            return record

    all_artifacts: list[str] = []
    seen_artifacts: set[str] = set()
    source_payloads: list[dict[str, Any]] = []
    for source in sources:
        source_artifacts = source.artifacts or extract_artifacts_from_payload(source.payload)
        for artifact in source_artifacts:
            if artifact not in seen_artifacts:
                seen_artifacts.add(artifact)
                all_artifacts.append(artifact)
        source_payloads.append(
            {
                "handoff_id": source.handoff_id,
                "producer": source.producer,
                "consumer": source.consumer,
                "contract": source.contract,
                "payload": source.payload,
                "summary": _build_generic_summary(source.payload),
                "artifacts": source_artifacts,
                "task_prompt": source.task_prompt,
            }
        )

    payload = {
        "source_handoff_ids": canonical_source_ids,
        "source_handoffs": source_payloads,
        "producers": [source.producer for source in sources],
        "contracts": [source.contract for source in sources],
        "artifacts": all_artifacts,
    }
    merged_task_prompt = task_prompt or _build_multi_source_prompt(
        consumer=consumer,
        source_payloads=source_payloads,
        artifacts=all_artifacts,
    )
    return store_derived_handoff(
        producer="multi-source",
        consumer=consumer,
        contract=contract,
        payload=payload,
        parent_handoff_id=canonical_source_ids[0],
        parent_handoff_ids=canonical_source_ids,
        task_prompt=merged_task_prompt,
        artifacts=all_artifacts,
    )


def build_handoff_for_consumer(
    *,
    consumer: str,
    source_handoff_id: str | None = None,
    producer: str | None = None,
    strategy: str = "latest",
) -> HandoffRecord:
    """Create a derived handoff for a downstream consumer."""
    if source_handoff_id:
        source = get_handoff(source_handoff_id)
    else:
        if not producer:
            raise ValueError("producer is required when source_handoff_id is not provided")
        if strategy != "latest":
            raise ValueError(f"unsupported strategy '{strategy}'")
        source = get_latest_result_handoff(producer=producer)

    if source is None:
        raise ValueError("source handoff not found")
    if source.status != "ok":
        raise ValueError(
            f"source handoff {source.handoff_id} is {source.status} and cannot be adapted"
        )

    typed_handoff = None
    if is_result_contract(source.contract):
        try:
            typed_handoff = build_typed_handoff(
                source,
                consumer,
                scope_user_query=get_scope_user_query(),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                "handoffs: typed adapter failed for %s -> %s (%s); falling back to generic context",
                source.producer,
                consumer,
                exc,
            )

    if typed_handoff is None:
        contract, payload, task_prompt = _adapt_generic_to_consumer(source, consumer)
    else:
        contract, payload, task_prompt = typed_handoff

    return store_derived_handoff(
        producer=source.producer,
        consumer=consumer,
        contract=contract,
        payload=payload,
        parent_handoff_id=source.handoff_id,
        task_prompt=task_prompt,
    )


def record_to_json_envelope(record: HandoffRecord) -> str:
    return json.dumps({"ok": True, "handoff": record.to_dict()}, indent=2)


def records_to_json_envelope(records: list[HandoffRecord]) -> str:
    return json.dumps(
        {
            "ok": True,
            "handoffs": [record.to_dict() for record in records],
        },
        indent=2,
    )
