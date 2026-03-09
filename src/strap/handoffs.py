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

    consumer_guidance = {
        "biosteam-analyst": "Translate the upstream context into simulation-ready scenarios and compare TEA/LCA implications.",
        "patent-researcher": "Use the upstream context to narrow patent search terms or extract the exact patent angles to investigate.",
        "rag-analyst": "Use the upstream context before any new retrieval and only search further if the payload is insufficient.",
        "safety-analyst": "Focus on the solvent/material safety implications contained in the upstream result.",
        "scholar-researcher": "Use the upstream context to narrow literature search terms or extract the exact questions to investigate.",
        "separation-engineer": "Use the upstream context to refine or extend the separation plan without re-running the upstream task unless necessary.",
        "statistics-ml": "Use the upstream context to perform the requested calculations or screening without repeating the upstream analysis.",
        "visualization-specialist": "Prefer plotting the provided numeric results or reusing upstream artifacts; do not re-run the upstream analysis.",
    }
    lines.append(
        consumer_guidance.get(
            consumer,
            "Use your domain tools only as needed; do not repeat the upstream task unless the payload is insufficient.",
        )
    )
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
