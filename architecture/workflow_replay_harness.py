"""Deterministic workflow replay harness for routed multi-agent plans.

This harness replays routed workflows without invoking any LLM/model path.
It uses the real routing plan, real handoff store, and real build_handoff
adapters, but supplies canned subagent outputs.

The goal is to test more than initial planning:
- root dispatch sequencing
- completion tracking from task ToolMessages
- pending handoff detection
- typed/generic handoff construction
- downstream dispatch gating
- branch/join behavior
- deadlock behavior after missing upstream outputs
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from strap.handoff_store import (
    cleanup_handoff_scope,
    initialize_handoff_scope,
    list_handoff_records,
    restore_handoff_scope_state,
    snapshot_handoff_scope_state,
    store_agent_failure,
    store_agent_result,
)
from strap.planning_graph import build_planning_graph
from strap.result_extractor import build_handoff
from strap.routing import RoutingMiddleware
from strap.routing_classifier import ROUTING_RULES, derive_workflow_dependencies
from strap.routing_guards import (
    _build_initial_route_task_response,
    _build_pending_handoff_response,
    _build_ready_handoff_response,
)
from strap.routing_handoff_state import (
    _get_pending_required_handoff,
    _get_ready_downstream_handoff,
)
from strap.routing_message_state import (
    _extract_all_task_subagents,
    _extract_completed_subagents,
    _extract_failed_subagent_calls,
    _extract_task_dispatches,
    _get_active_remaining_steps,
    _get_latest_dispatch_for_subagent,
    _get_ordered_plan,
)
from strap.testing_utils import block_model_access


@dataclass(frozen=True)
class ReplayOutcome:
    kind: Literal["ok", "missing"]
    payload: dict[str, Any] | None = None
    raw_text: str = ""


@dataclass(frozen=True)
class ReplayExpectation:
    status: Literal["complete", "deadlock"]
    completed_subagents: tuple[str, ...]
    missing_direct_handoffs: tuple[tuple[str, str], ...] = ()
    expected_contracts: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class WorkflowReplayCase:
    name: str
    query: str
    outcomes: dict[str, ReplayOutcome | tuple[ReplayOutcome, ...] | list[ReplayOutcome]]
    expectation: ReplayExpectation
    allowed_subagents: tuple[str, ...] = ()


@dataclass
class ReplayEvent:
    kind: str
    actor: str
    detail: str


@dataclass
class WorkflowReplayResult:
    name: str
    ok: bool
    status: str
    reason: str
    allowed_subagents: list[str]
    ordered_plan: list[dict[str, Any]]
    completed_subagents: list[str]
    failed_subagents: list[str]
    built_handoff_edges: list[tuple[str, str]]
    built_handoffs: list[dict[str, Any]]
    stored_multi_source_handoffs: list[dict[str, Any]]
    missing_direct_handoffs: list[tuple[str, str]]
    events: list[ReplayEvent]


@dataclass
class WorkflowReplayCheckpoint:
    case: WorkflowReplayCase
    allowed_subagents: list[str]
    status: str
    messages: list[Any]
    pending_outcomes: dict[str, list[ReplayOutcome]]
    events: list[ReplayEvent]
    scope_snapshot: dict[str, Any]


@dataclass
class WorkflowReplaySummary:
    total: int
    passed: int
    failed: int
    blocked_model_call_attempts: dict[str, int]
    results: list[WorkflowReplayResult]


def _structured_result_block(payload: dict[str, Any]) -> str:
    return f"<STRUCTURED_RESULT>\n{json.dumps(payload, indent=2, sort_keys=True)}\n</STRUCTURED_RESULT>"


def _default_separation_payload(polymers: list[str]) -> dict[str, Any]:
    sequence = list(polymers)
    solvent_mapping = {
        polymer: solvent
        for polymer, solvent in zip(
            sequence[:-1] or sequence,
            ["toluene", "xylene", "thf", "dmso"],
            strict=False,
        )
    }
    steps = [
        {
            "step": idx,
            "polymer": polymer,
            "solvent": solvent_mapping[polymer],
            "temperature_c": 95.0 + idx * 5.0,
            "selectivity_pct": 80.0 - idx,
        }
        for idx, polymer in enumerate(sequence[:-1] or sequence, start=1)
        if polymer in solvent_mapping
    ]
    return {
        "agent": "separation-engineer",
        "schema_version": "1.0",
        "polymers": sequence,
        "best_sequence": sequence,
        "steps": steps,
        "solvent_mapping": solvent_mapping,
        "top_solvents": list(dict.fromkeys(solvent_mapping.values())),
        "top_k_sequences": [
            {
                "rank": 1,
                "sequence": sequence,
                "min_selectivity": 72.0,
                "solvent_mapping": solvent_mapping,
            }
        ],
    }


def _default_contaminant_payload(
    *,
    target_polymer: str,
    contaminants: list[str],
    other_polymers: list[str],
    solvents: list[str],
) -> dict[str, Any]:
    candidate_solvents = [
        {"solvent": solvent, "screen_mode": "leaching", "passes": True}
        for solvent in solvents
    ]
    return {
        "agent": "contaminant-removal-analyst",
        "schema_version": "1.0",
        "mode": "comparison",
        "target_polymer": target_polymer,
        "other_polymers": other_polymers,
        "contaminants": contaminants,
        "supported_contaminants": contaminants,
        "unsupported_contaminants": [],
        "candidate_solvents": candidate_solvents,
        "recommended_solvents": solvents[:2] or solvents,
        "decision_basis": ["screened with canned replay payload"],
        "caveats": [],
    }


def _default_biosteam_payload(target_plastic: str, solvent: str) -> dict[str, Any]:
    return {
        "agent": "biosteam-analyst",
        "schema_version": "1.0",
        "target_plastic": target_plastic,
        "energy_case": "C1",
        "results": [
            {
                "solvent": solvent,
                "msp": 1.23,
                "tci": 1_500_000.0,
                "aoc": 420_000.0,
                "gwp": 2.8,
            }
        ],
        "n_simulations": 1,
        "n_failed": 0,
    }


def _default_scholar_payload(query: str) -> dict[str, Any]:
    return {
        "agent": "scholar-researcher",
        "schema_version": "1.0",
        "query": query,
        "n_results": 2,
        "papers": [
            {"title": "Paper A", "year": 2024},
            {"title": "Paper B", "year": 2023},
        ],
        "saved_to_rag": True,
    }


def _default_patent_payload(query: str) -> dict[str, Any]:
    return {
        "agent": "patent-researcher",
        "schema_version": "1.0",
        "query": query,
        "n_results": 2,
        "patents": [
            {"title": "Patent A", "number": "US1234567"},
            {"title": "Patent B", "number": "US2345678"},
        ],
        "saved_to_rag": True,
    }


def _default_rag_payload() -> dict[str, Any]:
    return {
        "agent": "rag-analyst",
        "schema_version": "1.0",
        "operation": "answer_question",
        "answer": "Canned synthesis from retrieved sources.",
    }


def _default_visualization_payload(plot_type: str = "comparison_dashboard") -> dict[str, Any]:
    return {
        "agent": "visualization-specialist",
        "schema_version": "1.0",
        "plot_type": plot_type,
        "plot_paths": [f"/plots/{plot_type}_{uuid4().hex[:8]}.png"],
        "format": "png",
    }


def _default_statistics_payload() -> dict[str, Any]:
    return {
        "agent": "statistics-ml",
        "schema_version": "1.0",
        "analysis_type": "tg_lookup",
        "summary": "Polycarbonate Tg summary.",
    }


def _default_safety_payload() -> dict[str, Any]:
    return {
        "agent": "safety-analyst",
        "schema_version": "1.0",
        "solvents_assessed": ["toluene", "xylene"],
        "gscore_results": [{"solvent": "toluene", "g_score": 6.4}],
        "ghs_results": [{"solvent": "toluene", "hazard_codes": ["H225"]}],
        "safest_solvent": "xylene",
    }


_DEFAULT_PAYLOADS: dict[str, Any] = {
    "biosteam-analyst": lambda: _default_biosteam_payload("HDPE", "toluene"),
    "contaminant-removal-analyst": lambda: _default_contaminant_payload(
        target_polymer="HDPE",
        other_polymers=["EVOH"],
        contaminants=["Phthalates"],
        solvents=["toluene", "xylene"],
    ),
    "patent-researcher": lambda: _default_patent_payload("graph-derived replay case"),
    "rag-analyst": _default_rag_payload,
    "safety-analyst": _default_safety_payload,
    "scholar-researcher": lambda: _default_scholar_payload("graph-derived replay case"),
    "separation-engineer": lambda: _default_separation_payload(["HDPE", "EVOH"]),
    "statistics-ml": _default_statistics_payload,
    "visualization-specialist": _default_visualization_payload,
}

_TYPED_CONTRACTS: dict[tuple[str, str], str] = {
    ("biosteam-analyst", "visualization-specialist"): "biosteam_plot.v1",
    ("contaminant-removal-analyst", "biosteam-analyst"): "contaminant_biosteam.v1",
    ("contaminant-removal-analyst", "separation-engineer"): "contaminant_guided_separation.v1",
    ("patent-researcher", "rag-analyst"): "patent_context.v1",
    ("scholar-researcher", "rag-analyst"): "literature_context.v1",
    ("separation-engineer", "biosteam-analyst"): "sequence_batch.v1",
    ("separation-engineer", "contaminant-removal-analyst"): "contaminant_screen.v1",
    ("separation-engineer", "visualization-specialist"): "separation_plot.v1",
    ("statistics-ml", "visualization-specialist"): "analysis_plot.v1",
}


def _expected_contract(producer: str, consumer: str) -> str:
    return _TYPED_CONTRACTS.get((producer, consumer), f"{producer}.to.{consumer}.context.v1")


def _default_outcome_for_subagent(subagent: str) -> ReplayOutcome:
    payload_factory = _DEFAULT_PAYLOADS.get(subagent)
    if payload_factory is None:
        raise KeyError(f"No default replay payload configured for {subagent}")
    payload = payload_factory()
    if not isinstance(payload, dict):
        raise TypeError(f"Replay payload factory for {subagent} must return a dict")
    return ReplayOutcome("ok", payload)


def _rule_subset(names: tuple[str, ...]) -> list[dict]:
    rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
    return [rules_by_name[name] for name in names]


def _build_graph_derived_replay_cases() -> list[WorkflowReplayCase]:
    graph = build_planning_graph()
    cases: list[WorkflowReplayCase] = []
    seen_case_names: set[str] = set()
    for edge in graph.capability_edges:
        if edge.producer not in _DEFAULT_PAYLOADS or edge.consumer not in _DEFAULT_PAYLOADS:
            continue
        dependency_map = derive_workflow_dependencies(
            f"Graph-derived replay coverage for {edge.producer} and {edge.consumer}.",
            {edge.producer, edge.consumer},
        )
        oriented_producer = edge.producer
        oriented_consumer = edge.consumer
        if edge.producer in dependency_map.get(edge.consumer, set()):
            oriented_producer = edge.producer
            oriented_consumer = edge.consumer
        elif edge.consumer in dependency_map.get(edge.producer, set()):
            oriented_producer = edge.consumer
            oriented_consumer = edge.producer
        else:
            continue
        case_name = f"edge-{oriented_producer}-to-{oriented_consumer}"
        if case_name in seen_case_names:
            continue
        seen_case_names.add(case_name)
        cases.append(
            WorkflowReplayCase(
                name=case_name,
                query=f"Graph-derived replay coverage for {oriented_producer} -> {oriented_consumer}.",
                allowed_subagents=(oriented_producer, oriented_consumer),
                outcomes={
                    oriented_producer: _default_outcome_for_subagent(oriented_producer),
                    oriented_consumer: _default_outcome_for_subagent(oriented_consumer),
                },
                expectation=ReplayExpectation(
                    status="complete",
                    completed_subagents=(oriented_producer, oriented_consumer),
                    expected_contracts=((oriented_producer, oriented_consumer, _expected_contract(oriented_producer, oriented_consumer)),),
                ),
            )
        )
    cases.sort(key=lambda case: case.name)
    return cases


def build_workflow_replay_cases() -> list[WorkflowReplayCase]:
    research_query = (
        "Do a literature search and patent search for multilayer polymer recycling methods, "
        "answer the question with RAG, then create a chart visualization of the retrieved findings."
    )
    seq_contam_query = (
        "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream using selective dissolution "
        "at atmospheric pressure. Propose up to 1 additional wash step for phthalate removal. Then run a "
        "techno-economic analysis on solvent recovery for the best option."
    )
    seq_bio_viz_query = (
        "Find an optimal separation sequence for LDPE and EVOH at atmospheric pressure. Then run a "
        "techno-economic analysis on solvent recovery for the best option and create a chart of the TEA results."
    )
    stats_viz_query = "Look up Tg for polycarbonate and then plot solubility curves."
    mixed_query = (
        "Do a literature search and patent search for solvent-based delamination of HDPE/EVOH food-packaging "
        "laminates, answer the question with RAG, then design an optimal atmospheric-pressure separation sequence "
        "for an HDPE/EVOH mixed waste stream using selective dissolution. Propose up to 1 additional wash step "
        "for phthalate removal, then run a techno-economic analysis on solvent recovery for the best option, and "
        "finally create a chart summarizing both the retrieved findings and the process results."
    )

    sep_payload = _default_separation_payload(["HDPE", "EVOH"])
    contam_payload = _default_contaminant_payload(
        target_polymer="HDPE",
        other_polymers=["EVOH"],
        contaminants=["Phthalates"],
        solvents=["toluene", "xylene"],
    )
    bio_payload = _default_biosteam_payload("HDPE", "toluene")

    cases = [
        WorkflowReplayCase(
            name="research-rag-viz",
            query=research_query,
            outcomes={
                "scholar-researcher": ReplayOutcome("ok", _default_scholar_payload(research_query)),
                "patent-researcher": ReplayOutcome("ok", _default_patent_payload(research_query)),
                "rag-analyst": ReplayOutcome("ok", _default_rag_payload()),
                "visualization-specialist": ReplayOutcome("ok", _default_visualization_payload("comparison_dashboard")),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=(
                    "scholar-researcher",
                    "patent-researcher",
                    "rag-analyst",
                    "visualization-specialist",
                ),
                expected_contracts=(
                    ("patent-researcher", "rag-analyst", "patent_context.v1"),
                    ("rag-analyst", "visualization-specialist", "rag-analyst.to.visualization-specialist.context.v1"),
                    ("scholar-researcher", "rag-analyst", "literature_context.v1"),
                ),
            ),
        ),
        WorkflowReplayCase(
            name="sep-contam-bio",
            query=seq_contam_query,
            outcomes={
                "separation-engineer": ReplayOutcome("ok", sep_payload),
                "contaminant-removal-analyst": ReplayOutcome("ok", contam_payload),
                "biosteam-analyst": ReplayOutcome("ok", bio_payload),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=(
                    "separation-engineer",
                    "contaminant-removal-analyst",
                    "biosteam-analyst",
                ),
                expected_contracts=(
                    ("contaminant-removal-analyst", "biosteam-analyst", "contaminant_biosteam.v1"),
                    ("separation-engineer", "contaminant-removal-analyst", "contaminant_screen.v1"),
                ),
            ),
        ),
        WorkflowReplayCase(
            name="sep-bio-viz",
            query=seq_bio_viz_query,
            outcomes={
                "separation-engineer": ReplayOutcome("ok", _default_separation_payload(["LDPE", "EVOH"])),
                "biosteam-analyst": ReplayOutcome("ok", _default_biosteam_payload("LDPE", "toluene")),
                "visualization-specialist": ReplayOutcome("ok", _default_visualization_payload("biosteam_dashboard")),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=(
                    "separation-engineer",
                    "biosteam-analyst",
                    "visualization-specialist",
                ),
                expected_contracts=(
                    ("biosteam-analyst", "visualization-specialist", "biosteam_plot.v1"),
                    ("separation-engineer", "biosteam-analyst", "sequence_batch.v1"),
                ),
            ),
        ),
        WorkflowReplayCase(
            name="stats-viz",
            query=stats_viz_query,
            outcomes={
                "statistics-ml": ReplayOutcome("ok", _default_statistics_payload()),
                "visualization-specialist": ReplayOutcome("ok", _default_visualization_payload("curve_plot")),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=("statistics-ml", "visualization-specialist"),
                expected_contracts=(("statistics-ml", "visualization-specialist", "analysis_plot.v1"),),
            ),
        ),
        WorkflowReplayCase(
            name="mixed-complex-success",
            query=mixed_query,
            outcomes={
                "separation-engineer": ReplayOutcome("ok", sep_payload),
                "scholar-researcher": ReplayOutcome("ok", _default_scholar_payload(mixed_query)),
                "patent-researcher": ReplayOutcome("ok", _default_patent_payload(mixed_query)),
                "rag-analyst": ReplayOutcome("ok", _default_rag_payload()),
                "contaminant-removal-analyst": ReplayOutcome("ok", contam_payload),
                "biosteam-analyst": ReplayOutcome("ok", bio_payload),
                "visualization-specialist": ReplayOutcome("ok", _default_visualization_payload("comparison_dashboard")),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=(
                    "separation-engineer",
                    "scholar-researcher",
                    "patent-researcher",
                    "rag-analyst",
                    "contaminant-removal-analyst",
                    "biosteam-analyst",
                    "visualization-specialist",
                ),
                expected_contracts=(
                    ("biosteam-analyst", "visualization-specialist", "biosteam_plot.v1"),
                    ("contaminant-removal-analyst", "biosteam-analyst", "contaminant_biosteam.v1"),
                    ("patent-researcher", "rag-analyst", "patent_context.v1"),
                    ("rag-analyst", "visualization-specialist", "rag-analyst.to.visualization-specialist.context.v1"),
                    ("scholar-researcher", "rag-analyst", "literature_context.v1"),
                    ("separation-engineer", "contaminant-removal-analyst", "contaminant_screen.v1"),
                ),
            ),
        ),
        WorkflowReplayCase(
            name="mixed-complex-missing-contaminant",
            query=mixed_query,
            outcomes={
                "separation-engineer": ReplayOutcome("ok", sep_payload),
                "scholar-researcher": ReplayOutcome("ok", _default_scholar_payload(mixed_query)),
                "patent-researcher": ReplayOutcome("ok", _default_patent_payload(mixed_query)),
                "rag-analyst": ReplayOutcome("ok", _default_rag_payload()),
                "contaminant-removal-analyst": ReplayOutcome("missing", raw_text="Only prose, no structured result."),
            },
            expectation=ReplayExpectation(
                status="deadlock",
                completed_subagents=(
                    "separation-engineer",
                    "scholar-researcher",
                    "patent-researcher",
                    "rag-analyst",
                ),
                expected_contracts=(
                    ("patent-researcher", "rag-analyst", "patent_context.v1"),
                    ("scholar-researcher", "rag-analyst", "literature_context.v1"),
                    ("separation-engineer", "contaminant-removal-analyst", "contaminant_screen.v1"),
                ),
            ),
        ),
        WorkflowReplayCase(
            name="sep-bio-retry-success",
            query=(
                "Find an optimal separation sequence for LDPE and EVOH at atmospheric pressure. "
                "Then run a techno-economic analysis on solvent recovery for the best option."
            ),
            outcomes={
                "separation-engineer": (
                    ReplayOutcome("missing", raw_text="Only prose, no structured result."),
                    ReplayOutcome("ok", _default_separation_payload(["LDPE", "EVOH"])),
                ),
                "biosteam-analyst": ReplayOutcome("ok", _default_biosteam_payload("LDPE", "toluene")),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=("separation-engineer", "biosteam-analyst"),
                expected_contracts=(("separation-engineer", "biosteam-analyst", "sequence_batch.v1"),),
            ),
        ),
        WorkflowReplayCase(
            name="research-rag-retry-scholar",
            query=(
                "Do a literature search and patent search for multilayer polymer recycling methods, "
                "answer the question with RAG."
            ),
            outcomes={
                "scholar-researcher": (
                    ReplayOutcome("missing", raw_text="Only prose, no structured result."),
                    ReplayOutcome(
                        "ok",
                        _default_scholar_payload(
                            "Do a literature search and patent search for multilayer polymer recycling methods, answer the question with RAG."
                        ),
                    ),
                ),
                "patent-researcher": ReplayOutcome(
                    "ok",
                    _default_patent_payload(
                        "Do a literature search and patent search for multilayer polymer recycling methods, answer the question with RAG."
                    ),
                ),
                "rag-analyst": ReplayOutcome("ok", _default_rag_payload()),
            },
            expectation=ReplayExpectation(
                status="complete",
                completed_subagents=("patent-researcher", "scholar-researcher", "rag-analyst"),
                expected_contracts=(
                    ("patent-researcher", "rag-analyst", "patent_context.v1"),
                    ("scholar-researcher", "rag-analyst", "literature_context.v1"),
                ),
            ),
        ),
    ]
    return cases + _build_graph_derived_replay_cases()


def _make_task_call(subagent: str, description: str = "") -> dict[str, Any]:
    args: dict[str, Any] = {"subagent_type": subagent}
    if description:
        args["description"] = description
    return {
        "id": f"tc_{subagent}_{uuid4().hex[:10]}",
        "name": "task",
        "args": args,
    }


def _parse_handoff(tool_message: ToolMessage) -> dict[str, Any] | None:
    try:
        payload = json.loads(tool_message.content if isinstance(tool_message.content, str) else str(tool_message.content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    handoff = payload.get("handoff")
    if not isinstance(handoff, dict):
        return None
    producer = handoff.get("producer")
    consumer = handoff.get("consumer")
    if not isinstance(producer, str) or not isinstance(consumer, str):
        return None
    return {
        "producer": producer,
        "consumer": consumer,
        "contract": handoff.get("contract"),
        "handoff_id": handoff.get("handoff_id"),
        "parent_handoff_id": handoff.get("parent_handoff_id"),
        "status": handoff.get("status"),
    }


def _parse_handoff_edge(tool_message: ToolMessage) -> tuple[str, str] | None:
    handoff = _parse_handoff(tool_message)
    if handoff is None:
        return None
    return handoff["producer"], handoff["consumer"]


def _extract_built_handoffs(messages: list) -> list[dict[str, Any]]:
    handoffs: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        handoff = _parse_handoff(message)
        if handoff is not None:
            handoffs.append(handoff)
    return handoffs


def _normalize_outcome_sequence(
    outcome: ReplayOutcome | tuple[ReplayOutcome, ...] | list[ReplayOutcome],
) -> list[ReplayOutcome]:
    if isinstance(outcome, ReplayOutcome):
        return [outcome]
    return list(outcome)


def _next_outcome(
    pending_outcomes: dict[str, list[ReplayOutcome]],
    subagent: str,
) -> ReplayOutcome:
    sequence = pending_outcomes.get(subagent)
    if not sequence:
        raise KeyError(f"No replay outcome configured for {subagent}")
    return sequence.pop(0)


def _append_task_result(
    *,
    messages: list,
    subagent: str,
    outcome: ReplayOutcome,
    events: list[ReplayEvent],
) -> None:
    dispatch = _get_latest_dispatch_for_subagent(messages, subagent)
    if dispatch is None:
        raise RuntimeError(f"Cannot append task result for undispatched subagent {subagent}")

    tool_call_id = dispatch["tool_call_id"]
    task_prompt = dispatch.get("description") or ""
    if outcome.kind == "ok":
        if not isinstance(outcome.payload, dict):
            raise ValueError(f"Replay outcome for {subagent} must include a payload")
        store_agent_result(
            producer=subagent,
            payload=outcome.payload,
            source_tool_call_id=tool_call_id,
            task_prompt=task_prompt,
        )
        content = f"Completed {subagent}.\n\n{_structured_result_block(outcome.payload)}"
        messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        events.append(ReplayEvent("result", subagent, "structured_result_ok"))
        return

    store_agent_failure(
        producer=subagent,
        error_kind="missing_structured_result",
        message="No usable structured result was returned",
        source_tool_call_id=tool_call_id,
        raw_text=outcome.raw_text or "Only prose, no structured result.",
        task_prompt=task_prompt,
    )
    messages.append(
        ToolMessage(
            content=outcome.raw_text or "Only prose, no structured result.",
            tool_call_id=tool_call_id,
        )
    )
    events.append(ReplayEvent("result", subagent, "missing_structured_result"))


def _dispatch_task(
    *,
    messages: list,
    subagent: str,
    description: str,
    events: list[ReplayEvent],
) -> None:
    task_call = _make_task_call(subagent, description)
    messages.append(AIMessage(content="", tool_calls=[task_call]))
    events.append(ReplayEvent("dispatch", subagent, description or ""))


def _unstarted_root_step(messages: list, remaining: list[dict[str, Any]]) -> dict[str, Any] | None:
    started = set(_extract_all_task_subagents(messages))
    for step in remaining:
        if step["depends_on"]:
            continue
        if step["subagent"] in started:
            continue
        return step
    return None


def _extract_built_handoff_edges(messages: list) -> list[tuple[str, str]]:
    return [
        (handoff["producer"], handoff["consumer"])
        for handoff in _extract_built_handoffs(messages)
    ]


def _compute_missing_direct_handoffs(
    *,
    plan: list[dict[str, Any]],
    messages: list,
) -> list[tuple[str, str]]:
    built_edges = set(_extract_built_handoff_edges(messages))
    dispatched_subagents = {dispatch["subagent"] for dispatch in _extract_task_dispatches(messages)}
    missing: list[tuple[str, str]] = []
    for step in plan:
        consumer = step["subagent"]
        if consumer not in dispatched_subagents:
            continue
        for producer in step.get("depends_on", ()):
            edge = (producer, consumer)
            if edge not in built_edges:
                missing.append(edge)
    return missing


def _retryable_step(
    *,
    messages: list,
    remaining: list[dict[str, Any]],
    pending_outcomes: dict[str, list[ReplayOutcome]],
) -> dict[str, Any] | None:
    failed_subagents = {dispatch["subagent"] for dispatch in _extract_failed_subagent_calls(messages)}
    completed_subagents = set(_extract_completed_subagents(messages))
    for step in remaining:
        subagent = step["subagent"]
        if subagent in completed_subagents or subagent not in failed_subagents:
            continue
        if not pending_outcomes.get(subagent):
            continue
        dependencies = step.get("depends_on", ())
        if any(dependency not in completed_subagents for dependency in dependencies):
            continue
        return step
    return None


def _run_replay_loop(
    *,
    case: WorkflowReplayCase,
    messages: list[Any],
    allowed_rules: list[dict[str, Any]],
    pending_outcomes: dict[str, list[ReplayOutcome]],
    events: list[ReplayEvent],
    stop_after_completed: tuple[str, ...] = (),
    stop_after_failed: tuple[str, ...] = (),
    max_iterations: int = 100,
) -> str:
    completed_target = set(stop_after_completed)
    failed_target = set(stop_after_failed)
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        completed_now = set(_extract_completed_subagents(messages))
        failed_now = {dispatch["subagent"] for dispatch in _extract_failed_subagent_calls(messages)}
        if completed_target and completed_target.issubset(completed_now):
            return "paused"
        if failed_target and failed_target.issubset(failed_now):
            return "paused"

        plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
        remaining = _get_active_remaining_steps(messages, plan)
        if not remaining:
            return "complete"

        if not _extract_all_task_subagents(messages):
            initial = _build_initial_route_task_response(messages, allowed_rules)
            if initial is not None:
                ai_msg = initial.result[0]
                messages.append(ai_msg)
                task_call = ai_msg.tool_calls[0]
                subagent = task_call["args"]["subagent_type"]
                description = task_call["args"].get("description", "")
                events.append(ReplayEvent("dispatch", subagent, description))
                outcome = _next_outcome(pending_outcomes, subagent)
                _append_task_result(messages=messages, subagent=subagent, outcome=outcome, events=events)
                continue

        pending = _get_pending_required_handoff(messages, allowed_rules)
        if pending is not None:
            response = _build_pending_handoff_response(pending)
            ai_msg = response.result[0]
            messages.append(ai_msg)
            tool_call = ai_msg.tool_calls[0]
            events.append(
                ReplayEvent(
                    "build_handoff",
                    f"{pending[0]}->{pending[1]}",
                    "pending_required_handoff",
                )
            )
            result_content = build_handoff(**tool_call["args"])
            messages.append(ToolMessage(content=result_content, tool_call_id=tool_call["id"]))
            continue

        ready = _get_ready_downstream_handoff(messages, allowed_rules)
        if ready is not None:
            response = _build_ready_handoff_response(ready)
            ai_msg = response.result[0]
            messages.append(ai_msg)
            task_call = ai_msg.tool_calls[0]
            subagent = task_call["args"]["subagent_type"]
            description = task_call["args"].get("description", "")
            events.append(ReplayEvent("dispatch", subagent, description or "ready_downstream_handoff"))
            outcome = _next_outcome(pending_outcomes, subagent)
            _append_task_result(messages=messages, subagent=subagent, outcome=outcome, events=events)
            continue

        root_step = _unstarted_root_step(messages, remaining)
        if root_step is not None:
            _dispatch_task(
                messages=messages,
                subagent=root_step["subagent"],
                description=case.query,
                events=events,
            )
            outcome = _next_outcome(pending_outcomes, root_step["subagent"])
            _append_task_result(
                messages=messages,
                subagent=root_step["subagent"],
                outcome=outcome,
                events=events,
            )
            continue

        retry_step = _retryable_step(
            messages=messages,
            remaining=remaining,
            pending_outcomes=pending_outcomes,
        )
        if retry_step is not None:
            prior_dispatch = _get_latest_dispatch_for_subagent(messages, retry_step["subagent"])
            retry_description = (
                (prior_dispatch or {}).get("description")
                or retry_step.get("description")
                or case.query
            )
            _dispatch_task(
                messages=messages,
                subagent=retry_step["subagent"],
                description=retry_description,
                events=events,
            )
            outcome = _next_outcome(pending_outcomes, retry_step["subagent"])
            _append_task_result(
                messages=messages,
                subagent=retry_step["subagent"],
                outcome=outcome,
                events=events,
            )
            continue

        events.append(ReplayEvent("deadlock", remaining[0]["subagent"], "no pending handoff or ready root"))
        return "deadlock"

    events.append(ReplayEvent("deadlock", "loop", "iteration limit reached"))
    return "deadlock"


def _finalize_replay_result(
    *,
    case: WorkflowReplayCase,
    messages: list[Any],
    allowed_rules: list[dict[str, Any]],
    status: str,
    events: list[ReplayEvent],
) -> WorkflowReplayResult:
    plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    completed = _extract_completed_subagents(messages)
    failed = [dispatch["subagent"] for dispatch in _extract_failed_subagent_calls(messages)]
    built_handoffs = _extract_built_handoffs(messages)
    stored_multi_source_handoffs = [
        record.to_dict()
        for record in list_handoff_records(
            producer="multi-source",
            status="ok",
        )
    ]
    built_edges = _extract_built_handoff_edges(messages)
    missing_handoffs = _compute_missing_direct_handoffs(plan=plan, messages=messages)
    ordered_plan_snapshot = [
        {
            "subagent": step["subagent"],
            "depends_on": tuple(step.get("depends_on", ())),
            "step_id": step["step_id"],
        }
        for step in plan
    ]
    built_contracts = {
        (handoff["producer"], handoff["consumer"], str(handoff.get("contract")))
        for handoff in built_handoffs
    }
    expected_contracts = set(case.expectation.expected_contracts)

    ok = (
        status == case.expectation.status
        and tuple(completed) == case.expectation.completed_subagents
        and tuple(sorted(set(missing_handoffs))) == tuple(sorted(set(case.expectation.missing_direct_handoffs)))
        and built_contracts == expected_contracts
    )
    reason = "ok" if ok else (
        f"expected status={case.expectation.status}, completed={case.expectation.completed_subagents}, "
        f"missing_handoffs={case.expectation.missing_direct_handoffs}, expected_contracts={tuple(sorted(expected_contracts))}; "
        f"got status={status}, completed={tuple(completed)}, "
        f"missing_handoffs={tuple(sorted(set(missing_handoffs)))}, "
        f"built_contracts={tuple(sorted(built_contracts))}"
    )

    return WorkflowReplayResult(
        name=case.name,
        ok=ok,
        status=status,
        reason=reason,
        allowed_subagents=[rule["subagent"] for rule in allowed_rules],
        ordered_plan=ordered_plan_snapshot,
        completed_subagents=completed,
        failed_subagents=failed,
        built_handoff_edges=built_edges,
        built_handoffs=built_handoffs,
        stored_multi_source_handoffs=stored_multi_source_handoffs,
        missing_direct_handoffs=missing_handoffs,
        events=events,
    )


def replay_workflow_case_until(
    case: WorkflowReplayCase,
    *,
    stop_after_completed: tuple[str, ...] = (),
    stop_after_failed: tuple[str, ...] = (),
) -> WorkflowReplayCheckpoint:
    events: list[ReplayEvent] = []
    temp_dir = tempfile.TemporaryDirectory(prefix="workflow_replay_")
    initialize_handoff_scope(
        run_id=f"workflow-replay-{case.name}",
        thread_id="workflow-replay",
        artifact_root=Path(temp_dir.name),
        user_query=case.query,
    )
    messages: list[Any] = [HumanMessage(content=case.query)]
    middleware = RoutingMiddleware(classifier_model=None)
    allowed_rules = (
        _rule_subset(case.allowed_subagents)
        if case.allowed_subagents
        else middleware._get_allowed_rules(messages)
    )
    pending_outcomes = {
        subagent: _normalize_outcome_sequence(outcome)
        for subagent, outcome in case.outcomes.items()
    }
    try:
        status = _run_replay_loop(
            case=case,
            messages=messages,
            allowed_rules=allowed_rules,
            pending_outcomes=pending_outcomes,
            events=events,
            stop_after_completed=stop_after_completed,
            stop_after_failed=stop_after_failed,
        )
        scope_snapshot = snapshot_handoff_scope_state()
    finally:
        cleanup_handoff_scope()

    temp_dir.cleanup()
    return WorkflowReplayCheckpoint(
        case=case,
        allowed_subagents=[rule["subagent"] for rule in allowed_rules],
        status=status,
        messages=list(messages),
        pending_outcomes={name: list(outcomes) for name, outcomes in pending_outcomes.items()},
        events=list(events),
        scope_snapshot=scope_snapshot,
    )


def resume_workflow_replay(checkpoint: WorkflowReplayCheckpoint) -> WorkflowReplayResult:
    restore_handoff_scope_state(checkpoint.scope_snapshot)
    try:
        allowed_rules = _rule_subset(tuple(checkpoint.allowed_subagents))
        messages = list(checkpoint.messages)
        pending_outcomes = {
            name: list(outcomes)
            for name, outcomes in checkpoint.pending_outcomes.items()
        }
        events = list(checkpoint.events)
        status = checkpoint.status
        if status == "paused":
            status = _run_replay_loop(
                case=checkpoint.case,
                messages=messages,
                allowed_rules=allowed_rules,
                pending_outcomes=pending_outcomes,
                events=events,
            )
        return _finalize_replay_result(
            case=checkpoint.case,
            messages=messages,
            allowed_rules=allowed_rules,
            status=status,
            events=events,
        )
    finally:
        cleanup_handoff_scope()


def replay_workflow_case(case: WorkflowReplayCase) -> WorkflowReplayResult:
    checkpoint = replay_workflow_case_until(case)
    return resume_workflow_replay(checkpoint)


def run_workflow_replay_suite(
    cases: list[WorkflowReplayCase] | None = None,
) -> WorkflowReplaySummary:
    selected_cases = list(cases) if cases is not None else build_workflow_replay_cases()
    results: list[WorkflowReplayResult] = []
    blocked_model_call_attempts: Counter[str] = Counter()
    with block_model_access(blocked_model_call_attempts):
        for case in selected_cases:
            results.append(replay_workflow_case(case))
    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    return WorkflowReplaySummary(
        total=len(results),
        passed=passed,
        failed=failed,
        blocked_model_call_attempts=dict(blocked_model_call_attempts),
        results=results,
    )


def _summary_payload(summary: WorkflowReplaySummary) -> dict[str, Any]:
    return {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "blocked_model_call_attempts": summary.blocked_model_call_attempts,
        "results": [
            {
                **asdict(result),
                "events": [asdict(event) for event in result.events],
            }
            for result in summary.results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic workflow replay harness.")
    parser.add_argument("--json-out", type=Path, help="Optional path for a JSON summary.")
    args = parser.parse_args(argv)

    summary = run_workflow_replay_suite()
    payload = _summary_payload(summary)
    print(
        json.dumps(
            {
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "blocked_model_call_attempts": summary.blocked_model_call_attempts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if summary.failed == 0 and not summary.blocked_model_call_attempts else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
