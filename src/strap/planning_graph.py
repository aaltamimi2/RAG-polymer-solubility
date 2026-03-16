"""Capability-graph loader for arbitrary multi-step orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .subagent_config import load_subagent_specs

GENERIC_CONTEXT_ARTIFACT = "generic.context.v1"
_VALID_HINTS = {"low", "medium", "high"}


def _unique_strings(values: Any, *, field_name: str, subagent: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"{subagent} planning.{field_name} must be a list of strings")

    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{subagent} planning.{field_name} entries must be non-empty strings")
        normalized = value.strip()
        if normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    return tuple(items)


def _normalize_hint(
    value: Any,
    *,
    field_name: str,
    subagent: str,
    default: str,
) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value.strip() not in _VALID_HINTS:
        allowed = ", ".join(sorted(_VALID_HINTS))
        raise ValueError(
            f"{subagent} planning.{field_name} must be one of: {allowed}"
        )
    return value.strip()


def _normalize_optional_str(
    value: Any,
    *,
    field_name: str,
    subagent: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{subagent} planning.{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class PlanningNode:
    """Normalized planning metadata for one subagent."""

    name: str
    description: str
    goals: tuple[str, ...]
    produces: tuple[str, ...]
    requires: tuple[str, ...]
    consumes: tuple[str, ...]
    prefers: tuple[str, ...]
    parallel_group: str | None
    cost_hint: str
    latency_hint: str


@dataclass(frozen=True)
class PlanningEdge:
    """Directed planning edge between two subagents."""

    producer: str
    consumer: str
    kind: Literal["capability", "generic"]
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class PlanningGraph:
    """Graph of planning nodes plus preferred and fallback edges."""

    nodes: dict[str, PlanningNode]
    capability_edges: tuple[PlanningEdge, ...]
    generic_edges: tuple[PlanningEdge, ...]

    @property
    def edges(self) -> tuple[PlanningEdge, ...]:
        return self.capability_edges + self.generic_edges

    def incoming(self, consumer: str, *, kind: str | None = None) -> tuple[PlanningEdge, ...]:
        edges = self.edges if kind is None else tuple(edge for edge in self.edges if edge.kind == kind)
        return tuple(edge for edge in edges if edge.consumer == consumer)

    def outgoing(self, producer: str, *, kind: str | None = None) -> tuple[PlanningEdge, ...]:
        edges = self.edges if kind is None else tuple(edge for edge in self.edges if edge.kind == kind)
        return tuple(edge for edge in edges if edge.producer == producer)


def _build_node(spec: dict[str, Any]) -> PlanningNode:
    subagent = str(spec.get("name") or "").strip()
    if not subagent:
        raise ValueError("Subagent spec is missing name")

    planning = spec.get("planning") or {}
    if not isinstance(planning, dict):
        raise ValueError(f"{subagent} planning metadata must be a mapping")

    return PlanningNode(
        name=subagent,
        description=str(spec.get("description") or "").strip(),
        goals=_unique_strings(planning.get("goals"), field_name="goals", subagent=subagent),
        produces=_unique_strings(planning.get("produces"), field_name="produces", subagent=subagent),
        requires=_unique_strings(planning.get("requires"), field_name="requires", subagent=subagent),
        consumes=_unique_strings(planning.get("consumes"), field_name="consumes", subagent=subagent),
        prefers=_unique_strings(planning.get("prefers"), field_name="prefers", subagent=subagent),
        parallel_group=_normalize_optional_str(
            planning.get("parallel_group"),
            field_name="parallel_group",
            subagent=subagent,
        ),
        cost_hint=_normalize_hint(
            planning.get("cost_hint"),
            field_name="cost_hint",
            subagent=subagent,
            default="medium",
        ),
        latency_hint=_normalize_hint(
            planning.get("latency_hint"),
            field_name="latency_hint",
            subagent=subagent,
            default="medium",
        ),
    )


def load_planning_nodes(config_path: str | Path | None = None) -> dict[str, PlanningNode]:
    """Load normalized planning metadata for every configured subagent."""
    specs = load_subagent_specs(config_path)
    nodes = [_build_node(spec) for spec in specs]
    return {node.name: node for node in nodes}


def build_planning_graph(config_path: str | Path | None = None) -> PlanningGraph:
    """Construct a capability graph from subagent planning metadata."""
    nodes = load_planning_nodes(config_path)
    capability_edges: list[PlanningEdge] = []
    generic_edges: list[PlanningEdge] = []

    for producer_name, producer in nodes.items():
        produced = set(producer.produces)
        if not produced:
            continue
        for consumer_name, consumer in nodes.items():
            if producer_name == consumer_name:
                continue

            shared = tuple(sorted(produced & (set(consumer.consumes) - {GENERIC_CONTEXT_ARTIFACT})))
            if shared:
                capability_edges.append(
                    PlanningEdge(
                        producer=producer_name,
                        consumer=consumer_name,
                        kind="capability",
                        artifacts=shared,
                    )
                )

            if GENERIC_CONTEXT_ARTIFACT in consumer.consumes:
                generic_edges.append(
                    PlanningEdge(
                        producer=producer_name,
                        consumer=consumer_name,
                        kind="generic",
                        artifacts=(GENERIC_CONTEXT_ARTIFACT,),
                    )
                )

    capability_edges.sort(key=lambda edge: (edge.producer, edge.consumer, edge.artifacts))
    generic_edges.sort(key=lambda edge: (edge.producer, edge.consumer))
    return PlanningGraph(
        nodes=nodes,
        capability_edges=tuple(capability_edges),
        generic_edges=tuple(generic_edges),
    )
