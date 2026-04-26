"""Typed runtime metadata for lightweight orchestrator decisions.

The CLI uses these small, serializable records to keep simple direct-tool
turns closed over the selected tool and to persist artifact context for later
follow-ups without injecting full transcript history back into the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SPECIALIST_SUBAGENTS: tuple[str, ...] = (
    "separation-engineer",
    "safety-analyst",
    "biosteam-analyst",
    "scholar-agent",
    "patent-agent",
    "rag-agent",
    "visualization-agent",
    "statistics-agent",
    "contaminant-removal-engineer",
    "optimization-engineer",
)


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def new_runtime_id(prefix: str) -> str:
    """Return a compact runtime identifier with a readable prefix."""
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(frozen=True)
class ArtifactFrame:
    """Compact artifact descriptor persisted across CLI turns."""

    artifact_id: str
    type: str
    producer: str
    entities: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    row_order: list[str] = field(default_factory=list)
    display_title: str = ""
    source_artifact_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    ttl: str = "session"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    """Auditable route decision made before executing a simple request."""

    route_id: str
    mode: str
    intent: str
    executor: str = "direct_tool_adapter"
    allowed_tools: list[str] = field(default_factory=list)
    allowed_subagents: list[str] = field(default_factory=list)
    denied_subagents: list[str] = field(default_factory=lambda: list(SPECIALIST_SUBAGENTS))
    model_call_budget: int = 0
    tool_call_budget: int = 1
    handoff_policy: str = "disabled"
    verifier_policy: str = "disabled_for_validated_tool_display"
    failure_policy: str = "fail_closed_with_explanation"
    reason: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunLedger:
    """Minimal execution ledger for route/tool budget auditing."""

    route_id: str
    status: str
    model_calls: int
    tool_calls: int
    tools: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_route_decision(
    *,
    mode: str,
    intent: str,
    allowed_tools: list[str],
    tool_call_budget: int,
    reason: str,
) -> dict[str, Any]:
    """Build a JSON-serializable direct-tool route decision."""
    decision = RouteDecision(
        route_id=new_runtime_id("route"),
        mode=mode,
        intent=intent,
        allowed_tools=allowed_tools,
        tool_call_budget=tool_call_budget,
        reason=reason,
    )
    return decision.to_dict()


def make_artifact(
    *,
    artifact_type: str,
    producer: str,
    entities: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    row_order: list[str] | None = None,
    display_title: str = "",
    source_artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable artifact frame."""
    artifact = ArtifactFrame(
        artifact_id=new_runtime_id("artifact"),
        type=artifact_type,
        producer=producer,
        entities=entities or {},
        data=data or {},
        row_order=row_order or [],
        display_title=display_title,
        source_artifact_ids=source_artifact_ids or [],
    )
    return artifact.to_dict()


def make_run_ledger(
    *,
    route_decision: dict[str, Any],
    tools: list[str],
    status: str = "ok",
) -> dict[str, Any]:
    """Build a JSON-serializable run ledger for a completed route."""
    ledger = RunLedger(
        route_id=str(route_decision.get("route_id", "")),
        status=status,
        model_calls=int(route_decision.get("model_call_budget", 0)),
        tool_calls=len(tools),
        tools=tools,
    )
    return ledger.to_dict()
