"""Offline replay of recorded live planner payloads — no API calls.

``tests/fixtures/route_planner_goldens.json`` holds raw payloads captured from
the production planner model by ``architecture/record_route_planner_goldens.py``.
These tests replay every payload through the CURRENT validation/gating code, so
routing regressions in validate_route_payload, plan projection, or the fast-path
and typed-runtime gates surface without spending a single model call.

If the planner prompt or model changes, re-record the goldens first:

    python architecture/record_route_planner_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strap.route_planner import (
    RESEARCH_SUBAGENTS,
    RoutePlanner,
    validate_route_payload,
)

GOLDENS_PATH = Path(__file__).parent / "fixtures" / "route_planner_goldens.json"

KNOWN_SUBAGENTS = [
    "separation-engineer", "safety-analyst", "biosteam-analyst",
    "scholar-researcher", "patent-researcher", "rag-analyst",
    "visualization-specialist", "statistics-ml",
    "contaminant-removal-analyst", "optimization-engineer",
]


def _load_entries() -> list[dict]:
    if not GOLDENS_PATH.exists():
        pytest.fail(
            f"Missing goldens fixture {GOLDENS_PATH}; "
            "run `python architecture/record_route_planner_goldens.py` to record it."
        )
    data = json.loads(GOLDENS_PATH.read_text())
    return data["entries"]


ENTRIES = _load_entries()
BANK_ENTRIES = [e for e in ENTRIES if e["source"].startswith("query_bank")]
CASE_STUDY_ENTRIES = [e for e in ENTRIES if e["source"] == "case_study"]


def _replay(entry: dict):
    plan = validate_route_payload(entry["query"], entry["payload"])
    assert plan is not None, f"golden payload for {entry['id']} no longer validates"
    return plan


def _expected_route(label: str) -> tuple[list[str], bool]:
    found = sorted(
        (label.find(name), name) for name in KNOWN_SUBAGENTS if name in label
    )
    return [name for _, name in found], "direct fast path" in label.lower()


class _ReplayPlanner(RoutePlanner):
    """Planner that serves one pre-validated plan without any backend."""

    def __init__(self, plan):
        super().__init__(backend=lambda q: None)
        self._plan = plan

    def plan(self, query_text: str, *, session_digest: str | None = None):
        return self._plan


@pytest.mark.parametrize("entry", BANK_ENTRIES, ids=lambda e: e["id"])
def test_bank_payload_routes_to_expected_specialists(entry):
    plan = _replay(entry)
    expected_names, expected_direct = _expected_route(entry["expected_label"])

    if expected_direct:
        assert plan.is_direct or set(plan.subagent_names()) <= {"safety-analyst"}, (
            f"{entry['id']}: expected direct-capable route, got {plan.mode} "
            f"{plan.subagent_names()}"
        )
        return
    if not expected_names:
        return

    planned = plan.subagent_names()
    primary = expected_names[0]
    assert primary in planned, (
        f"{entry['id']}: expected primary {primary!r} in plan, got {planned} "
        f"for query {entry['query'][:90]!r}"
    )
    chain = [name for name in expected_names if name in planned]
    assert chain == [name for name in planned if name in chain], (
        f"{entry['id']}: expected relative order {chain}, plan order {planned}"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["id"])
def test_research_plans_refuse_interception(entry):
    """Plans routing to research/RAG must gate out typed runtime and fast path."""
    plan = _replay(entry)
    if not (set(plan.subagent_names()) & RESEARCH_SUBAGENTS):
        pytest.skip("not a research-routed plan")

    from strap.direct_fast_path import _plan_allows_fast_path
    from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

    planner = _ReplayPlanner(plan)
    middleware = TypedRuntimeMiddleware(route_planner=planner)
    assert middleware._plan_permits_typed_runtime(entry["query"]) is False
    assert _plan_allows_fast_path(entry["query"], planner) is False


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["id"])
def test_direct_plans_permit_fast_path(entry):
    plan = _replay(entry)
    if not plan.is_direct:
        pytest.skip("not a direct plan")

    from strap.direct_fast_path import _plan_allows_fast_path

    assert _plan_allows_fast_path(entry["query"], _ReplayPlanner(plan)) is True


@pytest.mark.parametrize("entry", CASE_STUDY_ENTRIES, ids=lambda e: e["id"])
def test_case_study_route_snapshots(entry):
    """Re-validate case-study payloads and pin the resulting routes.

    A diff here means validation/projection behavior changed for a payload
    that previously produced the recorded route — inspect, then re-record if
    the change is intentional.
    """
    plan = _replay(entry)
    recorded = entry["validated"]
    assert recorded is not None
    assert plan.mode == recorded["mode"]
    assert plan.subagent_names() == [step["subagent"] for step in recorded["steps"]]
    assert {name: sorted(deps) for name, deps in plan.dependency_map().items()} == {
        step["subagent"]: sorted(step["depends_on"]) for step in recorded["steps"]
    }


FOLLOWUP_ENTRIES = [e for e in ENTRIES if e["source"] == "session_followup"]


@pytest.mark.parametrize("entry", FOLLOWUP_ENTRIES, ids=lambda e: e["id"])
def test_session_followup_payload_routes_as_expected(entry):
    """Replay recorded multi-turn planner payloads (real model output given a
    session digest) through current validation and pin the follow-up routing:
    answer-from-results stays with the orchestrator; new stages dispatch only
    the new specialist; explicit redo/retry re-dispatches."""
    plan = _replay(entry)
    assert entry["session_digest"], "follow-up golden must carry a session digest"

    if entry.get("expected_mode"):
        assert plan.mode == entry["expected_mode"], (
            f"{entry['id']}: mode={plan.mode} expected {entry['expected_mode']} "
            f"(steps={plan.subagent_names()})"
        )
    planned = plan.subagent_names()
    for name in entry.get("expected_specialists") or ():
        assert name in planned, f"{entry['id']}: expected {name} in {planned}"
    if entry.get("expected_specialists_any"):
        assert set(entry["expected_specialists_any"]) & set(planned), (
            f"{entry['id']}: none of {entry['expected_specialists_any']} in {planned}"
        )
    for name in entry.get("forbidden_specialists") or ():
        assert name not in planned, (
            f"{entry['id']}: {name} must not be re-planned (already completed); got {planned}"
        )
