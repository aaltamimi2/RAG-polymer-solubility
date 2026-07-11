"""Guarded mid-turn replanning: a typed step failure re-invokes the planner
with the prior plan + outcome; the revision atomically replaces the cached
plan for the rest of the turn. Triggers are narrow (task error, step-budget
exhaustion, infeasible structured result), each outcome revises at most once,
at most two revisions apply per turn, and a degraded planner never revises."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from strap.route_planner import RoutePlanner, clear_active_route_plans, get_active_route_plan
from strap.routing import RoutingMiddleware, _plan_step_statuses, _qualifying_step_failure

QUERY = "Shortlist solvents for PE/EVOH then run the cost-emissions Pareto and report the knee point"

PLAN_A = {
    "mode": "specialists",
    "steps": [
        {"subagent": "separation-engineer", "objective": "shortlist solvents", "depends_on": []},
        {"subagent": "optimization-engineer", "objective": "cost-emissions Pareto", "depends_on": ["separation-engineer"]},
    ],
    "excluded_subagents": [],
    "confidence": "high",
    "rationale": "staged flow",
}
REVISION = {
    "mode": "orchestrator",
    "steps": [],
    "excluded_subagents": [],
    "confidence": "high",
    "rationale": "optimization infeasible; synthesize the honest answer",
}


class RecordingBackend:
    """Returns PLAN_A for initial planning, REVISION for revision requests."""

    def __init__(self, revision_payload=REVISION, fail_revision=False):
        self.calls: list[tuple[str, str | None]] = []
        self.revision_payload = revision_payload
        self.fail_revision = fail_revision

    def __call__(self, query_text: str, *, session_digest: str | None = None):
        self.calls.append((query_text, session_digest))
        if session_digest and "[PLAN REVISION REQUEST]" in session_digest:
            if self.fail_revision:
                raise RuntimeError("revision backend down")
            return dict(self.revision_payload)
        return dict(PLAN_A)

    @property
    def revision_calls(self):
        return [c for c in self.calls if c[1] and "[PLAN REVISION REQUEST]" in c[1]]


def _messages(opt_outcome: str, opt_status: str | None = None) -> list:
    """One turn: sep-engineer completed ok, then opt-engineer returned `opt_outcome`."""
    sep_result = (
        'shortlist ready <STRUCTURED_RESULT>{"agent": "separation-engineer", '
        '"schema_version": "1.0", "polymer_solvent_candidates": {"PE": []}}</STRUCTURED_RESULT>'
    )
    tool_kwargs = {"status": opt_status} if opt_status else {}
    return [
        HumanMessage(QUERY),
        AIMessage("", tool_calls=[{"name": "task", "id": "t_sep",
                                   "args": {"subagent_type": "separation-engineer", "description": "shortlist"}}]),
        ToolMessage(sep_result, tool_call_id="t_sep"),
        AIMessage("", tool_calls=[{"name": "task", "id": "t_opt",
                                   "args": {"subagent_type": "optimization-engineer", "description": "pareto"}}]),
        ToolMessage(opt_outcome, tool_call_id="t_opt", **tool_kwargs),
    ]


INFEASIBLE = (
    'summary <STRUCTURED_RESULT>{"agent": "optimization-engineer", "schema_version": "1.3", '
    '"analysis_type": "infeasible", "failure_reason": "all_shortlisted_sims_failed", '
    '"message": "no workbook rows remain"}</STRUCTURED_RESULT>'
)
BUDGET_FAILURE = (
    "Subagent 'optimization-engineer' was terminated after exhausting its step budget "
    "(120 graph steps) without producing a final answer."
)
SUCCESS = (
    'done <STRUCTURED_RESULT>{"agent": "optimization-engineer", "schema_version": "1.5", '
    '"analysis_type": "pareto_front", "n_points_feasible": 3}</STRUCTURED_RESULT>'
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_active_route_plans()
    yield
    clear_active_route_plans()


def _middleware(backend) -> tuple[RoutingMiddleware, RoutePlanner]:
    planner = RoutePlanner(backend=backend)
    return RoutingMiddleware(planner=planner), planner


class TestTriggers:
    def test_infeasible_structured_result_triggers_revision(self):
        backend = RecordingBackend()
        mw, planner = _middleware(backend)
        messages = _messages(INFEASIBLE)

        mw._maybe_replan(messages)

        assert len(backend.revision_calls) == 1
        active = get_active_route_plan(QUERY)
        assert active.mode == "orchestrator"
        assert active.source == "planner_revision"
        assert any(str(n) == "revised_after:t_opt" for n in active.validation_notes)
        # the revision digest carried the prior plan with per-step statuses
        digest = backend.revision_calls[0][1]
        assert "separation-engineer — shortlist solvents [completed]" in digest
        assert "FAILED: infeasible — no workbook rows remain" in digest

    def test_step_budget_failure_triggers_revision(self):
        backend = RecordingBackend()
        mw, _ = _middleware(backend)
        mw._maybe_replan(_messages(BUDGET_FAILURE))
        assert len(backend.revision_calls) == 1
        assert "step budget exhausted" in backend.revision_calls[0][1]

    def test_task_error_status_triggers_revision(self):
        backend = RecordingBackend()
        mw, _ = _middleware(backend)
        mw._maybe_replan(_messages("tool blew up", opt_status="error"))
        assert len(backend.revision_calls) == 1

    def test_successful_result_does_not_trigger(self):
        backend = RecordingBackend()
        mw, _ = _middleware(backend)
        mw._maybe_replan(_messages(SUCCESS))
        assert backend.revision_calls == []
        assert get_active_route_plan(QUERY).source == "planner"


class TestGuards:
    def test_same_outcome_revises_only_once(self):
        backend = RecordingBackend()
        mw, _ = _middleware(backend)
        messages = _messages(INFEASIBLE)
        mw._maybe_replan(messages)
        mw._maybe_replan(messages)
        mw._maybe_replan(messages)
        assert len(backend.revision_calls) == 1

    def test_revision_replaces_plan_for_all_consumers(self):
        backend = RecordingBackend()
        mw, planner = _middleware(backend)
        messages = _messages(INFEASIBLE)
        mw._maybe_replan(messages)
        # any later plan lookup in the same turn returns the revision from cache
        again = mw._get_plan(QUERY, messages)
        assert again.source == "planner_revision"
        assert again.mode == "orchestrator"

    def test_revision_backend_failure_keeps_original_plan(self):
        backend = RecordingBackend(fail_revision=True)
        mw, _ = _middleware(backend)
        messages = _messages(INFEASIBLE)
        mw._maybe_replan(messages)  # revision raises internally; swallowed
        active = get_active_route_plan(QUERY)
        assert active.source == "planner"
        assert [s.subagent for s in active.steps] == ["separation-engineer", "optimization-engineer"]

    def test_revision_cap_two_per_turn(self):
        # revision payload keeps specialist steps so a further failure could
        # in principle re-trigger; markers must cap it at two
        revision_with_steps = {
            "mode": "specialists",
            "steps": [
                {"subagent": "separation-engineer", "objective": "redo shortlist", "depends_on": []},
                {"subagent": "optimization-engineer", "objective": "retry pareto", "depends_on": ["separation-engineer"]},
            ],
            "excluded_subagents": [],
            "confidence": "high",
            "rationale": "retry with narrower scope",
        }
        backend = RecordingBackend(revision_payload=revision_with_steps)
        mw, _ = _middleware(backend)

        mw._maybe_replan(_messages(INFEASIBLE))
        assert len(backend.revision_calls) == 1

        # second, distinct failed outcome → second revision (markers carried forward)
        messages2 = _messages(INFEASIBLE)
        messages2[3] = AIMessage("", tool_calls=[{"name": "task", "id": "t_opt2",
                                                  "args": {"subagent_type": "optimization-engineer",
                                                           "description": "pareto retry"}}])
        messages2[4] = ToolMessage(INFEASIBLE, tool_call_id="t_opt2")
        mw._maybe_replan(messages2)
        assert len(backend.revision_calls) == 2
        active = get_active_route_plan(QUERY)
        markers = [n for n in active.validation_notes if str(n).startswith("revised_after:")]
        assert len(markers) == 2

        # third outcome → capped
        messages3 = _messages(INFEASIBLE)
        messages3[3] = AIMessage("", tool_calls=[{"name": "task", "id": "t_opt3",
                                                  "args": {"subagent_type": "optimization-engineer",
                                                           "description": "pareto retry 2"}}])
        messages3[4] = ToolMessage(INFEASIBLE, tool_call_id="t_opt3")
        mw._maybe_replan(messages3)
        assert len(backend.revision_calls) == 2

    def test_degraded_planner_never_revises(self):
        planner = RoutePlanner(backend=None)  # keyword fallback only
        mw = RoutingMiddleware(planner=planner)
        # fallback plans with steps are authoritative without a backend, but
        # revise() requires a backend — the plan must survive unchanged
        messages = _messages(INFEASIBLE)
        mw._maybe_replan(messages)
        active = get_active_route_plan(QUERY)
        assert active is None or active.source != "planner_revision"


class TestHelpers:
    def test_qualifying_failure_ignores_non_plan_subagents(self):
        backend = RecordingBackend()
        planner = RoutePlanner(backend=backend)
        plan = planner.plan(QUERY)
        messages = _messages(SUCCESS)
        messages.append(AIMessage("", tool_calls=[{"name": "task", "id": "t_x",
                                                   "args": {"subagent_type": "scholar-researcher",
                                                            "description": "papers"}}]))
        messages.append(ToolMessage("boom", tool_call_id="t_x", status="error"))
        assert _qualifying_step_failure(messages, plan) is None

    def test_statuses_annotate_not_started_steps(self):
        backend = RecordingBackend()
        planner = RoutePlanner(backend=backend)
        plan = planner.plan(QUERY)
        messages = [
            HumanMessage(QUERY),
            AIMessage("", tool_calls=[{"name": "task", "id": "t_sep",
                                       "args": {"subagent_type": "separation-engineer", "description": "s"}}]),
            ToolMessage(BUDGET_FAILURE, tool_call_id="t_sep"),
        ]
        statuses = _plan_step_statuses(messages, plan, "separation-engineer", "step budget exhausted")
        assert statuses["separation-engineer"].startswith("FAILED")
        assert statuses["optimization-engineer"] == "not started"
