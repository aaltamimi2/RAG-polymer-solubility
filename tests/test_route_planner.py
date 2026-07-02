"""Tests for the planner-first routing core (strap.route_planner)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strap.route_planner import (
    LLMRoutePlannerBackend,
    RoutePlan,
    RoutePlanner,
    RouteStep,
    activate_route_plan,
    active_plan_dependency_map,
    build_planner_system_prompt,
    build_session_digest,
    clear_active_route_plans,
    extract_json_payload,
    fallback_route_plan,
    get_active_route_plan,
    is_direct_route,
    plan_query,
    validate_route_payload,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_active_route_plans()
    yield
    clear_active_route_plans()


def _payload(mode="specialists", steps=(), excluded=(), confidence="high", **extra):
    return {
        "mode": mode,
        "steps": list(steps),
        "excluded_subagents": list(excluded),
        "confidence": confidence,
        "rationale": "test",
        **extra,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_single_specialist_plan(self):
        plan = validate_route_payload(
            "find papers on TEA of recycling",
            _payload(steps=[{"subagent": "scholar-researcher", "objective": "find papers"}]),
        )
        assert plan is not None
        assert plan.mode == "specialists"
        assert plan.subagent_names() == ["scholar-researcher"]
        assert plan.source == "planner"

    def test_unknown_subagents_dropped_with_note(self):
        plan = validate_route_payload(
            "q",
            _payload(steps=[
                {"subagent": "made-up-agent"},
                {"subagent": "separation-engineer"},
            ]),
        )
        assert plan.subagent_names() == ["separation-engineer"]
        assert any("made-up-agent" in note for note in plan.validation_notes)

    def test_all_unknown_specialists_is_unusable(self):
        plan = validate_route_payload(
            "q", _payload(steps=[{"subagent": "nope"}, {"subagent": "also-nope"}])
        )
        assert plan is None

    def test_invalid_mode_without_steps_is_unusable(self):
        assert validate_route_payload("q", _payload(mode="banana")) is None
        assert validate_route_payload("q", {"not": "a plan"}) is None
        assert validate_route_payload("q", "not a dict") is None

    def test_direct_mode_with_steps_coerced_to_specialists(self):
        plan = validate_route_payload(
            "q", _payload(mode="direct", steps=[{"subagent": "separation-engineer"}])
        )
        assert plan.mode == "specialists"
        assert not plan.is_direct

    def test_specialists_mode_without_steps_coerced_to_orchestrator(self):
        plan = validate_route_payload("q", _payload(mode="specialists", steps=[]))
        assert plan.mode == "orchestrator"

    def test_dependency_cycle_reset(self):
        plan = validate_route_payload(
            "q",
            _payload(steps=[
                {"subagent": "separation-engineer", "depends_on": ["biosteam-analyst"]},
                {"subagent": "biosteam-analyst", "depends_on": ["separation-engineer"]},
            ]),
        )
        assert plan is not None
        assert all(step.depends_on == () for step in plan.steps)
        assert any("cycle" in note for note in plan.validation_notes)

    def test_dependencies_order_steps_topologically(self):
        plan = validate_route_payload(
            "q",
            _payload(steps=[
                {"subagent": "biosteam-analyst", "depends_on": ["separation-engineer"]},
                {"subagent": "separation-engineer"},
            ]),
        )
        assert plan.subagent_names() == ["separation-engineer", "biosteam-analyst"]

    def test_exclusions_win_over_steps(self):
        plan = validate_route_payload(
            "optimize profit, no TEA",
            _payload(
                steps=[
                    {"subagent": "optimization-engineer"},
                    {"subagent": "biosteam-analyst"},
                ],
                excluded=["biosteam-analyst"],
            ),
        )
        assert plan.subagent_names() == ["optimization-engineer"]
        assert plan.excluded_subagents == ("biosteam-analyst",)

    def test_missing_dependency_edges_enriched_from_capability_graph(self):
        plan = validate_route_payload(
            "design a separation route then run TEA",
            _payload(steps=[
                {"subagent": "separation-engineer"},
                {"subagent": "biosteam-analyst"},
            ]),
        )
        deps = plan.dependency_map()
        assert "separation-engineer" in deps["biosteam-analyst"]
        assert any("enriched" in note for note in plan.validation_notes)

    def test_explicit_dependency_edges_not_overridden_by_graph(self):
        plan = validate_route_payload(
            "q",
            _payload(steps=[
                {"subagent": "separation-engineer"},
                {"subagent": "safety-analyst", "depends_on": ["separation-engineer"]},
            ]),
        )
        assert plan.dependency_map()["safety-analyst"] == {"separation-engineer"}

    def test_string_steps_accepted(self):
        plan = validate_route_payload(
            "q", _payload(steps=["separation-engineer", "safety-analyst"])
        )
        assert set(plan.subagent_names()) == {"separation-engineer", "safety-analyst"}

    def test_duplicate_steps_deduped(self):
        plan = validate_route_payload(
            "q",
            _payload(steps=[
                {"subagent": "separation-engineer"},
                {"subagent": "separation-engineer"},
            ]),
        )
        assert plan.subagent_names() == ["separation-engineer"]


# ---------------------------------------------------------------------------
# Live-failure regression scenarios (query bank 2026-04-27)
# ---------------------------------------------------------------------------

class TestLiveFailureScenarios:
    """Token-bearing research queries must never be hijacked by numeric routes."""

    def test_tea_worded_literature_query_routes_to_scholar(self):
        payload = _payload(steps=[{"subagent": "scholar-researcher", "objective": "search"}])
        planner = RoutePlanner(backend=lambda q: payload)
        plan = planner.plan(
            "Find recent journal articles on techno-economic analysis of "
            "solvent-based polyolefin recycling"
        )
        assert plan.subagent_names() == ["scholar-researcher"]

        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        middleware = TypedRuntimeMiddleware(route_planner=planner)
        assert middleware._plan_permits_typed_runtime(plan.query) is False

    def test_rag_query_not_stolen_by_fast_path(self):
        payload = _payload(steps=[{"subagent": "rag-analyst", "objective": "retrieve"}])
        planner = RoutePlanner(backend=lambda q: payload)
        query = "What do our indexed documents say about EVOH solvents? Cite retrieved chunks."
        planner.plan(query)

        from strap.direct_fast_path import _plan_allows_fast_path

        assert _plan_allows_fast_path(query, planner) is False

    def test_hsp_worded_rag_query_permits_no_typed_interception(self):
        payload = _payload(steps=[{"subagent": "rag-analyst"}])
        planner = RoutePlanner(backend=lambda q: payload)
        query = "Search the indexed corpus for Hansen solubility parameter tables for EVOH."
        planner.plan(query)

        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        middleware = TypedRuntimeMiddleware(route_planner=planner)
        assert middleware._plan_permits_typed_runtime(query) is False

    def test_covered_numeric_workflow_still_permits_typed_runtime(self):
        payload = _payload(steps=[
            {"subagent": "separation-engineer"},
            {"subagent": "optimization-engineer", "depends_on": ["separation-engineer"]},
        ])
        planner = RoutePlanner(backend=lambda q: payload)
        query = "Optimize the LDPE/PP pathway for profit with a Pareto front."
        planner.plan(query)

        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        middleware = TypedRuntimeMiddleware(route_planner=planner)
        assert middleware._plan_permits_typed_runtime(query) is True

    def test_direct_plan_permits_fast_path(self):
        planner = RoutePlanner(backend=lambda q: _payload(mode="direct"))
        query = "What solvents dissolve LDPE?"
        planner.plan(query)

        from strap.direct_fast_path import _plan_allows_fast_path

        assert _plan_allows_fast_path(query, planner) is True

    def test_safety_only_plan_permits_fast_path_card_rendering(self):
        payload = _payload(steps=[{"subagent": "safety-analyst"}])
        planner = RoutePlanner(backend=lambda q: payload)
        query = "Show the safety card for toluene."
        planner.plan(query)

        from strap.direct_fast_path import _plan_allows_fast_path

        assert _plan_allows_fast_path(query, planner) is True


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_no_backend_uses_keyword_fallback(self):
        planner = RoutePlanner(backend=None)
        plan = planner.plan("Generate the best separation sequence for LDPE/EVOH/PET below 100 C.")
        assert plan.source == "fallback"
        assert "separation-engineer" in plan.subagent_names()

    def test_invalid_payload_falls_back(self):
        planner = RoutePlanner(backend=lambda q: {"subagents": ["separation-engineer"]})
        plan = planner.plan("Design a separation sequence for LDPE and PP below 120C")
        assert plan.source == "fallback"
        assert "separation-engineer" in plan.subagent_names()

    def test_backend_exception_falls_back(self):
        def _boom(query):
            raise RuntimeError("backend down")

        planner = RoutePlanner(backend=_boom)
        plan = planner.plan("Rank solvent selectivity for LDPE over PET below 100 C.")
        assert plan.source == "fallback"

    def test_fallback_direct_lookup_stays_direct(self):
        plan = fallback_route_plan("what are good solvents for dissolving LDPE")
        assert plan.is_direct
        assert plan.to_rules() == []

    def test_fallback_unmatched_query_is_orchestrator(self):
        plan = fallback_route_plan("hello there")
        assert plan.mode == "orchestrator"

    def test_empty_query_is_orchestrator(self):
        plan = RoutePlanner(backend=None).plan("")
        assert plan.mode == "orchestrator"


# ---------------------------------------------------------------------------
# Registry + dependency hook
# ---------------------------------------------------------------------------

class TestActivePlanRegistry:
    def test_plan_activation_and_lookup_normalizes_whitespace(self):
        plan = RoutePlan(query="Design a  route", mode="orchestrator")
        activate_route_plan(plan)
        assert get_active_route_plan("design a route") is plan

    def test_dependency_hook_prefers_active_plan(self):
        from strap.routing_classifier import derive_workflow_dependencies

        query = "custom workflow question"
        plan = RoutePlan(
            query=query,
            mode="specialists",
            steps=(
                RouteStep("separation-engineer"),
                RouteStep("visualization-specialist", depends_on=("separation-engineer",)),
            ),
        )
        activate_route_plan(plan)
        deps = derive_workflow_dependencies(
            query, {"separation-engineer", "visualization-specialist"}
        )
        assert deps["visualization-specialist"] == {"separation-engineer"}
        assert deps["separation-engineer"] == set()

    def test_dependency_hook_ignores_uncovered_name_sets(self):
        query = "another workflow question"
        plan = RoutePlan(
            query=query,
            mode="specialists",
            steps=(RouteStep("separation-engineer"),),
        )
        activate_route_plan(plan)
        assert active_plan_dependency_map(query, {"separation-engineer", "biosteam-analyst"}) is None

    def test_is_direct_route_prefers_plan_over_regex(self):
        query = "Rank solvent selectivity for LDPE over PET below 100 C."
        # Regex says this is not direct; an active direct plan overrides.
        activate_route_plan(RoutePlan(query=query, mode="direct"))
        assert is_direct_route(query) is True

    def test_is_direct_route_falls_back_to_regex_without_plan(self):
        assert is_direct_route("what are good solvents for dissolving LDPE") is True
        assert is_direct_route("Generate the best separation sequence for LDPE/EVOH/PET.") is False


# ---------------------------------------------------------------------------
# RoutePlan projections
# ---------------------------------------------------------------------------

class TestRoutePlanProjection:
    def test_to_rules_carries_rule_fields_and_plan_metadata(self):
        plan = validate_route_payload(
            "q",
            _payload(steps=[
                {"subagent": "separation-engineer", "objective": "design route"},
                {"subagent": "biosteam-analyst", "objective": "cost it",
                 "depends_on": ["separation-engineer"]},
            ]),
        )
        rules = plan.to_rules()
        assert [rule["subagent"] for rule in rules] == ["separation-engineer", "biosteam-analyst"]
        assert all("description" in rule for rule in rules)
        assert rules[1]["depends_on"] == ("separation-engineer",)
        assert rules[1]["objective"] == "cost it"

    def test_explain_is_json_friendly(self):
        import json

        plan = validate_route_payload(
            "q", _payload(steps=[{"subagent": "separation-engineer"}])
        )
        assert json.loads(json.dumps(plan.explain()))["mode"] == "specialists"


# ---------------------------------------------------------------------------
# LLM backend parsing
# ---------------------------------------------------------------------------

class TestLLMBackend:
    def test_extract_json_payload_variants(self):
        assert extract_json_payload('{"mode": "direct"}') == {"mode": "direct"}
        assert extract_json_payload('```json\n{"mode": "direct"}\n```') == {"mode": "direct"}
        assert extract_json_payload('Sure! {"mode": "direct"} hope that helps') == {"mode": "direct"}
        assert extract_json_payload("no json here") is None
        assert extract_json_payload("") is None

    def test_backend_returns_parsed_payload(self):
        response = MagicMock()
        response.content = '{"mode": "direct", "steps": []}'
        model = MagicMock()
        model.invoke.return_value = response
        backend = LLMRoutePlannerBackend(model)
        assert backend("q") == {"mode": "direct", "steps": []}
        assert model.invoke.call_count == 1

    def test_backend_retries_once_on_unparseable_output(self):
        bad = MagicMock()
        bad.content = "gibberish"
        good = MagicMock()
        good.content = '{"mode": "orchestrator", "steps": []}'
        model = MagicMock()
        model.invoke.side_effect = [bad, good]
        backend = LLMRoutePlannerBackend(model)
        assert backend("q") == {"mode": "orchestrator", "steps": []}
        assert model.invoke.call_count == 2

    def test_backend_gives_up_after_retry(self):
        bad = MagicMock()
        bad.content = "gibberish"
        model = MagicMock()
        model.invoke.return_value = bad
        backend = LLMRoutePlannerBackend(model)
        assert backend("q") is None
        assert model.invoke.call_count == 2

    def test_backend_model_error_returns_none(self):
        model = MagicMock()
        model.invoke.side_effect = RuntimeError("api down")
        assert LLMRoutePlannerBackend(model)("q") is None

    def test_prompt_lists_every_configured_specialist(self):
        from strap.routing_classifier import ROUTING_RULES

        prompt = build_planner_system_prompt()
        for rule in ROUTING_RULES:
            assert rule["subagent"] in prompt


# ---------------------------------------------------------------------------
# Planner caching
# ---------------------------------------------------------------------------

class TestPlannerCaching:
    def test_plan_cached_per_query(self):
        calls = []

        def backend(query):
            calls.append(query)
            return _payload(mode="direct")

        planner = RoutePlanner(backend=backend)
        planner.plan("What solvents dissolve LDPE?")
        planner.plan("What solvents dissolve LDPE?")
        planner.plan("  what solvents dissolve  LDPE? ".strip())
        assert len(calls) == 1

    def test_plan_query_convenience(self):
        plan = plan_query("what are good solvents for dissolving LDPE")
        assert plan.is_direct


# ---------------------------------------------------------------------------
# Plan-declared deliverables (typed-runtime intent without keywords)
# ---------------------------------------------------------------------------

class TestDeliverables:
    def test_deliverables_validated_deduped_and_capped(self):
        plan = validate_route_payload(
            "q",
            _payload(
                steps=[{"subagent": "biosteam-analyst"}],
                deliverables=["biosteam_tea_lca_result", "biosteam_tea_lca_result", "", 42],
            ),
        )
        assert plan.deliverables == ("biosteam_tea_lca_result", "42")

    def test_middleware_filters_deliverables_against_artifact_catalog(self):
        payload = _payload(
            steps=[{"subagent": "biosteam-analyst"}],
            deliverables=["biosteam_tea_lca_result", "not_a_real_artifact"],
        )
        planner = RoutePlanner(backend=lambda q: payload)
        query = "What would it cost per kg to recover PE with toluene?"
        planner.plan(query)

        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        middleware = TypedRuntimeMiddleware(route_planner=planner)
        context = middleware._plan_deliverable_context(query)
        assert context == {"plan_requested_artifact_types": ["biosteam_tea_lca_result"]}

    def test_fallback_plans_provide_no_deliverable_context(self):
        planner = RoutePlanner(backend=None)
        query = "Run a TEA for PE with toluene"
        planner.plan(query)

        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        middleware = TypedRuntimeMiddleware(route_planner=planner)
        assert middleware._plan_deliverable_context(query) is None

    def test_plan_deliverables_drive_compile_without_keywords(self):
        from strap.planning.compiler import compile_request

        query = ("What would it cost per kilogram to recover PE from packaging "
                 "waste using toluene at 8000 tonnes per year under case C1?")
        keyword_only = compile_request(query)
        assert keyword_only.status == "unsupported"

        plan_driven = compile_request(
            query, context={"plan_requested_artifact_types": ["biosteam_tea_lca_result"]}
        )
        assert plan_driven.status == "compiled"
        assert plan_driven.plan.intent_family == "biosteam_tea_lca"

    def test_guards_biosteam_intent_prefers_plan(self):
        from langchain_core.messages import HumanMessage

        from strap.route_planner import RoutePlan, RouteStep, activate_route_plan
        from strap.routing_guards import _query_has_explicit_biosteam_intent

        query = "Estimate what recovering PE would run us financially."
        # Regex finds no TEA intent in this wording...
        assert _query_has_explicit_biosteam_intent([HumanMessage(content=query)]) is False
        # ...but a planner decision including biosteam-analyst is authoritative.
        activate_route_plan(RoutePlan(
            query=query, mode="specialists",
            steps=(RouteStep("biosteam-analyst"),),
        ))
        assert _query_has_explicit_biosteam_intent([HumanMessage(content=query)]) is True


# ---------------------------------------------------------------------------
# Degraded-planner semantics: keyword fallback may inform, never execute
# ---------------------------------------------------------------------------

class TestDegradedPlanner:
    def test_fallback_plans_not_cached_so_planner_self_heals(self):
        calls = {"n": 0}

        def flaky_backend(query):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient outage")
            return _payload(mode="direct")

        planner = RoutePlanner(backend=flaky_backend)
        query = "What solvents dissolve LDPE?"
        first = planner.plan(query)
        assert first.source == "fallback"
        second = planner.plan(query)
        assert second.source == "planner"
        assert second.is_direct
        # planner-sourced plan is now cached
        third = planner.plan(query)
        assert third is second
        assert calls["n"] == 2

    def test_degraded_planner_refuses_fast_path_execution(self):
        def dead_backend(query):
            raise RuntimeError("outage")

        planner = RoutePlanner(backend=dead_backend)
        query = "what are good solvents for dissolving LDPE"
        # fallback classifies this direct — but with a configured (degraded)
        # backend, keyword intent must not trigger deterministic execution.
        assert planner.plan(query).is_direct

        from strap.direct_fast_path import _plan_allows_fast_path

        assert _plan_allows_fast_path(query, planner) is False

    def test_no_backend_deployment_keeps_legacy_fast_path(self):
        planner = RoutePlanner(backend=None)
        query = "what are good solvents for dissolving LDPE"
        planner.plan(query)

        from strap.direct_fast_path import _plan_allows_fast_path

        assert _plan_allows_fast_path(query, planner) is True

    def test_degraded_planner_refuses_typed_interception(self):
        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        def dead_backend(query):
            raise RuntimeError("outage")

        degraded = TypedRuntimeMiddleware(route_planner=RoutePlanner(backend=dead_backend))
        assert degraded._plan_permits_typed_runtime("Run BioSTEAM TEA for PE with toluene") is False

        keyword_mode = TypedRuntimeMiddleware(route_planner=RoutePlanner(backend=None))
        assert keyword_mode._plan_permits_typed_runtime("Run BioSTEAM TEA for PE with toluene") is True

    def test_degraded_planner_makes_task_guards_advisory(self):
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage

        from strap.routing import RoutingMiddleware

        def dead_backend(query):
            raise RuntimeError("outage")

        middleware = RoutingMiddleware(planner=RoutePlanner(backend=dead_backend))
        query = "what are good solvents for dissolving LDPE"
        request = MagicMock()
        request.tool_call = {"id": "t1", "name": "task",
                             "args": {"subagent_type": "separation-engineer", "description": "x"}}
        request.state = {"messages": [HumanMessage(content=query)]}
        result = middleware.wrap_tool_call(request, handler=lambda r: "ALLOWED")
        assert result == "ALLOWED"

    def test_no_backend_task_guard_stays_hard(self):
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage
        from langchain_core.messages import ToolMessage as _TM

        from strap.routing import RoutingMiddleware

        middleware = RoutingMiddleware(classifier_model=None)
        query = "what are good solvents for dissolving LDPE"
        request = MagicMock()
        request.tool_call = {"id": "t1", "name": "task",
                             "args": {"subagent_type": "separation-engineer", "description": "x"}}
        request.state = {"messages": [HumanMessage(content=query)]}
        result = middleware.wrap_tool_call(request, handler=lambda r: "ALLOWED")
        assert isinstance(result, _TM) and result.status == "error"

    def test_typed_failure_defers_to_planned_specialists(self, monkeypatch):
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage

        import strap.planning.typed_runtime_integration as tri
        from strap.planning.config import (
            DEFAULT_SELECTED_ENFORCEMENT_ARTIFACTS,
            PlannerConfig,
        )

        payload = _payload(steps=[{"subagent": "biosteam-analyst"}],
                           deliverables=["biosteam_tea_lca_result"])
        planner = RoutePlanner(backend=lambda q: payload)
        middleware = tri.TypedRuntimeMiddleware(
            route_planner=planner,
            config=PlannerConfig(
                mode="enforce_selected",
                selected_enforcement_artifacts=set(DEFAULT_SELECTED_ENFORCEMENT_ARTIFACTS),
            ),
        )
        failure = MagicMock()
        failure.status = "typed_failure"
        failure.reason = "compile failed for selected target"
        monkeypatch.setattr(tri, "maybe_run_typed_runtime", lambda *a, **k: failure)

        request = MagicMock()
        request.messages = [HumanMessage(content="Run BioSTEAM TEA for PE with toluene under case C1")]
        response = middleware.wrap_model_call(request, handler=lambda r: "SPECIALIST_PATH")
        assert response == "SPECIALIST_PATH"

    def test_plan_markers_derived_from_step_graph(self):
        from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware

        payload = _payload(
            steps=[
                {"subagent": "separation-engineer"},
                {"subagent": "optimization-engineer", "depends_on": ["separation-engineer"]},
            ],
            deliverables=["optimization_pareto_front"],
        )
        planner = RoutePlanner(backend=lambda q: payload)
        query = "Shortlist solvents then optimize the pathway for profit."
        planner.plan(query)

        middleware = TypedRuntimeMiddleware(route_planner=planner)
        context = middleware._plan_deliverable_context(query)
        assert context["plan_requested_artifact_types"] == ["optimization_pareto_front"]
        assert set(context["plan_workflow_markers"]) == {"separation", "optimization", "handoff"}


# ---------------------------------------------------------------------------
# Session-aware planning: digest construction, cache identity, threading
# ---------------------------------------------------------------------------

def _multi_turn_messages():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    return [
        HumanMessage(content="Generate a separation state map for LDPE/EVOH/PET under 100 C."),
        AIMessage(content="", tool_calls=[{
            "id": "t1", "name": "task",
            "args": {"subagent_type": "separation-engineer", "description": "map"},
        }]),
        ToolMessage(
            content=(
                '<STRUCTURED_RESULT>{"agent":"separation-engineer","schema_version":"1.0",'
                '"best_sequence":["LDPE","EVOH"],"steps":[{"step":1}],"polymers":["LDPE","EVOH","PET"]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="t1",
        ),
        AIMessage(content="Here is the state map.",
                  additional_kwargs={"strap_origin": "typed_runtime",
                                     "strap_typed_runtime_status": "executed",
                                     "strap_workflow_id": "separation_visualization"}),
        HumanMessage(content="From that state map, which sequence maximizes efficiency?"),
    ]


class TestSessionDigest:
    def test_no_history_yields_none(self):
        from langchain_core.messages import HumanMessage

        assert build_session_digest(None) is None
        assert build_session_digest([]) is None
        assert build_session_digest([HumanMessage(content="first question")]) is None

    def test_digest_summarizes_prior_turns_results_and_runtime(self):
        digest = build_session_digest(_multi_turn_messages())
        assert digest is not None
        assert digest.startswith("[SESSION CONTEXT]")
        assert '"Generate a separation state map' in digest
        assert "- separation-engineer: completed (has: best_sequence, steps, polymers)" in digest
        assert "- typed runtime executed: separation_visualization" in digest
        # current turn's question is NOT part of the digest
        assert "maximizes efficiency" not in digest

    def test_failed_specialists_reported(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        messages = [
            HumanMessage(content="Run the separation."),
            AIMessage(content="", tool_calls=[{
                "id": "t1", "name": "task",
                "args": {"subagent_type": "separation-engineer", "description": "x"},
            }]),
            ToolMessage(content="Tool error: solver crashed", tool_call_id="t1", status="error"),
            HumanMessage(content="try again"),
        ]
        digest = build_session_digest(messages)
        assert "- separation-engineer: FAILED" in digest

    def test_digest_is_deterministic_and_bounded(self):
        messages = _multi_turn_messages()
        assert build_session_digest(messages) == build_session_digest(messages)
        assert len(build_session_digest(messages)) <= 1400


class TestSessionAwarePlanning:
    def test_cache_distinguishes_session_digests(self):
        calls = []

        def backend(query, session_digest=None):
            calls.append(session_digest)
            return _payload(mode="orchestrator")

        planner = RoutePlanner(backend=backend)
        query = "plot the results"
        planner.plan(query)
        planner.plan(query, session_digest="[SESSION CONTEXT]\n- separation-engineer: completed")
        planner.plan(query, session_digest="[SESSION CONTEXT]\n- separation-engineer: completed")
        assert len(calls) == 2  # third call hit the digest-keyed cache
        assert calls[0] is None and calls[1] is not None

    def test_backend_without_digest_parameter_still_works(self):
        planner = RoutePlanner(backend=lambda q: _payload(mode="direct"))
        plan = planner.plan("What solvents dissolve LDPE?", session_digest="[SESSION CONTEXT]\nx")
        assert plan.is_direct

    def test_routing_middleware_threads_digest_to_backend(self):
        from strap.routing import RoutingMiddleware

        seen = {}

        def backend(query, session_digest=None):
            seen["digest"] = session_digest
            return _payload(mode="orchestrator")

        middleware = RoutingMiddleware(planner=RoutePlanner(backend=backend))
        middleware._get_allowed_rules(_multi_turn_messages())
        assert seen["digest"] is not None
        assert "separation-engineer: completed" in seen["digest"]

    def test_llm_backend_prepends_digest_to_current_request(self):
        response = MagicMock()
        response.content = '{"mode": "orchestrator", "steps": []}'
        model = MagicMock()
        model.invoke.return_value = response
        backend = LLMRoutePlannerBackend(model)
        backend("which was best?", session_digest="[SESSION CONTEXT]\nAlready produced this session:\n- separation-engineer: completed")

        human = model.invoke.call_args[0][0][1]
        assert human.content.startswith("[SESSION CONTEXT]")
        assert "[CURRENT REQUEST]\nwhich was best?" in human.content

    def test_followup_orchestrator_plan_yields_no_specialists(self):
        payload = _payload(mode="orchestrator")
        planner = RoutePlanner(backend=lambda q, session_digest=None: payload)
        from strap.routing import RoutingMiddleware

        middleware = RoutingMiddleware(planner=planner)
        rules = middleware._get_allowed_rules(_multi_turn_messages())
        assert rules == []
