"""Tests for routing progress with repeated subagent delegations."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents.middleware.types import ToolCallRequest


def _task_call(tool_call_id: str, subagent: str, description: str | None = None) -> dict:
    args = {"subagent_type": subagent}
    if description is not None:
        args["description"] = description
    return {
        "id": tool_call_id,
        "name": "task",
        "args": args,
    }


def _fs_call(tool_call_id: str, tool_name: str, **args) -> dict:
    return {
        "id": tool_call_id,
        "name": tool_name,
        "args": args,
    }


def _structured_result_content(agent: str) -> str:
    return (
        "<STRUCTURED_RESULT>"
        f'{{"agent":"{agent}","schema_version":"1.0","no_data":true}}'
        "</STRUCTURED_RESULT>"
    )


def _classifier_for(*subagents: str):
    response = MagicMock()
    response.content = (
        '{"subagents": ['
        + ", ".join(f'"{name}"' for name in subagents)
        + '], "confidence": "HIGH"}'
    )
    model = MagicMock()
    model.invoke.return_value = response
    return model


def test_classify_query_applies_runtime_normalization_for_hsp_process_design():
    from strap.routing_classifier import classify_query

    hint = classify_query(
        [
            HumanMessage(
                content=(
                    "Use Hansen solubility parameters to plan a selective separation sequence "
                    "for PE and EVOH."
                )
            )
        ]
    )

    assert hint is not None
    assert "separation-engineer" in hint
    assert "statistics-ml" not in hint


def test_classify_query_keeps_temperature_range_solubility_on_core_tools():
    from strap.routing_classifier import classify_query, explain_routing_decision

    query = "What is the solubility of LDPE in dodecane from room temperature up to 140C?"
    hint = classify_query([HumanMessage(content=query)])
    decision = explain_routing_decision(query)

    assert hint is None
    assert decision["direct_answer"]["is_direct"] is True
    assert decision["planned"] == []


def test_simple_solvent_lookup_stays_on_orchestrator_fast_path():
    from strap.routing_classifier import classify_query, explain_routing_decision, build_direct_answer_hint
    from strap.routing import RoutingMiddleware

    query = "i have an LDPE/EVOH/PET feedstock. what are good solvents for dissolving LDPE"

    hint = classify_query([HumanMessage(content=query)])
    direct_hint = build_direct_answer_hint(query)
    decision = explain_routing_decision(query)
    middleware = RoutingMiddleware(classifier_model=None)
    allowed = middleware._get_allowed_rules([HumanMessage(content=query)])

    assert hint is None
    assert direct_hint is not None
    assert "Do not delegate" in direct_hint
    assert "Do not invent temperature" in direct_hint
    assert decision["direct_answer"]["is_direct"] is True
    assert decision["planned"] == []
    assert allowed == []


def test_simple_solvent_lookup_overrides_overeager_llm_router():
    from strap.routing import RoutingMiddleware

    query = "what are good solvents for dissolving LDPE"
    model = _classifier_for("statistics-ml", "separation-engineer")
    middleware = RoutingMiddleware(classifier_model=model)

    allowed = middleware._get_allowed_rules([HumanMessage(content=query)])

    assert allowed == []
    model.invoke.assert_not_called()


def test_routine_multilayer_solvent_candidate_query_stays_out_of_statistics_ml():
    from strap.routing import RoutingMiddleware
    from strap.routing_classifier import classify_query, explain_routing_decision

    query = (
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
        "identify solvents that are promising for dissolving any one of the components "
        "below 100 deg C."
    )
    middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "separation-engineer"))

    hint = classify_query([HumanMessage(content=query)])
    decision = explain_routing_decision(query)
    allowed = middleware._get_allowed_rules([HumanMessage(content=query)])

    assert hint is None
    assert decision["direct_answer"]["is_direct"] is True
    assert allowed == []
    middleware._classifier_model.invoke.assert_not_called()


def test_routine_multilayer_solvent_candidate_temperature_spellings_stay_direct():
    from strap.routing import RoutingMiddleware
    from strap.routing_classifier import explain_routing_decision

    variants = (
        "below 100 C",
        "below 100C",
        "below 100 °C",
        "under 100 C",
        "up to 100 C",
        "below 212 F",
        "below 212 fahrenehit",
        "below 212 degrees Fahrenheit",
        "under 373.15 K",
        "up to 373.15 kelvn",
        "up to 373.15 degrees Kelvin",
    )
    for variant in variants:
        query = (
            "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
            "identify solvents that are promising for dissolving any one of the components "
            f"{variant}."
        )
        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "separation-engineer"))

        allowed = middleware._get_allowed_rules([HumanMessage(content=query)])
        decision = explain_routing_decision(query)

        assert decision["direct_answer"]["is_direct"] is True, variant
        assert allowed == []
        middleware._classifier_model.invoke.assert_not_called()


def test_router_blocks_statistics_ml_task_for_routine_solvent_candidate_query():
    from strap.routing import RoutingMiddleware

    query = (
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
        "identify solvents that are promising for dissolving any one component below 100 deg C."
    )
    middleware = RoutingMiddleware(classifier_model=None)
    request = ToolCallRequest(
        tool_call=_task_call("stats1", "statistics-ml"),
        tool=None,
        state={"messages": [HumanMessage(content=query)]},
        runtime=MagicMock(),
    )
    handler = MagicMock()

    result = middleware.wrap_tool_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "reserved for explicit HSP/RED" in result.content


def test_router_blocks_any_subagent_task_for_direct_solvent_candidate_query():
    from strap.routing import RoutingMiddleware

    query = (
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
        "identify solvents that are promising for dissolving any one component below 100 C."
    )
    middleware = RoutingMiddleware(classifier_model=None)
    request = ToolCallRequest(
        tool_call=_task_call("sep1", "separation-engineer"),
        tool=None,
        state={"messages": [HumanMessage(content=query)]},
        runtime=MagicMock(),
    )
    handler = MagicMock()

    result = middleware.wrap_tool_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "direct core-tool lookup" in result.content


def test_direct_solvent_lookup_injects_fast_path_system_hint():
    from strap.routing import RoutingMiddleware

    class _Request:
        def __init__(self, messages, system_message):
            self.messages = messages
            self.system_message = system_message

        def override(self, *, system_message):
            return _Request(self.messages, system_message)

    request = _Request(
        [HumanMessage(content="what are good solvents for dissolving LDPE")],
        SystemMessage(content="base system"),
    )
    middleware = RoutingMiddleware(classifier_model=None)

    patched = middleware._inject_hint(request)

    assert patched.system_message.content != "base system"
    system_text = str(patched.system_message.content)
    assert "DIRECT_ANSWER" in system_text
    assert "do not run separation/selectivity ranking" in system_text


def test_direct_solvent_lookup_blocks_temperature_range_tool():
    from strap.routing import RoutingMiddleware

    middleware = RoutingMiddleware(classifier_model=None)
    request = ToolCallRequest(
        tool_call={
            "id": "range1",
            "name": "predict_solubility_range",
            "args": {"polymer": "LDPE", "solvent": "toluene"},
        },
        tool=None,
        state={"messages": [HumanMessage(content="what are good solvents for dissolving LDPE")]},
        runtime=MagicMock(),
    )
    handler = MagicMock()

    result = middleware.wrap_tool_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "range1"
    assert result.status == "error"
    assert "direct solvent lookup" in result.content


def test_direct_solubility_lookup_allows_temperature_range_tool():
    from strap.routing import RoutingMiddleware

    middleware = RoutingMiddleware(classifier_model=None)
    request = ToolCallRequest(
        tool_call={
            "id": "range_ok",
            "name": "predict_solubility_range",
            "args": {"polymer_name": "LDPE", "solvent_name": "dodecane", "t_end_c": 80},
        },
        tool=None,
        state={
            "messages": [
                HumanMessage(
                    content="what is the solubility of LDPE in each of these solvents up to 80C"
                )
            ]
        },
        runtime=MagicMock(),
    )
    handler = MagicMock(return_value="ok")

    result = middleware.wrap_tool_call(request, handler)

    handler.assert_called_once()
    assert result == "ok"


def test_followup_solubility_lookup_with_session_context_stays_direct():
    from strap.routing_classifier import build_direct_answer_hint, explain_routing_decision

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: polymers=LDPE, EVOH, PET\n"
        "- Process: solvent_candidates=Cyclohexane, Hexane, n-Heptane, Dodecane\n\n"
        "User request:\n"
        "what is the solubility of LDPE in each of these solvents up to 80C"
    )

    decision = explain_routing_decision(query)
    hint = build_direct_answer_hint(query)

    assert decision["direct_answer"]["is_direct"] is True
    assert decision["planned"] == []
    assert hint is not None
    assert "predict_solubility_range" in hint


def test_followup_solubility_plot_with_session_context_stays_direct():
    from strap.routing_classifier import build_direct_answer_hint, explain_routing_decision
    from strap.routing import RoutingMiddleware

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last solubility lookup: polymer=EVOH; solvents=N,N-Dimethylformamide; temperature_range=25-80 C\n\n"
        "User request:\n"
        "plot it"
    )

    decision = explain_routing_decision(query)
    hint = build_direct_answer_hint(query)
    allowed = RoutingMiddleware(classifier_model=None)._get_allowed_rules([HumanMessage(content=query)])

    assert decision["direct_answer"]["is_direct"] is True
    assert decision["planned"] == []
    assert allowed == []
    assert hint is not None
    assert "plot_solubility_vs_temperature" in hint
    assert "Do not delegate" in hint


def test_plot_range_correction_with_session_context_stays_direct():
    from strap.routing_classifier import build_direct_answer_hint, explain_routing_decision
    from strap.routing import RoutingMiddleware

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: polymers=LDPE, EVOH, PET\n"
        "- Last solubility lookup: polymer=EVOH; solvents=N,N-Dimethylformamide; temperature_range=25-153 C\n\n"
        "User request:\n"
        "no just plot from 25C to 90C"
    )

    decision = explain_routing_decision(query)
    hint = build_direct_answer_hint(query)
    allowed = RoutingMiddleware(classifier_model=None)._get_allowed_rules([HumanMessage(content=query)])

    assert decision["direct_answer"]["is_direct"] is True
    assert decision["planned"] == []
    assert allowed == []
    assert hint is not None
    assert "plot_solubility_vs_temperature" in hint
    assert "current user request override the stored range" in hint


def test_dp_state_map_request_routes_to_visualization_not_solubility_direct(tmp_path):
    from strap.routing_classifier import build_direct_answer_hint, explain_routing_decision
    from strap.routing import RoutingMiddleware

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymers=LDPE, EVOH, PET; "
        "solvents=dodecane, 1,2-dimethylbenzene, toluene; temperature_range=25-100 C; "
        f"path={tmp_path / 'prior.png'}; output_dir={tmp_path}\n\n"
        "User request:\n"
        f'Generate a dynamic-programming separation state map for LDPE/EVOH/PET at 100C and save it to "{tmp_path}". '
        "Only include separation/selectivity visuals, not cost, greenness, or optimization plots."
    )

    decision = explain_routing_decision(query)
    hint = build_direct_answer_hint(query)
    allowed = RoutingMiddleware(classifier_model=None)._get_allowed_rules([HumanMessage(content=query)])

    assert decision["direct_answer"]["is_direct"] is False
    assert hint is None
    assert [rule["subagent"] for rule in allowed] == ["visualization-specialist"]


def test_routing_after_model_skips_classifier_after_direct_fast_path():
    from strap.routing import RoutingMiddleware

    classifier = MagicMock()
    middleware = RoutingMiddleware(classifier_model=classifier)
    state = {
        "messages": [
            HumanMessage(content="plot the solubility of each of these solvents in EVOH"),
            AIMessage(
                content="Solubility vs Temperature Plot Created",
                additional_kwargs={"strap_origin": "direct_tool_fast_path"},
            ),
        ]
    }

    assert middleware.after_model(state, runtime=MagicMock()) is None
    classifier.invoke.assert_not_called()


def test_simple_solvent_lookup_examples_do_not_route_to_specialists():
    from strap.routing_classifier import explain_routing_decision

    queries = [
        "what are good solvents for dissolving LDPE",
        "list available solvents for LDPE",
        "what solvents dissolve PET",
    ]

    for query in queries:
        decision = explain_routing_decision(query)
        assert decision["direct_answer"]["is_direct"] is True
        assert decision["planned"] == []


def test_complex_solvent_queries_still_route_to_specialists():
    from strap.routing_classifier import classify_query_keywords, plan_workflow_rules

    cases = {
        "rank solvents for separating LDPE from EVOH/PET": ["separation-engineer"],
        "plan a separation sequence for LDPE/EVOH/PET": ["separation-engineer"],
        "Use Hansen solubility parameters to screen LDPE solvents": ["statistics-ml"],
        "Estimate CAPEX/OPEX/GWP for LDPE recovered with Cyclohexane under energy case C1.": ["biosteam-analyst"],
    }

    for query, expected in cases.items():
        keyword_rules = classify_query_keywords([HumanMessage(content=query)])
        planned = plan_workflow_rules(query, keyword_rules)
        assert [rule["subagent"] for rule in planned] == expected


def test_classify_query_prefers_separation_then_contaminant_for_route_screen_query():
    from strap.routing_classifier import classify_query

    hint = classify_query(
        [
            HumanMessage(
                content=(
                    "First do process design for separation, then contaminant screening. "
                    "For an EVOH/PE multilayer contaminated with di-n-butyl phthalate (DBP), "
                    "identify the best atmospheric-pressure solvent route and then compare "
                    "leaching versus STRAP contaminant removal."
                )
            )
        ]
    )

    assert hint is not None
    assert "separation-engineer" in hint
    assert "contaminant-removal-analyst" in hint
    assert "biosteam-analyst" not in hint


def test_classify_query_prefers_contaminant_only_when_tea_and_safety_are_negated():
    from strap.routing_classifier import classify_query

    hint = classify_query(
        [
            HumanMessage(
                content=(
                    "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. "
                    "For EVOH contaminated with di-n-butyl phthalate (DBP), compare leaching versus STRAP contaminant removal."
                )
            )
        ]
    )

    assert hint is not None
    assert "contaminant-removal-analyst" in hint
    assert "biosteam-analyst" not in hint
    assert "safety-analyst" not in hint
    assert "separation-engineer" not in hint


def test_classify_query_supports_four_stage_research_workflow():
    from strap.routing_classifier import classify_query

    hint = classify_query(
        [
            HumanMessage(
                content=(
                    "Do a literature search and patent search for multilayer polymer recycling methods, "
                    "answer the question with RAG, then create a chart visualization of the retrieved findings."
                )
            )
        ]
    )

    assert hint is not None
    assert "scholar-researcher" in hint
    assert "patent-researcher" in hint
    assert "rag-analyst" in hint
    assert "visualization-specialist" in hint
    assert "staged workflow" in hint


def test_plan_workflow_rules_can_start_from_query_goals_without_seed_rules():
    from strap.routing_classifier import plan_workflow_rules

    planned = plan_workflow_rules(
        (
            "Do a literature search and patent search for multilayer polymer recycling methods, "
            "answer the question with RAG, then create a chart visualization of the retrieved findings."
        ),
        None,
    )

    assert [rule["subagent"] for rule in planned] == [
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "visualization-specialist",
    ]


def test_plan_workflow_rules_preserves_goal_chain_when_seed_is_empty_for_process_query():
    from strap.routing_classifier import plan_workflow_rules

    planned = plan_workflow_rules(
        (
            "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream using "
            "selective dissolution at atmospheric pressure. Propose up to 1 additional wash "
            "step for phthalate removal. Then run a techno-economic analysis on solvent recovery."
        ),
        [],
    )

    assert [rule["subagent"] for rule in planned] == [
        "separation-engineer",
        "contaminant-removal-analyst",
        "biosteam-analyst",
    ]


def test_runtime_planner_inserts_uncovered_contaminant_goal_for_complex_mixed_query():
    from strap.routing import RoutingMiddleware
    from strap.routing_message_state import _get_ordered_plan

    query = (
        "Do a literature search and patent search for solvent-based delamination of HDPE/EVOH "
        "food-packaging laminates, answer the question with RAG, then design an optimal "
        "atmospheric-pressure separation sequence for an HDPE/EVOH mixed waste stream using "
        "selective dissolution. Propose up to 1 additional wash step for phthalate removal, "
        "then run a techno-economic analysis on solvent recovery for the best option, and "
        "finally create a chart summarizing both the retrieved findings and the process results."
    )
    messages = [HumanMessage(content=query)]

    middleware = RoutingMiddleware(classifier_model=None)
    allowed = middleware._get_allowed_rules(messages)
    names = [rule["subagent"] for rule in allowed]

    assert set(names) == {
        "separation-engineer",
        "contaminant-removal-analyst",
        "biosteam-analyst",
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "visualization-specialist",
    }

    plan = _get_ordered_plan(messages, allowed_rules=allowed)
    dependency_map = {step["subagent"]: step["depends_on"] for step in plan}

    assert dependency_map["separation-engineer"] == ()
    assert dependency_map["scholar-researcher"] == ()
    assert dependency_map["patent-researcher"] == ()
    assert dependency_map["contaminant-removal-analyst"] == ("separation-engineer",)
    assert dependency_map["biosteam-analyst"] == ("contaminant-removal-analyst",)
    assert dependency_map["rag-analyst"] == ("scholar-researcher", "patent-researcher")
    assert set(dependency_map["visualization-specialist"]) == {"biosteam-analyst", "rag-analyst"}


def test_infer_requested_goals_uses_query_context_for_process_query():
    from strap.routing_classifier import infer_requested_goals

    goals = infer_requested_goals(
        (
            "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream using "
            "selective dissolution at atmospheric pressure. Propose up to 1 additional wash "
            "step for phthalate removal. Then run a techno-economic analysis on solvent recovery."
        )
    )

    assert {
        "separation.route",
        "separation.feasibility",
        "contaminant.screening",
        "contaminant.removal",
        "tea.economics",
    }.issubset(goals)
    assert "safety.assessment" not in goals


def test_infer_requested_goals_respects_negated_query_context_requests():
    from strap.routing_classifier import infer_requested_goals

    goals = infer_requested_goals(
        (
            "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. "
            "For EVOH contaminated with di-n-butyl phthalate (DBP), compare leaching versus STRAP contaminant removal."
        )
    )

    assert goals == {"contaminant.screening", "contaminant.removal"}


def test_infer_requested_goals_ignores_negated_solvent_screening_for_biosteam_only_query():
    from strap.routing_classifier import infer_requested_goals

    goals = infer_requested_goals(
        (
            "Only do BioSTEAM TEA/LCA. For LDPE dissolved in toluene, run a techno-economic "
            "analysis and life-cycle assessment under energy case C1. Do not do solvent screening "
            "or process design."
        )
    )

    assert goals == {"tea.economics", "lca.environmental"}


def test_infer_requested_goals_ignores_negated_optimization_for_biosteam_only_query():
    from strap.routing_classifier import infer_requested_goals

    goals = infer_requested_goals(
        (
            "Only do BioSTEAM TEA/LCA. Estimate CAPEX/OPEX/GWP for LDPE recovered "
            "with Cyclohexane under energy case C1. Do not do separation planning, "
            "waste optimization, or Pareto optimization."
        )
    )

    assert goals == {"tea.economics", "lca.environmental"}


def test_plan_workflow_excludes_negated_separation_and_optimization_for_tea_lca_query():
    from langchain_core.messages import HumanMessage

    from strap.routing_classifier import classify_query_keywords, plan_workflow_rules

    query = (
        "Only do BioSTEAM TEA/LCA. Estimate CAPEX/OPEX/GWP for a STRAP process case: "
        "LDPE recovered with Cyclohexane at 79.7 C, 8000 tonnes/year total feed capacity, "
        "60 wt% LDPE in the feed, energy case C1. Use the BioSTEAM TEA/LCA simulation tool. "
        "Report MSP, TCI/CAPEX, AOC/OPEX, and GWP with units. Do not do separation planning, "
        "waste optimization, or Pareto optimization."
    )

    keyword_rules = classify_query_keywords([HumanMessage(content=query)])
    planned = plan_workflow_rules(query, keyword_rules)

    assert [rule["subagent"] for rule in planned] == ["biosteam-analyst"]


def test_plan_workflow_routes_natural_capex_opex_gwp_query_to_biosteam_only():
    from langchain_core.messages import HumanMessage

    from strap.routing_classifier import classify_query_keywords, plan_workflow_rules

    query = (
        "Estimate CAPEX/OPEX/GWP for a STRAP process case: LDPE recovered with "
        "Cyclohexane at 79.7 C, 8000 tonnes/year total feed capacity, 60 wt% "
        "LDPE in the feed, energy case C1. Report MSP, TCI/CAPEX, AOC/OPEX, "
        "and GWP with units."
    )

    keyword_rules = classify_query_keywords([HumanMessage(content=query)])
    planned = plan_workflow_rules(query, keyword_rules)

    assert [rule["subagent"] for rule in planned] == ["biosteam-analyst"]


def test_plan_workflow_routes_fixed_process_temperature_caveat_query_to_biosteam_only():
    from langchain_core.messages import HumanMessage

    from strap.routing_classifier import classify_query_keywords, plan_workflow_rules

    query = (
        "Estimate CAPEX/OPEX/GWP for recovering EVOH with Dimethyl sulfoxide "
        "at 140 C in a STRAP process at 5000 tonnes/year under energy case C3. "
        "Report MSP, CAPEX, OPEX, GWP, and any boiling-point feasibility caveat."
    )

    keyword_rules = classify_query_keywords([HumanMessage(content=query)])
    planned = plan_workflow_rules(query, keyword_rules)

    assert [rule["subagent"] for rule in planned] == ["biosteam-analyst"]


def test_infer_requested_goals_uses_query_context_for_research_workflow():
    from strap.routing_classifier import infer_requested_goals

    goals = infer_requested_goals(
        (
            "Do a literature search and patent search for multilayer polymer recycling methods, "
            "answer the question with RAG, then create a chart visualization of the retrieved findings."
        )
    )

    assert goals == {
        "literature.search",
        "patent.search",
        "literature.answer",
        "rag.retrieval",
        "visualization.plot",
    }


def test_infer_requested_goals_prefers_optimization_goal_for_optimization_only_query():
    from strap.routing_classifier import infer_requested_goals

    goals = infer_requested_goals(
        (
            "Optimize waste management for an 8000 t/y multilayer feed of 40% PE, 40% PET, "
            "1% Nylon-6, and 19% EVOH. Maximize profit and report emissions."
        )
    )

    assert goals == {"optimization.pathway"}


def test_infer_available_query_inputs_maps_core_process_requirements():
    from strap.routing_classifier import infer_available_query_inputs

    available = infer_available_query_inputs(
        (
            "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream using "
            "selective dissolution at atmospheric pressure. Propose up to 1 additional wash "
            "step for phthalate removal. Then run a techno-economic analysis on solvent recovery."
        )
    )

    assert {
        "user.polymers",
        "user.target_plastic",
        "user.target_polymer",
        "user.contaminants",
        "user.solvents_or_route",
    }.issubset(available)


def test_plan_workflow_rules_routes_optimization_only_query_to_optimization_engineer():
    from strap.routing_classifier import plan_workflow_rules

    planned = plan_workflow_rules(
        (
            "Optimize waste management for an 8000 t/y multilayer feed of 40% PE, 40% PET, "
            "1% Nylon-6, and 19% EVOH. Maximize profit and report emissions."
        ),
        None,
    )

    assert [rule["subagent"] for rule in planned] == ["optimization-engineer"]


def test_plan_workflow_rules_keeps_explicit_process_and_optimization_goals_together():
    from strap.routing_classifier import derive_workflow_dependencies, plan_workflow_rules

    query = (
        "Find a separation route for a PE/PET/EVOH/Nylon-6 multilayer film, run TEA/LCA "
        "on the route, then optimize waste management for profit and emissions."
    )

    planned = plan_workflow_rules(query, None)
    names = [rule["subagent"] for rule in planned]

    assert names == [
        "separation-engineer",
        "biosteam-analyst",
        "optimization-engineer",
    ]

    dependencies = derive_workflow_dependencies(query, set(names))
    assert dependencies["separation-engineer"] == set()
    assert dependencies["biosteam-analyst"] == {"separation-engineer"}
    assert dependencies["optimization-engineer"] == {"separation-engineer"}


def test_plan_workflow_rules_routes_separation_to_optimization_to_visualization_without_biosteam_for_pareto_query():
    from strap.routing_classifier import plan_workflow_rules

    query = (
        "For an LDPE/EVOH/PET film, use the top separation routes as candidates, "
        "run route-constrained optimization, broaden to the optimizer-supported candidate catalog "
        "if the ranked routes are infeasible, and generate a Pareto front plot of total cost vs emissions."
    )

    planned = plan_workflow_rules(query, None)

    assert [rule["subagent"] for rule in planned] == [
        "separation-engineer",
        "optimization-engineer",
        "visualization-specialist",
    ]


def test_get_allowed_rules_falls_back_to_keyword_sequential_match_when_llm_route_is_weaker():
    from strap.routing import RoutingMiddleware

    response = MagicMock()
    response.content = '{"subagents": ["contaminant-removal-analyst"], "confidence": "HIGH"}'
    model = MagicMock()
    model.invoke.return_value = response

    middleware = RoutingMiddleware(classifier_model=model)
    messages = [
        HumanMessage(
            content=(
                "First do process design for separation, then contaminant screening. "
                "For an EVOH/PE multilayer contaminated with DBP, identify the best route "
                "and then compare leaching versus STRAP contaminant removal."
            )
        )
    ]

    allowed = middleware._get_allowed_rules(messages)

    names = [rule["subagent"] for rule in allowed]
    assert names[:2] == ["separation-engineer", "contaminant-removal-analyst"]


def test_routing_middleware_selects_optimization_engineer_for_optimization_only_query():
    from strap.routing import RoutingMiddleware

    query = (
        "Optimize waste management for an 8000 t/y multilayer feed of 40% PE, 40% PET, "
        "1% Nylon-6, and 19% EVOH. Maximize profit and report emissions."
    )
    messages = [HumanMessage(content=query)]

    middleware = RoutingMiddleware(classifier_model=None)
    allowed = middleware._get_allowed_rules(messages)

    assert [rule["subagent"] for rule in allowed] == ["optimization-engineer"]


def test_routing_middleware_keeps_explicit_process_chain_plus_optimization():
    from strap.routing import RoutingMiddleware
    from strap.routing_message_state import _get_ordered_plan

    query = (
        "Find a separation route for a PE/PET/EVOH/Nylon-6 multilayer film, run TEA/LCA "
        "on the route, then optimize waste management for profit and emissions."
    )
    messages = [HumanMessage(content=query)]

    middleware = RoutingMiddleware(classifier_model=None)
    allowed = middleware._get_allowed_rules(messages)

    assert set(rule["subagent"] for rule in allowed) == {
        "separation-engineer",
        "biosteam-analyst",
        "optimization-engineer",
    }

    plan = _get_ordered_plan(messages, allowed_rules=allowed)
    dependency_map = {step["subagent"]: step["depends_on"] for step in plan}

    assert dependency_map["optimization-engineer"] == ("separation-engineer",)
    assert dependency_map["separation-engineer"] == ()
    assert dependency_map["biosteam-analyst"] == ("separation-engineer",)


def test_get_allowed_rules_plans_four_stage_research_workflow():
    from strap.routing import RoutingMiddleware
    from strap.routing_message_state import _get_ordered_plan

    messages = [
        HumanMessage(
            content=(
                "Do a literature search and patent search for multilayer polymer recycling methods, "
                "answer the question with RAG, then create a chart visualization of the retrieved findings."
            )
        )
    ]

    middleware = RoutingMiddleware()
    allowed = middleware._get_allowed_rules(messages)

    assert [rule["subagent"] for rule in allowed] == [
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "visualization-specialist",
    ]

    plan = _get_ordered_plan(messages, allowed_rules=allowed)
    assert [step["subagent"] for step in plan] == [
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "visualization-specialist",
    ]
    assert plan[0]["depends_on"] == ()
    assert plan[1]["depends_on"] == ()
    assert plan[2]["depends_on"] == ("scholar-researcher", "patent-researcher")
    assert plan[3]["depends_on"] == ("rag-analyst",)


def test_wrap_tool_call_blocks_write_todos_before_first_sequential_dispatch():
    from strap.routing import RoutingMiddleware

    response = MagicMock()
    response.content = '{"subagents": ["contaminant-removal-analyst"], "confidence": "HIGH"}'
    model = MagicMock()
    model.invoke.return_value = response
    middleware = RoutingMiddleware(classifier_model=model)

    request = ToolCallRequest(
        tool_call={
            "id": "todo_seq",
            "name": "write_todos",
            "args": {"todos": [{"content": "start separation", "status": "pending"}]},
        },
        tool=None,
        state={
            "messages": [
                HumanMessage(
                    content=(
                        "First do process design for separation, then contaminant screening "
                        "for an EVOH/PE film with DBP contamination."
                    )
                )
            ]
        },
        runtime=MagicMock(),
    )

    handler = MagicMock()
    result = middleware.wrap_tool_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "todo_seq"
    assert "write_todos" in result.content
    assert "separation-engineer" in result.content


def test_incomplete_route_retry_hint_starts_multi_specialist_workflow():
    from strap.routing_guards import _build_incomplete_route_retry_hint

    messages = [
        HumanMessage(
            content=(
                "First do process design for separation, then contaminant screening "
                "for an EVOH/PE film with DBP contamination."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "contaminant-removal-analyst", "description": "contaminant screening"},
    ]

    hint = _build_incomplete_route_retry_hint(messages, allowed_rules)

    assert hint is not None
    assert 'task(subagent_type="separation-engineer")' in hint


def test_initial_route_task_response_dispatches_first_specialist():
    from strap.routing_guards import _build_initial_route_task_response

    messages = [
        HumanMessage(
            content=(
                "First do process design for separation, then contaminant screening "
                "for an EVOH/PE film with DBP contamination."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "contaminant-removal-analyst", "description": "contaminant screening"},
    ]

    response = _build_initial_route_task_response(messages, allowed_rules)

    assert response is not None
    ai_msg = response.result[0]
    assert ai_msg.tool_calls[0]["name"] == "task"
    assert ai_msg.tool_calls[0]["args"]["subagent_type"] == "separation-engineer"
    assert "DBP contamination" in ai_msg.tool_calls[0]["args"]["description"]


def test_ordered_plan_places_contaminant_before_biosteam_for_integrated_query():
    from strap.routing_message_state import _get_ordered_plan

    messages = [
        HumanMessage(
            content=(
                "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
                "using selective dissolution at atmospheric pressure. Propose up to 1 "
                "additional wash step for phthalate removal. Then run a techno-economic "
                "analysis on the solvent recovery for the best option."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "biosteam-analyst", "description": "tea"},
        {"subagent": "contaminant-removal-analyst", "description": "contaminant screening"},
    ]

    plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)

    assert [step["subagent"] for step in plan] == [
        "separation-engineer",
        "contaminant-removal-analyst",
        "biosteam-analyst",
    ]
    assert plan[0]["depends_on"] == ()
    assert plan[1]["depends_on"] == ("separation-engineer",)
    assert plan[2]["depends_on"] == ("contaminant-removal-analyst",)


def test_derive_workflow_dependencies_prefers_transitive_contaminant_handoff_for_biosteam():
    from strap.routing_classifier import derive_workflow_dependencies

    dependencies = derive_workflow_dependencies(
        (
            "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
            "using selective dissolution at atmospheric pressure. Propose up to 1 "
            "additional wash step for phthalate removal. Then run a techno-economic "
            "analysis on the solvent recovery for the best option."
        ),
        {
            "separation-engineer",
            "contaminant-removal-analyst",
            "biosteam-analyst",
        },
    )

    assert dependencies["separation-engineer"] == set()
    assert dependencies["contaminant-removal-analyst"] == {"separation-engineer"}
    assert dependencies["biosteam-analyst"] == {"contaminant-removal-analyst"}


def test_pending_required_handoff_tracks_depends_on_for_three_step_workflow():
    from strap.routing import _get_pending_required_handoff

    messages = [
        HumanMessage(
            content=(
                "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
                "using selective dissolution at atmospheric pressure. Propose up to 1 "
                "additional wash step for phthalate removal. Then run a techno-economic "
                "analysis on the solvent recovery for the best option."
            )
        ),
        AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
        ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "contaminant-removal-analyst", "description": "contaminant screening"},
        {"subagent": "biosteam-analyst", "description": "tea"},
    ]

    pending = _get_pending_required_handoff(messages, allowed_rules)

    assert pending == ("separation-engineer", "contaminant-removal-analyst")


def test_pending_required_handoff_tracks_each_missing_join_edge_in_order():
    from strap.routing import _get_pending_required_handoff

    query = (
        "Do a literature search and patent search for multilayer polymer recycling methods, "
        "answer the question with RAG."
    )
    allowed_rules = [
        {"subagent": "scholar-researcher", "description": "literature"},
        {"subagent": "patent-researcher", "description": "patents"},
        {"subagent": "rag-analyst", "description": "rag"},
    ]
    messages = [
        HumanMessage(content=query),
        AIMessage(content="", tool_calls=[_task_call("tc_scholar", "scholar-researcher")]),
        ToolMessage(content=_structured_result_content("scholar-researcher"), tool_call_id="tc_scholar"),
        AIMessage(content="", tool_calls=[_task_call("tc_patent", "patent-researcher")]),
        ToolMessage(content=_structured_result_content("patent-researcher"), tool_call_id="tc_patent"),
    ]

    pending = _get_pending_required_handoff(messages, allowed_rules)
    assert pending == ("scholar-researcher", "rag-analyst")

    messages.extend([
        AIMessage(content="", tool_calls=[{
            "id": "bh_scholar",
            "name": "build_handoff",
            "args": {"consumer": "rag-analyst", "producer": "scholar-researcher"},
        }]),
        ToolMessage(
            content=(
                '{"ok": true, "handoff": {"handoff_id": "h_scholar_rag", '
                '"producer": "scholar-researcher", "consumer": "rag-analyst", '
                '"contract": "literature_context.v1", "status": "ok", '
                '"task_prompt": "Use the literature findings."}}'
            ),
            tool_call_id="bh_scholar",
        ),
    ])

    pending = _get_pending_required_handoff(messages, allowed_rules)
    assert pending == ("patent-researcher", "rag-analyst")


def test_ready_downstream_handoff_for_join_requires_all_direct_handoffs():
    from strap.routing import _get_ready_downstream_handoff

    query = (
        "Do a literature search and patent search for multilayer polymer recycling methods, "
        "answer the question with RAG."
    )
    allowed_rules = [
        {"subagent": "scholar-researcher", "description": "literature"},
        {"subagent": "patent-researcher", "description": "patents"},
        {"subagent": "rag-analyst", "description": "rag"},
    ]
    messages = [
        HumanMessage(content=query),
        AIMessage(content="", tool_calls=[_task_call("tc_scholar", "scholar-researcher")]),
        ToolMessage(content=_structured_result_content("scholar-researcher"), tool_call_id="tc_scholar"),
        AIMessage(content="", tool_calls=[_task_call("tc_patent", "patent-researcher")]),
        ToolMessage(content=_structured_result_content("patent-researcher"), tool_call_id="tc_patent"),
        AIMessage(content="", tool_calls=[{
            "id": "bh_scholar",
            "name": "build_handoff",
            "args": {"consumer": "rag-analyst", "producer": "scholar-researcher"},
        }]),
        ToolMessage(
            content=(
                '{"ok": true, "handoff": {"handoff_id": "h_scholar_rag", '
                '"producer": "scholar-researcher", "consumer": "rag-analyst", '
                '"contract": "literature_context.v1", "status": "ok", '
                '"task_prompt": "Use the literature findings."}}'
            ),
            tool_call_id="bh_scholar",
        ),
    ]

    assert _get_ready_downstream_handoff(messages, allowed_rules) is None

    messages.extend([
        AIMessage(content="", tool_calls=[{
            "id": "bh_patent",
            "name": "build_handoff",
            "args": {"consumer": "rag-analyst", "producer": "patent-researcher"},
        }]),
        ToolMessage(
            content=(
                '{"ok": true, "handoff": {"handoff_id": "h_patent_rag", '
                '"producer": "patent-researcher", "consumer": "rag-analyst", '
                '"contract": "patent_context.v1", "status": "ok", '
                '"task_prompt": "Use the patent findings."}}'
            ),
            tool_call_id="bh_patent",
        ),
    ])

    ready = _get_ready_downstream_handoff(messages, allowed_rules)

    assert ready is not None
    assert ready["consumer"] == "rag-analyst"
    assert set(ready["handoff_ids"]) == {"h_scholar_rag", "h_patent_rag"}
    assert "scholar-researcher" in ready["task_prompt"]
    assert "patent-researcher" in ready["task_prompt"]


def test_ordered_plan_prefers_biosteam_closure_for_visualization_chain():
    from strap.routing_message_state import _get_ordered_plan

    messages = [
        HumanMessage(
            content=(
                "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
                "using selective dissolution at atmospheric pressure. Then run a techno-economic "
                "analysis on the solvent recovery for the best option and create a chart of the TEA results."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "biosteam-analyst", "description": "tea"},
        {"subagent": "visualization-specialist", "description": "visualization"},
    ]

    plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)

    assert [step["subagent"] for step in plan] == [
        "separation-engineer",
        "biosteam-analyst",
        "visualization-specialist",
    ]
    assert plan[0]["depends_on"] == ()
    assert plan[1]["depends_on"] == ("separation-engineer",)
    assert plan[2]["depends_on"] == ("biosteam-analyst",)


def test_ordered_plan_uses_generic_fallback_for_literature_then_visualization_query():
    from strap.routing_message_state import _get_ordered_plan

    messages = [
        HumanMessage(
            content=(
                "Do a literature search for multilayer polymer recycling methods, "
                "then create a chart summarizing the papers."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "scholar-researcher", "description": "literature search"},
        {"subagent": "visualization-specialist", "description": "visualization"},
    ]

    plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)

    assert [step["subagent"] for step in plan] == [
        "scholar-researcher",
        "visualization-specialist",
    ]
    assert plan[0]["depends_on"] == ()
    assert plan[1]["depends_on"] == ("scholar-researcher",)


def test_initial_route_task_response_dispatches_first_specialist_for_three_step_workflow():
    from strap.routing_guards import _build_initial_route_task_response

    messages = [
        HumanMessage(
            content=(
                "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
                "using selective dissolution at atmospheric pressure. Propose up to 1 "
                "additional wash step for phthalate removal. Then run a techno-economic "
                "analysis on the solvent recovery for the best option."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "contaminant-removal-analyst", "description": "contaminant screening"},
        {"subagent": "biosteam-analyst", "description": "tea"},
    ]

    response = _build_initial_route_task_response(messages, allowed_rules)

    assert response is not None
    ai_msg = response.result[0]
    assert ai_msg.tool_calls[0]["name"] == "task"
    assert ai_msg.tool_calls[0]["args"]["subagent_type"] == "separation-engineer"


def test_validate_task_tool_call_uses_active_ordered_plan_for_contaminant_sequence():
    from strap.routing_guards import _validate_task_tool_call

    messages = [
        HumanMessage(
            content=(
                "First do process design for separation, then contaminant screening "
                "for an EVOH/PE film with DBP contamination."
            )
        )
    ]
    allowed_rules = [
        {"subagent": "separation-engineer", "description": "process design"},
        {"subagent": "contaminant-removal-analyst", "description": "contaminant screening"},
    ]

    separation_call = _task_call("tc_sep_start", "separation-engineer", "Design the separation route first.")
    contaminant_call = _task_call(
        "tc_contam_early",
        "contaminant-removal-analyst",
        "Screen contaminant removal before separation is complete.",
    )

    assert _validate_task_tool_call(separation_call, messages, allowed_rules) is None
    validation_error = _validate_task_tool_call(contaminant_call, messages, allowed_rules)
    assert validation_error is not None
    assert "Complete the upstream specialist step first" in validation_error
    assert "separation-engineer" in validation_error


def test_validate_task_tool_call_blocks_biosteam_after_optimization_without_tea_intent():
    """Post-optimization biosteam-analyst dispatch is blocked unless the user explicitly asked for TEA/LCA."""
    from strap.routing_guards import _validate_task_tool_call

    messages = [
        HumanMessage(
            content=(
                "For a mixed plastic feedstock, run a Pareto sweep of total cost vs emissions "
                "for the shortlisted solvents."
            )
        ),
        AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
        ToolMessage(
            content=_structured_result_content("optimization-engineer"),
            tool_call_id="tc_opt",
        ),
    ]
    allowed_rules = [
        {"subagent": "optimization-engineer", "description": "run Pareto optimization"},
        {"subagent": "biosteam-analyst", "description": "tea/lca"},
    ]
    call = _task_call("tc_bio", "biosteam-analyst", "Re-run BioSTEAM for the five sequences.")

    err = _validate_task_tool_call(call, messages, allowed_rules)
    assert err is not None
    assert "biosteam-analyst" in err
    assert "optimization-engineer" in err
    assert "duplicate" in err.lower() or "internally" in err.lower()


def test_validate_task_tool_call_allows_biosteam_when_user_requests_tea_lca():
    """When the user explicitly asked for TEA/LCA, biosteam-analyst must still be dispatchable."""
    from strap.routing_guards import _validate_task_tool_call

    messages = [
        HumanMessage(
            content=(
                "Run the Pareto sweep, then give me a detailed TEA and LCA breakdown "
                "with capex/opex for the top route."
            )
        ),
        AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
        ToolMessage(
            content=_structured_result_content("optimization-engineer"),
            tool_call_id="tc_opt",
        ),
    ]
    allowed_rules = [
        {"subagent": "optimization-engineer", "description": "run Pareto"},
        {"subagent": "biosteam-analyst", "description": "detailed TEA/LCA"},
    ]
    call = _task_call("tc_bio", "biosteam-analyst", "Run detailed TEA and LCA.")

    # Explicit TEA intent means the post-opt dedup does NOT fire. Predecessor
    # guards may still block if a build_handoff is expected, but the dedup guard
    # itself must not be the reason.
    err = _validate_task_tool_call(call, messages, allowed_rules)
    if err is not None:
        assert "duplicate that work" not in err


class TestRoutingProgress:
    def test_completed_subagents_preserve_repeated_calls(self):
        from strap.routing import _extract_completed_subagents

        messages = [
            HumanMessage(content="run safety twice"),
            AIMessage(content="", tool_calls=[_task_call("tc1", "safety-analyst")]),
            ToolMessage(content=_structured_result_content("safety-analyst"), tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[_task_call("tc2", "safety-analyst")]),
            ToolMessage(content=_structured_result_content("safety-analyst"), tool_call_id="tc2"),
        ]

        completed = _extract_completed_subagents(messages)
        assert completed == ["safety-analyst", "safety-analyst"]

    def test_ordered_plan_keeps_duplicate_dispatches(self):
        from strap.routing import _get_ordered_plan

        messages = [
            HumanMessage(content="run safety twice and then viz"),
            AIMessage(content="", tool_calls=[_task_call("tc1", "safety-analyst")]),
            ToolMessage(content="done 1", tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[_task_call("tc2", "safety-analyst")]),
            ToolMessage(content="done 2", tool_call_id="tc2"),
            AIMessage(content="", tool_calls=[_task_call("tc3", "visualization-specialist")]),
        ]

        plan = _get_ordered_plan(
            messages,
            allowed_rules=[
                {"subagent": "safety-analyst", "description": "safety"},
                {"subagent": "visualization-specialist", "description": "viz"},
            ],
        )
        assert [step["subagent"] for step in plan[:3]] == [
            "safety-analyst",
            "safety-analyst",
            "visualization-specialist",
        ]
        assert [step["step_id"] for step in plan[:3]] == ["tc1", "tc2", "tc3"]

    def test_ordered_plan_ignores_router_guarded_task_calls(self):
        from strap.routing import _get_ordered_plan

        messages = [
            HumanMessage(content="run stats then viz"),
            AIMessage(content="", tool_calls=[_task_call("tc1", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[_task_call("tc2", "statistics-ml")]),
            ToolMessage(
                content="Router guard: statistics task should not be repeated.",
                tool_call_id="tc2",
                status="error",
            ),
        ]

        plan = _get_ordered_plan(
            messages,
            allowed_rules=[
                {"subagent": "statistics-ml", "description": "stats"},
                {"subagent": "visualization-specialist", "description": "viz"},
            ],
        )

        assert [step["subagent"] for step in plan] == [
            "statistics-ml",
            "visualization-specialist",
        ]

    def test_missing_handoff_does_not_count_task_as_completed(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware
        from strap.routing import (
            _build_progress_directive,
            _extract_completed_subagents,
            _extract_failed_subagent_calls,
            _get_ordered_plan,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        request = MagicMock()
        request.tool_call = _task_call("tc1", "statistics-ml")

        missing = ToolMessage(
            content="Only prose, no structured block.",
            tool_call_id="tc1",
        )
        mw.wrap_tool_call(request, MagicMock(return_value=missing))

        messages = [
            HumanMessage(content="run statistics"),
            AIMessage(content="", tool_calls=[_task_call("tc1", "statistics-ml")]),
            missing,
        ]

        completed = _extract_completed_subagents(messages)
        failed = _extract_failed_subagent_calls(messages)
        plan = _get_ordered_plan(messages)
        progress = _build_progress_directive(messages, set(), plan, failed_ids={"tc1"})

        assert completed == []
        assert [step["subagent"] for step in failed] == ["statistics-ml"]
        assert "Failed subagents: statistics-ml" in progress

    def test_progress_directive_treats_failed_ids_as_not_completed(self):
        from strap.routing import _build_progress_directive

        messages = [HumanMessage(content="run separation then tea")]
        ordered_plan = [
            {"subagent": "separation-engineer", "description": "sep", "step_id": "tc_sep"},
            {"subagent": "biosteam-analyst", "description": "tea", "step_id": "tc_tea"},
        ]

        progress = _build_progress_directive(
            messages,
            {"tc_sep", "tc_tea"},
            ordered_plan,
            failed_ids={"tc_tea"},
        )

        assert "Completed subagents: separation-engineer" in progress
        assert "Failed subagents: biosteam-analyst" in progress

    def test_completion_progress_anchor_uses_validated_separation_constraints(self):
        from strap.routing import _build_progress_directive

        messages = [
            HumanMessage(content="Find the optimal separation sequence for PS, PMMA, and PET up to 120C."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                    '"supported_polymers":["PS","PET"],"unsupported_polymers":["PMMA"],'
                    '"best_sequence":["PS","PET"],'
                    '"steps":[{"step":1,"polymer":"PS","solvent":"Tetrahydropyran","temperature_c":85.0}],'
                    '"solvent_mapping":{"PS":"Tetrahydropyran"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Tetrahydropyran"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz"),
        ]
        ordered_plan = [
            {"subagent": "separation-engineer", "description": "sep", "step_id": "tc_sep"},
            {"subagent": "visualization-specialist", "description": "viz", "step_id": "tc_viz"},
        ]

        progress = _build_progress_directive(messages, {"tc_sep", "tc_viz"}, ordered_plan)

        assert "Use only validated subagent outputs" in progress
        assert "supported subset (PS, PET)" in progress
        assert "unsupported: PMMA" in progress
        assert "phase behavior is unknown without additional data" in progress
        assert "Tetrahydropyran at 85.0C" in progress
        assert "remains below the boiling point at 1 atm" in progress
        assert "narrow atmospheric-pressure operating margin" in progress

    def test_completion_progress_directive_forces_final_synthesis(self):
        from strap.routing import _build_progress_directive

        messages = [HumanMessage(content="Run TEA on PE with Toluene.")]
        ordered_plan = [
            {"subagent": "biosteam-analyst", "description": "tea", "step_id": "tc_tea"},
        ]

        progress = _build_progress_directive(messages, {"tc_tea"}, ordered_plan)

        assert "Write the final answer now using the completed subagent outputs" in progress
        assert "Do not call any more tools" in progress

    def test_completion_progress_anchor_prioritizes_optimization_routes_over_separation_steps(self):
        from strap.routing import _build_progress_directive

        messages = [
            HumanMessage(content="Compare optimized routes for LDPE, EVOH, and PET."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["LDPE","EVOH","PET"],'
                    '"best_sequence":["LDPE","EVOH","PET"],'
                    '"steps":[{"step":1,"polymer":"LDPE","solvent":"Cyclohexane","temperature_c":76.0}],'
                    '"solvent_mapping":{"LDPE":"Cyclohexane"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["LDPE","EVOH","PET"],"solvent_mapping":{"LDPE":"Cyclohexane","EVOH":"Methanol"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.3","analysis_type":"pareto_front",'
                    '"x_metric":"total_cost","y_metric":"emissions","n_points_feasible":1,'
                    '"n_routes_requested":2,"n_routes_solved":1,'
                    '"points":[{"point_id":1,"route_id":"route_1","total_cost":1000,"emissions":100}],'
                    '"route_reports":[{"route_id":"route_1","status":"solved","polymer_solvent_map":{"PE":"Cyclohexane","EVOH":"Dimethyl sulfoxide"}},{"route_id":"route_2","status":"skipped","reason":"not in catalog","polymer_solvent_map":{"PE":"Cyclohexane","EVOH":"Methanol"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
        ]
        ordered_plan = [
            {"subagent": "separation-engineer", "description": "sep", "step_id": "tc_sep"},
            {"subagent": "optimization-engineer", "description": "opt", "step_id": "tc_opt"},
        ]

        progress = _build_progress_directive(messages, {"tc_sep", "tc_opt"}, ordered_plan)

        assert "Optimization-engineer returned the latest validated downstream result" in progress
        assert "route-constrained optimization result" in progress
        assert "Validated optimization route route_1: PE-Cyclohexane, EVOH-Dimethyl sulfoxide." in progress
        assert "Do not restate upstream separation step numbers" in progress
        assert "Preserve validated Step 1" not in progress

    def test_completion_progress_anchor_uses_multi_slice_optimization_summary(self):
        from strap.routing import _build_progress_directive

        messages = [
            HumanMessage(content="Run Pareto slices for 20/60/20 and 60/20/20."),
            AIMessage(content="", tool_calls=[_task_call("tc_opt_slices", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.0","analysis_type":"pareto_slices",'
                    '"x_metric":"total_cost","y_metric":"circularity","n_slices_requested":2,"n_slices_solved":2,'
                    '"n_points_requested_per_slice":100,"pareto_slices_payload_path":"/tmp/pareto_slices.json",'
                    '"slices":[{"slice_id":"slice_1","label":"ldpe20_evoh60_pet20","status":"solved","n_points_feasible":7,"max_circularity":0.82},'
                    '{"slice_id":"slice_2","label":"ldpe60_evoh20_pet20","status":"solved","n_points_feasible":5,"max_circularity":0.78}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt_slices",
            ),
        ]
        ordered_plan = [
            {"subagent": "optimization-engineer", "description": "opt", "step_id": "tc_opt_slices"},
        ]

        progress = _build_progress_directive(messages, {"tc_opt_slices"}, ordered_plan)

        assert "multi-composition Pareto-slice optimization result" in progress
        assert "Report 2 solved slices out of 2 requested" in progress
        assert "/tmp/pareto_slices.json" in progress
        assert "Validated slice ldpe20_evoh60_pet20: status=solved, frontier_points=7" in progress

    def test_completion_progress_anchor_includes_visualization_artifacts_after_optimization(self):
        from strap.routing import _build_progress_directive

        messages = [
            HumanMessage(content="Run Pareto slices and plot them."),
            AIMessage(content="", tool_calls=[_task_call("tc_opt_slices", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.0","analysis_type":"pareto_slices",'
                    '"x_metric":"total_cost","y_metric":"circularity","n_slices_requested":2,"n_slices_solved":2,'
                    '"n_points_requested_per_slice":100,"pareto_slices_payload_path":"/tmp/pareto_slices.json",'
                    '"slices":[{"slice_id":"slice_1","label":"ldpe20_evoh60_pet20","status":"solved","n_points_feasible":7,"max_circularity":0.82}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt_slices",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz_slices", "visualization-specialist")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"visualization-specialist","schema_version":"1.0","plot_type":"optimization_pareto_slices",'
                    '"plot_paths":["/tmp/combined.png","/tmp/slice1.png"],"format":"png"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_viz_slices",
            ),
        ]
        ordered_plan = [
            {"subagent": "optimization-engineer", "description": "opt", "step_id": "tc_opt_slices"},
            {"subagent": "visualization-specialist", "description": "viz", "step_id": "tc_viz_slices"},
        ]

        progress = _build_progress_directive(messages, {"tc_opt_slices", "tc_viz_slices"}, ordered_plan)

        assert "multi-composition Pareto-slice optimization result" in progress
        assert "Visualization-specialist returned validated optimization_pareto_slices artifacts" in progress
        assert "Validated plot artifact: /tmp/combined.png." in progress
        assert "Validated plot artifact: /tmp/slice1.png." in progress

    def test_invalid_handoff_without_scope_counts_as_failed(self):
        from strap.routing import (
            _build_progress_directive,
            _extract_completed_subagents,
            _extract_failed_subagent_calls,
            _get_ordered_plan,
        )

        invalid = ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"biosteam-analyst","schema_version":"1.0","target_plastic":"LDPE"}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_tea",
        )
        messages = [
            HumanMessage(content="run separation then tea"),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[_task_call("tc_tea", "biosteam-analyst")]),
            invalid,
        ]

        completed = _extract_completed_subagents(messages)
        failed = _extract_failed_subagent_calls(messages)
        plan = _get_ordered_plan(messages)
        progress = _build_progress_directive(
            messages,
            {"tc_sep"},
            plan,
            failed_ids={"tc_tea"},
        )

        assert completed == ["separation-engineer"]
        assert [step["subagent"] for step in failed] == ["biosteam-analyst"]
        assert "Failed subagents: biosteam-analyst" in progress

    def test_successful_retry_supersedes_earlier_failed_attempt(self):
        from strap.routing import (
            _build_progress_directive,
            _get_active_remaining_steps,
            _get_effective_completed_task_ids,
            _get_effective_failed_task_ids,
            _get_ordered_plan,
            _get_pending_required_handoff,
        )

        messages = [
            HumanMessage(content="run separation then tea"),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_1", "separation-engineer")]),
            ToolMessage(content="Only prose, no structured result.", tool_call_id="tc_sep_1"),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_2", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep_2"),
        ]
        allowed_rules = [
            {"subagent": "separation-engineer", "description": "sep"},
            {"subagent": "biosteam-analyst", "description": "tea"},
        ]

        plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
        completed_ids = _get_effective_completed_task_ids(messages)
        failed_ids = _get_effective_failed_task_ids(messages)
        remaining = _get_active_remaining_steps(messages, plan)
        progress = _build_progress_directive(messages, completed_ids, plan, failed_ids=failed_ids)
        pending_handoff = _get_pending_required_handoff(messages, allowed_rules)

        assert completed_ids == {"tc_sep_2"}
        assert failed_ids == set()
        assert [step["subagent"] for step in remaining] == ["biosteam-analyst"]
        assert "Failed subagents: separation-engineer" not in progress
        assert pending_handoff == ("separation-engineer", "biosteam-analyst")

    def test_consecutive_failed_retries_collapse_to_latest_active_attempt(self):
        from strap.routing import (
            _build_progress_directive,
            _get_active_remaining_steps,
            _get_effective_failed_task_ids,
            _get_ordered_plan,
        )

        messages = [
            HumanMessage(content="run separation then viz"),
            AIMessage(content="", tool_calls=[_task_call(
                "tc_sep_1",
                "separation-engineer",
                "Analyze the separation request and propose the sequence.",
            )]),
            ToolMessage(content="Only prose, no structured result.", tool_call_id="tc_sep_1"),
            AIMessage(content="", tool_calls=[_task_call(
                "tc_sep_2",
                "separation-engineer",
                "Analyze the same separation request and return the best scheme.",
            )]),
            ToolMessage(content="Still no structured result.", tool_call_id="tc_sep_2"),
        ]
        allowed_rules = [
            {"subagent": "separation-engineer", "description": "sep"},
            {"subagent": "visualization-specialist", "description": "viz"},
        ]

        plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
        failed_ids = _get_effective_failed_task_ids(messages)
        remaining = _get_active_remaining_steps(messages, plan)
        progress = _build_progress_directive(messages, set(), plan, failed_ids=failed_ids)

        assert failed_ids == {"tc_sep_2"}
        assert [step["step_id"] for step in remaining] == ["tc_sep_2", "advisory:visualization-specialist"]
        assert "Failed subagents: separation-engineer, separation-engineer" not in progress
        assert "Failed subagents: separation-engineer" in progress

    def test_write_todos_allowed_before_progress_activation(self):
        from strap.routing import _build_write_todos_guard_messages

        messages = [
            HumanMessage(content="plan the workflow"),
            AIMessage(content="", tool_calls=[{
                "id": "todo1",
                "name": "write_todos",
                "args": {"todos": [{"content": "Plan workflow", "status": "in_progress"}]},
            }]),
        ]

        assert _build_write_todos_guard_messages(messages) == []

    def test_write_todos_blocked_before_first_dispatch_for_multi_specialist_route(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find the safest solvent that selectively dissolves PS over PVC."),
            AIMessage(content="", tool_calls=[{
                "id": "todo_parallel",
                "name": "write_todos",
                "args": {"todos": [{"content": "Plan parallel specialist work", "status": "in_progress"}]},
            }]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "safety-analyst")
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "todo_parallel"
        assert "disabled before the first specialist dispatch" in update["messages"][0].content
        assert "separation-engineer" in update["messages"][0].content
        assert "safety-analyst" in update["messages"][0].content

    def test_ready_handoff_suppressed_after_failed_downstream_attempt(self):
        from strap.routing import _get_ready_downstream_handoff

        messages = [
            HumanMessage(content="run separation then visualize"),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"handoff_id": "h_sep_viz", '
                    '"producer": "separation-engineer", "consumer": "visualization-specialist", '
                    '"status": "ok", "task_prompt": "Create the plot."}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(content="Only prose, no structured result.", tool_call_id="tc_viz"),
        ]

        handoff = _get_ready_downstream_handoff(
            messages,
            allowed_rules=[
                {"subagent": "separation-engineer", "description": "sep"},
                {"subagent": "visualization-specialist", "description": "viz"},
            ],
        )

        assert handoff is None

    def test_downstream_join_task_blocked_until_all_required_handoffs_exist(self):
        from strap.routing_guards import _validate_task_tool_call

        query = (
            "Do a literature search and patent search for multilayer polymer recycling methods, "
            "answer the question with RAG."
        )
        allowed_rules = [
            {"subagent": "scholar-researcher", "description": "literature"},
            {"subagent": "patent-researcher", "description": "patents"},
            {"subagent": "rag-analyst", "description": "rag"},
        ]
        messages = [
            HumanMessage(content=query),
            AIMessage(content="", tool_calls=[_task_call("tc_scholar", "scholar-researcher")]),
            ToolMessage(content=_structured_result_content("scholar-researcher"), tool_call_id="tc_scholar"),
            AIMessage(content="", tool_calls=[_task_call("tc_patent", "patent-researcher")]),
            ToolMessage(content=_structured_result_content("patent-researcher"), tool_call_id="tc_patent"),
            AIMessage(content="", tool_calls=[{
                "id": "bh_scholar",
                "name": "build_handoff",
                "args": {"consumer": "rag-analyst", "producer": "scholar-researcher"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"handoff_id": "h_scholar_rag", '
                    '"producer": "scholar-researcher", "consumer": "rag-analyst", '
                    '"contract": "literature_context.v1", "status": "ok"}}'
                ),
                tool_call_id="bh_scholar",
            ),
        ]

        validation = _validate_task_tool_call(
            {
                "id": "tc_rag",
                "name": "task",
                "args": {"subagent_type": "rag-analyst", "description": "Use both upstream results."},
            },
            messages,
            allowed_rules,
        )

        assert validation is not None
        assert 'build_handoff(consumer="rag-analyst", producer="patent-researcher")' in validation

    def test_write_todos_blocked_once_progress_is_active(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="run stats then continue"),
            AIMessage(content="", tool_calls=[_task_call("tc1", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[{
                "id": "todo2",
                "name": "write_todos",
                "args": {"todos": [{"content": "Update plan", "status": "in_progress"}]},
            }]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert len(update["messages"]) == 1
        assert update["messages"][0].tool_call_id == "todo2"
        assert update["messages"][0].status == "error"
        assert "disabled once subagent progress tracking is active" in update["messages"][0].content

    def test_malformed_write_todos_is_blocked_before_tool_execution(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="plan the workflow"),
            AIMessage(content="", tool_calls=[{
                "id": "todo3",
                "name": "write_todos",
                "args": {"todos": [{"status": "completed"}]},
            }]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert len(update["messages"]) == 1
        assert update["messages"][0].tool_call_id == "todo3"
        assert update["messages"][0].status == "error"
        assert "missing a non-empty `content` field" in update["messages"][0].content

    def test_subagent_free_tools_bill_filesystem_exploration(self):
        from strap.agent import _ALWAYS_FREE_TOOLS

        assert "read_file" in _ALWAYS_FREE_TOOLS
        assert "write_file" in _ALWAYS_FREE_TOOLS
        assert "write_todos" in _ALWAYS_FREE_TOOLS
        assert "ls" not in _ALWAYS_FREE_TOOLS
        assert "glob" not in _ALWAYS_FREE_TOOLS
        assert "grep" not in _ALWAYS_FREE_TOOLS
        assert "edit_file" not in _ALWAYS_FREE_TOOLS
        assert "execute" not in _ALWAYS_FREE_TOOLS

    def test_subagent_guard_blocks_grep_for_handoff_search(self):
        from strap.guardrails import SubagentGuardMiddleware

        middleware = SubagentGuardMiddleware(agent_name="separation-engineer")
        request = ToolCallRequest(
            tool_call=_fs_call("grep_handoff", "grep", pattern="h_123", path="/home/aaltamimi2"),
            tool=None,
            state={"messages": [HumanMessage(content="Use the upstream handoff.")]},
            runtime=MagicMock(),
        )
        handler = MagicMock()

        result = middleware.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "grep_handoff"
        assert result.status == "error"
        assert "Subagent guard" in result.content
        assert "handoff payload" in result.content

    def test_ls_blocked_without_user_request_or_path_context(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Which solvent dissolves EVOH but not PE?"),
            AIMessage(content="", tool_calls=[_fs_call("ls1", "ls", path="/plots")]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert len(update["messages"]) == 1
        assert update["messages"][0].tool_call_id == "ls1"
        assert update["messages"][0].status == "error"
        assert "prior tool returned a concrete path" in update["messages"][0].content

    def test_ls_allowed_when_following_returned_artifact_path(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Create a separation tree plot for PE,PS,PVC."),
            AIMessage(content="", tool_calls=[_task_call("tc_plot", "visualization-specialist")]),
            ToolMessage(content="Plot saved: `/plots/separation_tree.png`", tool_call_id="tc_plot"),
            AIMessage(content="", tool_calls=[_fs_call("ls2", "ls", path="/plots")]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is None

    def test_ls_allowed_for_default_artifact_dir_after_file_producer(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Create a separation tree plot for PE,PS,PVC."),
            AIMessage(content="", tool_calls=[_task_call("tc_plot", "visualization-specialist")]),
            ToolMessage(content="Plot created successfully.", tool_call_id="tc_plot"),
            AIMessage(content="", tool_calls=[_fs_call("ls3", "ls", path="/plots")]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is None

    def test_execute_stays_blocked_without_explicit_file_request(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Create a separation tree plot for PE,PS,PVC."),
            AIMessage(content="", tool_calls=[_task_call("tc_plot", "visualization-specialist")]),
            ToolMessage(content="Plot saved: `/plots/separation_tree.png`", tool_call_id="tc_plot"),
            AIMessage(content="", tool_calls=[_fs_call("exec1", "execute", command="ls /plots")]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert len(update["messages"]) == 1
        assert update["messages"][0].tool_call_id == "exec1"
        assert update["messages"][0].status == "error"
        assert "disabled at the orchestrator layer" in update["messages"][0].content

    def test_grep_allowed_when_user_explicitly_requests_file_inspection(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Inspect the local files in /plots and grep for scenario labels."),
            AIMessage(content="", tool_calls=[_fs_call("grep1", "grep", pattern="scenario", path="/plots")]),
        ]

        middleware = RoutingMiddleware()
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is None

    def test_get_subagent_result_blocked_after_successful_sequential_step(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then run a techno-economic analysis."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "gsr1",
                "name": "get_subagent_result",
                "args": {"agent_name": "separation-engineer"},
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer", "biosteam-analyst"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "gsr1"
        assert "fallback-only" in update["messages"][0].content
        assert "build_handoff" in update["messages"][0].content

    def test_get_subagent_result_blocked_after_routed_workflow_is_complete(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then create a visualization."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"producer": "separation-engineer", '
                    '"consumer": "visualization-specialist", "status": "ok", '
                    '"task_prompt": "Create the requested visualization."}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz_1", "visualization-specialist")]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz_1"),
            AIMessage(content="", tool_calls=[{
                "id": "gsr_done",
                "name": "get_subagent_result",
                "args": {"agent_name": "separation-engineer"},
            }]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "visualization-specialist")
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "gsr_done"
        assert "All routed specialists already completed successfully" in update["messages"][0].content
        assert "Synthesize the final answer" in update["messages"][0].content

    def test_get_all_subagent_results_blocked_after_routed_workflow_is_complete(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then create a visualization."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"producer": "separation-engineer", '
                    '"consumer": "visualization-specialist", "status": "ok", '
                    '"task_prompt": "Create the requested visualization."}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz_1", "visualization-specialist")]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz_1"),
            AIMessage(content="", tool_calls=[{
                "id": "gar_done",
                "name": "get_all_subagent_results",
                "args": {},
            }]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "visualization-specialist")
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "gar_done"
        assert "All routed specialists already completed successfully" in update["messages"][0].content

    def test_downstream_task_blocked_until_build_handoff_exists(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then run a techno-economic analysis."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "tc_bio",
                "name": "task",
                "args": {
                    "subagent_type": "biosteam-analyst",
                    "description": "Run BioSTEAM for the best separation sequence.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer", "biosteam-analyst"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_bio"
        assert "build_handoff" in update["messages"][0].content

    def test_downstream_task_allowed_after_build_handoff(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then run a techno-economic analysis."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "biosteam-analyst", "producer": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"producer": "separation-engineer", '
                    '"consumer": "biosteam-analyst", "status": "ok"}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[{
                "id": "tc_bio",
                "name": "task",
                "args": {
                    "subagent_type": "biosteam-analyst",
                    "description": "Run BioSTEAM for the best separation sequence.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer", "biosteam-analyst"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is None

    def test_task_blocked_after_routed_workflow_is_complete(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then create a visualization."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"producer": "separation-engineer", '
                    '"consumer": "visualization-specialist", "status": "ok", '
                    '"task_prompt": "Create the requested visualization."}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz_1", "visualization-specialist")]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz_1"),
            AIMessage(content="", tool_calls=[{
                "id": "tc_viz_2",
                "name": "task",
                "args": {
                    "subagent_type": "visualization-specialist",
                    "description": "Create a selectivity heatmap too.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "visualization-specialist")
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_viz_2"
        assert "All routed specialists already completed successfully" in update["messages"][0].content

    def test_ready_handoff_blocks_non_consumer_task(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find a separation sequence and then run a techno-economic analysis."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "biosteam-analyst", "producer": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"producer": "separation-engineer", '
                    '"consumer": "biosteam-analyst", "status": "ok"}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[{
                "id": "tc_sep_repeat",
                "name": "task",
                "args": {
                    "subagent_type": "separation-engineer",
                    "description": "Search for another separation option.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer", "biosteam-analyst"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_sep_repeat"
        assert "already completed successfully" in update["messages"][0].content
        assert "biosteam-analyst" in update["messages"][0].content

    def test_ready_handoff_blocks_non_task_orchestrator_tools(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[_task_call("tc_stats", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc_stats"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "statistics-ml"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"producer": "statistics-ml", '
                    '"consumer": "visualization-specialist", "status": "ok"}}'
                ),
                tool_call_id="bh1",
            ),
            AIMessage(content="", tool_calls=[{
                "id": "lsafe1",
                "name": "list_handoffs",
                "args": {"producer": "statistics-ml"},
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "visualization-specialist"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "lsafe1"
        assert "validated handoff for `visualization-specialist` is already available" in update["messages"][0].content

    def test_unexpected_subagent_is_blocked_outside_allowed_route(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[_task_call("tc_stats", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc_stats"),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz"),
            AIMessage(content="", tool_calls=[{
                "id": "tc_scholar",
                "name": "task",
                "args": {
                    "subagent_type": "scholar-researcher",
                    "description": "Search the literature for polycarbonate solvents.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "visualization-specialist"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_scholar"
        assert "outside the active routed specialist set" in update["messages"][0].content

    def test_repeat_task_blocked_after_route_completion_without_new_scope(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[_task_call("tc_stats", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc_stats"),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz"),
            AIMessage(content="", tool_calls=[{
                "id": "tc_viz_repeat",
                "name": "task",
                "args": {
                    "subagent_type": "visualization-specialist",
                    "description": "Plot solubility curves for polycarbonate.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "visualization-specialist"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_viz_repeat"
        assert "already completed successfully" in update["messages"][0].content

    def test_repeat_task_blocked_when_visualization_runs_before_required_stats_step(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[{
                "id": "tc_viz",
                "name": "task",
                "args": {
                    "subagent_type": "visualization-specialist",
                    "description": "Plot solubility curves for polycarbonate.",
                },
            }]),
            ToolMessage(content=_structured_result_content("visualization-specialist"), tool_call_id="tc_viz"),
            AIMessage(content="", tool_calls=[{
                "id": "tc_viz_repeat",
                "name": "task",
                "args": {
                    "subagent_type": "visualization-specialist",
                    "description": "Create a precipitation dashboard for polycarbonate.",
                },
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("visualization-specialist"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_viz_repeat"
        assert "downstream of statistics-ml" in update["messages"][0].content

    def test_direct_domain_tool_blocked_when_multi_agent_route_is_active(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Separate PS from PVC, compare safety, and create a dashboard."),
            AIMessage(content="", tool_calls=[{
                "id": "rank1",
                "name": "rank_solvents_selectivity",
                "args": {"target_polymer": "PS", "other_polymer": "PVC"},
            }]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for(
                "separation-engineer",
                "safety-analyst",
                "visualization-specialist",
            )
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "rank1"
        assert "disabled at the orchestrator layer" in update["messages"][0].content

    def test_direct_separation_tool_blocked_before_first_task_for_single_specialist_route(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Find the best way to separate PS from PVC below 90C."),
            AIMessage(content="", tool_calls=[{
                "id": "rank1",
                "name": "rank_solvents_selectivity",
                "args": {"target_polymer": "PS", "other_polymer": "PVC"},
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "rank1"
        assert "belongs to the routed `separation-engineer` workflow" in update["messages"][0].content
        assert 'task(subagent_type="separation-engineer")' in update["messages"][0].content

    def test_wrap_model_call_autodispatches_ready_handoff(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[_task_call("tc_stats", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc_stats"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "statistics-ml"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"handoff_id": "h_stats_viz", '
                    '"producer": "statistics-ml", "consumer": "visualization-specialist", '
                    '"status": "ok", '
                    '"task_prompt": "Create a visualization for this statistics/ML result using the provided analysis summary."}}'
                ),
                tool_call_id="bh1",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        call_count = 0

        def handler(_request):
            nonlocal call_count
            call_count += 1
            return MagicMock(result=[AIMessage(content="premature final answer")])

        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "visualization-specialist"))
        response = middleware.wrap_model_call(request, handler)

        assert call_count == 1
        assert response.result[0].tool_calls[0]["args"]["subagent_type"] == "visualization-specialist"
        assert response.result[0].tool_calls[0]["args"]["handoff_id"] == "h_stats_viz"
        assert (
            response.result[0].tool_calls[0]["args"]["description"]
            == "Create a visualization for this statistics/ML result using the provided analysis summary."
        )
        assert request.system_message.content == "base system"

    def test_wrap_model_call_overrides_wrong_task_when_ready_handoff_exists(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[_task_call("tc_stats", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc_stats"),
            AIMessage(content="", tool_calls=[{
                "id": "bh1",
                "name": "build_handoff",
                "args": {"consumer": "visualization-specialist", "producer": "statistics-ml"},
            }]),
            ToolMessage(
                content=(
                    '{"ok": true, "handoff": {"handoff_id": "h_stats_viz", "producer": "statistics-ml", '
                    '"consumer": "visualization-specialist", "status": "ok", '
                    '"task_prompt": "Create a visualization for this statistics/ML result using the provided analysis summary."}}'
                ),
                tool_call_id="bh1",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))

        def handler(_request):
            return MagicMock(result=[AIMessage(content="", tool_calls=[_task_call("tc_wrong", "scholar-researcher")])])

        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "visualization-specialist"))
        response = middleware.wrap_model_call(request, handler)

        assert response.result[0].tool_calls[0]["name"] == "task"
        assert response.result[0].tool_calls[0]["args"]["subagent_type"] == "visualization-specialist"
        assert response.result[0].tool_calls[0]["args"]["handoff_id"] == "h_stats_viz"
        assert response.result[0].tool_calls[0]["id"].startswith("route_task_h_stats_viz_")

    def test_wrap_model_call_autobuilds_pending_handoff(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(content="Look up Tg for polycarbonate and then plot solubility curves."),
            AIMessage(content="", tool_calls=[_task_call("tc_stats", "statistics-ml")]),
            ToolMessage(content=_structured_result_content("statistics-ml"), tool_call_id="tc_stats"),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        call_count = 0

        def handler(_request):
            nonlocal call_count
            call_count += 1
            return MagicMock(result=[AIMessage(content="premature final answer")])

        middleware = RoutingMiddleware(classifier_model=_classifier_for("statistics-ml", "visualization-specialist"))
        response = middleware.wrap_model_call(request, handler)

        assert call_count == 1
        assert response.result[0].tool_calls[0]["name"] == "build_handoff"
        assert response.result[0].tool_calls[0]["args"]["consumer"] == "visualization-specialist"
        assert response.result[0].tool_calls[0]["args"]["producer"] == "statistics-ml"


class TestRoutingNormalization:
    def test_wrap_model_call_injects_initial_advisory_hint(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        request = _Request(
            [HumanMessage(content="Find a separation sequence for PS and PET.")],
            SystemMessage(content="base system"),
        )

        captured = {}

        def handler(patched_request):
            captured["system"] = patched_request.system_message.content
            return MagicMock(result=[AIMessage(content="ok")])

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        response = middleware.wrap_model_call(request, handler)
        system_text = str(captured["system"])

        assert response.result[0].content == "ok"
        assert "separation-engineer" in system_text
        assert "ADVISORY" in system_text
        assert "answer directly using your own tools" not in system_text

    def test_hsp_only_queries_prefer_statistics_ml_over_separation_engineer(self):
        from strap.routing import _normalize_matched_rules

        matched = [
            {"subagent": "statistics-ml", "description": "stats"},
            {"subagent": "separation-engineer", "description": "separation"},
        ]

        normalized = _normalize_matched_rules(
            "Evaluate using Hansen solubility parameters and compare RED values for DMSO with EVOH and PE.",
            matched,
        )

        assert [rule["subagent"] for rule in normalized] == ["statistics-ml"]

    def test_process_design_queries_keep_separation_engineer(self):
        from strap.routing import _normalize_matched_rules

        matched = [
            {"subagent": "statistics-ml", "description": "stats"},
            {"subagent": "separation-engineer", "description": "separation"},
        ]

        normalized = _normalize_matched_rules(
            "Use Hansen solubility parameters to screen 16 polymers against 10 solvents, build a selectivity matrix, and propose a room-temperature separation sequence.",
            matched,
        )

        assert [rule["subagent"] for rule in normalized] == [
            "statistics-ml",
            "separation-engineer",
        ]

    def test_process_design_queries_drop_statistics_ml_without_explicit_screening_deliverable(self):
        from strap.routing import _normalize_matched_rules

        matched = [
            {"subagent": "statistics-ml", "description": "stats"},
            {"subagent": "separation-engineer", "description": "separation"},
            {"subagent": "visualization-specialist", "description": "viz"},
        ]

        normalized = _normalize_matched_rules(
            "Find the optimal separation sequence for PS, PMMA, and PET at up to 120C, then create a selectivity heatmap showing the results.",
            matched,
        )

        assert [rule["subagent"] for rule in normalized] == [
            "separation-engineer",
            "visualization-specialist",
        ]

    def test_process_design_queries_drop_biosteam_without_explicit_tea_or_lca_intent(self):
        from strap.routing import _normalize_matched_rules

        matched = [
            {"subagent": "separation-engineer", "description": "separation"},
            {"subagent": "biosteam-analyst", "description": "biosteam"},
        ]

        normalized = _normalize_matched_rules(
            "Only do process design. At room temperature, what is the best dissolution-based separation sequence for PETG, PC, and PS? If the route is weak or non-selective, say so clearly.",
            matched,
        )

        assert [rule["subagent"] for rule in normalized] == ["separation-engineer"]

    def test_process_only_queries_drop_safety_without_explicit_safety_data_intent(self):
        from strap.routing import _normalize_matched_rules

        matched = [
            {"subagent": "separation-engineer", "description": "separation"},
            {"subagent": "safety-analyst", "description": "safety"},
        ]

        normalized = _normalize_matched_rules(
            "Only do process design. For separating EVOH from PE at atmospheric pressure and no higher than 120C, compare DMSO and DMF as candidate process solvents and recommend the safer executable operating window only if it is defensible.",
            matched,
        )

        assert [rule["subagent"] for rule in normalized] == ["separation-engineer"]

    def test_parallel_hint_with_separation_engineer_omits_direct_tool_advice(self):
        from strap.routing import _build_hint_from_matches

        hint = _build_hint_from_matches([
            {"subagent": "separation-engineer", "description": "separation"},
            {"subagent": "safety-analyst", "description": "safety"},
        ])

        assert hint is not None
        assert "rank_solvents_selectivity" not in hint


class TestSeparationRoutePurityAndFallback:
    def test_visualization_specialist_blocked_for_separation_only_query(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(
                content=(
                    "Only do process design. For a multilayer EVOH/LDPE/PET film, propose the best "
                    "atmospheric-pressure separation sequence up to 140C and highlight any boiling-point "
                    "or unsupported-data limits."
                )
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz_only", "visualization-specialist")]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "visualization-specialist")
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_viz_only"
        assert "did not explicitly request a plot" in update["messages"][0].content

    def test_build_handoff_to_visualization_blocked_for_separation_only_query(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(
                content="Only do process design. Find the best separation sequence for PS, PET, and PC up to 120C at 1 atm."
            ),
            AIMessage(content="", tool_calls=[{
                "id": "bh_viz_only",
                "name": "build_handoff",
                "args": {"producer": "separation-engineer", "consumer": "visualization-specialist"},
            }]),
        ]

        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "visualization-specialist")
        )
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "bh_viz_only"
        assert "did not explicitly request a visualization" in update["messages"][0].content

    def test_visualization_tool_blocked_for_separation_only_query(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(
                content="Only do process design. Find the best separation sequence for PS and PET up to 120C."
            ),
            AIMessage(content="", tool_calls=[_fs_call("plot_sep_only", "create_separation_tree_plot", polymers="PS,PET")]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "plot_sep_only"
        assert "process design only" in update["messages"][0].content

    def test_listing_tool_blocked_before_single_specialist_separation_dispatch(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(
                content=(
                    "Only do process design. At room temperature, what is the best dissolution-based "
                    "separation sequence for PETG, PC, and PS? If the route is weak or non-selective, say so clearly."
                )
            ),
            AIMessage(content="", tool_calls=[_fs_call("list_poly", "list_available_polymers")]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "list_poly"
        assert "Dispatch `task(subagent_type=\"separation-engineer\")` first" in update["messages"][0].content

    def test_wrap_model_call_short_circuits_from_completed_single_specialist_output(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Only do process design. Find the best separation sequence for PS and PET up to 120C."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_done", "separation-engineer")]),
            ToolMessage(
                content=(
                    "Use Toluene at 75C to recover PS, then filter PET.\n"
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                    '"best_sequence":["PS","PET"],'
                    '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                    '"solvent_mapping":{"PS":"Toluene"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep_done",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        assert response.result[0].content == "Use Toluene at 75C to recover PS, then filter PET."
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_prose"

    def test_wrap_model_call_short_circuits_to_payload_fallback_when_task_prose_is_empty(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Only do process design. Find the best separation sequence for PS and PET up to 120C at 1 atm."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_payload", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                    '"best_sequence":["PS","PET"],'
                    '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                    '"solvent_mapping":{"PS":"Toluene"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep_payload",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "Recommended separation sequence" in content
        assert "Toluene" in content
        assert "75.0" in content
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_separation_fallback"

    def test_wrap_model_call_uses_biosteam_payload_fallback_when_prose_omits_tci_and_aoc(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Only do TEA/LCA. Run a batch BioSTEAM comparison across all PE solvents under C1 and report the top five."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_bio_batch", "biosteam-analyst")]),
            ToolMessage(
                content=(
                    "Top solvents by MSP are Toluene and Heptane.\n"
                    "<STRUCTURED_RESULT>"
                    '{"agent":"biosteam-analyst","schema_version":"1.0","target_plastic":"PE","energy_case":"C1",'
                    '"results":['
                    '{"scenario_label":"Toluene","success":true,"tea":{"msp_usd_per_kg":1.02,"tci_usd":12000000,"aoc_usd_per_yr":5100000},"lca":{"gwp_kg_co2e_per_kg":0.97}},'
                    '{"scenario_label":"Heptane","success":true,"tea":{"msp_usd_per_kg":1.08,"tci_usd":12500000,"aoc_usd_per_yr":5300000},"lca":{"gwp_kg_co2e_per_kg":0.90}}'
                    '],"n_simulations":2,"n_failed":0}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_bio_batch",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("biosteam-analyst"))
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "Top scenarios by MSP" in content
        assert "1. Toluene: MSP $1.02/kg; GWP 0.97 kg CO2e/kg; TCI $12.00M; AOC $5.10M." in content
        assert "2. Heptane: MSP $1.08/kg; GWP 0.90 kg CO2e/kg; TCI $12.50M; AOC $5.30M." in content
        assert "TCI" in content
        assert "AOC" in content
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_biosteam_fallback"

    def test_wrap_model_call_uses_biosteam_payload_fallback_for_flat_result_rows(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Only do TEA/LCA. Run a batch BioSTEAM comparison across all PE solvents under C1 and report the top five."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_bio_batch_flat", "biosteam-analyst")]),
            ToolMessage(
                content=(
                    "Top solvents by MSP are Toluene and Heptane.\n"
                    "<STRUCTURED_RESULT>"
                    '{"agent":"biosteam-analyst","schema_version":"1.0","target_plastic":"PE","energy_case":"C1",'
                    '"results":['
                    '{"solvent":"Toluene","msp_usd_per_kg":1.02,"tci_usd":12000000,"aoc_usd_per_yr":5100000,"gwp_kg_co2e_per_kg":0.97},'
                    '{"solvent":"Heptane","msp_usd_per_kg":1.08,"tci_usd":12500000,"aoc_usd_per_yr":5300000,"gwp_kg_co2e_per_kg":0.90}'
                    '],"n_simulations":2,"n_failed":0}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_bio_batch_flat",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("biosteam-analyst"))
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "1. Toluene: MSP $1.02/kg; GWP 0.97 kg CO2e/kg; TCI $12.00M; AOC $5.10M." in content
        assert "2. Heptane: MSP $1.08/kg; GWP 0.90 kg CO2e/kg; TCI $12.50M; AOC $5.30M." in content
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_biosteam_fallback"

    def test_wrap_model_call_uses_contaminant_payload_fallback_when_prose_omits_logd_and_miscibility(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content=(
                    "Only do contaminant-removal screening. Compare leaching versus STRAP contaminant removal "
                    "for DBP from EVOH in the presence of PE."
                )
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_contam", "contaminant-removal-analyst")]),
            ToolMessage(
                content=(
                    "STRAP is recommended because it has viable solvents.\n"
                    "<STRUCTURED_RESULT>"
                    '{"agent":"contaminant-removal-analyst","schema_version":"1.0","mode":"comparison",'
                    '"target_polymer":"EVOH","other_polymers":["LDPE"],'
                    '"contaminants":["di-n-butyl phthalate (DBP)"],'
                    '"supported_contaminants":["di-n-butyl phthalate (DBP)"],'
                    '"unsupported_contaminants":[],"recommended_mode":"strap_contaminant_removal",'
                    '"modes":{"strap_contaminant_removal":{"candidate_solvents":[{"solvent":"dimethyl sulfoxide","passes":true,"operating_temperature_c":90.0,"contaminant_logd_min":0.31}]},'
                    '"leaching":{"candidate_solvents":[]}}}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_contam",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("contaminant-removal-analyst"))
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content.lower()
        assert "miscible" in content
        assert "logd" in content
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_contaminant_fallback"

    def test_wrap_model_call_short_circuits_completed_separation_contaminant_route(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content=(
                    "First do process design for separation, then contaminant screening. "
                    "For an EVOH/PE multilayer contaminated with DBP, identify the best route "
                    "and then compare leaching versus STRAP contaminant removal."
                )
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["EVOH","PE"],'
                    '"best_sequence":["EVOH","PE"],'
                    '"steps":[{"step":1,"polymer":"EVOH","solvent":"dimethyl sulfoxide","temperature_c":72.5}],'
                    '"top_solvents":["dimethyl sulfoxide","isopropylamine"],'
                    '"solvent_mapping":{"EVOH":"dimethyl sulfoxide"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["EVOH","PE"],"solvent_mapping":{"EVOH":"dimethyl sulfoxide"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_contam_done", "contaminant-removal-analyst")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"contaminant-removal-analyst","schema_version":"1.0","mode":"comparison",'
                    '"target_polymer":"EVOH","other_polymers":["PE"],'
                    '"contaminants":["di-n-butyl phthalate (DBP)"],'
                    '"supported_contaminants":["di-n-butyl phthalate (DBP)"],'
                    '"unsupported_contaminants":[],"recommended_mode":"strap_contaminant_removal",'
                    '"modes":{"strap_contaminant_removal":{"recommended_solvents":["dimethyl sulfoxide"],'
                    '"candidate_solvents":[{"solvent":"dimethyl sulfoxide","passes":true,"operating_temperature_c":105.0,'
                    '"precipitation_temperature_c":0.0,"boiling_point_c":189.0,"contaminant_logd_min":0.31}]},'
                    '"leaching":{"recommended_solvents":[],"candidate_solvents":[]}}}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_contam_done",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "contaminant-removal-analyst")
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content.lower()
        assert "miscible" in content
        assert "logd" in content
        assert "experimental" in content
        assert "dimethyl sulfoxide" in content
        assert (
            response.result[0].additional_kwargs["strap_origin"]
            == "routing_multi_specialist_separation_contaminant_fallback"
        )

    def test_wrap_model_call_contaminant_fallback_does_not_say_preferred_comparison_for_unsupported_contaminant(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content=(
                    "First do process design for separation, then contaminant screening for an EVOH/PE film "
                    "with DEHP contamination."
                )
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["EVOH","PE"],'
                    '"best_sequence":["EVOH"],'
                    '"steps":[{"step":1,"polymer":"EVOH","solvent":"Isopropylamine","temperature_c":72.5}],'
                    '"top_solvents":["Isopropylamine"],'
                    '"solvent_mapping":{"EVOH":"Isopropylamine"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["EVOH"],"solvent_mapping":{"EVOH":"Isopropylamine"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_contam", "contaminant-removal-analyst")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"contaminant-removal-analyst","schema_version":"1.0","no_data":true,'
                    '"mode":"comparison","target_polymer":"EVOH","other_polymers":["PE"],'
                    '"contaminants":["di-(2-ethylhexyl) phthalate (DEHP)"],'
                    '"supported_contaminants":[],"unsupported_contaminants":["di-(2-ethylhexyl) phthalate (DEHP)"],'
                    '"candidate_solvents":[{"solvent":"Isopropylamine"}],"recommended_solvents":[],'
                    '"decision_basis":["unsupported contaminant"],'
                    '"caveats":["requested contaminant is unsupported"]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_contam",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "contaminant-removal-analyst")
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content.lower()
        assert "preferred comparison" not in content
        assert "unsupported" in content
        assert "no supported contaminant-screening mode result is available" in content

    def test_wrap_model_call_optimization_fallback_uses_only_optimization_payload(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content=(
                    "For an LDPE/EVOH/PET film, use the top separation routes as candidates, "
                    "run route-constrained optimization, and summarize the Pareto result."
                )
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["LDPE","EVOH","PET"],'
                    '"best_sequence":["LDPE","EVOH","PET"],'
                    '"steps":[{"step":1,"polymer":"LDPE","solvent":"Cyclohexane","temperature_c":120.0},'
                    '{"step":2,"polymer":"EVOH","solvent":"Methanol","temperature_c":55.0}],'
                    '"solvent_mapping":{"LDPE":"Cyclohexane","EVOH":"Methanol"},'
                    '"top_k_sequences":[{"rank":1,"sequence":["LDPE","EVOH","PET"],"solvent_mapping":{"LDPE":"Cyclohexane","EVOH":"Methanol"}}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.3","analysis_type":"pareto_front",'
                    '"x_metric":"total_cost","y_metric":"emissions","n_points_feasible":1,'
                    '"n_routes_requested":2,"n_routes_solved":1,'
                    '"points":[{"point_id":1,"route_id":"route_1","total_cost":36635.0,"emissions":7110.0}],'
                    '"route_reports":[{"route_id":"route_1","status":"solved","polymer_solvent_map":{"PE":"Cyclohexane","EVOH":"Dimethyl sulfoxide"}},{"route_id":"route_2","status":"infeasible","reason":"cost-anchor solve failed under route enforcement","polymer_solvent_map":{"PE":"Cyclohexane","EVOH":"Methanol"}}],'
                    '"solvent_filter_warnings":[]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("separation-engineer", "optimization-engineer")
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "route-constrained Pareto front" in content
        assert "Route reports:" in content
        assert "route_1: solved | PE-Cyclohexane, EVOH-Dimethyl sulfoxide" in content
        assert "route_2: infeasible | PE-Cyclohexane, EVOH-Methanol" in content
        assert "Pareto points:" in content
        assert "Step 1" not in content
        assert "Predicted separation order" not in content
        assert (
            response.result[0].additional_kwargs["strap_origin"]
            == "routing_multi_specialist_separation_optimization_fallback"
        )

    def test_wrap_model_call_optimization_visualization_fallback_uses_plot_and_optimization_payloads(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Run optimization and generate a Pareto front plot for total cost vs emissions."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.3","analysis_type":"pareto_front",'
                    '"x_metric":"total_cost","y_metric":"emissions","n_points_feasible":1,'
                    '"n_routes_requested":1,"n_routes_solved":1,'
                    '"points":[{"point_id":1,"route_id":"route_1","total_cost":24080.0,"emissions":0.0}],'
                    '"route_reports":[{"route_id":"route_1","status":"solved","polymer_solvent_map":{"PE":"Heptane","EVOH":"gamma-butyrolactone"}}],'
                    '"solvent_filter_warnings":["Broadened to optimizer-supported catalog."]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"visualization-specialist","schema_version":"1.0",'
                    '"plot_type":"optimization_pareto_front","plot_paths":["./plots/optimization_pareto_emissions.png"],'
                    '"format":"png"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_viz",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("optimization-engineer", "visualization-specialist")
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "Optimization Pareto plot created." in content
        assert "./plots/optimization_pareto_emissions.png" in content
        assert "Validated Pareto result: 1 feasible point(s)" in content
        assert "route_1: solved | PE-Heptane, EVOH-gamma-butyrolactone" in content
        assert "Filter warnings:" in content
        assert "Step 1" not in content
        assert "Cyclohexane at 76.0C" not in content
        assert (
            response.result[0].additional_kwargs["strap_origin"]
            == "routing_multi_specialist_optimization_visualization_fallback"
        )
        artifacts = response.result[0].additional_kwargs["strap_artifacts"]
        assert artifacts[0]["type"] == "optimization_pareto_front"
        assert artifacts[0]["data"]["payload"]["analysis_type"] == "pareto_front"

    def test_wrap_model_call_optimization_visualization_fallback_handles_pareto_slices(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(content="Run multi-composition optimization and plot all Pareto slices."),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.0","analysis_type":"pareto_slices",'
                    '"x_metric":"total_cost","y_metric":"circularity","n_slices_requested":2,"n_slices_solved":2,'
                    '"n_points_requested_per_slice":100,"pareto_slices_payload_path":"/tmp/pareto_slices.json",'
                    '"slices":[{"slice_id":"slice_1","label":"ldpe20_evoh60_pet20","status":"solved","n_points_feasible":3,"max_circularity":0.5233},'
                    '{"slice_id":"slice_2","label":"ldpe34_evoh33_pet33","status":"solved","n_points_feasible":8,"max_circularity":0.6714}]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"visualization-specialist","schema_version":"1.0",'
                    '"plot_type":"optimization_pareto_slices","plot_paths":["./plots/optimization_pareto_slices.png","./plots/optimization_pareto_slices_ldpe20.png"],'
                    '"format":"png"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_viz",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("optimization-engineer", "visualization-specialist")
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "Optimization Pareto slice plots created." in content
        assert "./plots/optimization_pareto_slices.png" in content
        assert "Validated multi-slice Pareto result: 2 of 2 composition slice(s) solved" in content
        assert "ldpe34_evoh33_pet33: solved; frontier points 8; max circularity 0.67" in content

    def test_wrap_model_call_optimization_visualization_fallback_formats_circularity_alias(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(content="Run optimization and generate a Pareto front plot for total cost vs circularity."),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.5","analysis_type":"pareto_front",'
                    '"x_metric":"total_cost","y_metric":"circularity","n_points_feasible":1,'
                    '"points":[{"point_id":1,"total_cost":1432919.96,"circularity_score":0.4808}],'
                    '"pareto_payload_path":"/tmp/pareto_payload.json"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"visualization-specialist","schema_version":"1.0",'
                    '"plot_type":"optimization_pareto_front","plot_paths":["./plots/optimization_pareto_circularity.png"],'
                    '"format":"png"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_viz",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("optimization-engineer", "visualization-specialist")
        )
        response = middleware.wrap_model_call(request, MagicMock())

        content = response.result[0].content
        assert "circularity 0.48" in content
        assert "circularity N/A" not in content

    def test_wrap_model_call_optimization_point_visualization_fallback(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Run optimization and generate a plot for the selected point-optimum design."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.5","analysis_type":"point_optimum",'
                    '"total_cost":7014529.18,"emissions":8700.71,"profit":12075070.82,'
                    '"circularity_score":0.6469,'
                    '"optimal_washes":["PE-Cyclohexane @ 120C","EVOH-Dimethyl sulfoxide @ 120C"]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"visualization-specialist","schema_version":"1.0",'
                    '"plot_type":"optimization_point_result","plot_paths":["./plots/optimization_point_result.png"],'
                    '"format":"png"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_viz",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for("optimization-engineer", "visualization-specialist")
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        content = response.result[0].content
        assert "Optimization plot created." in content
        assert "./plots/optimization_point_result.png" in content
        assert "Selected washes: PE-Cyclohexane @ 120C, EVOH-Dimethyl sulfoxide @ 120C." in content
        assert "Validated point result:" in content
        assert (
            response.result[0].additional_kwargs["strap_origin"]
            == "routing_multi_specialist_optimization_visualization_fallback"
        )
        artifacts = response.result[0].additional_kwargs["strap_artifacts"]
        assert artifacts[0]["type"] == "optimization_point_result"
        assert artifacts[0]["data"]["payload"]["analysis_type"] == "point_optimum"

    def test_wrap_model_call_optimization_visualization_fallback_tolerates_extra_allowed_specialists(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Run optimization and generate a Pareto front plot for total cost vs emissions."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_opt", "optimization-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"optimization-engineer","schema_version":"1.3","analysis_type":"pareto_front",'
                    '"x_metric":"total_cost","y_metric":"emissions","n_points_feasible":1,'
                    '"n_routes_requested":1,"n_routes_solved":1,'
                    '"points":[{"point_id":1,"route_id":"route_1","total_cost":24080.0,"emissions":0.0}],'
                    '"route_reports":[{"route_id":"route_1","status":"solved","polymer_solvent_map":{"PE":"Heptane","EVOH":"gamma-butyrolactone"}}],'
                    '"solvent_filter_warnings":[]}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_opt",
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_viz", "visualization-specialist")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"visualization-specialist","schema_version":"1.0",'
                    '"plot_type":"optimization_pareto_front","plot_paths":["./plots/optimization_pareto_emissions.png"],'
                    '"format":"png"}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_viz",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(
            classifier_model=_classifier_for(
                "separation-engineer",
                "optimization-engineer",
                "biosteam-analyst",
                "visualization-specialist",
            )
        )
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        assert (
            response.result[0].additional_kwargs["strap_origin"]
            == "routing_multi_specialist_optimization_visualization_fallback"
        )
        assert "Optimization Pareto plot created." in response.result[0].content

    def test_wrap_model_call_short_circuits_single_specialist_separation_even_when_task_status_is_invalid(self):
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        messages = [
            HumanMessage(
                content="Only do process design. Find the best separation sequence for PS, PET, and PC up to 120C at 1 atm."
            ),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_invalid", "separation-engineer")]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET","PC"],'
                    '"best_sequence":["PS","PC","PET"],'
                    '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":105.0},'
                    '{"step":2,"polymer":"PC","solvent":"THF","temperature_c":60.0}],'
                    '"solvent_mapping":{"PS":"Toluene","PC":"THF"}}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep_invalid",
            ),
        ]

        request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        handler = MagicMock()

        response = middleware.wrap_model_call(request, handler)

        handler.assert_not_called()
        assert "Recommended separation sequence" in response.result[0].content
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_separation_fallback"
        assert response.result[0].additional_kwargs["strap_handoff_status"] == "invalid"

    def test_wrap_model_call_short_circuits_single_specialist_separation_when_structured_result_is_missing(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware
        from strap.routing import RoutingMiddleware

        class _Request:
            def __init__(self, messages, system_message):
                self.messages = messages
                self.system_message = system_message

            def override(self, *, system_message):
                return _Request(self.messages, system_message)

        extractor = StructuredResultExtractorMiddleware()
        extractor.before_agent(
            {"messages": [HumanMessage(content="Only do process design. For EVOH/LDPE/PET, find the best atmospheric-pressure sequence up to 140C.")]},
            None,
        )
        tool_call = _task_call("tc_sep_missing", "separation-engineer")
        request = MagicMock()
        request.tool_call = tool_call
        missing_message = ToolMessage(content="", tool_call_id="tc_sep_missing")
        extractor.wrap_tool_call(request, MagicMock(return_value=missing_message))

        messages = [
            HumanMessage(
                content="Only do process design. For EVOH/LDPE/PET, find the best atmospheric-pressure sequence up to 140C."
            ),
            AIMessage(content="", tool_calls=[tool_call]),
            missing_message,
        ]

        routing_request = _Request(messages, SystemMessage(content="base system"))
        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        handler = MagicMock()

        response = middleware.wrap_model_call(routing_request, handler)

        handler.assert_not_called()
        assert "validated step-by-step separation sequence could not be extracted" in response.result[0].content.lower()
        assert response.result[0].additional_kwargs["strap_origin"] == "routing_single_specialist_missing_fallback"
        assert response.result[0].additional_kwargs["strap_handoff_status"] == "missing"
        extractor.after_agent(None, None)

    def test_single_specialist_handoff_lookup_is_blocked_after_separation_returns(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Only do process design. For EVOH/LDPE/PET, find the best atmospheric-pressure sequence up to 140C."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_done", "separation-engineer")]),
            ToolMessage(content="Specialist prose without structured block.", tool_call_id="tc_sep_done"),
            AIMessage(content="", tool_calls=[{
                "id": "lookup_sep",
                "name": "get_subagent_result",
                "args": {"agent_name": "separation-engineer"},
            }]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "lookup_sep"
        assert "disabled after `separation-engineer` has already returned" in update["messages"][0].content

    def test_single_specialist_duplicate_task_is_blocked_after_first_return(self):
        from strap.routing import RoutingMiddleware

        messages = [
            HumanMessage(content="Only do process design. Find the best separation sequence for PS and PET up to 120C."),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_first", "separation-engineer")]),
            ToolMessage(content="Specialist answer", tool_call_id="tc_sep_first"),
            AIMessage(content="", tool_calls=[_task_call("tc_sep_repeat", "separation-engineer")]),
        ]

        middleware = RoutingMiddleware(classifier_model=_classifier_for("separation-engineer"))
        update = middleware.after_model({"messages": messages}, MagicMock())

        assert update is not None
        assert update["messages"][0].tool_call_id == "tc_sep_repeat"
        assert "already returned for this single-specialist route" in update["messages"][0].content
