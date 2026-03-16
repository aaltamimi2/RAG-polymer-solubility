from __future__ import annotations

from langchain_core.messages import HumanMessage

from architecture.plan_only_harness import build_plan_only_cases
from architecture.workflow_replay_harness import build_workflow_replay_cases
from strap.planning_graph import build_planning_graph
from strap.routing import RoutingMiddleware
from strap.routing_classifier import (
    ROUTING_RULES,
    infer_available_query_inputs,
    infer_requested_goals,
)
from strap.routing_message_state import _get_ordered_plan


def _build_runtime_plan(
    query: str,
    *,
    allowed_subagents: tuple[str, ...] = (),
) -> tuple[list[dict], list[dict]]:
    messages = [HumanMessage(content=query)]
    if allowed_subagents:
        rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
        allowed_rules = [rules_by_name[name] for name in allowed_subagents]
    else:
        middleware = RoutingMiddleware(classifier_model=None)
        allowed_rules = middleware._get_allowed_rules(messages)
    return allowed_rules, _get_ordered_plan(messages, allowed_rules=allowed_rules)


def _topologically_sorted(plan: list[dict]) -> bool:
    positions = {
        step["subagent"]: index
        for index, step in enumerate(plan)
    }
    for step in plan:
        consumer = step["subagent"]
        for dependency in step.get("depends_on", ()):
            if positions.get(dependency, -1) >= positions[consumer]:
                return False
    return True


def test_generated_query_suite_runtime_plans_stay_acyclic_and_dependency_closed():
    graph = build_planning_graph()
    edge_artifacts = {
        (edge.producer, edge.consumer): set(edge.artifacts)
        for edge in graph.edges
    }

    plan_only_cases = build_plan_only_cases()
    replay_cases = build_workflow_replay_cases()

    for case in [*plan_only_cases, *replay_cases]:
        allowed_rules, plan = _build_runtime_plan(
            case.query,
            allowed_subagents=getattr(case, "allowed_subagents", ()),
        )
        allowed_names = [rule["subagent"] for rule in allowed_rules]
        planned_names = [step["subagent"] for step in plan]

        assert planned_names, f"{case.name}: empty runtime plan"
        assert len(planned_names) == len(set(planned_names)), f"{case.name}: duplicate steps in plan"
        assert set(planned_names) == set(allowed_names), f"{case.name}: allowed/plan set mismatch"
        assert _topologically_sorted(plan), f"{case.name}: plan is not topologically sorted"

        for step in plan:
            consumer = step["subagent"]
            dependencies = tuple(step.get("depends_on", ()))
            assert consumer not in dependencies, f"{case.name}: self dependency for {consumer}"
            for producer in dependencies:
                assert producer in planned_names, f"{case.name}: dependency {producer} missing from plan"
                assert (producer, consumer) in edge_artifacts, (
                    f"{case.name}: dependency {producer} -> {consumer} is not a planning-graph edge"
                )
                assert edge_artifacts[(producer, consumer)], (
                    f"{case.name}: dependency {producer} -> {consumer} has no associated artifacts"
                )


def test_generated_query_suite_runtime_plans_cover_requested_goals_and_query_requirements():
    graph = build_planning_graph()

    plan_only_cases = build_plan_only_cases()
    replay_cases = build_workflow_replay_cases()

    for case in [*plan_only_cases, *replay_cases]:
        if getattr(case, "allowed_subagents", ()):
            continue
        _allowed_rules, plan = _build_runtime_plan(
            case.query,
            allowed_subagents=getattr(case, "allowed_subagents", ()),
        )
        planned_names = {step["subagent"] for step in plan}
        requested_goals = infer_requested_goals(case.query)
        available_inputs = infer_available_query_inputs(case.query)

        covered_goals: set[str] = set()
        for name in planned_names:
            node = graph.nodes[name]
            assert set(node.requires).issubset(available_inputs), (
                f"{case.name}: selected {name} without satisfying query requirements {node.requires}"
            )
            covered_goals.update(node.goals)

        assert requested_goals.issubset(covered_goals), (
            f"{case.name}: requested goals {sorted(requested_goals - covered_goals)} not covered by planned nodes"
        )
