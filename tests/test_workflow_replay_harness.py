from __future__ import annotations

from strap.planning_graph import build_planning_graph
from strap.routing_classifier import derive_workflow_dependencies

from architecture.workflow_replay_harness import (
    build_workflow_replay_cases,
    replay_workflow_case,
    replay_workflow_case_until,
    resume_workflow_replay,
    run_workflow_replay_suite,
)


def _case_by_name(name: str):
    cases = {case.name: case for case in build_workflow_replay_cases()}
    return cases[name]


def test_workflow_replay_suite_passes_without_model_access():
    cases = build_workflow_replay_cases()
    summary = run_workflow_replay_suite()

    assert summary.total == len(cases)
    assert summary.passed == len(cases)
    assert summary.failed == 0
    assert sum(summary.blocked_model_call_attempts.values()) == 0


def test_workflow_replay_cases_include_graph_derived_capability_edges():
    graph = build_planning_graph()
    case_names = {case.name for case in build_workflow_replay_cases()}
    expected_names: set[str] = set()

    for edge in graph.capability_edges:
        dependency_map = derive_workflow_dependencies(
            f"Graph-derived replay coverage for {edge.producer} and {edge.consumer}.",
            {edge.producer, edge.consumer},
        )
        if edge.producer in dependency_map.get(edge.consumer, set()):
            expected_names.add(f"edge-{edge.producer}-to-{edge.consumer}")
        elif edge.consumer in dependency_map.get(edge.producer, set()):
            expected_names.add(f"edge-{edge.consumer}-to-{edge.producer}")

    assert expected_names.issubset(case_names)


def test_workflow_replay_complex_success_builds_all_join_handoffs_and_contracts():
    result = replay_workflow_case(_case_by_name("mixed-complex-success"))

    assert result.ok is True
    assert result.status == "complete"
    assert result.failed_subagents == []
    assert result.completed_subagents == [
        "separation-engineer",
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "contaminant-removal-analyst",
        "biosteam-analyst",
        "visualization-specialist",
    ]
    assert result.missing_direct_handoffs == []
    assert {
        (handoff["producer"], handoff["consumer"], handoff["contract"])
        for handoff in result.built_handoffs
    } == {
        ("biosteam-analyst", "visualization-specialist", "biosteam_plot.v1"),
        ("contaminant-removal-analyst", "biosteam-analyst", "contaminant_biosteam.v1"),
        ("patent-researcher", "rag-analyst", "patent_context.v1"),
        ("rag-analyst", "visualization-specialist", "rag-analyst.to.visualization-specialist.context.v1"),
        ("scholar-researcher", "rag-analyst", "literature_context.v1"),
        ("separation-engineer", "contaminant-removal-analyst", "contaminant_screen.v1"),
    }
    assert {
        handoff["contract"]
        for handoff in result.stored_multi_source_handoffs
    } == {
        "multi-source.to.rag-analyst.context.v1",
        "multi-source.to.visualization-specialist.context.v1",
    }


def test_workflow_replay_missing_structured_result_deadlocks_downstream_branch():
    result = replay_workflow_case(_case_by_name("mixed-complex-missing-contaminant"))

    assert result.ok is True
    assert result.status == "deadlock"
    assert result.failed_subagents == ["contaminant-removal-analyst"]
    assert "biosteam-analyst" not in result.completed_subagents
    assert "visualization-specialist" not in result.completed_subagents
    assert result.missing_direct_handoffs == []


def test_workflow_replay_retry_case_recovers_after_failed_upstream_attempt():
    result = replay_workflow_case(_case_by_name("sep-bio-retry-success"))

    assert result.ok is True
    assert result.status == "complete"
    assert result.completed_subagents == ["separation-engineer", "biosteam-analyst"]
    assert ("separation-engineer", "biosteam-analyst") in result.built_handoff_edges


def test_workflow_replay_checkpoint_resume_matches_single_pass_result():
    case = _case_by_name("mixed-complex-success")

    checkpoint = replay_workflow_case_until(
        case,
        stop_after_completed=(
            "separation-engineer",
            "scholar-researcher",
            "patent-researcher",
            "contaminant-removal-analyst",
        ),
    )
    resumed = resume_workflow_replay(checkpoint)
    single_pass = replay_workflow_case(case)

    assert checkpoint.status == "paused"
    assert resumed.ok is True
    assert resumed.status == single_pass.status
    assert resumed.completed_subagents == single_pass.completed_subagents
    assert {
        (handoff["producer"], handoff["consumer"], handoff["contract"])
        for handoff in resumed.built_handoffs
    } == {
        (handoff["producer"], handoff["consumer"], handoff["contract"])
        for handoff in single_pass.built_handoffs
    }
    assert {
        handoff["contract"]
        for handoff in resumed.stored_multi_source_handoffs
    } == {
        handoff["contract"]
        for handoff in single_pass.stored_multi_source_handoffs
    }


def test_workflow_replay_checkpoint_resume_preserves_retry_recovery_path():
    case = _case_by_name("research-rag-retry-scholar")

    checkpoint = replay_workflow_case_until(
        case,
        stop_after_failed=("scholar-researcher",),
    )
    resumed = resume_workflow_replay(checkpoint)

    assert checkpoint.status == "paused"
    assert resumed.ok is True
    assert resumed.status == "complete"
    assert resumed.completed_subagents == [
        "patent-researcher",
        "scholar-researcher",
        "rag-analyst",
    ]
    assert resumed.failed_subagents == ["scholar-researcher"]
    assert ("scholar-researcher", "rag-analyst") in resumed.built_handoff_edges
    assert ("patent-researcher", "rag-analyst") in resumed.built_handoff_edges
