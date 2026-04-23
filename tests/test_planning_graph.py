from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from strap.planning_graph import (
    GENERIC_CONTEXT_ARTIFACT,
    build_planning_graph,
)


def _edge_lookup(graph, *, kind: str) -> dict[tuple[str, str], tuple[str, ...]]:
    edges = graph.capability_edges if kind == "capability" else graph.generic_edges
    return {
        (edge.producer, edge.consumer): edge.artifacts
        for edge in edges
    }


def test_build_planning_graph_loads_all_configured_subagents():
    graph = build_planning_graph()

    assert len(graph.nodes) == 10
    assert set(graph.nodes) == {
        "separation-engineer",
        "safety-analyst",
        "biosteam-analyst",
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "visualization-specialist",
        "statistics-ml",
        "contaminant-removal-analyst",
        "optimization-engineer",
    }

    for node in graph.nodes.values():
        assert node.goals
        assert node.produces
        assert node.requires
        assert GENERIC_CONTEXT_ARTIFACT in node.consumes
        assert node.cost_hint in {"low", "medium", "high"}
        assert node.latency_hint in {"low", "medium", "high"}

    assert graph.nodes["biosteam-analyst"].cost_hint == "high"
    assert graph.nodes["visualization-specialist"].parallel_group == "visualization"


def test_build_planning_graph_exposes_expected_capability_edges():
    graph = build_planning_graph()
    capability = _edge_lookup(graph, kind="capability")

    assert capability[("separation-engineer", "contaminant-removal-analyst")] == (
        "separation.route.v1",
        "solvent.shortlist.v1",
    )
    assert capability[("contaminant-removal-analyst", "biosteam-analyst")] == (
        "contaminant.screen.v1",
    )
    assert capability[("scholar-researcher", "rag-analyst")] == (
        "literature.findings.v1",
    )
    assert capability[("patent-researcher", "rag-analyst")] == (
        "patent.findings.v1",
    )
    assert capability[("biosteam-analyst", "visualization-specialist")] == (
        "tea.lca.v1",
    )
    assert capability[("optimization-engineer", "visualization-specialist")] == (
        "optimization.results.v1",
    )
    assert capability[("statistics-ml", "visualization-specialist")] == (
        "statistics.analysis.v1",
    )


def test_build_planning_graph_exposes_generic_fallback_edges_for_non_capability_pairs():
    graph = build_planning_graph()
    capability = _edge_lookup(graph, kind="capability")
    generic = _edge_lookup(graph, kind="generic")

    assert ("safety-analyst", "rag-analyst") not in capability
    assert generic[("safety-analyst", "rag-analyst")] == (GENERIC_CONTEXT_ARTIFACT,)


def test_build_planning_graph_rejects_invalid_planning_hint(tmp_path: Path):
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(
        yaml.safe_dump(
            {
                "subagents": [
                    {
                        "name": "alpha",
                        "description": "alpha desc",
                        "system_prompt": "prompt",
                        "tool_groups": ["reflection"],
                        "routing": {"priority": 1, "phrases": ["alpha"]},
                        "planning": {
                            "goals": ["alpha.goal"],
                            "produces": ["alpha.result.v1"],
                            "requires": ["user.alpha"],
                            "consumes": [GENERIC_CONTEXT_ARTIFACT],
                            "cost_hint": "fast",
                        },
                    }
                ]
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ValueError, match="cost_hint"):
        build_planning_graph(legacy_path)
