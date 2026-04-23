from __future__ import annotations

import pytest

from strap.handoff_adapters import build_typed_handoff
from strap.handoff_models import HandoffRecord, HandoffScope


def _scope() -> HandoffScope:
    return HandoffScope(invocation_id="inv", run_id="run", thread_id="thread")


def test_build_typed_handoff_separation_to_optimization_includes_solvent_filters():
    source = HandoffRecord(
        handoff_id="handoff-sep-1",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "Toluene", "temperature_c": 110.0},
                {"step": 2, "polymer": "EVOH", "solvent": "Ethylene Glycol", "temperature_c": 90.0},
            ],
            "solvent_mapping": {"LDPE": "Toluene", "EVOH": "Ethylene Glycol"},
            "top_solvents": ["Toluene", "Xylene", "Ethylene Glycol"],
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Toluene", "EVOH": "Ethylene Glycol"}},
                {"rank": 2, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Xylene", "EVOH": "Pyridazine"}},
            ],
        },
        created_at="2026-04-17T18:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Find a separation route and then optimize waste management for the shortlisted solvents.",
    )

    assert typed is not None
    contract, payload, task_prompt = typed

    assert contract == "optimization.stage_candidates.v1"
    assert payload["workflow_scope"] == "multi_stage"
    assert payload["constraint_mode"] == "ranked_soft"
    assert payload["fallback_policy"] == "broaden_disclosed"
    assert payload["polymer_solvent_filters"]["PE"] == ["Toluene", "Xylene"]
    assert payload["polymer_solvent_filters"]["EVOH"] == ["Ethylene Glycol", "Pyridazine"]
    assert payload["stages"][0]["target_polymer"] == "PE"
    assert payload["stages"][1]["target_polymer"] == "EVOH"
    assert "candidate_solvents" in payload
    assert "stage_candidates_json" in task_prompt
    assert "constraint mode" in task_prompt.lower()


def test_build_typed_handoff_separation_to_optimization_normalizes_catalog_names():
    source = HandoffRecord(
        handoff_id="handoff-sep-2",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "cyclohexane", "temperature_c": 95.0},
                {"step": 2, "polymer": "EVOH", "solvent": "isopropylamine", "temperature_c": 80.0},
            ],
            "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "isopropylamine"},
            "top_solvents": ["cyclohexane", "isopropylamine"],
            "top_k_sequences": [],
        },
        created_at="2026-04-17T18:00:00Z",
    )

    typed = build_typed_handoff(source, "optimization-engineer")

    assert typed is not None
    _, payload, _ = typed
    assert payload["polymer_solvent_filters"]["PE"] == ["Cyclohexane"]
    assert payload["polymer_solvent_filters"]["EVOH"] == ["isopropylamine"]
    assert payload["stages"][0]["candidate_pairs"][0]["solvent"] == "Cyclohexane"


def test_build_typed_handoff_optimization_pareto_to_visualization():
    """Opt → viz adapter must produce a pareto_result_json handoff pointing at plot_optimization_pareto_front."""
    source = HandoffRecord(
        handoff_id="handoff-opt-1",
        scope=_scope(),
        producer="optimization-engineer",
        consumer="orchestrator",
        contract="optimization.results.v1",
        status="ok",
        payload={
            "analysis_type": "pareto_front",
            "schema_version": "1.0",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "n_points_feasible": 3,
            "points": [
                {"point_id": 1, "total_cost": 100.0, "emissions": 4.0},
                {"point_id": 2, "total_cost": 150.0, "emissions": 3.0},
                {"point_id": 3, "total_cost": 200.0, "emissions": 2.5},
            ],
        },
        created_at="2026-04-17T19:00:00Z",
    )

    typed = build_typed_handoff(source, "visualization-specialist", scope_user_query="Plot the Pareto frontier.")

    assert typed is not None
    contract, payload, prompt = typed
    assert contract == "optimization_plot_context.v1"
    assert payload["analysis_type"] == "pareto_front"
    assert payload["requested_plot_tool"] == "plot_optimization_pareto_front"
    assert payload["source_handoff_id"] == "handoff-opt-1"
    assert payload["pareto_result_json"]["points"][0]["total_cost"] == 100.0
    assert "plot_optimization_pareto_front" in prompt
    assert 'source_handoff_id="handoff-opt-1"' in prompt


def test_build_typed_handoff_resolves_dmso_aliases_to_optimizer_catalog_name():
    """Solvent aliases like 'dmso' and 'dimethylsulfoxide' must resolve to the optimizer catalog name."""
    source = HandoffRecord(
        handoff_id="handoff-sep-dmso",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "cyclohexane"},
                {"step": 2, "polymer": "EVOH", "solvent": "dimethylsulfoxide"},
            ],
            "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "dmso"},
            "top_solvents": ["cyclohexane", "dmso", "methanol"],
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "methanol"}},
                {"rank": 2, "sequence": ["EVOH", "LDPE"], "solvent_mapping": {"EVOH": "dimethylsulfoxide", "LDPE": "cyclohexane"}},
            ],
        },
        created_at="2026-04-20T12:00:00Z",
    )

    typed = build_typed_handoff(source, "optimization-engineer", scope_user_query="Find a route, prefer shortlist.")

    assert typed is not None
    _, payload, _ = typed
    # 'dimethylsulfoxide' alias must land as 'Dimethyl sulfoxide' in the EVOH filter.
    assert "Dimethyl sulfoxide" in payload["polymer_solvent_filters"].get("EVOH", [])
    # 'cyclohexane' (lowercase) must land as 'Cyclohexane' in the PE filter.
    assert "Cyclohexane" in payload["polymer_solvent_filters"].get("PE", [])


def test_build_typed_handoff_preserves_route_candidates_with_canonical_names():
    """route_candidates must preserve polymer-solvent coupling and use canonical names."""
    source = HandoffRecord(
        handoff_id="handoff-sep-routes",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "cyclohexane"},
                {"step": 2, "polymer": "EVOH", "solvent": "methanol"},
            ],
            "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "methanol"},
            "top_solvents": ["cyclohexane", "methanol"],
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "methanol"}},
                {"rank": 2, "sequence": ["EVOH", "LDPE"], "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "dmso"}},
                {"rank": 3, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "cyclohexane", "EVOH": "methanol"}},
            ],
        },
        created_at="2026-04-20T12:10:00Z",
    )

    typed = build_typed_handoff(source, "optimization-engineer", scope_user_query="Use exactly these routes for optimization.")

    assert typed is not None
    _, payload, _ = typed
    routes = payload.get("route_candidates")
    assert routes, "route_candidates must be emitted"
    # Duplicates must be collapsed (rank 1 and rank 3 have the same polymer_solvent_map)
    maps = [r["polymer_solvent_map"] for r in routes]
    signatures = {tuple(sorted(m.items())) for m in maps}
    assert len(signatures) == len(routes), "duplicate routes must be deduplicated"
    # All routes must use canonical names that resolve in the optimizer catalog.
    for route in routes:
        mapping = route["polymer_solvent_map"]
        assert "PE" in mapping, "LDPE must be canonicalized to PE"
        assert route["polymer_solvent_map"]["PE"] == "Cyclohexane"
    # DMSO alias must canonicalize even in route_candidates.
    assert any(r["polymer_solvent_map"].get("EVOH") == "Dimethyl sulfoxide" for r in routes)


def test_build_typed_handoff_defaults_to_ranked_soft_when_top_k_sequences_present():
    """With a neutral user query, the presence of top_k_sequences>=2 must flip
    constraint_mode to ranked_soft so the per-route Pareto branch activates
    without needing special prompting."""
    source = HandoffRecord(
        handoff_id="handoff-neutral",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "Cyclohexane"},
                {"step": 2, "polymer": "EVOH", "solvent": "Dimethyl sulfoxide"},
            ],
            "solvent_mapping": {"LDPE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
            "top_solvents": ["Cyclohexane", "Dimethyl sulfoxide"],
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"}},
                {"rank": 2, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Cyclohexane", "EVOH": "Ethylene Glycol"}},
            ],
        },
        created_at="2026-04-20T12:45:00Z",
    )

    # Plain query — no "prefer", "shortlist", "exactly" keywords
    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Run the optimization and produce a Pareto frontier.",
    )

    assert typed is not None
    _, payload, _ = typed
    assert payload["constraint_mode"] == "ranked_soft", (
        "top_k_sequences>=2 must flip default mode to ranked_soft so route "
        "enforcement activates without keyword triggers"
    )
    # route_candidates must be present so the per-route Pareto branch fires
    assert payload.get("route_candidates")


def test_build_typed_handoff_stays_soft_without_top_k_sequences():
    """If the DP planner didn't produce multiple routes, default mode stays soft."""
    source = HandoffRecord(
        handoff_id="handoff-single",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "polymers": ["LDPE"],
            "steps": [{"step": 1, "polymer": "LDPE", "solvent": "Cyclohexane"}],
            "solvent_mapping": {"LDPE": "Cyclohexane"},
            "top_solvents": ["Cyclohexane"],
            # No top_k_sequences (or length < 2)
        },
        created_at="2026-04-20T12:50:00Z",
    )
    typed = build_typed_handoff(source, "optimization-engineer", scope_user_query="Optimize.")
    assert typed is not None
    _, payload, _ = typed
    assert payload["constraint_mode"] == "soft"


def test_build_typed_handoff_uses_ranked_polymer_solvent_candidates(monkeypatch: pytest.MonkeyPatch):
    def _fake_optimizer_sets() -> dict[str, list[str]]:
        return {
            "S_PE": [f"PE Solvent {i}" for i in range(1, 61)],
            "S_EV1": [f"EVOH Solvent {i}" for i in range(1, 61)],
            "S_EV2": [],
            "S": [*(f"PE Solvent {i}" for i in range(1, 61)), *(f"EVOH Solvent {i}" for i in range(1, 61))],
            "P": ["PE", "EVOH"],
            "W": ["Wash 1", "Wash 2"],
        }

    monkeypatch.setattr(
        "strap.waste_management.data_loader.get_optimizer_default_sets",
        _fake_optimizer_sets,
    )

    source = HandoffRecord(
        handoff_id="handoff-sep-candidates",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "PE Solvent 1"},
                {"step": 2, "polymer": "EVOH", "solvent": "EVOH Solvent 1"},
            ],
            "solvent_mapping": {"LDPE": "PE Solvent 1", "EVOH": "EVOH Solvent 1"},
            "polymer_solvent_candidates": {
                "LDPE": (
                    [{"rank": 1, "solvent": "PE Solvent 1"}]
                    + [f"PE Solvent {i}" for i in range(2, 56)]
                    + ["PE Solvent 2", "PE Solvent 10"]
                ),
                "EVOH": [
                    {"rank": 1, "solvent": "EVOH Solvent 1"},
                    {"rank": 2, "solvent": "EVOH Solvent 2"},
                    {"rank": 3, "solvent": "EVOH Solvent 2"},
                    {"rank": 4, "solvent": "EVOH Solvent 3"},
                ],
            },
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "PE Solvent 1", "EVOH": "EVOH Solvent 1"}},
                {"rank": 2, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "PE Solvent 2", "EVOH": "EVOH Solvent 2"}},
            ],
        },
        created_at="2026-04-22T11:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Route this into optimization and Pareto analysis.",
    )

    assert typed is not None
    _, payload, _ = typed
    assert len(payload["polymer_solvent_filters"]["PE"]) == 50
    assert payload["polymer_solvent_filters"]["PE"][:4] == [
        "PE Solvent 1",
        "PE Solvent 2",
        "PE Solvent 3",
        "PE Solvent 4",
    ]
    assert payload["candidate_counts_by_polymer"]["PE"] == 50
    assert payload["candidate_counts_by_polymer"]["EVOH"] == 3
    assert payload["max_unique_solvents_per_polymer"] == 50

    evoh_pairs = payload["stages"][1]["candidate_pairs"]
    assert [pair["solvent"] for pair in evoh_pairs] == [
        "EVOH Solvent 1",
        "EVOH Solvent 2",
        "EVOH Solvent 3",
    ]
    assert [pair["source_rank"] for pair in evoh_pairs] == [1, 2, 4]
    assert all(pair["source_reason"] == "upstream ranked solvent candidate" for pair in evoh_pairs)
    assert payload["route_candidates"], "broad solvent pools must not suppress route_candidates"


def test_build_typed_handoff_infers_slot_independent_from_broad_pool_query():
    source = HandoffRecord(
        handoff_id="handoff-sep-broad-pool",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "Cyclohexane"},
                {"step": 2, "polymer": "EVOH", "solvent": "Dimethyl sulfoxide"},
            ],
            "solvent_mapping": {"LDPE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"}},
                {"rank": 2, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Heptane", "EVOH": "Ethylene Glycol"}},
            ],
        },
        created_at="2026-04-22T12:05:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Pass the top 50 unique solvent choices to the optimizer using broader solvent-pool semantics.",
    )

    assert typed is not None
    _, payload, _ = typed
    assert payload["route_pool_mode"] == "slot_independent"


def test_build_typed_handoff_optimization_point_solve_to_visualization_falls_back():
    """Point-optimum analyses have no native plot tool and must surface that fact."""
    source = HandoffRecord(
        handoff_id="handoff-opt-2",
        scope=_scope(),
        producer="optimization-engineer",
        consumer="orchestrator",
        contract="optimization.results.v1",
        status="ok",
        payload={
            "analysis_type": "point_optimum",
            "schema_version": "1.0",
            "profit": 5_000_000,
            "emissions": 2_000,
            "total_cost": 1_500_000,
            "optimal_washes": ["PE-Heptane"],
        },
        created_at="2026-04-17T19:05:00Z",
    )

    typed = build_typed_handoff(source, "visualization-specialist")
    assert typed is not None
    contract, payload, prompt = typed
    assert contract == "optimization_plot_context.v1"
    assert payload["analysis_type"] == "point_optimum"
    assert "no dedicated plotting tool" in prompt.lower()
