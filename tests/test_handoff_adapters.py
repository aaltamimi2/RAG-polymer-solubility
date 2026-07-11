from __future__ import annotations

import json

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
    assert payload["fallback_policy"] == "fail_closed"
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
    assert payload["requested_plot_mode"] is None
    assert payload["source_handoff_id"] == "handoff-opt-1"
    assert payload["pareto_result_json"]["points"][0]["total_cost"] == 100.0
    assert "plot_optimization_pareto_front" in prompt
    assert 'source_handoff_id="handoff-opt-1"' in prompt


def test_build_typed_handoff_optimization_pareto_to_visualization_requests_landscape_mode():
    source = HandoffRecord(
        handoff_id="handoff-opt-landscape",
        scope=_scope(),
        producer="optimization-engineer",
        consumer="orchestrator",
        contract="optimization.results.v1",
        status="ok",
        payload={
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "n_points_feasible": 1,
            "points": [{"point_id": 1, "total_cost": 100.0, "emissions": 4.0}],
        },
        created_at="2026-04-24T15:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "visualization-specialist",
        scope_user_query="Create an optimization Pareto landscape including all feasible points and highlight the frontier.",
    )
    assert typed is not None
    contract, payload, prompt = typed
    assert contract == "optimization_plot_context.v1"
    assert payload["requested_plot_mode"] == "landscape"
    assert 'plot_mode="landscape"' in prompt


def test_build_typed_handoff_optimization_pareto_to_visualization_sets_composition_output_stem():
    source = HandoffRecord(
        handoff_id="handoff-opt-composition",
        scope=_scope(),
        producer="optimization-engineer",
        consumer="orchestrator",
        contract="optimization.results.v1",
        status="ok",
        payload={
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "n_points_feasible": 1,
            "points": [{"point_id": 1, "total_cost": 100.0, "emissions": 4.0}],
        },
        created_at="2026-04-25T15:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "visualization-specialist",
        scope_user_query=(
            "For a mixed plastic feedstock composed of 60% LDPE, 20% EVOH, and 20% PET, "
            "save the plot under a composition-specific filename."
        ),
    )

    assert typed is not None
    _contract, payload, prompt = typed
    assert payload["requested_output_stem"] == "optimization_pareto_emissions_ldpe60_evoh20_pet20"
    assert 'output_stem="optimization_pareto_emissions_ldpe60_evoh20_pet20"' in prompt


def test_build_typed_handoff_optimization_pareto_slices_to_visualization():
    source = HandoffRecord(
        handoff_id="handoff-opt-slices",
        scope=_scope(),
        producer="optimization-engineer",
        consumer="orchestrator",
        contract="optimization.results.v1",
        status="ok",
        payload={
            "analysis_type": "pareto_slices",
            "schema_version": "1.0",
            "x_metric": "total_cost",
            "y_metric": "circularity",
            "n_slices_requested": 2,
            "n_slices_solved": 2,
            "pareto_slices_payload_path": "/tmp/pareto_slices.json",
            "slices": [
                {"slice_id": "slice_1", "label": "20/60/20", "n_points_feasible": 9},
            ],
        },
        created_at="2026-04-25T00:00:00Z",
    )

    typed = build_typed_handoff(source, "visualization-specialist", scope_user_query="Plot all slices as a landscape.")

    assert typed is not None
    contract, payload, prompt = typed
    assert contract == "optimization_plot_context.v1"
    assert payload["analysis_type"] == "pareto_slices"
    assert payload["requested_plot_tool"] == "plot_optimization_pareto_slices"
    assert payload["requested_plot_mode"] == "landscape"
    assert payload["pareto_slices_json"]["pareto_slices_payload_path"] == "/tmp/pareto_slices.json"
    assert "plot_optimization_pareto_slices" in prompt


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


def test_build_typed_handoff_includes_pet_candidates_for_optimization():
    source = HandoffRecord(
        handoff_id="handoff-sep-pet",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH", "PET"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "cyclohexane"},
                {"step": 2, "polymer": "EVOH", "solvent": "dimethylsulfoxide"},
                {"step": 3, "polymer": "PET", "solvent": "n,n-dimethylformamide"},
            ],
            "solvent_mapping": {
                "LDPE": "cyclohexane",
                "EVOH": "dimethylsulfoxide",
                "PET": "n,n-dimethylformamide",
            },
            "top_solvents": ["cyclohexane", "dimethylsulfoxide", "n,n-dimethylformamide"],
            "top_k_sequences": [
                {
                    "rank": 1,
                    "sequence": ["LDPE", "EVOH", "PET"],
                    "solvent_mapping": {
                        "LDPE": "cyclohexane",
                        "EVOH": "dimethylsulfoxide",
                        "PET": "n,n-dimethylformamide",
                    },
                }
            ],
        },
        created_at="2026-04-23T10:00:00Z",
    )

    typed = build_typed_handoff(source, "optimization-engineer")

    assert typed is not None
    _, payload, _ = typed
    assert "PET" in payload["polymer_solvent_filters"]
    assert payload["polymer_solvent_filters"]["PET"] == ["N,N-Dimethylformamide"]
    assert any(stage["target_polymer"] == "PET" for stage in payload["stages"])
    assert any(route["polymer_solvent_map"].get("PET") == "N,N-Dimethylformamide" for route in payload["route_candidates"])


def test_build_typed_handoff_includes_extended_polymer_candidates_for_optimization():
    source = HandoffRecord(
        handoff_id="handoff-sep-extended-polymers",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["PP", "PS", "polyvinyl chloride", "polycarbonate"],
            "steps": [
                {"step": 1, "polymer": "PP", "solvent": "toluene"},
                {"step": 2, "polymer": "PS", "solvent": "n,n-dimethylformamide"},
                {"step": 3, "polymer": "polyvinyl chloride", "solvent": "dimethylsulfoxide"},
                {"step": 4, "polymer": "polycarbonate", "solvent": "toluene"},
            ],
            "solvent_mapping": {
                "PP": "toluene",
                "PS": "n,n-dimethylformamide",
                "polyvinyl chloride": "dimethylsulfoxide",
                "polycarbonate": "toluene",
            },
            "top_k_sequences": [
                {
                    "rank": 1,
                    "sequence": ["PP", "PS", "polyvinyl chloride", "polycarbonate"],
                    "solvent_mapping": {
                        "PP": "toluene",
                        "PS": "n,n-dimethylformamide",
                        "polyvinyl chloride": "dimethylsulfoxide",
                        "polycarbonate": "toluene",
                    },
                }
            ],
        },
        created_at="2026-04-25T10:00:00Z",
    )

    typed = build_typed_handoff(source, "optimization-engineer")

    assert typed is not None
    _, payload, _ = typed
    assert payload["polymer_solvent_filters"]["PP"] == ["Toluene"]
    assert payload["polymer_solvent_filters"]["PS"] == ["N,N-Dimethylformamide"]
    assert payload["polymer_solvent_filters"]["PVC"] == ["Dimethyl sulfoxide"]
    assert payload["polymer_solvent_filters"]["PC"] == ["Toluene"]
    assert {stage["target_polymer"] for stage in payload["stages"]} >= {"PP", "PS", "PVC", "PC"}
    assert any(
        route["polymer_solvent_map"].get("PVC") == "Dimethyl sulfoxide"
        and route["polymer_solvent_map"].get("PC") == "Toluene"
        for route in payload["route_candidates"]
    )




def test_build_typed_handoff_ranked_candidate_query_uses_fail_closed_fallback():
    source = HandoffRecord(
        handoff_id="handoff-sep-ranked",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH", "PET"],
            "polymer_solvent_candidates": {
                "LDPE": [{"rank": 1, "solvent": "cyclohexane"}],
                "EVOH": [{"rank": 1, "solvent": "ethylene glycol"}],
                "PET": [{"rank": 1, "solvent": "N,N-Dimethylformamide"}],
            },
            "top_k_sequences": [],
        },
        created_at="2026-04-23T00:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query=(
            "Have the separation engineer propose the top 3 solvent candidates per polymer "
            "and pass those shortlisted candidates to the optimization engineer."
        ),
    )

    assert typed is not None
    _, payload, _ = typed
    assert payload["constraint_mode"] == "ranked_soft"
    assert payload["fallback_policy"] == "fail_closed"

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


def test_build_typed_handoff_backfills_underreported_top_n_candidates(monkeypatch: pytest.MonkeyPatch):
    def fake_plan_sequential_separation(**kwargs):
        assert kwargs["top_k_solvents"] == 3
        return json.dumps(
            {
                "data": {
                    "polymer_solvent_candidates": {
                        "LDPE": [
                            {"rank": 1, "solvent": "cyclohexane", "temperature_c": 79.7},
                            {"rank": 2, "solvent": "hexane", "temperature_c": 67.7},
                            {"rank": 3, "solvent": "n-heptane", "temperature_c": 97.5},
                        ],
                        "EVOH": [
                            {"rank": 1, "solvent": "methanol", "temperature_c": 63.6},
                            {"rank": 2, "solvent": "dimethylsulfoxide", "temperature_c": 145.0},
                            {"rank": 3, "solvent": "ethanol", "temperature_c": 77.2},
                        ],
                        "PET": [
                            {"rank": 1, "solvent": "dimethylformamide", "temperature_c": 100.0},
                            {"rank": 2, "solvent": "pyridine", "temperature_c": 114.0},
                            {"rank": 3, "solvent": "dimethylsulfoxide", "temperature_c": 145.0},
                        ],
                    }
                }
            }
        )

    monkeypatch.setattr(
        "strap.tools.sequence_planning_tools.plan_sequential_separation",
        fake_plan_sequential_separation,
    )
    source = HandoffRecord(
        handoff_id="handoff-underfilled-topn",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH", "PET"],
            "polymer_solvent_candidates": {
                "LDPE": [{"rank": 1, "solvent": "cyclohexane", "temperature_c": 79.7}],
                "EVOH": [{"rank": 1, "solvent": "methanol", "temperature_c": 63.6}],
            },
            "top_k_sequences": [],
        },
        created_at="2026-04-24T00:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Propose the top 3 solvent candidates per polymer and optimize.",
    )

    assert typed is not None
    _, payload, prompt = typed
    assert payload["candidate_counts_by_polymer"] == {"PE": 3, "EVOH": 3, "PET": 3}
    assert payload["candidate_backfill_warnings"]
    assert "Candidate backfill warnings" in prompt
    pet_stage = next(stage for stage in payload["stages"] if stage["target_polymer"] == "PET")
    assert [pair["solvent"] for pair in pet_stage["candidate_pairs"]][:2] == [
        "N,N-Dimethylformamide",
        "pyridine",
    ]


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


def test_build_typed_handoff_infers_slot_independent_from_solvent_choices_query():
    source = HandoffRecord(
        handoff_id="handoff-sep-solvent-choices",
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
        created_at="2026-04-23T14:05:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query=(
            "Have the separation engineer identify the top 6 solvent choices for LDPE and EVOH recovery, "
            "then pass those shortlisted solvent candidates to the optimizer for a Pareto sweep."
        ),
    )

    assert typed is not None
    _, payload, _ = typed
    assert payload["route_pool_mode"] == "slot_independent"


def test_build_typed_handoff_optimization_infeasible_to_visualization_skips_plotting():
    source = HandoffRecord(
        handoff_id="handoff-opt-infeasible",
        scope=_scope(),
        producer="optimization-engineer",
        consumer="orchestrator",
        contract="optimization.results.v1",
        status="ok",
        payload={
            "analysis_type": "infeasible",
            "schema_version": "1.3",
            "failure_reason": "no_candidate_overlap",
            "message": "No valid candidates survived.",
        },
        created_at="2026-04-24T12:00:00Z",
    )

    typed = build_typed_handoff(source, "visualization-specialist")
    assert typed is not None
    contract, payload, prompt = typed
    assert contract == "optimization_plot_context.v1"
    assert payload["analysis_type"] == "infeasible"
    assert payload["requested_plot_tool"] is None
    assert "Do not call a plotting tool" in prompt


def test_build_typed_handoff_optimization_point_solve_to_visualization_uses_point_plot_tool():
    """Point-optimum analyses should route to the dedicated point-result plotting tool."""
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
    assert payload["requested_plot_tool"] == "plot_optimization_point_result"
    assert "plot_optimization_point_result" in prompt


def test_build_typed_handoff_preserves_temperature_distinct_optimizer_options():
    source = HandoffRecord(
        handoff_id="handoff-sep-temp-variants",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.1",
            "polymers": ["PET"],
            "steps": [
                {"step": 1, "polymer": "PET", "solvent": "dimethyl sulfoxide", "temperature_c": 135.0},
            ],
            "solvent_mapping": {"PET": "dimethyl sulfoxide"},
            "polymer_solvent_candidates": {
                "PET": [
                    {"rank": 1, "solvent": "dimethyl sulfoxide", "temperature_c": 135.0},
                    {"rank": 2, "solvent": "dimethyl sulfoxide", "temperature_c": 145.0},
                ]
            },
            "top_k_sequences": [
                {
                    "rank": 1,
                    "sequence": ["PET"],
                    "solvent_mapping": {"PET": "dimethyl sulfoxide"},
                    "steps": [{"step": 1, "polymer": "PET", "solvent": "dimethyl sulfoxide", "temperature_c": 135.0}],
                },
                {
                    "rank": 2,
                    "sequence": ["PET"],
                    "solvent_mapping": {"PET": "dimethyl sulfoxide"},
                    "steps": [{"step": 1, "polymer": "PET", "solvent": "dimethyl sulfoxide", "temperature_c": 145.0}],
                },
            ],
        },
        created_at="2026-04-23T18:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Use the shortlisted solvent candidates in optimization.",
    )

    assert typed is not None
    _, payload, _ = typed
    pet_stage = next(stage for stage in payload["stages"] if stage["target_polymer"] == "PET")
    options = [pair["optimizer_option"] for pair in pet_stage["candidate_pairs"]]
    assert "Dimethyl sulfoxide @ 135C" in options
    assert "Dimethyl sulfoxide @ 145C" in options
    assert len(options) == 2

    route_options = [
        route["polymer_solvent_map"]["PET"]
        for route in payload["route_candidates"]
    ]
    assert "Dimethyl sulfoxide @ 135C" in route_options
    assert "Dimethyl sulfoxide @ 145C" in route_options


def test_build_typed_handoff_skips_placeholder_step_solvents():
    source = HandoffRecord(
        handoff_id="handoff-sep-placeholders",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.1",
            "polymers": ["LDPE", "PET"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "cyclohexane", "temperature_c": 120.0},
                {"step": 2, "polymer": "PET", "solvent": "N/A (Solid Residue)", "temperature_c": 120.0},
            ],
            "polymer_solvent_candidates": {
                "LDPE": [{"rank": 1, "solvent": "cyclohexane"}],
                "PET": [{"rank": 1, "solvent": "dimethyl sulfoxide"}],
            },
            "top_k_sequences": [
                {
                    "rank": 1,
                    "sequence": ["LDPE", "PET"],
                    "solvent_mapping": {"LDPE": "cyclohexane"},
                }
            ],
        },
        created_at="2026-04-23T19:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Use the shortlisted solvent candidates in optimization.",
    )

    assert typed is not None
    _, payload, _ = typed
    pet_stage = next(stage for stage in payload["stages"] if stage["target_polymer"] == "PET")
    options = [pair["optimizer_option"] for pair in pet_stage["candidate_pairs"]]
    assert "N/A (Solid Residue) @ 120C" not in options
    assert options == ["Dimethyl sulfoxide"]


def test_build_typed_handoff_separation_to_optimization_carries_feed_metadata_without_inline_json():
    source = HandoffRecord(
        handoff_id="handoff-sep-feed",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH", "PET"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "Toluene", "temperature_c": 110.0},
                {"step": 2, "polymer": "EVOH", "solvent": "Ethylene Glycol", "temperature_c": 90.0},
            ],
            "solvent_mapping": {"LDPE": "Toluene", "EVOH": "Ethylene Glycol"},
            "top_solvents": ["Toluene", "Ethylene Glycol"],
            "top_k_sequences": [],
        },
        created_at="2026-04-23T12:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query=(
            "For a mixed plastic feedstock of 8000 tonnes/year composed of 5% LDPE, 5% EVOH, and 90% PET, "
            "optimize waste management for the shortlisted solvents."
        ),
    )

    assert typed is not None
    _, payload, task_prompt = typed
    assert payload["feed_capacity_tpy"] == 8000.0
    assert payload["feed_composition"] == {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9}
    assert "Exact `stage_candidates_json`" not in task_prompt
    assert "attached to your runtime state" in task_prompt
    assert "injects the attached payload" in task_prompt
