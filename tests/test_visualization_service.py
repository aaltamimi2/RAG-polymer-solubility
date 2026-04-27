"""Tests for visualization service helpers."""

import asyncio
import json
from pathlib import Path

from strap.handoff_models import HandoffRecord, HandoffScope


def test_normalize_solvent_names_expands_aliases():
    from strap.services.visualization_service import normalize_solvent_names

    normalized = normalize_solvent_names(["xylene", "DMSO", "THF"])

    assert "1,2-dimethylbenzene" in normalized
    assert "1,4-dimethylbenzene" in normalized
    assert "dimethylsulfoxide" in normalized
    assert "thf" in normalized


def test_normalize_solvent_names_reconstructs_fragmented_names():
    from strap.services.visualization_service import normalize_solvent_names

    normalized = normalize_solvent_names(["2", "3-dihydropyran", "toluene"])

    assert normalized[0] == "2,3-dihydropyran"
    assert normalized[1] == "toluene"


def test_get_plot_url_formats_path():
    from strap.services.visualization_service import get_plot_url

    assert get_plot_url("/tmp/plot.png") == "Plot saved: `/tmp/plot.png`"


def test_resolve_plot_output_path_accepts_wsl_unc_directory():
    from strap.tools._helpers import resolve_plot_output_path

    path = resolve_plot_output_path(
        "ldpe_dodecane_solubility",
        output_dir=(
            r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
            r"\langchain-STRAP-v9-contaminants\docs\case_studies"
            r"\separation_engineer_publication\cli-plots"
        ),
    )

    assert path == (
        "/home/aaltamimi2/langchain-STRAP-v9-contaminants/docs/case_studies/"
        "separation_engineer_publication/cli-plots/ldpe_dodecane_solubility.png"
    )


def test_plot_solubility_vs_temperature_writes_to_requested_dir(tmp_path):
    from strap.tools import visualization

    raw = visualization.plot_solubility_vs_temperature(
        table_name="common_solvents_database",
        polymer_column="polymer",
        solvent_column="solvent",
        temperature_column="temperature___c_",
        solubility_column="solubility",
        polymers="LDPE",
        solvents="Dodecane",
        temperature_max=140,
        output_dir=str(tmp_path),
    )
    payload = json.loads(raw)
    plot_path = payload["data"]["plot_filepath"]

    assert plot_path.startswith(str(tmp_path))
    assert plot_path.endswith(".png")
    assert (tmp_path / "ldpe_dodecane_solubility_vs_temp.png").is_file()
    assert payload["data"]["plot_url"] == f"Plot saved: `{plot_path}`"


def test_plot_solubility_vs_temperature_preserves_multi_polymer_multi_solvent_request(tmp_path):
    from strap.tools import visualization

    raw = visualization.plot_solubility_vs_temperature(
        table_name="common_solvents_database",
        polymer_column="polymer",
        solvent_column="solvent",
        temperature_column="temperature___c_",
        solubility_column="solubility",
        polymers="LDPE, PET, EVOH",
        solvents="dodecane, o-xylene",
        temperature_min=25,
        temperature_max=160,
        output_dir=str(tmp_path),
    )
    payload = json.loads(raw)

    assert payload["data"]["polymers"] == ["LDPE", "PET", "EVOH"]
    assert payload["data"]["solvents"] == ["dodecane", "1,2-dimethylbenzene"]
    assert payload["data"]["data_points"] == 168
    assert payload["data"]["excluded_pairs"] == []
    assert payload["data"]["y_axis_max"] == 100.0
    assert "Y-axis range: 0-100%" in payload["display"]
    assert (
        tmp_path
        / "ldpe_pet_evoh_dodecane_1_2_dimethylbenzene_solubility_vs_temp.png"
    ).is_file()


def test_plot_solubility_vs_temperature_excludes_quarantined_pair(tmp_path):
    from strap.tools import visualization

    raw = visualization.plot_solubility_vs_temperature(
        table_name="common_solvents_database",
        polymer_column="polymer",
        solvent_column="solvent",
        temperature_column="temperature___c_",
        solubility_column="solubility",
        polymers="EVOH",
        solvents="DMF, triethylamine",
        temperature_max=80,
        output_dir=str(tmp_path),
    )
    payload = json.loads(raw)

    assert payload["data"]["success"] is True
    assert payload["data"]["solvents"] == ["dimethylformamide"]
    assert payload["data"]["excluded_pairs"] == ["EVOH/triethylamine"]
    assert "Excluded data-quality pair(s): EVOH/triethylamine" in payload["display"]


def test_plot_solubility_vs_temperature_excludes_triethylamine_globally(tmp_path):
    from strap.tools import visualization

    raw = visualization.plot_solubility_vs_temperature(
        table_name="common_solvents_database",
        polymer_column="polymer",
        solvent_column="solvent",
        temperature_column="temperature___c_",
        solubility_column="solubility",
        polymers="LDPE, PET",
        solvents="dodecane, triethylamine",
        temperature_max=80,
        output_dir=str(tmp_path),
    )
    payload = json.loads(raw)

    assert payload["data"]["success"] is True
    assert payload["data"]["solvents"] == ["dodecane"]
    assert payload["data"]["excluded_pairs"] == ["LDPE/triethylamine", "PET/triethylamine"]


def test_create_separation_tree_plot_honors_output_dir(tmp_path):
    from strap.tools.advanced_separation import create_separation_tree_plot

    raw = create_separation_tree_plot("LDPE,EVOH,PET", temperature=100, output_dir=str(tmp_path))
    payload = json.loads(raw)

    assert payload["data"]["success"] is True
    assert payload["data"]["rank1_plot"] == str(tmp_path / "separation_sequence_rank1.png")
    assert payload["data"]["topk_plot"] == str(tmp_path / "separation_topk_comparison.png")
    assert (tmp_path / "separation_sequence_rank1.png").is_file()
    assert (tmp_path / "separation_topk_comparison.png").is_file()


def test_plot_dynamic_programming_separation_options_creates_publication_artifacts(tmp_path):
    from strap.tools.advanced_separation import plot_dynamic_programming_separation_options

    raw = plot_dynamic_programming_separation_options(
        "LDPE,EVOH,PET",
        temperature=100,
        output_dir=str(tmp_path),
        include_objective_paths=True,
        objectives="selectivity,greenness,energy",
    )
    payload = json.loads(raw)

    assert payload["data"]["success"] is True
    assert payload["data"]["total_sequences_evaluated"] == 6
    expected = {
        "separation_sequence_rank1.png",
        "separation_topk_comparison.png",
        "separation_dp_state_map.png",
        "separation_objective_paths.png",
    }
    assert {Path(path).name for path in payload["data"]["plot_paths"]} == expected
    for filename in expected:
        assert (tmp_path / filename).is_file()


def test_plot_dynamic_programming_separation_options_defaults_to_separation_only(tmp_path):
    from strap.tools.advanced_separation import plot_dynamic_programming_separation_options

    raw = plot_dynamic_programming_separation_options("LDPE,EVOH,PET", temperature=100, output_dir=str(tmp_path))
    payload = json.loads(raw)

    assert {Path(path).name for path in payload["data"]["plot_paths"]} == {
        "separation_sequence_rank1.png",
        "separation_topk_comparison.png",
        "separation_dp_state_map.png",
    }
    assert not (tmp_path / "separation_objective_paths.png").exists()
    assert payload["data"]["objectives_requested"] == ["selectivity"]


def test_execute_query_blocks_unsafe_sql():
    from strap.services.visualization_service import execute_query

    result = execute_query("DROP TABLE solvent_data")

    assert result["success"] is False
    assert "Unsafe" in result["error"]


def test_get_solvent_table_name_detects_property_table(conn):
    from strap.services.visualization_service import get_solvent_table_name

    table = get_solvent_table_name()

    assert table == "solvent_data"


def test_get_solvent_name_and_cosmobase_columns(conn):
    from strap.services.visualization_service import (
        get_cosmobase_column,
        get_solvent_name_column,
    )

    assert get_solvent_name_column("solvent_data") == "solvent_name"
    assert get_cosmobase_column("solvent_data") == "solvent_name_in_cosmobase"


def test_verify_inputs_accepts_known_table_and_columns(conn):
    from strap.services.visualization_service import verify_inputs

    ok, message = verify_inputs(
        "common_solvents_database",
        {
            "polymer": "polymer",
            "solvent": "solvent",
            "temperature": "temperature___c_",
        },
        {"polymer": ["LDPE"]},
    )

    assert ok is True
    assert message == "All inputs verified"


def test_lookup_solvent_properties_returns_expected_fields(conn):
    from strap.services.visualization_service import lookup_solvent_properties

    result = asyncio.run(lookup_solvent_properties(["Toluene", "THF"], "solvent_data"))

    assert "Toluene" in result
    assert "THF" in result
    assert result["Toluene"]["bp"] is not None


def test_plot_optimization_pareto_front_reads_authoritative_handoff(monkeypatch, tmp_path):
    from strap import handoffs
    from strap.tools import visualization

    handoff = HandoffRecord(
        handoff_id="h_opt_plot",
        scope=HandoffScope(invocation_id="inv", run_id="run", thread_id="thread"),
        producer="optimization-engineer",
        consumer="visualization-specialist",
        contract="optimization_plot_context.v1",
        status="ok",
        payload={
            "source_handoff_id": "h_opt_result",
            "analysis_type": "pareto_front",
            "pareto_result_json": {
                "analysis_type": "pareto_front",
                "x_metric": "total_cost",
                "y_metric": "emissions",
                "points": [
                    {"point_id": 1, "total_cost": 6313346.86, "emissions": 9449.13},
                ],
            },
        },
        created_at="2026-04-20T19:00:00Z",
    )

    output_path = tmp_path / "optimization_pareto_emissions.png"
    monkeypatch.setattr(handoffs, "get_handoff", lambda handoff_id: handoff if handoff_id == "h_opt_plot" else None)
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(source_handoff_id="h_opt_plot")
    payload = json.loads(raw)

    assert payload["data"]["plot_type"] == "optimization_pareto_front"
    assert payload["data"]["n_points"] == 1
    assert payload["data"]["source_handoff_id"] == "h_opt_plot"
    assert payload["data"]["color_by"] == "emissions"
    assert payload["data"]["plot_paths"] == [str(output_path)]
    assert payload["data"]["point_legend"] == ["P1: Cost: $6.31M | Emissions:\n    9,449.1"]


def test_plot_optimization_point_result_reads_authoritative_handoff(monkeypatch, tmp_path):
    from strap import handoffs
    from strap.tools import visualization

    handoff = HandoffRecord(
        handoff_id="h_opt_point_plot",
        scope=HandoffScope(invocation_id="inv", run_id="run", thread_id="thread"),
        producer="optimization-engineer",
        consumer="visualization-specialist",
        contract="optimization_plot_context.v1",
        status="ok",
        payload={
            "source_handoff_id": "h_opt_point_result",
            "analysis_type": "point_optimum",
            "optimization_result_json": {
                "analysis_type": "point_optimum",
                "scenario": "A",
                "feed_composition": {"PE": 0.6, "EVOH": 0.4},
                "profit": 12_000_000.0,
                "total_cost": 7_000_000.0,
                "emissions": 8_700.0,
                "circularity_score": 0.647,
                "optimal_washes": ["PE-Cyclohexane @ 120C", "EVOH-Dimethyl sulfoxide @ 120C"],
            },
        },
        created_at="2026-04-24T09:00:00Z",
    )

    output_path = tmp_path / "optimization_point_result.png"
    monkeypatch.setattr(handoffs, "get_handoff", lambda handoff_id: handoff if handoff_id == "h_opt_point_plot" else None)
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_point_result(source_handoff_id="h_opt_point_plot")
    payload = json.loads(raw)

    assert payload["data"]["plot_type"] == "optimization_point_result"
    assert payload["data"]["source_handoff_id"] == "h_opt_point_plot"
    assert payload["data"]["optimal_washes"] == [
        "PE-Cyclohexane @ 120C",
        "EVOH-Dimethyl sulfoxide @ 120C",
    ]
    assert payload["data"]["plot_paths"] == [str(output_path)]


def test_plot_optimization_point_result_honors_output_dir(monkeypatch, tmp_path):
    from strap.tools import visualization

    captured: dict[str, str] = {}

    def fake_save_plot(fig, stem, **kwargs):
        captured["output_dir"] = kwargs.get("output_dir")
        return str(tmp_path / "custom" / f"{stem}.png")

    monkeypatch.setattr(visualization, "save_plot", fake_save_plot)
    raw = visualization.plot_optimization_point_result(
        optimization_result_json={
            "analysis_type": "point_optimum",
            "scenario": "A",
            "feed_composition": {"PE": 0.6, "EVOH": 0.4},
            "profit": 12_000_000.0,
            "total_cost": 7_000_000.0,
            "emissions": 8_700.0,
            "circularity_score": 0.647,
            "optimal_washes": ["PE-Cyclohexane @ 120C"],
        },
        output_dir=str(tmp_path / "custom"),
    )
    payload = json.loads(raw)

    assert captured["output_dir"] == str(tmp_path / "custom")
    assert payload["data"]["plot_paths"] == [str(tmp_path / "custom" / "optimization_point_result.png")]


def test_plot_optimization_pareto_front_reads_requested_plot_mode_from_handoff(monkeypatch, tmp_path):
    from strap import handoffs
    from strap.tools import visualization

    captured: dict[str, str] = {}
    handoff = HandoffRecord(
        handoff_id="h_opt_plot_landscape",
        scope=HandoffScope(invocation_id="inv", run_id="run", thread_id="thread"),
        producer="optimization-engineer",
        consumer="visualization-specialist",
        contract="optimization_plot_context.v1",
        status="ok",
        payload={
            "source_handoff_id": "h_opt_plot_landscape",
            "analysis_type": "pareto_front",
            "requested_plot_mode": "landscape",
            "requested_output_stem": "optimization_pareto_emissions_ldpe60_evoh20_pet20",
            "pareto_result_json": {
                "analysis_type": "pareto_front",
                "x_metric": "total_cost",
                "y_metric": "emissions",
                "points": [
                    {"point_id": 1, "total_cost": 100.0, "emissions": 50.0, "point_status": "frontier"},
                ],
                "all_feasible_points": [
                    {"raw_point_id": 1, "total_cost": 100.0, "emissions": 50.0, "point_status": "frontier"},
                    {"raw_point_id": 2, "total_cost": 120.0, "emissions": 55.0, "point_status": "dominated"},
                ],
            },
        },
        created_at="2026-04-24T15:00:00Z",
    )

    output_path = tmp_path / "optimization_pareto_landscape.png"
    monkeypatch.setattr(handoffs, "get_handoff", lambda handoff_id: handoff if handoff_id == "h_opt_plot_landscape" else None)
    def _capture_save_plot(fig, stem):
        captured["stem"] = stem
        return str(output_path)

    monkeypatch.setattr(visualization, "save_plot", _capture_save_plot)

    raw = visualization.plot_optimization_pareto_front(source_handoff_id="h_opt_plot_landscape")
    payload = json.loads(raw)

    assert payload["data"]["plot_mode"] == "landscape"
    assert payload["data"]["output_stem"] == "optimization_pareto_emissions_ldpe60_evoh20_pet20"
    assert captured["stem"] == "optimization_pareto_emissions_ldpe60_evoh20_pet20"
    assert payload["data"]["n_all_feasible_points"] == 2
    assert payload["data"]["plot_paths"] == [str(output_path)]


def test_plot_optimization_pareto_front_normalizes_freeform_landscape_plot_mode(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_landscape.png"
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "points": [{"point_id": 1, "total_cost": 100.0, "emissions": 50.0}],
            "all_feasible_points": [{"raw_point_id": 1, "total_cost": 100.0, "emissions": 50.0}],
        },
        plot_mode="Pareto landscape including all feasible points",
    )
    payload = json.loads(raw)

    assert payload["data"]["plot_mode"] == "landscape"


def test_plot_optimization_pareto_front_generates_point_ids_and_fallback_color(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_emissions.png"
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "points": [
                {"total_cost": 100.0, "emissions": 50.0},
                {"total_cost": 150.0, "emissions": 40.0},
            ],
        }
    )
    payload = json.loads(raw)

    assert payload["data"]["n_points"] == 2
    assert payload["data"]["color_by"] == "emissions"
    assert payload["data"]["point_legend"] == [
        "P1: Cost: $100 | Emissions: 50.0",
        "P2: Cost: $150 | Emissions: 40.0",
    ]


def test_plot_optimization_pareto_front_prefers_structured_design_labels(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_circularity.png"
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "circularity",
            "points": [
                {
                    "point_id": 1,
                    "total_cost": 91919.92,
                    "circularity_score": 0.6949,
                    "wash1_selection": [],
                    "wash2_selection": [],
                    "equivalent_designs": [
                        {
                            "stage1_tech": ["lf"],
                            "wash1_selection": [],
                            "wash2_selection": [],
                        }
                    ],
                },
                {
                    "point_id": 2,
                    "total_cost": 5743520.201,
                    "circularity_score": 0.8447,
                    "wash1_selection": ["PE-Tetrachloroethylene"],
                    "wash2_selection": ["EVOH-gamma-butyrolactone"],
                    "stage3_variants": ["lf"],
                    "equivalent_designs": [
                        {
                            "stage1_tech": ["st1"],
                            "stage2_tech": ["st2"],
                            "stage3_tech": ["lf"],
                            "wash1_selection": ["PE-Tetrachloroethylene"],
                            "wash2_selection": ["EVOH-gamma-butyrolactone"],
                        }
                    ],
                },
            ],
        }
    )
    payload = json.loads(raw)

    assert payload["data"]["point_legend"] == [
        "P1: Baseline: Landfill | Cost:\n    $92k | Circularity:\n    0.695",
        "P2: W1: PE-Tetrachloroethylene |\n    W2: EVOH-gamma-\n    butyrolactone | End:\n    Landfill | Cost: $5.74M\n    | Circularity: 0.845",
    ]


def test_plot_optimization_pareto_front_labels_residual_polymers(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_circularity.png"
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "circularity",
            "points": [
                {
                    "point_id": 1,
                    "total_cost": 3446515.26,
                    "circularity_score": 0.5967,
                    "wash1_selection": ["PC-Toluene"],
                    "wash2_selection": ["PS-Toluene"],
                    "recovered_polymers": ["PC", "PS"],
                    "residual_polymers": ["PP", "PVC"],
                    "residual_destination_stage": "stage3",
                    "residual_destination_tech": ["we"],
                    "equivalent_designs": [
                        {
                            "stage1_tech": ["st1"],
                            "stage2_tech": ["st2"],
                            "stage3_tech": ["we"],
                            "wash1_selection": ["PC-Toluene"],
                            "wash2_selection": ["PS-Toluene"],
                            "recovered_polymers": ["PC", "PS"],
                            "residual_polymers": ["PP", "PVC"],
                            "residual_destination_stage": "stage3",
                            "residual_destination_tech": ["we"],
                        }
                    ],
                }
            ],
        }
    )
    payload = json.loads(raw)
    legend = " ".join(payload["data"]["point_legend"][0].split()).replace("- ", "-")

    assert "W1: PC-Toluene" in legend
    assert "W2: PS-Toluene" in legend
    assert "Waste: PP, PVC -> Waste-to-energy" in legend




def test_plot_optimization_pareto_front_landscape_mode_uses_all_feasible_points(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_landscape.png"
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "circularity",
            "points": [
                {"point_id": 1, "total_cost": 100.0, "circularity_score": 0.4, "point_status": "frontier"},
                {"point_id": 2, "total_cost": 200.0, "circularity_score": 0.6, "point_status": "frontier"},
            ],
            "all_feasible_points": [
                {"raw_point_id": 1, "total_cost": 100.0, "circularity_score": 0.4, "point_status": "frontier"},
                {"raw_point_id": 2, "total_cost": 150.0, "circularity_score": 0.5, "point_status": "dominated"},
                {"raw_point_id": 3, "total_cost": 200.0, "circularity_score": 0.6, "point_status": "frontier"},
            ],
        },
        plot_mode="landscape",
    )
    payload = json.loads(raw)

    assert payload["data"]["plot_mode"] == "landscape"
    assert payload["data"]["n_frontier_points"] == 2
    assert payload["data"]["n_all_feasible_points"] == 3
    assert payload["data"]["plot_paths"] == [str(output_path)]


def test_plot_optimization_pareto_front_landscape_mode_uses_landscape_points(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_landscape.png"
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "points": [
                {"point_id": 1, "total_cost": 100.0, "emissions": 10.0, "point_status": "frontier"},
            ],
            "all_feasible_points": [
                {"raw_point_id": 1, "total_cost": 100.0, "emissions": 10.0, "point_status": "frontier"},
            ],
            "landscape_points": [
                {
                    "raw_point_id": 2,
                    "total_cost": 140.0,
                    "emissions": 20.0,
                    "point_status": "landscape_sample",
                    "wash1_selection": ["PE-Cyclohexane"],
                    "wash2_selection": [],
                },
                {
                    "raw_point_id": 3,
                    "total_cost": 180.0,
                    "emissions": 25.0,
                    "point_status": "landscape_sample",
                    "wash1_selection": ["EVOH-Methanol"],
                    "wash2_selection": [],
                },
            ],
        },
        plot_mode="landscape",
    )
    payload = json.loads(raw)

    assert payload["data"]["plot_mode"] == "landscape"
    assert payload["data"]["n_frontier_points"] == 1
    assert payload["data"]["n_all_feasible_points"] == 3
    assert payload["data"]["n_landscape_points"] == 2


def test_plot_optimization_pareto_front_loads_pareto_payload_sidecar(monkeypatch, tmp_path):
    from strap.tools import visualization

    output_path = tmp_path / "optimization_pareto_landscape.png"
    sidecar_path = tmp_path / "pareto_payload.json"
    sidecar_path.write_text(json.dumps({
        "analysis_type": "pareto_front",
        "x_metric": "total_cost",
        "y_metric": "emissions",
        "points": [
            {"point_id": 1, "total_cost": 100.0, "emissions": 10.0, "point_status": "frontier"},
        ],
        "all_feasible_points": [
            {"raw_point_id": 1, "total_cost": 100.0, "emissions": 10.0, "point_status": "frontier"},
        ],
        "landscape_points": [
            {
                "raw_point_id": 2,
                "total_cost": 140.0,
                "emissions": 20.0,
                "point_status": "landscape_sample",
                "wash1_selection": ["PE-Cyclohexane"],
            },
            {
                "raw_point_id": 3,
                "total_cost": 180.0,
                "emissions": 25.0,
                "point_status": "landscape_sample",
                "wash1_selection": ["EVOH-Methanol"],
            },
        ],
        "pareto_payload_path": str(sidecar_path),
    }))
    monkeypatch.setattr(visualization, "save_plot", lambda fig, stem: str(output_path))

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "points": [{"point_id": 1, "total_cost": 100.0, "emissions": 10.0}],
            "pareto_payload_path": str(sidecar_path),
        },
        plot_mode="landscape",
    )
    payload = json.loads(raw)

    assert payload["data"]["n_all_feasible_points"] == 3
    assert payload["data"]["n_landscape_points"] == 2
    assert payload["data"]["pareto_payload_path"] == str(sidecar_path)


def test_plot_optimization_pareto_front_uses_explicit_output_stem(monkeypatch, tmp_path):
    from strap.tools import visualization

    captured = {}

    def fake_save_plot(fig, stem, plot_type="matplotlib"):
        captured["stem"] = stem
        return str(tmp_path / f"{stem}.png")

    monkeypatch.setattr(visualization, "save_plot", fake_save_plot)

    raw = visualization.plot_optimization_pareto_front(
        pareto_result_json={
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "circularity",
            "points": [
                {"point_id": 1, "total_cost": 100.0, "circularity_score": 0.5},
            ],
        },
        plot_title="Ignored Title",
        output_stem="pareto_ldpe50_evoh40_pet10",
    )
    payload = json.loads(raw)

    assert captured["stem"] == "pareto_ldpe50_evoh40_pet10"
    assert payload["data"]["plot_paths"] == [str(tmp_path / "pareto_ldpe50_evoh40_pet10.png")]
    assert payload["data"]["output_stem"] == "pareto_ldpe50_evoh40_pet10"


def test_plot_optimization_pareto_slices_creates_combined_and_per_slice_plots(monkeypatch, tmp_path):
    from strap.tools import visualization

    stems: list[str] = []

    def fake_save_plot(fig, stem, plot_type="matplotlib"):
        stems.append(stem)
        return str(tmp_path / f"{stem}.png")

    monkeypatch.setattr(visualization, "save_plot", fake_save_plot)

    slice_payload = {
        "analysis_type": "pareto_front",
        "x_metric": "total_cost",
        "y_metric": "circularity",
        "points": [
            {"point_id": 1, "total_cost": 100.0, "circularity_score": 0.4},
            {"point_id": 2, "total_cost": 200.0, "circularity_score": 0.6},
        ],
        "all_feasible_points": [
            {"raw_point_id": 1, "total_cost": 100.0, "circularity_score": 0.4},
            {"raw_point_id": 2, "total_cost": 150.0, "circularity_score": 0.45},
        ],
        "landscape_points": [
            {"landscape_point_id": 3, "total_cost": 250.0, "circularity_score": 0.5},
        ],
    }
    raw = visualization.plot_optimization_pareto_slices(
        pareto_slices_json={
            "analysis_type": "pareto_slices",
            "x_metric": "total_cost",
            "y_metric": "circularity",
            "slice_payloads": [
                {**slice_payload, "slice_label": "ldpe20_evoh60_pet20"},
                {**slice_payload, "slice_label": "ldpe05_evoh05_pet90"},
            ],
        },
        output_stem="composition_compare",
    )
    payload = json.loads(raw)["data"]

    assert payload["plot_type"] == "optimization_pareto_slices"
    assert payload["n_slices"] == 2
    assert payload["combined_plot_path"] == str(tmp_path / "composition_compare.png")
    assert len(payload["per_slice_plot_paths"]) == 2
    assert stems[-1] == "composition_compare"
