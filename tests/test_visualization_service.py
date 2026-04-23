"""Tests for visualization service helpers."""

import asyncio
import json

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
