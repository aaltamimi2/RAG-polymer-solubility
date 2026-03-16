import json
from types import SimpleNamespace
import pandas as pd
import pytest


def test_json_tool_response_infers_success_and_tool_name():
    from strap.services.tool_response_service import json_tool_response

    parsed = json.loads(
        json_tool_response("ok", {"found": True, "value": 1}, tool_name="demo_tool")
    )

    assert parsed["display"] == "ok"
    assert parsed["data"]["tool_name"] == "demo_tool"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["value"] == 1


def test_json_tool_error_sets_standard_failure_fields():
    from strap.services.tool_response_service import json_tool_error

    parsed = json.loads(
        json_tool_error(
            "bad input",
            tool_name="demo_tool",
            error_code="bad_input",
            field="solvent",
        )
    )

    assert parsed["data"]["tool_name"] == "demo_tool"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error"] == "bad input"
    assert parsed["data"]["error_code"] == "bad_input"
    assert parsed["data"]["field"] == "solvent"


def test_predict_solubility_ml_returns_standard_envelope(monkeypatch):
    from strap.tools import ml_prediction

    class FakePredictor:
        def predict(self, polymer_hsp, solvent_hsp, r0, molar_volume):
            return {
                "soluble": True,
                "probability": 0.92,
                "confidence": 0.88,
                "red": 0.57,
                "ra": 4.2,
                "r0": 7.4,
            }

    monkeypatch.setattr(ml_prediction, "get_predictor", lambda: FakePredictor())

    raw = ml_prediction.predict_solubility_ml(
        "PE",
        "Toluene",
        temperature=25.0,
        generate_visualizations=False,
    )
    parsed = json.loads(raw)

    assert parsed["data"]["tool_name"] == "predict_solubility_ml"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["polymer_name"].upper() == "PE"
    assert parsed["data"]["solvent_name"] == "Toluene"
    assert parsed["data"]["red"] == pytest.approx(0.57)
    assert "ML Solubility Prediction" in parsed["display"]


def test_get_solvent_gscore_returns_standard_envelope(monkeypatch):
    from strap.tools import safety_gsk

    monkeypatch.setattr(
        safety_gsk,
        "_lookup_gsk_exact",
        lambda _name: pd.DataFrame(
            [
                {
                    "solvent_common_name": "Toluene",
                    "classification": "Aromatics",
                    "g_score": 4.8,
                    "cas_number": "108-88-3",
                }
            ]
        ),
    )
    monkeypatch.setattr(safety_gsk, "get_logp", lambda _name: 2.7)

    raw = safety_gsk.get_solvent_gscore("Toluene")
    parsed = json.loads(raw)

    assert parsed["data"]["tool_name"] == "get_solvent_gscore"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["solvent_name"] == "Toluene"
    assert parsed["data"]["classification"] == "Aromatics"
    assert "G-Score" in parsed["display"]


def test_get_pubchem_safety_info_returns_standard_envelope(monkeypatch):
    from strap.tools import safety_pubchem

    monkeypatch.setattr(safety_pubchem, "fetch_pubchem_cid", lambda _name: 123)
    monkeypatch.setattr(
        safety_pubchem,
        "fetch_pubchem_properties",
        lambda _cid: {"MolecularFormula": "C7H8", "MolecularWeight": 92.14},
    )
    monkeypatch.setattr(
        safety_pubchem,
        "fetch_pubchem_ghs_data",
        lambda _cid: {
            "signal_word": "Danger",
            "pictograms": ["Flammable", "Health Hazard"],
            "hazard_statements": ["Highly flammable liquid and vapor"],
        },
    )

    raw = safety_pubchem.get_pubchem_safety_info("toluene")
    parsed = json.loads(raw)

    assert parsed["data"]["tool_name"] == "get_pubchem_safety_info"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["cid"] == 123
    assert parsed["data"]["compound_name"] == "toluene"
    assert "PubChem Safety Profile" in parsed["display"]


def test_compare_pubchem_safety_invalid_input_returns_standard_error():
    from strap.tools import safety_pubchem

    parsed = json.loads(safety_pubchem.compare_pubchem_safety(["toluene"]))

    assert parsed["data"]["tool_name"] == "compare_pubchem_safety"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "too_few_compounds"


def test_list_available_polymers_returns_standard_envelope():
    from strap.tools.listing import list_available_polymers

    parsed = json.loads(list_available_polymers())

    assert parsed["data"]["tool_name"] == "list_available_polymers"
    assert parsed["data"]["success"] is True
    assert "Available Polymers Summary" in parsed["display"]


def test_query_database_invalid_sql_returns_standard_error():
    from strap.tools.database_query import query_database

    parsed = json.loads(query_database("SELECT * FROM definitely_missing_table"))

    assert parsed["data"]["tool_name"] == "query_database"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "sql_query_failed"


def test_check_column_values_returns_standard_envelope():
    from strap.tools.database_query import check_column_values

    parsed = json.loads(check_column_values("common_solvents_database", "polymer", limit=5))

    assert parsed["data"]["tool_name"] == "check_column_values"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["column_name"] == "polymer"
    assert isinstance(parsed["data"]["values"], list)
    assert parsed["data"]["total_unique"] >= 1


def test_get_solvent_properties_returns_standard_envelope():
    from strap.tools.solvent_properties import get_solvent_properties

    parsed = json.loads(get_solvent_properties("toluene"))

    assert parsed["data"]["tool_name"] == "get_solvent_properties"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["found_count"] >= 1
    assert "Solvent Properties" in parsed["display"]


def test_rank_solvents_by_property_invalid_property_returns_standard_error():
    from strap.tools.solvent_properties import rank_solvents_by_property

    parsed = json.loads(rank_solvents_by_property("definitely_not_a_property"))

    assert parsed["data"]["tool_name"] == "rank_solvents_by_property"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "property_not_found"


def test_lookup_solvent_price_returns_standard_envelope():
    from strap.tools.solvent_lookup import lookup_local_solvent_market_data, lookup_solvent_price

    local = lookup_local_solvent_market_data("Toluene")
    assert local is not None
    assert local["price_usd_kg"] == pytest.approx(1.312)
    assert local["price_source"] == "60_common_solvents-TEA-LCA.csv price column"

    parsed = json.loads(lookup_solvent_price("Toluene"))

    assert parsed["data"]["tool_name"] == "lookup_solvent_price"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["solvent"] == "Toluene"
    assert parsed["data"]["price_usd_kg"] == pytest.approx(1.312)

    benzene = lookup_local_solvent_market_data("benzene")
    assert benzene is not None
    assert benzene["solvent"] == "Benzene"
    assert benzene["price_usd_kg"] == pytest.approx(1.42)
    assert benzene["cas"] == "71-43-2"
    assert benzene["gwp_kg_co2e"] is None


def test_search_google_scholar_empty_query_returns_standard_error():
    from strap.tools.literature import search_google_scholar

    parsed = json.loads(search_google_scholar(""))

    assert parsed["data"]["tool_name"] == "search_google_scholar"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "tool_reported_failure"
    assert "cannot be empty" in parsed["data"]["error"]


def test_search_literature_rag_not_ready_returns_standard_error(monkeypatch):
    from strap.tools import rag_core

    class FakeKbManager:
        def list_kbs(self):
            return []

    class FakeRagSystem:
        kb_manager = FakeKbManager()

    monkeypatch.setattr(
        rag_core,
        "get_rag_system",
        lambda: FakeRagSystem(),
    )

    parsed = json.loads(rag_core.search_literature_rag("polymer dissolution"))

    assert parsed["data"]["tool_name"] == "search_literature_rag"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "tool_reported_failure"
    assert "not ready" in parsed["data"]["error"].lower()


def test_safe_tool_wrapper_structured_exceptions_return_standard_error():
    from strap.tools._helpers import safe_tool_wrapper

    @safe_tool_wrapper(structured_output=True, tool_name="exploding_tool")
    def exploding_tool():
        raise RuntimeError("boom")

    parsed = json.loads(exploding_tool())

    assert parsed["data"]["tool_name"] == "exploding_tool"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "tool_execution_failed"
    assert parsed["data"]["exception_type"] == "RuntimeError"


def test_safe_tool_wrapper_normalizes_legacy_display_data_envelopes():
    from strap.tools._helpers import safe_tool_wrapper

    @safe_tool_wrapper(structured_output=True, tool_name="legacy_tool")
    def legacy_tool():
        return json.dumps({"display": "ok", "data": {"value": 1}})

    parsed = json.loads(legacy_tool())

    assert parsed["display"] == "ok"
    assert parsed["data"]["tool_name"] == "legacy_tool"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["value"] == 1


def test_statistical_summary_invalid_table_returns_standard_error():
    from strap.tools.statistical import statistical_summary

    parsed = json.loads(
        statistical_summary(
            table_name="definitely_missing_table",
            value_column="solubility",
        )
    )

    assert parsed["data"]["tool_name"] == "statistical_summary"
    assert parsed["data"]["success"] is False
    assert "Input validation failed" in parsed["display"]


def test_correlation_analysis_success_returns_standard_envelope(monkeypatch):
    import pandas as pd

    from strap.tools import statistical

    monkeypatch.setattr(statistical, "get_connection", lambda: object())
    monkeypatch.setattr(statistical, "_sanitize_identifier", lambda *args, **kwargs: None)
    monkeypatch.setattr(statistical, "save_plot", lambda *args, **kwargs: "/tmp/corr.png")
    monkeypatch.setattr(
        statistical,
        "_execute_query",
        lambda *args, **kwargs: {
            "success": True,
            "dataframe": pd.DataFrame(
                {
                    "temperature": [25.0, 50.0, 75.0, 100.0],
                    "solubility": [1.0, 2.0, 3.0, 4.0],
                }
            ),
        },
    )

    parsed = json.loads(
        statistical.correlation_analysis(
            table_name="common_solvents_database",
            columns="temperature,solubility",
        )
    )

    assert parsed["data"]["tool_name"] == "correlation_analysis"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["plot_filepath"] == "/tmp/corr.png"


def test_plot_solubility_vs_temperature_no_data_returns_standard_error():
    from strap.tools.visualization import plot_solubility_vs_temperature

    parsed = json.loads(
        plot_solubility_vs_temperature(
            table_name="unused",
            polymer_column="polymer",
            solvent_column="solvent",
            temperature_column="temperature",
            solubility_column="solubility",
            polymers="definitely_missing_polymer",
            solvents="definitely_missing_solvent",
        )
    )

    assert parsed["data"]["tool_name"] == "plot_solubility_vs_temperature"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "tool_reported_failure"


def test_find_optimal_separation_conditions_invalid_comparison_type_returns_standard_error():
    from strap.tools.adaptive_separation import find_optimal_separation_conditions

    parsed = json.loads(find_optimal_separation_conditions("LDPE", {"bad": "type"}))

    assert parsed["data"]["tool_name"] == "find_optimal_separation_conditions"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "invalid_comparison_polymers"


def test_analyze_selective_solubility_enhanced_target_not_found_returns_standard_error():
    from strap.tools.adaptive_separation import analyze_selective_solubility_enhanced

    parsed = json.loads(
        analyze_selective_solubility_enhanced(
            target_polymer="definitely_missing_polymer",
            comparison_polymers="LDPE,PP",
        )
    )

    assert parsed["data"]["tool_name"] == "analyze_selective_solubility_enhanced"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "target_polymer_not_found"


def test_create_separation_tree_plot_insufficient_polymers_returns_standard_error():
    from strap.tools.advanced_separation import create_separation_tree_plot

    parsed = json.loads(create_separation_tree_plot("LDPE"))

    assert parsed["data"]["tool_name"] == "create_separation_tree_plot"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "insufficient_polymers"


def test_create_separation_tree_plot_success_returns_standard_envelope(monkeypatch):
    from strap.tools import advanced_separation
    from strap.tools import separation_visualization_tools as vis_tools

    monkeypatch.setattr(
        vis_tools,
        "score_separation_sequences",
        lambda polymer_list, temperature: [
            {"sequence": polymer_list, "min_selectivity": 22.5, "steps": [{"target": polymer_list[0], "remaining": polymer_list[1:], "solvents": [{"solvent": "toluene", "selectivity": 22.5}]}]},
            {"sequence": list(reversed(polymer_list)), "min_selectivity": 10.0, "steps": []},
        ],
    )
    monkeypatch.setattr(vis_tools, "_plot_separation_sequence", lambda *args, **kwargs: "/tmp/rank1.png")
    monkeypatch.setattr(vis_tools, "_plot_topk_comparison", lambda *args, **kwargs: "/tmp/topk.png")
    monkeypatch.setattr(vis_tools, "_get_plot_url", lambda path: f"url:{path}")

    parsed = json.loads(advanced_separation.create_separation_tree_plot("PS,PE"))

    assert parsed["data"]["tool_name"] == "create_separation_tree_plot"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["rank1_plot"] == "/tmp/rank1.png"
    assert parsed["data"]["topk_plot"] == "/tmp/topk.png"
    assert "url:/tmp/rank1.png" in parsed["display"]


def test_create_selectivity_heatmap_no_data_returns_standard_error(monkeypatch):
    from strap.tools import advanced_separation
    from strap.tools import separation_visualization_tools as vis_tools

    class _DummyMatrixBuilder:
        def __init__(self, conn):
            self.conn = conn

        def build_matrix(self, *, polymers, solvents, temperature):
            return {}

    monkeypatch.setattr(vis_tools, "PolymerCompatibilityMatrix", _DummyMatrixBuilder)
    monkeypatch.setattr(vis_tools, "get_connection", lambda: object())

    parsed = json.loads(advanced_separation.create_selectivity_heatmap("PS,PE", "toluene"))

    assert parsed["data"]["tool_name"] == "create_selectivity_heatmap"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "no_data"


def test_create_selectivity_heatmap_success_returns_standard_envelope(monkeypatch):
    import strap.engines.visualization as viz_module
    from strap.tools import advanced_separation
    from strap.tools import separation_visualization_tools as vis_tools

    class _DummyMatrixBuilder:
        def __init__(self, conn):
            self.conn = conn

        def build_matrix(self, *, polymers, solvents, temperature):
            return {"PS": {"toluene": 88.0}, "PE": {"hexane": 55.0}}

    class _DummyPlotConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    class _DummyHeatmap:
        def __init__(self, config):
            self.config = config

        def create_polymer_solvent_heatmap(self, matrix):
            return "/tmp/heatmap.png"

    monkeypatch.setattr(vis_tools, "PolymerCompatibilityMatrix", _DummyMatrixBuilder)
    monkeypatch.setattr(vis_tools, "get_connection", lambda: object())
    monkeypatch.setattr(vis_tools, "get_plots_dir", lambda: "/tmp")
    monkeypatch.setattr(viz_module, "PlotConfig", _DummyPlotConfig)
    monkeypatch.setattr(vis_tools, "SelectivityHeatmap", _DummyHeatmap)

    parsed = json.loads(advanced_separation.create_selectivity_heatmap("PS,PE", "toluene,hexane"))

    assert parsed["data"]["tool_name"] == "create_selectivity_heatmap"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["filepath"] == "/tmp/heatmap.png"
    assert parsed["data"]["matrix_rows"] == 2
    assert parsed["data"]["matrix_solvents"] == ["hexane", "toluene"]


def test_create_process_flow_diagram_success_returns_standard_envelope(monkeypatch):
    from types import SimpleNamespace

    import strap.engines.visualization as viz_module
    from strap.tools import advanced_separation
    from strap.tools import separation_visualization_tools as vis_tools

    class _DummyPlotConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    class _DummyPfd:
        def __init__(self, config):
            self.config = config

        def create_flow_diagram(self, best_sequence):
            return "/tmp/pfd.png"

    async def _fake_find_best_separation(polymer_list, conn, temperature, algorithm):
        sequence = SimpleNamespace(
            steps=[
                SimpleNamespace(target_polymer="PS"),
                SimpleNamespace(target_polymer="PE"),
            ],
            unique_solvents={"toluene"},
        )
        return SimpleNamespace(best_sequence=sequence)

    monkeypatch.setattr(vis_tools, "get_connection", lambda: object())
    monkeypatch.setattr(vis_tools, "get_plots_dir", lambda: "/tmp")
    monkeypatch.setattr(viz_module, "PlotConfig", _DummyPlotConfig)
    monkeypatch.setattr(vis_tools, "ProcessFlowDiagram", _DummyPfd)
    monkeypatch.setattr(vis_tools, "find_best_separation", _fake_find_best_separation)

    parsed = json.loads(advanced_separation.create_process_flow_diagram("PS,PE"))

    assert parsed["data"]["tool_name"] == "create_process_flow_diagram"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["filepath"] == "/tmp/pfd.png"
    assert parsed["data"]["steps"] == 1
    assert parsed["data"]["solvents_used"] == ["toluene"]


def test_analyze_selective_antisolvent_precipitation_requires_two_polymers():
    from strap.tools.advanced_separation import analyze_selective_antisolvent_precipitation

    parsed = json.loads(analyze_selective_antisolvent_precipitation("LDPE"))

    assert parsed["data"]["tool_name"] == "analyze_selective_antisolvent_precipitation"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "insufficient_polymers"


def test_find_antisolvents_success_returns_standard_envelope(monkeypatch):
    import strap.solubility as solubility
    from strap.tools.advanced_separation import find_antisolvents

    monkeypatch.setattr(solubility, "get_available_solvents_for_polymer", lambda polymer: ["hexane", "toluene"])
    monkeypatch.setattr(
        solubility,
        "get_solubility",
        lambda polymer, solvent, temperature: 0.2 if solvent == "hexane" else 5.0,
    )

    parsed = json.loads(find_antisolvents("LDPE", max_solubility=1.0, temperature=25.0))

    assert parsed["data"]["tool_name"] == "find_antisolvents"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["antisolvent_count"] == 1
    assert parsed["data"]["antisolvents"][0]["solvent"] == "hexane"


def test_find_antisolvents_no_matches_returns_standard_error(monkeypatch):
    import strap.solubility as solubility
    from strap.tools.advanced_separation import find_antisolvents

    monkeypatch.setattr(solubility, "get_available_solvents_for_polymer", lambda polymer: ["toluene"])
    monkeypatch.setattr(solubility, "get_solubility", lambda polymer, solvent, temperature: 5.0)

    parsed = json.loads(find_antisolvents("LDPE", max_solubility=1.0, temperature=25.0))

    assert parsed["data"]["tool_name"] == "find_antisolvents"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "no_antisolvents_found"


def test_check_atmospheric_feasibility_failure_returns_standard_error(monkeypatch):
    from strap.tools import precipitation_analysis

    monkeypatch.setattr(precipitation_analysis, "PrecipitationAnalyzer", lambda conn: object())
    monkeypatch.setattr(precipitation_analysis, "get_connection", lambda: object())
    monkeypatch.setattr(
        precipitation_analysis,
        "build_atmospheric_feasibility_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    parsed = json.loads(precipitation_analysis.check_atmospheric_feasibility("LDPE", "EVOH"))

    assert parsed["data"]["tool_name"] == "check_atmospheric_feasibility"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "analysis_failed"


def test_plan_sequential_separation_insufficient_polymers_returns_standard_error():
    from strap.tools.advanced_separation import plan_sequential_separation

    parsed = json.loads(plan_sequential_separation("LDPE"))

    assert parsed["data"]["tool_name"] == "plan_sequential_separation"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "insufficient_polymers"


def test_plan_multiple_separation_schemes_insufficient_polymers_returns_standard_error():
    from strap.tools.advanced_separation import plan_multiple_separation_schemes

    parsed = json.loads(plan_multiple_separation_schemes("LDPE"))

    assert parsed["data"]["tool_name"] == "plan_multiple_separation_schemes"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "insufficient_polymers"


def test_analyze_integrated_separation_too_many_polymers_returns_standard_error():
    from strap.tools.advanced_separation import analyze_integrated_separation

    parsed = json.loads(analyze_integrated_separation("LDPE,HDPE,PP,PS"))

    assert parsed["data"]["tool_name"] == "analyze_integrated_separation"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "too_many_polymers"


def test_view_alternative_separation_sequence_requires_selector_returns_standard_error():
    from strap.tools.advanced_separation import view_alternative_separation_sequence

    parsed = json.loads(view_alternative_separation_sequence("LDPE,PP"))

    assert parsed["data"]["tool_name"] == "view_alternative_separation_sequence"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error_code"] == "missing_sequence_selector"


def test_get_supported_polymers_and_solvents_returns_standard_envelope():
    from strap.tools.advanced_separation import get_supported_polymers_and_solvents

    parsed = json.loads(get_supported_polymers_and_solvents())

    assert parsed["data"]["tool_name"] == "get_supported_polymers_and_solvents"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["polymer_count"] >= 1
    assert "Supported Polymers" in parsed["display"]
