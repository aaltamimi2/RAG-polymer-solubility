from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyomo.environ as pyo
import strap.tools.waste_optimization as waste_optimization
from pyomo.repn.standard_repn import generate_standard_repn

from strap.handoff_adapters import _adapt_separation_to_optimization
from strap.handoff_models import HandoffRecord, HandoffScope
from strap.services.biosteam_service import build_single_config
from strap.tools.waste_optimization import (
    _NUMERIC_WORKBOOK_COLUMNS,
    _apply_solvent_filters,
    _build_candidate_telemetry,
    _build_candidate_landscape_route_specs,
    _build_compiled_strap_coefficient_table,
    _build_optimization_infeasible_response,
    _build_source_handoff_summary,
    _derive_filters_from_stage_candidates,
    _extract_route_candidates,
    _frame_to_pareto_points,
    _materialize_optimizer_workbook_rows,
    _map_biosteam_to_strap_row,
    _non_dominated,
    _prepare_optimization_context,
    _retry_constrained_objective,
    _run_pareto_with_route_pool,
    _run_biosteam_updates,
    _solve_objective_with_fallback,
    run_waste_management_pareto,
    run_waste_management_pareto_slices,
)
from strap.waste_management.data_loader import (
    OTHERTECH_PAPER_GWP_FALLBACK,
    derive_available_othertechs,
    derive_optimizer_sets_from_df,
    get_optimizer_default_sets,
    load_all_data,
    load_othertech_data,
)
from strap.waste_management.model import build_model, estimate_metric_upper_bound
from strap.waste_management.solver import (
    _get_solver,
    _set_objective,
    _summarize_solver_log,
    extract_results,
    solve_single,
    summarize_constraint_residuals,
)


_EXCEL_PATH = Path("src/strap/waste_management/Data for model_Scenarios.xlsx")
_CONFIG = {
    "Feed": 8000,
    "PE_f": 0.40,
    "PET_f": 0.40,
    "N6_f": 0.01,
    "EV_f": 0.19,
    "Cpe": 1173,
    "Cevoh": 8100,
    "Cwte": 259.57,
    "UB_energy": 6.26e7,
    "UB_ghg": 21303.35985408156,
    "UB_withdrawal": 14468.80855,
    "UB_waste": 1.92e6,
    "fc_t": 3.01,
    "vc_t": 0.07,
    "products_heat": 583.33,
    "products_electricity": 724.3693,
    "price_heat": 0.13,
    "price_elec": 0.0996,
    "Cgas_pw": 110,
    "ce_weights": {"energy": 0.20, "ghg": 0.20, "water": 0.20, "waste": 0.20, "subs": 0.20},
    "distances": {"strap": 0, "lf": 9.2, "we": 151, "py": 1034, "gas_er": 0, "gas_h2": 2036, "gas_h2cc": 2036},
}


def _load_model():
    data = load_all_data(
        excel_path=_EXCEL_PATH,
        strap_sheet="StrapScenario3 Units",
        other_sheet="Othertech w TransportA",
        p_strap=1.0,
    )
    return data, build_model(data, _CONFIG)


def test_load_all_data_accepts_explicit_compiled_strap_table():
    strap_df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Cyclohexane", "CAPEX [USD/yr]": 111.0},
            {"Wash number": "Wash 2", "Polymer": "PE", "Solvents": "Toluene", "CAPEX [USD/yr]": 222.0},
            {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "Dimethyl sulfoxide", "CAPEX [USD/yr]": 333.0},
            {"Wash number": "Wash 2", "Polymer": "EVOH", "Solvents": "Ethylene Glycol", "CAPEX [USD/yr]": 444.0},
        ]
    )

    data = load_all_data(
        excel_path=_EXCEL_PATH,
        strap_sheet="StrapScenario3 Units",
        other_sheet="Othertech w TransportA",
        p_strap=1.0,
        strap_df=strap_df,
    )

    assert data["strap"]["capex"][("Wash 1", "PE", "Cyclohexane")] == 111.0
    assert data["strap"]["capex"][("Wash 2", "EVOH", "Ethylene Glycol")] == 444.0
    assert data["sets"]["S_PE"] == ["Cyclohexane", "Toluene"]
    assert data["sets"]["S_EV1"] == ["Dimethyl sulfoxide"]
    assert data["sets"]["S_EV2"] == ["Ethylene Glycol"]
    assert list(data["strap_df"]["Solvents"]) == ["Cyclohexane", "Toluene", "Dimethyl sulfoxide", "Ethylene Glycol"]


def test_load_othertech_data_applies_audited_piw_fallbacks_for_unresolved_formulas():
    other, telemetry = load_othertech_data(_EXCEL_PATH, "Othertech w TransportA", return_telemetry=True)

    assert derive_available_othertechs(other) == ["lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]
    for tech in ["py", "gas_er", "gas_h2", "gas_h2cc"]:
        assert other["gwp"][tech] == OTHERTECH_PAPER_GWP_FALLBACK[tech]
    assert other["gwp"]["lf"] == 0.0816
    assert other["gwp"]["we"] == 2.38
    assert {
        (event["tech"], event["metric"], event["source"])
        for event in telemetry
        if event["event"] == "othertech_metric_fallback"
    } >= {
        ("py", "gwp", "piw_paper_julia_reference"),
        ("gas_er", "gwp", "piw_paper_julia_reference"),
        ("gas_h2", "gwp", "piw_paper_julia_reference"),
        ("gas_h2cc", "gwp", "piw_paper_julia_reference"),
        ("gas_er", "capex", "strap_model_reference"),
    }


def test_load_all_data_surfaces_othertech_fallback_telemetry():
    data = load_all_data(
        excel_path=_EXCEL_PATH,
        strap_sheet="StrapScenario3 Units",
        other_sheet="Othertech w TransportA",
        p_strap=1.0,
    )

    assert data["sets"]["othertech"] == ["lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]
    assert any(
        event.get("event") == "othertech_metric_fallback" and event.get("tech") == "gas_h2"
        for event in data["othertech_telemetry"]
    )


def test_numeric_workbook_columns_are_castable_to_float_before_row_updates():
    df = pd.read_excel(_EXCEL_PATH, sheet_name="StrapScenario3 Units").head(1).copy()
    for column in _NUMERIC_WORKBOOK_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    updated = _map_biosteam_to_strap_row(
        df.iloc[0].copy(),
        {
            "tea": {"tci_usd": 1234.5, "aoc_usd_per_yr": 678.9},
            "lca": {
                "gwp_kg_co2e_per_kg": 1.2,
                "htc_ctuh_per_kg": 0.001,
                "htnc_ctuh_per_kg": 0.002,
                "etox_ctue_per_kg": 0.003,
            },
            "operations": {
                "total_energy_mj_per_kg": 4.0,
                "water_consumed_m3_yr": 5.5,
                "waste_generated_kg_yr": 6.5,
            },
        },
        capacity_tons_yr=10,
    )
    for column, value in updated.items():
        df.at[0, column] = value

    assert df["Water consumed/discarded [m3/yr]"].dtype.kind == "f"
    assert df["Waste generated - Non Hazardous [kg/yr]"].dtype.kind == "f"
    assert df["Total Energy indirect GHG emissions (Scope 2) [metric tons CO2 equivalent (t CO2e/yr)]"].dtype.kind == "f"


def test_apply_solvent_filters_restricts_by_polymer_and_falls_back_when_needed():
    df = pd.DataFrame(
        [
            {"Polymer": "PE", "Solvents": "Toluene"},
            {"Polymer": "PE", "Solvents": "Xylene"},
            {"Polymer": "EVOH", "Solvents": "Ethylene Glycol"},
            {"Polymer": "EVOH", "Solvents": "Pyridazine"},
        ]
    )

    filtered, applied, warnings, requested = _apply_solvent_filters(
        df,
        candidate_solvents=["Toluene", "Ethylene Glycol"],
        polymer_solvent_filters_json={"PE": ["Toluene"], "EVOH": ["NotInWorkbook"]},
    )

    assert requested["global"] == ["Toluene", "Ethylene Glycol"]
    assert applied["PE"] == ["Toluene"]
    assert "EVOH" not in applied
    assert any("No EVOH solvent overlap" in warning for warning in warnings)
    assert filtered.loc[filtered["Polymer"] == "PE", "Solvents"].tolist() == ["Toluene"]
    assert filtered.loc[filtered["Polymer"] == "EVOH", "Solvents"].tolist() == ["Ethylene Glycol", "Pyridazine"]


def test_shared_optimizer_catalog_exposes_full_strap_biosteam_space_plus_legacy_rows():
    sets = get_optimizer_default_sets()

    assert "Cyclohexane" in sets["S_PE"]
    assert "styrene" in sets["S_PE"]
    assert "isopropylamine" in sets["S_EV1"]
    assert "isopropylamine" in sets["S_EV2"]
    # Keep legacy workbook candidates that are still important to routed Pareto runs.
    assert "Tetrachloroethylene" in sets["S_PE"]
    assert "Pyridazine" in sets["S_EV1"]


def test_materialize_optimizer_workbook_rows_expands_to_shared_catalog():
    df = pd.read_excel(_EXCEL_PATH, sheet_name="StrapScenario3 Units")
    expanded = _materialize_optimizer_workbook_rows(df)

    pe_wash1 = expanded.loc[
        expanded["Wash number"].eq("Wash 1") & expanded["Polymer"].eq("PE"),
        "Solvents",
    ].astype(str).tolist()
    evoh_wash1 = expanded.loc[
        expanded["Wash number"].eq("Wash 1") & expanded["Polymer"].eq("EVOH"),
        "Solvents",
    ].astype(str).tolist()

    assert "Cyclohexane" in pe_wash1
    assert "Dimethyl sulfoxide" in evoh_wash1
    assert len(expanded) > len(df)


def test_apply_solvent_filters_matches_canonicalized_shortlist_after_materialization():
    df = pd.read_excel(_EXCEL_PATH, sheet_name="StrapScenario3 Units")
    expanded = _materialize_optimizer_workbook_rows(df)

    filtered, applied, warnings, requested = _apply_solvent_filters(
        expanded,
        polymer_solvent_filters_json={"PE": ["cyclohexane"], "EVOH": ["isopropylamine"]},
    )

    assert requested["PE"] == ["cyclohexane"]
    assert applied["PE"] == ["Cyclohexane"]
    assert applied["EVOH"] == ["isopropylamine"]
    assert warnings == []
    assert filtered.loc[filtered["Polymer"] == "PE", "Solvents"].tolist() == ["Cyclohexane", "Cyclohexane"]
    assert filtered.loc[filtered["Polymer"] == "EVOH", "Solvents"].tolist() == ["isopropylamine", "isopropylamine"]


def test_derive_filters_from_stage_candidates_reads_constraint_metadata():
    polymer_filters, global_candidates, constraint_mode, fallback_policy = _derive_filters_from_stage_candidates(
        {
            "stages": [
                {
                    "stage_id": "wash_1",
                    "target_polymer": "PE",
                    "candidate_pairs": [{"polymer": "PE", "solvent": "Cyclohexane"}],
                },
                {
                    "stage_id": "wash_2",
                    "target_polymer": "EVOH",
                    "candidate_pairs": [{"polymer": "EVOH", "solvent": "N,N-Dimethylformamide"}],
                },
            ],
            "constraint_mode": "hard",
            "fallback_policy": "fail_closed",
        }
    )

    assert polymer_filters == {
        "PE": ["Cyclohexane"],
        "EVOH": ["N,N-Dimethylformamide"],
    }
    assert global_candidates == ["Cyclohexane", "N,N-Dimethylformamide"]
    assert constraint_mode == "hard"
    assert fallback_policy == "fail_closed"


def test_derive_filters_admits_bare_solvent_names_alongside_option_labels():
    """Live regression (2026-07-07 multistage stress test): temperature-suffixed
    optimizer_option labels ("Toluene @ 105C") excluded the workbook-baseline
    rows ("Toluene") from the allowlist, so when every temperature-specific
    BioSTEAM sim failed the whole polymer had zero rows and the solve went
    infeasible. The bare solvent name must ride along so baselines stay
    eligible as a fallback."""
    polymer_filters, global_candidates, _, _ = _derive_filters_from_stage_candidates(
        {
            "stages": [
                {
                    "stage_id": "wash_1",
                    "target_polymer": "PE",
                    "candidate_pairs": [
                        {"polymer": "PE", "solvent": "Toluene", "optimizer_option": "Toluene @ 105C"},
                        {"polymer": "PE", "solvent": "Heptane", "optimizer_option": "Heptane @ 95C"},
                    ],
                },
                {
                    "stage_id": "wash_2",
                    "target_polymer": "EVOH",
                    "candidate_pairs": [
                        {"polymer": "EVOH", "solvent": "Ethylene Glycol", "optimizer_option": "Ethylene Glycol @ 140C"},
                    ],
                },
            ],
            "polymer_solvent_filters": {"PE": ["Xylene"]},
        }
    )

    assert polymer_filters["PE"] == ["Toluene @ 105C", "Toluene", "Heptane @ 95C", "Heptane", "Xylene"]
    assert polymer_filters["EVOH"] == ["Ethylene Glycol @ 140C", "Ethylene Glycol"]
    assert "Toluene" in global_candidates and "Toluene @ 105C" in global_candidates
    assert "Xylene" in global_candidates  # adapter's bare-name filters merge in


def test_apply_solvent_filters_can_fail_closed_on_missing_overlap():
    df = pd.DataFrame(
        [
            {"Polymer": "PE", "Solvents": "Cyclohexane"},
            {"Polymer": "EVOH", "Solvents": "N,N-Dimethylformamide"},
        ]
    )

    filtered, applied, warnings, requested = _apply_solvent_filters(
        df,
        polymer_solvent_filters_json={"PE": ["Cyclohexane"], "EVOH": ["NotInCatalog"]},
        constraint_mode="hard",
        fallback_policy="fail_closed",
    )

    assert filtered.empty
    assert applied == {"PE": ["Cyclohexane"]}
    assert requested["EVOH"] == ["NotInCatalog"]
    assert any("fail-closed semantics" in warning for warning in warnings)






def test_prepare_optimization_context_ignores_zero_fraction_polymers(monkeypatch, tmp_path):
    captured = {}

    def fake_build_compiled(*args, **kwargs):
        captured["polymer_fraction_pcts"] = dict(kwargs.get("polymer_fraction_pcts") or {})
        return (
            __import__("pandas").DataFrame(),
            {},
            {},
            [],
            "not_requested",
            [],
            [],
        )

    def fake_scenario(*args, **kwargs):
        return "A", {"other_sheet": "Othertech w TransportA", "config": {}}

    monkeypatch.setattr("strap.tools.waste_optimization._build_compiled_strap_coefficient_table", fake_build_compiled)
    monkeypatch.setattr("strap.tools.waste_optimization._get_scenario_config", fake_scenario)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    from strap.tools.waste_optimization import _prepare_optimization_context

    context = _prepare_optimization_context(
        feed=8000,
        pe_fraction=None,
        pet_fraction=None,
        n6_fraction=None,
        evoh_fraction=None,
        feed_composition_json={"PE": 0.6, "EVOH": 0.4},
        scenario="A",
        polymer_solvent_filters_json={"PE": ["Toluene"], "EVOH": ["Pyridazine"]},
        constraint_mode="hard",
        fallback_policy="fail_closed",
    )

    assert context["recoverable_polymers"] == ["PE", "EVOH"]
    assert captured["polymer_fraction_pcts"] == {"PE": 60.0, "EVOH": 40.0}

def test_apply_solvent_filters_ranked_soft_can_fail_closed_on_missing_overlap():
    df = pd.DataFrame(
        [
            {"Polymer": "PE", "Solvents": "Cyclohexane"},
            {"Polymer": "EVOH", "Solvents": "N,N-Dimethylformamide"},
        ]
    )

    filtered, applied, warnings, requested = _apply_solvent_filters(
        df,
        polymer_solvent_filters_json={"PE": ["Cyclohexane"], "EVOH": ["NotInCatalog"]},
        constraint_mode="ranked_soft",
        fallback_policy="fail_closed",
    )

    assert filtered.empty
    assert applied == {"PE": ["Cyclohexane"]}
    assert requested["EVOH"] == ["NotInCatalog"]
    assert any("fail-closed semantics" in warning for warning in warnings)

def test_build_optimization_infeasible_response_returns_structured_envelope():
    raw = _build_optimization_infeasible_response(
        failure_reason="no_candidate_overlap",
        message="No EVOH overlap.",
        constraint_mode="hard",
        fallback_policy="fail_closed",
        requested_filters={"EVOH": ["isopropylamine"]},
        applied_filters={"PE": ["Cyclohexane"]},
        suggested_relaxation="retry_with_soft_mode",
    )

    parsed = json.loads(raw)
    assert parsed["data"]["analysis_type"] == "infeasible"
    assert parsed["data"]["failure_reason"] == "no_candidate_overlap"
    assert parsed["data"]["suggested_relaxation"] == "retry_with_soft_mode"


def test_extract_results_reports_recovered_and_residual_polymer_flows():
    model = pyo.ConcreteModel()
    model.P = pyo.Set(initialize=["PP", "PS", "PVC", "PC"])
    model.S = pyo.Set(initialize=["Toluene", "Dimethyl sulfoxide"])
    model.I = pyo.Set(initialize=["st1"])
    model.J = pyo.Set(initialize=["st2"])
    model.K = pyo.Set(initialize=["we"])
    model.x = pyo.Var(model.I, initialize=1)
    model.y = pyo.Var(model.J, initialize=1)
    model.z = pyo.Var(model.K, initialize=1)
    model.a = pyo.Var(model.P, model.S, initialize=0)
    model.b = pyo.Var(model.P, model.S, initialize=0)
    model.R = pyo.Var(model.P, initialize=0)
    model.T = pyo.Var(model.P, initialize=0)
    for var in (model.x, model.y, model.z, model.a, model.b, model.R, model.T):
        for index in var:
            var[index].fix(pyo.value(var[index]))

    model.a["PVC", "Dimethyl sulfoxide"].fix(1)
    model.b["PP", "Toluene"].fix(1)
    model.R["PVC"].fix(2000)
    model.T["PP"].fix(2000)
    for name, value in {
        "Profit": 1.0,
        "TotalEmissions": 2.0,
        "CEoverall": 3.0,
        "Sales": 4.0,
        "CapitalCost": 5.0,
        "OperationalCost": 6.0,
        "TransportationCost": 7.0,
        "E_score": 0.1,
        "GHG_score": 0.2,
        "Water_score": 0.3,
        "Waste_score": 0.4,
        "Subs_score": 0.5,
    }.items():
        setattr(model, name, pyo.Var(initialize=value))
        getattr(model, name).fix(value)
    object.__setattr__(model, "_strap_feed", 8000.0)
    object.__setattr__(
        model,
        "_strap_polymer_fractions",
        {"PP": 0.25, "PS": 0.25, "PVC": 0.25, "PC": 0.25},
    )
    object.__setattr__(model, "_strap_polymer_recovery_yields", {"PP": 0.9, "PVC": 0.8})

    result = extract_results(model)

    assert result["wash1_selection"] == ["PVC-Dimethyl sulfoxide"]
    assert result["wash2_selection"] == ["PP-Toluene"]
    assert result["recovered_polymers"] == ["PP", "PVC"]
    assert result["residual_polymers"] == ["PS", "PC"]
    assert result["residual_destination_stage"] == "stage3"
    assert result["residual_destination_tech"] == ["we"]
    assert result["recovered_mass_tpy_by_polymer"] == {"PP": 2000.0, "PVC": 2000.0}
    assert result["saleable_recovered_mass_tpy_by_polymer"] == {"PP": 1800.0, "PVC": 1600.0}
    assert result["residual_mass_tpy_by_polymer"] == {"PS": 2000.0, "PC": 2000.0}


def test_frame_to_pareto_points_normalizes_and_deduplicates_rows():
    frame = pd.DataFrame(
        [
            {"epsilon": 1.0, "profit": 10.0, "emissions": 5.0, "CE": 200000.0, "total_cost": 100.0, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Cyclohexane"], "wash2": ["EVOH-DMSO"]},
            {"epsilon": 1.0, "profit": 10.0, "emissions": 5.0, "CE": 200000.0, "total_cost": 100.0, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Cyclohexane"], "wash2": ["EVOH-DMSO"]},
            {"epsilon": 2.0, "profit": 11.0, "emissions": 4.0, "CE": 300000.0, "total_cost": 120.0, "capital_cost": 50.0, "operational_cost": 60.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Toluene"], "wash2": ["EVOH-DMF"]},
        ]
    )

    points = _frame_to_pareto_points(frame)

    assert len(points) == 2
    assert points[0]["point_id"] == 1
    assert points[0]["total_cost"] == 100.0
    assert points[0]["circularity_score"] == 0.2
    assert points[0]["n_equivalent_designs"] == 1
    assert points[0]["stage3_variants"] == ["WtE"]
    assert points[1]["point_id"] == 2


def test_frame_to_pareto_points_preserves_stage3_equivalent_design_variants():
    frame = pd.DataFrame(
        [
            {"epsilon": 1.0, "profit": 10.0, "emissions": 5.0, "CE": 200000.0, "total_cost": 100.0, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Cyclohexane"], "wash2": ["EVOH-DMSO"]},
            {"epsilon": 1.0, "profit": 10.0, "emissions": 5.0, "CE": 200000.0, "total_cost": 100.0, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["Landfill"], "wash1": ["PE-Cyclohexane"], "wash2": ["EVOH-DMSO"]},
        ]
    )

    points = _frame_to_pareto_points(frame)

    assert len(points) == 1
    assert points[0]["n_equivalent_designs"] == 2
    assert points[0]["stage3_variants"] == ["Landfill", "WtE"]
    assert len(points[0]["equivalent_designs"]) == 2


def test_frame_to_pareto_points_preserves_residual_polymer_metadata():
    frame = pd.DataFrame(
        [
            {
                "epsilon": 1.0,
                "profit": 10.0,
                "emissions": 5.0,
                "CE": 200000.0,
                "total_cost": 100.0,
                "capital_cost": 40.0,
                "operational_cost": 50.0,
                "transportation_cost": 10.0,
                "stage1": ["st1"],
                "stage2": ["st2"],
                "stage3": ["we"],
                "wash1": ["PVC-Dimethyl sulfoxide"],
                "wash2": ["PP-Toluene"],
                "recovered_polymers": ["PP", "PVC"],
                "residual_polymers": ["PS", "PC"],
                "residual_destination_stage": "stage3",
                "residual_destination_tech": ["we"],
                "recovered_mass_tpy_by_polymer": {"PP": 2000.0, "PVC": 2000.0},
                "residual_mass_tpy_by_polymer": {"PS": 2000.0, "PC": 2000.0},
            }
        ]
    )

    points = _frame_to_pareto_points(frame)

    assert points[0]["recovered_polymers"] == ["PP", "PVC"]
    assert points[0]["residual_polymers"] == ["PS", "PC"]
    assert points[0]["residual_destination_stage"] == "stage3"
    assert points[0]["residual_destination_tech"] == ["we"]
    assert points[0]["recovered_mass_tpy_by_polymer"] == {"PP": 2000.0, "PVC": 2000.0}
    assert points[0]["residual_mass_tpy_by_polymer"] == {"PS": 2000.0, "PC": 2000.0}
    assert points[0]["equivalent_designs"][0]["residual_polymers"] == ["PS", "PC"]


def test_frame_to_pareto_points_deduplicates_floating_point_metric_noise():
    frame = pd.DataFrame(
        [
            {"epsilon": 2923.180200731392, "profit": 10.0, "emissions": 2695.491661214309, "CE": 418557.2200999567, "total_cost": 1427373.5171950008, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["st1"], "stage2": ["lf"], "stage3": [], "wash1": ["PET-N,N-Dimethylformamide @ 25C"], "wash2": []},
            {"epsilon": 2749.3534662613606, "profit": 10.0, "emissions": 2695.491661214309, "CE": 418557.22009995667, "total_cost": 1427373.5171950008, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["st1"], "stage2": ["lf"], "stage3": [], "wash1": ["PET-N,N-Dimethylformamide @ 25C"], "wash2": []},
        ]
    )

    points = _frame_to_pareto_points(frame)

    assert len(points) == 1
    assert points[0]["n_equivalent_designs"] == 1
    assert points[0]["wash1_selection"] == ["PET-N,N-Dimethylformamide @ 25C"]


def test_frame_to_pareto_points_sanitizes_nan_route_metadata():
    frame = pd.DataFrame(
        [
            {
                "epsilon": 1.0,
                "profit": 10.0,
                "emissions": 5.0,
                "CE": 200000.0,
                "total_cost": 100.0,
                "capital_cost": 40.0,
                "operational_cost": 50.0,
                "transportation_cost": 10.0,
                "stage1": ["STRAP"],
                "stage2": ["Reuse"],
                "stage3": ["WtE"],
                "wash1": ["PE-Cyclohexane"],
                "wash2": ["EVOH-DMSO"],
                "route_id": float("nan"),
                "matched_route_id": float("nan"),
                "rank": float("nan"),
                "selection_origin": float("nan"),
                "wash1_origin_route_id": float("nan"),
                "wash2_origin_route_id": float("nan"),
            }
        ]
    )

    points = _frame_to_pareto_points(frame)

    assert len(points) == 1
    assert points[0]["route_id"] == ""
    assert points[0]["matched_route_id"] is None
    assert points[0]["rank"] is None
    assert points[0]["selection_origin"] == "exact_route"
    assert points[0]["wash1_origin_route_id"] is None
    assert points[0]["wash2_origin_route_id"] is None


def test_run_waste_management_pareto_defaults_to_100_points():
    signature = inspect.signature(run_waste_management_pareto)
    assert signature.parameters["n_points"].default == 100


def test_run_waste_management_pareto_slices_runs_each_composition(monkeypatch):
    calls: list[dict] = []

    def fake_pareto(**kwargs):
        calls.append(kwargs)
        composition = dict(kwargs["feed_composition_json"])
        label = f"call_{len(calls)}"
        payload = {
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "success": True,
            "x_metric": kwargs["x_metric"],
            "y_metric": kwargs["y_metric"],
            "feed": kwargs["feed"],
            "feed_composition": composition,
            "n_points_requested": kwargs["n_points"],
            "n_points_raw_feasible": 3,
            "n_points_feasible": 2,
            "points": [
                {"point_id": 1, "total_cost": 100.0, "circularity_score": 0.4},
                {"point_id": 2, "total_cost": 200.0, "circularity_score": 0.6},
            ],
            "all_feasible_points": [
                {"raw_point_id": 1, "total_cost": 100.0, "circularity_score": 0.4},
            ],
            "landscape_points": [],
        }
        payload["pareto_payload_path"] = f"/tmp/{label}.json"
        return json.dumps({"display": "ok", "data": payload})

    monkeypatch.setattr(waste_optimization, "run_waste_management_pareto", fake_pareto)
    monkeypatch.setattr(
        waste_optimization,
        "_write_pareto_slices_payload_sidecar",
        lambda payload: payload.setdefault("pareto_slices_payload_path", "/tmp/slices.json"),
    )

    raw = run_waste_management_pareto_slices(
        feed=8000,
        composition_slices_json=[
            {"label": "20/60/20", "feed_composition": {"LDPE": 0.2, "EVOH": 0.6, "PET": 0.2}},
            {"label": "5/5/90", "feed_composition": {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9}},
        ],
        scenario="A",
        x_metric="total_cost",
        y_metric="circularity",
        n_points=12,
        min_active_washes=1,
        max_active_washes=2,
    )
    payload = json.loads(raw)["data"]

    assert payload["analysis_type"] == "pareto_slices"
    assert payload["n_slices_requested"] == 2
    assert payload["n_slices_solved"] == 2
    assert payload["n_points_requested_per_slice"] == 12
    assert payload["slices"][0]["feed_composition"] == {"PE": 0.2, "EVOH": 0.6, "PET": 0.2}
    assert payload["slices"][1]["feed_composition"] == {"PE": 0.05, "EVOH": 0.05, "PET": 0.9}
    assert len(calls) == 2
    assert all(call["min_active_washes"] == 1 and call["max_active_washes"] == 2 for call in calls)


def test_parse_composition_slices_strips_quoted_polymer_keys():
    slices = waste_optimization._parse_composition_slices_json(
        [
            {"label": "bad_keys", "feed_composition": {'"PE"': 0.2, '"EVOH"': 0.6, '"PET"': 0.2}},
        ]
    )

    assert slices == [
        {
            "slice_id": "slice_1",
            "label": "bad_keys",
            "feed_composition": {"PE": 0.2, "EVOH": 0.6, "PET": 0.2},
        }
    ]


def test_run_waste_management_pareto_accepts_active_wash_bounds(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.2, "PET": 0.7, "N6": 0.0, "EVOH": 0.1},
            "requested_filters": {"PE": ["Cyclohexane"], "PET": ["N,N-Dimethylformamide"]},
            "applied_filters": {"PE": ["Cyclohexane"], "PET": ["N,N-Dimethylformamide"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "constraint_mode": "soft",
            "fallback_policy": "broaden_disclosed",
            "route_candidates": [],
        },
    )

    apply_calls: list[tuple[int | None, int | None]] = []

    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._apply_active_wash_constraints",
        lambda model, *, min_active_washes=None, max_active_washes=None: apply_calls.append(
            (min_active_washes, max_active_washes)
        ),
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: {
            "min_total_cost": {"total_cost": 100.0, "emissions": 12.0, "CE": 200000.0},
            "max_circularity": {"total_cost": 150.0, "emissions": 5.0, "CE": 400000.0},
        }[objective],
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_ce",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 200000.0,
                    "profit": 20.0,
                    "emissions": 12.0,
                    "CE": 200000.0,
                    "total_cost": 100.0,
                    "capital_cost": 40.0,
                    "operational_cost": 50.0,
                    "transportation_cost": 10.0,
                    "stage1": ["STRAP"],
                    "stage2": ["Reuse"],
                    "stage3": ["WtE"],
                    "wash1": ["PET-N,N-Dimethylformamide"],
                    "wash2": [],
                }
            ]
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        feed_composition_json={"PE": 0.2, "EVOH": 0.1, "PET": 0.7},
        y_metric="circularity",
        n_points=8,
        min_active_washes=1,
        max_active_washes=2,
    )
    payload = json.loads(raw)["data"]

    assert payload["analysis_type"] == "pareto_front"
    assert payload["n_points_feasible"] == 1
    assert apply_calls
    assert all(call == (1, 2) for call in apply_calls)


def test_summarize_solver_log_flags_original_problem_infeasibility():
    summary = _summarize_solver_log(
        """
        SCIP Status        : problem is solved [optimal solution found]
        Warning: best solution is not feasible in original problem
        violated linear constraint <lc217> by 3.0517578125e-05
        """
    )

    assert summary is not None
    assert summary["original_problem_infeasible"] is True
    assert summary["violated_constraint"]["constraint"] == "lc217"
    assert summary["violated_constraint"]["violation"] == 3.0517578125e-05


def test_solve_objective_with_fallback_retries_after_unverified_scip_result(monkeypatch):
    calls: list[tuple[str | None, dict | None]] = []

    class DummyModel:
        def clone(self):
            return DummyModel()

    def fake_solve_single(model, objective, solver_name="gurobi", solver_options=None):
        calls.append((solver_name, solver_options))
        if solver_name == "scip" and solver_options is None:
            return None
        if solver_name == "scip" and solver_options in (
            {"presolving/maxrounds": 0},
            {"presolving/maxrounds": 0, "randomization/randomseedshift": 1},
        ):
            return {"total_cost": 789.0}
        raise AssertionError(f"Unexpected solver path: {(solver_name, solver_options)}")

    monkeypatch.setattr("strap.tools.waste_optimization.solve_single", fake_solve_single)
    debug_iter = iter([
        {"accepted": False, "rejection_reason": {"type": "original_problem_infeasible"}},
        {"accepted": True, "label": "max_circularity"},
        {"accepted": True, "label": "max_circularity"},
    ])
    monkeypatch.setattr(
        "strap.tools.waste_optimization.consume_last_solver_debug",
        lambda: next(debug_iter),
    )

    result, debug_attempts = _solve_objective_with_fallback(DummyModel(), "max_circularity", return_debug=True)

    assert result == {"total_cost": 789.0}
    assert calls == [
        ("scip", None),
        ("scip", {"presolving/maxrounds": 0}),
        ("scip", {"presolving/maxrounds": 0, "randomization/randomseedshift": 1}),
    ]
    assert len(debug_attempts) == 3
    assert debug_attempts[0]["accepted"] is False
    assert debug_attempts[1]["accepted"] is True
    assert debug_attempts[2]["accepted"] is True


def test_solve_objective_with_fallback_captures_legacy_solver_stdout(monkeypatch, capsys):
    def fake_solve_single(model, objective, solver_name="gurobi", solver_options=None):
        print("legacy solver table")
        return {"profit": 123.0}

    monkeypatch.setattr("strap.tools.waste_optimization.solve_single", fake_solve_single)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.consume_last_solver_debug",
        lambda: {"accepted": True, "label": "max_profit"},
    )

    result, debug_attempts = _solve_objective_with_fallback(object(), "max_profit", return_debug=True)

    assert result == {"profit": 123.0}
    assert capsys.readouterr().out == ""
    assert debug_attempts[0]["solver_stdout"] == "legacy solver table"


def test_run_waste_management_optimization_discloses_solver_retry(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **kwargs: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": kwargs["scenario"],
            "fractions": {"PE": 0.6, "EVOH": 0.4},
            "requested_filters": {"global": [], "PE": ["Heptane"], "EVOH": ["Ethylene Glycol"]},
            "applied_filters": {"PE": ["Heptane"], "EVOH": ["Ethylene Glycol"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "constraint_mode": kwargs.get("constraint_mode") or "soft",
            "fallback_policy": kwargs.get("fallback_policy") or "broaden_disclosed",
            "route_pool_mode": "exact",
            "route_candidates": [],
            "has_typed_handoff": False,
            "strap_table_rows": 4,
            "simulation_failures": [],
            "simulation_skips": [],
        },
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr("strap.tools.waste_optimization._apply_active_wash_constraints", lambda *args, **kwargs: None)

    def fake_solve_with_debug(model, objective, **kwargs):
        assert kwargs["return_debug"] is True
        return (
            {
                "profit": 12_345.0,
                "emissions": 678.0,
                "CE": 250_000.0,
                "total_cost": 1_000.0,
                "capital_cost": 100.0,
                "operational_cost": 800.0,
                "transportation_cost": 100.0,
                "wash1_selection": ["PE-Heptane"],
                "wash2_selection": ["EVOH-Ethylene Glycol"],
                "stage1_tech": ["st1"],
                "stage2_tech": ["st2"],
                "stage3_tech": ["lf"],
            },
            [
                {
                    "attempt_index": 1,
                    "accepted": False,
                    "termination_condition": "optimal",
                    "rejection_reason": {"type": "original_problem_infeasible"},
                },
                {
                    "attempt_index": 2,
                    "accepted": True,
                    "termination_condition": "optimal",
                    "residual_summary": {"count": 0, "max_residual": 0.0, "violations": []},
                },
            ],
        )

    monkeypatch.setattr("strap.tools.waste_optimization._solve_objective_with_fallback", fake_solve_with_debug)

    raw = waste_optimization.run_waste_management_optimization(
        feed=8000,
        feed_composition_json={"PE": 0.6, "EVOH": 0.4},
        scenario="A",
        objective="max_profit",
    )
    parsed = json.loads(raw)

    assert "Solver retry note" in parsed["display"]
    assert parsed["data"]["solver_debug"]["n_rejected_attempts"] == 1
    assert "original_problem_infeasible" in parsed["data"]["solver_retry_note"]


def test_run_waste_management_pareto_emits_solver_debug(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.2, "PET": 0.7, "N6": 0.0, "EVOH": 0.1},
            "requested_filters": {"PE": ["Cyclohexane"], "PET": ["N,N-Dimethylformamide"]},
            "applied_filters": {"PE": ["Cyclohexane"], "PET": ["N,N-Dimethylformamide"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "constraint_mode": "soft",
            "fallback_policy": "broaden_disclosed",
            "route_candidates": [],
        },
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: (
            {
                "min_total_cost": {"total_cost": 100.0, "emissions": 12.0, "CE": 200000.0},
                "max_circularity": {"total_cost": 150.0, "emissions": 5.0, "CE": 400000.0},
            }[objective],
            [{"accepted": True, "label": objective}],
        ),
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_ce",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 200000.0,
                    "profit": 20.0,
                    "emissions": 12.0,
                    "CE": 200000.0,
                    "total_cost": 100.0,
                    "capital_cost": 40.0,
                    "operational_cost": 50.0,
                    "transportation_cost": 10.0,
                    "stage1": ["STRAP"],
                    "stage2": ["Reuse"],
                    "stage3": ["WtE"],
                    "wash1": ["PET-N,N-Dimethylformamide"],
                    "wash2": [],
                }
            ]
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        feed_composition_json={"PE": 0.2, "EVOH": 0.1, "PET": 0.7},
        y_metric="circularity",
        n_points=8,
        min_active_washes=1,
        max_active_washes=2,
    )
    payload = json.loads(raw)["data"]

    assert payload["solver_debug"]["anchors"]["min_total_cost"][0]["accepted"] is True
    assert payload["solver_debug"]["anchors"]["max_circularity"][0]["solve_attempts"][0]["label"] == "max_circularity"
    assert isinstance(payload["solver_debug"]["sweep_attempts"], list)


def test_run_waste_management_pareto_returns_typed_front(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.5, "PET": 0.0, "N6": 0.0, "EVOH": 0.5},
            "requested_filters": {"PE": ["Cyclohexane"]},
            "applied_filters": {"PE": ["Cyclohexane"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "constraint_mode": "soft",
            "fallback_policy": "broaden_disclosed",
        },
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())

    def fake_solve_single(model, sense, solver_name="gurobi"):
        if sense == "min_total_cost":
            return {"total_cost": 100.0, "emissions": 8.0, "CE": 200000.0}
        if sense == "min_emissions":
            return {"total_cost": 150.0, "emissions": 5.0, "CE": 300000.0}
        raise AssertionError(f"Unexpected objective {sense}")

    monkeypatch.setattr("strap.tools.waste_optimization.solve_single", fake_solve_single)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"epsilon": 8.0, "profit": 20.0, "emissions": 8.0, "CE": 200000.0, "total_cost": 100.0, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Cyclohexane"], "wash2": ["EVOH-DMSO"]},
                {"epsilon": 5.0, "profit": 18.0, "emissions": 5.0, "CE": 300000.0, "total_cost": 150.0, "capital_cost": 60.0, "operational_cost": 75.0, "transportation_cost": 15.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Toluene"], "wash2": ["EVOH-DMF"]},
            ]
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        pe_fraction=0.5,
        pet_fraction=0.0,
        n6_fraction=0.0,
        evoh_fraction=0.5,
        y_metric="emissions",
        n_points=4,
    )
    parsed = json.loads(raw)

    assert parsed["data"]["analysis_type"] == "pareto_front"
    assert parsed["data"]["x_metric"] == "total_cost"
    assert parsed["data"]["y_metric"] == "emissions"
    assert parsed["data"]["n_points_feasible"] == 2
    assert parsed["data"]["candidate_summary"]["status"] == "applied"
    assert parsed["data"]["frontier_summary"]["n_distinct_stage3_techs"] == 1
    assert parsed["data"]["frontier_summary"]["distinct_stage3_techs"] == ["WtE"]


def test_estimate_metric_upper_bound_is_data_driven():
    data, _ = _load_model()

    energy_upper = estimate_metric_upper_bound(data["strap"], data["other"], "total_energy", _CONFIG["Feed"])
    ghg_upper = estimate_metric_upper_bound(data["strap"], data["other"], "direct_ghg", _CONFIG["Feed"])

    assert energy_upper > _CONFIG["UB_energy"]
    assert ghg_upper > 0


def test_model_uses_tighter_big_m_coefficients_than_legacy_scaling():
    _, model = _load_model()

    energy_bound = generate_standard_repn(model.e_ub.body)
    energy_lower = generate_standard_repn(model.e_lb.body)

    energy_bound_max = max(abs(float(value)) for value in energy_bound.linear_coefs)
    energy_lower_max = max(abs(float(value)) for value in energy_lower.linear_coefs)

    assert energy_bound_max < 1e12
    assert energy_lower_max < 1e9


def test_solver_residual_summary_filters_tiny_numerical_noise():
    _, model = _load_model()
    _set_objective(model, "max_profit")
    result = _get_solver("scip").solve(model, tee=False, load_solutions=False)

    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    model.solutions.load_from(result)
    residuals = summarize_constraint_residuals(model)

    assert residuals["count"] == 0


def test_solver_residual_summary_accepts_sub_micro_absolute_noise():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=1.0000005)
    model.c = pyo.Constraint(expr=model.x <= 1.0)

    residuals = summarize_constraint_residuals(model)

    assert residuals["count"] == 0


def test_solve_single_rejects_postsolve_constraint_violations(monkeypatch):
    class DummySolutions:
        def load_from(self, result):
            self.result = result

    class DummyModel:
        def __init__(self):
            self.solutions = DummySolutions()

    class DummySolver:
        def solve(self, model, tee=True, load_solutions=False):
            return SimpleNamespace(
                solver=SimpleNamespace(termination_condition=pyo.TerminationCondition.optimal),
                solution=[SimpleNamespace(status="feasible")],
            )

    monkeypatch.setattr("strap.waste_management.solver._set_objective", lambda *args, **kwargs: None)
    monkeypatch.setattr("strap.waste_management.solver._get_solver", lambda *args, **kwargs: DummySolver())
    monkeypatch.setattr(
        "strap.waste_management.solver.summarize_constraint_residuals",
        lambda model: {
            "count": 1,
            "violations": [{"constraint": "bad_con", "residual": 1e-4, "tolerance": 1e-7}],
        },
    )
    monkeypatch.setattr(
        "strap.waste_management.solver.extract_results",
        lambda model: {"total_cost": 1.0},
    )

    assert solve_single(DummyModel(), "min_total_cost", solver_name="scip") is None


def _biosteam_columns_template() -> dict[str, float]:
    return {column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS}


def test_run_biosteam_updates_treats_zero_tea_as_sim_failure(monkeypatch):
    """Rows are validated against post-update economics. A row is dropped if two
    or more of {CAPEX, OPEX, GWP} land at zero after BioSTEAM's update pass, so
    zero-metric ghost rows can't dominate the MINLP. Rows where BioSTEAM returns
    null TEA but workbook baseline supplied CAPEX/OPEX, or where only one of
    the three fields is zero, stay in the solve."""
    df = pd.DataFrame([
        # Row 1: full sim — all three fields become positive. Keep.
        {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "RealSolvent", **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS}},
        # Row 2: BioSTEAM returns null TEA (common for this env), but the row starts
        # with workbook baseline CAPEX/OPEX values. Keep — the MINLP has real data.
        {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "BaselineOnly",
         **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
         "CAPEX [USD/yr]": 500_000.0, "OPEX [USD/yr]": 50_000.0},
        # Row 3: Materialized ghost — starts at zero and BioSTEAM returns null TEA + null GWP.
        # Post-update row has CAPEX=0, OPEX=0, GWP=0 → drop.
        {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "GhostRow", **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS}},
    ])

    results_by_solvent = {
        "RealSolvent": {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.5},
            "operations": {"total_energy_mj_per_kg": 2.0},
        },
        "BaselineOnly": {
            "success": True,
            # null TEA — BioSTEAM couldn't compute economics this run
            "tea": {"tci_usd": None, "aoc_usd_per_yr": None},
            "lca": {"gwp_kg_co2e_per_kg": 0.9},
            "operations": {"total_energy_mj_per_kg": 2.5},
        },
        "GhostRow": {
            "success": True,
            # null TEA AND null GWP → nothing populates, row stays empty
            "tea": {"tci_usd": None, "aoc_usd_per_yr": None},
            "lca": {"gwp_kg_co2e_per_kg": None},
            "operations": {},
        },
    }

    def fake_build_config(**kwargs):
        return {"solvent": kwargs["solvent"]}

    def fake_run_sim(config):
        return results_by_solvent[config["solvent"]]

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_config)
    monkeypatch.setattr("strap.tools.waste_optimization.run_single_simulation", fake_run_sim)

    updated_df, failures, simulation_skips = _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    surviving_solvents = set(updated_df["Solvents"].astype(str))
    assert surviving_solvents == {"RealSolvent", "BaselineOnly"}, (
        f"expected real + baseline-only to survive, got: {surviving_solvents}"
    )

    # The GhostRow must land in failed_sims with a post-update-economics reason.
    failure_map = {(f["polymer"], f["solvent"]): f["reason"] for f in failures}
    assert ("EVOH", "GhostRow") in failure_map
    assert "row_missing_economics" in failure_map[("EVOH", "GhostRow")]
    # CAPEX + OPEX + GWP all ended up zero
    assert "CAPEX<=0" in failure_map[("EVOH", "GhostRow")]
    assert "OPEX<=0" in failure_map[("EVOH", "GhostRow")]
    assert simulation_skips == []


def test_run_biosteam_updates_drops_failed_sim_rows_and_reports_pairs(monkeypatch):
    """G1: failed-sim rows must not survive as zero-metric ghosts in the solve."""

    df = pd.DataFrame([
        {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "GoodSolvent", **_biosteam_columns_template()},
        {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "BadSolvent", **_biosteam_columns_template()},
        {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "BadEvohSolvent", **_biosteam_columns_template()},
    ])

    def fake_config(**kwargs):
        return {"solvent": kwargs["solvent"]}

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_config)

    def fake_run_single_simulation(config):
        if config["solvent"] == "GoodSolvent":
            return {
                "success": True,
                "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
                "lca": {"gwp_kg_co2e_per_kg": 0.5},
                "operations": {"total_energy_mj_per_kg": 2.0},
            }
        return {"success": False}

    monkeypatch.setattr("strap.tools.waste_optimization.run_single_simulation", fake_run_single_simulation)

    updated_df, failures, simulation_skips = _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    # Only the "GoodSolvent" row should survive
    assert len(updated_df) == 1
    assert updated_df.iloc[0]["Solvents"] == "GoodSolvent"

    failed_pairs = {(item["polymer"], item["solvent"]) for item in failures}
    assert ("PE", "BadSolvent") in failed_pairs
    assert ("EVOH", "BadEvohSolvent") in failed_pairs
    assert simulation_skips == []


def test_materialize_optimizer_workbook_rows_tags_row_provenance():
    df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Heptane", **_biosteam_columns_template()},
            {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "Ethylene Glycol", **_biosteam_columns_template()},
            {"Wash number": "Wash 2", "Polymer": "PE", "Solvents": "Heptane", **_biosteam_columns_template()},
            {"Wash number": "Wash 2", "Polymer": "EVOH", "Solvents": "Ethylene Glycol", **_biosteam_columns_template()},
        ]
    )

    expanded = _materialize_optimizer_workbook_rows(df)

    assert "coefficient_source" in expanded.columns
    original_sources = expanded.loc[
        expanded["Solvents"].isin(["Heptane", "Ethylene Glycol"]),
        "coefficient_source",
    ].astype(str).unique().tolist()
    assert "workbook_baseline" in original_sources
    assert "materialized_clone" in expanded["coefficient_source"].astype(str).tolist()


def test_materialize_optimizer_workbook_rows_respects_allowlist():
    df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Heptane", **_biosteam_columns_template()},
            {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "Ethylene Glycol", **_biosteam_columns_template()},
            {"Wash number": "Wash 2", "Polymer": "PE", "Solvents": "Heptane", **_biosteam_columns_template()},
            {"Wash number": "Wash 2", "Polymer": "EVOH", "Solvents": "Ethylene Glycol", **_biosteam_columns_template()},
        ]
    )

    expanded = _materialize_optimizer_workbook_rows(
        df,
        allowed_solvents_by_slot={
            ("Wash 1", "PE"): ["Cyclohexane"],
            ("Wash 2", "PE"): ["Cyclohexane"],
            ("Wash 1", "EVOH"): ["Dimethyl sulfoxide"],
            ("Wash 2", "EVOH"): ["gamma-butyrolactone"],
        },
    )

    pe_solvents = expanded.loc[expanded["Polymer"].eq("PE"), "Solvents"].astype(str).tolist()
    evoh_wash1 = expanded.loc[
        expanded["Wash number"].eq("Wash 1") & expanded["Polymer"].eq("EVOH"),
        "Solvents",
    ].astype(str).tolist()

    assert "Cyclohexane" in pe_solvents
    assert "Methylcyclohexane" not in pe_solvents
    assert "Dimethyl sulfoxide" in evoh_wash1
    assert "Pyridazine" not in evoh_wash1


def test_build_single_config_sanitizes_nonalpha_solvent_names_for_biosteam():
    config = build_single_config(
        solvent="2,3-Dihydropyran",
        target_plastic="PE",
        target_plastic_percent=60.0,
        processing_capacity=1000.0,
    )

    assert config["solvent"] == "Dihydropyran"
    assert config["solvent_input_name"] == "2,3-Dihydropyran"


def test_build_single_config_sanitizes_locant_prefixed_alcohol_names():
    config = build_single_config(
        solvent="1-butanol",
        target_plastic="PE",
        target_plastic_percent=60.0,
        processing_capacity=1000.0,
    )

    assert config["solvent"] == "Butanol"
    assert config["solvent_input_name"] == "1-butanol"


def test_run_biosteam_updates_skips_only_workbook_baseline_rows_with_sufficient_data(monkeypatch):
    df = pd.DataFrame([
        {
            "Wash number": "Wash 1",
            "Polymer": "PE",
            "Solvents": "BaselineBacked",
            **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
            "CAPEX [USD/yr]": 500_000.0,
            "OPEX [USD/yr]": 50_000.0,
            "GWP [tonCO2e/yr]": 100.0,
            "coefficient_source": "workbook_baseline",
        },
        {
            "Wash number": "Wash 1",
            "Polymer": "PE",
            "Solvents": "MaterializedNeedsSim",
            **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
            "CAPEX [USD/yr]": 500_000.0,
            "OPEX [USD/yr]": 50_000.0,
            "GWP [tonCO2e/yr]": 100.0,
            "coefficient_source": "materialized_clone",
        },
    ])

    captured_solvents: list[str] = []

    def fake_build_config(**kwargs):
        captured_solvents.append(kwargs["solvent"])
        return {"solvent": kwargs["solvent"]}

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_config)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config: {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.5},
            "operations": {"total_energy_mj_per_kg": 2.0},
        },
    )

    updated_df, failures, simulation_skips = _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    assert failures == []
    assert captured_solvents == ["MaterializedNeedsSim"]
    assert len(simulation_skips) == 1
    assert simulation_skips[0]["polymer"] == "PE"
    assert simulation_skips[0]["solvent"] == "BaselineBacked"
    assert simulation_skips[0]["reason"] == "baseline_sufficient"
    source_map = {
        str(row["Solvents"]): str(row["coefficient_source"])
        for _, row in updated_df.iterrows()
    }
    assert source_map["BaselineBacked"] == "workbook_baseline"
    assert source_map["MaterializedNeedsSim"] == "biosteam_updated"


def test_run_biosteam_updates_reuses_cached_simulations_across_calls(monkeypatch):
    waste_optimization._BIOSTEAM_SIM_CACHE.clear()
    df = pd.DataFrame([
        {
            "Wash number": "Wash 1",
            "Polymer": "PE",
            "Solvents": "NeedsSim",
            **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
            "coefficient_source": "materialized_clone",
        },
    ])

    calls = {"count": 0}

    def fake_run_single_simulation(config):
        calls["count"] += 1
        return {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.5},
            "operations": {"total_energy_mj_per_kg": 2.0},
        }

    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        fake_run_single_simulation,
    )

    updated_df_1, failures_1, simulation_skips_1 = _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )
    updated_df_2, failures_2, simulation_skips_2 = _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    assert calls["count"] == 1
    assert failures_1 == []
    assert failures_2 == []
    assert simulation_skips_1 == []
    assert simulation_skips_2 == []
    assert updated_df_1["CAPEX [USD/yr]"].iloc[0] > 0
    assert updated_df_2["CAPEX [USD/yr]"].iloc[0] > 0


def test_run_biosteam_updates_applies_runtime_denylist_without_runner_call(monkeypatch):
    waste_optimization._BIOSTEAM_SIM_CACHE.clear()
    df = pd.DataFrame([
        {
            "Wash number": "Wash 1",
            "Polymer": "PE",
            "Solvents": "hexamethylphosphoramide",
            **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
            "coefficient_source": "materialized_clone",
        },
    ])

    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config: (_ for _ in ()).throw(AssertionError("denylisted solvent should not be simulated")),
    )

    updated_df, failures, simulation_skips = _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    assert updated_df.empty
    assert simulation_skips == []
    assert len(failures) == 1
    assert failures[0]["polymer"] == "PE"
    assert failures[0]["solvent"] == "hexamethylphosphoramide"
    assert failures[0]["failure_class"] == "undefined_chemical_alias"
    assert failures[0]["source"] == "runtime_denylist"


def test_build_compiled_strap_coefficient_table_surfaces_sim_failure_warnings(monkeypatch, tmp_path):
    """G1: sim failures that eliminate a polymer must appear as filter warnings."""

    # Minimal workbook fixture: two rows, one PE + one EVOH.
    fake_df = pd.DataFrame([
        {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Heptane", **_biosteam_columns_template()},
        {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "Pyridazine", **_biosteam_columns_template()},
    ])
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: fake_df.copy())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._materialize_optimizer_workbook_rows",
        lambda df, **kwargs: df.copy(),
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", lambda **_: {})
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config: {"success": False},
    )

    df, _req, _app, warnings, status, failures, skips = _build_compiled_strap_coefficient_table(
        tmp_path / "dummy.xlsx",
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    assert df.empty
    assert failures, "Expected simulation_failures to be recorded"
    assert skips == []
    assert any("BioSTEAM simulation failed" in w for w in warnings)
    # Both polymers lost all rows — warnings should flag each
    assert any("All PE candidate rows" in w for w in warnings)
    assert any("All EVOH candidate rows" in w for w in warnings)


def test_prepare_optimization_context_typed_path_overrides_legacy_with_warning(monkeypatch):
    """C1/C3: typed stage_candidates_json wins; legacy overrides produce a warning."""

    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        captured["kwargs"] = kwargs
        return (
            pd.DataFrame([{"Polymer": "PE", "Solvents": "Heptane"}]),
            {"PE": ["Heptane"]},
            {"PE": ["Heptane"]},
            [],
            "applied",
            [],
            [],
        )

    monkeypatch.setattr(
        "strap.tools.waste_optimization._build_compiled_strap_coefficient_table",
        fake_build,
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.load_all_data",
        lambda **kwargs: {"strap": {}, "other": {}, "sets": {}},
    )

    stage_candidates = {
        "schema_version": "1.0",
        "workflow_scope": "multi_stage",
        "route_id": "r1",
        "constraint_mode": "hard",
        "fallback_policy": "fail_closed",
        "operating_constraints": {},
        "stages": [
            {"stage_id": "wash_1", "stage_kind": "selective_dissolution", "target_polymer": "PE", "candidate_pairs": [{"polymer": "PE", "solvent": "Heptane"}]},
        ],
        "candidate_pairs": [{"polymer": "PE", "solvent": "Heptane", "stage_id": "wash_1"}],
        "polymer_solvent_filters": {"PE": ["Heptane"]},
        "candidate_solvents": ["Heptane"],
    }

    context = _prepare_optimization_context(
        feed=1000,
        pe_fraction=1.0,
        pet_fraction=0.0,
        n6_fraction=0.0,
        evoh_fraction=0.0,
        scenario="A",
        candidate_solvents=["Legacy1"],
        polymer_solvent_filters_json={"PE": ["Legacy2"]},
        stage_candidates_json=stage_candidates,
        constraint_mode="soft",
        fallback_policy="broaden_disclosed",
    )

    # Typed handoff's hard+fail_closed must win — and because the stub build
    # returned warnings=[], the context must NOT trigger an infeasible
    # short-circuit; it must return a normal context with the precedence
    # warning appended.
    assert "infeasible_response" not in context
    assert any("Typed stage_candidates_json took precedence" in w for w in context["filter_warnings"])
    assert context["constraint_mode"] == "hard"
    assert context["fallback_policy"] == "fail_closed"
    # The typed handoff's candidate values are what reached the build call,
    # not the legacy overrides.
    assert captured["kwargs"]["candidate_solvents"] == ["Heptane"]
    assert captured["kwargs"]["polymer_solvent_filters_json"] == {"PE": ["Heptane"]}


def test_prepare_optimization_context_fail_closed_tolerates_partial_simulation_failures(monkeypatch):
    def fake_build(*args, **kwargs):
        return (
            pd.DataFrame([
                {"Polymer": "PE", "Solvents": "Cyclohexane"},
                {"Polymer": "EVOH", "Solvents": "Dimethyl sulfoxide"},
            ]),
            {"PE": ["Cyclohexane", "gvl"], "EVOH": ["Dimethyl sulfoxide", "gvl"]},
            {"PE": ["Cyclohexane"], "EVOH": ["Dimethyl sulfoxide"]},
            [
                "BioSTEAM simulation failed (or returned incomplete TEA/GWP) for these candidate pairs and the corresponding rows were dropped to prevent zero-metric ghost rows from polluting the optimization solve: PE+gvl (undefined_chemical_alias:chemical 'gvl' not recognized), EVOH+gvl (undefined_chemical_alias:chemical 'gvl' not recognized).",
            ],
            "partially_applied_with_fallback",
            [
                {"polymer": "PE", "solvent": "gvl", "failure_class": "undefined_chemical_alias", "reason": "chemical 'gvl' not recognized", "source": "biosteam_simulation"},
                {"polymer": "EVOH", "solvent": "gvl", "failure_class": "undefined_chemical_alias", "reason": "chemical 'gvl' not recognized", "source": "biosteam_simulation"},
            ],
            [],
        )

    monkeypatch.setattr(
        "strap.tools.waste_optimization._build_compiled_strap_coefficient_table",
        fake_build,
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.load_all_data",
        lambda **kwargs: {"strap": {}, "other": {}, "sets": {}},
    )

    context = _prepare_optimization_context(
        feed=1000,
        pe_fraction=0.6,
        pet_fraction=0.0,
        n6_fraction=0.0,
        evoh_fraction=0.4,
        scenario="A",
        polymer_solvent_filters_json={"PE": ["Cyclohexane", "gvl"], "EVOH": ["Dimethyl sulfoxide", "gvl"]},
        constraint_mode="ranked_soft",
        fallback_policy="fail_closed",
    )

    assert "infeasible_response" not in context
    assert context["simulation_failures"]
    assert any(item["solvent"] == "gvl" for item in context["simulation_failures"])


def test_prepare_optimization_context_allows_explicit_route_pool_override(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        captured["kwargs"] = kwargs
        return (
            pd.DataFrame([{"Polymer": "PE", "Solvents": "Heptane"}]),
            {"PE": ["Heptane"]},
            {"PE": ["Heptane"]},
            [],
            "applied",
            [],
            [],
        )

    monkeypatch.setattr(
        "strap.tools.waste_optimization._build_compiled_strap_coefficient_table",
        fake_build,
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.load_all_data",
        lambda **kwargs: {"strap": {}, "other": {}, "sets": {}},
    )

    stage_candidates = {
        "schema_version": "1.0",
        "workflow_scope": "multi_stage",
        "route_id": "r1",
        "constraint_mode": "ranked_soft",
        "fallback_policy": "broaden_disclosed",
        "route_pool_mode": "exact",
        "operating_constraints": {},
        "stages": [
            {"stage_id": "wash_1", "stage_kind": "selective_dissolution", "target_polymer": "PE", "candidate_pairs": [{"polymer": "PE", "solvent": "Heptane"}]},
        ],
        "candidate_pairs": [{"polymer": "PE", "solvent": "Heptane", "stage_id": "wash_1"}],
        "polymer_solvent_filters": {"PE": ["Heptane"]},
        "candidate_solvents": ["Heptane"],
    }

    context = _prepare_optimization_context(
        feed=1000,
        pe_fraction=1.0,
        pet_fraction=0.0,
        n6_fraction=0.0,
        evoh_fraction=0.0,
        scenario="A",
        stage_candidates_json=stage_candidates,
        route_pool_mode="slot_independent",
    )

    assert "infeasible_response" not in context
    assert context["route_pool_mode"] == "slot_independent"
    assert any("Explicit route_pool_mode override took precedence" in w for w in context["filter_warnings"])
    assert captured["kwargs"]["candidate_solvents"] == ["Heptane"]
    assert captured["kwargs"]["polymer_solvent_filters_json"] == {"PE": ["Heptane"]}


def test_prepare_optimization_context_slot_independent_keeps_routes_for_provenance_only(monkeypatch):
    captured = {}

    def fake_build(*args, **kwargs):
        captured["kwargs"] = kwargs
        return (
            pd.DataFrame([{"Polymer": "PE", "Solvents": "Heptane"}]),
            {"PE": ["Heptane"]},
            {"PE": ["Heptane"]},
            [],
            "applied",
            [],
            [],
        )

    monkeypatch.setattr(
        "strap.tools.waste_optimization._build_compiled_strap_coefficient_table",
        fake_build,
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.load_all_data",
        lambda **kwargs: {"strap": {}, "other": {}, "sets": {}},
    )

    stage_candidates = {
        "schema_version": "1.1",
        "workflow_scope": "multi_stage",
        "route_id": "r1",
        "constraint_mode": "ranked_soft",
        "fallback_policy": "fail_closed",
        "route_pool_mode": "slot_independent",
        "candidate_pairs": [
            {"polymer": "PE", "solvent": "Heptane", "stage_id": "candidate_pool_pe"},
            {"polymer": "EVOH", "solvent": "Methanol", "stage_id": "candidate_pool_evoh"},
        ],
        "route_candidates": [
            {
                "route_id": "route_1",
                "rank": 1,
                "sequence": ["PE", "EVOH"],
                "polymer_solvent_map": {"PE": "Heptane", "EVOH": "Methanol"},
            }
        ],
        "stages": [
            {"stage_id": "candidate_pool_pe", "stage_kind": "selective_dissolution", "target_polymer": "PE", "candidate_pairs": [{"polymer": "PE", "solvent": "Heptane"}]},
            {"stage_id": "candidate_pool_evoh", "stage_kind": "selective_dissolution", "target_polymer": "EVOH", "candidate_pairs": [{"polymer": "EVOH", "solvent": "Methanol"}]},
        ],
        "polymer_solvent_filters": {"PE": ["Heptane"], "EVOH": ["Methanol"]},
        "candidate_solvents": ["Heptane", "Methanol"],
    }

    context = _prepare_optimization_context(
        feed=1000,
        pe_fraction=0.5,
        pet_fraction=0.0,
        n6_fraction=0.0,
        evoh_fraction=0.5,
        scenario="A",
        stage_candidates_json=stage_candidates,
    )

    assert "infeasible_response" not in context
    assert context["route_pool_mode"] == "slot_independent"
    assert context["route_candidates"]
    assert context["route_candidates_for_enforcement"] == []
    assert any(
        "full candidate_pairs pool for optimization" in warning
        for warning in context["filter_warnings"]
    )


def test_build_candidate_landscape_route_specs_samples_one_and_two_wash_designs():
    context = {
        "fractions": {"PE": 0.6, "EVOH": 0.4},
        "recoverable_polymers": ["PE", "EVOH"],
        "data": {
            "sets": {
                "P": ["PE", "EVOH"],
                "S": ["PE-S1", "PE-S2", "EVOH-S1", "EVOH-S2"],
                "S_BY_STAGE_POLYMER": {
                    "Wash 1": {"PE": ["PE-S1"], "EVOH": ["EVOH-S1"]},
                    "Wash 2": {"PE": ["PE-S2"], "EVOH": ["EVOH-S2"]},
                },
            }
        },
        "strap_df": pd.DataFrame(
            [
                {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "PE-S1"},
                {"Wash number": "Wash 2", "Polymer": "PE", "Solvents": "PE-S2"},
                {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "EVOH-S1"},
                {"Wash number": "Wash 2", "Polymer": "EVOH", "Solvents": "EVOH-S2"},
            ]
        ),
    }

    specs, summary = _build_candidate_landscape_route_specs(
        context,
        min_active_washes=0,
        max_active_washes=2,
        max_routes=20,
    )

    signatures = {
        (tuple(spec["sequence"]), tuple(sorted(spec["polymer_solvent_map"].items())))
        for spec in specs
    }
    assert (("PE",), (("PE", "PE-S1"),)) in signatures
    assert (("EVOH",), (("EVOH", "EVOH-S1"),)) in signatures
    assert (("PE", "EVOH"), (("EVOH", "EVOH-S2"), ("PE", "PE-S1"))) in signatures
    assert (("EVOH", "PE"), (("EVOH", "EVOH-S1"), ("PE", "PE-S2"))) in signatures
    assert all(len(set(spec["sequence"])) == len(spec["sequence"]) for spec in specs)
    assert summary["n_candidate_designs_selected"] == len(specs)


def test_constrained_pareto_sweep_applies_default_scip_time_limit(monkeypatch):
    captured_options: list[dict[str, object]] = []

    def fake_builder():
        return object(), True, None

    def fake_pareto(*args, **kwargs):
        captured_options.append(dict(kwargs["solver_options"]))
        return pd.DataFrame(
            [
                {
                    "epsilon": 1.0,
                    "profit": 1.0,
                    "emissions": 1.0,
                    "CE": 2.0,
                    "total_cost": 3.0,
                    "capital_cost": 1.0,
                    "operational_cost": 1.0,
                    "transportation_cost": 1.0,
                    "stage1": ["st1"],
                    "stage2": ["lf"],
                    "stage3": [],
                    "wash1": ["PE-Cyclohexane"],
                    "wash2": [],
                }
            ]
        )

    monkeypatch.setattr("strap.tools.waste_optimization.pareto_cost_vs_ce", fake_pareto)

    frontier, reason = waste_optimization._run_constrained_pareto_sweep(
        fake_builder,
        y_metric="circularity",
        cost_opt={"CE": 1.0},
        y_opt={"CE": 2.0},
        n_points=4,
    )

    assert reason is None
    assert len(frontier) == 1
    assert captured_options
    assert captured_options[0]["limits/time"] == waste_optimization._PARETO_SWEEP_SOLVE_TIME_LIMIT_SEC


def test_sample_candidate_landscape_points_caps_forced_design_solves(monkeypatch):
    captured_options: list[dict[str, object]] = []

    monkeypatch.setattr(
        "strap.tools.waste_optimization._build_candidate_landscape_route_specs",
        lambda *args, **kwargs: (
            [
                {
                    "route_id": "landscape_1",
                    "sequence": ["PE"],
                    "polymer_solvent_map": {"PE": "Cyclohexane"},
                }
            ],
            {"status": "generated"},
        ),
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr("strap.tools.waste_optimization._apply_active_wash_constraints", lambda *args, **kwargs: None)
    monkeypatch.setattr("strap.tools.waste_optimization._apply_route_constraints", lambda *args, **kwargs: (True, None))

    def fake_solve(model, objective, **kwargs):
        captured_options.append(dict(kwargs["solver_options"]))
        return {
            "profit": 1.0,
            "emissions": 2.0,
            "CE": 300000.0,
            "total_cost": 4.0,
            "capital_cost": 1.0,
            "operational_cost": 2.0,
            "transportation_cost": 1.0,
            "stage1_tech": ["st1"],
            "stage2_tech": ["lf"],
            "stage3_tech": [],
            "wash1_selection": ["PE-Cyclohexane"],
            "wash2_selection": [],
        }, [{"accepted": True}]

    monkeypatch.setattr("strap.tools.waste_optimization._solve_objective_with_fallback", fake_solve)

    points, summary = waste_optimization._sample_candidate_landscape_points(
        {"data": {"sets": {"P": ["PE"], "S": ["Cyclohexane"]}}, "config": {}},
        min_active_washes=1,
        max_active_washes=2,
    )

    assert summary["n_candidate_designs_solved"] == 1
    assert len(points) == 1
    assert captured_options
    assert captured_options[0]["limits/time"] == waste_optimization._LANDSCAPE_SOLVE_TIME_LIMIT_SEC


def test_solve_objective_with_fallback_selects_best_verified_retry_after_failure(monkeypatch):
    class CloneableModel:
        def clone(self):
            return CloneableModel()

    calls: list[dict[str, object]] = []

    def fake_solve_single(model, objective, solver_name="scip", solver_options=None):
        calls.append(dict(solver_options or {}))
        if solver_options is None:
            return None
        if solver_options == {"presolving/maxrounds": 0}:
            return {"CE": 100.0, "profit": 0.0, "emissions": 10.0, "total_cost": 10.0}
        return {"CE": 250.0, "profit": 0.0, "emissions": 10.0, "total_cost": 10.0}

    monkeypatch.setattr("strap.tools.waste_optimization.solve_single", fake_solve_single)
    monkeypatch.setattr("strap.tools.waste_optimization.consume_last_solver_debug", lambda: {})

    result, debug = _solve_objective_with_fallback(
        CloneableModel(),
        "max_circularity",
        return_debug=True,
    )

    assert result["CE"] == 250.0
    assert len(calls) == 3
    assert [attempt["accepted"] for attempt in debug] == [False, True, True]


def test_derived_optimizer_sets_do_not_backfill_missing_polymers_when_explicit_table_is_authoritative():
    df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "Dimethyl sulfoxide"},
            {"Wash number": "Wash 2", "Polymer": "EVOH", "Solvents": "Ethylene Glycol"},
        ]
    )

    sets = derive_optimizer_sets_from_df(df, fallback_defaults=False)

    assert sets["S_BY_STAGE_POLYMER"]["Wash 1"]["EVOH"] == ["Dimethyl sulfoxide"]
    assert sets["S_BY_STAGE_POLYMER"]["Wash 2"]["EVOH"] == ["Ethylene Glycol"]
    assert sets["S_BY_STAGE_POLYMER"]["Wash 1"]["PE"] == []
    assert sets["S_BY_STAGE_POLYMER"]["Wash 2"]["PE"] == []
    assert "p-Xylene" not in sets["S"]


def test_optimizer_default_sets_include_extended_biosteam_polymers():
    sets = get_optimizer_default_sets()

    for polymer in ("PP", "PS", "PVC", "PC"):
        assert polymer in sets["P"]
        assert sets[f"S_{polymer}"]
        assert sets["S_BY_POLYMER"][polymer]
        assert sets["S_BY_STAGE_POLYMER"]["Wash 1"][polymer]
        assert sets["S_BY_STAGE_POLYMER"]["Wash 2"][polymer]
        assert set(sets["S_BY_POLYMER"][polymer]).issubset(set(sets["S"]))


def test_candidate_solvents_json_mapping_is_treated_as_polymer_filters():
    df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "PP", "Solvents": "Toluene"},
            {"Wash number": "Wash 1", "Polymer": "PP", "Solvents": "Cyclohexane"},
            {"Wash number": "Wash 1", "Polymer": "PS", "Solvents": "Toluene"},
            {"Wash number": "Wash 1", "Polymer": "PS", "Solvents": "N,N-Dimethylformamide"},
        ]
    )

    filtered, applied, warnings, requested = _apply_solvent_filters(
        df,
        candidate_solvents='{"PP": ["Toluene", "Cyclohexane"], "PS": ["Toluene", "N,N-Dimethylformamide"]}',
    )

    assert warnings == []
    assert requested["PP"] == ["Toluene", "Cyclohexane"]
    assert requested["PS"] == ["Toluene", "N,N-Dimethylformamide"]
    assert applied == {key: value for key, value in requested.items() if key != "global"}
    assert 'PP": ["Toluene' not in filtered["Solvents"].astype(str).tolist()


def test_run_biosteam_updates_uses_feed_fraction_percent_in_build_config(monkeypatch):
    df = pd.DataFrame([
        {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Cyclohexane", **_biosteam_columns_template()},
        {"Wash number": "Wash 1", "Polymer": "EVOH", "Solvents": "Ethylene Glycol", **_biosteam_columns_template()},
    ])

    captured_configs: list[dict[str, object]] = []

    def fake_build_config(**kwargs):
        captured_configs.append(dict(kwargs))
        return {"solvent": kwargs["solvent"]}

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_config)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config: {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.5},
            "operations": {"total_energy_mj_per_kg": 2.0},
        },
    )

    _run_biosteam_updates(
        df,
        capacity_pe=1000.0,
        capacity_evoh=500.0,
        pe_fraction_pct=60.0,
        evoh_fraction_pct=30.0,
    )

    percent_by_solvent = {
        cfg["solvent"]: cfg["target_plastic_percent"]
        for cfg in captured_configs
    }
    assert percent_by_solvent["Cyclohexane"] == 60.0
    assert percent_by_solvent["Ethylene Glycol"] == 30.0


def test_extract_route_candidates_normalizes_polymer_names():
    """_extract_route_candidates must surface the adapter's route_candidates intact."""
    payload = {
        "route_candidates": [
            {
                "route_id": "route_1",
                "rank": 1,
                "sequence": ["LDPE", "EVOH"],
                "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Methanol"},
            },
            {
                "route_id": "route_2",
                "rank": 2,
                "polymer_solvent_map": {"LDPE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
            },
            # Invalid entry — should be filtered out
            {"route_id": "bad", "polymer_solvent_map": {}},
        ],
    }
    routes = _extract_route_candidates(payload)
    assert len(routes) == 2
    assert routes[0]["polymer_solvent_map"] == {"PE": "Cyclohexane", "EVOH": "Methanol"}
    # LDPE alias in the second entry must normalize to PE.
    assert routes[1]["polymer_solvent_map"] == {"PE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"}


def test_non_dominated_filters_dominated_points_on_cost_vs_emissions():
    """Cross-route aggregation must drop dominated points."""
    points = [
        {"point_id": 1, "total_cost": 100.0, "emissions": 10.0, "route_id": "A"},
        {"point_id": 2, "total_cost": 120.0, "emissions": 12.0, "route_id": "A"},  # dominated by 1
        {"point_id": 3, "total_cost": 90.0, "emissions": 15.0, "route_id": "B"},
        {"point_id": 4, "total_cost": 150.0, "emissions": 5.0, "route_id": "B"},
    ]
    nondom = _non_dominated(points, y_key="emissions")
    route_ids = {p["point_id"] for p in nondom}
    assert 2 not in route_ids  # dominated by 1
    assert route_ids == {1, 3, 4}


def test_run_waste_management_pareto_skips_route_pool_when_slot_independent_pair_pool_is_authoritative(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.6, "PET": 0.0, "N6": 0.0, "EVOH": 0.4},
            "requested_filters": {"PE": ["Cyclohexane"], "EVOH": ["Methanol"]},
            "applied_filters": {"PE": ["Cyclohexane"], "EVOH": ["Methanol"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "simulation_failures": [],
            "constraint_mode": "ranked_soft",
            "fallback_policy": "fail_closed",
            "route_pool_mode": "slot_independent",
            "has_typed_handoff": True,
            "route_candidates": [{"route_id": "route_1"}],
            "route_candidates_for_enforcement": [],
            "strap_table_rows": 2,
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("route pool should not be used when candidate_pairs are authoritative")

    monkeypatch.setattr("strap.tools.waste_optimization._run_pareto_with_route_pool", fail_if_called)
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: {
            "min_total_cost": {"total_cost": 100.0, "emissions": 10.0, "CE": 700000.0},
            "min_emissions": {"total_cost": 100.0, "emissions": 10.0, "CE": 700000.0},
            "max_circularity": {"total_cost": 120.0, "emissions": 12.0, "CE": 710000.0},
        }[objective],
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 10.0,
                    "profit": -100.0,
                    "emissions": 10.0,
                    "CE": 700000.0,
                    "total_cost": 100.0,
                    "capital_cost": 0.0,
                    "operational_cost": 80.0,
                    "transportation_cost": 20.0,
                    "stage1": ["lf"],
                    "stage2": [],
                    "stage3": [],
                    "wash1": [],
                    "wash2": [],
                }
            ]
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        feed_composition_json={"PE": 0.6, "EVOH": 0.4},
        scenario="A",
        x_metric="total_cost",
        y_metric="emissions",
        n_points=4,
        stage_candidates_json={"schema_version": "1.1"},
        min_active_washes=0,
        max_active_washes=2,
    )
    payload = json.loads(raw)["data"]

    assert payload["analysis_type"] == "pareto_front"
    assert payload["n_points_feasible"] == 1


def test_run_waste_management_pareto_includes_landscape_samples(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.6, "PET": 0.0, "N6": 0.0, "EVOH": 0.4},
            "requested_filters": {"PE": ["Cyclohexane"], "EVOH": ["Methanol"]},
            "applied_filters": {"PE": ["Cyclohexane"], "EVOH": ["Methanol"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "simulation_failures": [],
            "simulation_skips": [],
            "constraint_mode": "soft",
            "fallback_policy": "broaden_disclosed",
            "route_pool_mode": "slot_independent",
            "has_typed_handoff": True,
            "route_candidates": [],
            "route_candidates_for_enforcement": [],
            "strap_table_rows": 2,
        },
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: {
            "min_total_cost": {"total_cost": 100.0, "emissions": 10.0, "CE": 700000.0},
            "min_emissions": {"total_cost": 100.0, "emissions": 10.0, "CE": 700000.0},
        }[objective],
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 10.0,
                    "profit": -100.0,
                    "emissions": 10.0,
                    "CE": 700000.0,
                    "total_cost": 100.0,
                    "capital_cost": 0.0,
                    "operational_cost": 80.0,
                    "transportation_cost": 20.0,
                    "stage1": ["lf"],
                    "stage2": [],
                    "stage3": [],
                    "wash1": [],
                    "wash2": [],
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._sample_candidate_landscape_points",
        lambda *args, **kwargs: (
            [
                {
                    "total_cost": 140.0,
                    "emissions": 20.0,
                    "circularity_score": 0.72,
                    "stage1_tech": ["st1"],
                    "stage2_tech": [],
                    "stage3_tech": ["lf"],
                    "wash1_selection": ["PE-Cyclohexane"],
                    "wash2_selection": [],
                    "point_status": "landscape_sample",
                    "is_frontier": False,
                    "selection_origin": "landscape_sample",
                }
            ],
            {
                "n_candidate_designs_attempted": 1,
                "n_candidate_designs_solved": 1,
                "n_landscape_points": 1,
            },
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        feed_composition_json={"PE": 0.6, "EVOH": 0.4},
        scenario="A",
        x_metric="total_cost",
        y_metric="emissions",
        n_points=4,
        min_active_washes=0,
        max_active_washes=2,
    )
    payload = json.loads(raw)["data"]

    assert payload["landscape_summary"]["n_landscape_points"] == 1
    assert payload["route_pool_mode"] == "slot_independent"
    assert len(payload["landscape_points"]) == 1
    assert len(payload["all_feasible_points"]) == 1
    assert payload["points"][0]["selection_origin"] == "candidate_pool"
    assert payload["landscape_points"][0]["point_status"] == "landscape_sample"


def test_run_waste_management_pareto_enforces_routes_when_stage_candidates_present(monkeypatch):
    """When route_candidates are present under hard/ranked_soft, the Pareto tool
    must use the pooled route-shortlist path rather than the aggregate solve."""

    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.5, "PET": 0.0, "N6": 0.0, "EVOH": 0.5},
            "requested_filters": {"PE": ["Cyclohexane"], "EVOH": ["Methanol", "Dimethyl sulfoxide"]},
            "applied_filters": {"PE": ["Cyclohexane"], "EVOH": ["Methanol", "Dimethyl sulfoxide"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "simulation_failures": [],
            "constraint_mode": "hard",
            "fallback_policy": "fail_closed",
            "has_typed_handoff": True,
            "route_candidates": [
                {
                    "route_id": "route_1",
                    "rank": 1,
                    "sequence": ["PE", "EVOH"],
                    "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Methanol"},
                },
                {
                    "route_id": "route_2",
                    "rank": 2,
                    "sequence": ["PE", "EVOH"],
                    "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
                },
            ],
        },
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._run_pareto_with_route_pool",
        lambda *args, **kwargs: json.dumps(
            {
                "display": "route pooled",
                "data": {
                    "analysis_type": "pareto_front",
                    "n_routes_requested": 2,
                    "n_routes_solved": 2,
                    "n_points_feasible": 3,
                    "points": [
                        {"point_id": 1, "route_id": "route_1", "stage3_tech": ["WtE"], "stage3_variants": ["WtE"]},
                        {"point_id": 2, "route_id": "route_1", "stage3_tech": ["WtE"], "stage3_variants": ["WtE"]},
                        {"point_id": 3, "route_id": "route_2", "stage3_tech": ["Landfill"], "stage3_variants": ["Landfill"]},
                    ],
                    "route_reports": [
                        {
                            "route_id": "route_1",
                            "status": "feasible",
                            "n_points_on_frontier": 2,
                            "stage3_techs_explored": ["WtE"],
                        },
                        {
                            "route_id": "route_2",
                            "status": "feasible",
                            "n_points_on_frontier": 1,
                            "stage3_techs_explored": ["Landfill"],
                        },
                    ],
                    "frontier_summary": {
                        "n_routes_on_frontier": 2,
                        "n_distinct_stage3_techs": 2,
                        "distinct_stage3_techs": ["Landfill", "WtE"],
                    },
                },
            }
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        pe_fraction=0.6,
        pet_fraction=0.1,
        n6_fraction=0.0,
        evoh_fraction=0.3,
        y_metric="emissions",
        n_points=4,
    )
    payload = json.loads(raw)["data"]

    # The pooled route path was used (not the un-routed aggregate solve).
    assert payload["analysis_type"] == "pareto_front"
    assert payload["n_routes_requested"] == 2
    assert payload["n_routes_solved"] == 2
    assert payload["n_points_feasible"] == 3
    assert all("route_id" in p for p in payload["points"])
    assert payload["frontier_summary"]["n_routes_on_frontier"] == 2
    assert payload["frontier_summary"]["n_distinct_stage3_techs"] == 2
    assert payload["frontier_summary"]["distinct_stage3_techs"] == ["Landfill", "WtE"]
    report_ids = {r["route_id"] for r in payload["route_reports"]}
    assert report_ids == {"route_1", "route_2"}
    route_report_map = {r["route_id"]: r for r in payload["route_reports"]}
    assert route_report_map["route_1"]["stage3_techs_explored"] == ["WtE"]
    assert route_report_map["route_2"]["stage3_techs_explored"] == ["Landfill"]
    assert route_report_map["route_1"]["n_points_on_frontier"] == 2
    assert route_report_map["route_2"]["n_points_on_frontier"] == 1


def test_run_waste_management_pareto_flattens_filter_fields(monkeypatch):
    """B1/H1: Pareto result must expose flat filter fields for the verifier."""

    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.5, "PET": 0.0, "N6": 0.0, "EVOH": 0.5},
            "requested_filters": {"PE": ["Cyclohexane"]},
            "applied_filters": {},
            "filter_warnings": ["No PE solvent overlap"],
            "filter_status": "fallback_to_full_catalog",
            "simulation_failures": [{"polymer": "PE", "solvent": "Cyclohexane"}],
            "simulation_skips": [{"polymer": "PE", "solvent": "Heptane", "reason": "baseline_sufficient", "source": "workbook_baseline"}],
            "constraint_mode": "soft",
            "fallback_policy": "broaden_disclosed",
            "route_pool_mode": "slot_independent",
            "has_typed_handoff": True,
            "stage_candidate_payload": {
                "schema_version": "1.1",
                "source_handoff_id": "sep-handoff-1",
                "route_pool_mode": "slot_independent",
                "candidate_pairs": [
                    {"polymer": "PE", "solvent": "Cyclohexane", "optimizer_option": "Cyclohexane"},
                ],
                "candidate_counts_by_polymer": {"PE": 1},
                "polymer_solvent_filters": {"PE": ["Cyclohexane"]},
                "polymer_option_filters": {"PE": ["Cyclohexane"]},
                "candidate_backfill_warnings": ["Filled under-reported candidates"],
            },
        },
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization.solve_single",
        lambda model, sense, solver_name="gurobi": {"total_cost": 100.0, "emissions": 5.0, "CE": 200000.0},
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [{"epsilon": 5.0, "profit": 10.0, "emissions": 5.0, "CE": 200000.0, "total_cost": 100.0, "capital_cost": 40.0, "operational_cost": 50.0, "transportation_cost": 10.0, "stage1": ["STRAP"], "stage2": ["Reuse"], "stage3": ["WtE"], "wash1": ["PE-Cyclohexane"], "wash2": []}]
        ),
    )

    raw = run_waste_management_pareto(
        feed=1000, pe_fraction=0.5, pet_fraction=0.0, n6_fraction=0.0, evoh_fraction=0.5,
        y_metric="emissions", n_points=3,
    )
    payload = json.loads(raw)["data"]

    # Flat fields matching point_optimum shape
    assert payload["requested_solvent_filters"] == {"PE": ["Cyclohexane"]}
    assert payload["solvent_filter_warnings"] == ["No PE solvent overlap"]
    assert payload["solvent_filter_status"] == "fallback_to_full_catalog"
    assert payload["simulation_failures"] == [{"polymer": "PE", "solvent": "Cyclohexane"}]
    assert payload["simulation_skips"] == [{"polymer": "PE", "solvent": "Heptane", "reason": "baseline_sufficient", "source": "workbook_baseline"}]
    # Nested summary is retained for backwards compatibility
    assert payload["candidate_summary"]["status"] == "fallback_to_full_catalog"
    assert payload["candidate_telemetry"]["requested"]["counts_by_polymer"] == {"PE": 1}
    assert payload["candidate_telemetry"]["simulation"]["failure_counts_by_class"] == {"unknown_failure": 1}
    assert payload["source_handoff_summary"]["status"] == "typed_handoff"
    assert payload["source_handoff_summary"]["source_handoff_id"] == "sep-handoff-1"
    assert payload["source_handoff_summary"]["candidate_counts_by_polymer"] == {"PE": 1}
    assert payload["source_handoff_summary"]["candidate_backfill_warnings"] == ["Filled under-reported candidates"]


def test_build_candidate_telemetry_exposes_requested_surviving_and_failure_breakdown():
    strap_df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Cyclohexane"},
            {"Wash number": "Wash 1", "Polymer": "PE", "Solvents": "Heptane"},
            {"Wash number": "Wash 2", "Polymer": "EVOH", "Solvents": "Ethylene Glycol"},
        ]
    )
    telemetry = _build_candidate_telemetry(
        {
            "has_typed_handoff": True,
            "route_candidates": [{"route_id": "route_1"}],
            "route_pool_mode": "slot_independent",
            "constraint_mode": "ranked_soft",
            "fallback_policy": "broaden_disclosed",
            "requested_filters": {
                "global": [],
                "PE": ["Cyclohexane", "Heptane", "Tetrahydropyran"],
                "EVOH": ["Ethylene Glycol", "Hexamethylphosphoramide"],
            },
            "applied_filters": {
                "PE": ["Cyclohexane", "Heptane", "Tetrahydropyran"],
                "EVOH": ["Ethylene Glycol", "Hexamethylphosphoramide"],
            },
            "stage_candidate_payload": {
                "candidate_counts_by_polymer": {"PE": 3, "EVOH": 2},
                "polymer_solvent_filters": {
                    "PE": ["Cyclohexane", "Heptane", "Tetrahydropyran"],
                    "EVOH": ["Ethylene Glycol", "Hexamethylphosphoramide"],
                },
            },
            "strap_df": strap_df,
            "strap_table_rows": 3,
            "simulation_failures": [
                {
                    "polymer": "PE",
                    "solvent": "Tetrahydropyran",
                    "failure_class": "vapor_pressure_extrapolation_failure",
                    "reason": "vapor_pressure_extrapolation_failure: failed to extrapolate",
                    "source": "runtime_denylist",
                },
                {
                    "polymer": "EVOH",
                    "solvent": "Hexamethylphosphoramide",
                    "failure_class": "undefined_chemical_alias",
                    "reason": "undefined_chemical_alias: P4O10",
                    "source": "runtime_denylist",
                },
            ],
            "simulation_skips": [
                {
                    "polymer": "PE",
                    "solvent": "Cyclohexane",
                    "reason": "baseline_sufficient",
                    "source": "workbook_baseline",
                }
            ],
        }
    )

    assert telemetry["requested"]["counts_by_polymer"] == {"PE": 3, "EVOH": 2}
    assert telemetry["requested"]["source_counts_by_polymer"] == {"PE": 3, "EVOH": 2}
    assert telemetry["surviving"]["counts_by_polymer"] == {"PE": 2, "EVOH": 1}
    assert telemetry["surviving"]["counts_by_stage"]["Wash 1"] == {"PE": 2}
    assert telemetry["surviving"]["counts_by_stage"]["Wash 2"] == {"EVOH": 1}
    assert telemetry["simulation"]["failure_counts_by_class"] == {
        "undefined_chemical_alias": 1,
        "vapor_pressure_extrapolation_failure": 1,
    }
    assert telemetry["simulation"]["skip_counts_by_reason"] == {"baseline_sufficient": 1}


def test_build_source_handoff_summary_reports_optimizer_consumed_candidates():
    summary = _build_source_handoff_summary(
        {
            "has_typed_handoff": True,
            "constraint_mode": "ranked_soft",
            "fallback_policy": "fail_closed",
            "route_pool_mode": "slot_independent",
            "fractions": {"PE": 0.2, "EVOH": 0.2, "PET": 0.6},
            "stage_candidate_payload": {
                "schema_version": "1.1",
                "workflow_scope": "multi_stage",
                "source_handoff_id": "handoff-sep-1",
                "route_id": "handoff-sep-1",
                "operating_constraints": {"temperature_max_c": 145.0},
                "feed_composition": {"LDPE": 0.2, "EVOH": 0.2, "PET": 0.6},
                "feed_capacity_tpy": 8000,
                "polymers": ["LDPE", "EVOH", "PET"],
                "stages": [{"stage_id": "candidate_pool_pe"}],
                "route_candidates": [{"route_id": "route_1"}],
                "candidate_pairs": [
                    {"polymer": "PE", "solvent": "Cyclohexane", "optimizer_option": "Cyclohexane@81C"},
                    {"polymer": "PE", "solvent": "Cyclohexane", "optimizer_option": "Cyclohexane@100C"},
                    {"polymer": "PET", "solvent": "Dichloromethane", "optimizer_option": "Dichloromethane@39C"},
                ],
                "candidate_counts_by_polymer": {"PE": 2, "PET": 1},
                "polymer_solvent_filters": {"PE": ["Cyclohexane"], "PET": ["Dichloromethane"]},
                "polymer_option_filters": {
                    "PE": ["Cyclohexane@81C", "Cyclohexane@100C"],
                    "PET": ["Dichloromethane@39C"],
                },
                "candidate_backfill_warnings": ["Filled under-reported candidates"],
                "route_candidate_warnings": ["Dropped unsupported route"],
                "source_user_query": "user query",
                "source_task_prompt": "task prompt",
            },
        }
    )

    assert summary["summary_kind"] == "optimizer_consumed_handoff"
    assert summary["status"] == "typed_handoff"
    assert summary["source_handoff_id"] == "handoff-sep-1"
    assert summary["candidate_counts_by_polymer"] == {"PE": 2, "PET": 1}
    assert summary["candidate_pair_counts_by_polymer"] == {"PE": 2, "PET": 1}
    assert summary["candidate_solvent_counts_by_polymer"] == {"PE": 1, "PET": 1}
    assert summary["candidate_option_counts_by_polymer"] == {"PE": 2, "PET": 1}
    assert summary["polymer_option_filter_counts_by_polymer"] == {"PE": 2, "PET": 1}
    assert summary["candidate_backfill_warnings"] == ["Filled under-reported candidates"]
    assert summary["source_user_query_present"] is True
    assert summary["source_task_prompt_present"] is True


def test_run_waste_management_pareto_broadens_after_ranked_route_failure(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.6, "PET": 0.1, "N6": 0.0, "EVOH": 0.3},
            "requested_filters": {"PE": ["Cyclohexane"], "EVOH": ["Dimethyl sulfoxide"]},
            "applied_filters": {},
            "filter_warnings": [],
            "filter_status": "applied",
            "simulation_failures": [],
            "constraint_mode": "ranked_soft",
            "fallback_policy": "broaden_disclosed",
            "route_candidates": [
                {"route_id": "route_1", "rank": 1, "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"}}
            ],
        },
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._run_pareto_with_route_pool",
        lambda *args, **kwargs: json.dumps(
            {
                "display": "route-enforced infeasible",
                "data": {
                    "analysis_type": "pareto_front",
                    "n_points_feasible": 0,
                    "n_routes_solved": 0,
                    "route_reports": [
                        {"route_id": "route_1", "status": "infeasible", "reason": "cost-anchor solve failed"}
                    ],
                },
            }
        ),
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization.solve_single",
        lambda model, sense, solver_name="gurobi": {"total_cost": 100.0, "emissions": 5.0, "CE": 200000.0},
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 5.0,
                    "profit": 10.0,
                    "emissions": 5.0,
                    "CE": 200000.0,
                    "total_cost": 100.0,
                    "capital_cost": 40.0,
                    "operational_cost": 50.0,
                    "transportation_cost": 10.0,
                    "stage1": ["STRAP"],
                    "stage2": ["Reuse"],
                    "stage3": ["WtE"],
                    "wash1": ["PE-Heptane"],
                    "wash2": ["EVOH-gamma-butyrolactone"],
                }
            ]
        ),
    )

    raw = run_waste_management_pareto(
        feed=1000, pe_fraction=0.6, pet_fraction=0.1, n6_fraction=0.0, evoh_fraction=0.3,
        y_metric="emissions", n_points=3,
    )
    payload = json.loads(raw)["data"]

    # When every route is skipped/infeasible, we no longer broaden to the
    # aggregate catalog because that produces ghost landfill pathways.
    # Return the route_response as-is so the caller sees the honest
    # "all routes infeasible" state.
    assert payload["n_points_feasible"] == 0
    assert payload["n_routes_solved"] == 0
    assert payload["route_reports"][0]["status"] == "infeasible"


def test_run_waste_management_pareto_filters_to_true_frontier_in_aggregate_path(monkeypatch):
    monkeypatch.setattr(
        "strap.tools.waste_optimization._prepare_optimization_context",
        lambda **_: {
            "temp_dir": None,
            "data": {"dummy": True},
            "config": {"dummy": True},
            "scenario": "A",
            "fractions": {"PE": 0.6, "PET": 0.1, "N6": 0.0, "EVOH": 0.3},
            "requested_filters": {"PE": ["Heptane"], "EVOH": ["Pyridazine"]},
            "applied_filters": {"PE": ["Heptane"], "EVOH": ["Pyridazine"]},
            "filter_warnings": [],
            "filter_status": "applied",
            "simulation_failures": [],
            "simulation_skips": [],
            "constraint_mode": "soft",
            "fallback_policy": "broaden_disclosed",
            "route_candidates": [],
            "strap_table_rows": 4,
        },
    )
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: {
            "min_total_cost": {"total_cost": 100.0, "emissions": 12.0, "CE": 200000.0},
            "min_emissions": {"total_cost": 150.0, "emissions": 5.0, "CE": 300000.0},
        }[objective],
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 12.0,
                    "profit": 20.0,
                    "emissions": 12.0,
                    "CE": 200000.0,
                    "total_cost": 100.0,
                    "capital_cost": 40.0,
                    "operational_cost": 50.0,
                    "transportation_cost": 10.0,
                    "stage1": ["STRAP"],
                    "stage2": ["Reuse"],
                    "stage3": ["WtE"],
                    "wash1": ["PE-Heptane"],
                    "wash2": ["EVOH-Pyridazine"],
                },
                {
                    "epsilon": 8.0,
                    "profit": 19.0,
                    "emissions": 8.0,
                    "CE": 240000.0,
                    "total_cost": 160.0,
                    "capital_cost": 60.0,
                    "operational_cost": 80.0,
                    "transportation_cost": 20.0,
                    "stage1": ["STRAP"],
                    "stage2": ["Reuse"],
                    "stage3": ["WtE"],
                    "wash1": ["PE-Heptane"],
                    "wash2": ["EVOH-Pyridazine"],
                },
                {
                    "epsilon": 5.0,
                    "profit": 18.0,
                    "emissions": 5.0,
                    "CE": 300000.0,
                    "total_cost": 150.0,
                    "capital_cost": 60.0,
                    "operational_cost": 75.0,
                    "transportation_cost": 15.0,
                    "stage1": ["STRAP"],
                    "stage2": ["Reuse"],
                    "stage3": ["WtE"],
                    "wash1": ["PE-Heptane"],
                    "wash2": ["EVOH-Pyridazine"],
                },
            ]
        ),
    )

    raw = run_waste_management_pareto(
        feed=8000,
        pe_fraction=0.6,
        pet_fraction=0.1,
        n6_fraction=0.0,
        evoh_fraction=0.3,
        y_metric="emissions",
        n_points=6,
    )
    payload = json.loads(raw)["data"]

    assert payload["n_points_raw_feasible"] == 3
    assert payload["n_points_feasible"] == 2
    assert [point["point_id"] for point in payload["points"]] == [1, 2]
    assert {(point["total_cost"], point["emissions"]) for point in payload["points"]} == {
        (100.0, 12.0),
        (150.0, 5.0),
    }


def test_run_pareto_with_route_pool_supports_arbitrary_n_points(monkeypatch):
    context = {
        "data": {
            "sets": {
                "P": ["PE", "EVOH"],
                "S": ["Cyclohexane", "Heptane", "Dimethyl sulfoxide", "Ethylene Glycol"],
            }
        },
        "config": {"dummy": True},
        "scenario": "A",
        "fractions": {"PE": 0.6, "PET": 0.1, "N6": 0.0, "EVOH": 0.3},
        "constraint_mode": "ranked_soft",
        "fallback_policy": "broaden_disclosed",
        "requested_filters": {"PE": ["Cyclohexane", "Heptane"], "EVOH": ["Dimethyl sulfoxide", "Ethylene Glycol"]},
        "applied_filters": {"PE": ["Cyclohexane", "Heptane"], "EVOH": ["Dimethyl sulfoxide", "Ethylene Glycol"]},
        "filter_warnings": [],
        "filter_status": "applied",
        "simulation_failures": [],
        "strap_table_rows": 4,
    }
    route_candidates = [
        {
            "route_id": "route_1",
            "rank": 1,
            "sequence": ["PE", "EVOH"],
            "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
        },
        {
            "route_id": "route_2",
            "rank": 2,
            "sequence": ["PE", "EVOH"],
            "polymer_solvent_map": {"PE": "Heptane", "EVOH": "Ethylene Glycol"},
        },
    ]

    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._apply_route_pool_constraints",
        lambda model, routes, **kwargs: (True, None),
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective: {
            "min_total_cost": {"total_cost": 1000.0, "emissions": 120.0, "CE": 200000.0},
            "min_emissions": {"total_cost": 2000.0, "emissions": 10.0, "CE": 500000.0},
        }[objective],
    )

    rows = []
    for idx in range(12):
        route_id = "route_1" if idx < 6 else "route_2"
        wash1 = "PE-Cyclohexane" if route_id == "route_1" else "PE-Heptane"
        wash2 = "EVOH-Dimethyl sulfoxide" if route_id == "route_1" else "EVOH-Ethylene Glycol"
        rows.append(
            {
                "epsilon": 120.0 - idx * 10.0,
                "profit": 100.0 - idx,
                "emissions": 120.0 - idx * 10.0,
                "CE": 300000.0 + idx * 1000.0,
                "total_cost": 1000.0 + idx * 100.0,
                "capital_cost": 400.0 + idx,
                "operational_cost": 500.0 + idx,
                "transportation_cost": 100.0,
                "stage1": ["STRAP"],
                "stage2": ["Reuse"],
                "stage3": ["WtE" if idx % 2 == 0 else "Landfill"],
                "wash1": [wash1],
                "wash2": [wash2],
            }
        )

    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(rows),
    )

    raw = _run_pareto_with_route_pool(
        context,
        route_candidates=route_candidates,
        feed=8000,
        x_metric="total_cost",
        y_metric="emissions",
        n_points=12,
    )
    payload = json.loads(raw)["data"]

    assert payload["n_points_requested"] == 12
    assert payload["n_points_raw_feasible"] == 12
    assert payload["n_points_feasible"] == 12
    assert len(payload["points"]) == 12
    assert payload["frontier_summary"]["n_routes_on_frontier"] == 2
    assert set(payload["frontier_summary"]["route_ids_on_frontier"]) == {"route_1", "route_2"}


def test_run_pareto_with_route_pool_defaults_to_exact_mode(monkeypatch):
    context = {
        "data": {"sets": {"P": ["PE", "EVOH"], "S": ["Cyclohexane", "Ethylene Glycol"]}},
        "config": {"dummy": True},
        "scenario": "A",
        "fractions": {"PE": 0.6, "PET": 0.1, "N6": 0.0, "EVOH": 0.3},
        "constraint_mode": "ranked_soft",
        "fallback_policy": "broaden_disclosed",
        "requested_filters": {"PE": ["Cyclohexane"], "EVOH": ["Ethylene Glycol"]},
        "applied_filters": {"PE": ["Cyclohexane"], "EVOH": ["Ethylene Glycol"]},
        "filter_warnings": [],
        "filter_status": "applied",
        "simulation_failures": [],
        "strap_table_rows": 2,
    }
    route_candidates = [
        {
            "route_id": "route_1",
            "rank": 1,
            "sequence": ["PE", "EVOH"],
            "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Ethylene Glycol"},
        }
    ]

    called = {"exact": 0, "slot_independent": 0}
    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())

    def fake_exact(model, routes, **kwargs):
        called["exact"] += 1
        return True, None

    def fake_slot_independent(model, routes, **kwargs):
        called["slot_independent"] += 1
        return True, None

    monkeypatch.setattr("strap.tools.waste_optimization._apply_route_pool_constraints", fake_exact)
    monkeypatch.setattr("strap.tools.waste_optimization._apply_slot_independent_constraints", fake_slot_independent)
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: {
            "min_total_cost": {"total_cost": 1000.0, "emissions": 120.0, "CE": 200000.0},
            "min_emissions": {"total_cost": 1200.0, "emissions": 100.0, "CE": 250000.0},
        }[objective],
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 120.0,
                    "profit": 10.0,
                    "emissions": 120.0,
                    "CE": 200000.0,
                    "total_cost": 1000.0,
                    "capital_cost": 400.0,
                    "operational_cost": 500.0,
                    "transportation_cost": 100.0,
                    "stage1": ["st1"],
                    "stage2": ["st2"],
                    "stage3": ["lf"],
                    "wash1": ["PE-Cyclohexane"],
                    "wash2": ["EVOH-Ethylene Glycol"],
                }
            ]
        ),
    )

    raw = _run_pareto_with_route_pool(
        context,
        route_candidates=route_candidates,
        feed=8000,
        x_metric="total_cost",
        y_metric="emissions",
        n_points=4,
    )
    payload = json.loads(raw)["data"]

    assert payload["route_pool_mode"] == "exact"
    assert called["exact"] > 0
    assert called["slot_independent"] == 0


def test_run_pareto_with_slot_independent_mode_emits_cross_product_provenance(monkeypatch):
    context = {
        "data": {"sets": {"P": ["PE", "EVOH"], "S": ["Cyclohexane", "Heptane", "Dimethyl sulfoxide", "Ethylene Glycol"]}},
        "config": {"dummy": True},
        "scenario": "A",
        "fractions": {"PE": 0.6, "PET": 0.1, "N6": 0.0, "EVOH": 0.3},
        "constraint_mode": "ranked_soft",
        "fallback_policy": "broaden_disclosed",
        "route_pool_mode": "slot_independent",
        "requested_filters": {"PE": ["Cyclohexane", "Heptane"], "EVOH": ["Dimethyl sulfoxide", "Ethylene Glycol"]},
        "applied_filters": {"PE": ["Cyclohexane", "Heptane"], "EVOH": ["Dimethyl sulfoxide", "Ethylene Glycol"]},
        "filter_warnings": [],
        "filter_status": "applied",
        "simulation_failures": [],
        "strap_table_rows": 4,
    }
    route_candidates = [
        {
            "route_id": "route_1",
            "rank": 1,
            "sequence": ["PE", "EVOH"],
            "polymer_solvent_map": {"PE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
        },
        {
            "route_id": "route_2",
            "rank": 2,
            "sequence": ["PE", "EVOH"],
            "polymer_solvent_map": {"PE": "Heptane", "EVOH": "Ethylene Glycol"},
        },
    ]

    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._apply_slot_independent_constraints",
        lambda model, routes, **kwargs: (True, None),
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, **kwargs: {
            "min_total_cost": {"total_cost": 90.0, "emissions": 90.0, "CE": 300000.0},
            "min_emissions": {"total_cost": 120.0, "emissions": 80.0, "CE": 320000.0},
        }[objective],
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization.pareto_cost_vs_emissions",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "epsilon": 100.0,
                    "profit": 10.0,
                    "emissions": 100.0,
                    "CE": 250000.0,
                    "total_cost": 100.0,
                    "capital_cost": 40.0,
                    "operational_cost": 50.0,
                    "transportation_cost": 10.0,
                    "stage1": ["st1"],
                    "stage2": ["st2"],
                    "stage3": ["lf"],
                    "wash1": ["PE-Cyclohexane"],
                    "wash2": ["EVOH-Dimethyl sulfoxide"],
                },
                {
                    "epsilon": 90.0,
                    "profit": 11.0,
                    "emissions": 90.0,
                    "CE": 300000.0,
                    "total_cost": 90.0,
                    "capital_cost": 35.0,
                    "operational_cost": 45.0,
                    "transportation_cost": 10.0,
                    "stage1": ["st1"],
                    "stage2": ["st2"],
                    "stage3": ["lf"],
                    "wash1": ["PE-Cyclohexane"],
                    "wash2": ["EVOH-Ethylene Glycol"],
                },
                {
                    "epsilon": 80.0,
                    "profit": 9.0,
                    "emissions": 80.0,
                    "CE": 320000.0,
                    "total_cost": 120.0,
                    "capital_cost": 50.0,
                    "operational_cost": 60.0,
                    "transportation_cost": 10.0,
                    "stage1": ["st1"],
                    "stage2": ["st2"],
                    "stage3": ["gas_h2"],
                    "wash1": ["PE-Heptane"],
                    "wash2": ["EVOH-Ethylene Glycol"],
                },
            ]
        ),
    )

    raw = _run_pareto_with_route_pool(
        context,
        route_candidates=route_candidates,
        feed=8000,
        x_metric="total_cost",
        y_metric="emissions",
        n_points=8,
    )
    payload = json.loads(raw)["data"]

    assert payload["route_pool_mode"] == "slot_independent"
    assert payload["n_points_feasible"] == 2
    cross_point = next(point for point in payload["points"] if point["selection_origin"] == "cross_product")
    assert cross_point["matched_route_id"] is None
    assert cross_point["wash1_origin_route_id"] == "route_1"
    assert cross_point["wash2_origin_route_id"] == "route_2"
    route_reports = {report["route_id"]: report for report in payload["route_reports"]}
    assert route_reports["route_1"]["status"] == "dominated"
    assert route_reports["route_1"]["dominating_point_id"] is not None
    assert route_reports["route_2"]["status"] == "feasible"
    assert payload["frontier_summary"]["n_cross_product_points"] == 1


def test_adapt_separation_to_optimization_sets_explicit_route_pool_mode():
    scope = HandoffScope(invocation_id="i1", run_id="r1", thread_id="t1")
    source = HandoffRecord(
        handoff_id="h1",
        scope=scope,
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "polymers": ["LDPE", "EVOH", "PET"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "Cyclohexane"},
                {"step": 2, "polymer": "EVOH", "solvent": "Dimethyl sulfoxide"},
            ],
            "solvent_mapping": {"LDPE": "Cyclohexane", "EVOH": "Dimethyl sulfoxide"},
            "top_k_sequences": [
                {
                    "rank": 1,
                    "sequence": ["LDPE", "EVOH"],
                    "solvent_mapping": {
                        "LDPE": "Cyclohexane",
                        "EVOH": "Dimethyl sulfoxide",
                    },
                }
            ],
        },
        created_at="2026-04-21T00:00:00Z",
    )

    _, payload, _ = _adapt_separation_to_optimization(source, scope_user_query="Optimize the shortlisted route.")

    assert payload["route_pool_mode"] == "exact"


def test_retry_constrained_objective_rebuilds_after_initial_failure(monkeypatch):
    attempts = {"n": 0}
    seen_options: list[dict | None] = []

    def fake_builder():
        return object(), True, None

    def fake_solve(model, objective, *, solver_options=None):
        attempts["n"] += 1
        seen_options.append(solver_options)
        if attempts["n"] == 1:
            return None
        return {"total_cost": 123.0}

    monkeypatch.setattr("strap.tools.waste_optimization._solve_objective_with_fallback", fake_solve)

    result, reason = _retry_constrained_objective(fake_builder, "min_total_cost", max_attempts=3)

    assert reason is None
    assert result == {"total_cost": 123.0}
    assert attempts["n"] == 2
    assert seen_options == [None, {"presolving/maxrounds": 0}]


def test_retry_constrained_objective_recovers_after_solver_exception(monkeypatch):
    calls: list[tuple[str | None, dict | None]] = []

    def fake_builder():
        return object(), True, None

    def fake_solve_single(model, objective, solver_name="gurobi", solver_options=None):
        calls.append((solver_name, solver_options))
        if solver_options is None:
            raise RuntimeError("scip numerical failure")
        return {"total_cost": 321.0}

    monkeypatch.setattr("strap.tools.waste_optimization.solve_single", fake_solve_single)

    result, reason = _retry_constrained_objective(fake_builder, "min_total_cost", max_attempts=3)

    assert reason is None
    assert result == {"total_cost": 321.0}
    assert calls == [
        ("scip", None),
        ("scip", {"presolving/maxrounds": 0}),
        ("scip", {"presolving/maxrounds": 0, "randomization/randomseedshift": 1}),
    ]


def test_solve_objective_with_fallback_retries_scip_option_ladder(monkeypatch):
    calls: list[tuple[str | None, dict | None]] = []

    class DummyModel:
        def clone(self):
            return DummyModel()

    def fake_solve_single(model, objective, solver_name="gurobi", solver_options=None):
        calls.append((solver_name, solver_options))
        if solver_name == "scip" and solver_options is None:
            raise RuntimeError("default scip numerical failure")
        if solver_name == "scip" and solver_options in (
            {"presolving/maxrounds": 0},
            {"presolving/maxrounds": 0, "randomization/randomseedshift": 1},
        ):
            return {"total_cost": 456.0}
        raise AssertionError(f"Unexpected solver path: {(solver_name, solver_options)}")

    monkeypatch.setattr("strap.tools.waste_optimization.solve_single", fake_solve_single)

    result = _solve_objective_with_fallback(DummyModel(), "max_circularity")

    assert result == {"total_cost": 456.0}
    assert calls == [
        ("scip", None),
        ("scip", {"presolving/maxrounds": 0}),
        ("scip", {"presolving/maxrounds": 0, "randomization/randomseedshift": 1}),
    ]


def test_run_pareto_with_route_pool_retries_sweep_with_scip_option_ladder(monkeypatch):
    context = {
        "data": {"sets": {"P": ["PE", "EVOH"], "S": ["Xylene", "Ethylene Glycol"]}},
        "config": {"dummy": True},
        "scenario": "A",
        "fractions": {"PE": 0.6, "PET": 0.1, "N6": 0.0, "EVOH": 0.3},
        "constraint_mode": "ranked_soft",
        "fallback_policy": "broaden_disclosed",
        "requested_filters": {"PE": ["Xylene"], "EVOH": ["Ethylene Glycol"]},
        "applied_filters": {"PE": ["Xylene"], "EVOH": ["Ethylene Glycol"]},
        "filter_warnings": [],
        "filter_status": "applied",
        "simulation_failures": [],
        "strap_table_rows": 2,
    }
    route_candidates = [
        {
            "route_id": "route_1",
            "rank": 1,
            "sequence": ["PE", "EVOH"],
            "polymer_solvent_map": {"PE": "Xylene", "EVOH": "Ethylene Glycol"},
        }
    ]

    monkeypatch.setattr("strap.tools.waste_optimization.build_model", lambda data, config: object())
    monkeypatch.setattr(
        "strap.tools.waste_optimization._apply_route_pool_constraints",
        lambda model, routes, **kwargs: (True, None),
    )
    monkeypatch.setattr(
        "strap.tools.waste_optimization._solve_objective_with_fallback",
        lambda model, objective, *, solver_options=None: {
            "min_total_cost": {"total_cost": 1000.0, "emissions": 120.0, "CE": 200000.0},
            "max_circularity": {"total_cost": 1800.0, "emissions": 80.0, "CE": 450000.0},
        }[objective],
    )

    sweep_options: list[dict | None] = []

    def fake_pareto(*args, **kwargs):
        sweep_options.append(kwargs.get("solver_options"))
        if len(sweep_options) == 1:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "epsilon": 200000.0,
                    "profit": 10.0,
                    "emissions": 110.0,
                    "CE": 300000.0,
                    "total_cost": 1100.0,
                    "capital_cost": 400.0,
                    "operational_cost": 500.0,
                    "transportation_cost": 200.0,
                    "stage1": ["st1"],
                    "stage2": ["st2"],
                    "stage3": ["lf"],
                    "wash1": ["PE-Xylene"],
                    "wash2": ["EVOH-Ethylene Glycol"],
                },
                {
                    "epsilon": 450000.0,
                    "profit": 9.0,
                    "emissions": 95.0,
                    "CE": 450000.0,
                    "total_cost": 1400.0,
                    "capital_cost": 450.0,
                    "operational_cost": 700.0,
                    "transportation_cost": 250.0,
                    "stage1": ["st1"],
                    "stage2": ["st2"],
                    "stage3": ["gas_h2"],
                    "wash1": ["PE-Xylene"],
                    "wash2": ["EVOH-Ethylene Glycol"],
                },
            ]
        )

    monkeypatch.setattr("strap.tools.waste_optimization.pareto_cost_vs_ce", fake_pareto)

    raw = _run_pareto_with_route_pool(
        context,
        route_candidates=route_candidates,
        feed=8000,
        x_metric="total_cost",
        y_metric="circularity_score",
        n_points=8,
    )
    payload = json.loads(raw)["data"]

    assert payload["n_points_raw_feasible"] == 2
    assert payload["n_points_feasible"] == 2
    assert sweep_options == [
        {"limits/time": 30},
        {"presolving/maxrounds": 0, "limits/time": 30},
    ]


def test_build_model_can_select_pet_recovery_with_generic_polymer_sets():
    strap_df = pd.DataFrame(
        [
            {
                "Wash number": "Wash 1",
                "Polymer": "PE",
                "Solvents": "Cyclohexane",
                **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
                "CAPEX [USD/yr]": 5_000_000.0,
                "OPEX [USD/yr]": 2_000_000.0,
                "GWP [tonCO2e/yr]": 5_000.0,
            },
            {
                "Wash number": "Wash 1",
                "Polymer": "PET",
                "Solvents": "N,N-Dimethylformamide",
                **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
                "CAPEX [USD/yr]": 50_000.0,
                "OPEX [USD/yr]": 25_000.0,
                "GWP [tonCO2e/yr]": 100.0,
            },
        ]
    )
    data = load_all_data(
        excel_path=_EXCEL_PATH,
        strap_sheet="StrapScenario3 Units",
        other_sheet="Othertech w TransportA",
        p_strap=1.0,
        strap_df=strap_df,
    )
    config = dict(_CONFIG)
    config["polymer_fractions"] = {"PE": 0.0, "PET": 1.0}
    config["polymer_market_values_per_ton"] = {"PE": 0.0, "PET": 2000.0}
    config["polymer_recovery_yields"] = {"PE": 0.97, "PET": 0.97}

    model = build_model(data, config)
    result = solve_single(model, "max_profit", solver_name="scip")

    assert result is not None
    assert (
        "PET-N,N-Dimethylformamide" in result["wash1_selection"]
        or "PET-N,N-Dimethylformamide" in result["wash2_selection"]
    )


def test_build_model_can_select_extended_polymer_recovery_with_generic_polymer_sets():
    strap_df = pd.DataFrame(
        [
            {
                "Wash number": "Wash 1",
                "Polymer": "PP",
                "Solvents": "Toluene",
                **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
                "CAPEX [USD/yr]": 50_000.0,
                "OPEX [USD/yr]": 25_000.0,
                "GWP [tonCO2e/yr]": 100.0,
            },
            {
                "Wash number": "Wash 1",
                "Polymer": "PC",
                "Solvents": "Toluene",
                **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
                "CAPEX [USD/yr]": 50_000.0,
                "OPEX [USD/yr]": 25_000.0,
                "GWP [tonCO2e/yr]": 100.0,
            },
        ]
    )
    data = load_all_data(
        excel_path=_EXCEL_PATH,
        strap_sheet="StrapScenario3 Units",
        other_sheet="Othertech w TransportA",
        p_strap=1.0,
        strap_df=strap_df,
    )
    config = dict(_CONFIG)
    config["polymer_fractions"] = {"PP": 0.5, "PC": 0.5}
    config["polymer_market_values_per_ton"] = {"PP": 5000.0, "PC": 5000.0}
    config["polymer_recovery_yields"] = {"PP": 0.97, "PC": 0.97}

    model = build_model(data, config)
    result = solve_single(model, "max_profit", solver_name="scip")

    assert result is not None
    recovered = set(result["wash1_selection"] + result["wash2_selection"])
    assert recovered & {"PP-Toluene", "PC-Toluene"}


def test_build_model_handles_empty_strap_solvent_set_without_boolean_constraint():
    empty_strap_df = pd.DataFrame(
        columns=["Wash number", "Polymer", "Solvents", *_NUMERIC_WORKBOOK_COLUMNS]
    )
    data = load_all_data(
        excel_path=_EXCEL_PATH,
        strap_sheet="StrapScenario3 Units",
        other_sheet="Othertech w TransportA",
        p_strap=1.0,
        strap_df=empty_strap_df,
    )
    config = dict(_CONFIG)
    config["polymer_fractions"] = {"PP": 1.0}
    config["polymer_market_values_per_ton"] = {"PP": 1000.0}
    config["polymer_recovery_yields"] = {"PP": 0.97}

    model = build_model(data, config)

    assert len(list(model.S)) == 0
    assert len(list(model.P)) >= 1


def test_run_waste_management_optimization_optimize_feed_mode_selects_best_composition(monkeypatch):
    def fake_prepare_context(**kwargs):
        fractions = waste_optimization._build_feed_fraction_map(
            pe_fraction=kwargs["pe_fraction"],
            pet_fraction=kwargs["pet_fraction"],
            n6_fraction=kwargs["n6_fraction"],
            evoh_fraction=kwargs["evoh_fraction"],
            feed_composition_json=kwargs.get("feed_composition_json"),
        )
        return {
            "temp_dir": None,
            "scenario": kwargs["scenario"],
            "fractions": fractions,
            "requested_filters": {},
            "applied_filters": {},
            "filter_warnings": [],
            "filter_status": "not_requested",
            "constraint_mode": kwargs.get("constraint_mode") or "soft",
            "fallback_policy": kwargs.get("fallback_policy") or "broaden_disclosed",
        }

    def fake_solve_point(context, *, objective, min_active_washes=None, max_active_washes=None):
        assert objective == "max_profit"
        assert min_active_washes == 1
        assert max_active_washes == 2
        pet_fraction = context["fractions"].get("PET", 0.0)
        profit = 1_000_000.0 * pet_fraction
        return {
            "profit": profit,
            "emissions": 1000.0 - 100.0 * pet_fraction,
            "CE": 200000.0 + 100000.0 * pet_fraction,
            "total_cost": 100000.0,
            "wash1_selection": ["PET-N,N-Dimethylformamide"],
            "wash2_selection": [],
            "stage1_tech": ["st1"],
            "stage2_tech": ["lf"],
            "stage3_tech": [],
        }

    monkeypatch.setattr("strap.tools.waste_optimization._prepare_optimization_context", fake_prepare_context)
    monkeypatch.setattr("strap.tools.waste_optimization._solve_point_optimum_for_context", fake_solve_point)

    raw = waste_optimization.run_waste_management_optimization(
        feed=8000,
        pe_fraction=0.34,
        pet_fraction=0.33,
        n6_fraction=0.0,
        evoh_fraction=0.33,
        scenario="A",
        objective="max_profit",
        feed_mode="optimize",
        composition_constraints_json={
            "PE": {"min": 0.2},
            "PET": {"min": 0.2},
            "EVOH": {"min": 0.2},
        },
        composition_step=0.2,
        min_active_washes=1,
        max_active_washes=2,
    )
    payload = json.loads(raw)["data"]

    assert payload["analysis_type"] == "composition_optimum"
    assert payload["best_result"]["feed_composition"]["PE"] == 0.2
    assert payload["best_result"]["feed_composition"]["PET"] == 0.6
    assert payload["best_result"]["feed_composition"]["EVOH"] == 0.2
    assert payload["best_result"]["feed_composition"]["N6"] == 0.0
    assert payload["n_compositions_evaluated"] == 6


def test_run_waste_management_optimization_accepts_feed_composition_json_without_legacy_fractions(monkeypatch):
    def fake_prepare_context(**kwargs):
        assert kwargs["pe_fraction"] is None
        assert kwargs["pet_fraction"] is None
        assert kwargs["n6_fraction"] is None
        assert kwargs["evoh_fraction"] is None
        fractions = waste_optimization._build_feed_fraction_map(
            pe_fraction=kwargs["pe_fraction"],
            pet_fraction=kwargs["pet_fraction"],
            n6_fraction=kwargs["n6_fraction"],
            evoh_fraction=kwargs["evoh_fraction"],
            feed_composition_json=kwargs.get("feed_composition_json"),
        )
        return {
            "temp_dir": None,
            "scenario": kwargs["scenario"],
            "fractions": fractions,
            "requested_filters": {},
            "applied_filters": {},
            "filter_warnings": [],
            "filter_status": "not_requested",
            "constraint_mode": kwargs.get("constraint_mode") or "soft",
            "fallback_policy": kwargs.get("fallback_policy") or "broaden_disclosed",
        }

    def fake_solve_point(context, *, objective, min_active_washes=None, max_active_washes=None):
        assert objective == "max_profit"
        return {
            "profit": 12345.0,
            "emissions": 678.0,
            "CE": 250000.0,
            "total_cost": 1000.0,
            "wash1_selection": ["PET-N,N-Dimethylformamide"],
            "wash2_selection": [],
            "stage1_tech": ["st1"],
            "stage2_tech": ["lf"],
            "stage3_tech": [],
        }

    monkeypatch.setattr("strap.tools.waste_optimization._prepare_optimization_context", fake_prepare_context)
    monkeypatch.setattr("strap.tools.waste_optimization._solve_point_optimum_for_context", fake_solve_point)

    raw = waste_optimization.run_waste_management_optimization(
        feed=8000,
        feed_composition_json={"PE": 0.5, "EVOH": 0.3, "PET": 0.2},
        scenario="A",
        objective="max_profit",
    )
    payload = json.loads(raw)["data"]

    assert payload["analysis_type"] == "point_optimum"
    assert payload["feed_composition"]["PE"] == 0.5
    assert payload["feed_composition"]["EVOH"] == 0.3
    assert payload["feed_composition"]["PET"] == 0.2


def test_materialize_optimizer_workbook_rows_preserves_temperature_distinct_variants():
    df = pd.DataFrame(
        [
            {"Wash number": "Wash 1", "Polymer": "PET", "Solvents": "Dimethyl sulfoxide", **_biosteam_columns_template()},
        ]
    )

    expanded = _materialize_optimizer_workbook_rows(
        df,
        allowed_solvents_by_slot={
            ("Wash 1", "PET"): [
                {
                    "solvent": "Dimethyl sulfoxide",
                    "optimizer_option": "Dimethyl sulfoxide @ 135C",
                    "dissolution_temp_c": 135.0,
                    "temperature_source": "upstream_explicit",
                },
                {
                    "solvent": "Dimethyl sulfoxide",
                    "optimizer_option": "Dimethyl sulfoxide @ 145C",
                    "dissolution_temp_c": 145.0,
                    "temperature_source": "upstream_explicit",
                },
            ]
        },
    )

    pet_rows = expanded.loc[
        expanded["Polymer"].eq("PET") & expanded["Wash number"].eq("Wash 1"),
        ["Solvents", "actual_solvent", "dissolution_temperature_c"],
    ]
    assert "Dimethyl sulfoxide @ 135C" in pet_rows["Solvents"].astype(str).tolist()
    assert "Dimethyl sulfoxide @ 145C" in pet_rows["Solvents"].astype(str).tolist()
    assert set(pet_rows["actual_solvent"].astype(str)) == {"Dimethyl sulfoxide"}


def test_run_biosteam_updates_passes_dissolution_temperature_into_build_single_config(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "Wash number": "Wash 1",
                "Polymer": "PET",
                "Solvents": "Dimethyl sulfoxide @ 135C",
                "actual_solvent": "Dimethyl sulfoxide",
                "dissolution_temperature_c": 135.0,
                "temperature_source": "upstream_explicit",
                **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
                "coefficient_source": "materialized_clone",
            },
        ]
    )

    captured = {}

    def fake_build_config(**kwargs):
        captured.update(kwargs)
        return {"solvent": kwargs["solvent"]}

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_config)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config: {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.5},
            "operations": {"total_energy_mj_per_kg": 2.0},
        },
    )

    updated_df, failures, simulation_skips = _run_biosteam_updates(
        df,
        polymer_capacities={"PET": 1000.0},
        polymer_fraction_pcts={"PET": 90.0},
    )

    assert failures == []
    assert simulation_skips == []
    assert not updated_df.empty
    assert captured["solvent"] == "Dimethyl sulfoxide"
    assert captured["dissolution_temp_c"] == 135.0



def test_run_biosteam_updates_omits_null_precipitation_temperature(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "Wash number": "Wash 1",
                "Polymer": "PE",
                "Solvents": "Cyclohexane",
                "actual_solvent": "Cyclohexane",
                "dissolution_temperature_c": pd.NA,
                "precipitation_temperature_c": pd.NA,
                "temperature_source": "biosteam_default",
                **{column: 0.0 for column in _NUMERIC_WORKBOOK_COLUMNS},
                "coefficient_source": "materialized_clone",
            },
        ]
    )

    captured = {}

    def fake_build_config(**kwargs):
        captured.update(kwargs)
        return {"solvent": kwargs["solvent"]}

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_config)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config: {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.5},
            "operations": {"total_energy_mj_per_kg": 2.0},
        },
    )

    updated_df, failures, simulation_skips = _run_biosteam_updates(
        df,
        polymer_capacities={"PE": 1000.0},
        polymer_fraction_pcts={"PE": 60.0},
    )

    assert failures == []
    assert simulation_skips == []
    assert not updated_df.empty
    assert captured["solvent"] == "Cyclohexane"
    assert "precipitation_temp_c" not in captured


def test_prepare_optimization_context_keeps_typed_candidates_polymer_scoped(monkeypatch):
    captured_calls = []

    def fake_build_single_config(**kwargs):
        captured_calls.append(kwargs.copy())
        return kwargs.copy()

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_single_config)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config, timeout=None: {"success": False, "error_type": "Synthetic", "error": "synthetic"},
    )

    context = _prepare_optimization_context(
        feed=8000.0,
        pe_fraction=None,
        pet_fraction=None,
        n6_fraction=None,
        evoh_fraction=None,
        feed_composition_json={"LDPE": 0.5, "PET": 0.5},
        scenario="A",
        stage_candidates_json={
            "schema_version": "1.1",
            "constraint_mode": "ranked_soft",
            "fallback_policy": "broaden_disclosed",
            "route_pool_mode": "slot_independent",
            "stages": [
                {
                    "stage_id": "candidate_pool_pe",
                    "target_polymer": "PE",
                    "candidate_pairs": [
                        {"polymer": "PE", "solvent": "Cyclohexane", "optimizer_option": "Cyclohexane @ 120C", "dissolution_temp_c": 120.0},
                        {"polymer": "PE", "solvent": "Hexane", "optimizer_option": "Hexane"},
                    ],
                },
                {
                    "stage_id": "candidate_pool_pet",
                    "target_polymer": "PET",
                    "candidate_pairs": [
                        {"polymer": "PET", "solvent": "Dimethyl sulfoxide", "optimizer_option": "Dimethyl sulfoxide"},
                        {"polymer": "PET", "solvent": "N/A (Solid Residue)", "optimizer_option": "N/A (Solid Residue) @ 120C", "dissolution_temp_c": 120.0},
                    ],
                },
            ],
        },
    )

    assert context.get("infeasible_response") is not None
    assert captured_calls
    assert {call["target_plastic"] for call in captured_calls} <= {"PE", "PET"}
    assert all("@" not in call["solvent"] for call in captured_calls)
    assert all("Solid Residue" not in call["solvent"] for call in captured_calls)
    assert any(call["solvent"] == "Cyclohexane" and call.get("dissolution_temp_c") == 120.0 for call in captured_calls)


def test_prepare_optimization_context_routes_extended_polymers_into_biosteam_configs(monkeypatch):
    captured_configs: list[dict[str, object]] = []

    def fake_build_single_config(**kwargs):
        captured_configs.append(kwargs.copy())
        return kwargs.copy()

    monkeypatch.setattr("strap.tools.waste_optimization.build_single_config", fake_build_single_config)
    monkeypatch.setattr(
        "strap.tools.waste_optimization.run_single_simulation",
        lambda config, timeout=None: {
            "success": True,
            "tea": {"tci_usd": 1_000_000, "aoc_usd_per_yr": 100_000},
            "lca": {"gwp_kg_co2e_per_kg": 0.75},
            "operations": {"total_energy_mj_per_kg": 2.0},
        },
    )

    context = _prepare_optimization_context(
        feed=8000.0,
        pe_fraction=None,
        pet_fraction=None,
        n6_fraction=None,
        evoh_fraction=None,
        feed_composition_json={"PP": 0.25, "PS": 0.25, "PVC": 0.25, "PC": 0.25},
        scenario="A",
        stage_candidates_json={
            "schema_version": "1.1",
            "constraint_mode": "ranked_soft",
            "fallback_policy": "fail_closed",
            "route_pool_mode": "slot_independent",
            "stages": [
                {"stage_id": "candidate_pool_pp", "target_polymer": "PP", "candidate_pairs": [{"polymer": "PP", "solvent": "Toluene"}]},
                {"stage_id": "candidate_pool_ps", "target_polymer": "PS", "candidate_pairs": [{"polymer": "PS", "solvent": "Toluene"}]},
                {"stage_id": "candidate_pool_pvc", "target_polymer": "PVC", "candidate_pairs": [{"polymer": "PVC", "solvent": "N,N-Dimethylformamide"}]},
                {"stage_id": "candidate_pool_pc", "target_polymer": "PC", "candidate_pairs": [{"polymer": "PC", "solvent": "Toluene"}]},
            ],
        },
    )

    assert context.get("infeasible_response") is None
    assert set(context["fractions"]) == {"PP", "PS", "PVC", "PC"}
    assert set(context["config"]["polymer_market_values_per_ton"]) >= {"PP", "PS", "PVC", "PC"}
    assert all(context["config"]["polymer_market_values_per_ton"][polymer] > 0 for polymer in ("PP", "PS", "PVC", "PC"))
    assert set(context["data"]["sets"]["P"]) >= {"PP", "PS", "PVC", "PC"}
    assert set(context["strap_df"]["Polymer"].unique()) == {"PP", "PS", "PVC", "PC"}
    assert {call["target_plastic"] for call in captured_configs} == {"PP", "PS", "PVC", "PC"}
    assert all(call["target_plastic_percent"] == 25.0 for call in captured_configs)
