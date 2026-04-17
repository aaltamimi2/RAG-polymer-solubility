from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from strap.tools.waste_optimization import (
    _NUMERIC_WORKBOOK_COLUMNS,
    _apply_solvent_filters,
    _map_biosteam_to_strap_row,
)
from strap.waste_management.data_loader import load_all_data
from strap.waste_management.model import build_model, estimate_metric_upper_bound
from strap.waste_management.solver import _get_solver, _set_objective, summarize_constraint_residuals


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
