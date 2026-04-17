"""
Data loader for the multilayer plastic waste management optimization model.
Reads STRAP scenario data and other technology data from the Excel file.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------
WASHES = ["Wash 1", "Wash 2"]
POLYMERS = ["PE", "EVOH"]

S_PE = [
    "sec-Butyl Acetate", "Isobutyl Acetate", "Tetrachloroethylene",
    "o-Chlorotoluene", "Methylcyclohexane", "Dodecanol", "Heptane",
    "Toluene", "Xylene",
]

S_EV1 = ["Ethylene Glycol", "Pyridazine"]

S_EV2 = [
    "butane-1,4-diol", "Diethanolamine", "Diethylene glycol",
    "Ethylene Glycol", "Propylene Glycol", "Pyridazine",
    "gamma-butyrolactone",
]

ALL_SOLVENTS = [
    *S_PE,
    "Ethylene Glycol", "Pyridazine",
    "butane-1,4-diol", "Diethanolamine", "Diethylene glycol",
    "Propylene Glycol", "gamma-butyrolactone",
]

# Technology sets for three-stage superstructure
I_SET = ["st1", "lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]
J_SET = ["st2", "lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]
K_SET = ["lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]
OTHERTECH = ["lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]

# Column name mappings: short key -> Excel header
STRAP_UNIT_COLS = {
    "total_energy":  "Total Energy Consumed [MJ/yr]",
    "renewable":     "Total Renewable Energy Consumed [MJ/yr]",
    "direct_ghg":    "Total Direct GHG emissions [Scope 1] [metric tons CO2 equivalent [tCO2e/yr]]",
    "indirect_ghg":  "Total Energy indirect GHG emissions (Scope 2) [metric tons CO2 equivalent (t CO2e/yr)]",
    "water_with":    "Water consumed/discarded [m3/yr]",
    "water_recyc":   "Water Recycled or Reused [m3/yr]",
    "waste":         "Waste generated - Non Hazardous [kg/yr]",
    "disposal":      "Waste to Disposal[ton/yr]",
    "gwp":           "GWP [tonCO2e/yr]",
    "htc":           "Human toxicity cancer [CTUh/yr]",
    "htnc":          "Human toxicity non-cancer [CTUh/yr]",
    "ecot":          "Ecotoxicity [CTUe/yr]",
    "capex":         "CAPEX [USD/yr]",
    "opex":          "OPEX [USD/yr]",
}

OTHER_COLS = {
    "total_energy":  "Total Energy Consumed [MJ/ton resin]",
    "renewable":     "Total Renewable Energy Consumed [MJ/ton resin]",
    "direct_ghg":    "Total Direct GHG emissions [Scope 1] [metric tons CO2 equivalent [tCO2e/ton]]",
    "indirect_ghg":  "Total Energy indirect GHG emissions (Scope 2) [metric tons CO2 equivalent (t CO2e)]",
    "water_with":    "Water consumed/discarded [m3/ton]",
    "water_recyc":   "Water Recycled or Reused [m3/ton]",
    "waste":         "Waste generated - Non Hazardous [kg/ton]",
    "disposal":      "Waste to Disposal[ton/ton]",
    "gwp":           "GWP [tonCO2e/ton resin]",
    "htc":           "Human toxicity cancer [CTUh/ton resin]",
    "htnc":          "Human toxicity non-cancer [CTUh/ton resin]",
    "ecot":          "Ecotoxicity [CTUe/ton resin]",
    "capex":         "CAPEX [USD/yr]",
    "opex":          "OPEX [USD/ton]",
}

TECH_NAME_MAP = {
    "Landfill": "lf",
    "Incineration": "we",
    "Pyrolysis": "py",
    "Gasif. For Energy Recovery": "gas_er",
    "Gasif. For H2": "gas_h2",
    "Gasif. For H2 + CCS": "gas_h2cc",
}


def load_strap_data(excel_path, sheet_name="StrapScenario3 Units", p_strap=1.0):
    """
    Load STRAP wash scenario data.

    Returns a dict of dicts:
        strap_data[metric][(wash, polymer, solvent)] = value * p_strap
    where metric is one of the keys in STRAP_UNIT_COLS.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    strap_data = {key: {} for key in STRAP_UNIT_COLS}

    for _, row in df.iterrows():
        w = row["Wash number"]
        p = row["Polymer"]
        s = row["Solvents"]
        if pd.isna(w) or pd.isna(p) or pd.isna(s):
            continue
        for key, col in STRAP_UNIT_COLS.items():
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            strap_data[key][(w, p, s)] = float(val) * p_strap

    return strap_data


def load_othertech_data(excel_path, sheet_name="Othertech w TransportA"):
    """
    Load data for non-STRAP technologies (landfill, incineration, pyrolysis,
    gasification variants).

    Returns a dict of dicts:
        other_data[metric][tech_key] = value
    where tech_key is one of: lf, we, py, gas_er, gas_h2, gas_h2cc.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    other_data = {key: {} for key in OTHER_COLS}

    for _, row in df.iterrows():
        tech_name = row.iloc[0]  # "Technology" column
        if pd.isna(tech_name) or tech_name not in TECH_NAME_MAP:
            continue
        tech_key = TECH_NAME_MAP[tech_name]
        for key, col in OTHER_COLS.items():
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            other_data[key][tech_key] = float(val)

    return other_data


def load_all_data(excel_path=None,
                  strap_sheet="StrapScenario3 Units",
                  other_sheet="Othertech w TransportA",
                  p_strap=1.0):
    """
    Convenience function to load all model data at once.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file. Defaults to "Data for model_Scenarios.xlsx"
        in the same directory as this script.
    strap_sheet : str
        Sheet name for STRAP scenario data with units in /yr.
    other_sheet : str
        Sheet name for other technologies data.
    p_strap : float
        STRAP capacity fraction (1.0 = dedicated, 0.4 = shared).

    Returns
    -------
    dict with keys:
        "strap": dict of STRAP data arrays
        "other": dict of other-tech data arrays
        "sets": dict of set definitions
    """
    if excel_path is None:
        excel_path = Path(__file__).parent / "Data for model_Scenarios.xlsx"

    strap = load_strap_data(excel_path, strap_sheet, p_strap)
    other = load_othertech_data(excel_path, other_sheet)

    sets = {
        "I": I_SET,
        "J": J_SET,
        "K": K_SET,
        "othertech": OTHERTECH,
        "P": POLYMERS,
        "S": ALL_SOLVENTS,
        "S_PE": S_PE,
        "S_EV1": S_EV1,
        "S_EV2": S_EV2,
        "W": WASHES,
    }

    return {"strap": strap, "other": other, "sets": sets}
