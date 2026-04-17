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

    The Excel has a BASE 'Othertech' sheet with most data, and transport-
    specific sheets ('Othertech w TransportA', 'B') that override some values.
    We merge them: base first, then overlay transport-specific non-NaN values.

    Row mapping BY POSITION (Julia uses setnames!):
        row 1=lf, row 2=we, row 3=py, row 4=gas_er, row 5=gas_h2, row 6=gas_h2cc

    Returns a dict of dicts:
        other_data[metric][tech_key] = value
    """
    import numpy as np

    tech_order = ["lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]

    # Column mapping by position (cols 1-14 after Technology col 0)
    col_map = {
        "total_energy":  1,
        "renewable":     2,
        "direct_ghg":    3,
        "indirect_ghg":  4,
        "water_with":    5,
        "water_recyc":   6,
        "waste":         7,
        "disposal":      8,
        "gwp":           9,
        "htc":          10,
        "htnc":         11,
        "ecot":         12,
        "capex":        13,
        "opex":         14,
    }

    other_data = {key: {} for key in col_map}

    def _read_sheet_into(sheet, target):
        """Read sheet by position and store non-NaN values into target dict."""
        df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
        for row_idx, tech_key in enumerate(tech_order):
            data_row = row_idx + 1  # skip header
            if data_row >= len(df_raw):
                continue
            for metric_key, col_idx in col_map.items():
                if col_idx >= len(df_raw.columns):
                    continue
                val = df_raw.iloc[data_row, col_idx]
                if pd.notna(val) and val is not None:
                    target[metric_key][tech_key] = float(val)

    # 1. Read BASE 'Othertech' sheet first (has the most complete data)
    try:
        _read_sheet_into("Othertech", other_data)
    except Exception:
        pass  # base sheet might not exist in all versions

    # 2. Overlay transport-specific sheet (overrides base values where non-NaN)
    try:
        _read_sheet_into(sheet_name, other_data)
    except Exception:
        pass

    # 3. Fill any remaining gaps with 0.0
    for metric_key in col_map:
        for tech_key in tech_order:
            if tech_key not in other_data[metric_key]:
                other_data[metric_key][tech_key] = 0.0

    # 4. GWP fallback: compute from direct_ghg + indirect_ghg if missing
    gwp_sum = sum(abs(other_data["gwp"].get(t, 0)) for t in tech_order)
    if gwp_sum < 1e-10:
        for t in tech_order:
            other_data["gwp"][t] = (
                other_data["direct_ghg"].get(t, 0.0)
                + other_data["indirect_ghg"].get(t, 0.0)
            )

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
