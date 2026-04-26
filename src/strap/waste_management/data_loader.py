"""
Data loader for the multilayer plastic waste management optimization model.
Reads STRAP scenario data and other technology data from the Excel file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
import ast
import re


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------
WASHES = ["Wash 1", "Wash 2"]
POLYMERS = ["PE", "EVOH", "PET", "PP", "PS", "PVC", "PC"]

LEGACY_S_PE = [
    "sec-Butyl Acetate", "Isobutyl Acetate", "Tetrachloroethylene",
    "o-Chlorotoluene", "Methylcyclohexane", "Dodecanol", "Heptane",
    "Toluene", "Xylene",
]

LEGACY_S_EV1 = ["Ethylene Glycol", "Pyridazine"]

LEGACY_S_EV2 = [
    "butane-1,4-diol", "Diethanolamine", "Diethylene glycol",
    "Ethylene Glycol", "Propylene Glycol", "Pyridazine",
    "gamma-butyrolactone",
]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def get_optimizer_default_sets() -> dict[str, list[str]]:
    """Return the default optimizer solvent universe.

    This intentionally exposes the full polymer-specific STRAP/BioSTEAM solvent
    catalogs as the optimizer's starting universe, then lets downstream
    simulation decide what survives. We still append legacy workbook solvents
    so existing validated pathways remain available even when they are not
    present in the current CSV-derived catalog.
    """

    from strap.services.biosteam_service import (
        EVOH_SOLVENTS,
        EVOH_SOLVENTS_E2,
        PC_SOLVENTS,
        PE_SOLVENTS,
        PET_SOLVENTS,
        PP_SOLVENTS,
        PS_SOLVENTS,
        PVC_SOLVENTS,
    )

    s_pe = _dedupe_keep_order(list(PE_SOLVENTS) + LEGACY_S_PE)
    s_ev1 = _dedupe_keep_order(list(EVOH_SOLVENTS) + LEGACY_S_EV1)
    s_ev2 = _dedupe_keep_order(list(EVOH_SOLVENTS_E2) + LEGACY_S_EV2)
    s_pet = _dedupe_keep_order(list(PET_SOLVENTS))
    s_pp = _dedupe_keep_order(list(PP_SOLVENTS))
    s_ps = _dedupe_keep_order(list(PS_SOLVENTS))
    s_pvc = _dedupe_keep_order(list(PVC_SOLVENTS))
    s_pc = _dedupe_keep_order(list(PC_SOLVENTS))
    solvents_by_stage_polymer = {
        "Wash 1": {
            "PE": list(s_pe),
            "EVOH": list(s_ev1),
            "PET": list(s_pet),
            "PP": list(s_pp),
            "PS": list(s_ps),
            "PVC": list(s_pvc),
            "PC": list(s_pc),
        },
        "Wash 2": {
            "PE": list(s_pe),
            "EVOH": list(s_ev2),
            "PET": list(s_pet),
            "PP": list(s_pp),
            "PS": list(s_ps),
            "PVC": list(s_pvc),
            "PC": list(s_pc),
        },
    }
    solvents_by_polymer = {
        polymer: _dedupe_keep_order(
            solvent
            for wash_map in solvents_by_stage_polymer.values()
            for solvent in wash_map.get(polymer, [])
        )
        for polymer in POLYMERS
    }
    return {
        "S_PE": s_pe,
        "S_EV1": s_ev1,
        "S_EV2": s_ev2,
        "S_PET": s_pet,
        "S_PP": s_pp,
        "S_PS": s_ps,
        "S_PVC": s_pvc,
        "S_PC": s_pc,
        "S": _dedupe_keep_order(
            solvent
            for polymer in POLYMERS
            for solvent in solvents_by_polymer.get(polymer, [])
        ),
        "P": list(POLYMERS),
        "W": list(WASHES),
        "S_BY_STAGE_POLYMER": solvents_by_stage_polymer,
        "S_BY_POLYMER": solvents_by_polymer,
    }


def derive_optimizer_sets_from_df(
    df: pd.DataFrame | None,
    *,
    fallback_defaults: bool = True,
) -> dict[str, list[str]]:
    """Derive solver index sets from the actual STRAP sheet rows for this run."""

    defaults = get_optimizer_default_sets() if fallback_defaults else {
        "S_PE": [],
        "S_EV1": [],
        "S_EV2": [],
        "S_PET": [],
        "S_PP": [],
        "S_PS": [],
        "S_PVC": [],
        "S_PC": [],
        "S": [],
        "P": list(POLYMERS),
        "W": list(WASHES),
        "S_BY_STAGE_POLYMER": {
            wash: {polymer: [] for polymer in POLYMERS}
            for wash in WASHES
        },
        "S_BY_POLYMER": {polymer: [] for polymer in POLYMERS},
    }
    if df is None or df.empty:
        return defaults

    solvent_defaults = defaults["S_BY_STAGE_POLYMER"]
    polymers = _dedupe_keep_order(
        [*POLYMERS, *df["Polymer"].dropna().astype(str).tolist()]
    )

    def _extract_solvents(wash: str, polymer: str) -> list[str]:
        mask = df["Wash number"].eq(wash) & df["Polymer"].eq(polymer)
        values = [
            str(solvent).strip()
            for solvent in df.loc[mask, "Solvents"].dropna().astype(str)
            if str(solvent).strip()
        ]
        if values:
            return _dedupe_keep_order(values)
        return list((solvent_defaults.get(wash) or {}).get(polymer, []))

    solvents_by_stage_polymer = {
        wash: {
            polymer: _extract_solvents(wash, polymer)
            for polymer in polymers
        }
        for wash in WASHES
    }
    solvents_by_polymer = {
        polymer: _dedupe_keep_order(
            solvent
            for wash in WASHES
            for solvent in solvents_by_stage_polymer[wash].get(polymer, [])
        )
        for polymer in polymers
    }
    all_solvents = _dedupe_keep_order(
        solvent
        for polymer in polymers
        for solvent in solvents_by_polymer.get(polymer, [])
    )
    s_pe = list(solvents_by_polymer.get("PE", defaults["S_PE"]))
    s_ev1 = list(solvents_by_stage_polymer.get("Wash 1", {}).get("EVOH", defaults["S_EV1"]))
    s_ev2 = list(solvents_by_stage_polymer.get("Wash 2", {}).get("EVOH", defaults["S_EV2"]))
    s_pet = list(solvents_by_polymer.get("PET", defaults["S_PET"]))
    s_pp = list(solvents_by_polymer.get("PP", defaults["S_PP"]))
    s_ps = list(solvents_by_polymer.get("PS", defaults["S_PS"]))
    s_pvc = list(solvents_by_polymer.get("PVC", defaults["S_PVC"]))
    s_pc = list(solvents_by_polymer.get("PC", defaults["S_PC"]))

    return {
        "S_PE": s_pe,
        "S_EV1": s_ev1,
        "S_EV2": s_ev2,
        "S_PET": s_pet,
        "S_PP": s_pp,
        "S_PS": s_ps,
        "S_PVC": s_pvc,
        "S_PC": s_pc,
        "S": all_solvents,
        "P": polymers,
        "W": list(WASHES),
        "S_BY_STAGE_POLYMER": solvents_by_stage_polymer,
        "S_BY_POLYMER": solvents_by_polymer,
    }


_DEFAULT_OPTIMIZER_SETS = get_optimizer_default_sets()
S_PE = list(_DEFAULT_OPTIMIZER_SETS["S_PE"])
S_EV1 = list(_DEFAULT_OPTIMIZER_SETS["S_EV1"])
S_EV2 = list(_DEFAULT_OPTIMIZER_SETS["S_EV2"])
S_PET = list(_DEFAULT_OPTIMIZER_SETS["S_PET"])
S_PP = list(_DEFAULT_OPTIMIZER_SETS["S_PP"])
S_PS = list(_DEFAULT_OPTIMIZER_SETS["S_PS"])
S_PVC = list(_DEFAULT_OPTIMIZER_SETS["S_PVC"])
S_PC = list(_DEFAULT_OPTIMIZER_SETS["S_PC"])
ALL_SOLVENTS = list(_DEFAULT_OPTIMIZER_SETS["S"])
SOLVENTS_BY_STAGE_POLYMER = {
    wash: {polymer: list(solvents) for polymer, solvents in polymer_map.items()}
    for wash, polymer_map in _DEFAULT_OPTIMIZER_SETS["S_BY_STAGE_POLYMER"].items()
}
SOLVENTS_BY_POLYMER = {
    polymer: list(solvents)
    for polymer, solvents in _DEFAULT_OPTIMIZER_SETS["S_BY_POLYMER"].items()
}

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

_OTHERTECH_ROW_BY_TECH = {
    "lf": 1,
    "we": 2,
    "py": 3,
    "gas_er": 4,
    "gas_h2": 5,
    "gas_h2cc": 6,
}
_OTHERTECH_TECH_BY_ROW = {row: tech for tech, row in _OTHERTECH_ROW_BY_TECH.items()}
_OTHERTECH_POSITIONAL_COLS = {
    "total_energy": 1,
    "renewable": 2,
    "direct_ghg": 3,
    "indirect_ghg": 4,
    "water_with": 5,
    "water_recyc": 6,
    "waste": 7,
    "disposal": 8,
    "gwp": 9,
    "htc": 10,
    "htnc": 11,
    "ecot": 12,
    "capex": 13,
    "opex": 14,
}
_TECH_REQUIRED_METRICS = {
    "lf": ("gwp", "opex"),
    "we": ("gwp",),
    "py": ("gwp", "opex"),
    "gas_er": ("gwp", "capex"),
    "gas_h2": ("gwp",),
    "gas_h2cc": ("gwp",),
}


def _coerce_strap_dataframe(
    *,
    excel_path=None,
    sheet_name="StrapScenario3 Units",
    strap_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if strap_df is not None:
        return strap_df.copy()
    return pd.read_excel(excel_path, sheet_name=sheet_name)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_CELL_REF_RE = re.compile(r"\b([A-Z]+)(\d+)\b")


def _column_letters_to_index(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _safe_eval_arithmetic(expr: str) -> float | None:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node_: ast.AST) -> float:
        if isinstance(node_, ast.Expression):
            return _eval(node_.body)
        if isinstance(node_, ast.Constant) and isinstance(node_.value, (int, float)):
            return float(node_.value)
        if isinstance(node_, ast.UnaryOp) and isinstance(node_.op, (ast.UAdd, ast.USub)):
            operand = _eval(node_.operand)
            return operand if isinstance(node_.op, ast.UAdd) else -operand
        if isinstance(node_, ast.BinOp) and isinstance(node_.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(node_.left)
            right = _eval(node_.right)
            if isinstance(node_.op, ast.Add):
                return left + right
            if isinstance(node_.op, ast.Sub):
                return left - right
            if isinstance(node_.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("unsupported expression")

    try:
        return float(_eval(node))
    except Exception:
        return None


def _evaluate_sheet_formula(
    formula: Any,
    formulas_df: pd.DataFrame,
    *,
    row_idx: int,
    cache: dict[tuple[int, int], float | None],
    visiting: set[tuple[int, int]],
) -> float | None:
    if not isinstance(formula, str):
        return _coerce_optional_float(formula)
    text = formula.strip()
    if not text.startswith("="):
        return _coerce_optional_float(text)
    expression = text[1:].strip()
    if not expression:
        return None
    if "[" in expression or "!" in expression:
        return None

    def _replace_ref(match: re.Match[str]) -> str:
        col_letters, row_text = match.groups()
        ref_row = int(row_text) - 1
        ref_col = _column_letters_to_index(col_letters)
        ref_key = (ref_row, ref_col)
        if ref_key in cache:
            ref_value = cache[ref_key]
        else:
            if ref_key in visiting:
                raise ValueError("circular formula reference")
            visiting.add(ref_key)
            ref_formula = formulas_df.iat[ref_row, ref_col]
            ref_value = _evaluate_sheet_formula(
                ref_formula,
                formulas_df,
                row_idx=ref_row,
                cache=cache,
                visiting=visiting,
            )
            visiting.remove(ref_key)
            cache[ref_key] = ref_value
        if ref_value is None:
            raise ValueError("unresolved reference")
        return str(ref_value)

    try:
        substituted = _CELL_REF_RE.sub(_replace_ref, expression)
    except ValueError:
        return None
    return _safe_eval_arithmetic(substituted)


def _read_othertech_sheet(excel_path, sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_excel(excel_path, sheet_name=sheet_name, header=None),
        pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl"),
    )


def _load_othertech_rows(excel_path, sheet_name: str) -> dict[str, dict[str, float | None]]:
    values_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    formulas_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    cache: dict[tuple[int, int], float | None] = {}
    rows: dict[str, dict[str, float | None]] = {}
    for row_idx, tech_key in _OTHERTECH_TECH_BY_ROW.items():
        metric_values: dict[str, float | None] = {}
        for metric, col_idx in _OTHERTECH_POSITIONAL_COLS.items():
            numeric_value = _coerce_optional_float(values_df.iat[row_idx, col_idx])
            if numeric_value is not None:
                metric_values[metric] = numeric_value
                cache[(row_idx, col_idx)] = numeric_value
                continue
            formula_value = _evaluate_sheet_formula(
                formulas_df.iat[row_idx, col_idx],
                formulas_df,
                row_idx=row_idx,
                cache=cache,
                visiting={(row_idx, col_idx)},
            )
            metric_values[metric] = formula_value
            cache[(row_idx, col_idx)] = formula_value
        rows[tech_key] = metric_values
    return rows


def _merge_othertech_rows(
    primary_rows: dict[str, dict[str, float | None]],
    fallback_rows: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    merged: dict[str, dict[str, float | None]] = {}
    for tech in _OTHERTECH_ROW_BY_TECH:
        merged_metrics: dict[str, float | None] = {}
        for metric in _OTHERTECH_POSITIONAL_COLS:
            primary = (primary_rows.get(tech) or {}).get(metric)
            fallback = (fallback_rows.get(tech) or {}).get(metric)
            merged_metrics[metric] = primary if primary is not None else fallback
        merged[tech] = merged_metrics
    return merged


def _tech_has_required_metrics(metrics: dict[str, float | None], tech_key: str) -> bool:
    required = _TECH_REQUIRED_METRICS.get(tech_key, ())
    return all(metrics.get(metric) is not None for metric in required)


def derive_available_othertechs(other_data: dict[str, dict[str, float]]) -> list[str]:
    available: list[str] = []
    for tech_key in OTHERTECH:
        metrics = {metric: values.get(tech_key) for metric, values in other_data.items()}
        if _tech_has_required_metrics(metrics, tech_key):
            available.append(tech_key)
    return available


def load_strap_data(excel_path=None, sheet_name="StrapScenario3 Units", p_strap=1.0, strap_df: pd.DataFrame | None = None):
    """
    Load STRAP wash scenario data.

    Returns a dict of dicts:
        strap_data[metric][(wash, polymer, solvent)] = value * p_strap
    where metric is one of the keys in STRAP_UNIT_COLS.
    """
    df = _coerce_strap_dataframe(excel_path=excel_path, sheet_name=sheet_name, strap_df=strap_df)

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
    primary_rows = _load_othertech_rows(excel_path, sheet_name)
    fallback_rows = (
        primary_rows
        if sheet_name == "Othertech"
        else _load_othertech_rows(excel_path, "Othertech")
    )
    merged_rows = _merge_othertech_rows(primary_rows, fallback_rows)

    other_data = {key: {} for key in OTHER_COLS}
    for tech_key, metrics in merged_rows.items():
        if not _tech_has_required_metrics(metrics, tech_key):
            continue
        for key in OTHER_COLS:
            value = metrics.get(key)
            if value is None:
                value = 0.0
            other_data[key][tech_key] = float(value)

    return other_data


def load_all_data(excel_path=None,
                  strap_sheet="StrapScenario3 Units",
                  other_sheet="Othertech w TransportA",
                  p_strap=1.0,
                  strap_df: pd.DataFrame | None = None):
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
    strap_df : pd.DataFrame | None
        Explicit in-memory STRAP coefficient table. When provided, this is the
        authoritative source for wash coefficients and solvent sets.

    Returns
    -------
    dict with keys:
        "strap": dict of STRAP data arrays
        "strap_df": compiled STRAP coefficient table used for this solve
        "other": dict of other-tech data arrays
        "sets": dict of set definitions
    """
    if excel_path is None:
        excel_path = Path(__file__).parent / "Data for model_Scenarios.xlsx"

    strap_source_df = _coerce_strap_dataframe(excel_path=excel_path, sheet_name=strap_sheet, strap_df=strap_df)
    strap = load_strap_data(excel_path, strap_sheet, p_strap, strap_df=strap_source_df)
    other = load_othertech_data(excel_path, other_sheet)
    available_othertech = derive_available_othertechs(other)
    # When a caller supplies an explicit compiled STRAP table, that table is
    # the authoritative optimization decision space. Falling back to default
    # solvent catalogs for polymers missing from the compiled table creates
    # zero-coefficient ghost wash choices, so defaults are only used for the
    # legacy workbook-only path.
    derived_sets = derive_optimizer_sets_from_df(
        strap_source_df,
        fallback_defaults=strap_df is None,
    )

    sets = {
        "I": ["st1", *available_othertech],
        "J": ["st2", *available_othertech],
        "K": list(available_othertech),
        "othertech": list(available_othertech),
        "P": derived_sets["P"],
        "S": derived_sets["S"],
        "S_PE": derived_sets["S_PE"],
        "S_EV1": derived_sets["S_EV1"],
        "S_EV2": derived_sets["S_EV2"],
        "S_PET": derived_sets["S_PET"],
        "S_PP": derived_sets["S_PP"],
        "S_PS": derived_sets["S_PS"],
        "S_PVC": derived_sets["S_PVC"],
        "S_PC": derived_sets["S_PC"],
        "W": derived_sets["W"],
        "S_BY_STAGE_POLYMER": derived_sets["S_BY_STAGE_POLYMER"],
        "S_BY_POLYMER": derived_sets["S_BY_POLYMER"],
    }

    return {"strap": strap, "strap_df": strap_source_df.copy(), "other": other, "sets": sets}
