"""Temperature-dependent solubility interpolation tools.

Pre-fitted ln(S) = A + B/T_K + C/T_K^2 coefficients enable instant
solubility predictions at arbitrary temperatures without SQL queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from strap.tools._helpers import safe_tool_wrapper

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy-loaded coefficient singleton
# ------------------------------------------------------------------

_COEFFICIENTS: Optional[dict] = None
_LOOKUP: Optional[dict[tuple[str, str], dict]] = None

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "solubility_coefficients.json"


def _load_coefficients() -> tuple[dict, dict[tuple[str, str], dict]]:
    """Load JSON and build normalized lookup dict."""
    global _COEFFICIENTS, _LOOKUP
    if _COEFFICIENTS is not None:
        return _COEFFICIENTS, _LOOKUP

    with open(_DATA_PATH) as f:
        _COEFFICIENTS = json.load(f)

    _LOOKUP = {}
    for entry in _COEFFICIENTS["entries"]:
        key = (entry["polymer"].strip().upper(), entry["solvent"].strip().lower())
        _LOOKUP[key] = entry

    return _COEFFICIENTS, _LOOKUP


# ------------------------------------------------------------------
# Fuzzy name matching
# ------------------------------------------------------------------

# Common aliases: display name → CSV key (lowercase)
_SOLVENT_ALIASES: dict[str, str] = {
    "water": "h2o",
    "acetone": "propanone",
    "dmf": "dimethylformamide",
    "n,n-dimethylformamide": "dimethylformamide",
    "dmso": "dimethylsulfoxide",
    "dimethyl sulfoxide": "dimethylsulfoxide",
    "dichloromethane": "ch2cl2",
    "dcm": "ch2cl2",
    "chloroform": "chcl3",
    "methyl ethyl ketone": "butanone",
    "mek": "butanone",
    "ethyl acetate": "ethylacetate",
    "methyl acetate": "methylacetate",
    "tetrahydrofuran": "thf",
    "tetrahydropyran": "thp",
    "ethylene glycol": "glycol",
    "propylene glycol": "propyleneglycol",
    "1,2-propanediol": "propyleneglycol",
    "diphenyl ether": "diphenylether",
    "o-xylene": "1,2-dimethylbenzene",
    "p-xylene": "1,4-dimethylbenzene",
    "heptane": "n-heptane",
    "n-hexane": "hexane",
    "1-propanol": "propanol",
    "2-propanol": "2-propanol",
    "isopropanol": "2-propanol",
    "ipa": "2-propanol",
    "tert-butyl alcohol": "tert-butanol",
    "2,4-pentanedione": "acetylacetone",
    "2,3-dihydropyran": "2,3-dihydropyran",
}

_POLYMER_ALIASES: dict[str, str] = {
    "POLYETHYLENE": "HDPE",
    "NYLON 6": "NYLON6",
    "NYLON 66": "NYLON66",
    "NYLON-6": "NYLON6",
    "NYLON-66": "NYLON66",
    "PA6": "NYLON6",
    "PA66": "NYLON66",
    "POLYCARBONATE": "PC",
    "POLYSTYRENE": "PS",
    "POLYETHERSULFONE": "PES",
    "POLYVINYLCHLORIDE": "PVC",
    "POLYVINYL CHLORIDE": "PVC",
    "POLYPROPYLENE": "PP",
}


def _resolve_polymer(name: str, known_polymers: set[str]) -> Optional[str]:
    """3-strategy fuzzy match: exact → alias → substring."""
    norm = name.strip().upper()
    if norm in known_polymers:
        return norm
    alias = _POLYMER_ALIASES.get(norm)
    if alias and alias in known_polymers:
        return alias
    for kp in known_polymers:
        if norm in kp or kp in norm:
            return kp
    return None


def _resolve_solvent(name: str, known_solvents: set[str]) -> Optional[str]:
    """3-strategy fuzzy match: exact → alias → substring."""
    norm = name.strip().lower()
    if norm in known_solvents:
        return norm
    alias = _SOLVENT_ALIASES.get(norm)
    if alias and alias in known_solvents:
        return alias
    for ks in known_solvents:
        if norm in ks or ks in norm:
            return ks
    return None


def _get_known_names(lookup: dict) -> tuple[set[str], set[str]]:
    polymers = set()
    solvents = set()
    for (p, s) in lookup:
        polymers.add(p)
        solvents.add(s)
    return polymers, solvents


# ------------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------------

def _predict(entry: dict, temp_c: float) -> dict:
    """Predict solubility for a single temperature using coefficients."""
    t_k = temp_c + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] / t_k**2
    s_pct = np.clip(np.exp(ln_s), 0.0, 100.0)

    extrapolation = ""
    if temp_c < entry["t_min_c"]:
        extrapolation = "below"
    elif temp_c > entry["t_max_c"]:
        extrapolation = "above"

    return {
        "solubility_pct": round(float(s_pct), 6),
        "temperature_c": temp_c,
        "extrapolation": extrapolation,
    }


# ------------------------------------------------------------------
# Tool 1: Single-point prediction
# ------------------------------------------------------------------

@safe_tool_wrapper
def predict_solubility(polymer_name: str, solvent_name: str, temperature_c: float) -> str:
    """Predict polymer solubility (%) at an arbitrary temperature using pre-fitted
    interpolation coefficients. Works for any temperature, not just the 5°C grid
    points in the database.

    Args:
        polymer_name: Polymer name (e.g. "HDPE", "PS", "Nylon6").
        solvent_name: Solvent name (e.g. "toluene", "acetone", "water").
        temperature_c: Temperature in °C (typically 25-160, extrapolation flagged).

    Returns:
        Markdown-formatted prediction with solubility, R², and warnings.
    """
    coeffs, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)

    polymer = _resolve_polymer(polymer_name, known_p)
    if polymer is None:
        return (
            f"**Unknown polymer**: '{polymer_name}'\n\n"
            f"Available polymers: {', '.join(sorted(known_p))}"
        )

    solvent = _resolve_solvent(solvent_name, known_s)
    if solvent is None:
        return (
            f"**Unknown solvent**: '{solvent_name}'\n\n"
            f"Available solvents: {', '.join(sorted(known_s))}"
        )

    key = (polymer, solvent)
    entry = lookup.get(key)
    if entry is None:
        return f"**No data** for {polymer} in {solvent}."

    cat = entry["category"]
    if cat == "insoluble":
        return (
            f"## {polymer} in {solvent} at {temperature_c}°C\n\n"
            f"**Insoluble** — solubility ≈ 0% across all temperatures "
            f"({entry['t_min_c']}–{entry['t_max_c']}°C)."
        )
    if cat == "anomalous":
        return (
            f"## {polymer} in {solvent} at {temperature_c}°C\n\n"
            f"**Anomalous fit** (R² = {entry['r_squared']:.4f} < 0.98). "
            f"The ln(S) = A + B/T + C/T² model does not fit this pair well. "
            f"Use `query_database` for exact grid-point values instead."
        )

    pred = _predict(entry, temperature_c)
    lines = [
        f"## {polymer} in {solvent} at {temperature_c}°C",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Solubility | **{pred['solubility_pct']:.4f}%** |",
        f"| R² | {entry['r_squared']:.6f} |",
        f"| Fit range | {entry['t_min_c']}–{entry['t_max_c']}°C |",
        f"| Data points | {entry['n_points']} |",
    ]
    if pred["extrapolation"]:
        lines.append(
            f"| ⚠ Extrapolation | {pred['extrapolation']} fit range |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Tool 2: Range prediction
# ------------------------------------------------------------------

@safe_tool_wrapper
def predict_solubility_range(
    polymer_name: str,
    solvent_name: str,
    t_start_c: float = 25.0,
    t_end_c: float = 160.0,
    t_step_c: float = 5.0,
) -> str:
    """Predict polymer solubility over a temperature range. Returns a markdown table.

    Args:
        polymer_name: Polymer name (e.g. "HDPE", "PS", "Nylon6").
        solvent_name: Solvent name (e.g. "toluene", "acetone", "water").
        t_start_c: Start temperature in °C (default 25).
        t_end_c: End temperature in °C (default 160).
        t_step_c: Step size in °C (default 5). Minimum 1°C.

    Returns:
        Markdown table of predicted solubilities across the range.
    """
    coeffs, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)

    polymer = _resolve_polymer(polymer_name, known_p)
    if polymer is None:
        return (
            f"**Unknown polymer**: '{polymer_name}'\n\n"
            f"Available polymers: {', '.join(sorted(known_p))}"
        )

    solvent = _resolve_solvent(solvent_name, known_s)
    if solvent is None:
        return (
            f"**Unknown solvent**: '{solvent_name}'\n\n"
            f"Available solvents: {', '.join(sorted(known_s))}"
        )

    key = (polymer, solvent)
    entry = lookup.get(key)
    if entry is None:
        return f"**No data** for {polymer} in {solvent}."

    cat = entry["category"]
    if cat == "insoluble":
        return (
            f"**{polymer} in {solvent}**: Insoluble across all temperatures "
            f"({entry['t_min_c']}–{entry['t_max_c']}°C)."
        )
    if cat == "anomalous":
        return (
            f"**{polymer} in {solvent}**: Anomalous fit (R² = {entry['r_squared']:.4f}). "
            f"Use `query_database` for exact values."
        )

    step = max(t_step_c, 1.0)
    temps = np.arange(t_start_c, t_end_c + step / 2, step)
    if len(temps) > 200:
        temps = temps[:200]

    lines = [
        f"## {polymer} in {solvent} ({t_start_c}–{t_end_c}°C, step {step}°C)",
        f"R² = {entry['r_squared']:.6f} | Fit range: {entry['t_min_c']}–{entry['t_max_c']}°C",
        "",
        "| T (°C) | Solubility (%) | Note |",
        "|--------|----------------|------|",
    ]

    for t in temps:
        pred = _predict(entry, float(t))
        note = ""
        if pred["extrapolation"]:
            note = f"extrapolation ({pred['extrapolation']})"
        lines.append(f"| {pred['temperature_c']:.1f} | {pred['solubility_pct']:.4f} | {note} |")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Tool 3: Coverage listing
# ------------------------------------------------------------------

@safe_tool_wrapper
def list_interpolation_coverage() -> str:
    """List all polymer-solvent pairs with interpolation data, grouped by polymer.

    Shows category (fitted/insoluble/anomalous) and R² for each pair.
    Use this to discover which predictions are available.

    Returns:
        Markdown summary of all 352 pairs grouped by polymer.
    """
    coeffs, lookup = _load_coefficients()

    by_polymer: dict[str, list[dict]] = {}
    for entry in coeffs["entries"]:
        by_polymer.setdefault(entry["polymer"], []).append(entry)

    lines = [
        "## Interpolation Coverage",
        "",
        f"**Total pairs**: {coeffs['n_entries']} | "
        f"Fitted: {coeffs['categories']['fitted']} | "
        f"Insoluble: {coeffs['categories']['insoluble']} | "
        f"Anomalous: {coeffs['categories']['anomalous']}",
        "",
    ]

    for polymer in sorted(by_polymer):
        entries = sorted(by_polymer[polymer], key=lambda e: e["solvent"])
        fitted = [e for e in entries if e["category"] == "fitted"]
        lines.append(f"### {polymer} ({len(fitted)}/{len(entries)} fitted)")
        lines.append("")
        lines.append("| Solvent | Category | R² |")
        lines.append("|---------|----------|----|")
        for e in entries:
            r2 = f"{e['r_squared']:.4f}" if e["r_squared"] is not None else "—"
            lines.append(f"| {e['solvent']} | {e['category']} | {r2} |")
        lines.append("")

    return "\n".join(lines)
