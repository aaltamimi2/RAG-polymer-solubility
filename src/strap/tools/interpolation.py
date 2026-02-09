"""Temperature-dependent solubility interpolation tools.

Thin agent-facing wrappers around the unified solubility API
(strap.solubility).  These format results as Markdown for the LLM.
"""

from __future__ import annotations

import numpy as np

from strap.solubility import (
    _load_coefficients,
    _get_known_names,
    get_entry,
    get_coefficients_metadata,
    predict,
    resolve_polymer,
    resolve_solvent,
)
from strap.tools._helpers import safe_tool_wrapper


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
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)

    polymer = resolve_polymer(polymer_name, known_p)
    if polymer is None:
        return (
            f"**Unknown polymer**: '{polymer_name}'\n\n"
            f"Available polymers: {', '.join(sorted(known_p))}"
        )

    solvent = resolve_solvent(solvent_name, known_s)
    if solvent is None:
        return (
            f"**Unknown solvent**: '{solvent_name}'\n\n"
            f"Available solvents: {', '.join(sorted(known_s))}"
        )

    entry = lookup.get((polymer, solvent))
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

    pred = predict(entry, temperature_c)
    lines = [
        f"## {polymer} in {solvent} at {temperature_c}°C",
        "",
        "| Property | Value |",
        "|----------|-------|",
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
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)

    polymer = resolve_polymer(polymer_name, known_p)
    if polymer is None:
        return (
            f"**Unknown polymer**: '{polymer_name}'\n\n"
            f"Available polymers: {', '.join(sorted(known_p))}"
        )

    solvent = resolve_solvent(solvent_name, known_s)
    if solvent is None:
        return (
            f"**Unknown solvent**: '{solvent_name}'\n\n"
            f"Available solvents: {', '.join(sorted(known_s))}"
        )

    entry = lookup.get((polymer, solvent))
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
        pred = predict(entry, float(t))
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
    coeffs = get_coefficients_metadata()
    _, lookup = _load_coefficients()

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
