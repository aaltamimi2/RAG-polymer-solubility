"""Temperature-dependent solubility interpolation tools.

Thin agent-facing wrappers around the unified solubility API
(strap.solubility).  These format results as Markdown for the LLM.
"""

from __future__ import annotations

import numpy as np

from strap.solubility import (
    _load_coefficients,
    _get_known_names,
    get_all_solvents_selectivity,
    get_boiling_point,
    get_entry,
    get_coefficients_metadata,
    get_solubility,
    predict,
    resolve_polymer,
    resolve_solvent,
)
from strap.tools._helpers import safe_tool_wrapper

# R² threshold: below this the interpolation is unreliable → silent SQL fallback
_R2_THRESHOLD = 0.98


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
        Markdown-formatted prediction with solubility and boiling point.
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
        return f"{polymer} in {solvent} at {temperature_c} C: **Insoluble** (0%)."

    # Anomalous or low-R² → silent SQL fallback
    if cat == "anomalous" or entry.get("r_squared", 0) < _R2_THRESHOLD:
        sol = get_solubility(polymer, solvent, temperature_c, method="sql")
        if sol is None:
            return f"No solubility data available for {polymer} in {solvent} at {temperature_c} C."
        bp = get_boiling_point(solvent)
        bp_str = f"{bp:.1f}" if bp is not None else "N/A"
        return (
            f"{polymer} in {solvent} at {temperature_c} C: "
            f"**{sol:.2f}%** | BP {bp_str} C"
        )

    pred = predict(entry, temperature_c)
    bp = get_boiling_point(solvent)
    bp_str = f"{bp:.1f}" if bp is not None else "N/A"
    r2 = entry.get("r_squared", 0)
    return (
        f"{polymer} in {solvent} at {temperature_c} C: "
        f"**{pred['solubility_pct']:.2f}%** | BP {bp_str} C | R\u00b2 = {r2:.2f}"
    )


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
        return f"**{polymer} in {solvent}**: Insoluble across all temperatures."

    # Anomalous or low-R² → use SQL fallback for the range
    if cat == "anomalous" or entry.get("r_squared", 0) < _R2_THRESHOLD:
        return (
            f"**{polymer} in {solvent}**: Interpolation unavailable for this pair. "
            f"Use `query_database` to look up exact grid-point values."
        )

    bp = get_boiling_point(solvent)
    bp_str = f"BP {bp:.1f} C" if bp is not None else ""

    step = max(t_step_c, 1.0)
    temps = np.arange(t_start_c, t_end_c + step / 2, step)
    if len(temps) > 200:
        temps = temps[:200]

    lines = [
        f"## {polymer} in {solvent} ({t_start_c}–{t_end_c} C) {bp_str}",
        "",
        "| T (C) | Solubility (%) |",
        "|-------|----------------|",
    ]

    for t in temps:
        pred = predict(entry, float(t))
        lines.append(f"| {pred['temperature_c']:.1f} | {pred['solubility_pct']:.2f} |")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Tool 3: Coverage listing
# ------------------------------------------------------------------

@safe_tool_wrapper
def list_interpolation_coverage() -> str:
    """List all polymer-solvent pairs with interpolation data, grouped by polymer.

    Shows category (fitted/insoluble/anomalous) for each pair.
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
        lines.append("| Solvent | Category |")
        lines.append("|---------|----------|")
        for e in entries:
            lines.append(f"| {e['solvent']} | {e['category']} |")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Tool 4: Selectivity ranking for separation
# ------------------------------------------------------------------

@safe_tool_wrapper
def rank_solvents_selectivity(
    target_polymer: str,
    other_polymers: str,
    temperature_c: float = 120.0,
    min_selectivity: float = 5.0,
) -> str:
    """Rank all solvents by how selectively they dissolve one polymer over others.

    Use this to answer "what solvents separate X from Y at T°C" or
    "which solvent best dissolves X but not Y".

    Args:
        target_polymer: The polymer to dissolve (e.g. "LDPE").
        other_polymers: Comma-separated polymers to reject (e.g. "EVOH" or "EVOH,PS").
        temperature_c: Temperature in °C (default 120).
        min_selectivity: Minimum selectivity to include (default 5.0).

    Returns:
        Markdown table of solvents ranked by selectivity (target_sol / max_other_sol).
    """
    _, lookup = _load_coefficients()
    known_p, _ = _get_known_names(lookup)

    target = resolve_polymer(target_polymer, known_p)
    if target is None:
        return (
            f"**Unknown polymer**: '{target_polymer}'\n\n"
            f"Available: {', '.join(sorted(known_p))}"
        )

    others_raw = [s.strip() for s in other_polymers.split(",") if s.strip()]
    others = []
    for name in others_raw:
        p = resolve_polymer(name, known_p)
        if p is None:
            return (
                f"**Unknown polymer**: '{name}'\n\n"
                f"Available: {', '.join(sorted(known_p))}"
            )
        others.append(p)

    if not others:
        return "**Error**: Provide at least one polymer in other_polymers."

    results = get_all_solvents_selectivity(target, others, temperature_c)
    filtered = [r for r in results if r["selectivity"] >= min_selectivity]

    if not filtered:
        return (
            f"No solvents found with selectivity ≥ {min_selectivity} for "
            f"{target} vs {', '.join(others)} at {temperature_c}°C."
        )

    # Enrich with boiling point data from DuckDB (avoids follow-up queries)
    bp_map: dict[str, float] = {}
    try:
        from strap.database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT solvent_name, solvent_name_in_cosmobase, bp__oc_ "
            "FROM solvent_data WHERE bp__oc_ IS NOT NULL"
        ).fetchall()
        for name_val, cosmo_val, bp_val in rows:
            bp = float(bp_val)
            if name_val:
                bp_map[str(name_val).lower().strip()] = bp
            if cosmo_val:
                bp_map[str(cosmo_val).lower().strip()] = bp
    except Exception:
        pass

    # Mark atmospheric feasibility
    atm_note = f"Solvents with BP > {temperature_c}°C are safe for atmospheric operation."
    lines = [
        f"## Solvents for separating {target} from {', '.join(others)} at {temperature_c}°C",
        f"Showing {len(filtered)} solvents with selectivity ≥ {min_selectivity}. {atm_note}",
        "",
        f"| Solvent | {target} Sol. (%) | Max Other Sol. (%) | Selectivity | BP (°C) | Atmospheric? |",
        "|---------|-------------------|--------------------|-------------|---------|-------------|",
    ]
    for r in filtered:
        solvent_lower = r["solvent"].lower().strip()
        bp = bp_map.get(solvent_lower)
        bp_str = f"{bp:.1f}" if bp is not None else "—"
        atm = ""
        if bp is not None:
            atm = "YES" if bp > temperature_c else "NO (above BP)"
        lines.append(
            f"| {r['solvent']} | {r['target_sol']:.2f} | "
            f"{r['max_other_sol']:.2f} | {r['selectivity']:.1f} | "
            f"{bp_str} | {atm} |"
        )

    return "\n".join(lines)
