"""Temperature-dependent solubility interpolation tools.
Thin agent-facing wrappers around the unified solubility API
(strap.solubility).  These format results as Markdown for the LLM.
"""
from __future__ import annotations
import numpy as np
from strap.services.tool_response_service import json_tool_error, json_tool_response, json_tool_success
from strap.solubility import (
    FITTED_TEMP_MAX_C,
    FITTED_TEMP_MIN_C,
    RECOMMENDED_EXTRAPOLATION_MAX_C,
    SENSITIVITY_EXTRAPOLATION_MAX_C,
    _load_coefficients,
    _get_known_names,
    get_all_solvents_selectivity,
    get_boiling_point,
    get_entry,
    get_coefficients_metadata,
    get_solubility_pair_exclusion_reason,
    get_solubility,
    predict,
    resolve_polymer,
    resolve_solvent,
    temperature_basis_note,
    temperature_extrapolation_status,
    temperature_use_regime,
)
from strap.tools._helpers import safe_tool_wrapper
# R² threshold: below this the interpolation is unreliable → silent SQL fallback
_R2_THRESHOLD = 0.98


def _bounded_temperature_grid(t_start_c: float, t_end_c: float, t_step_c: float) -> np.ndarray:
    """Return a temperature grid that never exceeds the requested end point."""
    step = max(float(t_step_c), 1.0)
    start = float(t_start_c)
    end = float(t_end_c)
    if end < start:
        return np.array([], dtype=float)
    temps = np.arange(start, end + 1e-9, step, dtype=float)
    if temps.size == 0:
        temps = np.array([start], dtype=float)
    if not np.isclose(float(temps[-1]), end):
        temps = np.append(temps, end)
    return temps


# ------------------------------------------------------------------
# Tool 1: Single-point prediction
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    if temperature_c > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return json_tool_error(
            (
                f"Temperature {temperature_c} C is above the supported sensitivity limit "
                f"of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C for Apelblat extrapolation."
            ),
            tool_name="predict_solubility",
            error_code="temperature_above_supported_extrapolation",
            temperature_c=temperature_c,
            max_temperature_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        )
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)
    polymer = resolve_polymer(polymer_name, known_p)
    if polymer is None:
        message = (
            f"**Unknown polymer**: '{polymer_name}'\n\n"
            f"Available polymers: {', '.join(sorted(known_p))}"
        )
        return json_tool_error(
            message,
            tool_name="predict_solubility",
            error_code="unknown_polymer",
            polymer_name=polymer_name,
            available_polymers=sorted(known_p),
        )
    solvent = resolve_solvent(solvent_name, known_s)
    if solvent is None:
        message = (
            f"**Unknown solvent**: '{solvent_name}'\n\n"
            f"Available solvents: {', '.join(sorted(known_s))}"
        )
        return json_tool_error(
            message,
            tool_name="predict_solubility",
            error_code="unknown_solvent",
            solvent_name=solvent_name,
            available_solvents=sorted(known_s),
        )
    entry = lookup.get((polymer, solvent))
    if entry is None:
        message = f"**No data** for {polymer} in {solvent}."
        return json_tool_error(
            message,
            tool_name="predict_solubility",
            error_code="pair_not_found",
            polymer_name=polymer,
            solvent_name=solvent,
        )
    if reason := get_solubility_pair_exclusion_reason(polymer, solvent):
        return json_tool_error(
            f"**Excluded data-quality pair**: {polymer} in {solvent}.\n\n{reason}",
            tool_name="predict_solubility",
            error_code="excluded_data_quality_pair",
            polymer_name=polymer,
            solvent_name=solvent,
            reason=reason,
        )
    cat = entry["category"]
    if cat == "insoluble":
        display = f"{polymer} in {solvent} at {temperature_c} C: **Insoluble** (0%)."
        return json_tool_success(
            display,
            tool_name="predict_solubility",
            polymer_name=polymer,
            solvent_name=solvent,
            temperature_c=temperature_c,
            category=cat,
            method="lookup",
            solubility_pct=0.0,
        )
    # Anomalous or low-R² → silent SQL fallback
    if cat == "anomalous" or entry.get("r_squared", 0) < _R2_THRESHOLD:
        sol = get_solubility(polymer, solvent, temperature_c, method="sql")
        if sol is None:
            message = f"No solubility data available for {polymer} in {solvent} at {temperature_c} C."
            return json_tool_error(
                message,
                tool_name="predict_solubility",
                error_code="sql_fallback_unavailable",
                polymer_name=polymer,
                solvent_name=solvent,
                temperature_c=temperature_c,
                category=cat,
            )
        bp = get_boiling_point(solvent)
        bp_str = f"{bp:.1f}" if bp is not None else "N/A"
        display = (
            f"{polymer} in {solvent} at {temperature_c} C: "
            f"**{sol:.2f}%** | BP {bp_str} C"
        )
        return json_tool_success(
            display,
            tool_name="predict_solubility",
            polymer_name=polymer,
            solvent_name=solvent,
            temperature_c=temperature_c,
            category=cat,
            method="sql",
            solubility_pct=sol,
            boiling_point_c=bp,
            r_squared=entry.get("r_squared"),
        )
    pred = predict(entry, temperature_c)
    bp = get_boiling_point(solvent)
    bp_str = f"{bp:.1f}" if bp is not None else "N/A"
    r2 = entry.get("r_squared", 0)
    basis_note = temperature_basis_note(temperature_c)
    display = (
        f"{polymer} in {solvent} at {temperature_c} C: "
        f"**{pred['solubility_pct']:.2f}%** | BP {bp_str} C | R\u00b2 = {r2:.2f}"
    )
    atmospheric_operation = None
    if bp is not None:
        atmospheric_operation = temperature_c < bp
        if not atmospheric_operation:
            display += " | Above solvent BP at 1 atm"
    if basis_note:
        display += f" | {basis_note}"
    return json_tool_success(
        display,
        tool_name="predict_solubility",
        polymer_name=polymer,
        solvent_name=solvent,
        temperature_c=temperature_c,
        category=cat,
        method="interpolation",
        solubility_pct=pred["solubility_pct"],
        boiling_point_c=bp,
        atmospheric_operation=atmospheric_operation,
        r_squared=r2,
        extrapolation=pred["extrapolation"] or "none",
        temperature_extrapolation=temperature_extrapolation_status(temperature_c),
        temperature_use_regime=temperature_use_regime(temperature_c),
        fitted_temperature_range_c=[FITTED_TEMP_MIN_C, FITTED_TEMP_MAX_C],
        recommended_extrapolation_max_c=RECOMMENDED_EXTRAPOLATION_MAX_C,
        sensitivity_extrapolation_max_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
    )
# ------------------------------------------------------------------
# Tool 2: Range prediction
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    if t_start_c > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return json_tool_error(
            (
                f"Start temperature {t_start_c} C is above the supported sensitivity limit "
                f"of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C for Apelblat extrapolation."
            ),
            tool_name="predict_solubility_range",
            error_code="temperature_above_supported_extrapolation",
            temperature_c=t_start_c,
            max_temperature_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        )
    requested_t_end_c = t_end_c
    range_was_capped = False
    if t_end_c > SENSITIVITY_EXTRAPOLATION_MAX_C:
        t_end_c = SENSITIVITY_EXTRAPOLATION_MAX_C
        range_was_capped = True
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)
    polymer = resolve_polymer(polymer_name, known_p)
    if polymer is None:
        message = (
            f"**Unknown polymer**: '{polymer_name}'\n\n"
            f"Available polymers: {', '.join(sorted(known_p))}"
        )
        return json_tool_error(
            message,
            tool_name="predict_solubility_range",
            error_code="unknown_polymer",
            polymer_name=polymer_name,
            available_polymers=sorted(known_p),
        )
    solvent = resolve_solvent(solvent_name, known_s)
    if solvent is None:
        message = (
            f"**Unknown solvent**: '{solvent_name}'\n\n"
            f"Available solvents: {', '.join(sorted(known_s))}"
        )
        return json_tool_error(
            message,
            tool_name="predict_solubility_range",
            error_code="unknown_solvent",
            solvent_name=solvent_name,
            available_solvents=sorted(known_s),
        )
    entry = lookup.get((polymer, solvent))
    if entry is None:
        message = f"**No data** for {polymer} in {solvent}."
        return json_tool_error(
            message,
            tool_name="predict_solubility_range",
            error_code="pair_not_found",
            polymer_name=polymer,
            solvent_name=solvent,
        )
    if reason := get_solubility_pair_exclusion_reason(polymer, solvent):
        return json_tool_error(
            f"**Excluded data-quality pair**: {polymer} in {solvent}.\n\n{reason}",
            tool_name="predict_solubility_range",
            error_code="excluded_data_quality_pair",
            polymer_name=polymer,
            solvent_name=solvent,
            reason=reason,
        )
    cat = entry["category"]
    if cat == "insoluble":
        display = f"**{polymer} in {solvent}**: Insoluble across all temperatures."
        return json_tool_success(
            display,
            tool_name="predict_solubility_range",
            polymer_name=polymer,
            solvent_name=solvent,
            category=cat,
            predictions=[],
            n_points=0,
        )
    # Anomalous or low-R² → use SQL fallback for the range
    if cat == "anomalous" or entry.get("r_squared", 0) < _R2_THRESHOLD:
        message = (
            f"**{polymer} in {solvent}**: Interpolation unavailable for this pair. "
            f"Use `query_database` to look up exact grid-point values."
        )
        return json_tool_error(
            message,
            tool_name="predict_solubility_range",
            error_code="interpolation_unavailable",
            polymer_name=polymer,
            solvent_name=solvent,
            category=cat,
            r_squared=entry.get("r_squared"),
        )
    bp = get_boiling_point(solvent)
    bp_str = f"BP {bp:.1f} C" if bp is not None else ""
    step = max(t_step_c, 1.0)
    temps = _bounded_temperature_grid(t_start_c, t_end_c, step)
    if len(temps) > 200:
        temps = temps[:200]
    lines = [
        f"## {polymer} in {solvent} ({t_start_c}–{t_end_c} C) {bp_str}",
        "",
        "| T (C) | Solubility (%) |",
        "|-------|----------------|",
    ]
    if range_was_capped:
        lines.insert(
            2,
            f"Requested end temperature {requested_t_end_c} C was capped at {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C.",
        )
    if bp is not None and t_end_c >= bp:
        lines.insert(
            2,
            f"Atmospheric note: part or all of this range is at/above the solvent BP ({bp:.1f} C).",
        )
    predictions: list[dict[str, float | str]] = []
    extrapolated_points = 0
    for t in temps:
        pred = predict(entry, float(t))
        if pred["extrapolation"]:
            extrapolated_points += 1
        predictions.append(
            {
                "temperature_c": pred["temperature_c"],
                "solubility_pct": pred["solubility_pct"],
                "extrapolation": pred["extrapolation"] or "none",
                "temperature_use_regime": temperature_use_regime(float(t)),
            }
        )
        lines.append(f"| {pred['temperature_c']:.1f} | {pred['solubility_pct']:.2f} |")
    if extrapolated_points:
        lines.append("")
        lines.append(
            f"Note: {extrapolated_points} point(s) are Apelblat extrapolations outside "
            f"the fitted {FITTED_TEMP_MIN_C:.0f}-{FITTED_TEMP_MAX_C:.0f} C range; treat them as lower-confidence estimates."
        )
    return json_tool_success(
        "\n".join(lines),
        tool_name="predict_solubility_range",
        polymer_name=polymer,
        solvent_name=solvent,
        category=cat,
        r_squared=entry.get("r_squared"),
        t_start_c=t_start_c,
        t_end_c=t_end_c,
        requested_t_end_c=requested_t_end_c,
        range_was_capped=range_was_capped,
        t_step_c=step,
        n_points=len(predictions),
        predictions=predictions,
        extrapolated_points=extrapolated_points,
        fitted_temperature_range_c=[FITTED_TEMP_MIN_C, FITTED_TEMP_MAX_C],
        sensitivity_extrapolation_max_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        boiling_point_c=bp,
        atmospheric_range_exceeds_bp=bool(bp is not None and t_end_c >= bp),
    )
# ------------------------------------------------------------------
# Tool 3: Coverage listing
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    return json_tool_success(
        "\n".join(lines),
        tool_name="list_interpolation_coverage",
        n_entries=coeffs["n_entries"],
        categories=coeffs["categories"],
        polymers=sorted(by_polymer),
    )
# ------------------------------------------------------------------
# Tool 4: Selectivity ranking for separation
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    if temperature_c > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return json_tool_error(
            (
                f"Temperature {temperature_c} C is above the supported sensitivity limit "
                f"of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C for Apelblat extrapolation."
            ),
            tool_name="rank_solvents_selectivity",
            error_code="temperature_above_supported_extrapolation",
            temperature_c=temperature_c,
            max_temperature_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        )
    _, lookup = _load_coefficients()
    known_p, _ = _get_known_names(lookup)
    target = resolve_polymer(target_polymer, known_p)
    if target is None:
        message = (
            f"**Unknown polymer**: '{target_polymer}'\n\n"
            f"Available: {', '.join(sorted(known_p))}"
        )
        return json_tool_error(
            message,
            tool_name="rank_solvents_selectivity",
            error_code="unknown_target_polymer",
            polymer_name=target_polymer,
            available_polymers=sorted(known_p),
        )
    others_raw = [s.strip() for s in other_polymers.split(",") if s.strip()]
    others = []
    for name in others_raw:
        p = resolve_polymer(name, known_p)
        if p is None:
            message = (
                f"**Unknown polymer**: '{name}'\n\n"
                f"Available: {', '.join(sorted(known_p))}"
            )
            return json_tool_error(
                message,
                tool_name="rank_solvents_selectivity",
                error_code="unknown_other_polymer",
                polymer_name=name,
                available_polymers=sorted(known_p),
            )
        others.append(p)
    if not others:
        return json_tool_error(
            "**Error**: Provide at least one polymer in other_polymers.",
            tool_name="rank_solvents_selectivity",
            error_code="missing_other_polymers",
        )
    results = get_all_solvents_selectivity(target, others, temperature_c)
    filtered = [r for r in results if r["selectivity"] >= min_selectivity]
    if not filtered:
        display = (
            f"No solvents found with selectivity ≥ {min_selectivity} for "
            f"{target} vs {', '.join(others)} at {temperature_c}°C."
        )
        return json_tool_response(
            display,
            {
                "target_polymer": target,
                "other_polymers": others,
                "temperature_c": temperature_c,
                "min_selectivity": min_selectivity,
                "n_results": 0,
                "solvents": [],
            },
            tool_name="rank_solvents_selectivity",
            success=True,
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
    use_regime = temperature_use_regime(temperature_c)
    high_temperature_screening = use_regime == "sensitivity_extrapolation"
    if high_temperature_screening:
        filtered = [
            r
            for r in filtered
            if (bp := bp_map.get(r["solvent"].lower().strip())) is not None
            and bp >= temperature_c + 5.0
        ]
        if not filtered:
            display = (
                f"No high-boiling solvents found with selectivity ≥ {min_selectivity} for "
                f"{target} vs {', '.join(others)} at {temperature_c}°C while keeping a 5°C boiling-point margin."
            )
            return json_tool_response(
                display,
                {
                    "target_polymer": target,
                    "other_polymers": others,
                    "temperature_c": temperature_c,
                    "temperature_extrapolation": temperature_extrapolation_status(temperature_c),
                    "temperature_use_regime": use_regime,
                    "min_selectivity": min_selectivity,
                    "n_results": 0,
                    "solvents": [],
                    "high_temperature_screening": True,
                    "required_boiling_point_c": temperature_c + 5.0,
                },
                tool_name="rank_solvents_selectivity",
                success=True,
            )
    # Mark atmospheric feasibility and interpolation basis
    atm_note = f"Solvents with BP > {temperature_c}°C are safe for atmospheric operation."
    basis_note = temperature_basis_note(temperature_c)
    lines = [
        f"## Solvents for separating {target} from {', '.join(others)} at {temperature_c}°C",
        f"Showing {len(filtered)} solvents with selectivity ≥ {min_selectivity}. {atm_note}",
        "",
        f"| Solvent | {target} Sol. (%) | Max Other Sol. (%) | Selectivity | BP (°C) | Atmospheric? |",
        "|---------|-------------------|--------------------|-------------|---------|-------------|",
    ]
    if basis_note:
        lines.insert(
            2,
            f"**Temperature basis:** {basis_note}",
        )
    if high_temperature_screening:
        lines.insert(
            3 if basis_note else 2,
            f"**High-temperature screening:** showing only solvents with BP ≥ {temperature_c + 5.0:.0f}°C; do not treat 180–200°C results as validated process recommendations.",
        )
    structured_results: list[dict[str, float | str | None]] = []
    for r in filtered:
        solvent_lower = r["solvent"].lower().strip()
        bp = bp_map.get(solvent_lower)
        bp_str = f"{bp:.1f}" if bp is not None else "—"
        atm = ""
        if bp is not None:
            atm = "YES" if bp > temperature_c else "NO (above BP)"
        structured_results.append(
            {
                "solvent": r["solvent"],
                "target_solubility_pct": r["target_sol"],
                "max_other_solubility_pct": r["max_other_sol"],
                "selectivity": r["selectivity"],
                "boiling_point_c": bp,
                "atmospheric_operation": atm == "YES",
                "temperature_extrapolation": temperature_extrapolation_status(temperature_c),
                "temperature_use_regime": use_regime,
            }
        )
        lines.append(
            f"| {r['solvent']} | {r['target_sol']:.2f} | "
            f"{r['max_other_sol']:.2f} | {r['selectivity']:.1f} | "
            f"{bp_str} | {atm} |"
        )
    return json_tool_success(
        "\n".join(lines),
        tool_name="rank_solvents_selectivity",
        target_polymer=target,
        other_polymers=others,
        temperature_c=temperature_c,
        temperature_extrapolation=temperature_extrapolation_status(temperature_c),
        temperature_use_regime=use_regime,
        fitted_temperature_range_c=[FITTED_TEMP_MIN_C, FITTED_TEMP_MAX_C],
        sensitivity_extrapolation_max_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        high_temperature_screening=high_temperature_screening,
        min_selectivity=min_selectivity,
        n_results=len(structured_results),
        solvents=structured_results,
    )
