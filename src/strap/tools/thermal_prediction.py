"""Thermal ML prediction tools for STRAP v7.
Agent-facing wrappers around the thermal ML pipeline (strap.thermal_ml),
COSMO-RS SLE interface (strap.cosmo_interface), and dynamic coefficient
store (strap.solubility).  All tools return Markdown-formatted strings.
"""
from __future__ import annotations
import logging
import math
import numpy as np
from strap.services.tool_response_service import json_tool_error, json_tool_response, json_tool_success
from strap.tools._helpers import safe_tool_wrapper
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _import_thermal_ml():
    """Lazy-import strap.thermal_ml with a friendly error on failure."""
    try:
        from strap.thermal_ml import predict_thermal_properties as _predict
        return _predict
    except ImportError as exc:
        raise ImportError(
            "The thermal ML module is not available. "
            "Install thermal_ml dependencies (torch, transformers) or check "
            f"that strap.thermal_ml is on the Python path. Original error: {exc}"
        ) from exc
def _import_cosmo_interface():
    """Lazy-import strap.cosmo_interface with a friendly error."""
    try:
        from strap.cosmo_interface import run_sle_calculation, list_available_cosmo_files
        return run_sle_calculation, list_available_cosmo_files
    except ImportError as exc:
        raise ImportError(
            "The COSMO-RS interface is not available. "
            f"Check that strap.cosmo_interface is installed. Original error: {exc}"
        ) from exc
def _fit_abc(temps_c: np.ndarray, sols_pct: np.ndarray) -> dict:
    """Fit ln(S%) = A + B/T + C*ln(T) and return coefficients + R^2 (modified Apelblat).
    Mirrors the fitting logic from scripts/fit_solubility_coefficients.py.
    """
    from scipy.optimize import curve_fit
    s_clamped = np.clip(sols_pct, 1e-12, 100.0)
    ln_s = np.log(s_clamped)
    t_k = temps_c + 273.15
    def model(t, a, b, c):
        return a + b / t + c * np.log(t)
    try:
        popt, _ = curve_fit(model, t_k, ln_s, p0=[0, 0, 0], maxfev=10000)
        ln_s_pred = model(t_k, *popt)
        ss_res = np.sum((ln_s - ln_s_pred) ** 2)
        ss_tot = np.sum((ln_s - np.mean(ln_s)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"A": float(popt[0]), "B": float(popt[1]), "C": float(popt[2]), "r_squared": float(r2)}
    except Exception as e:
        return {"A": None, "B": None, "C": None, "r_squared": -1.0, "error": str(e)}
def _fmt_val(value: float, std: float, unit: str) -> str:
    """Format a value +/- std with units, handling NaN gracefully."""
    if math.isnan(value):
        return f"N/A {unit}"
    if math.isnan(std):
        return f"{value:.2f} {unit}"
    return f"{value:.2f} +/- {std:.2f} {unit}"
def _confidence_explanation(confidence: str) -> str:
    """Return a short explanation of the confidence level."""
    explanations = {
        "high": "Low uncertainty and high group coverage -- predictions are reliable.",
        "medium": "Moderate uncertainty or partial group coverage -- use with caution.",
        "low": "High uncertainty or poor group coverage -- treat as rough estimates only.",
    }
    return explanations.get(confidence, "Unknown confidence level.")
def _thermal_response(tool_name: str, display: str, **data) -> str:
    return json_tool_response(display, data, tool_name=tool_name)
def _thermal_error(tool_name: str, message: str, *, error_code: str = "invalid_input", **data) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)
# ------------------------------------------------------------------
# Tool 1: Thermal property prediction
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def predict_thermal_properties(polymer_psmiles: str, polymer_name: str = "Unknown") -> str:
    """Predict melting temperature (Tm), enthalpy of fusion (delta Hf), and heat capacity
    difference (delta Cp) for a polymer from its PSMILES representation.
    Uses Van Krevelen group contribution baselines with optional polyBERT ML
    residual correction. Returns predictions with uncertainty estimates.
    Args:
        polymer_psmiles: Polymer SMILES with [*] attachment points (e.g. "[*]CC[*]" for PE).
        polymer_name: Optional human-readable name for display.
    Returns:
        Markdown-formatted thermal property predictions with confidence level.
    """
    _predict = _import_thermal_ml()
    result = _predict(polymer_psmiles)
    tm_k = result["Tm_K"]
    tm_std = result["Tm_std_K"]
    dhf = result["delta_Hf_J_per_mol"]
    dhf_std = result["delta_Hf_std"]
    dcp = result["delta_Cp_J_per_mol_K"]
    dcp_std = result["delta_Cp_std"]
    method = result["method"]
    confidence = result["confidence"]
    baselines = result.get("group_contribution_baselines", {})
    coverage = baselines.get("coverage", 0.0)
    # Tm in Celsius
    tm_c = tm_k - 273.15 if not math.isnan(tm_k) else float("nan")
    tm_c_std = tm_std  # same magnitude in K and C
    lines = [
        f"## Thermal Properties: {polymer_name}",
        f"**PSMILES**: `{polymer_psmiles}`",
        "",
        "| Property | Value | Uncertainty |",
        "|----------|-------|-------------|",
        f"| Tm | {_fmt_val(tm_k, tm_std, 'K')} ({_fmt_val(tm_c, tm_c_std, 'degC')}) | +/- {tm_std:.2f} K |"
        if not math.isnan(tm_k)
        else "| Tm | N/A | -- |",
        f"| Delta Hf | {_fmt_val(dhf, dhf_std, 'J/mol')} | +/- {dhf_std:.2f} J/mol |"
        if not math.isnan(dhf)
        else "| Delta Hf | N/A | -- |",
        f"| Delta Cp | {_fmt_val(dcp, dcp_std, 'J/(mol*K)')} | +/- {dcp_std:.2f} J/(mol*K) |"
        if not math.isnan(dcp)
        else "| Delta Cp | N/A | -- |",
        "",
        "| Detail | Value |",
        "|--------|-------|",
        f"| Method | {method} |",
        f"| Confidence | **{confidence}** |",
        f"| Group contribution coverage | {coverage:.0%} |",
        "",
        f"> {_confidence_explanation(confidence)}",
    ]
    return json_tool_success(
        "\n".join(lines),
        tool_name="predict_thermal_properties",
        polymer_name=polymer_name,
        polymer_psmiles=polymer_psmiles,
        tm_k=None if math.isnan(tm_k) else tm_k,
        tm_c=None if math.isnan(tm_c) else tm_c,
        tm_std_k=None if math.isnan(tm_std) else tm_std,
        delta_hf_j_per_mol=None if math.isnan(dhf) else dhf,
        delta_hf_std=None if math.isnan(dhf_std) else dhf_std,
        delta_cp_j_per_mol_k=None if math.isnan(dcp) else dcp,
        delta_cp_std=None if math.isnan(dcp_std) else dcp_std,
        method=method,
        confidence=confidence,
        group_contribution_coverage=coverage,
        group_contribution_baselines=baselines,
    )
# ------------------------------------------------------------------
# Tool 2: Solubility for a new polymer
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def generate_solubility_for_new_polymer(
    polymer_name: str,
    polymer_psmiles: str,
    solvent_name: str,
    temperature_c: float = 25.0,
) -> str:
    """Generate solubility prediction for a NEW polymer-solvent pair not in the
    existing database. Runs the full pipeline: ML thermal prediction -> COSMO-RS
    SLE -> coefficient fitting.
    Args:
        polymer_name: Human-readable polymer name (e.g. "PLA", "PEEK").
        polymer_psmiles: Polymer SMILES with [*] attachment points.
        solvent_name: Solvent name (e.g. "toluene", "acetone").
        temperature_c: Temperature for the prediction in degC (default 25).
    Returns:
        Markdown-formatted solubility prediction with fitted coefficients and uncertainty.
    """
    # --- Check if the pair already exists in the static database -----------
    from strap.solubility import _load_coefficients, _get_known_names, resolve_polymer, resolve_solvent
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)
    existing_polymer = resolve_polymer(polymer_name, known_p)
    existing_solvent = resolve_solvent(solvent_name, known_s)
    if existing_polymer and existing_solvent:
        key = (existing_polymer, existing_solvent)
        if key in lookup and lookup[key].get("source") == "static":
            display = (
                f"## {polymer_name} in {solvent_name}\n\n"
                f"This polymer-solvent pair **already exists** in the static "
                f"coefficient database as **{existing_polymer}** / **{existing_solvent}**.\n\n"
                f"Use `predict_solubility(\"{existing_polymer}\", \"{existing_solvent}\", "
                f"{temperature_c})` for predictions from the fitted experimental data."
            )
            return _thermal_response(
                "generate_solubility_for_new_polymer",
                display,
                existing_pair=True,
                static_pair=True,
                polymer_name=polymer_name,
                solvent_name=solvent_name,
                existing_polymer=existing_polymer,
                existing_solvent=existing_solvent,
                temperature_c=temperature_c,
                source="static",
                success=True,
            )
    # --- Step 1: ML thermal prediction ------------------------------------
    _predict = _import_thermal_ml()
    thermal = _predict(polymer_psmiles)
    tm_k = thermal["Tm_K"]
    dhf = thermal["delta_Hf_J_per_mol"]
    dcp = thermal["delta_Cp_J_per_mol_K"]
    thermal_method = thermal["method"]
    thermal_confidence = thermal["confidence"]
    coverage = thermal.get("group_contribution_baselines", {}).get("coverage", 0.0)
    if math.isnan(tm_k) or math.isnan(dhf) or math.isnan(dcp):
        display = (
            f"## {polymer_name} in {solvent_name}\n\n"
            f"**Cannot generate solubility prediction.** The thermal ML model "
            f"could not produce valid Tm/Delta Hf/Delta Cp estimates for "
            f"`{polymer_psmiles}` (group coverage: {coverage:.0%}).\n\n"
            f"Ensure the PSMILES is valid and contains recognised repeat-unit groups."
        )
        return _thermal_error(
            "generate_solubility_for_new_polymer",
            display,
            error_code="thermal_prediction_invalid",
            polymer_name=polymer_name,
            polymer_psmiles=polymer_psmiles,
            solvent_name=solvent_name,
            group_contribution_coverage=coverage,
        )
    # --- Step 2: COSMO-RS SLE calculation ---------------------------------
    run_sle, list_cosmo = _import_cosmo_interface()
    # Try to find COSMO files for this polymer/solvent
    available_cosmo = list_cosmo()
    polymer_cosmo = None
    solvent_cosmo = None
    # Match polymer COSMO file by name (best effort)
    for f in available_cosmo.get("polymers", []):
        if polymer_name.lower().replace(" ", "") in f.lower().replace(" ", ""):
            from pathlib import Path
            from strap.cosmo_interface import _POLYMER_DIR
            polymer_cosmo = str(_POLYMER_DIR / f)
            break
    for f in available_cosmo.get("solvents", []):
        if solvent_name.lower().replace(" ", "") in f.lower().replace(" ", ""):
            from pathlib import Path
            from strap.cosmo_interface import _SOLVENT_DIR
            solvent_cosmo = str(_SOLVENT_DIR / f)
            break
    sle_source = "ideal SLE"
    sle_warning = ""
    if polymer_cosmo is None or solvent_cosmo is None:
        sle_warning = (
            "No COSMO sigma-profile files found -- using **ideal SLE** "
            "(gamma = 1). Solubility will be **over-predicted** because the "
            "non-ideal activity-coefficient correction is absent."
        )
    sle_df = run_sle(
        polymer_cosmo_file=polymer_cosmo,
        solvent_cosmo_file=solvent_cosmo,
        Tm_K=tm_k,
        delta_Hf=dhf,
        delta_Cp=dcp,
        t_range_c=(25, 160),
        t_step_c=5.0,
    )
    if sle_df is not None and not sle_df.empty:
        sle_source = sle_df["source"].iloc[0]
    # --- Step 3: Fit A, B, C coefficients ---------------------------------
    if sle_df is None or sle_df.empty:
        display = (
            f"## {polymer_name} in {solvent_name}\n\n"
            f"SLE calculation returned no data. The COSMO-RS interface may not "
            f"support this polymer-solvent pair."
        )
        return _thermal_error(
            "generate_solubility_for_new_polymer",
            display,
            error_code="sle_no_data",
            polymer_name=polymer_name,
            solvent_name=solvent_name,
            temperature_c=temperature_c,
        )
    temps_c_arr = sle_df["temperature_c"].values
    sols_pct_arr = sle_df["solubility_pct"].values
    fit_result = _fit_abc(temps_c_arr, sols_pct_arr)
    a_coeff = fit_result["A"]
    b_coeff = fit_result["B"]
    c_coeff = fit_result["C"]
    r2 = fit_result["r_squared"]
    # --- Predict at the requested temperature -----------------------------
    if a_coeff is not None:
        t_k_req = temperature_c + 273.15
        ln_s = a_coeff + b_coeff / t_k_req + c_coeff * np.log(t_k_req)
        s_pct = float(np.clip(np.exp(ln_s), 0.0, 100.0))
    else:
        # Fitting failed -- interpolate directly from the SLE curve
        s_pct = float(np.interp(temperature_c, temps_c_arr, sols_pct_arr))
    # --- Format output ----------------------------------------------------
    lines = [
        f"## Predicted Solubility: {polymer_name} in {solvent_name} at {temperature_c} degC",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Solubility | **{s_pct:.4f}%** |",
        f"| SLE source | {sle_source} |",
    ]
    if a_coeff is not None:
        lines.extend([
            "",
            "### Fitted Coefficients (ln S% = A + B/T + C·ln T, modified Apelblat)",
            "",
            "| Coefficient | Value |",
            "|-------------|-------|",
            f"| A | {a_coeff:.6f} |",
            f"| B | {b_coeff:.4f} |",
            f"| C | {c_coeff:.2f} |",
            f"| R^2 | {r2:.6f} |",
            f"| Fit range | {temps_c_arr.min():.0f}--{temps_c_arr.max():.0f} degC |",
            f"| Data points | {len(temps_c_arr)} |",
        ])
    else:
        lines.append(f"| Fit status | **Failed** ({fit_result.get('error', 'unknown')}) |")
    lines.extend([
        "",
        "### Thermal Property Inputs",
        "",
        "| Property | Value | Uncertainty |",
        "|----------|-------|-------------|",
        f"| Tm | {tm_k:.2f} K ({tm_k - 273.15:.2f} degC) | +/- {thermal['Tm_std_K']:.2f} K |",
        f"| Delta Hf | {dhf:.2f} J/mol | +/- {thermal['delta_Hf_std']:.2f} J/mol |",
        f"| Delta Cp | {dcp:.2f} J/(mol*K) | +/- {thermal['delta_Cp_std']:.2f} J/(mol*K) |",
        f"| Method | {thermal_method} |",
        f"| Confidence | **{thermal_confidence}** |",
        f"| Group coverage | {coverage:.0%} |",
    ])
    if sle_warning:
        lines.extend(["", f"> **Warning**: {sle_warning}"])
    if thermal_confidence == "low":
        lines.extend([
            "",
            "> **Warning**: Thermal property confidence is **low**. "
            "The solubility prediction inherits this uncertainty and should "
            "be treated as a rough order-of-magnitude estimate.",
        ])
    return json_tool_success(
        "\n".join(lines),
        tool_name="generate_solubility_for_new_polymer",
        polymer_name=polymer_name,
        polymer_psmiles=polymer_psmiles,
        solvent_name=solvent_name,
        temperature_c=temperature_c,
        solubility_pct=s_pct,
        sle_source=sle_source,
        sle_warning=sle_warning or None,
        thermal_method=thermal_method,
        thermal_confidence=thermal_confidence,
        group_contribution_coverage=coverage,
        fit_coefficients={
            "A": a_coeff,
            "B": b_coeff,
            "C": c_coeff,
            "r_squared": r2,
            "fit_error": fit_result.get("error"),
            "n_points": len(temps_c_arr),
            "t_min_c": float(temps_c_arr.min()),
            "t_max_c": float(temps_c_arr.max()),
        },
        thermal_inputs={
            "Tm_K": tm_k,
            "Tm_std_K": thermal["Tm_std_K"],
            "delta_Hf_J_per_mol": dhf,
            "delta_Hf_std": thermal["delta_Hf_std"],
            "delta_Cp_J_per_mol_K": dcp,
            "delta_Cp_std": thermal["delta_Cp_std"],
        },
    )
# ------------------------------------------------------------------
# Tool 3: List generated polymers
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def lookup_tg(polymer_query: str, top_k: int = 5) -> str:
    """Look up glass transition temperature (Tg) for a polymer by name, abbreviation,
    structural keyword, or PSMILES string.
    Searches a pre-computed database of ~7,700 polymers with experimental Tg values
    and ML ensemble predictions (polyBERT+MLP, R²=0.888, 5-fold CV).
    Supports fuzzy name matching: "polystyrene", "PS", "PMMA", "nylon", etc.
    Also accepts PSMILES strings directly: "[*]CC([*])c1ccccc1".
    Structural keywords work too: "fluorinated", "aromatic", "siloxane".
    Args:
        polymer_query: Polymer name, abbreviation, structural keyword, or PSMILES.
        top_k: Maximum number of results to return (default 5).
    Returns:
        Markdown table of matching polymers with Tg values and uncertainty.
    """
    from strap.thermal_ml.tg_lookup import search_tg, get_lookup_stats
    results = search_tg(polymer_query, top_k=top_k)
    if not results:
        stats = get_lookup_stats()
        total = stats.get("stats", {}).get("total_entries", "~7,700")
        display = (
            f"## Tg Lookup: \"{polymer_query}\"\n\n"
            f"**No matching polymers found** in the database ({total} entries).\n\n"
            "Try:\n"
            "- A different name or abbreviation (e.g. \"PS\" for polystyrene)\n"
            "- A structural keyword (e.g. \"fluorinated\", \"aromatic\", \"siloxane\")\n"
            "- The PSMILES string directly (e.g. \"[*]CC([*])c1ccccc1\")"
        )
        return _thermal_error(
            "lookup_tg",
            display,
            error_code="polymer_not_found",
            polymer_query=polymer_query,
            top_k=top_k,
            total_entries=total,
        )
    lines = [
        f"## Tg Lookup: \"{polymer_query}\"",
        "",
        f"Found {len(results)} match{'es' if len(results) != 1 else ''}:",
        "",
        "| Polymer | PSMILES | Tg (K) | Tg (°C) | ± std (K) | Match |",
        "|---------|---------|--------|---------|-----------|-------|",
    ]
    for r in results:
        name = ", ".join(r["names"]) if r["names"] else ", ".join(r["tags"][:3])
        psmiles_short = r["psmiles"][:35] + "..." if len(r["psmiles"]) > 35 else r["psmiles"]
        tg_k = r["Tg_K"] if r["Tg_K"] else r["Tg_K_predicted"]
        tg_c = tg_k - 273.15 if tg_k else None
        tg_str = f"{tg_k:.1f}" if tg_k else "N/A"
        tg_c_str = f"{tg_c:.1f}" if tg_c else "N/A"
        std_str = f"{r['Tg_std']:.1f}" if r.get("Tg_std") else "--"
        match_str = r.get("match_type", "")
        lines.append(
            f"| {name} | `{psmiles_short}` | {tg_str} | {tg_c_str} | {std_str} | {match_str} |"
        )
    lines.extend([
        "",
        "> Model: polyBERT+MLP ensemble (R²=0.888, 5-fold CV). "
        "Tg values are experimental where available, ML-predicted otherwise. "
        "Uncertainty (std) is the ensemble spread across folds.",
    ])
    return json_tool_success(
        "\n".join(lines),
        tool_name="lookup_tg",
        polymer_query=polymer_query,
        top_k=top_k,
        matches=results,
        n_matches=len(results),
    )
@safe_tool_wrapper(structured_output=True)
def list_generated_polymers() -> str:
    """List all ML-generated polymer-solvent pairs in the dynamic coefficient store.
    Shows which new polymers have been added via the ML pipeline, their tier
    (dynamic/validated), and confidence levels.
    Returns:
        Markdown table of all generated entries grouped by polymer.
    """
    from strap.solubility import get_dynamic_entries
    entries = get_dynamic_entries()
    if not entries:
        display = (
            "## ML-Generated Polymer Entries\n\n"
            "**No generated entries found.**\n\n"
            "To generate predictions for a new polymer-solvent pair, use:\n"
            "```\n"
            "generate_solubility_for_new_polymer(\n"
            '    polymer_name="PLA",\n'
            '    polymer_psmiles="[*]OC(C)C(=O)[*]",\n'
            '    solvent_name="toluene",\n'
            "    temperature_c=25.0,\n"
            ")\n"
            "```"
        )
        return _thermal_response(
            "list_generated_polymers",
            display,
            n_entries=0,
            n_polymers=0,
            entries=[],
            success=True,
        )
    # Group entries by polymer
    by_polymer: dict[str, list[dict]] = {}
    for entry in entries:
        polymer = entry.get("polymer", "Unknown")
        by_polymer.setdefault(polymer, []).append(entry)
    total = len(entries)
    n_polymers = len(by_polymer)
    lines = [
        "## ML-Generated Polymer Entries",
        "",
        f"**Total entries**: {total} | **Polymers**: {n_polymers}",
        "",
    ]
    for polymer in sorted(by_polymer):
        polymer_entries = sorted(by_polymer[polymer], key=lambda e: e.get("solvent", ""))
        lines.append(f"### {polymer} ({len(polymer_entries)} solvents)")
        lines.append("")
        lines.append("| Solvent | Tier | R^2 | Source | Confidence |")
        lines.append("|---------|------|-----|--------|------------|")
        for e in polymer_entries:
            solvent = e.get("solvent", "?")
            tier = e.get("source", e.get("tier", "dynamic"))
            r2 = e.get("r_squared")
            r2_str = f"{r2:.4f}" if r2 is not None else "--"
            source = e.get("sle_source", e.get("source", "--"))
            confidence = e.get("confidence", "--")
            lines.append(f"| {solvent} | {tier} | {r2_str} | {source} | {confidence} |")
        lines.append("")
    return json_tool_success(
        "\n".join(lines),
        tool_name="list_generated_polymers",
        n_entries=total,
        n_polymers=n_polymers,
        entries=entries,
    )
