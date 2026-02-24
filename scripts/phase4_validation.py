#!/usr/bin/env python3
"""Phase 4 Validation: End-to-end validation of the STRAP v7 thermal ML pipeline.

Pipeline: PSMILES -> ML Model -> (Tm, dHf, dCp) -> COSMO-RS SLE -> Solubility Curve -> Fit A,B,C

This script validates the pipeline against the existing 352 polymer-solvent pairs.
Since a trained ML model is not yet available and COSMOtherm is not installed,
validation uses group contribution estimates and ideal SLE (no gamma correction)
as a baseline.

Usage:
    python scripts/phase4_validation.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Project root and path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from strap.thermal_ml import predict_thermal_properties, get_group_contribution_estimate
from strap.cosmo_interface import compute_ideal_sle, run_sle_calculation
from strap.solubility import _load_coefficients, predict as solubility_predict
from strap.tools.thermal_prediction import (
    predict_thermal_properties as tool_predict_thermal,
    generate_solubility_for_new_polymer as tool_generate,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol*K)

POLYMER_PSMILES = {
    "HDPE": "[*]CC[*]",
    "LDPE": "[*]CC[*]",
    "PP": "[*]CC(C)[*]",
    "PS": "[*]CC(c1ccccc1)[*]",
    "PVC": "[*]CC(Cl)[*]",
    "PET": "[*]OC(=O)c1ccc(C(=O)OCC[*])cc1",
    "Nylon6": "[*]CCCCCC(=O)N[*]",
    "Nylon66": "[*]NCCCCCCNC(=O)CCCCC(=O)[*]",
    "PC": "[*]OC(=O)Oc1ccc(C(C)(C)c2ccc(O[*])cc2)cc1",
    "PES": "[*]Oc1ccc(S(=O)(=O)c2ccc(O[*])cc2)cc1",
    "EVOH": "[*]CC(O)[*]",
}

TEST_POLYMERS = {
    "PLA": "[*]OC(C)C(=O)[*]",
    "PEEK": "[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1",
    "PVDF": "[*]CC(F)(F)[*]",
}

TEST_SOLVENTS = {
    "PLA": ["toluene", "acetone", "chcl3"],
    "PEEK": ["toluene", "dimethylformamide"],
    "PVDF": ["acetone", "thf"],
}

THERMAL_CSV = _PROJECT_ROOT / "data" / "thermal_properties" / "polymer_thermal_reference.csv"
COEFF_JSON = _PROJECT_ROOT / "data" / "solubility_coefficients.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_abc(temps_c: np.ndarray, sols_pct: np.ndarray) -> dict:
    """Fit ln(S%) = A + B/T + C*ln(T) and return coefficients + R^2 (modified Apelblat)."""
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


def solubility_from_abc(A: float, B: float, C: float, temps_c: np.ndarray) -> np.ndarray:
    """Compute S% from A,B,C coefficients at given temperatures."""
    t_k = temps_c + 273.15
    ln_s = A + B / t_k + C * np.log(t_k)
    return np.clip(np.exp(ln_s), 0.0, 100.0)


def max_abs_error_pct(s1: np.ndarray, s2: np.ndarray) -> float:
    """Max absolute difference between two solubility arrays (% units)."""
    return float(np.max(np.abs(s1 - s2)))


def mean_abs_error_pct(s1: np.ndarray, s2: np.ndarray) -> float:
    """Mean absolute difference between two solubility arrays (% units)."""
    return float(np.mean(np.abs(s1 - s2)))


def load_thermal_reference() -> pd.DataFrame:
    """Load the polymer thermal reference CSV."""
    df = pd.read_csv(THERMAL_CSV)
    # Normalise polymer column to uppercase for matching
    df["polymer_upper"] = df["polymer"].str.strip().str.upper()
    return df


def get_fitted_solvents_for_polymer(polymer_name: str, lookup: dict) -> list[dict]:
    """Get all fitted coefficient entries for a polymer from the lookup."""
    key_upper = polymer_name.strip().upper()
    results = []
    for (p, s), entry in lookup.items():
        if p == key_upper and entry.get("category") == "fitted":
            results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Part 1: Leave-One-Out Validation Against Known Pairs
# ---------------------------------------------------------------------------

def run_part1():
    """Validate GC-estimated and reference thermal properties against COSMO-RS fitted coefficients."""
    print("=" * 80)
    print("PART 1: Leave-One-Out Validation Against Known Pairs")
    print("=" * 80)
    print()

    ref_df = load_thermal_reference()
    _, lookup = _load_coefficients()

    # Temperature grid for comparisons
    temps_c = np.arange(25.0, 165.0, 5.0)

    # Collectors for summary
    thermal_results = []       # per-polymer thermal error
    solubility_results = []    # per-polymer-solvent solubility error

    for polymer_name, psmiles in POLYMER_PSMILES.items():
        polymer_upper = polymer_name.strip().upper()

        # Skip amorphous PES
        ref_row = ref_df[ref_df["polymer_upper"] == polymer_upper]
        if ref_row.empty:
            print(f"  [{polymer_name}] No reference data found -- SKIPPED")
            print()
            continue

        row = ref_row.iloc[0]
        crystalline = str(row.get("crystalline", "")).strip().lower()
        if crystalline == "amorphous":
            print(f"  [{polymer_name}] Amorphous polymer (no Tm/dHf) -- SKIPPED")
            print()
            continue

        # Reference thermal properties
        ref_Tm_K = row.get("Tm0_K")
        ref_dHf_J_mol = row.get("delta_Hf0_J_per_mol")
        ref_dCp_J_g_K = row.get("delta_Cp_J_per_g_K")
        ref_MW = row.get("repeat_unit_MW")

        if pd.isna(ref_Tm_K) or pd.isna(ref_dHf_J_mol):
            print(f"  [{polymer_name}] Missing Tm or dHf in reference -- SKIPPED")
            print()
            continue

        ref_Tm_K = float(ref_Tm_K)
        ref_dHf_J_mol = float(ref_dHf_J_mol)

        # Convert dCp from J/(g*K) to J/(mol*K) if available
        ref_dCp_J_mol_K = 0.0
        if not pd.isna(ref_dCp_J_g_K) and not pd.isna(ref_MW):
            ref_dCp_J_mol_K = float(ref_dCp_J_g_K) * float(ref_MW)

        # GC estimate
        gc = get_group_contribution_estimate(psmiles)
        gc_Tm_K = gc.get("Tm_K", float("nan"))
        gc_dHf = gc.get("delta_Hf_J_per_mol", float("nan"))
        gc_dCp = gc.get("delta_Cp_J_per_mol_K", float("nan"))

        # Thermal errors
        tm_err = abs(gc_Tm_K - ref_Tm_K) if not math.isnan(gc_Tm_K) else float("nan")
        dhf_err = abs(gc_dHf - ref_dHf_J_mol) if not math.isnan(gc_dHf) else float("nan")
        dhf_pct_err = (dhf_err / ref_dHf_J_mol * 100) if not math.isnan(dhf_err) and ref_dHf_J_mol != 0 else float("nan")

        thermal_results.append({
            "polymer": polymer_name,
            "ref_Tm_K": ref_Tm_K,
            "gc_Tm_K": gc_Tm_K,
            "Tm_err_K": tm_err,
            "ref_dHf": ref_dHf_J_mol,
            "gc_dHf": gc_dHf,
            "dHf_err_pct": dhf_pct_err,
            "ref_dCp": ref_dCp_J_mol_K,
            "gc_dCp": gc_dCp,
        })

        print(f"  [{polymer_name}]  PSMILES: {psmiles}")
        print(f"    Reference:  Tm={ref_Tm_K:.1f} K,  dHf={ref_dHf_J_mol:.0f} J/mol,  dCp={ref_dCp_J_mol_K:.1f} J/(mol*K)")
        if math.isnan(gc_Tm_K):
            print(f"    GC estimate: FAILED (coverage={gc.get('coverage', 0):.0%})")
        else:
            print(f"    GC estimate: Tm={gc_Tm_K:.1f} K,  dHf={gc_dHf:.0f} J/mol,  dCp={gc_dCp:.1f} J/(mol*K)")
            print(f"    Tm error: {tm_err:.1f} K  |  dHf error: {dhf_pct_err:.1f}%")

        # Get fitted solvents for this polymer
        solvents = get_fitted_solvents_for_polymer(polymer_name, lookup)
        if not solvents:
            print(f"    No fitted solvents found in coefficient database.")
            print()
            continue

        print(f"    Evaluating {len(solvents)} fitted solvent pairs...")

        solvent_errors_gc = []
        solvent_errors_ref = []

        for entry in solvents:
            solvent = entry["solvent"]
            A_fit, B_fit, C_fit = entry["A"], entry["B"], entry["C"]

            # COSMO-RS fitted solubility curve
            s_fitted = solubility_from_abc(A_fit, B_fit, C_fit, temps_c)

            # (a) Ideal SLE with GC-estimated thermal props
            if not math.isnan(gc_Tm_K) and not math.isnan(gc_dHf):
                gc_dCp_safe = gc_dCp if not math.isnan(gc_dCp) else 0.0
                x_gc = compute_ideal_sle(temps_c + 273.15, gc_Tm_K, gc_dHf, gc_dCp_safe)
                s_gc = x_gc * 100.0  # mole fraction -> percent approximation

                # Fit A,B,C to the GC-based curve
                fit_gc = fit_abc(temps_c, s_gc)

                if fit_gc["A"] is not None:
                    s_gc_fitted = solubility_from_abc(fit_gc["A"], fit_gc["B"], fit_gc["C"], temps_c)
                    max_err_gc = max_abs_error_pct(s_gc_fitted, s_fitted)
                    mean_err_gc = mean_abs_error_pct(s_gc_fitted, s_fitted)
                else:
                    max_err_gc = max_abs_error_pct(s_gc, s_fitted)
                    mean_err_gc = mean_abs_error_pct(s_gc, s_fitted)
            else:
                max_err_gc = float("nan")
                mean_err_gc = float("nan")

            # (b) Ideal SLE with reference thermal props
            x_ref = compute_ideal_sle(temps_c + 273.15, ref_Tm_K, ref_dHf_J_mol, ref_dCp_J_mol_K)
            s_ref = x_ref * 100.0

            fit_ref = fit_abc(temps_c, s_ref)
            if fit_ref["A"] is not None:
                s_ref_fitted = solubility_from_abc(fit_ref["A"], fit_ref["B"], fit_ref["C"], temps_c)
                max_err_ref = max_abs_error_pct(s_ref_fitted, s_fitted)
                mean_err_ref = mean_abs_error_pct(s_ref_fitted, s_fitted)
            else:
                max_err_ref = max_abs_error_pct(s_ref, s_fitted)
                mean_err_ref = mean_abs_error_pct(s_ref, s_fitted)

            solvent_errors_gc.append(max_err_gc)
            solvent_errors_ref.append(max_err_ref)

            solubility_results.append({
                "polymer": polymer_name,
                "solvent": solvent,
                "max_err_gc_pct": max_err_gc,
                "mean_err_gc_pct": mean_err_gc,
                "max_err_ref_pct": max_err_ref,
                "mean_err_ref_pct": mean_err_ref,
            })

        valid_gc = [e for e in solvent_errors_gc if not math.isnan(e)]
        valid_ref = [e for e in solvent_errors_ref if not math.isnan(e)]

        if valid_gc:
            print(f"    GC-based ideal SLE vs COSMO-RS fitted:")
            print(f"      Avg max error: {np.mean(valid_gc):.2f}%  |  Max: {np.max(valid_gc):.2f}%")
        if valid_ref:
            print(f"    Ref-based ideal SLE vs COSMO-RS fitted:")
            print(f"      Avg max error: {np.mean(valid_ref):.2f}%  |  Max: {np.max(valid_ref):.2f}%")
        print()

    return thermal_results, solubility_results


# ---------------------------------------------------------------------------
# Part 2: Pipeline End-to-End Test
# ---------------------------------------------------------------------------

def run_part2():
    """Run the full tool pipeline for 3 test polymers."""
    print("=" * 80)
    print("PART 2: Pipeline End-to-End Test (3 New Polymers)")
    print("=" * 80)
    print()

    tool_results = {}

    for polymer_name, psmiles in TEST_POLYMERS.items():
        print(f"  [{polymer_name}] PSMILES: {psmiles}")
        tool_results[polymer_name] = {"thermal": None, "solubility": {}}

        # Step 1: thermal prediction tool
        try:
            thermal_output = tool_predict_thermal(polymer_psmiles=psmiles, polymer_name=polymer_name)
            if thermal_output and isinstance(thermal_output, str) and len(thermal_output) > 20:
                tool_results[polymer_name]["thermal"] = "PASS"
                # Print first 3 lines for brevity
                lines = thermal_output.strip().split("\n")
                for line in lines[:5]:
                    print(f"    {line}")
                if len(lines) > 5:
                    print(f"    ... ({len(lines) - 5} more lines)")
            else:
                tool_results[polymer_name]["thermal"] = "FAIL (empty output)"
                print(f"    Thermal prediction: FAIL (empty or short output)")
        except Exception as e:
            tool_results[polymer_name]["thermal"] = f"FAIL ({e})"
            print(f"    Thermal prediction: FAIL ({e})")

        # Step 2: solubility generation for 2-3 solvents
        solvents = TEST_SOLVENTS.get(polymer_name, ["toluene", "acetone"])
        for solvent in solvents:
            try:
                sol_output = tool_generate(
                    polymer_name=polymer_name,
                    polymer_psmiles=psmiles,
                    solvent_name=solvent,
                    temperature_c=25.0,
                )
                if sol_output and isinstance(sol_output, str) and len(sol_output) > 20:
                    tool_results[polymer_name]["solubility"][solvent] = "PASS"
                    # Print header line
                    first_line = sol_output.strip().split("\n")[0]
                    print(f"    Solubility ({solvent}): PASS  [{first_line}]")
                else:
                    tool_results[polymer_name]["solubility"][solvent] = "FAIL (empty)"
                    print(f"    Solubility ({solvent}): FAIL (empty output)")
            except Exception as e:
                tool_results[polymer_name]["solubility"][solvent] = f"FAIL ({e})"
                print(f"    Solubility ({solvent}): FAIL ({e})")

        print()

    return tool_results


# ---------------------------------------------------------------------------
# Part 3: Summary Report
# ---------------------------------------------------------------------------

def run_part3(thermal_results, solubility_results, tool_results):
    """Print comprehensive summary of all validation results."""
    print("=" * 80)
    print("PART 3: Summary Report")
    print("=" * 80)
    print()

    # --- Thermal property errors ---
    print("-" * 60)
    print("Thermal Property Errors (GC vs Reference)")
    print("-" * 60)
    print(f"{'Polymer':<12} {'Ref Tm(K)':>10} {'GC Tm(K)':>10} {'Tm Err(K)':>10} {'dHf Err%':>10}")
    print("-" * 60)

    valid_tm_errs = []
    valid_dhf_errs = []
    best_polymer = (None, float("inf"))
    worst_polymer = (None, 0.0)

    for tr in thermal_results:
        tm_err = tr["Tm_err_K"]
        dhf_err = tr["dHf_err_pct"]
        ref_tm = tr["ref_Tm_K"]
        gc_tm = tr["gc_Tm_K"]

        tm_str = f"{tm_err:.1f}" if not math.isnan(tm_err) else "N/A"
        dhf_str = f"{dhf_err:.1f}" if not math.isnan(dhf_err) else "N/A"
        gc_str = f"{gc_tm:.1f}" if not math.isnan(gc_tm) else "N/A"

        print(f"{tr['polymer']:<12} {ref_tm:>10.1f} {gc_str:>10} {tm_str:>10} {dhf_str:>10}")

        if not math.isnan(tm_err):
            valid_tm_errs.append(tm_err)
            if tm_err < best_polymer[1]:
                best_polymer = (tr["polymer"], tm_err)
            if tm_err > worst_polymer[1]:
                worst_polymer = (tr["polymer"], tm_err)
        if not math.isnan(dhf_err):
            valid_dhf_errs.append(dhf_err)

    print("-" * 60)
    if valid_tm_errs:
        print(f"Average Tm error:  {np.mean(valid_tm_errs):.1f} K  (n={len(valid_tm_errs)})")
    if valid_dhf_errs:
        print(f"Average dHf error: {np.mean(valid_dhf_errs):.1f}%  (n={len(valid_dhf_errs)})")
    if best_polymer[0]:
        print(f"Best predicted:    {best_polymer[0]} (Tm err = {best_polymer[1]:.1f} K)")
    if worst_polymer[0]:
        print(f"Worst predicted:   {worst_polymer[0]} (Tm err = {worst_polymer[1]:.1f} K)")
    print()

    # --- Solubility errors ---
    print("-" * 60)
    print("Solubility Errors: Ideal SLE vs COSMO-RS Fitted A,B,C")
    print("-" * 60)

    # Aggregate by polymer
    polymer_gc_errs = {}
    polymer_ref_errs = {}
    for sr in solubility_results:
        p = sr["polymer"]
        if not math.isnan(sr["max_err_gc_pct"]):
            polymer_gc_errs.setdefault(p, []).append(sr["max_err_gc_pct"])
        if not math.isnan(sr["max_err_ref_pct"]):
            polymer_ref_errs.setdefault(p, []).append(sr["max_err_ref_pct"])

    print(f"{'Polymer':<12} {'N pairs':>8} {'GC avg err%':>12} {'GC max err%':>12} {'Ref avg err%':>13} {'Ref max err%':>13}")
    print("-" * 75)

    all_gc_errs = []
    all_ref_errs = []

    for polymer_name in POLYMER_PSMILES:
        gc_errs = polymer_gc_errs.get(polymer_name, [])
        ref_errs = polymer_ref_errs.get(polymer_name, [])
        n = max(len(gc_errs), len(ref_errs))

        if n == 0:
            continue

        gc_avg = f"{np.mean(gc_errs):.2f}" if gc_errs else "N/A"
        gc_max = f"{np.max(gc_errs):.2f}" if gc_errs else "N/A"
        ref_avg = f"{np.mean(ref_errs):.2f}" if ref_errs else "N/A"
        ref_max = f"{np.max(ref_errs):.2f}" if ref_errs else "N/A"

        print(f"{polymer_name:<12} {n:>8} {gc_avg:>12} {gc_max:>12} {ref_avg:>13} {ref_max:>13}")

        all_gc_errs.extend(gc_errs)
        all_ref_errs.extend(ref_errs)

    print("-" * 75)
    if all_gc_errs:
        print(f"Overall GC-based:  avg max err = {np.mean(all_gc_errs):.2f}%  |  worst = {np.max(all_gc_errs):.2f}%  (n={len(all_gc_errs)} pairs)")
    if all_ref_errs:
        print(f"Overall Ref-based: avg max err = {np.mean(all_ref_errs):.2f}%  |  worst = {np.max(all_ref_errs):.2f}%  (n={len(all_ref_errs)} pairs)")
    print()

    # --- Tool pipeline results ---
    print("-" * 60)
    print("Tool Pipeline Pass/Fail (3 Test Polymers)")
    print("-" * 60)
    print(f"{'Polymer':<8} {'Thermal':>10} {'Solvents':>40}")
    print("-" * 60)

    for polymer_name in TEST_POLYMERS:
        tr = tool_results.get(polymer_name, {})
        thermal_status = tr.get("thermal", "NOT RUN")
        sol_statuses = tr.get("solubility", {})
        sol_str = ", ".join(f"{s}: {st}" for s, st in sol_statuses.items()) if sol_statuses else "NONE"
        print(f"{polymer_name:<8} {thermal_status:>10} {sol_str:>40}")

    print("-" * 60)

    # Count passes
    total_tests = 0
    total_pass = 0
    for polymer_name in TEST_POLYMERS:
        tr = tool_results.get(polymer_name, {})
        total_tests += 1
        if tr.get("thermal") == "PASS":
            total_pass += 1
        for st in tr.get("solubility", {}).values():
            total_tests += 1
            if st == "PASS":
                total_pass += 1

    print(f"Total: {total_pass}/{total_tests} tests passed")
    print()


# ---------------------------------------------------------------------------
# Part 4: Sensitivity-Calibrated Confidence Tiers
# ---------------------------------------------------------------------------

def run_part4(thermal_results, solubility_results):
    """Compute recommended confidence tier thresholds from validation data."""
    print("=" * 80)
    print("PART 4: Sensitivity-Calibrated Confidence Tiers")
    print("=" * 80)
    print()

    # Build a mapping: polymer -> (Tm_error, avg max solubility error with GC)
    polymer_tm_err = {}
    for tr in thermal_results:
        if not math.isnan(tr["Tm_err_K"]):
            polymer_tm_err[tr["polymer"]] = tr["Tm_err_K"]

    polymer_avg_sol_err = {}
    for sr in solubility_results:
        p = sr["polymer"]
        if not math.isnan(sr["max_err_gc_pct"]):
            polymer_avg_sol_err.setdefault(p, []).append(sr["max_err_gc_pct"])

    # Build paired data
    tm_errs = []
    sol_errs = []
    polymer_labels = []

    for p in polymer_tm_err:
        if p in polymer_avg_sol_err and polymer_avg_sol_err[p]:
            tm_errs.append(polymer_tm_err[p])
            sol_errs.append(np.mean(polymer_avg_sol_err[p]))
            polymer_labels.append(p)

    if len(tm_errs) < 3:
        print("  Insufficient data to fit Tm_error -> solubility_error relationship.")
        print(f"  Only {len(tm_errs)} data points available (need >= 3).")
        print()
        _print_default_tiers()
        return

    tm_errs = np.array(tm_errs)
    sol_errs = np.array(sol_errs)

    print("  Per-polymer: Tm error vs average max solubility error")
    print(f"  {'Polymer':<12} {'Tm err (K)':>12} {'Avg sol err%':>14}")
    print("  " + "-" * 40)
    for label, te, se in zip(polymer_labels, tm_errs, sol_errs):
        print(f"  {label:<12} {te:>12.1f} {se:>14.2f}")
    print()

    # Fit linear relationship: sol_err = slope * tm_err + intercept
    try:
        coeffs = np.polyfit(tm_errs, sol_errs, 1)
        slope, intercept = coeffs
        r_corr = np.corrcoef(tm_errs, sol_errs)[0, 1] if len(tm_errs) > 1 else 0.0

        print(f"  Linear fit: sol_err% = {slope:.3f} * Tm_err(K) + {intercept:.3f}")
        print(f"  Correlation (r): {r_corr:.3f}")
        print()

        # Determine Tm error thresholds for target solubility error tiers
        # high = <5% solubility error, medium = <15%, low = >15%
        high_threshold = 5.0   # % solubility error
        medium_threshold = 15.0

        if slope > 0:
            tm_high = (high_threshold - intercept) / slope
            tm_medium = (medium_threshold - intercept) / slope
        else:
            # Negative or zero slope means Tm error doesn't drive sol error well
            tm_high = 15.0
            tm_medium = 40.0

        # Clamp to reasonable ranges
        tm_high = max(5.0, min(tm_high, 50.0))
        tm_medium = max(tm_high + 5.0, min(tm_medium, 80.0))

        print("  Recommended Confidence Tier Thresholds:")
        print("  " + "=" * 55)
        print(f"  {'Tier':<10} {'Tm Error (K)':<15} {'Expected Sol Error%':<22} {'Action'}")
        print("  " + "-" * 55)
        print(f"  {'HIGH':<10} {'< ' + f'{tm_high:.0f}':>12}   {'< ' + f'{high_threshold:.0f}%':>18}   Use directly")
        print(f"  {'MEDIUM':<10} {'< ' + f'{tm_medium:.0f}':>12}   {'< ' + f'{medium_threshold:.0f}%':>18}   Use with caution")
        print(f"  {'LOW':<10} {'> ' + f'{tm_medium:.0f}':>12}   {'> ' + f'{medium_threshold:.0f}%':>18}   Order-of-magnitude only")
        print("  " + "=" * 55)
        print()

        # Also show dHf error-based tiers if useful
        print("  Additional thresholds (from validation data):")
        print(f"    Tm error for HIGH confidence:   <= {tm_high:.0f} K")
        print(f"    Tm error for MEDIUM confidence: <= {tm_medium:.0f} K")
        print(f"    dHf error for HIGH confidence:  <= 20%")
        print(f"    dHf error for MEDIUM confidence:<= 50%")
        print(f"    Group coverage for HIGH:        >= 80%")
        print(f"    Group coverage for MEDIUM:      >= 50%")

    except Exception as e:
        print(f"  Fitting failed: {e}")
        _print_default_tiers()

    print()


def _print_default_tiers():
    """Print default conservative tiers when data is insufficient for calibration."""
    print()
    print("  Using default conservative thresholds:")
    print("  " + "=" * 55)
    print(f"  {'Tier':<10} {'Tm Error (K)':<15} {'Expected Sol Error%':<22} {'Action'}")
    print("  " + "-" * 55)
    print(f"  {'HIGH':<10} {'< 10':>12}   {'< 5%':>18}   Use directly")
    print(f"  {'MEDIUM':<10} {'< 25':>12}   {'< 15%':>18}   Use with caution")
    print(f"  {'LOW':<10} {'> 25':>12}   {'> 15%':>18}   Order-of-magnitude only")
    print("  " + "=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("*" * 80)
    print("  STRAP v7 Phase 4 Validation: Thermal ML Pipeline")
    print("  Pipeline: PSMILES -> GC/ML -> (Tm, dHf, dCp) -> Ideal SLE -> A,B,C")
    print("*" * 80)
    print()

    # Part 1: Leave-one-out validation
    thermal_results, solubility_results = run_part1()

    # Part 2: End-to-end tool test
    tool_results = run_part2()

    # Part 3: Summary
    run_part3(thermal_results, solubility_results, tool_results)

    # Part 4: Confidence tiers
    run_part4(thermal_results, solubility_results)

    print("*" * 80)
    print("  Validation complete.")
    print("*" * 80)
    print()


if __name__ == "__main__":
    main()
