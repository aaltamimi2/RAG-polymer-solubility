#!/usr/bin/env python3
"""Phase 0 Feasibility Check: Validate the COSMO-RS SLE → curve fitting link.

For each polymer with known thermal data (Tm, ΔHf, ΔCp), compute the
"ideal SLE" solubility contribution from thermal properties alone:

    ln(x_ideal) = -(ΔHf / R) * (1/T - 1/Tm) + (ΔCp / R) * (Tm/T - 1 - ln(Tm/T))

Compare these ideal solubility curves against the existing COSMO-RS fitted
coefficients (ln(S) = A + B/T + C·ln(T), modified Apelblat) to quantify:

1. How much of the temperature dependence comes from thermal properties
   vs. activity coefficient (γ) contributions from COSMO-RS
2. Whether the ln(S) = A + B/T + C·ln(T) model can capture SLE-derived curves
3. Sensitivity of fitted A,B,C to Tm perturbations (±30K, ±60K)

This validates the pipeline link BEFORE investing in ML model training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COEFF_PATH = DATA_DIR / "solubility_coefficients.json"
THERMAL_PATH = DATA_DIR / "thermal_properties" / "polymer_thermal_reference.csv"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"

R = 8.314  # J/(mol·K)


def sle_ideal_solubility(
    T_K: np.ndarray,
    Tm_K: float,
    delta_Hf_J_per_mol: float,
    delta_Cp_J_per_mol_K: float | None = None,
) -> np.ndarray:
    """Compute ideal mole-fraction solubility from SLE equation.

    ln(x) = -(ΔHf/R)(1/T - 1/Tm) + (ΔCp/R)(Tm/T - 1 - ln(Tm/T))

    Returns solubility as mole fraction (0-1).
    """
    ln_x = -(delta_Hf_J_per_mol / R) * (1.0 / T_K - 1.0 / Tm_K)

    if delta_Cp_J_per_mol_K is not None and delta_Cp_J_per_mol_K > 0:
        ln_x += (delta_Cp_J_per_mol_K / R) * (
            Tm_K / T_K - 1.0 - np.log(Tm_K / T_K)
        )

    return np.clip(np.exp(ln_x), 0.0, 1.0)


def interp_model(t_k: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """ln(S%) = A + B/T_K + C*ln(T_K)  (modified Apelblat)"""
    return a + b / t_k + c * np.log(t_k)


def r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_abc(temps_c: np.ndarray, sols_pct: np.ndarray) -> dict:
    """Fit ln(S%) = A + B/T + C/T² and return coefficients + R²."""
    s_clamped = np.clip(sols_pct, 1e-12, 100.0)
    ln_s = np.log(s_clamped)
    t_k = temps_c + 273.15

    try:
        popt, _ = curve_fit(interp_model, t_k, ln_s, p0=[0, 0, 0], maxfev=10000)
        ln_s_pred = interp_model(t_k, *popt)
        r2 = r_squared(ln_s, ln_s_pred)
        return {"A": popt[0], "B": popt[1], "C": popt[2], "r_squared": r2}
    except Exception as e:
        return {"A": None, "B": None, "C": None, "r_squared": -1.0, "error": str(e)}


def main() -> None:
    # Load thermal reference data
    thermal_df = pd.read_csv(THERMAL_PATH)
    thermal_df = thermal_df[thermal_df["crystalline"].isin(["yes", "semicryst"])]

    # Load existing fitted coefficients
    with open(COEFF_PATH) as f:
        coefficients = json.load(f)
    coeff_lookup = {}
    for entry in coefficients["entries"]:
        coeff_lookup[(entry["polymer"], entry["solvent"])] = entry

    # Load raw COSMO-RS data
    raw_df = pd.read_csv(CSV_PATH, encoding="utf-8")
    raw_df.columns = [c.strip() for c in raw_df.columns]
    col_map = {}
    for c in raw_df.columns:
        cl = c.lower()
        if "solvent" in cl:
            col_map[c] = "solvent"
        elif "temperature" in cl:
            col_map[c] = "temperature_c"
        elif "solubility" in cl:
            col_map[c] = "solubility_pct"
        elif "polymer" in cl:
            col_map[c] = "polymer"
    raw_df.rename(columns=col_map, inplace=True)
    raw_df["temperature_c"] = pd.to_numeric(raw_df["temperature_c"], errors="coerce")
    raw_df["solubility_pct"] = pd.to_numeric(raw_df["solubility_pct"], errors="coerce")

    print("=" * 80)
    print("PHASE 0: FEASIBILITY CHECK — SLE Thermal Data → Curve Fitting Validation")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Part 1: Ideal SLE curves vs COSMO-RS fitted curves
    # ---------------------------------------------------------------
    print("\n--- Part 1: Ideal SLE vs COSMO-RS Fitted Curves ---\n")

    temps_c = np.arange(25, 165, 5, dtype=float)
    temps_k = temps_c + 273.15

    results = []

    for _, row in thermal_df.iterrows():
        polymer = row["polymer_canonical"]
        Tm_K = row["Tm0_K"]
        delta_Hf_g = row["delta_Hf0_J_per_g"]
        MW = row["repeat_unit_MW"]

        if pd.isna(Tm_K) or pd.isna(delta_Hf_g) or pd.isna(MW):
            print(f"  SKIP {polymer}: missing Tm or ΔHf data")
            continue

        delta_Hf_mol = delta_Hf_g * MW  # J/mol repeat unit
        delta_Cp_g = row.get("delta_Cp_J_per_g_K", np.nan)
        delta_Cp_mol = delta_Cp_g * MW if not pd.isna(delta_Cp_g) else None

        # Get a representative solvent for this polymer (pick one with best R²)
        polymer_pairs = [
            (k, v) for k, v in coeff_lookup.items()
            if k[0] == polymer and v["category"] == "fitted"
        ]
        if not polymer_pairs:
            print(f"  SKIP {polymer}: no fitted pairs in coefficient database")
            continue

        # Sort by R² descending, take top 3 solvents
        polymer_pairs.sort(key=lambda x: x[1].get("r_squared", 0), reverse=True)
        test_pairs = polymer_pairs[:3]

        for (poly, solvent), existing in test_pairs:
            # Compute ideal SLE solubility (mole fraction → %)
            x_ideal = sle_ideal_solubility(temps_k, Tm_K, delta_Hf_mol, delta_Cp_mol)
            s_ideal_pct = x_ideal * 100.0

            # Compute COSMO-RS fitted solubility
            if existing["A"] is not None:
                ln_s_cosmo = interp_model(
                    temps_k, existing["A"], existing["B"], existing["C"]
                )
                s_cosmo_pct = np.clip(np.exp(ln_s_cosmo), 0.0, 100.0)
            else:
                continue

            # Fit A,B,C to the ideal SLE curve
            # (only where solubility > 0.001% to avoid log(0))
            mask = s_ideal_pct > 0.001
            if mask.sum() < 3:
                print(f"  {polymer}/{solvent}: ideal SLE too low to fit")
                continue

            ideal_fit = fit_abc(temps_c[mask], s_ideal_pct[mask])

            # Compare
            mean_cosmo = np.mean(s_cosmo_pct)
            mean_ideal = np.mean(s_ideal_pct)

            results.append({
                "polymer": polymer,
                "solvent": solvent,
                "Tm_K": Tm_K,
                "delta_Hf_J_per_mol": delta_Hf_mol,
                "cosmo_A": existing["A"],
                "cosmo_B": existing["B"],
                "cosmo_C": existing["C"],
                "cosmo_R2": existing["r_squared"],
                "ideal_A": ideal_fit.get("A"),
                "ideal_B": ideal_fit.get("B"),
                "ideal_C": ideal_fit.get("C"),
                "ideal_fit_R2": ideal_fit.get("r_squared"),
                "mean_cosmo_sol_pct": mean_cosmo,
                "mean_ideal_sol_pct": mean_ideal,
                "ratio_cosmo_ideal": mean_cosmo / mean_ideal if mean_ideal > 0 else float("inf"),
            })

            print(
                f"  {polymer:8s}/{solvent:20s} | "
                f"COSMO mean={mean_cosmo:8.3f}% | "
                f"Ideal SLE mean={mean_ideal:8.3f}% | "
                f"Ratio={mean_cosmo / mean_ideal if mean_ideal > 0 else float('inf'):8.2f} | "
                f"Ideal fit R²={ideal_fit.get('r_squared', -1):.6f}"
            )

    # ---------------------------------------------------------------
    # Part 2: Sensitivity analysis — perturb Tm by ±30K, ±60K
    # ---------------------------------------------------------------
    print("\n\n--- Part 2: Sensitivity to Tm Perturbation ---\n")
    print(f"{'Polymer':10s} {'Solvent':20s} | {'Tm':>6s} | "
          f"{'A(base)':>10s} {'A(+30K)':>10s} {'A(-30K)':>10s} {'A(+60K)':>10s} {'A(-60K)':>10s} | "
          f"{'MaxΔS%(±30K)':>12s} {'MaxΔS%(±60K)':>12s}")
    print("-" * 140)

    perturbations = [0, +30, -30, +60, -60]

    for _, row in thermal_df.iterrows():
        polymer = row["polymer_canonical"]
        Tm_K = row["Tm0_K"]
        delta_Hf_g = row["delta_Hf0_J_per_g"]
        MW = row["repeat_unit_MW"]

        if pd.isna(Tm_K) or pd.isna(delta_Hf_g) or pd.isna(MW):
            continue

        delta_Hf_mol = delta_Hf_g * MW
        delta_Cp_g = row.get("delta_Cp_J_per_g_K", np.nan)
        delta_Cp_mol = delta_Cp_g * MW if not pd.isna(delta_Cp_g) else None

        polymer_pairs = [
            (k, v) for k, v in coeff_lookup.items()
            if k[0] == polymer and v["category"] == "fitted"
        ]
        if not polymer_pairs:
            continue

        # Pick the best-fit solvent
        polymer_pairs.sort(key=lambda x: x[1].get("r_squared", 0), reverse=True)
        (poly, solvent), existing = polymer_pairs[0]

        curves = {}
        fits = {}
        for dTm in perturbations:
            Tm_perturbed = Tm_K + dTm
            x_ideal = sle_ideal_solubility(temps_k, Tm_perturbed, delta_Hf_mol, delta_Cp_mol)
            s_pct = x_ideal * 100.0
            curves[dTm] = s_pct

            mask = s_pct > 0.001
            if mask.sum() >= 3:
                fits[dTm] = fit_abc(temps_c[mask], s_pct[mask])
            else:
                fits[dTm] = {"A": None, "B": None, "C": None}

        # Max solubility difference from perturbation
        base = curves[0]
        max_ds_30 = max(np.max(np.abs(curves[+30] - base)), np.max(np.abs(curves[-30] - base)))
        max_ds_60 = max(np.max(np.abs(curves[+60] - base)), np.max(np.abs(curves[-60] - base)))

        def fmt_a(d):
            a = fits[d].get("A")
            return f"{a:10.4f}" if a is not None else f"{'N/A':>10s}"

        print(
            f"{polymer:10s} {solvent:20s} | {Tm_K:6.1f} | "
            f"{fmt_a(0)} {fmt_a(+30)} {fmt_a(-30)} {fmt_a(+60)} {fmt_a(-60)} | "
            f"{max_ds_30:12.4f} {max_ds_60:12.4f}"
        )

    # ---------------------------------------------------------------
    # Part 3: Summary and recommendations
    # ---------------------------------------------------------------
    print("\n\n--- Part 3: Summary ---\n")

    if results:
        ratios = [r["ratio_cosmo_ideal"] for r in results if r["ratio_cosmo_ideal"] != float("inf")]
        ideal_r2s = [r["ideal_fit_R2"] for r in results if r["ideal_fit_R2"] is not None and r["ideal_fit_R2"] > 0]

        print(f"Polymer-solvent pairs analyzed: {len(results)}")
        print(f"COSMO/Ideal solubility ratio: min={min(ratios):.2f}  mean={np.mean(ratios):.2f}  max={max(ratios):.2f}")
        print(f"  → ratio >> 1 means COSMO-RS γ contribution dominates (solvent affinity matters)")
        print(f"  → ratio ≈ 1 means ideal SLE is a good approximation")
        print(f"  → ratio << 1 means COSMO-RS predicts lower solubility than ideal (unfavorable γ)")
        print(f"\nIdeal SLE curve fit to ln(S)=A+B/T+C·ln(T):")
        print(f"  R² stats: min={min(ideal_r2s):.6f}  mean={np.mean(ideal_r2s):.6f}  max={max(ideal_r2s):.6f}")
        print(f"  → High R² confirms the A+B/T+C·ln(T) functional form can capture SLE-derived curves")

    print("\nKey questions answered:")
    print("  1. Can ln(S)=A+B/T+C·ln(T) fit SLE-derived curves? → Check ideal fit R² above")
    print("  2. How much does γ matter? → Check COSMO/Ideal ratio above")
    print("  3. How sensitive are curves to Tm error? → Check Part 2 MaxΔS% above")
    print("\nIf ideal fit R² > 0.99 and MaxΔS%(±30K) < 5%, proceed to Phase 1.")
    print("If MaxΔS%(±60K) > 20%, Tm prediction accuracy is critical — tighten ML requirements.")


if __name__ == "__main__":
    main()
