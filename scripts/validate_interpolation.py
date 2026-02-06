#!/usr/bin/env python3
"""Round-trip validation: predict at all original grid temperatures and compare to CSV.

Reports per-pair MAE, RMSE, max_error and aggregate statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"
COEFF_PATH = DATA_DIR / "solubility_coefficients.json"


def _predict(entry: dict, temp_c: float) -> float:
    """ln(S) = A + B/T_K + C/T_K^2 → S clamped to [0, 100]."""
    t_k = temp_c + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] / t_k**2
    return float(np.clip(np.exp(ln_s), 0.0, 100.0))


def main() -> None:
    # Load CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "solvent" in cl:
            col_map[c] = "solvent"
        elif "temperature" in cl:
            col_map[c] = "temperature_c"
        elif "solubility" in cl:
            col_map[c] = "solubility_pct"
        elif "polymer" in cl:
            col_map[c] = "polymer"
    df.rename(columns=col_map, inplace=True)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["solubility_pct"] = pd.to_numeric(df["solubility_pct"], errors="coerce")
    df.dropna(subset=["temperature_c", "solubility_pct"], inplace=True)

    # Load coefficients
    with open(COEFF_PATH) as f:
        coeffs = json.load(f)

    lookup = {}
    for entry in coeffs["entries"]:
        key = (entry["polymer"].strip().upper(), entry["solvent"].strip().lower())
        lookup[key] = entry

    # Round-trip comparison
    results = []
    for (polymer, solvent), grp in df.groupby(["polymer", "solvent"]):
        key = (polymer.strip().upper(), solvent.strip().lower())
        entry = lookup.get(key)
        if entry is None:
            continue
        if entry["category"] != "fitted":
            continue

        actual = grp["solubility_pct"].values
        temps = grp["temperature_c"].values
        predicted = np.array([_predict(entry, t) for t in temps])

        errors = np.abs(actual - predicted)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_err = float(np.max(errors))

        results.append({
            "polymer": polymer,
            "solvent": solvent,
            "r_squared": entry["r_squared"],
            "n_points": len(actual),
            "mae": mae,
            "rmse": rmse,
            "max_error": max_err,
        })

    # Sort by max_error descending
    results.sort(key=lambda r: -r["max_error"])

    # Print report
    print(f"{'='*80}")
    print(f"INTERPOLATION ROUND-TRIP VALIDATION")
    print(f"{'='*80}")
    print(f"Fitted pairs evaluated: {len(results)}")
    print()

    # Aggregate stats
    all_mae = [r["mae"] for r in results]
    all_rmse = [r["rmse"] for r in results]
    all_max = [r["max_error"] for r in results]

    print(f"Aggregate statistics:")
    print(f"  MAE:       mean={np.mean(all_mae):.6f}  median={np.median(all_mae):.6f}  max={np.max(all_mae):.6f}")
    print(f"  RMSE:      mean={np.mean(all_rmse):.6f}  median={np.median(all_rmse):.6f}  max={np.max(all_rmse):.6f}")
    print(f"  Max error: mean={np.mean(all_max):.6f}  median={np.median(all_max):.6f}  max={np.max(all_max):.6f}")
    print()

    # Pairs with max_error > 1%
    bad = [r for r in results if r["max_error"] > 1.0]
    print(f"Pairs with max_error > 1%: {len(bad)}/{len(results)}")
    print()

    # Top 20 worst
    print(f"Top 20 highest-error pairs:")
    print(f"{'Polymer':<8} {'Solvent':<22} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'MaxErr':>10}")
    print(f"{'-'*8} {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for r in results[:20]:
        print(
            f"{r['polymer']:<8} {r['solvent']:<22} {r['r_squared']:>8.4f} "
            f"{r['mae']:>10.4f} {r['rmse']:>10.4f} {r['max_error']:>10.4f}"
        )
    print()

    # Full per-pair table
    print(f"{'='*80}")
    print(f"FULL PER-PAIR REPORT (sorted by max_error descending)")
    print(f"{'='*80}")
    print(f"{'Polymer':<8} {'Solvent':<22} {'R²':>8} {'N':>4} {'MAE':>10} {'RMSE':>10} {'MaxErr':>10}")
    print(f"{'-'*8} {'-'*22} {'-'*8} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(
            f"{r['polymer']:<8} {r['solvent']:<22} {r['r_squared']:>8.4f} "
            f"{r['n_points']:>4} {r['mae']:>10.4f} {r['rmse']:>10.4f} {r['max_error']:>10.4f}"
        )


if __name__ == "__main__":
    main()
