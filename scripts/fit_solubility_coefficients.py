#!/usr/bin/env python3
"""Fit ln(S) = A + B/T_K + C/T_K^2 for every (polymer, solvent) pair.

Reads data/COMMON-SOLVENTS-DATABASE.csv and writes
data/solubility_coefficients.json with 352 coefficient entries.

Note: COSMO-RS returns NaN for fully miscible conditions. Those were
stored as S=100% in the CSV. We drop exact 100.0 points before fitting
because the ln(S) model cannot represent a flat ceiling — keeping them
degrades R² significantly (5 pairs rescued from anomalous → fitted,
all 26 affected pairs improve, zero regressions).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"
OUT_PATH = DATA_DIR / "solubility_coefficients.json"

# Clamp bounds for solubility before taking ln
S_MIN = 1e-12
S_MAX = 100.0

# Minimum R^2 for a "fitted" pair
R2_THRESHOLD = 0.98


def _model(t_k: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """ln(S) = A + B/T_K + C/T_K^2"""
    return a + b / t_k + c / t_k**2


def _r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    # Normalize column names (handle °C encoding variants)
    df.columns = [c.strip() for c in df.columns]
    # Rename to standard names
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

    entries = []
    categories = {"fitted": 0, "insoluble": 0, "saturated": 0, "anomalous": 0}

    total_dropped = 0

    for (polymer, solvent), grp in df.groupby(["polymer", "solvent"]):
        grp = grp.sort_values("temperature_c")
        n_total = len(grp)
        temps_c_all = grp["temperature_c"].values
        sols_all = grp["solubility_pct"].values

        # Edge case: all zero / near-zero
        if np.all(sols_all < 1e-10):
            entries.append({
                "polymer": polymer,
                "solvent": solvent,
                "category": "insoluble",
                "A": None, "B": None, "C": None,
                "r_squared": None,
                "n_points": int(n_total),
                "n_dropped_100": 0,
                "t_min_c": float(temps_c_all.min()),
                "t_max_c": float(temps_c_all.max()),
            })
            categories["insoluble"] += 1
            continue

        # Edge case: all saturated (>=99.999)
        if np.all(sols_all >= 99.999):
            entries.append({
                "polymer": polymer,
                "solvent": solvent,
                "category": "saturated",
                "A": None, "B": None, "C": None,
                "r_squared": None,
                "n_points": int(n_total),
                "n_dropped_100": int(np.sum(sols_all == 100.0)),
                "t_min_c": float(temps_c_all.min()),
                "t_max_c": float(temps_c_all.max()),
            })
            categories["saturated"] += 1
            continue

        # Drop exact 100.0 points (COSMO-RS NaN artifacts)
        mask = sols_all != 100.0
        n_dropped = int(np.sum(~mask))
        total_dropped += n_dropped
        temps_c = temps_c_all[mask]
        sols = sols_all[mask]

        if len(sols) < 3:
            # Too few points left after dropping 100s
            entries.append({
                "polymer": polymer,
                "solvent": solvent,
                "category": "saturated",
                "A": None, "B": None, "C": None,
                "r_squared": None,
                "n_points": int(n_total),
                "n_dropped_100": n_dropped,
                "t_min_c": float(temps_c_all.min()),
                "t_max_c": float(temps_c_all.max()),
            })
            categories["saturated"] += 1
            continue

        # Clamp and fit
        s_clamped = np.clip(sols, S_MIN, S_MAX)
        ln_s = np.log(s_clamped)
        t_k = temps_c + 273.15

        try:
            popt, _ = curve_fit(_model, t_k, ln_s, p0=[0, 0, 0], maxfev=10000)
            ln_s_pred = _model(t_k, *popt)
            r2 = _r_squared(ln_s, ln_s_pred)
        except Exception:
            r2 = -1.0
            popt = [0, 0, 0]

        if r2 < R2_THRESHOLD:
            cat = "anomalous"
            categories["anomalous"] += 1
        else:
            cat = "fitted"
            categories["fitted"] += 1

        entries.append({
            "polymer": polymer,
            "solvent": solvent,
            "category": cat,
            "A": float(popt[0]),
            "B": float(popt[1]),
            "C": float(popt[2]),
            "r_squared": round(float(r2), 6),
            "n_points": int(len(sols)),
            "n_dropped_100": n_dropped,
            "t_min_c": float(temps_c.min()),
            "t_max_c": float(temps_c.max()),
        })

    output = {
        "description": "Solubility interpolation coefficients: ln(S%) = A + B/T_K + C/T_K^2",
        "note": "Exact S=100% points (COSMO-RS NaN artifacts) excluded before fitting.",
        "source": "COMMON-SOLVENTS-DATABASE.csv",
        "model_path": "data/solubility_coefficients.json",
        "inference_module": "src/strap/tools/interpolation.py",
        "n_entries": len(entries),
        "n_points_dropped_100": total_dropped,
        "categories": categories,
        "entries": entries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(entries)} entries to {OUT_PATH}")
    print(f"Dropped {total_dropped} exact-100% data points across all pairs")
    print(f"Categories: {categories}")
    print(f"  fitted:    {categories['fitted']}")
    print(f"  insoluble: {categories['insoluble']}")
    print(f"  saturated: {categories['saturated']}")
    print(f"  anomalous: {categories['anomalous']}")

    # Quick sanity check
    fitted = [e for e in entries if e["category"] == "fitted"]
    if fitted:
        r2s = [e["r_squared"] for e in fitted]
        print(f"\nFitted R² stats: min={min(r2s):.6f}  mean={np.mean(r2s):.6f}  max={max(r2s):.6f}")

    # Report pairs that had 100% points dropped
    dropped_pairs = [e for e in entries if e.get("n_dropped_100", 0) > 0]
    if dropped_pairs:
        print(f"\nPairs with dropped 100% points ({len(dropped_pairs)}):")
        for e in sorted(dropped_pairs, key=lambda x: -x["n_dropped_100"]):
            print(f"  {e['polymer']}/{e['solvent']}: dropped {e['n_dropped_100']}, "
                  f"category={e['category']}, "
                  f"R²={e['r_squared'] if e['r_squared'] else 'N/A'}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
