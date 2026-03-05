#!/usr/bin/env python3
"""Compare ln(S) = A + B/T + C/T^2 (Maier-Kelley) vs A + B/T + C*ln(T) (Apelblat).

Reads data/COMMON-SOLVENTS-DATABASE.csv directly — no STRAP agent imports.
Applies identical preprocessing to scripts/fit_solubility_coefficients.py.

Outputs:
  experiments/apelblat_comparison/comparison_results.json
  experiments/apelblat_comparison/comparison_results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"
COEFF_PATH = DATA_DIR / "solubility_coefficients.json"
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "comparison_results.json"
OUT_CSV = OUT_DIR / "comparison_results.csv"

S_MIN = 1e-12
S_MAX = 100.0
R2_THRESHOLD = 0.98


# ── Model functions ──────────────────────────────────────────────
def model_mk(t_k: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Maier-Kelley: ln(S%) = A + B/T + C/T^2"""
    return a + b / t_k + c / t_k**2


def model_apelblat(t_k: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Modified Apelblat: ln(S%) = A + B/T + C*ln(T)"""
    return a + b / t_k + c * np.log(t_k)


# ── Metrics ──────────────────────────────────────────────────────
def r_squared(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_metrics(y_actual: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_pred - y_actual
    r2 = r_squared(y_actual, y_pred)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_err = float(np.max(np.abs(residuals)))
    return {"r2": round(r2, 8), "rmse": rmse, "mae": mae, "max_err": max_err}


def fit_model(model_fn, t_k, ln_s):
    """Fit a model and return (coeffs, metrics_lns, metrics_pct) or None on failure."""
    try:
        popt, _ = curve_fit(model_fn, t_k, ln_s, p0=[0.0, 0.0, 0.0], maxfev=10000)
    except Exception:
        return None

    ln_s_pred = model_fn(t_k, *popt)
    m_lns = compute_metrics(ln_s, ln_s_pred)

    # Back-transform to S% space
    s_actual = np.clip(np.exp(ln_s), 0.0, 100.0)
    s_pred = np.clip(np.exp(ln_s_pred), 0.0, 100.0)
    m_pct = compute_metrics(s_actual, s_pred)

    return {
        "A": float(popt[0]),
        "B": float(popt[1]),
        "C": float(popt[2]),
        "lns": m_lns,
        "pct": m_pct,
    }


# ── Main ─────────────────────────────────────────────────────────
def main() -> int:
    # Load raw data
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

    # Load existing MK coefficients for cross-validation
    with open(COEFF_PATH) as f:
        existing = json.load(f)
    existing_lookup = {
        (e["polymer"].strip(), e["solvent"].strip()): e
        for e in existing["entries"]
    }

    rows = []
    skipped = {"insoluble": 0, "saturated": 0}

    for (polymer, solvent), grp in df.groupby(["polymer", "solvent"]):
        grp = grp.sort_values("temperature_c")
        sols_all = grp["solubility_pct"].values
        temps_c_all = grp["temperature_c"].values

        # Skip insoluble
        if np.all(sols_all < 1e-10):
            skipped["insoluble"] += 1
            continue

        # Skip all-saturated
        if np.all(sols_all >= 99.999):
            skipped["saturated"] += 1
            continue

        # Drop exact 100.0 (COSMO-RS NaN artifacts) — same as original
        mask = sols_all != 100.0
        n_dropped = int(np.sum(~mask))
        temps_c = temps_c_all[mask]
        sols = sols_all[mask]

        if len(sols) < 3:
            skipped["saturated"] += 1
            continue

        # Clamp and transform
        s_clamped = np.clip(sols, S_MIN, S_MAX)
        ln_s = np.log(s_clamped)
        t_k = temps_c + 273.15

        # Fit both models
        mk_result = fit_model(model_mk, t_k, ln_s)
        apel_result = fit_model(model_apelblat, t_k, ln_s)

        row = {
            "polymer": polymer,
            "solvent": solvent,
            "n_points": int(len(sols)),
            "n_dropped_100": n_dropped,
            "t_min_c": float(temps_c.min()),
            "t_max_c": float(temps_c.max()),
        }

        # MK results
        if mk_result:
            row["mk_A"] = mk_result["A"]
            row["mk_B"] = mk_result["B"]
            row["mk_C"] = mk_result["C"]
            row["mk_r2"] = mk_result["lns"]["r2"]
            row["mk_rmse_lns"] = mk_result["lns"]["rmse"]
            row["mk_mae_pct"] = mk_result["pct"]["mae"]
            row["mk_rmse_pct"] = mk_result["pct"]["rmse"]
            row["mk_max_err_pct"] = mk_result["pct"]["max_err"]
            row["mk_category"] = "fitted" if mk_result["lns"]["r2"] >= R2_THRESHOLD else "anomalous"
        else:
            for k in ["mk_A", "mk_B", "mk_C", "mk_r2", "mk_rmse_lns",
                       "mk_mae_pct", "mk_rmse_pct", "mk_max_err_pct"]:
                row[k] = None
            row["mk_category"] = "failed"

        # Apelblat results
        if apel_result:
            row["apel_A"] = apel_result["A"]
            row["apel_B"] = apel_result["B"]
            row["apel_C"] = apel_result["C"]
            row["apel_r2"] = apel_result["lns"]["r2"]
            row["apel_rmse_lns"] = apel_result["lns"]["rmse"]
            row["apel_mae_pct"] = apel_result["pct"]["mae"]
            row["apel_rmse_pct"] = apel_result["pct"]["rmse"]
            row["apel_max_err_pct"] = apel_result["pct"]["max_err"]
            row["apel_category"] = "fitted" if apel_result["lns"]["r2"] >= R2_THRESHOLD else "anomalous"
        else:
            for k in ["apel_A", "apel_B", "apel_C", "apel_r2", "apel_rmse_lns",
                       "apel_mae_pct", "apel_rmse_pct", "apel_max_err_pct"]:
                row[k] = None
            row["apel_category"] = "failed"

        # Delta and winner
        if mk_result and apel_result:
            delta = apel_result["lns"]["r2"] - mk_result["lns"]["r2"]
            row["delta_r2"] = round(delta, 8)
            if delta > 1e-5:
                row["winner"] = "apelblat"
            elif delta < -1e-5:
                row["winner"] = "mk"
            else:
                row["winner"] = "tie"
        else:
            row["delta_r2"] = None
            row["winner"] = None

        rows.append(row)

    # ── Write outputs ────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)

    json_output = {
        "description": "Comparison: Maier-Kelley (A+B/T+C/T^2) vs Apelblat (A+B/T+C*ln(T))",
        "source": str(CSV_PATH),
        "n_compared": len(rows),
        "n_skipped": skipped,
        "entries": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    # ── Console summary ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MAIER-KELLEY  vs  APELBLAT  COMPARISON")
    print(f"{'='*70}")
    print(f"\nPairs compared:  {len(rows)}")
    print(f"Skipped:         {skipped}")

    # Category comparison
    mk_fitted = sum(1 for r in rows if r["mk_category"] == "fitted")
    mk_anom = sum(1 for r in rows if r["mk_category"] == "anomalous")
    apel_fitted = sum(1 for r in rows if r["apel_category"] == "fitted")
    apel_anom = sum(1 for r in rows if r["apel_category"] == "anomalous")

    print(f"\n  Category Breakdown:")
    print(f"  {'':30s} {'MK':>8s}  {'Apelblat':>8s}")
    print(f"  {'Fitted (R² >= 0.98)':30s} {mk_fitted:>8d}  {apel_fitted:>8d}")
    print(f"  {'Anomalous (R² < 0.98)':30s} {mk_anom:>8d}  {apel_anom:>8d}")

    # Winner tally
    both_valid = [r for r in rows if r["delta_r2"] is not None]
    apel_wins = sum(1 for r in both_valid if r["winner"] == "apelblat")
    mk_wins = sum(1 for r in both_valid if r["winner"] == "mk")
    ties = sum(1 for r in both_valid if r["winner"] == "tie")

    print(f"\n  Head-to-Head (among {len(both_valid)} jointly fitted):")
    print(f"    Apelblat wins:  {apel_wins:>4d}  ({100*apel_wins/len(both_valid):.1f}%)")
    print(f"    MK wins:        {mk_wins:>4d}  ({100*mk_wins/len(both_valid):.1f}%)")
    print(f"    Ties:           {ties:>4d}  ({100*ties/len(both_valid):.1f}%)")

    # R² statistics
    deltas = [r["delta_r2"] for r in both_valid]
    mk_r2s = [r["mk_r2"] for r in both_valid]
    apel_r2s = [r["apel_r2"] for r in both_valid]

    print(f"\n  R² Statistics (ln(S) space):")
    print(f"    {'':25s} {'MK':>12s}  {'Apelblat':>12s}")
    print(f"    {'Mean R²':25s} {np.mean(mk_r2s):>12.8f}  {np.mean(apel_r2s):>12.8f}")
    print(f"    {'Median R²':25s} {np.median(mk_r2s):>12.8f}  {np.median(apel_r2s):>12.8f}")
    print(f"    {'Min R²':25s} {np.min(mk_r2s):>12.8f}  {np.min(apel_r2s):>12.8f}")
    print(f"    {'Max R²':25s} {np.max(mk_r2s):>12.8f}  {np.max(apel_r2s):>12.8f}")

    print(f"\n  Delta R² (Apelblat - MK):")
    print(f"    Mean:    {np.mean(deltas):+.8f}")
    print(f"    Median:  {np.median(deltas):+.8f}")
    print(f"    Min:     {np.min(deltas):+.8f}")
    print(f"    Max:     {np.max(deltas):+.8f}")

    # S% space metrics
    mk_maes = [r["mk_mae_pct"] for r in both_valid if r["mk_mae_pct"] is not None]
    apel_maes = [r["apel_mae_pct"] for r in both_valid if r["apel_mae_pct"] is not None]
    mk_rmses = [r["mk_rmse_pct"] for r in both_valid if r["mk_rmse_pct"] is not None]
    apel_rmses = [r["apel_rmse_pct"] for r in both_valid if r["apel_rmse_pct"] is not None]

    print(f"\n  S% Space Error (back-transformed):")
    print(f"    {'':25s} {'MK':>12s}  {'Apelblat':>12s}")
    print(f"    {'Mean MAE (%)':25s} {np.mean(mk_maes):>12.6f}  {np.mean(apel_maes):>12.6f}")
    print(f"    {'Mean RMSE (%)':25s} {np.mean(mk_rmses):>12.6f}  {np.mean(apel_rmses):>12.6f}")

    # Rescue report: MK anomalous but Apelblat fitted
    rescued = [r for r in rows
               if r["mk_category"] == "anomalous" and r["apel_category"] == "fitted"]
    lost = [r for r in rows
            if r["mk_category"] == "fitted" and r["apel_category"] == "anomalous"]

    print(f"\n  Rescue Report:")
    print(f"    MK anomalous -> Apelblat fitted:  {len(rescued)}")
    for r in rescued:
        print(f"      {r['polymer']}/{r['solvent']}: "
              f"MK R²={r['mk_r2']:.6f} -> Apel R²={r['apel_r2']:.6f}")
    print(f"    MK fitted -> Apelblat anomalous:  {len(lost)}")
    for r in lost:
        print(f"      {r['polymer']}/{r['solvent']}: "
              f"MK R²={r['mk_r2']:.6f} -> Apel R²={r['apel_r2']:.6f}")

    # Top 10 biggest improvements
    sorted_by_delta = sorted(both_valid, key=lambda r: r["delta_r2"], reverse=True)
    print(f"\n  Top 10 Apelblat Improvements:")
    for r in sorted_by_delta[:10]:
        print(f"    {r['polymer']:>8s}/{r['solvent']:<25s} "
              f"delta_R²={r['delta_r2']:+.8f}  "
              f"(MK={r['mk_r2']:.6f} -> Apel={r['apel_r2']:.6f})")

    print(f"\n  Top 10 MK Advantages:")
    for r in sorted_by_delta[-10:]:
        print(f"    {r['polymer']:>8s}/{r['solvent']:<25s} "
              f"delta_R²={r['delta_r2']:+.8f}  "
              f"(MK={r['mk_r2']:.6f} -> Apel={r['apel_r2']:.6f})")

    # Per-polymer summary
    print(f"\n  Per-Polymer Mean Delta R²:")
    for polymer in sorted(set(r["polymer"] for r in rows)):
        p_rows = [r for r in both_valid if r["polymer"] == polymer]
        if p_rows:
            p_deltas = [r["delta_r2"] for r in p_rows]
            p_wins = sum(1 for d in p_deltas if d > 1e-5)
            p_losses = sum(1 for d in p_deltas if d < -1e-5)
            print(f"    {polymer:>8s}: mean={np.mean(p_deltas):+.8f}  "
                  f"wins={p_wins}/{len(p_rows)}  losses={p_losses}/{len(p_rows)}")

    print(f"\n{'='*70}")
    print(f"  Results written to:")
    print(f"    {OUT_CSV}")
    print(f"    {OUT_JSON}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
