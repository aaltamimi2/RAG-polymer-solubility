#!/usr/bin/env python3
"""Extensive visualization suite for the solubility interpolation model.

Generates publication-quality figures characterizing model quality,
coverage, and per-pair fit diagnostics.

Model location for inference:
  Coefficients: data/solubility_coefficients.json
  Tool module:  src/strap/tools/interpolation.py
  Loader:       interpolation._load_coefficients() (lazy singleton)

Figures saved to: plots/interpolation/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"
COEFF_PATH = DATA_DIR / "solubility_coefficients.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "interpolation"

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

POLYMER_ORDER = ["EVOH", "HDPE", "LDPE", "Nylon6", "Nylon66", "PC", "PES", "PET", "PP", "PS", "PVC"]
CAT_COLORS = {"fitted": "#2ecc71", "anomalous": "#e74c3c", "insoluble": "#95a5a6", "saturated": "#3498db"}


def _load_data():
    """Load CSV and coefficients."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "solvent" in cl: col_map[c] = "solvent"
        elif "temperature" in cl: col_map[c] = "temperature_c"
        elif "solubility" in cl: col_map[c] = "solubility_pct"
        elif "polymer" in cl: col_map[c] = "polymer"
    df.rename(columns=col_map, inplace=True)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["solubility_pct"] = pd.to_numeric(df["solubility_pct"], errors="coerce")
    df.dropna(subset=["temperature_c", "solubility_pct"], inplace=True)

    with open(COEFF_PATH) as f:
        coeffs = json.load(f)

    lookup = {}
    for entry in coeffs["entries"]:
        key = (entry["polymer"].strip(), entry["solvent"].strip())
        lookup[key] = entry

    return df, coeffs, lookup


def _predict_s(entry, temp_c):
    t_k = temp_c + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] * np.log(t_k)
    return float(np.clip(np.exp(ln_s), 0.0, 100.0))


# ==================================================================
# Figure 1: R² Heatmap (polymer × solvent)
# ==================================================================
def fig1_r2_heatmap(coeffs, lookup):
    solvents = sorted({e["solvent"] for e in coeffs["entries"]})
    polymers = [p for p in POLYMER_ORDER if p in {e["polymer"] for e in coeffs["entries"]}]

    matrix = np.full((len(polymers), len(solvents)), np.nan)
    cat_matrix = np.empty((len(polymers), len(solvents)), dtype=object)

    for i, p in enumerate(polymers):
        for j, s in enumerate(solvents):
            e = lookup.get((p, s))
            if e is None:
                continue
            cat_matrix[i, j] = e["category"]
            if e["r_squared"] is not None:
                matrix[i, j] = e["r_squared"]

    fig, ax = plt.subplots(figsize=(18, 6))

    # Custom colormap: grey for NaN, red→yellow→green for R²
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.90, vmax=1.0,
                   interpolation="nearest")

    # Mark insoluble/anomalous with symbols
    for i in range(len(polymers)):
        for j in range(len(solvents)):
            cat = cat_matrix[i, j]
            if cat == "insoluble":
                ax.text(j, i, "X", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="#555")
            elif cat == "anomalous":
                ax.text(j, i, "!", ha="center", va="center", fontsize=8,
                        fontweight="bold", color="#c0392b")
            elif cat == "fitted" and matrix[i, j] < 0.995:
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                        fontsize=5.5, color="black")

    ax.set_xticks(range(len(solvents)))
    ax.set_xticklabels(solvents, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(polymers)))
    ax.set_yticklabels(polymers, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("R²")

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Fitted (R² ≥ 0.98)"),
        Patch(facecolor="#e74c3c", label="! Anomalous (R² < 0.98)"),
        Patch(facecolor="#95a5a6", label="X Insoluble"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9)

    ax.set_title("Interpolation Model R² — Polymer × Solvent (352 pairs)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_r2_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_r2_heatmap.png")


# ==================================================================
# Figure 2: R² Distribution Histogram
# ==================================================================
def fig2_r2_distribution(coeffs):
    fitted = [e for e in coeffs["entries"] if e["category"] == "fitted"]
    anomalous = [e for e in coeffs["entries"] if e["category"] == "anomalous"]
    r2_fitted = [e["r_squared"] for e in fitted]
    r2_anom = [e["r_squared"] for e in anomalous]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all fitted
    ax1.hist(r2_fitted, bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax1.axvline(0.99, color="#e67e22", ls="--", lw=1.5, label="R² = 0.99")
    ax1.axvline(0.999, color="#e74c3c", ls="--", lw=1.5, label="R² = 0.999")
    ax1.set_xlabel("R²")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Fitted Pairs (n={len(fitted)})")
    ax1.legend()

    n_above_99 = sum(1 for r in r2_fitted if r >= 0.99)
    n_above_999 = sum(1 for r in r2_fitted if r >= 0.999)
    ax1.text(0.05, 0.95,
             f"≥ 0.99: {n_above_99} ({100*n_above_99/len(r2_fitted):.1f}%)\n"
             f"≥ 0.999: {n_above_999} ({100*n_above_999/len(r2_fitted):.1f}%)",
             transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Right: anomalous
    if r2_anom:
        ax2.barh(range(len(r2_anom)),
                 [r for r in sorted(r2_anom, reverse=True)],
                 color="#e74c3c", edgecolor="white")
        labels = [f"{e['polymer']}/{e['solvent']}" for e in
                  sorted(anomalous, key=lambda x: -x["r_squared"])]
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.axvline(0.98, color="#2ecc71", ls="--", lw=1.5, label="Threshold")
        ax2.set_xlabel("R²")
        ax2.set_title(f"Anomalous Pairs (n={len(anomalous)})")
        ax2.legend()
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "No anomalous pairs", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.suptitle("Model Fit Quality Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_r2_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_r2_distribution.png")


# ==================================================================
# Figure 3: Category Breakdown (donut + per-polymer bar)
# ==================================================================
def fig3_category_breakdown(coeffs):
    cats = coeffs["categories"]
    entries = coeffs["entries"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    # Donut chart
    labels = [k for k, v in cats.items() if v > 0]
    sizes = [cats[k] for k in labels]
    colors = [CAT_COLORS[k] for k in labels]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.set_title(f"Overall: {sum(sizes)} pairs")

    # Per-polymer stacked bar
    polymers = [p for p in POLYMER_ORDER if p in {e["polymer"] for e in entries}]
    cat_names = ["fitted", "anomalous", "insoluble", "saturated"]

    data = {cat: [] for cat in cat_names}
    for p in polymers:
        p_entries = [e for e in entries if e["polymer"] == p]
        for cat in cat_names:
            data[cat].append(sum(1 for e in p_entries if e["category"] == cat))

    x = np.arange(len(polymers))
    bottom = np.zeros(len(polymers))
    for cat in cat_names:
        vals = np.array(data[cat])
        if np.sum(vals) == 0:
            continue
        ax2.bar(x, vals, bottom=bottom, label=cat, color=CAT_COLORS[cat],
                edgecolor="white", linewidth=0.5)
        bottom += vals

    ax2.set_xticks(x)
    ax2.set_xticklabels(polymers, rotation=45, ha="right")
    ax2.set_ylabel("Number of solvent pairs")
    ax2.set_title("Per-Polymer Category Breakdown")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Interpolation Model Coverage", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_category_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_category_breakdown.png")


# ==================================================================
# Figure 4: Sample fit curves — 12 representative pairs (4×3 grid)
# ==================================================================
def fig4_sample_fit_curves(df, lookup):
    # Choose diverse pairs: some excellent, some marginal, some with dropped points
    sample_pairs = [
        ("HDPE", "toluene"),     ("PS", "chcl3"),          ("LDPE", "hexane"),
        ("PC", "ch2cl2"),        ("Nylon6", "methanol"),   ("PVC", "thf"),
        ("EVOH", "dimethylformamide"), ("PET", "diphenylether"), ("PP", "cyclohexane"),
        ("HDPE", "hexane"),      ("PS", "thf"),            ("EVOH", "triethylamine"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    axes = axes.flatten()

    for idx, (polymer, solvent) in enumerate(sample_pairs):
        ax = axes[idx]
        entry = lookup.get((polymer, solvent))
        grp = df[(df.polymer == polymer) & (df.solvent == solvent)].sort_values("temperature_c")

        if grp.empty or entry is None:
            ax.text(0.5, 0.5, f"No data\n{polymer}/{solvent}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        temps = grp["temperature_c"].values
        sols = grp["solubility_pct"].values

        # Mark 100% points
        mask_100 = sols == 100.0
        ax.scatter(temps[~mask_100], sols[~mask_100], c="#3498db", s=20,
                   zorder=3, label="Data (used)", alpha=0.8)
        if np.any(mask_100):
            ax.scatter(temps[mask_100], sols[mask_100], c="#e74c3c", s=30,
                       marker="x", zorder=3, label="Dropped (S=100%)", linewidths=1.5)

        # Model curve
        if entry["category"] == "fitted" or entry["category"] == "anomalous":
            t_fine = np.linspace(temps.min(), temps.max(), 200)
            s_pred = [_predict_s(entry, t) for t in t_fine]
            ax.plot(t_fine, s_pred, color="#e67e22", lw=2, label="Model", zorder=2)

        cat = entry["category"]
        r2 = entry["r_squared"]
        n_drop = entry.get("n_dropped_100", 0)
        title = f"{polymer} / {solvent}"
        info = f"Cat: {cat}"
        if r2 is not None:
            info += f" | R²={r2:.4f}"
        if n_drop > 0:
            info += f" | {n_drop} dropped"

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top",
                fontsize=7.5, bbox=dict(boxstyle="round,pad=0.2",
                facecolor="white", alpha=0.8))
        ax.set_xlabel("T (°C)")
        ax.set_ylabel("Solubility (%)")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Sample Fit Curves — Data vs Model", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_sample_fit_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_sample_fit_curves.png")


# ==================================================================
# Figure 5: Actual vs Predicted scatter (all fitted pairs)
# ==================================================================
def fig5_actual_vs_predicted(df, lookup):
    all_actual = []
    all_predicted = []
    pair_labels = []

    for (polymer, solvent), grp in df.groupby(["polymer", "solvent"]):
        entry = lookup.get((polymer, solvent))
        if entry is None or entry["category"] != "fitted":
            continue

        # Skip 100% points for consistency with fitting
        grp = grp[grp.solubility_pct != 100.0]
        for _, row in grp.iterrows():
            actual = row["solubility_pct"]
            pred = _predict_s(entry, row["temperature_c"])
            all_actual.append(actual)
            all_predicted.append(pred)
            pair_labels.append(polymer)

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: linear scale
    for p in POLYMER_ORDER:
        mask = np.array(pair_labels) == p
        if np.sum(mask) == 0:
            continue
        ax1.scatter(all_actual[mask], all_predicted[mask], s=3, alpha=0.4, label=p)
    ax1.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Perfect fit")
    ax1.set_xlabel("Actual Solubility (%)")
    ax1.set_ylabel("Predicted Solubility (%)")
    ax1.set_title("Linear Scale")
    ax1.legend(fontsize=7, markerscale=3, ncol=2)
    ax1.set_xlim(-2, 102)
    ax1.set_ylim(-2, 102)

    # Right: log scale (clip to avoid log(0))
    actual_log = np.clip(all_actual, 1e-12, None)
    pred_log = np.clip(all_predicted, 1e-12, None)
    for p in POLYMER_ORDER:
        mask = np.array(pair_labels) == p
        if np.sum(mask) == 0:
            continue
        ax2.scatter(actual_log[mask], pred_log[mask], s=3, alpha=0.4, label=p)
    lims = [1e-12, 100]
    ax2.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Actual Solubility (%)")
    ax2.set_ylabel("Predicted Solubility (%)")
    ax2.set_title("Log Scale (shows low-solubility precision)")
    ax2.set_xlim(1e-12, 200)
    ax2.set_ylim(1e-12, 200)

    residuals = all_predicted - all_actual
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    fig.suptitle(
        f"Actual vs Predicted — All Fitted Pairs "
        f"(n={len(all_actual):,} points, MAE={mae:.3f}%, RMSE={rmse:.3f}%)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_actual_vs_predicted.png")


# ==================================================================
# Figure 6: Residual analysis
# ==================================================================
def fig6_residual_analysis(df, lookup):
    residuals = []
    temps_all = []
    actual_all = []

    for (polymer, solvent), grp in df.groupby(["polymer", "solvent"]):
        entry = lookup.get((polymer, solvent))
        if entry is None or entry["category"] != "fitted":
            continue
        grp = grp[grp.solubility_pct != 100.0]
        for _, row in grp.iterrows():
            pred = _predict_s(entry, row["temperature_c"])
            residuals.append(pred - row["solubility_pct"])
            temps_all.append(row["temperature_c"])
            actual_all.append(row["solubility_pct"])

    residuals = np.array(residuals)
    temps_all = np.array(temps_all)
    actual_all = np.array(actual_all)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: residual histogram
    ax = axes[0, 0]
    ax.hist(residuals, bins=100, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Residual (Predicted - Actual) [%]")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (mean={np.mean(residuals):.4f}%)")
    p5, p95 = np.percentile(residuals, [5, 95])
    ax.text(0.95, 0.95,
            f"σ = {np.std(residuals):.4f}%\n"
            f"5th %ile = {p5:.3f}%\n"
            f"95th %ile = {p95:.3f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Top-right: residual vs temperature
    ax = axes[0, 1]
    ax.scatter(temps_all, residuals, s=1, alpha=0.15, c="#2c3e50")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residuals vs Temperature")

    # Bottom-left: residual vs actual (shows heteroscedasticity)
    ax = axes[1, 0]
    ax.scatter(actual_all, residuals, s=1, alpha=0.15, c="#8e44ad")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Actual Solubility (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residuals vs Actual Solubility")

    # Bottom-right: absolute error vs actual (log-log)
    ax = axes[1, 1]
    abs_err = np.abs(residuals)
    ax.scatter(np.clip(actual_all, 1e-12, None), np.clip(abs_err, 1e-12, None),
               s=1, alpha=0.15, c="#e67e22")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual Solubility (%)")
    ax.set_ylabel("|Residual| (%)")
    ax.set_title("Absolute Error vs Actual (log scale)")

    fig.suptitle("Residual Analysis — All Fitted Pairs", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_residual_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_residual_analysis.png")


# ==================================================================
# Figure 7: Per-pair error heatmap (MAE)
# ==================================================================
def fig7_error_heatmap(df, lookup, coeffs):
    solvents = sorted({e["solvent"] for e in coeffs["entries"]})
    polymers = [p for p in POLYMER_ORDER if p in {e["polymer"] for e in coeffs["entries"]}]

    mae_matrix = np.full((len(polymers), len(solvents)), np.nan)

    for (polymer, solvent), grp in df.groupby(["polymer", "solvent"]):
        entry = lookup.get((polymer, solvent))
        if entry is None or entry["category"] != "fitted":
            continue
        grp = grp[grp.solubility_pct != 100.0]
        if grp.empty:
            continue
        actual = grp["solubility_pct"].values
        predicted = np.array([_predict_s(entry, t) for t in grp["temperature_c"].values])
        mae = np.mean(np.abs(actual - predicted))

        pi = polymers.index(polymer) if polymer in polymers else None
        si = solvents.index(solvent) if solvent in solvents else None
        if pi is not None and si is not None:
            mae_matrix[pi, si] = mae

    fig, ax = plt.subplots(figsize=(18, 6))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(mae_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=5.0,
                   interpolation="nearest")

    # Annotate cells with MAE > 2
    for i in range(len(polymers)):
        for j in range(len(solvents)):
            val = mae_matrix[i, j]
            if not np.isnan(val) and val > 2.0:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 3.5 else "black")

    ax.set_xticks(range(len(solvents)))
    ax.set_xticklabels(solvents, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(polymers)))
    ax.set_yticklabels(polymers, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MAE (%)")

    ax.set_title("Mean Absolute Error — Polymer × Solvent (fitted pairs only)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_error_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig7_error_heatmap.png")


# ==================================================================
# Figure 8: Impact of dropping S=100% points
# ==================================================================
def fig8_drop100_impact(df, lookup, coeffs):
    """Before/after comparison for pairs that had 100% points dropped."""
    affected = [e for e in coeffs["entries"] if e.get("n_dropped_100", 0) > 0]
    if not affected:
        print("  fig8 skipped: no pairs with dropped 100% points")
        return

    affected.sort(key=lambda e: -e.get("n_dropped_100", 0))

    # Re-fit WITH 100% points to get comparison R²
    def model(t_k, a, b, c):
        return a + b/t_k + c * np.log(t_k)

    def r_sq(y, yp):
        ss_res = np.sum((y - yp)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        if ss_tot == 0: return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res/ss_tot

    pairs_data = []
    for entry in affected:
        p, s = entry["polymer"], entry["solvent"]
        grp = df[(df.polymer == p) & (df.solvent == s)].sort_values("temperature_c")
        sols_all = grp["solubility_pct"].values
        temps_all = grp["temperature_c"].values

        # Fit WITH 100%
        s_clamp = np.clip(sols_all, 1e-12, 100.0)
        ln_s = np.log(s_clamp)
        t_k = temps_all + 273.15
        try:
            popt_w, _ = curve_fit(model, t_k, ln_s, p0=[0,0,0], maxfev=10000)
            r2_with = r_sq(ln_s, model(t_k, *popt_w))
        except Exception:
            r2_with = 0.0

        r2_drop = entry["r_squared"] if entry["r_squared"] else 0.0
        pairs_data.append({
            "label": f"{p}/{s}",
            "r2_with": r2_with,
            "r2_drop": r2_drop,
            "n_dropped": entry.get("n_dropped_100", 0),
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: paired bar chart
    labels = [d["label"] for d in pairs_data]
    r2_with = [d["r2_with"] for d in pairs_data]
    r2_drop = [d["r2_drop"] for d in pairs_data]

    y = np.arange(len(labels))
    height = 0.35
    ax1.barh(y - height/2, r2_with, height, color="#e74c3c", alpha=0.7,
             label="With S=100%")
    ax1.barh(y + height/2, r2_drop, height, color="#2ecc71", alpha=0.7,
             label="Dropped S=100%")
    ax1.axvline(0.98, color="#333", ls="--", lw=1, alpha=0.7, label="Threshold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("R²")
    ax1.set_title("R² Before/After Dropping S=100%")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0.7, 1.01)
    ax1.invert_yaxis()

    # Right: delta R² with annotations
    deltas = [d["r2_drop"] - d["r2_with"] for d in pairs_data]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
    ax2.barh(y, deltas, color=colors, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_xlabel("ΔR² (drop - with)")
    ax2.set_title("Improvement from Dropping S=100%")
    ax2.axvline(0, color="#333", ls="-", lw=1)
    ax2.invert_yaxis()

    # Annotate rescued pairs
    for i, d in enumerate(pairs_data):
        if d["r2_with"] < 0.98 and d["r2_drop"] >= 0.98:
            ax2.text(deltas[i] + 0.002, i, "RESCUED", va="center",
                     fontsize=7, fontweight="bold", color="#27ae60")

    fig.suptitle(
        f"Impact of Removing COSMO-RS NaN→100% Artifacts "
        f"({len(pairs_data)} affected pairs, {sum(d['n_dropped'] for d in pairs_data)} points dropped)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_drop100_impact.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_drop100_impact.png")


# ==================================================================
# Figure 9: Model architecture summary (infographic)
# ==================================================================
def fig9_model_summary(coeffs):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    cats = coeffs["categories"]
    n_total = coeffs["n_entries"]
    n_dropped = coeffs.get("n_points_dropped_100", 0)
    fitted = [e for e in coeffs["entries"] if e["category"] == "fitted"]
    r2s = [e["r_squared"] for e in fitted]

    text = (
        "SOLUBILITY INTERPOLATION MODEL\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Model:     ln(S%) = A + B/T_K + C·ln(T_K)  (modified Apelblat)\n"
        f"Source:    COSMO-RS simulation data\n"
        f"CSV:       data/COMMON-SOLVENTS-DATABASE.csv\n\n"
        f"MODEL STORED AT\n"
        f"  Coefficients:  data/solubility_coefficients.json\n"
        f"  Inference:     src/strap/tools/interpolation.py\n"
        f"  Loader:        interpolation._load_coefficients()\n\n"
        f"COVERAGE\n"
        f"  Total pairs:   {n_total}\n"
        f"  Polymers:      11    (EVOH, HDPE, LDPE, Nylon6, Nylon66,\n"
        f"                        PC, PES, PET, PP, PS, PVC)\n"
        f"  Solvents:      32    (common organic + water)\n"
        f"  Temperatures:  25–160°C (5°C steps, 28 points/pair)\n\n"
        f"CATEGORIES\n"
        f"  Fitted:        {cats['fitted']:>3}  ({100*cats['fitted']/n_total:.1f}%)\n"
        f"  Anomalous:     {cats['anomalous']:>3}  (R² < 0.98)\n"
        f"  Insoluble:     {cats['insoluble']:>3}  (S ≈ 0 at all T)\n\n"
        f"FIT QUALITY (fitted pairs)\n"
        f"  R² min:        {min(r2s):.6f}\n"
        f"  R² mean:       {np.mean(r2s):.6f}\n"
        f"  R² ≥ 0.99:    {sum(1 for r in r2s if r >= 0.99)} / {len(r2s)}\n"
        f"  R² ≥ 0.999:   {sum(1 for r in r2s if r >= 0.999)} / {len(r2s)}\n\n"
        f"DATA CLEANING\n"
        f"  Dropped {n_dropped} exact S=100% points\n"
        f"  (COSMO-RS NaN → 100% artifacts)"
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1",
                      edgecolor="#bdc3c7", linewidth=2))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_model_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig9_model_summary.png")


# ==================================================================
# Figure 10: All anomalous pair curves
# ==================================================================
def fig10_anomalous_pairs(df, lookup, coeffs):
    anomalous = [e for e in coeffs["entries"] if e["category"] == "anomalous"]
    if not anomalous:
        print("  fig10 skipped: no anomalous pairs")
        return

    n = len(anomalous)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, entry in enumerate(sorted(anomalous, key=lambda e: e["r_squared"] or 0)):
        ax = axes_flat[idx]
        p, s = entry["polymer"], entry["solvent"]
        grp = df[(df.polymer == p) & (df.solvent == s)].sort_values("temperature_c")

        temps = grp["temperature_c"].values
        sols = grp["solubility_pct"].values

        mask_100 = sols == 100.0
        ax.scatter(temps[~mask_100], sols[~mask_100], c="#3498db", s=15, zorder=3)
        if np.any(mask_100):
            ax.scatter(temps[mask_100], sols[mask_100], c="#e74c3c", s=25,
                       marker="x", zorder=3, linewidths=1.5)

        if entry["A"] is not None:
            t_fine = np.linspace(temps.min(), temps.max(), 200)
            s_pred = [_predict_s(entry, t) for t in t_fine]
            ax.plot(t_fine, s_pred, color="#e67e22", lw=1.5, alpha=0.7)

        r2 = entry["r_squared"]
        ax.set_title(f"{p}/{s}\nR²={r2:.4f}" if r2 else f"{p}/{s}", fontsize=8)
        ax.set_xlabel("T (°C)", fontsize=7)
        ax.set_ylabel("S (%)", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"All Anomalous Pairs (n={n}) — Model Cannot Fit These Well",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_anomalous_pairs.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig10_anomalous_pairs.png")


# ==================================================================
# Main
# ==================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading data...")
    df, coeffs, lookup = _load_data()
    print(f"Generating figures to {OUT_DIR}/\n")

    fig1_r2_heatmap(coeffs, lookup)
    fig2_r2_distribution(coeffs)
    fig3_category_breakdown(coeffs)
    fig4_sample_fit_curves(df, lookup)
    fig5_actual_vs_predicted(df, lookup)
    fig6_residual_analysis(df, lookup)
    fig7_error_heatmap(df, lookup, coeffs)
    fig8_drop100_impact(df, lookup, coeffs)
    fig9_model_summary(coeffs)
    fig10_anomalous_pairs(df, lookup, coeffs)

    print(f"\nDone — 10 figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
