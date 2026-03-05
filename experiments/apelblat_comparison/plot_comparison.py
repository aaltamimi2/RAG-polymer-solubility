#!/usr/bin/env python3
"""Visualization for Maier-Kelley vs Apelblat comparison.

Run AFTER fit_compare.py. Reads comparison_results.csv.
Writes figures to experiments/apelblat_comparison/plots/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"
OUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUT_DIR / "comparison_results.csv"
PLOT_DIR = OUT_DIR / "plots"

# ── Style (matches visualize_interpolation_model.py) ─────────────
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

POLYMER_ORDER = ["EVOH", "HDPE", "LDPE", "Nylon6", "Nylon66",
                 "PC", "PES", "PET", "PP", "PS", "PVC"]

MK_COLOR = "#e74c3c"      # red
APEL_COLOR = "#2980b9"    # blue
TIE_COLOR = "#95a5a6"     # grey


def load_results():
    return pd.read_csv(RESULTS_CSV)


def load_raw_data():
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
    return df


# ── Model functions (for curve plotting) ─────────────────────────
def predict_mk(A, B, C, temp_c):
    t_k = temp_c + 273.15
    return np.clip(np.exp(A + B / t_k + C / t_k**2), 0.0, 100.0)


def predict_apel(A, B, C, temp_c):
    t_k = temp_c + 273.15
    return np.clip(np.exp(A + B / t_k + C * np.log(t_k)), 0.0, 100.0)


# ==================================================================
# Figure 1: R² scatter — MK vs Apelblat
# ==================================================================
def fig1_r2_scatter(res):
    both = res.dropna(subset=["mk_r2", "apel_r2"])

    fig, ax = plt.subplots(figsize=(8, 8))

    for polymer in POLYMER_ORDER:
        sub = both[both.polymer == polymer]
        if sub.empty:
            continue
        ax.scatter(sub.mk_r2, sub.apel_r2, s=25, alpha=0.7, label=polymer, zorder=3)

    lims = [min(both.mk_r2.min(), both.apel_r2.min()) - 0.01, 1.001]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="Equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("R² — Maier-Kelley (A + B/T + C/T²)")
    ax.set_ylabel("R² — Apelblat (A + B/T + C·ln T)")
    ax.set_title("Per-Pair R² Comparison")
    ax.legend(fontsize=7, markerscale=1.5, ncol=2, loc="lower right")
    ax.set_aspect("equal")

    above = (both.apel_r2 > both.mk_r2 + 1e-5).sum()
    below = (both.apel_r2 < both.mk_r2 - 1e-5).sum()
    ax.text(0.03, 0.97,
            f"Above diagonal (Apelblat better): {above}\n"
            f"Below diagonal (MK better): {below}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig1_r2_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_r2_scatter.png")


# ==================================================================
# Figure 2: Delta-R² distribution
# ==================================================================
def fig2_delta_r2_dist(res):
    both = res.dropna(subset=["delta_r2"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all pairs
    deltas = both.delta_r2.values
    colors = [APEL_COLOR if d > 1e-5 else MK_COLOR if d < -1e-5 else TIE_COLOR
              for d in deltas]

    ax1.hist(deltas, bins=60, color=APEL_COLOR, edgecolor="white", alpha=0.75)
    ax1.axvline(0, color="black", ls="-", lw=1.5)
    ax1.set_xlabel("Delta R² (Apelblat - MK)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"All Pairs (n={len(deltas)})")
    ax1.text(0.03, 0.97,
             f"Mean: {np.mean(deltas):+.2e}\nMedian: {np.median(deltas):+.2e}",
             transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # Right: zoom on anomalous MK pairs
    anom = both[both.mk_category == "anomalous"]
    if not anom.empty:
        labels = [f"{r.polymer}/{r.solvent}" for _, r in anom.iterrows()]
        y = np.arange(len(labels))
        bar_colors = [APEL_COLOR if d > 0 else MK_COLOR for d in anom.delta_r2]
        ax2.barh(y, anom.delta_r2.values, color=bar_colors, edgecolor="white")
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.axvline(0, color="black", ls="-", lw=1)
        ax2.set_xlabel("Delta R²")
        ax2.set_title(f"MK-Anomalous Pairs (n={len(anom)})")
        ax2.invert_yaxis()

        for i, (_, r) in enumerate(anom.iterrows()):
            if r.mk_category == "anomalous" and r.apel_category == "fitted":
                ax2.text(r.delta_r2, i, " RESCUED", va="center",
                         fontsize=7, fontweight="bold", color="#27ae60")
    else:
        ax2.text(0.5, 0.5, "No anomalous MK pairs", ha="center",
                 va="center", transform=ax2.transAxes)

    fig.suptitle("Delta R² Distribution (Apelblat - Maier-Kelley)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig2_delta_r2_dist.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_delta_r2_dist.png")


# ==================================================================
# Figure 3: Per-polymer bar chart of mean delta R²
# ==================================================================
def fig3_per_polymer(res):
    both = res.dropna(subset=["delta_r2"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: mean delta R²
    polymers = [p for p in POLYMER_ORDER if p in both.polymer.values]
    means = []
    stds = []
    for p in polymers:
        d = both[both.polymer == p].delta_r2.values
        means.append(np.mean(d))
        stds.append(np.std(d))

    bar_colors = [APEL_COLOR if m > 1e-5 else MK_COLOR if m < -1e-5 else TIE_COLOR
                  for m in means]
    x = np.arange(len(polymers))
    ax1.bar(x, means, yerr=stds, color=bar_colors, edgecolor="white",
            capsize=3, alpha=0.85)
    ax1.axhline(0, color="black", ls="-", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(polymers, rotation=45, ha="right")
    ax1.set_ylabel("Mean Delta R²")
    ax1.set_title("Per-Polymer Mean Improvement")

    # Right: win/loss counts
    win_counts = []
    loss_counts = []
    tie_counts = []
    for p in polymers:
        sub = both[both.polymer == p]
        win_counts.append((sub.winner == "apelblat").sum())
        loss_counts.append((sub.winner == "mk").sum())
        tie_counts.append((sub.winner == "tie").sum())

    ax2.bar(x, win_counts, label="Apelblat wins", color=APEL_COLOR, alpha=0.85)
    ax2.bar(x, tie_counts, bottom=win_counts, label="Tie", color=TIE_COLOR, alpha=0.85)
    bottom2 = np.array(win_counts) + np.array(tie_counts)
    ax2.bar(x, loss_counts, bottom=bottom2, label="MK wins", color=MK_COLOR, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(polymers, rotation=45, ha="right")
    ax2.set_ylabel("Pair count")
    ax2.set_title("Win/Loss per Polymer")
    ax2.legend(fontsize=8)

    fig.suptitle("Per-Polymer Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig3_per_polymer.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_per_polymer.png")


# ==================================================================
# Figure 4: MAE comparison scatter (S% space)
# ==================================================================
def fig4_mae_comparison(res):
    both = res.dropna(subset=["mk_mae_pct", "apel_mae_pct"])

    fig, ax = plt.subplots(figsize=(8, 8))

    for polymer in POLYMER_ORDER:
        sub = both[both.polymer == polymer]
        if sub.empty:
            continue
        ax.scatter(sub.mk_mae_pct, sub.apel_mae_pct, s=25, alpha=0.7, label=polymer)

    mx = max(both.mk_mae_pct.max(), both.apel_mae_pct.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("MAE (%) — Maier-Kelley")
    ax.set_ylabel("MAE (%) — Apelblat")
    ax.set_title("Per-Pair MAE in S% Space")
    ax.legend(fontsize=7, markerscale=1.5, ncol=2, loc="upper left")
    ax.set_aspect("equal")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig4_mae_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_mae_comparison.png")


# ==================================================================
# Figure 5: Top-12 most-improved pair curve overlays
# ==================================================================
def fig5_top12_curves(res, raw_df):
    both = res.dropna(subset=["delta_r2"])
    top12 = both.nlargest(12, "delta_r2")

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top12.iterrows()):
        ax = axes[idx]
        polymer, solvent = row["polymer"], row["solvent"]

        grp = raw_df[(raw_df.polymer == polymer) & (raw_df.solvent == solvent)]
        grp = grp.sort_values("temperature_c")
        temps = grp.temperature_c.values
        sols = grp.solubility_pct.values

        # Raw data
        mask_100 = sols == 100.0
        ax.scatter(temps[~mask_100], sols[~mask_100], c="#333", s=15,
                   zorder=4, label="Data", alpha=0.7)
        if np.any(mask_100):
            ax.scatter(temps[mask_100], sols[mask_100], c="#e74c3c", s=25,
                       marker="x", zorder=4, label="Dropped", linewidths=1.5)

        # Model curves
        t_fine = np.linspace(temps.min(), temps.max(), 300)
        if pd.notna(row.mk_A):
            s_mk = predict_mk(row.mk_A, row.mk_B, row.mk_C, t_fine)
            ax.plot(t_fine, s_mk, color=MK_COLOR, lw=2, label=f"MK (R²={row.mk_r2:.5f})")
        if pd.notna(row.apel_A):
            s_apel = predict_apel(row.apel_A, row.apel_B, row.apel_C, t_fine)
            ax.plot(t_fine, s_apel, color=APEL_COLOR, lw=2, ls="--",
                    label=f"Apel (R²={row.apel_r2:.5f})")

        ax.set_title(f"{polymer} / {solvent}", fontsize=10, fontweight="bold")
        ax.text(0.02, 0.98, f"ΔR²={row.delta_r2:+.2e}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax.set_xlabel("T (°C)")
        ax.set_ylabel("S (%)")
        ax.legend(fontsize=6.5, loc="lower right")

    fig.suptitle("Top 12 Pairs Where Apelblat Improves Most Over MK",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig5_top12_improved.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_top12_improved.png")


# ==================================================================
# Figure 6: Residual distributions side-by-side
# ==================================================================
def fig6_residual_overlay(res, raw_df):
    both = res.dropna(subset=["mk_r2", "apel_r2"])

    mk_residuals = []
    apel_residuals = []

    for _, row in both.iterrows():
        grp = raw_df[(raw_df.polymer == row.polymer) & (raw_df.solvent == row.solvent)]
        grp = grp[grp.solubility_pct != 100.0].sort_values("temperature_c")
        if grp.empty:
            continue

        temps_c = grp.temperature_c.values
        sols = grp.solubility_pct.values
        s_clamped = np.clip(sols, 1e-12, 100.0)
        ln_s = np.log(s_clamped)
        t_k = temps_c + 273.15

        if pd.notna(row.mk_A):
            ln_pred_mk = row.mk_A + row.mk_B / t_k + row.mk_C / t_k**2
            mk_residuals.extend((ln_pred_mk - ln_s).tolist())

        if pd.notna(row.apel_A):
            ln_pred_ap = row.apel_A + row.apel_B / t_k + row.apel_C * np.log(t_k)
            apel_residuals.extend((ln_pred_ap - ln_s).tolist())

    mk_residuals = np.array(mk_residuals)
    apel_residuals = np.array(apel_residuals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlaid histograms
    bins = np.linspace(
        min(mk_residuals.min(), apel_residuals.min()),
        max(mk_residuals.max(), apel_residuals.max()),
        100,
    )
    ax1.hist(mk_residuals, bins=bins, alpha=0.6, color=MK_COLOR,
             label=f"MK (σ={np.std(mk_residuals):.4f})", edgecolor="white")
    ax1.hist(apel_residuals, bins=bins, alpha=0.6, color=APEL_COLOR,
             label=f"Apelblat (σ={np.std(apel_residuals):.4f})", edgecolor="white")
    ax1.axvline(0, color="black", ls="-", lw=1)
    ax1.set_xlabel("Residual in ln(S) space")
    ax1.set_ylabel("Count")
    ax1.set_title("Residual Distribution (all fitted pairs)")
    ax1.legend()

    # Right: QQ-style cumulative comparison
    for arr, color, label in [(mk_residuals, MK_COLOR, "MK"),
                               (apel_residuals, APEL_COLOR, "Apelblat")]:
        sorted_abs = np.sort(np.abs(arr))
        cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
        ax2.plot(sorted_abs, cdf, color=color, lw=1.5, label=label)

    ax2.set_xlabel("|Residual| in ln(S) space")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("Cumulative |Residual| Distribution")
    ax2.legend()
    ax2.set_xlim(0, np.percentile(np.abs(mk_residuals), 99.5))

    fig.suptitle("Residual Analysis — MK vs Apelblat", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig6_residual_overlay.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_residual_overlay.png")


# ==================================================================
# Main
# ==================================================================
def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading comparison results...")
    res = load_results()
    print("Loading raw data for curve plots...")
    raw_df = load_raw_data()
    print(f"Generating figures to {PLOT_DIR}/\n")

    fig1_r2_scatter(res)
    fig2_delta_r2_dist(res)
    fig3_per_polymer(res)
    fig4_mae_comparison(res)
    fig5_top12_curves(res, raw_df)
    fig6_residual_overlay(res, raw_df)

    print(f"\nDone — 6 figures saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
