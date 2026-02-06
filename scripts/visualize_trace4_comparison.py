#!/usr/bin/env python3
"""Compare SQL (database) vs Interpolation predictions for Trace 4.

Trace 4: 3-scheme, 9-polymer separation query.
26 polymer-solvent-temperature pairs across 3 proposed schemes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "COMMON-SOLVENTS-DATABASE.csv"
COEFF_PATH = DATA_DIR / "solubility_coefficients.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "interpolation"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

SOLVENT_ALIASES = {
    "water": "h2o", "acetone": "propanone", "dmf": "dimethylformamide",
    "dmso": "dimethylsulfoxide", "dichloromethane": "ch2cl2",
    "chloroform": "chcl3", "methyl acetate": "methylacetate",
    "tetrahydrofuran": "thf", "ethylene glycol": "glycol",
    "o-xylene": "1,2-dimethylbenzene", "p-xylene": "1,4-dimethylbenzene",
    "n-heptane": "n-heptane", "isopropylamine": "isopropylamine",
    "ethyl acetate": "ethylacetate", "cyclohexanol": "cyclohexanol",
}


def _resolve(name):
    return SOLVENT_ALIASES.get(name.strip().lower(), name.strip().lower())


def _predict_s(entry, temp_c):
    t_k = temp_c + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] / t_k**2
    return float(np.clip(np.exp(ln_s), 0.0, 100.0))


# ------------------------------------------------------------------
# Trace 4 pairs
# ------------------------------------------------------------------
TRACE4_PAIRS = [
    # (polymer, solvent_display, temp, scheme, step)
    ("EVOH", "triethylamine", 25, "S1", 1),
    ("PS", "benzene", 55, "S1", 2),
    ("PVC", "THF", 55, "S1", 3),
    ("PP", "cyclohexane", 75, "S1", 4),
    ("LDPE", "o-xylene", 105, "S1", 5),
    ("HDPE", "o-xylene", 140, "S1", 6),
    ("Nylon6", "DMF", 145, "S1", 7),
    ("Nylon66", "DMSO", 155, "S1", 8),
    ("PET", "ethylene glycol", 190, "S1", 9),

    ("EVOH", "isopropylamine", 25, "S2", 1),
    ("PS", "methyl acetate", 45, "S2", 2),
    ("PVC", "DMF", 80, "S2", 3),
    ("PP", "p-xylene", 100, "S2", 4),
    ("LDPE", "dodecane", 110, "S2", 5),
    ("HDPE", "dodecane", 150, "S2", 6),
    ("Nylon6", "cyclohexanol", 150, "S2", 7),
    ("Nylon66", "ethylene glycol", 170, "S2", 8),
    ("PET", "DMSO", 160, "S2", 9),

    ("EVOH", "ethanol", 75, "S3", 1),
    ("PS", "toluene", 60, "S3", 2),
    ("PVC", "chloroform", 55, "S3", 3),
    ("PP", "n-heptane", 90, "S3", 4),
    ("LDPE", "p-xylene", 110, "S3", 5),
    ("HDPE", "p-xylene", 135, "S3", 6),
    ("Nylon6", "DMSO", 150, "S3", 7),
    ("Nylon66", "DMSO", 170, "S3", 8),
    # PET in S3 = residue (not dissolved), skip
]


def load_data():
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

    with open(COEFF_PATH) as f:
        coeffs = json.load(f)
    lookup = {}
    for e in coeffs["entries"]:
        lookup[(e["polymer"].strip().upper(), e["solvent"].strip().lower())] = e

    return df, coeffs, lookup


def build_comparison(df, lookup):
    results = []
    for polymer, solvent_disp, temp, scheme, step in TRACE4_PAIRS:
        solvent_key = _resolve(solvent_disp)

        # SQL ground truth
        mask = ((df.polymer.str.upper() == polymer.upper()) &
                (df.solvent.str.lower() == solvent_key) &
                (df.temperature_c == temp))
        row = df[mask]
        sql_val = float(row.solubility_pct.values[0]) if len(row) > 0 else None

        # Interpolation
        entry = lookup.get((polymer.upper(), solvent_key))
        interp_val = None
        if entry and entry["category"] == "fitted":
            interp_val = _predict_s(entry, temp)

        is_nan_artifact = sql_val is not None and sql_val == 100.0
        extrap = temp > 160 or temp < 25
        error = abs(interp_val - sql_val) if (interp_val is not None and sql_val is not None) else None

        results.append({
            "scheme": scheme, "step": step,
            "polymer": polymer, "solvent": solvent_disp, "temp": temp,
            "sql": sql_val, "interp": interp_val,
            "error": error, "extrap": extrap,
            "nan_artifact": is_nan_artifact,
            "label": f"{polymer}\n{solvent_disp}\n{temp}°C",
            "short_label": f"{polymer}/{solvent_disp}",
        })
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, coeffs, lookup = load_data()
    results = build_comparison(df, lookup)

    # ================================================================
    # Figure A: Side-by-side bar chart per scheme
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    scheme_colors = {"S1": ("#2c3e50", "#e67e22"), "S2": ("#2c3e50", "#27ae60"), "S3": ("#2c3e50", "#8e44ad")}

    for ax_idx, scheme in enumerate(["S1", "S2", "S3"]):
        ax = axes[ax_idx]
        scheme_data = [r for r in results if r["scheme"] == scheme]
        n = len(scheme_data)
        x = np.arange(n)
        width = 0.35

        sql_vals = []
        interp_vals = []
        labels = []
        for r in scheme_data:
            sql_vals.append(r["sql"] if r["sql"] is not None else 0)
            interp_vals.append(r["interp"] if r["interp"] is not None else 0)
            labels.append(f"Step {r['step']}\n{r['polymer']}\n{r['solvent']}\n{r['temp']}°C")

        c_sql, c_interp = scheme_colors[scheme]
        bars1 = ax.bar(x - width/2, sql_vals, width, label="SQL (database)",
                       color=c_sql, alpha=0.8, edgecolor="white")
        bars2 = ax.bar(x + width/2, interp_vals, width, label="Interpolation",
                       color=c_interp, alpha=0.8, edgecolor="white")

        # Annotate special cases
        for i, r in enumerate(scheme_data):
            if r["nan_artifact"]:
                ax.annotate("NaN→100%\nartifact", (x[i] - width/2, sql_vals[i]),
                           ha="center", va="bottom", fontsize=7, color="#e74c3c",
                           fontweight="bold")
            if r["extrap"]:
                ax.annotate("extrapolation\n(>160°C)", (x[i] + width/2, interp_vals[i] + 1),
                           ha="center", va="bottom", fontsize=7, color="#8e44ad",
                           fontweight="bold")
            if r["error"] is not None and r["error"] > 5 and not r["nan_artifact"]:
                ax.annotate(f"Δ={r['error']:.1f}%", (x[i], max(sql_vals[i], interp_vals[i]) + 1),
                           ha="center", va="bottom", fontsize=7, color="#c0392b")

        scheme_names = {"S1": "Scheme 1: Standard Multi-Solvent",
                       "S2": "Scheme 2: High-BP Alternatives",
                       "S3": "Scheme 3: Aromatic/Polar Focus"}
        ax.set_title(scheme_names[scheme], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, ha="center")
        ax.set_ylabel("Solubility (%)")
        ax.set_ylim(0, 115)
        ax.axhline(100, color="#bdc3c7", ls=":", lw=1)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Trace 4: SQL Database vs Interpolation Model — 3-Scheme 9-Polymer Separation",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig11_trace4_sql_vs_interp_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig11_trace4_sql_vs_interp_bars.png")

    # ================================================================
    # Figure B: Scatter (SQL vs Interp) + error distribution
    # ================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # --- Scatter ---
    valid = [r for r in results if r["sql"] is not None and r["interp"] is not None]
    normal = [r for r in valid if not r["nan_artifact"] and not r["extrap"]]
    nan_pts = [r for r in valid if r["nan_artifact"]]
    extrap_pts = [r for r in valid if r["extrap"]]

    if normal:
        ax1.scatter([r["sql"] for r in normal], [r["interp"] for r in normal],
                   c="#2ecc71", s=60, zorder=3, edgecolor="white", label="Grid match")
    if nan_pts:
        ax1.scatter([r["sql"] for r in nan_pts], [r["interp"] for r in nan_pts],
                   c="#e74c3c", s=80, marker="X", zorder=4, label="S=100% (NaN artifact)")
    if extrap_pts:
        ax1.scatter([r["sql"] for r in extrap_pts], [r["interp"] for r in extrap_pts],
                   c="#8e44ad", s=80, marker="D", zorder=4, label="Extrapolation")

    ax1.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Perfect agreement")
    ax1.set_xlabel("SQL Database (%)")
    ax1.set_ylabel("Interpolation Model (%)")
    ax1.set_title("SQL vs Interpolation")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-5, 110)
    ax1.set_ylim(-5, 110)

    # Annotate worst outliers
    for r in valid:
        if r["error"] and r["error"] > 20:
            ax1.annotate(r["short_label"],
                        (r["sql"], r["interp"]),
                        fontsize=6.5, ha="left",
                        xytext=(5, 5), textcoords="offset points")

    # --- Error histogram (excluding NaN artifacts and extrapolation) ---
    clean = [r for r in normal if r["error"] is not None]
    errors = [r["error"] for r in clean]

    # Split into low and high error for visibility
    low_err = [e for e in errors if e <= 2]
    high_err = [e for e in errors if e > 2]

    ax2.hist(low_err, bins=20, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Absolute Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Error Distribution (grid-point pairs, n={len(clean)})")
    ax2.text(0.95, 0.95,
             f"Median: {np.median(errors):.3f}%\n"
             f"MAE: {np.mean(errors):.3f}%\n"
             f"< 0.5%: {sum(1 for e in errors if e < 0.5)}/{len(errors)}\n"
             f"> 2%: {len(high_err)}/{len(errors)}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Per-pair error bar chart ---
    # Only non-NaN, non-extrapolation
    clean_sorted = sorted(clean, key=lambda r: r["error"], reverse=True)
    labels = [f"{r['polymer']}/{r['solvent']}\n{r['temp']}°C" for r in clean_sorted]
    errs = [r["error"] for r in clean_sorted]
    colors = ["#e74c3c" if e > 2 else "#e67e22" if e > 0.5 else "#2ecc71" for e in errs]

    y = np.arange(len(labels))
    ax3.barh(y, errs, color=colors, edgecolor="white")
    ax3.set_yticks(y)
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_xlabel("Absolute Error (%)")
    ax3.set_title("Per-Pair Error (excl. NaN artifacts)")
    ax3.invert_yaxis()

    fig.suptitle(
        "Interpolation vs SQL — Trace 4 Accuracy Assessment",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig12_trace4_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig12_trace4_accuracy.png")

    # ================================================================
    # Figure C: Full temperature curves for the 3 worst-error pairs
    # ================================================================
    # Find pairs with highest error that aren't NaN artifacts
    worst = [r for r in results if r["error"] is not None and not r["nan_artifact"]
             and not r["extrap"]]
    worst.sort(key=lambda r: -r["error"])
    worst3 = worst[:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, r in enumerate(worst3):
        ax = axes[idx]
        solvent_key = _resolve(r["solvent"])
        polymer = r["polymer"]

        # Get full temperature curve from CSV
        grp = df[(df.polymer.str.upper() == polymer.upper()) &
                 (df.solvent.str.lower() == solvent_key)].sort_values("temperature_c")
        temps = grp.temperature_c.values
        sols = grp.solubility_pct.values

        # Mark 100% and non-100%
        mask_100 = sols == 100.0
        ax.scatter(temps[~mask_100], sols[~mask_100], c="#3498db", s=20,
                   zorder=3, label="CSV data", alpha=0.8)
        if np.any(mask_100):
            ax.scatter(temps[mask_100], sols[mask_100], c="#e74c3c", s=30,
                       marker="x", zorder=3, label="S=100% (dropped)", linewidths=1.5)

        # Model curve
        entry = lookup.get((polymer.upper(), solvent_key))
        if entry and entry["A"] is not None:
            t_fine = np.linspace(25, 160, 300)
            s_pred = [_predict_s(entry, t) for t in t_fine]
            ax.plot(t_fine, s_pred, color="#e67e22", lw=2, label="Interp. model", zorder=2)

        # Mark the specific query point
        ax.scatter([r["temp"]], [r["sql"]], c="#2c3e50", s=100, marker="*",
                   zorder=5, label=f"Query point ({r['temp']}°C)")
        if r["interp"] is not None:
            ax.scatter([r["temp"]], [r["interp"]], c="#e67e22", s=100, marker="*",
                       zorder=5)
            # Draw error line
            ax.plot([r["temp"], r["temp"]], [r["sql"], r["interp"]],
                    color="#e74c3c", lw=2, ls="--", zorder=4)
            ax.annotate(f"Δ = {r['error']:.1f}%",
                       (r["temp"] + 2, (r["sql"] + r["interp"]) / 2),
                       fontsize=9, color="#e74c3c", fontweight="bold")

        ax.set_title(f"{polymer} / {r['solvent']} @ {r['temp']}°C\n"
                     f"SQL={r['sql']:.1f}% → Interp={r['interp']:.1f}%",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Solubility (%)")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Worst-Error Pairs — Why Interpolation Diverges in Steep Transition Zones",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig13_trace4_worst_pairs.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig13_trace4_worst_pairs.png")

    # ================================================================
    # Summary stats
    # ================================================================
    clean_no_nan = [r for r in results if r["error"] is not None
                    and not r["nan_artifact"] and not r["extrap"]]
    errors_clean = [r["error"] for r in clean_no_nan]

    print(f"\n{'='*60}")
    print(f"TRACE 4 COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs tested:      {len(results)}")
    print(f"Grid-point matches:      {len(clean_no_nan)}")
    print(f"NaN→100% artifacts:      {sum(1 for r in results if r['nan_artifact'])}")
    print(f"Extrapolations (>160°C): {sum(1 for r in results if r['extrap'])}")
    print(f"")
    print(f"Grid-point accuracy (excl. artifacts & extrapolations):")
    print(f"  Median error: {np.median(errors_clean):.4f}%")
    print(f"  MAE:          {np.mean(errors_clean):.4f}%")
    print(f"  Max error:    {np.max(errors_clean):.4f}%")
    print(f"  < 0.2%:       {sum(1 for e in errors_clean if e < 0.2)}/{len(errors_clean)}")
    print(f"  < 1%:         {sum(1 for e in errors_clean if e < 1.0)}/{len(errors_clean)}")
    print(f"  > 5%:         {sum(1 for e in errors_clean if e > 5.0)}/{len(errors_clean)}")


if __name__ == "__main__":
    main()
