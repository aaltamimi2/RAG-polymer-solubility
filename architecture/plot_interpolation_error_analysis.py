"""Error analysis of the solubility interpolation model.

Focuses on HDPE/LDPE as the weakest fits vs other polymers.
4-panel figure: polymer MAE ranking, error distributions, worst solvents, cumulative error.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
COEFF_PATH = ROOT / "data" / "solubility_coefficients.json"
CSV_PATH = ROOT / "data" / "COMMON-SOLVENTS-DATABASE.csv"

# ── Style ─────────────────────────────────────────────────────────
PUB_FONT = "Liberation Sans"
PUB_FONTSIZE = 8
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GREY = "#999999"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#E69F00"

HDPE_COLOR = VERMILLION
LDPE_COLOR = GREEN
OTHER_COLOR = BLUE


def apply_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [PUB_FONT, "Arial", "DejaVu Sans"],
        "font.size": PUB_FONTSIZE,
        "axes.labelsize": PUB_FONTSIZE,
        "axes.titlesize": PUB_FONTSIZE,
        "xtick.labelsize": PUB_FONTSIZE,
        "ytick.labelsize": PUB_FONTSIZE,
        "legend.fontsize": PUB_FONTSIZE - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def predict(entry, temp_c):
    t_k = temp_c + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] * np.log(t_k)
    return np.clip(np.exp(ln_s), 0.0, 100.0)


def build_errors(entries, df, t_max=130):
    """Build per-polymer, per-solvent error records."""
    lookup = {(e["polymer"].upper(), e["solvent"].lower()): e
              for e in entries if e["category"] == "fitted"}

    records = []
    for _, row in df.iterrows():
        poly = row["Polymer"].strip().upper()
        solv = row["Solvent"].strip().lower()
        sol = row["Solubility (%)"]
        temp = row["Temperature (°C)"]
        if sol >= 100.0 or temp > t_max:
            continue
        entry = lookup.get((poly, solv))
        if entry is None:
            continue
        pred = predict(entry, temp)
        records.append({
            "polymer": poly, "solvent": solv, "temp": temp,
            "actual": sol, "pred": pred,
            "error": sol - pred, "abs_error": abs(sol - pred),
        })
    return records


def plot_error_analysis(entries, df, t_max=130):
    apply_pub_style()

    records = build_errors(entries, df, t_max)

    # Group by polymer
    poly_errors = {}
    for r in records:
        poly_errors.setdefault(r["polymer"], []).append(r)

    # Group by (polymer, solvent)
    poly_solv_errors = {}
    for r in records:
        key = (r["polymer"], r["solvent"])
        poly_solv_errors.setdefault(key, []).append(r["abs_error"])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # ── Panel A: MAE by polymer (horizontal bar) ──────────────────
    ax = axes[0, 0]
    polymers_sorted = sorted(poly_errors.keys(),
                             key=lambda p: np.mean([r["abs_error"] for r in poly_errors[p]]))
    maes = [np.mean([r["abs_error"] for r in poly_errors[p]]) for p in polymers_sorted]
    maxes = [np.max([r["abs_error"] for r in poly_errors[p]]) for p in polymers_sorted]
    colors = [HDPE_COLOR if p == "HDPE" else LDPE_COLOR if p == "LDPE" else OTHER_COLOR
              for p in polymers_sorted]

    y_pos = range(len(polymers_sorted))
    bars = ax.barh(y_pos, maes, height=0.6, color=colors, edgecolor="white", linewidth=0.5)

    # Max error markers
    ax.scatter(maxes, y_pos, marker="|", color="black", s=30, zorder=5, linewidths=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(polymers_sorted)
    ax.set_xlabel("Mean absolute error (%)")
    ax.set_title("(a) MAE by polymer (T \u2264 130\u00b0C)", fontsize=PUB_FONTSIZE, loc="left")

    # Annotate MAE values
    for i, (mae, mx) in enumerate(zip(maes, maxes)):
        ax.text(mae + 0.03, i, f"{mae:.2f}%", va="center", fontsize=PUB_FONTSIZE - 2)
        ax.text(mx + 0.03, i, f"max {mx:.1f}%", va="center",
                fontsize=PUB_FONTSIZE - 2, color=GREY)

    ax.legend(handles=[
        mpatches.Patch(fc=HDPE_COLOR, label="HDPE"),
        mpatches.Patch(fc=LDPE_COLOR, label="LDPE"),
        mpatches.Patch(fc=OTHER_COLOR, label="Other"),
    ], fontsize=PUB_FONTSIZE - 2, loc="lower right", frameon=True,
       facecolor="white", edgecolor="none")

    # ── Panel B: Error distribution (box + strip) ────────────────
    ax = axes[0, 1]

    hdpe_errs = np.array([r["abs_error"] for r in poly_errors.get("HDPE", [])])
    ldpe_errs = np.array([r["abs_error"] for r in poly_errors.get("LDPE", [])])
    other_errs = np.array([r["abs_error"] for p, recs in poly_errors.items()
                           if p not in ("HDPE", "LDPE") for r in recs])

    bp = ax.boxplot(
        [other_errs, ldpe_errs, hdpe_errs],
        vert=True, widths=0.5, patch_artist=True,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    box_colors = [OTHER_COLOR, LDPE_COLOR, HDPE_COLOR]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(["Other\n(9 polymers)", "LDPE", "HDPE"])
    ax.set_ylabel("Absolute error (%)")
    ax.set_title("(b) Error distribution", fontsize=PUB_FONTSIZE, loc="left")

    # Annotate medians and 95th percentiles
    for i, (errs, label) in enumerate([(other_errs, "Other"), (ldpe_errs, "LDPE"), (hdpe_errs, "HDPE")]):
        med = np.median(errs)
        p95 = np.percentile(errs, 95)
        ax.text(i + 1.3, med, f"med={med:.3f}%", va="center",
                fontsize=PUB_FONTSIZE - 2, color=GREY)
        ax.text(i + 1.3, p95, f"p95={p95:.1f}%", va="center",
                fontsize=PUB_FONTSIZE - 2, color=GREY)

    # ── Panel C: Worst solvents for HDPE & LDPE ──────────────────
    ax = axes[1, 0]

    # Top 8 worst solvents across HDPE + LDPE combined
    combined = {}
    for (poly, solv), aerrs in poly_solv_errors.items():
        if poly in ("HDPE", "LDPE"):
            combined.setdefault(solv, {"HDPE": [], "LDPE": []})
            combined[solv][poly] = aerrs

    # Rank by max of (HDPE MAE, LDPE MAE)
    solv_rank = sorted(combined.keys(),
                       key=lambda s: max(
                           np.mean(combined[s]["HDPE"]) if combined[s]["HDPE"] else 0,
                           np.mean(combined[s]["LDPE"]) if combined[s]["LDPE"] else 0),
                       reverse=True)[:10]

    y_pos = range(len(solv_rank))
    hdpe_vals = [np.mean(combined[s]["HDPE"]) if combined[s]["HDPE"] else 0 for s in solv_rank]
    ldpe_vals = [np.mean(combined[s]["LDPE"]) if combined[s]["LDPE"] else 0 for s in solv_rank]

    bar_h = 0.35
    ax.barh([y - bar_h / 2 for y in y_pos], hdpe_vals, height=bar_h,
            color=HDPE_COLOR, label="HDPE", edgecolor="white", linewidth=0.3)
    ax.barh([y + bar_h / 2 for y in y_pos], ldpe_vals, height=bar_h,
            color=LDPE_COLOR, label="LDPE", edgecolor="white", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(solv_rank, fontsize=PUB_FONTSIZE - 1)
    ax.set_xlabel("Mean absolute error (%)")
    ax.set_title("(c) Worst solvents for HDPE/LDPE", fontsize=PUB_FONTSIZE, loc="left")
    ax.legend(fontsize=PUB_FONTSIZE - 2, loc="lower right", frameon=True,
              facecolor="white", edgecolor="none")
    ax.invert_yaxis()

    # ── Panel D: Cumulative error distribution ────────────────────
    ax = axes[1, 1]

    for errs, color, label in [
        (other_errs, OTHER_COLOR, "Other (9 polymers)"),
        (ldpe_errs, LDPE_COLOR, "LDPE"),
        (hdpe_errs, HDPE_COLOR, "HDPE"),
    ]:
        sorted_e = np.sort(errs)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e) * 100
        ax.plot(sorted_e, cdf, "-", color=color, linewidth=1.2, label=label)

    # Reference lines
    for thresh in [1, 2, 5]:
        ax.axvline(thresh, color=GREY, linewidth=0.5, linestyle=":", alpha=0.5)
        ax.text(thresh + 0.1, 5, f"{thresh}%", fontsize=PUB_FONTSIZE - 2,
                color=GREY, va="bottom")

    ax.axhline(90, color=GREY, linewidth=0.4, linestyle="--", alpha=0.4)
    ax.axhline(95, color=GREY, linewidth=0.4, linestyle="--", alpha=0.4)
    ax.text(0.1, 90.5, "90th", fontsize=PUB_FONTSIZE - 2, color=GREY)
    ax.text(0.1, 95.5, "95th", fontsize=PUB_FONTSIZE - 2, color=GREY)

    ax.set_xlabel("Absolute error (%)")
    ax.set_ylabel("Cumulative % of data points")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 101)
    ax.set_title("(d) Cumulative error distribution", fontsize=PUB_FONTSIZE, loc="left")
    ax.legend(fontsize=PUB_FONTSIZE - 2, loc="center right", frameon=True,
              facecolor="white", edgecolor="none")

    fig.tight_layout()
    out = HERE / "interpolation_error_analysis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    with open(COEFF_PATH) as f:
        entries = json.load(f)["entries"]
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    plot_error_analysis(entries, df, t_max=130)
