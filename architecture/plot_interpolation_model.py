"""Visualize the solubility interpolation model: ln(S) = A + B/T + C·ln(T) (modified Apelblat).

Generates 5 figures:
1. Representative solubility curves (predicted vs raw data)
2. R² distribution across all 328 fitted entries
3. Coverage heatmap (polymer × solvent, by category)
4. Predicted vs actual parity plot
5. Grouped panels — one subplot per polymer, all solvent curves
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
COEFF_PATH = DATA / "solubility_coefficients.json"
CSV_PATH = DATA / "COMMON-SOLVENTS-DATABASE.csv"

# ── Publication style ─────────────────────────────────────────────
PUB_FONT = "Liberation Sans"
PUB_FONTSIZE = 8
PUB_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
    "#882255",  # wine
    "#44AA99",  # teal
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # forest
    "#AA4499",  # magenta
    "#88CCEE",  # light blue
    "#999933",  # olive
    "#661100",  # dark red
    "#6699CC",  # steel blue
    "#DDDDDD",  # light grey
    "#CC6677",  # rose
    "#225522",  # dark green
    "#AA7744",  # tan
    "#774411",  # brown
    "#AADDCC",  # mint
    "#775566",  # mauve
    "#66CCEE",  # cyan
    "#EE6677",  # salmon
    "#AA3377",  # purple
    "#BBCC33",  # pear
    "#99DDFF",  # ice blue
    "#44BB99",  # jade
    "#FFDD44",  # gold
]


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


# ── Load data ─────────────────────────────────────────────────────

def load_coefficients():
    with open(COEFF_PATH) as f:
        data = json.load(f)
    return data["entries"]


def load_raw_csv():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def predict_curve(entry, temps_c):
    """Predict solubility for an array of temperatures."""
    t_k = np.array(temps_c) + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] * np.log(t_k)
    return np.clip(np.exp(ln_s), 0.0, 100.0)


# ==================================================================
# Figure 1: Representative solubility curves (predicted vs raw)
# ==================================================================

def plot_representative_curves(entries, df):
    """6 representative polymer-solvent pairs showing fit vs raw data."""
    apply_pub_style()

    # Pick diverse examples
    targets = [
        ("HDPE", "dodecane"),
        ("PS", "toluene"),
        ("PVC", "thf"),
        ("EVOH", "propanol"),
        ("PET", "dimethylformamide"),
        ("Nylon6", "glycol"),
    ]

    lookup = {(e["polymer"].upper(), e["solvent"].lower()): e for e in entries}

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    axes = axes.flatten()

    for i, (poly, solv) in enumerate(targets):
        ax = axes[i]
        key = (poly.upper(), solv.lower())
        entry = lookup.get(key)
        if entry is None:
            ax.text(0.5, 0.5, f"No fit:\n{poly}/{solv}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)
            continue

        # Raw data
        mask = ((df["Polymer"].str.upper() == poly.upper()) &
                (df["Solvent"].str.lower() == solv.lower()))
        raw = df[mask].copy()
        raw = raw[raw["Solubility (%)"] < 100.0]  # exclude 100% artifacts
        raw = raw.sort_values("Temperature (°C)")

        # Model curve
        t_min = entry["t_min_c"]
        t_max = entry["t_max_c"]
        temps = np.linspace(t_min, t_max, 200)
        preds = predict_curve(entry, temps)

        ax.plot(temps, preds, "-", color=PUB_COLORS[0], linewidth=1.2,
                label="Model", zorder=3)
        if len(raw) > 0:
            ax.scatter(raw["Temperature (°C)"], raw["Solubility (%)"],
                       s=12, color=PUB_COLORS[1], zorder=4, label="Data",
                       edgecolors="none", alpha=0.8)

        ax.set_xlabel("Temperature (\u00b0C)")
        ax.set_ylabel("Solubility (%)")
        ax.set_title(f"{poly} in {solv}", fontsize=PUB_FONTSIZE)

        # R² annotation
        r2 = entry.get("r_squared", 0)
        ax.text(0.97, 0.05, f"R\u00b2 = {r2:.4f}\nn = {entry['n_points']}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=PUB_FONTSIZE - 1,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc",
                          alpha=0.9, lw=0.5))

        if i == 0:
            ax.legend(fontsize=PUB_FONTSIZE - 1, loc="upper left",
                      frameon=True, facecolor="white", edgecolor="none")

    fig.tight_layout()
    out = HERE / "interpolation_representative_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==================================================================
# Figure 2: R² distribution
# ==================================================================

def plot_r2_distribution(entries):
    apply_pub_style()

    fitted = [e for e in entries if e["category"] == "fitted"]
    r2_vals = [e["r_squared"] for e in fitted]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    bins = np.arange(0.90, 1.001, 0.005)
    counts, edges, patches = ax.hist(r2_vals, bins=bins, color=PUB_COLORS[0],
                                      edgecolor="white", linewidth=0.5, alpha=0.85)

    # Color bins by quality
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge >= 0.999:
            patch.set_facecolor(PUB_COLORS[2])  # green — excellent
        elif left_edge >= 0.99:
            patch.set_facecolor(PUB_COLORS[0])  # blue — good
        elif left_edge >= 0.95:
            patch.set_facecolor(PUB_COLORS[6])  # orange — acceptable
        else:
            patch.set_facecolor(PUB_COLORS[1])  # red — poor

    ax.set_xlabel("R\u00b2")
    ax.set_ylabel("Count")
    ax.set_xlim(0.90, 1.001)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Stats annotation
    r2_arr = np.array(r2_vals)
    stats_text = (f"n = {len(r2_vals)}\n"
                  f"median = {np.median(r2_arr):.5f}\n"
                  f"mean = {np.mean(r2_arr):.5f}\n"
                  f"min = {np.min(r2_arr):.5f}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=PUB_FONTSIZE - 1,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      alpha=0.9, lw=0.5))

    # Legend
    legend_items = [
        mpatches.Patch(fc=PUB_COLORS[2], ec="white", label="R\u00b2 \u2265 0.999"),
        mpatches.Patch(fc=PUB_COLORS[0], ec="white", label="0.99 \u2264 R\u00b2 < 0.999"),
        mpatches.Patch(fc=PUB_COLORS[6], ec="white", label="0.95 \u2264 R\u00b2 < 0.99"),
        mpatches.Patch(fc=PUB_COLORS[1], ec="white", label="R\u00b2 < 0.95"),
    ]
    ax.legend(handles=legend_items, fontsize=PUB_FONTSIZE - 2, loc="upper right",
              frameon=True, facecolor="white", edgecolor="none")

    fig.tight_layout()
    out = HERE / "interpolation_r2_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==================================================================
# Figure 3: Coverage heatmap (polymer × solvent)
# ==================================================================

def plot_coverage_heatmap(entries):
    apply_pub_style()

    # Build polymer × solvent matrix
    polymers = sorted(set(e["polymer"] for e in entries))
    solvents = sorted(set(e["solvent"] for e in entries))
    lookup = {(e["polymer"], e["solvent"]): e for e in entries}

    # Category → numeric: 0=no data, 1=fitted, 2=anomalous, 3=insoluble
    cat_map = {"fitted": 1, "anomalous": 2, "insoluble": 3}
    matrix = np.zeros((len(polymers), len(solvents)))

    for i, p in enumerate(polymers):
        for j, s in enumerate(solvents):
            entry = lookup.get((p, s))
            if entry:
                matrix[i, j] = cat_map.get(entry["category"], 0)

    # Custom colormap: white=no data, blue=fitted, orange=anomalous, red=insoluble
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#f5f5f5", PUB_COLORS[0], PUB_COLORS[6], PUB_COLORS[1]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig_w = max(7.0, len(solvents) * 0.25 + 1.5)
    fig_h = max(3.0, len(polymers) * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(solvents)))
    ax.set_xticklabels(solvents, rotation=90, fontsize=PUB_FONTSIZE - 2)
    ax.set_yticks(range(len(polymers)))
    ax.set_yticklabels(polymers, fontsize=PUB_FONTSIZE - 1)

    # R² values inside fitted cells
    for i, p in enumerate(polymers):
        for j, s in enumerate(solvents):
            entry = lookup.get((p, s))
            if entry and entry["category"] == "fitted":
                r2 = entry["r_squared"]
                color = "white" if r2 < 0.99 else "#cccccc"
                ax.text(j, i, f".{str(round(r2, 3)).split('.')[1][:3]}",
                        ha="center", va="center", fontsize=4, color=color)

    # Legend
    legend_items = [
        mpatches.Patch(fc="#f5f5f5", ec="#cccccc", label="No data"),
        mpatches.Patch(fc=PUB_COLORS[0], ec="none", label=f"Fitted ({sum(1 for e in entries if e['category']=='fitted')})"),
        mpatches.Patch(fc=PUB_COLORS[6], ec="none", label=f"Anomalous ({sum(1 for e in entries if e['category']=='anomalous')})"),
        mpatches.Patch(fc=PUB_COLORS[1], ec="none", label=f"Insoluble ({sum(1 for e in entries if e['category']=='insoluble')})"),
    ]
    ax.legend(handles=legend_items, fontsize=PUB_FONTSIZE - 1, loc="upper right",
              bbox_to_anchor=(1.0, -0.15), ncol=4, frameon=False)

    fig.tight_layout()
    out = HERE / "interpolation_coverage_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==================================================================
# Figure 4: Predicted vs Actual parity plot
# ==================================================================

def plot_parity(entries, df, t_max=None):
    """Scatter all raw data vs model prediction — colored by polymer."""
    apply_pub_style()

    lookup = {(e["polymer"].upper(), e["solvent"].lower()): e for e in entries
              if e["category"] == "fitted"}

    actual_all = []
    pred_all = []
    poly_labels = []

    polymers_order = sorted(set(e["polymer"] for e in entries))
    poly_color_map = {p: PUB_COLORS[i % len(PUB_COLORS)] for i, p in enumerate(polymers_order)}

    for _, row in df.iterrows():
        poly = row["Polymer"].strip().upper()
        solv = row["Solvent"].strip().lower()
        sol = row["Solubility (%)"]
        temp = row["Temperature (°C)"]

        if sol >= 100.0:
            continue  # skip artifacts
        if t_max is not None and temp > t_max:
            continue

        key = (poly, solv)
        entry = lookup.get(key)
        if entry is None:
            continue

        pred = predict_curve(entry, [temp])[0]
        actual_all.append(sol)
        pred_all.append(pred)
        poly_labels.append(poly)

    actual_arr = np.array(actual_all)
    pred_arr = np.array(pred_all)

    # Filter out zero/negative for log space
    valid_log = (actual_arr > 0) & (pred_arr > 0)
    ln_actual = np.log(actual_arr[valid_log])
    ln_pred = np.log(pred_arr[valid_log])
    poly_labels_arr = np.array(poly_labels)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # ── Left panel: linear space ──
    for poly in polymers_order:
        mask = np.array([p == poly for p in poly_labels])
        if mask.sum() == 0:
            continue
        ax_lin.scatter(actual_arr[mask], pred_arr[mask], s=4, alpha=0.5,
                       color=poly_color_map[poly], label=poly, edgecolors="none")

    ax_lin.plot([0, 100], [0, 100], "--", color="#999999", linewidth=0.8, zorder=1)
    ax_lin.set_xlabel("Actual solubility (%)")
    ax_lin.set_ylabel("Predicted solubility (%)")
    ax_lin.set_xlim(-2, 102)
    ax_lin.set_ylim(-2, 102)
    ax_lin.set_aspect("equal")

    ss_res = np.sum((actual_arr - pred_arr) ** 2)
    ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
    r2_lin = 1 - ss_res / ss_tot
    rmse_lin = np.sqrt(np.mean((actual_arr - pred_arr) ** 2))
    mae_lin = np.mean(np.abs(actual_arr - pred_arr))

    t_label = f"\nT \u2264 {t_max}\u00b0C" if t_max else ""
    ax_lin.text(0.03, 0.97,
                f"Linear space{t_label}\nn = {len(actual_arr):,}\n"
                f"R\u00b2 = {r2_lin:.5f}\n"
                f"RMSE = {rmse_lin:.2f}%\nMAE = {mae_lin:.2f}%",
                transform=ax_lin.transAxes, ha="left", va="top",
                fontsize=PUB_FONTSIZE - 1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          alpha=0.9, lw=0.5))

    ax_lin.legend(fontsize=PUB_FONTSIZE - 2, loc="lower right",
                  frameon=True, facecolor="white", edgecolor="none",
                  markerscale=2, ncol=2)

    # ── Right panel: log space ──
    for poly in polymers_order:
        mask_full = np.array([p == poly for p in poly_labels])
        mask = mask_full[valid_log]
        if mask.sum() == 0:
            continue
        ax_log.scatter(ln_actual[mask], ln_pred[mask], s=4, alpha=0.5,
                       color=poly_color_map[poly], label=poly, edgecolors="none")

    lo = min(ln_actual.min(), ln_pred.min()) - 0.3
    hi = max(ln_actual.max(), ln_pred.max()) + 0.3
    ax_log.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=0.8, zorder=1)
    ax_log.set_xlabel("Actual ln(S%)")
    ax_log.set_ylabel("Predicted ln(S%)")
    ax_log.set_xlim(lo, hi)
    ax_log.set_ylim(lo, hi)
    ax_log.set_aspect("equal")

    ss_res_log = np.sum((ln_actual - ln_pred) ** 2)
    ss_tot_log = np.sum((ln_actual - np.mean(ln_actual)) ** 2)
    r2_log = 1 - ss_res_log / ss_tot_log
    rmse_log = np.sqrt(np.mean((ln_actual - ln_pred) ** 2))

    ax_log.text(0.03, 0.97,
                f"Log space (native){t_label}\nn = {valid_log.sum():,}\n"
                f"R\u00b2 = {r2_log:.5f}\nRMSE = {rmse_log:.4f}",
                transform=ax_log.transAxes, ha="left", va="top",
                fontsize=PUB_FONTSIZE - 1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          alpha=0.9, lw=0.5))

    ax_log.legend(fontsize=PUB_FONTSIZE - 2, loc="lower right",
                  frameon=True, facecolor="white", edgecolor="none",
                  markerscale=2, ncol=2)

    fig.tight_layout()
    suffix = f"_{t_max}C" if t_max else ""
    out = HERE / f"interpolation_parity_plot{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==================================================================
# Figure 5: Grouped panels — one subplot per polymer, all solvents
# ==================================================================

def plot_polymer_panels(entries, df, t_max=None):
    """One subplot per polymer showing ALL solvent curves with raw data."""
    apply_pub_style()

    fitted = [e for e in entries if e["category"] == "fitted"]
    polymers = sorted(set(e["polymer"] for e in fitted))
    n_poly = len(polymers)

    # Layout: try to be roughly square
    ncols = 4 if n_poly > 6 else 3
    nrows = int(np.ceil(n_poly / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.4))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, poly in enumerate(polymers):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Get all solvents for this polymer
        poly_entries = sorted(
            [e for e in fitted if e["polymer"] == poly],
            key=lambda e: e["solvent"]
        )

        for j, entry in enumerate(poly_entries):
            solv = entry["solvent"]
            color = PUB_COLORS[j % len(PUB_COLORS)]

            # Model curve
            t_hi = min(entry["t_max_c"], t_max) if t_max else entry["t_max_c"]
            if t_hi <= entry["t_min_c"]:
                continue
            temps = np.linspace(entry["t_min_c"], t_hi, 100)
            preds = predict_curve(entry, temps)
            ax.plot(temps, preds, "-", color=color, linewidth=0.8, alpha=0.85)

            # Raw data points
            mask = ((df["Polymer"].str.strip().str.upper() == poly.upper()) &
                    (df["Solvent"].str.strip().str.lower() == solv.lower()) &
                    (df["Solubility (%)"] < 100.0))
            if t_max:
                mask = mask & (df["Temperature (°C)"] <= t_max)
            raw = df[mask]
            if len(raw) > 0:
                ax.scatter(raw["Temperature (°C)"], raw["Solubility (%)"],
                           s=3, color=color, alpha=0.4, edgecolors="none")

        t_label = f", T\u2264{t_max}\u00b0C" if t_max else ""
        ax.set_title(f"{poly} ({len(poly_entries)} solvents{t_label})",
                     fontsize=PUB_FONTSIZE, fontweight="bold")
        ax.set_xlabel("Temperature (\u00b0C)")
        ax.set_ylabel("Solubility (%)")
        ax.set_ylim(-5, 105)

    # Hide unused axes
    for idx in range(n_poly, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    suffix = f"_{t_max}C" if t_max else ""
    out = HERE / f"interpolation_polymer_panels{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==================================================================
# Figure 5b: Filtered panels — only fits with R² < threshold
# ==================================================================

def plot_polymer_panels_filtered(entries, df, r2_threshold=0.998):
    """One subplot per polymer showing only solvent fits below R² threshold."""
    apply_pub_style()

    fitted = [e for e in entries if e["category"] == "fitted"]
    weak = [e for e in fitted if e["r_squared"] < r2_threshold]

    # Only polymers that have at least one weak fit
    polymers = sorted(set(e["polymer"] for e in weak))
    n_poly = len(polymers)
    if n_poly == 0:
        print(f"  No fits below R² < {r2_threshold}")
        return

    ncols = 3 if n_poly <= 6 else 4
    nrows = int(np.ceil(n_poly / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8),
                             squeeze=False)

    for idx, poly in enumerate(polymers):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Get weak fits for this polymer, sorted worst-first
        poly_entries = sorted(
            [e for e in weak if e["polymer"] == poly],
            key=lambda e: e["r_squared"]
        )

        for j, entry in enumerate(poly_entries):
            solv = entry["solvent"]
            r2 = entry["r_squared"]
            color = PUB_COLORS[j % len(PUB_COLORS)]

            # Model curve
            temps = np.linspace(entry["t_min_c"], entry["t_max_c"], 100)
            preds = predict_curve(entry, temps)
            ax.plot(temps, preds, "-", color=color, linewidth=1.0, alpha=0.9,
                    label=f"{solv} ({r2:.4f})")

            # Raw data points
            mask = ((df["Polymer"].str.strip().str.upper() == poly.upper()) &
                    (df["Solvent"].str.strip().str.lower() == solv.lower()) &
                    (df["Solubility (%)"] < 100.0))
            raw = df[mask]
            if len(raw) > 0:
                ax.scatter(raw["Temperature (°C)"], raw["Solubility (%)"],
                           s=8, color=color, alpha=0.6, edgecolors="none",
                           zorder=4)

        ax.set_title(f"{poly} ({len(poly_entries)} solvents with R\u00b2 < {r2_threshold})",
                     fontsize=PUB_FONTSIZE)
        ax.set_xlabel("Temperature (\u00b0C)")
        ax.set_ylabel("Solubility (%)")
        ax.set_ylim(-5, 105)

        # Legend — show solvent names + R² values
        n_show = min(len(poly_entries), 8)
        ax.legend(fontsize=PUB_FONTSIZE - 3, loc="upper left",
                  frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.85, ncol=1 if n_show <= 5 else 2,
                  handlelength=1.2, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.25)

    # Hide unused axes
    for idx in range(n_poly, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    out = HERE / "interpolation_polymer_panels_weak_fits.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==================================================================
# Run all
# ==================================================================

if __name__ == "__main__":
    print("Loading data...")
    entries = load_coefficients()
    df = load_raw_csv()

    fitted = [e for e in entries if e["category"] == "fitted"]
    anomalous = [e for e in entries if e["category"] == "anomalous"]
    insoluble = [e for e in entries if e["category"] == "insoluble"]
    print(f"  {len(entries)} total entries: {len(fitted)} fitted, "
          f"{len(anomalous)} anomalous, {len(insoluble)} insoluble")
    print(f"  {len(df)} raw data points")

    print("\n1. Representative curves...")
    plot_representative_curves(entries, df)

    print("\n2. R² distribution...")
    plot_r2_distribution(entries)

    print("\n3. Coverage heatmap...")
    plot_coverage_heatmap(entries)

    print("\n4. Parity plot (predicted vs actual)...")
    plot_parity(entries, df)

    print("\n5. Polymer panels (all solvents per polymer)...")
    plot_polymer_panels(entries, df)

    print("\n5b. Polymer panels (weak fits only, R² < 0.998)...")
    plot_polymer_panels_filtered(entries, df, r2_threshold=0.998)

    print("\nDone!")
