#!/usr/bin/env python
"""Multi-Polymer Sequence Pareto — Publication-quality figures.

Reads the sequence Pareto results JSON from run_multi_polymer_pareto.py
and generates figures:

  Fig 1 — Sequence Pareto Front (Avg MSP vs Avg GWP per sequence)
  Fig 2 — Per-Step Breakdown Table (PNG + CSV)
  Fig 3 — Step-by-Step Comparison Bar Chart

Usage:
    python multi_polymer_figures.py \
        --results sequence_pareto.json \
        --output-dir ./figures
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Path setup ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ARCH_DIR = _THIS_DIR.parent.parent
_ROOT_DIR = _ARCH_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / "src"))
_V8_SRC = Path("/home/aaltamimi2/langchain-STRAP-v8/src")
if _V8_SRC.is_dir():
    sys.path.insert(0, str(_V8_SRC))

# Reuse constants from trace20
_T20_DIR = _THIS_DIR.parent / "trace20-cs2-pareto"
sys.path.insert(0, str(_T20_DIR))
from cs2_pareto_figures import (  # noqa: E402
    PARETO_GOLD,
    DOMINATED_GRAY,
    compute_pareto_front,
    _is_finite,
)

# ── Publication style ────────────────────────────────────────────────
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)

# Sequence colors (distinct per sequence)
SEQ_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
    "#e67e22", "#1abc9c", "#e84393", "#34495e",
    "#f39c12", "#27ae60", "#8e44ad", "#2c3e50",
]


def _abbreviate_solvent(name: str) -> str:
    """Short label for solvent names."""
    abbrevs = {
        "Toluene": "Tol", "Dodecane": "Dod",
        "Ethylene Glycol": "EG", "Propylene Glycol": "PG",
        "Diethylene glycol": "DEG", "Dimethyl sulfoxide": "DMSO",
        "N,N-Dimethylformamide": "DMF", "Diphenyl ether": "DPE",
        "Ethanol": "EtOH", "Isopropanol": "iPrOH",
        "Ethyl Acetate": "EtOAc", "Acetonitrile": "ACN",
        "Cyclohexanol": "CyOH", "Cyclohexanone": "CyO",
        "o-Xylene": "oXyl", "p-Xylene": "pXyl",
        "Oleyl Alcohol": "OleOH", "n-Heptane": "Hep",
        "Propylene Carbonate": "PC", "2,3-Dihydropyran": "DHP",
    }
    return abbrevs.get(name, name[:5])


# ── Figure 1: Sequence Pareto Front ─────────────────────────────────

def fig1_sequence_pareto(data: dict, output_dir: Path):
    """Scatter: Avg MSP vs Avg GWP per sequence, with Pareto front."""
    sequences = data["sequences"]
    valid = [s for s in sequences
             if s.get("all_succeeded") and s.get("avg_msp") is not None]

    if not valid:
        print("  Fig 1 skipped: no valid sequences")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # All sequences
    all_msp = [s["avg_msp"] for s in valid]
    all_gwp = [s["avg_gwp"] for s in valid]

    # Non-Pareto as gray
    non_pareto = [s for s in valid if not s.get("is_pareto")]
    if non_pareto:
        np_msp = [s["avg_msp"] for s in non_pareto]
        np_gwp = [s["avg_gwp"] for s in non_pareto]
        ax.scatter(np_gwp, np_msp, c=DOMINATED_GRAY, s=60, alpha=0.5,
                   edgecolors="white", linewidths=0.4, zorder=2,
                   label=f"Non-Pareto (n={len(non_pareto)})")

    # Pareto-optimal
    pareto = [s for s in valid if s.get("is_pareto")]
    if pareto:
        p_sorted = sorted(pareto, key=lambda s: s["avg_gwp"])
        p_msp = [s["avg_msp"] for s in p_sorted]
        p_gwp = [s["avg_gwp"] for s in p_sorted]

        # Pareto line
        ax.plot(p_gwp, p_msp, "--", color=PARETO_GOLD, linewidth=2.5, zorder=4,
                label="Pareto front")
        ax.scatter(p_gwp, p_msp, c=PARETO_GOLD, s=180, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=5,
                   label=f"Pareto-optimal (n={len(pareto)})")

        # Annotate
        for i, s in enumerate(p_sorted):
            # Short label: first 30 chars
            label = s["label"][:30]
            offset_y = 10 if i % 2 == 0 else -14
            ax.annotate(
                label, (s["avg_gwp"], s["avg_msp"]),
                textcoords="offset points", xytext=(12, offset_y),
                fontsize=7, fontweight="bold", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
            )

    # Annotate non-Pareto too (if few enough)
    if len(non_pareto) <= 10:
        for s in non_pareto:
            label = s["label"][:25]
            ax.annotate(
                label, (s["avg_gwp"], s["avg_msp"]),
                textcoords="offset points", xytext=(8, -8),
                fontsize=6, color="#777777",
            )

    ax.set_xlabel("Average GWP per Step (kg CO₂-eq / kg polymer)")
    ax.set_ylabel("Average MSP per Step (USD / kg polymer)")
    ax.set_title("Separation Sequence Pareto Front\n(Avg MSP vs Avg GWP)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig1_sequence_pareto.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 1 saved: {path}")


# ── Figure 2: Per-Step Breakdown Table ───────────────────────────────

def fig2_step_table(data: dict, output_dir: Path):
    """Render per-step breakdown for all sequences as PNG table + CSV."""
    sequences = data["sequences"]
    valid = [s for s in sequences
             if s.get("all_succeeded") and s.get("avg_msp") is not None]

    if not valid:
        print("  Fig 2 skipped: no valid sequences")
        return

    # Sort by avg_msp
    sorted_seqs = sorted(valid, key=lambda s: s["avg_msp"])

    # Build table: columns = Rank, Label, Step 1 (polymer/solvent/temp/MSP), ..., Total, Pareto
    max_steps = max(len(s["steps"]) for s in sorted_seqs)

    col_labels = ["Rank", "Sequence"]
    for i in range(max_steps):
        col_labels.append(f"Step {i+1}")
    col_labels.extend(["Avg MSP", "Avg GWP", "Pareto?"])

    table_data = []
    csv_rows = []
    for rank, s in enumerate(sorted_seqs, 1):
        row = [str(rank), s["label"][:25]]
        csv_row = {
            "Rank": rank,
            "Label": s["label"],
            "avg_msp": s["avg_msp"],
            "avg_gwp": s["avg_gwp"],
            "total_msp": s["total_msp"],
            "total_gwp": s["total_gwp"],
            "is_pareto": s.get("is_pareto", False),
        }

        for i in range(max_steps):
            if i < len(s["steps"]):
                step = s["steps"][i]
                solv_abbr = _abbreviate_solvent(step.get("solvent_bst", step.get("solvent_interp", "?")))
                cell = (f"{step['polymer']}\n{solv_abbr}\n"
                        f"{step['temperature_c']:.0f}°C\n"
                        f"${step.get('msp', 0):.3f}")
                row.append(cell)
                csv_row[f"step{i+1}_polymer"] = step["polymer"]
                csv_row[f"step{i+1}_solvent"] = step.get("solvent_bst", "")
                csv_row[f"step{i+1}_temp_c"] = step["temperature_c"]
                csv_row[f"step{i+1}_msp"] = step.get("msp")
                csv_row[f"step{i+1}_gwp"] = step.get("gwp")
            else:
                row.append("—")

        row.append(f"${s['avg_msp']:.3f}")
        row.append(f"{s['avg_gwp']:.2f}")
        row.append("Yes" if s.get("is_pareto") else "")

        table_data.append(row)
        csv_rows.append(csv_row)

    # Render PNG
    n_cols = len(col_labels)
    fig_width = max(16, n_cols * 1.6)
    fig_height = max(6, len(table_data) * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.set_title("Separation Sequence Step Breakdown",
                 fontsize=13, fontweight="bold", pad=20)

    cell_colors = []
    for row in table_data:
        if row[-1] == "Yes":
            cell_colors.append(["#FFF8DC"] * n_cols)
        else:
            cell_colors.append(["#FFFFFF"] * n_cols)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.0)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    path_png = output_dir / "fig2_step_table.png"
    fig.savefig(path_png)
    plt.close(fig)
    print(f"  Fig 2 (PNG) saved: {path_png}")

    # Save CSV
    if csv_rows:
        path_csv = output_dir / "fig2_step_table.csv"
        fieldnames = list(csv_rows[0].keys())
        with open(path_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Fig 2 (CSV) saved: {path_csv}")


# ── Figure 3: Step-by-Step MSP Comparison ────────────────────────────

def fig3_step_comparison(data: dict, output_dir: Path):
    """Grouped bar chart comparing per-step MSP across sequences."""
    sequences = data["sequences"]
    valid = [s for s in sequences
             if s.get("all_succeeded") and s.get("avg_msp") is not None]

    if not valid or len(valid) < 2:
        print("  Fig 3 skipped: need at least 2 valid sequences")
        return

    # Take top N sequences by avg_msp
    sorted_seqs = sorted(valid, key=lambda s: s["avg_msp"])[:8]
    max_steps = max(len(s["steps"]) for s in sorted_seqs)

    fig, ax = plt.subplots(figsize=(max(12, max_steps * 2), 7))

    bar_width = 0.8 / len(sorted_seqs)
    x = np.arange(max_steps)

    for si, seq in enumerate(sorted_seqs):
        msps = []
        for i in range(max_steps):
            if i < len(seq["steps"]) and seq["steps"][i].get("msp") is not None:
                msps.append(seq["steps"][i]["msp"])
            else:
                msps.append(0)

        offset = (si - len(sorted_seqs) / 2 + 0.5) * bar_width
        color = SEQ_COLORS[si % len(SEQ_COLORS)]
        label = seq["label"][:25]
        pareto_tag = " *" if seq.get("is_pareto") else ""
        ax.bar(x + offset, msps, bar_width * 0.9, color=color,
               label=f"{label}{pareto_tag}", alpha=0.85, edgecolor="white",
               linewidth=0.5)

    # X-axis: Step labels (polymer/solvent from first sequence)
    step_labels = []
    for i in range(max_steps):
        labels_at_step = set()
        for seq in sorted_seqs:
            if i < len(seq["steps"]):
                step = seq["steps"][i]
                poly = step["polymer"]
                solv = _abbreviate_solvent(step.get("solvent_bst", "?"))
                labels_at_step.add(f"{poly}")
        step_labels.append(f"Step {i+1}\n" + "/".join(sorted(labels_at_step)))

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("MSP (USD / kg polymer)")
    ax.set_title("Per-Step MSP Comparison Across Sequences",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "fig3_step_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 3 saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Polymer Sequence Pareto — Generate figures"
    )
    parser.add_argument(
        "--results", required=True, metavar="PATH",
        help="Path to sequence Pareto results JSON",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory for figures (default: this script's dir)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else _THIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Multi-Polymer Sequence Pareto — Figure Generation")
    print(f"  Output: {output_dir}\n")

    print("Generating Fig 1: Sequence Pareto Front...")
    fig1_sequence_pareto(data, output_dir)

    print("\nGenerating Fig 2: Step Breakdown Table...")
    fig2_step_table(data, output_dir)

    print("\nGenerating Fig 3: Step MSP Comparison...")
    fig3_step_comparison(data, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
