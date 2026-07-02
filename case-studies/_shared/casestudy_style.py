"""Shared plotting style for v10 case-study figures.

One place to define the publication palette and matplotlib rcParams so every
case study renders with a consistent, polished look. Import and call
``apply_style()`` at the top of a case-study ``reproduce.py``.
"""

from __future__ import annotations

import matplotlib as mpl

# Colour-blind-safe qualitative palette (Okabe-Ito), extended for up to 8 series.
SERIES_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]
INK = "#1a1a2e"
FRONTIER = "#2c3e50"
ACCENT_OPTIMAL = "#d62728"   # DP-optimal marker
ACCENT_CHEAPEST = "#2ca02c"  # cheapest marker
ACCENT_KNEE = "#ff7f0e"      # knee marker
FILL = "#0072B2"


def apply_style() -> None:
    """Apply the shared rcParams. Idempotent."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def caption(fig, text: str) -> None:
    """Add a small methods/source caption along the bottom of a figure."""
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7.5,
             color="#666666", wrap=True)
