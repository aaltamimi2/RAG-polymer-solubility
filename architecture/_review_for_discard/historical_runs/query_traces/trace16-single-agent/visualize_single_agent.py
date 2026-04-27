#!/usr/bin/env python
"""Generate summary visualization for single-agent trace campaign.

Reads the results JSON produced by run_single_agent_traces.py and
generates a publication-quality summary figure showing all 10 queries.

Usage:
    python visualize_single_agent.py                          # latest results
    python visualize_single_agent.py results_20260224_*.json  # specific file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Liberation Sans", "Arial",
]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent.parent.parent
sys.path.insert(0, str(_ROOT / "architecture"))

from visualize_trace_cards import AGENT_COLORS

# ── Layout constants ─────────────────────────────────────────────────

FIG_W = 10.0
ROW_H = 0.55       # inches per query row
HEADER_H = 1.2     # inches for header
FOOTER_H = 0.5     # inches for footer
LEFT_LABEL = 0.02  # left edge for query name
BAR_LEFT = 0.26    # left edge for time bar
BAR_RIGHT = 0.62   # right edge for time bar
TOKEN_X = 0.64     # token count position
AGENTS_X = 0.76    # subagent pills start
ROUTE_X = 0.96     # routing check position

C_BG = "#FFFFFF"
C_TITLE = "#1E293B"
C_BODY = "#374151"
C_GRID = "#F1F5F9"
C_BAR_BG = "#E2E8F0"
C_CHECK = "#22C55E"
C_CROSS = "#EF4444"


def _find_latest_results(directory: Path) -> Path | None:
    """Find the most recent results_*.json in the directory."""
    candidates = sorted(directory.glob("results_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _agent_color(name: str) -> str:
    """Get color for a subagent, with fallback."""
    return AGENT_COLORS.get(name, "#6B7280")


def draw_summary(results: list[dict], output_path: str, campaign: str = "Single Agent"):
    """Draw the 10-row summary figure."""
    n = len(results)
    fig_h = HEADER_H + n * ROW_H + FOOTER_H
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    # ── Aggregate stats ──
    total_time = sum(r.get("wall_time_s", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    n_ok = sum(1 for r in results if r.get("routing_match", False))
    max_time = max((r.get("wall_time_s", 1) for r in results), default=1)

    # ── Header ──
    header_y = 1.0 - HEADER_H / fig_h
    ax.text(0.50, 1.0 - 0.04, f"DISSOLVE Trace Campaign: {campaign} Execution",
            ha="center", fontsize=13, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    ax.text(0.50, 1.0 - 0.08,
            f"{n} queries  |  {total_time:.0f}s total  |  "
            f"{total_tokens:,} tokens  |  {n_ok}/{n} routing match",
            ha="center", fontsize=9, color=C_BODY,
            transform=ax.transAxes)

    # Column headers
    col_y = header_y + 0.01
    ax.text(LEFT_LABEL, col_y, "Query", fontsize=7, fontweight="bold",
            color=C_BODY, transform=ax.transAxes, va="center")
    ax.text((BAR_LEFT + BAR_RIGHT) / 2, col_y, "Wall Time",
            ha="center", fontsize=7, fontweight="bold", color=C_BODY,
            transform=ax.transAxes, va="center")
    ax.text(TOKEN_X, col_y, "Tokens", fontsize=7, fontweight="bold",
            color=C_BODY, transform=ax.transAxes, va="center")
    ax.text(AGENTS_X, col_y, "Subagent(s)", fontsize=7, fontweight="bold",
            color=C_BODY, transform=ax.transAxes, va="center")
    ax.text(ROUTE_X, col_y, "Route", fontsize=7, fontweight="bold",
            color=C_BODY, ha="center", transform=ax.transAxes, va="center")

    # ── Rows ──
    for i, r in enumerate(results):
        # Row vertical center (top-down)
        row_frac = ROW_H / fig_h
        row_y = header_y - (i + 0.5) * row_frac

        # Alternating background
        if i % 2 == 0:
            stripe_y = header_y - (i + 1) * row_frac
            ax.add_patch(Rectangle(
                (0, stripe_y), 1, row_frac,
                facecolor=C_GRID, edgecolor="none",
                transform=ax.transAxes, zorder=0,
            ))

        # Query name
        name = r.get("name", f"query-{i}")
        ax.text(LEFT_LABEL, row_y, name,
                fontsize=7, color=C_TITLE, fontweight="bold",
                transform=ax.transAxes, va="center")

        # Wall-time bar
        wt = r.get("wall_time_s", 0)
        bar_w_max = BAR_RIGHT - BAR_LEFT
        bar_w = (wt / max_time) * bar_w_max if max_time > 0 else 0
        bar_h_frac = 0.6 * row_frac

        # Determine bar color from first subagent
        actual = r.get("actual_subagents", [])
        bar_color = _agent_color(actual[0]) if actual else "#94A3B8"

        ax.add_patch(FancyBboxPatch(
            (BAR_LEFT, row_y - bar_h_frac / 2), bar_w, bar_h_frac,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=bar_color, edgecolor="none", alpha=0.80,
            transform=ax.transAxes, zorder=2,
        ))
        # Time label
        ax.text(BAR_LEFT + bar_w + 0.005, row_y, f"{wt:.0f}s",
                fontsize=6, color=C_BODY,
                transform=ax.transAxes, va="center")

        # Token count
        tokens = r.get("total_tokens", 0)
        ax.text(TOKEN_X, row_y, f"{tokens:,}",
                fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="center")

        # Subagent pills
        px = AGENTS_X
        for sa in actual:
            color = _agent_color(sa)
            short = sa.replace("-engineer", "").replace("-analyst", "")
            short = short.replace("-specialist", "").replace("-researcher", "")
            tw = len(short) * 0.0055 + 0.012
            pill = FancyBboxPatch(
                (px, row_y - 0.008), tw, 0.016,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=color, edgecolor="none", alpha=0.85,
                transform=ax.transAxes, zorder=3,
            )
            ax.add_patch(pill)
            ax.text(px + tw / 2, row_y, short,
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)
            px += tw + 0.005

        # Routing match
        ok = r.get("routing_match", False)
        ax.text(ROUTE_X, row_y,
                "\u2713" if ok else "\u2717",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=C_CHECK if ok else C_CROSS,
                transform=ax.transAxes, zorder=3)

    # ── Footer ──
    footer_y = FOOTER_H / fig_h * 0.4
    ax.text(0.50, footer_y,
            f"DISSOLVE v8  |  {campaign} Execution  |  "
            f"Gemini 2.5 Pro + Flash  |  2026-02-24",
            ha="center", fontsize=7, color=C_BODY,
            transform=ax.transAxes)

    fig.savefig(output_path, dpi=200, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize single-agent trace campaign results"
    )
    parser.add_argument("results_json", nargs="?", default=None,
                        help="Results JSON file (default: latest)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    if args.results_json:
        results_path = Path(args.results_json)
    else:
        results_path = _find_latest_results(_DIR)
        if not results_path:
            print("No results_*.json found. Run the trace campaign first.")
            return

    print(f"Reading: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in JSON.")
        return

    output = args.output or str(_DIR / "summary_single_agent.png")
    campaign = data.get("campaign", "Single Agent")
    draw_summary(results, output, campaign=campaign.replace("-", " ").title())


if __name__ == "__main__":
    main()
