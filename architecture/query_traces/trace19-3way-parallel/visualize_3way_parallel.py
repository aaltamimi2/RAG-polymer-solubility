#!/usr/bin/env python
"""Generate summary visualization for 3-way parallel trace campaign.

Reads the results JSON produced by run_3way_parallel_traces.py and
generates a publication-quality summary figure showing all 10 queries
with fork/join arrows reflecting actual parallel vs sequential dispatch.

Usage:
    python visualize_3way_parallel.py                          # latest results
    python visualize_3way_parallel.py results_20260224_*.json  # specific file
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent.parent.parent
sys.path.insert(0, str(_ROOT / "architecture"))

from visualize_trace_cards import AGENT_COLORS

# ── Layout ────────────────────────────────────────────────────────────

FIG_W = 16.0
ROW_H = 0.85
HEADER_H = 1.4
FOOTER_H = 0.5

C_BG = "#FFFFFF"
C_TITLE = "#1E293B"
C_BODY = "#374151"
C_MUTED = "#9CA3AF"
C_GRID = "#F1F5F9"
C_CHECK = "#22C55E"
C_CROSS = "#EF4444"
C_ARROW = "#94A3B8"

# Short labels for pills
AGENT_SHORT = {
    "separation-engineer": "separation",
    "safety-analyst": "safety",
    "biosteam-analyst": "biosteam",
    "visualization-specialist": "visualization",
    "scholar-researcher": "scholar",
    "patent-researcher": "patent",
    "rag-analyst": "rag",
    "statistics-ml": "stats-ml",
}


def _agent_color(name: str) -> str:
    return AGENT_COLORS.get(name, "#6B7280")


def _find_latest_results(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("results_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _derive_execution_stages(tool_names: list[str], actual_subagents: list[str]) -> list[list[str]]:
    """Derive execution stages from the tool_names sequence.

    Returns a list of stages, where each stage is a list of agent names
    dispatched concurrently. E.g.:
      [["separation-engineer"], ["safety-analyst", "biosteam-analyst"]]
    means separation ran first, then safety + biosteam in parallel.
    """
    # Group consecutive 'task' calls into batches (parallel dispatches)
    task_batches: list[int] = []
    current_batch = 0
    prev_was_task = False
    for t in tool_names:
        if t == "task":
            if prev_was_task:
                current_batch += 1
            else:
                if current_batch > 0:
                    task_batches.append(current_batch)
                current_batch = 1
            prev_was_task = True
        else:
            if prev_was_task and current_batch > 0:
                task_batches.append(current_batch)
                current_batch = 0
            prev_was_task = False
    if current_batch > 0:
        task_batches.append(current_batch)

    # Map batches to agent names (consume actual_subagents in order)
    # Deduplicate agents
    seen = set()
    unique = []
    for a in actual_subagents:
        if a not in seen:
            seen.add(a)
            unique.append(a)

    stages: list[list[str]] = []
    idx = 0
    for batch_size in task_batches:
        stage = unique[idx:idx + batch_size]
        if stage:
            stages.append(stage)
        idx += batch_size

    # If any agents remain unaccounted, add them
    if idx < len(unique):
        stages.append(unique[idx:])

    return stages if stages else [unique]


def _draw_pill(ax, x, y, label, color, fontsize=5.5, alpha=0.90):
    """Draw a colored pill with text, return (x, width)."""
    tw = len(label) * 0.0038 + 0.012
    pill = FancyBboxPatch(
        (x, y - 0.008), tw, 0.016,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, zorder=5,
    )
    ax.add_patch(pill)
    ax.text(x + tw / 2, y, label,
            ha="center", va="center", fontsize=fontsize, color="white",
            fontweight="bold", transform=ax.transAxes, zorder=6)
    return tw


def draw_summary(results: list[dict], output_path: str, campaign: str = "3-Way Parallel"):
    n = len(results)
    fig_h = HEADER_H + n * ROW_H + FOOTER_H
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    total_time = sum(r.get("wall_time_s", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    n_ok = sum(1 for r in results if r.get("routing_match", False))
    max_time = max((r.get("wall_time_s", 1) for r in results), default=1)

    # Count true 3-parallel dispatches
    n_3par = 0
    for r in results:
        stages = _derive_execution_stages(
            r.get("tool_names", []), r.get("actual_subagents", []))
        if any(len(s) >= 3 for s in stages):
            n_3par += 1

    # ── Header ──
    header_y = 1.0 - HEADER_H / fig_h
    ax.text(0.50, 1.0 - 0.025, f"DISSOLVE Trace Campaign: {campaign} Execution",
            ha="center", fontsize=15, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    ax.text(0.50, 1.0 - 0.055,
            f"{n} queries  \u2502  {total_time:.0f}s total  \u2502  "
            f"{total_tokens:,} tokens  \u2502  {n_ok}/{n} routing match  \u2502  "
            f"{n_3par}/{n} true 3-way parallel",
            ha="center", fontsize=9.5, color=C_BODY,
            transform=ax.transAxes)

    # Column headers
    col_y = header_y + 0.006
    ax.text(0.01, col_y, "Query", fontsize=7.5, fontweight="bold",
            color=C_BODY, transform=ax.transAxes, va="center")
    ax.text(0.155, col_y, "Time", ha="center",
            fontsize=7.5, fontweight="bold", color=C_BODY,
            transform=ax.transAxes, va="center")
    ax.text(0.205, col_y, "Tokens", ha="center",
            fontsize=7.5, fontweight="bold", color=C_BODY,
            transform=ax.transAxes, va="center")
    ax.text(0.28, col_y, "Execution Pattern",
            fontsize=7.5, fontweight="bold", color=C_BODY,
            transform=ax.transAxes, va="center")
    ax.text(0.97, col_y, "Route", fontsize=7.5, fontweight="bold",
            color=C_BODY, ha="center", transform=ax.transAxes, va="center")

    # ── Rows ──
    for i, r in enumerate(results):
        row_frac = ROW_H / fig_h
        row_y = header_y - (i + 0.5) * row_frac

        if i % 2 == 0:
            stripe_y = header_y - (i + 1) * row_frac
            ax.add_patch(Rectangle(
                (0, stripe_y), 1, row_frac,
                facecolor=C_GRID, edgecolor="none",
                transform=ax.transAxes, zorder=0,
            ))

        # Query name
        name = r.get("name", f"query-{i}")
        ax.text(0.01, row_y, name,
                fontsize=6.5, color=C_TITLE, fontweight="bold",
                transform=ax.transAxes, va="center")

        # Time
        wt = r.get("wall_time_s", 0)
        ax.text(0.155, row_y, f"{wt:.0f}s", ha="center",
                fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="center")

        # Tokens
        tokens = r.get("total_tokens", 0)
        ax.text(0.205, row_y, f"{tokens:,}", ha="center",
                fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="center")

        # ── Execution pattern with fork/join arrows ──
        actual = r.get("actual_subagents", [])
        tool_names = r.get("tool_names", [])
        stages = _derive_execution_stages(tool_names, actual)

        # Layout stages left-to-right with arrows between them
        stage_x = 0.28
        arrow_len = 0.018
        stage_gap = 0.008  # gap between arrow tip and next pill

        for si, stage in enumerate(stages):
            n_in_stage = len(stage)

            if n_in_stage == 1:
                # Single agent — one pill centered on row_y
                sa = stage[0]
                label = AGENT_SHORT.get(sa, sa)
                color = _agent_color(sa)
                tw = _draw_pill(ax, stage_x, row_y, label, color)
                pill_right = stage_x + tw

                # Arrow to next stage
                if si < len(stages) - 1:
                    ax.annotate("", xy=(pill_right + arrow_len, row_y),
                                xytext=(pill_right + 0.003, row_y),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=dict(arrowstyle="->", color=C_ARROW,
                                                lw=1.2, shrinkA=0, shrinkB=0),
                                zorder=4)
                    stage_x = pill_right + arrow_len + stage_gap

            else:
                # Multiple agents in parallel — stacked with fork/join arrows
                pill_spacing = 0.022
                total_height = (n_in_stage - 1) * pill_spacing
                top_y = row_y + total_height / 2
                bottom_y = row_y - total_height / 2

                # Fork arrows from left
                fork_x = stage_x - 0.005
                max_tw = 0
                pill_positions = []

                for j, sa in enumerate(stage):
                    py = top_y - j * pill_spacing
                    label = AGENT_SHORT.get(sa, sa)
                    color = _agent_color(sa)
                    tw = _draw_pill(ax, stage_x, py, label, color)
                    pill_positions.append((py, stage_x + tw))
                    max_tw = max(max_tw, tw)

                # Fork lines: vertical line on left, small horizontals to each pill
                fork_line_x = stage_x - 0.012
                # Vertical fork line
                ax.plot([fork_line_x, fork_line_x],
                        [bottom_y, top_y],
                        color=C_ARROW, lw=1.2, transform=ax.transAxes,
                        zorder=3, solid_capstyle="round")
                # Horizontal ticks to each pill
                for j, sa in enumerate(stage):
                    py = top_y - j * pill_spacing
                    ax.annotate("", xy=(stage_x - 0.002, py),
                                xytext=(fork_line_x, py),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=dict(arrowstyle="->", color=C_ARROW,
                                                lw=1.0, shrinkA=0, shrinkB=0),
                                zorder=4)

                # Incoming arrow to fork
                if si > 0:
                    ax.annotate("", xy=(fork_line_x, row_y),
                                xytext=(fork_line_x - arrow_len + 0.003, row_y),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=dict(arrowstyle="->", color=C_ARROW,
                                                lw=1.2, shrinkA=0, shrinkB=0),
                                zorder=4)

                # Join lines on right side
                join_line_x = stage_x + max_tw + 0.008
                ax.plot([join_line_x, join_line_x],
                        [bottom_y, top_y],
                        color=C_ARROW, lw=1.2, transform=ax.transAxes,
                        zorder=3, solid_capstyle="round")
                # Horizontal ticks from each pill to join
                for py, pr in pill_positions:
                    ax.plot([pr + 0.002, join_line_x],
                            [py, py],
                            color=C_ARROW, lw=1.0, transform=ax.transAxes,
                            zorder=3)

                # Arrow from join to next stage
                if si < len(stages) - 1:
                    ax.annotate("", xy=(join_line_x + arrow_len, row_y),
                                xytext=(join_line_x + 0.003, row_y),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=dict(arrowstyle="->", color=C_ARROW,
                                                lw=1.2, shrinkA=0, shrinkB=0),
                                zorder=4)
                    stage_x = join_line_x + arrow_len + stage_gap
                else:
                    stage_x = join_line_x + 0.005

        # ── Parallel label ──
        # Determine pattern description
        pattern_parts = []
        for stage in stages:
            if len(stage) == 1:
                pattern_parts.append("1")
            else:
                pattern_parts.append(str(len(stage)))
        pattern_desc = "+".join(pattern_parts)
        if all(len(s) >= 3 for s in stages) and len(stages) == 1:
            pattern_label = "3-way parallel"
        elif any(len(s) >= 3 for s in stages):
            pattern_label = f"hybrid ({pattern_desc})"
        elif any(len(s) >= 2 for s in stages):
            pattern_label = f"1+2 parallel ({pattern_desc})"
        else:
            pattern_label = f"sequential ({pattern_desc})"

        ax.text(stage_x + 0.008, row_y, pattern_label,
                fontsize=5.5, color=C_MUTED, fontstyle="italic",
                transform=ax.transAxes, va="center", zorder=4)

        # Routing match
        ok = r.get("routing_match", False)
        ax.text(0.97, row_y,
                "PASS" if ok else "MISS",
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color=C_CHECK if ok else C_CROSS,
                transform=ax.transAxes, zorder=5)

    # ── Footer ──
    footer_y = FOOTER_H / fig_h * 0.4
    ax.text(0.50, footer_y,
            f"DISSOLVE v8  \u2502  {campaign} Execution  \u2502  "
            f"Gemini 2.5 Pro + Flash  \u2502  2026-02-24",
            ha="center", fontsize=8, color=C_BODY,
            transform=ax.transAxes)

    fig.savefig(output_path, dpi=200, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3-way parallel trace campaign results"
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

    output = args.output or str(_DIR / "summary_3way_parallel.png")
    campaign = data.get("campaign", "3-Way Parallel")
    draw_summary(results, output, campaign=campaign.replace("-", " ").title())


if __name__ == "__main__":
    main()
