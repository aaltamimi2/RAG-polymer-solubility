"""Regenerate scaling_benchmark.png comparing v6 (deterministic) vs v7 (advisory).

Uses the project's publication-quality plot style.

Usage:
    python architecture/plot_scaling_benchmark.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Publication style (matches visualization.py conventions) ──────────

_PUB_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#000000",  # black
]

_PUB_FONTSIZE = 8
_PUB_FONT = "Liberation Sans"


def _apply_pub_style(fig, axes):
    """Apply publication-quality style to figure and axes."""
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flat

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [_PUB_FONT, "Arial", "DejaVu Sans"],
        "font.size": _PUB_FONTSIZE,
        "mathtext.default": "regular",
    })

    for ax in axes:
        ax.tick_params(
            which="both", direction="in",
            top=True, right=True, bottom=True, left=True,
            labelsize=_PUB_FONTSIZE,
        )
        ax.xaxis.label.set_size(_PUB_FONTSIZE)
        ax.yaxis.label.set_size(_PUB_FONTSIZE)
        if ax.get_title():
            ax.title.set_size(_PUB_FONTSIZE)

    for t in fig.texts:
        t.set_fontsize(_PUB_FONTSIZE)


# ── Data loading ──────────────────────────────────────────────────────

def load_runs(json_files: list[Path]) -> dict[int, list[float]]:
    """Load wall times from JSON files, grouped by n_polymers."""
    times: dict[int, list[float]] = defaultdict(list)
    for jf in json_files:
        if not jf.exists():
            continue
        with open(jf) as f:
            data = json.load(f)
        for entry in data:
            if entry.get("error") is None:
                times[entry["n_polymers"]].append(entry["wall_time_s"])
    return dict(times)


def load_tokens(json_files: list[Path]) -> dict[int, list[int]]:
    """Load total token counts from JSON files, grouped by n_polymers."""
    tokens: dict[int, list[int]] = defaultdict(list)
    for jf in json_files:
        if not jf.exists():
            continue
        with open(jf) as f:
            data = json.load(f)
        for entry in data:
            if entry.get("error") is None:
                tokens[entry["n_polymers"]].append(entry["total_tokens"])
    return dict(tokens)


def load_tool_calls(json_files: list[Path]) -> dict[int, list[int]]:
    """Load tool call counts from JSON files, grouped by n_polymers."""
    calls: dict[int, list[int]] = defaultdict(list)
    for jf in json_files:
        if not jf.exists():
            continue
        with open(jf) as f:
            data = json.load(f)
        for entry in data:
            if entry.get("error") is None:
                calls[entry["n_polymers"]].append(entry["n_tool_calls"])
    return dict(calls)


# ── Plotting ──────────────────────────────────────────────────────────

def main():
    arch_dir = Path(__file__).resolve().parent

    # v6 data: original deterministic bypass runs
    v6_files = [
        arch_dir / "scaling_benchmark_run1.json",
        arch_dir / "scaling_benchmark_run2.json",
    ]
    # v7 data: advisory routing runs
    v7_files = [
        arch_dir / "scaling_benchmark_v7_run1.json",
        arch_dir / "scaling_benchmark_v7_run2.json",
    ]

    v6_times = load_runs(v6_files)
    v7_times = load_runs(v7_files)
    v6_tokens = load_tokens(v6_files)
    v7_tokens = load_tokens(v7_files)
    v6_tools = load_tool_calls(v6_files)
    v7_tools = load_tool_calls(v7_files)

    if not v6_times and not v7_times:
        print("No benchmark data found.")
        return

    # ── 3-panel figure: time, tokens, tool calls ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.4))
    _apply_pub_style(fig, np.array([ax1, ax2, ax3]))

    for data, color, label, marker, ax_set in [
        (v6_times, _PUB_COLORS[0], "v6: deterministic bypass", "o",
         [(ax1, v6_times), (ax2, v6_tokens), (ax3, v6_tools)]),
        (v7_times, _PUB_COLORS[1], "v7: advisory routing", "s",
         [(ax1, v7_times), (ax2, v7_tokens), (ax3, v7_tools)]),
    ]:
        for ax, series_data in ax_set:
            if not series_data:
                continue
            ns = sorted(series_data.keys())
            means = [np.mean(series_data[n]) for n in ns]
            stds = [np.std(series_data[n], ddof=1) if len(series_data[n]) > 1 else 0.0 for n in ns]

            ax.errorbar(
                ns, means, yerr=stds,
                color=color, marker=marker, markersize=3.5,
                linewidth=1.0, capsize=2, capthick=0.6,
                elinewidth=0.6, zorder=3,
                label=label if ax == ax1 else None,
            )

    # Axes labels
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Number of polymers")
        all_ns = sorted(set(list(v6_times.keys()) + list(v7_times.keys())))
        ax.set_xticks(all_ns)
        ax.set_ylim(bottom=0)

    ax1.set_ylabel("Query time (s)")
    ax2.set_ylabel("Total tokens")
    ax3.set_ylabel("Tool calls")

    # Format token axis with K suffix
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")
    )

    # Panel labels
    for ax, letter in zip([ax1, ax2, ax3], ["a", "b", "c"]):
        ax.text(
            -0.18, 1.05, f"({letter})",
            transform=ax.transAxes,
            fontsize=_PUB_FONTSIZE, fontweight="bold",
            va="bottom", ha="left",
        )

    # Single legend at top of first panel
    ax1.legend(
        loc="upper left", fontsize=5.5, framealpha=0.9,
        edgecolor="none", fancybox=False,
        handlelength=1.5, handletextpad=0.4,
    )

    fig.tight_layout(pad=0.4, w_pad=1.0)
    out_path = arch_dir / "scaling_benchmark.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
