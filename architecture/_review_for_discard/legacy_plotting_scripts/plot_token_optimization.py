"""Token optimization analysis for 9-polymer selectivity query.

Shows progression from v1 (unguarded) through v6 (deterministic) with
token usage, wall time, messages, and tool calls.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).parent

# ── Style ─────────────────────────────────────────────────────────
PUB_FONT = "Liberation Sans"
PUB_FONTSIZE = 8
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREY = "#999999"


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


# ── Data ──────────────────────────────────────────────────────────
# Each version's n=9 metrics (averaged over 2 runs where available)

versions = [
    {
        "label": "v1\n(no guardrails)",
        "tokens": 2_000_000,
        "time_s": 634,
        "messages": 71,
        "tool_calls": 35,
        "source": "MEMORY.md (pre-v6)",
        "color": VERMILLION,
    },
    {
        "label": "v2\n(early guardrails)",
        "tokens": 19558,  # total_chars as proxy (no token count available)
        "time_s": 82.2,
        "messages": 16,
        "tool_calls": 7,
        "source": "scaling_benchmark_results.json",
        "color": ORANGE,
        "note": "chars not tokens",
    },
    {
        "label": "v5\n(relaxed guardrails)",
        "tokens": (25126 + 26335) / 2,
        "time_s": (44.1 + 47.8) / 2,
        "messages": 12,
        "tool_calls": 9,
        "source": "scaling_benchmark_current",
        "color": SKY,
    },
    {
        "label": "v6\n(strict guardrails)",
        "tokens": (24825 + 24720) / 2,
        "time_s": (24.4 + 23.5) / 2,
        "messages": 12,
        "tool_calls": 9,
        "source": "scaling_benchmark_v7",
        "color": BLUE,
    },
    {
        "label": "v6\n(deterministic)",
        "tokens": 7013,
        "time_s": (11.0 + 16.0) / 2,
        "messages": 2,
        "tool_calls": 0,
        "source": "scaling_benchmark_run1/run2",
        "color": GREEN,
    },
]

# Pure computation reference
pure_comp = {
    "greedy_time": 0.001,
    "dp_time": 1.3,
    "bb_time": 0.14,
}


def plot_optimization():
    apply_pub_style()

    fig = plt.figure(figsize=(7.0, 8.5))

    # Use gridspec for custom layout: 3 rows
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35,
                          height_ratios=[1, 1, 1.0])

    ax_tokens = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_msgs = fig.add_subplot(gs[1, 0])
    ax_tools = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[2, :])

    labels = [v["label"] for v in versions]
    x = range(len(versions))
    colors = [v["color"] for v in versions]

    # ── Panel A: Token usage (log scale) ──────────────────────────
    tokens = [v["tokens"] for v in versions]
    bars = ax_tokens.bar(x, tokens, color=colors, edgecolor="white", linewidth=0.5, width=0.65)
    ax_tokens.set_yscale("log")
    ax_tokens.set_ylabel("Total tokens")
    ax_tokens.set_xticks(x)
    ax_tokens.set_xticklabels(labels, fontsize=PUB_FONTSIZE - 1)
    ax_tokens.set_title("(a) Token usage", fontsize=PUB_FONTSIZE, loc="left")

    for i, (bar, tok) in enumerate(zip(bars, tokens)):
        if tok >= 1_000_000:
            lbl = f"{tok/1_000_000:.0f}M"
        elif tok >= 1_000:
            lbl = f"{tok/1_000:.1f}K"
        else:
            lbl = str(int(tok))
        ax_tokens.text(bar.get_x() + bar.get_width() / 2, tok * 1.3,
                       lbl, ha="center", va="bottom", fontsize=PUB_FONTSIZE - 1,
                       fontweight="bold")

    # Reduction annotations
    ax_tokens.annotate("", xy=(4, tokens[4] * 0.7), xytext=(0, tokens[0] * 0.7),
                       arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8,
                                       connectionstyle="arc3,rad=-0.2"))
    ax_tokens.text(2, tokens[0] * 0.25, "285\u00d7\nreduction",
                   ha="center", fontsize=PUB_FONTSIZE - 1, color=GREY,
                   fontweight="bold")

    # ── Panel B: Wall time (log scale) ────────────────────────────
    times = [v["time_s"] for v in versions]
    bars = ax_time.bar(x, times, color=colors, edgecolor="white", linewidth=0.5, width=0.65)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Wall-clock time (s)")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, fontsize=PUB_FONTSIZE - 1)
    ax_time.set_title("(b) Wall-clock time", fontsize=PUB_FONTSIZE, loc="left")

    for bar, t in zip(bars, times):
        lbl = f"{t:.0f}s" if t >= 10 else f"{t:.1f}s"
        ax_time.text(bar.get_x() + bar.get_width() / 2, t * 1.3,
                     lbl, ha="center", va="bottom", fontsize=PUB_FONTSIZE - 1,
                     fontweight="bold")

    # Pure computation reference line
    ax_time.axhline(pure_comp["greedy_time"], color=GREY, linewidth=0.6,
                    linestyle="--", alpha=0.5)
    ax_time.text(4.4, pure_comp["greedy_time"] * 1.5, "greedy\n(0.001s)",
                 fontsize=PUB_FONTSIZE - 2, color=GREY, va="bottom")

    ax_time.annotate("", xy=(4, times[4] * 0.7), xytext=(0, times[0] * 0.7),
                     arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8,
                                     connectionstyle="arc3,rad=-0.2"))
    ax_time.text(2, times[0] * 0.25, "47\u00d7\nfaster",
                 ha="center", fontsize=PUB_FONTSIZE - 1, color=GREY,
                 fontweight="bold")

    # ── Panel C: Messages ─────────────────────────────────────────
    msgs = [v["messages"] for v in versions]
    bars = ax_msgs.bar(x, msgs, color=colors, edgecolor="white", linewidth=0.5, width=0.65)
    ax_msgs.set_ylabel("Messages")
    ax_msgs.set_xticks(x)
    ax_msgs.set_xticklabels(labels, fontsize=PUB_FONTSIZE - 1)
    ax_msgs.set_title("(c) LLM messages", fontsize=PUB_FONTSIZE, loc="left")

    for bar, m in zip(bars, msgs):
        ax_msgs.text(bar.get_x() + bar.get_width() / 2, m + 1,
                     str(m), ha="center", va="bottom", fontsize=PUB_FONTSIZE - 1,
                     fontweight="bold")

    # ── Panel D: Tool calls ───────────────────────────────────────
    tools = [v["tool_calls"] for v in versions]
    bars = ax_tools.bar(x, tools, color=colors, edgecolor="white", linewidth=0.5, width=0.65)
    ax_tools.set_ylabel("Tool calls")
    ax_tools.set_xticks(x)
    ax_tools.set_xticklabels(labels, fontsize=PUB_FONTSIZE - 1)
    ax_tools.set_title("(d) Tool calls", fontsize=PUB_FONTSIZE, loc="left")

    for bar, tc in zip(bars, tools):
        ax_tools.text(bar.get_x() + bar.get_width() / 2, tc + 0.5,
                      str(tc), ha="center", va="bottom", fontsize=PUB_FONTSIZE - 1,
                      fontweight="bold")

    # ── Panel E: Summary table ────────────────────────────────────
    ax_summary.axis("off")

    col_labels = ["Version", "Tokens", "Time", "Msgs", "Tools", "Key change"]
    row_data = [
        ["v1 (no guardrails)", "2,000K", "634s", "71", "35",
         "Subagents explored filesystem"],
        ["v2 (early guardrails)", "~20K*", "82s", "16", "7",
         "Iteration/token/tool-call caps"],
        ["v5 (relaxed guardrails)", "25.7K", "46s", "12", "9",
         "rank_solvents_selectivity tool"],
        ["v6 (strict guardrails)", "24.8K", "24s", "12", "9",
         "Truncation + synthesis injection"],
        ["v6 (deterministic)", "7.0K", "13.5s", "2", "0",
         "Bypass LLM for selectivity"],
    ]

    table = ax_summary.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=[GREY + "30"] * len(col_labels),
        colWidths=[0.22, 0.1, 0.08, 0.07, 0.07, 0.33],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(PUB_FONTSIZE - 1)
    table.scale(1, 1.5)

    # Color version column cells
    for i, v in enumerate(versions):
        table[i + 1, 0].set_facecolor(v["color"] + "30")

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")

    # Left-align "Key change" column
    for i in range(len(row_data) + 1):
        table[i, 5].set_text_props(ha="left")
        table[i, 0].set_text_props(ha="left")

    ax_summary.set_title("(e) Optimization progression \u2014 9 polymers at 120\u00b0C",
                         fontsize=PUB_FONTSIZE, loc="left", pad=10)

    out = HERE / "token_optimization_analysis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_optimization()
