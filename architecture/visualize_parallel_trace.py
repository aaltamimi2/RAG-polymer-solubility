"""Visualize a LangSmith trace as a Gantt-style swim-lane diagram.

Supports both parallel and sequential multi-agent execution patterns.
Auto-detects the pattern and renders accordingly.
Usage:
    python architecture/visualize_parallel_trace.py                          # latest trace
    python architecture/visualize_parallel_trace.py <trace-id>               # specific trace
    python architecture/visualize_parallel_trace.py <trace-id> -o output.png # custom output
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langsmith import Client

# ── Colours ──────────────────────────────────────────────────
C_BG        = "#FAFAFA"
C_TEXT      = "#2C3E50"
C_ORCH      = "#2C3E50"       # orchestrator - dark blue-grey
C_ORCH_FILL = "#D6EAF8"
C_SEP       = "#E74C3C"       # separation-engineer - red
C_SEP_FILL  = "#FADBD8"
C_TEA       = "#27AE60"       # tea-lca-analyst - green
C_TEA_FILL  = "#D5F5E3"
C_TOOL      = "#8E44AD"       # tool calls - purple
C_TOOL_FILL = "#F5EEF8"
C_MODEL     = "#F39C12"       # model calls - orange
C_MODEL_FILL= "#FEF9E7"
C_GRAY      = "#95A5A6"
C_WHITE     = "#FFFFFF"
C_PARALLEL  = "#E74C3C"       # parallel indicator

# Subagent colour map
AGENT_COLORS = {
    "separation-engineer": (C_SEP, C_SEP_FILL),
    "tea-lca-analyst": (C_TEA, C_TEA_FILL),
    "safety-analyst": ("#3498DB", "#D6EAF8"),
    "scholar-researcher": ("#E67E22", "#FEF5E7"),
    "patent-researcher": ("#E67E22", "#FEF5E7"),
    "rag-analyst": ("#1ABC9C", "#D1F2EB"),
    "visualization-specialist": ("#9B59B6", "#F5EEF8"),
    "statistics-ml": ("#34495E", "#EAEDED"),
}


def fetch_trace_data(client: Client, trace_id: str):
    """Fetch all runs and organize into orchestrator + subagent swim lanes."""
    all_runs = list(client.list_runs(trace_id=trace_id))
    all_runs.sort(key=lambda r: r.start_time)

    root = next((r for r in all_runs if r.parent_run_id is None), all_runs[0])
    root_start = root.start_time
    run_map = {str(r.id): r for r in all_runs}

    def ms(dt):
        return (dt - root_start).total_seconds() * 1000

    # Classify runs into lanes
    orchestrator_events = []
    subagent_lanes = {}  # name -> list of events
    parallel_groups = []  # list of (start_ms, end_ms, agents)

    # Find task tool calls (subagent dispatches)
    task_runs = [r for r in all_runs if r.name == "task" and r.run_type == "tool"]

    # Detect parallel dispatch by timing overlap.
    # LangGraph's Send dispatches each task to a separate tools node,
    # so they have different parents. We detect parallelism by start time
    # proximity (within 500ms of each other) and overlapping execution.
    PARALLEL_WINDOW_MS = 500
    used = set()
    for i, tr1 in enumerate(task_runs):
        if i in used:
            continue
        group = [tr1]
        for j, tr2 in enumerate(task_runs):
            if j <= i or j in used:
                continue
            if abs(ms(tr1.start_time) - ms(tr2.start_time)) < PARALLEL_WINDOW_MS:
                group.append(tr2)
                used.add(j)
        if len(group) >= 2:
            used.add(i)
            starts = [ms(t.start_time) for t in group]
            ends = [ms(t.end_time) for t in group if t.end_time]
            agents = []
            for t in group:
                inp = str(t.inputs) if t.inputs else ""
                for name in AGENT_COLORS:
                    if name in inp:
                        agents.append(name)
                        break
                else:
                    agents.append("unknown")
            if ends:
                parallel_groups.append((min(starts), max(ends), agents))

    # Build subagent lanes from task runs
    for tr in task_runs:
        inp = str(tr.inputs) if tr.inputs else ""
        agent_name = "unknown"
        for name in AGENT_COLORS:
            if name in inp:
                agent_name = name
                break

        start = ms(tr.start_time)
        dur = ms(tr.end_time) - start if tr.end_time else 0

        if agent_name not in subagent_lanes:
            subagent_lanes[agent_name] = []

        # Add the subagent bar
        subagent_lanes[agent_name].append({
            "type": "subagent",
            "name": agent_name,
            "start_ms": start,
            "duration_ms": dur,
        })

        # Find tool calls WITHIN this subagent
        subagent_chain_id = None
        for r in all_runs:
            if (r.parent_run_id and str(r.parent_run_id) == str(tr.id)
                    and r.run_type == "chain"):
                subagent_chain_id = str(r.id)
                break

        if subagent_chain_id:
            for r in all_runs:
                if r.run_type == "tool" and r.name != "task" and r.name != "write_todos":
                    # Check if this tool is a descendant of the subagent chain
                    pid = r.parent_run_id
                    depth = 0
                    is_descendant = False
                    seen = set()
                    while pid and str(pid) not in seen and depth < 10:
                        seen.add(str(pid))
                        if str(pid) == subagent_chain_id:
                            is_descendant = True
                            break
                        parent = run_map.get(str(pid))
                        pid = parent.parent_run_id if parent else None
                        depth += 1

                    if is_descendant:
                        t_start = ms(r.start_time)
                        t_dur = ms(r.end_time) - t_start if r.end_time else 0
                        subagent_lanes[agent_name].append({
                            "type": "tool",
                            "name": r.name,
                            "start_ms": t_start,
                            "duration_ms": t_dur,
                        })

    # Orchestrator model calls (top-level only)
    for r in all_runs:
        if r.run_type == "llm" and r.parent_run_id:
            parent = run_map.get(str(r.parent_run_id))
            if parent and parent.name == "model":
                grandparent = run_map.get(str(parent.parent_run_id)) if parent.parent_run_id else None
                if grandparent and grandparent.parent_run_id is None:
                    # Top-level model call
                    start = ms(r.start_time)
                    dur = ms(r.end_time) - start if r.end_time else 0
                    orchestrator_events.append({
                        "type": "model",
                        "name": "LLM call",
                        "start_ms": start,
                        "duration_ms": dur,
                        "tokens": r.total_tokens or 0,
                    })

    # Orchestrator tool calls (write_todos + core tools at top level)
    for r in all_runs:
        if r.run_type == "tool" and r.name != "task":
            parent = run_map.get(str(r.parent_run_id)) if r.parent_run_id else None
            if parent and parent.name == "tools":
                gp = run_map.get(str(parent.parent_run_id)) if parent.parent_run_id else None
                if gp and gp.parent_run_id is None:
                    start = ms(r.start_time)
                    dur = ms(r.end_time) - start if r.end_time else 0
                    orchestrator_events.append({
                        "type": "tool",
                        "name": r.name,
                        "start_ms": start,
                        "duration_ms": dur,
                    })

    total_ms = ms(root.end_time) if root.end_time else 0

    return {
        "orchestrator": sorted(orchestrator_events, key=lambda e: e["start_ms"]),
        "subagents": subagent_lanes,
        "parallel_groups": parallel_groups,
        "total_ms": total_ms,
        "total_tokens": root.total_tokens or 0,
        "root": root,
    }


def draw_parallel_trace(data, output_path: str):
    """Draw a Gantt-style swim-lane diagram for parallel or sequential execution."""
    total_ms = data["total_ms"]
    orchestrator = data["orchestrator"]
    subagents = data["subagents"]
    parallel_groups = data["parallel_groups"]
    is_parallel = len(parallel_groups) > 0

    # Calculate lane height based on events per lane
    n_lanes = 1 + len(subagents)
    fig_width = 16
    base_lane_height = 2.2

    # Orchestrator may need more height if it has many tool calls
    orch_tools = [e for e in orchestrator if e["type"] == "tool"]
    orch_lane_height = max(base_lane_height, 1.2 + len(orch_tools) * 0.42 + 0.4)

    sub_lane_heights = []
    for agent_name in subagents:
        n_tools = len([e for e in subagents[agent_name] if e["type"] == "tool"])
        h = max(base_lane_height, 1.2 + n_tools * 0.42 + 0.4)
        sub_lane_heights.append(h)

    total_lane_height = orch_lane_height + sum(sub_lane_heights)
    header_height = 2.0
    footer_height = 1.8
    fig_height = header_height + total_lane_height + footer_height + 0.5

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    margin_left = 3.5
    margin_right = 0.5
    plot_width = fig_width - margin_left - margin_right
    ax.set_xlim(0, fig_width)
    ax.set_ylim(-fig_height + 0.5, header_height)
    ax.axis("off")

    def x_pos(ms_val):
        if total_ms == 0:
            return margin_left
        return margin_left + (ms_val / total_ms) * plot_width

    def bw(dur_ms):
        if total_ms == 0:
            return 0.12
        return max((dur_ms / total_ms) * plot_width, 0.12)

    # ── Title ──
    pattern_label = "Parallel" if is_parallel else "Sequential"
    ax.text(fig_width / 2, header_height - 0.3,
            f"DISSOLVE Agent — {pattern_label} Execution Trace",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C_ORCH, fontfamily="sans-serif")

    meta = f"Total: {total_ms / 1000:.1f}s | Tokens: {data['total_tokens']:,} | Model: Gemini 2.0 Flash"
    ax.text(fig_width / 2, header_height - 0.85, meta,
            ha="center", va="center", fontsize=9, color=C_GRAY)

    # ── Time axis ──
    axis_y = header_height - 1.45
    ax.plot([margin_left, margin_left + plot_width], [axis_y, axis_y],
            color=C_GRAY, lw=0.8, alpha=0.6)

    nice_interval = 5000 if total_ms <= 30000 else 10000
    t = 0
    while t <= total_ms + 1:
        tx = x_pos(t)
        ax.plot([tx, tx], [axis_y - 0.08, axis_y + 0.08], color=C_GRAY, lw=0.5)
        ax.text(tx, axis_y - 0.22, f"{t / 1000:.0f}s",
                ha="center", va="top", fontsize=7, color=C_GRAY, fontfamily="monospace")
        t += nice_interval

    # ── Draw lanes ──
    lane_y_start = axis_y - 0.6
    lane_names = ["Orchestrator"] + list(subagents.keys())
    all_lane_heights = [orch_lane_height] + sub_lane_heights

    # Precompute lane y positions
    lane_y_tops = []
    lane_y_bots = []
    y_cursor = lane_y_start
    for lh in all_lane_heights:
        lane_y_tops.append(y_cursor)
        lane_y_bots.append(y_cursor - lh)
        y_cursor -= lh

    for lane_idx, lane_name in enumerate(lane_names):
        y_top = lane_y_tops[lane_idx]
        y_bot = lane_y_bots[lane_idx]
        lane_height_cur = all_lane_heights[lane_idx]
        y_mid = (y_top + y_bot) / 2

        # Lane colours
        if lane_idx == 0:
            color, fill = C_ORCH, C_ORCH_FILL
        else:
            agent_key = list(subagents.keys())[lane_idx - 1]
            color, fill = AGENT_COLORS.get(agent_key, (C_GRAY, "#EAEDED"))

        # Lane background
        ax.add_patch(FancyBboxPatch(
            (margin_left - 0.05, y_bot + 0.05), plot_width + 0.1, lane_height_cur - 0.1,
            boxstyle="round,pad=0.05", facecolor=fill, edgecolor="none", alpha=0.25))

        # Lane label
        display_name = lane_name.replace("-", "\n")
        ax.text(margin_left - 0.3, y_mid, display_name,
                ha="right", va="center", fontsize=8.5, fontweight="bold",
                color=color, fontfamily="sans-serif")

        # Separator
        if lane_idx > 0:
            ax.plot([margin_left - 0.1, margin_left + plot_width + 0.05],
                    [y_top + 0.02, y_top + 0.02], color=C_GRAY, lw=0.4, alpha=0.3, ls="--")

        # ── Draw events ──
        if lane_idx == 0:
            events = orchestrator
        else:
            events = subagents[list(subagents.keys())[lane_idx - 1]]

        # Separate subagent spans from tool calls
        span_events = [e for e in events if e["type"] == "subagent"]
        tool_events = [e for e in events if e["type"] == "tool"]
        model_events = [e for e in events if e["type"] == "model"]

        # Row positions within lane
        span_y = y_top - 0.35  # subagent span near top
        tool_y_base = y_top - 0.95  # tools below
        model_y = y_mid  # model calls centered

        # Draw subagent span bars (full-width background bar)
        for evt in span_events:
            ex = x_pos(evt["start_ms"])
            ew = bw(evt["duration_ms"])
            h = 0.45
            rect = FancyBboxPatch(
                (ex, span_y - h / 2), ew, h,
                boxstyle="round,pad=0.04",
                facecolor=fill, edgecolor=color, linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            dur_s = evt["duration_ms"] / 1000
            lbl = f"{evt['name']} ({dur_s:.1f}s)"
            ax.text(ex + 0.15, span_y, lbl,
                    ha="left", va="center", fontsize=7, fontweight="bold",
                    color=color, fontfamily="monospace")

        # Draw tool call bars (stacked below span) — subagent lanes only
        # Orchestrator tools are rendered separately below (core tools + write_todos markers)
        if lane_idx > 0:
            for ti, evt in enumerate(sorted(tool_events, key=lambda e: e["start_ms"])):
                ex = x_pos(evt["start_ms"])
                ew = bw(evt["duration_ms"])
                ty = tool_y_base - ti * 0.42
                h = 0.32
                rect = FancyBboxPatch(
                    (ex, ty - h / 2), ew, h,
                    boxstyle="round,pad=0.03",
                    facecolor=C_TOOL_FILL, edgecolor=C_TOOL, linewidth=1.2, alpha=0.9)
                ax.add_patch(rect)
                dur_s = evt["duration_ms"] / 1000
                lbl = f"{evt['name']} ({dur_s:.1f}s)" if dur_s >= 1 else f"{evt['name']} ({evt['duration_ms']:.0f}ms)"
                # Label to the right if bar is small
                if ew > 2.0:
                    ax.text(ex + ew / 2, ty, lbl,
                            ha="center", va="center", fontsize=6.5,
                            color=C_TOOL, fontweight="bold", fontfamily="monospace")
                else:
                    ax.text(ex + ew + 0.1, ty, lbl,
                            ha="left", va="center", fontsize=6.5,
                            color=C_TOOL, fontfamily="monospace")

        # Draw model call bars (orchestrator only)
        for evt in model_events:
            ex = x_pos(evt["start_ms"])
            ew = bw(evt["duration_ms"])
            h = 0.38
            rect = FancyBboxPatch(
                (ex, model_y - h / 2), ew, h,
                boxstyle="round,pad=0.03",
                facecolor=C_MODEL_FILL, edgecolor=C_MODEL, linewidth=1.5, alpha=0.9)
            ax.add_patch(rect)
            dur_ms = evt["duration_ms"]
            lbl = f"LLM ({dur_ms:.0f}ms)"
            if ew > 1.5:
                ax.text(ex + ew / 2, model_y, lbl,
                        ha="center", va="center", fontsize=6.5,
                        color=C_MODEL, fontweight="bold", fontfamily="monospace")
            else:
                ax.text(ex + ew + 0.1, model_y, lbl,
                        ha="left", va="center", fontsize=6,
                        color=C_MODEL, fontfamily="monospace")

        # Draw orchestrator core tool calls (non-subagent tools like analyze_selective_solubility)
        if lane_idx == 0:
            core_tools = [e for e in tool_events if e["name"] != "write_todos"]
            write_todos_events = [e for e in tool_events if e["name"] == "write_todos"]

            # Core tools get prominent rendering (same style as subagent tool calls)
            for ti, evt in enumerate(sorted(core_tools, key=lambda e: e["start_ms"])):
                ex = x_pos(evt["start_ms"])
                ew = bw(evt["duration_ms"])
                ty = tool_y_base - ti * 0.42
                h = 0.32
                rect = FancyBboxPatch(
                    (ex, ty - h / 2), ew, h,
                    boxstyle="round,pad=0.03",
                    facecolor=C_TOOL_FILL, edgecolor=C_TOOL, linewidth=1.2, alpha=0.9)
                ax.add_patch(rect)
                dur_s = evt["duration_ms"] / 1000
                lbl = f"{evt['name']} ({dur_s:.1f}s)" if dur_s >= 1 else f"{evt['name']} ({evt['duration_ms']:.0f}ms)"
                if ew > 2.0:
                    ax.text(ex + ew / 2, ty, lbl,
                            ha="center", va="center", fontsize=6.5,
                            color=C_TOOL, fontweight="bold", fontfamily="monospace")
                else:
                    ax.text(ex + ew + 0.1, ty, lbl,
                            ha="left", va="center", fontsize=6.5,
                            color=C_TOOL, fontfamily="monospace")

            # write_todos get subtle markers
            for evt in write_todos_events:
                ex = x_pos(evt["start_ms"])
                ax.plot([ex], [y_top - 0.2], marker="v", markersize=5,
                        color=C_GRAY, alpha=0.6)
                ax.text(ex, y_top - 0.08, "plan", ha="center", va="bottom",
                        fontsize=5, color=C_GRAY, fontfamily="monospace", alpha=0.7)

    # ── Parallel execution indicators ──
    for pg_start, pg_end, agents in parallel_groups:
        px1 = x_pos(pg_start)
        px2 = x_pos(pg_end)

        agent_indices = []
        for ag in agents:
            if ag in subagents:
                idx = list(subagents.keys()).index(ag) + 1
                agent_indices.append(idx)

        if len(agent_indices) >= 2:
            y_top_bracket = lane_y_tops[min(agent_indices)] + 0.1
            y_bot_bracket = lane_y_bots[max(agent_indices)] + 0.1

            # Parallel region highlight
            rect = plt.Rectangle(
                (px1 - 0.08, y_bot_bracket), px2 - px1 + 0.16,
                y_top_bracket - y_bot_bracket,
                facecolor=C_PARALLEL, alpha=0.035, edgecolor=C_PARALLEL,
                linewidth=2, linestyle="--", zorder=0)
            ax.add_patch(rect)

            # "PARALLEL" badge
            ax.text((px1 + px2) / 2, y_top_bracket + 0.12,
                    "PARALLEL DISPATCH",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=C_PARALLEL, fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=C_WHITE,
                              edgecolor=C_PARALLEL, linewidth=1.5, alpha=0.95))

            # Duration annotation
            dur_s = (pg_end - pg_start) / 1000
            fast_agent = min(agents, key=lambda a: next(
                (e["duration_ms"] for e in subagents.get(a, []) if e["type"] == "subagent"), 999999))
            fast_dur = next(
                (e["duration_ms"] for e in subagents.get(fast_agent, []) if e["type"] == "subagent"), 0)
            slow_dur = pg_end - pg_start
            savings = slow_dur - fast_dur
            if savings > 0:
                ax.text((px1 + px2) / 2, y_bot_bracket - 0.15,
                        f"Wall time: {dur_s:.1f}s (saved {savings / 1000:.1f}s vs sequential)",
                        ha="center", va="top", fontsize=7, color=C_PARALLEL,
                        fontfamily="monospace", style="italic")

    # ── Sequential execution indicators ──
    if not is_parallel and len(subagents) >= 1:
        # Draw sequential flow arrows between orchestrator tool calls and subagent tasks
        task_events = []
        for agent_name, events in subagents.items():
            for e in events:
                if e["type"] == "subagent":
                    agent_idx = list(subagents.keys()).index(agent_name) + 1
                    task_events.append((e["start_ms"], e["duration_ms"], agent_name, agent_idx))

        task_events.sort(key=lambda x: x[0])

        # Draw ordering badges on each sequential step
        step_num = 0
        # Core tools as steps
        orch_core_tools = sorted(
            [e for e in orchestrator if e["type"] == "tool" and e["name"] != "write_todos"],
            key=lambda e: e["start_ms"])
        all_steps = [(e["start_ms"], e["start_ms"] + e["duration_ms"], e["name"], 0)
                     for e in orch_core_tools]
        all_steps += [(s, s + d, name, idx) for s, d, name, idx in task_events]
        all_steps.sort(key=lambda x: x[0])

        for step_num, (s_start, s_end, s_name, s_lane_idx) in enumerate(all_steps):
            # Step badge
            lane_y_t = lane_y_tops[s_lane_idx]
            badge_x = x_pos(s_start) - 0.25
            badge_y = lane_y_t - 0.35
            ax.text(badge_x, badge_y, f"Step {step_num + 1}",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color=C_WHITE, fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#2C3E50",
                              edgecolor="none", alpha=0.85))

            # Draw arrow to next step
            if step_num < len(all_steps) - 1:
                next_start = all_steps[step_num + 1][0]
                next_lane = all_steps[step_num + 1][3]
                x1 = x_pos(s_end)
                x2 = x_pos(next_start)
                y1 = (lane_y_tops[s_lane_idx] + lane_y_bots[s_lane_idx]) / 2
                y2 = (lane_y_tops[next_lane] + lane_y_bots[next_lane]) / 2
                ax.annotate("", xy=(x2 - 0.1, y2), xytext=(x1 + 0.1, y1),
                            arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                           lw=1.8, connectionstyle="arc3,rad=-0.2"),
                            zorder=5)

        # "SEQUENTIAL" badge at top
        if len(all_steps) >= 2:
            first_x = x_pos(all_steps[0][0])
            last_x = x_pos(all_steps[-1][0] + all_steps[-1][1] - all_steps[-1][0])
            mid_x = (first_x + last_x) / 2
            badge_y = lane_y_tops[0] + 0.15
            ax.text(mid_x, badge_y, "SEQUENTIAL EXECUTION",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#2C3E50", fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=C_WHITE,
                              edgecolor="#2C3E50", linewidth=1.5, alpha=0.95))

    # ── Legend ──
    legend_y = lane_y_bots[-1] - 0.6
    legend_items = [
        (C_MODEL_FILL, C_MODEL, "LLM Call"),
        (C_TOOL_FILL, C_TOOL, "Tool Call"),
    ]
    for name in subagents:
        c, f = AGENT_COLORS.get(name, (C_GRAY, "#EAEDED"))
        legend_items.append((f, c, name.replace("-", " ").title()))

    for j, (lfill, lborder, llabel) in enumerate(legend_items):
        lx = margin_left + j * 3.2
        ax.add_patch(FancyBboxPatch(
            (lx, legend_y - 0.15), 0.5, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=lfill, edgecolor=lborder, linewidth=1.5))
        ax.text(lx + 0.65, legend_y, llabel, va="center", fontsize=7.5,
                color=C_TEXT, fontfamily="sans-serif")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize parallel execution trace")
    parser.add_argument("trace_id", nargs="?", default=None,
                        help="Trace ID (default: latest)")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("-p", "--project", default="strap-agent",
                        help="LangSmith project name")
    args = parser.parse_args()

    client = Client()

    if args.trace_id:
        trace_id = args.trace_id
    else:
        runs = list(client.list_runs(
            project_name=args.project, is_root=True, limit=1))
        if not runs:
            print("No traces found.")
            return
        trace_id = str(runs[0].trace_id)
        print(f"Using latest trace: {trace_id}")

    print(f"Fetching trace {trace_id}...")
    data = fetch_trace_data(client, trace_id)

    print(f"Trace: {data['total_ms']:.0f}ms total, {data['total_tokens']:,} tokens")
    print(f"Subagent lanes: {list(data['subagents'].keys())}")
    print(f"Parallel groups: {len(data['parallel_groups'])}")

    for pg in data["parallel_groups"]:
        print(f"  PARALLEL: {pg[2]} ({pg[0]:.0f}ms - {pg[1]:.0f}ms)")

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "parallel_trace.png"
    )
    draw_parallel_trace(data, output)


if __name__ == "__main__":
    main()
