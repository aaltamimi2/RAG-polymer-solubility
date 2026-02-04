"""Visualize a LangSmith trace as a publication-quality waterfall diagram.

Emphasizes: AI reasoning, tool calls, and their results.
Usage:
    python architecture/visualize_trace.py                          # latest trace
    python architecture/visualize_trace.py <run-id>                 # specific trace
    python architecture/visualize_trace.py <run-id> -o output.png   # custom output
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langsmith import Client

# ── Colours ──────────────────────────────────────────────────
C_BG       = "#FAFAFA"
C_TEXT     = "#2C3E50"
C_AI       = "#2C3E50"      # dark blue-grey
C_AI_FILL  = "#EBF5FB"      # light blue
C_TOOL     = "#27AE60"      # green
C_TOOL_FILL= "#EAFAF1"      # light green
C_USER     = "#8E44AD"      # purple
C_USER_FILL= "#F5EEF8"      # light purple
C_TIME     = "#E74C3C"      # red for time bars
C_TOKEN    = "#F39C12"      # orange for tokens
C_GRAY     = "#BDC3C7"
C_WHITE    = "#FFFFFF"
C_SUBAGENT = "#E67E22"      # orange for subagent calls
C_SUBAGENT_FILL = "#FEF5E7"


def _wrap(text: str, width: int = 80) -> str:
    """Wrap text for display in boxes."""
    if not text:
        return ""
    lines = text.strip().split("\n")
    wrapped = []
    for line in lines[:6]:  # Max 6 lines
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    if len(lines) > 6:
        wrapped.append("...")
    return "\n".join(wrapped[:8])


def _extract_ai_content(run) -> str:
    """Extract AI text content from a model run's output."""
    if not run.outputs:
        return ""
    msgs = run.outputs.get("messages", [])
    if not msgs:
        return ""
    msg = msgs[0] if isinstance(msgs, list) else msgs
    if isinstance(msg, dict):
        content = msg.get("content", "")
        # Also check for tool calls
        tc = msg.get("additional_kwargs", {}).get("function_call", {})
        if tc:
            tool_name = tc.get("name", "")
            if content:
                return f"{content}\n→ calls {tool_name}()"
            return f"→ calls {tool_name}()"
        return content
    return str(msg)[:300]


def _extract_tool_result(run) -> str:
    """Extract tool result snippet."""
    if not run.outputs:
        return ""
    msgs = run.outputs.get("messages", [])
    if msgs:
        msg = msgs[0] if isinstance(msgs, list) else msgs
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content and len(content) > 200:
                return content[:200] + "..."
            return content
    return str(run.outputs)[:200]


def _extract_tool_args(run) -> str:
    """Extract tool input arguments."""
    if not run.inputs:
        return ""
    inp = run.inputs.get("input", "")
    if isinstance(inp, str):
        # Parse the dict-like string
        try:
            import ast
            d = ast.literal_eval(inp)
            if isinstance(d, dict):
                parts = []
                for k, v in d.items():
                    parts.append(f"{k}={v!r}")
                return ", ".join(parts)
        except Exception:
            pass
        return inp[:150]
    return str(inp)[:150]


def fetch_trace_steps(client: Client, trace_id: str):
    """Fetch and organize trace into a list of steps for visualization."""
    all_runs = list(client.list_runs(trace_id=trace_id))
    all_runs.sort(key=lambda r: r.start_time)

    root = next((r for r in all_runs if r.parent_run_id is None), all_runs[0])
    root_start = root.start_time

    steps = []
    for run in all_runs:
        # Only show: model calls (with AI content), tool executions, subagent tasks
        if run.run_type == "chain" and run.name == "model":
            ai_content = _extract_ai_content(run)
            if ai_content:
                elapsed = (run.start_time - root_start).total_seconds() * 1000
                duration = 0
                if run.end_time:
                    duration = (run.end_time - run.start_time).total_seconds() * 1000
                steps.append({
                    "type": "ai",
                    "name": "AI Response",
                    "content": ai_content,
                    "elapsed_ms": elapsed,
                    "duration_ms": duration,
                    "tokens": run.total_tokens or 0,
                })

        elif run.run_type == "tool":
            elapsed = (run.start_time - root_start).total_seconds() * 1000
            duration = 0
            if run.end_time:
                duration = (run.end_time - run.start_time).total_seconds() * 1000

            args = _extract_tool_args(run)
            result = _extract_tool_result(run)

            step_type = "tool"
            if run.name == "task":
                step_type = "subagent"

            steps.append({
                "type": step_type,
                "name": run.name,
                "args": args,
                "result": result,
                "elapsed_ms": elapsed,
                "duration_ms": duration,
            })

    # Add user message at the start
    if root.inputs:
        msgs = root.inputs.get("messages", [])
        if msgs:
            user_content = msgs[0].get("content", "") if isinstance(msgs[0], dict) else str(msgs[0])
            steps.insert(0, {
                "type": "user",
                "name": "User Query",
                "content": user_content,
                "elapsed_ms": 0,
                "duration_ms": 0,
            })

    return steps, root


def draw_trace(steps, root, output_path: str):
    """Draw the waterfall trace diagram."""
    total_duration = 0
    if root.end_time and root.start_time:
        total_duration = (root.end_time - root.start_time).total_seconds() * 1000

    # Layout params
    left_margin = 0.5
    row_height = 1.8
    box_width = 11.5
    time_col_width = 2.5
    fig_width = 15
    fig_height = max(len(steps) * row_height + 3, 8)

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(-fig_height, 1.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Title
    ax.text(fig_width / 2, 0.8, "DISSOLVE Agent — Execution Trace",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C_AI, fontfamily="sans-serif")

    # Subtitle with metadata
    meta = f"Total: {total_duration:.0f}ms | Tokens: {root.total_tokens or '?'} | Model: Gemini 2.0 Flash"
    ax.text(fig_width / 2, 0.15, meta,
            ha="center", va="center", fontsize=9, color=C_GRAY)

    # Column headers
    y_header = -0.4
    ax.text(left_margin + 0.2, y_header, "Step", fontsize=9, fontweight="bold",
            color=C_GRAY, va="center")
    ax.text(left_margin + time_col_width, y_header, "Content", fontsize=9,
            fontweight="bold", color=C_GRAY, va="center")
    ax.text(fig_width - 0.5, y_header, "Time", fontsize=9, fontweight="bold",
            color=C_GRAY, va="center", ha="right")

    # Separator line
    ax.plot([left_margin, fig_width - 0.3], [y_header - 0.25, y_header - 0.25],
            color=C_GRAY, lw=0.5, alpha=0.5)

    # Draw each step
    for i, step in enumerate(steps):
        y_center = -(i * row_height + 1.2)
        stype = step["type"]

        # Pick colours
        if stype == "user":
            fill_color = C_USER_FILL
            border_color = C_USER
            icon = "USER"
            icon_color = C_USER
        elif stype == "ai":
            fill_color = C_AI_FILL
            border_color = C_AI
            icon = "AI"
            icon_color = C_AI
        elif stype == "subagent":
            fill_color = C_SUBAGENT_FILL
            border_color = C_SUBAGENT
            icon = "SUB"
            icon_color = C_SUBAGENT
        else:  # tool
            fill_color = C_TOOL_FILL
            border_color = C_TOOL
            icon = "TOOL"
            icon_color = C_TOOL

        # Step label badge
        badge_w = 1.6
        badge = FancyBboxPatch(
            (left_margin, y_center - 0.3), badge_w, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=border_color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(badge)
        ax.text(left_margin + badge_w / 2, y_center, icon,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=C_WHITE, fontfamily="monospace")

        # Content box
        content_x = left_margin + time_col_width
        content_w = box_width - time_col_width

        if stype == "user":
            display_text = _wrap(step.get("content", ""), width=70)
        elif stype == "ai":
            display_text = _wrap(step.get("content", ""), width=70)
        elif stype == "subagent":
            display_text = f"→ task(agent={step.get('args', '')})"
            result = step.get("result", "")
            if result:
                display_text += "\n" + _wrap(result, width=65)
        else:  # tool
            args = step.get("args", "")
            display_text = f"{step['name']}({args})"
            result = step.get("result", "")
            if result:
                # Show first 2 lines of result
                result_lines = result.split("\\n")[:2]
                display_text += "\n→ " + " | ".join(result_lines)[:120]

        # Content background box
        n_lines = display_text.count("\n") + 1
        box_h = max(0.6, n_lines * 0.28 + 0.2)

        content_box = FancyBboxPatch(
            (content_x, y_center - box_h / 2), content_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=fill_color, edgecolor=border_color,
            linewidth=1, alpha=0.9)
        ax.add_patch(content_box)

        ax.text(content_x + 0.15, y_center, display_text,
                ha="left", va="center", fontsize=7.5, color=C_TEXT,
                fontfamily="monospace", linespacing=1.3)

        # Time info (right column)
        elapsed = step.get("elapsed_ms", 0)
        duration = step.get("duration_ms", 0)
        tokens = step.get("tokens", 0)

        time_parts = []
        if duration > 0:
            time_parts.append(f"{duration:.0f}ms")
        if tokens:
            time_parts.append(f"{tokens} tok")

        if time_parts:
            time_text = " | ".join(time_parts)
            ax.text(fig_width - 0.5, y_center, time_text,
                    ha="right", va="center", fontsize=7.5, color=C_GRAY,
                    fontfamily="monospace")

        # Connecting arrow to next step
        if i < len(steps) - 1:
            arrow_y_start = y_center - box_h / 2 - 0.05
            arrow_y_end = -(((i + 1) * row_height + 1.2)) + 0.4
            ax.annotate("", xy=(left_margin + badge_w / 2, arrow_y_end),
                        xytext=(left_margin + badge_w / 2, arrow_y_start),
                        arrowprops=dict(arrowstyle="->", color=C_GRAY,
                                        lw=1, alpha=0.4))

    # Legend
    legend_y = -(len(steps) * row_height + 1.8)
    legend_items = [
        (C_USER, "User Input"),
        (C_AI, "AI Reasoning"),
        (C_TOOL, "Tool Call"),
        (C_SUBAGENT, "Subagent"),
    ]
    for j, (color, label) in enumerate(legend_items):
        lx = left_margin + j * 3.2
        ax.add_patch(FancyBboxPatch(
            (lx, legend_y - 0.15), 0.4, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=color, linewidth=1))
        ax.text(lx + 0.55, legend_y, label, va="center", fontsize=8, color=C_TEXT)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a LangSmith trace")
    parser.add_argument("run_id", nargs="?", default=None,
                        help="Run ID (default: latest trace)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path")
    parser.add_argument("-p", "--project", default="strap-agent",
                        help="LangSmith project name")
    args = parser.parse_args()

    client = Client()

    if args.run_id:
        trace_id = args.run_id
    else:
        # Get latest trace
        runs = list(client.list_runs(
            project_name=args.project,
            is_root=True,
            limit=1,
        ))
        if not runs:
            print("No traces found in project.")
            return
        trace_id = str(runs[0].trace_id)
        print(f"Using latest trace: {trace_id}")

    steps, root = fetch_trace_steps(client, trace_id)
    print(f"Trace has {len(steps)} visible steps")

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "trace_visualization.png"
    )

    draw_trace(steps, root, output)


if __name__ == "__main__":
    main()
