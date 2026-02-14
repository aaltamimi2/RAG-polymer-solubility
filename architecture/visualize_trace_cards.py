"""Visualize a LangSmith trace as a stacked-card diagram.

Uses the polished style from the static trace scripts (trace7, trace6, etc.)
but fetches data dynamically from LangSmith.

Usage:
    python architecture/visualize_trace_cards.py                          # latest trace
    python architecture/visualize_trace_cards.py <trace-id>               # specific trace
    python architecture/visualize_trace_cards.py <trace-id> -o output.png # custom output
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans", "Arial", "DejaVu Sans",
]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langsmith import Client


# ── Color palette (matches trace7 style) ─────────────────────────────

C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"
C_BOX_BORDER = "#E5E7EB"

C_USER       = "#6366F1"   # Indigo
C_ROUTER     = "#22C55E"   # Green
C_SYNTH      = "#8B5CF6"   # Violet
C_WARN       = "#F59E0B"   # Amber
C_DELTA      = "#0EA5E9"   # Sky

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"

# Subagent accent colors
AGENT_COLORS = {
    "separation-engineer":      "#F97316",   # Orange
    "safety-analyst":           "#3B82F6",   # Blue
    "scholar-researcher":       "#F59E0B",   # Amber
    "patent-researcher":        "#EF4444",   # Red
    "rag-analyst":              "#14B8A6",   # Teal
    "visualization-specialist": "#8B5CF6",   # Violet
    "statistics-ml":            "#6366F1",   # Indigo
    "biosteam-analyst":         "#2D8B72",   # Forest green
}


# ── Layout constants ─────────────────────────────────────────────────

FIG_W    = 7.0
LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
GAP      = 0.008
PILL_H   = 0.014      # vertical space per pill row
LINE_H   = 0.009      # vertical space per text line
HEADER_H = 0.012      # space for section header
HEADER_GAP = 0.004    # gap between header and body text

# Compute wrap width from box geometry + font metrics.
# Text area: cl..cr in axes fraction → physical inches → chars.
_TEXT_AREA_FRAC = (RIGHT - PAD) - (LEFT + ACCENT_W + PAD)  # ~0.944
_TEXT_AREA_INCHES = _TEXT_AREA_FRAC * FIG_W                # ~6.6"
# DejaVu Sans / Liberation Sans at 5.5pt: empirical avg char width ~5.1pt
_AVG_CHAR_PTS = 5.1
WRAP_CHARS = int(_TEXT_AREA_INCHES * 72 / _AVG_CHAR_PTS)   # ~93


# ── Visual primitives (from trace7) ─────────────────────────────────

def _box(ax, x, y_top, w, h, accent_color, radius=0.005):
    y_bot = y_top - h
    box = FancyBboxPatch(
        (x, y_bot), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(box)
    bar = Rectangle(
        (x + 0.002, y_bot + radius), ACCENT_W, h - 2 * radius,
        facecolor=accent_color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(bar)
    return y_bot


def _divider(ax, x1, x2, y, color="#D1D5DB", lw=0.6):
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            transform=ax.transAxes, zorder=3)


def _pill(ax, x, y_center, text, fg, bg, border, fs=5.5, mono=True):
    tw = len(text) * (0.0070 if mono else 0.0060) + 0.014
    pill = FancyBboxPatch(
        (x, y_center - 0.005), tw, 0.010,
        boxstyle="round,pad=0.002,rounding_size=0.003",
        facecolor=bg, edgecolor=border, linewidth=0.6,
        transform=ax.transAxes, clip_on=False, zorder=4,
    )
    ax.add_patch(pill)
    ax.text(
        x + tw / 2, y_center, text,
        ha="center", va="center", fontsize=fs, color=fg,
        fontfamily="monospace" if mono else "sans-serif",
        transform=ax.transAxes, zorder=5,
    )
    return tw


def _pills_row(ax, x_start, y_center, labels, max_w, fg, bg, border, fs=5.0):
    """Layout pills in rows, wrapping when exceeding max_w. Returns number of rows."""
    cx = x_start
    rows = 1
    for label in labels:
        tw = len(label) * 0.0070 + 0.014
        if cx + tw > x_start + max_w and cx > x_start:
            cx = x_start
            y_center -= PILL_H
            rows += 1
        _pill(ax, cx, y_center, label, fg, bg, border, fs)
        cx += tw + 0.006
    return rows


def _sanitize(text: str) -> str:
    """Strip characters that cause font/rendering issues."""
    import re
    return re.sub(r'[^\x20-\x7E\xA0-\xFF\u00B0\u00B1\u00B2\u00B3\u00B5\u00B7]', '?', text)


def _wrap(text, width=95):
    if not text:
        return ""
    if not isinstance(text, str):
        text = _extract_text(text)
    text = _sanitize(text)
    lines = text.split("\n")
    out = []
    for line in lines[:8]:
        out.extend(textwrap.wrap(line, width=width) if len(line) > width else [line])
    if len(lines) > 8:
        out.append("...")
    return "\n".join(out[:10])


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def _count_pill_rows(tools, cl, cr):
    """Pre-calculate how many rows of pills will be needed."""
    if not tools:
        return 1
    cx = cl
    max_w = cr - cl
    rows = 1
    for tool in tools:
        t_dur = tool["duration_ms"] / 1000
        if t_dur >= 1:
            label = f"{tool['name']} ({t_dur:.1f}s)"
        else:
            label = f"{tool['name']} ({tool['duration_ms']:.0f}ms)"
        tw = len(label) * 0.0070 + 0.014
        if cx + tw > cl + max_w and cx > cl:
            cx = cl
            rows += 1
        cx += tw + 0.006
    return rows


def _count_label_rows(labels, cl, cr):
    """Pre-calculate how many rows of pills needed for plain string labels."""
    if not labels:
        return 1
    cx = cl
    max_w = cr - cl
    rows = 1
    for label in labels:
        tw = len(label) * 0.0070 + 0.014
        if cx + tw > cl + max_w and cx > cl:
            cx = cl
            rows += 1
        cx += tw + 0.006
    return rows


# ── Trace data fetching ──────────────────────────────────────────────

def fetch_trace_structure(client: Client, trace_id: str) -> dict:
    """Fetch a LangSmith trace and organize it into card-friendly structure."""
    all_runs = list(client.list_runs(trace_id=trace_id))
    all_runs.sort(key=lambda r: r.start_time)

    root = next((r for r in all_runs if r.parent_run_id is None), all_runs[0])
    root_start = root.start_time
    run_map = {str(r.id): r for r in all_runs}

    def ms(dt):
        return (dt - root_start).total_seconds() * 1000 if dt else 0

    total_ms = ms(root.end_time) if root.end_time else 0

    # Extract user query
    user_query = ""
    if root.inputs:
        msgs = root.inputs.get("messages", [])
        if msgs:
            msg = msgs[0]
            if isinstance(msg, dict):
                user_query = _extract_text(msg.get("content", ""))
            else:
                user_query = str(msg)

    # Find task() calls (subagent dispatches)
    task_runs = [r for r in all_runs if r.name == "task" and r.run_type == "tool"]

    subagents = []
    for tr in task_runs:
        inp = str(tr.inputs) if tr.inputs else ""
        agent_name = "unknown"
        for name in AGENT_COLORS:
            if name in inp:
                agent_name = name
                break

        start = ms(tr.start_time)
        dur = ms(tr.end_time) - start if tr.end_time else 0

        # Find tools called within this subagent
        tools_used = []
        subagent_chain_id = None
        for r in all_runs:
            if (r.parent_run_id and str(r.parent_run_id) == str(tr.id)
                    and r.run_type == "chain"):
                subagent_chain_id = str(r.id)
                break

        if subagent_chain_id:
            for r in all_runs:
                if r.run_type == "tool" and r.name not in ("task", "write_todos", "think"):
                    pid = r.parent_run_id
                    depth = 0
                    seen = set()
                    is_descendant = False
                    while pid and str(pid) not in seen and depth < 10:
                        seen.add(str(pid))
                        if str(pid) == subagent_chain_id:
                            is_descendant = True
                            break
                        parent = run_map.get(str(pid))
                        pid = parent.parent_run_id if parent else None
                        depth += 1
                    if is_descendant:
                        t_dur = ms(r.end_time) - ms(r.start_time) if r.end_time else 0
                        tools_used.append({
                            "name": r.name,
                            "duration_ms": t_dur,
                        })

        subagents.append({
            "name": agent_name,
            "start_ms": start,
            "duration_ms": dur,
            "tools": tools_used,
        })

    # Find orchestrator core tool calls (non-task, non-write_todos at top level)
    orch_tools = []
    for r in all_runs:
        if r.run_type == "tool" and r.name not in ("task", "write_todos", "think"):
            parent = run_map.get(str(r.parent_run_id)) if r.parent_run_id else None
            if parent and parent.name == "tools":
                gp = run_map.get(str(parent.parent_run_id)) if parent.parent_run_id else None
                if gp and gp.parent_run_id is None:
                    t_dur = ms(r.end_time) - ms(r.start_time) if r.end_time else 0
                    orch_tools.append({
                        "name": r.name,
                        "start_ms": ms(r.start_time),
                        "duration_ms": t_dur,
                    })

    # Detect parallel dispatch
    is_parallel = False
    if len(task_runs) >= 2:
        starts = [ms(tr.start_time) for tr in task_runs]
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                if abs(starts[i] - starts[j]) < 500:
                    is_parallel = True
                    break

    # Extract final answer (truncated)
    final_answer = ""
    if root.outputs:
        msgs = root.outputs.get("messages") or root.outputs.get("output", "")
        if isinstance(msgs, list):
            for msg in reversed(msgs):
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if content:
                        final_answer = _extract_text(content)
                        break

    return {
        "trace_id": trace_id,
        "user_query": user_query,
        "total_ms": total_ms,
        "total_tokens": root.total_tokens or 0,
        "n_messages": len(all_runs),
        "subagents": subagents,
        "orch_tools": orch_tools,
        "is_parallel": is_parallel,
        "final_answer": final_answer[:600],
        "root": root,
    }


# ── Card-based rendering ─────────────────────────────────────────────

def draw_trace_cards(data: dict, output_path: str, title: str | None = None):
    """Draw a stacked-card trace diagram."""
    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD

    # Pre-compute figure height — generous estimate, we'll trim later
    n_subagents = len(data["subagents"])
    n_orch_tools = len(data["orch_tools"])
    fig_h = (
        1.2         # header + subtitle
        + 1.0       # user query
        + 0.6       # routing
        + n_orch_tools * 0.5
        + sum(1.0 + min(len(sa["tools"]), 6) * 0.3 for sa in data["subagents"])
        + 1.2       # synthesis
        + 1.2       # timeline
        + 1.5       # bottom panel + footer
        + 0.5       # padding
    )
    fig_h = max(8, min(fig_h, 45))

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    y = 0.993

    # ── HEADER ──
    total_s = data["total_ms"] / 1000
    total_tok = data["total_tokens"]
    n_sub = len(data["subagents"])
    n_tools = len(data["orch_tools"]) + sum(len(sa["tools"]) for sa in data["subagents"])

    header = title or "DISSOLVE Agent — Execution Trace"
    ax.text(MID, y, header,
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.014

    pattern = "Parallel" if data["is_parallel"] else "Sequential"
    if n_sub == 0:
        pattern = "Orchestrator-Only"
    elif n_sub == 1:
        pattern = "Single Subagent"

    subtitle = (
        f"{total_s:.1f}s  |  {total_tok:,} tokens  |  "
        f"{n_sub} subagent{'s' if n_sub != 1 else ''}  |  "
        f"{n_tools} tool calls  |  {pattern}"
    )
    ax.text(MID, y, subtitle,
            ha="center", fontsize=7, color=C_BODY,
            transform=ax.transAxes)
    y -= 0.020

    # ── USER QUERY ──
    query_text = _wrap(data["user_query"], WRAP_CHARS)
    n_lines = query_text.count("\n") + 1
    box_h = HEADER_H + HEADER_GAP + n_lines * LINE_H + 0.010
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.006
    ax.text(cl, cy, "User Query",
            fontsize=8, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= HEADER_H + HEADER_GAP
    ax.text(cl, cy, query_text,
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ── ROUTING ──
    agent_names = [sa["name"] for sa in data["subagents"]]
    if agent_names:
        routing_text = " -> ".join(agent_names) if not data["is_parallel"] else " || ".join(agent_names)
    else:
        routing_text = "Orchestrator handled directly (no subagent delegation)"

    box_h = HEADER_H + HEADER_GAP + LINE_H + 0.010
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    cy = y - 0.006
    pattern_label = "Parallel" if data["is_parallel"] else ("Sequential" if n_sub > 1 else "Single-Agent")
    ax.text(cl, cy, f"{pattern_label} Routing",
            fontsize=8, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= HEADER_H + HEADER_GAP
    ax.text(cl, cy, routing_text,
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ── ORCHESTRATOR TOOLS (if any) ──
    if data["orch_tools"]:
        pill_rows = _count_pill_rows(data["orch_tools"], cl, cr)
        box_h = HEADER_H + HEADER_GAP + pill_rows * PILL_H + 0.010
        _box(ax, LEFT, y, WIDTH, box_h, C_USER)
        cy = y - 0.006
        ax.text(cl, cy, "Orchestrator Tools",
                fontsize=8, fontweight="bold", color=C_USER,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= HEADER_H + HEADER_GAP

        tool_labels = []
        for tool in data["orch_tools"]:
            dur_s = tool["duration_ms"] / 1000
            label = f"{tool['name']}  ({dur_s:.1f}s)" if dur_s >= 1 else f"{tool['name']}  ({tool['duration_ms']:.0f}ms)"
            tool_labels.append(label)
        _pills_row(ax, cl, cy, tool_labels, cr - cl, C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT)

        y -= box_h + GAP

    # ── SUBAGENT BOXES ──
    for i, sa in enumerate(data["subagents"], 1):
        color = AGENT_COLORS.get(sa["name"], "#6B7280")
        n_sa_tools = len(sa["tools"])
        dur_s = sa["duration_ms"] / 1000

        # Calculate pill rows needed
        pill_rows = _count_pill_rows(sa["tools"], cl, cr) if sa["tools"] else 1
        box_h = HEADER_H + HEADER_GAP + pill_rows * PILL_H + 0.010
        _box(ax, LEFT, y, WIDTH, box_h, color)
        cy = y - 0.006

        ax.text(cl, cy, f"{i}. {sa['name']}",
                fontsize=8, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cr, cy,
                f"{dur_s:.1f}s  |  {n_sa_tools} tool call{'s' if n_sa_tools != 1 else ''}",
                fontsize=6, color=C_BODY, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= HEADER_H + HEADER_GAP

        # Tool pills
        if sa["tools"]:
            tool_labels = []
            for tool in sa["tools"]:
                t_dur = tool["duration_ms"] / 1000
                if t_dur >= 1:
                    label = f"{tool['name']} ({t_dur:.1f}s)"
                else:
                    label = f"{tool['name']} ({tool['duration_ms']:.0f}ms)"
                tool_labels.append(label)
            _pills_row(ax, cl, cy, tool_labels, cr - cl, C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT)
        else:
            ax.text(cl, cy, "(no domain tool calls recorded)",
                    fontsize=5.5, color="#9CA3AF", transform=ax.transAxes,
                    va="center", zorder=5)

        y -= box_h + GAP

    # ── SYNTHESIS ──
    raw_answer = _sanitize(data["final_answer"][:400])
    answer_preview = _wrap(raw_answer, WRAP_CHARS)[:300]
    if len(raw_answer) > 300:
        answer_preview += "..."
    n_lines = answer_preview.count("\n") + 1
    box_h = HEADER_H + HEADER_GAP + n_lines * LINE_H + 0.010
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.006
    ax.text(cl, cy, "Final Synthesis",
            fontsize=8, fontweight="bold", color=C_SYNTH,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= HEADER_H + HEADER_GAP
    if answer_preview:
        ax.text(cl, cy, answer_preview,
                fontsize=5, color=C_BODY, transform=ax.transAxes,
                va="top", zorder=5, linespacing=1.3)
    else:
        ax.text(cl, cy, "(answer not captured in trace output)",
                fontsize=5.5, color="#9CA3AF", transform=ax.transAxes,
                va="top", zorder=5)
    y -= box_h + GAP

    # ── EXECUTION TIMELINE ──
    y -= 0.005  # extra breathing room
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.016

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.016
    total_ms = max(data["total_ms"], 1)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)
    bar_y = y - bar_h
    ax.add_patch(FancyBboxPatch((bar_left, bar_y), bar_w, bar_h,
                                transform=ax.transAxes, zorder=2, **bg_kw))

    def _bar_seg(t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_ms) * bar_w
        w = max((t_dur / total_ms) * bar_w, 0.002)
        ax.add_patch(FancyBboxPatch(
            (x0, bar_y), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if w > 0.06:
            ax.text(x0 + w / 2, bar_y + bar_h / 2, label,
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    # Orchestrator tool segments
    for tool in data["orch_tools"]:
        _bar_seg(tool["start_ms"], tool["duration_ms"], C_USER,
                 tool["name"].replace("rank_solvents_", ""))

    # Subagent segments
    for sa in data["subagents"]:
        color = AGENT_COLORS.get(sa["name"], "#6B7280")
        short_name = sa["name"].split("-")[0][:6]
        _bar_seg(sa["start_ms"], sa["duration_ms"], color, short_name)

    # Time labels
    ty = bar_y - 0.010
    nice = 5000 if total_ms <= 30000 else 10000 if total_ms <= 120000 else 30000
    t = 0
    while t <= total_ms + 1:
        tx = bar_left + (t / total_ms) * bar_w
        ax.text(tx, ty, f"{t/1000:.0f}s",
                ha="center", fontsize=5.5, color=C_BODY,
                transform=ax.transAxes)
        t += nice

    # Timeline legend
    ly = ty - 0.014
    legend_items = [(C_USER, "Orchestrator")]
    for sa in data["subagents"]:
        color = AGENT_COLORS.get(sa["name"], "#6B7280")
        legend_items.append((color, sa["name"]))
    lx = bar_left + 0.02
    sp = min((bar_w - 0.04) / max(len(legend_items), 1), 0.18)
    for color, label in legend_items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=5.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.020

    # ── BOTTOM PANEL: Metadata ──
    panel_h = 0.060
    _box(ax, LEFT, y, WIDTH, panel_h, C_DELTA, radius=0.004)
    cy = y - 0.006
    ax.text(cl, cy, "Trace Metadata",
            fontsize=8, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= HEADER_H + 0.002

    metrics_left = [
        ("Trace ID", data["trace_id"][:36]),
        ("Run Time", f"{total_s:.1f}s"),
        ("Total Tokens", f"{total_tok:,}"),
    ]
    metrics_right = [
        ("Subagents", str(n_sub)),
        ("Tool Calls", str(n_tools)),
        ("Pattern", pattern),
    ]

    cy_l = cy
    for label, value in metrics_left:
        ax.text(cl, cy_l, f"{label}:", fontsize=5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(cl + 0.070, cy_l, value, fontsize=5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="center", zorder=5)
        cy_l -= 0.009

    cy_r = cy
    mid_x = MID + 0.1
    for label, value in metrics_right:
        ax.text(mid_x, cy_r, f"{label}:", fontsize=5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(mid_x + 0.070, cy_r, value, fontsize=5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="center", zorder=5)
        cy_r -= 0.009

    y -= panel_h + 0.008

    # ── Footer ──
    ax.text(MID, y,
            "DISSOLVE v8  |  Gemini 2.5 Pro + Flash  |  "
            f"{'Parallel' if data['is_parallel'] else 'Sequential'} Execution",
            ha="center", fontsize=6, color=C_BODY,
            transform=ax.transAxes)

    # Resize figure so all content (drawn in transAxes 0-1 coords) fits.
    # Content runs from y (bottom) to ~0.993 (top). If y < 0, content
    # overflows the default (0,1) axes box. We reposition the axes within
    # the figure so (y_bottom, y_top) in transAxes maps to the full figure.
    y_bottom = y - 0.02
    y_top    = 1.005
    content_span = y_top - y_bottom  # axes-fraction units used

    # Physical height: keep same scale as the initial estimate per axes-unit
    actual_h = content_span * fig_h
    actual_h = max(8, min(actual_h, 50))
    fig.set_size_inches(FIG_W, actual_h)

    # Reposition axes so that transAxes y_bottom..y_top fills the figure.
    # In figure-fraction coords the axes box [left, bottom, width, height]:
    #   bottom = (0 - y_bottom) / content_span = -y_bottom / content_span
    #   height = 1.0 / content_span
    ax_bottom = -y_bottom / content_span
    ax_height = 1.0 / content_span
    ax.set_position([0, ax_bottom, 1, ax_height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Save without bbox_inches="tight" to avoid matplotlib expanding the
    # figure to include artists outside the axes (the root cause of the
    # "119M pixel" figure-size errors).
    fig.savefig(output_path, dpi=200, facecolor=C_BG, pad_inches=0)
    plt.close()
    print(f"Saved: {output_path}")


# ── Compact card-based rendering ──────────────────────────────────────

def draw_compact_trace(data: dict, output_path: str, title: str | None = None):
    """Draw a compact stacked-card trace diagram (~40% shorter than full)."""

    # Tighter layout constants
    _gap = 0.004
    _line_h = 0.007
    _header_h = 0.009
    _header_gap = 0.002
    _pill_h = 0.011

    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD

    # Pre-compute figure height (generous, trimmed later)
    n_subagents = len(data["subagents"])
    fig_h = (
        0.8
        + 0.4         # header + subtitle (with inline routing)
        + 0.4         # user query (2 lines max)
        + sum(0.6 + min(len(sa["tools"]), 8) * 0.15 for sa in data["subagents"])
        + 0.5         # synthesis (3 lines max)
        + 0.8         # timeline
        + 0.3         # footer
    )
    fig_h = max(6, min(fig_h, 40))

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    y = 0.993

    # ── HEADER with inline metrics + routing ──
    total_s = data["total_ms"] / 1000
    total_tok = data["total_tokens"]
    n_sub = len(data["subagents"])
    n_tools = len(data["orch_tools"]) + sum(len(sa["tools"]) for sa in data["subagents"])

    header = title or "DISSOLVE Agent — Execution Trace"
    ax.text(MID, y, header,
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    # Inline routing pattern in subtitle
    pattern = "Parallel" if data["is_parallel"] else "Sequential"
    if n_sub == 0:
        pattern = "Orchestrator-Only"
    elif n_sub == 1:
        pattern = "Single Subagent"

    agent_names = [sa["name"] for sa in data["subagents"]]
    if agent_names:
        route_str = (" || " if data["is_parallel"] else " -> ").join(agent_names)
    else:
        route_str = "direct"

    subtitle = (
        f"{total_s:.1f}s  |  {total_tok:,} tok  |  "
        f"{n_tools} tools  |  {pattern}: {route_str}"
    )
    ax.text(MID, y, subtitle,
            ha="center", fontsize=5.5, color=C_BODY,
            transform=ax.transAxes)
    y -= 0.014

    # ── USER QUERY (2 lines max) ──
    query_text = _wrap(data["user_query"], WRAP_CHARS)
    query_lines = query_text.split("\n")[:2]
    if len(query_text.split("\n")) > 2:
        query_lines[-1] = query_lines[-1][:80] + "..."
    query_text = "\n".join(query_lines)
    n_lines = len(query_lines)

    box_h = _header_h + _header_gap + n_lines * _line_h + 0.006
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.004
    ax.text(cl, cy, "Query",
            fontsize=7, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= _header_h + _header_gap
    ax.text(cl, cy, query_text,
            fontsize=5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + _gap

    # ── SUBAGENT BOXES (name-only pills, no durations) ──
    for i, sa in enumerate(data["subagents"], 1):
        color = AGENT_COLORS.get(sa["name"], "#6B7280")
        n_sa_tools = len(sa["tools"])
        dur_s = sa["duration_ms"] / 1000

        # Name-only pill labels (no duration)
        tool_labels = [tool["name"] for tool in sa["tools"]] if sa["tools"] else []
        pill_rows = _count_label_rows(tool_labels, cl, cr) if tool_labels else 1

        box_h = _header_h + _header_gap + pill_rows * _pill_h + 0.006
        _box(ax, LEFT, y, WIDTH, box_h, color)
        cy = y - 0.004

        ax.text(cl, cy, f"{i}. {sa['name']}",
                fontsize=7, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cr, cy,
                f"{dur_s:.1f}s  |  {n_sa_tools} tools",
                fontsize=5, color=C_BODY, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= _header_h + _header_gap

        if tool_labels:
            _pills_row(ax, cl, cy, tool_labels, cr - cl,
                       C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=4.5)
        else:
            ax.text(cl, cy, "(no domain tools)",
                    fontsize=4.5, color="#9CA3AF", transform=ax.transAxes,
                    va="center", zorder=5)

        y -= box_h + _gap

    # ── SYNTHESIS (3 lines max) ──
    raw_answer = _sanitize(data["final_answer"][:300])
    answer_preview = _wrap(raw_answer, WRAP_CHARS)
    answer_lines = answer_preview.split("\n")[:3]
    if len(answer_preview.split("\n")) > 3:
        answer_lines[-1] = answer_lines[-1][:80] + "..."
    answer_preview = "\n".join(answer_lines)
    n_lines = len(answer_lines)

    box_h = _header_h + _header_gap + n_lines * _line_h + 0.006
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.004
    ax.text(cl, cy, "Synthesis",
            fontsize=7, fontweight="bold", color=C_SYNTH,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= _header_h + _header_gap
    if answer_preview:
        ax.text(cl, cy, answer_preview,
                fontsize=4.5, color=C_BODY, transform=ax.transAxes,
                va="top", zorder=5, linespacing=1.2)
    else:
        ax.text(cl, cy, "(answer not captured)",
                fontsize=4.5, color="#9CA3AF", transform=ax.transAxes,
                va="top", zorder=5)
    y -= box_h + _gap

    # ── EXECUTION TIMELINE (thinner bar) ──
    y -= 0.003
    ax.text(MID, y, "Timeline",
            ha="center", fontsize=7, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.012
    total_ms = max(data["total_ms"], 1)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)
    bar_y = y - bar_h
    ax.add_patch(FancyBboxPatch((bar_left, bar_y), bar_w, bar_h,
                                transform=ax.transAxes, zorder=2, **bg_kw))

    def _bar_seg(t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_ms) * bar_w
        w = max((t_dur / total_ms) * bar_w, 0.002)
        ax.add_patch(FancyBboxPatch(
            (x0, bar_y), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if w > 0.08:
            ax.text(x0 + w / 2, bar_y + bar_h / 2, label,
                    ha="center", va="center", fontsize=4.5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    for tool in data["orch_tools"]:
        _bar_seg(tool["start_ms"], tool["duration_ms"], C_USER,
                 tool["name"].replace("rank_solvents_", ""))

    for sa in data["subagents"]:
        color = AGENT_COLORS.get(sa["name"], "#6B7280")
        short_name = sa["name"].split("-")[0][:6]
        _bar_seg(sa["start_ms"], sa["duration_ms"], color, short_name)

    # Time labels
    ty = bar_y - 0.008
    nice = 5000 if total_ms <= 30000 else 10000 if total_ms <= 120000 else 30000
    t = 0
    while t <= total_ms + 1:
        tx = bar_left + (t / total_ms) * bar_w
        ax.text(tx, ty, f"{t/1000:.0f}s",
                ha="center", fontsize=5, color=C_BODY,
                transform=ax.transAxes)
        t += nice

    y = ty - 0.010

    # ── FOOTER (1 line) ──
    trace_short = data["trace_id"][:12] if data["trace_id"] else "?"
    ax.text(MID, y,
            f"DISSOLVE v8  |  Trace {trace_short}  |  "
            f"{'Parallel' if data['is_parallel'] else 'Sequential'}",
            ha="center", fontsize=5, color=C_BODY,
            transform=ax.transAxes)

    # Resize figure to fit content (same logic as draw_trace_cards)
    y_bottom = y - 0.015
    y_top = 1.005
    content_span = y_top - y_bottom

    actual_h = content_span * fig_h
    actual_h = max(4, min(actual_h, 40))
    fig.set_size_inches(FIG_W, actual_h)

    ax_bottom = -y_bottom / content_span
    ax_height = 1.0 / content_span
    ax.set_position([0, ax_bottom, 1, ax_height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.savefig(output_path, dpi=200, facecolor=C_BG, pad_inches=0)
    plt.close()
    print(f"Saved compact: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize trace as stacked cards")
    parser.add_argument("trace_id", nargs="?", default=None,
                        help="Trace ID (default: latest)")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("-p", "--project", default="strap-agent",
                        help="LangSmith project name")
    parser.add_argument("-t", "--title", default=None,
                        help="Custom title for the figure")
    parser.add_argument("--compact", action="store_true",
                        help="Generate compact version (~40%% shorter)")
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
    data = fetch_trace_structure(client, trace_id)

    print(f"Trace: {data['total_ms']/1000:.1f}s, {data['total_tokens']:,} tokens")
    print(f"Subagents: {[sa['name'] for sa in data['subagents']]}")

    default_name = "trace_compact.png" if args.compact else "trace_cards.png"
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), default_name)

    if args.compact:
        draw_compact_trace(data, output, title=args.title)
    else:
        draw_trace_cards(data, output, title=args.title)


if __name__ == "__main__":
    main()
