"""
Visualize DISSOLVE agent single-subagent trace (LDPE/EVOH/PET separation).
New guardrails: tool-call budget (8), synthesis injection, tool-result truncation.
Stacked-card style matching architecture/visualize_sequential_trace.py.
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans", "Arial", "DejaVu Sans",
]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import textwrap
import os


# ── Trace data (LangSmith trace 019c3002-e4eb-7102-85d7-03d65e540bb7) ──
TRACE = {
    "query": (
        "Find the optimal separation sequence for LDPE, EVOH, and PET. "
        "Use selective dissolution at temperatures up to 120C."
    ),
    "total_duration_s": 41,
    "total_tokens_in": 63056,
    "total_tokens_out": 4333,
    "total_runs": 43,
    "orchestrator_llm_calls": 2,
    "step1": {
        "agent": "separation-engineer",
        "duration_s": 24,
        "tokens_in": 47016,
        "llm_calls": 4,
        "tool_calls_total": 8,
        "tool_calls_executed": 6,
        "description": (
            "Find the optimal separation sequence for LDPE, EVOH, and PET "
            "using selective dissolution at temperatures up to 120\u00b0C."
        ),
        "tool_calls": [
            "plan_sequential_separation",
            "find_optimal_separation_conditions",
            "analyze_selective_solubility_enhanced",
        ],
        "guardrail_events": [
            "Synthesis tool detected: plan_sequential_separation",
            "[CRITICAL INSTRUCTION] injected into system prompt",
            "[LIMIT] Tool call budget exhausted (8/8)",
        ],
        "result_summary": (
            "Step 1: EVOH -> DMSO at 100\u00b0C (BP 189\u00b0C)\n"
            "Step 2: LDPE -> p-Xylene at 105\u00b0C (BP 138\u00b0C)\n"
            "Step 3: PET recovered as solid residue\n"
            "All selectivities >10, all temps below BP"
        ),
    },
}

TOOL_SHORT = {
    "plan_sequential_separation": "plan_sequential_separation",
    "find_optimal_separation_conditions": "find_optimal_sep_conditions",
    "analyze_selective_solubility_enhanced": "analyze_selective_solubility",
}


# ── Color palette ────────────────────────────────────────────────────
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"
C_BOX_BORDER = "#E5E7EB"

C_USER       = "#6366F1"   # Indigo
C_ROUTER     = "#22C55E"   # Green
C_SEP_ENG    = "#F97316"   # Orange — separation-engineer
C_SYNTH      = "#0EA5E9"   # Sky blue
C_GUARD      = "#EF4444"   # Red — guardrails

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"   # Light blue
C_GUARD_TEXT = "#1E40AF"   # Blue


# ── Layout constants (7-inch publication width) ──────────────────────
LEFT     = 0.003
MAIN_W   = 0.994
MID      = LEFT + MAIN_W / 2
INPAD    = 0.015
ACCENT_W = 0.006
CL       = LEFT + ACCENT_W + INPAD + 0.004
CR       = LEFT + MAIN_W - INPAD
CW       = CR - CL


# ── Helper functions ─────────────────────────────────────────────────

def draw_quntur_box(ax, x, y, w, h, accent_color, radius=0.005):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(box)
    accent = Rectangle(
        (x + 0.002, y + radius * 0.8),
        ACCENT_W, h - radius * 1.6,
        facecolor=accent_color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(accent)


def draw_divider(ax, x1, x2, y, color="#D1D5DB", lw=0.6):
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            transform=ax.transAxes, zorder=3)


def draw_tool_pills(ax, x, y, tools, max_w):
    col_x = x
    row_y = y
    for tool in tools:
        short = TOOL_SHORT.get(tool, tool)
        tw = len(short) * 0.011 + 0.016
        if col_x + tw > x + max_w:
            col_x = x
            row_y -= 0.028
        pill = FancyBboxPatch(
            (col_x, row_y - 0.010), tw, 0.022,
            boxstyle="round,pad=0.003,rounding_size=0.005",
            facecolor=C_TOOL_BG, edgecolor=C_TOOL_TEXT, linewidth=0.6,
            transform=ax.transAxes, clip_on=False, zorder=4,
        )
        ax.add_patch(pill)
        ax.text(
            col_x + tw / 2, row_y + 0.001, short,
            ha="center", va="center", fontsize=7, color=C_TOOL_TEXT,
            fontfamily="monospace", transform=ax.transAxes, zorder=5,
        )
        col_x += tw + 0.008


def draw_guard_pills(ax, x, y, events, max_w):
    """Draw guardrail event pills in blue style."""
    col_x = x
    row_y = y
    for event in events:
        tw = len(event) * 0.0085 + 0.016
        if col_x + tw > x + max_w:
            col_x = x
            row_y -= 0.028
        pill = FancyBboxPatch(
            (col_x, row_y - 0.010), tw, 0.022,
            boxstyle="round,pad=0.003,rounding_size=0.005",
            facecolor=C_GUARD_BG, edgecolor=C_GUARD_TEXT, linewidth=0.6,
            transform=ax.transAxes, clip_on=False, zorder=4,
        )
        ax.add_patch(pill)
        ax.text(
            col_x + tw / 2, row_y + 0.001, event,
            ha="center", va="center", fontsize=7, color=C_GUARD_TEXT,
            fontfamily="monospace", transform=ax.transAxes, zorder=5,
        )
        col_x += tw + 0.008


def wrap(text, width=90):
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) > width:
            wrapped.extend(textwrap.wrap(line, width=width))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


# ── Main figure ──────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(7, 12.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    # ── Stacked-card layout ──────────────────────────────────────
    y_top = 0.985

    qh  = 0.040    # User query
    ph  = 0.048    # Routing middleware
    s1h = 0.260    # Separation sub-agent (tools + guardrails + output)
    ah  = 0.090    # Synthesized answer

    sections = [
        (qh,  C_USER),
        (ph,  C_ROUTER),
        (s1h, C_SEP_ENG),
        (ah,  C_SYNTH),
    ]
    total_h = sum(h for h, _ in sections)
    stack_bot = y_top - total_h

    # One big rounded container
    big = FancyBboxPatch(
        (LEFT, stack_bot), MAIN_W, total_h,
        boxstyle="round,pad=0,rounding_size=0.005",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(big)

    # Accent bars
    sec_y = y_top
    for i, (h, color) in enumerate(sections):
        sec_y -= h
        ti = 0.004 if i == 0 else 0
        bi = 0.004 if i == len(sections) - 1 else 0
        bar = Rectangle(
            (LEFT + 0.003, sec_y + bi), ACCENT_W, h - ti - bi,
            facecolor=color, edgecolor="none",
            transform=ax.transAxes, clip_on=False, zorder=3,
        )
        ax.add_patch(bar)

    # Dividers between sections
    div_y = y_top
    for h, _ in sections[:-1]:
        div_y -= h
        ax.plot(
            [LEFT, LEFT + MAIN_W], [div_y, div_y],
            color=C_BOX_BORDER, lw=0.8,
            transform=ax.transAxes, zorder=4,
        )

    # Section y-positions
    qy  = y_top - qh
    py  = qy - ph
    s1y = py - s1h
    ay  = s1y - ah

    # ── 1. User Query ────────────────────────────────────────────
    ax.text(
        CL, qy + qh - 0.007, "User",
        fontsize=9.5, fontweight="bold", color=C_USER,
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, qy + qh - 0.018, wrap(TRACE["query"]),
        fontsize=8.5, color=C_BODY, transform=ax.transAxes,
        va="top", zorder=5,
    )

    # ── 2. Routing Middleware ────────────────────────────────────
    ax.text(
        CL, py + ph - 0.006,
        "Single-Agent Routing",
        fontsize=9, fontweight="bold", color="#15803D",
        transform=ax.transAxes, va="top", zorder=5,
    )
    plan_text = (
        'Keyword match: "separat" + "solvent" -> separation-engineer (score 1)\n'
        'Route: Delegate to separation-engineer for separation sequences, '
        'selectivity, dissolution'
    )
    ax.text(
        CL, py + 0.005, plan_text,
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.4,
    )

    # ── 3. Step 1: Separation Sub-Agent ──────────────────────────
    step = TRACE["step1"]
    ax.text(
        CL, s1y + s1h - 0.008,
        "Step 1: Separation Sub-Agent",
        fontsize=10, fontweight="bold", color=C_SEP_ENG,
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CR, s1y + s1h - 0.008,
        f"~{step['duration_s']}s  |  {step['llm_calls']} LLM calls  |  "
        f"{step['tool_calls_total']} tool calls ({step['tool_calls_executed']} executed)",
        fontsize=8, color=C_BODY, ha="right",
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, s1y + s1h - 0.024,
        wrap(step["description"]),
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="top", zorder=5, style="italic",
    )

    # Tool Calls
    ax.text(
        CL, s1y + s1h - 0.050,
        "Tool Calls (deduplicated):",
        fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
        transform=ax.transAxes, va="top", zorder=5,
    )
    draw_tool_pills(
        ax, CL, s1y + s1h - 0.064,
        step["tool_calls"], max_w=CW,
    )

    # Guardrail events
    ax.text(
        CL, s1y + s1h - 0.098,
        "Guardrail Events:",
        fontsize=8, fontweight="bold", color=C_GUARD_TEXT,
        transform=ax.transAxes, va="top", zorder=5,
    )
    draw_guard_pills(
        ax, CL, s1y + s1h - 0.112,
        step["guardrail_events"], max_w=CW,
    )

    # Sub-Agent Output
    draw_divider(ax, CL, CR, s1y + s1h - 0.170)
    ax.text(
        CL, s1y + s1h - 0.176,
        "Sub-Agent Output:",
        fontsize=8, fontweight="bold", color="#9A3412",
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, s1y + 0.006,
        step["result_summary"],
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.35,
    )

    # ── 4. Synthesized Answer ────────────────────────────────────
    ax.text(
        CL, ay + ah - 0.008,
        "Synthesized Answer",
        fontsize=9, fontweight="bold", color="#0369A1",
        transform=ax.transAxes, va="top", zorder=5,
    )
    synth = (
        "Recommended 3-step separation sequence:\n"
        "\n"
        "1. EVOH -> DMSO at 100\u00b0C (BP 189\u00b0C, selectivity >10)\n"
        "2. LDPE -> p-Xylene at 105\u00b0C (BP 138\u00b0C, selectivity >10)\n"
        "3. PET -> recovered as solid residue\n"
        "All steps at atmospheric pressure, temps well below solvent BPs."
    )
    ax.text(
        CL, ay + 0.005, synth,
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.35,
    )

    y = ay - 0.018

    # ── 5. Execution Timeline ────────────────────────────────────
    ax.text(
        0.50, y, "Execution Timeline",
        ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
        transform=ax.transAxes,
    )
    y -= 0.018

    bar_left = LEFT + 0.01
    bar_w = MAIN_W - 0.02
    bar_h = 0.030
    bar_y = y - bar_h

    # Background bar
    bg_bar = FancyBboxPatch(
        (bar_left, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.004",
        facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(bg_bar)

    total_s = 41

    # Orchestrator routing (0-8s)
    orch1_w = (8 / total_s) * bar_w
    ax.add_patch(FancyBboxPatch(
        (bar_left, bar_y), orch1_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.004",
        facecolor=C_USER, edgecolor="none", alpha=0.75,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(
        bar_left + orch1_w / 2, bar_y + bar_h / 2,
        "route",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        transform=ax.transAxes, zorder=4,
    )

    # Separation-engineer (8-32s = 24s)
    sep_start = bar_left + (8 / total_s) * bar_w
    sep_bw = (24 / total_s) * bar_w
    ax.add_patch(FancyBboxPatch(
        (sep_start, bar_y), sep_bw, bar_h,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_SEP_ENG, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(
        sep_start + sep_bw / 2, bar_y + bar_h / 2,
        "separation-engineer (24s)",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        transform=ax.transAxes, zorder=4,
    )

    # Orchestrator synthesis (32-41s)
    synth_start = bar_left + (32 / total_s) * bar_w
    synth_bw = (9 / total_s) * bar_w
    ax.add_patch(FancyBboxPatch(
        (synth_start, bar_y), synth_bw, bar_h,
        boxstyle="round,pad=0,rounding_size=0.004",
        facecolor=C_SYNTH, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(
        synth_start + synth_bw / 2, bar_y + bar_h / 2,
        "synth",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        transform=ax.transAxes, zorder=4,
    )

    # Time labels
    y_time = bar_y - 0.014
    for t_sec in [0, 10, 20, 30, 41]:
        tx = bar_left + (t_sec / total_s) * bar_w
        ax.text(
            tx, y_time, f"{t_sec}s",
            ha="center", fontsize=8, color=C_BODY,
            transform=ax.transAxes,
        )

    # Timeline legend
    y_legend = y_time - 0.022
    legend_items = [
        (C_USER,    "Orchestrator"),
        (C_SEP_ENG, "separation-engineer"),
        (C_SYNTH,   "Synthesis"),
    ]
    lx = bar_left + 0.08
    leg_spacing = (bar_w - 0.16) / 3
    for color, label in legend_items:
        ax.plot(
            [lx, lx + 0.02], [y_legend, y_legend],
            color=color, lw=5, transform=ax.transAxes,
            solid_capstyle="round", zorder=3, alpha=0.85,
        )
        ax.text(
            lx + 0.025, y_legend, label,
            fontsize=8, color=C_BODY, va="center",
            transform=ax.transAxes,
        )
        lx += leg_spacing

    y = y_legend - 0.025

    # ── 6. Bottom Panels ─────────────────────────────────────────
    panel_h = 0.138
    panel_y = y - panel_h
    panel_gap = 0.012
    pw = (MAIN_W - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.010

    # Panel 1: Trace Metadata
    draw_quntur_box(ax, p1x, panel_y, pw, panel_h, C_USER, radius=0.004)
    ax.text(
        p1x + pw / 2, panel_y + panel_h - 0.012,
        "Trace Metadata",
        ha="center", va="top", fontsize=9, fontweight="bold",
        color=C_TITLE, transform=ax.transAxes,
    )
    metrics = [
        (">", "Run Time",       "~41 s"),
        ("#", "Total Tokens",   "67K (63K in)"),
        ("$", "Subagent Calls", "1"),
        ("+", "LLM Calls",     "4 (subagent)"),
        ("~", "Tool Calls",    "8 (6 executed)"),
        ("*", "Pattern",        "single-agent"),
    ]
    my = panel_y + panel_h - 0.036
    for icon, label, value in metrics:
        ax.text(p1x + pcl, my, icon, fontsize=8, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pcl + 0.016, my, label, fontsize=8, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.008, my, value, fontsize=8, fontweight="bold",
                color=C_TITLE, ha="right", transform=ax.transAxes,
                va="center", zorder=5)
        my -= 0.018

    # Panel 2: Execution Pattern
    draw_quntur_box(ax, p2x, panel_y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(
        p2x + pw / 2, panel_y + panel_h - 0.012,
        "Execution Pattern",
        ha="center", va="top", fontsize=9, fontweight="bold",
        color=C_TITLE, transform=ax.transAxes,
    )
    pattern_text = (
        "Single-agent delegation:\n"
        "1. Routing: keyword match\n"
        "   sends to sep-engineer\n"
        "2. Sub-agent: plans sep,\n"
        "   verifies 2 conditions\n"
        "3. Guardrail: synthesis\n"
        "   injection + tool limit\n"
        "4. Orchestrator: formats\n"
        "   final answer for user"
    )
    ax.text(
        p2x + pcl, panel_y + panel_h - 0.034,
        pattern_text,
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="top", zorder=5, linespacing=1.2,
    )

    # Panel 3: Active Guardrails
    draw_quntur_box(ax, p3x, panel_y, pw, panel_h, C_GUARD, radius=0.004)
    ax.text(
        p3x + pw / 2, panel_y + panel_h - 0.012,
        "Active Guardrails",
        ha="center", va="top", fontsize=9, fontweight="bold",
        color=C_TITLE, transform=ax.transAxes,
    )
    guardrails = [
        "[x] Tool-call budget",
        "    (8 max, hit at 8)",
        "[x] Synthesis injection",
        "    (plan_sequential_sep)",
        "[x] Tool-result truncation",
        "    (2000 chars after iter 3)",
        "[x] Token budget 200K",
        "[x] Iteration limit 25",
    ]
    gy = panel_y + panel_h - 0.036
    for g in guardrails:
        ax.text(
            p3x + pcl, gy, g,
            fontsize=8, color="#166534", transform=ax.transAxes,
            va="top", zorder=5,
        )
        gy -= 0.0125

    # ── Footer ───────────────────────────────────────────────────
    ax.text(
        0.50, panel_y - 0.012,
        "DISSOLVE: Data Integrated Solubility Solver via LLM Evaluation  |  "
        "Model: Gemini 3 Flash Preview  |  "
        "Query: LDPE/EVOH/PET Single-Agent Separation",
        ha="center", fontsize=8, color=C_BODY,
        transform=ax.transAxes,
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "chain_trace.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
