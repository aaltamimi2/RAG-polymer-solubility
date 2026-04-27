"""
Visualize DISSOLVE agent trace in publication-quality style.
Inspired by El Agente Quntur (arXiv:2602.04850v1) grey agent output boxes.
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


# ── Trace data (from LangSmith trace 019c2ef5) ──────────────────────────
TRACE = {
    "query": (
        "Plan a separation sequence for HDPE and PS at 100\u00b0C, then estimate "
        "the operating cost and payback period for the solvents used."
    ),
    "total_duration_s": 75,
    "total_tokens": 192_891,
    "total_runs": 116,
    "orchestrator_llm_calls": 3,
    "step1": {
        "agent": "separation-engineer",
        "duration_s": 37,
        "tokens": 113_994,
        "tool_calls": [
            "analyze_selective_solubility_enhanced",
            "analyze_selective_solubility_enhanced",
            "rank_solvents_for_separation",
            "calculate_selectivity_detailed",
            "calculate_selectivity_detailed",
        ],
        "description": (
            "Plan a separation sequence for HDPE and PS at 100\u00b0C. Identify "
            "suitable solvents that can selectively dissolve one polymer while "
            "leaving the other as a solid. Ensure BP > 100\u00b0C."
        ),
        "result_summary": (
            "Selective dissolution of PS at 100\u00b0C.\n"
            "Primary: DMF (selectivity 28.0\u00d7, BP 153\u00b0C)\n"
            "Secondary: Toluene (selectivity 10.2\u00d7, BP 111\u00b0C)\n"
            "HDPE remains insoluble (\u22640% at 100\u00b0C)."
        ),
    },
    "step2": {
        "agent": "tea-lca-analyst",
        "duration_s": 26,
        "tokens": 49_961,
        "tool_calls": [
            "compare_solvents_tea_lca",
            "analyze_solvent_recovery_tea",
            "analyze_solvent_recovery_tea",
        ],
        "description": (
            "Estimate operating cost and payback period for the HDPE/PS "
            "separation using DMF and Toluene. Assume 100 kg/hr throughput, "
            "95% recovery."
        ),
        "result_summary": (
            "DMF: OPEX $1.75/kg, payback 0.28 yr (3.4 mo)\n"
            "Toluene: OPEX $1.42/kg, payback 0.52 yr (6.2 mo)\n"
            "Toluene: 32% lower carbon footprint (709 vs 1045 t CO2e/yr)"
        ),
    },
}

# Short display names for tool calls
TOOL_SHORT = {
    "analyze_selective_solubility_enhanced": "analyze_selective_solubility",
    "calculate_selectivity_detailed": "calculate_selectivity",
    "rank_solvents_for_separation": "rank_solvents",
    "compare_solvents_tea_lca": "compare_solvents_tea_lca",
    "analyze_solvent_recovery_tea": "analyze_solvent_recovery",
}


# ── Color palette (Quntur-inspired) ─────────────────────────────────────
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"   # Light grey box fill
C_BOX_BORDER = "#E5E7EB"   # Subtle border

# Accent colors (left bar + role icon)
C_USER       = "#6366F1"   # Indigo
C_ROUTER     = "#22C55E"   # Green
C_SEP_ENG    = "#F97316"   # Orange
C_TEA_LCA    = "#EC4899"   # Pink
C_SYNTH      = "#0EA5E9"   # Sky blue

# Text — all text uses C_BODY or C_TITLE (no grey)
C_TITLE      = "#1E293B"
C_BODY       = "#374151"

# Tool pills
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"

# Arrows (non-text, so grey is OK)
C_ARROW      = "#000000"


# ── Layout constants (optimised for 7-inch width) ───────────────────────
LEFT     = 0.003           # near-zero left margin
MAIN_W   = 0.994           # boxes span ~6.96 inches at 7" width
MID      = LEFT + MAIN_W / 2
INPAD    = 0.015           # internal padding (tighter for narrow figure)
ACCENT_W = 0.006           # left accent bar width
CL       = LEFT + ACCENT_W + INPAD + 0.004  # content left edge
CR       = LEFT + MAIN_W - INPAD             # content right edge
CW       = CR - CL                           # content width

# Spacing
ROLE_GAP  = 0.008          # gap between role label and its box (tight)
ARROW_GAP = 0.026          # gap between previous box and next role label (loose)


# ── Helper functions ────────────────────────────────────────────────────

def draw_quntur_box(ax, x, y, w, h, accent_color, radius=0.005):
    """Grey box with colored left accent bar (Quntur paper style)."""
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


def draw_role_label(ax, x, y, color, label):
    """Colored rounded-square icon + bold role name above a box."""
    icon = FancyBboxPatch(
        (x, y - 0.004), 0.009, 0.009,
        boxstyle="round,pad=0,rounding_size=0.002",
        facecolor=color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=5,
    )
    ax.add_patch(icon)
    ax.text(
        x + 0.013, y + 0.001, label,
        fontsize=9.5, fontweight="bold", color=C_TITLE,
        va="center", transform=ax.transAxes, zorder=5,
    )


def draw_arrow(ax, x, y1, y2, color=C_ARROW, lw=1.5):
    """Downward arrow between boxes."""
    ax.annotate(
        "", xy=(x, y2), xytext=(x, y1),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle="arc3,rad=0",
        ),
        zorder=3,
    )


def draw_divider(ax, x1, x2, y, color="#D1D5DB", lw=0.6):
    """Thin horizontal divider line."""
    ax.plot(
        [x1, x2], [y, y],
        color=color, lw=lw, transform=ax.transAxes, zorder=3,
    )


def draw_tool_pills(ax, x, y, tools, max_w):
    """Draw tool-call pills, wrapping to new rows as needed."""
    col_x = x
    row_y = y
    for tool in tools:
        short = TOOL_SHORT.get(tool, tool)
        # Character width calibrated for 7-inch figure at fontsize 8 monospace
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


def wrap(text, width=90):
    """Wrap long lines while preserving explicit newlines."""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) > width:
            wrapped.extend(textwrap.wrap(line, width=width))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


# ── Main figure ─────────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(7, 16))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    # ── Stacked-card layout: all sections in one rounded box ──────
    y_top = 0.985

    # Section heights (compact — deduplicated pills need less space)
    qh  = 0.044    # User query (with "User" title inside)
    ph  = 0.064    # Routing middleware
    s1h = 0.185    # Separation sub-agent (+ context handoff line)
    s2h = 0.155    # TEA/LCA sub-agent
    ah  = 0.095    # Orchestrator synthesized answer

    sections = [
        (qh,  C_USER),
        (ph,  C_ROUTER),
        (s1h, C_SEP_ENG),
        (s2h, C_TEA_LCA),
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

    # Colored accent bars for each section
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

    # Thin dividers between sections
    div_y = y_top
    for h, _ in sections[:-1]:
        div_y -= h
        ax.plot(
            [LEFT, LEFT + MAIN_W], [div_y, div_y],
            color=C_BOX_BORDER, lw=0.8,
            transform=ax.transAxes, zorder=4,
        )

    # Compute section y-positions
    qy  = y_top - qh
    py  = qy - ph
    s1y = py - s1h
    s2y = s1y - s2h
    ay  = s2y - ah

    # ── 1. User Query ─────────────────────────────────────────────
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

    # ── 2. Routing Middleware ─────────────────────────────────────
    ax.text(
        CL, py + ph - 0.006,
        "Sequential Delegation Plan",
        fontsize=9, fontweight="bold", color="#15803D",
        transform=ax.transAxes, va="top", zorder=5,
    )
    plan_text = (
        'Keyword match: "separation sequence" -> separation-engineer (score 3)\n'
        'Keyword match: "operating cost", "payback" -> tea-lca-analyst (score 2+)\n'
        'Pair (separation-engineer, tea-lca-analyst) in SEQUENTIAL_PAIRS\n'
        'Hint: "Step 1: separation-engineer  ->  Step 2: tea-lca-analyst"'
    )
    ax.text(
        CL, py + 0.005, plan_text,
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.4,
    )

    # ── 3. Step 1: Separation Sub-Agent ───────────────────────────
    ax.text(
        CL, s1y + s1h - 0.008,
        "Step 1: Separation Sub-Agent",
        fontsize=10, fontweight="bold", color=C_SEP_ENG,
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CR, s1y + s1h - 0.008,
        "37s  |  114K tokens  |  5 tool calls",
        fontsize=8, color=C_BODY, ha="right",
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, s1y + s1h - 0.024,
        wrap(TRACE["step1"]["description"]),
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="top", zorder=5, style="italic",
    )
    ax.text(
        CL, s1y + s1h - 0.050,
        "Tool Calls:",
        fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
        transform=ax.transAxes, va="top", zorder=5,
    )
    draw_tool_pills(
        ax, CL, s1y + s1h - 0.064,
        list(dict.fromkeys(TRACE["step1"]["tool_calls"])), max_w=CW,
    )
    draw_divider(ax, CL, CR, s1y + s1h - 0.092)
    ax.text(
        CL, s1y + s1h - 0.098,
        "Sub-Agent Output:",
        fontsize=8, fontweight="bold", color="#9A3412",
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, s1y + 0.020,
        TRACE["step1"]["result_summary"],
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.35,
    )
    ax.text(
        CL, s1y + 0.006,
        "Separation results passed as context to next agent.",
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, style="italic",
    )

    # ── 4. Step 2: TEA/LCA Sub-Agent ─────────────────────────────
    ax.text(
        CL, s2y + s2h - 0.008,
        "Step 2: TEA/LCA Sub-Agent",
        fontsize=10, fontweight="bold", color=C_TEA_LCA,
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CR, s2y + s2h - 0.008,
        "26s  |  50K tokens  |  3 tool calls",
        fontsize=8, color=C_BODY, ha="right",
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, s2y + s2h - 0.024,
        wrap(TRACE["step2"]["description"]),
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="top", zorder=5, style="italic",
    )
    ax.text(
        CL, s2y + s2h - 0.050,
        "Tool Calls:",
        fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
        transform=ax.transAxes, va="top", zorder=5,
    )
    draw_tool_pills(
        ax, CL, s2y + s2h - 0.064,
        list(dict.fromkeys(TRACE["step2"]["tool_calls"])), max_w=CW,
    )
    draw_divider(ax, CL, CR, s2y + s2h - 0.092)
    ax.text(
        CL, s2y + s2h - 0.098,
        "Sub-Agent Output:",
        fontsize=8, fontweight="bold", color="#9D174D",
        transform=ax.transAxes, va="top", zorder=5,
    )
    ax.text(
        CL, s2y + 0.006,
        TRACE["step2"]["result_summary"],
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.35,
    )

    # ── 5. Orchestrator Synthesized Answer ────────────────────────
    ax.text(
        CL, ay + ah - 0.008,
        "Synthesized Answer",
        fontsize=9, fontweight="bold", color="#0369A1",
        transform=ax.transAxes, va="top", zorder=5,
    )
    synth = (
        "Selective dissolution of PS at 100\u00b0C (HDPE stays solid).\n"
        "\n"
        "DMF: 28\u00d7 selectivity, BP 153\u00b0C  |  OPEX $1.75/kg, payback 3.4 months\n"
        "Toluene: 10.2\u00d7 selectivity, BP 111\u00b0C  |  OPEX $1.42/kg, payback 6.2 months\n"
        "\n"
        "Recommendation: Toluene for lowest cost & environmental impact;\n"
        "DMF for highest purity & fastest ROI."
    )
    ax.text(
        CL, ay + 0.005, synth,
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="bottom", zorder=5, linespacing=1.35,
    )

    y = ay - 0.018

    # ── 6. Execution Timeline ─────────────────────────────────────
    ax.text(
        0.50, y, "Execution Timeline",
        ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
        transform=ax.transAxes,
    )
    y -= 0.018

    bar_left = LEFT + 0.01
    bar_w = MAIN_W - 0.02
    bar_h = 0.026
    bar_y = y - bar_h

    bg_bar = FancyBboxPatch(
        (bar_left, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.004",
        facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(bg_bar)

    # Orchestrator init (0-3s)
    orch_w = (3 / 75) * bar_w
    ax.add_patch(FancyBboxPatch(
        (bar_left, bar_y), orch_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.004",
        facecolor=C_USER, edgecolor="none", alpha=0.7,
        transform=ax.transAxes, zorder=3,
    ))

    # Step 1 (3-40s)
    s1_start = bar_left + (3 / 75) * bar_w
    s1_bw = (37 / 75) * bar_w
    ax.add_patch(FancyBboxPatch(
        (s1_start, bar_y), s1_bw, bar_h,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_SEP_ENG, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(
        s1_start + s1_bw / 2, bar_y + bar_h / 2,
        "separation-engineer (37s)",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        transform=ax.transAxes, zorder=4,
    )

    # Handoff (40-43s)
    ho_start = bar_left + (40 / 75) * bar_w
    ho_w = (3 / 75) * bar_w
    ax.add_patch(FancyBboxPatch(
        (ho_start, bar_y), ho_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_USER, edgecolor="none", alpha=0.7,
        transform=ax.transAxes, zorder=3,
    ))

    # Step 2 (43-69s)
    s2_start = bar_left + (43 / 75) * bar_w
    s2_bw = (26 / 75) * bar_w
    ax.add_patch(FancyBboxPatch(
        (s2_start, bar_y), s2_bw, bar_h,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_TEA_LCA, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(
        s2_start + s2_bw / 2, bar_y + bar_h / 2,
        "tea-lca-analyst (26s)",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        transform=ax.transAxes, zorder=4,
    )

    # Final synthesis (69-75s)
    fs_start = bar_left + (69 / 75) * bar_w
    fs_w = (6 / 75) * bar_w
    ax.add_patch(FancyBboxPatch(
        (fs_start, bar_y), fs_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.004",
        facecolor=C_SYNTH, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))

    # Time labels
    y_time = bar_y - 0.012
    for t_sec in [0, 15, 30, 45, 60, 75]:
        tx = bar_left + (t_sec / 75) * bar_w
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
        (C_TEA_LCA, "tea-lca-analyst"),
        (C_SYNTH,   "Synthesis"),
    ]
    lx = bar_left
    leg_spacing = (bar_w - 0.02) / 4
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

    y = y_legend - 0.022

    # ── 7. Bottom Panels ────────────────────────────────────────────
    panel_h = 0.138
    panel_y = y - panel_h
    panel_gap = 0.012
    pw = (MAIN_W - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)

    # Panel content helper offsets
    pcl = ACCENT_W + 0.010  # content left inside panel

    # Panel 1: Trace Metadata
    draw_quntur_box(ax, p1x, panel_y, pw, panel_h, C_USER, radius=0.004)
    ax.text(
        p1x + pw / 2, panel_y + panel_h - 0.012,
        "Trace Metadata",
        ha="center", va="top", fontsize=9, fontweight="bold",
        color=C_TITLE, transform=ax.transAxes,
    )

    metrics = [
        (">", "Run Time",     "~75 s"),
        ("#", "Total Tokens", "~193K"),
        ("$", "Est. Cost",    "<$0.01"),
        ("~", "LLM Calls",    "19 total"),
        ("+", "Tool Calls",   "8 (subagent)"),
        ("*", "Orch. Calls",  "3"),
    ]
    my = panel_y + panel_h - 0.036
    for icon, label, value in metrics:
        ax.text(
            p1x + pcl, my, icon,
            fontsize=8, color=C_BODY, transform=ax.transAxes,
            va="center", zorder=5,
        )
        ax.text(
            p1x + pcl + 0.016, my, label,
            fontsize=8, color=C_BODY, transform=ax.transAxes,
            va="center", zorder=5,
        )
        ax.text(
            p1x + pw - 0.008, my, value,
            fontsize=8, fontweight="bold", color=C_TITLE,
            ha="right", transform=ax.transAxes, va="center", zorder=5,
        )
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
        "Sequential delegation:\n"
        "1. Routing middleware\n"
        "   classifies query ->\n"
        "   two-agent sequential\n"
        "2. separation-engineer\n"
        "   runs first\n"
        "3. Results forwarded to\n"
        "   tea-lca-analyst\n"
        "4. Orchestrator synthesizes\n"
        "   final answer"
    )
    ax.text(
        p2x + pcl, panel_y + panel_h - 0.034,
        pattern_text,
        fontsize=8, color=C_BODY, transform=ax.transAxes,
        va="top", zorder=5, linespacing=1.2,
    )

    # Panel 3: Active Guardrails
    draw_quntur_box(ax, p3x, panel_y, pw, panel_h, "#EF4444", radius=0.004)
    ax.text(
        p3x + pw / 2, panel_y + panel_h - 0.012,
        "Active Guardrails",
        ha="center", va="top", fontsize=9, fontweight="bold",
        color=C_TITLE, transform=ax.transAxes,
    )
    guardrails = [
        "[x] SubagentGuard",
        "    Middleware (25 iter.)",
        "[x] Token budget",
        "    (200K per subagent)",
        "[x] Routing middleware",
        "    (keyword classifier)",
        "[x] Boiling point",
        "    constraint (BP > T_op)",
        "[x] recursion_limit = 250",
    ]
    gy = panel_y + panel_h - 0.036
    for g in guardrails:
        ax.text(
            p3x + pcl, gy, g,
            fontsize=8, color="#166534", transform=ax.transAxes,
            va="top", zorder=5,
        )
        gy -= 0.0115

    # ── Footer ──────────────────────────────────────────────────────
    ax.text(
        0.50, panel_y - 0.012,
        "DISSOLVE: Data Integrated Solubility Solver via LLM Evaluation  |  "
        "Model: Gemini 3 Flash Preview  |  "
        "Trace ID: 019c2ef5-16b9-7543",
        ha="center", fontsize=8, color=C_BODY,
        transform=ax.transAxes,
    )

    out_path = "/home/aaltamimi2/langchain-STRAP/architecture/sequential_trace.png"
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
