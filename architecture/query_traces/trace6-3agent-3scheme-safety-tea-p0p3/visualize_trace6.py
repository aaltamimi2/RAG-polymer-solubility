"""
Visualize DISSOLVE agent Trace 6: P0-P3 validation run.
3-scheme separation + safety + TEA, with improved safety-analyst prompt (not invoked).

Key finding: orchestrator self-served safety assessment via direct GSK DB queries
instead of delegating to the safety-analyst subagent.

Trace ID: 019c3149-e973-77b0-9d84-9ed33f07cd68
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


# -- Color palette ------------------------------------------------------------
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"
C_BOX_BORDER = "#E5E7EB"

C_USER       = "#6366F1"   # Indigo  -- orchestrator / user
C_ROUTER     = "#22C55E"   # Green   -- routing
C_SEP_ENG    = "#F97316"   # Orange  -- separation-engineer
C_TEA        = "#0EA5E9"   # Sky     -- tea-lca-analyst
C_SAFETY     = "#EF4444"   # Red     -- safety (orchestrator self-served)
C_SYNTH      = "#8B5CF6"   # Violet  -- synthesis
C_WARN       = "#F59E0B"   # Amber   -- warnings

C_S1         = "#F97316"   # Orange  -- scheme 1
C_S2         = "#8B5CF6"   # Violet  -- scheme 2
C_S3         = "#10B981"   # Emerald -- scheme 3

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"


# -- Layout constants ----------------------------------------------------------
FIG_W  = 7.0
FIG_H  = 20.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
GAP      = 0.005
PILL_H   = 0.012


# -- Helpers -------------------------------------------------------------------

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


def _pill(ax, x, y_center, text, fg, bg, border, fs=6.5, mono=True):
    tw = len(text) * (0.0075 if mono else 0.0065) + 0.014
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


def _pills_row(ax, x_start, y_center, labels, max_w, fg, bg, border, fs=5.5):
    cx = x_start
    rows = 1
    for label in labels:
        tw = len(label) * 0.0075 + 0.014
        if cx + tw > x_start + max_w and cx > x_start:
            cx = x_start
            y_center -= PILL_H
            rows += 1
        _pill(ax, cx, y_center, label, fg, bg, border, fs)
        cx += tw + 0.006
    return rows


def _wrap(text, width=95):
    lines = text.split("\n")
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width) if len(line) > width else [line])
    return "\n".join(out)


# -- Main figure ---------------------------------------------------------------

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD
    cw = cr - cl

    y = 0.993

    # ==================================================================
    # 1. HEADER
    # ==================================================================
    ax.text(MID, y, "Trace 6: P0-P3 Validation  |  3-Scheme + Safety + TEA",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    # ==================================================================
    # 2. USER QUERY
    # ==================================================================
    query = (
        "Find the optimal separation sequence for PS, PVC, LDPE, HDPE, PP, "
        "EVOH, Nylon6, Nylon66, and PET. Propose THREE different dissolution "
        "schemes. Then run a safety assessment on each scheme. Finally, run a "
        "techno-economic analysis on each scheme to compare operating costs."
    )
    box_h = 0.026
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.005
    ax.text(cl, cy, "User Query",
            fontsize=8, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    ax.text(cl, cy, _wrap(query, 120),
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 3. ROUTING
    # ==================================================================
    box_h = 0.020
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    cy = y - 0.005
    ax.text(cl, cy, "3-Agent Sequential Routing",
            fontsize=8, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    ax.text(cl, cy,
            'Keywords: "separat"+"dissolut" -> sep-eng  |  '
            '"safe" -> safety  |  "techno-economic" -> tea-lca  |  '
            '3 agents -> sequential',
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5)
    y -= box_h + GAP

    # ==================================================================
    # 4. SEPARATION-ENGINEER (3 attempts)
    # ==================================================================
    box_h = 0.042
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP_ENG)
    cy = y - 0.005
    ax.text(cl, cy, "1. separation-engineer  (3 invocations)",
            fontsize=8, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "175s total  |  496K tokens",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    sep_attempts = [
        ("Attempt 1 (74.7s, 216K tok):", "Token budget exceeded at 216K/200K. Returned partial results."),
        ("Attempt 2 (75.7s, 220K tok):", "Token budget exceeded at 220K/200K. Returned partial results."),
        ("Attempt 3 (24.4s, 59K tok):", "Tool-call budget exhausted. Returned 3-scheme tables."),
    ]
    for label, desc in sep_attempts:
        ax.text(cl, cy, label, fontsize=5.5, fontweight="bold",
                color=C_SEP_ENG, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.17, cy, desc, fontsize=5.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ==================================================================
    # 5. ORCHESTRATOR SELF-SERVICE (DB queries)
    # ==================================================================
    box_h = 0.042
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.005
    ax.text(cl, cy, "Orchestrator: Direct DB Queries",
            fontsize=8, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "6 tool calls  |  list_polymers, list_tables, query_db x3, get_properties",
            fontsize=5.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    ax.text(cl, cy,
            "After sep-eng returned partial results, the orchestrator ran its own "
            "DB queries to fill gaps:\n"
            "list_available_polymers, list_tables, 3x query_database (solubility data, "
            "boiling points), get_solvent_properties.",
            fontsize=5.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 6. THREE SCHEME TABLES
    # ==================================================================
    ax.text(MID, y - 0.002, "Final Schemes (orchestrator-assembled from sep-eng + DB queries)",
            ha="center", fontsize=8, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    col_gap = 0.008
    col_w = (WIDTH - 2 * col_gap) / 3
    c1x = LEFT
    c2x = LEFT + col_w + col_gap
    c3x = LEFT + 2 * (col_w + col_gap)

    scheme_h = 0.130
    schemes = [
        ("Scheme 1: Industrial", C_S1, c1x, [
            ("PS",       "Toluene",          "25 C"),
            ("EVOH",     "Triethylamine",    "25 C"),
            ("PVC",      "THF",              "40 C"),
            ("LDPE",     "Xylene",           "75 C"),
            ("HDPE",     "Xylene",           "100 C"),
            ("PP",       "Xylene",           "120 C"),
            ("Nylon 6",  "Formic Acid",      "25 C"),
            ("Nylon 66", "m-Cresol",         "25 C"),
            ("PET",      "NMP",              "180 C"),
        ]),
        ("Scheme 2: Green/Alt", C_S2, c2x, [
            ("PS",       "Ethyl Acetate",    "25 C"),
            ("EVOH",     "Ethanol",          "70 C"),
            ("PVC",      "Cyclohexanone",    "60 C"),
            ("LDPE",     "Decalin",          "80 C"),
            ("HDPE",     "Decalin",          "100 C"),
            ("PP",       "Decalin",          "120 C"),
            ("Nylon 6",  "Benzyl Alcohol",   "100 C"),
            ("Nylon 66", "Benzyl Alcohol",   "150 C"),
            ("PET",      "DMSO",             "160 C"),
        ]),
        ("Scheme 3: Energy-Opt", C_S3, c3x, [
            ("PS",       "MEK",              "25 C"),
            ("EVOH",     "Isopropanol",      "80 C"),
            ("PVC",      "NMP",              "60 C"),
            ("LDPE",     "Toluene",          "70 C"),
            ("HDPE",     "Toluene",          "90 C"),
            ("PP",       "Toluene",          "105 C"),
            ("Nylon 6",  "Formic Acid",      "40 C"),
            ("Nylon 66", "Formic Acid",      "80 C"),
            ("PET",      "m-Cresol",         "180 C"),
        ]),
    ]

    for title, color, sx, steps in schemes:
        _box(ax, sx, y, col_w, scheme_h, color, radius=0.004)
        scl = sx + ACCENT_W + 0.006
        scr = sx + col_w - 0.005

        cy = y - 0.005
        ax.text(scl, cy, title,
                fontsize=6.5, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.010

        ax.text(scl, cy, "Polymer", fontsize=5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scl + 0.055, cy, "Solvent", fontsize=5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scr, cy, "Temp", fontsize=5, fontweight="bold",
                color=C_TITLE, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.007

        for polymer, solvent, temp in steps:
            ax.text(scl, cy, polymer, fontsize=5, fontweight="bold",
                    color=color, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scl + 0.055, cy, solvent, fontsize=5,
                    color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scr, cy, temp, fontsize=5,
                    color=C_BODY, ha="right",
                    transform=ax.transAxes, va="top", zorder=5)
            cy -= 0.011

    y -= scheme_h + GAP

    # ==================================================================
    # 7. TEA-LCA-ANALYST
    # ==================================================================
    box_h = 0.048
    _box(ax, LEFT, y, WIDTH, box_h, C_TEA)
    cy = y - 0.005
    ax.text(cl, cy, "2. tea-lca-analyst",
            fontsize=8, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "30.9s  |  42.8K tokens  |  9 analyze_solvent_recovery_tea calls",
            fontsize=5.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    ax.text(cl, cy,
            "Analyzed operating costs for all 3 schemes. Tool-call budget exhausted.",
            fontsize=5.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.010
    _divider(ax, cl, cr, cy)
    cy -= 0.008

    tea_ranks = [
        ("Most cost-effective:", "Scheme 3", C_S3,
         "Toluene reuse for LDPE->HDPE->PP (ramp temp, no solvent switch)."),
        ("Highest energy cost:", "Scheme 2", C_S2,
         "Benzyl Alcohol + Decalin (BP >190 C) -> expensive distillation recovery."),
    ]
    for rank, scheme, color, rationale in tea_ranks:
        ax.text(cl + 0.002, cy, rank, fontsize=5.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.12, cy, scheme, fontsize=5.5, fontweight="bold",
                color=color, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.185, cy, rationale, fontsize=5.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ==================================================================
    # 8. SAFETY ASSESSMENT (orchestrator self-served)
    # ==================================================================
    box_h = 0.078
    _box(ax, LEFT, y, WIDTH, box_h, C_SAFETY)
    cy = y - 0.005
    ax.text(cl, cy, "3. Safety Assessment  (orchestrator self-served, no safety-analyst subagent)",
            fontsize=7.5, fontweight="bold", color=C_SAFETY,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "2 query_database + 1 check_column_values + 1 get_solvent_properties",
            fontsize=5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    ax.text(cl, cy,
            "Orchestrator queried GSK dataset directly instead of delegating to safety-analyst.\n"
            "Absolute G-scores cited for each solvent — a key improvement over Trace 5.",
            fontsize=5.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.015
    _divider(ax, cl, cr, cy)
    cy -= 0.008

    # G-score pills — row 1: Problematic, row 2: Good
    ax.text(cl, cy, "G-Scores (from orchestrator DB query):",
            fontsize=6.5, fontweight="bold", color=C_SAFETY,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    row1 = [("THF 4.79", "Problematic"), ("NMP 5.49", "Problematic"),
            ("Toluene 5.96", "Problematic")]
    row2 = [("Ethyl Acetate 6.66", "Good"), ("Cyclohexanone 7.24", "Good"),
            ("Benzyl Alcohol 7.68", "Good")]

    for row in [row1, row2]:
        cx = cl
        for score, rating in row:
            bg = "#FEE2E2" if rating == "Problematic" else "#D1FAE5"
            fg = "#991B1B" if rating == "Problematic" else "#065F46"
            tw = _pill(ax, cx, cy, f"{score} ({rating})", fg, bg, fg, fs=4.5)
            cx += tw + 0.005
        cy -= PILL_H + 0.002

    cy -= 0.004
    ax.text(cl, cy,
            "Scheme 2 (Green/Alt) rated safest: Ethyl Acetate + Cyclohexanone + "
            "Benzyl Alcohol (all Good). No fabricated scores.",
            fontsize=5.5, color=C_BODY,
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)

    y -= box_h + GAP

    # ==================================================================
    # 9. KEY FINDING
    # ==================================================================
    box_h = 0.030
    _box(ax, LEFT, y, WIDTH, box_h, C_WARN)
    cy = y - 0.005
    ax.text(cl, cy, "Key Finding: Scheme 2 as Deliberate \"Green\" Design",
            fontsize=8, fontweight="bold", color="#92400E",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    finding = (
        "Unlike Trace 5 where all 3 schemes were designed purely for "
        "separation efficiency, Trace 6 proactively designed Scheme 2 as a "
        "\"Green/Alternative\" option using safer solvents (Ethyl Acetate, "
        "Benzyl Alcohol, Ethanol). The P0-P3 safety improvements influenced "
        "the orchestrator's scheme design even though the safety-analyst "
        "subagent was not invoked."
    )
    ax.text(cl, cy, _wrap(finding, 115),
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 10. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.023
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.005
    ax.text(cl, cy, "Orchestrator Synthesis",
            fontsize=8, fontweight="bold", color=C_SYNTH,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "6,930 chars  |  25K out tokens  |  30 messages",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    ax.text(cl, cy,
            "Integrated separation schemes, TEA comparison, and safety "
            "assessment with absolute G-scores into unified recommendation.",
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP * 2

    # ==================================================================
    # 11. EXECUTION TIMELINE
    # ==================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.010

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.014
    total_s = 274.9

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)

    bar_y = y - bar_h
    ax.add_patch(FancyBboxPatch((bar_left, bar_y), bar_w, bar_h,
                                transform=ax.transAxes, zorder=2, **bg_kw))

    def _bar_seg(t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_s) * bar_w
        w = (t_dur / total_s) * bar_w
        ax.add_patch(FancyBboxPatch(
            (x0, bar_y), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if w > 0.03:
            ax.text(x0 + w / 2, bar_y + bar_h / 2, label,
                    ha="center", va="center", fontsize=4.5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    # Timeline segments
    _bar_seg(0, 74.7, C_SEP_ENG, "sep-eng #1 (74.7s)")
    _bar_seg(74.7, 75.7, C_SEP_ENG, "sep-eng #2 (75.7s)")
    _bar_seg(150.4, 24.4, C_SEP_ENG, "sep #3")
    _bar_seg(174.8, 30.9, C_TEA, "tea-lca (30.9s)")
    _bar_seg(205.7, 40, C_USER, "orch DB queries")
    _bar_seg(245.7, 29.2, C_SYNTH, "synthesis")

    # Time labels
    ty = bar_y - 0.007
    for t in [0, 50, 100, 150, 200, 250, 275]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=5.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.010
    items = [(C_SEP_ENG, "sep-eng (x3)"), (C_TEA, "tea-lca"),
             (C_USER, "Orchestrator"), (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.02
    sp = (bar_w - 0.04) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=5.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.015

    # ==================================================================
    # 12. BOTTOM PANELS
    # ==================================================================
    panel_h = 0.062
    panel_gap = 0.008
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.006

    # Panel 1: Trace Metadata
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.005, "Trace Metadata",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "274.9 s"),
        ("Total Tokens",   "770K"),
        ("Input Tokens",   "745K"),
        ("Output Tokens",  "25K"),
        ("Messages",       "30"),
        ("LLM Calls",      "53"),
        ("Tool Runs",      "360"),
    ]
    my = y - 0.015
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.005, my, value, fontsize=5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.007

    # Panel 2: Trace 5 vs 6 Comparison
    _box(ax, p2x, y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.005, "Trace 5 vs 6",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    comparisons = (
        "Trace 5 (pre-P0P3):\n"
        "  311s / 628K tok / 3 subagents\n"
        "  Safety: hallucinated G-scores\n"
        "\n"
        "Trace 6 (P0-P3 active):\n"
        "  275s / 770K tok / 2 subagents\n"
        "  Safety: real G-scores, absolute\n"
        "  context, green scheme designed"
    )
    ax.text(p2x + pcl, y - 0.014, comparisons,
            fontsize=5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.1)

    # Panel 3: Observations
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.005, "Observations",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    observations = (
        "1. Safety-analyst subagent was\n"
        "   NOT invoked -- orchestrator\n"
        "   self-served via GSK DB queries\n"
        "2. Sep-eng called 3x (budget\n"
        "   limits hit on first 2 attempts)\n"
        "3. No fabricated G-scores\n"
        "4. Scheme 2 proactively green"
    )
    ax.text(p3x + pcl, y - 0.014, observations,
            fontsize=5, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.1)

    y -= panel_h + 0.005

    # -- Footer --
    ax.text(MID, y,
            "DISSOLVE  |  Gemini 3 Flash Preview  |  "
            "Trace 019c3149  |  P0-P3 Validation  |  "
            "3-Scheme + Safety + TEA",
            ha="center", fontsize=6, color=C_BODY,
            transform=ax.transAxes)

    # -- Clip whitespace by adjusting ylim to actual content extent --
    ax.set_ylim(y - 0.015, 1.0)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "019c3149-e973-77b0-9d84-9ed33f07cd68.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
