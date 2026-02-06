"""
Visualize DISSOLVE agent trace: 3-agent sequential -- 3 schemes + safety + TEA.
Query asked for THREE dissolution schemes, then safety assessment, then TEA.

Trace ID: 019c30ab-342a-72c3-bb97-91fc69afe030
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
C_SAFETY     = "#EF4444"   # Red     -- safety-analyst
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
FIG_H  = 28.0

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
    # 1. USER QUERY
    # ==================================================================
    query = (
        "Find the optimal separation sequence for PS, PVC, LDPE, HDPE, PP, "
        "EVOH, Nylon6, Nylon66, and PET. Propose THREE different dissolution "
        "schemes. Then run a safety assessment on each scheme. Finally, run a "
        "techno-economic analysis on each scheme to compare operating costs."
    )
    box_h = 0.028
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.006
    ax.text(cl, cy, "User Query",
            fontsize=8, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    ax.text(cl, cy, _wrap(query, 120),
            fontsize=6, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 2. ROUTING
    # ==================================================================
    box_h = 0.022
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    cy = y - 0.006
    ax.text(cl, cy, "3-Agent Sequential Routing",
            fontsize=8, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    ax.text(cl, cy,
            'Keyword matches: "separat"+"dissolut" -> sep-eng  |  '
            '"safe" -> safety  |  "techno-economic" -> tea-lca  |  '
            '3 agents -> sequential',
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5)
    y -= box_h + GAP

    # ==================================================================
    # 3. SEPARATION-ENGINEER SUBAGENT
    # ==================================================================
    box_h = 0.055
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP_ENG)
    cy = y - 0.006
    ax.text(cl, cy, "1. separation-engineer",
            fontsize=8, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "20 tool calls  |  token budget hit (204K/200K)",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011
    ax.text(cl, cy,
            "Explored 9-polymer dissolution space using query_database. "
            "Produced 3 differentiated schemes:\n"
            "Conventional (ethyl acetate, THF, xylene, DMSO, DMF), "
            "Aromatic Focus (toluene, acetone, methanol, nitrobenzene), "
            "Low-Temp/Chlorinated (DCM, MEK, decalin, phenol, TFA).",
            fontsize=6, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.025
    _divider(ax, cl, cr, cy)
    cy -= 0.007
    _pills_row(ax, cl, cy,
               ["token budget: 204K/200K", "tool limit: 20/20", "synth injection: off"],
               cw, C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=5)
    y -= box_h + GAP

    # ==================================================================
    # 4. THREE SCHEME TABLES (side by side)
    # ==================================================================
    ax.text(MID, y - 0.002, "Agent Output: Three Dissolution Schemes",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.013

    col_gap = 0.008
    col_w = (WIDTH - 2 * col_gap) / 3
    c1x = LEFT
    c2x = LEFT + col_w + col_gap
    c3x = LEFT + 2 * (col_w + col_gap)

    scheme_h = 0.130
    schemes = [
        ("Scheme 1: Conventional", C_S1, c1x, [
            ("PS",       "Ethyl Acetate",  "50 C"),
            ("PVC",      "THF",            "25 C"),
            ("LDPE",     "Xylene",         "75 C"),
            ("PP",       "Xylene",         "95 C"),
            ("HDPE",     "Xylene",         "110 C"),
            ("EVOH",     "DMSO",           "100 C"),
            ("Nylon 6",  "DMF",            "100 C"),
            ("Nylon 66", "Formic Acid",    "60 C"),
            ("PET",      "m-Cresol",       "110 C"),
        ]),
        ("Scheme 2: Aromatic", C_S2, c2x, [
            ("PS",       "Toluene",        "25 C"),
            ("PVC",      "Acetone",        "50 C"),
            ("LDPE",     "Toluene",        "85 C"),
            ("PP",       "Toluene",        "100 C"),
            ("HDPE",     "Toluene",        "108 C"),
            ("EVOH",     "Methanol",       "60 C"),
            ("Nylon 6",  "Acetic Acid",    "100 C"),
            ("Nylon 66", "Benzyl Alcohol", "100 C"),
            ("PET",      "Nitrobenzene",   "120 C"),
        ]),
        ("Scheme 3: Low-Temp", C_S3, c3x, [
            ("PS",       "DCM",            "25 C"),
            ("PVC",      "MEK",            "70 C"),
            ("LDPE",     "Decalin",        "80 C"),
            ("PP",       "Decalin",        "105 C"),
            ("HDPE",     "Decalin",        "120 C"),
            ("EVOH",     "EtOH/Water",     "75 C"),
            ("Nylon 6",  "Phenol",         "45 C"),
            ("Nylon 66", "TFA",            "25 C"),
            ("PET",      "TFA/DCM",        "25 C"),
        ]),
    ]

    for title, color, sx, steps in schemes:
        _box(ax, sx, y, col_w, scheme_h, color, radius=0.004)
        scl = sx + ACCENT_W + 0.006
        scr = sx + col_w - 0.005

        cy = y - 0.006
        ax.text(scl, cy, title,
                fontsize=6.5, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.011

        # Column headers
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
    # 5. TEA-LCA-ANALYST SUBAGENT
    # ==================================================================
    box_h = 0.072
    _box(ax, LEFT, y, WIDTH, box_h, C_TEA)
    cy = y - 0.006
    ax.text(cl, cy, "2. tea-lca-analyst",
            fontsize=8, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~22 tool calls  |  tool limit hit (22/15)",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011

    ax.text(cl, cy,
            "Compared operating costs (solvent heating + recovery energy) "
            "across all 3 schemes using compare_solvents_tea_lca and related tools.",
            fontsize=6, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.013
    _divider(ax, cl, cr, cy)
    cy -= 0.008

    # TEA Rankings
    ax.text(cl, cy, "TEA Rankings (by operating cost):",
            fontsize=7, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    tea_ranks = [
        ("1 (cheapest)",      "Scheme 3", C_S3,
         "Low temps (25 C for PS, Nylon66, PET) minimize heating. Decalin has wide atmospheric window."),
        ("2",                 "Scheme 1", C_S1,
         "Balanced. Xylene for polyolefins is industrial standard. DMSO/DMF/m-Cresol at 100-110 C."),
        ("3 (most expensive)","Scheme 2", C_S2,
         "Nitrobenzene + Benzyl Alcohol BP >200 C -> extremely energy-intensive recovery."),
    ]
    for rank, scheme, color, rationale in tea_ranks:
        ax.text(cl + 0.002, cy, f"#{rank}:", fontsize=5.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.090, cy, scheme, fontsize=5.5, fontweight="bold",
                color=color, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.155, cy, rationale, fontsize=5.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ==================================================================
    # 6. SAFETY-ANALYST SUBAGENT
    # ==================================================================
    box_h = 0.072
    _box(ax, LEFT, y, WIDTH, box_h, C_SAFETY)
    cy = y - 0.006
    ax.text(cl, cy, "3. safety-analyst",
            fontsize=8, fontweight="bold", color=C_SAFETY,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "15 tool calls (14 get_solvent_gscore)  |  tool limit hit (15/15)",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011

    ax.text(cl, cy,
            "Assessed GSK G-scores and hazard profiles for solvents in all 3 "
            "schemes. Data-grounded: 14 database lookups (vs 0 in Trace 2).",
            fontsize=6, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.013
    _divider(ax, cl, cr, cy)
    cy -= 0.008

    # Safety Rankings
    ax.text(cl, cy, "Safety Rankings (by hazard profile):",
            fontsize=7, fontweight="bold", color=C_SAFETY,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    safety_ranks = [
        ("1 (safest)",        "Scheme 1", C_S1,
         "DMF (reproductive toxin), m-Cresol (corrosive). Moderate overall GSK scores."),
        ("2",                 "Scheme 2", C_S2,
         "Nitrobenzene (GSK G-score 1, carcinogen), Toluene (reproductive toxin)."),
        ("3 (most hazardous)","Scheme 3", C_S3,
         "DCM (suspected carcinogen), TFA (extremely corrosive/toxic), Phenol (dermal toxicity)."),
    ]
    for rank, scheme, color, hazards in safety_ranks:
        ax.text(cl + 0.002, cy, f"#{rank}:", fontsize=5.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.095, cy, scheme, fontsize=5.5, fontweight="bold",
                color=color, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.155, cy, hazards, fontsize=5.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ==================================================================
    # 7. COST-SAFETY TRADE-OFF (key finding)
    # ==================================================================
    box_h = 0.038
    _box(ax, LEFT, y, WIDTH, box_h, C_WARN)
    cy = y - 0.006
    ax.text(cl, cy, "Key Finding: Cost-Safety Trade-off",
            fontsize=8, fontweight="bold", color="#92400E",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011
    tradeoff = (
        "The cheapest process uses the most dangerous solvents. "
        "Scheme 3 achieves low operating costs by using aggressive "
        "solvents (DCM, TFA, Phenol) that dissolve polymers at ambient "
        "temperature, eliminating heating costs but introducing severe "
        "safety/environmental risks. Scheme 1 represents the best balance: "
        "moderate costs with fewer extreme hazards."
    )
    ax.text(cl, cy, _wrap(tradeoff, 115),
            fontsize=6, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 8. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.025
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.006
    ax.text(cl, cy, "Orchestrator Synthesis",
            fontsize=8, fontweight="bold", color=C_SYNTH,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "39K out tokens  |  18 messages",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    ax.text(cl, cy,
            "Integrated separation feasibility, operating costs, and safety "
            "hazards into unified recommendation. Identified cost-safety "
            "trade-off as central finding.",
            fontsize=6, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP * 2

    # ==================================================================
    # 9. EXECUTION TIMELINE
    # ==================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.011

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.016
    total_s = 311.0

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
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    # Approximate timing: route ~8s, sep-eng ~120s, tea-lca ~90s, safety ~60s, synth ~33s
    _bar_seg(0, 8, C_USER, "route")
    _bar_seg(8, 120, C_SEP_ENG, "sep-eng (~120s, 20 tools)")
    _bar_seg(128, 90, C_TEA, "tea-lca (~90s, 22 tools)")
    _bar_seg(218, 60, C_SAFETY, "safety (~60s, 15 tools)")
    _bar_seg(278, 33, C_SYNTH, "synth")

    # Time labels
    ty = bar_y - 0.008
    for t in [0, 50, 100, 150, 200, 250, 311]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=5.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.011
    items = [(C_USER, "Orchestrator"), (C_SEP_ENG, "sep-eng"),
             (C_TEA, "tea-lca"), (C_SAFETY, "safety"),
             (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.02
    sp = (bar_w - 0.04) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=5.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.016

    # ==================================================================
    # 10. BOTTOM PANELS
    # ==================================================================
    panel_h = 0.068
    panel_gap = 0.008
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.006

    # Panel 1: Trace Metadata
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.006, "Trace Metadata",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "311 s"),
        ("Total Tokens",   "628K"),
        ("Input Tokens",   "588K"),
        ("Output Tokens",  "39K"),
        ("Messages",       "18"),
        ("Subagents",      "3 (sequential)"),
    ]
    my = y - 0.017
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=5.5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.005, my, value, fontsize=5.5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.008

    # Panel 2: Cost Comparison
    _box(ax, p2x, y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.006, "Token Efficiency",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    comparisons = (
        "3-scheme only (Trace 4):\n"
        "  234s / 1,573K tok / 1 agent\n"
        "\n"
        "3-scheme + safety + TEA:\n"
        "  311s / 628K tok / 3 agents\n"
        "\n"
        "3-agent is 2.5x more\n"
        "token-efficient despite more work"
    )
    ax.text(p2x + pcl, y - 0.016, comparisons,
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.1)

    # Panel 3: Guardrails
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.006, "Guardrails",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    guards = (
        "sep-eng:\n"
        "  [x] Token budget: 204K/200K\n"
        "  [x] Tool limit: 20/20\n"
        "tea-lca:\n"
        "  [x] Tool limit: 22/15\n"
        "safety:\n"
        "  [x] Tool limit: 15/15\n"
        "  [!] DuckDB race condition"
    )
    ax.text(p3x + pcl, y - 0.016, guards,
            fontsize=5.5, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.1)

    y -= panel_h + 0.005

    # -- Footer --
    ax.text(MID, y,
            "DISSOLVE  |  Gemini 3 Flash Preview  |  "
            "Trace 019c30ab  |  "
            "3-Agent Sequential: 3 Schemes + Safety + TEA",
            ha="center", fontsize=6, color=C_BODY,
            transform=ax.transAxes)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "019c30ab-342a-72c3-bb97-91fc69afe030.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
