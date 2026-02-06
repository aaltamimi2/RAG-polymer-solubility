"""
Visualize DISSOLVE agent trace: 3-scheme 9-polymer separation.
Query asked for THREE different dissolution schemes with comparison table.

Trace ID: 019c3073-97ae-7662-88b1-ae5d96d77aec
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
C_SYNTH      = "#0EA5E9"   # Sky     -- synthesis
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
FIG_H  = 22.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
GAP      = 0.006
PILL_H   = 0.015


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
        (x, y_center - 0.006), tw, 0.012,
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


def _pills_row(ax, x_start, y_center, labels, max_w, fg, bg, border, fs=6):
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

    y = 0.990

    # ==================================================================
    # 1. USER QUERY
    # ==================================================================
    query = (
        "Find the optimal separation sequence for a mixed polymer waste "
        "stream containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, "
        "Nylon66, and PET. Use selective dissolution at atmospheric "
        "pressure. Propose THREE different sets of solvents and conditions "
        "for this 9-polymer dissolution scheme."
    )
    box_h = 0.038
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.008
    ax.text(cl, cy, "User Query",
            fontsize=9, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.013
    ax.text(cl, cy, _wrap(query, 110),
            fontsize=6.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 2. ROUTING
    # ==================================================================
    box_h = 0.028
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    cy = y - 0.008
    ax.text(cl, cy, "Single-Agent Routing",
            fontsize=9, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012
    ax.text(cl, cy,
            'Keyword match: "separat" + "solvent" + "dissolut" '
            '-> separation-engineer  |  Patched: max_tool_calls=20, '
            'synthesis_injection=off',
            fontsize=6, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5)
    y -= box_h + GAP

    # ==================================================================
    # 3. SEPARATION-ENGINEER SUBAGENT
    # ==================================================================
    box_h = 0.078
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP_ENG)
    cy = y - 0.008
    ax.text(cl, cy, "separation-engineer",
            fontsize=9, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~210s  |  1,549K in  |  20 tool calls  |  54 messages",
            fontsize=6.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.014

    ax.text(cl, cy,
            "Explored 9-polymer dissolution space across multiple solvent families. "
            "Made 20 domain tool calls (query_database)\n"
            "to compare solubilities, find alternative solvents, and verify "
            "boiling point constraints for 3 distinct schemes.",
            fontsize=6.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.022

    _divider(ax, cl, cr, cy)
    cy -= 0.008

    # Guardrail pills
    ax.text(cl, cy, "Guardrails Hit:",
            fontsize=7, fontweight="bold", color=C_GUARD_TEXT,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012
    _pills_row(ax, cl, cy,
               ["token budget: 202K/200K", "tool limit: 20/20"],
               cw, C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=5.5)
    y -= box_h + GAP

    # ==================================================================
    # 4. THREE SCHEME TABLES (side by side)
    # ==================================================================
    # Section header
    ax.text(MID, y - 0.002, "Agent Output: Three Dissolution Schemes",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.016

    col_gap = 0.010
    col_w = (WIDTH - 2 * col_gap) / 3
    c1x = LEFT
    c2x = LEFT + col_w + col_gap
    c3x = LEFT + 2 * (col_w + col_gap)

    scheme_h = 0.170
    schemes = [
        ("Scheme 1: Standard", C_S1, c1x, [
            ("EVOH",     "Triethylamine",  "25 C"),
            ("PS",       "Benzene",        "55 C"),
            ("PVC",      "THF",            "55 C"),
            ("PP",       "Cyclohexane",    "75 C"),
            ("LDPE",     "o-Xylene",       "105 C"),
            ("HDPE",     "o-Xylene",       "140 C"),
            ("Nylon 6",  "DMF",            "145 C"),
            ("Nylon 66", "DMSO",           "155 C"),
            ("PET",      "Eth. Glycol",    "190 C"),
        ]),
        ("Scheme 2: High-BP", C_S2, c2x, [
            ("EVOH",     "Isopropylamine", "25 C"),
            ("PS",       "Methyl Acetate", "45 C"),
            ("PVC",      "DMF",            "80 C"),
            ("PP",       "p-Xylene",       "100 C"),
            ("LDPE",     "Dodecane",       "110 C"),
            ("HDPE",     "Dodecane",       "150 C"),
            ("Nylon 6",  "Cyclohexanol",   "150 C"),
            ("Nylon 66", "Eth. Glycol",    "170 C"),
            ("PET",      "DMSO",           "160 C"),
        ]),
        ("Scheme 3: Aromatic", C_S3, c3x, [
            ("EVOH",     "Ethanol",        "75 C"),
            ("PS",       "Toluene",        "60 C"),
            ("PVC",      "Chloroform",     "55 C"),
            ("PP",       "n-Heptane",      "90 C"),
            ("LDPE",     "p-Xylene",       "110 C"),
            ("HDPE",     "p-Xylene",       "135 C"),
            ("Nylon 6",  "DMSO",           "150 C"),
            ("Nylon 66", "DMSO",           "170 C"),
            ("PET",      "(residue)",      "--"),
        ]),
    ]

    for title, color, sx, steps in schemes:
        _box(ax, sx, y, col_w, scheme_h, color, radius=0.004)
        scl = sx + ACCENT_W + 0.008
        scr = sx + col_w - 0.006

        cy = y - 0.008
        ax.text(scl, cy, title,
                fontsize=7.5, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.014

        # Column headers
        ax.text(scl, cy, "Polymer", fontsize=5.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scl + 0.060, cy, "Solvent", fontsize=5.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scr, cy, "Temp", fontsize=5.5, fontweight="bold",
                color=C_TITLE, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

        for polymer, solvent, temp in steps:
            ax.text(scl, cy, polymer, fontsize=5.5, fontweight="bold",
                    color=color, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scl + 0.060, cy, solvent, fontsize=5.5,
                    color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scr, cy, temp, fontsize=5.5,
                    color=C_BODY, ha="right",
                    transform=ax.transAxes, va="top", zorder=5)
            cy -= 0.0145

    y -= scheme_h + GAP

    # ==================================================================
    # 5. PAPER COMPARISON
    # ==================================================================
    box_h = 0.170
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)

    cy = y - 0.008
    ax.text(cl, cy, "Comparison vs. Paper (Sanchez-Rivera et al. 2025)",
            fontsize=9, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.014

    # Paper sequence header
    ax.text(cl, cy, "Paper's 10-step STRAP sequence:",
            fontsize=7, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    paper_steps = [
        ("1. PS",     "Toluene @ 35 C"),
        ("2. PVC",    "THF @ 67 C"),
        ("3. LDPE",   "o-Xylene @ 80 C"),
        ("4. HDPE",   "o-Xylene @ 95 C"),
        ("5. PP",     "o-Xylene @ 115 C"),
        ("6. EVOH",   "DMSO/Water @ 95 C"),
        ("7. PA66/6", "1,2-PDO @ 125 C"),
        ("8. PET",    "GVL @ 160 C"),
        ("9. PA6",    "DMSO @ 145 C"),
        ("10. PA66",  "(residue)"),
    ]

    # Two columns for paper steps
    half = len(paper_steps) // 2
    for i, (step, cond) in enumerate(paper_steps):
        col_offset = 0 if i < half else 0.46
        row_offset = (i % half) * 0.009
        ax.text(cl + col_offset, cy - row_offset, f"{step}: {cond}",
                fontsize=6, color=C_BODY, transform=ax.transAxes,
                va="top", zorder=5)

    cy -= half * 0.009 + 0.010

    _divider(ax, cl, cr, cy)
    cy -= 0.008

    ax.text(cl, cy, "Key Differences:",
            fontsize=7, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    diffs = [
        "Ordering: All 3 agent schemes dissolve EVOH first (step 1); "
        "paper dissolves EVOH 6th after polyolefins",
        "Polyolefins: Agent uses varied aromatics/alkanes "
        "(toluene, xylenes, cyclohexane, dodecane, heptane); "
        "paper uses o-xylene for all three",
        "PET: Scheme 3 recovers PET as residue (paper does this for PA66); "
        "Schemes 1-2 use ethylene glycol/DMSO instead of paper's GVL",
        "Safety: Scheme 1 proposes benzene for PS (carcinogen); "
        "paper uses toluene. No safety-analyst invoked.",
        "Thermodynamic validity: All 27 solvent-temperature pairs across "
        "3 schemes operate below solvent boiling points",
    ]
    for diff in diffs:
        wrapped = textwrap.fill(diff, width=120)
        nlines = wrapped.count("\n") + 1
        ax.text(cl + 0.008, cy, wrapped,
                fontsize=5.5, color=C_BODY, transform=ax.transAxes,
                va="top", zorder=5, linespacing=1.2)
        # bullet
        ax.text(cl, cy + 0.001, "-",
                fontsize=6, color=C_TITLE, transform=ax.transAxes,
                va="top", zorder=5)
        cy -= nlines * 0.0075 + 0.004

    y -= box_h + GAP

    # ==================================================================
    # 6. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.038
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.008
    ax.text(cl, cy, "Orchestrator Synthesis",
            fontsize=9, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~24s  |  24K out tokens",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.014
    ax.text(cl, cy,
            "Formatted 3 schemes in comparison table with rationale. "
            "Highlighted key solvent selection logic:\n"
            "EVOH in amines/alcohols, PS/PVC in ethers/aromatics, "
            "polyolefins by temperature ramp, nylons in DMF/DMSO, "
            "PET in glycol/DMSO or as residue.",
            fontsize=6.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP * 2

    # ==================================================================
    # 7. EXECUTION TIMELINE
    # ==================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.014

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.020
    total_s = 234.0

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
        if w > 0.04:
            ax.text(x0 + w / 2, bar_y + bar_h / 2, label,
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    _bar_seg(0, 10, C_USER, "route")
    _bar_seg(10, 200, C_SEP_ENG, "separation-engineer (~200s, 20 tool calls)")
    _bar_seg(210, 24, C_SYNTH, "synth")

    # Time labels
    ty = bar_y - 0.010
    for t in [0, 50, 100, 150, 200, 234]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=6.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.014
    items = [(C_USER, "Orchestrator"), (C_SEP_ENG, "separation-engineer"),
             (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.08
    sp = (bar_w - 0.16) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.015], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.020, ly, label, fontsize=6.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.020

    # ==================================================================
    # 8. BOTTOM PANELS
    # ==================================================================
    panel_h = 0.090
    panel_gap = 0.010
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.008

    # Panel 1: Trace Metadata
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.008, "Trace Metadata",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "234 s"),
        ("Total Tokens",   "1,573K"),
        ("Input Tokens",   "1,549K"),
        ("Output Tokens",  "24K"),
        ("Messages",       "109"),
        ("Tool Calls",     "54 (20 domain)"),
    ]
    my = y - 0.022
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.006, my, value, fontsize=6.5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.010

    # Panel 2: Cost Comparison
    _box(ax, p2x, y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.008, "Cost vs. Single Scheme",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    comparisons = (
        "Single scheme (Trace 3):\n"
        "  69s / 127K tok / 8 tools\n"
        "\n"
        "3 schemes (this trace):\n"
        "  234s / 1,573K tok / 54 tools\n"
        "\n"
        "Multiplier: 3.4x time,\n"
        "  12.4x tokens"
    )
    ax.text(p2x + pcl, y - 0.020, comparisons,
            fontsize=6, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    # Panel 3: Guardrails
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.008, "Guardrails",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    guards = (
        "[x] Tool limit: 20 (patched)\n"
        "    Hit at 20/20 calls\n"
        "[x] Token budget: 200K\n"
        "    Exceeded at 202K\n"
        "[ ] Synthesis injection\n"
        "    (disabled for this run)\n"
        "[x] Truncation: 2000 chars"
    )
    ax.text(p3x + pcl, y - 0.020, guards,
            fontsize=6, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    y -= panel_h + 0.006

    # -- Footer --
    ax.text(MID, y,
            "DISSOLVE  |  Gemini 3 Flash Preview  |  "
            "Trace 019c3073  |  "
            "3-Scheme 9-Polymer Separation (Sanchez-Rivera et al. 2025 case study)",
            ha="center", fontsize=6.5, color=C_BODY,
            transform=ax.transAxes)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "019c3073-97ae-7662-88b1-ae5d96d77aec.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
