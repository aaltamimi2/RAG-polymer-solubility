"""
Visualize DISSOLVE agent single-subagent trace: 9-polymer separation case study.
Query: PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, PET separation sequence.

Trace ID: 019c3044-1e4d-73f0-a2e9-f47499716e98
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

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"


# -- Layout constants ----------------------------------------------------------
FIG_W  = 7.0
FIG_H  = 16.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
GAP      = 0.008
LINE_H   = 0.014
PILL_H   = 0.018


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
        (x, y_center - 0.008), tw, 0.016,
        boxstyle="round,pad=0.002,rounding_size=0.004",
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


def _pills_row(ax, x_start, y_center, labels, max_w, fg, bg, border, fs=6.5):
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

    # Descending y-cursor
    y = 0.985

    # ==================================================================
    # 1. USER QUERY
    # ==================================================================
    query = (
        "Find the optimal separation sequence for a mixed polymer waste "
        "stream containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, "
        "Nylon66, and PET. Use selective dissolution at atmospheric pressure."
    )
    box_h = 0.048
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)

    cy = y - 0.010
    ax.text(cl, cy, "User Query",
            fontsize=10, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016
    ax.text(cl, cy, _wrap(query, 105),
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 2. ROUTING
    # ==================================================================
    box_h = 0.042
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)

    cy = y - 0.010
    ax.text(cl, cy, "Single-Agent Routing",
            fontsize=10, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018
    route_text = (
        'Keyword match: "separat" + "solvent" + "dissolut" '
        '-> separation-engineer (score 3)\n'
        'Route: Delegate to separation-engineer for multi-polymer '
        'separation sequencing'
    )
    ax.text(cl, cy, route_text,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ==================================================================
    # 3. SEPARATION-ENGINEER SUBAGENT
    # ==================================================================
    box_h = 0.290
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP_ENG)

    cy = y - 0.010
    ax.text(cl, cy, "Step 1: separation-engineer",
            fontsize=10, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~50s  |  ~110K tok  |  8 tool calls (6 domain)",
            fontsize=7.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018

    ax.text(cl, cy,
            "Determine optimal separation sequence for 9 polymers (PS, PVC, LDPE, HDPE,\n"
            "PP, EVOH, Nylon6, Nylon66, PET) using selective dissolution at atmospheric pressure.",
            fontsize=7.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.030

    _divider(ax, cl, cr, cy)
    cy -= 0.010

    # Tool calls
    ax.text(cl, cy, "Tool Calls (6 domain, deduplicated):",
            fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    tool_labels = [
        "plan_sequential_separation",
        "find_optimal_sep_sequence",
        "build_compatibility_matrix",
        "analyze_selective_solubility x3",
    ]
    rows = _pills_row(ax, cl, cy, tool_labels, cw,
                       C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= rows * PILL_H + 0.006

    # Guardrail events
    ax.text(cl, cy, "Guardrail Events:",
            fontsize=8, fontweight="bold", color=C_GUARD_TEXT,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    guard_labels = [
        "synthesis injection",
        "LIMIT: 8/8 tool calls",
    ]
    rows = _pills_row(ax, cl, cy, guard_labels, cw,
                       C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=6)
    cy -= rows * PILL_H + 0.006

    _divider(ax, cl, cr, cy)
    cy -= 0.010

    # 9-step separation output
    ax.text(cl, cy, "Sub-Agent Output: 9-Step Separation Sequence",
            fontsize=8, fontweight="bold", color="#9A3412",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    steps = [
        ("1.", "PS",      "Toluene",          "25\u00b0C"),
        ("2.", "PVC",     "THF",              "40\u00b0C"),
        ("3.", "EVOH",    "1-Propanol/Water",  "75\u00b0C"),
        ("4.", "LDPE",    "Toluene",          "75\u00b0C"),
        ("5.", "HDPE",    "Toluene",          "95\u00b0C"),
        ("6.", "PP",      "Toluene",          "105\u00b0C"),
        ("7.", "Nylon6",  "Formic Acid",      "25\u00b0C"),
        ("8.", "Nylon66", "m-Cresol",         "100\u00b0C"),
        ("9.", "PET",     "NMP",              "180\u00b0C"),
    ]

    # Column headers
    ax.text(cl + 0.020, cy, "Polymer", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.130, cy, "Solvent", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.340, cy, "Temp", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    for num, polymer, solvent, temp in steps:
        ax.text(cl, cy, num, fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.020, cy, polymer, fontsize=7, fontweight="bold",
                color=C_SEP_ENG, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.130, cy, solvent, fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.340, cy, temp, fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.0105

    y -= box_h + GAP

    # ==================================================================
    # 4. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.065
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)

    cy = y - 0.010
    ax.text(cl, cy, "Orchestrator Synthesis",
            fontsize=10, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~11s  |  ~9K tok",
            fontsize=8, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018

    synth_text = (
        "Formatted 9-step selective dissolution sequence at atmospheric pressure.\n"
        "Key insight: Toluene reused at 4 temperature steps (25, 75, 95, 105 C)\n"
        "for PS/LDPE/HDPE/PP via temperature-based selectivity."
    )
    ax.text(cl, cy, synth_text,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP * 2

    # ==================================================================
    # 5. EXECUTION TIMELINE
    # ==================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.018

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.028
    total_s = 69.2

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.004",
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
        ax.text(x0 + w / 2, bar_y + bar_h / 2, label,
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", transform=ax.transAxes, zorder=4)

    _bar_seg(0, 8, C_USER, "route")
    _bar_seg(8, 50, C_SEP_ENG, "separation-engineer (~50s)")
    _bar_seg(58, 11, C_SYNTH, "synth")

    # Time labels
    ty = bar_y - 0.012
    for t in [0, 10, 20, 30, 40, 50, 60, 69]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=7.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.018
    items = [(C_USER, "Orchestrator"), (C_SEP_ENG, "separation-engineer"),
             (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.08
    sp = (bar_w - 0.16) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.018], [ly, ly], color=color, lw=5,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.024, ly, label, fontsize=7.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.028

    # ==================================================================
    # 6. BOTTOM PANELS
    # ==================================================================
    panel_h = 0.120
    panel_gap = 0.010
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.008

    # --- Panel 1: Trace Metadata ---
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.010, "Trace Metadata",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "~69.2 s"),
        ("Total Tokens",   "127K (120K in)"),
        ("Subagent Calls", "1"),
        ("LLM Calls",      "~6 (subagent)"),
        ("Tool Calls",     "8 (6 domain)"),
        ("Total Runs",     "61"),
        ("Pattern",        "single-agent"),
    ]
    my = y - 0.028
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=7.5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.006, my, value, fontsize=7.5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.012

    # --- Panel 2: Execution Pattern ---
    _box(ax, p2x, y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.010, "Execution Pattern",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    pattern = (
        "Single-agent delegation:\n"
        "1. Route: keyword match\n"
        "   -> separation-engineer\n"
        "2. Sub-agent: plans 9-step\n"
        "   sequence, builds matrix\n"
        "3. Guardrail: synthesis\n"
        "   injection + tool limit\n"
        "4. Orchestrator: formats\n"
        "   final answer"
    )
    ax.text(p2x + pcl, y - 0.026, pattern,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    # --- Panel 3: Limitations Found ---
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.010, "Limitations Found",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    limits = (
        "[!] Toluene reused 4x\n"
        "    (cross-contamination?)\n"
        "[!] No TEA/LCA check\n"
        "[!] 127K tok (+90% vs T1)\n"
        "[!] BP margin: Toluene\n"
        "    at 105C (BP 111C)\n"
        "[!] Single-agent only"
    )
    ax.text(p3x + pcl, y - 0.026, limits,
            fontsize=7, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    y -= panel_h + 0.008

    # -- Footer --
    ax.text(MID, y,
            "DISSOLVE  |  Gemini 3 Flash Preview  |  "
            "Trace 019c3044  |  "
            "9-Polymer Separation Case Study (Sanchez-Rivera et al. 2025)",
            ha="center", fontsize=7.5, color=C_BODY,
            transform=ax.transAxes)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "019c3044-1e4d-73f0-a2e9-f47499716e98.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
