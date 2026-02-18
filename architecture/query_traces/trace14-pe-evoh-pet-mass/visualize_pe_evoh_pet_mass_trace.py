"""
Visualize DISSOLVE agent pipeline trace:
Re-analysis of trace13 PE/EVOH/PET pipeline with MASS allocation
(vs. value-weighted allocation in trace13).

Style matches trace11/trace13 (hand-crafted card layout).
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


# ── Color palette ────────────────────────────────────────────────────
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"
C_BOX_BORDER = "#E5E7EB"

C_USER       = "#6366F1"   # Indigo — orchestrator / user
C_ROUTER     = "#22C55E"   # Green — routing
C_RAG        = "#14B8A6"   # Teal — RAG retrieval
C_SEP        = "#F97316"   # Orange — separation-engineer
C_BIO_LIT    = "#8B5CF6"   # Violet — biosteam (literature)
C_BIO_SAFE   = "#0EA5E9"   # Sky — biosteam (safety-optimal)
C_SYNTH      = "#64748B"   # Slate — synthesis
C_VERIFIER   = "#10B981"   # Emerald — verifier / win
C_ALLOC      = "#E11D48"   # Rose — allocation method highlight

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"


# ── Layout constants ─────────────────────────────────────────────────
FIG_W  = 7.5
FIG_H  = 17.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
LINE_H   = 0.013
LINE_S   = 0.011
PILL_H   = 0.016
GAP      = 0.006


# ── Helpers ──────────────────────────────────────────────────────────

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


def _wrap(text, width=95):
    lines = text.split("\n")
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width) if len(line) > width else [line])
    return "\n".join(out)


# ── Main figure ──────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    y = 0.988
    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD

    # ================================================================
    # 1. USER QUERY
    # ================================================================
    query = (
        "Re-run PE/EVOH/PET multi-polymer BioSTEAM analysis using MASS "
        "allocation (equal weight per polymer) instead of value-weighted "
        "allocation. Compare literature vs. safety-optimal sequences."
    )
    box_h = 0.042
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    ax.text(cl, y - 0.008, "User Query",
            fontsize=9, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl, y - 0.018, _wrap(query, 115),
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ================================================================
    # 2. ALLOCATION METHOD
    # ================================================================
    box_h = 0.058
    _box(ax, LEFT, y, WIDTH, box_h, C_ALLOC)
    ax.text(cl, y - 0.008, "Allocation Method: Mass vs. Value",
            fontsize=9, fontweight="bold", color=C_ALLOC,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.008, "trace13 -> trace14",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.020
    alloc_text = (
        "Value-weighted (trace13): weight = market value  "
        "(PE: $1.10, EVOH: $4.50, PET: $1.05/kg)\n"
        "Mass allocation  (trace14): weight = 1.0 for all polymers  "
        "(equal cost share per kg recovered)"
    )
    ax.text(cl, cy, alloc_text,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.5)
    cy -= 0.024
    ax.text(cl, cy,
            "Mass allocation assigns equal economic importance per kg — "
            "no premium for specialty polymers like EVOH.",
            fontsize=7, color="#9F1239", fontweight="bold", style="italic",
            transform=ax.transAxes, va="top", zorder=5)
    y -= box_h + GAP * 3

    # ================================================================
    # 3. PARALLEL SUBAGENTS (two columns)
    # ================================================================
    col_gap_inner = 0.010
    col_w = (WIDTH - col_gap_inner) / 2
    c1_x = LEFT
    c2_x = LEFT + col_w + col_gap_inner
    par_top = y

    # Fork arrows
    arrow_kw = dict(arrowstyle="-|>,head_length=0.4,head_width=0.25",
                    color=C_ROUTER, lw=2.0,
                    connectionstyle="arc3,rad=0")
    ax.annotate("", xy=(c1_x + col_w / 2, par_top + 0.002),
                xytext=(MID, par_top + GAP * 3 + 0.003),
                arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)
    ax.annotate("", xy=(c2_x + col_w / 2, par_top + 0.002),
                xytext=(MID, par_top + GAP * 3 + 0.003),
                arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)

    # --- Column 1: BioSTEAM Literature (mass) ---
    col_h = 0.168
    _box(ax, c1_x, par_top, col_w, col_h, C_BIO_LIT, radius=0.004)
    a_cl = c1_x + ACCENT_W + 0.008
    a_cr = c1_x + col_w - 0.006

    cy = par_top - 0.008
    ax.text(a_cl, cy, "biosteam-analyst (literature)",
            fontsize=8.5, fontweight="bold", color=C_BIO_LIT,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "34.3s  |  mass alloc",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    _pill(ax, a_cl, cy, "run_biosteam_multi_polymer", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= PILL_H + 0.004
    _divider(ax, a_cl, a_cr, cy)
    cy -= 0.008

    ax.text(a_cl, cy, "Literature Process (PE->EVOH->PET):",
            fontsize=8, fontweight="bold", color="#6D28D9",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.013

    lit_results = [
        ("PE",   "Toluene",     "110C", "$1.14/kg", "1.05"),
        ("EVOH", "DMSO",        " 95C", "$0.95/kg", "1.17"),
        ("PET",  "Eth. Glycol", "120C", "$0.97/kg", "1.30"),
    ]
    # Mini header
    ax.text(a_cl, cy, "Polymer", fontsize=6.5, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cl + 0.065, cy, "Solvent", fontsize=6.5, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cl + 0.185, cy, "MSP", fontsize=6.5, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "GWP", fontsize=6.5, fontweight="bold",
            color="#475569", ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    _divider(ax, a_cl, a_cr, cy, color="#94A3B8", lw=0.5)
    cy -= 0.008

    for polymer, solvent, temp, msp, gwp in lit_results:
        ax.text(a_cl, cy, polymer, fontsize=7, color=C_BODY,
                fontweight="bold", transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cl + 0.065, cy, f"{solvent} @{temp}", fontsize=6.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cl + 0.185, cy, msp, fontsize=7, color=C_BODY,
                fontfamily="monospace", transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cr, cy, gwp, fontsize=7, color=C_BODY,
                fontfamily="monospace", ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.011

    cy -= 0.004
    ax.text(a_cl, cy, "Blended: $1.02/kg  |  GWP 1.17  |  TCI $221.9M",
            fontsize=7, color="#6D28D9", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    # --- Column 2: BioSTEAM Safety-Optimal (mass) ---
    _box(ax, c2_x, par_top, col_w, col_h, C_BIO_SAFE, radius=0.004)
    a_cl = c2_x + ACCENT_W + 0.008
    a_cr = c2_x + col_w - 0.006

    cy = par_top - 0.008
    ax.text(a_cl, cy, "biosteam-analyst (safety-opt)",
            fontsize=8.5, fontweight="bold", color=C_BIO_SAFE,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "34.1s  |  mass alloc",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    _pill(ax, a_cl, cy, "run_biosteam_multi_polymer", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= PILL_H + 0.004
    _divider(ax, a_cl, a_cr, cy)
    cy -= 0.008

    ax.text(a_cl, cy, "Safety-Optimal (EVOH->PET->PE):",
            fontsize=8, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.013

    safe_results = [
        ("EVOH", "Methanol", " 65C", "$0.98/kg", "0.94"),
        ("PET",  "DCM",      " 40C", "$0.86/kg", "0.88"),
        ("PE",   "Toluene",  "110C", "$1.14/kg", "1.05"),
    ]
    ax.text(a_cl, cy, "Polymer", fontsize=6.5, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cl + 0.065, cy, "Solvent", fontsize=6.5, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cl + 0.185, cy, "MSP", fontsize=6.5, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "GWP", fontsize=6.5, fontweight="bold",
            color="#475569", ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    _divider(ax, a_cl, a_cr, cy, color="#94A3B8", lw=0.5)
    cy -= 0.008

    for polymer, solvent, temp, msp, gwp in safe_results:
        ax.text(a_cl, cy, polymer, fontsize=7, color=C_BODY,
                fontweight="bold", transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cl + 0.065, cy, f"{solvent} @{temp}", fontsize=6.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cl + 0.185, cy, msp, fontsize=7, color=C_BODY,
                fontfamily="monospace", transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cr, cy, gwp, fontsize=7, color=C_BODY,
                fontfamily="monospace", ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.011

    cy -= 0.004
    ax.text(a_cl, cy, "Blended: $0.99/kg  |  GWP 0.96  |  TCI $216.4M",
            fontsize=7, color="#0369A1", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    y = par_top - col_h - GAP * 2

    # ================================================================
    # 4. ORCHESTRATOR SYNTHESIS — MASS ALLOCATION COMPARISON
    # ================================================================
    box_h = 0.074
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    ax.text(cl, y - 0.008, "Orchestrator Synthesis (Mass Allocation)",
            fontsize=9, fontweight="bold", color="#334155",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.008, "comparison: PASS",
            fontsize=7, color=C_VERIFIER, ha="right", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    # Comparison table
    tab_top = y - 0.022
    tab_cols = [cl, cl + 0.28, cl + 0.45, cl + 0.62, cl + 0.78]
    headers = ["Sequence", "MSP ($/kg)", "TCI ($M)", "GWP", "Delta"]
    for cx, hdr in zip(tab_cols, headers):
        ax.text(cx, tab_top, hdr, fontsize=7, fontweight="bold",
                color="#475569", transform=ax.transAxes, va="top", zorder=5)
    _divider(ax, cl, cr - 0.02, tab_top - 0.008, color="#94A3B8", lw=0.8)

    rows = [
        ("Literature (PE->EVOH->PET)", "$1.02", "$221.9M", "1.17", "baseline"),
        ("Safety-opt (EVOH->PET->PE)", "$0.99", "$216.4M", "0.96", "-19% GWP"),
    ]
    row_colors = ["#B91C1C", "#15803D"]
    for i, (seq, msp, tci, gwp, delta) in enumerate(rows):
        ry = tab_top - 0.014 - i * 0.012
        vals = [seq, msp, tci, gwp, delta]
        for cx, val in zip(tab_cols, vals):
            ax.text(cx, ry, val, fontsize=7, color=row_colors[i],
                    fontweight="bold" if i == 1 else "normal",
                    transform=ax.transAxes, va="top", zorder=5)

    # Winner callout
    ax.text(cl, tab_top - 0.044,
            "Winner: EVOH->PET->PE  --  $0.99/kg MSP, GWP 0.96; "
            "19% GWP reduction with equal-weight cost allocation",
            fontsize=7, color="#15803D", fontweight="bold", style="italic",
            transform=ax.transAxes, va="top", zorder=5)

    y -= box_h + GAP * 2

    # ================================================================
    # 5. CROSS-TRACE COMPARISON (trace13 vs trace14)
    # ================================================================
    box_h = 0.105
    _box(ax, LEFT, y, WIDTH, box_h, C_ALLOC)
    ax.text(cl, y - 0.008, "Cross-Trace Comparison: Value vs. Mass Allocation",
            fontsize=9, fontweight="bold", color=C_ALLOC,
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.024
    xtab_cols = [cl, cl + 0.16, cl + 0.30, cl + 0.44, cl + 0.58, cl + 0.72]
    xtab_hdrs = ["", "Alloc", "Lit MSP", "Safe MSP", "Lit GWP", "Safe GWP"]
    for cx, hdr in zip(xtab_cols, xtab_hdrs):
        ax.text(cx, cy, hdr, fontsize=7, fontweight="bold",
                color="#475569", transform=ax.transAxes, va="top", zorder=5)
    _divider(ax, cl, cr - 0.08, cy - 0.008, color="#94A3B8", lw=0.8)

    cross_rows = [
        ("Trace 13", "value", "$2.45", "$2.40", "1.48", "1.13"),
        ("Trace 14", "mass",  "$1.02", "$0.99", "1.17", "0.96"),
    ]
    cross_colors = ["#6D28D9", "#0369A1"]
    for i, (trace, alloc, lit_msp, safe_msp, lit_gwp, safe_gwp) in enumerate(cross_rows):
        ry = cy - 0.014 - i * 0.012
        vals = [trace, alloc, lit_msp, safe_msp, lit_gwp, safe_gwp]
        for cx, val in zip(xtab_cols, vals):
            ax.text(cx, ry, val, fontsize=7,
                    color=cross_colors[i],
                    fontweight="bold" if i == 1 else "normal",
                    transform=ax.transAxes, va="top", zorder=5)

    # Explanation
    cy = cy - 0.042
    _divider(ax, cl, cr - 0.08, cy + 0.004, color="#D1D5DB", lw=0.5)
    explain = (
        "Mass allocation produces lower absolute MSP because EVOH's $4.50/kg market value "
        "no longer inflates its cost share.\n"
        "The safety-optimal advantage persists: -19% GWP (mass) vs -24% GWP (value). "
        "Both methods confirm EVOH->PET->PE as the optimal sequence."
    )
    ax.text(cl, cy, _wrap(explain, 110),
            fontsize=7, color="#9F1239", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)

    y -= box_h + GAP * 2

    # ================================================================
    # 6. EXECUTION TIMELINE (Gantt chart)
    # ================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.004

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.016
    total_s = 72.0

    def _bar_seg(ax, y_bar, t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_s) * bar_w
        w = (t_dur / total_s) * bar_w
        ax.add_patch(FancyBboxPatch(
            (x0, y_bar), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if label and w > 0.04:
            ax.text(x0 + w / 2, y_bar + bar_h / 2, label,
                    ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)

    # Two rows: BioSTEAM-lit | BioSTEAM-safe
    row_labels = ["BioSTEAM\n(lit)", "BioSTEAM\n(safe)"]
    row_colors_bar = [C_BIO_LIT, C_BIO_SAFE]
    rows_y = []

    for i in range(2):
        by = y - (i + 1) * (bar_h + 0.004)
        rows_y.append(by)
        ax.add_patch(FancyBboxPatch((bar_left, by), bar_w, bar_h,
                                    transform=ax.transAxes, zorder=2, **bg_kw))
        ax.text(bar_left - 0.005, by + bar_h / 2, row_labels[i],
                ha="right", va="center", fontsize=6.5, color=row_colors_bar[i],
                fontweight="bold", transform=ax.transAxes, linespacing=1.1)

    # BioSTEAM-lit row: runs 0-34.3s
    _bar_seg(ax, rows_y[0], 0, 34.3, C_BIO_LIT, "literature (34.3s)")

    # BioSTEAM-safe row: runs 0-34.1s (parallel)
    _bar_seg(ax, rows_y[1], 0, 34.1, C_BIO_SAFE, "safety-opt (34.1s)")

    # Synthesis block at the end
    synth_start = 35.0
    synth_dur = 3.0
    sx = bar_left + (synth_start / total_s) * bar_w
    sw = (synth_dur / total_s) * bar_w
    sh = rows_y[0] + bar_h - rows_y[1]
    ax.add_patch(FancyBboxPatch(
        (sx, rows_y[1]), sw, sh,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_SYNTH, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(sx + sw / 2, rows_y[1] + sh / 2, "synth",
            ha="center", va="center", fontsize=6, color="white",
            fontweight="bold", transform=ax.transAxes, zorder=4)

    # Time axis labels
    ty = rows_y[1] - 0.010
    for t in [0, 5, 10, 15, 20, 25, 30, 35, 38]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t:.0f}s",
                ha="center", fontsize=7, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.016
    items = [(C_BIO_LIT, "BioSTEAM (lit)"),
             (C_BIO_SAFE, "BioSTEAM (safe)"), (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.005
    sp = (bar_w - 0.01) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.014], [ly, ly], color=color, lw=4.5,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.019, ly, label, fontsize=6.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.022

    # ================================================================
    # 7. BOTTOM PANELS (three columns)
    # ================================================================
    panel_h = 0.090
    panel_gap = 0.010
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.008

    # --- Panel 1: Trace Metadata ---
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.008, "Trace Metadata",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "~38s"),
        ("Allocation",     "mass (1.0)"),
        ("Base Trace",     "trace13"),
        ("Tool Calls",     "2 domain"),
        ("BioSTEAM Sims",  "6 (3+3)"),
        ("Pattern",        "parallel re-run"),
    ]
    my = y - 0.022
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.006, my, value, fontsize=7,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.011

    # --- Panel 2: Results Summary ---
    _box(ax, p2x, y, pw, panel_h, C_VERIFIER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.008, "Key Results",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    ranking = (
        "Best sequence:\n"
        "  EVOH -> PET -> PE\n"
        "  MSP: $0.99/kg\n"
        "  GWP: 0.96 kg CO2e/kg\n"
        "  TCI: $216.4M\n"
        "  19% GWP reduction"
    )
    ax.text(p2x + pcl, y - 0.020, ranking,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    # --- Panel 3: Mass vs Value ---
    _box(ax, p3x, y, pw, panel_h, C_ALLOC, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.008, "Allocation Impact",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    pattern = (
        "Mass allocation effect:\n"
        "  EVOH MSP: lower (no\n"
        "    $4.50 premium weight)\n"
        "  Blended MSP: lower\n"
        "  GWP advantage: -19%\n"
        "  Same winner sequence"
    )
    ax.text(p3x + pcl, y - 0.020, pattern,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    y -= panel_h + 0.006

    # ── Footer ──
    ax.text(MID, y,
            "DISSOLVE v8  |  PE/EVOH/PET Mass Allocation  |  "
            "Trace 14 (re-analysis of Trace 13)",
            ha="center", fontsize=7, color=C_BODY,
            transform=ax.transAxes)

    # ── Save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace14-pe-evoh-pet-mass.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
