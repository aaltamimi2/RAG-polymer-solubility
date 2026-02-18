"""
Visualize DISSOLVE agent pipeline trace:
RAG retrieval -> parallel separation-engineer + biosteam-analyst
for PE/EVOH/PET multi-polymer sequential dissolution.

Style matches trace11-v8-tol-hep-xyl (hand-crafted card layout).
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
        "Which solvents were used to dissolve PE, EVOH, and PET in the STRAP "
        "literature? Find alternative separation sequences prioritizing safety, "
        "and run BioSTEAM TEA/LCA on the literature process and the best alternative."
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
    # 2. ROUTING
    # ================================================================
    box_h = 0.038
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    ax.text(cl, y - 0.008, "Sequential + Parallel Routing",
            fontsize=9, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    route_text = (
        'Phase 1: RAG retrieval (orchestrator)  ->  '
        'Phase 2: separation-engineer || biosteam-analyst (parallel)  ->  '
        'Phase 3: biosteam-analyst (safety-optimal)'
    )
    ax.text(cl, y - 0.020, route_text,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ================================================================
    # 3. RAG RETRIEVAL (orchestrator tool)
    # ================================================================
    box_h = 0.082
    _box(ax, LEFT, y, WIDTH, box_h, C_RAG)
    ax.text(cl, y - 0.008, "Phase 1: RAG Retrieval (STRAP-CORE)",
            fontsize=9, fontweight="bold", color="#0D9488",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.008, "4.2s  |  8 passages  |  0.978-0.998 relevance",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.020
    _pill(ax, cl, cy, "search_literature_rag", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.004

    _divider(ax, cl, cr, cy)
    cy -= 0.008

    ax.text(cl, cy, "Literature STRAP Process (from 4 papers):",
            fontsize=8, fontweight="bold", color="#0D9488",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012
    rag_results = (
        "Step 1: PE in Toluene @ 110C  (14.6 wt% solubility)       |  "
        "Step 2: EVOH in DMSO @ 95C       |  "
        "Step 3: PET remains as undissolved residue"
    )
    ax.text(cl, cy, rag_results,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP * 3

    # ================================================================
    # 4. PARALLEL SUBAGENTS (two columns)
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

    # --- Column 1: Separation Engineer ---
    col_h = 0.195
    _box(ax, c1_x, par_top, col_w, col_h, C_SEP, radius=0.004)
    a_cl = c1_x + ACCENT_W + 0.008
    a_cr = c1_x + col_w - 0.006

    cy = par_top - 0.008
    ax.text(a_cl, cy, "separation-engineer",
            fontsize=8.5, fontweight="bold", color=C_SEP,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "8.5s  |  ~18K tok",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    _pill(ax, a_cl, cy, "plan_sequential_separation", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= PILL_H + 0.004
    _divider(ax, a_cl, a_cr, cy)
    cy -= 0.008

    ax.text(a_cl, cy, "Sequence Ranking (6 evaluated):",
            fontsize=8, fontweight="bold", color="#C2410C",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.013

    rankings = [
        ("#1  EVOH->PET->PE", "19.6%", "#15803D"),
        ("#2  PET->EVOH->PE", "11.7%", C_BODY),
        ("#3  EVOH->PE->PET", " 7.5%", C_BODY),
        ("#4  PE-first seqs", "  neg.", "#B91C1C"),
    ]
    for label, sel, color in rankings:
        ax.text(a_cl, cy, label, fontsize=7, color=color,
                fontweight="bold" if color == "#15803D" else "normal",
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cr, cy, f"sel: {sel}", fontsize=7, color=color,
                ha="right", transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.011

    cy -= 0.004
    ax.text(a_cl, cy, "Best solvents: Methanol + DCM + Toluene",
            fontsize=7, color="#15803D", fontweight="bold", style="italic",
            transform=ax.transAxes, va="top", zorder=5)

    # --- Column 2: BioSTEAM Analyst (literature conditions) ---
    _box(ax, c2_x, par_top, col_w, col_h, C_BIO_LIT, radius=0.004)
    a_cl = c2_x + ACCENT_W + 0.008
    a_cr = c2_x + col_w - 0.006

    cy = par_top - 0.008
    ax.text(a_cl, cy, "biosteam-analyst",
            fontsize=8.5, fontweight="bold", color=C_BIO_LIT,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "35.1s  |  ~22K tok",
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
        ("PE",   "Toluene",  "110C", "$2.00/kg", "1.09"),
        ("EVOH", "DMSO",     " 95C", "$2.37/kg", "1.47"),
        ("PET",  "Eth. Glycol", "120C", "$3.23/kg", "1.97"),
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
    ax.text(a_cl, cy, "Blended: $2.45/kg  |  GWP 1.48  |  TCI $65.3M",
            fontsize=7, color="#6D28D9", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    y = par_top - col_h - GAP

    # ================================================================
    # 5. BIOSTEAM SAFETY-OPTIMAL (full width)
    # ================================================================
    box_h = 0.085
    _box(ax, LEFT, y, WIDTH, box_h, C_BIO_SAFE)
    ax.text(cl, y - 0.008, "Phase 3: BioSTEAM Safety-Optimal Sequence (EVOH->PET->PE)",
            fontsize=9, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.008, "33.2s  |  ~22K tok",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.022
    _pill(ax, cl, cy, "run_biosteam_multi_polymer", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.004
    _divider(ax, cl, cr, cy)
    cy -= 0.008

    safe_results = [
        ("EVOH", "Methanol", " 65C", "$2.40/kg", "1.12"),
        ("PET",  "DCM",      " 40C", "$2.85/kg", "1.21"),
        ("PE",   "Toluene",  "110C", "$2.00/kg", "1.09"),
    ]
    # Inline table
    ax.text(cl, cy, "Polymer", fontsize=7, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.085, cy, "Solvent", fontsize=7, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.26, cy, "MSP", fontsize=7, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.38, cy, "GWP", fontsize=7, fontweight="bold",
            color="#475569", transform=ax.transAxes, va="top", zorder=5)

    # Blended results on right side
    ax.text(cr - 0.28, cy, "Blended MSP:", fontsize=7, fontweight="bold",
            color="#0369A1", transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "$2.40/kg", fontsize=7, fontweight="bold",
            color="#15803D", ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    _divider(ax, cl, cl + 0.44, cy, color="#94A3B8", lw=0.5)
    ax.text(cr - 0.28, cy + 0.002, "Weighted GWP:", fontsize=7, fontweight="bold",
            color="#0369A1", transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy + 0.002, "1.13 kg CO2e/kg", fontsize=7, fontweight="bold",
            color="#15803D", ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.008
    ax.text(cr - 0.28, cy + 0.002, "Combined TCI:", fontsize=7, fontweight="bold",
            color="#0369A1", transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy + 0.002, "$63.6M", fontsize=7, fontweight="bold",
            color="#15803D", ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    for polymer, solvent, temp, msp, gwp in safe_results:
        ax.text(cl, cy, polymer, fontsize=7, color=C_BODY,
                fontweight="bold", transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.085, cy, f"{solvent} @{temp}", fontsize=6.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.26, cy, msp, fontsize=7, color=C_BODY,
                fontfamily="monospace", transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.38, cy, gwp, fontsize=7, color=C_BODY,
                fontfamily="monospace", transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.010

    y -= box_h + GAP * 2

    # ================================================================
    # 6. ORCHESTRATOR SYNTHESIS
    # ================================================================
    box_h = 0.074
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    ax.text(cl, y - 0.008, "Orchestrator Synthesis",
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
        ("Literature (PE->EVOH->PET)", "$2.45", "$65.3M", "1.48", "baseline"),
        ("Safety-opt (EVOH->PET->PE)", "$2.40", "$63.6M", "1.13", "-24% GWP"),
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
            "Winner: EVOH->PET->PE  --  lowest MSP, TCI, and GWP; "
            "24% GWP reduction from replacing DMSO/EG with Methanol/DCM",
            fontsize=7, color="#15803D", fontweight="bold", style="italic",
            transform=ax.transAxes, va="top", zorder=5)

    y -= box_h + GAP * 2

    # ================================================================
    # 7. EXECUTION TIMELINE (Gantt chart)
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

    # Three rows: RAG | sep-eng | biosteam
    row_labels = ["RAG", "Sep-Eng", "BioSTEAM"]
    row_colors_bar = [C_RAG, C_SEP, C_BIO_LIT]
    rows_y = []

    for i in range(3):
        by = y - (i + 1) * (bar_h + 0.004)
        rows_y.append(by)
        ax.add_patch(FancyBboxPatch((bar_left, by), bar_w, bar_h,
                                    transform=ax.transAxes, zorder=2, **bg_kw))

        # Row label on left
        ax.text(bar_left - 0.005, by + bar_h / 2, row_labels[i],
                ha="right", va="center", fontsize=7, color=row_colors_bar[i],
                fontweight="bold", transform=ax.transAxes)

    # RAG row: orchestrator route (0-1s) + RAG query (1-5.2s)
    _bar_seg(ax, rows_y[0], 0, 1.0, C_USER, "")
    _bar_seg(ax, rows_y[0], 1.0, 4.2, C_RAG, "RAG (4.2s)")

    # Sep-eng row: idle during RAG, then runs 6-14.5s
    _bar_seg(ax, rows_y[1], 6.0, 8.5, C_SEP, "sep-eng (8.5s)")

    # BioSTEAM row: literature 6-41s, then safety-optimal 41-74s
    _bar_seg(ax, rows_y[2], 6.0, 35.1, C_BIO_LIT, "literature (35.1s)")
    _bar_seg(ax, rows_y[2], 41.5, 33.2, C_BIO_SAFE, "safety-opt (33.2s)")

    # Synthesis block spanning all 3 rows at the end
    synth_start = 68.0
    synth_dur = 4.0
    sx = bar_left + (synth_start / total_s) * bar_w
    sw = (synth_dur / total_s) * bar_w
    sh = rows_y[0] + bar_h - rows_y[2]
    ax.add_patch(FancyBboxPatch(
        (sx, rows_y[2]), sw, sh,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_SYNTH, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(sx + sw / 2, rows_y[2] + sh / 2, "synth",
            ha="center", va="center", fontsize=6, color="white",
            fontweight="bold", transform=ax.transAxes, zorder=4)

    # Time axis labels
    ty = rows_y[2] - 0.010
    for t in [0, 10, 20, 30, 40, 50, 60, 72]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t:.0f}s",
                ha="center", fontsize=7, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.016
    items = [(C_USER, "Orchestrator"), (C_RAG, "RAG"),
             (C_SEP, "Sep-Engineer"), (C_BIO_LIT, "BioSTEAM (lit)"),
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
    # 8. BOTTOM PANELS (three columns)
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
        ("Run Time",       "72.0s"),
        ("Total Tokens",   "28,400"),
        ("Subagent Calls", "2 (parallel)"),
        ("Tool Calls",     "4 domain"),
        ("BioSTEAM Sims",  "6 (3+3)"),
        ("Pattern",        "seq + parallel"),
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
        "  MSP: $2.40/kg\n"
        "  GWP: 1.13 kg CO2e/kg\n"
        "  TCI: $63.6M\n"
        "  24% GWP reduction"
    )
    ax.text(p2x + pcl, y - 0.020, ranking,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    # --- Panel 3: Pipeline Pattern ---
    _box(ax, p3x, y, pw, panel_h, C_SEP, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.008, "Pipeline Pattern",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    pattern = (
        "Multi-agent pipeline:\n"
        "1. RAG: 8 passages, 4 papers\n"
        "2. Sep-eng: 6 sequences\n"
        "3. BioSTEAM: lit conditions\n"
        "4. BioSTEAM: safety-optimal\n"
        "5. Compare + recommend"
    )
    ax.text(p3x + pcl, y - 0.020, pattern,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    y -= panel_h + 0.006

    # ── Footer ──
    ax.text(MID, y,
            "DISSOLVE v8  |  PE/EVOH/PET Multi-Polymer Pipeline  |  "
            "RAG -> Separation Engineer || BioSTEAM Analyst",
            ha="center", fontsize=7, color=C_BODY,
            transform=ax.transAxes)

    # ── Save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace13-pe-evoh-pet-pipeline.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
