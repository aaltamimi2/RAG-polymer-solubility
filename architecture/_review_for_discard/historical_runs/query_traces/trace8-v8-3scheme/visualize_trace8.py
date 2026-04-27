"""
Visualize DISSOLVE agent Trace 8: v8 3-scheme run.
Separation-engineer with plan_multiple_separation_schemes.
Trace ID: 019c55c8-cbe0-7fe1-ac13-11b6c2f80a02
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
C_SYNTH      = "#8B5CF6"   # Violet  -- synthesis

C_S1         = "#F97316"   # Orange  -- scheme 1
C_S2         = "#0EA5E9"   # Sky     -- scheme 2
C_S3         = "#F59E0B"   # Amber   -- scheme 3

C_TITLE      = "#1E293B"
C_BODY       = "#374151"

# -- Uniform font size ---------------------------------------------------------
FS = 7  # all text in figure


# -- Layout constants ----------------------------------------------------------
FIG_W  = 7.0
FIG_H  = 14.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
GAP      = 0.005


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

    y = 0.993

    # ==================================================================
    # 1. USER QUERY — full-width wrap
    # ==================================================================
    query = (
        "Find the optimal separation sequence for a mixed polymer waste stream "
        "containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, and PET. "
        "Use selective dissolution at atmospheric pressure. Propose THREE different "
        "sets of solvents: (1) optimized for maximum selectivity, (2) optimized for "
        "green/safe solvents with high GSK G-scores, and (3) optimized for the "
        "cheapest solvents to minimize operating cost."
    )
    box_h = 0.055
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cy = y - 0.006
    ax.text(cl, cy, "User Query",
            fontsize=FS, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012
    ax.text(cl, cy, _wrap(query, 140),
            fontsize=FS, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ==================================================================
    # 2. ROUTING — title only
    # ==================================================================
    box_h = 0.014
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    cy = y - box_h / 2
    ax.text(cl, cy, "Single-Agent Routing",
            fontsize=FS, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="center", zorder=5)
    y -= box_h + GAP

    # ==================================================================
    # 3. SEPARATION-ENGINEER
    # ==================================================================
    box_h = 0.042
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP_ENG)
    cy = y - 0.006
    ax.text(cl, cy, "1. separation-engineer  (1 invocation, 1 tool call)",
            fontsize=FS, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~55s  |  1 tool call  |  synthesis injected",
            fontsize=FS, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.013

    ax.text(cl, cy, "Tool: plan_multiple_separation_schemes",
            fontsize=FS, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011
    args_text = (
        "Args: polymers=\"PS,PVC,LDPE,HDPE,PP,EVOH,Nylon6,Nylon66,PET\", "
        "temperature=120.0, min_selectivity=5.0"
    )
    ax.text(cl, cy, _wrap(args_text, 140),
            fontsize=FS, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5)

    y -= box_h + GAP

    # ==================================================================
    # 4. THREE SCHEME TABLES
    # ==================================================================
    ax.text(MID, y - 0.003,
            "Three Separation Schemes (from plan_multiple_separation_schemes, 0.23s)",
            ha="center", fontsize=FS, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.008

    col_gap = 0.008
    col_w = (WIDTH - 2 * col_gap) / 3
    c1x = LEFT
    c2x = LEFT + col_w + col_gap
    c3x = LEFT + 2 * (col_w + col_gap)

    ROW_STEP = 0.016
    scheme_h = 0.012 + 0.010 + 0.009 + 9 * ROW_STEP  # title + hdr + gap + 9 rows

    s1_rows = [
        ("PS",       "CH2Cl2",           "35",  "19"),
        ("PVC",      "propanone",        "51",  "21"),
        ("EVOH",     "DMSO",             "120", "27"),
        ("LDPE",     "ethanol",          "73",  "9"),
        ("PP",       "THF",              "60",  "29"),
        ("HDPE",     "cyclohexane",      "76",  "27"),
        ("PET",      "DMF",              "120", "12"),
        ("Nylon 6",  "chloroform",       "56",  "6"),
        ("Nylon 66", "(Isolated)",        "-",   "-"),
    ]
    s2_rows = [
        ("EVOH",     "glycol",           "120", "8.1"),
        ("LDPE",     "cyclohexanol",     "120", "7.5"),
        ("PS",       "ethyl acetate",    "72",  "6.7"),
        ("PVC",      "methyl acetate",   "52",  "6.7"),
        ("PP",       "tert-butanol",     "77",  "6.3"),
        ("HDPE",     "toluene",          "106", "6.0"),
        ("PET",      "DMF",              "120", "5.0"),
        ("Nylon 6",  "chloroform",       "56",  "4.4"),
        ("Nylon 66", "(Isolated)",        "-",   "-"),
    ]
    s3_rows = [
        ("PS",       "isopropylamine",   "27",  "17"),
        ("PVC",      "CH2Cl2",           "35",  "5"),
        ("EVOH",     "methanol",         "60",  "19"),
        ("PP",       "THF",              "60",  "8"),
        ("LDPE",     "propanone",        "51",  "6"),
        ("HDPE",     "hexane",           "64",  "27"),
        ("Nylon 6",  "chloroform",       "56",  "6"),
        ("PET",      "methyl acetate",   "52",  "6"),
        ("Nylon 66", "(Isolated)",        "-",   "-"),
    ]

    schemes = [
        ("Scheme 1: Max Selectivity", C_S1, c1x, s1_rows, ("T(\u00b0C)", "Sel%")),
        ("Scheme 2: Green/Safe (GSK)", C_S2, c2x, s2_rows, ("T(\u00b0C)", "GSK")),
        ("Scheme 3: Lowest Energy", C_S3, c3x, s3_rows, ("T(\u00b0C)", "Sel%")),
    ]

    for title, color, sx, steps, (h3, h4) in schemes:
        _box(ax, sx, y, col_w, scheme_h, color, radius=0.004)
        scl = sx + ACCENT_W + 0.006
        scr = sx + col_w - 0.005

        cy = y - 0.005
        ax.text(scl, cy, title,
                fontsize=FS, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.013

        # Headers
        ax.text(scl, cy, "Polymer", fontsize=FS, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scl + 0.068, cy, "Solvent", fontsize=FS, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scr - 0.040, cy, h3, fontsize=FS, fontweight="bold",
                color=C_TITLE, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(scr, cy, h4, fontsize=FS, fontweight="bold",
                color=C_TITLE, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

        for polymer, solvent, c3, c4 in steps:
            ax.text(scl, cy, polymer, fontsize=FS, fontweight="bold",
                    color=color, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scl + 0.068, cy, solvent, fontsize=FS,
                    color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scr - 0.040, cy, c3, fontsize=FS,
                    color=C_BODY, ha="right",
                    transform=ax.transAxes, va="top", zorder=5)
            ax.text(scr, cy, c4, fontsize=FS,
                    color=C_BODY, ha="right",
                    transform=ax.transAxes, va="top", zorder=5)
            cy -= ROW_STEP

    y -= scheme_h + GAP

    # ==================================================================
    # 5. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.038
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.006
    ax.text(cl, cy, "Orchestrator Synthesis",
            fontsize=FS, fontweight="bold", color=C_SYNTH,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "4 messages total  |  2 AI messages  |  1 tool message",
            fontsize=FS, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012
    synth_body = (
        "Separation-engineer returned complete 3-scheme analysis. Orchestrator "
        "formatted into final answer with rationale per step, pros/cons for each "
        "scheme, and atmospheric pressure compliance verification."
    )
    ax.text(cl, cy, _wrap(synth_body, 140),
            fontsize=FS, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ==================================================================
    # 6. EXECUTION TIMELINE
    # ==================================================================
    y -= 0.008
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=FS, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.016
    total_s = 60.0

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
                    ha="center", va="center", fontsize=FS, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    _bar_seg(0, 4, C_USER, "init")
    _bar_seg(4, 7, C_ROUTER, "route")
    _bar_seg(11, 40, C_SEP_ENG, "sep-eng (40s)")
    _bar_seg(51, 9, C_SYNTH, "synth (9s)")

    ty = bar_y - 0.008
    for t in [0, 10, 20, 30, 40, 50, 60]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=FS, color=C_BODY,
                transform=ax.transAxes)

    ly = ty - 0.012
    items = [(C_USER, "Init"), (C_ROUTER, "Routing"),
             (C_SEP_ENG, "sep-eng"), (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.02
    sp = (bar_w - 0.04) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=FS, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.005

    # -- Clip to content --
    ax.set_ylim(y, 1.0)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace8-v8-3scheme.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.04)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
