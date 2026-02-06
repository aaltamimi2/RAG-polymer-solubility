"""
Visualize DISSOLVE agent parallel-subagent trace (safety-analyst || tea-lca-analyst).
Parallel execution: safety assessment + cost analysis for LDPE/EVOH.
Stacked-card style matching architecture/visualize_chain_trace.py.

Trace ID: 019c3021-b123-70a2-bd02-45ac13f686fc
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
C_SAFETY     = "#10B981"   # Emerald — safety-analyst
C_TEA_LCA    = "#8B5CF6"   # Violet — tea-lca-analyst
C_SYNTH      = "#0EA5E9"   # Sky blue — synthesis
C_WARN       = "#F59E0B"   # Amber — warnings

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"


# ── Layout constants ─────────────────────────────────────────────────
FIG_W  = 7.0
FIG_H  = 14.0            # inches — will use bbox_inches="tight"

# All y-coordinates are in axes fraction (0=bottom, 1=top).
# We use a descending y-cursor so items are placed top-to-bottom.

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015          # inner padding from box edges
ACCENT_W = 0.006
LINE_H   = 0.016          # height consumed by one line of text (fontsize 8)
LINE_S   = 0.013          # height consumed by one line of small text (fontsize 7)
PILL_H   = 0.020          # height consumed by one row of pills
GAP      = 0.008          # vertical gap between sections


# ── Helpers ──────────────────────────────────────────────────────────

def _box(ax, x, y_top, w, h, accent_color, radius=0.005):
    """Draw a rounded box with left accent bar. Returns y_bottom."""
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
    """Draw one pill and return its width."""
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
    """Draw a row of pills, wrapping to next row if needed. Returns rows used."""
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


# ── Main figure ──────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    # Descending y-cursor
    y = 0.985

    # ================================================================
    # 1. USER QUERY
    # ================================================================
    query = (
        "Find the safest solvents and the lowest operating cost solvents "
        "relevant to LDPE and EVOH. Run the safety assessment and cost "
        "analysis in parallel, then combine results to determine which "
        "solvents could potentially treat one polymer differently from "
        "the other."
    )
    box_h = 0.052
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD
    ax.text(cl, y - 0.010, "User Query",
            fontsize=10, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl, y - 0.022, _wrap(query, 100),
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ================================================================
    # 2. ROUTING
    # ================================================================
    box_h = 0.048
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    ax.text(cl, y - 0.010, "Parallel Routing",
            fontsize=10, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    route_text = (
        'Keyword matches:  "safe" -> safety-analyst (score 2)   |   '
        '"operating cost" -> tea-lca-analyst (score 3)\n'
        'Pair {safety-analyst, tea-lca-analyst} in PARALLEL_PAIRS  '
        '->  launch both task() calls in one LLM response'
    )
    ax.text(cl, y - 0.024, route_text,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    # ================================================================
    # 3. PARALLEL SUBAGENTS (side-by-side)
    # ================================================================
    col_gap = 0.014
    col_w = (WIDTH - col_gap) / 2
    left_x = LEFT
    right_x = LEFT + col_w + col_gap
    par_top = y  # remember where we start

    # We'll draw both columns, tracking how far down each goes,
    # then set y to the lower of the two.

    # ── Left column: safety-analyst ──
    sa_h = 0.185
    _box(ax, left_x, par_top, col_w, sa_h, C_SAFETY, radius=0.004)
    sa_cl = left_x + ACCENT_W + 0.010
    sa_cr = left_x + col_w - 0.008
    sa_cw = sa_cr - sa_cl

    cy = par_top - 0.012  # cursor inside box
    ax.text(sa_cl, cy, "safety-analyst",
            fontsize=9, fontweight="bold", color=C_SAFETY,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(sa_cr, cy, "7.9s  |  9,892 tok",
            fontsize=7.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018

    ax.text(sa_cl, cy,
            "Assess safety profiles of solvents\nrelevant to LDPE and EVOH.",
            fontsize=7.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.032

    _divider(ax, sa_cl, sa_cr, cy)
    cy -= 0.012

    ax.text(sa_cl, cy, "Tool Calls: NONE",
            fontsize=8, fontweight="bold", color=C_WARN,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018

    _pill(ax, sa_cl, cy, "LLM knowledge only", C_WARN, "#FEF3C7", C_WARN, fs=7)
    cy -= PILL_H

    _divider(ax, sa_cl, sa_cr, cy)
    cy -= 0.012

    ax.text(sa_cl, cy, "Output:",
            fontsize=8, fontweight="bold", color="#065F46",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016
    ax.text(sa_cl, cy,
            "DMSO (G-score ~9/10, low tox)\n"
            "Ethanol/Water (safe for EVOH)\n"
            "Cyclohexane (greener for LDPE)",
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)

    # ── Right column: tea-lca-analyst ──
    tla_h = sa_h  # same height
    _box(ax, right_x, par_top, col_w, tla_h, C_TEA_LCA, radius=0.004)
    tla_cl = right_x + ACCENT_W + 0.010
    tla_cr = right_x + col_w - 0.008
    tla_cw = tla_cr - tla_cl

    cy = par_top - 0.012
    ax.text(tla_cl, cy, "tea-lca-analyst",
            fontsize=9, fontweight="bold", color=C_TEA_LCA,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(tla_cr, cy, "28.4s  |  82,444 tok",
            fontsize=7.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018

    ax.text(tla_cl, cy,
            "Analyze operating costs and economics\nof solvents for LDPE and EVOH.",
            fontsize=7.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.3)
    cy -= 0.032

    _divider(ax, tla_cl, tla_cr, cy)
    cy -= 0.012

    ax.text(tla_cl, cy, "Tool Calls (7 domain / 10 total):",
            fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016

    tool_labels = [
        "compare_tea_lca x2",
        "compare_strap",
        "gen_tea_lca_viz x2",
        "analyze_lca x2",
    ]
    rows = _pills_row(ax, tla_cl, cy, tool_labels, tla_cw,
                       C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= rows * PILL_H + 0.006

    # Guardrail pill
    _pill(ax, tla_cl, cy, "LIMIT: 10/10 tool calls", C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=6)
    cy -= PILL_H

    _divider(ax, tla_cl, tla_cr, cy)
    cy -= 0.012

    ax.text(tla_cl, cy, "Output:",
            fontsize=8, fontweight="bold", color="#5B21B6",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.016
    ax.text(tla_cl, cy,
            "Xylenes: cheapest for LDPE\n"
            "Ethanol/Water 70/30: cheapest EVOH\n"
            "STRAP process MSP analyzed",
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)

    # Fork arrows from routing to both boxes
    arrow_kw = dict(arrowstyle="-|>", color=C_ROUTER, lw=1.2,
                    connectionstyle="arc3,rad=0")
    ax.annotate("", xy=(left_x + col_w / 2, par_top + 0.002),
                xytext=(MID, par_top + GAP),
                arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)
    ax.annotate("", xy=(right_x + col_w / 2, par_top + 0.002),
                xytext=(MID, par_top + GAP),
                arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)

    y = par_top - sa_h - GAP

    # ================================================================
    # 4. ORCHESTRATOR SYNTHESIS
    # ================================================================
    box_h = 0.075
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    ax.text(cl, y - 0.010, "Orchestrator Synthesis",
            fontsize=10, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.010, "16.9s  |  11,113 tok",
            fontsize=8, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    synth = (
        "Recommended strategy for LDPE / EVOH:\n"
        "1. Ethanol/Water at 75 C  ->  dissolve EVOH (safest + cheapest)\n"
        "2. Cyclohexane at 90 C  ->  dissolve LDPE (safer than aromatics)\n"
        "Note: BP constraint flagged  --  Cyclohexane BP 81 C vs 90 C operation"
    )
    ax.text(cl, y - 0.026, synth,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP * 2

    # ================================================================
    # 5. EXECUTION TIMELINE
    # ================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.020

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.022
    total_s = 51.5

    def _bar_seg(ax, y_bar, t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_s) * bar_w
        w = (t_dur / total_s) * bar_w
        ax.add_patch(FancyBboxPatch(
            (x0, y_bar), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        ax.text(x0 + w / 2, y_bar + bar_h / 2, label,
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", transform=ax.transAxes, zorder=4)

    # Row 1: safety-analyst timeline
    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)
    by1 = y - bar_h
    ax.add_patch(FancyBboxPatch((bar_left, by1), bar_w, bar_h,
                                transform=ax.transAxes, zorder=2, **bg_kw))
    _bar_seg(ax, by1, 0, 6, C_USER, "route")
    _bar_seg(ax, by1, 6, 7.9, C_SAFETY, "safety (7.9s)")

    # Row 2: tea-lca-analyst timeline
    by2 = by1 - bar_h - 0.004
    ax.add_patch(FancyBboxPatch((bar_left, by2), bar_w, bar_h,
                                transform=ax.transAxes, zorder=2, **bg_kw))
    _bar_seg(ax, by2, 0, 6, C_USER, "")
    _bar_seg(ax, by2, 6, 28.4, C_TEA_LCA, "tea-lca-analyst (28.4s)")

    # Synthesis block spanning both rows
    synth_t = 34
    synth_dur = 17.5
    sx = bar_left + (synth_t / total_s) * bar_w
    sw = (synth_dur / total_s) * bar_w
    sh = bar_h * 2 + 0.004
    ax.add_patch(FancyBboxPatch(
        (sx, by2), sw, sh,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_SYNTH, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(sx + sw / 2, by2 + sh / 2, "synthesis (16.9s)",
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold", transform=ax.transAxes, zorder=4)

    # Time labels
    ty = by2 - 0.012
    for t in [0, 10, 20, 30, 40, 51.5]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t:.0f}s" if t == int(t) else f"{t}s",
                ha="center", fontsize=7.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.018
    items = [(C_USER, "Orchestrator"), (C_SAFETY, "safety-analyst"),
             (C_TEA_LCA, "tea-lca-analyst"), (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.03
    sp = (bar_w - 0.06) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.018], [ly, ly], color=color, lw=5,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.024, ly, label, fontsize=7.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.028

    # ================================================================
    # 6. BOTTOM PANELS (three columns)
    # ================================================================
    panel_h = 0.108
    panel_gap = 0.010
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.008  # inner left of panel content

    # --- Panel 1: Trace Metadata ---
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.010, "Trace Metadata",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "~51.5 s"),
        ("Total Tokens",   "112K (106K in)"),
        ("Subagent Calls", "2 (parallel)"),
        ("LLM Calls",      "14 total"),
        ("Tool Calls",     "7 domain"),
        ("Pattern",        "parallel"),
    ]
    my = y - 0.028
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=7.5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.006, my, value, fontsize=7.5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.013

    # --- Panel 2: Execution Pattern ---
    _box(ax, p2x, y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.010, "Execution Pattern",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    pattern = (
        "Parallel delegation:\n"
        "1. Route: match 2 agents\n"
        "2. safety: fast, 0 tools\n"
        "3. tea-lca: 7 tools, limit\n"
        "4. Orchestrator synthesizes"
    )
    ax.text(p2x + pcl, y - 0.026, pattern,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    # --- Panel 3: Limitations Found ---
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.010, "Limitations Found",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    limits = (
        "[!] safety: 0 tool calls\n"
        "    (claims ungrounded)\n"
        "[!] 3.6x duration gap\n"
        "[!] No DB cross-check\n"
        "[!] 112K tok (+67%)"
    )
    ax.text(p3x + pcl, y - 0.026, limits,
            fontsize=7.5, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    y -= panel_h + 0.008

    # ── Footer ──
    ax.text(MID, y,
            "DISSOLVE  |  Gemini 3 Flash Preview  |  "
            "Trace 019c3021  |  "
            "Parallel Safety + Cost for LDPE/EVOH",
            ha="center", fontsize=7.5, color=C_BODY,
            transform=ax.transAxes)

    # ── Save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "019c3021-b123-70a2-bd02-45ac13f686fc.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
