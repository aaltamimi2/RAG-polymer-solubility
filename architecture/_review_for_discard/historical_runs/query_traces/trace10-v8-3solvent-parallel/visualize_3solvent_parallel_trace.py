"""
Visualize DISSOLVE agent parallel-subagent trace:
3 x biosteam-analyst running Toluene, Heptane, Methylcyclohexane in parallel.

Trace ID: 019c58bb-0482-7a31-9206-51296ec37fc9
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
C_BIO1       = "#F97316"   # Orange — biosteam-analyst #1 (Toluene)
C_BIO2       = "#8B5CF6"   # Violet — biosteam-analyst #2 (Heptane)
C_BIO3       = "#0EA5E9"   # Sky — biosteam-analyst #3 (MCH)
C_SYNTH      = "#64748B"   # Slate — synthesis
C_VERIFIER   = "#10B981"   # Emerald — verifier

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"


# ── Layout constants ─────────────────────────────────────────────────
FIG_W  = 7.5
FIG_H  = 14.5

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
LINE_H   = 0.016
LINE_S   = 0.013
PILL_H   = 0.020
GAP      = 0.008


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

    y = 0.985

    # ================================================================
    # 1. USER QUERY
    # ================================================================
    query = (
        "Compare the techno-economic performance of three PE solvents in "
        "parallel: Toluene, Heptane, and Methylcyclohexane. Run BioSTEAM "
        "simulations for each under energy case C1 and report MSP, GWP, "
        "and TCI side by side."
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
        'LLM classifier:  "BioSTEAM" + "techno-economic" -> '
        'biosteam-analyst (confidence: HIGH)\n'
        '3 solvents requested  ->  orchestrator launches 3 parallel '
        'task() calls, one per solvent'
    )
    ax.text(cl, y - 0.024, route_text,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP * 3  # extra space for fork arrows

    # ================================================================
    # 3. PARALLEL SUBAGENTS (three columns side-by-side)
    # ================================================================
    col_gap = 0.010
    col_w = (WIDTH - 2 * col_gap) / 3
    c1_x = LEFT
    c2_x = LEFT + col_w + col_gap
    c3_x = LEFT + 2 * (col_w + col_gap)
    par_top = y

    # Subagent data
    agents = [
        {
            "name": "biosteam-analyst",
            "color": C_BIO1,
            "header_color": "#C2410C",
            "output_color": "#9A3412",
            "x": c1_x,
            "solvent": "Toluene",
            "time": "15.1s",
            "tokens": "~21K tok",
            "tool": "run_biosteam_simulation",
            "sim_time": "13.7s",
            "msp": "$1.14/kg",
            "gwp": "1.05 kg CO2e/kg",
            "tci": "$83.34M",
        },
        {
            "name": "biosteam-analyst",
            "color": C_BIO2,
            "header_color": "#6D28D9",
            "output_color": "#5B21B6",
            "x": c2_x,
            "solvent": "Heptane",
            "time": "13.9s",
            "tokens": "~21K tok",
            "tool": "run_biosteam_simulation",
            "sim_time": "11.8s",
            "msp": "$1.03/kg",
            "gwp": "0.98 kg CO2e/kg",
            "tci": "$75.38M",
        },
        {
            "name": "biosteam-analyst",
            "color": C_BIO3,
            "header_color": "#0369A1",
            "output_color": "#075985",
            "x": c3_x,
            "solvent": "Methylcyclohexane",
            "time": "14.5s",
            "tokens": "~21K tok",
            "tool": "run_biosteam_simulation",
            "sim_time": "12.4s",
            "msp": "$0.996/kg",
            "gwp": "0.985 kg CO2e/kg",
            "tci": "$72.53M",
        },
    ]

    col_h = 0.200
    for a in agents:
        _box(ax, a["x"], par_top, col_w, col_h, a["color"], radius=0.004)
        a_cl = a["x"] + ACCENT_W + 0.008
        a_cr = a["x"] + col_w - 0.006

        cy = par_top - 0.010
        ax.text(a_cl, cy, a["name"],
                fontsize=8, fontweight="bold", color=a["color"],
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(a_cr, cy, f'{a["time"]}  |  {a["tokens"]}',
                fontsize=6.5, color=C_BODY, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.016

        ax.text(a_cl, cy, f'Solvent: {a["solvent"]}',
                fontsize=7.5, color=C_BODY, style="italic",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.014
        ax.text(a_cl, cy, "Energy case: C1 (CHP)",
                fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.018

        _divider(ax, a_cl, a_cr, cy)
        cy -= 0.008

        ax.text(a_cl, cy, "Tool Calls (1 domain):",
                fontsize=7, fontweight="bold", color=C_TOOL_TEXT,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.016

        _pill(ax, a_cl, cy, a["tool"], C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=5.5)
        cy -= PILL_H

        # Simulation time pill
        _pill(ax, a_cl, cy, f'subprocess: {a["sim_time"]}',
               C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=5.5)
        cy -= PILL_H + 0.004

        _divider(ax, a_cl, a_cr, cy)
        cy -= 0.010

        ax.text(a_cl, cy, "Results:",
                fontsize=7.5, fontweight="bold", color=a["output_color"],
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.014
        ax.text(a_cl, cy,
                f'MSP: {a["msp"]}\n'
                f'GWP: {a["gwp"]}\n'
                f'TCI: {a["tci"]}',
                fontsize=7, color=C_BODY, transform=ax.transAxes,
                va="top", zorder=5, linespacing=1.3)

    # Fork arrows from routing to all three boxes
    arrow_kw = dict(arrowstyle="-|>,head_length=0.4,head_width=0.25",
                    color=C_ROUTER, lw=2.0,
                    connectionstyle="arc3,rad=0")
    for a in agents:
        ax.annotate("", xy=(a["x"] + col_w / 2, par_top + 0.002),
                    xytext=(MID, par_top + GAP * 3),
                    arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)

    y = par_top - col_h - GAP

    # ================================================================
    # 4. ORCHESTRATOR SYNTHESIS
    # ================================================================
    box_h = 0.090
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    ax.text(cl, y - 0.010, "Orchestrator Synthesis",
            fontsize=10, fontweight="bold", color="#334155",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.010, "7.4s  |  verifier: PASS",
            fontsize=8, color=C_VERIFIER, ha="right", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    # Mini comparison table inside synthesis box
    tab_top = y - 0.028
    tab_left = cl
    col_positions = [tab_left, tab_left + 0.22, tab_left + 0.42, tab_left + 0.62]
    headers = ["Solvent", "MSP ($/kg)", "GWP (CO2e/kg)", "TCI ($M)"]
    rows = [
        ("MCH",      "$0.996",  "0.985",  "$72.53"),
        ("Heptane",  "$1.03",   "0.98",   "$75.38"),
        ("Toluene",  "$1.14",   "1.05",   "$83.34"),
    ]
    rank_colors = ["#15803D", C_BODY, "#B91C1C"]  # green, neutral, red

    # Header row
    for cx, hdr in zip(col_positions, headers):
        ax.text(cx, tab_top, hdr, fontsize=7, fontweight="bold",
                color="#475569", transform=ax.transAxes, va="top", zorder=5)
    _divider(ax, tab_left, cr - 0.02, tab_top - 0.010, color="#94A3B8", lw=0.8)

    # Data rows
    for i, (solvent, msp, gwp, tci) in enumerate(rows):
        ry = tab_top - 0.016 - i * 0.014
        vals = [solvent, msp, gwp, tci]
        for cx, val in zip(col_positions, vals):
            ax.text(cx, ry, val, fontsize=7.5, color=rank_colors[i],
                    fontweight="bold" if i == 0 else "normal",
                    fontfamily="monospace" if cx != tab_left else "sans-serif",
                    transform=ax.transAxes, va="top", zorder=5)

    # Winner callout
    ax.text(cl, tab_top - 0.060, "Winner: Methylcyclohexane  --  lowest MSP and TCI",
            fontsize=7.5, color="#15803D", fontweight="bold", style="italic",
            transform=ax.transAxes, va="top", zorder=5)

    y -= box_h + GAP * 2

    # ================================================================
    # 5. EXECUTION TIMELINE (Gantt chart)
    # ================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.020

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.020
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
        if label:
            ax.text(x0 + w / 2, y_bar + bar_h / 2, label,
                    ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)

    # Row labels
    row_labels = ["Toluene", "Heptane", "MCH"]
    row_colors = [C_BIO1, C_BIO2, C_BIO3]

    # Timing: route ~11s, then 3 parallel sims ~14s each, synthesis ~7s
    route_dur = 11.5
    sim_durs = [15.1, 13.9, 14.5]
    synth_start = route_dur + max(sim_durs) + 1
    synth_dur = 7.4

    rows_y = []
    for i in range(3):
        by = y - (i + 1) * (bar_h + 0.004)
        rows_y.append(by)
        ax.add_patch(FancyBboxPatch((bar_left, by), bar_w, bar_h,
                                    transform=ax.transAxes, zorder=2, **bg_kw))
        # Route segment (shared)
        if i == 0:
            _bar_seg(ax, by, 0, route_dur, C_USER, "route + classify")
        else:
            _bar_seg(ax, by, 0, route_dur, C_USER, "")

        # Parallel sim segment
        _bar_seg(ax, by, route_dur, sim_durs[i], row_colors[i],
                 f'{row_labels[i]} ({sim_durs[i]:.0f}s)')

        # Row label on left
        ax.text(bar_left - 0.005, by + bar_h / 2, row_labels[i],
                ha="right", va="center", fontsize=7, color=row_colors[i],
                fontweight="bold", transform=ax.transAxes)

    # Synthesis block spanning all 3 rows
    sx = bar_left + (synth_start / total_s) * bar_w
    sw = (synth_dur / total_s) * bar_w
    sh = rows_y[0] + bar_h - rows_y[2]
    ax.add_patch(FancyBboxPatch(
        (sx, rows_y[2]), sw, sh,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=C_SYNTH, edgecolor="none", alpha=0.85,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(sx + sw / 2, rows_y[2] + sh / 2, "synthesis\n+ verify",
            ha="center", va="center", fontsize=7, color="white",
            fontweight="bold", transform=ax.transAxes, zorder=4,
            linespacing=1.2)

    # Time axis labels
    ty = rows_y[2] - 0.012
    for t in [0, 10, 20, 30, 40, 51.5]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t:.0f}s" if t == int(t) else f"{t}s",
                ha="center", fontsize=7.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.018
    items = [(C_USER, "Orchestrator"), (C_BIO1, "Toluene"),
             (C_BIO2, "Heptane"), (C_BIO3, "MCH"), (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.01
    sp = (bar_w - 0.02) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.016], [ly, ly], color=color, lw=5,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.022, ly, label, fontsize=7, color=C_BODY,
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
    pcl = ACCENT_W + 0.008

    # --- Panel 1: Trace Metadata ---
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.010, "Trace Metadata",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "51.5s"),
        ("Total Tokens",   "65,154"),
        ("Subagent Calls", "3 (parallel)"),
        ("LLM Calls",      "~10 total"),
        ("Tool Calls",     "3 domain"),
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

    # --- Panel 2: Results Summary ---
    _box(ax, p2x, y, pw, panel_h, C_ROUTER, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.010, "Results Ranking",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    ranking = (
        "Best MSP:\n"
        "  1. MCH     $0.996/kg\n"
        "  2. Heptane  $1.03/kg\n"
        "  3. Toluene  $1.14/kg\n"
        "All C1 energy case,\n"
        "LCA CFs fully applied"
    )
    ax.text(p2x + pcl, y - 0.026, ranking,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    # --- Panel 3: Execution Pattern ---
    _box(ax, p3x, y, pw, panel_h, C_VERIFIER, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.010, "Execution Pattern",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    pattern = (
        "Parallel delegation:\n"
        "1. LLM classifies query\n"
        "2. 3 task() calls at once\n"
        "3. Each subagent: 1 tool\n"
        "4. BioSTEAM subprocesses\n"
        "5. Orchestrator synthesizes"
    )
    ax.text(p3x + pcl, y - 0.026, pattern,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    y -= panel_h + 0.008

    # ── Footer ──
    ax.text(MID, y,
            "DISSOLVE v8  |  Gemini 2.5 Pro + Flash  |  "
            "Trace 019c58bb  |  "
            "3-Solvent Parallel BioSTEAM Comparison",
            ha="center", fontsize=7.5, color=C_BODY,
            transform=ax.transAxes)

    # ── Save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace10-v8-3solvent-parallel.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
