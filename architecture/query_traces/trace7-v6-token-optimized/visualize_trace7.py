"""
Visualize DISSOLVE agent Trace 7: v6 token-optimized run.
Single subagent invocation with plan_multiple_separation_schemes tool.
40x token reduction vs Trace 6 (2M -> ~50K), 12.7x faster (634s -> 50s).

Trace ID: (local run — no LangSmith trace available)
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
C_WARN       = "#F59E0B"   # Amber   -- warnings / key findings
C_DELTA      = "#0EA5E9"   # Sky     -- comparison / delta

C_S1         = "#F97316"   # Orange  -- scheme 1 (selectivity)
C_S2         = "#8B5CF6"   # Violet  -- scheme 2 (safety)
C_S3         = "#10B981"   # Emerald -- scheme 3 (energy)

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
    ax.text(MID, y, "Trace 7: v6 Token-Optimized  |  3-Scheme Multi-Criteria",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    # Subtitle with key metrics
    ax.text(MID, y,
            "50.0s  |  4 messages  |  1 tool call  |  1 subagent invocation",
            ha="center", fontsize=7, color=C_BODY,
            transform=ax.transAxes)
    y -= 0.015

    # ==================================================================
    # 2. USER QUERY
    # ==================================================================
    query = (
        "Find the optimal separation sequence for a mixed polymer waste stream "
        "containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, and PET. "
        "Use selective dissolution at atmospheric pressure. Propose THREE different "
        "sets of solvents and conditions for this 9-polymer dissolution scheme."
    )
    box_h = 0.030
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
    box_h = 0.023
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    cy = y - 0.005
    ax.text(cl, cy, "Single-Agent Routing",
            fontsize=8, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    ax.text(cl, cy,
            'Keywords: "separat"+"dissolut"+"scheme" -> separation-engineer  |  '
            'Multi-scheme hint: delegate ONCE (multi-scheme tool available)',
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 4. SEPARATION-ENGINEER (single invocation)
    # ==================================================================
    box_h = 0.052
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP_ENG)
    cy = y - 0.005
    ax.text(cl, cy, "1. separation-engineer  (1 invocation, 1 tool call)",
            fontsize=8, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "~18s  |  1 tool call  |  synthesis injected",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    # Tool call details
    ax.text(cl, cy, "Tool: plan_multiple_separation_schemes",
            fontsize=6.5, fontweight="bold", color=C_SEP_ENG,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    ax.text(cl, cy,
            "Args: polymers=\"PS,PVC,LDPE,HDPE,PP,EVOH,Nylon6,Nylon66,PET\", "
            "temperature=120.0, min_selectivity=5.0",
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5)
    cy -= 0.008

    _divider(ax, cl, cr, cy)
    cy -= 0.008

    # Guardrails info
    pills_data = [
        ("max_tool_calls=5", C_GUARD_BG, C_GUARD_TEXT),
        ("token_budget=100K", C_GUARD_BG, C_GUARD_TEXT),
        ("synthesis_tool_detected", "#D1FAE5", "#065F46"),
        ("truncate_after=800", C_GUARD_BG, C_GUARD_TEXT),
    ]
    cx = cl
    for label, bg, fg in pills_data:
        tw = _pill(ax, cx, cy, label, fg, bg, fg, fs=5)
        cx += tw + 0.005

    y -= box_h + GAP

    # ==================================================================
    # 5. THREE SCHEME TABLES
    # ==================================================================
    ax.text(MID, y - 0.002,
            "Three Schemes (from plan_multiple_separation_schemes, 2493 chars, 0.23s)",
            ha="center", fontsize=8, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.014

    col_gap = 0.008
    col_w = (WIDTH - 2 * col_gap) / 3
    c1x = LEFT
    c2x = LEFT + col_w + col_gap
    c3x = LEFT + 2 * (col_w + col_gap)

    scheme_h = 0.150
    schemes = [
        ("Scheme 1: Max Selectivity", C_S1, c1x, [
            ("PS",       "Dichloromethane", "35",  "19.1"),
            ("PVC",      "Acetone",         "50",  "21.3"),
            ("EVOH",     "DMSO",            "120", "27.2"),
            ("LDPE",     "Ethanol",         "70",  "9.4"),
            ("PP",       "THF",             "60",  "29.8"),
            ("HDPE",     "Cyclohexane",     "75",  "27.1"),
            ("PET",      "DMF",             "120", "12.3"),
            ("Nylon 6",  "Chloroform",      "55",  "6.2"),
            ("Nylon 66", "(Residue)",        "-",   "-"),
        ]),
        ("Scheme 2: Safest Process", C_S2, c2x, [
            ("EVOH",     "Glycol",          "120", "22.4"),
            ("LDPE",     "Cyclohexanol",    "120", "5.7"),
            ("PS",       "Ethyl Acetate",   "70",  "18.9"),
            ("PVC",      "Methyl Acetate",  "50",  "17.8"),
            ("PP",       "tert-Butanol",    "75",  "21.2"),
            ("HDPE",     "Toluene",         "100", "20.9"),
            ("PET",      "DMF",             "120", "8.8"),
            ("Nylon 6",  "Chloroform",      "55",  "5.7"),
            ("Nylon 66", "(Residue)",        "-",   "-"),
        ]),
        ("Scheme 3: Lowest Energy", C_S3, c3x, [
            ("PS",       "Isopropylamine",  "25",  "14.4"),
            ("PVC",      "Dichloromethane",  "35",  "11.2"),
            ("EVOH",     "Methanol",        "60",  "13.8"),
            ("PP",       "THF",             "60",  "21.1"),
            ("LDPE",     "Acetone",         "50",  "13.7"),
            ("HDPE",     "Hexane",          "60",  "18.8"),
            ("Nylon 6",  "Chloroform",      "55",  "5.4"),
            ("PET",      "MethylAcetate",   "50",  "6.9"),
            ("Nylon 66", "(Residue)",        "-",   "-"),
        ]),
    ]

    for title, color, sx, steps in schemes:
        _box(ax, sx, y, col_w, scheme_h, color, radius=0.004)
        scl = sx + ACCENT_W + 0.006
        scr = sx + col_w - 0.005

        cy = y - 0.005
        ax.text(scl, cy, title,
                fontsize=5.8, fontweight="bold", color=color,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.010

        # Headers
        ax.text(scl, cy, "Polymer", fontsize=4.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scl + 0.045, cy, "Solvent", fontsize=4.5, fontweight="bold",
                color=C_TITLE, transform=ax.transAxes, va="top", zorder=5)
        ax.text(scr - 0.025, cy, "T(C)", fontsize=4.5, fontweight="bold",
                color=C_TITLE, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(scr, cy, "Sel%", fontsize=4.5, fontweight="bold",
                color=C_TITLE, ha="right",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.007

        for polymer, solvent, temp, sel in steps:
            ax.text(scl, cy, polymer, fontsize=4.5, fontweight="bold",
                    color=color, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scl + 0.045, cy, solvent, fontsize=4.5,
                    color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
            ax.text(scr - 0.025, cy, temp, fontsize=4.5,
                    color=C_BODY, ha="right",
                    transform=ax.transAxes, va="top", zorder=5)
            ax.text(scr, cy, sel, fontsize=4.5,
                    color=C_BODY, ha="right",
                    transform=ax.transAxes, va="top", zorder=5)
            cy -= 0.013

    y -= scheme_h + GAP

    # ==================================================================
    # 6. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.028
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    cy = y - 0.005
    ax.text(cl, cy, "Orchestrator Synthesis",
            fontsize=8, fontweight="bold", color=C_SYNTH,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, cy, "4 messages total  |  2 AI messages  |  1 tool message",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    ax.text(cl, cy,
            "Separation-engineer returned complete 3-scheme analysis. Orchestrator "
            "formatted into final answer with technical summary, Nylon challenge notes, "
            "and atmospheric pressure compliance verification.",
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ==================================================================
    # 7. KEY FINDING: TOKEN REDUCTION
    # ==================================================================
    box_h = 0.048
    _box(ax, LEFT, y, WIDTH, box_h, C_WARN)
    cy = y - 0.005
    ax.text(cl, cy, "Key Achievement: 40x Token Reduction",
            fontsize=8, fontweight="bold", color="#92400E",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011

    changes = [
        ("1. Multi-scheme tool:", "3 schemes in 1 tool call (2493 chars, 0.23s) vs 3 separate subagent invocations"),
        ("2. Orchestrator guardrails:", "500K token budget + 30 tool-call cap prevents runaway loops"),
        ("3. Subagent tightening:", "max_tool_calls 8->5, token_budget 100K, truncate_after 800 chars"),
        ("4. Aggressive truncation:", "kicks in after iter 1 (was 3), keeps last 4 messages (was 6)"),
    ]
    for label, desc in changes:
        ax.text(cl, cy, label, fontsize=5.5, fontweight="bold",
                color="#92400E", transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.145, cy, desc, fontsize=5.5,
                color=C_BODY, transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ==================================================================
    # 8. EXECUTION TIMELINE
    # ==================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.012

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.016
    total_s = 50.0

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

    # Timeline segments (approximate from HTTP request timestamps)
    # 23:51:22 -> 23:51:25: init (3s)
    # 23:51:25 -> 23:51:30: routing + delegation (5s)
    # 23:51:30 -> 23:51:50: sep-eng tool execution (20s)
    # 23:51:50 -> 23:52:12: LLM synthesis (22s)
    _bar_seg(0, 3, C_USER, "init")
    _bar_seg(3, 5, C_ROUTER, "route")
    _bar_seg(8, 20, C_SEP_ENG, "sep-eng (20s)")
    _bar_seg(28, 22, C_SYNTH, "synthesis (22s)")

    # Time labels
    ty = bar_y - 0.008
    for t in [0, 10, 20, 30, 40, 50]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=5.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.012
    items = [(C_USER, "Init"), (C_ROUTER, "Routing"),
             (C_SEP_ENG, "sep-eng"), (C_SYNTH, "Synthesis")]
    lx = bar_left + 0.02
    sp = (bar_w - 0.04) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=5.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.018

    # ==================================================================
    # 9. BOTTOM PANELS
    # ==================================================================
    panel_h = 0.072
    panel_gap = 0.008
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.006

    # Panel 1: Trace Metadata
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.005, "Trace 7 Metadata",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",         "50.0 s"),
        ("Messages",         "4"),
        ("Tool Calls",       "1"),
        ("Subagent Calls",   "1"),
        ("HTTP Requests",    "7"),
        ("Peak RSS",         "898 MB"),
        ("Recursion Limit",  "500"),
    ]
    my = y - 0.016
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.005, my, value, fontsize=5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.008

    # Panel 2: Trace 6 vs 7 Comparison
    _box(ax, p2x, y, pw, panel_h, C_DELTA, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.005, "Trace 6 vs 7",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    comparisons = (
        "Trace 6 (pre-optimization):\n"
        "  275s / ~2M tok / 30 msgs\n"
        "  3x sep-eng invocations\n"
        "  496K sep-eng + orch queries\n"
        "\n"
        "Trace 7 (v6 optimized):\n"
        "  50s / ~50K tok / 4 msgs\n"
        "  1x sep-eng, 1 tool call\n"
        "  5.5x faster, ~40x fewer tok"
    )
    ax.text(p2x + pcl, y - 0.016, comparisons,
            fontsize=4.8, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    # Panel 3: What Changed
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.005, "What Changed (v6)",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    changes_text = (
        "1. Multi-scheme tool: 3\n"
        "   schemes in 1 call (0.23s)\n"
        "2. Orchestrator guardrails:\n"
        "   500K tok + 30 tool cap\n"
        "3. Subagent: 5 tools, 100K\n"
        "4. Truncation: iter 1, keep 4\n"
        "5. Trimmed system prompts\n"
        "6. Interpolation model (no SQL)"
    )
    ax.text(p3x + pcl, y - 0.016, changes_text,
            fontsize=4.8, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    y -= panel_h + 0.005

    # -- Footer --
    ax.text(MID, y,
            "DISSOLVE v6  |  Gemini 3 Flash Preview  |  "
            "Token-Optimized  |  Interpolation Model  |  "
            "3-Scheme Multi-Criteria",
            ha="center", fontsize=6, color=C_BODY,
            transform=ax.transAxes)

    # -- Clip whitespace by adjusting ylim to actual content extent --
    ax.set_ylim(y - 0.015, 1.0)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace7-v6-token-optimized.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
