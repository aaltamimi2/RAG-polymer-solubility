"""
Visualize DISSOLVE agent sequential pipeline trace:
separation-engineer -> 3 x biosteam-analyst for LDPE/PET/EVOH multi-polymer
dissolution sequence ranking.

Trace ID: 019c5a85-4241-7b60-8a16-500bd572fc89
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

C_USER       = "#6366F1"   # Indigo - orchestrator / user
C_ROUTER     = "#22C55E"   # Green - routing
C_SEP        = "#F97316"   # Orange - separation-engineer
C_BIO1       = "#2D8B72"   # Forest green - biosteam-analyst #1
C_BIO2       = "#0EA5E9"   # Sky - biosteam-analyst #2
C_BIO3       = "#8B5CF6"   # Violet - biosteam-analyst #3
C_SYNTH      = "#64748B"   # Slate - synthesis
C_HANDOFF    = "#EC4899"   # Pink - data handoff
C_VERIFIER   = "#10B981"   # Emerald - verifier

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"


# -- Layout constants ----------------------------------------------------------
FIG_W  = 7.5
FIG_H  = 22.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
LINE_H   = 0.010
LINE_S   = 0.008
PILL_H   = 0.013
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


def _divider(ax, x1, x2, y, color="#D1D5DB", lw=0.6):
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            transform=ax.transAxes, zorder=3)


def _pill(ax, x, y_center, text, fg, bg, border, fs=6, mono=True):
    tw = len(text) * (0.0068 if mono else 0.0058) + 0.012
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


def _wrap(text, width=100):
    lines = text.split("\n")
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width) if len(line) > width else [line])
    return "\n".join(out)


def _arrow_down(ax, x, y_from, y_to, color, lw=1.5):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.3,head_width=0.2",
                    color=color, lw=lw),
                transform=ax.transAxes, zorder=6)


# -- Main figure ---------------------------------------------------------------

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    y = 0.990
    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD

    # ==================================================================
    # HEADER
    # ==================================================================
    ax.text(MID, y, "DISSOLVE Agent -- LDPE/PET/EVOH Multi-Polymer Pipeline",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.010
    ax.text(MID, y,
            "264.5s  |  200,702 tokens  |  4 subagent calls  |  "
            "Sequential: separation-engineer -> biosteam-analyst (x3)",
            ha="center", fontsize=6.5, color=C_BODY,
            transform=ax.transAxes)
    y -= 0.015

    # ==================================================================
    # 1. USER QUERY
    # ==================================================================
    query = (
        "Propose all possible LDPE/PET/EVOH dissolution sequences and "
        "test each one via BioSTEAM TEA/LCA simulation. "
        "Rank the sequences by blended MSP."
    )
    box_h = 0.034
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    ax.text(cl, y - 0.006, "User Query",
            fontsize=8, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl, y - 0.016, _wrap(query, 115),
            fontsize=6, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # Arrow
    _arrow_down(ax, MID, y + 0.003, y - 0.008, C_ROUTER)
    y -= 0.012

    # ==================================================================
    # 2. ROUTING
    # ==================================================================
    box_h = 0.032
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    ax.text(cl, y - 0.006, "Sequential Routing",
            fontsize=8, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    route_text = (
        'LLM classifier: "dissolution sequences" + "BioSTEAM" -> '
        'separation-engineer -> biosteam-analyst (SEQUENTIAL_PAIR)\n'
        'Pipeline protocol: extract top_k_sequences -> build polymers_json -> '
        'run multi-polymer BioSTEAM for each'
    )
    ax.text(cl, y - 0.016, route_text,
            fontsize=5.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # Arrow
    _arrow_down(ax, MID, y + 0.003, y - 0.008, C_SEP)
    y -= 0.012

    # ==================================================================
    # 3. SEPARATION ENGINEER
    # ==================================================================
    box_h = 0.068
    _box(ax, LEFT, y, WIDTH, box_h, C_SEP)
    ax.text(cl, y - 0.006, "1. separation-engineer",
            fontsize=8, fontweight="bold", color=C_SEP,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.006, "21.8s  |  1 tool call",
            fontsize=6, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.018
    _pill(ax, cl, cy, "plan_sequential_separation", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=5.5)
    cy -= PILL_H + 0.004

    _divider(ax, cl, cr, cy)
    cy -= 0.008

    ax.text(cl, cy, "Output:",
            fontsize=6, fontweight="bold", color="#9A3412",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    sep_output = (
        "2 viable sequences at 120C:\n"
        "  Seq 1: LDPE (Dodecane) -> EVOH (Ethylene Glycol) -> PET (Toluene)\n"
        "  Seq 2: EVOH (Ethylene Glycol) -> LDPE (Dodecane) -> PET (Toluene)"
    )
    ax.text(cl, cy, sep_output,
            fontsize=5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3,
            fontfamily="monospace")
    y -= box_h + GAP

    # Arrow with handoff label
    _arrow_down(ax, MID, y + 0.003, y - 0.012, C_HANDOFF)
    ax.text(MID + 0.015, y - 0.003,
            "solvent_mapping -> polymers_json",
            fontsize=5.5, color=C_HANDOFF, fontweight="bold", style="italic",
            transform=ax.transAxes, va="center", zorder=7)
    y -= 0.018

    # ==================================================================
    # 4. BIOSTEAM-ANALYST x3 (sequential)
    # ==================================================================

    bio_agents = [
        {
            "label": "2. biosteam-analyst  (sequence validation)",
            "color": C_BIO1,
            "time": "6.4s",
            "tools": ["get_biosteam_solvents"],
            "desc": "Validated LDPE/PET/EVOH solvent compatibility with BioSTEAM model",
            "result": "Confirmed: LDPE (Dodecane), EVOH (Ethylene Glycol), PET (Toluene) all supported",
        },
        {
            "label": "3. biosteam-analyst  (Seq 1: LDPE -> EVOH -> PET)",
            "color": C_BIO2,
            "time": "63.6s",
            "tools": ["run_biosteam_multi_polymer", "run_biosteam_multi_polymer"],
            "desc": "Multi-polymer TEA/LCA for sequence LDPE -> EVOH -> PET",
            "result": (
                "3/3 completed  |  Blended MSP: $1.0124/kg  |  "
                "GWP: 1.2755 kg CO2e/kg  |  TCI: $230.19M"
            ),
        },
        {
            "label": "4. biosteam-analyst  (Seq 2: EVOH -> LDPE -> PET)",
            "color": C_BIO3,
            "time": "100.4s",
            "tools": ["run_biosteam_multi_polymer", "run_biosteam_multi_polymer"],
            "desc": "Multi-polymer TEA/LCA for sequence EVOH -> LDPE -> PET",
            "result": (
                "3/3 completed  |  Blended MSP: $1.0124/kg  |  "
                "GWP: 1.2755 kg CO2e/kg  |  TCI: $230.19M"
            ),
        },
    ]

    for i, ba in enumerate(bio_agents):
        n_tools = len(ba["tools"])
        box_h = 0.050
        _box(ax, LEFT, y, WIDTH, box_h, ba["color"])
        ax.text(cl, y - 0.006, ba["label"],
                fontsize=8, fontweight="bold", color=ba["color"],
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cr, y - 0.006,
                f'{ba["time"]}  |  {n_tools} tool call{"s" if n_tools != 1 else ""}',
                fontsize=6, color=C_BODY, ha="right",
                transform=ax.transAxes, va="top", zorder=5)

        cy = y - 0.018
        px = cl
        for tool_name in ba["tools"]:
            tw = _pill(ax, px, cy, tool_name, C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=5)
            px += tw + 0.006

        cy -= PILL_H + 0.004
        _divider(ax, cl, cr, cy)
        cy -= 0.007

        ax.text(cl, cy, ba["result"],
                fontsize=5, color=C_BODY, transform=ax.transAxes,
                va="top", zorder=5, linespacing=1.3)

        y -= box_h + GAP

        # Arrow between biosteam agents
        if i < len(bio_agents) - 1:
            _arrow_down(ax, MID, y + 0.003, y - 0.006, ba["color"], lw=1.0)
            y -= 0.010

    # Arrow to synthesis
    _arrow_down(ax, MID, y + 0.003, y - 0.008, C_SYNTH)
    y -= 0.013

    # ==================================================================
    # 5. ORCHESTRATOR SYNTHESIS
    # ==================================================================
    box_h = 0.080
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    ax.text(cl, y - 0.006, "Orchestrator Synthesis",
            fontsize=8, fontweight="bold", color="#334155",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.006, "verifier: PASS",
            fontsize=6, color=C_VERIFIER, ha="right", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    # Per-polymer breakdown table
    tab_top = y - 0.020
    tab_left = cl
    col_pos = [tab_left, tab_left + 0.15, tab_left + 0.35,
               tab_left + 0.52, tab_left + 0.66, tab_left + 0.82]
    headers = ["Polymer", "Solvent", "MSP ($/kg)", "TCI ($M)", "GWP", "Wt"]
    rows = [
        ("LDPE",  "Dodecane",        "$1.0628", "$77.03", "1.2967", "$1.10"),
        ("EVOH",  "Ethylene Glycol", "$0.9714", "$69.83", "1.3239", "$4.50"),
        ("PET",   "Toluene",         "$1.1351", "$83.34", "1.0460", "$1.05"),
    ]
    row_colors = ["#2D8B72", "#D97706", "#7C3AED"]

    for cx, hdr in zip(col_pos, headers):
        ax.text(cx, tab_top, hdr, fontsize=5.5, fontweight="bold",
                color="#475569", transform=ax.transAxes, va="top", zorder=5)
    _divider(ax, tab_left, cr - 0.01, tab_top - 0.007, color="#94A3B8", lw=0.8)

    for i, (polymer, solvent, msp, tci, gwp, wt) in enumerate(rows):
        ry = tab_top - 0.012 - i * 0.009
        vals = [polymer, solvent, msp, tci, gwp, wt]
        for cx, val in zip(col_pos, vals):
            ax.text(cx, ry, val, fontsize=5.5, color=row_colors[i],
                    fontfamily="monospace" if cx != tab_left else "sans-serif",
                    transform=ax.transAxes, va="top", zorder=5)

    # Blended result callout
    ax.text(cl, tab_top - 0.042,
            "Both sequences yield identical results (order-independent process model):\n"
            "Blended MSP: $1.0124/kg  |  Combined TCI: $230.19M  |  "
            "Weighted GWP: 1.2755 kg CO2e/kg",
            fontsize=5.5, color="#15803D", fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.4)

    y -= box_h + GAP * 2

    # ==================================================================
    # 6. EXECUTION TIMELINE (Gantt)
    # ==================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=8, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.010

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.012
    total_s = 264.5

    def _bar_seg(ax, y_bar, t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_s) * bar_w
        w = max((t_dur / total_s) * bar_w, 0.003)
        ax.add_patch(FancyBboxPatch(
            (x0, y_bar), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.002",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if w > 0.05:
            ax.text(x0 + w / 2, y_bar + bar_h / 2, label,
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.002",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)

    # Row labels and timing from trace data
    row_data = [
        ("sep-eng",    C_SEP,  "separation-engineer"),
        ("bio-1",      C_BIO1, "biosteam (validate)"),
        ("bio-2",      C_BIO2, "biosteam (seq 1)"),
        ("bio-3",      C_BIO3, "biosteam (seq 2)"),
    ]

    # Timing: orchestrator routing ~25s, then sequential subagent calls
    # sep: 25-47s (21.8s), bio1: 55-61s (6.4s), bio2: 75-139s (63.6s), bio3: 148-248s (100.4s)
    # synthesis: 248-264.5s
    timings = [
        (25, 21.8),    # separation-engineer
        (55, 6.4),     # biosteam validate
        (75, 63.6),    # biosteam seq 1
        (148, 100.4),  # biosteam seq 2
    ]

    rows_y = []
    for i, ((label, color, desc), (t_start, t_dur)) in enumerate(zip(row_data, timings)):
        by = y - (i + 1) * (bar_h + 0.004)
        rows_y.append(by)
        ax.add_patch(FancyBboxPatch((bar_left, by), bar_w, bar_h,
                                    transform=ax.transAxes, zorder=2, **bg_kw))

        # Orchestrator segment (routing/thinking before this subagent)
        if i == 0:
            _bar_seg(ax, by, 0, 25, C_USER, "route")

        # Subagent segment
        _bar_seg(ax, by, t_start, t_dur, color, desc)

        # Row label
        ax.text(bar_left - 0.005, by + bar_h / 2, label,
                ha="right", va="center", fontsize=5.5, color=color,
                fontweight="bold", transform=ax.transAxes)

    # Synthesis segment on last row
    _bar_seg(ax, rows_y[-1], 248, 16.5, C_SYNTH, "synth")

    # Time axis
    ty = rows_y[-1] - 0.008
    for t in [0, 30, 60, 90, 120, 150, 180, 210, 240, 264.5]:
        tx = bar_left + (t / total_s) * bar_w
        label = f"{t:.0f}s" if t == int(t) else f"{t}s"
        ax.text(tx, ty, label,
                ha="center", fontsize=5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.012
    items = [
        (C_USER, "Orchestrator"), (C_SEP, "separation-engineer"),
        (C_BIO1, "biosteam (validate)"), (C_BIO2, "biosteam (seq 1)"),
        (C_BIO3, "biosteam (seq 2)"), (C_SYNTH, "Synthesis"),
    ]
    lx = bar_left + 0.005
    sp = (bar_w - 0.01) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=4.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.018

    # ==================================================================
    # 7. BOTTOM PANELS (three columns)
    # ==================================================================
    panel_h = 0.075
    panel_gap = 0.008
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.006

    # --- Panel 1: Trace Metadata ---
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.006, "Trace Metadata",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes, zorder=5)
    metrics = [
        ("Run Time",       "264.5s"),
        ("Total Tokens",   "200,702"),
        ("Subagent Calls", "4 (sequential)"),
        ("Tool Calls",     "6 domain"),
        ("BioSTEAM Sims",  "6 (3 poly x 2 seq)"),
        ("Pattern",        "sep -> biosteam x3"),
    ]
    my = y - 0.018
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.005, my, value, fontsize=5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.009

    # --- Panel 2: Pipeline Protocol ---
    _box(ax, p2x, y, pw, panel_h, C_HANDOFF, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.006, "Pipeline Protocol",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes, zorder=5)
    protocol = (
        "1. sep-eng: sequences\n"
        "2. Extract solvent_mapping\n"
        "3. Build polymers_json[]\n"
        "4. biosteam: per sequence\n"
        "5. Rank by blended MSP\n"
        "6. Synthesize comparison"
    )
    ax.text(p2x + pcl, y - 0.018, protocol,
            fontsize=5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.2)

    # --- Panel 3: Key Results ---
    _box(ax, p3x, y, pw, panel_h, C_VERIFIER, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.006, "Key Results",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes, zorder=5)
    results = (
        "Sequences tested: 2\n"
        "Polymers per seq: 3\n"
        "All 6/6 sims: OK\n"
        "Blended MSP: $1.01/kg\n"
        "Combined TCI: $230M\n"
        "Best GWP: 1.28 CO2e/kg"
    )
    ax.text(p3x + pcl, y - 0.018, results,
            fontsize=5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.2)

    y -= panel_h + 0.008

    # -- Footer --
    ax.text(MID, y,
            "DISSOLVE v8  |  Gemini 2.5 Pro + Flash  |  "
            "Trace 019c5a85  |  "
            "LDPE/PET/EVOH Multi-Polymer Sequential Dissolution Pipeline",
            ha="center", fontsize=6, color=C_BODY,
            transform=ax.transAxes)

    # -- Save --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace12-v8-ldpe-pet-evoh-pipeline.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
