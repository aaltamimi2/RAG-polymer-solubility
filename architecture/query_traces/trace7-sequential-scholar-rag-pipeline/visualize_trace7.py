"""
Visualize DISSOLVE agent sequential scholar-researcher -> rag-analyst pipeline trace.
Sequential execution: arXiv search + PDF ingest -> RAG Q&A.
Stacked-card style matching architecture/visualize_chain_trace.py.

Trace 7: First successful full RAG pipeline (search -> ingest -> query).
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
C_SCHOLAR    = "#0891B2"   # Cyan — scholar-researcher
C_RAG        = "#D946EF"   # Fuchsia — rag-analyst
C_SYNTH      = "#0EA5E9"   # Sky blue — synthesis
C_SUCCESS    = "#16A34A"   # Green — success indicators
C_WARN       = "#F59E0B"   # Amber — warnings

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_GUARD_BG   = "#DBEAFE"
C_GUARD_TEXT = "#1E40AF"
C_INGEST_BG  = "#D1FAE5"
C_INGEST_TEXT = "#065F46"


# ── Layout constants ─────────────────────────────────────────────────
FIG_W  = 7.5
FIG_H  = 19.0

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


def _arrow(ax, x_from, y_from, x_to, y_to, color, lw=1.2):
    ax.annotate("", xy=(x_to, y_to),
                xytext=(x_from, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0"),
                transform=ax.transAxes, zorder=6)


# ── Main figure ──────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    y = 0.985
    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD

    # ================================================================
    # 1. USER QUERY
    # ================================================================
    query = (
        "Search arXiv for papers on green solvents for PET recycling. "
        "Save the top 1 paper to a knowledgebase called test-pipeline-agent. "
        "Then query that knowledgebase: what green solvents show promise for PET?"
    )
    box_h = 0.052
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
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
    box_h = 0.050
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    ax.text(cl, y - 0.010, "Sequential Routing",
            fontsize=10, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    route_text = (
        'Keyword matches:  "arxiv" -> scholar-researcher (score 2)   |   '
        '"knowledgebase" -> rag-analyst (score 2)\n'
        'Pair (scholar-researcher, rag-analyst) in SEQUENTIAL_PAIRS  '
        '->  Step 1: scholar, Step 2: rag'
    )
    ax.text(cl, y - 0.024, route_text,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP

    _arrow(ax, MID, y + GAP, MID, y + 0.002, C_ROUTER)

    # ================================================================
    # 3. SCHOLAR-RESEARCHER (Step 1)
    # ================================================================
    box_h = 0.230
    _box(ax, LEFT, y, WIDTH, box_h, C_SCHOLAR)
    ax.text(cl, y - 0.010, "Step 1: scholar-researcher",
            fontsize=10, fontweight="bold", color=C_SCHOLAR,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.010, "~45s",
            fontsize=8, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy = y - 0.026
    ax.text(cl, cy,
            '"Search arXiv for green solvents for PET recycling.\n'
            ' Save top 1 paper to knowledgebase test-pipeline-agent."',
            fontsize=7.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.4)
    cy -= 0.040

    _divider(ax, cl, cr, cy)
    cy -= 0.013

    ax.text(cl, cy, "Tool Calls (1 domain + think):",
            fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018
    tool_labels = ["search_google_scholar x1"]
    _pills_row(ax, cl, cy, tool_labels, cr - cl,
               C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.006

    # Synthesis injection pill
    _pill(ax, cl, cy, "SYNTHESIS INJECTION fired", C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=6.5)
    cy -= PILL_H + 0.006

    # Ingestion result
    _pill(ax, cl, cy, "1 paper ingested: 46 chunks, 30 indexed", C_INGEST_TEXT, C_INGEST_BG, C_INGEST_TEXT, fs=6.5)
    cy -= PILL_H + 0.004

    _divider(ax, cl, cr, cy)
    cy -= 0.013

    ax.text(cl, cy, "Output:",
            fontsize=8, fontweight="bold", color="#0E7490",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.015
    article_title = (
        '"Mimicking a Solvent Interface at the Substrate Access Channel\n'
        ' of Nylonase" — saved to KB test-pipeline-agent'
    )
    ax.text(cl, cy, article_title,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)

    y -= box_h + GAP

    _arrow(ax, MID, y + GAP, MID, y + 0.002, C_SCHOLAR)

    # ================================================================
    # 4. RAG-ANALYST (Step 2)
    # ================================================================
    box_h = 0.225
    _box(ax, LEFT, y, WIDTH, box_h, C_RAG)
    ax.text(cl, y - 0.010, "Step 2: rag-analyst",
            fontsize=10, fontweight="bold", color=C_RAG,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.010, "~230s",
            fontsize=8, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy = y - 0.026
    ax.text(cl, cy,
            '"Query knowledgebase test-pipeline-agent:\n'
            ' What green solvents show promise for PET?"',
            fontsize=7.5, color=C_BODY, style="italic",
            transform=ax.transAxes, va="top", zorder=5, linespacing=1.4)
    cy -= 0.040

    _divider(ax, cl, cr, cy)
    cy -= 0.013

    ax.text(cl, cy, "Tool Calls (5 domain + think):",
            fontsize=8, fontweight="bold", color=C_TOOL_TEXT,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.018
    tool_labels = ["ask_literature x5"]
    _pills_row(ax, cl, cy, tool_labels, cr - cl,
               C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.006

    _pill(ax, cl, cy, "SYNTHESIS INJECTION fired after 1st call", C_GUARD_TEXT, C_GUARD_BG, C_GUARD_TEXT, fs=6.5)
    cy -= PILL_H + 0.006

    _pill(ax, cl, cy, "KB switched: test-pipeline-agent (5x)", C_INGEST_TEXT, C_INGEST_BG, C_INGEST_TEXT, fs=6.5)
    cy -= PILL_H + 0.004

    _divider(ax, cl, cr, cy)
    cy -= 0.013

    ax.text(cl, cy, "Output:",
            fontsize=8, fontweight="bold", color="#A21CAF",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.015
    ax.text(cl, cy,
            "Enzymatic depolymerization, biomimetic solvent\n"
            "interfaces, 86.8% depolymerization efficiency",
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)

    y -= box_h + GAP

    _arrow(ax, MID, y + GAP, MID, y + 0.002, C_RAG)

    # ================================================================
    # 5. ORCHESTRATOR SYNTHESIS
    # ================================================================
    box_h = 0.082
    _box(ax, LEFT, y, WIDTH, box_h, C_SYNTH)
    ax.text(cl, y - 0.010, "Orchestrator Synthesis",
            fontsize=10, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.010, "~6s",
            fontsize=8, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    synth = (
        "Combined scholar + RAG results into final answer:\n"
        "1. DESs, Ionic Liquids, Supercritical Fluids (from scholar search)\n"
        "2. Enzymatic depolymerization + biomimetic interfaces (from RAG)\n"
        "3. Hybrid 2-stage: solvent swelling + enzymatic -> 86.8% depolymerization"
    )
    ax.text(cl, y - 0.026, synth,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    y -= box_h + GAP * 2

    # ================================================================
    # 6. EXECUTION TIMELINE
    # ================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=10, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.020

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.022
    total_s = 285.0

    def _bar_seg(ax, y_bar, t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_s) * bar_w
        w = max((t_dur / total_s) * bar_w, 0.003)
        ax.add_patch(FancyBboxPatch(
            (x0, y_bar), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if w > 0.06:
            ax.text(x0 + w / 2, y_bar + bar_h / 2, label,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)

    by = y - bar_h
    ax.add_patch(FancyBboxPatch((bar_left, by), bar_w, bar_h,
                                transform=ax.transAxes, zorder=2, **bg_kw))
    _bar_seg(ax, by, 0, 2, C_USER, "")
    _bar_seg(ax, by, 2, 43, C_SCHOLAR, "scholar (45s)")
    _bar_seg(ax, by, 45, 230, C_RAG, "rag-analyst (230s)")
    _bar_seg(ax, by, 275, 6, C_SYNTH, "")

    # Time labels
    ty = by - 0.012
    for t in [0, 45, 100, 150, 200, 281]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=7.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.018
    items = [(C_USER, "Orchestrator"), (C_SCHOLAR, "scholar-researcher"),
             (C_RAG, "rag-analyst"), (C_SYNTH, "Synthesis")]
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
    # 7. BOTTOM PANELS (three columns)
    # ================================================================
    panel_h = 0.115
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
        ("Run Time",       "281.4s"),
        ("Messages",       "8"),
        ("Orchestr. Tools","4 (2 task, 2 todo)"),
        ("Scholar Tools",  "1 domain + think"),
        ("RAG Tools",      "5 domain + think"),
        ("Pattern",        "sequential"),
    ]
    my = y - 0.030
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=7, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.006, my, value, fontsize=7,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.014

    # --- Panel 2: Pipeline Steps ---
    _box(ax, p2x, y, pw, panel_h, C_SUCCESS, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.010, "Pipeline Steps",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    pattern = (
        "1. Route: sequential pair\n"
        "2. Scholar: search + ingest\n"
        "3. RAG: ask_literature x5\n"
        "4. Orchestrator: synthesize\n"
        "Full chain: query->PDF->RAG"
    )
    ax.text(p2x + pcl, y - 0.026, pattern,
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    # --- Panel 3: Key Fixes Applied ---
    _box(ax, p3x, y, pw, panel_h, C_WARN, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.010, "Key Fixes Applied",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    fixes = (
        "[+] synthesis_tools YAML\n"
        "[+] rag_qa (2 vs 19 tools)\n"
        "[+] ask_literature +KB param\n"
        "[+] Prescriptive routing\n"
        "[!] RAG called 5x (4 extra)"
    )
    ax.text(p3x + pcl, y - 0.026, fixes,
            fontsize=7.5, color="#92400E", transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.25)

    y -= panel_h + 0.008

    # ── Footer ──
    ax.text(MID, y,
            "DISSOLVE  |  Gemini 3 Flash Preview  |  "
            "Trace 7  |  "
            "Sequential Scholar -> RAG Pipeline",
            ha="center", fontsize=7.5, color=C_BODY,
            transform=ax.transAxes)

    # ── Save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "trace7-sequential-scholar-rag-pipeline.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
