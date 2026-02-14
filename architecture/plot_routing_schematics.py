"""Generate publication-quality routing architecture schematics.

Style matches generate_diagrams.py (agent_architecture): white boxes with
coloured borders, coloured header bands, #FFFFFF background, 600 DPI,
design-grid scaling with Liberation Sans.
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans", "Arial", "Helvetica", "DejaVu Sans",
]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette (same as generate_diagrams.py) ──────────────
PAL = {
    "orch":     "#3b5998",
    "text":     "#1C2A3A",
    "text_lt":  "#5A6A7A",
    "text_wh":  "#FFFFFF",
    "arrow":    "#8899AA",
    # Layer identity colours
    "flash":    "#7D5BA6",   # purple  — classifier / verifier
    "orch_lyr": "#3b5998",   # blue    — orchestrator
    "verify":   "#B83230",   # red     — verifier
}
C_BG = "#FFFFFF"


# ═════════════════════════════════════════════════════════════════
# Figure 1: Three-Layer Middleware Architecture
# ═════════════════════════════════════════════════════════════════

def plot_three_layer():
    """Three-layer middleware in agent_architecture visual style."""
    _DESIGN_W = 15.0
    _DESIGN_H = 14.0
    _TARGET_W = 7.0
    S = _TARGET_W / _DESIGN_W

    def _fs(pt):
        return max(round(pt * S, 1), 8)

    def _lw(pt):
        return max(round(pt * S, 2), 0.5)

    fig_w = _TARGET_W
    fig_h = round(_DESIGN_H * S, 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, _DESIGN_W)
    ax.set_ylim(0, _DESIGN_H)
    ax.axis("off")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    fig.patch.set_facecolor(C_BG)

    # ── Title ───────────────────────────────────────────────────
    ax.text(_DESIGN_W / 2, 13.5, "DISSOLVE Middleware Architecture",
            ha="center", va="center", fontsize=_fs(20), fontweight="bold",
            color=PAL["text"])
    ax.text(_DESIGN_W / 2, 13.0,
            "Three-layer pipeline: route \u2192 execute \u2192 verify",
            ha="center", va="center", fontsize=_fs(10),
            color=PAL["text"], fontstyle="italic")

    # ── Layer box helper ────────────────────────────────────────
    box_w = 8.0
    box_h = 1.9
    hdr_h = 0.85
    bx = (_DESIGN_W - box_w) / 2
    cx = _DESIGN_W / 2

    def _layer_box(y, color, title, subtitle, role_text):
        """Draw a white box with coloured border and header band."""
        # White box with coloured border
        outer = FancyBboxPatch(
            (bx, y), box_w, box_h,
            boxstyle="round,pad=0.06",
            facecolor="#FFFFFF", edgecolor=color, linewidth=_lw(2.0),
        )
        ax.add_patch(outer)
        # Coloured header band
        hdr = FancyBboxPatch(
            (bx + 0.06, y + box_h - hdr_h), box_w - 0.12, hdr_h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor=color, linewidth=_lw(1.0),
        )
        ax.add_patch(hdr)
        # Title in header
        ax.text(cx, y + box_h - hdr_h / 2 + 0.05, title,
                ha="center", va="center", fontsize=_fs(13),
                fontweight="bold", color=PAL["text_wh"])
        # Subtitle in white area
        ax.text(cx, y + (box_h - hdr_h) / 2, subtitle,
                ha="center", va="center", fontsize=_fs(9),
                color=PAL["text"], fontstyle="italic")
        # Role annotation to the right
        ax.text(bx + box_w + 0.3, y + box_h / 2, role_text,
                ha="left", va="center", fontsize=_fs(10),
                color=color, fontstyle="italic")

    # ── Three layers ────────────────────────────────────────────
    gap = 0.85
    y1 = 8.1
    y2 = y1 - box_h - gap
    y3 = y2 - box_h - gap

    _layer_box(y1, PAL["flash"],
               "ROUTING CLASSIFIER",
               "Gemini 3 Flash  \u00b7  ~200 tokens  \u00b7  ~0.2s",
               '"Who should\n handle this?"')

    _layer_box(y2, PAL["orch_lyr"],
               "ORCHESTRATOR + SUBAGENTS",
               "Gemini 2.5 Pro  \u00b7  8 specialists  \u00b7  30\u2013120s",
               '"Do the work"')

    _layer_box(y3, PAL["verify"],
               "OUTPUT VERIFIER",
               "Gemini 3 Flash  \u00b7  ~300 tokens  \u00b7  ~0.3s",
               '"Is this correct?"')

    # ── Arrows between layers ───────────────────────────────────
    arr_kw = dict(arrowstyle="-|>", lw=_lw(2.5),
                  mutation_scale=round(15 * S))

    # User Query box (same card style as routing_detailed_flow)
    qh = 1.3
    qy = y1 + box_h + gap
    q_hdr_h = qh * 0.65
    q_outer = FancyBboxPatch(
        (bx, qy), box_w, qh,
        boxstyle="round,pad=0.06",
        facecolor="#FFFFFF", edgecolor=PAL["text"], linewidth=_lw(2.0),
    )
    ax.add_patch(q_outer)
    q_hdr = FancyBboxPatch(
        (bx + 0.06, qy + qh - q_hdr_h), box_w - 0.12, q_hdr_h,
        boxstyle="round,pad=0.04",
        facecolor=PAL["text"], edgecolor=PAL["text"], linewidth=_lw(1.0),
    )
    ax.add_patch(q_hdr)
    ax.text(cx, qy + qh - q_hdr_h / 2 + 0.02, "USER QUERY",
            ha="center", va="center", fontsize=_fs(13),
            fontweight="bold", color=PAL["text_wh"])

    # Arrow: User Query → Layer 1 (same gap as between layers)
    ax.annotate("", xy=(cx, y1 + box_h + 0.02),
                xytext=(cx, qy - 0.02),
                arrowprops=dict(color=PAL["text"], **arr_kw))

    # Layer 1 → Layer 2
    ax.annotate("", xy=(cx, y2 + box_h + 0.02),
                xytext=(cx, y1 - 0.02),
                arrowprops=dict(color=PAL["text"], **arr_kw))
    ax.text(cx + 0.55, (y1 + y2 + box_h) / 2,
            "advisory hint appended\nto system prompt",
            ha="left", va="center", fontsize=_fs(8.5), color=PAL["text"])

    # Layer 2 → Layer 3
    ax.annotate("", xy=(cx, y3 + box_h + 0.02),
                xytext=(cx, y2 - 0.02),
                arrowprops=dict(color=PAL["text"], **arr_kw))
    ax.text(cx + 0.55, (y2 + y3 + box_h) / 2,
            "draft response",
            ha="left", va="center", fontsize=_fs(8.5), color=PAL["text"])

    # Layer 3 → Output
    out_y = y3 - 0.9
    ax.annotate("", xy=(cx, out_y),
                xytext=(cx, y3 - 0.02),
                arrowprops=dict(color=PAL["text"], **arr_kw))
    ax.text(cx + 0.55, out_y + 0.25, "verified response",
            ha="left", va="center", fontsize=_fs(9), color=PAL["text"])

    # ── Revision loop (verifier → orchestrator, dashed) ─────────
    loop_x = bx - 0.15
    ax.annotate(
        "", xy=(loop_x, y2 + 0.15), xytext=(loop_x, y3 + box_h - 0.15),
        arrowprops=dict(
            arrowstyle="-|>", color=PAL["verify"], lw=_lw(1.8),
            connectionstyle="arc3,rad=0.4", linestyle="--",
            mutation_scale=round(12 * S),
        ),
    )
    ax.text(bx - 0.9, (y2 + y3 + box_h) / 2,
            "revision\n(HIGH conf.\nissues only)",
            ha="center", va="center", fontsize=_fs(8),
            color=PAL["verify"], fontstyle="italic")

    # ── Cost annotation ─────────────────────────────────────────
    ax.text(cx, out_y - 0.55,
            "Total middleware overhead: ~$0.001 and ~0.5s per query",
            ha="center", va="center", fontsize=_fs(9), color=PAL["text"])
    ax.text(cx, out_y - 0.95,
            "Both Flash layers share a single model instance",
            ha="center", va="center", fontsize=_fs(8.5),
            color=PAL["text"], fontstyle="italic")

    # ── Save ────────────────────────────────────────────────────
    for ext, dpi in [("png", 600), ("pdf", 600), ("svg", None)]:
        path = os.path.join(OUT_DIR, f"routing_three_layer.{ext}")
        kw = dict(bbox_inches="tight", facecolor=C_BG, pad_inches=0.08)
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        print(f"  Saved: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════
# Figure 2: Detailed Routing Flow
# ═════════════════════════════════════════════════════════════════

def plot_routing_flow():
    """Detailed routing classifier flow in agent_architecture style."""
    _DESIGN_W = 18.0
    _DESIGN_H = 14.0
    _TARGET_W = 7.0
    S = _TARGET_W / _DESIGN_W

    def _fs(pt):
        return max(round(pt * S, 1), 8)

    def _lw(pt):
        return max(round(pt * S, 2), 0.5)

    fig_w = _TARGET_W
    fig_h = round(_DESIGN_H * S, 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, _DESIGN_W)
    ax.set_ylim(0, _DESIGN_H)
    ax.axis("off")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    fig.patch.set_facecolor(C_BG)

    # Colours for the two paths
    C_LLM = PAL["flash"]      # purple — LLM path
    C_KW  = "#BF7B24"         # amber  — keyword path
    C_MERGE = "#1A7A6C"       # teal   — merge step
    C_OUT = PAL["orch_lyr"]   # blue   — output

    arr_kw = dict(arrowstyle="-|>", lw=_lw(2.2),
                  mutation_scale=round(14 * S))

    # ── Title ───────────────────────────────────────────────────
    ax.text(_DESIGN_W / 2, 13.5,
            "Routing Classifier \u2014 Detailed Flow",
            ha="center", va="center", fontsize=_fs(20), fontweight="bold",
            color=PAL["text"])
    ax.text(_DESIGN_W / 2, 13.0,
            "LLM-based semantic classification with keyword fallback",
            ha="center", va="center", fontsize=_fs(10),
            color=PAL["text"], fontstyle="italic")

    # ── Box helper ──────────────────────────────────────────────
    def _card(x, y, w, h, color, title, subtitle=None, hdr_frac=0.5):
        """White box + coloured header, like architecture subagent cards."""
        hdr_h = h * hdr_frac
        outer = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor="#FFFFFF", edgecolor=color, linewidth=_lw(1.8),
        )
        ax.add_patch(outer)
        hdr = FancyBboxPatch(
            (x + 0.04, y + h - hdr_h), w - 0.08, hdr_h,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor=color, linewidth=_lw(1.0),
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - hdr_h / 2, title,
                ha="center", va="center", fontsize=_fs(11),
                fontweight="bold", color=PAL["text_wh"])
        if subtitle:
            ax.text(x + w / 2, y + (h - hdr_h) / 2, subtitle,
                    ha="center", va="center", fontsize=_fs(8.5),
                    color=PAL["text"], fontstyle="italic")

    # ── LLM path (left) ────────────────────────────────────────
    lcx = 4.5
    cw, ch = 5.5, 1.5
    fy = 8.5

    # ── User Query (above Flash, left column) ──────────────────
    qy = fy + ch + 0.85          # same gap as Flash → JSON
    _card(lcx - cw / 2, qy, cw, ch, PAL["text"],
          "User Query", hdr_frac=0.65)

    # Arrow: User Query → Flash (vertical, standard gap)
    ax.annotate("", xy=(lcx, fy + ch + 0.02),
                xytext=(lcx, qy - 0.02),
                arrowprops=dict(color=C_LLM, **arr_kw))

    # Flash classifier
    _card(lcx - cw / 2, fy, cw, ch, C_LLM,
          "Gemini 3 Flash", "classify intent \u2192 JSON")

    # JSON response
    jy = 6.6
    _card(lcx - cw / 2, jy, cw, ch * 0.7, "#6C3483",
          '{"subagents": [...]}', hdr_frac=0.7)

    ax.annotate("", xy=(lcx, jy + ch * 0.7 + 0.02),
                xytext=(lcx, fy - 0.02),
                arrowprops=dict(color=C_LLM, **arr_kw))

    # Map to ROUTING_RULES
    my = 5.0
    _card(lcx - cw / 2, my, cw, ch * 0.7, "#6C3483",
          "Map names \u2192 ROUTING_RULES", hdr_frac=0.7)

    ax.annotate("", xy=(lcx, my + ch * 0.7 + 0.02),
                xytext=(lcx, jy - 0.02),
                arrowprops=dict(color="#6C3483", **arr_kw))

    # ── Keyword path (right) — fallback only ───────────────────
    rcx = 13.5

    ky = 8.5
    _card(rcx - cw / 2, ky, cw, ch, C_KW,
          "Keyword Matcher", "regex phrases + stem scoring")

    # Score + sort
    sy = 6.6
    _card(rcx - cw / 2, sy, cw, ch * 0.7, "#9E5523",
          "Score rules \u2192 sort by priority", hdr_frac=0.7)

    ax.annotate("", xy=(rcx, sy + ch * 0.7 + 0.02),
                xytext=(rcx, ky - 0.02),
                arrowprops=dict(color=C_KW, **arr_kw))

    # list[dict] output
    ly = 5.0
    _card(rcx - cw / 2, ly, cw, ch * 0.7, "#9E5523",
          "list[dict]  matched rules", hdr_frac=0.7)

    ax.annotate("", xy=(rcx, ly + ch * 0.7 + 0.02),
                xytext=(rcx, sy - 0.02),
                arrowprops=dict(color="#9E5523", **arr_kw))

    # ── Dashed fallback arrow (LLM parse failure → keywords) ────
    ax.annotate(
        "", xy=(rcx - cw / 2, ky + ch / 2),
        xytext=(lcx + cw / 2, fy + ch / 2),
        arrowprops=dict(
            arrowstyle="-|>", color=PAL["text"], lw=_lw(1.5),
            connectionstyle="arc3,rad=-0.15", linestyle="--",
            mutation_scale=round(12 * S),
        ),
    )
    ax.text(_DESIGN_W / 2, fy + ch / 2 + 0.35,
            "fallback on parse failure",
            ha="center", va="center", fontsize=_fs(8),
            color=PAL["text"], fontstyle="italic")

    # ── Merge: _build_hint_from_matches() ───────────────────────
    mw = 6.5
    mh = 1.5
    mx = (_DESIGN_W - mw) / 2
    mby = 2.8
    _card(mx, mby, mw, mh, C_MERGE,
          "_build_hint_from_matches()",
          "single / parallel / sequential hint")

    # Arrows from both paths to merge
    ax.annotate("", xy=(mx + 0.2, mby + mh / 2),
                xytext=(lcx, my - 0.02),
                arrowprops=dict(color="#6C3483", connectionstyle="arc3,rad=0.15",
                                **arr_kw))
    ax.annotate("", xy=(mx + mw - 0.2, mby + mh / 2),
                xytext=(rcx, ly - 0.02),
                arrowprops=dict(color="#9E5523", connectionstyle="arc3,rad=-0.15",
                                **arr_kw))

    # ── Output: advisory hint → system prompt ───────────────────
    ow, oh = 6.5, 1.1
    ox = (_DESIGN_W - ow) / 2
    oy = 1.0
    _card(ox, oy, ow, oh, C_OUT,
          "Advisory hint \u2192 system prompt", hdr_frac=0.65)

    ax.annotate("", xy=(_DESIGN_W / 2, oy + oh + 0.02),
                xytext=(_DESIGN_W / 2, mby - 0.02),
                arrowprops=dict(color=C_MERGE, **arr_kw))

    # ── Save ────────────────────────────────────────────────────
    for ext, dpi in [("png", 600), ("pdf", 600), ("svg", None)]:
        path = os.path.join(OUT_DIR, f"routing_detailed_flow.{ext}")
        kw = dict(bbox_inches="tight", facecolor=C_BG, pad_inches=0.08)
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        print(f"  Saved: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════
# Figure 3: Output Verifier — Detailed Flow
# ═════════════════════════════════════════════════════════════════

def plot_verifier_flow():
    """Output verifier detailed flow in agent_architecture style."""
    _DESIGN_W = 16.0
    _DESIGN_H = 15.5
    _TARGET_W = 7.0
    S = _TARGET_W / _DESIGN_W

    def _fs(pt):
        return max(round(pt * S, 1), 8)

    def _lw(pt):
        return max(round(pt * S, 2), 0.5)

    fig_w = _TARGET_W
    fig_h = round(_DESIGN_H * S, 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, _DESIGN_W)
    ax.set_ylim(0, _DESIGN_H)
    ax.axis("off")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    fig.patch.set_facecolor(C_BG)

    # Colours
    C_ORCH = PAL["orch"]        # blue — orchestrator
    C_FLASH = PAL["flash"]      # purple — verifier model
    C_VERDICT = "#1A7A6C"       # teal — verdict
    C_REVISE = PAL["verify"]    # red — revision path
    C_PASS = "#247A5F"          # green — pass through
    C_OUT = PAL["orch"]         # blue — output

    arr_kw = dict(arrowstyle="-|>", lw=_lw(2.2),
                  mutation_scale=round(14 * S))

    cx = _DESIGN_W / 2
    cw = 5.5    # main card width
    ch = 1.5    # main card height
    ch_sm = 1.05  # small card height

    # ── Card helper ────────────────────────────────────────────
    def _card(x, y, w, h, color, title, subtitle=None, hdr_frac=0.5):
        """White box + coloured header, like architecture subagent cards."""
        hdr_h = h * hdr_frac
        outer = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor="#FFFFFF", edgecolor=color, linewidth=_lw(1.8),
        )
        ax.add_patch(outer)
        hdr = FancyBboxPatch(
            (x + 0.04, y + h - hdr_h), w - 0.08, hdr_h,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor=color, linewidth=_lw(1.0),
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - hdr_h / 2, title,
                ha="center", va="center", fontsize=_fs(11),
                fontweight="bold", color=PAL["text_wh"])
        if subtitle:
            ax.text(x + w / 2, y + (h - hdr_h) / 2, subtitle,
                    ha="center", va="center", fontsize=_fs(8.5),
                    color=PAL["text"], fontstyle="italic")

    # ── Title ──────────────────────────────────────────────────
    ax.text(cx, 15.0, "Output Verifier \u2014 Detailed Flow",
            ha="center", va="center", fontsize=_fs(20), fontweight="bold",
            color=PAL["text"])
    ax.text(cx, 14.5, "Single reflection pass on final synthesis",
            ha="center", va="center", fontsize=_fs(10),
            color=PAL["text"], fontstyle="italic")

    # ── 1. Orchestrator Draft Response (top) ───────────────────
    y1 = 12.5
    _card(cx - cw / 2, y1, cw, ch, C_ORCH,
          "Orchestrator Response", "model output after agent loop")

    # ── Right branch: tool calls → pass through ────────────────
    pw, px = 3.2, cx + cw / 2 + 1.5
    py = y1 + ch / 2 - ch_sm / 2
    _card(px, py, pw, ch_sm, C_PASS, "Pass Through", hdr_frac=0.65)

    ax.annotate("", xy=(px, y1 + ch / 2),
                xytext=(cx + cw / 2 + 0.02, y1 + ch / 2),
                arrowprops=dict(color=C_PASS, **arr_kw))
    mid_px = (cx + cw / 2 + px) / 2
    ax.text(mid_px, y1 + ch / 2 + 0.35, "has tool calls",
            ha="center", va="center", fontsize=_fs(8),
            color=C_PASS, fontstyle="italic")
    ax.text(px + pw / 2, py - 0.3, "(agent still iterating)",
            ha="center", va="center", fontsize=_fs(7.5),
            color=PAL["text"], fontstyle="italic")

    # ── Arrow down: no tool calls ──────────────────────────────
    y2 = 9.8
    ax.annotate("", xy=(cx, y2 + ch + 0.02),
                xytext=(cx, y1 - 0.02),
                arrowprops=dict(color=PAL["text"], **arr_kw))
    ax.text(cx + 0.4, (y1 + y2 + ch) / 2,
            "no tool calls\n(final answer)",
            ha="left", va="center", fontsize=_fs(8.5),
            color=PAL["text"])

    # ── 2. Gemini 3 Flash Verifier ─────────────────────────────
    _card(cx - cw / 2, y2, cw, ch, C_FLASH,
          "Gemini 3 Flash", "scientific accuracy verification")

    # ── Right annotation: what it checks ───────────────────────
    chk_x = cx + cw / 2 + 0.5
    chk_y = y2 + ch / 2
    ax.text(chk_x, chk_y + 0.45, "Checks:",
            ha="left", va="center", fontsize=_fs(8),
            fontweight="bold", color=C_FLASH)
    for i, txt in enumerate([
        "Unsupported claims",
        "Internal contradictions",
        "Missing safety caveats",
        "Logical gaps",
    ]):
        ax.text(chk_x + 0.15, chk_y + 0.15 - i * 0.28,
                f"\u2022 {txt}",
                ha="left", va="center", fontsize=_fs(7.5),
                color=PAL["text"])

    # ── Arrow down ─────────────────────────────────────────────
    y3 = 7.5
    ax.annotate("", xy=(cx, y3 + ch_sm + 0.02),
                xytext=(cx, y2 - 0.02),
                arrowprops=dict(color=C_FLASH, **arr_kw))

    # ── 3. JSON Verdict ────────────────────────────────────────
    _card(cx - cw / 2, y3, cw, ch_sm, C_VERDICT,
          '{"pass", "confidence", "issues"}', hdr_frac=0.65)

    # ── Arrow down: pass path ──────────────────────────────────
    y4 = 5.2
    ax.annotate("", xy=(cx, y4 + ch_sm + 0.02),
                xytext=(cx, y3 - 0.02),
                arrowprops=dict(color=C_VERDICT, **arr_kw))
    ax.text(cx + 0.4, (y3 + y4 + ch_sm) / 2,
            "pass = true OR\nconfidence \u2260 HIGH",
            ha="left", va="center", fontsize=_fs(8.5),
            color=PAL["text"])

    # ── 4. Verified Response → User ────────────────────────────
    _card(cx - cw / 2, y4, cw, ch_sm, C_OUT,
          "Verified Response \u2192 User", hdr_frac=0.65)

    # ── Left revision loop (dashed, red — routes AROUND boxes) ──
    card_left = cx - cw / 2
    margin_x = card_left - 1.5          # vertical channel, clear of all cards
    vy_start = y3 + ch_sm / 2           # verdict card midpoint
    vy_end = y1 + ch / 2               # orchestrator card midpoint
    loop_lw = _lw(3.0)

    # Segment 1: horizontal — verdict left edge → margin
    ax.plot([card_left, margin_x], [vy_start, vy_start],
            color=C_REVISE, lw=loop_lw, linestyle="--",
            dash_capstyle="round", zorder=2)
    # Segment 2: vertical — up from verdict level to orchestrator level
    ax.plot([margin_x, margin_x], [vy_start, vy_end],
            color=C_REVISE, lw=loop_lw, linestyle="--",
            dash_capstyle="round", zorder=2)
    # Segment 3: horizontal — margin → orchestrator left edge (with arrowhead)
    ax.annotate("", xy=(card_left, vy_end),
                xytext=(margin_x, vy_end),
                arrowprops=dict(arrowstyle="-|>", color=C_REVISE,
                                lw=loop_lw, linestyle="--",
                                mutation_scale=round(14 * S)))

    # Labels alongside vertical segment
    lbl_x = margin_x - 1.2
    lbl_y = (vy_start + vy_end) / 2
    ax.text(lbl_x, lbl_y + 0.4,
            "HIGH conf. issues",
            ha="center", va="center", fontsize=_fs(8),
            color=C_REVISE, fontweight="bold")
    ax.text(lbl_x, lbl_y - 0.25,
            "inject feedback\nre-invoke model\n(max 1 revision)",
            ha="center", va="center", fontsize=_fs(7.5),
            color=C_REVISE, fontstyle="italic")

    # ── Bottom annotations ─────────────────────────────────────
    ax.text(cx, y4 - 0.55,
            "Max 1 revision per invocation \u00b7 fail-open on parse errors",
            ha="center", va="center", fontsize=_fs(9), color=PAL["text"])
    ax.text(cx, y4 - 0.95,
            "Shares Flash model instance with routing classifier",
            ha="center", va="center", fontsize=_fs(8.5),
            color=PAL["text"], fontstyle="italic")

    # ── Save ───────────────────────────────────────────────────
    for ext, dpi in [("png", 600), ("pdf", 600), ("svg", None)]:
        path = os.path.join(OUT_DIR, f"output_verifier_flow.{ext}")
        kw = dict(bbox_inches="tight", facecolor=C_BG, pad_inches=0.08)
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_three_layer()
    plot_routing_flow()
    plot_verifier_flow()
