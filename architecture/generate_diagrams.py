"""Generate architecture diagrams for DISSOLVE agent system.

Produces three figure types:
1. agent_architecture — high-level overview (publication-quality)
2. subagent_tool_trees — detailed 3-column tree of all tools
3. individual subagent PNGs — one per subagent with full tool lists
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans", "Arial", "Helvetica", "DejaVu Sans",
]
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text editable in SVG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour palette ────────────────────────────────────────────────
PAL = {
    "orch":     "#3b5998",   # classic Facebook blue (orchestrator only)
    "core_bg":  "#E8EDF2",
    "core_bd":  "#A8B8C8",
    "grp_bg":   "#F0F2F4",
    "grp_bd":   "#B8BFC6",
    "text":     "#1C2A3A",
    "text_lt":  "#5A6A7A",
    "text_wh":  "#FFFFFF",
    "arrow":    "#8899AA",
    "flow_arr": "#5C7080",
    # Subagent identity colours (muted, grayscale-distinguishable)
    "sep":      "#B83230",
    "safety":   "#7D5BA6",
    "tea":      "#1A7A6C",
    "scholar":  "#BF7B24",
    "patent":   "#9E5523",
    "rag":      "#2968A3",
    "viz":      "#6A4C93",
    "stats":    "#247A5F",
    "biosteam": "#2D8B72",
}

# Legacy aliases for tool-tree and individual-diagram functions
C_MAIN = PAL["orch"]
C_CORE = "#5B7FA5"
C_TOOL = "#27AE60"
C_BG   = "#FFFFFF"
C_TEXT  = PAL["text"]
C_WHITE = "#FFFFFF"

# ── data (matches subagents.yaml + __init__.py tool assignments) ─
CORE_TOOLS = {
    "Database Query": [
        "list_tables", "describe_table", "check_column_values",
        "query_database", "validate_and_query",
    ],
    "Listing": [
        "list_available_solvents", "list_available_polymers",
    ],
    "Solvent Properties": [
        "get_solvent_properties", "rank_solvents_by_property",
    ],
    "Interpolation": [
        "predict_solubility", "predict_solubility_range",
        "list_interpolation_coverage", "rank_solvents_selectivity",
    ],
}

SUBAGENTS = {
    "separation-engineer": {
        "color": PAL["sep"],
        "groups": {
            "Sequence Planning": [
                "find_optimal_separation_sequence",
                "plan_sequential_separation",
                "plan_multiple_separation_schemes",
                "view_alternative_separation_sequence",
            ],
            "Selectivity Analysis": [
                "optimize_separation_temperature",
                "calculate_selectivity_detailed",
                "rank_solvents_for_separation",
                "build_compatibility_matrix",
                "find_challenging_polymer_pairs",
            ],
            "Precipitation": [
                "find_differential_precipitation_solvents",
                "analyze_multi_polymer_precipitation",
                "analyze_precipitation_temperature",
                "compare_polymer_pairs_precipitation",
                "check_atmospheric_feasibility",
                "check_multi_polymer_atmospheric_feasibility",
            ],
            "Antisolvent": [
                "find_antisolvents",
                "find_antisolvent_pairs",
                "analyze_selective_antisolvent_precipitation",
            ],
            "Adaptive": [
                "find_optimal_separation_conditions",
                "analyze_selective_solubility_enhanced",
            ],
        },
    },
    "safety-analyst": {
        "color": PAL["safety"],
        "groups": {
            "GSK G-Scores": [
                "get_solvent_gscore",
                "get_family_alternatives",
                "visualize_gscores",
            ],
            "PubChem Safety": [
                "get_pubchem_safety_info",
                "compare_pubchem_safety",
                "visualize_pubchem_safety",
                "get_pubchem_toxicity",
            ],
        },
    },
    "biosteam-analyst": {
        "color": PAL["biosteam"],
        "groups": {
            "BioSTEAM Simulation": [
                "run_biosteam_simulation",
                "run_biosteam_batch",
                "compare_biosteam_scenarios",
                "get_biosteam_solvents",
            ],
            "Analysis & Viz": [
                "visualize_biosteam_results",
                "run_biosteam_multi_polymer",
            ],
        },
    },
    "scholar-researcher": {
        "color": PAL["scholar"],
        "groups": {
            "Academic Search": [
                "search_google_scholar",
                "search_web_of_science",
                "search_arxiv",
            ],
        },
    },
    "patent-researcher": {
        "color": PAL["patent"],
        "groups": {
            "Patent Search": [
                "search_google_patents",
                "search_patentsview",
                "lookup_patent",
            ],
        },
    },
    "rag-analyst": {
        "color": PAL["rag"],
        "groups": {
            "Core RAG": [
                "search_literature_rag",
                "ingest_pdf_to_rag",
                "get_rag_status",
                "ask_literature",
                "clear_rag_index",
                "download_pdf_to_rag",
            ],
            "Quality": [
                "visualize_rag_chunks",
                "check_rag_chunk_quality",
                "get_rag_chunk_report",
            ],
            "Diagnostics": [
                "analyze_search_diagnostics",
                "visualize_retrieval_patterns",
                "visualize_embedding_space",
                "analyze_document_similarity",
                "analyze_dense_vs_sparse",
                "analyze_reranking_impact",
                "analyze_section_boost",
                "analyze_query_expansion",
                "run_full_rag_diagnostics",
            ],
        },
    },
    "visualization-specialist": {
        "color": PAL["viz"],
        "groups": {
            "General Plots": [
                "plot_solubility_vs_temperature",
                "plot_solubility_vs_temperature_interactive",
                "plot_selectivity_heatmap",
                "plot_multi_panel_analysis",
                "plot_comparison_dashboard",
                "plot_solvent_properties",
                "plot_interpolation_vs_sql",
            ],
            "Separation Plots": [
                "create_separation_tree_plot",
                "create_selectivity_heatmap",
                "create_process_flow_diagram",
                "plot_precipitation_curves",
                "plot_atmospheric_feasibility",
            ],
        },
    },
    "statistics-ml": {
        "color": PAL["stats"],
        "groups": {
            "Statistics": [
                "statistical_summary",
                "correlation_analysis",
                "compare_groups_statistically",
                "regression_analysis",
            ],
            "ML Prediction": [
                "predict_solubility_ml",
            ],
            "Thermal Properties": [
                "predict_thermal_properties",
                "lookup_tg",
                "generate_solubility_for_new_polymer",
                "list_generated_polymers",
            ],
        },
    },
}


def _pretty(name: str) -> str:
    """Convert snake_case tool name to readable form."""
    return name.replace("_", " ").replace("  ", " ")


# ════════════════════════════════════════════════════════════════
# Figure 1 — Minimal publication-quality agent architecture
# ════════════════════════════════════════════════════════════════
def generate_agent_architecture():
    """Clean minimal architecture: orchestrator + grouped subagents.

    7-inch wide figure for journal publication (600 DPI).
    Coordinate system kept at the 15-unit design grid; figsize
    controls physical output.  Fonts and linewidths are scaled
    proportionally with an 8 pt floor.
    """
    # ── scaling ───────────────────────────────────────────────────
    _DESIGN_W = 15.0
    _TARGET_W = 7.0
    S = _TARGET_W / _DESIGN_W          # 0.4667

    def _fs(pt):
        """Scale font, floor at 8 pt."""
        return max(round(pt * S, 1), 8)

    def _lw(pt):
        """Scale linewidth, floor at 0.5 pt."""
        return max(round(pt * S, 2), 0.5)

    fig_w, fig_h = _TARGET_W, round(5.5 * S, 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, _DESIGN_W)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    fig.patch.set_facecolor(C_BG)

    # ── Title ─────────────────────────────────────────────────────
    ax.text(_DESIGN_W / 2, 5.2, "DISSOLVE Agent Architecture",
            ha="center", va="center", fontsize=_fs(20), fontweight="bold",
            color=PAL["text"])

    # ── Orchestrator box ──────────────────────────────────────────
    ox, ow = 3.5, 8.0
    oy, oh = 3.95, 0.88

    orch = FancyBboxPatch(
        (ox, oy), ow, oh,
        boxstyle="round,pad=0.12",
        facecolor=PAL["orch"], edgecolor=PAL["orch"], linewidth=_lw(3.5),
    )
    ax.add_patch(orch)

    ax.text(_DESIGN_W / 2, oy + oh / 2 + 0.12, "DISSOLVE Orchestrator",
            ha="center", va="center", fontsize=_fs(14), fontweight="bold",
            color=PAL["text_wh"])
    ax.text(_DESIGN_W / 2, oy + oh / 2 - 0.2, "13 core tools",
            ha="center", va="center", fontsize=_fs(9.5),
            color="#95A5A6")

    # ── Subagent grid: equal-width boxes in 3 groups ──────────────
    box_w = 1.52
    within_gap = 0.12
    between_gap = 0.5
    grp_pad_x = 0.12

    groups_data = [
        {
            "label": "Process Engineering",
            "agents": [
                ("Separation\nEngineer", "sep", "20 tools"),
                ("Safety\nAnalyst", "safety", "7 tools"),
                ("BioSTEAM\nAnalyst", "biosteam", "6 tools"),
            ],
        },
        {
            "label": "Research & Retrieval",
            "agents": [
                ("Scholar\nResearcher", "scholar", "3 tools"),
                ("Patent\nResearcher", "patent", "3 tools"),
                ("RAG\nAnalyst", "rag", "18 tools"),
            ],
        },
        {
            "label": "Analytics & Output",
            "agents": [
                ("Visualization\nSpecialist", "viz", "12 tools"),
                ("Statistics\n& ML", "stats", "9 tools"),
            ],
        },
    ]

    # Compute total width and centering offset
    total_w = 0
    for g in groups_data:
        n = len(g["agents"])
        total_w += n * box_w + (n - 1) * within_gap + 2 * grp_pad_x
    total_w += (len(groups_data) - 1) * between_gap
    start_x = (_DESIGN_W - total_w) / 2

    # Box geometry
    box_top = 2.95
    box_h = 1.6
    box_bot = box_top - box_h
    hdr_h = 0.7

    # Group geometry
    grp_top = 3.42
    grp_bot = box_bot - 0.16

    agent_centers = []
    cursor_x = start_x

    for group in groups_data:
        n = len(group["agents"])
        grp_w = n * box_w + (n - 1) * within_gap + 2 * grp_pad_x
        gx = cursor_x

        # Group background (prominent border)
        grp_bg = FancyBboxPatch(
            (gx, grp_bot), grp_w, grp_top - grp_bot,
            boxstyle="round,pad=0.08",
            facecolor="#F0F2F4", edgecolor="#B8BFC6", linewidth=_lw(1.5),
        )
        ax.add_patch(grp_bg)

        # Group label
        ax.text(gx + grp_w / 2, grp_top - 0.12, group["label"],
                ha="center", va="center", fontsize=_fs(11), fontweight="bold",
                color=PAL["text"])

        box_start_x = gx + grp_pad_x
        for j, (name, key, count) in enumerate(group["agents"]):
            bx = box_start_x + j * (box_w + within_gap)
            color = PAL[key]

            # White box with coloured border
            agent_box = FancyBboxPatch(
                (bx, box_bot), box_w, box_h,
                boxstyle="round,pad=0.05",
                facecolor="#FFFFFF", edgecolor=color, linewidth=_lw(1.8),
            )
            ax.add_patch(agent_box)

            # Coloured header band
            hdr = FancyBboxPatch(
                (bx + 0.03, box_top - hdr_h), box_w - 0.06, hdr_h,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor=color, linewidth=_lw(1.0),
            )
            ax.add_patch(hdr)

            # Agent name (in header)
            ax.text(bx + box_w / 2, box_top - hdr_h / 2, name,
                    ha="center", va="center", fontsize=_fs(10),
                    fontweight="bold",
                    color=PAL["text_wh"], linespacing=1.1)

            # Tool count (centred in white area)
            count_y = box_bot + (box_h - hdr_h) / 2
            ax.text(bx + box_w / 2, count_y, count,
                    ha="center", va="center", fontsize=_fs(10.5),
                    color=color, fontweight="bold")

            # Record centre for delegation arrows
            cx = bx + box_w / 2
            agent_centers.append((cx, grp_top, color))

        cursor_x += grp_w + between_gap

    # ── Delegation arrows ─────────────────────────────────────────
    for cx, top_y, color in agent_centers:
        src_x = max(ox + 0.3, min(ox + ow - 0.3, cx))
        ax.annotate(
            "", xy=(cx, top_y + 0.02), xytext=(src_x, oy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=_lw(2.0),
                            alpha=0.55, mutation_scale=round(15 * S)),
        )

    # ── Save (PNG at 600 DPI, plus PDF and SVG vector formats) ───
    for ext, dpi in [("png", 600), ("pdf", 600), ("svg", None)]:
        path = os.path.join(OUT_DIR, f"agent_architecture.{ext}")
        kw = dict(bbox_inches="tight", facecolor=C_BG, pad_inches=0.08)
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        print(f"  Saved: {path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Figure 1b — Separation Engineer detail (publication-quality)
# ════════════════════════════════════════════════════════════════
def generate_separation_engineer_detail():
    """Separation Engineer subagent detail diagram.

    Shows all 5 tool groups with individual tools listed inside each
    box, matching the clean style of the main architecture figure.
    Two-row layout (3 + 2) to give enough width for long tool names.
    7-inch wide, 600 DPI.
    """
    COLOR = PAL["sep"]
    groups = SUBAGENTS["separation-engineer"]["groups"]
    total_tools = sum(len(t) for t in groups.values())
    group_names = list(groups.keys())
    group_tools = list(groups.values())

    # ── scaling ────────────────────────────────────────────────
    _DESIGN_W = 20.0
    _TARGET_W = 7.0
    S = _TARGET_W / _DESIGN_W

    def _fs(pt):
        return max(round(pt * S, 1), 8)

    def _lw(pt):
        return max(round(pt * S, 2), 0.5)

    # Show only the 2 most representative tools per group
    display_tools = {
        "Sequence Planning": [
            "plan_multiple_separation_schemes",
            "find_optimal_separation_sequence",
        ],
        "Selectivity Analysis": [
            "calculate_selectivity_detailed",
            "rank_solvents_for_separation",
        ],
        "Precipitation": [
            "find_differential_precipitation_solvents",
            "check_atmospheric_feasibility",
        ],
        "Antisolvent": [
            "find_antisolvent_pairs",
            "analyze_selective_antisolvent_precipitation",
        ],
        "Adaptive": [
            "find_optimal_separation_conditions",
            "analyze_selective_solubility_enhanced",
        ],
    }

    # ── two-row grid layout ────────────────────────────────────
    # Row 1: Sequence Planning, Selectivity Analysis, Precipitation
    # Row 2: Antisolvent, Adaptive (centred)
    rows = [
        [0, 1, 2],   # indices into group_names
        [3, 4],
    ]

    col_gap = 0.3
    hdr_band_h = 0.7
    tool_row_h = 0.55
    tool_pad = 0.18
    row_gap = 0.55
    title_area_h = 1.6     # space for title box + gap

    # Column width: divide available width evenly per row (use wider row)
    n_max = max(len(r) for r in rows)
    col_w = (_DESIGN_W - 1.0 - (n_max - 1) * col_gap) / n_max  # ~6.1

    # Per-group box height (2 shown tools + optional "+ N more" line)
    def _box_h(idx):
        gname = group_names[idx]
        shown = display_tools.get(gname, group_tools[idx][:2])
        extra = len(group_tools[idx]) - len(shown)
        n_lines = len(shown) + (0.8 if extra > 0 else 0)
        return hdr_band_h + 2 * tool_pad + n_lines * tool_row_h

    # Row heights (max in each row)
    row_max_h = [max(_box_h(i) for i in r) for r in rows]

    fig_design_h = (
        title_area_h + 0.5
        + sum(row_max_h) + (len(rows) - 1) * row_gap
        + 0.4
    )
    fig_w = _TARGET_W
    fig_h = round(fig_design_h * S, 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, _DESIGN_W)
    ax.set_ylim(0, fig_design_h)
    ax.axis("off")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    fig.patch.set_facecolor(C_BG)

    # ── Title header ───────────────────────────────────────────
    title_top = fig_design_h - 0.15
    thdr_w = 10.0
    thdr_h = 1.0
    thdr_x = (_DESIGN_W - thdr_w) / 2
    thdr_y = title_top - thdr_h

    title_box = FancyBboxPatch(
        (thdr_x, thdr_y), thdr_w, thdr_h,
        boxstyle="round,pad=0.12",
        facecolor=COLOR, edgecolor=COLOR, linewidth=_lw(3.5),
    )
    ax.add_patch(title_box)
    ax.text(_DESIGN_W / 2, thdr_y + thdr_h / 2 + 0.1,
            "Separation Engineer",
            ha="center", va="center", fontsize=_fs(18), fontweight="bold",
            color=PAL["text_wh"])
    ax.text(_DESIGN_W / 2, thdr_y + thdr_h / 2 - 0.25,
            f"{total_tools} tools  |  {len(group_names)} groups",
            ha="center", va="center", fontsize=_fs(10),
            color="#DDAAAA")

    # ── Draw rows of group cards ───────────────────────────────
    cursor_y = thdr_y - 0.5   # top of first row

    for row_idx, row_indices in enumerate(rows):
        n_cols = len(row_indices)
        row_w = n_cols * col_w + (n_cols - 1) * col_gap
        row_start_x = (_DESIGN_W - row_w) / 2
        row_h = row_max_h[row_idx]

        for j, gi in enumerate(row_indices):
            gname = group_names[gi]
            tools = group_tools[gi]
            bx = row_start_x + j * (col_w + col_gap)
            box_h = row_h  # uniform height within each row
            by = cursor_y - box_h

            # White box with coloured border
            box = FancyBboxPatch(
                (bx, by), col_w, box_h,
                boxstyle="round,pad=0.06",
                facecolor="#FFFFFF", edgecolor=COLOR, linewidth=_lw(1.8),
            )
            ax.add_patch(box)

            # Coloured header band
            hdr = FancyBboxPatch(
                (bx + 0.05, by + box_h - hdr_band_h),
                col_w - 0.10, hdr_band_h,
                boxstyle="round,pad=0.04",
                facecolor=COLOR, edgecolor=COLOR, linewidth=_lw(1.0),
            )
            ax.add_patch(hdr)

            # Group name
            ax.text(bx + col_w / 2, by + box_h - hdr_band_h / 2,
                    gname, ha="center", va="center",
                    fontsize=_fs(11), fontweight="bold", color=PAL["text_wh"])

            # Tool names (sans-serif, left-aligned) — 2 key tools + "N more"
            shown = display_tools.get(gname, tools[:2])
            extra = len(tools) - len(shown)
            tool_top = by + box_h - hdr_band_h - tool_pad
            for k, tool in enumerate(shown):
                ty = tool_top - (k + 0.5) * tool_row_h
                ax.text(bx + 0.25, ty, _pretty(tool),
                        ha="left", va="center", fontsize=_fs(9),
                        color=PAL["text"])
            if extra > 0:
                ty = tool_top - (len(shown) + 0.3) * tool_row_h
                ax.text(bx + 0.25, ty, f"+ {extra} more",
                        ha="left", va="center", fontsize=_fs(8.5),
                        color=PAL["text_lt"], style="italic")

            # Arrow from title to group card
            arr_src_x = max(thdr_x + 0.4, min(thdr_x + thdr_w - 0.4,
                            bx + col_w / 2))
            ax.annotate(
                "", xy=(bx + col_w / 2, cursor_y + 0.02),
                xytext=(arr_src_x, thdr_y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR, lw=_lw(2.0),
                                alpha=0.45, mutation_scale=round(15 * S)),
            )

        cursor_y -= row_h + row_gap

    # ── Save ───────────────────────────────────────────────────
    for ext, dpi in [("png", 600), ("pdf", 600), ("svg", None)]:
        path = os.path.join(OUT_DIR, f"separation_engineer.{ext}")
        kw = dict(bbox_inches="tight", facecolor=C_BG, pad_inches=0.08)
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        print(f"  Saved: {path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Figure 2 — Combined tool tree (single tall image)
# ════════════════════════════════════════════════════════════════
def _draw_tree(ax, section_name, color, groups, x_off, y_top, col_width):
    """Draw one subagent tree at the given position. Returns y_bottom."""
    total_tools = sum(len(t) for t in groups.values())

    # Root node
    root_x = x_off + col_width / 2
    ax.text(root_x, y_top, section_name.replace("-", " ").title(),
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_WHITE, zorder=5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                      edgecolor=color, linewidth=2))
    ax.text(root_x, y_top - 0.55, f"{total_tools} tools",
            ha="center", va="center", fontsize=8, color="#7F8C8D")

    group_x = x_off + col_width * 0.35
    tool_x = x_off + 0.3
    y = y_top - 1.3

    for group_name, tools in groups.items():
        group_y = y
        # Line from root to group
        ax.plot([root_x, group_x], [y_top - 0.7, group_y],
                color=color, lw=1.5, alpha=0.4, zorder=1)
        # Group label
        ax.text(group_x, group_y, f"  {group_name}  ",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=C_TEXT, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color,
                          edgecolor=color, linewidth=1.5, alpha=0.18))
        y -= 0.65

        for tool in tools:
            # Line from group to tool
            ax.plot([group_x - 0.3, tool_x + 0.1], [group_y, y],
                    color=C_TOOL, lw=0.7, alpha=0.3, zorder=1)
            # Tool label
            ax.text(tool_x, y, _pretty(tool),
                    ha="left", va="center", fontsize=7.5, color=C_TEXT,
                    fontfamily="monospace", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                              edgecolor=C_TOOL, linewidth=0.6, alpha=0.85))
            y -= 0.52
        y -= 0.35

    return y


def generate_tool_trees():
    all_sections = {"Core Tools (Main Agent)": {
        "color": C_CORE, "groups": CORE_TOOLS
    }}
    for name, info in SUBAGENTS.items():
        all_sections[name] = info

    # Compute height needed per section
    def _section_height(groups):
        n_tools = sum(len(t) for t in groups.values())
        n_groups = len(groups)
        return 1.3 + n_groups * 0.65 + n_tools * 0.52 + n_groups * 0.35 + 1.0

    sections = list(all_sections.items())

    # Split into 3 columns, balancing total height
    col_assignments = [[], [], []]
    col_heights = [0.0, 0.0, 0.0]

    for name, info in sections:
        h = _section_height(info["groups"])
        min_col = col_heights.index(min(col_heights))
        col_assignments[min_col].append((name, info, h))
        col_heights[min_col] += h + 1.5

    max_height = max(col_heights) + 2
    fig_width = 28
    col_width = fig_width / 3

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, max_height * 0.58))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(-max_height, 2)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.text(fig_width / 2, 1.3, "DISSOLVE \u2014 Subagent Tool Trees",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C_MAIN)

    for col_idx, col in enumerate(col_assignments):
        x_off = col_idx * col_width + 0.3
        y_top = 0.0
        for name, info, h in col:
            _draw_tree(ax, name, info["color"], info["groups"],
                       x_off, y_top, col_width - 0.6)
            y_top -= h + 1.5

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "subagent_tool_trees.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════
# Figure 3 — Individual subagent PNGs
# ════════════════════════════════════════════════════════════════
def generate_individual_subagent_diagrams():
    """Generate one PNG per subagent (+ core), showing tools."""
    all_sections = {
        "Core Tools (Main Agent)": {"color": C_CORE, "groups": CORE_TOOLS},
    }
    for name, info in SUBAGENTS.items():
        all_sections[name] = info

    for section_name, info in all_sections.items():
        color = info["color"]
        groups = info["groups"]

        n_tools = sum(len(t) for t in groups.values())
        n_groups = len(groups)

        fig_width = 10
        content_height = 1.8 + n_groups * 0.65 + n_tools * 0.52 + n_groups * 0.35
        fig_height = max(content_height * 0.55 + 1.5, 4)

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.set_xlim(0, fig_width)
        ax.set_ylim(-content_height - 0.5, 2.5)
        ax.axis("off")
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_BG)

        display_title = section_name.replace("-", " ").title()
        ax.text(fig_width / 2, 2.0, display_title,
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=C_WHITE, zorder=5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color,
                          edgecolor=color, linewidth=2.5))
        ax.text(fig_width / 2, 1.3,
                f"{n_tools} tools across "
                f"{n_groups} group{'s' if n_groups != 1 else ''}",
                ha="center", va="center", fontsize=10, color="#7F8C8D")

        root_x = fig_width / 2
        group_x = fig_width * 0.35
        tool_x = 1.0
        y = 0.3

        for group_name, tools in groups.items():
            group_y = y
            ax.plot([root_x, group_x], [1.0, group_y],
                    color=color, lw=2, alpha=0.4, zorder=1)
            ax.text(group_x, group_y, f"  {group_name}  ",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=C_TEXT, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                              edgecolor=color, linewidth=1.5, alpha=0.18))
            y -= 0.7

            for tool in tools:
                ax.plot([group_x - 0.5, tool_x + 0.1], [group_y, y],
                        color=C_TOOL, lw=0.8, alpha=0.3, zorder=1)
                ax.text(tool_x, y, _pretty(tool),
                        ha="left", va="center", fontsize=9.5, color=C_TEXT,
                        fontfamily="monospace", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=C_BG,
                                  edgecolor=C_TOOL, linewidth=0.8, alpha=0.9))
                y -= 0.55
            y -= 0.4

        plt.tight_layout()
        fname = (section_name.lower()
                 .replace(" ", "_").replace("(", "").replace(")", "")
                 .replace("-", "_").replace("__", "_").strip("_"))
        path = os.path.join(OUT_DIR, f"{fname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Generating agent architecture (publication-quality)...")
    generate_agent_architecture()
    print("Generating tool trees...")
    generate_tool_trees()
    print("Generating individual subagent diagrams...")
    generate_individual_subagent_diagrams()
    # Run last so the pub-quality card version overwrites the tree version
    print("Generating separation engineer detail (publication-quality)...")
    generate_separation_engineer_detail()
    print("Done.")
