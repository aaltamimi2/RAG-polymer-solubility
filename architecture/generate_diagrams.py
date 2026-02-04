"""Generate architecture diagrams for DISSOLVE agent system."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour palette ──────────────────────────────────────────────
C_MAIN      = "#2C3E50"   # dark blue-grey (main agent)
C_CORE      = "#3498DB"   # blue (core tools)
C_SUB       = "#E67E22"   # orange (subagents)
C_TOOL      = "#27AE60"   # green (tools)
C_BG        = "#FAFAFA"
C_TEXT      = "#2C3E50"
C_WHITE     = "#FFFFFF"
C_LIGHT     = "#ECF0F1"

# ── data ────────────────────────────────────────────────────────
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
    "Adaptive Separation": [
        "find_optimal_separation_conditions",
        "analyze_selective_solubility_enhanced",
    ],
}

SUBAGENTS = {
    "separation-engineer": {
        "color": "#E74C3C",
        "groups": {
            "Sequence Planning": [
                "find_optimal_separation_sequence",
                "plan_sequential_separation",
                "analyze_integrated_separation",
                "view_alternative_separation_sequence",
            ],
            "Analysis": [
                "optimize_separation_temperature",
                "calculate_selectivity_detailed",
                "rank_solvents_for_separation",
                "build_compatibility_matrix",
                "find_challenging_polymer_pairs",
            ],
            "Visualization": [
                "create_separation_tree_plot",
                "create_selectivity_heatmap",
                "create_process_flow_diagram",
            ],
            "Precipitation": [
                "find_differential_precipitation_solvents",
                "analyze_multi_polymer_precipitation",
                "analyze_precipitation_temperature",
                "plot_precipitation_curves",
                "plot_atmospheric_feasibility",
                "compare_polymer_pairs_precipitation",
                "check_atmospheric_feasibility",
                "check_multi_polymer_atmospheric_feasibility",
            ],
            "Antisolvent": [
                "find_antisolvents",
                "find_antisolvent_pairs",
                "analyze_selective_antisolvent_precipitation",
            ],
        },
    },
    "safety-analyst": {
        "color": "#9B59B6",
        "groups": {
            "GSK Safety": [
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
    "tea-lca-analyst": {
        "color": "#1ABC9C",
        "groups": {
            "TEA / LCA": [
                "analyze_solvent_recovery_tea",
                "analyze_solvent_recovery_lca",
                "compare_solvents_tea_lca",
                "generate_tea_lca_visualizations",
            ],
            "STRAP Process": [
                "analyze_strap_process",
                "calculate_strap_msp",
                "compare_strap_scenarios",
                "generate_strap_visualizations",
            ],
        },
    },
    "scholar-researcher": {
        "color": "#F39C12",
        "groups": {
            "Search": [
                "search_google_scholar",
                "search_web_of_science",
            ],
        },
    },
    "patent-researcher": {
        "color": "#D35400",
        "groups": {
            "Search": [
                "search_google_patents",
                "lookup_patent",
            ],
        },
    },
    "rag-analyst": {
        "color": "#2980B9",
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
        "color": "#8E44AD",
        "groups": {
            "Plots": [
                "plot_solubility_vs_temperature",
                "plot_solubility_vs_temperature_interactive",
                "plot_selectivity_heatmap",
                "plot_multi_panel_analysis",
                "plot_comparison_dashboard",
                "plot_solvent_properties",
            ],
        },
    },
    "statistics-ml": {
        "color": "#16A085",
        "groups": {
            "Statistics": [
                "statistical_summary",
                "correlation_analysis",
                "compare_groups_statistically",
                "regression_analysis",
            ],
            "ML": [
                "predict_solubility_ml",
            ],
        },
    },
}


def _pretty(name: str) -> str:
    """Convert snake_case tool name to readable form."""
    return name.replace("_", " ").replace("  ", " ")


# ════════════════════════════════════════════════════════════════
# Figure 1 — Agent-level architecture
# ════════════════════════════════════════════════════════════════
def generate_agent_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── Title ───────────────────────────────────────────────────
    ax.text(10, 13.5, "DISSOLVE Agent Architecture",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=C_MAIN, fontfamily="sans-serif")

    # ── Main agent box ──────────────────────────────────────────
    main_box = FancyBboxPatch((6.5, 11.0), 7, 1.8,
                              boxstyle="round,pad=0.2",
                              facecolor=C_MAIN, edgecolor=C_MAIN, linewidth=2)
    ax.add_patch(main_box)
    ax.text(10, 12.15, "DISSOLVE Main Agent", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_WHITE)
    ax.text(10, 11.65, "(Claude Sonnet 4.5)", ha="center", va="center",
            fontsize=10, color="#BDC3C7")

    # ── Core tools (left side) ──────────────────────────────────
    core_x, core_y = 1.5, 11.0
    core_box = FancyBboxPatch((0.3, 10.2), 5.2, 2.6,
                              boxstyle="round,pad=0.15",
                              facecolor=C_CORE, edgecolor=C_CORE,
                              linewidth=2, alpha=0.15)
    ax.add_patch(core_box)
    ax.text(2.9, 12.5, "Core Tools (always loaded)", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_CORE)

    core_names = []
    for group, tools in CORE_TOOLS.items():
        core_names.append(f"{group} ({len(tools)})")

    for i, name in enumerate(core_names):
        y = 12.05 - i * 0.42
        ax.text(2.9, y, name, ha="center", va="center",
                fontsize=9, color=C_TEXT,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C_WHITE,
                          edgecolor=C_CORE, linewidth=1, alpha=0.9))

    # Arrow from core to main
    ax.annotate("", xy=(6.5, 11.9), xytext=(5.5, 11.9),
                arrowprops=dict(arrowstyle="->", color=C_CORE, lw=2))

    # ── Subagent boxes (below main) ────────────────────────────
    sub_names = list(SUBAGENTS.keys())
    n_subs = len(sub_names)
    total_width = 18.5
    margin = 0.75
    box_width = (total_width - margin * (n_subs - 1)) / n_subs
    start_x = (20 - total_width) / 2

    for i, name in enumerate(sub_names):
        info = SUBAGENTS[name]
        color = info["color"]
        n_tools = sum(len(t) for t in info["groups"].values())
        group_names = list(info["groups"].keys())

        bx = start_x + i * (box_width + margin)
        by = 2.0
        bh = 7.5

        # Subagent box
        sub_box = FancyBboxPatch((bx, by), box_width, bh,
                                 boxstyle="round,pad=0.15",
                                 facecolor=color, edgecolor=color,
                                 linewidth=2, alpha=0.12)
        ax.add_patch(sub_box)

        # Header
        header_box = FancyBboxPatch((bx + 0.05, by + bh - 1.2), box_width - 0.1, 1.1,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=color, linewidth=1.5)
        ax.add_patch(header_box)

        # Name (split long names)
        display_name = name.replace("-", "\n")
        ax.text(bx + box_width / 2, by + bh - 0.45, display_name,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=C_WHITE, linespacing=1.1)
        ax.text(bx + box_width / 2, by + bh - 1.05,
                f"{n_tools} tools", ha="center", va="center",
                fontsize=7, color="#FDEBD0")

        # Group labels
        y_offset = by + bh - 1.55
        for gname in group_names:
            n = len(info["groups"][gname])
            if y_offset < by + 0.2:
                break
            ax.text(bx + box_width / 2, y_offset,
                    f"{gname} ({n})", ha="center", va="center",
                    fontsize=6.5, color=C_TEXT,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=C_WHITE,
                              edgecolor=color, linewidth=0.8, alpha=0.85))
            y_offset -= 0.55

        # Arrow from main to subagent
        ax.annotate("", xy=(bx + box_width / 2, by + bh),
                    xytext=(10, 11.0),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.5, alpha=0.5,
                                    connectionstyle="arc3,rad=0.0"))

    # ── Legend ──────────────────────────────────────────────────
    ax.text(10, 0.9, "Arrows indicate delegation paths  |  "
            "Core tools are always available to the main agent  |  "
            "Subagents are spawned on demand",
            ha="center", va="center", fontsize=8, color="#7F8C8D",
            style="italic")

    total_tools = sum(len(t) for g in CORE_TOOLS.values() for t in [g])
    total_tools += sum(
        sum(len(t) for t in info["groups"].values())
        for info in SUBAGENTS.values()
    )
    ax.text(10, 0.4, f"Total: {total_tools} tools across "
            f"{n_subs} specialist subagents + core",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C_MAIN)

    path = os.path.join(OUT_DIR, "agent_architecture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {path}")


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
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=C_WHITE,
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
        # Put in shortest column
        min_col = col_heights.index(min(col_heights))
        col_assignments[min_col].append((name, info, h))
        col_heights[min_col] += h + 1.5  # gap between sections

    max_height = max(col_heights) + 2
    fig_width = 28
    col_width = fig_width / 3

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, max_height * 0.58))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(-max_height, 2)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.text(fig_width / 2, 1.3, "DISSOLVE — Subagent Tool Trees",
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
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════
# Figure 3 — Individual subagent PNGs
# ════════════════════════════════════════════════════════════════
def generate_individual_subagent_diagrams():
    """Generate one PNG per subagent (+ core), showing the subagent and its tools."""
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

        # Compute figure dimensions based on content
        fig_width = 10
        content_height = 1.8 + n_groups * 0.65 + n_tools * 0.52 + n_groups * 0.35
        fig_height = max(content_height * 0.55 + 1.5, 4)

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.set_xlim(0, fig_width)
        ax.set_ylim(-content_height - 0.5, 2.5)
        ax.axis("off")
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_BG)

        # Title
        display_title = section_name.replace("-", " ").title()
        ax.text(fig_width / 2, 2.0, display_title,
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=C_WHITE, zorder=5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color,
                          edgecolor=color, linewidth=2.5))
        ax.text(fig_width / 2, 1.3, f"{n_tools} tools across {n_groups} group{'s' if n_groups != 1 else ''}",
                ha="center", va="center", fontsize=10, color="#7F8C8D")

        # Draw groups and tools
        root_x = fig_width / 2
        group_x = fig_width * 0.35
        tool_x = 1.0
        y = 0.3

        for group_name, tools in groups.items():
            group_y = y
            # Line from root to group
            ax.plot([root_x, group_x], [1.0, group_y],
                    color=color, lw=2, alpha=0.4, zorder=1)
            # Group label
            ax.text(group_x, group_y, f"  {group_name}  ",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=C_TEXT, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                              edgecolor=color, linewidth=1.5, alpha=0.18))
            y -= 0.7

            for tool in tools:
                # Line from group to tool
                ax.plot([group_x - 0.5, tool_x + 0.1], [group_y, y],
                        color=C_TOOL, lw=0.8, alpha=0.3, zorder=1)
                # Tool label
                ax.text(tool_x, y, _pretty(tool),
                        ha="left", va="center", fontsize=9.5, color=C_TEXT,
                        fontfamily="monospace", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=C_WHITE,
                                  edgecolor=C_TOOL, linewidth=0.8, alpha=0.9))
                y -= 0.55
            y -= 0.4

        plt.tight_layout()
        # File name: snake_case of section name
        fname = section_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        fname = fname.replace("__", "_").strip("_")
        path = os.path.join(OUT_DIR, f"{fname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    generate_agent_architecture()
    generate_tool_trees()
    generate_individual_subagent_diagrams()
    print("Done.")
