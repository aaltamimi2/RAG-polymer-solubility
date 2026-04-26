"""Visualization tools for separation workflows."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from strap.analysis import PolymerCompatibilityMatrix
from strap.database import get_connection
from strap.services.advanced_separation_service import (
    build_process_flow_report,
    build_selectivity_heatmap_report,
    build_separation_tree_report,
    parse_polymer_list,
    parse_solvent_list,
    plot_separation_sequence as _plot_separation_sequence,
    plot_topk_comparison as _plot_topk_comparison,
    run_async,
    score_separation_sequences,
)
from strap.services.tool_response_service import json_tool_error, json_tool_success
from strap.services.visualization_service import get_plot_url as _get_plot_url
from strap.tools._helpers import get_cross_database_properties, get_plots_dir, safe_tool_wrapper, save_plot

logger = logging.getLogger(__name__)


def _visualization_error(
    tool_name: str,
    message: str,
    *,
    error_code: str = "invalid_input",
    **data: Any,
) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)


_advanced_error = _visualization_error

try:
    from strap.engines.separation import find_best_separation
except Exception as exc:  # noqa: BLE001
    logger.warning("strap.engines.separation unavailable: %s", exc)
    find_best_separation = None

try:
    from strap.engines.visualization import ProcessFlowDiagram, SelectivityHeatmap
except Exception as exc:  # noqa: BLE001
    logger.warning("strap.engines.visualization unavailable: %s", exc)
    ProcessFlowDiagram = None
    SelectivityHeatmap = None

@safe_tool_wrapper(structured_output=True)
def create_separation_tree_plot(
    polymers: str,
    temperature: float = 120.0,
    output_dir: str | None = None,
) -> str:
    """Create a decision tree visualization showing the optimal separation path with selectivity at each step.

    Generates two plots:
    1. Rank #1 recommended sequence (flowchart with solvents, selectivities, and color-coding)
    2. Top-k comparison (side-by-side comparison of the best sequences)

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius
        output_dir: Directory where generated PNG files should be saved.

    WHEN TO USE:
    - "Visualize separation options for LDPE, HDPE, PET, PP"
    - "Show decision tree for polymer separation"
    - "Create separation diagram"
    - "Make a diagram of this separation sequence"
    """
    polymer_list = parse_polymer_list(polymers)
    n_polymers = len(polymer_list)
    if n_polymers < 2:
        return _advanced_error(
            "create_separation_tree_plot",
            "Need at least 2 polymers.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
        )

    sequence_scores = score_separation_sequences(polymer_list, temperature=temperature)

    # Plot 1: rank-1 sequence
    rank1_plot = None
    topk_plot = None
    plot_errors: list[str] = []

    try:
        fp1 = _plot_separation_sequence(
            polymer_list, sequence_scores[0], temperature,
            total_sequences=len(sequence_scores), rank=1,
            output_dir=output_dir,
        )
        rank1_plot = fp1
    except Exception as e:
        logger.error("Rank-1 plot error: %s", e, exc_info=True)
        plot_errors.append(f"rank1: {e}")

    # Plot 2: top-k comparison
    if len(sequence_scores) >= 2:
        try:
            fp2 = _plot_topk_comparison(polymer_list, sequence_scores, temperature, output_dir=output_dir)
            topk_plot = fp2
        except Exception as e:
            logger.error("Top-K plot error: %s", e, exc_info=True)
            plot_errors.append(f"topk: {e}")
    best = sequence_scores[0]
    display = build_separation_tree_report(
        polymer_list=polymer_list,
        sequence_scores=sequence_scores,
        temperature=temperature,
        rank1_plot=rank1_plot,
        topk_plot=topk_plot,
        plot_url_builder=_get_plot_url,
    )

    return json_tool_success(
        display,
        tool_name="create_separation_tree_plot",
        polymers=polymer_list,
        temperature=temperature,
        best_sequence=best["sequence"],
        min_selectivity=best["min_selectivity"],
        total_sequences_evaluated=len(sequence_scores),
        rank1_plot=rank1_plot,
        topk_plot=topk_plot,
        plot_paths=[path for path in [rank1_plot, topk_plot] if path],
        plot_errors=plot_errors,
    )


def _mask_label(mask: int, polymers: list[str]) -> str:
    if mask == 0:
        return "done"
    return "\n".join(polymer for idx, polymer in enumerate(polymers) if mask & (1 << idx))


def _edge_color(selectivity: float | None, *, isolation: bool = False) -> str:
    if isolation:
        return "#9aa7a7"
    value = float(selectivity or 0.0)
    if value >= 30:
        return "#2fa866"
    if value >= 10:
        return "#f0a21a"
    if value > 0:
        return "#df7f2a"
    return "#c94f4f"


def _best_step_for_edge(
    sequence_scores: list[dict[str, Any]],
    target: str,
    remaining: list[str],
) -> dict[str, Any] | None:
    remaining_set = set(remaining)
    for sequence in sequence_scores:
        for step in sequence.get("steps", []):
            if step.get("target") == target and set(step.get("remaining", [])) == remaining_set:
                return step
    return None


def _best_solvent_for_step(step: dict[str, Any] | None) -> dict[str, Any]:
    if not step:
        return {}
    solvents = step.get("solvents") or []
    if solvents and isinstance(solvents, list):
        return solvents[0] if isinstance(solvents[0], dict) else {}
    return {}


def _sequence_step_rows(sequence_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in sequence_data.get("steps", []):
        solvent = _best_solvent_for_step(step)
        rows.append(
            {
                "target": step.get("target", "?"),
                "solvent": solvent.get("solvent", "N/A"),
                "selectivity": float(solvent.get("selectivity") or 0.0),
                "target_sol": solvent.get("target_sol"),
                "max_other": solvent.get("max_other"),
            }
        )
    last_polymer = next(
        (polymer for polymer in sequence_data.get("sequence", []) if polymer not in {row["target"] for row in rows}),
        None,
    )
    if last_polymer:
        rows.append({"target": last_polymer, "solvent": "Residue", "selectivity": None})
    return rows


def _score_sequence_objectives(sequence_scores: list[dict[str, Any]], objectives: str = "selectivity") -> dict[str, dict[str, Any]]:
    requested = {item.strip().lower() for item in objectives.split(",") if item.strip()}
    conn = get_connection()
    best_selectivity = max(sequence_scores, key=lambda row: row.get("min_selectivity", 0.0))

    def route_props(sequence_data: dict[str, Any]) -> tuple[float | None, float | None]:
        gscores: list[float] = []
        energies: list[float] = []
        for row in _sequence_step_rows(sequence_data):
            solvent = row.get("solvent")
            if not solvent or solvent == "Residue":
                continue
            props = get_cross_database_properties(str(solvent), conn)
            if props.get("g_score") is not None:
                gscores.append(float(props["g_score"]))
            if props.get("energy") is not None:
                energies.append(float(props["energy"]))
        avg_gscore = sum(gscores) / len(gscores) if gscores else None
        total_energy = sum(energies) if energies else None
        return avg_gscore, total_energy

    enriched: list[tuple[dict[str, Any], float | None, float | None]] = [
        (sequence_data, *route_props(sequence_data)) for sequence_data in sequence_scores
    ]
    green_candidates = [item for item in enriched if item[1] is not None]
    energy_candidates = [item for item in enriched if item[2] is not None]

    objectives = {
        "max_selectivity": {
            "label": "Max selectivity",
            "rule": "highest bottleneck selectivity",
            "sequence": best_selectivity,
            "metric": f"{best_selectivity.get('min_selectivity', 0.0):.1f}% bottleneck",
            "color": "#276fbf",
        }
    }
    if green_candidates and requested.intersection({"green", "greenness", "gscore", "safety"}):
        best_green = max(green_candidates, key=lambda item: item[1] or -1)[0]
        objectives["greenest"] = {
            "label": "Greenness proxy",
            "rule": "highest average GSK G-score",
            "sequence": best_green,
            "metric": f"G {max(green_candidates, key=lambda item: item[1] or -1)[1]:.1f}",
            "color": "#2fa866",
        }
    if energy_candidates and requested.intersection({"energy", "cost", "price", "economic"}):
        best_energy_tuple = min(energy_candidates, key=lambda item: item[2] if item[2] is not None else float("inf"))
        objectives["lowest_energy"] = {
            "label": "Energy proxy",
            "rule": "lowest summed solvent energy",
            "sequence": best_energy_tuple[0],
            "metric": f"{best_energy_tuple[2]:.0f} J/g",
            "color": "#c65f2e",
        }
    return objectives


def _plot_dp_state_map(
    polymer_list: list[str],
    sequence_scores: list[dict[str, Any]],
    temperature: float,
    objective_sequences: dict[str, dict[str, Any]],
    *,
    output_dir: str | None,
    filename: str = "separation_dp_state_map",
) -> str:
    n = len(polymer_list)
    full_mask = (1 << n) - 1
    positions: dict[int, tuple[float, float]] = {}
    x_spacing = 2.45
    y_spacing = 1.45
    for level in range(n + 1):
        masks = sorted(mask for mask in range(1 << n) if mask.bit_count() == level)
        for idx, mask in enumerate(masks):
            positions[mask] = ((idx - (len(masks) - 1) / 2) * x_spacing, level * y_spacing)

    selected_edges: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for objective in objective_sequences.values():
        sequence = objective["sequence"].get("sequence", [])
        current_mask = full_mask
        for target in sequence[:-1]:
            idx = polymer_list.index(target)
            next_mask = current_mask & ~(1 << idx)
            selected_edges.setdefault((current_mask, next_mask), []).append(objective)
            current_mask = next_mask

    fig, ax = plt.subplots(figsize=(10.2, 7.5))
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    ax.set_xlim(min(xs) - 2.2, max(xs) + 2.4)
    ax.set_ylim(min(ys) - 0.95, max(ys) + 1.2)
    ax.text(
        0,
        max(ys) + 0.95,
        f"Dynamic Programming State Map ({', '.join(polymer_list)})",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#253445",
    )
    ax.text(
        0,
        max(ys) + 0.58,
        f"{2 ** n} states | {n * 2 ** (n - 1)} transitions | {math.factorial(n)} complete orderings | {temperature:g} C",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#5b6770",
    )

    for from_mask, (x1, y1) in positions.items():
        if from_mask == 0:
            continue
        for idx, polymer in enumerate(polymer_list):
            if not (from_mask & (1 << idx)):
                continue
            to_mask = from_mask & ~(1 << idx)
            remaining = [polymer_list[j] for j in range(n) if to_mask & (1 << j)]
            step = _best_step_for_edge(sequence_scores, polymer, remaining)
            solvent = _best_solvent_for_step(step)
            selectivity = solvent.get("selectivity")
            isolation = to_mask == 0
            x2, y2 = positions[to_mask]
            edge_key = (from_mask, to_mask)
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=1.25,
                    color=_edge_color(selectivity, isolation=isolation),
                    alpha=0.28 if edge_key not in selected_edges else 0.18,
                    shrinkA=22,
                    shrinkB=22,
                    mutation_scale=11,
                ),
                zorder=1,
            )
            if edge_key in selected_edges:
                for offset_idx, objective in enumerate(selected_edges[edge_key]):
                    offset = (offset_idx - (len(selected_edges[edge_key]) - 1) / 2) * 0.08
                    ax.annotate(
                        "",
                        xy=(x2 + offset, y2),
                        xytext=(x1 + offset, y1),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            lw=3.2,
                            color=objective["color"],
                            alpha=0.92,
                            shrinkA=23,
                            shrinkB=23,
                            mutation_scale=14,
                        ),
                        zorder=4,
                    )

    selected_nodes = {0, full_mask}
    for edge in selected_edges:
        selected_nodes.update(edge)
    for mask, (x, y) in positions.items():
        if mask == full_mask:
            face, edge, text = "#276fbf", "#253445", "white"
        elif mask == 0:
            face, edge, text = "#2fa866", "#253445", "white"
        elif mask in selected_nodes:
            face, edge, text = "#fff0b3", "#f0a21a", "#253445"
        else:
            face, edge, text = "#f1f3f5", "#ced4da", "#6b7880"
        ax.add_patch(plt.Circle((x, y), 0.40, facecolor=face, edgecolor=edge, linewidth=1.5, zorder=8))
        ax.text(x, y, _mask_label(mask, polymer_list), ha="center", va="center", fontsize=7.8, color=text, fontweight="bold", zorder=9)

    objective_handles = [
        Patch(fc=objective["color"], ec="black", label=objective["label"])
        for objective in objective_sequences.values()
    ]
    transition_handles = [
        Patch(fc="#2fa866", ec="black", label="Sel >= 30%"),
        Patch(fc="#f0a21a", ec="black", label="Sel 10-30%"),
        Patch(fc="#df7f2a", ec="black", label="Sel 0-10%"),
        Patch(fc="#9aa7a7", ec="black", label="Isolation"),
    ]
    ax.legend(handles=objective_handles + transition_handles, loc="lower center", ncol=4, fontsize=7.3, frameon=True, bbox_to_anchor=(0.5, -0.05))
    return save_plot(fig, filename, output_dir=output_dir, dpi=300)


def _plot_objective_paths(
    polymer_list: list[str],
    objective_sequences: dict[str, dict[str, Any]],
    *,
    output_dir: str | None,
    filename: str = "separation_objective_paths",
) -> str:
    objectives = list(objective_sequences.values())
    n_rows = len(objectives)
    n_cols = len(polymer_list)
    fig_w = 3.0 + n_cols * 2.2
    fig_h = 1.0 + n_rows * 1.15 + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.text(fig_w / 2, fig_h - 0.25, f"Objective-Optimal Separation Paths ({', '.join(polymer_list)})", ha="center", va="top", fontsize=13, fontweight="bold", color="#253445")
    left = 2.65
    top = fig_h - 1.0
    cell_w = 2.15
    cell_h = 0.92
    for col in range(n_cols):
        ax.text(left + col * cell_w + cell_w / 2, top + 0.22, f"Step {col + 1}" if col < n_cols - 1 else "Residue", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for row_idx, objective in enumerate(objectives):
        y = top - row_idx * 1.12
        sequence_data = objective["sequence"]
        ax.text(2.45, y - cell_h / 2, f"{objective['label']}\n{objective['rule']}\n{objective['metric']}", ha="right", va="center", fontsize=7.2, fontweight="bold", color=objective["color"], linespacing=1.15)
        for col, step in enumerate(_sequence_step_rows(sequence_data)):
            x = left + col * cell_w
            selectivity = step.get("selectivity")
            face = _edge_color(selectivity, isolation=step.get("solvent") == "Residue")
            text_color = "white" if step.get("solvent") == "Residue" or (selectivity is not None and selectivity < 10) else "#253445"
            ax.add_patch(FancyBboxPatch((x + 0.05, y - cell_h + 0.05), cell_w - 0.10, cell_h - 0.10, boxstyle="round,pad=0.025", facecolor=face, edgecolor=objective["color"], linewidth=2.2))
            ax.text(x + cell_w / 2, y - 0.25, step["target"], ha="center", va="center", fontsize=9.4, fontweight="bold", color=text_color)
            solvent = step.get("solvent", "")
            sel = "" if selectivity is None else f"S {selectivity:.1f}%"
            ax.text(x + cell_w / 2, y - 0.50, solvent, ha="center", va="center", fontsize=6.7, color=text_color)
            ax.text(x + cell_w / 2, y - 0.70, sel, ha="center", va="center", fontsize=6.6, color=text_color, fontstyle="italic")
    return save_plot(fig, filename, output_dir=output_dir, dpi=300)


@safe_tool_wrapper(structured_output=True)
def plot_dynamic_programming_separation_options(
    polymers: str,
    temperature: float = 100.0,
    output_dir: str | None = None,
    objectives: str = "selectivity",
    include_sequence_plots: bool = True,
    include_state_map: bool = True,
    include_objective_paths: bool = False,
) -> str:
    """Plot dynamic-programming separation options for an N-polymer feedstock.

    Generates a ranked best-path plot, top-k route comparison, a DP state map
    over all remaining-polymer states, and objective-path cards when requested.

    Use for requests such as "Generate a dynamic-programming separation state
    map", "DP state map", or "plot all possible separation sequences". This is
    a separation/selectivity visualization tool, not a solubility curve plotter.
    """
    polymer_list = parse_polymer_list(polymers)
    if len(polymer_list) < 2:
        return _advanced_error(
            "plot_dynamic_programming_separation_options",
            "Need at least 2 polymers.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
        )
    sequence_scores = score_separation_sequences(polymer_list, temperature=temperature)
    objective_sequences = _score_sequence_objectives(sequence_scores, objectives=objectives)
    plot_paths: list[str] = []
    plot_errors: list[str] = []

    if include_sequence_plots:
        try:
            plot_paths.append(
                _plot_separation_sequence(
                    polymer_list,
                    sequence_scores[0],
                    temperature,
                    total_sequences=len(sequence_scores),
                    rank=1,
                    output_dir=output_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rank-1 DP plot failed")
            plot_errors.append(f"rank1: {exc}")
        try:
            plot_paths.append(_plot_topk_comparison(polymer_list, sequence_scores, temperature, output_dir=output_dir))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Top-k DP plot failed")
            plot_errors.append(f"topk: {exc}")
    if include_state_map:
        try:
            plot_paths.append(_plot_dp_state_map(polymer_list, sequence_scores, temperature, objective_sequences, output_dir=output_dir))
        except Exception as exc:  # noqa: BLE001
            logger.exception("DP state-map plot failed")
            plot_errors.append(f"state_map: {exc}")
    if include_objective_paths:
        try:
            plot_paths.append(_plot_objective_paths(polymer_list, objective_sequences, output_dir=output_dir))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Objective-path plot failed")
            plot_errors.append(f"objective_paths: {exc}")

    best = sequence_scores[0]
    display = [
        "Dynamic-programming separation visualizations created.",
        f"Polymers: {', '.join(polymer_list)}",
        f"Temperature: {temperature:g} C",
        f"Complete orderings evaluated: {len(sequence_scores)}",
        f"Best sequence: {' -> '.join(best['sequence'])}",
        f"Best bottleneck selectivity: {best['min_selectivity']:.2f}%",
        "",
        "Plots:",
    ]
    display.extend(f"- {_get_plot_url(path)}" for path in plot_paths)
    if plot_errors:
        display.append("\nPlot warnings:")
        display.extend(f"- {error}" for error in plot_errors)
    return json_tool_success(
        "\n".join(display),
        tool_name="plot_dynamic_programming_separation_options",
        polymers=polymer_list,
        temperature=temperature,
        total_sequences_evaluated=len(sequence_scores),
        best_sequence=best["sequence"],
        min_selectivity=best["min_selectivity"],
        plot_paths=plot_paths,
        objectives_requested=[item.strip() for item in objectives.split(",") if item.strip()],
        objective_paths=[
            {
                "objective": objective["label"],
                "rule": objective["rule"],
                "sequence": objective["sequence"].get("sequence"),
                "metric": objective["metric"],
            }
            for objective in objective_sequences.values()
        ],
        plot_errors=plot_errors,
    )


@safe_tool_wrapper(structured_output=True)
def create_selectivity_heatmap(
    polymers: str,
    solvents: str = "",
    temperature: float = 100.0,
) -> str:
    """Create a color-coded heatmap of polymer-solvent solubility values.

    Args:
        polymers: Comma-separated list of polymers
        solvents: Comma-separated list of solvents (optional)
        temperature: Temperature in Celsius

    WHEN TO USE:
    - "Create solubility heatmap for these polymers"
    - "Visualize polymer-solvent compatibility"
    """
    polymer_list = parse_polymer_list(polymers)
    solvent_list = parse_solvent_list(solvents)
    conn = get_connection()

    # Build matrix
    matrix_builder = PolymerCompatibilityMatrix(conn)
    matrix = matrix_builder.build_matrix(
        polymers=polymer_list,
        solvents=solvent_list,
        temperature=temperature,
    )

    if not matrix:
        return _advanced_error(
            "create_selectivity_heatmap",
            "No data available to create heatmap.",
            error_code="no_data",
            polymers=polymer_list,
            solvents=solvent_list or [],
            temperature=temperature,
        )

    # Create visualization
    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    from strap.engines.visualization import PlotConfig
    config = PlotConfig(output_dir=plots_dir)
    viz = SelectivityHeatmap(config)
    filepath = viz.create_polymer_solvent_heatmap(matrix)
    display = build_selectivity_heatmap_report(
        filepath=filepath,
        polymer_list=polymer_list,
        solvent_list=solvent_list,
        temperature=temperature,
        matrix=matrix,
    )
    return json_tool_success(
        display,
        tool_name="create_selectivity_heatmap",
        polymers=polymer_list,
        solvents=solvent_list or [],
        temperature=temperature,
        filepath=filepath,
        matrix_rows=len(matrix),
        matrix_solvents=sorted({solvent for row in matrix.values() for solvent in row}),
    )


@safe_tool_wrapper(structured_output=True)
def create_process_flow_diagram(
    polymers: str,
    temperature: float = 120.0,
) -> str:
    """Create a process flow diagram showing feed, separation units, solvents, and product streams.

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius

    WHEN TO USE:
    - "Create PFD for polymer separation process"
    - "Visualize the separation workflow"
    - "Generate process diagram"
    """
    polymer_list = parse_polymer_list(polymers)
    conn = get_connection()

    # Get separation result
    result = run_async(find_best_separation(polymer_list, conn, temperature, "greedy"))

    # Create visualization
    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    from strap.engines.visualization import PlotConfig
    config = PlotConfig(output_dir=plots_dir)
    viz = ProcessFlowDiagram(config)
    filepath = viz.create_flow_diagram(result.best_sequence)
    display = build_process_flow_report(
        filepath=filepath,
        polymer_list=polymer_list,
        result=result,
    )
    return json_tool_success(
        display,
        tool_name="create_process_flow_diagram",
        polymers=polymer_list,
        temperature=temperature,
        filepath=filepath,
        steps=len(result.best_sequence.steps) - 1,
        solvents_used=sorted(result.best_sequence.unique_solvents),
        sequence=[step.target_polymer for step in result.best_sequence.steps],
    )

__all__ = [
    "create_separation_tree_plot",
    "plot_dynamic_programming_separation_options",
    "create_selectivity_heatmap",
    "create_process_flow_diagram",
]
