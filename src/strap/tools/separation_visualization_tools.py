"""Visualization tools for separation workflows."""

from __future__ import annotations

import logging
import os
from typing import Any

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
from strap.tools._helpers import get_plots_dir, safe_tool_wrapper

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

def create_separation_tree_plot(
    polymers: str,
    temperature: float = 120.0,
) -> str:
    """Create a decision tree visualization showing the optimal separation path with selectivity at each step.

    Generates two plots:
    1. Rank #1 recommended sequence (flowchart with solvents, selectivities, and color-coding)
    2. Top-k comparison (side-by-side comparison of the best sequences)

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius

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
        )
        rank1_plot = fp1
    except Exception as e:
        logger.error("Rank-1 plot error: %s", e, exc_info=True)
        plot_errors.append(f"rank1: {e}")

    # Plot 2: top-k comparison
    if len(sequence_scores) >= 2:
        try:
            fp2 = _plot_topk_comparison(polymer_list, sequence_scores, temperature)
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
    "create_selectivity_heatmap",
    "create_process_flow_diagram",
]
