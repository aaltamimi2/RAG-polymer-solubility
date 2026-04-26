"""Advanced separation tools: algorithms, optimization, precipitation, compatibility, planning."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 – available for downstream use

from strap.database import get_connection
from strap.services.advanced_separation_service import (
    build_challenging_pairs_report,
    build_compatibility_matrix_report,
    build_process_flow_report,
    build_selectivity_heatmap_report,
    build_separation_tree_report,
    build_solvent_ranking_report,
    format_optimization_result,
    format_safety_result,
    format_selectivity_metrics,
    format_separation_result,
    format_top_k_results,
    format_top_k_safety_results,
    parse_polymer_list,
    parse_solvent_list,
    plot_separation_sequence as _plot_separation_sequence,
    plot_topk_comparison as _plot_topk_comparison,
    run_async,
    score_separation_sequences,
)
from strap.services.tool_response_service import (
    json_tool_error,
    json_tool_success,
)
from strap.services.visualization_service import (
    PUB_COLORS as _PUB_COLORS,
    PUB_FONTSIZE as _PUB_FONTSIZE,
    apply_pub_style as _apply_pub_style,
    get_plot_url as _get_plot_url,
    get_solvent_table_name as _get_solvent_table_name,
    lookup_solvent_properties as _lookup_solvent_properties,
)
from strap.tools._helpers import (
    safe_tool_wrapper,
    truncate_output,
    save_plot,
    get_plots_dir,
    DataValidator,
    AdaptiveAnalyzer,
    get_cross_database_properties,
    normalize_solvent_name,
)
from strap.tools.precipitation_analysis import (
    analyze_multi_polymer_precipitation,
    analyze_precipitation_temperature,
    analyze_selective_antisolvent_precipitation,
    check_atmospheric_feasibility,
    check_multi_polymer_atmospheric_feasibility,
    compare_polymer_pairs_precipitation,
    find_antisolvent_pairs,
    find_antisolvents,
    find_differential_precipitation_solvents,
    plot_atmospheric_feasibility,
    plot_precipitation_curves,
)
from strap.tools.separation_planning_tools import (
    analyze_integrated_separation,
    plan_multiple_separation_schemes,
    plan_sequential_separation,
    view_alternative_separation_sequence,
)
from strap.tools.separation_visualization_tools import (
    create_process_flow_diagram,
    create_selectivity_heatmap,
    create_separation_tree_plot,
    plot_dynamic_programming_separation_options,
)
from strap.analysis import SelectivityCalculator, SolventRanker, PolymerCompatibilityMatrix
from strap.solvent_registry import ABBREVIATION_MAP as _ABBREVIATION_MAP

logger = logging.getLogger(__name__)


def _advanced_error(
    tool_name: str,
    message: str,
    *,
    error_code: str = "invalid_input",
    **data: Any,
) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)

# ---------------------------------------------------------------------------
# Optional engine imports (installed separately)
# ---------------------------------------------------------------------------
try:
    from strap.engines.separation import (
        GreedySeparator,
        DPSeparator,
        find_best_separation,
    )
except Exception as e:  # noqa: BLE001
    logger.warning("strap.engines.separation unavailable: %s", e)
    GreedySeparator = None
    DPSeparator = None
    find_best_separation = None

try:
    from strap.engines.optimization import TemperatureOptimizer
except Exception as e:  # noqa: BLE001
    logger.warning("strap.engines.optimization unavailable: %s", e)
    TemperatureOptimizer = None

try:
    from strap.engines.visualization import (
        SelectivityHeatmap,
        ProcessFlowDiagram,
    )
except Exception as e:  # noqa: BLE001
    logger.warning("strap.engines.visualization unavailable: %s", e)
    SelectivityHeatmap = None
    ProcessFlowDiagram = None

# ============================================================================
# Separation Algorithm Tools
# ============================================================================

@safe_tool_wrapper(structured_output=True)
def find_optimal_separation_sequence(
    polymers: str,
    temperature: float = 120.0,
    algorithm: str = "auto",
    top_k: int = 1,
    objective: str = "max_min",
    min_selectivity: float = 5.0,
) -> str:
    """Find the optimal order to separate multiple polymers using greedy or dynamic programming.

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,HDPE,PET,PP")
        temperature: Target separation temperature in Celsius (default: 120)
        algorithm: "greedy", "dp", "auto", or "compare" (default: "auto").
            Use "compare" to run greedy vs DP side-by-side.
        top_k: Number of top sequences to return ranked by min selectivity (default: 1).
            Requires DP algorithm (n <= 12). Use top_k=10 for ranked comparison.
        objective: Optimization objective (default: "max_min"):
            - "max_min": maximize bottleneck selectivity (standard)
            - "max_min_safety": maximize bottleneck GSK G-score subject to
              selectivity >= min_selectivity. Use this when safety matters more
              than maximizing selectivity.
        min_selectivity: Selectivity floor for safety objective (default: 5.0).
            Only used when objective="max_min_safety". Lower values allow
            more solvent choices (potentially safer but less selective).

    WHEN TO USE:
    - "What's the best order to separate LDPE, HDPE, PET, and PP?"
    - "Optimize separation sequence for 5 polymers at 100C"
    - "Compare greedy vs optimal separation for multilayer film"
    - "Return top 10 separation sequences for these polymers"
    - "Find the safest separation sequence for these polymers"
    - "Optimize for safety with at least 7% selectivity"
    """
    polymer_list = parse_polymer_list(polymers)
    from strap.solubility import SENSITIVITY_EXTRAPOLATION_MAX_C

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers for separation planning."

    if len(polymer_list) > 12:
        return f"Error: Too many polymers ({len(polymer_list)}). Maximum 12 for computational feasibility."

    if temperature > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return (
            "Error: Temperature exceeds the supported Apelblat sensitivity limit "
            f"of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C."
        )

    conn = get_connection()

    # Compare mode: run greedy vs DP side-by-side
    if algorithm == "compare":
        if len(polymer_list) > 10:
            return "Error: Comparison requires n <= 10 polymers for DP algorithm."

        greedy_result = run_async(find_best_separation(polymer_list, conn, temperature, "greedy"))
        optimal_result = run_async(find_best_separation(polymer_list, conn, temperature, "dp"))

        greedy_seq = " -> ".join(s.target_polymer for s in greedy_result.best_sequence.steps)
        optimal_seq = " -> ".join(s.target_polymer for s in optimal_result.best_sequence.steps)
        same_sequence = greedy_seq == optimal_seq

        output = [
            "# Algorithm Comparison\n",
            f"**Polymers:** {', '.join(polymer_list)}",
            f"**Temperature:** {temperature}C\n",
            "## Greedy Algorithm (Fast)\n",
            f"- Sequence: {greedy_seq}",
            f"- Min Selectivity: {greedy_result.best_sequence.min_selectivity:.1f}%",
            f"- Avg Selectivity: {greedy_result.best_sequence.avg_selectivity:.1f}%",
            f"- Time: {greedy_result.computation_time_ms:.1f}ms\n",
            "## Dynamic Programming (Optimal)\n",
            f"- Sequence: {optimal_seq}",
            f"- Min Selectivity: {optimal_result.best_sequence.min_selectivity:.1f}%",
            f"- Avg Selectivity: {optimal_result.best_sequence.avg_selectivity:.1f}%",
            f"- Time: {optimal_result.computation_time_ms:.1f}ms\n",
            "## Conclusion\n",
        ]

        if same_sequence:
            output.append("The greedy algorithm found the OPTIMAL solution for this polymer set.")
        else:
            improvement = optimal_result.best_sequence.min_selectivity - greedy_result.best_sequence.min_selectivity
            output.append(f"The optimal algorithm improves min selectivity by {improvement:.1f}%.")
            output.append(f"Recommendation: Use {'greedy' if improvement < 2 else 'optimal'} for this case.")

        return "\n".join(output)

    # Safety objective: force DP with safety mode
    if objective == "max_min_safety":
        if len(polymer_list) > 12:
            return "Error: Safety objective requires n <= 12 for DP algorithm."
        result = run_async(find_best_separation(
            polymer_list, conn, temperature, "dp",
            objective="max_min_safety", min_selectivity=min_selectivity,
            top_k=top_k,
        ))
        if top_k > 1:
            return format_top_k_safety_results(result, ", ".join(polymer_list))
        return format_safety_result(result)

    # Force DP when top_k > 1 (beam DP requires the full selectivity cache)
    if top_k > 1:
        if len(polymer_list) > 12:
            return "Error: top_k > 1 requires n <= 12 for DP algorithm."
        result = run_async(find_best_separation(
            polymer_list, conn, temperature, "dp", top_k=top_k,
        ))
        return format_top_k_results(result, ", ".join(polymer_list))

    # Standard mode: run single algorithm
    result = run_async(find_best_separation(polymer_list, conn, temperature, algorithm))
    return format_separation_result(result)


# ============================================================================
# Temperature Optimization Tools
# ============================================================================

@safe_tool_wrapper(structured_output=True)
def optimize_separation_temperature(
    target_polymer: str,
    other_polymers: str,
    solvent: str,
    temp_min: float = 25.0,
    temp_max: float = 180.0,
) -> str:
    """Find the optimal temperature window that maximizes selectivity for separating a target polymer from others.

    Args:
        target_polymer: Polymer to dissolve (e.g., "LDPE")
        other_polymers: Polymers to NOT dissolve, comma-separated (e.g., "HDPE,PP")
        solvent: Solvent to analyze (e.g., "xylene")
        temp_min: Minimum temperature to scan (default: 25)
        temp_max: Maximum temperature to scan (default: 180). Values from 180-200 C are reported as sensitivity-only extrapolation data.

    WHEN TO USE:
    - "What temperature should I use to dissolve LDPE but not HDPE in xylene?"
    - "Find optimal temperature window for PET separation"
    - "Is 100C good enough for PS dissolution in toluene?"
    """
    others = parse_polymer_list(other_polymers)
    conn = get_connection()

    optimizer = TemperatureOptimizer(conn)
    result = run_async(optimizer.find_optimal_temperature(
        target_polymer=target_polymer.strip().upper(),
        other_polymers=others,
        solvent=solvent.strip(),
        temp_range=(temp_min, temp_max),
    ))

    return format_optimization_result(result)



# ============================================================================
# Analysis Tools
# ============================================================================

@safe_tool_wrapper(structured_output=True)
def calculate_selectivity_detailed(
    target_polymer: str,
    other_polymers: str,
    solvent: str,
    temperature: float = 100.0,
) -> str:
    """Calculate detailed selectivity metrics (value, ratio, confidence, viability) for a specific separation.

    Args:
        target_polymer: Polymer to dissolve
        other_polymers: Polymers to NOT dissolve, comma-separated
        solvent: Solvent to use
        temperature: Temperature in Celsius

    WHEN TO USE:
    - "What's the selectivity for dissolving LDPE vs HDPE in xylene at 100C?"
    - "Is cyclohexane selective enough for PET separation?"
    - "Check if toluene works for PS vs LDPE separation"
    """
    others = parse_polymer_list(other_polymers)
    conn = get_connection()

    calc = SelectivityCalculator(conn)
    metrics = calc.calculate(
        target=target_polymer.strip().upper(),
        others=others,
        solvent=solvent.strip(),
        temperature=temperature,
    )

    return format_selectivity_metrics(metrics)


@safe_tool_wrapper(structured_output=True)
def rank_solvents_for_separation(
    target_polymer: str,
    other_polymers: str,
    temperature: float = 100.0,
    top_k: int = 10,
) -> str:
    """Rank solvents by multi-criteria scoring (selectivity, safety, environmental impact, cost).

    Args:
        target_polymer: Polymer to dissolve
        other_polymers: Polymers to NOT dissolve, comma-separated
        temperature: Temperature in Celsius
        top_k: Number of top solvents to return (default: 10)

    WHEN TO USE:
    - "What's the greenest solvent for separating LDPE from HDPE?"
    - "Rank solvents by safety for PET dissolution"
    - "Find cost-effective solvents for PP separation"
    """
    others = parse_polymer_list(other_polymers)
    conn = get_connection()

    calc = SelectivityCalculator(conn)
    ranker = SolventRanker(calc)
    scores = ranker.rank_solvents(
        target=target_polymer.strip().upper(),
        others=others,
        temperature=temperature,
        top_k=top_k,
    )

    if not scores:
        return "No solvents found with data for this polymer combination."
    return build_solvent_ranking_report(
        scores,
        target_polymer=target_polymer,
        other_polymers=others,
        temperature=temperature,
    )


@safe_tool_wrapper(structured_output=True)
def build_compatibility_matrix(
    polymers: str,
    solvents: str = "",
    temperature: float = 100.0,
) -> str:
    """Build a matrix showing solubility of each polymer in each solvent at a given temperature.

    Args:
        polymers: Comma-separated list of polymers
        solvents: Comma-separated list of solvents (optional, auto-detects if empty)
        temperature: Temperature in Celsius

    WHEN TO USE:
    - "Show compatibility matrix for LDPE, HDPE, PET in common solvents"
    - "What solvents work for which polymers?"
    - "Build solubility matrix for 5 polymers"
    """
    polymer_list = parse_polymer_list(polymers)
    solvent_list = parse_solvent_list(solvents)
    conn = get_connection()

    matrix_builder = PolymerCompatibilityMatrix(conn)
    matrix = matrix_builder.build_matrix(
        polymers=polymer_list,
        solvents=solvent_list,
        temperature=temperature,
    )

    if not matrix or not any(matrix.values()):
        return "No compatibility data found for this polymer/solvent combination."
    return build_compatibility_matrix_report(
        matrix,
        polymers=polymer_list,
        temperature=temperature,
    )


@safe_tool_wrapper(structured_output=True)
def find_challenging_polymer_pairs(
    polymers: str,
    temperature: float = 100.0,
    selectivity_threshold: float = 10.0,
) -> str:
    """Identify polymer pairs where the best achievable selectivity is below a threshold.

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius
        selectivity_threshold: Minimum acceptable selectivity (default: 10)

    WHEN TO USE:
    - "Which polymer pairs in this mixture are hard to separate?"
    - "Are any polymers in this set too similar?"
    - "Identify separation challenges for this film composition"
    """
    polymer_list = parse_polymer_list(polymers)
    conn = get_connection()

    matrix_builder = PolymerCompatibilityMatrix(conn)
    pairs = matrix_builder.find_challenging_pairs(
        polymers=polymer_list,
        temperature=temperature,
        threshold=selectivity_threshold,
    )
    return build_challenging_pairs_report(
        pairs,
        polymers=polymer_list,
        temperature=temperature,
        selectivity_threshold=selectivity_threshold,
    )


# ============================================================================
# Introspection Tool
# ============================================================================

@safe_tool_wrapper(structured_output=True)
def get_supported_polymers_and_solvents() -> str:
    """List all polymers and their available solvents in the interpolation coefficient database.

    Returns a formatted catalogue of every polymer-solvent pair that has fitted
    ln(S) = A + B/T + C·ln(T) coefficients (modified Apelblat), along with the valid temperature range.
    Use this before running separation or solubility tools to confirm that the
    polymer or solvent of interest is supported — tools silently return null/zero
    for unsupported pairs.

    WHEN TO USE:
    - "Which polymers are supported?"
    - "Does the database have data for PET with toluene?"
    - "What solvents are available for HDPE?"
    - "List all supported polymer-solvent combinations"
    """
    from strap.solubility import (
        FITTED_TEMP_MAX_C,
        FITTED_TEMP_MIN_C,
        RECOMMENDED_EXTRAPOLATION_MAX_C,
        SENSITIVITY_EXTRAPOLATION_MAX_C,
        _load_coefficients,
        _get_known_names,
    )

    _, lookup = _load_coefficients()
    known_polymers, _ = _get_known_names(lookup)

    # Build a dict: polymer -> sorted list of solvents (fitted entries only)
    polymer_solvents: dict[str, list[str]] = {}
    for (polymer, solvent), entry in lookup.items():
        if entry.get("category") == "fitted":
            polymer_solvents.setdefault(polymer, []).append(solvent)

    if not polymer_solvents:
        return _advanced_error(
            "get_supported_polymers_and_solvents",
            "No fitted interpolation coefficients found in the database.",
            error_code="no_supported_pairs",
        )

    output = [
        "# Interpolation Coefficient Database — Supported Polymers & Solvents\n",
        f"**Fitted temperature range:** {FITTED_TEMP_MIN_C:.0f}–{FITTED_TEMP_MAX_C:.0f} °C\n",
        f"**Exploratory extrapolation:** up to {RECOMMENDED_EXTRAPOLATION_MAX_C:.0f} °C as lower-confidence Apelblat estimates when explicitly requested\n",
        f"**Sensitivity-only extrapolation:** {RECOMMENDED_EXTRAPOLATION_MAX_C:.0f}–{SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} °C for high-boiling-solvent screening only\n",
        f"**Total polymers:** {len(polymer_solvents)}\n",
    ]

    for polymer in sorted(polymer_solvents):
        solvents = sorted(polymer_solvents[polymer])
        output.append(f"\n## {polymer} ({len(solvents)} solvents)")
        output.append(", ".join(solvents))

    output.append(
        "\n\n---\n"
        "**Note:** Names are case-normalised (polymers UPPER, solvents lower). "
        "Common aliases (e.g. POLYSTYRENE→PS, PA6→NYLON6) are resolved automatically."
    )
    return json_tool_success(
        "\n".join(output),
        tool_name="get_supported_polymers_and_solvents",
        polymers=sorted(polymer_solvents.keys()),
        polymer_count=len(polymer_solvents),
        supported_pairs=sum(len(solvents) for solvents in polymer_solvents.values()),
        fitted_temperature_range_c=[FITTED_TEMP_MIN_C, FITTED_TEMP_MAX_C],
        recommended_extrapolation_max_c=RECOMMENDED_EXTRAPOLATION_MAX_C,
        sensitivity_extrapolation_max_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        polymer_solvents={polymer: sorted(solvents) for polymer, solvents in polymer_solvents.items()},
    )


# ============================================================================
# Tool Collection
# ============================================================================

ADVANCED_SEPARATION_TOOLS = [
    find_optimal_separation_sequence,
    optimize_separation_temperature,
    calculate_selectivity_detailed,
    rank_solvents_for_separation,
    build_compatibility_matrix,
    find_challenging_polymer_pairs,
    create_separation_tree_plot,
    plot_dynamic_programming_separation_options,
    create_selectivity_heatmap,
    create_process_flow_diagram,
    # Differential precipitation tools
    find_differential_precipitation_solvents,
    analyze_multi_polymer_precipitation,
    analyze_precipitation_temperature,
    plot_precipitation_curves,
    plot_atmospheric_feasibility,
    compare_polymer_pairs_precipitation,
    check_atmospheric_feasibility,
    check_multi_polymer_atmospheric_feasibility,
    # Antisolvent precipitation tools
    find_antisolvents,
    find_antisolvent_pairs,
    analyze_selective_antisolvent_precipitation,
    # Separation planning tools (merged from separation_planning.py)
    plan_sequential_separation,
    analyze_integrated_separation,
    view_alternative_separation_sequence,
    # Introspection tool
    get_supported_polymers_and_solvents,
]

__all__ = [
    "find_optimal_separation_sequence",
    "optimize_separation_temperature",
    "calculate_selectivity_detailed",
    "rank_solvents_for_separation",
    "build_compatibility_matrix",
    "find_challenging_polymer_pairs",
    "create_separation_tree_plot",
    "plot_dynamic_programming_separation_options",
    "create_selectivity_heatmap",
    "create_process_flow_diagram",
    "find_differential_precipitation_solvents",
    "analyze_multi_polymer_precipitation",
    "analyze_precipitation_temperature",
    "plot_precipitation_curves",
    "plot_atmospheric_feasibility",
    "compare_polymer_pairs_precipitation",
    "check_atmospheric_feasibility",
    "check_multi_polymer_atmospheric_feasibility",
    "find_antisolvents",
    "find_antisolvent_pairs",
    "analyze_selective_antisolvent_precipitation",
    "plan_sequential_separation",
    "analyze_integrated_separation",
    "view_alternative_separation_sequence",
    "get_supported_polymers_and_solvents",
    "ADVANCED_SEPARATION_TOOLS",
]
