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
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 – available for downstream use

from strap.database import get_connection
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
from strap.analysis import SelectivityCalculator, SolventRanker, PolymerCompatibilityMatrix
from strap.tools.visualization import _apply_pub_style, _PUB_COLORS, _PUB_FONTSIZE

logger = logging.getLogger(__name__)

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
    from strap.engines.precipitation import PrecipitationAnalyzer
except Exception as e:  # noqa: BLE001
    logger.warning("strap.engines.precipitation unavailable: %s", e)
    PrecipitationAnalyzer = None

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
# Helper Functions
# ============================================================================

def parse_polymer_list(polymers: str) -> List[str]:
    """Parse comma-separated polymer string."""
    return [p.strip().upper() for p in polymers.split(',') if p.strip()]


def parse_solvent_list(solvents: str) -> Optional[List[str]]:
    """Parse comma-separated solvent string, or None if empty."""
    if not solvents or not solvents.strip():
        return None
    return [s.strip() for s in solvents.split(',') if s.strip()]


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_separation_result(result) -> str:
    """Format SeparationResult as readable markdown."""
    seq = result.best_sequence

    output = [
        "# Optimal Separation Sequence\n",
        f"**Algorithm:** {result.algorithm}",
        f"**Computation Time:** {result.computation_time_ms:.1f}ms",
        f"**Nodes Explored:** {result.nodes_explored}\n",
    ]

    sequence_str = " -> ".join(s.target_polymer for s in seq.steps)
    output.append(f"**Sequence:** {sequence_str}")
    output.append(f"**Status:** {seq.status.value}")
    output.append(f"**Minimum Selectivity:** {seq.min_selectivity:.1f}%")
    output.append(f"**Average Selectivity:** {seq.avg_selectivity:.1f}%")
    output.append(f"**Unique Solvents:** {len(seq.unique_solvents)}\n")

    output.append("## Step-by-Step Breakdown\n")
    for step in seq.steps:
        if step.remaining_polymers:
            status = "OK" if step.is_viable else "LOW"
            output.append(
                f"**Step {step.step_number}: Separate {step.target_polymer}**\n"
                f"  - Solvent: {step.solvent}\n"
                f"  - Temperature: {step.temperature}C\n"
                f"  - Selectivity: {step.selectivity:.1f}% [{status}]\n"
                f"  - Target Solubility: {step.target_solubility:.1f}%\n"
                f"  - Max Other Solubility: {step.max_other_solubility:.1f}%\n"
                f"  - Remaining: {', '.join(step.remaining_polymers)}\n"
            )
        else:
            output.append(
                f"**Step {step.step_number}: {step.target_polymer} isolated**\n"
            )

    return "\n".join(output)


def format_safety_result(result) -> str:
    """Format safety-optimized SeparationResult as readable markdown."""
    seq = result.best_sequence

    output = [
        "# Safety-Optimized Separation Sequence\n",
        f"**Algorithm:** {result.algorithm}",
        f"**Computation Time:** {result.computation_time_ms:.1f}ms",
        f"**Nodes Explored:** {result.nodes_explored}\n",
    ]

    sequence_str = " -> ".join(s.target_polymer for s in seq.steps)
    output.append(f"**Sequence:** {sequence_str}")
    output.append(f"**Status:** {seq.status.value}")
    output.append(f"**Minimum Selectivity:** {seq.min_selectivity:.1f}%")
    output.append(f"**Average Selectivity:** {seq.avg_selectivity:.1f}%")

    safety_scores = [s.safety_score for s in seq.steps if s.safety_score is not None]
    if safety_scores:
        output.append(f"**Min G-Score:** {min(safety_scores):.1f}/10")
        output.append(f"**Avg G-Score:** {sum(safety_scores)/len(safety_scores):.1f}/10")
    output.append("")

    output.append("## Step-by-Step Breakdown\n")
    for step in seq.steps:
        if step.remaining_polymers:
            status = "OK" if step.is_viable else "LOW"
            gs_str = f"G-Score: {step.safety_score:.1f}/10" if step.safety_score else "G-Score: N/A"
            output.append(
                f"**Step {step.step_number}: Separate {step.target_polymer}**\n"
                f"  - Solvent: {step.solvent}\n"
                f"  - Temperature: {step.temperature}C\n"
                f"  - Selectivity: {step.selectivity:.1f}% [{status}]\n"
                f"  - {gs_str}\n"
                f"  - Remaining: {', '.join(step.remaining_polymers)}\n"
            )
        else:
            output.append(
                f"**Step {step.step_number}: {step.target_polymer} isolated**\n"
            )

    return "\n".join(output)


def format_top_k_safety_results(result, polymers_str: str) -> str:
    """Format top-K safety-optimized sequences as a ranked markdown table."""
    sequences = result.all_sequences
    if not sequences:
        return "No sequences found."

    output = [
        "# Top Safety-Optimized Sequences\n",
        f"**Polymers:** {polymers_str}",
        f"**Algorithm:** {result.algorithm}",
        f"**Sequences found:** {len(sequences)}",
        f"**Computation time:** {result.computation_time_ms:.1f}ms\n",
        "| Rank | Sequence | Min Sel (%) | Min G-Score | Bottleneck |",
        "|------|----------|-------------|-------------|------------|",
    ]

    for i, seq in enumerate(sequences, 1):
        seq_str = " \u2192 ".join(s.target_polymer for s in seq.steps)
        real_steps = [s for s in seq.steps if s.remaining_polymers]
        safety_scores = [s.safety_score for s in real_steps if s.safety_score is not None]
        min_gs = min(safety_scores) if safety_scores else 0.0
        if real_steps:
            bottleneck = min(
                (s for s in real_steps if s.safety_score is not None),
                key=lambda s: s.safety_score,
                default=real_steps[0],
            )
            bn_str = f"{bottleneck.target_polymer} (G:{bottleneck.safety_score:.1f})"
        else:
            bn_str = "N/A"
        output.append(
            f"| {i} | {seq_str} | {seq.min_selectivity:.1f} | "
            f"{min_gs:.1f} | {bn_str} |"
        )

    # Detail on rank 1
    best = sequences[0]
    output.append(f"\n## Rank 1 Detail\n")
    output.append(format_safety_result(result))

    return "\n".join(output)


def format_top_k_results(result, polymers_str: str) -> str:
    """Format top-K separation sequences as a ranked markdown table."""
    sequences = result.all_sequences
    if not sequences:
        return "No sequences found."

    output = [
        "# Top Separation Sequences\n",
        f"**Polymers:** {polymers_str}",
        f"**Algorithm:** {result.algorithm}",
        f"**Sequences found:** {len(sequences)}",
        f"**Computation time:** {result.computation_time_ms:.1f}ms\n",
        "| Rank | Sequence | Min Sel (%) | Avg Sel (%) | Bottleneck |",
        "|------|----------|-------------|-------------|------------|",
    ]

    for i, seq in enumerate(sequences, 1):
        seq_str = " \u2192 ".join(s.target_polymer for s in seq.steps)
        # Find bottleneck step (lowest selectivity, excluding last)
        real_steps = [s for s in seq.steps if s.remaining_polymers]
        if real_steps:
            bottleneck = min(real_steps, key=lambda s: s.selectivity)
            bn_str = f"{bottleneck.target_polymer} ({bottleneck.selectivity:.1f}%)"
        else:
            bn_str = "N/A"
        output.append(
            f"| {i} | {seq_str} | {seq.min_selectivity:.1f} | "
            f"{seq.avg_selectivity:.1f} | {bn_str} |"
        )

    # Detail on rank 1
    best = sequences[0]
    output.append(f"\n## Rank 1 Detail\n")
    output.append(format_separation_result(result))

    return "\n".join(output)


def format_optimization_result(result) -> str:
    """Format OptimizationResult as readable markdown."""
    output = [
        "# Temperature Optimization Result\n",
        f"**Optimal Temperature:** {result.optimal_temperature}C",
        f"**Overall Selectivity:** {result.overall_selectivity:.1f}%",
        f"**Energy Score:** {result.energy_score:.2f} (lower is better)",
        f"**Feasibility Score:** {result.feasibility_score:.1%}\n",
    ]

    if result.temperature_windows:
        output.append("## Viable Temperature Windows\n")
        for w in result.temperature_windows:
            output.append(
                f"- {w.temp_min:.0f}C - {w.temp_max:.0f}C "
                f"(best: {w.optimal_temp:.0f}C, selectivity: {w.selectivity_at_optimal:.1f}%)"
            )
        output.append("")

    if result.recommendations:
        output.append("## Recommendations\n")
        for rec in result.recommendations:
            output.append(f"- {rec}")

    return "\n".join(output)


def format_selectivity_metrics(metrics) -> str:
    """Format SelectivityMetrics as readable markdown."""
    status = "VIABLE" if metrics.is_viable else "NOT VIABLE"

    output = [
        "# Selectivity Analysis\n",
        f"**Target Polymer:** {metrics.target_polymer}",
        f"**Other Polymers:** {', '.join(metrics.other_polymers)}",
        f"**Solvent:** {metrics.solvent}",
        f"**Temperature:** {metrics.temperature}C\n",
        "## Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Selectivity | {metrics.selectivity:.1f}% |",
        f"| Target Solubility | {metrics.target_solubility:.1f}% |",
        f"| Max Other Solubility | {metrics.max_other_solubility:.1f}% |",
        f"| Selectivity Ratio | {metrics.selectivity_ratio:.2f}x |",
        f"| Data Confidence | {metrics.confidence:.1%} |",
        f"| Status | **{status}** |",
    ]

    return "\n".join(output)


# ============================================================================
# Separation Algorithm Tools
# ============================================================================

@safe_tool_wrapper
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

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers for separation planning."

    if len(polymer_list) > 12:
        return f"Error: Too many polymers ({len(polymer_list)}). Maximum 12 for computational feasibility."

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

@safe_tool_wrapper
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
        temp_max: Maximum temperature to scan (default: 180)

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

@safe_tool_wrapper
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


@safe_tool_wrapper
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

    output = [
        "# Solvent Ranking\n",
        f"**Target:** {target_polymer.upper()}",
        f"**Separate from:** {', '.join(others)}",
        f"**Temperature:** {temperature}C\n",
        "## Top Solvents\n",
        "| Rank | Solvent | Overall | Selectivity | BP | LogP | Cp | Energy |",
        "|------|---------|---------|-------------|-----|------|-----|--------|",
    ]

    for i, score in enumerate(scores, 1):
        output.append(
            f"| {i} | {score.solvent} | {score.overall_score:.2f} | "
            f"{score.selectivity_score:.2f} | {score.bp_score:.2f} | "
            f"{score.logp_score:.2f} | {score.cp_score:.2f} | {score.energy_score:.2f} |"
        )

    # Add notes for top solvent
    if scores and scores[0].notes:
        output.append(f"\n**Notes for {scores[0].solvent}:**")
        for note in scores[0].notes:
            output.append(f"- {note}")

    return "\n".join(output)


@safe_tool_wrapper
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

    # Get all solvents
    all_solvents = set()
    for sols in matrix.values():
        all_solvents.update(sols.keys())
    all_solvents = sorted(all_solvents)[:15]  # Limit columns

    # Build table
    output = [
        "# Polymer-Solvent Compatibility Matrix\n",
        f"**Temperature:** {temperature}C",
        f"**Polymers:** {len(polymer_list)}",
        f"**Solvents:** {len(all_solvents)}\n",
    ]

    # Header
    header = "| Polymer | " + " | ".join(s[:8] for s in all_solvents) + " |"
    separator = "|---------|" + "|".join("-" * 8 for _ in all_solvents) + "|"
    output.append(header)
    output.append(separator)

    # Rows
    for polymer in polymer_list:
        row = f"| {polymer} |"
        for solvent in all_solvents:
            sol = matrix.get(polymer, {}).get(solvent)
            if sol is not None:
                row += f" {sol:5.1f}% |"
            else:
                row += "   -   |"
        output.append(row)

    output.append("\n*Values are solubility percentages. Higher = more soluble.*")

    return "\n".join(output)


@safe_tool_wrapper
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

    output = [
        "# Challenging Polymer Pairs\n",
        f"**Polymers:** {', '.join(polymer_list)}",
        f"**Temperature:** {temperature}C",
        f"**Threshold:** {selectivity_threshold}% selectivity\n",
    ]

    if not pairs:
        output.append("No challenging pairs found. All polymer pairs can be separated with selectivity above threshold.")
    else:
        output.append("## Difficult Pairs\n")
        output.append("| Polymer 1 | Polymer 2 | Best Selectivity |")
        output.append("|-----------|-----------|------------------|")
        for p1, p2, sel in pairs:
            warning = " (CRITICAL)" if sel < 5 else ""
            output.append(f"| {p1} | {p2} | {sel:.1f}%{warning} |")

        output.append(f"\n**{len(pairs)} challenging pair(s) identified.**")
        output.append("Consider alternative temperatures or solvents for these pairs.")

    return "\n".join(output)


# ============================================================================
# Visualization Tools
# ============================================================================

@safe_tool_wrapper
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
    from itertools import permutations
    from strap.solubility import get_all_solvents_selectivity as _get_all_sel

    polymer_list = parse_polymer_list(polymers)
    n_polymers = len(polymer_list)
    if n_polymers < 2:
        return "Error: Need at least 2 polymers."

    MAX_EXHAUSTIVE = 6  # 6! = 720 permutations

    def _find_top_solvents(target: str, remaining: list, k: int = 3) -> list:
        if not remaining:
            return [{"solvent": "N/A", "selectivity": float("inf"), "target_sol": 100, "max_other": 0}]
        all_sel = _get_all_sel(target, remaining, temperature)
        if not all_sel:
            return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]
        return [
            {"solvent": e["solvent"], "selectivity": e["selectivity"],
             "target_sol": e["target_sol"], "max_other": e["max_other_sol"]}
            for e in all_sel[:k]
        ]

    def _analyze_sequence(sequence):
        steps = []
        total_min_sel = float("inf")
        for step_idx, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step_idx:])
            top_solvents = _find_top_solvents(target, remaining)
            steps.append({"step": step_idx, "target": target, "remaining": remaining, "solvents": top_solvents})
            if top_solvents and top_solvents[0]["selectivity"] < total_min_sel:
                total_min_sel = top_solvents[0]["selectivity"]
        return {"sequence": list(sequence), "min_selectivity": total_min_sel, "steps": steps}

    # Build ranked sequences
    if n_polymers <= MAX_EXHAUSTIVE:
        sequence_scores = sorted(
            [_analyze_sequence(seq) for seq in permutations(polymer_list)],
            key=lambda x: x["min_selectivity"], reverse=True,
        )
    else:
        # Greedy for large n
        remaining = list(polymer_list)
        greedy_seq, greedy_steps = [], []
        while len(remaining) > 1:
            best_cand, best_val = None, -float("inf")
            for target in remaining:
                others = [p for p in remaining if p != target]
                solvents = _find_top_solvents(target, others)
                top_sel = solvents[0]["selectivity"] if solvents else 0
                if top_sel > best_val:
                    best_val = top_sel
                    best_cand = (target, solvents)
            target, solvents = best_cand
            greedy_seq.append(target)
            remaining.remove(target)
            greedy_steps.append({"step": len(greedy_seq), "target": target,
                                 "remaining": remaining.copy(), "solvents": solvents})
        greedy_seq.append(remaining[0])
        min_sel = min(s["solvents"][0]["selectivity"] for s in greedy_steps) if greedy_steps else 0
        sequence_scores = [{"sequence": greedy_seq, "min_selectivity": min_sel, "steps": greedy_steps}]

    output = ["# Separation Tree Visualization\n"]

    # Plot 1: rank-1 sequence
    try:
        fp1 = _plot_separation_sequence(
            polymer_list, sequence_scores[0], temperature,
            total_sequences=len(sequence_scores), rank=1,
        )
        output.append(f"**Rank #1 sequence:** {_get_plot_url(fp1)}\n")
    except Exception as e:
        logger.error("Rank-1 plot error: %s", e, exc_info=True)
        output.append(f"Could not create rank-1 plot: {e}\n")

    # Plot 2: top-k comparison
    if len(sequence_scores) >= 2:
        try:
            fp2 = _plot_topk_comparison(polymer_list, sequence_scores, temperature)
            output.append(f"**Top-K comparison:** {_get_plot_url(fp2)}\n")
        except Exception as e:
            logger.error("Top-K plot error: %s", e, exc_info=True)
            output.append(f"Could not create top-K plot: {e}\n")

    # Text summary
    best = sequence_scores[0]
    output.append(f"**Best sequence:** {' -> '.join(best['sequence'])}")
    output.append(f"**Min Selectivity:** {best['min_selectivity']:.1f}%")
    if len(sequence_scores) > 1:
        output.append(f"**Total sequences evaluated:** {len(sequence_scores)}")

    return "\n".join(output)


@safe_tool_wrapper
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
        return "No data available to create heatmap."

    # Create visualization
    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    from strap.engines.visualization import PlotConfig
    config = PlotConfig(output_dir=plots_dir)
    viz = SelectivityHeatmap(config)
    filepath = viz.create_polymer_solvent_heatmap(matrix)

    return f"# Selectivity Heatmap\n\n**Plot saved to:** `{filepath}`"


@safe_tool_wrapper
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

    output = [
        "# Process Flow Diagram\n",
        f"**Plot saved to:** `{filepath}`\n",
        "## Process Summary\n",
        f"- **Feed:** {', '.join(polymer_list)}",
        f"- **Steps:** {len(result.best_sequence.steps) - 1}",
        f"- **Solvents Used:** {', '.join(result.best_sequence.unique_solvents)}",
    ]

    return "\n".join(output)


# ============================================================================
# Differential Precipitation Tools
# ============================================================================

@safe_tool_wrapper
def find_differential_precipitation_solvents(
    polymer_to_precipitate: str,
    polymer_to_retain: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    top_k: int = 10,
) -> str:
    """Find solvents where one polymer precipitates before another during cooling.

    Args:
        polymer_to_precipitate: Polymer that should precipitate FIRST at higher temperature
        polymer_to_retain: Polymer that should stay dissolved (precipitates later)
        min_temperature_gap: Minimum temperature separation required in Celsius (default: 20)
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 1)
        top_k: Number of top results to return (default: 10)

    WHEN TO USE:
    - "Find a solvent where EVOH precipitates before LDPE"
    - "What solvent gives the best separation window for PP vs PET?"
    - "Find solvents with at least 30C precipitation gap between HDPE and PS"
    """
    conn = get_connection()
    analyzer = PrecipitationAnalyzer(conn)

    from strap.engines.precipitation import format_differential_precipitation_results

    results = analyzer.find_differential_precipitation_solvents(
        polymer_to_precipitate=polymer_to_precipitate,
        polymer_to_retain=polymer_to_retain,
        min_temp_gap=min_temperature_gap,
        precip_threshold=precipitation_threshold,
        top_k=top_k,
    )

    if not results:
        # Try the reverse order
        reverse_results = analyzer.find_differential_precipitation_solvents(
            polymer_to_precipitate=polymer_to_retain,
            polymer_to_retain=polymer_to_precipitate,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            top_k=top_k,
        )
        if reverse_results:
            return (
                f"No solvents found where {polymer_to_precipitate} precipitates before {polymer_to_retain}.\n\n"
                f"However, the REVERSE order works:\n\n"
                + format_differential_precipitation_results(reverse_results)
            )
        return (
            f"No solvents found with {min_temperature_gap} deg C gap for {polymer_to_precipitate}/{polymer_to_retain}.\n"
            f"Try reducing min_temperature_gap or checking polymer names.\n"
            f"Available polymers: {', '.join(analyzer.get_available_polymers())}"
        )

    return format_differential_precipitation_results(results)


@safe_tool_wrapper
def analyze_multi_polymer_precipitation(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Determine the order in which multiple polymers precipitate during cooling in a single solvent.

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,PP,PET,EVOH")
        solvent: Solvent to analyze (e.g., "toluene", "dimethylformamide")
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 10)

    WHEN TO USE:
    - "What's the precipitation order for PP, PS, HDPE in toluene?"
    - "Design a cooling protocol to separate LDPE, PET, and EVOH using DMF"
    - "How do I sequentially recover 4 polymers by cooling?"
    """
    polymer_list = [p.strip() for p in polymers.split(",")]

    conn = get_connection()
    analyzer = PrecipitationAnalyzer(conn)

    from strap.engines.precipitation import format_multi_polymer_sequence

    result = analyzer.analyze_multi_polymer_precipitation(
        polymers=polymer_list,
        solvent=solvent,
        precip_threshold=precipitation_threshold,
    )

    if not result:
        available_solvents = analyzer.get_available_solvents()
        available_polymers = analyzer.get_available_polymers()
        return (
            f"Could not analyze precipitation for {polymers} in {solvent}.\n"
            f"Available solvents: {', '.join(available_solvents[:10])}...\n"
            f"Available polymers: {', '.join(available_polymers)}"
        )

    return format_multi_polymer_sequence(result)


@safe_tool_wrapper
def analyze_precipitation_temperature(
    polymer: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Analyze dissolution/precipitation temperatures and solubility curve for a single polymer-solvent pair.

    Args:
        polymer: Polymer name (e.g., "LDPE", "PET", "EVOH")
        solvent: Solvent name (e.g., "toluene", "dimethylformamide")
        precipitation_threshold: Solubility threshold for precipitation (default: 10%)

    WHEN TO USE:
    - "What's the precipitation temperature of LDPE in toluene?"
    - "At what temperature does PET dissolve in DMF?"
    - "Show me the solubility profile of EVOH in propanone"
    """
    conn = get_connection()
    analyzer = PrecipitationAnalyzer(conn)

    point = analyzer.analyze_precipitation(polymer, solvent, precipitation_threshold)

    if not point:
        return f"No data found for {polymer} in {solvent}."

    # Get full curve for display
    df = analyzer.get_solubility_curve(polymer, solvent)

    lines = [
        f"# Precipitation Analysis: {polymer} in {solvent}\n",
        "## Key Temperatures\n",
        "| Property | Value |",
        "|----------|-------|",
        f"| Max Solubility | {point.max_solubility:.1f}% at {point.max_solubility_temp:.0f} deg C |",
        f"| Cloud Point (50%) | {point.cloud_point:.0f} deg C |" if point.cloud_point else "| Cloud Point | N/A |",
        f"| Precipitation Temp (<{precipitation_threshold}%) | {point.precipitation_temp:.0f} deg C |" if point.precipitation_temp else f"| Precipitation Temp | Never below {precipitation_threshold}% |",
        f"| Transition Width | {point.transition_width:.0f} deg C |",
        f"| Data Points | {point.data_points} |",
        "\n## Temperature-Solubility Curve\n",
        "| Temp (deg C) | Solubility (%) |",
        "|-------------|----------------|",
    ]

    # Show key temperatures from the curve
    for _, row in df.iloc[::3].iterrows():  # Every 3rd point to keep output manageable
        lines.append(f"| {row['temperature']:.0f} | {row['solubility']:.1f} |")

    return "\n".join(lines)


@safe_tool_wrapper
def plot_precipitation_curves(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Plot temperature-dependent solubility curves for multiple polymers, highlighting precipitation temperatures.

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,EVOH")
        solvent: Solvent to analyze
        precipitation_threshold: Threshold line to draw (default: 10%)

    WHEN TO USE:
    - "Plot LDPE vs EVOH solubility in toluene"
    - "Visualize the precipitation curves for PP, PS, PET"
    - "Show me a graph of temperature-dependent solubility"
    """
    import matplotlib.pyplot as plt

    polymer_list = [p.strip() for p in polymers.split(",")]

    conn = get_connection()
    analyzer = PrecipitationAnalyzer(conn)

    # Create figure
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    precip_temps = {}

    for i, polymer in enumerate(polymer_list):
        df = analyzer.get_solubility_curve(polymer, solvent)
        if df.empty:
            continue

        color = _PUB_COLORS[i % len(_PUB_COLORS)]
        ax.plot(df['temperature'], df['solubility'], '-o', color=color,
                label=polymer, linewidth=1.2, markersize=3)

        # Find and mark precipitation temperature
        precip_temp = analyzer.find_precipitation_temperature(polymer, solvent, precipitation_threshold)
        if precip_temp:
            precip_temps[polymer] = precip_temp
            ax.axvline(x=precip_temp, color=color, linestyle=':', alpha=0.7, linewidth=0.8)
            ax.annotate(f'{polymer}\n{precip_temp:.0f}\u00b0C', xy=(precip_temp, precipitation_threshold + 5),
                       fontsize=_PUB_FONTSIZE - 2, color=color, ha='center')

    # Add threshold line
    ax.axhline(y=precipitation_threshold, color='gray', linestyle='--', alpha=0.5,
               label=f'Threshold ({precipitation_threshold}%)')

    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel('Solubility (%)')
    ax.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 170)
    ax.set_ylim(0, 105)

    # Save plot
    from strap.tools._helpers import descriptive_plot_name
    plot_name = descriptive_plot_name("precipitation_curves", polymers=polymer_list, solvents=[solvent])
    filename = save_plot(fig, plot_name, "matplotlib")

    # Build summary
    lines = [
        f"# Precipitation Curves: {', '.join(polymer_list)} in {solvent.upper()}\n",
        f"**Plot saved:** `{filename}`\n",
        "## Precipitation Temperatures\n",
        "| Polymer | Precip Temp |",
        "|---------|-------------|",
    ]

    for polymer, temp in sorted(precip_temps.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {polymer} | {temp:.0f} deg C |")

    if len(precip_temps) >= 2:
        temps = list(precip_temps.values())
        max_gap = max(temps) - min(temps)
        lines.append(f"\n**Maximum Temperature Gap:** {max_gap:.0f} deg C")

    return "\n".join(lines)


@safe_tool_wrapper
def plot_atmospheric_feasibility(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Plot solubility curves with solvent boiling point to show whether separation is feasible at 1 atm.

    Args:
        polymers: Comma-separated list of polymers (e.g., "HDPE,LDPE,PP" or "LDPE,EVOH")
        solvent: Solvent to analyze (must have boiling point data)
        precipitation_threshold: Solubility threshold for precipitation (default: 1%)

    WHEN TO USE:
    - "Plot the atmospheric feasibility for LDPE/EVOH in DMF"
    - "Visualize whether we can separate these polymers at 1 atm"
    - "Show me the precipitation curves with boiling point"
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from strap.engines.precipitation import SOLVENT_BOILING_POINTS

    polymer_list = [p.strip().upper() for p in polymers.split(",")]

    conn = get_connection()
    analyzer = PrecipitationAnalyzer(conn)

    # Get boiling point
    solvent_lower = solvent.lower()
    bp = SOLVENT_BOILING_POINTS.get(solvent_lower)
    if bp is None:
        solvent_clean = solvent_lower.replace(' ', '').replace('-', '')
        bp = SOLVENT_BOILING_POINTS.get(solvent_clean)

    if bp is None:
        return f"Error: No boiling point data for {solvent}. Available solvents: {', '.join(list(SOLVENT_BOILING_POINTS.keys())[:20])}..."

    # Create figure
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    precip_temps = {}
    max_solubilities = {}
    all_temps = []

    for i, polymer in enumerate(polymer_list):
        df = analyzer.get_solubility_curve(polymer, solvent)
        if df.empty:
            continue

        color = _PUB_COLORS[i % len(_PUB_COLORS)]
        ax.plot(df['temperature'], df['solubility'], '-o', color=color,
                label=polymer, linewidth=1.2, markersize=3, alpha=0.9)

        all_temps.extend(df['temperature'].tolist())
        max_solubilities[polymer] = df['solubility'].max()

        # Find precipitation temperature
        precip_temp = analyzer.find_precipitation_temperature(polymer, solvent, precipitation_threshold)
        if precip_temp:
            precip_temps[polymer] = precip_temp
            ax.axvline(x=precip_temp, color=color, linestyle=':', alpha=0.6, linewidth=0.8)
            # Annotate precipitation point
            ax.scatter([precip_temp], [precipitation_threshold], color=color, s=20, zorder=5, marker='v')

    if not precip_temps:
        plt.close()
        return f"Error: No precipitation data found for {', '.join(polymer_list)} in {solvent}"

    # Determine x-axis range
    min_temp = min(all_temps) if all_temps else 20
    max_temp = max(all_temps) if all_temps else 160
    x_max = max(max_temp + 20, bp + 30)

    # Add boiling point line (critical constraint)
    ax.axvline(x=bp, color='red', linestyle='--', linewidth=1.2, label=f'BP ({bp}\u00b0C)')

    # Add precipitation threshold line
    ax.axhline(y=precipitation_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=0.6)
    ax.text(min_temp + 2, precipitation_threshold + 2, f'Threshold ({precipitation_threshold}%)',
            fontsize=_PUB_FONTSIZE - 2, color='gray')

    # Calculate dissolution temperature needed
    max_precip_temp = max(precip_temps.values())
    dissolution_temp = max_precip_temp + 20

    # Determine if feasible at atmospheric pressure
    is_feasible = dissolution_temp < bp

    # Add shaded regions
    if is_feasible:
        # Green zone: atmospheric operation possible
        ax.axvspan(min_temp, bp, alpha=0.08, color='green', label='Atmospheric zone')
        ax.axvline(x=dissolution_temp, color='green', linestyle='-.', linewidth=0.8, alpha=0.7)
        ax.text(dissolution_temp + 1, 90, f'~{dissolution_temp:.0f}\u00b0C', fontsize=_PUB_FONTSIZE - 2, color='green')
        feasibility_text = f"FEASIBLE AT 1 ATM\nMargin: {bp - dissolution_temp:.0f}\u00b0C below BP"
        text_color = 'green'
    else:
        # Red zone: requires pressurization
        ax.axvspan(bp, x_max, alpha=0.1, color='red', label='Requires pressure')
        ax.axvline(x=dissolution_temp, color='orange', linestyle='-.', linewidth=0.8, alpha=0.7)
        ax.text(dissolution_temp + 1, 90, f'~{dissolution_temp:.0f}\u00b0C', fontsize=_PUB_FONTSIZE - 2, color='orange')
        feasibility_text = f"REQUIRES PRESSURIZATION\nNeeds {dissolution_temp - bp:.0f}\u00b0C above BP"
        text_color = 'red'

    # Add feasibility annotation box
    props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=text_color, alpha=0.9)
    ax.text(0.98, 0.98, feasibility_text, transform=ax.transAxes, fontsize=_PUB_FONTSIZE - 1,
            verticalalignment='top', horizontalalignment='right', bbox=props, color=text_color)

    # Precipitation sequence annotation
    sorted_precip = sorted(precip_temps.items(), key=lambda x: x[1], reverse=True)
    seq_text = " \u2192 ".join([f"{p}@{t:.0f}\u00b0C" for p, t in sorted_precip])
    ax.text(0.02, 0.02, seq_text, transform=ax.transAxes, fontsize=_PUB_FONTSIZE - 2,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel('Solubility (%)')
    ax.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_temp - 5, x_max)
    ax.set_ylim(0, 105)

    # Save plot
    from strap.tools._helpers import descriptive_plot_name
    plot_name = descriptive_plot_name("atmospheric_feasibility", polymers=polymer_list, solvents=[solvent])
    filename = save_plot(fig, plot_name, "matplotlib")

    # Build summary
    lines = [
        f"# Atmospheric Feasibility Visualization\n",
        f"**Plot saved:** `{filename}`\n",
        f"## System: {', '.join(polymer_list)} in {solvent.upper()}\n",
        f"**Solvent Boiling Point:** {bp} deg C at 1 atm",
        f"**Dissolution Temperature Needed:** ~{dissolution_temp:.0f} deg C\n",
    ]

    if is_feasible:
        lines.append("## Feasible at Atmospheric Pressure")
        lines.append(f"Safety margin: {bp - dissolution_temp:.0f} deg C below boiling point\n")
    else:
        lines.append("## Requires Pressurization")
        lines.append(f"Would need to operate {dissolution_temp - bp:.0f} deg C above boiling point\n")

    lines.append("## Precipitation Sequence (during cooling)\n")
    lines.append("| Order | Polymer | Precip Temp | Max Solubility |")
    lines.append("|-------|---------|-------------|----------------|")

    for i, (polymer, temp) in enumerate(sorted_precip, 1):
        max_sol = max_solubilities.get(polymer, 0)
        lines.append(f"| {i} | {polymer} | {temp:.0f} deg C | {max_sol:.1f}% |")

    # Temperature gaps
    if len(sorted_precip) >= 2:
        lines.append("\n## Temperature Gaps")
        for i in range(len(sorted_precip) - 1):
            p1, t1 = sorted_precip[i]
            p2, t2 = sorted_precip[i + 1]
            gap = t1 - t2
            lines.append(f"- {p1} -> {p2}: **{gap:.0f} deg C**")

    return "\n".join(lines)


@safe_tool_wrapper
def compare_polymer_pairs_precipitation(
    polymer_pairs: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
) -> str:
    """Compare differential precipitation feasibility across multiple polymer pairs, trying both orders.

    Args:
        polymer_pairs: Semicolon-separated polymer pairs (e.g., "LDPE,PET;LDPE,EVOH;PP,PS")
        min_temperature_gap: Minimum temperature gap required (default: 20 deg C)
        precipitation_threshold: Solubility below which polymer is precipitated (default: 1%)

    WHEN TO USE:
    - "Compare LDPE/PET vs LDPE/EVOH for differential precipitation"
    - "Which polymer pair is easier to separate by cooling?"
    - "Evaluate precipitation feasibility for multiple polymer combinations"
    """
    conn = get_connection()
    analyzer = PrecipitationAnalyzer(conn)

    pairs = [p.strip().split(",") for p in polymer_pairs.split(";")]
    results = []

    for pair in pairs:
        if len(pair) != 2:
            continue
        p1, p2 = pair[0].strip(), pair[1].strip()

        # Try both orders
        order1 = analyzer.find_differential_precipitation_solvents(
            polymer_to_precipitate=p1,
            polymer_to_retain=p2,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            top_k=5,
        )

        order2 = analyzer.find_differential_precipitation_solvents(
            polymer_to_precipitate=p2,
            polymer_to_retain=p1,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            top_k=5,
        )

        best_results = order1 if len(order1) >= len(order2) else order2
        best_order = f"{p1} first" if len(order1) >= len(order2) else f"{p2} first"

        results.append({
            "pair": f"{p1}/{p2}",
            "order1_count": len(order1),
            "order2_count": len(order2),
            "best_order": best_order,
            "best_results": best_results,
            "max_gap": best_results[0].temperature_gap if best_results else 0,
        })

    # Sort by feasibility (number of solvents * max gap)
    results.sort(key=lambda x: len(x["best_results"]) * x["max_gap"], reverse=True)

    # Format output
    lines = ["# Polymer Pair Comparison for Differential Precipitation\n"]
    lines.append(f"Minimum temperature gap: {min_temperature_gap} deg C\n")

    lines.append("## Summary\n")
    lines.append("| Pair | Solvents Found | Best Order | Max Gap |")
    lines.append("|------|----------------|------------|---------|")
    for r in results:
        lines.append(f"| {r['pair']} | {len(r['best_results'])} | {r['best_order']} | {r['max_gap']:.0f} deg C |")

    lines.append("\n## Recommendation\n")
    if results and results[0]["best_results"]:
        best = results[0]
        lines.append(f"**Most feasible pair:** {best['pair']}")
        lines.append(f"- **{len(best['best_results'])} solvents** found with >={min_temperature_gap} deg C gap")
        lines.append(f"- **Best order:** {best['best_order']}")
        lines.append(f"- **Maximum temperature gap:** {best['max_gap']:.0f} deg C")

        lines.append("\n### Top Solvents:\n")
        lines.append("| Solvent | Temp Gap | First Precip | Second Precip |")
        lines.append("|---------|----------|--------------|---------------|")
        for sol in best["best_results"][:5]:
            lines.append(
                f"| {sol.solvent} | {sol.temperature_gap:.0f} deg C | "
                f"{sol.polymer_first} @ {sol.polymer_first_precip_temp:.0f} deg C | "
                f"{sol.polymer_second} @ {sol.polymer_second_precip_temp:.0f} deg C |"
            )
    else:
        lines.append("No feasible pairs found with the specified temperature gap.")
        lines.append(f"Try reducing min_temperature_gap below {min_temperature_gap} deg C.")

    # Add comparison for non-feasible pairs
    non_feasible = [r for r in results if not r["best_results"]]
    if non_feasible:
        lines.append("\n## Non-Feasible Pairs\n")
        for r in non_feasible:
            lines.append(f"- **{r['pair']}**: No solvents with >={min_temperature_gap} deg C gap found")

    return "\n".join(lines)


@safe_tool_wrapper
def check_atmospheric_feasibility(
    polymer1: str,
    polymer2: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    min_solubility: float = 30.0,
) -> str:
    """Check if two-polymer differential precipitation can be performed below the solvent boiling point (1 atm).

    Args:
        polymer1: First polymer name (e.g., "LDPE", "EVOH", "PU")
        polymer2: Second polymer name
        min_temperature_gap: Minimum temperature gap required for separation (default: 20 deg C)
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 1%)
        min_solubility: Minimum max solubility required for both polymers (default: 30%)

    WHEN TO USE:
    - "Can I separate LDPE and EVOH at atmospheric pressure?"
    - "Which solvents allow separation without pressurized equipment?"
    """
    conn = get_connection()

    from strap.engines.precipitation import format_atmospheric_feasibility_results

    try:
        analyzer = PrecipitationAnalyzer(conn)
        results = analyzer.check_atmospheric_feasibility(
            polymer1=polymer1,
            polymer2=polymer2,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            min_solubility=min_solubility,
            top_k=10
        )

        if not results:
            return (
                f"No solvents found for {polymer1}/{polymer2} differential precipitation "
                f"with >={min_temperature_gap} deg C gap. Try:\n"
                f"- Reducing min_temperature_gap (currently {min_temperature_gap} deg C)\n"
                f"- Reducing min_solubility threshold (currently {min_solubility}%)\n"
                f"- Checking if both polymers have solubility data in the database"
            )

        return format_atmospheric_feasibility_results(results, include_infeasible=True)

    except Exception as e:
        logger.error(f"Error in atmospheric feasibility check: {e}")
        return f"Error analyzing atmospheric feasibility: {str(e)}"


@safe_tool_wrapper
def check_multi_polymer_atmospheric_feasibility(
    polymers: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    min_solubility: float = 30.0,
) -> str:
    """Check if sequential precipitation of 2+ polymers is feasible below solvent boiling point (1 atm).

    Args:
        polymers: Comma-separated polymer names (e.g., "EVOH,PVC,LDPE"); min 2 required
        min_temperature_gap: Minimum gap between consecutive precipitations (default: 20 deg C)
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 1%)
        min_solubility: Minimum max solubility required for each polymer (default: 30%)

    WHEN TO USE:
    - "Can I separate EVOH, PVC, and LDPE at atmospheric pressure?"
    - "Find solvents for 3-polymer sequential precipitation at 1 atm"
    """
    conn = get_connection()

    from strap.engines.precipitation import format_multi_polymer_atmospheric_results

    # Parse polymer list
    polymer_list = [p.strip().upper() for p in polymers.split(',') if p.strip()]

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers. Provide comma-separated list, e.g., 'LDPE,EVOH,PVC'"

    try:
        analyzer = PrecipitationAnalyzer(conn)
        results = analyzer.check_multi_polymer_atmospheric_feasibility(
            polymers=polymer_list,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            min_solubility=min_solubility,
            top_k=10
        )

        if not results:
            available = analyzer.get_available_polymers()
            return (
                f"No solvents found for {'/'.join(polymer_list)} sequential precipitation "
                f"with >={min_temperature_gap} deg C gaps between each step.\n\n"
                f"**Suggestions:**\n"
                f"- Reduce min_temperature_gap (currently {min_temperature_gap} deg C)\n"
                f"- Reduce min_solubility threshold (currently {min_solubility}%)\n"
                f"- Check polymer names are valid\n\n"
                f"**Available polymers:** {', '.join(available)}"
            )

        return format_multi_polymer_atmospheric_results(results, include_infeasible=True)

    except Exception as e:
        logger.error(f"Error in multi-polymer atmospheric feasibility check: {e}")
        return f"Error analyzing multi-polymer atmospheric feasibility: {str(e)}"


# ============================================================================
# Antisolvent Precipitation Tools
# ============================================================================

@safe_tool_wrapper
def find_antisolvents(
    polymer: str,
    max_solubility: float = 1.0,
    temperature: float = 25.0,
) -> str:
    """Find antisolvents (near-zero solubility solvents) for a polymer at a given temperature.

    Args:
        polymer: Polymer name (e.g., "LDPE", "PET", "PP")
        max_solubility: Maximum solubility threshold (%) to qualify as antisolvent (default: 1%)
        temperature: Temperature to check solubility at (default: 25 deg C)

    WHEN TO USE:
    - "Find antisolvents for LDPE"
    - "Which solvents don't dissolve PET at room temperature?"
    """
    try:
        from strap.solubility import get_solubility, get_available_solvents_for_polymer

        solvents = get_available_solvents_for_polymer(polymer)
        rows = []
        for sv in solvents:
            sol = get_solubility(polymer, sv, temperature)
            if sol is not None and sol <= max_solubility:
                rows.append({'solvent': sv, 'solubility': sol, 'temp': temperature})

        if not rows:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(rows).sort_values('solubility').reset_index(drop=True)

        if df.empty:
            return (
                f"No antisolvents found for {polymer} with solubility < {max_solubility}% at {temperature} deg C.\n\n"
                f"Try increasing max_solubility threshold or checking a different temperature."
            )

        # Deduplicate by solvent name
        df = df.drop_duplicates(subset=['solvent'])

        lines = [
            f"# Antisolvents for {polymer.upper()}\n",
            f"Solvents with solubility < {max_solubility}% at ~{temperature} deg C\n",
            f"**Found {len(df)} antisolvents** (polymer is essentially insoluble)\n",
            "| Rank | Antisolvent | Solubility | Temp |",
            "|------|-------------|------------|------|",
        ]

        for i, row in df.iterrows():
            sol = row['solubility']
            if sol < 0.001:
                sol_str = f"{sol:.2e}%"
            elif sol < 0.1:
                sol_str = f"{sol:.4f}%"
            else:
                sol_str = f"{sol:.2f}%"
            lines.append(f"| {i+1} | {row['solvent']} | {sol_str} | {row['temp']:.0f} deg C |")

        lines.append("\n## Usage")
        lines.append("These solvents can be used as antisolvents to precipitate "
                    f"{polymer} from solution by adding them to a dissolved polymer mixture.")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error finding antisolvents: {e}")
        return f"Error finding antisolvents: {str(e)}"


@safe_tool_wrapper
def find_antisolvent_pairs(
    polymer: str,
    min_good_solubility: float = 50.0,
    max_antisolvent_solubility: float = 1.0,
) -> str:
    """Find good solvent + antisolvent pairs for dissolving then precipitating a polymer.

    Args:
        polymer: Polymer name (e.g., "LDPE", "PET", "EVOH")
        min_good_solubility: Minimum solubility (%) for good solvent classification (default: 50%)
        max_antisolvent_solubility: Maximum solubility (%) for antisolvent classification (default: 1%)

    WHEN TO USE:
    - "Find solvent/antisolvent pairs for LDPE recovery"
    - "What combinations work for antisolvent precipitation of PET?"
    """
    try:
        from strap.solubility import get_solubility, get_solubility_curve, get_available_solvents_for_polymer

        solvents = get_available_solvents_for_polymer(polymer)

        # Find good solvents: max solubility across all temperatures >= threshold
        good_rows = []
        for sv in solvents:
            curve = get_solubility_curve(polymer, sv, t_start_c=25, t_end_c=160, t_step_c=5)
            if curve:
                max_sol = max(pt['solubility'] for pt in curve)
                max_temp = next(pt['temperature'] for pt in curve if pt['solubility'] == max_sol)
                if max_sol >= min_good_solubility:
                    good_rows.append({'solvent': sv, 'max_solubility': max_sol, 'dissolution_temp': max_temp})
        good_solvents = (
            pd.DataFrame(good_rows).sort_values('max_solubility', ascending=False).reset_index(drop=True)
            if good_rows else pd.DataFrame()
        )

        # Find antisolvents: low solubility at room temperature
        anti_rows = []
        for sv in solvents:
            sol = get_solubility(polymer, sv, 25.0)
            if sol is not None and sol <= max_antisolvent_solubility:
                anti_rows.append({'solvent': sv, 'min_solubility': sol, 'temp': 25.0})
        antisolvents = (
            pd.DataFrame(anti_rows).sort_values('min_solubility').reset_index(drop=True)
            if anti_rows else pd.DataFrame()
        )

        if good_solvents.empty:
            return f"No good solvents found for {polymer} with solubility > {min_good_solubility}%"

        if antisolvents.empty:
            return f"No antisolvents found for {polymer} with solubility < {max_antisolvent_solubility}%"

        lines = [
            f"# Antisolvent Precipitation Pairs for {polymer.upper()}\n",
            f"## Good Solvents (for dissolution)\n",
            f"Solvents with >{min_good_solubility}% solubility:\n",
            "| Good Solvent | Max Solubility | Dissolution Temp |",
            "|--------------|----------------|------------------|",
        ]

        for _, row in good_solvents.head(10).iterrows():
            lines.append(f"| {row['solvent']} | {row['max_solubility']:.1f}% | {row['dissolution_temp']:.0f} deg C |")

        lines.append(f"\n## Antisolvents (to induce precipitation)\n")
        lines.append(f"Solvents with <{max_antisolvent_solubility}% solubility at room temp:\n")
        lines.append("| Antisolvent | Solubility at RT |")
        lines.append("|-------------|------------------|")

        for _, row in antisolvents.head(10).iterrows():
            sol = row['min_solubility']
            if sol < 0.001:
                sol_str = f"{sol:.2e}%"
            else:
                sol_str = f"{sol:.4f}%"
            lines.append(f"| {row['solvent']} | {sol_str} |")

        # Recommend best pairs (check solvent miscibility conceptually)
        lines.append("\n## Recommended Pairs\n")
        lines.append("**Best combinations** (good solvent + antisolvent):\n")

        recommendations = []
        for _, gs in good_solvents.head(5).iterrows():
            for _, anti in antisolvents.head(5).iterrows():
                # Skip if same solvent
                if gs['solvent'].lower() == anti['solvent'].lower():
                    continue
                recommendations.append({
                    'good': gs['solvent'],
                    'good_sol': gs['max_solubility'],
                    'good_temp': gs['dissolution_temp'],
                    'anti': anti['solvent'],
                    'anti_sol': anti['min_solubility']
                })

        lines.append("| Good Solvent | Antisolvent | Process |")
        lines.append("|--------------|-------------|---------|")

        for rec in recommendations[:8]:
            process = f"Dissolve at {rec['good_temp']:.0f} deg C, add {rec['anti']} to precipitate"
            lines.append(f"| {rec['good']} ({rec['good_sol']:.0f}%) | {rec['anti']} | {process} |")

        lines.append("\n## Process Steps")
        lines.append(f"1. Dissolve {polymer} in good solvent at elevated temperature")
        lines.append("2. Cool solution to moderate temperature")
        lines.append("3. Slowly add antisolvent while stirring")
        lines.append(f"4. {polymer} precipitates out as antisolvent reduces solvent quality")
        lines.append("5. Filter to collect precipitated polymer")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error finding antisolvent pairs: {e}")
        return f"Error finding antisolvent pairs: {str(e)}"


@safe_tool_wrapper
def analyze_selective_antisolvent_precipitation(
    polymers: str,
    antisolvent: str = "auto",
) -> str:
    """Analyze whether adding an antisolvent can selectively precipitate one polymer before another.

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,PET" or "LDPE,PP,HDPE")
        antisolvent: Specific antisolvent to analyze, or "auto" to find best options

    WHEN TO USE:
    - "Can I selectively precipitate LDPE from a LDPE/PET mixture using an antisolvent?"
    - "Analyze antisolvent-based separation for LDPE, PP, HDPE"
    """
    polymer_list = [p.strip().upper() for p in polymers.split(',')]

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers for selective precipitation analysis"

    try:
        from strap.solubility import get_solubility, get_available_solvents

        all_solvents = get_available_solvents()

        # For each polymer, get solubility in all solvents at room temp (25 °C)
        results = {}
        for polymer in polymer_list:
            sol_dict = {}
            for sv in all_solvents:
                sol = get_solubility(polymer, sv, 25.0)
                if sol is not None:
                    sol_dict[sv] = sol
            if sol_dict:
                results[polymer] = sol_dict

        if len(results) < 2:
            return f"Insufficient solubility data for {', '.join(polymer_list)}"

        # Find antisolvents with differential response
        # (one polymer has higher solubility than another in the antisolvent)
        common_solvents = set.intersection(*[set(r.keys()) for r in results.values()])

        differential_antisolvents = []
        for solvent in common_solvents:
            solubilities = {p: results[p].get(solvent, 100) for p in polymer_list}
            max_sol = max(solubilities.values())
            min_sol = min(solubilities.values())

            # Both should be low (antisolvent), but with differential
            if max_sol < 10 and (max_sol - min_sol) > 0.1:
                differential_antisolvents.append({
                    'solvent': solvent,
                    'solubilities': solubilities,
                    'differential': max_sol - min_sol,
                    'max_sol': max_sol,
                    'min_sol': min_sol
                })

        # Sort by differential (larger = better selectivity)
        differential_antisolvents.sort(key=lambda x: x['differential'], reverse=True)

        lines = [
            f"# Selective Antisolvent Precipitation Analysis\n",
            f"**Polymers:** {', '.join(polymer_list)}\n",
        ]

        if not differential_antisolvents:
            lines.append("## No Differential Antisolvents Found\n")
            lines.append("All tested antisolvents show similar rejection of all polymers.")
            lines.append("Selective antisolvent precipitation may not be feasible for this polymer combination.\n")
            lines.append("**Alternative:** Consider differential precipitation by cooling instead.")
        else:
            lines.append(f"## Found {len(differential_antisolvents)} Antisolvents with Differential Response\n")
            lines.append("These antisolvents reject polymers at different rates, enabling selective precipitation.\n")
            lines.append("| Antisolvent | " + " | ".join([f"{p} Sol." for p in polymer_list]) + " | Differential |")
            lines.append("|-------------|" + "|".join(["--------" for _ in polymer_list]) + "|--------------|")

            for anti in differential_antisolvents[:10]:
                row = f"| {anti['solvent']} |"
                for p in polymer_list:
                    sol = anti['solubilities'][p]
                    if sol < 0.01:
                        row += f" {sol:.2e}% |"
                    else:
                        row += f" {sol:.3f}% |"
                row += f" {anti['differential']:.3f}% |"
                lines.append(row)

            # Process recommendation
            if differential_antisolvents:
                best = differential_antisolvents[0]
                sorted_by_sol = sorted(best['solubilities'].items(), key=lambda x: x[1], reverse=True)

                lines.append(f"\n## Recommended Process with {best['solvent'].upper()}\n")
                lines.append("**Precipitation order** (by antisolvent tolerance):\n")

                for i, (polymer, sol) in enumerate(sorted_by_sol, 1):
                    if sol < 0.01:
                        lines.append(f"{i}. **{polymer}** - precipitates first (solubility: {sol:.2e}%)")
                    else:
                        lines.append(f"{i}. **{polymer}** - precipitates {'last' if i == len(sorted_by_sol) else 'next'} (solubility: {sol:.3f}%)")

                lines.append(f"\n**Process:**")
                lines.append(f"1. Dissolve all polymers in a common good solvent at elevated temperature")
                lines.append(f"2. Cool to moderate temperature (~50-60 deg C)")
                lines.append(f"3. Slowly add {best['solvent']} while stirring")
                lines.append(f"4. {sorted_by_sol[0][0]} precipitates first (lowest antisolvent tolerance)")
                lines.append(f"5. Filter to collect {sorted_by_sol[0][0]}")
                if len(sorted_by_sol) > 2:
                    lines.append(f"6. Continue adding {best['solvent']} to precipitate remaining polymers sequentially")
                else:
                    lines.append(f"6. Add more {best['solvent']} to precipitate {sorted_by_sol[1][0]}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in selective antisolvent analysis: {e}")
        return f"Error analyzing selective antisolvent precipitation: {str(e)}"


# ============================================================================
# Separation Planning Tools (merged from separation_planning.py)
# ============================================================================

# ---------------------------------------------------------------------------
# Local helpers for separation planning
# ---------------------------------------------------------------------------

_SOLVENT_DATA_TABLE: Optional[str] = None


def _get_plot_url(filepath: str) -> str:
    """Convert filepath to displayable format."""
    return f"Plot saved: `{filepath}`"


def _selectivity_color(selectivity: float) -> str:
    """Return color hex for a selectivity value."""
    if selectivity > 30:
        return "#2ecc71"
    elif selectivity > 10:
        return "#f1c40f"
    elif selectivity > 0:
        return "#e67e22"
    else:
        return "#e74c3c"


_SELECTIVITY_LEGEND = [
    ("#2ecc71", "Excellent (>30%)"),
    ("#f1c40f", "Good (10-30%)"),
    ("#e67e22", "Marginal (0-10%)"),
    ("#e74c3c", "Poor (<0%)"),
]


def _plot_separation_sequence(
    polymer_list: list[str],
    sequence_data: dict,
    temperature: float,
    total_sequences: int,
    rank: int = 1,
    filename: str | None = None,
) -> str:
    """Plot a single ranked separation sequence (flowchart style).

    Args:
        polymer_list: Full list of polymers in the mixture.
        sequence_data: Dict with keys ``sequence``, ``min_selectivity``, ``steps``.
            Each step has ``target``, ``remaining``, ``solvents`` (list of dicts with
            ``solvent``, ``selectivity``, and optionally ``temperature``,
            ``optimal_temp``, ``optimal_selectivity``).
        temperature: Operating temperature in °C.
        total_sequences: Total number of evaluated sequences (for title).
        rank: Rank of this sequence.
        filename: Override output filename (default: ``separation_sequence_rank{rank}``).

    Returns:
        Filepath of the saved PNG.
    """
    sequence = sequence_data["sequence"]
    steps = sequence_data["steps"]
    min_sel = sequence_data["min_selectivity"]

    n_steps = len(steps)
    fig_height = max(3 + n_steps * 2.5, 8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.set_title(
        f"RECOMMENDED SEPARATION SEQUENCE (Rank #{rank} of {total_sequences})\n"
        f'Sequence: {" -> ".join(sequence)} | Min Selectivity: {min_sel:.1f}% | Temp: {temperature} C',
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n_steps + 2.5)
    ax.axis("off")

    # Starting mixture bar
    y_pos = n_steps + 1.5
    ax.add_patch(plt.Rectangle((2, y_pos - 0.3), 6, 0.6,
                               facecolor="#3498db", edgecolor="black", linewidth=2))
    ax.text(5, y_pos, f'STARTING MIXTURE: {", ".join(polymer_list)}',
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    for idx, step in enumerate(steps):
        y_pos = n_steps - idx
        target = step["target"]
        remaining = step.get("remaining", [])
        top_solvent = step["solvents"][0] if step.get("solvents") else {"solvent": "N/A", "selectivity": 0}
        solvent_name = top_solvent["solvent"]
        selectivity = top_solvent.get("selectivity", 0)
        step_temp = top_solvent.get("temperature", temperature)
        optimal_temp = top_solvent.get("optimal_temp", step_temp)
        optimal_sel = top_solvent.get("optimal_selectivity", selectivity)
        color = _selectivity_color(selectivity)

        # Arrow from previous level
        ax.annotate("", xy=(3.5, y_pos + 0.4), xytext=(3.5, y_pos + 0.9),
                    arrowprops=dict(arrowstyle="->", lw=4, color=color))

        # Step box
        ax.add_patch(plt.Rectangle((1.2, y_pos - 0.35), 4.6, 0.7,
                                   facecolor=color, edgecolor="black", linewidth=2.5, alpha=0.3))

        # Step number circle
        ax.add_patch(plt.Circle((1.9, y_pos), 0.25, facecolor=color, edgecolor="black", linewidth=2))
        ax.text(1.9, y_pos, str(idx + 1), ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

        ax.text(2.7, y_pos, f"SEPARATE: {target}",
                ha="left", va="center", fontsize=14, fontweight="bold")

        # Solvent info box
        ax.add_patch(plt.Rectangle((6.2, y_pos + 0.35), 3.5, 0.75,
                                   facecolor="white", edgecolor=color, linewidth=2))
        ax.text(7.95, y_pos + 0.95, f"Solvent: {solvent_name}",
                ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(7.95, y_pos + 0.72, f"Sel: {selectivity:.1f}% @ {step_temp:.0f} C",
                ha="center", va="center", fontsize=10, color=color, fontweight="bold")
        if abs(optimal_temp - step_temp) > 5 and optimal_sel > selectivity:
            ax.text(7.95, y_pos + 0.5, f"(Optimal: {optimal_sel:.1f}% @ {optimal_temp:.0f} C)",
                    ha="center", va="center", fontsize=8, color="#27ae60", style="italic")

        if remaining:
            ax.text(5.7, y_pos - 0.15, f'Remaining: {", ".join(remaining)}',
                    ha="right", va="center", fontsize=10, color="#34495e",
                    style="italic", weight="bold")
        else:
            ax.text(5.7, y_pos - 0.15, "(Last polymer - isolated)",
                    ha="right", va="center", fontsize=10, color="#27ae60",
                    style="italic", weight="bold")

    # Final bar
    ax.add_patch(plt.Rectangle((2, -0.3), 6, 0.6,
                               facecolor="#2ecc71", edgecolor="black", linewidth=2.5))
    ax.text(5, 0, "ALL POLYMERS SEPARATED",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                   markersize=15, markeredgecolor="black", linewidth=2, label=lbl)
        for c, lbl in _SELECTIVITY_LEGEND
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
              frameon=True, fancybox=True, title="Selectivity Quality", title_fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fname = filename or f"separation_sequence_rank{rank}"
    filepath = save_plot(fig, fname)
    plt.close(fig)
    return filepath


def _plot_topk_comparison(
    polymer_list: list[str],
    sequence_scores: list[dict],
    temperature: float,
    top_k: int = 3,
    filename: str = "separation_topk_comparison",
) -> str:
    """Plot side-by-side comparison of top-k separation sequences.

    Args:
        polymer_list: Full list of polymers.
        sequence_scores: Sorted list of sequence dicts (best first).
        temperature: Operating temperature in °C.
        top_k: Number of sequences to compare.
        filename: Output filename.

    Returns:
        Filepath of the saved PNG.
    """
    top_k = min(top_k, len(sequence_scores))
    n_steps = len(polymer_list) - 1

    fig, ax = plt.subplots(figsize=(5 * top_k, 8), dpi=150)
    ax.set_title(
        f"TOP {top_k} SEPARATION SEQUENCES COMPARISON\n"
        f'Temperature: {temperature} C | Polymers: {", ".join(polymer_list)}',
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xlim(0, top_k * 5)
    ax.set_ylim(-1, n_steps + 2)
    ax.axis("off")

    col_width = 5

    for col_idx, seq_data in enumerate(sequence_scores[:top_k]):
        x_offset = col_idx * col_width
        seq = seq_data["sequence"]
        min_sel = seq_data["min_selectivity"]
        seq_steps = seq_data["steps"]

        medal = "#1" if col_idx == 0 else "#2" if col_idx == 1 else "#3"
        header_color = "#2ecc71" if col_idx == 0 else "#95a5a6"
        ax.add_patch(plt.Rectangle((x_offset + 0.2, n_steps + 1), col_width - 0.4, 0.8,
                                   facecolor=header_color, edgecolor="black", linewidth=2))
        ax.text(x_offset + col_width / 2, n_steps + 1.4, f"{medal} Rank #{col_idx + 1}",
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")

        ax.text(x_offset + col_width / 2, n_steps + 0.6, " -> ".join(seq),
                ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        for step_idx, step in enumerate(seq_steps):
            y_pos = n_steps - step_idx - 0.5
            target = step.get("target", "?")
            solvents_list = step.get("solvents", [])
            if solvents_list and isinstance(solvents_list, list):
                best_sol = solvents_list[0]
                solvent = best_sol.get("solvent", "N/A")
                selectivity = best_sol.get("selectivity", 0)
                step_temp = best_sol.get("temperature", temperature)
                optimal_temp = best_sol.get("optimal_temp", step_temp)
                optimal_sel = best_sol.get("optimal_selectivity", selectivity)
            else:
                solvent, selectivity = "N/A", 0
                step_temp = optimal_temp = temperature
                optimal_sel = 0

            has_optimal = abs(optimal_temp - step_temp) > 5 and optimal_sel > selectivity
            color = _selectivity_color(selectivity)

            box_height = 0.85 if has_optimal else 0.7
            ax.add_patch(plt.Rectangle((x_offset + 0.3, y_pos - 0.4), col_width - 0.6, box_height,
                                       facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.3))

            y_circle = y_pos + 0.05 if has_optimal else y_pos
            ax.add_patch(plt.Circle((x_offset + 0.7, y_circle), 0.2, facecolor=color, edgecolor="black"))
            ax.text(x_offset + 0.7, y_circle, str(step_idx + 1), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

            y_target = y_pos + 0.22 if has_optimal else y_pos + 0.15
            ax.text(x_offset + 1.1, y_target, f"{target}",
                    ha="left", va="center", fontsize=12, fontweight="bold")

            y_solvent = y_pos + 0.0 if has_optimal else y_pos - 0.15
            ax.text(x_offset + 1.1, y_solvent, f"{solvent} ({selectivity:.1f}% @{step_temp:.0f} C)",
                    ha="left", va="center", fontsize=8, color="#34495e")

            if has_optimal:
                ax.text(x_offset + 1.1, y_pos - 0.22, f"Opt: {optimal_sel:.1f}% @{optimal_temp:.0f} C",
                        ha="left", va="center", fontsize=7, color="#27ae60", style="italic")

        summary_color = "#2ecc71" if min_sel > 10 else "#f39c12" if min_sel > 0 else "#e74c3c"
        ax.add_patch(plt.Rectangle((x_offset + 0.3, -0.8), col_width - 0.6, 0.5,
                                   facecolor=summary_color, edgecolor="black", linewidth=2))
        ax.text(x_offset + col_width / 2, -0.55, f"Min Sel: {min_sel:.1f}%",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                   markersize=12, markeredgecolor="black", label=lbl)
        for c, lbl in _SELECTIVITY_LEGEND
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              frameon=True, fancybox=True, title="Selectivity", title_fontsize=10)

    plt.tight_layout()
    filepath = save_plot(fig, filename)
    plt.close(fig)
    return filepath


def _get_solvent_table_name() -> Optional[str]:
    """Auto-detect the solvent data table from the DuckDB connection."""
    global _SOLVENT_DATA_TABLE

    conn = get_connection()

    # Check if already detected
    if _SOLVENT_DATA_TABLE is not None:
        try:
            conn.execute(f"SELECT 1 FROM {_SOLVENT_DATA_TABLE} LIMIT 1")
            return _SOLVENT_DATA_TABLE
        except Exception:
            _SOLVENT_DATA_TABLE = None

    # Discover tables
    try:
        tables_df = conn.execute("SHOW TABLES").fetchdf()
        table_names = tables_df["name"].tolist()
    except Exception:
        return None

    for table_name in table_names:
        if "solvent" in table_name.lower() and "solubility" not in table_name.lower():
            try:
                schema_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                cols_lower = [c.lower() for c in schema_df["column_name"].tolist()]
                if (
                    any("bp" in c or "boil" in c for c in cols_lower)
                    or any("logp" in c for c in cols_lower)
                    or any("energy" in c for c in cols_lower)
                ):
                    _SOLVENT_DATA_TABLE = table_name
                    logger.info(f"Auto-detected solvent data table: {table_name}")
                    return table_name
            except Exception:
                continue

    return None


def _get_solvent_name_column(table_name: str) -> Optional[str]:
    """Return the column that contains solvent names."""
    conn = get_connection()
    try:
        schema_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        cols = schema_df["column_name"].tolist()
    except Exception:
        return None

    priority_patterns = ["solvent_name", "solvent", "name", "compound"]
    for pattern in priority_patterns:
        for col in cols:
            if pattern in col.lower():
                return col

    # Fallback: first VARCHAR column
    types = schema_df["column_type"].tolist()
    for col, dtype in zip(cols, types):
        if "VARCHAR" in str(dtype).upper() or "TEXT" in str(dtype).upper():
            return col

    return cols[0] if cols else None


def _get_cosmobase_column(table_name: str) -> Optional[str]:
    """Return the 'Solvent name in cosmobase' column (if present)."""
    conn = get_connection()
    try:
        schema_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        cols = schema_df["column_name"].tolist()
    except Exception:
        return None

    for col in cols:
        if "cosmobase" in col.lower():
            return col
    return None


from strap.solvent_registry import ABBREVIATION_MAP as _ABBREVIATION_MAP


async def _lookup_solvent_properties(solvent_names: list, solvent_table: str) -> dict:
    """Look up solvent properties for *solvent_names* with fuzzy matching."""
    conn = get_connection()

    # Validate table exists
    try:
        schema_df = conn.execute(f"DESCRIBE {solvent_table}").fetchdf()
        cols = schema_df["column_name"].tolist()
    except Exception:
        return {}

    cols_lower = {c.lower(): c for c in cols}

    cosmobase_col = _get_cosmobase_column(solvent_table)
    name_col = _get_solvent_name_column(solvent_table)

    logp_col = next((cols_lower[k] for k in cols_lower if "logp" in k), None)
    bp_col = next((cols_lower[k] for k in cols_lower if "bp" in k or "boil" in k), None)
    energy_col = next((cols_lower[k] for k in cols_lower if "energy" in k), None)
    cp_col = next((cols_lower[k] for k in cols_lower if "cp" in k and "logp" not in k), None)

    match_col = cosmobase_col or name_col
    if not match_col:
        return {}

    async def find_solvent_match(solvent: str):
        """Try multiple strategies to locate a solvent row."""
        sol_lower = solvent.lower().strip()
        sol_normalized = sol_lower.replace("-", "").replace(" ", "").replace(",", "")

        # Strategy 1: Exact match
        query1 = f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") = \'{sol_lower}\''
        try:
            df = conn.execute(query1).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

        # Strategy 2: Try abbreviation mapping
        if sol_lower in _ABBREVIATION_MAP:
            full_name = _ABBREVIATION_MAP[sol_lower]
            query2 = f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") LIKE \'%{full_name}%\' ORDER BY LENGTH("{match_col}")'
            try:
                df = conn.execute(query2).fetchdf()
                if len(df) > 0:
                    return df.iloc[0]
            except Exception:
                pass

        # Strategy 3: Substring match
        query3 = f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") LIKE \'%{sol_lower}%\' ORDER BY LENGTH("{match_col}")'
        try:
            df = conn.execute(query3).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

        # Strategy 4: Normalised match
        query4 = (
            f"SELECT * FROM {solvent_table} "
            f"WHERE REPLACE(REPLACE(REPLACE(LOWER(\"{match_col}\"), '-', ''), ' ', ''), ',', '') "
            f"LIKE '%{sol_normalized}%' ORDER BY LENGTH(\"{match_col}\")"
        )
        try:
            df = conn.execute(query4).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

        # Strategy 5: Reverse-abbreviation check
        for abbrev, full in _ABBREVIATION_MAP.items():
            if abbrev in sol_lower or sol_lower in full:
                query5 = f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") LIKE \'%{full}%\' ORDER BY LENGTH("{match_col}")'
                try:
                    df = conn.execute(query5).fetchdf()
                    if len(df) > 0:
                        return df.iloc[0]
                except Exception:
                    pass

        return None

    # Gather all matches concurrently
    match_tasks = [find_solvent_match(solvent) for solvent in solvent_names]
    matches = await asyncio.gather(*match_tasks)

    props_map: Dict[str, Dict[str, Any]] = {}
    for solvent, row in zip(solvent_names, matches):
        props: Dict[str, Any] = {"logp": None, "bp": None, "energy": None, "cp": None}
        if row is not None:
            props = {
                "logp": row[logp_col] if logp_col and logp_col in row.index else None,
                "bp": row[bp_col] if bp_col and bp_col in row.index else None,
                "energy": row[energy_col] if energy_col and energy_col in row.index else None,
                "cp": row[cp_col] if cp_col and cp_col in row.index else None,
            }
        props_map[solvent] = props

    return props_map


# ===================================================================
# Helper: greedy separation planning (not a tool – no wrapper)
# ===================================================================

async def _greedy_separation_planning(
    polymer_list: list,
    temperature: float,
    top_k_solvents: int,
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
) -> str:
    """Greedy algorithm for separation planning when n > 3 polymers.

    At each step, selects the polymer that can be most selectively separated
    from all remaining polymers. This is O(n^2) instead of O(n!).
    """
    import math

    from strap.solubility import get_selectivity as _get_selectivity

    n_polymers = len(polymer_list)

    output = []
    output.append("# Greedy Separation Planning\n")
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Count:** {n_polymers} polymers")
    output.append(f"**Algorithm:** Greedy (O(n^2) ~ {n_polymers**2} evaluations)")
    output.append(f"**vs Exhaustive:** {n_polymers}! = {math.factorial(n_polymers):,} permutations avoided")
    output.append(f"**Temperature:** {temperature} C\n")

    output.append("## Algorithm Explanation\n")
    output.append("At each step, we select the polymer that can be **most selectively** separated")
    output.append("from all remaining polymers. This greedy approach finds a good (not necessarily optimal)")
    output.append("sequence efficiently.\n")

    remaining = list(polymer_list)
    sequence: list[str] = []
    steps: list[dict] = []
    used_solvents: set[str] = set()

    output.append("## Step-by-Step Greedy Selection\n")

    step_num = 0
    while len(remaining) > 1:
        step_num += 1
        output.append(f"### Step {step_num}: Evaluating {len(remaining)} candidates\n")
        output.append(f"**Remaining mixture:** {{{', '.join(remaining)}}}\n")

        candidates = []

        for target in remaining:
            others = [p for p in remaining if p != target]

            ret = _get_selectivity(target, others, temperature, used_solvents)
            if ret:
                best_solvent, selectivity, target_sol, max_other = ret
                candidates.append({
                    "polymer": target,
                    "solvent": best_solvent,
                    "selectivity": selectivity,
                    "target_sol": target_sol,
                    "others": others,
                })
            else:
                candidates.append({
                    "polymer": target,
                    "solvent": "N/A",
                    "selectivity": -999,
                    "target_sol": 0,
                    "others": others,
                })

        # Show candidate evaluations
        output.append("| Polymer | Best Solvent | Selectivity |")
        output.append("|---------|--------------|-------------|")
        for c in sorted(candidates, key=lambda x: x["selectivity"], reverse=True):
            sel_str = f"{c['selectivity']:.1f}%" if c["selectivity"] > -900 else "N/A"
            output.append(f"| {c['polymer']} | {c['solvent']} | {sel_str} |")
        output.append("")

        best = max(candidates, key=lambda x: x["selectivity"])

        if best["selectivity"] > -900:
            output.append(f"**Selected: {best['polymer']}** with {best['solvent']} (selectivity: {best['selectivity']:.1f}%)\n")
        else:
            output.append(f"**Selected: {best['polymer']}** (no solubility data available)\n")

        sequence.append(best["polymer"])
        steps.append({
            "step": step_num,
            "target": best["polymer"],
            "solvent": best["solvent"],
            "selectivity": best["selectivity"],
            "remaining_before": list(remaining),
        })
        used_solvents.add(best["solvent"])
        remaining.remove(best["polymer"])

    # Add the last polymer
    if remaining:
        sequence.append(remaining[0])
        output.append(f"### Step {step_num + 1}: {remaining[0]} is isolated\n")

    # Summary
    output.append("---\n")
    output.append("## Greedy Separation Sequence Summary\n")
    output.append(f"**Optimized Sequence:** {' -> '.join(sequence)}\n")

    output.append("### Step-by-Step Protocol\n")
    output.append("| Step | Separate | Using Solvent | Selectivity |")
    output.append("|------|----------|---------------|-------------|")

    valid_steps = [s for s in steps if s["selectivity"] > -900]
    for s in steps:
        sel_str = f"{s['selectivity']:.1f}%" if s["selectivity"] > -900 else "N/A"
        output.append(f"| {s['step']} | {s['target']} | {s['solvent']} | {sel_str} |")
    output.append(f"| {len(steps) + 1} | {sequence[-1]} | (isolated) | done |")
    output.append("")

    # Metrics
    if valid_steps:
        min_sel = min(s["selectivity"] for s in valid_steps)
        avg_sel = sum(s["selectivity"] for s in valid_steps) / len(valid_steps)
        unique_solvents = len(set(s["solvent"] for s in valid_steps if s["solvent"] != "N/A"))

        output.append("### Metrics\n")
        output.append(f"- **Minimum selectivity:** {min_sel:.1f}%")
        output.append(f"- **Average selectivity:** {avg_sel:.1f}%")
        output.append(f"- **Unique solvents needed:** {unique_solvents}")
        output.append(f"- **Evaluations performed:** ~{n_polymers * (n_polymers + 1) // 2}")

    output.append("\n---\n")
    output.append("*Note: Greedy algorithm finds a good sequence efficiently but may not be globally optimal.*")
    output.append("*For <=3 polymers, exhaustive search is used to find the true optimum.*")

    display = "\n".join(output)

    # Build structured data for programmatic access
    valid_steps = [s for s in steps if s["selectivity"] > -900]
    solvents_used = list(set(s["solvent"] for s in valid_steps if s["solvent"] != "N/A"))

    solvent_mapping = {s["target"]: s["solvent"] for s in valid_steps if s["solvent"] != "N/A"}

    top_k_sequences = [{
        "rank": 1,
        "sequence": sequence,
        "min_selectivity": min(s["selectivity"] for s in valid_steps) if valid_steps else 0,
        "solvent_mapping": solvent_mapping,
    }]

    structured_data = {
        "tool_name": "plan_sequential_separation",
        "success": True,
        "polymers_analyzed": polymer_list,
        "best_sequence": sequence,
        "solvents": solvents_used,
        "selectivities": [s["selectivity"] for s in valid_steps],
        "temperature": temperature,
        "algorithm_used": "greedy",
        "steps": [
            {
                "step": s["step"],
                "target": s["target"],
                "solvent": s["solvent"],
                "selectivity": s["selectivity"],
            }
            for s in steps
        ],
        "min_selectivity": min(s["selectivity"] for s in valid_steps) if valid_steps else None,
        "max_selectivity": max(s["selectivity"] for s in valid_steps) if valid_steps else None,
        "coverage_complete": len(sequence) == len(polymer_list),
        "top_k_sequences": top_k_sequences,
        "total_sequences_evaluated": 1,
    }

    return json.dumps({"display": display, "data": structured_data}, ensure_ascii=False)


# ===================================================================
# Tool: plan_multiple_separation_schemes (token-efficient multi-scheme)
# ===================================================================

@safe_tool_wrapper
async def plan_multiple_separation_schemes(
    polymers: str,
    temperature: float = 120.0,
    min_selectivity: float = 5.0,
    n_variants: int = 2,
) -> str:
    """Generate diverse separation schemes: max-selectivity, safest, lowest-energy, each with variants.

    Args:
        polymers: Comma-separated polymer list (e.g., "LDPE,HDPE,PP,PS,PET,PVC,EVOH,Nylon6,Nylon66")
        temperature: Target temperature in C (default: 120.0)
        min_selectivity: Min selectivity threshold for safety/energy schemes (default: 5.0)
        n_variants: Number of variants per scheme type (default: 2). Variant 1 is the greedy optimum;
            variant 2+ explore alternative first-step choices that cascade into different sequences.
    """
    from strap.solubility import (
        get_all_solvents_selectivity as _get_all_sel,
        get_available_solvents as _get_available_solvents,
    )

    polymer_list = [p.strip().upper() for p in polymers.split(",") if p.strip()]
    n = len(polymer_list)
    if n < 2:
        return "Error: Need at least 2 polymers."

    # ---- Pre-load solvent properties (BP, LogP, G-score) once ----
    all_solvents = _get_available_solvents()

    bp_map: dict[str, float] = {}
    logp_map: dict[str, float] = {}
    gscore_map: dict[str, float] = {}

    solvent_table = _get_solvent_table_name()
    if solvent_table:
        prop_dict = await _lookup_solvent_properties(list(all_solvents), solvent_table)
        for sname, props in prop_dict.items():
            if props.get("bp") is not None:
                try:
                    bp_map[sname] = float(props["bp"])
                except (ValueError, TypeError):
                    pass
            if props.get("logp") is not None:
                try:
                    logp_map[sname] = float(props["logp"])
                except (ValueError, TypeError):
                    pass

    conn = get_connection()
    try:
        gsk_df = conn.execute(
            "SELECT solvent_common_name, g_score FROM gsk_dataset"
        ).fetchdf()
        gsk_lower = {}
        for _, row in gsk_df.iterrows():
            if row["g_score"] is not None:
                gsk_lower[row["solvent_common_name"].lower()] = float(row["g_score"])
        # Match interpolation solvent names to GSK names
        for sname in all_solvents:
            sl = sname.lower()
            if sl in gsk_lower:
                gscore_map[sname] = gsk_lower[sl]
            else:
                # Try abbreviation expansion
                expanded = _ABBREVIATION_MAP.get(sl, sl)
                if expanded.lower() in gsk_lower:
                    gscore_map[sname] = gsk_lower[expanded.lower()]
                else:
                    # Substring match
                    norm = sl.replace("-", "").replace(" ", "")
                    for gk, gv in gsk_lower.items():
                        gk_norm = gk.replace("-", "").replace(" ", "")
                        if norm in gk_norm or gk_norm in norm:
                            gscore_map[sname] = gv
                            break
    except Exception:
        pass

    # ---- Ranking functions ----
    def _rank_selectivity(candidates: list[dict]) -> list[dict]:
        return sorted(candidates, key=lambda c: c.get("selectivity", -999), reverse=True)

    def _rank_safety(candidates: list[dict]) -> list[dict]:
        viable = [
            c for c in candidates
            if c.get("selectivity", -999) >= min_selectivity
            and c.get("solvent", "") in gscore_map
        ]
        if not viable:
            viable = [c for c in candidates if c.get("solvent", "") in gscore_map]
        if not viable:
            return _rank_selectivity(candidates)
        return sorted(viable, key=lambda c: gscore_map.get(c["solvent"], 0), reverse=True)

    def _rank_energy(candidates: list[dict]) -> list[dict]:
        viable = [
            c for c in candidates
            if c.get("selectivity", -999) >= min_selectivity
            and c.get("solvent", "") in bp_map
        ]
        if not viable:
            viable = [c for c in candidates if c.get("solvent", "") in bp_map]
        if not viable:
            return _rank_selectivity(candidates)
        return sorted(viable, key=lambda c: bp_map.get(c["solvent"], 999))

    # ---- Generalized greedy loop ----
    def _greedy_scheme(name: str, tag: str, rank_fn, first_step_pick: int = 0) -> dict:
        """Run greedy separation with a given ranking function.

        first_step_pick: index into ranked candidates at step 1 (0=best, 1=2nd-best, ...).
        After step 1, always picks ranked[0] (greedy optimum).
        """
        remaining = list(polymer_list)
        steps = []
        used_solvents: set[str] = set()
        is_first_step = True

        while len(remaining) > 1:
            candidates = []
            for target in remaining:
                others = [p for p in remaining if p != target]
                all_sel = _get_all_sel(target, others, temperature)
                # Filter used solvents
                if used_solvents:
                    all_sel = [s for s in all_sel if s["solvent"] not in used_solvents]
                if not all_sel:
                    candidates.append({"polymer": target, "solvent": "N/A", "selectivity": -999})
                    continue
                # Apply scheme ranking to find best solvent for this target
                ranked = rank_fn(all_sel)
                best = ranked[0]
                candidates.append({
                    "polymer": target,
                    "solvent": best["solvent"],
                    "selectivity": best["selectivity"],
                })

            # Pick (polymer, solvent) pair — use alternate pick on first step only
            ranked_cands = rank_fn(candidates)
            pick_idx = 0
            if is_first_step and first_step_pick > 0:
                pick_idx = min(first_step_pick, len(ranked_cands) - 1)
                is_first_step = False
            else:
                is_first_step = False
            winner = ranked_cands[pick_idx]

            solvent_bp = bp_map.get(winner["solvent"])
            # Operating temp: requested temp or 5°C below BP, whichever is lower
            if solvent_bp is not None:
                op_temp = min(temperature, solvent_bp - 5)
            else:
                op_temp = temperature
            steps.append({
                "step": len(steps) + 1,
                "target": winner["polymer"],
                "solvent": winner["solvent"],
                "sel": winner["selectivity"],
                "temp": op_temp,
                "bp": solvent_bp,
                "gsk": gscore_map.get(winner["solvent"]),
                "logp": logp_map.get(winner["solvent"]),
            })
            used_solvents.add(winner["solvent"])
            remaining.remove(winner["polymer"])

        # Last polymer isolated
        if remaining:
            steps.append({
                "step": len(steps) + 1,
                "target": remaining[0],
                "solvent": "-",
                "sel": None, "temp": None, "bp": None, "gsk": None, "logp": None,
            })

        valid = [s["sel"] for s in steps if s["sel"] is not None and s["sel"] > -900]
        return {
            "name": name,
            "tag": tag,
            "steps": steps,
            "seq": [s["target"] for s in steps],
            "min_sel": min(valid) if valid else 0,
            "avg_sel": sum(valid) / len(valid) if valid else 0,
            "n_solv": len(set(s["solvent"] for s in steps if s["solvent"] != "-")),
        }

    # ---- Run schemes with variants ----
    n_variants = max(1, min(n_variants, 5))  # clamp to [1, 5]
    scheme_defs = [
        ("Max Selectivity", "SEL", _rank_selectivity),
        ("Safest Process (GSK)", "SAFE", _rank_safety),
        ("Lowest Energy (BP)", "NRG", _rank_energy),
    ]
    schemes = []
    for base_name, base_tag, rank_fn in scheme_defs:
        for v in range(n_variants):
            suffix = f" (v{v + 1})" if n_variants > 1 else ""
            tag = f"{base_tag}-v{v + 1}" if n_variants > 1 else base_tag
            schemes.append(_greedy_scheme(base_name + suffix, tag, rank_fn, first_step_pick=v))

    # Deduplicate: drop variants with identical polymer order AND solvent choices
    seen_keys: set[str] = set()
    unique_schemes = []
    for s in schemes:
        solvent_list = [st["solvent"] for st in s["steps"] if st["solvent"] != "-"]
        seq_key = ">".join(s["seq"]) + "|" + ",".join(solvent_list)
        if seq_key not in seen_keys:
            seen_keys.add(seq_key)
            unique_schemes.append(s)
    schemes = unique_schemes

    # ---- Compact output ----
    out = []
    out.append(f"MULTI-SCHEME SEPARATION: {','.join(polymer_list)} @ {temperature}C")
    out.append(f"Polymers: {n} | Greedy O(n^2) | {len(schemes)} schemes ({n_variants} variants/type)\n")

    for r in schemes:
        out.append(f"== {r['tag']}: {r['name']} ==")
        out.append(f"Seq: {' > '.join(r['seq'])}")
        out.append(f"Min/Avg sel: {r['min_sel']:.1f}% / {r['avg_sel']:.1f}% | Solvents: {r['n_solv']}")
        out.append("Stp|Target  |Solvent       |Sel% |T(C) |GSK |LogP")
        out.append("---|--------|--------------|-----|-----|----|----")
        for s in r["steps"]:
            if s["solvent"] == "-":
                out.append(f" {s['step']} |{s['target']:<8}|(isolated)    | done|  -  | -  | -")
            else:
                sel_s = f"{s['sel']:.0f}" if s["sel"] is not None and s["sel"] > -900 else "?"
                t_s = f"{s['temp']:.0f}" if s.get("temp") is not None else "-"
                gs_s = f"{s['gsk']:.1f}" if s.get("gsk") is not None else "-"
                lp_s = f"{s['logp']:.1f}" if s.get("logp") is not None else "-"
                out.append(f" {s['step']} |{s['target']:<8}|{s['solvent']:<14}|{sel_s:>5}|{t_s:>5}|{gs_s:>4}|{lp_s:>4}")
        out.append("")

    # Comparison summary
    out.append("== COMPARISON ==")
    out.append("Scheme|MinSel|AvgSel|#Solvents|Bottleneck")
    out.append("------|------|------|---------|----------")
    for r in schemes:
        valid_steps = [s for s in r["steps"] if s["sel"] is not None and s["sel"] > -900]
        if valid_steps:
            bn = min(valid_steps, key=lambda s: s["sel"])
            out.append(f"{r['tag']:6}|{r['min_sel']:5.0f}%|{r['avg_sel']:5.0f}%|{r['n_solv']:9}|Step {bn['step']}:{bn['target']}")
        else:
            out.append(f"{r['tag']:6}|    ?|    ?|{r['n_solv']:9}|N/A")

    out.append("")
    out.append("SEL=max separation reliability, SAFE=regulatory/green, NRG=lowest operating cost.")

    return "\n".join(out)


@safe_tool_wrapper
async def plan_sequential_separation(
    polymers: str,
    temperature: float = 120.0,
    top_k_solvents: int = 5,
    create_decision_tree: bool = True,
    excluded_solvents: str = "",
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """Plan optimal sequential separation sequences for multiple polymers.

    Args:
        polymers: Comma-separated polymer list (e.g., "LDPE,HDPE,PP,PS")
        temperature: Target temperature in C (default: 120.0)
        top_k_solvents: Top solvents to show per step (default: 5)
        create_decision_tree: Generate decision tree plot (default: True)
        excluded_solvents: Comma-separated solvents to exclude

    WHEN TO USE:
    - "Plan sequential separation for LDPE, HDPE, PP, and PS"
    - "What's the best order to separate mixed plastics?"
    - "Design a multi-step polymer separation process"
    """
    from itertools import permutations

    conn = get_connection()

    # Parse polymers
    polymer_list = [p.strip() for p in polymers.split(",") if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return "Error: Need at least 2 polymers for separation planning."

    # Parse excluded solvents (from feedback loop cost constraints)
    excluded_set: set[str] = set()
    if excluded_solvents:
        excluded_set = {s.strip().lower() for s in excluded_solvents.split(",") if s.strip()}
        logger.info(f"Excluding solvents from feedback loop: {excluded_set}")

    # For >4 polymers, use greedy algorithm instead of exhaustive search
    USE_GREEDY = n_polymers > 4

    if USE_GREEDY:
        return await _greedy_separation_planning(
            polymer_list, temperature, top_k_solvents,
            table_name, polymer_column, solvent_column,
            temperature_column, solubility_column,
        )

    # Generate all permutations (only for <=4 polymers)
    all_sequences = list(permutations(polymer_list))
    n_sequences = len(all_sequences)

    output = [f"# Sequential Separation Planning\n"]
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Number of possible sequences:** {n_polymers}! = {n_sequences}")
    output.append(f"**Temperature:** {temperature} C")
    output.append(f"**Top solvents per step:** {top_k_solvents}")
    if excluded_set:
        output.append(f"**Excluded solvents (cost constraint):** {', '.join(sorted(excluded_set))}")
    output.append("")

    # List all sequences
    output.append("## All Possible Sequences\n")
    for i, seq in enumerate(all_sequences, 1):
        output.append(f"{i}. {' -> '.join(seq)}")
    output.append("")

    # Minimum selectivity threshold for viable separation
    MIN_SELECTIVITY = 5.0

    # Async function to find top-k solvents for separating target from remaining polymers
    async def find_top_solvents(
        target: str,
        remaining: list,
        k: int = 5,
        used_solvents: Optional[set] = None,
        excluded_solvents_inner: Optional[set] = None,
    ) -> list:
        """Find top-k solvents for separating target from remaining polymers.

        Enforces solvent diversity by excluding solvents already used in previous steps.
        Also supports excluding expensive solvents via feedback loops.
        Also finds optimal temperature for comparison.
        """
        if used_solvents is None:
            used_solvents = set()
        if excluded_solvents_inner is None:
            excluded_solvents_inner = set()

        if not remaining:
            return [{"solvent": "N/A", "selectivity": float("inf"), "target_sol": 100, "max_other": 0,
                      "temperature": temperature, "optimal_temp": temperature, "note": "Last polymer - no separation needed"}]

        all_polymers = [target] + remaining

        # Use interpolation model instead of SQL
        from strap.solubility import get_solubility as _get_sol, get_available_solvents_for_polymer as _get_svnts
        import pandas as pd

        solvents_avail = _get_svnts(target)

        # Build DataFrame at user-specified temperature (replaces SQL query)
        rows_spec = []
        for sv in solvents_avail:
            for poly in all_polymers:
                sol = _get_sol(poly, sv, temperature)
                if sol is not None:
                    rows_spec.append({solvent_column: sv, polymer_column: poly, "avg_sol": sol})
        df = pd.DataFrame(rows_spec) if rows_spec else pd.DataFrame(columns=[solvent_column, polymer_column, "avg_sol"])

        # Build DataFrame across all temperatures 25-160°C (replaces SQL query)
        rows_all = []
        for sv in solvents_avail:
            for poly in all_polymers:
                for t in range(25, 161, 5):
                    sol = _get_sol(poly, sv, float(t))
                    if sol is not None and sol > 0:
                        rows_all.append({solvent_column: sv, polymer_column: poly, "temp": float(t), "avg_sol": sol})
        df_all = pd.DataFrame(rows_all) if rows_all else pd.DataFrame(columns=[solvent_column, polymer_column, "temp", "avg_sol"])

        if len(df) == 0:
            return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

        # Find optimal temp-solvent combinations from all temps data
        optimal_by_solvent: Dict[str, dict] = {}
        if len(df_all) > 0:
            for temp in df_all["temp"].unique():
                temp_df = df_all[df_all["temp"] == temp]
                for solvent in temp_df[solvent_column].unique():
                    solvent_data = temp_df[temp_df[solvent_column] == solvent]
                    target_data = solvent_data[solvent_data[polymer_column] == target]
                    if len(target_data) == 0:
                        continue
                    target_sol = target_data["avg_sol"].values[0]
                    other_data = solvent_data[solvent_data[polymer_column].isin(remaining)]
                    max_other = other_data["avg_sol"].max() if len(other_data) > 0 else 0
                    selectivity = target_sol - max_other

                    if solvent not in optimal_by_solvent or selectivity > optimal_by_solvent[solvent]["selectivity"]:
                        optimal_by_solvent[solvent] = {"temp": temp, "selectivity": selectivity}

        # Process results at user-specified temperature
        results = []
        for solvent in df[solvent_column].unique():
            solvent_data = df[df[solvent_column] == solvent]

            target_data = solvent_data[solvent_data[polymer_column] == target]
            if len(target_data) == 0:
                continue
            target_sol = target_data["avg_sol"].values[0]

            other_data = solvent_data[solvent_data[polymer_column].isin(remaining)]
            if len(other_data) == 0:
                max_other = 0
            else:
                max_other = other_data["avg_sol"].max()

            selectivity = target_sol - max_other

            opt_info = optimal_by_solvent.get(solvent, {"temp": temperature, "selectivity": selectivity})

            results.append({
                "solvent": solvent,
                "selectivity": selectivity,
                "target_sol": target_sol,
                "max_other": max_other,
                "temperature": temperature,
                "optimal_temp": opt_info["temp"],
                "optimal_selectivity": opt_info["selectivity"],
            })

        results.sort(key=lambda x: x["selectivity"], reverse=True)

        # Filter out solvents already used in previous steps to ensure diversity
        if used_solvents:
            used_lower = {s.lower() for s in used_solvents}
            unused_results = [r for r in results if r["solvent"].lower() not in used_lower]
            if unused_results:
                results = unused_results
            else:
                for r in results:
                    if r["solvent"].lower() in used_lower:
                        r["reused"] = True

        # Filter out expensive solvents from feedback loop (cost optimisation)
        if excluded_solvents_inner:
            excluded_lower = {s.lower() for s in excluded_solvents_inner}
            filtered_results = [r for r in results if r["solvent"].lower() not in excluded_lower]
            if filtered_results:
                results = filtered_results
                for r in results[:3]:
                    r["feedback_constrained"] = True
            else:
                for r in results:
                    if r["solvent"].lower() in excluded_lower:
                        r["excluded_expensive"] = True

        # Filter by minimum selectivity threshold
        viable_results = [r for r in results if r.get("selectivity", 0) >= MIN_SELECTIVITY]
        if viable_results:
            results = viable_results

        # Add solvent properties if available
        solvent_table = _get_solvent_table_name()
        if solvent_table and results:
            try:
                solvent_names = [r["solvent"] for r in results[:k]]
                prop_lookup = await _lookup_solvent_properties(solvent_names, solvent_table)
                for r in results:
                    if r["solvent"] in prop_lookup:
                        r.update({kk: vv for kk, vv in prop_lookup[r["solvent"]].items() if vv is not None})
            except Exception as e:
                logger.debug(f"Could not fetch solvent properties: {e}")

        return results[:k] if results else [{"solvent": "None found", "selectivity": 0, "target_sol": 0, "max_other": 0}]

    # Async function to analyze a single sequence with solvent diversity tracking
    async def analyze_sequence(sequence, seq_idx):
        """Analyze a single sequence, enforcing different solvents for each step."""
        seq_output: list[str] = []
        seq_output.append(f"### Sequence {seq_idx}: {' -> '.join(sequence)}\n")

        used_solvents: set[str] = set()
        total_min_selectivity = float("inf")
        seq_steps: list[dict] = []

        for step, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step:])
            top_solvents = await find_top_solvents(target, remaining, top_k_solvents, used_solvents, excluded_set)

            if top_solvents and top_solvents[0].get("solvent") not in ["N/A", "No data", "None found", "Error"]:
                used_solvents.add(top_solvents[0]["solvent"])

            seq_output.append(f"**Step {step}: Separate {target} from {{{', '.join(remaining)}}}**")

            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents,
            }
            seq_steps.append(step_data)

            for rank, sol_info in enumerate(top_solvents, 1):
                if "error" in sol_info:
                    seq_output.append(f"  {rank}. Error: {sol_info['error']}")
                elif sol_info.get("solvent") in ["No data", "None found", "No viable solvent"]:
                    seq_output.append(f"  {rank}. No data available")
                else:
                    sel = sol_info.get("selectivity", 0)
                    symbol = "OK" if sel > 10 else "WARN" if sel > 0 else "POOR"
                    reused_marker = " *(REUSED)*" if sol_info.get("reused") else ""
                    target_sol = sol_info.get("target_sol", 0)
                    max_other = sol_info.get("max_other", 0)
                    line = (
                        f"  {rank}. [{symbol}] **{sol_info['solvent']}**{reused_marker}: "
                        f"selectivity={sel:.1f}% (target={target_sol:.1f}%, max_other={max_other:.1f}%)"
                    )

                    props = []
                    if sol_info.get("logp") is not None:
                        toxicity = "Low" if sol_info["logp"] < 0 else "Med" if sol_info["logp"] < 2 else "High"
                        props.append(f"LogP:{sol_info['logp']:.1f}({toxicity})")
                    if sol_info.get("energy") is not None:
                        props.append(f"Energy:{sol_info['energy']:.0f}J/g")
                    if sol_info.get("bp") is not None:
                        props.append(f"BP:{sol_info['bp']:.0f} C")

                    if props:
                        line += f" | {' '.join(props)}"

                    seq_output.append(line)

            if top_solvents and "selectivity" in top_solvents[0]:
                best_selectivity = top_solvents[0]["selectivity"]
                total_min_selectivity = min(total_min_selectivity, best_selectivity)

            seq_output.append("")

        seq_output.append(f"**Step {len(sequence)}: {sequence[-1]} is isolated**\n")

        best_solvents = [
            s["solvents"][0]["solvent"]
            for s in seq_steps
            if s["solvents"] and s["solvents"][0].get("solvent") not in ["N/A", "No data", "None found", "Error"]
        ]
        unique_solvents = set(best_solvents)
        if len(best_solvents) > len(unique_solvents):
            seq_output.append(f"**Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(best_solvents)} steps (some reused)\n")
        else:
            seq_output.append(f"**Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(best_solvents)} steps\n")

        seq_output.append("---\n")

        return {
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps,
            "output": seq_output,
            "unique_solvents": len(unique_solvents),
        }

    # Analyze all sequences in parallel with limited concurrency
    output.append("## Detailed Analysis of Each Sequence\n")
    semaphore = asyncio.Semaphore(10)

    async def analyze_with_limit(sequence, seq_idx):
        async with semaphore:
            return await analyze_sequence(sequence, seq_idx)

    sequence_results = await asyncio.gather(*[
        analyze_with_limit(seq, idx)
        for idx, seq in enumerate(all_sequences, 1)
    ])

    sequence_scores: list[dict] = []
    sequence_details: list[list] = []
    for result in sequence_results:
        sequence_scores.append({
            "sequence": result["sequence"],
            "min_selectivity": result["min_selectivity"],
            "steps": result["steps"],
        })
        sequence_details.append(result["steps"])
        output.extend(result["output"])

    sequence_scores.sort(key=lambda x: x["min_selectivity"], reverse=True)

    output.append("## Sequence Ranking (by worst-case selectivity)\n")
    output.append("*Higher minimum selectivity = more robust separation*\n")

    for rank, score_data in enumerate(sequence_scores[:10], 1):
        seq_str = " -> ".join(score_data["sequence"])
        min_sel = score_data["min_selectivity"]
        symbol = "#1" if rank == 1 else "#2" if rank == 2 else "#3" if rank == 3 else f"{rank}."
        output.append(f"{symbol} **{seq_str}** (min selectivity: {min_sel:.1f}%)")

    output.append("")

    # Create visualisations using shared helpers
    if create_decision_tree and sequence_scores:
        output.append("## Top Recommended Separation Sequence\n")

        try:
            filepath = _plot_separation_sequence(
                polymer_list, sequence_scores[0], temperature,
                total_sequences=len(sequence_scores), rank=1,
            )
            output.append(f"Visualisation saved: {_get_plot_url(filepath)}\n")

            if len(sequence_scores) > 1:
                output.append(f"**Note:** This shows the top-ranked sequence. There are {len(sequence_scores) - 1} other possible sequences.")
                output.append(f"    To view alternatives, ask: 'Show me the 2nd best sequence' or 'Show me {polymer_list[1]}-first separation'")
        except Exception as e:
            logger.error(f"Decision tree error: {e}", exc_info=True)
            output.append(f"Could not create visualisation: {e}")

        # Generate TOP-K COMPARISON visualisation (side-by-side)
        if len(sequence_scores) >= 2:
            try:
                filepath = _plot_topk_comparison(polymer_list, sequence_scores, temperature)
                output.append(f"\n**Top-K Comparison**: {_get_plot_url(filepath)}")
            except Exception as e:
                logger.error(f"Top-K comparison visualisation error: {e}", exc_info=True)

    # Summary recommendations
    output.append("\n## Recommendations\n")
    if sequence_scores and sequence_scores[0]["min_selectivity"] > 10:
        best = sequence_scores[0]
        output.append(f"**Best sequence:** {' -> '.join(best['sequence'])}")
        output.append(f"   - Minimum selectivity: {best['min_selectivity']:.1f}%")
        output.append(f"   - All steps have positive selectivity")
        if len(sequence_scores) > 1:
            output.append(f"\n**Alternative sequences available:** {len(sequence_scores) - 1} more options")
            output.append(f"   Ask to see specific sequences (e.g., 'Show 2nd best' or 'Show {polymer_list[0]}-first')")
    elif sequence_scores:
        output.append("**No sequence has all high-selectivity steps.**")
        output.append("Consider:")
        output.append("  - Exploring different temperatures")
        output.append("  - Using multi-stage extraction")
        output.append("  - Combining solvents")

    display = "\n".join(output)

    # Build structured data for programmatic access (exhaustive search results)
    best_seq = sequence_scores[0] if sequence_scores else {}
    best_steps = best_seq.get("steps", [])
    valid_steps = [s for s in best_steps if s.get("selectivity", -1000) > -900]
    solvents_used = list(set(s.get("solvent", "N/A") for s in valid_steps if s.get("solvent") != "N/A"))

    top_k_sequences = []
    for rank, seq_data in enumerate(sequence_scores[:3], 1):
        seq_steps = seq_data.get("steps", [])
        solvent_mapping: Dict[str, str] = {}
        for step in seq_steps:
            target = step.get("target")
            solvents_list = step.get("solvents", [])
            if solvents_list and isinstance(solvents_list, list):
                best_sol = solvents_list[0].get("solvent") if isinstance(solvents_list[0], dict) else None
                if target and best_sol and best_sol not in ["N/A", "No data", "None found", "Error"]:
                    solvent_mapping[target] = best_sol
        top_k_sequences.append({
            "rank": rank,
            "sequence": seq_data.get("sequence", []),
            "min_selectivity": seq_data.get("min_selectivity", 0),
            "solvent_mapping": solvent_mapping,
        })

    structured_data = {
        "tool_name": "plan_sequential_separation",
        "success": True,
        "polymers_analyzed": polymer_list,
        "best_sequence": list(best_seq.get("sequence", [])),
        "solvents": solvents_used,
        "selectivities": [s.get("selectivity", 0) for s in valid_steps],
        "temperature": temperature,
        "algorithm_used": "exhaustive",
        "excluded_solvents": list(excluded_set) if excluded_set else [],
        "feedback_iteration": len(excluded_set) > 0,
        "steps": [
            {
                "step": i + 1,
                "target": s.get("target", ""),
                "solvent": s["solvents"][0].get("solvent", "") if s.get("solvents") else "",
                "selectivity": s["solvents"][0].get("selectivity", 0) if s.get("solvents") else 0,
            }
            for i, s in enumerate(best_steps)
        ],
        "min_selectivity": min(s.get("selectivity", 0) for s in valid_steps) if valid_steps else None,
        "max_selectivity": max(s.get("selectivity", 0) for s in valid_steps) if valid_steps else None,
        "coverage_complete": len(best_seq.get("sequence", [])) == len(polymer_list),
        "top_k_sequences": top_k_sequences,
        "total_sequences_evaluated": len(sequence_scores),
    }

    return json.dumps({"display": display, "data": structured_data}, ensure_ascii=False)


# ===================================================================
# Tool: analyze_integrated_separation
# ===================================================================

@safe_tool_wrapper
async def analyze_integrated_separation(
    polymers: str,
    rank_by: str = "selectivity",
    top_k: int = 10,
    temperature_min: float = 25.0,
    temperature_max: float = 160.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """Multi-polymer separation analysis with optimal temperatures and integrated properties.

    Args:
        polymers: Comma-separated polymer list (e.g., "LDPE,EVOH,PET,PVC")
        rank_by: Ranking criterion - 'selectivity', 'cost', 'safety', 'toxicity', or 'bp'
        top_k: Top solvents per step (default: 10)
        temperature_min: Min search temperature in C (default: 25)
        temperature_max: Max search temperature in C (default: 160)

    WHEN TO USE:
    - "Find optimal temperatures for separating LDPE, EVOH, and PET"
    - "Rank solvents by safety for polymer separation"
    - "Comprehensive separation analysis with cost and toxicity"
    """
    from itertools import permutations

    conn = get_connection()

    polymer_list = [p.strip().upper() for p in polymers.split(",") if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return "Need at least 2 polymers for separation analysis."

    if n_polymers > 3:
        return (
            f"For {n_polymers} polymers, use `plan_sequential_separation` which uses efficient greedy algorithm. "
            "This exhaustive analysis tool is limited to <=3 polymers."
        )

    # Temperature range from interpolation model (25–160 °C, step 5)
    available_temps = [
        float(t) for t in range(
            max(int(temperature_min), 25),
            min(int(temperature_max), 160) + 1,
            5,
        )
    ]

    if not available_temps:
        return f"No temperature data found between {temperature_min} C and {temperature_max} C"

    output = [f"# Integrated Multi-Polymer Separation Analysis\n"]
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Temperature Range:** {temperature_min} C - {temperature_max} C ({len(available_temps)} temperatures)")
    output.append(f"**Ranking Criterion:** {rank_by}")
    import math
    output.append(f"**Number of Sequences:** {n_polymers}! = {math.factorial(n_polymers)}\n")

    # Helper to get solvent properties including GSK G-score
    async def get_full_properties(solvent_names: list) -> dict:
        """Get all properties for solvents including GSK G-scores."""
        prop_lookup: Dict[str, dict] = {}

        solvent_table = _get_solvent_table_name()
        if solvent_table:
            try:
                prop_dict = await _lookup_solvent_properties(solvent_names, solvent_table)
                if prop_dict:
                    prop_lookup.update(prop_dict)
            except Exception:
                pass

        # Get GSK G-scores
        try:
            placeholders = ", ".join(["?" for _ in solvent_names])
            gscore_query = f"""
            SELECT solvent_common_name, g_score, classification
            FROM gsk_dataset
            WHERE LOWER(solvent_common_name) IN ({placeholders})
            """
            gscore_df = conn.execute(gscore_query, [n.lower() for n in solvent_names]).fetchdf()
            if len(gscore_df) > 0:
                for _, row in gscore_df.iterrows():
                    name = row["solvent_common_name"]
                    for orig_name in solvent_names:
                        if orig_name.lower() == name.lower():
                            if orig_name not in prop_lookup:
                                prop_lookup[orig_name] = {}
                            prop_lookup[orig_name]["g_score"] = row["g_score"]
                            prop_lookup[orig_name]["gsk_class"] = row["classification"]
                            break
        except Exception:
            pass

        return prop_lookup

    # Minimum selectivity threshold
    MIN_SELECTIVITY_THRESHOLD = 5.0

    async def find_optimal_separation(target: str, remaining: list, used_solvents: Optional[set] = None) -> dict:
        """Find the best temperature-solvent combination for separating target from remaining."""
        if used_solvents is None:
            used_solvents = set()

        if not remaining:
            return {
                "solvent": "N/A",
                "temperature": 0,
                "selectivity": float("inf"),
                "target_sol": 100,
                "max_other": 0,
                "note": "Last polymer - no separation needed",
            }

        from strap.solubility import get_solubility as _get_sol, get_available_solvents as _get_svnts

        all_solvents = _get_svnts()

        results = []
        for temp in available_temps:
            for solvent in all_solvents:
                target_sol = _get_sol(target, solvent, temp)
                if target_sol is None or target_sol <= 0:
                    continue

                max_other = 0.0
                for poly in remaining:
                    sol = _get_sol(poly, solvent, temp)
                    if sol is not None and sol > max_other:
                        max_other = sol

                selectivity = target_sol - max_other
                results.append({
                    "solvent": solvent,
                    "temperature": temp,
                    "selectivity": selectivity,
                    "target_sol": target_sol,
                    "max_other": max_other,
                })

        if not results:
            return {"solvent": "None found", "temperature": 0, "selectivity": 0}

        # Filter out solvents already used in previous steps
        if used_solvents:
            unused_results = [r for r in results if r["solvent"].lower() not in {s.lower() for s in used_solvents}]
            if unused_results:
                results = unused_results
            else:
                for r in results:
                    if r["solvent"].lower() in {s.lower() for s in used_solvents}:
                        r["reused_solvent"] = True

        # Filter by minimum selectivity threshold
        results = [r for r in results if r.get("selectivity", 0) >= MIN_SELECTIVITY_THRESHOLD]
        if not results:
            return {
                "solvent": "No viable solvent",
                "temperature": 0,
                "selectivity": 0,
                "note": f"No solvent found with selectivity >= {MIN_SELECTIVITY_THRESHOLD}%",
            }

        # Get properties for all solvents found
        solvent_names = list(set(r["solvent"] for r in results))
        prop_lookup = await get_full_properties(solvent_names)

        for r in results:
            if r["solvent"] in prop_lookup:
                r.update(prop_lookup[r["solvent"]])

        # Sort based on rank_by criterion
        rank_lower = rank_by.lower()
        if rank_lower in ["cost", "energy"]:
            valid = [r for r in results if r.get("selectivity", 0) > 0 and r.get("energy") is not None]
            if valid:
                valid.sort(key=lambda x: x["energy"])
                return valid[0]
        elif rank_lower in ["safety", "gscore", "g_score"]:
            valid = [r for r in results if r.get("selectivity", 0) > 0 and r.get("g_score") is not None]
            if valid:
                valid.sort(key=lambda x: x["g_score"], reverse=True)
                return valid[0]
        elif rank_lower in ["toxicity", "logp"]:
            valid = [r for r in results if r.get("selectivity", 0) > 0 and r.get("logp") is not None]
            if valid:
                valid.sort(key=lambda x: x["logp"])
                return valid[0]
        elif rank_lower in ["bp", "boiling", "boiling_point"]:
            valid = [r for r in results if r.get("selectivity", 0) > 0 and r.get("bp") is not None]
            if valid:
                valid.sort(key=lambda x: x["bp"])
                return valid[0]

        # Default: sort by selectivity (higher = better)
        results.sort(key=lambda x: x.get("selectivity", 0), reverse=True)
        return results[0]

    async def analyze_sequence(sequence: tuple) -> dict:
        """Analyze one separation sequence finding optimal temp for each step."""
        steps: list[dict] = []
        total_score = 0
        used_solvents: set[str] = set()

        for step_idx, target in enumerate(sequence[:-1]):
            remaining = list(sequence[step_idx + 1:])
            best = await find_optimal_separation(target, remaining, used_solvents)

            if best.get("solvent") and best["solvent"] not in ["None found", "No data", "Error", "N/A", "No viable solvent"]:
                used_solvents.add(best["solvent"])

            step_data = {
                "step": step_idx + 1,
                "target": target,
                "remaining": remaining,
                "best": best,
            }
            steps.append(step_data)

            sel = best.get("selectivity", 0)
            if sel != float("inf"):
                total_score += sel

        steps.append({
            "step": len(sequence),
            "target": sequence[-1],
            "remaining": [],
            "best": {"solvent": "N/A", "temperature": 0, "selectivity": float("inf"), "note": "Isolated"},
        })

        min_sel = min(
            s["best"].get("selectivity", 0)
            for s in steps[:-1]
            if s["best"].get("selectivity", 0) != float("inf")
        ) if len(steps) > 1 else 0

        return {
            "sequence": sequence,
            "steps": steps,
            "total_score": total_score,
            "min_selectivity": min_sel,
        }

    # For large polymer sets, use greedy instead of exhaustive permutations
    MAX_EXHAUSTIVE = 6  # 6! = 720, 7! = 5040, 9! = 362880
    USE_GREEDY = n_polymers > MAX_EXHAUSTIVE

    if USE_GREEDY:
        import math
        output.append(f"## Greedy Analysis (n={n_polymers}, {math.factorial(n_polymers):,} permutations avoided)\n")
        output.append("Using greedy algorithm: at each step, select the polymer with the highest selectivity separation.\n")

        # Build a single greedy sequence
        remaining_g = list(polymer_list)
        greedy_sequence: list[str] = []
        greedy_steps: list[dict] = []
        used_solvents_g: set[str] = set()

        while len(remaining_g) > 1:
            best_candidate = None
            best_sel = -float("inf")
            for target in remaining_g:
                others = [p for p in remaining_g if p != target]
                result = await find_optimal_separation(target, others, used_solvents_g)
                sel = result.get("selectivity", 0)
                if sel == float("inf"):
                    sel = 0
                if sel > best_sel:
                    best_sel = sel
                    best_candidate = (target, result)

            target, best = best_candidate
            greedy_sequence.append(target)
            remaining_g.remove(target)
            if best.get("solvent") and best["solvent"] not in ["None found", "No data", "Error", "N/A", "No viable solvent"]:
                used_solvents_g.add(best["solvent"])
            greedy_steps.append({
                "step": len(greedy_sequence),
                "target": target,
                "remaining": remaining_g.copy(),
                "best": best,
            })

        # Add last polymer (isolated)
        greedy_sequence.append(remaining_g[0])
        greedy_steps.append({
            "step": len(greedy_sequence),
            "target": remaining_g[0],
            "remaining": [],
            "best": {"solvent": "N/A", "temperature": 0, "selectivity": float("inf"), "note": "Isolated"},
        })

        min_sel = min(
            s["best"].get("selectivity", 0)
            for s in greedy_steps[:-1]
            if s["best"].get("selectivity", 0) != float("inf")
        ) if len(greedy_steps) > 1 else 0

        all_results = [{
            "sequence": tuple(greedy_sequence),
            "steps": greedy_steps,
            "total_score": sum(s["best"].get("selectivity", 0) for s in greedy_steps[:-1] if s["best"].get("selectivity", 0) != float("inf")),
            "min_selectivity": min_sel,
        }]
    else:
        all_sequences = list(permutations(polymer_list))

        output.append("## Analyzing All Sequences...\n")

        semaphore = asyncio.Semaphore(5)

        async def analyze_with_limit(seq):
            async with semaphore:
                return await analyze_sequence(seq)

        all_results = await asyncio.gather(*[analyze_with_limit(seq) for seq in all_sequences])

        all_results.sort(key=lambda x: x["min_selectivity"], reverse=True)

    # Show top 3 sequences in detail
    output.append("## Top 3 Recommended Separation Sequences\n")

    for rank, result in enumerate(all_results[:3], 1):
        seq = result["sequence"]
        medal = "#1" if rank == 1 else "#2" if rank == 2 else "#3"

        output.append(f"### {medal} Rank #{rank}: {' -> '.join(seq)}")
        output.append(f"**Minimum Selectivity (Bottleneck):** {result['min_selectivity']:.1f}%\n")

        for step in result["steps"][:-1]:
            best = step["best"]
            target = step["target"]
            remaining = step["remaining"]

            sel = best.get("selectivity", 0)
            symbol = "OK" if sel > 30 else "FAIR" if sel > 10 else "WARN" if sel > 0 else "POOR"

            output.append(f"**Step {step['step']}: Separate {target} from {{{', '.join(remaining)}}}**")

            solvent_name = best.get("solvent", "N/A")
            if best.get("reused_solvent"):
                output.append(f"  **Solvent:** {solvent_name} @ **{best.get('temperature', 0):.0f} C** *(REUSED - limited options)*")
            elif best.get("note"):
                output.append(f"  **Solvent:** {solvent_name} - {best.get('note')}")
            else:
                output.append(f"  [{symbol}] **Solvent:** {solvent_name} @ **{best.get('temperature', 0):.0f} C**")
            output.append(f"  - Selectivity: {sel:.1f}% (target: {best.get('target_sol', 0):.1f}%, max_other: {best.get('max_other', 0):.1f}%)")

            props = []
            if best.get("g_score") is not None:
                gs = best["g_score"]
                rating = "Excellent" if gs >= 8 else "Good" if gs >= 6 else "Problematic" if gs >= 4 else "Hazardous"
                props.append(f"G-Score: {gs:.1f}/10 ({rating})")
            if best.get("logp") is not None:
                lp = best["logp"]
                tox = "Low" if lp < 0 else "Medium" if lp < 2 else "High"
                props.append(f"LogP: {lp:.2f} ({tox} toxicity)")
            if best.get("energy") is not None:
                props.append(f"Energy: {best['energy']:.1f} J/g")
            if best.get("bp") is not None:
                props.append(f"BP: {best['bp']:.0f} C")

            if props:
                output.append(f"  - Properties: {' | '.join(props)}")
            output.append("")

        output.append(f"**Step {len(seq)}: {seq[-1]} is isolated**\n")

        solvents_used = [
            s["best"].get("solvent", "N/A")
            for s in result["steps"][:-1]
            if s["best"].get("solvent") not in ["N/A", "None found", "No data", "Error", "No viable solvent"]
        ]
        unique_solvents = set(solvents_used)
        if len(solvents_used) > len(unique_solvents):
            duplicates = [s for s in unique_solvents if solvents_used.count(s) > 1]
            output.append(f"**Warning:** Solvent(s) {duplicates} used multiple times. This may indicate limited data or challenging separation.\n")
        else:
            output.append(f"**Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(solvents_used)} steps\n")

        output.append("---\n")

    # Create visualisation for top sequence
    try:
        best_result = all_results[0]
        seq = best_result["sequence"]
        steps = best_result["steps"][:-1]

        n_steps = len(steps)
        fig_height = max(5 + n_steps * 3.5, 12)
        fig, ax = plt.subplots(figsize=(16, fig_height), dpi=150)

        ax.set_title(
            f'OPTIMAL SEPARATION SEQUENCE: {" -> ".join(seq)}\n'
            + f'Ranked by: {rank_by} | Min Selectivity: {best_result["min_selectivity"]:.1f}%',
            fontsize=18,
            fontweight="bold",
            pad=25,
        )

        ax.set_xlim(0, 14)
        ax.set_ylim(-1.5, n_steps + 3.5)
        ax.axis("off")

        def get_color(selectivity):
            if selectivity > 30:
                return "#2ecc71"
            elif selectivity > 10:
                return "#f1c40f"
            elif selectivity > 0:
                return "#e67e22"
            else:
                return "#e74c3c"

        y_pos = n_steps + 2
        ax.add_patch(plt.Rectangle((1.5, y_pos - 0.5), 11, 1.0,
                                   facecolor="#3498db", edgecolor="black", linewidth=2.5))
        ax.text(7, y_pos, f'MIXTURE: {", ".join(polymer_list)}',
                ha="center", va="center", fontsize=16, fontweight="bold", color="white")

        for idx, step in enumerate(steps):
            y_pos = n_steps + 1 - idx
            best = step["best"]
            target = step["target"]
            remaining = step["remaining"]
            sel = best.get("selectivity", 0)
            temp = best.get("temperature", 0)
            color = get_color(sel)

            ax.annotate("", xy=(3.5, y_pos + 0.4), xytext=(3.5, y_pos + 0.9),
                        arrowprops=dict(arrowstyle="->", lw=3.5, color=color))

            ax.add_patch(plt.Rectangle((1, y_pos - 0.6), 5.5, 1.2,
                                       facecolor=color, edgecolor="black", linewidth=2.5, alpha=0.25))

            ax.add_patch(plt.Circle((1.6, y_pos), 0.35, facecolor=color, edgecolor="black", linewidth=2.5))
            ax.text(1.6, y_pos, str(idx + 1), ha="center", va="center",
                    fontsize=15, fontweight="bold", color="white")

            ax.text(2.4, y_pos + 0.25, f"SEPARATE: {target}",
                    ha="left", va="center", fontsize=14, fontweight="bold")
            ax.text(2.4, y_pos - 0.25, f'From: {", ".join(remaining)}',
                    ha="left", va="center", fontsize=12, color="#333")

            ax.add_patch(plt.Rectangle((7, y_pos - 0.6), 5.5, 1.2,
                                       facecolor="white", edgecolor=color, linewidth=2.5))
            ax.text(9.75, y_pos + 0.25, f'{best.get("solvent", "N/A")}',
                    ha="center", va="center", fontsize=14, fontweight="bold")
            ax.text(9.75, y_pos - 0.15, f"{temp:.0f} C  |  Selectivity: {sel:.1f}%",
                    ha="center", va="center", fontsize=13, color=color, fontweight="bold")

            props_text = []
            if best.get("g_score") is not None:
                props_text.append(f"G-Score: {best['g_score']:.0f}")
            if best.get("energy") is not None:
                props_text.append(f"Energy: {best['energy']:.0f} J/g")
            if best.get("bp") is not None:
                props_text.append(f"BP: {best['bp']:.0f} C")
            if props_text:
                ax.text(9.75, y_pos - 0.52, "  |  ".join(props_text),
                        ha="center", va="top", fontsize=12, fontweight="semibold", color="#222")

        ax.add_patch(plt.Rectangle((1.5, -0.5), 11, 1.0,
                                   facecolor="#2ecc71", edgecolor="black", linewidth=2.5))
        ax.text(7, 0, "ALL POLYMERS SEPARATED",
                ha="center", va="center", fontsize=16, fontweight="bold", color="white")

        legend_elements = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ecc71",
                       markersize=14, label="Excellent (>30%)"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#f1c40f",
                       markersize=14, label="Good (10-30%)"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e67e22",
                       markersize=14, label="Marginal (0-10%)"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c",
                       markersize=14, label="Poor (<0%)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=12,
                  framealpha=0.95, edgecolor="#333", fancybox=True)

        plt.tight_layout(pad=2.0)
        filepath = save_plot(fig, "integrated_separation_analysis")
        plt.close(fig)

        output.append("## Visualisation\n")
        output.append(f"![Separation Sequence]({_get_plot_url(filepath)})\n")

    except Exception as e:
        logger.error(f"Visualisation error: {e}", exc_info=True)
        output.append(f"Could not create visualisation: {e}\n")

    # Summary and recommendations
    output.append("## Summary & Recommendations\n")

    best = all_results[0]
    output.append(f"**Best Sequence:** {' -> '.join(best['sequence'])}")
    output.append(f"**Bottleneck Selectivity:** {best['min_selectivity']:.1f}%\n")

    output.append("**Optimal Conditions per Step:**")
    for step in best["steps"][:-1]:
        b = step["best"]
        output.append(f"  - Step {step['step']}: {step['target']} -> {b.get('solvent', 'N/A')} @ {b.get('temperature', 0):.0f} C")

    if rank_by.lower() == "selectivity":
        output.append(
            "\n**Tip:** Re-run with `rank_by='cost'` for cheapest solvents, "
            "`rank_by='safety'` for safest (highest G-score), or `rank_by='toxicity'` for least toxic (lowest LogP)."
        )

    return "\n".join(output)

# ===================================================================
# Tool: view_alternative_separation_sequence
# ===================================================================

@safe_tool_wrapper
async def view_alternative_separation_sequence(
    polymers: str,
    sequence_rank: Optional[int] = None,
    starting_polymer: Optional[str] = None,
    top_k_solvents: int = 5,
    temperature: float = 120.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """View a specific alternative separation sequence with clear visualisation.

    Use after plan_sequential_separation to explore different sequence options.

    Parameters:
    - polymers: Comma-separated list of polymers (e.g., "LDPE,HDPE,PP,PS")
    - sequence_rank: Rank of sequence to view (1=best, 2=2nd best, etc.)
    - starting_polymer: Name of polymer to start with (alternative to rank)
    - top_k_solvents: Number of top solvents to show per step (default: 5)
    - temperature: Target temperature in C (default: 120.0)

    WHEN TO USE:
    - "Show me the 2nd best separation sequence"
    - "What if we start with PET instead?"
    - "View LDPE-first separation option"
    """
    from itertools import permutations

    conn = get_connection()

    polymer_list = [p.strip() for p in polymers.split(",") if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return "Error: Need at least 2 polymers."

    MAX_EXHAUSTIVE = 6  # 6! = 720 permutations

    async def find_top_solvents(target: str, remaining: list, k: int = 5) -> list:
        """Find top-k solvents for separating target from remaining polymers."""
        if not remaining:
            return [{"solvent": "N/A", "selectivity": float("inf"), "target_sol": 100, "max_other": 0}]

        # Use interpolation model instead of SQL
        from strap.solubility import get_all_solvents_selectivity as _get_all_sel
        all_sel = _get_all_sel(target, remaining, temperature)
        if not all_sel:
            return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

        results = [
            {
                "solvent": entry["solvent"],
                "selectivity": entry["selectivity"],
                "target_sol": entry["target_sol"],
                "max_other": entry["max_other_sol"],
            }
            for entry in all_sel
        ]
        return results[:k]

    async def analyze_sequence(sequence, seq_idx):
        """Analyze single sequence."""
        step_tasks = []
        step_info = []
        for step, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step:])
            step_tasks.append(find_top_solvents(target, remaining, top_k_solvents))
            step_info.append((step, target, remaining))

        all_step_results = await asyncio.gather(*step_tasks)

        total_min_selectivity = float("inf")
        seq_steps: list[dict] = []

        for (step, target, remaining), top_solvents in zip(step_info, all_step_results):
            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents,
            }
            seq_steps.append(step_data)

            if top_solvents and top_solvents[0]["selectivity"] < total_min_selectivity:
                total_min_selectivity = top_solvents[0]["selectivity"]

        return {
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps,
        }

    # For large polymer sets, limit permutation enumeration
    if n_polymers <= MAX_EXHAUSTIVE:
        all_sequences = list(permutations(polymer_list))

        semaphore = asyncio.Semaphore(10)

        async def analyze_with_limit(seq, idx):
            async with semaphore:
                return await analyze_sequence(seq, idx)

        sequence_analyses = await asyncio.gather(*[
            analyze_with_limit(seq, idx) for idx, seq in enumerate(all_sequences, 1)
        ])

        sequence_scores = sorted(sequence_analyses, key=lambda x: x["min_selectivity"], reverse=True)
    elif starting_polymer is not None:
        # Only generate permutations starting with the specified polymer: (n-1)!
        from itertools import permutations as _perms
        starting_polymer_normalized = starting_polymer.strip().upper()
        if starting_polymer_normalized not in [p.upper() for p in polymer_list]:
            return f"Error: '{starting_polymer}' not found in polymer list: {', '.join(polymer_list)}"
        others = [p for p in polymer_list if p.upper() != starting_polymer_normalized]
        start_p = next(p for p in polymer_list if p.upper() == starting_polymer_normalized)
        all_sequences = [tuple([start_p] + list(perm)) for perm in _perms(others)]

        semaphore = asyncio.Semaphore(10)

        async def analyze_with_limit(seq, idx):
            async with semaphore:
                return await analyze_sequence(seq, idx)

        sequence_analyses = await asyncio.gather(*[
            analyze_with_limit(seq, idx) for idx, seq in enumerate(all_sequences, 1)
        ])

        sequence_scores = sorted(sequence_analyses, key=lambda x: x["min_selectivity"], reverse=True)
    else:
        # Greedy approach for large n without starting_polymer
        import math
        from strap.solubility import get_all_solvents_selectivity as _get_all_sel_greedy

        remaining_g = list(polymer_list)
        greedy_seq: list[str] = []
        greedy_steps: list[dict] = []

        while len(remaining_g) > 1:
            best_candidate = None
            best_sel_val = -float("inf")
            for target in remaining_g:
                others = [p for p in remaining_g if p != target]
                all_sel = _get_all_sel_greedy(target, others, temperature)
                top_sel = all_sel[0]["selectivity"] if all_sel else 0
                if top_sel > best_sel_val:
                    best_sel_val = top_sel
                    top_solvents = [
                        {"solvent": e["solvent"], "selectivity": e["selectivity"],
                         "target_sol": e["target_sol"], "max_other": e["max_other_sol"]}
                        for e in (all_sel[:top_k_solvents] if all_sel else [])
                    ]
                    best_candidate = (target, top_solvents)

            if best_candidate is None:
                greedy_seq.extend(remaining_g)
                break
            target, solvents = best_candidate
            greedy_seq.append(target)
            remaining_g.remove(target)
            greedy_steps.append({
                "step": len(greedy_seq),
                "target": target,
                "remaining": remaining_g.copy(),
                "solvents": solvents if solvents else [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}],
            })

        greedy_seq.append(remaining_g[0])
        min_sel = min(
            s["solvents"][0]["selectivity"] for s in greedy_steps if s["solvents"]
        ) if greedy_steps else 0

        sequence_scores = [{
            "sequence": tuple(greedy_seq),
            "min_selectivity": min_sel,
            "steps": greedy_steps,
        }]

    # Find the requested sequence
    target_seq = None
    rank = None

    if sequence_rank is not None:
        if 1 <= sequence_rank <= len(sequence_scores):
            target_seq = sequence_scores[sequence_rank - 1]
            rank = sequence_rank
        else:
            return f"Error: Rank {sequence_rank} is out of range (1-{len(sequence_scores)}). " + (
                f"Only greedy (rank 1) available for {n_polymers} polymers." if n_polymers > MAX_EXHAUSTIVE else ""
            )

    elif starting_polymer is not None:
        starting_polymer_normalized = starting_polymer.strip().upper()
        for idx, seq_data in enumerate(sequence_scores, 1):
            if seq_data["sequence"][0].upper() == starting_polymer_normalized:
                target_seq = seq_data
                rank = idx
                break

        if target_seq is None:
            return f"Error: No sequence found starting with '{starting_polymer}'. Available polymers: {', '.join(polymer_list)}"

    else:
        return "Error: Must specify either sequence_rank or starting_polymer"

    # Generate output with visualisation
    output: list[str] = []
    output.append(f"# Alternative Separation Sequence (Rank #{rank})\n")
    output.append(f"**Sequence:** {' -> '.join(target_seq['sequence'])}")
    output.append(f"**Minimum Selectivity:** {target_seq['min_selectivity']:.1f}%")
    output.append(f"**Temperature:** {temperature} C\n")

    # Create visualisation using shared helper
    try:
        filepath = _plot_separation_sequence(
            polymer_list, target_seq, temperature,
            total_sequences=len(sequence_scores), rank=rank,
        )
        output.append(f"\nVisualisation saved: {_get_plot_url(filepath)}")
    except Exception as e:
        logger.error(f"Visualisation error: {e}", exc_info=True)
        output.append(f"\nCould not create visualisation: {e}")

    # Step details
    output.append("\n## Separation Steps\n")
    for step_data in target_seq["steps"]:
        step_num = step_data["step"]
        target = step_data["target"]
        remaining = step_data["remaining"]
        solvents = step_data["solvents"]

        output.append(f"**Step {step_num}: Separate {target}**")
        if remaining:
            output.append(f"  - Remaining in mixture: {', '.join(remaining)}")
        output.append(f"  - Top solvents:")

        for rank_idx, sol in enumerate(solvents[:3], 1):
            sol_name = sol.get("solvent", "N/A")
            sel = sol.get("selectivity", 0)
            output.append(f"    {rank_idx}. {sol_name} (selectivity: {sel:.1f}%)")
        output.append("")

    # Comparison to best
    if rank > 1:
        best_seq = sequence_scores[0]
        output.append("## Comparison to Best Sequence\n")
        output.append(f"**Best sequence:** {' -> '.join(best_seq['sequence'])} (min selectivity: {best_seq['min_selectivity']:.1f}%)")
        output.append(f"**This sequence:** {' -> '.join(target_seq['sequence'])} (min selectivity: {target_seq['min_selectivity']:.1f}%)")
        output.append(f"**Difference:** {target_seq['min_selectivity'] - best_seq['min_selectivity']:.1f}% selectivity")

    return "\n".join(output)


# ============================================================================
# Introspection Tool
# ============================================================================

@safe_tool_wrapper
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
    from strap.solubility import _load_coefficients, _get_known_names

    _, lookup = _load_coefficients()
    known_polymers, _ = _get_known_names(lookup)

    # Build a dict: polymer -> sorted list of solvents (fitted entries only)
    polymer_solvents: dict[str, list[str]] = {}
    for (polymer, solvent), entry in lookup.items():
        if entry.get("category") == "fitted":
            polymer_solvents.setdefault(polymer, []).append(solvent)

    if not polymer_solvents:
        return "No fitted interpolation coefficients found in the database."

    output = [
        "# Interpolation Coefficient Database — Supported Polymers & Solvents\n",
        f"**Temperature range:** 25–160 °C (extrapolation flagged outside fitted range)\n",
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
    return "\n".join(output)


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
