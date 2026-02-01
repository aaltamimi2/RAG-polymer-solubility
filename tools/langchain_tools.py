"""
LangChain Tool Wrappers for Polymer Separation Tools

This module provides @tool decorated functions that wrap the modular
separation algorithms, optimization utilities, and analysis functions
for use by the multi-agent system.

Usage:
    from tools.langchain_tools import (
        find_optimal_separation_sequence,
        optimize_separation_temperature,
        rank_solvents_for_separation,
    )

    # Add to agent's tool list
    tools = [find_optimal_separation_sequence, ...]
"""

import asyncio
import functools
import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool

# Import modular tools
from .separation import (
    GreedySeparator,
    DPSeparator,
    BranchAndBoundSeparator,
    find_best_separation,
    SeparationSequence,
    SeparationResult,
)
from .optimization import (
    TemperatureOptimizer,
    ThroughputAnalyzer,
    OptimizationResult,
)
from .analysis import (
    SelectivityCalculator,
    SolventRanker,
    PolymerCompatibilityMatrix,
    SelectivityMetrics,
)
from .visualization import (
    SeparationTreeVisualizer,
    SelectivityHeatmap,
    ProcessFlowDiagram,
    PlotConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Database Connection Management
# =============================================================================

_db_connection = None


def set_db_connection(conn):
    """Set database connection for testing or custom usage."""
    global _db_connection
    _db_connection = conn


def get_db_connection():
    """Get database connection - uses global sql_db by default."""
    global _db_connection
    if _db_connection is not None:
        return _db_connection
    try:
        from agent_sql_final_1212_patched import sql_db
        return sql_db.conn
    except ImportError:
        raise RuntimeError(
            "No database connection available. "
            "Either import agent_sql_final_1212_patched first, "
            "or call set_db_connection() with a valid connection."
        )


# =============================================================================
# Safe Tool Wrapper (matches existing pattern)
# =============================================================================

MAX_OUTPUT_LENGTH = 15000


def safe_tool_wrapper(func):
    """Decorator for error handling and output truncation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)

            # Truncate if too long
            if isinstance(result, str) and len(result) > MAX_OUTPUT_LENGTH:
                result = result[:MAX_OUTPUT_LENGTH] + "\n\n... (output truncated)"

            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            return f"Error in {func.__name__}: {str(e)}"

    return wrapper


# =============================================================================
# Helper Functions
# =============================================================================

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
            # If loop is running (e.g., in Jupyter), use nest_asyncio pattern
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(coro)


def format_separation_result(result: SeparationResult) -> str:
    """Format SeparationResult as readable markdown."""
    seq = result.best_sequence

    output = [
        "# Optimal Separation Sequence\n",
        f"**Algorithm:** {result.algorithm}",
        f"**Computation Time:** {result.computation_time_ms:.1f}ms",
        f"**Nodes Explored:** {result.nodes_explored}\n",
    ]

    # Sequence summary
    sequence_str = " -> ".join(s.target_polymer for s in seq.steps)
    output.append(f"**Sequence:** {sequence_str}")
    output.append(f"**Status:** {seq.status.value}")
    output.append(f"**Minimum Selectivity:** {seq.min_selectivity:.1f}%")
    output.append(f"**Average Selectivity:** {seq.avg_selectivity:.1f}%")
    output.append(f"**Unique Solvents:** {len(seq.unique_solvents)}\n")

    # Step-by-step breakdown
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


def format_optimization_result(result: OptimizationResult) -> str:
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


def format_selectivity_metrics(metrics: SelectivityMetrics) -> str:
    """Format SelectivityMetrics as readable markdown."""
    status = "VIABLE" if metrics.is_viable else "NOT VIABLE"

    output = [
        "# Selectivity Analysis\n",
        f"**Target Polymer:** {metrics.target_polymer}",
        f"**Other Polymers:** {', '.join(metrics.other_polymers)}",
        f"**Solvent:** {metrics.solvent}",
        f"**Temperature:** {metrics.temperature}C\n",
        "## Results\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Selectivity | {metrics.selectivity:.1f}% |",
        f"| Target Solubility | {metrics.target_solubility:.1f}% |",
        f"| Max Other Solubility | {metrics.max_other_solubility:.1f}% |",
        f"| Selectivity Ratio | {metrics.selectivity_ratio:.2f}x |",
        f"| Data Confidence | {metrics.confidence:.1%} |",
        f"| Status | **{status}** |",
    ]

    return "\n".join(output)


# =============================================================================
# Separation Algorithm Tools
# =============================================================================

@tool
@safe_tool_wrapper
def find_optimal_separation_sequence(
    polymers: str,
    temperature: float = 120.0,
    algorithm: str = "auto",
) -> str:
    """Find the optimal sequence to separate multiple polymers.

    Uses sophisticated algorithms to maximize selectivity across all steps:
    - greedy: Fast O(n^2) heuristic, good for quick estimates
    - dp: Optimal dynamic programming, best for n <= 10 polymers
    - branch_and_bound: Optimal with pruning, good for 6-12 polymers
    - auto: Automatically chooses best algorithm based on polymer count

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,HDPE,PET,PP")
        temperature: Target separation temperature in Celsius (default: 120)
        algorithm: Algorithm choice - "greedy", "dp", "branch_and_bound", or "auto"

    Returns:
        Detailed separation sequence with solvents, selectivities, and metrics.

    WHEN TO USE:
    - "What's the best order to separate LDPE, HDPE, PET, and PP?"
    - "Optimize separation sequence for 5 polymers at 100C"
    - "Compare greedy vs optimal separation for multilayer film"
    """
    polymer_list = parse_polymer_list(polymers)

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers for separation planning."

    if len(polymer_list) > 12:
        return f"Error: Too many polymers ({len(polymer_list)}). Maximum 12 for computational feasibility."

    conn = get_db_connection()
    result = run_async(find_best_separation(polymer_list, conn, temperature, algorithm))

    return format_separation_result(result)


@tool
@safe_tool_wrapper
def compare_separation_algorithms(
    polymers: str,
    temperature: float = 120.0,
) -> str:
    """Compare greedy vs optimal algorithms for polymer separation.

    Runs both greedy and DP/Branch-and-Bound algorithms and compares:
    - Sequence order differences
    - Total selectivity scores
    - Computation time
    - Recommendation for this polymer set

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,HDPE,PET,PP")
        temperature: Target separation temperature in Celsius (default: 120)

    Returns:
        Comparison of algorithm results with recommendation.

    WHEN TO USE:
    - "Is the greedy solution good enough for these polymers?"
    - "Compare fast vs optimal separation planning"
    """
    polymer_list = parse_polymer_list(polymers)

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers for separation planning."

    if len(polymer_list) > 10:
        return "Error: Comparison requires n <= 10 polymers for DP algorithm."

    conn = get_db_connection()

    # Run both algorithms
    greedy_result = run_async(find_best_separation(polymer_list, conn, temperature, "greedy"))
    optimal_result = run_async(find_best_separation(polymer_list, conn, temperature, "dp"))

    # Compare
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


# =============================================================================
# Temperature Optimization Tools
# =============================================================================

@tool
@safe_tool_wrapper
def optimize_separation_temperature(
    target_polymer: str,
    other_polymers: str,
    solvent: str,
    temp_min: float = 25.0,
    temp_max: float = 180.0,
) -> str:
    """Find optimal temperature window for separating target polymer from others.

    Scans temperature range to find where selectivity is maximized.
    Returns temperature windows, energy scores, and feasibility assessment.

    Args:
        target_polymer: Polymer to dissolve (e.g., "LDPE")
        other_polymers: Polymers to NOT dissolve, comma-separated (e.g., "HDPE,PP")
        solvent: Solvent to analyze (e.g., "xylene")
        temp_min: Minimum temperature to scan (default: 25)
        temp_max: Maximum temperature to scan (default: 180)

    Returns:
        Optimal temperature, viable windows, and recommendations.

    WHEN TO USE:
    - "What temperature should I use to dissolve LDPE but not HDPE in xylene?"
    - "Find optimal temperature window for PET separation"
    - "Is 100C good enough for PS dissolution in toluene?"
    """
    others = parse_polymer_list(other_polymers)
    conn = get_db_connection()

    optimizer = TemperatureOptimizer(conn)
    result = run_async(optimizer.find_optimal_temperature(
        target_polymer=target_polymer.strip().upper(),
        other_polymers=others,
        solvent=solvent.strip(),
        temp_range=(temp_min, temp_max),
    ))

    return format_optimization_result(result)


@tool
@safe_tool_wrapper
def analyze_sequence_throughput(
    polymers: str,
    temperature: float = 120.0,
    base_rate_kg_hr: float = 100.0,
) -> str:
    """Analyze throughput and identify bottlenecks in a separation sequence.

    First finds the optimal separation sequence, then estimates dissolution
    rates for each step and identifies the rate-limiting step.

    Args:
        polymers: Comma-separated list of polymers
        temperature: Operating temperature in Celsius
        base_rate_kg_hr: Base dissolution rate in kg/hr (default: 100)

    Returns:
        Throughput analysis with bottleneck identification and recommendations.

    WHEN TO USE:
    - "What's the throughput for separating these 5 polymers?"
    - "Which step is the bottleneck in the separation?"
    - "How can I improve separation speed?"
    """
    polymer_list = parse_polymer_list(polymers)
    conn = get_db_connection()

    # Get optimal sequence first
    sep_result = run_async(find_best_separation(polymer_list, conn, temperature, "greedy"))

    # Build steps for throughput analysis
    steps = [
        {
            "polymer": s.target_polymer,
            "solvent": s.solvent,
            "temperature": s.temperature,
            "selectivity": s.selectivity,
        }
        for s in sep_result.best_sequence.steps
    ]

    analyzer = ThroughputAnalyzer(conn, base_rate_kg_hr)
    result = analyzer.analyze_sequence_throughput(steps)

    output = [
        "# Throughput Analysis\n",
        f"**Polymers:** {', '.join(polymer_list)}",
        f"**Base Rate:** {base_rate_kg_hr} kg/hr\n",
        "## Overall Results\n",
        f"- **Effective Throughput:** {result['overall_rate']:.1f} kg/hr",
        f"- **Bottleneck Step:** Step {result['bottleneck_step']}",
        f"- **Improvement Potential:** {result['improvement_potential']:.1%}\n",
        "## Step-by-Step Rates\n",
    ]

    for sr in result['step_rates']:
        marker = " (BOTTLENECK)" if sr.get('limiting') else ""
        output.append(f"- Step {sr['step']} ({sr['polymer']}): {sr['rate']:.1f} kg/hr{marker}")

    if result.get('recommendations'):
        output.append("\n## Recommendations\n")
        for rec in result['recommendations']:
            output.append(f"- {rec}")

    return "\n".join(output)


# =============================================================================
# Analysis Tools
# =============================================================================

@tool
@safe_tool_wrapper
def calculate_selectivity_detailed(
    target_polymer: str,
    other_polymers: str,
    solvent: str,
    temperature: float = 100.0,
) -> str:
    """Calculate detailed selectivity metrics for a separation.

    Returns comprehensive metrics including selectivity value and ratio,
    solubilities, confidence score, and viability assessment.

    Args:
        target_polymer: Polymer to dissolve
        other_polymers: Polymers to NOT dissolve, comma-separated
        solvent: Solvent to use
        temperature: Temperature in Celsius

    Returns:
        Detailed selectivity metrics and viability assessment.

    WHEN TO USE:
    - "What's the selectivity for dissolving LDPE vs HDPE in xylene at 100C?"
    - "Is cyclohexane selective enough for PET separation?"
    - "Check if toluene works for PS vs LDPE separation"
    """
    others = parse_polymer_list(other_polymers)
    conn = get_db_connection()

    calc = SelectivityCalculator(conn)
    metrics = calc.calculate(
        target=target_polymer.strip().upper(),
        others=others,
        solvent=solvent.strip(),
        temperature=temperature,
    )

    return format_selectivity_metrics(metrics)


@tool
@safe_tool_wrapper
def rank_solvents_for_separation(
    target_polymer: str,
    other_polymers: str,
    temperature: float = 100.0,
    top_k: int = 10,
) -> str:
    """Rank solvents for separation using multi-criteria scoring.

    Considers selectivity, safety, environmental impact, and cost.
    Returns ranked list with detailed scores for each criterion.

    Args:
        target_polymer: Polymer to dissolve
        other_polymers: Polymers to NOT dissolve, comma-separated
        temperature: Temperature in Celsius
        top_k: Number of top solvents to return (default: 10)

    Returns:
        Ranked list of solvents with scores for each criterion.

    WHEN TO USE:
    - "What's the greenest solvent for separating LDPE from HDPE?"
    - "Rank solvents by safety for PET dissolution"
    - "Find cost-effective solvents for PP separation"
    """
    others = parse_polymer_list(other_polymers)
    conn = get_db_connection()

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
        "| Rank | Solvent | Overall | Selectivity | Safety | Environmental | Cost |",
        "|------|---------|---------|-------------|--------|---------------|------|",
    ]

    for i, score in enumerate(scores, 1):
        output.append(
            f"| {i} | {score.solvent} | {score.overall_score:.2f} | "
            f"{score.selectivity_score:.2f} | {score.safety_score:.2f} | "
            f"{score.environmental_score:.2f} | {score.cost_score:.2f} |"
        )

    # Add notes for top solvent
    if scores and scores[0].notes:
        output.append(f"\n**Notes for {scores[0].solvent}:**")
        for note in scores[0].notes:
            output.append(f"- {note}")

    return "\n".join(output)


@tool
@safe_tool_wrapper
def build_compatibility_matrix(
    polymers: str,
    solvents: str = "",
    temperature: float = 100.0,
) -> str:
    """Build polymer-solvent compatibility matrix.

    Returns a matrix showing solubility of each polymer in each solvent.
    Useful for understanding separation feasibility at a glance.

    Args:
        polymers: Comma-separated list of polymers
        solvents: Comma-separated list of solvents (optional, auto-detects if empty)
        temperature: Temperature in Celsius

    Returns:
        Compatibility matrix with solubility percentages.

    WHEN TO USE:
    - "Show compatibility matrix for LDPE, HDPE, PET in common solvents"
    - "What solvents work for which polymers?"
    - "Build solubility matrix for 5 polymers"
    """
    polymer_list = parse_polymer_list(polymers)
    solvent_list = parse_solvent_list(solvents)
    conn = get_db_connection()

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


@tool
@safe_tool_wrapper
def find_challenging_polymer_pairs(
    polymers: str,
    temperature: float = 100.0,
    selectivity_threshold: float = 10.0,
) -> str:
    """Identify polymer pairs that are difficult to separate.

    Returns pairs where the best achievable selectivity is below threshold.
    Helps identify separation challenges upfront before planning.

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius
        selectivity_threshold: Minimum acceptable selectivity (default: 10)

    Returns:
        List of challenging pairs with best achievable selectivity.

    WHEN TO USE:
    - "Which polymer pairs in this mixture are hard to separate?"
    - "Are any polymers in this set too similar?"
    - "Identify separation challenges for this film composition"
    """
    polymer_list = parse_polymer_list(polymers)
    conn = get_db_connection()

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


# =============================================================================
# Visualization Tools
# =============================================================================

@tool
@safe_tool_wrapper
def create_separation_tree_plot(
    polymers: str,
    temperature: float = 120.0,
    algorithm: str = "greedy",
) -> str:
    """Create a decision tree visualization for separation sequences.

    Shows the separation path with selectivity values at each step.
    Highlights the optimal sequence.

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius
        algorithm: Algorithm to use for finding sequences

    Returns:
        Path to saved plot file and summary.

    WHEN TO USE:
    - "Visualize separation options for LDPE, HDPE, PET, PP"
    - "Show decision tree for polymer separation"
    - "Create separation diagram"
    """
    polymer_list = parse_polymer_list(polymers)
    conn = get_db_connection()

    # Get separation result
    result = run_async(find_best_separation(polymer_list, conn, temperature, algorithm))

    # Create visualization
    config = PlotConfig(output_dir="plots")
    viz = SeparationTreeVisualizer(config)
    filepath = viz.create_tree([result.best_sequence])

    output = [
        "# Separation Tree Visualization\n",
        f"**Plot saved to:** `{filepath}`\n",
        f"**Sequence:** {' -> '.join(s.target_polymer for s in result.best_sequence.steps)}",
        f"**Min Selectivity:** {result.best_sequence.min_selectivity:.1f}%",
    ]

    return "\n".join(output)


@tool
@safe_tool_wrapper
def create_selectivity_heatmap(
    polymers: str,
    solvents: str = "",
    temperature: float = 100.0,
) -> str:
    """Create heatmap showing polymer-solvent solubility.

    Visualizes the compatibility matrix with color-coding.
    Green = high solubility, Red = low solubility.

    Args:
        polymers: Comma-separated list of polymers
        solvents: Comma-separated list of solvents (optional)
        temperature: Temperature in Celsius

    Returns:
        Path to saved plot file.

    WHEN TO USE:
    - "Create solubility heatmap for these polymers"
    - "Visualize polymer-solvent compatibility"
    """
    polymer_list = parse_polymer_list(polymers)
    solvent_list = parse_solvent_list(solvents)
    conn = get_db_connection()

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
    config = PlotConfig(output_dir="plots")
    viz = SelectivityHeatmap(config)
    filepath = viz.create_polymer_solvent_heatmap(matrix)

    return f"# Selectivity Heatmap\n\n**Plot saved to:** `{filepath}`"


@tool
@safe_tool_wrapper
def create_process_flow_diagram(
    polymers: str,
    temperature: float = 120.0,
) -> str:
    """Create a process flow diagram for the optimal separation sequence.

    Shows the complete separation process with:
    - Input feed stream
    - Separation units with solvents and temperatures
    - Output product streams
    - Selectivity at each step

    Args:
        polymers: Comma-separated list of polymers
        temperature: Temperature in Celsius

    Returns:
        Path to saved plot file and summary.

    WHEN TO USE:
    - "Create PFD for polymer separation process"
    - "Visualize the separation workflow"
    - "Generate process diagram"
    """
    polymer_list = parse_polymer_list(polymers)
    conn = get_db_connection()

    # Get separation result
    result = run_async(find_best_separation(polymer_list, conn, temperature, "greedy"))

    # Create visualization
    config = PlotConfig(output_dir="plots")
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


# =============================================================================
# Tool Collection
# =============================================================================

# All tools in this module
ADVANCED_SEPARATION_TOOLS = [
    find_optimal_separation_sequence,
    compare_separation_algorithms,
    optimize_separation_temperature,
    analyze_sequence_throughput,
    calculate_selectivity_detailed,
    rank_solvents_for_separation,
    build_compatibility_matrix,
    find_challenging_polymer_pairs,
    create_separation_tree_plot,
    create_selectivity_heatmap,
    create_process_flow_diagram,
]

__all__ = [
    # Database utilities
    "set_db_connection",
    "get_db_connection",
    # Tools
    "find_optimal_separation_sequence",
    "compare_separation_algorithms",
    "optimize_separation_temperature",
    "analyze_sequence_throughput",
    "calculate_selectivity_detailed",
    "rank_solvents_for_separation",
    "build_compatibility_matrix",
    "find_challenging_polymer_pairs",
    "create_separation_tree_plot",
    "create_selectivity_heatmap",
    "create_process_flow_diagram",
    # Tool collection
    "ADVANCED_SEPARATION_TOOLS",
]
