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
from .precipitation import (
    PrecipitationAnalyzer,
    PrecipitationPoint,
    DifferentialPrecipitationResult,
    MultiPolymerPrecipitationSequence,
    AtmosphericFeasibilityResult,
    MultiPolymerAtmosphericResult,
    format_differential_precipitation_results,
    format_multi_polymer_sequence,
    format_atmospheric_feasibility_results,
    format_multi_polymer_atmospheric_results,
    SOLVENT_BOILING_POINTS,
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
# Differential Precipitation Tools
# =============================================================================

@tool
@safe_tool_wrapper
def find_differential_precipitation_solvents(
    polymer_to_precipitate: str,
    polymer_to_retain: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    top_k: int = 10,
) -> str:
    """Find solvents where one polymer precipitates before another during cooling.

    This tool searches the temperature-dependent solubility database to find solvents
    that enable selective/differential precipitation of polymer mixtures.

    The process works by:
    1. Dissolving both polymers at high temperature
    2. Cooling until the first polymer precipitates (can be filtered out)
    3. Continuing to cool until the second polymer precipitates

    Args:
        polymer_to_precipitate: Polymer that should precipitate FIRST at higher temperature
            (e.g., "EVOH", "LDPE", "PET", "PP", "PS", "HDPE", "PVC", "PC", "Nylon6")
        polymer_to_retain: Polymer that should stay dissolved (precipitates later at lower temp)
        min_temperature_gap: Minimum temperature separation required in Celsius (default: 20)
            Larger gaps give better separation but fewer solvents qualify
        precipitation_threshold: Solubility (%) below which polymer is precipitated, ~0% (default: 1)
        top_k: Number of top results to return (default: 10)

    Returns:
        Ranked list of solvents with precipitation temperatures, gaps, and process recommendations.

    WHEN TO USE:
    - "Find a solvent where EVOH precipitates before LDPE"
    - "What solvent gives the best separation window for PP vs PET?"
    - "Find solvents with at least 30C precipitation gap between HDPE and PS"
    - "Design a cooling-based separation for two polymers"
    """
    conn = get_db_connection()
    analyzer = PrecipitationAnalyzer(conn)

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
            f"No solvents found with {min_temperature_gap}°C gap for {polymer_to_precipitate}/{polymer_to_retain}.\n"
            f"Try reducing min_temperature_gap or checking polymer names.\n"
            f"Available polymers: {', '.join(analyzer.get_available_polymers())}"
        )

    return format_differential_precipitation_results(results)


@tool
@safe_tool_wrapper
def analyze_multi_polymer_precipitation(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Analyze precipitation sequence for multiple polymers in a single solvent.

    This tool determines the order in which polymers will precipitate as a solution
    is cooled, enabling multi-step sequential separation.

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,PP,PET,EVOH")
        solvent: Solvent to analyze (e.g., "toluene", "dimethylformamide")
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 10)

    Returns:
        Ordered precipitation sequence with recommended cooling protocol.

    WHEN TO USE:
    - "What's the precipitation order for PP, PS, HDPE in toluene?"
    - "Design a cooling protocol to separate LDPE, PET, and EVOH using DMF"
    - "How do I sequentially recover 4 polymers by cooling?"
    """
    polymer_list = [p.strip() for p in polymers.split(",")]

    conn = get_db_connection()
    analyzer = PrecipitationAnalyzer(conn)

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


@tool
@safe_tool_wrapper
def analyze_precipitation_temperature(
    polymer: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Analyze precipitation characteristics for a single polymer-solvent pair.

    Returns detailed information about dissolution and precipitation temperatures,
    maximum solubility, and transition behavior.

    Args:
        polymer: Polymer name (e.g., "LDPE", "PET", "EVOH")
        solvent: Solvent name (e.g., "toluene", "dimethylformamide")
        precipitation_threshold: Solubility threshold for precipitation (default: 10%)

    Returns:
        Detailed precipitation analysis including temperatures and solubility curve.

    WHEN TO USE:
    - "What's the precipitation temperature of LDPE in toluene?"
    - "At what temperature does PET dissolve in DMF?"
    - "Show me the solubility profile of EVOH in propanone"
    """
    conn = get_db_connection()
    analyzer = PrecipitationAnalyzer(conn)

    point = analyzer.analyze_precipitation(polymer, solvent, precipitation_threshold)

    if not point:
        return f"No data found for {polymer} in {solvent}."

    # Get full curve for display
    df = analyzer.get_solubility_curve(polymer, solvent)

    lines = [
        f"# Precipitation Analysis: {polymer} in {solvent}\n",
        "## Key Temperatures\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Max Solubility | {point.max_solubility:.1f}% at {point.max_solubility_temp:.0f}°C |",
        f"| Cloud Point (50%) | {point.cloud_point:.0f}°C |" if point.cloud_point else "| Cloud Point | N/A |",
        f"| Precipitation Temp (<{precipitation_threshold}%) | {point.precipitation_temp:.0f}°C |" if point.precipitation_temp else f"| Precipitation Temp | Never below {precipitation_threshold}% |",
        f"| Transition Width | {point.transition_width:.0f}°C |",
        f"| Data Points | {point.data_points} |",
        "\n## Temperature-Solubility Curve\n",
        "| Temp (°C) | Solubility (%) |",
        "|-----------|----------------|",
    ]

    # Show key temperatures from the curve
    for _, row in df.iloc[::3].iterrows():  # Every 3rd point to keep output manageable
        lines.append(f"| {row['temperature']:.0f} | {row['solubility']:.1f} |")

    return "\n".join(lines)


@tool
@safe_tool_wrapper
def plot_precipitation_curves(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Create a visualization of temperature-dependent solubility for multiple polymers.

    Generates a plot showing how solubility changes with temperature for each polymer,
    highlighting precipitation temperatures and separation windows.

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,EVOH")
        solvent: Solvent to analyze
        precipitation_threshold: Threshold line to draw (default: 10%)

    Returns:
        Path to saved plot file and summary of key findings.

    WHEN TO USE:
    - "Plot LDPE vs EVOH solubility in toluene"
    - "Visualize the precipitation curves for PP, PS, PET"
    - "Show me a graph of temperature-dependent solubility"
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    polymer_list = [p.strip() for p in polymers.split(",")]

    conn = get_db_connection()
    analyzer = PrecipitationAnalyzer(conn)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    precip_temps = {}

    for i, polymer in enumerate(polymer_list):
        df = analyzer.get_solubility_curve(polymer, solvent)
        if df.empty:
            continue

        color = colors[i % len(colors)]
        ax.plot(df['temperature'], df['solubility'], '-o', color=color,
                label=polymer, linewidth=2, markersize=4)

        # Find and mark precipitation temperature
        precip_temp = analyzer.find_precipitation_temperature(polymer, solvent, precipitation_threshold)
        if precip_temp:
            precip_temps[polymer] = precip_temp
            ax.axvline(x=precip_temp, color=color, linestyle=':', alpha=0.7)
            ax.annotate(f'{polymer}\n{precip_temp:.0f}°C', xy=(precip_temp, precipitation_threshold + 5),
                       fontsize=8, color=color, ha='center')

    # Add threshold line
    ax.axhline(y=precipitation_threshold, color='gray', linestyle='--', alpha=0.5,
               label=f'Precipitation threshold ({precipitation_threshold}%)')

    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Solubility (%)', fontsize=11)
    ax.set_title(f'Temperature-Dependent Solubility in {solvent.upper()}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 170)
    ax.set_ylim(0, 105)

    # Save plot
    plots_dir = os.environ.get('PLOTS_DIR', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{plots_dir}/precipitation_curves_{solvent}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Build summary
    lines = [
        f"# Precipitation Curves: {', '.join(polymer_list)} in {solvent.upper()}\n",
        f"**Plot saved:** `{filename}`\n",
        "## Precipitation Temperatures\n",
        "| Polymer | Precip Temp |",
        "|---------|-------------|",
    ]

    for polymer, temp in sorted(precip_temps.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {polymer} | {temp:.0f}°C |")

    if len(precip_temps) >= 2:
        temps = list(precip_temps.values())
        max_gap = max(temps) - min(temps)
        lines.append(f"\n**Maximum Temperature Gap:** {max_gap:.0f}°C")

    return "\n".join(lines)


@tool
@safe_tool_wrapper
def plot_atmospheric_feasibility(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Visualize multi-polymer differential precipitation with atmospheric feasibility.

    Creates a plot showing:
    1. Solubility curves for all polymers vs temperature
    2. Solvent boiling point line (red dashed) - critical 1 atm constraint
    3. Precipitation temperatures for each polymer
    4. Shaded "atmospheric operation zone" if feasible
    5. Clear indication of whether process works without pressurization

    Args:
        polymers: Comma-separated list of polymers (e.g., "HDPE,LDPE,PP" or "LDPE,EVOH")
        solvent: Solvent to analyze (must have boiling point data)
        precipitation_threshold: Solubility threshold for precipitation (default: 1%)

    Returns:
        Path to saved plot and atmospheric feasibility summary.

    WHEN TO USE:
    - After check_atmospheric_feasibility or check_multi_polymer_atmospheric_feasibility
    - "Plot the atmospheric feasibility for LDPE/EVOH in DMF"
    - "Visualize whether we can separate these polymers at 1 atm"
    - "Show me the precipitation curves with boiling point"
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    from datetime import datetime

    polymer_list = [p.strip().upper() for p in polymers.split(",")]

    conn = get_db_connection()
    if conn is None:
        return "Error: Database connection not available"

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
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    precip_temps = {}
    max_solubilities = {}
    all_temps = []

    for i, polymer in enumerate(polymer_list):
        df = analyzer.get_solubility_curve(polymer, solvent)
        if df.empty:
            continue

        color = colors[i % len(colors)]
        ax.plot(df['temperature'], df['solubility'], '-o', color=color,
                label=polymer, linewidth=2.5, markersize=5, alpha=0.9)

        all_temps.extend(df['temperature'].tolist())
        max_solubilities[polymer] = df['solubility'].max()

        # Find precipitation temperature
        precip_temp = analyzer.find_precipitation_temperature(polymer, solvent, precipitation_threshold)
        if precip_temp:
            precip_temps[polymer] = precip_temp
            ax.axvline(x=precip_temp, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
            # Annotate precipitation point
            ax.scatter([precip_temp], [precipitation_threshold], color=color, s=100, zorder=5, marker='v')

    if not precip_temps:
        plt.close()
        return f"Error: No precipitation data found for {', '.join(polymer_list)} in {solvent}"

    # Determine x-axis range
    min_temp = min(all_temps) if all_temps else 20
    max_temp = max(all_temps) if all_temps else 160
    x_max = max(max_temp + 20, bp + 30)

    # Add boiling point line (critical constraint)
    ax.axvline(x=bp, color='red', linestyle='--', linewidth=2.5, label=f'Boiling Point ({bp}°C)')

    # Add precipitation threshold line
    ax.axhline(y=precipitation_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(min_temp + 2, precipitation_threshold + 2, f'Precip. threshold ({precipitation_threshold}%)',
            fontsize=9, color='gray')

    # Calculate dissolution temperature needed
    max_precip_temp = max(precip_temps.values())
    dissolution_temp = max_precip_temp + 20

    # Determine if feasible at atmospheric pressure
    is_feasible = dissolution_temp < bp

    # Add shaded regions
    if is_feasible:
        # Green zone: atmospheric operation possible
        ax.axvspan(min_temp, bp, alpha=0.1, color='green', label='Atmospheric zone')
        ax.axvline(x=dissolution_temp, color='green', linestyle='-.', linewidth=1.5, alpha=0.7)
        ax.text(dissolution_temp + 1, 90, f'Dissolution\n~{dissolution_temp:.0f}°C', fontsize=9, color='green')
        feasibility_text = f"✅ FEASIBLE AT 1 ATM\nMargin: {bp - dissolution_temp:.0f}°C below BP"
        text_color = 'green'
    else:
        # Red zone: requires pressurization
        ax.axvspan(bp, x_max, alpha=0.15, color='red', label='Requires pressure')
        ax.axvline(x=dissolution_temp, color='orange', linestyle='-.', linewidth=1.5, alpha=0.7)
        ax.text(dissolution_temp + 1, 90, f'Dissolution\n~{dissolution_temp:.0f}°C', fontsize=9, color='orange')
        feasibility_text = f"❌ REQUIRES PRESSURIZATION\nNeeds {dissolution_temp - bp:.0f}°C above BP"
        text_color = 'red'

    # Add feasibility annotation box
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=text_color, alpha=0.9)
    ax.text(0.98, 0.98, feasibility_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props, color=text_color, fontweight='bold')

    # Precipitation sequence annotation
    sorted_precip = sorted(precip_temps.items(), key=lambda x: x[1], reverse=True)
    seq_text = "Precipitation sequence:\n" + " → ".join([f"{p}@{t:.0f}°C" for p, t in sorted_precip])
    ax.text(0.02, 0.02, seq_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Solubility (%)', fontsize=12)
    ax.set_title(f'Atmospheric Feasibility: {", ".join(polymer_list)} in {solvent.upper()}\n'
                 f'(Boiling Point: {bp}°C at 1 atm)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_temp - 5, x_max)
    ax.set_ylim(0, 105)

    # Save plot
    plots_dir = os.environ.get('PLOTS_DIR', 'plots')
    subdir = f"{plots_dir}/atmospheric_feasibility"
    os.makedirs(subdir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    polymer_str = "_".join(polymer_list)
    filename = f"{subdir}/{polymer_str}_{solvent}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Build summary
    lines = [
        f"# Atmospheric Feasibility Visualization\n",
        f"**Plot saved:** `{filename}`\n",
        f"## System: {', '.join(polymer_list)} in {solvent.upper()}\n",
        f"**Solvent Boiling Point:** {bp}°C at 1 atm",
        f"**Dissolution Temperature Needed:** ~{dissolution_temp:.0f}°C\n",
    ]

    if is_feasible:
        lines.append(f"## ✅ Feasible at Atmospheric Pressure")
        lines.append(f"Safety margin: {bp - dissolution_temp:.0f}°C below boiling point\n")
    else:
        lines.append(f"## ❌ Requires Pressurization")
        lines.append(f"Would need to operate {dissolution_temp - bp:.0f}°C above boiling point\n")

    lines.append("## Precipitation Sequence (during cooling)\n")
    lines.append("| Order | Polymer | Precip Temp | Max Solubility |")
    lines.append("|-------|---------|-------------|----------------|")

    for i, (polymer, temp) in enumerate(sorted_precip, 1):
        max_sol = max_solubilities.get(polymer, 0)
        lines.append(f"| {i} | {polymer} | {temp:.0f}°C | {max_sol:.1f}% |")

    # Temperature gaps
    if len(sorted_precip) >= 2:
        lines.append("\n## Temperature Gaps")
        for i in range(len(sorted_precip) - 1):
            p1, t1 = sorted_precip[i]
            p2, t2 = sorted_precip[i + 1]
            gap = t1 - t2
            lines.append(f"- {p1} → {p2}: **{gap:.0f}°C**")

    return "\n".join(lines)


@tool
@safe_tool_wrapper
def compare_polymer_pairs_precipitation(
    polymer_pairs: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
) -> str:
    """Compare differential precipitation feasibility for multiple polymer pairs.

    This tool evaluates multiple polymer pairs and determines which is most feasible
    for selective precipitation separation. It tries BOTH precipitation orders for
    each pair (A before B and B before A).

    Args:
        polymer_pairs: Semicolon-separated polymer pairs (e.g., "LDPE,PET;LDPE,EVOH;PP,PS")
        min_temperature_gap: Minimum temperature gap required (default: 20°C)
        precipitation_threshold: Solubility below which polymer is precipitated (default: 1%)

    Returns:
        Comparison of all polymer pairs with feasibility ranking.

    WHEN TO USE:
    - "Compare LDPE/PET vs LDPE/EVOH for differential precipitation"
    - "Which polymer pair is easier to separate by cooling?"
    - "Evaluate precipitation feasibility for multiple polymer combinations"
    """
    conn = get_db_connection()
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
    lines.append(f"Minimum temperature gap: {min_temperature_gap}°C\n")

    lines.append("## Summary\n")
    lines.append("| Pair | Solvents Found | Best Order | Max Gap |")
    lines.append("|------|----------------|------------|---------|")
    for r in results:
        lines.append(f"| {r['pair']} | {len(r['best_results'])} | {r['best_order']} | {r['max_gap']:.0f}°C |")

    lines.append("\n## Recommendation\n")
    if results and results[0]["best_results"]:
        best = results[0]
        lines.append(f"**Most feasible pair:** {best['pair']}")
        lines.append(f"- **{len(best['best_results'])} solvents** found with ≥{min_temperature_gap}°C gap")
        lines.append(f"- **Best order:** {best['best_order']}")
        lines.append(f"- **Maximum temperature gap:** {best['max_gap']:.0f}°C")

        lines.append("\n### Top Solvents:\n")
        lines.append("| Solvent | Temp Gap | First Precip | Second Precip |")
        lines.append("|---------|----------|--------------|---------------|")
        for sol in best["best_results"][:5]:
            lines.append(
                f"| {sol.solvent} | {sol.temperature_gap:.0f}°C | "
                f"{sol.polymer_first} @ {sol.polymer_first_precip_temp:.0f}°C | "
                f"{sol.polymer_second} @ {sol.polymer_second_precip_temp:.0f}°C |"
            )
    else:
        lines.append("No feasible pairs found with the specified temperature gap.")
        lines.append(f"Try reducing min_temperature_gap below {min_temperature_gap}°C.")

    # Add comparison for non-feasible pairs
    non_feasible = [r for r in results if not r["best_results"]]
    if non_feasible:
        lines.append("\n## Non-Feasible Pairs\n")
        for r in non_feasible:
            lines.append(f"- **{r['pair']}**: No solvents with ≥{min_temperature_gap}°C gap found")

    return "\n".join(lines)


@tool
@safe_tool_wrapper
def check_atmospheric_feasibility(
    polymer1: str,
    polymer2: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    min_solubility: float = 30.0,
) -> str:
    """Check if differential precipitation is feasible at atmospheric pressure (1 atm).

    This tool analyzes whether the entire differential precipitation process
    (dissolution → cooling → sequential precipitation) can be performed below
    the solvent's boiling point, eliminating the need for pressurized equipment.

    Key concept: Many solvents have boiling points lower than the dissolution
    temperature needed for polymer separation. This tool identifies solvents
    where atmospheric operation IS possible.

    Args:
        polymer1: First polymer name (e.g., "LDPE", "EVOH", "PU")
        polymer2: Second polymer name
        min_temperature_gap: Minimum temperature gap required for separation (default: 20°C)
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 1%)
        min_solubility: Minimum max solubility required for both polymers (default: 30%)

    Returns:
        Markdown table showing:
        - Which solvents work at atmospheric pressure
        - Boiling points and safety margins
        - Process recommendations for feasible solvents
        - Which solvents would require pressurization
    """
    conn = get_db_connection()
    if conn is None:
        return "Error: Database connection not available"

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
                f"with ≥{min_temperature_gap}°C gap. Try:\n"
                f"- Reducing min_temperature_gap (currently {min_temperature_gap}°C)\n"
                f"- Reducing min_solubility threshold (currently {min_solubility}%)\n"
                f"- Checking if both polymers have solubility data in the database"
            )

        return format_atmospheric_feasibility_results(results, include_infeasible=True)

    except Exception as e:
        logger.error(f"Error in atmospheric feasibility check: {e}")
        return f"Error analyzing atmospheric feasibility: {str(e)}"


@tool
@safe_tool_wrapper
def check_multi_polymer_atmospheric_feasibility(
    polymers: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    min_solubility: float = 30.0,
) -> str:
    """Check if multi-polymer differential precipitation is feasible at atmospheric pressure.

    This tool analyzes sequential precipitation of 2 or more polymers at 1 atm.
    For N polymers, it finds solvents where:
    1. All N polymers dissolve below the solvent's boiling point
    2. Each polymer precipitates at a different temperature during cooling
    3. Temperature gaps between consecutive precipitations are sufficient

    Example: For EVOH/PVC/LDPE, might find a solvent where:
    - EVOH precipitates at 100°C (first during cooling)
    - PVC precipitates at 75°C (second)
    - LDPE precipitates at 50°C (last)
    - All below solvent BP of 150°C

    Args:
        polymers: Comma-separated polymer names (e.g., "EVOH,PVC,LDPE" or "LDPE,HDPE")
                  Minimum 2 polymers required. Order doesn't matter - tool determines
                  precipitation order automatically.
        min_temperature_gap: Minimum temperature gap between consecutive precipitations (default: 20°C)
        precipitation_threshold: Solubility (%) below which polymer is precipitated (default: 1%)
        min_solubility: Minimum max solubility required for each polymer (default: 30%)

    Returns:
        Markdown report showing:
        - Which solvents work at atmospheric pressure for all polymers
        - Precipitation sequence (which polymer precipitates first, second, etc.)
        - Temperature gaps between each precipitation step
        - Step-by-step cooling protocol
        - Which solvents would require pressurization
    """
    conn = get_db_connection()
    if conn is None:
        return "Error: Database connection not available"

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
                f"with ≥{min_temperature_gap}°C gaps between each step.\n\n"
                f"**Suggestions:**\n"
                f"- Reduce min_temperature_gap (currently {min_temperature_gap}°C)\n"
                f"- Reduce min_solubility threshold (currently {min_solubility}%)\n"
                f"- Check polymer names are valid\n\n"
                f"**Available polymers:** {', '.join(available)}"
            )

        return format_multi_polymer_atmospheric_results(results, include_infeasible=True)

    except Exception as e:
        logger.error(f"Error in multi-polymer atmospheric feasibility check: {e}")
        return f"Error analyzing multi-polymer atmospheric feasibility: {str(e)}"


# =============================================================================
# Antisolvent Precipitation Tools
# =============================================================================

@tool
@safe_tool_wrapper
def find_antisolvents(
    polymer: str,
    max_solubility: float = 1.0,
    temperature: float = 25.0,
) -> str:
    """Find antisolvents for a polymer - solvents with near-zero solubility at room temperature.

    Antisolvents are solvents where the polymer has very low/no solubility. When added to
    a polymer solution, antisolvents induce precipitation by reducing overall solvent quality.

    This is useful for:
    - Antisolvent precipitation processes
    - Finding non-solvents for polymer recovery
    - Identifying solvents to avoid for dissolution

    Args:
        polymer: Polymer name (e.g., "LDPE", "PET", "PP")
        max_solubility: Maximum solubility threshold (%) to qualify as antisolvent (default: 1%)
        temperature: Temperature to check solubility at (default: 25°C room temp)

    Returns:
        List of antisolvents ranked by how effectively they reject the polymer (lowest solubility first).
    """
    conn = get_db_connection()
    if conn is None:
        return "Error: Database connection not available"

    try:
        # Query for low-solubility solvents at the specified temperature
        # Try both column name formats (original vs DuckDB sanitized)
        queries = [
            # DuckDB sanitized column names
            f"""
            SELECT solvent,
                   solubility____ as solubility,
                   temperature___c_ as temp
            FROM common_solvents_database
            WHERE UPPER(polymer) = UPPER('{polymer}')
            AND temperature___c_ BETWEEN {temperature - 5} AND {temperature + 5}
            AND solubility____ <= {max_solubility}
            ORDER BY solubility____ ASC
            """,
            # Original column names with quotes
            f"""
            SELECT "Solvent" as solvent,
                   "Solubility (%)" as solubility,
                   "Temperature (°C)" as temp
            FROM common_solvents_database
            WHERE UPPER("Polymer") = UPPER('{polymer}')
            AND "Temperature (°C)" BETWEEN {temperature - 5} AND {temperature + 5}
            AND "Solubility (%)" <= {max_solubility}
            ORDER BY "Solubility (%)" ASC
            """,
        ]

        df = None
        for query in queries:
            try:
                df = conn.execute(query).fetchdf()
                if not df.empty:
                    break
            except Exception:
                continue

        if df is None or df.empty:
            return (
                f"No antisolvents found for {polymer} with solubility < {max_solubility}% at {temperature}°C.\n\n"
                f"Try increasing max_solubility threshold or checking a different temperature."
            )

        # Deduplicate by solvent name
        df = df.drop_duplicates(subset=['solvent'])

        lines = [
            f"# Antisolvents for {polymer.upper()}\n",
            f"Solvents with solubility < {max_solubility}% at ~{temperature}°C\n",
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
            lines.append(f"| {i+1} | {row['solvent']} | {sol_str} | {row['temp']:.0f}°C |")

        lines.append("\n## Usage")
        lines.append("These solvents can be used as antisolvents to precipitate "
                    f"{polymer} from solution by adding them to a dissolved polymer mixture.")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error finding antisolvents: {e}")
        return f"Error finding antisolvents: {str(e)}"


@tool
@safe_tool_wrapper
def find_antisolvent_pairs(
    polymer: str,
    min_good_solubility: float = 50.0,
    max_antisolvent_solubility: float = 1.0,
) -> str:
    """Find good solvent + antisolvent pairs for antisolvent precipitation.

    Identifies combinations where:
    1. Good solvent: High solubility at elevated temperature (for dissolution)
    2. Antisolvent: Near-zero solubility at room temperature (to induce precipitation)

    The process: Dissolve polymer in good solvent at high temp, then add antisolvent
    to precipitate the polymer.

    Args:
        polymer: Polymer name (e.g., "LDPE", "PET", "EVOH")
        min_good_solubility: Minimum solubility (%) for good solvent classification (default: 50%)
        max_antisolvent_solubility: Maximum solubility (%) for antisolvent classification (default: 1%)

    Returns:
        Table of good solvent + antisolvent combinations with process recommendations.
    """
    conn = get_db_connection()
    if conn is None:
        return "Error: Database connection not available"

    try:
        # Find good solvents (high solubility at any temperature)
        # Try sanitized column names first, then original
        good_queries = [
            f"""
            SELECT solvent,
                   MAX(solubility____) as max_solubility,
                   MAX(temperature___c_) as dissolution_temp
            FROM common_solvents_database
            WHERE UPPER(polymer) = UPPER('{polymer}')
            GROUP BY solvent
            HAVING MAX(solubility____) >= {min_good_solubility}
            ORDER BY max_solubility DESC
            """,
            f"""
            SELECT "Solvent" as solvent,
                   MAX("Solubility (%)") as max_solubility,
                   MAX("Temperature (°C)") as dissolution_temp
            FROM common_solvents_database
            WHERE UPPER("Polymer") = UPPER('{polymer}')
            GROUP BY "Solvent"
            HAVING MAX("Solubility (%)") >= {min_good_solubility}
            ORDER BY max_solubility DESC
            """,
        ]

        good_solvents = None
        for gq in good_queries:
            try:
                good_solvents = conn.execute(gq).fetchdf()
                if not good_solvents.empty:
                    break
            except Exception:
                continue

        # Find antisolvents (near-zero solubility at room temp)
        anti_queries = [
            f"""
            SELECT solvent,
                   MIN(solubility____) as min_solubility,
                   temperature___c_ as temp
            FROM common_solvents_database
            WHERE UPPER(polymer) = UPPER('{polymer}')
            AND temperature___c_ <= 30
            GROUP BY solvent, temperature___c_
            HAVING MIN(solubility____) <= {max_antisolvent_solubility}
            ORDER BY min_solubility ASC
            """,
            f"""
            SELECT "Solvent" as solvent,
                   MIN("Solubility (%)") as min_solubility,
                   "Temperature (°C)" as temp
            FROM common_solvents_database
            WHERE UPPER("Polymer") = UPPER('{polymer}')
            AND "Temperature (°C)" <= 30
            GROUP BY "Solvent", "Temperature (°C)"
            HAVING MIN("Solubility (%)") <= {max_antisolvent_solubility}
            ORDER BY min_solubility ASC
            """,
        ]

        antisolvents = None
        for aq in anti_queries:
            try:
                antisolvents = conn.execute(aq).fetchdf()
                if not antisolvents.empty:
                    break
            except Exception:
                continue

        if good_solvents is None or good_solvents.empty:
            return f"No good solvents found for {polymer} with solubility > {min_good_solubility}%"

        if antisolvents is None or antisolvents.empty:
            return f"No antisolvents found for {polymer} with solubility < {max_antisolvent_solubility}%"

        # Deduplicate antisolvents
        antisolvents = antisolvents.drop_duplicates(subset=['solvent'])

        lines = [
            f"# Antisolvent Precipitation Pairs for {polymer.upper()}\n",
            f"## Good Solvents (for dissolution)\n",
            f"Solvents with >{min_good_solubility}% solubility:\n",
            "| Good Solvent | Max Solubility | Dissolution Temp |",
            "|--------------|----------------|------------------|",
        ]

        for _, row in good_solvents.head(10).iterrows():
            lines.append(f"| {row['solvent']} | {row['max_solubility']:.1f}% | {row['dissolution_temp']:.0f}°C |")

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

        # Simple heuristic: pair polar antisolvents with polar solvents, etc.
        polar_antisolvents = ['h2o', 'water', 'methanol', 'ethanol', 'glycol', 'propyleneglycol']
        nonpolar_solvents = ['hexane', 'n-heptane', 'cyclohexane', 'toluene', 'benzene', 'dodecane']

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
            process = f"Dissolve at {rec['good_temp']:.0f}°C, add {rec['anti']} to precipitate"
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


@tool
@safe_tool_wrapper
def analyze_selective_antisolvent_precipitation(
    polymers: str,
    antisolvent: str = "auto",
) -> str:
    """Analyze selective antisolvent precipitation for separating multiple polymers.

    When multiple polymers are dissolved together, adding an antisolvent can selectively
    precipitate one polymer before another if they have different solubility responses.

    This tool finds conditions where:
    1. Both/all polymers dissolve in a good solvent
    2. Adding antisolvent precipitates one polymer while keeping another dissolved
    3. Further antisolvent addition precipitates the remaining polymer(s)

    Args:
        polymers: Comma-separated list of polymers (e.g., "LDPE,PET" or "LDPE,PP,HDPE")
        antisolvent: Specific antisolvent to analyze, or "auto" to find best options

    Returns:
        Analysis of selective antisolvent precipitation feasibility with process recommendations.
    """
    conn = get_db_connection()
    if conn is None:
        return "Error: Database connection not available"

    polymer_list = [p.strip().upper() for p in polymers.split(',')]

    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers for selective precipitation analysis"

    try:
        # For each polymer, get solubility in potential antisolvents at room temp
        results = {}
        for polymer in polymer_list:
            queries = [
                # DuckDB sanitized column names
                f"""
                SELECT solvent,
                       solubility____ as solubility
                FROM common_solvents_database
                WHERE UPPER(polymer) = UPPER('{polymer}')
                AND temperature___c_ <= 30
                ORDER BY solubility____ ASC
                """,
                # Original column names
                f"""
                SELECT "Solvent" as solvent,
                       "Solubility (%)" as solubility
                FROM common_solvents_database
                WHERE UPPER("Polymer") = UPPER('{polymer}')
                AND "Temperature (°C)" <= 30
                ORDER BY "Solubility (%)" ASC
                """,
            ]

            df = None
            for query in queries:
                try:
                    df = conn.execute(query).fetchdf()
                    if not df.empty:
                        break
                except Exception:
                    continue

            if df is not None and not df.empty:
                df = df.drop_duplicates(subset=['solvent'])
                results[polymer] = dict(zip(df['solvent'], df['solubility']))

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
            lines.append("## ⚠️ No Differential Antisolvents Found\n")
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
                lines.append(f"**Precipitation order** (by antisolvent tolerance):\n")

                for i, (polymer, sol) in enumerate(sorted_by_sol, 1):
                    if sol < 0.01:
                        lines.append(f"{i}. **{polymer}** - precipitates first (solubility: {sol:.2e}%)")
                    else:
                        lines.append(f"{i}. **{polymer}** - precipitates {'last' if i == len(sorted_by_sol) else 'next'} (solubility: {sol:.3f}%)")

                lines.append(f"\n**Process:**")
                lines.append(f"1. Dissolve all polymers in a common good solvent at elevated temperature")
                lines.append(f"2. Cool to moderate temperature (~50-60°C)")
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
    # Differential precipitation tools
    "find_differential_precipitation_solvents",
    "analyze_multi_polymer_precipitation",
    "analyze_precipitation_temperature",
    "plot_precipitation_curves",
    "plot_atmospheric_feasibility",
    "compare_polymer_pairs_precipitation",
    "check_atmospheric_feasibility",
    "check_multi_polymer_atmospheric_feasibility",
    # Antisolvent precipitation tools
    "find_antisolvents",
    "find_antisolvent_pairs",
    "analyze_selective_antisolvent_precipitation",
    # Tool collection
    "ADVANCED_SEPARATION_TOOLS",
]
