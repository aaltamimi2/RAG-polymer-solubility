"""
Polymer Separation Tools Library

Modular tools for polymer separation planning, optimization, and analysis.
Designed to work with the multi-agent system and SQL database.

Modules:
    - separation: Algorithms for polymer separation sequence optimization
    - optimization: Temperature and throughput optimization utilities
    - visualization: Advanced plotting and visualization helpers
    - analysis: Data analysis and validation utilities
"""

from .separation import (
    SeparationStep,
    SeparationSequence,
    SeparationResult,
    GreedySeparator,
    DPSeparator,
    BranchAndBoundSeparator,
)

from .optimization import (
    TemperatureOptimizer,
    ThroughputAnalyzer,
    OptimizationResult,
)

from .visualization import (
    SeparationTreeVisualizer,
    SelectivityHeatmap,
    ProcessFlowDiagram,
)

from .analysis import (
    SelectivityCalculator,
    SolventRanker,
    PolymerCompatibilityMatrix,
)

# LangChain tool wrappers for agent integration
from .langchain_tools import (
    # Database utilities
    set_db_connection,
    get_db_connection,
    # Separation tools
    find_optimal_separation_sequence,
    compare_separation_algorithms,
    # Optimization tools
    optimize_separation_temperature,
    analyze_sequence_throughput,
    # Analysis tools
    calculate_selectivity_detailed,
    rank_solvents_for_separation,
    build_compatibility_matrix,
    find_challenging_polymer_pairs,
    # Visualization tools
    create_separation_tree_plot,
    create_selectivity_heatmap,
    create_process_flow_diagram,
    # Tool collection
    ADVANCED_SEPARATION_TOOLS,
)

__version__ = "0.1.0"
__all__ = [
    # Separation
    "SeparationStep",
    "SeparationSequence",
    "SeparationResult",
    "GreedySeparator",
    "DPSeparator",
    "BranchAndBoundSeparator",
    # Optimization
    "TemperatureOptimizer",
    "ThroughputAnalyzer",
    "OptimizationResult",
    # Visualization
    "SeparationTreeVisualizer",
    "SelectivityHeatmap",
    "ProcessFlowDiagram",
    # Analysis
    "SelectivityCalculator",
    "SolventRanker",
    "PolymerCompatibilityMatrix",
    # LangChain Tools
    "set_db_connection",
    "get_db_connection",
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
    "ADVANCED_SEPARATION_TOOLS",
]
