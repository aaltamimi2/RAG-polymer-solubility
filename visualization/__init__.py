"""
Visualization Module for DISSOLVE Multi-Agent System

Provides visualization capabilities for execution traces:
- Workflow graphs showing agent interactions
- Timeline/Gantt charts of agent execution
- Routing decision visualizations
- Publication-quality static figure export

Usage:
    from visualization import (
        render_workflow_graph,
        render_timeline,
        render_routing_decision,
        export_figure,
    )

    # Render workflow graph
    svg_bytes = render_workflow_graph(trace)

    # Render timeline
    png_bytes = render_timeline(trace, format="png")

    # Export publication-quality figure
    export_figure(trace, output_path="figure.pdf", dpi=300)

Note: Requires matplotlib and networkx for full functionality.
"""

import logging

logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import matplotlib
    import networkx
    HAS_VISUALIZATION_DEPS = True
except ImportError:
    HAS_VISUALIZATION_DEPS = False
    logger.warning(
        "Visualization dependencies not installed. "
        "Install with: pip install matplotlib networkx"
    )

# Tableau 10 colorblind-friendly palette (always available)
AGENT_COLORS = {
    "router": "#4E79A7",           # Blue
    "separation": "#F28E2B",       # Orange
    "tea_lca": "#E15759",          # Red
    "literature": "#76B7B2",       # Teal
    "smart_aggregator": "#59A14F", # Green
    "fast_agent": "#EDC948",       # Yellow
    "standard_agent": "#B07AA1",   # Purple
    "orchestrator": "#9C755F",     # Brown
    "collab_separation_agent": "#F28E2B",  # Orange (same as separation)
    "collab_tea_agent": "#E15759",         # Red (same as tea_lca)
    "integrated_orchestrator": "#9C755F",  # Brown
}

# Conditional imports based on dependencies
if HAS_VISUALIZATION_DEPS:
    from .graph_renderer import (
        WorkflowGraphRenderer,
        render_workflow_graph,
    )
    from .timeline_renderer import (
        TimelineRenderer,
        render_timeline,
    )
    from .routing_renderer import (
        RoutingRenderer,
        render_routing_decision,
    )
    from .static_figures import (
        StaticFigureExporter,
        export_figure,
        export_combined_figure,
    )

    __all__ = [
        # Graph rendering
        "WorkflowGraphRenderer",
        "render_workflow_graph",
        "AGENT_COLORS",
        # Timeline rendering
        "TimelineRenderer",
        "render_timeline",
        # Routing rendering
        "RoutingRenderer",
        "render_routing_decision",
        # Static export
        "StaticFigureExporter",
        "export_figure",
        "export_combined_figure",
        # Dependency flag
        "HAS_VISUALIZATION_DEPS",
    ]
else:
    # Provide stub functions that raise ImportError
    def render_workflow_graph(*args, **kwargs):
        raise ImportError("Visualization dependencies not installed. Install matplotlib and networkx.")

    def render_timeline(*args, **kwargs):
        raise ImportError("Visualization dependencies not installed. Install matplotlib and networkx.")

    def render_routing_decision(*args, **kwargs):
        raise ImportError("Visualization dependencies not installed. Install matplotlib and networkx.")

    def export_figure(*args, **kwargs):
        raise ImportError("Visualization dependencies not installed. Install matplotlib and networkx.")

    def export_combined_figure(*args, **kwargs):
        raise ImportError("Visualization dependencies not installed. Install matplotlib and networkx.")

    __all__ = [
        "AGENT_COLORS",
        "render_workflow_graph",
        "render_timeline",
        "render_routing_decision",
        "export_figure",
        "export_combined_figure",
        "HAS_VISUALIZATION_DEPS",
    ]
