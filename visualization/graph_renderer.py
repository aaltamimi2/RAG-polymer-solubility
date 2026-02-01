"""
Workflow Graph Renderer for Multi-Agent Traces

Renders NetworkX graphs showing agent interactions:
- Nodes: agents (colored by type using Tableau 10 palette)
- Edges: handoffs (width = duration, color = success/failure)
- Export: SVG, PNG, PDF
"""

import io
import logging
from typing import Optional, Dict, Any, Tuple, List

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

logger = logging.getLogger(__name__)

# Tableau 10 colorblind-friendly palette
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

# Default color for unknown agents
DEFAULT_AGENT_COLOR = "#BAB0AC"  # Gray

# Edge colors
SUCCESS_EDGE_COLOR = "#59A14F"  # Green
FAILURE_EDGE_COLOR = "#E15759"  # Red


class WorkflowGraphRenderer:
    """
    Renders workflow graphs from execution traces.

    Uses NetworkX for graph structure and Matplotlib for rendering.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100,
        font_size: int = 10,
        node_size: int = 2000,
        edge_width_scale: float = 5.0,
    ):
        """
        Initialize the renderer.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for raster output
            font_size: Font size for labels
            node_size: Size of agent nodes
            edge_width_scale: Scale factor for edge widths
        """
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "NetworkX and Matplotlib required for graph rendering. "
                "Install with: pip install networkx matplotlib"
            )

        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size
        self.node_size = node_size
        self.edge_width_scale = edge_width_scale

    def render(
        self,
        trace: Dict[str, Any],
        format: str = "svg",
        show_legend: bool = True,
        title: Optional[str] = None,
    ) -> bytes:
        """
        Render a workflow graph from a trace.

        Args:
            trace: StoredTrace dict or similar structure
            format: Output format ("svg", "png", "pdf")
            show_legend: Whether to include color legend
            title: Optional title for the graph

        Returns:
            Bytes of the rendered image
        """
        # Build the graph
        G = self._build_graph(trace)

        if len(G.nodes()) == 0:
            # Return empty placeholder if no data
            return self._render_empty_graph(format)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Layout
        pos = self._compute_layout(G)

        # Draw nodes
        self._draw_nodes(G, pos, ax)

        # Draw edges
        self._draw_edges(G, pos, ax)

        # Draw labels
        self._draw_labels(G, pos, ax)

        # Title
        if title:
            ax.set_title(title, fontsize=self.font_size + 2, fontweight="bold")
        elif "trace_id" in trace:
            ax.set_title(f"Trace: {trace['trace_id']}", fontsize=self.font_size + 2)

        # Legend
        if show_legend:
            self._add_legend(G, ax)

        # Clean up axes
        ax.set_axis_off()
        ax.margins(0.2)

        # Render to bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()

    def _build_graph(self, trace: Dict[str, Any]) -> "nx.DiGraph":
        """Build a directed graph from trace data."""
        G = nx.DiGraph()

        # Add nodes from agents_visited
        agents_visited = trace.get("agents_visited", [])
        for agent in agents_visited:
            G.add_node(agent, color=self._get_agent_color(agent))

        # Add edges from handoff_metrics
        handoff_metrics = trace.get("handoff_metrics", [])
        for handoff in handoff_metrics:
            if isinstance(handoff, dict):
                from_agent = handoff.get("from_agent")
                to_agent = handoff.get("to_agent")
                duration_ms = handoff.get("duration_ms", 0) or 0
                success = handoff.get("success", True)

                if from_agent and to_agent:
                    # Ensure nodes exist
                    if from_agent not in G:
                        G.add_node(from_agent, color=self._get_agent_color(from_agent))
                    if to_agent not in G:
                        G.add_node(to_agent, color=self._get_agent_color(to_agent))

                    G.add_edge(
                        from_agent,
                        to_agent,
                        duration_ms=duration_ms,
                        success=success,
                    )

        # If no handoffs but we have agents, create a simple chain
        if len(G.edges()) == 0 and len(agents_visited) > 1:
            for i in range(len(agents_visited) - 1):
                G.add_edge(agents_visited[i], agents_visited[i + 1], duration_ms=0, success=True)

        return G

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for an agent."""
        # Try exact match
        if agent_name in AGENT_COLORS:
            return AGENT_COLORS[agent_name]

        # Try partial match
        agent_lower = agent_name.lower()
        for key, color in AGENT_COLORS.items():
            if key in agent_lower or agent_lower in key:
                return color

        return DEFAULT_AGENT_COLOR

    def _compute_layout(self, G: "nx.DiGraph") -> Dict[str, Tuple[float, float]]:
        """Compute node positions."""
        if len(G.nodes()) <= 1:
            return {n: (0, 0) for n in G.nodes()}

        # Use spring layout for general graphs
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except Exception:
            # Fallback to shell layout
            pos = nx.shell_layout(G)

        return pos

    def _draw_nodes(
        self, G: "nx.DiGraph", pos: Dict, ax: plt.Axes
    ) -> None:
        """Draw graph nodes."""
        colors = [G.nodes[n].get("color", DEFAULT_AGENT_COLOR) for n in G.nodes()]

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=colors,
            node_size=self.node_size,
            alpha=0.9,
            edgecolors="white",
            linewidths=2,
        )

    def _draw_edges(
        self, G: "nx.DiGraph", pos: Dict, ax: plt.Axes
    ) -> None:
        """Draw graph edges with width based on duration."""
        if len(G.edges()) == 0:
            return

        # Separate edges by success
        success_edges = []
        failure_edges = []
        edge_widths = {}

        max_duration = max(
            (G.edges[e].get("duration_ms", 0) or 0 for e in G.edges()), default=1
        )
        max_duration = max(max_duration, 1)  # Avoid division by zero

        for edge in G.edges():
            duration = G.edges[edge].get("duration_ms", 0) or 0
            width = 1 + (duration / max_duration) * self.edge_width_scale
            edge_widths[edge] = width

            if G.edges[edge].get("success", True):
                success_edges.append(edge)
            else:
                failure_edges.append(edge)

        # Draw success edges
        if success_edges:
            widths = [edge_widths[e] for e in success_edges]
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=success_edges,
                width=widths,
                edge_color=SUCCESS_EDGE_COLOR,
                alpha=0.7,
                arrows=True,
                arrowsize=20,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
            )

        # Draw failure edges
        if failure_edges:
            widths = [edge_widths[e] for e in failure_edges]
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=failure_edges,
                width=widths,
                edge_color=FAILURE_EDGE_COLOR,
                alpha=0.7,
                arrows=True,
                arrowsize=20,
                arrowstyle="-|>",
                style="dashed",
                connectionstyle="arc3,rad=0.1",
            )

    def _draw_labels(
        self, G: "nx.DiGraph", pos: Dict, ax: plt.Axes
    ) -> None:
        """Draw node labels."""
        # Create shortened labels
        labels = {}
        for node in G.nodes():
            # Shorten common prefixes
            label = node.replace("collab_", "").replace("_agent", "").replace("_", "\n")
            labels[node] = label.title()

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            labels=labels,
            font_size=self.font_size,
            font_weight="bold",
            font_color="white",
        )

    def _add_legend(self, G: "nx.DiGraph", ax: plt.Axes) -> None:
        """Add a color legend."""
        # Get unique agent types in the graph
        agent_types = set()
        for node in G.nodes():
            for key in AGENT_COLORS:
                if key in node.lower():
                    agent_types.add(key)
                    break
            else:
                agent_types.add("other")

        # Create legend patches
        patches = []
        for agent_type in sorted(agent_types):
            color = AGENT_COLORS.get(agent_type, DEFAULT_AGENT_COLOR)
            label = agent_type.replace("_", " ").title()
            patches.append(mpatches.Patch(color=color, label=label))

        # Add edge legend items
        patches.append(mpatches.Patch(color=SUCCESS_EDGE_COLOR, label="Success"))
        patches.append(mpatches.Patch(color=FAILURE_EDGE_COLOR, label="Failure"))

        ax.legend(
            handles=patches,
            loc="upper left",
            fontsize=self.font_size - 2,
            framealpha=0.9,
        )

    def _render_empty_graph(self, format: str) -> bytes:
        """Render a placeholder for empty graphs."""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=self.dpi)
        ax.text(
            0.5, 0.5,
            "No workflow data available",
            ha="center", va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_axis_off()

        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()


# Global renderer instance with default settings
_default_renderer = None


def render_workflow_graph(
    trace: Dict[str, Any],
    format: str = "svg",
    show_legend: bool = True,
    title: Optional[str] = None,
) -> bytes:
    """
    Render a workflow graph from a trace.

    Convenience function using default renderer settings.

    Args:
        trace: StoredTrace dict or similar structure
        format: Output format ("svg", "png", "pdf")
        show_legend: Whether to include color legend
        title: Optional title for the graph

    Returns:
        Bytes of the rendered image
    """
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = WorkflowGraphRenderer()

    return _default_renderer.render(
        trace=trace,
        format=format,
        show_legend=show_legend,
        title=title,
    )
