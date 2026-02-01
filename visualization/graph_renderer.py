"""
Workflow Graph Renderer for Multi-Agent Traces

Renders execution flow diagrams showing:
- Agents as nodes in execution order (left to right)
- Handoffs as labeled edges with duration and tools
- Color coding by agent type (Tableau 10 palette)
"""

import io
import logging
from typing import Optional, Dict, Any, Tuple, List

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    import numpy as np
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

logger = logging.getLogger(__name__)

# Tableau 10 colorblind-friendly palette
AGENT_COLORS = {
    "router": "#4E79A7",           # Blue
    "separation": "#F28E2B",       # Orange
    "tea_lca": "#E15759",          # Red
    "tea": "#E15759",              # Red
    "literature": "#76B7B2",       # Teal
    "smart_aggregator": "#59A14F", # Green
    "aggregator": "#59A14F",       # Green
    "fast_agent": "#EDC948",       # Yellow
    "standard_agent": "#B07AA1",   # Purple
    "orchestrator": "#9C755F",     # Brown
    "integrated_orchestrator": "#9C755F",  # Brown
    "collab_separation_agent": "#F28E2B",  # Orange
    "collab_separation": "#F28E2B",        # Orange
    "collab_tea_agent": "#E15759",         # Red
    "collab_tea": "#E15759",               # Red
    "end": "#333333",              # Dark gray
}

DEFAULT_AGENT_COLOR = "#BAB0AC"  # Gray
SUCCESS_COLOR = "#59A14F"  # Green
FAILURE_COLOR = "#E15759"  # Red


class WorkflowGraphRenderer:
    """
    Renders workflow graphs showing handoff events as a flow diagram.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150,
        font_size: int = 10,
    ):
        if not HAS_DEPENDENCIES:
            raise ImportError("Matplotlib required. Install with: pip install matplotlib")

        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size

    def render(
        self,
        trace: Dict[str, Any],
        format: str = "svg",
        show_legend: bool = True,
        title: Optional[str] = None,
    ) -> bytes:
        """Render a workflow graph from a trace."""
        handoffs = trace.get("handoff_metrics", [])

        if not handoffs:
            return self._render_empty(format, "No handoff data available")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Build node list from handoffs
        nodes = []
        for h in handoffs:
            if isinstance(h, dict):
                from_agent = h.get("from_agent", "unknown")
                to_agent = h.get("to_agent", "unknown")
                if from_agent not in nodes:
                    nodes.append(from_agent)
                if to_agent not in nodes and to_agent.upper() != "END":
                    nodes.append(to_agent)

        # Add END node
        nodes.append("END")

        n_nodes = len(nodes)
        if n_nodes < 2:
            return self._render_empty(format, "Insufficient nodes for graph")

        # Layout: horizontal flow
        node_spacing = 2.5
        node_width = 1.8
        node_height = 0.8

        # Calculate positions
        node_positions = {}
        for i, node in enumerate(nodes):
            node_positions[node] = (i * node_spacing, 0)

        # Set axis limits
        ax.set_xlim(-1, (n_nodes - 1) * node_spacing + 1)
        ax.set_ylim(-2.5, 2)

        # Draw nodes
        for node, (x, y) in node_positions.items():
            color = self._get_agent_color(node)

            # Draw node box
            box = FancyBboxPatch(
                (x - node_width/2, y - node_height/2),
                node_width, node_height,
                boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                zorder=10,
            )
            ax.add_patch(box)

            # Node label
            label = self._format_agent_name(node)
            ax.text(x, y, label, ha="center", va="center",
                   fontsize=self.font_size, fontweight="bold", color="white", zorder=11)

        # Draw handoff arrows with labels
        for i, h in enumerate(handoffs):
            if not isinstance(h, dict):
                continue

            from_agent = h.get("from_agent", "unknown")
            to_agent = h.get("to_agent", "END")
            if to_agent.upper() == "END":
                to_agent = "END"

            duration_ms = h.get("duration_ms", 0) or 0
            success = h.get("success", True)
            tools = h.get("tools_called", [])

            if from_agent not in node_positions or to_agent not in node_positions:
                continue

            x1, y1 = node_positions[from_agent]
            x2, y2 = node_positions[to_agent]

            # Arrow from right edge of from_node to left edge of to_node
            start_x = x1 + node_width/2
            end_x = x2 - node_width/2

            # Draw arrow
            arrow_color = SUCCESS_COLOR if success else FAILURE_COLOR
            arrow = FancyArrowPatch(
                (start_x, y1),
                (end_x, y2),
                arrowstyle="-|>",
                mutation_scale=20,
                color=arrow_color,
                linewidth=3,
                zorder=5,
            )
            ax.add_patch(arrow)

            # Handoff label (duration + tools)
            mid_x = (start_x + end_x) / 2

            # Duration label above arrow
            if duration_ms > 0:
                if duration_ms >= 1000:
                    dur_str = f"{duration_ms/1000:.1f}s"
                else:
                    dur_str = f"{duration_ms:.0f}ms"
                ax.text(mid_x, y1 + 0.5, dur_str, ha="center", va="bottom",
                       fontsize=self.font_size, fontweight="bold", color=arrow_color)

            # Tools label below arrow
            if tools:
                tools_str = ", ".join(t.replace("_", " ")[:20] for t in tools[:2])
                if len(tools) > 2:
                    tools_str += f" +{len(tools)-2}"
                ax.text(mid_x, y1 - 0.6, tools_str, ha="center", va="top",
                       fontsize=self.font_size - 2, color="#666666", style="italic")

        # Title
        if title:
            ax.set_title(title, fontsize=self.font_size + 4, fontweight="bold", pad=20)

        # Legend
        if show_legend:
            self._add_legend(ax, nodes)

        ax.set_axis_off()
        ax.set_aspect("equal")

        # Render
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi,
                   facecolor="white", edgecolor="none")
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for an agent."""
        name_lower = agent_name.lower()

        # Direct match
        if name_lower in AGENT_COLORS:
            return AGENT_COLORS[name_lower]

        # Partial match
        for key, color in AGENT_COLORS.items():
            if key in name_lower or name_lower in key:
                return color

        return DEFAULT_AGENT_COLOR

    def _format_agent_name(self, name: str) -> str:
        """Format agent name for display."""
        name = name.replace("collab_", "").replace("_agent", "")
        name = name.replace("integrated_", "").replace("_", " ")
        return name.title()

    def _add_legend(self, ax: plt.Axes, nodes: List[str]) -> None:
        """Add legend."""
        # Get unique colors used
        seen_types = set()
        patches = []

        for node in nodes:
            color = self._get_agent_color(node)
            node_type = self._format_agent_name(node)
            if node_type not in seen_types:
                seen_types.add(node_type)
                patches.append(mpatches.Patch(color=color, label=node_type))

        # Add success/failure indicators
        patches.append(mpatches.Patch(color=SUCCESS_COLOR, label="Success"))
        patches.append(mpatches.Patch(color=FAILURE_COLOR, label="Failure"))

        ax.legend(handles=patches, loc="upper left", fontsize=self.font_size - 1,
                 framealpha=0.95, ncol=2)

    def _render_empty(self, format: str, message: str) -> bytes:
        """Render placeholder for empty data."""
        fig, ax = plt.subplots(figsize=(8, 4), dpi=self.dpi)
        ax.text(0.5, 0.5, message, ha="center", va="center",
               fontsize=14, color="gray", transform=ax.transAxes)
        ax.set_axis_off()

        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()


# Global instance
_renderer = None

def render_workflow_graph(
    trace: Dict[str, Any],
    format: str = "svg",
    show_legend: bool = True,
    title: Optional[str] = None,
) -> bytes:
    """Render a workflow graph from a trace."""
    global _renderer
    if _renderer is None:
        _renderer = WorkflowGraphRenderer()
    return _renderer.render(trace, format, show_legend, title)
