"""
Routing Decision Renderer for Multi-Agent Traces

Renders 3-panel visualization showing:
1. Complexity gauge (1-5 scale)
2. Path selection flowchart
3. Specialist assignment details
"""

import io
import logging
import math
from typing import Optional, Dict, Any, Tuple, List

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
    import matplotlib.patheffects as path_effects
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .graph_renderer import AGENT_COLORS, DEFAULT_AGENT_COLOR

logger = logging.getLogger(__name__)

# Path colors
PATH_COLORS = {
    "fast": "#59A14F",       # Green
    "standard": "#4E79A7",   # Blue
    "specialist": "#F28E2B", # Orange
    "integrated": "#E15759", # Red
}


class RoutingRenderer:
    """
    Renders routing decision visualizations.

    Creates a 3-panel figure showing complexity, path, and specialists.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 5),
        dpi: int = 100,
        font_size: int = 11,
    ):
        """
        Initialize the renderer.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for raster output
            font_size: Font size for labels
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "Matplotlib required for routing visualization. "
                "Install with: pip install matplotlib"
            )

        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size

    def render(
        self,
        trace: Dict[str, Any],
        format: str = "svg",
        title: Optional[str] = None,
    ) -> bytes:
        """
        Render a routing decision visualization.

        Args:
            trace: StoredTrace dict or similar structure
            format: Output format ("svg", "png", "pdf")
            title: Optional title for the figure

        Returns:
            Bytes of the rendered image
        """
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)

        # Extract data
        complexity = trace.get("complexity", 3)
        path = trace.get("path", "standard")
        if hasattr(path, "value"):
            path = path.value
        specialists = trace.get("collaboration_specialists", [])
        if not specialists and trace.get("specialist"):
            specialists = [trace["specialist"]]

        # Panel 1: Complexity gauge
        self._draw_complexity_gauge(axes[0], complexity)

        # Panel 2: Path flowchart
        self._draw_path_flowchart(axes[1], path, complexity)

        # Panel 3: Specialist assignment
        self._draw_specialist_panel(axes[2], specialists, path)

        # Title
        if title:
            fig.suptitle(title, fontsize=self.font_size + 2, fontweight="bold", y=1.02)
        elif "trace_id" in trace:
            fig.suptitle(
                f"Routing Decision: {trace['trace_id']}",
                fontsize=self.font_size + 2,
                fontweight="bold",
                y=1.02,
            )

        plt.tight_layout()

        # Render to bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()

    def _draw_complexity_gauge(self, ax: plt.Axes, complexity: int) -> None:
        """Draw a semicircular complexity gauge."""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Complexity Score", fontsize=self.font_size, fontweight="bold", pad=10)

        # Draw gauge background segments (1-5)
        colors = ["#59A14F", "#79BD6F", "#EDC948", "#F28E2B", "#E15759"]
        labels = ["1", "2", "3", "4", "5"]

        for i in range(5):
            start_angle = 180 - (i * 36)
            end_angle = 180 - ((i + 1) * 36)

            wedge = Wedge(
                center=(0, 0),
                r=1.0,
                theta1=end_angle,
                theta2=start_angle,
                facecolor=colors[i],
                edgecolor="white",
                linewidth=2,
                alpha=0.3 if (i + 1) != complexity else 0.9,
            )
            ax.add_patch(wedge)

            # Add segment labels
            mid_angle = math.radians((start_angle + end_angle) / 2)
            label_x = 0.75 * math.cos(mid_angle)
            label_y = 0.75 * math.sin(mid_angle)
            ax.text(
                label_x, label_y, labels[i],
                ha="center", va="center",
                fontsize=self.font_size,
                fontweight="bold",
                color="white" if (i + 1) == complexity else "gray",
            )

        # Inner white circle
        inner = Circle((0, 0), 0.4, facecolor="white", edgecolor="gray", linewidth=1)
        ax.add_patch(inner)

        # Needle
        needle_angle = math.radians(180 - (complexity - 0.5) * 36)
        needle_x = 0.85 * math.cos(needle_angle)
        needle_y = 0.85 * math.sin(needle_angle)
        ax.annotate(
            "",
            xy=(needle_x, needle_y),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                color="#333333",
                lw=3,
            ),
        )

        # Center dot
        center_dot = Circle((0, 0), 0.08, facecolor="#333333", edgecolor="none")
        ax.add_patch(center_dot)

        # Complexity value text
        ax.text(
            0, -0.15,
            str(complexity),
            ha="center", va="center",
            fontsize=self.font_size + 4,
            fontweight="bold",
            color=colors[complexity - 1],
        )

    def _draw_path_flowchart(
        self, ax: plt.Axes, selected_path: str, complexity: int
    ) -> None:
        """Draw a path selection flowchart."""
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_axis_off()
        ax.set_title("Path Selection", fontsize=self.font_size, fontweight="bold", pad=10)

        # Router box at top
        router_box = FancyBboxPatch(
            (1.0, 1.8), 1.0, 0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor="#4E79A7",
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(router_box)
        ax.text(1.5, 2.05, "Router", ha="center", va="center",
                fontsize=self.font_size, fontweight="bold", color="white")

        # Path boxes
        paths = [
            ("fast", 0.0, "#59A14F", "1-2"),
            ("standard", 1.0, "#4E79A7", "3"),
            ("specialist", 2.0, "#F28E2B", "4"),
            ("integrated", 3.0, "#E15759", "5"),
        ]

        for path_name, x, color, complexity_range in paths:
            is_selected = path_name == selected_path.lower()

            box = FancyBboxPatch(
                (x - 0.1, 0.3), 0.7, 0.6,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=color if is_selected else "#e0e0e0",
                edgecolor="white" if is_selected else "gray",
                linewidth=3 if is_selected else 1,
                alpha=1.0 if is_selected else 0.5,
            )
            ax.add_patch(box)

            # Path label
            label = path_name.title()
            ax.text(
                x + 0.25, 0.7,
                label,
                ha="center", va="center",
                fontsize=self.font_size - 1,
                fontweight="bold" if is_selected else "normal",
                color="white" if is_selected else "gray",
            )

            # Complexity range
            ax.text(
                x + 0.25, 0.45,
                f"({complexity_range})",
                ha="center", va="center",
                fontsize=self.font_size - 2,
                color="white" if is_selected else "gray",
            )

            # Arrow from router to path
            arrow = FancyArrowPatch(
                (1.5, 1.8),
                (x + 0.25, 0.9),
                arrowstyle="->",
                mutation_scale=15,
                color=color if is_selected else "lightgray",
                linewidth=2 if is_selected else 1,
            )
            ax.add_patch(arrow)

    def _draw_specialist_panel(
        self, ax: plt.Axes, specialists: List[str], path: str
    ) -> None:
        """Draw specialist assignment details."""
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_axis_off()
        ax.set_title("Specialist Assignment", fontsize=self.font_size, fontweight="bold", pad=10)

        if not specialists:
            ax.text(
                1.0, 1.0,
                f"No specialists\n(Path: {path})",
                ha="center", va="center",
                fontsize=self.font_size,
                color="gray",
            )
            return

        # Draw specialist boxes
        y_positions = [1.8 - (i * 0.8) for i in range(len(specialists))]

        for i, (specialist, y) in enumerate(zip(specialists, y_positions)):
            color = self._get_specialist_color(specialist)

            box = FancyBboxPatch(
                (0.2, y - 0.25), 1.6, 0.5,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=color,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(box)

            # Specialist label
            label = specialist.replace("_", " ").title()
            ax.text(
                1.0, y,
                label,
                ha="center", va="center",
                fontsize=self.font_size,
                fontweight="bold",
                color="white",
            )

            # Order number
            ax.text(
                0.0, y,
                str(i + 1),
                ha="center", va="center",
                fontsize=self.font_size + 2,
                fontweight="bold",
                color=color,
            )

            # Arrow to next specialist
            if i < len(specialists) - 1:
                arrow = FancyArrowPatch(
                    (1.0, y - 0.25),
                    (1.0, y_positions[i + 1] + 0.25),
                    arrowstyle="->",
                    mutation_scale=15,
                    color="gray",
                    linewidth=2,
                )
                ax.add_patch(arrow)

        # Collaboration mode label
        if len(specialists) > 1:
            ax.text(
                1.0, -0.3,
                "Collaboration Mode",
                ha="center", va="center",
                fontsize=self.font_size - 1,
                style="italic",
                color="gray",
            )

    def _get_specialist_color(self, specialist: str) -> str:
        """Get color for a specialist."""
        specialist_lower = specialist.lower()

        if "separation" in specialist_lower:
            return AGENT_COLORS["separation"]
        elif "tea" in specialist_lower or "lca" in specialist_lower:
            return AGENT_COLORS["tea_lca"]
        elif "literature" in specialist_lower:
            return AGENT_COLORS["literature"]
        elif "aggregator" in specialist_lower:
            return AGENT_COLORS["smart_aggregator"]
        else:
            return DEFAULT_AGENT_COLOR


# Global renderer instance
_default_renderer = None


def render_routing_decision(
    trace: Dict[str, Any],
    format: str = "svg",
    title: Optional[str] = None,
) -> bytes:
    """
    Render a routing decision visualization.

    Convenience function using default renderer settings.

    Args:
        trace: StoredTrace dict or similar structure
        format: Output format ("svg", "png", "pdf")
        title: Optional title for the figure

    Returns:
        Bytes of the rendered image
    """
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = RoutingRenderer()

    return _default_renderer.render(
        trace=trace,
        format=format,
        title=title,
    )
