"""
Timeline/Gantt Chart Renderer for Multi-Agent Traces

Renders Matplotlib Gantt charts showing:
- X-axis: time (ms)
- Y-axis: agent names
- Bars: agent execution duration with color coding
"""

import io
import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .graph_renderer import AGENT_COLORS, DEFAULT_AGENT_COLOR

logger = logging.getLogger(__name__)


class TimelineRenderer:
    """
    Renders timeline/Gantt charts from execution traces.

    Shows agent execution as horizontal bars with time on X-axis.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100,
        font_size: int = 10,
        bar_height: float = 0.6,
    ):
        """
        Initialize the renderer.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for raster output
            font_size: Font size for labels
            bar_height: Height of timeline bars (0-1 scale)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "Matplotlib required for timeline rendering. "
                "Install with: pip install matplotlib"
            )

        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size
        self.bar_height = bar_height

    def render(
        self,
        trace: Dict[str, Any],
        format: str = "svg",
        show_legend: bool = True,
        title: Optional[str] = None,
    ) -> bytes:
        """
        Render a timeline from a trace.

        Args:
            trace: StoredTrace dict or similar structure
            format: Output format ("svg", "png", "pdf")
            show_legend: Whether to include color legend
            title: Optional title for the chart

        Returns:
            Bytes of the rendered image
        """
        # Extract timeline data
        timeline_data = self._extract_timeline_data(trace)

        if not timeline_data:
            return self._render_empty_timeline(format)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Draw timeline bars
        self._draw_timeline(timeline_data, ax)

        # Title
        if title:
            ax.set_title(title, fontsize=self.font_size + 2, fontweight="bold", pad=20)
        elif "trace_id" in trace:
            duration = trace.get("total_duration_ms")
            duration_str = f" ({duration:.0f}ms)" if duration else ""
            ax.set_title(
                f"Execution Timeline: {trace['trace_id']}{duration_str}",
                fontsize=self.font_size + 2,
                fontweight="bold",
                pad=20,
            )

        # Legend
        if show_legend:
            self._add_legend(timeline_data, ax)

        # Axis labels
        ax.set_xlabel("Time (ms)", fontsize=self.font_size)
        ax.set_ylabel("Agent", fontsize=self.font_size)

        # Grid
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Tight layout
        plt.tight_layout()

        # Render to bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()

    def _extract_timeline_data(
        self, trace: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract timeline data from trace.

        Returns list of {agent, start_ms, duration_ms, color, success}.
        """
        timeline = []

        # First, try to use handoff_metrics for precise timing
        handoff_metrics = trace.get("handoff_metrics", [])

        if handoff_metrics:
            current_time = 0.0

            for i, handoff in enumerate(handoff_metrics):
                if not isinstance(handoff, dict):
                    continue

                from_agent = handoff.get("from_agent", "unknown")
                duration_ms = handoff.get("duration_ms") or 0
                success = handoff.get("success", True)

                timeline.append({
                    "agent": from_agent,
                    "start_ms": current_time,
                    "duration_ms": max(duration_ms, 50),  # Minimum visible width
                    "color": self._get_agent_color(from_agent),
                    "success": success,
                })

                current_time += duration_ms

        # Fallback: use agent_timings if no handoff_metrics
        elif trace.get("agent_timings"):
            agent_timings = trace["agent_timings"]

            # Sort by timing values (assumed to be timestamps)
            sorted_agents = sorted(agent_timings.items(), key=lambda x: x[1])

            # Calculate relative timing
            if sorted_agents:
                base_time = sorted_agents[0][1]
                prev_time = 0

                for i, (agent, timing) in enumerate(sorted_agents):
                    start_ms = (timing - base_time) * 1000 if base_time else 0

                    # Estimate duration from gap to next agent
                    if i < len(sorted_agents) - 1:
                        next_timing = sorted_agents[i + 1][1]
                        duration_ms = (next_timing - timing) * 1000
                    else:
                        # Last agent: use remaining time
                        total_duration = trace.get("total_duration_ms", 0)
                        duration_ms = max(total_duration - start_ms, 100) if total_duration else 500

                    timeline.append({
                        "agent": agent,
                        "start_ms": start_ms,
                        "duration_ms": max(duration_ms, 50),
                        "color": self._get_agent_color(agent),
                        "success": True,
                    })

        # Final fallback: use agents_visited with equal spacing
        elif trace.get("agents_visited"):
            agents = trace["agents_visited"]
            total_duration = trace.get("total_duration_ms", len(agents) * 500)
            duration_per_agent = total_duration / max(len(agents), 1)

            for i, agent in enumerate(agents):
                timeline.append({
                    "agent": agent,
                    "start_ms": i * duration_per_agent,
                    "duration_ms": duration_per_agent * 0.9,  # Small gap
                    "color": self._get_agent_color(agent),
                    "success": True,
                })

        return timeline

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for an agent."""
        if agent_name in AGENT_COLORS:
            return AGENT_COLORS[agent_name]

        agent_lower = agent_name.lower()
        for key, color in AGENT_COLORS.items():
            if key in agent_lower or agent_lower in key:
                return color

        return DEFAULT_AGENT_COLOR

    def _draw_timeline(
        self, timeline_data: List[Dict[str, Any]], ax: plt.Axes
    ) -> None:
        """Draw timeline bars."""
        # Get unique agents in order of appearance
        seen_agents = []
        for item in timeline_data:
            if item["agent"] not in seen_agents:
                seen_agents.append(item["agent"])

        # Create agent -> y position mapping
        agent_y = {agent: i for i, agent in enumerate(reversed(seen_agents))}

        # Draw bars
        for item in timeline_data:
            y = agent_y[item["agent"]]

            # Main bar
            bar = Rectangle(
                (item["start_ms"], y - self.bar_height / 2),
                item["duration_ms"],
                self.bar_height,
                facecolor=item["color"],
                edgecolor="white",
                linewidth=1,
                alpha=0.85 if item["success"] else 0.5,
            )
            ax.add_patch(bar)

            # Add failure pattern if not successful
            if not item["success"]:
                bar_fail = Rectangle(
                    (item["start_ms"], y - self.bar_height / 2),
                    item["duration_ms"],
                    self.bar_height,
                    facecolor="none",
                    edgecolor="#E15759",
                    linewidth=2,
                    linestyle="--",
                )
                ax.add_patch(bar_fail)

            # Duration label inside bar (if space permits)
            if item["duration_ms"] > 100:
                ax.text(
                    item["start_ms"] + item["duration_ms"] / 2,
                    y,
                    f"{item['duration_ms']:.0f}ms",
                    ha="center",
                    va="center",
                    fontsize=self.font_size - 2,
                    color="white",
                    fontweight="bold",
                )

        # Set axis limits
        max_time = max(
            (item["start_ms"] + item["duration_ms"] for item in timeline_data),
            default=1000,
        )
        ax.set_xlim(-max_time * 0.02, max_time * 1.05)
        ax.set_ylim(-0.5, len(seen_agents) - 0.5)

        # Y-axis labels
        ax.set_yticks(range(len(seen_agents)))
        ax.set_yticklabels(
            [a.replace("_agent", "").replace("_", " ").title() for a in reversed(seen_agents)],
            fontsize=self.font_size,
        )

    def _add_legend(
        self, timeline_data: List[Dict[str, Any]], ax: plt.Axes
    ) -> None:
        """Add a color legend."""
        # Get unique agent types
        agent_types = set()
        for item in timeline_data:
            agent_name = item["agent"].lower()
            for key in AGENT_COLORS:
                if key in agent_name:
                    agent_types.add(key)
                    break
            else:
                agent_types.add("other")

        # Create legend patches
        patches = []
        for agent_type in sorted(agent_types):
            color = AGENT_COLORS.get(agent_type, DEFAULT_AGENT_COLOR)
            label = agent_type.replace("_", " ").title()
            patches.append(mpatches.Patch(color=color, label=label, alpha=0.85))

        ax.legend(
            handles=patches,
            loc="upper right",
            fontsize=self.font_size - 2,
            framealpha=0.9,
        )

    def _render_empty_timeline(self, format: str) -> bytes:
        """Render a placeholder for empty timelines."""
        fig, ax = plt.subplots(figsize=(8, 4), dpi=self.dpi)
        ax.text(
            0.5, 0.5,
            "No timeline data available",
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


# Global renderer instance
_default_renderer = None


def render_timeline(
    trace: Dict[str, Any],
    format: str = "svg",
    show_legend: bool = True,
    title: Optional[str] = None,
) -> bytes:
    """
    Render a timeline from a trace.

    Convenience function using default renderer settings.

    Args:
        trace: StoredTrace dict or similar structure
        format: Output format ("svg", "png", "pdf")
        show_legend: Whether to include color legend
        title: Optional title for the chart

    Returns:
        Bytes of the rendered image
    """
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = TimelineRenderer()

    return _default_renderer.render(
        trace=trace,
        format=format,
        show_legend=show_legend,
        title=title,
    )
