"""
Timeline Renderer for Multi-Agent Traces

Renders handoff events as a timeline showing:
- Each handoff as a distinct event on the timeline
- Duration bars with agent colors
- Tool calls and success/failure indicators
- Cumulative time progression
"""

import io
import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Rectangle
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .graph_renderer import AGENT_COLORS, DEFAULT_AGENT_COLOR, SUCCESS_COLOR, FAILURE_COLOR

logger = logging.getLogger(__name__)


class TimelineRenderer:
    """
    Renders timeline showing each handoff as an event.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150,
        font_size: int = 10,
    ):
        if not HAS_MATPLOTLIB:
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
        """Render a timeline from a trace."""
        handoffs = trace.get("handoff_metrics", [])

        if not handoffs:
            return self._render_empty(format, "No handoff data available")

        # Create figure with two subplots: timeline and summary
        fig, (ax_timeline, ax_summary) = plt.subplots(
            2, 1, figsize=self.figsize, dpi=self.dpi,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # Process handoffs to get timing data
        events = []
        cumulative_time = 0

        for i, h in enumerate(handoffs):
            if not isinstance(h, dict):
                continue

            from_agent = h.get("from_agent", "unknown")
            to_agent = h.get("to_agent", "unknown")
            duration_ms = h.get("duration_ms", 0) or 0
            success = h.get("success", True)
            tools = h.get("tools_called", [])

            events.append({
                "index": i + 1,
                "from_agent": from_agent,
                "to_agent": to_agent,
                "start_ms": cumulative_time,
                "duration_ms": duration_ms,
                "end_ms": cumulative_time + duration_ms,
                "success": success,
                "tools": tools,
                "color": self._get_agent_color(from_agent),
            })

            cumulative_time += duration_ms

        if not events:
            return self._render_empty(format, "No valid handoff events")

        total_duration = cumulative_time
        n_events = len(events)

        # Draw timeline
        self._draw_timeline(ax_timeline, events, total_duration)

        # Draw summary table
        self._draw_summary(ax_summary, events, total_duration)

        # Title
        if title:
            fig.suptitle(title, fontsize=self.font_size + 4, fontweight="bold", y=0.98)
        else:
            total_str = f"{total_duration/1000:.2f}s" if total_duration >= 1000 else f"{total_duration:.0f}ms"
            fig.suptitle(f"Execution Timeline ({n_events} handoffs, {total_str} total)",
                        fontsize=self.font_size + 4, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Render
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi,
                   facecolor="white", edgecolor="none")
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()

    def _draw_timeline(self, ax: plt.Axes, events: List[Dict], total_duration: float) -> None:
        """Draw the main timeline visualization."""
        n_events = len(events)
        bar_height = 0.6

        # Y positions for events (bottom to top)
        y_positions = list(range(n_events))

        # Draw event bars
        for event in events:
            y = n_events - event["index"]  # Reverse so first event is at top
            x_start = event["start_ms"]
            width = max(event["duration_ms"], total_duration * 0.01)  # Minimum visible width

            # Main bar
            color = event["color"]
            alpha = 0.9 if event["success"] else 0.5

            bar = Rectangle(
                (x_start, y - bar_height/2),
                width, bar_height,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
                alpha=alpha,
                zorder=5,
            )
            ax.add_patch(bar)

            # Failure indicator (red border)
            if not event["success"]:
                bar_fail = Rectangle(
                    (x_start, y - bar_height/2),
                    width, bar_height,
                    facecolor="none",
                    edgecolor=FAILURE_COLOR,
                    linewidth=3,
                    linestyle="--",
                    zorder=6,
                )
                ax.add_patch(bar_fail)

            # Duration label inside bar (if fits)
            dur_ms = event["duration_ms"]
            if dur_ms >= 1000:
                dur_str = f"{dur_ms/1000:.1f}s"
            else:
                dur_str = f"{dur_ms:.0f}ms"

            if width > total_duration * 0.08:  # Only if bar is wide enough
                ax.text(x_start + width/2, y, dur_str,
                       ha="center", va="center",
                       fontsize=self.font_size - 1, fontweight="bold", color="white",
                       zorder=7)
            else:
                # Label to the right
                ax.text(x_start + width + total_duration * 0.01, y, dur_str,
                       ha="left", va="center",
                       fontsize=self.font_size - 1, fontweight="bold", color=color,
                       zorder=7)

        # Y-axis labels (agent names)
        y_labels = []
        for event in events:
            from_name = self._format_agent_name(event["from_agent"])
            to_name = self._format_agent_name(event["to_agent"])
            y_labels.append(f"{from_name} → {to_name}")

        ax.set_yticks(list(range(n_events)))
        ax.set_yticklabels(reversed(y_labels), fontsize=self.font_size)

        # X-axis
        ax.set_xlim(-total_duration * 0.02, total_duration * 1.05)
        ax.set_ylim(-0.5, n_events - 0.5)

        # Format x-axis
        if total_duration >= 1000:
            ax.set_xlabel("Time (ms)", fontsize=self.font_size)
            # Add secondary label showing seconds
            ax.text(1.02, -0.02, f"({total_duration/1000:.1f}s total)",
                   transform=ax.transAxes, fontsize=self.font_size - 2, va="top")
        else:
            ax.set_xlabel("Time (ms)", fontsize=self.font_size)

        ax.set_ylabel("Handoff Events", fontsize=self.font_size)

        # Grid
        ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)
        ax.set_axisbelow(True)

        # Vertical lines at event boundaries
        for event in events:
            ax.axvline(event["end_ms"], color="#cccccc", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

    def _draw_summary(self, ax: plt.Axes, events: List[Dict], total_duration: float) -> None:
        """Draw summary table of handoffs."""
        ax.set_axis_off()

        # Build summary text
        headers = ["#", "From", "To", "Duration", "Tools", "Status"]
        rows = []

        for event in events:
            dur_ms = event["duration_ms"]
            dur_str = f"{dur_ms/1000:.2f}s" if dur_ms >= 1000 else f"{dur_ms:.0f}ms"
            tools_str = ", ".join(event["tools"][:2]) if event["tools"] else "-"
            if len(event["tools"]) > 2:
                tools_str += f" +{len(event['tools'])-2}"
            # Success=False often means iteration, not failure
            status = "✓" if event["success"] else "⟳"

            rows.append([
                str(event["index"]),
                self._format_agent_name(event["from_agent"]),
                self._format_agent_name(event["to_agent"]),
                dur_str,
                tools_str[:30],
                status,
            ])

        # Create table
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colWidths=[0.05, 0.15, 0.15, 0.1, 0.45, 0.1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 1)
        table.scale(1, 1.5)

        # Style header row
        for i, key in enumerate(headers):
            table[(0, i)].set_facecolor("#4E79A7")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        # Color status cells (last column index is len(headers)-1)
        status_col = len(headers) - 1
        for i, row in enumerate(rows):
            status = row[-1]
            if status == "✓":
                table[(i + 1, status_col)].set_facecolor("#d4edda")
            else:
                table[(i + 1, status_col)].set_facecolor("#f8d7da")

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for an agent."""
        name_lower = agent_name.lower()
        if name_lower in AGENT_COLORS:
            return AGENT_COLORS[name_lower]
        for key, color in AGENT_COLORS.items():
            if key in name_lower or name_lower in key:
                return color
        return DEFAULT_AGENT_COLOR

    def _format_agent_name(self, name: str) -> str:
        """Format agent name for display."""
        name = name.replace("collab_", "").replace("_agent", "")
        name = name.replace("integrated_", "").replace("_", " ")
        return name.title()

    def _render_empty(self, format: str, message: str) -> bytes:
        """Render placeholder."""
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

def render_timeline(
    trace: Dict[str, Any],
    format: str = "svg",
    show_legend: bool = True,
    title: Optional[str] = None,
) -> bytes:
    """Render a timeline from a trace."""
    global _renderer
    if _renderer is None:
        _renderer = TimelineRenderer()
    return _renderer.render(trace, format, show_legend, title)
