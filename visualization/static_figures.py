"""
Static Figure Exporter for Publication-Quality Output

Generates combined multi-panel figures showing:
- Workflow graph (agent flow with handoffs)
- Execution timeline (handoff events)
- Summary statistics
"""

import io
import os
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch, Rectangle
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .graph_renderer import AGENT_COLORS, DEFAULT_AGENT_COLOR, SUCCESS_COLOR, FAILURE_COLOR

logger = logging.getLogger(__name__)

PUBLICATION_DPI = 150
PUBLICATION_FONT_SIZE = 10


class StaticFigureExporter:
    """
    Exports publication-quality combined figures.
    """

    def __init__(
        self,
        dpi: int = PUBLICATION_DPI,
        font_size: int = PUBLICATION_FONT_SIZE,
        output_dir: Optional[str] = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required. Install with: pip install matplotlib")

        self.dpi = dpi
        self.font_size = font_size
        self.output_dir = output_dir

        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 2,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
        })

    def export_combined(
        self,
        trace: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "pdf",
        title: Optional[str] = None,
    ) -> Union[bytes, str]:
        """
        Export a combined figure with workflow and timeline.

        Layout:
        - Top: Workflow graph (horizontal flow)
        - Bottom: Timeline with summary table
        """
        handoffs = trace.get("handoff_metrics", [])

        if not handoffs:
            return self._export_empty(output_path, format, "No handoff data")

        # Create figure
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.5, 0.8], hspace=0.3)

        # Top: Workflow graph
        ax_workflow = fig.add_subplot(gs[0])
        self._draw_workflow(ax_workflow, trace)

        # Middle: Timeline
        ax_timeline = fig.add_subplot(gs[1])
        self._draw_timeline(ax_timeline, trace)

        # Bottom: Summary
        ax_summary = fig.add_subplot(gs[2])
        self._draw_summary_table(ax_summary, trace)

        # Title
        if title:
            fig.suptitle(title, fontsize=self.font_size + 6, fontweight="bold", y=0.98)
        else:
            trace_id = trace.get("trace_id", "unknown")
            total_ms = trace.get("total_duration_ms", 0)
            total_str = f"{total_ms/1000:.2f}s" if total_ms else "N/A"
            fig.suptitle(f"Multi-Agent Execution Analysis\nTrace: {trace_id} | Duration: {total_str}",
                        fontsize=self.font_size + 4, fontweight="bold", y=0.99)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save or return bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi,
                   facecolor="white", edgecolor="none")
        plt.close(fig)
        buffer.seek(0)
        data = buffer.read()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(data)
            return output_path

        return data

    def _draw_workflow(self, ax: plt.Axes, trace: Dict[str, Any]) -> None:
        """Draw workflow graph in axes."""
        handoffs = trace.get("handoff_metrics", [])

        # Build node list
        nodes = []
        for h in handoffs:
            if isinstance(h, dict):
                from_agent = h.get("from_agent", "unknown")
                to_agent = h.get("to_agent", "unknown")
                if from_agent not in nodes:
                    nodes.append(from_agent)
                if to_agent not in nodes and to_agent.upper() != "END":
                    nodes.append(to_agent)
        nodes.append("END")

        n_nodes = len(nodes)
        node_spacing = 2.0
        node_width = 1.5
        node_height = 0.6

        # Calculate positions
        node_positions = {node: (i * node_spacing, 0) for i, node in enumerate(nodes)}

        ax.set_xlim(-0.5, (n_nodes - 1) * node_spacing + 0.5)
        ax.set_ylim(-1.5, 1.5)

        # Draw nodes
        for node, (x, y) in node_positions.items():
            color = self._get_agent_color(node)
            box = FancyBboxPatch(
                (x - node_width/2, y - node_height/2),
                node_width, node_height,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                facecolor=color, edgecolor="white", linewidth=2, zorder=10,
            )
            ax.add_patch(box)

            label = self._format_agent_name(node)
            ax.text(x, y, label, ha="center", va="center",
                   fontsize=self.font_size - 1, fontweight="bold", color="white", zorder=11)

        # Draw arrows
        from matplotlib.patches import FancyArrowPatch
        for h in handoffs:
            if not isinstance(h, dict):
                continue

            from_agent = h.get("from_agent", "unknown")
            to_agent = h.get("to_agent", "END")
            if to_agent.upper() == "END":
                to_agent = "END"

            duration_ms = h.get("duration_ms", 0) or 0
            success = h.get("success", True)

            if from_agent not in node_positions or to_agent not in node_positions:
                continue

            x1, _ = node_positions[from_agent]
            x2, _ = node_positions[to_agent]

            arrow_color = SUCCESS_COLOR if success else FAILURE_COLOR
            arrow = FancyArrowPatch(
                (x1 + node_width/2, 0), (x2 - node_width/2, 0),
                arrowstyle="-|>", mutation_scale=15,
                color=arrow_color, linewidth=2.5, zorder=5,
            )
            ax.add_patch(arrow)

            # Duration label
            mid_x = (x1 + x2) / 2
            if duration_ms >= 1000:
                dur_str = f"{duration_ms/1000:.1f}s"
            else:
                dur_str = f"{duration_ms:.0f}ms"
            ax.text(mid_x, 0.5, dur_str, ha="center", va="bottom",
                   fontsize=self.font_size - 2, fontweight="bold", color=arrow_color)

        ax.set_title("Execution Flow", fontsize=self.font_size + 2, fontweight="bold", pad=10)
        ax.set_axis_off()

    def _draw_timeline(self, ax: plt.Axes, trace: Dict[str, Any]) -> None:
        """Draw timeline in axes."""
        handoffs = trace.get("handoff_metrics", [])

        events = []
        cumulative_time = 0
        for i, h in enumerate(handoffs):
            if not isinstance(h, dict):
                continue
            duration_ms = h.get("duration_ms", 0) or 0
            events.append({
                "index": i + 1,
                "from_agent": h.get("from_agent", "unknown"),
                "to_agent": h.get("to_agent", "unknown"),
                "start_ms": cumulative_time,
                "duration_ms": duration_ms,
                "success": h.get("success", True),
                "tools": h.get("tools_called", []),
                "color": self._get_agent_color(h.get("from_agent", "")),
            })
            cumulative_time += duration_ms

        total_duration = cumulative_time or 1
        n_events = len(events)
        bar_height = 0.5

        for event in events:
            y = n_events - event["index"]
            x_start = event["start_ms"]
            width = max(event["duration_ms"], total_duration * 0.01)

            bar = Rectangle(
                (x_start, y - bar_height/2), width, bar_height,
                facecolor=event["color"], edgecolor="white", linewidth=1,
                alpha=0.9 if event["success"] else 0.5, zorder=5,
            )
            ax.add_patch(bar)

            # Duration label
            dur_ms = event["duration_ms"]
            dur_str = f"{dur_ms/1000:.1f}s" if dur_ms >= 1000 else f"{dur_ms:.0f}ms"
            if width > total_duration * 0.1:
                ax.text(x_start + width/2, y, dur_str, ha="center", va="center",
                       fontsize=self.font_size - 2, fontweight="bold", color="white", zorder=6)

        # Y labels
        y_labels = [f"{self._format_agent_name(e['from_agent'])} → {self._format_agent_name(e['to_agent'])}"
                   for e in events]
        ax.set_yticks(list(range(n_events)))
        ax.set_yticklabels(reversed(y_labels), fontsize=self.font_size - 1)

        ax.set_xlim(-total_duration * 0.02, total_duration * 1.05)
        ax.set_ylim(-0.5, n_events - 0.5)

        if total_duration >= 1000:
            ax.set_xlabel("Time (seconds)", fontsize=self.font_size)
        else:
            ax.set_xlabel("Time (ms)", fontsize=self.font_size)

        ax.set_title("Handoff Timeline", fontsize=self.font_size + 2, fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)

    def _draw_summary_table(self, ax: plt.Axes, trace: Dict[str, Any]) -> None:
        """Draw summary statistics table."""
        ax.set_axis_off()

        handoffs = trace.get("handoff_metrics", [])

        # Build table data
        headers = ["#", "From Agent", "To Agent", "Duration", "Tools Called", "Status"]
        rows = []

        for i, h in enumerate(handoffs):
            if not isinstance(h, dict):
                continue

            dur_ms = h.get("duration_ms", 0) or 0
            dur_str = f"{dur_ms/1000:.2f}s" if dur_ms >= 1000 else f"{dur_ms:.0f}ms"
            tools = h.get("tools_called", [])
            tools_str = ", ".join(tools[:2]) if tools else "-"
            if len(tools) > 2:
                tools_str += f" +{len(tools)-2}"
            # Success=False often means "iteration in progress" not actual failure
            if h.get("success", True):
                status = "✓ Success"
            elif h.get("error_message"):
                status = "✗ Failed"
            else:
                status = "⟳ Iterating"

            rows.append([
                str(i + 1),
                self._format_agent_name(h.get("from_agent", "?")),
                self._format_agent_name(h.get("to_agent", "?")),
                dur_str,
                tools_str[:35],
                status,
            ])

        if not rows:
            ax.text(0.5, 0.5, "No handoff data", ha="center", va="center",
                   fontsize=self.font_size, transform=ax.transAxes)
            return

        table = ax.table(
            cellText=rows, colLabels=headers,
            loc="center", cellLoc="center",
            colWidths=[0.05, 0.15, 0.15, 0.1, 0.40, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 1)
        table.scale(1, 1.8)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#4E79A7")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        # Color status cells (last column is len(headers)-1)
        status_col = len(headers) - 1
        for i, row in enumerate(rows):
            if "Success" in row[-1]:
                table[(i + 1, status_col)].set_facecolor("#d4edda")
            else:
                table[(i + 1, status_col)].set_facecolor("#f8d7da")

        ax.set_title("Handoff Summary", fontsize=self.font_size + 2, fontweight="bold", pad=5)

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for agent."""
        name_lower = agent_name.lower()
        if name_lower in AGENT_COLORS:
            return AGENT_COLORS[name_lower]
        for key, color in AGENT_COLORS.items():
            if key in name_lower or name_lower in key:
                return color
        return DEFAULT_AGENT_COLOR

    def _format_agent_name(self, name: str) -> str:
        """Format agent name."""
        name = name.replace("collab_", "").replace("_agent", "")
        name = name.replace("integrated_", "").replace("_", " ")
        return name.title()

    def _export_empty(self, output_path: Optional[str], format: str, message: str) -> Union[bytes, str]:
        """Export empty placeholder."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="gray",
               transform=ax.transAxes)
        ax.set_axis_off()

        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        buffer.seek(0)
        data = buffer.read()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(data)
            return output_path
        return data


def export_figure(
    trace: Dict[str, Any],
    output_path: str,
    figure_type: str = "combined",
    format: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = PUBLICATION_DPI,
) -> str:
    """Export a figure to file."""
    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "pdf"

    exporter = StaticFigureExporter(dpi=dpi)
    return exporter.export_combined(trace, output_path, format, title)


def export_combined_figure(
    trace: Dict[str, Any],
    output_path: str,
    format: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = PUBLICATION_DPI,
) -> str:
    """Export combined figure."""
    return export_figure(trace, output_path, "combined", format, title, dpi)
