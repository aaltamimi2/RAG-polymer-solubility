"""
Static Figure Exporter for Publication-Quality Output

Provides high-level API for generating publication-ready figures:
- Configurable DPI (300 default)
- Font size scaling
- Colorblind-friendly Tableau 10 palette
- Export formats: PDF, SVG, PNG
"""

import io
import os
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .graph_renderer import WorkflowGraphRenderer, AGENT_COLORS
from .timeline_renderer import TimelineRenderer
from .routing_renderer import RoutingRenderer

logger = logging.getLogger(__name__)

# Publication-quality defaults
PUBLICATION_DPI = 300
PUBLICATION_FONT_SIZE = 12
PUBLICATION_FIGSIZE_SINGLE = (8, 6)
PUBLICATION_FIGSIZE_COMBINED = (16, 10)


class StaticFigureExporter:
    """
    Exports publication-quality static figures from execution traces.

    Supports individual visualizations or combined multi-panel figures.
    """

    def __init__(
        self,
        dpi: int = PUBLICATION_DPI,
        font_size: int = PUBLICATION_FONT_SIZE,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the exporter.

        Args:
            dpi: Dots per inch for raster output (default 300)
            font_size: Base font size (default 12)
            output_dir: Default directory for saved files
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "Matplotlib required for static figure export. "
                "Install with: pip install matplotlib"
            )

        self.dpi = dpi
        self.font_size = font_size
        self.output_dir = output_dir

        # Configure matplotlib for publication quality
        self._configure_matplotlib()

        # Initialize renderers with publication settings
        self.graph_renderer = WorkflowGraphRenderer(
            figsize=PUBLICATION_FIGSIZE_SINGLE,
            dpi=dpi,
            font_size=font_size,
            node_size=3000,
        )
        self.timeline_renderer = TimelineRenderer(
            figsize=(12, 6),
            dpi=dpi,
            font_size=font_size,
        )
        self.routing_renderer = RoutingRenderer(
            figsize=(14, 5),
            dpi=dpi,
            font_size=font_size,
        )

    def _configure_matplotlib(self) -> None:
        """Configure matplotlib for publication quality."""
        plt.rcParams.update({
            "font.size": self.font_size,
            "axes.labelsize": self.font_size,
            "axes.titlesize": self.font_size + 2,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # TrueType fonts for PDF
            "ps.fonttype": 42,
        })

    def export_graph(
        self,
        trace: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "pdf",
        title: Optional[str] = None,
    ) -> Union[bytes, str]:
        """
        Export a workflow graph to file or bytes.

        Args:
            trace: StoredTrace dict
            output_path: Output file path (if None, returns bytes)
            format: Output format ("pdf", "svg", "png")
            title: Optional title

        Returns:
            File path if output_path provided, bytes otherwise
        """
        data = self.graph_renderer.render(
            trace=trace,
            format=format,
            show_legend=True,
            title=title,
        )

        if output_path:
            return self._save_to_file(data, output_path)
        return data

    def export_timeline(
        self,
        trace: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "pdf",
        title: Optional[str] = None,
    ) -> Union[bytes, str]:
        """
        Export a timeline to file or bytes.

        Args:
            trace: StoredTrace dict
            output_path: Output file path (if None, returns bytes)
            format: Output format ("pdf", "svg", "png")
            title: Optional title

        Returns:
            File path if output_path provided, bytes otherwise
        """
        data = self.timeline_renderer.render(
            trace=trace,
            format=format,
            show_legend=True,
            title=title,
        )

        if output_path:
            return self._save_to_file(data, output_path)
        return data

    def export_routing(
        self,
        trace: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "pdf",
        title: Optional[str] = None,
    ) -> Union[bytes, str]:
        """
        Export a routing decision visualization to file or bytes.

        Args:
            trace: StoredTrace dict
            output_path: Output file path (if None, returns bytes)
            format: Output format ("pdf", "svg", "png")
            title: Optional title

        Returns:
            File path if output_path provided, bytes otherwise
        """
        data = self.routing_renderer.render(
            trace=trace,
            format=format,
            title=title,
        )

        if output_path:
            return self._save_to_file(data, output_path)
        return data

    def export_combined(
        self,
        trace: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "pdf",
        title: Optional[str] = None,
        include_graph: bool = True,
        include_timeline: bool = True,
        include_routing: bool = True,
    ) -> Union[bytes, str]:
        """
        Export a combined multi-panel figure.

        Layout:
        - Top row: Routing decision (full width)
        - Bottom row: Workflow graph (left) + Timeline (right)

        Args:
            trace: StoredTrace dict
            output_path: Output file path (if None, returns bytes)
            format: Output format ("pdf", "svg", "png")
            title: Optional overall title
            include_graph: Include workflow graph panel
            include_timeline: Include timeline panel
            include_routing: Include routing decision panel

        Returns:
            File path if output_path provided, bytes otherwise
        """
        # Calculate figure layout
        panels = sum([include_graph, include_timeline, include_routing])
        if panels == 0:
            raise ValueError("At least one panel must be included")

        # Create figure with appropriate layout
        if panels == 1:
            fig = plt.figure(figsize=PUBLICATION_FIGSIZE_SINGLE, dpi=self.dpi)
            gs = gridspec.GridSpec(1, 1)
        elif panels == 2:
            fig = plt.figure(figsize=(14, 6), dpi=self.dpi)
            gs = gridspec.GridSpec(1, 2)
        else:
            fig = plt.figure(figsize=PUBLICATION_FIGSIZE_COMBINED, dpi=self.dpi)
            gs = gridspec.GridSpec(2, 2, height_ratios=[0.4, 0.6])

        # Track subplot index
        subplot_idx = 0

        # Add routing panel (top row, full width if 3 panels)
        if include_routing:
            if panels == 3:
                ax_routing = fig.add_subplot(gs[0, :])
            else:
                ax_routing = fig.add_subplot(gs.flat[subplot_idx])
                subplot_idx += 1

            self._draw_routing_in_axes(trace, ax_routing)

        # Add graph panel
        if include_graph:
            if panels == 3:
                ax_graph = fig.add_subplot(gs[1, 0])
            else:
                ax_graph = fig.add_subplot(gs.flat[subplot_idx])
                subplot_idx += 1

            self._draw_graph_in_axes(trace, ax_graph)

        # Add timeline panel
        if include_timeline:
            if panels == 3:
                ax_timeline = fig.add_subplot(gs[1, 1])
            else:
                ax_timeline = fig.add_subplot(gs.flat[subplot_idx])
                subplot_idx += 1

            self._draw_timeline_in_axes(trace, ax_timeline)

        # Overall title
        if title:
            fig.suptitle(title, fontsize=self.font_size + 4, fontweight="bold", y=1.02)
        elif "trace_id" in trace:
            fig.suptitle(
                f"Execution Analysis: {trace['trace_id']}",
                fontsize=self.font_size + 4,
                fontweight="bold",
                y=1.02,
            )

        plt.tight_layout()

        # Render to bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        buffer.seek(0)
        data = buffer.read()

        if output_path:
            return self._save_to_file(data, output_path)
        return data

    def _draw_graph_in_axes(self, trace: Dict[str, Any], ax: plt.Axes) -> None:
        """Draw workflow graph in existing axes (simplified version)."""
        ax.set_title("Workflow Graph", fontsize=self.font_size, fontweight="bold")
        ax.text(
            0.5, 0.5,
            f"Agents: {len(trace.get('agents_visited', []))}\n"
            f"Handoffs: {len(trace.get('handoff_metrics', []))}",
            ha="center", va="center",
            fontsize=self.font_size,
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    def _draw_timeline_in_axes(self, trace: Dict[str, Any], ax: plt.Axes) -> None:
        """Draw timeline in existing axes (simplified version)."""
        ax.set_title("Execution Timeline", fontsize=self.font_size, fontweight="bold")
        duration = trace.get("total_duration_ms", 0)
        ax.text(
            0.5, 0.5,
            f"Total Duration: {duration:.0f}ms" if duration else "No timing data",
            ha="center", va="center",
            fontsize=self.font_size,
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    def _draw_routing_in_axes(self, trace: Dict[str, Any], ax: plt.Axes) -> None:
        """Draw routing info in existing axes (simplified version)."""
        ax.set_title("Routing Decision", fontsize=self.font_size, fontweight="bold")

        complexity = trace.get("complexity", "?")
        path = trace.get("path", "unknown")
        if hasattr(path, "value"):
            path = path.value
        specialists = trace.get("collaboration_specialists", [])

        info_text = (
            f"Complexity: {complexity}/5\n"
            f"Path: {path.title()}\n"
            f"Specialists: {', '.join(specialists) if specialists else 'None'}"
        )
        ax.text(
            0.5, 0.5,
            info_text,
            ha="center", va="center",
            fontsize=self.font_size,
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    def _save_to_file(self, data: bytes, output_path: str) -> str:
        """Save bytes to file."""
        # Create directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(data)

        logger.info(f"Saved figure to {output_path}")
        return output_path


# Global exporter instance
_default_exporter = None


def export_figure(
    trace: Dict[str, Any],
    output_path: str,
    figure_type: str = "graph",
    format: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = PUBLICATION_DPI,
) -> str:
    """
    Export a single figure to file.

    Convenience function for quick exports.

    Args:
        trace: StoredTrace dict
        output_path: Output file path
        figure_type: Type of figure ("graph", "timeline", "routing")
        format: Output format (inferred from path if None)
        title: Optional title
        dpi: Dots per inch

    Returns:
        Output file path
    """
    global _default_exporter
    if _default_exporter is None:
        _default_exporter = StaticFigureExporter(dpi=dpi)

    # Infer format from extension
    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "pdf"

    if figure_type == "graph":
        return _default_exporter.export_graph(trace, output_path, format, title)
    elif figure_type == "timeline":
        return _default_exporter.export_timeline(trace, output_path, format, title)
    elif figure_type == "routing":
        return _default_exporter.export_routing(trace, output_path, format, title)
    else:
        raise ValueError(f"Unknown figure type: {figure_type}")


def export_combined_figure(
    trace: Dict[str, Any],
    output_path: str,
    format: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = PUBLICATION_DPI,
) -> str:
    """
    Export a combined multi-panel figure to file.

    Convenience function for quick combined exports.

    Args:
        trace: StoredTrace dict
        output_path: Output file path
        format: Output format (inferred from path if None)
        title: Optional title
        dpi: Dots per inch

    Returns:
        Output file path
    """
    global _default_exporter
    if _default_exporter is None:
        _default_exporter = StaticFigureExporter(dpi=dpi)

    # Infer format from extension
    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "pdf"

    return _default_exporter.export_combined(trace, output_path, format, title)
