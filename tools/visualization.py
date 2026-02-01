"""
Visualization Tools for Polymer Separation

Provides specialized visualizations:
- Decision trees for separation sequences
- Selectivity heatmaps
- Process flow diagrams
- Temperature-selectivity profiles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
import os

# Lazy imports for visualization libraries
def _get_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches

def _get_seaborn():
    import seaborn as sns
    return sns

def _get_plotly():
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    return go, px, make_subplots

if TYPE_CHECKING:
    from .separation import SeparationSequence, SeparationStep


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    output_dir: str = "plots"
    format: str = "png"
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "viridis"


class SeparationTreeVisualizer:
    """
    Create decision tree visualizations for separation sequences.

    Shows all possible paths through the separation process
    with selectivity values at each branch.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def create_tree(
        self,
        sequences: List["SeparationSequence"],
        title: str = "Polymer Separation Decision Tree",
        highlight_best: bool = True,
    ) -> str:
        """
        Create a decision tree visualization.

        Args:
            sequences: List of separation sequences to visualize
            title: Plot title
            highlight_best: Whether to highlight the best sequence

        Returns:
            Path to saved plot
        """
        plt, mpatches = _get_matplotlib()

        fig, ax = plt.subplots(figsize=self.config.figsize)

        if not sequences:
            ax.text(0.5, 0.5, "No sequences to display",
                   ha='center', va='center', fontsize=14)
            filepath = os.path.join(self.config.output_dir, "separation_tree.png")
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            return filepath

        # Find best sequence
        best_seq = max(sequences, key=lambda s: s.min_selectivity)

        # Calculate layout
        n_polymers = len(sequences[0].polymers)
        n_sequences = len(sequences)

        # Draw nodes and edges
        level_height = 1.0 / (n_polymers + 1)
        cmap = plt.colormaps.get_cmap(self.config.color_palette)
        colors = cmap([i / max(1, n_sequences - 1) for i in range(n_sequences)])

        for seq_idx, seq in enumerate(sequences):
            is_best = seq == best_seq
            color = colors[seq_idx]
            alpha = 1.0 if is_best else 0.3
            linewidth = 3 if is_best else 1

            x_offset = seq_idx / max(1, n_sequences - 1)

            for step_idx, step in enumerate(seq.steps):
                y = 1.0 - (step_idx + 1) * level_height

                # Draw node
                node_size = 800 if is_best else 400
                ax.scatter([x_offset], [y], s=node_size, c=[color],
                          alpha=alpha, zorder=3, edgecolors='black')

                # Label
                if is_best or seq_idx == 0:
                    ax.annotate(step.target_polymer,
                               (x_offset, y),
                               textcoords="offset points",
                               xytext=(10, 0),
                               fontsize=9 if is_best else 7,
                               fontweight='bold' if is_best else 'normal')

                # Edge to previous
                if step_idx > 0:
                    prev_y = 1.0 - step_idx * level_height
                    ax.plot([x_offset, x_offset], [prev_y, y],
                           c=color, alpha=alpha, linewidth=linewidth, zorder=2)

                    # Selectivity label
                    if is_best and step.remaining_polymers:
                        mid_y = (prev_y + y) / 2
                        ax.annotate(f"{step.selectivity:.0f}%",
                                   (x_offset, mid_y),
                                   textcoords="offset points",
                                   xytext=(-25, 0),
                                   fontsize=8,
                                   color='green' if step.selectivity >= 10 else 'red')

        # Title and labels
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Separation Steps", fontsize=12)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([])

        # Legend
        if highlight_best:
            best_patch = mpatches.Patch(color='black', alpha=0.8,
                                        label=f'Best: {" → ".join(s.target_polymer for s in best_seq.steps)}')
            ax.legend(handles=[best_patch], loc='upper right')

        # Add metrics box
        metrics_text = (
            f"Best Sequence Metrics:\n"
            f"Min Selectivity: {best_seq.min_selectivity:.1f}%\n"
            f"Avg Selectivity: {best_seq.avg_selectivity:.1f}%\n"
            f"Solvents: {len(best_seq.unique_solvents)}"
        )
        ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        filepath = os.path.join(self.config.output_dir, "separation_tree.png")
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return filepath

    def create_sequence_comparison(
        self,
        sequences: List["SeparationSequence"],
        top_k: int = 5,
    ) -> str:
        """Create a bar chart comparing top sequences."""
        plt, _ = _get_matplotlib()

        # Get top k sequences
        sorted_seqs = sorted(sequences, key=lambda s: s.min_selectivity, reverse=True)[:top_k]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Min selectivity comparison
        labels = [" → ".join(s.target_polymer for s in seq.steps) for seq in sorted_seqs]
        min_sels = [seq.min_selectivity for seq in sorted_seqs]
        colors = ['green' if s >= 10 else 'orange' if s >= 5 else 'red' for s in min_sels]

        axes[0].barh(range(len(sorted_seqs)), min_sels, color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(sorted_seqs)))
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_xlabel("Minimum Selectivity (%)")
        axes[0].set_title("Top Sequences by Min Selectivity")
        axes[0].axvline(x=10, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        axes[0].axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='Viable threshold')
        axes[0].legend()

        # Plot 2: Selectivity profile for best sequence
        best = sorted_seqs[0]
        steps = range(1, len(best.steps) + 1)
        selectivities = [s.selectivity for s in best.steps]

        axes[1].bar(steps, selectivities, color='steelblue', alpha=0.7)
        axes[1].axhline(y=10, color='green', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("Step Number")
        axes[1].set_ylabel("Selectivity (%)")
        axes[1].set_title(f"Best Sequence Step-by-Step Selectivity")

        # Add polymer labels
        for i, step in enumerate(best.steps):
            axes[1].annotate(step.target_polymer,
                           (i + 1, selectivities[i]),
                           textcoords="offset points",
                           xytext=(0, 5),
                           ha='center', fontsize=8)

        plt.tight_layout()
        filepath = os.path.join(self.config.output_dir, "sequence_comparison.png")
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return filepath


class SelectivityHeatmap:
    """
    Create heatmaps showing polymer-solvent selectivity.

    Useful for identifying optimal solvent choices across
    multiple polymers at various temperatures.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def create_polymer_solvent_heatmap(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Polymer-Solvent Solubility Matrix",
        annotate: bool = True,
    ) -> str:
        """
        Create heatmap of polymer-solvent solubility.

        Args:
            data: Dict of {polymer: {solvent: solubility}}
            title: Plot title
            annotate: Whether to annotate cells with values

        Returns:
            Path to saved plot
        """
        plt, _ = _get_matplotlib()
        sns = _get_seaborn()
        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame(data).T.fillna(0)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        sns.heatmap(
            df,
            annot=annotate,
            fmt='.0f',
            cmap='RdYlGn',
            center=50,
            ax=ax,
            cbar_kws={'label': 'Solubility (%)'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Solvent", fontsize=12)
        ax.set_ylabel("Polymer", fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filepath = os.path.join(self.config.output_dir, "selectivity_heatmap.png")
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return filepath

    def create_temperature_profile(
        self,
        polymer: str,
        solvent: str,
        temp_data: List[Tuple[float, float]],
        title: Optional[str] = None,
    ) -> str:
        """
        Create temperature-solubility profile for a polymer-solvent pair.

        Args:
            polymer: Polymer name
            solvent: Solvent name
            temp_data: List of (temperature, solubility) tuples
            title: Optional title override

        Returns:
            Path to saved plot
        """
        plt, _ = _get_matplotlib()

        temps, sols = zip(*sorted(temp_data))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(temps, sols, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax.fill_between(temps, sols, alpha=0.3)

        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Solubility (%)", fontsize=12)
        ax.set_title(title or f"{polymer} in {solvent}: Temperature Profile",
                    fontsize=14, fontweight='bold')

        ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = os.path.join(
            self.config.output_dir,
            f"temp_profile_{polymer}_{solvent}.png".replace(" ", "_")
        )
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return filepath


class ProcessFlowDiagram:
    """
    Create process flow diagrams for separation sequences.

    Shows the complete separation process with:
    - Input streams
    - Separation units
    - Output streams
    - Operating conditions
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def create_flow_diagram(
        self,
        sequence: "SeparationSequence",
        title: str = "Polymer Separation Process Flow",
    ) -> str:
        """
        Create a process flow diagram for a separation sequence.

        Uses matplotlib to create a schematic diagram.
        """
        plt, mpatches = _get_matplotlib()

        fig, ax = plt.subplots(figsize=(14, 8))

        n_steps = len([s for s in sequence.steps if s.remaining_polymers])

        # Layout parameters
        box_width = 0.12
        box_height = 0.15
        arrow_length = 0.08
        start_x = 0.1
        y_center = 0.5

        # Draw feed
        feed_polymers = ", ".join(sequence.polymers)
        ax.text(start_x - 0.05, y_center, f"Feed:\n{feed_polymers}",
               fontsize=9, ha='right', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        current_x = start_x

        for i, step in enumerate(sequence.steps):
            if not step.remaining_polymers:
                # Last polymer - just show output
                ax.annotate(step.target_polymer,
                           xy=(current_x, y_center),
                           fontsize=10, ha='left', va='center',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                break

            # Draw separator unit
            rect = mpatches.FancyBboxPatch(
                (current_x, y_center - box_height/2),
                box_width, box_height,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                facecolor='lightyellow',
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(rect)

            # Unit label
            ax.text(current_x + box_width/2, y_center,
                   f"Sep {i+1}\n{step.solvent}\n{step.temperature}°C",
                   fontsize=8, ha='center', va='center')

            # Top output (separated polymer)
            ax.annotate(
                '',
                xy=(current_x + box_width/2, y_center + box_height/2 + 0.08),
                xytext=(current_x + box_width/2, y_center + box_height/2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2)
            )
            ax.text(current_x + box_width/2, y_center + box_height/2 + 0.12,
                   f"{step.target_polymer}\n({step.selectivity:.0f}%)",
                   fontsize=8, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

            # Arrow to next unit
            if i < n_steps - 1:
                ax.annotate(
                    '',
                    xy=(current_x + box_width + arrow_length, y_center),
                    xytext=(current_x + box_width, y_center),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2)
                )

            current_x += box_width + arrow_length + 0.02

        # Title and formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Separator Unit'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.6, label='Product Stream'),
            mpatches.Patch(facecolor='lightblue', alpha=0.8, label='Feed Stream'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        filepath = os.path.join(self.config.output_dir, "process_flow.png")
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return filepath

    def create_interactive_flow(
        self,
        sequence: "SeparationSequence",
    ) -> str:
        """Create an interactive flow diagram using Plotly."""
        go, px, make_subplots = _get_plotly()

        fig = go.Figure()

        # Create Sankey diagram
        labels = ["Feed"]
        source = []
        target = []
        value = []
        colors = []

        # Add polymer labels
        for i, step in enumerate(sequence.steps):
            labels.append(f"Sep {i+1}")
            labels.append(step.target_polymer)

        # Build connections
        node_idx = 1
        for i, step in enumerate(sequence.steps):
            # Feed/previous -> Separator
            prev_idx = 0 if i == 0 else (i * 2)
            source.append(prev_idx)
            target.append(node_idx)
            value.append(100 - i * 10)
            colors.append('rgba(100,100,200,0.4)')

            # Separator -> Product
            source.append(node_idx)
            target.append(node_idx + 1)
            value.append(step.selectivity if step.remaining_polymers else 100)
            colors.append('rgba(100,200,100,0.6)')

            node_idx += 2

        fig.add_trace(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=labels,
                color=['lightblue'] + ['lightyellow', 'lightgreen'] * len(sequence.steps)
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=colors
            )
        ))

        fig.update_layout(
            title="Polymer Separation Flow",
            font_size=12
        )

        filepath = os.path.join(self.config.output_dir, "interactive_flow.html")
        fig.write_html(filepath)

        return filepath


# Convenience functions
def visualize_separation_tree(
    sequences: List["SeparationSequence"],
    output_dir: str = "plots",
) -> str:
    """Quick function to create separation tree visualization."""
    config = PlotConfig(output_dir=output_dir)
    viz = SeparationTreeVisualizer(config)
    return viz.create_tree(sequences)


def visualize_selectivity_heatmap(
    data: Dict[str, Dict[str, float]],
    output_dir: str = "plots",
) -> str:
    """Quick function to create selectivity heatmap."""
    config = PlotConfig(output_dir=output_dir)
    viz = SelectivityHeatmap(config)
    return viz.create_polymer_solvent_heatmap(data)
