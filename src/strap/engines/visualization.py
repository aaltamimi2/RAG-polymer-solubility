"""
Visualization Tools for Polymer Separation

Provides specialized visualizations:
- Selectivity heatmaps
- Process flow diagrams
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, TYPE_CHECKING
import os

# Lazy imports for visualization libraries
def _get_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches

def _get_seaborn():
    import seaborn as sns
    return sns

if TYPE_CHECKING:
    from .separation import SeparationSequence


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    output_dir: str = "plots"
    format: str = "png"
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "viridis"


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

