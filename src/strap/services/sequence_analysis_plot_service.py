"""Plot builders for integrated sequence analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strap.tools._helpers import save_plot


def plot_integrated_separation_analysis(
    polymer_list: list[str],
    best_result: dict,
    rank_by: str,
) -> str:
    """Create the integrated sequence plot and return the saved filepath."""
    sequence = best_result["sequence"]
    steps = best_result["steps"][:-1]

    n_steps = len(steps)
    fig_height = max(5 + n_steps * 3.5, 12)
    fig, ax = plt.subplots(figsize=(16, fig_height), dpi=150)

    ax.set_title(
        f'OPTIMAL SEPARATION SEQUENCE: {" -> ".join(sequence)}\n'
        + f'Ranked by: {rank_by} | Min Selectivity: {best_result["min_selectivity"]:.1f}%',
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xlim(0, 14)
    ax.set_ylim(-1.5, n_steps + 3.5)
    ax.axis("off")

    def get_color(selectivity: float) -> str:
        if selectivity > 30:
            return "#2ecc71"
        if selectivity > 10:
            return "#f1c40f"
        if selectivity > 0:
            return "#e67e22"
        return "#e74c3c"

    y_pos = n_steps + 2
    ax.add_patch(
        plt.Rectangle((1.5, y_pos - 0.5), 11, 1.0, facecolor="#3498db", edgecolor="black", linewidth=2.5)
    )
    ax.text(
        7,
        y_pos,
        f'MIXTURE: {", ".join(polymer_list)}',
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    for idx, step in enumerate(steps):
        y_pos = n_steps + 1 - idx
        best = step["best"]
        target = step["target"]
        remaining = step["remaining"]
        selectivity = best.get("selectivity", 0)
        temperature = best.get("temperature", 0)
        color = get_color(selectivity)

        ax.annotate(
            "",
            xy=(3.5, y_pos + 0.4),
            xytext=(3.5, y_pos + 0.9),
            arrowprops=dict(arrowstyle="->", lw=3.5, color=color),
        )
        ax.add_patch(
            plt.Rectangle((1, y_pos - 0.6), 5.5, 1.2, facecolor=color, edgecolor="black", linewidth=2.5, alpha=0.25)
        )
        ax.add_patch(plt.Circle((1.6, y_pos), 0.35, facecolor=color, edgecolor="black", linewidth=2.5))
        ax.text(1.6, y_pos, str(idx + 1), ha="center", va="center", fontsize=15, fontweight="bold", color="white")

        ax.text(2.4, y_pos + 0.25, f"SEPARATE: {target}", ha="left", va="center", fontsize=14, fontweight="bold")
        ax.text(2.4, y_pos - 0.25, f'From: {", ".join(remaining)}', ha="left", va="center", fontsize=12, color="#333")

        ax.add_patch(
            plt.Rectangle((7, y_pos - 0.6), 5.5, 1.2, facecolor="white", edgecolor=color, linewidth=2.5)
        )
        ax.text(9.75, y_pos + 0.25, f'{best.get("solvent", "N/A")}', ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(
            9.75,
            y_pos - 0.15,
            f"{temperature:.0f} C  |  Selectivity: {selectivity:.1f}%",
            ha="center",
            va="center",
            fontsize=13,
            color=color,
            fontweight="bold",
        )

        props_text: list[str] = []
        if best.get("g_score") is not None:
            props_text.append(f"G-Score: {best['g_score']:.0f}")
        if best.get("energy") is not None:
            props_text.append(f"Energy: {best['energy']:.0f} J/g")
        if best.get("bp") is not None:
            props_text.append(f"BP: {best['bp']:.0f} C")
        if props_text:
            ax.text(
                9.75,
                y_pos - 0.52,
                "  |  ".join(props_text),
                ha="center",
                va="top",
                fontsize=12,
                fontweight="semibold",
                color="#222",
            )

    ax.add_patch(
        plt.Rectangle((1.5, -0.5), 11, 1.0, facecolor="#2ecc71", edgecolor="black", linewidth=2.5)
    )
    ax.text(7, 0, "ALL POLYMERS SEPARATED", ha="center", va="center", fontsize=16, fontweight="bold", color="white")

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ecc71", markersize=14, label="Excellent (>30%)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#f1c40f", markersize=14, label="Good (10-30%)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e67e22", markersize=14, label="Marginal (0-10%)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=14, label="Poor (<0%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.95, edgecolor="#333", fancybox=True)

    plt.tight_layout(pad=2.0)
    filepath = save_plot(fig, "integrated_separation_analysis")
    plt.close(fig)
    return filepath
