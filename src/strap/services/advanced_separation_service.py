"""Shared helper functions for advanced separation tool adapters."""

from __future__ import annotations

import asyncio
from itertools import permutations
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strap.tools._helpers import save_plot

_SELECTIVITY_LEGEND = [
    ("#2ecc71", "Excellent (>30%)"),
    ("#f1c40f", "Good (10-30%)"),
    ("#e67e22", "Marginal (0-10%)"),
    ("#e74c3c", "Poor (<0%)"),
]


def _temperature_basis_suffix(temperature: float) -> str:
    from strap.solubility import temperature_basis_note

    note = temperature_basis_note(float(temperature))
    return f" ({note})" if note else ""


def parse_polymer_list(polymers: str) -> list[str]:
    """Parse a comma-separated polymer string."""
    return [polymer.strip().upper() for polymer in polymers.split(",") if polymer.strip()]


def parse_solvent_list(solvents: str) -> Optional[list[str]]:
    """Parse a comma-separated solvent string, or return None when empty."""
    if not solvents or not solvents.strip():
        return None
    return [solvent.strip() for solvent in solvents.split(",") if solvent.strip()]


def run_async(coro: Any) -> Any:
    """Run an async coroutine from sync tool code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def format_separation_result(result: Any) -> str:
    """Format a separation result as readable markdown."""
    seq = result.best_sequence

    output = [
        "# Optimal Separation Sequence\n",
        f"**Algorithm:** {result.algorithm}",
        f"**Computation Time:** {result.computation_time_ms:.1f}ms",
        f"**Nodes Explored:** {result.nodes_explored}\n",
    ]

    sequence_str = " -> ".join(step.target_polymer for step in seq.steps)
    output.append(f"**Sequence:** {sequence_str}")
    output.append(f"**Status:** {seq.status.value}")
    output.append(f"**Minimum Selectivity:** {seq.min_selectivity:.1f}%")
    output.append(f"**Average Selectivity:** {seq.avg_selectivity:.1f}%")
    output.append(f"**Unique Solvents:** {len(seq.unique_solvents)}\n")
    first_real_step = next((step for step in seq.steps if step.remaining_polymers), None)
    if first_real_step is not None:
        temp_suffix = _temperature_basis_suffix(first_real_step.temperature)
        if temp_suffix:
            output.append(f"**Temperature Basis:**{temp_suffix}\n")

    output.append("## Step-by-Step Breakdown\n")
    for step in seq.steps:
        if step.remaining_polymers:
            status = "OK" if step.is_viable else "LOW"
            output.append(
                f"**Step {step.step_number}: Separate {step.target_polymer}**\n"
                f"  - Solvent: {step.solvent}\n"
                f"  - Temperature: {step.temperature}C{_temperature_basis_suffix(step.temperature)}\n"
                f"  - Selectivity: {step.selectivity:.1f}% [{status}]\n"
                f"  - Target Solubility: {step.target_solubility:.1f}%\n"
                f"  - Max Other Solubility: {step.max_other_solubility:.1f}%\n"
                f"  - Remaining: {', '.join(step.remaining_polymers)}\n"
            )
        else:
            output.append(f"**Step {step.step_number}: {step.target_polymer} isolated**\n")

    return "\n".join(output)


def format_safety_result(result: Any) -> str:
    """Format a safety-optimized separation result as readable markdown."""
    seq = result.best_sequence

    output = [
        "# Safety-Optimized Separation Sequence\n",
        f"**Algorithm:** {result.algorithm}",
        f"**Computation Time:** {result.computation_time_ms:.1f}ms",
        f"**Nodes Explored:** {result.nodes_explored}\n",
    ]

    sequence_str = " -> ".join(step.target_polymer for step in seq.steps)
    output.append(f"**Sequence:** {sequence_str}")
    output.append(f"**Status:** {seq.status.value}")
    output.append(f"**Minimum Selectivity:** {seq.min_selectivity:.1f}%")
    output.append(f"**Average Selectivity:** {seq.avg_selectivity:.1f}%")
    first_real_step = next((step for step in seq.steps if step.remaining_polymers), None)
    if first_real_step is not None:
        temp_suffix = _temperature_basis_suffix(first_real_step.temperature)
        if temp_suffix:
            output.append(f"**Temperature Basis:**{temp_suffix}")

    safety_scores = [step.safety_score for step in seq.steps if step.safety_score is not None]
    if safety_scores:
        output.append(f"**Min G-Score:** {min(safety_scores):.1f}/10")
        output.append(f"**Avg G-Score:** {sum(safety_scores) / len(safety_scores):.1f}/10")
    output.append("")

    output.append("## Step-by-Step Breakdown\n")
    for step in seq.steps:
        if step.remaining_polymers:
            status = "OK" if step.is_viable else "LOW"
            gs_str = (
                f"G-Score: {step.safety_score:.1f}/10" if step.safety_score else "G-Score: N/A"
            )
            output.append(
                f"**Step {step.step_number}: Separate {step.target_polymer}**\n"
                f"  - Solvent: {step.solvent}\n"
                f"  - Temperature: {step.temperature}C{_temperature_basis_suffix(step.temperature)}\n"
                f"  - Selectivity: {step.selectivity:.1f}% [{status}]\n"
                f"  - {gs_str}\n"
                f"  - Remaining: {', '.join(step.remaining_polymers)}\n"
            )
        else:
            output.append(f"**Step {step.step_number}: {step.target_polymer} isolated**\n")

    return "\n".join(output)


def format_top_k_safety_results(result: Any, polymers_str: str) -> str:
    """Format top-k safety-optimized sequences as markdown."""
    sequences = result.all_sequences
    if not sequences:
        return "No sequences found."

    output = [
        "# Top Safety-Optimized Sequences\n",
        f"**Polymers:** {polymers_str}",
        f"**Algorithm:** {result.algorithm}",
        f"**Sequences found:** {len(sequences)}",
        f"**Computation time:** {result.computation_time_ms:.1f}ms\n",
        "| Rank | Sequence | Min Sel (%) | Min G-Score | Bottleneck |",
        "|------|----------|-------------|-------------|------------|",
    ]

    for index, seq in enumerate(sequences, 1):
        seq_str = " -> ".join(step.target_polymer for step in seq.steps)
        real_steps = [step for step in seq.steps if step.remaining_polymers]
        safety_scores = [step.safety_score for step in real_steps if step.safety_score is not None]
        min_gs = min(safety_scores) if safety_scores else 0.0
        if real_steps:
            bottleneck = min(
                (step for step in real_steps if step.safety_score is not None),
                key=lambda step: step.safety_score,
                default=real_steps[0],
            )
            bn_str = f"{bottleneck.target_polymer} (G:{bottleneck.safety_score:.1f})"
        else:
            bn_str = "N/A"
        output.append(
            f"| {index} | {seq_str} | {seq.min_selectivity:.1f} | {min_gs:.1f} | {bn_str} |"
        )

    output.append("\n## Rank 1 Detail\n")
    output.append(format_safety_result(result))
    return "\n".join(output)


def format_top_k_results(result: Any, polymers_str: str) -> str:
    """Format top-k separation sequences as markdown."""
    sequences = result.all_sequences
    if not sequences:
        return "No sequences found."

    output = [
        "# Top Separation Sequences\n",
        f"**Polymers:** {polymers_str}",
        f"**Algorithm:** {result.algorithm}",
        f"**Sequences found:** {len(sequences)}",
        f"**Computation time:** {result.computation_time_ms:.1f}ms\n",
        "| Rank | Sequence | Min Sel (%) | Avg Sel (%) | Bottleneck |",
        "|------|----------|-------------|-------------|------------|",
    ]

    for index, seq in enumerate(sequences, 1):
        seq_str = " -> ".join(step.target_polymer for step in seq.steps)
        real_steps = [step for step in seq.steps if step.remaining_polymers]
        if real_steps:
            bottleneck = min(real_steps, key=lambda step: step.selectivity)
            bn_str = f"{bottleneck.target_polymer} ({bottleneck.selectivity:.1f}%)"
        else:
            bn_str = "N/A"
        output.append(
            f"| {index} | {seq_str} | {seq.min_selectivity:.1f} | "
            f"{seq.avg_selectivity:.1f} | {bn_str} |"
        )

    output.append("\n## Rank 1 Detail\n")
    output.append(format_separation_result(result))
    return "\n".join(output)


def format_optimization_result(result: Any) -> str:
    """Format a temperature optimization result as markdown."""
    from strap.solubility import temperature_use_regime

    sensitivity_only = temperature_use_regime(result.optimal_temperature) == "sensitivity_extrapolation"
    temp_label = "Sensitivity-only Best Temperature" if sensitivity_only else "Optimal Temperature"
    output = [
        "# Temperature Optimization Result\n",
        f"**{temp_label}:** {result.optimal_temperature}C",
        f"**Overall Selectivity:** {result.overall_selectivity:.1f}%",
        f"**Energy Score:** {result.energy_score:.2f} (lower is better)",
        f"**Feasibility Score:** {result.feasibility_score:.1%}\n",
    ]
    if sensitivity_only:
        output.append(
            "**Use Constraint:** 180-200 C values are Apelblat sensitivity estimates for screening only, not validated process recommendations.\n"
        )

    if result.temperature_windows:
        output.append("## Viable Temperature Windows\n")
        for window in result.temperature_windows:
            line = (
                f"- {window.temp_min:.0f}C - {window.temp_max:.0f}C "
                f"(best: {window.optimal_temp:.0f}C, selectivity: {window.selectivity_at_optimal:.1f}%)"
            )
            if getattr(window, "notes", ""):
                line += f" -- {window.notes}"
            output.append(line)
        output.append("")

    if result.recommendations:
        output.append("## Recommendations\n")
        for recommendation in result.recommendations:
            output.append(f"- {recommendation}")

    return "\n".join(output)


def format_selectivity_metrics(metrics: Any) -> str:
    """Format selectivity metrics as markdown."""
    status = "VIABLE" if metrics.is_viable else "NOT VIABLE"
    output = [
        "# Selectivity Analysis\n",
        f"**Target Polymer:** {metrics.target_polymer}",
        f"**Other Polymers:** {', '.join(metrics.other_polymers)}",
        f"**Solvent:** {metrics.solvent}",
        f"**Temperature:** {metrics.temperature}C\n",
        "## Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Selectivity | {metrics.selectivity:.1f}% |",
        f"| Target Solubility | {metrics.target_solubility:.1f}% |",
        f"| Max Other Solubility | {metrics.max_other_solubility:.1f}% |",
        f"| Selectivity Ratio | {metrics.selectivity_ratio:.2f}x |",
        f"| Data Confidence | {metrics.confidence:.1%} |",
        f"| Status | **{status}** |",
    ]
    return "\n".join(output)


def build_solvent_ranking_report(
    scores: list[Any],
    *,
    target_polymer: str,
    other_polymers: list[str],
    temperature: float,
) -> str:
    """Render ranked solvent scores as markdown."""
    output = [
        "# Solvent Ranking\n",
        f"**Target:** {target_polymer.upper()}",
        f"**Separate from:** {', '.join(other_polymers)}",
        f"**Temperature:** {temperature}C\n",
        "## Top Solvents\n",
        "| Rank | Solvent | Overall | Selectivity | BP | LogP | Cp | Energy |",
        "|------|---------|---------|-------------|-----|------|-----|--------|",
    ]
    temp_suffix = _temperature_basis_suffix(temperature)
    if temp_suffix:
        output.insert(4, f"**Temperature Basis:**{temp_suffix}\n")

    for index, score in enumerate(scores, 1):
        output.append(
            f"| {index} | {score.solvent} | {score.overall_score:.2f} | "
            f"{score.selectivity_score:.2f} | {score.bp_score:.2f} | "
            f"{score.logp_score:.2f} | {score.cp_score:.2f} | {score.energy_score:.2f} |"
        )

    if scores and getattr(scores[0], "notes", None):
        output.append(f"\n**Notes for {scores[0].solvent}:**")
        for note in scores[0].notes:
            output.append(f"- {note}")

    return "\n".join(output)


def build_compatibility_matrix_report(
    matrix: dict[str, dict[str, float]],
    *,
    polymers: list[str],
    temperature: float,
) -> str:
    """Render a polymer-solvent compatibility matrix as markdown."""
    all_solvents = set()
    for solvents in matrix.values():
        all_solvents.update(solvents.keys())
    all_solvents = sorted(all_solvents)[:15]

    output = [
        "# Polymer-Solvent Compatibility Matrix\n",
        f"**Temperature:** {temperature}C",
        f"**Polymers:** {len(polymers)}",
        f"**Solvents:** {len(all_solvents)}\n",
    ]

    header = "| Polymer | " + " | ".join(solvent[:8] for solvent in all_solvents) + " |"
    separator = "|---------|" + "|".join("-" * 8 for _ in all_solvents) + "|"
    output.append(header)
    output.append(separator)

    for polymer in polymers:
        row = f"| {polymer} |"
        for solvent in all_solvents:
            solubility = matrix.get(polymer, {}).get(solvent)
            if solubility is not None:
                row += f" {solubility:5.1f}% |"
            else:
                row += "   -   |"
        output.append(row)

    output.append("\n*Values are solubility percentages. Higher = more soluble.*")
    return "\n".join(output)


def build_challenging_pairs_report(
    pairs: list[tuple[str, str, float]],
    *,
    polymers: list[str],
    temperature: float,
    selectivity_threshold: float,
) -> str:
    """Render challenging polymer-pair findings as markdown."""
    output = [
        "# Challenging Polymer Pairs\n",
        f"**Polymers:** {', '.join(polymers)}",
        f"**Temperature:** {temperature}C",
        f"**Threshold:** {selectivity_threshold}% selectivity\n",
    ]

    if not pairs:
        output.append(
            "No challenging pairs found. All polymer pairs can be separated with selectivity above threshold."
        )
        return "\n".join(output)

    output.append("## Difficult Pairs\n")
    output.append("| Polymer 1 | Polymer 2 | Best Selectivity |")
    output.append("|-----------|-----------|------------------|")
    for polymer_one, polymer_two, selectivity in pairs:
        warning = " (CRITICAL)" if selectivity < 5 else ""
        output.append(f"| {polymer_one} | {polymer_two} | {selectivity:.1f}%{warning} |")

    output.append(f"\n**{len(pairs)} challenging pair(s) identified.**")
    output.append("Consider alternative temperatures or solvents for these pairs.")
    return "\n".join(output)


def _find_top_sequence_solvents(
    target: str,
    remaining: list[str],
    *,
    temperature: float,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Return the best solvent candidates for separating one target polymer."""
    if not remaining:
        return [{"solvent": "N/A", "selectivity": float("inf"), "target_sol": 100, "max_other": 0}]

    from strap.solubility import get_all_solvents_selectivity as _get_all_sel

    all_selectivities = _get_all_sel(target, remaining, temperature)
    if not all_selectivities:
        return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

    return [
        {
            "solvent": entry["solvent"],
            "selectivity": entry["selectivity"],
            "target_sol": entry["target_sol"],
            "max_other": entry["max_other_sol"],
        }
        for entry in all_selectivities[:k]
    ]


def _analyze_ranked_sequence(
    sequence: tuple[str, ...],
    *,
    temperature: float,
) -> dict[str, Any]:
    """Score one complete polymer-separation order."""
    steps: list[dict[str, Any]] = []
    total_min_selectivity = float("inf")

    for step_index, target in enumerate(sequence[:-1], 1):
        remaining = list(sequence[step_index:])
        top_solvents = _find_top_sequence_solvents(target, remaining, temperature=temperature)
        steps.append(
            {
                "step": step_index,
                "target": target,
                "remaining": remaining,
                "solvents": top_solvents,
            }
        )
        if top_solvents and top_solvents[0]["selectivity"] < total_min_selectivity:
            total_min_selectivity = top_solvents[0]["selectivity"]

    return {
        "sequence": list(sequence),
        "min_selectivity": total_min_selectivity,
        "steps": steps,
    }


def score_separation_sequences(
    polymer_list: list[str],
    *,
    temperature: float,
    max_exhaustive: int = 6,
) -> list[dict[str, Any]]:
    """Rank candidate separation orders for tree/diagram tools."""
    if len(polymer_list) <= max_exhaustive:
        return sorted(
            [
                _analyze_ranked_sequence(sequence, temperature=temperature)
                for sequence in permutations(polymer_list)
            ],
            key=lambda entry: entry["min_selectivity"],
            reverse=True,
        )

    remaining = list(polymer_list)
    greedy_sequence: list[str] = []
    greedy_steps: list[dict[str, Any]] = []

    while len(remaining) > 1:
        best_candidate: tuple[str, list[dict[str, Any]]] | None = None
        best_value = -float("inf")
        for target in remaining:
            others = [polymer for polymer in remaining if polymer != target]
            solvents = _find_top_sequence_solvents(target, others, temperature=temperature)
            top_selectivity = solvents[0]["selectivity"] if solvents else 0
            if top_selectivity > best_value:
                best_value = top_selectivity
                best_candidate = (target, solvents)

        if best_candidate is None:
            break

        target, solvents = best_candidate
        greedy_sequence.append(target)
        remaining.remove(target)
        greedy_steps.append(
            {
                "step": len(greedy_sequence),
                "target": target,
                "remaining": remaining.copy(),
                "solvents": solvents,
            }
        )

    if remaining:
        greedy_sequence.append(remaining[0])

    min_selectivity = (
        min(step["solvents"][0]["selectivity"] for step in greedy_steps)
        if greedy_steps
        else 0
    )
    return [
        {
            "sequence": greedy_sequence,
            "min_selectivity": min_selectivity,
            "steps": greedy_steps,
        }
    ]


def build_separation_tree_report(
    *,
    polymer_list: list[str],
    sequence_scores: list[dict[str, Any]],
    temperature: float,
    rank1_plot: str | None,
    topk_plot: str | None,
    plot_url_builder,
) -> str:
    """Render the separation-tree tool response."""
    best = sequence_scores[0]
    output = ["# Separation Tree Visualization\n"]

    if rank1_plot:
        output.append(f"**Rank #1 sequence:** {plot_url_builder(rank1_plot)}\n")
    if topk_plot:
        output.append(f"**Top-K comparison:** {plot_url_builder(topk_plot)}\n")

    output.append(f"**Best sequence:** {' -> '.join(best['sequence'])}")
    output.append(f"**Min Selectivity:** {best['min_selectivity']:.1f}%")
    if len(sequence_scores) > 1:
        output.append(f"**Total sequences evaluated:** {len(sequence_scores)}")
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Temperature:** {temperature}C")

    return "\n".join(output)


def build_selectivity_heatmap_report(
    *,
    filepath: str,
    polymer_list: list[str],
    solvent_list: list[str] | None,
    temperature: float,
    matrix: dict[str, dict[str, float]],
) -> str:
    """Render the selectivity-heatmap tool response."""
    solvents_in_matrix = sorted({solvent for row in matrix.values() for solvent in row})
    output = [
        "# Selectivity Heatmap\n",
        f"**Plot saved to:** `{filepath}`",
        f"**Polymers:** {', '.join(polymer_list)}",
        f"**Temperature:** {temperature}C",
        f"**Matrix rows:** {len(matrix)}",
        f"**Matrix solvents:** {len(solvents_in_matrix)}",
    ]
    if solvent_list:
        output.append(f"**Requested solvents:** {', '.join(solvent_list)}")
    return "\n".join(output)


def build_process_flow_report(
    *,
    filepath: str,
    polymer_list: list[str],
    result: Any,
) -> str:
    """Render the process-flow-diagram tool response."""
    return "\n".join(
        [
            "# Process Flow Diagram\n",
            f"**Plot saved to:** `{filepath}`\n",
            "## Process Summary\n",
            f"- **Feed:** {', '.join(polymer_list)}",
            f"- **Steps:** {len(result.best_sequence.steps) - 1}",
            f"- **Solvents Used:** {', '.join(result.best_sequence.unique_solvents)}",
        ]
    )


def plot_separation_sequence(
    polymer_list: list[str],
    sequence_data: dict[str, Any],
    temperature: float,
    total_sequences: int,
    rank: int = 1,
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Plot a single ranked separation sequence."""
    sequence = sequence_data["sequence"]
    steps = sequence_data["steps"]
    min_sel = sequence_data["min_selectivity"]

    n_steps = len(steps)
    fig_height = max(3 + n_steps * 2.5, 8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.set_title(
        f"RECOMMENDED SEPARATION SEQUENCE (Rank #{rank} of {total_sequences})\n"
        f"Sequence: {' -> '.join(sequence)} | Min Selectivity: {min_sel:.1f}% | Temp: {temperature} C",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n_steps + 2.5)
    ax.axis("off")

    y_pos = n_steps + 1.5
    ax.add_patch(
        plt.Rectangle((2, y_pos - 0.3), 6, 0.6, facecolor="#3498db", edgecolor="black", linewidth=2)
    )
    ax.text(
        5,
        y_pos,
        f"STARTING MIXTURE: {', '.join(polymer_list)}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    for index, step in enumerate(steps):
        y_pos = n_steps - index
        target = step["target"]
        remaining = step.get("remaining", [])
        top_solvent = (
            step["solvents"][0]
            if step.get("solvents")
            else {"solvent": "N/A", "selectivity": 0}
        )
        solvent_name = top_solvent["solvent"]
        selectivity = top_solvent.get("selectivity", 0)
        step_temp = top_solvent.get("temperature", temperature)
        optimal_temp = top_solvent.get("optimal_temp", step_temp)
        optimal_sel = top_solvent.get("optimal_selectivity", selectivity)
        color = _selectivity_color(selectivity)

        ax.annotate(
            "",
            xy=(3.5, y_pos + 0.4),
            xytext=(3.5, y_pos + 0.9),
            arrowprops=dict(arrowstyle="->", lw=4, color=color),
        )

        ax.add_patch(
            plt.Rectangle(
                (1.2, y_pos - 0.35),
                4.6,
                0.7,
                facecolor=color,
                edgecolor="black",
                linewidth=2.5,
                alpha=0.3,
            )
        )
        ax.add_patch(
            plt.Circle((1.9, y_pos), 0.25, facecolor=color, edgecolor="black", linewidth=2)
        )
        ax.text(1.9, y_pos, str(index + 1), ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.text(2.7, y_pos, f"SEPARATE: {target}", ha="left", va="center", fontsize=14, fontweight="bold")

        ax.add_patch(
            plt.Rectangle((6.2, y_pos + 0.35), 3.5, 0.75, facecolor="white", edgecolor=color, linewidth=2)
        )
        ax.text(7.95, y_pos + 0.95, f"Solvent: {solvent_name}", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(
            7.95,
            y_pos + 0.72,
            f"Sel: {selectivity:.1f}% @ {step_temp:.0f} C",
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        if abs(optimal_temp - step_temp) > 5 and optimal_sel > selectivity:
            ax.text(
                7.95,
                y_pos + 0.5,
                f"(Optimal: {optimal_sel:.1f}% @ {optimal_temp:.0f} C)",
                ha="center",
                va="center",
                fontsize=8,
                color="#27ae60",
                style="italic",
            )

        if remaining:
            ax.text(
                5.7,
                y_pos - 0.15,
                f"Remaining: {', '.join(remaining)}",
                ha="right",
                va="center",
                fontsize=10,
                color="#34495e",
                style="italic",
                weight="bold",
            )
        else:
            ax.text(
                5.7,
                y_pos - 0.15,
                "(Last polymer - isolated)",
                ha="right",
                va="center",
                fontsize=10,
                color="#27ae60",
                style="italic",
                weight="bold",
            )

    ax.add_patch(plt.Rectangle((2, -0.3), 6, 0.6, facecolor="#2ecc71", edgecolor="black", linewidth=2.5))
    ax.text(5, 0, "ALL POLYMERS SEPARATED", ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=15,
            markeredgecolor="black",
            linewidth=2,
            label=label,
        )
        for color, label in _SELECTIVITY_LEGEND
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=11,
        frameon=True,
        fancybox=True,
        title="Selectivity Quality",
        title_fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    filepath = save_plot(fig, filename or f"separation_sequence_rank{rank}", output_dir=output_dir)
    plt.close(fig)
    return filepath


def plot_topk_comparison(
    polymer_list: list[str],
    sequence_scores: list[dict[str, Any]],
    temperature: float,
    top_k: int = 3,
    filename: str = "separation_topk_comparison",
    output_dir: Optional[str] = None,
) -> str:
    """Plot a side-by-side comparison of top-k separation sequences."""
    top_k = min(top_k, len(sequence_scores))
    n_steps = len(polymer_list) - 1

    fig, ax = plt.subplots(figsize=(5 * top_k, 8), dpi=150)
    ax.set_title(
        f"TOP {top_k} SEPARATION SEQUENCES COMPARISON\n"
        f"Temperature: {temperature} C | Polymers: {', '.join(polymer_list)}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(0, top_k * 5)
    ax.set_ylim(-1, n_steps + 2)
    ax.axis("off")

    col_width = 5
    for col_idx, seq_data in enumerate(sequence_scores[:top_k]):
        x_offset = col_idx * col_width
        sequence = seq_data["sequence"]
        min_sel = seq_data["min_selectivity"]
        seq_steps = seq_data["steps"]

        medal = "#1" if col_idx == 0 else "#2" if col_idx == 1 else "#3"
        header_color = "#2ecc71" if col_idx == 0 else "#95a5a6"
        ax.add_patch(
            plt.Rectangle((x_offset + 0.2, n_steps + 1), col_width - 0.4, 0.8, facecolor=header_color, edgecolor="black", linewidth=2)
        )
        ax.text(
            x_offset + col_width / 2,
            n_steps + 1.4,
            f"{medal} Rank #{col_idx + 1}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        ax.text(
            x_offset + col_width / 2,
            n_steps + 0.6,
            " -> ".join(sequence),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

        for step_idx, step in enumerate(seq_steps):
            y_pos = n_steps - step_idx - 0.5
            target = step.get("target", "?")
            solvents_list = step.get("solvents", [])
            if solvents_list and isinstance(solvents_list, list):
                best_solvent = solvents_list[0]
                solvent = best_solvent.get("solvent", "N/A")
                selectivity = best_solvent.get("selectivity", 0)
                step_temp = best_solvent.get("temperature", temperature)
                optimal_temp = best_solvent.get("optimal_temp", step_temp)
                optimal_sel = best_solvent.get("optimal_selectivity", selectivity)
            else:
                solvent, selectivity = "N/A", 0
                step_temp = optimal_temp = temperature
                optimal_sel = 0

            has_optimal = abs(optimal_temp - step_temp) > 5 and optimal_sel > selectivity
            color = _selectivity_color(selectivity)

            ax.add_patch(
                plt.Rectangle(
                    (x_offset + 0.3, y_pos - 0.4),
                    col_width - 0.6,
                    0.85 if has_optimal else 0.7,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.3,
                )
            )

            y_circle = y_pos + 0.05 if has_optimal else y_pos
            ax.add_patch(plt.Circle((x_offset + 0.7, y_circle), 0.2, facecolor=color, edgecolor="black"))
            ax.text(x_offset + 0.7, y_circle, str(step_idx + 1), ha="center", va="center", fontsize=10, fontweight="bold", color="white")

            y_target = y_pos + 0.22 if has_optimal else y_pos + 0.15
            ax.text(x_offset + 1.1, y_target, target, ha="left", va="center", fontsize=12, fontweight="bold")

            y_solvent = y_pos if has_optimal else y_pos - 0.15
            ax.text(
                x_offset + 1.1,
                y_solvent,
                f"{solvent} ({selectivity:.1f}% @{step_temp:.0f} C)",
                ha="left",
                va="center",
                fontsize=8,
                color="#34495e",
            )
            if has_optimal:
                ax.text(
                    x_offset + 1.1,
                    y_pos - 0.22,
                    f"Opt: {optimal_sel:.1f}% @{optimal_temp:.0f} C",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color="#27ae60",
                    style="italic",
                )

        summary_color = "#2ecc71" if min_sel > 10 else "#f39c12" if min_sel > 0 else "#e74c3c"
        ax.add_patch(
            plt.Rectangle((x_offset + 0.3, -0.8), col_width - 0.6, 0.5, facecolor=summary_color, edgecolor="black", linewidth=2)
        )
        ax.text(
            x_offset + col_width / 2,
            -0.55,
            f"Min Sel: {min_sel:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=12,
            markeredgecolor="black",
            label=label,
        )
        for color, label in _SELECTIVITY_LEGEND
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True, fancybox=True, title="Selectivity", title_fontsize=10)

    plt.tight_layout()
    filepath = save_plot(fig, filename, output_dir=output_dir)
    plt.close(fig)
    return filepath


def _selectivity_color(selectivity: float) -> str:
    if selectivity > 30:
        return "#2ecc71"
    if selectivity > 10:
        return "#f1c40f"
    if selectivity > 0:
        return "#e67e22"
    return "#e74c3c"
