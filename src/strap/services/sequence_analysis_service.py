"""Shared display builders for sequence analysis tools."""

from __future__ import annotations

import math
from typing import Any


def _selectivity_symbol(selectivity: float) -> str:
    if selectivity > 30:
        return "OK"
    if selectivity > 10:
        return "FAIR"
    if selectivity > 0:
        return "WARN"
    return "POOR"


def _format_property_line(best: dict[str, Any]) -> str | None:
    props: list[str] = []
    if best.get("g_score") is not None:
        g_score = best["g_score"]
        rating = (
            "Excellent"
            if g_score >= 8
            else "Good"
            if g_score >= 6
            else "Problematic"
            if g_score >= 4
            else "Hazardous"
        )
        props.append(f"G-Score: {g_score:.1f}/10 ({rating})")
    if best.get("logp") is not None:
        logp = best["logp"]
        toxicity = "Low" if logp < 0 else "Medium" if logp < 2 else "High"
        props.append(f"LogP: {logp:.2f} ({toxicity} toxicity)")
    if best.get("energy") is not None:
        props.append(f"Energy: {best['energy']:.1f} J/g")
    if best.get("bp") is not None:
        props.append(f"BP: {best['bp']:.0f} C")
    if not props:
        return None
    return f"  - Properties: {' | '.join(props)}"


def build_integrated_analysis_display(
    *,
    polymer_list: list[str],
    rank_by: str,
    temperature_min: float,
    temperature_max: float,
    available_temps: list[float],
    all_results: list[dict[str, Any]],
    plot_url: str | None,
    visualization_error: str | None,
    used_greedy: bool,
    temperature_basis_note: str | None = None,
) -> str:
    """Render the integrated multi-polymer separation report."""
    try:
        from strap.solubility import temperature_use_regime

        sensitivity_only = any(
            temperature_use_regime(float(temp)) == "sensitivity_extrapolation"
            for temp in available_temps
        )
    except Exception:
        sensitivity_only = False
    n_polymers = len(polymer_list)
    output = ["# Integrated Multi-Polymer Separation Analysis\n"]
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(
        f"**Temperature Range:** {temperature_min} C - {temperature_max} C ({len(available_temps)} temperatures)"
    )
    if temperature_basis_note:
        output.append(f"**Temperature Basis:** {temperature_basis_note}")
    output.append(f"**Ranking Criterion:** {rank_by}")
    output.append(f"**Number of Sequences:** {n_polymers}! = {math.factorial(n_polymers)}\n")

    if used_greedy:
        output.append(
            f"## Greedy Analysis (n={n_polymers}, {math.factorial(n_polymers):,} permutations avoided)\n"
        )
        output.append(
            "Using greedy algorithm: at each step, select the polymer with the highest selectivity separation.\n"
        )
    else:
        output.append("## Analyzing All Sequences...\n")

    output.append(
        "## Top 3 Sensitivity Screening Sequences\n"
        if sensitivity_only
        else "## Top 3 Recommended Separation Sequences\n"
    )
    for rank, result in enumerate(all_results[:3], 1):
        sequence = result["sequence"]
        medal = "#1" if rank == 1 else "#2" if rank == 2 else "#3"

        output.append(f"### {medal} Rank #{rank}: {' -> '.join(sequence)}")
        output.append(
            f"**Minimum Selectivity (Bottleneck):** {result['min_selectivity']:.1f}%\n"
        )

        for step in result["steps"][:-1]:
            best = step["best"]
            target = step["target"]
            remaining = step["remaining"]
            selectivity = best.get("selectivity", 0)
            symbol = _selectivity_symbol(selectivity)

            output.append(
                f"**Step {step['step']}: Separate {target} from {{{', '.join(remaining)}}}**"
            )

            solvent_name = best.get("solvent", "N/A")
            extrapolated = (
                " *(sensitivity-only extrapolation)*"
                if best.get("temperature_use_regime") == "sensitivity_extrapolation"
                else " *(Apelblat extrapolated above 160 C)*"
                if best.get("temperature_extrapolation") == "above_fit"
                else ""
            )
            if best.get("reused_solvent"):
                output.append(
                    f"  **Solvent:** {solvent_name} @ **{best.get('temperature', 0):.0f} C**{extrapolated} *(REUSED - limited options)*"
                )
            elif best.get("note"):
                output.append(f"  **Solvent:** {solvent_name} - {best.get('note')}")
            else:
                output.append(
                    f"  [{symbol}] **Solvent:** {solvent_name} @ **{best.get('temperature', 0):.0f} C**{extrapolated}"
                )

            output.append(
                f"  - Selectivity: {selectivity:.1f}% (target: {best.get('target_sol', 0):.1f}%, max_other: {best.get('max_other', 0):.1f}%)"
            )
            property_line = _format_property_line(best)
            if property_line:
                output.append(property_line)
            output.append("")

        output.append(f"**Step {len(sequence)}: {sequence[-1]} is isolated**\n")
        solvents_used = [
            step["best"].get("solvent", "N/A")
            for step in result["steps"][:-1]
            if step["best"].get("solvent")
            not in ["N/A", "None found", "No data", "Error", "No viable solvent"]
        ]
        unique_solvents = set(solvents_used)
        if len(solvents_used) > len(unique_solvents):
            duplicates = [solvent for solvent in unique_solvents if solvents_used.count(solvent) > 1]
            output.append(
                f"**Warning:** Solvent(s) {duplicates} used multiple times. This may indicate limited data or challenging separation.\n"
            )
        else:
            output.append(
                f"**Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(solvents_used)} steps\n"
            )
        output.append("---\n")

    if plot_url or visualization_error:
        output.append("## Visualisation\n")
        if plot_url:
            output.append(f"![Separation Sequence]({plot_url})\n")
        if visualization_error:
            output.append(f"Could not create visualisation: {visualization_error}\n")

    output.append("## Summary & Screening Notes\n" if sensitivity_only else "## Summary & Recommendations\n")
    best = all_results[0]
    best_label = "Best Screening Sequence" if sensitivity_only else "Best Sequence"
    output.append(f"**{best_label}:** {' -> '.join(best['sequence'])}")
    output.append(f"**Bottleneck Selectivity:** {best['min_selectivity']:.1f}%\n")
    if sensitivity_only:
        output.append(
            "**Use Constraint:** Steps above 180 C are sensitivity-only Apelblat estimates and should not be treated as validated operating recommendations.\n"
        )
    output.append("**Optimal Conditions per Step:**")
    for step in best["steps"][:-1]:
        best_step = step["best"]
        output.append(
            f"  - Step {step['step']}: {step['target']} -> {best_step.get('solvent', 'N/A')} @ {best_step.get('temperature', 0):.0f} C"
        )

    if rank_by.lower() == "selectivity":
        output.append(
            "\n**Tip:** Re-run with `rank_by='cost'` for cheapest solvents, "
            "`rank_by='safety'` for safest (highest G-score), or `rank_by='toxicity'` for least toxic (lowest LogP)."
        )

    return "\n".join(output)


def build_alternative_sequence_display(
    *,
    polymer_list: list[str],
    target_sequence: dict[str, Any],
    sequence_scores: list[dict[str, Any]],
    rank: int,
    temperature: float,
    plot_url: str | None,
    visualization_error: str | None,
) -> str:
    """Render one selected alternative separation sequence."""
    output: list[str] = []
    output.append(f"# Alternative Separation Sequence (Rank #{rank})\n")
    output.append(f"**Sequence:** {' -> '.join(target_sequence['sequence'])}")
    output.append(f"**Minimum Selectivity:** {target_sequence['min_selectivity']:.1f}%")
    output.append(f"**Temperature:** {temperature} C\n")

    if plot_url:
        output.append(f"\nVisualisation saved: {plot_url}")
    elif visualization_error:
        output.append(f"\nCould not create visualisation: {visualization_error}")

    output.append("\n## Separation Steps\n")
    for step_data in target_sequence["steps"]:
        step_num = step_data["step"]
        target = step_data["target"]
        remaining = step_data["remaining"]
        solvents = step_data["solvents"]

        output.append(f"**Step {step_num}: Separate {target}**")
        if remaining:
            output.append(f"  - Remaining in mixture: {', '.join(remaining)}")
        output.append("  - Top solvents:")
        for rank_idx, solvent in enumerate(solvents[:3], 1):
            solvent_name = solvent.get("solvent", "N/A")
            selectivity = solvent.get("selectivity", 0)
            output.append(
                f"    {rank_idx}. {solvent_name} (selectivity: {selectivity:.1f}%)"
            )
        output.append("")

    if rank > 1:
        best_sequence = sequence_scores[0]
        output.append("## Comparison to Best Sequence\n")
        output.append(
            f"**Best sequence:** {' -> '.join(best_sequence['sequence'])} (min selectivity: {best_sequence['min_selectivity']:.1f}%)"
        )
        output.append(
            f"**This sequence:** {' -> '.join(target_sequence['sequence'])} (min selectivity: {target_sequence['min_selectivity']:.1f}%)"
        )
        output.append(
            f"**Difference:** {target_sequence['min_selectivity'] - best_sequence['min_selectivity']:.1f}% selectivity"
        )

    return "\n".join(output)


def select_alternative_sequence(
    *,
    sequence_scores: list[dict[str, Any]],
    sequence_rank: int | None,
    starting_polymer: str | None,
    polymer_list: list[str],
    n_polymers: int,
    max_exhaustive: int,
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any] | None]:
    """Select an alternative sequence by rank or starting polymer."""
    if sequence_rank is not None:
        if 1 <= sequence_rank <= len(sequence_scores):
            return sequence_scores[sequence_rank - 1], sequence_rank, None
        suffix = (
            f" Only greedy (rank 1) available for {n_polymers} polymers."
            if n_polymers > max_exhaustive
            else ""
        )
        return None, None, {
            "message": f"Rank {sequence_rank} is out of range (1-{len(sequence_scores)}).{suffix}",
            "error_code": "sequence_rank_out_of_range",
            "sequence_rank": sequence_rank,
            "max_rank": len(sequence_scores),
        }

    if starting_polymer is not None:
        starting_polymer_normalized = starting_polymer.strip().upper()
        for index, seq_data in enumerate(sequence_scores, 1):
            if seq_data["sequence"][0].upper() == starting_polymer_normalized:
                return seq_data, index, None
        return None, None, {
            "message": (
                f"No sequence found starting with '{starting_polymer}'. "
                f"Available polymers: {', '.join(polymer_list)}"
            ),
            "error_code": "starting_sequence_not_found",
            "starting_polymer": starting_polymer,
        }

    return None, None, {
        "message": "Must specify either sequence_rank or starting_polymer",
        "error_code": "missing_sequence_selector",
    }
