"""Exhaustive-sequence display builders for sequence planning tools."""

from __future__ import annotations

from typing import Any


def _format_solvent_detail_lines(solvents: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for rank, sol_info in enumerate(solvents, 1):
        if "error" in sol_info:
            lines.append(f"  {rank}. Error: {sol_info['error']}")
            continue
        if sol_info.get("solvent") in ["No data", "None found", "No viable solvent"]:
            lines.append(f"  {rank}. No data available")
            continue

        sel = sol_info.get("selectivity", 0)
        symbol = "OK" if sel > 10 else "WARN" if sel > 0 else "POOR"
        reused_marker = " *(REUSED)*" if sol_info.get("reused") else ""
        target_sol = sol_info.get("target_sol", 0)
        max_other = sol_info.get("max_other", 0)
        line = (
            f"  {rank}. [{symbol}] **{sol_info['solvent']}**{reused_marker}: "
            f"selectivity={sel:.1f}% (target={target_sol:.1f}%, max_other={max_other:.1f}%)"
        )

        props = []
        if sol_info.get("logp") is not None:
            toxicity = (
                "Low"
                if sol_info["logp"] < 0
                else "Med"
                if sol_info["logp"] < 2
                else "High"
            )
            props.append(f"LogP:{sol_info['logp']:.1f}({toxicity})")
        if sol_info.get("energy") is not None:
            props.append(f"Energy:{sol_info['energy']:.0f}J/g")
        if sol_info.get("bp") is not None:
            props.append(f"BP:{sol_info['bp']:.0f} C")
        if props:
            line += f" | {' '.join(props)}"
        lines.append(line)

    return lines


def build_sequence_analysis_output(
    *,
    sequence: list[str] | tuple[str, ...],
    seq_idx: int,
    seq_steps: list[dict[str, Any]],
) -> list[str]:
    """Render the detailed analysis for one candidate sequence."""
    output = [f"### Sequence {seq_idx}: {' -> '.join(sequence)}\n"]
    for step_data in seq_steps:
        step = step_data["step"]
        target = step_data["target"]
        remaining = step_data["remaining"]
        solvents = step_data["solvents"]
        output.append(f"**Step {step}: Separate {target} from {{{', '.join(remaining)}}}**")
        output.extend(_format_solvent_detail_lines(solvents))
        output.append("")

    output.append(f"**Step {len(sequence)}: {sequence[-1]} is isolated**\n")
    best_solvents = [
        step["solvents"][0]["solvent"]
        for step in seq_steps
        if step["solvents"]
        and step["solvents"][0].get("solvent") not in ["N/A", "No data", "None found", "Error"]
    ]
    unique_solvents = set(best_solvents)
    if len(best_solvents) > len(unique_solvents):
        output.append(
            f"**Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(best_solvents)} steps (some reused)\n"
        )
    else:
        output.append(
            f"**Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(best_solvents)} steps\n"
        )
    output.append("---\n")
    return output


def build_sequential_planning_display(
    *,
    polymer_list: list[str],
    temperature: float,
    top_k_solvents: int,
    excluded_set: set[str],
    all_sequences: list[tuple[str, ...]],
    sequence_results: list[dict[str, Any]],
    sequence_scores: list[dict[str, Any]],
    rank1_plot_url: str | None,
    topk_plot_url: str | None,
    visualization_errors: list[str],
) -> str:
    """Render the exhaustive sequential planning output."""
    output = [
        "# Sequential Separation Planning\n",
        f"**Polymers:** {', '.join(polymer_list)}",
        f"**Number of possible sequences:** {len(polymer_list)}! = {len(all_sequences)}",
        f"**Temperature:** {temperature} C",
        f"**Top solvents per step:** {top_k_solvents}",
    ]
    if excluded_set:
        output.append(
            f"**Excluded solvents (cost constraint):** {', '.join(sorted(excluded_set))}"
        )
    output.append("")
    output.append("## All Possible Sequences\n")
    for index, sequence in enumerate(all_sequences, 1):
        output.append(f"{index}. {' -> '.join(sequence)}")
    output.append("")
    output.append("## Detailed Analysis of Each Sequence\n")
    for result in sequence_results:
        output.extend(result["output"])

    output.append("## Sequence Ranking (by worst-case selectivity)\n")
    output.append("*Higher minimum selectivity = more robust separation*\n")
    for rank, score_data in enumerate(sequence_scores[:10], 1):
        seq_str = " -> ".join(score_data["sequence"])
        min_sel = score_data["min_selectivity"]
        symbol = "#1" if rank == 1 else "#2" if rank == 2 else "#3" if rank == 3 else f"{rank}."
        output.append(f"{symbol} **{seq_str}** (min selectivity: {min_sel:.1f}%)")
    output.append("")

    if sequence_scores:
        output.append("## Top Recommended Separation Sequence\n")
        if rank1_plot_url:
            output.append(f"Visualisation saved: {rank1_plot_url}\n")
            if len(sequence_scores) > 1:
                output.append(
                    f"**Note:** This shows the top-ranked sequence. There are {len(sequence_scores) - 1} other possible sequences."
                )
                output.append(
                    f"    To view alternatives, ask: 'Show me the 2nd best sequence' or 'Show me {polymer_list[1]}-first separation'"
                )
        if topk_plot_url:
            output.append(f"\n**Top-K Comparison**: {topk_plot_url}")
        for error in visualization_errors:
            output.append(f"Could not create visualisation: {error}")

    output.append("\n## Recommendations\n")
    if sequence_scores and sequence_scores[0]["min_selectivity"] > 10:
        best = sequence_scores[0]
        output.append(f"**Best sequence:** {' -> '.join(best['sequence'])}")
        output.append(f"   - Minimum selectivity: {best['min_selectivity']:.1f}%")
        output.append("   - All steps have positive selectivity")
        if len(sequence_scores) > 1:
            output.append(
                f"\n**Alternative sequences available:** {len(sequence_scores) - 1} more options"
            )
            output.append(
                f"   Ask to see specific sequences (e.g., 'Show 2nd best' or 'Show {polymer_list[0]}-first')"
            )
    elif sequence_scores:
        output.append("**No sequence has all high-selectivity steps.**")
        output.append("Consider:")
        output.append("  - Exploring different temperatures")
        output.append("  - Using multi-stage extraction")
        output.append("  - Combining solvents")

    return "\n".join(output)
