"""Shared precipitation-analysis helpers for advanced separation tools."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strap.services.visualization_service import (
    PUB_COLORS,
    PUB_FONTSIZE,
    apply_pub_style,
    get_plot_url,
)
from strap.tools._helpers import descriptive_plot_name, save_plot


def build_differential_precipitation_report(
    analyzer: Any,
    format_results: Callable[[list[Any]], str],
    *,
    polymer_to_precipitate: str,
    polymer_to_retain: str,
    min_temperature_gap: float,
    precipitation_threshold: float,
    top_k: int,
) -> str:
    results = analyzer.find_differential_precipitation_solvents(
        polymer_to_precipitate=polymer_to_precipitate,
        polymer_to_retain=polymer_to_retain,
        min_temp_gap=min_temperature_gap,
        precip_threshold=precipitation_threshold,
        top_k=top_k,
    )

    if not results:
        reverse_results = analyzer.find_differential_precipitation_solvents(
            polymer_to_precipitate=polymer_to_retain,
            polymer_to_retain=polymer_to_precipitate,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            top_k=top_k,
        )
        if reverse_results:
            return (
                f"No solvents found where {polymer_to_precipitate} precipitates before {polymer_to_retain}.\n\n"
                f"However, the REVERSE order works:\n\n"
                + format_results(reverse_results)
            )
        return (
            f"No solvents found with {min_temperature_gap} deg C gap for {polymer_to_precipitate}/{polymer_to_retain}.\n"
            f"Try reducing min_temperature_gap or checking polymer names.\n"
            f"Available polymers: {', '.join(analyzer.get_available_polymers())}"
        )

    return format_results(results)


def build_multi_polymer_precipitation_report(
    analyzer: Any,
    format_sequence: Callable[[Any], str],
    *,
    polymers: str,
    solvent: str,
    precipitation_threshold: float,
) -> str:
    polymer_list = [polymer.strip() for polymer in polymers.split(",")]
    result = analyzer.analyze_multi_polymer_precipitation(
        polymers=polymer_list,
        solvent=solvent,
        precip_threshold=precipitation_threshold,
    )

    if not result:
        available_solvents = analyzer.get_available_solvents()
        available_polymers = analyzer.get_available_polymers()
        return (
            f"Could not analyze precipitation for {polymers} in {solvent}.\n"
            f"Available solvents: {', '.join(available_solvents[:10])}...\n"
            f"Available polymers: {', '.join(available_polymers)}"
        )

    return format_sequence(result)


def build_precipitation_temperature_report(
    analyzer: Any,
    *,
    polymer: str,
    solvent: str,
    precipitation_threshold: float,
) -> str:
    point = analyzer.analyze_precipitation(polymer, solvent, precipitation_threshold)
    if not point:
        return f"No data found for {polymer} in {solvent}."

    df = analyzer.get_solubility_curve(polymer, solvent)
    lines = [
        f"# Precipitation Analysis: {polymer} in {solvent}\n",
        "## Key Temperatures\n",
        "| Property | Value |",
        "|----------|-------|",
        f"| Max Solubility | {point.max_solubility:.1f}% at {point.max_solubility_temp:.0f} deg C |",
        f"| Cloud Point (50%) | {point.cloud_point:.0f} deg C |" if point.cloud_point else "| Cloud Point | N/A |",
        (
            f"| Precipitation Temp (<{precipitation_threshold}%) | {point.precipitation_temp:.0f} deg C |"
            if point.precipitation_temp
            else f"| Precipitation Temp | Never below {precipitation_threshold}% |"
        ),
        f"| Transition Width | {point.transition_width:.0f} deg C |",
        f"| Data Points | {point.data_points} |",
        "\n## Temperature-Solubility Curve\n",
        "| Temp (deg C) | Solubility (%) |",
        "|-------------|----------------|",
    ]

    for _, row in df.iloc[::3].iterrows():
        lines.append(f"| {row['temperature']:.0f} | {row['solubility']:.1f} |")

    return "\n".join(lines)


def build_precipitation_curves_report(
    analyzer: Any,
    *,
    polymers: str,
    solvent: str,
    precipitation_threshold: float,
) -> str:
    polymer_list = [polymer.strip() for polymer in polymers.split(",")]

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    precip_temps: dict[str, float] = {}

    for index, polymer in enumerate(polymer_list):
        df = analyzer.get_solubility_curve(polymer, solvent)
        if df.empty:
            continue

        color = PUB_COLORS[index % len(PUB_COLORS)]
        ax.plot(
            df["temperature"],
            df["solubility"],
            "-o",
            color=color,
            label=polymer,
            linewidth=1.2,
            markersize=3,
        )

        precip_temp = analyzer.find_precipitation_temperature(polymer, solvent, precipitation_threshold)
        if precip_temp:
            precip_temps[polymer] = precip_temp
            ax.axvline(x=precip_temp, color=color, linestyle=":", alpha=0.7, linewidth=0.8)
            ax.annotate(
                f"{polymer}\n{precip_temp:.0f}C",
                xy=(precip_temp, precipitation_threshold + 5),
                fontsize=PUB_FONTSIZE - 2,
                color=color,
                ha="center",
            )

    ax.axhline(
        y=precipitation_threshold,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Threshold ({precipitation_threshold}%)",
    )
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Solubility (%)")
    ax.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 170)
    ax.set_ylim(0, 105)

    plot_name = descriptive_plot_name("precipitation_curves", polymers=polymer_list, solvents=[solvent])
    filename = save_plot(fig, plot_name, "matplotlib")

    lines = [
        f"# Precipitation Curves: {', '.join(polymer_list)} in {solvent.upper()}\n",
        f"**Plot saved:** `{filename}`\n",
        "## Precipitation Temperatures\n",
        "| Polymer | Precip Temp |",
        "|---------|-------------|",
    ]

    for polymer, temp in sorted(precip_temps.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {polymer} | {temp:.0f} deg C |")

    if len(precip_temps) >= 2:
        temps = list(precip_temps.values())
        lines.append(f"\n**Maximum Temperature Gap:** {max(temps) - min(temps):.0f} deg C")

    return "\n".join(lines)


def build_atmospheric_feasibility_plot_report(
    analyzer: Any,
    *,
    polymers: str,
    solvent: str,
    precipitation_threshold: float,
    solvent_boiling_points: Mapping[str, float],
) -> str:
    polymer_list = [polymer.strip().upper() for polymer in polymers.split(",")]

    solvent_lower = solvent.lower()
    bp = solvent_boiling_points.get(solvent_lower)
    if bp is None:
        solvent_clean = solvent_lower.replace(" ", "").replace("-", "")
        bp = solvent_boiling_points.get(solvent_clean)
    if bp is None:
        available = ", ".join(list(solvent_boiling_points.keys())[:20])
        return f"Error: No boiling point data for {solvent}. Available solvents: {available}..."

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    precip_temps: dict[str, float] = {}
    max_solubilities: dict[str, float] = {}
    all_temps: list[float] = []

    for index, polymer in enumerate(polymer_list):
        df = analyzer.get_solubility_curve(polymer, solvent)
        if df.empty:
            continue

        color = PUB_COLORS[index % len(PUB_COLORS)]
        ax.plot(
            df["temperature"],
            df["solubility"],
            "-o",
            color=color,
            label=polymer,
            linewidth=1.2,
            markersize=3,
            alpha=0.9,
        )

        all_temps.extend(df["temperature"].tolist())
        max_solubilities[polymer] = df["solubility"].max()

        precip_temp = analyzer.find_precipitation_temperature(polymer, solvent, precipitation_threshold)
        if precip_temp:
            precip_temps[polymer] = precip_temp
            ax.axvline(x=precip_temp, color=color, linestyle=":", alpha=0.6, linewidth=0.8)
            ax.scatter([precip_temp], [precipitation_threshold], color=color, s=20, zorder=5, marker="v")

    if not precip_temps:
        plt.close(fig)
        return f"Error: No precipitation data found for {', '.join(polymer_list)} in {solvent}"

    min_temp = min(all_temps) if all_temps else 20
    max_temp = max(all_temps) if all_temps else 160
    x_max = max(max_temp + 20, bp + 30)

    ax.axvline(x=bp, color="red", linestyle="--", linewidth=1.2, label=f"BP ({bp}C)")
    ax.axhline(y=precipitation_threshold, color="gray", linestyle="--", alpha=0.5, linewidth=0.6)
    ax.text(min_temp + 2, precipitation_threshold + 2, f"Threshold ({precipitation_threshold}%)", fontsize=PUB_FONTSIZE - 2, color="gray")

    max_precip_temp = max(precip_temps.values())
    dissolution_temp = max_precip_temp + 20
    is_feasible = dissolution_temp < bp

    if is_feasible:
        ax.axvspan(min_temp, bp, alpha=0.08, color="green", label="Atmospheric zone")
        ax.axvline(x=dissolution_temp, color="green", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.text(dissolution_temp + 1, 90, f"~{dissolution_temp:.0f}C", fontsize=PUB_FONTSIZE - 2, color="green")
        feasibility_text = f"FEASIBLE AT 1 ATM\nMargin: {bp - dissolution_temp:.0f}C below BP"
        text_color = "green"
    else:
        ax.axvspan(bp, x_max, alpha=0.1, color="red", label="Requires pressure")
        ax.axvline(x=dissolution_temp, color="orange", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.text(dissolution_temp + 1, 90, f"~{dissolution_temp:.0f}C", fontsize=PUB_FONTSIZE - 2, color="orange")
        feasibility_text = f"REQUIRES PRESSURIZATION\nNeeds {dissolution_temp - bp:.0f}C above BP"
        text_color = "red"

    props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=text_color, alpha=0.9)
    ax.text(
        0.98,
        0.98,
        feasibility_text,
        transform=ax.transAxes,
        fontsize=PUB_FONTSIZE - 1,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        color=text_color,
    )

    sorted_precip = sorted(precip_temps.items(), key=lambda item: item[1], reverse=True)
    seq_text = " -> ".join([f"{polymer}@{temp:.0f}C" for polymer, temp in sorted_precip])
    ax.text(
        0.02,
        0.02,
        seq_text,
        transform=ax.transAxes,
        fontsize=PUB_FONTSIZE - 2,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Solubility (%)")
    ax.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_temp - 5, x_max)
    ax.set_ylim(0, 105)

    plot_name = descriptive_plot_name("atmospheric_feasibility", polymers=polymer_list, solvents=[solvent])
    filename = save_plot(fig, plot_name, "matplotlib")

    lines = [
        "# Atmospheric Feasibility Visualization\n",
        f"**Plot saved:** `{filename}`\n",
        f"## System: {', '.join(polymer_list)} in {solvent.upper()}\n",
        f"**Solvent Boiling Point:** {bp} deg C at 1 atm",
        f"**Dissolution Temperature Needed:** ~{dissolution_temp:.0f} deg C\n",
    ]
    if is_feasible:
        lines.append("## Feasible at Atmospheric Pressure")
        lines.append(f"Safety margin: {bp - dissolution_temp:.0f} deg C below boiling point\n")
    else:
        lines.append("## Requires Pressurization")
        lines.append(f"Would need to operate {dissolution_temp - bp:.0f} deg C above boiling point\n")

    lines.append("## Precipitation Sequence (during cooling)\n")
    lines.append("| Order | Polymer | Precip Temp | Max Solubility |")
    lines.append("|-------|---------|-------------|----------------|")
    for index, (polymer, temp) in enumerate(sorted_precip, 1):
        max_sol = max_solubilities.get(polymer, 0)
        lines.append(f"| {index} | {polymer} | {temp:.0f} deg C | {max_sol:.1f}% |")

    if len(sorted_precip) >= 2:
        lines.append("\n## Temperature Gaps")
        for idx in range(len(sorted_precip) - 1):
            polymer_a, temp_a = sorted_precip[idx]
            polymer_b, temp_b = sorted_precip[idx + 1]
            lines.append(f"- {polymer_a} -> {polymer_b}: **{temp_a - temp_b:.0f} deg C**")

    return "\n".join(lines)


def build_polymer_pair_comparison_report(
    analyzer: Any,
    *,
    polymer_pairs: str,
    min_temperature_gap: float,
    precipitation_threshold: float,
) -> str:
    pairs = [pair.strip().split(",") for pair in polymer_pairs.split(";")]
    results: list[dict[str, Any]] = []

    for pair in pairs:
        if len(pair) != 2:
            continue
        polymer_a, polymer_b = pair[0].strip(), pair[1].strip()
        order1 = analyzer.find_differential_precipitation_solvents(
            polymer_to_precipitate=polymer_a,
            polymer_to_retain=polymer_b,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            top_k=5,
        )
        order2 = analyzer.find_differential_precipitation_solvents(
            polymer_to_precipitate=polymer_b,
            polymer_to_retain=polymer_a,
            min_temp_gap=min_temperature_gap,
            precip_threshold=precipitation_threshold,
            top_k=5,
        )

        best_results = order1 if len(order1) >= len(order2) else order2
        best_order = f"{polymer_a} first" if len(order1) >= len(order2) else f"{polymer_b} first"
        results.append(
            {
                "pair": f"{polymer_a}/{polymer_b}",
                "order1_count": len(order1),
                "order2_count": len(order2),
                "best_order": best_order,
                "best_results": best_results,
                "max_gap": best_results[0].temperature_gap if best_results else 0,
            }
        )

    results.sort(key=lambda item: len(item["best_results"]) * item["max_gap"], reverse=True)

    lines = ["# Polymer Pair Comparison for Differential Precipitation\n"]
    lines.append(f"Minimum temperature gap: {min_temperature_gap} deg C\n")
    lines.append("## Summary\n")
    lines.append("| Pair | Solvents Found | Best Order | Max Gap |")
    lines.append("|------|----------------|------------|---------|")
    for result in results:
        lines.append(
            f"| {result['pair']} | {len(result['best_results'])} | {result['best_order']} | {result['max_gap']:.0f} deg C |"
        )

    lines.append("\n## Recommendation\n")
    if results and results[0]["best_results"]:
        best = results[0]
        lines.append(f"**Most feasible pair:** {best['pair']}")
        lines.append(f"- **{len(best['best_results'])} solvents** found with >={min_temperature_gap} deg C gap")
        lines.append(f"- **Best order:** {best['best_order']}")
        lines.append(f"- **Maximum temperature gap:** {best['max_gap']:.0f} deg C")
        lines.append("\n### Top Solvents:\n")
        lines.append("| Solvent | Temp Gap | First Precip | Second Precip |")
        lines.append("|---------|----------|--------------|---------------|")
        for solvent_result in best["best_results"][:5]:
            lines.append(
                f"| {solvent_result.solvent} | {solvent_result.temperature_gap:.0f} deg C | "
                f"{solvent_result.polymer_first} @ {solvent_result.polymer_first_precip_temp:.0f} deg C | "
                f"{solvent_result.polymer_second} @ {solvent_result.polymer_second_precip_temp:.0f} deg C |"
            )
    else:
        lines.append("No feasible pairs found with the specified temperature gap.")
        lines.append(f"Try reducing min_temperature_gap below {min_temperature_gap} deg C.")

    non_feasible = [result for result in results if not result["best_results"]]
    if non_feasible:
        lines.append("\n## Non-Feasible Pairs\n")
        for result in non_feasible:
            lines.append(
                f"- **{result['pair']}**: No solvents with >={min_temperature_gap} deg C gap found"
            )

    return "\n".join(lines)


def build_atmospheric_feasibility_report(
    analyzer: Any,
    format_results: Callable[[Any], str],
    *,
    polymer1: str,
    polymer2: str,
    min_temperature_gap: float,
    precipitation_threshold: float,
    min_solubility: float,
) -> str:
    results = analyzer.check_atmospheric_feasibility(
        polymer1=polymer1,
        polymer2=polymer2,
        min_temp_gap=min_temperature_gap,
        precip_threshold=precipitation_threshold,
        min_solubility=min_solubility,
        top_k=10,
    )
    if not results:
        return (
            f"No solvents found for {polymer1}/{polymer2} differential precipitation "
            f"with >={min_temperature_gap} deg C gap. Try:\n"
            f"- Reducing min_temperature_gap (currently {min_temperature_gap} deg C)\n"
            f"- Reducing min_solubility threshold (currently {min_solubility}%)\n"
            f"- Checking if both polymers have solubility data in the database"
        )
    return format_results(results, include_infeasible=True)


def build_multi_polymer_atmospheric_feasibility_report(
    analyzer: Any,
    format_results: Callable[[Any], str],
    *,
    polymers: str,
    min_temperature_gap: float,
    precipitation_threshold: float,
    min_solubility: float,
) -> str:
    polymer_list = [polymer.strip().upper() for polymer in polymers.split(",") if polymer.strip()]
    if len(polymer_list) < 2:
        return "Error: Need at least 2 polymers. Provide comma-separated list, e.g., 'LDPE,EVOH,PVC'"

    results = analyzer.check_multi_polymer_atmospheric_feasibility(
        polymers=polymer_list,
        min_temp_gap=min_temperature_gap,
        precip_threshold=precipitation_threshold,
        min_solubility=min_solubility,
        top_k=10,
    )
    if not results:
        available = analyzer.get_available_polymers()
        return (
            f"No solvents found for {'/'.join(polymer_list)} sequential precipitation "
            f"with >={min_temperature_gap} deg C gaps between each step.\n\n"
            f"**Suggestions:**\n"
            f"- Reduce min_temperature_gap (currently {min_temperature_gap} deg C)\n"
            f"- Reduce min_solubility threshold (currently {min_solubility}%)\n"
            f"- Check polymer names are valid\n\n"
            f"**Available polymers:** {', '.join(available)}"
        )
    return format_results(results, include_infeasible=True)
