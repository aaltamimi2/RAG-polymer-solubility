"""Precipitation and antisolvent analysis tools."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from strap.database import get_connection
from strap.services.precipitation_service import (
    build_atmospheric_feasibility_plot_report,
    build_atmospheric_feasibility_report,
    build_differential_precipitation_report,
    build_multi_polymer_atmospheric_feasibility_report,
    build_multi_polymer_precipitation_report,
    build_polymer_pair_comparison_report,
    build_precipitation_curves_report,
    build_precipitation_temperature_report,
)
from strap.services.tool_response_service import json_tool_error, json_tool_success
from strap.tools._helpers import safe_tool_wrapper

logger = logging.getLogger(__name__)

try:
    from strap.engines.precipitation import PrecipitationAnalyzer
except Exception as exc:  # noqa: BLE001
    logger.warning("strap.engines.precipitation unavailable: %s", exc)
    PrecipitationAnalyzer = None


def _precipitation_error(
    tool_name: str,
    message: str,
    *,
    error_code: str = "invalid_input",
    **data: Any,
) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)


def _get_analyzer():
    if PrecipitationAnalyzer is None:
        raise RuntimeError("precipitation_engine_unavailable")
    return PrecipitationAnalyzer(get_connection())


def _format_solubility_percent(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}%"
    if value < 0.1:
        return f"{value:.4f}%"
    return f"{value:.2f}%"


@safe_tool_wrapper(structured_output=True)
def find_differential_precipitation_solvents(
    polymer_to_precipitate: str,
    polymer_to_retain: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    top_k: int = 10,
) -> str:
    """Find solvents where one polymer precipitates before another during cooling."""
    from strap.engines.precipitation import format_differential_precipitation_results

    return build_differential_precipitation_report(
        _get_analyzer(),
        format_differential_precipitation_results,
        polymer_to_precipitate=polymer_to_precipitate,
        polymer_to_retain=polymer_to_retain,
        min_temperature_gap=min_temperature_gap,
        precipitation_threshold=precipitation_threshold,
        top_k=top_k,
    )


@safe_tool_wrapper(structured_output=True)
def analyze_multi_polymer_precipitation(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Determine the order in which multiple polymers precipitate during cooling."""
    from strap.engines.precipitation import format_multi_polymer_sequence

    return build_multi_polymer_precipitation_report(
        _get_analyzer(),
        format_multi_polymer_sequence,
        polymers=polymers,
        solvent=solvent,
        precipitation_threshold=precipitation_threshold,
    )


@safe_tool_wrapper(structured_output=True)
def analyze_precipitation_temperature(
    polymer: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Analyze dissolution/precipitation temperatures for one polymer-solvent pair."""
    return build_precipitation_temperature_report(
        _get_analyzer(),
        polymer=polymer,
        solvent=solvent,
        precipitation_threshold=precipitation_threshold,
    )


@safe_tool_wrapper(structured_output=True)
def plot_precipitation_curves(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Plot temperature-dependent solubility curves for multiple polymers."""
    return build_precipitation_curves_report(
        _get_analyzer(),
        polymers=polymers,
        solvent=solvent,
        precipitation_threshold=precipitation_threshold,
    )


@safe_tool_wrapper(structured_output=True)
def plot_atmospheric_feasibility(
    polymers: str,
    solvent: str,
    precipitation_threshold: float = 1.0,
) -> str:
    """Plot precipitation curves with the solvent boiling point at 1 atm."""
    from strap.engines.precipitation import SOLVENT_BOILING_POINTS

    return build_atmospheric_feasibility_plot_report(
        _get_analyzer(),
        polymers=polymers,
        solvent=solvent,
        precipitation_threshold=precipitation_threshold,
        solvent_boiling_points=SOLVENT_BOILING_POINTS,
    )


@safe_tool_wrapper(structured_output=True)
def compare_polymer_pairs_precipitation(
    polymer_pairs: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
) -> str:
    """Compare differential precipitation feasibility across multiple polymer pairs."""
    return build_polymer_pair_comparison_report(
        _get_analyzer(),
        polymer_pairs=polymer_pairs,
        min_temperature_gap=min_temperature_gap,
        precipitation_threshold=precipitation_threshold,
    )


@safe_tool_wrapper(structured_output=True)
def check_atmospheric_feasibility(
    polymer1: str,
    polymer2: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    min_solubility: float = 30.0,
) -> str:
    """Check if two-polymer differential precipitation is feasible below boiling point."""
    from strap.engines.precipitation import format_atmospheric_feasibility_results

    try:
        return build_atmospheric_feasibility_report(
            _get_analyzer(),
            format_atmospheric_feasibility_results,
            polymer1=polymer1,
            polymer2=polymer2,
            min_temperature_gap=min_temperature_gap,
            precipitation_threshold=precipitation_threshold,
            min_solubility=min_solubility,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in atmospheric feasibility check: %s", exc)
        return _precipitation_error(
            "check_atmospheric_feasibility",
            f"Error analyzing atmospheric feasibility: {exc}",
            error_code="analysis_failed",
            polymer1=polymer1,
            polymer2=polymer2,
            min_temperature_gap=min_temperature_gap,
            precipitation_threshold=precipitation_threshold,
            min_solubility=min_solubility,
        )


@safe_tool_wrapper(structured_output=True)
def check_multi_polymer_atmospheric_feasibility(
    polymers: str,
    min_temperature_gap: float = 20.0,
    precipitation_threshold: float = 1.0,
    min_solubility: float = 30.0,
) -> str:
    """Check if sequential precipitation of 2+ polymers is feasible below boiling point."""
    from strap.engines.precipitation import format_multi_polymer_atmospheric_results

    try:
        return build_multi_polymer_atmospheric_feasibility_report(
            _get_analyzer(),
            format_multi_polymer_atmospheric_results,
            polymers=polymers,
            min_temperature_gap=min_temperature_gap,
            precipitation_threshold=precipitation_threshold,
            min_solubility=min_solubility,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in multi-polymer atmospheric feasibility check: %s", exc)
        return _precipitation_error(
            "check_multi_polymer_atmospheric_feasibility",
            f"Error analyzing multi-polymer atmospheric feasibility: {exc}",
            error_code="analysis_failed",
            polymers=polymers,
            min_temperature_gap=min_temperature_gap,
            precipitation_threshold=precipitation_threshold,
            min_solubility=min_solubility,
        )


@safe_tool_wrapper(structured_output=True)
def find_antisolvents(
    polymer: str,
    max_solubility: float = 1.0,
    temperature: float = 25.0,
) -> str:
    """Find near-zero-solubility antisolvents for a polymer."""
    from strap.solubility import get_available_solvents_for_polymer, get_solubility

    solvents = get_available_solvents_for_polymer(polymer)
    rows = []
    for solvent in solvents:
        solubility = get_solubility(polymer, solvent, temperature)
        if solubility is not None and solubility <= max_solubility:
            rows.append({"solvent": solvent, "solubility": solubility, "temp": temperature})

    df = (
        pd.DataFrame(rows).sort_values("solubility").reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
    if df.empty:
        return _precipitation_error(
            "find_antisolvents",
            f"No antisolvents found for {polymer} with solubility < {max_solubility}% at {temperature} deg C.",
            error_code="no_antisolvents_found",
            polymer=polymer.upper(),
            temperature=temperature,
            max_solubility=max_solubility,
        )

    df = df.drop_duplicates(subset=["solvent"])
    lines = [
        f"# Antisolvents for {polymer.upper()}\n",
        f"Solvents with solubility < {max_solubility}% at ~{temperature} deg C\n",
        f"**Found {len(df)} antisolvents** (polymer is essentially insoluble)\n",
        "| Rank | Antisolvent | Solubility | Temp |",
        "|------|-------------|------------|------|",
    ]

    antisolvents = []
    for index, row in df.iterrows():
        lines.append(
            f"| {index + 1} | {row['solvent']} | {_format_solubility_percent(row['solubility'])} | {row['temp']:.0f} deg C |"
        )
        antisolvents.append(
            {
                "solvent": row["solvent"],
                "solubility": float(row["solubility"]),
                "temperature": float(row["temp"]),
            }
        )

    lines.append("\n## Usage")
    lines.append(
        "These solvents can be used as antisolvents to precipitate "
        f"{polymer} from solution by adding them to a dissolved polymer mixture."
    )

    return json_tool_success(
        "\n".join(lines),
        tool_name="find_antisolvents",
        polymer=polymer.upper(),
        temperature=temperature,
        max_solubility=max_solubility,
        antisolvent_count=len(antisolvents),
        antisolvents=antisolvents,
    )


@safe_tool_wrapper(structured_output=True)
def find_antisolvent_pairs(
    polymer: str,
    min_good_solubility: float = 50.0,
    max_antisolvent_solubility: float = 1.0,
) -> str:
    """Find good-solvent plus antisolvent pairs for dissolution and precipitation."""
    from strap.solubility import (
        get_available_solvents_for_polymer,
        get_solubility,
        get_solubility_curve,
    )

    solvents = get_available_solvents_for_polymer(polymer)
    good_rows = []
    for solvent in solvents:
        curve = get_solubility_curve(polymer, solvent, t_start_c=25, t_end_c=160, t_step_c=5)
        if curve:
            max_solubility = max(point["solubility"] for point in curve)
            max_temp = next(
                point["temperature"]
                for point in curve
                if point["solubility"] == max_solubility
            )
            if max_solubility >= min_good_solubility:
                good_rows.append(
                    {
                        "solvent": solvent,
                        "max_solubility": max_solubility,
                        "dissolution_temp": max_temp,
                    }
                )

    anti_rows = []
    for solvent in solvents:
        solubility = get_solubility(polymer, solvent, 25.0)
        if solubility is not None and solubility <= max_antisolvent_solubility:
            anti_rows.append({"solvent": solvent, "min_solubility": solubility, "temp": 25.0})

    good_solvents = (
        pd.DataFrame(good_rows).sort_values("max_solubility", ascending=False).reset_index(drop=True)
        if good_rows
        else pd.DataFrame()
    )
    antisolvents = (
        pd.DataFrame(anti_rows).sort_values("min_solubility").reset_index(drop=True)
        if anti_rows
        else pd.DataFrame()
    )

    if good_solvents.empty:
        return _precipitation_error(
            "find_antisolvent_pairs",
            f"No good solvents found for {polymer} with solubility > {min_good_solubility}%",
            error_code="no_good_solvents_found",
            polymer=polymer.upper(),
            min_good_solubility=min_good_solubility,
        )
    if antisolvents.empty:
        return _precipitation_error(
            "find_antisolvent_pairs",
            f"No antisolvents found for {polymer} with solubility < {max_antisolvent_solubility}%",
            error_code="no_antisolvents_found",
            polymer=polymer.upper(),
            max_antisolvent_solubility=max_antisolvent_solubility,
        )

    lines = [
        f"# Antisolvent Precipitation Pairs for {polymer.upper()}\n",
        "## Good Solvents (for dissolution)\n",
        f"Solvents with >{min_good_solubility}% solubility:\n",
        "| Good Solvent | Max Solubility | Dissolution Temp |",
        "|--------------|----------------|------------------|",
    ]

    good_payload = []
    for _, row in good_solvents.head(10).iterrows():
        lines.append(
            f"| {row['solvent']} | {row['max_solubility']:.1f}% | {row['dissolution_temp']:.0f} deg C |"
        )
        good_payload.append(
            {
                "solvent": row["solvent"],
                "max_solubility": float(row["max_solubility"]),
                "dissolution_temp": float(row["dissolution_temp"]),
            }
        )

    lines.append("\n## Antisolvents (to induce precipitation)\n")
    lines.append(f"Solvents with <{max_antisolvent_solubility}% solubility at room temp:\n")
    lines.append("| Antisolvent | Solubility at RT |")
    lines.append("|-------------|------------------|")

    anti_payload = []
    for _, row in antisolvents.head(10).iterrows():
        lines.append(f"| {row['solvent']} | {_format_solubility_percent(row['min_solubility'])} |")
        anti_payload.append(
            {"solvent": row["solvent"], "min_solubility": float(row["min_solubility"])}
        )

    recommendations = []
    for _, good in good_solvents.head(5).iterrows():
        for _, anti in antisolvents.head(5).iterrows():
            if good["solvent"].lower() == anti["solvent"].lower():
                continue
            recommendations.append(
                {
                    "good_solvent": good["solvent"],
                    "good_solubility": float(good["max_solubility"]),
                    "dissolution_temp": float(good["dissolution_temp"]),
                    "antisolvent": anti["solvent"],
                    "antisolvent_solubility": float(anti["min_solubility"]),
                }
            )

    lines.append("\n## Recommended Pairs\n")
    lines.append("**Best combinations** (good solvent + antisolvent):\n")
    lines.append("| Good Solvent | Antisolvent | Process |")
    lines.append("|--------------|-------------|---------|")

    for recommendation in recommendations[:8]:
        process = (
            f"Dissolve at {recommendation['dissolution_temp']:.0f} deg C, "
            f"add {recommendation['antisolvent']} to precipitate"
        )
        lines.append(
            f"| {recommendation['good_solvent']} ({recommendation['good_solubility']:.0f}%) | "
            f"{recommendation['antisolvent']} | {process} |"
        )

    lines.append("\n## Process Steps")
    lines.append(f"1. Dissolve {polymer} in good solvent at elevated temperature")
    lines.append("2. Cool solution to moderate temperature")
    lines.append("3. Slowly add antisolvent while stirring")
    lines.append(
        f"4. {polymer} precipitates out as antisolvent reduces solvent quality"
    )
    lines.append("5. Filter to collect precipitated polymer")

    return json_tool_success(
        "\n".join(lines),
        tool_name="find_antisolvent_pairs",
        polymer=polymer.upper(),
        min_good_solubility=min_good_solubility,
        max_antisolvent_solubility=max_antisolvent_solubility,
        good_solvent_count=len(good_payload),
        antisolvent_count=len(anti_payload),
        recommendation_count=min(len(recommendations), 8),
        good_solvents=good_payload,
        antisolvents=anti_payload,
        recommendations=recommendations[:8],
    )


@safe_tool_wrapper(structured_output=True)
def analyze_selective_antisolvent_precipitation(
    polymers: str,
    antisolvent: str = "auto",
) -> str:
    """Analyze whether adding an antisolvent can selectively precipitate polymers."""
    from strap.solubility import get_available_solvents, get_solubility

    polymer_list = [polymer.strip().upper() for polymer in polymers.split(",") if polymer.strip()]
    if len(polymer_list) < 2:
        return _precipitation_error(
            "analyze_selective_antisolvent_precipitation",
            "Need at least 2 polymers for selective precipitation analysis.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
            antisolvent=antisolvent,
        )

    all_solvents = get_available_solvents()
    results = {}
    for polymer in polymer_list:
        solubilities = {}
        for solvent in all_solvents:
            solubility = get_solubility(polymer, solvent, 25.0)
            if solubility is not None:
                solubilities[solvent] = solubility
        if solubilities:
            results[polymer] = solubilities

    if len(results) < 2:
        return _precipitation_error(
            "analyze_selective_antisolvent_precipitation",
            f"Insufficient solubility data for {', '.join(polymer_list)}.",
            error_code="insufficient_data",
            polymers=polymer_list,
            antisolvent=antisolvent,
        )

    common_solvents = set.intersection(*[set(result.keys()) for result in results.values()])
    differential_antisolvents = []
    for solvent in common_solvents:
        solubilities = {polymer: results[polymer].get(solvent, 100) for polymer in polymer_list}
        max_solubility = max(solubilities.values())
        min_solubility = min(solubilities.values())
        if max_solubility < 10 and (max_solubility - min_solubility) > 0.1:
            differential_antisolvents.append(
                {
                    "solvent": solvent,
                    "solubilities": solubilities,
                    "differential": max_solubility - min_solubility,
                    "max_solubility": max_solubility,
                    "min_solubility": min_solubility,
                }
            )

    differential_antisolvents.sort(key=lambda entry: entry["differential"], reverse=True)
    if antisolvent != "auto":
        differential_antisolvents = [
            entry
            for entry in differential_antisolvents
            if entry["solvent"].lower() == antisolvent.lower()
        ]

    lines = [
        "# Selective Antisolvent Precipitation Analysis\n",
        f"**Polymers:** {', '.join(polymer_list)}\n",
    ]

    if not differential_antisolvents:
        lines.append("## No Differential Antisolvents Found\n")
        lines.append("All tested antisolvents show similar rejection of all polymers.")
        lines.append(
            "Selective antisolvent precipitation may not be feasible for this polymer combination.\n"
        )
        lines.append("**Alternative:** Consider differential precipitation by cooling instead.")
        return json_tool_success(
            "\n".join(lines),
            tool_name="analyze_selective_antisolvent_precipitation",
            polymers=polymer_list,
            antisolvent=antisolvent,
            candidate_count=0,
            candidates=[],
        )

    lines.append(
        f"## Found {len(differential_antisolvents)} Antisolvents with Differential Response\n"
    )
    lines.append(
        "These antisolvents reject polymers at different rates, enabling selective precipitation.\n"
    )
    lines.append("| Antisolvent | " + " | ".join([f"{polymer} Sol." for polymer in polymer_list]) + " | Differential |")
    lines.append("|-------------|" + "|".join(["--------" for _ in polymer_list]) + "|--------------|")

    for candidate in differential_antisolvents[:10]:
        row = f"| {candidate['solvent']} |"
        for polymer in polymer_list:
            solubility = candidate["solubilities"][polymer]
            if solubility < 0.01:
                row += f" {solubility:.2e}% |"
            else:
                row += f" {solubility:.3f}% |"
        row += f" {candidate['differential']:.3f}% |"
        lines.append(row)

    best = differential_antisolvents[0]
    sorted_by_solubility = sorted(best["solubilities"].items(), key=lambda item: item[1], reverse=True)
    lines.append(f"\n## Recommended Process with {best['solvent'].upper()}\n")
    lines.append("**Precipitation order** (by antisolvent tolerance):\n")

    for index, (polymer, solubility) in enumerate(sorted_by_solubility, 1):
        if solubility < 0.01:
            lines.append(f"{index}. **{polymer}** - precipitates first (solubility: {solubility:.2e}%)")
        else:
            order = "last" if index == len(sorted_by_solubility) else "next"
            lines.append(
                f"{index}. **{polymer}** - precipitates {order} (solubility: {solubility:.3f}%)"
            )

    lines.append("\n**Process:**")
    lines.append("1. Dissolve all polymers in a common good solvent at elevated temperature")
    lines.append("2. Cool to moderate temperature (~50-60 deg C)")
    lines.append(f"3. Slowly add {best['solvent']} while stirring")
    lines.append(
        f"4. {sorted_by_solubility[0][0]} precipitates first (lowest antisolvent tolerance)"
    )
    lines.append(f"5. Filter to collect {sorted_by_solubility[0][0]}")
    if len(sorted_by_solubility) > 2:
        lines.append(
            f"6. Continue adding {best['solvent']} to precipitate remaining polymers sequentially"
        )
    else:
        lines.append(
            f"6. Add more {best['solvent']} to precipitate {sorted_by_solubility[1][0]}"
        )

    payload_candidates = [
        {
            "solvent": candidate["solvent"],
            "differential": float(candidate["differential"]),
            "solubilities": {
                polymer: float(solubility)
                for polymer, solubility in candidate["solubilities"].items()
            },
        }
        for candidate in differential_antisolvents[:10]
    ]
    return json_tool_success(
        "\n".join(lines),
        tool_name="analyze_selective_antisolvent_precipitation",
        polymers=polymer_list,
        antisolvent=antisolvent,
        candidate_count=len(differential_antisolvents),
        best_antisolvent=best["solvent"],
        precipitation_order=[polymer for polymer, _ in sorted_by_solubility],
        candidates=payload_candidates,
    )


__all__ = [
    "find_differential_precipitation_solvents",
    "analyze_multi_polymer_precipitation",
    "analyze_precipitation_temperature",
    "plot_precipitation_curves",
    "plot_atmospheric_feasibility",
    "compare_polymer_pairs_precipitation",
    "check_atmospheric_feasibility",
    "check_multi_polymer_atmospheric_feasibility",
    "find_antisolvents",
    "find_antisolvent_pairs",
    "analyze_selective_antisolvent_precipitation",
]
