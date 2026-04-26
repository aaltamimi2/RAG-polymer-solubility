"""Runtime search helpers for sequence planning tools."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from strap.solubility import (
    FITTED_TEMP_MAX_C,
    FITTED_TEMP_MIN_C,
    SENSITIVITY_EXTRAPOLATION_MAX_C,
    temperature_extrapolation_status,
    temperature_use_regime,
)


def rank_by_selectivity(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda candidate: candidate.get("selectivity", -999), reverse=True)


def rank_by_safety(
    candidates: list[dict[str, Any]],
    *,
    min_selectivity: float,
    gscore_map: dict[str, float],
) -> list[dict[str, Any]]:
    viable = [
        candidate
        for candidate in candidates
        if candidate.get("selectivity", -999) >= min_selectivity
        and candidate.get("solvent", "") in gscore_map
    ]
    if not viable:
        viable = [candidate for candidate in candidates if candidate.get("solvent", "") in gscore_map]
    if not viable:
        return rank_by_selectivity(candidates)
    return sorted(viable, key=lambda candidate: gscore_map.get(candidate["solvent"], 0), reverse=True)


def rank_by_energy(
    candidates: list[dict[str, Any]],
    *,
    min_selectivity: float,
    bp_map: dict[str, float],
) -> list[dict[str, Any]]:
    viable = [
        candidate
        for candidate in candidates
        if candidate.get("selectivity", -999) >= min_selectivity
        and candidate.get("solvent", "") in bp_map
    ]
    if not viable:
        viable = [candidate for candidate in candidates if candidate.get("solvent", "") in bp_map]
    if not viable:
        return rank_by_selectivity(candidates)
    return sorted(viable, key=lambda candidate: bp_map.get(candidate["solvent"], 999))


def build_greedy_scheme_variant(
    *,
    polymer_list: list[str],
    temperature: float,
    get_all_selectivity: Callable[[str, list[str], float], list[dict[str, Any]]],
    rank_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    name: str,
    tag: str,
    first_step_pick: int,
    bp_map: dict[str, float],
    gscore_map: dict[str, float],
    logp_map: dict[str, float],
) -> dict[str, Any]:
    """Run one greedy multi-scheme variant."""
    remaining = list(polymer_list)
    steps: list[dict[str, Any]] = []
    used_solvents: set[str] = set()
    is_first_step = True

    while len(remaining) > 1:
        candidates = []
        for target in remaining:
            others = [polymer for polymer in remaining if polymer != target]
            all_selectivities = get_all_selectivity(target, others, temperature)
            if used_solvents:
                all_selectivities = [
                    selectivity
                    for selectivity in all_selectivities
                    if selectivity["solvent"] not in used_solvents
                ]
            if not all_selectivities:
                candidates.append({"polymer": target, "solvent": "N/A", "selectivity": -999})
                continue
            ranked = rank_fn(all_selectivities)
            best = ranked[0]
            candidates.append(
                {
                    "polymer": target,
                    "solvent": best["solvent"],
                    "selectivity": best["selectivity"],
                }
            )

        ranked_candidates = rank_fn(candidates)
        pick_index = 0
        if is_first_step and first_step_pick > 0:
            pick_index = min(first_step_pick, len(ranked_candidates) - 1)
        is_first_step = False
        winner = ranked_candidates[pick_index]

        solvent_bp = bp_map.get(winner["solvent"])
        operating_temp = min(temperature, solvent_bp - 5) if solvent_bp is not None else temperature
        steps.append(
            {
                "step": len(steps) + 1,
                "target": winner["polymer"],
                "solvent": winner["solvent"],
                "sel": winner["selectivity"],
                "temp": operating_temp,
                "bp": solvent_bp,
                "gsk": gscore_map.get(winner["solvent"]),
                "logp": logp_map.get(winner["solvent"]),
            }
        )
        used_solvents.add(winner["solvent"])
        remaining.remove(winner["polymer"])

    if remaining:
        steps.append(
            {
                "step": len(steps) + 1,
                "target": remaining[0],
                "solvent": "-",
                "sel": None,
                "temp": None,
                "bp": None,
                "gsk": None,
                "logp": None,
            }
        )

    valid_selectivities = [
        step["sel"] for step in steps if step["sel"] is not None and step["sel"] > -900
    ]
    return {
        "name": name,
        "tag": tag,
        "steps": steps,
        "seq": [step["target"] for step in steps],
        "min_sel": min(valid_selectivities) if valid_selectivities else 0,
        "avg_sel": sum(valid_selectivities) / len(valid_selectivities) if valid_selectivities else 0,
        "n_solv": len({step["solvent"] for step in steps if step["solvent"] != "-"}),
    }


async def load_scheme_property_maps(
    *,
    all_solvents: list[str] | set[str],
    solvent_table: str | None,
    lookup_solvent_properties: Callable[[list[str], str], Awaitable[dict[str, dict[str, Any]]]],
    connection,
    abbreviation_map: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Load BP, LogP, and G-score maps for multi-scheme planning."""
    bp_map: dict[str, float] = {}
    logp_map: dict[str, float] = {}
    gscore_map: dict[str, float] = {}

    all_solvent_list = list(all_solvents)
    if solvent_table:
        prop_dict = await lookup_solvent_properties(all_solvent_list, solvent_table)
        for solvent_name, props in prop_dict.items():
            if props.get("bp") is not None:
                try:
                    bp_map[solvent_name] = float(props["bp"])
                except (ValueError, TypeError):
                    pass
            if props.get("logp") is not None:
                try:
                    logp_map[solvent_name] = float(props["logp"])
                except (ValueError, TypeError):
                    pass

    try:
        gsk_df = connection.execute(
            "SELECT solvent_common_name, g_score FROM gsk_dataset"
        ).fetchdf()
        gsk_lower: dict[str, float] = {}
        for _, row in gsk_df.iterrows():
            if row["g_score"] is not None:
                gsk_lower[row["solvent_common_name"].lower()] = float(row["g_score"])

        for solvent_name in all_solvent_list:
            solvent_lower = solvent_name.lower()
            if solvent_lower in gsk_lower:
                gscore_map[solvent_name] = gsk_lower[solvent_lower]
                continue

            expanded = abbreviation_map.get(solvent_lower, solvent_lower)
            if expanded.lower() in gsk_lower:
                gscore_map[solvent_name] = gsk_lower[expanded.lower()]
                continue

            normalized = solvent_lower.replace("-", "").replace(" ", "")
            for gsk_name, gsk_value in gsk_lower.items():
                gsk_normalized = gsk_name.replace("-", "").replace(" ", "")
                if normalized in gsk_normalized or gsk_normalized in normalized:
                    gscore_map[solvent_name] = gsk_value
                    break
    except Exception:
        pass

    return bp_map, logp_map, gscore_map


async def find_top_solvents_for_target(
    *,
    target: str,
    remaining: list[str],
    temperature: float,
    top_k: int,
    used_solvents: set[str] | None,
    excluded_solvents: set[str] | None,
    min_selectivity: float,
    solvent_column: str,
    polymer_column: str,
    get_solubility: Callable[[str, str, float], float | None],
    get_available_solvents_for_polymer: Callable[[str], list[str]],
    solvent_table: str | None,
    lookup_solvent_properties: Callable[[list[str], str], Awaitable[dict[str, dict[str, Any]]]],
    temperature_scan_min: float = FITTED_TEMP_MIN_C,
    temperature_scan_max: float | None = None,
) -> list[dict[str, Any]]:
    """Find the top-K solvents for one sequential separation step."""
    used_solvents = used_solvents or set()
    excluded_solvents = excluded_solvents or set()
    if temperature > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return [
            {
                "solvent": "None found",
                "selectivity": 0,
                "target_sol": 0,
                "max_other": 0,
                "note": f"Temperature exceeds supported sensitivity limit of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C",
            }
        ]
    if temperature_scan_max is None:
        temperature_scan_max = max(FITTED_TEMP_MAX_C, float(temperature))
    temperature_scan_max = min(float(temperature_scan_max), SENSITIVITY_EXTRAPOLATION_MAX_C)
    temperature_scan_min = max(float(temperature_scan_min), FITTED_TEMP_MIN_C)
    high_temperature_screening = temperature_use_regime(float(temperature)) == "sensitivity_extrapolation"

    if not remaining:
        return [
            {
                "solvent": "N/A",
                "selectivity": float("inf"),
                "target_sol": 100,
                "max_other": 0,
                "temperature": temperature,
                "optimal_temp": temperature,
                "note": "Last polymer - no separation needed",
            }
        ]

    all_polymers = [target] + remaining
    solvents_available = get_available_solvents_for_polymer(target)

    current_rows: list[dict[str, Any]] = []
    all_temp_rows: list[dict[str, Any]] = []
    for solvent in solvents_available:
        for polymer in all_polymers:
            current_solubility = get_solubility(polymer, solvent, temperature)
            if current_solubility is not None:
                current_rows.append({solvent_column: solvent, polymer_column: polymer, "avg_sol": current_solubility})
            start_temp = int(round(temperature_scan_min))
            stop_temp = int(round(temperature_scan_max))
            for temp in range(start_temp, stop_temp + 1, 5):
                solubility = get_solubility(polymer, solvent, float(temp))
                if solubility is not None and solubility > 0:
                    all_temp_rows.append(
                        {solvent_column: solvent, polymer_column: polymer, "temp": float(temp), "avg_sol": solubility}
                    )

    if not current_rows:
        return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

    optimal_by_solvent: dict[str, dict[str, float]] = {}
    for row in all_temp_rows:
        if row[polymer_column] != target:
            continue
        solvent = row[solvent_column]
        temp = row["temp"]
        target_sol = row["avg_sol"]
        max_other = max(
            (
                candidate["avg_sol"]
                for candidate in all_temp_rows
                if candidate[solvent_column] == solvent
                and candidate["temp"] == temp
                and candidate[polymer_column] in remaining
            ),
            default=0,
        )
        selectivity = target_sol - max_other
        if solvent not in optimal_by_solvent or selectivity > optimal_by_solvent[solvent]["selectivity"]:
            optimal_by_solvent[solvent] = {"temp": temp, "selectivity": selectivity}

    results: list[dict[str, Any]] = []
    for solvent in {row[solvent_column] for row in current_rows}:
        target_values = [
            row["avg_sol"]
            for row in current_rows
            if row[solvent_column] == solvent and row[polymer_column] == target
        ]
        if not target_values:
            continue
        target_sol = target_values[0]
        max_other = max(
            (
                row["avg_sol"]
                for row in current_rows
                if row[solvent_column] == solvent and row[polymer_column] in remaining
            ),
            default=0,
        )
        selectivity = target_sol - max_other
        optimal = optimal_by_solvent.get(solvent, {"temp": temperature, "selectivity": selectivity})
        results.append(
            {
                "solvent": solvent,
                "selectivity": selectivity,
                "target_sol": target_sol,
                "max_other": max_other,
                "temperature": temperature,
                "optimal_temp": optimal["temp"],
                "optimal_selectivity": optimal["selectivity"],
                "temperature_extrapolation": temperature_extrapolation_status(float(temperature)),
                "optimal_temp_extrapolation": temperature_extrapolation_status(float(optimal["temp"])),
                "temperature_use_regime": temperature_use_regime(float(temperature)),
                "high_temperature_screening": high_temperature_screening,
            }
        )

    results.sort(key=lambda result: result["selectivity"], reverse=True)

    if used_solvents:
        used_lower = {solvent.lower() for solvent in used_solvents}
        unused_results = [result for result in results if result["solvent"].lower() not in used_lower]
        if unused_results:
            results = unused_results
        else:
            for result in results:
                if result["solvent"].lower() in used_lower:
                    result["reused"] = True

    if excluded_solvents:
        excluded_lower = {solvent.lower() for solvent in excluded_solvents}
        filtered_results = [result for result in results if result["solvent"].lower() not in excluded_lower]
        if filtered_results:
            results = filtered_results
            for result in results[:3]:
                result["feedback_constrained"] = True
        else:
            for result in results:
                if result["solvent"].lower() in excluded_lower:
                    result["excluded_expensive"] = True

    if solvent_table and results:
        try:
            solvent_names = sorted({result["solvent"] for result in results})
            property_lookup = await lookup_solvent_properties(solvent_names, solvent_table)
            for result in results:
                if result["solvent"] in property_lookup:
                    result.update(
                        {
                            key: value
                            for key, value in property_lookup[result["solvent"]].items()
                            if value is not None
                        }
                    )
            atmospheric_results = []
            for result in results:
                bp = result.get("bp")
                if bp is None:
                    if high_temperature_screening:
                        result["missing_boiling_point"] = True
                        continue
                    atmospheric_results.append(result)
                    continue
                try:
                    bp_value = float(bp)
                except (TypeError, ValueError):
                    atmospheric_results.append(result)
                    continue
                if float(result.get("temperature", temperature)) <= bp_value - 5.0:
                    atmospheric_results.append(result)
                else:
                    result["atmospheric_infeasible"] = True
            if atmospheric_results:
                results = atmospheric_results
            else:
                results = []
        except Exception:
            pass

    viable_results = [result for result in results if result.get("selectivity", 0) >= min_selectivity]
    if viable_results:
        results = viable_results

    if solvent_table and results:
        try:
            solvent_names = [result["solvent"] for result in results[:top_k]]
            property_lookup = await lookup_solvent_properties(solvent_names, solvent_table)
            for result in results:
                if result["solvent"] in property_lookup:
                    result.update(
                        {
                            key: value
                            for key, value in property_lookup[result["solvent"]].items()
                            if value is not None
                        }
                    )
        except Exception:
            pass

    return results[:top_k] if results else [{"solvent": "None found", "selectivity": 0, "target_sol": 0, "max_other": 0}]
