"""Runtime search helpers for integrated sequence analysis tools."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from strap.solubility import temperature_extrapolation_status, temperature_use_regime


async def get_full_solvent_properties(
    solvent_names: list[str],
    *,
    solvent_table: str | None,
    lookup_solvent_properties: Callable[[list[str], str], Awaitable[dict[str, dict[str, Any]]]],
    connection,
) -> dict[str, dict[str, Any]]:
    """Load solvent properties and GSK metadata for integrated analysis."""
    property_lookup: dict[str, dict[str, Any]] = {}

    if solvent_table:
        try:
            prop_dict = await lookup_solvent_properties(solvent_names, solvent_table)
            if prop_dict:
                property_lookup.update(prop_dict)
        except Exception:
            pass

    try:
        placeholders = ", ".join(["?" for _ in solvent_names])
        gscore_query = f"""
        SELECT solvent_common_name, g_score, classification
        FROM gsk_dataset
        WHERE LOWER(solvent_common_name) IN ({placeholders})
        """
        gscore_df = connection.execute(gscore_query, [name.lower() for name in solvent_names]).fetchdf()
        if len(gscore_df) > 0:
            for _, row in gscore_df.iterrows():
                name = row["solvent_common_name"]
                for original_name in solvent_names:
                    if original_name.lower() == name.lower():
                        if original_name not in property_lookup:
                            property_lookup[original_name] = {}
                        property_lookup[original_name]["g_score"] = row["g_score"]
                        property_lookup[original_name]["gsk_class"] = row["classification"]
                        break
    except Exception:
        pass

    return property_lookup


async def find_optimal_separation(
    *,
    target: str,
    remaining: list[str],
    available_temps: list[float],
    rank_by: str,
    used_solvents: set[str] | None,
    get_solubility: Callable[[str, str, float], float | None],
    get_available_solvents: Callable[[], list[str] | set[str]],
    solvent_table: str | None,
    lookup_solvent_properties: Callable[[list[str], str], Awaitable[dict[str, dict[str, Any]]]],
    connection,
    min_selectivity_threshold: float,
) -> dict[str, Any]:
    """Find the best temperature-solvent combination for separating one polymer."""
    used_solvents = used_solvents or set()

    if not remaining:
        return {
            "solvent": "N/A",
            "temperature": 0,
            "selectivity": float("inf"),
            "target_sol": 100,
            "max_other": 0,
            "note": "Last polymer - no separation needed",
        }

    results: list[dict[str, Any]] = []
    for temp in available_temps:
        for solvent in get_available_solvents():
            target_sol = get_solubility(target, solvent, temp)
            if target_sol is None or target_sol <= 0:
                continue

            max_other = 0.0
            for polymer in remaining:
                solubility = get_solubility(polymer, solvent, temp)
                if solubility is not None and solubility > max_other:
                    max_other = solubility

            results.append(
                {
                    "solvent": solvent,
                    "temperature": temp,
                    "temperature_extrapolation": temperature_extrapolation_status(float(temp)),
                    "temperature_use_regime": temperature_use_regime(float(temp)),
                    "high_temperature_screening": temperature_use_regime(float(temp)) == "sensitivity_extrapolation",
                    "selectivity": target_sol - max_other,
                    "target_sol": target_sol,
                    "max_other": max_other,
                }
            )

    if not results:
        return {"solvent": "None found", "temperature": 0, "selectivity": 0}

    if used_solvents:
        unused_results = [result for result in results if result["solvent"].lower() not in {s.lower() for s in used_solvents}]
        if unused_results:
            results = unused_results
        else:
            for result in results:
                if result["solvent"].lower() in {s.lower() for s in used_solvents}:
                    result["reused_solvent"] = True

    results = [result for result in results if result.get("selectivity", 0) >= min_selectivity_threshold]
    if not results:
        return {
            "solvent": "No viable solvent",
            "temperature": 0,
            "selectivity": 0,
            "note": f"No solvent found with selectivity >= {min_selectivity_threshold}%",
        }

    solvent_names = list({result["solvent"] for result in results})
    property_lookup = await get_full_solvent_properties(
        solvent_names,
        solvent_table=solvent_table,
        lookup_solvent_properties=lookup_solvent_properties,
        connection=connection,
    )
    for result in results:
        if result["solvent"] in property_lookup:
            result.update(property_lookup[result["solvent"]])

    atmospheric_results = []
    for result in results:
        bp = result.get("bp")
        if bp is None:
            if result.get("high_temperature_screening"):
                result["missing_boiling_point"] = True
                continue
            atmospheric_results.append(result)
            continue
        try:
            bp_value = float(bp)
        except (TypeError, ValueError):
            atmospheric_results.append(result)
            continue
        if float(result.get("temperature", 0)) <= bp_value - 5.0:
            atmospheric_results.append(result)
        else:
            result["atmospheric_infeasible"] = True
    if atmospheric_results:
        results = atmospheric_results
    else:
        return {
            "solvent": "No viable solvent",
            "temperature": 0,
            "selectivity": 0,
            "note": "No solvent found within atmospheric boiling-point constraints",
        }

    rank_lower = rank_by.lower()
    if rank_lower in ["cost", "energy"]:
        valid = [result for result in results if result.get("selectivity", 0) > 0 and result.get("energy") is not None]
        if valid:
            valid.sort(key=lambda result: result["energy"])
            return valid[0]
    elif rank_lower in ["safety", "gscore", "g_score"]:
        valid = [result for result in results if result.get("selectivity", 0) > 0 and result.get("g_score") is not None]
        if valid:
            valid.sort(key=lambda result: result["g_score"], reverse=True)
            return valid[0]
    elif rank_lower in ["toxicity", "logp"]:
        valid = [result for result in results if result.get("selectivity", 0) > 0 and result.get("logp") is not None]
        if valid:
            valid.sort(key=lambda result: result["logp"])
            return valid[0]
    elif rank_lower in ["bp", "boiling", "boiling_point"]:
        valid = [result for result in results if result.get("selectivity", 0) > 0 and result.get("bp") is not None]
        if valid:
            valid.sort(key=lambda result: result["bp"])
            return valid[0]

    results.sort(key=lambda result: result.get("selectivity", 0), reverse=True)
    return results[0]


async def analyze_integrated_sequence(
    sequence: tuple[str, ...],
    *,
    find_optimal_separation_fn: Callable[..., Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Analyze one exhaustive integrated sequence."""
    steps: list[dict[str, Any]] = []
    total_score = 0.0
    used_solvents: set[str] = set()

    for step_idx, target in enumerate(sequence[:-1]):
        remaining = list(sequence[step_idx + 1:])
        best = await find_optimal_separation_fn(target=target, remaining=remaining, used_solvents=used_solvents)
        if best.get("solvent") and best["solvent"] not in ["None found", "No data", "Error", "N/A", "No viable solvent"]:
            used_solvents.add(best["solvent"])

        steps.append(
            {
                "step": step_idx + 1,
                "target": target,
                "remaining": remaining,
                "best": best,
            }
        )

        selectivity = best.get("selectivity", 0)
        if selectivity != float("inf"):
            total_score += selectivity

    steps.append(
        {
            "step": len(sequence),
            "target": sequence[-1],
            "remaining": [],
            "best": {"solvent": "N/A", "temperature": 0, "selectivity": float("inf"), "note": "Isolated"},
        }
    )

    min_selectivity = min(
        step["best"].get("selectivity", 0)
        for step in steps[:-1]
        if step["best"].get("selectivity", 0) != float("inf")
    ) if len(steps) > 1 else 0

    return {
        "sequence": sequence,
        "steps": steps,
        "total_score": total_score,
        "min_selectivity": min_selectivity,
    }


async def build_greedy_integrated_results(
    polymer_list: list[str],
    *,
    find_optimal_separation_fn: Callable[..., Awaitable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build the greedy fallback result set for larger integrated analyses."""
    remaining = list(polymer_list)
    greedy_sequence: list[str] = []
    greedy_steps: list[dict[str, Any]] = []
    used_solvents: set[str] = set()

    while len(remaining) > 1:
        best_candidate = None
        best_selectivity = -float("inf")
        for target in remaining:
            others = [polymer for polymer in remaining if polymer != target]
            result = await find_optimal_separation_fn(target=target, remaining=others, used_solvents=used_solvents)
            selectivity = result.get("selectivity", 0)
            if selectivity == float("inf"):
                selectivity = 0
            if selectivity > best_selectivity:
                best_selectivity = selectivity
                best_candidate = (target, result)

        if best_candidate is None:
            break

        target, best = best_candidate
        greedy_sequence.append(target)
        remaining.remove(target)
        if best.get("solvent") and best["solvent"] not in ["None found", "No data", "Error", "N/A", "No viable solvent"]:
            used_solvents.add(best["solvent"])
        greedy_steps.append(
            {
                "step": len(greedy_sequence),
                "target": target,
                "remaining": remaining.copy(),
                "best": best,
            }
        )

    if remaining:
        greedy_sequence.append(remaining[0])
        greedy_steps.append(
            {
                "step": len(greedy_sequence),
                "target": remaining[0],
                "remaining": [],
                "best": {"solvent": "N/A", "temperature": 0, "selectivity": float("inf"), "note": "Isolated"},
            }
        )

    min_selectivity = min(
        step["best"].get("selectivity", 0)
        for step in greedy_steps[:-1]
        if step["best"].get("selectivity", 0) != float("inf")
    ) if len(greedy_steps) > 1 else 0

    return [
        {
            "sequence": tuple(greedy_sequence),
            "steps": greedy_steps,
            "total_score": sum(
                step["best"].get("selectivity", 0)
                for step in greedy_steps[:-1]
                if step["best"].get("selectivity", 0) != float("inf")
            ),
            "min_selectivity": min_selectivity,
        }
    ]


async def run_exhaustive_integrated_analysis(
    sequences: list[tuple[str, ...]],
    *,
    analyze_sequence_fn: Callable[[tuple[str, ...]], Awaitable[dict[str, Any]]],
    concurrency: int,
) -> list[dict[str, Any]]:
    """Analyze all candidate integrated sequences with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def analyze_with_limit(sequence: tuple[str, ...]) -> dict[str, Any]:
        async with semaphore:
            return await analyze_sequence_fn(sequence)

    results = await asyncio.gather(*[analyze_with_limit(sequence) for sequence in sequences])
    results.sort(key=lambda result: result["min_selectivity"], reverse=True)
    return results
