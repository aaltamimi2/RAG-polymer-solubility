"""Payload builders and serialization helpers for sequence planning tools."""

from __future__ import annotations

import json
from typing import Any

from strap.solvent_registry import resolve_to_biosteam, resolve_to_property_db

_INVALID_SOLVENT_NAMES = {"", "N/A", "No data", "None found", "Error", "No viable solvent"}


def canonicalize_sequence_solvent_name(name: Any) -> str:
    """Return a user-facing canonical solvent name for sequence outputs."""

    text = str(name or "").strip()
    if text in _INVALID_SOLVENT_NAMES:
        return text
    normalized = (
        text.replace("₂", "2")
        .replace("₃", "3")
        .replace("₄", "4")
        .replace("₅", "5")
        .replace("₆", "6")
    )
    for candidate in (text, normalized):
        canonical = resolve_to_property_db(candidate) or resolve_to_biosteam(candidate)
        if canonical:
            return canonical
    return text


def _candidate_temperature(solvent_info: dict[str, Any], default_temperature: float) -> float | None:
    temperature: float | None = None
    for key in ("optimal_temp", "temperature", "temp", "temperature_c"):
        value = solvent_info.get(key)
        if value is None:
            continue
        try:
            temperature = float(value)
            break
        except (TypeError, ValueError):
            continue
    if temperature is None:
        try:
            temperature = float(default_temperature)
        except (TypeError, ValueError):
            return None

    bp = solvent_info.get("bp")
    if bp is not None:
        try:
            bp_value = float(bp)
        except (TypeError, ValueError):
            bp_value = None
        if bp_value is not None and temperature >= bp_value:
            return max(0.0, bp_value - 1.0)
    return temperature


def _build_polymer_solvent_candidates(
    *,
    sequence_scores: list[dict[str, Any]],
    temperature: float,
    max_candidates_per_polymer: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate ranked solvent candidates across all evaluated sequences.

    The best separation route leaves the final polymer as residue, so the route's
    own steps cannot expose solvent candidates for that final polymer. Aggregating
    across all evaluated permutations preserves candidates for every polymer when
    it appears as an active recovery target in an alternate sequence.
    """

    candidates: dict[str, list[dict[str, Any]]] = {}
    seen: dict[str, set[str]] = {}
    for sequence_rank, seq_data in enumerate(sequence_scores, start=1):
        for step in seq_data.get("steps", []) or []:
            polymer = str(step.get("target") or step.get("polymer") or "").strip()
            if not polymer:
                continue
            seen.setdefault(polymer, set())
            candidates.setdefault(polymer, [])
            if len(candidates[polymer]) >= max_candidates_per_polymer:
                continue
            for solvent_rank, solvent_info in enumerate(step.get("solvents", []) or [], start=1):
                if not isinstance(solvent_info, dict):
                    continue
                source_solvent = str(solvent_info.get("solvent") or "").strip()
                if source_solvent in _INVALID_SOLVENT_NAMES:
                    continue
                solvent = canonicalize_sequence_solvent_name(source_solvent)
                canonical = solvent.lower()
                if canonical in seen[polymer]:
                    continue
                seen[polymer].add(canonical)
                entry: dict[str, Any] = {
                    "rank": len(candidates[polymer]) + 1,
                    "solvent": solvent,
                    "source_sequence_rank": sequence_rank,
                    "source_step": step.get("step"),
                    "source_solvent_rank": solvent_rank,
                }
                if source_solvent and source_solvent != solvent:
                    entry["source_solvent"] = source_solvent
                temperature_c = _candidate_temperature(solvent_info, temperature)
                if temperature_c is not None:
                    entry["temperature_c"] = temperature_c
                if solvent_info.get("selectivity") is not None:
                    entry["selectivity_pct"] = solvent_info.get("selectivity")
                if solvent_info.get("optimal_selectivity") is not None:
                    entry["optimal_selectivity_pct"] = solvent_info.get("optimal_selectivity")
                if solvent_info.get("target_sol") is not None:
                    entry["target_solubility_pct"] = solvent_info.get("target_sol")
                if solvent_info.get("max_other") is not None:
                    entry["max_other_solubility_pct"] = solvent_info.get("max_other")
                if solvent_info.get("temperature_extrapolation") is not None:
                    entry["temperature_extrapolation"] = solvent_info.get("temperature_extrapolation")
                if solvent_info.get("temperature_use_regime") is not None:
                    entry["temperature_use_regime"] = solvent_info.get("temperature_use_regime")
                if solvent_info.get("optimal_temp_extrapolation") is not None:
                    entry["optimal_temp_extrapolation"] = solvent_info.get("optimal_temp_extrapolation")
                candidates[polymer].append(entry)
                if len(candidates[polymer]) >= max_candidates_per_polymer:
                    break
    return {polymer: values for polymer, values in candidates.items() if values}


def build_greedy_planning_payload(
    *,
    polymer_list: list[str],
    temperature: float,
    sequence: list[str],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable greedy planning payload."""
    valid_steps = [step for step in steps if step["selectivity"] > -900]
    solvents_used = list(
        {
            canonicalize_sequence_solvent_name(step["solvent"])
            for step in valid_steps
            if step["solvent"] != "N/A"
        }
    )
    solvent_mapping = {
        step["target"]: canonicalize_sequence_solvent_name(step["solvent"])
        for step in valid_steps
        if step["solvent"] != "N/A"
    }

    return {
        "tool_name": "plan_sequential_separation",
        "success": True,
        "polymers_analyzed": polymer_list,
        "best_sequence": sequence,
        "solvents": solvents_used,
        "selectivities": [step["selectivity"] for step in valid_steps],
        "temperature": temperature,
        "algorithm_used": "greedy",
        "steps": [
            {
                "step": step["step"],
                "target": step["target"],
                "solvent": canonicalize_sequence_solvent_name(step["solvent"]),
                "selectivity": step["selectivity"],
            }
            for step in steps
        ],
        "min_selectivity": min(step["selectivity"] for step in valid_steps)
        if valid_steps
        else None,
        "max_selectivity": max(step["selectivity"] for step in valid_steps)
        if valid_steps
        else None,
        "coverage_complete": len(sequence) == len(polymer_list),
        "top_k_sequences": [
            {
                "rank": 1,
                "sequence": sequence,
                "min_selectivity": min(step["selectivity"] for step in valid_steps)
                if valid_steps
                else 0,
                "solvent_mapping": solvent_mapping,
            }
        ],
        "total_sequences_evaluated": 1,
    }


def build_sequential_planning_payload(
    *,
    polymer_list: list[str],
    temperature: float,
    excluded_set: set[str],
    sequence_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable exhaustive planning payload."""
    best_seq = sequence_scores[0] if sequence_scores else {}
    best_steps = best_seq.get("steps", [])
    valid_steps = [
        step for step in best_steps if step.get("solvents") and step["solvents"][0].get("selectivity", -1000) > -900
    ]
    solvents_used = list(
        {
            step["solvents"][0].get("solvent", "N/A")
            for step in valid_steps
            if step["solvents"][0].get("solvent") != "N/A"
        }
    )
    solvents_used = [canonicalize_sequence_solvent_name(solvent) for solvent in solvents_used]

    top_k_sequences = []
    for rank, seq_data in enumerate(sequence_scores[:3], 1):
        solvent_mapping: dict[str, str] = {}
        route_steps: list[dict[str, Any]] = []
        for step in seq_data.get("steps", []):
            solvents_list = step.get("solvents", [])
            if not solvents_list or not isinstance(solvents_list[0], dict):
                continue
            source_best_sol = solvents_list[0].get("solvent")
            best_sol = canonicalize_sequence_solvent_name(source_best_sol)
            target = step.get("target")
            if target and best_sol and best_sol not in ["N/A", "No data", "None found", "Error"]:
                solvent_mapping[target] = best_sol
                route_step: dict[str, Any] = {
                    "step": step.get("step"),
                    "polymer": target,
                    "solvent": best_sol,
                }
                if source_best_sol and source_best_sol != best_sol:
                    route_step["source_solvent"] = source_best_sol
                temperature_c = _candidate_temperature(solvents_list[0], temperature)
                if temperature_c is not None:
                    route_step["temperature_c"] = temperature_c
                if solvents_list[0].get("selectivity") is not None:
                    route_step["selectivity_pct"] = solvents_list[0].get("selectivity")
                if solvents_list[0].get("temperature_extrapolation") is not None:
                    route_step["temperature_extrapolation"] = solvents_list[0].get("temperature_extrapolation")
                if solvents_list[0].get("temperature_use_regime") is not None:
                    route_step["temperature_use_regime"] = solvents_list[0].get("temperature_use_regime")
                route_steps.append(route_step)
        top_k_sequences.append(
            {
                "rank": rank,
                "sequence": seq_data.get("sequence", []),
                "min_selectivity": seq_data.get("min_selectivity", 0),
                "solvent_mapping": solvent_mapping,
                "steps": route_steps,
            }
        )

    return {
        "tool_name": "plan_sequential_separation",
        "success": True,
        "polymers_analyzed": polymer_list,
        "best_sequence": list(best_seq.get("sequence", [])),
        "solvents": solvents_used,
        "selectivities": [
            step["solvents"][0].get("selectivity", 0) for step in valid_steps
        ],
        "temperature": temperature,
        "algorithm_used": "exhaustive",
        "excluded_solvents": list(excluded_set) if excluded_set else [],
        "feedback_iteration": len(excluded_set) > 0,
        "steps": [
            {
                "step": index + 1,
                "target": step.get("target", ""),
                "solvent": canonicalize_sequence_solvent_name(step["solvents"][0].get("solvent", ""))
                if step.get("solvents")
                else "",
                "selectivity": step["solvents"][0].get("selectivity", 0) if step.get("solvents") else 0,
                "temperature_extrapolation": step["solvents"][0].get("temperature_extrapolation")
                if step.get("solvents")
                else None,
                "temperature_use_regime": step["solvents"][0].get("temperature_use_regime")
                if step.get("solvents")
                else None,
            }
            for index, step in enumerate(best_steps)
        ],
        "min_selectivity": min(
            step["solvents"][0].get("selectivity", 0) for step in valid_steps
        )
        if valid_steps
        else None,
        "max_selectivity": max(
            step["solvents"][0].get("selectivity", 0) for step in valid_steps
        )
        if valid_steps
        else None,
        "coverage_complete": len(best_seq.get("sequence", [])) == len(polymer_list),
        "polymer_solvent_candidates": _build_polymer_solvent_candidates(
            sequence_scores=sequence_scores,
            temperature=temperature,
        ),
        "top_k_sequences": top_k_sequences,
        "total_sequences_evaluated": len(sequence_scores),
    }


def dumps_tool_payload(display: str, payload: dict[str, Any]) -> str:
    """Serialize a tool response with preserved non-ASCII text."""
    return json.dumps({"display": display, "data": payload}, ensure_ascii=False)
