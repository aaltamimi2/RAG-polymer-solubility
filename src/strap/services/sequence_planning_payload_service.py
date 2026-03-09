"""Payload builders and serialization helpers for sequence planning tools."""

from __future__ import annotations

import json
from typing import Any


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
        {step["solvent"] for step in valid_steps if step["solvent"] != "N/A"}
    )
    solvent_mapping = {
        step["target"]: step["solvent"]
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
                "solvent": step["solvent"],
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

    top_k_sequences = []
    for rank, seq_data in enumerate(sequence_scores[:3], 1):
        solvent_mapping: dict[str, str] = {}
        for step in seq_data.get("steps", []):
            solvents_list = step.get("solvents", [])
            if not solvents_list or not isinstance(solvents_list[0], dict):
                continue
            best_sol = solvents_list[0].get("solvent")
            target = step.get("target")
            if target and best_sol and best_sol not in ["N/A", "No data", "None found", "Error"]:
                solvent_mapping[target] = best_sol
        top_k_sequences.append(
            {
                "rank": rank,
                "sequence": seq_data.get("sequence", []),
                "min_selectivity": seq_data.get("min_selectivity", 0),
                "solvent_mapping": solvent_mapping,
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
                "solvent": step["solvents"][0].get("solvent", "") if step.get("solvents") else "",
                "selectivity": step["solvents"][0].get("selectivity", 0) if step.get("solvents") else 0,
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
        "top_k_sequences": top_k_sequences,
        "total_sequences_evaluated": len(sequence_scores),
    }


def dumps_tool_payload(display: str, payload: dict[str, Any]) -> str:
    """Serialize a tool response with preserved non-ASCII text."""
    return json.dumps({"display": display, "data": payload}, ensure_ascii=False)
