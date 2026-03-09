"""Sequence planning tools for multi-polymer separation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from strap.database import get_connection
from strap.services.sequence_planning_exhaustive_display_service import (
    build_sequence_analysis_output,
    build_sequential_planning_display,
)
from strap.services.sequence_planning_greedy_display_service import (
    build_greedy_planning_display,
    build_multi_scheme_display,
)
from strap.services.sequence_planning_payload_service import (
    build_greedy_planning_payload,
    build_sequential_planning_payload,
    dumps_tool_payload,
)
from strap.services.sequence_runtime_service import (
    build_greedy_scheme_variant,
    find_top_solvents_for_target,
    load_scheme_property_maps,
    rank_by_energy,
    rank_by_safety,
    rank_by_selectivity,
)
from strap.services.tool_response_service import json_tool_error
from strap.services.visualization_service import (
    get_plot_url as _get_plot_url,
    get_solvent_table_name as _get_solvent_table_name,
    lookup_solvent_properties as _lookup_solvent_properties,
)
from strap.services.advanced_separation_service import (
    plot_separation_sequence as _plot_separation_sequence,
    plot_topk_comparison as _plot_topk_comparison,
)
from strap.solvent_registry import ABBREVIATION_MAP as _ABBREVIATION_MAP
from strap.tools._helpers import safe_tool_wrapper

logger = logging.getLogger(__name__)


def _planning_error(
    tool_name: str,
    message: str,
    *,
    error_code: str = "invalid_input",
    **data: Any,
) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)


_advanced_error = _planning_error

async def _greedy_separation_planning(
    polymer_list: list,
    temperature: float,
    top_k_solvents: int,
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
) -> str:
    """Greedy algorithm for separation planning when n > 3 polymers.

    At each step, selects the polymer that can be most selectively separated
    from all remaining polymers. This is O(n^2) instead of O(n!).
    """
    from strap.solubility import get_selectivity as _get_selectivity

    remaining = list(polymer_list)
    sequence: list[str] = []
    steps: list[dict] = []
    evaluations: list[dict[str, Any]] = []
    used_solvents: set[str] = set()

    step_num = 0
    while len(remaining) > 1:
        step_num += 1
        candidates = []

        for target in remaining:
            others = [p for p in remaining if p != target]

            ret = _get_selectivity(target, others, temperature, used_solvents)
            if ret:
                best_solvent, selectivity, target_sol, max_other = ret
                candidates.append({
                    "polymer": target,
                    "solvent": best_solvent,
                    "selectivity": selectivity,
                    "target_sol": target_sol,
                    "others": others,
                })
            else:
                candidates.append({
                    "polymer": target,
                    "solvent": "N/A",
                    "selectivity": -999,
                    "target_sol": 0,
                    "others": others,
                })

        sorted_candidates = sorted(candidates, key=lambda x: x["selectivity"], reverse=True)
        best = max(candidates, key=lambda x: x["selectivity"])
        evaluations.append(
            {
                "step": step_num,
                "remaining": list(remaining),
                "candidates": sorted_candidates,
                "selected": best,
            }
        )

        sequence.append(best["polymer"])
        steps.append({
            "step": step_num,
            "target": best["polymer"],
            "solvent": best["solvent"],
            "selectivity": best["selectivity"],
            "remaining_before": list(remaining),
        })
        used_solvents.add(best["solvent"])
        remaining.remove(best["polymer"])

    if remaining:
        sequence.append(remaining[0])
    display = build_greedy_planning_display(
        polymer_list=polymer_list,
        temperature=temperature,
        evaluations=evaluations,
        sequence=sequence,
        steps=steps,
    )
    structured_data = build_greedy_planning_payload(
        polymer_list=polymer_list,
        temperature=temperature,
        sequence=sequence,
        steps=steps,
    )
    return dumps_tool_payload(display, structured_data)


# ===================================================================
# Tool: plan_multiple_separation_schemes (token-efficient multi-scheme)
# ===================================================================

@safe_tool_wrapper(structured_output=True)
async def plan_multiple_separation_schemes(
    polymers: str,
    temperature: float = 120.0,
    min_selectivity: float = 5.0,
    n_variants: int = 2,
) -> str:
    """Generate diverse separation schemes: max-selectivity, safest, lowest-energy, each with variants.

    Args:
        polymers: Comma-separated polymer list (e.g., "LDPE,HDPE,PP,PS,PET,PVC,EVOH,Nylon6,Nylon66")
        temperature: Target temperature in C (default: 120.0)
        min_selectivity: Min selectivity threshold for safety/energy schemes (default: 5.0)
        n_variants: Number of variants per scheme type (default: 2). Variant 1 is the greedy optimum;
            variant 2+ explore alternative first-step choices that cascade into different sequences.
    """
    from strap.solubility import (
        get_all_solvents_selectivity as _get_all_sel,
        get_available_solvents as _get_available_solvents,
    )

    polymer_list = [p.strip().upper() for p in polymers.split(",") if p.strip()]
    n = len(polymer_list)
    if n < 2:
        return _advanced_error(
            "plan_multiple_separation_schemes",
            "Need at least 2 polymers.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
        )

    # ---- Pre-load solvent properties (BP, LogP, G-score) once ----
    all_solvents = _get_available_solvents()

    solvent_table = _get_solvent_table_name()
    conn = get_connection()
    bp_map, logp_map, gscore_map = await load_scheme_property_maps(
        all_solvents=all_solvents,
        solvent_table=solvent_table,
        lookup_solvent_properties=_lookup_solvent_properties,
        connection=conn,
        abbreviation_map=_ABBREVIATION_MAP,
    )

    # ---- Run schemes with variants ----
    n_variants = max(1, min(n_variants, 5))  # clamp to [1, 5]
    scheme_defs = [
        ("Max Selectivity", "SEL", rank_by_selectivity),
        ("Safest Process (GSK)", "SAFE", lambda candidates: rank_by_safety(candidates, min_selectivity=min_selectivity, gscore_map=gscore_map)),
        ("Lowest Energy (BP)", "NRG", lambda candidates: rank_by_energy(candidates, min_selectivity=min_selectivity, bp_map=bp_map)),
    ]
    schemes = []
    for base_name, base_tag, rank_fn in scheme_defs:
        for v in range(n_variants):
            suffix = f" (v{v + 1})" if n_variants > 1 else ""
            tag = f"{base_tag}-v{v + 1}" if n_variants > 1 else base_tag
            schemes.append(
                build_greedy_scheme_variant(
                    polymer_list=polymer_list,
                    temperature=temperature,
                    get_all_selectivity=_get_all_sel,
                    rank_fn=rank_fn,
                    name=base_name + suffix,
                    tag=tag,
                    first_step_pick=v,
                    bp_map=bp_map,
                    gscore_map=gscore_map,
                    logp_map=logp_map,
                )
            )

    # Deduplicate: drop variants with identical polymer order AND solvent choices
    seen_keys: set[str] = set()
    unique_schemes = []
    for s in schemes:
        solvent_list = [st["solvent"] for st in s["steps"] if st["solvent"] != "-"]
        seq_key = ">".join(s["seq"]) + "|" + ",".join(solvent_list)
        if seq_key not in seen_keys:
            seen_keys.add(seq_key)
            unique_schemes.append(s)
    schemes = unique_schemes

    return build_multi_scheme_display(
        polymer_list=polymer_list,
        temperature=temperature,
        n_variants=n_variants,
        schemes=schemes,
    )


@safe_tool_wrapper(structured_output=True)
async def plan_sequential_separation(
    polymers: str,
    temperature: float = 120.0,
    top_k_solvents: int = 5,
    create_decision_tree: bool = True,
    excluded_solvents: str = "",
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """Plan optimal sequential separation sequences for multiple polymers.

    Args:
        polymers: Comma-separated polymer list (e.g., "LDPE,HDPE,PP,PS")
        temperature: Target temperature in C (default: 120.0)
        top_k_solvents: Top solvents to show per step (default: 5)
        create_decision_tree: Generate decision tree plot (default: True)
        excluded_solvents: Comma-separated solvents to exclude

    WHEN TO USE:
    - "Plan sequential separation for LDPE, HDPE, PP, and PS"
    - "What's the best order to separate mixed plastics?"
    - "Design a multi-step polymer separation process"
    """
    from itertools import permutations

    # Parse polymers
    polymer_list = [p.strip() for p in polymers.split(",") if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return _advanced_error(
            "plan_sequential_separation",
            "Need at least 2 polymers for separation planning.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
        )

    # Parse excluded solvents (from feedback loop cost constraints)
    excluded_set: set[str] = set()
    if excluded_solvents:
        excluded_set = {s.strip().lower() for s in excluded_solvents.split(",") if s.strip()}
        logger.info(f"Excluding solvents from feedback loop: {excluded_set}")

    # For >4 polymers, use greedy algorithm instead of exhaustive search
    USE_GREEDY = n_polymers > 4

    if USE_GREEDY:
        return await _greedy_separation_planning(
            polymer_list, temperature, top_k_solvents,
            table_name, polymer_column, solvent_column,
            temperature_column, solubility_column,
        )

    # Generate all permutations (only for <=4 polymers)
    all_sequences = list(permutations(polymer_list))
    n_sequences = len(all_sequences)

    # Minimum selectivity threshold for viable separation
    MIN_SELECTIVITY = 5.0

    # Async function to analyze a single sequence with solvent diversity tracking
    async def analyze_sequence(sequence, seq_idx):
        """Analyze a single sequence, enforcing different solvents for each step."""
        used_solvents: set[str] = set()
        total_min_selectivity = float("inf")
        seq_steps: list[dict] = []

        for step, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step:])
            from strap.solubility import (
                get_available_solvents_for_polymer as _get_available_solvents_for_polymer,
                get_solubility as _get_solubility,
            )

            top_solvents = await find_top_solvents_for_target(
                target=target,
                remaining=remaining,
                temperature=temperature,
                top_k=top_k_solvents,
                used_solvents=used_solvents,
                excluded_solvents=excluded_set,
                min_selectivity=MIN_SELECTIVITY,
                solvent_column=solvent_column,
                polymer_column=polymer_column,
                get_solubility=_get_solubility,
                get_available_solvents_for_polymer=_get_available_solvents_for_polymer,
                solvent_table=_get_solvent_table_name(),
                lookup_solvent_properties=_lookup_solvent_properties,
            )

            if top_solvents and top_solvents[0].get("solvent") not in ["N/A", "No data", "None found", "Error"]:
                used_solvents.add(top_solvents[0]["solvent"])

            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents,
            }
            seq_steps.append(step_data)

            if top_solvents and "selectivity" in top_solvents[0]:
                best_selectivity = top_solvents[0]["selectivity"]
                total_min_selectivity = min(total_min_selectivity, best_selectivity)

        return {
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps,
            "output": build_sequence_analysis_output(
                sequence=sequence,
                seq_idx=seq_idx,
                seq_steps=seq_steps,
            ),
        }

    # Analyze all sequences in parallel with limited concurrency
    semaphore = asyncio.Semaphore(10)

    async def analyze_with_limit(sequence, seq_idx):
        async with semaphore:
            return await analyze_sequence(sequence, seq_idx)

    sequence_results = await asyncio.gather(*[
        analyze_with_limit(seq, idx)
        for idx, seq in enumerate(all_sequences, 1)
    ])

    sequence_scores: list[dict] = []
    for result in sequence_results:
        sequence_scores.append({
            "sequence": result["sequence"],
            "min_selectivity": result["min_selectivity"],
            "steps": result["steps"],
        })

    sequence_scores.sort(key=lambda x: x["min_selectivity"], reverse=True)

    rank1_plot_url = None
    topk_plot_url = None
    visualization_errors: list[str] = []
    if create_decision_tree and sequence_scores:
        try:
            filepath = _plot_separation_sequence(
                polymer_list, sequence_scores[0], temperature,
                total_sequences=len(sequence_scores), rank=1,
            )
            rank1_plot_url = _get_plot_url(filepath)
        except Exception as e:
            logger.error(f"Decision tree error: {e}", exc_info=True)
            visualization_errors.append(str(e))

        if len(sequence_scores) >= 2:
            try:
                filepath = _plot_topk_comparison(polymer_list, sequence_scores, temperature)
                topk_plot_url = _get_plot_url(filepath)
            except Exception as e:
                logger.error(f"Top-K comparison visualisation error: {e}", exc_info=True)
                visualization_errors.append(str(e))

    display = build_sequential_planning_display(
        polymer_list=polymer_list,
        temperature=temperature,
        top_k_solvents=top_k_solvents,
        excluded_set=excluded_set,
        all_sequences=all_sequences,
        sequence_results=sequence_results,
        sequence_scores=sequence_scores,
        rank1_plot_url=rank1_plot_url,
        topk_plot_url=topk_plot_url,
        visualization_errors=visualization_errors,
    )
    structured_data = build_sequential_planning_payload(
        polymer_list=polymer_list,
        temperature=temperature,
        excluded_set=excluded_set,
        sequence_scores=sequence_scores,
    )
    return dumps_tool_payload(display, structured_data)

__all__ = [
    "_planning_error",
    "_greedy_separation_planning",
    "plan_multiple_separation_schemes",
    "plan_sequential_separation",
]
