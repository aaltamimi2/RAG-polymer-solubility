"""Sequence analysis and alternative-view tools for multi-polymer separation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from strap.database import get_connection
from strap.services.sequence_analysis_plot_service import (
    plot_integrated_separation_analysis,
)
from strap.services.sequence_analysis_runtime_service import (
    analyze_integrated_sequence,
    build_greedy_integrated_results,
    find_optimal_separation,
    run_exhaustive_integrated_analysis,
)
from strap.services.sequence_analysis_service import (
    build_alternative_sequence_display,
    build_integrated_analysis_display,
    select_alternative_sequence,
)
from strap.services.tool_response_service import json_tool_error, json_tool_success
from strap.services.visualization_service import (
    get_plot_url as _get_plot_url,
    get_solvent_table_name as _get_solvent_table_name,
    lookup_solvent_properties as _lookup_solvent_properties,
)
from strap.services.advanced_separation_service import (
    plot_separation_sequence as _plot_separation_sequence,
)
from strap.tools._helpers import safe_tool_wrapper

logger = logging.getLogger(__name__)


def _analysis_error(
    tool_name: str,
    message: str,
    *,
    error_code: str = "invalid_input",
    **data: Any,
) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)


_advanced_error = _analysis_error

@safe_tool_wrapper(structured_output=True)
async def analyze_integrated_separation(
    polymers: str,
    rank_by: str = "selectivity",
    top_k: int = 10,
    temperature_min: float = 25.0,
    temperature_max: float = 160.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """Multi-polymer separation analysis with optimal temperatures and integrated properties.

    Args:
        polymers: Comma-separated polymer list (e.g., "LDPE,EVOH,PET,PVC")
        rank_by: Ranking criterion - 'selectivity', 'cost', 'safety', 'toxicity', or 'bp'
        top_k: Top solvents per step (default: 10)
        temperature_min: Min search temperature in C (default: 25)
        temperature_max: Max search temperature in C (default: 160)

    WHEN TO USE:
    - "Find optimal temperatures for separating LDPE, EVOH, and PET"
    - "Rank solvents by safety for polymer separation"
    - "Comprehensive separation analysis with cost and toxicity"
    """
    from itertools import permutations

    conn = get_connection()

    polymer_list = [p.strip().upper() for p in polymers.split(",") if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return _advanced_error(
            "analyze_integrated_separation",
            "Need at least 2 polymers for separation analysis.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
        )

    if n_polymers > 3:
        message = (
            f"For {n_polymers} polymers, use `plan_sequential_separation` which uses efficient greedy algorithm. "
            "This exhaustive analysis tool is limited to <=3 polymers."
        )
        return _advanced_error(
            "analyze_integrated_separation",
            message,
            error_code="too_many_polymers",
            polymers=polymer_list,
            max_supported=3,
        )

    # Temperature range from interpolation model (25–160 °C, step 5)
    available_temps = [
        float(t) for t in range(
            max(int(temperature_min), 25),
            min(int(temperature_max), 160) + 1,
            5,
        )
    ]

    if not available_temps:
        return _advanced_error(
            "analyze_integrated_separation",
            f"No temperature data found between {temperature_min} C and {temperature_max} C",
            error_code="empty_temperature_window",
            polymers=polymer_list,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
        )

    # Minimum selectivity threshold
    MIN_SELECTIVITY_THRESHOLD = 5.0
    from strap.solubility import get_available_solvents as _get_available_solvents, get_solubility as _get_solubility

    async def _find_optimal_separation(target: str, remaining: list, used_solvents: Optional[set] = None) -> dict:
        return await find_optimal_separation(
            target=target,
            remaining=remaining,
            available_temps=available_temps,
            rank_by=rank_by,
            used_solvents=used_solvents,
            get_solubility=_get_solubility,
            get_available_solvents=_get_available_solvents,
            solvent_table=_get_solvent_table_name(),
            lookup_solvent_properties=_lookup_solvent_properties,
            connection=conn,
            min_selectivity_threshold=MIN_SELECTIVITY_THRESHOLD,
        )

    async def _analyze_sequence(sequence: tuple[str, ...]) -> dict:
        return await analyze_integrated_sequence(
            sequence,
            find_optimal_separation_fn=_find_optimal_separation,
        )

    # For large polymer sets, use greedy instead of exhaustive permutations
    MAX_EXHAUSTIVE = 6  # 6! = 720, 7! = 5040, 9! = 362880
    USE_GREEDY = n_polymers > MAX_EXHAUSTIVE

    if USE_GREEDY:
        all_results = await build_greedy_integrated_results(
            polymer_list,
            find_optimal_separation_fn=_find_optimal_separation,
        )
    else:
        all_sequences = list(permutations(polymer_list))
        all_results = await run_exhaustive_integrated_analysis(
            all_sequences,
            analyze_sequence_fn=_analyze_sequence,
            concurrency=5,
        )

    # Create visualisation for top sequence
    plot_url = None
    visualization_error = None
    try:
        filepath = plot_integrated_separation_analysis(
            polymer_list=polymer_list,
            best_result=all_results[0],
            rank_by=rank_by,
        )
        plot_url = _get_plot_url(filepath)

    except Exception as e:
        logger.error(f"Visualisation error: {e}", exc_info=True)
        visualization_error = str(e)

    return build_integrated_analysis_display(
        polymer_list=polymer_list,
        rank_by=rank_by,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        available_temps=available_temps,
        all_results=all_results,
        plot_url=plot_url,
        visualization_error=visualization_error,
        used_greedy=USE_GREEDY,
    )

# ===================================================================
# Tool: view_alternative_separation_sequence
# ===================================================================

@safe_tool_wrapper(structured_output=True)
async def view_alternative_separation_sequence(
    polymers: str,
    sequence_rank: Optional[int] = None,
    starting_polymer: Optional[str] = None,
    top_k_solvents: int = 5,
    temperature: float = 120.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """View a specific alternative separation sequence with clear visualisation.

    Use after plan_sequential_separation to explore different sequence options.

    Parameters:
    - polymers: Comma-separated list of polymers (e.g., "LDPE,HDPE,PP,PS")
    - sequence_rank: Rank of sequence to view (1=best, 2=2nd best, etc.)
    - starting_polymer: Name of polymer to start with (alternative to rank)
    - top_k_solvents: Number of top solvents to show per step (default: 5)
    - temperature: Target temperature in C (default: 120.0)

    WHEN TO USE:
    - "Show me the 2nd best separation sequence"
    - "What if we start with PET instead?"
    - "View LDPE-first separation option"
    """
    from itertools import permutations

    polymer_list = [p.strip() for p in polymers.split(",") if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return _advanced_error(
            "view_alternative_separation_sequence",
            "Need at least 2 polymers.",
            error_code="insufficient_polymers",
            polymers=polymer_list,
        )

    MAX_EXHAUSTIVE = 6  # 6! = 720 permutations

    async def find_top_solvents(target: str, remaining: list, k: int = 5) -> list:
        """Find top-k solvents for separating target from remaining polymers."""
        if not remaining:
            return [{"solvent": "N/A", "selectivity": float("inf"), "target_sol": 100, "max_other": 0}]

        # Use interpolation model instead of SQL
        from strap.solubility import get_all_solvents_selectivity as _get_all_sel
        all_sel = _get_all_sel(target, remaining, temperature)
        if not all_sel:
            return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

        results = [
            {
                "solvent": entry["solvent"],
                "selectivity": entry["selectivity"],
                "target_sol": entry["target_sol"],
                "max_other": entry["max_other_sol"],
            }
            for entry in all_sel
        ]
        return results[:k]

    async def analyze_sequence(sequence, seq_idx):
        """Analyze single sequence."""
        step_tasks = []
        step_info = []
        for step, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step:])
            step_tasks.append(find_top_solvents(target, remaining, top_k_solvents))
            step_info.append((step, target, remaining))

        all_step_results = await asyncio.gather(*step_tasks)

        total_min_selectivity = float("inf")
        seq_steps: list[dict] = []

        for (step, target, remaining), top_solvents in zip(step_info, all_step_results):
            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents,
            }
            seq_steps.append(step_data)

            if top_solvents and top_solvents[0]["selectivity"] < total_min_selectivity:
                total_min_selectivity = top_solvents[0]["selectivity"]

        return {
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps,
        }

    # For large polymer sets, limit permutation enumeration
    if n_polymers <= MAX_EXHAUSTIVE:
        all_sequences = list(permutations(polymer_list))

        semaphore = asyncio.Semaphore(10)

        async def analyze_with_limit(seq, idx):
            async with semaphore:
                return await analyze_sequence(seq, idx)

        sequence_analyses = await asyncio.gather(*[
            analyze_with_limit(seq, idx) for idx, seq in enumerate(all_sequences, 1)
        ])

        sequence_scores = sorted(sequence_analyses, key=lambda x: x["min_selectivity"], reverse=True)
    elif starting_polymer is not None:
        # Only generate permutations starting with the specified polymer: (n-1)!
        from itertools import permutations as _perms
        starting_polymer_normalized = starting_polymer.strip().upper()
        if starting_polymer_normalized not in [p.upper() for p in polymer_list]:
            return _advanced_error(
                "view_alternative_separation_sequence",
                f"'{starting_polymer}' not found in polymer list: {', '.join(polymer_list)}",
                error_code="starting_polymer_not_found",
                polymers=polymer_list,
                starting_polymer=starting_polymer,
            )
        others = [p for p in polymer_list if p.upper() != starting_polymer_normalized]
        start_p = next(p for p in polymer_list if p.upper() == starting_polymer_normalized)
        all_sequences = [tuple([start_p] + list(perm)) for perm in _perms(others)]

        semaphore = asyncio.Semaphore(10)

        async def analyze_with_limit(seq, idx):
            async with semaphore:
                return await analyze_sequence(seq, idx)

        sequence_analyses = await asyncio.gather(*[
            analyze_with_limit(seq, idx) for idx, seq in enumerate(all_sequences, 1)
        ])

        sequence_scores = sorted(sequence_analyses, key=lambda x: x["min_selectivity"], reverse=True)
    else:
        # Greedy approach for large n without starting_polymer
        import math
        from strap.solubility import get_all_solvents_selectivity as _get_all_sel_greedy

        remaining_g = list(polymer_list)
        greedy_seq: list[str] = []
        greedy_steps: list[dict] = []

        while len(remaining_g) > 1:
            best_candidate = None
            best_sel_val = -float("inf")
            for target in remaining_g:
                others = [p for p in remaining_g if p != target]
                all_sel = _get_all_sel_greedy(target, others, temperature)
                top_sel = all_sel[0]["selectivity"] if all_sel else 0
                if top_sel > best_sel_val:
                    best_sel_val = top_sel
                    top_solvents = [
                        {"solvent": e["solvent"], "selectivity": e["selectivity"],
                         "target_sol": e["target_sol"], "max_other": e["max_other_sol"]}
                        for e in (all_sel[:top_k_solvents] if all_sel else [])
                    ]
                    best_candidate = (target, top_solvents)

            if best_candidate is None:
                greedy_seq.extend(remaining_g)
                break
            target, solvents = best_candidate
            greedy_seq.append(target)
            remaining_g.remove(target)
            greedy_steps.append({
                "step": len(greedy_seq),
                "target": target,
                "remaining": remaining_g.copy(),
                "solvents": solvents if solvents else [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}],
            })

        greedy_seq.append(remaining_g[0])
        min_sel = min(
            s["solvents"][0]["selectivity"] for s in greedy_steps if s["solvents"]
        ) if greedy_steps else 0

        sequence_scores = [{
            "sequence": tuple(greedy_seq),
            "min_selectivity": min_sel,
            "steps": greedy_steps,
        }]

    target_seq, rank, selection_error = select_alternative_sequence(
        sequence_scores=sequence_scores,
        sequence_rank=sequence_rank,
        starting_polymer=starting_polymer,
        polymer_list=polymer_list,
        n_polymers=n_polymers,
        max_exhaustive=MAX_EXHAUSTIVE,
    )
    if selection_error is not None:
        return _advanced_error(
            "view_alternative_separation_sequence",
            selection_error["message"],
            error_code=selection_error["error_code"],
            polymers=polymer_list,
            **{
                key: value
                for key, value in selection_error.items()
                if key not in {"message", "error_code"}
            },
        )

    # Create visualisation using shared helper
    plot_url = None
    visualization_error = None
    try:
        filepath = _plot_separation_sequence(
            polymer_list, target_seq, temperature,
            total_sequences=len(sequence_scores), rank=rank,
        )
        plot_url = _get_plot_url(filepath)
    except Exception as e:
        logger.error(f"Visualisation error: {e}", exc_info=True)
        visualization_error = str(e)

    return json_tool_success(
        build_alternative_sequence_display(
            polymer_list=polymer_list,
            target_sequence=target_seq,
            sequence_scores=sequence_scores,
            rank=rank,
            temperature=temperature,
            plot_url=plot_url,
            visualization_error=visualization_error,
        ),
        tool_name="view_alternative_separation_sequence",
        polymers=polymer_list,
        sequence_rank=rank,
        starting_polymer=starting_polymer,
        temperature=temperature,
        selected_sequence=list(target_seq["sequence"]),
        min_selectivity=target_seq["min_selectivity"],
        steps=target_seq["steps"],
        total_sequences_evaluated=len(sequence_scores),
    )



__all__ = [
    "analyze_integrated_separation",
    "view_alternative_separation_sequence",
]
