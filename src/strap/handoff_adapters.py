"""Typed downstream handoff adapters.

These adapters translate validated upstream result payloads into
consumer-specific contracts and task prompts. Generic fallback handoffs stay
in ``handoffs.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .handoff_models import HandoffRecord


def _build_biosteam_prompt(sequence_candidates: list[dict[str, Any]]) -> str:
    lines = ["Run multi-polymer BioSTEAM for these alternative sequences:"]
    for candidate in sequence_candidates:
        polymers_json = json.dumps(candidate["polymers_json"], ensure_ascii=False)
        rank = candidate.get("rank", "?")
        lines.append(f"Seq {rank}: {polymers_json}")
    return "\n".join(lines)


def _infer_contaminant_mode(prompt: str | None) -> str:
    text = (prompt or "").lower()
    if "leach" in text or "leaching" in text:
        return "leaching"
    if "strap contaminant" in text or "temperature-swing" in text or "temperature swing" in text:
        return "strap_contaminant_removal"
    return "comparison"


def _infer_visualization_request(task_prompt: str | None) -> tuple[str, str, str]:
    prompt = (task_prompt or "").lower()
    if "heatmap" in prompt:
        return (
            "selectivity_heatmap",
            "create_selectivity_heatmap",
            "selectivity heatmap",
        )
    if "dashboard" in prompt:
        return (
            "comparison_dashboard",
            "plot_comparison_dashboard",
            "comparison dashboard",
        )
    if "process flow" in prompt or "flow diagram" in prompt:
        return (
            "process_flow_diagram",
            "create_process_flow_diagram",
            "process flow diagram",
        )
    if "multi-panel" in prompt or "multi panel" in prompt:
        return (
            "multi_panel_analysis",
            "plot_multi_panel_analysis",
            "multi-panel analysis",
        )
    if "atmospheric feasibility" in prompt or "feasibility plot" in prompt:
        return (
            "atmospheric_feasibility",
            "plot_atmospheric_feasibility",
            "atmospheric feasibility plot",
        )
    return (
        "separation_tree",
        "create_separation_tree_plot",
        "separation tree plot",
    )


def _partition_supported_polymers(polymers: list[str]) -> tuple[list[str], list[str]]:
    from .solubility import get_available_polymers, resolve_polymer

    known_polymers = get_available_polymers()
    supported: list[str] = []
    unsupported: list[str] = []
    for polymer in polymers:
        normalized = str(polymer).strip().upper()
        if not normalized:
            continue
        resolved = resolve_polymer(normalized, known_polymers)
        if resolved:
            supported.append(resolved)
        else:
            unsupported.append(normalized)
    return supported, unsupported


def _canonical_optimization_polymer(polymer: Any) -> str | None:
    text = str(polymer or "").strip().upper()
    if text in {"PE", "HDPE", "LDPE"}:
        return "PE"
    if text == "EVOH":
        return "EVOH"
    return None


def _build_optimization_solvent_filters(payload: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    from .waste_management.data_loader import S_EV1, S_EV2, S_PE

    pe_known = set(S_PE)
    evoh_known = set(S_EV1) | set(S_EV2)

    polymer_filters: dict[str, list[str]] = {"PE": [], "EVOH": []}
    global_candidates: list[str] = []

    def add_global(solvent: str) -> None:
        if solvent and solvent not in global_candidates:
            global_candidates.append(solvent)

    def add_polymer(polymer: str, solvent: str) -> None:
        if solvent and solvent not in polymer_filters[polymer]:
            polymer_filters[polymer].append(solvent)
            add_global(solvent)

    def route_solvent(solvent: Any, polymer: Any = None) -> None:
        solvent_name = str(solvent or "").strip()
        if not solvent_name:
            return
        canonical_polymer = _canonical_optimization_polymer(polymer)
        if canonical_polymer == "PE" and solvent_name in pe_known:
            add_polymer("PE", solvent_name)
            return
        if canonical_polymer == "EVOH" and solvent_name in evoh_known:
            add_polymer("EVOH", solvent_name)
            return
        if solvent_name in pe_known:
            add_polymer("PE", solvent_name)
        if solvent_name in evoh_known:
            add_polymer("EVOH", solvent_name)

    for step in payload.get("steps") or []:
        if isinstance(step, dict):
            route_solvent(step.get("solvent"), step.get("polymer"))

    for polymer, solvent in (payload.get("solvent_mapping") or {}).items():
        route_solvent(solvent, polymer)

    for sequence in payload.get("top_k_sequences") or []:
        if not isinstance(sequence, dict):
            continue
        for polymer, solvent in (sequence.get("solvent_mapping") or {}).items():
            route_solvent(solvent, polymer)

    for solvent in payload.get("top_solvents") or []:
        route_solvent(solvent)

    polymer_filters = {polymer: solvents for polymer, solvents in polymer_filters.items() if solvents}
    return polymer_filters, global_candidates


def _build_visualization_task_prompt(
    *,
    polymers: list[str],
    plot_polymers: list[str] | None = None,
    temperature: float | None,
    requested_label: str,
    preferred_tool: str,
    suggested_solvents: list[str],
    request_context: str | None = None,
    unsupported_polymers: list[str] | None = None,
) -> str:
    display_polymers = plot_polymers or polymers
    polymer_text = ",".join(display_polymers)
    temp_suffix = f" at {temperature}C" if temperature is not None else ""
    tool_args = [f'polymers="{polymer_text}"']
    if suggested_solvents:
        tool_args.append(f'solvents="{",".join(suggested_solvents)}"')
    if temperature is not None:
        tool_args.append(f"temperature={temperature}")

    lines = [
        f"The user specifically requested a {requested_label} for {polymer_text}{temp_suffix}.",
        f"Required tool: {preferred_tool}",
        f"Required call pattern: {preferred_tool}({', '.join(tool_args)})",
        "Use the provided separation result and existing handoff context; do not rerun the upstream separation analysis.",
    ]
    if request_context:
        lines.append(f"Original user request: {request_context}")
    if preferred_tool != "create_separation_tree_plot":
        lines.append("Do not substitute create_separation_tree_plot or any other plot type for this request.")
    if unsupported_polymers:
        lines.append(
            "Only visualize the supported subset with actual data coverage. "
            f"Do not imply the plot contains data for unsupported polymers: {', '.join(unsupported_polymers)}."
        )
    if suggested_solvents:
        lines.append(f"Preferred solvent list for the plot: {', '.join(suggested_solvents)}.")
    lines.append("Create one visualization that directly answers the request, then synthesize.")
    return " ".join(lines)


def _adapt_separation_to_biosteam(source: HandoffRecord) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    candidates = payload.get("top_k_sequences", [])
    sequence_candidates: list[dict[str, Any]] = []

    for item in candidates:
        solvent_mapping = item.get("solvent_mapping") or {}
        ordered_sequence = item.get("sequence") or []
        polymers_json = [
            {"polymer": polymer, "solvent": solvent_mapping[polymer]}
            for polymer in ordered_sequence
            if polymer in solvent_mapping
        ]
        sequence_candidates.append(
            {
                "rank": item.get("rank"),
                "sequence": ordered_sequence,
                "polymers_json": polymers_json,
                "solvent_mapping": solvent_mapping,
                "min_selectivity": item.get("min_selectivity"),
            }
        )

    if not sequence_candidates:
        raise ValueError("source handoff has no usable top_k_sequences")

    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "polymers": payload.get("polymers", []),
        "sequence_candidates": sequence_candidates,
    }
    return (
        "sequence_batch.v1",
        handoff_payload,
        _build_biosteam_prompt(sequence_candidates),
    )


def _adapt_separation_to_visualization(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    polymers = payload.get("polymers", [])
    supported_polymers = payload.get("supported_polymers") or []
    unsupported_polymers = payload.get("unsupported_polymers") or []
    if not supported_polymers and not unsupported_polymers:
        supported_polymers, unsupported_polymers = _partition_supported_polymers(polymers)
    plot_polymers = supported_polymers or polymers
    temperature = None
    steps = payload.get("steps") or []
    if steps:
        temperature = steps[0].get("temperature_c")
    requested_plot_type, preferred_tool, requested_label = _infer_visualization_request(
        scope_user_query
    )
    if preferred_tool == "create_separation_tree_plot" and source.task_prompt:
        requested_plot_type, preferred_tool, requested_label = _infer_visualization_request(
            source.task_prompt
        )
    suggested_solvents = list(
        dict.fromkeys(
            [
                *[
                    str(solvent)
                    for solvent in (payload.get("solvent_mapping") or {}).values()
                    if solvent
                ],
                *[str(solvent) for solvent in (payload.get("top_solvents") or []) if solvent],
            ]
        )
    )

    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "polymers": polymers,
        "plot_polymers": plot_polymers,
        "supported_polymers": supported_polymers,
        "unsupported_polymers": unsupported_polymers,
        "best_sequence": payload.get("best_sequence", []),
        "solvent_mapping": payload.get("solvent_mapping", {}),
        "steps": steps,
        "requested_plot_type": requested_plot_type,
        "preferred_tool": preferred_tool,
        "suggested_solvents": suggested_solvents,
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }

    task_prompt = _build_visualization_task_prompt(
        polymers=polymers,
        plot_polymers=plot_polymers,
        temperature=temperature,
        requested_label=requested_label,
        preferred_tool=preferred_tool,
        suggested_solvents=suggested_solvents,
        request_context=scope_user_query,
        unsupported_polymers=unsupported_polymers,
    )
    return ("separation_plot.v1", handoff_payload, task_prompt)


def _adapt_separation_to_optimization(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    polymer_filters, global_candidates = _build_optimization_solvent_filters(payload)
    filters_json = json.dumps(polymer_filters, ensure_ascii=False)
    global_json = json.dumps(global_candidates, ensure_ascii=False)

    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "polymers": payload.get("polymers", []),
        "best_sequence": payload.get("best_sequence", []),
        "steps": payload.get("steps", []),
        "solvent_mapping": payload.get("solvent_mapping", {}),
        "top_solvents": payload.get("top_solvents", []),
        "polymer_solvent_filters": polymer_filters,
        "candidate_solvents": global_candidates,
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }

    lines = [
        "Use the upstream separation-route solvent shortlist to constrain the waste optimization solve.",
        f"Polymer-specific optimization solvent filters: {filters_json}.",
        f"Global candidate solvents: {global_json}.",
        "Call run_waste_management_optimization with the upstream shortlist when running the optimization.",
        "Pass `candidate_solvents` using the listed global candidate solvents.",
        "Pass `polymer_solvent_filters_json` using the listed polymer-specific filter JSON.",
        "The optimization tool applies these as soft filters: if a polymer-specific shortlist has no overlap with the optimization workbook, it falls back to the full candidate set for that polymer and reports a filter note.",
        "Do not broaden the candidate set beyond the upstream route shortlist unless the tool reports no overlap.",
    ]
    if scope_user_query:
        lines.append(f"Original user request: {scope_user_query}")
    return ("optimization_route_context.v1", handoff_payload, " ".join(lines))


def _adapt_statistics_to_visualization(
    source: HandoffRecord,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    analysis_type = payload.get("analysis_type", "analysis")
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "analysis_type": analysis_type,
        "summary": payload.get("summary"),
        "plot_paths": payload.get("plot_paths"),
        "table": payload.get("table"),
    }
    task_prompt = (
        "Create a visualization for this statistics/ML result using the provided "
        f"{analysis_type} summary."
    )
    return ("analysis_plot.v1", handoff_payload, task_prompt)


def _adapt_biosteam_to_visualization(
    source: HandoffRecord,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "target_plastic": payload.get("target_plastic"),
        "energy_case": payload.get("energy_case"),
        "results_json": json.dumps(payload, ensure_ascii=False),
        "results": payload.get("results"),
        "existing_artifacts": source.artifacts,
    }
    task_prompt = (
        "Visualize this BioSTEAM result using `payload.results_json` as the input to "
        "`visualize_biosteam_results`. Reuse any existing charts if they already answer the request."
    )
    return ("biosteam_plot.v1", handoff_payload, task_prompt)


def _adapt_separation_to_contaminant(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    mode = _infer_contaminant_mode(scope_user_query or source.task_prompt)
    solvents = list(
        dict.fromkeys(
            [
                *[
                    str(solvent)
                    for solvent in (payload.get("solvent_mapping") or {}).values()
                    if solvent
                ],
                *[str(solvent) for solvent in (payload.get("top_solvents") or []) if solvent],
            ]
        )
    )
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "polymers": payload.get("polymers", []),
        "supported_polymers": payload.get("supported_polymers", []),
        "unsupported_polymers": payload.get("unsupported_polymers", []),
        "best_sequence": payload.get("best_sequence", []),
        "steps": payload.get("steps", []),
        "solvent_mapping": payload.get("solvent_mapping", {}),
        "top_solvents": payload.get("top_solvents", []),
        "candidate_solvents": solvents,
        "suggested_mode": mode,
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }
    lines = [
        "Screen the separation-route solvent candidates for contaminant removal using the user request as the objective source.",
        f"Suggested mode: {mode}.",
        f"Candidate solvents: {', '.join(solvents) if solvents else 'none supplied from the separation result'}.",
        "Use the target polymer, non-target polymers, contaminants, and temperature bound from the user request.",
    ]
    if scope_user_query:
        lines.append(f"Original user request: {scope_user_query}")
    lines.append("Do not recompute the upstream separation route; screen or compare contaminant-removal modes for these solvents.")
    return ("contaminant_screen.v1", handoff_payload, " ".join(lines))


def _adapt_contaminant_to_separation(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    recommended_solvents = payload.get("recommended_solvents", [])
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "mode": payload.get("mode"),
        "target_polymer": payload.get("target_polymer"),
        "other_polymers": payload.get("other_polymers", []),
        "contaminants": payload.get("contaminants", []),
        "supported_contaminants": payload.get("supported_contaminants", []),
        "unsupported_contaminants": payload.get("unsupported_contaminants", []),
        "recommended_solvents": recommended_solvents,
        "candidate_solvents": payload.get("candidate_solvents", []),
        "decision_basis": payload.get("decision_basis", []),
        "caveats": payload.get("caveats", []),
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }
    lines = [
        "Refine the separation route using the contaminant-removal screening results.",
        f"Recommended solvents from contaminant screening: {', '.join(recommended_solvents) if recommended_solvents else 'none'}.",
        f"Requested contaminant-removal mode: {payload.get('mode')}.",
        "Respect all contaminant-screening caveats and disqualified solvents when proposing the revised sequence.",
    ]
    if scope_user_query:
        lines.append(f"Original user request: {scope_user_query}")
    return ("contaminant_guided_separation.v1", handoff_payload, " ".join(lines))


def _adapt_contaminant_to_biosteam(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    target_polymer = payload.get("target_polymer")
    if not isinstance(target_polymer, str) or not target_polymer.strip():
        raise ValueError("source handoff is missing target_polymer")

    recommended_solvents = [
        str(item).strip()
        for item in (payload.get("recommended_solvents") or [])
        if str(item).strip()
    ]
    candidate_solvents = [
        str(item.get("solvent", "")).strip()
        for item in (payload.get("candidate_solvents") or [])
        if isinstance(item, dict) and str(item.get("solvent", "")).strip()
    ]
    solvents = list(dict.fromkeys(recommended_solvents + candidate_solvents))
    if not solvents:
        raise ValueError("source handoff has no usable solvent candidates")

    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "mode": payload.get("mode"),
        "target_plastic": target_polymer,
        "other_polymers": payload.get("other_polymers", []),
        "contaminants": payload.get("contaminants", []),
        "supported_contaminants": payload.get("supported_contaminants", []),
        "unsupported_contaminants": payload.get("unsupported_contaminants", []),
        "recommended_solvents": recommended_solvents,
        "candidate_solvents": solvents,
        "best_solvent": solvents[0],
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }
    lines = [
        f"Run TEA/LCA for the contaminant-screened recovery option for {target_polymer}.",
        f"Recommended solvents from contaminant screening: {', '.join(recommended_solvents) if recommended_solvents else 'none explicitly recommended'}.",
        f"Candidate solvents to assess: {', '.join(solvents)}.",
        "Use the best screened solvent first. If multiple recommended solvents are available, compare them in one batch before selecting the best option.",
        "Use the contaminant-screening result as the solvent shortlist; do not repeat contaminant screening.",
    ]
    if scope_user_query:
        lines.append(f"Original user request: {scope_user_query}")
    return ("contaminant_biosteam.v1", handoff_payload, " ".join(lines))


def _adapt_scholar_to_rag(
    source: HandoffRecord,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "query": payload.get("query"),
        "papers": payload.get("papers", []),
        "n_results": payload.get("n_results"),
        "saved_to_rag": payload.get("saved_to_rag"),
    }
    task_prompt = (
        "Use these upstream literature findings as context before any new RAG retrieval. "
        "Prioritize the cited papers and only search further if the payload is insufficient."
    )
    return ("literature_context.v1", handoff_payload, task_prompt)


def _adapt_patent_to_rag(
    source: HandoffRecord,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "query": payload.get("query"),
        "patents": payload.get("patents", []),
        "n_results": payload.get("n_results"),
        "saved_to_rag": payload.get("saved_to_rag"),
    }
    task_prompt = (
        "Use these upstream patent findings as context before any new RAG retrieval. "
        "Prioritize the cited patents and only search further if the payload is insufficient."
    )
    return ("patent_context.v1", handoff_payload, task_prompt)


_ADAPTERS: dict[tuple[str, str], Any] = {
    ("biosteam-analyst", "visualization-specialist"): _adapt_biosteam_to_visualization,
    ("contaminant-removal-analyst", "biosteam-analyst"): _adapt_contaminant_to_biosteam,
    ("contaminant-removal-analyst", "separation-engineer"): _adapt_contaminant_to_separation,
    ("patent-researcher", "rag-analyst"): _adapt_patent_to_rag,
    ("scholar-researcher", "rag-analyst"): _adapt_scholar_to_rag,
    ("separation-engineer", "biosteam-analyst"): _adapt_separation_to_biosteam,
    ("separation-engineer", "contaminant-removal-analyst"): _adapt_separation_to_contaminant,
    ("separation-engineer", "optimization-engineer"): _adapt_separation_to_optimization,
    ("separation-engineer", "visualization-specialist"): _adapt_separation_to_visualization,
    ("statistics-ml", "visualization-specialist"): _adapt_statistics_to_visualization,
}


def build_typed_handoff(
    source: HandoffRecord,
    consumer: str,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str] | None:
    adapter = _ADAPTERS.get((source.producer, consumer))
    if adapter is None:
        return None
    if adapter is _adapt_separation_to_visualization:
        return adapter(source, scope_user_query=scope_user_query)
    if adapter is _adapt_separation_to_contaminant:
        return adapter(source, scope_user_query=scope_user_query)
    if adapter is _adapt_separation_to_optimization:
        return adapter(source, scope_user_query=scope_user_query)
    if adapter is _adapt_contaminant_to_separation:
        return adapter(source, scope_user_query=scope_user_query)
    if adapter is _adapt_contaminant_to_biosteam:
        return adapter(source, scope_user_query=scope_user_query)
    return adapter(source)
