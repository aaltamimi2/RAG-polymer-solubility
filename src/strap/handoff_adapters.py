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


_MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER = 50


def _normalize_solvent_key(solvent: Any) -> str:
    text = str(solvent or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _build_solvent_lookup(solvents: list[str]) -> dict[str, str]:
    from .solvent_registry import resolve_to_biosteam

    lookup: dict[str, str] = {}
    for solvent in solvents:
        actual = str(solvent).strip()
        if not actual:
            continue
        variants = {
            actual,
            resolve_to_biosteam(actual) or "",
            actual.replace("_", " "),
            actual.replace("-", " "),
        }
        for variant in variants:
            key = _normalize_solvent_key(variant)
            if key:
                lookup.setdefault(key, actual)
    return lookup


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


def _infer_optimization_constraint_mode(
    scope_user_query: str | None,
    payload: dict[str, Any] | None = None,
) -> str:
    """Infer the optimization constraint mode.

    Signals in order of precedence:
      1. Explicit strong-enforcement phrasing in the user query → "hard"
      2. Explicit relaxation phrasing → "soft"
      3. A DP-produced top_k_sequences list with two or more ranked
         alternatives → "ranked_soft" (the separation planner chose this
         structure specifically to offer the optimizer a bounded decision
         set; soft mode would let the optimizer ignore the whole thing)
      4. Preference/shortlist keywords paired with route/solvent language
         in the user query → "ranked_soft"
      5. Default → "soft"
    """
    text = (scope_user_query or "").lower()
    if any(phrase in text for phrase in ("exactly these", "use exactly", "must only use", "strictly use")):
        return "hard"
    relax_signals = ("ignore shortlist", "don't constrain", "do not constrain", "broad search", "unconstrained")
    if any(phrase in text for phrase in relax_signals):
        return "soft"

    top_k = (payload or {}).get("top_k_sequences") or []
    if isinstance(top_k, list) and sum(1 for item in top_k if isinstance(item, dict)) >= 2:
        return "ranked_soft"

    if any(phrase in text for phrase in ("prefer", "shortlist", "candidate", "top ", "best ")) and any(
        phrase in text for phrase in ("route", "solvent", "pair", "shortlist", "sequence")
    ):
        return "ranked_soft"
    return "soft"


def _infer_optimization_fallback_policy(constraint_mode: str) -> str:
    if constraint_mode in {"fixed", "hard"}:
        return "fail_closed"
    return "broaden_disclosed"


def _infer_route_pool_mode(scope_user_query: str | None) -> str:
    """Infer whether route tuples should stay exact or become slot-independent.

    Default is exact route preservation. We only opt into slot-independent
    pooling when the user explicitly asks to mix, combine, or treat wash slots
    independently.
    """
    text = (scope_user_query or "").lower()
    slot_independent_signals = (
        "slot independent",
        "slot-independent",
        "mix and match",
        "cross product",
        "cross-product",
        "combine independently",
        "independently combine",
        "independent wash",
        "independent slots",
        "broader solvent pool",
        "broader solvent-pool",
        "broad solvent pool",
        "top 50 unique",
        "unique solvent choices",
    )
    if any(signal in text for signal in slot_independent_signals):
        return "slot_independent"
    return "exact"


def _infer_operating_constraints(
    payload: dict[str, Any],
    scope_user_query: str | None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    max_temp: float | None = None
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        temp = step.get("temperature_c")
        try:
            temp_value = float(temp)
        except (TypeError, ValueError):
            continue
        max_temp = temp_value if max_temp is None else max(max_temp, temp_value)
    if max_temp is not None:
        constraints["temperature_max_c"] = max_temp
    query_text = (scope_user_query or "").lower()
    constraints["pressure"] = "atmospheric" if "atmospheric" in query_text else "unspecified"
    return constraints


def _build_optimization_solvent_filters(
    payload: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str], dict[tuple[str, str], int]]:
    from .waste_management.data_loader import get_optimizer_default_sets

    optimizer_sets = get_optimizer_default_sets()
    pe_lookup = _build_solvent_lookup(optimizer_sets["S_PE"])
    evoh_lookup = _build_solvent_lookup(list(dict.fromkeys([*optimizer_sets["S_EV1"], *optimizer_sets["S_EV2"]])))

    polymer_filters: dict[str, list[str]] = {"PE": [], "EVOH": []}
    global_candidates: list[str] = []
    candidate_rank_lookup: dict[tuple[str, str], int] = {}

    def add_global(solvent: str) -> None:
        if solvent and solvent not in global_candidates:
            global_candidates.append(solvent)

    def add_polymer(polymer: str, solvent: str, *, rank: int | None = None) -> None:
        existing = polymer_filters[polymer]
        if solvent in existing:
            if rank is not None:
                candidate_rank_lookup.setdefault((polymer, solvent), rank)
            return
        if len(existing) >= _MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER:
            return
        if solvent and solvent not in polymer_filters[polymer]:
            polymer_filters[polymer].append(solvent)
            add_global(solvent)
            if rank is not None:
                candidate_rank_lookup.setdefault((polymer, solvent), rank)

    from .solvent_registry import resolve_to_biosteam

    def _lookup_keys(raw: str) -> list[str]:
        """All normalization variants to try against the optimizer lookup.

        Incoming names like 'dimethylsulfoxide' (no space) don't match the
        optimizer-side 'Dimethyl sulfoxide' under pure whitespace/case folding;
        we have to route through the registry first so aliases, abbreviations,
        and spacing differences resolve to the same canonical key.
        """
        raw_key = _normalize_solvent_key(raw)
        keys: list[str] = [raw_key] if raw_key else []
        canonical = resolve_to_biosteam(raw) or ""
        if canonical:
            canonical_key = _normalize_solvent_key(canonical)
            if canonical_key and canonical_key not in keys:
                keys.append(canonical_key)
        return keys

    def _first_match(lookup: dict[str, str], raw: str) -> str | None:
        for key in _lookup_keys(raw):
            match = lookup.get(key)
            if match:
                return match
        return None

    def route_solvent(solvent: Any, polymer: Any, *, rank: int | None = None) -> None:
        """Route a polymer-LABELED (polymer, solvent) pair into the filters.

        Polymer-labeled sources — steps, solvent_mapping, top_k_sequences —
        carry explicit intent about which polymer each solvent targets. Only
        those should drive per-polymer filter population. Unlabeled top_solvents
        flow through add_to_globals below so downstream consumers still see the
        full candidate universe, but they don't cross-contaminate the
        per-polymer shortlist.
        """
        solvent_name = str(solvent or "").strip()
        if not solvent_name:
            return
        canonical_polymer = _canonical_optimization_polymer(polymer)
        if canonical_polymer is None:
            return
        lookup = pe_lookup if canonical_polymer == "PE" else evoh_lookup
        match = _first_match(lookup, solvent_name)
        if match:
            add_polymer(canonical_polymer, match, rank=rank)

    def add_to_globals(solvent: Any) -> None:
        """Add a solvent to global_candidates without touching per-polymer filters."""
        solvent_name = str(solvent or "").strip()
        if not solvent_name:
            return
        # Match against either polymer's lookup to get the canonical form.
        match = _first_match(pe_lookup, solvent_name) or _first_match(evoh_lookup, solvent_name)
        if match:
            add_global(match)

    polymer_solvent_candidates = payload.get("polymer_solvent_candidates") or {}
    if isinstance(polymer_solvent_candidates, dict):
        for polymer_raw, entries in polymer_solvent_candidates.items():
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries, start=1):
                if isinstance(entry, dict):
                    solvent = entry.get("solvent")
                    rank_value = entry.get("rank")
                    try:
                        rank = int(rank_value) if rank_value is not None else index
                    except (TypeError, ValueError):
                        rank = index
                else:
                    solvent = entry
                    rank = index
                route_solvent(solvent, polymer_raw, rank=rank)

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
        # top_solvents are polymer-agnostic nominations; they populate globals
        # only, never per-polymer filters. Routing them into a specific filter
        # causes cross-polymer leakage (e.g., DMSO landing in the PE list just
        # because the shared catalog has DMSO on both sides).
        add_to_globals(solvent)

    polymer_filters = {polymer: solvents for polymer, solvents in polymer_filters.items() if solvents}
    return polymer_filters, global_candidates, candidate_rank_lookup


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


def _build_optimization_route_candidates(
    payload: dict[str, Any],
    *,
    warnings_sink: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Preserve (polymer, solvent) coupling per DP-proposed sequence.

    The separation engine's top_k_sequences each carries a full polymer→solvent
    mapping; flattening these into per-polymer solvent sets loses the fact that
    e.g. "LDPE+cyclohexane with EVOH+methanol" is a distinct route from
    "LDPE+cyclohexane with EVOH+DMSO". The route_candidates field preserves
    this coupling so the optimizer can solve each route as an enforced
    decision set rather than a loose filter.

    Only routes whose (polymer, solvent) pairs ALL resolve to the shared
    optimizer catalog are emitted; routes with any unresolvable pair are
    skipped. The caller still sees the flattened per-polymer filter as a
    backup for soft-mode or legacy consumers.
    """
    from .solvent_registry import resolve_to_biosteam
    from .waste_management.data_loader import get_optimizer_default_sets

    optimizer_sets = get_optimizer_default_sets()
    pe_lookup = _build_solvent_lookup(optimizer_sets["S_PE"])
    evoh_lookup = _build_solvent_lookup(list(dict.fromkeys([*optimizer_sets["S_EV1"], *optimizer_sets["S_EV2"]])))

    def _resolve_for_polymer(solvent: Any, polymer: str) -> str | None:
        raw = str(solvent or "").strip()
        if not raw:
            return None
        canonical = resolve_to_biosteam(raw) or raw
        variants = [_normalize_solvent_key(raw), _normalize_solvent_key(canonical)]
        lookup = pe_lookup if polymer == "PE" else evoh_lookup
        for key in variants:
            match = lookup.get(key)
            if match:
                return match
        return None

    seen_routes: set[tuple[tuple[str, str], ...]] = set()
    routes: list[dict[str, Any]] = []

    def _try_route(
        source: str,
        rank: Any,
        sequence: list[str] | None,
        mapping: dict[str, Any] | None,
    ) -> None:
        if not mapping:
            return
        resolved_pairs: list[tuple[str, str]] = []
        for polymer_raw, solvent in mapping.items():
            polymer = _canonical_optimization_polymer(polymer_raw)
            if polymer is None:
                # Non-PE/EVOH polymers are ignored for route enforcement — they
                # can't be forced through the current workbook anyway.
                continue
            resolved = _resolve_for_polymer(solvent, polymer)
            if resolved is None:
                if warnings_sink is not None:
                    warnings_sink.append(
                        f"Dropped route from {source} (rank={rank}): solvent "
                        f"'{solvent}' for polymer '{polymer_raw}' is not in the optimizer catalog."
                    )
                return
            resolved_pairs.append((polymer, resolved))
        if not resolved_pairs:
            return
        signature = tuple(sorted(resolved_pairs))
        if signature in seen_routes:
            return
        seen_routes.add(signature)
        try:
            rank_int: int | None = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_int = None
        polymer_solvent_map = {polymer: solvent for polymer, solvent in resolved_pairs}
        routes.append(
            {
                "route_id": f"route_{len(routes) + 1}",
                "rank": rank_int,
                "source": source,
                "sequence": [str(p) for p in (sequence or []) if isinstance(p, str)],
                "polymer_solvent_map": polymer_solvent_map,
            }
        )

    # Primary source: top_k_sequences preserves multiple alternative routes.
    for sequence_record in payload.get("top_k_sequences") or []:
        if not isinstance(sequence_record, dict):
            continue
        _try_route(
            source="top_k_sequences",
            rank=sequence_record.get("rank"),
            sequence=sequence_record.get("sequence"),
            mapping=sequence_record.get("solvent_mapping"),
        )

    # Fallback: the single-sequence solvent_mapping if no top_k was given.
    _try_route(
        source="solvent_mapping",
        rank=1,
        sequence=payload.get("best_sequence") or payload.get("polymers"),
        mapping=payload.get("solvent_mapping"),
    )

    return routes


def _adapt_separation_to_optimization(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = source.payload
    polymer_filters, global_candidates, candidate_rank_lookup = _build_optimization_solvent_filters(payload)
    route_dropped_warnings: list[str] = []
    route_candidates = _build_optimization_route_candidates(payload, warnings_sink=route_dropped_warnings)
    filters_json = json.dumps(polymer_filters, ensure_ascii=False)
    global_json = json.dumps(global_candidates, ensure_ascii=False)
    constraint_mode = _infer_optimization_constraint_mode(scope_user_query, payload)
    fallback_policy = _infer_optimization_fallback_policy(constraint_mode)
    route_pool_mode = _infer_route_pool_mode(scope_user_query)
    operating_constraints = _infer_operating_constraints(payload, scope_user_query)

    step_rank_lookup: dict[tuple[str, str], int] = {}
    for idx, step in enumerate(payload.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        polymer = _canonical_optimization_polymer(step.get("polymer"))
        solvent = str(step.get("solvent") or "").strip()
        if polymer and solvent:
            step_rank_lookup[(polymer, solvent)] = idx

    stages: list[dict[str, Any]] = []
    flattened_pairs: list[dict[str, Any]] = []
    stage_order = (
        ("wash_1", "selective_dissolution", "PE"),
        ("wash_2", "selective_dissolution", "EVOH"),
    )
    for stage_id, stage_kind, target_polymer in stage_order:
        solvents = polymer_filters.get(target_polymer, [])
        stage_pairs: list[dict[str, Any]] = []
        for solvent in solvents:
            pair = {
                "polymer": target_polymer,
                "solvent": solvent,
                "source_agent": source.producer,
                "source_rank": candidate_rank_lookup.get(
                    (target_polymer, solvent),
                    step_rank_lookup.get((target_polymer, solvent)),
                ),
                "source_reason": (
                    "upstream ranked solvent candidate"
                    if (target_polymer, solvent) in candidate_rank_lookup
                    else "upstream separation route candidate"
                ),
                "constraint_mode": constraint_mode,
            }
            stage_pairs.append(pair)
            flattened_pairs.append({**pair, "stage_id": stage_id})
        stages.append(
            {
                "stage_id": stage_id,
                "stage_kind": stage_kind,
                "target_polymer": target_polymer,
                "candidate_pairs": stage_pairs,
            }
        )

    handoff_payload = {
        "schema_version": "1.0",
        "workflow_scope": "multi_stage",
        "route_id": source.handoff_id,
        "constraint_mode": constraint_mode,
        "fallback_policy": fallback_policy,
        "route_pool_mode": route_pool_mode,
        "operating_constraints": operating_constraints,
        "stages": stages,
        "candidate_pairs": flattened_pairs,
        "route_candidates": route_candidates,
        "route_candidate_warnings": route_dropped_warnings,
        "source_handoff_id": source.handoff_id,
        "polymers": payload.get("polymers", []),
        "best_sequence": payload.get("best_sequence", []),
        "steps": payload.get("steps", []),
        "solvent_mapping": payload.get("solvent_mapping", {}),
        "top_solvents": payload.get("top_solvents", []),
        "polymer_solvent_filters": polymer_filters,
        "candidate_solvents": global_candidates,
        "max_unique_solvents_per_polymer": _MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER,
        "candidate_counts_by_polymer": {
            polymer: len(solvents) for polymer, solvents in polymer_filters.items()
        },
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }
    stage_candidates_json = json.dumps(handoff_payload, ensure_ascii=False)

    routes_json = json.dumps(route_candidates, ensure_ascii=False)
    lines = [
        "Use the adapter-produced optimization stage-candidate handoff as the authoritative decision set for the waste optimization solve.",
        f"Constraint mode: {constraint_mode}.",
        f"Fallback policy: {fallback_policy}.",
        f"Route pool mode: {route_pool_mode}.",
        f"Operating constraints: {json.dumps(operating_constraints, ensure_ascii=False)}.",
        f"Exact `stage_candidates_json`: {stage_candidates_json}.",
        f"Polymer-specific optimization solvent filters: {filters_json}.",
        f"Global candidate solvents: {global_json}.",
        (
            "The solvent filters preserve up to "
            f"{_MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER} unique polymer-aware STRAP/BioSTEAM "
            "solvents per target polymer from the separation-engineer output when available."
        ),
        f"Route candidates (preserve polymer-solvent coupling per DP route): {routes_json}.",
        (
            "If route_candidates is non-empty, the optimizer will enforce each route's "
            "exact (polymer, solvent) washes and run a Pareto sweep per route. Do not "
            "rely on the flattened per-polymer filter as the authoritative decision set."
        ),
        (
            "The optimizer now accepts the broader polymer-aware STRAP/BioSTEAM solvent "
            "space and lets simulation plus post-sim validation decide what survives; "
            "do not pre-prune candidates in prose."
        ),
        "Call run_waste_management_optimization or run_waste_management_pareto with `stage_candidates_json` using this handoff payload.",
        "Do not rebuild or broaden the candidate set yourself. Respect the tool's fallback behavior and report any disclosed broadening explicitly.",
    ]
    if scope_user_query:
        lines.append(f"Original user request: {scope_user_query}")
    return ("optimization.stage_candidates.v1", handoff_payload, " ".join(lines))


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


def _adapt_optimization_to_visualization(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    """Forward a stored optimization payload into the visualization specialist.

    Handles both pareto_front and point_optimum analyses. The Pareto path maps to
    plot_optimization_pareto_front; the point-optimum path currently has no
    dedicated plot, so it falls back to a dashboard request.
    """
    payload = source.payload
    analysis_type = str(payload.get("analysis_type") or "").strip()

    if analysis_type == "pareto_front":
        # Deep-copy-safe: the handoff store already owns the dict; we only pass
        # the reference forward. The plotting tool accepts either a JSON string
        # or the dict directly.
        pareto_payload = dict(payload)
        handoff_payload = {
            "source_handoff_id": source.handoff_id,
            "analysis_type": "pareto_front",
            "x_metric": payload.get("x_metric"),
            "y_metric": payload.get("y_metric"),
            "n_points_feasible": payload.get("n_points_feasible"),
            "pareto_result_json": pareto_payload,
            "requested_plot_tool": "plot_optimization_pareto_front",
            "source_user_query": scope_user_query,
            "source_task_prompt": source.task_prompt,
        }
        lines = [
            "Plot the upstream waste-optimization Pareto frontier.",
            "Required tool: plot_optimization_pareto_front.",
            f"Call `plot_optimization_pareto_front(source_handoff_id=\"{source.handoff_id}\")`.",
            "Do not rebuild or paraphrase the Pareto payload from earlier separation prose.",
            f"Axes: x={payload.get('x_metric') or 'total_cost'}, y={payload.get('y_metric') or 'emissions'}.",
        ]
        if scope_user_query:
            lines.append(f"Original user request: {scope_user_query}")
        return ("optimization_plot_context.v1", handoff_payload, " ".join(lines))

    # Point optimum or unknown analysis — no native plot tool, surface the
    # structured result as a dashboard request.
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "analysis_type": analysis_type or "point_optimum",
        "profit": payload.get("profit"),
        "emissions": payload.get("emissions"),
        "total_cost": payload.get("total_cost"),
        "circularity_score": payload.get("circularity_score"),
        "optimal_washes": payload.get("optimal_washes") or payload.get("wash1_selection") or [],
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }
    lines = [
        "The optimization completed a point solve. There is no dedicated plotting tool "
        "for single-point optimization results today, so report the result fields rather "
        "than inventing a plot path.",
    ]
    if scope_user_query:
        lines.append(f"Original user request: {scope_user_query}")
    return ("optimization_plot_context.v1", handoff_payload, " ".join(lines))


_ADAPTERS: dict[tuple[str, str], Any] = {
    ("biosteam-analyst", "visualization-specialist"): _adapt_biosteam_to_visualization,
    ("contaminant-removal-analyst", "biosteam-analyst"): _adapt_contaminant_to_biosteam,
    ("contaminant-removal-analyst", "separation-engineer"): _adapt_contaminant_to_separation,
    ("optimization-engineer", "visualization-specialist"): _adapt_optimization_to_visualization,
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
    if adapter is _adapt_optimization_to_visualization:
        return adapter(source, scope_user_query=scope_user_query)
    return adapter(source)
