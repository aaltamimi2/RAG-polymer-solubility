"""Typed downstream handoff adapters.

These adapters translate validated upstream result payloads into
consumer-specific contracts and task prompts. Generic fallback handoffs stay
in ``handoffs.py``.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

from .query_context import extract_query_context

if TYPE_CHECKING:
    from .handoff_models import HandoffRecord


_MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER = 50


def _coerce_temperature_c(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer():
        return float(int(numeric))
    return round(numeric, 4)


def _format_optimizer_option(
    solvent: str,
    *,
    dissolution_temp_c: float | None = None,
    precipitation_temp_c: float | None = None,
) -> str:
    solvent_text = str(solvent or "").strip()
    if not solvent_text:
        return ""
    if dissolution_temp_c is None and precipitation_temp_c is None:
        return solvent_text
    if dissolution_temp_c is not None and precipitation_temp_c is not None:
        return f"{solvent_text} @ {dissolution_temp_c:g}C / ppt {precipitation_temp_c:g}C"
    if dissolution_temp_c is not None:
        return f"{solvent_text} @ {dissolution_temp_c:g}C"
    return f"{solvent_text} @ ppt {precipitation_temp_c:g}C"


def _normalize_solvent_key(solvent: Any) -> str:
    text = str(solvent or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _is_placeholder_solvent(solvent: Any) -> bool:
    text = str(solvent or "").strip().lower()
    if not text:
        return True
    placeholder_signals = (
        "n/a",
        "solid residue",
        "residue",
        "no solvent",
        "not applicable",
    )
    return any(signal in text for signal in placeholder_signals)


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
    if text in {"PE", "HDPE", "LDPE", "POLYETHYLENE"}:
        return "PE"
    if text == "EVOH":
        return "EVOH"
    if text in {"PET", "POLYETHYLENE TEREPHTHALATE"}:
        return "PET"
    if text in {"PP", "POLYPROPYLENE"}:
        return "PP"
    if text in {"PS", "POLYSTYRENE"}:
        return "PS"
    if text in {"PVC", "POLYVINYL CHLORIDE", "POLY(VINYL CHLORIDE)"}:
        return "PVC"
    if text in {"PC", "POLYCARBONATE"}:
        return "PC"
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
    if constraint_mode in {"fixed", "hard", "ranked_soft"}:
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
        "top solvent choices",
        "solvent choices",
        "solvent candidates",
        "shortlisted solvent candidates",
        "shortlisted wash candidates",
        "candidate pool",
        "unique solvent choices",
    )
    if any(signal in text for signal in slot_independent_signals):
        return "slot_independent"
    return "exact"


def _infer_requested_top_n_solvent_candidates(scope_user_query: str | None) -> int | None:
    text = (scope_user_query or "").lower()
    patterns = (
        r"top\s+(\d+)\s+(?:unique\s+)?solvent\s+candidates?\s+per\s+polymer",
        r"top\s+(\d+)\s+(?:unique\s+)?solvent\s+choices?\s+per\s+polymer",
        r"top\s+(\d+)\s+(?:unique\s+)?(?:polymer-)?solvent\s+pairs?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return min(value, _MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER)
    return None


def _augment_underfilled_polymer_solvent_candidates(
    payload: dict[str, Any],
    *,
    scope_user_query: str | None,
) -> dict[str, Any]:
    """Backfill under-filled top-N candidate lists from the deterministic planner.

    The LLM synthesis can under-copy the tool payload (for example only three
    candidates or omitting the final-residue polymer). The typed optimizer
    handoff should use the separation tool data, not the shortened prose.
    """

    requested_n = _infer_requested_top_n_solvent_candidates(scope_user_query)
    if requested_n is None:
        return payload

    requested_polymers = [
        str(polymer).strip()
        for polymer in payload.get("polymers", []) or []
        if _canonical_optimization_polymer(polymer)
    ]
    if not requested_polymers:
        return payload

    existing = payload.get("polymer_solvent_candidates")
    if not isinstance(existing, dict):
        existing = {}
    counts_by_canonical: dict[str, int] = {}
    for polymer_raw, entries in existing.items():
        polymer = _canonical_optimization_polymer(polymer_raw)
        if polymer is None:
            continue
        counts_by_canonical[polymer] = max(
            counts_by_canonical.get(polymer, 0),
            len(entries) if isinstance(entries, list) else 0,
        )
    requested_canonical = {
        _canonical_optimization_polymer(polymer)
        for polymer in requested_polymers
    }
    requested_canonical.discard(None)
    if requested_canonical and all(counts_by_canonical.get(polymer, 0) >= requested_n for polymer in requested_canonical):
        return payload

    try:
        from strap.tools.sequence_planning_tools import plan_sequential_separation

        raw = plan_sequential_separation(
            polymers=",".join(requested_polymers),
            top_k_solvents=requested_n,
            create_decision_tree=False,
        )
        tool_payload = json.loads(raw)
        tool_candidates = (tool_payload.get("data") or {}).get("polymer_solvent_candidates") or {}
    except Exception:
        return payload
    if not isinstance(tool_candidates, dict):
        return payload

    merged_payload = dict(payload)
    merged_candidates = {
        str(polymer): list(entries) if isinstance(entries, list) else []
        for polymer, entries in existing.items()
    }
    by_canonical_key: dict[str, str] = {}
    for polymer in requested_polymers:
        canonical = _canonical_optimization_polymer(polymer)
        if canonical:
            by_canonical_key.setdefault(canonical, polymer)
    for polymer_raw, entries in tool_candidates.items():
        canonical = _canonical_optimization_polymer(polymer_raw)
        if canonical is None or canonical not in requested_canonical:
            continue
        target_key = by_canonical_key.get(canonical) or str(polymer_raw)
        target_entries = merged_candidates.setdefault(target_key, [])
        seen = {
            str(entry.get("solvent") if isinstance(entry, dict) else entry).strip().lower()
            for entry in target_entries
            if str(entry.get("solvent") if isinstance(entry, dict) else entry).strip()
        }
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            solvent = str(entry.get("solvent") or "").strip()
            if not solvent or solvent.lower() in seen:
                continue
            next_entry = dict(entry)
            next_entry.setdefault("source_reason", "deterministic separation planner top-N backfill")
            target_entries.append(next_entry)
            seen.add(solvent.lower())
            if len(target_entries) >= requested_n:
                break
    merged_payload["polymer_solvent_candidates"] = merged_candidates
    warnings = list(merged_payload.get("candidate_backfill_warnings") or [])
    warnings.append(
        f"Filled under-reported polymer_solvent_candidates to requested top {requested_n} per polymer using plan_sequential_separation output."
    )
    merged_payload["candidate_backfill_warnings"] = warnings
    return merged_payload


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
    optimizer_polymers = list(optimizer_sets.get("P", []))
    stage_polymer_map = optimizer_sets.get("S_BY_POLYMER", {})
    polymer_lookups = {
        polymer: _build_solvent_lookup(list(stage_polymer_map.get(polymer, [])))
        for polymer in optimizer_polymers
    }

    polymer_filters: dict[str, list[str]] = {polymer: [] for polymer in optimizer_polymers}
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
        if canonical_polymer is None or canonical_polymer not in polymer_filters:
            return
        lookup = polymer_lookups.get(canonical_polymer, {})
        match = _first_match(lookup, solvent_name)
        if match is None:
            match = resolve_to_biosteam(solvent_name) or solvent_name
        if match:
            add_polymer(canonical_polymer, match, rank=rank)

    def add_to_globals(solvent: Any) -> None:
        """Add a solvent to global_candidates without touching per-polymer filters."""
        solvent_name = str(solvent or "").strip()
        if not solvent_name:
            return
        match = None
        for lookup in polymer_lookups.values():
            match = _first_match(lookup, solvent_name)
            if match:
                break
        if match is None:
            match = resolve_to_biosteam(solvent_name) or solvent_name
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


def _build_temperature_aware_candidate_pairs(
    payload: dict[str, Any],
    *,
    canonical_solvent_lookup_by_polymer: dict[str, dict[str, str]],
    candidate_rank_lookup: dict[tuple[str, str], int],
) -> dict[str, list[dict[str, Any]]]:
    from .solvent_registry import resolve_to_biosteam

    variants_by_polymer: dict[str, list[dict[str, Any]]] = {}
    seen_keys: set[tuple[str, str, float | None, float | None]] = set()
    seen_explicit_by_base: set[tuple[str, str]] = set()

    def _canonical_solvent(polymer: str, solvent: Any) -> str | None:
        raw = str(solvent or "").strip()
        if not raw:
            return None
        if _is_placeholder_solvent(raw):
            return None
        lookup = canonical_solvent_lookup_by_polymer.get(polymer, {})
        variants = [raw, resolve_to_biosteam(raw) or ""]
        for variant in variants:
            key = _normalize_solvent_key(variant)
            match = lookup.get(key)
            if match:
                return match
        return resolve_to_biosteam(raw) or raw

    def _add_candidate(
        *,
        polymer_raw: Any,
        solvent_raw: Any,
        rank: int | None,
        source_reason: str,
        dissolution_temp_c: Any = None,
        precipitation_temp_c: Any = None,
        temperature_source: str | None = None,
    ) -> None:
        polymer = _canonical_optimization_polymer(polymer_raw)
        if polymer is None:
            return
        solvent = _canonical_solvent(polymer, solvent_raw)
        if not solvent:
            return
        dissolution_temp = _coerce_temperature_c(dissolution_temp_c)
        precipitation_temp = _coerce_temperature_c(precipitation_temp_c)
        if dissolution_temp is None and precipitation_temp is None and (polymer, solvent) in seen_explicit_by_base:
            return
        key = (polymer, solvent, dissolution_temp, precipitation_temp)
        if key in seen_keys:
            return
        seen_keys.add(key)
        if dissolution_temp is not None or precipitation_temp is not None:
            seen_explicit_by_base.add((polymer, solvent))
        option_label = _format_optimizer_option(
            solvent,
            dissolution_temp_c=dissolution_temp,
            precipitation_temp_c=precipitation_temp,
        )
        variants_by_polymer.setdefault(polymer, []).append(
            {
                "polymer": polymer,
                "solvent": solvent,
                "optimizer_option": option_label,
                "dissolution_temp_c": dissolution_temp,
                "precipitation_temp_c": precipitation_temp,
                "temperature_source": temperature_source or ("upstream_explicit" if dissolution_temp is not None else "biosteam_default"),
                "source_rank": rank,
                "source_reason": source_reason,
            }
        )

    for polymer_raw, entries in (payload.get("polymer_solvent_candidates") or {}).items():
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
                _add_candidate(
                    polymer_raw=polymer_raw,
                    solvent_raw=solvent,
                    rank=rank,
                    source_reason="upstream ranked solvent candidate",
                    dissolution_temp_c=entry.get("temperature_c"),
                    precipitation_temp_c=entry.get("precipitation_temp_c"),
                    temperature_source="polymer_solvent_candidates" if entry.get("temperature_c") is not None else None,
                )
            else:
                _add_candidate(
                    polymer_raw=polymer_raw,
                    solvent_raw=entry,
                    rank=index,
                    source_reason="upstream ranked solvent candidate",
                )

    for index, step in enumerate(payload.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        _add_candidate(
            polymer_raw=step.get("polymer"),
            solvent_raw=step.get("solvent"),
            rank=index,
            source_reason="upstream separation step",
            dissolution_temp_c=step.get("temperature_c"),
            precipitation_temp_c=step.get("precipitation_temp_c"),
            temperature_source="steps" if step.get("temperature_c") is not None else None,
        )

    for sequence_record in payload.get("top_k_sequences") or []:
        if not isinstance(sequence_record, dict):
            continue
        rank_value = sequence_record.get("rank")
        try:
            route_rank = int(rank_value) if rank_value is not None else None
        except (TypeError, ValueError):
            route_rank = None
        step_conditions = sequence_record.get("steps") or []
        if isinstance(step_conditions, list) and step_conditions:
            for step in step_conditions:
                if not isinstance(step, dict):
                    continue
                _add_candidate(
                    polymer_raw=step.get("polymer"),
                    solvent_raw=step.get("solvent"),
                    rank=route_rank,
                    source_reason="upstream route candidate",
                    dissolution_temp_c=step.get("temperature_c"),
                    precipitation_temp_c=step.get("precipitation_temp_c"),
                    temperature_source="top_k_sequences.steps" if step.get("temperature_c") is not None else None,
                )
            continue
        for polymer_raw, solvent_raw in (sequence_record.get("solvent_mapping") or {}).items():
            _add_candidate(
                polymer_raw=polymer_raw,
                solvent_raw=solvent_raw,
                rank=route_rank,
                source_reason="upstream route candidate",
            )

    for polymer_raw, solvent_raw in (payload.get("solvent_mapping") or {}).items():
        polymer = _canonical_optimization_polymer(polymer_raw)
        solvent = _canonical_solvent(polymer, solvent_raw) if polymer else None
        rank = candidate_rank_lookup.get((polymer, solvent)) if polymer and solvent else None
        _add_candidate(
            polymer_raw=polymer_raw,
            solvent_raw=solvent_raw,
            rank=rank,
            source_reason="upstream separation route candidate",
        )

    return variants_by_polymer


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
    """Preserve polymer, solvent, and step-temperature coupling per route."""
    from .solvent_registry import resolve_to_biosteam
    from .waste_management.data_loader import get_optimizer_default_sets

    optimizer_sets = get_optimizer_default_sets()
    polymer_lookups = {
        polymer: _build_solvent_lookup(list((optimizer_sets.get("S_BY_POLYMER", {}) or {}).get(polymer, [])))
        for polymer in optimizer_sets.get("P", [])
    }

    def _resolve_for_polymer(solvent: Any, polymer: str) -> str | None:
        raw = str(solvent or "").strip()
        if not raw:
            return None
        canonical = resolve_to_biosteam(raw) or raw
        variants = [_normalize_solvent_key(raw), _normalize_solvent_key(canonical)]
        lookup = polymer_lookups.get(polymer, {})
        for key in variants:
            match = lookup.get(key)
            if match:
                return match
        return canonical or raw

    seen_routes: set[tuple[tuple[str, str], ...]] = set()
    routes: list[dict[str, Any]] = []

    def _normalize_step_conditions(
        raw_steps: list[dict[str, Any]] | None,
        *,
        source: str,
        rank: Any,
    ) -> list[dict[str, Any]] | None:
        if not isinstance(raw_steps, list):
            return []
        normalized: list[dict[str, Any]] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            polymer_raw = step.get("polymer")
            polymer = _canonical_optimization_polymer(polymer_raw)
            if polymer is None:
                continue
            resolved = _resolve_for_polymer(step.get("solvent"), polymer)
            if resolved is None:
                if warnings_sink is not None:
                    warnings_sink.append(
                        f"Dropped route from {source} (rank={rank}): solvent '{step.get('solvent')}' for polymer '{polymer_raw}' is not in the optimizer catalog."
                    )
                return None
            dissolution_temp = _coerce_temperature_c(step.get("temperature_c"))
            precipitation_temp = _coerce_temperature_c(step.get("precipitation_temp_c"))
            normalized.append(
                {
                    "polymer": polymer,
                    "solvent": resolved,
                    "optimizer_option": _format_optimizer_option(
                        resolved,
                        dissolution_temp_c=dissolution_temp,
                        precipitation_temp_c=precipitation_temp,
                    ),
                    "dissolution_temp_c": dissolution_temp,
                    "precipitation_temp_c": precipitation_temp,
                    "temperature_source": "route_step" if dissolution_temp is not None else "biosteam_default",
                }
            )
        return normalized

    def _try_route(
        source: str,
        rank: Any,
        sequence: list[str] | None,
        mapping: dict[str, Any] | None,
        *,
        raw_steps: list[dict[str, Any]] | None = None,
    ) -> None:
        if not mapping and not raw_steps:
            return
        step_conditions = _normalize_step_conditions(raw_steps, source=source, rank=rank)
        if step_conditions is None:
            return
        condition_by_polymer = {
            str(item.get("polymer") or ""): dict(item)
            for item in (step_conditions or [])
            if str(item.get("polymer") or "")
        }
        for polymer_raw, solvent in (mapping or {}).items():
            polymer = _canonical_optimization_polymer(polymer_raw)
            if polymer is None or polymer in condition_by_polymer:
                continue
            resolved = _resolve_for_polymer(solvent, polymer)
            if resolved is None:
                if warnings_sink is not None:
                    warnings_sink.append(
                        f"Dropped route from {source} (rank={rank}): solvent '{solvent}' for polymer '{polymer_raw}' is not in the optimizer catalog."
                    )
                return
            condition_by_polymer[polymer] = {
                "polymer": polymer,
                "solvent": resolved,
                "optimizer_option": resolved,
                "dissolution_temp_c": None,
                "precipitation_temp_c": None,
                "temperature_source": "biosteam_default",
            }
        if not condition_by_polymer:
            return
        signature = tuple(sorted((polymer, str(condition.get("optimizer_option") or condition.get("solvent") or "")) for polymer, condition in condition_by_polymer.items()))
        if signature in seen_routes:
            return
        seen_routes.add(signature)
        try:
            rank_int: int | None = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_int = None
        ordered_sequence = [str(p) for p in (sequence or []) if isinstance(p, str)]
        polymer_option_map = {
            polymer: str(condition.get("optimizer_option") or condition.get("solvent") or "")
            for polymer, condition in condition_by_polymer.items()
            if str(condition.get("optimizer_option") or condition.get("solvent") or "")
        }
        actual_solvent_map = {
            polymer: str(condition.get("solvent") or "")
            for polymer, condition in condition_by_polymer.items()
            if str(condition.get("solvent") or "")
        }
        routes.append(
            {
                "route_id": f"route_{len(routes) + 1}",
                "rank": rank_int,
                "source": source,
                "sequence": ordered_sequence,
                "polymer_solvent_map": polymer_option_map,
                "actual_solvent_map": actual_solvent_map,
                "step_conditions": list(condition_by_polymer.values()),
            }
        )

    for sequence_record in payload.get("top_k_sequences") or []:
        if not isinstance(sequence_record, dict):
            continue
        _try_route(
            source="top_k_sequences",
            rank=sequence_record.get("rank"),
            sequence=sequence_record.get("sequence"),
            mapping=sequence_record.get("solvent_mapping"),
            raw_steps=sequence_record.get("steps"),
        )

    _try_route(
        source="solvent_mapping",
        rank=1,
        sequence=payload.get("best_sequence") or payload.get("polymers"),
        mapping=payload.get("solvent_mapping"),
        raw_steps=payload.get("steps"),
    )

    return routes


def _adapt_separation_to_optimization(
    source: HandoffRecord,
    *,
    scope_user_query: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    payload = _augment_underfilled_polymer_solvent_candidates(
        source.payload,
        scope_user_query=scope_user_query,
    )
    polymer_filters, global_candidates, candidate_rank_lookup = _build_optimization_solvent_filters(payload)
    canonical_solvent_lookup_by_polymer = {
        polymer: _build_solvent_lookup(list(solvents))
        for polymer, solvents in polymer_filters.items()
    }
    variant_candidates_by_polymer = _build_temperature_aware_candidate_pairs(
        payload,
        canonical_solvent_lookup_by_polymer=canonical_solvent_lookup_by_polymer,
        candidate_rank_lookup=candidate_rank_lookup,
    )
    route_dropped_warnings: list[str] = []
    route_candidates = _build_optimization_route_candidates(payload, warnings_sink=route_dropped_warnings)
    query_context = extract_query_context(scope_user_query or "")
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
    ordered_polymers = ["PE", "EVOH", "PET", "PP", "PS", "PVC", "PC"]
    for polymer in list(polymer_filters) + list(variant_candidates_by_polymer):
        if polymer not in ordered_polymers:
            ordered_polymers.append(polymer)
    option_filters: dict[str, list[str]] = {}
    candidate_counts_by_polymer: dict[str, int] = {}
    for target_polymer in ordered_polymers:
        stage_pairs: list[dict[str, Any]] = []
        variants = list(variant_candidates_by_polymer.get(target_polymer, []))[:_MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER]
        if not variants:
            for solvent in polymer_filters.get(target_polymer, []):
                variants.append(
                    {
                        "polymer": target_polymer,
                        "solvent": solvent,
                        "optimizer_option": solvent,
                        "dissolution_temp_c": None,
                        "precipitation_temp_c": None,
                        "temperature_source": "biosteam_default",
                        "source_rank": candidate_rank_lookup.get((target_polymer, solvent), step_rank_lookup.get((target_polymer, solvent))),
                        "source_reason": "upstream separation route candidate",
                    }
                )
        if not variants:
            continue
        stage_id = f"candidate_pool_{target_polymer.lower()}"
        option_filters[target_polymer] = []
        for variant in variants:
            option_label = str(variant.get("optimizer_option") or variant.get("solvent") or "").strip()
            if not option_label:
                continue
            pair = {
                "polymer": target_polymer,
                "solvent": variant.get("solvent"),
                "optimizer_option": option_label,
                "dissolution_temp_c": variant.get("dissolution_temp_c"),
                "precipitation_temp_c": variant.get("precipitation_temp_c"),
                "temperature_source": variant.get("temperature_source") or "biosteam_default",
                "source_agent": source.producer,
                "source_rank": variant.get("source_rank"),
                "source_reason": variant.get("source_reason") or "upstream separation route candidate",
                "constraint_mode": constraint_mode,
            }
            stage_pairs.append(pair)
            flattened_pairs.append({**pair, "stage_id": stage_id})
            option_filters[target_polymer].append(option_label)
        if not stage_pairs:
            continue
        candidate_counts_by_polymer[target_polymer] = len(stage_pairs)
        stages.append(
            {
                "stage_id": stage_id,
                "stage_kind": "selective_dissolution_candidate_pool",
                "target_polymer": target_polymer,
                "candidate_pairs": stage_pairs,
            }
        )

    handoff_payload = {
        "schema_version": "1.1",
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
        "candidate_backfill_warnings": payload.get("candidate_backfill_warnings", []),
        "source_handoff_id": source.handoff_id,
        "polymers": payload.get("polymers", []),
        "best_sequence": payload.get("best_sequence", []),
        "steps": payload.get("steps", []),
        "solvent_mapping": payload.get("solvent_mapping", {}),
        "top_solvents": payload.get("top_solvents", []),
        "polymer_solvent_filters": polymer_filters,
        "polymer_option_filters": option_filters,
        "candidate_solvents": global_candidates,
        "max_unique_solvents_per_polymer": _MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER,
        "candidate_counts_by_polymer": candidate_counts_by_polymer,
        "feed_composition": query_context.feed_composition,
        "feed_capacity_tpy": query_context.feed_capacity_tpy,
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }

    lines = [
        "Use the adapter-produced optimization stage-candidate handoff as the authoritative decision set for the waste optimization solve.",
        "A validated structured optimization handoff is attached by the orchestrator. Pass that attached payload exactly as `stage_candidates_json`; do not reconstruct or reserialize it from prose.",
        f"Constraint mode: {constraint_mode}.",
        f"Fallback policy: {fallback_policy}.",
        f"Route pool mode: {route_pool_mode}.",
        f"Operating constraints: {json.dumps(operating_constraints, ensure_ascii=False)}.",
        (
            "The typed candidate pairs are temperature-aware optimizer options. "
            "If the same solvent appears at different temperatures, preserve those as distinct options downstream."
        ),
        (
            "The solvent filters preserve up to "
            f"{_MAX_OPTIMIZATION_SOLVENTS_PER_POLYMER} unique polymer-aware STRAP/BioSTEAM "
            "solvents per target polymer from the separation-engineer output when available."
        ),
        (
            "If route_pool_mode is exact and route_candidates is non-empty, the optimizer "
            "will enforce each route's exact candidate options and run a Pareto sweep per route. "
            "If route_pool_mode is slot_independent, the flattened temperature-aware candidate_pairs "
            "are the authoritative decision pool and route_candidates are retained only for provenance."
        ),
        (
            "The optimizer now accepts the broader polymer-aware STRAP/BioSTEAM solvent "
            "space and lets simulation plus post-sim validation decide what survives; "
            "do not pre-prune candidates in prose."
        ),
        (
            "Use feed metadata from the attached handoff when it is available. "
            "If the user supplied composition but not legacy fractions, pass `feed_composition_json` instead of inventing per-polymer fraction args."
        ),
        (
            f"Candidate counts by polymer: {json.dumps(candidate_counts_by_polymer, ensure_ascii=False)}. "
            f"Route candidates available: {len(route_candidates)}."
        ),
        "Respect the tool's fallback behavior and report any disclosed broadening explicitly.",
    ]
    if payload.get("candidate_backfill_warnings"):
        lines.append(
            "Candidate backfill warnings: "
            + json.dumps(payload.get("candidate_backfill_warnings"), ensure_ascii=False)
        )
    if query_context.feed_composition:
        lines.append(
            "Feed composition from the user request: "
            f"{json.dumps(query_context.feed_composition, ensure_ascii=False)}."
        )
    if query_context.feed_capacity_tpy is not None:
        lines.append(
            f"Feed capacity from the user request: {query_context.feed_capacity_tpy:g} tonnes/year."
        )
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

    Handles both pareto_front and point_optimum analyses with dedicated plot
    tools for each optimization result type.
    """
    payload = source.payload
    analysis_type = str(payload.get("analysis_type") or "").strip()
    requested_plot_mode = None
    if scope_user_query:
        lower_query = scope_user_query.lower()
        if "landscape" in lower_query or "all feasible" in lower_query:
            requested_plot_mode = "landscape"

    def _composition_output_stem(payload: dict[str, Any]) -> str | None:
        fractions = payload.get("feed_composition")
        if not isinstance(fractions, dict):
            feed = payload.get("feed")
            if isinstance(feed, dict):
                fractions = feed.get("composition") or feed.get("feed_composition")
        if not isinstance(fractions, dict) and scope_user_query:
            parsed_context = extract_query_context(scope_user_query)
            fractions = parsed_context.feed_composition
        if not isinstance(fractions, dict) or not fractions:
            return None

        query_text = str(scope_user_query or "").lower()
        order = ["PE", "LDPE", "EVOH", "PET", "PP", "PS", "PVC", "PC", "N6"]
        normalized: dict[str, float] = {}
        for polymer, value in fractions.items():
            label = str(polymer or "").strip().upper().strip('"').strip("'")
            if label == "PE" and "ldpe" in query_text:
                label = "LDPE"
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                continue
            if fraction <= 0:
                continue
            normalized[label] = fraction
        if not normalized:
            return None

        ordered_labels = [label for label in order if label in normalized]
        ordered_labels.extend(sorted(label for label in normalized if label not in set(ordered_labels)))

        parts: list[str] = []
        for label in ordered_labels:
            pct = normalized[label] * 100.0
            pct_text = f"{pct:.0f}" if abs(pct - round(pct)) < 1e-6 else f"{pct:.1f}".replace(".", "p")
            parts.append(f"{label.lower()}{pct_text}")
        y_metric = str(payload.get("y_metric") or "emissions").strip().lower()
        y_slug = "circularity" if "circular" in y_metric else "emissions" if "emission" in y_metric else re.sub(r"[^a-z0-9]+", "_", y_metric).strip("_")
        return f"optimization_pareto_{y_slug}_{'_'.join(parts)}"

    if analysis_type == "pareto_front":
        # Deep-copy-safe: the handoff store already owns the dict; we only pass
        # the reference forward. The plotting tool accepts either a JSON string
        # or the dict directly.
        pareto_payload = dict(payload)
        requested_output_stem = _composition_output_stem(payload)
        handoff_payload = {
            "source_handoff_id": source.handoff_id,
            "analysis_type": "pareto_front",
            "x_metric": payload.get("x_metric"),
            "y_metric": payload.get("y_metric"),
            "n_points_feasible": payload.get("n_points_feasible"),
            "pareto_result_json": pareto_payload,
            "requested_plot_tool": "plot_optimization_pareto_front",
            "requested_plot_mode": requested_plot_mode,
            "requested_output_stem": requested_output_stem,
            "source_user_query": scope_user_query,
            "source_task_prompt": source.task_prompt,
        }
        call_args = f'source_handoff_id="{source.handoff_id}"'
        if requested_output_stem:
            call_args += f', output_stem="{requested_output_stem}"'
        lines = [
            "Plot the upstream waste-optimization Pareto frontier.",
            "Required tool: plot_optimization_pareto_front.",
            f"Call `plot_optimization_pareto_front({call_args})`.",
            "Do not rebuild or paraphrase the Pareto payload from earlier separation prose.",
            f"Axes: x={payload.get('x_metric') or 'total_cost'}, y={payload.get('y_metric') or 'emissions'}.",
        ]
        if requested_output_stem:
            lines.append(f"Use output_stem=\"{requested_output_stem}\".")
        if requested_plot_mode:
            lines.append(f"Use plot_mode=\"{requested_plot_mode}\".")
        if scope_user_query:
            lines.append(f"Original user request: {scope_user_query}")
        return ("optimization_plot_context.v1", handoff_payload, " ".join(lines))

    if analysis_type == "pareto_slices":
        slices_payload = dict(payload)
        handoff_payload = {
            "source_handoff_id": source.handoff_id,
            "analysis_type": "pareto_slices",
            "x_metric": payload.get("x_metric"),
            "y_metric": payload.get("y_metric"),
            "n_slices_requested": payload.get("n_slices_requested"),
            "n_slices_solved": payload.get("n_slices_solved"),
            "pareto_slices_json": slices_payload,
            "requested_plot_tool": "plot_optimization_pareto_slices",
            "requested_plot_mode": requested_plot_mode or "landscape",
            "source_user_query": scope_user_query,
            "source_task_prompt": source.task_prompt,
        }
        lines = [
            "Plot the upstream multi-composition waste-optimization Pareto slices.",
            "Required tool: plot_optimization_pareto_slices.",
            f"Call `plot_optimization_pareto_slices(source_handoff_id=\"{source.handoff_id}\")`.",
            "This tool should create one PNG per solved composition slice and one combined comparison plot.",
            "Do not rebuild or paraphrase the Pareto payloads from earlier separation prose.",
            f"Axes: x={payload.get('x_metric') or 'total_cost'}, y={payload.get('y_metric') or 'circularity'}.",
            "Use plot_mode=\"landscape\".",
        ]
        if scope_user_query:
            lines.append(f"Original user request: {scope_user_query}")
        return ("optimization_plot_context.v1", handoff_payload, " ".join(lines))

    if analysis_type == "infeasible":
        handoff_payload = {
            "source_handoff_id": source.handoff_id,
            "analysis_type": "infeasible",
            "optimization_result_json": dict(payload),
            "requested_plot_tool": None,
            "source_user_query": scope_user_query,
            "source_task_prompt": source.task_prompt,
        }
        lines = [
            "The upstream waste-optimization result is infeasible.",
            "Do not call a plotting tool for this handoff.",
            "Report that no optimization figure was generated because the optimization payload is infeasible.",
        ]
        if scope_user_query:
            lines.append(f"Original user request: {scope_user_query}")
        return ("optimization_plot_context.v1", handoff_payload, " ".join(lines))

    # Point optimum or unknown analysis.
    handoff_payload = {
        "source_handoff_id": source.handoff_id,
        "analysis_type": analysis_type or "point_optimum",
        "optimization_result_json": dict(payload),
        "profit": payload.get("profit"),
        "emissions": payload.get("emissions"),
        "total_cost": payload.get("total_cost"),
        "circularity_score": payload.get("circularity_score"),
        "optimal_washes": payload.get("optimal_washes") or payload.get("wash1_selection") or [],
        "requested_plot_tool": "plot_optimization_point_result",
        "source_user_query": scope_user_query,
        "source_task_prompt": source.task_prompt,
    }
    lines = [
        "Plot the upstream waste-optimization point result.",
        "Required tool: plot_optimization_point_result.",
        f"Call `plot_optimization_point_result(source_handoff_id=\"{source.handoff_id}\")`.",
        "Do not rebuild or paraphrase the point-optimum payload from earlier separation prose.",
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
