"""Routing guard and synthetic-response helpers."""

from __future__ import annotations

import json
import posixpath
import re
from uuid import uuid4

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .guardrail_utils import extract_user_temperature_limit_c
from .handoffs import (
    build_multi_source_handoff_for_consumer,
    get_latest_result_handoff,
    normalize_agent_payload,
)
from .routing_classifier import derive_workflow_dependencies
from .routing_progress import (
    _extract_completed_subagent_calls,
    _extract_returned_subagent_calls,
    _get_active_remaining_steps,
    _get_allowed_subagent_names,
    _get_downstream_subagents,
    _get_last_human_message,
    _get_latest_dispatch_for_subagent,
    _get_missing_required_handoffs_for_consumer,
    _get_ordered_plan,
    _get_pending_required_handoff,
    _get_ready_downstream_handoff,
    _get_step_dependencies,
    _get_task_handoff_statuses,
    _get_tool_message_registry,
    _has_built_handoff_since,
    _normalize_task_description,
    _parse_tool_json_content,
    _task_descriptions_match,
    _workflow_is_active,
)
from .solubility import get_boiling_point

_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)

_WORKFLOW_SAFE_TOOLS = {
    "think",
    "task",
    "read_file",
    "write_file",
    "write_todos",
    "get_subagent_result",
    "get_subagent_results",
    "get_all_subagent_results",
    "list_handoffs",
    "get_handoff_details",
    "build_handoff",
    "list_tables",
    "describe_table",
    "query_database",
    "list_available_polymers",
    "list_available_solvents",
}
_SEPARATION_PREFLIGHT_ONLY_TOOLS = {
    "list_available_polymers",
    "list_available_solvents",
    "list_tables",
    "describe_table",
}
_SEPARATION_SPECIALIST_TOOLS = {
    "find_optimal_separation_conditions",
    "analyze_selective_solubility_enhanced",
    "lookup_solvent_price",
    "lookup_solvent_gwp",
    "plan_sequential_separation",
    "view_alternative_separation_sequence",
    "plan_multiple_separation_schemes",
    "find_optimal_separation_sequence",
    "analyze_integrated_separation",
    "find_antisolvents",
    "find_antisolvent_pairs",
    "analyze_selective_antisolvent_precipitation",
    "find_differential_precipitation_solvents",
    "analyze_multi_polymer_precipitation",
    "analyze_precipitation_temperature",
    "compare_polymer_pairs_precipitation",
    "check_atmospheric_feasibility",
    "check_multi_polymer_atmospheric_feasibility",
    "optimize_separation_temperature",
    "calculate_selectivity_detailed",
    "rank_solvents_for_separation",
    "build_compatibility_matrix",
    "find_challenging_polymer_pairs",
    "get_supported_polymers_and_solvents",
    "rank_solvents_selectivity",
    "find_optimal_separation_conditions",
    "create_separation_tree_plot",
}
_HANDOFF_REQUIRED_TOOLS = {
    "build_handoff",
    "list_handoffs",
    "get_handoff_details",
    "think",
    "read_file",
    "write_file",
}
_GUARDED_FILESYSTEM_TOOLS = {"ls", "glob", "grep", "edit_file", "execute"}
_CONTEXTUAL_FILESYSTEM_TOOLS = {"ls", "glob"}
_FILE_PRODUCING_SUBAGENTS = {
    "biosteam-analyst",
    "rag-analyst",
    "safety-analyst",
    "separation-engineer",
    "statistics-ml",
    "visualization-specialist",
}
_DEFAULT_ARTIFACT_DIRS = {"/plots", "/rag_pdfs"}
_VISUALIZATION_ROUTE_TOOLS = {
    "create_separation_tree_plot",
    "create_selectivity_heatmap",
    "create_process_flow_diagram",
    "plot_atmospheric_feasibility",
    "plot_multi_panel_analysis",
    "plot_comparison_dashboard",
    "plot_selectivity_heatmap",
    "visualize_biosteam_results",
}
_VISUALIZATION_REQUEST_RE = re.compile(
    r"\b("
    r"plot|plots|chart|charts|diagram|diagrams|figure|figures|heatmap|heatmaps|"
    r"dashboard|dashboards|visualize|visualization|flow diagram|flowchart|show .*plot"
    r")\b",
    re.IGNORECASE,
)
_CONTAMINANT_SCREEN_MODES = {"leaching", "strap_contaminant_removal", "tie"}
_USER_FILE_REQUEST_RE = re.compile(
    r"\b("
    r"inspect (?:the )?(?:local )?(?:files?|directories?|folders?|repo|repository|codebase)|"
    r"search (?:the )?(?:repo|repository|codebase|files?)|"
    r"list (?:the )?(?:files?|directories?|folders?)|"
    r"open (?:the )?(?:file|directory|folder)|"
    r"read (?:the )?file|"
    r"check (?:the )?(?:repo|repository|codebase|files?)|"
    r"\b(?:ls|grep|glob)\b|"
    r"file path|local path"
    r")\b",
    re.IGNORECASE,
)
_PATH_TOKEN_RE = re.compile(r"`([^`\n]*(?:/|\./)[^`\n]+)`|(?<![\w])((?:\./|/)[A-Za-z0-9._~\-/]+)")


def _query_explicitly_requests_visualization(messages: list) -> bool:
    query = _get_last_human_message(messages)
    if not query:
        return False
    return bool(_VISUALIZATION_REQUEST_RE.search(query))


def _strip_structured_result_block(text: str) -> str:
    stripped = _STRUCTURED_RESULT_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def _extract_structured_payload_from_task_message(message: ToolMessage) -> dict | None:
    content = message.content if isinstance(message.content, str) else str(message.content)
    match = _STRUCTURED_RESULT_RE.search(content)
    if not match:
        return None
    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()
    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_separation_payload_fallback(messages: list, payload: dict) -> str:
    polymers = [str(polymer) for polymer in payload.get("polymers", []) if str(polymer).strip()]
    supported = [str(polymer) for polymer in payload.get("supported_polymers", []) if str(polymer).strip()]
    unsupported = [str(polymer) for polymer in payload.get("unsupported_polymers", []) if str(polymer).strip()]
    sequence = [str(polymer) for polymer in payload.get("best_sequence", []) if str(polymer).strip()]
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    max_temp_c = extract_user_temperature_limit_c(messages)

    lines: list[str] = []
    if sequence:
        lines.append(f"Recommended separation sequence: {' -> '.join(sequence)}.")
    else:
        lines.append("Recommended separation sequence: no fully executable atmospheric-pressure sequence was established.")

    if supported or unsupported:
        supported_text = ", ".join(supported) if supported else "none"
        lines.append(f"Supported subset: {supported_text}.")
        if unsupported:
            lines.append(
                "Unsupported polymers: "
                f"{', '.join(unsupported)}. Conclusions apply only to the supported subset."
            )

    if not steps:
        lines.append(
            "No feasible step-level solvent and temperature sequence was established from the validated separation result."
        )
        return "\n\n".join(lines)

    lines.append("Step-by-step recommendation:")
    any_infeasible = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_no = step.get("step", "?")
        polymer = str(step.get("polymer", "")).strip() or "target polymer"
        solvent = str(step.get("solvent", "")).strip() or "unspecified solvent"
        temp_value = step.get("temperature_c", step.get("temp"))
        try:
            operating_temp_c = float(temp_value)
            temp_text = f"{operating_temp_c:.1f}°C"
        except (TypeError, ValueError):
            operating_temp_c = None
            temp_text = "an unspecified temperature"

        line = f"- Step {step_no}: use {solvent} to recover {polymer} at {temp_text}."
        boiling_point_c = get_boiling_point(solvent)
        if boiling_point_c is not None and operating_temp_c is not None:
            if operating_temp_c >= boiling_point_c:
                any_infeasible = True
                line += (
                    f" This step is infeasible at atmospheric pressure because {solvent} boils at "
                    f"{boiling_point_c:.1f}°C."
                )
            else:
                line += (
                    f" This step remains below the {boiling_point_c:.1f}°C boiling point of {solvent} at 1 atm."
                )
                bp_margin = boiling_point_c - operating_temp_c
                if bp_margin <= 5.0:
                    line += " The atmospheric-pressure operating margin is narrow and needs careful temperature control."
                elif max_temp_c is not None and max_temp_c >= boiling_point_c - 10.0:
                    line += (
                        f" Operate at {operating_temp_c:.1f}°C rather than the upper bound of {max_temp_c:.1f}°C."
                    )
        lines.append(line)

    if any_infeasible:
        lines.append("Overall assessment: no fully feasible atmospheric-pressure route was established from these steps.")
    else:
        lines.append("Overall assessment: this is the recommended feasible atmospheric-pressure route within the current validated result.")
    return "\n\n".join(lines)


def _biosteam_prose_missing_economic_metrics(prose: str) -> bool:
    text = (prose or "").lower()
    has_tci = any(term in text for term in ("tci", "capex", "capital cost"))
    has_aoc = any(term in text for term in ("aoc", "opex", "operating cost"))
    return not (has_tci and has_aoc)


def _format_money_millions(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    try:
        return f"${float(value) / 1e6:.2f}M"
    except (TypeError, ValueError):
        return "N/A"


def _extract_biosteam_metric(item: dict, nested_group: str, key: str):
    nested = item.get(nested_group)
    if isinstance(nested, dict) and nested.get(key) is not None:
        return nested.get(key)
    return item.get(key)


def _build_biosteam_payload_fallback(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None

    lines: list[str] = []
    target = str(payload.get("target_plastic", "")).strip()
    energy_case = str(payload.get("energy_case", "")).strip()
    if target or energy_case:
        header = "BioSTEAM summary"
        if target and energy_case:
            header += f" for {target} ({energy_case})"
        elif target:
            header += f" for {target}"
        elif energy_case:
            header += f" ({energy_case})"
        lines.append(header + ".")

    results = payload.get("results")
    if isinstance(results, list) and results:
        successes: list[dict] = [item for item in results if isinstance(item, dict) and item.get("success", True)]
        ranked: list[dict] = []
        for item in successes:
            msp = _extract_biosteam_metric(item, "tea", "msp_usd_per_kg")
            try:
                msp_key = float(msp) if msp is not None else float("inf")
            except (TypeError, ValueError):
                msp_key = float("inf")
            ranked.append({"item": item, "msp_key": msp_key})
        ranked.sort(key=lambda entry: entry["msp_key"])
        top = ranked[:5]
        if top:
            lines.append("Top scenarios by MSP:")
            for idx, entry in enumerate(top, 1):
                item = entry["item"]
                label = (
                    item.get("scenario_label")
                    or item.get("solvent")
                    or item.get("label")
                    or f"Scenario {idx}"
                )
                msp = _extract_biosteam_metric(item, "tea", "msp_usd_per_kg")
                gwp = _extract_biosteam_metric(item, "lca", "gwp_kg_co2e_per_kg")
                tci = _extract_biosteam_metric(item, "tea", "tci_usd")
                aoc = _extract_biosteam_metric(item, "tea", "aoc_usd_per_yr")
                msp_text = f"${float(msp):.2f}/kg" if msp is not None else "N/A"
                gwp_text = f"{float(gwp):.2f} kg CO2e/kg" if gwp is not None else "N/A"
                lines.append(
                    f"{idx}. {label}: MSP {msp_text}; GWP {gwp_text}; "
                    f"TCI {_format_money_millions(tci)}; AOC {_format_money_millions(aoc)}."
                )
            return "\n\n".join(lines)

    if "combined_tci_usd" in payload or "combined_aoc_usd_per_yr" in payload:
        blended_msp = payload.get("blended_msp_usd_per_kg")
        blended_gwp = payload.get("weighted_gwp_kg_co2e_per_kg")
        lines.append(
            "Combined metrics: "
            f"MSP {f'${float(blended_msp):.2f}/kg' if blended_msp is not None else 'N/A'}; "
            f"GWP {f'{float(blended_gwp):.2f} kg CO2e/kg' if blended_gwp is not None else 'N/A'}; "
            f"TCI {_format_money_millions(payload.get('combined_tci_usd'))}; "
            f"AOC {_format_money_millions(payload.get('combined_aoc_usd_per_yr'))}."
        )
        return "\n\n".join(lines)

    if "tci_percentiles" in payload or "msp_percentiles" in payload:
        msp = payload.get("msp_percentiles") if isinstance(payload.get("msp_percentiles"), dict) else {}
        gwp = payload.get("gwp_percentiles") if isinstance(payload.get("gwp_percentiles"), dict) else {}
        tci = payload.get("tci_percentiles") if isinstance(payload.get("tci_percentiles"), dict) else {}
        msp_p50 = msp.get("p50")
        gwp_p50 = gwp.get("p50")
        msp_text = f"${float(msp_p50):.2f}/kg" if msp_p50 is not None else "N/A"
        gwp_text = f"{float(gwp_p50):.2f} kg CO2e/kg" if gwp_p50 is not None else "N/A"
        lines.append(
            "Uncertainty summary: "
            f"MSP median {msp_text}; "
            f"GWP median {gwp_text}; "
            f"TCI median {_format_money_millions(tci.get('p50'))}."
        )
        return "\n\n".join(lines)

    return None


def _contaminant_prose_missing_screening_metrics(prose: str) -> bool:
    text = prose.lower()
    has_logd = "logd" in text or "partition" in text
    has_miscibility = "miscible" in text or "miscibility" in text
    return not (has_logd and has_miscibility)


def _build_contaminant_payload_fallback(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None

    payload = normalize_agent_payload("contaminant-removal-analyst", payload)
    mode = payload.get("mode")
    no_data = bool(payload.get("no_data"))
    target = payload.get("target_polymer")
    contaminants = payload.get("supported_contaminants") or payload.get("contaminants") or []
    unsupported = payload.get("unsupported_contaminants") or []
    candidate_solvents = payload.get("candidate_solvents") or []
    recommended_mode = payload.get("recommended_mode") or mode
    recommended_raw = payload.get("recommended_solvents")
    if isinstance(recommended_raw, dict):
        recommended = [
            str(item)
            for item in (recommended_raw.get(recommended_mode or mode) or [])
            if str(item).strip()
        ]
    else:
        recommended = [str(item) for item in (recommended_raw or []) if str(item).strip()]

    lines: list[str] = []
    if target:
        lines.append(f"Contaminant-removal screening summary for {target}.")
    if contaminants:
        lines.append(f"Supported contaminants screened: {', '.join(str(item) for item in contaminants)}.")
    if unsupported:
        lines.append(f"Unsupported contaminants: {', '.join(str(item) for item in unsupported)}.")
    if mode == "comparison":
        if recommended_mode in {"leaching", "strap_contaminant_removal"}:
            lines.append(f"Recommended mode: {recommended_mode}.")
            lines.append(
                f"Recommended solvents for that mode: {', '.join(str(item) for item in recommended) if recommended else 'None'}."
            )
        elif recommended_mode == "tie":
            lines.append("No mode clearly won the screening comparison.")
        elif no_data and unsupported:
            lines.append(
                "No supported comparison result is available because the requested contaminant set is unsupported."
            )
        else:
            lines.append("No mode clearly won the screening comparison.")
    else:
        lines.append(f"Mode screened: {mode}.")
        lines.append(
            f"Recommended solvents: {', '.join(str(item) for item in recommended) if recommended else 'None'}."
        )

    passing = [
        row for row in candidate_solvents
        if isinstance(row, dict)
        and row.get("passes")
        and (mode != "comparison" or row.get("screen_mode") == recommended_mode)
    ]
    if passing:
        if recommended_mode not in _CONTAMINANT_SCREEN_MODES:
            recommended_mode = str(passing[0].get("screen_mode") or recommended_mode or mode)
        lines.append(
            "Recommended candidates kept the contaminants miscible in the solvent and maintained positive contaminant logD."
        )
        for row in passing[:3]:
            solvent = row.get("solvent", "unknown solvent")
            fragments: list[str] = []
            if row.get("contaminant_logd_min") is not None:
                fragments.append(f"minimum contaminant logD {float(row['contaminant_logd_min']):.2f}")
            if row.get("operating_temperature_c") is not None:
                fragments.append(f"screened operating temperature {float(row['operating_temperature_c']):.1f}°C")
            lines.append(f"- {solvent}: {'; '.join(fragments) if fragments else 'passes the screening criteria'}.")
    else:
        lines.append(
            "No robust recommendation passed the full screening criteria for contaminant miscibility, positive logD, and polymer behavior."
        )

    caveats = payload.get("caveats") or []
    if caveats:
        lines.append("Caveats:")
        for item in caveats[:4]:
            lines.append(f"- {item}")

    answer = "\n".join(lines).strip()
    return answer or None


def _get_latest_task_payload_bundle(messages: list, subagent: str) -> tuple[dict | None, dict | None, str, str | None]:
    latest_dispatch = _get_latest_dispatch_for_subagent(messages, subagent)
    if latest_dispatch is None:
        return None, None, "", None
    tool_result = _get_tool_message_registry(messages).get(latest_dispatch["tool_call_id"])
    if tool_result is None:
        return latest_dispatch, None, "", None
    task_message = tool_result["message"]
    content = task_message.content if isinstance(task_message.content, str) else str(task_message.content)
    prose = _strip_structured_result_block(content)
    payload = _extract_structured_payload_from_task_message(task_message)
    status = _get_task_handoff_statuses(messages).get(latest_dispatch["tool_call_id"])
    return latest_dispatch, payload, prose, status


def _build_separation_contaminant_payload_fallback(messages: list) -> AIMessage | None:
    sep_dispatch, sep_payload, _, sep_status = _get_latest_task_payload_bundle(messages, "separation-engineer")
    contam_dispatch, contam_payload, _, contam_status = _get_latest_task_payload_bundle(
        messages,
        "contaminant-removal-analyst",
    )
    if not isinstance(sep_payload, dict) or not isinstance(contam_payload, dict):
        return None

    contam_payload = normalize_agent_payload("contaminant-removal-analyst", contam_payload)
    contam_no_data = bool(contam_payload.get("no_data"))
    target_polymer = str(contam_payload.get("target_polymer", "")).strip()
    other_polymers = [
        str(item) for item in contam_payload.get("other_polymers", []) if str(item).strip()
    ]
    contaminants = [
        str(item) for item in (contam_payload.get("supported_contaminants") or contam_payload.get("contaminants") or [])
        if str(item).strip()
    ]
    recommended_mode = str(contam_payload.get("recommended_mode") or contam_payload.get("mode") or "").strip()
    recommended_raw = contam_payload.get("recommended_solvents")
    if isinstance(recommended_raw, dict):
        recommended_solvents = [
            str(item)
            for item in (recommended_raw.get(recommended_mode) or [])
            if str(item).strip()
        ]
    else:
        recommended_solvents = [str(item) for item in (recommended_raw or []) if str(item).strip()]

    top_solvents = [str(item) for item in sep_payload.get("top_solvents", []) if str(item).strip()]
    if not top_solvents:
        top_solvents = [
            str(solvent)
            for solvent in (sep_payload.get("solvent_mapping") or {}).values()
            if str(solvent).strip()
        ]
    if not top_solvents:
        for sequence in sep_payload.get("top_k_sequences", []) or []:
            if not isinstance(sequence, dict):
                continue
            for solvent in (sequence.get("solvent_mapping") or {}).values():
                solvent_text = str(solvent).strip()
                if solvent_text and solvent_text not in top_solvents:
                    top_solvents.append(solvent_text)

    passing_rows = [
        row for row in (contam_payload.get("candidate_solvents") or [])
        if isinstance(row, dict)
        and row.get("passes")
        and (
            not recommended_mode
            or row.get("screen_mode") in {None, "", recommended_mode}
            or contam_payload.get("mode") != "comparison"
        )
    ]
    selected_row = None
    for solvent in recommended_solvents:
        selected_row = next(
            (
                row for row in passing_rows
                if str(row.get("solvent", "")).strip().lower() == solvent.lower()
            ),
            None,
        )
        if selected_row is not None:
            break
    if selected_row is None and passing_rows:
        overlap = {
            solvent.lower(): solvent
            for solvent in top_solvents
        }
        selected_row = next(
            (
                row for row in passing_rows
                if str(row.get("solvent", "")).strip().lower() in overlap
            ),
            passing_rows[0],
        )
    if recommended_mode not in _CONTAMINANT_SCREEN_MODES and isinstance(selected_row, dict):
        recommended_mode = str(selected_row.get("screen_mode") or recommended_mode or contam_payload.get("mode") or "").strip()

    best_sequence = [str(item) for item in sep_payload.get("best_sequence", []) if str(item).strip()]
    lines: list[str] = []
    if target_polymer:
        if other_polymers:
            lines.append(
                f"Integrated screening summary for recovering {target_polymer} from "
                f"{', '.join(other_polymers)} while removing {', '.join(contaminants) if contaminants else 'the supported contaminants'}."
            )
        else:
            lines.append(
                f"Integrated screening summary for {target_polymer} with "
                f"{', '.join(contaminants) if contaminants else 'the supported contaminants'}."
            )

    if best_sequence or top_solvents:
        lines.append(
            "Separation screening identified the following predicted selective-dissolution candidates: "
            f"{', '.join(top_solvents[:3]) if top_solvents else 'none'}."
        )
        if best_sequence:
            lines.append(
                "Predicted separation order: " + " -> ".join(best_sequence) + "."
            )
        sep_steps = sep_payload.get("steps") if isinstance(sep_payload.get("steps"), list) else []
        target_step = None
        for step in sep_steps:
            if not isinstance(step, dict):
                continue
            if target_polymer and str(step.get("polymer", "")).strip().lower() == target_polymer.lower():
                target_step = step
                break
        if target_step is None and sep_steps:
            target_step = sep_steps[0]
        if isinstance(target_step, dict):
            solvent = str(target_step.get("solvent", "")).strip()
            temp_value = target_step.get("temperature_c", target_step.get("temp"))
            try:
                temp_text = f"{float(temp_value):.1f}°C"
            except (TypeError, ValueError):
                temp_text = None
            if solvent and temp_text:
                lines.append(
                    f"The upstream separation result screened {solvent} at {temp_text} as a candidate operating point."
                )

    if recommended_mode in {"leaching", "strap_contaminant_removal"}:
        mode_text = recommended_mode.replace("_", " ")
        lines.append(f"Contaminant screening preferred {mode_text}.")
    elif recommended_mode == "tie":
        lines.append("No mode clearly won the contaminant-screening comparison.")
    elif contam_no_data and contam_payload.get("unsupported_contaminants"):
        lines.append(
            "No supported contaminant-screening mode result is available because the requested contaminant set is unsupported."
        )
    else:
        lines.append("No mode clearly won the contaminant-screening comparison.")
    if contaminants:
        lines.append(
            f"Target contaminants evaluated: {', '.join(contaminants)}."
        )

    if selected_row is not None:
        solvent = str(selected_row.get("solvent", "")).strip() or "the selected solvent"
        mode_prefix = "In the recommended contaminant-removal screen,"
        details: list[str] = [f"{mode_prefix} {solvent} kept the contaminants miscible"]
        logd_min = selected_row.get("contaminant_logd_min")
        if logd_min is not None:
            details.append(f"with minimum logD {float(logd_min):.2f}")
        details_text = " ".join(details) + "."
        lines.append(details_text)

        operating_temperature_c = selected_row.get("operating_temperature_c")
        precipitation_temperature_c = selected_row.get("precipitation_temperature_c")
        boiling_point_c = selected_row.get("boiling_point_c")
        if operating_temperature_c is not None:
            line = f"Screened operating temperature: {float(operating_temperature_c):.1f}°C"
            if precipitation_temperature_c is not None:
                line += f"; screened precipitation temperature: {float(precipitation_temperature_c):.1f}°C"
            if boiling_point_c is not None:
                line += f"; boiling point: {float(boiling_point_c):.1f}°C at 1 atm"
            lines.append(line + ".")
    elif recommended_solvents:
        lines.append(
            "Recommended contaminant-removal solvents: " + ", ".join(recommended_solvents) + "."
        )
        lines.append(
            "These recommendations require explicit review of contaminant miscibility and positive logD from the screening payload."
        )
    else:
        lines.append(
            "No solvent passed the full contaminant-removal criteria for explicit contaminant miscibility, positive logD, and polymer behavior."
        )

    if selected_row is not None and target_polymer:
        solvent = str(selected_row.get("solvent", "")).strip()
        overlap = {item.lower() for item in top_solvents}
        if solvent and solvent.lower() in overlap:
            lines.append(
                f"Final recommendation: use {solvent} as the leading integrated screening candidate for dissolving {target_polymer}, "
                "keeping the non-target polymer(s) out of solution, and retaining the contaminants in the solvent phase."
            )
        elif solvent:
            lines.append(
                f"Final recommendation: {solvent} is the leading contaminant-removal candidate, but it should be treated as a follow-on "
                "screening result rather than a fully validated integrated separation solvent because it was not the top-ranked upstream separation solvent."
            )

    lines.append(
        "Treat this as a screening-based recommendation. Confirm separation selectivity, contaminant retention, and polymer precipitation experimentally before scale-up."
    )

    origin_status = contam_status or sep_status
    origin_tool_call_id = (
        contam_dispatch["tool_call_id"] if contam_dispatch is not None else sep_dispatch["tool_call_id"]
    )
    return _build_origin_tagged_ai_message(
        "\n\n".join(line for line in lines if line).strip(),
        origin="routing_multi_specialist_separation_contaminant_fallback",
        subagent="contaminant-removal-analyst",
        tool_call_id=origin_tool_call_id,
        status=origin_status,
    )


def _build_origin_tagged_ai_message(
    content: str,
    *,
    origin: str,
    subagent: str,
    tool_call_id: str,
    status: str | None = None,
) -> AIMessage:
    additional_kwargs = {
        "strap_origin": origin,
        "strap_subagent": subagent,
        "strap_tool_call_id": tool_call_id,
    }
    if status:
        additional_kwargs["strap_handoff_status"] = status
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


def _build_missing_separation_failure_fallback(
    messages: list,
    *,
    task_prompt: str | None = None,
    error_message: str | None = None,
) -> str:
    """Deterministic fallback when separation returned without a structured result."""
    query = _get_last_human_message(messages) or task_prompt or "the requested separation"
    lines = [
        "A validated step-by-step separation sequence could not be extracted from the latest separation-engineer return.",
        "No solvent/temperature sequence should be treated as recommended from this failed summary alone.",
    ]
    if error_message:
        lines.append(f"Recorded issue: {error_message}")
    lines.append(
        "Current conclusion: treat the route as unresolved rather than feasible until the separation-engineer returns a usable final result."
    )
    lines.append(f"Original request context: {query}")
    return "\n\n".join(lines)


def build_single_specialist_separation_ai_message(messages: list) -> AIMessage | None:
    """Build a terminal answer directly from the latest returned separation task.

    This is intentionally narrow: it only applies when the executed route is
    exactly a single returned `separation-engineer` task. It is used both by
    routing short-circuiting and by the harness as an emergency answer-source
    when the agent returned messages but no final prose answer.
    """
    returned_calls = [
        dispatch
        for dispatch in _extract_returned_subagent_calls(messages)
        if dispatch["subagent"] == "separation-engineer"
    ]
    if not returned_calls:
        return None

    all_returned_subagents = {
        dispatch["subagent"] for dispatch in _extract_returned_subagent_calls(messages)
    }
    if all_returned_subagents != {"separation-engineer"}:
        return None

    latest_dispatch = max(returned_calls, key=lambda dispatch: dispatch["message_index"])
    tool_result = _get_tool_message_registry(messages).get(latest_dispatch["tool_call_id"])
    if tool_result is None:
        return None

    task_message = tool_result["message"]
    content = task_message.content if isinstance(task_message.content, str) else str(task_message.content)
    prose = _strip_structured_result_block(content)
    status = _get_task_handoff_statuses(messages).get(latest_dispatch["tool_call_id"])
    if prose:
        return _build_origin_tagged_ai_message(
            prose,
            origin="routing_single_specialist_prose",
            subagent="separation-engineer",
            tool_call_id=latest_dispatch["tool_call_id"],
            status=status,
        )

    payload = _extract_structured_payload_from_task_message(task_message)
    if not isinstance(payload, dict):
        latest_record = get_latest_result_handoff(producer="separation-engineer")
        if (
            latest_record is not None
            and latest_record.source_tool_call_id == latest_dispatch["tool_call_id"]
            and latest_record.status == "missing"
        ):
            raw_preview = str(latest_record.payload.get("raw_text_preview", "")).strip()
            if raw_preview:
                prose = _strip_structured_result_block(raw_preview)
                if prose:
                    return _build_origin_tagged_ai_message(
                        prose,
                        origin="routing_single_specialist_missing_preview",
                        subagent="separation-engineer",
                        tool_call_id=latest_dispatch["tool_call_id"],
                        status=latest_record.status,
                    )
            return _build_origin_tagged_ai_message(
                _build_missing_separation_failure_fallback(
                    messages,
                    task_prompt=latest_record.task_prompt,
                    error_message=latest_record.payload.get("message"),
                ),
                origin="routing_single_specialist_missing_fallback",
                subagent="separation-engineer",
                tool_call_id=latest_dispatch["tool_call_id"],
                status=latest_record.status,
            )
        return None

    return _build_origin_tagged_ai_message(
        _build_separation_payload_fallback(messages, payload),
        origin="routing_single_specialist_separation_fallback",
        subagent="separation-engineer",
        tool_call_id=latest_dispatch["tool_call_id"],
        status=status,
    )


def _build_single_specialist_completion_response(
    messages: list,
    allowed_rules: list[dict],
) -> ModelResponse | None:
    completed_calls = _extract_completed_subagent_calls(messages)
    completed_names = {dispatch["subagent"] for dispatch in completed_calls}
    allowed_names = _get_allowed_subagent_names(allowed_rules)
    if len(allowed_names) != 1:
        return None

    subagent = next(iter(allowed_names))
    returned_calls = [
        dispatch
        for dispatch in _extract_returned_subagent_calls(messages)
        if dispatch["subagent"] == subagent
    ]
    if not returned_calls:
        return None

    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if ordered_plan:
        remaining = _get_active_remaining_steps(messages, ordered_plan)
        if remaining and completed_names != allowed_names:
            if subagent == "separation-engineer":
                ai_message = build_single_specialist_separation_ai_message(messages)
                if ai_message is not None:
                    return ModelResponse(result=[ai_message])
            return None

    latest_dispatch = max(returned_calls, key=lambda dispatch: dispatch["message_index"])
    if latest_dispatch is None:
        return None

    tool_result = _get_tool_message_registry(messages).get(latest_dispatch["tool_call_id"])
    if tool_result is None:
        return None

    task_message = tool_result["message"]
    content = task_message.content if isinstance(task_message.content, str) else str(task_message.content)
    prose = _strip_structured_result_block(content)
    payload = _extract_structured_payload_from_task_message(task_message)
    status = _get_task_handoff_statuses(messages).get(latest_dispatch["tool_call_id"])
    if subagent == "biosteam-analyst" and isinstance(payload, dict):
        fallback = _build_biosteam_payload_fallback(payload)
        if fallback and (not prose or _biosteam_prose_missing_economic_metrics(prose)):
            return ModelResponse(result=[_build_origin_tagged_ai_message(
                fallback,
                origin="routing_single_specialist_biosteam_fallback",
                subagent=subagent,
                tool_call_id=latest_dispatch["tool_call_id"],
                status=status,
            )])
    if subagent == "contaminant-removal-analyst" and isinstance(payload, dict):
        fallback = _build_contaminant_payload_fallback(payload)
        if fallback and (not prose or _contaminant_prose_missing_screening_metrics(prose)):
            return ModelResponse(result=[_build_origin_tagged_ai_message(
                fallback,
                origin="routing_single_specialist_contaminant_fallback",
                subagent=subagent,
                tool_call_id=latest_dispatch["tool_call_id"],
                status=status,
            )])
    if prose:
        return ModelResponse(result=[_build_origin_tagged_ai_message(
            prose,
            origin="routing_single_specialist_prose",
            subagent=subagent,
            tool_call_id=latest_dispatch["tool_call_id"],
            status=status,
        )])

    if not isinstance(payload, dict):
        return None

    if subagent == "separation-engineer":
        fallback = _build_separation_payload_fallback(messages, payload)
        return ModelResponse(result=[_build_origin_tagged_ai_message(
            fallback,
            origin="routing_single_specialist_separation_fallback",
            subagent=subagent,
            tool_call_id=latest_dispatch["tool_call_id"],
            status=_get_task_handoff_statuses(messages).get(latest_dispatch["tool_call_id"]),
        )])

    return None


def _build_multi_specialist_completion_response(
    messages: list,
    allowed_rules: list[dict],
) -> ModelResponse | None:
    allowed_names = _get_allowed_subagent_names(allowed_rules)
    if allowed_names != {"separation-engineer", "contaminant-removal-analyst"}:
        return None

    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if not ordered_plan:
        return None
    remaining = _get_active_remaining_steps(messages, ordered_plan)
    if remaining:
        return None

    returned_names = {
        dispatch["subagent"] for dispatch in _extract_returned_subagent_calls(messages)
    }
    if allowed_names - returned_names:
        return None

    ai_message = _build_separation_contaminant_payload_fallback(messages)
    if ai_message is None:
        return None
    return ModelResponse(result=[ai_message])

def _build_filesystem_guard_messages(messages: list) -> list[ToolMessage]:
    """Create error ToolMessages for unjustified filesystem exploration calls."""
    last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if not last_ai_msg or not getattr(last_ai_msg, "tool_calls", None):
        return []

    guard_messages: list[ToolMessage] = []
    for tool_call in last_ai_msg.tool_calls:
        validation_error = _validate_filesystem_tool_call(tool_call, messages)
        if validation_error is None:
            continue

        guard_messages.append(
            ToolMessage(
                content=(
                    f"Router guard: {validation_error} "
                    "If you need filesystem access, cite the exact returned path or rely on "
                    "`get_subagent_results(...)`, `list_handoffs(...)`, or domain tools instead."
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )
        )

    return guard_messages


def _normalize_virtual_path(path: str | None) -> str | None:
    """Normalize a tool or message path into the virtual filesystem form."""
    if not isinstance(path, str):
        return None

    cleaned = path.strip().strip("`'\"")
    cleaned = re.sub(r"[),.;:]+$", "", cleaned)
    if not cleaned:
        return None
    if cleaned.startswith("./"):
        cleaned = f"/{cleaned[2:]}"
    if not cleaned.startswith("/"):
        return None

    normalized = posixpath.normpath(cleaned)
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized.rstrip("/") or "/"


def _extract_paths_from_text(text: str) -> set[str]:
    """Extract path-like tokens from tool outputs or tool arguments."""
    paths: set[str] = set()
    if not isinstance(text, str):
        return paths

    for match in _PATH_TOKEN_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        normalized = _normalize_virtual_path(raw)
        if normalized:
            paths.add(normalized)
    return paths


def _expand_parent_paths(paths: set[str]) -> set[str]:
    """Include parent directories so `ls` can follow file-producing tool outputs."""
    expanded = set(paths)
    for path in list(paths):
        current = path
        while current not in {"", "/"}:
            current = posixpath.dirname(current.rstrip("/")) or "/"
            if current == "/":
                break
            expanded.add(current)
    return expanded


def _iter_string_values(value) -> list[str]:
    """Collect string leaves from nested tool-call arguments."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_iter_string_values(item))
        return items
    if isinstance(value, dict):
        items: list[str] = []
        for item in value.values():
            items.extend(_iter_string_values(item))
        return items
    return []


def _collect_recent_path_context(messages: list) -> set[str]:
    """Collect concrete file paths that were already mentioned by tool outputs."""
    paths: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            paths.update(_extract_paths_from_text(content))
    return _expand_parent_paths(paths)


def _extract_tool_call_paths(tool_call: dict) -> set[str]:
    """Extract concrete filesystem paths referenced by a pending tool call."""
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return set()

    paths: set[str] = set()
    for value in _iter_string_values(args):
        paths.update(_extract_paths_from_text(value))
    return _expand_parent_paths(paths)


def _user_requested_local_file_inspection(messages: list) -> bool:
    """Return True when the user explicitly asked to inspect the local filesystem."""
    query = _get_last_human_message(messages)
    if not query:
        return False
    return bool(_USER_FILE_REQUEST_RE.search(query))


def _get_latest_returned_subagent(messages: list) -> str | None:
    """Return the most recently completed subagent name, if any."""
    returned = _extract_returned_subagent_calls(messages)
    if not returned:
        return None
    return returned[-1]["subagent"]


def _tool_paths_match_context(tool_call: dict, allowed_paths: set[str]) -> bool:
    """Return True when the pending tool targets a recently mentioned path."""
    if not allowed_paths:
        return False
    return bool(_extract_tool_call_paths(tool_call) & allowed_paths)


def _is_default_artifact_followup(tool_call: dict, messages: list) -> bool:
    """Allow lightweight directory follow-up after plot/pdf-producing subagents."""
    if tool_call.get("name") not in _CONTEXTUAL_FILESYSTEM_TOOLS:
        return False

    latest_subagent = _get_latest_returned_subagent(messages)
    if latest_subagent not in _FILE_PRODUCING_SUBAGENTS:
        return False

    target_paths = _extract_tool_call_paths(tool_call)
    if not target_paths:
        return False

    for target in target_paths:
        for artifact_dir in _DEFAULT_ARTIFACT_DIRS:
            if target == artifact_dir or target.startswith(f"{artifact_dir}/"):
                return True
    return False


def _validate_filesystem_tool_call(tool_call: dict, messages: list) -> str | None:
    """Return a validation error message for unjustified filesystem tool calls."""
    tool_name = tool_call.get("name")
    if tool_name not in _GUARDED_FILESYSTEM_TOOLS:
        return None

    if _user_requested_local_file_inspection(messages):
        return None

    recent_paths = _collect_recent_path_context(messages)
    if tool_name in _CONTEXTUAL_FILESYSTEM_TOOLS:
        if _tool_paths_match_context(tool_call, recent_paths):
            return None
        if _is_default_artifact_followup(tool_call, messages):
            return None
        return (
            f"`{tool_name}` is disabled unless the user explicitly asked to inspect local files "
            "or a prior tool returned a concrete path that this call is following up on."
        )

    return (
        f"`{tool_name}` is disabled at the orchestrator layer unless the user explicitly asked "
        "to inspect local files. Use handoff tools or domain tools instead."
    )


def _is_materially_different_task(tool_call: dict, messages: list) -> bool:
    """Allow repeated specialist calls only when the task description materially changes."""
    pending_tool_call_id = tool_call.get("id")
    pending_subagent = tool_call.get("args", {}).get("subagent_type", "")
    pending_description = tool_call.get("args", {}).get("description", "")
    if not pending_subagent or not _normalize_task_description(pending_description):
        return False

    prior_descriptions = {
        dispatch.get("description", "")
        for dispatch in _extract_task_dispatches(messages)
        if dispatch["subagent"] == pending_subagent
        and dispatch["tool_call_id"] != pending_tool_call_id
    }
    prior_descriptions = {
        description for description in prior_descriptions
        if _normalize_task_description(description)
    }
    if not prior_descriptions:
        return False
    return not any(
        _task_descriptions_match(pending_description, prior_description)
        for prior_description in prior_descriptions
    )


def _validate_handoff_lookup_call(
    tool_call: dict,
    messages: list,
    allowed_rules: list[dict],
) -> str | None:
    """Block get_subagent_result on successful sequential paths; require build_handoff."""
    tool_name = tool_call.get("name")
    if tool_name not in {
        "get_subagent_result",
        "get_subagent_results",
        "get_all_subagent_results",
    }:
        return None

    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if ordered_plan:
        remaining = _get_active_remaining_steps(messages, ordered_plan)
        if not remaining:
            return (
                "All routed specialists already completed successfully. "
                f"Synthesize the final answer instead of calling `{tool_name}(...)`. "
                "If the user wants more specialist output, wait for a new user request."
            )

    allowed_names = _get_allowed_subagent_names(allowed_rules)
    args = tool_call.get("args", {})
    producer = args.get("agent_name") or args.get("producer")
    if (
        allowed_names == {"separation-engineer"}
        and any(dispatch["subagent"] == "separation-engineer" for dispatch in _extract_returned_subagent_calls(messages))
    ):
        return (
            f"`{tool_name}` is disabled after `separation-engineer` has already returned for this single-specialist route. "
            "Synthesize the final answer directly from the existing separation result or fallback instead."
        )
    if not isinstance(producer, str) or producer not in allowed_names:
        return None

    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    downstream = [
        consumer
        for consumer in _get_downstream_subagents(ordered_plan, producer)
        if consumer in allowed_names
    ]
    if not downstream:
        return None

    latest_dispatch = _get_latest_dispatch_for_subagent(messages, producer)
    if latest_dispatch is None:
        return None

    latest_status = _get_task_handoff_statuses(messages).get(latest_dispatch["tool_call_id"])
    completed_names = {d["subagent"] for d in _extract_completed_subagent_calls(messages)}
    pending_downstream = [consumer for consumer in downstream if consumer not in completed_names]
    if latest_status != "ok" or not pending_downstream:
        return None

    next_consumer = pending_downstream[0]
    return (
        f"`{tool_name}` is fallback-only when `{producer}` returned a missing or invalid handoff. "
        f"Use `list_handoffs(producer=\"{producer}\")` to inspect source results, then "
        f"`build_handoff(consumer=\"{next_consumer}\", producer=\"{producer}\")` "
        f"before dispatching `{next_consumer}`."
    )


def _validate_task_tool_call(
    tool_call: dict,
    messages: list,
    allowed_rules: list[dict],
) -> str | None:
    """Validate task() dispatches against the active specialist route."""
    if tool_call.get("name") != "task":
        return None

    allowed_names = _get_allowed_subagent_names(allowed_rules)
    if not allowed_names:
        return None

    subagent = tool_call.get("args", {}).get("subagent_type", "")
    if not isinstance(subagent, str) or not subagent:
        return "`task` requires a non-empty `subagent_type`."

    if (
        subagent == "visualization-specialist"
        and not _query_explicitly_requests_visualization(messages)
    ):
        return (
            "`visualization-specialist` is disabled for this separation-only route because the user "
            "did not explicitly request a plot, diagram, heatmap, dashboard, or other visualization."
        )

    if subagent not in allowed_names:
        return (
            f"`{subagent}` is outside the active routed specialist set for this query. "
            f"Allowed subagents: {', '.join(sorted(allowed_names))}."
        )

    if len(allowed_names) == 1:
        prior_returns = [
            dispatch
            for dispatch in _extract_returned_subagent_calls(messages)
            if dispatch["subagent"] == subagent
        ]
        if prior_returns:
            return (
                f"`{subagent}` already returned for this single-specialist route. "
                "Synthesize the final answer from the existing specialist result instead of "
                "dispatching another `task()`."
            )

    completed_calls = _extract_completed_subagent_calls(messages)
    completed_names = {call["subagent"] for call in completed_calls}
    if allowed_names.issubset(completed_names):
        return (
            "All routed specialists already completed successfully. "
            "Synthesize the final answer instead of dispatching more `task()` calls. "
            "If the user wants additional specialist work, wait for a new user request."
        )

    ready_handoff = _get_ready_downstream_handoff(messages, allowed_rules)
    if ready_handoff is not None:
        ready_consumer = ready_handoff["consumer"]
        ready_prompt = _normalize_task_description(ready_handoff.get("task_prompt", ""))
        current_prompt = _normalize_task_description(tool_call.get("args", {}).get("description", ""))
        if subagent != ready_consumer:
            return (
                f"A validated handoff for `{ready_consumer}` is already available. "
                f"Dispatch `task(subagent_type=\"{ready_consumer}\")` next instead of `{subagent}`."
            )
        if ready_prompt and current_prompt != ready_prompt:
            return (
                f"`{ready_consumer}` must use the handoff-provided task prompt. "
                f'Use description="{ready_handoff.get("task_prompt", "")}".'
            )

    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    downstream_pending = [
        consumer
        for consumer in _get_downstream_subagents(ordered_plan, subagent)
        if consumer in allowed_names and consumer not in completed_names
    ]
    if subagent in completed_names and downstream_pending:
        return (
            f"`{subagent}` already completed successfully. "
            f"Use `build_handoff(consumer=\"{downstream_pending[0]}\", producer=\"{subagent}\")` "
            "instead of repeating the same upstream task."
        )

    predecessors = [
        producer
        for producer in _get_step_dependencies(ordered_plan, subagent)
        if producer in allowed_names
    ]
    if not predecessors:
        return None

    successful_predecessors = [
        producer
        for producer in predecessors
        if _get_latest_dispatch_for_subagent(messages, producer, status="ok") is not None
    ]
    if not successful_predecessors:
        return (
            f"`{subagent}` is downstream of {', '.join(predecessors)}. "
            "Complete the upstream specialist step first."
        )
    missing_predecessors = [producer for producer in predecessors if producer not in successful_predecessors]
    if missing_predecessors:
        return (
            f"`{subagent}` is downstream of {', '.join(predecessors)}. "
            f"Complete the remaining upstream specialist step(s) first: {', '.join(missing_predecessors)}."
        )

    missing_handoffs = list(_get_missing_required_handoffs_for_consumer(messages, ordered_plan, subagent))
    if not missing_handoffs:
        return None

    producer, _consumer = missing_handoffs[0]
    other_missing = [missing_producer for missing_producer, _ in missing_handoffs[1:]]
    additional_note = ""
    if other_missing:
        additional_note = f" Additional required upstream handoffs remain from: {', '.join(other_missing)}."
    return (
        f"`{subagent}` requires validated upstream handoffs from {', '.join(predecessors)}. "
        f"Call `build_handoff(consumer=\"{subagent}\", producer=\"{producer}\")` "
        "or build from a specific `source_handoff_id` before dispatching the downstream task."
        f"{additional_note}"
    )


def _validate_workflow_tool_call(
    tool_call: dict,
    messages: list,
    allowed_rules: list[dict],
) -> str | None:
    """Block direct specialist-owned tools while a multi-agent workflow is active."""
    tool_name = tool_call.get("name")
    allowed_names = _get_allowed_subagent_names(allowed_rules)
    explicitly_requests_visualization = _query_explicitly_requests_visualization(messages)

    if tool_name == "build_handoff":
        consumer = tool_call.get("args", {}).get("consumer")
        if (
            consumer == "visualization-specialist"
            and not explicitly_requests_visualization
        ):
            return (
                "`build_handoff(..., consumer=\"visualization-specialist\")` is disabled because "
                "the user did not explicitly request a visualization for this route."
            )

    if tool_name in _VISUALIZATION_ROUTE_TOOLS and not explicitly_requests_visualization:
        return (
            f"`{tool_name}` is disabled because the user asked for process design only and did not "
            "explicitly request a visualization."
        )

    if (
        allowed_names == {"separation-engineer"}
        and not _extract_returned_subagent_calls(messages)
        and not _extract_completed_subagent_calls(messages)
        and not _workflow_is_active(messages, allowed_rules)
        and tool_name in (_SEPARATION_SPECIALIST_TOOLS | _SEPARATION_PREFLIGHT_ONLY_TOOLS)
    ):
        return (
            f"`{tool_name}` belongs to the routed `separation-engineer` workflow for this query. "
            'Dispatch `task(subagent_type="separation-engineer")` first instead of calling '
            "top-level lookup or separation tools at the orchestrator layer."
        )

    if not _workflow_is_active(messages, allowed_rules):
        return None

    ready_handoff = _get_ready_downstream_handoff(messages, allowed_rules)
    if ready_handoff and tool_name != "task":
        ready_consumer = ready_handoff["consumer"]
        return (
            f"A validated handoff for `{ready_consumer}` is already available. "
            f"Dispatch `task(subagent_type=\"{ready_consumer}\")` next instead of calling `{tool_name}`."
        )

    pending_handoff = _get_pending_required_handoff(messages, allowed_rules)
    if pending_handoff and tool_name not in _HANDOFF_REQUIRED_TOOLS:
        producer, consumer = pending_handoff
        return (
            f"A validated `{producer}` -> `{consumer}` handoff is required before other orchestrator tools "
            f"can run. Use `list_handoffs(producer=\"{producer}\")` and "
            f"`build_handoff(consumer=\"{consumer}\", producer=\"{producer}\")` first."
        )

    if tool_name in _WORKFLOW_SAFE_TOOLS or tool_name in _GUARDED_FILESYSTEM_TOOLS:
        return None

    return (
        f"`{tool_name}` is disabled at the orchestrator layer while the routed multi-agent "
        "workflow is active. Delegate specialist work via `task(...)` and move results "
        "between specialists with `build_handoff(...)`."
    )


def _build_workflow_guard_messages(
    messages: list,
    allowed_rules: list[dict],
) -> list[ToolMessage]:
    """Create ToolMessages for task/handoff/workflow violations."""
    last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if not last_ai_msg or not getattr(last_ai_msg, "tool_calls", None):
        return []

    guard_messages: list[ToolMessage] = []
    for tool_call in last_ai_msg.tool_calls:
        validation_error = _validate_handoff_lookup_call(tool_call, messages, allowed_rules)
        if validation_error is None:
            validation_error = _validate_task_tool_call(tool_call, messages, allowed_rules)
        if validation_error is None:
            validation_error = _validate_workflow_tool_call(tool_call, messages, allowed_rules)
        if validation_error is None:
            continue

        guard_messages.append(
            ToolMessage(
                content=f"Router guard: {validation_error}",
                tool_call_id=tool_call["id"],
                status="error",
            )
        )

    return guard_messages


def _build_incomplete_route_retry_hint(
    messages: list,
    allowed_rules: list[dict],
) -> str | None:
    """Return a hard retry hint when the model tries to stop too early."""
    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if not ordered_plan:
        return None

    returned_calls = _extract_returned_subagent_calls(messages)
    if not returned_calls:
        allowed_names = _get_allowed_subagent_names(allowed_rules)
        if len(allowed_names) <= 1:
            return None
        next_name = ordered_plan[0]["subagent"]
        return (
            "\n\n[ROUTER_RETRY: This routed multi-specialist workflow has not started yet. "
            f'Call task(subagent_type="{next_name}") now.]'
        )

    remaining = _get_active_remaining_steps(messages, ordered_plan)
    if not remaining:
        return None

    allowed_names = _get_allowed_subagent_names(allowed_rules)
    if (
        allowed_names == {"separation-engineer"}
        and any(dispatch["subagent"] == "separation-engineer" for dispatch in _extract_returned_subagent_calls(messages))
    ):
        return (
            "\n\n[ROUTER_RETRY: The separation specialist already returned. "
            "Write the final answer now from the existing separation result or deterministic fallback. "
            "Do not call `get_subagent_result(...)`, direct separation tools, or another `task()`.]"
        )

    ready_handoff = _get_ready_downstream_handoff(messages, allowed_rules)
    if ready_handoff:
        ready_consumer = ready_handoff["consumer"]
        task_prompt = ready_handoff.get("task_prompt") or ""
        prompt_suffix = (
            f', description="{task_prompt}"'
            if task_prompt
            else ""
        )
        return (
            "\n\n[ROUTER_RETRY: The routed workflow is not complete. "
            f'A validated handoff is ready for "{ready_consumer}". '
            f'Call task(subagent_type="{ready_consumer}"{prompt_suffix}) now.]'
        )

    pending_handoff = _get_pending_required_handoff(messages, allowed_rules)
    if pending_handoff:
        producer, consumer = pending_handoff
        return (
            "\n\n[ROUTER_RETRY: The routed workflow is not complete. "
            f'Call build_handoff(consumer="{consumer}", producer="{producer}") now.]'
        )

    next_name = remaining[0]["subagent"]
    return (
        "\n\n[ROUTER_RETRY: The routed workflow is not complete. "
        f'Call task(subagent_type="{next_name}") now.]'
    )


def _get_response_ai_message(response) -> AIMessage | None:
    """Return the leading AIMessage from a middleware response."""
    if isinstance(response, AIMessage):
        return response

    result = getattr(response, "result", None)
    if not result:
        return None

    first = result[0]
    if isinstance(first, AIMessage):
        return first
    return None


def _build_ready_handoff_task_call(ready_handoff: dict) -> dict:
    """Create the exact downstream task() call for a validated handoff."""
    consumer = ready_handoff["consumer"]
    handoff_id = ready_handoff.get("handoff_id") or ready_handoff.get("parent_handoff_id") or consumer
    task_prompt = ready_handoff.get("task_prompt") or ""

    handoff_ids = ready_handoff.get("handoff_ids")
    if isinstance(handoff_ids, list) and len(handoff_ids) > 1:
        merged_record = build_multi_source_handoff_for_consumer(
            consumer=consumer,
            source_handoff_ids=[str(item) for item in handoff_ids if str(item).strip()],
            task_prompt=task_prompt,
        )
        handoff_id = merged_record.handoff_id
        task_prompt = merged_record.task_prompt or task_prompt

    args = {"subagent_type": consumer}
    if task_prompt:
        args["description"] = task_prompt

    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(handoff_id))
    return {
        "id": f"route_task_{safe_id}_{uuid4().hex[:10]}",
        "name": "task",
        "args": args,
    }


def _build_initial_route_task_response(
    messages: list,
    allowed_rules: list[dict],
) -> ModelResponse | None:
    """Synthesize the first task() call when a routed multi-specialist workflow never started."""
    if not allowed_rules:
        return None
    if _extract_returned_subagent_calls(messages):
        return None

    allowed_names = _get_allowed_subagent_names(allowed_rules)
    if len(allowed_names) <= 1:
        return None

    dependencies = derive_workflow_dependencies(
        _get_last_human_message(messages) or "",
        allowed_names,
    )
    if not any(dependencies.values()):
        return None

    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    if not ordered_plan:
        return None

    remaining = _get_active_remaining_steps(messages, ordered_plan)
    if not remaining:
        return None

    next_name = remaining[0]["subagent"]
    query_text = (_get_last_human_message(messages) or "").strip()
    args = {"subagent_type": next_name}
    if query_text:
        args["description"] = query_text
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": f"route_task_init_{uuid4().hex[:10]}",
            "name": "task",
            "args": args,
        }],
        additional_kwargs={"strap_origin": "routing_initial_multistep_dispatch"},
    )
    return ModelResponse(result=[ai_msg])


def _response_matches_ready_handoff(response, ready_handoff: dict) -> bool:
    """Return True when the response already dispatches the exact downstream task."""
    ai_msg = _get_response_ai_message(response)
    if ai_msg is None:
        return False

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if len(tool_calls) != 1:
        return False

    tool_call = tool_calls[0]
    if tool_call.get("name") != "task":
        return False

    consumer = ready_handoff["consumer"]
    if tool_call.get("args", {}).get("subagent_type") != consumer:
        return False

    expected_prompt = _normalize_task_description(ready_handoff.get("task_prompt", ""))
    actual_prompt = _normalize_task_description(tool_call.get("args", {}).get("description", ""))
    return not expected_prompt or actual_prompt == expected_prompt


def _build_ready_handoff_response(ready_handoff: dict, response=None) -> ModelResponse:
    """Return a synthetic model response that dispatches the ready downstream task."""
    task_call = _build_ready_handoff_task_call(ready_handoff)
    ai_msg = AIMessage(content="", tool_calls=[task_call])
    structured_response = getattr(response, "structured_response", None)
    return ModelResponse(result=[ai_msg], structured_response=structured_response)


def _response_matches_pending_handoff(response, pending_handoff: tuple[str, str]) -> bool:
    """Return True when the response already emits the required build_handoff call."""
    ai_msg = _get_response_ai_message(response)
    if ai_msg is None:
        return False

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if len(tool_calls) != 1:
        return False

    tool_call = tool_calls[0]
    if tool_call.get("name") != "build_handoff":
        return False

    producer, consumer = pending_handoff
    args = tool_call.get("args", {})
    if args.get("consumer") != consumer:
        return False
    return args.get("producer") == producer or bool(args.get("source_handoff_id"))


def _build_pending_handoff_response(
    pending_handoff: tuple[str, str],
    response=None,
) -> ModelResponse:
    """Return a synthetic model response that builds the required handoff."""
    producer, consumer = pending_handoff
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{producer}_{consumer}")
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": f"route_handoff_{safe_id}_{uuid4().hex[:10]}",
            "name": "build_handoff",
            "args": {
                "producer": producer,
                "consumer": consumer,
                "strategy": "latest",
            },
        }],
    )
    structured_response = getattr(response, "structured_response", None)
    return ModelResponse(result=[ai_msg], structured_response=structured_response)
